//! GPU probe — three independent access paths are tested:
//!
//! 1. **DRM** — presence of `/dev/dri/renderD128` and the driver name reported
//!    by sysfs. This is the baseline kernel interface used by Mesa/Turnip on
//!    upstream-kernel images of the QRB2210.
//! 2. **Vulkan** — Turnip ICD via `vulkaninfo --summary`. We parse the
//!    well-defined summary format rather than dlopen'ing `libvulkan.so.1`,
//!    which would require an unsafe FFI shim in Phase 1.
//! 3. **OpenCL** — Rusticl via `clinfo`. Same parser rationale as Vulkan.
//! 4. **KGSL** — legacy `/dev/kgsl-3d0` (Qualcomm downstream driver). Recorded
//!    for completeness; on a mainline kernel this is absent.
//!
//! Each path is independently optional; absence is recorded as `None`, not an
//! error.

use dragonwing_core::capabilities::{
    GpuInfo, KgslPath, OpenClDevice, OpenClPath, OpenClPlatform, VulkanDevice, VulkanPath,
};
use std::path::Path;
use std::process::Command;

#[must_use]
pub fn probe() -> GpuInfo {
    let drm_render_node = if Path::new("/dev/dri/renderD128").exists() {
        Some("/dev/dri/renderD128".to_string())
    } else {
        None
    };
    let drm_driver = read_drm_driver();
    let of_compatible = read_of_compatible();
    let kgsl = probe_kgsl();
    let vulkan = probe_vulkan();
    let opencl = probe_opencl();

    let present = drm_render_node.is_some() || kgsl.is_some() || vulkan.is_some();

    GpuInfo {
        present,
        drm_render_node,
        drm_driver,
        of_compatible,
        vulkan,
        opencl,
        kgsl,
    }
}

fn read_drm_driver() -> Option<String> {
    // /sys/class/drm/renderD128/device/uevent contains `DRIVER=msm` on Adreno.
    let text = std::fs::read_to_string("/sys/class/drm/renderD128/device/uevent").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("DRIVER=") {
            return Some(rest.trim().to_string());
        }
    }
    None
}

fn read_of_compatible() -> Option<String> {
    // Devicetree compatible string. Multiple null-separated entries possible;
    // we report the first.
    let bytes = std::fs::read("/sys/class/drm/renderD128/device/of_node/compatible").ok()?;
    let first_null = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    Some(String::from_utf8_lossy(&bytes[..first_null]).into_owned())
}

fn probe_kgsl() -> Option<KgslPath> {
    if !Path::new("/dev/kgsl-3d0").exists() {
        return None;
    }
    let gpu_model = std::fs::read_to_string("/sys/class/kgsl/kgsl-3d0/gpu_model")
        .ok()
        .map(|s| s.trim().to_string());
    let firmware_version = std::fs::read_to_string("/sys/class/kgsl/kgsl-3d0/fw_version")
        .ok()
        .map(|s| s.trim().to_string());
    Some(KgslPath {
        device_node: "/dev/kgsl-3d0".to_string(),
        gpu_model,
        firmware_version,
    })
}

fn probe_vulkan() -> Option<VulkanPath> {
    // Confirm the loader exists. We don't try to load it ourselves — the OS
    // loader is invoked by vulkaninfo.
    let loader_candidates = [
        "/lib/aarch64-linux-gnu/libvulkan.so.1",
        "/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
        "/usr/lib/libvulkan.so.1",
    ];
    let loader_path = loader_candidates
        .iter()
        .find(|p| Path::new(p).exists())
        .map(|s| (*s).to_string())?;

    let output = Command::new("vulkaninfo").arg("--summary").output().ok()?;
    if !output.status.success() {
        // vulkaninfo can warn about missing surfaces but still print devices
        // and exit 0. If it failed outright, we can't trust anything.
        return Some(VulkanPath {
            loader_path,
            instance_api_version: String::new(),
            devices: Vec::new(),
        });
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let (instance_api_version, devices) = parse_vulkaninfo_summary(&text);
    Some(VulkanPath {
        loader_path,
        instance_api_version,
        devices,
    })
}

/// Parse the deterministic portion of `vulkaninfo --summary` output. The
/// format is documented in the Vulkan SDK and stable: an `Instance Version`
/// line followed by `GPUN:` blocks containing `key = value` pairs.
fn parse_vulkaninfo_summary(text: &str) -> (String, Vec<VulkanDevice>) {
    let mut instance_api_version = String::new();
    let mut devices: Vec<VulkanDevice> = Vec::new();
    let mut current: Option<VulkanDevice> = None;

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Vulkan Instance Version:") {
            instance_api_version = rest.trim().to_string();
            continue;
        }
        if trimmed.starts_with("GPU") && trimmed.ends_with(':') {
            if let Some(d) = current.take() {
                devices.push(d);
            }
            current = Some(VulkanDevice::default());
            continue;
        }
        let Some(dev) = current.as_mut() else {
            continue;
        };
        let Some((k, v)) = trimmed.split_once('=') else {
            continue;
        };
        let key = k.trim();
        let val = v.trim().to_string();
        match key {
            "apiVersion" => dev.api_version = val,
            "deviceName" => dev.device_name = val,
            "driverName" => dev.driver_name = val,
            "driverInfo" => dev.driver_info = val,
            "deviceType" => dev.device_type = val,
            "vendorID" => dev.vendor_id = parse_hex_u32(&val),
            "deviceID" => dev.device_id = parse_hex_u32(&val),
            _ => {}
        }
    }
    if let Some(d) = current {
        devices.push(d);
    }
    (instance_api_version, devices)
}

fn parse_hex_u32(s: &str) -> u32 {
    let t = s.trim().trim_start_matches("0x");
    u32::from_str_radix(t, 16).unwrap_or(0)
}

fn probe_opencl() -> Option<OpenClPath> {
    let loader_candidates = [
        "/lib/aarch64-linux-gnu/libOpenCL.so.1",
        "/usr/lib/aarch64-linux-gnu/libOpenCL.so.1",
        "/usr/lib/libOpenCL.so.1",
    ];
    let loader_path = loader_candidates
        .iter()
        .find(|p| Path::new(p).exists())
        .map(|s| (*s).to_string())?;

    let output = Command::new("clinfo").output().ok();
    let platforms = match output {
        Some(o) if o.status.success() => {
            parse_clinfo(&String::from_utf8_lossy(&o.stdout))
        }
        _ => Vec::new(),
    };
    Some(OpenClPath {
        loader_path,
        platforms,
    })
}

/// Parse `clinfo` default output.
///
/// `clinfo` output has two top-level sections:
///
///   * `Number of platforms  N` followed by N platform metadata blocks (each
///     starting with `Platform Name`).
///   * Then, for each platform: another `Platform Name` header followed by
///     `Number of devices M` and M device blocks (each starting with
///     `Device Name`).
///
/// To avoid double-counting platforms we only treat the *first occurrence* of
/// each platform name as the canonical platform; subsequent identical
/// `Platform Name` headers are interpreted as "we are now inside this
/// platform's device list" and we look up the existing platform by name.
fn parse_clinfo(text: &str) -> Vec<OpenClPlatform> {
    let mut platforms: Vec<OpenClPlatform> = Vec::new();
    let mut cur_plat_idx: Option<usize> = None;
    let mut cur_dev: Option<OpenClDevice> = None;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Platform Name") {
            // Flush pending device into prior platform.
            if let (Some(d), Some(idx)) = (cur_dev.take(), cur_plat_idx) {
                platforms[idx].devices.push(d);
            }
            let name = split_clinfo_value(trimmed).unwrap_or_default();
            cur_plat_idx = if let Some(idx) = platforms.iter().position(|p| p.name == name) {
                Some(idx)
            } else {
                platforms.push(OpenClPlatform {
                    name,
                    ..Default::default()
                });
                Some(platforms.len() - 1)
            };
            continue;
        }
        let Some(idx) = cur_plat_idx else { continue };
        let p = &mut platforms[idx];
        if trimmed.starts_with("Platform Vendor") && p.vendor.is_empty() {
            p.vendor = split_clinfo_value(trimmed).unwrap_or_default();
        } else if trimmed.starts_with("Platform Version") && p.version.is_empty() {
            p.version = split_clinfo_value(trimmed).unwrap_or_default();
        } else if trimmed.starts_with("Device Name") {
            // Same dedup pattern as for platforms: clinfo repeats
            // `Device Name` headers in successive sub-sections (general,
            // extensions, IL versions, ...). If we already have a device
            // with this name on this platform, point cur_dev at it rather
            // than appending a duplicate.
            let name = split_clinfo_value(trimmed).unwrap_or_default();
            if let Some(d) = cur_dev.take() {
                p.devices.push(d);
            }
            if let Some(existing) = p.devices.iter().position(|d| d.name == name) {
                // Move the existing device back into cur_dev so further
                // attribute lines (Device Vendor, Max compute units, etc.)
                // continue to populate it. We swap-remove and reinsert at
                // end on flush to keep ordering stable.
                cur_dev = Some(p.devices.swap_remove(existing));
            } else {
                cur_dev = Some(OpenClDevice {
                    name,
                    ..Default::default()
                });
            }
        } else if let Some(d) = cur_dev.as_mut() {
            if trimmed.starts_with("Device Vendor")
                && !trimmed.starts_with("Device Vendor ID")
                && d.vendor.is_empty()
            {
                d.vendor = split_clinfo_value(trimmed).unwrap_or_default();
            } else if trimmed.starts_with("Device Version") && d.version.is_empty() {
                d.version = split_clinfo_value(trimmed).unwrap_or_default();
            } else if trimmed.starts_with("Max compute units") && d.compute_units == 0 {
                d.compute_units = split_clinfo_value(trimmed)
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0);
            } else if trimmed.starts_with("Global memory size") && d.global_mem_bytes == 0 {
                // Format: "Global memory size    1234567 (1.234GiB)"
                d.global_mem_bytes = split_clinfo_value(trimmed)
                    .and_then(|s| s.split_whitespace().next().map(str::to_string))
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
            }
        }
    }
    if let (Some(d), Some(idx)) = (cur_dev, cur_plat_idx) {
        platforms[idx].devices.push(d);
    }
    platforms
}

/// `clinfo` separates label and value with at least two spaces. We split on
/// the first run of two-or-more spaces.
fn split_clinfo_value(line: &str) -> Option<String> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b' ' && bytes[i + 1] == b' ' {
            // Found the separator; skip the whole whitespace run.
            let mut j = i + 2;
            while j < bytes.len() && bytes[j] == b' ' {
                j += 1;
            }
            if j < bytes.len() {
                return Some(line[j..].trim().to_string());
            }
            return None;
        }
        i += 1;
    }
    None
}
