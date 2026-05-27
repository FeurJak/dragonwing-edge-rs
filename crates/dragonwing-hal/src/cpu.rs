//! CPU probe — reads /proc/cpuinfo and /sys/devices/system/cpu.

use dragonwing_core::capabilities::{CpuCore, CpuInfo};
use std::fs;
use std::path::Path;

#[must_use]
pub fn probe() -> CpuInfo {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let (features, implementer, part) = parse_cpuinfo(&cpuinfo);

    let logical_cores = count_cpus("/sys/devices/system/cpu");
    let mut cores = Vec::with_capacity(logical_cores as usize);
    let mut online_cores = 0u32;
    for i in 0..logical_cores {
        let core = probe_core(i);
        if core.online {
            online_cores += 1;
        }
        cores.push(core);
    }

    let (vendor, part_name) = decode_part(implementer, part);

    CpuInfo {
        logical_cores,
        online_cores,
        cores,
        features,
        vendor,
        part: part_name,
        implementer_raw: implementer,
        part_raw: part,
    }
}

fn count_cpus(base: &str) -> u32 {
    let Ok(entries) = fs::read_dir(base) else {
        return 0;
    };
    let mut max_seen: Option<u32> = None;
    for e in entries.flatten() {
        let name = e.file_name();
        let s = name.to_string_lossy();
        // Match exactly `cpuN` where N is digits.
        if let Some(rest) = s.strip_prefix("cpu")
            && let Ok(n) = rest.parse::<u32>()
            && max_seen.is_none_or(|m| n > m)
        {
            max_seen = Some(n);
        }
    }
    max_seen.map_or(0, |m| m + 1)
}

fn probe_core(i: u32) -> CpuCore {
    let base = format!("/sys/devices/system/cpu/cpu{i}");
    // cpu0 has no `online` file on most kernels (it can't be offlined). Treat
    // missing file as online.
    let online = if let Ok(s) = fs::read_to_string(format!("{base}/online")) {
        s.trim() == "1"
    } else {
        Path::new(&base).exists()
    };
    let freq_min_khz = read_u64(&format!("{base}/cpufreq/cpuinfo_min_freq"));
    let freq_max_khz = read_u64(&format!("{base}/cpufreq/cpuinfo_max_freq"));
    let freq_cur_khz = read_u64(&format!("{base}/cpufreq/scaling_cur_freq"));
    let governor = fs::read_to_string(format!("{base}/cpufreq/scaling_governor"))
        .ok()
        .map(|s| s.trim().to_string());
    CpuCore {
        index: i,
        online,
        freq_min_khz,
        freq_max_khz,
        freq_cur_khz,
        governor,
    }
}

fn read_u64(path: &str) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
}

/// Parse the relevant fields out of `/proc/cpuinfo`. We grab the *last*
/// occurrence of each header; on heterogeneous Arm SoCs that means the
/// highest-numbered core, which is fine because the QRB2210 is homogeneous
/// (4× Kryo). For a heterogeneous SoC we'd record per-core implementer/part.
fn parse_cpuinfo(text: &str) -> (Vec<String>, u32, u32) {
    let mut features = Vec::new();
    let mut implementer = 0u32;
    let mut part = 0u32;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Features\t: ") {
            features = rest.split_whitespace().map(str::to_string).collect();
        } else if let Some(rest) = line.strip_prefix("CPU implementer\t: ") {
            implementer = parse_hex(rest).unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("CPU part\t: ") {
            part = parse_hex(rest).unwrap_or(0);
        }
    }
    (features, implementer, part)
}

fn parse_hex(s: &str) -> Option<u32> {
    let t = s.trim().trim_start_matches("0x");
    u32::from_str_radix(t, 16).ok()
}

/// Decode `CPU implementer` and `CPU part` into human-readable strings.
/// Reference: Arm Architecture Reference Manual + linux/arch/arm64/include/asm/cputype.h.
fn decode_part(implementer: u32, part: u32) -> (String, String) {
    let vendor = match implementer {
        0x41 => "ARM",
        0x42 => "Broadcom",
        0x43 => "Cavium",
        0x44 => "DEC",
        0x48 => "HiSilicon",
        0x49 => "Infineon",
        0x4E => "Nvidia",
        0x50 => "Applied Micro",
        0x51 => "Qualcomm",
        0x53 => "Samsung",
        0x56 => "Marvell",
        0x61 => "Apple",
        0x66 => "Faraday",
        0x69 => "Intel",
        0xC0 => "Ampere",
        _ => "Unknown",
    };
    // Qualcomm parts on QRB2210 / QCM2290: 0x801 == Kryo (Cortex-A53 derived).
    let part_name = match (implementer, part) {
        (0x51, 0x800) => "Qualcomm Kryo (Cortex-A73 derivative)",
        (0x51, 0x801) => "Qualcomm Kryo (Cortex-A53 derivative)",
        (0x51, 0x802) => "Qualcomm Kryo (Cortex-A75 derivative, gold)",
        (0x51, 0x803 | 0x805) => "Qualcomm Kryo (Cortex-A55 derivative, silver)",
        (0x51, 0x804) => "Qualcomm Kryo (Cortex-A76 derivative, gold)",
        (0x41, 0xD03) => "Arm Cortex-A53",
        (0x41, 0xD07) => "Arm Cortex-A57",
        (0x41, 0xD08) => "Arm Cortex-A72",
        _ => "Unknown",
    };
    (vendor.to_string(), part_name.to_string())
}
