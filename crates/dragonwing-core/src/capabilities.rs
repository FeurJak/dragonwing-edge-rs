//! Hardware capability descriptors.
//!
//! These types are produced by `dragonwing-hal` and consumed by anything that
//! needs to make a runtime decision about which backend to dispatch to. They
//! are plain data — no behaviour — so they live in `core`.
//!
//! All fields are public to keep this crate boring; the canonical producer
//! (`dragonwing-hal`) fills them in via struct-literal syntax.

use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Top-level capability snapshot for a single device.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HardwareCapabilities {
    /// Schema version. Bump when the shape of this struct changes in a
    /// breaking way; consumers can use it to refuse to parse newer reports.
    pub schema_version: u32,
    /// ISO-8601 timestamp the probe was captured. Filled in by the probe
    /// binary, not by the HAL.
    pub captured_at: String,
    /// Free-form identifier of the host (e.g. `uname -n`).
    pub host: String,
    pub kernel: KernelInfo,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub thermal: Vec<ThermalZone>,
    pub gpu: GpuInfo,
    pub dsp: DspInfo,
    pub mcu_link: McuLinkInfo,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KernelInfo {
    pub uname: String,
    pub arch: String,
    pub os_pretty_name: String,
    pub libc: String,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CpuInfo {
    pub logical_cores: u32,
    pub online_cores: u32,
    /// One entry per logical core.
    pub cores: Vec<CpuCore>,
    /// Aggregated feature flags taken from `/proc/cpuinfo` line `Features:`.
    pub features: Vec<String>,
    /// Vendor as decoded from `CPU implementer` (e.g. `0x51` → `"Qualcomm"`).
    pub vendor: String,
    /// Decoded part name (e.g. `0x801` → `"Kryo / Cortex-A53 derivative"`).
    pub part: String,
    pub implementer_raw: u32,
    pub part_raw: u32,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CpuCore {
    pub index: u32,
    pub online: bool,
    /// kHz. None if cpufreq is not exposed for this core.
    pub freq_min_khz: Option<u64>,
    pub freq_max_khz: Option<u64>,
    pub freq_cur_khz: Option<u64>,
    pub governor: Option<String>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryInfo {
    pub total_kib: u64,
    pub available_kib: u64,
    pub free_kib: u64,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ThermalZone {
    pub index: u32,
    pub type_name: String,
    /// millidegrees Celsius (kernel convention).
    pub temp_mc: i64,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuInfo {
    /// Whether any usable GPU access path was detected.
    pub present: bool,
    /// Linux DRM render-node path, e.g. `/dev/dri/renderD128`. None if DRM
    /// isn't exposed.
    pub drm_render_node: Option<String>,
    /// Kernel DRM driver name (`msm`, `kgsl`, etc).
    pub drm_driver: Option<String>,
    /// Devicetree compatible string of the GPU node, if extractable.
    pub of_compatible: Option<String>,

    /// Vulkan access path, populated when `libvulkan.so.1` is loadable and
    /// at least one physical device enumerates.
    pub vulkan: Option<VulkanPath>,
    /// OpenCL access path, populated when `libOpenCL.so.1` is loadable and
    /// reports at least one device.
    pub opencl: Option<OpenClPath>,
    /// Direct KGSL ioctl access (Qualcomm downstream driver). None on
    /// mainline-kernel images.
    pub kgsl: Option<KgslPath>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VulkanPath {
    pub loader_path: String,
    pub instance_api_version: String,
    pub devices: Vec<VulkanDevice>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VulkanDevice {
    pub device_name: String,
    pub driver_name: String,
    pub driver_info: String,
    pub api_version: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: String,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OpenClPath {
    pub loader_path: String,
    pub platforms: Vec<OpenClPlatform>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OpenClPlatform {
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub devices: Vec<OpenClDevice>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OpenClDevice {
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub compute_units: u32,
    pub global_mem_bytes: u64,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KgslPath {
    pub device_node: String,
    pub gpu_model: Option<String>,
    pub firmware_version: Option<String>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DspInfo {
    pub present: bool,
    /// FastRPC device nodes that exist.
    pub fastrpc_nodes: Vec<String>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct McuLinkInfo {
    /// Candidate serial/USB devices that may be the STM32U585 bridge.
    pub candidate_tty: Vec<String>,
    pub usb_devices: Vec<UsbDevice>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UsbDevice {
    pub bus_address: String,
    pub vendor_id: u16,
    pub product_id: u16,
    pub description: String,
}

impl HardwareCapabilities {
    /// Current schema version emitted by the probe.
    pub const SCHEMA_VERSION: u32 = 1;
}
