//! Vulkan instance / physical device / logical device / queue setup.
//!
//! Everything here happens exactly once per [`VulkanBackend`](crate::VulkanBackend)
//! at construction time and is then immutable. Errors from any step
//! produce a clean `Error::Backend(...)` with a human-readable message so
//! integration debugging on unknown drivers is straightforward.
//!
//! # Selection policy
//!
//! Physical devices are scored as follows and the highest-scoring device
//! is picked (ties broken by enumeration order):
//!
//! * Turnip on Adreno (driver id `MESA_TURNIP`): **+100**
//! * Any integrated GPU: **+10**
//! * Any discrete GPU: **+5**
//! * CPU fallback (`llvmpipe`): **+1**
//!
//! Devices that don't support the required extensions are filtered out
//! before scoring.
//!
//! # Required extensions (task 002)
//!
//! * `VK_KHR_storage_buffer_storage_class` — SSBO access in shaders.
//! * `VK_KHR_synchronization2` — modern barriers / queue submit.
//! * `VK_KHR_timeline_semaphore` — single-counter sync used by `synchronize`.
//! * `VK_KHR_8bit_storage` / `VK_KHR_16bit_storage` — reserved for task 003.
//! * `VK_KHR_shader_float16_int8` — reserved for task 003.
//! * `VK_KHR_shader_integer_dot_product` — reserved for task 003.
//!
//! All of these are present on the device per the task-001 hardware probe.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Mutex;

use ash::vk;
use dragonwing_core::{Error, Result};

use crate::error::{loading_err, vk_err};

/// User-facing knobs for backend construction.
#[derive(Debug, Clone)]
pub struct VulkanConfig {
    /// Preferred device type. The selection logic still prefers Turnip on
    /// Adreno when this is `INTEGRATED_GPU`; this field is a tie-breaker
    /// for non-Adreno hosts.
    pub prefer_device_type: vk::PhysicalDeviceType,

    /// Enable `VK_LAYER_KHRONOS_validation` (only useful when the layer
    /// is installed; not the case on the UNO Q stock image).
    pub validation: bool,

    /// Optional fixed device index. If set, skip the scoring logic and
    /// use that index directly. Useful in tests.
    pub device_index: Option<usize>,
}

impl Default for VulkanConfig {
    fn default() -> Self {
        Self {
            prefer_device_type: vk::PhysicalDeviceType::INTEGRATED_GPU,
            validation: cfg!(feature = "validation"),
            device_index: None,
        }
    }
}

/// Shared Vulkan context held inside every [`VulkanBackend`](crate::VulkanBackend)
/// clone. Owns the instance, device, and queue.
pub struct Context {
    /// Held to keep `libvulkan.so.1` loaded for the lifetime of the
    /// context. Drop order matters: `entry` must outlive `instance`.
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    timeline: Mutex<Timeline>,
    /// KHR extension loader for timeline semaphores (needed for Vulkan 1.0/1.1).
    timeline_semaphore_khr: ash::khr::timeline_semaphore::Device,

    device_name: String,
    driver_info: String,
    physical_device_properties: vk::PhysicalDeviceProperties,
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("device_name", &self.device_name)
            .field("driver_info", &self.driver_info)
            .finish_non_exhaustive()
    }
}

/// Timeline-semaphore counter used by [`Context::wait_idle`].
struct Timeline {
    semaphore: vk::Semaphore,
    next_signal: u64,
}

impl Context {
    /// Build a new context. See module docs for the selection policy.
    pub fn new(config: VulkanConfig) -> Result<Self> {
        // SAFETY: `Entry::load` dlopens `libvulkan.so.1`. The returned
        // entry owns the loaded library; safe under standard dlopen
        // semantics.
        let entry = unsafe { ash::Entry::load() }.map_err(loading_err)?;

        let instance = create_instance(&entry, &config)?;
        let (physical_device, queue_family_index, props, driver_info) =
            select_physical_device(&instance, &config)?;

        let device = create_logical_device(&instance, physical_device, queue_family_index)?;

        // SAFETY: the queue family index we picked is in range, and we
        // created the device asking for at least one queue in it.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Create the KHR extension loader for timeline semaphores.
        // This is needed for Vulkan 1.0/1.1 devices that expose
        // VK_KHR_timeline_semaphore but not the core 1.2 function.
        let timeline_semaphore_khr = ash::khr::timeline_semaphore::Device::new(&instance, &device);

        // Create the timeline semaphore used by `wait_idle`.
        let semaphore = create_timeline_semaphore(&device)?;

        // SAFETY: `props.device_name` is a fixed-size C array of bytes
        // ending in a NUL within the array.
        let device_name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            timeline: Mutex::new(Timeline {
                semaphore,
                next_signal: 0,
            }),
            timeline_semaphore_khr,
            device_name,
            driver_info,
            physical_device_properties: props,
        })
    }

    /// Borrow the `ash::Device`. Used by op modules to record command
    /// buffers and submit work.
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Borrow the `ash::Instance`. Rarely needed by user code.
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// The chosen `VkPhysicalDevice`. Needed for memory-type queries
    /// in [`crate::memory`].
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Cached physical-device properties (limits, vendor IDs, etc.).
    pub fn physical_device_properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.physical_device_properties
    }

    /// The single compute queue. Op submissions go here.
    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// Queue family index the [`queue`](Self::queue) belongs to. Needed
    /// for command-pool creation.
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Human-readable device name (e.g. `"Turnip Adreno (TM) 702"`).
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// `"driverName - driverInfo"` from `VkPhysicalDeviceDriverProperties`.
    pub fn driver_info(&self) -> &str {
        &self.driver_info
    }

    /// Block until every submission made through this context has
    /// completed.
    ///
    /// Implemented via a single timeline semaphore counter: we wait on
    /// `next_signal - 1`. If no submission has happened yet, this is a
    /// trivial wait that returns immediately.
    pub fn wait_idle(&self) -> Result<()> {
        let signal = {
            let t = self.timeline.lock().expect("timeline mutex poisoned");
            if t.next_signal == 0 {
                return Ok(());
            }
            t.next_signal - 1
        };
        let semaphores = [self.timeline_semaphore()];
        let values = [signal];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: handles owned by self; spec-compliant struct.
        // Use the KHR extension loader for Vulkan 1.0/1.1 compatibility.
        unsafe {
            self.timeline_semaphore_khr
                .wait_semaphores(&wait_info, u64::MAX)
                .map_err(|r| vk_err("wait_semaphores", r))?;
        }
        Ok(())
    }

    /// Get the timeline-semaphore handle. Op code calls this and then
    /// [`next_signal_value`](Self::next_signal_value) when building a
    /// `VkSubmitInfo`.
    pub fn timeline_semaphore(&self) -> vk::Semaphore {
        self.timeline
            .lock()
            .expect("timeline mutex poisoned")
            .semaphore
    }

    /// Reserve the next signal value. Returns the value to put in the
    /// submit's `pSignalSemaphoreValues`. Atomically increments
    /// `next_signal`.
    pub fn next_signal_value(&self) -> u64 {
        let mut t = self.timeline.lock().expect("timeline mutex poisoned");
        t.next_signal += 1;
        t.next_signal
    }
}

// Drop order: device -> instance -> entry. ash's Drop on Instance/Device
// does NOT call vkDestroy*; we must do it explicitly.
impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: we own these handles and are not in concurrent use
        // because Drop has exclusive access to self.
        unsafe {
            // Best-effort idle. If wait_idle fails (device lost) we still
            // proceed to destroy handles to avoid leaks.
            let _ = self.wait_idle();
            let sem = self.timeline_semaphore();
            self.device.destroy_semaphore(sem, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ---------------------------------------------------------------------------
// Instance creation
// ---------------------------------------------------------------------------

fn create_instance(entry: &ash::Entry, config: &VulkanConfig) -> Result<ash::Instance> {
    let app_name = CString::new("dragonwing-edge").unwrap();
    let engine_name = CString::new("dragonwing-vulkan").unwrap();

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 0, 1))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 0, 1))
        .api_version(vk::API_VERSION_1_1);

    let mut layer_names: Vec<CString> = Vec::new();
    if config.validation {
        layer_names.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
    }
    let layer_ptrs: Vec<*const c_char> = layer_names.iter().map(|s| s.as_ptr()).collect();

    // No instance-level extensions required for headless compute.
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_ptrs);

    // SAFETY: pointers in create_info reference data alive in this stack
    // frame until vkCreateInstance returns.
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .map_err(|r| vk_err("create_instance", r))?;
    Ok(instance)
}

// ---------------------------------------------------------------------------
// Physical-device selection
// ---------------------------------------------------------------------------

const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    // Always-required for storage buffers + sync2 + timeline semaphores.
    c"VK_KHR_storage_buffer_storage_class",
    c"VK_KHR_synchronization2",
    c"VK_KHR_timeline_semaphore",
    // 16-bit storage is widely available and needed for FP16 buffers.
    c"VK_KHR_16bit_storage",
    c"VK_KHR_shader_float16_int8",
];

// Note: VK_KHR_8bit_storage and VK_KHR_shader_integer_dot_product are optional
// and checked at device creation time. They're not in the required list because
// not all drivers (e.g., Turnip on A702) support them.

fn select_physical_device(
    instance: &ash::Instance,
    config: &VulkanConfig,
) -> Result<(vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties, String)> {
    // SAFETY: instance owned by caller.
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|r| vk_err("enumerate_physical_devices", r))?;
    if devices.is_empty() {
        return Err(Error::Backend(
            "vulkan: no physical devices enumerated".into(),
        ));
    }

    if let Some(idx) = config.device_index {
        let pd = *devices
            .get(idx)
            .ok_or_else(|| Error::Backend(format!("vulkan: device_index {idx} out of range")))?;
        return finalise_device(instance, pd, config);
    }

    // Score and pick.
    let mut best: Option<(i32, vk::PhysicalDevice)> = None;
    for &pd in &devices {
        let Ok(()) = check_required_extensions(instance, pd) else {
            continue;
        };
        if find_compute_queue_family(instance, pd).is_none() {
            continue;
        }
        let score = score_device(instance, pd, config);
        if best.is_none_or(|(s, _)| score > s) {
            best = Some((score, pd));
        }
    }
    let pd = best
        .map(|(_, pd)| pd)
        .ok_or_else(|| Error::Backend("vulkan: no suitable physical device".into()))?;
    finalise_device(instance, pd, config)
}

fn finalise_device(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    _config: &VulkanConfig,
) -> Result<(vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties, String)> {
    let qfi = find_compute_queue_family(instance, pd)
        .ok_or_else(|| Error::Backend("vulkan: no compute queue family".into()))?;
    // SAFETY: read-only Vulkan getter.
    let props = unsafe { instance.get_physical_device_properties(pd) };

    // Pull driver name + info via VK_KHR_driver_properties (core in 1.2; on
    // 1.1 it's an extension. Mesa exposes it as core via the property
    // chain regardless).
    let driver_info = read_driver_info(instance, pd);

    Ok((pd, qfi, props, driver_info))
}

fn check_required_extensions(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Result<()> {
    // SAFETY: read-only enumeration.
    let exts = unsafe { instance.enumerate_device_extension_properties(pd) }
        .map_err(|r| vk_err("enumerate_device_extension_properties", r))?;
    for req in REQUIRED_DEVICE_EXTENSIONS {
        let found = exts.iter().any(|e| {
            // SAFETY: extension_name is NUL-terminated within the array.
            let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            name == *req
        });
        if !found {
            return Err(Error::Backend(format!(
                "vulkan: missing required extension {req:?}"
            )));
        }
    }
    Ok(())
}

fn find_compute_queue_family(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Option<u32> {
    // SAFETY: read-only Vulkan getter.
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    // Prefer compute-only (no graphics) family if one exists.
    let mut compute_only = None;
    let mut any_compute = None;
    for (i, f) in families.iter().enumerate() {
        if !f.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            continue;
        }
        if any_compute.is_none() {
            any_compute = Some(i as u32);
        }
        if !f.queue_flags.contains(vk::QueueFlags::GRAPHICS) && compute_only.is_none() {
            compute_only = Some(i as u32);
        }
    }
    compute_only.or(any_compute)
}

fn score_device(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    config: &VulkanConfig,
) -> i32 {
    // SAFETY: read-only getter.
    let props = unsafe { instance.get_physical_device_properties(pd) };
    let driver_info = read_driver_info(instance, pd);
    let is_turnip = driver_info.to_lowercase().contains("turnip");

    let type_score = match (props.device_type, config.prefer_device_type) {
        (vk::PhysicalDeviceType::INTEGRATED_GPU, _) => 10,
        (vk::PhysicalDeviceType::DISCRETE_GPU, _) => 5,
        (vk::PhysicalDeviceType::CPU, _) => 1,
        _ => 0,
    };
    let mut score = type_score;
    if is_turnip {
        score += 100;
    }
    if props.device_type == config.prefer_device_type {
        score += 1;
    }
    score
}

fn read_driver_info(instance: &ash::Instance, pd: vk::PhysicalDevice) -> String {
    let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver_props);
    // SAFETY: chain is valid for the duration of the call.
    unsafe {
        instance.get_physical_device_properties2(pd, &mut props2);
    }
    // SAFETY: NUL-terminated strings inside fixed arrays.
    let name = unsafe { CStr::from_ptr(driver_props.driver_name.as_ptr()) }.to_string_lossy();
    let info = unsafe { CStr::from_ptr(driver_props.driver_info.as_ptr()) }.to_string_lossy();
    format!("{name} - {info}")
}

// ---------------------------------------------------------------------------
// Logical device + queue
// ---------------------------------------------------------------------------

fn create_logical_device(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<ash::Device> {
    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities)];

    // Enumerate available extensions to decide which optional ones to enable.
    // SAFETY: read-only enumeration.
    let available_exts = unsafe { instance.enumerate_device_extension_properties(pd) }
        .map_err(|r| vk_err("enumerate_device_extension_properties", r))?;

    let has_ext = |name: &CStr| -> bool {
        available_exts.iter().any(|e| {
            // SAFETY: extension_name is NUL-terminated within the array.
            let ext_name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            ext_name == name
        })
    };

    // Build extension list: required + available optional.
    let mut extensions_to_enable: Vec<&CStr> = REQUIRED_DEVICE_EXTENSIONS.to_vec();
    let has_8bit = has_ext(c"VK_KHR_8bit_storage");
    let has_dot_product = has_ext(c"VK_KHR_shader_integer_dot_product");

    if has_8bit {
        extensions_to_enable.push(c"VK_KHR_8bit_storage");
    }
    if has_dot_product {
        extensions_to_enable.push(c"VK_KHR_shader_integer_dot_product");
    }

    let ext_ptrs: Vec<*const c_char> = extensions_to_enable
        .iter()
        .map(|c| c.as_ptr())
        .collect();

    // Feature chain: timeline semaphores + synchronization2 + sized types.
    // Only include features for extensions we're actually enabling.
    let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
    let mut timeline =
        vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
    let mut storage16 =
        vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true);
    let mut float16_int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
        .shader_float16(true)
        .shader_int8(true);

    // Optional features - only enable if extension is present.
    let mut storage8 =
        vk::PhysicalDevice8BitStorageFeatures::default().storage_buffer8_bit_access(has_8bit);
    let mut dot_product = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
        .shader_integer_dot_product(has_dot_product);

    let mut create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&ext_ptrs)
        .push_next(&mut sync2)
        .push_next(&mut timeline)
        .push_next(&mut storage16)
        .push_next(&mut float16_int8);

    // Only push optional feature structs if the extension is present.
    if has_8bit {
        create_info = create_info.push_next(&mut storage8);
    }
    if has_dot_product {
        create_info = create_info.push_next(&mut dot_product);
    }

    // SAFETY: pointers in create_info reference data alive on this stack
    // frame until vkCreateDevice returns.
    let device = unsafe { instance.create_device(pd, &create_info, None) }
        .map_err(|r| vk_err("create_device", r))?;
    Ok(device)
}

// ---------------------------------------------------------------------------
// Timeline semaphore for wait_idle
// ---------------------------------------------------------------------------

fn create_timeline_semaphore(device: &ash::Device) -> Result<vk::Semaphore> {
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
    // SAFETY: spec-compliant struct chain, no aliasing.
    let sem = unsafe { device.create_semaphore(&create_info, None) }
        .map_err(|r| vk_err("create_semaphore (timeline)", r))?;
    Ok(sem)
}
