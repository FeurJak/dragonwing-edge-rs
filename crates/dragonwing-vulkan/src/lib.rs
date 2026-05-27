//! `dragonwing-vulkan` — Vulkan 1.1 compute backend.
//!
//! Targets the **Adreno A702** GPU on the Qualcomm QRB2210, reached
//! through the **Mesa/Turnip** Vulkan driver. The same binary will also
//! enumerate `llvmpipe` (Mesa's CPU fallback) on host development
//! machines that don't have Turnip; that's intentional and is handled by
//! the physical-device selection logic in [`context::Context`].
//!
//! # Architecture
//!
//! ```text
//!   VulkanBackend          (public — implements dragonwing_core::Backend)
//!      │
//!      ├── Context         (instance, physical device, logical device, queue)
//!      │     └── ash::Entry  (loader → libvulkan.so.1)
//!      │
//!      ├── Allocator       (per-buffer vkAllocateMemory; replace later)
//!      │
//!      └── PipelineCache   (one VkPipelineCache, one descriptor pool,
//!                           one compute pipeline per shader)
//! ```
//!
//! See `docs/vulkan-backend.md` for a longer walkthrough including memory-type
//! selection, synchronisation strategy, and known Turnip quirks.
//!
//! # Quickstart
//!
//! ```no_run
//! use dragonwing_core::{Backend, BackendBuffer, BufferKind};
//! use dragonwing_vulkan::{VulkanBackend, VulkanConfig};
//!
//! let backend = VulkanBackend::new(VulkanConfig::default())
//!     .expect("Vulkan init failed");
//! println!("Selected device: {}", backend.device_name());
//!
//! let n = 1024;
//! let mut buf = backend.alloc(n * 4, BufferKind::Storage).unwrap();
//! dragonwing_vulkan::ops::fill_f32(&backend, &mut buf, 1.5).unwrap();
//! backend.synchronize().unwrap();
//!
//! let mut readback = vec![0u8; buf.len_bytes()];
//! backend.download(&buf, &mut readback).unwrap();
//! ```
//!
//! # Memory model (QRB2210 specific)
//!
//! The QRB2210 has unified host/device memory: the same physical DRAM
//! backs both CPU and GPU. The Vulkan memory-type query reports a single
//! heap (~870 MiB usable at probe time) with at least one type that has
//! both `HOST_VISIBLE` and `DEVICE_LOCAL` flags set. We always pick that
//! combination so `upload`/`download` reduce to `memcpy` through a
//! persistent map — no staging buffer needed. The allocator falls back
//! to staging if the unified type is somehow unavailable; this hasn't
//! been observed on Turnip.
//!
//! # Synchronisation
//!
//! Each op submission allocates a new timeline-semaphore signal value.
//! [`VulkanBackend::synchronize`](VulkanBackend) waits on the latest
//! signal. This is simpler than tracking per-buffer waits and matches the
//! sequential-submission style we use today (one op per submit).
//!
//! # Known Turnip-on-A702 properties (from task 001 hardware probe)
//!
//! * `subgroupSize = 4` — workgroup-level reductions through shared
//!   memory are mandatory for sums.
//! * `maxComputeWorkGroupInvocations = 512` — 8×8×8, 16×16 OK; 32×32 not.
//! * `maxComputeSharedMemorySize = 16 KiB` — gemm tile sizes constrained.
//! * `maxStorageBufferRange = 128 MiB` — per-binding limit.
//! * `shaderFloat16` / `shaderInt8` / `shaderIntegerDotProduct` available.
//!   Task 002 ships FP32 only; these are reserved for task 003.

#![warn(missing_docs)]

pub mod context;
pub mod error;
pub mod memory;
pub mod ops;
pub mod pipeline;

use std::sync::Arc;

use dragonwing_core::{Backend, BackendBuffer, BufferKind, Error, Result};

pub use context::{Context, VulkanConfig};
pub use memory::VulkanBuffer;

/// The Vulkan compute backend.
///
/// Cheap to clone (`Arc` internally). All clones share the same
/// `VkInstance`, `VkDevice`, and pipeline cache.
#[derive(Clone)]
pub struct VulkanBackend {
    ctx: Arc<Context>,
    pipelines: Arc<pipeline::PipelineCache>,
}

impl std::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBackend")
            .field("device", &self.ctx.device_name())
            .field("driver", &self.ctx.driver_info())
            .finish()
    }
}

impl VulkanBackend {
    /// Construct a new backend. Loads `libvulkan.so.1`, creates an
    /// instance + device, and primes the pipeline cache.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * `libvulkan.so.1` is not loadable (the OS Vulkan loader is missing).
    /// * No physical device matches the selection criteria
    ///   (`config.prefer_device_type`).
    /// * Required extensions or features are not supported by the chosen
    ///   device.
    pub fn new(config: VulkanConfig) -> Result<Self> {
        let ctx = Arc::new(Context::new(config)?);
        let pipelines = Arc::new(pipeline::PipelineCache::new(ctx.clone())?);
        Ok(Self { ctx, pipelines })
    }

    /// Human-readable device name as reported by Vulkan, e.g.
    /// `"Turnip Adreno (TM) 702"`.
    pub fn device_name(&self) -> &str {
        self.ctx.device_name()
    }

    /// Driver name + driver info string, e.g.
    /// `"turnip Mesa driver - Mesa 25.2.6"`.
    pub fn driver_info(&self) -> &str {
        self.ctx.driver_info()
    }

    /// Borrow the shared context. Most user code does not need this; it's
    /// exposed so that op modules in `dragonwing-vulkan::ops::*` can
    /// access raw Vulkan handles.
    pub fn context(&self) -> &Arc<Context> {
        &self.ctx
    }

    /// Borrow the shared pipeline cache.
    pub fn pipelines(&self) -> &Arc<pipeline::PipelineCache> {
        &self.pipelines
    }
}

impl Backend for VulkanBackend {
    type Buffer = VulkanBuffer;

    fn name(&self) -> &'static str {
        // We can't return self.ctx.driver_name() here because Backend::name
        // requires &'static str. The HAL/probe report carries the actual
        // driver string when needed.
        "vulkan-turnip"
    }

    fn alloc(&self, len_bytes: usize, _kind: BufferKind) -> Result<Self::Buffer> {
        if len_bytes == 0 {
            return Err(Error::Backend(
                "Vulkan alloc: zero-length buffers are not supported".into(),
            ));
        }
        VulkanBuffer::new(self.ctx.clone(), len_bytes)
    }

    fn upload(&self, dst: &mut Self::Buffer, src: &[u8]) -> Result<()> {
        if dst.len_bytes() != src.len() {
            return Err(Error::Backend(format!(
                "Vulkan upload size mismatch: dst={} src={}",
                dst.len_bytes(),
                src.len()
            )));
        }
        dst.write_from_host(src)
    }

    fn download(&self, src: &Self::Buffer, dst: &mut [u8]) -> Result<()> {
        if src.len_bytes() != dst.len() {
            return Err(Error::Backend(format!(
                "Vulkan download size mismatch: src={} dst={}",
                src.len_bytes(),
                dst.len()
            )));
        }
        // Wait on prior work before reading.
        self.synchronize()?;
        src.read_to_host(dst)
    }

    fn synchronize(&self) -> Result<()> {
        self.ctx.wait_idle()
    }
}

// Re-export commonly used types so users don't need to import ash directly.
pub use ash::vk::PhysicalDeviceType;
