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
pub mod record;
pub mod slab;

use std::sync::Arc;

use dragonwing_core::{Backend, BackendBuffer, BufferKind, Error, Result};

pub use context::{Context, VulkanConfig};
pub use memory::VulkanBuffer;
pub use record::OpsRecorder;
pub use slab::{SlabAllocator, SlabAllocation, SlabStats};

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

    /// Find the preferred memory-type index for storage buffers on the
    /// current physical device.
    ///
    /// Mirrors the selection logic in `VulkanBuffer::new`: prefer
    /// `HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL` (the unified type
    /// on QRB2210), fall back to `HOST_VISIBLE | HOST_COHERENT`.
    ///
    /// `type_filter` should be the `memoryTypeBits` from
    /// `vkGetBufferMemoryRequirements` for a probe buffer of the same
    /// usage flags as production buffers. For a typical storage buffer
    /// this is identical across allocations, so `find_storage_memory_type`
    /// computes it from a one-byte probe buffer.
    pub fn find_storage_memory_type(&self) -> Result<u32> {
        // Create a small probe buffer to discover the memory type filter.
        let buffer_info = ash::vk::BufferCreateInfo::default()
            .size(64)
            .usage(
                ash::vk::BufferUsageFlags::STORAGE_BUFFER
                    | ash::vk::BufferUsageFlags::TRANSFER_DST
                    | ash::vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);
        // SAFETY: spec-compliant struct.
        let probe = unsafe { self.ctx.device().create_buffer(&buffer_info, None) }
            .map_err(|r| Error::Backend(format!("probe create_buffer: {r:?}")))?;
        // SAFETY: probe just created.
        let reqs = unsafe { self.ctx.device().get_buffer_memory_requirements(probe) };
        // SAFETY: probe owned by us, no work submitted.
        unsafe { self.ctx.device().destroy_buffer(probe, None) };

        memory::find_memory_type(
            self.ctx.instance(),
            self.ctx.physical_device(),
            reqs.memory_type_bits,
            ash::vk::MemoryPropertyFlags::HOST_VISIBLE
                | ash::vk::MemoryPropertyFlags::HOST_COHERENT
                | ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| {
            memory::find_memory_type(
                self.ctx.instance(),
                self.ctx.physical_device(),
                reqs.memory_type_bits,
                ash::vk::MemoryPropertyFlags::HOST_VISIBLE
                    | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })
        .ok_or_else(|| Error::Backend("no suitable storage memory type".into()))
    }

    /// Allocate a `VulkanBuffer` of `len_bytes` from a [`SlabAllocator`].
    ///
    /// The returned buffer holds a sub-range of a slab's `VkDeviceMemory`.
    /// It must be dropped **before** the `SlabAllocator` (typically by
    /// arranging both inside the same owning struct).
    ///
    /// Caller is responsible for tracking the returned `SlabAllocation`
    /// alongside the buffer if it later wants to call
    /// [`SlabAllocator::free`] explicitly. For graph-runtime workloads
    /// where allocations live as long as the runtime, freeing happens
    /// implicitly when the slab is dropped.
    pub fn alloc_slab_buffer(
        &self,
        slab: &SlabAllocator,
        len_bytes: usize,
    ) -> Result<(VulkanBuffer, SlabAllocation)> {
        if len_bytes == 0 {
            return Err(Error::Backend(
                "alloc_slab_buffer: zero-length buffers are not supported".into(),
            ));
        }
        let allocation = slab.alloc(len_bytes)?;
        let memory_handle = slab.memory(&allocation)?;
        let mapped = slab
            .mapped_ptr(&allocation)
            .ok_or_else(|| Error::Backend("alloc_slab_buffer: slab memory not host-visible".into()))?;
        // Use the caller's `len_bytes` (the logical size) for the
        // VkBuffer; the slab may have rounded up to MIN_ALIGNMENT (256 B)
        // and binding a larger VkBuffer than requested would make
        // `len_bytes()` return the wrong value for downstream size checks.
        let buffer = VulkanBuffer::from_slab(
            self.ctx.clone(),
            len_bytes,
            allocation.size(),
            memory_handle,
            allocation.offset(),
            mapped,
        )?;
        Ok((buffer, allocation))
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
