//! Vulkan buffer allocation and memory management.
//!
//! # Memory-type selection (QRB2210 / Turnip)
//!
//! The QRB2210 has **unified memory**: a single physical DRAM pool is
//! accessible to both CPU and GPU. The Vulkan memory-type query exposes
//! this via a memory type that has both `HOST_VISIBLE` and `DEVICE_LOCAL`
//! flags. We always prefer that combination so that:
//!
//! * `upload` / `download` reduce to `memcpy` through a persistent map.
//! * No staging buffer is needed (one fewer allocation + copy).
//! * Cache coherence is handled by the driver via `HOST_COHERENT`.
//!
//! If for some reason the unified type is not available we fall back to a
//! staging path (not yet implemented; errors out).
//!
//! # Allocation strategy
//!
//! Task 002 uses **one `vkAllocateMemory` per buffer**. This is simple
//! and correct but incurs Vulkan's per-allocation overhead (~4 KB driver
//! bookkeeping on Mesa). For the small number of buffers in our initial
//! ops this is fine; task 003 may replace this with a slab allocator
//! (e.g. `gpu-allocator`).
//!
//! # Safety invariants
//!
//! * `VulkanBuffer` holds an `Arc<Context>` to ensure the `VkDevice`
//!   outlives the buffer.
//! * The buffer is **persistently mapped** at construction; the pointer
//!   remains valid until `Drop`.
//! * Concurrent CPU access to the mapped range while GPU work is in
//!   flight is **undefined behaviour**. Callers must call
//!   [`VulkanBackend::synchronize`](crate::VulkanBackend::synchronize)
//!   before reading back results.

use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;
use dragonwing_core::{BackendBuffer, Error, Result};

use crate::context::Context;
use crate::error::vk_err;

/// A GPU-visible buffer backed by a single `VkBuffer` + `VkDeviceMemory`.
///
/// Implements [`BackendBuffer`] so it can be used with the generic
/// [`Backend`](dragonwing_core::Backend) trait.
pub struct VulkanBuffer {
    ctx: Arc<Context>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    /// Persistent host-mapped pointer (only valid when memory is
    /// HOST_VISIBLE). Wrapped in NonNull for !Send/!Sync safety;
    /// the actual pointer is *mut c_void.
    mapped: NonNull<u8>,
}

impl std::fmt::Debug for VulkanBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBuffer")
            .field("size", &self.size)
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

// SAFETY: VulkanBuffer's mapped pointer refers to driver-owned memory that
// is safe to access from any thread as long as no GPU work is in flight.
// The user is responsible for calling synchronize() before reading; we do
// not enforce this at the type level.
unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

impl VulkanBuffer {
    /// Allocate a new buffer of `size` bytes.
    ///
    /// The buffer is created with `STORAGE_BUFFER` usage and bound to a
    /// HOST_VISIBLE + DEVICE_LOCAL memory type. The memory is
    /// persistently mapped.
    pub fn new(ctx: Arc<Context>, size: usize) -> Result<Self> {
        let size = size as vk::DeviceSize;

        // 1. Create VkBuffer.
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: spec-compliant struct, device owned by ctx.
        let buffer = unsafe { ctx.device().create_buffer(&buffer_info, None) }
            .map_err(|r| vk_err("create_buffer", r))?;

        // 2. Query memory requirements.
        // SAFETY: buffer just created, device valid.
        let mem_reqs = unsafe { ctx.device().get_buffer_memory_requirements(buffer) };

        // 3. Find suitable memory type.
        let mem_type_index = find_memory_type(
            ctx.instance(),
            ctx.physical_device(),
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| {
            // Fallback: HOST_VISIBLE + HOST_COHERENT without DEVICE_LOCAL.
            find_memory_type(
                ctx.instance(),
                ctx.physical_device(),
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })
        .ok_or_else(|| {
            // Clean up buffer before returning error.
            // SAFETY: we own buffer, no work submitted.
            unsafe { ctx.device().destroy_buffer(buffer, None) };
            Error::Backend("vulkan: no suitable memory type for buffer".into())
        })?;

        // 4. Allocate memory.
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        // SAFETY: spec-compliant struct.
        let memory = unsafe { ctx.device().allocate_memory(&alloc_info, None) }.map_err(|r| {
            // Clean up buffer on failure.
            unsafe { ctx.device().destroy_buffer(buffer, None) };
            vk_err("allocate_memory", r)
        })?;

        // 5. Bind buffer to memory.
        // SAFETY: buffer and memory just created, not yet bound.
        unsafe { ctx.device().bind_buffer_memory(buffer, memory, 0) }.map_err(|r| {
            unsafe {
                ctx.device().free_memory(memory, None);
                ctx.device().destroy_buffer(buffer, None);
            }
            vk_err("bind_buffer_memory", r)
        })?;

        // 6. Map memory persistently.
        // SAFETY: memory is HOST_VISIBLE, size in range.
        let ptr = unsafe {
            ctx.device()
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .map_err(|r| {
            unsafe {
                ctx.device().free_memory(memory, None);
                ctx.device().destroy_buffer(buffer, None);
            }
            vk_err("map_memory", r)
        })?;

        let mapped = NonNull::new(ptr.cast::<u8>()).ok_or_else(|| {
            unsafe {
                ctx.device().unmap_memory(memory);
                ctx.device().free_memory(memory, None);
                ctx.device().destroy_buffer(buffer, None);
            }
            Error::Backend("vulkan: map_memory returned null".into())
        })?;

        Ok(Self {
            ctx,
            buffer,
            memory,
            size,
            mapped,
        })
    }

    /// Raw `VkBuffer` handle. Used by pipeline dispatch code.
    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// Raw `VkDeviceMemory` handle.
    pub fn vk_memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    /// Write `src` bytes into the buffer via the persistent map.
    ///
    /// # Safety contract
    ///
    /// The caller must ensure no GPU work is reading this buffer
    /// concurrently. Typically this means calling `synchronize()` before
    /// uploading to a buffer that was previously used in a dispatch.
    pub fn write_from_host(&mut self, src: &[u8]) -> Result<()> {
        if src.len() as vk::DeviceSize != self.size {
            return Err(Error::Backend(format!(
                "vulkan write_from_host size mismatch: buf={} src={}",
                self.size,
                src.len()
            )));
        }
        // SAFETY: mapped pointer valid, size matches, no concurrent GPU
        // access (caller's responsibility).
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.mapped.as_ptr(), src.len());
        }
        Ok(())
    }

    /// Read buffer contents into `dst` via the persistent map.
    ///
    /// # Safety contract
    ///
    /// The caller must ensure all GPU work writing this buffer has
    /// completed (call `synchronize()` first).
    pub fn read_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() as vk::DeviceSize != self.size {
            return Err(Error::Backend(format!(
                "vulkan read_to_host size mismatch: buf={} dst={}",
                self.size,
                dst.len()
            )));
        }
        // SAFETY: mapped pointer valid, size matches, GPU work completed
        // (caller's responsibility).
        unsafe {
            std::ptr::copy_nonoverlapping(self.mapped.as_ptr(), dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }
}

impl BackendBuffer for VulkanBuffer {
    fn len_bytes(&self) -> usize {
        self.size as usize
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        // SAFETY: we own these handles, and Drop has exclusive access.
        // Best practice: unmap before free, destroy buffer, then free
        // memory.
        unsafe {
            self.ctx.device().unmap_memory(self.memory);
            self.ctx.device().destroy_buffer(self.buffer, None);
            self.ctx.device().free_memory(self.memory, None);
        }
    }
}

/// Find a memory type index that satisfies `type_filter` (from
/// `memoryTypeBits`) and has all `required_flags`.
fn find_memory_type(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    type_filter: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    // SAFETY: read-only Vulkan getter.
    let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };
    for i in 0..mem_props.memory_type_count {
        let type_bit = 1 << i;
        if (type_filter & type_bit) == 0 {
            continue;
        }
        let flags = mem_props.memory_types[i as usize].property_flags;
        if flags.contains(required_flags) {
            return Some(i);
        }
    }
    None
}
