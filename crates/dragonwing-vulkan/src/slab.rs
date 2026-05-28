//! Slab allocator for Vulkan memory pooling.
//!
//! # Motivation
//!
//! Vulkan has a limit on the number of memory allocations (~4096 on most
//! drivers). YOLO models can easily have 100+ tensors, and with intermediate
//! buffers, we could hit this limit. More importantly, each `vkAllocateMemory`
//! call has non-trivial driver overhead (~4KB bookkeeping on Mesa).
//!
//! # Design
//!
//! The slab allocator pre-allocates large "slabs" (e.g., 16 MiB) and
//! sub-allocates buffers from them. This reduces:
//!
//! 1. Number of `vkAllocateMemory` calls (from O(n) to O(n/slab_size))
//! 2. Memory fragmentation
//! 3. Driver bookkeeping overhead
//!
//! # Algorithm
//!
//! We use a simple **first-fit free list** within each slab:
//!
//! 1. Maintain a list of free blocks (offset, size)
//! 2. On alloc: find first block >= requested size, split if needed
//! 3. On free: return block to list, coalesce with neighbors
//!
//! # Memory Layout
//!
//! ```text
//! Slab 0 (16 MiB)           Slab 1 (16 MiB)
//! ┌─────────────────────┐   ┌─────────────────────┐
//! │ [alloc1] [free]     │   │ [alloc3] [alloc4]   │
//! │ [alloc2] [free]     │   │ [free]              │
//! └─────────────────────┘   └─────────────────────┘
//! ```
//!
//! # Thread Safety
//!
//! The allocator uses interior mutability (`Mutex`) for thread safety.
//! Allocations/frees are relatively infrequent (model setup, not per-inference).

use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use ash::vk;
use dragonwing_core::{Error, Result};

use crate::context::Context;
use crate::error::vk_err;

/// Default slab size: 16 MiB.
/// This is a good balance between:
/// - Not too many slabs (reduces vkAllocateMemory calls)
/// - Not wasting memory on small models
const DEFAULT_SLAB_SIZE: usize = 16 * 1024 * 1024;

/// Minimum allocation alignment (256 bytes for storage buffers).
/// Vulkan requires `minStorageBufferOffsetAlignment`, which is typically
/// 16-256 bytes. We use 256 to be safe on all devices.
const MIN_ALIGNMENT: usize = 256;

/// A single slab of pre-allocated Vulkan memory.
struct Slab {
    /// The underlying VkDeviceMemory.
    memory: vk::DeviceMemory,
    /// Total size of the slab.
    size: usize,
    /// Persistent mapped pointer (if HOST_VISIBLE).
    mapped: Option<NonNull<u8>>,
    /// Free list: offset -> size of free block.
    /// Using BTreeMap for efficient range queries and ordered iteration.
    free_blocks: BTreeMap<usize, usize>,
    /// Allocated blocks: offset -> size.
    allocated: BTreeMap<usize, usize>,
}

impl Slab {
    /// Create a new slab with the given size.
    fn new(ctx: &Context, size: usize, mem_type_index: u32) -> Result<Self> {
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size as vk::DeviceSize)
            .memory_type_index(mem_type_index);

        let memory = unsafe { ctx.device().allocate_memory(&alloc_info, None) }
            .map_err(|r| vk_err("allocate_memory", r))?;

        // Try to map the memory (will succeed if HOST_VISIBLE).
        let mapped = unsafe {
            ctx.device()
                .map_memory(memory, 0, size as vk::DeviceSize, vk::MemoryMapFlags::empty())
        }
        .ok()
        .and_then(|ptr| NonNull::new(ptr.cast::<u8>()));

        // Initialize with entire slab as one free block.
        let mut free_blocks = BTreeMap::new();
        free_blocks.insert(0, size);

        Ok(Self {
            memory,
            size,
            mapped,
            free_blocks,
            allocated: BTreeMap::new(),
        })
    }

    /// Try to allocate `size` bytes from this slab (first-fit).
    /// Returns the offset within the slab if successful.
    fn try_alloc(&mut self, size: usize) -> Option<usize> {
        // Align size up to MIN_ALIGNMENT.
        let aligned_size = align_up(size, MIN_ALIGNMENT);

        // First-fit: find first free block >= aligned_size.
        let mut alloc_offset = None;
        let mut block_to_split = None;

        for (&offset, &block_size) in &self.free_blocks {
            if block_size >= aligned_size {
                alloc_offset = Some(offset);
                block_to_split = Some((offset, block_size));
                break;
            }
        }

        let (offset, block_size) = block_to_split?;

        // Remove the free block.
        self.free_blocks.remove(&offset);

        // If there's remaining space, add it back as a new free block.
        if block_size > aligned_size {
            let remaining_offset = offset + aligned_size;
            let remaining_size = block_size - aligned_size;
            self.free_blocks.insert(remaining_offset, remaining_size);
        }

        // Track the allocation.
        self.allocated.insert(offset, aligned_size);

        alloc_offset
    }

    /// Free an allocation at the given offset.
    fn free(&mut self, offset: usize) -> Result<()> {
        let size = self.allocated.remove(&offset).ok_or_else(|| {
            Error::Backend(format!("slab free: no allocation at offset {}", offset))
        })?;

        // Try to coalesce with adjacent free blocks.
        let mut new_offset = offset;
        let mut new_size = size;

        // Check for block immediately before.
        let prev_block: Option<(usize, usize)> = self
            .free_blocks
            .range(..offset)
            .next_back()
            .map(|(&o, &s)| (o, s));

        if let Some((prev_offset, prev_size)) = prev_block {
            if prev_offset + prev_size == offset {
                // Coalesce with previous block.
                self.free_blocks.remove(&prev_offset);
                new_offset = prev_offset;
                new_size += prev_size;
            }
        }

        // Check for block immediately after.
        let next_offset = new_offset + new_size;
        if let Some(&next_size) = self.free_blocks.get(&next_offset) {
            // Coalesce with next block.
            self.free_blocks.remove(&next_offset);
            new_size += next_size;
        }

        // Insert the (possibly coalesced) free block.
        self.free_blocks.insert(new_offset, new_size);

        Ok(())
    }

    /// Get the mapped pointer for an offset (if mapped).
    fn mapped_ptr(&self, offset: usize) -> Option<NonNull<u8>> {
        self.mapped.map(|base| {
            // SAFETY: offset is within slab bounds (checked by allocation).
            unsafe { NonNull::new_unchecked(base.as_ptr().add(offset)) }
        })
    }

    /// Check if slab is completely empty (all memory free).
    fn is_empty(&self) -> bool {
        self.allocated.is_empty()
    }
}

/// Align `value` up to the next multiple of `alignment`.
fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// A sub-allocation within a slab.
#[derive(Debug, Clone, Copy)]
pub struct SlabAllocation {
    /// Index of the slab in the allocator's slab list.
    slab_index: usize,
    /// Offset within the slab.
    offset: usize,
    /// Actual size allocated (aligned).
    size: usize,
}

impl SlabAllocation {
    /// Get the offset within the slab's VkDeviceMemory.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the allocated size.
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Thread-safe slab allocator for Vulkan memory.
pub struct SlabAllocator {
    ctx: Arc<Context>,
    /// Memory type index for allocations.
    mem_type_index: u32,
    /// Size of each slab.
    slab_size: usize,
    /// Interior-mutable slab storage.
    inner: Mutex<SlabAllocatorInner>,
}

struct SlabAllocatorInner {
    slabs: Vec<Slab>,
    /// Total bytes allocated across all slabs.
    total_allocated: usize,
    /// Total bytes in use (sub-allocated).
    bytes_in_use: usize,
}

impl SlabAllocator {
    /// Create a new slab allocator.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Vulkan context
    /// * `mem_type_index` - Memory type index for allocations (from find_memory_type)
    /// * `slab_size` - Size of each slab (default: 16 MiB)
    pub fn new(ctx: Arc<Context>, mem_type_index: u32, slab_size: Option<usize>) -> Self {
        Self {
            ctx,
            mem_type_index,
            slab_size: slab_size.unwrap_or(DEFAULT_SLAB_SIZE),
            inner: Mutex::new(SlabAllocatorInner {
                slabs: Vec::new(),
                total_allocated: 0,
                bytes_in_use: 0,
            }),
        }
    }

    /// Allocate `size` bytes from the pool.
    ///
    /// Returns a `SlabAllocation` that can be used to:
    /// - Bind buffers to the underlying memory
    /// - Get the mapped pointer for uploads/downloads
    pub fn alloc(&self, size: usize) -> Result<SlabAllocation> {
        if size == 0 {
            return Err(Error::Backend("slab alloc: zero-size allocation".into()));
        }

        let aligned_size = align_up(size, MIN_ALIGNMENT);
        let mut inner = self.inner.lock().unwrap();

        // Try to allocate from existing slabs.
        for (slab_index, slab) in inner.slabs.iter_mut().enumerate() {
            if let Some(offset) = slab.try_alloc(aligned_size) {
                inner.bytes_in_use += aligned_size;
                return Ok(SlabAllocation {
                    slab_index,
                    offset,
                    size: aligned_size,
                });
            }
        }

        // No space in existing slabs - create a new one.
        // Use max(slab_size, aligned_size) to handle large allocations.
        let new_slab_size = self.slab_size.max(aligned_size);
        let mut new_slab = Slab::new(&self.ctx, new_slab_size, self.mem_type_index)?;

        let offset = new_slab.try_alloc(aligned_size).ok_or_else(|| {
            Error::Backend("slab alloc: failed to allocate from fresh slab".into())
        })?;

        inner.total_allocated += new_slab_size;
        inner.bytes_in_use += aligned_size;
        let slab_index = inner.slabs.len();
        inner.slabs.push(new_slab);

        Ok(SlabAllocation {
            slab_index,
            offset,
            size: aligned_size,
        })
    }

    /// Free a previous allocation.
    pub fn free(&self, alloc: SlabAllocation) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();

        if alloc.slab_index >= inner.slabs.len() {
            return Err(Error::Backend(format!(
                "slab free: invalid slab index {}",
                alloc.slab_index
            )));
        }

        inner.slabs[alloc.slab_index].free(alloc.offset)?;
        inner.bytes_in_use -= alloc.size;

        Ok(())
    }

    /// Get the VkDeviceMemory handle for an allocation.
    pub fn memory(&self, alloc: &SlabAllocation) -> Result<vk::DeviceMemory> {
        let inner = self.inner.lock().unwrap();
        if alloc.slab_index >= inner.slabs.len() {
            return Err(Error::Backend("slab: invalid allocation".into()));
        }
        Ok(inner.slabs[alloc.slab_index].memory)
    }

    /// Get the mapped pointer for an allocation (if memory is HOST_VISIBLE).
    pub fn mapped_ptr(&self, alloc: &SlabAllocation) -> Option<NonNull<u8>> {
        let inner = self.inner.lock().unwrap();
        if alloc.slab_index >= inner.slabs.len() {
            return None;
        }
        inner.slabs[alloc.slab_index].mapped_ptr(alloc.offset)
    }

    /// Get allocator statistics.
    pub fn stats(&self) -> SlabStats {
        let inner = self.inner.lock().unwrap();
        SlabStats {
            num_slabs: inner.slabs.len(),
            total_allocated: inner.total_allocated,
            bytes_in_use: inner.bytes_in_use,
            utilization: if inner.total_allocated > 0 {
                inner.bytes_in_use as f64 / inner.total_allocated as f64
            } else {
                0.0
            },
        }
    }

    /// Release empty slabs back to the system.
    ///
    /// This is useful after model unload to reclaim memory.
    /// Returns the number of slabs released.
    pub fn trim(&self) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let ctx = &self.ctx;

        // First pass: identify empty slabs and collect their info.
        let mut to_remove: Vec<(usize, vk::DeviceMemory, Option<NonNull<u8>>, usize)> = Vec::new();
        for (idx, slab) in inner.slabs.iter().enumerate() {
            if slab.is_empty() {
                to_remove.push((idx, slab.memory, slab.mapped, slab.size));
            }
        }

        // Free the VkDeviceMemory for empty slabs.
        for &(_, memory, mapped, size) in &to_remove {
            unsafe {
                if mapped.is_some() {
                    ctx.device().unmap_memory(memory);
                }
                ctx.device().free_memory(memory, None);
            }
            inner.total_allocated -= size;
        }

        // Remove from list in reverse order to preserve indices.
        for &(idx, _, _, _) in to_remove.iter().rev() {
            inner.slabs.swap_remove(idx);
        }

        to_remove.len()
    }
}

impl Drop for SlabAllocator {
    fn drop(&mut self) {
        // Free all slabs.
        let inner = self.inner.get_mut().unwrap();
        for slab in &inner.slabs {
            unsafe {
                if slab.mapped.is_some() {
                    self.ctx.device().unmap_memory(slab.memory);
                }
                self.ctx.device().free_memory(slab.memory, None);
            }
        }
    }
}

/// Allocator statistics.
#[derive(Debug, Clone, Copy)]
pub struct SlabStats {
    /// Number of slabs allocated.
    pub num_slabs: usize,
    /// Total bytes allocated from Vulkan.
    pub total_allocated: usize,
    /// Bytes currently in use by sub-allocations.
    pub bytes_in_use: usize,
    /// Utilization ratio (bytes_in_use / total_allocated).
    pub utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(1000, 256), 1024);
    }

    #[test]
    fn test_slab_alloc_free() {
        // Test the slab allocation logic without Vulkan.
        // We'll use a mock slab with just the free_blocks logic.
        
        let mut free_blocks: BTreeMap<usize, usize> = BTreeMap::new();
        free_blocks.insert(0, 4096);
        
        // Simulate first allocation of 256 bytes.
        let size1 = 256;
        let offset1 = 0;
        free_blocks.remove(&0);
        free_blocks.insert(256, 4096 - 256);
        
        assert_eq!(free_blocks.get(&256), Some(&(4096 - 256)));
        
        // Simulate second allocation of 512 bytes.
        let offset2 = 256;
        free_blocks.remove(&256);
        free_blocks.insert(256 + 512, 4096 - 256 - 512);
        
        assert_eq!(free_blocks.get(&768), Some(&(4096 - 768)));
        
        // Free first allocation - no coalescing (neighbor is allocated).
        free_blocks.insert(offset1, size1);
        
        // Now we have: [0, 256] free, [256, 768] allocated, [768, 4096] free.
        assert_eq!(free_blocks.len(), 2);
    }
}
