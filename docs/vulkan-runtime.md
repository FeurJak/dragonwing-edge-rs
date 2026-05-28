# Vulkan Runtime & Memory Management

This document describes the Vulkan runtime infrastructure and memory pooling system in dragonwing.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Vulkan Execution Stack                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 VulkanGraphRuntime                          │   │
│  │  - Graph execution                                          │   │
│  │  - Buffer management                                        │   │
│  │  - Command buffer batching                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    VulkanBackend                            │   │
│  │  - Pipeline management                                      │   │
│  │  - Compute dispatch                                         │   │
│  │  - Synchronization                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    SlabAllocator                            │   │
│  │  - Memory pooling                                           │   │
│  │  - Buffer aliasing                                          │   │
│  │  - First-fit allocation                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Slab Allocator

### Motivation

Vulkan has constraints that make per-buffer allocation problematic:

1. **Allocation limit**: ~4096 allocations on most drivers
2. **Driver overhead**: ~4 KB bookkeeping per allocation on Mesa
3. **Fragmentation**: Small allocations waste memory

YOLO models have 100+ tensors. Without pooling, we'd exhaust allocation limits.

### Design

The slab allocator pre-allocates large memory blocks ("slabs") and sub-allocates from them:

```rust
// crates/dragonwing-vulkan/src/slab.rs:52-61
const DEFAULT_SLAB_SIZE: usize = 16 * 1024 * 1024;  // 16 MiB
const MIN_ALIGNMENT: usize = 256;  // Vulkan storage buffer requirement
```

Memory layout:
```
Slab 0 (16 MiB)                 Slab 1 (16 MiB)
┌───────────────────────────┐   ┌───────────────────────────┐
│ [tensor A  ] [  free    ] │   │ [tensor D     ] [tensor E]│
│ [tensor B  ] [tensor C  ] │   │ [   free                 ]│
└───────────────────────────┘   └───────────────────────────┘
```

### Algorithm: First-Fit Free List

```rust
pub struct Slab {
    memory: vk::DeviceMemory,
    size: usize,
    mapped: Option<NonNull<u8>>,  // For HOST_VISIBLE memory
    free_blocks: BTreeMap<usize, usize>,  // offset -> size
    allocated: BTreeMap<usize, usize>,    // offset -> size
}
```

**Allocation**:
1. Scan free blocks in offset order
2. Find first block >= requested size (with alignment)
3. Split block if larger than needed
4. Return offset within slab

**Deallocation**:
1. Return block to free list
2. Coalesce with adjacent free blocks

### Usage

```rust
// Create allocator
let allocator = SlabAllocator::new(ctx, mem_type_index)?;

// Allocate buffer
let (slab_idx, offset) = allocator.allocate(size)?;
let memory = allocator.get_slab_memory(slab_idx);

// Bind buffer to memory at offset
vkBindBufferMemory(device, buffer, memory, offset);

// Free when done
allocator.free(slab_idx, offset)?;
```

### Memory Savings

For YOLOv8n at 640×640:

| Metric | Per-Buffer | Slab Allocator |
|--------|------------|----------------|
| vkAllocateMemory calls | ~150 | ~6 |
| Peak memory | ~150 MiB | ~95 MiB |
| Fragmentation | High | Low |

## Command Buffer Batching

### Problem

Per-op submission is expensive:
```rust
// Slow: one submit per op
for op in &graph.ops {
    record_op(cmd, op);
    vkQueueSubmit(...);  // Expensive!
    vkWaitForFences(...);
}
```

YOLO has ~200 ops. At ~100µs per submit, that's 20ms of overhead alone.

### Solution

Batch all ops into a single command buffer:

```rust
// Fast: one submit for entire graph
vkBeginCommandBuffer(cmd);
for op in &graph.ops {
    record_op(cmd, op);
    // Insert barriers between dependent ops
    if needs_barrier(op) {
        vkCmdPipelineBarrier(cmd, ...);
    }
}
vkEndCommandBuffer(cmd);
vkQueueSubmit(...);  // Single submit
vkWaitForFences(...);
```

### Memory Barriers

Read-after-write hazards require barriers between dependent ops:

```rust
fn insert_barrier(cmd: vk::CommandBuffer, device: &ash::Device) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}
```

### Timeline Semaphores

For async execution with proper ordering:

```rust
// Use VK_KHR_timeline_semaphore
let semaphore_type_info = vk::SemaphoreTypeCreateInfo::default()
    .semaphore_type(vk::SemaphoreType::TIMELINE)
    .initial_value(0);

// Signal value N after batch completes
let signal_info = vk::TimelineSemaphoreSubmitInfo::default()
    .signal_semaphore_values(&[next_value]);

// Wait for value N-1 before starting
let wait_info = vk::TimelineSemaphoreSubmitInfo::default()
    .wait_semaphore_values(&[current_value]);
```

## Pipeline Cache

Shader compilation is expensive (~100ms per pipeline). We cache compiled pipelines:

```rust
// crates/dragonwing-vulkan/src/pipeline.rs
// Load cache from disk
let cache_data = std::fs::read(cache_path).ok();
let cache = vkCreatePipelineCache(device, &cache_data)?;

// Create pipeline with cache
vkCreateComputePipelines(device, cache, &create_info, &mut pipeline)?;

// Save cache to disk after compilation
let data = vkGetPipelineCacheData(device, cache)?;
std::fs::write(cache_path, &data)?;
```

Cache file location:
```
$XDG_CACHE_HOME/dragonwing/pipeline-<device-uuid>-<shader-hash>.cache
```

### Cold vs Warm Start

| Metric | Cold Start | Warm Start |
|--------|------------|------------|
| Pipeline creation | ~3s | ~50ms |
| First inference | ~3.8s | ~800ms |

## Synchronization Model

```
                    Timeline
    ─────────────────────────────────────────────►

    Upload    Compute    Compute    Download
    ┌────┐    ┌────┐     ┌────┐    ┌────┐
    │ W  │    │ R→W│     │ R→W│    │ R  │
    └────┘    └────┘     └────┘    └────┘
       │         │          │         │
       └─────────┴──────────┴─────────┘
              Timeline Semaphore
              Values: 0 → 1 → 2 → 3
```

1. **Upload**: Write to staging buffer, signal value 1
2. **Compute ops**: Each batch waits for previous, signals next
3. **Download**: Wait for final compute, read result

## Vulkan 1.0.318 Constraints

The Arduino UNO Q runs Turnip Mesa driver at Vulkan 1.0.318:

| Feature | Status | Workaround |
|---------|--------|------------|
| Core 1.0 | Available | Use directly |
| Timeline semaphores | KHR extension | Use `ash::khr::timeline_semaphore` |
| 16-bit storage | KHR extension | FP16 ops work |
| 8-bit storage | **Not available** | INT8 needs UINT32 packing |
| Push constants | 128 bytes max | Keep under 64 bytes |

## Performance Tips

### 1. Minimize Memory Transfers

```rust
// Bad: transfer every frame
for frame in frames {
    upload(weights);    // Expensive!
    run_inference();
    download(output);
}

// Good: upload once
upload(weights);
for frame in frames {
    upload(input);      // Only input changes
    run_inference();
    download(output);
}
```

### 2. Use Unified Memory

The Adreno GPU has unified memory. Skip staging buffers when possible:

```rust
// Check for HOST_VISIBLE + DEVICE_LOCAL
let unified = mem_props.memory_types.iter().any(|t| {
    t.property_flags.contains(
        vk::MemoryPropertyFlags::HOST_VISIBLE |
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    )
});
```

### 3. Right-Size Workgroups

```glsl
// Good for Adreno A702
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Max is 512 total invocations
// 8×8×1 = 64 invocations per workgroup
```

## File References

- `crates/dragonwing-vulkan/src/slab.rs:1-470` — Slab allocator implementation
- `crates/dragonwing-vulkan/src/backend.rs:1-500` — VulkanBackend
- `crates/dragonwing-vulkan/src/pipeline.rs:1-400` — Pipeline cache
- `crates/dragonwing-vulkan/src/context.rs:1-200` — Vulkan context setup
