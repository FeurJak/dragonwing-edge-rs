# Vulkan Backend (`dragonwing-vulkan`)

This document describes the Vulkan compute backend for dragonwing-edge, targeting the **Adreno A702** GPU on the Arduino UNO Q (Qualcomm QRB2210) via the **Mesa/Turnip** open-source driver.

## Overview

The Vulkan backend provides GPU-accelerated compute operations using Vulkan 1.0 with KHR extensions. It serves as the primary high-performance execution path for dragonwing-edge workloads.

### Key Characteristics

| Property | Value |
|----------|-------|
| Target GPU | Adreno A702 |
| Driver | Mesa Turnip 25.2.6 |
| Vulkan API | 1.0.318 |
| Memory Model | Unified (HOST_VISIBLE + DEVICE_LOCAL) |
| Synchronization | Timeline semaphores |

## Architecture

```
VulkanBackend          (public API — implements Backend trait)
   │
   ├── Context         (instance, physical device, logical device, queue)
   │     └── ash::Entry  (loader → libvulkan.so.1)
   │
   ├── VulkanBuffer    (per-buffer VkBuffer + VkDeviceMemory, persistent map)
   │
   └── PipelineCache   (VkPipelineCache, VkDescriptorPool, compute pipelines)
```

### Module Structure

- `lib.rs` — `VulkanBackend` struct implementing `dragonwing_core::Backend`
- `context.rs` — Vulkan instance/device setup, physical device selection
- `memory.rs` — `VulkanBuffer` with unified memory allocation
- `pipeline.rs` — `PipelineCache` managing shader pipelines
- `ops.rs` — Individual op implementations (fill, axpy, relu, gemm)
- `error.rs` — Error translation from Vulkan to dragonwing errors

## Device Selection

The backend automatically selects the best available physical device:

1. **Turnip on Adreno** (score +100) — preferred
2. **Integrated GPU** (score +10)
3. **Discrete GPU** (score +5)
4. **CPU fallback** (llvmpipe, score +1)

Force a specific device by setting `VulkanConfig::device_index`:

```rust
let config = VulkanConfig {
    device_index: Some(0),  // Force first device
    ..Default::default()
};
let backend = VulkanBackend::new(config)?;
```

## Required Extensions

The following extensions are required (device creation fails without them):

- `VK_KHR_storage_buffer_storage_class`
- `VK_KHR_synchronization2`
- `VK_KHR_timeline_semaphore`
- `VK_KHR_16bit_storage`
- `VK_KHR_shader_float16_int8`

Optional (enabled if present):

- `VK_KHR_8bit_storage` — Not available on Turnip A702
- `VK_KHR_shader_integer_dot_product` — Not available on Turnip A702

## Memory Model

The QRB2210 has **unified memory**: a single physical DRAM pool accessible to both CPU and GPU. The Vulkan memory type query returns a type with both `HOST_VISIBLE` and `DEVICE_LOCAL` flags.

Benefits:
- `upload`/`download` reduce to `memcpy` through persistent mapping
- No staging buffers needed
- Zero-copy potential for large tensors

### Buffer Allocation

```rust
let backend = VulkanBackend::new(VulkanConfig::default())?;
let mut buf = backend.alloc(1024 * 4, BufferKind::Storage)?;

// Upload data
backend.upload(&mut buf, &host_bytes)?;

// Run compute ops...
ops::fill_f32(&backend, &mut buf, 1.5)?;

// Synchronize and download
backend.synchronize()?;
backend.download(&buf, &mut output_bytes)?;
```

## Synchronization

The backend uses a single **timeline semaphore** for synchronization:

1. Each op submission increments the semaphore counter
2. `synchronize()` waits on the latest counter value
3. Download implicitly calls `synchronize()` first

This provides a simple serial execution model. Overlapping dispatches require a more sophisticated approach (future work).

## Compute Shaders

Shaders are written in GLSL 450 and compiled to SPIR-V via `glslangValidator`. Prebuilt `.spv` files are checked into `crates/dragonwing-shaders/spv/` for contributors without the Vulkan SDK.

### Shader Properties

| Op | local_size | Push Constants | Bindings |
|----|------------|----------------|----------|
| fill_f32 | 64×1×1 | n, value, pad×2 (16B) | 1 SSBO (output) |
| axpy_f32 | 64×1×1 | n, alpha, pad×2 (16B) | 2 SSBOs (Y r/w, X ro) |
| relu_f32 | 64×1×1 | n, pad×3 (16B) | 1 SSBO (in-place) |
| gemm_f32 | 16×16×1 | M, N, K, pad (16B) | 3 SSBOs (C, A, B) |

### Dispatch Calculations

```rust
// Element-wise ops (fill, axpy, relu)
let groups = n.div_ceil(64);  // local_size_x = 64

// GEMM
let groups_x = n.div_ceil(16);  // local_size_x = 16
let groups_y = m.div_ceil(16);  // local_size_y = 16
```

## Hardware Limits (Adreno A702)

From the task-001 hardware probe:

| Property | Value |
|----------|-------|
| maxComputeWorkGroupInvocations | 512 |
| maxComputeWorkGroupSize | [512, 512, 512] |
| maxComputeSharedMemorySize | 16 KiB |
| maxStorageBufferRange | 128 MiB |
| subgroupSize | 4 |

## Error Handling

All Vulkan errors are translated to `dragonwing_core::Error::Backend(String)` with descriptive messages:

```rust
match backend.alloc(size, BufferKind::Storage) {
    Ok(buf) => { /* use buffer */ }
    Err(Error::Backend(msg)) => eprintln!("Vulkan error: {msg}"),
    _ => unreachable!(),
}
```

## Usage Example

```rust
use dragonwing_core::{Backend, BufferKind};
use dragonwing_vulkan::{VulkanBackend, VulkanConfig, ops};

fn main() -> dragonwing_core::Result<()> {
    // Initialize backend
    let backend = VulkanBackend::new(VulkanConfig::default())?;
    println!("Using: {} ({})", backend.device_name(), backend.driver_info());

    // Allocate buffers
    let n = 1024;
    let mut a = backend.alloc(n * 4, BufferKind::Storage)?;
    let mut b = backend.alloc(n * 4, BufferKind::Storage)?;

    // Initialize with fill
    ops::fill_f32(&backend, &mut a, 1.0)?;
    ops::fill_f32(&backend, &mut b, 2.0)?;

    // Compute y = 3*x + y (axpy)
    ops::axpy_f32(&backend, &a, &mut b, 3.0)?;

    // Wait for completion
    backend.synchronize()?;

    // Download result
    let mut result = vec![0u8; n * 4];
    backend.download(&b, &mut result)?;

    Ok(())
}
```

## Known Issues

1. **Vulkan 1.0 KHR functions**: Timeline semaphore functions must use the KHR extension loader (`ash::khr::timeline_semaphore::Device`) since core functions don't exist on Vulkan 1.0.

2. **Optional extensions**: `VK_KHR_8bit_storage` is not available on Turnip A702. INT8 ops (task 003) may need a different approach.

3. **No overlapping dispatch**: Current model waits for each dispatch before the next. Pipeline parallelism is future work.

## Testing

Run the Vulkan-specific tests on device:

```bash
cargo build --release --bin vulkan-test
adb push target/release/vulkan-test /home/arduino/
adb shell /home/arduino/vulkan-test
```

Expected output:
```
=== dragonwing-vulkan test ===

Device: Turnip Adreno (TM) 702
Driver: turnip Mesa driver - Mesa 25.2.6-1~bpo13+1

Test fill_f32... PASS
Test axpy_f32... PASS
Test relu_f32... PASS
Test gemm_f32... PASS

=== Results: 4 passed, 0 failed ===
```

## Future Work (Task 003)

- [ ] FP16 compute ops using `shaderFloat16`
- [ ] INT8 dot product ops (if VK_KHR_8bit_storage becomes available)
- [ ] Tiled GEMM using shared memory
- [ ] Pipeline parallelism for overlapping dispatches
- [ ] Memory pool/slab allocator to reduce allocation overhead
