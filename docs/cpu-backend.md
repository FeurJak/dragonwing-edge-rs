# CPU Backend (`dragonwing-cpu`)

This document describes the CPU reference backend for dragonwing-edge. It serves as both a correctness oracle and a fallback execution path when no GPU is available.

## Overview

The CPU backend implements `dragonwing_core::Backend` using host memory and NEON SIMD intrinsics on AArch64. It provides the reference implementations against which GPU backends are validated.

### Key Characteristics

| Property | Value |
|----------|-------|
| Target | AArch64 (Cortex-A53 on QRB2210) |
| SIMD | NEON (128-bit vectors) |
| Threading | Single-threaded (task 003 adds parallelism) |
| Memory | Standard heap allocation |

## Architecture

The CPU backend is simple:

```
CpuBackend     (implements Backend trait)
   │
   └── CpuBuffer    (Vec<u8> wrapper with f32 view methods)
```

### Module Structure

- `lib.rs` — `CpuBackend` and `CpuBuffer` types
- `ops.rs` — NEON-accelerated op implementations

## Op Implementations

All ops work on `&mut [f32]` slices directly:

| Op | Signature | NEON Intrinsics |
|----|-----------|-----------------|
| `fill_f32` | `(y: &mut [f32], v: f32)` | `vdupq_n_f32`, `vst1q_f32` |
| `axpy_f32` | `(y: &mut [f32], a: f32, x: &[f32])` | `vld1q_f32`, `vfmaq_f32`, `vst1q_f32` |
| `relu_f32` | `(y: &mut [f32], x: &[f32])` | `vld1q_f32`, `vmaxq_f32`, `vst1q_f32` |
| `gemm_f32_naive` | `(c, a, b, m, n, k)` | Scalar (NEON gemm is future work) |

### NEON Vectorization

Ops process 4 floats at a time using 128-bit NEON registers:

```rust
// fill_f32 inner loop (simplified)
let v4 = vdupq_n_f32(value);  // Broadcast scalar to 4-lane vector
for chunk in y.chunks_exact_mut(4) {
    vst1q_f32(chunk.as_mut_ptr(), v4);  // Store 4 floats
}
```

### Scalar Fallback

For non-AArch64 targets (x86_64 dev machines), ops fall back to portable scalar implementations:

```rust
#[cfg(not(target_arch = "aarch64"))]
pub fn fill_f32(y: &mut [f32], v: f32) {
    y.fill(v);
}
```

## Buffer API

`CpuBuffer` provides type-safe views into raw bytes:

```rust
let backend = CpuBackend::new();
let mut buf = backend.alloc(1024 * 4, BufferKind::Storage)?;

// Get f32 views
let floats: &[f32] = buf.as_f32();
let floats_mut: &mut [f32] = buf.as_f32_mut();

// Get raw bytes
let bytes: &[u8] = buf.as_bytes();
let bytes_mut: &mut [u8] = buf.as_bytes_mut();
```

### Upload/Download

For CPU backend, upload/download are simple memcpy operations:

```rust
// Upload host data to buffer
backend.upload(&mut buf, &host_bytes)?;

// Download buffer to host
backend.download(&buf, &mut host_bytes)?;
```

## Usage Example

```rust
use dragonwing_cpu::{CpuBackend, ops};
use dragonwing_core::{Backend, BufferKind};

fn main() -> dragonwing_core::Result<()> {
    let backend = CpuBackend::new();
    
    // Allocate and initialize
    let n = 1024;
    let mut buf = backend.alloc(n * 4, BufferKind::Storage)?;
    
    // Work directly with slices
    ops::fill_f32(buf.as_f32_mut(), 1.0);
    
    // Or use Backend trait methods
    let host_data = vec![0u8; n * 4];
    backend.upload(&mut buf, &host_data)?;
    
    Ok(())
}
```

## Direct Slice Operations

For maximum performance, use ops directly on slices without going through buffers:

```rust
use dragonwing_cpu::ops;

let mut y = vec![0.0f32; 1024];
let x = vec![1.0f32; 1024];

// NEON-accelerated ops on raw slices
ops::fill_f32(&mut y, 2.0);
ops::axpy_f32(&mut y, 3.0, &x);  // y = 3*x + y
ops::relu_f32(&mut y, &x);       // y = max(0, x)
```

## Why `unsafe`?

The NEON intrinsics in `std::arch::aarch64` are `unsafe fn` because the compiler cannot verify the target CPU supports them. On the UNO Q, we've verified NEON is available via the hardware probe (`cpu.features` contains `"asimd"`).

Every `unsafe` block is annotated with a `// SAFETY:` comment explaining the invariant:

```rust
// SAFETY: target_arch = "aarch64" guarantees NEON is available.
// The pointer and length come from a valid &mut [f32] slice.
unsafe {
    let v4 = vdupq_n_f32(value);
    vst1q_f32(ptr, v4);
}
```

## Performance Notes

1. **NEON is 4× wider than scalar**: Processing 4 floats per instruction provides significant speedup for large arrays.

2. **Tail handling**: Ops handle arrays not divisible by 4 with scalar cleanup loops.

3. **FMA precision**: `axpy_f32` uses `vfmaq_f32` (fused multiply-add) which performs `a*b+c` with a single rounding, matching the GPU's `fma()` behavior for parity.

4. **GEMM is scalar**: The naive GEMM implementation doesn't use NEON yet. Task 003 will add NEON-tiled GEMM.

## Role in Testing

The CPU backend is the **correctness oracle** for parity testing. When CPU and Vulkan results differ, the CPU result is considered correct unless proven otherwise.

This works because:
- CPU ops are simpler and easier to verify
- FP32 behavior is well-understood on ARM
- No driver/firmware bugs to worry about

## Testing

Run CPU op tests:

```bash
cargo test -p dragonwing-cpu
```

Expected output:
```
running 4 tests
test tests::test_fill_f32 ... ok
test tests::test_axpy_f32 ... ok
test tests::test_relu_f32 ... ok
test tests::test_gemm_f32_naive ... ok
```

## Future Work (Task 003)

- [ ] Multi-threaded dispatch with work-stealing scheduler
- [ ] NEON-tiled GEMM implementation
- [ ] FP16 ops using NEON half-precision intrinsics
- [ ] INT8 quantized ops
