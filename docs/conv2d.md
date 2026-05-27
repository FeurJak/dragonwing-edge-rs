# Convolution and Pooling Ops

This document describes the 2D convolution and pooling operations added in task 003.

## Overview

Task 003 ships layer-level ops needed to compose vision graphs:

| Op | F32 | FP16 | CPU | Vulkan |
|----|-----|------|-----|--------|
| `conv2d_*_nhwc` | ✓ | ✓ (CPU only) | ✓ | F32 only |
| `maxpool2d_f32_nhwc` | ✓ | — | ✓ | ✓ |
| `avgpool2d_f32_nhwc` | ✓ | — | ✓ | — |
| `softmax_f32` | ✓ | — | ✓ | ✓ |

All ops use **NHWC** layout (no silent transpose). The natural stride layout is
`((n*H + h)*W + w)*C + c` — easy to inline, easy to debug.

## Conv2D

### Algorithm

Direct convolution only in this task. Each output element is computed by one
work item:

```
for batch in 0..N:
  for oh in 0..H_out:
    for ow in 0..W_out:
      for oc in 0..C_out:
        acc = 0
        for kh in 0..K_h:
          for kw in 0..K_w:
            for ic in 0..C_in:
              ih = oh*stride_h - pad_h + kh
              iw = ow*stride_w - pad_w + kw
              if in_bounds(ih, iw):
                acc += input[n,ih,iw,ic] * kernel[oc,kh,kw,ic]
        output[n,oh,ow,oc] = acc
```

Im2col + GEMM is **not** in this task. Direct is correct for all kernel sizes
and clearly faster for 1×1 and 3×3 — task 003's bench either confirms or
overturns the small-kernel cutoff that task 004 must respect.

### Parameters

Conv2D is parameterised by:

| Param | Meaning |
|-------|---------|
| `N` | Batch size |
| `H_in`, `W_in` | Input spatial dimensions |
| `C_in` | Input channels |
| `C_out` | Output channels |
| `K_h`, `K_w` | Kernel spatial dimensions |
| `stride_h`, `stride_w` | Spatial stride |
| `pad_h`, `pad_w` | Implicit zero padding (top/left, computed bottom/right) |

Output dimensions are `H_out = (H_in + 2*pad_h - K_h)/stride_h + 1` and
similarly for `W_out`. The caller is responsible for sizing the output buffer
to `N * H_out * W_out * C_out * dtype.size_bytes()`.

### Tensor layout (NHWC)

Stride math is consistent across input, output, and kernel:

- **Input**: `[N][H_in][W_in][C_in]` → `((n*H_in + ih)*W_in + iw)*C_in + ic`
- **Output**: `[N][H_out][W_out][C_out]` → `((n*H_out + oh)*W_out + ow)*C_out + oc`
- **Kernel**: `[C_out][K_h][K_w][C_in]` → `((oc*K_h + kh)*K_w + kw)*C_in + ic`

The kernel layout is OHWI (output-major). This matches the inner loop's access
pattern: for a fixed output position the kernel is walked contiguously in
`(kh, kw, ic)`.

## Vulkan implementation

### Push constants (64 bytes)

Task 002 used 16-byte push constants. Conv2D needs 12 dimension parameters and
spills the budget; the project rule was relaxed for the Phase 5 ops:

```rust
#[repr(C)]
struct PushConv {
    dims0: [u32; 4],  // h_in, w_in, c_in, c_out
    dims1: [u32; 4],  // k_h, k_w, stride_h, stride_w
    dims2: [u32; 4],  // pad_h, pad_w, h_out, w_out
    dims3: [u32; 4],  // n, _pad, _pad, _pad
}
```

64 bytes is well under the Vulkan spec's 128-byte guaranteed minimum and the
A702's actual `maxPushConstantsSize`. Anything bigger should go in a uniform
buffer — push constants beyond ~128 bytes are a portability liability.

### Workgroup geometry

| Op | `local_size` | Threads per group | Notes |
|----|--------------|--------------------|-------|
| `conv2d_f32_nhwc` | (8, 8, 1) | 64 | One output `(h, w)` per thread; `gz = n * c_out` |
| `maxpool2d_f32` | (8, 8, 1) | 64 | Same dispatch shape as conv |
| `softmax_f32` | (64, 1, 1) | 64 | One row per workgroup, lanes reduce in shared memory |

The 8×8 choice for conv/pool balances:
- **Occupancy** (well under the 512 invocation cap)
- **Register pressure** (each thread holds its accumulator + indexing locals)
- **Spatial locality** (8×8 = 64 output positions read overlapping kernel data)

### Shader pattern

Each conv thread computes one output element. The `(oh, ow)` come from
`gl_GlobalInvocationID.xy`; the `(batch, oc)` come from
`gl_GlobalInvocationID.z` decomposed as `batch = gid_z / C_out`,
`oc = gid_z % C_out`. The accumulator is FP32 even for the FP16 variant
(when it lands).

See `crates/dragonwing-shaders/glsl/conv2d_f32_nhwc.comp` for the canonical
implementation.

## Pooling

### `maxpool2d_f32_nhwc`

Standard 2D max pool. Each thread:

1. Maps `(gl_GlobalInvocationID.xy)` to an output spatial position.
2. Reads the pool window from the input (no shared memory — windows are small,
   typically 2×2 or 3×3, and non-overlapping with `stride == pool_size`).
3. Reduces with `max()`.
4. Writes a single output element.

No padding support in this task — the caller must size inputs so the pool
window stays in-bounds. (Padded pool is a 3-line change but the GLSL gets
ugly fast; deferred.)

### `avgpool2d_f32_nhwc` (CPU only)

Same dispatch pattern as max pool, but accumulates and divides by the window
size. Vulkan variant is deferred to task 004 (real graphs use global average
pool, which is a different shape and gets its own op).

## Softmax

### Algorithm: three-pass with shared-memory reduction

Numerical stability requires subtracting the row max before `exp`:

1. **Pass 1 — row max**: each thread finds the max over its strided portion of
   the row; workgroup reduction in shared memory.
2. **Pass 2 — `exp(x - max)` and sum**: each thread accumulates a partial sum
   of the unnormalised exponentials; workgroup reduction in shared memory.
3. **Pass 3 — normalise**: each thread divides its slice by the reduced sum.

The 64-thread workgroup processes rows up to about 4K elements efficiently
(each thread loops over `n / 64` elements). Beyond that, a multi-workgroup
reduction would be needed — for vocabularies of 32k+ (LLM logits), the current
implementation is correct but suboptimal. Real models in task 004's scope
(MobileNetV2, YOLO classification head) have ≤1k classes.

### Why not the subgroup intrinsic?

Adreno A702's `subgroupSize` is **4**, not the 32 or 64 found on desktop GPUs.
Subgroup reductions collapse only 4 lanes; a workgroup-level reduction through
shared memory is mandatory. This is the same reason `gemm_f32_tiled` uses
shared memory for the K-strip rather than relying on subgroup operations.

## CPU implementations

### NEON paths

All F32 ops have NEON variants. The patterns from task 002 carry over:

- 4-wide vector unit (`float32x4_t`)
- 16-wide unrolled inner loop (4 vector instructions per iteration)
- Tail handling via scalar fallback
- `vfmaq_f32` for single-rounded FMA matching the scalar `f32::mul_add`

The conv NEON path uses register tiling on the `oc` dimension: each iteration
of the inner loop reads four kernel values (one vector load) and adds them
into four accumulators.

### Multi-threaded variants

`gemm_f32_mt`, `gemm_fp16_mt`, `conv2d_f32_nhwc_mt`, and `conv2d_fp16_nhwc_mt`
partition work across the thread pool (see `docs/cpu-multithreading.md`).

## Parity policy

| Op | Tolerance | Rationale |
|----|-----------|-----------|
| `conv2d_f32` | 1e-4 relative | Accumulation order may differ; depth-K of conv is `K_h * K_w * C_in` |
| `conv2d_fp16` | 5e-3 relative | FP16 mantissa precision |
| `maxpool2d` | 1e-5 | Selection op, no arithmetic; small diff possible only from boundary handling |
| `avgpool2d` | 1e-5 | Window sums are short (typically 4 elements for 2×2) |
| `softmax_f32` | 1e-4 | `exp` is the dominant error source; intermediate sums in FP32 |

The micro-graph end-to-end test (`crates/dragonwing-test/src/micro_graph.rs`)
exercises all of these in sequence with a `1e-4` per-op tolerance.

## Known limitations

| Limitation | Affects | Workaround | Target |
|------------|---------|------------|--------|
| No im2col path | Large kernels (5×5+) suboptimal | Use 3×3 stacks instead | Task 004 |
| Maxpool has no padding | Caller must size inputs so windows fit | Pad input manually | Task 004 |
| Softmax limited to ~4K elements/row | Large vocabularies | Multi-workgroup reduction | Task 005 |
| No FP16 conv on Vulkan | FP16 path stops at CPU | Use F32 on Vulkan | Task 004 |
| No fused conv-bias-relu | Three ops, three memory round-trips | Run separately | Task 004 |

## Local references

- `crates/dragonwing-cpu/src/ops.rs::conv2d_f32_nhwc` (single-thread) and `::conv2d_f32_nhwc_mt` (multi-thread).
- `crates/dragonwing-cpu/src/ops.rs::maxpool2d_f32_nhwc`, `::avgpool2d_f32_nhwc`, `::softmax_f32`.
- `crates/dragonwing-vulkan/src/ops.rs::conv2d_f32_nhwc`, `::maxpool2d_f32`, `::softmax_f32`.
- `crates/dragonwing-shaders/glsl/conv2d_f32_nhwc.comp` — direct convolution shader.
- `crates/dragonwing-shaders/glsl/maxpool2d_f32.comp` — max-pool shader.
- `crates/dragonwing-shaders/glsl/softmax_f32.comp` — three-pass softmax with shared-mem reduction.
- `crates/dragonwing-test/src/micro_graph.rs` — end-to-end MNIST-shaped graph that wires all of the above.
