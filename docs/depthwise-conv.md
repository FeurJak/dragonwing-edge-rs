# Depthwise Convolution

This document describes the depthwise convolution implementation in dragonwing, used by MobileNet-style efficient architectures.

## What is Depthwise Convolution?

Standard convolution: Each output channel is a weighted sum of ALL input channels.
```
Output[c_out] = sum over c_in: Conv(Input[c_in], Weight[c_out, c_in])
```

Depthwise convolution: Each input channel is convolved independently.
```
Output[c] = Conv(Input[c], Weight[c])
```

This reduces computation from `O(C_in * C_out * K^2 * H * W)` to `O(C * K^2 * H * W)`.

## ONNX Representation

In ONNX, depthwise conv is represented as a standard `Conv` op with `group == C_in`:

```
Conv {
    inputs: [X, W, B?]
    attributes: {
        group: 32,      // == C_in for depthwise
        kernel_shape: [3, 3],
        strides: [1, 1],
        pads: [1, 1, 1, 1],
    }
}
```

Weight shape: `[C_out, 1, kH, kW]` (C_in per group = 1)

## Detection Logic

```rust
// crates/dragonwing-onnx/src/builder.rs (ConvBuilder)
let group = node.get_attr_int("group", 1) as usize;
let is_depthwise = group == c_in && group > 1;
```

The builder routes to different kernels based on this:

| Condition | Kernel |
|-----------|--------|
| `group == 1` | Standard conv2d |
| `group == c_in && group > 1` | Depthwise conv2d |
| `group > 1 && group != c_in` | Grouped conv (not supported) |

## CPU Implementation

### Scalar Reference

```rust
// crates/dragonwing-cpu/src/ops.rs:1400-1500
pub fn depthwise_conv2d_f32_nhwc(
    output: &mut [f32],
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    n: usize, h_in: usize, w_in: usize, c: usize,
    k_h: usize, k_w: usize,
    stride_h: usize, stride_w: usize,
    pad_t: usize, pad_l: usize,
) {
    let h_out = (h_in + pad_t + pad_b - k_h) / stride_h + 1;
    let w_out = (w_in + pad_l + pad_r - k_w) / stride_w + 1;

    for batch in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                for ch in 0..c {
                    let mut sum = 0.0f32;
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;
                            if ih >= pad_t && ih < h_in + pad_t &&
                               iw >= pad_l && iw < w_in + pad_l {
                                let in_idx = ((batch * h_in + ih - pad_t) * w_in + iw - pad_l) * c + ch;
                                let k_idx = (kh * k_w + kw) * c + ch;
                                sum += input[in_idx] * kernel[k_idx];
                            }
                        }
                    }
                    if let Some(b) = bias {
                        sum += b[ch];
                    }
                    let out_idx = ((batch * h_out + oh) * w_out + ow) * c + ch;
                    output[out_idx] = sum;
                }
            }
        }
    }
}
```

### Multi-Threaded Version

```rust
pub fn depthwise_conv2d_f32_nhwc_mt(
    output: &mut [f32],
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    // ... same params ...
    num_threads: usize,
) {
    // Partition across output height dimension
    parallel_for_scoped(
        0..h_out,
        num_threads,
        8192,  // Work threshold: 8K MACs per task
        |oh_range| {
            for oh in oh_range {
                // ... same inner loop ...
            }
        }
    );
}
```

The height dimension is chosen for parallelism because:
- Each output row is independent
- Memory access is contiguous within a row (NHWC layout)
- Load balancing is good (h_out typically 14-112 for ImageNet models)

## Weight Layout

After NCHW→NHWC conversion:

| Original (ONNX) | Converted (dragonwing) |
|-----------------|------------------------|
| `[C_out, 1, kH, kW]` | `[kH, kW, 1, C_out]` |

For depthwise where `C_out == C_in`, the converted layout is `[kH, kW, 1, C]`.

The kernel indexing in the inner loop:
```rust
let k_idx = (kh * k_w + kw) * c + ch;
```

This assumes the weight is stored as `[kH, kW, C]` (the middle dimension of 1 is squeezed).

## MobileNetV2 Depthwise Stats

MobileNetV2 contains 17 depthwise conv layers:

| Layer | Input Shape | Kernel | Stride | Output Shape |
|-------|-------------|--------|--------|--------------|
| conv2d_1 | 112×112×32 | 3×3 | 1 | 112×112×32 |
| conv2d_2 | 56×56×96 | 3×3 | 2 | 28×28×96 |
| conv2d_3 | 28×28×144 | 3×3 | 1 | 28×28×144 |
| ... | | | | |
| conv2d_17 | 7×7×960 | 3×3 | 1 | 7×7×960 |

Total depthwise conv MACs: ~15% of model
Total depthwise memory bandwidth: ~25% of model (memory-bound)

## Performance Characteristics

Depthwise conv is **memory-bound**, not compute-bound:
- Only 1 MAC per weight element (vs K for standard conv)
- Arithmetic intensity: ~0.1-0.5 FLOP/byte
- Performance limited by memory bandwidth, not ALU throughput

### Optimization Strategies

1. **Channel vectorization** (current): Process 4 channels per NEON instruction
2. **Spatial tiling**: Keep kernel and tile of input in L1 cache
3. **Fused bias+relu**: Avoid extra memory pass for activation

## Vulkan Implementation

### Workgroup Geometry

```glsl
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
```

Each thread computes one output pixel `(oh, ow)` for all channels:

```glsl
void main() {
    uint oh = gl_GlobalInvocationID.y;
    uint ow = gl_GlobalInvocationID.x;
    
    for (uint c = 0; c < C; c++) {
        float sum = 0.0;
        for (uint kh = 0; kh < K_H; kh++) {
            for (uint kw = 0; kw < K_W; kw++) {
                // ... accumulate ...
            }
        }
        output[index(oh, ow, c)] = sum + bias[c];
    }
}
```

### Why Not One Thread Per (oh, ow, c)?

For depthwise conv, the kernel loop (`kh × kw`) is short (typically 9 iterations for 3×3). The thread setup overhead would dominate.

Instead, each thread handles all channels for a spatial position, amortizing launch overhead.

## Common Bugs

### 1. Wrong Channel Indexing

```rust
// WRONG: treats depthwise like standard conv
let k_idx = ((c_out * c_in + c_in) * k_h + kh) * k_w + kw;

// CORRECT: depthwise has c_in == 1 per group
let k_idx = (kh * k_w + kw) * c + ch;
```

### 2. Weight Layout Mismatch

After NCHW→NHWC transpose with permutation `[2, 3, 1, 0]`:
- Original: `[C_out, 1, kH, kW]`
- Result: `[kH, kW, 1, C_out]`

If you use the wrong permutation, the kernel will silently produce wrong output.

### 3. Stride vs Dilation

MobileNetV2 uses strided depthwise conv (`stride=2`) for downsampling, NOT dilated conv. The implementations are different:

| Strided | Dilated |
|---------|---------|
| `ih = oh * stride + kh` | `ih = oh + kh * dilation` |
| Output smaller than input | Output same size as input |

## Test Coverage

The depthwise conv implementation is tested in:

1. **Unit test**: 4×4×4 input with explicit per-channel weights
   - `crates/dragonwing-cpu/src/ops.rs` (test module)

2. **Integration test**: MobileNetV2 end-to-end
   - `crates/dragonwing-onnx/src/graph.rs::test_compile_mobilenetv2_nhwc`

3. **Parity test**: CPU single-thread vs multi-thread
   - Verifies MT implementation matches ST exactly

## File References

- `crates/dragonwing-cpu/src/ops.rs:1380-1500` — CPU depthwise conv implementation
- `crates/dragonwing-onnx/src/builder.rs:390-530` — ConvBuilder with depthwise detection
- `crates/dragonwing-onnx/src/runtime.rs:1250-1380` — CpuGraphRuntime conv dispatch
- `crates/dragonwing-shaders/glsl/depthwise_conv2d_f32_nhwc.comp` — Vulkan shader (task 005)
