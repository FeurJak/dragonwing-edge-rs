# INT8 Vulkan Implementation

This document describes the INT8 quantized inference implementation for Vulkan, specifically designed for devices like the Adreno A702 that lack native 8-bit storage extensions.

## Hardware Constraints

### Adreno A702 (Arduino UNO Q)

| Feature | Status | Implication |
|---------|--------|-------------|
| `VK_KHR_8bit_storage` | **Absent** | Cannot use `int8_t` in shaders |
| `VK_KHR_16bit_storage` | Present | FP16 works natively |
| Shader INT32 | Present | Use for accumulation |
| UINT32 buffers | Present | Pack 4×INT8 into UINT32 |

Without `VK_KHR_8bit_storage`, we cannot declare INT8 storage buffers. The solution is to **pack 4 INT8 values into each UINT32** and unpack in the shader.

## UINT32 Packing Format

### Packing Convention (Little-Endian)

```
Memory layout:  [byte0][byte1][byte2][byte3][byte4][byte5]...
UINT32 packing: [    packed0 (bytes 0-3)   ][    packed1    ]

packed_u32 = (i8[0] & 0xFF) | 
             ((i8[1] & 0xFF) << 8) | 
             ((i8[2] & 0xFF) << 16) | 
             ((i8[3] & 0xFF) << 24)
```

### Rust Packing Code

```rust
/// Pack 4 INT8 values into a UINT32 (little-endian)
fn pack_i8x4(values: [i8; 4]) -> u32 {
    (values[0] as u8 as u32)
        | ((values[1] as u8 as u32) << 8)
        | ((values[2] as u8 as u32) << 16)
        | ((values[3] as u8 as u32) << 24)
}

/// Pack entire INT8 tensor into UINT32 buffer
fn pack_tensor_i8_to_u32(input: &[i8]) -> Vec<u32> {
    assert!(input.len() % 4 == 0, "Length must be multiple of 4");
    input
        .chunks_exact(4)
        .map(|chunk| pack_i8x4([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
```

### GLSL Unpacking Code

```glsl
/// Unpack 4 INT8 values from a UINT32 (with sign extension)
ivec4 unpack_i8x4(uint packed) {
    // Extract unsigned bytes
    int i0 = int(packed & 0xFFu);
    int i1 = int((packed >> 8u) & 0xFFu);
    int i2 = int((packed >> 16u) & 0xFFu);
    int i3 = int((packed >> 24u) & 0xFFu);
    
    // Sign extension: if value >= 128, it's negative
    if (i0 >= 128) i0 -= 256;
    if (i1 >= 128) i1 -= 256;
    if (i2 >= 128) i2 -= 256;
    if (i3 >= 128) i3 -= 256;
    
    return ivec4(i0, i1, i2, i3);
}
```

## INT8 Shader Inventory

| Shader | Input | Output | Purpose |
|--------|-------|--------|---------|
| `gemm_i8_packed.comp` | Packed UINT32 | INT32 | Matrix multiplication |
| `conv2d_i8_nhwc_packed.comp` | Packed UINT32 | INT32 | 2D convolution |
| `requantize_i32_to_i8_packed.comp` | INT32 | Packed UINT32 | Layer transition |
| `quantize_f32_to_i8_packed.comp` | F32 | Packed UINT32 | Input quantization |
| `dequantize_i8_packed_to_f32.comp` | Packed UINT32 | F32 | Output dequantization |
| `add_i8_packed.comp` | Packed UINT32 | Packed UINT32 | Element-wise add |
| `relu_i8_packed.comp` | Packed UINT32 | Packed UINT32 | ReLU activation |

## GEMM Shader Design

### Buffer Layout

```glsl
layout(set = 0, binding = 0, std430) writeonly buffer CBuf {
    int c[];  // Output: INT32 accumulator [m × n]
};

layout(set = 0, binding = 1, std430) readonly buffer ABuf {
    uint a_packed[];  // Input A: packed INT8 [m × k/4]
};

layout(set = 0, binding = 2, std430) readonly buffer BBuf {
    uint b_packed[];  // Input B: packed INT8 [k × n/4]
};
```

### Push Constants

```glsl
layout(push_constant) uniform Pc {
    uint m;    // Rows of A, rows of C
    uint n;    // Cols of B, cols of C
    uint k;    // Cols of A, rows of B (must be multiple of 4)
    uint _pad; // Padding for alignment
} pc;
```

### Computation Pattern

Each invocation computes one output element C[row, col]:

```glsl
void main() {
    uint col = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.y;
    
    if (row >= pc.m || col >= pc.n) return;
    
    int acc = 0;
    uint k_packed = pc.k / 4u;
    
    for (uint p = 0u; p < k_packed; p++) {
        // Load and unpack 4 elements from A row
        ivec4 a_vals = unpack_i8x4(a_packed[row * k_packed + p]);
        
        // Load and unpack 4 elements from B column
        // (requires 4 separate row accesses for column-major access)
        // ... (see full shader for details)
        
        // 4-element dot product
        acc += a_vals.x * b0 + a_vals.y * b1 + 
               a_vals.z * b2 + a_vals.w * b3;
    }
    
    c[row * pc.n + col] = acc;
}
```

### Workgroup Configuration

```glsl
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
```

Dispatch dimensions:
- `gx = ceil(n / 8)`
- `gy = ceil(m / 8)`
- `gz = 1`

## Convolution Shader Design

The INT8 convolution follows similar patterns to GEMM but operates on NHWC tensors.

### Buffer Layout

```glsl
layout(set = 0, binding = 0, std430) writeonly buffer OutBuf {
    int output[];  // [N × H_out × W_out × C_out] INT32
};

layout(set = 0, binding = 1, std430) readonly buffer InBuf {
    uint input_packed[];  // [N × H × W × C_in/4] packed
};

layout(set = 0, binding = 2, std430) readonly buffer WBuf {
    uint weights_packed[]; // [K_h × K_w × C_in × C_out/4] packed
};
```

### Computation

```glsl
// For each output position
for (uint kh = 0; kh < K_H; kh++) {
    for (uint kw = 0; kw < K_W; kw++) {
        for (uint ic_packed = 0; ic_packed < C_IN_PACKED; ic_packed++) {
            // Load 4 input channels
            ivec4 in_vals = unpack_i8x4(input_packed[...]);
            
            // Load 4 weights for current output channel
            ivec4 w_vals = unpack_i8x4(weights_packed[...]);
            
            // Accumulate
            acc += dot(in_vals, w_vals);
        }
    }
}
```

## Requantization Shader

After INT8 computation produces INT32 accumulators, requantization converts back to INT8 for the next layer.

### Scale Application

```glsl
layout(push_constant) uniform Pc {
    uint count;
    float scale;  // requant_scale = scale_in * scale_weight / scale_out
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x * 4;
    if (idx >= pc.count) return;
    
    // Load 4 INT32 values
    int i0 = input[idx + 0];
    int i1 = input[idx + 1];
    int i2 = input[idx + 2];
    int i3 = input[idx + 3];
    
    // Scale, round, and clamp to INT8 range
    int o0 = clamp(int(round(float(i0) * pc.scale)), -128, 127);
    int o1 = clamp(int(round(float(i1) * pc.scale)), -128, 127);
    int o2 = clamp(int(round(float(i2) * pc.scale)), -128, 127);
    int o3 = clamp(int(round(float(i3) * pc.scale)), -128, 127);
    
    // Pack into UINT32
    output_packed[idx / 4] = pack_i8x4(o0, o1, o2, o3);
}
```

## Alignment Requirements

### Tensor Dimensions

| Dimension | Requirement | Reason |
|-----------|-------------|--------|
| Channels (C) | Multiple of 4 | UINT32 packing |
| K (GEMM inner) | Multiple of 4 | UINT32 packing |
| Other dims | No restriction | - |

### Buffer Alignment

```rust
// Ensure tensor size is multiple of 4 for packing
fn align_to_4(size: usize) -> usize {
    (size + 3) & !3
}

// Pad tensor if needed
fn pad_tensor_for_packing(tensor: &[i8]) -> Vec<i8> {
    let aligned_len = align_to_4(tensor.len());
    let mut padded = tensor.to_vec();
    padded.resize(aligned_len, 0);
    padded
}
```

## Performance Considerations

### Memory Bandwidth

| Format | Bytes per Element | Bandwidth Reduction |
|--------|-------------------|---------------------|
| F32 | 4 | Baseline |
| INT8 (packed) | 1 (effective) | 4× |

### Compute Efficiency

Unpacking adds overhead, but:
- 4 INT8 values fit in one 32-bit register
- INT32 MAD is typically as fast as F32 MAD
- Reduced memory traffic often dominates

### Optimal Patterns

1. **Coalesced access**: Access packed UINT32s sequentially
2. **Shared memory**: Cache unpacked values for reuse
3. **4-element vectorization**: Process 4 values per operation

## Debugging Tips

### Verify Packing

```rust
// Test round-trip
let original: [i8; 4] = [-128, -1, 0, 127];
let packed = pack_i8x4(original);
let unpacked = unpack_i8x4(packed);
assert_eq!(original, unpacked);
```

### Check Sign Extension

Common bug: forgetting sign extension when unpacking.

```glsl
// WRONG: treats all values as unsigned
int i0 = int(packed & 0xFFu);  // -1 becomes 255!

// CORRECT: sign extend
int i0 = int(packed & 0xFFu);
if (i0 >= 128) i0 -= 256;  // -1 stays -1
```

### Validate Alignment

```rust
assert!(input.len() % 4 == 0, "Tensor length must be multiple of 4");
assert!(channels % 4 == 0, "Channels must be multiple of 4");
```

## Code References

- `crates/dragonwing-shaders/glsl/gemm_i8_packed.comp` - GEMM shader
- `crates/dragonwing-shaders/glsl/conv2d_i8_nhwc_packed.comp` - Conv2D shader
- `crates/dragonwing-shaders/glsl/requantize_i32_to_i8_packed.comp` - Requantization
- `crates/dragonwing-shaders/src/lib.rs` - Shader constants and registry
- `crates/dragonwing-vulkan/src/pipeline.rs` - Pipeline creation for INT8 ops

## Testing

### CPU-Vulkan Parity

```bash
cargo test -p dragonwing-test parity
```

Tests verify:
- Packed GEMM matches CPU INT8 GEMM
- Packed Conv2D matches CPU INT8 Conv2D
- Requantization produces same output

### Tolerance

INT8 Vulkan ops should match CPU within 1 quantization level:
- Max absolute difference: 1 (due to rounding)
- Relative error: <1% for typical workloads

## Future Improvements

- [ ] Tiled GEMM with shared memory caching
- [ ] Fused conv + requantize + ReLU kernel
- [ ] Asymmetric quantization support (non-zero zero_point)
- [ ] UINT8 support for unsigned activations
- [ ] Batch GEMM for multiple outputs
