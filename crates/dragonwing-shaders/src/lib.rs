//! `dragonwing-shaders` — compiled SPIR-V compute shaders for the Vulkan
//! backend.
//!
//! Each public `&'static [u8]` constant is the **raw SPIR-V binary** for
//! one compute shader. The binary is suitable for direct
//! `vkCreateShaderModule` consumption (SPIR-V is a sequence of 32-bit
//! words; the slice length is always a multiple of 4).
//!
//! # Source of truth
//!
//! The `.comp` GLSL sources live in `glsl/`. The build script
//! (`build.rs`) compiles them to SPIR-V using `glslangValidator` if that
//! tool is on `PATH`; otherwise it copies pre-compiled blobs from `spv/`.
//! See `build.rs` for the full policy.
//!
//! # How to add a new shader
//!
//! 1. Drop a new `glsl/<name>.comp` file. Follow the conventions in the
//!    existing files (push-constant layout commented at the top, single
//!    `main`).
//! 2. Add a `pub const <UPPER_NAME>: &[u8] = include_bytes!(...)` line
//!    here.
//! 3. `cargo build -p dragonwing-shaders`.
//! 4. Commit the freshly-produced `spv/<name>.spv` so contributors
//!    without `glslangValidator` can still build.
//!
//! # Cross-host determinism
//!
//! `glslangValidator` is deterministic given the same input and the same
//! version of the compiler. Different versions can produce different
//! (but functionally equivalent) SPIR-V. Pinning the compiler version is
//! recommended for reproducible builds; this is documented in
//! `docs/vulkan-backend.md`.

#![no_std]
#![warn(missing_docs)]

// ===========================================================================
// F32 shaders
// ===========================================================================

/// `fill_f32` — write a scalar to every element of an f32 buffer.
///
/// Push-constant layout: `{ uint n; float v; float _; float _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const FILL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fill_f32.spv"));

/// `axpy_f32` — `y[i] = a * x[i] + y[i]` using `fma()`.
///
/// Push-constant layout: `{ uint n; float a; float _; float _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const AXPY_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/axpy_f32.spv"));

/// `relu_f32` — `y[i] = max(0, x[i])`.
///
/// Push-constant layout: `{ uint n; uint _; uint _; uint _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const RELU_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/relu_f32.spv"));

/// `add_f32` — `y[i] = a[i] + b[i]`.
///
/// Push-constant layout: `{ uint n; uint _; uint _; uint _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const ADD_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_f32.spv"));

/// `gemm_f32_naive` — `C = A * B`, row-major, naive triple-loop with
/// fma() inner.
///
/// Push-constant layout: `{ uint m; uint n; uint k; uint _; }`.
/// Dispatch: `gx = ceil(n / 16), gy = ceil(m / 16), gz = 1`.
pub const GEMM_F32_NAIVE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_f32_naive.spv"));

/// `gemm_f32_tiled` — `C = A * B`, row-major, tiled with shared memory.
/// Uses 16x16 tiles with K-blocking for improved memory locality.
///
/// Push-constant layout: `{ uint m; uint n; uint k; uint _; }`.
/// Dispatch: `gx = ceil(n / 16), gy = ceil(m / 16), gz = 1`.
pub const GEMM_F32_TILED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_f32_tiled.spv"));

/// `gemm_fp16` — `C = A * B`, row-major, F16 inputs/outputs with F32
/// accumulator for precision.
///
/// Push-constant layout: `{ uint m; uint n; uint k; uint _; }`.
/// Dispatch: `gx = ceil(n / 16), gy = ceil(m / 16), gz = 1`.
pub const GEMM_FP16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_fp16.spv"));

// ===========================================================================
// Convolution and pooling shaders
// ===========================================================================

/// `conv2d_f32_nhwc` — Direct 2D convolution in NHWC format.
///
/// Push-constant layout: 64 bytes (4 uvec4).
/// Dispatch: `gx = ceil(w_out/8), gy = ceil(h_out/8), gz = n * c_out`.
pub const CONV2D_F32_NHWC: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/conv2d_f32_nhwc.spv"));

/// `conv2d_fp16_nhwc` — Direct 2D convolution in NHWC format with FP16 I/O.
///
/// Uses FP16 for input/output buffers but accumulates in FP32 for precision.
/// Push-constant layout: 64 bytes (4 uvec4).
/// Dispatch: `gx = ceil(w_out/8), gy = ceil(h_out/8), gz = n * c_out`.
pub const CONV2D_FP16_NHWC: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/conv2d_fp16_nhwc.spv"));

/// `maxpool2d_f32` — 2D max pooling in NHWC format.
///
/// Push-constant layout: 64 bytes (4 uvec4).
/// Dispatch: `gx = ceil(w_out/8), gy = ceil(h_out/8), gz = n * c`.
pub const MAXPOOL2D_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/maxpool2d_f32.spv"));

/// `softmax_f32` — Softmax along last axis with workgroup reduction.
///
/// Push-constant layout: `{ uint n; uint rows; uint _; uint _; }`.
/// Dispatch: `gx = 1, gy = rows, gz = 1`.
pub const SOFTMAX_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softmax_f32.spv"));

// ===========================================================================
// YOLO ops (Task 005)
// ===========================================================================

/// `sigmoid_f32` — element-wise sigmoid: y[i] = 1/(1+exp(-x[i])).
///
/// Push-constant layout: `{ uint n; uint _; uint _; uint _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const SIGMOID_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sigmoid_f32.spv"));

/// `mul_f32` — element-wise multiplication: y[i] = a[i] * b[i].
///
/// Push-constant layout: `{ uint n; uint _; uint _; uint _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const MUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_f32.spv"));

// ===========================================================================
// FP16 shaders (require VK_KHR_16bit_storage + shaderFloat16)
// ===========================================================================

/// `fill_fp16` — write a scalar to every element of an f16 buffer.
///
/// Push-constant layout: `{ uint n; float v; float _; float _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const FILL_FP16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fill_fp16.spv"));

/// `axpy_fp16` — `y[i] = a * x[i] + y[i]` with F16 buffers.
/// Computes in F32 internally for precision.
///
/// Push-constant layout: `{ uint n; float a; float _; float _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const AXPY_FP16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/axpy_fp16.spv"));

/// `relu_fp16` — `y[i] = max(0, x[i])` with F16 buffers.
///
/// Push-constant layout: `{ uint n; uint _; uint _; uint _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const RELU_FP16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/relu_fp16.spv"));

/// `add_fp16` — `y[i] = a[i] + b[i]` with F16 buffers.
///
/// Push-constant layout: `{ uint n; uint _; uint _; uint _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const ADD_FP16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_fp16.spv"));

// ===========================================================================
// INT8 quantized shaders (Task 006)
//
// These shaders use UINT32 packing to work around lack of VK_KHR_8bit_storage
// on Adreno A702. Four INT8 values are packed into each UINT32.
// ===========================================================================

/// `gemm_i8_packed` — INT8 matrix multiplication with UINT32 packing.
///
/// Reads packed INT8 data (4 values per UINT32), accumulates in INT32.
/// Push-constant layout: `{ uint m; uint n; uint k; uint _; }` (k must be multiple of 4).
/// Dispatch: `gx = ceil(n / 8), gy = ceil(m / 8), gz = 1`.
pub const GEMM_I8_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_i8_packed.spv"));

/// `conv2d_i8_nhwc_packed` — INT8 2D convolution with UINT32 packing.
///
/// Input/kernel packed along channel dimension (C must be multiple of 4).
/// Output is INT32 accumulator; caller handles requantization.
/// Push-constant layout: 64 bytes (4 uvec4).
/// Dispatch: `gx = ceil(w_out/8), gy = ceil(h_out/8), gz = n * c_out`.
pub const CONV2D_I8_NHWC_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/conv2d_i8_nhwc_packed.spv"));

/// `requantize_i32_to_i8_packed` — Requantize INT32 accumulators to packed INT8.
///
/// Applies scale and packs 4 INT8 outputs into each UINT32.
/// Push-constant layout: `{ uint n; float requant_scale; uint _; uint _; }`.
/// Dispatch: `gx = ceil(n / (64 * 4)), gy = gz = 1`.
pub const REQUANTIZE_I32_TO_I8_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/requantize_i32_to_i8_packed.spv"));

/// `dequantize_i8_packed_to_f32` — Convert packed INT8 to F32.
///
/// Reads packed INT8 (4 per UINT32) and outputs F32.
/// Push-constant layout: `{ uint n; float scale; uint _; uint _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const DEQUANTIZE_I8_PACKED_TO_F32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/dequantize_i8_packed_to_f32.spv"));

/// `quantize_f32_to_i8_packed` — Convert F32 to packed INT8.
///
/// Reads F32 and outputs packed INT8 (4 per UINT32).
/// Push-constant layout: `{ uint n; float inv_scale; uint _; uint _; }`.
/// Dispatch: `gx = ceil(n / (64 * 4)), gy = gz = 1`.
pub const QUANTIZE_F32_TO_I8_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/quantize_f32_to_i8_packed.spv"));

/// `add_i8_packed` — INT8 element-wise addition with requantization.
///
/// Adds two packed INT8 tensors, requantizes to output scale.
/// Push-constant layout: `{ uint n; float scale_a_over_y; float scale_b_over_y; uint _; }`.
/// Dispatch: `gx = ceil(n / (64 * 4)), gy = gz = 1`.
pub const ADD_I8_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/add_i8_packed.spv"));

/// `relu_i8_packed` — INT8 ReLU with packed storage.
///
/// max(0, x) for packed INT8 data.
/// Push-constant layout: `{ uint n; uint _; uint _; uint _; }`.
/// Dispatch: `gx = ceil(n / (64 * 4)), gy = gz = 1`.
pub const RELU_I8_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/relu_i8_packed.spv"));

// ---------------------------------------------------------------------------
// Fused kernels (Task 007)
// ---------------------------------------------------------------------------

/// `silu_f32` — Fused SiLU activation: `y[i] = x[i] * sigmoid(x[i])`.
///
/// Saves one full-tensor pass vs the separate `sigmoid_f32 + mul_f32`
/// chain. YOLOv8 uses SiLU extensively, so this fusion is the highest-ROI
/// F32 op fusion in the codebase.
///
/// Push-constant layout: `{ uint n; uint _; uint _; uint _; }`.
/// Dispatch: `gx = ceil(n / 64), gy = gz = 1`.
pub const SILU_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu_f32.spv"));

/// `conv2d_requant_relu_i8_packed` — Fused INT8 Conv2D + Requantize + ReLU.
///
/// Dominant compute pattern in YOLO INT8 inference. Each invocation
/// produces 4 INT8 output channels (packed UINT32). Requires `c_out` to
/// be a multiple of 4.
///
/// Push-constant layout (80 bytes):
/// `{ uvec4 dims0; uvec4 dims1; uvec4 dims2; uvec4 dims3; float requant_scale; uint do_relu; }`.
/// Dispatch: `gx = ceil(w_out / 8), gy = ceil(h_out / 8), gz = n * c_out_packed`.
pub const CONV2D_REQUANT_RELU_I8_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/conv2d_requant_relu_i8_packed.spv"));

/// Convenience: every shader byte-slice in the crate, indexed by a
/// stable string id. Useful for a `--list-shaders` developer tool.
pub const ALL: &[(&str, &[u8])] = &[
    ("fill_f32", FILL_F32),
    ("axpy_f32", AXPY_F32),
    ("relu_f32", RELU_F32),
    ("add_f32", ADD_F32),
    ("gemm_f32_naive", GEMM_F32_NAIVE),
    ("gemm_f32_tiled", GEMM_F32_TILED),
    ("conv2d_f32_nhwc", CONV2D_F32_NHWC),
    ("conv2d_fp16_nhwc", CONV2D_FP16_NHWC),
    ("maxpool2d_f32", MAXPOOL2D_F32),
    ("softmax_f32", SOFTMAX_F32),
    ("sigmoid_f32", SIGMOID_F32),
    ("mul_f32", MUL_F32),
    ("fill_fp16", FILL_FP16),
    ("axpy_fp16", AXPY_FP16),
    ("relu_fp16", RELU_FP16),
    ("add_fp16", ADD_FP16),
    ("gemm_fp16", GEMM_FP16),
    // INT8 packed shaders (Task 006)
    ("gemm_i8_packed", GEMM_I8_PACKED),
    ("conv2d_i8_nhwc_packed", CONV2D_I8_NHWC_PACKED),
    ("requantize_i32_to_i8_packed", REQUANTIZE_I32_TO_I8_PACKED),
    ("dequantize_i8_packed_to_f32", DEQUANTIZE_I8_PACKED_TO_F32),
    ("quantize_f32_to_i8_packed", QUANTIZE_F32_TO_I8_PACKED),
    ("add_i8_packed", ADD_I8_PACKED),
    ("relu_i8_packed", RELU_I8_PACKED),
    // Fused kernels (Task 007)
    ("silu_f32", SILU_F32),
    ("conv2d_requant_relu_i8_packed", CONV2D_REQUANT_RELU_I8_PACKED),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shader_is_nonempty_and_word_aligned() {
        // SPIR-V is a stream of 32-bit words; a valid module is at least
        // 5 words (magic, version, generator, bound, schema).
        for (name, blob) in ALL {
            assert!(blob.len() >= 20, "shader {name}: only {} bytes", blob.len());
            assert_eq!(blob.len() % 4, 0, "shader {name}: length not multiple of 4");
            // SPIR-V magic number, little-endian: 0x07230203.
            let magic = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
            assert_eq!(
                magic, 0x0723_0203,
                "shader {name}: bad SPIR-V magic 0x{magic:08x}"
            );
        }
    }
}
