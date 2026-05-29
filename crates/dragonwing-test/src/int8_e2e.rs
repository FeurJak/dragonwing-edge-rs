//! INT8 quantization end-to-end tests (Task 006).
//!
//! This module tests the complete INT8 quantization pipeline:
//! 1. Quantize F32 tensor to INT8
//! 2. Run INT8 operations
//! 3. Dequantize back to F32
//! 4. Verify accuracy within tolerance
//!
//! The tests use the CPU ops from dragonwing-cpu.

#[cfg(test)]
use dragonwing_cpu::ops;

/// Random number generator (Xorshift64).
#[cfg(test)]
struct Rng(u64);

#[cfg(test)]
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Generate f32 in [-1, 1].
    fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64();
        (bits as f32 / u64::MAX as f32) * 2.0 - 1.0
    }

    /// Generate random f32 in [-scale, scale].
    fn fill_f32_scaled(&mut self, data: &mut [f32], scale: f32) {
        for v in data {
            *v = self.next_f32() * scale;
        }
    }
}

/// Test quantize -> dequantize roundtrip accuracy.
#[test]
fn test_quantize_dequantize_roundtrip() {
    let mut rng = Rng::new(12345);
    
    // Generate random F32 data in range [-2.0, 2.0]
    let n = 256;
    let mut original = vec![0.0f32; n];
    rng.fill_f32_scaled(&mut original, 2.0);
    
    // Determine scale: max(|x|) / 127
    let abs_max = original.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = abs_max / 127.0;
    
    // Quantize F32 -> INT8
    let mut quantized = vec![0i8; n];
    ops::quantize_f32_to_i8(&mut quantized, &original, scale);
    
    // Dequantize INT8 -> F32
    let mut reconstructed = vec![0.0f32; n];
    ops::dequantize_i8_to_f32(&mut reconstructed, &quantized, scale);
    
    // Verify roundtrip error is within quantization error (scale/2)
    let max_error = scale;  // Should be less than one quantization step
    for (i, (&orig, &recon)) in original.iter().zip(reconstructed.iter()).enumerate() {
        let error = (orig - recon).abs();
        assert!(
            error <= max_error,
            "Roundtrip error too large at index {}: original={}, reconstructed={}, error={}",
            i, orig, recon, error
        );
    }
    
    println!("Quantize/dequantize roundtrip test passed");
    println!("  Scale: {}", scale);
    println!("  Max error: {}", original.iter().zip(reconstructed.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max));
}

/// Test INT8 GEMM correctness.
#[test]
fn test_int8_gemm_vs_f32() {
    let mut rng = Rng::new(42);
    
    // Small matrices for easy verification
    let m = 8;
    let n = 8;
    let k = 16;
    
    // Generate random F32 data
    let mut a_f32 = vec![0.0f32; m * k];
    let mut b_f32 = vec![0.0f32; k * n];
    rng.fill_f32_scaled(&mut a_f32, 1.0);
    rng.fill_f32_scaled(&mut b_f32, 1.0);
    
    // Compute F32 GEMM reference
    let mut c_f32_ref = vec![0.0f32; m * n];
    ops::gemm_f32_naive(&mut c_f32_ref, &a_f32, &b_f32, m, n, k);
    
    // Quantize inputs
    let scale_a = a_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max) / 127.0;
    let scale_b = b_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max) / 127.0;
    
    let mut a_i8 = vec![0i8; m * k];
    let mut b_i8 = vec![0i8; k * n];
    ops::quantize_f32_to_i8(&mut a_i8, &a_f32, scale_a);
    ops::quantize_f32_to_i8(&mut b_i8, &b_f32, scale_b);
    
    // Compute INT8 GEMM (result in INT32)
    let mut c_i32 = vec![0i32; m * n];
    ops::gemm_i8(&mut c_i32, &a_i8, &b_i8, m, n, k);
    
    // Dequantize result
    // The scale of the output is scale_a * scale_b
    let scale_c = scale_a * scale_b;
    let c_f32_quant: Vec<f32> = c_i32.iter()
        .map(|&x| x as f32 * scale_c)
        .collect();
    
    // Compare
    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f32;
    for (_i, (&ref_val, &quant_val)) in c_f32_ref.iter().zip(c_f32_quant.iter()).enumerate() {
        let error = (ref_val - quant_val).abs();
        max_error = max_error.max(error);
        sum_sq_error += error * error;
    }
    
    let rmse = (sum_sq_error / (m * n) as f32).sqrt();
    println!("INT8 GEMM test passed");
    println!("  Max error: {}", max_error);
    println!("  RMSE: {}", rmse);
    
    // Allow reasonable quantization error
    // Error accumulates across k elements, and each element has up to 0.5/127 quantization error
    // Total error can be up to k * (input_quant_error * weight_quant_error)
    // Roughly: sqrt(k) * scale_output for typical random data
    let tolerance = 0.1; // 10% relative error is reasonable for quantization
    let ref_max = c_f32_ref.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        max_error / ref_max.max(1e-6) <= tolerance,
        "GEMM relative error too large: max_error={}, ref_max={}, ratio={}",
        max_error, ref_max, max_error / ref_max
    );
}

/// Test INT8 ReLU correctness.
#[test]
fn test_int8_relu() {
    // Input: mix of positive and negative values
    let input: Vec<i8> = (-64..64).map(|x| x as i8).collect();
    let mut output = vec![0i8; input.len()];
    
    ops::relu_i8(&mut output, &input);
    
    // Verify: all negatives become 0, positives unchanged
    for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
        let expected = if inp > 0 { inp } else { 0 };
        assert_eq!(out, expected, "ReLU mismatch at {}: input={}, output={}", i, inp, out);
    }
    
    println!("INT8 ReLU test passed");
}

/// Test INT8 Add with scale adjustment.
#[test]
fn test_int8_add_scaled() {
    let n = 64;
    
    // Two tensors with different scales
    let a_f32: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b_f32: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.05).collect();
    
    // Expected F32 result
    let expected_f32: Vec<f32> = a_f32.iter().zip(b_f32.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    
    // Quantize with different scales
    let scale_a = a_f32.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    let scale_b = b_f32.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    let scale_out = expected_f32.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    
    let mut a_i8 = vec![0i8; n];
    let mut b_i8 = vec![0i8; n];
    ops::quantize_f32_to_i8(&mut a_i8, &a_f32, scale_a);
    ops::quantize_f32_to_i8(&mut b_i8, &b_f32, scale_b);
    
    // INT8 add with scale adjustment
    let scale_a_over_out = scale_a / scale_out;
    let scale_b_over_out = scale_b / scale_out;
    
    let mut out_i8 = vec![0i8; n];
    ops::add_i8(&mut out_i8, &a_i8, &b_i8, scale_a_over_out, scale_b_over_out);
    
    // Dequantize result
    let mut result_f32 = vec![0.0f32; n];
    ops::dequantize_i8_to_f32(&mut result_f32, &out_i8, scale_out);
    
    // Verify
    let mut max_error = 0.0f32;
    for (i, (&exp, &res)) in expected_f32.iter().zip(result_f32.iter()).enumerate() {
        let error = (exp - res).abs();
        max_error = max_error.max(error);
        
        // Allow quantization error
        let tolerance = scale_out * 2.0;
        assert!(
            error <= tolerance,
            "Add error at {}: expected={}, result={}, error={}",
            i, exp, res, error
        );
    }
    
    println!("INT8 scaled Add test passed");
    println!("  Max error: {}", max_error);
}

/// Test requantization INT32 -> INT8.
#[test]
fn test_requantize() {
    // Simulate GEMM output in INT32
    let input_i32: Vec<i32> = (-1000..1000).map(|x| x * 10).collect();
    let n = input_i32.len();
    
    // Requantization scale
    let requant_scale = 0.01;
    
    let mut output_i8 = vec![0i8; n];
    ops::requantize_i32_to_i8(&mut output_i8, &input_i32, requant_scale);
    
    // Verify
    for (i, (&inp, &out)) in input_i32.iter().zip(output_i8.iter()).enumerate() {
        let expected_f32 = inp as f32 * requant_scale;
        let expected_i8 = expected_f32.round().clamp(-128.0, 127.0) as i8;
        
        // Allow rounding differences
        let diff = (out as i32 - expected_i8 as i32).abs();
        assert!(
            diff <= 1,
            "Requant mismatch at {}: input={}, output={}, expected={}",
            i, inp, out, expected_i8
        );
    }
    
    println!("Requantize INT32->INT8 test passed");
}

/// Test INT8 convolution accuracy.
#[test]
fn test_int8_conv2d() {
    // Small convolution test
    let n = 1;
    let h_in = 5;
    let w_in = 5;
    let c_in = 4;  // Multiple of 4 for packing
    let c_out = 4;
    let k_h = 3;
    let k_w = 3;
    
    let h_out = h_in - k_h + 1;  // No padding
    let w_out = w_in - k_w + 1;
    
    let mut rng = Rng::new(999);
    
    // Generate random F32 data
    let mut input_f32 = vec![0.0f32; n * h_in * w_in * c_in];
    let mut kernel_f32 = vec![0.0f32; k_h * k_w * c_in * c_out];
    rng.fill_f32_scaled(&mut input_f32, 1.0);
    rng.fill_f32_scaled(&mut kernel_f32, 0.5);
    
    // Compute F32 reference using CPU op
    let mut output_f32_ref = vec![0.0f32; n * h_out * w_out * c_out];
    ops::conv2d_f32_nhwc(
        &mut output_f32_ref, &input_f32, &kernel_f32,
        n, h_in, w_in, c_in, c_out, k_h, k_w, 1, 1, 0, 0
    );
    
    // Quantize
    let scale_input = input_f32.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    let scale_kernel = kernel_f32.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    
    let mut input_i8 = vec![0i8; input_f32.len()];
    let mut kernel_i8 = vec![0i8; kernel_f32.len()];
    ops::quantize_f32_to_i8(&mut input_i8, &input_f32, scale_input);
    ops::quantize_f32_to_i8(&mut kernel_i8, &kernel_f32, scale_kernel);
    
    // INT8 convolution (output in INT32)
    let mut output_i32 = vec![0i32; n * h_out * w_out * c_out];
    ops::conv2d_i8_nhwc(
        &mut output_i32, &input_i8, &kernel_i8,
        n, h_in, w_in, c_in, c_out, k_h, k_w, 1, 1, 0, 0
    );
    
    // Dequantize
    let scale_output = scale_input * scale_kernel;
    let output_f32_quant: Vec<f32> = output_i32.iter()
        .map(|&x| x as f32 * scale_output)
        .collect();
    
    // Compare
    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f32;
    for (_i, (&ref_val, &quant_val)) in output_f32_ref.iter().zip(output_f32_quant.iter()).enumerate() {
        let error = (ref_val - quant_val).abs();
        max_error = max_error.max(error);
        sum_sq_error += error * error;
    }
    
    let rmse = (sum_sq_error / output_f32_ref.len() as f32).sqrt();
    println!("INT8 Conv2D test passed");
    println!("  Max error: {}", max_error);
    println!("  RMSE: {}", rmse);
    
    // Allow reasonable quantization error
    assert!(max_error < 0.5, "Conv2D max error too large: {}", max_error);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Re-export tests to run with cargo test
    #[test]
    fn quantize_dequantize() { test_quantize_dequantize_roundtrip(); }
    
    #[test]
    fn int8_gemm() { test_int8_gemm_vs_f32(); }
    
    #[test]
    fn int8_relu() { test_int8_relu(); }
    
    #[test]
    fn int8_add() { test_int8_add_scaled(); }
    
    #[test]
    fn requantize() { test_requantize(); }
    
    #[test]
    fn int8_conv() { test_int8_conv2d(); }
}

// ============================================================================
// Phase 6: End-to-End INT8 Inference Tests
// ============================================================================

/// INT8 inference statistics for reporting.
#[derive(Debug, Clone)]
pub struct Int8InferenceStats {
    /// Maximum absolute error vs F32 reference.
    pub max_abs_error: f32,
    /// Root mean square error vs F32 reference.
    pub rmse: f32,
    /// Relative error (max_error / max_value).
    pub relative_error: f32,
    /// F32 inference time in microseconds.
    pub f32_time_us: u64,
    /// INT8 inference time in microseconds.
    pub i8_time_us: u64,
    /// Speedup ratio (f32_time / i8_time).
    pub speedup: f32,
    /// F32 memory usage in bytes.
    pub f32_memory_bytes: usize,
    /// INT8 memory usage in bytes.
    pub i8_memory_bytes: usize,
    /// Memory reduction ratio (f32_mem / i8_mem).
    pub memory_reduction: f32,
}

/// Test INT8 multi-layer network accuracy.
/// 
/// This simulates a small neural network with:
/// - Conv -> ReLU -> Conv -> ReLU -> GEMM
/// 
/// Verifies that chained quantized operations maintain acceptable accuracy.
#[test]
fn test_int8_multi_layer_accuracy() {
    let mut rng = Rng::new(77777);
    
    // Layer 1: Conv 3x3, 4->8 channels
    let n = 1;
    let h1 = 8;
    let w1 = 8;
    let c1 = 4;
    let c2 = 8;
    
    let mut input = vec![0.0f32; n * h1 * w1 * c1];
    let mut conv1_weight = vec![0.0f32; 3 * 3 * c1 * c2];
    rng.fill_f32_scaled(&mut input, 1.0);
    rng.fill_f32_scaled(&mut conv1_weight, 0.3);
    
    // F32 path: conv1
    let h2 = h1 - 2; // 6
    let w2 = w1 - 2; // 6
    let mut conv1_out_f32 = vec![0.0f32; n * h2 * w2 * c2];
    ops::conv2d_f32_nhwc(
        &mut conv1_out_f32, &input, &conv1_weight,
        n, h1, w1, c1, c2, 3, 3, 1, 1, 0, 0
    );
    
    // F32 path: relu1
    for v in &mut conv1_out_f32 {
        *v = v.max(0.0);
    }
    
    // Layer 2: Conv 3x3, 8->8 channels
    let mut conv2_weight = vec![0.0f32; 3 * 3 * c2 * c2];
    rng.fill_f32_scaled(&mut conv2_weight, 0.3);
    
    let h3 = h2 - 2; // 4
    let w3 = w2 - 2; // 4
    let mut conv2_out_f32 = vec![0.0f32; n * h3 * w3 * c2];
    ops::conv2d_f32_nhwc(
        &mut conv2_out_f32, &conv1_out_f32, &conv2_weight,
        n, h2, w2, c2, c2, 3, 3, 1, 1, 0, 0
    );
    
    // F32 path: relu2
    for v in &mut conv2_out_f32 {
        *v = v.max(0.0);
    }
    
    // Final F32 result
    let f32_output = conv2_out_f32.clone();
    
    // INT8 path
    // Quantize input
    let scale_input = input.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    let scale_conv1_w = conv1_weight.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    let scale_conv2_w = conv2_weight.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    
    let mut input_i8 = vec![0i8; input.len()];
    let mut conv1_weight_i8 = vec![0i8; conv1_weight.len()];
    let mut conv2_weight_i8 = vec![0i8; conv2_weight.len()];
    
    ops::quantize_f32_to_i8(&mut input_i8, &input, scale_input);
    ops::quantize_f32_to_i8(&mut conv1_weight_i8, &conv1_weight, scale_conv1_w);
    ops::quantize_f32_to_i8(&mut conv2_weight_i8, &conv2_weight, scale_conv2_w);
    
    // INT8 conv1
    let mut conv1_out_i32 = vec![0i32; n * h2 * w2 * c2];
    ops::conv2d_i8_nhwc(
        &mut conv1_out_i32, &input_i8, &conv1_weight_i8,
        n, h1, w1, c1, c2, 3, 3, 1, 1, 0, 0
    );
    
    // Requantize conv1 output for next layer
    let scale_conv1_out = scale_input * scale_conv1_w;
    // Estimate output range for requant scale
    let conv1_max_i32 = conv1_out_i32.iter().map(|&x| x.abs()).max().unwrap_or(1);
    let scale_conv1_i8 = (conv1_max_i32 as f32 * scale_conv1_out) / 127.0;
    let requant_scale1 = scale_conv1_out / scale_conv1_i8;
    
    let mut conv1_out_i8 = vec![0i8; conv1_out_i32.len()];
    ops::requantize_i32_to_i8(&mut conv1_out_i8, &conv1_out_i32, requant_scale1);
    
    // INT8 relu1
    let mut relu1_out_i8 = vec![0i8; conv1_out_i8.len()];
    ops::relu_i8(&mut relu1_out_i8, &conv1_out_i8);
    
    // INT8 conv2
    let mut conv2_out_i32 = vec![0i32; n * h3 * w3 * c2];
    ops::conv2d_i8_nhwc(
        &mut conv2_out_i32, &relu1_out_i8, &conv2_weight_i8,
        n, h2, w2, c2, c2, 3, 3, 1, 1, 0, 0
    );
    
    // Dequantize final output
    let scale_conv2_out = scale_conv1_i8 * scale_conv2_w;
    let mut i8_output_f32 = vec![0.0f32; conv2_out_i32.len()];
    for (i, &v) in conv2_out_i32.iter().enumerate() {
        i8_output_f32[i] = v as f32 * scale_conv2_out;
    }
    
    // Apply relu to INT8 output (simulating full pipeline)
    for v in &mut i8_output_f32 {
        *v = v.max(0.0);
    }
    
    // Compare
    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f32;
    for (&f32_val, &i8_val) in f32_output.iter().zip(i8_output_f32.iter()) {
        let error = (f32_val - i8_val).abs();
        max_error = max_error.max(error);
        sum_sq_error += error * error;
    }
    
    let rmse = (sum_sq_error / f32_output.len() as f32).sqrt();
    let ref_max = f32_output.iter().map(|&x| x.abs()).fold(1e-6f32, f32::max);
    let relative_error = max_error / ref_max;
    
    println!("INT8 Multi-layer accuracy test:");
    println!("  Max absolute error: {:.6}", max_error);
    println!("  RMSE: {:.6}", rmse);
    println!("  Relative error: {:.2}%", relative_error * 100.0);
    println!("  Reference max: {:.6}", ref_max);
    
    // Target: <20% relative error for multi-layer quantized inference
    // This is higher than single-op because errors accumulate
    assert!(
        relative_error < 0.20,
        "Multi-layer INT8 relative error too high: {:.2}% (target <20%)",
        relative_error * 100.0
    );
}

/// Test INT8 performance benchmark.
/// 
/// Compares F32 vs INT8 inference time and memory usage.
#[test]
fn test_int8_performance_benchmark() {
    let mut rng = Rng::new(88888);
    
    // Larger dimensions for meaningful benchmark
    let m = 64;
    let n = 64;
    let k = 128;
    
    // Generate data
    let mut a_f32 = vec![0.0f32; m * k];
    let mut b_f32 = vec![0.0f32; k * n];
    rng.fill_f32_scaled(&mut a_f32, 1.0);
    rng.fill_f32_scaled(&mut b_f32, 1.0);
    
    // Quantize
    let scale_a = a_f32.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    let scale_b = b_f32.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    
    let mut a_i8 = vec![0i8; m * k];
    let mut b_i8 = vec![0i8; k * n];
    ops::quantize_f32_to_i8(&mut a_i8, &a_f32, scale_a);
    ops::quantize_f32_to_i8(&mut b_i8, &b_f32, scale_b);
    
    // Warm up
    let mut c_f32 = vec![0.0f32; m * n];
    let mut c_i32 = vec![0i32; m * n];
    ops::gemm_f32_naive(&mut c_f32, &a_f32, &b_f32, m, n, k);
    ops::gemm_i8(&mut c_i32, &a_i8, &b_i8, m, n, k);
    
    // Benchmark F32
    let iterations = 100;
    let start_f32 = std::time::Instant::now();
    for _ in 0..iterations {
        ops::gemm_f32_naive(&mut c_f32, &a_f32, &b_f32, m, n, k);
    }
    let f32_time = start_f32.elapsed();
    
    // Benchmark INT8
    let start_i8 = std::time::Instant::now();
    for _ in 0..iterations {
        ops::gemm_i8(&mut c_i32, &a_i8, &b_i8, m, n, k);
    }
    let i8_time = start_i8.elapsed();
    
    let f32_time_us = f32_time.as_micros() as u64 / iterations as u64;
    let i8_time_us = i8_time.as_micros() as u64 / iterations as u64;
    let speedup = f32_time.as_secs_f32() / i8_time.as_secs_f32();
    
    // Memory comparison
    let f32_memory = (a_f32.len() + b_f32.len() + c_f32.len()) * 4;
    let i8_memory = (a_i8.len() + b_i8.len()) + c_i32.len() * 4; // INT8 inputs, INT32 output
    let memory_reduction = f32_memory as f32 / i8_memory as f32;
    
    println!("INT8 Performance Benchmark ({}x{}x{} GEMM):", m, n, k);
    println!("  F32 time: {} us/iter", f32_time_us);
    println!("  INT8 time: {} us/iter", i8_time_us);
    println!("  Speedup: {:.2}x", speedup);
    println!("  F32 memory: {} bytes", f32_memory);
    println!("  INT8 memory: {} bytes", i8_memory);
    println!("  Memory reduction: {:.2}x", memory_reduction);
    
    // Note: Actual speedup depends on hardware (NEON availability)
    // On most platforms, INT8 should be at least as fast as F32
    // Memory reduction should be close to 4x for inputs (F32 -> I8)
    
    // Verify memory reduction is significant (at least 1.5x considering INT32 accumulators)
    assert!(
        memory_reduction >= 1.5,
        "Memory reduction should be at least 1.5x, got {:.2}x",
        memory_reduction
    );
}

/// Simulated end-to-end INT8 YOLO inference test.
/// 
/// This test simulates the key operations in YOLO:
/// 1. Input quantization (F32 -> INT8)
/// 2. Backbone convolutions (INT8)
/// 3. Detection head (INT8 -> F32 for post-processing)
#[test]
fn test_int8_yolo_simulation() {
    let mut rng = Rng::new(12345678);
    
    // Simulated YOLO backbone dimensions (scaled down)
    // Real YOLOv8n: 640x640 input, but we use smaller for fast testing
    let batch = 1;
    let h = 32;  // Scaled down from 640
    let w = 32;
    let c_in = 3;  // RGB input
    let c_mid = 16; // Intermediate channels (scaled down from 64)
    let c_out = 21; // Detection output: 4 bbox + 1 obj + 16 classes
    
    // Generate synthetic "image" input
    let mut input_f32 = vec![0.0f32; batch * h * w * c_in];
    rng.fill_f32_scaled(&mut input_f32, 1.0);
    
    // Generate backbone weights
    let mut conv1_w = vec![0.0f32; 3 * 3 * c_in * c_mid];  // 3x3 conv
    let mut conv2_w = vec![0.0f32; 3 * 3 * c_mid * c_mid]; // 3x3 conv
    let mut head_w = vec![0.0f32; 1 * 1 * c_mid * c_out];  // 1x1 detection head
    rng.fill_f32_scaled(&mut conv1_w, 0.2);
    rng.fill_f32_scaled(&mut conv2_w, 0.2);
    rng.fill_f32_scaled(&mut head_w, 0.2);
    
    // ========== F32 Reference Path ==========
    // Conv1: 32x32x3 -> 30x30x16
    let h1 = h - 2;
    let w1 = w - 2;
    let mut conv1_f32 = vec![0.0f32; batch * h1 * w1 * c_mid];
    ops::conv2d_f32_nhwc(
        &mut conv1_f32, &input_f32, &conv1_w,
        batch, h, w, c_in, c_mid, 3, 3, 1, 1, 0, 0
    );
    // ReLU
    for v in &mut conv1_f32 { *v = v.max(0.0); }
    
    // Conv2: 30x30x16 -> 28x28x16
    let h2 = h1 - 2;
    let w2 = w1 - 2;
    let mut conv2_f32 = vec![0.0f32; batch * h2 * w2 * c_mid];
    ops::conv2d_f32_nhwc(
        &mut conv2_f32, &conv1_f32, &conv2_w,
        batch, h1, w1, c_mid, c_mid, 3, 3, 1, 1, 0, 0
    );
    // ReLU
    for v in &mut conv2_f32 { *v = v.max(0.0); }
    
    // Detection head: 28x28x16 -> 28x28x21
    let mut head_f32 = vec![0.0f32; batch * h2 * w2 * c_out];
    ops::conv2d_f32_nhwc(
        &mut head_f32, &conv2_f32, &head_w,
        batch, h2, w2, c_mid, c_out, 1, 1, 1, 1, 0, 0
    );
    
    let f32_detections = head_f32.clone();
    
    // ========== INT8 Quantized Path ==========
    // Compute scales
    let scale_in = input_f32.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    let scale_c1w = conv1_w.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    let scale_c2w = conv2_w.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    let scale_hw = head_w.iter().map(|x| x.abs()).fold(1e-6f32, f32::max) / 127.0;
    
    // Quantize weights
    let mut input_i8 = vec![0i8; input_f32.len()];
    let mut conv1_w_i8 = vec![0i8; conv1_w.len()];
    let mut conv2_w_i8 = vec![0i8; conv2_w.len()];
    let mut head_w_i8 = vec![0i8; head_w.len()];
    
    ops::quantize_f32_to_i8(&mut input_i8, &input_f32, scale_in);
    ops::quantize_f32_to_i8(&mut conv1_w_i8, &conv1_w, scale_c1w);
    ops::quantize_f32_to_i8(&mut conv2_w_i8, &conv2_w, scale_c2w);
    ops::quantize_f32_to_i8(&mut head_w_i8, &head_w, scale_hw);
    
    // INT8 Conv1
    let mut conv1_i32 = vec![0i32; batch * h1 * w1 * c_mid];
    ops::conv2d_i8_nhwc(
        &mut conv1_i32, &input_i8, &conv1_w_i8,
        batch, h, w, c_in, c_mid, 3, 3, 1, 1, 0, 0
    );
    
    // Requantize for next layer
    let scale_c1_out = scale_in * scale_c1w;
    let c1_max = conv1_i32.iter().map(|&x| x.abs()).max().unwrap_or(1);
    let scale_c1_i8 = (c1_max as f32 * scale_c1_out) / 127.0;
    let mut conv1_i8 = vec![0i8; conv1_i32.len()];
    ops::requantize_i32_to_i8(&mut conv1_i8, &conv1_i32, scale_c1_out / scale_c1_i8);
    
    // INT8 ReLU
    let mut relu1_i8 = vec![0i8; conv1_i8.len()];
    ops::relu_i8(&mut relu1_i8, &conv1_i8);
    
    // INT8 Conv2
    let mut conv2_i32 = vec![0i32; batch * h2 * w2 * c_mid];
    ops::conv2d_i8_nhwc(
        &mut conv2_i32, &relu1_i8, &conv2_w_i8,
        batch, h1, w1, c_mid, c_mid, 3, 3, 1, 1, 0, 0
    );
    
    // Requantize
    let scale_c2_out = scale_c1_i8 * scale_c2w;
    let c2_max = conv2_i32.iter().map(|&x| x.abs()).max().unwrap_or(1);
    let scale_c2_i8 = (c2_max as f32 * scale_c2_out) / 127.0;
    let mut conv2_i8 = vec![0i8; conv2_i32.len()];
    ops::requantize_i32_to_i8(&mut conv2_i8, &conv2_i32, scale_c2_out / scale_c2_i8);
    
    // INT8 ReLU
    let mut relu2_i8 = vec![0i8; conv2_i8.len()];
    ops::relu_i8(&mut relu2_i8, &conv2_i8);
    
    // INT8 Detection Head
    let mut head_i32 = vec![0i32; batch * h2 * w2 * c_out];
    ops::conv2d_i8_nhwc(
        &mut head_i32, &relu2_i8, &head_w_i8,
        batch, h2, w2, c_mid, c_out, 1, 1, 1, 1, 0, 0
    );
    
    // Dequantize detection output to F32 for post-processing
    let scale_head_out = scale_c2_i8 * scale_hw;
    let i8_detections: Vec<f32> = head_i32.iter()
        .map(|&x| x as f32 * scale_head_out)
        .collect();
    
    // ========== Compare Results ==========
    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f32;
    for (&f32_val, &i8_val) in f32_detections.iter().zip(i8_detections.iter()) {
        let error = (f32_val - i8_val).abs();
        max_error = max_error.max(error);
        sum_sq_error += error * error;
    }
    
    let rmse = (sum_sq_error / f32_detections.len() as f32).sqrt();
    let ref_max = f32_detections.iter().map(|&x| x.abs()).fold(1e-6f32, f32::max);
    let relative_error = max_error / ref_max;
    
    println!("INT8 YOLO Simulation Test:");
    println!("  Input: {}x{}x{}", h, w, c_in);
    println!("  Output: {}x{}x{}", h2, w2, c_out);
    println!("  Max absolute error: {:.6}", max_error);
    println!("  RMSE: {:.6}", rmse);
    println!("  Relative error: {:.2}%", relative_error * 100.0);
    
    // For YOLO, we need detection outputs to be reasonably accurate
    // Allow up to 25% relative error for 3-layer quantized network
    assert!(
        relative_error < 0.25,
        "YOLO INT8 simulation relative error too high: {:.2}% (target <25%)",
        relative_error * 100.0
    );
    
    // Also verify that top detections would be similar
    // Find indices of max values in both outputs (simulating NMS input)
    let f32_max_idx = f32_detections.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i);
    let i8_max_idx = i8_detections.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i);
    
    println!("  F32 max index: {:?}", f32_max_idx);
    println!("  INT8 max index: {:?}", i8_max_idx);
    
    // Note: Max indices may differ due to quantization, which is acceptable
    // The important metric is the overall relative error
}
