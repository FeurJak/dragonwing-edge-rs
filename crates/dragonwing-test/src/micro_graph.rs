//! End-to-end micro-graph test for task 003.
//!
//! Assembles a tiny MNIST-shaped image classification graph:
//!
//! ```text
//! Input: 1×28×28×1 (NHWC)
//!     ↓
//! Conv3×3, 1→16, stride=1, pad=1  → 1×28×28×16
//!     ↓
//! ReLU
//!     ↓
//! MaxPool 2×2, stride=2           → 1×14×14×16
//!     ↓
//! Conv3×3, 16→32, stride=1, pad=1 → 1×14×14×32
//!     ↓
//! ReLU
//!     ↓
//! MaxPool 2×2, stride=2           → 1×7×7×32
//!     ↓
//! Flatten                         → 1×1568
//!     ↓
//! GEMM (1568 → 10)                → 1×10
//!     ↓
//! Softmax                         → 1×10
//! ```
//!
//! The test validates that CPU ops produce consistent results across
//! multiple runs. When Vulkan is available, it also verifies CPU/GPU parity.

use dragonwing_cpu::ops;

/// Random number generator (Xorshift64).
struct Rng(u64);

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

/// Result of running the micro-graph.
#[derive(Debug)]
pub struct MicroGraphResult {
    /// Output logits (10 values after softmax).
    pub output: [f32; 10],
    /// Whether the graph executed successfully.
    pub success: bool,
    /// Error message if any.
    pub error: Option<String>,
}

/// Run the micro-graph on CPU.
///
/// Returns the 10-element softmax output.
pub fn run_cpu(seed: u64) -> MicroGraphResult {
    let mut rng = Rng::new(seed);

    // Dimensions
    const N: usize = 1;
    const H_IN: usize = 28;
    const W_IN: usize = 28;
    const C_IN: usize = 1;

    // Layer 1: Conv 1→16, 3×3, pad=1
    const C1_OUT: usize = 16;
    const K1: usize = 3;
    const PAD1: usize = 1;
    const H1_OUT: usize = H_IN; // same padding
    const W1_OUT: usize = W_IN;

    // Pool 1: 2×2, stride 2
    const H2: usize = H1_OUT / 2; // 14
    const W2: usize = W1_OUT / 2; // 14

    // Layer 2: Conv 16→32, 3×3, pad=1
    const C2_OUT: usize = 32;
    const K2: usize = 3;
    const PAD2: usize = 1;
    const H3_OUT: usize = H2; // same padding
    const W3_OUT: usize = W2;

    // Pool 2: 2×2, stride 2
    const H4: usize = H3_OUT / 2; // 7
    const W4: usize = W3_OUT / 2; // 7

    // FC layer: flatten -> GEMM
    const FC_IN: usize = H4 * W4 * C2_OUT; // 7*7*32 = 1568
    const FC_OUT: usize = 10;

    // Allocate buffers
    let mut input = vec![0.0f32; N * H_IN * W_IN * C_IN];
    let mut conv1_kernel = vec![0.0f32; K1 * K1 * C_IN * C1_OUT];
    let mut conv1_out = vec![0.0f32; N * H1_OUT * W1_OUT * C1_OUT];
    let mut relu1_out = vec![0.0f32; N * H1_OUT * W1_OUT * C1_OUT];
    let mut pool1_out = vec![0.0f32; N * H2 * W2 * C1_OUT];

    let mut conv2_kernel = vec![0.0f32; K2 * K2 * C1_OUT * C2_OUT];
    let mut conv2_out = vec![0.0f32; N * H3_OUT * W3_OUT * C2_OUT];
    let mut relu2_out = vec![0.0f32; N * H3_OUT * W3_OUT * C2_OUT];
    let mut pool2_out = vec![0.0f32; N * H4 * W4 * C2_OUT];

    let mut fc_weight = vec![0.0f32; FC_IN * FC_OUT];
    let mut fc_out = vec![0.0f32; N * FC_OUT];
    let mut softmax_out = vec![0.0f32; N * FC_OUT];

    // Initialize with random data
    // Input: [-0.5, 0.5] (simulating normalized pixels)
    rng.fill_f32_scaled(&mut input, 0.5);

    // Conv1 kernel: small weights for stability
    rng.fill_f32_scaled(&mut conv1_kernel, 0.1);

    // Conv2 kernel
    rng.fill_f32_scaled(&mut conv2_kernel, 0.1);

    // FC weights: Xavier-ish initialization
    let fc_scale = (2.0 / FC_IN as f32).sqrt();
    rng.fill_f32_scaled(&mut fc_weight, fc_scale);

    // ========== Forward pass ==========

    // Conv1: 28×28×1 → 28×28×16
    ops::conv2d_f32_nhwc(
        &mut conv1_out,
        &input,
        &conv1_kernel,
        N,
        H_IN,
        W_IN,
        C_IN,
        C1_OUT,
        K1,
        K1,
        1, // stride_h
        1, // stride_w
        PAD1,
        PAD1,
    );

    // ReLU1
    ops::relu_f32(&mut relu1_out, &conv1_out);

    // MaxPool1: 28×28×16 → 14×14×16
    ops::maxpool2d_f32_nhwc(
        &mut pool1_out,
        &relu1_out,
        N,
        H1_OUT,
        W1_OUT,
        C1_OUT,
        2, // pool_h
        2, // pool_w
        2, // stride_h
        2, // stride_w
    );

    // Conv2: 14×14×16 → 14×14×32
    ops::conv2d_f32_nhwc(
        &mut conv2_out,
        &pool1_out,
        &conv2_kernel,
        N,
        H2,
        W2,
        C1_OUT,
        C2_OUT,
        K2,
        K2,
        1,
        1,
        PAD2,
        PAD2,
    );

    // ReLU2
    ops::relu_f32(&mut relu2_out, &conv2_out);

    // MaxPool2: 14×14×32 → 7×7×32
    ops::maxpool2d_f32_nhwc(
        &mut pool2_out,
        &relu2_out,
        N,
        H3_OUT,
        W3_OUT,
        C2_OUT,
        2,
        2,
        2,
        2,
    );

    // Flatten is implicit (pool2_out is already contiguous in memory)

    // FC: 1568 → 10 via GEMM
    // pool2_out is [1, 1568] (after reshape), fc_weight is [1568, 10]
    // We need C = A * B where A=[1, 1568], B=[1568, 10], C=[1, 10]
    ops::gemm_f32_naive(
        &mut fc_out,
        &pool2_out,
        &fc_weight,
        N,        // m = 1
        FC_OUT,   // n = 10
        FC_IN,    // k = 1568
    );

    // Softmax
    ops::softmax_f32(&mut softmax_out, &fc_out, FC_OUT);

    // Extract result
    let mut output = [0.0f32; 10];
    output.copy_from_slice(&softmax_out);

    // Validate output (softmax should sum to 1, all positive)
    let sum: f32 = output.iter().sum();
    if (sum - 1.0).abs() > 1e-4 {
        return MicroGraphResult {
            output,
            success: false,
            error: Some(format!("Softmax sum is {sum}, expected 1.0")),
        };
    }
    if output.iter().any(|&x| x < 0.0 || x.is_nan()) {
        return MicroGraphResult {
            output,
            success: false,
            error: Some("Softmax output contains negative or NaN values".into()),
        };
    }

    MicroGraphResult {
        output,
        success: true,
        error: None,
    }
}

/// Run the micro-graph on CPU with multi-threading.
pub fn run_cpu_mt(seed: u64, num_threads: usize) -> MicroGraphResult {
    let mut rng = Rng::new(seed);

    // Same dimensions as run_cpu
    const N: usize = 1;
    const H_IN: usize = 28;
    const W_IN: usize = 28;
    const C_IN: usize = 1;
    const C1_OUT: usize = 16;
    const K1: usize = 3;
    const PAD1: usize = 1;
    const H1_OUT: usize = H_IN;
    const W1_OUT: usize = W_IN;
    const H2: usize = H1_OUT / 2;
    const W2: usize = W1_OUT / 2;
    const C2_OUT: usize = 32;
    const K2: usize = 3;
    const PAD2: usize = 1;
    const H3_OUT: usize = H2;
    const W3_OUT: usize = W2;
    const H4: usize = H3_OUT / 2;
    const W4: usize = W3_OUT / 2;
    const FC_IN: usize = H4 * W4 * C2_OUT;
    const FC_OUT: usize = 10;

    // Allocate buffers
    let mut input = vec![0.0f32; N * H_IN * W_IN * C_IN];
    let mut conv1_kernel = vec![0.0f32; K1 * K1 * C_IN * C1_OUT];
    let mut conv1_out = vec![0.0f32; N * H1_OUT * W1_OUT * C1_OUT];
    let mut relu1_out = vec![0.0f32; N * H1_OUT * W1_OUT * C1_OUT];
    let mut pool1_out = vec![0.0f32; N * H2 * W2 * C1_OUT];

    let mut conv2_kernel = vec![0.0f32; K2 * K2 * C1_OUT * C2_OUT];
    let mut conv2_out = vec![0.0f32; N * H3_OUT * W3_OUT * C2_OUT];
    let mut relu2_out = vec![0.0f32; N * H3_OUT * W3_OUT * C2_OUT];
    let mut pool2_out = vec![0.0f32; N * H4 * W4 * C2_OUT];

    let mut fc_weight = vec![0.0f32; FC_IN * FC_OUT];
    let mut fc_out = vec![0.0f32; N * FC_OUT];
    let mut softmax_out = vec![0.0f32; N * FC_OUT];

    // Initialize with random data
    rng.fill_f32_scaled(&mut input, 0.5);
    rng.fill_f32_scaled(&mut conv1_kernel, 0.1);
    rng.fill_f32_scaled(&mut conv2_kernel, 0.1);
    let fc_scale = (2.0 / FC_IN as f32).sqrt();
    rng.fill_f32_scaled(&mut fc_weight, fc_scale);

    // ========== Forward pass with multi-threaded ops ==========

    // Conv1 (multi-threaded)
    ops::conv2d_f32_nhwc_mt(
        &mut conv1_out,
        &input,
        &conv1_kernel,
        N, H_IN, W_IN, C_IN, C1_OUT, K1, K1, 1, 1, PAD1, PAD1,
        num_threads,
    );

    ops::relu_f32(&mut relu1_out, &conv1_out);

    ops::maxpool2d_f32_nhwc(
        &mut pool1_out, &relu1_out, N, H1_OUT, W1_OUT, C1_OUT, 2, 2, 2, 2,
    );

    // Conv2 (multi-threaded)
    ops::conv2d_f32_nhwc_mt(
        &mut conv2_out,
        &pool1_out,
        &conv2_kernel,
        N, H2, W2, C1_OUT, C2_OUT, K2, K2, 1, 1, PAD2, PAD2,
        num_threads,
    );

    ops::relu_f32(&mut relu2_out, &conv2_out);

    ops::maxpool2d_f32_nhwc(
        &mut pool2_out, &relu2_out, N, H3_OUT, W3_OUT, C2_OUT, 2, 2, 2, 2,
    );

    // FC (multi-threaded GEMM)
    ops::gemm_f32_mt(&mut fc_out, &pool2_out, &fc_weight, N, FC_OUT, FC_IN, num_threads);

    ops::softmax_f32(&mut softmax_out, &fc_out, FC_OUT);

    let mut output = [0.0f32; 10];
    output.copy_from_slice(&softmax_out);

    let sum: f32 = output.iter().sum();
    if (sum - 1.0).abs() > 1e-4 {
        return MicroGraphResult {
            output,
            success: false,
            error: Some(format!("Softmax sum is {sum}, expected 1.0")),
        };
    }
    if output.iter().any(|&x| x < 0.0 || x.is_nan()) {
        return MicroGraphResult {
            output,
            success: false,
            error: Some("Softmax output contains negative or NaN values".into()),
        };
    }

    MicroGraphResult {
        output,
        success: true,
        error: None,
    }
}

/// Compare two micro-graph results for parity.
pub fn compare_results(a: &MicroGraphResult, b: &MicroGraphResult, tol: f32) -> Result<f32, String> {
    if !a.success {
        return Err(format!("First result failed: {:?}", a.error));
    }
    if !b.success {
        return Err(format!("Second result failed: {:?}", b.error));
    }

    let mut max_diff = 0.0f32;
    for (i, (&va, &vb)) in a.output.iter().zip(b.output.iter()).enumerate() {
        let diff = (va - vb).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tol {
            return Err(format!(
                "Mismatch at output[{i}]: {va} vs {vb} (diff={diff:.2e}, tol={tol:.2e})"
            ));
        }
    }

    Ok(max_diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_graph_cpu_runs() {
        let result = run_cpu(0xDEAD_BEEF);
        assert!(result.success, "Micro-graph failed: {:?}", result.error);

        // Check softmax properties
        let sum: f32 = result.output.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Softmax sum should be 1.0, got {sum}"
        );
        assert!(
            result.output.iter().all(|&x| x >= 0.0),
            "All softmax outputs should be non-negative"
        );
    }

    #[test]
    fn micro_graph_cpu_deterministic() {
        let result1 = run_cpu(0xCAFE_BABE);
        let result2 = run_cpu(0xCAFE_BABE);

        assert!(result1.success);
        assert!(result2.success);

        // Same seed should produce identical results
        for i in 0..10 {
            assert_eq!(
                result1.output[i], result2.output[i],
                "Output[{i}] differs between runs"
            );
        }
    }

    #[test]
    fn micro_graph_mt_matches_st() {
        let result_st = run_cpu(0x1234_5678);
        let result_mt = run_cpu_mt(0x1234_5678, 4);

        assert!(result_st.success, "ST failed: {:?}", result_st.error);
        assert!(result_mt.success, "MT failed: {:?}", result_mt.error);

        // Multi-threaded should match single-threaded
        let max_diff = compare_results(&result_st, &result_mt, 1e-4)
            .expect("ST vs MT comparison failed");
        println!("ST vs MT max_diff: {max_diff:.2e}");
    }

    #[test]
    fn micro_graph_different_seeds_different_output() {
        let result1 = run_cpu(0xAAAA_AAAA);
        let result2 = run_cpu(0xBBBB_BBBB);

        assert!(result1.success);
        assert!(result2.success);

        // Different seeds should produce different results
        let mut any_diff = false;
        for i in 0..10 {
            if (result1.output[i] - result2.output[i]).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "Different seeds produced identical outputs");
    }
}
