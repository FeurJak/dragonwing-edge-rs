//! Cross-backend parity test harness.
//!
//! Runs the same operations on CPU and Vulkan backends and compares results.

use dragonwing_core::{Backend, BufferKind};
use dragonwing_cpu::CpuBackend;
use dragonwing_vulkan::{VulkanBackend, ops as vk_ops};

use crate::generators;

/// Configuration for parity tests.
#[derive(Debug, Clone)]
pub struct TestConfig {
    /// Tolerance for element-wise ops (fill, axpy, relu).
    pub elementwise_tol: f32,
    /// Tolerance for GEMM (higher due to accumulated FMA differences).
    pub gemm_tol: f32,
    /// Number of random test cases per op.
    pub num_trials: usize,
    /// Base seed for random number generation.
    pub seed: u64,
    /// Print verbose output.
    pub verbose: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            // Tolerance for element-wise ops. Set to 1e-4 to account for
            // FMA rounding differences between NEON vfmaq_f32 and GPU fma().
            elementwise_tol: 1e-4,
            // Tolerance for GEMM (higher due to accumulated FMA differences).
            gemm_tol: 1e-3,
            num_trials: 5,
            seed: 0xDEAD_BEEF_CAFE_BABEu64,
            verbose: false,
        }
    }
}

/// Result of a single test.
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Name of the test.
    pub name: String,
    /// Whether the test passed.
    pub passed: bool,
    /// Error message if failed.
    pub error: Option<String>,
    /// Maximum absolute difference observed.
    pub max_diff: f32,
}

/// Collection of test results.
#[derive(Debug, Default)]
pub struct TestResults {
    /// Individual test results.
    pub results: Vec<TestResult>,
}

impl TestResults {
    /// Count passed tests.
    pub fn passed(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Count failed tests.
    pub fn failed(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Print a summary of results.
    pub fn print_summary(&self) {
        println!("\n=== Parity Test Results ===\n");
        for result in &self.results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            print!("{}: {} (max_diff={:.2e})", result.name, status, result.max_diff);
            if let Some(err) = &result.error {
                print!(" - {err}");
            }
            println!();
        }
        println!("\nTotal: {} passed, {} failed", self.passed(), self.failed());
    }

    /// Returns true if all tests passed.
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }
}

/// Parity test runner.
#[derive(Debug)]
pub struct ParityTest;

impl ParityTest {
    /// Run all parity tests.
    pub fn run_all(
        cpu: &CpuBackend,
        vulkan: &VulkanBackend,
        config: &TestConfig,
    ) -> TestResults {
        let mut results = TestResults::default();

        for trial in 0..config.num_trials {
            let seed = config.seed.wrapping_add(trial as u64 * 12345);

            results.results.push(Self::test_fill(cpu, vulkan, config, seed, trial));
            results.results.push(Self::test_axpy(cpu, vulkan, config, seed, trial));
            results.results.push(Self::test_relu(cpu, vulkan, config, seed, trial));
            results.results.push(Self::test_gemm(cpu, vulkan, config, seed, trial));
        }

        results
    }

    fn test_fill(
        _cpu: &CpuBackend,
        vulkan: &VulkanBackend,
        config: &TestConfig,
        seed: u64,
        trial: usize,
    ) -> TestResult {
        let name = format!("fill_f32[{trial}]");
        let (n, value) = generators::fill_test_data(seed);

        // CPU - use the ops directly on a Vec
        let mut cpu_result = vec![0.0f32; n];
        dragonwing_cpu::ops::fill_f32(&mut cpu_result, value);

        // Vulkan
        let mut vk_buf = match vulkan.alloc(n * 4, BufferKind::Storage) {
            Ok(b) => b,
            Err(e) => return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan alloc failed: {e}")),
                max_diff: f32::NAN,
            },
        };
        if let Err(e) = vk_ops::fill_f32(vulkan, &mut vk_buf, value) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan fill failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        if let Err(e) = vulkan.synchronize() {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan sync failed: {e}")),
                max_diff: f32::NAN,
            };
        }

        let mut vk_result = vec![0u8; n * 4];
        if let Err(e) = vulkan.download(&vk_buf, &mut vk_result) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan download failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        let vk_floats: &[f32] = bytemuck_cast(&vk_result);

        // Compare
        compare_f32(&name, &cpu_result, vk_floats, config.elementwise_tol)
    }

    fn test_axpy(
        _cpu: &CpuBackend,
        vulkan: &VulkanBackend,
        config: &TestConfig,
        seed: u64,
        trial: usize,
    ) -> TestResult {
        let name = format!("axpy_f32[{trial}]");
        let (x_data, y_data, alpha) = generators::axpy_test_data(seed);
        let n = x_data.len();

        // CPU - axpy_f32(y, a, x) computes y = a*x + y
        let mut cpu_result = y_data.clone();
        dragonwing_cpu::ops::axpy_f32(&mut cpu_result, alpha, &x_data);

        // Vulkan
        let mut vk_x = match vulkan.alloc(n * 4, BufferKind::Storage) {
            Ok(b) => b,
            Err(e) => return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan alloc x failed: {e}")),
                max_diff: f32::NAN,
            },
        };
        let mut vk_y = match vulkan.alloc(n * 4, BufferKind::Storage) {
            Ok(b) => b,
            Err(e) => return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan alloc y failed: {e}")),
                max_diff: f32::NAN,
            },
        };
        if let Err(e) = vulkan.upload(&mut vk_x, bytemuck_cast_slice(&x_data)) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan upload x failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        if let Err(e) = vulkan.upload(&mut vk_y, bytemuck_cast_slice(&y_data)) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan upload y failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        if let Err(e) = vk_ops::axpy_f32(vulkan, &vk_x, &mut vk_y, alpha) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan axpy failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        if let Err(e) = vulkan.synchronize() {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan sync failed: {e}")),
                max_diff: f32::NAN,
            };
        }

        let mut vk_result = vec![0u8; n * 4];
        if let Err(e) = vulkan.download(&vk_y, &mut vk_result) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan download failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        let vk_floats: &[f32] = bytemuck_cast(&vk_result);

        compare_f32(&name, &cpu_result, vk_floats, config.elementwise_tol)
    }

    fn test_relu(
        _cpu: &CpuBackend,
        vulkan: &VulkanBackend,
        config: &TestConfig,
        seed: u64,
        trial: usize,
    ) -> TestResult {
        let name = format!("relu_f32[{trial}]");
        let x_data = generators::relu_test_data(seed);
        let n = x_data.len();

        // CPU - relu_f32(y, x) computes y = max(0, x)
        // For in-place, we use the same buffer for input and output
        let mut cpu_result = vec![0.0f32; n];
        dragonwing_cpu::ops::relu_f32(&mut cpu_result, &x_data);

        // Vulkan - in-place relu
        let mut vk_x = match vulkan.alloc(n * 4, BufferKind::Storage) {
            Ok(b) => b,
            Err(e) => return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan alloc failed: {e}")),
                max_diff: f32::NAN,
            },
        };
        if let Err(e) = vulkan.upload(&mut vk_x, bytemuck_cast_slice(&x_data)) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan upload failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        if let Err(e) = vk_ops::relu_f32(vulkan, &mut vk_x) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan relu failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        if let Err(e) = vulkan.synchronize() {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan sync failed: {e}")),
                max_diff: f32::NAN,
            };
        }

        let mut vk_result = vec![0u8; n * 4];
        if let Err(e) = vulkan.download(&vk_x, &mut vk_result) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan download failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        let vk_floats: &[f32] = bytemuck_cast(&vk_result);

        compare_f32(&name, &cpu_result, vk_floats, config.elementwise_tol)
    }

    fn test_gemm(
        _cpu: &CpuBackend,
        vulkan: &VulkanBackend,
        config: &TestConfig,
        seed: u64,
        trial: usize,
    ) -> TestResult {
        let name = format!("gemm_f32[{trial}]");
        let (a_data, b_data, m, n, k) = generators::gemm_test_data(seed);

        // CPU - gemm_f32_naive(c, a, b, m, n, k)
        let mut cpu_result = vec![0.0f32; m * n];
        dragonwing_cpu::ops::gemm_f32_naive(&mut cpu_result, &a_data, &b_data, m, n, k);

        // Vulkan
        let mut vk_a = match vulkan.alloc(m * k * 4, BufferKind::Storage) {
            Ok(b) => b,
            Err(e) => return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan alloc A failed: {e}")),
                max_diff: f32::NAN,
            },
        };
        let mut vk_b = match vulkan.alloc(k * n * 4, BufferKind::Storage) {
            Ok(b) => b,
            Err(e) => return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan alloc B failed: {e}")),
                max_diff: f32::NAN,
            },
        };
        let mut vk_c = match vulkan.alloc(m * n * 4, BufferKind::Storage) {
            Ok(b) => b,
            Err(e) => return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan alloc C failed: {e}")),
                max_diff: f32::NAN,
            },
        };
        if let Err(e) = vulkan.upload(&mut vk_a, bytemuck_cast_slice(&a_data)) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan upload A failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        if let Err(e) = vulkan.upload(&mut vk_b, bytemuck_cast_slice(&b_data)) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan upload B failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        if let Err(e) = vk_ops::gemm_f32(vulkan, &vk_a, &vk_b, &mut vk_c, m, n, k) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan gemm failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        if let Err(e) = vulkan.synchronize() {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan sync failed: {e}")),
                max_diff: f32::NAN,
            };
        }

        let mut vk_result = vec![0u8; m * n * 4];
        if let Err(e) = vulkan.download(&vk_c, &mut vk_result) {
            return TestResult {
                name,
                passed: false,
                error: Some(format!("Vulkan download failed: {e}")),
                max_diff: f32::NAN,
            };
        }
        let vk_floats: &[f32] = bytemuck_cast(&vk_result);

        compare_f32(&name, &cpu_result, vk_floats, config.gemm_tol)
    }
}

/// Compare two f32 slices with tolerance.
fn compare_f32(name: &str, expected: &[f32], actual: &[f32], tol: f32) -> TestResult {
    if expected.len() != actual.len() {
        return TestResult {
            name: name.to_string(),
            passed: false,
            error: Some(format!(
                "length mismatch: expected {} got {}",
                expected.len(),
                actual.len()
            )),
            max_diff: f32::NAN,
        };
    }

    let mut max_diff = 0.0f32;
    let mut first_mismatch: Option<(usize, f32, f32)> = None;

    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tol && first_mismatch.is_none() {
            first_mismatch = Some((i, e, a));
        }
    }

    if let Some((idx, exp, act)) = first_mismatch {
        TestResult {
            name: name.to_string(),
            passed: false,
            error: Some(format!(
                "mismatch at [{idx}]: expected {exp}, got {act} (diff={:.2e})",
                (exp - act).abs()
            )),
            max_diff,
        }
    } else {
        TestResult {
            name: name.to_string(),
            passed: true,
            error: None,
            max_diff,
        }
    }
}

// Minimal bytemuck-like helpers (no external dep).
fn bytemuck_cast<T>(bytes: &[u8]) -> &[T] {
    let len = bytes.len() / std::mem::size_of::<T>();
    // SAFETY: T is f32 which is valid for any bit pattern.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len) }
}

fn bytemuck_cast_slice<T>(data: &[T]) -> &[u8] {
    let len = data.len() * std::mem::size_of::<T>();
    // SAFETY: reinterpreting T as bytes is always safe.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), len) }
}
