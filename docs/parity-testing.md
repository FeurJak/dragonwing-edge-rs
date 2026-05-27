# Cross-Backend Parity Testing

This document describes the parity testing framework in `dragonwing-test`, which verifies that CPU and Vulkan backends produce numerically equivalent results.

## Purpose

Parity testing serves two goals:

1. **Correctness verification**: Ensure GPU compute shaders implement the same algorithms as CPU reference code.

2. **Regression detection**: Catch changes that break numerical equivalence between backends.

## Architecture

```
dragonwing-test
├── generators.rs   — Deterministic test data generation
├── harness.rs      — Test runner and comparison logic
└── bin/
    └── parity_test.rs  — CLI test binary
```

### Test Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Generate    │───►│ Run on CPU   │───►│ Compare     │
│ Test Data   │    │ Run on GPU   │    │ Results     │
│ (seed-based)│    │ (same data)  │    │ (tolerance) │
└─────────────┘    └──────────────┘    └─────────────┘
```

## Running Tests

On the device:

```bash
cargo build --release --bin parity-test
./target/release/parity-test
```

Expected output:
```
=== dragonwing-edge cross-backend parity test ===

CPU backend: cpu-neon
Vulkan backend: Turnip Adreno (TM) 702 (turnip Mesa driver - Mesa 25.2.6)

=== Parity Test Results ===

fill_f32[0]: PASS (max_diff=0.00e0)
axpy_f32[0]: PASS (max_diff=6.10e-5)
relu_f32[0]: PASS (max_diff=0.00e0)
gemm_f32[0]: PASS (max_diff=1.43e-6)
...

Total: 20 passed, 0 failed

All parity tests passed!
```

## Test Configuration

```rust
use dragonwing_test::TestConfig;

let config = TestConfig {
    // Tolerance for element-wise ops (fill, axpy, relu)
    elementwise_tol: 1e-4,
    
    // Higher tolerance for GEMM due to accumulated FMA differences
    gemm_tol: 1e-3,
    
    // Number of random trials per op
    num_trials: 5,
    
    // Base seed for reproducible random data
    seed: 0xDEAD_BEEF_CAFE_BABE,
    
    // Print verbose output
    verbose: true,
};
```

## Tolerance Selection

### Why Non-Zero Tolerance?

Floating-point results can differ between CPU and GPU due to:

1. **FMA rounding**: GPU `fma(a,b,c)` vs CPU `a*b+c` may round differently
2. **Instruction scheduling**: Different evaluation order of associative ops
3. **Hardware precision**: Minor differences in FP unit implementations

### Recommended Tolerances

| Op Type | Tolerance | Rationale |
|---------|-----------|-----------|
| fill | 0.0 | Exact copy, no arithmetic |
| axpy | 1e-4 | Single FMA per element |
| relu | 0.0 | Exact comparison (max) |
| gemm | 1e-3 | O(K) FMAs accumulate error |

### Scaling with Problem Size

For GEMM, error grows with `K` (inner dimension):
- K=32: ~1e-6 max error
- K=256: ~1e-4 max error
- K=1024: ~1e-3 max error

## Data Generation

Test data is generated deterministically from a seed:

```rust
use dragonwing_test::generators;

// Same seed → same data every time
let (n, value) = generators::fill_test_data(0x12345);
let (x, y, alpha) = generators::axpy_test_data(0x12345);
let x_data = generators::relu_test_data(0x12345);
let (a, b, m, n, k) = generators::gemm_test_data(0x12345);
```

### Data Ranges

| Op | Range | Rationale |
|----|-------|-----------|
| fill | [-100, 100] | General range |
| axpy | x,y: [-100, 100], α: [-10, 10] | Moderate magnitudes |
| relu | [-100, 100] | Mix of negative/positive |
| gemm | [-1, 1] | Small values prevent accumulation overflow |

## Programmatic Usage

```rust
use dragonwing_cpu::CpuBackend;
use dragonwing_vulkan::{VulkanBackend, VulkanConfig};
use dragonwing_test::{ParityTest, TestConfig};

fn main() {
    let cpu = CpuBackend::new();
    let vulkan = VulkanBackend::new(VulkanConfig::default()).unwrap();
    
    let config = TestConfig::default();
    let results = ParityTest::run_all(&cpu, &vulkan, &config);
    
    if results.all_passed() {
        println!("All tests passed!");
    } else {
        for r in &results.results {
            if !r.passed {
                println!("FAIL: {} - {:?}", r.name, r.error);
            }
        }
    }
}
```

## Debugging Failures

### Step 1: Identify the Mismatch

```
axpy_f32[2]: FAIL (max_diff=1.23e-3) - mismatch at [42]: expected -5.678, got -5.680
```

### Step 2: Check Tolerance

Is `1.23e-3` reasonable for this op? For axpy, `1e-4` should suffice. If max_diff >> tolerance, there's likely a bug.

### Step 3: Reproduce with Fixed Seed

```rust
let config = TestConfig {
    seed: 0xDEAD_BEEF_CAFE_BABE + 2 * 12345,  // Trial 2
    num_trials: 1,  // Just this one
    ..Default::default()
};
```

### Step 4: Compare Intermediate Values

Add debug output to both backends:
- CPU: Print intermediate values in ops.rs
- GPU: Use RenderDoc or printf debugging in shaders

### Common Causes

| Symptom | Likely Cause |
|---------|--------------|
| All zeros from GPU | Shader not dispatched, or wrong binding order |
| Wrong but consistent | Incorrect push constants or buffer binding |
| Random garbage | Uninitialized memory, race condition |
| Small systematic diff | FMA vs mul+add, different rounding |

## Adding New Op Tests

1. Add generator in `generators.rs`:
```rust
pub fn new_op_test_data(seed: u64) -> (...) {
    let mut rng = Rng::new(seed);
    // Generate appropriate test data
}
```

2. Add test method in `harness.rs`:
```rust
fn test_new_op(
    cpu: &CpuBackend,
    vulkan: &VulkanBackend,
    config: &TestConfig,
    seed: u64,
    trial: usize,
) -> TestResult {
    // Run op on both backends
    // Compare results
}
```

3. Call from `run_all()`:
```rust
results.results.push(Self::test_new_op(cpu, vulkan, config, seed, trial));
```

## CI Integration

Parity tests should run as part of the device test suite:

```bash
#!/bin/bash
adb shell /home/arduino/dragonwing-edge-rs/target/release/parity-test
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Parity tests failed!"
    exit 1
fi
```

## Future Improvements

- [ ] Parallel test execution
- [ ] Statistical analysis (mean, std dev of differences)
- [ ] Stress testing with larger problem sizes
- [ ] Edge case testing (denormals, infinities, NaN)
- [ ] Differential testing against reference libraries (e.g., BLAS)
