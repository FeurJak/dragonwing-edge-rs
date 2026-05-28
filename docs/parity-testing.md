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

#### F32 ops (carried from task 002)

| Op Type | Tolerance | Rationale |
|---------|-----------|-----------|
| `fill_f32` | 0.0 | Exact copy, no arithmetic |
| `axpy_f32` | 1e-4 | Single FMA per element |
| `relu_f32` | 0.0 | Exact comparison (max) |
| `add_f32` | 1e-5 | Single addition per element |
| `gemm_f32_naive` / `_tiled` | 1e-3 | O(K) FMAs accumulate error |
| `conv2d_f32_nhwc` | 1e-4 relative | Accumulation depth is `K_h * K_w * C_in`; differs by accumulation order across backends |
| `maxpool2d_f32` | 1e-5 | Selection op; only round-off boundary cases produce diff |
| `avgpool2d_f32` | 1e-5 | Short window sums (typically 2×2 = 4 elements) |
| `softmax_f32` | 1e-4 relative | `exp` is the dominant error term; intermediate sums in F32 |

#### FP16 ops (task 003)

| Op Type | Tolerance | Rationale |
|---------|-----------|-----------|
| `fill_fp16` | exact bit pattern on representable inputs | Deterministic round-to-nearest-even |
| `axpy_fp16` | 1e-3 relative | F32 accumulator, FP16 round on store |
| `relu_fp16` | exact | Pass-through; no arithmetic |
| `add_fp16` | 1e-3 relative | Same model as axpy without the scalar |
| `gemm_fp16` | 5e-3 relative | F32 accumulator over K terms, FP16 result; tolerance reflects mantissa precision (10 bits) |
| `conv2d_fp16_nhwc` (CPU only) | 5e-3 relative | Same accumulation model as `gemm_fp16` |

#### Task 004 ops

| Op Type | Tolerance | Rationale |
|---------|-----------|-----------|
| `depthwise_conv2d_f32_nhwc` | 1e-4 relative | Same as standard conv; kernel is 3×3 so K=9 |
| `depthwise_conv2d_fp16_nhwc` | 5e-3 relative | F32 accumulator, FP16 inputs/outputs |
| `relu6_f32` | 0.0 | Exact clamp operation |
| `relu6_fp16` | 0.0 | Exact clamp (within FP16 representability) |
| `global_avg_pool_f32_nhwc` | 1e-5 | Sum over H×W elements; typical spatial is 7×7=49 |
| `global_avg_pool_fp16_nhwc` | 1e-3 | F32 accumulator, FP16 result |

#### End-to-end MobileNetV2

| Comparison | Tolerance | Notes |
|------------|-----------|-------|
| CPU F32 vs reference | 1e-3 relative | Top-5 logit comparison |
| CPU F16 vs reference | 5e-2 relative | Wider tolerance for FP16 accumulation |
| Vulkan F32 vs CPU F32 | 1e-3 relative | (Task 005 — not yet tested) |

### Scaling with Problem Size

For GEMM, error grows with `K` (inner dimension):
- K=32: ~1e-6 max error (F32) / ~1e-3 (FP16)
- K=256: ~1e-4 max error (F32) / ~3e-3 (FP16)
- K=1024: ~1e-3 max error (F32) / ~5e-3 (FP16, near tolerance limit)

For Conv2D, effective `K` is `K_h * K_w * C_in`:
- 3×3 conv with C_in=64: K=576 ≈ GEMM K=512 behaviour.
- 1×1 conv with C_in=256: K=256 ≈ same as GEMM K=256.

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
| add | [-10, 10] | Moderate magnitudes |
| gemm | [-1, 1] | Small values prevent accumulation overflow at large K |
| conv2d | input [-1, 1], weights [-0.5, 0.5] | Same accumulation budget as GEMM |
| maxpool / avgpool | [-10, 10] | Mix of magnitudes |
| softmax | [-5, 5] | After max-subtraction, exp inputs land in [-10, 0]; tractable |
| depthwise conv | input [-1, 1], weights [-0.5, 0.5] | Same as standard conv (K=9 for 3×3) |
| global_avg_pool | [-10, 10] | Sum over spatial dims |
| relu6 / clip | [-10, 10] | Mix of values inside and outside [0, 6] |
| FP16 variants of above | same range, narrower distribution | FP16 max ≈ 65504; we stay well clear |

### End-to-end micro-graph

The `dragonwing-test::micro_graph` test exercises the new ops in sequence:

```
Input 1×28×28×1
  → Conv3×3 (1→16, stride 1, pad 1) → ReLU
  → MaxPool 2×2 (stride 2)
  → Conv3×3 (16→32, stride 1, pad 1) → ReLU
  → MaxPool 2×2 (stride 2)
  → Flatten (reshape, zero cost)
  → GEMM (32·7·7 → 10)
  → Softmax
```

Tolerance: `1e-4` relative for the whole pipeline. The test validates CPU
multi-thread parity, deterministic output for a fixed seed, and that different
seeds produce different output.

### End-to-end MobileNetV2 (task 004)

The `dragonwing-onnx` crate includes an end-to-end MobileNetV2 test:

```
Input 1×224×224×3 (NHWC, after layout conversion)
  → 52 Conv layers (standard + depthwise)
  → BatchNorm (folded into Conv at load time)
  → ReLU6 activations
  → GlobalAveragePool
  → Gemm (1280 → 1000)
  → Softmax
```

The test:
1. Loads `artifacts/models/mobilenetv2-12.onnx`
2. Folds BatchNorm into preceding Conv
3. Converts NCHW → NHWC
4. Runs inference on CPU
5. Compares top-5 predictions against reference

Tolerance: `1e-3` relative for F32, `5e-2` relative for FP16.

Note: Vulkan end-to-end parity testing is planned for task 005.

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
