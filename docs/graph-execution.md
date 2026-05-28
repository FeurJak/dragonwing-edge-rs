# Graph Execution

This document describes the graph compilation and execution model in `dragonwing-onnx`.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         dragonwing-onnx                             │
├─────────────────────────────────────────────────────────────────────┤
│  Model          │  Graph           │  Runtime                       │
│  (ONNX parsing) │  (compilation)   │  (execution)                   │
│                 │                  │                                │
│  load_model()   │  validate_model()│  GraphRuntime<B> (generic)     │
│  parse_model()  │  compile_model() │  CpuGraphRuntime (optimized)   │
│                 │  fold_batchnorm()│                                │
│                 │  convert_nchw_*()│                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Two-Phase Compilation

### Why Two Phases?

The two-phase model prevents resource leaks and provides better error reporting:

1. **Validation fails fast**: If op #47 is unsupported, you learn immediately without allocating buffers for ops #1-46.

2. **All-or-nothing compilation**: Either the entire graph compiles successfully, or nothing is allocated.

3. **Better error aggregation**: Validation collects ALL unsupported ops, not just the first one.

### Phase 1: Validation

```rust
pub fn validate_model(model: &Model, dtype: Dtype) -> ValidationReport
```

The validation pass:

1. **Registers input shapes** — Model inputs get their shapes from the ONNX graph definition. Dynamic dimensions (typically batch size = -1) are replaced with 1.

2. **Registers initializer shapes** — Constant tensors (weights, biases) are read from the ONNX file and their shapes recorded.

3. **Validates each node** — For each node in topological order:
   - Look up the `OpBuilder` by op type
   - Call `is_supported()` for quick compatibility check
   - Call `build()` to populate shapes and constants (needed for constant folding)
   - Record result in `ValidationReport`

```rust
pub struct ValidationReport {
    pub supported: Vec<String>,      // Node names that passed
    pub unsupported: Vec<(String, UnsupportedReason)>,  // Failures
}
```

### Phase 2: Compilation

```rust
pub fn compile_model(model: &Model, dtype: Dtype) -> Result<Graph>
```

After validation passes, compilation:

1. **Re-walks the graph** — Same topological order
2. **Builds each op** — Creates `CompiledOp` with runtime parameters
3. **Collects initializers** — Filters to only F32/F16 tensors actually used by ops
4. **Returns executable Graph**

```rust
pub struct Graph {
    pub ops: Vec<CompiledOp>,           // Ops in execution order
    pub shapes: HashMap<String, TensorShape>,  // All tensor shapes
    pub inputs: Vec<String>,             // Input tensor names
    pub outputs: Vec<String>,            // Output tensor names
    pub initializers: HashMap<String, Vec<u8>>,  // Weight data
    pub dtype: Dtype,                    // Graph dtype (F32 or F16)
}
```

## CompiledOp Structure

Each compiled op carries everything needed for runtime dispatch:

```rust
pub struct CompiledOp {
    pub name: String,           // Node name (for debugging)
    pub op_type: String,        // ONNX op type ("Conv", "Relu", etc.)
    pub inputs: Vec<String>,    // Input tensor names
    pub outputs: Vec<String>,   // Output tensor names
    pub params: OpParams,       // Op-specific parameters
}

pub enum OpParams {
    None,  // Relu, etc.
    Clip { min: f32, max: f32 },
    Conv2d {
        kernel_shape: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        dilations: [usize; 2],
        group: usize,
    },
    Gemm { alpha: f32, beta: f32, trans_a: bool, trans_b: bool },
    Softmax { axis: i64 },
    Reshape { shape: Vec<usize> },
    GlobalAvgPool,
    Add,
    MaxPool { kernel_shape: [usize; 2], strides: [usize; 2], pads: [usize; 4] },
    AvgPool { kernel_shape: [usize; 2], strides: [usize; 2], pads: [usize; 4] },
}
```

## Tensor Lifetime and Memory Planning

### Current Model: Per-Tensor Allocation

Task 004 uses the simplest viable memory plan: **one buffer per tensor, no reuse**.

For MobileNetV2 at F32:
- ~100 tensors
- ~30 MiB total activation memory
- ~10 MiB total weight memory
- Well under the 870 MiB heap budget

### Future: Arena Allocation (Task 005+)

For larger models (YOLO), tensor lifetime analysis enables buffer reuse:

```
Tensor A: [────────]
Tensor B:      [────────]
Tensor C:                [────────]
          
A and C can share the same buffer (non-overlapping lifetimes)
```

This is tracked in task 005 as "Vulkan slab allocator."

## Runtime Variants

### Generic Runtime: `GraphRuntime<B: Backend>`

Works with any backend but uses download/upload for every op:

```rust
fn dispatch_op(&mut self, op: &CompiledOp) -> Result<()> {
    // 1. Download input tensors from backend buffers to CPU memory
    // 2. Execute op on CPU
    // 3. Upload output tensors to backend buffers
}
```

This is correct but slow — useful for debugging and as a fallback.

### CPU Runtime: `CpuGraphRuntime`

Operates directly on CPU buffer slices, avoiding download/upload overhead:

```rust
fn dispatch_relu(&mut self, op: &CompiledOp) -> Result<()> {
    let input = self.buffers.get(input_name).as_f32();
    let output = self.buffers.get_mut(output_name).as_f32_mut();
    ops::relu_f32(output, input);  // Direct slice access
    Ok(())
}
```

Uses optimized ops from `dragonwing_cpu::ops`:
- Multi-threaded GEMM (`gemm_f32_mt`)
- Multi-threaded Conv2D (`conv2d_f32_nhwc_mt`)
- NEON-accelerated element-wise ops

### Vulkan Runtime (Task 005+)

Will use `GraphRuntime<VulkanBackend>` with proper command buffer batching.

## Execution Flow

```rust
// 1. Create runtime (allocates buffers, uploads weights)
let mut runtime = CpuGraphRuntime::new(graph)?;

// 2. Set inputs
runtime.set_input_f32("input", &input_data)?;

// 3. Run inference
runtime.run()?;  // Executes all ops in order

// 4. Get outputs
let output = runtime.get_output_f32("output")?;
```

### Inside `run()`

```rust
pub fn run(&mut self) -> Result<()> {
    let ops = self.graph.ops.clone();
    for op in &ops {
        self.dispatch_op(op)?;
    }
    Ok(())
}
```

Each op is dispatched based on `OpParams`:

```rust
fn dispatch_op(&mut self, op: &CompiledOp) -> Result<()> {
    match &op.params {
        OpParams::None => match op.op_type.as_str() {
            "Relu" => self.dispatch_relu(op),
            "Flatten" => self.dispatch_reshape(op),
            _ => Ok(()),
        },
        OpParams::Conv2d { .. } => self.dispatch_conv2d(op, ...),
        OpParams::Gemm { .. } => self.dispatch_gemm(op, ...),
        // ... etc
    }
}
```

## Dtype Handling

### Graph-Wide Dtype

The entire graph uses a single dtype (F32 or F16), specified at compile time:

```rust
let graph = compile_model(&model, Dtype::F16)?;
```

### Weight Conversion

Weights are stored as F32 in ONNX files. On runtime creation, they're converted to the graph dtype:

```rust
let upload_data = match (shape.dtype, graph.dtype) {
    (Dtype::F32, Dtype::F16) => convert_f32_to_f16(data),
    (Dtype::F16, Dtype::F32) => convert_f16_to_f32(data),
    _ => data.clone(),
};
```

### Mixed Precision (Task 005+)

For models that need specific layers in F32 (e.g., normalization), per-op dtype override is planned but not yet implemented.

## Error Handling

Runtime errors include context for debugging:

```rust
Error::Runtime(format!("buffer not found: {name}"))
Error::Runtime(format!("input size mismatch: expected {}, got {}", expected, actual))
Error::Runtime(format!("Conv2D dispatch not implemented for op: {}", op.name))
```

## Performance Considerations

### Current Bottlenecks

1. **Per-op cloning**: `self.graph.ops.clone()` creates a full copy of the op list on every `run()`. This should be refactored to use references.

2. **No batching**: Ops execute one at a time. Vulkan runtime should batch command buffers.

3. **No buffer reuse**: Each tensor gets its own allocation. Arena packing would reduce memory pressure.

### Profiling Points

The `CpuGraphRuntime` logs per-op timing when compiled with `--features profiling`:

```
[profile] Conv /conv1: 12.3ms
[profile] Relu /relu1: 0.1ms
[profile] MaxPool /pool1: 1.2ms
```

## File References

- `crates/dragonwing-onnx/src/graph.rs:129-236` — `compile_model()` implementation
- `crates/dragonwing-onnx/src/graph.rs:57-127` — `validate_model()` implementation
- `crates/dragonwing-onnx/src/runtime.rs:16-131` — Generic `GraphRuntime<B>`
- `crates/dragonwing-onnx/src/runtime.rs:744-1500` — `CpuGraphRuntime` implementation
- `crates/dragonwing-onnx/src/builder.rs:170-232` — `CompiledOp` and `OpParams` definitions
