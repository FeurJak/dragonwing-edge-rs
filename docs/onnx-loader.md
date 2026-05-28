# ONNX Model Loader

This document describes the `dragonwing-onnx` crate — a minimal, zero-dependency ONNX parser and graph compiler for static-shape inference on edge hardware.

## Design Philosophy

### Why Hand-Rolled Protobuf?

ONNX models are stored in Google Protocol Buffer format. Standard approaches would use `prost` or `protobuf-rs`, adding 7+ transitive dependencies and a build script.

Instead, `dragonwing-onnx` implements a minimal protobuf reader (~300 lines) that handles only the wire types ONNX actually uses:

| Wire Type | ONNX Usage |
|-----------|------------|
| Varint (0) | Field numbers, enum values, int64 dims |
| Length-delimited (2) | Nested messages, strings, packed arrays |
| Fixed64 (1) | Unused in ONNX |
| Fixed32 (5) | Float values in raw_data |

This keeps the crate at **zero external runtime dependencies** and simplifies debugging — the entire parser fits in `proto.rs:1-250`.

### Opset Support

We accept **opset 12-19** (the range covering MobileNetV2 and most common ImageNet classifiers). Models outside this range fail with `Error::UnsupportedOpset` and a clear message.

The opset check validates minimum supported features, not exact version match. A model at opset 15 works fine if it only uses ops that haven't changed since opset 12.

## Supported Ops

| Op Type | Status | Notes |
|---------|--------|-------|
| `Conv` | Full | Including depthwise via `group == C_in` |
| `Relu` | Full | |
| `Clip` | Full | Used for Relu6 (`min=0, max=6`) |
| `Add` | Full | Same-shape or broadcast to last dim |
| `GlobalAveragePool` | Full | |
| `Gemm` | Partial | `alpha=1, beta=1` only |
| `MatMul` | Full | |
| `Softmax` | Full | Last axis only |
| `Reshape` | Full | Static shapes only |
| `Flatten` | Full | |
| `Squeeze` | Partial | |
| `Unsqueeze` | Partial | |
| `Transpose` | Partial | |
| `BatchNormalization` | Folded | Always folded into preceding Conv |
| `MaxPool` | Full | |
| `AveragePool` | Full | |
| `Shape` | Folded | Constant-folded at compile time |
| `Gather` | Folded | Constant-folded at compile time |
| `Concat` | Partial | Only for shape tensors |
| `Constant` | Full | |
| `Pad` | Partial | |

### Unsupported Ops (Not Yet Implemented)

These ops are needed for YOLO and other detection models — planned for task 005:

- `Concat` (runtime, for activation tensors)
- `Split`
- `Resize` / `Upsample`
- `Sigmoid`
- `Mul` (runtime)

## Two-Phase Compilation

Following the QNN EP pattern from task 002's analysis, the loader uses a two-phase compilation model:

```
                    ┌─────────────────┐
      ONNX file ───►│ 1. Validate     │
                    │    - Check ops  │
                    │    - Compute    │
                    │      shapes     │
                    └────────┬────────┘
                             │ ValidationReport
                             ▼
                    ┌─────────────────┐
                    │ 2. Compile      │
                    │    - Allocate   │
                    │      buffers    │
                    │    - Build ops  │
                    └────────┬────────┘
                             │ Graph
                             ▼
                    ┌─────────────────┐
                    │ 3. Run          │
                    │    - Upload     │
                    │      inputs     │
                    │    - Execute    │
                    │    - Download   │
                    │      outputs    │
                    └─────────────────┘
```

### Phase 1: Validation

```rust
let report = validate_model(&model, Dtype::F32);
if !report.is_fully_supported() {
    for (name, reason) in &report.unsupported {
        eprintln!("Unsupported: {name}: {reason}");
    }
}
```

The validation pass:
- Walks the graph in topological order
- Checks each op via `OpBuilder::is_supported()`
- Computes output shapes without allocating buffers
- Returns a `ValidationReport` with supported/unsupported op lists

This allows fast "can we run this model?" checks without resource allocation.

### Phase 2: Compilation

```rust
let graph = compile_model(&model, Dtype::F32)?;
```

The compile pass:
- Re-walks the graph (after validation passes)
- Calls `OpBuilder::build()` for each node
- Computes tensor shapes and stores in `Graph::shapes`
- Collects initializer data (weights) for later upload

### Phase 3: Execution

```rust
let mut runtime = CpuGraphRuntime::new(graph)?;
runtime.set_input_f32("input", &input_data)?;
runtime.run()?;
let output = runtime.get_output_f32("output")?;
```

## NCHW to NHWC Conversion

ONNX models use **NCHW** layout (batch, channels, height, width).
Dragonwing kernels use **NHWC** layout for efficient memory access on the target hardware.

The conversion happens at compile time:

```rust
let mut graph = compile_model(&model, Dtype::F32)?;
convert_nchw_to_nhwc(&mut graph)?;
```

This:
1. Transposes Conv weight initializers: `[C_out, C_in, kH, kW]` → `[kH, kW, C_in, C_out]`
2. Updates all 4D activation shapes: `[N, C, H, W]` → `[N, H, W, C]`

The caller must transpose input data before feeding to the graph:

```rust
let input_nhwc = transpose_nchw_to_nhwc(&input_nchw, &[1, 3, 224, 224]);
```

## BatchNorm Folding

BatchNormalization is **always** folded into the preceding Conv at load time. We never ship a runtime BN op.

The folding math (computed in f64 for precision):
```
std_inv = 1 / sqrt(var + epsilon)
scale_factor = scale * std_inv
weight_new[c] = weight[c] * scale_factor
bias_new[c] = (bias[c] - mean[c]) * scale_factor + bn_bias[c]
```

If a model has a BN op that cannot be folded (e.g., BN not preceded by Conv, or BN inputs are not constants), the load fails with a clear error.

## Op Builder Pattern

Each ONNX op type has a corresponding `OpBuilder` implementation:

```rust
pub trait OpBuilder: Send + Sync {
    fn op_type(&self) -> &'static str;
    
    fn is_supported(&self, node: &OnnxNode, ctx: &BuildContext<'_>) 
        -> Result<(), UnsupportedReason>;
    
    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) 
        -> Result<TensorShape>;
    
    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) 
        -> Result<CompiledOp>;
}
```

The registry uses a simple `match` statement (~20 cases) rather than `phf` or other proc-macro dependencies:

```rust
// crates/dragonwing-onnx/src/builder.rs:257
pub fn get_builder(op_type: &str) -> Option<&'static dyn OpBuilder> {
    match op_type {
        "Relu" => Some(&RELU_BUILDER),
        "Conv" => Some(&CONV_BUILDER),
        // ... ~20 more cases
        _ => None,
    }
}
```

## Constant Folding

Shape-manipulation ops (`Shape`, `Gather`, `Concat` on shape tensors) are constant-folded at compile time. The runtime never sees these ops.

Example: MobileNetV2's classifier head uses `Shape → Gather → Concat → Reshape` to compute the flatten dimensions. All of this is folded at compile time; the runtime only sees a single `Reshape` op with a static target shape.

## Error Handling

All errors use a single `Error` enum:

```rust
pub enum Error {
    Io(String),
    Parse(String),
    UnsupportedOpset { found: i64, expected: i64 },
    UnsupportedOp { op_type: String, node_name: String, reason: String },
    Validation(String),
    Compile(String),
    Runtime(String),
}
```

Errors include context (node name, op type, tensor name) to help debugging.

## Usage Example

```rust
use dragonwing_onnx::{load_model, compile_model, convert_nchw_to_nhwc};
use dragonwing_onnx::CpuGraphRuntime;
use dragonwing_core::Dtype;

// Load and compile
let model = load_model("mobilenetv2.onnx")?;
let mut graph = compile_model(&model, Dtype::F32)?;
convert_nchw_to_nhwc(&mut graph)?;

// Create runtime and run inference
let mut runtime = CpuGraphRuntime::new(graph)?;
runtime.set_input_f32("input", &input_nhwc)?;
runtime.run()?;
let logits = runtime.get_output_f32("output")?;

// Find top prediction
let (class_id, _) = logits.iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .unwrap();
println!("Predicted class: {class_id}");
```

## Implementation Notes from Task 004

### Non-packed Repeated Fields

ONNX uses non-packed encoding for repeated `int64`/`float` fields in `TensorProto` (dims, float_data, int64_data). The parser handles both packed (length-delimited blob) and non-packed (one wire element per value) encodings.

### Weight Transpose Permutation

Standard conv weights `[C_out, C_in, kH, kW]` must be transposed with permutation `[2, 3, 1, 0]` to get `[kH, kW, C_in, C_out]`. The initial implementation used `[0, 2, 3, 1]` which produced wrong results.

### INT64 Initializers

Shape ops produce INT64 tensors. These are constant-folded at compile time and must be filtered out before uploading weights to the backend. The runtime only uploads F32/F16 initializers that are actually consumed by ops.

## File References

- `crates/dragonwing-onnx/src/proto.rs:1-250` — Hand-rolled protobuf reader
- `crates/dragonwing-onnx/src/model.rs:1-300` — ONNX model parsing
- `crates/dragonwing-onnx/src/builder.rs:250-1400` — Op builder implementations
- `crates/dragonwing-onnx/src/graph.rs:1-750` — Two-phase compilation, NCHW→NHWC conversion
- `crates/dragonwing-onnx/src/runtime.rs:730-1500` — `CpuGraphRuntime` implementation
