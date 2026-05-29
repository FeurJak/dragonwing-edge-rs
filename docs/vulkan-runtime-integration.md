# Vulkan Runtime Integration (Task 007)

This document describes `VulkanGraphRuntime` — the glue between the
compiled ONNX `Graph` representation and the low-level Vulkan compute
ops in `dragonwing-vulkan`. It complements `docs/vulkan-runtime.md`
(architecture overview) and `docs/int8-vulkan.md` (UINT32 packing
reference).

## Where it lives

```
crates/dragonwing-onnx/
    Cargo.toml          # adds [features] vulkan = ["dep:dragonwing-vulkan"]
    src/
        vulkan_runtime.rs   # VulkanGraphRuntime
```

The runtime sits in the **onnx** crate rather than the **vulkan**
crate because:

- `Graph`, `CompiledOp`, and `OpParams` are defined in onnx.
- Without an onnx dependency, the vulkan crate stays a thin
  wrapper around `ash` (matching the design goal in
  `docs/vulkan-backend.md`).
- This mirrors `CpuGraphRuntime` (also onnx, behind feature `cpu`).

Enable with `--features vulkan`:

```sh
cargo build -p dragonwing-onnx --features vulkan
```

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    VulkanGraphRuntime                          │
│                                                                │
│   graph: Graph                  (compiled ONNX from task 004)  │
│   backend: VulkanBackend        (one device, one pipeline      │
│                                  cache; cheap to clone)        │
│   buffers: HashMap<            (one VulkanBuffer per tensor,   │
│       String, VulkanBuffer>     allocated up-front in new())   │
│                                                                │
│   set_input_f32 / set_input_bytes                              │
│   run()  ──► for op in graph.ops { dispatch_op(op); }          │
│              synchronize()                                     │
│   get_output_f32 / get_output_bytes                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   dispatch_op() matches OpParams → calls dragonwing_vulkan::ops::*
   ┌─────────────┬─────────────────────────────────────────────┐
   │  OpParams   │  shader / wrapper                            │
   ├─────────────┼─────────────────────────────────────────────┤
   │  Add        │  add_f32                                     │
   │  Mul        │  mul_f32                                     │
   │  Sigmoid    │  sigmoid_f32                                 │
   │  Gemm       │  gemm_f32 (non-transposed, no bias)          │
   │  Conv2d     │  conv2d_f32_nhwc (group=1, no bias)          │
   │  MaxPool    │  maxpool2d_f32                               │
   │  Softmax    │  softmax_f32                                 │
   │  Clip(0,∞)  │  relu_f32                                    │
   │  Quantize   │  quantize_f32_to_i8_packed                   │
   │  Dequantize │  dequantize_i8_packed_to_f32                 │
   │  Requantize │  requantize_i32_to_i8_packed                 │
   │  AddQuant   │  add_i8_packed                               │
   │  None+"Relu"│  relu_f32 (in-place or out-of-place)         │
   │  None+"SiLU"│  silu_f32 (fused: sigmoid + mul)             │
   └─────────────┴─────────────────────────────────────────────┘
```

## Buffer lifecycle

```
new(graph, backend)
  ├── for (name, shape) in graph.shapes:
  │       buf = backend.alloc(shape.size_bytes(), Storage)
  │       buffers.insert(name, buf)
  └── for (name, data) in graph.initializers:
          if buffers.contains(name) && len matches:
              backend.upload(buf, data)
```

All tensors get a Vulkan buffer at construction. Initializers
(weights, biases) are uploaded once. The runtime never reallocates
buffers between `run()` calls — only inputs are re-uploaded.

**Memory pressure note (unfixed).** Each `backend.alloc` calls
`vkAllocateMemory`. Adreno A702's allocation count is bounded
(typically 4096); YOLOv8n has ~200 tensors, well within the limit but
each allocation has its own latency. Switching to
`dragonwing_vulkan::SlabAllocator` (already implemented in task 005)
will reduce both allocation count and overhead. Deferred to task 008.

## Synchronisation model

Phase 1–4: **per-op submit + wait**. Every `dragonwing_vulkan::ops::*`
wrapper allocates a one-shot command pool, records one dispatch,
submits it with a timeline-semaphore signal, then *waits for the
queue to go idle on drop*. `run()` calls `backend.synchronize()` at
the end so `get_output_*` reads coherent data.

Phase 6 (deferred): switch to **one command buffer per `run()`** with
`vkCmdPipelineBarrier(SHADER_WRITE → SHADER_READ)` between dependent
ops. This eliminates the per-op `device_wait_idle()` cost. The bench
binary in `crates/dragonwing-test/src/bin/device_benchmark.rs` will
quantify whether the current per-op pattern is the bottleneck before
we make the change.

## Borrow-checker pattern

Op wrappers in `dragonwing_vulkan::ops::*` take `&VulkanBackend`
plus per-buffer `&` or `&mut` references. Calling these from a
method on `VulkanGraphRuntime` that also holds `&mut
self.buffers` runs into the standard split-borrow problem.

Solution: the helpers `get_two_buffers` / `get_three_buffers` are
**associated functions**, not methods:

```rust
fn get_three_buffers<'a>(
    buffers: &'a mut HashMap<String, VulkanBuffer>,
    a_name: &str,
    b_name: &str,
    c_name: &str,
) -> Result<(&'a VulkanBuffer, &'a VulkanBuffer, &'a mut VulkanBuffer)>
```

Call sites do `Self::get_three_buffers(&mut self.buffers, …)` instead
of `self.get_three_buffers(…)`. This leaves `&self.backend` free for
the inner op call.

Safety: the helpers verify the three names are distinct, then
construct disjoint raw pointers from `HashMap::get` / `get_mut`.
Sound because `HashMap::get*` does not invalidate other entries.

## OpParams dispatch table

| `OpParams` variant | Vulkan op | Status | Notes |
|--------------------|-----------|--------|-------|
| `None` + `op_type="Relu"` | `relu_f32` | ✅ | in-place if input=output |
| `None` + `op_type="SiLU"` | `silu_f32` | ✅ | fused sigmoid+mul |
| `None` + `op_type="Reshape"/"Flatten"` | (memcpy) | ✅ | metadata-only; uses download+upload (Phase 6 will switch to `vkCmdCopyBuffer`) |
| `Reshape` | (memcpy) | ✅ | same as above |
| `Add` | `add_f32` | ✅ | equal-shape only |
| `Mul` | `mul_f32` | ✅ | equal-shape only |
| `Sigmoid` | `sigmoid_f32` | ✅ | not in-place |
| `Clip {0, ∞}` | `relu_f32` | ✅ | ReLU pattern |
| `Clip {min, max}` | — | ❌ | general clip needs new shader |
| `Gemm` | `gemm_f32` | ⚠ | non-transposed, no bias |
| `Conv2d` | `conv2d_f32_nhwc` | ⚠ | group=1, no bias |
| `MaxPool` | `maxpool2d_f32` | ✅ | NHWC |
| `Softmax` | `softmax_f32` | ✅ | last-axis |
| `Quantize {scale}` | `quantize_f32_to_i8_packed` | ✅ | n must be %4 |
| `Dequantize {scale}` | `dequantize_i8_packed_to_f32` | ✅ | n must be %4 |
| `Requantize {scale}` | `requantize_i32_to_i8_packed` | ✅ | n must be %4 |
| `AddQuantized` | `add_i8_packed` | ✅ | scale_a, scale_b |
| `Sub`, `Div`, `Concat`, `Resize`, `Split`, `Transpose`, `Slice`, `GlobalAvgPool`, `AvgPool` | — | ❌ | error; use CpuGraphRuntime |

**Legend:** ✅ wired and tested · ⚠ wired with restrictions · ❌ errors out

## INT8 helpers and constraints

INT8 ops use UINT32 packing along the channel/inner dimension. The
constraint is **the element count must be a multiple of 4**. The
runtime validates this at dispatch time and returns
`Error::Backend("…: n=… must be multiple of 4")` if violated.

Sizes for INT8 tensors:
- Packed I8 buffer length (bytes) = element count (1 byte/element).
- INT32 accumulator buffer length = element count × 4 bytes.

Two helpers on the runtime simplify INT8 testing:

```rust
pub fn set_input_bytes(&mut self, name: &str, data: &[u8]) -> Result<()>
pub fn get_output_bytes(&self, name: &str) -> Result<Vec<u8>>
```

These bypass dtype conversion. They are useful for tests that feed
packed-UINT32 buffers or INT32 accumulator buffers directly.

## Fused kernels (Phase 4)

Two new shaders, callable via `dragonwing_vulkan::ops::*`:

1. `silu_f32` — fused sigmoid + mul. Routed from
   `OpParams::None` + `op_type="SiLU"` (the op-type emitted by the
   existing fusion pass in `onnx::fusion`).

2. `conv2d_requant_relu_i8_packed` — fused INT8 conv + requantize +
   optional ReLU. Output is packed UINT32, so `c_out` must be a
   multiple of 4. **Not yet routed from `OpParams`** — there is no
   `OpParams::Conv2dRequantReluI8` variant. Callable from low-level
   code only. Wiring it through requires a graph-rewrite pass in
   `onnx::quantize` or `onnx::fusion`; deferred to a follow-up task.

## Testing

Host tests use `VulkanBackend::new(VulkanConfig::default())` and
**skip gracefully** when no Vulkan device is available (returns
`None`, test prints `skipping: no Vulkan device available` and
returns). On the device (Adreno A702) they all pass.

| Test | Validates |
|------|-----------|
| `empty_graph_constructs_and_runs` | new() + run() + get_output_f32 path |
| `relu_inplace_matches_cpu` | in-place ReLU |
| `add_matches_cpu` | out-of-place 3-buffer F32 add |
| `silu_two_op_graph_matches_cpu` | sigmoid + mul chain |
| `fused_silu_matches_reference` | fused silu_f32 |
| `gemm_matches_reference` | 2×3 × 3×2 known-answer GEMM |
| `quantize_roundtrip_via_dequantize` | F32 → packed I8 → F32 |
| `requantize_clamps_and_scales` | INT32 → packed I8, with clamping |
| `relu_i8_clamps_negatives` | packed I8 ReLU |

On-device timings live in
`crates/dragonwing-test/src/bin/device_benchmark.rs` — see Task 007
findings for results when the device is reconnected.

## Building on the Arduino UNO Q

Cross-compiling with the Vulkan loader linked in is fragile (the
loader paths differ from the host); we recommend **building on
device**:

```sh
# On host:
cd ~/dragonwing-edge-rs
tar czf /tmp/dw-src.tgz \
    --exclude='target' --exclude='.git' --exclude='artifacts/models' \
    Cargo.toml Cargo.lock crates docs README.md \
    rust-toolchain.toml LICENSE
adb push /tmp/dw-src.tgz /home/arduino/

# On device:
cd ~ && rm -rf dragonwing-edge-rs && mkdir dragonwing-edge-rs && \
    cd dragonwing-edge-rs && tar xzf ~/dw-src.tgz
# Patch the toolchain file (remove musl target — not used on device):
cat > rust-toolchain.toml <<EOF
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
targets = ["aarch64-unknown-linux-gnu"]
profile = "minimal"
EOF
# Remove cross-compile linker config (only relevant on host):
rm -f .cargo/config.toml

PATH=~/.cargo/bin:$PATH cargo build -p dragonwing-test \
    --bin device-benchmark --release
./target/release/device-benchmark
```

Initial build takes ~1m42s on the Arduino UNO Q. Subsequent
incremental builds are seconds.

## Future work (Phase 6 deferred items)

See `.task/implementation-task-007.md` for the full list. The
highest-priority items are:

1. Slab-allocator wiring (`VulkanBackend::alloc → SlabAllocator`).
2. Single command buffer per `run()` with explicit barriers.
3. Graph-rewrite pass that emits the fused conv op-type.
4. Workgroup-size tuning for INT8 GEMM and Conv.
