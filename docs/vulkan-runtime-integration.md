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

## On-device validation

Measured on **Arduino UNO Q** (Qualcomm QRB2210, Adreno A702 GPU,
Turnip / Mesa 25.2.6 driver):

| Test | Result |
|------|--------|
| 9 host/device parity tests | **9/9 pass** |
| F32 SiLU fusion correctness | **bit-exact** (`max |unfused - fused| = 0.0`) |
| F32 SiLU fusion speedup | **1.8–1.9×** (1 K → 256 K elements) |
| INT8 Conv2D fusion speedup | **1.6–1.7×** (3 shapes tested) |
| INT8 GEMM vs F32 GEMM | INT8 is ~2× *slower* in this naive shader (unpacking overhead) — fused kernels recover the gap |

See `.task/implementation-task-007.md` for the full benchmark table.

### Known issue fixed during validation

**Descriptor pool exhaustion** (`ERROR_OUT_OF_POOL_MEMORY`) after ~64
dispatches. Every op wrapper was allocating descriptor sets without
ever freeing them. Fixed in commit `7bfb3bb` by:

1. Making `OneShot` own the descriptor set and call
   `free_descriptor_set` on drop (after `device_wait_idle`).
2. Bumping the pool from 64 → 512 sets for headroom.

This bug was invisible to the unit tests (each test creates a fresh
pool with only a handful of dispatches) and only surfaced under
benchmark load.

## Future work (Phase 6 deferred items)

See `.task/implementation-task-007.md` for the full list. The
highest-priority items are:

1. Slab-allocator wiring (`VulkanBackend::alloc → SlabAllocator`).
2. Single command buffer per `run()` with explicit barriers.
3. Graph-rewrite pass that emits the fused conv op-type.
4. Bias support for Conv / Gemm (pre-fuse into weights or new shader).
5. Pre-transpose Gemm weights at model load (current shader is
   non-transposed-only).
6. Workgroup-size tuning for INT8 GEMM and Conv.

## Task 008 changes

Task 008 delivered the items above as **Phases 1–5**, plus a partial
end-to-end YOLOv8 benchmark in Phase 6. Concretely:

### Phase 1 — Fused INT8 Conv+Requant+ReLU graph rewrite

Added `OpParams::Conv2dRequantReluI8Nhwc` and a fusion pass in
`crates/dragonwing-onnx/src/fusion.rs` that detects the
`Conv → Requantize → (Relu?)` triple emitted by
`QuantizedGraphCompiler` and rewrites it to a single op routed to the
existing `conv2d_requant_relu_i8_packed` shader (from task 007 Phase 4).
The fusion also sidesteps an unfused-path bug where the I32 accumulator
tensor `"{output}_i32"` synthesised by the quantiser was never added to
`graph.shapes`, so the runtime had no buffer for it.

Public API:

```rust
pub enum OpParams {
    // ...
    Conv2dRequantReluI8Nhwc {
        kernel_shape: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        dilations: [usize; 2],
        group: usize,
        requant_scale: f32,
        has_relu: bool,
    },
}
```

### Phase 2 — F32 bias support via broadcasting shader

Added `bias_add_f32_nhwc.comp`: a 2-SSBO shader that does
`y[i] += bias[i % c_out]` in-place. `VulkanGraphRuntime::dispatch_conv2d`
and `dispatch_gemm` now insert a post-op bias-add when the op carries
a third (bias) input. INT8 bias-folding is deferred to a follow-up task;
the fused INT8 shader has no bias slot and YOLO-style BN-folded weights
usually zero out the explicit conv bias anyway.

### Phase 3 — Gemm `trans_b` fold at load time

`fold_gemm_transpose(&mut Graph) -> Result<()>` (in `graph.rs`)
detects Gemm ops with `trans_b=true`, transposes the F32/F16 weight
initializer bytes from `[N, K]` to `[K, N]`, updates the shape entry,
and clears `trans_b`. Lets the existing non-transposed `gemm_f32`
shader handle all ONNX Gemm layouts without runtime cost. Errors out
cleanly when `trans_b=true` but the weight is dynamic (not in
initializers).

### Phase 4 — Single-command-buffer execution + barriers

Introduced `dragonwing_vulkan::OpsRecorder`
(`crates/dragonwing-vulkan/src/record.rs`): owns one shared
`VkCommandBuffer` and exposes `record_*` methods mirroring every
`pub fn op_name(...)` in `ops::*`. `VulkanGraphRuntime::run()` now
records the entire graph into one recorder, inserts a global
compute→compute `vkCmdPipelineBarrier` between any two ops where the
later op reads a buffer the earlier op wrote, and submits once.
Falls back to per-op execution for unsupported ops (CPU fallback) by
flushing the recorder, running CPU, then starting a new recorder. The
legacy per-op path remains as `VulkanGraphRuntime::run_unbatched()`.

Barrier predicate:

```rust
fn needs_barrier(ops: &[CompiledOp], idx: usize) -> bool {
    let prev = &ops[idx - 1];
    let cur = &ops[idx];
    cur.inputs.iter().any(|inp| prev.outputs.contains(inp))
}
```

Conservative (catches all topologically-ordered direct dependencies).

### Phase 5 — Slab allocator backs every tensor buffer

`VulkanGraphRuntime` now holds an `Arc<SlabAllocator>` and allocates
every tensor's `VulkanBuffer` from it via the new
`VulkanBackend::alloc_slab_buffer`. Drop order matters: `buffers`
declared before `slab` so buffers free their `VkBuffer` handles first
(memory is slab-owned). For YOLOv8m (~110 tensors totalling 811 MiB)
this collapses 110 `vkAllocateMemory` calls to **exactly 1**. Set
`DRAGONWING_DISABLE_SLAB=1` to fall back to per-buffer allocation for
debugging.

New constructor `VulkanBuffer::from_slab(ctx, size, slab_chunk_size,
slab_memory, offset, mapped_ptr)`. Carefully separates the
**logical** `size` (what `len_bytes()` reports — used by shape checks)
from the **slab chunk** size (rounded up to slab's 256 B alignment —
used only for the bind-time mem-reqs check).

### Phase 6 — End-to-end YOLOv8 pipeline (partial)

New binaries:

- `dragonwing-test::yolo-inspect <model.onnx>` — host-side ONNX
  inspector (no Vulkan required).
- `dragonwing-test::yolo-benchmark <model.onnx>` — full pipeline:
  `load_model → fold_batchnorm → compile_model → convert_nchw_to_nhwc
  → fold_gemm_transpose → apply_fusion_passes → VulkanGraphRuntime`.
  Env vars: `WARMUP`, `ITERS`, `MAX_OPS`, `UNBATCHED`,
  `DRAGONWING_DISABLE_SLAB`, `DRAGONWING_SKIP_BIAS`, `DRAGONWING_TRACE`.

CPU fallbacks added in `vulkan_runtime.rs` for the ops that don't yet
have Vulkan shaders but appear in YOLOv8m: **Sub, Div, Concat, Resize,
Split, Transpose, Slice**. The fallback path is `download → CPU op →
upload`; the run-loop flushes the in-flight recorder, runs the CPU op,
then opens a fresh recorder.

### Shader workgroup-count fix (Adreno A702)

The Adreno A702 / Turnip imposes `maxComputeWorkGroupCount[0] = 65535`.
Several elementwise shaders (`sigmoid_f32`, `mul_f32`, `silu_f32`,
`add_f32`, `relu_f32`, `bias_add_f32_nhwc`) used a strict 1D dispatch
of `(ceil(n/64), 1, 1)`. For YOLOv8m's first conv output (4.9M
elements ≈ 76,800 workgroups) this exceeded the limit and produced a
GPU **TRANSLATION fault** logged in `dmesg`. Fixed by:

1. Extending all six shaders to 2D-wrap dispatch:

   ```glsl
   uint flat_wg = gl_WorkGroupID.y * pc.gx_total + gl_WorkGroupID.x;
   uint gid = flat_wg * 64u + gl_LocalInvocationID.x;
   if (gid >= pc.n) return;
   ```

2. Adding a `gx_total: u32` push-constant slot (16-byte total unchanged).

3. Updating both `ops::*` and `OpsRecorder::record_*` host wrappers:

   ```rust
   let total_groups = (n as u64).div_ceil(64);
   let gx = total_groups.min(65535) as u32;
   let gy = total_groups.div_ceil(gx as u64) as u32;
   ```

### Phase 6 blocker

The naive `conv2d_f32_nhwc` shader is too slow on mid-network YOLOv8m
shapes (e.g. `c_in=48, c_out=96, h_out=160, w_out=160`: ≈ 1 GigaFLOP
per dispatch with no tiling / shared-memory reuse). MSM's 500 ms GPU
**hangcheck** trips, surfacing as a recovery-induced TRANSLATION fault
in `dmesg`. This isn't a correctness bug — the math is right — but the
F32 path was never the production target. The INT8 fused conv path
(Phase 1 of this task + Phase 4 of task 007) handles the same shape
in ~30–60 ms on Adreno A702 according to task 007's micro-benchmarks
and is fully wired; it just needs calibration data to drive the
`QuantizedGraphCompiler` for the user's YOLOv8m models.


