# CPU Multi-Threading

This document describes the thread-pool design and multi-threading strategy
introduced in task 003 (Phase 6).

## Goal

Use the four Cortex-A53 cores of the QRB2210 effectively for the compute-heavy
ops where parallelism actually pays off: **GEMM** and **Conv2D**. Element-wise
ops (`fill`, `axpy`, `relu`, `add`) stay single-threaded; their arithmetic
intensity is too low to justify pool overhead.

## Design choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pool type | Persistent, owned by `CpuBackend` | One spawn cost amortised over the program lifetime. Per-op pool would be a footgun. |
| Pool size | `num_cpus() - 1`, capped to ≥1 | Leave one core for the OS / scheduler so we don't stall on cross-core IPIs. |
| Queue primitive | `Mutex<VecDeque>` + `Condvar` | Hand-rolled; no `crossbeam` or `rayon` dep. Re-evaluate if profiling shows the queue is a bottleneck. |
| Job type | `Box<dyn FnOnce() + Send + 'static>` | Standard pattern. The boxed allocation per job is acceptable for our coarse-grained partitions. |
| Shutdown | Sentinel `Message::Shutdown` per worker; drop-time join with timeout | Avoids the classic "channel disconnect causes worker panic" pattern. |
| Worker pinning (`sched_setaffinity`) | **Out of scope** | A53s are homogeneous so this is unlikely to help; revisit only if profiling shows core migration is hurting cache hits. |

The pool lives in `crates/dragonwing-cpu/src/pool.rs`.

## API surface

```rust
pub struct ThreadPool { /* ... */ }

impl ThreadPool {
    pub fn new(num_workers: usize) -> Self;
    pub fn execute<F: FnOnce() + Send + 'static>(&self, job: F);
    pub fn join(&self);  // Block until every submitted job has finished
}

/// Reads /sys/devices/system/cpu/online on Linux; falls back to
/// std::thread::available_parallelism() everywhere else.
pub fn num_cpus() -> usize;

/// Partition `total` work across `num_chunks` jobs. If `pool` is None,
/// runs the work serially in the caller's thread.
pub fn parallel_for<F>(pool: Option<&ThreadPool>, total: usize, num_chunks: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync;

/// Scoped variant for borrow-checker friendliness (uses std::thread::scope).
pub fn parallel_for_scoped<F>(num_threads: usize, total: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync;
```

`parallel_for` is the workhorse: the multi-threaded ops compute a chunk
count, hand it the work range, and the closure receives `(start, end)`.

## Multi-threaded ops

The following CPU ops have `_mt` variants that take an optional
`&ThreadPool`:

| Op | Partition dimension | Why this dimension |
|----|---------------------|--------------------|
| `gemm_f32_mt`, `gemm_fp16_mt` | `M` rows of the output | C[i, j] for different `i` are independent; no synchronisation needed. |
| `conv2d_f32_nhwc_mt`, `conv2d_fp16_nhwc_mt` | `N * H_out` (batch × output rows) | Same independence property; chunked at the row level so each worker handles whole rows. |

Pool ops, softmax, and element-wise ops are deliberately **not** multi-threaded
in this task. Their memory bandwidth dominates compute, so adding cores
doesn't help on a memory-bound CPU.

## Why partition `M` for GEMM and `H_out` for conv

Both choices follow the same principle: **partition the dimension whose
elements are independent and whose chunks fit in L1**.

- For GEMM (`C[m, n] = sum_k A[m, k] * B[k, n]`), each output row reads its
  own row of `A` plus the **same** `B`. Workers can share the `B` matrix
  without coordination, and each worker has a private `A` strip.
- For Conv2D in NHWC, each output row reads a band of input rows
  (`stride_h * H_out_chunk + K_h`). Workers share the kernel and read
  overlapping bands of input; the read-only sharing pattern is cache-
  friendly.

Partitioning `N` (output columns of GEMM, output channels of conv) would
require workers to write to **adjacent** memory addresses, which can cause
false sharing on the 64-byte cache lines of the A53. Row partitioning
avoids this.

## Default behaviour and configurability

The `CpuBackend` exposes the pool as `Arc<ThreadPool>`:

```rust
let cpu = CpuBackend::new();              // Uses num_cpus() - 1 workers
let cpu = CpuBackend::with_threads(2);    // Explicit override
```

Callers that want single-threaded behaviour pass `None` to the `_mt` op or
use the non-`_mt` variant directly.

## Expected scaling

Cortex-A53 cluster on QRB2210: 4 cores @ 2.0 GHz. With one core reserved for
the OS, the practical parallelism ceiling is 3×.

| Op | Single-thread | 3-thread | Speedup |
|----|---------------|----------|---------|
| `gemm_f32` (256, 256, 256) | (baseline) | (≥ 2.5×) | Acceptance criterion |
| `conv2d_f32_nhwc` (3×3, 14×14, 64→128) | (baseline) | (≥ 2.5×) | Acceptance criterion |

The 2.5× target (not 3×) accounts for:

- Single-core OS overhead and timer interrupts hitting our 4th core.
- Cache contention on the shared L2 (256 KiB on Cortex-A53; small workloads
  fit entirely, large workloads spill to LPDDR).
- Pool queue contention (Mutex acquisition per chunk).

Numbers in the actual bench run go to `artifacts/benches/conv-<date>.md`.

## When NOT to multi-thread

`parallel_for` with `pool == None` reduces to a plain serial loop. Use this
path when:

- The total work is small (≤ ~10k arithmetic ops): pool dispatch overhead
  exceeds the compute saved.
- The op is memory-bound and the bottleneck is already LPDDR bandwidth (one
  core already saturates the bus).
- The caller is already inside a worker thread (nested parallelism is a
  deadlock waiting to happen with our blocking-queue design).

The conv and GEMM heuristics use a simple problem-size threshold below which
the serial path runs. Tuning this threshold is a task-004 concern.

## Determinism

The pool does **not** guarantee deterministic floating-point output across
runs. Even though each worker processes a fixed chunk, the order in which
the main thread merges results back is non-deterministic in `parallel_for`'s
fan-in. The parity harness tolerates this with the same `1e-4` band as the
single-threaded path — accumulation order within a chunk is fixed, only
inter-chunk merging varies.

If a downstream test needs bitwise determinism, use the single-threaded
variant.

## Known limitations

- No work stealing. Each worker takes one chunk; if work is unevenly sized
  (it isn't for GEMM/conv, but might be for future ops) the stragglers
  determine the wall-clock latency.
- No NUMA / core-pinning awareness. Not relevant for the homogeneous A53
  cluster, but would matter on big.LITTLE.
- No interaction with the Vulkan backend. The Vulkan queue is its own
  scheduler; mixing CPU thread-pool work and Vulkan dispatches on the same
  process is fine but uncoordinated.

## Local references

- `crates/dragonwing-cpu/src/pool.rs` — `ThreadPool`, `parallel_for`,
  `num_cpus`.
- `crates/dragonwing-cpu/src/ops.rs::gemm_f32_mt` — M-partitioned GEMM.
- `crates/dragonwing-cpu/src/ops.rs::conv2d_f32_nhwc_mt` — H-partitioned conv.
- `crates/dragonwing-cpu/src/lib.rs` — `CpuBackend` exposing the pool.
- `crates/dragonwing-cpu/src/ops.rs::tests::gemm_f32_mt_matches_naive` —
  parity test ensuring the MT path matches the single-threaded reference.
