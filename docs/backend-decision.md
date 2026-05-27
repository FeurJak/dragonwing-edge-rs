# Backend Decision — based on `artifacts/probes/uno-q-2026-05-27.json`

This document records the recommendation produced at the end of
implementation task 001. The four candidate paths are taken verbatim from
`.task/implementation-task-001.md` §Phase 6.

## Evidence summary

Pulled directly from the committed probe artifact:

| Capability path     | Probe field                          | Status on UNO Q |
| ------------------- | ------------------------------------ | --------------- |
| Vulkan / Turnip     | `gpu.vulkan.devices[0].driver_name`  | `"turnip Mesa driver"` (Mesa 25.2.6, API 1.0.318) — **available** |
| Vulkan device type  | `gpu.vulkan.devices[0].device_type`  | `PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU` — confirmed Adreno A702 |
| OpenCL / Rusticl    | `gpu.opencl.platforms[0].name`       | `"rusticl"`, OpenCL 3.0, device `FD702` — **available** |
| Direct KGSL ioctl   | `gpu.kgsl`                           | `null` — **unavailable** (no `/dev/kgsl-3d0`) |
| DRM render node     | `gpu.drm_render_node`                | `/dev/dri/renderD128` — **available** (user is in `render` group) |
| Hexagon DSP         | `dsp.present`                        | `false` — no FastRPC nodes |
| CPU NEON            | `cpu.features`                       | contains `asimd` — **available** on all 4 cores |
| Half-precision      | `cpu.features`                       | does **not** contain `asimdhp` (no ARMv8.2 FP16 ASIMD on this part) |

## Path analysis

### Path A — Fork Edge-Impulse SDK, reimplement in Rust, layer Adreno opts

- **Pros:** Edge-Impulse is the path Arduino already supports for the UNO Q,
  so model authoring tooling (`.eim` files, training UI) maps cleanly. Lots
  of pre-trained YOLO models already published in EI format.
- **Cons:** The C++ SDK is large (~50k LOC) and a faithful Rust port is a
  multi-month effort with high churn risk. The "layer TinyGrad/Comma.ai
  optimisations on top" step is blocked because **the KGSL ioctl interface
  that TinyGrad's QCOM compiler emits to does not exist on this image**
  (`gpu.kgsl == null`). We would have to additionally re-target those
  optimisations onto Vulkan or Mesa, which negates the value of forking
  Edge-Impulse to begin with.
- **Verdict:** Highest cost, lowest leverage given the kernel image. Park.

### Path B — TinyGrad-style direct Adreno backend via KGSL ioctl

- **Pros:** Theoretically minimum overhead — TinyGrad demonstrates that
  hand-emitted shader binaries through KGSL outperform OpenCL/Vulkan on
  comparable Adreno parts.
- **Cons:** **Blocked by the kernel.** `gpu.kgsl == null` and the DRM
  driver is `msm_dpu` (upstream). Making this path work requires either
  rebuilding the kernel with the downstream KGSL driver, or porting
  TinyGrad's QCOM emitter to talk to the upstream `msm` DRM ioctl
  (`DRM_IOCTL_MSM_GEM_*`, `DRM_IOCTL_MSM_SUBMITQUEUE_*`) — a non-trivial
  re-targeting effort. Either option is out of scope for an initial
  framework drop.
- **Verdict:** Compelling long-term, but premature without kernel changes.
  Park.

### Path C — Vulkan-compute backend via Mesa/Turnip

- **Pros:** **Works out of the box on the stock image.** Turnip enumerates
  the A702 as an integrated GPU with Vulkan 1.0.318 + the standard compute
  pipeline. SPIR-V compute shaders are a portable, well-documented target
  and there are mature Rust crates (`ash`, `vulkano`, `wgpu`) we can use or
  draw on. Mesa Turnip is actively developed (Mesa 25.2.6 is current) and
  the same code will work on every future kernel that keeps the upstream
  `msm` driver. The conformance report shipped in the ICD
  (`conformanceVersion = 1.2.7.1`) means we can rely on standard behaviour.
- **Cons:** Adreno A702 is the entry-tier Adreno 7-series part — fewer SPs
  than A6xx parts. Half-precision (FP16) compute is supported by the GPU
  (Adreno 700-series has FP16 ALUs) but we will need explicit SPIR-V FP16
  extensions; the CPU `asimd` does **not** include `asimdhp`, so FP16
  fallback to NEON is not free.
- **Verdict:** **This is the path that exists today on real hardware.**

### Path D — CPU/NEON-only baseline first, GPU later

- **Pros:** Lowest risk, smallest surface, easy to make `no_std`. Useful as
  a correctness oracle for whichever GPU path we add later.
- **Cons:** A 4× 2.0 GHz A53 cluster will not run a modern YOLO model at
  useful frame rates. The entire point of `dragonwing-edge` (per
  `project.md`) is GPU-accelerated edge inference; a CPU-only baseline
  doesn't satisfy the success criterion of integrating with `cortex-edge-rs`
  for the `alcoa_excavator` / `crusher_ai` applications.
- **Verdict:** Not sufficient as the primary path, but valuable as a
  **second-priority** fallback that we ship alongside C for parity testing.

## Recommendation

> **Adopt Path C (Vulkan compute via Mesa/Turnip) as the primary backend,
> with Path D (CPU/NEON) as a co-shipped reference backend used for
> correctness checks and as a fallback when no GPU is reachable.**

Rationale in one paragraph: Turnip is **already installed**, the Adreno A702
**enumerates successfully** as a Vulkan integrated GPU, the user has render
permission, and SPIR-V compute is the only path that doesn't require either a
kernel rebuild (Path B) or a multi-month C++→Rust port that depends on the
same kernel rebuild (Path A). Path D alone cannot meet the inference
throughput requirement implied by `project.md`'s YOLO integration goal but is
indispensable as a deterministic reference.

OpenCL via Rusticl is also available and **kept as a future option**, but it
is not the recommended primary path because Rusticl on this device reports
only one OpenCL compute unit (an artefact of Rusticl's current device-model
abstraction over the A702), whereas Vulkan exposes the full physical-device
properties and we have direct control over workgroup sizing.

## Implications for task 002

1. Add a `dragonwing-vulkan` crate that wraps Mesa/Turnip via `ash` (chosen
   for being the thinnest `libvulkan` binding in the Rust ecosystem; revisit
   if `vulkano`'s abstractions prove worth the dep weight).
2. Add a `dragonwing-cpu` crate using `std::arch::aarch64` NEON intrinsics
   for the baseline ops kernel.
3. Implement `Backend` for both, dispatched at runtime based on the same
   `HardwareCapabilities` snapshot produced by task 001.
4. **Do not** add Edge-Impulse SDK bindings to the core workspace; if a user
   wants `.eim` model loading, that lives in a separate optional crate.
5. **Do not** start KGSL-direct work until either we have a downstream
   kernel on the UNO Q or we have a working `msm`-ioctl prototype — either
   is its own task.

## Re-evaluation triggers

This recommendation should be revisited if any of the following change:

- The Arduino-shipped image switches to a downstream Qualcomm kernel that
  exposes `/dev/kgsl-3d0`. (Re-run the probe; if `gpu.kgsl != null`, Path B
  becomes viable.)
- Mesa Turnip support for the A702 regresses or is dropped.
- An Edge-Impulse-Rust crate appears that wraps the C++ SDK at acceptable
  binary cost and license.
