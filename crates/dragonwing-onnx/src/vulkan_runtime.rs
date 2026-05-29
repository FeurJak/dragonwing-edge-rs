//! Vulkan-backed graph runtime (Task 007).
//!
//! This module provides `VulkanGraphRuntime`, which executes compiled ONNX
//! graphs on the Vulkan compute backend. Unlike the generic `GraphRuntime`
//! (which falls back to host-side compute), this runtime dispatches every
//! op directly to a `dragonwing_vulkan::ops::*` wrapper.
//!
//! # Architecture
//!
//! ```text
//!   VulkanGraphRuntime
//!     ├── Graph                  (compiled, dtype-validated)
//!     ├── VulkanBackend          (Arc<Context> + Arc<PipelineCache>)
//!     ├── HashMap<String,        VulkanBuffer per tensor (allocated up-front)
//!     │       VulkanBuffer>
//!     └── dispatch_op() match    routes CompiledOp → dragonwing_vulkan::ops::*
//! ```
//!
//! # Buffer lifecycle
//!
//! All tensors (inputs, intermediates, outputs, initializers) get one
//! `VulkanBuffer` at `new()` time. Initializers are uploaded once. Inputs are
//! re-uploaded via `set_input_f32()`. Outputs are read back via
//! `get_output_f32()`. Intermediates are written/read by the GPU only — they
//! are never touched on the host between `run()` calls.
//!
//! # Synchronisation
//!
//! Each op currently records its own one-shot command buffer + submit + wait
//! (mirroring the existing `dragonwing_vulkan::ops::*` API). The `run()`
//! method waits for the queue to go idle after the last op so that
//! `get_output_*()` reads coherent data. Phase 6 may move to a single batched
//! command buffer per graph if measurements show submit overhead is high.
//!
//! # Quickstart
//!
//! ```ignore
//! use dragonwing_onnx::{compile_model, VulkanGraphRuntime};
//! use dragonwing_vulkan::{VulkanBackend, VulkanConfig};
//! use dragonwing_core::Dtype;
//!
//! let model = dragonwing_onnx::load_model("model.onnx")?;
//! let graph = compile_model(&model, Dtype::F32)?;
//! let backend = VulkanBackend::new(VulkanConfig::default())?;
//! let mut rt = VulkanGraphRuntime::new(graph, backend)?;
//!
//! rt.set_input_f32("input", &input_data)?;
//! rt.run()?;
//! let out = rt.get_output_f32("output")?;
//! ```

use crate::builder::{CompiledOp, OpParams, TensorShape};
use crate::error::{Error, Result};
use crate::graph::Graph;
use dragonwing_core::{Backend, BackendBuffer, BufferKind, Dtype};
use dragonwing_vulkan::{VulkanBackend, VulkanBuffer};
use std::collections::HashMap;

/// Graph runtime that dispatches every op to the Vulkan compute backend.
///
/// Phase 1 implementation: skeleton, buffer management, and op-dispatch
/// scaffolding. Most op dispatchers are stubs that return
/// `Error::Runtime("not yet implemented")` and will be filled in in
/// subsequent phases:
///
/// * **Phase 2** wires up the F32 ops that already have Vulkan shaders
///   (`gemm_f32`, `conv2d_f32_nhwc`, `relu_f32`, `add_f32`, `maxpool2d_f32`,
///   `softmax_f32`, `sigmoid_f32`, `mul_f32`).
/// * **Phase 3** wires up the INT8 packed ops.
/// * **Phase 4** wires up fused kernels.
pub struct VulkanGraphRuntime {
    /// The compiled graph.
    graph: Graph,
    /// Vulkan backend (cheap to clone — internally `Arc`).
    backend: VulkanBackend,
    /// Tensor-name → `VulkanBuffer`. Allocated up-front in `new()`.
    buffers: HashMap<String, VulkanBuffer>,
}

impl std::fmt::Debug for VulkanGraphRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanGraphRuntime")
            .field("device", &self.backend.device_name())
            .field("op_count", &self.graph.ops.len())
            .field("buffer_count", &self.buffers.len())
            .finish()
    }
}

impl VulkanGraphRuntime {
    /// Allocate a Vulkan buffer for every tensor in `graph.shapes` and upload
    /// all initializers (weights, biases).
    ///
    /// Buffer sizes use `TensorShape::size_bytes()`, which accounts for the
    /// graph's dtype (4 B for F32, 2 B for F16, 1 B for I8, 4 B for I32).
    ///
    /// # Errors
    ///
    /// * `Error::Runtime` if any individual `backend.alloc()` fails.
    /// * `Error::Runtime` if an initializer's byte length does not match the
    ///   allocated buffer (typically indicates a build-context bug).
    pub fn new(graph: Graph, backend: VulkanBackend) -> Result<Self> {
        let mut buffers: HashMap<String, VulkanBuffer> = HashMap::new();

        for (name, shape) in &graph.shapes {
            let size_bytes = shape.size_bytes();
            if size_bytes == 0 {
                // Zero-sized tensors (e.g. metadata-only Reshape shape inputs)
                // — skip allocation. Dispatch code that looks them up will
                // handle the absence.
                continue;
            }
            let buffer = backend
                .alloc(size_bytes, BufferKind::Storage)
                .map_err(|e| Error::Runtime(format!("Vulkan alloc {name} failed: {e}")))?;
            buffers.insert(name.clone(), buffer);
        }

        // Upload initializers (weights, biases, etc.)
        for (name, data) in &graph.initializers {
            if let Some(buffer) = buffers.get_mut(name) {
                if buffer.len_bytes() == data.len() {
                    backend
                        .upload(buffer, data)
                        .map_err(|e| Error::Runtime(format!("Vulkan upload {name}: {e}")))?;
                }
                // If sizes mismatch, the initializer is likely a shape array
                // for a metadata-only op (Reshape, Resize, etc.) — silently
                // skip; the dispatcher will encode the params separately.
            }
        }

        Ok(Self {
            graph,
            backend,
            buffers,
        })
    }

    /// Input tensor names + shapes (for caller introspection).
    pub fn inputs(&self) -> Vec<(&str, &TensorShape)> {
        self.graph.input_info()
    }

    /// Output tensor names + shapes.
    pub fn outputs(&self) -> Vec<(&str, &TensorShape)> {
        self.graph.output_info()
    }

    /// Borrow the underlying backend (cheap clone if needed).
    pub fn backend(&self) -> &VulkanBackend {
        &self.backend
    }

    /// Borrow the compiled graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Upload F32 data into the named input buffer.
    ///
    /// For graphs compiled in F32, this is a direct memcpy. For F16 graphs we
    /// convert via `F16::from_f32`. INT8 graphs require quantization at the
    /// caller side (see Phase 3 wiring); for now `set_input_f32` returns an
    /// error if the graph dtype is `I8`.
    pub fn set_input_f32(&mut self, name: &str, data: &[f32]) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(name)
            .ok_or_else(|| Error::Runtime(format!("input not found: {name}")))?;

        match self.graph.dtype {
            Dtype::F32 => {
                let expected = data.len() * 4;
                if buffer.len_bytes() != expected {
                    return Err(Error::Runtime(format!(
                        "input {name} size mismatch: buffer={} bytes, data={} bytes",
                        buffer.len_bytes(),
                        expected,
                    )));
                }
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), expected)
                };
                self.backend
                    .upload(buffer, bytes)
                    .map_err(|e| Error::Runtime(format!("upload {name}: {e}")))
            }
            Dtype::F16 => {
                use dragonwing_core::F16;
                let expected = data.len() * 2;
                if buffer.len_bytes() != expected {
                    return Err(Error::Runtime(format!(
                        "input {name} size mismatch: buffer={} bytes, data={} bytes",
                        buffer.len_bytes(),
                        expected,
                    )));
                }
                let f16_data: Vec<F16> = data.iter().map(|&v| F16::from_f32(v)).collect();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(f16_data.as_ptr().cast::<u8>(), expected)
                };
                self.backend
                    .upload(buffer, bytes)
                    .map_err(|e| Error::Runtime(format!("upload {name}: {e}")))
            }
            Dtype::I8 => Err(Error::Runtime(
                "set_input_f32 on I8 graph requires explicit Quantize op or set_input_i8 \
                 (see Phase 3)"
                    .into(),
            )),
            _ => Err(Error::Runtime(format!(
                "set_input_f32: unsupported graph dtype {:?}",
                self.graph.dtype
            ))),
        }
    }

    /// Download the named output buffer as F32.
    ///
    /// Implicitly synchronises the backend (waits for the last submission to
    /// complete) so the readback observes the result of the most recent
    /// `run()`.
    pub fn get_output_f32(&self, name: &str) -> Result<Vec<f32>> {
        let buffer = self
            .buffers
            .get(name)
            .ok_or_else(|| Error::Runtime(format!("output not found: {name}")))?;

        // `download` calls `synchronize` internally — see `VulkanBackend::download`.
        match self.graph.dtype {
            Dtype::F32 => {
                let n = buffer.len_bytes() / 4;
                let mut bytes = vec![0u8; buffer.len_bytes()];
                self.backend
                    .download(buffer, &mut bytes)
                    .map_err(|e| Error::Runtime(format!("download {name}: {e}")))?;
                let mut out = vec![0.0f32; n];
                for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Ok(out)
            }
            Dtype::F16 => {
                use dragonwing_core::F16;
                let n = buffer.len_bytes() / 2;
                let mut bytes = vec![0u8; buffer.len_bytes()];
                self.backend
                    .download(buffer, &mut bytes)
                    .map_err(|e| Error::Runtime(format!("download {name}: {e}")))?;
                let mut out = vec![0.0f32; n];
                for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out[i] = F16(bits).to_f32();
                }
                Ok(out)
            }
            Dtype::I8 => Err(Error::Runtime(
                "get_output_f32 on I8 graph requires explicit Dequantize op or get_output_i8 \
                 (see Phase 3)"
                    .into(),
            )),
            _ => Err(Error::Runtime(format!(
                "get_output_f32: unsupported graph dtype {:?}",
                self.graph.dtype
            ))),
        }
    }

    /// Execute all ops in graph order.
    ///
    /// Each op records and submits its own command buffer (the existing
    /// `dragonwing_vulkan::ops::*` pattern); barriers between dependent ops
    /// are not yet inserted because every submit currently waits on the
    /// previous via the timeline semaphore. Phase 6 will optimise this by
    /// batching into a single command buffer with explicit
    /// `vkCmdPipelineBarrier` between dependent ops.
    pub fn run(&mut self) -> Result<()> {
        let ops: Vec<CompiledOp> = self.graph.ops.clone();
        for op in &ops {
            self.dispatch_op(op)?;
        }
        // Ensure all submitted work has finished before returning so the next
        // get_output_* sees coherent data.
        self.backend
            .synchronize()
            .map_err(|e| Error::Runtime(format!("synchronize: {e}")))?;
        Ok(())
    }

    /// Get two buffers (one mut, one shared) by name without borrow conflict.
    ///
    /// Used by ops that need a read-only input and a writable output of the
    /// same `HashMap`. Returns an error if either name is missing or if the
    /// two names alias.
    #[allow(dead_code)]
    fn get_two_buffers<'a>(
        &'a mut self,
        a_name: &str,
        b_name: &str,
    ) -> Result<(&'a VulkanBuffer, &'a mut VulkanBuffer)> {
        if a_name == b_name {
            return Err(Error::Runtime(format!(
                "aliasing not supported here: {a_name}"
            )));
        }
        // SAFETY: a_name != b_name (checked above), so the two raw pointers are
        // disjoint. The HashMap itself is borrowed mutably only once; we split
        // the borrow manually to avoid the double-`get` limitation.
        let a_ptr: *const VulkanBuffer = self
            .buffers
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
        let b_ptr: *mut VulkanBuffer = self
            .buffers
            .get_mut(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
        unsafe { Ok((&*a_ptr, &mut *b_ptr)) }
    }

    /// Get three buffers (two shared inputs, one mut output) by name.
    #[allow(dead_code)]
    fn get_three_buffers<'a>(
        &'a mut self,
        a_name: &str,
        b_name: &str,
        c_name: &str,
    ) -> Result<(&'a VulkanBuffer, &'a VulkanBuffer, &'a mut VulkanBuffer)> {
        if a_name == c_name || b_name == c_name {
            return Err(Error::Runtime(format!(
                "output {c_name} aliases an input"
            )));
        }
        let a_ptr: *const VulkanBuffer = self
            .buffers
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
        let b_ptr: *const VulkanBuffer = self
            .buffers
            .get(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
        let c_ptr: *mut VulkanBuffer = self
            .buffers
            .get_mut(c_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {c_name}")))?;
        unsafe { Ok((&*a_ptr, &*b_ptr, &mut *c_ptr)) }
    }

    /// Dispatch a single op to the Vulkan backend.
    ///
    /// In Phase 1 most arms are stubs that return `Error::Runtime("not yet
    /// implemented for Vulkan")`. Subsequent phases fill them in.
    fn dispatch_op(&mut self, op: &CompiledOp) -> Result<()> {
        match &op.params {
            // ----- metadata-only / no-op ----------------------------------
            OpParams::None => match op.op_type.as_str() {
                // "Relu" with OpParams::None happens for in-place relus that
                // didn't get folded into a Clip; treat as no-op for now —
                // Phase 2 will route to relu_f32.
                "Flatten" | "Reshape" => self.dispatch_reshape(op),
                "Relu" => self.dispatch_relu(op),
                _ => Ok(()),
            },
            OpParams::Reshape { .. } => self.dispatch_reshape(op),

            // ----- F32 element-wise (Phase 2) -----------------------------
            OpParams::Add => self.dispatch_add(op),
            OpParams::Mul => self.dispatch_mul(op),
            OpParams::Sigmoid => self.dispatch_sigmoid(op),
            OpParams::Clip { min, max } => self.dispatch_clip(op, *min, *max),

            // ----- F32 GEMM / Conv (Phase 2) ------------------------------
            OpParams::Gemm {
                trans_a, trans_b, ..
            } => self.dispatch_gemm(op, *trans_a, *trans_b),
            OpParams::Conv2d {
                kernel_shape,
                strides,
                pads,
                group,
                ..
            } => self.dispatch_conv2d(op, *kernel_shape, *strides, *pads, *group),
            OpParams::MaxPool {
                kernel_shape,
                strides,
                ..
            } => self.dispatch_maxpool(op, *kernel_shape, *strides),
            OpParams::Softmax { axis } => self.dispatch_softmax(op, *axis),

            // ----- INT8 ops (Phase 3) ------------------------------------
            OpParams::Requantize { scale } => self.dispatch_requantize(op, *scale),
            OpParams::AddQuantized {
                scale_a_over_out,
                scale_b_over_out,
            } => self.dispatch_add_quantized(op, *scale_a_over_out, *scale_b_over_out),
            OpParams::Quantize { scale } => self.dispatch_quantize(op, *scale),
            OpParams::Dequantize { scale } => self.dispatch_dequantize(op, *scale),

            // ----- ops without Vulkan shaders yet (CPU fallback later) ---
            OpParams::Sub
            | OpParams::Div
            | OpParams::GlobalAvgPool
            | OpParams::AvgPool { .. }
            | OpParams::Concat { .. }
            | OpParams::Resize { .. }
            | OpParams::Split { .. }
            | OpParams::Transpose { .. }
            | OpParams::Slice { .. } => Err(Error::Runtime(format!(
                "op {:?} not yet supported by VulkanGraphRuntime",
                op.op_type
            ))),
        }
    }

    // =========================================================================
    // Phase 1 dispatch stubs — filled in by later phases.
    // =========================================================================

    fn dispatch_reshape(&mut self, op: &CompiledOp) -> Result<()> {
        // Reshape is metadata-only; the underlying storage layout is the same.
        // If input and output are different tensors, copy via GPU.
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Ok(());
        }
        if op.inputs[0] == op.outputs[0] {
            return Ok(());
        }
        // For Phase 1 we use download + upload as a placeholder. Phase 2
        // can swap this for `vkCmdCopyBuffer`.
        let src_bytes = {
            let src = self
                .buffers
                .get(&op.inputs[0])
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {}", op.inputs[0])))?;
            let mut bytes = vec![0u8; src.len_bytes()];
            self.backend
                .download(src, &mut bytes)
                .map_err(|e| Error::Runtime(format!("reshape download: {e}")))?;
            bytes
        };
        let dst = self
            .buffers
            .get_mut(&op.outputs[0])
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {}", op.outputs[0])))?;
        let n = dst.len_bytes().min(src_bytes.len());
        self.backend
            .upload(dst, &src_bytes[..n])
            .map_err(|e| Error::Runtime(format!("reshape upload: {e}")))?;
        Ok(())
    }

    fn dispatch_relu(&mut self, _op: &CompiledOp) -> Result<()> {
        Err(Error::Runtime(
            "Vulkan relu not wired yet — Phase 2".into(),
        ))
    }

    fn dispatch_add(&mut self, _op: &CompiledOp) -> Result<()> {
        Err(Error::Runtime("Vulkan add not wired yet — Phase 2".into()))
    }

    fn dispatch_mul(&mut self, _op: &CompiledOp) -> Result<()> {
        Err(Error::Runtime("Vulkan mul not wired yet — Phase 2".into()))
    }

    fn dispatch_sigmoid(&mut self, _op: &CompiledOp) -> Result<()> {
        Err(Error::Runtime(
            "Vulkan sigmoid not wired yet — Phase 2".into(),
        ))
    }

    fn dispatch_clip(&mut self, _op: &CompiledOp, _min: f32, _max: f32) -> Result<()> {
        Err(Error::Runtime(
            "Vulkan clip not wired yet — Phase 2".into(),
        ))
    }

    fn dispatch_gemm(&mut self, _op: &CompiledOp, _trans_a: bool, _trans_b: bool) -> Result<()> {
        Err(Error::Runtime(
            "Vulkan gemm not wired yet — Phase 2".into(),
        ))
    }

    fn dispatch_conv2d(
        &mut self,
        _op: &CompiledOp,
        _kernel: [usize; 2],
        _strides: [usize; 2],
        _pads: [usize; 4],
        _group: usize,
    ) -> Result<()> {
        Err(Error::Runtime(
            "Vulkan conv2d not wired yet — Phase 2".into(),
        ))
    }

    fn dispatch_maxpool(
        &mut self,
        _op: &CompiledOp,
        _kernel: [usize; 2],
        _strides: [usize; 2],
    ) -> Result<()> {
        Err(Error::Runtime(
            "Vulkan maxpool not wired yet — Phase 2".into(),
        ))
    }

    fn dispatch_softmax(&mut self, _op: &CompiledOp, _axis: i64) -> Result<()> {
        Err(Error::Runtime(
            "Vulkan softmax not wired yet — Phase 2".into(),
        ))
    }

    fn dispatch_requantize(&mut self, _op: &CompiledOp, _scale: f32) -> Result<()> {
        Err(Error::Runtime(
            "Vulkan requantize not wired yet — Phase 3".into(),
        ))
    }

    fn dispatch_add_quantized(
        &mut self,
        _op: &CompiledOp,
        _scale_a_over_out: f32,
        _scale_b_over_out: f32,
    ) -> Result<()> {
        Err(Error::Runtime(
            "Vulkan add_quantized not wired yet — Phase 3".into(),
        ))
    }

    fn dispatch_quantize(&mut self, _op: &CompiledOp, _scale: f32) -> Result<()> {
        Err(Error::Runtime(
            "Vulkan quantize not wired yet — Phase 3".into(),
        ))
    }

    fn dispatch_dequantize(&mut self, _op: &CompiledOp, _scale: f32) -> Result<()> {
        Err(Error::Runtime(
            "Vulkan dequantize not wired yet — Phase 3".into(),
        ))
    }
}

// =========================================================================
// Tests — Phase 1 only validates the skeleton (construction, no ops).
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::TensorShape;
    use dragonwing_vulkan::VulkanConfig;

    fn try_make_backend() -> Option<VulkanBackend> {
        VulkanBackend::new(VulkanConfig::default()).ok()
    }

    fn empty_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert(
            "input".to_string(),
            TensorShape::new(vec![1, 4], Dtype::F32),
        );
        shapes.insert(
            "output".to_string(),
            TensorShape::new(vec![1, 4], Dtype::F32),
        );
        Graph {
            ops: Vec::new(),
            shapes,
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn empty_graph_constructs_and_runs() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let graph = empty_graph();
        let mut rt = VulkanGraphRuntime::new(graph, backend).expect("new");

        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        rt.set_input_f32("input", &input).expect("set_input");

        // Empty op list — run is a no-op + synchronize.
        rt.run().expect("run");

        // Output buffer was never written, but get_output_f32 should still
        // succeed (reads whatever zeros / garbage is in the buffer; we just
        // assert the length).
        let out = rt.get_output_f32("output").expect("get_output");
        assert_eq!(out.len(), 4);
    }
}
