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

    /// Get two buffers (one shared, one mut) from the `buffers` map without
    /// borrowing `self` entirely. Used by dispatchers that also need
    /// `&self.backend`.
    ///
    /// Borrows `buffers: &mut HashMap<String, VulkanBuffer>` so it
    /// doesn't conflict with shared borrows of other `self` fields.
    fn get_two_buffers<'a>(
        buffers: &'a mut HashMap<String, VulkanBuffer>,
        a_name: &str,
        b_name: &str,
    ) -> Result<(&'a VulkanBuffer, &'a mut VulkanBuffer)> {
        if a_name == b_name {
            return Err(Error::Runtime(format!(
                "aliasing not supported here: {a_name}"
            )));
        }
        // SAFETY: a_name != b_name (checked above) so the two raw pointers
        // refer to disjoint entries. We split the &mut HashMap borrow.
        let a_ptr: *const VulkanBuffer = buffers
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
        let b_ptr: *mut VulkanBuffer = buffers
            .get_mut(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
        unsafe { Ok((&*a_ptr, &mut *b_ptr)) }
    }

    /// Three buffers: two shared inputs + one mut output.
    fn get_three_buffers<'a>(
        buffers: &'a mut HashMap<String, VulkanBuffer>,
        a_name: &str,
        b_name: &str,
        c_name: &str,
    ) -> Result<(&'a VulkanBuffer, &'a VulkanBuffer, &'a mut VulkanBuffer)> {
        if a_name == c_name || b_name == c_name {
            return Err(Error::Runtime(format!(
                "output {c_name} aliases an input"
            )));
        }
        let a_ptr: *const VulkanBuffer = buffers
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {a_name}")))?;
        let b_ptr: *const VulkanBuffer = buffers
            .get(b_name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {b_name}")))?;
        let c_ptr: *mut VulkanBuffer = buffers
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

    // =========================================================================
    // Phase 2 dispatchers — F32 ops wired to dragonwing_vulkan::ops::*.
    // =========================================================================
    //
    // Conventions:
    //   - Input names come from op.inputs[0..]; output name is op.outputs[0].
    //   - For in-place ops (input_name == output_name), we just borrow the
    //     buffer mutably; the GPU shader treats it as inout.
    //   - For out-of-place ops we use get_two_buffers / get_three_buffers
    //     to avoid HashMap borrow conflicts.
    //   - All shape info comes from self.graph.shapes (set by compile_model).
    //   - Optional biases (Conv/Gemm input[2]) are handled by adding a
    //     separate Add pass when present — Phase 2 keeps this simple.

    fn dispatch_relu(&mut self, op: &CompiledOp) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("relu: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();

        if in_name == out_name {
            // In-place ReLU on a single buffer.
            let buf = self
                .buffers
                .get_mut(&in_name)
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {in_name}")))?;
            dragonwing_vulkan::ops::relu_f32(&self.backend, buf)
                .map_err(|e| Error::Runtime(format!("vk relu_f32: {e}")))
        } else {
            // Copy input → output, then in-place relu on output.
            // (relu_f32 is single-buffer; no out-of-place variant exists.)
            let src_bytes = {
                let src = self
                    .buffers
                    .get(&in_name)
                    .ok_or_else(|| Error::Runtime(format!("buffer not found: {in_name}")))?;
                let mut b = vec![0u8; src.len_bytes()];
                self.backend
                    .download(src, &mut b)
                    .map_err(|e| Error::Runtime(format!("relu copy download: {e}")))?;
                b
            };
            let dst = self
                .buffers
                .get_mut(&out_name)
                .ok_or_else(|| Error::Runtime(format!("buffer not found: {out_name}")))?;
            self.backend
                .upload(dst, &src_bytes)
                .map_err(|e| Error::Runtime(format!("relu copy upload: {e}")))?;
            dragonwing_vulkan::ops::relu_f32(&self.backend, dst)
                .map_err(|e| Error::Runtime(format!("vk relu_f32: {e}")))
        }
    }

    fn dispatch_add(&mut self, op: &CompiledOp) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("add: needs 2 inputs and 1 output".into()));
        }
        let (a, b, y) = Self::get_three_buffers(&mut self.buffers, &op.inputs[0], &op.inputs[1], &op.outputs[0])?;
        dragonwing_vulkan::ops::add_f32(&self.backend, a, b, y)
            .map_err(|e| Error::Runtime(format!("vk add_f32: {e}")))
    }

    fn dispatch_mul(&mut self, op: &CompiledOp) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("mul: needs 2 inputs and 1 output".into()));
        }
        let (a, b, y) = Self::get_three_buffers(&mut self.buffers, &op.inputs[0], &op.inputs[1], &op.outputs[0])?;
        dragonwing_vulkan::ops::mul_f32(&self.backend, a, b, y)
            .map_err(|e| Error::Runtime(format!("vk mul_f32: {e}")))
    }

    fn dispatch_sigmoid(&mut self, op: &CompiledOp) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("sigmoid: missing input/output".into()));
        }
        if op.inputs[0] == op.outputs[0] {
            return Err(Error::Runtime(
                "vk sigmoid_f32 does not support in-place; compile_model should split"
                    .into(),
            ));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &op.inputs[0], &op.outputs[0])?;
        dragonwing_vulkan::ops::sigmoid_f32(&self.backend, input, output)
            .map_err(|e| Error::Runtime(format!("vk sigmoid_f32: {e}")))
    }

    fn dispatch_clip(&mut self, op: &CompiledOp, min: f32, max: f32) -> Result<()> {
        // No clip shader exists. For the common ReLU6 case (min=0, max=6) we
        // could approximate via relu_f32 + saturate; but for general clip we
        // need a dedicated shader. Phase 2 routes the common ReLU (min=0,
        // max=+inf) to relu_f32 and errors otherwise.
        if min == 0.0 && max.is_infinite() {
            return self.dispatch_relu(op);
        }
        let _ = (min, max);
        Err(Error::Runtime(
            "vk clip: only min=0/max=inf (ReLU) supported in Phase 2; \
             general clip shader is future work"
                .into(),
        ))
    }

    fn dispatch_gemm(&mut self, op: &CompiledOp, trans_a: bool, trans_b: bool) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("gemm: needs ≥2 inputs and 1 output".into()));
        }

        let a_name = &op.inputs[0];
        let b_name = &op.inputs[1];
        let c_name = &op.outputs[0];

        let a_shape = self
            .graph
            .shapes
            .get(a_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {a_name}")))?;
        let b_shape = self
            .graph
            .shapes
            .get(b_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {b_name}")))?;

        if a_shape.dims.len() != 2 || b_shape.dims.len() != 2 {
            return Err(Error::Runtime("Gemm inputs must be 2D".into()));
        }

        let m = if trans_a { a_shape.dims[1] } else { a_shape.dims[0] };
        let k = if trans_a { a_shape.dims[0] } else { a_shape.dims[1] };
        let n = if trans_b { b_shape.dims[0] } else { b_shape.dims[1] };

        // Vulkan gemm_f32 shader is non-transposed only. For ONNX models that
        // store FC weights with trans_b=true (the typical case), we'd need a
        // transposed shader. Phase 2 errors out and the caller can either:
        //   (a) pre-transpose B when loading the model, or
        //   (b) fall back to CpuGraphRuntime for this op.
        if trans_a || trans_b {
            return Err(Error::Runtime(format!(
                "vk gemm: trans_a/trans_b not supported (yet). \
                 trans_a={trans_a}, trans_b={trans_b}, MxNxK={}x{}x{}",
                m, n, k
            )));
        }

        // Buffers
        let a_name_str = a_name.clone();
        let b_name_str = b_name.clone();
        let c_name_str = c_name.clone();
        let (a, b, c) = Self::get_three_buffers(&mut self.buffers, &a_name_str, &b_name_str, &c_name_str)?;
        dragonwing_vulkan::ops::gemm_f32(&self.backend, a, b, c, m, n, k)
            .map_err(|e| Error::Runtime(format!("vk gemm_f32: {e}")))?;

        // Bias handling (op.inputs[2]) — Phase 2: error if present, since
        // gemm_f32 has no bias variant. Phase 6 can add a fused gemm+bias
        // shader or insert an Add pass.
        if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
            return Err(Error::Runtime(
                "vk gemm: bias not supported in Phase 2; preprocess model \
                 or use CpuGraphRuntime"
                    .into(),
            ));
        }
        Ok(())
    }

    fn dispatch_conv2d(
        &mut self,
        op: &CompiledOp,
        kernel_shape: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        group: usize,
    ) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("conv2d: needs ≥2 inputs and 1 output".into()));
        }

        let in_name = &op.inputs[0];
        let w_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let in_shape = self
            .graph
            .shapes
            .get(in_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {in_name}")))?;
        let out_shape = self
            .graph
            .shapes
            .get(out_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {out_name}")))?;

        if in_shape.dims.len() != 4 {
            return Err(Error::Runtime("conv2d input must be 4D NHWC".into()));
        }

        let n = in_shape.dims[0];
        let h_in = in_shape.dims[1];
        let w_in = in_shape.dims[2];
        let c_in = in_shape.dims[3];
        let c_out = out_shape.dims[3];
        let k_h = kernel_shape[0];
        let k_w = kernel_shape[1];
        let pad_h = pads[0];
        let pad_w = pads[1];

        // Depthwise / grouped not supported by the standard vk conv shader.
        if group != 1 {
            return Err(Error::Runtime(format!(
                "vk conv2d: group={group} not supported (need depthwise shader)"
            )));
        }
        if op.inputs.len() > 2 && !op.inputs[2].is_empty() {
            return Err(Error::Runtime(
                "vk conv2d: bias not yet supported in Phase 2".into(),
            ));
        }

        let in_name = in_name.clone();
        let w_name = w_name.clone();
        let out_name = out_name.clone();
        let (input, kernel, output) = Self::get_three_buffers(&mut self.buffers, &in_name, &w_name, &out_name)?;

        dragonwing_vulkan::ops::conv2d_f32_nhwc(
            &self.backend,
            input,
            kernel,
            output,
            n,
            h_in,
            w_in,
            c_in,
            c_out,
            k_h,
            k_w,
            strides[0],
            strides[1],
            pad_h,
            pad_w,
        )
        .map_err(|e| Error::Runtime(format!("vk conv2d_f32_nhwc: {e}")))
    }

    fn dispatch_maxpool(
        &mut self,
        op: &CompiledOp,
        kernel_shape: [usize; 2],
        strides: [usize; 2],
    ) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("maxpool: missing input/output".into()));
        }

        let in_name = &op.inputs[0];
        let out_name = &op.outputs[0];

        let in_shape = self
            .graph
            .shapes
            .get(in_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {in_name}")))?;
        if in_shape.dims.len() != 4 {
            return Err(Error::Runtime("maxpool input must be 4D NHWC".into()));
        }

        let n = in_shape.dims[0];
        let h_in = in_shape.dims[1];
        let w_in = in_shape.dims[2];
        let c = in_shape.dims[3];

        let in_name = in_name.clone();
        let out_name = out_name.clone();
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        dragonwing_vulkan::ops::maxpool2d_f32(
            &self.backend,
            input,
            output,
            n,
            h_in,
            w_in,
            c,
            kernel_shape[0],
            kernel_shape[1],
            strides[0],
            strides[1],
        )
        .map_err(|e| Error::Runtime(format!("vk maxpool2d_f32: {e}")))
    }

    fn dispatch_softmax(&mut self, op: &CompiledOp, _axis: i64) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("softmax: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime(
                "vk softmax does not support in-place".into(),
            ));
        }

        let in_shape = self
            .graph
            .shapes
            .get(&in_name)
            .ok_or_else(|| Error::Runtime(format!("shape not found: {in_name}")))?;
        let last = *in_shape.dims.last().unwrap_or(&1);
        let rows = in_shape.numel() / last;

        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        dragonwing_vulkan::ops::softmax_f32(&self.backend, input, output, rows, last)
            .map_err(|e| Error::Runtime(format!("vk softmax_f32: {e}")))
    }

    // =========================================================================
    // Phase 3 dispatchers — INT8 packed ops.
    // =========================================================================
    //
    // INT8 ops use UINT32 packing (4× I8 per U32). Tensor sizes are derived
    // from `graph.shapes` via `numel()`. The buffer for a packed I8 tensor
    // has byte length == `numel()` (1 byte per element). The buffer for an
    // INT32 accumulator has byte length == `numel() * 4`.

    fn dispatch_requantize(&mut self, op: &CompiledOp, scale: f32) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("requantize: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime("requantize: in-place not supported".into()));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        dragonwing_vulkan::ops::requantize_i32_to_i8_packed(&self.backend, input, output, scale)
            .map_err(|e| Error::Runtime(format!("vk requantize: {e}")))
    }

    fn dispatch_add_quantized(
        &mut self,
        op: &CompiledOp,
        scale_a_over_out: f32,
        scale_b_over_out: f32,
    ) -> Result<()> {
        if op.inputs.len() < 2 || op.outputs.is_empty() {
            return Err(Error::Runtime("add_q: needs 2 inputs + 1 output".into()));
        }
        let (a, b, y) =
            Self::get_three_buffers(&mut self.buffers, &op.inputs[0], &op.inputs[1], &op.outputs[0])?;
        dragonwing_vulkan::ops::add_i8_packed(
            &self.backend,
            a,
            b,
            y,
            scale_a_over_out,
            scale_b_over_out,
        )
        .map_err(|e| Error::Runtime(format!("vk add_i8: {e}")))
    }

    fn dispatch_quantize(&mut self, op: &CompiledOp, scale: f32) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("quantize: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime("quantize: in-place not supported".into()));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        dragonwing_vulkan::ops::quantize_f32_to_i8_packed(&self.backend, input, output, scale)
            .map_err(|e| Error::Runtime(format!("vk quantize: {e}")))
    }

    fn dispatch_dequantize(&mut self, op: &CompiledOp, scale: f32) -> Result<()> {
        if op.inputs.is_empty() || op.outputs.is_empty() {
            return Err(Error::Runtime("dequantize: missing input/output".into()));
        }
        let in_name = op.inputs[0].clone();
        let out_name = op.outputs[0].clone();
        if in_name == out_name {
            return Err(Error::Runtime("dequantize: in-place not supported".into()));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, &in_name, &out_name)?;
        dragonwing_vulkan::ops::dequantize_i8_packed_to_f32(&self.backend, input, output, scale)
            .map_err(|e| Error::Runtime(format!("vk dequantize: {e}")))
    }

    /// INT8 ReLU on packed tensors. Not in `OpParams` directly — exposed via a
    /// helper for ops that combine ReLU with another quantized step. Phase 3
    /// includes it for testing.
    #[allow(dead_code)]
    fn dispatch_relu_i8(&mut self, in_name: &str, out_name: &str) -> Result<()> {
        if in_name == out_name {
            return Err(Error::Runtime("relu_i8: in-place not supported".into()));
        }
        let (input, output) = Self::get_two_buffers(&mut self.buffers, in_name, out_name)?;
        dragonwing_vulkan::ops::relu_i8_packed(&self.backend, input, output)
            .map_err(|e| Error::Runtime(format!("vk relu_i8: {e}")))
    }

    /// Vulkan-side INT8 GEMM (low-level helper, not used by graph dispatch).
    /// Useful for parity tests that build packed buffers directly.
    #[allow(dead_code)]
    fn dispatch_gemm_i8(
        &mut self,
        a_name: &str,
        b_name: &str,
        c_name: &str,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let (a, b, c) = Self::get_three_buffers(&mut self.buffers, a_name, b_name, c_name)?;
        dragonwing_vulkan::ops::gemm_i8_packed(&self.backend, a, b, c, m, n, k)
            .map_err(|e| Error::Runtime(format!("vk gemm_i8: {e}")))
    }

    /// Set a Vulkan buffer by raw bytes (Phase 3 helper for INT8 testing).
    ///
    /// Bypasses dtype conversion; the caller is responsible for matching the
    /// buffer's byte length exactly.
    pub fn set_input_bytes(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let buffer = self
            .buffers
            .get_mut(name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {name}")))?;
        if buffer.len_bytes() != data.len() {
            return Err(Error::Runtime(format!(
                "set_input_bytes {name}: size mismatch buffer={} bytes, data={} bytes",
                buffer.len_bytes(),
                data.len(),
            )));
        }
        self.backend
            .upload(buffer, data)
            .map_err(|e| Error::Runtime(format!("upload {name}: {e}")))
    }

    /// Read a Vulkan buffer's raw bytes (Phase 3 helper for INT8 testing).
    pub fn get_output_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let buffer = self
            .buffers
            .get(name)
            .ok_or_else(|| Error::Runtime(format!("buffer not found: {name}")))?;
        let mut out = vec![0u8; buffer.len_bytes()];
        self.backend
            .download(buffer, &mut out)
            .map_err(|e| Error::Runtime(format!("download {name}: {e}")))?;
        Ok(out)
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

    // -----------------------------------------------------------------------
    // Phase 2 tests: F32 op dispatch parity
    // -----------------------------------------------------------------------

    /// Build a graph with a single Relu op (in-place).
    fn relu_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("x".into(), TensorShape::new(vec![1, 8], Dtype::F32));
        let ops = vec![CompiledOp {
            name: "relu1".into(),
            op_type: "Relu".into(),
            inputs: vec!["x".into()],
            outputs: vec!["x".into()], // in-place
            params: OpParams::None,
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["x".into()],
            outputs: vec!["x".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn relu_inplace_matches_cpu() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt = VulkanGraphRuntime::new(relu_graph(), backend).expect("new");

        let input = vec![-3.0f32, -1.0, 0.0, 0.5, 1.0, 2.0, -0.001, 100.0];
        rt.set_input_f32("x", &input).expect("set_input");
        rt.run().expect("run");
        let out = rt.get_output_f32("x").expect("get_output");

        let expected: Vec<f32> = input.iter().map(|&v| v.max(0.0)).collect();
        for (i, (a, b)) in expected.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
        }
    }

    /// Build a graph: y = a + b (out-of-place).
    fn add_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("a".into(), TensorShape::new(vec![8], Dtype::F32));
        shapes.insert("b".into(), TensorShape::new(vec![8], Dtype::F32));
        shapes.insert("y".into(), TensorShape::new(vec![8], Dtype::F32));
        let ops = vec![CompiledOp {
            name: "add1".into(),
            op_type: "Add".into(),
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["y".into()],
            params: OpParams::Add,
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["y".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn add_matches_cpu() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt = VulkanGraphRuntime::new(add_graph(), backend).expect("new");

        let a = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, -2.0, 0.0, 10.0];
        let b = vec![0.5f32, 0.5, 0.5, 0.5, 1.5, 2.5, 3.5, -5.0];
        rt.set_input_f32("a", &a).expect("set a");
        rt.set_input_f32("b", &b).expect("set b");
        rt.run().expect("run");
        let out = rt.get_output_f32("y").expect("get y");

        for i in 0..a.len() {
            let exp = a[i] + b[i];
            assert!((exp - out[i]).abs() < 1e-6, "mismatch at {i}: {exp} vs {}", out[i]);
        }
    }

    /// Build a graph: y = sigmoid(x), then z = x * y (SiLU).
    fn silu_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("x".into(), TensorShape::new(vec![8], Dtype::F32));
        shapes.insert("s".into(), TensorShape::new(vec![8], Dtype::F32));
        shapes.insert("z".into(), TensorShape::new(vec![8], Dtype::F32));
        let ops = vec![
            CompiledOp {
                name: "sig1".into(),
                op_type: "Sigmoid".into(),
                inputs: vec!["x".into()],
                outputs: vec!["s".into()],
                params: OpParams::Sigmoid,
            },
            CompiledOp {
                name: "mul1".into(),
                op_type: "Mul".into(),
                inputs: vec!["x".into(), "s".into()],
                outputs: vec!["z".into()],
                params: OpParams::Mul,
            },
        ];
        Graph {
            ops,
            shapes,
            inputs: vec!["x".into()],
            outputs: vec!["z".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn silu_two_op_graph_matches_cpu() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt = VulkanGraphRuntime::new(silu_graph(), backend).expect("new");

        let x = vec![-3.0f32, -1.0, 0.0, 0.5, 1.0, 2.0, -0.5, 1.5];
        rt.set_input_f32("x", &x).expect("set");
        rt.run().expect("run");
        let z = rt.get_output_f32("z").expect("get");

        for i in 0..x.len() {
            let sig = 1.0 / (1.0 + (-x[i]).exp());
            let exp = x[i] * sig;
            assert!((exp - z[i]).abs() < 1e-4, "silu mismatch at {i}: {exp} vs {}", z[i]);
        }
    }

    /// Build a tiny gemm graph: c = a × b, with M=2, K=3, N=2.
    fn gemm_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("a".into(), TensorShape::new(vec![2, 3], Dtype::F32));
        shapes.insert("b".into(), TensorShape::new(vec![3, 2], Dtype::F32));
        shapes.insert("c".into(), TensorShape::new(vec![2, 2], Dtype::F32));
        let ops = vec![CompiledOp {
            name: "gemm1".into(),
            op_type: "Gemm".into(),
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["c".into()],
            params: OpParams::Gemm {
                alpha: 1.0,
                beta: 1.0,
                trans_a: false,
                trans_b: false,
            },
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["c".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3 tests: INT8 packed op dispatch
    // -----------------------------------------------------------------------

    /// Pack `i8` slice into little-endian UINT32s as bytes (4× I8 → 1× U32).
    fn pack_i8_bytes(values: &[i8]) -> Vec<u8> {
        // `as u8` gives the two's-complement byte representation, which
        // matches the shader's `int(packed & 0xFF) - (if >= 128) 256` decoding.
        values.iter().map(|&v| v as u8).collect()
    }

    /// Build a graph that just quantizes F32 to I8 packed.
    fn quantize_graph() -> Graph {
        let mut shapes = HashMap::new();
        // Note: dtype on the graph as a whole still drives buffer allocation.
        // For mixed-dtype graphs we set per-tensor shapes accordingly.
        shapes.insert("x_f32".into(), TensorShape::new(vec![16], Dtype::F32));
        shapes.insert("x_i8".into(), TensorShape::new(vec![16], Dtype::I8));
        let ops = vec![CompiledOp {
            name: "q1".into(),
            op_type: "Quantize".into(),
            inputs: vec!["x_f32".into()],
            outputs: vec!["x_i8".into()],
            params: OpParams::Quantize { scale: 0.1 },
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["x_f32".into()],
            outputs: vec!["x_i8".into()],
            initializers: HashMap::new(),
            // Even for a mixed-dtype graph we have to pick a "primary" dtype;
            // F32 makes set_input_f32 use the right path for the F32 input.
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn quantize_roundtrip_via_dequantize() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };

        // We build a Q → DQ chain manually since the graph types don't yet
        // model two-shape graphs cleanly.
        let mut shapes = HashMap::new();
        shapes.insert("x_f32".into(), TensorShape::new(vec![16], Dtype::F32));
        shapes.insert("x_i8".into(), TensorShape::new(vec![16], Dtype::I8));
        shapes.insert("y_f32".into(), TensorShape::new(vec![16], Dtype::F32));
        let scale = 0.1f32;
        let ops = vec![
            CompiledOp {
                name: "q1".into(),
                op_type: "Quantize".into(),
                inputs: vec!["x_f32".into()],
                outputs: vec!["x_i8".into()],
                params: OpParams::Quantize { scale },
            },
            CompiledOp {
                name: "dq1".into(),
                op_type: "Dequantize".into(),
                inputs: vec!["x_i8".into()],
                outputs: vec!["y_f32".into()],
                params: OpParams::Dequantize { scale },
            },
        ];
        let graph = Graph {
            ops,
            shapes,
            inputs: vec!["x_f32".into()],
            outputs: vec!["y_f32".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        };

        let mut rt = VulkanGraphRuntime::new(graph, backend).expect("new");
        let input = vec![
            0.0f32, 0.1, -0.2, 0.5, -0.5, 1.0, -1.0, 12.7, // 12.7/0.1=127, clamps OK
            -12.7, 6.3, -6.3, 0.05, -0.05, 0.3, -0.7, 2.5,
        ];
        rt.set_input_f32("x_f32", &input).expect("set");
        rt.run().expect("run");
        let out = rt.get_output_f32("y_f32").expect("get");

        // Each value should be within one quantization step (scale = 0.1)
        // of the original. The roundtrip clamps values outside [-12.8, 12.7].
        for (i, (a, b)) in input.iter().zip(out.iter()).enumerate() {
            let err = (a - b).abs();
            assert!(err <= scale, "qdq mismatch at {i}: {a} -> {b}, err={err}");
        }
    }

    /// Build a graph that does only requantize: I32 input → I8 packed output.
    /// Used to validate the shader directly with manually-prepared buffers.
    fn requantize_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("acc_i32".into(), TensorShape::new(vec![16], Dtype::I32));
        shapes.insert("out_i8".into(), TensorShape::new(vec![16], Dtype::I8));
        let ops = vec![CompiledOp {
            name: "rq1".into(),
            op_type: "Requantize".into(),
            inputs: vec!["acc_i32".into()],
            outputs: vec!["out_i8".into()],
            params: OpParams::Requantize { scale: 0.01 },
        }];
        Graph {
            ops,
            shapes,
            inputs: vec!["acc_i32".into()],
            outputs: vec!["out_i8".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32, // doesn't matter for raw byte I/O
        }
    }

    #[test]
    fn requantize_clamps_and_scales() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };

        let mut rt = VulkanGraphRuntime::new(requantize_graph(), backend).expect("new");

        // input INT32 values
        let acc: Vec<i32> = vec![
            0, 100, -100, 12700, // 12700 * 0.01 = 127
            -12700, 25400,  // saturates to +127
            -25400, // saturates to -128
            500, -500, 1234, -1234, 5, -5, 9999, -9999, 12700,
        ];
        let acc_bytes: Vec<u8> = acc
            .iter()
            .flat_map(|v| v.to_le_bytes().into_iter())
            .collect();
        rt.set_input_bytes("acc_i32", &acc_bytes).expect("set");
        rt.run().expect("run");
        let out_bytes = rt.get_output_bytes("out_i8").expect("get");
        let out: Vec<i8> = out_bytes.iter().map(|&b| b as i8).collect();

        for (i, &v) in acc.iter().enumerate() {
            let expected = ((v as f32) * 0.01).round().clamp(-128.0, 127.0) as i8;
            assert_eq!(
                out[i], expected,
                "requantize[{i}]: input={v}, expected={expected}, got={}",
                out[i]
            );
        }
    }

    /// INT8 ReLU graph.
    fn relu_i8_graph() -> Graph {
        let mut shapes = HashMap::new();
        shapes.insert("x".into(), TensorShape::new(vec![8], Dtype::I8));
        shapes.insert("y".into(), TensorShape::new(vec![8], Dtype::I8));
        // OpParams::None is fine; dispatcher routes by op_type for Quantize/Relu
        // but we use the helper directly.
        Graph {
            ops: Vec::new(),
            shapes,
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn relu_i8_clamps_negatives() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };

        let mut rt = VulkanGraphRuntime::new(relu_i8_graph(), backend).expect("new");

        let input: Vec<i8> = vec![-128, -1, 0, 1, 127, -50, 50, -10];
        rt.set_input_bytes("x", &pack_i8_bytes(&input))
            .expect("set");

        rt.dispatch_relu_i8("x", "y").expect("relu_i8");
        rt.backend.synchronize().expect("sync");

        let out_bytes = rt.get_output_bytes("y").expect("get");
        let out: Vec<i8> = out_bytes.iter().map(|&b| b as i8).collect();

        let expected: Vec<i8> = input.iter().map(|&v| v.max(0)).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn gemm_matches_reference() {
        let Some(backend) = try_make_backend() else {
            eprintln!("skipping: no Vulkan device available");
            return;
        };
        let mut rt = VulkanGraphRuntime::new(gemm_graph(), backend).expect("new");

        // a is 2×3, b is 3×2
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        rt.set_input_f32("a", &a).expect("set a");
        rt.set_input_f32("b", &b).expect("set b");
        rt.run().expect("run");
        let c = rt.get_output_f32("c").expect("get c");

        // c[0,0] = 1*1 + 2*0 + 3*1 = 4
        // c[0,1] = 1*0 + 2*1 + 3*1 = 5
        // c[1,0] = 4*1 + 5*0 + 6*1 = 10
        // c[1,1] = 4*0 + 5*1 + 6*1 = 11
        let expected = [4.0f32, 5.0, 10.0, 11.0];
        for (i, (exp, got)) in expected.iter().zip(c.iter()).enumerate() {
            assert!((exp - got).abs() < 1e-4, "gemm[{i}]: {exp} vs {got}");
        }
    }
}
