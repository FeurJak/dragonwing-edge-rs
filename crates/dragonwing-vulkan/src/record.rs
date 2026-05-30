//! Batched command-buffer recording (Task 008 Phase 4).
//!
//! The original `ops::*` API records and submits one op per call
//! (allocating a fresh `VkCommandPool` and `VkCommandBuffer` each time,
//! then waiting on `device_wait_idle`). That's fine for tests and
//! single-op micro-benchmarks but it pessimises full-graph execution:
//! YOLOv8n's ~100 ops would do ~100 submits, ~100 wait_idles, and
//! ~100 command-pool allocations per inference call.
//!
//! [`OpsRecorder`] lets [`VulkanGraphRuntime`] amortise that overhead:
//!
//! ```text
//! let rec = OpsRecorder::begin(backend)?;
//! rec.record_add_f32(&a, &b, &mut y)?;
//! rec.record_memory_barrier();
//! rec.record_conv2d_f32_nhwc(...)?;
//! rec.finish_and_submit()?;
//! backend.synchronize()?;
//! ```
//!
//! Each `record_*` method:
//! 1. Gets/creates the pipeline (no-op after first run).
//! 2. Allocates a fresh descriptor set bound to the supplied buffers.
//! 3. Records `cmd_bind_pipeline`, `cmd_bind_descriptor_sets`,
//!    `cmd_push_constants`, `cmd_dispatch` into the shared command
//!    buffer.
//!
//! Descriptor sets are tracked and freed when the recorder is dropped
//! (after `device_wait_idle` confirms the GPU is done).
//!
//! `record_memory_barrier` inserts a global compute→compute storage
//! barrier (`VK_ACCESS_SHADER_WRITE_BIT → VK_ACCESS_SHADER_READ_BIT`).
//! The graph runtime should call this between any two ops where the
//! later op reads a buffer the earlier op wrote.

use std::sync::Arc;

use ash::vk;
use dragonwing_core::{BackendBuffer, Error, Result};

use crate::context::Context;
use crate::error::vk_err;
use crate::memory::VulkanBuffer;
use crate::pipeline::{OpKind, PipelineCache};
use crate::VulkanBackend;

/// Records many compute dispatches into a single Vulkan command buffer,
/// then submits once.
pub struct OpsRecorder {
    ctx: Arc<Context>,
    cache: Arc<PipelineCache>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    /// Descriptor sets allocated during recording; freed on `drop`.
    descriptors: Vec<vk::DescriptorSet>,
    /// Number of ops recorded so far (for diagnostics).
    op_count: usize,
    /// True once `finish_and_submit` has been called.
    finished: bool,
}

impl OpsRecorder {
    /// Begin recording. Allocates a fresh command pool + primary command
    /// buffer and calls `vkBeginCommandBuffer`.
    pub fn begin(backend: &VulkanBackend) -> Result<Self> {
        let ctx = backend.context().clone();
        let cache = backend.pipelines().clone();

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family_index())
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        // SAFETY: spec-compliant struct.
        let pool = unsafe { ctx.device().create_command_pool(&pool_info, None) }
            .map_err(|r| vk_err("OpsRecorder::create_command_pool", r))?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: pool just created.
        let cmds = unsafe { ctx.device().allocate_command_buffers(&alloc_info) }.map_err(|r| {
            unsafe { ctx.device().destroy_command_pool(pool, None) };
            vk_err("OpsRecorder::allocate_command_buffers", r)
        })?;
        let cmd = cmds[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: cmd valid.
        unsafe { ctx.device().begin_command_buffer(cmd, &begin_info) }
            .map_err(|r| vk_err("OpsRecorder::begin_command_buffer", r))?;

        Ok(Self {
            ctx,
            cache,
            pool,
            cmd,
            descriptors: Vec::with_capacity(32),
            op_count: 0,
            finished: false,
        })
    }

    /// Number of ops recorded so far.
    pub fn op_count(&self) -> usize {
        self.op_count
    }

    /// Insert a global compute→compute storage barrier.
    ///
    /// Use between any pair of ops where the later op reads a buffer
    /// written by the earlier op. We use a single global memory barrier
    /// (cheaper than per-buffer barriers on tile-based mobile GPUs).
    pub fn record_memory_barrier(&mut self) {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        // SAFETY: cmd recording.
        unsafe {
            self.ctx.device().cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
    }

    /// End the command buffer and submit to the queue, signalling the
    /// context's timeline semaphore. Caller should subsequently call
    /// `VulkanBackend::synchronize()` to wait for completion before
    /// reading any output buffer.
    pub fn finish_and_submit(mut self) -> Result<()> {
        self.finish_inner()
    }

    fn finish_inner(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        // SAFETY: cmd recording.
        unsafe { self.ctx.device().end_command_buffer(self.cmd) }
            .map_err(|r| vk_err("OpsRecorder::end_command_buffer", r))?;

        let signal_value = self.ctx.next_signal_value();
        let sem = self.ctx.timeline_semaphore();
        let cmd_bufs = [self.cmd];
        let signal_sems = [sem];
        let signal_values = [signal_value];
        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
        let submit_info = [vk::SubmitInfo::default()
            .command_buffers(&cmd_bufs)
            .signal_semaphores(&signal_sems)
            .push_next(&mut timeline_info)];
        // SAFETY: handles valid, struct chain correct.
        unsafe {
            self.ctx
                .device()
                .queue_submit(self.ctx.queue(), &submit_info, vk::Fence::null())
        }
        .map_err(|r| vk_err("OpsRecorder::queue_submit", r))
    }

    /// Allocate a descriptor set for `op`, bind `output_buf` at binding 0
    /// and `input_bufs[i]` at bindings 1..=N. Tracks the set for later
    /// cleanup.
    fn alloc_and_bind(
        &mut self,
        op: OpKind,
        output_buf: vk::Buffer,
        input_bufs: &[vk::Buffer],
    ) -> Result<(crate::pipeline::CachedPipeline, vk::DescriptorSet)> {
        let cached = self.cache.get_or_create(op)?;
        let desc_set = self.cache.allocate_descriptor_set(cached.descriptor_set_layout)?;
        self.descriptors.push(desc_set);

        let mut buf_infos: Vec<vk::DescriptorBufferInfo> =
            Vec::with_capacity(1 + input_bufs.len());
        buf_infos.push(vk::DescriptorBufferInfo {
            buffer: output_buf,
            offset: 0,
            range: vk::WHOLE_SIZE,
        });
        for &b in input_bufs {
            buf_infos.push(vk::DescriptorBufferInfo {
                buffer: b,
                offset: 0,
                range: vk::WHOLE_SIZE,
            });
        }
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        // SAFETY: handles valid.
        unsafe { self.ctx.device().update_descriptor_sets(&writes, &[]) };

        Ok((cached, desc_set))
    }

    /// Common tail for every record_* op: bind pipeline + set + push, dispatch.
    fn record_pipeline_dispatch(
        &mut self,
        cached: crate::pipeline::CachedPipeline,
        desc_set: vk::DescriptorSet,
        push_constants: &[u8],
        groups: (u32, u32, u32),
    ) {
        let device = self.ctx.device();
        // SAFETY: cmd recording.
        unsafe {
            device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
            device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                cached.layout,
                0,
                &[desc_set],
                &[],
            );
            if !push_constants.is_empty() {
                device.cmd_push_constants(
                    self.cmd,
                    cached.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            device.cmd_dispatch(self.cmd, groups.0, groups.1, groups.2);
        }
        self.op_count += 1;
    }

    // =====================================================================
    // F32 ops
    // =====================================================================

    /// Record `relu_f32` (in-place).
    pub fn record_relu_f32(&mut self, x: &mut VulkanBuffer) -> Result<()> {
        if !x.len_bytes().is_multiple_of(4) {
            return Err(Error::Backend(
                "record_relu_f32: buffer size must be multiple of 4".into(),
            ));
        }
        let n = x.len_bytes() / 4;
        // ReluF32 has 1 SSBO (inout); use alloc_and_bind with an empty input list.
        let (cached, desc_set) = self.alloc_and_bind(OpKind::ReluF32, x.vk_buffer(), &[])?;
        let pc: [u8; 16] = push_n_u32(n as u32);
        let groups = ((n as u32).div_ceil(64), 1, 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `add_f32`: `y = a + b`.
    pub fn record_add_f32(
        &mut self,
        a: &VulkanBuffer,
        b: &VulkanBuffer,
        y: &mut VulkanBuffer,
    ) -> Result<()> {
        check_same_size_2("add_f32", a, b)?;
        check_same_size_2("add_f32", a, y)?;
        let n = a.len_bytes() / 4;
        let (cached, desc_set) =
            self.alloc_and_bind(OpKind::AddF32, y.vk_buffer(), &[a.vk_buffer(), b.vk_buffer()])?;
        let pc = push_n_u32(n as u32);
        let groups = ((n as u32).div_ceil(64), 1, 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `mul_f32`: `y = a * b`.
    pub fn record_mul_f32(
        &mut self,
        a: &VulkanBuffer,
        b: &VulkanBuffer,
        y: &mut VulkanBuffer,
    ) -> Result<()> {
        check_same_size_2("mul_f32", a, b)?;
        check_same_size_2("mul_f32", a, y)?;
        let n = a.len_bytes() / 4;
        let (cached, desc_set) =
            self.alloc_and_bind(OpKind::MulF32, y.vk_buffer(), &[a.vk_buffer(), b.vk_buffer()])?;
        let pc = push_n_u32(n as u32);
        let groups = ((n as u32).div_ceil(64), 1, 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `sigmoid_f32`: `y = sigmoid(x)`.
    pub fn record_sigmoid_f32(
        &mut self,
        input: &VulkanBuffer,
        output: &mut VulkanBuffer,
    ) -> Result<()> {
        check_same_size_2("sigmoid_f32", input, output)?;
        let n = input.len_bytes() / 4;
        let (cached, desc_set) =
            self.alloc_and_bind(OpKind::SigmoidF32, output.vk_buffer(), &[input.vk_buffer()])?;
        let pc = push_n_u32(n as u32);
        let groups = ((n as u32).div_ceil(64), 1, 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `silu_f32`: fused `y = x * sigmoid(x)`.
    pub fn record_silu_f32(
        &mut self,
        input: &VulkanBuffer,
        output: &mut VulkanBuffer,
    ) -> Result<()> {
        check_same_size_2("silu_f32", input, output)?;
        let n = input.len_bytes() / 4;
        let (cached, desc_set) =
            self.alloc_and_bind(OpKind::SiluF32, output.vk_buffer(), &[input.vk_buffer()])?;
        let pc = push_n_u32(n as u32);
        let groups = ((n as u32).div_ceil(64), 1, 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `gemm_f32`: `c = a · b`, row-major, MxK × KxN.
    pub fn record_gemm_f32(
        &mut self,
        a: &VulkanBuffer,
        b: &VulkanBuffer,
        c: &mut VulkanBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len_bytes() != m * k * 4 {
            return Err(Error::Backend(format!(
                "record_gemm_f32: A size mismatch: have {} bytes, expected {}",
                a.len_bytes(),
                m * k * 4
            )));
        }
        if b.len_bytes() != k * n * 4 {
            return Err(Error::Backend(format!(
                "record_gemm_f32: B size mismatch: have {} bytes, expected {}",
                b.len_bytes(),
                k * n * 4
            )));
        }
        if c.len_bytes() != m * n * 4 {
            return Err(Error::Backend(format!(
                "record_gemm_f32: C size mismatch: have {} bytes, expected {}",
                c.len_bytes(),
                m * n * 4
            )));
        }
        let (cached, desc_set) = self.alloc_and_bind(
            OpKind::GemmF32,
            c.vk_buffer(),
            &[a.vk_buffer(), b.vk_buffer()],
        )?;
        let pc = push_3_u32(m as u32, n as u32, k as u32);
        let groups = ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `conv2d_f32_nhwc` (no bias; insert `record_bias_add_f32_nhwc`
    /// after if bias is present).
    #[allow(clippy::too_many_arguments)]
    pub fn record_conv2d_f32_nhwc(
        &mut self,
        input: &VulkanBuffer,
        kernel: &VulkanBuffer,
        output: &mut VulkanBuffer,
        n: usize,
        h_in: usize,
        w_in: usize,
        c_in: usize,
        c_out: usize,
        k_h: usize,
        k_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Result<()> {
        let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;
        let (cached, desc_set) = self.alloc_and_bind(
            OpKind::Conv2dF32Nhwc,
            output.vk_buffer(),
            &[input.vk_buffer(), kernel.vk_buffer()],
        )?;
        let pc = push_conv_64(
            h_in, w_in, c_in, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w, h_out, w_out, n,
        );
        let groups = (
            (w_out as u32).div_ceil(8),
            (h_out as u32).div_ceil(8),
            (n * c_out) as u32,
        );
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `bias_add_f32_nhwc`: `y[i] += bias[i % c_out]` in-place.
    pub fn record_bias_add_f32_nhwc(
        &mut self,
        y_inout: &mut VulkanBuffer,
        bias: &VulkanBuffer,
        c_out: usize,
    ) -> Result<()> {
        if !y_inout.len_bytes().is_multiple_of(4) {
            return Err(Error::Backend(
                "record_bias_add_f32_nhwc: y size must be multiple of 4".into(),
            ));
        }
        if bias.len_bytes() != c_out * 4 {
            return Err(Error::Backend(format!(
                "record_bias_add_f32_nhwc: bias has {} bytes, expected {}",
                bias.len_bytes(),
                c_out * 4
            )));
        }
        let n = y_inout.len_bytes() / 4;
        if n % c_out != 0 {
            return Err(Error::Backend(format!(
                "record_bias_add_f32_nhwc: total {n} not divisible by c_out {c_out}"
            )));
        }
        let (cached, desc_set) = self.alloc_and_bind(
            OpKind::BiasAddF32Nhwc,
            y_inout.vk_buffer(),
            &[bias.vk_buffer()],
        )?;
        let pc = push_2_u32(n as u32, c_out as u32);
        let groups = ((n as u32).div_ceil(64), 1, 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `maxpool2d_f32`.
    #[allow(clippy::too_many_arguments)]
    pub fn record_maxpool2d_f32(
        &mut self,
        input: &VulkanBuffer,
        output: &mut VulkanBuffer,
        n: usize,
        h_in: usize,
        w_in: usize,
        c: usize,
        k_h: usize,
        k_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<()> {
        let h_out = (h_in - k_h) / stride_h + 1;
        let w_out = (w_in - k_w) / stride_w + 1;
        let (cached, desc_set) =
            self.alloc_and_bind(OpKind::Maxpool2dF32, output.vk_buffer(), &[input.vk_buffer()])?;
        // The maxpool shader uses the same 4-uvec4 layout (64 B) as conv.
        // The shader ignores fields it doesn't need; reuse push_conv_64
        // with c_in == c_out == c and k = pad=stride passed through.
        let pc = push_conv_64(
            h_in, w_in, c, c, k_h, k_w, stride_h, stride_w, 0, 0, h_out, w_out, n,
        );
        let groups = (
            (w_out as u32).div_ceil(8),
            (h_out as u32).div_ceil(8),
            (n * c) as u32,
        );
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `softmax_f32` over rows of `[rows, cols]`.
    pub fn record_softmax_f32(
        &mut self,
        input: &VulkanBuffer,
        output: &mut VulkanBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if input.len_bytes() != rows * cols * 4 || output.len_bytes() != rows * cols * 4 {
            return Err(Error::Backend(format!(
                "record_softmax_f32: size mismatch rows={rows} cols={cols}"
            )));
        }
        let (cached, desc_set) =
            self.alloc_and_bind(OpKind::SoftmaxF32, output.vk_buffer(), &[input.vk_buffer()])?;
        let pc = push_2_u32(cols as u32, rows as u32);
        let groups = (1, rows as u32, 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    // =====================================================================
    // INT8 ops
    // =====================================================================

    /// Record `requantize_i32_to_i8_packed`.
    pub fn record_requantize_i32_to_i8_packed(
        &mut self,
        input_i32: &VulkanBuffer,
        output_packed: &mut VulkanBuffer,
        requant_scale: f32,
    ) -> Result<()> {
        // input is I32: 4 bytes/elem; output is packed I8: 1 byte/elem.
        // Element count comes from the output buffer.
        let n_elem = output_packed.len_bytes();
        if input_i32.len_bytes() != n_elem * 4 {
            return Err(Error::Backend(format!(
                "record_requantize: I32 input {} bytes != {n_elem}*4",
                input_i32.len_bytes()
            )));
        }
        let (cached, desc_set) = self.alloc_and_bind(
            OpKind::RequantizeI32ToI8,
            output_packed.vk_buffer(),
            &[input_i32.vk_buffer()],
        )?;
        let pc = push_n_f32(n_elem as u32, requant_scale);
        let groups = ((n_elem as u32).div_ceil(64 * 4), 1, 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `quantize_f32_to_i8_packed`.
    pub fn record_quantize_f32_to_i8_packed(
        &mut self,
        input_f32: &VulkanBuffer,
        output_packed: &mut VulkanBuffer,
        scale: f32,
    ) -> Result<()> {
        let n_elem = output_packed.len_bytes();
        if input_f32.len_bytes() != n_elem * 4 {
            return Err(Error::Backend(format!(
                "record_quantize: F32 input {} bytes != {n_elem}*4",
                input_f32.len_bytes()
            )));
        }
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let (cached, desc_set) = self.alloc_and_bind(
            OpKind::QuantizeF32ToI8,
            output_packed.vk_buffer(),
            &[input_f32.vk_buffer()],
        )?;
        let pc = push_n_f32(n_elem as u32, inv_scale);
        let groups = ((n_elem as u32).div_ceil(64 * 4), 1, 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `dequantize_i8_packed_to_f32`.
    pub fn record_dequantize_i8_packed_to_f32(
        &mut self,
        input_packed: &VulkanBuffer,
        output_f32: &mut VulkanBuffer,
        scale: f32,
    ) -> Result<()> {
        let n_elem = input_packed.len_bytes();
        if output_f32.len_bytes() != n_elem * 4 {
            return Err(Error::Backend(format!(
                "record_dequantize: F32 output {} bytes != {n_elem}*4",
                output_f32.len_bytes()
            )));
        }
        let (cached, desc_set) = self.alloc_and_bind(
            OpKind::DequantizeI8ToF32,
            output_f32.vk_buffer(),
            &[input_packed.vk_buffer()],
        )?;
        let pc = push_n_f32(n_elem as u32, scale);
        let groups = ((n_elem as u32).div_ceil(64), 1, 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record `add_i8_packed` with per-input scale ratios.
    pub fn record_add_i8_packed(
        &mut self,
        a: &VulkanBuffer,
        b: &VulkanBuffer,
        y: &mut VulkanBuffer,
        scale_a_over_y: f32,
        scale_b_over_y: f32,
    ) -> Result<()> {
        if a.len_bytes() != b.len_bytes() || a.len_bytes() != y.len_bytes() {
            return Err(Error::Backend("record_add_i8_packed: size mismatch".into()));
        }
        let n_elem = y.len_bytes();
        let (cached, desc_set) =
            self.alloc_and_bind(OpKind::AddI8Packed, y.vk_buffer(), &[a.vk_buffer(), b.vk_buffer()])?;
        let pc = push_n_2f32(n_elem as u32, scale_a_over_y, scale_b_over_y);
        let groups = ((n_elem as u32).div_ceil(64 * 4), 1, 1);
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }

    /// Record fused INT8 `conv2d + requantize + (optional) relu`.
    #[allow(clippy::too_many_arguments)]
    pub fn record_conv2d_requant_relu_i8_packed(
        &mut self,
        input_packed: &VulkanBuffer,
        kernel_packed: &VulkanBuffer,
        output_packed: &mut VulkanBuffer,
        n: usize,
        h_in: usize,
        w_in: usize,
        c_in: usize,
        c_out: usize,
        k_h: usize,
        k_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        requant_scale: f32,
        do_relu: bool,
    ) -> Result<()> {
        if c_in % 4 != 0 || c_out % 4 != 0 {
            return Err(Error::Backend(format!(
                "record_conv2d_requant_relu_i8_packed: c_in ({c_in}) and c_out ({c_out}) must be multiples of 4"
            )));
        }
        let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

        let (cached, desc_set) = self.alloc_and_bind(
            OpKind::Conv2dRequantReluI8Packed,
            output_packed.vk_buffer(),
            &[input_packed.vk_buffer(), kernel_packed.vk_buffer()],
        )?;
        // 80-byte push constants: 4× uvec4 + float + uint.
        let pc = push_conv_fused_80(
            h_in,
            w_in,
            c_in,
            c_out,
            k_h,
            k_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            h_out,
            w_out,
            n,
            requant_scale,
            if do_relu { 1 } else { 0 },
        );
        let groups = (
            (w_out as u32).div_ceil(8),
            (h_out as u32).div_ceil(8),
            (n * (c_out / 4)) as u32,
        );
        self.record_pipeline_dispatch(cached, desc_set, &pc, groups);
        Ok(())
    }
}

impl Drop for OpsRecorder {
    fn drop(&mut self) {
        // If finish_and_submit was never called we still need to end the cmd
        // buffer; otherwise the validation layers complain.
        if !self.finished {
            // SAFETY: cmd recording.
            let _ = unsafe { self.ctx.device().end_command_buffer(self.cmd) };
            self.finished = true;
        }
        // Wait for any submitted work to finish before tearing down. This is
        // conservative but matches the OneShot semantics.
        // SAFETY: best-effort wait.
        let _ = unsafe { self.ctx.device().device_wait_idle() };
        // Free all descriptor sets.
        for set in self.descriptors.drain(..) {
            let _ = self.cache.free_descriptor_set(set);
        }
        // SAFETY: pool owned by us; cmd implicitly freed.
        unsafe {
            self.ctx.device().destroy_command_pool(self.pool, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Push-constant builders
// ---------------------------------------------------------------------------

fn check_same_size_2(label: &str, a: &VulkanBuffer, b: &VulkanBuffer) -> Result<()> {
    if a.len_bytes() != b.len_bytes() {
        return Err(Error::Backend(format!(
            "{label}: buffer size mismatch ({} vs {})",
            a.len_bytes(),
            b.len_bytes()
        )));
    }
    if !a.len_bytes().is_multiple_of(4) {
        return Err(Error::Backend(format!(
            "{label}: buffer size must be multiple of 4"
        )));
    }
    Ok(())
}

fn push_n_u32(n: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&n.to_le_bytes());
    out
}

fn push_2_u32(a: u32, b: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a.to_le_bytes());
    out[4..8].copy_from_slice(&b.to_le_bytes());
    out
}

fn push_3_u32(a: u32, b: u32, c: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a.to_le_bytes());
    out[4..8].copy_from_slice(&b.to_le_bytes());
    out[8..12].copy_from_slice(&c.to_le_bytes());
    out
}

fn push_n_f32(n: u32, f: f32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&n.to_le_bytes());
    out[4..8].copy_from_slice(&f.to_le_bytes());
    out
}

fn push_n_2f32(n: u32, f1: f32, f2: f32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&n.to_le_bytes());
    out[4..8].copy_from_slice(&f1.to_le_bytes());
    out[8..12].copy_from_slice(&f2.to_le_bytes());
    out
}

#[allow(clippy::too_many_arguments)]
fn push_conv_64(
    h_in: usize,
    w_in: usize,
    c_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    h_out: usize,
    w_out: usize,
    n: usize,
) -> [u8; 64] {
    let mut out = [0u8; 64];
    let words: [u32; 16] = [
        h_in as u32, w_in as u32, c_in as u32, c_out as u32,
        k_h as u32, k_w as u32, stride_h as u32, stride_w as u32,
        pad_h as u32, pad_w as u32, h_out as u32, w_out as u32,
        n as u32, 0, 0, 0,
    ];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn push_conv_fused_80(
    h_in: usize,
    w_in: usize,
    c_in: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    h_out: usize,
    w_out: usize,
    n: usize,
    requant_scale: f32,
    do_relu: u32,
) -> [u8; 80] {
    let mut out = [0u8; 80];
    let words: [u32; 16] = [
        h_in as u32, w_in as u32, c_in as u32, c_out as u32,
        k_h as u32, k_w as u32, stride_h as u32, stride_w as u32,
        pad_h as u32, pad_w as u32, h_out as u32, w_out as u32,
        n as u32, 0, 0, 0,
    ];
    for (i, w) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    // Trailing 16 bytes: requant_scale (f32) + do_relu (u32) + 2× padding.
    out[64..68].copy_from_slice(&requant_scale.to_le_bytes());
    out[68..72].copy_from_slice(&do_relu.to_le_bytes());
    out
}
