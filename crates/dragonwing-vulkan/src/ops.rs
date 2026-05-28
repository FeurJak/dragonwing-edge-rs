//! Vulkan compute ops.
//!
//! Each op follows this pattern:
//!
//! 1. Get or create the pipeline via [`PipelineCache::get_or_create`].
//! 2. Allocate a descriptor set.
//! 3. Write buffer bindings into the descriptor set.
//! 4. Allocate a one-shot command buffer.
//! 5. Record: bind pipeline, bind descriptor set, push constants, dispatch.
//! 6. Submit with a timeline-semaphore signal.
//! 7. Free the descriptor set (after GPU work completes — but we do it
//!    lazily or rely on pool reset).
//!
//! Command buffers are allocated from a per-submission pool that is reset
//! after each submit. This is simple but not optimal for high-frequency
//! dispatch; a ring of command pools could be used later.
//!
//! # Synchronisation
//!
//! Each submit signals the next value of the context's timeline semaphore.
//! [`VulkanBackend::synchronize`](crate::VulkanBackend) waits on the
//! latest signalled value. This means ops are serialised; overlapping
//! dispatch requires a more sophisticated model (out of scope for task 002).

use std::sync::Arc;

use ash::vk;
use dragonwing_core::{BackendBuffer, Error, Result};

use crate::context::Context;
use crate::error::vk_err;
use crate::memory::VulkanBuffer;
use crate::pipeline::OpKind;
use crate::VulkanBackend;

// ---------------------------------------------------------------------------
// Shared dispatch helper
// ---------------------------------------------------------------------------

/// One-shot command pool + buffer. Freed on drop.
struct OneShot {
    ctx: Arc<Context>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
}

impl OneShot {
    fn new(ctx: Arc<Context>) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family_index())
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        // SAFETY: spec-compliant struct.
        let pool = unsafe { ctx.device().create_command_pool(&pool_info, None) }
            .map_err(|r| vk_err("create_command_pool", r))?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: pool just created.
        let cmds = unsafe { ctx.device().allocate_command_buffers(&alloc_info) }.map_err(|r| {
            unsafe { ctx.device().destroy_command_pool(pool, None) };
            vk_err("allocate_command_buffers", r)
        })?;
        let cmd = cmds[0];

        Ok(Self { ctx, pool, cmd })
    }

    fn begin(&self) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: cmd valid.
        unsafe { self.ctx.device().begin_command_buffer(self.cmd, &begin_info) }
            .map_err(|r| vk_err("begin_command_buffer", r))
    }

    fn end(&self) -> Result<()> {
        // SAFETY: cmd recording.
        unsafe { self.ctx.device().end_command_buffer(self.cmd) }
            .map_err(|r| vk_err("end_command_buffer", r))
    }

    /// Submit and signal timeline semaphore.
    fn submit(&self) -> Result<()> {
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
        unsafe { self.ctx.device().queue_submit(self.ctx.queue(), &submit_info, vk::Fence::null()) }
            .map_err(|r| vk_err("queue_submit", r))
    }
}

impl Drop for OneShot {
    fn drop(&mut self) {
        // Wait for GPU to finish before destroying the command pool.
        // This ensures the command buffer is no longer in use.
        // SAFETY: device is valid, this is a best-effort wait.
        let _ = unsafe { self.ctx.device().device_wait_idle() };
        // SAFETY: we own pool and cmd.
        unsafe {
            // Command buffer freed implicitly when pool is destroyed.
            self.ctx.device().destroy_command_pool(self.pool, None);
        }
    }
}

// ---------------------------------------------------------------------------
// fill_f32
// ---------------------------------------------------------------------------

/// Fill `dst` with `value`. `dst.len_bytes()` must be a multiple of 4.
///
/// Push constants: `{ n: u32, value: f32 }` where `n` is element count.
pub fn fill_f32(backend: &VulkanBackend, dst: &mut VulkanBuffer, value: f32) -> Result<()> {
    let n = dst.len_bytes() / 4;
    if !dst.len_bytes().is_multiple_of(4) {
        return Err(Error::Backend(
            "fill_f32: buffer size must be multiple of 4".into(),
        ));
    }

    let cached = backend.pipelines().get_or_create(OpKind::FillF32)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    // Write descriptor: binding 0 = dst.
    let buf_info = [vk::DescriptorBufferInfo {
        buffer: dst.vk_buffer(),
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let write = [vk::WriteDescriptorSet::default()
        .dst_set(desc_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&buf_info)];
    // SAFETY: handles valid.
    unsafe { backend.context().device().update_descriptor_sets(&write, &[]) };

    // Record command buffer.
    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    // Push constants: must match shader's layout (16 bytes with padding).
    #[repr(C)]
    struct PushFill {
        n: u32,
        value: f32,
        _pad0: f32,
        _pad1: f32,
    }
    let pc = PushFill {
        n: n as u32,
        value,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), std::mem::size_of::<PushFill>()) };

    let device = backend.context().device();
    // SAFETY: cmd recording, handles valid.
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(
            one_shot.cmd,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_push_constants(
            one_shot.cmd,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );

        // Dispatch: local_size_x=64 in shader, so groups = ceil(n / 64).
        let groups = (n as u32).div_ceil(64);
        device.cmd_dispatch(one_shot.cmd, groups, 1, 1);
    }

    one_shot.end()?;
    one_shot.submit()?;

    // Descriptor set freed when pool is destroyed or on explicit free.
    // For simplicity we leave it allocated; the pool has enough capacity.
    Ok(())
}

// ---------------------------------------------------------------------------
// axpy_f32
// ---------------------------------------------------------------------------

/// Compute `y[i] = alpha * x[i] + y[i]` for `n` elements.
///
/// Both buffers must have the same size, a multiple of 4 bytes.
///
/// Push constants: `{ n: u32, alpha: f32 }`.
pub fn axpy_f32(
    backend: &VulkanBackend,
    x: &VulkanBuffer,
    y: &mut VulkanBuffer,
    alpha: f32,
) -> Result<()> {
    if x.len_bytes() != y.len_bytes() {
        return Err(Error::Backend("axpy_f32: buffer size mismatch".into()));
    }
    let n = x.len_bytes() / 4;
    if !x.len_bytes().is_multiple_of(4) {
        return Err(Error::Backend(
            "axpy_f32: buffer size must be multiple of 4".into(),
        ));
    }

    let cached = backend.pipelines().get_or_create(OpKind::AxpyF32)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    // Shader bindings: 0=Y (read/write), 1=X (readonly).
    let buf_infos = [
        vk::DescriptorBufferInfo {
            buffer: y.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: x.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[1..2]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    // Push constants: must match shader layout (16 bytes with padding).
    #[repr(C)]
    struct PushAxpy {
        n: u32,
        alpha: f32,
        _pad0: f32,
        _pad1: f32,
    }
    let pc = PushAxpy {
        n: n as u32,
        alpha,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), std::mem::size_of::<PushAxpy>()) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(
            one_shot.cmd,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_push_constants(
            one_shot.cmd,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        let groups = (n as u32).div_ceil(64);
        device.cmd_dispatch(one_shot.cmd, groups, 1, 1);
    }

    one_shot.end()?;
    one_shot.submit()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// relu_f32
// ---------------------------------------------------------------------------

/// Apply ReLU in-place: `x[i] = max(x[i], 0)`.
///
/// Push constants: `{ n: u32 }`.
pub fn relu_f32(backend: &VulkanBackend, x: &mut VulkanBuffer) -> Result<()> {
    let n = x.len_bytes() / 4;
    if !x.len_bytes().is_multiple_of(4) {
        return Err(Error::Backend(
            "relu_f32: buffer size must be multiple of 4".into(),
        ));
    }

    let cached = backend.pipelines().get_or_create(OpKind::ReluF32)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_info = [vk::DescriptorBufferInfo {
        buffer: x.vk_buffer(),
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let write = [vk::WriteDescriptorSet::default()
        .dst_set(desc_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&buf_info)];
    unsafe { backend.context().device().update_descriptor_sets(&write, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    // Push constants: must match shader layout (16 bytes with padding).
    #[repr(C)]
    struct PushRelu {
        n: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }
    let pc = PushRelu {
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), std::mem::size_of::<PushRelu>()) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(
            one_shot.cmd,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_push_constants(
            one_shot.cmd,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        let groups = (n as u32).div_ceil(64);
        device.cmd_dispatch(one_shot.cmd, groups, 1, 1);
    }

    one_shot.end()?;
    one_shot.submit()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// gemm_f32 (naive)
// ---------------------------------------------------------------------------

/// Compute C = A × B (row-major, no transpose).
///
/// * A: M×K
/// * B: K×N
/// * C: M×N
///
/// All buffers must be sized correctly (`M*K*4`, `K*N*4`, `M*N*4`).
///
/// Push constants: `{ M: u32, N: u32, K: u32 }`.
///
/// This is the naive O(MNK) algorithm matching the CPU reference. A tiled
/// version is planned for task 003.
pub fn gemm_f32(
    backend: &VulkanBackend,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    c: &mut VulkanBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if a.len_bytes() != m * k * 4 {
        return Err(Error::Backend(format!(
            "gemm_f32: A size mismatch: expected {} got {}",
            m * k * 4,
            a.len_bytes()
        )));
    }
    if b.len_bytes() != k * n * 4 {
        return Err(Error::Backend(format!(
            "gemm_f32: B size mismatch: expected {} got {}",
            k * n * 4,
            b.len_bytes()
        )));
    }
    if c.len_bytes() != m * n * 4 {
        return Err(Error::Backend(format!(
            "gemm_f32: C size mismatch: expected {} got {}",
            m * n * 4,
            c.len_bytes()
        )));
    }

    let cached = backend.pipelines().get_or_create(OpKind::GemmF32)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    // bindings: 0 = A, 1 = B, 2 = C.
    // Shader bindings: 0=C (output), 1=A, 2=B
    let buf_infos = [
        vk::DescriptorBufferInfo {
            buffer: c.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: a.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: b.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[1..2]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[2..3]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    // Push constants: must match shader layout (16 bytes with padding).
    #[repr(C)]
    struct PushGemm {
        m: u32,
        n: u32,
        k: u32,
        _pad: u32,
    }
    let pc = PushGemm {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        _pad: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), std::mem::size_of::<PushGemm>()) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(
            one_shot.cmd,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_push_constants(
            one_shot.cmd,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        // gemm shader: local_size = (16, 16, 1). Dispatch (ceil(N/16), ceil(M/16), 1).
        let groups_x = (n as u32).div_ceil(16);
        let groups_y = (m as u32).div_ceil(16);
        device.cmd_dispatch(one_shot.cmd, groups_x, groups_y, 1);
    }

    one_shot.end()?;
    one_shot.submit()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// add_f32
// ---------------------------------------------------------------------------

/// Compute y[i] = a[i] + b[i] (element-wise addition).
///
/// All buffers must have the same size, a multiple of 4 bytes.
///
/// Push constants: `{ n: u32, _pad0: u32, _pad1: u32, _pad2: u32 }`.
pub fn add_f32(
    backend: &VulkanBackend,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    y: &mut VulkanBuffer,
) -> Result<()> {
    if a.len_bytes() != b.len_bytes() || a.len_bytes() != y.len_bytes() {
        return Err(Error::Backend("add_f32: buffer size mismatch".into()));
    }
    let n = a.len_bytes() / 4;
    if !a.len_bytes().is_multiple_of(4) {
        return Err(Error::Backend(
            "add_f32: buffer size must be multiple of 4".into(),
        ));
    }

    let cached = backend.pipelines().get_or_create(OpKind::AddF32)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    // Bindings: 0=Y (output), 1=A, 2=B
    let buf_infos = [
        vk::DescriptorBufferInfo {
            buffer: y.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: a.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: b.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[1..2]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[2..3]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    #[repr(C)]
    struct PushAdd {
        n: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }
    let pc = PushAdd {
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), std::mem::size_of::<PushAdd>()) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(
            one_shot.cmd,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_push_constants(
            one_shot.cmd,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        let groups = (n as u32).div_ceil(64);
        device.cmd_dispatch(one_shot.cmd, groups, 1, 1);
    }

    one_shot.end()?;
    one_shot.submit()?;
    Ok(())
}

// ===========================================================================
// FP16 ops
// ===========================================================================

// ---------------------------------------------------------------------------
// fill_fp16
// ---------------------------------------------------------------------------

/// Fill `dst` with `value` (F16 buffer). `dst.len_bytes()` must be a multiple of 2.
///
/// Push constants: `{ n: u32, value: f32 }` where `n` is element count.
pub fn fill_fp16(backend: &VulkanBackend, dst: &mut VulkanBuffer, value: f32) -> Result<()> {
    let n = dst.len_bytes() / 2;
    if !dst.len_bytes().is_multiple_of(2) {
        return Err(Error::Backend(
            "fill_fp16: buffer size must be multiple of 2".into(),
        ));
    }

    let cached = backend.pipelines().get_or_create(OpKind::FillFp16)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_info = [vk::DescriptorBufferInfo {
        buffer: dst.vk_buffer(),
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let write = [vk::WriteDescriptorSet::default()
        .dst_set(desc_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&buf_info)];
    unsafe { backend.context().device().update_descriptor_sets(&write, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    #[repr(C)]
    struct PushFill {
        n: u32,
        value: f32,
        _pad0: f32,
        _pad1: f32,
    }
    let pc = PushFill {
        n: n as u32,
        value,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), std::mem::size_of::<PushFill>()) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(
            one_shot.cmd,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_push_constants(
            one_shot.cmd,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        let groups = (n as u32).div_ceil(64);
        device.cmd_dispatch(one_shot.cmd, groups, 1, 1);
    }

    one_shot.end()?;
    one_shot.submit()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// axpy_fp16
// ---------------------------------------------------------------------------

/// Compute `y[i] = alpha * x[i] + y[i]` for `n` elements (F16 buffers).
///
/// Both buffers must have the same size, a multiple of 2 bytes.
pub fn axpy_fp16(
    backend: &VulkanBackend,
    x: &VulkanBuffer,
    y: &mut VulkanBuffer,
    alpha: f32,
) -> Result<()> {
    if x.len_bytes() != y.len_bytes() {
        return Err(Error::Backend("axpy_fp16: buffer size mismatch".into()));
    }
    let n = x.len_bytes() / 2;
    if !x.len_bytes().is_multiple_of(2) {
        return Err(Error::Backend(
            "axpy_fp16: buffer size must be multiple of 2".into(),
        ));
    }

    let cached = backend.pipelines().get_or_create(OpKind::AxpyFp16)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_infos = [
        vk::DescriptorBufferInfo {
            buffer: y.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: x.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[1..2]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    #[repr(C)]
    struct PushAxpy {
        n: u32,
        alpha: f32,
        _pad0: f32,
        _pad1: f32,
    }
    let pc = PushAxpy {
        n: n as u32,
        alpha,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), std::mem::size_of::<PushAxpy>()) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(
            one_shot.cmd,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_push_constants(
            one_shot.cmd,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        let groups = (n as u32).div_ceil(64);
        device.cmd_dispatch(one_shot.cmd, groups, 1, 1);
    }

    one_shot.end()?;
    one_shot.submit()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// relu_fp16
// ---------------------------------------------------------------------------

/// Apply ReLU in-place: `x[i] = max(x[i], 0)` (F16 buffer).
pub fn relu_fp16(backend: &VulkanBackend, x: &mut VulkanBuffer) -> Result<()> {
    let n = x.len_bytes() / 2;
    if !x.len_bytes().is_multiple_of(2) {
        return Err(Error::Backend(
            "relu_fp16: buffer size must be multiple of 2".into(),
        ));
    }

    let cached = backend.pipelines().get_or_create(OpKind::ReluFp16)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_info = [vk::DescriptorBufferInfo {
        buffer: x.vk_buffer(),
        offset: 0,
        range: vk::WHOLE_SIZE,
    }];
    let write = [vk::WriteDescriptorSet::default()
        .dst_set(desc_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&buf_info)];
    unsafe { backend.context().device().update_descriptor_sets(&write, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    #[repr(C)]
    struct PushRelu {
        n: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }
    let pc = PushRelu {
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), std::mem::size_of::<PushRelu>()) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(
            one_shot.cmd,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_push_constants(
            one_shot.cmd,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        let groups = (n as u32).div_ceil(64);
        device.cmd_dispatch(one_shot.cmd, groups, 1, 1);
    }

    one_shot.end()?;
    one_shot.submit()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// add_fp16
// ---------------------------------------------------------------------------

/// Compute y[i] = a[i] + b[i] (element-wise addition, F16 buffers).
///
/// All buffers must have the same size, a multiple of 2 bytes.
pub fn add_fp16(
    backend: &VulkanBackend,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    y: &mut VulkanBuffer,
) -> Result<()> {
    if a.len_bytes() != b.len_bytes() || a.len_bytes() != y.len_bytes() {
        return Err(Error::Backend("add_fp16: buffer size mismatch".into()));
    }
    let n = a.len_bytes() / 2;
    if !a.len_bytes().is_multiple_of(2) {
        return Err(Error::Backend(
            "add_fp16: buffer size must be multiple of 2".into(),
        ));
    }

    let cached = backend.pipelines().get_or_create(OpKind::AddFp16)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_infos = [
        vk::DescriptorBufferInfo {
            buffer: y.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: a.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: b.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[1..2]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[2..3]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    #[repr(C)]
    struct PushAdd {
        n: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }
    let pc = PushAdd {
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), std::mem::size_of::<PushAdd>()) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(
            one_shot.cmd,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_push_constants(
            one_shot.cmd,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        let groups = (n as u32).div_ceil(64);
        device.cmd_dispatch(one_shot.cmd, groups, 1, 1);
    }

    one_shot.end()?;
    one_shot.submit()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// gemm_f32_tiled
// ---------------------------------------------------------------------------

/// Compute C = A × B (row-major, no transpose) with tiled shared-memory blocking.
///
/// This is the optimized version of gemm_f32. Uses 16x16 tiles with K-blocking
/// for improved memory locality on Adreno A702.
///
/// * A: M×K
/// * B: K×N
/// * C: M×N
///
/// All buffers must be sized correctly (`M*K*4`, `K*N*4`, `M*N*4`).
///
/// Push constants: `{ M: u32, N: u32, K: u32, _: u32 }`.
pub fn gemm_f32_tiled(
    backend: &VulkanBackend,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    c: &mut VulkanBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if a.len_bytes() != m * k * 4 {
        return Err(Error::Backend(format!(
            "gemm_f32_tiled: A size mismatch: expected {} got {}",
            m * k * 4,
            a.len_bytes()
        )));
    }
    if b.len_bytes() != k * n * 4 {
        return Err(Error::Backend(format!(
            "gemm_f32_tiled: B size mismatch: expected {} got {}",
            k * n * 4,
            b.len_bytes()
        )));
    }
    if c.len_bytes() != m * n * 4 {
        return Err(Error::Backend(format!(
            "gemm_f32_tiled: C size mismatch: expected {} got {}",
            m * n * 4,
            c.len_bytes()
        )));
    }

    let cached = backend.pipelines().get_or_create(OpKind::GemmF32Tiled)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_infos = [
        vk::DescriptorBufferInfo {
            buffer: c.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: a.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: b.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[1..2]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[2..3]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    #[repr(C)]
    struct PushGemm {
        m: u32,
        n: u32,
        k: u32,
        _pad: u32,
    }
    let pc = PushGemm {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        _pad: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), std::mem::size_of::<PushGemm>()) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(
            one_shot.cmd,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_push_constants(
            one_shot.cmd,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        let groups_x = (n as u32).div_ceil(16);
        let groups_y = (m as u32).div_ceil(16);
        device.cmd_dispatch(one_shot.cmd, groups_x, groups_y, 1);
    }

    one_shot.end()?;
    one_shot.submit()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// gemm_fp16
// ---------------------------------------------------------------------------

/// Compute C = A × B (row-major, no transpose) with F16 buffers.
///
/// Uses F32 accumulator internally for precision (mixed-precision pattern).
///
/// * A: M×K (F16)
/// * B: K×N (F16)
/// * C: M×N (F16)
///
/// All buffers must be sized correctly (`M*K*2`, `K*N*2`, `M*N*2`).
///
/// Push constants: `{ M: u32, N: u32, K: u32, _: u32 }`.
pub fn gemm_fp16(
    backend: &VulkanBackend,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    c: &mut VulkanBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if a.len_bytes() != m * k * 2 {
        return Err(Error::Backend(format!(
            "gemm_fp16: A size mismatch: expected {} got {}",
            m * k * 2,
            a.len_bytes()
        )));
    }
    if b.len_bytes() != k * n * 2 {
        return Err(Error::Backend(format!(
            "gemm_fp16: B size mismatch: expected {} got {}",
            k * n * 2,
            b.len_bytes()
        )));
    }
    if c.len_bytes() != m * n * 2 {
        return Err(Error::Backend(format!(
            "gemm_fp16: C size mismatch: expected {} got {}",
            m * n * 2,
            c.len_bytes()
        )));
    }

    let cached = backend.pipelines().get_or_create(OpKind::GemmFp16)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_infos = [
        vk::DescriptorBufferInfo {
            buffer: c.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: a.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: b.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[1..2]),
        vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos[2..3]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    #[repr(C)]
    struct PushGemmFp16 {
        m: u32,
        n: u32,
        k: u32,
        _pad: u32,
    }
    let pc = PushGemmFp16 {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        _pad: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), std::mem::size_of::<PushGemmFp16>()) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(
            one_shot.cmd,
            vk::PipelineBindPoint::COMPUTE,
            cached.layout,
            0,
            &[desc_set],
            &[],
        );
        device.cmd_push_constants(
            one_shot.cmd,
            cached.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        let groups_x = (n as u32).div_ceil(16);
        let groups_y = (m as u32).div_ceil(16);
        device.cmd_dispatch(one_shot.cmd, groups_x, groups_y, 1);
    }

    one_shot.end()?;
    one_shot.submit()?;
    Ok(())
}

// ===========================================================================
// Convolution and pooling ops (task 003)
// ===========================================================================

/// 2D convolution in NHWC format.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32_nhwc(
    backend: &VulkanBackend,
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
    let expected_input = n * h_in * w_in * c_in * 4;
    let expected_kernel = k_h * k_w * c_in * c_out * 4;
    let expected_output = n * h_out * w_out * c_out * 4;

    if input.len_bytes() != expected_input {
        return Err(Error::Backend(format!("conv2d: input size mismatch: expected {expected_input}, got {}", input.len_bytes())));
    }
    if kernel.len_bytes() != expected_kernel {
        return Err(Error::Backend(format!("conv2d: kernel size mismatch: expected {expected_kernel}, got {}", kernel.len_bytes())));
    }
    if output.len_bytes() != expected_output {
        return Err(Error::Backend(format!("conv2d: output size mismatch: expected {expected_output}, got {}", output.len_bytes())));
    }

    let cached = backend.pipelines().get_or_create(OpKind::Conv2dF32Nhwc)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_infos = [
        vk::DescriptorBufferInfo { buffer: output.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: input.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: kernel.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
    ];
    let writes = [
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[1..2]),
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[2..3]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    #[repr(C)]
    struct PushConv { dims0: [u32; 4], dims1: [u32; 4], dims2: [u32; 4], dims3: [u32; 4] }
    let pc = PushConv {
        dims0: [h_in as u32, w_in as u32, c_in as u32, c_out as u32],
        dims1: [k_h as u32, k_w as u32, stride_h as u32, stride_w as u32],
        dims2: [pad_h as u32, pad_w as u32, h_out as u32, w_out as u32],
        dims3: [n as u32, 0, 0, 0],
    };
    let pc_bytes: &[u8] = unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 64) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.layout, 0, &[desc_set], &[]);
        device.cmd_push_constants(one_shot.cmd, cached.layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
        device.cmd_dispatch(one_shot.cmd, (w_out as u32).div_ceil(8), (h_out as u32).div_ceil(8), (n * c_out) as u32);
    }
    one_shot.end()?;
    one_shot.submit()
}

/// 2D convolution in NHWC format with FP16 input/output.
///
/// Uses FP16 for input/output and kernel buffers but accumulates in FP32.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_fp16_nhwc(
    backend: &VulkanBackend,
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
    // FP16 = 2 bytes per element
    let expected_input = n * h_in * w_in * c_in * 2;
    let expected_kernel = k_h * k_w * c_in * c_out * 2;
    let expected_output = n * h_out * w_out * c_out * 2;

    if input.len_bytes() != expected_input {
        return Err(Error::Backend(format!("conv2d_fp16: input size mismatch: expected {expected_input}, got {}", input.len_bytes())));
    }
    if kernel.len_bytes() != expected_kernel {
        return Err(Error::Backend(format!("conv2d_fp16: kernel size mismatch: expected {expected_kernel}, got {}", kernel.len_bytes())));
    }
    if output.len_bytes() != expected_output {
        return Err(Error::Backend(format!("conv2d_fp16: output size mismatch: expected {expected_output}, got {}", output.len_bytes())));
    }

    let cached = backend.pipelines().get_or_create(OpKind::Conv2dFp16Nhwc)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_infos = [
        vk::DescriptorBufferInfo { buffer: output.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: input.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: kernel.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
    ];
    let writes = [
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[1..2]),
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[2..3]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    #[repr(C)]
    struct PushConv { dims0: [u32; 4], dims1: [u32; 4], dims2: [u32; 4], dims3: [u32; 4] }
    let pc = PushConv {
        dims0: [h_in as u32, w_in as u32, c_in as u32, c_out as u32],
        dims1: [k_h as u32, k_w as u32, stride_h as u32, stride_w as u32],
        dims2: [pad_h as u32, pad_w as u32, h_out as u32, w_out as u32],
        dims3: [n as u32, 0, 0, 0],
    };
    let pc_bytes: &[u8] = unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 64) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.layout, 0, &[desc_set], &[]);
        device.cmd_push_constants(one_shot.cmd, cached.layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
        device.cmd_dispatch(one_shot.cmd, (w_out as u32).div_ceil(8), (h_out as u32).div_ceil(8), (n * c_out) as u32);
    }
    one_shot.end()?;
    one_shot.submit()
}

/// 2D max pooling in NHWC format.
#[allow(clippy::too_many_arguments)]
pub fn maxpool2d_f32(
    backend: &VulkanBackend,
    input: &VulkanBuffer,
    output: &mut VulkanBuffer,
    n: usize,
    h_in: usize,
    w_in: usize,
    c: usize,
    pool_h: usize,
    pool_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<()> {
    let h_out = (h_in - pool_h) / stride_h + 1;
    let w_out = (w_in - pool_w) / stride_w + 1;

    if input.len_bytes() != n * h_in * w_in * c * 4 { return Err(Error::Backend("maxpool2d: input size mismatch".into())); }
    if output.len_bytes() != n * h_out * w_out * c * 4 { return Err(Error::Backend("maxpool2d: output size mismatch".into())); }

    let cached = backend.pipelines().get_or_create(OpKind::Maxpool2dF32)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_infos = [
        vk::DescriptorBufferInfo { buffer: output.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: input.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
    ];
    let writes = [
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[1..2]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    #[repr(C)]
    struct PushPool { dims0: [u32; 4], dims1: [u32; 4], dims2: [u32; 4], dims3: [u32; 4] }
    let pc = PushPool {
        dims0: [h_in as u32, w_in as u32, c as u32, n as u32],
        dims1: [pool_h as u32, pool_w as u32, stride_h as u32, stride_w as u32],
        dims2: [h_out as u32, w_out as u32, 0, 0],
        dims3: [0, 0, 0, 0],
    };
    let pc_bytes: &[u8] = unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 64) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.layout, 0, &[desc_set], &[]);
        device.cmd_push_constants(one_shot.cmd, cached.layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
        device.cmd_dispatch(one_shot.cmd, (w_out as u32).div_ceil(8), (h_out as u32).div_ceil(8), (n * c) as u32);
    }
    one_shot.end()?;
    one_shot.submit()
}

/// Softmax along the last axis.
pub fn softmax_f32(
    backend: &VulkanBackend,
    input: &VulkanBuffer,
    output: &mut VulkanBuffer,
    rows: usize,
    n: usize,
) -> Result<()> {
    let expected = rows * n * 4;
    if input.len_bytes() != expected || output.len_bytes() != expected {
        return Err(Error::Backend("softmax: buffer size mismatch".into()));
    }

    let cached = backend.pipelines().get_or_create(OpKind::SoftmaxF32)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_infos = [
        vk::DescriptorBufferInfo { buffer: output.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: input.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
    ];
    let writes = [
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[1..2]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let one_shot = OneShot::new(backend.context().clone())?;
    one_shot.begin()?;

    #[repr(C)]
    struct PushSoftmax { n: u32, rows: u32, _p0: u32, _p1: u32 }
    let pc = PushSoftmax { n: n as u32, rows: rows as u32, _p0: 0, _p1: 0 };
    let pc_bytes: &[u8] = unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 16) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.layout, 0, &[desc_set], &[]);
        device.cmd_push_constants(one_shot.cmd, cached.layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
        device.cmd_dispatch(one_shot.cmd, 1, rows as u32, 1);
    }
    one_shot.end()?;
    one_shot.submit()
}
