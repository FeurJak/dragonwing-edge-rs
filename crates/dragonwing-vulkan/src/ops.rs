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
///
/// Optionally tracks a descriptor set (and the pool that owns it) so the
/// set is returned to the pool when this `OneShot` drops — i.e. after
/// `device_wait_idle()` confirms the GPU is no longer using it. Without
/// this freeing, the descriptor pool is exhausted after ~64 dispatches
/// (the pool's `max_sets` configured in `pipeline::PipelineCache::new`).
struct OneShot {
    ctx: Arc<Context>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    /// Optional descriptor set + the cache it was allocated from. Set
    /// via `attach_descriptor_set` after recording.
    desc: Option<(Arc<crate::pipeline::PipelineCache>, vk::DescriptorSet)>,
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

        Ok(Self {
            ctx,
            pool,
            cmd,
            desc: None,
        })
    }

    /// Register a descriptor set for cleanup when this `OneShot` drops.
    fn attach_descriptor_set(
        &mut self,
        cache: Arc<crate::pipeline::PipelineCache>,
        set: vk::DescriptorSet,
    ) {
        self.desc = Some((cache, set));
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
        // Wait for GPU to finish before destroying the command pool /
        // freeing the descriptor set.
        // SAFETY: device is valid, this is a best-effort wait.
        let _ = unsafe { self.ctx.device().device_wait_idle() };
        // Return the descriptor set to the pool *before* destroying the
        // command pool — Vulkan does not require any particular order, but
        // doing it first matches the allocation order.
        if let Some((cache, set)) = self.desc.take() {
            let _ = cache.free_descriptor_set(set);
        }
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
    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
    one_shot.begin()?;

    // 2D-wrap dispatch to avoid Adreno's maxComputeWorkGroupCount[0]=65535.
    let total_groups = (n as u64).div_ceil(64);
    let gx = total_groups.min(65535) as u32;
    let gy = total_groups.div_ceil(gx as u64) as u32;

    #[repr(C)]
    struct PushRelu {
        n: u32,
        gx_total: u32,
        _pad1: u32,
        _pad2: u32,
    }
    let pc = PushRelu {
        n: n as u32,
        gx_total: gx,
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
        device.cmd_dispatch(one_shot.cmd, gx, gy, 1);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
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

// ===========================================================================
// YOLO ops (Task 005)
// ===========================================================================

/// Element-wise sigmoid: y[i] = 1 / (1 + exp(-x[i])).
pub fn sigmoid_f32(
    backend: &VulkanBackend,
    input: &VulkanBuffer,
    output: &mut VulkanBuffer,
) -> Result<()> {
    let n = input.len_bytes() / 4;
    if output.len_bytes() != input.len_bytes() {
        return Err(Error::Backend("sigmoid: buffer size mismatch".into()));
    }

    let cached = backend.pipelines().get_or_create(OpKind::SigmoidF32)?;
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

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
    one_shot.begin()?;

    let total_groups = (n as u64).div_ceil(64);
    let gx = total_groups.min(65535) as u32;
    let gy = total_groups.div_ceil(gx as u64) as u32;

    #[repr(C)]
    struct PushSigmoid { n: u32, gx_total: u32, _p1: u32, _p2: u32 }
    let pc = PushSigmoid { n: n as u32, gx_total: gx, _p1: 0, _p2: 0 };
    let pc_bytes: &[u8] = unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 16) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.layout, 0, &[desc_set], &[]);
        device.cmd_push_constants(one_shot.cmd, cached.layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
        device.cmd_dispatch(one_shot.cmd, gx, gy, 1);
    }
    one_shot.end()?;
    one_shot.submit()
}

/// Element-wise multiplication: y[i] = a[i] * b[i].
pub fn mul_f32(
    backend: &VulkanBackend,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    output: &mut VulkanBuffer,
) -> Result<()> {
    let n = a.len_bytes() / 4;
    if a.len_bytes() != b.len_bytes() || a.len_bytes() != output.len_bytes() {
        return Err(Error::Backend("mul: buffer size mismatch".into()));
    }

    let cached = backend.pipelines().get_or_create(OpKind::MulF32)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    let buf_infos = [
        vk::DescriptorBufferInfo { buffer: output.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: a.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
        vk::DescriptorBufferInfo { buffer: b.vk_buffer(), offset: 0, range: vk::WHOLE_SIZE },
    ];
    let writes = [
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[0..1]),
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[1..2]),
        vk::WriteDescriptorSet::default().dst_set(desc_set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_infos[2..3]),
    ];
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
    one_shot.begin()?;

    let total_groups = (n as u64).div_ceil(64);
    let gx = total_groups.min(65535) as u32;
    let gy = total_groups.div_ceil(gx as u64) as u32;

    #[repr(C)]
    struct PushMul { n: u32, gx_total: u32, _p1: u32, _p2: u32 }
    let pc = PushMul { n: n as u32, gx_total: gx, _p1: 0, _p2: 0 };
    let pc_bytes: &[u8] = unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 16) };

    let device = backend.context().device();
    unsafe {
        device.cmd_bind_pipeline(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.pipeline);
        device.cmd_bind_descriptor_sets(one_shot.cmd, vk::PipelineBindPoint::COMPUTE, cached.layout, 0, &[desc_set], &[]);
        device.cmd_push_constants(one_shot.cmd, cached.layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);
        device.cmd_dispatch(one_shot.cmd, gx, gy, 1);
    }
    one_shot.end()?;
    one_shot.submit()
}

// ===========================================================================
// INT8 packed ops (Task 007 — wires Task 006's shaders into the backend)
// ===========================================================================
//
// All INT8 ops use UINT32-packed storage (4× INT8 per UINT32, little-endian)
// because Adreno A702 lacks `VK_KHR_8bit_storage`. The Rust-side buffers are
// `VulkanBuffer`s allocated by the caller; their `.len_bytes()` is expected
// to be `(n / 4) * 4` for packed I8 tensors and `n * 4` for INT32
// accumulators / scale-bearing F32 tensors.
//
// The graph runtime is responsible for ensuring `n` (the element count) is
// a multiple of 4 — the shaders early-out on out-of-range invocations but
// will still corrupt the tail if a non-multiple-of-4 length is passed.

/// Bind helper: 1 mut output (binding 0) + N readonly inputs.
fn write_descriptors_n(
    backend: &VulkanBackend,
    desc_set: vk::DescriptorSet,
    output: vk::Buffer,
    inputs: &[vk::Buffer],
) {
    let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(1 + inputs.len());
    buf_infos.push(vk::DescriptorBufferInfo {
        buffer: output,
        offset: 0,
        range: vk::WHOLE_SIZE,
    });
    for &b in inputs {
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
    unsafe { backend.context().device().update_descriptor_sets(&writes, &[]) };
}

/// INT8 GEMM: C[i32, m×n] = A[i8 packed, m×k] · B[i8 packed, k×n].
///
/// `k` must be a multiple of 4 (UINT32 packing constraint).
///
/// Buffer sizes:
///   - `a_packed`: `m * k / 4` UINT32 = `m * k` bytes
///   - `b_packed`: `k * n / 4` UINT32 = `k * n` bytes
///   - `c_i32`:    `m * n` INT32 = `m * n * 4` bytes
pub fn gemm_i8_packed(
    backend: &VulkanBackend,
    a_packed: &VulkanBuffer,
    b_packed: &VulkanBuffer,
    c_i32: &mut VulkanBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if !k.is_multiple_of(4) {
        return Err(Error::Backend(format!(
            "gemm_i8_packed: k={k} must be multiple of 4"
        )));
    }
    if a_packed.len_bytes() != m * k {
        return Err(Error::Backend(format!(
            "gemm_i8_packed: A size mismatch: expected {} bytes (m*k packed), got {}",
            m * k,
            a_packed.len_bytes()
        )));
    }
    if b_packed.len_bytes() != k * n {
        return Err(Error::Backend(format!(
            "gemm_i8_packed: B size mismatch: expected {} bytes (k*n packed), got {}",
            k * n,
            b_packed.len_bytes()
        )));
    }
    if c_i32.len_bytes() != m * n * 4 {
        return Err(Error::Backend(format!(
            "gemm_i8_packed: C size mismatch: expected {} bytes (m*n*4 int32), got {}",
            m * n * 4,
            c_i32.len_bytes()
        )));
    }

    let cached = backend.pipelines().get_or_create(OpKind::GemmI8Packed)?;
    let desc_set = backend
        .pipelines()
        .allocate_descriptor_set(cached.descriptor_set_layout)?;
    write_descriptors_n(
        backend,
        desc_set,
        c_i32.vk_buffer(),
        &[a_packed.vk_buffer(), b_packed.vk_buffer()],
    );

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
    one_shot.begin()?;

    #[repr(C)]
    struct PushGemmI8 {
        m: u32,
        n: u32,
        k: u32,
        _pad: u32,
    }
    let pc = PushGemmI8 {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        _pad: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 16) };

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
        // Workgroup 8×8; dispatch ceil(n/8) × ceil(m/8) × 1.
        let gx = (n as u32).div_ceil(8);
        let gy = (m as u32).div_ceil(8);
        device.cmd_dispatch(one_shot.cmd, gx, gy, 1);
    }
    one_shot.end()?;
    one_shot.submit()
}

/// INT8 2D convolution in NHWC layout with UINT32 channel packing.
///
/// `c_in` and `c_out` must be multiples of 4 (channel packing).
/// Output is INT32 accumulator; caller is responsible for requantization.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_i8_nhwc_packed(
    backend: &VulkanBackend,
    input_packed: &VulkanBuffer,
    kernel_packed: &VulkanBuffer,
    output_i32: &mut VulkanBuffer,
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
    if !c_in.is_multiple_of(4) {
        return Err(Error::Backend(format!(
            "conv2d_i8: c_in={c_in} must be multiple of 4"
        )));
    }
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;
    let expected_input = n * h_in * w_in * c_in; // packed: same total bytes
    let expected_kernel = k_h * k_w * c_in * c_out;
    let expected_output = n * h_out * w_out * c_out * 4;
    if input_packed.len_bytes() != expected_input {
        return Err(Error::Backend(format!(
            "conv2d_i8: input mismatch: expected {expected_input}, got {}",
            input_packed.len_bytes()
        )));
    }
    if kernel_packed.len_bytes() != expected_kernel {
        return Err(Error::Backend(format!(
            "conv2d_i8: kernel mismatch: expected {expected_kernel}, got {}",
            kernel_packed.len_bytes()
        )));
    }
    if output_i32.len_bytes() != expected_output {
        return Err(Error::Backend(format!(
            "conv2d_i8: output mismatch: expected {expected_output}, got {}",
            output_i32.len_bytes()
        )));
    }

    let cached = backend.pipelines().get_or_create(OpKind::Conv2dI8NhwcPacked)?;
    let desc_set = backend
        .pipelines()
        .allocate_descriptor_set(cached.descriptor_set_layout)?;
    write_descriptors_n(
        backend,
        desc_set,
        output_i32.vk_buffer(),
        &[input_packed.vk_buffer(), kernel_packed.vk_buffer()],
    );

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
    one_shot.begin()?;

    #[repr(C)]
    struct PushConvI8 {
        dims0: [u32; 4],
        dims1: [u32; 4],
        dims2: [u32; 4],
        dims3: [u32; 4],
    }
    let pc = PushConvI8 {
        dims0: [h_in as u32, w_in as u32, c_in as u32, c_out as u32],
        dims1: [k_h as u32, k_w as u32, stride_h as u32, stride_w as u32],
        dims2: [pad_h as u32, pad_w as u32, h_out as u32, w_out as u32],
        dims3: [n as u32, (c_in / 4) as u32, 0, 0],
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 64) };

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
        device.cmd_dispatch(
            one_shot.cmd,
            (w_out as u32).div_ceil(8),
            (h_out as u32).div_ceil(8),
            (n * c_out) as u32,
        );
    }
    one_shot.end()?;
    one_shot.submit()
}

/// Internal helper for 16-byte `{ n: u32, scale: f32, _pad0, _pad1 }` ops.
///
/// Wraps the boilerplate for `requantize`, `quantize`, `dequantize`, `relu`
/// (when scale is unused). `elements_per_group_x` is the number of elements
/// covered by each workgroup (= local_size_x × elements_per_thread). For
/// packed-output ops it's 256 (64 threads × 4 elements/thread); for
/// unpacked-output ops it's 64.
fn dispatch_simple_i8(
    backend: &VulkanBackend,
    op: OpKind,
    output: vk::Buffer,
    inputs: &[vk::Buffer],
    n: usize,
    scale: f32,
    elements_per_group_x: u32,
) -> Result<()> {
    let cached = backend.pipelines().get_or_create(op)?;
    let desc_set = backend
        .pipelines()
        .allocate_descriptor_set(cached.descriptor_set_layout)?;
    write_descriptors_n(backend, desc_set, output, inputs);

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
    one_shot.begin()?;

    #[repr(C)]
    struct PushScalar {
        n: u32,
        scale: f32,
        _pad0: u32,
        _pad1: u32,
    }
    let pc = PushScalar {
        n: n as u32,
        scale,
        _pad0: 0,
        _pad1: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 16) };

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
        device.cmd_dispatch(one_shot.cmd, (n as u32).div_ceil(elements_per_group_x), 1, 1);
    }
    one_shot.end()?;
    one_shot.submit()
}

/// Requantize: `output_i8[i] = clamp(round(input_i32[i] * requant_scale), -128, 127)`.
///
/// `n` must be a multiple of 4 (output is packed UINT32).
/// Input is INT32 (n*4 bytes); output is packed I8 (n bytes).
pub fn requantize_i32_to_i8_packed(
    backend: &VulkanBackend,
    input_i32: &VulkanBuffer,
    output_packed: &mut VulkanBuffer,
    requant_scale: f32,
) -> Result<()> {
    let n = output_packed.len_bytes(); // 1 byte per i8 → element count
    if !n.is_multiple_of(4) {
        return Err(Error::Backend(format!(
            "requantize: n={n} must be multiple of 4"
        )));
    }
    if input_i32.len_bytes() != n * 4 {
        return Err(Error::Backend(format!(
            "requantize: input must be {n}*4 bytes, got {}",
            input_i32.len_bytes()
        )));
    }
    dispatch_simple_i8(
        backend,
        OpKind::RequantizeI32ToI8,
        output_packed.vk_buffer(),
        &[input_i32.vk_buffer()],
        n,
        requant_scale,
        // Each invocation packs 4 elements → workgroup of 64 covers 256.
        256,
    )
}

/// Quantize: `output_i8[i] = clamp(round(input_f32[i] / scale), -128, 127)`.
///
/// `n` must be a multiple of 4.
/// Input is F32 (n*4 bytes); output is packed I8 (n bytes).
pub fn quantize_f32_to_i8_packed(
    backend: &VulkanBackend,
    input_f32: &VulkanBuffer,
    output_packed: &mut VulkanBuffer,
    scale: f32,
) -> Result<()> {
    let n = output_packed.len_bytes();
    if !n.is_multiple_of(4) {
        return Err(Error::Backend(format!(
            "quantize: n={n} must be multiple of 4"
        )));
    }
    if input_f32.len_bytes() != n * 4 {
        return Err(Error::Backend(format!(
            "quantize: input must be {n}*4 bytes, got {}",
            input_f32.len_bytes()
        )));
    }
    if scale == 0.0 {
        return Err(Error::Backend("quantize: scale must be nonzero".into()));
    }
    let inv_scale = 1.0 / scale;
    dispatch_simple_i8(
        backend,
        OpKind::QuantizeF32ToI8,
        output_packed.vk_buffer(),
        &[input_f32.vk_buffer()],
        n,
        inv_scale,
        256,
    )
}

/// Dequantize: `output_f32[i] = input_i8[i] * scale`.
///
/// `n` must be a multiple of 4 (input is packed UINT32).
/// Input is packed I8 (n bytes); output is F32 (n*4 bytes).
pub fn dequantize_i8_packed_to_f32(
    backend: &VulkanBackend,
    input_packed: &VulkanBuffer,
    output_f32: &mut VulkanBuffer,
    scale: f32,
) -> Result<()> {
    let n = input_packed.len_bytes();
    if !n.is_multiple_of(4) {
        return Err(Error::Backend(format!(
            "dequantize: n={n} must be multiple of 4"
        )));
    }
    if output_f32.len_bytes() != n * 4 {
        return Err(Error::Backend(format!(
            "dequantize: output must be {n}*4 bytes, got {}",
            output_f32.len_bytes()
        )));
    }
    // dequantize shader has 1 element per invocation, so workgroup_x of 64
    // covers 64 elements.
    dispatch_simple_i8(
        backend,
        OpKind::DequantizeI8ToF32,
        output_f32.vk_buffer(),
        &[input_packed.vk_buffer()],
        n,
        scale,
        64,
    )
}

/// INT8 element-wise add with scale adjustment.
///
/// `output_i8[i] = clamp(round(a_i8[i] * scale_a_over_y + b_i8[i] * scale_b_over_y), -128, 127)`.
///
/// All three buffers are packed I8 (n bytes each).
pub fn add_i8_packed(
    backend: &VulkanBackend,
    a_packed: &VulkanBuffer,
    b_packed: &VulkanBuffer,
    output_packed: &mut VulkanBuffer,
    scale_a_over_y: f32,
    scale_b_over_y: f32,
) -> Result<()> {
    let n = output_packed.len_bytes();
    if !n.is_multiple_of(4) {
        return Err(Error::Backend(format!("add_i8: n={n} must be multiple of 4")));
    }
    if a_packed.len_bytes() != n || b_packed.len_bytes() != n {
        return Err(Error::Backend("add_i8: input size mismatch".into()));
    }

    let cached = backend.pipelines().get_or_create(OpKind::AddI8Packed)?;
    let desc_set = backend
        .pipelines()
        .allocate_descriptor_set(cached.descriptor_set_layout)?;
    write_descriptors_n(
        backend,
        desc_set,
        output_packed.vk_buffer(),
        &[a_packed.vk_buffer(), b_packed.vk_buffer()],
    );

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
    one_shot.begin()?;

    #[repr(C)]
    struct PushAddI8 {
        n: u32,
        scale_a_over_y: f32,
        scale_b_over_y: f32,
        _pad: u32,
    }
    let pc = PushAddI8 {
        n: n as u32,
        scale_a_over_y,
        scale_b_over_y,
        _pad: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 16) };

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
        device.cmd_dispatch(one_shot.cmd, (n as u32).div_ceil(256), 1, 1);
    }
    one_shot.end()?;
    one_shot.submit()
}

/// INT8 ReLU on packed storage. `output_i8[i] = max(0, input_i8[i])`.
///
/// `n` must be a multiple of 4. Out-of-place (separate input/output buffers).
pub fn relu_i8_packed(
    backend: &VulkanBackend,
    input_packed: &VulkanBuffer,
    output_packed: &mut VulkanBuffer,
) -> Result<()> {
    let n = output_packed.len_bytes();
    if !n.is_multiple_of(4) {
        return Err(Error::Backend(format!("relu_i8: n={n} must be multiple of 4")));
    }
    if input_packed.len_bytes() != n {
        return Err(Error::Backend("relu_i8: input size mismatch".into()));
    }
    // ReLU has scale param ignored (0.0).
    dispatch_simple_i8(
        backend,
        OpKind::ReluI8Packed,
        output_packed.vk_buffer(),
        &[input_packed.vk_buffer()],
        n,
        0.0,
        256,
    )
}

// ===========================================================================
// Fused kernels (Task 007)
// ===========================================================================

/// Fused SiLU activation: `y[i] = x[i] * sigmoid(x[i])`.
///
/// Replaces the `sigmoid_f32 + mul_f32` pair with a single shader. Saves
/// one full-tensor memory pass — the dominant cost for elementwise YOLO
/// activations on Adreno A702.
///
/// Buffers must be the same size, a multiple of 4 bytes (F32 elements).
pub fn silu_f32(
    backend: &VulkanBackend,
    input: &VulkanBuffer,
    output: &mut VulkanBuffer,
) -> Result<()> {
    let n = input.len_bytes() / 4;
    if input.len_bytes() != output.len_bytes() {
        return Err(Error::Backend("silu_f32: buffer size mismatch".into()));
    }
    if !input.len_bytes().is_multiple_of(4) {
        return Err(Error::Backend(
            "silu_f32: buffer size must be multiple of 4".into(),
        ));
    }

    let cached = backend.pipelines().get_or_create(OpKind::SiluF32)?;
    let desc_set = backend
        .pipelines()
        .allocate_descriptor_set(cached.descriptor_set_layout)?;
    write_descriptors_n(backend, desc_set, output.vk_buffer(), &[input.vk_buffer()]);

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
    one_shot.begin()?;

    let total_groups = (n as u64).div_ceil(64);
    let gx = total_groups.min(65535) as u32;
    let gy = total_groups.div_ceil(gx as u64) as u32;

    #[repr(C)]
    struct PushSilu {
        n: u32,
        gx_total: u32,
        _p1: u32,
        _p2: u32,
    }
    let pc = PushSilu {
        n: n as u32,
        gx_total: gx,
        _p1: 0,
        _p2: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 16) };

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
        device.cmd_dispatch(one_shot.cmd, gx, gy, 1);
    }
    one_shot.end()?;
    one_shot.submit()
}

/// Fused INT8 Conv2D + Requantize + ReLU (NHWC, channel-packed).
///
/// One shader produces packed-I8 output directly from packed-I8 inputs and
/// weights. Each invocation produces 4 output channels (packed into a
/// UINT32), so `c_out` must be a multiple of 4. `c_in` must also be a
/// multiple of 4 (input packing).
///
/// `do_relu`: if `true`, applies ReLU (clip to ≥ 0) after requantize.
/// Useful when followed by a layer that expects unsigned-only activations;
/// `false` keeps the full INT8 range.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_requant_relu_i8_packed(
    backend: &VulkanBackend,
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
    if !c_in.is_multiple_of(4) {
        return Err(Error::Backend(format!(
            "conv_fused: c_in={c_in} must be multiple of 4"
        )));
    }
    if !c_out.is_multiple_of(4) {
        return Err(Error::Backend(format!(
            "conv_fused: c_out={c_out} must be multiple of 4 (output is packed)"
        )));
    }
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;
    let expected_input = n * h_in * w_in * c_in;
    let expected_kernel = k_h * k_w * c_in * c_out;
    let expected_output = n * h_out * w_out * c_out; // packed: 1 byte per element
    if input_packed.len_bytes() != expected_input {
        return Err(Error::Backend(format!(
            "conv_fused: input mismatch: expected {expected_input}, got {}",
            input_packed.len_bytes()
        )));
    }
    if kernel_packed.len_bytes() != expected_kernel {
        return Err(Error::Backend(format!(
            "conv_fused: kernel mismatch: expected {expected_kernel}, got {}",
            kernel_packed.len_bytes()
        )));
    }
    if output_packed.len_bytes() != expected_output {
        return Err(Error::Backend(format!(
            "conv_fused: output mismatch: expected {expected_output}, got {}",
            output_packed.len_bytes()
        )));
    }

    let cached = backend
        .pipelines()
        .get_or_create(OpKind::Conv2dRequantReluI8Packed)?;
    let desc_set = backend
        .pipelines()
        .allocate_descriptor_set(cached.descriptor_set_layout)?;
    write_descriptors_n(
        backend,
        desc_set,
        output_packed.vk_buffer(),
        &[input_packed.vk_buffer(), kernel_packed.vk_buffer()],
    );

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
    one_shot.begin()?;

    #[repr(C)]
    struct PushConvFused {
        dims0: [u32; 4],
        dims1: [u32; 4],
        dims2: [u32; 4],
        dims3: [u32; 4],
        requant_scale: f32,
        do_relu: u32,
        _pad0: u32,
        _pad1: u32,
    }
    // Size = 64 + 16 = 80 bytes. Matches OpKind::Conv2dRequantReluI8Packed
    // push_constant_size() above.
    let pc = PushConvFused {
        dims0: [h_in as u32, w_in as u32, c_in as u32, c_out as u32],
        dims1: [k_h as u32, k_w as u32, stride_h as u32, stride_w as u32],
        dims2: [pad_h as u32, pad_w as u32, h_out as u32, w_out as u32],
        dims3: [n as u32, (c_in / 4) as u32, (c_out / 4) as u32, 0],
        requant_scale,
        do_relu: if do_relu { 1 } else { 0 },
        _pad0: 0,
        _pad1: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 80) };

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
        device.cmd_dispatch(
            one_shot.cmd,
            (w_out as u32).div_ceil(8),
            (h_out as u32).div_ceil(8),
            (n * (c_out / 4)) as u32,
        );
    }
    one_shot.end()?;
    one_shot.submit()
}

/// In-place broadcasting bias add over an NHWC tensor.
///
/// Computes `y[i] += bias[i % c_out]` for `i in 0..n_elem`, where `n_elem`
/// is the full element count of `y` (typically `N * H * W * C`) and
/// `c_out` is the per-pixel channel count.
///
/// Used to add ONNX Conv/Gemm biases after the corresponding F32 op
/// (which has no bias input on its shader binding). For Gemm with output
/// shape `[M, N]`, pass `c_out = N` and `n_elem = M * N`.
///
/// Buffer sizes:
///   - `y_inout`: `n_elem * 4` bytes (F32).
///   - `bias`:    `c_out * 4` bytes (F32).
pub fn bias_add_f32_nhwc(
    backend: &VulkanBackend,
    y_inout: &mut VulkanBuffer,
    bias: &VulkanBuffer,
    c_out: usize,
) -> Result<()> {
    if !y_inout.len_bytes().is_multiple_of(4) {
        return Err(Error::Backend(
            "bias_add_f32_nhwc: y size must be multiple of 4".into(),
        ));
    }
    if !bias.len_bytes().is_multiple_of(4) {
        return Err(Error::Backend(
            "bias_add_f32_nhwc: bias size must be multiple of 4".into(),
        ));
    }
    let n = y_inout.len_bytes() / 4;
    let bias_n = bias.len_bytes() / 4;
    if bias_n != c_out {
        return Err(Error::Backend(format!(
            "bias_add_f32_nhwc: bias has {bias_n} elements, expected {c_out}"
        )));
    }
    if c_out == 0 || n == 0 {
        return Ok(());
    }
    if n % c_out != 0 {
        return Err(Error::Backend(format!(
            "bias_add_f32_nhwc: total elements {n} not divisible by c_out {c_out}"
        )));
    }

    let cached = backend.pipelines().get_or_create(OpKind::BiasAddF32Nhwc)?;
    let desc_set = backend
        .pipelines()
        .allocate_descriptor_set(cached.descriptor_set_layout)?;
    write_descriptors_n(backend, desc_set, y_inout.vk_buffer(), &[bias.vk_buffer()]);

    let mut one_shot = OneShot::new(backend.context().clone())?;
    one_shot.attach_descriptor_set(backend.pipelines().clone(), desc_set);
    one_shot.begin()?;

    // Adreno A702 / Turnip limits maxComputeWorkGroupCount[0] to 65535.
    // Our shader uses a 64-wide workgroup, so a single-axis dispatch tops
    // out at 65535*64 = ~4.2M elements. For larger tensors (any YOLO
    // conv output) we wrap into 2D: gx covers 65535 workgroups, gy
    // covers the rest.
    let total_groups = (n as u64).div_ceil(64);
    let gx = total_groups.min(65535) as u32;
    let gy = total_groups.div_ceil(gx as u64) as u32;

    #[repr(C)]
    struct PushBiasAdd {
        n: u32,
        c_out: u32,
        gx_total: u32,
        _p1: u32,
    }
    let pc = PushBiasAdd {
        n: n as u32,
        c_out: c_out as u32,
        gx_total: gx,
        _p1: 0,
    };
    let pc_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts((&raw const pc).cast::<u8>(), 16) };

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
        device.cmd_dispatch(one_shot.cmd, gx, gy, 1);
    }
    one_shot.end()?;
    one_shot.submit()
}
