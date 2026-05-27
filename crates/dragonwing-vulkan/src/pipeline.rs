//! Compute pipeline and descriptor management.
//!
//! # Design
//!
//! `PipelineCache` owns:
//!
//! * A single `VkPipelineCache` (serialisation not implemented yet).
//! * A `VkDescriptorPool` sized for the maximum concurrent descriptors
//!   we expect (conservative: 64 sets × 4 bindings).
//! * One `VkDescriptorSetLayout` per op signature (fill=1 SSBO,
//!   axpy/relu=2 SSBOs, gemm=3 SSBOs).
//! * One `VkPipelineLayout` per descriptor-set layout.
//! * One `VkPipeline` per shader (lazily created on first use).
//!
//! Each op module calls [`PipelineCache::get_or_create`] with the op
//! name (e.g. `"fill_f32"`) and receives handles to the pipeline and
//! layout. The op module then allocates a descriptor set, writes the
//! buffer bindings, records the dispatch, and submits.
//!
//! # Thread safety
//!
//! `PipelineCache` is internally synchronised via `Mutex`. Multiple
//! threads can call `get_or_create` concurrently; only one will actually
//! compile the pipeline.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ash::vk;
use dragonwing_core::Result;

use crate::context::Context;
use crate::error::vk_err;

/// Identifies a compute pipeline by op name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// `fill_f32.spv` — 1 SSBO (output)
    FillF32,
    /// `axpy_f32.spv` — 2 SSBOs (x, y inout)
    AxpyF32,
    /// `relu_f32.spv` — 1 SSBO (inout)
    ReluF32,
    /// `gemm_f32.spv` — 3 SSBOs (A, B, C)
    GemmF32,
}

impl OpKind {
    /// Number of storage-buffer bindings required by this op.
    pub const fn binding_count(self) -> u32 {
        match self {
            OpKind::FillF32 => 1,
            OpKind::AxpyF32 => 2,
            OpKind::ReluF32 => 1,
            OpKind::GemmF32 => 3,
        }
    }

    /// SPIR-V blob for this op (embedded via `dragonwing_shaders`).
    pub fn spirv(self) -> &'static [u8] {
        match self {
            OpKind::FillF32 => dragonwing_shaders::FILL_F32,
            OpKind::AxpyF32 => dragonwing_shaders::AXPY_F32,
            OpKind::ReluF32 => dragonwing_shaders::RELU_F32,
            OpKind::GemmF32 => dragonwing_shaders::GEMM_F32_NAIVE,
        }
    }

    /// Push-constant size in bytes. Must match the shader's push_constant layout.
    pub const fn push_constant_size(self) -> u32 {
        match self {
            // fill_f32: { n: u32, v: f32, _pad0: f32, _pad1: f32 } = 16 bytes
            OpKind::FillF32 => 16,
            // axpy_f32: { n: u32, alpha: f32, _pad0: f32, _pad1: f32 } = 16 bytes
            OpKind::AxpyF32 => 16,
            // relu_f32: { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 } = 16 bytes
            OpKind::ReluF32 => 16,
            // gemm_f32: { M: u32, N: u32, K: u32, _pad: u32 } = 16 bytes
            OpKind::GemmF32 => 16,
        }
    }
}

/// Cached pipeline + layout for a single op.
#[derive(Debug, Clone, Copy)]
pub struct CachedPipeline {
    /// The compiled compute pipeline.
    pub pipeline: vk::Pipeline,
    /// The pipeline layout.
    pub layout: vk::PipelineLayout,
    /// The descriptor set layout.
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

/// Shared cache of compute pipelines.
pub struct PipelineCache {
    ctx: Arc<Context>,
    vk_cache: vk::PipelineCache,
    descriptor_pool: vk::DescriptorPool,
    inner: Mutex<HashMap<OpKind, CachedPipeline>>,
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache")
            .field("ctx", &self.ctx)
            .finish_non_exhaustive()
    }
}

impl PipelineCache {
    /// Create a new cache. Allocates the descriptor pool up-front.
    pub fn new(ctx: Arc<Context>) -> Result<Self> {
        // Create VkPipelineCache (empty, no disk serialisation yet).
        let cache_info = vk::PipelineCacheCreateInfo::default();
        // SAFETY: spec-compliant struct.
        let vk_cache = unsafe { ctx.device().create_pipeline_cache(&cache_info, None) }
            .map_err(|r| vk_err("create_pipeline_cache", r))?;

        // Create descriptor pool.
        // Conservative sizing: 64 sets, each with up to 4 SSBOs.
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 256,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(64)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        // SAFETY: spec-compliant struct.
        let descriptor_pool = unsafe { ctx.device().create_descriptor_pool(&pool_info, None) }
            .map_err(|r| vk_err("create_descriptor_pool", r))?;

        Ok(Self {
            ctx,
            vk_cache,
            descriptor_pool,
            inner: Mutex::new(HashMap::new()),
        })
    }

    /// Get or create the pipeline for `op`. Returns references valid for
    /// the lifetime of the cache.
    pub fn get_or_create(&self, op: OpKind) -> Result<CachedPipeline> {
        let mut map = self.inner.lock().expect("pipeline cache mutex poisoned");
        if let Some(cached) = map.get(&op) {
            return Ok(CachedPipeline {
                pipeline: cached.pipeline,
                layout: cached.layout,
                descriptor_set_layout: cached.descriptor_set_layout,
            });
        }

        // Create descriptor set layout.
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..op.binding_count())
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: spec-compliant struct.
        let descriptor_set_layout =
            unsafe { self.ctx.device().create_descriptor_set_layout(&ds_layout_info, None) }
                .map_err(|r| vk_err("create_descriptor_set_layout", r))?;

        // Create pipeline layout with push constants.
        let push_range = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: op.push_constant_size(),
        }];
        let set_layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);
        // SAFETY: spec-compliant struct.
        let layout = unsafe { self.ctx.device().create_pipeline_layout(&layout_info, None) }
            .map_err(|r| {
                unsafe { self.ctx.device().destroy_descriptor_set_layout(descriptor_set_layout, None) };
                vk_err("create_pipeline_layout", r)
            })?;

        // Create shader module from SPIR-V.
        let spirv = op.spirv();
        // ash expects &[u32]; SPIR-V is little-endian u32 words.
        // SAFETY: dragonwing_shaders guarantees 4-byte alignment and valid SPIR-V.
        let code: &[u32] = unsafe {
            std::slice::from_raw_parts(spirv.as_ptr().cast::<u32>(), spirv.len() / 4)
        };
        let module_info = vk::ShaderModuleCreateInfo::default().code(code);
        let shader_module = unsafe { self.ctx.device().create_shader_module(&module_info, None) }
            .map_err(|r| {
                unsafe {
                    self.ctx.device().destroy_pipeline_layout(layout, None);
                    self.ctx.device().destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                vk_err("create_shader_module", r)
            })?;

        // Create compute pipeline.
        let entry_name = c"main";
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_name);
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout)];

        // SAFETY: spec-compliant structs; shader module valid.
        let pipelines = unsafe {
            self.ctx
                .device()
                .create_compute_pipelines(self.vk_cache, &pipeline_info, None)
        }
        .map_err(|(_, r)| {
            unsafe {
                self.ctx.device().destroy_shader_module(shader_module, None);
                self.ctx.device().destroy_pipeline_layout(layout, None);
                self.ctx.device().destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            vk_err("create_compute_pipelines", r)
        })?;
        let pipeline = pipelines[0];

        // Shader module no longer needed after pipeline creation.
        // SAFETY: pipeline created, module can be destroyed.
        unsafe { self.ctx.device().destroy_shader_module(shader_module, None) };

        let cached = CachedPipeline {
            pipeline,
            layout,
            descriptor_set_layout,
        };
        map.insert(op, CachedPipeline {
            pipeline,
            layout,
            descriptor_set_layout,
        });

        Ok(cached)
    }

    /// Allocate a descriptor set for the given layout.
    pub fn allocate_descriptor_set(
        &self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        // SAFETY: spec-compliant struct.
        let sets = unsafe { self.ctx.device().allocate_descriptor_sets(&alloc_info) }
            .map_err(|r| vk_err("allocate_descriptor_sets", r))?;
        Ok(sets[0])
    }

    /// Free a descriptor set back to the pool.
    pub fn free_descriptor_set(&self, set: vk::DescriptorSet) -> Result<()> {
        // SAFETY: set was allocated from our pool.
        unsafe {
            self.ctx
                .device()
                .free_descriptor_sets(self.descriptor_pool, &[set])
        }
        .map_err(|r| vk_err("free_descriptor_sets", r))
    }

    /// Borrow the context.
    pub fn context(&self) -> &Arc<Context> {
        &self.ctx
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        // SAFETY: exclusive access via Drop. Destroy pipelines, layouts,
        // descriptor set layouts, pool, cache.
        unsafe {
            let map = self.inner.get_mut().expect("mutex poisoned in drop");
            for (_, cached) in map.drain() {
                self.ctx.device().destroy_pipeline(cached.pipeline, None);
                self.ctx.device().destroy_pipeline_layout(cached.layout, None);
                self.ctx
                    .device()
                    .destroy_descriptor_set_layout(cached.descriptor_set_layout, None);
            }
            self.ctx
                .device()
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.ctx.device().destroy_pipeline_cache(self.vk_cache, None);
        }
    }
}
