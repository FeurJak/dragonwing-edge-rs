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
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ash::vk;
use dragonwing_core::Result;

use crate::context::Context;
use crate::error::vk_err;

/// Identifies a compute pipeline by op name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    // -----------------------------------------------------------------------
    // F32 ops
    // -----------------------------------------------------------------------
    /// `fill_f32.spv` — 1 SSBO (output)
    FillF32,
    /// `axpy_f32.spv` — 2 SSBOs (y inout, x readonly)
    AxpyF32,
    /// `relu_f32.spv` — 1 SSBO (inout)
    ReluF32,
    /// `add_f32.spv` — 3 SSBOs (y output, a readonly, b readonly)
    AddF32,
    /// `gemm_f32_naive.spv` — 3 SSBOs (C output, A, B), naive reference
    GemmF32,
    /// `gemm_f32_tiled.spv` — 3 SSBOs (C output, A, B), tiled with shared memory
    GemmF32Tiled,
    /// `conv2d_f32_nhwc.spv` — 3 SSBOs (output, input, kernel)
    Conv2dF32Nhwc,
    /// `conv2d_fp16_nhwc.spv` — 3 SSBOs (output, input, kernel), FP16 I/O
    Conv2dFp16Nhwc,
    /// `maxpool2d_f32.spv` — 2 SSBOs (output, input)
    Maxpool2dF32,
    /// `softmax_f32.spv` — 2 SSBOs (output, input)
    SoftmaxF32,

    // -----------------------------------------------------------------------
    // FP16 ops
    // -----------------------------------------------------------------------
    /// `fill_fp16.spv` — 1 SSBO (output, F16)
    FillFp16,
    /// `axpy_fp16.spv` — 2 SSBOs (y inout, x readonly, F16)
    AxpyFp16,
    /// `relu_fp16.spv` — 1 SSBO (inout, F16)
    ReluFp16,
    /// `add_fp16.spv` — 3 SSBOs (y output, a readonly, b readonly, F16)
    AddFp16,
    /// `gemm_fp16.spv` — 3 SSBOs (C output, A, B), F16 with F32 accumulator
    GemmFp16,
}

impl OpKind {
    /// Number of storage-buffer bindings required by this op.
    pub const fn binding_count(self) -> u32 {
        match self {
            OpKind::FillF32 | OpKind::FillFp16 => 1,
            OpKind::AxpyF32 | OpKind::AxpyFp16 => 2,
            OpKind::ReluF32 | OpKind::ReluFp16 => 1,
            OpKind::AddF32 | OpKind::AddFp16 => 3,
            OpKind::GemmF32 | OpKind::GemmF32Tiled | OpKind::GemmFp16 => 3,
            OpKind::Conv2dF32Nhwc | OpKind::Conv2dFp16Nhwc => 3,
            OpKind::Maxpool2dF32 | OpKind::SoftmaxF32 => 2,
        }
    }

    /// SPIR-V blob for this op (embedded via `dragonwing_shaders`).
    pub fn spirv(self) -> &'static [u8] {
        match self {
            OpKind::FillF32 => dragonwing_shaders::FILL_F32,
            OpKind::AxpyF32 => dragonwing_shaders::AXPY_F32,
            OpKind::ReluF32 => dragonwing_shaders::RELU_F32,
            OpKind::AddF32 => dragonwing_shaders::ADD_F32,
            OpKind::GemmF32 => dragonwing_shaders::GEMM_F32_NAIVE,
            OpKind::GemmF32Tiled => dragonwing_shaders::GEMM_F32_TILED,
            OpKind::Conv2dF32Nhwc => dragonwing_shaders::CONV2D_F32_NHWC,
            OpKind::Conv2dFp16Nhwc => dragonwing_shaders::CONV2D_FP16_NHWC,
            OpKind::Maxpool2dF32 => dragonwing_shaders::MAXPOOL2D_F32,
            OpKind::SoftmaxF32 => dragonwing_shaders::SOFTMAX_F32,
            OpKind::FillFp16 => dragonwing_shaders::FILL_FP16,
            OpKind::AxpyFp16 => dragonwing_shaders::AXPY_FP16,
            OpKind::ReluFp16 => dragonwing_shaders::RELU_FP16,
            OpKind::AddFp16 => dragonwing_shaders::ADD_FP16,
            OpKind::GemmFp16 => dragonwing_shaders::GEMM_FP16,
        }
    }

    /// Push-constant size in bytes. Must match the shader's push_constant layout.
    pub const fn push_constant_size(self) -> u32 {
        match self {
            OpKind::FillF32 | OpKind::FillFp16 => 16,
            OpKind::AxpyF32 | OpKind::AxpyFp16 => 16,
            OpKind::ReluF32 | OpKind::ReluFp16 => 16,
            OpKind::AddF32 | OpKind::AddFp16 => 16,
            OpKind::GemmF32 | OpKind::GemmF32Tiled | OpKind::GemmFp16 => 16,
            OpKind::SoftmaxF32 => 16,
            // Conv2d and Maxpool have larger push constants (64 bytes)
            OpKind::Conv2dF32Nhwc | OpKind::Conv2dFp16Nhwc | OpKind::Maxpool2dF32 => 64,
        }
    }
}

// ===========================================================================
// Pipeline cache persistence
// ===========================================================================

/// Compute a hash of all shader contents for cache invalidation.
///
/// When any shader changes, this hash changes, and the old cache file
/// is ignored (new filename).
fn shader_content_hash() -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    
    let mut hasher = DefaultHasher::new();
    for (name, blob) in dragonwing_shaders::ALL {
        name.hash(&mut hasher);
        blob.hash(&mut hasher);
    }
    hasher.finish()
}

/// Get the pipeline cache file path.
///
/// Format: `$XDG_CACHE_HOME/dragonwing/pipeline-<uuid>-<driver_ver>-<shader_hash>.cache`
fn cache_file_path(ctx: &Context) -> Option<PathBuf> {
    let cache_dir = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .map(|p| p.join("dragonwing"))?;
    
    let props = ctx.physical_device_properties();
    
    // Build a stable identifier from pipeline cache UUID
    let uuid = props.pipeline_cache_uuid;
    let uuid_hex: String = uuid.iter().map(|b| format!("{b:02x}")).collect();
    
    let driver_ver = props.driver_version;
    let shader_hash = shader_content_hash();
    
    let filename = format!("pipeline-{uuid_hex}-{driver_ver:08x}-{shader_hash:016x}.cache");
    
    Some(cache_dir.join(filename))
}

/// Load pipeline cache data from disk if available.
fn load_cache_from_disk(path: &std::path::Path) -> Option<Vec<u8>> {
    match std::fs::read(path) {
        Ok(data) => {
            #[cfg(feature = "verbose")]
            eprintln!("[dragonwing-vulkan] Loaded pipeline cache from {}", path.display());
            Some(data)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            #[cfg(feature = "verbose")]
            eprintln!("[dragonwing-vulkan] No pipeline cache found at {}", path.display());
            None
        }
        Err(e) => {
            eprintln!("[dragonwing-vulkan] Warning: Failed to load pipeline cache: {e}");
            None
        }
    }
}

/// Save pipeline cache data to disk atomically.
fn save_cache_to_disk(path: &std::path::Path, data: &[u8]) {
    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("[dragonwing-vulkan] Warning: Failed to create cache directory: {e}");
            return;
        }
    }
    
    // Write to temp file then rename for atomicity
    let temp_path = path.with_extension("cache.tmp");
    if let Err(e) = std::fs::write(&temp_path, data) {
        eprintln!("[dragonwing-vulkan] Warning: Failed to write pipeline cache: {e}");
        return;
    }
    
    if let Err(e) = std::fs::rename(&temp_path, path) {
        eprintln!("[dragonwing-vulkan] Warning: Failed to rename pipeline cache: {e}");
        let _ = std::fs::remove_file(&temp_path);
        return;
    }
    
    #[cfg(feature = "verbose")]
    eprintln!("[dragonwing-vulkan] Saved pipeline cache to {}", path.display());
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
    /// Path to the on-disk cache file (if available).
    cache_path: Option<PathBuf>,
    /// Whether the cache has been modified since load.
    dirty: Mutex<bool>,
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache")
            .field("ctx", &self.ctx)
            .finish_non_exhaustive()
    }
}

impl PipelineCache {
    /// Create a new cache. Loads from disk if available.
    pub fn new(ctx: Arc<Context>) -> Result<Self> {
        // Try to load existing cache from disk
        let cache_path = cache_file_path(&ctx);
        let initial_data = cache_path.as_ref().and_then(|p| load_cache_from_disk(p));
        
        // Create VkPipelineCache with loaded data (or empty if none)
        let cache_info = if let Some(ref data) = initial_data {
            vk::PipelineCacheCreateInfo::default().initial_data(data)
        } else {
            vk::PipelineCacheCreateInfo::default()
        };
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
            cache_path,
            dirty: Mutex::new(false),
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

        // Mark cache as dirty so it gets saved on drop
        *self.dirty.lock().expect("dirty mutex poisoned") = true;

        Ok(cached)
    }

    /// Save the pipeline cache to disk if it has been modified.
    ///
    /// Called automatically on drop, but can be called explicitly for
    /// early persistence.
    pub fn save_if_dirty(&self) {
        let dirty = *self.dirty.lock().expect("dirty mutex poisoned");
        if !dirty {
            return;
        }
        
        let Some(ref path) = self.cache_path else {
            return;
        };
        
        // Get cache data
        // SAFETY: vk_cache is valid.
        let data = match unsafe { self.ctx.device().get_pipeline_cache_data(self.vk_cache) } {
            Ok(data) if !data.is_empty() => data,
            Ok(_) => {
                eprintln!("[dragonwing-vulkan] Warning: Pipeline cache is empty");
                return;
            }
            Err(e) => {
                eprintln!("[dragonwing-vulkan] Warning: Failed to get pipeline cache data: {e:?}");
                return;
            }
        };
        
        save_cache_to_disk(path, &data);
        *self.dirty.lock().expect("dirty mutex poisoned") = false;
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
        // Save cache to disk before destroying
        self.save_if_dirty();
        
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
