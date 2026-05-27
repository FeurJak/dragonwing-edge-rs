# Integration Guide for LLM Agents

This document provides comprehensive guidance for LLM coding agents working on dragonwing-edge-rs. It covers project structure, conventions, and workflows for extending the compute framework.

## Project Overview

dragonwing-edge-rs is a GPU compute framework for the Arduino UNO Q (Qualcomm QRB2210), featuring:

- **Vulkan backend** via Mesa/Turnip for Adreno A702 GPU
- **CPU backend** with NEON SIMD for correctness reference
- **Cross-backend parity testing** to verify numerical equivalence

## Crate Structure

```
dragonwing-edge-rs/
├── Cargo.toml              # Workspace root
├── crates/
│   ├── dragonwing-core/    # Shared traits and types
│   ├── dragonwing-cpu/     # CPU/NEON reference backend
│   ├── dragonwing-vulkan/  # Vulkan compute backend
│   ├── dragonwing-shaders/ # GLSL→SPIR-V shader compilation
│   ├── dragonwing-test/    # Cross-backend parity tests
│   ├── dragonwing-hal/     # Hardware abstraction (future)
│   ├── dragonwing-probe/   # Hardware capability probe
│   └── dragonwing-edge/    # Top-level API (future)
└── docs/                   # Documentation
```

## Adding a New Op

### Step 1: Define the Op Contract

Decide the signature, matching between CPU and GPU:

```rust
// CPU: operates on slices directly
pub fn new_op(output: &mut [f32], input: &[f32], param: f32);

// GPU: operates on VulkanBuffers
pub fn new_op(backend: &VulkanBackend, out: &mut VulkanBuffer, inp: &VulkanBuffer, param: f32) -> Result<()>;
```

### Step 2: Implement CPU Version

In `crates/dragonwing-cpu/src/ops.rs`:

```rust
/// Brief description of what the op does.
///
/// # Arguments
/// * `output` - Output buffer (written)
/// * `input` - Input buffer (read)
/// * `param` - Scalar parameter
///
/// # Panics
/// Panics if `output.len() != input.len()`.
pub fn new_op(output: &mut [f32], input: &[f32], param: f32) {
    assert_eq!(output.len(), input.len(), "new_op: length mismatch");
    
    #[cfg(target_arch = "aarch64")]
    {
        new_op_neon(output, input, param);
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    {
        new_op_scalar(output, input, param);
    }
}

#[cfg(target_arch = "aarch64")]
fn new_op_neon(output: &mut [f32], input: &[f32], param: f32) {
    use std::arch::aarch64::*;
    // NEON implementation processing 4 floats at a time
}

fn new_op_scalar(output: &mut [f32], input: &[f32], param: f32) {
    // Portable scalar fallback
}
```

### Step 3: Write the GLSL Shader

In `crates/dragonwing-shaders/glsl/new_op.comp`:

```glsl
#version 450
//
// new_op — brief description
//
// Bindings:
//   set=0 binding=0  std430 buffer Output[]  (write-only)
//   set=0 binding=1  std430 buffer Input[]   (read-only)
//
// Push constants (16 bytes):
//   uint  n         — element count
//   float param     — scalar parameter
//   float _pad0     — padding for alignment
//   float _pad1     — padding for alignment
//
// Dispatch:
//   gx = ceil(n / 64). gy = gz = 1.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer OutBuf {
    float output[];
};

layout(set = 0, binding = 1, std430) readonly buffer InBuf {
    float input[];
};

layout(push_constant) uniform Pc {
    uint  n;
    float param;
    float _pad0;
    float _pad1;
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.n) {
        return;
    }
    output[gid] = /* compute result */;
}
```

### Step 4: Compile the Shader

```bash
cd crates/dragonwing-shaders
glslangValidator -V -o spv/new_op.spv glsl/new_op.comp
```

### Step 5: Register in dragonwing-shaders

In `crates/dragonwing-shaders/src/lib.rs`:

```rust
/// SPIR-V blob for the new_op shader.
pub const NEW_OP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/new_op.spv"));
```

Add to `build.rs` sources list if using the build script.

### Step 6: Add to Pipeline Cache

In `crates/dragonwing-vulkan/src/pipeline.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    FillF32,
    AxpyF32,
    ReluF32,
    GemmF32,
    NewOp,  // Add new variant
}

impl OpKind {
    pub const fn binding_count(self) -> u32 {
        match self {
            // ... existing ...
            OpKind::NewOp => 2,  // Output + Input
        }
    }

    pub fn spirv(self) -> &'static [u8] {
        match self {
            // ... existing ...
            OpKind::NewOp => dragonwing_shaders::NEW_OP,
        }
    }

    pub const fn push_constant_size(self) -> u32 {
        match self {
            // ... existing ...
            OpKind::NewOp => 16,  // n + param + pad×2
        }
    }
}
```

### Step 7: Implement Vulkan Op

In `crates/dragonwing-vulkan/src/ops.rs`:

```rust
/// New op description.
///
/// # Errors
/// Returns error if buffer sizes don't match or Vulkan calls fail.
pub fn new_op(
    backend: &VulkanBackend,
    output: &mut VulkanBuffer,
    input: &VulkanBuffer,
    param: f32,
) -> Result<()> {
    use dragonwing_core::BackendBuffer;
    
    if output.len_bytes() != input.len_bytes() {
        return Err(Error::Backend("new_op: size mismatch".into()));
    }
    let n = output.len_bytes() / 4;

    let cached = backend.pipelines().get_or_create(OpKind::NewOp)?;
    let desc_set = backend.pipelines().allocate_descriptor_set(cached.descriptor_set_layout)?;

    // Bindings: 0=output, 1=input
    let buf_infos = [
        vk::DescriptorBufferInfo {
            buffer: output.vk_buffer(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        },
        vk::DescriptorBufferInfo {
            buffer: input.vk_buffer(),
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
    struct PushNewOp {
        n: u32,
        param: f32,
        _pad0: f32,
        _pad1: f32,
    }
    let pc = PushNewOp {
        n: n as u32,
        param,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let pc_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            (&raw const pc).cast::<u8>(),
            std::mem::size_of::<PushNewOp>(),
        )
    };

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
```

### Step 8: Add Parity Test

In `crates/dragonwing-test/src/generators.rs`:

```rust
pub fn new_op_test_data(seed: u64) -> (Vec<f32>, f32) {
    let mut rng = Rng::new(seed);
    let n = 1024 + (rng.next_u64() % 1024) as usize;
    let param = rng.next_f32_range(-10.0, 10.0);
    let mut input = vec![0.0f32; n];
    rng.fill_f32(&mut input, -100.0, 100.0);
    (input, param)
}
```

In `crates/dragonwing-test/src/harness.rs`:

```rust
fn test_new_op(
    _cpu: &CpuBackend,
    vulkan: &VulkanBackend,
    config: &TestConfig,
    seed: u64,
    trial: usize,
) -> TestResult {
    let name = format!("new_op[{trial}]");
    let (input, param) = generators::new_op_test_data(seed);
    let n = input.len();

    // CPU
    let mut cpu_result = vec![0.0f32; n];
    dragonwing_cpu::ops::new_op(&mut cpu_result, &input, param);

    // Vulkan
    let mut vk_inp = match vulkan.alloc(n * 4, BufferKind::Storage) {
        Ok(b) => b,
        Err(e) => return TestResult { name, passed: false, error: Some(format!("{e}")), max_diff: f32::NAN },
    };
    let mut vk_out = match vulkan.alloc(n * 4, BufferKind::Storage) {
        Ok(b) => b,
        Err(e) => return TestResult { name, passed: false, error: Some(format!("{e}")), max_diff: f32::NAN },
    };
    
    // Upload, run, sync, download...
    // Compare results...
    
    compare_f32(&name, &cpu_result, vk_floats, config.elementwise_tol)
}
```

## Common Pitfalls

### 1. Push Constant Size Mismatch

**Problem**: Pipeline layout declares different size than shader expects.

**Fix**: Ensure push constant struct in Rust matches shader's `layout(push_constant)`:
- Both must have same size (16 bytes recommended for alignment)
- Add padding fields explicitly

### 2. Buffer Binding Order

**Problem**: Vulkan bindings don't match shader's `binding=N`.

**Fix**: Carefully match descriptor writes to shader bindings:
```glsl
// Shader
layout(set = 0, binding = 0, std430) buffer A { ... };
layout(set = 0, binding = 1, std430) buffer B { ... };

// Rust - must write A to binding 0, B to binding 1
```

### 3. Workgroup Size Mismatch

**Problem**: Dispatch calculation uses wrong local_size.

**Fix**: Match dispatch to shader's `layout(local_size_x = N)`:
```rust
// If shader has local_size_x = 64
let groups = n.div_ceil(64);  // NOT 256!
```

### 4. Forgetting Synchronization

**Problem**: Download reads stale data before GPU finishes.

**Fix**: Always call `backend.synchronize()` before download:
```rust
ops::new_op(backend, &mut out, &inp, param)?;
backend.synchronize()?;  // Wait for GPU
backend.download(&out, &mut result)?;
```

### 5. Timeline Semaphore KHR

**Problem**: Core Vulkan 1.2 functions don't exist on 1.0 devices.

**Fix**: Use KHR extension loader for timeline semaphores:
```rust
// Wrong (fails on Vulkan 1.0)
device.wait_semaphores(&wait_info, timeout)?;

// Correct
timeline_semaphore_khr.wait_semaphores(&wait_info, timeout)?;
```

## Testing Workflow

1. **Local check**: `cargo check -p dragonwing-vulkan`
2. **Push to device**: Create tarball and adb push
3. **Build on device**: `cargo build --release`
4. **Run tests**: `./target/release/vulkan-test && ./target/release/parity-test`

## Hardware Constraints

When designing ops, respect Adreno A702 limits:

| Limit | Value | Implication |
|-------|-------|-------------|
| maxComputeWorkGroupInvocations | 512 | local_size_x × y × z ≤ 512 |
| maxComputeSharedMemorySize | 16 KiB | Tiled algorithms constrained |
| maxStorageBufferRange | 128 MiB | Single buffer limit |
| subgroupSize | 4 | Reduction algorithms need shared memory |

## Code Style

- Use `#![warn(missing_docs)]` on all public items
- Every `unsafe` block needs `// SAFETY:` comment
- Match CPU and GPU op signatures as closely as possible
- Prefer explicit padding in push constants over implicit
- Use `div_ceil()` for dispatch calculations

## Debugging Tips

1. **Shader compilation**: Check `glslangValidator` output for errors
2. **Vulkan validation**: Enable `VulkanConfig::validation = true` on host
3. **Print debugging**: Add eprintln! in Rust or printf in shaders (if supported)
4. **Binary inspection**: Use `spirv-dis` to decompile SPIR-V
5. **GPU capture**: RenderDoc works with Turnip on some setups
