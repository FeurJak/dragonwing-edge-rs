//! Vulkan backend test binary.
//!
//! Tests all four ops (fill, axpy, relu, gemm) against expected results.
//! Run on the device to verify Turnip/Adreno A702 functionality.

use dragonwing_core::{Backend, BufferKind};
use dragonwing_vulkan::{ops, VulkanBackend, VulkanConfig};

fn main() {
    println!("=== dragonwing-vulkan test ===\n");

    // Initialize backend.
    let backend = match VulkanBackend::new(VulkanConfig::default()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERROR: Failed to initialize Vulkan backend: {e}");
            std::process::exit(1);
        }
    };

    println!("Device: {}", backend.device_name());
    println!("Driver: {}", backend.driver_info());
    println!();

    let mut passed = 0;
    let mut failed = 0;

    // Test 1: fill_f32
    print!("Test fill_f32... ");
    match test_fill(&backend) {
        Ok(()) => {
            println!("PASS");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // Test 2: axpy_f32
    print!("Test axpy_f32... ");
    match test_axpy(&backend) {
        Ok(()) => {
            println!("PASS");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // Test 3: relu_f32
    print!("Test relu_f32... ");
    match test_relu(&backend) {
        Ok(()) => {
            println!("PASS");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // Test 4: gemm_f32
    print!("Test gemm_f32... ");
    match test_gemm(&backend) {
        Ok(()) => {
            println!("PASS");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    println!("\n=== Results: {passed} passed, {failed} failed ===");
    if failed > 0 {
        std::process::exit(1);
    }
}

fn test_fill(backend: &VulkanBackend) -> Result<(), String> {
    let n = 1024;
    let value = 3.14159_f32;

    let mut buf = backend
        .alloc(n * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc: {e}"))?;

    ops::fill_f32(backend, &mut buf, value).map_err(|e| format!("fill_f32: {e}"))?;
    backend.synchronize().map_err(|e| format!("sync: {e}"))?;

    let mut readback = vec![0u8; n * 4];
    backend
        .download(&buf, &mut readback)
        .map_err(|e| format!("download: {e}"))?;

    // Verify.
    let floats: &[f32] = bytemuck_cast(&readback);
    for (i, &v) in floats.iter().enumerate() {
        if (v - value).abs() > 1e-6 {
            return Err(format!("mismatch at {i}: expected {value}, got {v}"));
        }
    }
    Ok(())
}

fn test_axpy(backend: &VulkanBackend) -> Result<(), String> {
    let n = 1024;
    let alpha = 2.0_f32;

    // x = [1, 2, 3, ...], y = [10, 20, 30, ...]
    let x_data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let y_data: Vec<f32> = (1..=n).map(|i| (i * 10) as f32).collect();

    let mut x_buf = backend
        .alloc(n * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc x: {e}"))?;
    let mut y_buf = backend
        .alloc(n * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc y: {e}"))?;

    backend
        .upload(&mut x_buf, bytemuck_cast_slice(&x_data))
        .map_err(|e| format!("upload x: {e}"))?;
    backend
        .upload(&mut y_buf, bytemuck_cast_slice(&y_data))
        .map_err(|e| format!("upload y: {e}"))?;

    ops::axpy_f32(backend, &x_buf, &mut y_buf, alpha).map_err(|e| format!("axpy_f32: {e}"))?;
    backend.synchronize().map_err(|e| format!("sync: {e}"))?;

    let mut readback = vec![0u8; n * 4];
    backend
        .download(&y_buf, &mut readback)
        .map_err(|e| format!("download: {e}"))?;

    // Verify: y[i] = alpha * x[i] + y[i] = 2 * i + 10*i = 12*i
    let floats: &[f32] = bytemuck_cast(&readback);
    for i in 0..n {
        let expected = alpha * x_data[i] + y_data[i];
        if (floats[i] - expected).abs() > 1e-4 {
            return Err(format!(
                "mismatch at {i}: expected {expected}, got {}",
                floats[i]
            ));
        }
    }
    Ok(())
}

fn test_relu(backend: &VulkanBackend) -> Result<(), String> {
    let n = 1024;

    // Half negative, half positive.
    let data: Vec<f32> = (0..n).map(|i| (i as f32) - 512.0).collect();

    let mut buf = backend
        .alloc(n * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc: {e}"))?;

    backend
        .upload(&mut buf, bytemuck_cast_slice(&data))
        .map_err(|e| format!("upload: {e}"))?;

    ops::relu_f32(backend, &mut buf).map_err(|e| format!("relu_f32: {e}"))?;
    backend.synchronize().map_err(|e| format!("sync: {e}"))?;

    let mut readback = vec![0u8; n * 4];
    backend
        .download(&buf, &mut readback)
        .map_err(|e| format!("download: {e}"))?;

    // Verify: relu(x) = max(x, 0).
    let floats: &[f32] = bytemuck_cast(&readback);
    for i in 0..n {
        let expected = data[i].max(0.0);
        if (floats[i] - expected).abs() > 1e-6 {
            return Err(format!(
                "mismatch at {i}: expected {expected}, got {}",
                floats[i]
            ));
        }
    }
    Ok(())
}

fn test_gemm(backend: &VulkanBackend) -> Result<(), String> {
    // Small GEMM: 4x3 * 3x5 = 4x5.
    let m = 4_usize;
    let k = 3_usize;
    let n = 5_usize;

    // A: M×K row-major.
    let a: Vec<f32> = (0..(m * k)).map(|i| (i + 1) as f32).collect();
    // B: K×N row-major.
    let b: Vec<f32> = (0..(k * n)).map(|i| ((i + 1) * 2) as f32).collect();

    let mut a_buf = backend
        .alloc(m * k * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc A: {e}"))?;
    let mut b_buf = backend
        .alloc(k * n * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc B: {e}"))?;
    let mut c_buf = backend
        .alloc(m * n * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc C: {e}"))?;

    backend
        .upload(&mut a_buf, bytemuck_cast_slice(&a))
        .map_err(|e| format!("upload A: {e}"))?;
    backend
        .upload(&mut b_buf, bytemuck_cast_slice(&b))
        .map_err(|e| format!("upload B: {e}"))?;

    ops::gemm_f32(backend, &a_buf, &b_buf, &mut c_buf, m, n, k)
        .map_err(|e| format!("gemm_f32: {e}"))?;
    backend.synchronize().map_err(|e| format!("sync: {e}"))?;

    let mut readback = vec![0u8; m * n * 4];
    backend
        .download(&c_buf, &mut readback)
        .map_err(|e| format!("download: {e}"))?;

    // Compute expected C on CPU.
    let expected = cpu_gemm(&a, &b, m, n, k);
    let floats: &[f32] = bytemuck_cast(&readback);

    for i in 0..(m * n) {
        if (floats[i] - expected[i]).abs() > 1e-3 {
            return Err(format!(
                "mismatch at {i}: expected {}, got {}",
                expected[i], floats[i]
            ));
        }
    }
    Ok(())
}

fn cpu_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// Minimal bytemuck-like helpers (no external dep).
fn bytemuck_cast<T>(bytes: &[u8]) -> &[T] {
    let len = bytes.len() / std::mem::size_of::<T>();
    // SAFETY: T is f32 which is valid for any bit pattern, and we trust
    // the buffer was allocated with correct alignment (4-byte aligned on
    // VkBuffer).
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len) }
}

fn bytemuck_cast_slice<T>(data: &[T]) -> &[u8] {
    let len = data.len() * std::mem::size_of::<T>();
    // SAFETY: reinterpreting T as bytes is always safe.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), len) }
}
