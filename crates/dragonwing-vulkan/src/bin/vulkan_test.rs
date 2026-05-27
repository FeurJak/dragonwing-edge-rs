//! Vulkan backend test binary.
//!
//! Tests all ops (fill, axpy, relu, gemm, conv2d, maxpool2d, softmax) against expected results.
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

    // Test 5: conv2d_f32_nhwc
    print!("Test conv2d_f32_nhwc... ");
    match test_conv2d(&backend) {
        Ok(()) => {
            println!("PASS");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // Test 6: maxpool2d_f32
    print!("Test maxpool2d_f32... ");
    match test_maxpool2d(&backend) {
        Ok(()) => {
            println!("PASS");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // Test 7: softmax_f32
    print!("Test softmax_f32... ");
    match test_softmax(&backend) {
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

fn test_conv2d(backend: &VulkanBackend) -> Result<(), String> {
    // Small conv2d test: 1x4x4x2 input, 3x3 kernel, 2->3 channels, no padding, stride 1.
    // Output: 1x2x2x3.
    let n = 1_usize;
    let h_in = 4_usize;
    let w_in = 4_usize;
    let c_in = 2_usize;
    let c_out = 3_usize;
    let k_h = 3_usize;
    let k_w = 3_usize;
    let stride_h = 1_usize;
    let stride_w = 1_usize;
    let pad_h = 0_usize;
    let pad_w = 0_usize;

    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1; // 2
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1; // 2

    // Input: sequential values, NHWC layout.
    let input: Vec<f32> = (0..(n * h_in * w_in * c_in)).map(|i| i as f32).collect();
    // Kernel: simple values, layout [k_h, k_w, c_in, c_out].
    let kernel: Vec<f32> = (0..(k_h * k_w * c_in * c_out)).map(|i| (i as f32) * 0.01).collect();

    let mut input_buf = backend
        .alloc(input.len() * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc input: {e}"))?;
    let mut kernel_buf = backend
        .alloc(kernel.len() * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc kernel: {e}"))?;
    let mut output_buf = backend
        .alloc(n * h_out * w_out * c_out * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc output: {e}"))?;

    backend
        .upload(&mut input_buf, bytemuck_cast_slice(&input))
        .map_err(|e| format!("upload input: {e}"))?;
    backend
        .upload(&mut kernel_buf, bytemuck_cast_slice(&kernel))
        .map_err(|e| format!("upload kernel: {e}"))?;

    ops::conv2d_f32_nhwc(
        backend,
        &input_buf,
        &kernel_buf,
        &mut output_buf,
        n,
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
    )
    .map_err(|e| format!("conv2d_f32_nhwc: {e}"))?;
    backend.synchronize().map_err(|e| format!("sync: {e}"))?;

    let mut readback = vec![0u8; n * h_out * w_out * c_out * 4];
    backend
        .download(&output_buf, &mut readback)
        .map_err(|e| format!("download: {e}"))?;

    // Compute expected on CPU.
    let expected = cpu_conv2d(&input, &kernel, n, h_in, w_in, c_in, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w);
    let floats: &[f32] = bytemuck_cast(&readback);

    for i in 0..expected.len() {
        if (floats[i] - expected[i]).abs() > 1e-3 {
            return Err(format!(
                "mismatch at {i}: expected {}, got {}",
                expected[i], floats[i]
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cpu_conv2d(
    input: &[f32],
    kernel: &[f32],
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
) -> Vec<f32> {
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;
    let mut output = vec![0.0_f32; n * h_out * w_out * c_out];

    for batch in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                for oc in 0..c_out {
                    let mut acc = 0.0_f32;
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                            let iw = (ow * stride_w + kw) as isize - pad_w as isize;
                            if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                let ih = ih as usize;
                                let iw = iw as usize;
                                for ic in 0..c_in {
                                    let input_idx = ((batch * h_in + ih) * w_in + iw) * c_in + ic;
                                    let kernel_idx = ((kh * k_w + kw) * c_in + ic) * c_out + oc;
                                    acc += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                    let output_idx = ((batch * h_out + oh) * w_out + ow) * c_out + oc;
                    output[output_idx] = acc;
                }
            }
        }
    }
    output
}

fn test_maxpool2d(backend: &VulkanBackend) -> Result<(), String> {
    // Small maxpool test: 1x4x4x2 input, 2x2 pool, stride 2.
    // Output: 1x2x2x2.
    let n = 1_usize;
    let h_in = 4_usize;
    let w_in = 4_usize;
    let c = 2_usize;
    let pool_h = 2_usize;
    let pool_w = 2_usize;
    let stride_h = 2_usize;
    let stride_w = 2_usize;

    let h_out = (h_in - pool_h) / stride_h + 1; // 2
    let w_out = (w_in - pool_w) / stride_w + 1; // 2

    // Input: sequential values, NHWC layout.
    let input: Vec<f32> = (0..(n * h_in * w_in * c)).map(|i| i as f32).collect();

    let mut input_buf = backend
        .alloc(input.len() * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc input: {e}"))?;
    let mut output_buf = backend
        .alloc(n * h_out * w_out * c * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc output: {e}"))?;

    backend
        .upload(&mut input_buf, bytemuck_cast_slice(&input))
        .map_err(|e| format!("upload input: {e}"))?;

    ops::maxpool2d_f32(
        backend,
        &input_buf,
        &mut output_buf,
        n,
        h_in,
        w_in,
        c,
        pool_h,
        pool_w,
        stride_h,
        stride_w,
    )
    .map_err(|e| format!("maxpool2d_f32: {e}"))?;
    backend.synchronize().map_err(|e| format!("sync: {e}"))?;

    let mut readback = vec![0u8; n * h_out * w_out * c * 4];
    backend
        .download(&output_buf, &mut readback)
        .map_err(|e| format!("download: {e}"))?;

    // Compute expected on CPU.
    let expected = cpu_maxpool2d(&input, n, h_in, w_in, c, pool_h, pool_w, stride_h, stride_w);
    let floats: &[f32] = bytemuck_cast(&readback);

    for i in 0..expected.len() {
        if (floats[i] - expected[i]).abs() > 1e-6 {
            return Err(format!(
                "mismatch at {i}: expected {}, got {}",
                expected[i], floats[i]
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cpu_maxpool2d(
    input: &[f32],
    n: usize,
    h_in: usize,
    w_in: usize,
    c: usize,
    pool_h: usize,
    pool_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Vec<f32> {
    let h_out = (h_in - pool_h) / stride_h + 1;
    let w_out = (w_in - pool_w) / stride_w + 1;
    let mut output = vec![f32::NEG_INFINITY; n * h_out * w_out * c];

    for batch in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                for ch in 0..c {
                    let mut max_val = f32::NEG_INFINITY;
                    for ph in 0..pool_h {
                        for pw in 0..pool_w {
                            let ih = oh * stride_h + ph;
                            let iw = ow * stride_w + pw;
                            let idx = ((batch * h_in + ih) * w_in + iw) * c + ch;
                            max_val = max_val.max(input[idx]);
                        }
                    }
                    let output_idx = ((batch * h_out + oh) * w_out + ow) * c + ch;
                    output[output_idx] = max_val;
                }
            }
        }
    }
    output
}

fn test_softmax(backend: &VulkanBackend) -> Result<(), String> {
    // Test softmax: 4 rows of 8 elements each.
    let rows = 4_usize;
    let n = 8_usize;

    // Input: sequential values.
    let input: Vec<f32> = (0..(rows * n)).map(|i| (i as f32) * 0.1 - 1.5).collect();

    let mut input_buf = backend
        .alloc(input.len() * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc input: {e}"))?;
    let mut output_buf = backend
        .alloc(input.len() * 4, BufferKind::Storage)
        .map_err(|e| format!("alloc output: {e}"))?;

    backend
        .upload(&mut input_buf, bytemuck_cast_slice(&input))
        .map_err(|e| format!("upload input: {e}"))?;

    ops::softmax_f32(backend, &input_buf, &mut output_buf, rows, n)
        .map_err(|e| format!("softmax_f32: {e}"))?;
    backend.synchronize().map_err(|e| format!("sync: {e}"))?;

    let mut readback = vec![0u8; rows * n * 4];
    backend
        .download(&output_buf, &mut readback)
        .map_err(|e| format!("download: {e}"))?;

    // Compute expected on CPU.
    let expected = cpu_softmax(&input, rows, n);
    let floats: &[f32] = bytemuck_cast(&readback);

    for i in 0..expected.len() {
        if (floats[i] - expected[i]).abs() > 1e-5 {
            return Err(format!(
                "mismatch at {i}: expected {}, got {}",
                expected[i], floats[i]
            ));
        }
    }

    // Also verify each row sums to 1.
    for row in 0..rows {
        let sum: f32 = (0..n).map(|i| floats[row * n + i]).sum();
        if (sum - 1.0).abs() > 1e-4 {
            return Err(format!("row {row} doesn't sum to 1: {sum}"));
        }
    }
    Ok(())
}

fn cpu_softmax(input: &[f32], rows: usize, n: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; rows * n];
    for row in 0..rows {
        let row_start = row * n;
        // Find max for numerical stability.
        let max_val = (0..n).map(|i| input[row_start + i]).fold(f32::NEG_INFINITY, f32::max);
        // Compute exp(x - max).
        let exp_vals: Vec<f32> = (0..n).map(|i| (input[row_start + i] - max_val).exp()).collect();
        // Sum.
        let sum: f32 = exp_vals.iter().sum();
        // Normalize.
        for i in 0..n {
            output[row_start + i] = exp_vals[i] / sum;
        }
    }
    output
}
