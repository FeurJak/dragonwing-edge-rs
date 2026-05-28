//! Compute ops implemented on the CPU.
//!
//! Each op exists in two flavours:
//!
//! * **`*_neon`** — AArch64 NEON-accelerated path. Compiled on AArch64 only.
//! * **`*_scalar`** — Portable scalar path. Always compiled; used as a
//!   fallback on non-AArch64 hosts and as the reference for the NEON
//!   implementation's own unit tests.
//!
//! The public-facing function (e.g. [`axpy_f32`]) picks the right path at
//! compile time via `#[cfg(target_arch = ...)]`. We do **not** do runtime
//! feature detection because the target hardware (QRB2210, all Cortex-A53
//! derivatives) is known to support ASIMD universally and host targets
//! are checked at compile time.
//!
//! # FP16 ops
//!
//! FP16 ops (e.g. [`fill_fp16`], [`axpy_fp16`], [`relu_fp16`]) convert to F32
//! in registers, compute in F32, and convert back on store. This is the
//! standard pattern for Cortex-A53 which has FP16 storage but **not** FP16
//! arithmetic (no `armv8.2-a+fp16`). The FP16 path saves memory bandwidth
//! (2 bytes per element vs 4) at the cost of conversion overhead.
//!
//! # Tail handling
//!
//! NEON paths process 4 floats per instruction (`float32x4_t`) and an
//! unrolled inner loop processes 16 floats per iteration (four NEON
//! registers). Any leftover elements at the tail of the slice fall
//! through to the scalar path. This means the NEON and scalar paths must
//! produce *bitwise-identical* results, which is true for `fill`, `axpy`
//! (assuming the same FMA contraction), and `relu`.
//!
//! # Why `fma` and not `mul + add`
//!
//! `vfmaq_f32` is a fused multiply-add: `y = a * x + y` with a **single**
//! rounding. The scalar implementation uses `f32::mul_add` to match,
//! producing bitwise-identical output as long as the host CPU's `mul_add`
//! is also single-rounded (true on every supported target).

use core::cmp::Ordering;

use dragonwing_core::F16;

// ---------------------------------------------------------------------------
// fill
// ---------------------------------------------------------------------------

/// Write `v` to every element of `y`.
///
/// `y` may have any length; an empty slice is a no-op.
pub fn fill_f32(y: &mut [f32], v: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: ASIMD is mandatory on every AArch64 target Rust supports.
        unsafe { fill_f32_neon(y, v) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fill_f32_scalar(y, v);
    }
}

/// Portable scalar implementation of [`fill_f32`]. Exposed for tests and
/// for ports to non-AArch64 hosts.
pub fn fill_f32_scalar(y: &mut [f32], v: f32) {
    for slot in y {
        *slot = v;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fill_f32_neon(y: &mut [f32], v: f32) {
    use core::arch::aarch64::{vdupq_n_f32, vst1q_f32};
    let len = y.len();
    // SAFETY: NEON is enabled at this function via `#[target_feature]`;
    // the indexing arithmetic below maintains `i + 16 <= len` as a loop
    // invariant so all `add`/`vst1q_f32` calls are in bounds. `vst1q_f32`
    // needs only 4-byte alignment, which is the natural alignment of an
    // `&mut [f32]` slice. `vdupq_n_f32` is a pure value intrinsic.
    unsafe {
        let v4 = vdupq_n_f32(v);
        let mut i = 0;
        while i + 16 <= len {
            let p = y.as_mut_ptr().add(i);
            vst1q_f32(p, v4);
            vst1q_f32(p.add(4), v4);
            vst1q_f32(p.add(8), v4);
            vst1q_f32(p.add(12), v4);
            i += 16;
        }
        // Tail (0..15 elements) goes through the scalar path so the
        // result is bitwise-identical to `fill_f32_scalar`.
        while i < len {
            *y.get_unchecked_mut(i) = v;
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// axpy: y = a * x + y
// ---------------------------------------------------------------------------

/// `y[i] = a * x[i] + y[i]` for every `i`.
///
/// # Panics
///
/// Panics if `y.len() != x.len()`. Callers are expected to size their
/// tensors consistently; a length mismatch indicates a bug upstream.
pub fn axpy_f32(y: &mut [f32], a: f32, x: &[f32]) {
    assert_eq!(y.len(), x.len(), "axpy_f32: length mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: ASIMD is mandatory on AArch64.
        unsafe { axpy_f32_neon(y, a, x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        axpy_f32_scalar(y, a, x);
    }
}

/// Portable scalar implementation of [`axpy_f32`], using
/// [`f32::mul_add`] so it matches the NEON `vfmaq_f32` rounding exactly.
pub fn axpy_f32_scalar(y: &mut [f32], a: f32, x: &[f32]) {
    assert_eq!(y.len(), x.len(), "axpy_f32_scalar: length mismatch");
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = a.mul_add(*xi, *yi);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_f32_neon(y: &mut [f32], a: f32, x: &[f32]) {
    use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
    debug_assert_eq!(y.len(), x.len());
    let len = y.len();
    // SAFETY: NEON enabled by `#[target_feature]`; `i + 16 <= len`
    // invariant keeps all `add`/`vld1q_f32`/`vst1q_f32` in bounds.
    // `vfmaq_f32` is `acc + a*b` single-rounded — matches scalar
    // `f32::mul_add`, which is what the tail uses, so the NEON path is
    // bitwise-equivalent to the scalar path.
    unsafe {
        let av = vdupq_n_f32(a);
        let mut i = 0;
        while i + 16 <= len {
            let py = y.as_mut_ptr().add(i);
            let px = x.as_ptr().add(i);

            let x0 = vld1q_f32(px);
            let x1 = vld1q_f32(px.add(4));
            let x2 = vld1q_f32(px.add(8));
            let x3 = vld1q_f32(px.add(12));

            let y0 = vld1q_f32(py);
            let y1 = vld1q_f32(py.add(4));
            let y2 = vld1q_f32(py.add(8));
            let y3 = vld1q_f32(py.add(12));

            let r0 = vfmaq_f32(y0, av, x0);
            let r1 = vfmaq_f32(y1, av, x1);
            let r2 = vfmaq_f32(y2, av, x2);
            let r3 = vfmaq_f32(y3, av, x3);

            vst1q_f32(py, r0);
            vst1q_f32(py.add(4), r1);
            vst1q_f32(py.add(8), r2);
            vst1q_f32(py.add(12), r3);
            i += 16;
        }
        while i < len {
            let yi = y.get_unchecked_mut(i);
            *yi = a.mul_add(*x.get_unchecked(i), *yi);
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// relu: y[i] = max(0, x[i])
// ---------------------------------------------------------------------------

/// `y[i] = max(0, x[i])` for every `i`.
///
/// # Panics
///
/// Panics if `y.len() != x.len()`.
pub fn relu_f32(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len(), "relu_f32: length mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: ASIMD is mandatory on AArch64.
        unsafe { relu_f32_neon(y, x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        relu_f32_scalar(y, x);
    }
}

/// Portable scalar [`relu_f32`].
///
/// Uses `partial_cmp` on `f32` to preserve the IEEE-754 NaN handling that
/// `vmaxq_f32` follows: NaN inputs propagate to NaN outputs. This matters
/// for parity testing — see the tolerance policy in
/// `docs/parity-testing.md`.
pub fn relu_f32_scalar(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len(), "relu_f32_scalar: length mismatch");
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = match xi.partial_cmp(&0.0) {
            Some(Ordering::Greater) => *xi,
            // Negative *and* zero map to 0.0 (positive zero), matching
            // vmaxq_f32 behaviour for finite inputs.
            Some(_) => 0.0,
            // NaN propagates.
            None => f32::NAN,
        };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn relu_f32_neon(y: &mut [f32], x: &[f32]) {
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmaxq_f32, vst1q_f32};
    debug_assert_eq!(y.len(), x.len());
    let len = y.len();
    // SAFETY: as in `axpy_f32_neon`.
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= len {
            let py = y.as_mut_ptr().add(i);
            let px = x.as_ptr().add(i);

            let x0 = vld1q_f32(px);
            let x1 = vld1q_f32(px.add(4));
            let x2 = vld1q_f32(px.add(8));
            let x3 = vld1q_f32(px.add(12));

            vst1q_f32(py, vmaxq_f32(zero, x0));
            vst1q_f32(py.add(4), vmaxq_f32(zero, x1));
            vst1q_f32(py.add(8), vmaxq_f32(zero, x2));
            vst1q_f32(py.add(12), vmaxq_f32(zero, x3));
            i += 16;
        }
        while i < len {
            let xi = *x.get_unchecked(i);
            *y.get_unchecked_mut(i) = if xi > 0.0 { xi } else { 0.0 };
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// gemm_f32_naive: C[m,n] = A[m,k] * B[k,n]
// ---------------------------------------------------------------------------

/// Naive single-precision matrix multiplication: `C = A * B`.
///
/// * `a` is laid out as `[m, k]` row-major, length `m * k`.
/// * `b` is laid out as `[k, n]` row-major, length `k * n`.
/// * `c` is laid out as `[m, n]` row-major, length `m * n`. Overwritten
///   (not accumulated).
///
/// Triple-loop with the K-loop innermost. **Not tiled, not multi-threaded,
/// no NEON.** This is the reference implementation; performance work is
/// task 003.
///
/// # Panics
///
/// Panics if `a.len() != m * k`, `b.len() != k * n`, or `c.len() != m * n`.
pub fn gemm_f32_naive(c: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "gemm_f32_naive: A size mismatch");
    assert_eq!(b.len(), k * n, "gemm_f32_naive: B size mismatch");
    assert_eq!(c.len(), m * n, "gemm_f32_naive: C size mismatch");
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for p in 0..k {
                // Using mul_add here matches what an FMA-capable backend
                // (Vulkan via Turnip) is allowed to do, keeping parity
                // tight.
                acc = a[i * k + p].mul_add(b[p * n + j], acc);
            }
            c[i * n + j] = acc;
        }
    }
}

// ===========================================================================
// FP16 ops
// ===========================================================================
//
// FP16 ops convert to F32 in registers, compute in F32, and convert back on
// store. This is the standard pattern for Cortex-A53 which has FP16 storage
// but not FP16 arithmetic.

// ---------------------------------------------------------------------------
// fill_fp16
// ---------------------------------------------------------------------------

/// Write `v` (converted from f32 to F16) to every element of `y`.
///
/// `y` may have any length; an empty slice is a no-op.
pub fn fill_fp16(y: &mut [F16], v: f32) {
    let v16 = F16::from_f32(v);
    for slot in y {
        *slot = v16;
    }
}

// ---------------------------------------------------------------------------
// axpy_fp16: y = a * x + y (in FP16)
// ---------------------------------------------------------------------------

/// `y[i] = a * x[i] + y[i]` for every `i`, where x and y are F16.
///
/// Computes in F32 internally and converts back to F16.
///
/// # Panics
///
/// Panics if `y.len() != x.len()`.
pub fn axpy_fp16(y: &mut [F16], a: f32, x: &[F16]) {
    assert_eq!(y.len(), x.len(), "axpy_fp16: length mismatch");
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        let x_f32 = xi.to_f32();
        let y_f32 = yi.to_f32();
        let result = a.mul_add(x_f32, y_f32);
        *yi = F16::from_f32(result);
    }
}

// ---------------------------------------------------------------------------
// relu_fp16: y[i] = max(0, x[i])
// ---------------------------------------------------------------------------

/// `y[i] = max(0, x[i])` for every `i`, where x and y are F16.
///
/// Computes in F32 internally and converts back to F16.
///
/// # Panics
///
/// Panics if `y.len() != x.len()`.
pub fn relu_fp16(y: &mut [F16], x: &[F16]) {
    assert_eq!(y.len(), x.len(), "relu_fp16: length mismatch");
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        let x_f32 = xi.to_f32();
        let result = if x_f32 > 0.0 { x_f32 } else { 0.0 };
        *yi = F16::from_f32(result);
    }
}

// ---------------------------------------------------------------------------
// add_f32: y[i] = a[i] + b[i]
// ---------------------------------------------------------------------------

/// Element-wise addition: `y[i] = a[i] + b[i]`.
///
/// # Panics
///
/// Panics if lengths don't match.
pub fn add_f32(y: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(y.len(), a.len(), "add_f32: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "add_f32: y/b length mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: ASIMD is mandatory on AArch64.
        unsafe { add_f32_neon(y, a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        add_f32_scalar(y, a, b);
    }
}

/// Portable scalar implementation of [`add_f32`].
pub fn add_f32_scalar(y: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(y.len(), a.len(), "add_f32_scalar: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "add_f32_scalar: y/b length mismatch");
    for i in 0..y.len() {
        y[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_f32_neon(y: &mut [f32], a: &[f32], b: &[f32]) {
    use core::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32};
    debug_assert_eq!(y.len(), a.len());
    debug_assert_eq!(y.len(), b.len());
    let len = y.len();
    unsafe {
        let mut i = 0;
        while i + 16 <= len {
            let py = y.as_mut_ptr().add(i);
            let pa = a.as_ptr().add(i);
            let pb = b.as_ptr().add(i);

            let a0 = vld1q_f32(pa);
            let a1 = vld1q_f32(pa.add(4));
            let a2 = vld1q_f32(pa.add(8));
            let a3 = vld1q_f32(pa.add(12));

            let b0 = vld1q_f32(pb);
            let b1 = vld1q_f32(pb.add(4));
            let b2 = vld1q_f32(pb.add(8));
            let b3 = vld1q_f32(pb.add(12));

            vst1q_f32(py, vaddq_f32(a0, b0));
            vst1q_f32(py.add(4), vaddq_f32(a1, b1));
            vst1q_f32(py.add(8), vaddq_f32(a2, b2));
            vst1q_f32(py.add(12), vaddq_f32(a3, b3));
            i += 16;
        }
        while i < len {
            *y.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// add_fp16: y[i] = a[i] + b[i] (in FP16)
// ---------------------------------------------------------------------------

/// Element-wise addition in F16: `y[i] = a[i] + b[i]`.
///
/// Computes in F32 internally and converts back to F16.
///
/// # Panics
///
/// Panics if lengths don't match.
pub fn add_fp16(y: &mut [F16], a: &[F16], b: &[F16]) {
    assert_eq!(y.len(), a.len(), "add_fp16: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "add_fp16: y/b length mismatch");
    for i in 0..y.len() {
        let a_f32 = a[i].to_f32();
        let b_f32 = b[i].to_f32();
        y[i] = F16::from_f32(a_f32 + b_f32);
    }
}

// ---------------------------------------------------------------------------
// gemm_fp16: C[m,n] = A[m,k] * B[k,n] with F16 data and F32 accumulator
// ---------------------------------------------------------------------------

/// Half-precision matrix multiplication with F32 accumulator: `C = A * B`.
///
/// * `a` is laid out as `[m, k]` row-major, length `m * k`, F16 elements.
/// * `b` is laid out as `[k, n]` row-major, length `k * n`, F16 elements.
/// * `c` is laid out as `[m, n]` row-major, length `m * n`, F16 elements.
///
/// The accumulation is done in F32 for numerical stability (FP16 accumulator
/// loses precision significantly at K >= 256). The result is converted back
/// to F16 on store.
///
/// # Panics
///
/// Panics if `a.len() != m * k`, `b.len() != k * n`, or `c.len() != m * n`.
pub fn gemm_fp16(c: &mut [F16], a: &[F16], b: &[F16], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "gemm_fp16: A size mismatch");
    assert_eq!(b.len(), k * n, "gemm_fp16: B size mismatch");
    assert_eq!(c.len(), m * n, "gemm_fp16: C size mismatch");
    for i in 0..m {
        for j in 0..n {
            // F32 accumulator for precision
            let mut acc: f32 = 0.0;
            for p in 0..k {
                let a_f32 = a[i * k + p].to_f32();
                let b_f32 = b[p * n + j].to_f32();
                acc = a_f32.mul_add(b_f32, acc);
            }
            c[i * n + j] = F16::from_f32(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — verify NEON path matches scalar path on aarch64 hosts (e.g. M-series Mac).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() <= f32::EPSILON * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn fill_matches_scalar() {
        let mut a = vec![0.0f32; 67];
        let mut b = vec![0.0f32; 67];
        fill_f32(&mut a, 1.25);
        fill_f32_scalar(&mut b, 1.25);
        assert_eq!(a, b);
    }

    #[test]
    fn axpy_matches_scalar() {
        let x: Vec<f32> = (0..101).map(|i| i as f32 * 0.5).collect();
        let mut y_neon: Vec<f32> = (0..101).map(|i| -(i as f32)).collect();
        let mut y_ref = y_neon.clone();
        axpy_f32(&mut y_neon, 2.0, &x);
        axpy_f32_scalar(&mut y_ref, 2.0, &x);
        for (a, b) in y_neon.iter().zip(y_ref.iter()) {
            assert!(approx_eq(*a, *b), "{a} vs {b}");
        }
    }

    #[test]
    fn relu_matches_scalar() {
        let x: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let mut y_neon = vec![0.0f32; x.len()];
        let mut y_ref = vec![0.0f32; x.len()];
        relu_f32(&mut y_neon, &x);
        relu_f32_scalar(&mut y_ref, &x);
        assert_eq!(y_neon, y_ref);
    }

    #[test]
    fn gemm_small_identity() {
        // 3x3 * 3x3 identity check: A * I = A.
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let id = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0_f32; 9];
        gemm_f32_naive(&mut c, &a, &id, 3, 3, 3);
        assert_eq!(c, a);
    }

    // -----------------------------------------------------------------------
    // FP16 tests
    // -----------------------------------------------------------------------

    fn approx_eq_f16(a: F16, b: F16, tol: f32) -> bool {
        let a_f = a.to_f32();
        let b_f = b.to_f32();
        if a_f.is_nan() && b_f.is_nan() {
            return true;
        }
        (a_f - b_f).abs() <= tol
    }

    #[test]
    fn fill_fp16_basic() {
        let mut y = vec![F16::ZERO; 100];
        fill_fp16(&mut y, 1.5);
        for v in &y {
            assert!(
                approx_eq_f16(*v, F16::from_f32(1.5), 1e-3),
                "expected ~1.5, got {}",
                v.to_f32()
            );
        }
    }

    #[test]
    fn axpy_fp16_basic() {
        let n = 64;
        let x: Vec<F16> = (0..n).map(|i| F16::from_f32(i as f32 * 0.1)).collect();
        let mut y: Vec<F16> = (0..n).map(|i| F16::from_f32(i as f32 * 0.05)).collect();

        // Save original y for reference
        let y_orig: Vec<f32> = y.iter().map(|v| v.to_f32()).collect();
        let x_f32: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();

        let alpha = 2.0f32;
        axpy_fp16(&mut y, alpha, &x);

        // Verify: y_new = alpha * x + y_old
        for i in 0..n {
            let expected = alpha * x_f32[i] + y_orig[i];
            let actual = y[i].to_f32();
            let diff = (expected - actual).abs();
            // F16 tolerance is about 0.1% relative error
            assert!(
                diff < 0.01 * expected.abs().max(1.0),
                "axpy_fp16 mismatch at {i}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn relu_fp16_basic() {
        let x: Vec<F16> = (-50..50)
            .map(|i| F16::from_f32(i as f32 * 0.1))
            .collect();
        let mut y = vec![F16::ZERO; x.len()];

        relu_fp16(&mut y, &x);

        for (i, (yi, xi)) in y.iter().zip(x.iter()).enumerate() {
            let x_f = xi.to_f32();
            let y_f = yi.to_f32();
            let expected = if x_f > 0.0 { x_f } else { 0.0 };
            assert!(
                approx_eq_f16(*yi, F16::from_f32(expected), 1e-3),
                "relu_fp16 mismatch at {i}: x={x_f}, expected {expected}, got {y_f}"
            );
        }
    }

    #[test]
    fn add_f32_basic() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 * 2.0).collect();
        let mut y = vec![0.0f32; 100];

        add_f32(&mut y, &a, &b);

        for i in 0..100 {
            assert_eq!(y[i], a[i] + b[i], "add_f32 mismatch at {i}");
        }
    }

    #[test]
    fn add_f32_matches_scalar() {
        let a: Vec<f32> = (0..67).map(|i| i as f32 * 1.5).collect();
        let b: Vec<f32> = (0..67).map(|i| -(i as f32) * 0.5).collect();
        let mut y_neon = vec![0.0f32; 67];
        let mut y_scalar = vec![0.0f32; 67];

        add_f32(&mut y_neon, &a, &b);
        add_f32_scalar(&mut y_scalar, &a, &b);

        assert_eq!(y_neon, y_scalar);
    }

    #[test]
    fn add_fp16_basic() {
        let a: Vec<F16> = (0..64).map(|i| F16::from_f32(i as f32)).collect();
        let b: Vec<F16> = (0..64).map(|i| F16::from_f32(i as f32 * 0.5)).collect();
        let mut y = vec![F16::ZERO; 64];

        add_fp16(&mut y, &a, &b);

        for i in 0..64 {
            let expected = a[i].to_f32() + b[i].to_f32();
            let actual = y[i].to_f32();
            let diff = (expected - actual).abs();
            assert!(
                diff < 0.01 * expected.abs().max(1.0),
                "add_fp16 mismatch at {i}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn conv2d_f32_1x1_identity() {
        // 1x1 conv with identity kernel (c_in == c_out, each channel passes through)
        let n = 1;
        let h = 4;
        let w = 4;
        let c_in = 2;
        let c_out = 2;

        // Input: simple values
        let input: Vec<f32> = (0..(n * h * w * c_in))
            .map(|i| i as f32 * 0.1)
            .collect();

        // Identity 1x1 kernel: [1,1,c_in,c_out] where kernel[0,0,i,i]=1, else 0
        let mut kernel = vec![0.0f32; 1 * 1 * c_in * c_out];
        for i in 0..c_in.min(c_out) {
            kernel[i * c_out + i] = 1.0;
        }

        let mut output = vec![0.0f32; n * h * w * c_out];
        super::conv2d_f32_nhwc(&mut output, &input, &kernel, n, h, w, c_in, c_out, 1, 1, 1, 1, 0, 0);

        // With identity kernel, output should equal input
        for (i, (&out, &inp)) in output.iter().zip(input.iter()).enumerate() {
            let diff = (out - inp).abs();
            assert!(diff < 1e-5, "conv2d 1x1 identity mismatch at {i}: {out} vs {inp}");
        }
    }

    #[test]
    fn conv2d_f32_3x3_simple() {
        // 3x3 conv on a small input
        let n = 1;
        let h_in = 5;
        let w_in = 5;
        let c_in = 1;
        let c_out = 1;
        let k = 3;
        let stride = 1;
        let pad = 0;

        // All ones input
        let input = vec![1.0f32; n * h_in * w_in * c_in];
        // All ones kernel
        let kernel = vec![1.0f32; k * k * c_in * c_out];

        let h_out = (h_in + 2 * pad - k) / stride + 1; // = 3
        let w_out = (w_in + 2 * pad - k) / stride + 1; // = 3
        let mut output = vec![0.0f32; n * h_out * w_out * c_out];

        super::conv2d_f32_nhwc(&mut output, &input, &kernel, n, h_in, w_in, c_in, c_out, k, k, stride, stride, pad, pad);

        // Each output element should be 3*3 = 9 (sum of all ones in 3x3 window)
        for (i, &out) in output.iter().enumerate() {
            assert!(
                (out - 9.0).abs() < 1e-5,
                "conv2d 3x3 simple mismatch at {i}: expected 9.0, got {out}"
            );
        }
    }

    #[test]
    fn maxpool2d_basic() {
        // 2x2 max pooling on a 4x4 input
        let n = 1;
        let h_in = 4;
        let w_in = 4;
        let c = 1;
        let pool = 2;
        let stride = 2;

        // Input with increasing values
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let h_out = (h_in - pool) / stride + 1;
        let w_out = (w_in - pool) / stride + 1;
        let mut output = vec![0.0f32; n * h_out * w_out * c];

        super::maxpool2d_f32_nhwc(&mut output, &input, n, h_in, w_in, c, pool, pool, stride, stride);

        // Expected: max of each 2x2 block
        // [0,1,2,3; 4,5,6,7; 8,9,10,11; 12,13,14,15] -> [5, 7, 13, 15]
        let expected = [5.0, 7.0, 13.0, 15.0];
        for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!((out - exp).abs() < 1e-5, "maxpool2d mismatch at {i}: {out} vs {exp}");
        }
    }

    #[test]
    fn avgpool2d_basic() {
        // 2x2 avg pooling on a 4x4 input
        let n = 1;
        let h_in = 4;
        let w_in = 4;
        let c = 1;
        let pool = 2;
        let stride = 2;

        // All ones input
        let input = vec![1.0f32; n * h_in * w_in * c];
        let h_out = (h_in - pool) / stride + 1;
        let w_out = (w_in - pool) / stride + 1;
        let mut output = vec![0.0f32; n * h_out * w_out * c];

        super::avgpool2d_f32_nhwc(&mut output, &input, n, h_in, w_in, c, pool, pool, stride, stride);

        // Average of ones should be 1.0
        for (i, &out) in output.iter().enumerate() {
            assert!((out - 1.0).abs() < 1e-5, "avgpool2d mismatch at {i}: {out}");
        }
    }

    #[test]
    fn softmax_basic() {
        // Simple softmax test
        let input = [1.0f32, 2.0, 3.0];
        let mut output = [0.0f32; 3];

        super::softmax_f32(&mut output, &input, 3);

        // Verify: sum to 1, all positive, order preserved
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum should be 1.0, got {sum}");
        assert!(output[0] < output[1] && output[1] < output[2], "softmax should preserve order");
        assert!(output.iter().all(|&x| x > 0.0), "softmax outputs should be positive");
    }

    // -----------------------------------------------------------------------
    // Multi-threaded tests
    // -----------------------------------------------------------------------

    #[test]
    fn gemm_f32_mt_matches_naive() {
        let m = 64;
        let n = 64;
        let k = 64;

        // Random-ish input
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02).cos()).collect();

        let mut c_naive = vec![0.0f32; m * n];
        let mut c_mt = vec![0.0f32; m * n];

        super::gemm_f32_naive(&mut c_naive, &a, &b, m, n, k);
        super::gemm_f32_mt(&mut c_mt, &a, &b, m, n, k, 4);

        for i in 0..m * n {
            let diff = (c_naive[i] - c_mt[i]).abs();
            let tol = 1e-4 * c_naive[i].abs().max(1.0);
            assert!(
                diff < tol,
                "gemm_f32_mt mismatch at {i}: naive={} mt={}",
                c_naive[i],
                c_mt[i]
            );
        }
    }

    #[test]
    fn conv2d_f32_mt_matches_st() {
        let n = 1;
        let h_in = 14;
        let w_in = 14;
        let c_in = 16;
        let c_out = 32;
        let k = 3;
        let stride = 1;
        let pad = 1;

        let input: Vec<f32> = (0..n * h_in * w_in * c_in)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let kernel: Vec<f32> = (0..k * k * c_in * c_out)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();

        let h_out = (h_in + 2 * pad - k) / stride + 1;
        let w_out = (w_in + 2 * pad - k) / stride + 1;

        let mut output_st = vec![0.0f32; n * h_out * w_out * c_out];
        let mut output_mt = vec![0.0f32; n * h_out * w_out * c_out];

        super::conv2d_f32_nhwc(
            &mut output_st, &input, &kernel,
            n, h_in, w_in, c_in, c_out, k, k, stride, stride, pad, pad
        );
        super::conv2d_f32_nhwc_mt(
            &mut output_mt, &input, &kernel,
            n, h_in, w_in, c_in, c_out, k, k, stride, stride, pad, pad, 4
        );

        for i in 0..output_st.len() {
            let diff = (output_st[i] - output_mt[i]).abs();
            let tol = 1e-4 * output_st[i].abs().max(1.0);
            assert!(
                diff < tol,
                "conv2d_f32_mt mismatch at {i}: st={} mt={}",
                output_st[i],
                output_mt[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // YOLO ops tests (Task 005)
    // -----------------------------------------------------------------------

    #[test]
    fn sigmoid_f32_basic() {
        let x = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let mut y = vec![0.0f32; 5];
        
        super::sigmoid_f32(&mut y, &x);
        
        // sigmoid(-2) ≈ 0.119, sigmoid(-1) ≈ 0.269, sigmoid(0) = 0.5
        // sigmoid(1) ≈ 0.731, sigmoid(2) ≈ 0.881
        let expected = [0.119, 0.269, 0.5, 0.731, 0.881];
        for (i, (&out, &exp)) in y.iter().zip(expected.iter()).enumerate() {
            let diff = (out - exp).abs();
            assert!(diff < 0.01, "sigmoid mismatch at {i}: {out} vs {exp}");
        }
    }

    #[test]
    fn sigmoid_f32_matches_scalar() {
        let x: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let mut y_main = vec![0.0f32; x.len()];
        let mut y_scalar = vec![0.0f32; x.len()];
        
        super::sigmoid_f32(&mut y_main, &x);
        super::sigmoid_f32_scalar(&mut y_scalar, &x);
        
        for (i, (&a, &b)) in y_main.iter().zip(y_scalar.iter()).enumerate() {
            assert!(approx_eq(a, b), "sigmoid mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn mul_f32_basic() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
        let mut y = vec![0.0f32; 100];
        
        super::mul_f32(&mut y, &a, &b);
        
        for i in 0..100 {
            let expected = a[i] * b[i];
            assert_eq!(y[i], expected, "mul_f32 mismatch at {i}");
        }
    }

    #[test]
    fn mul_f32_matches_scalar() {
        let a: Vec<f32> = (0..67).map(|i| i as f32 * 1.5).collect();
        let b: Vec<f32> = (0..67).map(|i| (i as f32).sin()).collect();
        let mut y_main = vec![0.0f32; 67];
        let mut y_scalar = vec![0.0f32; 67];
        
        super::mul_f32(&mut y_main, &a, &b);
        super::mul_f32_scalar(&mut y_scalar, &a, &b);
        
        for (i, (&a, &b)) in y_main.iter().zip(y_scalar.iter()).enumerate() {
            assert!(approx_eq(a, b), "mul_f32 mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn concat_f32_channels() {
        // Concat two [1,2,2,2] tensors along channel axis -> [1,2,2,4]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [1,2,2,2]
        let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]; // [1,2,2,2]
        let mut output = vec![0.0f32; 16]; // [1,2,2,4]
        
        super::concat_f32(
            &mut output,
            &[&a[..], &b[..]],
            &[[1, 2, 2, 2], [1, 2, 2, 2]],
            3
        );
        
        // At each spatial position, channels from a then b
        // (0,0): [1,2] + [10,20] = [1,2,10,20]
        // (0,1): [3,4] + [30,40] = [3,4,30,40]
        // etc.
        let expected = [
            1.0, 2.0, 10.0, 20.0,  // (0,0,0)
            3.0, 4.0, 30.0, 40.0,  // (0,0,1)
            5.0, 6.0, 50.0, 60.0,  // (0,1,0)
            7.0, 8.0, 70.0, 80.0,  // (0,1,1)
        ];
        assert_eq!(output, expected, "concat along channels failed");
    }

    #[test]
    fn concat_f32_batch() {
        // Concat two [1,2,2,1] tensors along batch axis -> [2,2,2,1]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 8];
        
        super::concat_f32(
            &mut output,
            &[&a[..], &b[..]],
            &[[1, 2, 2, 1], [1, 2, 2, 1]],
            0
        );
        
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(output, expected, "concat along batch failed");
    }

    #[test]
    fn resize_f32_nearest_2x() {
        // Upsample [1,2,2,1] to [1,4,4,1] with nearest neighbor
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 16];
        
        super::resize_f32(&mut output, &input, 1, 2, 2, 4, 4, 1, super::ResizeMode::Nearest);
        
        // Each input pixel becomes a 2x2 block
        // Input: [1,2; 3,4] -> Output: [1,1,2,2; 1,1,2,2; 3,3,4,4; 3,3,4,4]
        let expected = [
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ];
        for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (out - exp).abs();
            assert!(diff < 0.1, "resize nearest mismatch at {i}: {out} vs {exp}");
        }
    }

    #[test]
    fn resize_f32_bilinear_2x() {
        // Upsample [1,2,2,1] to [1,4,4,1] with bilinear
        let input = vec![0.0f32, 1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 16];
        
        super::resize_f32(&mut output, &input, 1, 2, 2, 4, 4, 1, super::ResizeMode::Bilinear);
        
        // Bilinear interpolation should produce smooth gradients
        // The exact values depend on the coordinate mapping convention
        // Just verify the output has reasonable interpolated values
        let min_val: f32 = *output.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_val: f32 = *output.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        // Values should be within input range (0-3) with some margin for edge handling
        assert!(min_val >= -0.5, "min should be near 0, got {min_val}");
        assert!(max_val <= 3.5, "max should be near 3, got {max_val}");
        
        // Output should vary smoothly - check that there's a gradient
        assert!(output[15] > output[0], "should have gradient from top-left to bottom-right");
    }

    #[test]
    fn split_f32_channels() {
        // Split [1,2,2,4] into two [1,2,2,2] along channels
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out1 = vec![0.0f32; 8];
        let mut out2 = vec![0.0f32; 8];
        
        super::split_f32(
            &mut [&mut out1[..], &mut out2[..]],
            &input,
            [1, 2, 2, 4],
            3,
            &[2, 2]
        );
        
        // First output gets channels 0-1, second gets channels 2-3
        // Input at (0,0): [0,1,2,3] -> out1[0,0]=[0,1], out2[0,0]=[2,3]
        assert_eq!(out1[0], 0.0);
        assert_eq!(out1[1], 1.0);
        assert_eq!(out2[0], 2.0);
        assert_eq!(out2[1], 3.0);
    }

    #[test]
    fn transpose_f32_nchw_to_nhwc() {
        // Transpose [1,2,3,4] (NCHW) to [1,3,4,2] (NHWC)
        // This is a common operation when converting between formats
        let input: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 24];
        
        super::transpose_f32(&mut output, &input, [1, 2, 3, 4], [0, 2, 3, 1]);
        
        // Original: input[n,c,h,w] at index n*24 + c*12 + h*4 + w
        // Output: output[n,h,w,c] at index n*24 + h*8 + w*2 + c
        // input[0,0,0,0] = 0 -> output[0,0,0,0] = 0
        // input[0,1,0,0] = 12 -> output[0,0,0,1] = 1st output position channel 1
        assert_eq!(output[0], 0.0);  // [0,0,0,0]
        assert_eq!(output[1], 12.0); // [0,0,0,1] = input[0,1,0,0]
    }

    #[test]
    fn sub_f32_basic() {
        let a: Vec<f32> = (0..100).map(|i| i as f32 * 2.0).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
        let mut y = vec![0.0f32; 100];
        
        super::sub_f32(&mut y, &a, &b);
        
        for i in 0..100 {
            let expected = a[i] - b[i];
            assert_eq!(y[i], expected, "sub_f32 mismatch at {i}");
        }
    }

    #[test]
    fn sub_f32_matches_scalar() {
        let a: Vec<f32> = (0..67).map(|i| i as f32 * 1.5).collect();
        let b: Vec<f32> = (0..67).map(|i| (i as f32).sin()).collect();
        let mut y_main = vec![0.0f32; 67];
        let mut y_scalar = vec![0.0f32; 67];
        
        super::sub_f32(&mut y_main, &a, &b);
        super::sub_f32_scalar(&mut y_scalar, &a, &b);
        
        for (i, (&a, &b)) in y_main.iter().zip(y_scalar.iter()).enumerate() {
            assert!(approx_eq(a, b), "sub_f32 mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn div_f32_basic() {
        let a: Vec<f32> = (1..101).map(|i| i as f32 * 2.0).collect();
        let b: Vec<f32> = (1..101).map(|i| i as f32 * 0.5).collect();
        let mut y = vec![0.0f32; 100];
        
        super::div_f32(&mut y, &a, &b);
        
        for i in 0..100 {
            let expected = a[i] / b[i];
            assert!(approx_eq(y[i], expected), "div_f32 mismatch at {i}: {} vs {}", y[i], expected);
        }
    }

    #[test]
    fn div_f32_matches_scalar() {
        let a: Vec<f32> = (1..68).map(|i| i as f32 * 1.5).collect();
        let b: Vec<f32> = (1..68).map(|i| (i as f32).sin().abs() + 0.1).collect(); // Avoid division by zero
        let mut y_main = vec![0.0f32; 67];
        let mut y_scalar = vec![0.0f32; 67];
        
        super::div_f32(&mut y_main, &a, &b);
        super::div_f32_scalar(&mut y_scalar, &a, &b);
        
        for (i, (&a, &b)) in y_main.iter().zip(y_scalar.iter()).enumerate() {
            assert!(approx_eq(a, b), "div_f32 mismatch at {i}: {a} vs {b}");
        }
    }
}

// ===========================================================================
// Convolution operations
// ===========================================================================

// ---------------------------------------------------------------------------
// conv2d_f32_nhwc: Direct convolution for NHWC layout
// ---------------------------------------------------------------------------

/// 2D convolution in NHWC format.
///
/// * `input`: `[N, H, W, C_in]` row-major
/// * `kernel`: `[K_h, K_w, C_in, C_out]` row-major
/// * `output`: `[N, H_out, W_out, C_out]` row-major
///
/// Parameters:
/// * `n`: batch size
/// * `h_in, w_in`: input spatial dimensions
/// * `c_in`: input channels
/// * `c_out`: output channels
/// * `k_h, k_w`: kernel height and width
/// * `stride_h, stride_w`: stride
/// * `pad_h, pad_w`: zero-padding (symmetric)
///
/// Output dimensions are computed as:
/// * `h_out = (h_in + 2*pad_h - k_h) / stride_h + 1`
/// * `w_out = (w_in + 2*pad_w - k_w) / stride_w + 1`
///
/// # Panics
///
/// Panics if buffer sizes don't match expected dimensions.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32_nhwc(
    output: &mut [f32],
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
) {
    // Compute output dimensions
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    // Verify buffer sizes
    assert_eq!(
        input.len(),
        n * h_in * w_in * c_in,
        "conv2d_f32_nhwc: input size mismatch"
    );
    assert_eq!(
        kernel.len(),
        k_h * k_w * c_in * c_out,
        "conv2d_f32_nhwc: kernel size mismatch"
    );
    assert_eq!(
        output.len(),
        n * h_out * w_out * c_out,
        "conv2d_f32_nhwc: output size mismatch"
    );

    // Direct convolution: iterate over each output position
    for batch in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                for oc in 0..c_out {
                    let mut acc: f32 = 0.0;

                    // Convolve over the kernel
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            // Input position (with padding offset)
                            let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                            let iw = (ow * stride_w + kw) as isize - pad_w as isize;

                            // Skip if outside input bounds (zero-padding)
                            if ih < 0 || ih >= h_in as isize || iw < 0 || iw >= w_in as isize {
                                continue;
                            }

                            let ih = ih as usize;
                            let iw = iw as usize;

                            // Sum over input channels
                            for ic in 0..c_in {
                                // Input index: [batch, ih, iw, ic] in NHWC
                                let input_idx =
                                    ((batch * h_in + ih) * w_in + iw) * c_in + ic;
                                // Kernel index: [kh, kw, ic, oc]
                                let kernel_idx =
                                    ((kh * k_w + kw) * c_in + ic) * c_out + oc;

                                acc = input[input_idx].mul_add(kernel[kernel_idx], acc);
                            }
                        }
                    }

                    // Output index: [batch, oh, ow, oc] in NHWC
                    let output_idx = ((batch * h_out + oh) * w_out + ow) * c_out + oc;
                    output[output_idx] = acc;
                }
            }
        }
    }
}

/// 2D convolution in NHWC format with F16 data.
///
/// Same as `conv2d_f32_nhwc` but with F16 inputs/outputs and F32 accumulator.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_fp16_nhwc(
    output: &mut [F16],
    input: &[F16],
    kernel: &[F16],
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
) {
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    assert_eq!(input.len(), n * h_in * w_in * c_in);
    assert_eq!(kernel.len(), k_h * k_w * c_in * c_out);
    assert_eq!(output.len(), n * h_out * w_out * c_out);

    for batch in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                for oc in 0..c_out {
                    let mut acc: f32 = 0.0;

                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                            let iw = (ow * stride_w + kw) as isize - pad_w as isize;

                            if ih < 0 || ih >= h_in as isize || iw < 0 || iw >= w_in as isize {
                                continue;
                            }

                            let ih = ih as usize;
                            let iw = iw as usize;

                            for ic in 0..c_in {
                                let input_idx = ((batch * h_in + ih) * w_in + iw) * c_in + ic;
                                let kernel_idx = ((kh * k_w + kw) * c_in + ic) * c_out + oc;

                                let i_f32 = input[input_idx].to_f32();
                                let k_f32 = kernel[kernel_idx].to_f32();
                                acc = i_f32.mul_add(k_f32, acc);
                            }
                        }
                    }

                    let output_idx = ((batch * h_out + oh) * w_out + ow) * c_out + oc;
                    output[output_idx] = F16::from_f32(acc);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pooling operations
// ---------------------------------------------------------------------------

/// 2D max pooling in NHWC format.
///
/// * `input`: `[N, H_in, W_in, C]`
/// * `output`: `[N, H_out, W_out, C]`
///
/// Pool size and stride are both `(pool_h, pool_w)`.
#[allow(clippy::too_many_arguments)]
pub fn maxpool2d_f32_nhwc(
    output: &mut [f32],
    input: &[f32],
    n: usize,
    h_in: usize,
    w_in: usize,
    c: usize,
    pool_h: usize,
    pool_w: usize,
    stride_h: usize,
    stride_w: usize,
) {
    let h_out = (h_in - pool_h) / stride_h + 1;
    let w_out = (w_in - pool_w) / stride_w + 1;

    assert_eq!(input.len(), n * h_in * w_in * c);
    assert_eq!(output.len(), n * h_out * w_out * c);

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
}

/// 2D average pooling in NHWC format.
#[allow(clippy::too_many_arguments)]
pub fn avgpool2d_f32_nhwc(
    output: &mut [f32],
    input: &[f32],
    n: usize,
    h_in: usize,
    w_in: usize,
    c: usize,
    pool_h: usize,
    pool_w: usize,
    stride_h: usize,
    stride_w: usize,
) {
    let h_out = (h_in - pool_h) / stride_h + 1;
    let w_out = (w_in - pool_w) / stride_w + 1;
    let pool_size = (pool_h * pool_w) as f32;

    assert_eq!(input.len(), n * h_in * w_in * c);
    assert_eq!(output.len(), n * h_out * w_out * c);

    for batch in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                for ch in 0..c {
                    let mut sum: f32 = 0.0;

                    for ph in 0..pool_h {
                        for pw in 0..pool_w {
                            let ih = oh * stride_h + ph;
                            let iw = ow * stride_w + pw;
                            let idx = ((batch * h_in + ih) * w_in + iw) * c + ch;
                            sum += input[idx];
                        }
                    }

                    let output_idx = ((batch * h_out + oh) * w_out + ow) * c + ch;
                    output[output_idx] = sum / pool_size;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Softmax along the last axis.
///
/// * `input`: `[..., N]` — any shape, softmax over the last dimension
/// * `output`: same shape as input
/// * `n`: size of the last dimension
///
/// For numerical stability, uses the max-subtraction trick.
pub fn softmax_f32(output: &mut [f32], input: &[f32], n: usize) {
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len() % n, 0);

    let num_rows = input.len() / n;

    for row in 0..num_rows {
        let start = row * n;
        let end = start + n;
        let row_in = &input[start..end];
        let row_out = &mut output[start..end];

        // Find max for numerical stability
        let max_val = row_in.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) and sum
        let mut sum: f32 = 0.0;
        for (out, &inp) in row_out.iter_mut().zip(row_in.iter()) {
            let exp_val = (inp - max_val).exp();
            *out = exp_val;
            sum += exp_val;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        for out in row_out.iter_mut() {
            *out *= inv_sum;
        }
    }
}

// ===========================================================================
// Multi-threaded ops
// ===========================================================================

use crate::pool::{num_cpus, parallel_for_scoped, SendPtr};

/// Multi-threaded GEMM: `C = A * B`.
///
/// Parallelizes over the M dimension (rows of C). Uses `std::thread::scope`
/// for scoped parallelism without 'static lifetime requirements.
///
/// * `a`: `[m, k]` row-major
/// * `b`: `[k, n]` row-major  
/// * `c`: `[m, n]` row-major (output)
/// * `num_threads`: number of threads to use (clamped to 1..=num_cpus)
///
/// # Panics
///
/// Panics if buffer sizes don't match.
pub fn gemm_f32_mt(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    num_threads: usize,
) {
    assert_eq!(a.len(), m * k, "gemm_f32_mt: A size mismatch");
    assert_eq!(b.len(), k * n, "gemm_f32_mt: B size mismatch");
    assert_eq!(c.len(), m * n, "gemm_f32_mt: C size mismatch");

    // For small matrices, single-threaded is faster due to thread overhead
    if m * n * k < 4096 || num_threads <= 1 {
        gemm_f32_naive(c, a, b, m, n, k);
        return;
    }

    let num_threads = num_threads.min(m).min(num_cpus());

    // SAFETY: We partition the output by row so threads write non-overlapping regions.
    // Input arrays are read-only and shared safely. The pointers remain valid
    // for the scope of parallel_for_scoped (scoped threads).
    let c_ptr = unsafe { SendPtr::new(c.as_mut_ptr()) };
    let a_ptr = unsafe { SendPtr::from_const(a.as_ptr()) };
    let b_ptr = unsafe { SendPtr::from_const(b.as_ptr()) };

    parallel_for_scoped(num_threads, m, |row_start, row_end| {
        // SAFETY: Each thread writes to non-overlapping rows of c.
        // a and b are read-only and shared safely.
        unsafe {
            for i in row_start..row_end {
                for j in 0..n {
                    let mut acc: f32 = 0.0;
                    for p in 0..k {
                        let a_val = *a_ptr.as_const_ptr().add(i * k + p);
                        let b_val = *b_ptr.as_const_ptr().add(p * n + j);
                        acc = a_val.mul_add(b_val, acc);
                    }
                    *c_ptr.as_ptr().add(i * n + j) = acc;
                }
            }
        }
    });
}

/// Multi-threaded FP16 GEMM with F32 accumulator.
///
/// Same as `gemm_f32_mt` but with F16 inputs/outputs.
pub fn gemm_fp16_mt(
    c: &mut [F16],
    a: &[F16],
    b: &[F16],
    m: usize,
    n: usize,
    k: usize,
    num_threads: usize,
) {
    assert_eq!(a.len(), m * k, "gemm_fp16_mt: A size mismatch");
    assert_eq!(b.len(), k * n, "gemm_fp16_mt: B size mismatch");
    assert_eq!(c.len(), m * n, "gemm_fp16_mt: C size mismatch");

    if m * n * k < 4096 || num_threads <= 1 {
        gemm_fp16(c, a, b, m, n, k);
        return;
    }

    let num_threads = num_threads.min(m).min(num_cpus());

    // SAFETY: Same as gemm_f32_mt
    let c_ptr = unsafe { SendPtr::new(c.as_mut_ptr()) };
    let a_ptr = unsafe { SendPtr::from_const(a.as_ptr()) };
    let b_ptr = unsafe { SendPtr::from_const(b.as_ptr()) };

    parallel_for_scoped(num_threads, m, |row_start, row_end| {
        unsafe {
            for i in row_start..row_end {
                for j in 0..n {
                    let mut acc: f32 = 0.0;
                    for p in 0..k {
                        let a_f32 = (*a_ptr.as_const_ptr().add(i * k + p)).to_f32();
                        let b_f32 = (*b_ptr.as_const_ptr().add(p * n + j)).to_f32();
                        acc = a_f32.mul_add(b_f32, acc);
                    }
                    *c_ptr.as_ptr().add(i * n + j) = F16::from_f32(acc);
                }
            }
        }
    });
}

/// Multi-threaded 2D convolution in NHWC format.
///
/// Parallelizes over the H_out dimension (output rows). For small outputs,
/// falls back to single-threaded.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32_nhwc_mt(
    output: &mut [f32],
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
    num_threads: usize,
) {
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    assert_eq!(input.len(), n * h_in * w_in * c_in);
    assert_eq!(kernel.len(), k_h * k_w * c_in * c_out);
    assert_eq!(output.len(), n * h_out * w_out * c_out);

    // For small outputs, single-threaded is faster
    let total_work = n * h_out * w_out * c_out * k_h * k_w * c_in;
    if total_work < 8192 || num_threads <= 1 {
        conv2d_f32_nhwc(output, input, kernel, n, h_in, w_in, c_in, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w);
        return;
    }

    let num_threads = num_threads.min(n * h_out).min(num_cpus());

    // SAFETY: Same reasoning as gemm_f32_mt - threads write non-overlapping output rows
    let output_ptr = unsafe { SendPtr::new(output.as_mut_ptr()) };
    let input_ptr = unsafe { SendPtr::from_const(input.as_ptr()) };
    let kernel_ptr = unsafe { SendPtr::from_const(kernel.as_ptr()) };

    // Parallelize over batch * h_out
    let total_rows = n * h_out;

    parallel_for_scoped(num_threads, total_rows, |row_start, row_end| {
        unsafe {
            for row_idx in row_start..row_end {
                let batch = row_idx / h_out;
                let oh = row_idx % h_out;

                for ow in 0..w_out {
                    for oc in 0..c_out {
                        let mut acc: f32 = 0.0;

                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                let iw = (ow * stride_w + kw) as isize - pad_w as isize;

                                if ih < 0 || ih >= h_in as isize || iw < 0 || iw >= w_in as isize {
                                    continue;
                                }

                                let ih = ih as usize;
                                let iw = iw as usize;

                                for ic in 0..c_in {
                                    let input_idx = ((batch * h_in + ih) * w_in + iw) * c_in + ic;
                                    let kernel_idx = ((kh * k_w + kw) * c_in + ic) * c_out + oc;

                                    let i_val = *input_ptr.as_const_ptr().add(input_idx);
                                    let k_val = *kernel_ptr.as_const_ptr().add(kernel_idx);
                                    acc = i_val.mul_add(k_val, acc);
                                }
                            }
                        }

                        let output_idx = ((batch * h_out + oh) * w_out + ow) * c_out + oc;
                        *output_ptr.as_ptr().add(output_idx) = acc;
                    }
                }
            }
        }
    });
}

// ===========================================================================
// YOLO Ops (Task 005)
// ===========================================================================

// ---------------------------------------------------------------------------
// sigmoid: y[i] = 1 / (1 + exp(-x[i]))
// ---------------------------------------------------------------------------

/// Element-wise sigmoid: `y[i] = 1 / (1 + exp(-x[i]))`.
///
/// # Panics
///
/// Panics if `y.len() != x.len()`.
pub fn sigmoid_f32(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len(), "sigmoid_f32: length mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sigmoid_f32_neon(y, x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sigmoid_f32_scalar(y, x);
    }
}

/// Portable scalar implementation of [`sigmoid_f32`].
pub fn sigmoid_f32_scalar(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len(), "sigmoid_f32_scalar: length mismatch");
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = 1.0 / (1.0 + (-*xi).exp());
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sigmoid_f32_neon(y: &mut [f32], x: &[f32]) {
    // NEON doesn't have a direct exp intrinsic, so we fall back to scalar
    // For production, consider using a polynomial approximation
    sigmoid_f32_scalar(y, x);
}

/// FP16 sigmoid with F32 compute.
pub fn sigmoid_fp16(y: &mut [F16], x: &[F16]) {
    assert_eq!(y.len(), x.len(), "sigmoid_fp16: length mismatch");
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        let xf = xi.to_f32();
        let result = 1.0 / (1.0 + (-xf).exp());
        *yi = F16::from_f32(result);
    }
}

// ---------------------------------------------------------------------------
// mul: y[i] = a[i] * b[i]
// ---------------------------------------------------------------------------

/// Element-wise multiplication: `y[i] = a[i] * b[i]`.
///
/// # Panics
///
/// Panics if lengths don't match.
pub fn mul_f32(y: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(y.len(), a.len(), "mul_f32: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "mul_f32: y/b length mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { mul_f32_neon(y, a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        mul_f32_scalar(y, a, b);
    }
}

/// Portable scalar implementation of [`mul_f32`].
pub fn mul_f32_scalar(y: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(y.len(), a.len(), "mul_f32_scalar: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "mul_f32_scalar: y/b length mismatch");
    for i in 0..y.len() {
        y[i] = a[i] * b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mul_f32_neon(y: &mut [f32], a: &[f32], b: &[f32]) {
    use core::arch::aarch64::{vmulq_f32, vld1q_f32, vst1q_f32};
    debug_assert_eq!(y.len(), a.len());
    debug_assert_eq!(y.len(), b.len());
    let len = y.len();
    unsafe {
        let mut i = 0;
        while i + 16 <= len {
            let py = y.as_mut_ptr().add(i);
            let pa = a.as_ptr().add(i);
            let pb = b.as_ptr().add(i);

            let a0 = vld1q_f32(pa);
            let a1 = vld1q_f32(pa.add(4));
            let a2 = vld1q_f32(pa.add(8));
            let a3 = vld1q_f32(pa.add(12));

            let b0 = vld1q_f32(pb);
            let b1 = vld1q_f32(pb.add(4));
            let b2 = vld1q_f32(pb.add(8));
            let b3 = vld1q_f32(pb.add(12));

            vst1q_f32(py, vmulq_f32(a0, b0));
            vst1q_f32(py.add(4), vmulq_f32(a1, b1));
            vst1q_f32(py.add(8), vmulq_f32(a2, b2));
            vst1q_f32(py.add(12), vmulq_f32(a3, b3));
            i += 16;
        }
        while i < len {
            *y.get_unchecked_mut(i) = *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
    }
}

/// FP16 element-wise multiplication.
pub fn mul_fp16(y: &mut [F16], a: &[F16], b: &[F16]) {
    assert_eq!(y.len(), a.len(), "mul_fp16: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "mul_fp16: y/b length mismatch");
    for i in 0..y.len() {
        let a_f32 = a[i].to_f32();
        let b_f32 = b[i].to_f32();
        y[i] = F16::from_f32(a_f32 * b_f32);
    }
}

// ---------------------------------------------------------------------------
// sub: y[i] = a[i] - b[i]
// ---------------------------------------------------------------------------

/// Element-wise subtraction: `y[i] = a[i] - b[i]`.
///
/// # Panics
///
/// Panics if lengths don't match.
pub fn sub_f32(y: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(y.len(), a.len(), "sub_f32: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "sub_f32: y/b length mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { sub_f32_neon(y, a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sub_f32_scalar(y, a, b);
    }
}

/// Portable scalar implementation of [`sub_f32`].
pub fn sub_f32_scalar(y: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(y.len(), a.len(), "sub_f32_scalar: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "sub_f32_scalar: y/b length mismatch");
    for i in 0..y.len() {
        y[i] = a[i] - b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sub_f32_neon(y: &mut [f32], a: &[f32], b: &[f32]) {
    use core::arch::aarch64::{vsubq_f32, vld1q_f32, vst1q_f32};
    debug_assert_eq!(y.len(), a.len());
    debug_assert_eq!(y.len(), b.len());
    let len = y.len();
    unsafe {
        let mut i = 0;
        while i + 16 <= len {
            let py = y.as_mut_ptr().add(i);
            let pa = a.as_ptr().add(i);
            let pb = b.as_ptr().add(i);

            let a0 = vld1q_f32(pa);
            let a1 = vld1q_f32(pa.add(4));
            let a2 = vld1q_f32(pa.add(8));
            let a3 = vld1q_f32(pa.add(12));

            let b0 = vld1q_f32(pb);
            let b1 = vld1q_f32(pb.add(4));
            let b2 = vld1q_f32(pb.add(8));
            let b3 = vld1q_f32(pb.add(12));

            vst1q_f32(py, vsubq_f32(a0, b0));
            vst1q_f32(py.add(4), vsubq_f32(a1, b1));
            vst1q_f32(py.add(8), vsubq_f32(a2, b2));
            vst1q_f32(py.add(12), vsubq_f32(a3, b3));
            i += 16;
        }
        while i < len {
            *y.get_unchecked_mut(i) = *a.get_unchecked(i) - *b.get_unchecked(i);
            i += 1;
        }
    }
}

/// FP16 element-wise subtraction.
pub fn sub_fp16(y: &mut [F16], a: &[F16], b: &[F16]) {
    assert_eq!(y.len(), a.len(), "sub_fp16: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "sub_fp16: y/b length mismatch");
    for i in 0..y.len() {
        let a_f32 = a[i].to_f32();
        let b_f32 = b[i].to_f32();
        y[i] = F16::from_f32(a_f32 - b_f32);
    }
}

// ---------------------------------------------------------------------------
// div: y[i] = a[i] / b[i]
// ---------------------------------------------------------------------------

/// Element-wise division: `y[i] = a[i] / b[i]`.
///
/// # Panics
///
/// Panics if lengths don't match.
pub fn div_f32(y: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(y.len(), a.len(), "div_f32: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "div_f32: y/b length mismatch");
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { div_f32_neon(y, a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        div_f32_scalar(y, a, b);
    }
}

/// Portable scalar implementation of [`div_f32`].
pub fn div_f32_scalar(y: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(y.len(), a.len(), "div_f32_scalar: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "div_f32_scalar: y/b length mismatch");
    for i in 0..y.len() {
        y[i] = a[i] / b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn div_f32_neon(y: &mut [f32], a: &[f32], b: &[f32]) {
    use core::arch::aarch64::{vdivq_f32, vld1q_f32, vst1q_f32};
    debug_assert_eq!(y.len(), a.len());
    debug_assert_eq!(y.len(), b.len());
    let len = y.len();
    unsafe {
        let mut i = 0;
        while i + 16 <= len {
            let py = y.as_mut_ptr().add(i);
            let pa = a.as_ptr().add(i);
            let pb = b.as_ptr().add(i);

            let a0 = vld1q_f32(pa);
            let a1 = vld1q_f32(pa.add(4));
            let a2 = vld1q_f32(pa.add(8));
            let a3 = vld1q_f32(pa.add(12));

            let b0 = vld1q_f32(pb);
            let b1 = vld1q_f32(pb.add(4));
            let b2 = vld1q_f32(pb.add(8));
            let b3 = vld1q_f32(pb.add(12));

            vst1q_f32(py, vdivq_f32(a0, b0));
            vst1q_f32(py.add(4), vdivq_f32(a1, b1));
            vst1q_f32(py.add(8), vdivq_f32(a2, b2));
            vst1q_f32(py.add(12), vdivq_f32(a3, b3));
            i += 16;
        }
        while i < len {
            *y.get_unchecked_mut(i) = *a.get_unchecked(i) / *b.get_unchecked(i);
            i += 1;
        }
    }
}

/// FP16 element-wise division.
pub fn div_fp16(y: &mut [F16], a: &[F16], b: &[F16]) {
    assert_eq!(y.len(), a.len(), "div_fp16: y/a length mismatch");
    assert_eq!(y.len(), b.len(), "div_fp16: y/b length mismatch");
    for i in 0..y.len() {
        let a_f32 = a[i].to_f32();
        let b_f32 = b[i].to_f32();
        y[i] = F16::from_f32(a_f32 / b_f32);
    }
}

// ---------------------------------------------------------------------------
// concat: concatenate tensors along an axis
// ---------------------------------------------------------------------------

/// Concatenate multiple tensors along an axis (NHWC format, typically axis=3 for channels).
///
/// * `output`: pre-allocated output buffer
/// * `inputs`: slice of input tensor data
/// * `shapes`: shapes of each input tensor (NHWC format)
/// * `axis`: axis to concatenate along (0=N, 1=H, 2=W, 3=C)
///
/// # Panics
///
/// Panics if shapes don't match on non-concat axes or buffer sizes are wrong.
pub fn concat_f32(
    output: &mut [f32],
    inputs: &[&[f32]],
    shapes: &[[usize; 4]], // Each input's [N, H, W, C]
    axis: usize,
) {
    assert!(axis < 4, "concat_f32: axis must be 0-3");
    assert!(!inputs.is_empty(), "concat_f32: need at least one input");
    
    let n = shapes[0][0];
    let h = shapes[0][1];
    let w = shapes[0][2];
    
    // Verify all inputs have same dims except on concat axis
    for shape in shapes {
        for (i, (&d1, &d2)) in shapes[0].iter().zip(shape.iter()).enumerate() {
            if i != axis {
                assert_eq!(d1, d2, "concat_f32: shapes must match except on concat axis");
            }
        }
    }
    
    // Calculate total size on concat axis
    let total_concat_dim: usize = shapes.iter().map(|s| s[axis]).sum();
    
    // Verify output size
    let mut output_shape = shapes[0];
    output_shape[axis] = total_concat_dim;
    let expected_output_len: usize = output_shape.iter().product();
    assert_eq!(output.len(), expected_output_len, "concat_f32: output size mismatch");
    
    // Verify input sizes
    for (input, shape) in inputs.iter().zip(shapes.iter()) {
        let expected: usize = shape.iter().product();
        assert_eq!(input.len(), expected, "concat_f32: input size mismatch");
    }
    
    match axis {
        3 => {
            // Concat along channels (most common for YOLO FPN)
            // For each (n, h, w) position, copy channels from each input sequentially
            for batch in 0..n {
                for y in 0..h {
                    for x in 0..w {
                        let mut out_offset = ((batch * h + y) * w + x) * total_concat_dim;
                        for (input, shape) in inputs.iter().zip(shapes.iter()) {
                            let c = shape[3];
                            let in_offset = ((batch * h + y) * w + x) * c;
                            output[out_offset..out_offset + c]
                                .copy_from_slice(&input[in_offset..in_offset + c]);
                            out_offset += c;
                        }
                    }
                }
            }
        }
        0 => {
            // Concat along batch - just copy each input sequentially
            let mut offset = 0;
            for input in inputs {
                output[offset..offset + input.len()].copy_from_slice(input);
                offset += input.len();
            }
        }
        1 => {
            // Concat along height
            let c = shapes[0][3];
            for batch in 0..n {
                let mut h_offset = 0;
                for (input, shape) in inputs.iter().zip(shapes.iter()) {
                    let h_in = shape[1];
                    for y in 0..h_in {
                        let total_h = total_concat_dim;
                        let out_start = ((batch * total_h + h_offset + y) * w) * c;
                        let in_start = (y * w) * c;
                        let row_len = w * c;
                        output[out_start..out_start + row_len]
                            .copy_from_slice(&input[batch * h_in * w * c + in_start..batch * h_in * w * c + in_start + row_len]);
                    }
                    h_offset += h_in;
                }
            }
        }
        2 => {
            // Concat along width
            let c = shapes[0][3];
            for batch in 0..n {
                for y in 0..h {
                    let mut w_offset = 0;
                    let total_w = total_concat_dim;
                    for (input, shape) in inputs.iter().zip(shapes.iter()) {
                        let w_in = shape[2];
                        let out_start = ((batch * h + y) * total_w + w_offset) * c;
                        let in_start = ((batch * h + y) * w_in) * c;
                        let row_len = w_in * c;
                        output[out_start..out_start + row_len]
                            .copy_from_slice(&input[in_start..in_start + row_len]);
                        w_offset += w_in;
                    }
                }
            }
        }
        _ => unreachable!(),
    }
}

/// FP16 concatenation.
pub fn concat_fp16(
    output: &mut [F16],
    inputs: &[&[F16]],
    shapes: &[[usize; 4]],
    axis: usize,
) {
    assert!(axis < 4, "concat_fp16: axis must be 0-3");
    assert!(!inputs.is_empty(), "concat_fp16: need at least one input");
    
    let n = shapes[0][0];
    let h = shapes[0][1];
    let w = shapes[0][2];
    let total_concat_dim: usize = shapes.iter().map(|s| s[axis]).sum();
    
    match axis {
        3 => {
            for batch in 0..n {
                for y in 0..h {
                    for x in 0..w {
                        let mut out_offset = ((batch * h + y) * w + x) * total_concat_dim;
                        for (input, shape) in inputs.iter().zip(shapes.iter()) {
                            let c = shape[3];
                            let in_offset = ((batch * h + y) * w + x) * c;
                            output[out_offset..out_offset + c]
                                .copy_from_slice(&input[in_offset..in_offset + c]);
                            out_offset += c;
                        }
                    }
                }
            }
        }
        0 => {
            let mut offset = 0;
            for input in inputs {
                output[offset..offset + input.len()].copy_from_slice(input);
                offset += input.len();
            }
        }
        _ => {
            // Convert to f32, concat, convert back (fallback for less common axes)
            let f32_inputs: Vec<Vec<f32>> = inputs.iter()
                .map(|inp| inp.iter().map(|v| v.to_f32()).collect())
                .collect();
            let f32_refs: Vec<&[f32]> = f32_inputs.iter().map(|v| v.as_slice()).collect();
            let mut f32_output = vec![0.0f32; output.len()];
            concat_f32(&mut f32_output, &f32_refs, shapes, axis);
            for (out, val) in output.iter_mut().zip(f32_output.iter()) {
                *out = F16::from_f32(*val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// resize: bilinear/nearest upsampling
// ---------------------------------------------------------------------------

/// Resize mode for upsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// Nearest-neighbor interpolation.
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
}

/// Resize a tensor using bilinear or nearest-neighbor interpolation (NHWC format).
///
/// * `output`: `[N, H_out, W_out, C]`
/// * `input`: `[N, H_in, W_in, C]`
/// * `mode`: interpolation mode
#[allow(clippy::too_many_arguments)]
pub fn resize_f32(
    output: &mut [f32],
    input: &[f32],
    n: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    c: usize,
    mode: ResizeMode,
) {
    assert_eq!(input.len(), n * h_in * w_in * c);
    assert_eq!(output.len(), n * h_out * w_out * c);
    
    let scale_h = h_in as f32 / h_out as f32;
    let scale_w = w_in as f32 / w_out as f32;
    
    match mode {
        ResizeMode::Nearest => {
            for batch in 0..n {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        // Map output coordinate to input
                        let ih = ((oh as f32 + 0.5) * scale_h - 0.5).round().max(0.0).min((h_in - 1) as f32) as usize;
                        let iw = ((ow as f32 + 0.5) * scale_w - 0.5).round().max(0.0).min((w_in - 1) as f32) as usize;
                        
                        let in_offset = ((batch * h_in + ih) * w_in + iw) * c;
                        let out_offset = ((batch * h_out + oh) * w_out + ow) * c;
                        output[out_offset..out_offset + c].copy_from_slice(&input[in_offset..in_offset + c]);
                    }
                }
            }
        }
        ResizeMode::Bilinear => {
            for batch in 0..n {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        // Map output coordinate to input (align_corners=false convention)
                        let ih_f = (oh as f32 + 0.5) * scale_h - 0.5;
                        let iw_f = (ow as f32 + 0.5) * scale_w - 0.5;
                        
                        let ih0 = ih_f.floor().max(0.0) as usize;
                        let iw0 = iw_f.floor().max(0.0) as usize;
                        let ih1 = (ih0 + 1).min(h_in - 1);
                        let iw1 = (iw0 + 1).min(w_in - 1);
                        
                        let fh = ih_f - ih_f.floor();
                        let fw = iw_f - iw_f.floor();
                        
                        // Bilinear weights
                        let w00 = (1.0 - fh) * (1.0 - fw);
                        let w01 = (1.0 - fh) * fw;
                        let w10 = fh * (1.0 - fw);
                        let w11 = fh * fw;
                        
                        let idx00 = ((batch * h_in + ih0) * w_in + iw0) * c;
                        let idx01 = ((batch * h_in + ih0) * w_in + iw1) * c;
                        let idx10 = ((batch * h_in + ih1) * w_in + iw0) * c;
                        let idx11 = ((batch * h_in + ih1) * w_in + iw1) * c;
                        let out_idx = ((batch * h_out + oh) * w_out + ow) * c;
                        
                        for ch in 0..c {
                            output[out_idx + ch] = 
                                w00 * input[idx00 + ch] +
                                w01 * input[idx01 + ch] +
                                w10 * input[idx10 + ch] +
                                w11 * input[idx11 + ch];
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// split: split tensor along an axis
// ---------------------------------------------------------------------------

/// Split a tensor along an axis (NHWC format).
///
/// * `outputs`: pre-allocated output buffers
/// * `input`: input tensor data
/// * `shape`: input shape [N, H, W, C]
/// * `axis`: axis to split along
/// * `split_sizes`: size of each split along the axis
#[allow(clippy::too_many_arguments)]
pub fn split_f32(
    outputs: &mut [&mut [f32]],
    input: &[f32],
    shape: [usize; 4],
    axis: usize,
    split_sizes: &[usize],
) {
    assert!(axis < 4, "split_f32: axis must be 0-3");
    assert_eq!(outputs.len(), split_sizes.len(), "split_f32: outputs and split_sizes must match");
    
    let [n, h, w, c] = shape;
    let total_split: usize = split_sizes.iter().sum();
    assert_eq!(total_split, shape[axis], "split_f32: split sizes must sum to axis dimension");
    
    match axis {
        3 => {
            // Split along channels
            for batch in 0..n {
                for y in 0..h {
                    for x in 0..w {
                        let in_base = ((batch * h + y) * w + x) * c;
                        let mut c_offset = 0;
                        for (out, &split_c) in outputs.iter_mut().zip(split_sizes.iter()) {
                            let out_base = ((batch * h + y) * w + x) * split_c;
                            out[out_base..out_base + split_c]
                                .copy_from_slice(&input[in_base + c_offset..in_base + c_offset + split_c]);
                            c_offset += split_c;
                        }
                    }
                }
            }
        }
        0 => {
            // Split along batch
            let elem_per_batch = h * w * c;
            let mut batch_offset = 0;
            for (out, &split_n) in outputs.iter_mut().zip(split_sizes.iter()) {
                let src_start = batch_offset * elem_per_batch;
                let src_end = src_start + split_n * elem_per_batch;
                out.copy_from_slice(&input[src_start..src_end]);
                batch_offset += split_n;
            }
        }
        _ => {
            // Generic implementation for other axes
            // This is less common in YOLO, so we use a simpler approach
            unimplemented!("split_f32: axis {} not yet implemented", axis);
        }
    }
}

// ---------------------------------------------------------------------------
// transpose: permute dimensions
// ---------------------------------------------------------------------------

/// Transpose a 4D tensor with arbitrary permutation (NHWC format).
///
/// * `output`: pre-allocated output buffer
/// * `input`: input tensor data
/// * `shape`: input shape [D0, D1, D2, D3]
/// * `perm`: permutation array, e.g., [0, 2, 3, 1] for NCHW->NHWC
pub fn transpose_f32(
    output: &mut [f32],
    input: &[f32],
    shape: [usize; 4],
    perm: [usize; 4],
) {
    let [d0, d1, d2, d3] = shape;
    let out_shape = [shape[perm[0]], shape[perm[1]], shape[perm[2]], shape[perm[3]]];
    
    let expected_len: usize = shape.iter().product();
    assert_eq!(input.len(), expected_len);
    assert_eq!(output.len(), expected_len);
    
    // Input strides (row-major)
    let in_strides = [d1 * d2 * d3, d2 * d3, d3, 1];
    // Output strides
    let out_strides = [
        out_shape[1] * out_shape[2] * out_shape[3],
        out_shape[2] * out_shape[3],
        out_shape[3],
        1,
    ];
    
    // Inverse permutation for index mapping
    let mut inv_perm = [0usize; 4];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }
    
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                for i3 in 0..d3 {
                    let in_idx = i0 * in_strides[0] + i1 * in_strides[1] + 
                                 i2 * in_strides[2] + i3 * in_strides[3];
                    
                    let indices = [i0, i1, i2, i3];
                    let out_indices = [indices[perm[0]], indices[perm[1]], 
                                       indices[perm[2]], indices[perm[3]]];
                    let out_idx = out_indices[0] * out_strides[0] + 
                                  out_indices[1] * out_strides[1] +
                                  out_indices[2] * out_strides[2] + 
                                  out_indices[3] * out_strides[3];
                    
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// slice: extract subtensor
// ---------------------------------------------------------------------------

/// Extract a slice from a tensor along specified axes.
///
/// * `output`: pre-allocated output buffer
/// * `input`: input tensor data  
/// * `input_shape`: input shape [N, H, W, C]
/// * `starts`: start indices for each axis
/// * `ends`: end indices for each axis
/// * `axes`: which axes to slice (others are kept fully)
/// * `steps`: step size for each axis (1 = all elements)
pub fn slice_f32(
    output: &mut [f32],
    input: &[f32],
    input_shape: [usize; 4],
    starts: &[isize],
    ends: &[isize],
    axes: &[usize],
    steps: &[isize],
) {
    let [n, h, w, c] = input_shape;
    
    // Compute effective ranges for each axis
    let mut ranges: [(usize, usize, usize); 4] = [
        (0, n, 1), (0, h, 1), (0, w, 1), (0, c, 1)
    ];
    
    for (i, &axis) in axes.iter().enumerate() {
        let dim = input_shape[axis] as isize;
        let start = if starts[i] < 0 { (dim + starts[i]).max(0) } else { starts[i].min(dim) } as usize;
        let end = if ends[i] < 0 { (dim + ends[i]).max(0) } else { ends[i].min(dim) } as usize;
        let step = steps.get(i).copied().unwrap_or(1).max(1) as usize;
        ranges[axis] = (start, end, step);
    }
    
    let mut out_idx = 0;
    for i0 in (ranges[0].0..ranges[0].1).step_by(ranges[0].2) {
        for i1 in (ranges[1].0..ranges[1].1).step_by(ranges[1].2) {
            for i2 in (ranges[2].0..ranges[2].1).step_by(ranges[2].2) {
                for i3 in (ranges[3].0..ranges[3].1).step_by(ranges[3].2) {
                    let in_idx = ((i0 * h + i1) * w + i2) * c + i3;
                    output[out_idx] = input[in_idx];
                    out_idx += 1;
                }
            }
        }
    }
}

/// Multi-threaded FP16 2D convolution.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_fp16_nhwc_mt(
    output: &mut [F16],
    input: &[F16],
    kernel: &[F16],
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
    num_threads: usize,
) {
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    assert_eq!(input.len(), n * h_in * w_in * c_in);
    assert_eq!(kernel.len(), k_h * k_w * c_in * c_out);
    assert_eq!(output.len(), n * h_out * w_out * c_out);

    let total_work = n * h_out * w_out * c_out * k_h * k_w * c_in;
    if total_work < 8192 || num_threads <= 1 {
        conv2d_fp16_nhwc(output, input, kernel, n, h_in, w_in, c_in, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w);
        return;
    }

    let num_threads = num_threads.min(n * h_out).min(num_cpus());

    // SAFETY: Same reasoning as above
    let output_ptr = unsafe { SendPtr::new(output.as_mut_ptr()) };
    let input_ptr = unsafe { SendPtr::from_const(input.as_ptr()) };
    let kernel_ptr = unsafe { SendPtr::from_const(kernel.as_ptr()) };

    let total_rows = n * h_out;

    parallel_for_scoped(num_threads, total_rows, |row_start, row_end| {
        unsafe {
            for row_idx in row_start..row_end {
                let batch = row_idx / h_out;
                let oh = row_idx % h_out;

                for ow in 0..w_out {
                    for oc in 0..c_out {
                        let mut acc: f32 = 0.0;

                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                let iw = (ow * stride_w + kw) as isize - pad_w as isize;

                                if ih < 0 || ih >= h_in as isize || iw < 0 || iw >= w_in as isize {
                                    continue;
                                }

                                let ih = ih as usize;
                                let iw = iw as usize;

                                for ic in 0..c_in {
                                    let input_idx = ((batch * h_in + ih) * w_in + iw) * c_in + ic;
                                    let kernel_idx = ((kh * k_w + kw) * c_in + ic) * c_out + oc;

                                    let i_f32 = (*input_ptr.as_const_ptr().add(input_idx)).to_f32();
                                    let k_f32 = (*kernel_ptr.as_const_ptr().add(kernel_idx)).to_f32();
                                    acc = i_f32.mul_add(k_f32, acc);
                                }
                            }
                        }

                        let output_idx = ((batch * h_out + oh) * w_out + ow) * c_out + oc;
                        *output_ptr.as_ptr().add(output_idx) = F16::from_f32(acc);
                    }
                }
            }
        }
    });
}

// ===========================================================================
// Depthwise Convolution (group == c_in)
// ===========================================================================

/// Depthwise 2D convolution in NHWC format (group == c_in).
///
/// Each input channel has its own filter (or `channel_multiplier` filters).
/// For MobileNet, channel_multiplier is typically 1.
///
/// * Input: `[N, H_in, W_in, C_in]` in NHWC format
/// * Kernel: `[k_h, k_w, C_in, channel_multiplier]`
/// * Output: `[N, H_out, W_out, C_in * channel_multiplier]`
///
/// When `channel_multiplier == 1`, this is a standard depthwise conv where
/// `c_out == c_in`.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d_f32_nhwc(
    output: &mut [f32],
    input: &[f32],
    kernel: &[f32],
    n: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    channel_multiplier: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let c_out = c_in * channel_multiplier;
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    assert_eq!(input.len(), n * h_in * w_in * c_in);
    assert_eq!(kernel.len(), k_h * k_w * c_in * channel_multiplier);
    assert_eq!(output.len(), n * h_out * w_out * c_out);

    for batch in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                for ic in 0..c_in {
                    for m in 0..channel_multiplier {
                        let oc = ic * channel_multiplier + m;
                        let mut acc: f32 = 0.0;

                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                let iw = (ow * stride_w + kw) as isize - pad_w as isize;

                                if ih < 0 || ih >= h_in as isize || iw < 0 || iw >= w_in as isize {
                                    continue;
                                }

                                let ih = ih as usize;
                                let iw = iw as usize;

                                // Input index: [batch, ih, iw, ic]
                                let input_idx = ((batch * h_in + ih) * w_in + iw) * c_in + ic;
                                // Kernel index: [kh, kw, ic, m]
                                let kernel_idx = ((kh * k_w + kw) * c_in + ic) * channel_multiplier + m;

                                acc = input[input_idx].mul_add(kernel[kernel_idx], acc);
                            }
                        }

                        // Output index: [batch, oh, ow, oc]
                        let output_idx = ((batch * h_out + oh) * w_out + ow) * c_out + oc;
                        output[output_idx] = acc;
                    }
                }
            }
        }
    }
}

/// Multi-threaded depthwise 2D convolution in NHWC format.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d_f32_nhwc_mt(
    output: &mut [f32],
    input: &[f32],
    kernel: &[f32],
    n: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    channel_multiplier: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    num_threads: usize,
) {
    let c_out = c_in * channel_multiplier;
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    assert_eq!(input.len(), n * h_in * w_in * c_in);
    assert_eq!(kernel.len(), k_h * k_w * c_in * channel_multiplier);
    assert_eq!(output.len(), n * h_out * w_out * c_out);

    let total_work = n * h_out * w_out * c_out * k_h * k_w;
    if total_work < 4096 || num_threads <= 1 {
        depthwise_conv2d_f32_nhwc(output, input, kernel, n, h_in, w_in, c_in, channel_multiplier,
            k_h, k_w, stride_h, stride_w, pad_h, pad_w);
        return;
    }

    let num_threads = num_threads.min(n * h_out).min(num_cpus());

    let output_ptr = unsafe { SendPtr::new(output.as_mut_ptr()) };
    let input_ptr = unsafe { SendPtr::from_const(input.as_ptr()) };
    let kernel_ptr = unsafe { SendPtr::from_const(kernel.as_ptr()) };

    let total_rows = n * h_out;

    parallel_for_scoped(num_threads, total_rows, |row_start, row_end| {
        unsafe {
            for row_idx in row_start..row_end {
                let batch = row_idx / h_out;
                let oh = row_idx % h_out;

                for ow in 0..w_out {
                    for ic in 0..c_in {
                        for m in 0..channel_multiplier {
                            let oc = ic * channel_multiplier + m;
                            let mut acc: f32 = 0.0;

                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                    let iw = (ow * stride_w + kw) as isize - pad_w as isize;

                                    if ih < 0 || ih >= h_in as isize || iw < 0 || iw >= w_in as isize {
                                        continue;
                                    }

                                    let ih = ih as usize;
                                    let iw = iw as usize;

                                    let input_idx = ((batch * h_in + ih) * w_in + iw) * c_in + ic;
                                    let kernel_idx = ((kh * k_w + kw) * c_in + ic) * channel_multiplier + m;

                                    let i_val = *input_ptr.as_const_ptr().add(input_idx);
                                    let k_val = *kernel_ptr.as_const_ptr().add(kernel_idx);
                                    acc = i_val.mul_add(k_val, acc);
                                }
                            }

                            let output_idx = ((batch * h_out + oh) * w_out + ow) * c_out + oc;
                            *output_ptr.as_ptr().add(output_idx) = acc;
                        }
                    }
                }
            }
        }
    });
}

/// Depthwise 2D convolution in NHWC format with F16 data.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d_fp16_nhwc_mt(
    output: &mut [F16],
    input: &[F16],
    kernel: &[F16],
    n: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    channel_multiplier: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    num_threads: usize,
) {
    let c_out = c_in * channel_multiplier;
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

    assert_eq!(input.len(), n * h_in * w_in * c_in);
    assert_eq!(kernel.len(), k_h * k_w * c_in * channel_multiplier);
    assert_eq!(output.len(), n * h_out * w_out * c_out);

    let total_work = n * h_out * w_out * c_out * k_h * k_w;
    if total_work < 4096 || num_threads <= 1 {
        // Inline single-threaded version
        for batch in 0..n {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    for ic in 0..c_in {
                        for m in 0..channel_multiplier {
                            let oc = ic * channel_multiplier + m;
                            let mut acc: f32 = 0.0;

                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                    let iw = (ow * stride_w + kw) as isize - pad_w as isize;

                                    if ih < 0 || ih >= h_in as isize || iw < 0 || iw >= w_in as isize {
                                        continue;
                                    }

                                    let ih = ih as usize;
                                    let iw = iw as usize;

                                    let input_idx = ((batch * h_in + ih) * w_in + iw) * c_in + ic;
                                    let kernel_idx = ((kh * k_w + kw) * c_in + ic) * channel_multiplier + m;

                                    acc = input[input_idx].to_f32().mul_add(kernel[kernel_idx].to_f32(), acc);
                                }
                            }

                            let output_idx = ((batch * h_out + oh) * w_out + ow) * c_out + oc;
                            output[output_idx] = F16::from_f32(acc);
                        }
                    }
                }
            }
        }
        return;
    }

    let num_threads = num_threads.min(n * h_out).min(num_cpus());

    let output_ptr = unsafe { SendPtr::new(output.as_mut_ptr()) };
    let input_ptr = unsafe { SendPtr::from_const(input.as_ptr()) };
    let kernel_ptr = unsafe { SendPtr::from_const(kernel.as_ptr()) };

    let total_rows = n * h_out;

    parallel_for_scoped(num_threads, total_rows, |row_start, row_end| {
        unsafe {
            for row_idx in row_start..row_end {
                let batch = row_idx / h_out;
                let oh = row_idx % h_out;

                for ow in 0..w_out {
                    for ic in 0..c_in {
                        for m in 0..channel_multiplier {
                            let oc = ic * channel_multiplier + m;
                            let mut acc: f32 = 0.0;

                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                    let iw = (ow * stride_w + kw) as isize - pad_w as isize;

                                    if ih < 0 || ih >= h_in as isize || iw < 0 || iw >= w_in as isize {
                                        continue;
                                    }

                                    let ih = ih as usize;
                                    let iw = iw as usize;

                                    let input_idx = ((batch * h_in + ih) * w_in + iw) * c_in + ic;
                                    let kernel_idx = ((kh * k_w + kw) * c_in + ic) * channel_multiplier + m;

                                    let i_f32 = (*input_ptr.as_const_ptr().add(input_idx)).to_f32();
                                    let k_f32 = (*kernel_ptr.as_const_ptr().add(kernel_idx)).to_f32();
                                    acc = i_f32.mul_add(k_f32, acc);
                                }
                            }

                            let output_idx = ((batch * h_out + oh) * w_out + ow) * c_out + oc;
                            *output_ptr.as_ptr().add(output_idx) = F16::from_f32(acc);
                        }
                    }
                }
            }
        }
    });
}
