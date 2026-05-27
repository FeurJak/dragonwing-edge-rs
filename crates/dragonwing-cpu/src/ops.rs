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
}
