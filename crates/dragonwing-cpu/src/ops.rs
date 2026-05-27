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
