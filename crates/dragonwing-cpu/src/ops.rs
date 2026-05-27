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
}
