//! Data type representation for tensor elements.
//!
//! # Design (task 003 decision)
//!
//! This module follows **Option B** from the task spec: carry a separate `Dtype`
//! parameter at op-call sites. This avoids combinatorial explosion of `BufferKind`
//! variants and keeps the backend trait simple (byte-oriented).
//!
//! # F16 representation
//!
//! The `F16` type is a transparent wrapper around `u16` storing IEEE 754
//! binary16 (half-precision) format. Conversion to/from `f32` uses pure-bits
//! software conversion.
//!
//! ## Why software conversion, not NEON intrinsics?
//!
//! The NEON `vcvt_f16_f32` / `vcvt_f32_f16` intrinsics require unstable Rust
//! features (`stdarch_neon_f16`). More importantly, Cortex-A53 (our target)
//! only has FP16 *storage* support, not FP16 *arithmetic* (no `armv8.2-a+fp16`).
//! This means even with hardware intrinsics, we'd still need to convert to F32
//! for any actual math. The software conversion is fast enough for our use case
//! (loading weights, not per-element compute).
//!
//! # Why no `half` crate
//!
//! The conversion is straightforward (~60 lines) and keeps the zero-dep
//! posture of `dragonwing-core`. The `half` crate would add a dependency
//! for no additional capability on our target hardware.

use core::fmt;

/// Element data type for tensor operations.
///
/// This enum describes how bytes in a buffer should be interpreted.
/// Ops carry this parameter explicitly rather than encoding it in
/// `BufferKind` to avoid combinatorial explosion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Dtype {
    /// 32-bit IEEE 754 single-precision floating point.
    F32,
    /// 16-bit IEEE 754 half-precision floating point.
    F16,
}

impl Dtype {
    /// Size of one element in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dtype::F32 => write!(f, "f32"),
            Dtype::F16 => write!(f, "f16"),
        }
    }
}

/// IEEE 754 binary16 (half-precision) floating-point value.
///
/// This is a transparent wrapper around `u16` storing the raw bit pattern.
/// It does **not** support arithmetic directly — convert to `f32` first.
///
/// # Conversion
///
/// * [`F16::from_f32`] rounds to nearest even (default IEEE rounding mode).
/// * [`F16::to_f32`] is exact for all finite values.
///
/// # Special values
///
/// * Positive zero: `F16(0x0000)`
/// * Negative zero: `F16(0x8000)`
/// * Positive infinity: `F16(0x7C00)`
/// * Negative infinity: `F16(0xFC00)`
/// * NaN (quiet, canonical): `F16(0x7E00)`
///
/// Denormals are supported but may be flushed to zero on some GPU hardware.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct F16(pub u16);

impl F16 {
    /// Positive zero.
    pub const ZERO: Self = Self(0x0000);
    /// Negative zero.
    pub const NEG_ZERO: Self = Self(0x8000);
    /// Positive infinity.
    pub const INFINITY: Self = Self(0x7C00);
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(0xFC00);
    /// Canonical quiet NaN.
    pub const NAN: Self = Self(0x7E00);
    /// Largest finite positive value: ~65504.
    pub const MAX: Self = Self(0x7BFF);
    /// Smallest positive normal value: ~6.1e-5.
    pub const MIN_POSITIVE: Self = Self(0x0400);
    /// One.
    pub const ONE: Self = Self(0x3C00);

    /// Convert an `f32` to `F16`, rounding to nearest even.
    ///
    /// Values outside the F16 range saturate to ±infinity.
    /// NaN inputs produce NaN output (sign preserved, payload truncated).
    ///
    /// # Implementation note
    ///
    /// Uses software conversion. The NEON `vcvt_f16_f32` intrinsic is
    /// available on AArch64 but requires unstable Rust features. Since
    /// Cortex-A53 (our target) only has FP16 *storage* support, not FP16
    /// *arithmetic*, we use software conversion everywhere for simplicity.
    /// The conversion is fast enough for our use case (weight loading,
    /// not per-element compute).
    #[must_use]
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        Self::from_f32_soft(v)
    }

    /// Convert `F16` to `f32`. This is exact for all finite values.
    #[must_use]
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.to_f32_soft()
    }

    /// Software conversion from f32 to f16.
    #[must_use]
    fn from_f32_soft(v: f32) -> Self {
        let bits = v.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x007F_FFFF;

        if exp == 255 {
            // Inf or NaN
            if mant == 0 {
                // Infinity
                return Self((sign | 0x7C00) as u16);
            }
            // NaN — preserve sign, use quiet NaN with some mantissa bits
            return Self((sign | 0x7E00 | (mant >> 13).min(0x1FF)) as u16);
        }

        // Bias conversion: f32 bias = 127, f16 bias = 15
        let new_exp = exp - 127 + 15;

        if new_exp >= 31 {
            // Overflow to infinity
            return Self((sign | 0x7C00) as u16);
        }

        if new_exp <= 0 {
            // Denormal or underflow
            if new_exp < -10 {
                // Too small, flush to zero
                return Self(sign as u16);
            }
            // Denormal: shift mantissa right
            let shift = 1 - new_exp;
            let mant_with_hidden = mant | 0x0080_0000; // Add hidden bit
            let shifted = mant_with_hidden >> (13 + shift);
            // Round to nearest even
            let round_bit = (mant_with_hidden >> (12 + shift)) & 1;
            let sticky = if (mant_with_hidden & ((1 << (12 + shift)) - 1)) != 0 { 1 } else { 0 };
            let rounded = shifted + ((round_bit & (sticky | (shifted & 1))) as u32);
            return Self((sign | rounded) as u16);
        }

        // Normal number
        let mant_16 = mant >> 13;
        // Round to nearest even
        let round_bit = (mant >> 12) & 1;
        let sticky = if (mant & 0x0FFF) != 0 { 1 } else { 0 };
        let rounded = mant_16 + ((round_bit & (sticky | (mant_16 & 1))) as u32);

        if rounded >= 0x400 {
            // Mantissa overflow, increment exponent
            let new_exp = new_exp + 1;
            if new_exp >= 31 {
                return Self((sign | 0x7C00) as u16);
            }
            return Self((sign | ((new_exp as u32) << 10)) as u16);
        }

        Self((sign | ((new_exp as u32) << 10) | rounded) as u16)
    }

    /// Software conversion from f16 to f32.
    #[must_use]
    fn to_f32_soft(self) -> f32 {
        let bits = self.0 as u32;
        let sign = (bits & 0x8000) << 16;
        let exp = (bits >> 10) & 0x1F;
        let mant = bits & 0x03FF;

        if exp == 0 {
            if mant == 0 {
                // Zero (positive or negative)
                return f32::from_bits(sign);
            }
            // Denormal: normalize
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e += 1;
            }
            // Remove hidden bit
            let mant_32 = (m & 0x03FF) << 13;
            let exp_32 = (127 - 15 - e) as u32;
            return f32::from_bits(sign | (exp_32 << 23) | mant_32);
        }

        if exp == 31 {
            if mant == 0 {
                // Infinity
                return f32::from_bits(sign | 0x7F80_0000);
            }
            // NaN
            return f32::from_bits(sign | 0x7FC0_0000 | (mant << 13));
        }

        // Normal number
        let exp_32 = (exp + 127 - 15) << 23;
        let mant_32 = mant << 13;
        f32::from_bits(sign | exp_32 | mant_32)
    }

    /// Returns `true` if this value is NaN.
    #[must_use]
    #[inline]
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mant = self.0 & 0x03FF;
        exp == 31 && mant != 0
    }

    /// Returns `true` if this value is positive or negative infinity.
    #[must_use]
    #[inline]
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mant = self.0 & 0x03FF;
        exp == 31 && mant == 0
    }

    /// Returns `true` if this value is finite (not NaN or infinity).
    #[must_use]
    #[inline]
    pub fn is_finite(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        exp != 31
    }

    /// Returns `true` if this value is a denormal (subnormal).
    #[must_use]
    #[inline]
    pub fn is_subnormal(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mant = self.0 & 0x03FF;
        exp == 0 && mant != 0
    }
}

impl fmt::Debug for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F16({:.4})", self.to_f32())
    }
}

impl fmt::Display for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for F16 {
    #[inline]
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl From<F16> for f32 {
    #[inline]
    fn from(v: F16) -> Self {
        v.to_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_zero_roundtrip() {
        assert_eq!(F16::from_f32(0.0).to_f32(), 0.0);
        assert_eq!(F16::from_f32(-0.0).0, 0x8000); // Negative zero preserved
    }

    #[test]
    fn f16_one_roundtrip() {
        let one = F16::from_f32(1.0);
        assert_eq!(one.0, 0x3C00);
        assert!((one.to_f32() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn f16_neg_one_roundtrip() {
        let neg_one = F16::from_f32(-1.0);
        assert_eq!(neg_one.0, 0xBC00);
        assert!((neg_one.to_f32() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn f16_infinity() {
        let pos_inf = F16::from_f32(f32::INFINITY);
        assert_eq!(pos_inf.0, 0x7C00);
        assert!(pos_inf.is_infinite());
        assert!(pos_inf.to_f32().is_infinite() && pos_inf.to_f32() > 0.0);

        let neg_inf = F16::from_f32(f32::NEG_INFINITY);
        assert_eq!(neg_inf.0, 0xFC00);
        assert!(neg_inf.is_infinite());
        assert!(neg_inf.to_f32().is_infinite() && neg_inf.to_f32() < 0.0);
    }

    #[test]
    fn f16_nan() {
        let nan = F16::from_f32(f32::NAN);
        assert!(nan.is_nan());
        assert!(nan.to_f32().is_nan());
    }

    #[test]
    fn f16_max_finite() {
        let max = F16::MAX;
        assert!(!max.is_infinite());
        assert!(!max.is_nan());
        // F16 max is approximately 65504
        let f = max.to_f32();
        assert!((f - 65504.0).abs() < 1.0);
    }

    #[test]
    fn f16_overflow_to_inf() {
        // Values larger than F16 max should become infinity
        let big = F16::from_f32(100000.0);
        assert!(big.is_infinite());
    }

    #[test]
    fn f16_underflow_to_zero() {
        // Very small values should become zero
        let tiny = F16::from_f32(1e-10);
        assert_eq!(tiny.to_f32(), 0.0);
    }

    #[test]
    fn f16_denormal() {
        // Smallest positive denormal in f16 is 2^-24 ≈ 5.96e-8
        let small = F16::from_f32(1e-6);
        assert!(small.is_subnormal() || small.0 == 0);
    }

    #[test]
    fn f16_representative_values() {
        // Test a range of representative values
        let test_cases: &[(f32, u16)] = &[
            (0.0, 0x0000),
            (1.0, 0x3C00),
            (-1.0, 0xBC00),
            (2.0, 0x4000),
            (0.5, 0x3800),
            (0.25, 0x3400),
        ];
        for &(f, expected_bits) in test_cases {
            let h = F16::from_f32(f);
            assert_eq!(
                h.0, expected_bits,
                "from_f32({f}) = 0x{:04X}, expected 0x{expected_bits:04X}",
                h.0
            );
        }
    }

    #[test]
    fn f16_roundtrip_accuracy() {
        // For values within F16 range, roundtrip should be close
        let test_values = [0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0];
        for &v in &test_values {
            let h = F16::from_f32(v);
            let back = h.to_f32();
            let rel_err = (back - v).abs() / v.abs().max(1e-6);
            assert!(
                rel_err < 0.001,
                "roundtrip({v}): got {back}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn dtype_size() {
        assert_eq!(Dtype::F32.size_bytes(), 4);
        assert_eq!(Dtype::F16.size_bytes(), 2);
    }
}
