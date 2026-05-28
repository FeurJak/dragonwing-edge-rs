//! Quantization types and utilities for INT8 inference.
//!
//! # Quantization Scheme
//!
//! We use **symmetric per-channel quantization** for weights and
//! **symmetric per-tensor quantization** for activations.
//!
//! ## Symmetric Quantization
//!
//! ```text
//! x_float = x_int8 * scale
//! x_int8 = round(x_float / scale)
//! ```
//!
//! The zero point is always 0, which simplifies computations:
//! - No zero-point bias correction in convolutions
//! - Simpler shader/kernel code
//! - Minimal accuracy loss for ReLU-activated networks
//!
//! ## Per-Channel vs Per-Tensor
//!
//! - **Weights**: Per-channel scales (one scale per output channel)
//! - **Activations**: Per-tensor scales (one scale for the entire tensor)
//!
//! Per-channel weight quantization provides better accuracy because
//! different channels can have vastly different value ranges.
//!
//! # Implementation Notes
//!
//! This module is `no_std` compatible with `extern crate alloc`.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::collections::BTreeMap;
use libm::roundf;

/// Clamp a float value to a range.
/// 
/// This is a `no_std` compatible replacement for `f32::clamp`.
#[inline]
fn clamp_f32(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Quantization scale and optional zero point.
///
/// For symmetric quantization (our default), `zero_point` is always 0.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantScale {
    /// Scale factor: `x_float = x_quantized * scale`
    pub scale: f32,
    /// Zero point (always 0 for symmetric quantization).
    pub zero_point: i8,
}

impl QuantScale {
    /// Create a new symmetric quantization scale.
    #[must_use]
    pub const fn symmetric(scale: f32) -> Self {
        Self { scale, zero_point: 0 }
    }

    /// Create a quantization scale with explicit zero point.
    ///
    /// This is provided for compatibility with asymmetric quantization
    /// schemes, but symmetric (zero_point = 0) is preferred.
    #[must_use]
    pub const fn asymmetric(scale: f32, zero_point: i8) -> Self {
        Self { scale, zero_point }
    }

    /// Compute scale from the min/max range of values.
    ///
    /// For symmetric quantization, we use the max absolute value:
    /// ```text
    /// scale = max(|min|, |max|) / 127
    /// ```
    #[must_use]
    pub fn from_range(min: f32, max: f32) -> Self {
        let abs_max = min.abs().max(max.abs());
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
        Self::symmetric(scale)
    }

    /// Quantize a float value to int8.
    #[must_use]
    #[inline]
    pub fn quantize(&self, value: f32) -> i8 {
        let scaled = value / self.scale;
        let clamped = clamp_f32(scaled, -128.0, 127.0);
        roundf(clamped) as i8
    }

    /// Dequantize an int8 value to float.
    #[must_use]
    #[inline]
    pub fn dequantize(&self, value: i8) -> f32 {
        (value as i32 - self.zero_point as i32) as f32 * self.scale
    }
}

impl Default for QuantScale {
    fn default() -> Self {
        Self::symmetric(1.0)
    }
}

/// Per-channel quantization scales for a weight tensor.
///
/// Each output channel has its own scale factor. This provides
/// better accuracy than per-tensor quantization because different
/// channels can have very different value ranges.
#[derive(Debug, Clone, PartialEq)]
pub struct PerChannelScale {
    /// One scale per output channel.
    pub scales: Vec<f32>,
}

impl PerChannelScale {
    /// Create per-channel scales from a vector of scale values.
    #[must_use]
    pub fn new(scales: Vec<f32>) -> Self {
        Self { scales }
    }

    /// Number of channels.
    #[must_use]
    pub fn num_channels(&self) -> usize {
        self.scales.len()
    }

    /// Get the scale for a specific channel.
    #[must_use]
    pub fn get(&self, channel: usize) -> f32 {
        self.scales.get(channel).copied().unwrap_or(1.0)
    }

    /// Compute per-channel scales from weight tensor.
    ///
    /// The weight tensor is expected to be in [C_out, ...] layout
    /// where C_out is the number of output channels (first dimension).
    ///
    /// For each channel, we compute:
    /// ```text
    /// scale[c] = max(|weight[c, :]|) / 127
    /// ```
    #[must_use]
    pub fn from_weights(weights: &[f32], num_channels: usize) -> Self {
        let mut scales = Vec::with_capacity(num_channels);
        let per_channel = weights.len() / num_channels;

        for c in 0..num_channels {
            let start = c * per_channel;
            let end = start + per_channel;
            let abs_max = weights[start..end]
                .iter()
                .map(|&w| w.abs())
                .fold(0.0f32, f32::max);
            
            let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
            scales.push(scale);
        }

        Self { scales }
    }

    /// Quantize weights using per-channel scales.
    ///
    /// The weight tensor is expected to be in [C_out, ...] layout.
    /// Returns the quantized weights as i8.
    #[must_use]
    pub fn quantize_weights(&self, weights: &[f32], num_channels: usize) -> Vec<i8> {
        let mut quantized = Vec::with_capacity(weights.len());
        let per_channel = weights.len() / num_channels;

        for c in 0..num_channels {
            let scale = self.get(c);
            let start = c * per_channel;
            let end = start + per_channel;

            for &w in &weights[start..end] {
                let scaled = w / scale;
                let clamped = clamp_f32(scaled, -128.0, 127.0);
                quantized.push(roundf(clamped) as i8);
            }
        }

        quantized
    }
}

/// Quantization parameters for a model.
///
/// Contains scale information for all tensors that need quantization:
/// - Input scale (for quantizing float inputs)
/// - Weight scales (per-channel for each weight tensor)
/// - Activation scales (per-tensor for intermediate activations)
#[derive(Debug, Clone, Default)]
pub struct QuantizationParams {
    /// Scale for the model input tensor.
    pub input_scale: QuantScale,
    
    /// Per-channel scales for weight tensors.
    /// Key is the tensor name.
    pub weight_scales: BTreeMap<String, PerChannelScale>,
    
    /// Per-tensor scales for activation tensors.
    /// Key is the tensor name.
    pub activation_scales: BTreeMap<String, QuantScale>,
}

impl QuantizationParams {
    /// Create empty quantization parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input scale.
    pub fn set_input_scale(&mut self, scale: QuantScale) {
        self.input_scale = scale;
    }

    /// Add a weight tensor's per-channel scales.
    pub fn add_weight_scales(&mut self, name: impl Into<String>, scales: PerChannelScale) {
        self.weight_scales.insert(name.into(), scales);
    }

    /// Add an activation tensor's scale.
    pub fn add_activation_scale(&mut self, name: impl Into<String>, scale: QuantScale) {
        self.activation_scales.insert(name.into(), scale);
    }

    /// Get the weight scales for a tensor.
    #[must_use]
    pub fn get_weight_scales(&self, name: &str) -> Option<&PerChannelScale> {
        self.weight_scales.get(name)
    }

    /// Get the activation scale for a tensor.
    #[must_use]
    pub fn get_activation_scale(&self, name: &str) -> Option<&QuantScale> {
        self.activation_scales.get(name)
    }
}

/// Quantize a float slice to int8 using a single scale.
#[inline]
pub fn quantize_tensor(output: &mut [i8], input: &[f32], scale: &QuantScale) {
    debug_assert_eq!(output.len(), input.len());
    for (out, &inp) in output.iter_mut().zip(input.iter()) {
        *out = scale.quantize(inp);
    }
}

/// Dequantize an int8 slice to float using a single scale.
#[inline]
pub fn dequantize_tensor(output: &mut [f32], input: &[i8], scale: &QuantScale) {
    debug_assert_eq!(output.len(), input.len());
    for (out, &inp) in output.iter_mut().zip(input.iter()) {
        *out = scale.dequantize(inp);
    }
}

/// Requantize int32 accumulator to int8 with combined scale.
///
/// After INT8 convolution: `acc_i32 = sum(input_i8 * weight_i8)`
/// The combined scale is: `scale_combined = scale_input * scale_weight`
/// 
/// To get output in the new output scale:
/// ```text
/// output_i8 = round(acc_i32 * scale_combined / scale_output)
///           = round(acc_i32 * requant_scale)
/// ```
/// 
/// where `requant_scale = scale_combined / scale_output`
#[inline]
pub fn requantize_i32_to_i8(output: &mut [i8], input: &[i32], requant_scale: f32) {
    debug_assert_eq!(output.len(), input.len());
    for (out, &inp) in output.iter_mut().zip(input.iter()) {
        let scaled = (inp as f32) * requant_scale;
        let clamped = clamp_f32(scaled, -128.0, 127.0);
        *out = roundf(clamped) as i8;
    }
}

/// Pack four int8 values into a u32 for UINT32-packed storage.
///
/// This is used when the target doesn't support `VK_KHR_8bit_storage`.
/// Values are packed as: `[a, b, c, d]` -> `(d << 24) | (c << 16) | (b << 8) | a`
#[inline]
#[must_use]
pub const fn pack_i8x4_to_u32(a: i8, b: i8, c: i8, d: i8) -> u32 {
    let a = (a as u8) as u32;
    let b = (b as u8) as u32;
    let c = (c as u8) as u32;
    let d = (d as u8) as u32;
    a | (b << 8) | (c << 16) | (d << 24)
}

/// Unpack a u32 into four int8 values.
#[inline]
#[must_use]
pub const fn unpack_u32_to_i8x4(packed: u32) -> [i8; 4] {
    [
        (packed & 0xFF) as u8 as i8,
        ((packed >> 8) & 0xFF) as u8 as i8,
        ((packed >> 16) & 0xFF) as u8 as i8,
        ((packed >> 24) & 0xFF) as u8 as i8,
    ]
}

/// Pack an int8 slice into UINT32-packed format.
///
/// Input length must be a multiple of 4. If not, it will be padded with zeros.
#[must_use]
pub fn pack_i8_to_u32(input: &[i8]) -> Vec<u32> {
    let packed_len = (input.len() + 3) / 4;
    let mut output = Vec::with_capacity(packed_len);

    let mut i = 0;
    while i + 4 <= input.len() {
        output.push(pack_i8x4_to_u32(
            input[i],
            input[i + 1],
            input[i + 2],
            input[i + 3],
        ));
        i += 4;
    }

    // Handle remainder (pad with zeros)
    if i < input.len() {
        let a = input.get(i).copied().unwrap_or(0);
        let b = input.get(i + 1).copied().unwrap_or(0);
        let c = input.get(i + 2).copied().unwrap_or(0);
        let d = input.get(i + 3).copied().unwrap_or(0);
        output.push(pack_i8x4_to_u32(a, b, c, d));
    }

    output
}

/// Unpack UINT32-packed format to int8.
#[must_use]
pub fn unpack_u32_to_i8(input: &[u32], output_len: usize) -> Vec<i8> {
    let mut output = Vec::with_capacity(output_len);

    for &packed in input {
        let unpacked = unpack_u32_to_i8x4(packed);
        for &v in &unpacked {
            if output.len() >= output_len {
                break;
            }
            output.push(v);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_scale_symmetric() {
        let scale = QuantScale::symmetric(0.1);
        assert_eq!(scale.zero_point, 0);
        assert!((scale.scale - 0.1).abs() < 1e-6);
    }

    #[test]
    fn quant_scale_from_range() {
        // Range [-1.27, 1.27] should give scale ≈ 0.01
        let scale = QuantScale::from_range(-1.27, 1.27);
        assert!((scale.scale - 0.01).abs() < 1e-6);

        // Asymmetric range [-0.5, 2.0] should use max abs = 2.0
        let scale = QuantScale::from_range(-0.5, 2.0);
        assert!((scale.scale - (2.0 / 127.0)).abs() < 1e-6);
    }

    #[test]
    fn quant_scale_quantize_dequantize() {
        let scale = QuantScale::symmetric(0.1);
        
        // Quantize 1.0: 1.0 / 0.1 = 10 -> 10
        assert_eq!(scale.quantize(1.0), 10);
        
        // Quantize -1.5: -1.5 / 0.1 = -15 -> -15
        assert_eq!(scale.quantize(-1.5), -15);
        
        // Dequantize 10: 10 * 0.1 = 1.0
        let dequant = scale.dequantize(10);
        assert!((dequant - 1.0).abs() < 1e-6);

        // Roundtrip
        let original = 1.25f32;
        let quant = scale.quantize(original);
        let dequant = scale.dequantize(quant);
        // Should be close (quantization error)
        assert!((dequant - original).abs() < scale.scale);
    }

    #[test]
    fn quant_scale_clamps() {
        let scale = QuantScale::symmetric(0.01);
        
        // Value too large should clamp to 127
        assert_eq!(scale.quantize(100.0), 127);
        
        // Value too small should clamp to -128
        assert_eq!(scale.quantize(-100.0), -128);
    }

    #[test]
    fn per_channel_from_weights() {
        // 2 channels, 4 values each
        let weights = vec![
            // Channel 0: max abs = 1.27
            1.27, -0.5, 0.3, 0.1,
            // Channel 1: max abs = 2.54
            2.54, 1.0, -1.5, 0.2,
        ];
        
        let scales = PerChannelScale::from_weights(&weights, 2);
        assert_eq!(scales.num_channels(), 2);
        assert!((scales.get(0) - (1.27 / 127.0)).abs() < 1e-6);
        assert!((scales.get(1) - (2.54 / 127.0)).abs() < 1e-6);
    }

    #[test]
    fn per_channel_quantize_weights() {
        let weights = vec![
            1.27, 0.635, 0.0, -0.635,  // Channel 0
            2.54, 1.27, 0.0, -1.27,    // Channel 1
        ];
        
        let scales = PerChannelScale::from_weights(&weights, 2);
        let quantized = scales.quantize_weights(&weights, 2);
        
        // Channel 0: scale = 1.27/127 = 0.01
        // 1.27 / 0.01 = 127
        assert_eq!(quantized[0], 127);
        // 0.635 / 0.01 = 63.5 -> 64 (rounded)
        assert_eq!(quantized[1], 64);
        // 0 / 0.01 = 0
        assert_eq!(quantized[2], 0);
        // -0.635 / 0.01 = -63.5 -> -64
        assert_eq!(quantized[3], -64);
    }

    #[test]
    fn pack_unpack_i8() {
        let values: [i8; 4] = [10, -20, 30, -40];
        let packed = pack_i8x4_to_u32(values[0], values[1], values[2], values[3]);
        let unpacked = unpack_u32_to_i8x4(packed);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_unpack_slice() {
        let input: Vec<i8> = (0..17).map(|i| i as i8 - 8).collect();
        let packed = pack_i8_to_u32(&input);
        let unpacked = unpack_u32_to_i8(packed.as_slice(), input.len());
        assert_eq!(unpacked, input);
    }

    #[test]
    fn quantize_tensor_basic() {
        let input = [0.0, 0.5, 1.0, -0.5, -1.0];
        let mut output = [0i8; 5];
        let scale = QuantScale::symmetric(0.01);
        
        quantize_tensor(&mut output, &input, &scale);
        
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 50);
        assert_eq!(output[2], 100);
        assert_eq!(output[3], -50);
        assert_eq!(output[4], -100);
    }

    #[test]
    fn dequantize_tensor_basic() {
        let input = [0i8, 50, 100, -50, -100];
        let mut output = [0.0f32; 5];
        let scale = QuantScale::symmetric(0.01);
        
        dequantize_tensor(&mut output, &input, &scale);
        
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 0.5).abs() < 1e-6);
        assert!((output[2] - 1.0).abs() < 1e-6);
        assert!((output[3] + 0.5).abs() < 1e-6);
        assert!((output[4] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn requantize_i32_basic() {
        // Accumulator values after INT8 matmul
        let acc = [0i32, 1000, 5000, -2000, -5000];
        let mut output = [0i8; 5];
        
        // If input scale = 0.01, weight scale = 0.01, output scale = 0.01
        // combined = 0.01 * 0.01 = 0.0001
        // requant = 0.0001 / 0.01 = 0.01
        let requant_scale = 0.01;
        
        requantize_i32_to_i8(&mut output, &acc, requant_scale);
        
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 10);
        assert_eq!(output[2], 50);
        assert_eq!(output[3], -20);
        assert_eq!(output[4], -50);
    }
}
