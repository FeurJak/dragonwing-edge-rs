//! Quantized graph compilation.
//!
//! This module transforms a calibrated F32 graph into an INT8 quantized graph
//! ready for execution on CPU (NEON) or GPU (Vulkan packed INT8).
//!
//! # Workflow
//!
//! 1. Compile an F32 graph from ONNX model
//! 2. Run calibration to collect activation statistics
//! 3. Use `QuantizedGraphCompiler` to transform to INT8
//!
//! # Example
//!
//! ```ignore
//! use dragonwing_onnx::{compile_model, QuantizedGraphCompiler};
//! use dragonwing_onnx::calibration::Calibrator;
//!
//! // Compile F32 graph
//! let f32_graph = compile_model(&model, Dtype::F32)?;
//!
//! // Calibrate
//! let mut calibrator = Calibrator::new(f32_graph.clone())?;
//! for sample in calibration_data {
//!     calibrator.feed("input", &sample)?;
//! }
//! let quant_params = calibrator.compute_params()?;
//!
//! // Quantize
//! let compiler = QuantizedGraphCompiler::new(quant_params);
//! let i8_graph = compiler.compile(f32_graph)?;
//! ```
//!
//! # Quantization Strategy
//!
//! - **Weights**: Per-channel symmetric quantization (one scale per output channel)
//! - **Activations**: Per-tensor symmetric quantization (one scale per tensor)
//! - **Accumulator**: INT32 (for Conv/GEMM intermediate results)
//! - **Requantization**: After each Conv/GEMM, requantize INT32 → INT8

use crate::builder::{CompiledOp, OpParams};
use crate::error::{Error, Result};
use crate::graph::Graph;
use dragonwing_core::{Dtype, PerChannelScale, QuantScale, QuantizationParams};

/// Quantized graph compiler.
///
/// Transforms an F32 graph into an INT8 graph using calibrated quantization
/// parameters.
pub struct QuantizedGraphCompiler {
    /// Quantization parameters from calibration.
    params: QuantizationParams,
}

impl QuantizedGraphCompiler {
    /// Create a new quantized graph compiler.
    ///
    /// # Arguments
    /// * `params` - Quantization parameters from calibration
    pub fn new(params: QuantizationParams) -> Self {
        Self { params }
    }

    /// Compile an F32 graph into an INT8 graph.
    ///
    /// # Arguments
    /// * `graph` - The F32 graph to quantize
    ///
    /// # Returns
    /// A new graph with:
    /// - INT8 weights (quantized from F32)
    /// - INT8 activations (with scale metadata)
    /// - Requantization ops inserted after Conv/GEMM
    pub fn compile(&self, mut graph: Graph) -> Result<Graph> {
        if graph.dtype != Dtype::F32 {
            return Err(Error::Validation(
                "QuantizedGraphCompiler requires F32 input graph".into(),
            ));
        }

        // Quantize weights
        self.quantize_weights(&mut graph)?;

        // Update tensor shapes with quantization scales
        self.update_tensor_scales(&mut graph)?;

        // Transform ops for INT8 execution
        let new_ops = self.transform_ops(&graph)?;
        graph.ops = new_ops;

        // Update dtype
        graph.dtype = Dtype::I8;

        Ok(graph)
    }

    /// Quantize weight tensors in the graph.
    fn quantize_weights(&self, graph: &mut Graph) -> Result<()> {
        // Find all weight tensors (initializers used by Conv/Gemm)
        let weight_names: Vec<String> = graph
            .ops
            .iter()
            .filter(|op| matches!(op.op_type.as_str(), "Conv" | "Gemm" | "MatMul"))
            .filter_map(|op| op.inputs.get(1).cloned())
            .filter(|name| graph.initializers.contains_key(name))
            .collect();

        for weight_name in weight_names {
            if let Some(weight_data) = graph.initializers.get_mut(&weight_name) {
                if let Some(shape) = graph.shapes.get(&weight_name) {
                    // Get or compute per-channel scales
                    let scales = if let Some(s) = self.params.get_weight_scales(&weight_name) {
                        s.clone()
                    } else {
                        // Compute scales from weight data
                        let f32_data = bytes_to_f32(weight_data);
                        let num_channels = shape.dims[0]; // First dim is output channels
                        PerChannelScale::from_weights(&f32_data, num_channels)
                    };

                    // Quantize weights
                    let f32_weights = bytes_to_f32(weight_data);
                    let i8_weights = quantize_weights_per_channel(&f32_weights, &shape.dims, &scales);

                    // Update data (pack for Vulkan or keep raw for CPU)
                    *weight_data = i8_weights;

                    // Update shape with per-channel scales
                    if let Some(s) = graph.shapes.get_mut(&weight_name) {
                        s.dtype = Dtype::I8;
                        s.per_channel_scales = Some(scales);
                    }
                }
            }
        }

        Ok(())
    }

    /// Update tensor shapes with quantization scales.
    fn update_tensor_scales(&self, graph: &mut Graph) -> Result<()> {
        // Update activation tensor shapes
        for (name, shape) in graph.shapes.iter_mut() {
            // Skip weight tensors (already handled)
            if shape.per_channel_scales.is_some() {
                continue;
            }

            // Skip if not an activation (e.g., INT64 shape tensors)
            if !shape.dtype.is_float() && !shape.dtype.is_quantized() {
                continue;
            }

            // Get scale from calibration params
            if let Some(scale) = self.params.get_activation_scale(name) {
                shape.dtype = Dtype::I8;
                shape.scale = Some(scale.clone());
            } else {
                // Use a default scale for tensors without calibration data
                // This shouldn't happen if calibration was done properly
                shape.dtype = Dtype::I8;
                shape.scale = Some(QuantScale::symmetric(1.0 / 127.0));
            }
        }

        Ok(())
    }

    /// Transform ops for INT8 execution.
    ///
    /// This inserts requantization ops and updates op types for INT8.
    fn transform_ops(&self, graph: &Graph) -> Result<Vec<CompiledOp>> {
        let mut new_ops = Vec::new();

        for op in &graph.ops {
            match op.op_type.as_str() {
                "Conv" => {
                    // Conv stays as Conv but with INT8 inputs
                    new_ops.push(op.clone());

                    // Insert requantization after Conv
                    let requant_op = self.create_requant_op(op, graph)?;
                    new_ops.push(requant_op);
                }
                "Gemm" | "MatMul" => {
                    // GEMM/MatMul stays but with INT8 inputs
                    new_ops.push(op.clone());

                    // Insert requantization after GEMM
                    let requant_op = self.create_requant_op(op, graph)?;
                    new_ops.push(requant_op);
                }
                "Add" => {
                    // Add needs special handling for different input scales
                    let add_op = self.transform_add_op(op, graph)?;
                    new_ops.push(add_op);
                }
                "Relu" | "Clip" => {
                    // ReLU is straightforward in INT8 (max(0, x))
                    new_ops.push(op.clone());
                }
                "MaxPool" | "GlobalAveragePool" | "AveragePool" => {
                    // Pooling ops work similarly in INT8
                    new_ops.push(op.clone());
                }
                "Softmax" => {
                    // Softmax needs dequantization before and quantization after
                    // For now, keep as-is (will run in F32)
                    new_ops.push(op.clone());
                }
                "Reshape" | "Flatten" | "Transpose" | "Concat" | "Split" | "Slice" => {
                    // Shape ops just pass through
                    new_ops.push(op.clone());
                }
                _ => {
                    // Unknown op - keep as-is
                    new_ops.push(op.clone());
                }
            }
        }

        Ok(new_ops)
    }

    /// Create a requantization op after Conv/GEMM.
    fn create_requant_op(&self, preceding_op: &CompiledOp, graph: &Graph) -> Result<CompiledOp> {
        let output_name = preceding_op
            .outputs
            .first()
            .ok_or_else(|| Error::Compile("Op has no outputs".into()))?;

        // Get scales for requantization calculation
        let input_scale = self
            .get_tensor_scale(&preceding_op.inputs[0], graph)
            .unwrap_or(1.0 / 127.0);

        let weight_scale = self
            .get_weight_scale(&preceding_op.inputs[1], graph)
            .unwrap_or(1.0 / 127.0);

        let output_scale = self
            .get_tensor_scale(output_name, graph)
            .unwrap_or(1.0 / 127.0);

        // Requantization scale: (input_scale * weight_scale) / output_scale
        let requant_scale = (input_scale * weight_scale) / output_scale;

        // Create a new intermediate tensor name for INT32 accumulator
        let accum_name = format!("{}_i32", output_name);

        // The requant op converts INT32 accumulator to INT8
        Ok(CompiledOp {
            name: format!("{}_requant", preceding_op.name),
            op_type: "Requantize".to_string(),
            inputs: vec![accum_name],
            outputs: vec![output_name.clone()],
            params: OpParams::Requantize {
                scale: requant_scale,
            },
        })
    }

    /// Transform Add op for INT8.
    fn transform_add_op(&self, op: &CompiledOp, graph: &Graph) -> Result<CompiledOp> {
        let a_name = &op.inputs[0];
        let b_name = &op.inputs[1];
        let out_name = &op.outputs[0];

        let scale_a = self.get_tensor_scale(a_name, graph).unwrap_or(1.0 / 127.0);
        let scale_b = self.get_tensor_scale(b_name, graph).unwrap_or(1.0 / 127.0);
        let scale_out = self
            .get_tensor_scale(out_name, graph)
            .unwrap_or(1.0 / 127.0);

        // Precompute scale ratios
        let scale_a_over_out = scale_a / scale_out;
        let scale_b_over_out = scale_b / scale_out;

        Ok(CompiledOp {
            name: op.name.clone(),
            op_type: "AddQuantized".to_string(),
            inputs: op.inputs.clone(),
            outputs: op.outputs.clone(),
            params: OpParams::AddQuantized {
                scale_a_over_out,
                scale_b_over_out,
            },
        })
    }

    /// Get the quantization scale for a tensor.
    fn get_tensor_scale(&self, name: &str, graph: &Graph) -> Option<f32> {
        // Check calibration params first
        if let Some(scale) = self.params.get_activation_scale(name) {
            return Some(scale.scale);
        }

        // Check graph shapes
        if let Some(shape) = graph.shapes.get(name) {
            if let Some(scale) = &shape.scale {
                return Some(scale.scale);
            }
        }

        None
    }

    /// Get the per-channel scale for a weight tensor (returns average for simplicity).
    fn get_weight_scale(&self, name: &str, graph: &Graph) -> Option<f32> {
        // Check calibration params
        if let Some(scales) = self.params.get_weight_scales(name) {
            // Return average scale for requantization calculation
            let sum: f32 = scales.scales.iter().sum();
            return Some(sum / scales.scales.len() as f32);
        }

        // Check graph shapes
        if let Some(shape) = graph.shapes.get(name) {
            if let Some(scales) = &shape.per_channel_scales {
                let sum: f32 = scales.scales.iter().sum();
                return Some(sum / scales.scales.len() as f32);
            }
        }

        None
    }
}

/// Convert bytes to f32 slice.
fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Quantize weights using per-channel scales.
///
/// # Arguments
/// * `weights` - F32 weight data
/// * `shape` - Weight shape (C_out, C_in, H, W) or (C_out, C_in)
/// * `scales` - Per-channel scales (one per C_out)
///
/// # Returns
/// INT8 weight data as bytes
fn quantize_weights_per_channel(
    weights: &[f32],
    shape: &[usize],
    scales: &PerChannelScale,
) -> Vec<u8> {
    let num_channels = shape[0];
    let elements_per_channel = weights.len() / num_channels;

    let mut quantized = Vec::with_capacity(weights.len());

    for c in 0..num_channels {
        let scale = scales.scales[c];
        let inv_scale = 1.0 / scale;

        let start = c * elements_per_channel;
        let end = start + elements_per_channel;

        for &w in &weights[start..end] {
            let q = (w * inv_scale).round().clamp(-128.0, 127.0) as i8;
            quantized.push(q as u8);
        }
    }

    quantized
}

/// Information about a quantized graph.
#[derive(Debug)]
pub struct QuantizedGraphInfo {
    /// Number of quantized ops.
    pub num_quantized_ops: usize,
    /// Number of requantization ops inserted.
    pub num_requant_ops: usize,
    /// Total weight size reduction (F32 → I8).
    pub weight_size_reduction: f32,
    /// List of ops that remain in F32.
    pub f32_ops: Vec<String>,
}

impl QuantizedGraphInfo {
    /// Analyze a quantized graph.
    pub fn from_graph(graph: &Graph) -> Self {
        let num_requant_ops = graph
            .ops
            .iter()
            .filter(|op| op.op_type == "Requantize")
            .count();

        let f32_ops: Vec<String> = graph
            .ops
            .iter()
            .filter(|op| op.op_type == "Softmax") // Ops that stay in F32
            .map(|op| op.name.clone())
            .collect();

        let num_quantized_ops = graph.ops.len() - f32_ops.len();

        // Weight size reduction: F32 (4 bytes) → I8 (1 byte) = 75% reduction
        let weight_size_reduction = 0.75;

        Self {
            num_quantized_ops,
            num_requant_ops,
            weight_size_reduction,
            f32_ops,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_weights_per_channel() {
        // 2 output channels, 2 elements per channel
        let weights = vec![0.5f32, 1.0, -0.5, -1.0];
        let shape = vec![2, 2]; // [C_out, C_in]

        // Scale 0.01 means max representable is 1.27
        let scales = PerChannelScale {
            scales: vec![0.01, 0.01],
        };

        let quantized = quantize_weights_per_channel(&weights, &shape, &scales);

        // 0.5 / 0.01 = 50, 1.0 / 0.01 = 100
        // -0.5 / 0.01 = -50, -1.0 / 0.01 = -100
        assert_eq!(quantized.len(), 4);
        assert_eq!(quantized[0] as i8, 50);
        assert_eq!(quantized[1] as i8, 100);
        assert_eq!(quantized[2] as i8, -50);
        assert_eq!(quantized[3] as i8, -100);
    }

    #[test]
    fn test_bytes_to_f32() {
        let f32_val = 1.5f32;
        let bytes = f32_val.to_le_bytes().to_vec();

        let result = bytes_to_f32(&bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.5).abs() < 1e-6);
    }
}
