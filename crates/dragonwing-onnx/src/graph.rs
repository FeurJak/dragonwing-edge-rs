//! Graph compilation and execution.
//!
//! This module handles:
//! 1. Two-phase compilation (validate + compile)
//! 2. Graph execution with proper tensor lifetime management

use crate::error::{Error, Result};
use crate::model::{Model, DataType};
use crate::builder::{
    BuildContext, TensorShape, CompiledOp,
    ValidationReport, UnsupportedReason, get_builder,
};
use dragonwing_core::Dtype;
use std::collections::HashMap;

/// A compiled graph ready for execution.
#[derive(Debug)]
pub struct Graph {
    /// Compiled ops in topological order.
    pub ops: Vec<CompiledOp>,
    /// Tensor shapes (name → shape).
    pub shapes: HashMap<String, TensorShape>,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Initializer data (name → raw bytes).
    pub initializers: HashMap<String, Vec<u8>>,
    /// The dtype the graph was compiled with.
    pub dtype: Dtype,
}

impl Graph {
    /// Get the input tensor names and shapes.
    pub fn input_info(&self) -> Vec<(&str, &TensorShape)> {
        self.inputs.iter()
            .filter_map(|name| self.shapes.get(name).map(|s| (name.as_str(), s)))
            .collect()
    }

    /// Get the output tensor names and shapes.
    pub fn output_info(&self) -> Vec<(&str, &TensorShape)> {
        self.outputs.iter()
            .filter_map(|name| self.shapes.get(name).map(|s| (name.as_str(), s)))
            .collect()
    }

    /// Get the total number of ops.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }
}

/// Validate a model without compiling it.
///
/// Returns a report of supported and unsupported ops.
pub fn validate_model(model: &Model, dtype: Dtype) -> ValidationReport {
    let initializers: HashMap<String, _> = model.initializers.iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect();
    
    let mut ctx = BuildContext::new(dtype, &initializers);
    let mut report = ValidationReport::default();

    // Register input shapes
    for input in &model.inputs {
        // Skip inputs that are actually initializers
        if initializers.contains_key(&input.name) {
            continue;
        }
        
        let shape = TensorShape::new(
            input.shape.iter().map(|&d| if d < 0 { 1 } else { d as usize }).collect(),
            dtype,
        );
        ctx.set_shape(input.name.clone(), shape);
    }

    // Register initializer shapes
    for init in &model.initializers {
        let shape = TensorShape::new(
            init.dims.iter().map(|&d| d as usize).collect(),
            onnx_dtype_to_dragonwing(init.data_type, dtype),
        );
        ctx.set_shape(init.name.clone(), shape);
    }

    // Validate each node
    for node in &model.nodes {
        let builder = match get_builder(&node.op_type) {
            Some(b) => b,
            None => {
                report.unsupported.push((
                    node.name.clone(),
                    UnsupportedReason::UnknownOpType,
                ));
                continue;
            }
        };

        match builder.is_supported(node, &ctx) {
            Ok(()) => {}
            Err(reason) => {
                report.unsupported.push((node.name.clone(), reason));
                continue;
            }
        }

        match builder.validate(node, &ctx) {
            Ok(shape) => {
                // Update context with output shape for downstream ops
                for output in &node.outputs {
                    ctx.set_shape(output.clone(), shape.clone());
                }
                report.supported.push(node.name.clone());
            }
            Err(e) => {
                report.unsupported.push((
                    node.name.clone(),
                    UnsupportedReason::Other(e.to_string()),
                ));
            }
        }
    }

    report
}

/// Compile a model into an executable graph.
///
/// This is the two-phase compilation:
/// 1. Validate all ops and compute shapes
/// 2. Build the compiled ops
pub fn compile_model(model: &Model, dtype: Dtype) -> Result<Graph> {
    // First, validate
    let report = validate_model(model, dtype);
    if !report.is_fully_supported() {
        let errors: Vec<String> = report.unsupported.iter()
            .map(|(name, reason)| format!("{name}: {reason}"))
            .collect();
        return Err(Error::Validation(format!(
            "Model has unsupported ops:\n  {}",
            errors.join("\n  ")
        )));
    }

    // Now compile
    let initializers: HashMap<String, _> = model.initializers.iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect();
    
    let mut ctx = BuildContext::new(dtype, &initializers);
    let mut ops = Vec::new();

    // Register input shapes
    let mut graph_inputs = Vec::new();
    for input in &model.inputs {
        if initializers.contains_key(&input.name) {
            continue;
        }
        
        let shape = TensorShape::new(
            input.shape.iter().map(|&d| if d < 0 { 1 } else { d as usize }).collect(),
            dtype,
        );
        ctx.set_shape(input.name.clone(), shape);
        graph_inputs.push(input.name.clone());
    }

    // Register initializer shapes
    for init in &model.initializers {
        let shape = TensorShape::new(
            init.dims.iter().map(|&d| d as usize).collect(),
            onnx_dtype_to_dragonwing(init.data_type, dtype),
        );
        ctx.set_shape(init.name.clone(), shape);
    }

    // Build each node
    for node in &model.nodes {
        let builder = get_builder(&node.op_type)
            .ok_or_else(|| Error::Compile(format!("no builder for {}", node.op_type)))?;
        
        let compiled = builder.build(node, &mut ctx)?;
        
        // Skip pure metadata ops (Reshape with no actual work)
        // Actually, keep them for now - the runtime will handle them
        ops.push(compiled);
    }

    // Collect output names
    let graph_outputs: Vec<String> = model.outputs.iter()
        .map(|o| o.name.clone())
        .collect();

    // Collect initializer data
    let initializer_data: HashMap<String, Vec<u8>> = model.initializers.iter()
        .map(|t| (t.name.clone(), t.data.clone()))
        .collect();

    Ok(Graph {
        ops,
        shapes: ctx.shapes,
        inputs: graph_inputs,
        outputs: graph_outputs,
        initializers: initializer_data,
        dtype,
    })
}

/// Convert ONNX DataType to dragonwing Dtype.
fn onnx_dtype_to_dragonwing(onnx_dtype: DataType, default: Dtype) -> Dtype {
    match onnx_dtype {
        DataType::Float => Dtype::F32,
        DataType::Float16 => Dtype::F16,
        // For weights stored as other types, use the graph's default dtype
        _ => default,
    }
}

/// Fold BatchNormalization into preceding Conv.
///
/// This modifies the model in place, removing BN nodes and adjusting
/// Conv weights/biases.
pub fn fold_batchnorm(model: &mut Model) -> Result<()> {
    let initializers: HashMap<String, _> = model.initializers.iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect();

    // Find Conv -> BN patterns
    let mut bn_to_remove = Vec::new();
    let mut conv_updates: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();

    for (i, node) in model.nodes.iter().enumerate() {
        if node.op_type != "BatchNormalization" {
            continue;
        }

        // Check if input comes from a Conv
        let input_name = &node.inputs[0];
        let conv_idx = model.nodes.iter()
            .position(|n| n.outputs.contains(input_name) && n.op_type == "Conv");
        
        let Some(conv_idx) = conv_idx else {
            // BN not after Conv - this is an error in task 004 scope
            return Err(Error::Compile(format!(
                "BatchNorm {} not preceded by Conv - cannot fold", node.name
            )));
        };

        let conv_node = &model.nodes[conv_idx];

        // Get BN parameters: scale, bias, mean, var
        // BN inputs: X, scale, B, mean, var
        if node.inputs.len() < 5 {
            return Err(Error::Compile(format!(
                "BatchNorm {} has insufficient inputs", node.name
            )));
        }

        let scale = initializers.get(&node.inputs[1])
            .and_then(|t| t.as_f32_slice())
            .ok_or_else(|| Error::Compile(format!("BN scale not found: {}", node.inputs[1])))?;
        let bn_bias = initializers.get(&node.inputs[2])
            .and_then(|t| t.as_f32_slice())
            .ok_or_else(|| Error::Compile(format!("BN bias not found: {}", node.inputs[2])))?;
        let mean = initializers.get(&node.inputs[3])
            .and_then(|t| t.as_f32_slice())
            .ok_or_else(|| Error::Compile(format!("BN mean not found: {}", node.inputs[3])))?;
        let var = initializers.get(&node.inputs[4])
            .and_then(|t| t.as_f32_slice())
            .ok_or_else(|| Error::Compile(format!("BN var not found: {}", node.inputs[4])))?;

        let epsilon = node.get_attr_float("epsilon", 1e-5);

        // Get Conv weight
        let conv_weight = initializers.get(&conv_node.inputs[1])
            .ok_or_else(|| Error::Compile(format!("Conv weight not found: {}", conv_node.inputs[1])))?;
        let conv_weight_data = conv_weight.as_f32_slice()
            .ok_or_else(|| Error::Compile("Conv weight must be f32".into()))?;

        // Get Conv bias (optional)
        let conv_bias = if conv_node.inputs.len() > 2 && !conv_node.inputs[2].is_empty() {
            initializers.get(&conv_node.inputs[2])
                .and_then(|t| t.as_f32_slice())
                .map(|s| s.to_vec())
        } else {
            None
        };

        // Fold BN into Conv
        // weight_new[c] = weight[c] * scale[c] / sqrt(var[c] + eps)
        // bias_new[c] = (bias[c] - mean[c]) * scale[c] / sqrt(var[c] + eps) + bn_bias[c]
        
        let c_out = conv_weight.dims[0] as usize;
        let weight_per_channel = conv_weight_data.len() / c_out;
        
        let mut new_weight = Vec::with_capacity(conv_weight_data.len());
        let mut new_bias = Vec::with_capacity(c_out);

        for c in 0..c_out {
            // Use f64 for intermediate calculations to avoid precision loss
            let std_inv = 1.0 / ((var[c] as f64 + epsilon as f64).sqrt());
            let scale_factor = scale[c] as f64 * std_inv;
            
            // Scale weights for this output channel
            let start = c * weight_per_channel;
            let end = start + weight_per_channel;
            for &w in &conv_weight_data[start..end] {
                new_weight.push((w as f64 * scale_factor) as f32);
            }
            
            // Compute new bias
            let old_bias = conv_bias.as_ref().map(|b| b[c]).unwrap_or(0.0) as f64;
            let new_b = (old_bias - mean[c] as f64) * scale_factor + bn_bias[c] as f64;
            new_bias.push(new_b as f32);
        }

        conv_updates.insert(conv_node.inputs[1].clone(), (new_weight, new_bias));
        bn_to_remove.push(i);
    }

    // Apply updates
    for (weight_name, (new_weight, new_bias)) in conv_updates {
        // Update weight initializer
        if let Some(init) = model.initializers.iter_mut().find(|t| t.name == weight_name) {
            let mut bytes = Vec::with_capacity(new_weight.len() * 4);
            for w in &new_weight {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            init.data = bytes;
        }

        // Find the conv node and its bias name
        for node in &mut model.nodes {
            if node.op_type == "Conv" && node.inputs.len() > 1 && node.inputs[1] == weight_name {
                // Add or update bias
                let bias_name = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                    node.inputs[2].clone()
                } else {
                    let name = format!("{}_bias_folded", weight_name);
                    if node.inputs.len() > 2 {
                        node.inputs[2] = name.clone();
                    } else {
                        node.inputs.push(name.clone());
                    }
                    name
                };

                // Update or add bias initializer
                let bias_dims = vec![new_bias.len() as i64];
                let mut bias_bytes = Vec::with_capacity(new_bias.len() * 4);
                for b in &new_bias {
                    bias_bytes.extend_from_slice(&b.to_le_bytes());
                }

                if let Some(init) = model.initializers.iter_mut().find(|t| t.name == bias_name) {
                    init.data = bias_bytes;
                } else {
                    model.initializers.push(crate::model::OnnxTensor {
                        name: bias_name,
                        dims: bias_dims,
                        data_type: DataType::Float,
                        data: bias_bytes,
                    });
                }
                break;
            }
        }
    }

    // Remove BN nodes (in reverse order to preserve indices)
    for &idx in bn_to_remove.iter().rev() {
        let bn_node = &model.nodes[idx];
        let bn_output = bn_node.outputs[0].clone();
        let bn_input = bn_node.inputs[0].clone();

        // Redirect all references from BN output to Conv output (which is BN input)
        for node in &mut model.nodes {
            for input in &mut node.inputs {
                if *input == bn_output {
                    *input = bn_input.clone();
                }
            }
        }

        model.nodes.remove(idx);
    }

    // Update outputs if they referenced BN outputs
    // (handled by the redirection above)

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape() {
        let shape = TensorShape::new(vec![1, 3, 224, 224], Dtype::F32);
        assert_eq!(shape.numel(), 1 * 3 * 224 * 224);
        assert_eq!(shape.size_bytes(), 1 * 3 * 224 * 224 * 4);
    }
}
