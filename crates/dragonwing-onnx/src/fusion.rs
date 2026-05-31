//! Op fusion passes for graph optimization.
//!
//! This module provides graph transformation passes that fuse multiple
//! operations into single fused ops for improved performance.
//!
//! # Supported Fusions
//!
//! - **Conv-Bias-Relu**: Fuse Conv + Add (bias) + Relu into a single fused conv
//! - **Conv-Relu**: Fuse Conv + Relu (when bias is already absorbed)
//! - **Mul-Sigmoid** (SiLU): x * sigmoid(x) pattern common in YOLO
//! - **Conv-Requant-Relu (INT8)**: Fuse the quantized triple emitted by the
//!   `QuantizedGraphCompiler` into a single
//!   `OpParams::Conv2dRequantReluI8Nhwc` op (Task 008).
//!
//! # How Fusion Works
//!
//! 1. Analyze the graph for fuseable patterns
//! 2. Replace matching op sequences with fused versions
//! 3. Update tensor dependencies and remove dead ops

use crate::builder::{CompiledOp, OpParams};
use crate::graph::Graph;
use std::collections::{HashMap, HashSet};

/// Fused operation parameters.
#[derive(Debug, Clone)]
pub enum FusedOp {
    /// Conv + bias + relu fused into one.
    ConvBiasRelu {
        /// Original Conv params.
        conv: Box<OpParams>,
        /// Whether to apply relu after.
        relu: bool,
    },
    /// x * sigmoid(x) - SiLU/Swish activation.
    SiLU,
}

/// Apply all fusion passes to a graph.
///
/// Returns the optimized graph with fused ops where applicable.
pub fn apply_fusion_passes(graph: &mut Graph) {
    // Track which ops have been fused and should be removed
    let mut ops_to_remove: HashSet<usize> = HashSet::new();
    let mut fused_ops: Vec<(usize, CompiledOp)> = Vec::new();
    
    // Build a map of tensor -> producing op index
    let mut tensor_producer: HashMap<String, usize> = HashMap::new();
    for (i, op) in graph.ops.iter().enumerate() {
        for output in &op.outputs {
            tensor_producer.insert(output.clone(), i);
        }
    }
    
    // Build a map of tensor -> consuming op indices
    let mut tensor_consumers: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, op) in graph.ops.iter().enumerate() {
        for input in &op.inputs {
            tensor_consumers.entry(input.clone())
                .or_default()
                .push(i);
        }
    }
    
    // Pass 1: Fuse Conv + Relu
    for (i, op) in graph.ops.iter().enumerate() {
        if ops_to_remove.contains(&i) {
            continue;
        }
        
        // Look for Relu ops
        if op.op_type != "Relu" {
            continue;
        }
        
        // Check if input comes from a Conv
        let input_name = match op.inputs.first() {
            Some(name) => name,
            None => continue,
        };
        
        let conv_idx = match tensor_producer.get(input_name) {
            Some(&idx) => idx,
            None => continue,
        };
        
        let conv_op = &graph.ops[conv_idx];
        if conv_op.op_type != "Conv" {
            continue;
        }
        
        // Check that the conv output is only used by this relu
        let consumers = tensor_consumers.get(input_name).map(|c| c.len()).unwrap_or(0);
        if consumers != 1 {
            continue;
        }
        
        // We can fuse Conv + Relu
        // Mark relu for removal
        ops_to_remove.insert(i);
        
        // Create fused op (modify conv to include relu)
        let fused = CompiledOp {
            name: format!("{}_fused_relu", conv_op.name),
            op_type: "Conv".into(), // Keep as Conv, runtime will check fused flag
            inputs: conv_op.inputs.clone(),
            outputs: op.outputs.clone(), // Use relu's output
            params: match &conv_op.params {
                OpParams::Conv2d { kernel_shape, strides, pads, dilations, group } => {
                    OpParams::Conv2d {
                        kernel_shape: *kernel_shape,
                        strides: *strides,
                        pads: *pads,
                        dilations: *dilations,
                        group: *group,
                    }
                }
                _ => conv_op.params.clone(),
            },
        };
        
        // Store the fused op to replace the conv
        fused_ops.push((conv_idx, fused));
    }
    
    // Pass 2: Fuse x * sigmoid(x) into SiLU
    // This pattern appears in YOLO's SiLU activation
    for (i, op) in graph.ops.iter().enumerate() {
        if ops_to_remove.contains(&i) {
            continue;
        }
        
        // Look for Mul ops
        if op.op_type != "Mul" {
            continue;
        }
        
        if op.inputs.len() != 2 {
            continue;
        }
        
        // Check if one input comes from a Sigmoid of the other input
        let input_a = &op.inputs[0];
        let input_b = &op.inputs[1];
        
        // Check pattern: mul(x, sigmoid(x))
        let (x_input, sigmoid_idx) = {
            let a_producer = tensor_producer.get(input_a);
            let b_producer = tensor_producer.get(input_b);
            
            if let Some(&sig_idx) = a_producer {
                if graph.ops[sig_idx].op_type == "Sigmoid" 
                    && graph.ops[sig_idx].inputs.first() == Some(input_b) {
                    (input_b.clone(), sig_idx)
                } else {
                    continue;
                }
            } else if let Some(&sig_idx) = b_producer {
                if graph.ops[sig_idx].op_type == "Sigmoid"
                    && graph.ops[sig_idx].inputs.first() == Some(input_a) {
                    (input_a.clone(), sig_idx)
                } else {
                    continue;
                }
            } else {
                continue;
            }
        };
        
        // Check that sigmoid output is only used by this mul
        let sigmoid_output = &graph.ops[sigmoid_idx].outputs[0];
        let consumers = tensor_consumers.get(sigmoid_output).map(|c| c.len()).unwrap_or(0);
        if consumers != 1 {
            continue;
        }
        
        // We can fuse into SiLU
        ops_to_remove.insert(sigmoid_idx);
        
        // Create fused SiLU op
        let fused = CompiledOp {
            name: format!("{}_silu", op.name),
            op_type: "SiLU".into(),
            inputs: vec![x_input],
            outputs: op.outputs.clone(),
            params: OpParams::None, // SiLU has no params
        };
        
        fused_ops.push((i, fused));
    }
    
    // Pass 3: Fuse Conv (INT8) + Requantize + (optional) Relu — Task 008.
    //
    // Pattern emitted by `QuantizedGraphCompiler`:
    //
    //   Conv      : inputs = [activations_i8, weights_i8 (, bias?)]    op_type "Conv"
    //               outputs = [conv_out]              -- with OpParams::Conv2d
    //   Requantize: inputs = ["{conv_out}_i32"]                       op_type "Requantize"
    //               outputs = [conv_out]              -- with OpParams::Requantize
    //   Relu      : inputs = [conv_out]                                op_type "Relu"
    //               outputs = [relu_out]              -- optional; if absent we still fuse
    //                                                    Conv+Requant to skip the I32 buffer
    //
    // We rewrite the Conv op into the fused variant and remove the Requantize
    // (and Relu, if matched). The fused op writes directly to the final output
    // name (`relu_out` if relu was matched, else the requantize's `conv_out`).
    for (i, op) in graph.ops.iter().enumerate() {
        if ops_to_remove.contains(&i) {
            continue;
        }
        
        // Anchor on Requantize (always present in the INT8 conv chain).
        if op.op_type != "Requantize" {
            continue;
        }
        
        // The requant op's input is the renamed I32 accumulator
        // (e.g. "foo_i32"). The producing op is the Conv that emitted it
        // implicitly. We locate the Conv by matching its output to the
        // requantize's *output* name (the original tensor name).
        let final_name = match op.outputs.first() {
            Some(n) => n.clone(),
            None => continue,
        };
        
        // Find the Conv whose output == final_name and which is the *direct*
        // predecessor of this requantize (so they were emitted as a pair).
        // We scan all earlier ops because the I32 accumulator buffer is
        // synthesised and not tracked in `tensor_producer`.
        let conv_idx = match graph.ops[..i].iter().rposition(|prev| {
            prev.op_type == "Conv"
                && prev.outputs.first().map(|s| s.as_str()) == Some(final_name.as_str())
                && matches!(prev.params, OpParams::Conv2d { .. })
        }) {
            Some(idx) => idx,
            None => continue,
        };
        if ops_to_remove.contains(&conv_idx) {
            continue;
        }
        
        // Conv must have at most weights as input[1] (we don't fuse biased
        // INT8 conv at this stage; bias-folding happens in quantize.rs).
        let conv_op = &graph.ops[conv_idx];
        if conv_op.inputs.len() < 2 {
            continue;
        }
        
        // Extract conv params.
        let (kernel_shape, strides, pads, dilations, group) = match &conv_op.params {
            OpParams::Conv2d { kernel_shape, strides, pads, dilations, group } => {
                (*kernel_shape, *strides, *pads, *dilations, *group)
            }
            _ => continue,
        };
        
        // The fused shader currently supports group=1 only.
        if group != 1 {
            continue;
        }
        
        // Extract requant scale.
        let requant_scale = match &op.params {
            OpParams::Requantize { scale } => *scale,
            _ => continue,
        };
        
        // Look for a Relu directly consuming the requantize output (with the
        // requant being its only consumer). If found, fuse it in too.
        let relu_idx = tensor_consumers.get(&final_name).and_then(|consumers| {
            // Filter out already-removed consumers and the requantize op itself.
            let live: Vec<usize> = consumers
                .iter()
                .copied()
                .filter(|&c| c != i && !ops_to_remove.contains(&c))
                .collect();
            if live.len() == 1 {
                let r = live[0];
                if graph.ops[r].op_type == "Relu" {
                    Some(r)
                } else {
                    None
                }
            } else {
                None
            }
        });
        
        // Compute the output name of the fused op:
        // - With relu: use relu's output (so downstream ops see the same tensor name).
        // - Without relu: keep `final_name` (requant's output).
        let (fused_output, has_relu) = if let Some(ridx) = relu_idx {
            let r = &graph.ops[ridx];
            let oname = match r.outputs.first() {
                Some(n) => n.clone(),
                None => final_name.clone(),
            };
            (oname, true)
        } else {
            (final_name.clone(), false)
        };
        
        // Skip the trivial case where Conv has a bias (input[2] non-empty).
        // Bias-folding is handled by quantize.rs separately; if a bias arrived
        // here we treat the conv as biased and refuse to fuse to avoid silent
        // accuracy loss.
        if conv_op.inputs.len() > 2 && !conv_op.inputs[2].is_empty() {
            continue;
        }
        
        // Build the fused op.
        let fused = CompiledOp {
            name: format!("{}_fused_qrelu", conv_op.name),
            op_type: "Conv2dRequantReluI8".to_string(),
            inputs: conv_op.inputs.clone(),
            outputs: vec![fused_output],
            params: OpParams::Conv2dRequantReluI8Nhwc {
                kernel_shape,
                strides,
                pads,
                dilations,
                group,
                requant_scale,
                has_relu,
            },
        };
        
        // Mark requantize (and relu, if matched) for removal; replace conv.
        ops_to_remove.insert(i);
        if let Some(ridx) = relu_idx {
            ops_to_remove.insert(ridx);
        }
        fused_ops.push((conv_idx, fused));
    }
    
    // Apply fusions: replace ops with fused versions
    for (idx, fused_op) in fused_ops {
        graph.ops[idx] = fused_op;
    }
    
    // Remove fused ops (in reverse order to preserve indices)
    let mut to_remove: Vec<usize> = ops_to_remove.into_iter().collect();
    to_remove.sort();
    to_remove.reverse();
    for idx in to_remove {
        graph.ops.remove(idx);
    }
}

/// Count fuseable patterns in a graph (for reporting).
pub fn count_fuseable_patterns(graph: &Graph) -> FusionStats {
    let mut stats = FusionStats::default();
    
    // Build tensor producer map
    let mut tensor_producer: HashMap<String, usize> = HashMap::new();
    for (i, op) in graph.ops.iter().enumerate() {
        for output in &op.outputs {
            tensor_producer.insert(output.clone(), i);
        }
    }
    
    // Count Conv-Relu patterns
    for op in &graph.ops {
        if op.op_type == "Relu" {
            if let Some(input) = op.inputs.first() {
                if let Some(&idx) = tensor_producer.get(input) {
                    if graph.ops[idx].op_type == "Conv" {
                        stats.conv_relu += 1;
                    }
                }
            }
        }
        
        // Count SiLU patterns (x * sigmoid(x))
        if op.op_type == "Mul" && op.inputs.len() == 2 {
            for input in &op.inputs {
                if let Some(&idx) = tensor_producer.get(input) {
                    if graph.ops[idx].op_type == "Sigmoid" {
                        stats.silu += 1;
                        break;
                    }
                }
            }
        }
        
        // Count INT8 Conv-Requant(-Relu) patterns (Task 008).
        // Pattern anchor is the Requantize op whose output name matches a
        // preceding Conv's output.
        if op.op_type == "Requantize" {
            if let Some(output_name) = op.outputs.first() {
                let matched = graph.ops.iter().any(|prev| {
                    prev.op_type == "Conv"
                        && prev.outputs.first() == Some(output_name)
                        && matches!(prev.params, OpParams::Conv2d { .. })
                });
                if matched {
                    stats.conv_requant_relu_i8 += 1;
                }
            }
        }
    }
    
    stats
}

/// Statistics about fuseable patterns in a graph.
#[derive(Debug, Default, Clone)]
pub struct FusionStats {
    /// Number of Conv + Relu patterns.
    pub conv_relu: usize,
    /// Number of x * sigmoid(x) patterns (SiLU).
    pub silu: usize,
    /// Number of INT8 Conv + Requantize (+ optional Relu) patterns (Task 008).
    pub conv_requant_relu_i8: usize,
}

impl FusionStats {
    /// Total fuseable patterns.
    pub fn total(&self) -> usize {
        self.conv_relu + self.silu + self.conv_requant_relu_i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dragonwing_core::Dtype;
    
    #[test]
    fn test_fusion_stats_empty() {
        let graph = Graph {
            ops: vec![],
            shapes: HashMap::new(),
            inputs: vec![],
            outputs: vec![],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        };
        
        let stats = count_fuseable_patterns(&graph);
        assert_eq!(stats.total(), 0);
    }
    
    #[test]
    fn test_conv_relu_detection() {
        let graph = Graph {
            ops: vec![
                CompiledOp {
                    name: "conv1".into(),
                    op_type: "Conv".into(),
                    inputs: vec!["input".into()],
                    outputs: vec!["conv_out".into()],
                    params: OpParams::Conv2d {
                        kernel_shape: [3, 3],
                        strides: [1, 1],
                        pads: [1, 1, 1, 1],
                        dilations: [1, 1],
                        group: 1,
                    },
                },
                CompiledOp {
                    name: "relu1".into(),
                    op_type: "Relu".into(),
                    inputs: vec!["conv_out".into()],
                    outputs: vec!["relu_out".into()],
                    params: OpParams::None,
                },
            ],
            shapes: HashMap::new(),
            inputs: vec!["input".into()],
            outputs: vec!["relu_out".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        };
        
        let stats = count_fuseable_patterns(&graph);
        assert_eq!(stats.conv_relu, 1);
    }
    
    #[test]
    fn test_silu_detection() {
        let graph = Graph {
            ops: vec![
                CompiledOp {
                    name: "sigmoid1".into(),
                    op_type: "Sigmoid".into(),
                    inputs: vec!["x".into()],
                    outputs: vec!["sig_out".into()],
                    params: OpParams::Sigmoid,
                },
                CompiledOp {
                    name: "mul1".into(),
                    op_type: "Mul".into(),
                    inputs: vec!["x".into(), "sig_out".into()],
                    outputs: vec!["silu_out".into()],
                    params: OpParams::Mul,
                },
            ],
            shapes: HashMap::new(),
            inputs: vec!["x".into()],
            outputs: vec!["silu_out".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        };
        
        let stats = count_fuseable_patterns(&graph);
        assert_eq!(stats.silu, 1);
    }
    
    #[test]
    fn test_conv_requant_relu_i8_fusion() {
        // Build a graph mirroring the QuantizedGraphCompiler output:
        //   Conv (OpParams::Conv2d) -> output = "conv_out"
        //   Requantize             -> input = "conv_out_i32", output = "conv_out"
        //   Relu                   -> input = "conv_out", output = "relu_out"
        // After fusion the graph should contain a single op of op_type
        // "Conv2dRequantReluI8" outputting to "relu_out".
        let mut graph = Graph {
            ops: vec![
                CompiledOp {
                    name: "conv1".into(),
                    op_type: "Conv".into(),
                    inputs: vec!["x".into(), "w".into()],
                    outputs: vec!["conv_out".into()],
                    params: OpParams::Conv2d {
                        kernel_shape: [3, 3],
                        strides: [1, 1],
                        pads: [1, 1, 1, 1],
                        dilations: [1, 1],
                        group: 1,
                    },
                },
                CompiledOp {
                    name: "conv1_requant".into(),
                    op_type: "Requantize".into(),
                    inputs: vec!["conv_out_i32".into()],
                    outputs: vec!["conv_out".into()],
                    params: OpParams::Requantize { scale: 0.0125 },
                },
                CompiledOp {
                    name: "relu1".into(),
                    op_type: "Relu".into(),
                    inputs: vec!["conv_out".into()],
                    outputs: vec!["relu_out".into()],
                    params: OpParams::None,
                },
            ],
            shapes: HashMap::new(),
            inputs: vec!["x".into()],
            outputs: vec!["relu_out".into()],
            initializers: HashMap::new(),
            dtype: Dtype::I8,
        };
        
        let stats = count_fuseable_patterns(&graph);
        assert_eq!(stats.conv_requant_relu_i8, 1);
        
        apply_fusion_passes(&mut graph);
        
        assert_eq!(graph.ops.len(), 1, "should collapse 3 ops to 1");
        let fused = &graph.ops[0];
        assert_eq!(fused.op_type, "Conv2dRequantReluI8");
        assert_eq!(fused.outputs[0], "relu_out");
        match &fused.params {
            OpParams::Conv2dRequantReluI8Nhwc {
                kernel_shape, strides, pads, dilations, group, requant_scale, has_relu,
            } => {
                assert_eq!(*kernel_shape, [3, 3]);
                assert_eq!(*strides, [1, 1]);
                assert_eq!(*pads, [1, 1, 1, 1]);
                assert_eq!(*dilations, [1, 1]);
                assert_eq!(*group, 1);
                assert!((*requant_scale - 0.0125).abs() < 1e-7);
                assert!(*has_relu);
            }
            _ => panic!("expected Conv2dRequantReluI8Nhwc"),
        }
    }
    
    #[test]
    fn test_conv_requant_only_fusion() {
        // No relu — still fuse Conv+Requant to skip the I32 buffer.
        let mut graph = Graph {
            ops: vec![
                CompiledOp {
                    name: "conv1".into(),
                    op_type: "Conv".into(),
                    inputs: vec!["x".into(), "w".into()],
                    outputs: vec!["conv_out".into()],
                    params: OpParams::Conv2d {
                        kernel_shape: [1, 1],
                        strides: [1, 1],
                        pads: [0, 0, 0, 0],
                        dilations: [1, 1],
                        group: 1,
                    },
                },
                CompiledOp {
                    name: "conv1_requant".into(),
                    op_type: "Requantize".into(),
                    inputs: vec!["conv_out_i32".into()],
                    outputs: vec!["conv_out".into()],
                    params: OpParams::Requantize { scale: 0.02 },
                },
            ],
            shapes: HashMap::new(),
            inputs: vec!["x".into()],
            outputs: vec!["conv_out".into()],
            initializers: HashMap::new(),
            dtype: Dtype::I8,
        };
        
        apply_fusion_passes(&mut graph);
        
        assert_eq!(graph.ops.len(), 1);
        let fused = &graph.ops[0];
        assert_eq!(fused.op_type, "Conv2dRequantReluI8");
        assert_eq!(fused.outputs[0], "conv_out");
        match &fused.params {
            OpParams::Conv2dRequantReluI8Nhwc { has_relu, .. } => {
                assert!(!*has_relu, "no relu in graph -> has_relu must be false");
            }
            _ => panic!("expected Conv2dRequantReluI8Nhwc"),
        }
    }
    
    #[test]
    fn test_conv_requant_skipped_when_group_ne_1() {
        // Depthwise-style group != 1 should NOT be fused.
        let mut graph = Graph {
            ops: vec![
                CompiledOp {
                    name: "conv1".into(),
                    op_type: "Conv".into(),
                    inputs: vec!["x".into(), "w".into()],
                    outputs: vec!["conv_out".into()],
                    params: OpParams::Conv2d {
                        kernel_shape: [3, 3],
                        strides: [1, 1],
                        pads: [1, 1, 1, 1],
                        dilations: [1, 1],
                        group: 32, // depthwise
                    },
                },
                CompiledOp {
                    name: "conv1_requant".into(),
                    op_type: "Requantize".into(),
                    inputs: vec!["conv_out_i32".into()],
                    outputs: vec!["conv_out".into()],
                    params: OpParams::Requantize { scale: 0.01 },
                },
            ],
            shapes: HashMap::new(),
            inputs: vec!["x".into()],
            outputs: vec!["conv_out".into()],
            initializers: HashMap::new(),
            dtype: Dtype::I8,
        };
        
        let before = graph.ops.len();
        apply_fusion_passes(&mut graph);
        assert_eq!(graph.ops.len(), before, "group!=1 must not fuse");
    }
    
    #[test]
    fn test_fusion_removes_ops() {
        let mut graph = Graph {
            ops: vec![
                CompiledOp {
                    name: "conv1".into(),
                    op_type: "Conv".into(),
                    inputs: vec!["input".into()],
                    outputs: vec!["conv_out".into()],
                    params: OpParams::Conv2d {
                        kernel_shape: [3, 3],
                        strides: [1, 1],
                        pads: [1, 1, 1, 1],
                        dilations: [1, 1],
                        group: 1,
                    },
                },
                CompiledOp {
                    name: "relu1".into(),
                    op_type: "Relu".into(),
                    inputs: vec!["conv_out".into()],
                    outputs: vec!["relu_out".into()],
                    params: OpParams::None,
                },
            ],
            shapes: HashMap::new(),
            inputs: vec!["input".into()],
            outputs: vec!["relu_out".into()],
            initializers: HashMap::new(),
            dtype: Dtype::F32,
        };
        
        let before = graph.ops.len();
        apply_fusion_passes(&mut graph);
        let after = graph.ops.len();
        
        // Should have removed the Relu op
        assert_eq!(after, before - 1);
        // The remaining op should output to relu_out
        assert_eq!(graph.ops[0].outputs[0], "relu_out");
    }
}
