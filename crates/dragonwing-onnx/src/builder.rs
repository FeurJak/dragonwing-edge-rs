//! Op builder trait and registry.
//!
//! Following the QNN EP two-phase pattern:
//! 1. `is_supported` — quick check if we can handle this op
//! 2. `validate` — full validation, compute output shape
//! 3. `build` — allocate resources, create the runtime op

use crate::error::{Error, Result};
use crate::model::{OnnxNode, OnnxTensor, DataType};
use dragonwing_core::Dtype;
use std::collections::HashMap;

/// Reason why an op is not supported.
#[derive(Debug, Clone)]
pub enum UnsupportedReason {
    /// Unknown op type.
    UnknownOpType,
    /// Unsupported attribute value.
    UnsupportedAttribute { name: String, value: String, reason: String },
    /// Unsupported data type.
    UnsupportedDtype { found: DataType, expected: &'static str },
    /// Unsupported input count.
    UnsupportedInputCount { found: usize, expected: &'static str },
    /// Unsupported output count.
    UnsupportedOutputCount { found: usize, expected: &'static str },
    /// Generic reason.
    Other(String),
}

impl std::fmt::Display for UnsupportedReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownOpType => write!(f, "unknown op type"),
            Self::UnsupportedAttribute { name, value, reason } => {
                write!(f, "unsupported attribute {name}={value}: {reason}")
            }
            Self::UnsupportedDtype { found, expected } => {
                write!(f, "unsupported dtype {found:?}, expected {expected}")
            }
            Self::UnsupportedInputCount { found, expected } => {
                write!(f, "unsupported input count {found}, expected {expected}")
            }
            Self::UnsupportedOutputCount { found, expected } => {
                write!(f, "unsupported output count {found}, expected {expected}")
            }
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

/// Tensor shape (static, all dimensions known).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    /// Dimensions.
    pub dims: Vec<usize>,
    /// Data type.
    pub dtype: Dtype,
}

impl TensorShape {
    /// Create a new shape.
    pub fn new(dims: Vec<usize>, dtype: Dtype) -> Self {
        Self { dims, dtype }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }
}

/// Context for building ops.
pub struct BuildContext<'a> {
    /// The dtype the graph is being compiled in.
    pub dtype: Dtype,
    /// Symbol table: tensor name → shape.
    pub shapes: HashMap<String, TensorShape>,
    /// Initializers (constant tensors).
    pub initializers: &'a HashMap<String, OnnxTensor>,
}

impl<'a> BuildContext<'a> {
    /// Create a new build context.
    pub fn new(dtype: Dtype, initializers: &'a HashMap<String, OnnxTensor>) -> Self {
        Self {
            dtype,
            shapes: HashMap::new(),
            initializers,
        }
    }

    /// Get the shape of a tensor by name.
    pub fn get_shape(&self, name: &str) -> Option<&TensorShape> {
        self.shapes.get(name)
    }

    /// Set the shape of a tensor.
    pub fn set_shape(&mut self, name: String, shape: TensorShape) {
        self.shapes.insert(name, shape);
    }

    /// Check if a tensor is an initializer (constant).
    pub fn is_initializer(&self, name: &str) -> bool {
        self.initializers.contains_key(name)
    }

    /// Get an initializer by name.
    pub fn get_initializer(&self, name: &str) -> Option<&OnnxTensor> {
        self.initializers.get(name)
    }
}

/// Trait for op builders.
///
/// Each ONNX op type has a corresponding builder that knows how to:
/// 1. Check if the op is supported with the given attributes
/// 2. Validate and compute output shapes
/// 3. Build the runtime dispatch
pub trait OpBuilder: Send + Sync {
    /// The ONNX op type this builder handles (e.g., "Conv", "Relu").
    fn op_type(&self) -> &'static str;

    /// Quick check if this op configuration is supported.
    ///
    /// Should be fast and not allocate. Returns `Ok(())` if supported,
    /// `Err(reason)` if not.
    fn is_supported(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason>;

    /// Full validation: check all inputs exist, compute output shape.
    ///
    /// Called after `is_supported` succeeds. May inspect input shapes
    /// from the context.
    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape>;

    /// Build the runtime op.
    ///
    /// Called after `validate` succeeds. Returns a `CompiledOp` that
    /// can be executed later.
    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp>;
}

/// A compiled op ready for execution.
#[derive(Debug, Clone)]
pub struct CompiledOp {
    /// Node name (for debugging).
    pub name: String,
    /// Op type.
    pub op_type: String,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Op-specific parameters encoded as bytes (interpreted by the runtime).
    pub params: OpParams,
}

/// Op-specific parameters.
#[derive(Debug, Clone)]
pub enum OpParams {
    /// No parameters (e.g., Relu).
    None,
    /// Reshape: target shape.
    Reshape { shape: Vec<usize> },
    /// Conv2D parameters.
    Conv2d {
        kernel_shape: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4], // [top, left, bottom, right]
        dilations: [usize; 2],
        group: usize,
    },
    /// Gemm parameters.
    Gemm {
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    },
    /// Softmax parameters.
    Softmax { axis: i64 },
    /// Clip parameters (for Relu6).
    Clip { min: f32, max: f32 },
    /// GlobalAveragePool (no params, but distinct from None for clarity).
    GlobalAvgPool,
    /// Add (elementwise, no params).
    Add,
}

/// Validation report from the validate pass.
#[derive(Debug, Default)]
pub struct ValidationReport {
    /// Nodes that passed validation.
    pub supported: Vec<String>,
    /// Nodes that failed validation: (node_name, reason).
    pub unsupported: Vec<(String, UnsupportedReason)>,
}

impl ValidationReport {
    /// Check if all nodes are supported.
    pub fn is_fully_supported(&self) -> bool {
        self.unsupported.is_empty()
    }
}

// =============================================================================
// Op Builder Registry
// =============================================================================

/// Get an op builder by ONNX op type.
///
/// Returns `None` for unknown op types.
pub fn get_builder(op_type: &str) -> Option<&'static dyn OpBuilder> {
    // Hand-rolled match instead of phf for zero deps
    match op_type {
        "Relu" => Some(&RELU_BUILDER),
        "Clip" => Some(&CLIP_BUILDER),
        "Conv" => Some(&CONV_BUILDER),
        "Add" => Some(&ADD_BUILDER),
        "GlobalAveragePool" => Some(&GLOBAL_AVG_POOL_BUILDER),
        "Gemm" => Some(&GEMM_BUILDER),
        "MatMul" => Some(&MATMUL_BUILDER),
        "Softmax" => Some(&SOFTMAX_BUILDER),
        "Reshape" => Some(&RESHAPE_BUILDER),
        "Flatten" => Some(&FLATTEN_BUILDER),
        "Squeeze" => Some(&SQUEEZE_BUILDER),
        "Unsqueeze" => Some(&UNSQUEEZE_BUILDER),
        "Transpose" => Some(&TRANSPOSE_BUILDER),
        "BatchNormalization" => Some(&BATCHNORM_BUILDER),
        "Pad" => Some(&PAD_BUILDER),
        "MaxPool" => Some(&MAXPOOL_BUILDER),
        "AveragePool" => Some(&AVGPOOL_BUILDER),
        "Shape" => Some(&SHAPE_BUILDER),
        "Gather" => Some(&GATHER_BUILDER),
        "Concat" => Some(&CONCAT_BUILDER),
        "Constant" => Some(&CONSTANT_BUILDER),
        _ => None,
    }
}

// =============================================================================
// Individual Op Builders
// =============================================================================

// --- Relu ---
struct ReluBuilder;
static RELU_BUILDER: ReluBuilder = ReluBuilder;

impl OpBuilder for ReluBuilder {
    fn op_type(&self) -> &'static str { "Relu" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.len() != 1 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "1",
            });
        }
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        Ok(input_shape.clone())
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Relu".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::None,
        })
    }
}

// --- Clip (for Relu6) ---
struct ClipBuilder;
static CLIP_BUILDER: ClipBuilder = ClipBuilder;

impl OpBuilder for ClipBuilder {
    fn op_type(&self) -> &'static str { "Clip" }

    fn is_supported(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        // Clip can have 1-3 inputs: X, min (optional), max (optional)
        if node.inputs.is_empty() || node.inputs.len() > 3 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "1-3",
            });
        }
        // Check if min/max are constants (required for static compilation)
        for (i, name) in node.inputs.iter().skip(1).enumerate() {
            if !name.is_empty() && !ctx.is_initializer(name) {
                return Err(UnsupportedReason::Other(
                    format!("Clip {} input must be constant", if i == 0 { "min" } else { "max" })
                ));
            }
        }
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        Ok(input_shape.clone())
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        
        // Get min/max from initializers or attributes
        let min_val = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
            ctx.get_initializer(&node.inputs[1])
                .and_then(|t| t.as_f32_slice())
                .and_then(|s| s.first().copied())
                .unwrap_or(f32::NEG_INFINITY)
        } else {
            node.get_attr_float("min", f32::NEG_INFINITY)
        };

        let max_val = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
            ctx.get_initializer(&node.inputs[2])
                .and_then(|t| t.as_f32_slice())
                .and_then(|s| s.first().copied())
                .unwrap_or(f32::INFINITY)
        } else {
            node.get_attr_float("max", f32::INFINITY)
        };

        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Clip".into(),
            inputs: vec![node.inputs[0].clone()], // Only pass the data input
            outputs: node.outputs.clone(),
            params: OpParams::Clip { min: min_val, max: max_val },
        })
    }
}

// --- Conv ---
struct ConvBuilder;
static CONV_BUILDER: ConvBuilder = ConvBuilder;

impl OpBuilder for ConvBuilder {
    fn op_type(&self) -> &'static str { "Conv" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        // Conv has 2-3 inputs: X, W, B (optional)
        if node.inputs.len() < 2 || node.inputs.len() > 3 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "2-3",
            });
        }
        
        // Check dilations (we only support 1)
        let dilations = node.get_attr_ints("dilations");
        if !dilations.is_empty() && dilations.iter().any(|&d| d != 1) {
            return Err(UnsupportedReason::UnsupportedAttribute {
                name: "dilations".into(),
                value: format!("{dilations:?}"),
                reason: "only dilation=1 supported".into(),
            });
        }

        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        
        // Get kernel shape from weight tensor or attribute
        let weight = ctx.get_initializer(&node.inputs[1])
            .ok_or_else(|| Error::Validation(format!("weight {} must be constant", node.inputs[1])))?;
        
        // Weight shape for Conv2D: [C_out, C_in/group, K_h, K_w] (OIHW)
        // Input shape: [N, C_in, H_in, W_in] (NCHW) for ONNX
        // We'll convert to NHWC internally
        
        if input_shape.dims.len() != 4 {
            return Err(Error::Validation("Conv input must be 4D (NCHW)".into()));
        }
        if weight.dims.len() != 4 {
            return Err(Error::Validation("Conv weight must be 4D (OIHW)".into()));
        }

        let n = input_shape.dims[0];
        let c_in = input_shape.dims[1];
        let h_in = input_shape.dims[2];
        let w_in = input_shape.dims[3];

        let c_out = weight.dims[0] as usize;
        let k_h = weight.dims[2] as usize;
        let k_w = weight.dims[3] as usize;

        let kernel_shape = node.get_attr_ints("kernel_shape");
        let (k_h, k_w) = if kernel_shape.len() == 2 {
            (kernel_shape[0] as usize, kernel_shape[1] as usize)
        } else {
            (k_h, k_w)
        };

        let strides = node.get_attr_ints("strides");
        let (stride_h, stride_w) = if strides.len() == 2 {
            (strides[0] as usize, strides[1] as usize)
        } else {
            (1, 1)
        };

        let pads = node.get_attr_ints("pads");
        let (pad_t, pad_l, pad_b, pad_r) = if pads.len() == 4 {
            (pads[0] as usize, pads[1] as usize, pads[2] as usize, pads[3] as usize)
        } else {
            (0, 0, 0, 0)
        };

        let group = node.get_attr_int("group", 1) as usize;
        
        // Validate group divides channels
        if c_in % group != 0 || c_out % group != 0 {
            return Err(Error::Validation(format!(
                "group {group} must divide both c_in {c_in} and c_out {c_out}"
            )));
        }

        // Output shape
        let h_out = (h_in + pad_t + pad_b - k_h) / stride_h + 1;
        let w_out = (w_in + pad_l + pad_r - k_w) / stride_w + 1;

        Ok(TensorShape::new(vec![n, c_out, h_out, w_out], ctx.dtype))
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        
        let kernel_shape_attr = node.get_attr_ints("kernel_shape");
        let weight = ctx.get_initializer(&node.inputs[1]).unwrap();
        let k_h = if kernel_shape_attr.len() == 2 { kernel_shape_attr[0] as usize } else { weight.dims[2] as usize };
        let k_w = if kernel_shape_attr.len() == 2 { kernel_shape_attr[1] as usize } else { weight.dims[3] as usize };

        let strides = node.get_attr_ints("strides");
        let (stride_h, stride_w) = if strides.len() == 2 {
            (strides[0] as usize, strides[1] as usize)
        } else {
            (1, 1)
        };

        let pads = node.get_attr_ints("pads");
        let (pad_t, pad_l, pad_b, pad_r) = if pads.len() == 4 {
            (pads[0] as usize, pads[1] as usize, pads[2] as usize, pads[3] as usize)
        } else {
            (0, 0, 0, 0)
        };

        let dilations = node.get_attr_ints("dilations");
        let (dil_h, dil_w) = if dilations.len() == 2 {
            (dilations[0] as usize, dilations[1] as usize)
        } else {
            (1, 1)
        };

        let group = node.get_attr_int("group", 1) as usize;

        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Conv".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::Conv2d {
                kernel_shape: [k_h, k_w],
                strides: [stride_h, stride_w],
                pads: [pad_t, pad_l, pad_b, pad_r],
                dilations: [dil_h, dil_w],
                group,
            },
        })
    }
}

// --- Add ---
struct AddBuilder;
static ADD_BUILDER: AddBuilder = AddBuilder;

impl OpBuilder for AddBuilder {
    fn op_type(&self) -> &'static str { "Add" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.len() != 2 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "2",
            });
        }
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let a_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?
            .clone();
        
        // B might be an initializer (bias)
        let b_shape = if let Some(shape) = ctx.get_shape(&node.inputs[1]) {
            shape.clone()
        } else if let Some(t) = ctx.get_initializer(&node.inputs[1]) {
            TensorShape::new(t.dims.iter().map(|&d| d as usize).collect(), ctx.dtype)
        } else {
            return Err(Error::Validation(format!("input {} not found", node.inputs[1])));
        };

        // For now, require same shape or broadcast-compatible
        // Simple broadcast: if shapes differ, the smaller must be 1D and match last dim
        if a_shape.dims == b_shape.dims {
            Ok(a_shape)
        } else if b_shape.dims.len() == 1 && a_shape.dims.last() == b_shape.dims.first() {
            // Bias add case: [N,C,H,W] + [C] broadcasts
            Ok(a_shape)
        } else if a_shape.dims.len() == b_shape.dims.len() {
            // Check numpy-style broadcast
            let mut out_dims = Vec::new();
            for (a, b) in a_shape.dims.iter().zip(&b_shape.dims) {
                if a == b {
                    out_dims.push(*a);
                } else if *a == 1 {
                    out_dims.push(*b);
                } else if *b == 1 {
                    out_dims.push(*a);
                } else {
                    return Err(Error::Validation(format!(
                        "Add shapes incompatible: {:?} vs {:?}", a_shape.dims, b_shape.dims
                    )));
                }
            }
            Ok(TensorShape::new(out_dims, ctx.dtype))
        } else {
            Err(Error::Validation(format!(
                "Add shapes incompatible: {:?} vs {:?}", a_shape.dims, b_shape.dims
            )))
        }
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Add".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::Add,
        })
    }
}

// --- GlobalAveragePool ---
struct GlobalAvgPoolBuilder;
static GLOBAL_AVG_POOL_BUILDER: GlobalAvgPoolBuilder = GlobalAvgPoolBuilder;

impl OpBuilder for GlobalAvgPoolBuilder {
    fn op_type(&self) -> &'static str { "GlobalAveragePool" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.len() != 1 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "1",
            });
        }
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        
        if input_shape.dims.len() != 4 {
            return Err(Error::Validation("GlobalAveragePool input must be 4D".into()));
        }

        // Output: [N, C, 1, 1]
        let n = input_shape.dims[0];
        let c = input_shape.dims[1];
        Ok(TensorShape::new(vec![n, c, 1, 1], ctx.dtype))
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "GlobalAveragePool".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::GlobalAvgPool,
        })
    }
}

// --- Gemm ---
struct GemmBuilder;
static GEMM_BUILDER: GemmBuilder = GemmBuilder;

impl OpBuilder for GemmBuilder {
    fn op_type(&self) -> &'static str { "Gemm" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.len() < 2 || node.inputs.len() > 3 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "2-3",
            });
        }
        
        let alpha = node.get_attr_float("alpha", 1.0);
        let beta = node.get_attr_float("beta", 1.0);
        if (alpha - 1.0).abs() > 1e-6 || (beta - 1.0).abs() > 1e-6 {
            return Err(UnsupportedReason::UnsupportedAttribute {
                name: "alpha/beta".into(),
                value: format!("alpha={alpha}, beta={beta}"),
                reason: "only alpha=1, beta=1 supported".into(),
            });
        }

        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let a_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?
            .clone();
        
        // B is usually a constant weight
        let b_shape = if let Some(shape) = ctx.get_shape(&node.inputs[1]) {
            shape.clone()
        } else if let Some(t) = ctx.get_initializer(&node.inputs[1]) {
            TensorShape::new(t.dims.iter().map(|&d| d as usize).collect(), ctx.dtype)
        } else {
            return Err(Error::Validation(format!("input {} not found", node.inputs[1])));
        };

        let trans_a = node.get_attr_int("transA", 0) != 0;
        let trans_b = node.get_attr_int("transB", 0) != 0;

        if a_shape.dims.len() != 2 || b_shape.dims.len() != 2 {
            return Err(Error::Validation("Gemm inputs must be 2D".into()));
        }

        let m = if trans_a { a_shape.dims[1] } else { a_shape.dims[0] };
        let k_a = if trans_a { a_shape.dims[0] } else { a_shape.dims[1] };
        let k_b = if trans_b { b_shape.dims[1] } else { b_shape.dims[0] };
        let n = if trans_b { b_shape.dims[0] } else { b_shape.dims[1] };

        if k_a != k_b {
            return Err(Error::Validation(format!(
                "Gemm dimension mismatch: A has K={k_a}, B has K={k_b}"
            )));
        }

        Ok(TensorShape::new(vec![m, n], ctx.dtype))
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        
        let alpha = node.get_attr_float("alpha", 1.0);
        let beta = node.get_attr_float("beta", 1.0);
        let trans_a = node.get_attr_int("transA", 0) != 0;
        let trans_b = node.get_attr_int("transB", 0) != 0;

        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Gemm".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::Gemm { alpha, beta, trans_a, trans_b },
        })
    }
}

// --- MatMul ---
struct MatMulBuilder;
static MATMUL_BUILDER: MatMulBuilder = MatMulBuilder;

impl OpBuilder for MatMulBuilder {
    fn op_type(&self) -> &'static str { "MatMul" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.len() != 2 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "2",
            });
        }
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let a_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?
            .clone();
        
        let b_shape = if let Some(shape) = ctx.get_shape(&node.inputs[1]) {
            shape.clone()
        } else if let Some(t) = ctx.get_initializer(&node.inputs[1]) {
            TensorShape::new(t.dims.iter().map(|&d| d as usize).collect(), ctx.dtype)
        } else {
            return Err(Error::Validation(format!("input {} not found", node.inputs[1])));
        };

        if a_shape.dims.len() != 2 || b_shape.dims.len() != 2 {
            return Err(Error::Validation("MatMul inputs must be 2D".into()));
        }

        let m = a_shape.dims[0];
        let k = a_shape.dims[1];
        let n = b_shape.dims[1];

        if k != b_shape.dims[0] {
            return Err(Error::Validation(format!(
                "MatMul dimension mismatch: A has K={k}, B has K={}", b_shape.dims[0]
            )));
        }

        Ok(TensorShape::new(vec![m, n], ctx.dtype))
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "MatMul".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::Gemm { alpha: 1.0, beta: 0.0, trans_a: false, trans_b: false },
        })
    }
}

// --- Softmax ---
struct SoftmaxBuilder;
static SOFTMAX_BUILDER: SoftmaxBuilder = SoftmaxBuilder;

impl OpBuilder for SoftmaxBuilder {
    fn op_type(&self) -> &'static str { "Softmax" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.len() != 1 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "1",
            });
        }
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        Ok(input_shape.clone())
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        let axis = node.get_attr_int("axis", -1);
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Softmax".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::Softmax { axis },
        })
    }
}

// --- Reshape ---
struct ReshapeBuilder;
static RESHAPE_BUILDER: ReshapeBuilder = ReshapeBuilder;

impl OpBuilder for ReshapeBuilder {
    fn op_type(&self) -> &'static str { "Reshape" }

    fn is_supported(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.len() != 2 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "2",
            });
        }
        // Shape must be constant
        if !ctx.is_initializer(&node.inputs[1]) {
            return Err(UnsupportedReason::Other("Reshape shape must be constant".into()));
        }
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        
        let shape_tensor = ctx.get_initializer(&node.inputs[1])
            .ok_or_else(|| Error::Validation("Reshape shape not found".into()))?;
        
        let target_shape: Vec<i64> = shape_tensor.as_i64_slice()
            .ok_or_else(|| Error::Validation("Reshape shape must be int64".into()))?
            .to_vec();

        let input_numel = input_shape.numel();
        let mut output_dims = Vec::new();
        let mut infer_dim = None;

        for (i, &d) in target_shape.iter().enumerate() {
            if d == -1 {
                if infer_dim.is_some() {
                    return Err(Error::Validation("Reshape: only one dimension can be -1".into()));
                }
                infer_dim = Some(i);
                output_dims.push(0); // placeholder
            } else if d == 0 {
                // Copy from input
                if i < input_shape.dims.len() {
                    output_dims.push(input_shape.dims[i]);
                } else {
                    return Err(Error::Validation("Reshape: 0 dimension out of bounds".into()));
                }
            } else {
                output_dims.push(d as usize);
            }
        }

        // Infer the -1 dimension
        if let Some(idx) = infer_dim {
            let known_product: usize = output_dims.iter().filter(|&&d| d != 0).product();
            if known_product == 0 || input_numel % known_product != 0 {
                return Err(Error::Validation("Reshape: cannot infer dimension".into()));
            }
            output_dims[idx] = input_numel / known_product;
        }

        Ok(TensorShape::new(output_dims, ctx.dtype))
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        let shape = output_shape.dims.clone();
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Reshape".into(),
            inputs: vec![node.inputs[0].clone()], // Only data input
            outputs: node.outputs.clone(),
            params: OpParams::Reshape { shape },
        })
    }
}

// --- Flatten ---
struct FlattenBuilder;
static FLATTEN_BUILDER: FlattenBuilder = FlattenBuilder;

impl OpBuilder for FlattenBuilder {
    fn op_type(&self) -> &'static str { "Flatten" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.len() != 1 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "1",
            });
        }
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        
        let axis = node.get_attr_int("axis", 1) as usize;
        
        // Flatten to 2D: dims before axis, dims from axis onward
        let dim0: usize = input_shape.dims[..axis].iter().product();
        let dim1: usize = input_shape.dims[axis..].iter().product();

        Ok(TensorShape::new(vec![dim0, dim1], ctx.dtype))
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        let shape = output_shape.dims.clone();
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Flatten".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::Reshape { shape },
        })
    }
}

// Placeholder builders for less common ops
macro_rules! placeholder_builder {
    ($struct_name:ident, $static_name:ident, $op_type:expr) => {
        struct $struct_name;
        static $static_name: $struct_name = $struct_name;
        
        impl OpBuilder for $struct_name {
            fn op_type(&self) -> &'static str { $op_type }
            
            fn is_supported(&self, _node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
                // TODO: implement
                Ok(())
            }
            
            fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
                // Passthrough for now
                if let Some(shape) = ctx.get_shape(&node.inputs[0]) {
                    Ok(shape.clone())
                } else {
                    Err(Error::Validation(format!("{} validation not implemented", $op_type)))
                }
            }
            
            fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
                let output_shape = self.validate(node, ctx)?;
                ctx.set_shape(node.outputs[0].clone(), output_shape);
                Ok(CompiledOp {
                    name: node.name.clone(),
                    op_type: $op_type.into(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                    params: OpParams::None,
                })
            }
        }
    };
}

placeholder_builder!(SqueezeBuilderImpl, SQUEEZE_BUILDER, "Squeeze");
placeholder_builder!(UnsqueezeBuilderImpl, UNSQUEEZE_BUILDER, "Unsqueeze");
placeholder_builder!(TransposeBuilderImpl, TRANSPOSE_BUILDER, "Transpose");
placeholder_builder!(BatchNormBuilderImpl, BATCHNORM_BUILDER, "BatchNormalization");
placeholder_builder!(PadBuilderImpl, PAD_BUILDER, "Pad");
placeholder_builder!(MaxPoolBuilderImpl, MAXPOOL_BUILDER, "MaxPool");
placeholder_builder!(AvgPoolBuilderImpl, AVGPOOL_BUILDER, "AveragePool");
placeholder_builder!(ShapeBuilderImpl, SHAPE_BUILDER, "Shape");
placeholder_builder!(GatherBuilderImpl, GATHER_BUILDER, "Gather");
placeholder_builder!(ConcatBuilderImpl, CONCAT_BUILDER, "Concat");
placeholder_builder!(ConstantBuilderImpl, CONSTANT_BUILDER, "Constant");
