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
    /// Initializers (constant tensors from model).
    pub initializers: &'a HashMap<String, OnnxTensor>,
    /// Runtime-computed constants (for constant folding).
    /// These are tensors that can be computed at compile time.
    pub constants: HashMap<String, Vec<u8>>,
}

impl<'a> BuildContext<'a> {
    /// Create a new build context.
    pub fn new(dtype: Dtype, initializers: &'a HashMap<String, OnnxTensor>) -> Self {
        Self {
            dtype,
            shapes: HashMap::new(),
            initializers,
            constants: HashMap::new(),
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
    
    /// Check if a tensor is a compile-time constant (initializer or folded).
    pub fn is_constant(&self, name: &str) -> bool {
        self.initializers.contains_key(name) || self.constants.contains_key(name)
    }
    
    /// Get constant data by name (checks both initializers and folded constants).
    pub fn get_constant_data(&self, name: &str) -> Option<&[u8]> {
        if let Some(init) = self.initializers.get(name) {
            return Some(&init.data);
        }
        self.constants.get(name).map(|v| v.as_slice())
    }
    
    /// Set a runtime-computed constant.
    pub fn set_constant(&mut self, name: String, data: Vec<u8>) {
        self.constants.insert(name, data);
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
    /// MaxPool parameters.
    MaxPool {
        /// Kernel size [H, W].
        kernel_shape: [usize; 2],
        /// Stride [H, W].
        strides: [usize; 2],
        /// Padding [top, left, bottom, right].
        pads: [usize; 4],
    },
    /// AveragePool parameters.
    AvgPool {
        /// Kernel size [H, W].
        kernel_shape: [usize; 2],
        /// Stride [H, W].
        strides: [usize; 2],
        /// Padding [top, left, bottom, right].
        pads: [usize; 4],
    },
    /// Sigmoid (no params).
    Sigmoid,
    /// Mul (elementwise, no params).
    Mul,
    /// Sub (elementwise, no params).
    Sub,
    /// Div (elementwise, no params).
    Div,
    /// Concat parameters.
    Concat {
        /// Axis to concatenate along.
        axis: usize,
    },
    /// Resize parameters.
    Resize {
        /// Output height.
        out_h: usize,
        /// Output width.
        out_w: usize,
        /// Interpolation mode: "nearest" or "linear".
        mode: String,
    },
    /// Split parameters.
    Split {
        /// Axis to split along.
        axis: usize,
        /// Sizes of each split.
        split_sizes: Vec<usize>,
    },
    /// Transpose parameters.
    Transpose {
        /// Permutation array.
        perm: Vec<usize>,
    },
    /// Slice parameters.
    Slice {
        /// Start indices.
        starts: Vec<isize>,
        /// End indices.
        ends: Vec<isize>,
        /// Axes to slice.
        axes: Vec<usize>,
        /// Steps.
        steps: Vec<isize>,
    },
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
        // YOLO ops (Task 005)
        "Sigmoid" => Some(&SIGMOID_BUILDER),
        "Mul" => Some(&MUL_BUILDER),
        "Resize" => Some(&RESIZE_BUILDER),
        "Split" => Some(&SPLIT_BUILDER),
        "Slice" => Some(&SLICE_BUILDER),
        "Sub" => Some(&SUB_BUILDER),
        "Div" => Some(&DIV_BUILDER),
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

// --- MaxPool ---
struct MaxPoolBuilder;
static MAXPOOL_BUILDER: MaxPoolBuilder = MaxPoolBuilder;

impl OpBuilder for MaxPoolBuilder {
    fn op_type(&self) -> &'static str { "MaxPool" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.len() != 1 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "1",
            });
        }

        // Check for unsupported attributes
        let ceil_mode = node.get_attr_int("ceil_mode", 0);
        if ceil_mode != 0 {
            return Err(UnsupportedReason::UnsupportedAttribute {
                name: "ceil_mode".into(),
                value: ceil_mode.to_string(),
                reason: "only ceil_mode=0 supported".into(),
            });
        }

        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;

        if input_shape.dims.len() != 4 {
            return Err(Error::Validation("MaxPool input must be 4D".into()));
        }

        // Extract attributes
        let kernel_shape = node.get_attr_ints("kernel_shape");
        if kernel_shape.is_empty() {
            return Err(Error::Validation("MaxPool requires kernel_shape".into()));
        }
        let strides = node.get_attr_ints("strides");
        let pads = node.get_attr_ints("pads");

        if kernel_shape.len() != 2 {
            return Err(Error::Validation("only 2D pooling supported".into()));
        }

        let n = input_shape.dims[0];
        let h_in = input_shape.dims[1];
        let w_in = input_shape.dims[2];
        let c = input_shape.dims[3];

        let k_h = kernel_shape[0] as usize;
        let k_w = kernel_shape[1] as usize;
        let stride_h = strides.first().map(|&v| v as usize).unwrap_or(1);
        let stride_w = strides.get(1).map(|&v| v as usize).unwrap_or(stride_h);
        let pad_h = pads.first().map(|&v| v as usize).unwrap_or(0);
        let pad_w = pads.get(1).map(|&v| v as usize).unwrap_or(0);

        // Output dimensions
        let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

        Ok(TensorShape::new(vec![n, h_out, w_out, c], ctx.dtype))
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        ctx.set_shape(node.outputs[0].clone(), output_shape);

        let kernel_shape = node.get_attr_ints("kernel_shape");
        let strides = node.get_attr_ints("strides");
        let pads = node.get_attr_ints("pads");

        let k_h = kernel_shape[0] as usize;
        let k_w = kernel_shape[1] as usize;
        let stride_h = strides.first().map(|&v| v as usize).unwrap_or(1);
        let stride_w = strides.get(1).map(|&v| v as usize).unwrap_or(stride_h);

        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "MaxPool".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::MaxPool {
                kernel_shape: [k_h, k_w],
                strides: [stride_h, stride_w],
                pads: [
                    pads.first().map(|&v| v as usize).unwrap_or(0),
                    pads.get(1).map(|&v| v as usize).unwrap_or(0),
                    pads.get(2).map(|&v| v as usize).unwrap_or(0),
                    pads.get(3).map(|&v| v as usize).unwrap_or(0),
                ],
            },
        })
    }
}

// --- AveragePool ---
struct AvgPoolBuilder;
static AVGPOOL_BUILDER: AvgPoolBuilder = AvgPoolBuilder;

impl OpBuilder for AvgPoolBuilder {
    fn op_type(&self) -> &'static str { "AveragePool" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.len() != 1 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "1",
            });
        }

        let ceil_mode = node.get_attr_int("ceil_mode", 0);
        if ceil_mode != 0 {
            return Err(UnsupportedReason::UnsupportedAttribute {
                name: "ceil_mode".into(),
                value: ceil_mode.to_string(),
                reason: "only ceil_mode=0 supported".into(),
            });
        }

        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;

        if input_shape.dims.len() != 4 {
            return Err(Error::Validation("AveragePool input must be 4D".into()));
        }

        let kernel_shape = node.get_attr_ints("kernel_shape");
        if kernel_shape.is_empty() {
            return Err(Error::Validation("AveragePool requires kernel_shape".into()));
        }
        let strides = node.get_attr_ints("strides");
        let pads = node.get_attr_ints("pads");

        if kernel_shape.len() != 2 {
            return Err(Error::Validation("only 2D pooling supported".into()));
        }

        let n = input_shape.dims[0];
        let h_in = input_shape.dims[1];
        let w_in = input_shape.dims[2];
        let c = input_shape.dims[3];

        let k_h = kernel_shape[0] as usize;
        let k_w = kernel_shape[1] as usize;
        let stride_h = strides.first().map(|&v| v as usize).unwrap_or(1);
        let stride_w = strides.get(1).map(|&v| v as usize).unwrap_or(stride_h);
        let pad_h = pads.first().map(|&v| v as usize).unwrap_or(0);
        let pad_w = pads.get(1).map(|&v| v as usize).unwrap_or(0);

        let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;

        Ok(TensorShape::new(vec![n, h_out, w_out, c], ctx.dtype))
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        ctx.set_shape(node.outputs[0].clone(), output_shape);

        let kernel_shape = node.get_attr_ints("kernel_shape");
        let strides = node.get_attr_ints("strides");
        let pads = node.get_attr_ints("pads");

        let k_h = kernel_shape[0] as usize;
        let k_w = kernel_shape[1] as usize;
        let stride_h = strides.first().map(|&v| v as usize).unwrap_or(1);
        let stride_w = strides.get(1).map(|&v| v as usize).unwrap_or(stride_h);

        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "AveragePool".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::AvgPool {
                kernel_shape: [k_h, k_w],
                strides: [stride_h, stride_w],
                pads: [
                    pads.first().map(|&v| v as usize).unwrap_or(0),
                    pads.get(1).map(|&v| v as usize).unwrap_or(0),
                    pads.get(2).map(|&v| v as usize).unwrap_or(0),
                    pads.get(3).map(|&v| v as usize).unwrap_or(0),
                ],
            },
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
        // Shape must be constant (initializer or constant-folded)
        if !ctx.is_constant(&node.inputs[1]) {
            return Err(UnsupportedReason::Other("Reshape shape must be constant".into()));
        }
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        
        // Get shape from constant data (initializer or folded constant)
        let shape_data = ctx.get_constant_data(&node.inputs[1])
            .ok_or_else(|| Error::Validation("Reshape shape not found".into()))?;
        
        // Parse as int64 array
        let target_shape: Vec<i64> = shape_data.chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

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
placeholder_builder!(BatchNormBuilderImpl, BATCHNORM_BUILDER, "BatchNormalization");
placeholder_builder!(PadBuilderImpl, PAD_BUILDER, "Pad");
// MaxPool and AveragePool are implemented above

// --- Transpose (full implementation) ---
struct TransposeBuilderImpl;
static TRANSPOSE_BUILDER: TransposeBuilderImpl = TransposeBuilderImpl;

impl OpBuilder for TransposeBuilderImpl {
    fn op_type(&self) -> &'static str { "Transpose" }
    
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
            .ok_or_else(|| Error::Validation(format!("Transpose: input {} not found", node.inputs[0])))?;
        
        // Get perm attribute or default to reverse
        let perm = node.get_attr_ints("perm");
        let perm: Vec<usize> = if !perm.is_empty() {
            perm.iter().map(|&p| p as usize).collect()
        } else {
            (0..input_shape.dims.len()).rev().collect()
        };
        
        // Apply permutation to output shape
        let out_dims: Vec<usize> = perm.iter().map(|&p| input_shape.dims[p]).collect();
        
        Ok(TensorShape::new(out_dims, input_shape.dtype))
    }
    
    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("Transpose: input {} not found", node.inputs[0])))?;
        let output_shape = self.validate(node, ctx)?;
        
        let perm = node.get_attr_ints("perm");
        let perm: Vec<usize> = if !perm.is_empty() {
            perm.iter().map(|&p| p as usize).collect()
        } else {
            (0..input_shape.dims.len()).rev().collect()
        };
        
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Transpose".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::Transpose { perm },
        })
    }
}

// =============================================================================
// Shape manipulation ops for constant folding
// =============================================================================

// --- Shape ---
// Produces the shape of input tensor as an int64 tensor
struct ShapeBuilderImpl;
static SHAPE_BUILDER: ShapeBuilderImpl = ShapeBuilderImpl;

impl OpBuilder for ShapeBuilderImpl {
    fn op_type(&self) -> &'static str { "Shape" }
    
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
            .ok_or_else(|| Error::Validation(format!("Shape: input {} not found", node.inputs[0])))?;
        
        // Output shape is [rank] - dtype is int64 but we use F32 as placeholder
        // since this gets constant-folded anyway
        Ok(TensorShape::new(vec![input_shape.dims.len()], Dtype::F32))
    }
    
    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("Shape: input {} not found", node.inputs[0])))?;
        
        // Store the shape values as a constant (int64 bytes)
        let shape_data: Vec<u8> = input_shape.dims.iter()
            .flat_map(|&d| (d as i64).to_le_bytes())
            .collect();
        let output_shape = TensorShape::new(vec![input_shape.dims.len()], Dtype::F32);
        
        ctx.set_constant(node.outputs[0].clone(), shape_data);
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        
        // No-op at runtime (constant folded)
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Shape".into(),
            inputs: vec![],  // No runtime inputs needed
            outputs: node.outputs.clone(),
            params: OpParams::None,
        })
    }
}

// --- Constant ---
// Creates a constant tensor from attributes
struct ConstantBuilderImpl;
static CONSTANT_BUILDER: ConstantBuilderImpl = ConstantBuilderImpl;

impl OpBuilder for ConstantBuilderImpl {
    fn op_type(&self) -> &'static str { "Constant" }
    
    fn is_supported(&self, _node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        Ok(())
    }
    
    fn validate(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> Result<TensorShape> {
        // Get value from attribute
        if let Some(attr) = node.get_attr("value") {
            if let Some(tensor) = attr.value.as_tensor() {
                let dims: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
                // All constant-folded tensors use F32 dtype since we only care about bytes
                return Ok(TensorShape::new(dims, Dtype::F32));
            }
        }
        // Scalar constant from value_int or value_float
        if node.get_attr("value_int").is_some() {
            return Ok(TensorShape::new(vec![], Dtype::F32));
        }
        if node.get_attr("value_float").is_some() {
            return Ok(TensorShape::new(vec![], Dtype::F32));
        }
        Err(Error::Validation("Constant: no value attribute found".into()))
    }
    
    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        
        // Get constant data
        let data = if let Some(attr) = node.get_attr("value") {
            if let Some(tensor) = attr.value.as_tensor() {
                tensor.data.clone()
            } else {
                return Err(Error::Validation("Constant: value is not a tensor".into()));
            }
        } else if let Some(attr) = node.get_attr("value_int") {
            if let Some(v) = attr.value.as_int() {
                v.to_le_bytes().to_vec()
            } else {
                return Err(Error::Validation("Constant: value_int is not an int".into()));
            }
        } else if let Some(attr) = node.get_attr("value_float") {
            if let Some(v) = attr.value.as_float() {
                v.to_le_bytes().to_vec()
            } else {
                return Err(Error::Validation("Constant: value_float is not a float".into()));
            }
        } else {
            return Err(Error::Validation("Constant: no value found".into()));
        };
        
        ctx.set_constant(node.outputs[0].clone(), data);
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Constant".into(),
            inputs: vec![],
            outputs: node.outputs.clone(),
            params: OpParams::None,
        })
    }
}

// --- Gather ---
// Selects elements from tensor using indices
struct GatherBuilderImpl;
static GATHER_BUILDER: GatherBuilderImpl = GatherBuilderImpl;

impl OpBuilder for GatherBuilderImpl {
    fn op_type(&self) -> &'static str { "Gather" }
    
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
        let data_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("Gather: data {} not found", node.inputs[0])))?;
        let indices_shape = ctx.get_shape(&node.inputs[1])
            .ok_or_else(|| Error::Validation(format!("Gather: indices {} not found", node.inputs[1])))?;
        
        let axis = node.get_attr_int("axis", 0) as usize;
        
        // Output shape: data.shape[:axis] + indices.shape + data.shape[axis+1:]
        let mut out_dims = data_shape.dims[..axis].to_vec();
        out_dims.extend(&indices_shape.dims);
        if axis + 1 < data_shape.dims.len() {
            out_dims.extend(&data_shape.dims[axis + 1..]);
        }
        
        Ok(TensorShape::new(out_dims, data_shape.dtype))
    }
    
    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        
        // If both inputs are constants, we can fold
        if ctx.is_constant(&node.inputs[0]) && ctx.is_constant(&node.inputs[1]) {
            let data_bytes = ctx.get_constant_data(&node.inputs[0])
                .ok_or_else(|| Error::Validation("Gather: data not found".into()))?;
            let indices_bytes = ctx.get_constant_data(&node.inputs[1])
                .ok_or_else(|| Error::Validation("Gather: indices not found".into()))?;
            
            let data_shape = ctx.get_shape(&node.inputs[0]).unwrap();
            let _axis = node.get_attr_int("axis", 0) as usize;
            
            // Simple case: scalar index on 1D data (common for shape manipulation)
            // Note: shape tensors use int64 (8 bytes) even though we track them with F32 dtype
            if data_shape.dims.len() == 1 {
                // Infer element size from total data length and shape
                let num_elements = data_shape.dims[0];
                let elem_size = if num_elements > 0 { data_bytes.len() / num_elements } else { 8 };
                
                // Get index value
                let index = if indices_bytes.len() == 8 {
                    i64::from_le_bytes(indices_bytes[..8].try_into().unwrap()) as usize
                } else if indices_bytes.len() == 4 {
                    i32::from_le_bytes(indices_bytes[..4].try_into().unwrap()) as usize
                } else {
                    // Can't determine index, skip constant folding
                    ctx.set_shape(node.outputs[0].clone(), output_shape);
                    return Ok(CompiledOp {
                        name: node.name.clone(),
                        op_type: "Gather".into(),
                        inputs: node.inputs.clone(),
                        outputs: node.outputs.clone(),
                        params: OpParams::None,
                    });
                };
                
                let start = index * elem_size;
                let end = start + elem_size;
                if end <= data_bytes.len() {
                    let result = data_bytes[start..end].to_vec();
                    ctx.set_constant(node.outputs[0].clone(), result);
                }
            }
            // For other cases, fall back to runtime (not implemented for constant folding)
        }
        
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Gather".into(),
            inputs: if ctx.is_constant(&node.outputs[0]) { vec![] } else { node.inputs.clone() },
            outputs: node.outputs.clone(),
            params: OpParams::None,
        })
    }
}

// --- Unsqueeze ---
// Adds dimensions to tensor
struct UnsqueezeBuilderImpl;
static UNSQUEEZE_BUILDER: UnsqueezeBuilderImpl = UnsqueezeBuilderImpl;

impl OpBuilder for UnsqueezeBuilderImpl {
    fn op_type(&self) -> &'static str { "Unsqueeze" }
    
    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.is_empty() {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: 0,
                expected: "1 or 2",
            });
        }
        Ok(())
    }
    
    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("Unsqueeze: input {} not found", node.inputs[0])))?;
        
        // Get axes (either from attribute or second input)
        let axes: Vec<i64> = if node.inputs.len() > 1 {
            // Opset 13+: axes from input
            if let Some(data) = ctx.get_constant_data(&node.inputs[1]) {
                // Parse as i64 array
                data.chunks_exact(8)
                    .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect()
            } else {
                return Err(Error::Validation("Unsqueeze: axes must be constant".into()));
            }
        } else {
            // Opset <13: axes from attribute
            node.get_attr_ints("axes")
        };
        
        let rank = input_shape.dims.len() as i64 + axes.len() as i64;
        let mut out_dims = input_shape.dims.clone();
        
        // Sort axes and insert 1s
        let mut sorted_axes: Vec<i64> = axes.iter().map(|&a| {
            if a < 0 { a + rank } else { a }
        }).collect();
        sorted_axes.sort();
        
        for &axis in &sorted_axes {
            let idx = axis as usize;
            if idx <= out_dims.len() {
                out_dims.insert(idx, 1);
            }
        }
        
        Ok(TensorShape::new(out_dims, input_shape.dtype))
    }
    
    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        
        // If input is constant, propagate
        if ctx.is_constant(&node.inputs[0]) {
            if let Some(data) = ctx.get_constant_data(&node.inputs[0]) {
                ctx.set_constant(node.outputs[0].clone(), data.to_vec());
            }
        }
        
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Unsqueeze".into(),
            inputs: if ctx.is_constant(&node.outputs[0]) { vec![] } else { vec![node.inputs[0].clone()] },
            outputs: node.outputs.clone(),
            params: OpParams::None,
        })
    }
}

// --- Concat ---
// Concatenates tensors along an axis
struct ConcatBuilderImpl;
static CONCAT_BUILDER: ConcatBuilderImpl = ConcatBuilderImpl;

impl OpBuilder for ConcatBuilderImpl {
    fn op_type(&self) -> &'static str { "Concat" }
    
    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.is_empty() {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: 0,
                expected: ">=1",
            });
        }
        Ok(())
    }
    
    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let first_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("Concat: input {} not found", node.inputs[0])))?;
        
        let axis = {
            let a = node.get_attr_int("axis", 0);
            if a < 0 { (first_shape.dims.len() as i64 + a) as usize } else { a as usize }
        };
        
        let mut out_dims = first_shape.dims.clone();
        
        // Sum the axis dimension across all inputs
        for input in &node.inputs[1..] {
            let shape = ctx.get_shape(input)
                .ok_or_else(|| Error::Validation(format!("Concat: input {} not found", input)))?;
            out_dims[axis] += shape.dims[axis];
        }
        
        Ok(TensorShape::new(out_dims, first_shape.dtype))
    }
    
    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        let first_shape = ctx.get_shape(&node.inputs[0]).unwrap();
        
        let axis = {
            let a = node.get_attr_int("axis", 0);
            if a < 0 { (first_shape.dims.len() as i64 + a) as usize } else { a as usize }
        };
        
        // If all inputs are constants, fold
        let all_constant = node.inputs.iter().all(|name| ctx.is_constant(name));
        if all_constant {
            // Simple case: 1D tensors (common for shape manipulation)
            if first_shape.dims.len() == 1 && axis == 0 {
                let mut result = Vec::new();
                for input in &node.inputs {
                    if let Some(data) = ctx.get_constant_data(input) {
                        result.extend_from_slice(data);
                    }
                }
                ctx.set_constant(node.outputs[0].clone(), result);
            }
        }
        
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Concat".into(),
            inputs: if ctx.is_constant(&node.outputs[0]) { vec![] } else { node.inputs.clone() },
            outputs: node.outputs.clone(),
            params: OpParams::Concat { axis },
        })
    }
}

// =============================================================================
// YOLO Ops (Task 005)
// =============================================================================

// --- Sigmoid ---
struct SigmoidBuilder;
static SIGMOID_BUILDER: SigmoidBuilder = SigmoidBuilder;

impl OpBuilder for SigmoidBuilder {
    fn op_type(&self) -> &'static str { "Sigmoid" }

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
            op_type: "Sigmoid".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::Sigmoid,
        })
    }
}

// --- Mul ---
struct MulBuilder;
static MUL_BUILDER: MulBuilder = MulBuilder;

impl OpBuilder for MulBuilder {
    fn op_type(&self) -> &'static str { "Mul" }

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
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        // For now, assume both inputs have the same shape (broadcasting handled at runtime)
        Ok(a_shape.clone())
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Mul".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::Mul,
        })
    }
}

// --- Resize ---
struct ResizeBuilder;
static RESIZE_BUILDER: ResizeBuilder = ResizeBuilder;

impl OpBuilder for ResizeBuilder {
    fn op_type(&self) -> &'static str { "Resize" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        // Resize has multiple inputs: X, roi (optional), scales (optional), sizes (optional)
        if node.inputs.is_empty() || node.inputs.len() > 4 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "1-4",
            });
        }
        
        // Check mode - we support nearest and linear
        let mode = node.get_attr_string("mode", "nearest");
        if mode != "nearest" && mode != "linear" {
            return Err(UnsupportedReason::UnsupportedAttribute {
                name: "mode".into(),
                value: mode,
                reason: "only 'nearest' and 'linear' modes supported".into(),
            });
        }
        
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        
        if input_shape.dims.len() != 4 {
            return Err(Error::Validation("Resize input must be 4D (NCHW)".into()));
        }
        
        // Get output size from 'sizes' input (index 3) or 'scales' input (index 2)
        let (out_h, out_w) = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
            // sizes input
            if let Some(sizes_data) = ctx.get_constant_data(&node.inputs[3]) {
                let sizes: &[i64] = unsafe {
                    std::slice::from_raw_parts(sizes_data.as_ptr() as *const i64, sizes_data.len() / 8)
                };
                if sizes.len() >= 4 {
                    (sizes[2] as usize, sizes[3] as usize)
                } else {
                    return Err(Error::Validation("Resize sizes must have 4 elements".into()));
                }
            } else {
                return Err(Error::Validation("Resize sizes must be constant".into()));
            }
        } else if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
            // scales input
            if let Some(scales_data) = ctx.get_constant_data(&node.inputs[2]) {
                let scales: &[f32] = unsafe {
                    std::slice::from_raw_parts(scales_data.as_ptr() as *const f32, scales_data.len() / 4)
                };
                if scales.len() >= 4 {
                    let h = (input_shape.dims[2] as f32 * scales[2]).round() as usize;
                    let w = (input_shape.dims[3] as f32 * scales[3]).round() as usize;
                    (h, w)
                } else {
                    return Err(Error::Validation("Resize scales must have 4 elements".into()));
                }
            } else {
                return Err(Error::Validation("Resize scales must be constant".into()));
            }
        } else {
            return Err(Error::Validation("Resize requires either scales or sizes".into()));
        };
        
        let out_shape = vec![input_shape.dims[0], input_shape.dims[1], out_h, out_w];
        Ok(TensorShape::new(out_shape, input_shape.dtype))
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        let out_h = output_shape.dims[2];
        let out_w = output_shape.dims[3];
        let mode = node.get_attr_string("mode", "nearest");
        
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Resize".into(),
            inputs: vec![node.inputs[0].clone()], // Only pass data input
            outputs: node.outputs.clone(),
            params: OpParams::Resize { out_h, out_w, mode },
        })
    }
}

// --- Split ---
struct SplitBuilder;
static SPLIT_BUILDER: SplitBuilder = SplitBuilder;

impl OpBuilder for SplitBuilder {
    fn op_type(&self) -> &'static str { "Split" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        if node.inputs.is_empty() || node.inputs.len() > 2 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "1-2",
            });
        }
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        
        let axis = {
            let a = node.get_attr_int("axis", 0);
            if a < 0 { (input_shape.dims.len() as i64 + a) as usize } else { a as usize }
        };
        
        // Get split sizes from attribute or input
        let split_sizes: Vec<usize> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
            if let Some(split_data) = ctx.get_constant_data(&node.inputs[1]) {
                let splits: &[i64] = unsafe {
                    std::slice::from_raw_parts(split_data.as_ptr() as *const i64, split_data.len() / 8)
                };
                splits.iter().map(|&s| s as usize).collect()
            } else {
                return Err(Error::Validation("Split sizes must be constant".into()));
            }
        } else {
            // Use num_outputs attribute or split evenly
            let attr_splits = node.get_attr_ints("split");
            if !attr_splits.is_empty() {
                attr_splits.iter().map(|&s| s as usize).collect()
            } else {
                let num_outputs = node.outputs.len();
                let size = input_shape.dims[axis] / num_outputs;
                vec![size; num_outputs]
            }
        };
        
        // Return first output shape
        let mut out_shape = input_shape.dims.clone();
        out_shape[axis] = split_sizes[0];
        
        Ok(TensorShape::new(out_shape, input_shape.dtype))
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?
            .clone();
        
        let axis = {
            let a = node.get_attr_int("axis", 0);
            if a < 0 { (input_shape.dims.len() as i64 + a) as usize } else { a as usize }
        };
        
        let split_sizes: Vec<usize> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
            if let Some(split_data) = ctx.get_constant_data(&node.inputs[1]) {
                let splits: &[i64] = unsafe {
                    std::slice::from_raw_parts(split_data.as_ptr() as *const i64, split_data.len() / 8)
                };
                splits.iter().map(|&s| s as usize).collect()
            } else {
                return Err(Error::Validation("Split sizes must be constant".into()));
            }
        } else {
            let attr_splits = node.get_attr_ints("split");
            if !attr_splits.is_empty() {
                attr_splits.iter().map(|&s| s as usize).collect()
            } else {
                let num_outputs = node.outputs.len();
                let size = input_shape.dims[axis] / num_outputs;
                vec![size; num_outputs]
            }
        };
        
        // Set shapes for all outputs
        for (i, (output_name, &split_size)) in node.outputs.iter().zip(split_sizes.iter()).enumerate() {
            let mut out_shape = input_shape.dims.clone();
            out_shape[axis] = split_size;
            ctx.set_shape(output_name.clone(), TensorShape::new(out_shape, input_shape.dtype));
            
            // Only set first output in validate
            if i == 0 {
                continue;
            }
        }
        
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Split".into(),
            inputs: vec![node.inputs[0].clone()],
            outputs: node.outputs.clone(),
            params: OpParams::Split { axis, split_sizes },
        })
    }
}

// --- Slice ---
struct SliceBuilder;
static SLICE_BUILDER: SliceBuilder = SliceBuilder;

impl OpBuilder for SliceBuilder {
    fn op_type(&self) -> &'static str { "Slice" }

    fn is_supported(&self, node: &OnnxNode, _ctx: &BuildContext<'_>) -> std::result::Result<(), UnsupportedReason> {
        // Slice inputs: data, starts, ends, axes (optional), steps (optional)
        if node.inputs.len() < 3 || node.inputs.len() > 5 {
            return Err(UnsupportedReason::UnsupportedInputCount {
                found: node.inputs.len(),
                expected: "3-5",
            });
        }
        Ok(())
    }

    fn validate(&self, node: &OnnxNode, ctx: &BuildContext<'_>) -> Result<TensorShape> {
        let input_shape = ctx.get_shape(&node.inputs[0])
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        
        // Get starts, ends, axes, steps from constant inputs
        let starts: Vec<i64> = get_i64_constant(ctx, &node.inputs[1])?;
        let ends: Vec<i64> = get_i64_constant(ctx, &node.inputs[2])?;
        
        let axes: Vec<usize> = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
            get_i64_constant(ctx, &node.inputs[3])?.iter().map(|&a| {
                if a < 0 { (input_shape.dims.len() as i64 + a) as usize } else { a as usize }
            }).collect()
        } else {
            (0..starts.len()).collect()
        };
        
        let steps: Vec<i64> = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
            get_i64_constant(ctx, &node.inputs[4])?
        } else {
            vec![1; starts.len()]
        };
        
        // Calculate output shape
        let mut out_dims = input_shape.dims.clone();
        for (i, &axis) in axes.iter().enumerate() {
            let dim = input_shape.dims[axis] as i64;
            let s = starts[i];
            let e = ends[i];
            let start = if s < 0 { (dim + s).max(0) } else { s.min(dim) } as usize;
            let end = if e < 0 { (dim + e).max(0) } else { e.min(dim) } as usize;
            let step = steps[i].unsigned_abs() as usize;
            out_dims[axis] = (end.saturating_sub(start) + step - 1) / step;
        }
        
        Ok(TensorShape::new(out_dims, input_shape.dtype))
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let input_shape = ctx.get_shape(&node.inputs[0]).unwrap().clone();
        let output_shape = self.validate(node, ctx)?;
        
        let starts: Vec<isize> = get_i64_constant(ctx, &node.inputs[1])?.iter().map(|&s| s as isize).collect();
        let ends: Vec<isize> = get_i64_constant(ctx, &node.inputs[2])?.iter().map(|&e| e as isize).collect();
        
        let axes: Vec<usize> = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
            get_i64_constant(ctx, &node.inputs[3])?.iter().map(|&a| {
                if a < 0 { (input_shape.dims.len() as i64 + a) as usize } else { a as usize }
            }).collect()
        } else {
            (0..starts.len()).collect()
        };
        
        let steps: Vec<isize> = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
            get_i64_constant(ctx, &node.inputs[4])?.iter().map(|&s| s as isize).collect()
        } else {
            vec![1; starts.len()]
        };
        
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Slice".into(),
            inputs: vec![node.inputs[0].clone()],
            outputs: node.outputs.clone(),
            params: OpParams::Slice { starts, ends, axes, steps },
        })
    }
}

/// Helper to extract i64 constant data
fn get_i64_constant(ctx: &BuildContext<'_>, name: &str) -> Result<Vec<i64>> {
    let data = ctx.get_constant_data(name)
        .ok_or_else(|| Error::Validation(format!("Slice: {} must be constant", name)))?;
    Ok(unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const i64, data.len() / 8).to_vec()
    })
}

// --- Sub ---
struct SubBuilder;
static SUB_BUILDER: SubBuilder = SubBuilder;

impl OpBuilder for SubBuilder {
    fn op_type(&self) -> &'static str { "Sub" }

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
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        Ok(a_shape.clone())
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Sub".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::Sub,
        })
    }
}

// --- Div ---
struct DivBuilder;
static DIV_BUILDER: DivBuilder = DivBuilder;

impl OpBuilder for DivBuilder {
    fn op_type(&self) -> &'static str { "Div" }

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
            .ok_or_else(|| Error::Validation(format!("input {} not found", node.inputs[0])))?;
        Ok(a_shape.clone())
    }

    fn build(&self, node: &OnnxNode, ctx: &mut BuildContext<'_>) -> Result<CompiledOp> {
        let output_shape = self.validate(node, ctx)?;
        ctx.set_shape(node.outputs[0].clone(), output_shape);
        Ok(CompiledOp {
            name: node.name.clone(),
            op_type: "Div".into(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            params: OpParams::Div,
        })
    }
}
