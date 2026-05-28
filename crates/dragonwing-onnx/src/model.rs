//! ONNX model representation and parser.
//!
//! This module parses the ONNX protobuf format into our internal `Model` struct.
//! The ONNX protobuf schema (simplified for what we use):
//!
//! ```text
//! ModelProto {
//!   1: ir_version (int64)
//!   7: graph (GraphProto)
//!   8: opset_import (repeated OperatorSetIdProto)
//! }
//!
//! GraphProto {
//!   1: node (repeated NodeProto)
//!   5: initializer (repeated TensorProto)
//!   11: input (repeated ValueInfoProto)
//!   12: output (repeated ValueInfoProto)
//! }
//!
//! NodeProto {
//!   1: input (repeated string)
//!   2: output (repeated string)
//!   3: name (string)
//!   4: op_type (string)
//!   5: attribute (repeated AttributeProto)
//! }
//!
//! TensorProto {
//!   1: dims (repeated int64)
//!   2: data_type (int32)
//!   3: segment (deprecated)
//!   4: float_data (repeated float, packed)
//!   5: int32_data (repeated int32)
//!   6: string_data (repeated bytes)
//!   7: int64_data (repeated int64)
//!   8: name (string)
//!   9: raw_data (bytes)
//!   10: double_data (repeated double)
//!   11: uint64_data (repeated uint64)
//! }
//!
//! AttributeProto {
//!   1: name (string)
//!   2: f (float)
//!   3: i (int64)
//!   4: s (bytes)
//!   6: t (TensorProto)
//!   7: floats (repeated float)
//!   8: ints (repeated int64)
//!   20: type (AttributeType enum)
//! }
//! ```

use crate::error::{Error, Result};
use crate::proto::{ProtoReader, parse_packed_int64, parse_packed_floats};

/// ONNX data types (from onnx.proto TensorProto.DataType).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DataType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    Bfloat16 = 16,
}

impl TryFrom<i32> for DataType {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self> {
        match value {
            0 => Ok(Self::Undefined),
            1 => Ok(Self::Float),
            2 => Ok(Self::Uint8),
            3 => Ok(Self::Int8),
            4 => Ok(Self::Uint16),
            5 => Ok(Self::Int16),
            6 => Ok(Self::Int32),
            7 => Ok(Self::Int64),
            8 => Ok(Self::String),
            9 => Ok(Self::Bool),
            10 => Ok(Self::Float16),
            11 => Ok(Self::Double),
            12 => Ok(Self::Uint32),
            13 => Ok(Self::Uint64),
            14 => Ok(Self::Complex64),
            15 => Ok(Self::Complex128),
            16 => Ok(Self::Bfloat16),
            _ => Err(Error::Parse(format!("unknown data type: {value}"))),
        }
    }
}

/// A parsed ONNX model.
#[derive(Debug)]
pub struct Model {
    /// IR version.
    pub ir_version: i64,
    /// Opset version (we require opset 12).
    pub opset_version: i64,
    /// Producer name (for debugging).
    pub producer_name: String,
    /// Model nodes (ops) in topological order.
    pub nodes: Vec<OnnxNode>,
    /// Initializers (constant tensors, typically weights).
    pub initializers: Vec<OnnxTensor>,
    /// Graph inputs (name + shape).
    pub inputs: Vec<ValueInfo>,
    /// Graph outputs (name + shape).
    pub outputs: Vec<ValueInfo>,
}

/// A node (op) in the ONNX graph.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// Node name (for debugging/error messages).
    pub name: String,
    /// ONNX op type (e.g., "Conv", "Relu", "Add").
    pub op_type: String,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Node attributes.
    pub attributes: Vec<OnnxAttribute>,
}

impl OnnxNode {
    /// Get an attribute by name.
    pub fn get_attr(&self, name: &str) -> Option<&OnnxAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Get an int attribute, or default if not present.
    pub fn get_attr_int(&self, name: &str, default: i64) -> i64 {
        self.get_attr(name)
            .and_then(|a| a.value.as_int())
            .unwrap_or(default)
    }

    /// Get an int array attribute, or empty if not present.
    pub fn get_attr_ints(&self, name: &str) -> Vec<i64> {
        self.get_attr(name)
            .and_then(|a| a.value.as_ints())
            .unwrap_or_default()
    }

    /// Get a float attribute, or default if not present.
    pub fn get_attr_float(&self, name: &str, default: f32) -> f32 {
        self.get_attr(name)
            .and_then(|a| a.value.as_float())
            .unwrap_or(default)
    }

    /// Get a string attribute, or default if not present.
    pub fn get_attr_string(&self, name: &str, default: &str) -> String {
        self.get_attr(name)
            .and_then(|a| a.value.as_string())
            .unwrap_or_else(|| default.to_string())
    }
}

/// An attribute on an ONNX node.
#[derive(Debug, Clone)]
pub struct OnnxAttribute {
    /// Attribute name.
    pub name: String,
    /// Attribute value.
    pub value: AttributeValue,
}

/// Possible attribute values.
#[derive(Debug, Clone)]
pub enum AttributeValue {
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f32),
    /// String value.
    String(String),
    /// Tensor value.
    Tensor(OnnxTensor),
    /// Repeated integers.
    Ints(Vec<i64>),
    /// Repeated floats.
    Floats(Vec<f32>),
    /// Repeated strings.
    Strings(Vec<String>),
}

impl AttributeValue {
    /// Get as int.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as float.
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as string.
    pub fn as_string(&self) -> Option<String> {
        match self {
            Self::String(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Get as int array.
    pub fn as_ints(&self) -> Option<Vec<i64>> {
        match self {
            Self::Ints(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Get as float array.
    pub fn as_floats(&self) -> Option<Vec<f32>> {
        match self {
            Self::Floats(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Get as tensor.
    pub fn as_tensor(&self) -> Option<&OnnxTensor> {
        match self {
            Self::Tensor(t) => Some(t),
            _ => None,
        }
    }
}

/// A tensor (used for initializers and attribute tensors).
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    /// Tensor name.
    pub name: String,
    /// Dimensions.
    pub dims: Vec<i64>,
    /// Data type.
    pub data_type: DataType,
    /// Raw data bytes.
    pub data: Vec<u8>,
}

impl OnnxTensor {
    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Get data as f32 slice (only valid for Float tensors with raw_data).
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.data_type != DataType::Float {
            return None;
        }
        if self.data.len() % 4 != 0 {
            return None;
        }
        // SAFETY: f32 has the same alignment as u8 when sliced from a Vec<u8>
        // that was originally f32 data. We check length is multiple of 4.
        Some(unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().cast::<f32>(),
                self.data.len() / 4,
            )
        })
    }

    /// Get data as i64 slice (only valid for Int64 tensors with raw_data).
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        if self.data_type != DataType::Int64 {
            return None;
        }
        if self.data.len() % 8 != 0 {
            return None;
        }
        Some(unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().cast::<i64>(),
                self.data.len() / 8,
            )
        })
    }
}

/// Value info (input/output metadata).
#[derive(Debug, Clone)]
pub struct ValueInfo {
    /// Tensor name.
    pub name: String,
    /// Shape dimensions (-1 for dynamic).
    pub shape: Vec<i64>,
    /// Data type.
    pub data_type: DataType,
}

// =============================================================================
// Parsing
// =============================================================================

/// Parse an ONNX model from bytes.
pub fn parse_model(bytes: &[u8]) -> Result<Model> {
    let mut reader = ProtoReader::new(bytes);
    
    let mut ir_version: i64 = 0;
    let mut opset_version: i64 = 0;
    let mut producer_name = String::new();
    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    while let Some(field) = reader.read_field()? {
        match field.field_number {
            1 => ir_version = field.data.as_int64()?,
            2 => producer_name = field.data.as_string()?.to_string(),
            7 => {
                // GraphProto
                let graph_reader = field.data.as_message()?;
                let (n, i, inp, outp) = parse_graph(graph_reader)?;
                nodes = n;
                initializers = i;
                inputs = inp;
                outputs = outp;
            }
            8 => {
                // OperatorSetIdProto
                let opset_reader = field.data.as_message()?;
                let version = parse_opset(opset_reader)?;
                if version > opset_version {
                    opset_version = version;
                }
            }
            _ => {} // Ignore unknown fields
        }
    }

    // Validate opset version
    if opset_version != 12 && opset_version != 13 && opset_version != 14 && opset_version != 15 {
        // Accept opset 12-15 for compatibility
        if opset_version < 12 || opset_version > 20 {
            return Err(Error::UnsupportedOpset {
                found: opset_version,
                expected: 12,
            });
        }
    }

    Ok(Model {
        ir_version,
        opset_version,
        producer_name,
        nodes,
        initializers,
        inputs,
        outputs,
    })
}

fn parse_opset(mut reader: ProtoReader<'_>) -> Result<i64> {
    let mut version: i64 = 0;
    while let Some(field) = reader.read_field()? {
        if field.field_number == 2 {
            version = field.data.as_int64()?;
        }
    }
    Ok(version)
}

fn parse_graph(mut reader: ProtoReader<'_>) -> Result<(Vec<OnnxNode>, Vec<OnnxTensor>, Vec<ValueInfo>, Vec<ValueInfo>)> {
    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    while let Some(field) = reader.read_field()? {
        match field.field_number {
            1 => {
                // NodeProto
                let node = parse_node(field.data.as_message()?)?;
                nodes.push(node);
            }
            5 => {
                // Initializer (TensorProto)
                let tensor = parse_tensor(field.data.as_message()?)?;
                initializers.push(tensor);
            }
            11 => {
                // Input (ValueInfoProto)
                let info = parse_value_info(field.data.as_message()?)?;
                inputs.push(info);
            }
            12 => {
                // Output (ValueInfoProto)
                let info = parse_value_info(field.data.as_message()?)?;
                outputs.push(info);
            }
            _ => {} // Ignore name (field 2), etc.
        }
    }

    Ok((nodes, initializers, inputs, outputs))
}

fn parse_node(mut reader: ProtoReader<'_>) -> Result<OnnxNode> {
    let mut name = String::new();
    let mut op_type = String::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut attributes = Vec::new();

    while let Some(field) = reader.read_field()? {
        match field.field_number {
            1 => inputs.push(field.data.as_string()?.to_string()),
            2 => outputs.push(field.data.as_string()?.to_string()),
            3 => name = field.data.as_string()?.to_string(),
            4 => op_type = field.data.as_string()?.to_string(),
            5 => {
                let attr = parse_attribute(field.data.as_message()?)?;
                attributes.push(attr);
            }
            _ => {}
        }
    }

    // Generate a name if not provided
    if name.is_empty() {
        name = format!("{}_{}", op_type, outputs.first().unwrap_or(&"unnamed".to_string()));
    }

    Ok(OnnxNode {
        name,
        op_type,
        inputs,
        outputs,
        attributes,
    })
}

fn parse_attribute(mut reader: ProtoReader<'_>) -> Result<OnnxAttribute> {
    let mut name = String::new();
    let mut value = AttributeValue::Int(0);
    let mut has_value = false;
    // For accumulating repeated non-packed values
    let mut ints_accum = Vec::new();
    let mut floats_accum = Vec::new();

    while let Some(field) = reader.read_field()? {
        match field.field_number {
            1 => name = field.data.as_string()?.to_string(),
            2 => {
                value = AttributeValue::Float(field.data.as_float()?);
                has_value = true;
            }
            3 => {
                value = AttributeValue::Int(field.data.as_int64()?);
                has_value = true;
            }
            4 => {
                value = AttributeValue::String(
                    String::from_utf8_lossy(field.data.as_bytes()?).to_string()
                );
                has_value = true;
            }
            5 => {
                // Tensor attribute (field 5 in ONNX AttributeProto is 't' for tensor)
                // Note: field 6 is 'g' for graph, which we don't support
                let tensor = parse_tensor(field.data.as_message()?)?;
                value = AttributeValue::Tensor(tensor);
                has_value = true;
            }
            7 => {
                // Repeated floats - can be packed or non-packed
                match &field.data {
                    crate::proto::FieldData::LengthDelimited(bytes) => {
                        floats_accum.extend(parse_packed_floats(bytes)?);
                    }
                    crate::proto::FieldData::Fixed32(v) => {
                        floats_accum.push(f32::from_bits(*v));
                    }
                    _ => return Err(Error::Parse("unexpected wire type for floats".into())),
                }
                has_value = true;
            }
            8 => {
                // Repeated ints - can be packed or non-packed
                match &field.data {
                    crate::proto::FieldData::LengthDelimited(bytes) => {
                        ints_accum.extend(parse_packed_int64(bytes)?);
                    }
                    crate::proto::FieldData::Varint(v) => {
                        ints_accum.push(*v as i64);
                    }
                    _ => return Err(Error::Parse("unexpected wire type for ints".into())),
                }
                has_value = true;
            }
            _ => {}
        }
    }

    // If we accumulated repeated values, use those
    if !ints_accum.is_empty() {
        value = AttributeValue::Ints(ints_accum);
    } else if !floats_accum.is_empty() {
        value = AttributeValue::Floats(floats_accum);
    } else if !has_value {
        // Default to empty ints for missing values
        value = AttributeValue::Ints(Vec::new());
    }

    Ok(OnnxAttribute { name, value })
}

fn parse_tensor(mut reader: ProtoReader<'_>) -> Result<OnnxTensor> {
    let mut name = String::new();
    let mut dims = Vec::new();
    let mut data_type = DataType::Float;
    let mut raw_data: Option<Vec<u8>> = None;
    let mut float_data = Vec::new();
    let mut int64_data = Vec::new();
    let mut int32_data = Vec::new();

    while let Some(field) = reader.read_field()? {
        match field.field_number {
            1 => {
                // dims - can be packed (length-delimited) or non-packed (repeated varint)
                match &field.data {
                    crate::proto::FieldData::LengthDelimited(bytes) => {
                        dims = parse_packed_int64(bytes)?;
                    }
                    crate::proto::FieldData::Varint(v) => {
                        dims.push(*v as i64);
                    }
                    _ => return Err(Error::Parse("unexpected wire type for dims".into())),
                }
            }
            2 => {
                data_type = DataType::try_from(field.data.as_int32()?)?;
            }
            4 => {
                // float_data (packed float)
                float_data = parse_packed_floats(field.data.as_bytes()?)?;
            }
            5 => {
                // int32_data (packed int32)
                let bytes = field.data.as_bytes()?;
                for chunk in bytes.chunks_exact(4) {
                    int32_data.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            7 => {
                // int64_data (packed int64)
                int64_data = parse_packed_int64(field.data.as_bytes()?)?;
            }
            8 => {
                name = field.data.as_string()?.to_string();
            }
            9 => {
                // raw_data
                raw_data = Some(field.data.as_bytes()?.to_vec());
            }
            _ => {}
        }
    }

    // Determine the final data bytes
    let data = if let Some(raw) = raw_data {
        raw
    } else if !float_data.is_empty() {
        // Convert float_data to raw bytes
        let mut bytes = Vec::with_capacity(float_data.len() * 4);
        for f in &float_data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        bytes
    } else if !int64_data.is_empty() {
        // Convert int64_data to raw bytes
        let mut bytes = Vec::with_capacity(int64_data.len() * 8);
        for i in &int64_data {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        bytes
    } else if !int32_data.is_empty() {
        // Convert int32_data to raw bytes
        let mut bytes = Vec::with_capacity(int32_data.len() * 4);
        for i in &int32_data {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        bytes
    } else {
        Vec::new()
    };

    Ok(OnnxTensor {
        name,
        dims,
        data_type,
        data,
    })
}

fn parse_value_info(mut reader: ProtoReader<'_>) -> Result<ValueInfo> {
    let mut name = String::new();
    let mut shape = Vec::new();
    let mut data_type = DataType::Float;

    while let Some(field) = reader.read_field()? {
        match field.field_number {
            1 => name = field.data.as_string()?.to_string(),
            2 => {
                // TypeProto
                let type_reader = field.data.as_message()?;
                let (dt, sh) = parse_type_proto(type_reader)?;
                data_type = dt;
                shape = sh;
            }
            _ => {}
        }
    }

    Ok(ValueInfo { name, shape, data_type })
}

fn parse_type_proto(mut reader: ProtoReader<'_>) -> Result<(DataType, Vec<i64>)> {
    let mut data_type = DataType::Float;
    let mut shape = Vec::new();

    while let Some(field) = reader.read_field()? {
        if field.field_number == 1 {
            // tensor_type
            let tensor_type_reader = field.data.as_message()?;
            let (dt, sh) = parse_tensor_type(tensor_type_reader)?;
            data_type = dt;
            shape = sh;
        }
    }

    Ok((data_type, shape))
}

fn parse_tensor_type(mut reader: ProtoReader<'_>) -> Result<(DataType, Vec<i64>)> {
    let mut data_type = DataType::Float;
    let mut shape = Vec::new();

    while let Some(field) = reader.read_field()? {
        match field.field_number {
            1 => data_type = DataType::try_from(field.data.as_int32()?)?,
            2 => {
                // TensorShapeProto
                shape = parse_tensor_shape(field.data.as_message()?)?;
            }
            _ => {}
        }
    }

    Ok((data_type, shape))
}

fn parse_tensor_shape(mut reader: ProtoReader<'_>) -> Result<Vec<i64>> {
    let mut dims = Vec::new();

    while let Some(field) = reader.read_field()? {
        if field.field_number == 1 {
            // Dimension
            let dim = parse_dimension(field.data.as_message()?)?;
            dims.push(dim);
        }
    }

    Ok(dims)
}

fn parse_dimension(mut reader: ProtoReader<'_>) -> Result<i64> {
    let mut value: i64 = -1; // -1 for dynamic

    while let Some(field) = reader.read_field()? {
        if field.field_number == 1 {
            // dim_value
            value = field.data.as_int64()?;
        }
        // field 2 is dim_param (string for dynamic dims), we return -1 for those
    }

    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_conversion() {
        assert_eq!(DataType::try_from(1).unwrap(), DataType::Float);
        assert_eq!(DataType::try_from(7).unwrap(), DataType::Int64);
        assert_eq!(DataType::try_from(10).unwrap(), DataType::Float16);
    }

    #[test]
    fn test_onnx_node_attr_helpers() {
        let node = OnnxNode {
            name: "test".into(),
            op_type: "Conv".into(),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            attributes: vec![
                OnnxAttribute {
                    name: "kernel_shape".into(),
                    value: AttributeValue::Ints(vec![3, 3]),
                },
                OnnxAttribute {
                    name: "strides".into(),
                    value: AttributeValue::Ints(vec![1, 1]),
                },
                OnnxAttribute {
                    name: "group".into(),
                    value: AttributeValue::Int(1),
                },
            ],
        };

        assert_eq!(node.get_attr_ints("kernel_shape"), vec![3, 3]);
        assert_eq!(node.get_attr_ints("strides"), vec![1, 1]);
        assert_eq!(node.get_attr_int("group", 1), 1);
        assert_eq!(node.get_attr_int("dilations", 1), 1); // default
    }

    #[test]
    #[ignore] // Requires artifacts/models/mobilenetv2-12.onnx
    fn test_parse_mobilenetv2() {
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../artifacts/models/mobilenetv2-12.onnx");
        let bytes = std::fs::read(model_path).expect("Failed to read model file");
        let model = parse_model(&bytes).expect("Failed to parse model");

        // MobileNetV2 basic properties
        assert_eq!(model.opset_version, 12);
        
        // Should have a reasonable number of nodes (MobileNetV2 has ~150+ nodes)
        assert!(model.nodes.len() > 100, "Expected >100 nodes, got {}", model.nodes.len());
        
        // Should have initializers (weights)
        assert!(model.initializers.len() > 50, "Expected >50 initializers, got {}", model.initializers.len());
        
        // Check input shape: [batch, 3, 224, 224] (NCHW in ONNX)
        assert_eq!(model.inputs.len(), 1);
        let input = &model.inputs[0];
        assert_eq!(input.shape.len(), 4);
        assert_eq!(input.shape[1], 3);  // Channels
        assert_eq!(input.shape[2], 224); // Height
        assert_eq!(input.shape[3], 224); // Width
        
        // Check output: should be [batch, 1000] for ImageNet classes
        assert_eq!(model.outputs.len(), 1);
        let output = &model.outputs[0];
        assert_eq!(output.shape.len(), 2);
        assert_eq!(output.shape[1], 1000); // 1000 ImageNet classes
        
        // Count op types
        let mut op_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for node in &model.nodes {
            *op_counts.entry(&node.op_type).or_insert(0) += 1;
        }
        
        println!("MobileNetV2 parsed successfully:");
        println!("  IR version: {}", model.ir_version);
        println!("  Opset version: {}", model.opset_version);
        println!("  Nodes: {}", model.nodes.len());
        println!("  Initializers: {}", model.initializers.len());
        println!("  Input: {} {:?}", input.name, input.shape);
        println!("  Output: {} {:?}", output.name, output.shape);
        println!("  Op counts:");
        let mut sorted_ops: Vec<_> = op_counts.iter().collect();
        sorted_ops.sort_by_key(|(k, _)| *k);
        for (op, count) in sorted_ops {
            println!("    {}: {}", op, count);
        }
        
        // Print special ops (non-standard ones we may need to handle)
        println!("\n  Shape manipulation ops:");
        for node in &model.nodes {
            match node.op_type.as_str() {
                "Shape" | "Gather" | "Unsqueeze" | "Concat" | "Constant" | "Reshape" => {
                    println!("    {} ({}): inputs={:?} outputs={:?}", 
                        node.op_type, node.name, node.inputs, node.outputs);
                    for attr in &node.attributes {
                        println!("      attr: {} = {:?}", attr.name, attr.value);
                    }
                }
                _ => {}
            }
        }
        
        // Check for tensor 630
        println!("\n  Looking for tensor '630':");
        for init in &model.initializers {
            if init.name == "630" {
                println!("    Found as initializer: dims={:?}, dtype={:?}, data_len={}", 
                    init.dims, init.data_type, init.data.len());
                if init.data.len() <= 16 {
                    println!("    data: {:?}", init.data);
                }
            }
        }
    }
}
