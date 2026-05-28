//! Error types for the ONNX loader.

use std::fmt;

/// Error type for ONNX loading and compilation.
#[derive(Debug)]
pub enum Error {
    /// I/O error (file not found, read failure, etc.).
    Io(String),
    /// Protobuf parsing error (malformed varint, unexpected wire type, etc.).
    Parse(String),
    /// Unsupported ONNX opset version (we only support opset 12).
    UnsupportedOpset { found: i64, expected: i64 },
    /// Unsupported ONNX operator.
    UnsupportedOp { op_type: String, node_name: String, reason: String },
    /// Shape inference error.
    Shape(String),
    /// Graph validation error.
    Validation(String),
    /// Compilation error.
    Compile(String),
    /// Runtime execution error.
    Runtime(String),
    /// Backend error (from dragonwing-core).
    Backend(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
            Self::UnsupportedOpset { found, expected } => {
                write!(f, "unsupported opset version {found}, expected {expected}")
            }
            Self::UnsupportedOp { op_type, node_name, reason } => {
                write!(f, "unsupported op '{op_type}' at node '{node_name}': {reason}")
            }
            Self::Shape(msg) => write!(f, "shape error: {msg}"),
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::Compile(msg) => write!(f, "compilation error: {msg}"),
            Self::Runtime(msg) => write!(f, "runtime error: {msg}"),
            Self::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type alias for ONNX operations.
pub type Result<T> = std::result::Result<T, Error>;
