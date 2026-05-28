//! Hand-rolled minimal protobuf reader for ONNX.
//!
//! ONNX uses only a subset of protobuf wire types:
//! - Varint (type 0): integers, bools, enums
//! - Length-delimited (type 2): strings, bytes, embedded messages, packed repeated
//! - Fixed64 (type 1): double
//! - Fixed32 (type 5): float
//!
//! We do not need groups (deprecated) or start/end group markers.
//!
//! This reader is ~300 lines and has zero external dependencies.

use crate::error::{Error, Result};

/// Wire types as defined by protobuf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WireType {
    Varint = 0,
    Fixed64 = 1,
    LengthDelimited = 2,
    // StartGroup = 3,  // Deprecated, not used in ONNX
    // EndGroup = 4,    // Deprecated, not used in ONNX
    Fixed32 = 5,
}

impl TryFrom<u8> for WireType {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Varint),
            1 => Ok(Self::Fixed64),
            2 => Ok(Self::LengthDelimited),
            5 => Ok(Self::Fixed32),
            _ => Err(Error::Parse(format!("unknown wire type: {value}"))),
        }
    }
}

/// A protobuf field: tag (field number + wire type) plus value.
#[derive(Debug)]
pub struct Field<'a> {
    pub field_number: u32,
    pub wire_type: WireType,
    pub data: FieldData<'a>,
}

/// The raw data of a protobuf field.
#[derive(Debug)]
pub enum FieldData<'a> {
    Varint(u64),
    Fixed64(u64),
    Fixed32(u32),
    LengthDelimited(&'a [u8]),
}

impl<'a> FieldData<'a> {
    /// Get as varint (u64).
    pub fn as_varint(&self) -> Result<u64> {
        match self {
            Self::Varint(v) => Ok(*v),
            _ => Err(Error::Parse("expected varint".into())),
        }
    }

    /// Get as i64 (signed varint via zigzag decoding).
    pub fn as_sint64(&self) -> Result<i64> {
        let v = self.as_varint()?;
        // Zigzag decode: (v >> 1) ^ -(v & 1)
        Ok(((v >> 1) as i64) ^ (-((v & 1) as i64)))
    }

    /// Get as i32 (signed varint).
    pub fn as_int32(&self) -> Result<i32> {
        self.as_varint().map(|v| v as i32)
    }

    /// Get as i64 (signed varint, no zigzag).
    pub fn as_int64(&self) -> Result<i64> {
        self.as_varint().map(|v| v as i64)
    }

    /// Get as bool.
    pub fn as_bool(&self) -> Result<bool> {
        self.as_varint().map(|v| v != 0)
    }

    /// Get as f32 (fixed32).
    pub fn as_float(&self) -> Result<f32> {
        match self {
            Self::Fixed32(v) => Ok(f32::from_bits(*v)),
            _ => Err(Error::Parse("expected fixed32 for float".into())),
        }
    }

    /// Get as f64 (fixed64).
    pub fn as_double(&self) -> Result<f64> {
        match self {
            Self::Fixed64(v) => Ok(f64::from_bits(*v)),
            _ => Err(Error::Parse("expected fixed64 for double".into())),
        }
    }

    /// Get as bytes (length-delimited).
    pub fn as_bytes(&self) -> Result<&'a [u8]> {
        match self {
            Self::LengthDelimited(b) => Ok(b),
            _ => Err(Error::Parse("expected length-delimited".into())),
        }
    }

    /// Get as UTF-8 string (length-delimited).
    pub fn as_string(&self) -> Result<&'a str> {
        let bytes = self.as_bytes()?;
        std::str::from_utf8(bytes)
            .map_err(|e| Error::Parse(format!("invalid UTF-8: {e}")))
    }

    /// Get as embedded message reader.
    pub fn as_message(&self) -> Result<ProtoReader<'a>> {
        let bytes = self.as_bytes()?;
        Ok(ProtoReader::new(bytes))
    }
}

/// A streaming protobuf reader.
#[derive(Debug, Clone)]
pub struct ProtoReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    /// Create a new reader over the given bytes.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Returns true if there's more data to read.
    pub fn has_remaining(&self) -> bool {
        self.pos < self.data.len()
    }

    /// Read a single byte, advancing the position.
    fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(Error::Parse("unexpected end of input".into()));
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    /// Read a varint (up to 10 bytes for u64).
    pub fn read_varint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            let b = self.read_byte()?;
            result |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(Error::Parse("varint too long".into()));
            }
        }
    }

    /// Read a fixed32 (4 bytes, little-endian).
    fn read_fixed32(&mut self) -> Result<u32> {
        if self.pos + 4 > self.data.len() {
            return Err(Error::Parse("unexpected end of input for fixed32".into()));
        }
        let bytes = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a fixed64 (8 bytes, little-endian).
    fn read_fixed64(&mut self) -> Result<u64> {
        if self.pos + 8 > self.data.len() {
            return Err(Error::Parse("unexpected end of input for fixed64".into()));
        }
        let bytes = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Read length-delimited bytes.
    fn read_length_delimited(&mut self) -> Result<&'a [u8]> {
        let len = self.read_varint()? as usize;
        if self.pos + len > self.data.len() {
            return Err(Error::Parse(format!(
                "length-delimited field extends past end: pos={}, len={}, total={}",
                self.pos, len, self.data.len()
            )));
        }
        let result = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(result)
    }

    /// Read the next field (tag + data).
    pub fn read_field(&mut self) -> Result<Option<Field<'a>>> {
        if !self.has_remaining() {
            return Ok(None);
        }

        let tag = self.read_varint()?;
        let field_number = (tag >> 3) as u32;
        let wire_type = WireType::try_from((tag & 0x07) as u8)?;

        let data = match wire_type {
            WireType::Varint => FieldData::Varint(self.read_varint()?),
            WireType::Fixed64 => FieldData::Fixed64(self.read_fixed64()?),
            WireType::Fixed32 => FieldData::Fixed32(self.read_fixed32()?),
            WireType::LengthDelimited => FieldData::LengthDelimited(self.read_length_delimited()?),
        };

        Ok(Some(Field { field_number, wire_type, data }))
    }

    /// Skip to end, useful for ignoring unknown fields.
    pub fn skip_to_end(&mut self) {
        self.pos = self.data.len();
    }
}

/// Parse packed repeated int64 values.
pub fn parse_packed_int64(data: &[u8]) -> Result<Vec<i64>> {
    let mut reader = ProtoReader::new(data);
    let mut result = Vec::new();
    while reader.has_remaining() {
        result.push(reader.read_varint()? as i64);
    }
    Ok(result)
}

/// Parse packed repeated float values.
pub fn parse_packed_floats(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() % 4 != 0 {
        return Err(Error::Parse("packed floats length not multiple of 4".into()));
    }
    let mut result = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        result.push(f32::from_bits(bits));
    }
    Ok(result)
}

/// Parse packed repeated double values.
pub fn parse_packed_doubles(data: &[u8]) -> Result<Vec<f64>> {
    if data.len() % 8 != 0 {
        return Err(Error::Parse("packed doubles length not multiple of 8".into()));
    }
    let mut result = Vec::with_capacity(data.len() / 8);
    for chunk in data.chunks_exact(8) {
        let bits = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        result.push(f64::from_bits(bits));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_simple() {
        // 0x01 = 1
        let mut reader = ProtoReader::new(&[0x01]);
        assert_eq!(reader.read_varint().unwrap(), 1);
    }

    #[test]
    fn test_varint_multi_byte() {
        // 0x96 0x01 = 150 (0b10010110 0b00000001 -> 0b0010110 + 0b0000001 << 7 = 22 + 128 = 150)
        let mut reader = ProtoReader::new(&[0x96, 0x01]);
        assert_eq!(reader.read_varint().unwrap(), 150);
    }

    #[test]
    fn test_field_parsing() {
        // Field 1, wire type 0 (varint), value 150
        // Tag: (1 << 3) | 0 = 0x08
        // Value: 0x96 0x01
        let data = [0x08, 0x96, 0x01];
        let mut reader = ProtoReader::new(&data);
        let field = reader.read_field().unwrap().unwrap();
        assert_eq!(field.field_number, 1);
        assert_eq!(field.wire_type, WireType::Varint);
        assert_eq!(field.data.as_varint().unwrap(), 150);
    }

    #[test]
    fn test_length_delimited() {
        // Field 2, wire type 2 (length-delimited), string "testing"
        // Tag: (2 << 3) | 2 = 0x12
        // Length: 7 (0x07)
        // Data: "testing"
        let data = [0x12, 0x07, b't', b'e', b's', b't', b'i', b'n', b'g'];
        let mut reader = ProtoReader::new(&data);
        let field = reader.read_field().unwrap().unwrap();
        assert_eq!(field.field_number, 2);
        assert_eq!(field.wire_type, WireType::LengthDelimited);
        assert_eq!(field.data.as_string().unwrap(), "testing");
    }
}
