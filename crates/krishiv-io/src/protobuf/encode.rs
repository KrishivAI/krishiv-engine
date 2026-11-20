// SPDX-License-Identifier: Apache-2.0
//! `to_protobuf`: a struct column as messages (Spark's `ProtobufSerializer`).
//!
//! Fields match by name, ignoring case (two proto fields that differ only by case are `PROTOBUF_FIELD_MISSING`); the Spark type of
//! each struct field must be the one the proto kind takes (`int` into the 32-bit kinds, `bigint` into the 64-bit kinds and
//! `uint32`, `decimal` into `uint64`, `timestamp` and `interval day to second` into messages written as `Timestamp` and `Duration`,
//! primitives into the wrappers); anything else is `UNABLE_TO_CONVERT_TO_PROTOBUF_MESSAGE_TYPE` naming the root message. NULL values
//! and (proto3) zero values of plain scalars are not written. The options of `to_protobuf` are never read.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::desc::{EnumDescriptor, FieldDescriptor, Kind, MessageDescriptor};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, DurationMicrosecondType, Float32Type, Float64Type, Int8Type,
    Int16Type, Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType};
use arrow_schema::{DataType as ArrowType, Field};
use krishiv_common::error::{
    CannotConvertSqlValueToProtobufEnumType, InternalError, KrishivError,
    NoSqlTypeInProtobufSchema, ProtobufFieldMissing, ProtobufFieldMissingInSqlSchema,
    UnableToConvertToProtobufMessageType,
};
use krishiv_types::DataType;

use super::DescriptorSet;
use crate::error::unsupported;

/// How one struct field is written.
#[derive(Debug, Clone)]
enum Conv {
    /// A `void` (NULL-typed) value: never written.
    Null,
    Bool,
    /// `int`, `smallint`, `tinyint` or `date` into a 32-bit kind.
    Int32,
    /// `bigint` into `uint32` (wraps).
    U32FromLong,
    Int64,
    /// `decimal` into `uint64`.
    U64FromDecimal,
    Float,
    Double,
    EnumName(EnumDescriptor),
    EnumInt(EnumDescriptor),
    Str,
    Bin,
    Timestamp,
    Duration,
    /// A primitive into a wrapper message.
    Wrapper(Wrapped),
    Struct(Box<EncStruct>),
    List(Box<Conv>),
    Map(Box<FieldEnc>, Box<FieldEnc>),
}

/// The primitive a wrapper message holds.
#[derive(Debug, Clone, Copy)]
enum Wrapped {
    Bool,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Str,
    Bin,
    Float,
    Double,
}

#[derive(Debug, Clone)]
struct FieldEnc {
    fd: FieldDescriptor,
    conv: Conv,
    /// The path of the Spark field, for error texts.
    path: String,
}

#[derive(Debug, Clone)]
struct EncStruct {
    /// The Spark struct field index of each proto field written, in struct order.
    fields: Vec<(usize, FieldEnc)>,
    /// Required proto fields (a NULL scalar there writes the default).
    msg: MessageDescriptor,
}

/// Writes struct values of one Spark type as messages of one type (see the module documentation).
#[derive(Debug, Clone)]
pub struct Encoder {
    root: EncStruct,
}

fn quote(path: &[&str]) -> String {
    path.iter()
        .map(|p| format!("`{p}`"))
        .collect::<Vec<_>>()
        .join(".")
}

impl Encoder {
    /// An encoder of values of Spark type `input` (a struct) as `message`.
    ///
    /// # Errors
    /// `PROTOBUF_MESSAGE_NOT_FOUND`; `UNABLE_TO_CONVERT_TO_PROTOBUF_MESSAGE_TYPE` for a struct that does not fit the message.
    pub fn new(set: &DescriptorSet, message: &str, input: &DataType) -> Result<Self, KrishivError> {
        let msg = set.message(message)?;
        let DataType::Struct(st) = input else {
            return Err(unsupported(
                "to_protobuf of a value that is not a struct".to_owned(),
            ));
        };
        // Spark wraps every analysis error of the conversion in one naming the root.
        let root = struct_conv(st, &msg, &mut Vec::new()).map_err(|_| {
            KrishivError::from(UnableToConvertToProtobufMessageType {
                to_type: format!("\"{}\"", input.sql()),
                protobuf_type: msg.name().to_owned(),
            })
        })?;
        Ok(Self { root })
    }

    /// Writes each row of `array` (a struct array); NULL stays NULL.
    ///
    /// # Errors
    /// `CANNOT_CONVERT_SQL_VALUE_TO_PROTOBUF_ENUM_TYPE` for an enum value that is not defined.
    pub fn encode(&self, array: &ArrayRef) -> Result<Vec<Option<Vec<u8>>>, KrishivError> {
        let Some(st) = array.as_struct_opt() else {
            return Err(internal("to_protobuf: the argument is not a struct array"));
        };
        let mut out = Vec::with_capacity(st.len());
        for row in 0..st.len() {
            if st.is_null(row) {
                out.push(None);
            } else {
                let mut buf = Vec::new();
                write_struct(&self.root, st, row, &mut buf)?;
                out.push(Some(buf));
            }
        }
        Ok(out)
    }
}

fn internal(m: &str) -> KrishivError {
    InternalError {
        message: m.to_owned(),
    }
    .into()
}

fn mismatch() -> KrishivError {
    // The text is replaced by the root error; any analysis error will do.
    NoSqlTypeInProtobufSchema {
        catalyst_field_path: String::new(),
    }
    .into()
}

/// The proto field of `msg` named like `name`, ignoring case.
fn match_field(
    msg: &MessageDescriptor,
    name: &str,
    path: &[&str],
) -> Result<Option<FieldDescriptor>, KrishivError> {
    let lower = name.to_lowercase();
    let mut found: Vec<FieldDescriptor> = msg
        .fields()
        .filter(|f| f.name().to_lowercase() == lower)
        .collect();
    match found.len() {
        0 => Ok(None),
        1 => Ok(found.pop()),
        n => Err(ProtobufFieldMissing {
            field: name.to_owned(),
            protobuf_schema: if path.is_empty() {
                "top-level record".to_owned()
            } else {
                format!("field '{}'", path.join("."))
            },
            match_size: n.to_string(),
            matches: format!(
                "[{}]",
                found
                    .iter()
                    .map(|f| f.name().to_owned())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
        .into()),
    }
}

fn struct_conv(
    st: &krishiv_types::StructType,
    msg: &MessageDescriptor,
    path: &mut Vec<String>,
) -> Result<EncStruct, KrishivError> {
    let proto_path: Vec<&str> = path.iter().map(String::as_str).collect();
    let mut matched: Vec<(usize, FieldDescriptor)> = Vec::new();
    for (i, f) in st.fields().iter().enumerate() {
        if let Some(fd) = match_field(msg, f.name(), &proto_path)? {
            matched.push((i, fd));
        }
    }
    for f in st.fields() {
        if match_field(msg, f.name(), &proto_path)?.is_none() {
            let mut p = proto_path.clone();
            p.push(f.name());
            return Err(NoSqlTypeInProtobufSchema {
                catalyst_field_path: format!("field '{}'", p.join(".")),
            }
            .into());
        }
    }
    for fd in msg.fields() {
        if fd.is_required() && !matched.iter().any(|(_, m)| m.number() == fd.number()) {
            return Err(ProtobufFieldMissingInSqlSchema {
                field: format!("field '{}'", fd.name()),
            }
            .into());
        }
    }
    let mut fields = Vec::new();
    for (i, fd) in matched {
        let f = &st.fields()[i];
        path.push(f.name().to_owned());
        let conv = conv_for(f.data_type(), &fd, path)?;
        let p: Vec<&str> = path.iter().map(String::as_str).collect();
        fields.push((
            i,
            FieldEnc {
                fd,
                conv,
                path: quote(&p),
            },
        ));
        path.pop();
    }
    Ok(EncStruct {
        fields,
        msg: msg.clone(),
    })
}

fn wrapper(full: &str) -> Option<Wrapped> {
    Some(match full {
        "google.protobuf.BoolValue" => Wrapped::Bool,
        "google.protobuf.Int32Value" => Wrapped::Int32,
        "google.protobuf.UInt32Value" => Wrapped::UInt32,
        "google.protobuf.Int64Value" => Wrapped::Int64,
        "google.protobuf.UInt64Value" => Wrapped::UInt64,
        "google.protobuf.StringValue" => Wrapped::Str,
        "google.protobuf.BytesValue" => Wrapped::Bin,
        "google.protobuf.FloatValue" => Wrapped::Float,
        "google.protobuf.DoubleValue" => Wrapped::Double,
        _ => return None,
    })
}

/// The writer of a value of Spark type `t` into field `fd` (`serializer.newConverter`).
fn conv_for(
    t: &DataType,
    fd: &FieldDescriptor,
    path: &mut Vec<String>,
) -> Result<Conv, KrishivError> {
    use Kind as K;
    let kind = fd.kind();
    if let DataType::Array(a) = t {
        if !fd.is_list() {
            return Err(mismatch());
        }
        return Ok(Conv::List(Box::new(conv_for(a.element_type(), fd, path)?)));
    }
    Ok(match (t, &kind) {
        (DataType::Null, _) => Conv::Null,
        (DataType::Boolean, K::Bool) => Conv::Bool,
        (
            DataType::Byte | DataType::Short | DataType::Integer | DataType::Date,
            K::Int32 | K::Sint32 | K::Sfixed32 | K::Uint32 | K::Fixed32,
        ) => Conv::Int32,
        (DataType::Long, K::Uint32) => Conv::U32FromLong,
        (DataType::Long, K::Int64 | K::Sint64 | K::Sfixed64 | K::Uint64 | K::Fixed64) => {
            Conv::Int64
        }
        (DataType::Decimal(_), K::Uint64) => Conv::U64FromDecimal,
        (DataType::Float, K::Float) => Conv::Float,
        (DataType::Double, K::Double) => Conv::Double,
        (DataType::String, K::Enum(e)) => Conv::EnumName(e.clone()),
        (DataType::Integer, K::Enum(e)) => Conv::EnumInt(e.clone()),
        (DataType::String | DataType::CollatedString(_), K::String) => Conv::Str,
        (DataType::Binary, K::Bytes) => Conv::Bin,
        (DataType::Timestamp, K::Message(_)) => Conv::Timestamp,
        (DataType::DayTimeInterval { .. }, K::Message(_)) => Conv::Duration,
        (_, K::Message(m)) => {
            if let Some(w) = wrapper(m.full_name()) {
                let ok = matches!(
                    (w, t),
                    (Wrapped::Bool, DataType::Boolean)
                        | (Wrapped::Int32 | Wrapped::UInt32, DataType::Integer)
                        | (Wrapped::Int64 | Wrapped::UInt64, DataType::Long)
                        | (Wrapped::Str, DataType::String)
                        | (Wrapped::Bin, DataType::Binary)
                        | (Wrapped::Float, DataType::Float)
                        | (Wrapped::Double, DataType::Double)
                );
                if ok {
                    return Ok(Conv::Wrapper(w));
                }
            }
            match t {
                DataType::Struct(st) => Conv::Struct(Box::new(struct_conv(st, m, path)?)),
                DataType::Map(mt) if fd.is_map() => {
                    let (kf, vf) = m
                        .map_entry_fields()
                        .ok_or_else(|| internal("a map entry without its key and value"))?;
                    path.push("key".to_owned());
                    let k = conv_for(mt.key_type(), &kf, path)?;
                    path.pop();
                    path.push("value".to_owned());
                    let v = conv_for(mt.value_type(), &vf, path)?;
                    path.pop();
                    Conv::Map(
                        Box::new(FieldEnc {
                            fd: kf,
                            conv: k,
                            path: String::new(),
                        }),
                        Box::new(FieldEnc {
                            fd: vf,
                            conv: v,
                            path: String::new(),
                        }),
                    )
                }
                _ => return Err(mismatch()),
            }
        }
        _ => return Err(mismatch()),
    })
}

// ---- wire writing ----

fn put_varint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

fn put_tag(buf: &mut Vec<u8>, number: u32, wire: u8) {
    put_varint(buf, (u64::from(number) << 3) | u64::from(wire));
}

fn put_len(buf: &mut Vec<u8>, number: u32, bytes: &[u8]) {
    put_tag(buf, number, 2);
    put_varint(buf, bytes.len() as u64);
    buf.extend_from_slice(bytes);
}

/// A scalar value ready for the wire.
enum Scalar {
    Varint(u64),
    Fixed32(u32),
    Fixed64(u64),
    Len(Vec<u8>),
    /// An embedded message (always written, empty or not).
    Message(Vec<u8>),
    /// An embedded group: the body between a start-group and an end-group tag.
    Group(Vec<u8>),
}

impl Scalar {
    fn is_default(&self) -> bool {
        match self {
            Self::Varint(v) => *v == 0,
            Self::Fixed32(v) => *v == 0,
            Self::Fixed64(v) => *v == 0,
            Self::Len(b) => b.is_empty(),
            Self::Message(_) | Self::Group(_) => false,
        }
    }

    fn write(&self, number: u32, buf: &mut Vec<u8>) {
        match self {
            Self::Varint(v) => {
                put_tag(buf, number, 0);
                put_varint(buf, *v);
            }
            Self::Fixed32(v) => {
                put_tag(buf, number, 5);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Self::Fixed64(v) => {
                put_tag(buf, number, 1);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Self::Len(b) | Self::Message(b) => put_len(buf, number, b),
            Self::Group(b) => {
                put_tag(buf, number, 3);
                buf.extend_from_slice(b);
                put_tag(buf, number, 4);
            }
        }
    }

    /// The value inside a packed field.
    fn write_packed(&self, buf: &mut Vec<u8>) {
        match self {
            Self::Varint(v) => put_varint(buf, *v),
            Self::Fixed32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            Self::Fixed64(v) => buf.extend_from_slice(&v.to_le_bytes()),
            Self::Len(_) | Self::Message(_) | Self::Group(_) => {}
        }
    }
}

fn prim<T: ArrowPrimitiveType>(a: &dyn Array, row: usize) -> Result<T::Native, KrishivError> {
    a.as_primitive_opt::<T>()
        .map(|p| p.value(row))
        .ok_or_else(|| internal("to_protobuf: unexpected column type"))
}

/// The value of an integer column as i64.
fn int_of(a: &dyn Array, row: usize) -> Result<i64, KrishivError> {
    Ok(match a.data_type() {
        ArrowType::Int8 => i64::from(prim::<Int8Type>(a, row)?),
        ArrowType::Int16 => i64::from(prim::<Int16Type>(a, row)?),
        ArrowType::Int32 => i64::from(prim::<Int32Type>(a, row)?),
        ArrowType::Date32 => i64::from(prim::<Date32Type>(a, row)?),
        ArrowType::Int64 => prim::<Int64Type>(a, row)?,
        _ => return Err(internal("to_protobuf: unexpected integer column type")),
    })
}

fn str_of(a: &dyn Array, row: usize) -> Result<String, KrishivError> {
    let s: &str = match a.data_type() {
        ArrowType::Utf8View => a.as_string_view().value(row),
        ArrowType::Utf8 => a.as_string::<i32>().value(row),
        ArrowType::LargeUtf8 => a.as_string::<i64>().value(row),
        _ => return Err(internal("to_protobuf: unexpected string column type")),
    };
    Ok(s.to_owned())
}

fn bin_of(a: &dyn Array, row: usize) -> Result<Vec<u8>, KrishivError> {
    Ok(match a.data_type() {
        ArrowType::BinaryView => a.as_binary_view().value(row).to_vec(),
        ArrowType::Binary => a.as_binary::<i32>().value(row).to_vec(),
        ArrowType::LargeBinary => a.as_binary::<i64>().value(row).to_vec(),
        // A string whose bytes are not UTF-8 is stored as binary.
        ArrowType::Utf8View => a.as_string_view().value(row).as_bytes().to_vec(),
        _ => return Err(internal("to_protobuf: unexpected binary column type")),
    })
}

fn decimal_unscaled(a: &dyn Array, row: usize) -> Result<i128, KrishivError> {
    Ok(match a.data_type() {
        ArrowType::Decimal128(..) => prim::<Decimal128Type>(a, row)?,
        ArrowType::Decimal64(..) => i128::from(
            a.as_primitive_opt::<arrow_array::types::Decimal64Type>()
                .map(|p| p.value(row))
                .ok_or_else(|| internal("to_protobuf: unexpected decimal column"))?,
        ),
        ArrowType::Decimal32(..) => i128::from(
            a.as_primitive_opt::<arrow_array::types::Decimal32Type>()
                .map(|p| p.value(row))
                .ok_or_else(|| internal("to_protobuf: unexpected decimal column"))?,
        ),
        _ => return Err(internal("to_protobuf: unexpected decimal column type")),
    })
}

/// A scalar of kind `kind` from a 32-bit signed value.
fn from_i32(kind: &Kind, v: i32) -> Scalar {
    match kind {
        Kind::Sint32 => Scalar::Varint(u64::from(((v << 1) ^ (v >> 31)) as u32)),
        Kind::Sfixed32 | Kind::Fixed32 => Scalar::Fixed32(v as u32),
        Kind::Uint32 => Scalar::Varint(u64::from(v as u32)),
        _ => Scalar::Varint(i64::from(v) as u64),
    }
}

fn from_i64(kind: &Kind, v: i64) -> Scalar {
    match kind {
        Kind::Sint64 => Scalar::Varint(((v << 1) ^ (v >> 63)) as u64),
        Kind::Sfixed64 | Kind::Fixed64 => Scalar::Fixed64(v as u64),
        _ => Scalar::Varint(v as u64),
    }
}

fn wrapper_message(w: Wrapped, a: &dyn Array, row: usize) -> Result<Vec<u8>, KrishivError> {
    let mut b = Vec::new();
    let s = match w {
        Wrapped::Bool => {
            Scalar::Varint(u64::from(a.as_boolean_opt().is_some_and(|x| x.value(row))))
        }
        Wrapped::Int32 => Scalar::Varint(int_of(a, row)? as u64),
        Wrapped::UInt32 => Scalar::Varint(u64::from(int_of(a, row)? as u32)),
        Wrapped::Int64 | Wrapped::UInt64 => Scalar::Varint(int_of(a, row)? as u64),
        Wrapped::Str => Scalar::Len(str_of(a, row)?.into_bytes()),
        Wrapped::Bin => Scalar::Len(bin_of(a, row)?),
        Wrapped::Float => Scalar::Fixed32(prim::<Float32Type>(a, row)?.to_bits()),
        Wrapped::Double => Scalar::Fixed64(prim::<Float64Type>(a, row)?.to_bits()),
    };
    if !s.is_default() {
        s.write(1, &mut b);
    }
    Ok(b)
}

fn seconds_nanos(seconds: i64, nanos: i32) -> Vec<u8> {
    let mut b = Vec::new();
    if seconds != 0 {
        put_tag(&mut b, 1, 0);
        put_varint(&mut b, seconds as u64);
    }
    if nanos != 0 {
        put_tag(&mut b, 2, 0);
        put_varint(&mut b, i64::from(nanos) as u64);
    }
    b
}

fn enum_error(path: &str, fd: &FieldDescriptor, data: String, listed: String) -> KrishivError {
    CannotConvertSqlValueToProtobufEnumType {
        sql_column: path.to_owned(),
        protobuf_column: format!("field '{}'", fd.name()),
        data,
        enum_string: listed,
    }
    .into()
}

/// One non-null value as a scalar; `None` for a repeated or map value, which `write_field` handles.
fn scalar(f: &FieldEnc, conv: &Conv, a: &dyn Array, row: usize) -> Result<Scalar, KrishivError> {
    let kind = f.fd.kind();
    Ok(match conv {
        Conv::Bool => Scalar::Varint(u64::from(
            a.as_boolean_opt()
                .ok_or_else(|| internal("to_protobuf: expected a boolean column"))?
                .value(row),
        )),
        Conv::Int32 => from_i32(&kind, int_of(a, row)? as i32),
        Conv::U32FromLong => Scalar::Varint(u64::from(int_of(a, row)? as u32)),
        Conv::Int64 => from_i64(&kind, int_of(a, row)?),
        Conv::U64FromDecimal => {
            let v = decimal_unscaled(a, row)?;
            let v = i64::try_from(v).map_err(|_| {
                // Spark raises java.lang.ArithmeticException (no error class): "BigInteger out of long range".
                internal("BigInteger out of long range")
            })?;
            Scalar::Varint(v as u64)
        }
        Conv::Float => Scalar::Fixed32(prim::<Float32Type>(a, row)?.to_bits()),
        Conv::Double => Scalar::Fixed64(prim::<Float64Type>(a, row)?.to_bits()),
        Conv::EnumName(e) => {
            let name = str_of(a, row)?;
            match e.get_value_by_name(&name) {
                Some(v) => Scalar::Varint(i64::from(v.number()) as u64),
                None => {
                    let listed = e
                        .values()
                        .iter()
                        .map(|v| format!("\"{}\"", v.name()))
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(enum_error(&f.path, &f.fd, name, listed));
                }
            }
        }
        Conv::EnumInt(e) => {
            let n = int_of(a, row)? as i32;
            match e.get_value(n) {
                Some(_) => Scalar::Varint(i64::from(n) as u64),
                None => {
                    let listed = e
                        .values()
                        .iter()
                        .map(|v| v.number().to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(enum_error(&f.path, &f.fd, n.to_string(), listed));
                }
            }
        }
        Conv::Str => Scalar::Len(str_of(a, row)?.into_bytes()),
        Conv::Bin => Scalar::Len(bin_of(a, row)?),
        Conv::Timestamp => {
            let micros = prim::<TimestampMicrosecondType>(a, row)?;
            let millis = micros.div_euclid(1000);
            Scalar::Message(seconds_nanos(
                millis / 1000,
                ((millis % 1000) * 1_000_000) as i32,
            ))
        }
        Conv::Duration => {
            let micros = prim::<DurationMicrosecondType>(a, row)?;
            let millis = micros.div_euclid(1000);
            let (mut seconds, mut nanos) = (millis / 1000, ((millis % 1000) * 1_000_000) as i32);
            if seconds < 0 && nanos > 0 {
                seconds += 1;
                nanos -= 1_000_000_000;
            } else if seconds > 0 && nanos < 0 {
                seconds -= 1;
                nanos += 1_000_000_000;
            }
            Scalar::Message(seconds_nanos(seconds, nanos))
        }
        Conv::Wrapper(w) => Scalar::Message(wrapper_message(*w, a, row)?),
        Conv::Struct(s) => {
            let st = a
                .as_struct_opt()
                .ok_or_else(|| internal("to_protobuf: expected a struct column"))?;
            let mut b = Vec::new();
            write_struct(s, st, row, &mut b)?;
            if f.fd.is_group() {
                Scalar::Group(b)
            } else {
                Scalar::Message(b)
            }
        }
        Conv::Null => return Err(internal("to_protobuf: a NULL-typed value")),
        Conv::List(_) | Conv::Map(..) => return Err(internal("to_protobuf: a nested collection")),
    })
}

/// Writes the fields of one struct row, in field-number order, a oneof keeping its last member.
fn write_struct(
    s: &EncStruct,
    st: &arrow_array::StructArray,
    row: usize,
    out: &mut Vec<u8>,
) -> Result<(), KrishivError> {
    let mut written: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    for (i, f) in &s.fields {
        let column = st.column(*i);
        if column.is_null(row) {
            // A NULL required scalar writes the type's default; a NULL required message is left out (Spark fails to build it).
            if f.fd.is_required() && !f.fd.is_list() && !matches!(f.fd.kind(), Kind::Message(_)) {
                let mut b = Vec::new();
                default_scalar(&f.fd).write(f.fd.number(), &mut b);
                written.insert(f.fd.number(), b);
            } else if f.fd.is_required() && matches!(f.fd.kind(), Kind::Message(_)) {
                // `UninitializedMessageException` in Spark, without an error class.
                return Err(internal(&format!(
                    "Message missing required fields: {}",
                    f.fd.name()
                )));
            }
            continue;
        }
        let mut b = Vec::new();
        write_field(f, column.as_ref(), row, &mut b)?;
        if let Some(oneof) = f.fd.containing_oneof()
            && !oneof.is_synthetic()
        {
            for other in oneof.fields() {
                written.remove(&other.number());
            }
        }
        if b.is_empty() {
            written.remove(&f.fd.number());
        } else {
            written.insert(f.fd.number(), b);
        }
    }
    for fd in s.msg.fields() {
        if fd.is_required()
            && !written.contains_key(&fd.number())
            && !s.fields.iter().any(|(_, f)| f.fd.number() == fd.number())
        {
            return Err(internal(&format!(
                "Message missing required fields: {}",
                fd.name()
            )));
        }
    }
    for bytes in written.values() {
        out.extend_from_slice(bytes);
    }
    Ok(())
}

/// The value Spark writes for a NULL field: the descriptor's default (an explicit proto2 `[default = ..]`, else the first
/// value of an enum, else the type's zero).
fn default_scalar(fd: &FieldDescriptor) -> Scalar {
    use super::desc::DefaultValue;
    if let Some(d) = fd.explicit_default() {
        return match d {
            DefaultValue::Varint(v) => Scalar::Varint(v),
            DefaultValue::Fixed32(v) => Scalar::Fixed32(v),
            DefaultValue::Fixed64(v) => Scalar::Fixed64(v),
            DefaultValue::Bytes(b) => Scalar::Len(b),
        };
    }
    match fd.kind() {
        Kind::Float | Kind::Fixed32 | Kind::Sfixed32 => Scalar::Fixed32(0),
        Kind::Double | Kind::Fixed64 | Kind::Sfixed64 => Scalar::Fixed64(0),
        Kind::String | Kind::Bytes => Scalar::Len(Vec::new()),
        Kind::Message(_) => Scalar::Message(Vec::new()),
        Kind::Enum(e) => Scalar::Varint(
            e.values()
                .first()
                .map_or(0, |v| i64::from(v.number()) as u64),
        ),
        _ => Scalar::Varint(0),
    }
}

/// Writes one field of a row (a list, a map or one value).
fn write_field(
    f: &FieldEnc,
    a: &dyn Array,
    row: usize,
    out: &mut Vec<u8>,
) -> Result<(), KrishivError> {
    let number = f.fd.number();
    match &f.conv {
        Conv::List(inner) => {
            let (values, start, end) = list_range(a, row)?;
            let element = FieldEnc {
                fd: f.fd.clone(),
                conv: (**inner).clone(),
                path: f.path.clone(),
            };
            let mut scalars = Vec::with_capacity(end - start);
            for i in start..end {
                if values.is_null(i) {
                    // Spark: a NULL element fails the plan (an optimizer-phase internal error).
                    return Err(internal("to_protobuf: an array with a NULL element"));
                }
                scalars.push(scalar(&element, &element.conv, values.as_ref(), i)?);
            }
            if scalars.is_empty() {
                return Ok(());
            }
            if f.fd.is_packed()
                && !matches!(f.fd.kind(), Kind::String | Kind::Bytes | Kind::Message(_))
            {
                let mut body = Vec::new();
                for s in &scalars {
                    s.write_packed(&mut body);
                }
                put_len(out, number, &body);
            } else {
                for s in &scalars {
                    s.write(number, out);
                }
            }
            Ok(())
        }
        Conv::Map(kf, vf) => {
            let m = a
                .as_map_opt()
                .ok_or_else(|| internal("to_protobuf: expected a map column"))?;
            let entries = m.entries();
            let offsets = m.value_offsets();
            let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
            for i in start..end {
                let mut entry = Vec::new();
                // Both fields of an entry are set explicitly, so they are written even at their defaults.
                scalar(kf, &kf.conv, entries.column(0).as_ref(), i)?.write(1, &mut entry);
                let values = entries.column(1);
                let value = if values.is_null(i) {
                    // A NULL value writes the field's default (nothing, for proto3 scalars).
                    default_scalar(&vf.fd)
                } else {
                    scalar(vf, &vf.conv, values.as_ref(), i)?
                };
                value.write(2, &mut entry);
                put_len(out, number, &entry);
            }
            Ok(())
        }
        conv => {
            let s = scalar(f, conv, a, row)?;
            // proto3 plain scalars at their zero are not written (bitwise: -0.0 is written); everything with presence is.
            if f.fd.supports_presence() || !s.is_default() {
                s.write(number, out);
            }
            Ok(())
        }
    }
}

/// The values array and the range of `row` of a list column.
fn list_range(a: &dyn Array, row: usize) -> Result<(ArrayRef, usize, usize), KrishivError> {
    match a.data_type() {
        ArrowType::List(_) => {
            let l = a.as_list::<i32>();
            let o = l.value_offsets();
            Ok((Arc::clone(l.values()), o[row] as usize, o[row + 1] as usize))
        }
        ArrowType::LargeList(_) => {
            let l = a.as_list::<i64>();
            let o = l.value_offsets();
            Ok((Arc::clone(l.values()), o[row] as usize, o[row + 1] as usize))
        }
        _ => Err(internal("to_protobuf: expected a list column")),
    }
}

/// The Spark type of an Arrow struct argument.
///
/// # Errors
/// A type with no Spark equivalent.
pub fn spark_type_of(ty: &ArrowType, nullable: bool) -> Result<DataType, KrishivError> {
    DataType::from_arrow_field(&Field::new("value", ty.clone(), nullable))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, StringViewArray, StructArray};
    use krishiv_common::error::ErrorClass;
    use krishiv_types::{StructField, StructType};

    use super::*;

    const GOLDEN: &[u8] = include_bytes!(
        "../../../../tests/spark-golden/functions/from_to_protobuf/fixtures/krishiv_golden.desc"
    );
    const P2: &[u8] = include_bytes!(
        "../../../../tests/spark-golden/functions/from_to_protobuf/fixtures/krishiv_golden_p2.desc"
    );

    fn st(fields: &[(&str, DataType)]) -> StructType {
        StructType::new(
            fields
                .iter()
                .map(|(n, t)| StructField::new(*n, t.clone(), true)),
        )
    }

    fn conv(desc: &[u8], message: &str, s: &StructType) -> KrishivError {
        let set = DescriptorSet::parse(desc).unwrap();
        let msg = set.message(message).unwrap();
        struct_conv(s, &msg, &mut Vec::new()).err().unwrap()
    }

    #[test]
    fn a_struct_field_the_message_lacks_is_no_sql_type_in_protobuf_schema() {
        let e = conv(GOLDEN, "Scalars", &st(&[("nope", DataType::Integer)]));
        assert_eq!(e.error_class(), ErrorClass::NoSqlTypeInProtobufSchema);
    }

    #[test]
    fn a_missing_required_field_is_protobuf_field_missing_in_sql_schema() {
        let e = conv(P2, "P2Required", &st(&[("c", DataType::Integer)]));
        assert_eq!(e.error_class(), ErrorClass::ProtobufFieldMissingInSqlSchema);
    }

    #[test]
    fn two_fields_matching_a_name_ignoring_case_is_protobuf_field_missing() {
        let e = conv(P2, "CaseClash", &st(&[("abc", DataType::String)]));
        assert_eq!(e.error_class(), ErrorClass::ProtobufFieldMissing);
    }

    #[test]
    fn the_encoder_reports_the_root_conversion_failure() {
        let set = DescriptorSet::parse(GOLDEN).unwrap();
        let input = DataType::from(st(&[("nope", DataType::Integer)]));
        let e = Encoder::new(&set, "Scalars", &input).err().unwrap();
        assert_eq!(
            e.error_class(),
            ErrorClass::UnableToConvertToProtobufMessageType
        );
    }

    #[test]
    fn a_string_that_is_not_an_enum_name_cannot_convert_to_the_enum() {
        let set = DescriptorSet::parse(GOLDEN).unwrap();
        let input = DataType::from(st(&[("color", DataType::String)]));
        let enc = Encoder::new(&set, "Scalars", &input).unwrap();
        let ArrowType::Struct(fields) = input.to_arrow() else {
            panic!("not a struct");
        };
        let col: ArrayRef = Arc::new(StringViewArray::from(vec![Some("BLUE")]));
        let array: ArrayRef = Arc::new(StructArray::new(fields, vec![col], None));
        let e = enc.encode(&array).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::CannotConvertSqlValueToProtobufEnumType
        );
    }
}
