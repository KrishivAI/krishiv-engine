// SPDX-License-Identifier: Apache-2.0
//! The Spark type of a protobuf message and the plan that reads a message into it (Spark's `SchemaConverters.toSqlType`).
//!
//! A message is a struct of ALL its fields in declaration order: proto3 scalars are nullable, repeated fields are
//! `array<T NOT NULL>`, maps `map<K, V NOT NULL>`, a proto2 `required` field is `NOT NULL`. `Timestamp` and `Duration` (by message
//! name and the names of their two fields) are `TIMESTAMP` and `INTERVAL DAY TO SECOND`; `Any` is `struct<type_url, value>` or, with
//! `convert.any.fields.to.json`, a JSON string; the primitive wrappers unwrap with `unwrap.primitive.wrapper.types`; a message without
//! fields is dropped from its parent (or kept as `struct<__dummy_field_in_empty_struct: string>` with `retain.empty.message.types`);
//! a recursive message is an error unless `recursive.fields.max.depth` allows it, and deeper levels are cut.

use std::collections::HashMap;

use super::desc::{FieldDescriptor, Kind, MessageDescriptor};
use krishiv_common::error::{CannotParseProtobufDescriptor, KrishivError, RecursiveProtobufSchema};
use krishiv_types::{ArrayType, DataType, DecimalType, MapType, StructField, StructType};

use super::options::ProtoOptions;
use crate::error::unsupported;

/// The most levels of message structs a schema may have, the root's included: past it the schema is refused
/// (`KRISHIV_UNSUPPORTED_FEATURE`) instead of recursing on (a chain of distinct types is not bounded by the recursion depth
/// option). Spark has no cap of its own (a deep enough chain is a `StackOverflowError` there); 100 is protobuf-java's recursion
/// limit for the messages themselves, so no value nested deeper could be read anyway.
pub(crate) const MAX_NESTING: usize = 100;

/// The most message structs one schema may expand to, counted over every field path (a descriptor of 30 types whose fields
/// each refer twice to the next would otherwise expand to about 2^30 leaves). Ordinary schemas are far below it.
const MAX_EXPANDED_MESSAGES: usize = 100_000;

std::thread_local! {
    /// Message structs expanded so far by the schema derivation running on this thread.
    static EXPANDED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How a value of a field becomes a Spark value.
#[derive(Debug, Clone)]
pub(crate) enum Out {
    /// The kind's own type (`int`, `bigint`, `string` for an enum name, ...).
    Plain,
    /// `uint32` as `bigint`.
    UpU32,
    /// `uint64` as `decimal(20,0)`.
    UpU64,
    /// An enum as its number.
    EnumInt,
    /// `google.protobuf.Timestamp`.
    Timestamp,
    /// `google.protobuf.Duration`.
    Duration,
    /// `google.protobuf.Any` as JSON text.
    AnyJson,
    /// A primitive wrapper as its `value`.
    Unwrap(Box<FieldConv>),
    /// A message as a struct.
    Struct(Box<StructPlan>),
}

/// The conversion of one value: the field it comes from and what it becomes.
#[derive(Debug, Clone)]
pub(crate) struct FieldConv {
    pub(crate) fd: FieldDescriptor,
    pub(crate) out: Out,
}

/// How a field's values are shaped.
#[derive(Debug, Clone)]
pub(crate) enum Shape {
    /// One value, or none.
    Single(FieldConv),
    /// `repeated`.
    List(FieldConv),
    /// A map: key, value and the entry message.
    Map(FieldConv, FieldConv, MessageDescriptor),
}

/// One field of a struct plan.
#[derive(Debug, Clone)]
pub(crate) struct PlanField {
    pub(crate) fd: FieldDescriptor,
    pub(crate) shape: Shape,
}

/// The fields of a message that make its struct.
#[derive(Debug, Clone)]
pub(crate) struct StructPlan {
    pub(crate) msg: MessageDescriptor,
    pub(crate) fields: Vec<PlanField>,
    /// A message without fields kept as `struct<__dummy_field_in_empty_struct: string>`.
    pub(crate) dummy: bool,
}

/// The root plan and Spark type of `msg`.
pub(crate) fn build(
    msg: &MessageDescriptor,
    opts: &ProtoOptions,
) -> Result<(DataType, StructPlan), KrishivError> {
    let existing = HashMap::from([(msg.full_name().to_owned(), 1)]);
    EXPANDED.with(|c| c.set(0));
    let (fields, plan) = struct_of(msg, &existing, opts, 0)?;
    Ok((struct_type(&fields, plan.dummy), plan))
}

fn struct_type(fields: &[StructField], dummy: bool) -> DataType {
    if dummy {
        return dummy_struct();
    }
    DataType::from(StructType::new(fields.to_vec()))
}

fn dummy_struct() -> DataType {
    DataType::from(StructType::new([StructField::new(
        "__dummy_field_in_empty_struct",
        DataType::String,
        true,
    )]))
}

/// The struct of `msg` at nesting `level` (the root is 0).
fn struct_of(
    msg: &MessageDescriptor,
    existing: &HashMap<String, i32>,
    opts: &ProtoOptions,
    level: usize,
) -> Result<(Vec<StructField>, StructPlan), KrishivError> {
    let expanded = EXPANDED.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if expanded > MAX_EXPANDED_MESSAGES {
        return Err(unsupported(format!(
            "a protobuf schema that expands to more than {MAX_EXPANDED_MESSAGES} messages (at {})",
            msg.full_name()
        )));
    }
    let mut fields = Vec::new();
    let mut plans = Vec::new();
    for fd in msg.fields() {
        if let Some((f, p)) = field(&fd, existing, opts, level)? {
            fields.push(f);
            plans.push(p);
        }
    }
    let dummy = fields.is_empty() && opts.retain_empty;
    Ok((
        fields,
        StructPlan {
            msg: msg.clone(),
            fields: plans,
            dummy,
        },
    ))
}

/// A field: its struct field and plan, `None` when it is dropped.
fn field(
    fd: &FieldDescriptor,
    existing: &HashMap<String, i32>,
    opts: &ProtoOptions,
    level: usize,
) -> Result<Option<(StructField, PlanField)>, KrishivError> {
    if fd.is_map() {
        let Kind::Message(entry) = fd.kind() else {
            return Ok(None);
        };
        let Some((kf, vf)) = entry.map_entry_fields() else {
            return Err(CannotParseProtobufDescriptor {}.into());
        };
        let Some((kt, kc)) = element(&kf, existing, opts, level)? else {
            return Ok(None);
        };
        let Some((vt, vc)) = element(&vf, existing, opts, level)? else {
            return Ok(None);
        };
        let dt = DataType::from(MapType::new(kt, vt, false));
        return Ok(Some((
            StructField::new(fd.name(), dt, true),
            PlanField {
                fd: fd.clone(),
                shape: Shape::Map(kc, vc, entry),
            },
        )));
    }
    let Some((dt, conv)) = element(fd, existing, opts, level)? else {
        return Ok(None);
    };
    if fd.is_list() {
        // Spark declares `array<T NOT NULL>` here (its goldens, 3.5.7 to 4.3.0). An unwrapped wrapper element can still
        // be NULL (an empty wrapper message), so those arrays say `containsNull = true` rather than lose the element
        // (known nullability divergence, recorded in the golden README).
        let contains_null = matches!(conv.out, Out::Unwrap(_));
        return Ok(Some((
            StructField::new(
                fd.name(),
                DataType::from(ArrayType::new(dt, contains_null)),
                true,
            ),
            PlanField {
                fd: fd.clone(),
                shape: Shape::List(conv),
            },
        )));
    }
    Ok(Some((
        StructField::new(fd.name(), dt, !fd.is_required()),
        PlanField {
            fd: fd.clone(),
            shape: Shape::Single(conv),
        },
    )))
}

const WRAPPERS: &[&str] = &[
    "google.protobuf.BoolValue",
    "google.protobuf.Int32Value",
    "google.protobuf.UInt32Value",
    "google.protobuf.Int64Value",
    "google.protobuf.UInt64Value",
    "google.protobuf.StringValue",
    "google.protobuf.BytesValue",
    "google.protobuf.FloatValue",
    "google.protobuf.DoubleValue",
];

fn conv(fd: &FieldDescriptor, out: Out) -> FieldConv {
    FieldConv {
        fd: fd.clone(),
        out,
    }
}

fn pair_named(m: &MessageDescriptor, name: &str) -> bool {
    let mut fields = m.fields();
    m.name() == name
        && m.fields().len() == 2
        && fields.next().is_some_and(|f| f.name() == "seconds")
        && fields.next().is_some_and(|f| f.name() == "nanos")
}

/// The type and conversion of one value of `fd` (an element when `fd` is repeated); `None` when it is dropped.
fn element(
    fd: &FieldDescriptor,
    existing: &HashMap<String, i32>,
    opts: &ProtoOptions,
    level: usize,
) -> Result<Option<(DataType, FieldConv)>, KrishivError> {
    let plain = |t: DataType| Ok(Some((t, conv(fd, Out::Plain))));
    match fd.kind() {
        Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 | Kind::Fixed32 => plain(DataType::Integer),
        Kind::Uint32 if opts.upcast_unsigned => Ok(Some((DataType::Long, conv(fd, Out::UpU32)))),
        Kind::Uint32 => plain(DataType::Integer),
        Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 | Kind::Fixed64 => plain(DataType::Long),
        Kind::Uint64 if opts.upcast_unsigned => Ok(Some((long_decimal()?, conv(fd, Out::UpU64)))),
        Kind::Uint64 => plain(DataType::Long),
        Kind::Float => plain(DataType::Float),
        Kind::Double => plain(DataType::Double),
        Kind::Bool => plain(DataType::Boolean),
        Kind::String => plain(DataType::String),
        Kind::Bytes => plain(DataType::Binary),
        Kind::Enum(_) if opts.enums_as_ints => {
            Ok(Some((DataType::Integer, conv(fd, Out::EnumInt))))
        }
        Kind::Enum(_) => plain(DataType::String),
        Kind::Message(m) => message(fd, &m, existing, opts, level),
    }
}

fn long_decimal() -> Result<DataType, KrishivError> {
    Ok(DataType::Decimal(DecimalType::try_new(20, 0)?))
}

fn message(
    fd: &FieldDescriptor,
    m: &MessageDescriptor,
    existing: &HashMap<String, i32>,
    opts: &ProtoOptions,
    level: usize,
) -> Result<Option<(DataType, FieldConv)>, KrishivError> {
    if pair_named(m, "Duration") {
        return Ok(Some((
            DataType::DEFAULT_DAY_TIME_INTERVAL,
            conv(fd, Out::Duration),
        )));
    }
    if pair_named(m, "Timestamp") {
        return Ok(Some((DataType::Timestamp, conv(fd, Out::Timestamp))));
    }
    if opts.any_to_json && m.full_name() == "google.protobuf.Any" {
        return Ok(Some((DataType::String, conv(fd, Out::AnyJson))));
    }
    if opts.unwrap_wrappers
        && WRAPPERS.contains(&m.full_name())
        && let Some(value) = m.get_field(1)
    {
        let (dt, out) = match m.full_name() {
            "google.protobuf.BoolValue" => (DataType::Boolean, Out::Plain),
            "google.protobuf.Int32Value" => (DataType::Integer, Out::Plain),
            "google.protobuf.UInt32Value" if opts.upcast_unsigned => (DataType::Long, Out::UpU32),
            "google.protobuf.UInt32Value" => (DataType::Integer, Out::Plain),
            "google.protobuf.Int64Value" => (DataType::Long, Out::Plain),
            "google.protobuf.UInt64Value" if opts.upcast_unsigned => (long_decimal()?, Out::UpU64),
            "google.protobuf.UInt64Value" => (DataType::Long, Out::Plain),
            "google.protobuf.StringValue" => (DataType::String, Out::Plain),
            "google.protobuf.BytesValue" => (DataType::Binary, Out::Plain),
            "google.protobuf.FloatValue" => (DataType::Float, Out::Plain),
            _ => (DataType::Double, Out::Plain),
        };
        return Ok(Some((
            dt,
            conv(fd, Out::Unwrap(Box::new(conv(&value, out)))),
        )));
    }
    let name = m.full_name();
    let depth = existing.get(name).copied().unwrap_or(0);
    let max = opts.recursive_depth;
    if existing.contains_key(name) && (max <= 0 || max > 10) {
        return Err(RecursiveProtobufSchema {
            field_descriptor: fd.full_name(),
        }
        .into());
    }
    if existing.contains_key(name) && depth >= max {
        return Ok(None);
    }
    if level + 1 >= MAX_NESTING {
        return Err(unsupported(format!(
            "a protobuf message nested more than {MAX_NESTING} levels deep (field {})",
            fd.full_name()
        )));
    }
    let mut next = existing.clone();
    next.insert(name.to_owned(), depth + 1);
    let (fields, plan) = struct_of(m, &next, opts, level + 1)?;
    if fields.is_empty() && !opts.retain_empty {
        return Ok(None);
    }
    let dt = struct_type(&fields, plan.dummy);
    Ok(Some((dt, conv(fd, Out::Struct(Box::new(plan))))))
}
