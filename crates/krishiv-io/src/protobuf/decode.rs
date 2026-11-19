// SPDX-License-Identifier: Apache-2.0
//! `from_protobuf`: a column of messages into one Arrow array of the Spark type of the message (see `schema`).

use std::sync::Arc;

use super::desc::{DefaultValue, FieldDescriptor, Kind};
use arrow_array::builder::{BinaryViewBuilder, StringViewBuilder};
use arrow_array::{
    ArrayRef, BooleanArray, Decimal128Array, DurationMicrosecondArray, Float32Array, Float64Array,
    Int32Array, Int64Array, ListArray, MapArray, StructArray, TimestampMicrosecondArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{ArrowError, DataType as ArrowType, Fields, TimeUnit};
use krishiv_common::error::{KrishivError, MalformedProtobufMessage, ParseModeUnsupported};
use krishiv_types::DataType;

use super::DescriptorSet;
use super::json;
use super::options::{ParseMode, ProtoOptions};
use super::schema::{self, FieldConv, Out, PlanField, Shape, StructPlan};
use super::wire::{Malformed, Parsed, Raw, Slot, collect};
use crate::error::unsupported;

/// A decoded value, before it is put in an Arrow array.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Cell {
    Null,
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Dec(i128),
    Str(String),
    Bin(Vec<u8>),
    Ts(i64),
    Dur(i64),
    Struct(Vec<Cell>),
    List(Vec<Cell>),
    Map(Vec<(Cell, Cell)>),
}

/// Why one message did not decode.
#[derive(Debug)]
pub(crate) enum Failure {
    /// The input is not a message of the type (a record-level failure, `mode` decides).
    Malformed,
    /// Something this implementation refuses, whatever the mode.
    Refused(KrishivError),
}

impl From<Malformed> for Failure {
    fn from(_: Malformed) -> Self {
        Self::Malformed
    }
}

/// Reads messages of one type (see the module documentation).
#[derive(Debug, Clone)]
pub struct Decoder {
    plan: StructPlan,
    spark: DataType,
    arrow: ArrowType,
    opts: ProtoOptions,
    set: DescriptorSet,
    /// Two fields of a message that differ only by case: Spark's field matching fails on the first message it reads.
    ambiguous: bool,
}

/// Whether any struct of `plan` has two proto fields whose names differ only by case.
fn has_case_clash(plan: &StructPlan) -> bool {
    let mut seen = std::collections::HashSet::new();
    let own = plan
        .msg
        .fields()
        .any(|f| !seen.insert(f.name().to_lowercase()));
    let nested = |c: &FieldConv| matches!(&c.out, Out::Struct(p) if has_case_clash(p));
    own || plan.fields.iter().any(|f| match &f.shape {
        Shape::Single(c) | Shape::List(c) => nested(c),
        Shape::Map(k, v, _) => nested(k) || nested(v),
    })
}

impl Decoder {
    /// A decoder of `message` of the descriptor set `set`.
    ///
    /// # Errors
    /// `PROTOBUF_MESSAGE_NOT_FOUND`, `RECURSIVE_PROTOBUF_SCHEMA`, and a type Arrow cannot carry.
    pub fn new<'a>(
        set: &DescriptorSet,
        message: &str,
        options: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, KrishivError> {
        let msg = set.message(message)?;
        let opts = ProtoOptions::parse(options)?;
        let (spark, plan) = schema::build(&msg, &opts)?;
        let arrow = spark.to_arrow();
        let ambiguous = has_case_clash(&plan);
        Ok(Self {
            ambiguous,
            plan,
            spark,
            arrow,
            opts,
            set: set.clone(),
        })
    }

    /// The Spark type of a decoded message (always nullable at the root).
    #[must_use]
    pub fn spark_type(&self) -> &DataType {
        &self.spark
    }

    /// The Arrow type [`Self::decode`] returns.
    #[must_use]
    pub fn arrow_type(&self) -> &ArrowType {
        &self.arrow
    }

    /// Decodes `rows` (NULL stays NULL). A malformed message raises `MALFORMED_PROTOBUF_MESSAGE` under `FAILFAST`, is NULL under
    /// `PERMISSIVE`, and raises `PARSE_MODE_UNSUPPORTED` under `DROPMALFORMED`.
    ///
    /// # Errors
    /// As above, and `KRISHIV_UNSUPPORTED_FEATURE` for what is refused.
    pub fn decode(&self, rows: &[Option<&[u8]>]) -> Result<ArrayRef, KrishivError> {
        let mut cells = Vec::with_capacity(rows.len());
        for row in rows {
            let Some(bytes) = row else {
                cells.push(Cell::Null);
                continue;
            };
            match self.message(bytes) {
                Ok(c) => cells.push(c),
                Err(Failure::Refused(e)) => return Err(e),
                Err(Failure::Malformed) => match self.opts.mode {
                    ParseMode::Permissive => cells.push(Cell::Null),
                    ParseMode::FailFast => {
                        return Err(MalformedProtobufMessage {
                            fail_fast_mode: "FAILFAST".to_owned(),
                        }
                        .into());
                    }
                    ParseMode::DropMalformed => {
                        return Err(ParseModeUnsupported {
                            func_name: "`from_protobuf`".to_owned(),
                            mode: "DROPMALFORMED".to_owned(),
                        }
                        .into());
                    }
                },
            }
        }
        build(&self.arrow, &cells).map_err(|e| unsupported(format!("from_protobuf result: {e}")))
    }

    fn message(&self, bytes: &[u8]) -> Result<Cell, Failure> {
        if self.ambiguous {
            return Err(Failure::Malformed);
        }
        let ctx = Ctx {
            opts: &self.opts,
            set: &self.set,
        };
        let parsed = collect(&self.plan.msg, bytes, 0)?;
        // Only the top-level message is checked for fields protobuf-java kept as unknown (Spark's `regularFieldNumbers`).
        if parsed.mismatch {
            return Err(Failure::Malformed);
        }
        convert_struct(&ctx, &self.plan, &parsed, 0)
    }
}

pub(crate) struct Ctx<'a> {
    pub(crate) opts: &'a ProtoOptions,
    pub(crate) set: &'a DescriptorSet,
}

fn convert_struct(
    ctx: &Ctx<'_>,
    plan: &StructPlan,
    parsed: &Parsed<'_>,
    depth: u32,
) -> Result<Cell, Failure> {
    if plan.dummy {
        return Ok(Cell::Struct(vec![Cell::Null]));
    }
    let mut out = Vec::with_capacity(plan.fields.len());
    for pf in &plan.fields {
        out.push(field_cell(ctx, pf, parsed, depth)?);
    }
    Ok(Cell::Struct(out))
}

/// The value of a field of a parsed message (Spark's `getFieldValue` and the writer of the field).
fn field_cell(
    ctx: &Ctx<'_>,
    pf: &PlanField,
    parsed: &Parsed<'_>,
    depth: u32,
) -> Result<Cell, Failure> {
    let slot = parsed.get(pf.fd.number());
    match &pf.shape {
        Shape::List(c) => {
            let mut items = Vec::new();
            if let Some(Slot::Many(raws)) = slot {
                for raw in raws {
                    let cell = value(ctx, c, *raw, depth)?;
                    items.push(cell);
                }
            }
            Ok(Cell::List(items))
        }
        Shape::Map(kc, vc, entry) => {
            let mut items = Vec::new();
            if let Some(Slot::Many(raws)) = slot {
                for raw in raws {
                    let Raw::Len(bytes) = raw else {
                        return Err(Failure::Malformed);
                    };
                    let e = collect(entry, bytes, depth + 1)?;
                    let k = entry_value(ctx, kc, &e, depth)?;
                    let v = entry_value(ctx, vc, &e, depth)?;
                    items.push((k, v));
                }
            }
            Ok(Cell::Map(items))
        }
        Shape::Single(c) => {
            let present = match slot {
                None => None,
                Some(Slot::One(raw)) => {
                    // A proto3 scalar without presence reads as absent when it is the type's zero.
                    if pf.fd.supports_presence() || !is_zero(&pf.fd, *raw) {
                        Some(value(ctx, c, *raw, depth)?)
                    } else {
                        None
                    }
                }
                Some(Slot::Many(raws)) => Some(merged_message(ctx, c, raws, depth)?),
            };
            if let Some(cell) = present {
                return Ok(cell);
            }
            if pf.fd.has_explicit_default() {
                return default_cell(ctx, c, depth, true);
            }
            if !pf.fd.supports_presence() && ctx.opts.emit_defaults {
                return default_cell(ctx, c, depth, false);
            }
            Ok(Cell::Null)
        }
    }
}

/// The key or value of a map entry: its value, or the type's default when the entry lacks it.
fn entry_value(
    ctx: &Ctx<'_>,
    c: &FieldConv,
    entry: &Parsed<'_>,
    depth: u32,
) -> Result<Cell, Failure> {
    match entry.get(c.fd.number()) {
        Some(Slot::One(raw)) => value(ctx, c, *raw, depth),
        Some(Slot::Many(raws)) => merged_message(ctx, c, raws, depth),
        None => match &c.out {
            Out::Struct(plan) => convert_struct(ctx, plan, &Parsed::default(), depth + 1),
            _ => default_cell(ctx, c, depth, false),
        },
    }
}

/// The pieces of a singular message field merge: their encodings concatenate.
fn merged_message(
    ctx: &Ctx<'_>,
    c: &FieldConv,
    raws: &[Raw<'_>],
    depth: u32,
) -> Result<Cell, Failure> {
    let mut joined: Vec<u8> = Vec::new();
    for raw in raws {
        let Raw::Len(b) = raw else {
            return Err(Failure::Malformed);
        };
        joined.extend_from_slice(b);
    }
    value(ctx, c, Raw::Len(&joined), depth)
}

/// Whether a scalar is its type's zero (what `hasField` is false for in proto3).
fn is_zero(fd: &FieldDescriptor, raw: Raw<'_>) -> bool {
    match (fd.kind(), raw) {
        (Kind::Int32 | Kind::Uint32 | Kind::Enum(_), Raw::Varint(v)) => v as u32 == 0,
        (Kind::Sint32, Raw::Varint(v)) => v as u32 == 0,
        (_, Raw::Varint(v)) => v == 0,
        (_, Raw::Fixed32(v)) => v == 0,
        (_, Raw::Fixed64(v)) => v == 0,
        (_, Raw::Len(b)) => b.is_empty(),
    }
}

/// The default of a field: the proto2 `[default = ..]` (`explicit`) or the type's zero.
fn default_cell(ctx: &Ctx<'_>, c: &FieldConv, depth: u32, explicit: bool) -> Result<Cell, Failure> {
    let owned = if explicit {
        c.fd.explicit_default()
    } else {
        Some(zero_default(&c.fd.kind()))
    };
    let raw = match &owned {
        Some(DefaultValue::Varint(v)) => Raw::Varint(*v),
        Some(DefaultValue::Fixed32(v)) => Raw::Fixed32(*v),
        Some(DefaultValue::Fixed64(v)) => Raw::Fixed64(*v),
        Some(DefaultValue::Bytes(b)) => Raw::Len(b),
        None => return Ok(Cell::Null),
    };
    value(ctx, c, raw, depth)
}

/// The type's zero: the first value of an enum, empty text, zero.
fn zero_default(kind: &Kind) -> DefaultValue {
    match kind {
        Kind::Float | Kind::Fixed32 | Kind::Sfixed32 => DefaultValue::Fixed32(0),
        Kind::Double | Kind::Fixed64 | Kind::Sfixed64 => DefaultValue::Fixed64(0),
        Kind::String | Kind::Bytes | Kind::Message(_) => DefaultValue::Bytes(Vec::new()),
        Kind::Enum(e) => DefaultValue::Varint(
            e.values()
                .first()
                .map_or(0, |v| i64::from(v.number()) as u64),
        ),
        _ => DefaultValue::Varint(0),
    }
}

fn zigzag32(v: u64) -> i32 {
    let n = v as u32;
    ((n >> 1) as i32) ^ -((n & 1) as i32)
}

fn zigzag64(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// One wire value as a Spark value.
fn value(ctx: &Ctx<'_>, c: &FieldConv, raw: Raw<'_>, depth: u32) -> Result<Cell, Failure> {
    match &c.out {
        Out::Plain => scalar(c, raw),
        Out::UpU32 => match raw {
            Raw::Varint(v) => Ok(Cell::I64(i64::from(v as u32))),
            _ => Err(Failure::Malformed),
        },
        Out::UpU64 => match raw {
            Raw::Varint(v) => Ok(Cell::Dec(i128::from(v))),
            _ => Err(Failure::Malformed),
        },
        Out::EnumInt => match raw {
            Raw::Varint(v) => Ok(Cell::I32(v as i32)),
            _ => Err(Failure::Malformed),
        },
        Out::Timestamp | Out::Duration => {
            let Kind::Message(m) = c.fd.kind() else {
                return Err(Failure::Malformed);
            };
            let Raw::Len(bytes) = raw else {
                return Err(Failure::Malformed);
            };
            let p = collect(&m, bytes, depth + 1)?;
            let mut it = m.fields();
            let (seconds, nanos) = match (it.next(), it.next()) {
                (Some(s), Some(n)) => (
                    varint_of(&p, s.number()).unwrap_or(0) as i64,
                    varint_of(&p, n.number()).unwrap_or(0) as i32,
                ),
                _ => return Err(Failure::Malformed),
            };
            // `millisToMicros(seconds * 1000)`: the product wraps, the conversion to microseconds is checked.
            let micros = seconds
                .wrapping_mul(1000)
                .checked_mul(1000)
                .ok_or(Failure::Malformed)?
                .wrapping_add(i64::from(nanos) / 1000);
            Ok(if matches!(c.out, Out::Timestamp) {
                Cell::Ts(micros)
            } else {
                Cell::Dur(micros)
            })
        }
        Out::AnyJson => {
            let Raw::Len(bytes) = raw else {
                return Err(Failure::Malformed);
            };
            json::any_to_json(ctx.set, bytes, ctx.opts.enums_as_ints, depth + 1).map(Cell::Str)
        }
        Out::Unwrap(inner) => {
            let Kind::Message(m) = c.fd.kind() else {
                return Err(Failure::Malformed);
            };
            let Raw::Len(bytes) = raw else {
                return Err(Failure::Malformed);
            };
            let p = collect(&m, bytes, depth + 1)?;
            match p.get(inner.fd.number()) {
                Some(Slot::One(r)) if !is_zero(&inner.fd, *r) => value(ctx, inner, *r, depth),
                // The wrapper's `value` has no presence: a zero reads as absent, unless defaults are emitted.
                Some(Slot::One(r)) if ctx.opts.emit_defaults => value(ctx, inner, *r, depth),
                None if ctx.opts.emit_defaults => default_cell(ctx, inner, depth, false),
                _ => Ok(Cell::Null),
            }
        }
        Out::Struct(plan) => {
            let Raw::Len(bytes) = raw else {
                return Err(Failure::Malformed);
            };
            let p = collect(&plan.msg, bytes, depth + 1)?;
            convert_struct(ctx, plan, &p, depth + 1)
        }
    }
}

fn varint_of(p: &Parsed<'_>, number: u32) -> Option<u64> {
    match p.get(number) {
        Some(Slot::One(Raw::Varint(v))) => Some(*v),
        _ => None,
    }
}

/// A scalar of the kind of the field, as Spark reads it.
fn scalar(c: &FieldConv, raw: Raw<'_>) -> Result<Cell, Failure> {
    let cell = match (c.fd.kind(), raw) {
        (Kind::Int32 | Kind::Uint32, Raw::Varint(v)) => Cell::I32(v as i32),
        (Kind::Sint32, Raw::Varint(v)) => Cell::I32(zigzag32(v)),
        (Kind::Fixed32 | Kind::Sfixed32, Raw::Fixed32(v)) => Cell::I32(v as i32),
        (Kind::Int64 | Kind::Uint64, Raw::Varint(v)) => Cell::I64(v as i64),
        (Kind::Sint64, Raw::Varint(v)) => Cell::I64(zigzag64(v)),
        (Kind::Fixed64 | Kind::Sfixed64, Raw::Fixed64(v)) => Cell::I64(v as i64),
        (Kind::Float, Raw::Fixed32(v)) => Cell::F32(f32::from_bits(v)),
        (Kind::Double, Raw::Fixed64(v)) => Cell::F64(f64::from_bits(v)),
        (Kind::Bool, Raw::Varint(v)) => Cell::Bool(v != 0),
        (Kind::String, Raw::Len(b)) => Cell::Str(String::from_utf8_lossy(b).into_owned()),
        (Kind::Bytes, Raw::Len(b)) => Cell::Bin(b.to_vec()),
        (Kind::Enum(e), Raw::Varint(v)) => {
            let n = v as i32;
            Cell::Str(e.get_value(n).map_or_else(
                // protobuf-java's `findValueByNumberCreatingIfUnknown`.
                || format!("UNKNOWN_ENUM_VALUE_{}_{n}", e.name()),
                |ev| ev.name().to_owned(),
            ))
        }
        _ => return Err(Failure::Malformed),
    };
    Ok(cell)
}

// ---- Arrow ----

/// The Arrow array of type `ty` holding `cells`.
fn build(ty: &ArrowType, cells: &[Cell]) -> Result<ArrayRef, ArrowError> {
    let nulls = || -> Option<NullBuffer> {
        let validity: Vec<bool> = cells.iter().map(|c| !matches!(c, Cell::Null)).collect();
        validity
            .iter()
            .any(|v| !v)
            .then(|| NullBuffer::from(validity))
    };
    Ok(match ty {
        ArrowType::Boolean => Arc::new(BooleanArray::from_iter(cells.iter().map(|c| match c {
            Cell::Bool(b) => Some(*b),
            _ => None,
        }))),
        ArrowType::Int32 => Arc::new(Int32Array::from_iter(cells.iter().map(|c| match c {
            Cell::I32(v) => Some(*v),
            _ => None,
        }))),
        ArrowType::Int64 => Arc::new(Int64Array::from_iter(cells.iter().map(|c| match c {
            Cell::I64(v) => Some(*v),
            _ => None,
        }))),
        ArrowType::Float32 => Arc::new(Float32Array::from_iter(cells.iter().map(|c| match c {
            Cell::F32(v) => Some(*v),
            _ => None,
        }))),
        ArrowType::Float64 => Arc::new(Float64Array::from_iter(cells.iter().map(|c| match c {
            Cell::F64(v) => Some(*v),
            _ => None,
        }))),
        ArrowType::Decimal128(p, s) => Arc::new(
            Decimal128Array::from_iter(cells.iter().map(|c| match c {
                Cell::Dec(v) => Some(*v),
                _ => None,
            }))
            .with_precision_and_scale(*p, *s)?,
        ),
        ArrowType::Utf8View => {
            let mut b = StringViewBuilder::new();
            for c in cells {
                match c {
                    Cell::Str(s) => b.append_value(s),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ArrowType::BinaryView => {
            let mut b = BinaryViewBuilder::new();
            for c in cells {
                match c {
                    Cell::Bin(v) => b.append_value(v),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ArrowType::Timestamp(TimeUnit::Microsecond, tz) => Arc::new(
            TimestampMicrosecondArray::from_iter(cells.iter().map(|c| match c {
                Cell::Ts(v) => Some(*v),
                _ => None,
            }))
            .with_timezone_opt(tz.clone()),
        ),
        ArrowType::Duration(TimeUnit::Microsecond) => Arc::new(
            DurationMicrosecondArray::from_iter(cells.iter().map(|c| match c {
                Cell::Dur(v) => Some(*v),
                _ => None,
            })),
        ),
        ArrowType::Struct(fields) => struct_array(fields, cells, nulls())?,
        ArrowType::List(field) => {
            let mut offsets = vec![0i32];
            let mut flat: Vec<Cell> = Vec::new();
            for c in cells {
                if let Cell::List(items) = c {
                    flat.extend(items.iter().cloned());
                }
                offsets.push(
                    i32::try_from(flat.len())
                        .map_err(|_| ArrowError::OffsetOverflowError(flat.len()))?,
                );
            }
            let values = build(field.data_type(), &flat)?;
            Arc::new(ListArray::try_new(
                Arc::clone(field),
                OffsetBuffer::new(offsets.into()),
                values,
                nulls(),
            )?)
        }
        ArrowType::Map(entries, sorted) => {
            let ArrowType::Struct(kv) = entries.data_type() else {
                return Err(ArrowError::SchemaError(
                    "a map without an entries struct".to_owned(),
                ));
            };
            let mut offsets = vec![0i32];
            let mut keys: Vec<Cell> = Vec::new();
            let mut vals: Vec<Cell> = Vec::new();
            for c in cells {
                if let Cell::Map(items) = c {
                    for (k, v) in items {
                        keys.push(k.clone());
                        vals.push(v.clone());
                    }
                }
                offsets.push(
                    i32::try_from(keys.len())
                        .map_err(|_| ArrowError::OffsetOverflowError(keys.len()))?,
                );
            }
            let (Some(kf), Some(vf)) = (kv.first(), kv.get(1)) else {
                return Err(ArrowError::SchemaError(
                    "a map entry without key and value".to_owned(),
                ));
            };
            let columns = vec![build(kf.data_type(), &keys)?, build(vf.data_type(), &vals)?];
            let entries_array = StructArray::try_new(kv.clone(), columns, None)?;
            Arc::new(MapArray::try_new(
                Arc::clone(entries),
                OffsetBuffer::new(offsets.into()),
                entries_array,
                nulls(),
                *sorted,
            )?)
        }
        other => {
            return Err(ArrowError::SchemaError(format!(
                "no Arrow builder for {other}"
            )));
        }
    })
}

fn struct_array(
    fields: &Fields,
    cells: &[Cell],
    nulls: Option<NullBuffer>,
) -> Result<ArrayRef, ArrowError> {
    if fields.is_empty() {
        return Ok(Arc::new(StructArray::new_empty_fields(cells.len(), nulls)));
    }
    let mut columns = Vec::with_capacity(fields.len());
    for (j, f) in fields.iter().enumerate() {
        let column: Vec<Cell> = cells
            .iter()
            .map(|c| match c {
                Cell::Struct(v) => v.get(j).cloned().unwrap_or(Cell::Null),
                _ => Cell::Null,
            })
            .collect();
        columns.push(build(f.data_type(), &column)?);
    }
    Ok(Arc::new(StructArray::try_new(
        fields.clone(),
        columns,
        nulls,
    )?))
}
