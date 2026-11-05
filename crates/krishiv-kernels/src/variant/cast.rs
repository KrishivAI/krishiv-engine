// SPDX-License-Identifier: Apache-2.0
//! `VariantGet.cast`: a variant value into a Spark type.
//!
//! A variant null is SQL `NULL`. A UUID casts to a string only. An object
//! or array casts to a string as its JSON, to an array, map (string keys)
//! or struct element by element (a struct takes the object's fields by
//! exact name; absent ones are `NULL`), and to nothing else. A scalar is
//! read as its own type (integers as `BIGINT`, a decimal with trailing zeros
//! stripped) and cast with Spark's `Cast` in `TRY` mode when
//! `Cast.canAnsiCast` allows the pair; `BIGINT`/`DECIMAL` to `TIMESTAMP`
//! multiply by 10^6 exactly. Anything else, and any cast that gives
//! `NULL`, is invalid: `INVALID_VARIANT_CAST` (first failing row) when
//! `fail_on_error`, else `NULL`.
//!
//! Rows are grouped by their scalar type, so each group is one kernel cast.

use std::sync::Arc;

use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Float32Builder, Float64Builder, Int32Builder, Int64Builder,
    StringBuilder, StringViewBuilder,
};
use arrow_array::cast::AsArray as _;
use arrow_array::types::{Decimal128Type, TimestampMicrosecondType};
use arrow_array::{
    Array, ArrayRef, ListArray, MapArray, PrimitiveArray, StructArray, new_null_array,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::{ErrorClass, InvalidVariantCast, KrishivError};
use krishiv_types::DataType;
use krishiv_types::variant::{
    VariantKind, VariantRef, VariantWriter, digits, strip_trailing_zeros,
};

use super::{VariantColumn, VariantColumnBuilder, internal, unknown_type, uuid_string, write_json};
use crate::cast::{CastMode, CastOptions, TimeZoneRules, cast};

/// Microseconds per second.
const MICROS: i128 = 1_000_000;

/// Casts every row of `col` (a variant column) to `to`.
///
/// # Errors
/// `INVALID_VARIANT_CAST` for the first row that does not cast when
/// `fail_on_error`; `MALFORMED_VARIANT` for bad bytes.
pub fn cast_from_variant(
    col: &dyn Array,
    to: &DataType,
    fail_on_error: bool,
    zone: &dyn TimeZoneRules,
) -> Result<ArrayRef, KrishivError> {
    let col = VariantColumn::new(col)?;
    let values = (0..col.len())
        .map(|i| col.get(i))
        .collect::<Result<Vec<_>, _>>()?;
    cast_values(&values, to, fail_on_error, zone)
}

/// The shared state of one cast.
struct Ctx<'z> {
    fail: bool,
    zone: &'z dyn TimeZoneRules,
}

impl Ctx<'_> {
    /// An invalid cast of `v` to `to`: the error, or `Ok` for `NULL`.
    fn invalid(&self, v: &VariantRef<'_>, to: &DataType) -> Result<(), KrishivError> {
        if !self.fail {
            return Ok(());
        }
        let mut value = String::new();
        write_json(v, self.zone, &mut value)?;
        Err(InvalidVariantCast {
            value,
            data_type: format!("\"{}\"", to.sql()),
        }
        .into())
    }
}

/// Casts `values` (`None` is SQL `NULL`) to `to`, as an array of
/// `to.to_arrow()`.
///
/// # Errors
/// As [`cast_from_variant`].
pub fn cast_values(
    values: &[Option<VariantRef<'_>>],
    to: &DataType,
    fail_on_error: bool,
    zone: &dyn TimeZoneRules,
) -> Result<ArrayRef, KrishivError> {
    let ctx = Ctx {
        fail: fail_on_error,
        zone,
    };
    match cast_to(&ctx, values, to) {
        // The batch cast groups rows and casts containers by field: the
        // error Spark raises is the first one in its row-by-row walk.
        Err(e) if fail_on_error && e.error_class() == ErrorClass::InvalidVariantCast => {
            for v in values.iter().flatten() {
                check_one(&ctx, v, to)?;
            }
            Err(e)
        }
        other => other,
    }
}

/// `VariantGet.cast` of one value, in Spark's order (array elements in
/// order, object fields in stored order), for the error it raises first.
fn check_one(ctx: &Ctx<'_>, v: &VariantRef<'_>, to: &DataType) -> Result<(), KrishivError> {
    if *to == DataType::Variant || v.kind() == VariantKind::Null {
        return Ok(());
    }
    match (to, v.kind()) {
        (DataType::Array(a), VariantKind::Array) => {
            for i in 0..v.len()? {
                check_one(ctx, &v.get_index(i)?, a.element_type())?;
            }
            Ok(())
        }
        (DataType::Map(m), VariantKind::Object) if is_string(m.key_type()) => {
            for i in 0..v.len()? {
                check_one(ctx, &v.field_at(i)?, m.value_type())?;
            }
            Ok(())
        }
        (DataType::Struct(s), VariantKind::Object) => {
            for i in 0..v.len()? {
                let key = v.field_key_at(i)?;
                if let Some(f) = s.fields().iter().find(|f| f.name() == key) {
                    check_one(ctx, &v.field_at(i)?, f.data_type())?;
                }
            }
            Ok(())
        }
        (DataType::Array(_) | DataType::Map(_) | DataType::Struct(_), _) => ctx.invalid(v, to),
        _ => cast_atomic(ctx, &[Some(*v)], to).map(|_| ()),
    }
}

fn cast_to(
    ctx: &Ctx<'_>,
    values: &[Option<VariantRef<'_>>],
    to: &DataType,
) -> Result<ArrayRef, KrishivError> {
    if *to == DataType::Variant {
        // A fresh value without the keys it does not use.
        let mut out = VariantColumnBuilder::with_capacity(values.len());
        for v in values {
            match v {
                Some(v) => {
                    let mut w = VariantWriter::new();
                    w.append_variant(v)?;
                    let (m, value) = w.finish()?;
                    out.append(&m, &value);
                }
                None => out.append_null(),
            }
        }
        return out.finish();
    }
    // A variant null is SQL NULL for every target.
    let values: Vec<Option<VariantRef<'_>>> = values
        .iter()
        .map(|v| v.filter(|v| v.kind() != VariantKind::Null))
        .collect();
    match to {
        DataType::Array(a) => {
            let mut offsets = vec![0_i32];
            let mut valid = Vec::with_capacity(values.len());
            let mut elements = vec![];
            for v in &values {
                match v {
                    Some(v) if v.kind() == VariantKind::Array => {
                        for i in 0..v.len()? {
                            elements.push(Some(v.get_index(i)?));
                        }
                        valid.push(true);
                    }
                    Some(v) => {
                        ctx.invalid(v, to)?;
                        valid.push(false);
                    }
                    None => valid.push(false),
                }
                offsets.push(i32::try_from(elements.len()).map_err(|_| too_big())?);
            }
            let child = cast_to(ctx, &elements, a.element_type())?;
            let ArrowType::List(field) = to.to_arrow() else {
                return Err(internal("an array type without a list carrier"));
            };
            ListArray::try_new(
                field,
                OffsetBuffer::new(offsets.into()),
                child,
                nulls(valid),
            )
            .map(|a| Arc::new(a) as ArrayRef)
            .map_err(|e| internal(&e.to_string()))
        }
        DataType::Map(m) if is_string(m.key_type()) => {
            let mut offsets = vec![0_i32];
            let mut valid = Vec::with_capacity(values.len());
            let mut keys = StringViewBuilder::new();
            let mut elements = vec![];
            for v in &values {
                match v {
                    Some(v) if v.kind() == VariantKind::Object => {
                        for i in 0..v.len()? {
                            keys.append_value(v.field_key_at(i)?);
                            elements.push(Some(v.field_at(i)?));
                        }
                        valid.push(true);
                    }
                    Some(v) => {
                        ctx.invalid(v, to)?;
                        valid.push(false);
                    }
                    None => valid.push(false),
                }
                offsets.push(i32::try_from(elements.len()).map_err(|_| too_big())?);
            }
            let child = cast_to(ctx, &elements, m.value_type())?;
            let ArrowType::Map(entries, sorted) = to.to_arrow() else {
                return Err(internal("a map type without a map carrier"));
            };
            let ArrowType::Struct(kv) = entries.data_type() else {
                return Err(internal("map entries that are not a struct"));
            };
            let keys: ArrayRef = Arc::new(keys.finish());
            let entry_struct = StructArray::try_new(kv.clone(), vec![keys, child], None)
                .map_err(|e| internal(&e.to_string()))?;
            MapArray::try_new(
                entries,
                OffsetBuffer::new(offsets.into()),
                entry_struct,
                nulls(valid),
                sorted,
            )
            .map(|a| Arc::new(a) as ArrayRef)
            .map_err(|e| internal(&e.to_string()))
        }
        DataType::Struct(s) => {
            let mut valid = Vec::with_capacity(values.len());
            let mut columns: Vec<Vec<Option<VariantRef<'_>>>> =
                vec![Vec::with_capacity(values.len()); s.fields().len()];
            for v in &values {
                let object = match v {
                    Some(v) if v.kind() == VariantKind::Object => Some(v),
                    Some(v) => {
                        ctx.invalid(v, to)?;
                        None
                    }
                    None => None,
                };
                valid.push(object.is_some());
                for (f, column) in s.fields().iter().zip(columns.iter_mut()) {
                    column.push(match object {
                        Some(o) => o.get_field(f.name())?,
                        None => None,
                    });
                }
            }
            let ArrowType::Struct(fields) = to.to_arrow() else {
                return Err(internal("a struct type without a struct carrier"));
            };
            let children = s
                .fields()
                .iter()
                .zip(&columns)
                .map(|(f, c)| cast_to(ctx, c, f.data_type()))
                .collect::<Result<Vec<_>, _>>()?;
            StructArray::try_new(fields, children, nulls(valid))
                .map(|a| Arc::new(a) as ArrayRef)
                .map_err(|e| internal(&e.to_string()))
        }
        DataType::Null
        | DataType::Map(_)
        | DataType::Geometry { .. }
        | DataType::Geography { .. }
        | DataType::YearMonthInterval { .. }
        | DataType::DayTimeInterval { .. }
        | DataType::CalendarInterval => {
            Err(internal(&format!("a cast of VARIANT to {}", to.sql())))
        }
        atomic => cast_atomic(ctx, &values, atomic),
    }
}

fn too_big() -> KrishivError {
    internal("a variant cast result over 2^31 elements")
}

fn nulls(valid: Vec<bool>) -> Option<NullBuffer> {
    valid.contains(&false).then(|| NullBuffer::from(valid))
}

fn is_string(t: &DataType) -> bool {
    matches!(t, DataType::String | DataType::CollatedString(_))
}

/// The scalar type a variant value is read as, for the cast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Source {
    Boolean,
    Long,
    Double,
    Float,
    Decimal(u8, u8),
    Date,
    Timestamp,
    TimestampNtz,
    Binary,
    String,
}

impl Source {
    fn spark(self) -> DataType {
        match self {
            Self::Boolean => DataType::Boolean,
            Self::Long => DataType::Long,
            Self::Double => DataType::Double,
            Self::Float => DataType::Float,
            Self::Decimal(p, s) => krishiv_types::DecimalType::try_new(u32::from(p), u32::from(s))
                .map_or(DataType::Long, DataType::Decimal),
            Self::Date => DataType::Date,
            Self::Timestamp => DataType::Timestamp,
            Self::TimestampNtz => DataType::TimestampNtz,
            Self::Binary => DataType::Binary,
            Self::String => DataType::String,
        }
    }
}

fn is_numeric(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Double
            | DataType::Decimal(_)
    )
}

/// Spark's `Cast.canAnsiCast` for a variant scalar's type.
fn can_ansi_cast(from: Source, to: &DataType) -> bool {
    let f = from.spark();
    if f == *to || is_string(to) {
        return true;
    }
    match from {
        Source::String => {
            matches!(
                to,
                DataType::Binary
                    | DataType::Boolean
                    | DataType::Timestamp
                    | DataType::TimestampNtz
                    | DataType::Date
                    | DataType::Time { .. }
            ) || is_numeric(to)
        }
        Source::Long | Source::Double | Source::Float | Source::Decimal(..) => {
            matches!(to, DataType::Boolean | DataType::Timestamp) || is_numeric(to)
        }
        Source::Boolean => is_numeric(to),
        Source::Date => matches!(to, DataType::Timestamp | DataType::TimestampNtz),
        Source::TimestampNtz => matches!(to, DataType::Timestamp | DataType::Date),
        Source::Timestamp => {
            matches!(to, DataType::TimestampNtz | DataType::Date) || is_numeric(to)
        }
        Source::Binary => false,
    }
}

/// The rows of one source type: their row indexes and the column of their
/// values.
enum Lane {
    Boolean(BooleanBuilder),
    Long(Int64Builder),
    Double(Float64Builder),
    Float(Float32Builder),
    Decimal(Vec<i128>),
    Date(Int32Builder),
    Timestamp(Vec<i64>),
    TimestampNtz(Vec<i64>),
    Binary(BinaryBuilder),
    String(StringBuilder),
}

impl Lane {
    fn new(s: Source) -> Self {
        match s {
            Source::Boolean => Self::Boolean(BooleanBuilder::new()),
            Source::Long => Self::Long(Int64Builder::new()),
            Source::Double => Self::Double(Float64Builder::new()),
            Source::Float => Self::Float(Float32Builder::new()),
            Source::Decimal(..) => Self::Decimal(vec![]),
            Source::Date => Self::Date(Int32Builder::new()),
            Source::Timestamp => Self::Timestamp(vec![]),
            Source::TimestampNtz => Self::TimestampNtz(vec![]),
            Source::Binary => Self::Binary(BinaryBuilder::new()),
            Source::String => Self::String(StringBuilder::new()),
        }
    }

    /// Appends `v`, whose source type is this lane's.
    fn push(&mut self, v: &VariantRef<'_>) -> Result<(), KrishivError> {
        match self {
            Self::Boolean(b) => b.append_value(v.as_bool()?),
            Self::Long(b) => b.append_value(v.as_i64()?),
            Self::Double(b) => b.append_value(v.as_f64()?),
            Self::Float(b) => b.append_value(v.as_f32()?),
            Self::Decimal(b) => {
                let (u, s) = v.as_decimal()?;
                b.push(strip_trailing_zeros(u, s).0);
            }
            Self::Date(b) => b.append_value(v.as_date()?),
            Self::Timestamp(b) | Self::TimestampNtz(b) => b.push(v.as_timestamp_micros()?),
            Self::Binary(b) => b.append_value(v.as_bytes()?),
            Self::String(b) => b.append_value(v.as_str()?),
        }
        Ok(())
    }

    fn finish(self, source: Source) -> Result<ArrayRef, KrishivError> {
        Ok(match self {
            Self::Boolean(mut b) => Arc::new(b.finish()),
            Self::Long(mut b) => Arc::new(b.finish()),
            Self::Double(mut b) => Arc::new(b.finish()),
            Self::Float(mut b) => Arc::new(b.finish()),
            Self::Decimal(v) => {
                let Source::Decimal(p, s) = source else {
                    return Err(internal("a decimal lane without a decimal type"));
                };
                Arc::new(
                    PrimitiveArray::<Decimal128Type>::from(v)
                        .with_precision_and_scale(p, s.cast_signed())
                        .map_err(|e| internal(&e.to_string()))?,
                )
            }
            Self::Date(mut b) => Arc::new(b.finish()),
            Self::Timestamp(v) => Arc::new(
                PrimitiveArray::<TimestampMicrosecondType>::from(v)
                    .with_timezone(krishiv_types::arrow::TIMESTAMP_ARROW_TIME_ZONE),
            ),
            Self::TimestampNtz(v) => Arc::new(PrimitiveArray::<TimestampMicrosecondType>::from(v)),
            Self::Binary(mut b) => Arc::new(b.finish()),
            Self::String(mut b) => Arc::new(b.finish()),
        })
    }
}

/// The source type of scalar `v`, or `None` for one Spark cannot read.
fn source_of(v: &VariantRef<'_>) -> Result<Option<Source>, KrishivError> {
    Ok(Some(match v.kind() {
        VariantKind::Bool => Source::Boolean,
        VariantKind::Int8 | VariantKind::Int16 | VariantKind::Int32 | VariantKind::Int64 => {
            Source::Long
        }
        VariantKind::Double => Source::Double,
        VariantKind::Float => Source::Float,
        VariantKind::Decimal { .. } => {
            let (u, s) = v.as_decimal()?;
            let (u, s) = strip_trailing_zeros(u, s);
            Source::Decimal(digits(u).max(s), s)
        }
        VariantKind::Date => Source::Date,
        VariantKind::Timestamp => Source::Timestamp,
        VariantKind::TimestampNtz => Source::TimestampNtz,
        VariantKind::Binary => Source::Binary,
        VariantKind::String => Source::String,
        _ => return Ok(None),
    }))
}

/// Where row `i`'s result comes from.
#[derive(Clone, Copy)]
enum Slot {
    Null,
    /// Lane `lane`, element `at`.
    Lane(usize, usize),
    /// The `at`-th precomputed string (JSON of a container, a UUID).
    Text(usize),
}

fn cast_atomic(
    ctx: &Ctx<'_>,
    values: &[Option<VariantRef<'_>>],
    to: &DataType,
) -> Result<ArrayRef, KrishivError> {
    // The kernels cast on the plain string and binary carriers.
    let target = match to.to_arrow() {
        ArrowType::Utf8View => ArrowType::Utf8,
        ArrowType::BinaryView => ArrowType::Binary,
        other => other,
    };
    let mut lanes: Vec<(Source, Lane, Vec<usize>)> = vec![];
    let mut texts = StringBuilder::new();
    let mut n_texts = 0_usize;
    let mut slots = vec![Slot::Null; values.len()];
    // Rows that failed before any kernel ran, by row.
    let mut first_invalid: Option<usize> = None;
    let mark = |row: usize, first: &mut Option<usize>| {
        if first.is_none_or(|f| row < f) {
            *first = Some(row);
        }
    };
    for (row, v) in values.iter().enumerate() {
        let Some(v) = v else { continue };
        match v.kind() {
            // Spark compares the target with `StringType` itself here.
            VariantKind::Uuid if *to == DataType::String => {
                texts.append_value(uuid_string(v.as_uuid()?));
                slots[row] = Slot::Text(n_texts);
                n_texts = n_texts.saturating_add(1);
            }
            VariantKind::Object | VariantKind::Array if is_string(to) => {
                let mut s = String::new();
                write_json(v, ctx.zone, &mut s)?;
                texts.append_value(s);
                slots[row] = Slot::Text(n_texts);
                n_texts = n_texts.saturating_add(1);
            }
            VariantKind::Uuid | VariantKind::Object | VariantKind::Array => {
                mark(row, &mut first_invalid);
            }
            _ => {
                let Some(source) = source_of(v)? else {
                    return Err(unknown_type(v));
                };
                let lane = match lanes.iter().position(|(s, ..)| *s == source) {
                    Some(l) => l,
                    None => {
                        lanes.push((source, Lane::new(source), vec![]));
                        lanes.len().saturating_sub(1)
                    }
                };
                let (_, builder, rows) = &mut lanes[lane];
                builder.push(v)?;
                slots[row] = Slot::Lane(lane, rows.len());
                rows.push(row);
            }
        }
    }
    let opts = CastOptions {
        mode: CastMode::Try,
        time_zone: ctx.zone,
    };
    let mut results: Vec<ArrayRef> = Vec::with_capacity(lanes.len());
    for (source, lane, rows) in lanes {
        let input = lane.finish(source)?;
        let out = if !can_ansi_cast(source, to) {
            new_null_array(&target, rows.len())
        } else if *to == DataType::Timestamp && matches!(source, Source::Long | Source::Decimal(..))
        {
            seconds_to_timestamp(&input, source)?
        } else {
            cast(input.as_ref(), &target, &opts).map_err(KrishivError::from)?
        };
        for (i, row) in rows.iter().enumerate() {
            if out.is_null(i) {
                mark(*row, &mut first_invalid);
                slots[*row] = Slot::Null;
            }
        }
        results.push(out);
    }
    if let Some(row) = first_invalid
        && let Some(v) = &values[row]
    {
        ctx.invalid(v, to)?;
    }
    let texts: ArrayRef = Arc::new(texts.finish());
    let texts = if n_texts > 0 {
        cast(texts.as_ref(), &target, &opts).map_err(KrishivError::from)?
    } else {
        new_null_array(&target, 0)
    };
    let null = new_null_array(&target, 1);
    let mut sources: Vec<&dyn Array> = results.iter().map(AsRef::as_ref).collect();
    let text_at = sources.len();
    sources.push(texts.as_ref());
    let null_at = sources.len();
    sources.push(null.as_ref());
    let picks: Vec<(usize, usize)> = slots
        .iter()
        .map(|s| match *s {
            Slot::Null => (null_at, 0),
            Slot::Lane(l, at) => (l, at),
            Slot::Text(at) => (text_at, at),
        })
        .collect();
    let out = arrow_select::interleave::interleave(&sources, &picks)
        .map_err(|e| internal(&e.to_string()))?;
    Ok(match out.data_type() {
        ArrowType::Utf8 => Arc::new(arrow_array::StringViewArray::from(out.as_string::<i32>())),
        ArrowType::Binary => Arc::new(arrow_array::BinaryViewArray::from(out.as_binary::<i32>())),
        _ => out,
    })
}

/// `castLongToTimestamp` / `castDecimalToTimestamp`: whole microseconds of
/// the seconds, `NULL` past the `BIGINT` range (an exact multiply, then
/// `BigDecimal.toBigInteger`, which truncates).
fn seconds_to_timestamp(input: &ArrayRef, source: Source) -> Result<ArrayRef, KrishivError> {
    let micros: Vec<Option<i64>> = match source {
        Source::Long => input
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .ok_or_else(|| internal("a long lane that is not Int64"))?
            .iter()
            .map(|v| v.and_then(|s| s.checked_mul(1_000_000)))
            .collect(),
        Source::Decimal(_, scale) => {
            let div = 10_i128.checked_pow(u32::from(scale)).unwrap_or(i128::MAX);
            input
                .as_any()
                .downcast_ref::<PrimitiveArray<Decimal128Type>>()
                .ok_or_else(|| internal("a decimal lane that is not Decimal128"))?
                .iter()
                .map(|v| {
                    v.and_then(|u| u.checked_mul(MICROS))
                        .and_then(|m| m.checked_div(div))
                        .and_then(|m| i64::try_from(m).ok())
                })
                .collect()
        }
        _ => return Err(internal("a timestamp lane of another type")),
    };
    Ok(Arc::new(
        PrimitiveArray::<TimestampMicrosecondType>::from(micros)
            .with_timezone(krishiv_types::arrow::TIMESTAMP_ARROW_TIME_ZONE),
    ))
}
