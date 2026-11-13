// SPDX-License-Identifier: Apache-2.0
//! Spark map functions over Arrow nested data: `map`, `map_concat`,
//! `map_contains_key`, `map_keys`, `map_values`, `map_entries`,
//! `map_from_arrays`, `map_from_entries` and `str_to_map`.
//!
//! Every builder works row by row over Arrow arrays directly (offsets and
//! child values, as the string kernels do): keys and values are never boxed
//! into a row enum. A row's entries are deduplicated by their Murmur3 hash
//! ([`crate::hash`], the observable Spark hash, which normalises `-0.0` and
//! NaN exactly as Spark's map-key normaliser does) and confirmed with
//! [`keys_equal`]; a hash collision only costs a comparison. Key equality is
//! Spark's: NaN equals NaN, `-0.0` equals `0.0`, strings and binary compare
//! byte-wise (the default binary collation) and arrays and structs compare
//! element by element, where two nulls are equal.
//!
//! A `NULL` key is `NULL_MAP_KEY` the moment it is met, in row-major pair
//! order, as Spark's builders raise it. A duplicate key is
//! `DUPLICATED_MAP_KEY` under the `EXCEPTION` policy and overwrites the
//! first value under `LAST_WIN` (keeping the first position, as Spark's
//! builder does).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{
    ArrayBuilder, BooleanBuilder, Date32Builder, Decimal64Builder, Decimal128Builder,
    DurationMicrosecondBuilder, Float32Builder, Float64Builder, Int8Builder, Int16Builder,
    Int32Builder, Int64Builder, IntervalMonthDayNanoBuilder, IntervalYearMonthBuilder, ListBuilder,
    MapBuilder, NullBuilder, StringViewBuilder, StructBuilder, TimestampMicrosecondBuilder,
    UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder, make_builder,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, DurationMicrosecondType, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, IntervalMonthDayNanoType, IntervalYearMonthType,
    TimestampMicrosecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, GenericListArray, ListArray, MapArray, PrimitiveArray,
    StructArray,
};
use arrow_buffer::{ArrowNativeType, NullBuffer, NullBufferBuilder, OffsetBuffer};
use arrow_schema::{DataType, FieldRef, IntervalUnit, TimeUnit};
use krishiv_common::error::{DuplicatedMapKey, KrishivError, LegacyErrorTemp2128, NullMapKey};
use krishiv_types::interval::calendar_string;

use crate::compare::SparkOrd;
use crate::error::KernelError;
use crate::hash;

/// How duplicate map keys resolve (`spark.sql.mapKeyDedupPolicy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupPolicy {
    /// Fail the query (`DUPLICATED_MAP_KEY`).
    Exception,
    /// The value inserted last wins.
    LastWin,
}

impl DedupPolicy {
    /// The policy of the session value (`EXCEPTION` when unset; the config
    /// registry only allows the two spellings).
    #[must_use]
    pub fn of(value: Option<&str>) -> Self {
        match value {
            Some("LAST_WIN") => Self::LastWin,
            _ => Self::Exception,
        }
    }
}

/// Whether the kernel builds maps over keys of Arrow type `t`: the types
/// the analyzer's casts produce (integers, floats, decimals, strings,
/// binary, dates, microsecond timestamps, ANSI intervals, and arrays and structs
/// of those). Anything else (a map, dictionary, union, calendar interval, time of
/// day, decimal beyond 128 bits, non-microsecond timestamps) cannot arrive here;
/// the binding refuses it first.
#[must_use]
pub fn supports_key(t: &DataType) -> bool {
    match t {
        // Spark dedups a calendar interval key but does not order it.
        t if krishiv_types::arrow::is_calendar_struct(t) => false,
        DataType::Null
        | DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Decimal64(..)
        | DataType::Decimal128(..)
        | DataType::Utf8
        | DataType::Utf8View
        | DataType::LargeUtf8
        | DataType::Binary
        | DataType::BinaryView
        | DataType::LargeBinary
        | DataType::Date32
        | DataType::Timestamp(TimeUnit::Microsecond, _)
        | DataType::Interval(IntervalUnit::YearMonth)
        | DataType::Duration(TimeUnit::Microsecond) => true,
        DataType::List(f) | DataType::LargeList(f) => supports_key(f.data_type()),
        DataType::Struct(fs) => fs
            .iter()
            .all(|f| supports_value(f.data_type()) && !holds_calendar(f.data_type())),
        _ => false,
    }
}

/// Whether `t` holds a calendar interval somewhere.
fn holds_calendar(t: &DataType) -> bool {
    match t {
        t if krishiv_types::arrow::is_calendar_struct(t) => true,
        DataType::List(f) | DataType::LargeList(f) | DataType::Map(f, _) => {
            holds_calendar(f.data_type())
        }
        DataType::Struct(fs) => fs.iter().any(|f| holds_calendar(f.data_type())),
        _ => false,
    }
}

/// Whether a key of Arrow type `t` is stored at the top level of a map: the key
/// types, plus a calendar interval, which Spark dedups by its three fields but
/// doesn't order (so only `map`, `map_from_arrays`, `map_from_entries` and
/// `map_concat` meet it, never as an array element).
#[must_use]
pub fn supports_map_key(t: &DataType) -> bool {
    krishiv_types::arrow::is_calendar_struct(t) || supports_key(t)
}

/// Whether the kernel stores values of Arrow type `t`: the key types, plus
/// calendar intervals and maps (a map value may itself be a map), at any depth.
#[must_use]
pub fn supports_value(t: &DataType) -> bool {
    match t {
        t if krishiv_types::arrow::is_calendar_struct(t) => true,
        DataType::List(f) | DataType::LargeList(f) => supports_value(f.data_type()),
        DataType::Struct(fs) => fs.iter().all(|f| supports_value(f.data_type())),
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(fs) if fs.len() == 2 => {
                supports_map_key(fs[0].data_type()) && supports_value(fs[1].data_type())
            }
            _ => false,
        },
        _ => supports_key(t),
    }
}

fn internal(e: impl std::fmt::Display) -> KernelError {
    KernelError::internal(e.to_string())
}

/// The hash at `at`: a missing entry means the hash run is shorter than the
/// entries it dedups, which is an internal error rather than a hash of zero.
fn hash_at(hash: &[i32], at: usize) -> Result<i32, KernelError> {
    hash.get(at)
        .copied()
        .ok_or_else(|| internal("map entries past their hashes"))
}

/// Spark's equality of two key values: `a[i] = b[j]`. Both sides already
/// share the Spark type (the analyzer cast them), but their carriers may
/// differ (a plain `Utf8` against a view, a `List` against a `LargeList`).
/// Floats use [`SparkOrd`] (NaN equals NaN, `-0.0` equals `0.0`); nested
/// values compare element by element with two nulls equal.
///
/// # Errors
/// `INTERNAL_ERROR` for a type [`supports_key`] refuses (unreachable: the
/// binding checks first).
pub fn keys_equal(a: &dyn Array, i: usize, b: &dyn Array, j: usize) -> Result<bool, KernelError> {
    use DataType as T;
    if a.is_null(i) || b.is_null(j) {
        return Ok(a.is_null(i) && b.is_null(j));
    }
    match (a.data_type(), b.data_type()) {
        (T::Null, T::Null) => Ok(true),
        (T::Boolean, T::Boolean) => Ok(a.as_boolean().value(i) == b.as_boolean().value(j)),
        (T::Int8, T::Int8) => {
            Ok(a.as_primitive::<Int8Type>().value(i) == b.as_primitive::<Int8Type>().value(j))
        }
        (T::Int16, T::Int16) => {
            Ok(a.as_primitive::<Int16Type>().value(i) == b.as_primitive::<Int16Type>().value(j))
        }
        (T::Int32, T::Int32) => {
            Ok(a.as_primitive::<Int32Type>().value(i) == b.as_primitive::<Int32Type>().value(j))
        }
        (T::Int64, T::Int64) => {
            Ok(a.as_primitive::<Int64Type>().value(i) == b.as_primitive::<Int64Type>().value(j))
        }
        (T::UInt8, T::UInt8) => {
            Ok(a.as_primitive::<UInt8Type>().value(i) == b.as_primitive::<UInt8Type>().value(j))
        }
        (T::UInt16, T::UInt16) => {
            Ok(a.as_primitive::<UInt16Type>().value(i) == b.as_primitive::<UInt16Type>().value(j))
        }
        (T::UInt32, T::UInt32) => {
            Ok(a.as_primitive::<UInt32Type>().value(i) == b.as_primitive::<UInt32Type>().value(j))
        }
        (T::UInt64, T::UInt64) => {
            Ok(a.as_primitive::<UInt64Type>().value(i) == b.as_primitive::<UInt64Type>().value(j))
        }
        (T::Float32, T::Float32) => Ok(f32::spark_eq(
            a.as_primitive::<Float32Type>().value(i),
            b.as_primitive::<Float32Type>().value(j),
        )),
        (T::Float64, T::Float64) => Ok(f64::spark_eq(
            a.as_primitive::<Float64Type>().value(i),
            b.as_primitive::<Float64Type>().value(j),
        )),
        (T::Decimal64(..), T::Decimal64(..)) => Ok(a.as_primitive::<Decimal64Type>().value(i)
            == b.as_primitive::<Decimal64Type>().value(j)),
        (T::Decimal128(..), T::Decimal128(..)) => Ok(a.as_primitive::<Decimal128Type>().value(i)
            == b.as_primitive::<Decimal128Type>().value(j)),
        (x, y) if is_string(x) && is_string(y) => Ok(string_value(a, i) == string_value(b, j)),
        (x, y) if is_binary(x) && is_binary(y) => Ok(binary_value(a, i) == binary_value(b, j)),
        (T::Date32, T::Date32) => {
            Ok(a.as_primitive::<Date32Type>().value(i) == b.as_primitive::<Date32Type>().value(j))
        }
        (T::Timestamp(TimeUnit::Microsecond, _), T::Timestamp(TimeUnit::Microsecond, _)) => {
            Ok(a.as_primitive::<TimestampMicrosecondType>().value(i)
                == b.as_primitive::<TimestampMicrosecondType>().value(j))
        }
        (T::Interval(IntervalUnit::YearMonth), T::Interval(IntervalUnit::YearMonth)) => {
            Ok(a.as_primitive::<IntervalYearMonthType>().value(i)
                == b.as_primitive::<IntervalYearMonthType>().value(j))
        }
        (T::Duration(TimeUnit::Microsecond), T::Duration(TimeUnit::Microsecond)) => {
            Ok(a.as_primitive::<DurationMicrosecondType>().value(i)
                == b.as_primitive::<DurationMicrosecondType>().value(j))
        }
        (T::Interval(IntervalUnit::MonthDayNano), T::Interval(IntervalUnit::MonthDayNano)) => {
            Ok(a.as_primitive::<IntervalMonthDayNanoType>().value(i)
                == b.as_primitive::<IntervalMonthDayNanoType>().value(j))
        }
        (T::List(_) | T::LargeList(_), T::List(_) | T::LargeList(_)) => list_equal(a, i, b, j),
        (T::Struct(_), T::Struct(_)) => struct_equal(a.as_struct(), i, b.as_struct(), j),
        (x, y) => Err(internal(format!(
            "map keys of {x} and {y} are not comparable"
        ))),
    }
}

fn is_string(t: &DataType) -> bool {
    matches!(t, DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8)
}

fn is_binary(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Binary | DataType::BinaryView | DataType::LargeBinary
    )
}

fn string_value(a: &dyn Array, i: usize) -> &str {
    match a.data_type() {
        DataType::Utf8 => a.as_string::<i32>().value(i),
        DataType::LargeUtf8 => a.as_string::<i64>().value(i),
        _ => a.as_string_view().value(i),
    }
}

fn binary_value(a: &dyn Array, i: usize) -> &[u8] {
    match a.data_type() {
        DataType::Binary => a.as_binary::<i32>().value(i),
        DataType::LargeBinary => a.as_binary::<i64>().value(i),
        _ => a.as_binary_view().value(i),
    }
}

/// The child range of list row `i`, whatever the offset width.
fn list_range(a: &dyn Array, i: usize) -> Result<(usize, usize), KernelError> {
    let err = || internal("a list row past its offsets");
    let (start, end) = match a.data_type() {
        DataType::List(_) => {
            let off = a.as_list::<i32>().value_offsets();
            let (Some(s), Some(e)) = (off.get(i), off.get(i.saturating_add(1))) else {
                return Err(err());
            };
            (s.as_usize(), e.as_usize())
        }
        DataType::LargeList(_) => {
            let off = a.as_list::<i64>().value_offsets();
            let (Some(s), Some(e)) = (off.get(i), off.get(i.saturating_add(1))) else {
                return Err(err());
            };
            (s.as_usize(), e.as_usize())
        }
        t => return Err(internal(format!("a list row of type {t}"))),
    };
    Ok((start, end))
}

fn list_values(a: &dyn Array) -> ArrayRef {
    match a.data_type() {
        DataType::List(_) => Arc::clone(a.as_list::<i32>().values()),
        _ => Arc::clone(a.as_list::<i64>().values()),
    }
}

fn list_equal(a: &dyn Array, i: usize, b: &dyn Array, j: usize) -> Result<bool, KernelError> {
    let (from, to) = list_range(a, i)?;
    let (other_from, other_to) = list_range(b, j)?;
    let (len, other_len) = (to.saturating_sub(from), other_to.saturating_sub(other_from));
    if len != other_len {
        return Ok(false);
    }
    let (values, other) = (list_values(a), list_values(b));
    for k in 0..len {
        let (x, y) = (from.saturating_add(k), other_from.saturating_add(k));
        if !keys_equal(values.as_ref(), x, other.as_ref(), y)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn struct_equal(a: &StructArray, i: usize, b: &StructArray, j: usize) -> Result<bool, KernelError> {
    let (DataType::Struct(af), DataType::Struct(bf)) = (a.data_type(), b.data_type()) else {
        return Err(internal("a struct key without struct fields"));
    };
    if af.len() != bf.len() {
        return Ok(false);
    }
    for (x, y) in a.columns().iter().zip(b.columns()) {
        if !keys_equal(x.as_ref(), i, y.as_ref(), j)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Appends `array[idx]` to `builder`, whose type is the canonical carrier
/// of `array`'s Spark type (views for strings and binary). A null appends
/// null; a null struct appends nulls to every field, which readers ignore.
///
/// # Errors
/// `INTERNAL_ERROR` for a type [`supports_value`] refuses, or when the
/// builder and the value disagree (both unreachable: the binding checks the
/// types and builds the builders from the same output field).
#[expect(clippy::too_many_lines, reason = "one arm per Arrow carrier")]
fn append_value(
    builder: &mut dyn ArrayBuilder,
    array: &dyn Array,
    idx: usize,
) -> Result<(), KernelError> {
    use DataType as T;
    macro_rules! prim {
        ($t:ty, $b:ty) => {{
            let a = array
                .as_primitive_opt::<$t>()
                .ok_or_else(|| internal("a map value of the wrong type"))?;
            let b = builder
                .as_any_mut()
                .downcast_mut::<$b>()
                .ok_or_else(|| internal("a map builder of the wrong type"))?;
            if a.is_null(idx) {
                b.append_null();
            } else {
                b.append_value(a.value(idx));
            }
            Ok(())
        }};
    }
    match array.data_type() {
        T::Null => {
            downcast::<NullBuilder>(builder)?.append_null();
            Ok(())
        }
        T::Boolean => {
            let (a, b) = (array.as_boolean(), downcast::<BooleanBuilder>(builder)?);
            if a.is_null(idx) {
                b.append_null();
            } else {
                b.append_value(a.value(idx));
            }
            Ok(())
        }
        T::Int8 => prim!(Int8Type, Int8Builder),
        T::Int16 => prim!(Int16Type, Int16Builder),
        T::Int32 => prim!(Int32Type, Int32Builder),
        T::Int64 => prim!(Int64Type, Int64Builder),
        T::UInt8 => prim!(UInt8Type, UInt8Builder),
        T::UInt16 => prim!(UInt16Type, UInt16Builder),
        T::UInt32 => prim!(UInt32Type, UInt32Builder),
        T::UInt64 => prim!(UInt64Type, UInt64Builder),
        T::Float32 => prim!(Float32Type, Float32Builder),
        T::Float64 => prim!(Float64Type, Float64Builder),
        T::Decimal64(..) => prim!(Decimal64Type, Decimal64Builder),
        T::Decimal128(..) => prim!(Decimal128Type, Decimal128Builder),
        t if is_string(t) => {
            let b = downcast::<StringViewBuilder>(builder)?;
            if array.is_null(idx) {
                b.append_null();
            } else {
                b.append_value(string_value(array, idx));
            }
            Ok(())
        }
        t if is_binary(t) => {
            let b = downcast::<arrow_array::builder::BinaryViewBuilder>(builder)?;
            if array.is_null(idx) {
                b.append_null();
            } else {
                b.append_value(binary_value(array, idx));
            }
            Ok(())
        }
        T::Date32 => prim!(Date32Type, Date32Builder),
        T::Timestamp(TimeUnit::Microsecond, _) => {
            prim!(TimestampMicrosecondType, TimestampMicrosecondBuilder)
        }
        T::Interval(IntervalUnit::YearMonth) => {
            prim!(IntervalYearMonthType, IntervalYearMonthBuilder)
        }
        T::Interval(IntervalUnit::MonthDayNano) => {
            prim!(IntervalMonthDayNanoType, IntervalMonthDayNanoBuilder)
        }
        T::Duration(TimeUnit::Microsecond) => {
            prim!(DurationMicrosecondType, DurationMicrosecondBuilder)
        }
        T::List(_) | T::LargeList(_) => {
            if array.is_null(idx) {
                append_list_null(builder)?;
                return Ok(());
            }
            let (from, to) = list_range(array, idx)?;
            append_list(builder, list_values(array).as_ref(), from, to)
        }
        T::Struct(_) => {
            let a = array.as_struct();
            let b = downcast::<StructBuilder>(builder)?;
            if array.is_null(idx) {
                for (f, col) in b.field_builders_mut().iter_mut().zip(a.columns()) {
                    append_value(f.as_mut(), col.as_ref(), idx)?;
                }
                b.append(false);
                return Ok(());
            }
            for (f, col) in b.field_builders_mut().iter_mut().zip(a.columns()) {
                append_value(f.as_mut(), col.as_ref(), idx)?;
            }
            b.append(true);
            Ok(())
        }
        T::Map(..) => {
            let Some(a) = array.as_map_opt() else {
                return Err(internal("a map value without map data"));
            };
            let b = downcast::<MapBuilder<Box<dyn ArrayBuilder>, Box<dyn ArrayBuilder>>>(builder)?;
            if array.is_null(idx) {
                b.append(false).map_err(internal)?;
                return Ok(());
            }
            let (from, to) = map_range(a, idx)?;
            for p in from..to {
                append_value(b.keys().as_mut(), a.keys().as_ref(), p)?;
                append_value(b.values().as_mut(), a.values().as_ref(), p)?;
            }
            b.append(true).map_err(internal)?;
            Ok(())
        }
        t => Err(internal(format!("a map value of type {t}"))),
    }
}

fn downcast<B: ArrayBuilder>(builder: &mut dyn ArrayBuilder) -> Result<&mut B, KernelError> {
    builder
        .as_any_mut()
        .downcast_mut::<B>()
        .ok_or_else(|| internal("a map builder of the wrong type"))
}

type DynListBuilder = ListBuilder<Box<dyn ArrayBuilder>>;
type DynLargeListBuilder = arrow_array::builder::LargeListBuilder<Box<dyn ArrayBuilder>>;

fn append_list(
    builder: &mut dyn ArrayBuilder,
    values: &dyn Array,
    from: usize,
    to: usize,
) -> Result<(), KernelError> {
    if let Some(b) = builder.as_any_mut().downcast_mut::<DynListBuilder>() {
        for p in from..to {
            append_value(b.values().as_mut(), values, p)?;
        }
        b.append(true);
        return Ok(());
    }
    let b = downcast::<DynLargeListBuilder>(builder)?;
    for p in from..to {
        append_value(b.values().as_mut(), values, p)?;
    }
    b.append(true);
    Ok(())
}

fn append_list_null(builder: &mut dyn ArrayBuilder) -> Result<(), KernelError> {
    if let Some(b) = builder.as_any_mut().downcast_mut::<DynListBuilder>() {
        b.append(false);
        return Ok(());
    }
    downcast::<DynLargeListBuilder>(builder)?.append(false);
    Ok(())
}

/// The child range of map row `i`.
fn map_range(m: &MapArray, i: usize) -> Result<(usize, usize), KernelError> {
    let offsets = m.value_offsets();
    let (Some(from), Some(to)) = (offsets.get(i), offsets.get(i.saturating_add(1))) else {
        return Err(internal("a map row past its offsets"));
    };
    Ok((from.as_usize(), to.as_usize()))
}

/// One entry of a row under construction: the key and value arrays and
/// their positions. Positions only; values are read once, when the entry
/// survives deduplication.
#[derive(Clone, Copy)]
struct Entry<'a> {
    keys: &'a dyn Array,
    key: usize,
    values: &'a dyn Array,
    value: usize,
}

/// The deduplicated entries of every row, in first-occurrence order with
/// the value position remapped under `LAST_WIN`.
struct Deduped<'a> {
    rows: Vec<Vec<Entry<'a>>>,
}

impl<'a> Deduped<'a> {
    /// Deduplicates `per_row` in order against `hash` (its entries in
    /// row-major entry order): a null key is `NULL_MAP_KEY`, a repeat is
    /// `DUPLICATED_MAP_KEY` under `EXCEPTION` and overwrites the value
    /// under `LAST_WIN`.
    fn of(
        per_row: Vec<Vec<Entry<'a>>>,
        policy: DedupPolicy,
        hash: &[i32],
    ) -> Result<Self, KernelError> {
        let mut rows = Vec::with_capacity(per_row.len());
        let mut at = 0_usize;
        for (row, entries) in per_row.into_iter().enumerate() {
            let mut seen: HashMap<i32, Vec<usize>> = HashMap::with_capacity(entries.len());
            let mut kept: Vec<Entry<'a>> = Vec::with_capacity(entries.len());
            for e in entries {
                if e.keys.is_null(e.key) {
                    return Err(KernelError::with_row(Some(row), NullMapKey {}.into()));
                }
                let h = hash_at(hash, at)?;
                at = at.saturating_add(1);
                let mut found = None;
                if let Some(same) = seen.get(&h) {
                    for &p in same {
                        if keys_equal(kept[p].keys, kept[p].key, e.keys, e.key)? {
                            found = Some(p);
                            break;
                        }
                    }
                }
                match found {
                    Some(p) => {
                        if policy == DedupPolicy::Exception {
                            return Err(KernelError::with_row(
                                Some(row),
                                duplicate_key(e.keys, e.key)?,
                            ));
                        }
                        kept[p].values = e.values;
                        kept[p].value = e.value;
                    }
                    None => {
                        seen.entry(h).or_default().push(kept.len());
                        kept.push(e);
                    }
                }
            }
            rows.push(kept);
        }
        Ok(Self { rows })
    }
}

/// `DUPLICATED_MAP_KEY` for the key value, rendered as Spark renders it.
fn duplicate_key(keys: &dyn Array, key: usize) -> Result<KrishivError, KernelError> {
    Ok(DuplicatedMapKey {
        key: key_text(keys, key)?,
        map_key_dedup_policy: "\"spark.sql.mapKeyDedupPolicy\"".to_owned(),
    }
    .into())
}

/// Spark's `key.toString` for a key value: strings as they are, numbers by
/// their display (`1.0` for floating and decimal one, `NaN` for NaN), dates
/// and timestamps by their internal number, binary by its bytes. Nested
/// keys render in brackets; only the error class is compared anywhere, so
/// this is best-effort display, not a format pin.
fn key_text(keys: &dyn Array, key: usize) -> Result<String, KernelError> {
    use DataType as T;
    match keys.data_type() {
        T::Utf8 | T::Utf8View | T::LargeUtf8 => Ok(string_value(keys, key).to_owned()),
        T::Boolean => Ok(keys.as_boolean().value(key).to_string()),
        T::Int8 => Ok(keys.as_primitive::<Int8Type>().value(key).to_string()),
        T::Int16 => Ok(keys.as_primitive::<Int16Type>().value(key).to_string()),
        T::Int32 => Ok(keys.as_primitive::<Int32Type>().value(key).to_string()),
        T::Int64 => Ok(keys.as_primitive::<Int64Type>().value(key).to_string()),
        T::UInt8 => Ok(keys.as_primitive::<UInt8Type>().value(key).to_string()),
        T::UInt16 => Ok(keys.as_primitive::<UInt16Type>().value(key).to_string()),
        T::UInt32 => Ok(keys.as_primitive::<UInt32Type>().value(key).to_string()),
        T::UInt64 => Ok(keys.as_primitive::<UInt64Type>().value(key).to_string()),
        T::Float32 => {
            let v = keys.as_primitive::<Float32Type>().value(key);
            if v.is_nan() {
                Ok("NaN".to_owned())
            } else {
                Ok(format!("{v:?}"))
            }
        }
        T::Float64 => {
            let v = keys.as_primitive::<Float64Type>().value(key);
            if v.is_nan() {
                Ok("NaN".to_owned())
            } else {
                Ok(format!("{v:?}"))
            }
        }
        T::Decimal64(_, scale) => decimal_text(
            i128::from(keys.as_primitive::<Decimal64Type>().value(key)),
            *scale,
        ),
        T::Decimal128(_, scale) => {
            decimal_text(keys.as_primitive::<Decimal128Type>().value(key), *scale)
        }
        T::Binary | T::BinaryView | T::LargeBinary => Ok(format!("{:?}", binary_value(keys, key))),
        T::Date32 => Ok(keys.as_primitive::<Date32Type>().value(key).to_string()),
        T::Timestamp(..) => Ok(keys
            .as_primitive::<TimestampMicrosecondType>()
            .value(key)
            .to_string()),
        T::Interval(IntervalUnit::YearMonth) => Ok(keys
            .as_primitive::<IntervalYearMonthType>()
            .value(key)
            .to_string()),
        T::Duration(TimeUnit::Microsecond) => Ok(keys
            .as_primitive::<DurationMicrosecondType>()
            .value(key)
            .to_string()),
        t if krishiv_types::arrow::is_calendar_struct(t) => {
            let v = krishiv_types::calendar::Calendars::of(keys)
                .ok_or_else(|| internal("a calendar key that is not one".to_owned()))?
                .value(key);
            Ok(calendar_string(v.months, v.days, v.micros))
        }
        T::List(_) | T::LargeList(_) => {
            let (from, to) = list_range(keys, key)?;
            let values = list_values(keys);
            let mut parts = Vec::new();
            for p in from..to {
                parts.push(if values.is_null(p) {
                    "null".to_owned()
                } else {
                    key_text(values.as_ref(), p)?
                });
            }
            Ok(format!("[{}]", parts.join(", ")))
        }
        T::Struct(_) => {
            let s = keys.as_struct();
            let mut parts = Vec::new();
            for col in s.columns() {
                parts.push(if col.is_null(key) {
                    "null".to_owned()
                } else {
                    key_text(col.as_ref(), key)?
                });
            }
            Ok(format!("[{}]", parts.join(", ")))
        }
        _ => Ok(format!("{:?}", keys)),
    }
}

/// A decimal unscaled value with `scale` digits after the point.
fn decimal_text(unscaled: i128, scale: i8) -> Result<String, KernelError> {
    let scale = usize::try_from(scale.max(0)).map_err(internal)?;
    let digits = unscaled.unsigned_abs().to_string();
    let padded = format!("{:0>width$}", digits, width = scale.saturating_add(1));
    let point = padded.len().saturating_sub(scale);
    let (int, frac) = padded.split_at(point);
    let sign = if unscaled < 0 { "-" } else { "" };
    if scale == 0 {
        Ok(format!("{sign}{int}"))
    } else {
        Ok(format!("{sign}{int}.{frac}"))
    }
}

/// Assembles the deduplicated rows into a [`MapArray`] over `entries` (the
/// `entries` struct field of the output map type).
fn assemble(
    entries: &FieldRef,
    deduped: &Deduped<'_>,
    nulls: Option<NullBuffer>,
) -> Result<MapArray, KernelError> {
    let fields = match entries.data_type() {
        DataType::Struct(fs) if fs.len() == 2 => fs.clone(),
        t => return Err(internal(format!("map entries of type {t}"))),
    };
    let total: usize = deduped.rows.iter().map(Vec::len).sum();
    let mut keys = make_builder(fields[0].data_type(), total);
    let mut values = make_builder(fields[1].data_type(), total);
    let mut offsets = Vec::with_capacity(deduped.rows.len().saturating_add(1));
    offsets.push(0_i32);
    for row in &deduped.rows {
        for e in row {
            append_value(keys.as_mut(), e.keys, e.key)?;
            append_value(values.as_mut(), e.values, e.value)?;
        }
        let next = i32::try_from(keys.len())
            .map_err(|_| KernelError::out_of_memory(keys.len().saturating_mul(16)))?;
        offsets.push(next);
    }
    let values = StructArray::try_new(fields, vec![keys.finish(), values.finish()], None)
        .map_err(internal)?;
    let offsets = OffsetBuffer::new(offsets.into());
    MapArray::try_new(Arc::clone(entries), offsets, values, nulls, false).map_err(internal)
}

/// The Murmur3 hash of every key of `arrays[i]` (of `counts[i]` values),
/// concatenated: equal keys hash equally whatever array holds them.
fn hash_all(arrays: &[&dyn Array], counts: &[usize]) -> Result<Vec<i32>, KernelError> {
    let mut out = Vec::new();
    for (a, n) in arrays.iter().zip(counts) {
        if *n == 0 {
            continue;
        }
        let plain = hashable(*a)?;
        let h = hash::hash(&[plain.as_ref()], *n)?;
        out.extend(h.values().iter().copied());
    }
    Ok(out)
}

/// `map(k1, v1, ...)`: one entry per key/value column pair per row.
/// `keys[i]` pairs with `values[i]`; every column has `rows` values and all
/// keys share the Arrow type, as do all values.
pub fn create_map(
    keys: &[ArrayRef],
    values: &[ArrayRef],
    rows: usize,
    entries: &FieldRef,
    policy: DedupPolicy,
) -> Result<MapArray, KernelError> {
    if keys.len() != values.len() {
        return Err(internal("map keys and values of different counts"));
    }
    for (a, b) in keys.iter().zip(values) {
        if a.len() != rows || b.len() != rows {
            return Err(internal("a map column of the wrong length"));
        }
    }
    let key_refs: Vec<&dyn Array> = keys.iter().map(AsRef::as_ref).collect();
    // Column-major hashes: pair `i`, row `r` hashed at `i * rows + r`.
    let hash = hash_all(&key_refs, &vec![rows; keys.len()])?;
    let mut per_row = Vec::with_capacity(rows);
    let mut ordered = Vec::with_capacity(rows.saturating_mul(keys.len()));
    for r in 0..rows {
        let mut row = Vec::with_capacity(keys.len());
        for (i, (k, v)) in keys.iter().zip(values).enumerate() {
            let at = i
                .checked_mul(rows)
                .and_then(|at| at.checked_add(r))
                .ok_or_else(|| internal("map entries past their hashes"))?;
            ordered.push(hash_at(&hash, at)?);
            row.push(Entry {
                keys: k.as_ref(),
                key: r,
                values: v.as_ref(),
                value: r,
            });
        }
        per_row.push(row);
    }
    assemble(entries, &Deduped::of(per_row, policy, &ordered)?, None)
}

/// `map_concat(m1, ...)`: the entries of every map per row, in order. A
/// null map makes its row null, whatever the other maps hold.
pub fn concat(
    maps: &[ArrayRef],
    rows: usize,
    entries: &FieldRef,
    policy: DedupPolicy,
) -> Result<MapArray, KernelError> {
    let marrays: Vec<MapArray> = maps
        .iter()
        .map(|m| {
            if m.len() != rows {
                return Err(internal("a map_concat argument of the wrong length"));
            }
            let Some(m) = m.as_map_opt() else {
                return Err(internal("a map_concat argument without map data"));
            };
            Ok(m.clone())
        })
        .collect::<Result<_, _>>()?;
    let key_refs: Vec<&dyn Array> = marrays.iter().map(|m| m.keys().as_ref()).collect();
    let counts: Vec<usize> = marrays.iter().map(|m| m.keys().len()).collect();
    let hash = hash_all(&key_refs, &counts)?;
    // The whole-array hash of map `mi` starts at `base[mi]`.
    let mut base = Vec::with_capacity(marrays.len());
    let mut len = 0_usize;
    for n in &counts {
        base.push(len);
        len = len.saturating_add(*n);
    }
    let mut nulls = NullBufferBuilder::new(rows);
    let mut per_row = Vec::with_capacity(rows);
    let mut ordered = Vec::new();
    for r in 0..rows {
        if marrays.iter().any(|m| m.is_null(r)) {
            nulls.append(false);
            per_row.push(Vec::new());
            continue;
        }
        nulls.append(true);
        let mut row = Vec::new();
        for (mi, m) in marrays.iter().enumerate() {
            let (from, to) = map_range(m, r)?;
            for p in from..to {
                let base_at = base
                    .get(mi)
                    .copied()
                    .ok_or_else(|| internal("map entries past their hashes"))?;
                let h = hash_at(&hash, base_at.saturating_add(p))?;
                ordered.push(h);
                row.push(Entry {
                    keys: m.keys().as_ref(),
                    key: p,
                    values: m.values().as_ref(),
                    value: p,
                });
            }
        }
        per_row.push(row);
    }
    assemble(
        entries,
        &Deduped::of(per_row, policy, &ordered)?,
        nulls.finish(),
    )
}

/// `map_contains_key(map, key)`: whether the key occurs in each row's map
/// (a linear scan, as Spark's `ArrayContains` does). A null map or a null
/// key is null.
pub fn contains_key(map: &ArrayRef, key: &ArrayRef) -> Result<BooleanArray, KernelError> {
    let rows = map.len();
    if key.len() != rows {
        return Err(internal("map_contains_key arguments of different lengths"));
    }
    let Some(m) = map.as_map_opt() else {
        return Err(internal("a map function over non-map data"));
    };
    let mut out = BooleanArray::builder(rows);
    for r in 0..rows {
        if m.is_null(r) || key.is_null(r) {
            out.append_null();
            continue;
        }
        let (from, to) = map_range(m, r)?;
        let mut found = false;
        for p in from..to {
            if keys_equal(m.keys().as_ref(), p, key.as_ref(), r)? {
                found = true;
                break;
            }
        }
        out.append_value(found);
    }
    Ok(out.finish())
}

/// `map_keys` / `map_values`: the key or value slice of every row, sharing
/// the map's children and offsets (a null map stays null).
fn project(m: &MapArray, child: &ArrayRef, element: &FieldRef) -> Result<ArrayRef, KernelError> {
    Ok(Arc::new(ListArray::new(
        Arc::clone(element),
        m.offsets().clone(),
        Arc::clone(child),
        m.nulls().cloned(),
    )))
}

/// `map_keys(map)`.
pub fn keys(map: &ArrayRef, element: &FieldRef) -> Result<ArrayRef, KernelError> {
    let Some(m) = map.as_map_opt() else {
        return Err(internal("a map function over non-map data"));
    };
    project(m, m.keys(), element)
}

/// `map_values(map)`.
pub fn values(map: &ArrayRef, element: &FieldRef) -> Result<ArrayRef, KernelError> {
    let Some(m) = map.as_map_opt() else {
        return Err(internal("a map function over non-map data"));
    };
    project(m, m.values(), element)
}

/// `map_entries(map)`: one `{key, value}` struct per entry, over the shared
/// children (a null map stays null).
pub fn entries(map: &ArrayRef, element: &FieldRef) -> Result<ArrayRef, KernelError> {
    let Some(m) = map.as_map_opt() else {
        return Err(internal("a map function over non-map data"));
    };
    let fields = match element.data_type() {
        DataType::Struct(fs) => fs.clone(),
        t => return Err(internal(format!("map entries of type {t}"))),
    };
    let structs = StructArray::try_new(
        fields,
        vec![Arc::clone(m.keys()), Arc::clone(m.values())],
        None,
    )
    .map_err(internal)?;
    Ok(Arc::new(ListArray::new(
        Arc::clone(element),
        m.offsets().clone(),
        Arc::new(structs),
        m.nulls().cloned(),
    )))
}

/// The list rows of `a` (a `List` of `rows` rows) as child ranges.
fn list_rows(a: &GenericListArray<i32>, rows: usize) -> Result<Vec<(usize, usize)>, KernelError> {
    if a.len() != rows {
        return Err(internal("a map input of the wrong length"));
    }
    (0..rows)
        .map(|r| {
            if a.is_null(r) {
                Ok(None)
            } else {
                let (Some(from), Some(to)) = (
                    a.value_offsets().get(r),
                    a.value_offsets().get(r.saturating_add(1)),
                ) else {
                    return Err(internal("a list row past its offsets"));
                };
                Ok(Some((from.as_usize(), to.as_usize())))
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|v| v.into_iter().map(|o| o.unwrap_or((0, 0))).collect())
}

/// `map_from_arrays(keys, values)`: row `r` maps the `r`-th key slice to
/// the `r`-th value slice. Slices of different lengths are
/// `_LEGACY_ERROR_TEMP_2128`; a null slice makes its row null.
pub fn from_arrays(
    keys: &ArrayRef,
    values: &ArrayRef,
    rows: usize,
    entries: &FieldRef,
    policy: DedupPolicy,
) -> Result<MapArray, KernelError> {
    let (k, v) = match (keys.data_type(), values.data_type()) {
        (DataType::List(_), DataType::List(_)) => (keys.as_list::<i32>(), values.as_list::<i32>()),
        (x, y) => return Err(internal(format!("map_from_arrays over {x} and {y}"))),
    };
    let (kr, vr) = (list_rows(k, rows)?, list_rows(v, rows)?);
    let hash = hash_all(&[k.values().as_ref()], &[k.values().len()])?;
    let mut nulls = NullBufferBuilder::new(rows);
    let mut per_row = Vec::with_capacity(rows);
    let mut ordered = Vec::new();
    for r in 0..rows {
        if k.is_null(r) || v.is_null(r) {
            nulls.append(false);
            per_row.push(Vec::new());
            continue;
        }
        nulls.append(true);
        let ((kf, kt), (vf, vt)) = (kr[r], vr[r]);
        if kt.saturating_sub(kf) != vt.saturating_sub(vf) {
            return Err(KernelError::with_row(
                Some(r),
                LegacyErrorTemp2128 {}.into(),
            ));
        }
        let mut row = Vec::with_capacity(kt.saturating_sub(kf));
        for (p, q) in (kf..kt).zip(vf..vt) {
            ordered.push(hash_at(&hash, p)?);
            row.push(Entry {
                keys: k.values().as_ref(),
                key: p,
                values: v.values().as_ref(),
                value: q,
            });
        }
        per_row.push(row);
    }
    assemble(
        entries,
        &Deduped::of(per_row, policy, &ordered)?,
        nulls.finish(),
    )
}

/// `map_from_entries(entries)`: row `r` maps the `r`-th struct slice's
/// first field to its second. A null element (of a nullable array) makes
/// only its own row null; a null key field is `NULL_MAP_KEY`.
pub fn from_entries(
    array: &ArrayRef,
    rows: usize,
    entries: &FieldRef,
    policy: DedupPolicy,
) -> Result<MapArray, KernelError> {
    let list = match array.data_type() {
        DataType::List(_) => array.as_list::<i32>(),
        t => return Err(internal(format!("map_from_entries over {t}"))),
    };
    let (keys, values) = match list.values().data_type() {
        DataType::Struct(fs) if fs.len() == 2 => {
            let s = list.values().as_struct();
            (Arc::clone(&s.columns()[0]), Arc::clone(&s.columns()[1]))
        }
        t => return Err(internal(format!("map_from_entries over entries of {t}"))),
    };
    let ranges = list_rows(list, rows)?;
    let hash = hash_all(&[keys.as_ref()], &[keys.len()])?;
    let mut nulls = NullBufferBuilder::new(rows);
    let mut per_row = Vec::with_capacity(rows);
    let mut ordered = Vec::new();
    for (r, range) in ranges.iter().enumerate() {
        if list.is_null(r) {
            nulls.append(false);
            per_row.push(Vec::new());
            continue;
        }
        let (from, to) = *range;
        let base = ordered.len();
        let mut row = Vec::with_capacity(to.saturating_sub(from));
        let mut null_element = false;
        for p in from..to {
            if list.values().is_null(p) {
                null_element = true;
                break;
            }
            ordered.push(hash_at(&hash, p)?);
            row.push(Entry {
                keys: keys.as_ref(),
                key: p,
                values: values.as_ref(),
                value: p,
            });
        }
        if null_element {
            // A null element of a nullable array nulls only its own row.
            ordered.truncate(base);
            nulls.append(false);
            per_row.push(Vec::new());
            continue;
        }
        nulls.append(true);
        per_row.push(row);
    }
    assemble(
        entries,
        &Deduped::of(per_row, policy, &ordered)?,
        nulls.finish(),
    )
}

/// `str_to_map` assembly: `keys` and `values` hold every row's pairs in
/// row-major order (`row_of[p]` is pair `p`'s row). A row with no pair had
/// a null text or delimiter and is null; any other row keeps its pairs,
/// even when there is only one.
pub fn from_pairs(
    keys: &ArrayRef,
    values: &ArrayRef,
    row_of: &[usize],
    rows: usize,
    entries: &FieldRef,
    policy: DedupPolicy,
) -> Result<MapArray, KernelError> {
    if keys.len() != values.len() || keys.len() != row_of.len() {
        return Err(internal("str_to_map pairs of different lengths"));
    }
    let hash = hash_all(&[keys.as_ref()], &[keys.len()])?;
    let mut nulls = NullBufferBuilder::new(rows);
    let mut per_row: Vec<Vec<Entry<'_>>> = (0..rows).map(|_| Vec::new()).collect();
    let mut ordered = Vec::with_capacity(row_of.len());
    for (p, r) in row_of.iter().enumerate() {
        let Some(row) = per_row.get_mut(*r) else {
            return Err(internal("a str_to_map pair past its rows"));
        };
        ordered.push(hash_at(&hash, p)?);
        row.push(Entry {
            keys: keys.as_ref(),
            key: p,
            values: values.as_ref(),
            value: p,
        });
    }
    for row in &per_row {
        nulls.append(!row.is_empty());
    }
    assemble(
        entries,
        &Deduped::of(per_row, policy, &ordered)?,
        nulls.finish(),
    )
}

/// The `entries` field of a map of `key_type` to `value_type` (non-null
/// keys, values nullable as `value_nullable`).
fn entries_field_of(key_type: &DataType, value_type: &DataType, value_nullable: bool) -> FieldRef {
    Arc::new(arrow_schema::Field::new(
        "entries",
        DataType::Struct(
            vec![
                arrow_schema::Field::new("key", key_type.clone(), false),
                arrow_schema::Field::new("value", value_type.clone(), value_nullable),
            ]
            .into(),
        ),
        false,
    ))
}

/// A map over `keys` and `values` with `offsets` (one entry per element)
/// and the row `nulls`, for the map higher-order functions that rebuild a
/// map from flattened entries.
///
/// # Errors
/// An engine error when the key and value counts disagree or the offsets
/// do not fit the entries.
pub fn build_map(
    keys: &ArrayRef,
    values: &ArrayRef,
    offsets: OffsetBuffer<i32>,
    nulls: Option<NullBuffer>,
    value_nullable: bool,
) -> Result<MapArray, KernelError> {
    if keys.len() != values.len() {
        return Err(internal(format!(
            "a map over {} keys and {} values",
            keys.len(),
            values.len()
        )));
    }
    let entries = entries_field_of(keys.data_type(), values.data_type(), value_nullable);
    let DataType::Struct(fields) = entries.data_type() else {
        return Err(internal("map entries that are no struct"));
    };
    let body = StructArray::try_new(
        fields.clone(),
        vec![Arc::clone(keys), Arc::clone(values)],
        None,
    )
    .map_err(internal)?;
    MapArray::try_new(entries, offsets, body, nulls, false).map_err(internal)
}

/// `array` with view string and binary carriers read as the plain ones the
/// hash kernel takes (the same bytes hash the same).
fn hashable(array: &dyn Array) -> Result<ArrayRef, KernelError> {
    Ok(match array.data_type() {
        // The hash kernel has no calendar form, and this one is only for dedup:
        // equal intervals mix to one `bigint`.
        t if krishiv_types::arrow::is_calendar_struct(t) => {
            let c = krishiv_types::calendar::Calendars::of(array)
                .ok_or_else(|| internal("a calendar key that is not one".to_owned()))?;
            let mixed: PrimitiveArray<Int64Type> = c
                .iter()
                .map(|v| {
                    v.map(|v| {
                        i64::from(v.months).rotate_left(40)
                            ^ i64::from(v.days).rotate_left(20)
                            ^ v.micros
                    })
                })
                .collect();
            Arc::new(mixed)
        }
        DataType::Utf8View => Arc::new(arrow_array::StringArray::from_iter(
            array.as_string_view().iter(),
        )),
        DataType::BinaryView => Arc::new(arrow_array::BinaryArray::from_iter(
            array.as_binary_view().iter(),
        )),
        DataType::List(field) if has_view(field.data_type()) => {
            let list = array.as_list::<i32>();
            let values = hashable(list.values().as_ref())?;
            let field = Arc::new(arrow_schema::Field::new(
                field.name(),
                values.data_type().clone(),
                field.is_nullable(),
            ));
            Arc::new(
                ListArray::try_new(field, list.offsets().clone(), values, list.nulls().cloned())
                    .map_err(internal)?,
            )
        }
        DataType::Struct(fields) if fields.iter().any(|f| has_view(f.data_type())) => {
            let s = array.as_struct();
            let columns = s
                .columns()
                .iter()
                .map(|c| hashable(c.as_ref()))
                .collect::<Result<Vec<_>, _>>()?;
            let fields: Vec<_> = fields
                .iter()
                .zip(&columns)
                .map(|(f, c)| {
                    arrow_schema::Field::new(f.name(), c.data_type().clone(), f.is_nullable())
                })
                .collect();
            Arc::new(
                StructArray::try_new(fields.into(), columns, s.nulls().cloned())
                    .map_err(internal)?,
            )
        }
        _ => arrow_array::make_array(array.to_data()),
    })
}

/// Whether `t` holds a view string or binary at any depth the hash kernel
/// would meet.
fn has_view(t: &DataType) -> bool {
    match t {
        DataType::Utf8View | DataType::BinaryView => true,
        DataType::List(f) => has_view(f.data_type()),
        DataType::Struct(fs) => fs.iter().any(|f| has_view(f.data_type())),
        _ => false,
    }
}

/// A row index as `u32`: a row that does not fit is an engine error, never
/// a wrap.
fn index32(n: usize) -> Result<u32, KernelError> {
    u32::try_from(n).map_err(|_| internal(format!("a map of {n} entries")))
}

/// Spark's key order over `keys[a]` and `other[b]`: the total order of
/// [`crate::order::cmp_rows`] (NaN equals NaN, `-0.0` equals `0.0`, views
/// and nested keys included), where equal means the same map key.
fn key_order(
    keys: &dyn Array,
    a: usize,
    other: &dyn Array,
    b: usize,
) -> Result<std::cmp::Ordering, KernelError> {
    crate::order::cmp_rows(keys, a, other, b)
}

/// The deduplicated entries of one map's `len` keys (the first `len` of
/// `keys`): the kept key indices and the value indices, which differ under
/// `LAST_WIN` (the last value wins but the key keeps its first-seen
/// position).
///
/// Keys are grouped by sorting their indices once under `key_order`, so
/// nested and view keys need no hashing.
///
/// # Errors
/// `NULL_MAP_KEY` for a null key, `DUPLICATED_MAP_KEY` for a repeat under
/// `EXCEPTION` (the earlier of the two failures in key order, as a scan
/// would meet it), or an engine error for a key type [`supports_key`]
/// refuses.
pub fn dedup_entries(
    keys: &dyn Array,
    len: usize,
    last_win: bool,
) -> Result<(Vec<u32>, Vec<u32>), KernelError> {
    if keys.len() < len {
        return Err(internal(format!(
            "{len} keys over an array of {}",
            keys.len()
        )));
    }
    if let Some(null) = (0..len).find(|&e| keys.is_null(e)) {
        // A scan stops at the first null unless a duplicate comes first.
        let mut head = Vec::with_capacity(null);
        head.extend(0..null);
        if let Some(dup) = first_duplicate(keys, &head)?
            && !last_win
        {
            return Err(KernelError::with_row(None, duplicate_key(keys, dup)?));
        }
        return Err(KernelError::with_row(None, NullMapKey {}.into()));
    }
    let mut order: Vec<usize> = (0..len).collect();
    let mut failure = None;
    order.sort_by(|&a, &b| match key_order(keys, a, keys, b) {
        Ok(ord) => ord.then(a.cmp(&b)),
        Err(e) => {
            failure.get_or_insert(e);
            std::cmp::Ordering::Equal
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    // Group boundaries: a key equal to its sorted predecessor repeats it.
    let mut first_of: Vec<usize> = (0..len).collect();
    let mut last_of: Vec<usize> = (0..len).collect();
    let mut duplicate: Option<usize> = None;
    let mut group_first = 0;
    for pair in order.windows(2) {
        let (prev, cur) = (pair[0], pair[1]);
        if key_order(keys, prev, keys, cur)? == std::cmp::Ordering::Equal {
            first_of[cur] = group_first;
            last_of[group_first] = cur;
            duplicate = Some(duplicate.map_or(cur, |d| d.min(cur)));
        } else {
            group_first = cur;
        }
    }
    if let Some(dup) = duplicate
        && !last_win
    {
        return Err(KernelError::with_row(None, duplicate_key(keys, dup)?));
    }
    let mut kept: Vec<u32> = Vec::with_capacity(len);
    let mut values: Vec<u32> = Vec::with_capacity(len);
    for e in 0..len {
        if first_of[e] == e {
            kept.push(index32(e)?);
            values.push(index32(last_of[e])?);
        }
    }
    Ok((kept, values))
}

/// The earliest index among `head` whose key repeats an earlier one.
fn first_duplicate(keys: &dyn Array, head: &[usize]) -> Result<Option<usize>, KernelError> {
    for (n, &e) in head.iter().enumerate() {
        for &p in &head[..n] {
            if key_order(keys, p, keys, e)? == std::cmp::Ordering::Equal {
                return Ok(Some(e));
            }
        }
    }
    Ok(None)
}

/// One key of the union of two maps' key sets: the key's index into the
/// concatenated keys, and each map's value index (`None` for a key the map
/// does not hold). A map holding a key twice contributes its first entry
/// (Spark's `MapZipWith`).
pub type UnionEntry = (u32, Option<u32>, Option<u32>);

/// The union of two maps' key sets, in first-seen order (see
/// [`UnionEntry`]); `keys1` and `keys2` hold each map's keys in order. The
/// second map's key index counts past the first map's `len1` keys, while
/// each value index counts within its own map.
///
/// # Errors
/// `NULL_MAP_KEY` for a null key, or an engine error for a key type
/// [`supports_key`] refuses.
pub fn union_keys(
    keys1: &dyn Array,
    len1: usize,
    keys2: &dyn Array,
    len2: usize,
) -> Result<Vec<UnionEntry>, KernelError> {
    if keys1.len() < len1 || keys2.len() < len2 {
        return Err(internal(format!(
            "zip over {len1} and {len2} keys of {} and {}",
            keys1.len(),
            keys2.len()
        )));
    }
    if (0..len1).any(|e| keys1.is_null(e)) || (0..len2).any(|e| keys2.is_null(e)) {
        return Err(KernelError::with_row(None, NullMapKey {}.into()));
    }
    // Entries as (map, index), grouped by key under [`key_order`]; within
    // a group the first map's entries come first, each in key order.
    let at = |w: usize| if w == 0 { keys1 } else { keys2 };
    let mut entries: Vec<(usize, usize)> = (0..len1)
        .map(|e| (0, e))
        .chain((0..len2).map(|e| (1, e)))
        .collect();
    let mut failure = None;
    entries.sort_by(|&(wa, a), &(wb, b)| match key_order(at(wa), a, at(wb), b) {
        Ok(ord) => ord.then((wa, a).cmp(&(wb, b))),
        Err(e) => {
            failure.get_or_insert(e);
            std::cmp::Ordering::Equal
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    let mut groups: Vec<(usize, UnionEntry)> = Vec::new();
    let mut previous: Option<(usize, usize)> = None;
    for &(w, e) in &entries {
        let same = match previous {
            Some((pw, pe)) => key_order(at(pw), pe, at(w), e)? == std::cmp::Ordering::Equal,
            None => false,
        };
        previous = Some((w, e));
        let value = Some(index32(e)?);
        if same {
            let (_, entry) = groups
                .last_mut()
                .ok_or_else(|| internal("a map key group without a first entry"))?;
            let slot = if w == 0 { &mut entry.1 } else { &mut entry.2 };
            if slot.is_none() {
                *slot = value;
            }
        } else {
            // First-seen position: the first map's keys, then the second's.
            let seen = if w == 0 { e } else { len1.saturating_add(e) };
            let key = index32(seen)?;
            let entry = if w == 0 {
                (key, value, None)
            } else {
                (key, None, value)
            };
            groups.push((seen, entry));
        }
    }
    groups.sort_by_key(|&(seen, _)| seen);
    Ok(groups.into_iter().map(|(_, entry)| entry).collect())
}

#[cfg(test)]
mod tests;
