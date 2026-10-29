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
    ArrayBuilder, BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder, Float64Builder,
    Int8Builder, Int16Builder, Int32Builder, Int64Builder, ListBuilder, MapBuilder, NullBuilder,
    StringViewBuilder, StructBuilder, TimestampMicrosecondBuilder, UInt8Builder, UInt16Builder,
    UInt32Builder, UInt64Builder, make_builder,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, GenericListArray, ListArray, MapArray, StructArray,
};
use arrow_buffer::{ArrowNativeType, NullBuffer, NullBufferBuilder, OffsetBuffer};
use arrow_schema::{DataType, FieldRef, TimeUnit};
use krishiv_common::error::{DuplicatedMapKey, KrishivError, LegacyErrorTemp2128, NullMapKey};

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
/// binary, dates, microsecond timestamps, and arrays and structs of those).
/// Anything else (a map, dictionary, union, interval, time of day, decimal
/// beyond 128 bits, non-microsecond timestamps) cannot arrive here; the
/// binding refuses it first.
#[must_use]
pub fn supports_key(t: &DataType) -> bool {
    match t {
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
        | DataType::Decimal128(..)
        | DataType::Utf8
        | DataType::Utf8View
        | DataType::LargeUtf8
        | DataType::Binary
        | DataType::BinaryView
        | DataType::LargeBinary
        | DataType::Date32
        | DataType::Timestamp(TimeUnit::Microsecond, _) => true,
        DataType::List(f) | DataType::LargeList(f) => supports_key(f.data_type()),
        DataType::Struct(fs) => fs.iter().all(|f| supports_value(f.data_type())),
        _ => false,
    }
}

/// Whether the kernel stores values of Arrow type `t`: the key types, plus
/// maps of those (a map value may itself be a map).
#[must_use]
pub fn supports_value(t: &DataType) -> bool {
    if supports_key(t) {
        return true;
    }
    match t {
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(fs) if fs.len() == 2 => {
                supports_key(fs[0].data_type()) && supports_value(fs[1].data_type())
            }
            _ => false,
        },
        _ => false,
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
        T::Decimal128(_, scale) => {
            decimal_text(keys.as_primitive::<Decimal128Type>().value(key), *scale)
        }
        T::Binary | T::BinaryView | T::LargeBinary => Ok(format!("{:?}", binary_value(keys, key))),
        T::Date32 => Ok(keys.as_primitive::<Date32Type>().value(key).to_string()),
        T::Timestamp(..) => Ok(keys
            .as_primitive::<TimestampMicrosecondType>()
            .value(key)
            .to_string()),
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
        let plain = hashable(*a);
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
fn hashable(array: &dyn Array) -> ArrayRef {
    match array.data_type() {
        DataType::Utf8View => Arc::new(arrow_array::StringArray::from_iter(
            array.as_string_view().iter(),
        )),
        DataType::BinaryView => Arc::new(arrow_array::BinaryArray::from_iter(
            array.as_binary_view().iter(),
        )),
        _ => arrow_array::make_array(array.to_data()),
    }
}

/// The Murmur3 hashes of the first `len` keys (empty for none).
fn hash_keys(keys: &dyn Array, len: usize) -> Result<Vec<i32>, KernelError> {
    if keys.len() < len {
        return Err(internal(format!(
            "{len} keys over an array of {}",
            keys.len()
        )));
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    // The hash kernel reads plain string and binary carriers; the view
    // carriers (Spark's `STRING` here) hash as the same bytes.
    let sliced = hashable(keys.slice(0, len).as_ref());
    Ok(hash::hash(&[sliced.as_ref()], len)?.values().to_vec())
}

/// A row index as `u32`: a row that does not fit is an engine error, never
/// a wrap.
fn index32(n: usize) -> Result<u32, KernelError> {
    u32::try_from(n).map_err(|_| internal(format!("a map of {n} entries")))
}

/// The deduplicated entries of one map's `len` keys (the first `len` of
/// `keys`): the kept key indices and the value indices, which differ under
/// `LAST_WIN` (the last value wins but the key keeps its first-seen
/// position).
///
/// # Errors
/// `NULL_MAP_KEY` for a null key, `DUPLICATED_MAP_KEY` for a repeat under
/// `EXCEPTION`, or an engine error for a key type [`supports_key`] refuses.
pub fn dedup_entries(
    keys: &dyn Array,
    len: usize,
    last_win: bool,
) -> Result<(Vec<u32>, Vec<u32>), KernelError> {
    let hash = hash_keys(keys, len)?;
    let mut seen: HashMap<i32, Vec<usize>> = HashMap::with_capacity(len);
    let mut kept: Vec<u32> = Vec::with_capacity(len);
    let mut values: Vec<u32> = Vec::with_capacity(len);
    for (e, h) in hash.iter().enumerate() {
        if keys.is_null(e) {
            return Err(KernelError::with_row(None, NullMapKey {}.into()));
        }
        let mut found = None;
        if let Some(same) = seen.get(h) {
            for &p in same {
                if keys_equal(keys, usize::try_from(kept[p]).map_err(internal)?, keys, e)? {
                    found = Some(p);
                    break;
                }
            }
        }
        let at = index32(e)?;
        match found {
            Some(p) if last_win => values[p] = at,
            Some(_) => return Err(KernelError::with_row(None, duplicate_key(keys, e)?)),
            None => {
                seen.entry(*h).or_default().push(kept.len());
                kept.push(at);
                values.push(at);
            }
        }
    }
    Ok((kept, values))
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
    let h1 = hash_keys(keys1, len1)?;
    let h2 = hash_keys(keys2, len2)?;
    // Slot indices of `out` by hash; the owner of a slot's key is the first
    // map when its key index is below `len1`.
    let mut seen: HashMap<i32, Vec<usize>> = HashMap::with_capacity(len1.saturating_add(len2));
    let mut out: Vec<UnionEntry> = Vec::with_capacity(len1.saturating_add(len2));
    for (second, (keys, hashes)) in [(keys1, &h1), (keys2, &h2)].into_iter().enumerate() {
        for (e, h) in hashes.iter().enumerate() {
            if keys.is_null(e) {
                return Err(KernelError::with_row(None, NullMapKey {}.into()));
            }
            let mut found = None;
            if let Some(same) = seen.get(h) {
                for &p in same {
                    let (key, ..) = out[p];
                    let key = usize::try_from(key).map_err(internal)?;
                    let equal = if key < len1 {
                        keys_equal(keys1, key, keys, e)?
                    } else {
                        keys_equal(keys2, key.saturating_sub(len1), keys, e)?
                    };
                    if equal {
                        found = Some(p);
                        break;
                    }
                }
            }
            let value = Some(index32(e)?);
            match found {
                Some(p) => {
                    let slot = if second == 0 {
                        &mut out[p].1
                    } else {
                        &mut out[p].2
                    };
                    if slot.is_none() {
                        *slot = value;
                    }
                }
                None => {
                    let key = if second == 0 {
                        index32(e)?
                    } else {
                        index32(
                            len1.checked_add(e)
                                .ok_or_else(|| internal("map keys overflow"))?,
                        )?
                    };
                    seen.entry(*h).or_default().push(out.len());
                    out.push(if second == 0 {
                        (key, value, None)
                    } else {
                        (key, None, value)
                    });
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Decimal128Array, Int32Array, StringArray, StringViewArray};
    use arrow_buffer::NullBufferBuilder;
    use arrow_schema::Field;
    use krishiv_common::error::ErrorClass;

    fn test_entries(key: DataType, value: DataType, value_nullable: bool) -> FieldRef {
        let entries = DataType::Struct(
            vec![
                Field::new("key", key, false),
                Field::new("value", value, value_nullable),
            ]
            .into(),
        );
        Arc::new(Field::new("entries", entries, false))
    }

    fn strings(v: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringArray::from(v.to_vec()))
    }

    fn views(v: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringViewArray::from(
            v.iter().map(|s| s.as_deref()).collect::<Vec<_>>(),
        ))
    }

    fn ints(v: &[Option<i32>]) -> ArrayRef {
        Arc::new(Int32Array::from(v.to_vec()))
    }

    fn int_field_entries() -> FieldRef {
        test_entries(DataType::Int32, DataType::Utf8View, true)
    }

    type Pairs = Vec<Option<Vec<(String, Option<String>)>>>;

    /// The `(key, value)` pairs of every map row, rendered for assertions.
    fn pairs(m: &MapArray) -> Pairs {
        (0..m.len())
            .map(|r| {
                if m.is_null(r) {
                    return None;
                }
                let (from, to) = map_range(m, r).unwrap();
                let mut row = Vec::new();
                for p in from..to {
                    let k = key_text(m.keys().as_ref(), p).expect("key text renders");
                    let v = if m.values().is_null(p) {
                        None
                    } else {
                        Some(string_value(m.values().as_ref(), p).to_owned())
                    };
                    row.push((k, v));
                }
                Some(row)
            })
            .collect()
    }

    /// A `MapArray` input over `entries`, with `counts[r]` entries in row
    /// `r` (`null_rows[r]` is null).
    fn map_input(
        keys: ArrayRef,
        values: ArrayRef,
        entries: &FieldRef,
        counts: &[usize],
        null_rows: &[bool],
    ) -> MapArray {
        let mut offsets = vec![0_i32];
        for (n, null) in counts.iter().zip(null_rows) {
            let next = if *null {
                offsets.last().copied().unwrap_or(0)
            } else {
                offsets
                    .last()
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(i32::try_from(*n).unwrap_or(0))
            };
            offsets.push(next);
        }
        let fields = match entries.data_type() {
            DataType::Struct(fs) => fs.clone(),
            _ => unreachable!(),
        };
        let structs = StructArray::try_new(fields, vec![keys, values], None).unwrap();
        let mut nulls = NullBufferBuilder::new(counts.len());
        for null in null_rows {
            nulls.append(!null);
        }
        MapArray::try_new(
            Arc::clone(entries),
            OffsetBuffer::new(offsets.into()),
            structs,
            nulls.finish(),
            false,
        )
        .unwrap()
    }

    fn list_of(values: ArrayRef, breaks: &[usize], nulls: &[bool]) -> ArrayRef {
        let mut offsets = vec![0_i32];
        for (n, null) in breaks.iter().zip(nulls) {
            let next = if *null {
                offsets.last().copied().unwrap_or(0)
            } else {
                offsets
                    .last()
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(i32::try_from(*n).unwrap_or(0))
            };
            offsets.push(next);
        }
        let mut valid = NullBufferBuilder::new(breaks.len());
        for null in nulls {
            valid.append(!null);
        }
        Arc::new(ListArray::new(
            Arc::new(Field::new("item", values.data_type().clone(), true)),
            OffsetBuffer::new(offsets.into()),
            values,
            valid.finish(),
        ))
    }

    #[test]
    fn equal_keys_cover_every_carrier() {
        assert!(keys_equal(&ints(&[Some(1)]), 0, &ints(&[Some(1)]), 0).unwrap());
        assert!(!keys_equal(&ints(&[Some(1)]), 0, &ints(&[Some(2)]), 0).unwrap());
        // A null equals only a null.
        assert!(keys_equal(&ints(&[None]), 0, &ints(&[None]), 0).unwrap());
        assert!(!keys_equal(&ints(&[None]), 0, &ints(&[Some(1)]), 0).unwrap());
        // Strings compare byte-wise across carriers.
        assert!(keys_equal(&strings(&[Some("a")]), 0, &views(&[Some("a")]), 0).unwrap());
        assert!(!keys_equal(&strings(&[Some("a")]), 0, &views(&[Some("b")]), 0).unwrap());
        // Floats: NaN equals NaN, -0.0 equals 0.0.
        let nan = Arc::new(arrow_array::Float64Array::from(vec![f64::NAN])) as ArrayRef;
        assert!(keys_equal(nan.as_ref(), 0, nan.as_ref(), 0).unwrap());
        let (neg, pos) = (
            Arc::new(arrow_array::Float64Array::from(vec![-0.0])) as ArrayRef,
            Arc::new(arrow_array::Float64Array::from(vec![0.0])) as ArrayRef,
        );
        assert!(keys_equal(neg.as_ref(), 0, pos.as_ref(), 0).unwrap());
        // Decimals compare by unscaled value.
        let dec = |u: i128| {
            Arc::new(
                Decimal128Array::from(vec![u])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ) as ArrayRef
        };
        assert!(keys_equal(&dec(100), 0, &dec(100), 0).unwrap());
        assert!(!keys_equal(&dec(100), 0, &dec(101), 0).unwrap());
        // Nested values compare element by element.
        let l1 = list_of(ints(&[Some(1), None]), &[2], &[false]);
        let l2 = list_of(ints(&[Some(1), None]), &[2], &[false]);
        let l3 = list_of(ints(&[Some(1), Some(2)]), &[2], &[false]);
        assert!(keys_equal(l1.as_ref(), 0, l2.as_ref(), 0).unwrap());
        assert!(!keys_equal(l1.as_ref(), 0, l3.as_ref(), 0).unwrap());
        // Types that cannot meet are an internal error, not a panic.
        assert!(keys_equal(&ints(&[Some(1)]), 0, &strings(&[Some("1")]), 0).is_err());
    }

    #[test]
    fn create_map_builds_a_map_per_row() {
        let keys = [ints(&[Some(1), Some(3)]), ints(&[Some(2), Some(4)])];
        let values = [
            strings(&[Some("a"), None]),
            strings(&[Some("b"), Some("d")]),
        ];
        let m = create_map(
            &keys,
            &values,
            2,
            &int_field_entries(),
            DedupPolicy::Exception,
        )
        .unwrap();
        assert_eq!(
            pairs(&m),
            [
                Some(vec![
                    ("1".to_owned(), Some("a".to_owned())),
                    ("2".to_owned(), Some("b".to_owned())),
                ]),
                Some(vec![
                    ("3".to_owned(), None),
                    ("4".to_owned(), Some("d".to_owned())),
                ]),
            ]
        );
        // The output carries the entries field: struct `key` (not null) and
        // `value`, over `Utf8View` values.
        assert_eq!(m.data_type(), &DataType::Map(int_field_entries(), false));
        assert_eq!(m.value_offsets().len(), 3);
    }

    #[test]
    fn create_map_rejects_null_and_duplicate_keys() {
        let keys = [ints(&[Some(1), None])];
        let values = [strings(&[Some("a"), Some("b")])];
        let e = create_map(
            &keys,
            &values,
            2,
            &int_field_entries(),
            DedupPolicy::Exception,
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::NullMapKey);
        assert_eq!(e.row(), Some(1));

        let keys = [ints(&[Some(1)]), ints(&[Some(1)])];
        let values = [strings(&[Some("a")]), strings(&[Some("b")])];
        let e = create_map(
            &keys,
            &values,
            1,
            &int_field_entries(),
            DedupPolicy::Exception,
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);
        assert_eq!(e.row(), Some(0));
        assert!(
            e.error()
                .message()
                .contains("Duplicate map key 1 was found")
        );
        let params: Vec<(&str, String)> = e
            .error()
            .message_parameters()
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        assert!(params.contains(&("key", "1".to_owned())));
        assert!(params.contains(&(
            "mapKeyDedupPolicy",
            "\"spark.sql.mapKeyDedupPolicy\"".to_owned()
        )));

        // `LAST_WIN` keeps the first position with the last value.
        let m = create_map(
            &keys,
            &values,
            1,
            &int_field_entries(),
            DedupPolicy::LastWin,
        )
        .unwrap();
        assert_eq!(
            pairs(&m),
            [Some(vec![("1".to_owned(), Some("b".to_owned()))])]
        );
    }

    #[test]
    fn create_map_empty_is_an_empty_map_per_row() {
        let m = create_map(&[], &[], 2, &int_field_entries(), DedupPolicy::Exception).unwrap();
        assert_eq!(pairs(&m), [Some(vec![]), Some(vec![])]);
    }

    #[test]
    fn create_map_hashes_align_with_rows() {
        // The dedup hash of pair `i`, row `r` is read row-major: a repeat
        // in a later row is still found (a column-major read would miss
        // it, as the hashes differ).
        let keys = [ints(&[Some(1), Some(1)]), ints(&[Some(2), Some(1)])];
        let values = [
            strings(&[Some("a"), Some("b")]),
            strings(&[Some("c"), Some("d")]),
        ];
        let e = create_map(
            &keys,
            &values,
            2,
            &int_field_entries(),
            DedupPolicy::Exception,
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);
        assert_eq!(e.row(), Some(1));
    }

    #[test]
    fn create_map_nests_arrays_structs_and_maps() {
        // Array keys.
        let k1 = list_of(ints(&[Some(1), Some(2)]), &[2], &[false]);
        let k2 = list_of(ints(&[Some(1), Some(2)]), &[2], &[false]);
        let v = [strings(&[Some("a")]), strings(&[Some("b")])];
        let e = create_map(
            &[Arc::clone(&k1), Arc::clone(&k2)],
            &v,
            1,
            &test_entries(
                DataType::List(Arc::new(Field::new("element", DataType::Int32, true))),
                DataType::Utf8View,
                true,
            ),
            DedupPolicy::Exception,
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);

        // Map values.
        let inner = create_map(
            &[ints(&[Some(7)])],
            &[strings(&[Some("x")])],
            1,
            &int_field_entries(),
            DedupPolicy::Exception,
        )
        .unwrap();
        let m = create_map(
            &[strings(&[Some("m")])],
            &[Arc::new(inner) as ArrayRef],
            1,
            &test_entries(
                DataType::Utf8View,
                DataType::Map(int_field_entries(), false),
                false,
            ),
            DedupPolicy::Exception,
        )
        .unwrap();
        assert_eq!(m.len(), 1);
        assert!(!m.is_null(0));
        let (from, to) = map_range(&m, 0).unwrap();
        assert_eq!((to - from), 1);
    }

    #[test]
    fn create_map_scales_to_thousands_of_rows() {
        let rows = 2_000_usize;
        let keys = [Arc::new(Int32Array::from_iter_values(
            0..i32::try_from(rows).unwrap_or(0),
        )) as ArrayRef];
        let values = [strings(&vec![Some("v"); rows])];
        let m = create_map(
            &keys,
            &values,
            rows,
            &int_field_entries(),
            DedupPolicy::Exception,
        )
        .unwrap();
        assert_eq!(m.len(), rows);
        assert_eq!(
            m.value_offsets().last().copied().unwrap_or(0),
            i32::try_from(rows).unwrap_or(-1)
        );
        let (from, to) = map_range(&m, rows - 1).unwrap();
        assert_eq!((to - from), 1);
        assert_eq!(
            key_text(m.keys().as_ref(), from).expect("key text renders"),
            (i32::try_from(rows).unwrap_or(0).saturating_sub(1)).to_string()
        );
    }

    #[test]
    fn concat_merges_with_nulls_and_policy() {
        let entries = int_field_entries();
        let left = map_input(
            ints(&[Some(1)]),
            views(&[Some("a")]),
            &entries,
            &[1],
            &[false],
        );
        let right = Arc::new(map_input(
            ints(&[Some(1)]),
            views(&[Some("B")]),
            &entries,
            &[1],
            &[false],
        )) as ArrayRef;
        // Key 1 in both maps is a duplicate under `EXCEPTION`.
        let e = concat(
            &[Arc::new(left.clone()) as ArrayRef, Arc::clone(&right)],
            1,
            &entries,
            DedupPolicy::Exception,
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);
        assert_eq!(e.row(), Some(0));

        // Under `LAST_WIN` the later map's value wins in place.
        let m = concat(
            &[Arc::new(left.clone()) as ArrayRef, Arc::clone(&right)],
            1,
            &entries,
            DedupPolicy::LastWin,
        )
        .unwrap();
        assert_eq!(
            pairs(&m),
            [Some(vec![("1".to_owned(), Some("B".to_owned()))])]
        );
        let _ = right;

        // Disjoint maps merge; a null map voids its row under either
        // policy, whatever the other maps hold.
        let other = Arc::new(map_input(
            ints(&[Some(2)]),
            views(&[Some("b")]),
            &entries,
            &[1],
            &[false],
        )) as ArrayRef;
        let m = concat(
            &[Arc::new(left) as ArrayRef, Arc::clone(&other)],
            1,
            &entries,
            DedupPolicy::Exception,
        )
        .unwrap();
        assert_eq!(
            pairs(&m),
            [Some(vec![
                ("1".to_owned(), Some("a".to_owned())),
                ("2".to_owned(), Some("b".to_owned())),
            ])]
        );

        let void = Arc::new(map_input(ints(&[]), views(&[]), &entries, &[0], &[true])) as ArrayRef;
        for policy in [DedupPolicy::Exception, DedupPolicy::LastWin] {
            let m = concat(
                &[Arc::clone(&void), Arc::clone(&other)],
                1,
                &entries,
                policy,
            )
            .unwrap();
            assert_eq!(pairs(&m), [None]);
        }

        // No maps is an empty map per row.
        let m = concat(&[], 2, &entries, DedupPolicy::Exception).unwrap();
        assert_eq!(pairs(&m), [Some(vec![]), Some(vec![])]);
    }

    #[test]
    fn contains_key_scans_with_nulls() {
        let entries = int_field_entries();
        let map = map_input(
            ints(&[Some(1), Some(2), Some(0)]),
            views(&[Some("a"), Some("b"), Some("c")]),
            &entries,
            &[2, 0, 0],
            &[false, false, true],
        );
        let out = contains_key(
            &(Arc::new(map) as ArrayRef),
            &ints(&[Some(2), Some(1), None]),
        )
        .unwrap();
        assert_eq!(out, BooleanArray::from(vec![Some(true), Some(false), None]));
    }

    #[test]
    fn contains_key_normalises_floats() {
        // `-0.0` finds `0.0`, NaN finds NaN: Spark's key equality.
        let entries = test_entries(DataType::Float64, DataType::Utf8View, true);
        let map = map_input(
            Arc::new(arrow_array::Float64Array::from(vec![Some(0.0)])) as ArrayRef,
            views(&[Some("zero")]),
            &entries,
            &[1],
            &[false],
        );
        let out = contains_key(
            &(Arc::new(map) as ArrayRef),
            &(Arc::new(arrow_array::Float64Array::from(vec![Some(-0.0)])) as ArrayRef),
        )
        .unwrap();
        assert_eq!(out, BooleanArray::from(vec![Some(true)]));
    }

    #[test]
    fn projections_share_children() {
        let map_field = test_entries(DataType::Int32, DataType::Utf8View, true);
        let map = map_input(
            ints(&[Some(1), Some(2)]),
            views(&[Some("a"), None]),
            &map_field,
            &[2, 0],
            &[false, true],
        );
        let map = Arc::new(map) as ArrayRef;
        let key_field = Arc::new(Field::new("element", DataType::Int32, true));
        let k = keys(&map, &key_field).unwrap();
        let l = k.as_list::<i32>();
        assert_eq!(l.len(), 2);
        assert!(l.is_null(1));
        assert_eq!(l.value_offsets(), &[0_i32, 2, 2][..]);

        let value_field = Arc::new(Field::new("element", DataType::Utf8View, true));
        let v = values(&map, &value_field).unwrap();
        assert_eq!(v.len(), 2);

        let entry_field = Arc::new(Field::new(
            "element",
            DataType::Struct(
                vec![
                    Field::new("key", DataType::Int32, false),
                    Field::new("value", DataType::Utf8View, true),
                ]
                .into(),
            ),
            false,
        ));
        let e = entries(&map, &entry_field).unwrap();
        let l = e.as_list::<i32>();
        assert_eq!(l.len(), 2);
        let s = l.values().as_struct();
        assert_eq!(s.num_columns(), 2);
        assert_eq!(s.column(0).len(), 2);
    }

    #[test]
    fn from_arrays_maps_slices_with_policy() {
        let entries = test_entries(DataType::Decimal128(10, 2), DataType::Utf8View, true);
        let key_values = Arc::new(
            Decimal128Array::from(vec![Some(100), Some(200), Some(300)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ) as ArrayRef;
        let keys = list_of(key_values, &[2, 1, 0], &[false, false, true]);
        let values = list_of(
            strings(&[Some("a"), Some("b"), Some("c")]),
            &[2, 1, 0],
            &[false, false, true],
        );
        let m = from_arrays(&keys, &values, 3, &entries, DedupPolicy::Exception).unwrap();
        assert_eq!(m.len(), 3);
        assert!(m.is_null(2));
        let (from, to) = map_range(&m, 0).unwrap();
        assert_eq!((to - from), 2);

        // Different slice lengths are `_LEGACY_ERROR_TEMP_2128`.
        let keys = list_of(ints(&[Some(1)]), &[1], &[false]);
        let values = list_of(strings(&[Some("a"), Some("b")]), &[2], &[false]);
        let e = from_arrays(
            &keys,
            &values,
            1,
            &int_field_entries(),
            DedupPolicy::Exception,
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::LegacyErrorTemp2128);
        assert_eq!(e.row(), Some(0));

        // A null key element is `NULL_MAP_KEY`, wherever it sits.
        let keys = list_of(ints(&[Some(1), None]), &[2], &[false]);
        let values = list_of(strings(&[Some("a"), Some("b")]), &[2], &[false]);
        let e = from_arrays(
            &keys,
            &values,
            1,
            &int_field_entries(),
            DedupPolicy::Exception,
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::NullMapKey);

        // Duplicate keys follow the policy.
        let keys = list_of(ints(&[Some(1), Some(1)]), &[2], &[false]);
        let values = list_of(strings(&[Some("a"), Some("b")]), &[2], &[false]);
        let e = from_arrays(
            &keys,
            &values,
            1,
            &int_field_entries(),
            DedupPolicy::Exception,
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);
        let m = from_arrays(
            &keys,
            &values,
            1,
            &int_field_entries(),
            DedupPolicy::LastWin,
        )
        .unwrap();
        assert_eq!(
            pairs(&m),
            [Some(vec![("1".to_owned(), Some("b".to_owned()))])]
        );
    }

    #[test]
    fn null_struct_values_append_typed_nulls() {
        let struct_ty = DataType::Struct(vec![Field::new("a", DataType::Int32, true)].into());
        let struct_fields = match &struct_ty {
            DataType::Struct(fs) => fs.clone(),
            _ => unreachable!(),
        };
        // A null struct as a map value: the map holds a null.
        let null_struct = Arc::new(
            StructArray::try_new(
                struct_fields.clone(),
                vec![ints(&[Some(1)])],
                Some(arrow_buffer::NullBuffer::from(vec![false])),
            )
            .unwrap(),
        ) as ArrayRef;
        let entries = test_entries(DataType::Int32, struct_ty.clone(), true);
        let m = create_map(
            &[ints(&[Some(1)])],
            &[null_struct],
            1,
            &entries,
            DedupPolicy::Exception,
        )
        .unwrap();
        assert!(!m.is_null(0));
        assert!(m.values().is_null(0));

        // A null struct element of a list value: the list keeps a null.
        let inner = Arc::new(
            StructArray::try_new(
                struct_fields,
                vec![ints(&[Some(1), Some(2)])],
                Some(arrow_buffer::NullBuffer::from(vec![true, false])),
            )
            .unwrap(),
        ) as ArrayRef;
        let list_value = list_of(inner, &[2], &[false]);
        let list_ty = DataType::List(Arc::new(Field::new("element", struct_ty, true)));
        let entries = test_entries(DataType::Int32, list_ty, true);
        let m = create_map(
            &[ints(&[Some(1)])],
            &[list_value],
            1,
            &entries,
            DedupPolicy::Exception,
        )
        .unwrap();
        assert!(!m.is_null(0));
        let lists = m.values().as_list::<i32>();
        assert!(!lists.is_null(0));
        assert!(lists.values().as_struct().is_null(1));
    }

    #[test]
    fn from_entries_maps_struct_slices() {
        let fields: arrow_schema::Fields = vec![
            Field::new("key", DataType::Int32, true),
            Field::new("value", DataType::Utf8View, true),
        ]
        .into();
        let structs = Arc::new(
            StructArray::try_new(
                fields.clone(),
                vec![ints(&[Some(1), Some(2)]), views(&[Some("a"), Some("b")])],
                None,
            )
            .unwrap(),
        ) as ArrayRef;
        let array = list_of(structs, &[2, 0], &[false, true]);
        let entries = test_entries(DataType::Int32, DataType::Utf8View, true);
        let m = from_entries(&array, 2, &entries, DedupPolicy::Exception).unwrap();
        assert_eq!(m.len(), 2);
        assert!(m.is_null(1));

        // A null element nulls only its own row.
        let values = Arc::new(
            StructArray::try_new(
                fields.clone(),
                vec![ints(&[Some(9)]), views(&[Some("z")])],
                Some(arrow_buffer::NullBuffer::from(vec![false])),
            )
            .unwrap(),
        ) as ArrayRef;
        let array = list_of(values, &[1, 0], &[false, false]);
        let m = from_entries(&array, 2, &entries, DedupPolicy::Exception).unwrap();
        assert!(m.is_null(0));
        assert!(!m.is_null(1));
        assert_eq!(pairs(&m), [None, Some(vec![])]);

        // A null key field is `NULL_MAP_KEY`.
        let structs = Arc::new(
            StructArray::try_new(fields, vec![ints(&[None]), views(&[Some("a")])], None).unwrap(),
        ) as ArrayRef;
        let array = list_of(structs, &[1], &[false]);
        let e = from_entries(&array, 1, &entries, DedupPolicy::Exception).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::NullMapKey);

        // Entries that are not two-field structs are refused.
        let bad = list_of(ints(&[Some(1)]), &[1], &[false]);
        assert!(from_entries(&bad, 1, &entries, DedupPolicy::Exception).is_err());
    }

    #[test]
    fn from_pairs_rows_empty_and_null() {
        let entries = test_entries(DataType::Utf8View, DataType::Utf8View, true);
        let m = from_pairs(
            &strings(&[Some("a"), Some("b"), Some("c")]),
            &strings(&[Some("1"), None, Some("3")]),
            &[0, 0, 2],
            3,
            &entries,
            DedupPolicy::Exception,
        )
        .unwrap();
        assert_eq!(
            pairs(&m),
            [
                Some(vec![
                    ("a".to_owned(), Some("1".to_owned())),
                    ("b".to_owned(), None),
                ]),
                None,
                Some(vec![("c".to_owned(), Some("3".to_owned()))]),
            ]
        );
    }

    #[test]
    fn policy_parses_and_shapes_hold() {
        assert_eq!(DedupPolicy::of(None), DedupPolicy::Exception);
        assert_eq!(DedupPolicy::of(Some("EXCEPTION")), DedupPolicy::Exception);
        assert_eq!(DedupPolicy::of(Some("LAST_WIN")), DedupPolicy::LastWin);
        assert_eq!(DedupPolicy::of(Some("other")), DedupPolicy::Exception);
        // Every key type the analyzer can produce is supported.
        for t in [
            DataType::Boolean,
            DataType::Int32,
            DataType::Float64,
            DataType::Decimal128(10, 2),
            DataType::Utf8View,
            DataType::BinaryView,
            DataType::Date32,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::List(Arc::new(Field::new("element", DataType::Int32, true))),
        ] {
            assert!(supports_key(&t), "{t}");
            assert!(supports_value(&t), "{t}");
        }
        assert!(supports_value(&DataType::Map(int_field_entries(), false)));
        assert!(!supports_key(&DataType::Map(int_field_entries(), false)));
        assert!(!supports_key(&DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8View),
        )));
    }

    #[test]
    fn decimal_text_renders() {
        assert_eq!(decimal_text(100, 2).expect("decimal text"), "1.00");
        assert_eq!(decimal_text(-5, 3).expect("decimal text"), "-0.005");
        assert_eq!(decimal_text(0, 0).expect("decimal text"), "0");
        assert_eq!(decimal_text(12345, 0).expect("decimal text"), "12345");
    }

    #[test]
    fn large_list_keys_compare() {
        use arrow_array::builder::{Int32Builder, LargeListBuilder};
        let mut a = LargeListBuilder::new(Int32Builder::new());
        a.values().append_value(1);
        a.append(true);
        let mut b = LargeListBuilder::new(Int32Builder::new());
        b.values().append_value(1);
        b.append(true);
        let (x, y) = (a.finish(), b.finish());
        assert!(keys_equal(&x, 0, &y, 0).unwrap());
        let m = create_map(
            &[Arc::new(x) as ArrayRef],
            &[strings(&[Some("a")])],
            1,
            &test_entries(
                DataType::LargeList(Arc::new(Field::new("element", DataType::Int32, true))),
                DataType::Utf8View,
                true,
            ),
            DedupPolicy::Exception,
        )
        .unwrap();
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn equal_keys_cover_every_primitive_and_struct_carrier() {
        use arrow_array::{
            BinaryArray, BooleanArray, Date32Array, Float32Array, Int8Array, Int16Array,
            Int64Array, NullArray, StructArray, TimestampMicrosecondArray, UInt8Array, UInt16Array,
            UInt32Array, UInt64Array,
        };
        // Each pair: (a, b) where row 0 of `a` equals row 0 of `b` and row 1 differs.
        let pairs: Vec<(ArrayRef, ArrayRef)> = vec![
            (
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(BooleanArray::from(vec![true, true])),
            ),
            (
                Arc::new(Int8Array::from(vec![1, 2])),
                Arc::new(Int8Array::from(vec![1, 3])),
            ),
            (
                Arc::new(Int16Array::from(vec![1, 2])),
                Arc::new(Int16Array::from(vec![1, 3])),
            ),
            (
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(Int64Array::from(vec![1, 3])),
            ),
            (
                Arc::new(UInt8Array::from(vec![1, 2])),
                Arc::new(UInt8Array::from(vec![1, 3])),
            ),
            (
                Arc::new(UInt16Array::from(vec![1, 2])),
                Arc::new(UInt16Array::from(vec![1, 3])),
            ),
            (
                Arc::new(UInt32Array::from(vec![1, 2])),
                Arc::new(UInt32Array::from(vec![1, 3])),
            ),
            (
                Arc::new(UInt64Array::from(vec![1, 2])),
                Arc::new(UInt64Array::from(vec![1, 3])),
            ),
            (
                Arc::new(Float32Array::from(vec![1.5, 2.5])),
                Arc::new(Float32Array::from(vec![1.5, 3.5])),
            ),
            (
                Arc::new(Date32Array::from(vec![1, 2])),
                Arc::new(Date32Array::from(vec![1, 3])),
            ),
            (
                Arc::new(TimestampMicrosecondArray::from(vec![1, 2])),
                Arc::new(TimestampMicrosecondArray::from(vec![1, 3])),
            ),
            (
                Arc::new(BinaryArray::from(vec![&b"a"[..], &b"b"[..]])),
                Arc::new(BinaryArray::from(vec![&b"a"[..], &b"c"[..]])),
            ),
        ];
        for (a, b) in &pairs {
            assert!(
                keys_equal(a.as_ref(), 0, b.as_ref(), 0).unwrap(),
                "{:?}",
                a.data_type()
            );
            assert!(
                !keys_equal(a.as_ref(), 1, b.as_ref(), 1).unwrap(),
                "{:?}",
                a.data_type()
            );
        }
        let n = NullArray::new(1);
        assert!(keys_equal(&n, 0, &n, 0).unwrap());
        // Structs compare field by field, by position.
        let st = |x: Vec<i32>, y: Vec<&str>| -> ArrayRef {
            Arc::new(StructArray::from(vec![
                (
                    Arc::new(Field::new("x", DataType::Int32, true)),
                    Arc::new(Int32Array::from(x)) as ArrayRef,
                ),
                (
                    Arc::new(Field::new("y", DataType::Utf8, true)),
                    Arc::new(StringArray::from(y)) as ArrayRef,
                ),
            ]))
        };
        let (s1, s2) = (
            st(vec![1, 2], vec!["a", "b"]),
            st(vec![1, 2], vec!["a", "c"]),
        );
        assert!(keys_equal(s1.as_ref(), 0, s2.as_ref(), 0).unwrap());
        assert!(!keys_equal(s1.as_ref(), 1, s2.as_ref(), 1).unwrap());
        // Structs of different arity are never equal; fields pair by position.
        let narrow: ArrayRef = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("x", DataType::Int32, true)),
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        )]));
        assert!(!keys_equal(s1.as_ref(), 0, narrow.as_ref(), 0).unwrap());
        let renamed: ArrayRef = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("x", DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("z", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
            ),
        ]));
        assert!(keys_equal(s1.as_ref(), 0, renamed.as_ref(), 0).unwrap());
    }

    fn strs(values: Vec<&str>) -> ArrayRef {
        Arc::new(StringArray::from(values))
    }

    #[test]
    fn dedup_raises_on_duplicates_and_keeps_last_under_last_win() {
        let keys = strs(vec!["b", "a", "b"]);
        let err = KrishivError::from(dedup_entries(keys.as_ref(), 3, false).unwrap_err());
        assert_eq!(err.error_class(), ErrorClass::DuplicatedMapKey);
        assert!(err.to_string().contains("b"), "{err}");
        let (kept, taken) = dedup_entries(keys.as_ref(), 3, true).unwrap();
        assert_eq!((kept, taken), (vec![0, 1], vec![2, 1]));
    }

    #[test]
    fn dedup_rejects_null_keys_and_short_arrays() {
        let nulls: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None]));
        let err = KrishivError::from(dedup_entries(nulls.as_ref(), 2, true).unwrap_err());
        assert_eq!(err.error_class(), ErrorClass::NullMapKey);
        assert!(dedup_entries(nulls.as_ref(), 3, true).is_err());
        assert_eq!(
            dedup_entries(nulls.as_ref(), 0, true).unwrap(),
            (vec![], vec![])
        );
    }

    #[test]
    fn union_merges_in_first_seen_order() {
        let (k1, k2) = (strs(vec!["a", "b", "b"]), strs(vec!["b", "c", "c"]));
        let m = union_keys(k1.as_ref(), 3, k2.as_ref(), 3).unwrap();
        // `a` only left, `b` both (first entries), `c` only right.
        assert_eq!(
            m,
            vec![
                (0, Some(0), None),
                (1, Some(1), Some(0)),
                (4, None, Some(1)),
            ]
        );
        let nulls: ArrayRef = Arc::new(StringArray::from(vec![None::<&str>]));
        let err = KrishivError::from(union_keys(k1.as_ref(), 1, nulls.as_ref(), 1).unwrap_err());
        assert_eq!(err.error_class(), ErrorClass::NullMapKey);
        let err = KrishivError::from(union_keys(nulls.as_ref(), 1, k1.as_ref(), 1).unwrap_err());
        assert_eq!(err.error_class(), ErrorClass::NullMapKey);
        assert!(union_keys(k1.as_ref(), 9, k2.as_ref(), 1).is_err());
    }

    #[test]
    fn build_map_checks_its_lengths() {
        let keys = strs(vec!["a", "b"]);
        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let offsets = OffsetBuffer::new(vec![0_i32, 2].into());
        let m = build_map(&keys, &values, offsets.clone(), None, false).unwrap();
        assert_eq!(m.len(), 1);
        let short: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        assert!(build_map(&keys, &short, offsets, None, false).is_err());
    }

    #[test]
    fn view_keys_hash_like_plain_keys() {
        let views: ArrayRef = Arc::new(StringViewArray::from(vec!["a", "b", "a"]));
        let err = KrishivError::from(dedup_entries(views.as_ref(), 3, false).unwrap_err());
        assert_eq!(err.error_class(), ErrorClass::DuplicatedMapKey);
        let bins: ArrayRef = Arc::new(arrow_array::BinaryViewArray::from(vec![
            b"x".as_slice(),
            b"x".as_slice(),
        ]));
        let (kept, taken) = dedup_entries(bins.as_ref(), 2, true).unwrap();
        assert_eq!((kept, taken), (vec![0], vec![1]));
    }
}
