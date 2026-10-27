// SPDX-License-Identifier: Apache-2.0
//! Canonical key encodings and total row orderings over Arrow values, for
//! the map builders and the comparator-free `array_sort` (Spark's
//! `ArrayBasedMapBuilder` key handling and `RowOrdering`).
//!
//! [`normalize_key`] encodes one value to bytes so that equal Spark values
//! encode equally: integers big-endian, floats with `-0.0` folded to `0.0`
//! and every NaN folded to one canonical NaN (Spark's
//! `NormalizeFloatingNumbers`), strings and binary as their bytes, structs
//! and lists recursively. A NULL encodes as `None` (a NULL map key is a
//! `NULL_MAP_KEY` error, never a value). Dictionary carriers resolve
//! through their values array.
//!
//! [`cmp_rows`] is the total order Spark sorts and compares by: NULL is
//! less than every value, NaN is greater than every non-NaN float,
//! `-0.0` equals `0.0`, strings and binary compare byte-wise, structs and
//! lists compare lexicographically. Types the analyzer refuses as keys or
//! sort elements (maps, and anything it never produces) are engine errors.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Date32Type, Date64Type, Decimal32Type, Decimal64Type, Decimal128Type,
    Decimal256Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    IntervalDayTimeType, IntervalMonthDayNanoType, IntervalYearMonthType, Time32MillisecondType,
    Time32SecondType, Time64MicrosecondType, Time64NanosecondType, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, PrimitiveArray};
use arrow_buffer::{ArrowNativeType as _, i256};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};

use super::error::KernelError;
use super::text::{java_double_string, java_float_string32};

/// What does not fit a 32-bit offset or index.
fn too_big(n: usize) -> KernelError {
    KernelError::internal(format!("a map key encoding of {n} rows"))
}

/// A NULL map key is a `NULL_MAP_KEY` error, never a value.
fn null_key() -> Option<Vec<u8>> {
    None
}

/// Canonical NaN bits (Spark's float normalization folds every NaN here).
const CANONICAL_F32_NAN: u32 = 0x7fc0_0000;
/// Canonical NaN bits for doubles.
const CANONICAL_F64_NAN: u64 = 0x7ff8_0000_0000_0000;

/// Normalized `f32` bits: NaN folds to one canonical NaN, `-0.0` to `0.0`.
fn norm_f32(v: f32) -> u32 {
    if v.is_nan() {
        CANONICAL_F32_NAN
    } else if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

/// Normalized `f64` bits.
fn norm_f64(v: f64) -> u64 {
    if v.is_nan() {
        CANONICAL_F64_NAN
    } else if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

/// Pushes a tag, a `u32` length and the bytes.
fn push_bytes(out: &mut Vec<u8>, tag: u8, b: &[u8]) -> Result<(), KernelError> {
    out.push(tag);
    out.extend_from_slice(
        &u32::try_from(b.len())
            .map_err(|_| too_big(b.len()))?
            .to_be_bytes(),
    );
    out.extend_from_slice(b);
    Ok(())
}

/// Encodes the non-null value at `idx` of the primitive `array`.
fn primitive<T: ArrowPrimitiveType>(
    out: &mut Vec<u8>,
    tag: u8,
    array: &PrimitiveArray<T>,
    idx: usize,
    encode: impl Fn(T::Native, &mut Vec<u8>),
) {
    out.push(tag);
    encode(array.value(idx), out);
}

/// Whether the primitive `array` holds `idx`'s type: encodes `tag` plus
/// the native widened to `wide` (identity where the carrier holds it).
/// Tags separate the Arrow carriers that share a payload shape.
macro_rules! keyed {
    ($array:ident, $out:ident, $idx:ident, $( $pat:pat => $tag:expr, $prim:ty, $wide:ty ),* $(,)?) => {
        match $array.data_type() {
            $( $pat => {
                primitive(&mut $out, $tag, $array.as_primitive::<$prim>(), $idx, |v, o| {
                    o.extend_from_slice(&<$wide>::from(v).to_be_bytes());
                });
                true
            } )*
            _ => false,
        }
    };
}

/// The canonical encoding of the value at `idx` of `array`, `None` for
/// NULL (see the module doc).
///
/// # Errors
/// [`KernelError`] for a type the analyzer never lets through (maps, run-end
/// encoded and unknown dictionary keys): an engine bug, never user input.
pub fn normalize_key(array: &dyn Array, idx: usize) -> Result<Option<Vec<u8>>, KernelError> {
    if array.is_null(idx) {
        return Ok(null_key());
    }
    let mut out = Vec::new();
    if !keyed!(array, out, idx,
        DataType::Int8 => 0x02, Int8Type, i64,
        DataType::Int16 => 0x02, Int16Type, i64,
        DataType::Int32 => 0x02, Int32Type, i64,
        DataType::Int64 => 0x02, Int64Type, i64,
        DataType::UInt8 => 0x03, UInt8Type, u64,
        DataType::UInt16 => 0x03, UInt16Type, u64,
        DataType::UInt32 => 0x03, UInt32Type, u64,
        DataType::UInt64 => 0x03, UInt64Type, u64,
        DataType::Decimal32(_, _) => 0x06, Decimal32Type, i64,
        DataType::Decimal64(_, _) => 0x06, Decimal64Type, i64,
        DataType::Decimal128(_, _) => 0x06, Decimal128Type, i128,
        DataType::Decimal256(_, _) => 0x06, Decimal256Type, i256,
        DataType::Date32 => 0x09, Date32Type, i32,
        DataType::Date64 => 0x09, Date64Type, i64,
        DataType::Time32(TimeUnit::Second) => 0x0a, Time32SecondType, i32,
        DataType::Time32(TimeUnit::Millisecond) => 0x0a, Time32MillisecondType, i32,
        DataType::Time64(TimeUnit::Microsecond) => 0x0a, Time64MicrosecondType, i64,
        DataType::Time64(TimeUnit::Nanosecond) => 0x0a, Time64NanosecondType, i64,
        DataType::Timestamp(TimeUnit::Second, _) => 0x0b, TimestampSecondType, i64,
        DataType::Timestamp(TimeUnit::Millisecond, _) => 0x0b, TimestampMillisecondType, i64,
        DataType::Timestamp(TimeUnit::Microsecond, _) => 0x0b, TimestampMicrosecondType, i64,
        DataType::Timestamp(TimeUnit::Nanosecond, _) => 0x0b, TimestampNanosecondType, i64,
        DataType::Interval(IntervalUnit::YearMonth) => 0x0c, IntervalYearMonthType, i32,
    ) {
        match array.data_type() {
            DataType::Boolean => {
                let a = array.as_boolean();
                out.push(0x01);
                out.push(u8::from(a.value(idx)));
            }
            DataType::Float32 => {
                primitive(
                    &mut out,
                    0x04,
                    array.as_primitive::<Float32Type>(),
                    idx,
                    |v, o| {
                        o.extend_from_slice(&norm_f32(v).to_be_bytes());
                    },
                );
            }
            DataType::Float64 => {
                primitive(
                    &mut out,
                    0x05,
                    array.as_primitive::<Float64Type>(),
                    idx,
                    |v, o| {
                        o.extend_from_slice(&norm_f64(v).to_be_bytes());
                    },
                );
            }
            DataType::Utf8 => {
                push_bytes(
                    &mut out,
                    0x07,
                    array.as_string::<i32>().value(idx).as_bytes(),
                )?;
            }
            DataType::LargeUtf8 => {
                push_bytes(
                    &mut out,
                    0x07,
                    array.as_string::<i64>().value(idx).as_bytes(),
                )?;
            }
            DataType::Utf8View => {
                push_bytes(&mut out, 0x07, array.as_string_view().value(idx).as_bytes())?;
            }
            DataType::Binary => {
                push_bytes(&mut out, 0x08, array.as_binary::<i32>().value(idx))?;
            }
            DataType::LargeBinary => {
                push_bytes(&mut out, 0x08, array.as_binary::<i64>().value(idx))?;
            }
            DataType::BinaryView => {
                push_bytes(&mut out, 0x08, array.as_binary_view().value(idx))?;
            }
            DataType::Interval(IntervalUnit::DayTime) => {
                primitive(
                    &mut out,
                    0x0c,
                    array.as_primitive::<IntervalDayTimeType>(),
                    idx,
                    |v, o| {
                        o.extend_from_slice(&v.days.to_be_bytes());
                        o.extend_from_slice(&v.milliseconds.to_be_bytes());
                    },
                );
            }
            DataType::Interval(IntervalUnit::MonthDayNano) => {
                primitive(
                    &mut out,
                    0x0c,
                    array.as_primitive::<IntervalMonthDayNanoType>(),
                    idx,
                    |v: arrow_buffer::IntervalMonthDayNano, o| {
                        o.extend_from_slice(&v.months.to_be_bytes());
                        o.extend_from_slice(&v.days.to_be_bytes());
                        o.extend_from_slice(&v.nanoseconds.to_be_bytes());
                    },
                );
            }
            DataType::Struct(_) => {
                let s = array.as_struct();
                out.push(0x0d);
                for child in s.columns() {
                    match normalize_key(child.as_ref(), idx)? {
                        Some(mut bytes) => {
                            out.push(1);
                            out.append(&mut bytes);
                        }
                        None => out.push(0),
                    }
                }
            }
            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _)
            | DataType::ListView(_)
            | DataType::LargeListView(_) => {
                let (values, start, len) = list_slice(array, idx)?;
                out.push(0x0e);
                out.extend_from_slice(&u32::try_from(len).map_err(|_| too_big(len))?.to_be_bytes());
                let end = start.checked_add(len).ok_or_else(|| too_big(len))?;
                for e in start..end {
                    match normalize_key(values.as_ref(), e)? {
                        Some(mut bytes) => {
                            out.push(1);
                            out.append(&mut bytes);
                        }
                        None => out.push(0),
                    }
                }
            }
            DataType::Dictionary(_, _) => {
                return dictionary_key(array, idx);
            }
            // A NULL array is all NULL, handled above.
            DataType::Null => return Ok(null_key()),
            other => {
                return Err(KernelError::internal(format!(
                    "a map key comparison of {other}"
                )));
            }
        }
    }
    Ok(Some(out))
}

/// The canonical encoding of one dictionary value: through the values
/// array at the key's index.
fn dictionary_key(array: &dyn Array, idx: usize) -> Result<Option<Vec<u8>>, KernelError> {
    macro_rules! key {
        ($t:ty) => {{
            let d = array
                .as_any()
                .downcast_ref::<arrow_array::DictionaryArray<$t>>()
                .ok_or_else(|| KernelError::internal("a map key of a dictionary array"))?;
            if d.is_null(idx) {
                return Ok(null_key());
            }
            let at = d.keys().value(idx).as_usize();
            return normalize_key(d.values().as_ref(), at);
        }};
    }
    use arrow_array::types::{Int32Type, Int64Type, UInt32Type, UInt64Type};
    match array.data_type() {
        DataType::Dictionary(k, _) => match k.as_ref() {
            DataType::Int32 => key!(Int32Type),
            DataType::Int64 => key!(Int64Type),
            DataType::UInt32 => key!(UInt32Type),
            DataType::UInt64 => key!(UInt64Type),
            other => Err(KernelError::internal(format!(
                "a map key of a dictionary with {other} keys"
            ))),
        },
        other => Err(KernelError::internal(format!(
            "a map key comparison of {other}"
        ))),
    }
}

/// The child values, start and length of the list at `idx`.
fn list_slice(array: &dyn Array, idx: usize) -> Result<(ArrayRef, usize, usize), KernelError> {
    // Every arm reads `offsets[idx]` and `offsets[idx + 1]`; the row count
    // was validated upstream, so the neighbor always exists.
    let next = idx
        .checked_add(1)
        .ok_or_else(|| KernelError::internal("a map key list of the wrong length"))?;
    if let Some(list) = array.as_list_opt::<i32>() {
        let offsets = list.value_offsets();
        let (start, stop) = (offsets[idx].as_usize(), offsets[next].as_usize());
        let len = stop
            .checked_sub(start)
            .ok_or_else(|| KernelError::internal("a map key list with decreasing offsets"))?;
        return Ok((Arc::clone(list.values()), start, len));
    }
    if let Some(list) = array.as_list_opt::<i64>() {
        let offsets = list.value_offsets();
        let (start, stop) = (offsets[idx].as_usize(), offsets[next].as_usize());
        let len = stop
            .checked_sub(start)
            .ok_or_else(|| KernelError::internal("a map key list with decreasing offsets"))?;
        return Ok((Arc::clone(list.values()), start, len));
    }
    if let Some(view) = array.as_any().downcast_ref::<arrow_array::ListViewArray>() {
        let (start, len) = (
            view.value_offset(idx).as_usize(),
            view.value_size(idx).as_usize(),
        );
        return Ok((Arc::clone(view.values()), start, len));
    }
    if let Some(view) = array
        .as_any()
        .downcast_ref::<arrow_array::LargeListViewArray>()
    {
        let (start, len) = (
            view.value_offset(idx).as_usize(),
            view.value_size(idx).as_usize(),
        );
        return Ok((Arc::clone(view.values()), start, len));
    }
    if let Some(list) = array
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeListArray>()
    {
        let len = usize::try_from(list.value_length())
            .map_err(|_| KernelError::internal("a map key list with a negative length"))?;
        return Ok((
            Arc::clone(list.values()),
            idx.checked_mul(len).ok_or_else(|| too_big(idx))?,
            len,
        ));
    }
    Err(KernelError::internal(format!(
        "a map key comparison of {}",
        array.data_type()
    )))
}

/// Spark's float order: equal values (including `-0.0` and `0.0`) tie, NaN
/// is greater than every non-NaN value.
macro_rules! spark_float {
    ($name:ident, $ty:ty) => {
        fn $name(a: $ty, b: $ty) -> Ordering {
            if a == b {
                return Ordering::Equal;
            }
            match (a.is_nan(), b.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
            }
        }
    };
}

spark_float!(spark_f32, f32);
spark_float!(spark_f64, f64);

/// The total order of the non-null values at `l` and `r` of one primitive
/// type, and of ` cmp` of their natives.
fn primitive_ord<T: ArrowPrimitiveType>(
    l: &PrimitiveArray<T>,
    li: usize,
    r: &PrimitiveArray<T>,
    ri: usize,
    cmp: impl Fn(T::Native, T::Native) -> Ordering,
) -> Ordering {
    cmp(l.value(li), r.value(ri))
}

/// The order of the non-null values at `l` and `r` of one listed primitive
/// type, or `None` for any other type.
macro_rules! ordered {
    ($l:ident, $r:ident, $li:ident, $ri:ident, $( $pat:pat => $prim:ty, $cmp:expr ),* $(,)?) => {
        match $l.data_type() {
            $( $pat => Some(primitive_ord($l.as_primitive::<$prim>(), $li, $r.as_primitive::<$prim>(), $ri, $cmp)), )*
            _ => None,
        }
    };
}

/// The byte-wise order of the non-null values at `l` and `r` of one
/// listed string or binary carrier, or `None` for any other type.
macro_rules! ordered_bytes {
    ($l:ident, $r:ident, $li:ident, $ri:ident, $( $pat:pat => $get:ident<$prim:ty> ),* $(,)?) => {
        match $l.data_type() {
            $( $pat => Some($l.$get::<$prim>().value($li).cmp($r.$get::<$prim>().value($ri))), )*
            _ => None,
        }
    };
}

/// The total order Spark sorts and compares by (see the module doc): NULL
/// is less than every value.
///
/// # Errors
/// [`KernelError`] for a type the analyzer never lets through.
pub fn cmp_rows(
    l: &dyn Array,
    li: usize,
    r: &dyn Array,
    ri: usize,
) -> Result<Ordering, KernelError> {
    match (l.is_null(li), r.is_null(ri)) {
        (true, true) => return Ok(Ordering::Equal),
        (true, false) => return Ok(Ordering::Less),
        (false, true) => return Ok(Ordering::Greater),
        (false, false) => {}
    }
    if l.data_type() != r.data_type() {
        return Err(KernelError::internal(format!(
            "a comparison of {} with {}",
            l.data_type(),
            r.data_type()
        )));
    }
    let ord = match ordered!(l, r, li, ri,
        DataType::Int8 => Int8Type, |a, b| a.cmp(&b),
        DataType::Int16 => Int16Type, |a, b| a.cmp(&b),
        DataType::Int32 => Int32Type, |a, b| a.cmp(&b),
        DataType::Int64 => Int64Type, |a, b| a.cmp(&b),
        DataType::UInt8 => UInt8Type, |a, b| a.cmp(&b),
        DataType::UInt16 => UInt16Type, |a, b| a.cmp(&b),
        DataType::UInt32 => UInt32Type, |a, b| a.cmp(&b),
        DataType::UInt64 => UInt64Type, |a, b| a.cmp(&b),
        DataType::Float32 => Float32Type, spark_f32,
        DataType::Float64 => Float64Type, spark_f64,
        DataType::Decimal32(_, _) => Decimal32Type, |a, b| a.cmp(&b),
        DataType::Decimal64(_, _) => Decimal64Type, |a, b| a.cmp(&b),
        DataType::Decimal128(_, _) => Decimal128Type, |a, b| a.cmp(&b),
        DataType::Decimal256(_, _) => Decimal256Type, |a, b| a.cmp(&b),
        DataType::Date32 => Date32Type, |a, b| a.cmp(&b),
        DataType::Date64 => Date64Type, |a, b| a.cmp(&b),
        DataType::Time32(TimeUnit::Second) => Time32SecondType, |a, b| a.cmp(&b),
        DataType::Time32(TimeUnit::Millisecond) => Time32MillisecondType, |a, b| a.cmp(&b),
        DataType::Time64(TimeUnit::Microsecond) => Time64MicrosecondType, |a, b| a.cmp(&b),
        DataType::Time64(TimeUnit::Nanosecond) => Time64NanosecondType, |a, b| a.cmp(&b),
        DataType::Timestamp(TimeUnit::Second, _) => TimestampSecondType, |a, b| a.cmp(&b),
        DataType::Timestamp(TimeUnit::Millisecond, _) => TimestampMillisecondType, |a, b| a.cmp(&b),
        DataType::Timestamp(TimeUnit::Microsecond, _) => TimestampMicrosecondType, |a, b| a.cmp(&b),
        DataType::Timestamp(_, _) => TimestampNanosecondType, |a, b| a.cmp(&b),
        DataType::Interval(IntervalUnit::YearMonth) => IntervalYearMonthType, |a, b| a.cmp(&b),
        DataType::Interval(IntervalUnit::DayTime) => IntervalDayTimeType, |a, b| a.cmp(&b),
    ) {
        Some(ord) => ord,
        None => match ordered_bytes!(l, r, li, ri,
            DataType::Utf8 => as_string<i32>,
            DataType::LargeUtf8 => as_string<i64>,
            DataType::Binary => as_binary<i32>,
            DataType::LargeBinary => as_binary<i64>,
        ) {
            Some(ord) => ord,
            None => match l.data_type() {
                DataType::Boolean => {
                    let (a, b) = (l.as_boolean(), r.as_boolean());
                    a.value(li).cmp(&b.value(ri))
                }
                DataType::Utf8View => l
                    .as_string_view()
                    .value(li)
                    .cmp(r.as_string_view().value(ri)),
                DataType::BinaryView => l
                    .as_binary_view()
                    .value(li)
                    .cmp(r.as_binary_view().value(ri)),
                DataType::Interval(IntervalUnit::MonthDayNano) => {
                    let (a, b) = (
                        l.as_primitive::<IntervalMonthDayNanoType>(),
                        r.as_primitive::<IntervalMonthDayNanoType>(),
                    );
                    let (a, b) = (a.value(li), b.value(ri));
                    a.months
                        .cmp(&b.months)
                        .then(a.days.cmp(&b.days))
                        .then(a.nanoseconds.cmp(&b.nanoseconds))
                }
                DataType::Struct(_) => {
                    let (a, b) = (l.as_struct(), r.as_struct());
                    let mut ord = Ordering::Equal;
                    for (x, y) in a.columns().iter().zip(b.columns().iter()) {
                        ord = cmp_rows(x.as_ref(), li, y.as_ref(), ri)?;
                        if ord != Ordering::Equal {
                            break;
                        }
                    }
                    ord
                }
                DataType::List(_)
                | DataType::LargeList(_)
                | DataType::FixedSizeList(_, _)
                | DataType::ListView(_)
                | DataType::LargeListView(_) => {
                    let (lv, ls, ll) = list_slice(l, li)?;
                    let (rv, rs, rl) = list_slice(r, ri)?;
                    let mut ord = Ordering::Equal;
                    for e in 0..ll.min(rl) {
                        ord = cmp_rows(
                            lv.as_ref(),
                            ls.checked_add(e).ok_or_else(|| too_big(e))?,
                            rv.as_ref(),
                            rs.checked_add(e).ok_or_else(|| too_big(e))?,
                        )?;
                        if ord != Ordering::Equal {
                            break;
                        }
                    }
                    ord.then(ll.cmp(&rl))
                }
                DataType::Dictionary(_, _) => {
                    let (a, b) = (dictionary_value(l, li)?, dictionary_value(r, ri)?);
                    cmp_rows(a.as_ref(), 0, b.as_ref(), 0)?
                }
                other => {
                    return Err(KernelError::internal(format!("a comparison of {other}")));
                }
            },
        },
    };
    Ok(ord)
}

/// One decoded dictionary value as a one-row array.
fn dictionary_value(array: &dyn Array, idx: usize) -> Result<ArrayRef, KernelError> {
    macro_rules! value {
        ($t:ty) => {{
            let d = array
                .as_any()
                .downcast_ref::<arrow_array::DictionaryArray<$t>>()
                .ok_or_else(|| KernelError::internal("a comparison of a dictionary array"))?;
            if d.is_null(idx) {
                return Err(KernelError::internal(
                    "a comparison of a null dictionary value",
                ));
            }
            return Ok(d.values().slice(d.keys().value(idx).as_usize(), 1));
        }};
    }
    use arrow_array::types::{Int32Type, Int64Type, UInt32Type, UInt64Type};
    match array.data_type() {
        DataType::Dictionary(k, _) => match k.as_ref() {
            DataType::Int32 => value!(Int32Type),
            DataType::Int64 => value!(Int64Type),
            DataType::UInt32 => value!(UInt32Type),
            DataType::UInt64 => value!(UInt64Type),
            other => Err(KernelError::internal(format!(
                "a comparison of a dictionary with {other} keys"
            ))),
        },
        other => Err(KernelError::internal(format!("a comparison of {other}"))),
    }
}

/// The `to_string` of the non-null value at `idx` of one listed primitive
/// type, or `None` for any other type.
macro_rules! rendered {
    ($array:ident, $idx:ident, $( $pat:pat => $prim:ty ),* $(,)?) => {
        match $array.data_type() {
            $( $pat => Some($array.as_primitive::<$prim>().value($idx).to_string()), )*
            _ => None,
        }
    };
}

/// Spark's rendering of a value in an error message (`key.toString` in the
/// JVM): numbers plain, strings bare, floats in Java's notation, arrays
/// and structs bracketed and comma-joined.
pub fn render_value(array: &dyn Array, idx: usize) -> String {
    if array.is_null(idx) {
        return "null".to_owned();
    }
    if let Some(s) = rendered!(array, idx,
        DataType::Int8 => Int8Type,
        DataType::Int16 => Int16Type,
        DataType::Int32 => Int32Type,
        DataType::Int64 => Int64Type,
        DataType::UInt8 => UInt8Type,
        DataType::UInt16 => UInt16Type,
        DataType::UInt32 => UInt32Type,
        DataType::UInt64 => UInt64Type,
        DataType::Date32 => Date32Type,
        DataType::Date64 => Date64Type,
        DataType::Time32(TimeUnit::Second) => Time32SecondType,
        DataType::Time32(TimeUnit::Millisecond) => Time32MillisecondType,
        DataType::Time64(TimeUnit::Microsecond) => Time64MicrosecondType,
        DataType::Time64(TimeUnit::Nanosecond) => Time64NanosecondType,
        DataType::Timestamp(TimeUnit::Second, _) => TimestampSecondType,
        DataType::Timestamp(TimeUnit::Millisecond, _) => TimestampMillisecondType,
        DataType::Timestamp(TimeUnit::Microsecond, _) => TimestampMicrosecondType,
        DataType::Timestamp(_, _) => TimestampNanosecondType,
        DataType::Interval(IntervalUnit::YearMonth) => IntervalYearMonthType,
    ) {
        return s;
    }
    // The rest reads each carrier directly.
    match array.data_type() {
        DataType::Boolean => array.as_boolean().value(idx).to_string(),
        DataType::Float32 => java_float_string32(array.as_primitive::<Float32Type>().value(idx)),
        DataType::Float64 => java_double_string(array.as_primitive::<Float64Type>().value(idx)),
        DataType::Decimal32(_, s) => super::text::java_decimal_string(
            i128::from(array.as_primitive::<Decimal32Type>().value(idx)),
            *s,
        ),
        DataType::Decimal64(_, s) => super::text::java_decimal_string(
            array.as_primitive::<Decimal64Type>().value(idx).into(),
            *s,
        ),
        DataType::Decimal128(_, s) => {
            super::text::java_decimal_string(array.as_primitive::<Decimal128Type>().value(idx), *s)
        }
        DataType::Utf8 => array.as_string::<i32>().value(idx).to_owned(),
        DataType::LargeUtf8 => array.as_string::<i64>().value(idx).to_owned(),
        DataType::Utf8View => array.as_string_view().value(idx).to_owned(),
        DataType::Binary => render_hex(array.as_binary::<i32>().value(idx)),
        DataType::LargeBinary => render_hex(array.as_binary::<i64>().value(idx)),
        DataType::BinaryView => render_hex(array.as_binary_view().value(idx)),
        DataType::Interval(IntervalUnit::DayTime) => {
            let v = array.as_primitive::<IntervalDayTimeType>().value(idx);
            format!("{},{}", v.days, v.milliseconds)
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let v = array.as_primitive::<IntervalMonthDayNanoType>().value(idx);
            format!("{},{},{}", v.months, v.days, v.nanoseconds)
        }
        DataType::Struct(_) => {
            let s = array.as_struct();
            let parts: Vec<String> = s
                .columns()
                .iter()
                .map(|c| render_value(c.as_ref(), idx))
                .collect();
            format!("[{}]", parts.join(","))
        }
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::FixedSizeList(_, _)
        | DataType::ListView(_)
        | DataType::LargeListView(_) => match list_slice(array, idx) {
            Ok((values, start, len)) => {
                // `start + len` stays inside the values; the fallback keeps
                // the total rendering on the impossible overflow.
                let end = start.checked_add(len).unwrap_or(start);
                let parts: Vec<String> = (start..end)
                    .map(|e| render_value(values.as_ref(), e))
                    .collect();
                format!("[{}]", parts.join(","))
            }
            Err(_) => render_opaque(array, idx),
        },
        _ => render_opaque(array, idx),
    }
}

/// The data type and row, for values with no JVM-shaped rendering here.
fn render_opaque(array: &dyn Array, idx: usize) -> String {
    format!("{}[{idx}]", array.data_type())
}

/// Hexadecimal bytes, for binary keys with no text rendering here.
fn render_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for &b in bytes {
        out.push(HEX[usize::from(b.wrapping_shr(4))] as char);
        out.push(HEX[usize::from(b & 0x0f)] as char);
    }
    out
}

#[cfg(test)]
mod tests;
