// SPDX-License-Identifier: Apache-2.0
//! Total row orderings over Arrow values, for the comparator-free
//! `array_sort` (Spark's `RowOrdering`) and the key rendering of map
//! errors.
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
use arrow_buffer::ArrowNativeType as _;
use arrow_schema::{DataType, IntervalUnit, TimeUnit};

use super::error::KernelError;
use super::text::{java_double_string, java_float_string32};

/// The child values, start and length of the list at `idx`.
fn list_slice(array: &dyn Array, idx: usize) -> Result<(ArrayRef, usize, usize), KernelError> {
    // The neighbor of `idx` must exist: an index past the rows is an engine
    // error, never a panic.
    let next = idx
        .checked_add(1)
        .ok_or_else(|| KernelError::internal("a map key list of the wrong length"))?;
    if let Some(list) = array.as_list_opt::<i32>() {
        let offsets = list.value_offsets();
        let (Some(start), Some(stop)) = (offsets.get(idx), offsets.get(next)) else {
            return Err(KernelError::internal("a map key list past its offsets"));
        };
        let (start, stop) = (start.as_usize(), stop.as_usize());
        let len = stop
            .checked_sub(start)
            .ok_or_else(|| KernelError::internal("a map key list with decreasing offsets"))?;
        return Ok((Arc::clone(list.values()), start, len));
    }
    if let Some(list) = array.as_list_opt::<i64>() {
        let offsets = list.value_offsets();
        let (Some(start), Some(stop)) = (offsets.get(idx), offsets.get(next)) else {
            return Err(KernelError::internal("a map key list past its offsets"));
        };
        let (start, stop) = (start.as_usize(), stop.as_usize());
        let len = stop
            .checked_sub(start)
            .ok_or_else(|| KernelError::internal("a map key list with decreasing offsets"))?;
        return Ok((Arc::clone(list.values()), start, len));
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
                DataType::List(_) | DataType::LargeList(_) => {
                    let (lv, ls, ll) = list_slice(l, li)?;
                    let (rv, rs, rl) = list_slice(r, ri)?;
                    let mut ord = Ordering::Equal;
                    for e in 0..ll.min(rl) {
                        ord = cmp_rows(
                            lv.as_ref(),
                            ls.checked_add(e).ok_or_else(|| {
                                KernelError::internal("a list position past usize")
                            })?,
                            rv.as_ref(),
                            rs.checked_add(e).ok_or_else(|| {
                                KernelError::internal("a list position past usize")
                            })?,
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
        DataType::List(_) | DataType::LargeList(_) => match list_slice(array, idx) {
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
