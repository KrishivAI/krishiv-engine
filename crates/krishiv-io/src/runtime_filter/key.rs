// SPDX-License-Identifier: Apache-2.0
//! Key normalization: the one path by which a join build hashes and
//! compares its keys and every scan stage hashes and compares the column it
//! filters, so both sides see identical values and bytes.
//!
//! A column is first cast to the filter's **key type**, losslessly only
//! ([`widens_to`]: integer widening, integers to a decimal or double that
//! holds them, decimal widening, `float` to `double`, the representations
//! of one string, binary or timestamp type); any other pair is not
//! filtered. Then values are normalized as Spark normalizes join keys:
//! `-0.0` is `0.0` and every NaN is one NaN. The bytes hashed are the key
//! type's fixed-width little-endian value (Parquet's plain encoding) or the
//! string's bytes.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef};
use arrow_buffer::{NullBuffer, ScalarBuffer};
use arrow_schema::{DataType, TimeUnit};

use crate::request::ScalarValue;

/// Whether `t` can be a runtime filter's key type.
#[must_use]
pub fn supported_key_type(t: &DataType) -> bool {
    use DataType as D;
    matches!(
        t,
        D::Boolean
            | D::Int8
            | D::Int16
            | D::Int32
            | D::Int64
            | D::Float32
            | D::Float64
            | D::Decimal128(..)
            | D::Date32
            | D::Timestamp(TimeUnit::Microsecond, _)
            | D::Utf8
            | D::LargeUtf8
            | D::Utf8View
            | D::Binary
            | D::LargeBinary
            | D::BinaryView
    )
}

fn int_digits(t: &DataType) -> Option<u8> {
    Some(match t {
        DataType::Int8 => 3,
        DataType::Int16 => 5,
        DataType::Int32 => 10,
        DataType::Int64 => 19,
        _ => return None,
    })
}

fn int_rank(t: &DataType) -> Option<u8> {
    Some(match t {
        DataType::Int8 => 1,
        DataType::Int16 => 2,
        DataType::Int32 => 3,
        DataType::Int64 => 4,
        _ => return None,
    })
}

/// Whether every value of `from` converts to a distinct, equal value of
/// `to` (see the module docs). Equal types widen to themselves.
#[must_use]
pub fn widens_to(from: &DataType, to: &DataType) -> bool {
    use DataType as D;
    if from.equals_datatype(to) {
        return supported_key_type(to);
    }
    let strings = |t: &DataType| matches!(t, D::Utf8 | D::LargeUtf8 | D::Utf8View);
    let binaries = |t: &DataType| matches!(t, D::Binary | D::LargeBinary | D::BinaryView);
    if (strings(from) && strings(to)) || (binaries(from) && binaries(to)) {
        return true;
    }
    match (from, to) {
        (D::Float32 | D::Int8 | D::Int16 | D::Int32, D::Float64)
        | (D::Int8 | D::Int16, D::Float32)
        | (D::Timestamp(TimeUnit::Microsecond, _), D::Timestamp(TimeUnit::Microsecond, _)) => true,
        (D::Decimal128(p1, s1), D::Decimal128(p2, s2)) => {
            s2 >= s1 && i16::from(*p2) - i16::from(*s2) >= i16::from(*p1) - i16::from(*s1)
        }
        (f, D::Decimal128(p, s)) => {
            int_digits(f).is_some_and(|d| i16::from(*p) - i16::from(*s) >= i16::from(d))
        }
        (f, t) => matches!((int_rank(f), int_rank(t)), (Some(a), Some(b)) if a < b),
    }
}

/// One normalized key value: how a value set holds its members.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum KeyValue {
    /// Integers, decimals (unscaled), booleans and float bit patterns.
    Fixed(i128),
    /// String and binary bytes.
    Bytes(Box<[u8]>),
}

/// Normalized values of one column (see the module docs).
#[derive(Debug)]
pub(crate) enum Norm {
    /// Integers, dates, timestamps and booleans (0/1), widened to `i64`;
    /// `width` is the key type's size in bytes.
    Int {
        values: ScalarBuffer<i64>,
        width: usize,
    },
    /// Unscaled decimals at the key type's scale.
    Decimal(Vec<i128>),
    /// Floats after Spark's normalization; `single` for a `float` key.
    Float { values: Vec<f64>, single: bool },
    /// String or binary values, as stored.
    Bytes(ArrayRef),
}

/// The values of a column normalized to a key type, with its nulls: see
/// [`normalize_key_bytes`].
#[derive(Debug)]
pub struct NormalizedKeys {
    pub(crate) norm: Norm,
    nulls: Option<NullBuffer>,
    len: usize,
}

fn spark_float(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

fn pow10(n: u32) -> Option<i128> {
    10_i128.checked_pow(n)
}

fn ints(a: &dyn Array) -> Option<ScalarBuffer<i64>> {
    Some(
        match a.data_type() {
            DataType::Int8 => a
                .as_primitive::<Int8Type>()
                .values()
                .iter()
                .map(|&v| i64::from(v))
                .collect::<Vec<i64>>(),
            DataType::Int16 => a
                .as_primitive::<Int16Type>()
                .values()
                .iter()
                .map(|&v| i64::from(v))
                .collect::<Vec<i64>>(),
            DataType::Int32 => a
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .map(|&v| i64::from(v))
                .collect::<Vec<i64>>(),
            // Zero-copy: the column's own buffer.
            DataType::Int64 => return Some(a.as_primitive::<Int64Type>().values().clone()),
            _ => return None,
        }
        .into(),
    )
}

/// `array` normalized to `key_type` (see the module docs), or `None` when
/// its type does not [widen](widens_to) to the key type: such a column is
/// never filtered.
#[must_use]
pub fn normalize_key_bytes(array: &dyn Array, key_type: &DataType) -> Option<NormalizedKeys> {
    use DataType as D;
    let from = array.data_type();
    if !widens_to(from, key_type) {
        return None;
    }
    let norm = match key_type {
        D::Boolean => Norm::Int {
            values: array
                .as_boolean()
                .values()
                .iter()
                .map(i64::from)
                .collect::<Vec<_>>()
                .into(),
            width: 1,
        },
        D::Int8 | D::Int16 | D::Int32 | D::Int64 => Norm::Int {
            values: ints(array)?,
            width: key_type.primitive_width()?,
        },
        D::Date32 => Norm::Int {
            values: array
                .as_primitive::<Date32Type>()
                .values()
                .iter()
                .map(|&d| i64::from(d))
                .collect::<Vec<_>>()
                .into(),
            width: 4,
        },
        D::Timestamp(..) => Norm::Int {
            values: array
                .as_primitive::<TimestampMicrosecondType>()
                .values()
                .clone(),
            width: 8,
        },
        D::Float32 | D::Float64 => {
            #[expect(
                clippy::cast_precision_loss,
                reason = "only exact widenings reach here"
            )]
            let values: Vec<f64> = match from {
                D::Float32 => array
                    .as_primitive::<Float32Type>()
                    .values()
                    .iter()
                    .map(|&v| spark_float(f64::from(v)))
                    .collect(),
                D::Float64 => array
                    .as_primitive::<Float64Type>()
                    .values()
                    .iter()
                    .map(|&v| spark_float(v))
                    .collect(),
                _ => ints(array)?.iter().map(|&v| v as f64).collect(),
            };
            Norm::Float {
                values,
                single: *key_type == D::Float32,
            }
        }
        D::Decimal128(_, s) => {
            let (unscaled, from_scale): (Vec<i128>, i8) = match from {
                D::Decimal128(_, fs) => (
                    array.as_primitive::<Decimal128Type>().values().to_vec(),
                    *fs,
                ),
                _ => (ints(array)?.iter().map(|&v| i128::from(v)).collect(), 0),
            };
            let up = u32::try_from(i16::from(*s) - i16::from(from_scale)).ok()?;
            let factor = pow10(up)?;
            // A widening fits the target precision; a stored value beyond
            // its declared precision is not filtered at all.
            Norm::Decimal(
                unscaled
                    .into_iter()
                    .map(|v| v.checked_mul(factor))
                    .collect::<Option<Vec<_>>>()?,
            )
        }
        _ => Norm::Bytes(Arc::new(arrow_array::make_array(array.to_data()))),
    };
    Some(NormalizedKeys {
        norm,
        nulls: array.logical_nulls(),
        len: array.len(),
    })
}

/// Calls `f(i, bytes)` with each row's string or binary bytes.
pub(crate) fn for_each_bytes(a: &dyn Array, mut f: impl FnMut(usize, &[u8])) {
    match a.data_type() {
        DataType::Utf8 => a
            .as_string::<i32>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| f(i, v.unwrap_or_default().as_bytes())),
        DataType::LargeUtf8 => a
            .as_string::<i64>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| f(i, v.unwrap_or_default().as_bytes())),
        DataType::Utf8View => a
            .as_string_view()
            .iter()
            .enumerate()
            .for_each(|(i, v)| f(i, v.unwrap_or_default().as_bytes())),
        DataType::Binary => a
            .as_binary::<i32>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| f(i, v.unwrap_or_default())),
        DataType::LargeBinary => a
            .as_binary::<i64>()
            .iter()
            .enumerate()
            .for_each(|(i, v)| f(i, v.unwrap_or_default())),
        DataType::BinaryView => a
            .as_binary_view()
            .iter()
            .enumerate()
            .for_each(|(i, v)| f(i, v.unwrap_or_default())),
        _ => {}
    }
}

impl NormalizedKeys {
    /// Rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether row `i` is NULL.
    #[must_use]
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        self.nulls.as_ref().is_some_and(|n| n.is_null(i))
    }

    /// The row hash `h` of each row ([`super::hash`]: XXH64 of the
    /// normalized bytes; NULL rows hash their placeholder value, callers
    /// skip them).
    #[must_use]
    pub fn hashes(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.len);
        super::hash::row_hashes(self, None, &mut out);
        out
    }

    /// The validity of the rows (`None`: no NULL).
    pub(crate) fn nulls(&self) -> Option<&NullBuffer> {
        self.nulls.as_ref()
    }
}

#[inline]
pub(crate) fn float_bits(v: f64, single: bool) -> i128 {
    if single {
        #[expect(clippy::cast_possible_truncation, reason = "a widened float32")]
        let f = v as f32;
        i128::from(f.to_bits())
    } else {
        i128::from(v.to_bits())
    }
}

/// The smallest and largest normalized non-null keys.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum KeyRange {
    Int(i64, i64),
    Decimal(i128, i128),
    /// Never holds NaN (a NaN key drops the range).
    Float(f64, f64),
    Bytes(Box<[u8]>, Box<[u8]>),
}

impl KeyRange {
    /// The range widened to hold `other`.
    pub(crate) fn union(&self, other: &Self) -> Option<Self> {
        Some(match (self, other) {
            (Self::Int(a, b), Self::Int(c, d)) => Self::Int(*a.min(c), *b.max(d)),
            (Self::Decimal(a, b), Self::Decimal(c, d)) => Self::Decimal(*a.min(c), *b.max(d)),
            (Self::Float(a, b), Self::Float(c, d)) => Self::Float(a.min(*c), b.max(*d)),
            (Self::Bytes(a, b), Self::Bytes(c, d)) => {
                Self::Bytes(a.min(c).clone(), b.max(d).clone())
            }
            _ => return None,
        })
    }

    /// The range's bounds as literals of `key_type`, for statistics.
    pub(crate) fn scalars(&self, key_type: &DataType) -> Option<(ScalarValue, ScalarValue)> {
        use DataType as D;
        let int = |v: i64| -> Option<ScalarValue> {
            Some(match key_type {
                D::Boolean => ScalarValue::Boolean(v != 0),
                D::Int8 => ScalarValue::Int8(i8::try_from(v).ok()?),
                D::Int16 => ScalarValue::Int16(i16::try_from(v).ok()?),
                D::Int32 => ScalarValue::Int32(i32::try_from(v).ok()?),
                D::Int64 => ScalarValue::Int64(v),
                D::Date32 => ScalarValue::Date32(i32::try_from(v).ok()?),
                D::Timestamp(..) => ScalarValue::TimestampMicros(v),
                _ => return None,
            })
        };
        Some(match (self, key_type) {
            (Self::Int(a, b), _) => (int(*a)?, int(*b)?),
            (Self::Decimal(a, b), D::Decimal128(p, s)) => {
                let d = |u: i128| ScalarValue::Decimal128 {
                    unscaled: u,
                    precision: *p,
                    scale: *s,
                };
                (d(*a), d(*b))
            }
            #[expect(clippy::cast_possible_truncation, reason = "widened float32 values")]
            (Self::Float(a, b), D::Float32) => (
                ScalarValue::Float32(*a as f32),
                ScalarValue::Float32(*b as f32),
            ),
            (Self::Float(a, b), D::Float64) => (ScalarValue::Float64(*a), ScalarValue::Float64(*b)),
            (Self::Bytes(a, b), D::Utf8 | D::LargeUtf8 | D::Utf8View) => (
                ScalarValue::Utf8(String::from_utf8(a.to_vec()).ok()?),
                ScalarValue::Utf8(String::from_utf8(b.to_vec()).ok()?),
            ),
            (Self::Bytes(a, b), _) => (
                ScalarValue::Binary(a.to_vec()),
                ScalarValue::Binary(b.to_vec()),
            ),
            _ => return None,
        })
    }
}

/// A value-set member as a literal of `key_type`, for statistics.
pub(crate) fn value_scalar(v: &KeyValue, key_type: &DataType) -> Option<ScalarValue> {
    use DataType as D;
    Some(match (v, key_type) {
        (KeyValue::Fixed(n), D::Decimal128(p, s)) => ScalarValue::Decimal128 {
            unscaled: *n,
            precision: *p,
            scale: *s,
        },
        (KeyValue::Fixed(n), D::Float32) => {
            ScalarValue::Float32(f32::from_bits(u32::try_from(*n).ok()?))
        }
        (KeyValue::Fixed(n), D::Float64) => {
            ScalarValue::Float64(f64::from_bits(u64::try_from(*n).ok()?))
        }
        (KeyValue::Fixed(n), _) => {
            let v = i64::try_from(*n).ok()?;
            KeyRange::Int(v, v).scalars(key_type)?.0
        }
        (KeyValue::Bytes(b), D::Utf8 | D::LargeUtf8 | D::Utf8View) => {
            ScalarValue::Utf8(String::from_utf8(b.to_vec()).ok()?)
        }
        (KeyValue::Bytes(b), _) => ScalarValue::Binary(b.to_vec()),
    })
}
