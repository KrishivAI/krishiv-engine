// SPDX-License-Identifier: Apache-2.0
//! Iceberg single-value binary serialisation (table spec appendix D) decoded
//! **by the stored byte width** into the read type (review-1 S10, as Java's
//! `Conversions.fromByteBuffer`): a `long` bound of 4 bytes is an `int` written
//! before the column was promoted, a `double` bound of 4 bytes a `float`; a
//! decimal is read at the read type's scale (promotion widens only the
//! precision). A width that the read type cannot have, or any other failure,
//! gives `None`: that one bound is dropped and the file is kept by it.
//!
//! The values are `krishiv_io`'s [`ScalarValue`]s, in the types Spark sees
//! (`uuid` as its hyphenated string, `fixed` as binary).

use std::cmp::Ordering;

use krishiv_io::ScalarValue;

use super::types::IcebergType;

const MICROS_PER_DAY: i64 = 86_400_000_000;

/// `bytes` (a lower or upper bound, a partition value or a partition summary
/// bound) as a value of `ty`, or `None` when the width or content cannot be one.
#[must_use]
pub fn decode(bytes: &[u8], ty: &IcebergType) -> Option<ScalarValue> {
    use IcebergType as T;
    Some(match ty {
        T::Boolean => match bytes {
            [b] => ScalarValue::Boolean(*b != 0),
            _ => return None,
        },
        T::Int => ScalarValue::Int32(i32::from_le_bytes(bytes.try_into().ok()?)),
        T::Long => match bytes.len() {
            4 => ScalarValue::Int64(i64::from(i32::from_le_bytes(bytes.try_into().ok()?))),
            _ => ScalarValue::Int64(i64::from_le_bytes(bytes.try_into().ok()?)),
        },
        T::Float => ScalarValue::Float32(f32::from_le_bytes(bytes.try_into().ok()?)),
        T::Double => match bytes.len() {
            4 => ScalarValue::Float64(f64::from(f32::from_le_bytes(bytes.try_into().ok()?))),
            _ => ScalarValue::Float64(f64::from_le_bytes(bytes.try_into().ok()?)),
        },
        T::Decimal { precision, scale } => ScalarValue::Decimal128 {
            unscaled: i128_from_be(bytes)?,
            precision: *precision,
            scale: i8::try_from(*scale).ok()?,
        },
        T::Date => ScalarValue::Date32(i32::from_le_bytes(bytes.try_into().ok()?)),
        T::Timestamp | T::Timestamptz => match bytes.len() {
            // A `date` promoted to a timestamp (format v3): the spec reads the day
            // count as midnight of that day (Java reads the raw count, FC S5; no
            // 1.12 writer can produce it).
            4 => ScalarValue::TimestampMicros(
                i64::from(i32::from_le_bytes(bytes.try_into().ok()?))
                    .checked_mul(MICROS_PER_DAY)?,
            ),
            _ => ScalarValue::TimestampMicros(i64::from_le_bytes(bytes.try_into().ok()?)),
        },
        T::String => ScalarValue::Utf8(std::str::from_utf8(bytes).ok()?.to_owned()),
        T::Uuid => ScalarValue::Utf8(uuid_string(bytes.try_into().ok()?)),
        T::Fixed(_) | T::Binary => ScalarValue::Binary(bytes.to_vec()),
        // Spark refuses these types; nothing compares against them.
        T::Time | T::TimestampNs | T::TimestamptzNs => return None,
    })
}

/// A big-endian two's-complement integer of 1 to 16 bytes.
fn i128_from_be(bytes: &[u8]) -> Option<i128> {
    let (&first, _) = bytes.split_first()?;
    if bytes.len() > 16 {
        return None;
    }
    let fill = if first & 0x80 == 0 { 0 } else { 0xff };
    let mut buf = [fill; 16];
    buf[16 - bytes.len()..].copy_from_slice(bytes);
    Some(i128::from_be_bytes(buf))
}

fn uuid_string(b: [u8; 16]) -> String {
    let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// SQL order of two values of the same type (strings and binaries by bytes);
/// `None` when they are not comparable (different types or scales, NaN).
#[must_use]
pub(crate) fn compare(a: &ScalarValue, b: &ScalarValue) -> Option<Ordering> {
    use ScalarValue as S;
    match (a, b) {
        (S::Boolean(x), S::Boolean(y)) => Some(x.cmp(y)),
        (S::Int8(x), S::Int8(y)) => Some(x.cmp(y)),
        (S::Int16(x), S::Int16(y)) => Some(x.cmp(y)),
        (S::Int32(x), S::Int32(y)) | (S::Date32(x), S::Date32(y)) => Some(x.cmp(y)),
        (S::Int64(x), S::Int64(y)) | (S::TimestampMicros(x), S::TimestampMicros(y)) => {
            Some(x.cmp(y))
        }
        (S::Float32(x), S::Float32(y)) => x.partial_cmp(y),
        (S::Float64(x), S::Float64(y)) => x.partial_cmp(y),
        (
            S::Decimal128 {
                unscaled: x,
                scale: sx,
                ..
            },
            S::Decimal128 {
                unscaled: y,
                scale: sy,
                ..
            },
        ) if sx == sy => Some(x.cmp(y)),
        (S::Utf8(x), S::Utf8(y)) => Some(x.as_bytes().cmp(y.as_bytes())),
        (S::Binary(x), S::Binary(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// Whether `v` is NaN (only floats can be).
#[must_use]
pub(crate) fn is_nan(v: &ScalarValue) -> bool {
    match v {
        ScalarValue::Float32(f) => f.is_nan(),
        ScalarValue::Float64(f) => f.is_nan(),
        _ => false,
    }
}

/// Literals whose order in statistics does not match SQL comparison: NaN
/// (greatest in Spark, unordered in bounds), signed zeros (equal in SQL,
/// distinct in some writers' bounds) and `NULL`.
#[must_use]
pub(crate) fn is_unordered(v: &ScalarValue) -> bool {
    match v {
        ScalarValue::Float32(f) => f.is_nan() || *f == 0.0,
        ScalarValue::Float64(f) => f.is_nan() || *f == 0.0,
        ScalarValue::Null => true,
        _ => false,
    }
}

/// The spec's single-value serialisation of `v` as a value of `ty` (the inverse of
/// [`decode`] at the type's full width). Used by tests and by T027's writers.
#[must_use]
pub fn encode(v: &ScalarValue, ty: &IcebergType) -> Option<Vec<u8>> {
    use IcebergType as T;
    use ScalarValue as S;
    Some(match (ty, v) {
        (T::Boolean, S::Boolean(b)) => vec![u8::from(*b)],
        (T::Int, S::Int32(x)) | (T::Date, S::Date32(x)) => x.to_le_bytes().to_vec(),
        (T::Long, S::Int64(x)) | (T::Timestamp | T::Timestamptz, S::TimestampMicros(x)) => {
            x.to_le_bytes().to_vec()
        }
        (T::Float, S::Float32(x)) => x.to_le_bytes().to_vec(),
        (T::Double, S::Float64(x)) => x.to_le_bytes().to_vec(),
        (T::Decimal { .. }, S::Decimal128 { unscaled, .. }) => min_be_bytes(*unscaled),
        (T::String, S::Utf8(s)) => s.as_bytes().to_vec(),
        (T::Fixed(_) | T::Binary, S::Binary(b)) => b.clone(),
        _ => return None,
    })
}

/// The shortest big-endian two's-complement bytes of `v` (Java's
/// `BigInteger.toByteArray`).
#[must_use]
pub(crate) fn min_be_bytes(v: i128) -> Vec<u8> {
    let b = v.to_be_bytes();
    let mut start = 0;
    while start < 15 {
        let (cur, next) = (b[start], b[start + 1]);
        let redundant = (cur == 0 && next & 0x80 == 0) || (cur == 0xff && next & 0x80 != 0);
        if !redundant {
            break;
        }
        start += 1;
    }
    b[start..].to_vec()
}
