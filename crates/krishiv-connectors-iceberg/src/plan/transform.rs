// SPDX-License-Identifier: Apache-2.0
//! Partition transforms applied to a predicate literal (table spec "Partition
//! Transforms" and appendix B for `bucket`), so a predicate on a source column
//! can be projected onto the partition values (inclusive projection, in
//! `prune`). Every transform other than `bucket` is monotone: `a <= b` implies
//! `t(a) <= t(b)`, which is what the projection relies on.

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_io::ScalarValue;

use super::bounds::min_be_bytes;
use super::types::{IcebergType, Transform};

const MICROS_PER_DAY: i64 = 86_400_000_000;
const MICROS_PER_HOUR: i64 = 3_600_000_000;

/// `transform` of `value`, a value of the source column whose type is `source`;
/// `None` when the transform does not apply to it (then nothing is pruned).
#[must_use]
pub fn apply(
    transform: Transform,
    source: &IcebergType,
    value: &ScalarValue,
) -> Option<ScalarValue> {
    use ScalarValue as S;
    // A literal of another type than the source (a decimal at another scale
    // above all) would hash or truncate to another value and prune wrongly;
    // T027's validation passes literals that were not coerced (review F7).
    if !fits(source, value) {
        return None;
    }
    match transform {
        Transform::Identity => Some(value.clone()),
        Transform::Bucket(n) => {
            let n = i32::try_from(n).ok().filter(|n| *n > 0)?;
            let h = bucket_hash(source, value)?;
            Some(S::Int32((h & i32::MAX) % n))
        }
        Transform::Truncate(w) => truncate(w, value),
        Transform::Year | Transform::Month | Transform::Day | Transform::Hour => {
            temporal(transform, value)
        }
        Transform::Void | Transform::Unknown => None,
    }
}

/// The partition value `transform` gives a row whose source column (of type
/// `source`) holds `value` (`None`: null), for a write (design note §6.1).
/// Unlike [`apply`], whose `None` means "cannot tell" and keeps a file, this
/// separates a null partition value (`void`, or a null source) from a
/// transform Krishiv cannot compute, which must fail the write rather than put
/// the row in the null partition.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` naming the transform and the source type.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the write task of PR-36a (later commit) calls it")
)]
pub(crate) fn partition_value(
    transform: Transform,
    source: &IcebergType,
    value: Option<&ScalarValue>,
) -> Result<Option<ScalarValue>, KrishivError> {
    if matches!(transform, Transform::Void) {
        return Ok(None);
    }
    let unsupported = || -> KrishivError {
        KrishivUnsupportedFeature {
            feature: format!(
                "writing the Iceberg partition transform {transform:?} of a {source:?} column"
            ),
        }
        .into()
    };
    if matches!(transform, Transform::Unknown) {
        return Err(unsupported());
    }
    match value {
        None => Ok(None),
        Some(v) => apply(transform, source, v)
            .map(Some)
            .ok_or_else(unsupported),
    }
}

/// Whether `value` is a value of the Iceberg type `source` as the reader types
/// it (`uuid` as its string, `fixed` as binary; a decimal at the same scale).
fn fits(source: &IcebergType, value: &ScalarValue) -> bool {
    use IcebergType as T;
    use ScalarValue as S;
    match (source, value) {
        (T::Boolean, S::Boolean(_))
        | (T::Int, S::Int32(_))
        | (T::Date, S::Date32(_))
        | (T::Long, S::Int64(_))
        | (T::Float, S::Float32(_))
        | (T::Double, S::Float64(_))
        | (T::Timestamp | T::Timestamptz, S::TimestampMicros(_))
        | (T::String | T::Uuid, S::Utf8(_))
        | (T::Fixed(_) | T::Binary, S::Binary(_)) => true,
        (T::Decimal { scale, .. }, S::Decimal128 { scale: s, .. }) => {
            i16::from(*scale) == i16::from(*s)
        }
        _ => false,
    }
}

/// Murmur3 of the value's bucket serialisation (spec appendix B), as a signed
/// 32-bit value.
pub(crate) fn bucket_hash(source: &IcebergType, value: &ScalarValue) -> Option<i32> {
    use IcebergType as T;
    use ScalarValue as S;
    let bytes: Vec<u8> = match (source, value) {
        (T::Int | T::Date, S::Int32(v) | S::Date32(v)) => i64::from(*v).to_le_bytes().to_vec(),
        (T::Long, S::Int64(v)) | (T::Timestamp | T::Timestamptz, S::TimestampMicros(v)) => {
            v.to_le_bytes().to_vec()
        }
        (T::Decimal { .. }, S::Decimal128 { unscaled, .. }) => min_be_bytes(*unscaled),
        (T::String, S::Utf8(s)) => s.as_bytes().to_vec(),
        (T::Uuid, S::Utf8(s)) => uuid_bytes(s)?.to_vec(),
        (T::Fixed(_) | T::Binary, S::Binary(b)) => b.clone(),
        _ => return None,
    };
    Some(i32::from_ne_bytes(murmur3_32(&bytes).to_ne_bytes()))
}

fn uuid_bytes(s: &str) -> Option<[u8; 16]> {
    let hex: Vec<u8> = s.bytes().filter(|b| *b != b'-').collect();
    if hex.len() != 32 || s.len() != 36 {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, pair) in hex.as_chunks::<2>().0.iter().enumerate() {
        let text = std::str::from_utf8(pair).ok()?;
        out[i] = u8::from_str_radix(text, 16).ok()?;
    }
    Some(out)
}

/// Murmur3 x86 32-bit, seed 0 (Guava's `murmur3_32`, which Iceberg uses).
#[must_use]
pub(crate) fn murmur3_32(data: &[u8]) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mix = |k: u32| k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
    let mut h: u32 = 0;
    let (chunks, tail) = data.as_chunks::<4>();
    for c in chunks {
        let k = u32::from_le_bytes(*c);
        h ^= mix(k);
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    if !tail.is_empty() {
        let mut k = 0u32;
        for (i, b) in tail.iter().enumerate() {
            k |= u32::from(*b) << (8 * i);
        }
        h ^= mix(k);
    }
    // Guava mixes in the length as an int.
    h ^= u32::try_from(data.len()).unwrap_or(u32::MAX);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// `truncate[W]`: integers and decimals to the multiple of `W` at or below the
/// value (Java's wrapping `int`/`long` arithmetic), strings to `W` code points,
/// binaries to `W` bytes.
fn truncate(w: u32, value: &ScalarValue) -> Option<ScalarValue> {
    use ScalarValue as S;
    if w == 0 {
        return None;
    }
    Some(match value {
        S::Int32(v) => {
            let w = i32::try_from(w).ok()?;
            S::Int32(v.wrapping_sub((v % w).wrapping_add(w) % w))
        }
        S::Int64(v) => {
            let w = i64::from(w);
            S::Int64(v.wrapping_sub((v % w).wrapping_add(w) % w))
        }
        S::Decimal128 {
            unscaled,
            precision,
            scale,
        } => {
            let w = i128::from(w);
            S::Decimal128 {
                unscaled: unscaled.checked_sub(((unscaled % w) + w) % w)?,
                precision: *precision,
                scale: *scale,
            }
        }
        S::Utf8(s) => S::Utf8(truncate_str(s, w)),
        S::Binary(b) => S::Binary(b.iter().take(usize::try_from(w).ok()?).copied().collect()),
        _ => return None,
    })
}

/// The first `w` code points of `s`.
#[must_use]
pub(crate) fn truncate_str(s: &str, w: u32) -> String {
    s.chars()
        .take(usize::try_from(w).unwrap_or(usize::MAX))
        .collect()
}

fn temporal(t: Transform, value: &ScalarValue) -> Option<ScalarValue> {
    use ScalarValue as S;
    let (days, micros) = match value {
        S::Date32(d) => (i64::from(*d), None),
        S::TimestampMicros(us) => (us.div_euclid(MICROS_PER_DAY), Some(*us)),
        _ => return None,
    };
    let out = match t {
        Transform::Day => return Some(S::Date32(i32::try_from(days).ok()?)),
        Transform::Hour => micros?.div_euclid(MICROS_PER_HOUR),
        Transform::Year => civil_from_days(days).0 - 1970,
        Transform::Month => {
            let (y, m) = civil_from_days(days);
            (y - 1970) * 12 + (m - 1)
        }
        _ => return None,
    };
    Some(S::Int32(i32::try_from(out).ok()?))
}

/// Year and month (1–12) of the proleptic Gregorian day `days` since 1970-01-01
/// (H. Hinnant's `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m)
}

/// Whether `t` is one of the time transforms whose pre-epoch values some old Java
/// writers rounded toward zero instead of down (Iceberg's
/// `fixInclusiveTimeProjection` widens the projection for them).
#[must_use]
pub(crate) fn is_time(t: Transform) -> bool {
    matches!(
        t,
        Transform::Year | Transform::Month | Transform::Day | Transform::Hour
    )
}
