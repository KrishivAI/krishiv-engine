// SPDX-License-Identifier: Apache-2.0
//! The routing hash `H_kg` and key groups (design P2-T171 §5.1 and §7.2,
//! frozen in P2-T157 Appendix C; P2-T175).
//!
//! `h = H_kg(keys)` is a **seedless** 64-bit hash, computed column-wise once
//! per row, that every operator instance, worker and process computes
//! alike. It is in every checkpoint key (`(key group, key)`), so it is fixed
//! and versioned: this module is version 1, and a change is a new version
//! named in the plan (the `key_group_hash` feature of `krishiv.plan.v1`).
//!
//! # Definition (version 1)
//! - A column's value is hashed as XXH64, seed 0, of its **normalized key
//!   bytes**, the bytes of the runtime filters' single normalizer (P2-T163,
//!   the Parquet Bloom-filter hash):
//!
//! | Key type | Bytes hashed |
//! |---|---|
//! | `boolean` | one byte, 0 or 1 |
//! | `tinyint` .. `bigint`, `date`, timestamps, intervals | the value's own width, little-endian |
//! | `float`, `double` | the 4 or 8 bytes of the bits, after `-0.0` → `0.0` and one canonical NaN |
//! | `decimal(p, s)` | the unscaled value as 16 bytes, little-endian, whatever its carrier width |
//! | `string`, `binary` | the bytes |
//!
//!   A NULL hashes to [`NULL_HASH`]. The bytes are written out
//!   little-endian whatever the machine's byte order, so `h` is the same on
//!   every platform.
//! - **One key column:** `h` is that column's hash, bit-identical to the
//!   runtime filters' Bloom hash (the Parquet Bloom-filter hash).
//! - **Several key columns:** `h = FOLD_INIT`, then for each column in key
//!   order `h = mix(h ^ h_col)` ([`fold_column`]), a folded multiply by a
//!   fixed odd constant. Starting from a constant (not from the first
//!   column's hash) keeps the fold order-dependent and keeps rows whose
//!   columns are equal (`(a, a)`) from collapsing to one value.
//!
//! Equal logical keys hash equally whatever their physical carrier:
//! dictionary, view and offset arrays of one type agree
//! ([`hash_columns`]).
//!
//! # Bits of `h`
//! - top [`KEY_GROUP_BITS`] bits: the **key group** ([`key_group`]);
//! - top `k` bits: the partition of an exchange of `2^k` partitions; for
//!   any count `n <= 4096` an exchange partition owns a contiguous range of key
//!   groups ([`partition_of`], [`key_group_range`]), so rescaling and AQE
//!   coalescing move or merge whole ranges and never rehash;
//! - top `R0` bits: a radix partition of `2^R0` ([`radix`]), a union of
//!   exchange partitions when `n = 2^k` with `k >= R0`.
//!
//! In-memory hash tables (aggregate, join) do not use `H_kg`: they keep
//! the row layout's faster seedless table hash, which never leaves the
//! process, and place rows by a seeded finalizer of it, local to one
//! operator instance (P2-T157 Appendix C; the P2-T171 §5.1 fallback).
//! `H_kg` is computed only where a value is observable across processes:
//! exchange routing, key groups, and anything in a plan or in state.

use std::ops::Range;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Date64Type, Decimal32Type, Decimal64Type, Decimal128Type, DurationMicrosecondType,
    DurationMillisecondType, DurationNanosecondType, DurationSecondType, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, IntervalYearMonthType, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrowPrimitiveType};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};

use super::SparkHasher;
use super::xxh64::Xxh64;
use crate::error::KernelError;

/// Bits of `h` that name a key group (its top bits).
pub const KEY_GROUP_BITS: u32 = 12;
/// Key groups: 4096 (ADR 0020; pending E20).
pub const KEY_GROUPS: u32 = 1 << KEY_GROUP_BITS;

/// The hash of a NULL column value (version 1).
pub const NULL_HASH: u64 = 0xA5A5_5A5A_C3C3_3C3C;
/// The start of the fold over several key columns (version 1; the
/// fractional digits of pi).
pub const FOLD_INIT: u64 = 0x243F_6A88_85A3_08D3;
/// The odd multiplier of [`fold_column`] (version 1; the fractional digits
/// of e).
const FOLD_MUL: u64 = 0xB7E1_5162_8AED_2A6B;

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline]
fn avalanche(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

#[inline]
fn lane8(h: u64, k: u64) -> u64 {
    let k1 = k.wrapping_mul(P2).rotate_left(31).wrapping_mul(P1);
    (h ^ k1).rotate_left(27).wrapping_mul(P1).wrapping_add(P4)
}

/// XXH64 (seed 0) of the `N` little-endian bytes of `v` (`N` is 1, 2, 4,
/// 8 or 16): the general algorithm's short-input path, unrolled for one
/// width.
#[inline]
#[must_use]
pub fn xxh64_fixed<const N: usize>(v: u128) -> u64 {
    // seed + P5 + len.
    let mut h = P5.wrapping_add(N as u64);
    let mut rest = v;
    let mut left = N;
    while left >= 8 {
        #[expect(clippy::cast_possible_truncation, reason = "the low eight bytes")]
        let k = rest as u64;
        h = lane8(h, k);
        rest >>= 64;
        left = left.wrapping_sub(8);
    }
    if left >= 4 {
        #[expect(clippy::cast_possible_truncation, reason = "the low four bytes")]
        let k = u64::from(rest as u32);
        h = (h ^ k.wrapping_mul(P1))
            .rotate_left(23)
            .wrapping_mul(P2)
            .wrapping_add(P3);
        rest >>= 32;
        left = left.wrapping_sub(4);
    }
    while left > 0 {
        #[expect(clippy::cast_possible_truncation, reason = "one byte")]
        let b = u64::from(rest as u8);
        h = (h ^ b.wrapping_mul(P5)).rotate_left(11).wrapping_mul(P1);
        rest >>= 8;
        left = left.wrapping_sub(1);
    }
    avalanche(h)
}

/// XXH64, seed 0, of `bytes`.
#[inline]
#[must_use]
pub fn xxh64(bytes: &[u8]) -> u64 {
    Xxh64::hash_bytes(bytes, 0).cast_unsigned()
}

/// One step of the fold over several key columns: `mix(h ^ h_col)`, a
/// folded multiply (the 128-bit product's halves xor-ed).
#[inline]
#[must_use]
pub fn fold_column(h: u64, h_col: u64) -> u64 {
    let p = u128::from(h ^ h_col).wrapping_mul(u128::from(FOLD_MUL));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the low half of the product is intended"
    )]
    let low = p as u64;
    let high = (p >> 64) as u64;
    low ^ high
}

/// The key group of `h`: its top [`KEY_GROUP_BITS`] bits.
#[inline]
#[must_use]
pub fn key_group(h: u64) -> u32 {
    (h >> (64 - KEY_GROUP_BITS)) as u32
}

/// The radix partition of `h` among `2^bits` (`bits <= 32`): its top
/// `bits` bits; 0 when `bits` is 0.
#[inline]
#[must_use]
pub fn radix(h: u64, bits: u32) -> u32 {
    if bits == 0 {
        return 0;
    }
    #[expect(clippy::cast_possible_truncation, reason = "at most 32 bits")]
    let r = (h >> 64u32.wrapping_sub(bits.min(32))) as u32;
    r
}

/// The partition of `n` (a count of 0 counts as 1) that a row with
/// routing hash `h` goes to.
///
/// Up to [`KEY_GROUPS`] partitions, partition `p` owns the contiguous key
/// groups [`key_group_range(p, n)`](key_group_range), `[p·4096/n,
/// (p+1)·4096/n)`, so a key group is never split; for `n = 2^k` that is the
/// top `k` bits of `h`. Above, partition `p` is `[p, p + 1) · 2^64 / n` of
/// `h`: for `n` a multiple of [`KEY_GROUPS`] (`n = 2^k`: the top `k` bits)
/// each key group is split evenly by the next bits of `h`.
#[inline]
#[must_use]
pub fn partition_of(h: u64, n: u32) -> u32 {
    let n = n.max(1);
    if n <= KEY_GROUPS {
        // < 2^12 · 2^13: no overflow.
        key_group(h).wrapping_mul(n) >> KEY_GROUP_BITS
    } else {
        #[expect(clippy::cast_possible_truncation, reason = "below n")]
        let p = (u128::from(h).wrapping_mul(u128::from(n)) >> 64) as u32;
        p
    }
}

/// The key groups partition `p` of `n` owns (see [`partition_of`]); for
/// `n` above [`KEY_GROUPS`], the key group it starts in, its only one when
/// `n` is a multiple of [`KEY_GROUPS`]. Empty when `p >= n`.
#[must_use]
pub fn key_group_range(p: u32, n: u32) -> Range<u32> {
    let n = n.max(1);
    if p >= n {
        return 0..0;
    }
    // p < n <= 2^32 and 4096 = 2^12: no product overflows a u64.
    let groups = u64::from(KEY_GROUPS);
    let (p, n) = (u64::from(p), u64::from(n));
    if n <= groups {
        // ceil(p · 4096 / n) .. ceil((p + 1) · 4096 / n)
        let start = p.wrapping_mul(groups).div_ceil(n);
        let end = p.wrapping_add(1).wrapping_mul(groups).div_ceil(n);
        #[expect(clippy::cast_possible_truncation, reason = "at most 4096")]
        (start as u32..end as u32)
    } else {
        #[expect(clippy::cast_possible_truncation, reason = "below 4096")]
        let kg = p.wrapping_mul(groups).checked_div(n).unwrap_or(0) as u32;
        kg..kg.wrapping_add(1)
    }
}

/// `H_kg` of each row of `columns` (the key columns, in key order; all of
/// one length), written to `out` (cleared first). See the module docs.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a key type without normalized bytes
/// (nested, 256-bit decimal, time of day, month-day-nano interval);
/// `INTERNAL_ERROR` for columns of different lengths.
pub fn hash_columns(columns: &[&dyn Array], out: &mut Vec<u64>) -> Result<(), KernelError> {
    out.clear();
    let Some(first) = columns.first() else {
        return Ok(());
    };
    let n = first.len();
    if columns.iter().any(|c| c.len() != n) {
        return Err(KernelError::internal("key columns of different lengths"));
    }
    if let [only] = columns {
        out.resize(n, 0);
        return column_hashes(*only, out);
    }
    out.resize(n, FOLD_INIT);
    let mut col = vec![0; n];
    for c in columns {
        column_hashes(*c, &mut col)?;
        for (h, &hc) in out.iter_mut().zip(&col) {
            *h = fold_column(*h, hc);
        }
    }
    Ok(())
}

/// Each row's column hash of `a` into `out` (as long as `a`).
fn column_hashes(a: &dyn Array, out: &mut [u64]) -> Result<(), KernelError> {
    fn prim<T: ArrowPrimitiveType>(a: &dyn Array, out: &mut [u64], f: impl Fn(T::Native) -> u64) {
        for (dst, &v) in out.iter_mut().zip(a.as_primitive::<T>().values()) {
            *dst = f(v);
        }
    }
    fn bytes<'a>(out: &mut [u64], values: impl Iterator<Item = Option<&'a [u8]>>) {
        for (dst, v) in out.iter_mut().zip(values) {
            *dst = v.map_or(NULL_HASH, xxh64);
        }
    }
    use DataType as D;
    match a.data_type() {
        D::Null => out.fill(NULL_HASH),
        D::Boolean => {
            for (dst, v) in out.iter_mut().zip(a.as_boolean().values()) {
                *dst = xxh64_fixed::<1>(u128::from(v));
            }
        }
        D::Int8 => prim::<Int8Type>(a, out, |v| xxh64_fixed::<1>(u128::from(v.cast_unsigned()))),
        D::UInt8 => prim::<UInt8Type>(a, out, |v| xxh64_fixed::<1>(u128::from(v))),
        D::Int16 => prim::<Int16Type>(a, out, |v| xxh64_fixed::<2>(u128::from(v.cast_unsigned()))),
        D::UInt16 => prim::<UInt16Type>(a, out, |v| xxh64_fixed::<2>(u128::from(v))),
        D::Int32 => prim::<Int32Type>(a, out, i32_hash),
        D::Date32 => prim::<Date32Type>(a, out, i32_hash),
        D::Interval(IntervalUnit::YearMonth) => prim::<IntervalYearMonthType>(a, out, i32_hash),
        D::UInt32 => prim::<UInt32Type>(a, out, |v| xxh64_fixed::<4>(u128::from(v))),
        D::Int64 => prim::<Int64Type>(a, out, i64_hash),
        D::UInt64 => prim::<UInt64Type>(a, out, |v| xxh64_fixed::<8>(u128::from(v))),
        D::Date64 => prim::<Date64Type>(a, out, i64_hash),
        D::Timestamp(TimeUnit::Second, _) => prim::<TimestampSecondType>(a, out, i64_hash),
        D::Timestamp(TimeUnit::Millisecond, _) => {
            prim::<TimestampMillisecondType>(a, out, i64_hash);
        }
        D::Timestamp(TimeUnit::Microsecond, _) => {
            prim::<TimestampMicrosecondType>(a, out, i64_hash);
        }
        D::Timestamp(TimeUnit::Nanosecond, _) => prim::<TimestampNanosecondType>(a, out, i64_hash),
        D::Duration(TimeUnit::Second) => prim::<DurationSecondType>(a, out, i64_hash),
        D::Duration(TimeUnit::Millisecond) => prim::<DurationMillisecondType>(a, out, i64_hash),
        D::Duration(TimeUnit::Microsecond) => prim::<DurationMicrosecondType>(a, out, i64_hash),
        D::Duration(TimeUnit::Nanosecond) => prim::<DurationNanosecondType>(a, out, i64_hash),
        D::Decimal32(..) => prim::<Decimal32Type>(a, out, |v| decimal_hash(i128::from(v))),
        D::Decimal64(..) => prim::<Decimal64Type>(a, out, |v| decimal_hash(i128::from(v))),
        D::Decimal128(..) => prim::<Decimal128Type>(a, out, decimal_hash),
        D::Float32 => prim::<Float32Type>(a, out, f32_hash),
        D::Float64 => prim::<Float64Type>(a, out, f64_hash),
        D::Utf8 => bytes(
            out,
            a.as_string::<i32>().iter().map(|v| v.map(str::as_bytes)),
        ),
        D::LargeUtf8 => bytes(
            out,
            a.as_string::<i64>().iter().map(|v| v.map(str::as_bytes)),
        ),
        D::Utf8View => bytes(out, a.as_string_view().iter().map(|v| v.map(str::as_bytes))),
        D::Binary => bytes(out, a.as_binary::<i32>().iter()),
        D::LargeBinary => bytes(out, a.as_binary::<i64>().iter()),
        D::BinaryView => bytes(out, a.as_binary_view().iter()),
        D::FixedSizeBinary(_) => bytes(out, a.as_fixed_size_binary().iter()),
        D::Dictionary(..) => {
            // Each distinct value hashed once, then gathered by key.
            let d = a
                .as_any_dictionary_opt()
                .ok_or_else(|| super::unsupported(a.data_type()))?;
            let values = d.values();
            if values.is_empty() {
                // No value to point to: every row is NULL (and
                // `normalized_keys` would panic).
                out.fill(NULL_HASH);
                return Ok(());
            }
            let mut value_hashes = vec![0; values.len()];
            column_hashes(values.as_ref(), &mut value_hashes)?;
            for (dst, k) in out.iter_mut().zip(d.normalized_keys()) {
                *dst = value_hashes.get(k).copied().unwrap_or(NULL_HASH);
            }
        }
        t => return Err(super::unsupported(t)),
    }
    // NULL slots hold arbitrary values: overwrite them. Logical nulls
    // include a dictionary's NULL values.
    if let Some(nulls) = a.logical_nulls() {
        for i in nulls.iter().enumerate().filter(|(_, v)| !v).map(|(i, _)| i) {
            if let Some(dst) = out.get_mut(i) {
                *dst = NULL_HASH;
            }
        }
    }
    Ok(())
}

/// The column hash of a 4-byte value.
#[inline]
#[must_use]
pub fn i32_hash(v: i32) -> u64 {
    xxh64_fixed::<4>(u128::from(v.cast_unsigned()))
}

/// The column hash of an 8-byte value.
#[inline]
#[must_use]
pub fn i64_hash(v: i64) -> u64 {
    xxh64_fixed::<8>(u128::from(v.cast_unsigned()))
}

/// The column hash of a decimal's unscaled value (16 bytes, any carrier).
#[inline]
#[must_use]
pub fn decimal_hash(v: i128) -> u64 {
    xxh64_fixed::<16>(v.cast_unsigned())
}

/// The column hash of a `float`, after Spark's key normalization.
#[inline]
#[must_use]
pub fn f32_hash(v: f32) -> u64 {
    let v = if v.is_nan() { f32::NAN } else { v + 0.0 };
    xxh64_fixed::<4>(u128::from(v.to_bits()))
}

/// The column hash of a `double`, after Spark's key normalization.
#[inline]
#[must_use]
pub fn f64_hash(v: f64) -> u64 {
    let v = if v.is_nan() { f64::NAN } else { v + 0.0 };
    xxh64_fixed::<8>(u128::from(v.to_bits()))
}
