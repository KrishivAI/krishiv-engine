// SPDX-License-Identifier: Apache-2.0
//! HyperLogLog++ (`approx_count_distinct`): Spark's
//! `HyperLogLogPlusPlusHelper` at tag v4.2.0, one-to-one.
//!
//! The partial state is the register array (one byte per register;
//! Spark packs six bits into words, which merges identically);
//! `merge` is element-wise `max`, so the operator runs partial,
//! exchange, final phases unchanged. The addressing bits `p` come from
//! `relativeSD` by Spark's formula (`p` of at least 4); precisions
//! above 18 need no bias tables (as in Spark, whose tables stop at 18),
//! and precisions above 30 are refused (Spark shifts `1 << p` as an
//! `Int` there, which overflows).
//!
//! Hashing is XXH64 with seed 42 over Spark's per-type bytes
//! (`XxHash64Function.hash` with `isCollationAware`, which is the raw
//! bytes for the default collation), after the float normalizer
//! (canonical NaN, `-0.0` as `0.0`).

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};

use crate::error::{Result, internal};

/// Spark's HLL++ seed.
const SEED: i64 = 42;

/// The largest addressing bits kept (see the module docs).
const MAX_P: u32 = 30;

/// Whether `t` is a type `approx_count_distinct` hashes here: Spark
/// hashes every type, but the engine has no calendar interval, nested or
/// collated-string hashing in this path yet.
pub(crate) fn hll_supported(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Date32
            | DataType::Timestamp(TimeUnit::Microsecond, _)
            | DataType::Interval(IntervalUnit::YearMonth)
            | DataType::Duration(TimeUnit::Microsecond)
            | DataType::Utf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::BinaryView
    )
}

/// Spark's addressing bits for `relativeSD`: `p` in `4..=30`, else
/// `None` (Spark's `require` fails below 4; above 30 its `1 << p`
/// overflows).
pub(crate) fn precision_for(relative_sd: f64) -> Option<u32> {
    if !relative_sd.is_finite() || relative_sd <= 0.0 {
        return None;
    }
    let p = (2.0 * (1.106 / relative_sd).ln() / std::f64::consts::LN_2).ceil();
    if !p.is_finite() || p < 4.0 || p > f64::from(MAX_P) {
        return None;
    }
    // `p` is in `4..=30` (checked above).
    #[expect(clippy::cast_possible_truncation, reason = "p is in 4..=30")]
    #[expect(clippy::cast_sign_loss, reason = "p is in 4..=30")]
    Some(p as u32)
}

/// The HyperLogLog++ state of one aggregate: per-group registers.
#[derive(Debug)]
pub(crate) struct HllAcc {
    regs: Vec<Vec<u8>>,
    p: u32,
    m: usize,
}

impl HllAcc {
    pub(crate) fn new(p: u32) -> Self {
        let m = 1_usize << p.min(MAX_P);
        Self {
            regs: Vec::new(),
            p,
            m,
        }
    }

    pub(crate) fn resize(&mut self, n: usize) {
        self.regs.resize(n, vec![0; self.m]);
    }

    pub(crate) fn reserve(&mut self, new: usize) {
        self.regs.reserve(new);
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        self.regs
            .iter()
            .map(|r| r.capacity().saturating_mul(1))
            .sum::<usize>()
    }

    /// The projected size after adding groups.
    pub(crate) fn projected(&self, new: usize) -> usize {
        self.size() + new.saturating_mul(self.m)
    }

    /// Folds the hash `x` of one value into group `g` (Spark's `update`).
    pub(crate) fn add(&mut self, g: usize, x: u64) {
        // The `p`-bit register index (`p <= 30`).
        #[expect(clippy::cast_possible_truncation, reason = "a p-bit index fits")]
        let idx = (x >> (u64::BITS - self.p)) as usize;
        let w = (x << self.p) | (1 << (self.p - 1));
        // The leading zeros plus one (`w` is nonzero, so at most 61).
        #[expect(clippy::cast_possible_truncation, reason = "at most 61")]
        let pw = (w.leading_zeros() + 1).min(u32::from(u8::MAX)) as u8;
        let r = &mut self.regs[g][idx];
        *r = (*r).max(pw);
    }

    /// Merges another register array into group `g` (Spark's `merge`).
    pub(crate) fn merge_regs(&mut self, g: usize, other: &[u8]) -> Result<()> {
        if other.len() != self.m {
            return Err(internal(
                "a HyperLogLog register array of another length than the registers".to_owned(),
            ));
        }
        for (r, &o) in self.regs[g].iter_mut().zip(other.iter()) {
            *r = (*r).max(o);
        }
        Ok(())
    }

    /// Spark's `query`: the estimated cardinality of group `g`.
    pub(crate) fn estimate(&self, g: usize) -> i64 {
        let (p, m) = (self.p, self.m as f64);
        let mut z_inverse = 0.0_f64;
        let mut zeros = 0.0_f64;
        for &r in &self.regs[g] {
            z_inverse += 1.0 / (1_u64 << r.min(63)) as f64;
            if r == 0 {
                zeros += 1.0;
            }
        }
        let e = alpha_m2(self.p, self.m) / z_inverse;
        let corrected = if p < 19 && e < 5.0 * m {
            e - estimate_bias(p, e)
        } else {
            e
        };
        let estimate = if zeros > 0.0 {
            let h = m * (m / zeros).ln();
            if (p < 19 && h <= THRESHOLDS[(p - 4) as usize]) || e <= 2.5 * m {
                h
            } else {
                corrected
            }
        } else {
            corrected
        };
        // `Math.round`, which saturates like Rust's `as`.
        #[expect(clippy::cast_possible_truncation, reason = "Math.round saturates")]
        let rounded = estimate.round() as i64;
        rounded
    }

    /// The partial state of groups `ids`: the register bytes.
    pub(crate) fn state(&self, ids: &[u32]) -> Vec<ArrayRef> {
        let mut out = Vec::with_capacity(ids.len());
        for &i in ids {
            out.push(self.regs[i as usize].clone());
        }
        vec![Arc::new(BinaryArray::from_iter_values(out.iter().map(Vec::as_slice))) as ArrayRef]
    }

    /// Merges the partial-state register arrays into `groups`.
    pub(crate) fn merge(&mut self, groups: &[u32], states: &BinaryArray) -> Result<()> {
        if states.len() != groups.len() {
            return Err(internal(
                "a HyperLogLog state of another length than its groups".to_owned(),
            ));
        }
        for (row, &g) in groups.iter().enumerate() {
            if states.is_null(row) {
                continue;
            }
            self.merge_regs(g as usize, states.value(row))?;
        }
        Ok(())
    }
}

/// Spark's `alpha * m * m`.
fn alpha_m2(p: u32, m: usize) -> f64 {
    let m = m as f64;
    match p {
        4 => 0.673 * m * m,
        5 => 0.697 * m * m,
        6 => 0.709 * m * m,
        _ => (0.7213 / (1.0 + 1.079 / m)) * m * m,
    }
}

/// `java.util.Arrays.binarySearch(double[], double)`: the index of a match,
/// else the insertion point. Spark's raw-estimate rows are not all sorted
/// (precision 5 holds `128.3464, 128.3462`), and on such rows the index a
/// search returns depends on the exact bisection, so Java's is ported as it
/// is.
fn java_binary_search(a: &[f64], key: f64) -> usize {
    let (mut low, mut high) = (0_usize, a.len());
    // `[low, high)` is the window still to search.
    while low < high {
        let mid = low + (high - low) / 2;
        let mid_val = a[mid];
        if mid_val < key {
            low = mid + 1;
        } else if mid_val > key {
            high = mid;
        } else {
            let (mid_bits, key_bits) =
                (mid_val.to_bits().cast_signed(), key.to_bits().cast_signed());
            if mid_bits == key_bits {
                return mid;
            } else if mid_bits < key_bits {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
    }
    low
}

/// Spark's `estimateBias`: KNN (`K = 6`) interpolation over the bias
/// table of precision `p`.
fn estimate_bias(p: u32, e: f64) -> f64 {
    const K: usize = 6;
    let estimates = RAW_ESTIMATE_DATA[(p - 4) as usize];
    let biases = BIAS_DATA[(p - 4) as usize];
    let n = estimates.len();
    let nearest = java_binary_search(estimates, e);
    let dist = |i: usize| {
        let d = e - estimates[i.min(n - 1)];
        d * d
    };
    let mut low = nearest.saturating_sub(K - 1);
    let mut high = (low + K).min(n);
    while high < n && dist(high) < dist(low) {
        low += 1;
        high += 1;
    }
    let mut sum = 0.0;
    for b in &biases[low..high] {
        sum += b;
    }
    sum / (high - low).max(1) as f64
}

/// XXH64 of one value with Spark's HLL++ seed (42), over Spark's
/// per-type bytes: `None` for a NULL (skipped), an internal error for
/// a type [`hll_supported`] rejects.
pub(crate) fn hash_of(a: &ArrayRef, i: usize) -> Result<Option<u64>> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{
        Date32Type, Decimal64Type, Decimal128Type, DurationMicrosecondType, Float32Type, Int8Type,
        Int16Type, Int32Type, Int64Type, IntervalYearMonthType, TimestampMicrosecondType,
    };
    use krishiv_kernels::hash::routing::{xxh64_seed, xxh64_seed_int, xxh64_seed_long};
    if a.is_null(i) {
        return Ok(None);
    }
    Ok(Some(
        match a.data_type() {
            DataType::Boolean => xxh64_seed_int(i32::from(a.as_boolean().value(i)), SEED),
            DataType::Int8 => {
                xxh64_seed_int(i32::from(a.as_primitive::<Int8Type>().value(i)), SEED)
            }
            DataType::Int16 => {
                xxh64_seed_int(i32::from(a.as_primitive::<Int16Type>().value(i)), SEED)
            }
            DataType::Int32 => xxh64_seed_int(a.as_primitive::<Int32Type>().value(i), SEED),
            DataType::Int64 => xxh64_seed_long(a.as_primitive::<Int64Type>().value(i), SEED),
            DataType::Float32 => {
                let v = a.as_primitive::<Float32Type>().value(i);
                let v = if v.is_nan() {
                    f32::NAN
                } else if v == 0.0 {
                    0.0
                } else {
                    v
                };
                xxh64_seed_int(v.to_bits().cast_signed(), SEED)
            }
            DataType::Float64 => {
                let v = a.as_primitive::<arrow_array::types::Float64Type>().value(i);
                let v = if v.is_nan() {
                    f64::NAN
                } else if v == 0.0 {
                    0.0
                } else {
                    v
                };
                xxh64_seed_long(v.to_bits().cast_signed(), SEED)
            }
            DataType::Decimal64(p, _) if *p <= 18 => {
                xxh64_seed_long(a.as_primitive::<Decimal64Type>().value(i), SEED)
            }
            DataType::Decimal128(p, _) if *p <= 18 => {
                // Precision 18 fits an `i64` (`DecimalType::MAX_LONG_DIGITS`).
                let v = i64::try_from(a.as_primitive::<Decimal128Type>().value(i))
                    .map_err(|_| internal("a HyperLogLog decimal of precision 18".to_owned()))?;
                xxh64_seed_long(v, SEED)
            }
            DataType::Decimal128(..) => {
                let v = a.as_primitive::<Decimal128Type>().value(i);
                xxh64_seed(&minimal_twos_complement(v), SEED)
            }
            DataType::Decimal64(..) => {
                let v = i128::from(a.as_primitive::<Decimal64Type>().value(i));
                xxh64_seed(&minimal_twos_complement(v), SEED)
            }
            DataType::Date32 => xxh64_seed_int(a.as_primitive::<Date32Type>().value(i), SEED),
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                xxh64_seed_long(a.as_primitive::<TimestampMicrosecondType>().value(i), SEED)
            }
            DataType::Interval(IntervalUnit::YearMonth) => {
                xxh64_seed_int(a.as_primitive::<IntervalYearMonthType>().value(i), SEED)
            }
            DataType::Duration(TimeUnit::Microsecond) => {
                xxh64_seed_long(a.as_primitive::<DurationMicrosecondType>().value(i), SEED)
            }
            DataType::Utf8View => xxh64_seed(a.as_string_view().value(i).as_bytes(), SEED),
            DataType::Utf8 => xxh64_seed(a.as_string::<i32>().value(i).as_bytes(), SEED),
            DataType::BinaryView => xxh64_seed(a.as_binary_view().value(i), SEED),
            DataType::Binary => xxh64_seed(a.as_binary::<i32>().value(i), SEED),
            t => return Err(internal(format!("a HyperLogLog value of type {t}"))),
        }
        .cast_unsigned(),
    ))
}

/// The minimal big-endian two's-complement bytes of `v` (as Java's
/// `BigInteger.toByteArray` gives them).
fn minimal_twos_complement(v: i128) -> Vec<u8> {
    let full = v.to_be_bytes();
    let mut start = 0;
    while start < 15 {
        let (a, b) = (full[start], full[start + 1]);
        if (a == 0x00 && b & 0x80 == 0) || (a == 0xFF && b & 0x80 != 0) {
            start += 1;
        } else {
            break;
        }
    }
    full[start..].to_vec()
}
// SPDX-License-Identifier: Apache-2.0
// HyperLogLog++ tables: bias-correction data transcribed verbatim from
// `HyperLogLogPlusPlusHelper.scala` at tag v4.2.0 (see above for the algorithm).
// The decimal literals below parse to the same `f64`s as the Scala `Double`
// literals (both correctly rounded).

#[cfg(test)]
mod tests;

mod tables;
pub(in crate::aggregate) use tables::*;
