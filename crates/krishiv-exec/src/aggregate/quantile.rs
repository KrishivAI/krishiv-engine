// SPDX-License-Identifier: Apache-2.0
//! The quantile accumulators: the exact value-count map of Spark's
//! `PercentileBase` (`percentile`, `median`, `percentile_cont`,
//! `percentile_disc`) and its Greenwald-Khanna `QuantileSummaries`
//! (`percentile_approx`, `approx_percentile`), at tag v4.2.0.
//!
//! The exact accumulator keeps, per group, the distinct values with
//! their counts (frequencies fold into the counts, as Spark's
//! `OpenHashMap[AnyRef, Long]` does), sorts them at evaluation and
//! interpolates exactly as Spark's `getPercentile` does: `position =
//! (total - 1) * p`, `(higher - position) * lower + (position - lower) *
//! higher`. `percentile_disc` instead takes the value at
//! `ceil(total * p)` (or the continuous lower bound with
//! `spark.sql.legacy.percentileDiscCalculation`). Descending order sorts
//! the values down first, as Spark's `reverse` does. Partial states are
//! the values with their counts, so merging re-folds them; the state
//! columns do not depend on the input type (every value folds to a
//! `double`, as Spark converts its boundary keys).
//!
//! The approximate accumulator ports Spark's `QuantileSummaries`
//! one-to-one (the head buffer, the insertion, the compression, the
//! merge with its additional-delta adjustments, the query): the partial
//! state is the compressed summary in Spark's
//! `PercentileDigestSerializer` format (big-endian), so merging
//! deserializes and merges. Results of an approximate aggregate depend
//! on the partitioning, as Spark's do.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::ArrayBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, BinaryArray, Float64Array, Int64Array, ListArray};
use arrow_schema::{DataType, Field, TimeUnit};

use crate::error::{Result, internal};

/// The element field of the exact states and results, as Spark's
/// `array<double>` lays it out (`to_arrow`): `element`, non-nullable.
pub(crate) fn values_list_field() -> Arc<Field> {
    Arc::new(Field::new("element", DataType::Float64, false))
}

/// The element field of the exact count states (`element`, non-nullable).
pub(crate) fn counts_list_field() -> Arc<Field> {
    Arc::new(Field::new("element", DataType::Int64, false))
}

/// A `ListArray` of `values` cut by `offsets` (one list per group, all
/// present), with element field `field`.
pub(crate) fn finish_lists(values: ArrayRef, offsets: Vec<i32>, field: Arc<Field>) -> ArrayRef {
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    Arc::new(ListArray::new(
        field,
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        values,
        None,
    )) as ArrayRef
}

/// The head-buffer size of Spark's `QuantileSummaries`
/// (`defaultHeadSize`).
const HEAD_SIZE: usize = 50_000;

/// The compression threshold (`defaultCompressThreshold`).
const COMPRESS_THRESHOLD: usize = 10_000;

/// Whether `t` is a type the exact percentiles take here: Spark's
/// numeric input (`NumericAndAnsiInterval` has no interval here); the
/// approximate one also reads dates and timestamps as doubles.
pub(crate) fn quantile_supported(t: &DataType, approx: bool) -> bool {
    match t {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Float32
        | DataType::Float64
        | DataType::Decimal64(..)
        | DataType::Decimal128(..) => true,
        DataType::Date32 | DataType::Timestamp(TimeUnit::Microsecond, _) if approx => true,
        _ => false,
    }
}

/// The canonical `f64` bits of `v`: NaNs canonicalized (as Java's
/// `doubleToLongBits` canonicalizes Spark's map keys), `-0.0` kept
/// distinct from `0.0` (as Java's `Double.equals` does).
fn key(v: f64) -> u64 {
    if v.is_nan() {
        f64::NAN.to_bits()
    } else {
        v.to_bits()
    }
}

/// The exact value-count state of one aggregate: per-group maps of
/// canonical value bits to counts, with the groups' totals.
#[derive(Debug, Default)]
pub(crate) struct ExactAcc {
    maps: Vec<HashMap<u64, i64>>,
    totals: Vec<i64>,
}

impl ExactAcc {
    pub(crate) fn resize(&mut self, n: usize) {
        self.maps.resize_with(n, HashMap::new);
        self.totals.resize(n, 0);
    }

    pub(crate) fn reserve(&mut self, new: usize) {
        self.maps.reserve(new);
        self.totals.reserve(new);
    }

    /// Bytes held: the maps (an entry is two words) and the totals.
    pub(crate) fn size(&self) -> usize {
        self.maps
            .iter()
            .map(|m| m.capacity().saturating_mul(32))
            .sum::<usize>()
            + self.totals.capacity().saturating_mul(8)
    }

    /// The projected size after adding groups (empty maps).
    pub(crate) fn projected(&self, new: usize) -> usize {
        self.size() + new.saturating_mul(32 + 8)
    }

    /// Folds `count` of `v` into `g`.
    pub(crate) fn add(&mut self, g: usize, v: f64, count: i64) {
        if count == 0 {
            return;
        }
        let e = self.maps[g].entry(key(v)).or_insert(0);
        *e = e.wrapping_add(count);
        self.totals[g] = self.totals[g].wrapping_add(count);
    }

    /// The sorted values of `g` with their cumulative counts.
    fn sorted(&self, g: usize, reverse: bool) -> (Vec<f64>, Vec<i64>) {
        let mut pairs: Vec<(f64, i64)> = self.maps[g]
            .iter()
            .map(|(&k, &c)| (f64::from_bits(k), c))
            .collect();
        pairs.sort_by(|a, b| {
            if reverse {
                b.0.total_cmp(&a.0)
            } else {
                a.0.total_cmp(&b.0)
            }
        });
        let mut total = 0_i64;
        let mut cum = Vec::with_capacity(pairs.len());
        for (_, c) in &pairs {
            total = total.wrapping_add(*c);
            cum.push(total);
        }
        (pairs.into_iter().map(|(v, _)| v).collect(), cum)
    }

    /// The exact percentiles `percentages` of group `g`: `None` for no
    /// values. `cont` selects the continuous interpolation,
    /// `legacy_disc` the continuous lower bound, else the discrete
    /// value at `ceil(total * p)`.
    pub(crate) fn evaluate(
        &self,
        g: usize,
        percentages: &[f64],
        reverse: bool,
        cont: bool,
        legacy_disc: bool,
    ) -> Option<Vec<f64>> {
        let total = self.totals[g];
        if total == 0 || self.maps[g].is_empty() {
            return None;
        }
        let (values, cum) = self.sorted(g, reverse);
        let total_f = total as f64;
        let mut out = Vec::with_capacity(percentages.len());
        for &p in percentages {
            out.push(if cont || legacy_disc {
                // Spark's `getPercentile`.
                let position = (total_f - 1.0) * p;
                let lower = position.floor();
                let higher = position.ceil();
                // `as` saturates like Java narrowing.
                #[expect(clippy::cast_possible_truncation, reason = "Java narrowing parity")]
                let lower_key = values[binary_search_count(&cum, lower as i64 + 1)];
                if higher == lower {
                    lower_key
                } else {
                    #[expect(clippy::cast_possible_truncation, reason = "Java narrowing parity")]
                    let higher_key = values[binary_search_count(&cum, higher as i64 + 1)];
                    if higher_key == lower_key || legacy_disc {
                        lower_key
                    } else {
                        (higher - position) * lower_key + (position - lower) * higher_key
                    }
                }
            } else {
                // Spark's `PercentileDisc.getPercentile`.
                let position = total_f * p;
                let higher = position.ceil();
                // `as` saturates like Java narrowing.
                #[expect(clippy::cast_possible_truncation, reason = "Java narrowing parity")]
                values[binary_search_count(&cum, higher as i64)]
            });
        }
        Some(out)
    }

    /// The partial state of groups `ids`: the values and their counts.
    pub(crate) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        use arrow_array::builder::{Float64Builder, Int64Builder};
        let mut values = Float64Builder::new();
        let mut counts = Int64Builder::new();
        let mut vo = Vec::with_capacity(ids.len() + 1);
        let mut co = Vec::with_capacity(ids.len() + 1);
        vo.push(0);
        co.push(0);
        for &i in ids {
            let g = i as usize;
            let mut pairs: Vec<(u64, i64)> = self.maps[g].iter().map(|(&k, &c)| (k, c)).collect();
            pairs.sort_unstable_by_key(|&(k, _)| k);
            for (k, c) in pairs {
                values.append_value(f64::from_bits(k));
                counts.append_value(c);
            }
            let too_many = || internal("a percentile of too many values".to_owned());
            vo.push(i32::try_from(values.len()).map_err(|_| too_many())?);
            co.push(i32::try_from(counts.len()).map_err(|_| too_many())?);
        }
        Ok(vec![
            finish_lists(Arc::new(values.finish()), vo, values_list_field()),
            finish_lists(Arc::new(counts.finish()), co, counts_list_field()),
        ])
    }

    /// Merges the partial states (`values`, `counts` lists) into `groups`.
    pub(crate) fn merge(
        &mut self,
        groups: &[u32],
        values: &ListArray,
        counts: &ListArray,
    ) -> Result<()> {
        let bad = || internal("a quantile state of other lists than (values, counts)".to_owned());
        let vv = values
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(bad)?;
        let cc = counts
            .values()
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(bad)?;
        if values.len() != groups.len() || counts.len() != groups.len() {
            return Err(bad());
        }
        for (row, &g) in groups.iter().enumerate() {
            if values.is_null(row) || counts.is_null(row) {
                continue;
            }
            let (vo, co) = (values.value_offsets(), counts.value_offsets());
            let (vs, ve) = (
                usize::try_from(vo[row]).map_err(|_| bad())?,
                usize::try_from(vo[row + 1]).map_err(|_| bad())?,
            );
            let (cs, ce) = (
                usize::try_from(co[row]).map_err(|_| bad())?,
                usize::try_from(co[row + 1]).map_err(|_| bad())?,
            );
            if ve - vs != ce - cs {
                return Err(bad());
            }
            for (vi, ci) in (vs..ve).zip(cs..ce) {
                if vv.is_valid(vi) && cc.is_valid(ci) {
                    self.add(g as usize, vv.value(vi), cc.value(ci));
                }
            }
        }
        Ok(())
    }
}

/// Spark's `Arrays.binarySearch(counts, 0, end, value)`: the index of the
/// value, or its insertion point, clamped into range. Cumulative counts
/// are strictly increasing, so a match is unique.
fn binary_search_count(counts: &[i64], value: i64) -> usize {
    match counts.binary_search(&value) {
        Ok(ix) | Err(ix) => ix.min(counts.len().saturating_sub(1)),
    }
}

/// One Greenwald-Khanna statistic: Spark's `QuantileSummaries.Stats`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Stat {
    value: f64,
    g: i64,
    delta: i64,
}

/// The Greenwald-Khanna state of one aggregate: per-group compressed
/// summaries with uncompressed head buffers, as Spark's
/// `QuantileSummaries` holds them.
#[derive(Debug, Default)]
pub(crate) struct GkAcc {
    sampled: Vec<Vec<Stat>>,
    heads: Vec<Vec<f64>>,
    counts: Vec<i64>,
}

impl GkAcc {
    pub(crate) fn resize(&mut self, n: usize) {
        self.sampled.resize(n, Vec::new());
        self.heads.resize(n, Vec::new());
        self.counts.resize(n, 0);
    }

    pub(crate) fn reserve(&mut self, new: usize) {
        self.sampled.reserve(new);
        self.heads.reserve(new);
        self.counts.reserve(new);
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        self.sampled
            .iter()
            .map(|s| s.capacity().saturating_mul(24))
            .sum::<usize>()
            + self
                .heads
                .iter()
                .map(|h| h.capacity().saturating_mul(8))
                .sum::<usize>()
            + self.counts.capacity().saturating_mul(8)
    }

    /// Inserts `x` into group `g` (Spark's `insert`).
    pub(crate) fn insert(&mut self, g: usize, x: f64, relative_error: f64) {
        self.heads[g].push(x);
        if self.heads[g].len() >= HEAD_SIZE {
            let head = std::mem::take(&mut self.heads[g]);
            let sampled = std::mem::take(&mut self.sampled[g]);
            let (sampled, count) =
                with_head_buffer_inserted(head, sampled, self.counts[g], relative_error);
            self.sampled[g] = sampled;
            self.counts[g] = count;
            if self.sampled[g].len() >= COMPRESS_THRESHOLD {
                let sampled = std::mem::take(&mut self.sampled[g]);
                let (sampled, count) = compress(&sampled, self.counts[g], relative_error);
                self.sampled[g] = sampled;
                self.counts[g] = count;
            }
        }
    }

    /// The compressed summary of group `g` (the head buffer drained),
    /// without touching the live state.
    fn compressed(&self, g: usize, relative_error: f64) -> (Vec<Stat>, i64, f64) {
        let (mut sampled, mut count) = (self.sampled[g].clone(), self.counts[g]);
        if !self.heads[g].is_empty() {
            (sampled, count) =
                with_head_buffer_inserted(self.heads[g].clone(), sampled, count, relative_error);
        }
        let (sampled, count) = compress(&sampled, count, relative_error);
        (sampled, count, relative_error)
    }

    /// Merges the compressed `other` summary into group `g` (Spark's
    /// `PercentileDigest.merge`: both sides compressed, the merged
    /// error is the larger one).
    pub(crate) fn merge_summary(
        &mut self,
        g: usize,
        other: (Vec<Stat>, i64, f64),
        relative_error: f64,
    ) {
        let mine = self.compressed(g, relative_error);
        let (sampled, count) = merge_summaries(mine, other);
        self.sampled[g] = sampled;
        self.counts[g] = count;
        self.heads[g].clear();
    }

    /// The approximate percentiles of group `g` (Spark's `query` over
    /// the compressed summary): `None` for no values.
    pub(crate) fn evaluate(
        &self,
        g: usize,
        percentages: &[f64],
        relative_error: f64,
    ) -> Option<Vec<f64>> {
        let (sampled, count, _) = self.compressed(g, relative_error);
        query(&sampled, count, percentages, relative_error)
    }

    /// The partial state of groups `ids`: the compressed summaries in
    /// Spark's `PercentileDigestSerializer` format (big-endian).
    pub(crate) fn state(&self, ids: &[u32], relative_error: f64) -> Vec<ArrayRef> {
        let mut out = Vec::with_capacity(ids.len());
        for &i in ids {
            let (sampled, count, _) = self.compressed(i as usize, relative_error);
            out.push(serialize_summary(&sampled, count, relative_error));
        }
        vec![Arc::new(BinaryArray::from_iter_values(out.iter().map(Vec::as_slice))) as ArrayRef]
    }

    /// Merges the partial-state summaries (a `BinaryArray` in the
    /// serializer format) into `groups`.
    pub(crate) fn merge(
        &mut self,
        groups: &[u32],
        states: &BinaryArray,
        relative_error: f64,
    ) -> Result<()> {
        if states.len() != groups.len() {
            return Err(internal(
                "a quantile summary state of another length than its groups".to_owned(),
            ));
        }
        for (row, &g) in groups.iter().enumerate() {
            if states.is_null(row) {
                continue;
            }
            let other = deserialize_summary(states.value(row))?;
            self.merge_summary(g as usize, other, relative_error);
        }
        Ok(())
    }
}

/// Inserts the sorted head buffer into the samples (Spark's
/// `withHeadBufferInserted`).
fn with_head_buffer_inserted(
    head: Vec<f64>,
    sampled: Vec<Stat>,
    count: i64,
    relative_error: f64,
) -> (Vec<Stat>, i64) {
    if head.is_empty() {
        return (sampled, count);
    }
    let mut sorted = head;
    sorted.sort_by(f64::total_cmp);
    let mut out: Vec<Stat> = Vec::with_capacity(sampled.len() + sorted.len());
    let mut current = count;
    let mut s = 0;
    for (o, &x) in sorted.iter().enumerate() {
        while s < sampled.len() && sampled[s].value <= x {
            out.push(sampled[s]);
            s += 1;
        }
        current += 1;
        let delta = if out.is_empty() || (s == sampled.len() && o == sorted.len() - 1) {
            0
        } else {
            // `as` saturates like Java narrowing; counts are small and finite.
            #[expect(clippy::cast_possible_truncation, reason = "Java narrowing parity")]
            let delta = (2.0 * relative_error * current as f64).floor() as i64;
            delta
        };
        out.push(Stat {
            value: x,
            g: 1,
            delta,
        });
    }
    out.extend_from_slice(&sampled[s..]);
    (out, current)
}

/// Compresses the samples (Spark's `compress`).
fn compress(sampled: &[Stat], count: i64, relative_error: f64) -> (Vec<Stat>, i64) {
    let threshold = 2.0 * relative_error * count as f64;
    (compress_immut(sampled, threshold), count)
}

/// Spark's `compressImmut`.
fn compress_immut(current: &[Stat], merge_threshold: f64) -> Vec<Stat> {
    if current.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<Stat> = Vec::new();
    let mut head = current[current.len() - 1];
    if current.len() > 1 {
        for i in (1..current.len() - 1).rev() {
            let sample = current[i];
            if ((sample.g.wrapping_add(head.g).wrapping_add(head.delta)) as f64) < merge_threshold {
                head.g = head.g.wrapping_add(sample.g);
            } else {
                out.push(head);
                head = sample;
            }
        }
    }
    out.push(head);
    let first = current[0];
    if first.value <= head.value && current.len() > 1 {
        out.push(first);
    }
    out.reverse();
    out
}

/// Merges two compressed summaries (Spark's `merge`).
fn merge_summaries(left: (Vec<Stat>, i64, f64), right: (Vec<Stat>, i64, f64)) -> (Vec<Stat>, i64) {
    let ((a, na, ea), (b, nb, eb)) = (left, right);
    if nb == 0 {
        return (a, na);
    }
    if na == 0 {
        return (b, nb);
    }
    let merged_error = ea.max(eb);
    let merged_count = na.wrapping_add(nb);
    // `as` saturates like Java narrowing; counts are small and finite.
    #[expect(clippy::cast_possible_truncation, reason = "Java narrowing parity")]
    let additional_self = (2.0 * eb * nb as f64).floor() as i64;
    #[expect(clippy::cast_possible_truncation, reason = "Java narrowing parity")]
    let additional_other = (2.0 * ea * na as f64).floor() as i64;
    let mut out: Vec<Stat> = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let (next, additional) = if a[i].value < b[j].value {
            i += 1;
            (a[i - 1], if j > 0 { additional_self } else { 0 })
        } else {
            j += 1;
            (b[j - 1], if i > 0 { additional_other } else { 0 })
        };
        out.push(Stat {
            value: next.value,
            g: next.g,
            delta: next.delta.wrapping_add(additional),
        });
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    let threshold = 2.0 * merged_error * merged_count as f64;
    (compress_immut(&out, threshold), merged_count)
}

/// Runs the query (Spark's `query`): the approximate quantiles in the
/// input order, or `None` for no samples.
fn query(
    sampled: &[Stat],
    count: i64,
    percentages: &[f64],
    relative_error: f64,
) -> Option<Vec<f64>> {
    if sampled.is_empty() {
        return None;
    }
    let mut exact = i64::MIN;
    for s in sampled {
        exact = exact.max(s.delta.wrapping_add(s.g));
    }
    // Spark's integer division, then to double.
    let target_error = (exact / 2) as f64;
    let mut order: Vec<usize> = (0..percentages.len()).collect();
    order.sort_by(|&a, &b| {
        percentages[a]
            .total_cmp(&percentages[b])
            .then_with(|| a.cmp(&b))
    });
    let mut result = vec![0.0; percentages.len()];
    let mut index = 0;
    let mut min_rank = sampled[0].g;
    for &pos in &order {
        let p = percentages[pos];
        if p <= relative_error {
            result[pos] = sampled[0].value;
        } else if p >= 1.0 - relative_error {
            result[pos] = sampled[sampled.len() - 1].value;
        } else {
            let (next, rank, value) =
                find_approx_quantile(sampled, count, index, min_rank, target_error, p);
            index = next;
            min_rank = rank;
            result[pos] = value;
        }
    }
    Some(result)
}

/// Spark's `findApproxQuantile`.
fn find_approx_quantile(
    sampled: &[Stat],
    count: i64,
    index: usize,
    min_rank_at_index: i64,
    target_error: f64,
    percentile: f64,
) -> (usize, i64, f64) {
    let mut cur = sampled[index];
    // `as` saturates like Java narrowing.
    #[expect(clippy::cast_possible_truncation, reason = "Java narrowing parity")]
    let rank = (percentile * count as f64).ceil() as i64;
    let mut i = index;
    let mut min_rank = min_rank_at_index;
    while i < sampled.len() - 1 {
        let max_rank = min_rank.wrapping_add(cur.delta) as f64;
        if max_rank - target_error <= rank as f64 && (rank as f64) <= min_rank as f64 + target_error
        {
            return (i, min_rank, cur.value);
        }
        i += 1;
        cur = sampled[i];
        min_rank = min_rank.wrapping_add(cur.g);
    }
    (sampled.len() - 1, 0, sampled[sampled.len() - 1].value)
}

/// Spark's `PercentileDigestSerializer`: big-endian
/// (`compressThreshold`, `relativeError`, `count`, samples).
fn serialize_summary(sampled: &[Stat], count: i64, relative_error: f64) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 8 + 8 + 4 + sampled.len() * 24);
    out.extend_from_slice(
        &i32::try_from(COMPRESS_THRESHOLD)
            .unwrap_or(i32::MAX)
            .to_be_bytes(),
    );
    out.extend_from_slice(&relative_error.to_be_bytes());
    out.extend_from_slice(&count.to_be_bytes());
    // Summaries are memory-bound, far below `i32::MAX` entries.
    #[expect(clippy::cast_possible_truncation, reason = "summaries fit i32")]
    out.extend_from_slice(&(sampled.len() as i32).to_be_bytes());
    for s in sampled {
        out.extend_from_slice(&s.value.to_be_bytes());
        out.extend_from_slice(&s.g.to_be_bytes());
        out.extend_from_slice(&s.delta.to_be_bytes());
    }
    out
}

/// Reads [`serialize_summary`] back.
fn deserialize_summary(bytes: &[u8]) -> Result<(Vec<Stat>, i64, f64)> {
    let bad = || internal("a corrupt quantile summary state".to_owned());
    if bytes.len() < 24 {
        return Err(bad());
    }
    let i32_at = |o: usize| -> Result<i32> {
        Ok(i32::from_be_bytes(
            bytes[o..o + 4].try_into().map_err(|_| bad())?,
        ))
    };
    let i64_at = |o: usize| -> Result<i64> {
        Ok(i64::from_be_bytes(
            bytes[o..o + 8].try_into().map_err(|_| bad())?,
        ))
    };
    let f64_at = |o: usize| -> Result<f64> {
        Ok(f64::from_be_bytes(
            bytes[o..o + 8].try_into().map_err(|_| bad())?,
        ))
    };
    let _threshold = i32_at(0)?;
    let error = f64_at(4)?;
    let count = i64_at(12)?;
    let len = i32_at(20)?;
    let len = usize::try_from(len).map_err(|_| bad())?;
    if bytes.len() != 24 + len * 24 {
        return Err(bad());
    }
    let mut sampled = Vec::with_capacity(len);
    for i in 0..len {
        let o = 24 + i * 24;
        sampled.push(Stat {
            value: f64_at(o)?,
            g: i64_at(o + 8)?,
            delta: i64_at(o + 16)?,
        });
    }
    Ok((sampled, count, error))
}

/// Reads one quantile value as a `double`, as Spark's `toDoubleValue`
/// does for the approximate path (dates and timestamps read as their
/// numeric forms).
pub(crate) fn double_of(a: &ArrayRef, i: usize) -> Result<Option<f64>> {
    if a.is_null(i) {
        return Ok(None);
    }
    Ok(Some(match a.data_type() {
        DataType::Int8 => f64::from(a.as_primitive::<Int8Type>().value(i)),
        DataType::Int16 => f64::from(a.as_primitive::<Int16Type>().value(i)),
        DataType::Int32 => f64::from(a.as_primitive::<Int32Type>().value(i)),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i) as f64,
        DataType::Float32 => f64::from(a.as_primitive::<Float32Type>().value(i)),
        DataType::Float64 => a.as_primitive::<Float64Type>().value(i),
        DataType::Decimal64(_, s) => {
            decimal_to_double(i128::from(a.as_primitive::<Decimal64Type>().value(i)), *s)
        }
        DataType::Decimal128(_, s) => {
            decimal_to_double(a.as_primitive::<Decimal128Type>().value(i), *s)
        }
        DataType::Date32 => f64::from(a.as_primitive::<Date32Type>().value(i)),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            a.as_primitive::<TimestampMicrosecondType>().value(i) as f64
        }
        t => return Err(internal(format!("a quantile value of type {t}"))),
    }))
}

/// The unscaled decimal `unscaled` at `scale` as the nearest `f64`, as
/// Java's `BigDecimal.doubleValue` gives it (correctly rounded, through
/// the decimal text).
pub(crate) fn decimal_to_double(unscaled: i128, scale: i8) -> f64 {
    let digits = unscaled.unsigned_abs().to_string();
    let scale = usize::try_from(scale).unwrap_or(0);
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (int, frac) = padded.split_at(padded.len() - scale);
    format!("{}{int}.{frac}", if unscaled < 0 { "-" } else { "" })
        .parse::<f64>()
        .unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;

    use super::*;
    use crate::aggregate::{Accumulator, AggInput, AggregateExpr, AggregateFunction};

    fn expr(function: AggregateFunction, t: DataType) -> AggregateExpr {
        let args = match &function {
            AggregateFunction::Percentile { .. } => vec![0, 1],
            _ => vec![0],
        };
        AggregateExpr {
            function,
            args,
            input: AggInput::Raw,
            arg_type: t,
            name: "q".to_owned(),
        }
    }

    fn acc(function: AggregateFunction, t: DataType) -> Accumulator {
        let mut a = Accumulator::new(&expr(function, t), true).unwrap();
        a.resize(1);
        a
    }

    fn ints(v: &[Option<i64>]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec())) as ArrayRef
    }

    fn decimals(v: &[Option<i128>]) -> ArrayRef {
        use arrow_array::Decimal128Array;
        Arc::new(Decimal128Array::from(v.to_vec()).with_data_type(DataType::Decimal128(10, 2)))
            as ArrayRef
    }

    fn ones(n: usize) -> ArrayRef {
        Arc::new(Int64Array::from(vec![1; n])) as ArrayRef
    }

    /// The final values of group 0 after folding `values` through `parts`
    /// partial accumulators and merging their states in order.
    fn run(
        function: &AggregateFunction,
        t: &DataType,
        values: &[Option<i64>],
        parts: usize,
    ) -> Vec<Option<f64>> {
        let n = values.len();
        let mut top = acc(function.clone(), t.clone());
        for chunk in values.chunks(n.div_ceil(parts).max(1)) {
            let mut a = acc(function.clone(), t.clone());
            let cols = if matches!(function, AggregateFunction::Percentile { .. }) {
                vec![ints(chunk), ones(chunk.len())]
            } else {
                vec![ints(chunk)]
            };
            a.update(
                &vec![0; chunk.len()],
                &cols,
                &krishiv_types::batch::Weights::Const(1),
            )
            .unwrap();
            let states = a.state(&[0]).unwrap();
            top.merge(&[0], &states).unwrap();
        }
        let out = top.evaluate(&[0]).unwrap();
        let out = out.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..out.len())
            .map(|i| out.is_valid(i).then(|| out.value(i)))
            .collect()
    }

    fn exact(pcts: &[f64], reverse: bool) -> AggregateFunction {
        AggregateFunction::Percentile {
            percentages: pcts.to_vec(),
            array: false,
            reverse,
        }
    }

    /// Exact results do not depend on the partitioning; `None` is no rows.
    #[test]
    fn exact_percentiles_merge_identically() {
        let values: Vec<Option<i64>> = (0..100)
            .map(|i| if i % 10 == 0 { None } else { Some(i % 25) })
            .collect();
        for pcts in [vec![0.0], vec![0.25, 0.5, 0.75], vec![1.0]] {
            for reverse in [false, true] {
                assert_eq!(
                    run(&exact(&pcts, reverse), &DataType::Int64, &values, 1),
                    run(&exact(&pcts, reverse), &DataType::Int64, &values, 8),
                    "{pcts:?} reverse={reverse}"
                );
            }
        }
        // The 0.5 percentile of 1..=4 is 2.5; descending mirrors it.
        let four: Vec<Option<i64>> = (1..=4).map(Some).collect();
        assert_eq!(
            run(&exact(&[0.5], false), &DataType::Int64, &four, 8),
            vec![Some(2.5)]
        );
        assert_eq!(
            run(&exact(&[0.5], true), &DataType::Int64, &four, 8),
            vec![Some(2.5)]
        );
        // No rows, and only NULLs, are NULL.
        assert_eq!(
            run(&exact(&[0.5], false), &DataType::Int64, &[], 1),
            vec![None]
        );
        assert_eq!(
            run(&exact(&[0.5], false), &DataType::Int64, &[None, None], 8),
            vec![None]
        );
        // Decimals read as doubles.
        let dec = DataType::Decimal128(10, 2);
        let mut top = acc(exact(&[0.5], false), dec);
        let cols = vec![decimals(&[Some(150), Some(250)]), ones(2)];
        top.update(&[0, 0], &cols, &krishiv_types::batch::Weights::Const(1))
            .unwrap();
        let out = top.evaluate(&[0]).unwrap();
        let out = out.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(out.value(0), 2.0);
    }

    /// `percentile_disc` takes the value at `ceil(total * p)` (or the
    /// continuous lower bound with the legacy flag), however partitioned.
    #[test]
    fn discrete_percentiles_merge_identically() {
        let values: Vec<Option<i64>> = (0..50).map(|i| Some(i % 7)).collect();
        for (legacy, reverse) in [(false, false), (false, true), (true, false)] {
            let f = AggregateFunction::PercentileDisc {
                percentages: vec![0.1, 0.5, 0.9],
                array: false,
                reverse,
                legacy,
            };
            assert_eq!(
                run(&f, &DataType::Int64, &values, 1),
                run(&f, &DataType::Int64, &values, 8),
            );
        }
        let four: Vec<Option<i64>> = (1..=4).map(Some).collect();
        let disc = AggregateFunction::PercentileDisc {
            percentages: vec![0.5],
            array: false,
            reverse: false,
            legacy: false,
        };
        assert_eq!(run(&disc, &DataType::Int64, &four, 3), vec![Some(2.0)]);
        let legacy = AggregateFunction::PercentileDisc {
            percentages: vec![0.5],
            array: false,
            reverse: false,
            legacy: true,
        };
        assert_eq!(run(&legacy, &DataType::Int64, &four, 3), vec![Some(2.0)]);
    }

    /// Approximate results stay within Spark's documented relative error of
    /// the exact value, however partitioned.
    #[test]
    fn approximate_percentiles_stay_within_error() {
        let ints: Vec<Option<i64>> = (1..=1000).map(Some).collect();
        let values: Vec<Option<f64>> = ints.iter().map(|v| v.map(|x| x as f64)).collect();
        let floats: ArrayRef = Arc::new(Float64Array::from(values)) as ArrayRef;
        let runf = |pcts: &[f64], parts: usize| {
            let f = AggregateFunction::PercentileApprox {
                percentages: pcts.to_vec(),
                array: false,
                accuracy: 100,
            };
            let n = floats.len();
            let mut top = acc(f.clone(), DataType::Float64);
            for chunk in (0..n).collect::<Vec<_>>().chunks(n.div_ceil(parts).max(1)) {
                let mut a = acc(f.clone(), DataType::Float64);
                let col = floats.slice(chunk[0], chunk.len());
                a.update(
                    &vec![0; chunk.len()],
                    &[col],
                    &krishiv_types::batch::Weights::Const(1),
                )
                .unwrap();
                let states = a.state(&[0]).unwrap();
                top.merge(&[0], &states).unwrap();
            }
            let out = top.evaluate(&[0]).unwrap();
            let out = out.as_any().downcast_ref::<Float64Array>().unwrap();
            out.value(0)
        };
        let rel = 1.0 / 100.0;
        for pct in [0.5, 0.9, 0.99] {
            let bound = 2.0 * rel * 1000.0;
            for parts in [1, 8] {
                let got = runf(&[pct], parts);
                let want = 1.0 + pct * 999.0;
                assert!(
                    (got - want).abs() <= bound,
                    "{pct} x{parts}: {got} vs {want}"
                );
            }
        }
        let empty = acc(
            AggregateFunction::PercentileApprox {
                percentages: vec![0.5],
                array: false,
                accuracy: 100,
            },
            DataType::Float64,
        );
        assert!(empty.evaluate(&[0]).unwrap().is_null(0));
    }
}
