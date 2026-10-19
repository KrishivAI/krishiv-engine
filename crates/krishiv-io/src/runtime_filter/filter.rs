// SPDX-License-Identifier: Apache-2.0
//! [`KeyFilter`]: what a producer learned about its keys (exact min/max, a
//! small value set, a Bloom filter, whether NULL can match), built by
//! [`KeyFilterBuilder`] partials that merge, and evaluated on rows or
//! lowered to statistics predicates.

use std::sync::Arc;

use arrow_array::{Array, BooleanArray};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType;

use super::FilterId;
use super::bloom::{RuntimeBloom, xxh64};
use super::key::{
    KeyRange, KeyValue, Norm, NormalizedKeys, float_bits, float_hash, for_each_bytes,
    normalize_key_bytes, supported_key_type, value_scalar, widens_to,
};
use crate::request::{CmpOp, ColumnPath, PushdownExpr, ScalarValue};
use crate::schema_map::{Retyped, retype};

/// What a producer collects (the cost model's decisions and caps).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyFilterOptions {
    /// Distinct keys kept as a value set (`krishiv.runtimeFilter.inList.maxValues`).
    pub max_values: usize,
    /// Collect hashes for a Bloom filter.
    pub bloom: bool,
    /// Most keys a Bloom filter is built for (`krishiv.runtimeFilter.bloom.maxItems`).
    pub bloom_max_items: u64,
    /// Largest Bloom filter (`krishiv.runtimeFilter.bloom.maxBytes`).
    pub bloom_max_bytes: usize,
    /// The false-positive rate a Bloom filter is sized for.
    pub bloom_fpp: f64,
    /// A Bloom filter expected to pass more false positives is not built.
    pub bloom_max_fpp: f64,
}

impl Default for KeyFilterOptions {
    fn default() -> Self {
        Self {
            max_values: 50,
            bloom: true,
            bloom_max_items: 4_000_000,
            bloom_max_bytes: 8 << 20,
            bloom_fpp: 0.01,
            bloom_max_fpp: 0.05,
        }
    }
}

/// One producer's partial knowledge of its keys (one per build driver);
/// partials [merge](Self::merge) and [finish](Self::finish) into a
/// [`KeyFilter`].
#[derive(Debug, Clone)]
pub struct KeyFilterBuilder {
    key_type: DataType,
    options: KeyFilterOptions,
    range: Option<KeyRange>,
    /// A NaN key was seen: no range.
    nan: bool,
    /// `None` once more than `max_values` distinct keys were seen.
    /// Sorted.
    values: Option<Vec<KeyValue>>,
    /// `None` when no Bloom is collected or too many keys were.
    hashes: Option<Vec<u64>>,
    nulls: bool,
    rows: u64,
}

impl KeyFilterBuilder {
    /// A builder for keys of `key_type`, or `None` when that type has no
    /// runtime filter.
    #[must_use]
    pub fn new(key_type: DataType, options: KeyFilterOptions) -> Option<Self> {
        supported_key_type(&key_type).then(|| Self {
            hashes: options.bloom.then(Vec::new),
            values: Some(Vec::new()),
            key_type,
            options,
            range: None,
            nan: false,
            nulls: false,
            rows: 0,
        })
    }

    /// Adds the keys of `column` (of the key type) in the rows `include`
    /// marks valid (every row when `None`): the rows the producer keeps. A
    /// NULL key in such a row (a null-safe key) lets NULLs pass.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if `column` is not of the key type.
    pub fn add(&mut self, column: &dyn Array, include: Option<&NullBuffer>) -> crate::Result<()> {
        if !column.data_type().equals_datatype(&self.key_type) {
            return Err(crate::error::internal(format!(
                "runtime filter keys of type {} added to a {} filter",
                column.data_type(),
                self.key_type
            )));
        }
        let keys = normalize_key_bytes(column, &self.key_type)
            .ok_or_else(|| crate::error::internal("unsupported runtime filter key".to_owned()))?;
        let kept = |i: usize| include.is_none_or(|v| v.is_valid(i));
        let mut rows = Vec::with_capacity(keys.len());
        for i in (0..keys.len()).filter(|&i| kept(i)) {
            if keys.is_null(i) {
                self.nulls = true;
            } else {
                rows.push(i);
            }
        }
        if rows.is_empty() {
            return Ok(());
        }
        self.rows += rows.len() as u64;
        self.add_range(&keys, &rows);
        self.add_values(&keys, &rows);
        if let Some(h) = &mut self.hashes {
            if self.rows > self.options.bloom_max_items {
                self.hashes = None;
            } else {
                let all = keys.hashes();
                h.extend(rows.iter().map(|&i| all[i]));
            }
        }
        Ok(())
    }

    fn add_range(&mut self, keys: &NormalizedKeys, rows: &[usize]) {
        let r = match &keys.norm {
            Norm::Int { values, .. } => {
                let it = rows.iter().map(|&i| values[i]);
                let (lo, hi) = it.fold((i64::MAX, i64::MIN), |(l, h), v| (l.min(v), h.max(v)));
                KeyRange::Int(lo, hi)
            }
            Norm::Decimal(values) => {
                let it = rows.iter().map(|&i| values[i]);
                let (lo, hi) = it.fold((i128::MAX, i128::MIN), |(l, h), v| (l.min(v), h.max(v)));
                KeyRange::Decimal(lo, hi)
            }
            Norm::Float { values, .. } => {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for &i in rows {
                    let v = values[i];
                    if v.is_nan() {
                        self.nan = true;
                    }
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                KeyRange::Float(lo, hi)
            }
            Norm::Bytes(a) => {
                let mut lo: Option<Box<[u8]>> = None;
                let mut hi: Option<Box<[u8]>> = None;
                let mut want = vec![false; a.len()];
                for &i in rows {
                    want[i] = true;
                }
                for_each_bytes(a.as_ref(), |i, b| {
                    if !want[i] {
                        return;
                    }
                    if lo.as_deref().is_none_or(|l| b < l) {
                        lo = Some(Box::from(b));
                    }
                    if hi.as_deref().is_none_or(|h| b > h) {
                        hi = Some(Box::from(b));
                    }
                });
                match (lo, hi) {
                    (Some(l), Some(h)) => KeyRange::Bytes(l, h),
                    _ => return,
                }
            }
        };
        self.range = match self.range.take() {
            None => Some(r),
            Some(old) => old.union(&r),
        };
    }

    fn add_values(&mut self, keys: &NormalizedKeys, rows: &[usize]) {
        let Some(set) = &mut self.values else {
            return;
        };
        let max = self.options.max_values;
        // A sorted vector of at most `max + 1` members: a binary search per
        // key, an insertion only for a new one.
        let mut push = |v: KeyValue| {
            if let Err(at) = set.binary_search(&v) {
                set.insert(at, v);
            }
            set.len() <= max
        };
        let ok = match &keys.norm {
            Norm::Int { values, .. } => rows
                .iter()
                .all(|&i| push(KeyValue::Fixed(i128::from(values[i])))),
            Norm::Decimal(values) => rows.iter().all(|&i| push(KeyValue::Fixed(values[i]))),
            Norm::Float { values, single } => rows
                .iter()
                .all(|&i| push(KeyValue::Fixed(float_bits(values[i], *single)))),
            Norm::Bytes(a) => {
                let mut want = vec![false; a.len()];
                for &i in rows {
                    want[i] = true;
                }
                let mut ok = true;
                for_each_bytes(a.as_ref(), |i, b| {
                    if ok && want[i] {
                        let found = set.binary_search_by(|v| cmp_bytes(v, b));
                        if let Err(at) = found {
                            set.insert(at, KeyValue::Bytes(Box::from(b)));
                            ok = set.len() <= max;
                        }
                    }
                });
                ok
            }
        };
        if !ok {
            self.values = None;
        }
    }

    /// Heap bytes held (what a producer reserves for this partial).
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let values = self.values.as_ref().map_or(0, |s| {
            s.iter()
                .map(|v| match v {
                    KeyValue::Fixed(_) => 48,
                    KeyValue::Bytes(b) => 48 + b.len(),
                })
                .sum()
        });
        let hashes = self.hashes.as_ref().map_or(0, |h| h.capacity() * 8);
        let range = match &self.range {
            Some(KeyRange::Bytes(a, b)) => a.len() + b.len(),
            _ => 0,
        };
        values + hashes + range
    }

    /// Stops collecting hashes (a failed reservation): no Bloom is built.
    pub fn drop_bloom(&mut self) {
        self.hashes = None;
    }

    /// Merges another partial of the same filter.
    pub fn merge(&mut self, other: Self) {
        self.range = match (self.range.take(), other.range) {
            (Some(a), Some(b)) => a.union(&b),
            (a, b) => a.or(b),
        };
        self.nan |= other.nan;
        self.nulls |= other.nulls;
        self.rows += other.rows;
        self.values = match (self.values.take(), other.values) {
            (Some(mut a), Some(b)) => {
                a.extend(b);
                a.sort_unstable();
                a.dedup();
                (a.len() <= self.options.max_values).then_some(a)
            }
            _ => None,
        };
        self.hashes = match (self.hashes.take(), other.hashes) {
            (Some(mut a), Some(b)) if self.rows <= self.options.bloom_max_items => {
                a.extend(b);
                Some(a)
            }
            _ => None,
        };
    }

    /// The filter: a value set when the keys are few, else a Bloom filter
    /// sized exactly for the keys collected (unless over budget or too
    /// inaccurate), and the range unless a key was NaN.
    #[must_use]
    pub fn finish(self) -> KeyFilter {
        let values = self.values;
        let bloom = match (&values, self.hashes) {
            (None, Some(hashes)) if !hashes.is_empty() => {
                let items = hashes.len() as u64;
                let bytes = RuntimeBloom::bytes_for(items, self.options.bloom_fpp)
                    .min(self.options.bloom_max_bytes);
                let mut bloom = RuntimeBloom::with_bytes(bytes);
                for h in hashes {
                    bloom.insert_hash(h);
                }
                (bloom.expected_fpp(items) <= self.options.bloom_max_fpp).then_some(bloom)
            }
            _ => None,
        };
        KeyFilter {
            range: self.range.filter(|_| !self.nan),
            values,
            bloom,
            nulls_pass: self.nulls,
            empty: self.rows == 0,
            key_type: self.key_type,
        }
    }
}

/// A member's order relative to bytes `b` (fixed members sort first).
fn cmp_bytes(v: &KeyValue, b: &[u8]) -> std::cmp::Ordering {
    match v {
        KeyValue::Bytes(x) => x.as_ref().cmp(b),
        KeyValue::Fixed(_) => std::cmp::Ordering::Less,
    }
}

/// What a producer knows about its keys (see the module docs). Immutable
/// once published.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyFilter {
    key_type: DataType,
    range: Option<KeyRange>,
    /// Sorted, at most `max_values`.
    values: Option<Vec<KeyValue>>,
    bloom: Option<RuntimeBloom>,
    nulls_pass: bool,
    empty: bool,
}

impl KeyFilter {
    /// The key type values are compared in.
    #[must_use]
    pub fn key_type(&self) -> &DataType {
        &self.key_type
    }

    /// No non-null key can match: every row whose key is not NULL can be
    /// dropped (and every row when [`Self::nulls_pass`] is false).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.empty
    }

    /// A NULL key can match (a null-safe key with a NULL build key).
    #[must_use]
    pub fn nulls_pass(&self) -> bool {
        self.nulls_pass
    }

    /// Whether the filter has a min/max range.
    #[must_use]
    pub fn has_range(&self) -> bool {
        self.range.is_some()
    }

    /// Distinct keys held as a value set, if any.
    #[must_use]
    pub fn value_count(&self) -> Option<usize> {
        self.values.as_ref().map(Vec::len)
    }

    /// The Bloom filter's size in bytes, if any.
    #[must_use]
    pub fn bloom_bytes(&self) -> Option<usize> {
        self.bloom.as_ref().map(RuntimeBloom::num_bytes)
    }

    /// Heap bytes held (what the producer reserves for the published
    /// filter).
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let values = self.values.as_ref().map_or(0, |v| {
            v.iter()
                .map(|x| match x {
                    KeyValue::Fixed(_) => 32,
                    KeyValue::Bytes(b) => 32 + b.len(),
                })
                .sum()
        });
        values + self.bloom_bytes().unwrap_or(0)
    }

    /// The same filter with only its min/max range (a failed reservation
    /// keeps the form that costs nothing).
    #[must_use]
    pub fn range_only(&self) -> Self {
        Self {
            values: None,
            bloom: None,
            ..self.clone()
        }
    }

    /// The range as literals of the key type.
    #[must_use]
    pub fn range_scalars(&self) -> Option<(ScalarValue, ScalarValue)> {
        self.range.as_ref()?.scalars(&self.key_type)
    }

    /// The value set as literals of the key type.
    #[must_use]
    pub fn value_scalars(&self) -> Option<Vec<ScalarValue>> {
        self.values
            .as_ref()?
            .iter()
            .map(|v| value_scalar(v, &self.key_type))
            .collect()
    }

    /// Whether each row of `column` may match: `false` only for rows the
    /// producer would drop. `None` when the column's type does not widen to
    /// the key type (the filter does not apply).
    #[must_use]
    pub fn evaluate(&self, column: &dyn Array) -> Option<BooleanArray> {
        let keys = normalize_key_bytes(column, &self.key_type)?;
        Some(BooleanArray::new(self.evaluate_keys(&keys), None))
    }

    /// As [`Self::evaluate`], over keys already normalized.
    #[must_use]
    pub fn evaluate_keys(&self, keys: &NormalizedKeys) -> BooleanBuffer {
        let n = keys.len();
        if self.empty {
            return BooleanBuffer::collect_bool(n, |i| self.nulls_pass && keys.is_null(i));
        }
        let nulls = self.nulls_pass;
        match &keys.norm {
            Norm::Int { values, width } => {
                let (lo, hi) = match &self.range {
                    Some(KeyRange::Int(l, h)) => (*l, *h),
                    _ => (i64::MIN, i64::MAX),
                };
                let w = *width;
                BooleanBuffer::collect_bool(n, |i| {
                    if keys.is_null(i) {
                        return nulls;
                    }
                    let v = values[i];
                    v >= lo
                        && v <= hi
                        && self.member(
                            || KeyValue::Fixed(i128::from(v)),
                            || xxh64(&v.to_le_bytes()[..w]),
                        )
                })
            }
            Norm::Decimal(values) => {
                let (lo, hi) = match &self.range {
                    Some(KeyRange::Decimal(l, h)) => (*l, *h),
                    _ => (i128::MIN, i128::MAX),
                };
                BooleanBuffer::collect_bool(n, |i| {
                    if keys.is_null(i) {
                        return nulls;
                    }
                    let v = values[i];
                    v >= lo
                        && v <= hi
                        && self.member(|| KeyValue::Fixed(v), || xxh64(&v.to_le_bytes()))
                })
            }
            Norm::Float { values, single } => {
                let range = match &self.range {
                    Some(KeyRange::Float(l, h)) => Some((*l, *h)),
                    _ => None,
                };
                BooleanBuffer::collect_bool(n, |i| {
                    if keys.is_null(i) {
                        return nulls;
                    }
                    let v = values[i];
                    // The range holds no NaN; a NaN key only matches a NaN
                    // build key, and a range exists only without one.
                    range.is_none_or(|(l, h)| v >= l && v <= h)
                        && self.member(
                            || KeyValue::Fixed(float_bits(v, *single)),
                            || float_hash(v, *single),
                        )
                })
            }
            Norm::Bytes(a) => {
                let range = match &self.range {
                    Some(KeyRange::Bytes(l, h)) => Some((l.as_ref(), h.as_ref())),
                    _ => None,
                };
                let mut out = vec![false; n];
                for_each_bytes(a.as_ref(), |i, b| {
                    out[i] = if keys.is_null(i) {
                        nulls
                    } else {
                        range.is_none_or(|(l, h)| b >= l && b <= h) && self.member_bytes(b)
                    };
                });
                BooleanBuffer::from(out)
            }
        }
    }

    #[inline]
    fn member(&self, value: impl FnOnce() -> KeyValue, hash: impl FnOnce() -> u64) -> bool {
        if let Some(values) = &self.values {
            return values.binary_search(&value()).is_ok();
        }
        self.bloom.as_ref().is_none_or(|b| b.check_hash(hash()))
    }

    fn member_bytes(&self, b: &[u8]) -> bool {
        if let Some(values) = &self.values {
            return values.binary_search_by(|v| cmp_bytes(v, b)).is_ok();
        }
        self.bloom.as_ref().is_none_or(|f| f.check_hash(xxh64(b)))
    }

    /// The filter as a predicate on `column` of type `column_type` (both at
    /// the level of the caller: table or file): the range and value set as
    /// comparisons statistics can prune with, re-typed to the column, and
    /// a [`PushdownExpr::Membership`] for stages that evaluate values.
    /// `None` when the column does not widen to the key type.
    #[must_use]
    pub fn lower(
        self: &Arc<Self>,
        column: &ColumnPath,
        column_type: &DataType,
        id: FilterId,
        generation: u64,
    ) -> Option<PushdownExpr> {
        if !widens_to(column_type, &self.key_type) {
            return None;
        }
        let null_or = |e: PushdownExpr| {
            if self.nulls_pass {
                PushdownExpr::Or(vec![e, PushdownExpr::IsNull(column.clone())])
            } else {
                e
            }
        };
        if self.empty {
            return Some(if self.nulls_pass {
                PushdownExpr::IsNull(column.clone())
            } else {
                PushdownExpr::Constant(false)
            });
        }
        let mut conj = Vec::new();
        if let Some((lo, hi)) = self.range_scalars() {
            for (v, op) in [(lo, CmpOp::GtEq), (hi, CmpOp::LtEq)] {
                match retype(&v, &self.key_type, column_type, op) {
                    Some(Retyped::Exact(value)) => {
                        conj.push(PushdownExpr::cmp(column.clone(), op, value));
                    }
                    Some(Retyped::Never) => conj.push(PushdownExpr::Constant(false)),
                    Some(Retyped::Always) | None => {}
                }
            }
        }
        if let Some(values) = self.value_scalars() {
            let mut kept = Vec::with_capacity(values.len());
            let mut usable = true;
            for v in &values {
                match retype(v, &self.key_type, column_type, CmpOp::Eq) {
                    Some(Retyped::Exact(x)) => kept.push(x),
                    Some(Retyped::Never) => {}
                    Some(Retyped::Always) | None => usable = false,
                }
            }
            if usable {
                conj.push(if kept.is_empty() {
                    PushdownExpr::Constant(false)
                } else {
                    PushdownExpr::In {
                        column: column.clone(),
                        values: kept,
                        collation: crate::request::Collation::Utf8Binary,
                    }
                });
            }
        }
        conj.push(PushdownExpr::Membership {
            column: column.clone(),
            filter: crate::request::MembershipFilter {
                id,
                generation,
                keys: Arc::clone(self),
            },
        });
        Some(null_or(PushdownExpr::And(conj)))
    }
}
