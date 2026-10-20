// SPDX-License-Identifier: Apache-2.0
//! [`KeyFilter`]: what a producer learned about its keys (exact min/max,
//! the distinct keys when they are few, a Bloom filter, whether NULL can
//! match), built by [`KeyFilterBuilder`] partials that merge, and evaluated
//! on rows (in the [form](FilterForm) its build size picks) or lowered to
//! statistics predicates.

use std::sync::Arc;

use arrow_array::{Array, BooleanArray};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType;
use hashbrown::HashTable;

use super::FilterId;
use super::bloom::RuntimeBloom;
use super::hash::{RowHash, row_hashes};
use super::key::{
    KeyRange, KeyValue, Norm, NormalizedKeys, float_bits, for_each_bytes, normalize_key_bytes,
    supported_key_type, value_scalar, widens_to,
};
use super::member::{Class, EXACT_SET_MAX, FilterForm, Member, evaluate};
use crate::request::{CmpOp, ColumnPath, PushdownExpr, ScalarValue};
use crate::schema_map::{Retyped, retype};

/// What a producer collects (the cost model's decisions and caps).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyFilterOptions {
    /// Distinct keys lowered to an `IN` list for statistics
    /// (`krishiv.runtimeFilter.inList.maxValues`).
    pub max_values: usize,
    /// Most distinct keys tested on rows as an exact set
    /// (`krishiv.runtimeFilter.exactSet.maxValues`); more get a Bloom
    /// filter (when collected) or the range only.
    pub exact_max_values: usize,
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
            exact_max_values: EXACT_SET_MAX,
            bloom: true,
            bloom_max_items: 4_000_000,
            bloom_max_bytes: 8 << 20,
            bloom_fpp: 0.01,
            bloom_max_fpp: 0.05,
        }
    }
}

/// Distinct keys with their row hashes, while few.
#[derive(Clone, Default)]
struct Distinct {
    table: HashTable<(u64, KeyValue)>,
    /// Bytes of byte-string members.
    bytes: usize,
}

impl std::fmt::Debug for Distinct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Distinct")
            .field("len", &self.table.len())
            .finish()
    }
}

impl Distinct {
    fn heap_bytes(&self) -> usize {
        self.table.capacity() * (size_of::<(u64, KeyValue)>() + 1) + self.bytes
    }

    /// Adds `v` (hash `h`); `false` once more than `max` distinct keys.
    fn insert(&mut self, h: u64, v: KeyValue, max: usize) -> bool {
        if self.table.find(h, |(_, x)| *x == v).is_some() {
            return true;
        }
        if self.table.len() >= max {
            return false;
        }
        if let KeyValue::Bytes(b) = &v {
            self.bytes += b.len();
        }
        self.table.insert_unique(h, (h, v), |e| e.0);
        true
    }

    /// Adds the bytes `b` (hash `h`), copying them only when new.
    fn insert_bytes(&mut self, h: u64, b: &[u8], max: usize) -> bool {
        let found = self.table.find(
            h,
            |(_, x)| matches!(x, KeyValue::Bytes(y) if y.as_ref() == b),
        );
        found.is_some() || self.insert(h, KeyValue::Bytes(Box::from(b)), max)
    }
}

/// One producer's partial knowledge of its keys (one per build driver);
/// partials [merge](Self::merge) and [finish](Self::finish) into a
/// [`KeyFilter`].
#[derive(Debug, Clone)]
pub struct KeyFilterBuilder {
    key_type: DataType,
    class: Class,
    options: KeyFilterOptions,
    range: Option<KeyRange>,
    /// A NaN key was seen: no range.
    nan: bool,
    /// `None` once more distinct keys were seen than any value form holds.
    distinct: Option<Distinct>,
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
        if !supported_key_type(&key_type) {
            return None;
        }
        Some(Self {
            class: Class::of(&key_type)?,
            hashes: options.bloom.then(Vec::new),
            distinct: Some(Distinct::default()),
            key_type,
            options,
            range: None,
            nan: false,
            nulls: false,
            rows: 0,
        })
    }

    /// Most distinct keys a value form holds.
    fn distinct_max(&self) -> usize {
        self.options.max_values.max(self.options.exact_max_values)
    }

    /// Adds the keys of `column` (of the key type) in the rows `include`
    /// marks valid (every row when `None`): the rows the producer keeps. A
    /// NULL key in such a row (a null-safe key) lets NULLs pass.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if `column` is not of the key type.
    pub fn add(&mut self, column: &dyn Array, include: Option<&NullBuffer>) -> crate::Result<()> {
        self.add_hashed(column, include, None)
    }

    /// As [`Self::add`], with the row hash `h` of every row of `column`
    /// already computed under `hash` (a join that stores `h` for its own
    /// table passes it instead of the keys being hashed twice). Hashes of
    /// another function, or of the wrong length, are not used: the
    /// builder hashes the keys itself.
    ///
    /// # Errors
    /// As [`Self::add`].
    pub fn add_hashed(
        &mut self,
        column: &dyn Array,
        include: Option<&NullBuffer>,
        hashes: Option<(RowHash, &[u64])>,
    ) -> crate::Result<()> {
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
            } else if let Ok(i) = u32::try_from(i) {
                rows.push(i);
            }
        }
        if rows.is_empty() {
            return Ok(());
        }
        self.rows += rows.len() as u64;
        self.add_range(&keys, &rows);
        if self.hashes.is_some() && self.rows > self.options.bloom_max_items {
            self.hashes = None;
        }
        if self.distinct.is_none() && self.hashes.is_none() {
            return Ok(());
        }
        // One row hash per kept row, for the distinct set and the Bloom.
        let h: Vec<u64> = match hashes {
            Some((RowHash::Xxh64Seed0, given)) if given.len() == keys.len() => rows
                .iter()
                .map(|&i| given.get(i as usize).copied().unwrap_or(0))
                .collect(),
            _ => {
                let mut h = Vec::with_capacity(rows.len());
                row_hashes(&keys, Some(&rows), &mut h);
                h
            }
        };
        self.add_values(&keys, &rows, &h);
        if let Some(all) = &mut self.hashes {
            all.extend_from_slice(&h);
        }
        Ok(())
    }

    fn add_range(&mut self, keys: &NormalizedKeys, rows: &[u32]) {
        let at = |i: u32| i as usize;
        let r = match &keys.norm {
            Norm::Int { values, .. } => {
                let it = rows.iter().filter_map(|&i| values.get(at(i)).copied());
                let (lo, hi) = it.fold((i64::MAX, i64::MIN), |(l, h), v| (l.min(v), h.max(v)));
                KeyRange::Int(lo, hi)
            }
            Norm::Decimal(values) => {
                let it = rows.iter().filter_map(|&i| values.get(at(i)).copied());
                let (lo, hi) = it.fold((i128::MAX, i128::MIN), |(l, h), v| (l.min(v), h.max(v)));
                KeyRange::Decimal(lo, hi)
            }
            Norm::Float { values, .. } => {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for v in rows.iter().filter_map(|&i| values.get(at(i)).copied()) {
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
                let want = wanted(a.len(), rows);
                for_each_bytes(a.as_ref(), |i, b| {
                    if !want.get(i).copied().unwrap_or(false) {
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

    /// Adds the kept `rows` (with their hashes `h`) to the distinct set.
    fn add_values(&mut self, keys: &NormalizedKeys, rows: &[u32], h: &[u64]) {
        let max = self.distinct_max();
        let Some(set) = &mut self.distinct else {
            return;
        };
        let pairs = rows.iter().zip(h);
        let ok = match &keys.norm {
            Norm::Int { values, .. } => pairs.into_iter().all(|(&i, &h)| {
                values
                    .get(i as usize)
                    .is_none_or(|&v| set.insert(h, KeyValue::Fixed(i128::from(v)), max))
            }),
            Norm::Decimal(values) => pairs.into_iter().all(|(&i, &h)| {
                values
                    .get(i as usize)
                    .is_none_or(|&v| set.insert(h, KeyValue::Fixed(v), max))
            }),
            Norm::Float { values, single } => pairs.into_iter().all(|(&i, &h)| {
                values
                    .get(i as usize)
                    .is_none_or(|&v| set.insert(h, KeyValue::Fixed(float_bits(v, *single)), max))
            }),
            Norm::Bytes(a) => {
                // Row index -> its position among `rows` (and `h`).
                let mut pos = vec![u32::MAX; a.len()];
                for (k, &i) in rows.iter().enumerate() {
                    if let (Some(p), Ok(k)) = (pos.get_mut(i as usize), u32::try_from(k)) {
                        *p = k;
                    }
                }
                let mut ok = true;
                for_each_bytes(a.as_ref(), |i, b| {
                    let k = pos.get(i).copied().unwrap_or(u32::MAX);
                    if ok && let Some(&h) = h.get(k as usize) {
                        ok = set.insert_bytes(h, b, max);
                    }
                });
                ok
            }
        };
        if !ok {
            self.distinct = None;
        }
    }

    /// Heap bytes held (what a producer reserves for this partial).
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let values = self.distinct.as_ref().map_or(0, Distinct::heap_bytes);
        let hashes = self.hashes.as_ref().map_or(0, |h| h.capacity() * 8);
        let range = match &self.range {
            Some(KeyRange::Bytes(a, b)) => a.len() + b.len(),
            _ => 0,
        };
        values + hashes + range
    }

    /// Keeps only what costs almost nothing (a failed reservation): no
    /// Bloom hashes, and the distinct keys only while they fit the `IN`
    /// list.
    pub fn drop_bloom(&mut self) {
        self.hashes = None;
        if self
            .distinct
            .as_ref()
            .is_some_and(|d| d.table.len() > self.options.max_values)
        {
            self.distinct = None;
        }
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
        let max = self.distinct_max();
        self.distinct = match (self.distinct.take(), other.distinct) {
            (Some(mut a), Some(b)) => b
                .table
                .into_iter()
                .all(|(h, v)| a.insert(h, v, max))
                .then_some(a),
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

    /// The filter: its row form picked by the distinct key count (see
    /// [`FilterForm`]) — the range alone for dense integers, an exact set
    /// while the keys are few, else a Bloom filter sized exactly for the
    /// keys collected (unless over budget or too inaccurate) — the `IN`
    /// list for statistics while the keys fit it, and the range unless a
    /// key was NaN.
    #[must_use]
    pub fn finish(self) -> KeyFilter {
        let range = self.range.filter(|_| !self.nan);
        let members: Option<Vec<(u64, KeyValue)>> =
            self.distinct.map(|d| d.table.into_iter().collect());
        let values = members
            .as_ref()
            .filter(|m| m.len() <= self.options.max_values)
            .map(|m| {
                let mut v: Vec<KeyValue> = m.iter().map(|(_, v)| v.clone()).collect();
                v.sort_unstable();
                v
            });
        let member = match members {
            Some(m) if dense(range.as_ref(), m.len()) => Member::Range,
            Some(m) if m.len() <= self.options.exact_max_values => {
                Member::exact(self.class, m).unwrap_or(Member::Range)
            }
            _ => self
                .hashes
                .and_then(|hashes| bloom(&hashes, &self.options))
                .map_or(Member::Range, Member::Bloom),
        };
        KeyFilter {
            range,
            values,
            member,
            nulls_pass: self.nulls,
            empty: self.rows == 0,
            keys: self.rows,
            key_type: self.key_type,
        }
    }
}

/// `len` marks: `true` at each of `rows`.
fn wanted(len: usize, rows: &[u32]) -> Vec<bool> {
    let mut want = vec![false; len];
    for &i in rows {
        if let Some(w) = want.get_mut(i as usize) {
            *w = true;
        }
    }
    want
}

/// Whether `count` distinct integer keys fill their range: the range is
/// then an exact membership test.
fn dense(range: Option<&KeyRange>, count: usize) -> bool {
    match range {
        Some(&KeyRange::Int(lo, hi)) => {
            i128::from(hi) - i128::from(lo) + 1 == i128::try_from(count).unwrap_or(-1)
        }
        _ => false,
    }
}

/// A Bloom filter sized exactly for `hashes`, unless empty or expected to
/// pass too many false positives.
fn bloom(hashes: &[u64], options: &KeyFilterOptions) -> Option<RuntimeBloom> {
    if hashes.is_empty() {
        return None;
    }
    let items = hashes.len() as u64;
    let bytes = RuntimeBloom::bytes_for(items, options.bloom_fpp).min(options.bloom_max_bytes);
    let mut bloom = RuntimeBloom::with_bytes(bytes);
    for &h in hashes {
        bloom.insert_hash(h);
    }
    (bloom.expected_fpp(items) <= options.bloom_max_fpp).then_some(bloom)
}

/// What a producer knows about its keys (see the module docs). Immutable
/// once published; a versioned, serializable value
/// ([`Self::encode`], [`Self::decode`]).
#[derive(Debug, Clone, PartialEq)]
pub struct KeyFilter {
    pub(crate) key_type: DataType,
    pub(crate) range: Option<KeyRange>,
    /// The `IN` list for statistics: sorted, at most `max_values`.
    pub(crate) values: Option<Vec<KeyValue>>,
    /// The row-level form.
    pub(crate) member: Member,
    pub(crate) nulls_pass: bool,
    pub(crate) empty: bool,
    /// Non-null keys the producer stored (with duplicates).
    pub(crate) keys: u64,
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

    /// Non-null keys the producer stored, duplicates included.
    #[must_use]
    pub fn key_count(&self) -> u64 {
        self.keys
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

    /// Distinct keys held as the statistics `IN` list, if any.
    #[must_use]
    pub fn value_count(&self) -> Option<usize> {
        self.values.as_ref().map(Vec::len)
    }

    /// The Bloom filter's size in bytes, if the row form is one.
    #[must_use]
    pub fn bloom_bytes(&self) -> Option<usize> {
        match &self.member {
            Member::Bloom(b) => Some(b.num_bytes()),
            _ => None,
        }
    }

    /// The row-level form.
    #[must_use]
    pub fn form(&self) -> FilterForm {
        self.member.form()
    }

    /// Estimated cost of testing one row, in nanoseconds (see the row
    /// stage's adaptive switch-off).
    #[must_use]
    pub fn row_cost_ns(&self) -> f64 {
        self.member.row_cost_ns()
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
        values + self.member.heap_bytes()
    }

    /// The same filter with only its min/max range (a failed reservation
    /// keeps the form that costs nothing).
    #[must_use]
    pub fn range_only(&self) -> Self {
        Self {
            values: None,
            member: Member::Range,
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

    /// As [`Self::evaluate`], over keys already normalized: one range pass
    /// over the batch, then the row form over the rows still passing
    /// (hashed once, probed in a loop), then NULLs.
    #[must_use]
    pub fn evaluate_keys(&self, keys: &NormalizedKeys) -> BooleanBuffer {
        let n = keys.len();
        let valid = keys.nulls().map(NullBuffer::inner);
        if self.empty {
            return match (valid, self.nulls_pass) {
                (Some(v), true) => !v,
                _ => BooleanBuffer::new_unset(n),
            };
        }
        let pass = evaluate(self.range.as_ref(), &self.member, keys, valid);
        match valid {
            Some(v) if self.nulls_pass => &pass | &!v,
            _ => pass,
        }
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
