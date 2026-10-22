// SPDX-License-Identifier: Apache-2.0
//! Late materialisation (P2-T087): the scan's pushed predicate evaluated on
//! the predicate columns of a row group first, so the other projected
//! columns are decoded only for the rows that may match (arrow-rs
//! `RowFilter`; pages without a selected row are skipped undecompressed).
//!
//! - **Superset only.** Every row the predicate may keep survives: a leaf
//!   that cannot be evaluated exactly here (floats, other collations, nested
//!   paths, a type it does not know) keeps every row, and the `Filter` above
//!   the scan still does the exact work. Evaluation never raises.
//! - **One pass.** All evaluable conjuncts are evaluated together on the
//!   predicate columns; several passes over fragmented selections cost
//!   more than they save (E08 follow-up: a 15% leading predicate lost
//!   20-36%). A batch that keeps more than [`DENSE`] of its rows passes
//!   whole, so a wrong choice costs one extra decode of the predicate
//!   columns, not a fragmented decode of every column.
//! - **Adaptive per row group** (as streaming Parquet readers re-plan
//!   their predicate passes per row group): the decoder measures the
//!   fraction of rows the predicate keeps, from the filter itself while it
//!   is on and from the first batch of a row group while it is off, and
//!   turns it on for the next row group only when that fraction is at most
//!   [`KEEP_MAX`]. It is never used when the projected columns outside the
//!   predicate are small next to the predicate columns (the decoder's size
//!   rule). All state is the decoder's own (one split, one task).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Int8Type, Int16Type, Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrowPrimitiveType, BooleanArray, RecordBatch};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{ArrowError, DataType, TimeUnit};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ArrowPredicate, RowFilter};

use crate::request::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue};
use crate::stats::FileStatistics;

/// The largest kept fraction at which a row group is decoded with the
/// filter. The filter pays where pages or whole row groups of the other
/// columns hold no kept row (they are neither fetched nor decompressed);
/// rows kept at random hit every page above about one row per page.
/// Measured (P2-T087): no gain on a TPC-H `lineitem` file at kept
/// fractions of 1e-4 to 1e-2 (1.0-1.2x scan CPU), while ClickBench's
/// `URL LIKE '%google%'` over all 105 columns (1.6e-4) took 60x less.
pub(crate) const KEEP_MAX: f64 = 0.001;

/// The kept fraction above which a batch passes whole (see
/// [`ScanPredicate`]'s `evaluate`): above about one kept row per hundred,
/// rows kept at random leave no page of the other columns unread, and a
/// fragmented selection decodes slower than a whole batch.
const DENSE: f64 = 0.01;

/// What a node may evaluate to, per row, as two bitmaps: `may_true` holds
/// every row that may be TRUE, `may_false` every row that may be FALSE
/// (SQL NULL is neither). `None` stands for all rows (nothing is known).
#[derive(Debug, Clone, Default)]
struct Maybe {
    may_true: Option<BooleanBuffer>,
    may_false: Option<BooleanBuffer>,
}

impl Maybe {
    /// An exactly evaluated boolean column (NULL rows neither true nor false).
    fn exact(values: &BooleanBuffer, valid: Option<&BooleanBuffer>) -> Self {
        let (t, f) = match valid {
            Some(v) => (values & v, &!values & v),
            None => (values.clone(), !values),
        };
        Self {
            may_true: Some(t),
            may_false: Some(f),
        }
    }

    fn constant(b: bool, n: usize) -> Self {
        let (t, f) = if b {
            (BooleanBuffer::new_set(n), BooleanBuffer::new_unset(n))
        } else {
            (BooleanBuffer::new_unset(n), BooleanBuffer::new_set(n))
        };
        Self {
            may_true: Some(t),
            may_false: Some(f),
        }
    }

    fn not(self) -> Self {
        Self {
            may_true: self.may_false,
            may_false: self.may_true,
        }
    }
}

fn and(a: Option<BooleanBuffer>, b: Option<BooleanBuffer>) -> Option<BooleanBuffer> {
    match (a, b) {
        (Some(a), Some(b)) => Some(&a & &b),
        (x, None) | (None, x) => x,
    }
}

fn or(a: Option<BooleanBuffer>, b: Option<BooleanBuffer>) -> Option<BooleanBuffer> {
    Some(&a? | &b?)
}

/// The column a path names in `batch`: top-level paths only.
fn column<'a>(batch: &'a RecordBatch, c: &ColumnPath) -> Option<&'a dyn Array> {
    match c.0.as_slice() {
        [name] => batch.column_by_name(name).map(AsRef::as_ref),
        _ => None,
    }
}

/// `op` over two ordered values.
fn test<T: PartialOrd>(op: CmpOp, a: &T, b: &T) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::NotEq => a != b,
        CmpOp::Lt => a < b,
        CmpOp::LtEq => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::GtEq => a >= b,
    }
}

fn prim<T: ArrowPrimitiveType>(a: &dyn Array, op: CmpOp, v: T::Native) -> BooleanBuffer
where
    T::Native: PartialOrd,
{
    let values = a.as_primitive::<T>().values();
    BooleanBuffer::collect_bool(values.len(), |i| test(op, &values[i], &v))
}

/// `op(value, lit)` for every row of a byte-string column, or `None` for a
/// type that is not one.
fn bytes(a: &dyn Array, f: impl Fn(&[u8]) -> bool) -> Option<BooleanBuffer> {
    let n = a.len();
    Some(match a.data_type() {
        DataType::Utf8View => {
            let s = a.as_string_view();
            BooleanBuffer::collect_bool(n, |i| f(s.value(i).as_bytes()))
        }
        DataType::Utf8 => {
            let s = a.as_string::<i32>();
            BooleanBuffer::collect_bool(n, |i| f(s.value(i).as_bytes()))
        }
        DataType::LargeUtf8 => {
            let s = a.as_string::<i64>();
            BooleanBuffer::collect_bool(n, |i| f(s.value(i).as_bytes()))
        }
        DataType::BinaryView => {
            let s = a.as_binary_view();
            BooleanBuffer::collect_bool(n, |i| f(s.value(i)))
        }
        DataType::Binary => {
            let s = a.as_binary::<i32>();
            BooleanBuffer::collect_bool(n, |i| f(s.value(i)))
        }
        DataType::LargeBinary => {
            let s = a.as_binary::<i64>();
            BooleanBuffer::collect_bool(n, |i| f(s.value(i)))
        }
        _ => return None,
    })
}

/// `column op lit` evaluated exactly, or `None` when the pair of types is
/// not one this module compares exactly (floats are never: Spark orders
/// NaN last and equates -0.0 with 0.0).
fn compare(a: &dyn Array, op: CmpOp, lit: &ScalarValue) -> Option<BooleanBuffer> {
    use ScalarValue as S;
    Some(match (a.data_type(), lit) {
        (DataType::Int8, S::Int8(v)) => prim::<Int8Type>(a, op, *v),
        (DataType::Int16, S::Int16(v)) => prim::<Int16Type>(a, op, *v),
        (DataType::Int32, S::Int32(v)) => prim::<Int32Type>(a, op, *v),
        (DataType::Int64, S::Int64(v)) => prim::<Int64Type>(a, op, *v),
        (DataType::Date32, S::Date32(v)) => prim::<Date32Type>(a, op, *v),
        (DataType::Timestamp(TimeUnit::Microsecond, _), S::TimestampMicros(v)) => {
            prim::<TimestampMicrosecondType>(a, op, *v)
        }
        (
            DataType::Decimal128(_, s),
            S::Decimal128 {
                unscaled, scale, ..
            },
        ) if s == scale => prim::<Decimal128Type>(a, op, *unscaled),
        (DataType::Boolean, S::Boolean(v)) => {
            let values = a.as_boolean().values();
            BooleanBuffer::collect_bool(values.len(), |i| test(op, &values.value(i), v))
        }
        (_, S::Utf8(v))
            if matches!(
                a.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            ) =>
        {
            let v = v.as_bytes();
            bytes(a, |x| test(op, &x, &v))?
        }
        (_, S::Binary(v))
            if matches!(
                a.data_type(),
                DataType::Binary | DataType::LargeBinary | DataType::BinaryView
            ) =>
        {
            bytes(a, |x| test(op, &x, &v.as_slice()))?
        }
        _ => return None,
    })
}

/// Evaluates `e` (negation normal form) on `batch`.
fn eval(e: &PushdownExpr, batch: &RecordBatch) -> Maybe {
    let n = batch.num_rows();
    let exact = |c: &ColumnPath, f: &dyn Fn(&dyn Array) -> Option<BooleanBuffer>| {
        column(batch, c)
            .and_then(|a| {
                let values = f(a)?;
                let nulls = a.logical_nulls();
                Some(Maybe::exact(
                    &values,
                    nulls.as_ref().map(arrow_buffer::NullBuffer::inner),
                ))
            })
            .unwrap_or_default()
    };
    let binary = |c: &Collation| *c == Collation::Utf8Binary;
    match e {
        PushdownExpr::Constant(b) => Maybe::constant(*b, n),
        PushdownExpr::And(xs) => xs.iter().fold(Maybe::constant(true, n), |acc, x| {
            let m = eval(x, batch);
            Maybe {
                may_true: and(acc.may_true, m.may_true),
                may_false: or(acc.may_false, m.may_false),
            }
        }),
        PushdownExpr::Or(xs) => xs.iter().fold(Maybe::constant(false, n), |acc, x| {
            let m = eval(x, batch);
            Maybe {
                may_true: or(acc.may_true, m.may_true),
                may_false: and(acc.may_false, m.may_false),
            }
        }),
        PushdownExpr::Not(x) => eval(x, batch).not(),
        PushdownExpr::IsNull(c) | PushdownExpr::IsNotNull(c) => {
            let Some(a) = column(batch, c) else {
                return Maybe::default();
            };
            let null = match a.logical_nulls() {
                Some(nulls) => !nulls.inner(),
                None => BooleanBuffer::new_unset(n),
            };
            let m = Maybe::exact(&null, None);
            if matches!(e, PushdownExpr::IsNull(_)) {
                m
            } else {
                m.not()
            }
        }
        PushdownExpr::Cmp {
            column: c,
            op,
            value,
            collation,
        } if binary(collation) => exact(c, &|a| compare(a, *op, value)),
        PushdownExpr::In {
            column: c,
            values,
            collation,
        } if binary(collation) => exact(c, &|a| {
            values
                .iter()
                .try_fold(BooleanBuffer::new_unset(a.len()), |acc, v| {
                    Some(&acc | &compare(a, CmpOp::Eq, v)?)
                })
        }),
        PushdownExpr::StartsWith {
            column: c,
            prefix,
            collation,
        } if binary(collation) => exact(c, &|a| {
            matches!(
                a.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            )
            .then(|| bytes(a, |x| x.starts_with(prefix.as_bytes())))?
        }),
        PushdownExpr::Contains {
            column: c,
            needle,
            collation,
        } if binary(collation) => exact(c, &|a| {
            let finder = memchr::memmem::Finder::new(needle.as_bytes());
            matches!(
                a.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            )
            .then(|| bytes(a, |x| finder.find(x).is_some()))?
        }),
        PushdownExpr::Membership { column: c, filter } => {
            // The filter keeps a superset (Bloom false positives): a kept
            // row may be true, a dropped one is certainly not (or NULL).
            let may_true = column(batch, c)
                .and_then(|a| filter.keys.evaluate(a))
                .map(|b| b.values().clone());
            Maybe {
                may_true,
                may_false: None,
            }
        }
        _ => Maybe::default(),
    }
}

/// The rows of `batch` the predicate may keep.
pub(crate) fn evaluate(e: &PushdownExpr, batch: &RecordBatch) -> BooleanBuffer {
    eval(e, batch)
        .may_true
        .unwrap_or_else(|| BooleanBuffer::new_set(batch.num_rows()))
}

/// `v` on a number line, for interpolation between statistics bounds.
#[expect(clippy::cast_precision_loss, reason = "an estimate")]
fn position(v: &ScalarValue) -> Option<f64> {
    Some(match v {
        ScalarValue::Int8(x) => f64::from(*x),
        ScalarValue::Int16(x) => f64::from(*x),
        ScalarValue::Int32(x) | ScalarValue::Date32(x) => f64::from(*x),
        ScalarValue::Int64(x) | ScalarValue::TimestampMicros(x) => *x as f64,
        ScalarValue::Decimal128 {
            unscaled, scale, ..
        } => *unscaled as f64 / 10f64.powi(i32::from(*scale)),
        _ => return None,
    })
}

/// The fraction of rows `e` (negation normal form) keeps, estimated from
/// `stats` with values spread evenly between min and max and conjuncts
/// independent (System R's rules); `None` where nothing is known
/// (equality, strings, runtime filters).
pub(crate) fn estimate(e: &PushdownExpr, stats: &FileStatistics) -> Option<f64> {
    match e {
        PushdownExpr::Constant(b) => Some(if *b { 1.0 } else { 0.0 }),
        PushdownExpr::And(xs) => {
            let known: Vec<f64> = xs.iter().filter_map(|x| estimate(x, stats)).collect();
            (!known.is_empty()).then(|| known.iter().product())
        }
        PushdownExpr::Or(xs) => xs
            .iter()
            .map(|x| estimate(x, stats))
            .sum::<Option<f64>>()
            .map(|f| f.min(1.0)),
        PushdownExpr::Cmp {
            column, op, value, ..
        } if matches!(op, CmpOp::Lt | CmpOp::LtEq | CmpOp::Gt | CmpOp::GtEq) => {
            let c = stats.column(column)?;
            let (lo, hi) = (position(c.min.value()?)?, position(c.max.value()?)?);
            let v = position(value)?;
            if hi <= lo {
                return None;
            }
            let below = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            Some(if matches!(op, CmpOp::Lt | CmpOp::LtEq) {
                below
            } else {
                1.0 - below
            })
        }
        _ => None,
    }
}

/// The top-level columns `e` reads, deduplicated, in order of appearance
/// (leaves on nested paths are not evaluated here and add none).
pub(crate) fn columns(e: &PushdownExpr) -> Vec<ColumnPath> {
    fn walk(e: &PushdownExpr, out: &mut Vec<ColumnPath>) {
        let c = match e {
            PushdownExpr::And(xs) | PushdownExpr::Or(xs) => {
                xs.iter().for_each(|x| walk(x, out));
                return;
            }
            PushdownExpr::Not(x) => return walk(x, out),
            PushdownExpr::Constant(_) => return,
            PushdownExpr::IsNull(c)
            | PushdownExpr::IsNotNull(c)
            | PushdownExpr::Cmp { column: c, .. }
            | PushdownExpr::In { column: c, .. }
            | PushdownExpr::StartsWith { column: c, .. }
            | PushdownExpr::Contains { column: c, .. }
            | PushdownExpr::Membership { column: c, .. } => c,
        };
        if c.0.len() == 1 && !out.contains(c) {
            out.push(c.clone());
        }
    }
    let mut out = Vec::new();
    walk(e, &mut out);
    out
}

/// Rows the filter saw and kept, since the decoder last read them.
#[derive(Debug, Default)]
pub(crate) struct Counters {
    seen: AtomicU64,
    kept: AtomicU64,
}

impl Counters {
    pub(crate) fn add(&self, seen: usize, kept: usize) {
        self.seen.fetch_add(seen as u64, Ordering::Relaxed);
        self.kept.fetch_add(kept as u64, Ordering::Relaxed);
    }

    /// `(seen, kept)`, and resets both.
    pub(crate) fn take(&self) -> (u64, u64) {
        (
            self.seen.swap(0, Ordering::Relaxed),
            self.kept.swap(0, Ordering::Relaxed),
        )
    }
}

/// Widens a decoded column to the type the file-level predicate was typed
/// against (see `decode::spark_read_array`); `None` if it cannot be.
pub(crate) type Widen = fn(&arrow_array::ArrayRef) -> Option<arrow_array::ArrayRef>;

/// The predicate as arrow-rs evaluates it while decoding.
pub(crate) struct ScanPredicate {
    expr: Arc<PushdownExpr>,
    projection: ProjectionMask,
    widen: Option<Widen>,
    counters: Arc<Counters>,
}

impl ScanPredicate {
    pub(crate) fn new(
        expr: Arc<PushdownExpr>,
        projection: ProjectionMask,
        widen: Option<Widen>,
        counters: Arc<Counters>,
    ) -> Self {
        Self {
            expr,
            projection,
            widen,
            counters,
        }
    }

    /// A row filter of this one predicate.
    pub(crate) fn into_filter(self) -> RowFilter {
        RowFilter::new(vec![Box::new(self)])
    }
}

/// `batch` with every column widened by `widen`; a column that cannot be
/// is left out (its leaves then keep every row).
pub(crate) fn widened(batch: &RecordBatch, widen: Option<Widen>) -> RecordBatch {
    let Some(widen) = widen else {
        return batch.clone();
    };
    let (fields, columns): (Vec<_>, Vec<_>) = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .filter_map(|(f, c)| {
            let c = widen(c)?;
            Some((
                Arc::new(f.as_ref().clone().with_data_type(c.data_type().clone())),
                c,
            ))
        })
        .unzip();
    RecordBatch::try_new_with_options(
        Arc::new(arrow_schema::Schema::new(fields)),
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .unwrap_or_else(|_| RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty())))
}

impl ArrowPredicate for ScanPredicate {
    fn projection(&self) -> &ProjectionMask {
        &self.projection
    }

    fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
        let n = batch.num_rows();
        let batch = widened(&batch, self.widen);
        let keep = if batch.num_rows() == n {
            evaluate(&self.expr, &batch)
        } else {
            BooleanBuffer::new_set(n)
        };
        let kept = keep.count_set_bits();
        self.counters.add(n, kept);
        // A dense batch is passed whole (a superset): decoding the other
        // columns with a fragmented selection costs more than the rows it
        // drops, and the `Filter` above drops them anyway.
        #[expect(clippy::cast_precision_loss, reason = "a ratio")]
        let dense = kept as f64 > n as f64 * DENSE;
        Ok(if dense {
            BooleanArray::new(BooleanBuffer::new_set(n), None)
        } else {
            BooleanArray::new(keep, None)
        })
    }
}

/// Bytes of decoded predicate columns arrow-rs may keep per row group to
/// hand them to the output without decoding them again (its predicate
/// cache; arrow-rs's default is 100 MiB per decoder). Off: that memory
/// would sit outside the query's reservation, and it pays only where the
/// filter is on, i.e. where few rows (and pages) of the predicate columns
/// are decoded a second time.
pub(crate) const CACHE_BYTES: usize = 0;

/// Whether filtering a row group's predicate columns first may pay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Worth {
    /// Yes.
    Yes,
    /// The other columns are too small to pay for the predicate columns'
    /// second decode; only a forced policy uses the filter.
    Small,
    /// Never: every projected leaf is a predicate leaf.
    Never,
}

/// The late-materialisation state of one decoder (see the module docs).
pub(crate) struct LateMaterialization {
    policy: crate::request::RowFilterPolicy,
    /// The file-level static predicate, in negation normal form.
    base: Option<PushdownExpr>,
    widen: Option<Widen>,
    counters: Arc<Counters>,
    /// The current row group is decoded with the filter.
    on: bool,
    /// The filter cannot apply to this scan at all.
    disabled: bool,
    /// Evaluate the next decoded batch to measure the kept fraction.
    sample: bool,
    /// A kept fraction was measured.
    measured: bool,
    /// What the last measurement (or the first guess) asked for.
    wanted: bool,
}

impl LateMaterialization {
    /// `eligible`: the scan reads no row index and no column needs a
    /// calendar check, which must see every row of a decoded row group.
    /// `kept`: the fraction of rows the static predicate keeps, estimated
    /// from statistics; above [`DENSE`] the first row group is read
    /// without the filter (and measured).
    pub(crate) fn new(
        policy: crate::request::RowFilterPolicy,
        base: Option<&PushdownExpr>,
        widen: Option<Widen>,
        eligible: bool,
        kept: Option<f64>,
    ) -> Self {
        let disabled = !eligible || policy == crate::request::RowFilterPolicy::Off;
        let guess = !disabled
            && (policy == crate::request::RowFilterPolicy::On || kept.is_none_or(|f| f <= DENSE));
        Self {
            policy,
            base: base.map(PushdownExpr::to_nnf),
            widen,
            counters: Arc::new(Counters::default()),
            // With nothing measured yet the first row group uses the filter:
            // where the size rule allows it, wrongly on costs a second decode
            // of the (smaller) predicate columns of one row group, wrongly
            // off a whole decode of the (larger) rest.
            on: guess,
            disabled,
            sample: false,
            measured: false,
            wanted: guess,
        }
    }

    /// Whether the current row group is decoded with the filter.
    pub(crate) fn is_on(&self) -> bool {
        self.on
    }

    /// Whether the filter is on by default, before any measurement.
    pub(crate) fn is_guess(&self) -> bool {
        self.on && !self.measured
    }

    /// The predicate over the static and the current runtime filters, if
    /// it reads at least one column.
    fn expr(&self, dynamic: &[PushdownExpr]) -> Option<PushdownExpr> {
        let mut all: Vec<PushdownExpr> = self.base.iter().cloned().collect();
        all.extend(dynamic.iter().map(PushdownExpr::to_nnf));
        let e = match all.len() {
            0 => return None,
            1 => all.pop()?,
            _ => PushdownExpr::And(all),
        };
        (!columns(&e).is_empty()).then_some(e)
    }

    /// The row filter for the next row group (an empty one when off).
    /// `mask(columns)` gives the predicate's leaves and whether filtering
    /// may pay; when it cannot, the filter stays off for the rest of the
    /// scan (the predicate columns only grow with later runtime filters).
    pub(crate) fn filter(
        &mut self,
        on: bool,
        dynamic: &[PushdownExpr],
        mask: impl FnOnce(&[ColumnPath]) -> (ProjectionMask, Worth),
    ) -> RowFilter {
        let built = if on && !self.disabled {
            self.expr(dynamic).and_then(|e| {
                let (projection, worth) = mask(&columns(&e));
                let forced = self.policy == crate::request::RowFilterPolicy::On;
                let use_it = worth == Worth::Yes || (forced && worth == Worth::Small);
                self.disabled |= !use_it;
                use_it.then(|| {
                    ScanPredicate::new(
                        Arc::new(e),
                        projection,
                        self.widen,
                        Arc::clone(&self.counters),
                    )
                    .into_filter()
                })
            })
        } else {
            None
        };
        self.on = built.is_some();
        built.unwrap_or_else(|| RowFilter::new(Vec::new()))
    }

    /// At a row-group boundary: whether the next row group should use the
    /// filter, from what the last one measured.
    pub(crate) fn want(&mut self) -> bool {
        if self.disabled {
            return false;
        }
        if self.policy == crate::request::RowFilterPolicy::On {
            return true;
        }
        let (seen, kept) = self.counters.take();
        if seen > 0 {
            self.measured = true;
            #[expect(clippy::cast_precision_loss, reason = "a ratio")]
            let fraction = kept as f64 / seen as f64;
            self.wanted = fraction <= KEEP_MAX;
        }
        self.wanted
    }

    /// Whether the next decoded batch is to be measured.
    pub(crate) fn sampling(&self) -> bool {
        self.sample
    }

    /// A row group starts: while off, its first batch is measured.
    pub(crate) fn start_row_group(&mut self) {
        self.sample = !self.on && !self.disabled;
    }

    /// Measures `batch` (decoded without the filter) if a sample is due.
    pub(crate) fn observe(&mut self, batch: &RecordBatch, dynamic: &[PushdownExpr]) {
        if !std::mem::take(&mut self.sample) {
            return;
        }
        let Some(e) = self.expr(dynamic) else {
            return;
        };
        let n = batch.num_rows();
        let w = widened(batch, self.widen);
        if w.num_rows() == n {
            self.counters.add(n, evaluate(&e, &w).count_set_bits());
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int8Array,
        Int16Array, Int32Array, Int64Array, StringArray, StringViewArray,
        TimestampMicrosecondArray,
    };
    use arrow_schema::{Field, Schema};

    use super::*;
    use crate::runtime_filter::{KeyFilterBuilder, KeyFilterOptions};

    fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let fields: Vec<Field> = cols
            .iter()
            .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
            .collect();
        RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            cols.into_iter().map(|(_, a)| a).collect(),
        )
        .unwrap()
    }

    fn kept(e: &PushdownExpr, b: &RecordBatch) -> Vec<bool> {
        evaluate(&e.to_nnf(), b).iter().collect()
    }

    /// The number of predicates of `f` (arrow-rs keeps them private).
    fn n(f: &RowFilter) -> usize {
        let text = format!("{f:?}");
        text.split_whitespace()
            .find_map(|w| w.parse().ok())
            .unwrap_or_else(|| panic!("{text}"))
    }

    fn c(name: &str) -> ColumnPath {
        ColumnPath::new(name)
    }

    fn cmp(name: &str, op: CmpOp, v: ScalarValue) -> PushdownExpr {
        PushdownExpr::cmp(c(name), op, v)
    }

    fn ints() -> RecordBatch {
        batch(vec![(
            "a",
            Arc::new(Int32Array::from(vec![Some(1), None, Some(3), Some(5)])) as ArrayRef,
        )])
    }

    #[test]
    fn comparisons_follow_sql_and_drop_nulls() {
        let b = ints();
        let v = || ScalarValue::Int32(3);
        assert_eq!(
            kept(&cmp("a", CmpOp::Eq, v()), &b),
            [false, false, true, false]
        );
        assert_eq!(
            kept(&cmp("a", CmpOp::NotEq, v()), &b),
            [true, false, false, true]
        );
        assert_eq!(
            kept(&cmp("a", CmpOp::Lt, v()), &b),
            [true, false, false, false]
        );
        assert_eq!(
            kept(&cmp("a", CmpOp::LtEq, v()), &b),
            [true, false, true, false]
        );
        assert_eq!(
            kept(&cmp("a", CmpOp::Gt, v()), &b),
            [false, false, false, true]
        );
        assert_eq!(
            kept(&cmp("a", CmpOp::GtEq, v()), &b),
            [false, false, true, true]
        );
        // NOT (a = 3) keeps neither the match nor the NULL.
        let not = PushdownExpr::Not(Box::new(cmp("a", CmpOp::Eq, v())));
        assert_eq!(kept(&not, &b), [true, false, false, true]);
        assert_eq!(
            kept(&PushdownExpr::IsNull(c("a")), &b),
            [false, true, false, false]
        );
        let not_null = PushdownExpr::Not(Box::new(PushdownExpr::IsNull(c("a"))));
        assert_eq!(kept(&not_null, &b), [true, false, true, true]);
        let within = PushdownExpr::In {
            column: c("a"),
            values: vec![ScalarValue::Int32(1), ScalarValue::Int32(5)],
            collation: Collation::Utf8Binary,
        };
        assert_eq!(kept(&within, &b), [true, false, false, true]);
        // NOT IN: a conjunction of <> after NNF.
        let outside = PushdownExpr::Not(Box::new(within));
        assert_eq!(kept(&outside, &b), [false, false, true, false]);
    }

    #[test]
    fn connectives_are_three_valued() {
        let b = ints();
        let eq = |v| cmp("a", CmpOp::Eq, ScalarValue::Int32(v));
        let or = PushdownExpr::Or(vec![eq(1), eq(5)]);
        assert_eq!(kept(&or, &b), [true, false, false, true]);
        let and = PushdownExpr::And(vec![or.clone(), cmp("a", CmpOp::Gt, ScalarValue::Int32(2))]);
        assert_eq!(kept(&and, &b), [false, false, false, true]);
        // NOT (a = 1 OR a = 5) = a <> 1 AND a <> 5: NULL stays out.
        assert_eq!(
            kept(&PushdownExpr::Not(Box::new(or)), &b),
            [false, false, true, false]
        );
        assert_eq!(kept(&PushdownExpr::Constant(false), &b), [false; 4]);
        assert_eq!(kept(&PushdownExpr::Constant(true), &b), [true; 4]);
    }

    #[test]
    fn what_cannot_be_evaluated_exactly_keeps_every_row() {
        let f = batch(vec![(
            "f",
            Arc::new(Float64Array::from(vec![0.0, -0.0, f64::NAN, 1.0])) as ArrayRef,
        )]);
        // Spark: -0.0 = 0.0 and NaN = NaN; floats are left to the Filter.
        let e = cmp("f", CmpOp::Eq, ScalarValue::Float64(0.0));
        assert_eq!(kept(&e, &f), [true; 4]);
        assert_eq!(kept(&PushdownExpr::Not(Box::new(e)), &f), [true; 4]);
        let b = ints();
        // A literal of another type, a missing or nested column, another
        // collation: unknown, never a dropped row.
        assert_eq!(
            kept(&cmp("a", CmpOp::Eq, ScalarValue::Int64(3)), &b),
            [true; 4]
        );
        assert_eq!(
            kept(&cmp("zz", CmpOp::Eq, ScalarValue::Int32(3)), &b),
            [true; 4]
        );
        let nested = PushdownExpr::cmp(
            ColumnPath::nested(["a", "x"]),
            CmpOp::Eq,
            ScalarValue::Int32(3),
        );
        assert_eq!(kept(&nested, &b), [true; 4]);
        let s = batch(vec![(
            "s",
            Arc::new(StringViewArray::from(vec!["A", "a"])) as ArrayRef,
        )]);
        let lcase = PushdownExpr::Cmp {
            column: c("s"),
            op: CmpOp::Eq,
            value: ScalarValue::Utf8("a".to_owned()),
            collation: Collation::Other("UTF8_LCASE".to_owned()),
        };
        assert_eq!(kept(&lcase, &s), [true, true]);
        assert_eq!(kept(&PushdownExpr::Not(Box::new(lcase)), &s), [true, true]);
        // An unknown conjunct leaves the rest; an unknown disjunct keeps all.
        let and = PushdownExpr::And(vec![
            cmp("zz", CmpOp::Eq, ScalarValue::Int32(3)),
            cmp("a", CmpOp::Eq, ScalarValue::Int32(3)),
        ]);
        assert_eq!(kept(&and, &b), [false, false, true, false]);
        let or = PushdownExpr::Or(vec![
            cmp("zz", CmpOp::Eq, ScalarValue::Int32(3)),
            cmp("a", CmpOp::Eq, ScalarValue::Int32(3)),
        ]);
        assert_eq!(kept(&or, &b), [true; 4]);
    }

    #[test]
    fn every_exact_type_compares_by_value() {
        let b = batch(vec![
            ("i8", Arc::new(Int8Array::from(vec![1, 2])) as ArrayRef),
            ("i16", Arc::new(Int16Array::from(vec![1, 2]))),
            ("i64", Arc::new(Int64Array::from(vec![1, 2]))),
            ("d", Arc::new(Date32Array::from(vec![1, 2]))),
            (
                "ts",
                Arc::new(TimestampMicrosecondArray::from(vec![1, 2]).with_timezone("UTC")),
            ),
            (
                "dec",
                Arc::new(
                    Decimal128Array::from(vec![100, 250])
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
            ),
            ("b", Arc::new(BooleanArray::from(vec![false, true]))),
            ("s", Arc::new(StringArray::from(vec!["ab", "b"]))),
            ("v", Arc::new(StringViewArray::from(vec!["ab", "b"]))),
            (
                "bin",
                Arc::new(BinaryArray::from(vec![b"ab".as_ref(), b"b".as_ref()])),
            ),
        ]);
        let ge = |n: &str, v| kept(&cmp(n, CmpOp::GtEq, v), &b);
        assert_eq!(ge("i8", ScalarValue::Int8(2)), [false, true]);
        assert_eq!(ge("i16", ScalarValue::Int16(2)), [false, true]);
        assert_eq!(ge("i64", ScalarValue::Int64(2)), [false, true]);
        assert_eq!(ge("d", ScalarValue::Date32(2)), [false, true]);
        assert_eq!(ge("ts", ScalarValue::TimestampMicros(2)), [false, true]);
        let dec = |scale| ScalarValue::Decimal128 {
            unscaled: 250,
            precision: 10,
            scale,
        };
        assert_eq!(ge("dec", dec(2)), [false, true]);
        // Another scale is another value: not compared here.
        assert_eq!(ge("dec", dec(3)), [true, true]);
        assert_eq!(ge("b", ScalarValue::Boolean(true)), [false, true]);
        for s in ["s", "v"] {
            assert_eq!(ge(s, ScalarValue::Utf8("b".to_owned())), [false, true]);
            // Byte order: "ab" < "b".
            assert_eq!(
                kept(&cmp(s, CmpOp::Lt, ScalarValue::Utf8("b".to_owned())), &b),
                [true, false]
            );
        }
        assert_eq!(ge("bin", ScalarValue::Binary(b"b".to_vec())), [false, true]);
        // A string literal against binary (and back) is not compared.
        assert_eq!(ge("bin", ScalarValue::Utf8("b".to_owned())), [true, true]);
        assert_eq!(ge("s", ScalarValue::Binary(b"b".to_vec())), [true, true]);
    }

    #[test]
    fn prefixes_and_substrings_match_bytes() {
        let b = batch(vec![(
            "u",
            Arc::new(StringViewArray::from(vec![
                Some("http://google.com"),
                Some("mail.google"),
                Some("bing"),
                None,
            ])) as ArrayRef,
        )]);
        let contains = PushdownExpr::Contains {
            column: c("u"),
            needle: "google".to_owned(),
            collation: Collation::Utf8Binary,
        };
        assert_eq!(kept(&contains, &b), [true, true, false, false]);
        assert_eq!(
            kept(&PushdownExpr::Not(Box::new(contains)), &b),
            [false, false, true, false]
        );
        let prefix = PushdownExpr::StartsWith {
            column: c("u"),
            prefix: "http".to_owned(),
            collation: Collation::Utf8Binary,
        };
        assert_eq!(kept(&prefix, &b), [true, false, false, false]);
        // Not a string column: unknown.
        let on_ints = PushdownExpr::Contains {
            column: c("a"),
            needle: "1".to_owned(),
            collation: Collation::Utf8Binary,
        };
        assert_eq!(kept(&on_ints, &ints()), [true; 4]);
    }

    #[test]
    fn membership_keeps_a_superset_and_never_negates() {
        let mut builder = KeyFilterBuilder::new(DataType::Int64, KeyFilterOptions::default())
            .unwrap_or_else(|| panic!("Int64 keys"));
        builder.add(&Int64Array::from(vec![1, 5]), None).unwrap();
        let member = PushdownExpr::Membership {
            column: c("a"),
            filter: crate::MembershipFilter {
                id: crate::FilterId(1),
                generation: 1,
                keys: Arc::new(builder.finish()),
            },
        };
        let b = ints();
        assert_eq!(kept(&member, &b), [true, false, false, true]);
        // NOT of a superset is not a superset: every row stays.
        assert_eq!(kept(&PushdownExpr::Not(Box::new(member)), &b), [true; 4]);
    }

    #[test]
    fn columns_are_the_top_level_leaves() {
        let e = PushdownExpr::And(vec![
            cmp("a", CmpOp::Eq, ScalarValue::Int32(1)),
            PushdownExpr::Not(Box::new(PushdownExpr::IsNull(c("b")))),
            PushdownExpr::Or(vec![
                cmp("a", CmpOp::Lt, ScalarValue::Int32(1)),
                PushdownExpr::cmp(
                    ColumnPath::nested(["s", "x"]),
                    CmpOp::Eq,
                    ScalarValue::Int32(1),
                ),
            ]),
            PushdownExpr::Constant(true),
        ]);
        assert_eq!(columns(&e), [c("a"), c("b")]);
    }

    #[test]
    fn the_policy_follows_the_measured_fraction() {
        let mut late = LateMaterialization::new(
            crate::request::RowFilterPolicy::Auto,
            Some(&cmp("a", CmpOp::Eq, ScalarValue::Int32(3))),
            None,
            true,
            None,
        );
        let all = |_: &[ColumnPath]| (ProjectionMask::all(), Worth::Yes);
        // Nothing measured: the first row group filters.
        assert!(late.is_on());
        assert_eq!(n(&late.filter(true, &[], all)), 1);
        // A quarter kept: off for the next row group.
        late.counters.add(4, 1);
        assert!(!late.want());
        assert_eq!(n(&late.filter(false, &[], all)), 0);
        assert!(!late.is_on());
        // Off: the first batch of a row group is measured, then no other.
        late.start_row_group();
        assert!(late.sampling());
        let many = batch(vec![(
            "a",
            Arc::new(Int32Array::from_iter_values((0..1000).map(|i| i % 500 + 4))) as ArrayRef,
        )]);
        late.observe(&many, &[]);
        assert!(!late.sampling());
        late.observe(&ints(), &[]);
        // 0 of 1000 kept (the second batch was not counted): on again.
        assert!(late.want());
        // Too little outside the predicate: off for good unless forced.
        let small = |_: &[ColumnPath]| (ProjectionMask::all(), Worth::Small);
        let mut auto = LateMaterialization::new(
            crate::request::RowFilterPolicy::Auto,
            Some(&cmp("a", CmpOp::Eq, ScalarValue::Int32(3))),
            None,
            true,
            None,
        );
        assert_eq!(n(&auto.filter(true, &[], small)), 0);
        assert!(!auto.want());
        let mut forced = LateMaterialization::new(
            crate::request::RowFilterPolicy::On,
            Some(&cmp("a", CmpOp::Eq, ScalarValue::Int32(3))),
            None,
            true,
            None,
        );
        assert_eq!(n(&forced.filter(true, &[], small)), 1);
        assert!(forced.want());
        // No projected column outside the predicate: nothing to save, ever.
        let only = |_: &[ColumnPath]| (ProjectionMask::all(), Worth::Never);
        assert_eq!(n(&late.filter(true, &[], only)), 0);
        assert!(!late.is_on());
        assert!(!late.want());
    }

    #[test]
    fn statistics_estimate_ranges_only() {
        use crate::stats::{ColumnStatistics, Precision};
        let mut stats = FileStatistics::with_rows(Precision::Exact(100));
        let mut a = ColumnStatistics::new(c("a"));
        a.min = Precision::Exact(ScalarValue::Int32(0));
        a.max = Precision::Exact(ScalarValue::Int32(100));
        let mut d = ColumnStatistics::new(c("d"));
        d.min = Precision::Inexact(ScalarValue::Decimal128 {
            unscaled: 0,
            precision: 10,
            scale: 2,
        });
        d.max = Precision::Exact(ScalarValue::Decimal128 {
            unscaled: 1000,
            precision: 10,
            scale: 2,
        });
        stats.columns = vec![a, d];
        let est = |e: &PushdownExpr| estimate(&e.to_nnf(), &stats);
        let lt = |v| cmp("a", CmpOp::Lt, ScalarValue::Int32(v));
        assert_eq!(est(&lt(25)), Some(0.25));
        assert_eq!(
            est(&cmp("a", CmpOp::GtEq, ScalarValue::Int32(25))),
            Some(0.75)
        );
        assert_eq!(est(&lt(-5)), Some(0.0));
        assert_eq!(est(&lt(500)), Some(1.0));
        let dec = cmp(
            "d",
            CmpOp::LtEq,
            ScalarValue::Decimal128 {
                unscaled: 250,
                precision: 10,
                scale: 2,
            },
        );
        assert_eq!(est(&dec), Some(0.25));
        // Conjuncts multiply; unknown ones are left out; disjuncts add.
        let eq = cmp("a", CmpOp::Eq, ScalarValue::Int32(3));
        assert_eq!(
            est(&PushdownExpr::And(vec![lt(50), dec.clone(), eq.clone()])),
            Some(0.125)
        );
        assert_eq!(est(&PushdownExpr::Or(vec![lt(50), dec])), Some(0.75));
        assert_eq!(est(&PushdownExpr::Or(vec![lt(90), lt(90)])), Some(1.0));
        assert_eq!(est(&PushdownExpr::Or(vec![lt(50), eq.clone()])), None);
        assert_eq!(est(&eq), None);
        assert_eq!(est(&PushdownExpr::Constant(false)), Some(0.0));
        assert_eq!(est(&cmp("zz", CmpOp::Lt, ScalarValue::Int32(1))), None);
        assert_eq!(
            est(&cmp("a", CmpOp::Lt, ScalarValue::Utf8("x".into()))),
            None
        );
        let mut flat = ColumnStatistics::new(c("f"));
        flat.min = Precision::Exact(ScalarValue::Int64(7));
        flat.max = Precision::Exact(ScalarValue::Int64(7));
        stats.columns.push(flat);
        let f = cmp("f", CmpOp::Lt, ScalarValue::Int64(9));
        assert_eq!(estimate(&f, &stats), None);
        // An estimate above 1% starts without the filter (and measures).
        let e = lt(50);
        let auto = crate::request::RowFilterPolicy::Auto;
        let mut late = LateMaterialization::new(auto, Some(&e), None, true, Some(0.5));
        assert!(!late.is_on());
        late.start_row_group();
        assert!(late.sampling());
        assert!(LateMaterialization::new(auto, Some(&e), None, true, Some(0.005)).is_on());
        let on = crate::request::RowFilterPolicy::On;
        assert!(LateMaterialization::new(on, Some(&e), None, true, Some(0.5)).is_on());
    }

    #[test]
    fn forced_and_ineligible_policies() {
        let e = cmp("a", CmpOp::Eq, ScalarValue::Int32(3));
        let mut on = LateMaterialization::new(
            crate::request::RowFilterPolicy::On,
            Some(&e),
            None,
            true,
            None,
        );
        on.counters.add(10, 10);
        assert!(on.want());
        let mut off = LateMaterialization::new(
            crate::request::RowFilterPolicy::Off,
            Some(&e),
            None,
            true,
            None,
        );
        assert!(!off.is_on());
        assert!(!off.want());
        off.start_row_group();
        assert!(!off.sampling());
        let mut row_index = LateMaterialization::new(
            crate::request::RowFilterPolicy::On,
            Some(&e),
            None,
            false,
            None,
        );
        assert!(!row_index.want());
        let all = |_: &[ColumnPath]| (ProjectionMask::all(), Worth::Yes);
        assert_eq!(n(&row_index.filter(true, &[], all)), 0);
        // No predicate at all: nothing to evaluate.
        let mut none =
            LateMaterialization::new(crate::request::RowFilterPolicy::On, None, None, true, None);
        assert_eq!(n(&none.filter(true, &[], all)), 0);
    }
}
