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
//!   rule). All state is the decoder's own (one piece of a split, one
//!   driver), except that a later piece of the file may start from what the
//!   last one measured, so cutting a split finer does not guess again
//!   ([`LateMaterialization::resuming`]).
//! - **A top-N bound goes first, in its own pass.** It usually reads one
//!   small column and, once tight, drops most rows, so the larger predicate
//!   and output columns are decoded only where it left something. It doesn't
//!   adapt per row group: it tightens, so an earlier row group says nothing
//!   about the next one, and a useless pass costs one decode of a column
//!   that the size rule requires to be small.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType,
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

/// `collect_bool` over `n` rows of `$at(i) $cmp $v` for the comparison
/// `$op` names, chosen once outside the loop so each loop is one
/// monomorphic comparison (vectorized for primitives).
macro_rules! by_op {
    ($op:expr, $n:expr, |$i:ident| $at:expr, $v:expr) => {{
        let n = $n;
        match $op {
            CmpOp::Eq => BooleanBuffer::collect_bool(n, |$i| $at == $v),
            CmpOp::NotEq => BooleanBuffer::collect_bool(n, |$i| $at != $v),
            CmpOp::Lt => BooleanBuffer::collect_bool(n, |$i| $at < $v),
            CmpOp::LtEq => BooleanBuffer::collect_bool(n, |$i| $at <= $v),
            CmpOp::Gt => BooleanBuffer::collect_bool(n, |$i| $at > $v),
            CmpOp::GtEq => BooleanBuffer::collect_bool(n, |$i| $at >= $v),
        }
    }};
}

fn prim<T: ArrowPrimitiveType>(a: &dyn Array, op: CmpOp, v: T::Native) -> BooleanBuffer
where
    T::Native: PartialOrd,
{
    let values: &[T::Native] = a.as_primitive::<T>().values();
    by_op!(op, values.len(), |i| values[i], v)
}

/// `op(value, lit)` for every row of a byte-string column (see [`bytes`]).
fn bytes_cmp(a: &dyn Array, op: CmpOp, lit: &[u8]) -> Option<BooleanBuffer> {
    match op {
        CmpOp::Eq => bytes(a, |x| x == lit),
        CmpOp::NotEq => bytes(a, |x| x != lit),
        CmpOp::Lt => bytes(a, |x| x < lit),
        CmpOp::LtEq => bytes(a, |x| x <= lit),
        CmpOp::Gt => bytes(a, |x| x > lit),
        CmpOp::GtEq => bytes(a, |x| x >= lit),
    }
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
            DataType::Decimal64(_, s),
            S::Decimal128 {
                unscaled, scale, ..
            },
        ) if s == scale => match i64::try_from(*unscaled) {
            Ok(v) => prim::<Decimal64Type>(a, op, v),
            // Beyond every value of the column: not filtered here.
            Err(_) => return None,
        },
        (
            DataType::Decimal128(_, s),
            S::Decimal128 {
                unscaled, scale, ..
            },
        ) if s == scale => prim::<Decimal128Type>(a, op, *unscaled),
        (DataType::Boolean, S::Boolean(v)) => {
            let values = a.as_boolean().values();
            by_op!(op, values.len(), |i| values.value(i), *v)
        }
        (_, S::Utf8(v))
            if matches!(
                a.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            ) =>
        {
            bytes_cmp(a, op, v.as_bytes())?
        }
        (_, S::Binary(v))
            if matches!(
                a.data_type(),
                DataType::Binary | DataType::LargeBinary | DataType::BinaryView
            ) =>
        {
            bytes_cmp(a, op, v)?
        }
        _ => return None,
    })
}

/// The `IN` lists at most this long are compared value by value (one
/// vectorised pass each); longer ones look each row up in the sorted values.
const IN_PASSES_MAX: usize = 8;

/// `column IN values` evaluated exactly (NULL rows aside), or `None` as for
/// [`compare`] when a value's type is not one compared exactly here.
fn in_list(a: &dyn Array, values: &[ScalarValue]) -> Option<BooleanBuffer> {
    use ScalarValue as S;
    fn passes(a: &dyn Array, values: &[ScalarValue]) -> Option<BooleanBuffer> {
        values
            .iter()
            .try_fold(BooleanBuffer::new_unset(a.len()), |acc, v| {
                Some(&acc | &compare(a, CmpOp::Eq, v)?)
            })
    }
    /// The values as `T`'s natives, sorted, or `None` if one is not one.
    fn natives<'v, N: Ord>(
        values: &'v [ScalarValue],
        f: impl Fn(&'v ScalarValue) -> Option<N>,
    ) -> Option<Vec<N>> {
        let mut v = values.iter().map(f).collect::<Option<Vec<N>>>()?;
        v.sort_unstable();
        Some(v)
    }
    fn lookup<T: ArrowPrimitiveType>(
        a: &dyn Array,
        values: &[ScalarValue],
        f: impl Fn(&ScalarValue) -> Option<T::Native>,
    ) -> Option<BooleanBuffer>
    where
        T::Native: Ord,
    {
        let sorted = natives(values, f)?;
        let col = a.as_primitive::<T>().values();
        Some(BooleanBuffer::collect_bool(col.len(), |i| {
            sorted.binary_search(&col[i]).is_ok()
        }))
    }
    if values.len() <= IN_PASSES_MAX {
        return passes(a, values);
    }
    let decimal = |s: i8| {
        move |v: &ScalarValue| match v {
            S::Decimal128 {
                unscaled, scale, ..
            } if *scale == s => Some(*unscaled),
            _ => None,
        }
    };
    match a.data_type() {
        DataType::Int8 => lookup::<Int8Type>(a, values, |v| match v {
            S::Int8(x) => Some(*x),
            _ => None,
        }),
        DataType::Int16 => lookup::<Int16Type>(a, values, |v| match v {
            S::Int16(x) => Some(*x),
            _ => None,
        }),
        DataType::Int32 => lookup::<Int32Type>(a, values, |v| match v {
            S::Int32(x) => Some(*x),
            _ => None,
        }),
        DataType::Int64 => lookup::<Int64Type>(a, values, |v| match v {
            S::Int64(x) => Some(*x),
            _ => None,
        }),
        DataType::Date32 => lookup::<Date32Type>(a, values, |v| match v {
            S::Date32(x) => Some(*x),
            _ => None,
        }),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            lookup::<TimestampMicrosecondType>(a, values, |v| match v {
                S::TimestampMicros(x) => Some(*x),
                _ => None,
            })
        }
        DataType::Decimal128(_, s) => lookup::<Decimal128Type>(a, values, decimal(*s)),
        // A literal beyond every Decimal64 value is not filtered here (as in
        // `compare`): the whole list is then not evaluated.
        DataType::Decimal64(_, s) => {
            let d = decimal(*s);
            lookup::<Decimal64Type>(a, values, |v| i64::try_from(d(v)?).ok())
        }
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            let sorted = natives(values, |v: &ScalarValue| match v {
                S::Utf8(x) => Some(x.as_bytes()),
                _ => None,
            })?;
            bytes(a, |x| sorted.binary_search(&x).is_ok())
        }
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
            let sorted = natives(values, |v: &ScalarValue| match v {
                S::Binary(x) => Some(x.as_slice()),
                _ => None,
            })?;
            bytes(a, |x| sorted.binary_search(&x).is_ok())
        }
        _ => passes(a, values),
    }
}

/// The surviving fraction at or below which an `And`'s later conjuncts
/// run on a compacted copy of the surviving rows: below it, copying the
/// few rows of the columns they read costs less than evaluating a Bloom
/// probe or an `IN` on every row.
const SPARSE: f64 = 0.25;

/// One conjunct of a flattened `And`, with the `And` it was written in.
type Conjunct<'e> = (usize, &'e PushdownExpr);

/// Appends the conjuncts of `xs` (the `And` numbered `group`) to `out`,
/// nested `And`s flattened, each numbered from `next`. A runtime filter's
/// `In` is skipped only beside the exact membership of its own `And`
/// ([`eval_rest`]), so the numbers keep the nesting that matters.
fn conjuncts<'e>(
    xs: &'e [PushdownExpr],
    group: usize,
    next: &mut usize,
    out: &mut Vec<Conjunct<'e>>,
) {
    for x in xs {
        if let PushdownExpr::And(ys) = x {
            *next += 1;
            let inner = *next;
            conjuncts(ys, inner, next, out);
        } else {
            out.push((group, x));
        }
    }
}

/// Whether a conjunct is one vectorized pass over one column (or none):
/// these run first, over the whole batch.
fn cheap(e: &PushdownExpr) -> bool {
    matches!(
        e,
        PushdownExpr::Constant(_)
            | PushdownExpr::Cmp { .. }
            | PushdownExpr::IsNull(_)
            | PushdownExpr::IsNotNull(_)
    )
}

/// The `And` of `xs`: the cheap conjuncts over the whole batch, then the
/// others only on the rows those may keep, over a compacted copy of the
/// columns they read when few rows survive, and not at all when none does
/// (a runtime filter's Bloom probe and `In` after its range and the
/// query's own comparisons, which often drop almost every row).
///
/// Where the later conjuncts are not evaluated on a row the `And` is not
/// known to be false there (`may_false` unknown), which is a superset.
fn eval_and(xs: &[Conjunct<'_>], batch: &RecordBatch) -> Maybe {
    let n = batch.num_rows();
    let (first, rest): (Vec<_>, Vec<_>) = xs.iter().copied().partition(|(_, x)| cheap(x));
    let mut acc = Maybe::constant(true, n);
    for (_, x) in &first {
        acc = conj(acc, eval(x, batch));
    }
    if rest.is_empty() {
        return acc;
    }
    let Some(live) = acc.may_true.clone() else {
        return conj(acc, eval_rest(&rest, batch));
    };
    let survivors = live.count_set_bits();
    if survivors == 0 {
        return Maybe {
            may_true: Some(live),
            may_false: None,
        };
    }
    #[expect(clippy::cast_precision_loss, reason = "a ratio")]
    let sparse = survivors as f64 <= n as f64 * SPARSE;
    if !sparse {
        return conj(acc, eval_rest(&rest, batch));
    }
    let Some(compact) = compacted(batch, &rest, &live).filter(|c| c.num_rows() == survivors) else {
        return conj(acc, eval_rest(&rest, batch));
    };
    let m = eval_rest(&rest, &compact);
    Maybe {
        may_true: Some(match m.may_true {
            Some(t) if t.len() == survivors => scatter(&live, &t),
            _ => live,
        }),
        may_false: None,
    }
}

/// The conjunction of two results of an `And`'s conjuncts.
fn conj(acc: Maybe, m: Maybe) -> Maybe {
    Maybe {
        may_true: and(acc.may_true, m.may_true),
        may_false: or(acc.may_false, m.may_false),
    }
}

/// The rows of `live` of the columns `xs` read, or `None` if they cannot
/// be copied (the caller then evaluates `xs` on the whole batch).
fn compacted(
    batch: &RecordBatch,
    xs: &[Conjunct<'_>],
    live: &BooleanBuffer,
) -> Option<RecordBatch> {
    let schema = batch.schema();
    let mut read: Vec<usize> = Vec::new();
    for (_, x) in xs {
        for c in columns(x) {
            if let [name] = c.0.as_slice()
                && let Ok(i) = schema.index_of(name)
                && !read.contains(&i)
            {
                read.push(i);
            }
        }
    }
    read.sort_unstable();
    let projected = batch.project(&read).ok()?;
    let mask = BooleanArray::new(live.clone(), None);
    arrow_select::filter::filter_record_batch(&projected, &mask).ok()
}

/// `compact` (one bit per set bit of `live`, in order) spread back over
/// the rows of `live`; unset elsewhere.
fn scatter(live: &BooleanBuffer, compact: &BooleanBuffer) -> BooleanBuffer {
    let mut out = arrow_buffer::BooleanBufferBuilder::new(live.len());
    out.append_n(live.len(), false);
    for (row, keep) in live.set_indices().zip(compact.iter()) {
        if keep {
            out.set_bit(row, true);
        }
    }
    out.finish()
}

/// The `And` of conjuncts that are not [`cheap`]: memberships first, then
/// the rest. A runtime filter lowers to an `In` of its keys (for
/// statistics) beside a `Membership` over the same keys. Where that
/// membership is evaluated and exact, the `In` keeps no row fewer: it is
/// skipped (as unknown), not compared value by value.
fn eval_rest(xs: &[Conjunct<'_>], batch: &RecordBatch) -> Maybe {
    let mut covered: Vec<(usize, &ColumnPath)> = Vec::new();
    let mut acc = Maybe::constant(true, batch.num_rows());
    for (group, x) in xs {
        if let PushdownExpr::Membership { column: c, filter } = x {
            let m = eval(x, batch);
            if m.may_true.is_some() && filter.keys.exact_rows() {
                covered.push((*group, c));
            }
            acc = conj(acc, m);
        }
    }
    for (group, x) in xs {
        match x {
            PushdownExpr::Membership { .. } => {}
            PushdownExpr::In { column: c, .. } if covered.contains(&(*group, c)) => {
                acc = conj(acc, Maybe::default());
            }
            _ => acc = conj(acc, eval(x, batch)),
        }
    }
    acc
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
        PushdownExpr::And(xs) => {
            let mut flat = Vec::new();
            conjuncts(xs, 0, &mut 0, &mut flat);
            eval_and(&flat, batch)
        }
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
        } if binary(collation) => exact(c, &|a| in_list(a, values)),
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

/// Rows the filter saw and kept, since the decoder last read them; and,
/// for `EXPLAIN ANALYZE`, the rows the filter evaluated while decoding and
/// kept, over the decoder's life.
#[derive(Debug, Default)]
pub(crate) struct Counters {
    seen: AtomicU64,
    kept: AtomicU64,
    filtered: AtomicU64,
    passed: AtomicU64,
}

impl Counters {
    pub(crate) fn add(&self, seen: usize, kept: usize) {
        self.seen.fetch_add(seen as u64, Ordering::Relaxed);
        self.kept.fetch_add(kept as u64, Ordering::Relaxed);
    }

    /// As [`add`](Self::add), for rows the filter evaluated while decoding
    /// (not a measurement sample): also counted in [`totals`](Self::totals).
    pub(crate) fn add_filtered(&self, seen: usize, kept: usize) {
        self.add(seen, kept);
        self.filtered.fetch_add(seen as u64, Ordering::Relaxed);
        self.passed.fetch_add(kept as u64, Ordering::Relaxed);
    }

    /// `(evaluated, kept)` by the filter while decoding, never reset.
    pub(crate) fn totals(&self) -> (u64, u64) {
        (
            self.filtered.load(Ordering::Relaxed),
            self.passed.load(Ordering::Relaxed),
        )
    }

    /// `(seen, kept)` since the last [`take`](Self::take).
    pub(crate) fn pending(&self) -> (u64, u64) {
        (
            self.seen.load(Ordering::Relaxed),
            self.kept.load(Ordering::Relaxed),
        )
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
        self.counters.add_filtered(n, kept);
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

/// Whether a measured fraction is low enough for the filter to pay.
fn keeps_few(seen: u64, kept: u64) -> bool {
    #[expect(clippy::cast_precision_loss, reason = "a ratio")]
    let fraction = kept as f64 / seen as f64;
    fraction <= KEEP_MAX
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
    /// There is no predicate to filter with yet (no static one, no runtime
    /// filter): the decoder stays as it is until one is published.
    idle: bool,
    /// Row groups decoded with the filter, and without it.
    groups_on: u64,
    groups_off: u64,
    /// No row filter at all (`eligible` false, or policy off).
    forbidden: bool,
    /// The top-N bound's pass is in the current row filter.
    bound_on: bool,
    /// The bound's pass: apart from `counters`, whose fraction decides
    /// about the other pass.
    bound_counters: Arc<Counters>,
}

impl LateMaterialization {
    /// `eligible`: no column needs a calendar check, which must see every
    /// row of a decoded row group.
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
            idle: false,
            groups_on: 0,
            groups_off: 0,
            forbidden: !eligible || policy == crate::request::RowFilterPolicy::Off,
            bound_on: false,
            bound_counters: Arc::new(Counters::default()),
        }
    }

    /// Starts from what an earlier piece of the file measured (`prior`)
    /// instead of the first guess, as the next row group of that piece
    /// would. Only [`RowFilterPolicy::Auto`] adapts.
    ///
    /// [`RowFilterPolicy::Auto`]: crate::request::RowFilterPolicy::Auto
    #[must_use]
    pub(crate) fn resuming(mut self, prior: Option<bool>) -> Self {
        if let Some(on) =
            prior.filter(|_| !self.disabled && self.policy == crate::request::RowFilterPolicy::Auto)
        {
            self.on = on;
            self.wanted = on;
            self.measured = true;
        }
        self
    }

    /// What the row groups so far ask of the next one: `None` before any
    /// measurement, and for a filter that does not adapt.
    pub(crate) fn measurement(&self) -> Option<bool> {
        if self.disabled || self.idle || self.policy != crate::request::RowFilterPolicy::Auto {
            return None;
        }
        let (seen, kept) = self.counters.pending();
        if seen > 0 {
            Some(keeps_few(seen, kept))
        } else {
            self.measured.then_some(self.wanted)
        }
    }

    /// `(row groups decoded with the filter, without it, rows the filter
    /// evaluated, rows it kept)`, for `EXPLAIN ANALYZE`; rows count once per
    /// pass.
    pub(crate) fn report(&self) -> (u64, u64, u64, u64) {
        let (filtered, passed) = self.counters.totals();
        let (bound_filtered, bound_passed) = self.bound_counters.totals();
        (
            self.groups_on,
            self.groups_off,
            filtered + bound_filtered,
            passed + bound_passed,
        )
    }

    /// Whether the current row group is decoded with the filter.
    pub(crate) fn is_on(&self) -> bool {
        self.on
    }

    /// Whether the current row group decodes some columns first, by either
    /// pass, so the others may not be needed.
    pub(crate) fn filters_first(&self) -> bool {
        self.on || self.bound_on
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
    /// `bounds` are the top-N bounds, each asked about on its own every time.
    pub(crate) fn filter(
        &mut self,
        on: bool,
        dynamic: &[PushdownExpr],
        bounds: &[PushdownExpr],
        mut mask: impl FnMut(&[ColumnPath]) -> (ProjectionMask, Worth),
    ) -> RowFilter {
        let bound = self.bound_pass(bounds, &mut mask);
        self.bound_on = bound.is_some();
        let expr = self.expr(dynamic);
        // Without a predicate there is nothing to switch on: asking again
        // at every row-group boundary would rebuild the decoder for
        // nothing (`runtime_filters_changed` asks again once one exists).
        self.idle = expr.is_none();
        if self.idle {
            self.wanted = false;
        }
        let built = if on && !self.disabled {
            expr.and_then(|e| {
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
                })
            })
        } else {
            None
        };
        self.on = built.is_some();
        RowFilter::new(
            bound
                .into_iter()
                .chain(built)
                .map(|p| Box::new(p) as Box<dyn ArrowPredicate>)
                .collect(),
        )
    }

    /// The pass over `bounds`, if it may pay.
    fn bound_pass(
        &self,
        bounds: &[PushdownExpr],
        mask: &mut impl FnMut(&[ColumnPath]) -> (ProjectionMask, Worth),
    ) -> Option<ScanPredicate> {
        if self.forbidden || bounds.is_empty() {
            return None;
        }
        let e = match bounds {
            [one] => one.to_nnf(),
            _ => PushdownExpr::And(bounds.iter().map(PushdownExpr::to_nnf).collect()),
        };
        let cols = columns(&e);
        if cols.is_empty() {
            return None;
        }
        let (projection, worth) = mask(&cols);
        (worth == Worth::Yes).then(|| {
            ScanPredicate::new(
                Arc::new(e),
                projection,
                self.widen,
                Arc::clone(&self.bound_counters),
            )
        })
    }

    /// The runtime filters changed. If the scan had no predicate to filter
    /// with, the first one it gets starts the guess over (module docs).
    pub(crate) fn runtime_filters_changed(&mut self, dynamic: &[PushdownExpr]) {
        if self.idle && !self.disabled && self.expr(dynamic).is_some() {
            self.idle = false;
            self.wanted = true;
        }
    }

    /// At a row-group boundary: whether the next row group should use the
    /// filter, from what the last one measured.
    pub(crate) fn want(&mut self) -> bool {
        if self.disabled || self.idle {
            return false;
        }
        if self.policy == crate::request::RowFilterPolicy::On {
            return true;
        }
        let (seen, kept) = self.counters.take();
        if seen > 0 {
            self.measured = true;
            self.wanted = keeps_few(seen, kept);
        }
        self.wanted
    }

    /// Whether the next decoded batch is to be measured.
    pub(crate) fn sampling(&self) -> bool {
        self.sample
    }

    /// A row group starts: while off, its first batch is measured.
    pub(crate) fn start_row_group(&mut self) {
        if self.filters_first() {
            self.groups_on += 1;
        } else {
            self.groups_off += 1;
        }
        self.sample = !self.on && !self.disabled && !self.idle;
    }

    /// Measures `batch` (decoded without the filter) if a sample is due.
    /// When the projection omits a predicate column the batch cannot
    /// measure anything (`evaluate` keeps every row for a missing
    /// column): filtering is wanted for the next row group, which then
    /// decodes the predicate columns through its own projection.
    pub(crate) fn observe(&mut self, batch: &RecordBatch, dynamic: &[PushdownExpr]) {
        if !std::mem::take(&mut self.sample) {
            return;
        }
        let Some(e) = self.expr(dynamic) else {
            return;
        };
        let n = batch.num_rows();
        let w = widened(batch, self.widen);
        if w.num_rows() != n {
            return;
        }
        if columns(&e).iter().any(|c| column(&w, c).is_none()) {
            self.wanted = true;
            return;
        }
        self.counters.add(n, evaluate(&e, &w).count_set_bits());
    }
}

#[cfg(test)]
mod tests;
