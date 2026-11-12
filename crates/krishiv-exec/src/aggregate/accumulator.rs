// SPDX-License-Identifier: Apache-2.0
//! Per-group accumulators, one vector per aggregate (column-oriented), with
//! Spark 4.2 arithmetic: `sum` of integers in `bigint` (ANSI raises, legacy
//! wraps), decimals in `i128`, `avg` of decimals by Spark's division
//! (quotient at scale 39, then rounded half-up to the result scale).
//!
//! Decimal sums overflow as soon as a running sum leaves the sum type's
//! precision, even if later rows bring it back (`ARITHMETIC_OVERFLOW
//! "Overflow in sum of decimals"` under ANSI, NULL otherwise), grouped or
//! not. Whether a temporary overflow happens depends on the summation
//! order, as in Spark 4.2's grouped and partial buffers. Spark's global
//! aggregate (when no partial buffer overflows) and its grouped aggregate
//! over a local relation tolerate a temporary overflow; Krishiv does not (a
//! documented divergence, ADR 0023).
//!
//! **Per-batch proof (P2-T160):** each decimal `sum`/`avg` keeps an upper
//! bound of every group's `|sum|`; a batch whose `Σ |weight·value|` keeps
//! that bound below the sum type's `10^p` cannot overflow any group in any
//! order, so it folds without per-row checks. Other batches take the
//! per-row checked path. Decimal `avg` with a result scale of at most 19
//! divides once instead of digit by digit (see [`avg_fast`]).

use std::borrow::Cow;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, DurationMicrosecondType, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, IntervalYearMonthType, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Decimal128Array, Float32Array,
    Float64Array, Int64Array, ListArray, PrimitiveArray, StringArray, StringViewArray, UInt32Array,
};
use arrow_buffer::{NullBuffer, i256};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};
use krishiv_kernels::KernelCtx;
use krishiv_kernels::compare::CompareKind;
use krishiv_kernels::decimal::DecimalNative;
use krishiv_types::batch::Weights;

use super::collect::{CollectAcc, VectorAcc};
use super::hll::{self, HllAcc};
use super::listagg::ListAggAcc;
use super::mode::ModeAcc;
use super::moments::Moments;
use super::quantile::{self, ExactAcc, GkAcc};
use super::{
    AggInput, AggregateExpr, AggregateFunction, decimal_sum_overflow, interval_sum_overflow,
    is_ansi_interval, long_overflow, out_of_range, unsupported,
};
use krishiv_common::error::KrishivError;

pub(crate) use super::hll::hll_supported;
pub(crate) use super::quantile::quantile_supported;
use crate::error::{Result, internal};
use crate::planner::eval::Value;
use crate::planner::eval::logic::cmp;
use krishiv_common::error::NegativeValuesInFrequencyExpression;

/// Whether `min`/`max` take arguments of type `t`.
pub(crate) fn min_max_supported(t: &DataType) -> bool {
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

/// Whether `min`/`max` take structs and arrays of Arrow type `t`: the same comparator the
/// comparison operators use accepts them ([`krishiv_kernels::r#struct::comparable`]).
pub(crate) fn struct_min_max_supported(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Struct(_) | DataType::List(_) | DataType::LargeList(_)
    ) && krishiv_kernels::r#struct::comparable(t)
}

/// `min`/`max` values, by physical kind.
#[derive(Debug)]
enum Values {
    /// Integral, boolean, decimal, date and timestamp values.
    Int(Vec<i128>),
    Float(Vec<f64>),
    Bytes(Vec<Box<[u8]>>),
}

impl Values {
    /// No groups yet, of the physical kind of Spark type `t`.
    fn empty(t: &DataType) -> Self {
        match t {
            DataType::Float32 | DataType::Float64 => Self::Float(Vec::new()),
            DataType::Utf8 | DataType::Utf8View | DataType::Binary | DataType::BinaryView => {
                Self::Bytes(Vec::new())
            }
            _ => Self::Int(Vec::new()),
        }
    }

    fn resize(&mut self, n: usize) {
        match self {
            Self::Int(v) => v.resize(n, 0),
            Self::Float(v) => v.resize(n, 0.0),
            Self::Bytes(v) => v.resize(n, Box::default()),
        }
    }

    fn reserve(&mut self, n: usize) {
        match self {
            Self::Int(v) => v.reserve(n),
            Self::Float(v) => v.reserve(n),
            Self::Bytes(v) => v.reserve(n),
        }
    }

    fn vec(&self) -> (usize, usize, usize) {
        fn v<T>(x: &Vec<T>) -> (usize, usize, usize) {
            (x.len(), x.capacity(), size_of::<T>())
        }
        match self {
            Self::Int(x) => v(x),
            Self::Float(x) => v(x),
            Self::Bytes(x) => v(x),
        }
    }

    /// Stores row `i` of `src` as group `g`'s value, keeping `heap` (the
    /// bytes of `Bytes` values) current.
    fn put(&mut self, g: usize, src: &Col<'_>, i: usize, heap: &mut usize) -> Result<()> {
        match (self, src) {
            (Self::Int(v), Col::Int(c)) => v[g] = c[i],
            (Self::Float(v), Col::Float(c)) => v[g] = c[i],
            (Self::Bytes(v), Col::Bytes(c)) => {
                *heap = heap.saturating_sub(v[g].len()) + c[i].len();
                v[g] = c[i].into();
            }
            _ => {
                return Err(internal(
                    "a value of another kind than its state".to_owned(),
                ));
            }
        }
        Ok(())
    }

    /// Clears group `g`'s value (a NULL win), keeping `heap` current.
    fn clear(&mut self, g: usize, heap: &mut usize) {
        match self {
            Self::Int(v) => v[g] = 0,
            Self::Float(v) => v[g] = 0.0,
            Self::Bytes(v) => {
                *heap = heap.saturating_sub(v[g].len());
                v[g] = Box::default();
            }
        }
    }

    /// Copies group `from`'s value to group `to` (a running frame's
    /// snapshot). Byte values are never snapshotted: the copies would not
    /// be accounted for.
    fn copy_slot(&mut self, from: usize, to: usize) -> Result<()> {
        match self {
            Self::Int(v) => v[to] = v[from],
            Self::Float(v) => v[to] = v[from],
            Self::Bytes(_) => return Err(internal("a snapshot of byte values".to_owned())),
        }
        Ok(())
    }
}

/// A column read as one physical kind (the kinds of [`Values`]).
enum Col<'a> {
    Int(Cow<'a, [i128]>),
    Float(Cow<'a, [f64]>),
    Bytes(Vec<&'a [u8]>),
}

impl<'a> Col<'a> {
    fn of(a: &'a ArrayRef) -> Result<Self> {
        Ok(match a.data_type() {
            DataType::Float32 | DataType::Float64 => Self::Float(floats(a)?),
            DataType::Utf8 | DataType::Utf8View | DataType::Binary | DataType::BinaryView => {
                Self::Bytes(byte_values(a)?)
            }
            _ => Self::Int(int_like(a)?),
        })
    }
}

#[derive(Debug)]
enum Kind {
    Count(Vec<i64>),
    /// A session aggregate UDF's accumulator (ADR 0045).
    Udf(Box<super::udf::UdfAcc>),
    SumInt {
        sum: Vec<i64>,
        seen: Vec<bool>,
    },
    SumFloat {
        sum: Vec<f64>,
        seen: Vec<bool>,
    },
    /// `state`: 0 no value, 1 a value, 2 overflowed.
    SumDec {
        sum: Vec<i128>,
        state: Vec<u8>,
    },
    AvgFloat {
        sum: Vec<f64>,
        count: Vec<i64>,
    },
    AvgDec {
        sum: Vec<i128>,
        count: Vec<i64>,
        overflow: Vec<bool>,
    },
    MinMax {
        max: bool,
        values: Values,
        set: Vec<bool>,
        /// Heap bytes of `Values::Bytes`.
        heap: usize,
    },
    /// `min`/`max` over structs: one owned winning row per group (`None`:
    /// no row yet). Rows compare with the same field-wise comparator the
    /// comparison operators use (NULL fields sort smallest and equal one
    /// another); a NULL struct row is skipped like any NULL input.
    MinMaxStruct {
        max: bool,
        /// The winning row of each group, as an owned one-row struct.
        values: Vec<Option<ArrayRef>>,
        /// Owned bytes of each group's winning row.
        bytes: Vec<usize>,
        set: Vec<bool>,
        /// Heap bytes of the winning rows.
        heap: usize,
        kernels: KernelCtx,
    },
    /// `first` / `last`: `set` = a row was taken, `null` = its value is NULL.
    Pick {
        last: bool,
        ignore_nulls: bool,
        values: Values,
        set: Vec<bool>,
        null: Vec<bool>,
        heap: usize,
    },
    /// `max_by` / `min_by`: `set` = a row with a non-NULL key was taken
    /// (`keys` holds its key), `null` = its value is NULL. `heap` counts
    /// the bytes of both `Bytes` vectors.
    By {
        max: bool,
        /// A tie keeps the held row (`min`/`max` of a collated string) instead of taking the new.
        first_wins: bool,
        values: Values,
        keys: Values,
        set: Vec<bool>,
        null: Vec<bool>,
        heap: usize,
    },
    /// The moment aggregates: Spark's buffers of doubles (see `moments`).
    Moments(Moments),
    /// `collect_list` / `collect_set`: appended value chunks per group, in
    /// fold order (see `collect`).
    Collect(CollectAcc),
    /// `vector_sum` / `vector_avg`: running sums with the non-NULL vector
    /// count per group (see `collect`).
    Vector(VectorAcc),
    /// `try_sum` of integers that gives NULL instead of an overflow
    /// error: `state`: 0 no value, 1 a value, 2 overflowed (sticky).
    TrySumInt {
        sum: Vec<i64>,
        state: Vec<u8>,
    },
    /// `sum`, `try_sum`, `avg`, `try_avg` of an ANSI interval, in months or
    /// microseconds. Every addition checks the family's range, ANSI on or
    /// off: `sum` and `avg` raise, the `try_` forms keep `over` (sticky).
    SumInterval {
        sum: Vec<i64>,
        count: Vec<i64>,
        over: Vec<bool>,
        months: bool,
    },
    /// `bit_and` / `bit_or` / `bit_xor`: the running value with whether
    /// a non-NULL input arrived.
    Bit {
        op: BitOp,
        values: Values,
        set: Vec<bool>,
    },
    /// `mode` and its ordered forms: distinct values in first-seen order
    /// with their counts and samples (see `mode`).
    Mode(ModeAcc),
    /// `listagg`: the non-NULL values with their ordering keys (see
    /// `listagg`).
    ListAgg(ListAggAcc),
    /// The exact percentiles: value-count maps (see `quantile`).
    Exact(ExactAcc),
    /// `percentile_approx`: Greenwald-Khanna summaries (see `quantile`).
    Approx(GkAcc),
    /// `approx_count_distinct`: HyperLogLog++ registers (see `hll`).
    Hll(HllAcc),
}

/// A bitwise aggregate operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BitOp {
    And,
    Or,
    Xor,
}

/// `a` combined with `b` by `op`.
fn bit(op: BitOp, a: i128, b: i128) -> i128 {
    match op {
        BitOp::And => a & b,
        BitOp::Or => a | b,
        BitOp::Xor => a ^ b,
    }
}

/// One aggregate's state for every group of a table.
#[derive(Debug)]
pub(crate) struct Accumulator {
    kind: Kind,
    expr: AggregateExpr,
    result: DataType,
    ansi: bool,
    /// The bound of a running decimal sum's magnitude (`10^p` of the sum
    /// type); `None` for other aggregates.
    sum_bound: Option<u128>,
    /// An upper bound of `|sum|` over every group of a decimal `sum`/`avg`,
    /// refreshed after each fold ([`refresh_max`]).
    max_sum: u128,
    /// Whether decimal folds may take the per-batch proof path (tests turn
    /// it off to compare against the per-row path).
    proofs: bool,
}

/// The snapshots of a running fold ([`Accumulator::update_running`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Running<'a> {
    /// The group holding each row's partition state.
    pub(crate) parts: &'a [u32],
    /// The group that takes the state after each row, or [`Running::NONE`].
    pub(crate) snapshots: &'a [u32],
}

impl Running<'_> {
    /// No snapshot after this row.
    pub(crate) const NONE: u32 = u32::MAX;
}

impl Accumulator {
    /// An accumulator for `expr` (decimal sums are bounded while running,
    /// see the module docs). `try_sum` and `try_avg` always fold as if
    /// ANSI were off (an overflow gives NULL instead of an error).
    pub(crate) fn new(expr: &AggregateExpr, ansi: bool) -> Result<Self> {
        use AggregateFunction as F;
        let result = expr.result_type()?;
        let ansi = if matches!(expr.function, F::TrySum | F::TryAvg) {
            false
        } else {
            ansi
        };
        let kind = match (&expr.function, &result) {
            (F::Count, _) => Kind::Count(Vec::new()),
            (F::Sum | F::TrySum | F::Avg | F::TryAvg, t) if is_ansi_interval(t) => {
                Kind::SumInterval {
                    sum: Vec::new(),
                    count: Vec::new(),
                    over: Vec::new(),
                    months: matches!(t, DataType::Interval(_)),
                }
            }
            (F::Sum | F::TrySum, DataType::Int64) => {
                if matches!(expr.function, F::TrySum) {
                    Kind::TrySumInt {
                        sum: Vec::new(),
                        state: Vec::new(),
                    }
                } else {
                    Kind::SumInt {
                        sum: Vec::new(),
                        seen: Vec::new(),
                    }
                }
            }
            (F::Sum | F::TrySum, DataType::Float64) => Kind::SumFloat {
                sum: Vec::new(),
                seen: Vec::new(),
            },
            (F::Sum | F::TrySum, _) => Kind::SumDec {
                sum: Vec::new(),
                state: Vec::new(),
            },
            (F::Avg | F::TryAvg, DataType::Float64) => Kind::AvgFloat {
                sum: Vec::new(),
                count: Vec::new(),
            },
            (F::Avg | F::TryAvg, _) => Kind::AvgDec {
                sum: Vec::new(),
                count: Vec::new(),
                overflow: Vec::new(),
            },
            (
                f @ (F::Min | F::Max),
                DataType::Struct(_) | DataType::List(_) | DataType::LargeList(_),
            ) => Kind::MinMaxStruct {
                max: *f == F::Max,
                values: Vec::new(),
                bytes: Vec::new(),
                set: Vec::new(),
                heap: 0,
                kernels: KernelCtx::new(),
            },
            (f @ (F::Min | F::Max), t) => Kind::MinMax {
                max: *f == F::Max,
                values: Values::empty(t),
                set: Vec::new(),
                heap: 0,
            },
            (F::MaxBy | F::MinBy | F::MaxByKeyFirst | F::MinByKeyFirst, _) => {
                let (v, k) = expr
                    .by_types()
                    .ok_or_else(|| internal("max_by without value and key types".to_owned()))?;
                Kind::By {
                    max: matches!(expr.function, F::MaxBy | F::MaxByKeyFirst),
                    first_wins: matches!(expr.function, F::MaxByKeyFirst | F::MinByKeyFirst),
                    values: Values::empty(v),
                    keys: Values::empty(k),
                    set: Vec::new(),
                    null: Vec::new(),
                    heap: 0,
                }
            }
            (
                f @ (F::First { ignore_nulls }
                | F::Last { ignore_nulls }
                | F::AnyValue { ignore_nulls }),
                t,
            ) => Kind::Pick {
                last: matches!(f, F::Last { .. }),
                ignore_nulls: *ignore_nulls,
                values: Values::empty(t),
                set: Vec::new(),
                null: Vec::new(),
                heap: 0,
            },
            (
                f @ (F::StddevSamp
                | F::StddevPop
                | F::VarSamp
                | F::VarPop
                | F::Skewness
                | F::Kurtosis
                | F::Corr
                | F::CovarSamp
                | F::CovarPop
                | F::RegrCount
                | F::RegrAvgx
                | F::RegrAvgy
                | F::RegrSxx
                | F::RegrSyy
                | F::RegrSxy
                | F::RegrSlope
                | F::RegrIntercept
                | F::RegrR2),
                _,
            ) => Kind::Moments(Moments::new(f.clone(), ansi)?),
            (F::Percentile { .. } | F::PercentileDisc { .. }, _) => {
                Kind::Exact(ExactAcc::default())
            }
            (F::PercentileApprox { .. }, _) => Kind::Approx(GkAcc::default()),
            (F::ApproxCountDistinct { p }, _) => Kind::Hll(HllAcc::new(*p)),
            (F::CollectList, _) => Kind::Collect(CollectAcc::new(expr.arg_type.clone(), false)?),
            // `CollectAcc::new` refuses the element types the shared key
            // encoding does not order, as `result_type` does.
            (F::CollectSet, _) => Kind::Collect(CollectAcc::new(expr.arg_type.clone(), true)?),
            (F::VectorSum, _) => Kind::Vector(VectorAcc::new(false)),
            (F::VectorAvg, _) => Kind::Vector(VectorAcc::new(true)),
            (f @ (F::BitAnd | F::BitOr | F::BitXor), t) => Kind::Bit {
                op: if *f == F::BitAnd {
                    BitOp::And
                } else if *f == F::BitOr {
                    BitOp::Or
                } else {
                    BitOp::Xor
                },
                values: Values::empty(t),
                set: Vec::new(),
            },
            // Built by `accumulator_of` with the session's library; a path
            // that reaches here has none.
            (F::Udf(_), _) => {
                return Err(unsupported(
                    "an aggregate UDF where the session's UDFs are not available".to_owned(),
                ));
            }
            (f @ (F::Mode | F::ModeMin | F::ModeMax), _) => {
                Kind::Mode(ModeAcc::new(expr.arg_type.clone(), f)?)
            }
            (F::ListAgg { delimiter, orders }, _) => {
                let (value, keys) = expr.listagg_types().ok_or_else(|| {
                    internal(format!(
                        "listagg of {} without value and keys",
                        expr.arg_type
                    ))
                })?;
                if keys.len() != orders.len() {
                    return Err(internal(format!(
                        "listagg has {} ordering keys for {} orders",
                        keys.len(),
                        orders.len()
                    )));
                }
                Kind::ListAgg(ListAggAcc::new(
                    delimiter.clone(),
                    orders.clone(),
                    value.clone(),
                    keys.into_iter().cloned().collect(),
                )?)
            }
        };
        let sum_bound = match (&expr.function, expr.avg_sum_type()) {
            (
                F::Sum | F::Avg | F::TrySum | F::TryAvg,
                DataType::Decimal64(p, _) | DataType::Decimal128(p, _),
            ) => Some(pow10(p).unsigned_abs()),
            _ => None,
        };
        Ok(Self {
            kind,
            expr: expr.clone(),
            result,
            ansi,
            sum_bound,
            max_sum: 0,
            proofs: true,
        })
    }

    /// The accumulator of an aggregate UDF call `expr` over `acc`.
    pub(crate) fn udf(expr: &AggregateExpr, acc: super::udf::UdfAcc) -> Result<Self> {
        Ok(Self {
            kind: Kind::Udf(Box::new(acc)),
            expr: expr.clone(),
            result: expr.result_type()?,
            ansi: false,
            sum_bound: None,
            max_sum: 0,
            proofs: false,
        })
    }

    /// Whether the sample moment aggregates give NaN instead of NULL where
    /// Spark divides by zero (`spark.sql.legacy.statisticalAggregate`); the
    /// other aggregates ignore it.
    #[must_use]
    pub(crate) fn legacy_statistical(mut self, on: bool) -> Self {
        if let Kind::Moments(m) = &mut self.kind {
            m.legacy(on);
        }
        self
    }

    /// Bytes one group's state takes (heap bytes of string values aside).
    pub(crate) fn group_bytes(&self) -> usize {
        self.vecs().iter().map(|&(_, _, e)| e).sum()
    }

    /// Visits every per-group vector as (length, capacity, element size).
    fn vecs(&self) -> Vec<(usize, usize, usize)> {
        fn v<T>(x: &Vec<T>) -> (usize, usize, usize) {
            (x.len(), x.capacity(), size_of::<T>())
        }
        match &self.kind {
            Kind::Count(c) => vec![v(c)],
            Kind::SumInt { sum, seen } => vec![v(sum), v(seen)],
            Kind::SumFloat { sum, seen } => vec![v(sum), v(seen)],
            Kind::SumDec { sum, state } => vec![v(sum), v(state)],
            Kind::AvgFloat { sum, count } => vec![v(sum), v(count)],
            Kind::AvgDec {
                sum,
                count,
                overflow,
            } => vec![v(sum), v(count), v(overflow)],
            Kind::MinMax { values, set, .. } | Kind::Bit { values, set, .. } => {
                vec![values.vec(), v(set)]
            }
            Kind::MinMaxStruct {
                values, bytes, set, ..
            } => {
                vec![v(values), v(bytes), v(set)]
            }
            Kind::Pick {
                values, set, null, ..
            } => vec![values.vec(), v(set), v(null)],
            Kind::By {
                values,
                keys,
                set,
                null,
                ..
            } => vec![values.vec(), keys.vec(), v(set), v(null)],
            Kind::Moments(m) => m.vecs(),
            Kind::Collect(c) => c.vecs(),
            Kind::Vector(v) => v.vecs(),
            Kind::TrySumInt { sum, state } => vec![v(sum), v(state)],
            Kind::SumInterval {
                sum, count, over, ..
            } => vec![v(sum), v(count), v(over)],
            Kind::Mode(m) => m.vecs(),
            Kind::ListAgg(l) => l.vecs(),
            // The quantile and HLL states are maps and buffers, not
            // plain vectors; `size` counts them exactly.
            Kind::Exact(_) | Kind::Approx(_) | Kind::Hll(_) | Kind::Udf(_) => Vec::new(),
        }
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        match &self.kind {
            Kind::Collect(c) => c.size(),
            Kind::Vector(v) => v.size(),
            Kind::Mode(m) => m.size(),
            Kind::ListAgg(l) => l.size(),
            Kind::Exact(e) => e.size(),
            Kind::Approx(g) => g.size(),
            Kind::Hll(h) => h.size(),
            Kind::Udf(u) => u.size(),
            _ => {
                let heap = match &self.kind {
                    Kind::MinMax { heap, .. }
                    | Kind::MinMaxStruct { heap, .. }
                    | Kind::Pick { heap, .. }
                    | Kind::By { heap, .. } => *heap,
                    _ => 0,
                };
                heap + self.vecs().iter().map(|(_, c, e)| c * e).sum::<usize>()
            }
        }
    }

    /// [`size`](Self::size) after [`resize`](Self::resize) to `new` more
    /// groups and an update with `input` (string values it may keep).
    /// `arg` is the value column the update reads (raw or state); `key` the
    /// ordering key column of `max_by`/`min_by`.
    pub(crate) fn projected(
        &self,
        new: usize,
        arg: Option<&ArrayRef>,
        key: Option<&ArrayRef>,
    ) -> usize {
        // The collect, vector, mode and listagg accumulators account
        // their variable-width growth themselves (see `collect`, `mode`
        // and `listagg`).
        if let Kind::Collect(c) = &self.kind {
            return c.projected(new, arg);
        }
        if let Kind::Vector(v) = &self.kind {
            return v.projected(new, arg);
        }
        if let Kind::Mode(m) = &self.kind {
            return m.projected(new, arg);
        }
        if let Kind::ListAgg(l) = &self.kind {
            return l.projected(new, arg);
        }
        // A UDF's growth is unknown until it folds: its last fold's.
        // (The row tables count the difference an argument makes.)
        if let Kind::Udf(u) = &self.kind {
            return if arg.is_some() {
                u.projected()
            } else {
                u.size()
            };
        }
        let vecs: usize = self
            .vecs()
            .iter()
            .map(|&(len, cap, e)| {
                let need = len.saturating_add(new);
                let c = if need <= cap {
                    cap
                } else {
                    need.max(cap * 2).max(if e == 1 { 8 } else { 4 })
                };
                c * e
            })
            .sum();
        let heap = match &self.kind {
            Kind::MinMax {
                values: Values::Bytes(_),
                heap,
                ..
            }
            | Kind::Pick {
                values: Values::Bytes(_),
                heap,
                ..
            }
            | Kind::MinMaxStruct { heap, .. } => heap + arg.map_or(0, value_bytes),
            Kind::By {
                values, keys, heap, ..
            } => {
                let bytes = |v: &Values, a: Option<&ArrayRef>| match v {
                    Values::Bytes(_) => a.map_or(0, value_bytes),
                    _ => 0,
                };
                heap + bytes(values, arg) + bytes(keys, key)
            }
            Kind::MinMax { heap, .. } | Kind::Pick { heap, .. } => *heap,
            Kind::Exact(e) => e.projected(new, arg.map_or(0, arrow_array::Array::len)),
            Kind::Approx(g) => g.projected(new, arg.map_or(0, arrow_array::Array::len)),
            Kind::Hll(h) => h.projected(new),
            _ => 0,
        };
        vecs + heap
    }

    /// Adds groups up to `n` with empty states.
    pub(crate) fn resize(&mut self, n: usize) {
        match &mut self.kind {
            Kind::Count(c) => c.resize(n, 0),
            Kind::SumInt { sum, seen } => {
                sum.resize(n, 0);
                seen.resize(n, false);
            }
            Kind::SumFloat { sum, seen } => {
                sum.resize(n, 0.0);
                seen.resize(n, false);
            }
            Kind::SumDec { sum, state } => {
                sum.resize(n, 0);
                state.resize(n, 0);
            }
            Kind::AvgFloat { sum, count } => {
                sum.resize(n, 0.0);
                count.resize(n, 0);
            }
            Kind::AvgDec {
                sum,
                count,
                overflow,
            } => {
                sum.resize(n, 0);
                count.resize(n, 0);
                overflow.resize(n, false);
            }
            Kind::MinMax { values, set, .. } | Kind::Bit { values, set, .. } => {
                set.resize(n, false);
                values.resize(n);
            }
            Kind::MinMaxStruct {
                values, bytes, set, ..
            } => {
                values.resize(n, None);
                bytes.resize(n, 0);
                set.resize(n, false);
            }
            Kind::Pick {
                values, set, null, ..
            } => {
                set.resize(n, false);
                null.resize(n, false);
                values.resize(n);
            }
            Kind::Moments(m) => m.resize(n),
            Kind::Collect(c) => c.resize(n),
            Kind::Vector(v) => v.resize(n),
            Kind::TrySumInt { sum, state } => {
                sum.resize(n, 0);
                state.resize(n, 0);
            }
            Kind::SumInterval {
                sum, count, over, ..
            } => {
                sum.resize(n, 0);
                count.resize(n, 0);
                over.resize(n, false);
            }
            Kind::Mode(m) => m.resize(n),
            Kind::ListAgg(l) => l.resize(n),
            Kind::Exact(e) => e.resize(n),
            Kind::Approx(g) => g.resize(n),
            Kind::Hll(h) => h.resize(n),
            Kind::Udf(u) => u.resize(n),
            Kind::By {
                values,
                keys,
                set,
                null,
                ..
            } => {
                set.resize(n, false);
                null.resize(n, false);
                values.resize(n);
                keys.resize(n);
            }
        }
    }

    /// Reserves room for `new` more groups (as [`projected`](Self::projected)).
    pub(crate) fn reserve(&mut self, new: usize) {
        match &mut self.kind {
            Kind::Count(c) => c.reserve(new),
            Kind::SumInt { sum, seen } => {
                sum.reserve(new);
                seen.reserve(new);
            }
            Kind::SumFloat { sum, seen } => {
                sum.reserve(new);
                seen.reserve(new);
            }
            Kind::SumDec { sum, state } => {
                sum.reserve(new);
                state.reserve(new);
            }
            Kind::AvgFloat { sum, count } => {
                sum.reserve(new);
                count.reserve(new);
            }
            Kind::AvgDec {
                sum,
                count,
                overflow,
            } => {
                sum.reserve(new);
                count.reserve(new);
                overflow.reserve(new);
            }
            Kind::MinMax { values, set, .. } | Kind::Bit { values, set, .. } => {
                set.reserve(new);
                values.reserve(new);
            }
            Kind::MinMaxStruct {
                values, bytes, set, ..
            } => {
                values.reserve(new);
                bytes.reserve(new);
                set.reserve(new);
            }
            Kind::Pick {
                values, set, null, ..
            } => {
                set.reserve(new);
                null.reserve(new);
                values.reserve(new);
            }
            Kind::Moments(m) => m.reserve(new),
            Kind::Collect(c) => c.reserve(new),
            Kind::Vector(v) => v.reserve(new),
            Kind::TrySumInt { sum, state } => {
                sum.reserve(new);
                state.reserve(new);
            }
            Kind::SumInterval {
                sum, count, over, ..
            } => {
                sum.reserve(new);
                count.reserve(new);
                over.reserve(new);
            }
            Kind::Mode(m) => m.reserve(new),
            Kind::ListAgg(l) => l.reserve(new),
            Kind::Exact(e) => e.reserve(new),
            Kind::Approx(g) => g.reserve(new),
            Kind::Hll(h) => h.reserve(new),
            Kind::Udf(_) => {}
            Kind::By {
                values,
                keys,
                set,
                null,
                ..
            } => {
                set.reserve(new);
                null.reserve(new);
                values.reserve(new);
                keys.reserve(new);
            }
        }
    }

    /// Folds rows of `cols` into `groups` (one group id per row): raw
    /// arguments, or partial states, per the aggregate's input. Raw
    /// updates skip the rows the mask column drops; merges never see one
    /// (partial states hold the rows their mask passed).
    pub(crate) fn update(&mut self, groups: &[u32], cols: &[ArrayRef], w: &Weights) -> Result<()> {
        let args: Vec<&ArrayRef> = self
            .expr
            .args
            .iter()
            .map(|&i| arg(cols, i))
            .collect::<Result<_>>()?;
        let state = self.expr.input == AggInput::State;
        let mask = if state { None } else { self.mask(cols)? };
        self.fold(groups, &args, state, w, None, mask)
    }

    /// The boolean mask column of a raw update, if the aggregate has one.
    fn mask<'a>(&self, cols: &'a [ArrayRef]) -> Result<Option<&'a BooleanArray>> {
        let Some(m) = self.expr.mask else {
            return Ok(None);
        };
        let col = cols
            .get(m)
            .ok_or_else(|| internal(format!("mask column {m} out of range")))?;
        col.as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| internal(format!("mask column {m} is not boolean")))
            .map(Some)
    }

    /// [`update`](Self::update) over raw arguments, keeping the state after
    /// chosen rows: after row `i` is folded, if `run.snapshots[i]` is a
    /// group, `run.parts[i]`'s state is copied into it. A window's running
    /// frame takes the state after each row or peer group this way: one
    /// pass over the rows, folded one by one and in order as Spark's frames
    /// fold them, with the argument columns, validity and conversions read
    /// once.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for the aggregates a window evaluates elsewhere
    /// (decimal `sum`/`avg`, string `min`/`max`/`first`/`last`), partial
    /// states, or slices of other lengths than the rows.
    pub(crate) fn update_running(
        &mut self,
        groups: &[u32],
        cols: &[ArrayRef],
        run: &Running<'_>,
    ) -> Result<()> {
        if run.parts.len() != groups.len() || run.snapshots.len() != groups.len() {
            return Err(internal(
                "a running fold over slices of different lengths".to_owned(),
            ));
        }
        let args: Vec<&ArrayRef> = self
            .expr
            .args
            .iter()
            .map(|&i| arg(cols, i))
            .collect::<Result<_>>()?;
        if self.expr.input == AggInput::State {
            return Err(internal("a running fold over partial states".to_owned()));
        }
        if matches!(self.kind, Kind::Exact(_)) {
            return self.update_running_exact(groups, cols, run);
        }
        self.fold(groups, &args, false, &Weights::Const(1), Some(run), None)
    }

    /// The flags of an exact percentile call: its percentages, `reverse`,
    /// whether it interpolates (`cont`), the legacy discrete rule and whether
    /// it returns an array.
    fn exact_params(&self) -> Result<(Vec<f64>, bool, bool, bool, bool)> {
        use AggregateFunction as F;
        match &self.expr.function {
            F::Percentile {
                percentages,
                array,
                reverse,
            } => Ok((percentages.clone(), *reverse, true, false, *array)),
            F::PercentileDisc {
                percentages,
                array,
                reverse,
                legacy,
            } => Ok((percentages.clone(), *reverse, false, *legacy, *array)),
            _ => Err(internal(
                "an exact percentile over another aggregate".to_owned(),
            )),
        }
    }

    /// [`update_running`](Self::update_running) for the exact percentiles,
    /// whose state is a value map: the rows are folded in runs ending at a
    /// snapshot row. A window evaluates them
    /// over a whole partition or a running frame (`percentile` accepts an
    /// ordered window); a snapshot freezes the answer at its row instead of
    /// copying the value map.
    fn update_running_exact(
        &mut self,
        groups: &[u32],
        cols: &[ArrayRef],
        run: &Running<'_>,
    ) -> Result<()> {
        let mut lo = 0;
        while lo < groups.len() {
            let hi = (lo..groups.len())
                .find(|&i| run.snapshots[i] != Running::NONE)
                .map_or(groups.len(), |i| i + 1);
            let part: Vec<ArrayRef> = cols.iter().map(|c| c.slice(lo, hi - lo)).collect();
            let args: Vec<&ArrayRef> = self
                .expr
                .args
                .iter()
                .map(|&i| arg(&part, i))
                .collect::<Result<_>>()?;
            self.fold(
                &groups[lo..hi],
                &args,
                false,
                &Weights::Const(1),
                None,
                None,
            )?;
            let last = hi - 1;
            if run.snapshots[last] != Running::NONE {
                let (percentages, reverse, cont, legacy, _) = self.exact_params()?;
                if let Kind::Exact(acc) = &mut self.kind {
                    acc.freeze(
                        run.parts[last] as usize,
                        run.snapshots[last] as usize,
                        &percentages,
                        reverse,
                        cont,
                        legacy,
                    );
                }
            }
            lo = hi;
        }
        Ok(())
    }

    /// Merges partial states (`states`: this aggregate's state columns).
    /// A mask never reaches a merge: it filtered the rows the states hold.
    pub(crate) fn merge(&mut self, groups: &[u32], states: &[ArrayRef]) -> Result<()> {
        let args: Vec<&ArrayRef> = states.iter().collect();
        self.fold(groups, &args, true, &Weights::Const(1), None, None)
    }

    /// The raw argument columns' indices.
    pub(crate) fn args(&self) -> &[usize] {
        &self.expr.args
    }

    /// Number of state columns.
    pub(crate) fn state_len(&self) -> usize {
        match self.kind {
            Kind::Count(_)
            | Kind::SumInt { .. }
            | Kind::SumFloat { .. }
            | Kind::MinMax { .. }
            | Kind::Bit { .. }
            | Kind::MinMaxStruct { .. }
            | Kind::Collect(_)
            | Kind::Approx(_)
            | Kind::Hll(_) => 1,
            Kind::TrySumInt { .. }
            | Kind::SumInterval { .. }
            | Kind::SumDec { .. }
            | Kind::AvgFloat { .. }
            | Kind::AvgDec { .. }
            | Kind::Pick { .. }
            | Kind::By { .. }
            | Kind::Vector(_)
            | Kind::Mode(_)
            | Kind::Exact(_) => 2,
            Kind::ListAgg(_) => self.expr.listagg_types().map_or(1, |(_, k)| k.len() + 1),
            Kind::Moments(ref m) => m.width(),
            Kind::Udf(ref u) => u.state_len(),
        }
    }
}

mod small_fold;

#[cfg(test)]
#[cfg(not(loom))]
mod differential;
pub(crate) mod paired;

/// The per-batch proofs (design P2-T160) against the per-row path.
#[cfg(test)]
#[cfg(not(loom))]
#[expect(
    clippy::disallowed_methods,
    reason = "test-only golden-file read; never runs on a compute worker"
)]
mod proof_tests;

#[cfg(test)]
#[cfg(not(loom))]
mod tests;

/// The batch fold of a struct `MIN`/`MAX` against feeding the same rows one
/// at a time (the row-at-a-time fold the batch path replaces).
#[cfg(test)]
#[cfg(not(loom))]
mod struct_extremum_tests;

mod results;
use results::*;

mod output;

mod fold;

mod helpers;
pub(crate) use helpers::*;
mod by_key;
