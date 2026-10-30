// SPDX-License-Identifier: Apache-2.0
//! Hash aggregation (ADR 0011): `count`, `sum`, `avg`, `min`, `max`,
//! `first`, `last`, `any_value`, `max_by`, `min_by`, the moment and
//! regression aggregates (`stddev`, `variance`, `skewness`, `kurtosis`,
//! `covar_*`, `corr`, `regr_*`), `bit_and`/`bit_or`/`bit_xor`, `mode`,
//! `listagg`/`string_agg` and `try_sum`/`try_avg` with Spark 4.2 result
//! types, in two phases, with spill and `DISTINCT`. `count_if` and the
//! boolean aggregates are `count` and `min`/`max` over a rewritten
//! argument (the physical planner).
//!
//! # Operators
//! - [`HashAggregate`] is a pipeline breaker. Each driver aggregates into
//!   its own table (thread-local pre-aggregation); `combine` hands the
//!   tables to the global state and `finalize` merges them. Under memory
//!   pressure a driver spills its table as one *run*: its groups, as
//!   partial states, split into [`PARTITIONS`](crate::hash::PARTITIONS)
//!   files by hash. `finalize` then merges partition by partition (a
//!   partition that still does not fit is split again by the next hash bits),
//!   and answers spill requests through [`Sink::spill_global`](crate::Sink::spill_global).
//! - [`PartialAggregate`] is the streaming first phase (before an exchange):
//!   it emits partial states, flushes its table when it is full instead of
//!   spilling, and switches to pass-through (one state row per input row)
//!   when it does not reduce its input (see [`BypassPolicy`]).
//!
//! # Row-layout tables (`krishiv.exec.agg.rowLayout`, on by default)
//! With the setting on and every grouping key of a type with a row
//! encoding (integers, decimals, dates, timestamps, floats, booleans,
//! strings and binaries), [`HashAggregate`] keeps its keys as rows in radix
//! partitions of a seedless row hash (P2-T159, `rowtable`): strings of up
//! to 12 bytes inline in the row, longer ones in their partition's heap.
//! Each driver's table has a fixed-capacity index that is cleared when
//! full (a group seen again becomes a partial duplicate), or skips lookups
//! when its input is nearly unique; `finalize` merges each range of
//! partitions from every driver in its own task, in parallel, sized from
//! per-partition sketches. Accumulators, spill files and the disk phase are
//! the same as the columnar path's, so results are the same; deterministic
//! mode never clears an index. Dictionary and nested keys, global
//! aggregates and [`PartialAggregate`] keep the columnar tables.
//!
//! # Plans
//! An [`AggregateSpec`] lists the grouping columns and the aggregates. Each
//! aggregate reads raw input columns ([`AggInput::Raw`]) or the partial
//! states of a previous phase ([`AggInput::State`]); the node emits final
//! values or partial states ([`AggregateOutput`]). Output columns are the
//! keys, then one column per aggregate (final) or its state columns
//! (partial, see [`AggregateExpr::state_fields`]). A global aggregate (no
//! grouping columns) returns one row on empty input; a grouped one returns
//! none.
//!
//! `DISTINCT` aggregates and `UNION` (distinct) reuse the same operator:
//! [`distinct_plan`] groups by the keys and the distinct arguments first, then
//! aggregates the distinct values; [`union_distinct`] is a grouping on every
//! column with no aggregates.
//!
//! # Weights
//! Rows carry Z-set weights ([`Weights`](krishiv_types::batch::Weights)):
//! `count` adds the weight, `sum` adds `value × weight`, `avg` divides the
//! weighted sum by the weighted count. `min`/`max`, `first`, `last`,
//! `any_value`, `max_by`, `min_by` and the moment aggregates need
//! retractions to be maintained incrementally (P4 IVM), so they and
//! `DISTINCT` reject any weight other than 1 with
//! `KRISHIV_UNSUPPORTED_FEATURE`. Output rows have weight 1; a group exists
//! once it received a row.

mod accumulator;
mod collect;
mod listagg;
mod mode;
mod moments;
mod node;
mod partial;
mod rowtable;
mod sink;
mod table;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use krishiv_common::error::{
    ArithmeticOverflow, KrishivError, KrishivUnsupportedFeature,
    NumericValueOutOfRangeWithSuggestion,
};
use krishiv_memory::SpillScope;

#[cfg(all(test, not(loom)))]
pub(crate) use accumulator::plain;
pub(crate) use accumulator::{Accumulator, Running, avg_decimal_wide, wide_sum_value};
pub(crate) use collect::{collect_set_supported, kept_bytes};
pub(crate) use moments::arity;
pub use node::{AggregateNode, PartialNode, distinct_plan, union_distinct};
pub use partial::{BypassPolicy, PartialAggregate};
#[cfg(test)]
use sink::row_merge_running;
pub use sink::{AggregateMetrics, HashAggregate};

use crate::error::{Result, internal};

/// Whether `f` is a moment aggregate (`stddev`, `variance`, `skewness`,
/// `kurtosis`, `covar_*`, `corr`, `regr_*`): it reads its arguments as doubles, and
/// the window planner refuses it.
pub fn is_moment(f: &AggregateFunction) -> bool {
    moments::buffer(f).is_some()
}

/// An aggregate function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AggregateFunction {
    /// `count(*)` (no arguments) or `count(a, …)`: rows whose arguments are
    /// all non-NULL. `bigint`, never NULL.
    Count,
    /// `sum(x)`: `bigint` for integral input, `double` for floating point,
    /// `decimal(min(38, p + 10), s)` for `decimal(p, s)`.
    Sum,
    /// `avg(x)`: `double`, or `decimal(min(38, p + 4), min(38, s + 4))` for
    /// `decimal(p, s)`.
    Avg,
    /// `min(x)`: the input type (Spark ordering: NaN is largest, `-0.0 = 0.0`,
    /// strings by UTF-8 bytes).
    Min,
    /// `max(x)`, as `min`.
    Max,
    /// `first(x)`: the first row's value of the group in input order (a
    /// NULL counts unless `ignore_nulls`); NULL for no value. The input type.
    First {
        /// `first(x, true)` / `IGNORE NULLS`: the first non-NULL value.
        ignore_nulls: bool,
    },
    /// `last(x)`, as `first` for the last row.
    Last {
        /// `last(x, true)` / `IGNORE NULLS`: the last non-NULL value.
        ignore_nulls: bool,
    },
    /// `stddev(x)` / `std` / `stddev_samp`: `double`; NULL for no rows, and
    /// for one row unless `legacy_statistical` (then NaN). Rows with a NULL
    /// are skipped. Likewise for the moment aggregates below.
    StddevSamp,
    /// `stddev_pop(x)`: `double`; NULL for no rows.
    StddevPop,
    /// `variance(x)` / `var_samp`, as `StddevSamp` without the root.
    VarSamp,
    /// `var_pop(x)`, as `StddevPop` without the root.
    VarPop,
    /// `skewness(x)`: `double`; NULL for no rows, and for no variance
    /// unless `legacy_statistical` (then NaN).
    Skewness,
    /// `kurtosis(x)`, as `Skewness`.
    Kurtosis,
    /// `corr(x, y)`: Pearson correlation, `double`; both arguments are read
    /// as `double` and a row with a NULL in either is skipped. NULL for no
    /// rows, and for one row unless `legacy_statistical` (then NaN). Over
    /// several rows of which one argument is constant it is
    /// `DIVIDE_BY_ZERO` with `ansi` and NULL without, whatever
    /// `legacy_statistical` says.
    Corr,
    /// `covar_samp(x, y)`: as `Corr` for its arguments and its single-row
    /// rule.
    CovarSamp,
    /// `covar_pop(x, y)`: as `Corr` for its arguments; NULL for no rows.
    CovarPop,
    /// `any_value(x)`: some row's value of the group; here the first row's,
    /// as `first` (a legal choice); NULL for no value. The input type.
    /// In-order merges keep the first row in input order; a `DISTINCT`
    /// first phase groups in hash order.
    AnyValue {
        /// `any_value(x, true)` / `IGNORE NULLS`: the first non-NULL value.
        ignore_nulls: bool,
    },
    /// `max_by(x, y)`: `x` of the row with the largest non-NULL `y` (Spark
    /// ordering), the first such row on a tie for in-order merges; rows
    /// with a NULL `y` are ignored and a NULL `x` is a result. NULL for no
    /// row. The type of `x`. Built with [`AggregateExpr::by`]. A
    /// `DISTINCT` first phase groups in hash order, so a tie there keeps
    /// the first row in hash order, not in input order.
    MaxBy,
    /// `min_by(x, y)`, as `max_by` for the smallest `y` (same tie caveat).
    MinBy,
    /// `collect_list(x)` (and `array_agg`, which the analyzer resolves as
    /// `collect_list` for Spark's display name): the non-NULL values in
    /// input order within a partition, partitions merged in partition-index
    /// order; `[]` for no values, never NULL.
    CollectList,
    /// `collect_set(x)`: the distinct non-NULL values in first-seen order;
    /// `[]` for no values, never NULL. Element identity is the grouping
    /// key encoding (`crate::hash::KeyEncoder`), which treats NaN as equal
    /// to NaN and `-0.0` as equal to `0.0` as grouping keys do (not yet
    /// checked against a Spark golden).
    CollectSet,
    /// `vector_sum(x)`: element-wise sums over `ARRAY<FLOAT>` (NULL vectors
    /// skipped); NULL for no non-NULL vector.
    VectorSum,
    /// `vector_avg(x)`: element-wise means over `ARRAY<FLOAT>` (NULL vectors
    /// skipped); NULL for no non-NULL vector.
    VectorAvg,
    /// `bit_and(x)`, `bit_or(x)`, `bit_xor(x)`: the bitwise AND, OR, XOR
    /// of the non-NULL `TINYINT` .. `BIGINT` values; NULL for no
    /// non-NULL value. The input type.
    BitAnd,
    /// `bit_or(x)`, as `bit_and`.
    BitOr,
    /// `bit_xor(x)`, as `bit_and`.
    BitXor,
    /// `regr_count(y, x)`: the number of rows where both are non-NULL.
    /// `bigint`, never NULL. Shares the pair-moment buffer with
    /// [`Corr`](AggregateFunction::Corr) (see `moments`).
    RegrCount,
    /// `regr_avgx(y, x)`: the mean of `x` over non-NULL pairs. `double`.
    RegrAvgx,
    /// `regr_avgy(y, x)`: the mean of `y` over non-NULL pairs. `double`.
    RegrAvgy,
    /// `regr_sxx(y, x)`: the sum of squares of `x` about its mean.
    /// `double`.
    RegrSxx,
    /// `regr_syy(y, x)`: the sum of squares of `y` about its mean.
    /// `double`.
    RegrSyy,
    /// `regr_sxy(y, x)`: the sum of cross-products about the means.
    /// `double`.
    RegrSxy,
    /// `regr_slope(y, x)`: the slope of the regression line; NULL when
    /// `x` does not vary. `double`.
    RegrSlope,
    /// `regr_intercept(y, x)`: the intercept of the regression line.
    /// `double`.
    RegrIntercept,
    /// `regr_r2(y, x)`: the coefficient of determination; NULL when `x`
    /// does not vary, 1 when only `y` does not. `double`.
    RegrR2,
    /// `mode(col)`: the most frequent non-NULL value; ties broken by
    /// Spark's hash-map order. The input type, NULL for no value.
    Mode,
    /// `mode(col, true)` and `mode() WITHIN GROUP (ORDER BY col)` (ASC):
    /// the smallest value among the most frequent.
    ModeMin,
    /// `mode() WITHIN GROUP (ORDER BY col DESC)`: the largest value
    /// among the most frequent.
    ModeMax,
    /// `listagg(expr[, delimiter])` (and `string_agg`): the non-NULL
    /// values joined by `delimiter` (empty when none or NULL was
    /// written), in input order within a partition (partitions merged in
    /// partition-index order) or sorted by the ordering keys. The input
    /// type (`STRING` or `BINARY`), NULL for no value.
    ListAgg {
        /// The delimiter bytes (empty when none or NULL was written).
        delimiter: Vec<u8>,
        /// One per ordering key, in order: descending, nulls first.
        orders: Vec<(bool, bool)>,
    },
    /// `try_sum(x)`: `sum` that gives NULL instead of an overflow error,
    /// ANSI on or off.
    TrySum,
    /// `try_avg(x)`: `avg` that gives NULL instead of an overflow error,
    /// ANSI on or off.
    TryAvg,
}

/// Where an aggregate's arguments come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AggInput {
    /// Raw input columns.
    Raw,
    /// The state columns of the same aggregate from a previous phase.
    State,
}

/// What the node emits per aggregate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AggregateOutput {
    /// Final values.
    Final,
    /// Partial states, for a later phase.
    Partial,
}

/// One aggregate of an [`AggregateSpec`].
#[derive(Clone, Debug, PartialEq)]
pub struct AggregateExpr {
    /// The function.
    pub function: AggregateFunction,
    /// Argument column indices: raw columns, or the state columns in order.
    pub args: Vec<usize>,
    /// Where the arguments come from.
    pub input: AggInput,
    /// The type of the raw argument (`count` ignores it). Fixed by the
    /// planner; state columns derive from it. For `max_by`/`min_by`, a
    /// struct of the value's type then the ordering key's (see
    /// [`AggregateExpr::by`]).
    pub arg_type: DataType,
    /// The output column name.
    pub name: String,
    /// A boolean input column: the aggregate reads only the rows where it
    /// is true (`FILTER (WHERE ...)`); `None` reads all rows. Set only on
    /// raw input: partial states are already filtered, so [`merging`](Self::merging)
    /// clears it. A NULL is not true.
    pub mask: Option<usize>,
}

impl AggregateExpr {
    /// A raw-input aggregate of column `arg` of type `arg_type`.
    pub fn new(
        function: AggregateFunction,
        arg: usize,
        arg_type: DataType,
        name: impl Into<String>,
    ) -> Self {
        Self {
            function,
            args: vec![arg],
            input: AggInput::Raw,
            arg_type,
            name: name.into(),
            mask: None,
        }
    }

    /// `max_by(x, y)` / `min_by(x, y)` of columns `value` and `key`, of
    /// types `value_type` and `key_type`. Both must be types `min`/`max`
    /// take.
    pub fn by(
        function: AggregateFunction,
        (value, key): (usize, usize),
        (value_type, key_type): (DataType, DataType),
        name: impl Into<String>,
    ) -> Self {
        Self {
            function,
            args: vec![value, key],
            input: AggInput::Raw,
            arg_type: DataType::Struct(Fields::from(vec![
                Field::new("value", value_type, true),
                Field::new("key", key_type, true),
            ])),
            name: name.into(),
            mask: None,
        }
    }

    /// `listagg(expr[, delimiter])` of column `value` (of type
    /// `value_type`) with ordering-key columns `orders` (of types
    /// `order_types`), of `function` (`ListAgg` with the fused delimiter
    /// and directions). The argument type is the struct of the value's
    /// type then the keys' (as [`by`](Self::by) does for `max_by`).
    pub fn listagg(
        function: AggregateFunction,
        value: usize,
        value_type: DataType,
        orders: Vec<usize>,
        order_types: Vec<DataType>,
        name: impl Into<String>,
    ) -> Self {
        let mut fields = Vec::with_capacity(1 + order_types.len());
        fields.push(Field::new("value", value_type, true));
        for (i, t) in order_types.into_iter().enumerate() {
            fields.push(Field::new(format!("key{i}"), t, true));
        }
        let mut args = Vec::with_capacity(1 + orders.len());
        args.push(value);
        args.extend(orders);
        Self {
            function,
            args,
            input: AggInput::Raw,
            arg_type: DataType::Struct(Fields::from(fields)),
            name: name.into(),
            mask: None,
        }
    }

    /// The value and ordering-key types of a `listagg`: the struct's
    /// first field, then the rest.
    fn listagg_types(&self) -> Option<(&DataType, Vec<&DataType>)> {
        match &self.arg_type {
            DataType::Struct(f) if !f.is_empty() => {
                let mut keys = Vec::with_capacity(f.len() - 1);
                for field in f.iter().skip(1) {
                    keys.push(field.data_type());
                }
                Some((f[0].data_type(), keys))
            }
            _ => None,
        }
    }

    /// `count(*)`.
    pub fn count_star(name: impl Into<String>) -> Self {
        Self {
            function: AggregateFunction::Count,
            args: Vec::new(),
            input: AggInput::Raw,
            arg_type: DataType::Null,
            name: name.into(),
            mask: None,
        }
    }

    /// The same aggregate reading its state columns at `first..`. A mask
    /// is not carried: the states were built from the rows it passed, and
    /// the state columns do not hold it.
    pub fn merging(&self, first: usize) -> Self {
        let n = self.state_fields().map_or(0, |f| f.len());
        Self {
            args: (first..first + n).collect(),
            input: AggInput::State,
            mask: None,
            ..self.clone()
        }
    }

    /// The Spark 4.2 result type.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for an argument type the function does
    /// not take here.
    pub fn result_type(&self) -> Result<DataType> {
        use AggregateFunction as F;
        let t = &self.arg_type;
        Ok(match (&self.function, t) {
            (F::Count, _) => DataType::Int64,
            (F::Sum | F::TrySum, t) if is_integral(t) => DataType::Int64,
            (F::Sum | F::TrySum | F::Avg | F::TryAvg, DataType::Float32 | DataType::Float64) => {
                DataType::Float64
            }
            (F::Avg | F::TryAvg, t) if is_integral(t) => DataType::Float64,
            // Spark refuses negative decimal scales by default
            // (`spark.sql.legacy.allowNegativeScaleOfDecimal`).
            (F::Sum | F::TrySum, DataType::Decimal64(p, s) | DataType::Decimal128(p, s))
                if *s >= 0 =>
            {
                sum_decimal(t, *p, *s)
            }
            (F::Avg | F::TryAvg, DataType::Decimal64(p, s) | DataType::Decimal128(p, s))
                if *s >= 0 =>
            {
                decimal_on(t, (*p + 4).min(38), s.saturating_add(4).min(38))
            }
            (F::Min | F::Max, t)
                if accumulator::min_max_supported(t)
                    || accumulator::struct_min_max_supported(t) =>
            {
                t.clone()
            }
            (F::First { .. } | F::Last { .. } | F::AnyValue { .. }, t)
                if accumulator::min_max_supported(t) =>
            {
                t.clone()
            }
            // `regr_count` reads its arguments as doubles (like the other
            // moment aggregates) but counts the non-NULL pairs.
            (F::RegrCount, t) if moments::reads_as_double(t) => DataType::Int64,
            // The analyzer casts every argument to `double` (decimals
            // included); the executor also reads integers and `float`.
            (f, t) if moments::buffer(f).is_some() && moments::reads_as_double(t) => {
                DataType::Float64
            }
            // Bitwise aggregates keep the integral input type.
            (F::BitAnd | F::BitOr | F::BitXor, t) if is_integral(t) => t.clone(),
            // `mode`: the most frequent value, of the input type (any
            // type; the accumulator keeps first-seen samples).
            (F::Mode | F::ModeMin | F::ModeMax, t) => t.clone(),
            // `listagg`: the value type; the ordering keys ride in the
            // struct with it (see [`AggregateExpr::listagg`]).
            (F::ListAgg { .. }, DataType::Struct(f)) if !f.is_empty() => f[0].data_type().clone(),
            // `collect_list` / `collect_set` keep the input type under an
            // array; the vector aggregates need a list of `float`. The set
            // needs the shared key encoding for its elements.
            (F::CollectList, t) => collect_result(t),
            (F::CollectSet, t) if collect_set_supported(t) => collect_result(t),
            (F::VectorSum | F::VectorAvg, DataType::List(f))
                if *f.data_type() == DataType::Float32 =>
            {
                vector_result()
            }
            (F::MaxBy | F::MinBy, _) => match self.by_types() {
                Some((v, k))
                    if accumulator::min_max_supported(v) && accumulator::min_max_supported(k) =>
                {
                    v.clone()
                }
                _ => {
                    return Err(unsupported(format!(
                        "{:?} of {}",
                        self.function, self.arg_type
                    )));
                }
            },
            (f, t) => return Err(unsupported(format!("{f:?} of {t}"))),
        })
    }

    /// The partial-state columns (names derive from [`name`](Self::name)).
    ///
    /// # Errors
    /// As [`result_type`](Self::result_type).
    pub fn state_fields(&self) -> Result<Vec<Field>> {
        use AggregateFunction as F;
        let out = self.result_type()?;
        let n = |suffix: &str| format!("{}#{suffix}", self.name);
        Ok(match (&self.function, &out) {
            (F::Count, _) => vec![Field::new(n("count"), DataType::Int64, false)],
            // `try_sum` of integers sticks overflow in its state, like a
            // decimal `sum` does.
            (F::Sum | F::TrySum, DataType::Decimal64(..) | DataType::Decimal128(..))
            | (F::TrySum, DataType::Int64) => vec![
                Field::new(n("sum"), out.clone(), true),
                Field::new(n("nonempty"), DataType::Boolean, false),
            ],
            (F::Sum | F::TrySum, _) => vec![Field::new(n("sum"), out.clone(), true)],
            (F::Avg | F::TryAvg, _) => vec![
                Field::new(n("sum"), self.avg_sum_type(), true),
                Field::new(n("count"), DataType::Int64, false),
            ],
            (F::Min | F::Max, _) => vec![Field::new(n("value"), out, true)],
            // Whether a row was taken: a NULL value is a value.
            (F::First { .. } | F::Last { .. } | F::AnyValue { .. }, _) => vec![
                Field::new(n("value"), out, true),
                Field::new(n("valueSet"), DataType::Boolean, false),
            ],
            (
                F::StddevSamp
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
                | F::RegrR2,
                _,
            ) => moments::state_fields(&self.function, &self.name),
            // Bitwise aggregates keep one nullable value (NULL until a
            // non-NULL input arrives).
            (F::BitAnd | F::BitOr | F::BitXor, _) => {
                vec![Field::new(n("value"), out, true)]
            }
            // `mode`: the distinct first-seen values with their counts.
            (F::Mode | F::ModeMin | F::ModeMax, _) => {
                let element = Field::new("key", self.arg_type.clone(), true);
                vec![
                    Field::new(n("keys"), DataType::List(Arc::new(element)), false),
                    Field::new(
                        n("counts"),
                        DataType::List(Arc::new(Field::new("count", DataType::Int64, false))),
                        false,
                    ),
                ]
            }
            // `listagg`: the non-NULL values with the ordering keys they
            // were folded with (one list per struct field).
            (F::ListAgg { .. }, _) => {
                let fields = match &self.arg_type {
                    DataType::Struct(f) => f.to_vec(),
                    _ => {
                        return Err(unsupported(format!("listagg of {}", self.arg_type)));
                    }
                };
                fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let name = if i == 0 {
                            n("values")
                        } else {
                            n(&format!("key{i}"))
                        };
                        Field::new(
                            name,
                            DataType::List(Arc::new(Field::new(
                                "element",
                                f.data_type().clone(),
                                true,
                            ))),
                            false,
                        )
                    })
                    .collect()
            }
            // The key is NULL until a row with a key was taken: it says
            // whether the value (which may be NULL) counts.
            (F::MaxBy | F::MinBy, _) => {
                let key = self.by_types().map_or(DataType::Null, |(_, k)| k.clone());
                vec![
                    Field::new(n("value"), out, true),
                    Field::new(n("key"), key, true),
                ]
            }
            // The collect and vector aggregates merge in driver-partition
            // order (see [`Run::partition`]).
            (F::CollectList | F::CollectSet, _) => {
                vec![Field::new(n("values"), out.clone(), true)]
            }
            // The vector aggregates keep the running sums and the count of
            // non-NULL vectors; merging adds them element-wise.
            (F::VectorSum | F::VectorAvg, _) => vec![
                Field::new(n("sums"), vector_sums_type(), true),
                Field::new(n("count"), DataType::Int64, false),
            ],
        })
    }

    /// The value and ordering key types of a `max_by`/`min_by`.
    fn by_types(&self) -> Option<(&DataType, &DataType)> {
        match &self.arg_type {
            DataType::Struct(f) if f.len() == 2 => Some((f[0].data_type(), f[1].data_type())),
            _ => None,
        }
    }

    /// The type `avg` sums in: `double`, or `sum`'s decimal type.
    fn avg_sum_type(&self) -> DataType {
        match self.arg_type {
            DataType::Decimal64(p, s) | DataType::Decimal128(p, s) => {
                sum_decimal(&self.arg_type, p, s)
            }
            _ => DataType::Float64,
        }
    }
}

fn sum_decimal(like: &DataType, p: u8, s: i8) -> DataType {
    decimal_on(like, (p + 10).min(38), s)
}

/// `collect_list` / `collect_set` of `t`: an array of `t`, laid out exactly
/// as the analyzer's `array<t>` ([`krishiv_types::DataType::to_arrow`]), so
/// the planner's check of the executor's result type against the analyzer's
/// holds.
fn collect_result(t: &DataType) -> DataType {
    DataType::List(Arc::new(Field::new("element", t.clone(), false)))
}

/// `vector_sum` / `vector_avg`: an array of `float`, laid out as the
/// analyzer's.
fn vector_result() -> DataType {
    DataType::List(Arc::new(Field::new("element", DataType::Float32, false)))
}

/// The partial-state sums of the vector aggregates: an array of `double`,
/// one entry per dimension (`NULL` while no non-NULL vector was seen).
fn vector_sums_type() -> DataType {
    DataType::List(Arc::new(Field::new("element", DataType::Float64, false)))
}

/// Whether `f` observes input order: its merges must run in partition-index
/// order (the sink sorts its tables by driver partition), not biggest-first.
pub fn is_order_sensitive(f: &AggregateFunction) -> bool {
    use AggregateFunction as F;
    matches!(
        f,
        F::CollectList
            | F::CollectSet
            | F::VectorSum
            | F::VectorAvg
            // `listagg` keeps input order (or sorts once, at evaluation);
            // `mode` iterates its map in slot order.
            | F::Mode
            | F::ModeMin
            | F::ModeMax
            | F::ListAgg { .. }
    )
}

/// `decimal(p, s)` on the carrier of the input type `like` (design
/// P2-T160): `Decimal64` when `like` is `Decimal64` and `p <= 18`, else
/// `Decimal128`. Inputs on the canonical carrier by precision give results
/// on the canonical carrier.
fn decimal_on(like: &DataType, p: u8, s: i8) -> DataType {
    if matches!(like, DataType::Decimal64(..)) && p <= 18 {
        DataType::Decimal64(p, s)
    } else {
        DataType::Decimal128(p, s)
    }
}

fn is_integral(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
    )
}

/// A hash aggregation: grouping columns, aggregates and the output kind.
#[derive(Clone, Debug)]
pub struct AggregateSpec {
    /// The input schema.
    pub input: SchemaRef,
    /// Grouping column indices (empty: a global aggregate).
    pub group_by: Vec<usize>,
    /// The aggregates.
    pub aggregates: Vec<AggregateExpr>,
    /// Final values or partial states.
    pub output: AggregateOutput,
    /// `spark.sql.ansi.enabled`: overflow raises instead of wrapping (`sum`
    /// of integers) or becoming NULL (decimals).
    pub ansi: bool,
    /// Where tables spill; `None` makes the aggregate unspillable.
    pub spill: Option<SpillScope>,
    /// Reject input weights other than 1 (`DISTINCT`, `UNION`: set
    /// semantics over Z-sets need IVM).
    pub require_unit_weights: bool,
    /// `spark.sql.legacy.statisticalAggregate`: where Spark divides by zero,
    /// the sample moment aggregates (`stddev_samp`, `var_samp`, `covar_samp`,
    /// `corr`, `skewness`, `kurtosis`) give NaN instead of NULL.
    pub legacy_statistical: bool,
}

impl AggregateSpec {
    /// The output schema: keys, then each aggregate's value or states.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a grouping column outside the input; the
    /// aggregates' typing errors.
    pub fn output_schema(&self) -> Result<SchemaRef> {
        let mut fields = Vec::new();
        for &k in &self.group_by {
            let f = self.input.fields().get(k);
            let f = f.ok_or_else(|| internal(format!("grouping column {k} out of range")))?;
            fields.push(f.as_ref().clone());
        }
        for a in &self.aggregates {
            match self.output {
                AggregateOutput::Final => {
                    let nullable = a.function != AggregateFunction::Count;
                    fields.push(Field::new(&a.name, a.result_type()?, nullable));
                }
                AggregateOutput::Partial => fields.extend(a.state_fields()?),
            }
        }
        Ok(Arc::new(Schema::new(fields)))
    }

    /// The schema of the partial states (keys, then every state column):
    /// what spill files and a partial phase hold.
    pub(crate) fn state_schema(&self) -> Result<SchemaRef> {
        Self {
            output: AggregateOutput::Partial,
            ..self.clone()
        }
        .output_schema()
    }
}

/// `spark.sql.ansi.enabled` as Spark's error messages quote it.
const ANSI_CONF: &str = "\"spark.sql.ansi.enabled\"";

/// `KRISHIV_UNSUPPORTED_FEATURE`.
pub(crate) fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

/// Spark's ANSI `ARITHMETIC_OVERFLOW` of a `long` addition in `sum`.
pub(crate) fn long_overflow() -> KrishivError {
    ArithmeticOverflow {
        message: "overflow".to_owned(),
        alternative: " Use 'try_add' to tolerate overflow and return NULL instead.".to_owned(),
        config: ANSI_CONF.to_owned(),
    }
    .into()
}

/// Spark's `overflowInSumOfDecimalError`; `function` is `sum` or `avg`.
pub(crate) fn decimal_sum_overflow(function: &str) -> KrishivError {
    ArithmeticOverflow {
        message: "Overflow in sum of decimals".to_owned(),
        alternative: format!(" Use 'try_{function}' to tolerate overflow and return NULL instead."),
        config: ANSI_CONF.to_owned(),
    }
    .into()
}

/// `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION`.
pub(crate) fn out_of_range(value: String, precision: u8, scale: i8) -> KrishivError {
    NumericValueOutOfRangeWithSuggestion {
        value,
        precision: precision.to_string(),
        scale: scale.to_string(),
        config: ANSI_CONF.to_owned(),
    }
    .into()
}
