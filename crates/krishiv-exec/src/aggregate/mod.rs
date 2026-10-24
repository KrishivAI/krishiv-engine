// SPDX-License-Identifier: Apache-2.0
//! Hash aggregation (ADR 0011): `count`, `sum`, `avg`, `min`, `max`,
//! `first`, `last` and the moment aggregates (`stddev`, `variance`,
//! `skewness`, `kurtosis`, `covar_*`, `corr`) with Spark 4.2 result types, in
//! two phases, with spill and `DISTINCT`.
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
//! # Row-layout tables (`krishiv.exec.agg.rowLayout`, off by default)
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
//! the same as the default path's, so results are the same; deterministic
//! mode never clears an index. Dictionary and nested keys, global
//! aggregates and [`PartialAggregate`] keep the default tables.
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
//! weighted sum by the weighted count. `min`/`max` need retractions to be
//! maintained incrementally (P4 IVM), so they, the moment aggregates and
//! `DISTINCT` reject any weight other than 1 with
//! `KRISHIV_UNSUPPORTED_FEATURE`. Output rows have weight 1; a group exists
//! once it received a row.

mod accumulator;
mod moments;
mod node;
mod partial;
mod rowtable;
mod sink;
mod table;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::{
    ArithmeticOverflow, KrishivError, KrishivUnsupportedFeature,
    NumericValueOutOfRangeWithSuggestion,
};
use krishiv_memory::SpillScope;

#[cfg(all(test, not(loom)))]
pub(crate) use accumulator::plain;
pub(crate) use accumulator::{Accumulator, Running, avg_decimal_wide, wide_sum_value};
pub(crate) use moments::arity;
pub use node::{AggregateNode, PartialNode, distinct_plan, union_distinct};
pub use partial::{BypassPolicy, PartialAggregate};
#[cfg(test)]
use sink::row_merge_running;
pub use sink::{AggregateMetrics, HashAggregate};

use crate::error::{Result, internal};

/// An aggregate function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    /// planner; state columns derive from it.
    pub arg_type: DataType,
    /// The output column name.
    pub name: String,
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
        }
    }

    /// The same aggregate reading its state columns at `first..`.
    pub fn merging(&self, first: usize) -> Self {
        let n = self.state_fields().map_or(0, |f| f.len());
        Self {
            args: (first..first + n).collect(),
            input: AggInput::State,
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
        Ok(match (self.function, t) {
            (F::Count, _) => DataType::Int64,
            (F::Sum, t) if is_integral(t) => DataType::Int64,
            (F::Sum | F::Avg, DataType::Float32 | DataType::Float64) => DataType::Float64,
            (F::Avg, t) if is_integral(t) => DataType::Float64,
            // Spark refuses negative decimal scales by default
            // (`spark.sql.legacy.allowNegativeScaleOfDecimal`).
            (F::Sum, DataType::Decimal64(p, s) | DataType::Decimal128(p, s)) if *s >= 0 => {
                sum_decimal(t, *p, *s)
            }
            (F::Avg, DataType::Decimal64(p, s) | DataType::Decimal128(p, s)) if *s >= 0 => {
                decimal_on(t, (*p + 4).min(38), s.saturating_add(4).min(38))
            }
            (F::Min | F::Max | F::First { .. } | F::Last { .. }, t)
                if accumulator::min_max_supported(t) =>
            {
                t.clone()
            }
            // The analyzer casts every argument to `double` (decimals
            // included); the executor also reads integers and `float`.
            (f, t) if moments::buffer(f).is_some() && moments::reads_as_double(t) => {
                DataType::Float64
            }
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
        Ok(match (self.function, &out) {
            (F::Count, _) => vec![Field::new(n("count"), DataType::Int64, false)],
            (F::Sum, DataType::Decimal64(..) | DataType::Decimal128(..)) => vec![
                Field::new(n("sum"), out.clone(), true),
                Field::new(n("nonempty"), DataType::Boolean, false),
            ],
            (F::Sum, _) => vec![Field::new(n("sum"), out.clone(), true)],
            (F::Avg, _) => vec![
                Field::new(n("sum"), self.avg_sum_type(), true),
                Field::new(n("count"), DataType::Int64, false),
            ],
            (F::Min | F::Max, _) => vec![Field::new(n("value"), out, true)],
            // Whether a row was taken: a NULL value is a value.
            (F::First { .. } | F::Last { .. }, _) => vec![
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
                | F::CovarPop,
                _,
            ) => moments::state_fields(self.function, &self.name),
        })
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
