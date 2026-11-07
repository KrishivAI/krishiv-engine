// SPDX-License-Identifier: Apache-2.0
//! Signatures of `time_bucket` (Spark 4.2 `TimeBucket`) and of the `TIME` operators the analyzer builds
//! (`__time_add_interval` for `TimeAddInterval`, `__subtract_times` for `SubtractTimes`).

use krishiv_common::error::{
    DatatypeMismatchNonFoldableInput, DatatypeMismatchUnexpectedInputType,
    DatatypeMismatchValueOutOfRange, KrishivError,
};
use krishiv_types::{DataType, DayTimeField, YearMonthField};

use super::defs::{Spec, scalar, typed};
use crate::signature::{ArgType, CallTypes, Constant, InputBuilder, InputType, Inputs};

pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(
        &["__time_add_interval"],
        &Inputs::Built(InputBuilder(as_built_inputs)),
        time_add_interval_result,
    );
    out.extend(scalar(
        &["__subtract_times"],
        &Inputs::Built(InputBuilder(as_built_inputs)),
        subtract_times_result,
    ));
    out.extend(scalar(
        &["time_bucket"],
        &Inputs::Built(InputBuilder(time_bucket_inputs)),
        time_bucket_result,
    ));
    out
}

/// The operand types of `__time_add_interval(time, interval, end_field)` and `__subtract_times(end, start)`, as
/// the analyzer builds them: each argument keeps its type.
fn as_built_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    (0..c.args().len())
        .map(|i| {
            c.data_type(i)
                .map_or(InputType::Any, |t| InputType::Strict(t.clone()))
        })
        .collect()
}

/// `TimeAddInterval`: `TIME(max(p, 6))` for an interval ending in `SECOND`, else `TIME(p)`.
fn time_add_interval_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let p = match c.data_type(0) {
        Some(DataType::Time { precision }) => *precision,
        _ => 6,
    };
    let seconds = matches!(
        c.data_type(1),
        Some(DataType::DayTimeInterval {
            end: DayTimeField::Second,
            ..
        })
    );
    let precision = if seconds { p.max(6) } else { p };
    Ok(ArgType::new(
        DataType::Time { precision },
        c.args().iter().take(2).any(|a| a.nullable),
    ))
}

/// `SubtractTimes`: an `INTERVAL HOUR TO SECOND`.
fn subtract_times_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(
        DataType::DayTimeInterval {
            start: DayTimeField::Hour,
            end: DayTimeField::Second,
        },
        c,
    )
}

/// `time_bucket(bucketSize, ts, origin)` (Spark 4.2 `TimeBucket` through `TimeBucketExpressionBuilder`): no
/// implicit casts (`ExpectsInputTypes`); an untyped `NULL` size is a day-time interval, an untyped `NULL` `ts` or
/// `origin` takes the other's timestamp type (else `TIMESTAMP`), as the builder's `retypeNull` does.
fn time_bucket_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let timestamp = |t: Option<&DataType>| {
        t.filter(|t| matches!(t, DataType::Timestamp | DataType::TimestampNtz))
            .cloned()
    };
    let size = match c.data_type(0) {
        Some(t @ (DataType::DayTimeInterval { .. } | DataType::YearMonthInterval { .. })) => {
            InputType::Strict(t.clone())
        }
        _ => InputType::StrictOneOf(vec![
            DataType::DEFAULT_DAY_TIME_INTERVAL,
            DataType::YearMonthInterval {
                start: YearMonthField::Year,
                end: YearMonthField::Month,
            },
        ]),
    };
    let any_timestamp =
        || InputType::StrictOneOf(vec![DataType::Timestamp, DataType::TimestampNtz]);
    let ts_type = match c.data_type(1) {
        None => timestamp(c.data_type(2)).or(Some(DataType::Timestamp)),
        t => timestamp(t),
    };
    let ts = ts_type
        .clone()
        .map_or_else(any_timestamp, InputType::Strict);
    let origin = match c.data_type(2) {
        None => match &ts_type {
            Some(t) => InputType::Strict(t.clone()),
            None => any_timestamp(),
        },
        t => timestamp(t).map_or_else(any_timestamp, InputType::Strict),
    };
    vec![size, ts, origin]
}

/// `time_bucket`'s checks after the input types, in Spark's order: a foldable, positive bucket size, a foldable
/// origin, then an origin of `ts`'s type. The result has `ts`'s type.
fn time_bucket_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let sql = |i: usize| c.sql(i).unwrap_or_default();
    let call = || format!("\"time_bucket({}, {}, {})\"", sql(0), sql(1), sql(2));
    let type_sql = |i: usize| {
        c.data_type(i)
            .map_or_else(|| "\"VOID\"".to_owned(), |t| format!("\"{}\"", t.sql()))
    };
    let non_foldable = |i: usize, name: &str| DatatypeMismatchNonFoldableInput {
        sql_expr: call(),
        input_name: format!("`{name}`"),
        input_type: type_sql(i),
        input_expr: format!("\"{}\"", sql(i)),
    };
    if !c.foldable(0) {
        return Err(non_foldable(0, "bucketSize").into());
    }
    match (c.constant(0), c.data_type(0)) {
        (Some(Constant::Error(e)), _) => return Err(e.clone()),
        (Some(Constant::Interval(v)), Some(t)) if *v <= 0 => {
            let current_value = match t {
                DataType::YearMonthInterval { start, end } => {
                    krishiv_types::interval::year_month_string(
                        i32::try_from(*v).unwrap_or(i32::MIN),
                        *start,
                        *end,
                    )
                }
                DataType::DayTimeInterval { start, end } => {
                    krishiv_types::interval::day_time_string(*v, *start, *end)
                }
                _ => v.to_string(),
            };
            return Err(DatatypeMismatchValueOutOfRange {
                sql_expr: call(),
                expr_name: "`bucketSize`".to_owned(),
                value_range: "(0, inf)".to_owned(),
                current_value,
            }
            .into());
        }
        _ => {}
    }
    if !c.foldable(2) {
        return Err(non_foldable(2, "origin").into());
    }
    if c.data_type(1) != c.data_type(2) {
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call(),
            param_index: "third".to_owned(),
            required_type: type_sql(1),
            input_sql: format!("\"{}\"", sql(2)),
            input_type: type_sql(2),
        }
        .into());
    }
    typed(c.data_type(1).cloned().unwrap_or(DataType::Timestamp), c)
}
