// SPDX-License-Identifier: Apache-2.0
//! The signatures of `__time_windows`, the call the analyzer builds for Spark's
//! `window(time_column, window_duration, slide_duration, start_time)`, and of `window_time`.
//!
//! `window_time(w)` (registered as `__window_time`, which the analyzer renames the calls to: the Spark catalog's `window_time` is an aggregate-kind row) is valid only for a column the window rules produced (Spark marks the attribute with
//! metadata). The engine cannot mark a column, so it reads the column's name: `window` and `session_window`
//! (the names the rewrite gives) are windows, any other struct is `_LEGACY_ERROR_TEMP_3101`. A window used
//! as the time column of `window(..)` (without `window_time`) is refused the same way.

use krishiv_common::error::{
    DatatypeMismatchUnexpectedInputType, KrishivError, LegacyErrorTemp3101,
};
use krishiv_types::{ArrayType, DataType, StructField, StructType};

use super::defs::{Spec, exact, scalar};
use super::hof::{call_sql, ordinal, quoted};
use crate::signature::{ArgType, CallTypes, InputBuilder, InputType, Inputs};

/// The names a window column has when the rewrite made it (`__tw_inner` is the inner window of a nested
/// `window(window(ts, ..), ..)`).
fn is_window_column(sql: Option<&str>) -> bool {
    sql.is_some_and(|s| {
        ["window", "session_window", "__tw_inner"]
            .iter()
            .any(|n| s.eq_ignore_ascii_case(n))
    })
}

/// The type of the time of a window struct `struct<start: T, end: T>` (`T` a timestamp type), if it is one.
fn window_time_type(t: &DataType) -> Option<DataType> {
    let DataType::Struct(s) = t else {
        return None;
    };
    match s.fields() {
        [a, b]
            if a.name() == "start"
                && b.name() == "end"
                && a.data_type() == b.data_type()
                && matches!(a.data_type(), DataType::Timestamp | DataType::TimestampNtz) =>
        {
            Some(a.data_type().clone())
        }
        _ => None,
    }
}

fn not_a_window(name: &str, c: &CallTypes<'_>) -> KrishivError {
    LegacyErrorTemp3101 {
        window_time: format!(
            "{name}({})",
            (0..c.args().len())
                .map(|i| c.sql(i).unwrap_or_default())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
    .into()
}

/// The first input of `__time_windows`: a timestamp type (a string is cast), or a struct for the result rule
/// to judge.
fn windows_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let first = if matches!(c.data_type(0), Some(DataType::Struct(_))) {
        InputType::Any
    } else {
        InputType::OneOf(vec![DataType::Timestamp, DataType::TimestampNtz])
    };
    vec![
        first,
        exact(DataType::Long),
        exact(DataType::Long),
        exact(DataType::Long),
    ]
}

/// `array<struct<start: T, end: T>>` for the time column's type `T` (`timestamp` or `timestamp_ntz`):
/// NULL for a NULL time.
fn windows(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let t = match c.data_type(0) {
        // A window as the time column is Spark's `window_time(window)` of a marked column: the engine reads
        // `window_time(..)` for that (see the module documentation), so it is refused here.
        Some(DataType::Struct(_)) => return Err(not_a_window("window_time", c)),
        Some(t @ (DataType::Timestamp | DataType::TimestampNtz)) => t.clone(),
        _ => DataType::Timestamp,
    };
    let window = DataType::Struct(StructType::new(vec![
        StructField::new("start", t.clone(), true),
        StructField::new("end", t, true),
    ]));
    Ok(ArgType::new(
        DataType::Array(std::sync::Arc::new(ArrayType::new(window, false))),
        true,
    ))
}

/// `window_time(window)`: `window.end - 1 microsecond`, of the window's time type.
fn window_time(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let Some(t) = c.data_type(0) else {
        return Ok(ArgType::new(DataType::Timestamp, true));
    };
    if !matches!(t, DataType::Struct(_)) {
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql("window_time", c),
            param_index: ordinal(0).to_owned(),
            required_type: "\"STRUCT\"".to_owned(),
            input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            input_type: quoted(Some(t)),
        }
        .into());
    }
    if !is_window_column(c.sql(0).as_deref()) {
        return Err(not_a_window("window_time", c));
    }
    let time = window_time_type(t).ok_or_else(|| not_a_window("window_time", c))?;
    Ok(ArgType::new(time, true))
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(
        &["__time_windows"],
        &Inputs::Built(InputBuilder(windows_inputs)),
        windows,
    );
    out.extend(scalar(
        &["__window_time"],
        &Inputs::Expects(vec![InputType::Any]),
        window_time,
    ));
    out
}
