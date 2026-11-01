// SPDX-License-Identifier: Apache-2.0
//! The array reshaping signatures: `flatten`, `slice` and `array_join`
//! (Spark 4.2's `Flatten`, `Slice` and `ArrayJoin`). Checked against a
//! Spark 4.2.0 run.
//!
//! `flatten` takes an array of arrays (`UNEXPECTED_INPUT_TYPE` otherwise,
//! also for an untyped `NULL`); its result holds NULL when either level did.
//! `slice`'s start and length are cast to `INT`; the argument checks (start
//! 0, negative length) happen per row. `array_join` takes an array of
//! strings (other element types are cast), a string delimiter and an
//! optional string for NULL elements.

use krishiv_common::error::{
    DatatypeMismatchSequenceWrongInputTypes, DatatypeMismatchUnexpectedInputType, KrishivError,
    KrishivUnsupportedFeature, WrongNumArgsWithoutSuggestion,
};
use krishiv_types::{ArrayType, DataType};

use super::defs::{Spec, scalar};
use super::hof::{call_sql, ordinal, quoted};
use crate::signature::{ArgType, CallTypes, InputBuilder, InputType, Inputs};

/// `flatten(array<array<T>>)`: the elements of the inner arrays in order.
fn flatten(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if let Some(DataType::Array(outer)) = c.data_type(0)
        && let DataType::Array(inner) = outer.element_type()
    {
        return Ok(ArgType::new(
            DataType::Array(
                ArrayType::new(inner.element_type().clone(), inner.contains_null()).into(),
            ),
            c.args()[0].nullable || outer.contains_null(),
        ));
    }
    Err(DatatypeMismatchUnexpectedInputType {
        sql_expr: call_sql("flatten", c),
        param_index: ordinal(0).to_owned(),
        required_type: "\"ARRAY\" of \"ARRAY\"".to_owned(),
        input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
        input_type: quoted(c.data_type(0)),
    }
    .into())
}

/// `slice(array, start, length)`: the array's type.
fn slice(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let Some(array @ DataType::Array(_)) = c.data_type(0) else {
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql("slice", c),
            param_index: ordinal(0).to_owned(),
            required_type: "\"ARRAY\"".to_owned(),
            input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            input_type: quoted(c.data_type(0)),
        }
        .into());
    };
    Ok(ArgType::new(array.clone(), c.any_nullable()))
}

/// `array_join(array, delimiter[, nullReplacement])`: a string, NULL for a
/// NULL array or delimiter.
fn join(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, c.any_nullable()))
}

/// `array_join`'s inputs: an array of strings, a delimiter and, when given,
/// the NULL replacement.
fn join_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let strings = DataType::Array(ArrayType::new(DataType::String, true).into());
    // Spark casts atomic elements to strings; arrays, maps and structs are
    // refused rather than cast.
    let nested = matches!(
        c.data_type(0),
        Some(DataType::Array(a))
            if matches!(a.element_type(), DataType::Array(_) | DataType::Map(_) | DataType::Struct(_))
    );
    let mut inputs = vec![
        if nested {
            InputType::Strict(strings)
        } else {
            InputType::Exact(strings)
        },
        InputType::Strict(DataType::String),
    ];
    if c.args().len() > 2 {
        inputs.push(InputType::Strict(DataType::String));
    }
    inputs
}

/// The rank of an integral type, widest last.
fn rank(t: &DataType) -> Option<u8> {
    match t {
        DataType::Byte => Some(0),
        DataType::Short => Some(1),
        DataType::Integer => Some(2),
        DataType::Long => Some(3),
        _ => None,
    }
}

/// The widest integral type among the typed arguments (`INT` when all are
/// untyped `NULL`), or `None` when an argument is not integral.
fn sequence_type(c: &CallTypes<'_>) -> Option<DataType> {
    let mut widest: Option<DataType> = None;
    for i in 0..c.args().len() {
        let Some(t) = c.data_type(i) else { continue };
        let r = rank(t)?;
        if widest.as_ref().is_none_or(|w| rank(w) < Some(r)) {
            widest = Some(t.clone());
        }
    }
    Some(widest.unwrap_or(DataType::Integer))
}

/// `sequence`'s inputs: every argument is cast to the widest integral type.
fn sequence_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    match sequence_type(c) {
        Some(t) => vec![InputType::Exact(t); c.args().len()],
        None => vec![InputType::Any; c.args().len()],
    }
}

/// `sequence(start, stop[, step])` over integral types: `array<T>` of the
/// widest type, without NULL elements.
fn sequence(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let n = c.args().len();
    if !(2..=3).contains(&n) {
        return Err(WrongNumArgsWithoutSuggestion {
            function_name: "`sequence`".to_owned(),
            expected_num: "[2, 3]".to_owned(),
            actual_num: n.to_string(),
            docroot: "https://spark.apache.org/docs/latest".to_owned(),
        }
        .into());
    }
    if let Some(t) = sequence_type(c) {
        return Ok(ArgType::new(
            DataType::Array(ArrayType::new(t, false).into()),
            c.any_nullable(),
        ));
    }
    let temporal = |t: Option<&DataType>| {
        matches!(
            t,
            Some(DataType::Date | DataType::Timestamp | DataType::TimestampNtz)
        )
    };
    if temporal(c.data_type(0))
        || temporal(c.data_type(1))
        || matches!(c.data_type(0), Some(DataType::String))
        || matches!(c.data_type(1), Some(DataType::String))
    {
        return Err(KrishivUnsupportedFeature {
            feature: "sequence over dates, timestamps or strings".to_owned(),
        }
        .into());
    }
    Err(DatatypeMismatchSequenceWrongInputTypes {
        sql_expr: call_sql("sequence", c),
        function_name: "`sequence`".to_owned(),
        start_type: "(\"TIMESTAMP\" or \"TIMESTAMP_NTZ\" or \"DATE\")".to_owned(),
        step_type: "(\"INTERVAL\" or \"INTERVAL YEAR TO MONTH\" or \"INTERVAL DAY TO SECOND\")"
            .to_owned(),
        other_start_type: "\"INTEGRAL\"".to_owned(),
    }
    .into())
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(&["flatten"], &Inputs::Any, flatten);
    out.extend(scalar(
        &["slice"],
        &Inputs::Expects(vec![
            InputType::Array,
            InputType::Exact(DataType::Integer),
            InputType::Exact(DataType::Integer),
        ]),
        slice,
    ));
    out.extend(scalar(
        &["array_join"],
        &Inputs::Built(InputBuilder(join_inputs)),
        join,
    ));
    out.extend(scalar(
        &["sequence"],
        &Inputs::Built(InputBuilder(sequence_inputs)),
        sequence,
    ));
    out
}
