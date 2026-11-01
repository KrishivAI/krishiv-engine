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

use krishiv_common::error::{DatatypeMismatchUnexpectedInputType, KrishivError};
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
                ArrayType::new(
                    inner.element_type().clone(),
                    outer.contains_null() || inner.contains_null(),
                )
                .into(),
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
        InputType::Exact(DataType::String),
    ];
    if c.args().len() > 2 {
        inputs.push(InputType::Exact(DataType::String));
    }
    inputs
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
    out
}
