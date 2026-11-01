// SPDX-License-Identifier: Apache-2.0
//! The array search signatures: `array_max`, `array_min`, `array_position`,
//! `array_remove` and `array_repeat` (Spark 4.2's `ArrayMax`, `ArrayMin`,
//! `ArrayPosition`, `ArrayRemove` and `ArrayRepeat`). Checked against a
//! Spark 4.2.0 run.
//!
//! The element type of all but `array_repeat` must order
//! (`INVALID_ORDERING_TYPE` names the element type). `array_position` and
//! `array_remove` take the value at the element's tightest common type
//! (`Inputs::ArrayItem`; an untyped `NULL` value takes the element type),
//! else `ARRAY_FUNCTION_DIFF_TYPES`; a first argument that is not an array
//! is `UNEXPECTED_INPUT_TYPE` for `array_position` (an untyped `NULL` is
//! `NULL_TYPE`) and `ARRAY_FUNCTION_DIFF_TYPES` for `array_remove`.
//! `array_repeat`'s count must already be an `INT`.

use krishiv_common::error::{
    DatatypeMismatchArrayFunctionDiffTypes, DatatypeMismatchInvalidOrderingType,
    DatatypeMismatchNullType, DatatypeMismatchUnexpectedInputType, KrishivError,
};
use krishiv_types::{ArrayType, DataType};

use super::defs::{Spec, scalar, spark_orderable};
use super::hof::{call_sql, ordinal, quoted};
use crate::signature::{ArgType, CallTypes, InputType, Inputs};

/// `INVALID_ORDERING_TYPE` for the element type `data_type`.
fn not_orderable(name: &str, c: &CallTypes<'_>, data_type: &DataType) -> KrishivError {
    DatatypeMismatchInvalidOrderingType {
        sql_expr: call_sql(name, c),
        function_name: format!("`{name}`"),
        data_type: quoted(Some(data_type)),
    }
    .into()
}

/// `array_max` and `array_min`: the extreme element, NULL for an empty
/// array or one of NULLs. An untyped `NULL` argument is NULL.
fn extreme(name: &str, c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let element = match c.data_type(0) {
        Some(DataType::Array(a)) => a.element_type().clone(),
        None => DataType::Null,
        Some(_) => {
            return Err(DatatypeMismatchUnexpectedInputType {
                sql_expr: call_sql(name, c),
                param_index: ordinal(0).to_owned(),
                required_type: "\"ARRAY\"".to_owned(),
                input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
                input_type: quoted(c.data_type(0)),
            }
            .into());
        }
    };
    if !spark_orderable(&element) {
        return Err(not_orderable(name, c, &element));
    }
    Ok(ArgType::new(element, true))
}

fn max(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    extreme("array_max", c)
}

fn min(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    extreme("array_min", c)
}

/// `ARRAY_FUNCTION_DIFF_TYPES` for the arguments of `name`.
fn different(name: &str, c: &CallTypes<'_>) -> KrishivError {
    DatatypeMismatchArrayFunctionDiffTypes {
        sql_expr: call_sql(name, c),
        function_name: format!("`{name}`"),
        data_type: "\"ARRAY\"".to_owned(),
        left_type: quoted(c.data_type(0)),
        right_type: quoted(c.data_type(1)),
    }
    .into()
}

/// The array of `array_position`/`array_remove` once the value met its
/// element type; the errors otherwise.
fn searched<'a>(name: &str, c: &CallTypes<'a>) -> Result<&'a ArrayType, KrishivError> {
    let array = match c.data_type(0) {
        Some(DataType::Array(a)) => a,
        None if name == "array_position" => {
            return Err(DatatypeMismatchNullType {
                sql_expr: call_sql(name, c),
                function_name: format!("`{name}`"),
            }
            .into());
        }
        other if name == "array_position" => {
            return Err(DatatypeMismatchUnexpectedInputType {
                sql_expr: call_sql(name, c),
                param_index: ordinal(0).to_owned(),
                required_type: "\"ARRAY\"".to_owned(),
                input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
                input_type: quoted(other),
            }
            .into());
        }
        _ => return Err(different(name, c)),
    };
    // An untyped NULL value took the element type in the cast.
    let met = c
        .data_type(1)
        .is_none_or(|v| array.element_type().equals_ignore_nullability(v));
    if !met {
        return Err(different(name, c));
    }
    if !spark_orderable(array.element_type()) {
        return Err(not_orderable(name, c, array.element_type()));
    }
    Ok(array)
}

/// `array_position(array, value)`: the 1-based index of the first match as
/// a BIGINT, `0` for none, NULL for a NULL array or value.
fn position(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    searched("array_position", c)?;
    Ok(ArgType::new(DataType::Long, c.any_nullable()))
}

/// `array_remove(array, value)`: the array without the elements equal to
/// `value`, NULL for a NULL array or value.
fn remove(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let array = searched("array_remove", c)?;
    Ok(ArgType::new(
        DataType::Array(array.clone().into()),
        c.any_nullable(),
    ))
}

/// `array_repeat(element, count)`: `count` copies of `element`, the array
/// holding NULL when the element can be NULL. A negative count is empty.
fn repeat(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let element = c.data_type(0).cloned().unwrap_or(DataType::Null);
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(element, c.args()[0].nullable).into()),
        c.args()[1].nullable,
    ))
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let array = || Inputs::Expects(vec![InputType::Array]);
    let mut out = scalar(&["array_max"], &array(), max);
    out.extend(scalar(&["array_min"], &array(), min));
    out.extend(scalar(&["array_position"], &Inputs::ArrayItem, position));
    out.extend(scalar(&["array_remove"], &Inputs::ArrayItem, remove));
    out.extend(scalar(
        &["array_repeat"],
        &Inputs::Expects(vec![InputType::Any, InputType::Strict(DataType::Integer)]),
        repeat,
    ));
    out
}
