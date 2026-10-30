// SPDX-License-Identifier: Apache-2.0
//! The array-editing signatures: `array_append`, `array_prepend`,
//! `array_insert` and `array_compact` (Spark 4.2's `ArrayAppend`,
//! `ArrayPrepend`, `ArrayInsert` and `ArrayCompact`). Checked against the
//! `array.sql` goldens and a Spark 4.2.0 run.
//!
//! The item must meet the array's element type (`Inputs::ArrayItem` casts
//! both to it). A first argument that is not an array is
//! `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` for `array_append` and
//! `array_prepend` and `DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES` for
//! `array_insert`, which also gives the latter for an item that does not meet
//! the element type. The result
//! array may hold NULL when the array or the item may.

use krishiv_common::error::{
    DatatypeMismatchArrayFunctionDiffTypes, DatatypeMismatchUnexpectedInputType, KrishivError,
};
use krishiv_types::{ArrayType, DataType};

use super::defs::{Spec, scalar};
use super::hof::{call_sql, ordinal, quoted};
use crate::signature::{ArgType, CallTypes, InputType, Inputs};

/// The result of an append, prepend or insert: the array type, NULL when the
/// array (or, for an insert, the position) is NULL.
fn grow(name: &'static str, c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let item = c.args().len() - 1;
    // Spark types an append or prepend by `ImplicitCastInputTypes`: a first
    // argument that is not an array is an input-type error; `array_insert`
    // reports the shape mismatch instead, and checks its position's type.
    if name != "array_insert" && !matches!(c.data_type(0), Some(DataType::Array(_))) {
        return Err(unexpected(name, c, 0, "\"ARRAY\""));
    }
    if name == "array_insert"
        && c.data_type(1) != Some(&DataType::Integer)
        && matches!(c.data_type(0), Some(DataType::Array(_)))
    {
        return Err(unexpected(name, c, 1, "\"INT\""));
    }
    let element = match (c.data_type(0), c.data_type(item)) {
        (Some(DataType::Array(a)), Some(t)) if a.element_type().equals_ignore_nullability(t) => {
            Some(a.element_type().clone())
        }
        // An untyped NULL item took the element type in the cast.
        (Some(DataType::Array(a)), None) => Some(a.element_type().clone()),
        _ => None,
    };
    let (Some(element), Some(DataType::Array(a))) = (element, c.data_type(0)) else {
        return Err(DatatypeMismatchArrayFunctionDiffTypes {
            data_type: "\"ARRAY\"".to_owned(),
            function_name: format!("`{name}`"),
            left_type: quoted(c.data_type(0)),
            right_type: quoted(c.data_type(item)),
            sql_expr: call_sql(name, c),
        }
        .into());
    };
    let contains_null = a.contains_null() || c.args()[item].nullable;
    let nullable = c.args()[0].nullable || (item == 2 && c.args()[1].nullable);
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(element, contains_null).into()),
        nullable,
    ))
}

fn append(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    grow("array_append", c)
}

fn prepend(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    grow("array_prepend", c)
}

fn insert(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    grow("array_insert", c)
}

/// The `UNEXPECTED_INPUT_TYPE` error of argument `index`.
fn unexpected(name: &str, c: &CallTypes<'_>, index: usize, required: &str) -> KrishivError {
    DatatypeMismatchUnexpectedInputType {
        sql_expr: call_sql(name, c),
        param_index: ordinal(index).to_owned(),
        required_type: required.to_owned(),
        input_sql: format!("\"{}\"", c.sql(index).unwrap_or_default()),
        input_type: quoted(c.data_type(index)),
    }
    .into()
}

/// `array_compact(array)`: the array without its NULLs.
fn compact(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let Some(DataType::Array(a)) = c.data_type(0) else {
        return Err(unexpected("array_compact", c, 0, "\"ARRAY\""));
    };
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(a.element_type().clone(), false).into()),
        c.args()[0].nullable,
    ))
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(&["array_append"], &Inputs::ArrayItem, append);
    out.extend(scalar(&["array_prepend"], &Inputs::ArrayItem, prepend));
    out.extend(scalar(&["array_insert"], &Inputs::ArrayItem, insert));
    out.extend(scalar(
        &["array_compact"],
        &Inputs::Expects(vec![InputType::Array]),
        compact,
    ));
    out
}
