// SPDX-License-Identifier: Apache-2.0
//! The element-access signatures: `array_size`, `get`, `element_at` and
//! `try_element_at` (Spark 4.2's `ArraySize`, `GetArrayItem`-based `get`,
//! `ElementAt` and `TryElementAt`). Checked against the `array.sql`,
//! `map.sql` and `try_element_at.sql` goldens.
//!
//! `array_size` casts an untyped `NULL` to an array (and is NULL for it);
//! `get` takes a real array only (an untyped `NULL` is `VOID`, refused) and
//! an `INT` index; `element_at` takes an array and an `INT` index, or a map
//! and a key of the map's key type.

use krishiv_common::error::{
    DatatypeMismatchUnexpectedInputType, InvalidExtractBaseFieldType, KrishivError,
};
use krishiv_types::DataType;

use super::defs::{Spec, exact, scalar};
use super::hof::{call_sql, ordinal, quoted};
use crate::signature::{ArgType, CallTypes, InputBuilder, InputType, Inputs};

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

/// `array_size(array)`: the number of elements, NULL for NULL.
fn array_size(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if !matches!(c.data_type(0), Some(DataType::Array(_))) {
        return Err(unexpected("array_size", c, 0, "\"ARRAY\""));
    }
    Ok(ArgType::new(DataType::Integer, c.any_nullable()))
}

/// The element of the array argument, or the `UNEXPECTED_INPUT_TYPE` error.
fn element_of(name: &str, c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    match c.data_type(0) {
        Some(DataType::Array(a)) => Ok(ArgType::new(a.element_type().clone(), true)),
        _ => Err(unexpected(name, c, 0, "\"ARRAY\"")),
    }
}

/// `get(array, index)`: the element, NULL out of range.
fn get(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    element_of("get", c)
}

/// `element_at` / `try_element_at`: the array element or the map value.
fn element_at_named(name: &'static str, c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    match c.data_type(0) {
        Some(DataType::Map(m)) => Ok(ArgType::new(m.value_type().clone(), true)),
        Some(DataType::Array(_)) => element_of(name, c),
        _ => Err(unexpected(name, c, 0, "(\"ARRAY\" or \"MAP\")")),
    }
}

fn element_at(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    element_at_named("element_at", c)
}

fn try_element_at(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    element_at_named("try_element_at", c)
}

/// `base[index]` (the lowering of a subscript): the array element or the map
/// value, NULL when absent. A struct base never gets here: the analyzer
/// reads it as a field.
fn subscript(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    // An array index must be integral (Spark's `GetArrayItem`); a string,
    // fraction or boolean is not cast to one.
    if matches!(c.data_type(0), Some(DataType::Array(_)))
        && !matches!(
            c.data_type(1),
            None | Some(
                DataType::Null
                    | DataType::Byte
                    | DataType::Short
                    | DataType::Integer
                    | DataType::Long
            )
        )
    {
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: format!(
                "\"{}[{}]\"",
                c.sql(0).unwrap_or_default(),
                c.sql(1).unwrap_or_default()
            ),
            param_index: ordinal(1).to_owned(),
            required_type: "\"INTEGRAL\"".to_owned(),
            input_sql: format!("\"{}\"", c.sql(1).unwrap_or_default()),
            input_type: quoted(c.data_type(1)),
        }
        .into());
    }
    match c.data_type(0) {
        Some(DataType::Map(m)) => Ok(ArgType::new(m.value_type().clone(), true)),
        Some(DataType::Array(a)) => Ok(ArgType::new(a.element_type().clone(), true)),
        other => Err(InvalidExtractBaseFieldType {
            base: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            other: quoted(other),
        }
        .into()),
    }
}

/// The second argument: the map's key type, else an `INT` index.
fn element_at_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let second = match c.data_type(0) {
        Some(DataType::Map(m)) => exact(m.key_type().clone()),
        _ => exact(DataType::Integer),
    };
    vec![InputType::Any, second]
}

/// `base[index]`: as `element_at`, but a non-integral array index stays
/// as it is for the result rule to refuse.
fn subscript_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let integral = matches!(
        c.data_type(1),
        None | Some(
            DataType::Null | DataType::Byte | DataType::Short | DataType::Integer | DataType::Long
        )
    );
    if matches!(c.data_type(0), Some(DataType::Array(_))) && !integral {
        return vec![InputType::Any, InputType::Any];
    }
    element_at_inputs(c)
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(
        &["array_size"],
        &Inputs::Expects(vec![InputType::Array]),
        array_size,
    );
    out.extend(scalar(
        &["get"],
        &Inputs::Expects(vec![InputType::Any, exact(DataType::Integer)]),
        get,
    ));
    let built = Inputs::Built(InputBuilder(element_at_inputs));
    let sub = Inputs::Built(InputBuilder(subscript_inputs));
    out.extend(scalar(&["__subscript"], &sub, subscript));
    out.extend(scalar(&["element_at"], &built, element_at));
    out.extend(scalar(&["try_element_at"], &built, try_element_at));
    out
}
