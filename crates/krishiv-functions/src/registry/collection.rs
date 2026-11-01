// SPDX-License-Identifier: Apache-2.0
//! The analysis signatures of the collection functions the higher-order
//! functions need: `size` and `split` (Spark's `Size` and `StringSplit`).
//! The `map` constructor is in the map family.
//!
//! `split` fills its default `limit` of `-1` in the analyzer (see
//! `krishiv-plan`'s materialized defaults), so the rule always sees three
//! arguments.

use krishiv_common::error::{DatatypeMismatchUnexpectedInputType, KrishivError};
use krishiv_types::{ArrayType, DataType};

use super::defs::Spec;
use super::hof::{call_sql, quoted};
use crate::signature::{ArgType, CallTypes, FunctionKind, Inputs};

/// `size(expr)`: the number of elements of an array or map (Spark's
/// `Size`). A NULL is `-1` with ANSI off (Spark's `legacySizeOfNull`
/// default) and NULL with ANSI on, so the result is non-nullable with
/// ANSI off and as nullable as the argument otherwise.
fn size_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    collection_size("size", c.ansi(), c)
}

/// `cardinality(expr)`: another name of `size` (Spark 4.2 has no separate
/// class).
fn cardinality_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    collection_size("cardinality", c.ansi(), c)
}

fn collection_size(
    name: &str,
    null_is_null: bool,
    c: &CallTypes<'_>,
) -> Result<ArgType, KrishivError> {
    let arg = c.args().first();
    let nullable = match arg.and_then(|a| a.data_type.as_ref()) {
        Some(DataType::Array(_) | DataType::Map(_)) | None => {
            if null_is_null {
                arg.is_none_or(|a| a.nullable)
            } else {
                false
            }
        }
        other => {
            return Err(DatatypeMismatchUnexpectedInputType {
                sql_expr: call_sql(name, c),
                param_index: "first".to_owned(),
                required_type: "(\"ARRAY\" or \"MAP\")".to_owned(),
                input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
                input_type: quoted(other),
            }
            .into());
        }
    };
    Ok(ArgType::new(DataType::Integer, nullable))
}

/// `split(str, pattern[, limit])`: the pieces of `str` around the matches
/// of `pattern` (Spark's `StringSplit`). The analyzer fills the default
/// `limit` of `-1` and casts the arguments, so the rule only types the
/// result: an array of non-null strings, nullable when an argument is.
pub(super) fn split_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(DataType::String, false).into()),
        c.any_nullable(),
    ))
}

/// A collection signature: `Scalar` kind and execution (the analyzer
/// coerces to `split`'s inputs; `map` and `size` take none).
fn spec(
    name: &'static str,
    inputs: Inputs,
    result: fn(&CallTypes<'_>) -> Result<ArgType, KrishivError>,
) -> Spec {
    Spec {
        name,
        kind: FunctionKind::Scalar,
        inputs,
        result,
        execution: super::Execution::Scalar,
    }
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    vec![
        spec("cardinality", Inputs::Any, cardinality_result),
        spec("size", Inputs::Any, size_result),
        spec(
            "split",
            Inputs::Expects(vec![
                super::defs::exact(DataType::String),
                super::defs::exact(DataType::String),
                super::defs::exact(DataType::Integer),
            ]),
            split_result,
        ),
    ]
}
