// SPDX-License-Identifier: Apache-2.0
//! The analysis signatures of the collection functions the higher-order
//! functions need: the `map` constructor, `size` and `split` (Spark's
//! `CreateMap`, `Size` and `StringSplit`).
//!
//! `map` takes no implicit casts (Spark's `CreateMap` checks the written
//! types): keys and values must already share a type. `split` fills its
//! default `limit` of `-1` in the analyzer (see `krishiv-plan`'s materialized
//! defaults), so the rule always sees three arguments.

use krishiv_common::error::{
    DatatypeMismatchCreateMapKeyDiffTypes, DatatypeMismatchCreateMapValueDiffTypes,
    DatatypeMismatchUnexpectedInputType, KrishivError, WrongNumArgsWithoutSuggestion,
};
use krishiv_types::{ArrayType, DataType, MapType};

use super::defs::Spec;
use super::hof::{call_sql, quoted};
use crate::signature::{ArgType, CallTypes, FunctionKind, Inputs};

/// Spark's documentation root, as the arity error cites it.
const SPARK_DOC_ROOT: &str = "https://spark.apache.org/docs/latest";

/// `map(k0, v0, ...)`: a map of the keys' common type to the values'
/// common type (Spark's `CreateMap`; the empty map is `MAP<VOID, VOID>`).
/// Keys and values must each share a type already, and the key type must
/// be usable as a map key.
fn map_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let n = c.args().len();
    if !n.is_multiple_of(2) {
        return Err(WrongNumArgsWithoutSuggestion {
            function_name: "`map`".to_owned(),
            expected_num: "2n (n > 0)".to_owned(),
            actual_num: n.to_string(),
            docroot: SPARK_DOC_ROOT.to_owned(),
        }
        .into());
    }
    let keys: Vec<Option<&DataType>> = (0..n).step_by(2).map(|i| c.data_type(i)).collect();
    let values: Vec<Option<&DataType>> = (1..n).step_by(2).map(|i| c.data_type(i)).collect();
    let same = |ts: &[Option<&DataType>]| {
        ts.iter()
            .filter_map(|t| *t)
            .collect::<Vec<_>>()
            .windows(2)
            .all(|w| w[0].equals_structurally(w[1], true))
    };
    if !same(&keys) {
        return Err(DatatypeMismatchCreateMapKeyDiffTypes {
            sql_expr: call_sql("map", c),
            function_name: "`map`".to_owned(),
            data_type: format!(
                "[{}]",
                keys.iter()
                    .map(|t| t.map_or_else(|| "VOID".to_owned(), DataType::sql))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
        .into());
    }
    if !same(&values) {
        return Err(DatatypeMismatchCreateMapValueDiffTypes {
            sql_expr: call_sql("map", c),
            function_name: "`map`".to_owned(),
            data_type: format!(
                "[{}]",
                values
                    .iter()
                    .map(|t| t.map_or_else(|| "VOID".to_owned(), DataType::sql))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
        .into());
    }
    let key = keys
        .iter()
        .find_map(|t| (*t).cloned())
        .unwrap_or(DataType::Null);
    let value = values
        .iter()
        .find_map(|t| (*t).cloned())
        .unwrap_or(DataType::Null);
    super::hof::check_key_type("map", c, &key)?;
    let value_nullable = c.args().iter().skip(1).step_by(2).any(|a| a.nullable);
    Ok(ArgType::new(
        DataType::Map(MapType::new(key, value, value_nullable).into()),
        false,
    ))
}

/// `size(expr)`: the number of elements of an array or map (Spark's
/// `Size`). A NULL is `-1` with ANSI off (Spark's `legacySizeOfNull`
/// default) and NULL with ANSI on, so the result is non-nullable with
/// ANSI off and as nullable as the argument otherwise.
fn size_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let arg = c.args().first();
    let nullable = match arg.and_then(|a| a.data_type.as_ref()) {
        Some(DataType::Array(_) | DataType::Map(_)) | None => {
            if c.ansi() {
                arg.is_none_or(|a| a.nullable)
            } else {
                false
            }
        }
        other => {
            return Err(DatatypeMismatchUnexpectedInputType {
                sql_expr: call_sql("size", c),
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
fn split_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
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
        spec("map", Inputs::Any, map_result),
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
