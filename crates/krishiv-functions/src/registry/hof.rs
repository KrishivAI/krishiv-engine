// SPDX-License-Identifier: Apache-2.0
//! The analysis signatures of the higher-order functions over arrays
//! (`transform`, `filter`, `exists`, `forall`) and the `array`
//! constructor they need (Spark's `ArrayHigherOrderFunction`s and
//! `CreateArray`).
//!
//! The analyzer binds a lambda's variables before the generic path types
//! the call (see `krishiv-plan`'s `hof`): at a lambda position the
//! argument's type is its body's, so these rules read the body type from
//! the argument. The executor compiles the bodies (see `krishiv-exec`'s
//! `hof`); only `array` binds a kernel here.
//!
//! `array` is the minimal constructor this family needs (all arguments at
//! their common type, as Spark's `CreateArray` types them). The collection
//! functions own its completion (nested coercion edge cases).

use krishiv_common::error::{DatatypeMismatchUnexpectedInputType, InternalError, KrishivError};
use krishiv_types::{ArrayType, DataType};

use super::Execution;
use super::defs::Spec;
use crate::signature::{ArgType, CallTypes, FunctionKind, Inputs};

/// `array(...)`: an array of the arguments' common type (Spark's
/// `CreateArray`; untyped `NULL` is `VOID`). The analyzer casts the
/// arguments to their common type first.
fn array_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let element = c.args().first().and_then(|a| a.data_type.clone());
    let nullable = c.args().iter().any(|a| a.nullable);
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(element.unwrap_or(DataType::Null), nullable).into()),
        false,
    ))
}

/// The call's SQL as Spark quotes it in errors: `name(arg, ...)`.
fn call_sql(name: &str, c: &CallTypes<'_>) -> String {
    let args: Vec<String> = (0..c.args().len())
        .map(|i| c.sql(i).unwrap_or_default())
        .collect();
    format!("\"{name}({})\"", args.join(", "))
}

/// A type as Spark's message parameters quote it (`"VOID"` for untyped
/// `NULL`).
fn quoted(t: Option<&DataType>) -> String {
    format!("\"{}\"", t.map_or_else(|| "VOID".to_owned(), DataType::sql))
}

/// The array argument's element type, `containsNull` and nullability, or
/// the `UNEXPECTED_INPUT_TYPE` error quoting it (Spark's
/// `checkInputDataTypes`). The arity check runs before typing, so a
/// missing argument is an internal error.
fn array_argument(name: &str, c: &CallTypes<'_>) -> Result<(DataType, bool, bool), KrishivError> {
    let Some(arg) = c.args().first() else {
        return Err(InternalError {
            message: format!("{name} typed without its array argument"),
        }
        .into());
    };
    match &arg.data_type {
        Some(DataType::Array(element)) => Ok((
            element.element_type().clone(),
            element.contains_null(),
            arg.nullable,
        )),
        other => Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql(name, c),
            param_index: "first".to_owned(),
            required_type: "\"ARRAY\"".to_owned(),
            input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            input_type: quoted(other.as_ref()),
        }
        .into()),
    }
}

/// Whether the lambda body (the second argument's type) is boolean, or the
/// `UNEXPECTED_INPUT_TYPE` error quoting it (Spark quotes the lambda,
/// constant lambdas wrapped over their hidden variable by the analyzer).
fn boolean_body(name: &str, c: &CallTypes<'_>) -> Result<bool, KrishivError> {
    let body = c.args().get(1);
    match body.and_then(|a| a.data_type.as_ref()) {
        Some(DataType::Boolean) => Ok(body.is_some_and(|a| a.nullable)),
        other => Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql(name, c),
            param_index: "second".to_owned(),
            required_type: "\"BOOLEAN\"".to_owned(),
            input_sql: c.sql(1).unwrap_or_default(),
            input_type: quoted(other),
        }
        .into()),
    }
}

/// `transform(arr, f)`: an array of the body's type.
fn transform_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let (_, _, nullable) = array_argument("transform", c)?;
    let body = c.args().get(1);
    let element = body
        .and_then(|a| a.data_type.clone())
        .unwrap_or(DataType::Null);
    let contains_null = body.is_none_or(|a| a.nullable);
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(element, contains_null).into()),
        nullable,
    ))
}

/// `filter(arr, p)`: the array itself, with a boolean body.
fn filter_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let (element, contains_null, nullable) = array_argument("filter", c)?;
    let _ = boolean_body("filter", c)?;
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(element, contains_null).into()),
        nullable,
    ))
}

/// `exists(arr, p)`: a boolean, with a boolean body. The result is
/// nullable when the array is or when the body is (a NULL mark decides
/// nothing, so the fold stays NULL). Spark derives nullability from the
/// array and body alone; `containsNull` is not part of the rule.
fn exists_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let (_, _, nullable) = array_argument("exists", c)?;
    let body_nullable = boolean_body("exists", c)?;
    Ok(ArgType::new(DataType::Boolean, nullable || body_nullable))
}

/// `forall(arr, p)`: a boolean, with a boolean body (same nullability
/// as `exists`: a NULL mark decides nothing either way).
fn forall_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let (_, _, nullable) = array_argument("forall", c)?;
    let body_nullable = boolean_body("forall", c)?;
    Ok(ArgType::new(DataType::Boolean, nullable || body_nullable))
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    vec![
        Spec {
            name: "array",
            kind: FunctionKind::Scalar,
            inputs: Inputs::Common,
            result: array_result,
            execution: Execution::Scalar,
        },
        Spec {
            name: "transform",
            kind: FunctionKind::Scalar,
            inputs: Inputs::Any,
            result: transform_result,
            execution: Execution::HigherOrder,
        },
        Spec {
            name: "filter",
            kind: FunctionKind::Scalar,
            inputs: Inputs::Any,
            result: filter_result,
            execution: Execution::HigherOrder,
        },
        Spec {
            name: "exists",
            kind: FunctionKind::Scalar,
            inputs: Inputs::Any,
            result: exists_result,
            execution: Execution::HigherOrder,
        },
        Spec {
            name: "forall",
            kind: FunctionKind::Scalar,
            inputs: Inputs::Any,
            result: forall_result,
            execution: Execution::HigherOrder,
        },
    ]
}
