// SPDX-License-Identifier: Apache-2.0
//! The analysis signatures of the higher-order functions over arrays and
//! maps (`transform`, `filter`, `exists`, `forall`, `aggregate`/`reduce`,
//! `zip_with`, `map_filter`, `map_zip_with`, `transform_keys`,
//! `transform_values` and `array_sort`) and the `array` constructor they
//! need (Spark's `ArrayHigherOrderFunction`s and `CreateArray`).
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

use krishiv_common::error::{
    DatatypeMismatchInvalidMapKeyType, DatatypeMismatchInvalidOrderingType,
    DatatypeMismatchMapZipWithDiffTypes, DatatypeMismatchUnexpectedInputType,
    DatatypeMismatchUnexpectedReturnType, InternalError, KrishivError,
};
use krishiv_types::{ArrayType, DataType, MapType};

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
///
/// [`bound_call_sql`] quotes the bound call instead (lambdas with Spark's
/// `namedlambdavariable()` variables), as quotes of a lambda argument and
/// output names show it.
pub(super) fn call_sql(name: &str, c: &CallTypes<'_>) -> String {
    let args: Vec<String> = (0..c.args().len())
        .map(|i| c.sql(i).unwrap_or_default())
        .collect();
    format!("\"{name}({})\"", args.join(", "))
}

/// The call's SQL with bound lambdas; `extra` appends a final argument
/// (the identity `finish` of a three-argument `aggregate`/`reduce`).
fn bound_call_sql(name: &str, c: &CallTypes<'_>, extra: Option<&str>) -> String {
    let mut args: Vec<String> = (0..c.args().len())
        .map(|i| c.bound_sql(i).unwrap_or_default())
        .collect();
    if let Some(last) = extra {
        args.push(last.to_owned());
    }
    format!("\"{name}({})\"", args.join(", "))
}

/// The identity `finish` Spark's `ArrayAggregate` constructor fills in for
/// a three-argument `aggregate`/`reduce`, as quotes show it.
const IDENTITY_FINISH: &str = "lambdafunction(namedlambdavariable(), namedlambdavariable())";

/// A type as Spark's message parameters quote it (`"VOID"` for untyped
/// `NULL`).
pub(super) fn quoted(t: Option<&DataType>) -> String {
    format!("\"{}\"", t.map_or_else(|| "VOID".to_owned(), DataType::sql))
}

/// The array argument's element type, `containsNull` and nullability, or the
/// `UNEXPECTED_INPUT_TYPE` error quoting it (Spark's
/// `checkInputDataTypes`). The arity check runs before typing, so a
/// missing argument is an internal error.
pub(super) fn array_argument(
    name: &str,
    c: &CallTypes<'_>,
) -> Result<(DataType, bool, bool), KrishivError> {
    array_argument_at(name, c, 0)
}

/// The array argument at `index` (see [`array_argument`]).
fn array_argument_at(
    name: &str,
    c: &CallTypes<'_>,
    index: usize,
) -> Result<(DataType, bool, bool), KrishivError> {
    let Some(arg) = c.args().get(index) else {
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
            param_index: ordinal(index).to_owned(),
            required_type: "\"ARRAY\"".to_owned(),
            input_sql: format!("\"{}\"", c.sql(index).unwrap_or_default()),
            input_type: quoted(other.as_ref()),
        }
        .into()),
    }
}

/// The lambda body's type and nullability: the body's, or `NULL` when it
/// is absent (an unbound body reads nothing).
fn body_type(c: &CallTypes<'_>, index: usize) -> (DataType, bool) {
    let body = c.args().get(index);
    (
        body.and_then(|a| a.data_type.clone())
            .unwrap_or(DataType::Null),
        body.is_none_or(|a| a.nullable),
    )
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
    let (element, contains_null) = body_type(c, 1);
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

/// The map argument's key and value types, or the `UNEXPECTED_INPUT_TYPE`
/// error quoting it (Spark's `checkInputDataTypes` over the abstract
/// `MapType`).
pub(super) fn map_argument(
    name: &str,
    c: &CallTypes<'_>,
    index: usize,
) -> Result<MapType, KrishivError> {
    let arg = c.args().get(index);
    match arg.and_then(|a| a.data_type.as_ref()) {
        Some(DataType::Map(map)) => Ok(map.as_ref().clone()),
        other => Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql(name, c),
            param_index: ordinal(index).to_owned(),
            required_type: "\"MAP\"".to_owned(),
            input_sql: format!("\"{}\"", c.sql(index).unwrap_or_default()),
            input_type: quoted(other),
        }
        .into()),
    }
}

/// Spark's `ordinalNumber`: `first`, `second`, `third`, else `Nth`.
pub(super) fn ordinal(index: usize) -> &'static str {
    match index {
        0 => "first",
        1 => "second",
        2 => "third",
        _ => "nth",
    }
}

/// `aggregate(arr, zero, merge[, finish])` and `reduce`: the fold over the
/// array. The accumulator type is the type of `zero` (Spark's
/// `ArrayAggregate`: the merge body's type must equal it structurally,
/// ignoring nullability, or the `UNEXPECTED_INPUT_TYPE` error quotes the
/// merge; the result is the finish body's type, or the accumulator's when
/// `finish` is absent). The accumulator is always nullable, so the result
/// is nullable when the finish body is.
fn aggregate_result(name: &str, c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let (_, _, nullable) = array_argument(name, c)?;
    let zero = c.args().get(1).and_then(|a| a.data_type.clone());
    let merge = c.args().get(2);
    let merge_ty = merge.and_then(|a| a.data_type.clone());
    match (&zero, &merge_ty) {
        (Some(z), Some(m)) if z.equals_structurally(m, true) => {}
        _ => {
            // The merge quotes bound (see the module doc); a missing
            // `finish` still shows as the identity lambda Spark fills in.
            let extra = (c.args().len() == 3).then_some(IDENTITY_FINISH);
            return Err(DatatypeMismatchUnexpectedInputType {
                sql_expr: bound_call_sql(name, c, extra),
                param_index: "third".to_owned(),
                required_type: quoted(zero.as_ref()),
                input_sql: c.bound_sql(2).unwrap_or_default(),
                input_type: quoted(merge_ty.as_ref()),
            }
            .into());
        }
    }
    let finish = c.args().get(3);
    let (data_type, finish_nullable) = match finish.and_then(|a| a.data_type.clone()) {
        Some(t) => {
            let nullable = finish.is_some_and(|a| a.nullable);
            (t, nullable)
        }
        // No `finish`: the identity lambda over the accumulator, which is
        // always nullable.
        None => (zero.unwrap_or(DataType::Null), true),
    };
    Ok(ArgType::new(data_type, nullable || finish_nullable))
}

/// `zip_with(a1, a2, (x, y) -> ...)`: the merged array of the body's type
/// (the shorter array is padded with NULLs, so both lambda variables are
/// always nullable).
fn zip_with_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let (_, _, nullable) = array_argument("zip_with", c)?;
    let (_, _, second_nullable) = array_argument_at("zip_with", c, 1)?;
    let (element, contains_null) = body_type(c, 2);
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(element, contains_null).into()),
        nullable || second_nullable,
    ))
}

/// `map_filter(m, (k, v) -> ...)`: the map itself, with a boolean body.
fn map_filter_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let map = map_argument("map_filter", c, 0)?;
    let nullable = c.args().first().is_some_and(|a| a.nullable);
    let _ = boolean_body("map_filter", c)?;
    Ok(ArgType::new(DataType::Map(map.into()), nullable))
}

/// Spark's `TypeUtils.checkForMapKeyType`: only a key type holding a map
/// (or variant, which never reaches here) is refused.
pub(super) fn check_key_type(
    name: &str,
    c: &CallTypes<'_>,
    key: &DataType,
) -> Result<(), KrishivError> {
    if holds_map(key) {
        return Err(DatatypeMismatchInvalidMapKeyType {
            sql_expr: bound_call_sql(name, c, None),
            key_type: quoted(Some(key)),
        }
        .into());
    }
    Ok(())
}

/// Whether `t` is or holds a map type.
pub(super) fn holds_map(t: &DataType) -> bool {
    match t {
        DataType::Map(_) => true,
        DataType::Array(a) => holds_map(a.element_type()),
        DataType::Struct(s) => s.fields().iter().any(|f| holds_map(f.data_type())),
        _ => false,
    }
}

/// `transform_keys(m, (k, v) -> ...)`: a map of the body's key type over
/// the map's values (duplicate keys raise at run time).
fn transform_keys_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let map = map_argument("transform_keys", c, 0)?;
    let nullable = c.args().first().is_some_and(|a| a.nullable);
    let (key, _) = body_type(c, 1);
    check_key_type("transform_keys", c, &key)?;
    Ok(ArgType::new(
        DataType::Map(
            MapType::new(key, map.value_type().clone(), map.value_contains_null()).into(),
        ),
        nullable,
    ))
}

/// `transform_values(m, (k, v) -> ...)`: a map of the map's keys over the
/// body's value type.
fn transform_values_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let map = map_argument("transform_values", c, 0)?;
    let nullable = c.args().first().is_some_and(|a| a.nullable);
    let (value, value_nullable) = body_type(c, 1);
    Ok(ArgType::new(
        DataType::Map(MapType::new(map.key_type().clone(), value, value_nullable).into()),
        nullable,
    ))
}

/// Whether `t` orders (Spark's `RowOrdering.isOrderable` over the types
/// that reach here: a map orders nowhere, and neither does an array or
/// struct holding one).
pub(super) fn orderable(t: &DataType) -> bool {
    match t {
        DataType::Map(_) => false,
        DataType::Array(a) => orderable(a.element_type()),
        DataType::Struct(s) => s.fields().iter().all(|f| orderable(f.data_type())),
        _ => true,
    }
}

/// `map_zip_with(m1, m2, (k, v1, v2) -> ...)`: the union of the key sets in
/// first-seen order, with missing values NULL. The key types must be the
/// same (ignoring nullability) and orderable, as Spark's
/// `checkInputDataTypes` requires.
fn map_zip_with_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let first = map_argument("map_zip_with", c, 0)?;
    let second = map_argument("map_zip_with", c, 1)?;
    let nullable = c.args().iter().take(2).any(|a| a.nullable);
    if !first
        .key_type()
        .equals_structurally(second.key_type(), true)
    {
        return Err(DatatypeMismatchMapZipWithDiffTypes {
            sql_expr: bound_call_sql("map_zip_with", c, None),
            function_name: "`map_zip_with`".to_owned(),
            left_type: quoted(Some(first.key_type())),
            right_type: quoted(Some(second.key_type())),
        }
        .into());
    }
    if !orderable(first.key_type()) {
        return Err(DatatypeMismatchInvalidOrderingType {
            sql_expr: bound_call_sql("map_zip_with", c, None),
            function_name: "`map_zip_with`".to_owned(),
            data_type: quoted(Some(first.key_type())),
        }
        .into());
    }
    let (value, value_nullable) = body_type(c, 2);
    Ok(ArgType::new(
        DataType::Map(MapType::new(first.key_type().clone(), value, value_nullable).into()),
        nullable,
    ))
}

/// `array_sort(arr[, (l, r) -> ...])`: the array itself. The comparator
/// body must be `INT`; without one the array sorts in ascending order with
/// NULLs last (Spark's default comparator), which needs an orderable
/// element type.
fn array_sort_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let (element, contains_null, nullable) = array_argument("array_sort", c)?;
    if c.args().len() > 1 {
        let body = c.args().get(1);
        match body.and_then(|a| a.data_type.as_ref()) {
            Some(DataType::Integer) => {}
            other => {
                return Err(DatatypeMismatchUnexpectedReturnType {
                    sql_expr: bound_call_sql("array_sort", c, None),
                    function_name: "`lambdafunction`".to_owned(),
                    expected_type: "\"INT\"".to_owned(),
                    actual_type: quoted(other),
                }
                .into());
            }
        }
    } else if !orderable(&element) {
        return Err(DatatypeMismatchInvalidOrderingType {
            sql_expr: bound_call_sql("array_sort", c, None),
            function_name: "`array_sort`".to_owned(),
            data_type: quoted(Some(&element)),
        }
        .into());
    }
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(element, contains_null).into()),
        nullable,
    ))
}

/// A higher-order signature: `Scalar` kind, `Any` inputs (the analyzer
/// binds the lambdas, so there is nothing to coerce) and `HigherOrder`
/// execution.
fn higher(name: &'static str, result: fn(&CallTypes<'_>) -> Result<ArgType, KrishivError>) -> Spec {
    Spec {
        name,
        kind: FunctionKind::Scalar,
        inputs: Inputs::Any,
        result,
        execution: Execution::HigherOrder,
    }
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
        higher("transform", transform_result),
        higher("filter", filter_result),
        higher("exists", exists_result),
        higher("forall", forall_result),
        higher("aggregate", |c| aggregate_result("aggregate", c)),
        higher("reduce", |c| aggregate_result("reduce", c)),
        higher("zip_with", zip_with_result),
        higher("map_filter", map_filter_result),
        higher("map_zip_with", map_zip_with_result),
        higher("transform_keys", transform_keys_result),
        higher("transform_values", transform_values_result),
        higher("array_sort", array_sort_result),
    ]
}
