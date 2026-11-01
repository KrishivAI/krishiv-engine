// SPDX-License-Identifier: Apache-2.0
//! The analysis signatures of the JSON functions: `from_json` and
//! `to_json` (nested documents), `schema_of_json` (inference),
//! `get_json_object` (JSONPath), `json_array_length` and `json_object_keys`
//! (structural queries).
//!
//! Spark checks these functions' arguments itself instead of coercing
//! them: `from_json` reports the first argument's mismatch against a call
//! that names only that argument, the schema must be a non-null string
//! literal (anything else is `INVALID_SCHEMA.NON_STRING_LITERAL`), the
//! options must be a non-null `MAP<STRING, STRING>` literal, and `to_json`
//! only takes structs, maps and arrays (`INVALID_JSON_SCHEMA`
//! otherwise). `schema_of_json` takes any JSON text: a non-string fails at
//! run time in Spark, so analysis only rejects a null literal and a
//! non-foldable argument.

use krishiv_common::error::{
    DatatypeMismatchInvalidJsonSchema, DatatypeMismatchNonFoldableInput,
    DatatypeMismatchNonStringType, DatatypeMismatchUnexpectedInputType,
    DatatypeMismatchUnexpectedNull, InvalidOptionsNonMapFunction, InvalidOptionsNonStringType,
    InvalidSchemaNonStringLiteral, KrishivError,
};
use krishiv_types::{ArrayType, DataType, StructField, StructType, parse_data_type};

use super::Execution;
use super::defs::Spec;
use super::defs::{exact, scalar};
use crate::signature::{ArgType, CallTypes, Constant, FunctionKind, InputType, Inputs};

/// Every JSON function: the schema reads as Spark's DDL (a bare field list
/// is wrapped in `struct<...>`), the options as Spark's JSON data source
/// reads them.
pub(super) fn specs() -> Vec<Spec> {
    // `LengthOfJsonArray` and `JsonObjectKeys` are `ExpectsInputTypes`: a
    // non-string is a type mismatch, never an implicit cast.
    let strict = Inputs::Expects(vec![InputType::Strict(DataType::String)]);
    let mut out = scalar(&["json_array_length"], &strict, int_result);
    out.extend(scalar(&["json_object_keys"], &strict, array_result));
    out.extend(scalar(
        &["get_json_object"],
        &Inputs::Expects(vec![exact(DataType::String), exact(DataType::String)]),
        string_result,
    ));
    out.extend(scalar(&["schema_of_json"], &Inputs::Any, schema_of_json));
    out.extend(scalar(&["to_json"], &Inputs::Any, to_json));
    out.extend(scalar(&["from_json"], &Inputs::Any, from_json));
    out.push(Spec {
        name: "json_tuple",
        kind: FunctionKind::Generator,
        inputs: Inputs::Any,
        result: json_tuple,
        execution: Execution::Scalar,
    });
    out
}

/// `json_array_length`: an `INT`, always nullable.
fn int_result(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Integer, true))
}

/// `json_object_keys`: an `ARRAY<STRING>` of non-null keys, always
/// nullable.
fn array_result(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        DataType::Array(ArrayType::new(DataType::String, false).into()),
        true,
    ))
}

/// `get_json_object`: a `STRING`, always nullable.
fn string_result(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, true))
}

/// `schema_of_json`: the JSON text must be foldable (a null literal is an
/// error, a non-foldable argument is an error, anything else is read at
/// bind time); the result is a `STRING`.
fn schema_of_json(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if let Some(Constant::Error(e)) = c.constant(0) {
        return Err(e.clone());
    }
    if matches!(c.constant(0), Some(Constant::Null)) {
        return Err(DatatypeMismatchUnexpectedNull {
            sql_expr: call_sql(c, "schema_of_json"),
            expr_name: "json".to_owned(),
        }
        .into());
    }
    if c.constant(0).is_none() {
        let input = c.data_type(0).cloned().unwrap_or(DataType::Null);
        return Err(DatatypeMismatchNonFoldableInput {
            sql_expr: call_sql(c, "schema_of_json"),
            input_name: "`json`".to_owned(),
            input_type: format!("\"{}\"", input.sql()),
            input_expr: format!("\"{}\"", c.sql(0).unwrap_or_default()),
        }
        .into());
    }
    check_options(c, 1)?;
    super::defs::string(c)
}

/// `to_json`: only a struct, map, array or untyped null reads; the result
/// is a `STRING`, always nullable.
fn to_json(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let ok = matches!(
        c.data_type(0),
        None | Some(DataType::Struct(_) | DataType::Map(_) | DataType::Array(_))
    );
    if !ok {
        let input = c.data_type(0).cloned().unwrap_or(DataType::Null);
        return Err(DatatypeMismatchInvalidJsonSchema {
            sql_expr: format!("\"to_json({})\"", c.sql(0).unwrap_or_default()),
            schema: format!("\"{}\"", input.sql()),
        }
        .into());
    }
    check_options(c, 1)?;
    string_result(c)
}

/// `from_json`: the JSON text must be a string, the schema a non-null
/// string literal, the options a non-null string map; the result is the
/// schema's type, always nullable.
fn from_json(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let strings = matches!(c.data_type(0), None | Some(DataType::String));
    if !strings {
        let input = c.data_type(0).cloned().unwrap_or(DataType::Null);
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: format!("\"from_json({})\"", c.sql(0).unwrap_or_default()),
            param_index: "first".to_owned(),
            required_type: InputType::Exact(DataType::String).sql(),
            input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            input_type: format!("\"{}\"", input.sql()),
        }
        .into());
    }
    let schema = match c.constant(1) {
        Some(Constant::String(schema)) => schema.clone(),
        Some(Constant::Error(e)) => return Err(e.clone()),
        _ => {
            return Err(InvalidSchemaNonStringLiteral {
                input_schema: format!("\"{}\"", c.sql(1).unwrap_or_default()),
            }
            .into());
        }
    };
    let parsed = parse_json_schema(&schema)?;
    check_options(c, 2)?;
    Ok(ArgType::new(parsed, true))
}

/// Parses a `from_json` schema: a data type, or a bare struct field list
/// (which Spark accepts without `struct<...>`). The direct parse's error
/// is reported when the wrapped parse fails too.
pub(crate) fn parse_json_schema(schema: &str) -> Result<DataType, KrishivError> {
    match parse_data_type(schema).and_then(DataType::refuse_collations) {
        Ok(parsed) => Ok(parsed),
        Err(first) => {
            if schema.trim().is_empty() {
                return Err(first);
            }
            match parse_data_type(&format!("struct<{schema}>"))
                .and_then(DataType::refuse_collations)
            {
                Ok(parsed) => Ok(parsed),
                Err(_) => Err(first),
            }
        }
    }
}

/// Checks the options argument at `at`: absent is the defaults, a non-map
/// is `INVALID_OPTIONS.NON_MAP_FUNCTION`, a map whose keys or values are
/// not strings is `INVALID_OPTIONS.NON_STRING_TYPE`, and a null map is not
/// a map either. A non-foldable map passes here and fails when bound.
fn check_options(c: &CallTypes<'_>, at: usize) -> Result<(), KrishivError> {
    if c.args().len() <= at {
        return Ok(());
    }
    match c.data_type(at) {
        Some(DataType::Map(map))
            if map.key_type() == &DataType::String && map.value_type() == &DataType::String => {}
        Some(DataType::Map(_)) => {
            let input = c.data_type(at).cloned().unwrap_or(DataType::Null);
            return Err(InvalidOptionsNonStringType {
                map_type: format!("\"{}\"", input.sql()),
            }
            .into());
        }
        _ => return Err(InvalidOptionsNonMapFunction {}.into()),
    }
    if matches!(c.constant(at), Some(Constant::Null)) {
        return Err(InvalidOptionsNonMapFunction {}.into());
    }
    Ok(())
}

/// The full call text as Spark's errors quote it (`"name(args)"`).
fn call_sql(c: &CallTypes<'_>, name: &str) -> String {
    let args: Vec<String> = (0..c.args().len())
        .map(|i| c.sql(i).unwrap_or_default())
        .collect();
    format!("\"{name}({})\"", args.join(", "))
}

/// `json_tuple(json, field1, ...)` (Spark 4.2's `JsonTuple`, a generator):
/// every argument must be a `STRING` (anything else, an untyped `NULL`
/// included, is `DATATYPE_MISMATCH.NON_STRING_TYPE`; the analyzer applies no
/// implicit casts), field names may be non-foldable, and the result is the
/// struct of N nullable `STRING`s `c0` .. `c(N-1)` the kernel in
/// `crate::scalar::json_tuple` returns (the struct itself is never null).
/// The analyzer expands a `json_tuple` select item into field accesses of
/// one shared struct computation. Arity (`> 1`) comes from Spark's catalog.
fn json_tuple(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let n = c.args().len();
    if n < 2 {
        // Unreachable through the analyzer (its arity check runs first);
        // the signature stays total for direct callers.
        return Err(krishiv_common::error::WrongNumArgsWithoutSuggestion {
            function_name: "`json_tuple`".to_owned(),
            expected_num: "> 1".to_owned(),
            actual_num: n.to_string(),
            docroot: "https://spark.apache.org/docs/latest".to_owned(),
        }
        .into());
    }
    let shown: Vec<String> = (0..n).map(|i| c.sql(i).unwrap_or_default()).collect();
    for i in 0..n {
        if c.data_type(i) != Some(&DataType::String) {
            return Err(DatatypeMismatchNonStringType {
                func_name: "`json_tuple`".to_owned(),
                sql_expr: format!("\"json_tuple({})\"", shown.join(", ")),
            }
            .into());
        }
    }
    let fields: Vec<StructField> = (0..n - 1)
        .map(|i| StructField::new(format!("c{i}"), DataType::String, true))
        .collect();
    Ok(ArgType::new(
        DataType::Struct(StructType::new(fields)),
        false,
    ))
}
