// SPDX-License-Identifier: Apache-2.0
//! `schema_of_csv`, `from_csv` and `to_csv` analysis signatures.
//!
//! `schema_of_csv(csv[, options])`: Spark's `SchemaOfCsv` takes any first
//! argument and checks it by hand (whether it folds, whether it is null,
//! whether it is exactly a `STRING`); a second argument must be a
//! `MAP<STRING, STRING>` (anything else is `INVALID_OPTIONS.NON_MAP_FUNCTION`,
//! a map of other types `INVALID_OPTIONS.NON_STRING_TYPE`). The analyzer
//! replaces a well-typed call with the DDL literal, so the call never
//! executes: its execution is `Execution::Evaluator`, which needs no scalar
//! binding.
//!
//! `from_csv(csvStr, schema[, options])`: Spark's `CsvToStructs`. The first
//! argument is a `STRING` (no implicit cast, as Spark's `ExpectsInputTypes`
//! refuses other types); the schema is a foldable `STRING` literal holding
//! a DDL struct type (or the DDL a `schema_of_csv` call resolved to);
//! options are a `MAP<STRING, STRING>` from `map(...)`. The result is the
//! struct the schema names, nullable when the input may be null.
//!
//! `to_csv(struct[, options])`: Spark's `StructsToCsv`. The first argument
//! must be a struct; options are a `MAP<STRING, STRING>`. The result is a
//! nullable `STRING`.

use krishiv_common::error::{
    DatatypeMismatchNonFoldableInput, DatatypeMismatchUnexpectedInputType,
    DatatypeMismatchUnexpectedNull, DatatypeMismatchUnsupportedInputType,
    InvalidOptionsNonMapFunction, InvalidOptionsNonStringType, InvalidSchemaNonStringLiteral,
    InvalidSchemaNonStructType, KrishivError,
};
use krishiv_types::DataType;

use super::Execution;
use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, Constant, FunctionKind, InputType, Inputs};

/// `schema_of_csv(csv[, options])`: Spark's checks in Spark's order (the
/// options map, then whether the CSV folds, whether it is null, whether it
/// is a string). The DDL itself is resolved by the analyzer.
fn schema_of_csv(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    schema_of_text(c, "schema_of_csv", "csv")
}

/// The checks of `schema_of_csv` and `schema_of_xml` (the same in Spark):
/// `func` and the text argument's `name` are what the errors quote.
pub(super) fn schema_of_text(
    c: &CallTypes<'_>,
    func: &str,
    name: &str,
) -> Result<ArgType, KrishivError> {
    if c.args().len() > 1 {
        match check_options_map(c, 1) {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(e),
            None => return Err(KrishivError::from(InvalidOptionsNonMapFunction {})),
        }
    }
    let sql = |i: usize| c.sql(i).unwrap_or_default();
    let call = format!("\"{func}({})\"", sql(0));
    let input_type = || {
        format!(
            "\"{}\"",
            c.data_type(0)
                .map_or_else(|| "VOID".to_owned(), DataType::sql)
        )
    };
    // A plain literal the folder has no constant for (a timestamp or a
    // binary) still folds in Spark: it is non-null, so only the type
    // check below can refuse it.
    let folded = c.constant(0);
    if folded.is_none() && !c.is_plain_literal(0) {
        return Err(DatatypeMismatchNonFoldableInput {
            sql_expr: call,
            input_name: format!("`{name}`"),
            input_type: input_type(),
            input_expr: format!("\"{}\"", sql(0)),
        }
        .into());
    }
    match folded {
        Some(Constant::Error(e)) => Err(e.clone()),
        Some(Constant::Null) => Err(DatatypeMismatchUnexpectedNull {
            sql_expr: call,
            expr_name: name.to_owned(),
        }
        .into()),
        _ => {
            if c.data_type(0) != Some(&DataType::String) {
                return Err(DatatypeMismatchUnexpectedInputType {
                    sql_expr: call,
                    param_index: "first".to_owned(),
                    required_type: "\"STRING\"".to_owned(),
                    input_sql: format!("\"{}\"", sql(0)),
                    input_type: input_type(),
                }
                .into());
            }
            Ok(ArgType::new(DataType::String, false))
        }
    }
}

/// Whether the options argument `i` is a `MAP<STRING, STRING>`: `None` when
/// it is not a map at all, `Some(Err)` for a map with non-string keys or
/// values, `Some(Ok(()))` for a string map.
pub(super) fn check_options_map(c: &CallTypes<'_>, i: usize) -> Option<Result<(), KrishivError>> {
    let ty = c.data_type(i)?;
    let DataType::Map(m) = ty else {
        return None;
    };
    let key_ok = matches!(m.key_type(), DataType::String);
    let val_ok = matches!(m.value_type(), DataType::String);
    if key_ok && val_ok {
        Some(Ok(()))
    } else {
        Some(Err(InvalidOptionsNonStringType {
            map_type: format!("\"{}\"", ty.sql()),
        }
        .into()))
    }
}

/// The struct type the DDL `ddl` names (Spark's `DataType.fromDDL`: a data
/// type, or else a table schema, which becomes a `STRUCT` of its columns).
///
/// # Errors
/// `PARSE_SYNTAX_ERROR`, `UNSUPPORTED_DATATYPE`, `KRISHIV_UNSUPPORTED_FEATURE`
/// and the other DDL errors for a malformed schema; `INVALID_SCHEMA.NON_STRUCT_TYPE`
/// when it is not a struct.
pub(super) fn struct_of_ddl(ddl: &str) -> Result<DataType, KrishivError> {
    let single: Result<DataType, KrishivError> = ddl.parse();
    match single {
        Ok(parsed) => {
            if matches!(parsed, DataType::Struct(_)) {
                Ok(parsed)
            } else {
                Err(InvalidSchemaNonStructType {
                    input_schema: format!("\"{ddl}\""),
                    data_type: format!("\"{}\"", parsed.sql()),
                }
                .into())
            }
        }
        Err(single_err) => {
            // A table schema (`a INT, b DOUBLE`) becomes `STRUCT` of its columns.
            let wrapped = format!("STRUCT<{ddl}>");
            let table: Result<DataType, KrishivError> = wrapped.parse();
            // `STRUCT<...>` is a struct whenever it parses; when it does not,
            // the error is the data type's (a bare `ARRAY` is
            // `INCOMPLETE_TYPE_DEFINITION.ARRAY`, as Spark reports it).
            match table {
                Ok(t) => Ok(t),
                Err(_) => Err(single_err),
            }
        }
    }
}

/// `from_csv(csvStr, schema[, options])`.
fn from_csv(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    // Options first, as Spark checks them before the schema.
    if c.args().len() > 2 {
        match check_options_map(c, 2) {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(e),
            None => return Err(KrishivError::from(InvalidOptionsNonMapFunction {})),
        }
    }
    let sql = |i: usize| c.sql(i).unwrap_or_default();
    // The schema must be a foldable string literal.
    let folded = c.constant(1);
    if folded.is_none() && !c.is_plain_literal(1) {
        // Spark: a schema that is not a string literal (a column, say) is
        // `INVALID_SCHEMA.NON_STRING_LITERAL`.
        return Err(InvalidSchemaNonStringLiteral {
            input_schema: format!("\"{}\"", sql(1)),
        }
        .into());
    }
    match folded {
        Some(Constant::Error(e)) => Err(e.clone()),
        Some(Constant::String(s)) => {
            let struct_type = struct_of_ddl(s)?;
            let nullable = c.args().first().is_some_and(|a| a.nullable);
            Ok(ArgType::new(struct_type, nullable))
        }
        _ => {
            // NULL, a non-string literal, or a plain literal the folder has
            // no constant for: Spark reports the SQL text as not a schema.
            Err(InvalidSchemaNonStringLiteral {
                input_schema: format!("\"{}\"", sql(1)),
            }
            .into())
        }
    }
}

/// `to_csv(struct[, options])`.
fn to_csv(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if c.args().len() > 1 {
        match check_options_map(c, 1) {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(e),
            None => return Err(KrishivError::from(InvalidOptionsNonMapFunction {})),
        }
    }
    let sql = |i: usize| c.sql(i).unwrap_or_default();
    match c.data_type(0) {
        Some(DataType::Struct(_)) => Ok(ArgType::new(DataType::String, true)),
        other => Err(DatatypeMismatchUnsupportedInputType {
            sql_expr: format!("\"to_csv({})\"", sql(0)),
            function_name: "`to_csv`".to_owned(),
            data_type: format!(
                "\"{}\"",
                other.map_or_else(|| "VOID".to_owned(), DataType::sql)
            ),
        }
        .into()),
    }
}

/// The first argument of `from_csv` is a `STRING` (no implicit cast: an
/// `int` is refused, as Spark's `ExpectsInputTypes` refuses it); the schema
/// and options are taken as they are.
fn from_csv_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let mut out = vec![
        InputType::Strict(DataType::String),
        InputType::Any,
        InputType::Any,
    ];
    out.truncate(c.args().len().max(2));
    out
}

pub(super) fn specs() -> Vec<Spec> {
    let mut out = vec![Spec {
        name: "schema_of_csv",
        kind: FunctionKind::Scalar,
        inputs: Inputs::Any,
        result: schema_of_csv,
        execution: Execution::Evaluator,
    }];
    out.extend(scalar(
        &["from_csv"],
        &Inputs::Built(crate::signature::InputBuilder(from_csv_inputs)),
        from_csv,
    ));
    out.extend(scalar(&["to_csv"], &Inputs::Any, to_csv));
    out
}
