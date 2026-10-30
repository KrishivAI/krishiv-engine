// SPDX-License-Identifier: Apache-2.0
//! `from_xml` and `to_xml` analysis signatures.
//!
//! `from_xml(xmlStr, schema[, options])`: Spark's `XmlToStructs`. The first
//! argument is a `STRING` (no implicit cast); the schema is a foldable `STRING`
//! literal holding a DDL struct type; options are a `MAP<STRING, STRING>` from
//! `map(...)`. The result is the struct the schema names, nullable when the
//! input may be null.
//!
//! `to_xml(struct[, options])`: Spark's `StructsToXml`. The first argument
//! must be a struct (an array or a map is
//! `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` naming `STRUCT`); options are a
//! `MAP<STRING, STRING>`. The result is a nullable `STRING`.

use krishiv_common::error::{
    DatatypeMismatchUnexpectedInputType, InvalidOptionsNonMapFunction,
    InvalidSchemaNonStringLiteral, KrishivError,
};
use krishiv_types::DataType;

use super::csv::{check_options_map, struct_of_ddl};
use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, Constant, InputType, Inputs};

/// The options argument at `at`, if any: absent is the defaults, anything but
/// a string map is refused with Spark's class.
fn check_options(c: &CallTypes<'_>, at: usize) -> Result<(), KrishivError> {
    if c.args().len() > at {
        match check_options_map(c, at) {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(e),
            None => return Err(KrishivError::from(InvalidOptionsNonMapFunction {})),
        }
    }
    Ok(())
}

/// `from_xml(xmlStr, schema[, options])`.
fn from_xml(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    // Options first, as Spark checks them before the schema.
    check_options(c, 2)?;
    let not_a_schema = || {
        KrishivError::from(InvalidSchemaNonStringLiteral {
            input_schema: format!("\"{}\"", c.sql(1).unwrap_or_default()),
        })
    };
    if c.constant(1).is_none() && !c.is_plain_literal(1) {
        return Err(not_a_schema());
    }
    match c.constant(1) {
        Some(Constant::Error(e)) => Err(e.clone()),
        Some(Constant::String(s)) => {
            let struct_type = struct_of_ddl(s)?;
            let nullable = c.args().first().is_some_and(|a| a.nullable);
            Ok(ArgType::new(struct_type, nullable))
        }
        _ => Err(not_a_schema()),
    }
}

/// `to_xml(struct[, options])`.
fn to_xml(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    check_options(c, 1)?;
    let sql = |i: usize| c.sql(i).unwrap_or_default();
    match c.data_type(0) {
        Some(DataType::Struct(_)) => Ok(ArgType::new(DataType::String, true)),
        other => Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: format!("\"to_xml({})\"", sql(0)),
            param_index: "first".to_owned(),
            required_type: "\"STRUCT\"".to_owned(),
            input_sql: format!("\"{}\"", sql(0)),
            input_type: format!(
                "\"{}\"",
                other.map_or_else(|| "VOID".to_owned(), DataType::sql)
            ),
        }
        .into()),
    }
}

/// The first argument of `from_xml` is a `STRING` (no implicit cast: an `int`
/// is refused, as Spark's `ExpectsInputTypes` refuses it); the schema and
/// options are taken as they are.
fn from_xml_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let mut out = vec![
        InputType::Strict(DataType::String),
        InputType::Any,
        InputType::Any,
    ];
    out.truncate(c.args().len().max(2));
    out
}

pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(
        &["from_xml"],
        &Inputs::Built(crate::signature::InputBuilder(from_xml_inputs)),
        from_xml,
    );
    out.extend(scalar(&["to_xml"], &Inputs::Any, to_xml));
    out
}
