// SPDX-License-Identifier: Apache-2.0
//! The analysis signatures of `from_avro` and `to_avro` (Spark 4.0+, the
//! SQL functions of the `spark-avro` module).
//!
//! Spark checks the arguments itself: the schema is a foldable string and
//! the options a foldable `MAP<STRING, STRING>` (`DATATYPE_MISMATCH.
//! TYPE_CHECK_FAILURE_WITH_HINT` with Spark's text). `from_avro`'s result is
//! the Avro schema as a Spark type; it is always the all-nullable form
//! (what `PERMISSIVE` returns; Spark's `FAILFAST` keeps the schema's own
//! nullability inside records, which only shows in a printed schema).
//! A first argument that is not binary is a type mismatch here; Spark
//! raises `INTERNAL_ERROR` for it (a Spark bug, goldens `errors_args`).

use krishiv_common::error::{
    DatatypeMismatchTypeCheckFailureWithHint, DatatypeMismatchUnexpectedInputType, KrishivError,
    KrishivUnsupportedFeature,
};
use krishiv_io::avro::datum::DatumDecoder;
use krishiv_types::DataType;

use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, Constant, Inputs};

/// `from_avro` and `to_avro`.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(&["from_avro"], &Inputs::Any, from_avro);
    out.extend(scalar(&["to_avro"], &Inputs::Any, to_avro));
    out
}

/// The full call text as Spark's errors quote it (`"name(args)"`).
fn call_sql(c: &CallTypes<'_>, name: &str) -> String {
    let args: Vec<String> = (0..c.args().len())
        .map(|i| c.sql(i).unwrap_or_default())
        .collect();
    format!("\"{name}({})\"", args.join(", "))
}

/// `DATATYPE_MISMATCH.TYPE_CHECK_FAILURE_WITH_HINT` with `msg`.
fn mismatch(c: &CallTypes<'_>, name: &str, msg: String) -> KrishivError {
    DatatypeMismatchTypeCheckFailureWithHint {
        sql_expr: call_sql(c, name),
        msg,
        hint: String::new(),
    }
    .into()
}

/// The schema argument: a foldable string or NULL, else Spark's error.
fn schema_text(
    c: &CallTypes<'_>,
    name: &str,
    upper: &str,
    direction: &str,
) -> Result<Option<String>, KrishivError> {
    match c.constant(1) {
        Some(Constant::String(s)) => Ok(Some(s.clone())),
        Some(Constant::Null) => Ok(None),
        Some(Constant::Error(e)) => Err(e.clone()),
        _ => Err(mismatch(
            c,
            name,
            format!(
                "The second argument of the {upper} SQL function must be a constant string containing the JSON representation of the schema to use for converting the value {direction} AVRO format"
            ),
        )),
    }
}

/// `from_avro(binary, jsonSchema[, options])`.
fn from_avro(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let schema = schema_text(c, "from_avro", "FROM_AVRO", "from")?;
    if c.args().len() > 2 {
        let map_of_strings = match c.data_type(2) {
            Some(DataType::Map(m)) => {
                matches!(m.key_type(), DataType::String | DataType::Null)
                    && matches!(m.value_type(), DataType::String | DataType::Null)
            }
            Some(DataType::Null) | None => matches!(c.constant(2), Some(Constant::Null)),
            _ => false,
        };
        if !map_of_strings || !c.foldable(2) {
            return Err(mismatch(
                c,
                "from_avro",
                "The third argument of the FROM_AVRO SQL function must be a constant map of strings to strings containing the options to use for converting the value from AVRO format".to_owned(),
            ));
        }
    }
    // Spark checks the schema and options first and never types the first argument
    // (it is an INTERNAL_ERROR there), so a bad schema or options wins over a bad value.
    if !matches!(c.data_type(0), Some(DataType::Binary)) {
        let input = c.data_type(0).cloned().unwrap_or(DataType::Null);
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql(c, "from_avro"),
            param_index: "first".to_owned(),
            required_type: "\"BINARY\"".to_owned(),
            input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            input_type: format!("\"{}\"", input.sql()),
        }
        .into());
    }
    let Some(schema) = schema else {
        return Err(KrishivUnsupportedFeature {
            feature: "from_avro with a NULL schema (Spark fails with a SchemaParseException)"
                .to_owned(),
        }
        .into());
    };
    let decoder = DatumDecoder::new(&schema, None, true).map_err(|e| {
        KrishivError::from(KrishivUnsupportedFeature {
            feature: format!("from_avro with this schema: {e}"),
        })
    })?;
    let field = arrow_schema::Field::new("value", decoder.data_type().clone(), true);
    let spark = DataType::from_arrow_field(&field)?;
    Ok(ArgType::new(spark, true))
}

/// `to_avro(value[, jsonSchema])`: `BINARY`, NULL for a NULL value.
fn to_avro(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if c.args().len() > 1 {
        schema_text(c, "to_avro", "TO_AVRO", "to")?;
    }
    Ok(ArgType::new(DataType::Binary, true))
}
