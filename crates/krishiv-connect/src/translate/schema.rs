// SPDX-License-Identifier: Apache-2.0
//! Schema texts a Connect plan carries (`LocalRelation.schema`, `Read.DataSource.schema`): DDL or
//! JSON, parsed as Spark's `SparkConnectPlanner` does (v4.3.0).

use krishiv_common::error::{ConnectInvalidPlanInvalidSchemaNonStructType, KrishivError};
use krishiv_types::{DataType, StructField, StructType};

/// The planner's `parseDatatypeString` (`SparkConnectPlanner.scala:1511-1528`):
/// a table schema (`a INT, b STRING`), else a data type (`int`), else the
/// text as the fields of a struct; the first failure is the error.
///
/// Krishiv parses the table schema as the fields of a `STRUCT<...>`, so
/// clauses only a table schema takes (`DEFAULT`, `GENERATED`) are refused,
/// and the error is the data type's (Spark reports the table schema's): a
/// recorded difference.
///
/// # Errors
/// The parse error of the text as a data type.
pub fn parse_datatype_string(text: &str) -> Result<DataType, KrishivError> {
    if let Ok(t) = krishiv_types::parse_data_type(&format!("STRUCT<{text}>")) {
        return Ok(t);
    }
    match krishiv_types::parse_data_type(text) {
        Ok(t) => Ok(t),
        Err(e) => {
            krishiv_types::parse_data_type(&format!("STRUCT<{}>", text.trim())).map_err(|_| e)
        }
    }
}

/// `DataType.parseTypeWithFallback`: `parser`'s result, else `fallback`'s; when both fail, the
/// first error.
fn with_fallback(
    text: &str,
    parser: impl FnOnce(&str) -> Result<DataType, KrishivError>,
    fallback: impl FnOnce(&str) -> Result<DataType, KrishivError>,
) -> Result<DataType, KrishivError> {
    parser(text).or_else(|first| fallback(text).map_err(|_| first))
}

/// A `LocalRelation` schema (`SparkConnectPlanner.transformLocalRelation`): DDL by
/// [`parse_datatype_string`], else JSON, and a type that is not a struct wrapped as
/// `struct<value: t>` (`toStructTypeOrWrap`).
///
/// # Errors
/// The parse error of the text as DDL when it is not JSON either.
pub fn parse_local_schema(text: &str) -> Result<StructType, KrishivError> {
    Ok(
        match with_fallback(text, parse_datatype_string, DataType::from_json)? {
            DataType::Struct(s) => s,
            other => StructType::new(vec![StructField::new("value", other, true)]),
        },
    )
}

/// A data source's user schema (`parseSchema`, `SparkConnectPlanner.scala:1682`): a table schema
/// by `StructType.fromDDL`, else JSON.
///
/// # Errors
/// The parse error of the DDL when the text is not JSON either;
/// `CONNECT_INVALID_PLAN.INVALID_SCHEMA_NON_STRUCT_TYPE` for JSON of a type that is not a struct
/// (golden-pending: the class is the registry's Connect one for
/// `invalidSchemaStringNonStructType`, its use unverified against a real run).
pub fn parse_table_schema(text: &str) -> Result<StructType, KrishivError> {
    let ddl = |t: &str| krishiv_types::parse_data_type(&format!("STRUCT<{t}>"));
    match with_fallback(text, ddl, DataType::from_json)? {
        DataType::Struct(s) => Ok(s),
        other => Err(ConnectInvalidPlanInvalidSchemaNonStructType {
            input_schema: format!("\"{text}\""),
            data_type: format!("\"{}\"", other.sql()),
        }
        .into()),
    }
}

#[cfg(test)]
mod tests;
