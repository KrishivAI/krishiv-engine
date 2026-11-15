// SPDX-License-Identifier: Apache-2.0
//! Schema texts `AnalyzePlan` answers with: Spark's `StructType.treeString`
//! (`tree_string`), `StructType.toDDL` (`json_to_ddl`) and the planner's
//! `parseDatatypeString` (`ddl_parse`), ported from apache/spark v4.3.0
//! (`sql/api/.../types/{StructType,StructField,ArrayType,MapType,DataType}.scala`,
//! `SparkConnectPlanner.scala:1511-1528`).

use krishiv_common::error::{InvalidSchemaNonStructType, KrishivError};
use krishiv_types::{ArrayType, DataType, MapType, StructField, StructType};

/// `StructType.treeString(maxDepth)` (`StructType.scala:384-393`): `root`,
/// then one line per field, nested types indented by `"    |"`. A level of
/// 0 or below, or none, prints every level.
#[must_use]
pub fn tree_string(schema: &StructType, level: Option<i32>) -> String {
    let mut out = String::from("root\n");
    let depth = match level {
        Some(l) if l > 0 => l,
        _ => i32::MAX,
    };
    for f in schema.fields() {
        field(f, " |", &mut out, depth);
    }
    out
}

/// `StructField.buildFormattedString` (`StructField.scala:57-67`).
fn field(f: &StructField, prefix: &str, out: &mut String, max_depth: i32) {
    if max_depth > 0 {
        out.push_str(&format!(
            "{prefix}-- {}: {} (nullable = {})\n",
            escape_meta_characters(f.name()),
            f.data_type().type_name(),
            f.nullable()
        ));
        nested(f.data_type(), &format!("{prefix}    |"), out, max_depth);
    }
}

/// `DataType.buildFormattedString` (`DataType.scala:492-506`): one level
/// down for arrays, structs and maps.
fn nested(t: &DataType, prefix: &str, out: &mut String, max_depth: i32) {
    let depth = max_depth.saturating_sub(1);
    match t {
        DataType::Array(a) => array(a, prefix, out, depth),
        DataType::Struct(s) => {
            for f in s.fields() {
                field(f, prefix, out, depth);
            }
        }
        DataType::Map(m) => map(m, prefix, out, depth),
        _ => {}
    }
}

/// `ArrayType.buildFormattedString` (`ArrayType.scala:71-80`).
fn array(a: &ArrayType, prefix: &str, out: &mut String, max_depth: i32) {
    if max_depth > 0 {
        out.push_str(&format!(
            "{prefix}-- element: {} (containsNull = {})\n",
            a.element_type().type_name(),
            a.contains_null()
        ));
        nested(a.element_type(), &format!("{prefix}    |"), out, max_depth);
    }
}

/// `MapType.buildFormattedString` (`MapType.scala:45-57`).
fn map(m: &MapType, prefix: &str, out: &mut String, max_depth: i32) {
    if max_depth > 0 {
        let inner = format!("{prefix}    |");
        out.push_str(&format!("{prefix}-- key: {}\n", m.key_type().type_name()));
        nested(m.key_type(), &inner, out, max_depth);
        out.push_str(&format!(
            "{prefix}-- value: {} (valueContainsNull = {})\n",
            m.value_type().type_name(),
            m.value_contains_null()
        ));
        nested(m.value_type(), &inner, out, max_depth);
    }
}

/// `SparkSchemaUtils.escapeMetaCharacters` (`common/utils/.../SparkSchemaUtils.scala:27-35`).
fn escape_meta_characters(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{c}' => out.push_str("\\f"),
            '\u{8}' => out.push_str("\\b"),
            '\u{b}' => out.push_str("\\v"),
            '\u{7}' => out.push_str("\\a"),
            c => out.push(c),
        }
    }
    out
}

/// `QuotingUtils.quoteIfNeeded` (`QuotingUtils.scala:46-54`): a name that is
/// not `[a-zA-Z_][a-zA-Z0-9_]*` is backquoted, backquotes doubled.
fn quote_if_needed(name: &str) -> String {
    let mut chars = name.chars();
    let plain = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if plain {
        name.to_owned()
    } else {
        format!("`{}`", name.replace('`', "``"))
    }
}

/// `QuotingUtils.escapeSingleQuotedString` (`QuotingUtils.scala:80-90`).
fn escape_single_quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            c => out.push(c),
        }
    }
    out
}

/// `StructType.toDDL` (`StructType.scala:458`): each field's
/// `StructField.toDDL` (`StructField.scala:295-323`), joined by `,`.
#[must_use]
pub fn to_ddl(schema: &StructType) -> String {
    // A `default` or `comment` value that is not a string is left out;
    // Spark's `metadata.getString` throws on one (an edge case: Krishiv
    // builds such metadata only from strings).
    let metadata = |f: &StructField, key: &str| -> Option<String> {
        f.metadata()
            .as_map()
            .get(key)
            .and_then(|v| v.as_str().map(str::to_owned))
    };
    schema
        .fields()
        .iter()
        .map(|f| {
            let mut out = format!("{} {}", quote_if_needed(f.name()), f.data_type().sql());
            if !f.nullable() {
                out.push_str(" NOT NULL");
            }
            // `getDefault()` (a SQL function parameter's default) before the
            // column's `CURRENT_DEFAULT`.
            if let Some(d) = metadata(f, "default").or_else(|| metadata(f, "CURRENT_DEFAULT")) {
                out.push_str(" DEFAULT ");
                out.push_str(&d);
            }
            if let Some(c) = metadata(f, "comment") {
                out.push_str(" COMMENT '");
                out.push_str(&escape_single_quoted(&c));
                out.push('\'');
            }
            out
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// `json_to_ddl`: Spark's `DataType.fromJson(json).asInstanceOf[StructType].toDDL`
/// (`SparkConnectAnalyzeHandler.scala:232-241`).
///
/// # Errors
/// The errors of [`DataType::from_json`]; `INVALID_SCHEMA.NON_STRUCT_TYPE`
/// for a type that is not a struct (Spark fails with a `ClassCastException`,
/// which reaches the client as a classless `INTERNAL`: a recorded
/// difference).
pub fn json_to_ddl(json: &str) -> Result<String, KrishivError> {
    match DataType::from_json(json)? {
        DataType::Struct(s) => Ok(to_ddl(&s)),
        other => Err(InvalidSchemaNonStructType {
            input_schema: format!("\"{json}\""),
            data_type: format!("\"{}\"", other.sql()),
        }
        .into()),
    }
}

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

#[cfg(test)]
mod tests;
