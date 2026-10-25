// SPDX-License-Identifier: Apache-2.0
//! Spark's JSON form of a type (`DataType.json` and `DataType.fromJson`),
//! as Spark writes schemas in files, plans and Spark Connect metadata:
//! an atomic type is its `typeName` as a JSON string (`"integer"`), a
//! nested type an object whose keys come in Spark's order
//! (`{"type":"array","elementType":"integer","containsNull":true}`).

use std::fmt::Write as _;

use krishiv_common::error::{ErrorClass, KrishivError};
use serde_json::{Map, Value};

use crate::error::{self, Result};
use crate::nested::{ArrayType, MapType, Metadata, StructField, StructType};
use crate::{DataType, DecimalType};

impl DataType {
    /// Spark's `DataType.json`: compact, keys in Spark's order.
    pub fn json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        match self {
            Self::Array(a) => {
                out.push_str(r#"{"type":"array","elementType":"#);
                a.element_type().write_json(out);
                let _ = write!(out, r#","containsNull":{}}}"#, a.contains_null());
            }
            Self::Map(m) => {
                out.push_str(r#"{"type":"map","keyType":"#);
                m.key_type().write_json(out);
                out.push_str(r#","valueType":"#);
                m.value_type().write_json(out);
                let _ = write!(out, r#","valueContainsNull":{}}}"#, m.value_contains_null());
            }
            Self::Struct(s) => {
                out.push_str(r#"{"type":"struct","fields":["#);
                for (i, f) in s.fields().iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(r#"{"name":"#);
                    write_string(f.name(), out);
                    out.push_str(r#","type":"#);
                    f.data_type().write_json(out);
                    let _ = write!(out, r#","nullable":{},"metadata":"#, f.nullable());
                    write_value(&Value::Object(f.metadata().as_map().clone()), out);
                    out.push('}');
                }
                out.push_str("]}");
            }
            atomic => write_string(&atomic.type_name(), out),
        }
    }

    /// Spark's `DataType.fromJson`.
    ///
    /// # Errors
    /// `INVALID_JSON_DATA_TYPE` for JSON that is not a Spark type,
    /// `FAILED_TO_PARSE_TOO_COMPLEX` for a type nested deeper than
    /// [`Self::MAX_NESTING_DEPTH`], and `KRISHIV_UNSUPPORTED_FEATURE` for a
    /// Spark type Krishiv does not support yet.
    pub fn from_json(text: &str) -> Result<Self> {
        let value: Value =
            serde_json::from_str(text).map_err(|_| error::invalid_json_type(text))?;
        Self::from_json_value(&value, 0)
    }

    fn from_json_value(v: &Value, depth: usize) -> Result<Self> {
        let invalid = || error::invalid_json_type(&v.to_string());
        let obj = match v {
            Value::String(name) => return atomic_from_name(name).ok_or_else(invalid)?,
            Value::Object(obj) => obj,
            _ => return Err(invalid()),
        };
        if depth >= Self::MAX_NESTING_DEPTH {
            return Err(error::too_deep());
        }
        let child = |key: &str| {
            obj.get(key)
                .ok_or_else(invalid)
                .and_then(|c| Self::from_json_value(c, depth + 1))
        };
        let flag = |key: &str| obj.get(key).and_then(Value::as_bool).ok_or_else(invalid);
        let keys = |expected: &[&str]| {
            obj.len() == expected.len() && expected.iter().all(|k| obj.contains_key(*k))
        };
        match obj.get("type").and_then(Value::as_str) {
            Some("array") if keys(&["type", "elementType", "containsNull"]) => {
                Ok(ArrayType::new(child("elementType")?, flag("containsNull")?).into())
            }
            Some("map") if keys(&["type", "keyType", "valueType", "valueContainsNull"]) => {
                Ok(MapType::new(
                    child("keyType")?,
                    child("valueType")?,
                    flag("valueContainsNull")?,
                )
                .into())
            }
            Some("struct") if keys(&["type", "fields"]) => {
                let fields = obj
                    .get("fields")
                    .and_then(Value::as_array)
                    .ok_or_else(invalid)?;
                let fields = fields
                    .iter()
                    .map(|f| struct_field(f, depth))
                    .collect::<Result<Vec<_>>>()?;
                Ok(StructType::new(fields).into())
            }
            _ => Err(invalid()),
        }
    }
}

/// A struct field: `name`, `type` and `nullable`, and `metadata` (optional,
/// as in Spark).
fn struct_field(v: &Value, depth: usize) -> Result<StructField> {
    let invalid = || error::invalid_json_type(&v.to_string());
    let obj = v.as_object().ok_or_else(invalid)?;
    let has_metadata = obj.contains_key("metadata");
    let expected = 3 + usize::from(has_metadata);
    if obj.len() != expected {
        return Err(invalid());
    }
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(invalid)?;
    let nullable = obj
        .get("nullable")
        .and_then(Value::as_bool)
        .ok_or_else(invalid)?;
    let data_type = DataType::from_json_value(obj.get("type").ok_or_else(invalid)?, depth + 1)?;
    let metadata = match obj.get("metadata") {
        None => Metadata::empty(),
        Some(Value::Object(m)) => Metadata::from_map(m.clone()),
        Some(_) => return Err(invalid()),
    };
    Ok(StructField::new(name, data_type, nullable).with_metadata(metadata))
}

/// Spark's `nameToType`: the `typeName` of an atomic type, or `decimal`
/// (`decimal(10,0)`); `None` for a name that is no type.
fn atomic_from_name(name: &str) -> Option<Result<DataType>> {
    let atomic = [
        DataType::Null,
        DataType::Boolean,
        DataType::Byte,
        DataType::Short,
        DataType::Integer,
        DataType::Long,
        DataType::Float,
        DataType::Double,
        DataType::String,
        DataType::Binary,
        DataType::Date,
        DataType::Timestamp,
        DataType::TimestampNtz,
    ];
    if let Some(t) = atomic.into_iter().find(|t| t.type_name() == name) {
        return Some(Ok(t));
    }
    if name == "decimal" {
        return Some(Ok(DataType::Decimal(DecimalType::USER_DEFAULT)));
    }
    if let Some(args) = name
        .strip_prefix("decimal(")
        .and_then(|r| r.strip_suffix(')'))
    {
        let (p, s) = args.split_once(',')?;
        let (p, s) = (p.trim(), s.trim());
        let digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
        if !digits(p) || !(digits(s) || s.strip_prefix('-').is_some_and(digits)) {
            return None;
        }
        return Some(match (p.parse::<u32>(), s.parse::<u32>()) {
            (Ok(p), Ok(s)) => DecimalType::try_new(p, s).map(DataType::Decimal),
            (Ok(_), Err(_)) if s.starts_with('-') => {
                Err(error::negative_scale(s.parse::<i64>().unwrap_or(i64::MIN)))
            }
            _ => Err(error::precision_exceeds_max(p)),
        });
    }
    // A Spark type Krishiv does not have yet (`char(3)`, `variant`, ...).
    match crate::parse_data_type(name) {
        Err(e) if e.error_class() == ErrorClass::KrishivUnsupportedFeature => Some(Err(e)),
        _ => None,
    }
}

/// A JSON string as json4s writes it: `"`, `\` and control characters
/// escaped, everything else as is.
fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04X}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A metadata value: objects with their keys sorted (see [`Metadata`]).
fn write_value(v: &Value, out: &mut String) {
    match v {
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => write_object(map, out),
        // `null`, booleans and numbers print the same in json4s.
        other => out.push_str(&other.to_string()),
    }
}

/// `metadata` as JSON.
pub(crate) fn metadata_json(metadata: &Metadata) -> String {
    let mut out = String::new();
    write_object(metadata.as_map(), &mut out);
    out
}

fn write_object(map: &Map<String, Value>, out: &mut String) {
    out.push('{');
    for (i, (k, v)) in map.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_string(k, out);
        out.push(':');
        write_value(v, out);
    }
    out.push('}');
}

/// Parses Spark `Metadata` JSON (a Connect struct field's `metadata`).
///
/// # Errors
/// `INVALID_JSON_DATA_TYPE` when `text` is not a JSON object.
pub fn parse_metadata(text: &str) -> Result<Metadata, KrishivError> {
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(map)) => Ok(Metadata::from_map(map)),
        _ => Err(error::invalid_json_type(text)),
    }
}
