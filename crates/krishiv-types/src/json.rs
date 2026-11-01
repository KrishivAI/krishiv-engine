// SPDX-License-Identifier: Apache-2.0
//! Spark's JSON form of a type (`DataType.json` and `DataType.fromJson`),
//! as Spark writes schemas in files, plans and Spark Connect metadata:
//! an atomic type is its `typeName` as a JSON string (`"integer"`), a
//! nested type an object whose keys come in Spark's order
//! (`{"type":"array","elementType":"integer","containsNull":true}`).

use std::fmt::Write as _;

use krishiv_common::error::{
    CollationInvalidProvider, ErrorClass, InvalidJsonDataTypeForCollations, KrishivError,
};
use serde_json::{Map, Value};

use crate::collation::{Collation, CollationFamily};
use crate::error::{self, Result};
use crate::nested::{ArrayType, MapType, Metadata, StructField, StructType};
use crate::{DataType, DecimalType, Srid};

impl DataType {
    /// Spark's `DataType.json`: compact, keys in Spark's order.
    pub fn json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        self.write_json_in(out, false);
    }

    /// `in_field`: the type is a struct field's, whose collations are in the
    /// field's `__COLLATIONS` metadata, so a collated string is plain `string`.
    fn write_json_in(&self, out: &mut String, in_field: bool) {
        match self {
            Self::CollatedString(c) if !in_field => {
                write_string(&format!("string collate {c}"), out);
            }
            Self::CollatedString(_) => write_string("string", out),
            Self::Array(a) => {
                out.push_str(r#"{"type":"array","elementType":"#);
                a.element_type().write_json_in(out, in_field);
                let _ = write!(out, r#","containsNull":{}}}"#, a.contains_null());
            }
            Self::Map(m) => {
                out.push_str(r#"{"type":"map","keyType":"#);
                m.key_type().write_json_in(out, in_field);
                out.push_str(r#","valueType":"#);
                m.value_type().write_json_in(out, in_field);
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
                    f.data_type().write_json_in(out, true);
                    let _ = write!(out, r#","nullable":{},"metadata":"#, f.nullable());
                    let mut metadata = f.metadata().as_map().clone();
                    let mut collations = Map::new();
                    collect_collations(f.data_type(), f.name(), &mut collations);
                    if !collations.is_empty() {
                        metadata.insert(COLLATIONS_KEY.to_owned(), Value::Object(collations));
                    }
                    write_value(&Value::Object(metadata), out);
                    out.push('}');
                }
                out.push_str("]}");
            }
            Self::Time { precision } => {
                write_string(&format!("time({precision})"), out);
            }
            Self::Geometry { srid } => {
                write_string(&format!("geometry({})", json_srid(*srid, false)), out);
            }
            Self::Geography { srid } => {
                write_string(
                    &format!("geography({}, SPHERICAL)", json_srid(*srid, true)),
                    out,
                );
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
    let (data_type, metadata) = match obj.get("metadata") {
        None => (data_type, Metadata::empty()),
        Some(Value::Object(m)) => {
            let mut m = m.clone();
            let data_type = match m.remove(COLLATIONS_KEY) {
                Some(Value::Object(map)) => apply_collations(&data_type, name, &map)?,
                Some(_) => return Err(invalid()),
                None => data_type,
            };
            (data_type, Metadata::from_map(m))
        }
        Some(_) => return Err(invalid()),
    };
    Ok(StructField::new(name, data_type, nullable).with_metadata(metadata))
}

/// The field-metadata key under which Spark's JSON form records the
/// collations of a field's string leaves, by path.
const COLLATIONS_KEY: &str = "__COLLATIONS";

/// `spark.NAME` or `icu.NAME`: Spark's collation provider prefix.
fn provider_name(c: Collation) -> String {
    let provider = match c.family() {
        CollationFamily::Icu { .. } => "icu",
        _ => "spark",
    };
    format!("{provider}.{}", c.name())
}

/// The collated string leaves of `t` below `path` (`.element`, `.key` and
/// `.value` under arrays and maps; a nested struct's fields carry their own).
fn collect_collations(t: &DataType, path: &str, out: &mut Map<String, Value>) {
    match t {
        DataType::CollatedString(c) => {
            out.insert(path.to_owned(), Value::String(provider_name(*c)));
        }
        DataType::Array(a) => {
            collect_collations(a.element_type(), &format!("{path}.element"), out);
        }
        DataType::Map(m) => {
            collect_collations(m.key_type(), &format!("{path}.key"), out);
            collect_collations(m.value_type(), &format!("{path}.value"), out);
        }
        _ => {}
    }
}

/// `t` with the collations of `map` applied to its string leaves.
fn apply_collations(t: &DataType, path: &str, map: &Map<String, Value>) -> Result<DataType> {
    Ok(match t {
        DataType::String => match map.get(path) {
            Some(Value::String(spec)) => DataType::collated_string(parse_provider_name(spec)?),
            _ => t.clone(),
        },
        DataType::Array(a) => ArrayType::new(
            apply_collations(a.element_type(), &format!("{path}.element"), map)?,
            a.contains_null(),
        )
        .into(),
        DataType::Map(m) => MapType::new(
            apply_collations(m.key_type(), &format!("{path}.key"), map)?,
            apply_collations(m.value_type(), &format!("{path}.value"), map)?,
            m.value_contains_null(),
        )
        .into(),
        other => {
            if map.contains_key(path) {
                return Err(InvalidJsonDataTypeForCollations {
                    json_type: other.json(),
                }
                .into());
            }
            other.clone()
        }
    })
}

/// `provider.NAME`, checked as Spark's `CollationFactory.fromFullyQualifiedName`
/// does.
fn parse_provider_name(spec: &str) -> Result<Collation> {
    let invalid_provider = |provider: &str| -> KrishivError {
        CollationInvalidProvider {
            provider: provider.to_owned(),
            supported_providers: "spark, icu".to_owned(),
        }
        .into()
    };
    let Some((provider, name)) = spec.split_once('.') else {
        return Err(invalid_provider(spec));
    };
    let collation = Collation::parse(name)?;
    let is_icu = matches!(collation.family(), CollationFamily::Icu { .. });
    match provider {
        "spark" if !is_icu => Ok(collation),
        "icu" if is_icu => Ok(collation),
        "spark" | "icu" => Err(crate::collation::invalid_name(name)),
        other => Err(invalid_provider(other)),
    }
}

/// Spark's `nameToType`: the `typeName` of an atomic type, `decimal`
/// (`decimal(10,0)`), `time(p)` and the spatial JSON names (`geometry` by
/// reference system, `geography` by reference system plus `SPHERICAL`);
/// `None` for a name that is no type.
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
    if let Some(t) = interval_from_name(name) {
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
    if let Some(rest) = name.strip_prefix("string collate ") {
        return Some(Collation::parse(rest).map(DataType::collated_string));
    }
    if let Some(t) = time_from_name(name) {
        return Some(t);
    }
    if let Some(t) = geometry_from_name(name) {
        return Some(t);
    }
    if let Some(t) = geography_from_name(name) {
        return Some(t);
    }
    // A Spark type Krishiv does not have yet (`char(3)`, `variant`, ...).
    match crate::parse_data_type(name) {
        Err(e) if e.error_class() == ErrorClass::KrishivUnsupportedFeature => Some(Err(e)),
        _ => None,
    }
}

/// The reference-system name Spark's JSON form of a spatial type uses, from
/// the type-string fixture (`SRID:0`, `EPSG:3857`, `OGC:CRS84`, `SRID:ANY`).
/// A fixed SRID Spark does not support cannot occur in a validated type;
/// it falls back to the integer so `json` never panics.
fn json_srid(srid: Srid, geographic: bool) -> String {
    match srid {
        Srid::Any => "SRID:ANY".to_owned(),
        Srid::Fixed(s) => Srid::crs(s, geographic)
            .map(str::to_owned)
            .unwrap_or_else(|| s.to_string()),
    }
}

/// The inverse of [`json_srid`]: `None` for a name that is no reference
/// system.
fn crs_to_srid(name: &str) -> Option<Srid> {
    match name {
        "SRID:0" => Some(Srid::Fixed(0)),
        "EPSG:3857" => Some(Srid::Fixed(3857)),
        "OGC:CRS84" => Some(Srid::Fixed(4326)),
        "SRID:ANY" => Some(Srid::Any),
        _ => None,
    }
}

/// Digits as written, without leading zeros (as in `ddl`).
fn json_digits(s: &str) -> String {
    match s.trim_start_matches('0') {
        "" => "0".to_owned(),
        d => d.to_owned(),
    }
}

/// `time(p)` in Spark's JSON form; `None` when `name` is no time name.
/// A numeric precision that overflows `u32` is `UNSUPPORTED_TIME_PRECISION`,
/// as in Spark's DDL path.
fn time_from_name(name: &str) -> Option<Result<DataType>> {
    let inner = name.strip_prefix("time(")?.strip_suffix(')')?.trim();
    if inner.is_empty() || !inner.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(match inner.parse::<u32>() {
        Ok(p) => DataType::time(p),
        Err(_) => Err(error::time_precision(&json_digits(inner))),
    })
}

/// `geometry(...)` in Spark's JSON form (`geometry(SRID:0)`,
/// `geometry(OGC:CRS84)`, `geometry(SRID:ANY)`), or the DDL form
/// (`geometry(0)`, `geometry(any)`); `None` when `name` is no geometry name.
fn geometry_from_name(name: &str) -> Option<Result<DataType>> {
    let inner = name.strip_prefix("geometry(")?.strip_suffix(')')?.trim();
    if let Some(srid) = crs_to_srid(inner) {
        return Some(DataType::geometry(srid));
    }
    if inner.eq_ignore_ascii_case("any") {
        return Some(DataType::geometry(Srid::Any));
    }
    if !inner.is_empty() && inner.bytes().all(|b| b.is_ascii_digit()) {
        return Some(match inner.parse::<i32>() {
            Ok(s) => DataType::geometry(Srid::Fixed(s)),
            Err(_) => Err(error::invalid_srid(&json_digits(inner))),
        });
    }
    // A signed integer (DDL would reject `+`/`-` with a syntax error at the
    // `(`, but JSON only ever carries what `json` wrote).
    if let Some(rest) = inner.strip_prefix('-')
        && !rest.is_empty()
        && rest.bytes().all(|b| b.is_ascii_digit())
    {
        return Some(match inner.parse::<i32>() {
            Ok(s) => DataType::geometry(Srid::Fixed(s)),
            Err(_) => Err(error::invalid_srid(&format!("-{}", json_digits(rest)))),
        });
    }
    None
}

/// `geography(...)` in Spark's JSON form (`geography(OGC:CRS84, SPHERICAL)`,
/// `geography(SRID:ANY, SPHERICAL)`), or the DDL form (`geography(4326)`,
/// `geography(any)`); `None` when `name` is no geography name.
fn geography_from_name(name: &str) -> Option<Result<DataType>> {
    let inner = name.strip_prefix("geography(")?.strip_suffix(')')?;
    if let Some((crs, shape)) = inner.split_once(',') {
        if shape.trim() != "SPHERICAL" {
            return None;
        }
        let srid = crs_to_srid(crs.trim())?;
        return Some(DataType::geography(srid));
    }
    let inner = inner.trim();
    if inner.eq_ignore_ascii_case("any") {
        return Some(DataType::geography(Srid::Any));
    }
    if !inner.is_empty() && inner.bytes().all(|b| b.is_ascii_digit()) {
        return Some(match inner.parse::<i32>() {
            Ok(s) => DataType::geography(Srid::Fixed(s)),
            Err(_) => Err(error::invalid_srid(&json_digits(inner))),
        });
    }
    None
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

/// The interval type whose JSON name is `name`, spelled exactly as Spark's
/// `typeName` spells it.
fn interval_from_name(name: &str) -> Option<DataType> {
    use crate::{DayTimeField as D, YearMonthField as Y};
    if name == "interval" {
        return Some(DataType::CalendarInterval);
    }
    let ym = [
        (Y::Year, Y::Year),
        (Y::Month, Y::Month),
        (Y::Year, Y::Month),
    ]
    .map(|(start, end)| DataType::YearMonthInterval { start, end });
    let dt = [
        (D::Day, D::Day),
        (D::Day, D::Hour),
        (D::Day, D::Minute),
        (D::Day, D::Second),
        (D::Hour, D::Hour),
        (D::Hour, D::Minute),
        (D::Hour, D::Second),
        (D::Minute, D::Minute),
        (D::Minute, D::Second),
        (D::Second, D::Second),
    ]
    .map(|(start, end)| DataType::DayTimeInterval { start, end });
    ym.into_iter().chain(dt).find(|t| t.type_name() == name)
}
