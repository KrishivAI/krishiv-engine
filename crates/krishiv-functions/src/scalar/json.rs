// SPDX-License-Identifier: Apache-2.0
//! JSON functions: `from_json` and `to_json` (nested documents),
//! `schema_of_json` (inference), `get_json_object` (JSONPath),
//! `json_array_length` and `json_object_keys` (structural queries).
//!
//! The analyzer types every call (the schema must be a string literal,
//! the options a string map); the binding only re-reads those values, so
//! a mismatch here is unreachable. The schema parses once per binding and
//! the options map compiles once per binding; every row then streams
//! through the kernels.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, StringArray};
use arrow_schema::DataType as ArrowDataType;
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_kernels::json::{
    JsonOptions, get_json_object, infer_json, json_array_length, json_object_keys, read_json,
    write_json,
};
use krishiv_types::{ArrayType, DataType};

use super::args::{Inputs, is_string, to_utf8};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "from_json",
    "to_json",
    "schema_of_json",
    "get_json_object",
    "json_array_length",
    "json_object_keys",
];

/// `INTERNAL_ERROR`.
fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// `KRISHIV_UNSUPPORTED_FEATURE`.
fn unsupported(feature: String) -> KrishivError {
    krishiv_common::error::KrishivUnsupportedFeature { feature }.into()
}

/// Binds the JSON function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let binding = match listed {
        "json_array_length" => array_length(args),
        "json_object_keys" => object_keys(args),
        "get_json_object" => get_object(args),
        "schema_of_json" => schema_of(args),
        "from_json" => from(args, opts),
        "to_json" => to(args, opts),
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, binding))
}

/// `json_array_length(json)`: an `INT` per row.
fn array_length(args: &[BindArg]) -> Result<Binding, KrishivError> {
    let [json] = args else {
        return Err(bad_types("json_array_length", args));
    };
    if !is_string(&json.data_type()) {
        return Err(bad_types("json_array_length", args));
    }
    Ok(Binding::new(ArrowDataType::Int32, |x: &Inputs<'_>| {
        Ok(json_array_length(&x.strings(0)?))
    }))
}

/// `json_object_keys(json)`: an array of the top-level keys per row.
fn object_keys(args: &[BindArg]) -> Result<Binding, KrishivError> {
    let [json] = args else {
        return Err(bad_types("json_object_keys", args));
    };
    if !is_string(&json.data_type()) {
        return Err(bad_types("json_object_keys", args));
    }
    let element = DataType::Array(ArrayType::new(DataType::String, false).into()).to_arrow();
    Ok(Binding::new(element, |x: &Inputs<'_>| {
        json_object_keys(&x.strings(0)?)
    }))
}

/// `get_json_object(json, path)`: the JSONPath lookup per row.
fn get_object(args: &[BindArg]) -> Result<Binding, KrishivError> {
    let [json, path] = args else {
        return Err(bad_types("get_json_object", args));
    };
    if !is_string(&json.data_type()) || !is_string(&path.data_type()) {
        return Err(bad_types("get_json_object", args));
    }
    Ok(Binding::new(ArrowDataType::Utf8, |x: &Inputs<'_>| {
        Ok(get_json_object(&x.strings(0)?, &x.strings(1)?))
    }))
}

/// `schema_of_json(json[, options])`: the inferred type's `sql` form per
/// row (`NULL` for malformed input and for types deeper than Spark's
/// nesting limit, which no DDL string could name back).
fn schema_of(args: &[BindArg]) -> Result<Binding, KrishivError> {
    let (json, options) = match args {
        [json] => (json, None),
        [json, options] => (json, Some(options)),
        _ => return Err(bad_types("schema_of_json", args)),
    };
    if !is_string(&json.data_type()) {
        return Err(bad_types("schema_of_json", args));
    }
    let options = read_options(options)?;
    Ok(Binding::new(ArrowDataType::Utf8, move |x: &Inputs<'_>| {
        let inferred = infer_json(&x.strings(0)?, &options)?;
        let mut out = Vec::with_capacity(inferred.len());
        for ty in inferred {
            match ty {
                None => out.push(None),
                Some(arrow) => match DataType::from_arrow(&arrow) {
                    Ok(parsed) => out.push(Some(parsed.sql())),
                    Err(_) => out.push(None),
                },
            }
        }
        Ok(Arc::new(StringArray::from(out)) as ArrayRef)
    }))
}

/// `from_json(json, schema[, options])`: the schema's Arrow type per row.
fn from(args: &[BindArg], opts: &BindOptions) -> Result<Binding, KrishivError> {
    let (json, schema, options) = match args {
        [json, schema] => (json, schema, None),
        [json, schema, options] => (json, schema, Some(options)),
        _ => return Err(bad_types("from_json", args)),
    };
    if !is_string(&json.data_type()) {
        return Err(bad_types("from_json", args));
    }
    let target = read_schema(schema)?.to_arrow();
    let options = read_options(options)?;
    let zone = Arc::clone(&opts.time_zone);
    let start = opts.query_start_micros;
    Ok(Binding::new(target.clone(), move |x: &Inputs<'_>| {
        read_json(&x.strings(0)?, &target, &options, zone.as_ref(), start)
    }))
}

/// `to_json(value[, options])`: compact JSON text per row.
fn to(args: &[BindArg], opts: &BindOptions) -> Result<Binding, KrishivError> {
    let (value, options) = match args {
        [value] => (value, None),
        [value, options] => (value, Some(options)),
        _ => return Err(bad_types("to_json", args)),
    };
    if !matches!(
        value.data_type(),
        ArrowDataType::Struct(_)
            | ArrowDataType::List(_)
            | ArrowDataType::LargeList(_)
            | ArrowDataType::Map(..)
    ) {
        return Err(bad_types("to_json", args));
    }
    let options = read_options(options)?;
    let zone = Arc::clone(&opts.time_zone);
    Ok(Binding::new(ArrowDataType::Utf8, move |x: &Inputs<'_>| {
        write_json(&x.column(0)?, &options, zone.as_ref())
    }))
}

/// The schema argument's value, parsed as Spark parses a `from_json`
/// schema (a data type, or a bare struct field list).
fn read_schema(arg: &BindArg) -> Result<DataType, KrishivError> {
    let BindArg::Literal(a) = arg else {
        return Err(unsupported(
            "a JSON schema that is not a literal".to_owned(),
        ));
    };
    let schema = to_utf8(a.as_ref())?;
    if !schema.is_valid(0) {
        return Err(internal("a null JSON schema".to_owned()));
    }
    crate::registry::parse_json_schema(schema.value(0))
        .map_err(|e| internal(format!("a JSON schema the analyzer accepted: {e}")))
}

/// The options argument's compiled form (absent is the defaults).
fn read_options(arg: Option<&BindArg>) -> Result<JsonOptions, KrishivError> {
    let Some(arg) = arg else {
        return Ok(JsonOptions::new());
    };
    let BindArg::Literal(a) = arg else {
        return Err(unsupported(
            "JSON options that are not a literal".to_owned(),
        ));
    };
    if !matches!(a.data_type(), ArrowDataType::Map(..)) {
        return Err(internal("JSON options that are not a map".to_owned()));
    }
    let map = a.as_map();
    let entries = map.entries();
    let keys = to_utf8(entries.column(0).as_ref())?;
    let values = to_utf8(entries.column(1).as_ref())?;
    let mut options = BTreeMap::new();
    for row in 0..map.len() {
        if map.is_null(row) {
            continue;
        }
        if keys.is_null(row) || values.is_null(row) {
            return Err(internal("a null JSON option".to_owned()));
        }
        options.insert(keys.value(row).to_owned(), values.value(row).to_owned());
    }
    JsonOptions::from_map(&options)
}
