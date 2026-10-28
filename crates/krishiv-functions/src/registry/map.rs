// SPDX-License-Identifier: Apache-2.0
//! The map functions' analysis signatures: `map`, `map_concat`,
//! `map_contains_key`, `map_keys`, `map_values`, `map_entries`,
//! `map_from_arrays`, `map_from_entries` and `str_to_map`.
//!
//! Typing follows Spark 4.2 (`CreateMap`, `MapConcat`, `MapContainsKey`,
//! `MapKeys`, `MapValues`, `MapEntries`, `MapFromArrays`, `MapFromEntries`,
//! `StringToMap` at the v4.2.0 tag): the implicit casts of
//! `TypeCoercionHelper` (keys and values of `map` meet separately,
//! `map_concat` needs every map and, with ANSI, the same type) and the
//! result types, checked against the function-matrix fixtures. Coercion
//! lives in the analyzer (`Inputs::Map`, `Inputs::MapConcat`,
//! `Inputs::MapContainsKey`); the rules below read the cast types.

use krishiv_common::error::{
    DatatypeMismatchInvalidMapKeyType, DatatypeMismatchMapFunctionDiffTypes,
    DatatypeMismatchNullType, DatatypeMismatchUnexpectedInputType, KrishivError,
};
use krishiv_types::{ArrayType, DataType, MapType, StructField, StructType};

use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, InputType, Inputs};

/// The whole call as Spark quotes it (`"map_contains_key(map(1, a), 1)"`).
fn call_sql(name: &str, c: &CallTypes<'_>) -> String {
    let args: Vec<String> = (0..c.args().len())
        .map(|i| c.sql(i).unwrap_or_default())
        .collect();
    format!("\"{name}({})\"", args.join(", "))
}

/// The map type of argument `i`, when it is one.
fn map_at(c: &CallTypes<'_>, i: usize) -> Option<MapType> {
    match c.data_type(i) {
        Some(DataType::Map(m)) => Some((**m).clone()),
        _ => None,
    }
}

/// Whether the type contains a map (Spark's `checkForMapKeyType`, without
/// the variant case: Krishiv has no variant type).
fn contains_map(t: &DataType) -> bool {
    match t {
        DataType::Map(_) => true,
        DataType::Array(a) => contains_map(a.element_type()),
        DataType::Struct(s) => s.fields().iter().any(|f| contains_map(f.data_type())),
        _ => false,
    }
}

/// `map(k1, v1, ...)`: the common key and value types (the analyzer met
/// them), `INVALID_MAP_KEY_TYPE` for a key holding a map. Never null.
fn map_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let key = c
        .args()
        .iter()
        .step_by(2)
        .filter_map(|a| a.data_type.clone())
        .next()
        .unwrap_or(DataType::Null);
    let value = c
        .args()
        .iter()
        .skip(1)
        .step_by(2)
        .filter_map(|a| a.data_type.clone())
        .next()
        .unwrap_or(DataType::Null);
    let value_nullable = c.args().iter().skip(1).step_by(2).any(|a| a.nullable);
    if contains_map(&key) {
        return Err(DatatypeMismatchInvalidMapKeyType {
            sql_expr: call_sql("map", c),
            key_type: format!("\"{}\"", key.sql()),
        }
        .into());
    }
    Ok(ArgType::new(
        MapType::new(key, value, value_nullable).into(),
        false,
    ))
}

/// `map_concat(m1, ...)`: the merged map type (the analyzer met the maps,
/// or they share it); `INVALID_MAP_KEY_TYPE` for a key holding a map.
/// Null when any argument may be.
fn map_concat_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let mut merged: Option<DataType> = None;
    for a in c.args() {
        let Some(t) = a.data_type.clone() else {
            continue;
        };
        merged = Some(match merged {
            None => t,
            Some(m) => m.union_like_merge(&t),
        });
    }
    let merged = merged.unwrap_or(MapType::new(DataType::String, DataType::String, true).into());
    if let DataType::Map(m) = &merged
        && contains_map(m.key_type())
    {
        return Err(DatatypeMismatchInvalidMapKeyType {
            sql_expr: call_sql("map_concat", c),
            key_type: format!("\"{}\"", m.key_type().sql()),
        }
        .into());
    }
    Ok(ArgType::new(merged, c.any_nullable()))
}

/// `map_contains_key(map, key)`: `NULL_TYPE` for an untyped null key;
/// `MAP_FUNCTION_DIFF_TYPES` unless the (cast) key shares the map's key
/// type. Always null: Spark evaluates it as `array_contains`, whose array
/// (`map_keys`) may hold nulls.
fn map_contains_key_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let sql_expr = call_sql("map_contains_key", c);
    if c.data_type(1).is_none() {
        return Err(DatatypeMismatchNullType {
            sql_expr,
            function_name: "`map_contains_key`".to_owned(),
        }
        .into());
    }
    let same = match (c.data_type(0), c.data_type(1)) {
        (Some(DataType::Map(m)), Some(k)) => m.key_type().equals_ignore_nullability(k),
        _ => false,
    };
    if !same {
        return Err(DatatypeMismatchMapFunctionDiffTypes {
            sql_expr,
            function_name: "`map_contains_key`".to_owned(),
            data_type: "\"MAP\"".to_owned(),
            left_type: quoted(c.data_type(0)),
            right_type: quoted(c.data_type(1)),
        }
        .into());
    }
    Ok(ArgType::new(DataType::Boolean, true))
}

fn quoted(t: Option<&DataType>) -> String {
    format!("\"{}\"", t.map_or_else(|| "VOID".to_owned(), DataType::sql))
}

/// `map_keys`, `map_values`, `map_entries`: the projection of the (cast)
/// map argument. The argument is always a map here (the analyzer refused
/// anything else); anything else is the same `UNEXPECTED_INPUT_TYPE`.
fn projection(
    c: &CallTypes<'_>,
    name: &str,
    pick: impl FnOnce(&MapType) -> DataType,
) -> Result<ArgType, KrishivError> {
    let Some(m) = map_at(c, 0) else {
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql(name, c),
            param_index: "first".to_owned(),
            required_type: "\"MAP\"".to_owned(),
            input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            input_type: quoted(c.data_type(0)),
        }
        .into());
    };
    let nullable = c.args().first().is_some_and(|a| a.nullable);
    Ok(ArgType::new(pick(&m), nullable))
}

fn map_keys_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    projection(c, "map_keys", |m| {
        ArrayType::new(m.key_type().clone(), true).into()
    })
}

fn map_values_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    projection(c, "map_values", |m| {
        ArrayType::new(m.value_type().clone(), true).into()
    })
}

fn map_entries_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    projection(c, "map_entries", |m| {
        ArrayType::new(
            StructType::new([
                StructField::new("key", m.key_type().clone(), false),
                StructField::new("value", m.value_type().clone(), m.value_contains_null()),
            ])
            .into(),
            false,
        )
        .into()
    })
}

/// `map_from_arrays(keys, values)`: the map of the (cast) array element
/// types. `INVALID_MAP_KEY_TYPE` for a key holding a map. Null when any
/// argument may be.
fn map_from_arrays_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let (keys, values) = (array_at(c, 0), array_at(c, 1));
    let (kt, vt, contains_null) = match (keys, values) {
        (Some(k), Some(v)) => (
            k.element_type().clone(),
            v.element_type().clone(),
            v.contains_null(),
        ),
        _ => (DataType::Null, DataType::Null, true),
    };
    if contains_map(&kt) {
        return Err(DatatypeMismatchInvalidMapKeyType {
            sql_expr: call_sql("map_from_arrays", c),
            key_type: format!("\"{}\"", kt.sql()),
        }
        .into());
    }
    Ok(ArgType::new(
        MapType::new(kt, vt, contains_null).into(),
        c.any_nullable(),
    ))
}

fn array_at(c: &CallTypes<'_>, i: usize) -> Option<ArrayType> {
    match c.data_type(i) {
        Some(DataType::Array(a)) => Some((**a).clone()),
        _ => None,
    }
}

/// `map_from_entries(entries)`: the map of the entry struct's fields; the
/// argument must be an array of two-field structs
/// (`UNEXPECTED_INPUT_TYPE` otherwise). `INVALID_MAP_KEY_TYPE` for a key
/// holding a map. Null when the array may be null or hold nulls.
fn map_from_entries_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let entry = match c.data_type(0) {
        Some(DataType::Array(a))
            if matches!(
                a.element_type(),
                DataType::Struct(s) if s.fields().len() == 2
            ) =>
        {
            std::sync::Arc::clone(a)
        }
        _ => {
            return Err(DatatypeMismatchUnexpectedInputType {
                sql_expr: call_sql("map_from_entries", c),
                param_index: "first".to_owned(),
                required_type: "\"ARRAY\" of pair \"STRUCT\"".to_owned(),
                input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
                input_type: quoted(c.data_type(0)),
            }
            .into());
        }
    };
    let DataType::Struct(fields) = entry.element_type() else {
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql("map_from_entries", c),
            param_index: "first".to_owned(),
            required_type: "\"ARRAY\" of pair \"STRUCT\"".to_owned(),
            input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            input_type: quoted(c.data_type(0)),
        }
        .into());
    };
    let (key, value) = (&fields.fields()[0], &fields.fields()[1]);
    if contains_map(key.data_type()) {
        return Err(DatatypeMismatchInvalidMapKeyType {
            sql_expr: call_sql("map_from_entries", c),
            key_type: format!("\"{}\"", key.data_type().sql()),
        }
        .into());
    }
    let nullable = c.args().first().is_some_and(|a| a.nullable) || entry.contains_null();
    Ok(ArgType::new(
        MapType::new(
            key.data_type().clone(),
            value.data_type().clone(),
            value.nullable(),
        )
        .into(),
        nullable,
    ))
}

/// `str_to_map(text[, pairDelim[, keyValueDelim]])`: `map<string, string>`.
/// Null when any argument may be.
fn str_to_map_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        MapType::new(DataType::String, DataType::String, true).into(),
        c.any_nullable(),
    ))
}

pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(&["map"], &Inputs::Map, map_result);
    out.extend(scalar(
        &["map_concat"],
        &Inputs::MapConcat,
        map_concat_result,
    ));
    out.extend(scalar(
        &["map_contains_key"],
        &Inputs::MapContainsKey,
        map_contains_key_result,
    ));
    out.extend(scalar(
        &["map_keys"],
        &Inputs::Expects(vec![InputType::Map]),
        map_keys_result,
    ));
    out.extend(scalar(
        &["map_values"],
        &Inputs::Expects(vec![InputType::Map]),
        map_values_result,
    ));
    out.extend(scalar(
        &["map_entries"],
        &Inputs::Expects(vec![InputType::Map]),
        map_entries_result,
    ));
    out.extend(scalar(
        &["map_from_arrays"],
        &Inputs::Expects(vec![InputType::Array, InputType::Array]),
        map_from_arrays_result,
    ));
    out.extend(scalar(
        &["map_from_entries"],
        &Inputs::Any,
        map_from_entries_result,
    ));
    out.extend(scalar(
        &["str_to_map"],
        &Inputs::Expects(vec![InputType::Strict(DataType::String)]),
        str_to_map_result,
    ));
    out
}
