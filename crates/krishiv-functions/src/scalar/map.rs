// SPDX-License-Identifier: Apache-2.0
//! Map functions (P2-T029 kernels): `map`, `map_concat`,
//! `map_contains_key`, `map_keys`, `map_values`, `map_entries`,
//! `map_from_arrays`, `map_from_entries` and `str_to_map`.
//!
//! Arguments arrive after the analyzer's implicit casts, so keys and values
//! already share their Spark types (the binding only maps Arrow carriers to
//! the canonical ones) and delimiters are strings. The kernels build the
//! maps; this family computes the output Arrow types from the Spark types
//! (exactly the analyzer's, via [`krishiv_types`]) and reads the session's
//! map-key dedup policy from [`BindOptions`].

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StringArray};
use arrow_schema::{DataType, Field, FieldRef};
use krishiv_common::error::KrishivError;
use krishiv_kernels::map::{self as kh, DedupPolicy};
use krishiv_types::{DataType as SparkType, MapType};

use super::args::{Inputs, internal, is_string, to_utf8};
use super::regex::Splitter;
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "map",
    "map_concat",
    "map_contains_key",
    "map_entries",
    "map_from_arrays",
    "map_from_entries",
    "map_keys",
    "map_values",
    "str_to_map",
];

/// The Spark type of an Arrow carrier (what the analyzer typed).
fn spark_type(name: &str, args: &[BindArg], t: &DataType) -> Result<SparkType, KrishivError> {
    SparkType::from_arrow(t).map_err(|_| bad_types(name, args))
}

/// The output `entries` field of a map of `key`/`value` Spark types: the
/// analyzer's field, exactly.
fn entries_field(key: &SparkType, value: &SparkType, value_nullable: bool) -> FieldRef {
    let entries = DataType::Struct(
        vec![
            key.to_arrow_field("key", false),
            value.to_arrow_field("value", value_nullable),
        ]
        .into(),
    );
    Arc::new(Field::new("entries", entries, false))
}

/// The output map type of `key`/`value` Spark types.
fn map_type(key: &SparkType, value: &SparkType, value_nullable: bool) -> DataType {
    DataType::Map(entries_field(key, value, value_nullable), false)
}

/// The output list type of `element` under the field `name`.
fn list_type(element: &Field, nullable: bool) -> DataType {
    DataType::List(Arc::new(Field::new(
        "element",
        element.data_type().clone(),
        nullable,
    )))
}

/// Binds the map function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let policy: DedupPolicy = opts.map_key_dedup_policy;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("map", _) if args.len().is_multiple_of(2) => map_bind(args, &types, policy),
        ("map_concat", []) => map_concat_bind(args, &[], policy),
        ("map_concat", _) if !args.is_empty() => map_concat_bind(args, &types, policy),
        ("map_contains_key", [m, k]) if matches!(m, DataType::Map(_, _)) && kh::supports_key(k) => {
            Ok(Binding::new(DataType::Boolean, move |x: &Inputs<'_>| {
                let (map, key) = (x.column(0)?, x.column(1)?);
                Ok(Arc::new(kh::contains_key(&map, &key)?) as ArrayRef)
            }))
        }
        ("map_keys" | "map_values" | "map_entries", [m @ DataType::Map(entries, _)])
            if kh::supports_value(m) =>
        {
            projection_bind(listed, entries)
        }
        ("map_from_arrays", [k, v])
            if matches!(k, DataType::List(_) | DataType::LargeList(_))
                && matches!(v, DataType::List(_) | DataType::LargeList(_))
                && kh::supports_key(k)
                && kh::supports_value(v) =>
        {
            from_arrays_bind(args, k, v, policy)
        }
        ("map_from_entries", [e]) if kh::supports_value(e) => from_entries_bind(args, e, policy),
        ("str_to_map", [t] | [t, _] | [t, _, _]) if is_string(t) => str_to_map_bind(args, policy),
        _ => Err(bad_types(listed, args)),
    };
    Some((listed, binding))
}

/// `map(k1, v1, ...)`: one entry per key/value column pair per row.
fn map_bind(
    args: &[BindArg],
    types: &[DataType],
    policy: DedupPolicy,
) -> Result<Binding, KrishivError> {
    let listed = "map";
    let mut key = SparkType::Null;
    let mut value = SparkType::Null;
    let mut first = true;
    for (k, v) in types.iter().step_by(2).zip(types.iter().skip(1).step_by(2)) {
        if !kh::supports_key(k) || !kh::supports_value(v) {
            return Err(bad_types(listed, args));
        }
        let (ks, vs) = (spark_type(listed, args, k)?, spark_type(listed, args, v)?);
        if first {
            key = ks;
            value = vs;
            first = false;
        } else if ks != key || vs != value {
            return Err(bad_types(listed, args));
        }
    }
    let value_nullable = args.iter().skip(1).step_by(2).any(BindArg::nullable);
    let out = map_type(&key, &value, value_nullable);
    let entries = map_entries_of(&out).ok_or_else(|| bad_types(listed, args))?;
    let n = args.len();
    Ok(Binding::new(out, move |x: &Inputs<'_>| {
        let mut keys = Vec::with_capacity(n / 2);
        let mut values = Vec::with_capacity(n / 2);
        for i in (0..n).step_by(2) {
            keys.push(x.column(i)?);
            values.push(x.column(i + 1)?);
        }
        Ok(Arc::new(kh::create_map(&keys, &values, x.rows(), &entries, policy)?) as ArrayRef)
    }))
}

/// `map_concat(m1, ...)`: the entries of every map per row, in order.
fn map_concat_bind(
    args: &[BindArg],
    types: &[DataType],
    policy: DedupPolicy,
) -> Result<Binding, KrishivError> {
    let listed = "map_concat";
    let mut merged: Option<SparkType> = None;
    for t in types {
        if !matches!(t, DataType::Map(_, _)) || !kh::supports_value(t) {
            return Err(bad_types(listed, args));
        }
        let s = spark_type(listed, args, t)?;
        merged = Some(match merged {
            None => s,
            Some(m) => m.union_like_merge(&s),
        });
    }
    let merged = merged.unwrap_or(SparkType::Map(Arc::new(MapType::new(
        SparkType::String,
        SparkType::String,
        true,
    ))));
    let out = merged.to_arrow();
    let entries = map_entries_of(&out).ok_or_else(|| bad_types(listed, args))?;
    let n = args.len();
    Ok(Binding::new(out, move |x: &Inputs<'_>| {
        let maps = (0..n).map(|i| x.column(i)).collect::<Result<Vec<_>, _>>()?;
        Ok(Arc::new(kh::concat(&maps, x.rows(), &entries, policy)?) as ArrayRef)
    }))
}

/// `map_keys`, `map_values`, `map_entries`: the projection of the map.
fn projection_bind(listed: &'static str, entries: &FieldRef) -> Result<Binding, KrishivError> {
    let out = match listed {
        "map_keys" => {
            let key = spark_map_key(entries)?;
            list_type(&key.to_arrow_field("element", true), true)
        }
        "map_values" => {
            let value = spark_map_value(entries)?;
            list_type(&value.to_arrow_field("element", true), true)
        }
        _ => DataType::List(Arc::new(Field::new(
            "element",
            DataType::Struct(
                vec![
                    spark_map_key(entries)?.to_arrow_field("key", false),
                    spark_map_value(entries)?.to_arrow_field("value", map_value_nullable(entries)?),
                ]
                .into(),
            ),
            false,
        ))),
    };
    let element = field_of(&out)?;
    Ok(Binding::new(out, move |x: &Inputs<'_>| {
        let map = x.column(0)?;
        Ok(match listed {
            "map_keys" => kh::keys(&map, &element)?,
            "map_values" => kh::values(&map, &element)?,
            _ => kh::entries(&map, &element)?,
        })
    }))
}

/// `map_from_arrays(keys, values)`: row `r` maps the `r`-th key slice to
/// the `r`-th value slice.
fn from_arrays_bind(
    args: &[BindArg],
    k: &DataType,
    v: &DataType,
    policy: DedupPolicy,
) -> Result<Binding, KrishivError> {
    let listed = "map_from_arrays";
    let (ks, vs) = (spark_type(listed, args, k)?, spark_type(listed, args, v)?);
    let (SparkType::Array(kt), SparkType::Array(vt)) = (ks, vs) else {
        return Err(bad_types(listed, args));
    };
    let out = map_type(kt.element_type(), vt.element_type(), vt.contains_null());
    let entries = map_entries_of(&out).ok_or_else(|| bad_types(listed, args))?;
    Ok(Binding::new(out, move |x: &Inputs<'_>| {
        let (keys, values) = (x.column(0)?, x.column(1)?);
        Ok(Arc::new(kh::from_arrays(&keys, &values, x.rows(), &entries, policy)?) as ArrayRef)
    }))
}

/// `map_from_entries(entries)`: the entry structs' fields become the map's
/// key and value.
fn from_entries_bind(
    args: &[BindArg],
    e: &DataType,
    policy: DedupPolicy,
) -> Result<Binding, KrishivError> {
    let entries = entries_field_of("map_from_entries", args, e)?;
    let out = DataType::Map(Arc::clone(&entries), false);
    Ok(Binding::new(out, move |x: &Inputs<'_>| {
        let array = x.column(0)?;
        Ok(Arc::new(kh::from_entries(&array, x.rows(), &entries, policy)?) as ArrayRef)
    }))
}

/// `str_to_map(text[, pairDelim[, keyValueDelim]])`: splits every row's
/// text and assembles the pairs.
fn str_to_map_bind(args: &[BindArg], policy: DedupPolicy) -> Result<Binding, KrishivError> {
    let out = map_type(&SparkType::String, &SparkType::String, true);
    let entries = map_entries_of(&out).ok_or_else(|| bad_types("str_to_map", args))?;
    let n = args.len();
    Ok(Binding::new(out, move |x: &Inputs<'_>| {
        str_to_map(x, n, &entries, policy)
    }))
}

/// The `entries` field of an output map type.
fn map_entries_of(out: &DataType) -> Option<FieldRef> {
    match out {
        DataType::Map(e, _) => Some(Arc::clone(e)),
        _ => None,
    }
}

/// The key/value Spark type of a map entries field.
fn spark_map_key(entries: &FieldRef) -> Result<SparkType, KrishivError> {
    match entries.data_type() {
        DataType::Struct(fs) if fs.len() == 2 => {
            SparkType::from_arrow(fs[0].data_type()).map_err(|e| internal(e.to_string()))
        }
        t => Err(internal(format!("map entries of type {t}"))),
    }
}

fn spark_map_value(entries: &FieldRef) -> Result<SparkType, KrishivError> {
    match entries.data_type() {
        DataType::Struct(fs) if fs.len() == 2 => {
            SparkType::from_arrow(fs[1].data_type()).map_err(|e| internal(e.to_string()))
        }
        t => Err(internal(format!("map entries of type {t}"))),
    }
}

fn map_value_nullable(entries: &FieldRef) -> Result<bool, KrishivError> {
    match entries.data_type() {
        DataType::Struct(fs) if fs.len() == 2 => Ok(fs[1].is_nullable()),
        t => Err(internal(format!("map entries of type {t}"))),
    }
}

fn field_of(t: &DataType) -> Result<FieldRef, KrishivError> {
    match t {
        DataType::List(f) => Ok(Arc::clone(f)),
        t => Err(internal(format!("a list type {t}"))),
    }
}

/// The `entries` field of the array-of-structs argument of
/// `map_from_entries`: the struct's fields become the map's key and value.
fn entries_field_of(name: &str, args: &[BindArg], t: &DataType) -> Result<FieldRef, KrishivError> {
    let bad = || bad_types(name, args);
    let SparkType::Array(a) = spark_type(name, args, t)? else {
        return Err(bad());
    };
    let SparkType::Struct(s) = a.element_type().clone() else {
        return Err(bad());
    };
    if s.fields().len() != 2 {
        return Err(bad());
    }
    let (key, value) = (&s.fields()[0], &s.fields()[1]);
    Ok(entries_field(
        key.data_type(),
        value.data_type(),
        value.nullable(),
    ))
}

/// `str_to_map(text[, pairDelim[, keyValueDelim]])` over one batch: splits
/// every row's text and assembles the pairs.
fn str_to_map(
    x: &Inputs<'_>,
    n: usize,
    entries: &FieldRef,
    policy: DedupPolicy,
) -> Result<ArrayRef, KrishivError> {
    let rows = x.rows();
    let text = to_utf8(x.column(0)?.as_ref())?;
    let pair_delim = if n > 1 {
        Some(to_utf8(x.column(1)?.as_ref())?)
    } else {
        None
    };
    let kv_delim = if n > 2 {
        Some(to_utf8(x.column(2)?.as_ref())?)
    } else {
        None
    };
    fn delim_at(col: &arrow_array::StringArray, r: usize) -> Option<&str> {
        col.is_valid(r).then(|| col.value(r))
    }
    let mut pair_splitters: HashMap<String, Splitter> = HashMap::new();
    let mut kv_splitters: HashMap<String, Splitter> = HashMap::new();
    // The splitter for `d`, compiled once per distinct delimiter.
    let splitter =
        |cache: &mut HashMap<String, Splitter>, d: &str| -> Result<Splitter, KrishivError> {
            if let Some(s) = cache.get(d) {
                return Ok(s.clone());
            }
            let s = Splitter::new(d)?;
            cache.insert(d.to_owned(), s.clone());
            Ok(s)
        };
    let mut keys: Vec<Option<String>> = Vec::new();
    let mut values: Vec<Option<String>> = Vec::new();
    let mut row_of: Vec<usize> = Vec::new();
    for r in 0..rows {
        let Some(t) = text.is_valid(r).then(|| text.value(r)) else {
            continue;
        };
        let pair = pair_delim.as_ref().map_or(Some(","), |c| delim_at(c, r));
        let kv = kv_delim.as_ref().map_or(Some(":"), |c| delim_at(c, r));
        let (Some(pair), Some(kv)) = (pair, kv) else {
            continue;
        };
        let pair_split = splitter(&mut pair_splitters, pair)?;
        let kv_split = splitter(&mut kv_splitters, kv)?;
        for piece in pair_split.split(t, -1)? {
            let kvp = kv_split.split(&piece, 2)?;
            let mut kvp = kvp.into_iter();
            keys.push(kvp.next());
            values.push(kvp.next());
            row_of.push(r);
        }
    }
    let keys = Arc::new(StringArray::from(keys)) as ArrayRef;
    let values = Arc::new(StringArray::from(values)) as ArrayRef;
    Ok(Arc::new(kh::from_pairs(
        &keys, &values, &row_of, rows, entries, policy,
    )?) as ArrayRef)
}
