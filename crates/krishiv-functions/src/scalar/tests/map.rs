// SPDX-License-Identifier: Apache-2.0
//! The map family's bindings: row maps, projections, entry builders and
//! `str_to_map` over columns and literals.

use std::sync::Arc;

use arrow_array::builder::{ArrayBuilder, Int32Builder, ListBuilder, StringBuilder, StructBuilder};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, MapArray, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::map::DedupPolicy;

use super::super::{BindArg, BindOptions, bind_scalar};
use super::{bind, bind_with, col, ints, lit, run, strs};
use arrow_array::types::Int32Type;

fn last_win() -> BindOptions {
    BindOptions::new().with_map_key_dedup_policy(DedupPolicy::LastWin)
}

type Pairs = Vec<Option<Vec<(String, Option<String>)>>>;

fn view_pairs(m: &MapArray) -> Pairs {
    let key = |p: usize| match m.keys().data_type() {
        DataType::Int32 => m.keys().as_primitive::<Int32Type>().value(p).to_string(),
        _ => m.keys().as_string_view().value(p).to_owned(),
    };
    (0..m.len())
        .map(|r| {
            if m.is_null(r) {
                return None;
            }
            let offsets = m.value_offsets();
            let (from, to) = (
                usize::try_from(offsets[r]).unwrap_or(0),
                usize::try_from(offsets[r + 1]).unwrap_or(0),
            );
            let mut row = Vec::new();
            for p in from..to {
                let v = m
                    .values()
                    .is_valid(p)
                    .then(|| m.values().as_string_view().value(p).to_owned());
                row.push((key(p), v));
            }
            Some(row)
        })
        .collect()
}

/// `map(k, v)` over two columns, as the analyzer casts them.
fn int_str_map(keys: &ArrayRef, values: &ArrayRef) -> ArrayRef {
    run("map", &[keys, values])
}

#[test]
fn map_builds_a_map_per_row() {
    let keys = ints(&[Some(1), Some(2), Some(1)]);
    let values = strs(&[Some("a"), None, Some("c")]);
    let out = int_str_map(&keys, &values);
    // Keys are `Int32`, values the canonical `Utf8View`.
    assert_eq!(
        out.data_type(),
        &DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("key", DataType::Int32, false),
                        Field::new("value", DataType::Utf8View, true),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        )
    );
    let m = out.as_map();
    assert_eq!(
        view_pairs(m),
        [
            Some(vec![("1".to_owned(), Some("a".to_owned()))]),
            Some(vec![("2".to_owned(), None)]),
            Some(vec![("1".to_owned(), Some("c".to_owned()))]),
        ]
    );
}

#[test]
fn map_value_nullability_follows_the_arguments() {
    let keys = ints(&[Some(1)]);
    let values = strs(&[Some("a")]);
    // Nullable value columns give nullable map values; literals do not.
    let bound = vec![
        BindArg::Column {
            ty: DataType::Int32,
            nullable: false,
        },
        BindArg::Column {
            ty: DataType::Utf8,
            nullable: true,
        },
    ];
    let f = bind_scalar("map", &bound, &BindOptions::new()).unwrap();
    let nullable = match f.return_type() {
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(fs) => fs[1].is_nullable(),
            _ => false,
        },
        _ => false,
    };
    assert!(nullable);
    let lit_args = vec![
        BindArg::Literal(Arc::clone(&keys)),
        BindArg::Literal(Arc::clone(&values)),
    ];
    let f = bind_scalar("map", &lit_args, &BindOptions::new()).unwrap();
    let nullable = match f.return_type() {
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(fs) => fs[1].is_nullable(),
            _ => true,
        },
        _ => true,
    };
    assert!(!nullable);
}

#[test]
fn map_rejects_null_and_duplicate_keys() {
    let keys = ints(&[Some(1), None]);
    let values = strs(&[Some("a"), Some("b")]);
    let e = bind("map", &[&keys, &values])
        .invoke(&[col(&keys), col(&values)], 2)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::NullMapKey);

    // A null key still fails under `LAST_WIN`.
    let keys = ints(&[None]);
    let values = strs(&[Some("a")]);
    let f = bind_with("map", &[&keys, &values], &last_win());
    let e = f.invoke(&[col(&keys), col(&values)], 1).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::NullMapKey);

    // Duplicate keys within a row.
    let k1 = ints(&[Some(1), Some(1)]);
    let v1 = strs(&[Some("a"), Some("x")]);
    let k2 = ints(&[Some(1), Some(1)]);
    let v2 = strs(&[Some("b"), Some("y")]);
    let e = bind("map", &[&k1, &v1, &k2, &v2])
        .invoke(&[col(&k1), col(&v1), col(&k2), col(&v2)], 2)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);

    // `LAST_WIN` keeps the last value.
    let f = bind_with("map", &[&k1, &v1, &k2, &v2], &last_win());
    let out = f
        .invoke(&[col(&k1), col(&v1), col(&k2), col(&v2)], 2)
        .unwrap();
    assert_eq!(
        view_pairs(out.as_map()),
        [
            Some(vec![("1".to_owned(), Some("b".to_owned()))]),
            Some(vec![("1".to_owned(), Some("y".to_owned()))]),
        ]
    );
}

#[test]
fn map_rejects_wrong_shapes() {
    // Odd arguments and non-key types never bind.
    assert!(
        bind_scalar(
            "map",
            &[BindArg::from(DataType::Int32)],
            &BindOptions::new()
        )
        .is_err()
    );
    let fields: arrow_schema::Fields = vec![Field::new("f", DataType::Int32, true)].into();
    let st = Arc::new(
        StructBuilder::new(
            fields,
            vec![Box::new(Int32Builder::new()) as Box<dyn ArrayBuilder>],
        )
        .finish(),
    ) as ArrayRef;
    assert!(
        bind_scalar(
            "map",
            &[BindArg::Literal(st), BindArg::from(DataType::Int32)],
            &BindOptions::new(),
        )
        .is_err()
    );
}

#[test]
fn map_concat_merges_in_order() {
    let left = int_str_map(&ints(&[Some(1)]), &strs(&[Some("a")]));
    let right = int_str_map(&ints(&[Some(2)]), &strs(&[Some("b")]));
    let out = run("map_concat", &[&left, &right]);
    assert_eq!(
        view_pairs(out.as_map()),
        [Some(vec![
            ("1".to_owned(), Some("a".to_owned())),
            ("2".to_owned(), Some("b".to_owned())),
        ])]
    );

    // No maps is an empty map per row.
    let f = bind_scalar("map_concat", &[], &BindOptions::new()).unwrap();
    assert_eq!(
        f.return_type(),
        &DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("key", DataType::Utf8View, false),
                        Field::new("value", DataType::Utf8View, true),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        )
    );
    assert_eq!(f.invoke(&[], 2).unwrap().len(), 2);

    // A null map voids its row under either policy.
    let null_keys = arrow_array::new_null_array(
        &DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        1,
    );
    let null_values = arrow_array::new_null_array(
        &DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        1,
    );
    let null = run("map_from_arrays", &[&null_keys, &null_values]);
    assert_eq!(null.null_count(), 1);

    // Duplicates follow the policy.
    let e = bind("map_concat", &[&left, &left])
        .invoke(&[col(&left), col(&left)], 1)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);

    // Non-maps and untyped nulls never bind.
    assert!(
        bind_scalar(
            "map_concat",
            &[
                BindArg::from(DataType::Int32),
                BindArg::from(DataType::Int32)
            ],
            &BindOptions::new(),
        )
        .is_err()
    );
}

#[test]
fn map_contains_key_compares() {
    let map = int_str_map(&ints(&[Some(1), Some(2)]), &strs(&[Some("a"), Some("b")]));
    let one = ints(&[Some(1), Some(3)]);
    let out = run("map_contains_key", &[&map, &one]);
    assert_eq!(
        out.as_boolean().iter().collect::<Vec<_>>(),
        [Some(true), Some(false)]
    );
    // A null map or key is null; an untyped null key never binds.
    let null = arrow_array::new_null_array(out.data_type(), 2);
    let out = run("map_contains_key", &[&map, &null]);
    assert_eq!(out.null_count(), 2);
    assert!(
        bind_scalar(
            "map_contains_key",
            &[
                BindArg::from(DataType::Int32),
                BindArg::from(DataType::Int32)
            ],
            &BindOptions::new(),
        )
        .is_err()
    );
}

#[test]
fn map_projections_share_rows() {
    let map = int_str_map(&ints(&[Some(1), Some(2)]), &strs(&[Some("a"), None]));
    let keys = run("map_keys", &[&map]);
    assert_eq!(
        keys.as_list::<i32>()
            .values()
            .as_primitive::<arrow_array::types::Int32Type>()
            .values(),
        &[1, 2]
    );
    let values = run("map_values", &[&map]);
    assert_eq!(values.null_count(), 0);
    assert_eq!(values.as_list::<i32>().values().null_count(), 1);
    let entries = run("map_entries", &[&map]);
    let list = entries.as_list::<i32>();
    assert_eq!(list.len(), 2);
    let structs = list.values().as_struct();
    assert_eq!(structs.num_columns(), 2);
    assert_eq!(structs.len(), 2);
}

fn int_list(values: &[Option<i32>], breaks: &[usize]) -> ArrayRef {
    let mut b = ListBuilder::new(Int32Builder::new());
    let mut at = 0_usize;
    for n in breaks {
        let n = *n;
        for v in &values[at..at.saturating_add(n)] {
            match v {
                Some(v) => b.values().append_value(*v),
                None => b.values().append_null(),
            }
        }
        at = at.saturating_add(n);
        b.append(true);
    }
    Arc::new(b.finish())
}

fn str_list(values: &[Option<&str>], breaks: &[usize]) -> ArrayRef {
    let mut b = ListBuilder::new(StringBuilder::new());
    let mut at = 0_usize;
    for n in breaks {
        let n = *n;
        for v in &values[at..at.saturating_add(n)] {
            match v {
                Some(v) => b.values().append_value(v),
                None => b.values().append_null(),
            }
        }
        at = at.saturating_add(n);
        b.append(true);
    }
    Arc::new(b.finish())
}

#[test]
fn map_from_arrays_maps_slices() {
    let keys = int_list(&[Some(1), Some(2)], &[2]);
    let values = str_list(&[Some("a"), Some("b")], &[2]);
    let out = run("map_from_arrays", &[&keys, &values]);
    let m = out.as_map();
    assert_eq!(m.len(), 1);
    assert_eq!(
        view_pairs(m),
        [Some(vec![
            ("1".to_owned(), Some("a".to_owned())),
            ("2".to_owned(), Some("b".to_owned())),
        ])]
    );

    // Slices of different lengths are `_LEGACY_ERROR_TEMP_2128`.
    let short = int_list(&[Some(1)], &[1]);
    let e = bind("map_from_arrays", &[&short, &values])
        .invoke(&[col(&short), col(&values)], 1)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::LegacyErrorTemp2128);

    // A null key element is `NULL_MAP_KEY`.
    let null_key = int_list(&[Some(1), None], &[2]);
    let e = bind("map_from_arrays", &[&null_key, &values])
        .invoke(&[col(&null_key), col(&values)], 1)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::NullMapKey);

    // Duplicates follow the policy.
    let dup = int_list(&[Some(1), Some(1)], &[2]);
    let e = bind("map_from_arrays", &[&dup, &values])
        .invoke(&[col(&dup), col(&values)], 1)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);
    let f = bind_with("map_from_arrays", &[&dup, &values], &last_win());
    let out = f.invoke(&[col(&dup), col(&values)], 1).unwrap();
    assert_eq!(
        view_pairs(out.as_map()),
        [Some(vec![("1".to_owned(), Some("b".to_owned()))])]
    );
}

fn struct_list(keys: &ArrayRef, values: &ArrayRef, breaks: &[usize], nulls: &[bool]) -> ArrayRef {
    // keys/values hold every row's entries in order; `breaks[r]` counts
    // row r's.
    let fields: arrow_schema::Fields = vec![
        Field::new("key", DataType::Int32, true),
        Field::new("value", DataType::Utf8View, true),
    ]
    .into();
    let mut key_builder = Int32Builder::new();
    let mut value_builder = arrow_array::builder::StringViewBuilder::new();
    let va = values.as_string::<i32>();
    let mut at = 0_usize;
    let mut offsets = vec![0_i32];
    for (r, n) in breaks.iter().enumerate() {
        if nulls[r] {
            offsets.push(offsets[offsets.len() - 1]);
            continue;
        }
        let ka = keys.as_primitive::<Int32Type>();
        for _ in 0..*n {
            if ka.is_null(at) {
                key_builder.append_null();
            } else {
                key_builder.append_value(ka.value(at));
            }
            if va.is_null(at) {
                value_builder.append_null();
            } else {
                value_builder.append_value(va.value(at));
            }
            at = at.saturating_add(1);
        }
        offsets.push(i32::try_from(key_builder.len()).unwrap_or(i32::MAX));
    }
    let structs = StructArray::new(
        fields,
        vec![
            Arc::new(key_builder.finish()) as ArrayRef,
            Arc::new(value_builder.finish()),
        ],
        None,
    );
    Arc::new(arrow_array::ListArray::new(
        Arc::new(Field::new(
            "item",
            DataType::Struct(
                vec![
                    Field::new("key", DataType::Int32, true),
                    Field::new("value", DataType::Utf8View, true),
                ]
                .into(),
            ),
            true,
        )),
        arrow_buffer::OffsetBuffer::new(offsets.into()),
        Arc::new(structs),
        nulls
            .iter()
            .any(|n| *n)
            .then(|| NullBuffer::from_iter(nulls.iter().map(|n| !n))),
    ))
}

#[test]
fn map_from_entries_maps_structs() {
    let keys = ints(&[Some(1), Some(2)]);
    let values = strs(&[Some("a"), Some("b")]);
    let array = struct_list(&keys, &values, &[2], &[false]);
    let out = run("map_from_entries", &[&array]);
    assert_eq!(
        view_pairs(out.as_map()),
        [Some(vec![
            ("1".to_owned(), Some("a".to_owned())),
            ("2".to_owned(), Some("b".to_owned())),
        ])]
    );

    // A null element voids the whole result.
    let array = struct_list(&keys, &values, &[2], &[true]);
    let out = run("map_from_entries", &[&array]);
    assert_eq!(out.null_count(), 1);

    // A null key field is `NULL_MAP_KEY`.
    let keys = ints(&[None]);
    let values = strs(&[Some("a")]);
    let array = struct_list(&keys, &values, &[1], &[false]);
    let e = bind("map_from_entries", &[&array])
        .invoke(&[col(&array)], 1)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::NullMapKey);

    // Anything but an array of pairs never binds.
    assert!(
        bind_scalar(
            "map_from_entries",
            &[BindArg::from(DataType::Int32)],
            &BindOptions::new()
        )
        .is_err()
    );
}

#[test]
fn str_to_map_splits_rows() {
    let text = strs(&[Some("a:1,b:2,c:3"), Some(""), None]);
    let out = run("str_to_map", &[&text]);
    assert_eq!(
        view_pairs(out.as_map()),
        [
            Some(vec![
                ("a".to_owned(), Some("1".to_owned())),
                ("b".to_owned(), Some("2".to_owned())),
                ("c".to_owned(), Some("3".to_owned())),
            ]),
            Some(vec![("".to_owned(), None)]),
            None,
        ]
    );

    // Custom delimiters, and a missing value is null.
    let pair = strs(&[Some(";")]);
    let kv = strs(&[Some("=")]);
    let f = bind_scalar(
        "str_to_map",
        &[
            BindArg::from(DataType::Utf8),
            BindArg::Literal(Arc::clone(&pair)),
            BindArg::Literal(Arc::clone(&kv)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    let text = strs(&[Some("a=1;b")]);
    let out = f.invoke(&[col(&text), lit(&pair), lit(&kv)], 1).unwrap();
    assert_eq!(
        view_pairs(out.as_map()),
        [Some(vec![
            ("a".to_owned(), Some("1".to_owned())),
            ("b".to_owned(), None),
        ])]
    );

    // Duplicate keys follow the policy.
    let dup = strs(&[Some("a:1,a:2")]);
    let e = bind("str_to_map", &[&dup])
        .invoke(&[col(&dup)], 1)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);
    let f = bind_with("str_to_map", &[&dup], &last_win());
    let out = f.invoke(&[col(&dup)], 1).unwrap();
    assert_eq!(
        view_pairs(out.as_map()),
        [Some(vec![("a".to_owned(), Some("2".to_owned())),])]
    );
}
