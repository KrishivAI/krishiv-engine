// SPDX-License-Identifier: Apache-2.0
//! Tests of the map kernels: key dedup policies and map construction.

use super::*;
use arrow_array::{Decimal128Array, Int32Array, StringArray, StringViewArray};
use arrow_buffer::NullBufferBuilder;
use arrow_schema::Field;
use krishiv_common::error::ErrorClass;

fn test_entries(key: DataType, value: DataType, value_nullable: bool) -> FieldRef {
    let entries = DataType::Struct(
        vec![
            Field::new("key", key, false),
            Field::new("value", value, value_nullable),
        ]
        .into(),
    );
    Arc::new(Field::new("entries", entries, false))
}

fn strings(v: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(v.to_vec()))
}

fn views(v: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringViewArray::from(
        v.iter().map(|s| s.as_deref()).collect::<Vec<_>>(),
    ))
}

fn ints(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(v.to_vec()))
}

fn int_field_entries() -> FieldRef {
    test_entries(DataType::Int32, DataType::Utf8View, true)
}

type Pairs = Vec<Option<Vec<(String, Option<String>)>>>;

/// The `(key, value)` pairs of every map row, rendered for assertions.
fn pairs(m: &MapArray) -> Pairs {
    (0..m.len())
        .map(|r| {
            if m.is_null(r) {
                return None;
            }
            let (from, to) = map_range(m, r).unwrap();
            let mut row = Vec::new();
            for p in from..to {
                let k = key_text(m.keys().as_ref(), p).expect("key text renders");
                let v = if m.values().is_null(p) {
                    None
                } else {
                    Some(string_value(m.values().as_ref(), p).to_owned())
                };
                row.push((k, v));
            }
            Some(row)
        })
        .collect()
}

/// A `MapArray` input over `entries`, with `counts[r]` entries in row
/// `r` (`null_rows[r]` is null).
fn map_input(
    keys: ArrayRef,
    values: ArrayRef,
    entries: &FieldRef,
    counts: &[usize],
    null_rows: &[bool],
) -> MapArray {
    let mut offsets = vec![0_i32];
    for (n, null) in counts.iter().zip(null_rows) {
        let next = if *null {
            offsets.last().copied().unwrap_or(0)
        } else {
            offsets
                .last()
                .copied()
                .unwrap_or(0)
                .saturating_add(i32::try_from(*n).unwrap_or(0))
        };
        offsets.push(next);
    }
    let fields = match entries.data_type() {
        DataType::Struct(fs) => fs.clone(),
        _ => unreachable!(),
    };
    let structs = StructArray::try_new(fields, vec![keys, values], None).unwrap();
    let mut nulls = NullBufferBuilder::new(counts.len());
    for null in null_rows {
        nulls.append(!null);
    }
    MapArray::try_new(
        Arc::clone(entries),
        OffsetBuffer::new(offsets.into()),
        structs,
        nulls.finish(),
        false,
    )
    .unwrap()
}

fn list_of(values: ArrayRef, breaks: &[usize], nulls: &[bool]) -> ArrayRef {
    let mut offsets = vec![0_i32];
    for (n, null) in breaks.iter().zip(nulls) {
        let next = if *null {
            offsets.last().copied().unwrap_or(0)
        } else {
            offsets
                .last()
                .copied()
                .unwrap_or(0)
                .saturating_add(i32::try_from(*n).unwrap_or(0))
        };
        offsets.push(next);
    }
    let mut valid = NullBufferBuilder::new(breaks.len());
    for null in nulls {
        valid.append(!null);
    }
    Arc::new(ListArray::new(
        Arc::new(Field::new("item", values.data_type().clone(), true)),
        OffsetBuffer::new(offsets.into()),
        values,
        valid.finish(),
    ))
}

#[test]
fn equal_keys_cover_every_carrier() {
    assert!(keys_equal(&ints(&[Some(1)]), 0, &ints(&[Some(1)]), 0).unwrap());
    assert!(!keys_equal(&ints(&[Some(1)]), 0, &ints(&[Some(2)]), 0).unwrap());
    // A null equals only a null.
    assert!(keys_equal(&ints(&[None]), 0, &ints(&[None]), 0).unwrap());
    assert!(!keys_equal(&ints(&[None]), 0, &ints(&[Some(1)]), 0).unwrap());
    // Strings compare byte-wise across carriers.
    assert!(keys_equal(&strings(&[Some("a")]), 0, &views(&[Some("a")]), 0).unwrap());
    assert!(!keys_equal(&strings(&[Some("a")]), 0, &views(&[Some("b")]), 0).unwrap());
    // Floats: NaN equals NaN, -0.0 equals 0.0.
    let nan = Arc::new(arrow_array::Float64Array::from(vec![f64::NAN])) as ArrayRef;
    assert!(keys_equal(nan.as_ref(), 0, nan.as_ref(), 0).unwrap());
    let (neg, pos) = (
        Arc::new(arrow_array::Float64Array::from(vec![-0.0])) as ArrayRef,
        Arc::new(arrow_array::Float64Array::from(vec![0.0])) as ArrayRef,
    );
    assert!(keys_equal(neg.as_ref(), 0, pos.as_ref(), 0).unwrap());
    // Decimals compare by unscaled value.
    let dec = |u: i128| {
        Arc::new(
            Decimal128Array::from(vec![u])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ) as ArrayRef
    };
    assert!(keys_equal(&dec(100), 0, &dec(100), 0).unwrap());
    assert!(!keys_equal(&dec(100), 0, &dec(101), 0).unwrap());
    // Nested values compare element by element.
    let l1 = list_of(ints(&[Some(1), None]), &[2], &[false]);
    let l2 = list_of(ints(&[Some(1), None]), &[2], &[false]);
    let l3 = list_of(ints(&[Some(1), Some(2)]), &[2], &[false]);
    assert!(keys_equal(l1.as_ref(), 0, l2.as_ref(), 0).unwrap());
    assert!(!keys_equal(l1.as_ref(), 0, l3.as_ref(), 0).unwrap());
    // Types that cannot meet are an internal error, not a panic.
    assert!(keys_equal(&ints(&[Some(1)]), 0, &strings(&[Some("1")]), 0).is_err());
}

#[test]
fn create_map_builds_a_map_per_row() {
    let keys = [ints(&[Some(1), Some(3)]), ints(&[Some(2), Some(4)])];
    let values = [
        strings(&[Some("a"), None]),
        strings(&[Some("b"), Some("d")]),
    ];
    let m = create_map(
        &keys,
        &values,
        2,
        &int_field_entries(),
        DedupPolicy::Exception,
    )
    .unwrap();
    assert_eq!(
        pairs(&m),
        [
            Some(vec![
                ("1".to_owned(), Some("a".to_owned())),
                ("2".to_owned(), Some("b".to_owned())),
            ]),
            Some(vec![
                ("3".to_owned(), None),
                ("4".to_owned(), Some("d".to_owned())),
            ]),
        ]
    );
    // The output carries the entries field: struct `key` (not null) and
    // `value`, over `Utf8View` values.
    assert_eq!(m.data_type(), &DataType::Map(int_field_entries(), false));
    assert_eq!(m.value_offsets().len(), 3);
}

#[test]
fn create_map_rejects_null_and_duplicate_keys() {
    let keys = [ints(&[Some(1), None])];
    let values = [strings(&[Some("a"), Some("b")])];
    let e = create_map(
        &keys,
        &values,
        2,
        &int_field_entries(),
        DedupPolicy::Exception,
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::NullMapKey);
    assert_eq!(e.row(), Some(1));

    let keys = [ints(&[Some(1)]), ints(&[Some(1)])];
    let values = [strings(&[Some("a")]), strings(&[Some("b")])];
    let e = create_map(
        &keys,
        &values,
        1,
        &int_field_entries(),
        DedupPolicy::Exception,
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);
    assert_eq!(e.row(), Some(0));
    assert!(
        e.error()
            .message()
            .contains("Duplicate map key 1 was found")
    );
    let params: Vec<(&str, String)> = e
        .error()
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert!(params.contains(&("key", "1".to_owned())));
    assert!(params.contains(&(
        "mapKeyDedupPolicy",
        "\"spark.sql.mapKeyDedupPolicy\"".to_owned()
    )));

    // `LAST_WIN` keeps the first position with the last value.
    let m = create_map(
        &keys,
        &values,
        1,
        &int_field_entries(),
        DedupPolicy::LastWin,
    )
    .unwrap();
    assert_eq!(
        pairs(&m),
        [Some(vec![("1".to_owned(), Some("b".to_owned()))])]
    );
}

#[test]
fn create_map_empty_is_an_empty_map_per_row() {
    let m = create_map(&[], &[], 2, &int_field_entries(), DedupPolicy::Exception).unwrap();
    assert_eq!(pairs(&m), [Some(vec![]), Some(vec![])]);
}

#[test]
fn create_map_hashes_align_with_rows() {
    // The dedup hash of pair `i`, row `r` is read row-major: a repeat
    // in a later row is still found (a column-major read would miss
    // it, as the hashes differ).
    let keys = [ints(&[Some(1), Some(1)]), ints(&[Some(2), Some(1)])];
    let values = [
        strings(&[Some("a"), Some("b")]),
        strings(&[Some("c"), Some("d")]),
    ];
    let e = create_map(
        &keys,
        &values,
        2,
        &int_field_entries(),
        DedupPolicy::Exception,
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);
    assert_eq!(e.row(), Some(1));
}

#[test]
fn create_map_nests_arrays_structs_and_maps() {
    // Array keys.
    let k1 = list_of(ints(&[Some(1), Some(2)]), &[2], &[false]);
    let k2 = list_of(ints(&[Some(1), Some(2)]), &[2], &[false]);
    let v = [strings(&[Some("a")]), strings(&[Some("b")])];
    let e = create_map(
        &[Arc::clone(&k1), Arc::clone(&k2)],
        &v,
        1,
        &test_entries(
            DataType::List(Arc::new(Field::new("element", DataType::Int32, true))),
            DataType::Utf8View,
            true,
        ),
        DedupPolicy::Exception,
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);

    // Map values.
    let inner = create_map(
        &[ints(&[Some(7)])],
        &[strings(&[Some("x")])],
        1,
        &int_field_entries(),
        DedupPolicy::Exception,
    )
    .unwrap();
    let m = create_map(
        &[strings(&[Some("m")])],
        &[Arc::new(inner) as ArrayRef],
        1,
        &test_entries(
            DataType::Utf8View,
            DataType::Map(int_field_entries(), false),
            false,
        ),
        DedupPolicy::Exception,
    )
    .unwrap();
    assert_eq!(m.len(), 1);
    assert!(!m.is_null(0));
    let (from, to) = map_range(&m, 0).unwrap();
    assert_eq!((to - from), 1);
}

#[test]
fn create_map_scales_to_thousands_of_rows() {
    let rows = 2_000_usize;
    let keys = [Arc::new(Int32Array::from_iter_values(
        0..i32::try_from(rows).unwrap_or(0),
    )) as ArrayRef];
    let values = [strings(&vec![Some("v"); rows])];
    let m = create_map(
        &keys,
        &values,
        rows,
        &int_field_entries(),
        DedupPolicy::Exception,
    )
    .unwrap();
    assert_eq!(m.len(), rows);
    assert_eq!(
        m.value_offsets().last().copied().unwrap_or(0),
        i32::try_from(rows).unwrap_or(-1)
    );
    let (from, to) = map_range(&m, rows - 1).unwrap();
    assert_eq!((to - from), 1);
    assert_eq!(
        key_text(m.keys().as_ref(), from).expect("key text renders"),
        (i32::try_from(rows).unwrap_or(0).saturating_sub(1)).to_string()
    );
}

#[test]
fn concat_merges_with_nulls_and_policy() {
    let entries = int_field_entries();
    let left = map_input(
        ints(&[Some(1)]),
        views(&[Some("a")]),
        &entries,
        &[1],
        &[false],
    );
    let right = Arc::new(map_input(
        ints(&[Some(1)]),
        views(&[Some("B")]),
        &entries,
        &[1],
        &[false],
    )) as ArrayRef;
    // Key 1 in both maps is a duplicate under `EXCEPTION`.
    let e = concat(
        &[Arc::new(left.clone()) as ArrayRef, Arc::clone(&right)],
        1,
        &entries,
        DedupPolicy::Exception,
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);
    assert_eq!(e.row(), Some(0));

    // Under `LAST_WIN` the later map's value wins in place.
    let m = concat(
        &[Arc::new(left.clone()) as ArrayRef, Arc::clone(&right)],
        1,
        &entries,
        DedupPolicy::LastWin,
    )
    .unwrap();
    assert_eq!(
        pairs(&m),
        [Some(vec![("1".to_owned(), Some("B".to_owned()))])]
    );
    let _ = right;

    // Disjoint maps merge; a null map voids its row under either
    // policy, whatever the other maps hold.
    let other = Arc::new(map_input(
        ints(&[Some(2)]),
        views(&[Some("b")]),
        &entries,
        &[1],
        &[false],
    )) as ArrayRef;
    let m = concat(
        &[Arc::new(left) as ArrayRef, Arc::clone(&other)],
        1,
        &entries,
        DedupPolicy::Exception,
    )
    .unwrap();
    assert_eq!(
        pairs(&m),
        [Some(vec![
            ("1".to_owned(), Some("a".to_owned())),
            ("2".to_owned(), Some("b".to_owned())),
        ])]
    );

    let void = Arc::new(map_input(ints(&[]), views(&[]), &entries, &[0], &[true])) as ArrayRef;
    for policy in [DedupPolicy::Exception, DedupPolicy::LastWin] {
        let m = concat(
            &[Arc::clone(&void), Arc::clone(&other)],
            1,
            &entries,
            policy,
        )
        .unwrap();
        assert_eq!(pairs(&m), [None]);
    }

    // No maps is an empty map per row.
    let m = concat(&[], 2, &entries, DedupPolicy::Exception).unwrap();
    assert_eq!(pairs(&m), [Some(vec![]), Some(vec![])]);
}

#[test]
fn contains_key_scans_with_nulls() {
    let entries = int_field_entries();
    let map = map_input(
        ints(&[Some(1), Some(2), Some(0)]),
        views(&[Some("a"), Some("b"), Some("c")]),
        &entries,
        &[2, 0, 0],
        &[false, false, true],
    );
    let out = contains_key(
        &(Arc::new(map) as ArrayRef),
        &ints(&[Some(2), Some(1), None]),
    )
    .unwrap();
    assert_eq!(out, BooleanArray::from(vec![Some(true), Some(false), None]));
}

#[test]
fn contains_key_normalises_floats() {
    // `-0.0` finds `0.0`, NaN finds NaN: Spark's key equality.
    let entries = test_entries(DataType::Float64, DataType::Utf8View, true);
    let map = map_input(
        Arc::new(arrow_array::Float64Array::from(vec![Some(0.0)])) as ArrayRef,
        views(&[Some("zero")]),
        &entries,
        &[1],
        &[false],
    );
    let out = contains_key(
        &(Arc::new(map) as ArrayRef),
        &(Arc::new(arrow_array::Float64Array::from(vec![Some(-0.0)])) as ArrayRef),
    )
    .unwrap();
    assert_eq!(out, BooleanArray::from(vec![Some(true)]));
}

#[test]
fn projections_share_children() {
    let map_field = test_entries(DataType::Int32, DataType::Utf8View, true);
    let map = map_input(
        ints(&[Some(1), Some(2)]),
        views(&[Some("a"), None]),
        &map_field,
        &[2, 0],
        &[false, true],
    );
    let map = Arc::new(map) as ArrayRef;
    let key_field = Arc::new(Field::new("element", DataType::Int32, true));
    let k = keys(&map, &key_field).unwrap();
    let l = k.as_list::<i32>();
    assert_eq!(l.len(), 2);
    assert!(l.is_null(1));
    assert_eq!(l.value_offsets(), &[0_i32, 2, 2][..]);

    let value_field = Arc::new(Field::new("element", DataType::Utf8View, true));
    let v = values(&map, &value_field).unwrap();
    assert_eq!(v.len(), 2);

    let entry_field = Arc::new(Field::new(
        "element",
        DataType::Struct(
            vec![
                Field::new("key", DataType::Int32, false),
                Field::new("value", DataType::Utf8View, true),
            ]
            .into(),
        ),
        false,
    ));
    let e = entries(&map, &entry_field).unwrap();
    let l = e.as_list::<i32>();
    assert_eq!(l.len(), 2);
    let s = l.values().as_struct();
    assert_eq!(s.num_columns(), 2);
    assert_eq!(s.column(0).len(), 2);
}

#[test]
fn from_arrays_maps_slices_with_policy() {
    let entries = test_entries(DataType::Decimal128(10, 2), DataType::Utf8View, true);
    let key_values = Arc::new(
        Decimal128Array::from(vec![Some(100), Some(200), Some(300)])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    ) as ArrayRef;
    let keys = list_of(key_values, &[2, 1, 0], &[false, false, true]);
    let values = list_of(
        strings(&[Some("a"), Some("b"), Some("c")]),
        &[2, 1, 0],
        &[false, false, true],
    );
    let m = from_arrays(&keys, &values, 3, &entries, DedupPolicy::Exception).unwrap();
    assert_eq!(m.len(), 3);
    assert!(m.is_null(2));
    let (from, to) = map_range(&m, 0).unwrap();
    assert_eq!((to - from), 2);

    // Different slice lengths are `_LEGACY_ERROR_TEMP_2128`.
    let keys = list_of(ints(&[Some(1)]), &[1], &[false]);
    let values = list_of(strings(&[Some("a"), Some("b")]), &[2], &[false]);
    let e = from_arrays(
        &keys,
        &values,
        1,
        &int_field_entries(),
        DedupPolicy::Exception,
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::LegacyErrorTemp2128);
    assert_eq!(e.row(), Some(0));

    // A null key element is `NULL_MAP_KEY`, wherever it sits.
    let keys = list_of(ints(&[Some(1), None]), &[2], &[false]);
    let values = list_of(strings(&[Some("a"), Some("b")]), &[2], &[false]);
    let e = from_arrays(
        &keys,
        &values,
        1,
        &int_field_entries(),
        DedupPolicy::Exception,
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::NullMapKey);

    // Duplicate keys follow the policy.
    let keys = list_of(ints(&[Some(1), Some(1)]), &[2], &[false]);
    let values = list_of(strings(&[Some("a"), Some("b")]), &[2], &[false]);
    let e = from_arrays(
        &keys,
        &values,
        1,
        &int_field_entries(),
        DedupPolicy::Exception,
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DuplicatedMapKey);
    let m = from_arrays(
        &keys,
        &values,
        1,
        &int_field_entries(),
        DedupPolicy::LastWin,
    )
    .unwrap();
    assert_eq!(
        pairs(&m),
        [Some(vec![("1".to_owned(), Some("b".to_owned()))])]
    );
}

#[test]
fn null_struct_values_append_typed_nulls() {
    let struct_ty = DataType::Struct(vec![Field::new("a", DataType::Int32, true)].into());
    let struct_fields = match &struct_ty {
        DataType::Struct(fs) => fs.clone(),
        _ => unreachable!(),
    };
    // A null struct as a map value: the map holds a null.
    let null_struct = Arc::new(
        StructArray::try_new(
            struct_fields.clone(),
            vec![ints(&[Some(1)])],
            Some(arrow_buffer::NullBuffer::from(vec![false])),
        )
        .unwrap(),
    ) as ArrayRef;
    let entries = test_entries(DataType::Int32, struct_ty.clone(), true);
    let m = create_map(
        &[ints(&[Some(1)])],
        &[null_struct],
        1,
        &entries,
        DedupPolicy::Exception,
    )
    .unwrap();
    assert!(!m.is_null(0));
    assert!(m.values().is_null(0));

    // A null struct element of a list value: the list keeps a null.
    let inner = Arc::new(
        StructArray::try_new(
            struct_fields,
            vec![ints(&[Some(1), Some(2)])],
            Some(arrow_buffer::NullBuffer::from(vec![true, false])),
        )
        .unwrap(),
    ) as ArrayRef;
    let list_value = list_of(inner, &[2], &[false]);
    let list_ty = DataType::List(Arc::new(Field::new("element", struct_ty, true)));
    let entries = test_entries(DataType::Int32, list_ty, true);
    let m = create_map(
        &[ints(&[Some(1)])],
        &[list_value],
        1,
        &entries,
        DedupPolicy::Exception,
    )
    .unwrap();
    assert!(!m.is_null(0));
    let lists = m.values().as_list::<i32>();
    assert!(!lists.is_null(0));
    assert!(lists.values().as_struct().is_null(1));
}

#[test]
fn from_entries_maps_struct_slices() {
    let fields: arrow_schema::Fields = vec![
        Field::new("key", DataType::Int32, true),
        Field::new("value", DataType::Utf8View, true),
    ]
    .into();
    let structs = Arc::new(
        StructArray::try_new(
            fields.clone(),
            vec![ints(&[Some(1), Some(2)]), views(&[Some("a"), Some("b")])],
            None,
        )
        .unwrap(),
    ) as ArrayRef;
    let array = list_of(structs, &[2, 0], &[false, true]);
    let entries = test_entries(DataType::Int32, DataType::Utf8View, true);
    let m = from_entries(&array, 2, &entries, DedupPolicy::Exception).unwrap();
    assert_eq!(m.len(), 2);
    assert!(m.is_null(1));

    // A null element nulls only its own row.
    let values = Arc::new(
        StructArray::try_new(
            fields.clone(),
            vec![ints(&[Some(9)]), views(&[Some("z")])],
            Some(arrow_buffer::NullBuffer::from(vec![false])),
        )
        .unwrap(),
    ) as ArrayRef;
    let array = list_of(values, &[1, 0], &[false, false]);
    let m = from_entries(&array, 2, &entries, DedupPolicy::Exception).unwrap();
    assert!(m.is_null(0));
    assert!(!m.is_null(1));
    assert_eq!(pairs(&m), [None, Some(vec![])]);

    // A null key field is `NULL_MAP_KEY`.
    let structs = Arc::new(
        StructArray::try_new(fields, vec![ints(&[None]), views(&[Some("a")])], None).unwrap(),
    ) as ArrayRef;
    let array = list_of(structs, &[1], &[false]);
    let e = from_entries(&array, 1, &entries, DedupPolicy::Exception).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::NullMapKey);

    // Entries that are not two-field structs are refused.
    let bad = list_of(ints(&[Some(1)]), &[1], &[false]);
    assert!(from_entries(&bad, 1, &entries, DedupPolicy::Exception).is_err());
}

#[test]
fn from_pairs_rows_empty_and_null() {
    let entries = test_entries(DataType::Utf8View, DataType::Utf8View, true);
    let m = from_pairs(
        &strings(&[Some("a"), Some("b"), Some("c")]),
        &strings(&[Some("1"), None, Some("3")]),
        &[0, 0, 2],
        3,
        &entries,
        DedupPolicy::Exception,
    )
    .unwrap();
    assert_eq!(
        pairs(&m),
        [
            Some(vec![
                ("a".to_owned(), Some("1".to_owned())),
                ("b".to_owned(), None),
            ]),
            None,
            Some(vec![("c".to_owned(), Some("3".to_owned()))]),
        ]
    );
}

#[test]
fn policy_parses_and_shapes_hold() {
    assert_eq!(DedupPolicy::of(None), DedupPolicy::Exception);
    assert_eq!(DedupPolicy::of(Some("EXCEPTION")), DedupPolicy::Exception);
    assert_eq!(DedupPolicy::of(Some("LAST_WIN")), DedupPolicy::LastWin);
    assert_eq!(DedupPolicy::of(Some("other")), DedupPolicy::Exception);
    // Every key type the analyzer can produce is supported.
    for t in [
        DataType::Boolean,
        DataType::Int32,
        DataType::Float64,
        DataType::Decimal128(10, 2),
        DataType::Utf8View,
        DataType::BinaryView,
        DataType::Date32,
        DataType::Timestamp(TimeUnit::Microsecond, None),
        DataType::Interval(IntervalUnit::YearMonth),
        DataType::Duration(TimeUnit::Microsecond),
        DataType::List(Arc::new(Field::new("element", DataType::Int32, true))),
    ] {
        assert!(supports_key(&t), "{t}");
        assert!(supports_value(&t), "{t}");
    }
    let calendar = krishiv_types::arrow::calendar_struct();
    let calendars = DataType::List(Arc::new(Field::new("element", calendar.clone(), true)));
    assert!(!supports_key(&calendar));
    assert!(supports_map_key(&calendar));
    assert!(supports_value(&calendar));
    assert!(!supports_map_key(&calendars));
    assert!(supports_value(&calendars));
    assert!(supports_value(&DataType::Map(int_field_entries(), false)));
    assert!(!supports_key(&DataType::Map(int_field_entries(), false)));
    assert!(!supports_key(&DataType::Dictionary(
        Box::new(DataType::Int32),
        Box::new(DataType::Utf8View),
    )));
}

#[test]
fn decimal_text_renders() {
    assert_eq!(decimal_text(100, 2).expect("decimal text"), "1.00");
    assert_eq!(decimal_text(-5, 3).expect("decimal text"), "-0.005");
    assert_eq!(decimal_text(0, 0).expect("decimal text"), "0");
    assert_eq!(decimal_text(12345, 0).expect("decimal text"), "12345");
}

#[test]
fn large_list_keys_compare() {
    use arrow_array::builder::{Int32Builder, LargeListBuilder};
    let mut a = LargeListBuilder::new(Int32Builder::new());
    a.values().append_value(1);
    a.append(true);
    let mut b = LargeListBuilder::new(Int32Builder::new());
    b.values().append_value(1);
    b.append(true);
    let (x, y) = (a.finish(), b.finish());
    assert!(keys_equal(&x, 0, &y, 0).unwrap());
    let m = create_map(
        &[Arc::new(x) as ArrayRef],
        &[strings(&[Some("a")])],
        1,
        &test_entries(
            DataType::LargeList(Arc::new(Field::new("element", DataType::Int32, true))),
            DataType::Utf8View,
            true,
        ),
        DedupPolicy::Exception,
    )
    .unwrap();
    assert_eq!(m.len(), 1);
}

#[test]
fn equal_keys_cover_every_primitive_and_struct_carrier() {
    use arrow_array::{
        BinaryArray, BooleanArray, Date32Array, Float32Array, Int8Array, Int16Array, Int64Array,
        NullArray, StructArray, TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array,
        UInt64Array,
    };
    // Each pair: (a, b) where row 0 of `a` equals row 0 of `b` and row 1 differs.
    let pairs: Vec<(ArrayRef, ArrayRef)> = vec![
        (
            Arc::new(BooleanArray::from(vec![true, false])),
            Arc::new(BooleanArray::from(vec![true, true])),
        ),
        (
            Arc::new(Int8Array::from(vec![1, 2])),
            Arc::new(Int8Array::from(vec![1, 3])),
        ),
        (
            Arc::new(Int16Array::from(vec![1, 2])),
            Arc::new(Int16Array::from(vec![1, 3])),
        ),
        (
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(Int64Array::from(vec![1, 3])),
        ),
        (
            Arc::new(UInt8Array::from(vec![1, 2])),
            Arc::new(UInt8Array::from(vec![1, 3])),
        ),
        (
            Arc::new(UInt16Array::from(vec![1, 2])),
            Arc::new(UInt16Array::from(vec![1, 3])),
        ),
        (
            Arc::new(UInt32Array::from(vec![1, 2])),
            Arc::new(UInt32Array::from(vec![1, 3])),
        ),
        (
            Arc::new(UInt64Array::from(vec![1, 2])),
            Arc::new(UInt64Array::from(vec![1, 3])),
        ),
        (
            Arc::new(Float32Array::from(vec![1.5, 2.5])),
            Arc::new(Float32Array::from(vec![1.5, 3.5])),
        ),
        (
            Arc::new(Date32Array::from(vec![1, 2])),
            Arc::new(Date32Array::from(vec![1, 3])),
        ),
        (
            Arc::new(TimestampMicrosecondArray::from(vec![1, 2])),
            Arc::new(TimestampMicrosecondArray::from(vec![1, 3])),
        ),
        (
            Arc::new(BinaryArray::from(vec![&b"a"[..], &b"b"[..]])),
            Arc::new(BinaryArray::from(vec![&b"a"[..], &b"c"[..]])),
        ),
    ];
    for (a, b) in &pairs {
        assert!(
            keys_equal(a.as_ref(), 0, b.as_ref(), 0).unwrap(),
            "{:?}",
            a.data_type()
        );
        assert!(
            !keys_equal(a.as_ref(), 1, b.as_ref(), 1).unwrap(),
            "{:?}",
            a.data_type()
        );
    }
    let n = NullArray::new(1);
    assert!(keys_equal(&n, 0, &n, 0).unwrap());
    // Structs compare field by field, by position.
    let st = |x: Vec<i32>, y: Vec<&str>| -> ArrayRef {
        Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("x", DataType::Int32, true)),
                Arc::new(Int32Array::from(x)) as ArrayRef,
            ),
            (
                Arc::new(Field::new("y", DataType::Utf8, true)),
                Arc::new(StringArray::from(y)) as ArrayRef,
            ),
        ]))
    };
    let (s1, s2) = (
        st(vec![1, 2], vec!["a", "b"]),
        st(vec![1, 2], vec!["a", "c"]),
    );
    assert!(keys_equal(s1.as_ref(), 0, s2.as_ref(), 0).unwrap());
    assert!(!keys_equal(s1.as_ref(), 1, s2.as_ref(), 1).unwrap());
    // Structs of different arity are never equal; fields pair by position.
    let narrow: ArrayRef = Arc::new(StructArray::from(vec![(
        Arc::new(Field::new("x", DataType::Int32, true)),
        Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
    )]));
    assert!(!keys_equal(s1.as_ref(), 0, narrow.as_ref(), 0).unwrap());
    let renamed: ArrayRef = Arc::new(StructArray::from(vec![
        (
            Arc::new(Field::new("x", DataType::Int32, true)),
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("z", DataType::Utf8, true)),
            Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
        ),
    ]));
    assert!(keys_equal(s1.as_ref(), 0, renamed.as_ref(), 0).unwrap());
}

fn strs(values: Vec<&str>) -> ArrayRef {
    Arc::new(StringArray::from(values))
}

#[test]
fn dedup_raises_on_duplicates_and_keeps_last_under_last_win() {
    let keys = strs(vec!["b", "a", "b"]);
    let err = KrishivError::from(dedup_entries(keys.as_ref(), 3, false).unwrap_err());
    assert_eq!(err.error_class(), ErrorClass::DuplicatedMapKey);
    assert!(err.to_string().contains("b"), "{err}");
    let (kept, taken) = dedup_entries(keys.as_ref(), 3, true).unwrap();
    assert_eq!((kept, taken), (vec![0, 1], vec![2, 1]));
}

#[test]
fn dedup_rejects_null_keys_and_short_arrays() {
    let nulls: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None]));
    let err = KrishivError::from(dedup_entries(nulls.as_ref(), 2, true).unwrap_err());
    assert_eq!(err.error_class(), ErrorClass::NullMapKey);
    assert!(dedup_entries(nulls.as_ref(), 3, true).is_err());
    assert_eq!(
        dedup_entries(nulls.as_ref(), 0, true).unwrap(),
        (vec![], vec![])
    );
}

#[test]
fn union_merges_in_first_seen_order() {
    let (k1, k2) = (strs(vec!["a", "b", "b"]), strs(vec!["b", "c", "c"]));
    let m = union_keys(k1.as_ref(), 3, k2.as_ref(), 3).unwrap();
    // `a` only left, `b` both (first entries), `c` only right.
    assert_eq!(
        m,
        vec![
            (0, Some(0), None),
            (1, Some(1), Some(0)),
            (4, None, Some(1)),
        ]
    );
    let nulls: ArrayRef = Arc::new(StringArray::from(vec![None::<&str>]));
    let err = KrishivError::from(union_keys(k1.as_ref(), 1, nulls.as_ref(), 1).unwrap_err());
    assert_eq!(err.error_class(), ErrorClass::NullMapKey);
    let err = KrishivError::from(union_keys(nulls.as_ref(), 1, k1.as_ref(), 1).unwrap_err());
    assert_eq!(err.error_class(), ErrorClass::NullMapKey);
    assert!(union_keys(k1.as_ref(), 9, k2.as_ref(), 1).is_err());
}

#[test]
fn build_map_checks_its_lengths() {
    let keys = strs(vec!["a", "b"]);
    let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
    let offsets = OffsetBuffer::new(vec![0_i32, 2].into());
    let m = build_map(&keys, &values, offsets.clone(), None, false).unwrap();
    assert_eq!(m.len(), 1);
    let short: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    assert!(build_map(&keys, &short, offsets, None, false).is_err());
}

#[test]
fn view_keys_hash_like_plain_keys() {
    let views: ArrayRef = Arc::new(StringViewArray::from(vec!["a", "b", "a"]));
    let err = KrishivError::from(dedup_entries(views.as_ref(), 3, false).unwrap_err());
    assert_eq!(err.error_class(), ErrorClass::DuplicatedMapKey);
    let bins: ArrayRef = Arc::new(arrow_array::BinaryViewArray::from(vec![
        b"x".as_slice(),
        b"x".as_slice(),
    ]));
    let (kept, taken) = dedup_entries(bins.as_ref(), 2, true).unwrap();
    assert_eq!((kept, taken), (vec![0], vec![1]));
}

#[test]
fn nested_and_view_keys_group_by_value() {
    use arrow_array::builder::{ListBuilder, StringViewBuilder};
    let mut b = ListBuilder::new(StringViewBuilder::new());
    for row in [["a"], ["b"], ["a"]] {
        b.values().append_value(row[0]);
        b.append(true);
    }
    let keys: ArrayRef = Arc::new(b.finish());
    let err = KrishivError::from(dedup_entries(keys.as_ref(), 3, false).unwrap_err());
    assert_eq!(err.error_class(), ErrorClass::DuplicatedMapKey);
    let (kept, taken) = dedup_entries(keys.as_ref(), 3, true).unwrap();
    assert_eq!((kept, taken), (vec![0, 1], vec![2, 1]));
    // A duplicate before a NULL key raises first, as a scan meets it.
    let nulls: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), Some("a"), None]));
    let err = KrishivError::from(dedup_entries(nulls.as_ref(), 3, false).unwrap_err());
    assert_eq!(err.error_class(), ErrorClass::DuplicatedMapKey);
    let err = KrishivError::from(dedup_entries(nulls.as_ref(), 3, true).unwrap_err());
    assert_eq!(err.error_class(), ErrorClass::NullMapKey);
    let hashed = hashable(keys.as_ref()).unwrap();
    assert!(!has_view(hashed.data_type()));
}
