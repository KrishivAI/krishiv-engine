// SPDX-License-Identifier: Apache-2.0
//! JSON kernels against Spark 4.2 behaviour (goldens and the function
//! matrix in `tests/fixtures/spark-4.2.0/function-matrix/cases/json.jsonl`):
//! `NULL`s, empty input, nesting, large offsets, malformed text and the
//! element/field failure rules of `from_json`.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Float64Type, Int32Type, TimestampMicrosecondType};
use arrow_array::{Array, ArrayRef, Int32Array, ListArray, MapArray, StringArray, StructArray};
use arrow_schema::{DataType, Field};
use krishiv_kernels::cast::FixedOffset;
use krishiv_kernels::json::{
    JsonOptions, get_json_object, infer_json, json_array_length, json_object_keys, read_json,
    write_json,
};

fn strs(v: &[Option<&str>]) -> StringArray {
    StringArray::from(v.to_vec())
}

fn zone() -> FixedOffset {
    FixedOffset(0)
}

fn rows(a: &StringArray) -> Vec<Option<String>> {
    a.iter().map(|s| s.map(str::to_owned)).collect()
}

fn ints(a: &Int32Array) -> Vec<Option<i32>> {
    a.iter().collect()
}

fn utf8_field(name: &str, nullable: bool) -> Arc<Field> {
    Arc::new(Field::new(name, DataType::Utf8View, nullable))
}

fn list_of(element: Arc<Field>) -> DataType {
    DataType::List(element)
}

fn struct_of(fields: Vec<Field>) -> DataType {
    DataType::Struct(fields.into())
}

fn map_of(key: Field, value: Field) -> DataType {
    let entries = DataType::Struct(vec![key, value].into());
    DataType::Map(Arc::new(Field::new("entries", entries, false)), false)
}

/// `read_json` of `texts` into `target`.
fn read(texts: &[Option<&str>], target: &DataType, opts: &JsonOptions) -> ArrayRef {
    read_json(&strs(texts), target, opts, &zone(), 0).unwrap()
}

/// `write_json` of `col`.
fn write(col: &ArrayRef, opts: &JsonOptions) -> Vec<Option<String>> {
    let out = write_json(col, opts, &zone()).unwrap();
    rows(out.as_string::<i32>())
}

#[test]
fn array_length_counts_top_level_elements() {
    let out = json_array_length(&strs(&[
        None,
        Some("[1,2,3,4]"),
        Some("[1,2,3,{\"f1\":1,\"f2\":[5,6]},4]"),
        Some("[]"),
        Some("[[1,2],[5,6,7]]"),
        Some("[{\"a\":123},{\"b\":\"hello\"}]"),
        Some("[1,2,3,[33,44],{\"key\":[2,3,4]}]"),
        Some("{\"key\":\"not a json array\"}"),
        Some(""),
        Some("[1,2,3,4,5"),
        Some("[1,2"),
        Some("   [7]   "),
    ]));
    assert_eq!(
        ints(out.as_primitive::<Int32Type>()),
        [
            None,
            Some(4),
            Some(5),
            Some(0),
            Some(2),
            Some(2),
            Some(5),
            None,
            None,
            None,
            None,
            Some(1),
        ]
    );
}

#[test]
fn object_keys_lists_top_level_keys_in_order() {
    let out = json_object_keys(&strs(&[
        None,
        Some("{}"),
        Some("{\"key\": 1}"),
        Some("{\"key\": \"value\", \"key2\": 2}"),
        Some("{\"arrayKey\": [1, 2, 3]}"),
        Some("{\"f1\":\"abc\",\"f2\":{\"f3\":\"a\", \"f4\":\"b\"}}"),
        Some(""),
        Some("{[1,2]}"),
        Some("{\"key\": 45, \"random_string\"}"),
        Some("[1, 2, 3]"),
        Some("{\"a\":1,\"a\":2}"),
    ]))
    .unwrap();
    let list = out.as_list::<i32>();
    let values = list.values().as_string_view();
    let offsets = list.value_offsets();
    let keys: Vec<Option<Vec<String>>> = (0..list.len())
        .map(|i| {
            if list.is_null(i) {
                return None;
            }
            let (start, end) = (
                usize::try_from(offsets[i]).unwrap_or(0),
                usize::try_from(offsets[i.saturating_add(1)]).unwrap_or(0),
            );
            Some(
                (start..end)
                    .map(|j| values.value(j).to_owned())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    assert_eq!(
        keys,
        [
            None,
            Some(vec![]),
            Some(vec!["key".to_owned()]),
            Some(vec!["key".to_owned(), "key2".to_owned()]),
            Some(vec!["arrayKey".to_owned()]),
            Some(vec!["f1".to_owned(), "f2".to_owned()]),
            None,
            None,
            None,
            None,
            Some(vec!["a".to_owned(), "a".to_owned()]),
        ]
    );
}

#[test]
fn get_json_object_selects_fields_indexes_and_wildcards() {
    let json = strs(&[
        None,
        Some("{\"a\":\"b\"}"),
        Some("[{\"a\":\"b\"},{\"a\":\"c\"}]"),
        Some("[{\"a\":\"b\"},{\"a\":\"c\"}]"),
        Some("{\"a\":{\"b\":[1,2]}}"),
        Some("{\"a\":1}"),
        Some("not json"),
        Some("{\"a\":\"b\"}"),
        Some("{\"a\":null}"),
    ]);
    let path = strs(&[
        Some("$.a"),
        Some("$.a"),
        Some("$[0].a"),
        Some("$[*].a"),
        Some("$.a.b[1]"),
        Some("$.missing"),
        Some("$.a"),
        Some("a"),
        Some("$.a"),
    ]);
    let got = get_json_object(&json, &path);
    assert_eq!(
        rows(got.as_string::<i32>()),
        [
            None,
            Some("b".to_owned()),
            Some("b".to_owned()),
            Some("[\"b\",\"c\"]".to_owned()),
            Some("2".to_owned()),
            None,
            None,
            None,
            None,
        ]
    );
}

#[test]
fn read_struct_fields_missing_and_malformed() {
    let target = struct_of(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8View, true),
    ]);
    let opts = JsonOptions::new();
    let out = read(
        &[
            None,
            Some("{\"a\":1, \"b\":\"x\"}"),
            Some("{\"a\":1}"),
            Some("{\"b\":\"x\", \"c\":true}"),
            Some("{\"a\":null, \"b\":null}"),
            // Not objects: structs of NULLs.
            Some("abc"),
            Some("1"),
            Some("[1,2]"),
            Some(""),
            Some("null"),
        ],
        &target,
        &opts,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    let a: Vec<Option<i32>> = st.column(0).as_primitive::<Int32Type>().iter().collect();
    let b: Vec<Option<String>> = st
        .column(1)
        .as_string_view()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect();
    assert_eq!(
        a,
        [
            None,
            Some(1),
            Some(1),
            None,
            None,
            None,
            None,
            None,
            None,
            None
        ]
    );
    assert_eq!(
        b,
        [
            None,
            Some("x".to_owned()),
            None,
            Some("x".to_owned()),
            None,
            None,
            None,
            None,
            None,
            None
        ]
    );
    assert!(st.is_null(0));
    assert!(!st.is_null(5));
}

#[test]
fn read_arrays_fail_whole_on_bad_elements() {
    let target = list_of(Arc::new(Field::new("element", DataType::Int32, true)));
    let out = read(
        &[
            Some("[1, 2, 3]"),
            Some("[1, \"2\", 3]"),
            Some("[1, 2, null]"),
            None,
        ],
        &target,
        &JsonOptions::new(),
    );
    let list = out.as_any().downcast_ref::<ListArray>().unwrap();
    assert!(!list.is_null(0));
    assert_eq!(list.value_length(0), 3);
    assert!(list.is_null(1));
    assert!(!list.is_null(2));
    assert_eq!(list.value_length(2), 3);
    assert!(list.is_null(3));
}

#[test]
fn read_single_object_wraps_to_one_element() {
    let target = list_of(Arc::new(Field::new(
        "element",
        struct_of(vec![Field::new("a", DataType::Int32, true)]),
        true,
    )));
    let out = read(
        &[Some("{\"a\": 1}"), Some("[null, {\"a\":2}]"), Some("1")],
        &target,
        &JsonOptions::new(),
    );
    let list = out.as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(list.value_length(0), 1);
    assert_eq!(list.value_length(1), 2);
    assert!(list.is_null(2));
}

#[test]
fn read_maps_entries_and_poisoning() {
    let target = map_of(
        Field::new("key", DataType::Utf8View, false),
        Field::new("value", DataType::Int32, true),
    );
    let out = read(
        &[
            Some("{\"a\":1, \"b\":2}"),
            Some("{}"),
            Some("{\"a\":1, \"b\":\"x\"}"),
            Some("[1,2]"),
        ],
        &target,
        &JsonOptions::new(),
    );
    let map = out.as_any().downcast_ref::<MapArray>().unwrap();
    assert!(!map.is_null(0));
    assert_eq!(map.value_length(0), 2);
    assert!(!map.is_null(1));
    assert_eq!(map.value_length(1), 0);
    assert!(map.is_null(2));
    assert!(map.is_null(3));
}

#[test]
fn read_scalars_strictly() {
    let target = struct_of(vec![
        Field::new("i", DataType::Int32, true),
        Field::new("d", DataType::Float64, true),
        Field::new("s", DataType::Utf8View, true),
        Field::new("b", DataType::Boolean, true),
    ]);
    let out = read(
        &[
            Some("{\"i\":1, \"d\":0.8, \"s\":\"x\", \"b\":true}"),
            Some("{\"i\":\"2\", \"d\":1, \"s\":1, \"b\":\"true\"}"),
            Some("{\"i\":1.5, \"d\":\"x\", \"s\":null, \"b\":1}"),
            Some("{\"i\":2147483648, \"d\":1e400, \"s\":\"\", \"b\":false}"),
        ],
        &target,
        &JsonOptions::new(),
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    let i: Vec<Option<i32>> = st.column(0).as_primitive::<Int32Type>().iter().collect();
    let d: Vec<Option<f64>> = st.column(1).as_primitive::<Float64Type>().iter().collect();
    assert_eq!(i, [Some(1), None, None, None]);
    assert_eq!(d[0], Some(0.8));
    assert_eq!(d[1], Some(1.0));
    assert_eq!(d[2], None);
    assert_eq!(d[3], Some(f64::INFINITY));
}

#[test]
fn read_dates_timestamps_default_and_custom() {
    let target = struct_of(vec![
        Field::new("d", DataType::Date32, true),
        Field::new(
            "t",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
    ]);
    let opts = JsonOptions::new();
    let out = read(
        &[
            Some("{\"d\": \"2012-12-15\", \"t\": \"2012-12-15 15:15:15\"}"),
            Some("{\"d\": \"02-29\", \"t\": \"02-29\"}"),
        ],
        &target,
        &opts,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    let d: Vec<Option<i32>> = st.column(0).as_primitive::<Date32Type>().iter().collect();
    assert_eq!(d[0], Some(15689));
    assert_eq!(d[1], None);
    let t: Vec<Option<i64>> = st
        .column(1)
        .as_primitive::<TimestampMicrosecondType>()
        .iter()
        .collect();
    assert_eq!(t[0], Some(1_355_584_515_000_000));
    assert_eq!(t[1], None);

    let mut opts = JsonOptions::new();
    opts.date_format = Some(
        krishiv_kernels::datetime_pattern::DateTimePattern::new(
            "MM/dd yyyy",
            krishiv_kernels::datetime_pattern::Purpose::Parse,
        )
        .unwrap(),
    );
    opts.timestamp_format = Some(
        krishiv_kernels::datetime_pattern::DateTimePattern::new(
            "MM/dd yyyy HH:mm:ss",
            krishiv_kernels::datetime_pattern::Purpose::Parse,
        )
        .unwrap(),
    );
    let out = read(
        &[Some(
            "{\"d\": \"12/15 2012\", \"t\": \"12/15 2012 15:15:15\"}",
        )],
        &target,
        &opts,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    let d: Vec<Option<i32>> = st.column(0).as_primitive::<Date32Type>().iter().collect();
    assert_eq!(d[0], Some(15689));
}

#[test]
fn read_large_and_nested_offsets() {
    // 10 000 elements with nesting: offsets stay 32-bit and aligned.
    let inner = struct_of(vec![Field::new("a", DataType::Int32, true)]);
    let target = list_of(Arc::new(Field::new(
        "element",
        list_of(Arc::new(Field::new("element", inner, true))),
        true,
    )));
    let row = format!(
        "[{}]",
        (0..10_000)
            .map(|i| format!("[{{\"a\":{i}}}]"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let out = read(&[Some(&row), None], &target, &JsonOptions::new());
    let list = out.as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(list.value_length(0), 10_000);
    assert!(list.is_null(1));
}

#[test]
fn write_structs_maps_arrays_and_scalars() {
    let opts = JsonOptions::new();
    // Struct, map and array literals through the reader, then back out.
    let struct_target = struct_of(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
    ]);
    let col = read(&[Some("{\"a\":1, \"b\":2}"), None], &struct_target, &opts);
    assert_eq!(
        write(&col, &opts),
        [Some("{\"a\":1,\"b\":2}".to_owned()), None]
    );
    let map_target = map_of(
        Field::new("key", DataType::Utf8View, false),
        Field::new("value", DataType::Int32, true),
    );
    let col = read(&[Some("{\"a\":1}"), None], &map_target, &opts);
    assert_eq!(write(&col, &opts), [Some("{\"a\":1}".to_owned()), None]);
    let list_target = list_of(utf8_field("element", true));
    let col = read(&[Some("[\"1\",\"2\",\"3\"]"), None], &list_target, &opts);
    assert_eq!(
        write(&col, &opts),
        [Some("[\"1\",\"2\",\"3\"]".to_owned()), None]
    );
}

#[test]
fn write_escapes_sorts_and_formats() {
    let opts = JsonOptions::new();
    let col = Arc::new(StringArray::from(vec![Some("a\"b\\c\né"), None])) as ArrayRef;
    assert_eq!(
        write(&col, &opts),
        [Some("\"a\\\"b\\\\c\\né\"".to_owned()), None]
    );
    // Struct keys sort with `sortKeys`.
    let mut sort = JsonOptions::new();
    sort.sort_keys = true;
    let target = struct_of(vec![
        Field::new("b", DataType::Int32, true),
        Field::new("a", DataType::Int32, true),
    ]);
    let col = read(&[Some("{\"b\":1, \"a\":2}")], &target, &JsonOptions::new());
    assert_eq!(write(&col, &sort), [Some("{\"a\":2,\"b\":1}".to_owned())]);
    // A string that is not a struct poisons a struct-keyed map.
    let key_struct = struct_of(vec![Field::new("a", DataType::Int32, true)]);
    let entries = DataType::Struct(
        vec![
            Field::new("key", key_struct, false),
            Field::new("value", DataType::Int32, true),
        ]
        .into(),
    );
    let map_target = DataType::Map(Arc::new(Field::new("entries", entries, false)), false);
    let col = read(&[Some("{\"[1]\":2}")], &map_target, &JsonOptions::new());
    assert_eq!(write(&col, &opts), [None]);
}

#[test]
fn write_dates_timestamps_decimals_and_doubles() {
    use arrow_array::{Date32Array, Decimal128Array, Float64Array, TimestampMicrosecondArray};
    let opts = JsonOptions::new();
    let col = Arc::new(Date32Array::from(vec![Some(15689), None])) as ArrayRef;
    assert_eq!(
        write(&col, &opts),
        [Some("\"2012-12-15\"".to_owned()), None]
    );
    let col = Arc::new(
        TimestampMicrosecondArray::from(vec![Some(1_355_584_515_000_000), None])
            .with_timezone("UTC"),
    ) as ArrayRef;
    assert_eq!(
        write(&col, &opts),
        [Some("\"2012-12-15T15:15:15.000Z\"".to_owned()), None]
    );
    let col = Arc::new(
        Decimal128Array::from(vec![Some(15), None])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    ) as ArrayRef;
    assert_eq!(write(&col, &opts), [Some("0.15".to_owned()), None]);
    let col = Arc::new(Float64Array::from(vec![Some(0.8), Some(1.0), None])) as ArrayRef;
    assert_eq!(
        write(&col, &opts),
        [Some("0.8".to_owned()), Some("1.0".to_owned()), None]
    );
}

#[test]
fn infer_types_primitives_nesting_and_options() {
    let opts = JsonOptions::new();
    let json = strs(&[
        Some("{\"c1\":0, \"c2\":[1]}"),
        Some("{\"c1\":1}"),
        Some("[{\"col\":0}]"),
        Some("true"),
        Some("null"),
        Some("1.5"),
        Some("\"x\""),
        Some(""),
        Some("abc"),
        None,
    ]);
    let got = infer_json(&json, &opts).unwrap();
    let show = |t: &Option<DataType>| match t {
        None => "NULL".to_owned(),
        Some(t) => format!("{t:?}"),
    };
    let kinds: Vec<String> = got.iter().map(show).collect();
    assert!(kinds[0].contains("Struct"), "{kinds:?}");
    assert_eq!(kinds[3], "Boolean");
    assert_eq!(kinds[4], "Null");
    assert_eq!(kinds[5], "Float64");
    assert!(kinds[6].contains("Utf8"));
    assert_eq!(kinds[7], "Null");
    assert_eq!(kinds[8], "NULL");
    assert_eq!(kinds[9], "NULL");
    // `prefersDecimal` and `primitivesAsString`.
    let mut decimal = JsonOptions::new();
    decimal.prefers_decimal = true;
    decimal.allow_numeric_leading_zeros = true;
    let got = infer_json(&strs(&[Some("{\"c1\":01, \"c2\":0.1}")]), &decimal).unwrap();
    assert!(matches!(got[0], Some(DataType::Struct(_))), "{got:?}");
    let mut strings = JsonOptions::new();
    strings.primitives_as_string = true;
    let got = infer_json(&strs(&[Some("{\"c1\":1}")]), &strings).unwrap();
    assert!(matches!(got[0], Some(DataType::Struct(_))), "{got:?}");
}

#[test]
fn options_parse_booleans_and_reject_bad_patterns() {
    use std::collections::BTreeMap;
    let opts = JsonOptions::from_map(&BTreeMap::from([
        ("timestampFormat".to_owned(), "dd/MM/yyyy".to_owned()),
        ("primitivesAsString".to_owned(), "True".to_owned()),
        ("bogus".to_owned(), "x".to_owned()),
    ]))
    .unwrap();
    assert!(opts.timestamp_format.is_some());
    assert!(opts.primitives_as_string);
    assert!(
        JsonOptions::from_map(&BTreeMap::from([(
            "dateFormat".to_owned(),
            "QQQ".to_owned(),
        )]))
        .is_err()
    );
}

#[test]
fn base64_round_trips_binary() {
    let target = DataType::BinaryView;
    let out = read(
        &[Some("\"aGVsbG8=\""), Some("\"!!!\""), None],
        &target,
        &JsonOptions::new(),
    );
    assert_eq!(
        write(&out, &JsonOptions::new()),
        [Some("\"aGVsbG8=\"".to_owned()), None, None]
    );
}
