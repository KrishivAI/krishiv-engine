// SPDX-License-Identifier: Apache-2.0
//! Diff-coverage batch two for the JSON kernels: merges, skips, escapes,
//! JSONPath edges, nested drains and every render branch.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Float32Type, Int32Type};
use arrow_array::{Array, ArrayRef, ListArray, MapArray, StringArray, StructArray};
use arrow_schema::{DataType, Field};
use krishiv_kernels::cast::FixedOffset;
use krishiv_kernels::json::{
    JsonOptions, get_json_object, infer_json, json_array_length, json_object_keys, read_json,
};

fn strs(v: &[Option<&str>]) -> StringArray {
    StringArray::from(v.to_vec())
}

fn zone() -> FixedOffset {
    FixedOffset(0)
}

fn read(texts: &[Option<&str>], target: &DataType, opts: &JsonOptions) -> ArrayRef {
    read_json(&strs(texts), target, opts, &zone(), 0).unwrap()
}

/// A malformed record: a valid struct row whose every field is `NULL`.
fn unreadable(out: &ArrayRef, row: usize) -> bool {
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    st.is_valid(row) && st.columns().iter().all(|c| c.is_null(row))
}

fn struct_of(fields: Vec<Field>) -> DataType {
    DataType::Struct(fields.into())
}

fn list_of(element: Arc<Field>) -> DataType {
    DataType::List(element)
}

fn map_of(key: Field, value: Field) -> DataType {
    let entries = DataType::Struct(vec![key, value].into());
    DataType::Map(Arc::new(Field::new("entries", entries, false)), false)
}

#[test]
fn infer_merges_nulls_floats_objects_and_rejects_fragments() {
    let opts = JsonOptions::new();
    // `null` merges into whatever follows it, and vice versa.
    for text in [
        "[null,1]",
        "[1,null]",
        "[null,null]",
        "[null,\"x\"]",
        "[null,true]",
        "[null,[1]]",
    ] {
        assert!(infer_json(&strs(&[Some(text)]), &opts).is_ok(), "{text}");
    }
    // Float pairs compare decimal shapes both ways.
    for text in ["[1.5,2.5]", "[1.55,1.5]", "[1.5,1.55]"] {
        let got = infer_json(&strs(&[Some(text)]), &opts).unwrap();
        assert_eq!(
            got[0],
            Some(DataType::List(Arc::new(Field::new(
                "element",
                DataType::Float64,
                true
            )))),
            "{text}"
        );
    }
    // Objects merge field-wise inside arrays.
    let got = infer_json(&strs(&[Some("[{\"a\":1},{\"b\":2}]")]), &opts).unwrap();
    assert!(matches!(got[0], Some(DataType::List(_))));
    let got = infer_json(&strs(&[Some("[{\"a\":1},{\"a\":2.5}]")]), &opts).unwrap();
    assert!(matches!(got[0], Some(DataType::List(_))));
    // Fragments fail.
    for text in ["[1 2]", "{\"a\" 1}", "{\"a\":1 \"b\":2}", "{\"a\":1"] {
        assert!(infer_json(&strs(&[Some(text)]), &opts).is_err(), "{text}");
    }
}

#[test]
fn lexer_edges_for_skips_escapes_and_keys() {
    let opts = JsonOptions::new();
    // An empty bare word is malformed.
    let mut bare = JsonOptions::new();
    bare.allow_unquoted_field_names = true;
    let t = struct_of(vec![Field::new("a", DataType::Int32, true)]);
    let out = read(&[Some("{,}")], &t, &bare);
    assert!(unreadable(&out, 0));
    // A 1001-deep skipped value is too deep.
    let deep_val = format!("{}1{}", "[".repeat(1001), "]".repeat(1001));
    let out = read(
        &[Some(format!("{{\"a\":{deep_val}}}").as_str())],
        &struct_of(vec![Field::new("a", DataType::Utf8View, true)]),
        &opts,
    );
    assert!(unreadable(&out, 0));
    // A missing colon inside a skipped value fails the query.
    let out = json_object_keys(&strs(&[Some("{\"a\": {\"b\" 1}}")])).unwrap();
    assert!(out.as_any().downcast_ref::<ListArray>().unwrap().is_null(0));
    // A bad separator inside a skipped array fails the query.
    let out = json_array_length(&strs(&[Some("[1 2]")]));
    assert!(out.as_primitive::<Int32Type>().iter().collect::<Vec<_>>() == [None]);
    // `\/` unescapes to a slash.
    let t = struct_of(vec![Field::new("a", DataType::Utf8View, true)]);
    let out = read(&[Some("{\"a\":\"a\\/b\"}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    assert_eq!(st.column(0).as_string_view().value(0), "a/b");
    // A pair whose second half is not `\u` fails.
    let out = read(&[Some("{\"a\":\"\\uD834ab\"}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    assert!(st.column(0).is_null(0));
    // `+` and `/` decode, and every padding shape works.
    let target = DataType::BinaryView;
    for text in ["\"++++\"", "\"////\"", "\"ABCD\"", "\"ABC=\"", "\"AB==\""] {
        let out = read(&[Some(text)], &target, &opts);
        assert!(!out.is_null(0), "{text}");
    }
}

#[test]
fn path_edges_render_branchy_values() {
    let null = || None::<String>;
    let one = |s: &str| {
        get_json_object(&strs(&[Some(s)]), &strs(&[Some("$")]))
            .as_string::<i32>()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect::<Vec<_>>()
    };
    // Booleans render by value.
    let got = get_json_object(&strs(&[Some("{\"a\":false}")]), &strs(&[Some("$.a")]));
    assert_eq!(
        got.as_string::<i32>().iter().collect::<Vec<_>>(),
        [Some("false")]
    );
    // A two-field root renders its comma.
    assert_eq!(
        one("{\"a\":1,\"b\":2}"),
        [Some("{\"a\":1,\"b\":2}".to_owned())]
    );
    // Empty input with steps, empty selections and bad shapes are NULL.
    for (j, p) in [
        ("", "$.a"),
        ("   ", "$.a"),
        ("{}", "$.a"),
        ("{\"a\" 1}", "$.a"),
        ("{\"a\":1 \"b\":2}", "$.a"),
        ("{\"a\":1", "$.a"),
        ("[]", "$[0]"),
        ("[]", "$[*]"),
        ("[1 2]", "$[0]"),
        ("[1", "$[0]"),
        ("[1 2]", "$"),
        ("[1", "$"),
        ("{\"a\" 1}", "$"),
        ("{\"a\":1", "$"),
        ("]", "$"),
    ] {
        let got = get_json_object(&strs(&[Some(j)]), &strs(&[Some(p)]));
        assert_eq!(
            got.as_string::<i32>()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect::<Vec<_>>(),
            [null()],
            "{j} {p}"
        );
    }
    // Empty containers capture and render.
    assert_eq!(one("[]"), [Some("[]".to_owned())]);
    assert_eq!(one("{}"), [Some("{}".to_owned())]);
    // Nesting past 1000 captures nothing.
    let deep = format!("{}1{}", "[".repeat(1001), "]".repeat(1001));
    assert_eq!(one(&deep), [null()]);
}

#[test]
fn read_struct_and_nested_edges() {
    let opts = JsonOptions::new();
    let fail = {
        let mut o = JsonOptions::new();
        o.mode = krishiv_kernels::json::ParseMode::FailFast;
        o
    };
    let ints = struct_of(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
    ]);
    // A truncated string poisons a non-struct row.
    let list_int = list_of(Arc::new(Field::new("element", DataType::Int32, true)));
    let out = read(&[Some("[\"ab]")], &list_int, &opts);
    assert!(out.as_any().downcast_ref::<ListArray>().unwrap().is_null(0));
    // Bare garbage fails the peek itself.
    let out = read(&[Some("abc")], &list_int, &opts);
    assert!(out.as_any().downcast_ref::<ListArray>().unwrap().is_null(0));
    // A non-object poisons the row in FAILFAST.
    assert!(read_json(&strs(&[Some("1")]), &ints, &fail, &zone(), 0).is_err());
    // An empty object reads null fields but a valid row.
    let out = read(&[Some("{}")], &ints, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    assert!(st.column(0).is_null(0));
    assert!(st.column(1).is_null(0));
    // A missing colon or trailing garbage fails the row.
    for bad in ["{\"a\" 1}", "{\"a\":1 \"b\":2}"] {
        let out = read(&[Some(bad)], &ints, &opts);
        assert!(unreadable(&out, 0), "{bad}");
    }
    // A JSON null into a nested list is a null list, not a poisoned row.
    let nested = struct_of(vec![Field::new(
        "a",
        list_of(Arc::new(Field::new("element", DataType::Int32, true))),
        true,
    )]);
    let out = read(&[Some("{\"a\":null}")], &nested, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    assert!(st.column(0).as_list::<i32>().is_null(0));
    let list_list = list_of(Arc::new(Field::new(
        "element",
        list_of(Arc::new(Field::new("element", DataType::Int32, true))),
        true,
    )));
    let out = read(&[Some("[[null]]")], &list_list, &opts);
    let list = out.as_any().downcast_ref::<ListArray>().unwrap();
    assert!(!list.is_null(0));
    // A missing separator fails the row.
    let out = read(&[Some("[1 2]")], &list_int, &opts);
    assert!(out.as_any().downcast_ref::<ListArray>().unwrap().is_null(0));
    // Nested drains track depth.
    let map_int = map_of(
        Field::new("key", DataType::Utf8View, false),
        Field::new("value", DataType::Int32, true),
    );
    let out = read(
        &[Some("{\"a\": \"x\", \"b\": {\"c\": 1}}")],
        &map_int,
        &opts,
    );
    assert!(out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    let out = read(&[Some("[1, \"x\", [2]]")], &list_int, &opts);
    assert!(out.as_any().downcast_ref::<ListArray>().unwrap().is_null(0));
}

#[test]
fn read_string_field_renders_every_token() {
    let opts = JsonOptions::new();
    let t = struct_of(vec![Field::new("a", DataType::Utf8View, true)]);
    let value = |text: &str| {
        let out = read(&[Some(text)], &t, &opts);
        let st = out.as_any().downcast_ref::<StructArray>().unwrap();
        assert!(!st.is_null(0), "{text}");
        st.column(0)
            .as_string_view()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        value("{\"a\":[null,true,false]}"),
        [Some("[null,true,false]".to_owned())]
    );
    assert_eq!(value("{\"a\":[\"x\"]}"), [Some("[\"x\"]".to_owned())]);
    assert_eq!(value("{\"a\":[]}"), [Some("[]".to_owned())]);
    assert_eq!(value("{\"a\":{}}"), [Some("{}".to_owned())]);
    assert_eq!(
        value("{\"a\":{\"x\":1,\"y\":2}}"),
        [Some("{\"x\":1,\"y\":2}".to_owned())]
    );
    // Lexically broken nesting fails the row (span splitting fails first).
    for bad in [
        "{\"a\":[1 2]}",
        "{\"a\":{\"x\":1 \"y\":2}}",
        "{\"a\":[}",
        "{\"a\":{\"x\":1}",
    ] {
        let out = read(&[Some(bad)], &t, &opts);
        assert!(unreadable(&out, 0), "{bad}");
    }
    // Bare keys work with the flag.
    let mut bare = JsonOptions::new();
    bare.allow_unquoted_field_names = true;
    let out = read(&[Some("{\"a\":{b:1}}")], &t, &bare);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(
        st.column(0)
            .as_string_view()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("{\"b\":1}".to_owned())]
    );
    // Nested arrays and objects render recursively.
    assert_eq!(value("{\"a\":[[]]}"), [Some("[[]]".to_owned())]);
    assert_eq!(value("{\"a\":[[1,2]]}"), [Some("[[1,2]]".to_owned())]);
    assert_eq!(value("{\"a\":[{}]}"), [Some("[{}]".to_owned())]);
    assert_eq!(
        value("{\"a\":[{\"x\":1,\"y\":2}]}"),
        [Some("[{\"x\":1,\"y\":2}]".to_owned())]
    );
    let out = read(&[Some("{\"a\":[{b:1}]}")], &t, &bare);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(
        st.column(0)
            .as_string_view()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("[{\"b\":1}]".to_owned())]
    );
    let out = read(&[Some("{\"a\":{b 1}}")], &t, &bare);
    assert!(unreadable(&out, 0));
}

#[test]
fn read_scalars_cover_every_arm() {
    let opts = JsonOptions::new();
    // A boolean into an integral is a field NULL.
    let t = struct_of(vec![Field::new("a", DataType::Int32, true)]);
    let out = read(&[Some("{\"a\":true}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    assert!(st.column(0).is_null(0));
    // Float text into a float field narrows like Java.
    let t = struct_of(vec![Field::new("f", DataType::Float32, true)]);
    let out = read(&[Some("{\"f\":1.5}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(st.column(0).as_primitive::<Float32Type>().value(0), 1.5);
    // Integral text into a decimal field scales.
    let t = struct_of(vec![Field::new("d", DataType::Decimal128(10, 2), true)]);
    let out = read(&[Some("{\"d\":5}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(st.column(0).as_primitive::<Decimal128Type>().value(0), 500);
    // An overflowing integral into decimal is a field NULL.
    let out = read(&[Some("{\"d\":99999999999999999999999}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(st.column(0).is_null(0));
    // Exponents with an explicit sign read too.
    let out = read(&[Some("{\"d\":1e+2}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(
        st.column(0).as_primitive::<Decimal128Type>().value(0),
        10_000
    );
    let out = read(&[Some("{\"d\":1e-2}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(st.column(0).as_primitive::<Decimal128Type>().value(0), 1);
    // A timestamp_ntz reads with the default grammar.
    let t = struct_of(vec![Field::new(
        "n",
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
        true,
    )]);
    let out = read(&[Some("{\"n\":\"2012-12-15 15:15:15\"}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.column(0).is_null(0));
    let out = read(&[Some("{\"n\":\"xx\"}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(st.column(0).is_null(0));
    // `len_total` runs over every scalar builder behind a list; a JSON
    // null reads as one null element.
    for target in [
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int64,
        DataType::Float32,
        DataType::Float64,
        DataType::Decimal128(10, 2),
        DataType::BinaryView,
        DataType::Date32,
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
    ] {
        let list = list_of(Arc::new(Field::new("element", target.clone(), true)));
        let out = read(&[Some("[null]")], &list, &opts);
        let list = out.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(!list.is_null(0), "{target:?}");
        assert_eq!(list.value_length(0), 1, "{target:?}");
        assert_eq!(list.values().null_count(), 1, "{target:?}");
    }
    let void_list = list_of(Arc::new(Field::new("element", DataType::Null, true)));
    let out = read(&[Some("[null]")], &void_list, &opts);
    assert!(!out.as_any().downcast_ref::<ListArray>().unwrap().is_null(0));
}

#[test]
fn read_decimal_map_keys_cover_sign_and_exponent() {
    let opts = JsonOptions::new();
    let target = map_of(
        Field::new("key", DataType::Decimal128(10, 2), false),
        Field::new("value", DataType::Int32, true),
    );
    // A bare minus is not a decimal.
    let out = read(&[Some("{\"-\":1}")], &target, &opts);
    assert!(out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    // An empty exponent is not a decimal.
    let out = read(&[Some("{\"1e\":1}")], &target, &opts);
    assert!(out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    // A non-digit exponent is not a decimal.
    let out = read(&[Some("{\"1ex\":1}")], &target, &opts);
    assert!(out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
}
