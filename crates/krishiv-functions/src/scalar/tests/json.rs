// SPDX-License-Identifier: Apache-2.0
//! The JSON bindings: values, options maps, return types and bind errors.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, MapArray, StringArray, StructArray};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowDataType, Field};
use krishiv_common::error::ErrorClass;

use super::{
    Arg, BindArg, BindOptions, bind, bind_scalar, class, col, ints, lit, owned, run, strs, text,
};

fn opts(pairs: &[(&str, &str)]) -> ArrayRef {
    let keys: ArrayRef = Arc::new(StringArray::from(
        pairs.iter().map(|(k, _)| Some(*k)).collect::<Vec<_>>(),
    ));
    let values: ArrayRef = Arc::new(StringArray::from(
        pairs.iter().map(|(_, v)| Some(*v)).collect::<Vec<_>>(),
    ));
    let fields: Vec<Field> = vec![
        Field::new("key", ArrowDataType::Utf8, false),
        Field::new("value", ArrowDataType::Utf8, true),
    ];
    let entries = StructArray::new(fields.into(), vec![keys, values], None);
    // One map row spanning every pair, as a `map(k1, v1, ...)` literal
    // reads (one entry per row would be several one-entry maps).
    let len = i32::try_from(pairs.len()).unwrap_or(0);
    Arc::new(
        MapArray::try_new(
            Arc::new(Field::new("entries", entries.data_type().clone(), false)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0, len])),
            entries,
            None,
            false,
        )
        .unwrap(),
    )
}

fn schema_arg(s: &str) -> BindArg {
    BindArg::Literal(strs(&[Some(s)]))
}

fn options(pairs: &[(&str, &str)]) -> BindArg {
    BindArg::Literal(opts(pairs))
}

fn run_with_schema(name: &str, json: &[Option<&str>], schema: &str) -> ArrayRef {
    let json = strs(json);
    let schema = strs(&[Some(schema)]);
    let f = bind_scalar(
        name,
        &[json.data_type().clone().into(), schema_arg_of(&schema)],
        &BindOptions::new(),
    )
    .unwrap();
    let rows = json.len();
    f.invoke(&[col(&json), lit(&schema)], rows).unwrap()
}

fn schema_arg_of(a: &ArrayRef) -> BindArg {
    BindArg::Literal(Arc::clone(a))
}

#[test]
fn array_length_counts_and_nulls() {
    let json = strs(&[
        Some("[1,2,3,4]"),
        Some("[]"),
        Some("[[1,2],[5,6,7]]"),
        Some("{\"k\":1}"),
        Some("[1,2"),
        Some(""),
        None,
    ]);
    let out = run("json_array_length", &[&json]);
    assert_eq!(
        out.as_primitive::<arrow_array::types::Int32Type>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(4), Some(0), Some(2), None, None, None, None]
    );
    let f = bind("json_array_length", &[&json]);
    assert_eq!(f.return_type(), &ArrowDataType::Int32);
    assert_eq!(f.name(), "json_array_length");
    // Only strings bind.
    let one = ints(&[Some(1)]);
    assert!(
        bind_scalar(
            "json_array_length",
            &[BindArg::Column(one.data_type().clone())],
            &BindOptions::new(),
        )
        .is_err()
    );
    assert!(bind_scalar("json_array_length", &[], &BindOptions::new()).is_err());
}

#[test]
fn object_keys_lists_and_nulls() {
    let json = strs(&[
        Some("{\"key\": 1}"),
        Some("{\"key\": \"value\", \"key2\": 2}"),
        Some("{}"),
        Some("[1,2]"),
        Some("{bad}"),
        None,
    ]);
    let f = bind("json_object_keys", &[&json]);
    let out = f.invoke(&[col(&json)], json.len()).unwrap();
    let list = out.as_list::<i32>();
    assert_eq!(list.len(), 6);
    assert!(!list.is_null(0));
    assert!(!list.is_null(2));
    assert!(list.is_null(3));
    assert!(list.is_null(5));
    // The keys come back as plain strings.
    let values = list.values().as_string_view();
    assert_eq!(values.value(0), "key");
    // A non-string does not bind.
    let one = ints(&[Some(1)]);
    assert!(
        bind_scalar(
            "json_object_keys",
            &[BindArg::Column(one.data_type().clone())],
            &BindOptions::new(),
        )
        .is_err()
    );
}

#[test]
fn get_object_selects_and_nulls() {
    let json = strs(&[
        Some("{\"a\":\"b\"}"),
        Some("[{\"a\":\"b\"},{\"a\":\"c\"}]"),
        Some("{\"a\":1}"),
        None,
    ]);
    let path = strs(&[Some("$.a"), Some("$[1].a"), Some("$.missing"), Some("$.a")]);
    let f = bind_scalar(
        "get_json_object",
        &[
            json.data_type().clone().into(),
            path.data_type().clone().into(),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    assert_eq!(f.return_type(), &ArrowDataType::Utf8);
    let out = f.invoke(&[col(&json), col(&path)], json.len()).unwrap();
    assert_eq!(text(&out), owned(&[Some("b"), Some("c"), None, None]));
    // A non-string path does not bind.
    let one = ints(&[Some(1)]);
    assert!(
        bind_scalar(
            "get_json_object",
            &[
                json.data_type().clone().into(),
                BindArg::Column(one.data_type().clone()),
            ],
            &BindOptions::new(),
        )
        .is_err()
    );
}

#[test]
fn schema_of_json_infers_and_reads_options() {
    let json = strs(&[Some("{\"c1\":0, \"c2\":[1]}"), None]);
    let f = bind("schema_of_json", &[&json]);
    assert_eq!(f.return_type(), &ArrowDataType::Utf8);
    let out = f.invoke(&[col(&json)], json.len()).unwrap();
    assert_eq!(
        text(&out),
        owned(&[Some("STRUCT<c1: BIGINT, c2: ARRAY<BIGINT>>"), None])
    );
    // `primitivesAsString` reads through a literal options map.
    let options = opts(&[("primitivesAsString", "true")]);
    let options_ty = options.data_type().clone();
    let f = bind_scalar(
        "schema_of_json",
        &[
            json.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&options)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    let out = f
        .invoke(&[col(&json), Arg::Scalar(Arc::clone(&options))], json.len())
        .unwrap();
    assert_eq!(
        text(&out),
        owned(&[Some("STRUCT<c1: STRING, c2: ARRAY<STRING>>"), None])
    );
    // A `NULL` field renders as `STRING`, recursively; fields sort by
    // name, as Spark's `JsonInferSchema` sorts them.
    let nulls = strs(&[
        Some("{\"a\":null}"),
        Some("{\"a\":[null]}"),
        Some("{\"b\":1,\"a\":2}"),
    ]);
    let f = bind("schema_of_json", &[&nulls]);
    let out = f.invoke(&[col(&nulls)], nulls.len()).unwrap();
    assert_eq!(
        text(&out),
        owned(&[
            Some("STRUCT<a: STRING>"),
            Some("STRUCT<a: ARRAY<STRING>>"),
            Some("STRUCT<a: BIGINT, b: BIGINT>"),
        ])
    );
    // `dropFieldIfAllNull` drops the void fields instead.
    let drop = opts(&[("dropFieldIfAllNull", "true")]);
    let f = bind_scalar(
        "schema_of_json",
        &[
            nulls.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&drop)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    let out = f.invoke(&[col(&nulls), lit(&drop)], nulls.len()).unwrap();
    assert_eq!(
        text(&out),
        owned(&[
            Some("STRUCT<>"),
            Some("STRUCT<a: ARRAY<STRING>>"),
            Some("STRUCT<a: BIGINT, b: BIGINT>"),
        ])
    );
    // A non-literal options map does not bind.
    assert!(
        bind_scalar(
            "schema_of_json",
            &[json.data_type().clone().into(), BindArg::Column(options_ty),],
            &BindOptions::new(),
        )
        .is_err()
    );
}

#[test]
fn from_json_reads_structs_arrays_and_maps() {
    let out = run_with_schema(
        "from_json",
        &[
            Some("{\"a\":1, \"b\":\"x\"}"),
            Some("{\"a\":null}"),
            None,
            Some("[1, 2]"),
        ],
        "a INT, b STRING",
    );
    assert_eq!(
        out.data_type(),
        &ArrowDataType::Struct(
            vec![
                Field::new("a", ArrowDataType::Int32, true),
                Field::new("b", ArrowDataType::Utf8View, true),
            ]
            .into()
        )
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(st.is_null(2));
    assert!(!st.is_null(3));
    // A non-object reads as a struct of `NULL`s.
    let a: Vec<Option<i32>> = st
        .column(0)
        .as_primitive::<arrow_array::types::Int32Type>()
        .iter()
        .collect();
    let b: Vec<Option<String>> = st
        .column(1)
        .as_string_view()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect();
    assert_eq!(a, [Some(1), None, None, None]);
    assert_eq!(b, [Some("x".to_owned()), None, None, None]);
    let out = run_with_schema("from_json", &[Some("[1, 2, null]"), None], "ARRAY<INT>");
    let list = out
        .as_any()
        .downcast_ref::<arrow_array::ListArray>()
        .unwrap();
    assert_eq!(list.value_length(0), 3);
    assert!(list.is_null(1));
    let values: Vec<Option<i32>> = list
        .values()
        .as_primitive::<arrow_array::types::Int32Type>()
        .iter()
        .collect();
    assert_eq!(values, [Some(1), Some(2), None]);
    let out = run_with_schema(
        "from_json",
        &[Some("{\"a\":1, \"b\":2}"), None],
        "MAP<STRING, INT>",
    );
    let map = out.as_any().downcast_ref::<MapArray>().unwrap();
    assert_eq!(map.value_length(0), 2);
    assert!(map.is_null(1));
    let entries = map.entries();
    let keys: Vec<Option<String>> = entries
        .column(0)
        .as_string_view()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect();
    let values: Vec<Option<i32>> = entries
        .column(1)
        .as_primitive::<arrow_array::types::Int32Type>()
        .iter()
        .collect();
    assert_eq!(keys, [Some("a".to_owned()), Some("b".to_owned())]);
    assert_eq!(values, [Some(1), Some(2)]);
}

#[test]
fn from_json_reads_options_and_dates() {
    let json = strs(&[Some("{\"time\":\"26/08/2015\"}")]);
    let schema = strs(&[Some("time Timestamp")]);
    let fmt = opts(&[("timestampFormat", "dd/MM/yyyy")]);
    let f = bind_scalar(
        "from_json",
        &[
            json.data_type().clone().into(),
            schema_arg_of(&schema),
            BindArg::Literal(Arc::clone(&fmt)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    let out = f
        .invoke(&[col(&json), lit(&schema), lit(&fmt)], json.len())
        .unwrap();
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    // `26/08/2015` parses to midnight UTC with `dd/MM/yyyy`.
    let micros: Vec<Option<i64>> = st
        .column(0)
        .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
        .iter()
        .collect();
    assert_eq!(micros, [Some(1_440_547_200_000_000)]);
    // FAILFAST with valid input is unaffected.
    let one = strs(&[Some("{\"a\":1}")]);
    let aschema = strs(&[Some("a INT")]);
    let mode = opts(&[("mode", "FAILFAST")]);
    let f = bind_scalar(
        "from_json",
        &[
            json.data_type().clone().into(),
            schema_arg_of(&aschema),
            BindArg::Literal(Arc::clone(&mode)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    let out = f
        .invoke(&[col(&one), lit(&aschema), lit(&mode)], 1)
        .unwrap();
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    let a: Vec<Option<i32>> = st
        .column(0)
        .as_primitive::<arrow_array::types::Int32Type>()
        .iter()
        .collect();
    assert_eq!(a, [Some(1)]);
    // DROPMALFORMED is rejected with PARSE_MODE_UNSUPPORTED.
    let drop = opts(&[("mode", "DROPMALFORMED")]);
    let err = bind_scalar(
        "from_json",
        &[
            json.data_type().clone().into(),
            schema_arg_of(&aschema),
            BindArg::Literal(Arc::clone(&drop)),
        ],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::ParseModeUnsupported);
    // A bad pattern fails the bind.
    assert!(
        bind_scalar(
            "from_json",
            &[
                json.data_type().clone().into(),
                schema_arg("a INT"),
                options(&[("timestampFormat", "QQQ")]),
            ],
            &BindOptions::new(),
        )
        .is_err()
    );
    // Only strings and a literal schema bind.
    let one = ints(&[Some(1)]);
    assert!(
        bind_scalar(
            "from_json",
            &[
                BindArg::Column(one.data_type().clone()),
                schema_arg("a INT")
            ],
            &BindOptions::new(),
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "from_json",
            &[
                json.data_type().clone().into(),
                BindArg::Column(ArrowDataType::Utf8),
            ],
            &BindOptions::new(),
        )
        .is_err()
    );
}

#[test]
fn from_json_reads_every_option_entry() {
    // A two-entry options map applies both entries: the format parses the
    // valid row, and the mode fails the malformed one (dropping either
    // entry would read `NULL`s or succeed instead).
    let json = strs(&[Some("{\"time\":\"26/08/2015\"}"), Some("not json")]);
    let schema = strs(&[Some("time Timestamp")]);
    let both = opts(&[("timestampFormat", "dd/MM/yyyy"), ("mode", "FAILFAST")]);
    let f = bind_scalar(
        "from_json",
        &[
            json.data_type().clone().into(),
            schema_arg_of(&schema),
            BindArg::Literal(Arc::clone(&both)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    let err = f
        .invoke(&[col(&json), lit(&schema), lit(&both)], json.len())
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::MalformedRecordInParsingWithoutSuggestion
    );
    let valid = strs(&[Some("{\"time\":\"26/08/2015\"}")]);
    let out = f
        .invoke(&[col(&valid), lit(&schema), lit(&both)], valid.len())
        .unwrap();
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    let micros: Vec<Option<i64>> = st
        .column(0)
        .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
        .iter()
        .collect();
    assert_eq!(micros, [Some(1_440_547_200_000_000)]);
}

#[test]
fn to_json_writes_structs_and_nulls() {
    let structs = run_with_schema(
        "from_json",
        &[Some("{\"a\":1, \"b\":\"x\"}"), None],
        "a INT, b STRING",
    );
    let f = bind("to_json", &[&structs]);
    assert_eq!(f.return_type(), &ArrowDataType::Utf8);
    let out = f.invoke(&[col(&structs)], structs.len()).unwrap();
    assert_eq!(text(&out), owned(&[Some("{\"a\":1,\"b\":\"x\"}"), None]));
    // Only structs, maps and arrays bind.
    let one = ints(&[Some(1)]);
    assert!(
        bind_scalar(
            "to_json",
            &[BindArg::Column(one.data_type().clone())],
            &BindOptions::new(),
        )
        .is_err()
    );
    // A non-literal options map does not bind.
    assert!(
        bind_scalar(
            "to_json",
            &[
                structs.data_type().clone().into(),
                BindArg::Column(ArrowDataType::Utf8),
            ],
            &BindOptions::new(),
        )
        .is_err()
    );
}

#[test]
fn from_json_string_field_accepts_any_token() {
    // Spark 4.2 oracle: to STRING accepts any token as its JSON text.
    let out = run_with_schema(
        "from_json",
        &[Some(
            "{\"a\":1,\"b\":1.5,\"c\":true,\"d\":[1,2],\"e\":{\"x\":1}}",
        )],
        "a STRING, b STRING, c STRING, d STRING, e STRING",
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    let col = |i: usize| {
        st.column(i)
            .as_string_view()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>()
    };
    assert_eq!(col(0), [Some("1".to_owned())]);
    assert_eq!(col(1), [Some("1.5".to_owned())]);
    assert_eq!(col(2), [Some("true".to_owned())]);
    assert_eq!(col(3), [Some("[1,2]".to_owned())]);
    assert_eq!(col(4), [Some("{\"x\":1}".to_owned())]);
}

#[test]
fn from_json_failfast_fails_malformed_valid_unaffected() {
    let json = strs(&[Some("{\"a\":1}"), Some("not json")]);
    let schema = strs(&[Some("a INT")]);
    let mode = opts(&[("mode", "FAILFAST")]);
    let f = bind_scalar(
        "from_json",
        &[
            json.data_type().clone().into(),
            schema_arg_of(&schema),
            BindArg::Literal(Arc::clone(&mode)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    // One malformed row fails the whole batch.
    let err = f
        .invoke(&[col(&json), lit(&schema), lit(&mode)], json.len())
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::MalformedRecordInParsingWithoutSuggestion
    );
    // Valid input alone is unaffected.
    let json = strs(&[Some("{\"a\":1}"), None]);
    let out = f
        .invoke(&[col(&json), lit(&schema), lit(&mode)], json.len())
        .unwrap();
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    assert!(st.is_null(1));
}

#[test]
fn schema_of_json_malformed_is_an_error() {
    let json = strs(&[Some("not json")]);
    let f = bind("schema_of_json", &[&json]);
    let err = f.invoke(&[col(&json)], json.len()).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivMalformedJson);
    // SQL NULL stays NULL; nesting past 1000 stays NULL.
    let json = strs(&[None]);
    let f = bind("schema_of_json", &[&json]);
    let out = f.invoke(&[col(&json)], json.len()).unwrap();
    assert_eq!(text(&out), owned(&[None]));
}

#[test]
fn from_json_maps_keep_duplicate_keys() {
    let out = run_with_schema(
        "from_json",
        &[Some("{\"a\":1, \"a\":2}")],
        "MAP<STRING, INT>",
    );
    let map = out.as_any().downcast_ref::<MapArray>().unwrap();
    assert!(!map.is_null(0));
    assert_eq!(map.value_length(0), 2);
}

#[test]
fn json_names_are_canonical() {
    for name in [
        "from_json",
        "to_json",
        "schema_of_json",
        "get_json_object",
        "json_array_length",
        "json_object_keys",
    ] {
        assert!(super::scalar_names().contains(&name), "{name}");
    }
    assert_eq!(
        class(&bind_scalar("nope_json", &[], &BindOptions::new()).unwrap_err()),
        ErrorClass::KrishivUnsupportedFeature
    );
}

#[test]
fn json_bind_rejects_bad_arities_and_types() {
    let one = ints(&[Some(1)]);
    let one_col = BindArg::Column(one.data_type().clone());
    let s = strs(&[Some("x")]);
    let s_col: BindArg = s.data_type().clone().into();
    // `json_array_length(json)`.
    assert!(bind_scalar("json_array_length", &[], &BindOptions::new()).is_err());
    assert!(
        bind_scalar(
            "json_array_length",
            &[s_col.clone(), s_col.clone()],
            &BindOptions::new(),
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "json_array_length",
            std::slice::from_ref(&one_col),
            &BindOptions::new()
        )
        .is_err()
    );
    // `json_object_keys(json)`.
    assert!(bind_scalar("json_object_keys", &[], &BindOptions::new()).is_err());
    assert!(
        bind_scalar(
            "json_object_keys",
            &[s_col.clone(), s_col.clone()],
            &BindOptions::new(),
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "json_object_keys",
            std::slice::from_ref(&one_col),
            &BindOptions::new()
        )
        .is_err()
    );
    // `get_json_object(json, path)`.
    assert!(bind_scalar("get_json_object", &[], &BindOptions::new()).is_err());
    assert!(
        bind_scalar(
            "get_json_object",
            std::slice::from_ref(&s_col),
            &BindOptions::new()
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "get_json_object",
            &[s_col.clone(), s_col.clone(), s_col.clone()],
            &BindOptions::new(),
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "get_json_object",
            &[one_col.clone(), s_col.clone()],
            &BindOptions::new(),
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "get_json_object",
            &[s_col.clone(), one_col.clone()],
            &BindOptions::new(),
        )
        .is_err()
    );
    // `schema_of_json(json[, options])`.
    assert!(bind_scalar("schema_of_json", &[], &BindOptions::new()).is_err());
    assert!(
        bind_scalar(
            "schema_of_json",
            &[s_col.clone(), s_col.clone(), s_col.clone()],
            &BindOptions::new(),
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "schema_of_json",
            std::slice::from_ref(&one_col),
            &BindOptions::new()
        )
        .is_err()
    );
    // `from_json(json, schema[, options])`.
    assert!(
        bind_scalar(
            "from_json",
            std::slice::from_ref(&s_col),
            &BindOptions::new()
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "from_json",
            &[one_col.clone(), schema_arg("a INT")],
            &BindOptions::new(),
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "from_json",
            &[
                s_col.clone(),
                schema_arg("a INT"),
                options(&[("mode", "PERMISSIVE")]),
                s_col.clone(),
            ],
            &BindOptions::new(),
        )
        .is_err()
    );
    // `to_json(value[, options])`.
    assert!(bind_scalar("to_json", &[], &BindOptions::new()).is_err());
    assert!(
        bind_scalar(
            "to_json",
            std::slice::from_ref(&one_col),
            &BindOptions::new()
        )
        .is_err()
    );
    let three = [s_col.clone(), s_col.clone(), s_col];
    assert!(bind_scalar("to_json", &three, &BindOptions::new()).is_err());
}

#[test]
fn json_schema_argument_errors() {
    let s = strs(&[Some("x")]);
    let s_col: BindArg = s.data_type().clone().into();
    // A column schema is not a literal.
    assert_eq!(
        class(
            &bind_scalar(
                "from_json",
                &[s_col.clone(), s_col.clone()],
                &BindOptions::new()
            )
            .unwrap_err()
        ),
        ErrorClass::KrishivUnsupportedFeature
    );
    // A null literal schema is an internal error.
    assert_eq!(
        class(
            &bind_scalar(
                "from_json",
                &[s_col.clone(), BindArg::Literal(strs(&[None]))],
                &BindOptions::new(),
            )
            .unwrap_err()
        ),
        ErrorClass::InternalError
    );
    // A schema the analyzer would reject is an internal error at bind time.
    assert_eq!(
        class(
            &bind_scalar(
                "from_json",
                &[s_col, schema_arg("{{{")],
                &BindOptions::new(),
            )
            .unwrap_err()
        ),
        ErrorClass::InternalError
    );
}

#[test]
fn json_options_argument_errors() {
    use arrow_array::StringArray;
    let s = strs(&[Some("x")]);
    let s_col: BindArg = s.data_type().clone().into();
    let schema = schema_arg("a INT");
    // Options that are not a map literal are an internal error.
    assert_eq!(
        class(
            &bind_scalar(
                "from_json",
                &[
                    s_col.clone(),
                    schema.clone(),
                    BindArg::Literal(Arc::clone(&s))
                ],
                &BindOptions::new(),
            )
            .unwrap_err()
        ),
        ErrorClass::InternalError
    );
    // A null option value is an internal error (Arrow forbids nullable map
    // keys, so only values can be null in a valid options map).
    let keys: ArrayRef = Arc::new(StringArray::from(vec![Some("mode")]));
    let values: ArrayRef = Arc::new(StringArray::from(vec![None as Option<&str>]));
    let fields: Vec<Field> = vec![
        Field::new("key", ArrowDataType::Utf8, false),
        Field::new("value", ArrowDataType::Utf8, true),
    ];
    let entries = StructArray::new(fields.into(), vec![keys, values], None);
    let map = Arc::new(
        MapArray::try_new(
            Arc::new(Field::new("entries", entries.data_type().clone(), false)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 1])),
            entries,
            None,
            false,
        )
        .unwrap(),
    );
    let null_value_args = [s_col.clone(), schema.clone(), BindArg::Literal(map)];
    assert_eq!(
        class(&bind_scalar("from_json", &null_value_args, &BindOptions::new()).unwrap_err()),
        ErrorClass::InternalError
    );
    // A null map row is skipped, so binding succeeds with the defaults.
    let keys: ArrayRef = Arc::new(StringArray::from(vec![Some("mode")]));
    let values: ArrayRef = Arc::new(StringArray::from(vec![Some("PERMISSIVE")]));
    let fields: Vec<Field> = vec![
        Field::new("key", ArrowDataType::Utf8, false),
        Field::new("value", ArrowDataType::Utf8, true),
    ];
    let entries = StructArray::new(fields.into(), vec![keys, values], None);
    let map = Arc::new(
        MapArray::try_new(
            Arc::new(Field::new("entries", entries.data_type().clone(), false)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 0])),
            entries,
            Some(arrow_buffer::NullBuffer::from(vec![false])),
            false,
        )
        .unwrap(),
    );
    let null_row_args = [s_col, schema, BindArg::Literal(map)];
    assert!(bind_scalar("from_json", &null_row_args, &BindOptions::new(),).is_ok());
}

#[test]
fn schema_of_json_deep_nesting_is_null() {
    // Deeper than `DataType::MAX_NESTING_DEPTH` (32): the inferred Arrow
    // type has no DDL form, so the row renders as NULL instead of failing.
    let deep = format!("{}1{}", "[".repeat(40), "]".repeat(40));
    let json = strs(&[Some(deep.as_str())]);
    let f = bind("schema_of_json", &[&json]);
    let out = f.invoke(&[col(&json)], json.len()).unwrap();
    assert_eq!(text(&out), owned(&[None]));
}
