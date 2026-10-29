// SPDX-License-Identifier: Apache-2.0
//! The `json_tuple` binding: value semantics, malformed documents, and the
//! scanner's limits.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, StringArray, StringViewArray};
use arrow_schema::{DataType, Field};

use super::{BindArg, BindOptions, ScalarFunction, bind_scalar, col, lit, strs};

/// Binds `json_tuple` over one JSON column and `fields` field columns (all
/// `STRING`, as the analyzer types them).
fn bind(json: &ArrayRef, fields: &[ArrayRef]) -> Arc<dyn ScalarFunction> {
    let mut args = vec![BindArg::Column {
        ty: json.data_type().clone(),
        nullable: true,
    }];
    for f in fields {
        args.push(BindArg::Column {
            ty: f.data_type().clone(),
            nullable: true,
        });
    }
    bind_scalar("json_tuple", &args, &BindOptions::new()).unwrap()
}

/// Invokes `f` over `rows` rows of `json` and `fields`, returning the
/// output columns as `Vec<Option<String>>` each.
fn run(
    f: &Arc<dyn ScalarFunction>,
    json: &ArrayRef,
    fields: &[ArrayRef],
    rows: usize,
) -> Vec<Vec<Option<String>>> {
    let mut args = vec![col(json)];
    for field in fields {
        args.push(col(field));
    }
    let out = f.invoke(&args, rows).unwrap();
    let s = out.as_struct();
    s.columns()
        .iter()
        .map(|c| {
            let v = c.as_string_view();
            (0..rows)
                .map(|r| (!v.is_null(r)).then(|| v.value(r).to_owned()))
                .collect()
        })
        .collect()
}

fn fields(names: &[&str]) -> Vec<ArrayRef> {
    names
        .iter()
        .map(|n| Arc::new(StringArray::from(vec![Some(*n)])) as ArrayRef)
        .collect()
}

#[test]
fn return_type_is_a_nullable_struct_of_nullable_strings() {
    let json = strs(&[Some("{}")]);
    let f = bind(&json, &fields(&["a", "b"]));
    assert_eq!(
        f.return_type(),
        &DataType::Struct(
            vec![
                Field::new("c0", DataType::Utf8View, true),
                Field::new("c1", DataType::Utf8View, true),
            ]
            .into()
        )
    );
    let out = run(&f, &json, &fields(&["a", "b"]), 1);
    assert_eq!(out, vec![vec![None], vec![None]]);
}

#[test]
fn every_value_kind_reads_as_spark_reads_it() {
    let doc = r#"{
        "s": "a\"b\\c/d\u00e9X\uD83D\uDE00",
        "int": -42,
        "dec": 1.50,
        "exp": 1e3,
        "neg_exp": -2.5E-2,
        "t": true,
        "f": false,
        "n": null,
        "obj": {"x": 1, "y": [1, "a"]},
        "arr": [1, "a", null, {"k": "v"}],
        "empty_obj": {},
        "empty_arr": [],
        "ws": "  padded  "
    }"#;
    let json = strs(&[Some(doc)]);
    let names = [
        "s",
        "int",
        "dec",
        "exp",
        "neg_exp",
        "t",
        "f",
        "n",
        "obj",
        "arr",
        "empty_obj",
        "empty_arr",
        "ws",
        "missing",
    ];
    let f = bind(&json, &fields(&names));
    let out = run(&f, &json, &fields(&names), 1);
    let row: Vec<Option<&str>> = out.iter().map(|c| c[0].as_deref()).collect();
    assert_eq!(
        row,
        [
            Some("a\"b\\c/d\u{e9}X\u{1f600}"),
            Some("-42"),
            Some("1.5"),
            Some("1000.0"),
            Some("-0.025"),
            Some("true"),
            Some("false"),
            None,
            Some(r#"{"x":1,"y":[1,"a"]}"#),
            Some(r#"[1,"a",null,{"k":"v"}]"#),
            Some("{}"),
            Some("[]"),
            Some("  padded  "),
            None,
        ]
    );
}

#[test]
fn duplicate_keys_resolve_to_the_last_occurrence() {
    let json = strs(&[Some(r#"{"a": 1, "b": 2, "a": 3}"#)]);
    let f = bind(&json, &fields(&["a", "b"]));
    let out = run(&f, &json, &fields(&["a", "b"]), 1);
    assert_eq!(
        out,
        vec![vec![Some("3".to_owned())], vec![Some("2".to_owned())]]
    );
}

#[test]
fn nested_strings_escape_as_a_json_generator_writes_them() {
    // A tab and a control character inside a nested string are normalised;
    // the forward slash is not escaped and non-ASCII rides through.
    let json = strs(&[Some("{\"o\": {\"s\": \"a\\tb\\u0001/éé\"}}")]);
    let f = bind(&json, &fields(&["o"]));
    let out = run(&f, &json, &fields(&["o"]), 1);
    assert_eq!(
        out,
        vec![vec![Some("{\"s\":\"a\\tb\\u0001/éé\"}".to_owned())]]
    );
}

#[test]
fn one_to_twenty_fields() {
    let mut doc = String::from("{");
    for i in 0..20 {
        if i > 0 {
            doc.push(',');
        }
        doc.push_str(&format!("\"k{i}\": {i}"));
    }
    doc.push('}');
    let json = strs(&[Some(doc.as_str())]);
    let names: Vec<String> = (0..20).map(|i| format!("k{i}")).collect();
    let borrowed: Vec<&str> = names.iter().map(String::as_str).collect();
    let f = bind(&json, &fields(&borrowed));
    assert_eq!(
        f.return_type(),
        &DataType::Struct(
            (0..20)
                .map(|i| Field::new(format!("c{i}"), DataType::Utf8View, true))
                .collect::<Vec<_>>()
                .into()
        )
    );
    let out = run(&f, &json, &fields(&borrowed), 1);
    assert_eq!(out.len(), 20);
    for (i, column) in out.iter().enumerate() {
        assert_eq!(column, &vec![Some(i.to_string())]);
    }
}

#[test]
fn invalid_json_anywhere_yields_all_nulls() {
    let bad = [
        "",
        "   ",
        "{",
        "{\"a\": 1",
        "{\"a\": 1,}",
        "{\"a\": trux}",
        "{\"a\": 01}",
        "{\"a\": +1}",
        "{\"a\": .5}",
        "{\"a\": 1.}",
        "{\"a\": 1e}",
        "{\"a\": 1e+}",
        "{\"a\": NaN}",
        "{\"a\": Infinity}",
        "{\"a\": \"unterminated}",
        "{\"a\": \"bad \\x escape\"}",
        "{\"a\": \"lone \\uD83D surrogate\"}",
        "{\"a\": \"lone \\uDE00 surrogate\"}",
        "{\"a\": \"\\u12\"}",
        "{\"a\": [1, 2}",
        "{\"a\": {\"b\": 1}",
        "{\"a\" 1}",
        "{\"a\": 1, \"a\": }",
        "[{\"a\": 1}] extra ]",
        "{\"a\": undefined}",
    ];
    for doc in bad {
        let json = strs(&[Some(doc)]);
        let f = bind(&json, &fields(&["a", "b"]));
        let out = run(&f, &json, &fields(&["a", "b"]), 1);
        assert_eq!(out, vec![vec![None], vec![None]], "{doc:?}");
    }
}

#[test]
fn non_object_top_levels_yield_all_nulls() {
    for doc in [
        "null",
        "true",
        "false",
        "1",
        "1.50",
        "\"just a string\"",
        "[1, 2]",
        "[]",
        "{} ",
    ] {
        let json = strs(&[Some(doc)]);
        let f = bind(&json, &fields(&["a"]));
        let out = run(&f, &json, &fields(&["a"]), 1);
        assert_eq!(out, vec![vec![None]], "{doc:?}");
    }
    // `{}` is a valid empty object: missing keys, not invalid input.
    let json = strs(&[Some("{}")]);
    let f = bind(&json, &fields(&["a"]));
    let out = run(&f, &json, &fields(&["a"]), 1);
    assert_eq!(out, vec![vec![None]]);
}

#[test]
fn null_inputs_yield_all_nulls() {
    let json = strs(&[None]);
    let f = bind(&json, &fields(&["a", "b"]));
    let out = run(&f, &json, &fields(&["a", "b"]), 1);
    assert_eq!(out, vec![vec![None], vec![None]]);
}

#[test]
fn null_field_names_yield_null_columns() {
    let json = strs(&[Some(r#"{"a": "1", "b": "2"}"#)]);
    let present = strs(&[Some("a")]);
    let missing: ArrayRef = Arc::new(StringArray::from(vec![None::<&str>]));
    let f = bind(&json, &[Arc::clone(&present), Arc::clone(&missing)]);
    let out = f
        .invoke(&[col(&json), col(&present), col(&missing)], 1)
        .unwrap();
    let s = out.as_struct();
    assert_eq!(s.column(0).as_string_view().value(0), "1");
    assert!(s.column(1).is_null(0));
}

#[test]
fn rows_are_independent_and_views_read_as_strings() {
    let json: ArrayRef = Arc::new(StringViewArray::from(vec![
        Some(r#"{"a": "x"}"#),
        None,
        Some("not json"),
        Some(r#"{"a": 1.50, "b": "y"}"#),
    ]));
    let a: ArrayRef = Arc::new(StringViewArray::from(vec![
        Some("a"),
        Some("a"),
        Some("a"),
        Some("b"),
    ]));
    let f = bind(&json, std::slice::from_ref(&a));
    let out = run(&f, &json, std::slice::from_ref(&a), 4);
    assert_eq!(
        out,
        vec![vec![Some("x".to_owned()), None, None, Some("y".to_owned())]]
    );
}

#[test]
fn non_foldable_field_names_resolve_per_row() {
    // A column of field names: each row looks up its own key.
    let json = strs(&[
        Some(r#"{"a": "1", "b": "2"}"#),
        Some(r#"{"a": "3", "b": "4"}"#),
    ]);
    let key = strs(&[Some("a"), Some("b")]);
    let f = bind(&json, std::slice::from_ref(&key));
    let out = run(&f, &json, std::slice::from_ref(&key), 2);
    assert_eq!(out, vec![vec![Some("1".to_owned()), Some("4".to_owned())]]);
}

#[test]
fn deeply_nested_input_is_invalid_past_the_depth_limit() {
    let mut doc = "{\"a\": ".to_owned();
    doc.push_str(&"[".repeat(1_001));
    doc.push_str(&"]".repeat(1_001));
    doc.push('}');
    let json = strs(&[Some(doc.as_str())]);
    let f = bind(&json, &fields(&["a"]));
    let out = run(&f, &json, &fields(&["a"]), 1);
    assert_eq!(out, vec![vec![None]]);
    // Just inside the limit still compacts.
    let mut ok = "{\"a\": ".to_owned();
    ok.push_str(&"[".repeat(999));
    ok.push_str(&"]".repeat(999));
    ok.push('}');
    let json = strs(&[Some(ok.as_str())]);
    let f = bind(&json, &fields(&["a"]));
    let out = run(&f, &json, &fields(&["a"]), 1);
    assert_eq!(out[0][0].as_deref().map(str::len), Some(1998));
}

#[test]
fn very_long_values_and_a_one_megabyte_document() {
    let big = "x".repeat(1 << 20);
    let doc = format!("{{\"a\": \"{big}\", \"b\": \"tail\"}}");
    let json = strs(&[Some(doc.as_str())]);
    let f = bind(&json, &fields(&["a", "b", "missing"]));
    let out = run(&f, &json, &fields(&["a", "b", "missing"]), 1);
    assert_eq!(out[0][0].as_deref(), Some(big.as_str()));
    assert_eq!(out[1][0].as_deref(), Some("tail"));
    assert_eq!(out[2][0], None);
}

#[test]
fn literals_broadcast_and_binding_rejects_bad_shapes() {
    let json = strs(&[Some(r#"{"a": "1"}"#)]);
    let f = bind(&json, &fields(&["a", "b"]));
    let one = |a: &ArrayRef| Arc::new(a.slice(0, 1)) as ArrayRef;
    let out = f
        .invoke(
            &[
                lit(&one(&json)),
                lit(&one(&strs(&[Some("a")]))),
                lit(&one(&strs(&[Some("b")]))),
            ],
            3,
        )
        .unwrap();
    assert_eq!(out.len(), 3);
    let s = out.as_struct();
    assert_eq!(s.column(0).as_string_view().value(2), "1");
    assert!(s.column(1).is_null(2));
    // Fewer than two arguments, and a non-string argument, do not bind.
    let opts = BindOptions::new();
    let single = BindArg::Column {
        ty: DataType::Utf8,
        nullable: true,
    };
    assert!(bind_scalar("json_tuple", std::slice::from_ref(&single), &opts).is_err());
    assert!(bind_scalar("json_tuple", &[], &opts).is_err());
    let bad = BindArg::Column {
        ty: DataType::Int32,
        nullable: true,
    };
    assert!(bind_scalar("json_tuple", &[single, bad], &opts).is_err());
}

#[test]
fn top_level_whitespace_and_case_sensitive_keys() {
    let json = strs(&[Some(" \t\r\n { \"A\": \"up\", \"a\": \"down\" } \t ")]);
    let f = bind(&json, &fields(&["A", "a"]));
    let out = run(&f, &json, &fields(&["A", "a"]), 1);
    assert_eq!(
        out,
        vec![vec![Some("up".to_owned())], vec![Some("down".to_owned())]]
    );
}

/// Every assertion above resolves `json_tuple` through the registry: the
/// kernel tests fail without the change because the name does not bind.
#[test]
fn the_registry_resolves_json_tuple_as_a_generator() {
    use crate::registry::Lookup;
    use crate::signature::SparkProfile;
    let def = match crate::registry::builtins().lookup("json_tuple", SparkProfile::V4_2) {
        Lookup::Found(def) => def,
        other => panic!("json_tuple: {other:?}"),
    };
    assert_eq!(def.kind, crate::signature::FunctionKind::Generator);
    let f = bind_scalar(
        "JSON_TUPLE",
        &[BindArg::from(DataType::Utf8), BindArg::from(DataType::Utf8)],
        &BindOptions::new(),
    )
    .unwrap();
    assert_eq!(f.name(), "json_tuple");
    // The signature stays total for a direct caller with too few arguments
    // (the analyzer's arity check raises first): Spark's `> 1` arity error.
    let one = [crate::signature::ArgType::new(
        krishiv_types::DataType::String,
        true,
    )];
    let err = (def.result)(&crate::signature::CallTypes::new(&one, &[])).unwrap_err();
    assert_eq!(
        err.error_class(),
        krishiv_common::error::ErrorClass::WrongNumArgsWithoutSuggestion
    );
}

/// Fields the query does not ask for are skipped whole, whatever their
/// shape, without disturbing the requested ones after them.
#[test]
fn unrequested_values_of_every_shape_are_skipped() {
    let doc = r#"{"skip1":{"x":[1,{"y":null},[],{}],"z":"s\"q"},"skip2":[],"skip3":{},
        "skip4":true,"skip5":false,"skip6":null,"skip7":-1.5e3,"skip8":"é\nA",
        "skip9":[1,[2,[3]],{"k":{"j":[]}}],"want":"v"}"#;
    let json = strs(&[Some(doc)]);
    let f = bind(&json, &fields(&["want"]));
    assert_eq!(
        run(&f, &json, &fields(&["want"]), 1),
        vec![vec![Some("v".to_owned())]]
    );
}

/// A malformed value inside a skipped field makes the whole document
/// invalid (every column NULL), as a malformed value anywhere does.
#[test]
fn malformed_skipped_values_invalidate_the_document() {
    for doc in [
        r#"{"a":{"x" 1},"want":"v"}"#,
        r#"{"a":{"x":1 "y":2},"want":"v"}"#,
        r#"{"a":{"x":1,},"want":"v"}"#,
        r#"{"a":[1 2],"want":"v"}"#,
        r#"{"a":[1,],"want":"v"}"#,
        r#"{"a":[1,2,"want":"v"}"#,
        r#"{"a":trux,"want":"v"}"#,
        r#"{"a":nul,"want":"v"}"#,
        r#"{"a":falsx,"want":"v"}"#,
        r#"{"a":-,"want":"v"}"#,
        r#"{"a":1.,"want":"v"}"#,
        r#"{"a":"unterminated,"want":"v"}"#,
        r#"{"a":?,"want":"v"}"#,
    ] {
        let json = strs(&[Some(doc)]);
        let f = bind(&json, &fields(&["want"]));
        assert_eq!(
            run(&f, &json, &fields(&["want"]), 1),
            vec![vec![None]],
            "{doc}"
        );
    }
}

/// Nesting past the depth limit inside a skipped value is invalid too.
#[test]
fn deeply_nested_skipped_values_are_invalid_past_the_limit() {
    for (open, close) in [("[", "]"), ("{\"k\":", "}")] {
        let doc = format!(
            "{{\"a\":{}1{},\"want\":\"v\"}}",
            open.repeat(1_001),
            close.repeat(1_001)
        );
        let json = strs(&[Some(doc.as_str())]);
        let f = bind(&json, &fields(&["want"]));
        assert_eq!(
            run(&f, &json, &fields(&["want"]), 1),
            vec![vec![None]],
            "{open}"
        );
    }
}

/// Numbers print as Jackson prints them: integers keep their digits (`-0`
/// is `0`), anything with a fraction or exponent is a `double` printed as
/// Java prints one, at the top level and inside objects and arrays (values
/// from Spark 4.2.0).
#[test]
fn numbers_are_normalised_like_jackson() {
    let doc = r#"{"a":1.0E5,"b":1.10,"c":-0,"d":1e3,"e":-2.5E-2,"f":12345678901234567890,
        "g":0.1e1,"h":1E+2,"i":1.5e300,"o":{"b":1.10},"l":[1.0E5,-0,2]}"#;
    let json = strs(&[Some(doc)]);
    let names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "o", "l"];
    let f = bind(&json, &fields(&names));
    let out = run(&f, &json, &fields(&names), 1);
    let row: Vec<Option<&str>> = out.iter().map(|c| c[0].as_deref()).collect();
    assert_eq!(
        row,
        [
            Some("100000.0"),
            Some("1.1"),
            Some("0"),
            Some("1000.0"),
            Some("-0.025"),
            Some("12345678901234567890"),
            Some("1.0"),
            Some("100.0"),
            Some("1.5E300"),
            Some(r#"{"b":1.1}"#),
            Some("[100000.0,0,2]"),
        ]
    );
}

/// What Spark's parser accepts beyond strict JSON: text after the closing
/// brace is ignored, strings may be single-quoted, and raw control
/// characters may sit inside strings (values from Spark 4.2.0).
#[test]
fn the_parser_is_as_lenient_as_sparks() {
    for (doc, want) in [
        ("{\"a\":1} trailing", "1"),
        ("{\"a\":1}{\"a\":2}", "1"),
        ("{'a':'x'}", "x"),
        ("{\"a\":\"x\ty\"}", "x\ty"),
        ("{\"a\": \"raw \u{1} control\"}", "raw \u{1} control"),
    ] {
        let json = strs(&[Some(doc)]);
        let f = bind(&json, &fields(&["a"]));
        assert_eq!(
            run(&f, &json, &fields(&["a"]), 1),
            vec![vec![Some(want.to_owned())]],
            "{doc:?}"
        );
    }
}

/// Literal field names build the key index once and still give every row
/// its own values, and a changed name column rebuilds it.
#[test]
fn key_index_is_reused_across_rows_and_rebuilt_when_names_change() {
    let json = strs(&[
        Some(r#"{"a":1,"b":2}"#),
        Some(r#"{"a":3,"b":4}"#),
        Some(r#"{"a":5,"b":6}"#),
    ]);
    let names = strs(&[Some("a"), Some("a"), Some("b")]);
    let f = bind(&json, std::slice::from_ref(&names));
    let out = run(&f, &json, std::slice::from_ref(&names), 3);
    assert_eq!(
        out,
        vec![vec![
            Some("1".to_owned()),
            Some("3".to_owned()),
            Some("6".to_owned())
        ]]
    );
}

/// Strings inside returned objects and arrays are escaped as a JSON
/// generator writes them, and the escapes of a top-level string decode
/// (values from Spark 4.2.0).
#[test]
fn escapes_round_trip_through_compact_text_and_decoding() {
    let doc = "{\"o\":{\"k\":\"q\\\"b\\\\\\n\\t\\f\\r\\b/ctl\\u0001\"},\
               \"arr\":[true,false,null,\"\u{e9}\"],\"s\":\"\\/\\b\\f\\n\\r\\t\\'x\"}";
    let json = strs(&[Some(doc)]);
    let names = ["o", "arr", "s"];
    let f = bind(&json, &fields(&names));
    let out = run(&f, &json, &fields(&names), 1);
    let row: Vec<Option<&str>> = out.iter().map(|c| c[0].as_deref()).collect();
    assert_eq!(
        row,
        [
            Some("{\"k\":\"q\\\"b\\\\\\n\\t\\f\\r\\b/ctl\\u0001\"}"),
            Some("[true,false,null,\"\u{e9}\"]"),
            Some("/\u{8}\u{c}\n\r\t'x"),
        ]
    );
}

/// A malformed value inside a requested object or array makes the document
/// invalid.
#[test]
fn malformed_requested_values_are_all_null() {
    for doc in [
        r#"{"o":{"a" 1}}"#,
        r#"{"o":{"a":1 "b":2}}"#,
        r#"{"o":[1 2]}"#,
        r#"{"o":[?]}"#,
        r#"{"o":{"a":?}}"#,
        r#"{"o":{1:2}}"#,
        r#"{"o":[tru]}"#,
        r#"{"o":-}"#,
        r#"{"o":"\uZZZZ"}"#,
        r#"{"o":[1,]}"#,
        r#"{"o":{"a":1,}}"#,
    ] {
        let json = strs(&[Some(doc)]);
        let f = bind(&json, &fields(&["o"]));
        assert_eq!(
            run(&f, &json, &fields(&["o"]), 1),
            vec![vec![None]],
            "{doc}"
        );
    }
    // Nesting past the limit inside a requested value.
    for (open, close) in [("[", "]"), ("{\"k\":", "}")] {
        let doc = format!("{{\"o\":{}1{}}}", open.repeat(1_001), close.repeat(1_001));
        let json = strs(&[Some(doc.as_str())]);
        let f = bind(&json, &fields(&["o"]));
        assert_eq!(
            run(&f, &json, &fields(&["o"]), 1),
            vec![vec![None]],
            "{open}"
        );
    }
}
