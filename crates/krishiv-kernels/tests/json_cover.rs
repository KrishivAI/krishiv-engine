// SPDX-License-Identifier: Apache-2.0
//! Diff-coverage push for the JSON kernels: every value kind, merge,
//! escape, path, option, writer and reader branch the precheck lists.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Float32Type, Float64Type, Int32Type};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeListArray,
    LargeStringArray, ListArray, MapArray, NullArray, StringArray, StringViewArray, StructArray,
    TimestampMicrosecondArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};
use krishiv_kernels::cast::FixedOffset;
use krishiv_kernels::json::{
    JsonOptions, ParseMode, get_json_object, infer_json, json_array_length, json_object_keys,
    read_json, write_json,
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

fn write(col: &ArrayRef, opts: &JsonOptions) -> Vec<Option<String>> {
    let out = write_json(col, opts, &zone()).unwrap();
    out.as_string::<i32>()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect()
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

fn opts_map(pairs: &[(&str, &str)]) -> JsonOptions {
    let m: BTreeMap<String, String> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    JsonOptions::from_map(&m).unwrap()
}

// ---- infer: every kind, nesting, conflicting merges ----

#[test]
fn infer_every_kind_and_nesting() {
    let opts = JsonOptions::new();
    // Null, bool, int, float, string, empty array/object, nesting.
    let json = strs(&[
        Some("null"),
        Some("true"),
        Some("false"),
        Some("0"),
        Some("-12"),
        Some("1.5"),
        Some("-2.25e3"),
        Some("\"x\""),
        Some("[]"),
        Some("{}"),
        Some("[1,2]"),
        Some("{\"a\":1,\"b\":[true,null]}"),
        Some("[[1],[2,3]]"),
        Some(""),
        Some("   "),
        None,
    ]);
    let got = infer_json(&json, &opts).unwrap();
    assert_eq!(got[0], Some(DataType::Null));
    assert_eq!(got[1], Some(DataType::Boolean));
    assert_eq!(got[2], Some(DataType::Boolean));
    assert_eq!(got[3], Some(DataType::Int64));
    assert_eq!(got[4], Some(DataType::Int64));
    assert_eq!(got[5], Some(DataType::Float64));
    assert_eq!(got[6], Some(DataType::Float64));
    assert!(matches!(got[7], Some(DataType::Utf8View)));
    assert_eq!(
        got[8],
        Some(DataType::List(Arc::new(Field::new(
            "element",
            DataType::Null,
            true
        ))))
    );
    assert!(matches!(got[9], Some(DataType::Struct(_))));
    assert!(matches!(got[10], Some(DataType::List(_))));
    assert!(matches!(got[11], Some(DataType::Struct(_))));
    assert!(matches!(got[12], Some(DataType::List(_))));
    assert_eq!(got[13], Some(DataType::Null));
    assert_eq!(got[14], Some(DataType::Null));
    assert_eq!(got[15], None);
}

#[test]
fn infer_conflicting_merges_fall_back_to_string() {
    let opts = JsonOptions::new();
    // int + string element -> STRING; bool + int -> STRING; object + array -> STRING.
    for (text, element) in [
        ("[1,\"x\"]", DataType::Utf8View),
        ("[true,1]", DataType::Utf8View),
        ("[1,1.5]", DataType::Float64), // int+float -> double, not string
        ("[true,false]", DataType::Boolean),
        ("[1,2]", DataType::Int64),
        ("[\"a\",\"b\"]", DataType::Utf8View),
    ] {
        let got = infer_json(&strs(&[Some(text)]), &opts).unwrap();
        assert_eq!(
            got[0],
            Some(DataType::List(Arc::new(Field::new(
                "element", element, true
            )))),
            "{text}"
        );
    }
    for (text, field) in [
        // Same field conflicting -> STRING field; otherwise the field type.
        ("{\"a\":1,\"a\":\"x\"}", DataType::Utf8View),
        ("{\"a\":1}", DataType::Int64),
    ] {
        let got = infer_json(&strs(&[Some(text)]), &opts).unwrap();
        assert_eq!(
            got[0],
            Some(DataType::Struct(vec![Field::new("a", field, true)].into())),
            "{text}"
        );
    }
    // Each row infers on its own: int+float in one row stays DOUBLE, and
    // every level sorts fields by name.
    let got = infer_json(
        &strs(&[Some("{\"a\":1,\"b\":2}"), Some("{\"a\":1.5,\"c\":true}")]),
        &opts,
    )
    .unwrap();
    assert_eq!(
        got,
        [
            Some(struct_of(vec![
                Field::new("a", DataType::Int64, true),
                Field::new("b", DataType::Int64, true),
            ])),
            Some(struct_of(vec![
                Field::new("a", DataType::Float64, true),
                Field::new("c", DataType::Boolean, true),
            ])),
        ]
    );
    // Array merging: empty + non-empty, non-empty + empty.
    let got = infer_json(&strs(&[Some("[[], [1]]"), Some("[[1], []]")]), &opts).unwrap();
    assert!(matches!(got[0], Some(DataType::List(_))));
    assert!(matches!(got[1], Some(DataType::List(_))));
    // Bool vs bool stays bool; int vs int stays int.
    let got = infer_json(&strs(&[Some("[true,true]"), Some("[1,2]")]), &opts).unwrap();
    assert!(format!("{:?}", got[0]).contains("Boolean"));
    assert!(format!("{:?}", got[1]).contains("Int64"));
}

#[test]
fn infer_decimal_and_string_options() {
    // prefersDecimal: float literal becomes DECIMAL; wide falls back to DOUBLE.
    let mut pref = JsonOptions::new();
    pref.prefers_decimal = true;
    let got = infer_json(&strs(&[Some("1.5")]), &pref).unwrap();
    assert!(
        matches!(got[0], Some(DataType::Decimal128(_, _))),
        "{got:?}"
    );
    // Negative scale widens the precision: 1e3 -> DECIMAL(4,0).
    let got = infer_json(&strs(&[Some("1e3")]), &pref).unwrap();
    assert_eq!(got[0], Some(DataType::Decimal128(4, 0)));
    // Over-wide decimal falls back to DOUBLE.
    let wide = format!("0.{}", "1".repeat(50));
    let got = infer_json(&strs(&[Some(&wide)]), &pref).unwrap();
    assert_eq!(got[0], Some(DataType::Float64));
    // Empty float raw cannot happen via lexer, but decimal_or_double with
    // prefersDecimal off is DOUBLE.
    let plain = JsonOptions::new();
    let got = infer_json(&strs(&[Some("1.5")]), &plain).unwrap();
    assert_eq!(got[0], Some(DataType::Float64));
    // primitivesAsString: every primitive reads as STRING, structure kept.
    let mut prim = JsonOptions::new();
    prim.primitives_as_string = true;
    let got = infer_json(
        &strs(&[
            Some("1"),
            Some("true"),
            Some("1.5"),
            Some("\"x\""),
            Some("[1,2]"),
            Some("{\"a\":1}"),
            Some("null"),
        ]),
        &prim,
    )
    .unwrap();
    assert!(matches!(got[0], Some(DataType::Utf8View)));
    assert!(matches!(got[1], Some(DataType::Utf8View)));
    assert!(matches!(got[2], Some(DataType::Utf8View)));
    assert!(matches!(got[3], Some(DataType::Utf8View)));
    assert!(matches!(got[4], Some(DataType::List(_))));
    assert!(matches!(got[5], Some(DataType::Struct(_))));
    assert_eq!(got[6], Some(DataType::Null));
    // Malformed fails; SQL NULL stays NULL.
    assert!(infer_json(&strs(&[Some("{bad}")]), &plain).is_err());
    assert!(infer_json(&strs(&[Some("[1,")]), &plain).is_err());
    let got = infer_json(&strs(&[None]), &plain).unwrap();
    assert_eq!(got, vec![None]);
    // Duplicate object fields merge.
    let got = infer_json(&strs(&[Some("{\"a\":1,\"a\":2}")]), &plain).unwrap();
    assert!(matches!(got[0], Some(DataType::Struct(_))));
    // Bare field names with the flag.
    let mut bare = JsonOptions::new();
    bare.allow_unquoted_field_names = true;
    let got = infer_json(&strs(&[Some("{a:1}")]), &bare).unwrap();
    assert!(matches!(got[0], Some(DataType::Struct(_))));
    // Bad bare usage still errors.
    assert!(infer_json(&strs(&[Some("{a:}")]), &bare).is_err());
    // Decimal shape with exponent and sign: -12.340e-2 -> DECIMAL(5,5).
    let got = infer_json(&strs(&[Some("-12.340e-2")]), &pref).unwrap();
    assert_eq!(got[0], Some(DataType::Decimal128(5, 5)));
}

#[test]
fn infer_sorts_fields_empty_strings_and_big_ints() {
    let opts = JsonOptions::new();
    // Fields sort by name, as Spark's `JsonInferSchema` sorts them.
    let got = infer_json(&strs(&[Some("{\"b\":1,\"a\":2}")]), &opts).unwrap();
    assert_eq!(
        got[0],
        Some(struct_of(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
        ]))
    );
    // Merged objects sort too: new fields slot into name order.
    let got = infer_json(&strs(&[Some("[{\"b\":1},{\"a\":2}]")]), &opts).unwrap();
    assert_eq!(
        got[0],
        Some(list_of(Arc::new(Field::new(
            "element",
            struct_of(vec![
                Field::new("a", DataType::Int64, true),
                Field::new("b", DataType::Int64, true),
            ]),
            true
        ))))
    );
    // An empty string infers as JSON `null`: it merges away under an
    // integer, as Spark infers `STRUCT<a: BIGINT>`.
    let got = infer_json(&strs(&[Some("[{\"a\":\"\"},{\"a\":1}]")]), &opts).unwrap();
    assert_eq!(
        got[0],
        Some(list_of(Arc::new(Field::new(
            "element",
            struct_of(vec![Field::new("a", DataType::Int64, true)]),
            true
        ))))
    );
    // An integer past `i64` infers `DECIMAL(p, 0)` (past precision 38 a
    // `DOUBLE`), with or without `prefersDecimal`.
    let got = infer_json(&strs(&[Some("{\"a\":12345678901234567890}")]), &opts).unwrap();
    assert_eq!(
        got[0],
        Some(struct_of(vec![Field::new(
            "a",
            DataType::Decimal128(20, 0),
            true
        )]))
    );
    let got = infer_json(
        &strs(&[Some("{\"a\":123456789012345678901234567890123456789}")]),
        &opts,
    )
    .unwrap();
    assert_eq!(
        got[0],
        Some(struct_of(vec![Field::new("a", DataType::Float64, true)]))
    );
    // `prefersDecimal` does not count leading zeros: `0.01` is
    // `DECIMAL(2,2)` and `0.1` is `DECIMAL(1,1)`.
    let mut pref = JsonOptions::new();
    pref.prefers_decimal = true;
    let got = infer_json(&strs(&[Some("0.01"), Some("0.1")]), &pref).unwrap();
    assert_eq!(
        got,
        [
            Some(DataType::Decimal128(2, 2)),
            Some(DataType::Decimal128(1, 1))
        ]
    );
}

// ---- options: every flag, unknown, bad mode, bad patterns ----

#[test]
fn options_every_flag_and_mode() {
    let opts = opts_map(&[
        ("dateFormat", "yyyy-MM-dd"),
        ("timestampFormat", "yyyy-MM-dd HH:mm:ss"),
        ("timestampNTZFormat", "yyyy-MM-dd HH:mm:ss"),
        ("primitivesAsString", "true"),
        ("prefersDecimal", "TRUE"),
        ("allowComments", "true"),
        ("allowSingleQuotes", " True "),
        ("allowNumericLeadingZeros", "true"),
        ("allowBackslashEscapingAnyCharacter", "true"),
        ("allowUnquotedFieldNames", "true"),
        ("allowUnquotedControlChars", "true"),
        ("sortKeys", "true"),
        ("mode", "permissive"),
    ]);
    assert!(opts.date_format.is_some());
    assert!(opts.date_output.is_some());
    assert!(opts.timestamp_format.is_some());
    assert!(opts.timestamp_output.is_some());
    assert!(opts.timestamp_ntz_format.is_some());
    assert!(opts.timestamp_ntz_output.is_some());
    assert!(opts.primitives_as_string);
    assert!(opts.prefers_decimal);
    assert!(opts.allow_comments);
    assert!(opts.allow_single_quotes);
    assert!(opts.allow_numeric_leading_zeros);
    assert!(opts.allow_backslash_escaping_any);
    assert!(opts.allow_unquoted_field_names);
    assert!(opts.allow_unquoted_control_chars);
    assert!(opts.sort_keys);
    assert_eq!(opts.mode, ParseMode::Permissive);

    for (mode, expect) in [
        ("PERMISSIVE", ParseMode::Permissive),
        ("FAILFAST", ParseMode::FailFast),
        ("DROPMALFORMED", ParseMode::DropMalformed),
        ("failfast", ParseMode::FailFast),
        ("  permissive ", ParseMode::Permissive),
    ] {
        let o = opts_map(&[("mode", mode)]);
        assert_eq!(o.mode, expect, "{mode}");
    }
    // Unknown mode keeps the default; unknown and file-only options ignored.
    let o = opts_map(&[("mode", "BOGUS"), ("multiLine", "true"), ("whatever", "1")]);
    assert_eq!(o.mode, ParseMode::Permissive);
    // False spellings stay false.
    let o = opts_map(&[("sortKeys", "yes"), ("primitivesAsString", "0")]);
    assert!(!o.sort_keys);
    assert!(!o.primitives_as_string);
    // Bad patterns fail for each of the three.
    for key in ["dateFormat", "timestampFormat", "timestampNTZFormat"] {
        let m: BTreeMap<String, String> = BTreeMap::from([(key.to_owned(), "QQQ".to_owned())]);
        assert!(JsonOptions::from_map(&m).is_err(), "{key}");
    }
}

// ---- base64 via binary (encode lengths, decode errors) ----

#[test]
fn binary_base64_lengths_and_bad_inputs() {
    for bytes in [
        vec![],
        vec![0x41],
        vec![0x41, 0x42],
        vec![0x41, 0x42, 0x43],
        vec![0x41, 0x42, 0x43, 0x44],
        vec![0xff, 0x00, 0x80, 0x01, 0x02],
    ] {
        // Every byte string round-trips through base64 and back, covering
        // the encode padding branches.
        let col: ArrayRef = Arc::new(BinaryViewArray::from(vec![Some(bytes.as_slice()), None]));
        let written = write(&col, &JsonOptions::new());
        assert_eq!(written.len(), 2);
        assert_eq!(written[1], None);
        let back = read(
            &[written[0].as_deref(), None],
            &DataType::BinaryView,
            &JsonOptions::new(),
        );
        let back = back
            .as_binary_view()
            .iter()
            .map(|b| b.map(<[u8]>::to_vec))
            .collect::<Vec<_>>();
        assert_eq!(back, [Some(bytes), None]);
    }
    // Encode padding: 1, 2, 0 leftover bytes.
    use arrow_array::BinaryViewArray;
    let col: ArrayRef = Arc::new(BinaryViewArray::from(vec![
        Some(b"".as_slice()),
        Some(b"f".as_slice()),
        Some(&b"foo"[..2]),
        Some(b"foo".as_slice()),
        Some(b"foob".as_slice()),
        None,
    ]));
    let got = write(&col, &JsonOptions::new());
    assert_eq!(
        got,
        [
            Some("\"\"".to_owned()),
            Some("\"Zg==\"".to_owned()),
            Some("\"Zm8=\"".to_owned()),
            Some("\"Zm9v\"".to_owned()),
            Some("\"Zm9vYg==\"".to_owned()),
            None,
        ]
    );
    // Decode errors: bad length, bad chars, bad padding placement.
    let target = DataType::BinaryView;
    for bad in [
        "\"ABC\"",
        "\"!!!\"",
        "\"AB=C\"",
        "\"A===\"",
        "\"AB==CD\"",
        "\"ab\"",
        "\"====\"",
        "\"A\"",
    ] {
        let out = read(&[Some(bad)], &target, &JsonOptions::new());
        assert!(out.is_null(0), "{bad}");
    }
    // Lone decode of empty string is empty bytes.
    let out = read(&[Some("\"\"")], &target, &JsonOptions::new());
    assert!(!out.is_null(0));
    let w = write(&out, &JsonOptions::new());
    assert_eq!(w, [Some("\"\"".to_owned())]);
}

// ---- lexer leniency via read_json ----

#[test]
fn leniency_flags_open_up_the_lexer() {
    let target = struct_of(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
    ]);
    // Comments.
    let mut o = JsonOptions::new();
    o.allow_comments = true;
    let out = read(
        &[
            Some("{\"a\":1 /* hi */, \"b\":2}"),
            Some("// lead\n{\"a\":1,\"b\":2}"),
        ],
        &target,
        &o,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0) && !st.is_null(1));
    // Unterminated block comment poisons.
    let out = read(&[Some("{\"a\":1 /* xx")], &target, &o);
    assert!(
        out.as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .is_null(0)
    );
    // A bare slash outside comments is malformed even with the flag.
    let out = read(&[Some("/{\"a\":1}")], &target, &o);
    assert!(
        out.as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .is_null(0)
    );

    // Single quotes.
    let mut o = JsonOptions::new();
    o.allow_single_quotes = true;
    let t = struct_of(vec![Field::new("a", DataType::Utf8View, true)]);
    let out = read(&[Some("{'a':'x'}")], &t, &o);
    assert!(
        !out.as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .is_null(0)
    );

    // Leading zeros.
    let mut o = JsonOptions::new();
    o.allow_numeric_leading_zeros = true;
    let t = struct_of(vec![Field::new("a", DataType::Int32, true)]);
    let out = read(&[Some("{\"a\":007}")], &t, &o);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(st.column(0).as_primitive::<Int32Type>().value(0), 7);
    // Without the flag that poisons.
    let out = read(&[Some("{\"a\":007}")], &t, &JsonOptions::new());
    assert!(
        out.as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .is_null(0)
    );

    // Backslash-escaping-any + unquoted names + unquoted control chars.
    let mut o = JsonOptions::new();
    o.allow_backslash_escaping_any = true;
    o.allow_unquoted_field_names = true;
    o.allow_unquoted_control_chars = true;
    let t = struct_of(vec![Field::new("a", DataType::Utf8View, true)]);
    let out = read(&[Some("{a:\"\\q\"}")], &t, &o);
    // `allowBackslashEscapingAnyCharacter` reads `\q` as `q` itself.
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    assert_eq!(
        st.column(0)
            .as_string_view()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("q".to_owned())]
    );
    // Without backslash-any that poisons.
    let mut o2 = JsonOptions::new();
    o2.allow_unquoted_field_names = true;
    let out = read(&[Some("{a:\"\\q\"}")], &t, &o2);
    assert!(
        out.as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .is_null(0)
    );
    // Bare non-ascii names read with the flag, matching the field.
    let mut o3 = JsonOptions::new();
    o3.allow_unquoted_field_names = true;
    let out = read(
        &[Some("{é:1}")],
        &struct_of(vec![Field::new("é", DataType::Int32, true)]),
        &o3,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    assert_eq!(
        st.column(0)
            .as_primitive::<Int32Type>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(1)]
    );
    // Without the flag the bare name poisons the row.
    let out = read(
        &[Some("{é:1}")],
        &struct_of(vec![Field::new("é", DataType::Int32, true)]),
        &JsonOptions::new(),
    );
    assert!(
        out.as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .is_null(0)
    );
}

#[test]
fn escapes_numbers_and_literals() {
    let t = struct_of(vec![Field::new("a", DataType::Utf8View, true)]);
    let opts = JsonOptions::new();
    // Valid escapes incl. unicode and surrogate pair (𝄞 U+1D11E).
    let out = read(
        &[Some(
            "{\"a\":\"\\\"\\\\/\\b\\f\\n\\r\\t\\u00e9\\uD834\\uDD1E\"}",
        )],
        &t,
        &opts,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    // Syntax-invalid escapes fail the whole row (span splitting fails);
    // well-formed escapes with lone surrogates poison only the field.
    for bad in [
        "{\"a\":\"\\x\"}",
        "{\"a\":\"\\u00G1\"}",
        "{\"a\":\"\\u12\"}",
        "{\"a\":\"\\q\"}",
        "{\"a\":\"abc}",
        "{\"a\":\"a\x01b\"}",
    ] {
        let out = read(&[Some(bad)], &t, &opts);
        assert!(
            out.as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .is_null(0),
            "{bad}"
        );
    }
    for bad in [
        "{\"a\":\"\\uD834\"}",
        "{\"a\":\"\\uD834\\u0041\"}",
        "{\"a\":\"\\uDC00\"}",
    ] {
        let out = read(&[Some(bad)], &t, &opts);
        let st = out.as_any().downcast_ref::<StructArray>().unwrap();
        assert!(!st.is_null(0), "{bad}");
        assert!(st.column(0).is_null(0), "{bad}");
    }
    // Numbers: bad int, trailing dot, bad frac, bad exp, leading zero.
    // Lexically bad numbers fail the whole row (span splitting fails).
    let ti = struct_of(vec![Field::new("a", DataType::Int32, true)]);
    for bad in [
        "{\"a\":-}",
        "{\"a\":1.}",
        "{\"a\":1e}",
        "{\"a\":1e+}",
        "{\"a\":01}",
        "{\"a\":trvX}",
        "{\"a\":nul}",
        "{\"a\":trueX}",
        "{\"a\":--1}",
    ] {
        let out = read(&[Some(bad)], &ti, &opts);
        assert!(
            out.as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .is_null(0),
            "{bad}"
        );
    }
    // Control chars rejected without the flag, accepted with it.
    let mut o = JsonOptions::new();
    o.allow_unquoted_control_chars = true;
    let out = read(&[Some("{\"a\":\"a\x01b\"}")], &t, &o);
    assert!(
        !out.as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .is_null(0)
    );
}

// ---- JSONPath features ----

#[test]
fn path_wildcards_indexes_quoted_and_bad_paths() {
    // Wildcard over object values + multiple matches render as JSON array.
    let json = strs(&[Some("{\"a\":1,\"b\":2,\"c\":3}")]);
    let got = get_json_object(&json, &strs(&[Some("$[*]")]));
    assert_eq!(
        got.as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("[1,2,3]".to_owned())]
    );
    // Wildcard over array with field step: 3 matches.
    let json = strs(&[Some("[{\"a\":1},{\"a\":2},{\"a\":3}]")]);
    let got = get_json_object(&json, &strs(&[Some("$[*].a")]));
    assert_eq!(
        got.as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("[1,2,3]".to_owned())]
    );
    // Index then wildcard.
    let json = strs(&[Some("{\"a\":[10,20,30]}")]);
    let got = get_json_object(&json, &strs(&[Some("$.a[*]")]));
    assert_eq!(
        got.as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("[10,20,30]".to_owned())]
    );
    // Quoted names, both quote kinds, with escapes.
    let json = strs(&[Some("{\"a.b\":1,\"c'd\":2}")]);
    let got = get_json_object(&json, &strs(&[Some("$[\"a.b\"]")]));
    assert_eq!(
        got.as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("1".to_owned())]
    );
    let got = get_json_object(&json, &strs(&[Some("$['c\\'d']")]));
    // The escaped quote unescapes, so the dotted-and-quoted field matches.
    assert_eq!(
        got.as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("2".to_owned())]
    );
    // Index out of range / wrong shape -> NULL.
    for (j, p) in [
        ("[1,2]", "$[5]"),
        ("{\"a\":1}", "$[0]"),
        ("[1,2]", "$.a"),
        ("{\"a\":1}", "$.missing"),
        ("1", "$.a"),
        ("null", "$.a"),
        ("{\"a\":null}", "$.a"),
        ("not json", "$.a"),
    ] {
        let got = get_json_object(&strs(&[Some(j)]), &strs(&[Some(p)]));
        assert_eq!(
            got.as_string::<i32>()
                .iter()
                .map(|s| s.map(str::to_owned))
                .collect::<Vec<_>>(),
            [None],
            "{j} {p}"
        );
    }
    // Bad paths -> NULL (no $, empty field, bad brackets).
    // (`$` alone selects the root, so it is valid.)
    for p in [
        "a", "$.", "$[", "$[]", "$[x]", "$['a'", "$.a]", "$[*x]", "$[1x]",
    ] {
        let got = get_json_object(&strs(&[Some("{\"a\":1}")]), &strs(&[Some(p)]));
        assert_eq!(
            got.as_string::<i32>()
                .iter()
                .map(|s| s.map(str::to_owned))
                .collect::<Vec<_>>(),
            [None],
            "{p}"
        );
    }
    // Root selects the whole compact value; strings come back bare.
    let got = get_json_object(&strs(&[Some("{\"a\":[1,2]}")]), &strs(&[Some("$")]));
    assert_eq!(
        got.as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("{\"a\":[1,2]}".to_owned())]
    );
    let got = get_json_object(&strs(&[Some("\"hi\"")]), &strs(&[Some("$")]));
    assert_eq!(
        got.as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("hi".to_owned())]
    );
    // Null json / null path -> NULL; booleans and numbers render.
    let got = get_json_object(
        &strs(&[None, Some("{\"a\":true}")]),
        &strs(&[Some("$.a"), None]),
    );
    assert_eq!(
        got.as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [None, None]
    );
    let got = get_json_object(
        &strs(&[Some("{\"a\":true,\"b\":1.5}")]),
        &strs(&[Some("$.a")]),
    );
    assert_eq!(
        got.as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("true".to_owned())]
    );
    // Nested object/array rendering through Owned.
    let got = get_json_object(
        &strs(&[Some("{\"a\":{\"x\":[1,true,null]}}")]),
        &strs(&[Some("$.a")]),
    );
    assert_eq!(
        got.as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some("{\"x\":[1,true,null]}".to_owned())]
    );
}

#[test]
fn path_renders_numbers_like_jackson_and_takes_first_match() {
    let one = |json: &str, path: &str| {
        get_json_object(&strs(&[Some(json)]), &strs(&[Some(path)]))
            .as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>()
    };
    // Numbers render as Jackson renders the parsed value: a float as
    // Java's `Double.toString` (`1.0e2` reads `100.0`), an integer as
    // written.
    assert_eq!(one("{\"a\":1.0e2}", "$.a"), [Some("100.0".to_owned())]);
    assert_eq!(one("{\"a\":2.50}", "$.a"), [Some("2.5".to_owned())]);
    assert_eq!(one("{\"a\":1.5}", "$.a"), [Some("1.5".to_owned())]);
    assert_eq!(one("{\"a\":10}", "$.a"), [Some("10".to_owned())]);
    assert_eq!(
        one("{\"a\":{\"b\":1.0e2}}", "$"),
        [Some("{\"a\":{\"b\":100.0}}".to_owned())]
    );
    // Duplicate keys: the first match wins, as Jackson's field lookup does.
    assert_eq!(one("{\"a\":1,\"a\":2}", "$.a"), [Some("1".to_owned())]);
    // A wildcard still collects every match.
    assert_eq!(
        one("[{\"a\":1.0e2},{\"a\":2}]", "$[*].a"),
        [Some("[100.0,2]".to_owned())]
    );
}

#[test]
fn structural_queries_edges() {
    // array_length: nesting, whitespace, trailing content ignored, bad.
    let out = json_array_length(&strs(&[
        Some("[]"),
        Some("[null]"),
        Some("[[],{}]"),
        Some("{\"a\":1}"),
        Some(""),
        Some("[1,]"),
        Some("[1 2]"),
        None,
    ]));
    assert_eq!(
        out.as_primitive::<Int32Type>().iter().collect::<Vec<_>>(),
        [Some(0), Some(1), Some(2), None, None, None, None, None]
    );
    // object_keys: values of every shape skipped, escapes unescaped.
    let out = json_object_keys(&strs(&[
        Some("{\"a\\\"b\":1,\"c\":[1,{\"d\":null}]}"),
        Some("{\"z\":1,\"a\":2}"),
        Some("[1]"),
        Some(""),
        Some("{\"a\":1"),
        None,
    ]))
    .unwrap();
    let list = out.as_any().downcast_ref::<ListArray>().unwrap();
    assert!(!list.is_null(0));
    assert_eq!(list.value_length(1), 2);
    assert!(list.is_null(2));
    assert!(list.is_null(3));
    assert!(list.is_null(4));
    assert!(list.is_null(5));
    // Null input row for object_keys keeps offsets aligned.
    let out = json_object_keys(&strs(&[None])).unwrap();
    assert!(out.as_any().downcast_ref::<ListArray>().unwrap().is_null(0));
}

// ---- read: all scalars, keys, nesting, failfast ----

#[test]
fn read_every_scalar_type() {
    let opts = JsonOptions::new();
    // Boolean incl. string "true" stays null; int overflow nulls.
    let t = struct_of(vec![
        Field::new("b", DataType::Boolean, true),
        Field::new("s", DataType::Utf8View, true),
    ]);
    let out = read(&[Some("{\"b\":true,\"s\":false}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(st.column(0).as_boolean().value(0));
    assert_eq!(st.column(1).as_string_view().value(0), "false");

    for (ty, json, check) in [
        (DataType::Int8, "{\"a\":127}", true),
        (DataType::Int8, "{\"a\":128}", false),
        (DataType::Int16, "{\"a\":32000}", true),
        (DataType::Int16, "{\"a\":99999}", false),
        (DataType::Int32, "{\"a\":2147483647}", true),
        (DataType::Int32, "{\"a\":2147483648}", false),
        (DataType::Int64, "{\"a\":9223372036854775807}", true),
        (DataType::Int64, "{\"a\":9223372036854775808}", false),
    ] {
        let target = struct_of(vec![Field::new("a", ty.clone(), true)]);
        let out = read(&[Some(json)], &target, &opts);
        let st = out.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(!st.column(0).is_null(0), check, "{ty:?} {json}");
    }
    // Float/double via integers and floats; float narrowing.
    let t = struct_of(vec![
        Field::new("f", DataType::Float32, true),
        Field::new("d", DataType::Float64, true),
    ]);
    let out = read(&[Some("{\"f\":1,\"d\":2}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(st.column(0).as_primitive::<Float32Type>().value(0), 1.0);
    assert_eq!(st.column(1).as_primitive::<Float64Type>().value(0), 2.0);
    // Decimal incl. exponent, rounding, overflow.
    let t = struct_of(vec![Field::new("d", DataType::Decimal128(10, 2), true)]);
    let out = read(
        &[
            Some("{\"d\":1.235}"),
            Some("{\"d\":\"x\"}"),
            Some("{\"d\":1e40}"),
        ],
        &t,
        &opts,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.column(0).is_null(0));
    assert!(st.column(0).is_null(1));
    assert!(st.column(0).is_null(2));
    // Dates/timestamps/binary tested elsewhere; void target.
    let out = read(&[Some("{\"a\":1}"), None, Some("")], &DataType::Null, &opts);
    assert_eq!(out.len(), 3);
}

#[test]
fn read_decimal_rounds_once_and_bounds_exponents() {
    let opts = JsonOptions::new();
    // `HALF_UP` decides on the most significant discarded digit only:
    // `1.2349` reads `1.23`, not `1.24` (rounding twice would give the
    // latter), and the same holds below zero.
    let t = struct_of(vec![Field::new("d", DataType::Decimal128(10, 2), true)]);
    let out = read(
        &[
            Some("{\"d\":1.2349}"),
            Some("{\"d\":1.235}"),
            Some("{\"d\":-1.2349}"),
            Some("{\"d\":2.5}"),
        ],
        &t,
        &opts,
    );
    let values: Vec<Option<i128>> = out
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap()
        .column(0)
        .as_primitive::<Decimal128Type>()
        .iter()
        .collect();
    assert_eq!(values, [Some(123), Some(124), Some(-123), Some(250)]);
    // An untrusted exponent cannot set a loop count: these return at once.
    // A scale past 38 never fits (`1e-2000000000` is `NULL`), while zero
    // of any exponent reads as zero (`0e2000000000` is `0.00`).
    let out = read(
        &[
            Some("{\"d\":1e-2000000000}"),
            Some("{\"d\":0e2000000000}"),
            Some("{\"d\":1e2000000000}"),
        ],
        &t,
        &opts,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(st.column(0).is_null(0));
    let values: Vec<Option<i128>> = st
        .column(0)
        .as_primitive::<Decimal128Type>()
        .iter()
        .collect();
    assert_eq!(values, [None, Some(0), None]);
}

#[test]
fn read_huge_integers_into_string_double_and_decimal() {
    let opts = JsonOptions::new();
    let big = "1234567890123456789012345678901234567890123";
    // A 43-digit integer poisons no target: `STRING` reads its text,
    // `DOUBLE` its double, and `DECIMAL(10,2)` (which cannot hold it) a
    // field `NULL`.
    let t = struct_of(vec![
        Field::new("s", DataType::Utf8View, true),
        Field::new("d", DataType::Float64, true),
        Field::new("e", DataType::Decimal128(10, 2), true),
    ]);
    let out = read(
        &[Some(&format!("{{\"s\":{big},\"d\":{big},\"e\":{big}}}"))],
        &t,
        &opts,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    assert_eq!(
        st.column(0)
            .as_string_view()
            .iter()
            .map(|s| s.map(str::to_owned))
            .collect::<Vec<_>>(),
        [Some(big.to_owned())]
    );
    assert_eq!(
        st.column(1)
            .as_primitive::<Float64Type>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(1.2345678901234567e42)]
    );
    assert!(st.column(2).is_null(0));
}

#[test]
fn read_non_numeric_numbers_with_the_flag() {
    // `allowNonNumericNumbers` (Spark's default) reads the `NaN`,
    // `Infinity` and `INF` spellings, signed, into float fields; quoted
    // spellings read too.
    let opts = JsonOptions::new();
    assert!(opts.allow_non_numeric_numbers);
    let t = struct_of(vec![
        Field::new("d", DataType::Float64, true),
        Field::new("f", DataType::Float32, true),
    ]);
    let out = read(
        &[
            Some("{\"d\":NaN,\"f\":Infinity}"),
            Some("{\"d\":-INF,\"f\":+INF}"),
            Some("{\"d\":\"NaN\",\"f\":\"-Infinity\"}"),
        ],
        &t,
        &opts,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    let d: Vec<Option<f64>> = st.column(0).as_primitive::<Float64Type>().iter().collect();
    assert_eq!(d.len(), 3);
    assert!(d[0].is_some_and(f64::is_nan));
    assert_eq!(d[1], Some(f64::NEG_INFINITY));
    assert!(d[2].is_some_and(f64::is_nan));
    let f: Vec<Option<f32>> = st.column(1).as_primitive::<Float32Type>().iter().collect();
    assert_eq!(
        f,
        [
            Some(f32::INFINITY),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY)
        ]
    );
    // `NaN` has no decimal form, and an infinity is no integer.
    let t = struct_of(vec![
        Field::new("e", DataType::Decimal128(10, 2), true),
        Field::new("i", DataType::Int32, true),
    ]);
    let out = read(&[Some("{\"e\":NaN,\"i\":Infinity}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.is_null(0));
    assert!(st.column(0).is_null(0));
    assert!(st.column(1).is_null(0));
    // Without the flag the spellings are malformed: span splitting fails,
    // so the whole row is `NULL`.
    let mut strict = JsonOptions::new();
    strict.allow_non_numeric_numbers = false;
    let t = struct_of(vec![Field::new("d", DataType::Float64, true)]);
    let out = read(&[Some("{\"d\":NaN}")], &t, &strict);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(st.is_null(0));
    // `schema_of_json` infers a `DOUBLE` for them.
    let got = infer_json(&strs(&[Some("{\"a\":NaN}")]), &opts).unwrap();
    assert_eq!(
        got[0],
        Some(struct_of(vec![Field::new("a", DataType::Float64, true)]))
    );
}

#[test]
fn read_map_keys_of_every_type() {
    let opts = JsonOptions::new();
    // Bool keys.
    let target = map_of(
        Field::new("key", DataType::Boolean, false),
        Field::new("value", DataType::Int32, true),
    );
    let out = read(&[Some("{\"true\":1,\"false\":2}")], &target, &opts);
    assert_eq!(
        out.as_any()
            .downcast_ref::<MapArray>()
            .unwrap()
            .value_length(0),
        2
    );
    let out = read(&[Some("{\"yes\":1}")], &target, &opts);
    assert!(out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    // Integral keys incl. bad.
    for ty in [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
    ] {
        let target = map_of(
            Field::new("key", ty.clone(), false),
            Field::new("value", DataType::Int32, true),
        );
        let out = read(&[Some("{\"3\":1}")], &target, &opts);
        assert!(
            !out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0),
            "{ty:?}"
        );
        let out = read(&[Some("{\"x\":1}")], &target, &opts);
        assert!(
            out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0),
            "{ty:?}"
        );
    }
    // Float keys (Java parse incl. Infinity).
    let target = map_of(
        Field::new("key", DataType::Float64, false),
        Field::new("value", DataType::Int32, true),
    );
    let out = read(&[Some("{\"1.5\":1}")], &target, &opts);
    assert!(!out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    // Java doubles accept NaN/Infinity as keys; other text fails.
    let out = read(&[Some("{\"NaN\":1}")], &target, &opts);
    assert!(!out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    let out = read(&[Some("{\"xyz\":1}")], &target, &opts);
    assert!(out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    // Decimal keys.
    let target = map_of(
        Field::new("key", DataType::Decimal128(10, 2), false),
        Field::new("value", DataType::Int32, true),
    );
    let out = read(&[Some("{\"1.50\":1}")], &target, &opts);
    assert!(!out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    // Date keys.
    let target = map_of(
        Field::new("key", DataType::Date32, false),
        Field::new("value", DataType::Int32, true),
    );
    let out = read(&[Some("{\"2012-12-15\":1}")], &target, &opts);
    assert!(!out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    let out = read(&[Some("{\"nope\":1}")], &target, &opts);
    assert!(out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    // Binary keys (base64).
    let target = map_of(
        Field::new("key", DataType::BinaryView, false),
        Field::new("value", DataType::Int32, true),
    );
    let out = read(&[Some("{\"aGVsbG8=\":1}")], &target, &opts);
    assert!(!out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    // Timestamp keys are never valid.
    let target = map_of(
        Field::new(
            "key",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("value", DataType::Int32, true),
    );
    let out = read(&[Some("{\"2012-12-15 15:15:15\":1}")], &target, &opts);
    assert!(out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    // Malformed maps: missing colon, missing comma, truncated, bare bad key.
    let target = map_of(
        Field::new("key", DataType::Utf8View, false),
        Field::new("value", DataType::Int32, true),
    );
    for bad in [
        "{\"a\" 1}",
        "{\"a\":1 \"b\":2}",
        "{\"a\":}",
        "{\"a\":1",
        "{bad}",
    ] {
        let out = read(&[Some(bad)], &target, &opts);
        assert!(
            out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0),
            "{bad}"
        );
    }
    // Duplicate map keys kept.
    let out = read(&[Some("{\"a\":1,\"a\":2,\"a\":3}")], &target, &opts);
    assert_eq!(
        out.as_any()
            .downcast_ref::<MapArray>()
            .unwrap()
            .value_length(0),
        3
    );
}

#[test]
fn read_nested_lists_structs_and_failfast() {
    let opts = JsonOptions::new();
    // Empty list, nested empty struct, struct with missing/extra.
    let target = list_of(Arc::new(Field::new("element", DataType::Int32, true)));
    for (text, valid) in [
        ("[]", true),
        ("[1,2]", true),
        ("[1,\"x\"]", false),
        ("1", false),
        ("null", false),
        ("{\"a\":1}", false),
    ] {
        let out = read(&[Some(text)], &target, &opts);
        assert_eq!(
            !out.as_any().downcast_ref::<ListArray>().unwrap().is_null(0),
            valid,
            "{text}"
        );
    }
    // Struct: non-object values read as structs of NULLs (row valid);
    // JSON null is a row NULL.
    let target = struct_of(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
    ]);
    for (text, is_null_row) in [
        ("1", false),
        ("[1]", false),
        ("null", true),
        ("{\"a\":1,\"a\":2}", false),
    ] {
        let out = read(&[Some(text)], &target, &opts);
        let st = out.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(st.is_null(0), is_null_row, "{text}");
    }
    // Trailing garbage after struct fields poisons in PERMISSIVE.
    let out = read(&[Some("{\"a\":1} trailing")], &target, &opts);
    // The first value reads; trailing text is ignored per docs.
    assert!(
        !out.as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .is_null(0)
    );
    // FAILFAST poisons lists, maps, structs and scalars alike.
    let fail = opts_map(&[("mode", "FAILFAST")]);
    for (target, text) in [
        (
            list_of(Arc::new(Field::new("element", DataType::Int32, true))),
            "[1,\"x\"]",
        ),
        (
            map_of(
                Field::new("key", DataType::Utf8View, false),
                Field::new("value", DataType::Int32, true),
            ),
            "{\"a\":\"x\"}",
        ),
        (
            struct_of(vec![Field::new("a", DataType::Int32, true)]),
            "{\"a\":\"x\"}",
        ),
    ] {
        assert!(
            read_json(&strs(&[Some(text)]), &target, &fail, &zone(), 0).is_err(),
            "{text}"
        );
    }
    // Builder errors: LargeList, bad map entries, unknown Arrow type.
    assert!(
        read_json(
            &strs(&[Some("[1]")]),
            &DataType::LargeList(Arc::new(Field::new("element", DataType::Int32, true))),
            &opts,
            &zone(),
            0
        )
        .is_err()
    );
    assert!(
        read_json(
            &strs(&[Some("1")]),
            &DataType::Duration(arrow_schema::TimeUnit::Microsecond),
            &opts,
            &zone(),
            0
        )
        .is_err()
    );
    // Null target reads anything as NULL.
    let out = read(
        &[Some("[1,2]"), Some("{\"a\":1}"), Some("1")],
        &DataType::Null,
        &opts,
    );
    assert_eq!(out.len(), 3);
}

#[test]
fn read_dates_timestamps_binary_custom_patterns() {
    // Custom dateFormat success and failure.
    let mut o = JsonOptions::new();
    o.date_format = Some(
        krishiv_kernels::datetime_pattern::DateTimePattern::new(
            "MM/dd/yyyy",
            krishiv_kernels::datetime_pattern::Purpose::Parse,
        )
        .unwrap(),
    );
    let t = struct_of(vec![Field::new("d", DataType::Date32, true)]);
    let out = read(
        &[Some("{\"d\":\"12/15/2012\"}"), Some("{\"d\":\"nope\"}")],
        &t,
        &o,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.column(0).is_null(0));
    assert!(st.column(0).is_null(1));
    // Custom timestamp + ntz patterns.
    let mut o = JsonOptions::new();
    o.timestamp_format = Some(
        krishiv_kernels::datetime_pattern::DateTimePattern::new(
            "MM/dd/yyyy HH:mm:ss",
            krishiv_kernels::datetime_pattern::Purpose::Parse,
        )
        .unwrap(),
    );
    o.timestamp_ntz_format = Some(
        krishiv_kernels::datetime_pattern::DateTimePattern::new(
            "MM/dd/yyyy HH:mm:ss",
            krishiv_kernels::datetime_pattern::Purpose::Parse,
        )
        .unwrap(),
    );
    let t = struct_of(vec![
        Field::new(
            "t",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new(
            "n",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            true,
        ),
    ]);
    let out = read(
        &[Some(
            "{\"t\":\"12/15/2012 01:02:03\",\"n\":\"12/15/2012 01:02:03\"}",
        )],
        &t,
        &o,
    );
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!st.column(0).is_null(0));
    assert!(!st.column(1).is_null(0));
    // Bad pattern text poisons the field (parse error -> Poison).
    let mut o = JsonOptions::new();
    o.timestamp_format = Some(
        krishiv_kernels::datetime_pattern::DateTimePattern::new(
            "yyyy-MM-dd",
            krishiv_kernels::datetime_pattern::Purpose::Parse,
        )
        .unwrap(),
    );
    let t = struct_of(vec![Field::new(
        "t",
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
        true,
    )]);
    let out = read(&[Some("{\"t\":\"not-a-date\"}")], &t, &o);
    assert!(
        out.as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .column(0)
            .is_null(0)
    );
}

// ---- write: every Arrow type, keys, sort, errors ----

#[test]
fn write_every_arrow_type() {
    let opts = JsonOptions::new();
    // Null, boolean.
    let col: ArrayRef = Arc::new(NullArray::new(2));
    assert_eq!(
        write(&col, &opts),
        [Some("null".to_owned()), Some("null".to_owned())]
    );
    let col: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), Some(false), None]));
    assert_eq!(
        write(&col, &opts),
        [Some("true".to_owned()), Some("false".to_owned()), None]
    );
    // All integral widths.
    let col: ArrayRef = Arc::new(Int8Array::from(vec![Some(-8), None]));
    assert_eq!(write(&col, &opts), [Some("-8".to_owned()), None]);
    let col: ArrayRef = Arc::new(Int16Array::from(vec![Some(-16), None]));
    assert_eq!(write(&col, &opts), [Some("-16".to_owned()), None]);
    let col: ArrayRef = Arc::new(Int32Array::from(vec![Some(-32), None]));
    assert_eq!(write(&col, &opts), [Some("-32".to_owned()), None]);
    let col: ArrayRef = Arc::new(Int64Array::from(vec![Some(-64), None]));
    assert_eq!(write(&col, &opts), [Some("-64".to_owned()), None]);
    // Unsigned is rejected.
    for ty in [
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
    ] {
        let col: ArrayRef = match ty {
            DataType::UInt8 => Arc::new(arrow_array::UInt8Array::from(vec![Some(1)])),
            DataType::UInt16 => Arc::new(arrow_array::UInt16Array::from(vec![Some(1)])),
            DataType::UInt32 => Arc::new(arrow_array::UInt32Array::from(vec![Some(1)])),
            _ => Arc::new(arrow_array::UInt64Array::from(vec![Some(1)])),
        };
        assert!(write_json(&col, &opts, &zone()).is_err(), "{ty:?}");
    }
    // Float32/64 incl. special values.
    let col: ArrayRef = Arc::new(Float32Array::from(vec![Some(1.5), None]));
    assert_eq!(write(&col, &opts), [Some("1.5".to_owned()), None]);
    let col: ArrayRef = Arc::new(Float64Array::from(vec![
        Some(f64::INFINITY),
        Some(f64::NAN),
        Some(f64::NEG_INFINITY),
        None,
    ]));
    // Non-finite values write quoted, as Spark's `JacksonGenerator` does.
    assert_eq!(
        write(&col, &opts),
        [
            Some("\"Infinity\"".to_owned()),
            Some("\"NaN\"".to_owned()),
            Some("\"-Infinity\"".to_owned()),
            None
        ]
    );
    // Decimal.
    let col: ArrayRef = Arc::new(
        Decimal128Array::from(vec![Some(12345), None])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    );
    assert_eq!(write(&col, &opts), [Some("123.45".to_owned()), None]);
    // Strings in every carrier.
    let col: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None]));
    assert_eq!(write(&col, &opts), [Some("\"a\"".to_owned()), None]);
    let col: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("b"), None]));
    assert_eq!(write(&col, &opts), [Some("\"b\"".to_owned()), None]);
    let col: ArrayRef = Arc::new(StringViewArray::from(vec![Some("c"), None]));
    assert_eq!(write(&col, &opts), [Some("\"c\"".to_owned()), None]);
    // Binaries in every carrier render base64.
    let col: ArrayRef = Arc::new(BinaryArray::from(vec![Some(b"hi".as_slice()), None]));
    assert_eq!(write(&col, &opts), [Some("\"aGk=\"".to_owned()), None]);
    let col: ArrayRef = Arc::new(LargeBinaryArray::from(vec![Some(b"hi".as_slice()), None]));
    assert_eq!(write(&col, &opts), [Some("\"aGk=\"".to_owned()), None]);
    // Dates default + custom; timestamps tz + ntz default + custom.
    let col: ArrayRef = Arc::new(Date32Array::from(vec![Some(15689), None]));
    assert_eq!(
        write(&col, &opts),
        [Some("\"2012-12-15\"".to_owned()), None]
    );
    let col: ArrayRef = Arc::new(
        TimestampMicrosecondArray::from(vec![Some(1_355_584_515_000_000), None])
            .with_timezone("UTC"),
    );
    assert_eq!(
        write(&col, &opts),
        [Some("\"2012-12-15T15:15:15.000Z\"".to_owned()), None]
    );
    let col: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
        Some(1_355_584_515_000_000),
        None,
    ]));
    assert_eq!(
        write(&col, &opts),
        [Some("\"2012-12-15T15:15:15.000\"".to_owned()), None]
    );
    // Custom output patterns.
    let mut custom = JsonOptions::new();
    custom.date_output = Some(
        krishiv_kernels::datetime_pattern::DateTimePattern::new(
            "MM/dd/yyyy",
            krishiv_kernels::datetime_pattern::Purpose::Format,
        )
        .unwrap(),
    );
    let col: ArrayRef = Arc::new(Date32Array::from(vec![Some(15689)]));
    assert_eq!(write(&col, &custom), [Some("\"12/15/2012\"".to_owned())]);
    // Struct null fields are dropped by default (`ignoreNullFields`);
    // `sortKeys` sorts what is written.
    let target = struct_of(vec![
        Field::new("b", DataType::Int32, true),
        Field::new("a", DataType::Int32, true),
    ]);
    let col = read(&[Some("{\"b\":1}")], &target, &JsonOptions::new());
    assert_eq!(write(&col, &opts), [Some("{\"b\":1}".to_owned())]);
    let mut sort = JsonOptions::new();
    sort.sort_keys = true;
    assert_eq!(write(&col, &sort), [Some("{\"b\":1}".to_owned())]);
    // With `ignoreNullFields` off the nulls write again; maps and arrays
    // still write `null` either way.
    let mut keep = JsonOptions::new();
    keep.ignore_null_fields = false;
    assert_eq!(
        write(&col, &keep),
        [Some("{\"b\":1,\"a\":null}".to_owned())]
    );
    let mut keep_sort = JsonOptions::new();
    keep_sort.ignore_null_fields = false;
    keep_sort.sort_keys = true;
    assert_eq!(
        write(&col, &keep_sort),
        [Some("{\"a\":null,\"b\":1}".to_owned())]
    );
    // Other Arrow types are rejected.
    let dur: ArrayRef = Arc::new(arrow_array::DurationMicrosecondArray::from(vec![Some(1)]));
    assert!(write_json(&dur, &opts, &zone()).is_err());
    // Top-level NULL rows stay NULL.
    let col: ArrayRef = Arc::new(Int32Array::from(vec![None]));
    assert_eq!(write(&col, &opts), [None]);
    // Control chars and quotes escape.
    let col: ArrayRef = Arc::new(StringArray::from(vec![Some("a\x01\"\\\u{8}\u{c}\n\r\t")]));
    let got = write(&col, &opts);
    assert!(got[0].as_ref().unwrap().contains("\\u0001"), "{got:?}");
}

#[test]
fn write_lists_maps_and_keys() {
    let opts = JsonOptions::new();
    // List and LargeList.
    let target = list_of(Arc::new(Field::new("element", DataType::Int32, true)));
    let col = read(&[Some("[1,2]"), None], &target, &opts);
    assert_eq!(write(&col, &opts), [Some("[1,2]".to_owned()), None]);
    // LargeListArray directly.
    let values: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), Some(2)]));
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0_i64, 2, 2]));
    let field = Arc::new(Field::new("element", DataType::Int32, true));
    let col: ArrayRef = Arc::new(LargeListArray::new(field, offsets, values, None));
    assert_eq!(
        write(&col, &opts),
        [Some("[1,2]".to_owned()), Some("[]".to_owned())]
    );
    // Maps with string keys keep order; sortKeys sorts by rendered key.
    let target = map_of(
        Field::new("key", DataType::Utf8View, false),
        Field::new("value", DataType::Int32, true),
    );
    let col = read(&[Some("{\"b\":1,\"a\":2}")], &target, &opts);
    assert_eq!(write(&col, &opts), [Some("{\"b\":1,\"a\":2}".to_owned())]);
    let mut sort = JsonOptions::new();
    sort.sort_keys = true;
    assert_eq!(write(&col, &sort), [Some("{\"a\":2,\"b\":1}".to_owned())]);
    // Integer, bool, float, decimal, date, binary keys render as text.
    for (key_ty, doc) in [
        (DataType::Int32, "{\"2\":1}"),
        (DataType::Boolean, "{\"true\":1}"),
        (DataType::Float64, "{\"1.5\":1}"),
        (DataType::Date32, "{\"2012-12-15\":1}"),
    ] {
        let target = map_of(
            Field::new("key", key_ty.clone(), false),
            Field::new("value", DataType::Int32, true),
        );
        let col = read(&[Some(doc)], &target, &opts);
        assert!(
            !col.as_any().downcast_ref::<MapArray>().unwrap().is_null(0),
            "{key_ty:?}"
        );
        let got = write(&col, &opts);
        assert_eq!(got.len(), 1, "{key_ty:?}");
    }
    // Struct keys render as the array of their values.
    let key_struct = struct_of(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
    ]);
    let entries = DataType::Struct(
        vec![
            Field::new("key", key_struct, false),
            Field::new("value", DataType::Int32, true),
        ]
        .into(),
    );
    let map_target = DataType::Map(Arc::new(Field::new("entries", entries, false)), false);
    let cols = read(&[Some("{\"[1,2]\":3}")], &map_target, &opts);
    // A struct-typed key never converts, so the whole map is `NULL`.
    assert_eq!(write(&cols, &opts), [None]);
    // Binary keys write as base64.
    let target = map_of(
        Field::new("key", DataType::BinaryView, false),
        Field::new("value", DataType::Int32, true),
    );
    let col = read(&[Some("{\"aGk=\":1}")], &target, &opts);
    assert!(!col.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    assert_eq!(write(&col, &opts), [Some("{\"aGk=\":1}".to_owned())]);
    // A struct-keyed map built by hand writes keys as value arrays.
    let struct_keys = StructArray::new(
        vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8View, true),
        ]
        .into(),
        vec![
            Arc::new(Int32Array::from(vec![Some(1)])) as ArrayRef,
            Arc::new(StringViewArray::from(vec![Some("x")])) as ArrayRef,
        ],
        None,
    );
    let struct_vals: ArrayRef = Arc::new(Int32Array::from(vec![Some(9)]));
    let struct_entries = StructArray::new(
        vec![
            Field::new(
                "key",
                DataType::Struct(
                    vec![
                        Field::new("a", DataType::Int32, true),
                        Field::new("b", DataType::Utf8View, true),
                    ]
                    .into(),
                ),
                false,
            ),
            Field::new("value", DataType::Int32, true),
        ]
        .into(),
        vec![Arc::new(struct_keys) as ArrayRef, struct_vals],
        None,
    );
    let struct_map_field = Arc::new(Field::new(
        "entries",
        struct_entries.data_type().clone(),
        false,
    ));
    let struct_map: ArrayRef = Arc::new(MapArray::new(
        struct_map_field,
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1])),
        struct_entries,
        None,
        false,
    ));
    assert_eq!(
        write(&struct_map, &opts),
        [Some("{\"[1,\\\"x\\\"]\":9}".to_owned())]
    );
    assert_eq!(
        write(&struct_map, &sort),
        [Some("{\"[1,\\\"x\\\"]\":9}".to_owned())]
    );
    // Non-UTC zones render with their offset; custom timestamp outputs apply.
    let plus_hour = FixedOffset(3600);
    let col: ArrayRef = Arc::new(
        TimestampMicrosecondArray::from(vec![Some(1_355_584_515_000_000)]).with_timezone("UTC"),
    );
    let got = write_json(&col, &opts, &plus_hour).unwrap();
    assert_eq!(
        got.as_string::<i32>().iter().collect::<Vec<_>>(),
        [Some("\"2012-12-15T16:15:15.000+01:00\"")]
    );
    let minus = FixedOffset(-19_800);
    let got = write_json(&col, &opts, &minus).unwrap();
    assert_eq!(
        got.as_string::<i32>().iter().collect::<Vec<_>>(),
        [Some("\"2012-12-15T09:45:15.000-05:30\"")]
    );
    let mut custom = JsonOptions::new();
    custom.timestamp_output = Some(
        krishiv_kernels::datetime_pattern::DateTimePattern::new(
            "MM/dd/yyyy HH:mm:ss",
            krishiv_kernels::datetime_pattern::Purpose::Format,
        )
        .unwrap(),
    );
    custom.timestamp_ntz_output = Some(
        krishiv_kernels::datetime_pattern::DateTimePattern::new(
            "MM/dd/yyyy",
            krishiv_kernels::datetime_pattern::Purpose::Format,
        )
        .unwrap(),
    );
    let got = write_json(&col, &custom, &zone()).unwrap();
    assert_eq!(
        got.as_string::<i32>().iter().collect::<Vec<_>>(),
        [Some("\"12/15/2012 15:15:15\"")]
    );
    let ntz: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![Some(
        1_355_584_515_000_000,
    )]));
    let got = write_json(&ntz, &custom, &zone()).unwrap();
    assert_eq!(
        got.as_string::<i32>().iter().collect::<Vec<_>>(),
        [Some("\"12/15/2012\"")]
    );
}

#[test]
fn read_write_nulls_and_void() {
    let opts = JsonOptions::new();
    // Empty input and JSON null are NULL for every shaped target.
    for target in [
        struct_of(vec![Field::new("a", DataType::Int32, true)]),
        list_of(Arc::new(Field::new("element", DataType::Int32, true))),
        map_of(
            Field::new("key", DataType::Utf8View, false),
            Field::new("value", DataType::Int32, true),
        ),
        DataType::Int32,
        DataType::Utf8View,
    ] {
        let out = read(&[Some(""), Some("null"), None], &target, &opts);
        assert_eq!(out.null_count(), 3, "{target:?}");
    }
    // SQL NULL rows append NULL to every scalar builder.
    for target in [
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::Float32,
        DataType::Float64,
        DataType::Decimal128(10, 2),
        DataType::Utf8View,
        DataType::BinaryView,
        DataType::Date32,
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
    ] {
        let out = read(&[None, Some("null"), Some("")], &target, &opts);
        assert_eq!(out.null_count(), 3, "{target:?}");
    }
    // A void target appends NULL per row too (its null count stays zero by
    // Arrow's `NullArray` convention, so the length is checked instead).
    let out = read(&[None, Some("null"), Some("")], &DataType::Null, &opts);
    assert_eq!(out.len(), 3);
    // A map target whose entries are not two fields is rejected.
    let bad_entries = DataType::Struct(vec![Field::new("key", DataType::Utf8View, false)].into());
    let bad_map = DataType::Map(Arc::new(Field::new("entries", bad_entries, false)), false);
    assert!(read_json(&strs(&[Some("{\"a\":1}")]), &bad_map, &opts, &zone(), 0).is_err());
}

#[test]
fn read_more_map_key_shapes() {
    let opts = JsonOptions::new();
    // Float32 keys narrow like Float64 keys.
    let target = map_of(
        Field::new("key", DataType::Float32, false),
        Field::new("value", DataType::Int32, true),
    );
    let out = read(&[Some("{\"1.5\":1}")], &target, &opts);
    assert!(!out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    // A bad decimal key nulls the whole map.
    let target = map_of(
        Field::new("key", DataType::Decimal128(10, 2), false),
        Field::new("value", DataType::Int32, true),
    );
    let out = read(&[Some("{\"xyz\":1}")], &target, &opts);
    assert!(out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
    // Negative decimals round half up.
    let t = struct_of(vec![Field::new("d", DataType::Decimal128(10, 2), true)]);
    let out = read(&[Some("{\"d\":-1.235}")], &t, &opts);
    let st = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(st.column(0).as_primitive::<Decimal128Type>().value(0), -124);
    // Decimal keys with an exponent read too.
    let target = map_of(
        Field::new("key", DataType::Decimal128(10, 2), false),
        Field::new("value", DataType::Int32, true),
    );
    let out = read(&[Some("{\"1e2\":1}")], &target, &opts);
    assert!(!out.as_any().downcast_ref::<MapArray>().unwrap().is_null(0));
}
