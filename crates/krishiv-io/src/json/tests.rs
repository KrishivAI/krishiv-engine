// SPDX-License-Identifier: Apache-2.0
//! The JSON format against JSON-lines records: the tokenizer, parse modes,
//! every option, inference, split boundaries, `multiLine`, encodings and
//! write round trips.
//!
//! Behaviour follows Spark 4.2's documented JSON semantics and JacksonParser
//! coercion rules; the real-Spark fixture (`tests/fixtures/spark-4.2.0/json.json`,
//! made by `tests/scripts/gen_json_goldens.py`) is not yet in this tree, so
//! the disputed coercions (boolean from string/number incl. `NaN`, nested
//! values as malformed records, base64 edge cases) are asserted explicitly
//! below against those rules.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use bytes::Bytes;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::OptionValue;
use krishiv_kernels::text::{java_double_string, java_float_string32};

use super::JsonFormat;
use super::decode::{JVal, Top, parse_line, parse_next};
use super::encode::JsonEncoder;
use super::options::Syntax;
use crate::format::{Decoder, Encoder, FileFormat, FileMetadata, Step};
use crate::io::{ByteRange, FileRef, IoConfig, IoScheduler, StoreId};
use crate::options::FormatOptions;
use crate::request::{ROW_INDEX_COLUMN, ScanRequest};
use crate::text::TextMetadata;

fn options(map: &serde_json::Value) -> FormatOptions {
    let mut user = BTreeMap::new();
    if let Some(m) = map.as_object() {
        for (k, v) in m {
            user.insert(
                k.clone(),
                OptionValue::from_user(k, v.as_str().unwrap_or_default().to_owned()),
            );
        }
    }
    JsonFormat
        .options()
        .parse(&user, &Config::new())
        .expect("options parse")
        .0
}

fn syntax_of(map: &serde_json::Value) -> Syntax {
    Syntax::from(&super::options::JsonOptions::new(&options(map), false).expect("options"))
}

/// An arrow type from a Spark DDL type name.
fn ddl_type(t: &str) -> DataType {
    match t {
        "string" => DataType::Utf8View,
        "binary" => DataType::BinaryView,
        "tinyint" => DataType::Int8,
        "smallint" => DataType::Int16,
        "int" => DataType::Int32,
        "bigint" => DataType::Int64,
        "float" => DataType::Float32,
        "double" => DataType::Float64,
        "boolean" => DataType::Boolean,
        "date" => DataType::Date32,
        "timestamp" => DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        "timestamp_ntz" => DataType::Timestamp(TimeUnit::Microsecond, None),
        d if d.starts_with("decimal(") => {
            let (p, s) = d["decimal(".len()..d.len() - 1]
                .split_once(',')
                .expect("decimal(p,s)");
            DataType::Decimal128(p.trim().parse().expect("p"), s.trim().parse().expect("s"))
        }
        other => panic!("unknown type {other}"),
    }
}

fn ddl(s: &str) -> SchemaRef {
    let mut fields = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    let mut parts = Vec::new();
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    for p in parts {
        let (n, t) = p.trim().split_once(' ').expect("name type");
        fields.push(Field::new(n, ddl_type(t.trim()), true));
    }
    Arc::new(Schema::new(fields))
}

fn timestamp_text(micros: i64) -> String {
    let t = chrono::DateTime::from_timestamp_micros(micros).expect("timestamp");
    let base = t.format("%Y-%m-%dT%H:%M:%S").to_string();
    format!("{base}.{:06}+00:00", micros.rem_euclid(1_000_000))
}

/// Spark's `CAST(value AS STRING)`, with binary as base64 (as JSON shows it).
fn render(a: &dyn Array, i: usize) -> Option<String> {
    if a.is_null(i) {
        return None;
    }
    Some(match a.data_type() {
        DataType::Boolean => a.as_boolean().value(i).to_string(),
        DataType::Int8 => a.as_primitive::<Int8Type>().value(i).to_string(),
        DataType::Int16 => a.as_primitive::<Int16Type>().value(i).to_string(),
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Float32 => java_float_string32(a.as_primitive::<Float32Type>().value(i)),
        DataType::Float64 => java_double_string(a.as_primitive::<Float64Type>().value(i)),
        DataType::Decimal128(_, s) => {
            let v = a.as_primitive::<Decimal128Type>().value(i);
            krishiv_kernels::text::java_decimal_string(v, *s)
        }
        DataType::Utf8View => a.as_string_view().value(i).to_owned(),
        DataType::BinaryView => super::convert::encode_base64(a.as_binary_view().value(i)),
        DataType::Date32 => {
            let days = a.as_primitive::<Date32Type>().value(i);
            chrono::NaiveDate::from_num_days_from_ce_opt(days + 719_163)?
                .format("%Y-%m-%d")
                .to_string()
        }
        DataType::Timestamp(..) => {
            timestamp_text(a.as_primitive::<TimestampMicrosecondType>().value(i))
        }
        other => panic!("{other}"),
    })
}

/// The bytes of `range` in `bytes`.
fn slice<'a>(bytes: &'a [u8], range: &std::ops::Range<u64>) -> &'a [u8] {
    let (a, b) = (usize::try_from(range.start), usize::try_from(range.end));
    &bytes[a.expect("range")..b.expect("range")]
}

/// Runs a decoder over `bytes` for `split`.
fn read_split(
    bytes: &[u8],
    o: &FormatOptions,
    req: &ScanRequest,
    split: ByteRange,
) -> Result<Vec<RecordBatch>, KrishivError> {
    let m: Arc<dyn FileMetadata> = Arc::new(TextMetadata {
        size: bytes.len() as u64,
    });
    let units = JsonFormat
        .plan_units(m.as_ref(), &split, req)
        .map_err(|e| e.into_error("t.json"))?;
    let mut d =
        super::decode::JsonDecoder::new(&m, &units, req, o).map_err(|e| e.into_error("t.json"))?;
    let mut out = Vec::new();
    loop {
        match d.step().map_err(|e| e.into_error("t.json"))? {
            Step::Done => return Ok(out),
            Step::Batch(b) => out.push(b),
            Step::NeedData(ranges) => {
                let data = ranges
                    .into_iter()
                    .map(|r| {
                        let slice = Bytes::copy_from_slice(slice(bytes, &r));
                        (r, slice)
                    })
                    .collect();
                d.push(data).map_err(|e| e.into_error("t.json"))?;
            }
        }
    }
}

fn read(
    bytes: &[u8],
    o: &FormatOptions,
    req: &ScanRequest,
) -> Result<Vec<RecordBatch>, KrishivError> {
    read_split(bytes, o, req, ByteRange::whole(bytes.len() as u64))
}

fn rows_of(batches: &[RecordBatch]) -> Vec<Vec<Option<String>>> {
    let mut rows = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            rows.push(b.columns().iter().map(|c| render(c.as_ref(), i)).collect());
        }
    }
    rows
}

fn row_indexed(schema: &SchemaRef) -> ScanRequest {
    let mut req = ScanRequest::all(Arc::clone(schema));
    req.metadata_cols.row_index = true;
    req
}

fn classes_of(e: &KrishivError) -> Vec<ErrorClass> {
    std::iter::once(e)
        .chain(e.causes())
        .map(KrishivError::error_class)
        .collect()
}

fn in_memory_file(bytes: &[u8], name: &str) -> (Arc<IoScheduler>, FileRef) {
    let sched = IoScheduler::new(IoConfig::default());
    let store = Arc::new(object_store::memory::InMemory::new());
    let id = StoreId::new("memory://json");
    sched.register_store(
        id.clone(),
        Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
    );
    let path = object_store::path::Path::from(name);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("runtime");
    let file = rt.block_on(async {
        object_store::ObjectStoreExt::put(store.as_ref(), &path, bytes.to_vec().into())
            .await
            .expect("put");
        let meta = object_store::ObjectStoreExt::head(store.as_ref(), &path)
            .await
            .expect("head");
        FileRef::from_meta(id, &meta)
    });
    (sched, file)
}

fn infer(bytes: &[u8], o: &FormatOptions) -> Result<SchemaRef, KrishivError> {
    infer_named(bytes, "f.json", o)
}

fn infer_named(bytes: &[u8], name: &str, o: &FormatOptions) -> Result<SchemaRef, KrishivError> {
    let (sched, file) = in_memory_file(bytes, name);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .map_err(|e| crate::error::internal(e.to_string()))?;
    let io = sched.context();
    rt.block_on(JsonFormat.infer_schema(&io, std::slice::from_ref(&file), o))
        .map_err(|e| e.into_error("f.json"))
}

fn encode(batch: &RecordBatch, o: &FormatOptions) -> Result<Vec<u8>, KrishivError> {
    let mut e = JsonEncoder::new(&batch.schema(), o).map_err(|e| e.into_error("t.json"))?;
    e.write(batch).map_err(|e| e.into_error("t.json"))?;
    let mut out = Vec::new();
    while let Some(b) = e.take_bytes() {
        out.extend_from_slice(&b);
    }
    let done = Box::new(e).finish().map_err(|e| e.into_error("t.json"))?;
    out.extend_from_slice(&done.tail);
    Ok(out)
}

fn schema_string(s: &Schema) -> String {
    let fields: Vec<String> = s
        .fields()
        .iter()
        .map(|f| format!("{}:{}", f.name(), f.data_type()))
        .collect();
    format!("struct<{}>", fields.join(","))
}

#[test]
fn tokenizer_reads_flat_values() {
    let s = syntax_of(&serde_json::json!({}));
    // Every scalar kind, escapes, an empty object and later duplicates.
    let top = parse_line(
        r#"{"n":null,"t":true,"f":false,"i":-12,"d":1.5e3,"s":"a\"b\\c\/d\be\ff\ng\rh\ti","u":"é☃","e":{},"a":[1],"z":{} }"#,
        s,
    )
    .expect("record");
    let Top::Object(o) = top else {
        panic!("object");
    };
    assert_eq!(o.fields.len(), 10);
    assert_eq!(o.fields[0], ("n".to_owned(), JVal::Null));
    assert_eq!(o.fields[1], ("t".to_owned(), JVal::Bool(true)));
    assert_eq!(o.fields[2], ("f".to_owned(), JVal::Bool(false)));
    assert_eq!(
        o.fields[3],
        ("i".to_owned(), JVal::Num("-12".to_owned(), false))
    );
    assert_eq!(
        o.fields[4],
        ("d".to_owned(), JVal::Num("1.5e3".to_owned(), true))
    );
    assert_eq!(
        o.fields[5],
        (
            "s".to_owned(),
            JVal::Str("a\"b\\c/d\u{8}e\u{c}f\ng\rh\ti".to_owned())
        )
    );
    assert_eq!(o.fields[6], ("u".to_owned(), JVal::Str("é☃".to_owned())));
    assert!(matches!(o.fields[7].1, JVal::Nested(_)));
    assert!(matches!(o.fields[8].1, JVal::Nested(_)));
    assert!(matches!(o.fields[9].1, JVal::Nested(_)));
    // Surrogate pairs and lone surrogates.
    let top = parse_line(r#"{"s":"Aß𝄞"}"#, s).expect("surrogates");
    let Top::Object(o) = top else {
        panic!("object")
    };
    assert_eq!(o.fields[0].1, JVal::Str("Aß𝄞".to_owned()));
    assert!(parse_line("{\"s\":\"\\uD83D\"}", s).is_err());
    assert!(parse_line("{\"s\":\"\\uDE00\"}", s).is_err());
    assert!(parse_line("{\"s\":\"\\u12\"}", s).is_err());
    // Duplicates: the last wins.
    let Top::Object(o) = parse_line(r#"{"a":1,"a":2}"#, s).expect("dup") else {
        panic!("object")
    };
    assert_eq!(
        o.fields,
        vec![("a".to_owned(), JVal::Num("2".to_owned(), false))]
    );
    // Empty objects and blank input.
    assert!(matches!(
        parse_line("{}", s),
        Ok(Top::Object(o)) if o.fields.is_empty()
    ));
    assert!(parse_line("  ", s).is_err());
    // Trailing text and structural defects are malformed.
    for bad in [
        r#"{"a":1} {"b":2}"#,
        r#"{"a":1,}"#,
        r#"{"a"}"#,
        r#"{"a":}"#,
        r#"{"a":1"#,
        r#"{"a""#,
        r#"{"a":01}"#,
        r#"{"a":1.}"#,
        r#"{"a":1e}"#,
        r#"{"a":t}"#,
        r#"{"a":nul}"#,
        r#"{"a":1}x"#,
        r#"{"a":"x"#,
        r#"{"a":'x'}"#,
        "{a:1}",
    ] {
        assert!(parse_line(bad, s).is_err(), "{bad:?} is malformed");
    }
    // Top-level arrays and scalars parse (the decoder treats them as
    // malformed records, not tokenizer failures).
    assert!(matches!(parse_line("[1]", s), Ok(Top::Other(_))));
    assert!(matches!(parse_line("42", s), Ok(Top::Other(_))));
}

#[test]
fn tokenizer_options_change_the_syntax() {
    // Comments.
    let comments = syntax_of(&serde_json::json!({"allowComments": "true"}));
    let Top::Object(o) =
        parse_line("{\"a\":1 /* c */, // d\n \"b\":/* e */2}", comments).expect("comments")
    else {
        panic!("object")
    };
    assert_eq!(o.fields.len(), 2);
    assert!(parse_line("{\"a\":1 /* x", comments).is_err());
    assert!(parse_line("{\"a\":1}", syntax_of(&serde_json::json!({}))).is_ok());
    // Unquoted field names.
    let bare = syntax_of(&serde_json::json!({"allowUnquotedFieldNames": "true"}));
    let Top::Object(o) = parse_line("{a1:1,$b:\"x\",_c:true}", bare).expect("bare") else {
        panic!("object")
    };
    assert_eq!(o.fields.len(), 3);
    assert!(parse_line("{1a:1}", bare).is_err());
    // Single quotes.
    let single = syntax_of(&serde_json::json!({"allowSingleQuotes": "true"}));
    let Top::Object(o) = parse_line("{'a':'b\\'c'}", single).expect("single") else {
        panic!("object")
    };
    assert_eq!(o.fields[0].1, JVal::Str("b'c".to_owned()));
    // Leading zeros.
    let zeros = syntax_of(&serde_json::json!({"allowNumericLeadingZeros": "true"}));
    assert!(matches!(
        parse_line("{\"a\":007}", zeros),
        Ok(Top::Object(_))
    ));
    // Backslash before any character.
    let any = syntax_of(&serde_json::json!({"allowBackslashEscapingAnyCharacter": "true"}));
    let Top::Object(o) = parse_line("{\"a\":\"\\q\\u0041\"}", any).expect("any") else {
        panic!("object")
    };
    assert_eq!(o.fields[0].1, JVal::Str("qA".to_owned()));
    assert!(parse_line("{\"a\":\"\\q\"}", syntax_of(&serde_json::json!({}))).is_err());
    // Unquoted control characters.
    let controls = syntax_of(&serde_json::json!({"allowUnquotedControlChars": "true"}));
    assert!(parse_line("{\"a\":\"\u{1}\"}", controls).is_ok());
    assert!(parse_line("{\"a\":\"\u{1}\"}", syntax_of(&serde_json::json!({}))).is_err());
    // `multiLine` splits top-level values across lines and comments.
    let text = "{\n \"a\" : 1 ,\n \"b\" : [1,\n2]\n}\n{\"a\":2}\n";
    let pos = 0;
    let (first, first_end) = parse_next(text, pos, syntax_of(&serde_json::json!({})))
        .expect("first")
        .expect("value");
    assert!(matches!(first, Top::Object(_)));
    let (second, second_end) = parse_next(text, first_end, syntax_of(&serde_json::json!({})))
        .expect("second")
        .expect("value");
    assert!(matches!(second, Top::Object(_)));
    assert_eq!(text.get(first_end..second_end), Some("\n{\"a\":2}"));
    assert!(
        parse_next(text, second_end, syntax_of(&serde_json::json!({})))
            .expect("end")
            .is_none()
    );
}

#[test]
fn parse_modes_and_the_corrupt_column() {
    let schema = ddl("a int, b string, _corrupt_record string");
    let bytes = b"{\"a\":1,\"b\":\"x\"}\n{\"a\":\"bad\",\"b\":\"y\"}\n{\"a\":2";
    // PERMISSIVE keeps good fields and reports the raw record.
    let o = options(&serde_json::json!({}));
    let got = read(bytes, &o, &ScanRequest::all(Arc::clone(&schema))).expect("read");
    assert_eq!(
        rows_of(&got),
        [
            vec![Some("1".into()), Some("x".into()), None],
            vec![
                None,
                Some("y".into()),
                Some("{\"a\":\"bad\",\"b\":\"y\"}".into())
            ],
            vec![None, None, Some("{\"a\":2".into())],
        ]
    );
    // Without the corrupt column the good fields still come back.
    let plain = ddl("a int, b string");
    let got = read(bytes, &o, &ScanRequest::all(Arc::clone(&plain))).expect("read");
    assert_eq!(
        rows_of(&got),
        [
            vec![Some("1".into()), Some("x".into())],
            vec![None, Some("y".into())],
            vec![None, None],
        ]
    );
    // DROPMALFORMED drops bad records.
    let o = options(&serde_json::json!({"mode": "DROPMALFORMED"}));
    let got = read(bytes, &o, &ScanRequest::all(Arc::clone(&plain))).expect("read");
    assert_eq!(rows_of(&got), [vec![Some("1".into()), Some("x".into())]]);
    // FAILFAST fails with Spark's classes: a syntactic defect carries no
    // classified cause.
    let o = options(&serde_json::json!({"mode": "FAILFAST"}));
    let e = read(
        b"{\"a\":1,\"b\":\"x\"}\n{\"a\":2",
        &o,
        &ScanRequest::all(Arc::clone(&plain)),
    )
    .expect_err("failfast");
    assert_eq!(
        classes_of(&e),
        [
            ErrorClass::FailedReadFileNoHint,
            ErrorClass::MalformedRecordInParsingWithoutSuggestion,
        ]
    );
    // A value that cannot convert fails the same way.
    let e = read(
        b"{\"a\":\"bad\"}\n",
        &o,
        &ScanRequest::all(Arc::clone(&plain)),
    )
    .expect_err("conversion");
    assert_eq!(
        classes_of(&e),
        [
            ErrorClass::FailedReadFileNoHint,
            ErrorClass::MalformedRecordInParsingWithoutSuggestion,
        ]
    );
    // A value of the wrong kind carries its classified cause.
    let e = read(
        b"{\"a\":1,\"b\":2}\n",
        &o,
        &ScanRequest::all(ddl("a int, b date")),
    )
    .expect_err("kind");
    assert_eq!(
        classes_of(&e),
        [
            ErrorClass::FailedReadFileNoHint,
            ErrorClass::MalformedRecordInParsingWithoutSuggestion,
            ErrorClass::CannotParseJsonField,
        ]
    );
    // A custom corrupt column name is honoured.
    let o = options(&serde_json::json!({"columnNameOfCorruptRecord": "bad"}));
    let schema = ddl("a int, bad string");
    let got = read(
        b"{\"a\":\"bad\"}\n",
        &o,
        &ScanRequest::all(Arc::clone(&schema)),
    )
    .expect("read");
    assert_eq!(rows_of(&got), [vec![None, Some("{\"a\":\"bad\"}".into())]]);
}

#[test]
fn top_level_arrays_and_scalars_are_malformed() {
    use ErrorClass::{
        FailedReadFileNoHint as NoHint, InvalidJsonRootField as Root,
        MalformedRecordInParsingCannotParseJsonArraysAsStructs as Arrays,
        MalformedRecordInParsingWithoutSuggestion as Without,
    };
    let schema = ddl("a int, _corrupt_record string");
    let req = ScanRequest::all(Arc::clone(&schema));
    let permissive = options(&serde_json::json!({}));
    let got = read(b"[1,2]\n42\n\"s\"\n", &permissive, &req).expect("read");
    assert_eq!(
        rows_of(&got),
        [
            vec![None, Some("[1,2]".into())],
            vec![None, Some("42".into())],
            vec![None, Some("\"s\"".into())],
        ]
    );
    let fail = options(&serde_json::json!({"mode": "FAILFAST"}));
    let e = read(b"[1,2]\n", &fail, &req).expect_err("array");
    assert_eq!(classes_of(&e), [NoHint, Without, Arrays]);
    let e = read(b"42\n", &fail, &req).expect_err("scalar");
    assert_eq!(classes_of(&e), [NoHint, Without, Root]);
    let drop = options(&serde_json::json!({"mode": "DROPMALFORMED"}));
    let got = read(b"[1,2]\n{\"a\":1}\n", &drop, &req).expect("read");
    assert_eq!(rows_of(&got), [vec![Some("1".into()), None]]);
}

#[test]
fn all_flat_types_read() {
    let schema = ddl(
        "b boolean, tiny tinyint, small smallint, i int, big bigint, f float, d double, \
         dec decimal(10,2), wide decimal(38,10), s string, bin binary, dt date, ts timestamp, ntz timestamp_ntz",
    );
    let text = concat!(
        "{\"b\":true,\"tiny\":1,\"small\":-2,\"i\":3,\"big\":9223372036854775807,",
        "\"f\":1.5,\"d\":-1e3,\"dec\":\"12.30\",\"wide\":0.0000000001,",
        "\"s\":\"a\\\"b\",\"bin\":\"qw==\",\"dt\":\"2020-01-02\",\"ts\":\"2020-01-02T03:04:05.123456+00:00\",",
        "\"ntz\":\"2020-01-02T03:04:05.123456\"}\n",
        "{\"b\":\"TRUE\",\"tiny\":\"4\",\"small\":5,\"i\":6,\"big\":7,\"f\":\"NaN\",\"d\":\"Infinity\",",
        "\"dec\":1,\"wide\":2,\"s\":3,\"bin\":null,\"dt\":null,\"ts\":null,\"ntz\":null}\n",
        "{\"b\":1,\"tiny\":true,\"i\":1.0}\n",
    );
    let o = options(&serde_json::json!({}));
    let got = read(text.as_bytes(), &o, &ScanRequest::all(Arc::clone(&schema))).expect("read");
    let rows = rows_of(&got);
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows[0],
        [
            Some("true".into()),
            Some("1".into()),
            Some("-2".into()),
            Some("3".into()),
            Some("9223372036854775807".into()),
            Some("1.5".into()),
            Some("-1000.0".into()),
            Some("12.30".into()),
            Some("1E-10".into()),
            Some("a\"b".into()),
            Some("qw==".into()),
            Some("2020-01-02".into()),
            Some("2020-01-02T03:04:05.123456+00:00".into()),
            Some("2020-01-02T03:04:05.123456+00:00".into()),
        ]
    );
    assert_eq!(rows[1][0], Some("true".into()));
    assert_eq!(rows[1][5], Some("NaN".into()));
    assert_eq!(rows[1][6], Some("Infinity".into()));
    assert_eq!(rows[1][9], Some("3".into()));
    assert_eq!(rows[1][10], None);
    // Numeric `1` reads as boolean `true`; `1.0` never reads as an integer:
    // the record is malformed in every mode.
    assert_eq!(rows[2][0], Some("true".into()));
    assert_eq!(rows[2][3], None);
    assert_eq!(rows[2][1], Some("1".into()));
}

#[test]
fn boolean_strings_and_numbers_follow_spark_rules() {
    // Only `true`/`false` (any case) read as booleans: anything else is a
    // malformed record, as in the CSV converter and Spark's JacksonParser.
    // Numbers read as `!= 0` (`NaN` reads `true`, as `NaN != 0` in Java).
    let schema = ddl("b boolean, _corrupt_record string");
    let permissive = options(&serde_json::json!({}));
    for (text, want) in [
        ("{\"b\":\"true\"}\n", Some("true")),
        ("{\"b\":\"TRUE\"}\n", Some("true")),
        ("{\"b\":\"False\"}\n", Some("false")),
        ("{\"b\":true}\n", Some("true")),
        ("{\"b\":1}\n", Some("true")),
        ("{\"b\":0}\n", Some("false")),
        ("{\"b\":0.0}\n", Some("false")),
        ("{\"b\":1.5}\n", Some("true")),
        ("{\"b\":NaN}\n", Some("true")),
    ] {
        let got = read(
            text.as_bytes(),
            &permissive,
            &ScanRequest::all(Arc::clone(&schema)),
        )
        .expect("read");
        assert_eq!(rows_of(&got)[0][0].as_deref(), want, "{text:?}");
    }
    for text in [
        "{\"b\":\"yes\"}\n",
        "{\"b\":\"1\"}\n",
        "{\"b\":\"\"}\n",
        "{\"b\":\"TRUE \"}\n",
    ] {
        let got = read(
            text.as_bytes(),
            &permissive,
            &ScanRequest::all(Arc::clone(&schema)),
        )
        .expect("malformed permissive");
        assert_eq!(
            rows_of(&got),
            [vec![None, Some(text.trim_end().to_owned())]],
            "{text:?}"
        );
        let drop = options(&serde_json::json!({"mode": "DROPMALFORMED"}));
        let got = read(
            text.as_bytes(),
            &drop,
            &ScanRequest::all(Arc::clone(&schema)),
        )
        .expect("malformed drop");
        assert!(rows_of(&got).is_empty(), "{text:?}");
        let fail = options(&serde_json::json!({"mode": "FAILFAST"}));
        let e = read(
            text.as_bytes(),
            &fail,
            &ScanRequest::all(Arc::clone(&schema)),
        )
        .expect_err("malformed failfast");
        assert_eq!(
            classes_of(&e),
            [
                ErrorClass::FailedReadFileNoHint,
                ErrorClass::MalformedRecordInParsingWithoutSuggestion,
            ],
            "{text:?}"
        );
    }
}

#[test]
fn primitives_as_string_reads_scalars_as_text() {
    let schema = ddl("a string, b string, c string, d string, _corrupt_record string");
    let o = options(&serde_json::json!({"primitivesAsString": "true"}));
    let got = read(
        b"{\"a\":1,\"b\":1.5,\"c\":true,\"d\":null}\n{\"a\":oops}\n",
        &o,
        &ScanRequest::all(Arc::clone(&schema)),
    )
    .expect("read");
    assert_eq!(
        rows_of(&got),
        [
            vec![
                Some("1".into()),
                Some("1.5".into()),
                Some("true".into()),
                None,
                None
            ],
            vec![None, None, None, None, Some("{\"a\":oops}".into())],
        ]
    );
    // A nested value in a string column reads as its compact JSON text.
    let got = read(
        b"{\"a\":[1, {\"b\": \"x y\"}]}\n",
        &o,
        &ScanRequest::all(Arc::clone(&schema)),
    )
    .expect("nested as text");
    assert_eq!(
        rows_of(&got),
        [vec![
            Some("[1,{\"b\":\"x y\"}]".into()),
            None,
            None,
            None,
            None
        ]]
    );
    // In a non-string column it is a malformed record: PERMISSIVE keeps
    // nulls plus the raw record, DROPMALFORMED drops the record.
    let ints = ddl("a int, _corrupt_record string");
    let got = read(b"{\"a\":[1]}\n", &o, &ScanRequest::all(Arc::clone(&ints)))
        .expect("nested permissive");
    assert_eq!(rows_of(&got), [vec![None, Some("{\"a\":[1]}".into())]]);
    let drop = options(&serde_json::json!({"mode": "DROPMALFORMED"}));
    let got = read(
        b"{\"a\":[1]}\n{\"a\":2}\n",
        &drop,
        &ScanRequest::all(Arc::clone(&ints)),
    )
    .expect("nested drop");
    assert_eq!(rows_of(&got), [vec![Some("2".into()), None]]);
}

#[test]
fn nested_data_is_malformed_and_nested_schemas_are_unsupported() {
    use ErrorClass::{KrishivUnsupportedFeature as Unsupported, UnsupportedDataTypeForDatasource};
    let o = options(&serde_json::json!({}));
    // A nested value in a non-string column is a malformed record, not a failed scan:
    // PERMISSIVE keeps the good fields plus the raw record in the corrupt
    // column (or nulls without it).
    let corrupt = ddl("a int, n int, _corrupt_record string");
    for record in [
        "{\"a\":1,\"n\":{\"b\":2}}",
        "{\"a\":1,\"n\":[1,2]}",
        "{\"a\":1,\"n\":[[[]]]}",
        "{\"n\":{}}",
    ] {
        let got = read(
            format!("{record}\n").as_bytes(),
            &o,
            &ScanRequest::all(Arc::clone(&corrupt)),
        )
        .expect("nested permissive");
        let rows = rows_of(&got);
        assert_eq!(rows.len(), 1, "{record}");
        assert_eq!(rows[0][2].as_deref(), Some(record), "{record}");
        if record.starts_with("{\"a\":1") {
            assert_eq!(rows[0][0].as_deref(), Some("1"), "{record}");
        }
        assert_eq!(rows[0][1], None, "{record}");
    }
    let plain = ddl("a int, n int");
    let got = read(
        b"{\"a\":1,\"n\":{\"b\":2}}\n{\"a\":2,\"n\":7}\n",
        &o,
        &ScanRequest::all(Arc::clone(&plain)),
    )
    .expect("nested permissive without corrupt column");
    assert_eq!(
        rows_of(&got),
        [
            vec![Some("1".into()), None],
            vec![Some("2".into()), Some("7".into())],
        ]
    );
    // DROPMALFORMED drops nested records; FAILFAST fails with the nested
    // cause under `MALFORMED_RECORD_IN_PARSING`.
    let drop = options(&serde_json::json!({"mode": "DROPMALFORMED"}));
    let got = read(
        b"{\"a\":1,\"n\":{\"b\":2}}\n{\"a\":2,\"n\":7}\n",
        &drop,
        &ScanRequest::all(Arc::clone(&plain)),
    )
    .expect("nested drop");
    assert_eq!(rows_of(&got), [vec![Some("2".into()), Some("7".into())]]);
    let fail = options(&serde_json::json!({"mode": "FAILFAST"}));
    let e = read(
        b"{\"a\":1,\"n\":{\"b\":2}}\n",
        &fail,
        &ScanRequest::all(Arc::clone(&plain)),
    )
    .expect_err("nested failfast");
    assert_eq!(
        classes_of(&e),
        [
            ErrorClass::FailedReadFileNoHint,
            ErrorClass::MalformedRecordInParsingWithoutSuggestion,
        ]
    );
    // Unrequested nested fields are pruned away, not an error.
    let got = read(
        b"{\"a\":1,\"n\":{\"b\":2}}\n",
        &o,
        &ScanRequest::all(ddl("a int")),
    )
    .expect("pruned");
    assert_eq!(rows_of(&got), [vec![Some("1".into())]]);
    // A nested requested schema is refused at open and at write planning.
    let nested = Arc::new(Schema::new(vec![Field::new(
        "n",
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        true,
    )]));
    let e = read(b"{}\n", &o, &ScanRequest::all(Arc::clone(&nested))).expect_err("nested");
    assert_eq!(e.error_class(), Unsupported);
    let batch = RecordBatch::new_empty(nested);
    let e = encode(&batch, &o).expect_err("nested write");
    assert_eq!(e.error_class(), UnsupportedDataTypeForDatasource);
    let e = JsonFormat
        .check_write(&batch.schema(), &o)
        .expect_err("nested check");
    assert_eq!(
        e,
        crate::error::FormatError::Other(
            krishiv_common::error::UnsupportedDataTypeForDatasource {
                format: "JSON".to_owned(),
                column_name: "`n`".to_owned(),
                column_type: "\"array<int>\"".to_owned(),
            }
            .into()
        )
    );
}

#[test]
fn decimal_merges_fall_back_to_double_past_38_digits() {
    let o = options(&serde_json::json!({"prefersDecimal": "true"}));
    let ty = |text: &str| {
        let s = infer(text.as_bytes(), &o).expect("infer");
        s.field_with_name("a").expect("a").data_type().clone()
    };
    assert_eq!(
        ty("{\"a\":1.5}\n{\"a\":12.25}\n"),
        DataType::Decimal128(4, 2)
    );
    // 30 integer digits next to 10 fractional ones needs 40 digits.
    assert_eq!(
        ty("{\"a\":12345678901234567890.1234567890}\n{\"a\":123456789012345678901234567890.5}\n"),
        DataType::Float64
    );
    // A long next to a decimal merges as DECIMAL(20, 0).
    assert_eq!(
        ty("{\"a\":1}\n{\"a\":123456789012345678901234567890.5}\n"),
        DataType::Decimal128(31, 1)
    );
}

#[test]
fn non_finite_floats_write_as_quoted_strings() {
    let o = options(&serde_json::json!({}));
    let batch = RecordBatch::try_new(
        ddl("d double, f float"),
        vec![
            Arc::new(arrow_array::Float64Array::from(vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                1.5,
            ])),
            Arc::new(arrow_array::Float32Array::from(vec![
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                2.5,
            ])),
        ],
    )
    .expect("batch");
    let bytes = encode(&batch, &o).expect("encode");
    assert_eq!(
        String::from_utf8_lossy(&bytes),
        concat!(
            "{\"d\":\"NaN\",\"f\":\"NaN\"}\n",
            "{\"d\":\"Infinity\",\"f\":\"Infinity\"}\n",
            "{\"d\":\"-Infinity\",\"f\":\"-Infinity\"}\n",
            "{\"d\":1.5,\"f\":2.5}\n"
        )
    );
}

#[test]
fn inference_follows_spark_rules() {
    let data = concat!(
        "{\"long\":1,\"dbl\":1.5,\"str\":\"x\",\"b\":true,\"ts\":\"2020-01-02T03:04:05\",\"n\":null}\n",
        "{\"long\":2,\"dbl\":2,\"str\":3,\"b\":false,\"ts\":\"2021-05-06T07:08:09\",\"n\":null}\n",
        "{\"long\":3.5,\"dbl\":null,\"str\":\"y\",\"b\":1,\"ts\":null,\"n\":null}\n",
    );
    // Longs widen to doubles, numbers and booleans widen to strings,
    // timestamp-like strings stay strings unless `inferTimestamp` is set,
    // null columns stay strings.
    let s = infer(data.as_bytes(), &options(&serde_json::json!({}))).expect("infer");
    assert_eq!(
        schema_string(&s),
        "struct<long:Float64,dbl:Float64,str:Utf8View,b:Utf8View,\
         ts:Utf8View,n:Utf8View>"
    );
    let s = infer(
        data.as_bytes(),
        &options(&serde_json::json!({"inferTimestamp": "true"})),
    )
    .expect("infer");
    assert_eq!(
        schema_string(&s),
        "struct<long:Float64,dbl:Float64,str:Utf8View,b:Utf8View,\
         ts:Timestamp(µs, \"UTC\"),n:Utf8View>"
    );
    // prefersDecimal infers decimals.
    let s = infer(
        data.as_bytes(),
        &options(&serde_json::json!({"prefersDecimal": "true"})),
    )
    .expect("infer");
    let dec = s.field_with_name("long").expect("long");
    assert_eq!(dec.data_type(), &DataType::Decimal128(21, 1));
    // dropFieldIfAllNull drops the null column; field order is by first appearance.
    let s = infer(
        b"{\"b\":1,\"a\":2}\n{\"a\":3,\"c\":4}\n",
        &options(&serde_json::json!({"dropFieldIfAllNull": "true"})),
    )
    .expect("infer");
    assert_eq!(
        s.fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>(),
        ["b", "a", "c"]
    );
    let s = infer(
        "{\"n\":null}\n".as_bytes(),
        &options(&serde_json::json!({"dropFieldIfAllNull": "true"})),
    )
    .expect("infer");
    assert!(s.fields().is_empty());
    // With `inferTimestamp`, timestamp-like strings infer timestamps and
    // other strings stay strings.
    let s = infer(
        "{\"a\":\"2020-01-02T03:04:05\"}\n".as_bytes(),
        &options(&serde_json::json!({"inferTimestamp": "true"})),
    )
    .expect("infer");
    assert!(matches!(
        s.field_with_name("a").expect("a").data_type(),
        DataType::Timestamp(_, _)
    ));
    // An empty file infers an empty schema.
    let s = infer(b"", &options(&serde_json::json!({}))).expect("infer");
    assert!(s.fields().is_empty());
    // Malformed lines shape nothing, nested fields are refused.
    let s = infer(b"nope\n{\"a\":1}\n", &options(&serde_json::json!({}))).expect("infer");
    assert_eq!(s.fields().len(), 1);
    let e = infer(b"{\"a\":{\"b\":1}}\n", &options(&serde_json::json!({}))).expect_err("nested");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // multiLine inference reads the whole file.
    let s = infer(
        b"{\n\"a\": 1,\n\"b\": \"x\"\n}\n{\n\"a\": 2.5\n}\n",
        &options(&serde_json::json!({"multiLine": "true"})),
    )
    .expect("infer");
    assert_eq!(
        s.fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(
        s.field_with_name("a").expect("a").data_type(),
        &DataType::Float64
    );
}

#[test]
fn dropped_records_keep_their_row_index_across_splits() {
    let text = "{\"a\":0}\n{\"a\":oops}\n{\"a\":2}\n{\"a\":bad}\n{\"a\":4}\n";
    let bytes = text.as_bytes();
    let o = options(&serde_json::json!({"mode": "DROPMALFORMED"}));
    let schema = ddl("a int");
    let req = row_indexed(&schema);
    let size = bytes.len() as u64;
    let indexes = |range: ByteRange| -> Vec<(i64, i32)> {
        let got = read_split(bytes, &o, &req, range).expect("read");
        let mut seen = Vec::new();
        for b in &got {
            let a = b.column(0).as_primitive::<Int32Type>();
            let idx = b.column_by_name(ROW_INDEX_COLUMN).expect("row index");
            let idx = idx.as_primitive::<Int64Type>();
            seen.extend((0..b.num_rows()).map(|i| (idx.value(i), a.value(i))));
        }
        seen
    };
    // The index is the record's place in the file, dropped records included.
    let want = vec![(0, 0), (2, 2), (4, 4)];
    assert_eq!(indexes(ByteRange::new(0, size)), want);
    let mut halves = indexes(ByteRange::new(0, 18));
    halves.extend(indexes(ByteRange::new(18, size)));
    halves.sort_unstable();
    assert_eq!(halves, want);
}

#[test]
fn splits_read_each_line_once() {
    let mut text = String::new();
    for i in 0..600 {
        if i % 11 == 0 {
            text.push('\n');
        } else if i % 13 == 0 {
            text.push_str(&format!("{{\"a\":broken{i}}}\n"));
        } else {
            text.push_str(&format!("{{\"a\":{i},\"s\":\"v{i}\"}}\n"));
        }
    }
    let bytes = text.as_bytes();
    let o = options(&serde_json::json!({}));
    let schema = ddl("a int, s string");
    let req = row_indexed(&schema);
    let whole = read(bytes, &o, &req).expect("read");
    let size = bytes.len() as u64;
    for cuts in [
        vec![0, size],
        vec![0, 7, 100, size / 2, size - 3, size],
        vec![0, 1, 2, 3, size],
    ] {
        let mut seen = Vec::new();
        for w in cuts.windows(2) {
            let got = read_split(bytes, &o, &req, ByteRange::new(w[0], w[1])).expect("read");
            for b in &got {
                let a = b.column(0).as_primitive::<Int32Type>();
                let idx = b.column_by_name(ROW_INDEX_COLUMN).expect("row index");
                let idx = idx.as_primitive::<Int64Type>();
                seen.extend(
                    (0..b.num_rows())
                        .filter(|i| !a.is_null(*i))
                        .map(|i| (idx.value(i), a.value(i))),
                );
            }
        }
        seen.sort_unstable();
        let mut want: Vec<(i64, i32)> = rows_of(&whole)
            .iter()
            .enumerate()
            .filter_map(|(n, r)| r[0].as_deref().map(|v| (n as i64, v.parse().expect("int"))))
            .collect();
        want.sort_unstable();
        assert_eq!(seen, want, "cuts {cuts:?}");
    }
    let o = options(&serde_json::json!({"multiLine": "true"}));
    let all = read(bytes, &o, &ScanRequest::all(Arc::clone(&schema))).expect("read");
    assert!(!all.is_empty());
    let later = read_split(
        bytes,
        &o,
        &ScanRequest::all(Arc::clone(&schema)),
        ByteRange::new(3, size),
    )
    .expect("read");
    assert!(later.is_empty());
    assert_eq!(
        JsonFormat.capabilities(&o).splittable,
        crate::format::Splittable::No
    );
    assert_eq!(
        JsonFormat
            .capabilities(&options(&serde_json::json!({})))
            .splittable,
        crate::format::Splittable::ByCodec
    );
}

#[test]
fn multi_line_records_span_lines() {
    let o = options(&serde_json::json!({"multiLine": "true"}));
    let text = "{\n  \"a\": 1,\n  \"b\": \"x\\ty\"\n}\n{\n\"a\": 2\n}\n";
    let schema = ddl("a int, b string");
    let got = read(text.as_bytes(), &o, &ScanRequest::all(Arc::clone(&schema))).expect("read");
    assert_eq!(
        rows_of(&got),
        [
            vec![Some("1".into()), Some("x\ty".into())],
            vec![Some("2".into()), None],
        ]
    );
    // Comments inside multi-line records honour allowComments.
    let o = options(&serde_json::json!({"multiLine": "true", "allowComments": "true"}));
    let got = read(
        "// lead\n{\"a\": /* mid */ 3}\n".as_bytes(),
        &o,
        &ScanRequest::all(Arc::clone(&schema)),
    )
    .expect("read");
    assert_eq!(rows_of(&got)[0][0], Some("3".into()));
    // A truncated tail is one malformed record, not silent loss: PERMISSIVE
    // keeps nulls plus the raw tail in the corrupt column.
    let o = options(&serde_json::json!({"multiLine": "true"}));
    let got = read(
        "{\"a\": 1}\n{\"a\": ".as_bytes(),
        &o,
        &ScanRequest::all(Arc::clone(&schema)),
    )
    .expect("read");
    assert_eq!(rows_of(&got).len(), 2);
    let corrupt = ddl("a int, _corrupt_record string");
    let got = read(
        "{\"a\": 1}\n{\"a\": ".as_bytes(),
        &o,
        &ScanRequest::all(Arc::clone(&corrupt)),
    )
    .expect("read corrupt");
    assert_eq!(
        rows_of(&got),
        [
            vec![Some("1".into()), None],
            vec![None, Some("\n{\"a\": ".into())],
        ]
    );
    let o = options(&serde_json::json!({"multiLine": "true", "mode": "DROPMALFORMED"}));
    let got = read(
        "{\"a\": 1}\n{\"a\": ".as_bytes(),
        &o,
        &ScanRequest::all(Arc::clone(&corrupt)),
    )
    .expect("read drop");
    assert_eq!(rows_of(&got), [vec![Some("1".into()), None]]);
    let o = options(&serde_json::json!({"multiLine": "true", "mode": "FAILFAST"}));
    let e = read(
        "{\"a\": 1}\n{\"a\": ".as_bytes(),
        &o,
        &ScanRequest::all(Arc::clone(&schema)),
    )
    .expect_err("truncated");
    assert_eq!(
        classes_of(&e),
        [
            ErrorClass::FailedReadFileNoHint,
            ErrorClass::MalformedRecordInParsingWithoutSuggestion,
        ]
    );
}

#[test]
fn encodings_and_line_separators() {
    use ErrorClass::KrishivUnsupportedFeature as Unsupported;
    // Latin-1 and ASCII decode; invalid bytes become U+FFFD like Spark.
    let latin = [b'{', b'"', b'a', b'"', b':', b'"', 0xE9, b'"', b'}', b'\n'];
    let schema = ddl("a string");
    for (name, value) in [("encoding", "ISO-8859-1"), ("charset", "LATIN1")] {
        let o = options(&serde_json::json!({name: value}));
        let got = read(&latin, &o, &ScanRequest::all(Arc::clone(&schema))).expect("read");
        assert_eq!(rows_of(&got), [vec![Some("é".into())]]);
    }
    let o = options(&serde_json::json!({"encoding": "US-ASCII"}));
    let got = read(&latin, &o, &ScanRequest::all(Arc::clone(&schema))).expect("read");
    assert_eq!(rows_of(&got), [vec![Some("�".into())]]);
    let o = options(&serde_json::json!({"encoding": "EBCDIC"}));
    let e = read(b"{}\n", &o, &ScanRequest::all(Arc::clone(&schema))).expect_err("charset");
    assert_eq!(e.error_class(), Unsupported);
    let o = options(&serde_json::json!({"lineSep": ";"}));
    let got = read(b"{\"a\":1};{\"a\":2};", &o, &ScanRequest::all(ddl("a int"))).expect("read");
    assert_eq!(
        rows_of(&got),
        [vec![Some("1".into())], vec![Some("2".into())]]
    );
    let e = options(&serde_json::json!({"lineSep": ""}));
    assert!(super::options::JsonOptions::new(&e, false).is_err());
    let batch = RecordBatch::try_new(
        ddl("a int"),
        vec![Arc::new(arrow_array::Int32Array::from(vec![
            Some(1),
            Some(2),
        ]))],
    )
    .expect("batch");
    let bytes = encode(&batch, &o).expect("encode");
    assert_eq!(bytes, b"{\"a\":1};{\"a\":2};");
}

#[test]
fn write_then_read_round_trips() {
    use arrow_array::{
        BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
        Int32Array, StringViewArray, TimestampMicrosecondArray,
    };
    let days = |y: i32, m: u32, d: u32| {
        chrono::NaiveDate::from_ymd_opt(y, m, d)
            .expect("date")
            .signed_duration_since(chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch"))
            .num_days()
            .try_into()
            .expect("days")
    };
    let micros = |y: i32, m: u32, d: u32, h: u32, mi: u32, s: u32, us: u32| {
        chrono::NaiveDate::from_ymd_opt(y, m, d)
            .expect("date")
            .and_hms_micro_opt(h, mi, s, us)
            .expect("time")
            .and_utc()
            .timestamp_micros()
    };
    let utc: Arc<str> = "UTC".into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, true),
        Field::new("d", DataType::Float64, true),
        Field::new("f", DataType::Float32, true),
        Field::new("dec", DataType::Decimal128(10, 2), true),
        Field::new("dt", DataType::Date32, true),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::clone(&utc))),
            true,
        ),
        Field::new(
            "ntz",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("b", DataType::Boolean, true),
        Field::new("bin", DataType::BinaryView, true),
        Field::new("s", DataType::Utf8View, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), None, Some(-7)])),
            Arc::new(Float64Array::from(vec![
                Some(1.5),
                Some(f64::NAN),
                Some(f64::NEG_INFINITY),
            ])),
            Arc::new(Float32Array::from(vec![
                Some(f32::INFINITY),
                Some(-0.0),
                None,
            ])),
            Arc::new(
                Decimal128Array::from(vec![Some(1230), None, Some(-5)])
                    .with_precision_and_scale(10, 2)
                    .expect("decimal"),
            ),
            Arc::new(Date32Array::from(vec![
                Some(days(2020, 1, 2)),
                None,
                Some(days(1999, 12, 31)),
            ])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![
                    Some(micros(2020, 1, 2, 3, 4, 5, 123_456)),
                    None,
                    Some(micros(1970, 1, 1, 0, 0, 0, 0)),
                ])
                .with_timezone(Arc::clone(&utc)),
            ),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(micros(2020, 1, 2, 3, 4, 5, 123_456)),
                None,
                Some(micros(2021, 6, 7, 8, 9, 10, 11)),
            ])),
            Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
            Arc::new(BinaryViewArray::from(vec![
                Some(&[0x00_u8, 0xFF, 0x41][..]),
                None,
                Some(&[][..]),
            ])),
            Arc::new(StringViewArray::from(vec![
                Some("plain"),
                None,
                Some("q\"\\\n\té☃"),
            ])),
        ],
    )
    .expect("batch");
    let o = options(&serde_json::json!({
        "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
        "timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
        "compression": "none",
    }));
    assert_eq!(JsonFormat.file_suffix(&o), ".json");
    let bytes = encode(&batch, &o).expect("encode");
    let r = options(&serde_json::json!({
        "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
        "timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
    }));
    let got = read(&bytes, &r, &ScanRequest::all(Arc::clone(&schema))).expect("read back");
    assert_eq!(rows_of(&got), rows_of(std::slice::from_ref(&batch)));
    // NULL fields are skipped by default (`ignoreNullFields` is true) and
    // write as nulls when it is false.
    assert!(!String::from_utf8_lossy(&bytes).contains("null"));
    let o = options(&serde_json::json!({
        "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
        "timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
        "ignoreNullFields": "false",
    }));
    let bytes = encode(&batch, &o).expect("encode");
    assert!(String::from_utf8_lossy(&bytes).contains("null"));
    let got = read(&bytes, &r, &ScanRequest::all(Arc::clone(&schema))).expect("read back");
    assert_eq!(rows_of(&got), rows_of(std::slice::from_ref(&batch)));
    // Compression needs codecs this build lacks.
    for name in ["gzip", "snappy", "zstd", "bzip2", "lz4"] {
        let w = options(&serde_json::json!({"compression": name}));
        let e = encode(&batch, &w).expect_err("unsupported codec");
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{name}"
        );
    }
    for name in ["a.json.gz", "a.json.zst"] {
        let e = infer_named(b"{}\n", name, &options(&serde_json::json!({}))).expect_err(name);
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{name}"
        );
    }
}

#[test]
fn very_long_lines_read_and_write() {
    let big = "x".repeat(8 << 20);
    let line = format!("{{\"a\":\"{big}\",\"b\":1}}\n");
    let o = options(&serde_json::json!({}));
    let schema = ddl("a string, b int");
    let got = read(line.as_bytes(), &o, &ScanRequest::all(Arc::clone(&schema))).expect("read");
    assert_eq!(rows_of(&got)[0][0].as_deref(), Some(big.as_str()));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::StringViewArray::from(vec![Some(big.as_str())])),
            Arc::new(arrow_array::Int32Array::from(vec![Some(1)])),
        ],
    )
    .expect("batch");
    let bytes = encode(&batch, &o).expect("encode");
    let back = read(&bytes, &o, &ScanRequest::all(Arc::clone(&schema))).expect("read back");
    assert_eq!(rows_of(&back)[0][0].as_deref(), Some(big.as_str()));
}

#[test]
fn truncated_and_garbage_inputs_never_panic() {
    let corpus = [
        "{\"a\":1,\"b\":\"x\",\"c\":true,\"d\":null,\"e\":1.5,\"f\":[1],\"g\":{\"h\":2}}",
        "{\"s\":\"a\\\"b\\u0041\\n\",\"n\":-12.5e-3,\"t\":\"2020-01-01\"}",
        "{unquoted:1,'sq':'v'} // comment",
        "[1,2,{\"a\":[]}]",
    ];
    let option_sets = [
        serde_json::json!({}),
        serde_json::json!({"mode": "DROPMALFORMED"}),
        serde_json::json!({"mode": "FAILFAST"}),
        serde_json::json!({"mode": "FAILFAST", "multiLine": "true"}),
        serde_json::json!({"multiLine": "true", "allowComments": "true"}),
        serde_json::json!({"allowUnquotedFieldNames": "true", "allowSingleQuotes": "true", "allowComments": "true"}),
        serde_json::json!({"allowNumericLeadingZeros": "true", "allowBackslashEscapingAnyCharacter": "true", "allowUnquotedControlChars": "true"}),
        serde_json::json!({"primitivesAsString": "true", "columnNameOfCorruptRecord": "bad"}),
    ];
    let plain = ddl("a int, b string, c boolean, d string, e double, s string, n double");
    for record in corpus {
        // Every truncation parses or fails cleanly.
        for i in 0..=record.len() {
            if !record.is_char_boundary(i) {
                continue;
            }
            let prefix = &record[..i];
            for opts in &option_sets {
                let o = options(opts);
                let syntax = syntax_of(opts);
                let _ = parse_line(prefix, syntax);
                let _ = parse_next(prefix, 0, syntax);
                let _ = read(prefix.as_bytes(), &o, &ScanRequest::all(Arc::clone(&plain)));
                let _ = infer(prefix.as_bytes(), &o);
            }
        }
        // Every single-byte mutation parses or fails cleanly.
        for (i, b) in record.bytes().enumerate() {
            for alt in [b'\"', b'\\', b'{', b'}', b'[', b']', 0, b'\n', b'x'] {
                if alt == b {
                    continue;
                }
                let mut v = record.as_bytes().to_vec();
                v[i] = alt;
                for opts in &option_sets {
                    let o = options(opts);
                    let syntax = syntax_of(opts);
                    let _ = parse_line(std::str::from_utf8(&v).unwrap_or(""), syntax);
                    let _ = read(&v, &o, &ScanRequest::all(Arc::clone(&plain)));
                }
            }
        }
    }
    // Deep nesting is a malformed record, not a stack overflow: the
    // scanner is iterative with `checked_add` depth counting.
    let deep = format!("{{\"a\":{}}}", "[".repeat(10_000) + &"]".repeat(10_000));
    let o = options(&serde_json::json!({}));
    let got = read(
        deep.as_bytes(),
        &o,
        &ScanRequest::all(ddl("a int, _corrupt_record string")),
    )
    .expect("deep permissive");
    assert_eq!(rows_of(&got).len(), 1);
    assert_eq!(rows_of(&got)[0][0], None);
    let o = options(&serde_json::json!({"mode": "DROPMALFORMED"}));
    let got = read(deep.as_bytes(), &o, &ScanRequest::all(ddl("a int"))).expect("deep drop");
    assert!(rows_of(&got).is_empty());
    let top = " [".to_owned() + &"[".repeat(9_999) + &"]".repeat(10_000);
    let o = options(&serde_json::json!({"mode": "FAILFAST"}));
    let e = read(top.as_bytes(), &o, &ScanRequest::all(ddl("a string"))).expect_err("deep top");
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
}

#[test]
fn base64_edge_cases() {
    use super::convert::{decode_base64, encode_base64};
    // Round trips, including empty and unpadded lengths.
    for bytes in [
        &[][..],
        &[0x00][..],
        &[0xFF][..],
        b"qw".as_slice(),
        b"hello world".as_slice(),
    ] {
        let text = encode_base64(bytes);
        assert_eq!(decode_base64(&text).as_deref(), Some(bytes));
    }
    assert_eq!(encode_base64(b""), "");
    assert_eq!(decode_base64(""), Some(Vec::new()));
    assert_eq!(decode_base64("qw==").expect("pad"), vec![0xAB]);
    // Padding only ends the input, at most two chars, length multiple of four.
    for bad in [
        "q", "qw=", "qw===", "=====", "qw=qw==", "qw==qw==", "q=w=", "q===", "q w=", "qw\n=",
        " qw==", "qw== ", "q*==", "q/w", "====",
    ] {
        assert_eq!(decode_base64(bad), None, "{bad:?}");
    }
    // Binary columns reject bad base64 as malformed records.
    let schema = ddl("bin binary, _corrupt_record string");
    let o = options(&serde_json::json!({}));
    let got = read(
        b"{\"bin\":\"qw==\"}\n{\"bin\":\"!!!\"}\n",
        &o,
        &ScanRequest::all(Arc::clone(&schema)),
    )
    .expect("read");
    assert_eq!(
        rows_of(&got),
        [
            vec![Some("qw==".into()), None],
            vec![None, Some("{\"bin\":\"!!!\"}".into())],
        ]
    );
}

#[test]
fn read_path_option_negatives() {
    // `samplingRatio` above 1 is accepted (as in CSV): inference reads all rows.
    let o = options(&serde_json::json!({"samplingRatio": "2.0"}));
    let s = infer(b"{\"a\":1}\n", &o).expect("ratio above 1 reads all rows");
    assert_eq!(s.fields().len(), 1);
    // An invalid `mode` is rejected at option-parse level, like CSV.
    let user = BTreeMap::from([(
        "mode".to_owned(),
        OptionValue::from_user("mode", "WHATEVER"),
    )]);
    let e = JsonFormat
        .options()
        .parse(&user, &Config::new())
        .expect_err("bad mode");
    assert_eq!(e.error_class(), ErrorClass::KrishivInvalidFormatOptionValue);
    // An invalid `encoding` fails the read; an empty `lineSep` fails option parsing.
    let e = read(
        b"{}\n",
        &options(&serde_json::json!({"encoding": "EBCDIC"})),
        &ScanRequest::all(ddl("a int")),
    )
    .expect_err("bad charset");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    let e = super::options::JsonOptions::new(&options(&serde_json::json!({"lineSep": ""})), false)
        .expect_err("empty lineSep");
    assert_eq!(e.error_class(), ErrorClass::KrishivInvalidFormatOptionValue);
}

#[test]
fn bad_option_values_are_typed_errors() {
    use ErrorClass::{
        KrishivInvalidFormatOptionValue as Invalid, KrishivUnsupportedFeature as Unsupported,
    };
    for opts in [
        serde_json::json!({"mode": "WHATEVER"}),
        serde_json::json!({"samplingRatio": "0"}),
        serde_json::json!({"samplingRatio": "-1"}),
        serde_json::json!({"samplingRatio": "x"}),
        serde_json::json!({"samplingRatio": "NaN"}),
        serde_json::json!({"lineSep": ""}),
        serde_json::json!({"inferenceMaxPartitionBytes": "12 parsecs"}),
        serde_json::json!({"inferenceMinPartitionNum": "0"}),
    ] {
        let parsed = JsonFormat.options().parse(
            &opts
                .as_object()
                .expect("object")
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        OptionValue::from_user(k, v.as_str().unwrap_or_default().to_owned()),
                    )
                })
                .collect(),
            &Config::new(),
        );
        let e = match parsed {
            Err(e) => e,
            Ok((o, _)) => match super::options::JsonOptions::new(&o, false) {
                Err(e) => e,
                Ok(_) => infer(b"{}\n", &o).expect_err("want an error"),
            },
        };
        assert_eq!(e.error_class(), Invalid, "{opts}");
    }
    for opts in [
        serde_json::json!({"encoding": "EBCDIC"}),
        serde_json::json!({"compression": "gzip"}),
    ] {
        let o = options(&opts);
        let Err(e) = super::options::JsonOptions::new(&o, true) else {
            panic!("want an error for {opts}")
        };
        assert_eq!(e.error_class(), Unsupported, "{opts}");
    }
}

#[test]
fn time_zones_and_formats() {
    // A fixed-offset session zone shifts instants on read and write.
    let o = options(&serde_json::json!({"timeZone": "+05:00"}));
    let schema = ddl("ts timestamp");
    let got = read(
        b"{\"ts\":\"2020-01-02T03:04:05\"}\n",
        &o,
        &ScanRequest::all(Arc::clone(&schema)),
    )
    .expect("read");
    let micros = got[0]
        .column(0)
        .as_primitive::<TimestampMicrosecondType>()
        .value(0);
    assert_eq!(
        micros,
        chrono::NaiveDate::from_ymd_opt(2020, 1, 1)
            .expect("date")
            .and_hms_opt(22, 4, 5)
            .expect("time")
            .and_utc()
            .timestamp_micros()
    );
    // Explicit patterns govern both directions.
    let o = options(&serde_json::json!({
        "dateFormat": "dd/MM/yyyy",
        "timestampFormat": "dd/MM/yyyy HH:mm:ss",
        "timestampNTZFormat": "dd/MM/yyyy HH:mm:ss",
    }));
    let schema = ddl("d date, ts timestamp, ntz timestamp_ntz");
    let got = read(
        "{\"d\":\"02/01/2020\",\"ts\":\"02/01/2020 03:04:05\",\"ntz\":\"02/01/2020 03:04:05\"}\n"
            .as_bytes(),
        &o,
        &ScanRequest::all(Arc::clone(&schema)),
    )
    .expect("read");
    assert_eq!(
        rows_of(&got)[0][0].as_deref(),
        Some("2020-01-02"),
        "dates write in the default pattern"
    );
    // A region zone fails only timestamp conversions.
    let o = options(&serde_json::json!({"timeZone": "America/New_York"}));
    let ints = ScanRequest::all(ddl("a int"));
    assert_eq!(
        rows_of(&read(b"{\"a\":1}\n", &o, &ints).expect("no timestamps")).len(),
        1
    );
    let e = read(
        b"{\"ts\":\"2020-01-01 10:00:00\"}\n",
        &o,
        &ScanRequest::all(ddl("ts timestamp")),
    )
    .expect_err("zone");
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn options_round_trip_through_the_schema() {
    // Unknown options warn; strict mode refuses them.
    let user = BTreeMap::from([(
        "bogus".to_owned(),
        OptionValue::from_user("bogus", "1".to_owned()),
    )]);
    let (_, warnings) = JsonFormat
        .options()
        .parse(&user, &Config::new())
        .expect("parse");
    assert_eq!(warnings.len(), 1);
    let e = infer(
        b"{}\n",
        &options(&serde_json::json!({"mode": "PERMISSIVE"})),
    );
    assert!(e.is_ok());
    assert_eq!(JsonFormat.name(), "json");
    // Capabilities: line mode splits by codec, multiLine does not.
    assert!(
        JsonFormat
            .capabilities(&options(&serde_json::json!({})))
            .write
    );
    assert!(
        JsonFormat
            .capabilities(&options(&serde_json::json!({})))
            .row_index
    );
}

/// Random text over the characters that matter never panics the tokenizer,
/// the decoder or inference, whatever the options, and every split reads
/// the same rows as the whole file.
#[test]
fn random_inputs_never_panic_and_splits_agree() {
    const ATOMS: &[&str] = &[
        "\"",
        "{",
        "}",
        "[",
        "]",
        ",",
        ":",
        "\n",
        "\\",
        " ",
        "\t",
        "a",
        "1",
        "-1.5",
        "x",
        "e",
        "E",
        ".",
        "+",
        "-",
        "true",
        "false",
        "null",
        "NaN",
        "Infinity",
        "2020-01-01",
        "0",
        "\"\"",
        "''",
        "\\\"",
        "\\u0041",
        "/",
        "*",
    ];
    let option_sets = [
        serde_json::json!({}),
        serde_json::json!({"mode": "DROPMALFORMED"}),
        serde_json::json!({"mode": "FAILFAST"}),
        serde_json::json!({"mode": "FAILFAST", "multiLine": "true"}),
        serde_json::json!({"multiLine": "true", "allowComments": "true"}),
        serde_json::json!({"allowUnquotedFieldNames": "true", "allowSingleQuotes": "true"}),
        serde_json::json!({"allowNumericLeadingZeros": "true", "allowBackslashEscapingAnyCharacter": "true"}),
        serde_json::json!({"primitivesAsString": "true", "allowUnquotedControlChars": "true"}),
    ];
    let schema = ddl("a int, b string, c double, d boolean, _corrupt_record string");
    let mut state = 17_u64;
    let mut next = |n: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from((state >> 33) % n as u64).expect("small")
    };
    for _ in 0..200 {
        let atoms = 1 + next(40);
        let text: String = (0..atoms).map(|_| ATOMS[next(ATOMS.len())]).collect();
        for opts in &option_sets {
            let o = options(opts);
            let _ = infer(text.as_bytes(), &o);
            let req = row_indexed(&schema);
            let whole = read(text.as_bytes(), &o, &req);
            let size = text.len() as u64;
            if size == 0 {
                continue;
            }
            let cut = (1 + next(text.len().max(2) - 1) as u64).min(size.saturating_sub(1));
            if cut == 0 {
                continue;
            }
            let first = read_split(text.as_bytes(), &o, &req, ByteRange::new(0, cut));
            let second = read_split(text.as_bytes(), &o, &req, ByteRange::new(cut, size));
            let splittable =
                JsonFormat.capabilities(&o).splittable == crate::format::Splittable::ByCodec;
            if splittable {
                assert_eq!(
                    whole.is_ok(),
                    first.is_ok() && second.is_ok(),
                    "{opts} cut {cut} of {text:?}"
                );
            }
            if let (Ok(w), Ok(f), Ok(s)) = (&whole, &first, &second)
                && splittable
            {
                {
                    let mut got: Vec<_> = rows_of(f).into_iter().chain(rows_of(s)).collect();
                    let mut want = rows_of(w);
                    got.sort();
                    want.sort();
                    assert_eq!(got, want, "{opts} cut {cut} of {text:?}");
                }
            }
        }
    }
}
