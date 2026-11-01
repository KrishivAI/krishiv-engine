// SPDX-License-Identifier: Apache-2.0
//! The CSV format against records Spark 4.2 tokenized, read and wrote
//! (`tests/fixtures/spark-4.2.0/csv.json`, made by
//! `tests/scripts/gen_csv_goldens.py`), and against the behaviour that
//! needs no Spark (compression, splits, row indexes, inference sampling).

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use bytes::Bytes;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_common::secret::OptionValue;
use krishiv_kernels::text::{java_double_string, java_float_string32};
use serde_json::Value;

use super::CsvFormat;
use super::encode::CsvEncoder;
use super::options::CsvOptions;
use super::tokenizer::Tokenizer;
use crate::format::{Decoder, Encoder, FileFormat, FileMetadata, Step};
use crate::io::{ByteRange, FileRef, IoConfig, IoScheduler, StoreId};
use crate::options::FormatOptions;
use crate::request::{ColumnPath, ROW_INDEX_COLUMN, ScanRequest};
use crate::text::{Datetimes, TextMetadata};

const GOLDEN: &str = include_str!("../../../../tests/fixtures/spark-4.2.0/csv.json");

fn golden() -> Value {
    serde_json::from_str(GOLDEN).expect("the golden file is JSON")
}

fn b64(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            _ => 63,
        }
    };
    let mut out = Vec::new();
    let clean: Vec<u8> = s.bytes().filter(|c| *c != b'=').collect();
    for chunk in clean.chunks(4) {
        let mut n = 0_u32;
        for (i, c) in chunk.iter().enumerate() {
            n |= val(*c) << (18 - 6 * i);
        }
        let bytes = n.to_be_bytes();
        out.extend_from_slice(&bytes[1..chunk.len()]);
    }
    out
}

fn options(map: &Value) -> FormatOptions {
    let mut user = BTreeMap::new();
    if let Some(m) = map.as_object() {
        for (k, v) in m {
            user.insert(
                k.clone(),
                OptionValue::from_user(k, v.as_str().unwrap_or_default().to_owned()),
            );
        }
    }
    CsvFormat
        .options()
        .parse(&user, &Config::new())
        .expect("options parse")
        .0
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
            krishiv_types::arrow::canonical_arrow_type(&DataType::Decimal128(
                p.trim().parse().expect("p"),
                s.trim().parse().expect("s"),
            ))
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

/// Spark's `simpleString` of a type.
fn type_name(dt: &DataType) -> String {
    match dt {
        DataType::Utf8View => "string".into(),
        DataType::Int32 => "int".into(),
        DataType::Int64 => "bigint".into(),
        DataType::Float64 => "double".into(),
        DataType::Boolean => "boolean".into(),
        DataType::Date32 => "date".into(),
        DataType::Timestamp(_, Some(_)) => "timestamp".into(),
        DataType::Decimal64(p, s) | DataType::Decimal128(p, s) => format!("decimal({p},{s})"),
        other => panic!("{other}"),
    }
}

fn schema_string(s: &Schema) -> String {
    let fields: Vec<String> = s
        .fields()
        .iter()
        .map(|f| format!("{}:{}", f.name(), type_name(f.data_type())))
        .collect();
    format!("struct<{}>", fields.join(","))
}

fn plain_decimal(v: i128, scale: i8) -> String {
    let digits = v.unsigned_abs().to_string();
    let scale = usize::try_from(scale).unwrap_or(0);
    let sign = if v < 0 { "-" } else { "" };
    if scale == 0 {
        return format!("{sign}{digits}");
    }
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (int, frac) = padded.split_at(padded.len() - scale);
    format!("{sign}{int}.{frac}")
}

fn timestamp_text(micros: i64) -> String {
    let t = chrono::DateTime::from_timestamp_micros(micros).expect("timestamp");
    let base = t.format("%Y-%m-%d %H:%M:%S").to_string();
    let frac = format!("{:06}", micros.rem_euclid(1_000_000));
    let frac = frac.trim_end_matches('0');
    if frac.is_empty() {
        base
    } else {
        format!("{base}.{frac}")
    }
}

/// Spark's `CAST(value AS STRING)`.
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
        DataType::Decimal64(_, s) => {
            plain_decimal(i128::from(a.as_primitive::<Decimal64Type>().value(i)), *s)
        }
        DataType::Decimal128(_, s) => {
            plain_decimal(a.as_primitive::<Decimal128Type>().value(i), *s)
        }
        DataType::Utf8View => a.as_string_view().value(i).to_owned(),
        DataType::BinaryView => String::from_utf8_lossy(a.as_binary_view().value(i)).into_owned(),
        DataType::Date32 => {
            let days = a.as_primitive::<arrow_array::types::Date32Type>().value(i);
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
    let units = CsvFormat
        .plan_units(m.as_ref(), &split, req)
        .map_err(|e| e.into_error("t.csv"))?;
    let mut d =
        super::decode::CsvDecoder::new(&m, &units, req, o).map_err(|e| e.into_error("t.csv"))?;
    let mut out = Vec::new();
    loop {
        match d.step().map_err(|e| e.into_error("t.csv"))? {
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
                d.push(data).map_err(|e| e.into_error("t.csv"))?;
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

/// The error classes of `e` and its causes, outermost first, as Spark names them.
fn classes(e: &KrishivError) -> Vec<String> {
    std::iter::once(e)
        .chain(e.causes())
        .map(|e| format!("{:?}", e.error_class()))
        .collect()
}

fn camel(class: &str) -> String {
    class
        .split(['.', '_'])
        .map(|w| {
            let mut c = w.chars();
            c.next()
                .map(|f| f.to_string() + &c.as_str().to_lowercase())
                .unwrap_or_default()
        })
        .collect()
}

fn in_memory_file(bytes: &[u8], name: &str) -> (Arc<IoScheduler>, FileRef) {
    let sched = IoScheduler::new(IoConfig::default());
    let store = Arc::new(object_store::memory::InMemory::new());
    let id = StoreId::new("memory://csv");
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
    infer_named(bytes, "f.csv", o)
}

fn infer_named(bytes: &[u8], name: &str, o: &FormatOptions) -> Result<SchemaRef, KrishivError> {
    let (sched, file) = in_memory_file(bytes, name);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .map_err(|e| crate::error::internal(e.to_string()))?;
    let io = sched.context();
    rt.block_on(CsvFormat.infer_schema(&io, std::slice::from_ref(&file), o))
        .map_err(|e| e.into_error("f.csv"))
}

#[test]
fn tokens_match_spark() {
    let g = golden();
    let mut checked = 0;
    let mut divergent = 0;
    for case in g["tokens"].as_array().expect("tokens") {
        let o = CsvOptions::new(&options(&case["options"]), false).expect("options");
        let tok = Tokenizer::new(o.syntax.clone());
        let mut misses = Vec::new();
        for (line, want) in case["lines"]
            .as_array()
            .expect("lines")
            .iter()
            .zip(case["rows"].as_array().expect("rows"))
        {
            let line = line.as_str().expect("line");
            let mut tokens = Vec::new();
            let res = tok.record(line, 0, &mut tokens);
            if want.get("error").is_some() {
                // With `ignoreTrailingWhiteSpace` alone Univocity crashes (a null
                // pointer, or a lost record) on a record that starts with a
                // white-space only value or holds a quoted one: Krishiv reads
                // them like any other (ADR 0023 divergence). Every other
                // error is a record Univocity rejects, and so is it here.
                if case["options"] == serde_json::json!({"ignoreTrailingWhiteSpace": "true"}) {
                    assert!(res.is_ok(), "{line:?} is read");
                    divergent += 1;
                } else {
                    assert!(res.is_err(), "{} {line:?} should fail", case["options"]);
                }
                continue;
            }
            let want: Vec<Option<&str>> = want
                .as_array()
                .expect("row")
                .iter()
                .map(Value::as_str)
                .collect();
            let got: Vec<Option<String>> = (0..8)
                .map(|i| {
                    tokens
                        .get(i)
                        .and_then(Clone::clone)
                        .filter(|t| *t != o.null_value)
                })
                .collect();
            // `from_csv` of an empty string is a row of NULLs, not a bad record.
            let got_corrupt =
                (!line.is_empty() && (res.is_err() || tokens.len() != 8)).then_some(line);
            let got_ref: Vec<Option<&str>> = got.iter().map(Option::as_deref).collect();
            if got_ref[..] != want[..8] || got_corrupt != want[8] {
                misses.push(format!(
                    "{line:?}: got {got_ref:?} {got_corrupt:?}, want {want:?}"
                ));
            }
            checked += 1;
        }
        assert!(
            misses.is_empty(),
            "options {}: {} of {} records differ, first:\n{}",
            case["options"],
            misses.len(),
            case["lines"].as_array().map_or(0, Vec::len),
            misses
                .iter()
                .take(14)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    assert!(checked > 2000);
    assert_eq!(divergent, 14);
}

#[test]
fn reads_match_spark() {
    let g = golden();
    let mut failures = Vec::new();
    for case in g["reads"].as_array().expect("reads") {
        let name = case["name"].as_str().expect("name");
        let bytes = b64(case["file"].as_str().expect("file"));
        let o = options(&case["options"]);
        let schema = match case["schema"].as_str() {
            Some(s) => ddl(s),
            None => match infer(&bytes, &o) {
                Ok(s) => {
                    let want = case["inferred"].as_str().expect("inferred");
                    if schema_string(&s) != want {
                        failures.push(format!(
                            "{name}: inferred {}, want {want}",
                            schema_string(&s)
                        ));
                    }
                    s
                }
                Err(e) => {
                    failures.push(format!("{name}: inference failed: {e}"));
                    continue;
                }
            },
        };
        let projection: Vec<ColumnPath> = match case["select"].as_array() {
            Some(sel) => sel
                .iter()
                .map(|c| ColumnPath::new(c.as_str().expect("col")))
                .collect(),
            None => schema
                .fields()
                .iter()
                .map(|f| ColumnPath::new(f.name()))
                .collect(),
        };
        let req = ScanRequest::new(Arc::clone(&schema), projection);
        let got = read(&bytes, &o, &req);
        match (got, case.get("error"), case.get("count"), case.get("rows")) {
            (Ok(b), None, Some(count), _) => {
                let n: usize = b.iter().map(RecordBatch::num_rows).sum();
                if Some(n as u64) != count.as_u64() {
                    failures.push(format!("{name}: {n} rows, want {count}"));
                }
            }
            (Ok(b), None, None, Some(rows)) => {
                let want: Vec<Vec<Option<String>>> = rows
                    .as_array()
                    .expect("rows")
                    .iter()
                    .map(|r| {
                        r.as_array()
                            .expect("row")
                            .iter()
                            .map(|v| v.as_str().map(str::to_owned))
                            .collect()
                    })
                    .collect();
                let got = rows_of(&b);
                if got != want {
                    failures.push(format!("{name}: got {got:?}, want {want:?}"));
                }
            }
            (Err(e), Some(want), ..) => {
                let want: Vec<String> = want
                    .as_array()
                    .expect("classes")
                    .iter()
                    .map(|c| camel(c.as_str().expect("class")))
                    .collect();
                // Classes the error carries must be Spark's, outermost first.
                let got = classes(&e);
                if got != want {
                    failures.push(format!("{name}: error {got:?}, want {want:?}"));
                }
            }
            (got, want, ..) => failures.push(format!(
                "{name}: got {:?}, want {want:?}",
                got.map(|b| rows_of(&b))
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} reads differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn strings_batch() -> RecordBatch {
    let ints: Vec<Option<i32>> = (1..=18).map(Some).collect();
    let strings = [
        Some("plain"),
        Some("a,b"),
        Some("q\"uote"),
        Some("line\nbreak"),
        Some(""),
        None,
        Some(" lead"),
        Some("trail "),
        Some("back\\slash"),
        Some("#hash"),
        Some("a\rb"),
        Some("back\\slash,comma"),
        Some(" \t"),
        Some("\"lead"),
        Some("\\\""),
        Some("é"),
        Some("x;y"),
        Some("NA"),
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int32, true),
        Field::new("s", DataType::Utf8View, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int32Array::from(ints)),
            Arc::new(arrow_array::StringViewArray::from(strings.to_vec())),
        ],
    )
    .expect("batch")
}

fn types_batch() -> RecordBatch {
    use arrow_array::{
        BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
        Int8Array, Int16Array, Int32Array, Int64Array, TimestampMicrosecondArray,
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
    let fields = vec![
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
        Field::new("tiny", DataType::Int8, true),
        Field::new("small", DataType::Int16, true),
        Field::new("big", DataType::Int64, true),
        Field::new("wide", DataType::Decimal128(38, 10), true),
    ];
    let ts = vec![
        Some(micros(2020, 1, 2, 3, 4, 5, 123_456)),
        Some(micros(2020, 1, 2, 3, 4, 5, 0)),
        Some(micros(1, 1, 1, 0, 0, 0, 0)),
        None,
        None,
        None,
        None,
    ];
    let ntz = vec![
        Some(micros(2020, 1, 2, 3, 4, 5, 123_456)),
        Some(micros(2020, 1, 2, 3, 4, 5, 0)),
        Some(micros(1, 1, 1, 0, 0, 0, 100_000)),
        None,
        None,
        None,
        None,
    ];
    let cols: Vec<arrow_array::ArrayRef> = vec![
        Arc::new(Int32Array::from((1..=7).collect::<Vec<_>>())),
        Arc::new(Float64Array::from(vec![
            Some(1.0),
            Some(1e7),
            Some(1e-4),
            Some(f64::NEG_INFINITY),
            Some(f64::NAN),
            Some(-0.0),
            Some(123_456_789.125),
        ])),
        Arc::new(Float32Array::from(vec![
            Some(1.5),
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(0.0),
            Some(1e10),
            Some(0.001),
            Some(3.4e38),
        ])),
        Arc::new(
            Decimal128Array::from(vec![Some(1230), Some(-50), None, None, None, None, None])
                .with_precision_and_scale(10, 2)
                .expect("decimal"),
        ),
        Arc::new(Date32Array::from(vec![
            Some(days(2020, 1, 2)),
            Some(days(1, 1, 1)),
            Some(days(9999, 12, 31)),
            None,
            None,
            None,
            None,
        ])),
        Arc::new(TimestampMicrosecondArray::from(ts).with_timezone(utc)),
        Arc::new(TimestampMicrosecondArray::from(ntz)),
        Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            None,
            None,
            None,
            None,
        ])),
        Arc::new(BinaryViewArray::from(vec![
            Some(&[0xAB_u8, 0x01, 0xFF][..]),
            Some(&[][..]),
            None,
            None,
            None,
            None,
            None,
        ])),
        Arc::new(Int8Array::from(vec![
            Some(1),
            None,
            None,
            None,
            None,
            None,
            None,
        ])),
        Arc::new(Int16Array::from(vec![
            Some(2),
            None,
            None,
            None,
            None,
            None,
            None,
        ])),
        Arc::new(Int64Array::from(vec![
            Some(3),
            None,
            None,
            None,
            None,
            None,
            None,
        ])),
        Arc::new(
            Decimal128Array::from(vec![Some(1), Some(0), None, Some(-1000), None, None, None])
                .with_precision_and_scale(38, 10)
                .expect("decimal"),
        ),
    ];
    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).expect("batch")
}

fn encode(batch: &RecordBatch, o: &FormatOptions) -> Result<Vec<u8>, KrishivError> {
    let mut e = CsvEncoder::new(&batch.schema(), o).map_err(|e| e.into_error("t.csv"))?;
    e.write(batch).map_err(|e| e.into_error("t.csv"))?;
    let mut out = Vec::new();
    while let Some(b) = e.take_bytes() {
        out.extend_from_slice(&b);
    }
    let done = Box::new(e).finish().map_err(|e| e.into_error("t.csv"))?;
    out.extend_from_slice(&done.tail);
    Ok(out)
}

#[test]
fn writes_match_spark() {
    let g = golden();
    let mut failures = Vec::new();
    for case in g["writes"].as_array().expect("writes") {
        let name = case["name"].as_str().expect("name");
        let o = options(&case["options"]);
        let batch = match case["schema"].as_str() {
            Some("struct<i:int,s:string>") => strings_batch(),
            Some(s) if s.starts_with("struct<i:int,d:double") => types_batch(),
            Some("struct<a:int,b:string>") => RecordBatch::new_empty(Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Utf8View, true),
            ]))),
            Some("struct<a:int>") => {
                RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
                    "a",
                    DataType::Int32,
                    true,
                )])))
            }
            Some(s) if s.starts_with("struct<a,b:int") => {
                // The header names case: columns a,b / q" / " sp ".
                let schema = Arc::new(Schema::new(vec![
                    Field::new("a,b", DataType::Int32, true),
                    Field::new("q\"", DataType::Int32, true),
                    Field::new(" sp ", DataType::Int32, true),
                ]));
                RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(arrow_array::Int32Array::from(vec![1])),
                        Arc::new(arrow_array::Int32Array::from(vec![2])),
                        Arc::new(arrow_array::Int32Array::from(vec![3])),
                    ],
                )
                .expect("batch")
            }
            other => panic!("unknown schema {other:?}"),
        };
        let want = b64(case["content"].as_str().expect("content"));
        let got = encode(&batch, &o).expect("encode");
        if got != want {
            failures.push(format!(
                "{name}:\n got  {:?}\n want {:?}",
                String::from_utf8_lossy(&got),
                String::from_utf8_lossy(&want)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} writes differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

const SAMPLING: &str = include_str!("../../../../tests/fixtures/spark-4.2.0/csv-sampling.json");

/// `samplingRatio` picks the lines Spark picks: one draw per line from the
/// partition's `XORShiftRandom`, seeded with the partition's index (the
/// Spark that produced the fixture ran on 2 cores, so its leaf parallelism was 2).
#[test]
fn sampling_ratio_matches_spark() {
    let g: Value = serde_json::from_str(SAMPLING).expect("json");
    let rows = usize::try_from(g["rows"].as_u64().expect("rows")).expect("rows");
    let cols = usize::try_from(g["cols"].as_u64().expect("cols")).expect("cols");
    let mut file = String::new();
    for r in 0..rows {
        let line: Vec<&str> = (0..cols).map(|c| if r == c { "x" } else { "1" }).collect();
        file.push_str(&line.join(","));
        file.push('\n');
    }
    for case in g["cases"].as_array().expect("cases") {
        let mut opts = serde_json::json!({
            "inferSchema": "true",
            "samplingRatio": case["ratio"].as_f64().expect("ratio").to_string(),
            "inferenceParallelism": "2",
        });
        if let Some(b) = case["maxPartitionBytes"].as_u64() {
            opts["inferenceMaxPartitionBytes"] = b.to_string().into();
        }
        if let Some(c) = case["openCostInBytes"].as_u64() {
            opts["inferenceOpenCostInBytes"] = c.to_string().into();
        }
        let schema = infer(file.as_bytes(), &options(&opts)).expect("infer");
        let got: Vec<String> = schema
            .fields()
            .iter()
            .map(|f| type_name(f.data_type()))
            .collect();
        let want: Vec<&str> = case["types"]
            .as_array()
            .expect("types")
            .iter()
            .map(|t| t.as_str().expect("type"))
            .collect();
        assert_eq!(got, want, "samplingRatio case {case}");
    }
}

fn int_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8View, true),
    ]))
}

fn row_indexed(schema: &SchemaRef) -> ScanRequest {
    let mut req = ScanRequest::all(Arc::clone(schema));
    req.metadata_cols.row_index = true;
    req
}

/// A split that starts after byte 0 numbers its rows from the rows before
/// it: any split boundaries read every row once with its index, headers,
/// comments, empty lines and dropped records included.
#[test]
fn row_indexes_continue_across_splits() {
    let mut text = String::from("a,b\n");
    let mut r = 1_u64;
    let mut want = Vec::new();
    for i in 0..400 {
        r = r
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        match (r >> 40) % 7 {
            0 => text.push_str("# a comment\n"),
            1 => text.push('\n'),
            2 => text.push_str(&format!("bad{i},x\n")),
            _ => {
                text.push_str(&format!("{i},v{i}\r\n"));
                want.push(i);
            }
        }
    }
    let o =
        options(&serde_json::json!({"header": "true", "mode": "DROPMALFORMED", "comment": "#"}));
    let schema = int_schema();
    let req = row_indexed(&schema);
    let size = text.len() as u64;
    for cuts in [
        vec![0, size],
        vec![0, 7, 100, size / 2, size - 3, size],
        vec![0, 1, 2, size],
    ] {
        let mut seen = Vec::new();
        for w in cuts.windows(2) {
            let got =
                read_split(text.as_bytes(), &o, &req, ByteRange::new(w[0], w[1])).expect("read");
            for b in &got {
                let a = b.column(0).as_primitive::<Int32Type>();
                let idx = b.column_by_name(ROW_INDEX_COLUMN).expect("row index");
                let idx = idx.as_primitive::<Int64Type>();
                seen.extend((0..b.num_rows()).map(|i| (idx.value(i), a.value(i))));
            }
        }
        seen.sort_unstable();
        let expected: Vec<(i64, i32)> = want
            .iter()
            .enumerate()
            .map(|(n, v)| (n as i64, *v))
            .collect();
        assert_eq!(seen, expected, "cuts {cuts:?}");
    }
}

#[test]
fn multi_line_files_belong_to_the_first_split() {
    let o = options(&serde_json::json!({"multiLine": "true"}));
    let text = b"1,\"a\nb\"\n2,c\n";
    let req = ScanRequest::all(int_schema());
    let all = read(text, &o, &req).expect("read");
    assert_eq!(
        rows_of(&all),
        [
            vec![Some("1".into()), Some("a\nb".into())],
            vec![Some("2".into()), Some("c".into())]
        ]
    );
    let later = read_split(text, &o, &req, ByteRange::new(3, 14)).expect("read");
    assert!(later.is_empty());
    assert_eq!(
        CsvFormat.capabilities(&o).splittable,
        crate::format::Splittable::No
    );
}

#[test]
fn written_files_read_back_and_codecs_are_refused() {
    let w = options(&serde_json::json!({"header": "true", "compression": "none"}));
    assert_eq!(CsvFormat.file_suffix(&w), ".csv");
    let bytes = encode(&strings_batch(), &w).expect("encode");
    let r =
        options(&serde_json::json!({"header": "true", "nullValue": "\u{1}", "multiLine": "true"}));
    let got = read(&bytes, &r, &ScanRequest::all(ddl("i int, s string"))).expect("read back");
    let rows = rows_of(&got);
    assert_eq!(rows.len(), 18);
    assert_eq!(rows[3][1].as_deref(), Some("line\nbreak"));
    // A multi-line reader turns every line end inside quotes into `\n`.
    assert_eq!(rows[10][1].as_deref(), Some("a\nb"));
    assert_eq!(rows[1][1].as_deref(), Some("a,b"));
    // Compression needs codecs this build lacks: a typed error, not a wrong file.
    let unsupported = krishiv_common::error::ErrorClass::KrishivUnsupportedFeature;
    for name in ["gzip", "deflate", "snappy", "zstd", "bzip2", "lz4"] {
        let w = options(&serde_json::json!({"compression": name}));
        let e = encode(&strings_batch(), &w).expect_err("unsupported codec");
        assert_eq!(e.error_class(), unsupported, "{name}");
    }
    for name in ["a.csv.gz", "a.csv.deflate", "a.csv.bz2", "a.csv.zst"] {
        let e = infer_named(b"a,b\n", name, &options(&serde_json::json!({}))).expect_err(name);
        assert_eq!(e.error_class(), unsupported, "{name}");
    }
}

#[test]
fn bad_option_values_are_typed_errors() {
    use krishiv_common::error::ErrorClass::{
        KrishivInvalidFormatOptionValue as Invalid, KrishivUnsupportedFeature as Unsupported,
    };
    let invalid = [
        serde_json::json!({"sep": ""}),
        serde_json::json!({"quote": "ab"}),
        serde_json::json!({"escape": "ab"}),
        serde_json::json!({"comment": "ab"}),
        serde_json::json!({"lineSep": ""}),
        serde_json::json!({"sep": "\\q"}),
        serde_json::json!({"samplingRatio": "0"}),
        serde_json::json!({"samplingRatio": "x"}),
        serde_json::json!({"inferenceMaxPartitionBytes": "12 parsecs"}),
        serde_json::json!({"inferenceMinPartitionNum": "0"}),
    ];
    let unsupported = [
        serde_json::json!({"encoding": "UTF-16"}),
        serde_json::json!({"locale": "de-DE"}),
    ];
    for (opts, class) in invalid
        .iter()
        .map(|o| (o, Invalid))
        .chain(unsupported.iter().map(|o| (o, Unsupported)))
    {
        let o = options(opts);
        let parse = Datetimes::new(&o, krishiv_kernels::datetime_pattern::Purpose::Parse).err();
        let e = CsvOptions::new(&o, false)
            .err()
            .or(parse)
            .or_else(|| infer(b"1\n", &o).err());
        assert_eq!(e.expect("error").error_class(), class, "{opts}");
    }
    let user = BTreeMap::from([(
        "mode".to_owned(),
        OptionValue::from_user("mode", "WHATEVER".to_owned()),
    )]);
    let e = CsvFormat
        .options()
        .parse(&user, &Config::new())
        .expect_err("mode");
    assert_eq!(e.error_class(), Invalid);
}

#[test]
fn spark_written_files_are_read_back() {
    // Spark wrote these files (`tests/scripts/gen_csv_goldens.py`): the values
    // it wrote come back, with the formats it wrote them in.
    let g = golden();
    let batch = types_batch();
    let want = rows_of(std::slice::from_ref(&batch));
    for name in ["types default", "types formats"] {
        let case = g["writes"]
            .as_array()
            .expect("writes")
            .iter()
            .find(|w| w["name"] == name)
            .expect("case");
        let bytes = b64(case["content"].as_str().expect("content"));
        let mut opts = case["options"].clone();
        opts["timestampFormat"] = case["options"]
            .get("timestampFormat")
            .cloned()
            .unwrap_or_else(|| "yyyy-MM-dd'T'HH:mm:ss.SSSXXX".into());
        let o = options(&opts);
        let mut schema = (*batch.schema()).clone();
        // `bin` is shown as `[AB 01 FF]` by Spark: it reads back as that text.
        schema = Schema::new(
            schema
                .fields()
                .iter()
                .map(|f| {
                    if f.name() == "bin" {
                        Field::new("bin", DataType::Utf8View, true)
                    } else {
                        f.as_ref().clone()
                    }
                })
                .collect::<Vec<_>>(),
        );
        let got = read(&bytes, &o, &ScanRequest::all(Arc::new(schema))).expect("read");
        let got = rows_of(&got);
        assert_eq!(got.len(), want.len(), "{name}");
        for (g, w) in got.iter().zip(&want) {
            for (col, (gv, wv)) in g.iter().zip(w).enumerate() {
                let field = batch.schema().field(col).name().clone();
                // Spark writes timestamps with milliseconds by default and shows binary in hex.
                let lossy = match name {
                    "types default" => field == "ts" || field == "ntz",
                    _ => field == "dt",
                };
                if lossy || field == "bin" {
                    continue;
                }
                assert_eq!(gv, wv, "{name} column {field}");
            }
        }
    }
}

/// Random text over the characters that matter never panics the tokenizer,
/// the decoder or inference, whatever the options, and every split reads
/// the same rows as the whole file.
#[test]
fn random_inputs_never_panic_and_splits_agree() {
    const ATOMS: &[&str] = &[
        "\"",
        "\"",
        ",",
        ",",
        "\n",
        "\r\n",
        "\r",
        "\\",
        " ",
        "\t",
        "a",
        "1",
        "-1.5",
        "x",
        "#",
        "||",
        ";",
        "é",
        "\u{0}",
        "true",
        "2020-01-01",
        "NA",
        "''",
        "\"\"",
        "\\\"",
    ];
    let option_sets = [
        serde_json::json!({}),
        serde_json::json!({"header": "true", "mode": "DROPMALFORMED"}),
        serde_json::json!({"mode": "FAILFAST", "multiLine": "true"}),
        serde_json::json!({"unescapedQuoteHandling": "BACK_TO_DELIMITER", "sep": "||"}),
        serde_json::json!({"unescapedQuoteHandling": "STOP_AT_CLOSING_QUOTE", "escape": "\""}),
        serde_json::json!({"unescapedQuoteHandling": "SKIP_VALUE", "comment": "#", "ignoreLeadingWhiteSpace": "true"}),
        serde_json::json!({"charToEscapeQuoteEscaping": "#", "quote": "'", "maxColumns": "3", "maxCharsPerColumn": "6"}),
        serde_json::json!({"escape": "", "quote": "", "lineSep": ";", "multiLine": "true"}),
    ];
    let schema = ddl("a int, b string, c double, d date, e timestamp, _corrupt_record string");
    let mut state = 17_u64;
    let mut next = |n: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from((state >> 33) % n as u64).expect("small")
    };
    for _ in 0..400 {
        let atoms = 1 + next(40);
        let text: String = (0..atoms).map(|_| ATOMS[next(ATOMS.len())]).collect();
        for opts in &option_sets {
            let o = options(opts);
            let _ = infer(text.as_bytes(), &o);
            let req = row_indexed(&schema);
            let whole = read(text.as_bytes(), &o, &req);
            let size = text.len() as u64;
            let cut = (1 + next(text.len().max(2) - 1) as u64).min(size.saturating_sub(1));
            if cut == 0 {
                continue;
            }
            let first = read_split(text.as_bytes(), &o, &req, ByteRange::new(0, cut));
            let second = read_split(text.as_bytes(), &o, &req, ByteRange::new(cut, size));
            // Splits are only defined for files that can be split.
            let splittable =
                CsvFormat.capabilities(&o).splittable == crate::format::Splittable::ByCodec;
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

#[test]
fn every_arrow_representation_reads_and_nested_types_are_refused() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, true),
        Field::new("b", DataType::LargeUtf8, true),
        Field::new("c", DataType::Binary, true),
        Field::new("d", DataType::BinaryView, true),
        Field::new("e", DataType::LargeBinary, true),
        Field::new("f", DataType::Null, true),
        Field::new("g", DataType::Int8, true),
        Field::new("h", DataType::Int16, true),
        Field::new("i", DataType::Float32, true),
        Field::new("j", DataType::Float64, true),
        Field::new("k", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("l", DataType::Date32, true),
    ]));
    let o = options(&serde_json::json!({}));
    let req = ScanRequest::all(Arc::clone(&schema));
    let file = "x,y,z,w,v,n,1,2,Inf,\"1,000.5\",2020-01-01 10:00:00,2020-01-02\n\
                ,,,,,,300,x,1,,,\n\
                x,y,z,w,v,n,-1,-2,-5x,1e+3z,x,x\n\
                x,y,z,w,v,n,2,3,\"1,5\",1e+,,\n";
    let got = read(file.as_bytes(), &o, &req).expect("read");
    let b = &got[0];
    assert_eq!(b.num_rows(), 4);
    assert_eq!(b.column(0).as_string::<i32>().value(0), "x");
    assert_eq!(b.column(1).as_string::<i64>().value(0), "y");
    assert_eq!(b.column(2).as_binary::<i32>().value(0), b"z");
    assert_eq!(b.column(3).as_binary_view().value(0), b"w");
    assert_eq!(b.column(4).as_binary::<i64>().value(0), b"v");
    assert_eq!(b.column(5).logical_null_count(), 4);
    assert_eq!(b.column(6).as_primitive::<Int8Type>().value(0), 1);
    assert_eq!(b.column(7).as_primitive::<Int16Type>().value(0), 2);
    assert_eq!(
        b.column(8).as_primitive::<Float32Type>().value(0),
        f32::INFINITY
    );
    assert!(b.column(9).is_null(0));
    // `300` does not fit a byte: the record is malformed, its value NULL.
    assert!(b.column(6).is_null(1) && b.column(7).is_null(1));
    // A number with trailing text is no number (Spark 4.2 has no lenient
    // number-format fallback: `5x`, `1,5` and `1e+3z` read as NULL).
    assert!(b.column(8).is_null(2) && b.column(8).is_null(3));
    assert!(b.column(9).is_null(2) && b.column(9).is_null(3));
    // Nested types have no CSV representation.
    let nested = Arc::new(Schema::new(vec![Field::new(
        "n",
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        true,
    )]));
    let e = read(b"1\n", &o, &ScanRequest::all(Arc::clone(&nested))).expect_err("nested");
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature
    );
    let batch = RecordBatch::new_empty(nested);
    let e = encode(&batch, &o).expect_err("nested");
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::UnsupportedDataTypeForDatasource
    );
}

#[test]
fn a_region_time_zone_fails_only_timestamp_conversions() {
    let o = options(&serde_json::json!({"timeZone": "America/New_York"}));
    let ints = ScanRequest::all(int_schema());
    assert_eq!(
        rows_of(&read(b"1,x\n", &o, &ints).expect("no timestamps")).len(),
        1
    );
    let ts = ddl("a int, t timestamp");
    let e = read(b"1,2020-01-01 10:00:00\n", &o, &ScanRequest::all(ts)).expect_err("zone");
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature
    );
    let e = encode(&types_batch(), &o).expect_err("zone");
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature
    );
    assert!(encode(&strings_batch(), &o).is_ok());
}

#[test]
fn failfast_and_tokenizer_errors_carry_spark_classes() {
    use krishiv_common::error::ErrorClass::{
        FailedReadFileNoHint, MalformedCsvRecord, MalformedRecordInParsingWithoutSuggestion,
    };
    let req = ScanRequest::all(int_schema());
    // A short record under FAILFAST: the read fails, caused by the parse
    // error, caused by the malformed record.
    let o = options(&serde_json::json!({"mode": "FAILFAST"}));
    let e = read(b"1,x\n2\n", &o, &req).expect_err("short record");
    assert_eq!(
        classes_of(&e),
        [
            FailedReadFileNoHint,
            MalformedRecordInParsingWithoutSuggestion,
            MalformedCsvRecord
        ]
    );
    // A value that is no number has no malformed-record cause.
    let e = read(b"x,y\n", &o, &req).expect_err("bad value");
    assert_eq!(
        classes_of(&e),
        [
            FailedReadFileNoHint,
            MalformedRecordInParsingWithoutSuggestion
        ]
    );
    // Too many columns fails in every mode.
    let o = options(&serde_json::json!({"maxColumns": "1"}));
    let e = read(b"1,x\n", &o, &req).expect_err("columns");
    assert_eq!(classes_of(&e), [FailedReadFileNoHint, MalformedCsvRecord]);
}

fn classes_of(e: &KrishivError) -> Vec<krishiv_common::error::ErrorClass> {
    std::iter::once(e)
        .chain(e.causes())
        .map(KrishivError::error_class)
        .collect()
}

/// `schema_of_csv`'s DDL over literals, through the same inference file
/// reads use (cases from Spark 4.2's `csv-functions.sql` golden and the
/// function matrix, plus the `|` delimiter the golden passes as options;
/// the empty literal and the over-wide record are checked against real
/// Spark 4.2.0 runs, which fail both during constant folding).
#[test]
fn schema_of_csv_literals() {
    use super::infer::{ddl_of_string, ddl_type};
    let plain = options(&serde_json::json!({}));
    for (csv, want) in [
        ("1", "STRUCT<_c0: INT>"),
        ("1,abc", "STRUCT<_c0: INT, _c1: STRING>"),
        ("abc", "STRUCT<_c0: STRING>"),
        (" \t a\u{3000}\n ", "STRUCT<_c0: STRING>"),
        ("\n", "STRUCT<_c0: STRING>"),
        ("14:30:45", "STRUCT<_c0: TIMESTAMP>"),
        ("14:30:45.123456", "STRUCT<_c0: TIMESTAMP>"),
        ("true", "STRUCT<_c0: BOOLEAN>"),
        ("2024-02-29", "STRUCT<_c0: DATE>"),
        ("1.5", "STRUCT<_c0: DOUBLE>"),
        ("123456789012", "STRUCT<_c0: BIGINT>"),
        ("45678901234567890123", "STRUCT<_c0: DECIMAL(20,0)>"),
        ("\"a\nb\",c", "STRUCT<_c0: STRING, _c1: STRING>"),
    ] {
        assert_eq!(ddl_of_string(csv, &plain).unwrap(), want, "{csv:?}");
    }
    // An empty literal parses to no record: Spark's evaluator asserts the
    // record is not null, so the query fails with `INTERNAL_ERROR`; so
    // does this one.
    let err = ddl_of_string("", &plain).unwrap_err();
    match &err {
        crate::FormatError::Other(e) => assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::InternalError,
            "{err}"
        ),
        other => panic!("{other}"),
    }
    // Options are the CSV reader's: a pipe delimiter (the golden's
    // `map('delimiter', '|')`, which SQL cannot pass yet).
    let pipes = options(&serde_json::json!({"delimiter": "|"}));
    assert_eq!(
        ddl_of_string("1|abc", &pipes).unwrap(),
        "STRUCT<_c0: INT, _c1: STRING>"
    );
    // A bad option value is rejected, not ignored.
    let bad_sep = options(&serde_json::json!({"sep": ""}));
    let err = ddl_of_string("1", &bad_sep).unwrap_err();
    match &err {
        crate::FormatError::Other(e) => assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::KrishivInvalidFormatOptionValue,
            "{err}"
        ),
        other => panic!("{other}"),
    }
    // A record past `maxColumns` is an internal error: Spark 4.2.0's own
    // parser throws `TextParsingException` there (past its 20480 columns),
    // failing the query during constant folding just as this one fails.
    let huge = "a,".repeat(20_481);
    let err = ddl_of_string(&huge, &plain).unwrap_err();
    match &err {
        crate::FormatError::Other(e) => assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::InternalError,
            "{err}"
        ),
        other => panic!("{other}"),
    }
    // Every inferred column type has a DDL name.
    for (t, want) in [
        (DataType::Int32, "INT"),
        (DataType::Int64, "BIGINT"),
        (DataType::Decimal128(20, 0), "DECIMAL(20,0)"),
        (DataType::Float64, "DOUBLE"),
        (DataType::Date32, "DATE"),
        (
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            "TIMESTAMP",
        ),
        (DataType::Boolean, "BOOLEAN"),
        (DataType::Utf8View, "STRING"),
    ] {
        assert_eq!(ddl_type(&t).unwrap(), want);
    }
    assert!(ddl_type(&DataType::Int8).is_err());
}
