// SPDX-License-Identifier: Apache-2.0
//! The JSON reader against real Spark 4.2.0 runs: nested types
//! (`tests/fixtures/spark-4.2.0/json-nested.json`, made by
//! `tests/scripts/gen_json_nested_goldens.py`) and the flat coercions
//! (`tests/fixtures/spark-4.2.0/json.json`, `tests/scripts/gen_json_goldens.py`).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType,
};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Schema, SchemaRef};
use krishiv_common::error::KrishivError;
use krishiv_kernels::text::java_double_string;
use serde_json::{Map, Value};

use super::tests::{classes_of, infer, options, read};
use crate::options::FormatOptions;
use crate::request::ScanRequest;

const NESTED: &str = include_str!("../../../../tests/fixtures/spark-4.2.0/json-nested.json");
const FLAT: &str = include_str!("../../../../tests/fixtures/spark-4.2.0/json.json");

/// Cases where Krishiv knowingly differs from the recorded Spark run: the
/// reason, and what Krishiv gives instead (asserted, so a change in the
/// difference fails the test).
const KNOWN: &[(&str, &str)] = &[
    (
        "depth 1000",
        "Krishiv types nest at most 32 deep (DataType::MAX_NESTING_DEPTH), so inference refuses it \
         with FAILED_TO_PARSE_TOO_COMPLEX; Spark's error here is PySpark failing to parse the \
         1000-deep schema",
    ),
    (
        "depth 1001",
        "the inferred schema matches (`_corrupt_record` alone); Spark then refuses a query of only \
         that column in its analysis (QUERY_ONLY_CORRUPT_RECORD_COLUMN), not in the reader",
    ),
    (
        "depth 1001 read",
        "past Jackson's depth limit Spark keeps parsing after the error and fills `a` with a stray \
         `]`; Krishiv makes the whole record malformed, as for the deeper `deep nesting` case",
    ),
];

/// What Krishiv gives for a [`KNOWN`] case, or `None` for any other case.
fn known(name: &str, case: &Value) -> Option<Value> {
    KNOWN.iter().find(|(n, _)| *n == name)?;
    Some(match name {
        "depth 1000" => serde_json::json!({ "error": ["FAILED_TO_PARSE_TOO_COMPLEX"] }),
        "depth 1001" => serde_json::json!({ "inferred": "struct<_corrupt_record:string>" }),
        _ => {
            let text = String::from_utf8(file_of(case)).expect("utf-8");
            serde_json::json!({
                "inferred": case["inferred"],
                "rows": [[null, null, text.trim_end()]],
            })
        }
    })
}

fn file_of(case: &Value) -> Vec<u8> {
    let b64 = case["file"].as_str().expect("file");
    super::convert::decode_base64(b64).expect("base64")
}

fn options_of(case: &Value) -> FormatOptions {
    options(&case["options"])
}

/// A DDL table schema (`a int, s struct<...>`) as Arrow.
fn schema_of(ddl: &str) -> SchemaRef {
    let t = krishiv_types::parse_data_type(&format!("struct<{ddl}>")).expect("ddl");
    let DataType::Struct(fields) = t.to_arrow() else {
        panic!("struct");
    };
    Arc::new(Schema::new(fields))
}

/// Spark's `simpleString` of a schema.
fn simple_string(s: &Schema) -> String {
    krishiv_types::DataType::from_arrow(&DataType::Struct(s.fields().clone()))
        .expect("spark type")
        .simple_string()
}

/// The error classes of `e`, outermost first, as the generator records them.
fn class_names(e: &KrishivError) -> Vec<String> {
    classes_of(e)
        .into_iter()
        .map(|c| c.name().to_owned())
        .collect()
}

/// The value at `i` as the generator's `plain()` writes it.
fn plain(a: &dyn Array, i: usize) -> Value {
    if a.is_null(i) {
        return Value::Null;
    }
    match a.data_type() {
        DataType::Boolean => Value::Bool(a.as_boolean().value(i)),
        DataType::Int8 => a.as_primitive::<Int8Type>().value(i).into(),
        DataType::Int16 => a.as_primitive::<Int16Type>().value(i).into(),
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).into(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).into(),
        DataType::Float32 => float(f64::from(a.as_primitive::<Float32Type>().value(i))),
        DataType::Float64 => float(a.as_primitive::<Float64Type>().value(i)),
        DataType::Decimal64(_, s) | DataType::Decimal128(_, s) => Value::String(
            krishiv_kernels::text::java_decimal_string(crate::csv::convert::decimal_at(a, i), *s),
        ),
        DataType::Utf8View => Value::String(a.as_string_view().value(i).to_owned()),
        DataType::BinaryView => {
            Value::String(super::convert::encode_base64(a.as_binary_view().value(i)))
        }
        DataType::Date32 => {
            let days = a.as_primitive::<Date32Type>().value(i);
            Value::String(
                chrono::NaiveDate::from_num_days_from_ce_opt(days + 719_163)
                    .expect("date")
                    .format("%Y-%m-%d")
                    .to_string(),
            )
        }
        DataType::Timestamp(..) => {
            let micros = a.as_primitive::<TimestampMicrosecondType>().value(i);
            let t = chrono::DateTime::from_timestamp_micros(micros).expect("timestamp");
            let base = t.format("%Y-%m-%dT%H:%M:%S").to_string();
            Value::String(if micros.rem_euclid(1_000_000) == 0 {
                base
            } else {
                format!("{base}.{:06}", micros.rem_euclid(1_000_000))
            })
        }
        DataType::Struct(fields) => {
            let s = a.as_struct();
            let mut m = Map::new();
            for (f, c) in fields.iter().zip(s.columns()) {
                m.insert(f.name().clone(), plain(c.as_ref(), i));
            }
            Value::Object(m)
        }
        DataType::List(_) => {
            let l = a.as_list::<i32>().value(i);
            Value::Array((0..l.len()).map(|j| plain(l.as_ref(), j)).collect())
        }
        DataType::Map(..) => {
            let m = a.as_map();
            let (start, end) = (m.value_offsets()[i], m.value_offsets()[i + 1]);
            let mut out = Map::new();
            for j in usize::try_from(start).expect("offset")..usize::try_from(end).expect("offset")
            {
                out.insert(
                    m.keys().as_string_view().value(j).to_owned(),
                    plain(m.values().as_ref(), j),
                );
            }
            Value::Object(out)
        }
        other => panic!("{other}"),
    }
}

fn float(v: f64) -> Value {
    if v.is_nan() {
        Value::String("NaN".to_owned())
    } else if v.is_infinite() {
        Value::String(if v > 0.0 { "Infinity" } else { "-Infinity" }.to_owned())
    } else {
        serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number)
    }
}

fn rows(batches: &[RecordBatch]) -> Vec<Value> {
    let mut out = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            out.push(Value::Array(
                b.columns().iter().map(|c| plain(c.as_ref(), i)).collect(),
            ));
        }
    }
    out
}

/// The schema a read case uses: its DDL, or the inferred one.
fn read_schema(case: &Value, file: &[u8], o: &FormatOptions) -> Result<SchemaRef, KrishivError> {
    match case["schema"].as_str() {
        Some(ddl) => Ok(schema_of(ddl)),
        None => infer(file, o),
    }
}

#[test]
fn nested_json_matches_spark() {
    let golden: Value = serde_json::from_str(NESTED).expect("fixture");
    assert_eq!(golden["spark_version"], "4.2.0");
    let mut failures = Vec::new();
    for case in golden["infers"].as_array().expect("infers") {
        let name = case["name"].as_str().expect("name");
        let file = file_of(case);
        let got = match infer(&file, &options_of(case)) {
            Ok(s) => serde_json::json!({ "inferred": simple_string(&s) }),
            Err(e) => serde_json::json!({ "error": class_names(&e) }),
        };
        let want = known(name, case).unwrap_or_else(|| match case.get("inferred") {
            Some(s) => serde_json::json!({ "inferred": s }),
            None => serde_json::json!({ "error": case["error"] }),
        });
        if got != want {
            failures.push(format!("infer {name}: got {got}, want {want}"));
        }
    }
    for case in golden["reads"].as_array().expect("reads") {
        let name = case["name"].as_str().expect("name");
        let file = file_of(case);
        let o = options_of(case);
        let got = read_schema(case, &file, &o).and_then(|schema| {
            let batches = read(&file, &o, &ScanRequest::all(Arc::clone(&schema)))?;
            Ok(serde_json::json!({
                "inferred": simple_string(&schema),
                "rows": rows(&batches),
            }))
        });
        let got = got.unwrap_or_else(|e| serde_json::json!({ "error": class_names(&e) }));
        let want = known(name, case).unwrap_or_else(|| match case.get("error") {
            Some(e) => serde_json::json!({ "error": e }),
            None => serde_json::json!({ "inferred": case["inferred"], "rows": case["rows"] }),
        });
        if got != want {
            failures.push(format!("read {name}: got {got}, want {want}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Values the row encoding cannot show: duplicate map keys and a wrapped
/// integral timestamp, against the expressions Spark ran over the read.
#[test]
fn map_duplicates_and_timestamp_overflow_match_spark() {
    let golden: Value = serde_json::from_str(NESTED).expect("fixture");
    let selects = golden["selects"].as_array().expect("selects");
    let case = |name: &str| {
        selects
            .iter()
            .find(|c| c["name"] == name)
            .unwrap_or_else(|| panic!("{name}"))
    };
    // `size(m)` and `map_keys(m)`: every key is kept.
    let c = case("map duplicate keys");
    let schema = schema_of(c["schema"].as_str().expect("schema"));
    let got = read(&file_of(c), &options_of(c), &ScanRequest::all(schema)).expect("read");
    let m = got[0].column(0).as_map();
    let keys: Vec<Value> = (0..m.keys().len())
        .map(|j| Value::String(m.keys().as_string_view().value(j).to_owned()))
        .collect();
    assert_eq!(
        serde_json::json!([[keys.len(), keys]]),
        c["rows"],
        "map duplicate keys"
    );
    // `unix_micros(ts)` and the corrupt column: seconds wrap past i64.
    let c = case("integral timestamp overflow");
    let schema = schema_of(c["schema"].as_str().expect("schema"));
    let got = read(&file_of(c), &options_of(c), &ScanRequest::all(schema)).expect("read");
    let ts = got[0].column(0).as_primitive::<TimestampMicrosecondType>();
    let corrupt = got[0].column(1);
    let rows: Vec<Value> = (0..ts.len())
        .map(|i| serde_json::json!([ts.value(i), plain(corrupt.as_ref(), i)]))
        .collect();
    assert_eq!(Value::Array(rows), c["rows"], "integral timestamp overflow");
}

/// Spark's `CAST(value AS STRING)` of a flat value, as the flat generator
/// records it.
fn cast_string(a: &dyn Array, i: usize) -> Value {
    if a.is_null(i) {
        return Value::Null;
    }
    Value::String(match a.data_type() {
        DataType::Float64 => java_double_string(a.as_primitive::<Float64Type>().value(i)),
        DataType::BinaryView => String::from_utf8_lossy(a.as_binary_view().value(i)).into_owned(),
        _ => match plain(a, i) {
            Value::String(s) => s,
            other => other.to_string(),
        },
    })
}

#[test]
fn flat_json_matches_spark() {
    let golden: Value = serde_json::from_str(FLAT).expect("fixture");
    assert_eq!(golden["spark_version"], "4.2.0");
    let mut failures = Vec::new();
    for case in golden["reads"].as_array().expect("reads") {
        let name = case["name"].as_str().expect("name");
        let file = file_of(case);
        let o = options_of(case);
        let schema = read_schema(case, &file, &o).expect("schema");
        let got = match read(&file, &o, &ScanRequest::all(Arc::clone(&schema))) {
            Ok(batches) => {
                let mut out = Vec::new();
                for b in &batches {
                    for i in 0..b.num_rows() {
                        out.push(Value::Array(
                            b.columns()
                                .iter()
                                .map(|c| cast_string(c.as_ref(), i))
                                .collect(),
                        ));
                    }
                }
                serde_json::json!({ "rows": out })
            }
            Err(e) => serde_json::json!({ "error": class_names(&e) }),
        };
        let want = match case.get("error") {
            Some(e) => serde_json::json!({ "error": e }),
            None => serde_json::json!({ "rows": case["rows"] }),
        };
        if got != want {
            failures.push(format!("read {name}: got {got}, Spark {want}"));
        }
    }
    for case in golden["infers"].as_array().expect("infers") {
        let name = case["name"].as_str().expect("name");
        let s = infer(&file_of(case), &options_of(case)).expect("infer");
        if simple_string(&s) != case["inferred"] {
            failures.push(format!(
                "infer {name}: got {}, Spark {}",
                simple_string(&s),
                case["inferred"]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
