// SPDX-License-Identifier: Apache-2.0
//! `spark.sql.legacy.timeParserPolicy=LEGACY` in the JSON, CSV and XML
//! functions: Spark's lenient `FastDateFormat` (goldens from a Spark 4.2.0
//! run, including its lenient fallback), end to end through the bindings.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, TimestampMicrosecondType};
use arrow_array::{Array, ArrayRef, MapArray, StringArray, StructArray};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowDataType, Field};
use krishiv_kernels::datetime::resolve_zone;

use super::{BindArg, BindOptions, bind_scalar, col, lit, strs};

mod probes {
    include!("legacy_fast_probes.rs");
}

fn map_lit(pairs: &[(&str, &str)]) -> ArrayRef {
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

fn legacy_in(zone: &str) -> BindOptions {
    BindOptions::new()
        .with_legacy_time_parser(true)
        .with_time_zone(Arc::new(resolve_zone(zone).unwrap()))
        .with_time_zone_name(Arc::from(zone))
        .with_configured_time_zone(Arc::from(zone))
}

/// `from_json({"c": text}, "c <ty>"[, {option: pattern}])`.
fn from_json(zone: &str, ty: &str, text: &str, option: Option<(&str, &str)>) -> ArrayRef {
    let json = strs(&[Some(&format!("{{\"c\":{}}}", serde_json_string(text)))]);
    let schema: ArrayRef = strs(&[Some(&format!("c {ty}"))]);
    let mut types = vec![
        json.data_type().clone().into(),
        BindArg::Literal(Arc::clone(&schema)),
    ];
    let mut args = vec![col(&json), lit(&schema)];
    if let Some((name, pattern)) = option {
        let opts = map_lit(&[(name, pattern)]);
        types.push(BindArg::Literal(Arc::clone(&opts)));
        args.push(lit(&opts));
    }
    let f = bind_scalar("from_json", &types, &legacy_in(zone)).unwrap();
    f.invoke(&args, 1).unwrap()
}

fn serde_json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 32 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `to_json(named_struct('c', value), {option: pattern})`.
fn to_json(zone: &str, value: ArrayRef, option: Option<(&str, &str)>) -> String {
    let field = Field::new("c", value.data_type().clone(), true);
    let st: ArrayRef = Arc::new(StructArray::new(vec![field].into(), vec![value], None));
    let mut types = vec![BindArg::Column {
        ty: st.data_type().clone(),
        nullable: true,
    }];
    let mut args = vec![col(&st)];
    if let Some((name, pattern)) = option {
        let opts = map_lit(&[(name, pattern)]);
        types.push(BindArg::Literal(Arc::clone(&opts)));
        args.push(lit(&opts));
    }
    let f = bind_scalar("to_json", &types, &legacy_in(zone)).unwrap();
    let out = f.invoke(&args, 1).unwrap();
    out.as_string::<i32>().value(0).to_owned()
}

fn unwrap_c(json: &str) -> String {
    // `{"c":"..."}`; the probes' texts hold no escapes but quotes.
    json.strip_prefix("{\"c\":\"")
        .and_then(|s| s.strip_suffix("\"}"))
        .unwrap_or(json)
        .to_owned()
}

#[test]
fn from_json_timestamps_match_spark_under_legacy() {
    let mut wrong = Vec::new();
    for &(zone, text, pattern, want) in probes::PARSE_TS {
        let out = from_json(
            zone,
            "TIMESTAMP",
            text,
            pattern.map(|p| ("timestampFormat", p)),
        );
        let c = out
            .as_struct()
            .column(0)
            .as_primitive::<TimestampMicrosecondType>();
        let got = (!c.is_null(0)).then(|| c.value(0));
        if got != want {
            wrong.push(format!(
                "{zone} {text:?} with {pattern:?}: {got:?}, Spark {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {}:\n{}",
        wrong.len(),
        probes::PARSE_TS.len(),
        wrong.join("\n")
    );
}

#[test]
fn from_json_dates_match_spark_under_legacy() {
    let mut wrong = Vec::new();
    for &(text, pattern, want) in probes::PARSE_DATE {
        let out = from_json("UTC", "DATE", text, pattern.map(|p| ("dateFormat", p)));
        let c = out.as_struct().column(0).as_primitive::<Date32Type>();
        let got = (!c.is_null(0)).then(|| c.value(0));
        if got != want {
            wrong.push(format!(
                "{text:?} with {pattern:?}: {got:?}, Spark {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {}:\n{}",
        wrong.len(),
        probes::PARSE_DATE.len(),
        wrong.join("\n")
    );
}

#[test]
fn to_json_timestamps_match_spark_under_legacy() {
    let mut wrong = Vec::new();
    for &(zone, micros, pattern, want) in probes::FORMAT_TS {
        let v: ArrayRef = Arc::new(
            arrow_array::TimestampMicrosecondArray::from(vec![Some(micros)])
                .with_timezone(Arc::from("UTC")),
        );
        let got = unwrap_c(&to_json(zone, v, pattern.map(|p| ("timestampFormat", p))));
        if got != want {
            wrong.push(format!(
                "{zone} {micros} with {pattern:?}: {got:?}, Spark {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {}:\n{}",
        wrong.len(),
        probes::FORMAT_TS.len(),
        wrong.join("\n")
    );
}

#[test]
fn to_json_dates_match_spark_under_legacy() {
    let mut wrong = Vec::new();
    for &(days, pattern, want) in probes::FORMAT_DATE {
        let v: ArrayRef = Arc::new(arrow_array::Date32Array::from(vec![Some(days)]));
        let got = unwrap_c(&to_json("UTC", v, pattern.map(|p| ("dateFormat", p))));
        if got != want {
            wrong.push(format!("{days} with {pattern:?}: {got:?}, Spark {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {}:\n{}",
        wrong.len(),
        probes::FORMAT_DATE.len(),
        wrong.join("\n")
    );
}

/// Spark builds the writer's timestamp formatter eagerly, the date one only
/// when it formats a date, and a reader's lazily (the fallback then reads
/// the value).
#[test]
fn a_bad_pattern_fails_where_spark_builds_the_formatter() {
    let ts: ArrayRef = Arc::new(
        arrow_array::TimestampMicrosecondArray::from(vec![Some(0)]).with_timezone(Arc::from("UTC")),
    );
    let st: ArrayRef = Arc::new(StructArray::new(
        vec![Field::new("c", ts.data_type().clone(), true)].into(),
        vec![ts],
        None,
    ));
    let opts = map_lit(&[("timestampFormat", "qq")]);
    let err = bind_scalar(
        "to_json",
        &[
            BindArg::Column {
                ty: st.data_type().clone(),
                nullable: true,
            },
            BindArg::Literal(opts),
        ],
        &legacy_in("UTC"),
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(err.contains("INVALID_DATETIME_PATTERN"), "{err}");
    // A reader falls back: `2024` is 2024-01-01 by the default grammar.
    let out = from_json("UTC", "TIMESTAMP", "2024", Some(("timestampFormat", "qq")));
    let c = out
        .as_struct()
        .column(0)
        .as_primitive::<TimestampMicrosecondType>();
    assert_eq!(c.value(0), 1_704_067_200_000_000);
}

/// `from_csv` / `to_csv` and `from_xml` / `to_xml` take the same formatter.
#[test]
fn csv_and_xml_use_the_lenient_formatter_under_legacy() {
    let csv = strs(&[Some("2024-02-30,2024-02-30 10:00:00.000123")]);
    let schema: ArrayRef = strs(&[Some("d DATE, t TIMESTAMP")]);
    let opts = map_lit(&[
        ("dateFormat", "yyyy-MM-dd"),
        ("timestampFormat", "yyyy-MM-dd HH:mm:ss.SSSSSS"),
    ]);
    let f = bind_scalar(
        "from_csv",
        &[
            csv.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&schema)),
            BindArg::Literal(Arc::clone(&opts)),
        ],
        &legacy_in("UTC"),
    )
    .unwrap();
    let out = f.invoke(&[col(&csv), lit(&schema), lit(&opts)], 1).unwrap();
    let st = out.as_struct();
    // March 1 2024, and `.000123` is 123 microseconds.
    assert_eq!(st.column(0).as_primitive::<Date32Type>().value(0), 19_783);
    assert_eq!(
        st.column(1)
            .as_primitive::<TimestampMicrosecondType>()
            .value(0),
        1_709_287_200_000_123
    );

    let xml = strs(&[Some(
        "<ROW><d>2024-02-30</d><t>2024-02-30 10:00:00.000123</t></ROW>",
    )]);
    let f = bind_scalar(
        "from_xml",
        &[
            xml.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&schema)),
            BindArg::Literal(Arc::clone(&opts)),
        ],
        &legacy_in("UTC"),
    )
    .unwrap();
    let out = f.invoke(&[col(&xml), lit(&schema), lit(&opts)], 1).unwrap();
    let st = out.as_struct();
    assert_eq!(st.column(0).as_primitive::<Date32Type>().value(0), 19_783);
    assert_eq!(
        st.column(1)
            .as_primitive::<TimestampMicrosecondType>()
            .value(0),
        1_709_287_200_000_123
    );
}
