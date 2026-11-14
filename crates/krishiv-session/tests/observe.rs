// SPDX-License-Identifier: Apache-2.0
//! `DataFrame::observe` / `Observation` end to end (P3-T084): the classic
//! cases of `tests/fixtures/spark-4.2.0/observe-goldens.json` that do not
//! depend on row-at-a-time pulling, parallelism exactness, first action
//! wins, `LIMIT 0`, reuse errors and cancellation.
#![cfg(test)]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use futures::executor::block_on;
use krishiv_session::functions::{col, count, lit, max, sum};
use krishiv_session::{Column, DataFrame, Observation, Session};
use serde_json::Value;

/// The generator's `base()`: `k string, v int, x double`.
fn base(s: &Session) -> DataFrame {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, true),
        Field::new("v", DataType::Int32, true),
        Field::new("x", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
            Arc::new(Int32Array::from(vec![Some(1), Some(2), None])),
            Arc::new(Float64Array::from(vec![Some(1.5), None, Some(3.0)])),
        ],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    s.table("t")
}

/// One metric cell as the fixture writes it: an integer or null.
fn cell(row: &RecordBatch, i: usize) -> Value {
    let c = row.column(i);
    if c.is_null(0) {
        return Value::Null;
    }
    match c.data_type() {
        DataType::Int64 => Value::from(c.as_primitive::<Int64Type>().value(0)),
        DataType::Int32 => Value::from(c.as_primitive::<Int32Type>().value(0)),
        DataType::Float64 => Value::from(c.as_primitive::<Float64Type>().value(0)),
        DataType::Utf8 | DataType::Utf8View => Value::from(c.as_string_opt::<i32>().map_or_else(
            || c.as_string_view().value(0).to_owned(),
            |s| s.value(0).to_owned(),
        )),
        other => panic!("unexpected metric type {other}"),
    }
}

fn metrics(row: &RecordBatch) -> serde_json::Map<String, Value> {
    row.schema()
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name().clone(), cell(row, i)))
        .collect()
}

fn golden(case: &str) -> serde_json::Map<String, Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.2.0/observe-goldens.json"
    );
    let all: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    all["classic"][case]["metrics"].as_object().unwrap().clone()
}

/// Runs `df.count()` and returns what the observation holds.
fn run(df: &DataFrame, obs: &Observation) -> serde_json::Map<String, Value> {
    block_on(df.count()).unwrap();
    metrics(&obs.get().unwrap())
}

/// A golden case: its name, its metrics and what the frame is filtered by below the observation.
type Case = (
    &'static str,
    Vec<Column>,
    Box<dyn Fn(DataFrame) -> DataFrame>,
);

#[test]
fn classic_goldens() {
    let s = Session::new().unwrap();
    let t = base(&s);
    let n = || count(lit(1)).alias("n");
    let cases: Vec<Case> = vec![
        (
            "basic_alias",
            vec![
                count(lit(1)).alias("rows"),
                sum(col("v")).alias("sv"),
                max(col("k")).alias("mk"),
            ],
            Box::new(|d| d),
        ),
        (
            "literal_only",
            vec![lit(42).alias("c"), lit("s").alias("s")],
            Box::new(|d| d),
        ),
        (
            "filter_before_observe",
            vec![n()],
            Box::new(|d| d.filter(col("v").eq(1))),
        ),
        (
            "empty_input",
            vec![
                count(lit(1)).alias("c"),
                sum(col("v")).alias("s"),
                max(col("v")).alias("mx"),
            ],
            Box::new(|d| d.filter(lit(1).eq(0))),
        ),
        (
            "nulls_all",
            vec![
                sum(col("v")).alias("s"),
                count(col("v")).alias("c"),
                max(col("v")).alias("m"),
            ],
            Box::new(|d| d.filter(col("v").is_null())),
        ),
    ];
    for (name, metrics_of, below) in cases {
        let obs = Observation::new();
        let df = below(t.clone()).observe(&obs, metrics_of).unwrap();
        let got = run(&df, &obs);
        let want = golden(name);
        // `empty_input` and `nulls_all` fixtures list more columns than the
        // cases above build; every column built here must agree.
        for (k, v) in &got {
            assert_eq!(Some(v), want.get(k), "{name}: {k}");
        }
    }
}

#[test]
fn filter_above_counts_the_rows_before_it() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s)
        .observe(&obs, [count(lit(1)).alias("n")])
        .unwrap()
        .filter(col("v").eq(1));
    assert_eq!(block_on(df.count()).unwrap(), 1);
    assert_eq!(metrics(&obs.get().unwrap())["n"], 3);
}

#[test]
fn unaliased_metrics_are_named_as_spark_names_them() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s)
        .observe(&obs, [count(lit(1)), max(col("v")), lit(42)])
        .unwrap();
    let got = run(&df, &obs);
    assert_eq!(got["count(1)"], 3);
    assert_eq!(got["max(v)"], 2);
    assert_eq!(got["42"], 42);
}

#[test]
fn exact_under_parallelism() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = s
        .range(0, 100_000, 1, Some(8))
        .observe(
            &obs,
            [
                count(lit(1)).alias("n"),
                sum(col("id")).alias("s"),
                max(col("id")).alias("mx"),
            ],
        )
        .unwrap();
    let got = run(&df, &obs);
    assert_eq!(got["n"], 100_000);
    assert_eq!(got["s"], 4_999_950_000_i64);
    assert_eq!(got["mx"], 99_999);
}

#[test]
fn the_first_action_wins() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    block_on(df.count()).unwrap();
    let first = metrics(&obs.get().unwrap());
    block_on(df.filter(col("v").eq(1)).count()).unwrap();
    block_on(df.count()).unwrap();
    assert_eq!(metrics(&obs.get().unwrap()), first);
    assert_eq!(first["n"], 3);
}

#[test]
fn limit_zero_reports_nothing() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s)
        .observe(&obs, [count(lit(1)).alias("n")])
        .unwrap()
        .limit(0);
    block_on(df.count()).unwrap();
    assert_eq!(obs.get().unwrap().num_columns(), 0);
}

#[test]
fn reuse_and_get_before_observe_fail() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    assert_eq!(
        obs.get().unwrap_err().error_class().name(),
        "KRISHIV_NO_OBSERVE_BEFORE_GET"
    );
    let t = base(&s);
    t.observe(&obs, [count(lit(1))]).unwrap();
    assert_eq!(
        t.observe(&obs, [count(lit(1))])
            .unwrap_err()
            .error_class()
            .name(),
        "KRISHIV_OBSERVATION_REUSED"
    );
    // No action yet: `get` waits, and a bounded wait gives up.
    assert!(
        obs.get_timeout(Duration::from_millis(50))
            .unwrap()
            .is_none()
    );
}

#[test]
fn a_cancelled_action_completes_the_observation() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = s
        .range(0, 50_000_000, 1, Some(4))
        .observe(&obs, [count(lit(1)).alias("n")])
        .unwrap();
    block_on(async {
        let mut stream = df.execute_stream().await.unwrap();
        let _ = stream.next().await;
        drop(stream);
    });
    let got = obs
        .get_timeout(Duration::from_secs(30))
        .unwrap()
        .expect("completes after the stream is dropped");
    // Whatever flowed before the cancellation was counted, no more than the input.
    let n = metrics(&got).get("n").and_then(Value::as_i64).unwrap_or(0);
    assert!((0..=50_000_000).contains(&n));
}

#[test]
fn invalid_metrics_fail_the_action_with_spark_classes() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [col("v")]).unwrap();
    let e = block_on(df.count()).unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "INVALID_OBSERVED_METRICS.NON_AGGREGATE_FUNC_ARG_IS_ATTRIBUTE"
    );
}
