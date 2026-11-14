// SPDX-License-Identifier: Apache-2.0
//! `DataFrame::observe` / `Observation` end to end (P3-T084): the actions that must complete an
//! observation (writes, table writes, metrics and profile runs, result-cache hits), a limit above
//! capping the count, the engine's own errors, memory and cancellation. The fixture's cases are
//! in `observe_goldens.rs`.
#![cfg(test)]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use futures::executor::block_on;
use krishiv_session::functions::{col, count, expr, lit, max, sum};
use krishiv_session::{DataFrame, Observation, Session};
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

/// One metric cell: an integer or null.
fn cell(row: &RecordBatch, i: usize) -> Value {
    let c = row.column(i);
    if c.is_null(0) {
        return Value::Null;
    }
    match c.data_type() {
        DataType::Int64 => Value::from(c.as_primitive::<Int64Type>().value(0)),
        DataType::Int32 => Value::from(c.as_primitive::<Int32Type>().value(0)),
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

/// Runs `df.count()` and returns what the observation holds.
fn run(df: &DataFrame, obs: &Observation) -> serde_json::Map<String, Value> {
    block_on(df.count()).unwrap();
    metrics(&obs.get().unwrap())
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

/// What a dropped stream guarantees: the observation completes, and what it holds is
/// `{}` (the aborted drivers' partials had not arrived) or a count no larger than the input.
/// Spark also reports a partial value for a cancelled action.
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
    match metrics(&got).get("n") {
        None => assert_eq!(got.num_columns(), 0),
        Some(n) => assert!((0..=50_000_000).contains(&n.as_i64().unwrap())),
    }
}

/// Every way an action runs the plan completes the observation (one hook,
/// `Session::run_on_query_logged`), not only `execute_stream`.
#[test]
fn a_parquet_write_completes_the_observation() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s)
        .observe(&obs, [count(lit(1)).alias("n"), sum(col("v")).alias("s")])
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    block_on(df.write().parquet(dir.path().join("o").to_str().unwrap())).unwrap();
    let got = metrics(
        &obs.get_timeout(Duration::from_secs(10))
            .unwrap()
            .expect("completes"),
    );
    assert_eq!((got["n"].as_i64(), got["s"].as_i64()), (Some(3), Some(3)));
}

#[test]
fn a_table_write_completes_the_observation() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    block_on(df.write().save_as_table("observed_out")).unwrap();
    let got = obs
        .get_timeout(Duration::from_secs(10))
        .unwrap()
        .expect("completes");
    assert_eq!(metrics(&got)["n"], 3);
    // Appending is another write through the same hook.
    let again = Observation::new();
    let df = base(&s)
        .observe(&again, [count(lit(1)).alias("n")])
        .unwrap();
    block_on(df.write().mode("append").insert_into("observed_out")).unwrap();
    let got = again
        .get_timeout(Duration::from_secs(10))
        .unwrap()
        .expect("completes");
    assert_eq!(metrics(&got)["n"], 3);
}

#[test]
fn collect_with_metrics_and_profile_complete_the_observation() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    block_on(df.collect_with_metrics()).unwrap();
    let got = obs
        .get_timeout(Duration::from_secs(10))
        .unwrap()
        .expect("completes");
    assert_eq!(metrics(&got)["n"], 3);
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    block_on(df.explain(krishiv_session::ExplainMode::Simple)).unwrap();
    assert!(
        obs.get_timeout(Duration::from_millis(200))
            .unwrap()
            .is_none()
    );
    block_on(df.collect_with_profile()).unwrap();
    let got = obs
        .get_timeout(Duration::from_secs(10))
        .unwrap()
        .expect("completes");
    assert_eq!(metrics(&got)["n"], 3);
}

/// A plan that observes is never served from the result cache: every action runs it, so every
/// observation completes.
#[test]
fn a_result_cache_hit_does_not_skip_the_observation() {
    let s = Session::new().unwrap();
    s.conf().set("krishiv.resultCache.enabled", "true").unwrap();
    block_on(
        block_on(s.sql(
            "CREATE TABLE rc_observed USING parquet AS SELECT 1 AS a UNION ALL SELECT 2 AS a",
        ))
        .unwrap()
        .collect(),
    )
    .unwrap();
    for _ in 0..3 {
        let obs = Observation::named("m").unwrap();
        let df = block_on(s.sql("SELECT a FROM rc_observed"))
            .unwrap()
            .observe(&obs, [count(lit(1)).alias("n")])
            .unwrap();
        assert_eq!(block_on(df.count()).unwrap(), 2);
        let got = obs
            .get_timeout(Duration::from_secs(10))
            .unwrap()
            .expect("completes");
        assert_eq!(metrics(&got)["n"], 2);
    }
}

/// Spark counts the rows its consumer pulled: a limit above (through row-for-row operators)
/// takes no more than that, not the batch it read.
#[test]
fn a_limit_above_caps_the_count_at_the_rows_it_takes() {
    let s = Session::new().unwrap();
    let count_of = |df: &DataFrame, limit: i32| {
        let obs = Observation::new();
        let observed = df.observe(&obs, [count(lit(1)).alias("n")]).unwrap();
        block_on(observed.limit(limit).collect()).unwrap();
        metrics(&obs.get_timeout(Duration::from_secs(10)).unwrap().unwrap())["n"]
            .as_i64()
            .unwrap()
    };
    assert_eq!(count_of(&base(&s), 1), 1);
    assert_eq!(count_of(&base(&s), 2), 2);
    assert_eq!(count_of(&base(&s), 10), 3);
    assert_eq!(count_of(&s.range(0, 1000, 1, Some(4)), 4), 4);
    // A projection between passes every row on.
    let obs = Observation::new();
    let df = s
        .range(0, 1000, 1, Some(4))
        .observe(&obs, [count(lit(1)).alias("n")])
        .unwrap()
        .select([col("id")]);
    block_on(df.limit(7).collect()).unwrap();
    assert_eq!(metrics(&obs.get().unwrap())["n"], 7);
    // A filter between changes which rows the limit pulled: the count is whole batches at least.
    let obs = Observation::new();
    let df = base(&s)
        .observe(&obs, [count(lit(1)).alias("n")])
        .unwrap()
        .filter(col("v").is_not_null());
    block_on(df.limit(1).collect()).unwrap();
    assert!(metrics(&obs.get().unwrap())["n"].as_i64().unwrap() >= 1);
}

/// A scalar subquery in a metric is refused as an unsupported feature, never an internal error.
#[test]
fn a_scalar_subquery_metric_is_unsupported_not_internal() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s)
        .observe(
            &obs,
            [expr("(select max(id) from range(3))").unwrap().alias("s")],
        )
        .unwrap();
    let e = block_on(df.count()).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE", "{e}");
}

/// The state of a metric is charged to the query's pool: an unbounded `collect_list` fails with
/// `UNABLE_TO_ACQUIRE_MEMORY` under a tiny limit and runs under a sufficient one.
#[test]
fn metric_state_is_charged_to_the_memory_pool() {
    let tiny = Session::builder()
        .config("krishiv.memory.limit", "64kb")
        .build()
        .unwrap();
    let obs = Observation::new();
    let df = tiny
        .range(0, 1_000_000, 1, Some(2))
        .observe(&obs, [expr("collect_list(id)").unwrap().alias("l")])
        .unwrap();
    let e = block_on(df.count()).unwrap_err();
    assert_eq!(e.error_class().name(), "UNABLE_TO_ACQUIRE_MEMORY", "{e}");
    let roomy = Session::builder()
        .config("krishiv.memory.limit", "512mb")
        .build()
        .unwrap();
    let obs = Observation::new();
    let df = roomy
        .range(0, 100_000, 1, Some(2))
        .observe(&obs, [expr("size(collect_list(id))").unwrap().alias("l")])
        .unwrap();
    assert_eq!(block_on(df.count()).unwrap(), 100_000);
    assert_eq!(metrics(&obs.get().unwrap())["l"], 100_000);
}

/// A metric that raises while the states are finalized is what `get` returns (Spark rethrows the
/// root cause), not an empty row.
#[test]
fn a_failed_finalization_is_what_get_returns() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    // The state folds; the metric over the results divides by zero when finalized.
    let df = base(&s)
        .observe(
            &obs,
            [expr("count(1) / (count(1) - 3)").unwrap().alias("r")],
        )
        .unwrap();
    s.conf().set("spark.sql.ansi.enabled", "true").unwrap();
    block_on(df.count()).unwrap();
    let e = obs
        .get_timeout(Duration::from_secs(10))
        .expect_err("the finalization error, not a silent {}");
    assert_eq!(e.error_class().name(), "DIVIDE_BY_ZERO", "{e}");
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
