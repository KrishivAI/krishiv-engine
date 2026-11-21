// SPDX-License-Identifier: Apache-2.0
//! `DataFrame::observe` / `Observation` end to end (P3-T084): the actions that must complete an
//! observation (writes, table writes, metrics and profile runs, result-cache hits), a limit above
//! capping the count, the engine's own errors, memory and cancellation. The fixture's cases are
//! in `observe_goldens.rs`.
#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    assert_eq!(first["n"], 3);
    // A later action whose own metrics differ (a fresh observation of it gives n = 1) leaves the
    // first action's.
    assert_eq!(block_on(df.limit(1).collect()).unwrap()[0].num_rows(), 1);
    assert_eq!(metrics(&obs.get().unwrap()), first);
    block_on(df.filter(col("v").eq(1)).count()).unwrap();
    assert_eq!(metrics(&obs.get().unwrap()), first);
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
    // A filter between changes which rows the limit pulled (known-differences.md, "a filter in
    // between"): over the one-batch, one-partition fixture the engine's count is deterministic.
    let obs = Observation::new();
    let df = base(&s)
        .observe(&obs, [count(lit(1)).alias("n")])
        .unwrap()
        .filter(col("v").is_not_null());
    block_on(df.limit(1).collect()).unwrap();
    assert_eq!(metrics(&obs.get().unwrap())["n"], FILTER_LIMIT_N);
}

/// The fixture's `observe -> filter(v IS NOT NULL) -> limit(1)` count: no cap reaches through the
/// filter, so the node counts the one batch the limit's pull read, all 3 rows (at most the input).
const FILTER_LIMIT_N: i64 = 3;

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

/// The states a driver retained after it ended and the table that merges them are charged to the
/// pool too: the drivers' own states fit, their merge does not.
#[test]
fn retained_states_and_the_merge_are_charged_to_the_memory_pool() {
    let s = Session::builder()
        .config("krishiv.memory.limit", "2200kb")
        .build()
        .unwrap();
    let obs = Observation::new();
    let df = s
        .range(0, 200_000, 1, Some(2))
        .observe(&obs, [expr("collect_list(id)").unwrap().alias("l")])
        .unwrap();
    // Drained, not counted: `count()` is an aggregate of its own, which the
    // pool would refuse first.
    let ran = block_on(async {
        let mut stream = df.execute_stream().await?;
        while let Some(batch) = stream.next().await {
            batch?;
        }
        Ok::<_, krishiv_common::error::KrishivError>(())
    });
    let got = obs.get_timeout(Duration::from_secs(10));
    let e = ran
        .err()
        .or(got.err())
        .expect("the limit is exceeded by the retained states plus their merge");
    assert_eq!(e.error_class().name(), "UNABLE_TO_ACQUIRE_MEMORY", "{e}");
}

/// One observation whose metrics fail to finalize does not take the others' with it, and the
/// failure reaches the Connect sink instead of leaving it silently unset.
#[test]
fn a_failed_observation_reaches_the_sink_beside_the_good_ones() {
    let s = Session::new().unwrap();
    s.conf().set("spark.sql.ansi.enabled", "true").unwrap();
    let sink = krishiv_session::ObservedSink::new();
    let df = base(&s)
        .observe_named("good", [count(lit(1)).alias("n")])
        .observe_named(
            "bad",
            [expr("count(1) / (count(1) - 3)").unwrap().alias("r")],
        )
        .with_observed_sink(&sink);
    block_on(df.count()).unwrap();
    let all = sink.take().expect("the action finished");
    assert_eq!(metrics(all["good"].as_ref().unwrap())["n"], 3);
    let e = all["bad"].as_ref().expect_err("the metric's own error");
    assert_eq!(e.error_class().name(), "DIVIDE_BY_ZERO", "{e}");
}

/// An observed frame registered as a temporary view completes its `Observation` when the view is
/// read through `Session::table` or SQL.
#[test]
fn a_temp_view_of_an_observed_frame_completes_the_observation() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    block_on(df.create_or_replace_temp_view("v")).unwrap();
    assert_eq!(block_on(s.table("v").count()).unwrap(), 3);
    assert_eq!(
        metrics(&obs.get_timeout(Duration::from_secs(10)).unwrap().unwrap())["n"],
        3
    );
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    block_on(df.create_or_replace_temp_view("w")).unwrap();
    block_on(async { s.sql("SELECT * FROM w").await.unwrap().count().await }).unwrap();
    assert_eq!(
        metrics(&obs.get_timeout(Duration::from_secs(10)).unwrap().unwrap())["n"],
        3
    );
}

/// A view of a view of an observed frame completes the observation too, though the first view was
/// never executed.
#[test]
fn a_view_over_a_view_of_an_observed_frame_completes_the_observation() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    block_on(df.create_or_replace_temp_view("v1")).unwrap();
    block_on(s.table("v1").create_or_replace_temp_view("v2")).unwrap();
    assert_eq!(block_on(s.table("v2").count()).unwrap(), 3);
    assert_eq!(
        metrics(&obs.get_timeout(Duration::from_secs(10)).unwrap().unwrap())["n"],
        3
    );
}

/// A shared CTE over an observed view, read first under `LIMIT 0` and then live: the `LIMIT 0`
/// reference does not plan the definition the live one reuses without its observed node (Spark
/// prunes only the zero-row reference; the live one reports the rows it read).
#[test]
fn a_cte_read_under_limit_zero_first_still_observes_the_live_read() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    block_on(df.create_or_replace_temp_view("ov")).unwrap();
    // `rand()` makes the definition nondeterministic, so its references always share one plan.
    let q = "WITH c AS (SELECT k, rand() AS r FROM ov) \
             SELECT k FROM (SELECT * FROM c LIMIT 0) UNION ALL SELECT k FROM c";
    let rows: usize = block_on(async { s.sql(q).await.unwrap().collect().await })
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 3);
    let got = obs.get_timeout(Duration::from_secs(10)).unwrap().unwrap();
    assert_eq!(
        metrics(&got),
        serde_json::Map::from_iter([("n".to_owned(), Value::from(3))])
    );
}

/// A nondeterministic echo of a string column that counts the rows it is evaluated over.
#[derive(Debug)]
struct Tick(Arc<AtomicUsize>);

impl krishiv_session::ScalarUdf for Tick {
    fn name(&self) -> &str {
        "tick"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &krishiv_session::Signature {
        static SIG: std::sync::OnceLock<krishiv_session::Signature> = std::sync::OnceLock::new();
        SIG.get_or_init(|| {
            krishiv_session::Signature::Exact(vec![krishiv_session::UdfInputType::Exact(
                krishiv_types::DataType::String,
            )])
        })
    }

    fn return_field(
        &self,
        _: &[krishiv_session::ArgField],
    ) -> Result<krishiv_session::ReturnField, krishiv_session::UdfError> {
        Ok(krishiv_session::ReturnField {
            data_type: krishiv_types::DataType::String,
            nullable: true,
        })
    }

    fn properties(&self) -> krishiv_session::UdfProperties {
        krishiv_session::UdfProperties {
            deterministic: false,
            null_handling: krishiv_session::NullHandling::Strict,
        }
    }

    fn invoke(
        &self,
        args: &[krishiv_session::UdfArg<'_>],
        rows: usize,
        _: &krishiv_session::InvokeContext,
    ) -> Result<arrow_array::ArrayRef, krishiv_session::UdfError> {
        self.0.fetch_add(rows, Ordering::Relaxed);
        let column = args[0].array();
        let strings = arrow_cast::cast(&column, &DataType::Utf8)
            .map_err(|e| krishiv_session::UdfError::new(e.to_string()))?;
        Ok(strings)
    }
}

/// A shared (nondeterministic) CTE read under `LIMIT 0` and live runs its body once: the
/// zero-row reference reads the one shared result, not a second copy of the body.
#[test]
fn a_shared_cte_under_limit_zero_runs_its_body_once() {
    let s = Session::new().unwrap();
    let evaluated = Arc::new(AtomicUsize::new(0));
    s.register_udf(Tick(Arc::clone(&evaluated))).unwrap();
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    block_on(df.create_or_replace_temp_view("ov")).unwrap();
    let q = "WITH c AS (SELECT tick(k) AS k FROM ov) \
             SELECT k FROM (SELECT * FROM c LIMIT 0) UNION ALL SELECT k FROM c";
    let rows: usize = block_on(async { s.sql(q).await.unwrap().collect().await })
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 3);
    assert_eq!(evaluated.load(Ordering::Relaxed), 3, "the body ran once");
    let got = obs.get_timeout(Duration::from_secs(10)).unwrap().unwrap();
    assert_eq!(
        metrics(&got),
        serde_json::Map::from_iter([("n".to_owned(), Value::from(3))])
    );
}

/// Runs `q` over an observed view `ov` and returns the rows and what the observation reported.
fn observed_rows(q: &str) -> (usize, serde_json::Map<String, Value>) {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    block_on(df.create_or_replace_temp_view("ov")).unwrap();
    let rows: usize = block_on(async { s.sql(q).await.unwrap().collect().await })
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    let got = obs.get_timeout(Duration::from_secs(10)).unwrap().unwrap();
    (rows, metrics(&got))
}

/// A shared CTE read only under `LIMIT 0` (twice, so the CTE is one shared plan) reports nothing:
/// Spark replaces each zero-row read with an empty relation and drops the definition with its
/// observed node. Without the liveness strip the shared plan reports the view's rows.
#[test]
fn a_shared_cte_read_only_under_limit_zero_reports_nothing() {
    let (rows, got) = observed_rows(
        "WITH c AS (SELECT k, rand() AS r FROM ov) \
         SELECT k FROM (SELECT * FROM c LIMIT 0) \
         UNION ALL SELECT k FROM (SELECT * FROM c LIMIT 0) \
         UNION ALL SELECT 'x'",
    );
    assert_eq!(rows, 1);
    assert!(got.is_empty(), "{got:?}");
}

/// The liveness is transitive: `a` is read live only by `b`, which only `LIMIT 0` reads, so `a`'s
/// observed node is dropped with `b`.
#[test]
fn a_shared_cte_read_live_only_by_a_dead_one_reports_nothing() {
    let (rows, got) = observed_rows(
        "WITH a AS (SELECT k, rand() AS r FROM ov), \
              b AS (SELECT x.k, rand() AS r FROM a x CROSS JOIN a y) \
         SELECT k FROM (SELECT * FROM b LIMIT 0) \
         UNION ALL SELECT k FROM (SELECT * FROM b LIMIT 0) \
         UNION ALL SELECT 'x'",
    );
    assert_eq!(rows, 1);
    assert!(got.is_empty(), "{got:?}");
}

/// `spark_partition_id()` in a metric under the second `UNION ALL` branch (its partitions
/// numbered from the first branch's count) is what the same expression gives in a projection at
/// the same position (the engine's `Project`, the stated oracle; no Spark golden covers it):
/// the branch's own index, 0 and 1 over five rows each, so 5.
#[test]
fn spark_partition_id_in_a_metric_matches_a_projection_at_its_place() {
    let s = Session::new().unwrap();
    let pid = || expr("spark_partition_id()").unwrap();
    let first = s.range(0, 10, 1, Some(2)).select([lit(0).alias("p")]);
    let oracle = first.union(&s.range(0, 10, 1, Some(2)).select([pid().alias("p")]));
    let summed = block_on(oracle.agg([sum(col("p")).alias("s")]).collect()).unwrap();
    let want = summed[0].column(0).as_primitive::<Int64Type>().value(0);
    assert_eq!(want, 5);
    let obs = Observation::new();
    let observed = s
        .range(0, 10, 1, Some(2))
        .observe(&obs, [sum(pid()).alias("p")])
        .unwrap()
        .select([lit(0).alias("p")]);
    assert_eq!(block_on(first.union(&observed).count()).unwrap(), 20);
    assert_eq!(metrics(&obs.get().unwrap())["p"], want);
}

/// Once the observation is delivered, the query keeps no observation state charged to its pool,
/// though the caller still holds the ended result stream (and so the query).
#[test]
fn a_delivered_observation_releases_its_retained_states() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = s
        .range(0, 100_000, 1, Some(2))
        .observe(&obs, [expr("size(collect_list(id))").unwrap().alias("l")])
        .unwrap();
    let held = block_on(async {
        let mut stream = df.execute_stream().await.unwrap();
        while let Some(b) = stream.next().await {
            b.unwrap();
        }
        stream
    });
    assert_eq!(metrics(&obs.get().unwrap())["l"], 100_000);
    let reserved = block_on(async {
        s.sql(
            "SELECT CAST(coalesce(sum(reserved_bytes), 0) AS BIGINT) FROM system.memory \
             WHERE kind = 'consumer' AND path LIKE '%observed-metrics%'",
        )
        .await
        .unwrap()
        .collect()
        .await
    })
    .unwrap();
    let reserved = reserved[0].column(0).as_primitive::<Int64Type>().value(0);
    drop(held);
    assert_eq!(reserved, 0);
}

/// What a bounded wait on `obs` gives: the error class it completed with. A pending observation
/// (no run completed it) fails the test instead of blocking.
fn failed_with(obs: &Observation) -> String {
    match obs.get_timeout(Duration::from_secs(10)) {
        Err(e) => e.error_class().name().to_owned(),
        Ok(Some(row)) => panic!("completed with metrics: {row:?}"),
        Ok(None) => panic!("still pending: no run completed the observation"),
    }
}

/// A session whose runtime admits one query at a time and queues `queued` more.
fn one_slot_session(queued: u32) -> Session {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resource-groups.json");
    std::fs::write(
        &path,
        format!(
            r#"{{"rootGroups": [{{"name": "root", "hardConcurrencyLimit": 1, "maxQueued": {queued}}}],
                "defaultGroup": "root"}}"#
        ),
    )
    .unwrap();
    let rt = krishiv_session::EngineRuntime::builder()
        .config(
            "krishiv.workload.resourceGroups.file",
            path.to_string_lossy().into_owned(),
        )
        .build()
        .unwrap();
    Session::builder().runtime(rt).build().unwrap()
}

/// Every run completes its observations, also the ones that end before the query starts.
#[test]
fn an_action_on_a_closed_session_completes_the_observation() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = s
        .range(0, 10, 1, None)
        .observe(&obs, [count(lit(1)).alias("n")])
        .unwrap();
    s.close();
    let err = block_on(df.count()).unwrap_err();
    assert_eq!(failed_with(&obs), err.error_class().name());
}

/// A write on a closed session is refused as its query starts (`run_on_query_logged`), after its
/// analysis.
#[test]
fn a_write_on_a_closed_session_completes_the_observation() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    s.close();
    let dir = tempfile::tempdir().unwrap();
    let err = block_on(df.write().parquet(dir.path().join("o").to_str().unwrap())).unwrap_err();
    assert_eq!(failed_with(&obs), err.error_class().name());
}

#[test]
fn a_refused_admission_completes_the_observation() {
    let s = one_slot_session(0);
    let held = block_on(s.range(0, 1, 1, None).execute_stream()).unwrap();
    let obs = Observation::new();
    let df = s
        .range(0, 10, 1, None)
        .observe(&obs, [count(lit(1)).alias("n")])
        .unwrap();
    let err = block_on(df.count()).unwrap_err();
    assert_eq!(
        err.error_class().name(),
        "KRISHIV_RESOURCE_GROUP_QUEUE_FULL"
    );
    assert_eq!(failed_with(&obs), "KRISHIV_RESOURCE_GROUP_QUEUE_FULL");
    drop(held);
}

#[tokio::test]
async fn an_action_dropped_while_it_waits_for_admission_completes_the_observation() {
    let s = one_slot_session(1);
    let held = s.range(0, 1, 1, None).execute_stream().await.unwrap();
    let obs = Observation::new();
    let df = s
        .range(0, 10, 1, None)
        .observe(&obs, [count(lit(1)).alias("n")])
        .unwrap();
    // Queued behind `held`, so the action is still in its setup when the caller gives up.
    let gave_up = tokio::time::timeout(Duration::from_millis(200), df.count()).await;
    assert!(gave_up.is_err(), "admitted past a full slot: {gave_up:?}");
    assert_eq!(failed_with(&obs), "OPERATION_CANCELED");
    drop(held);
}

/// An action an operator kills while it waits for admission completes its observation with the
/// kill (`KRISHIV_QUERY_KILLED`, as the action fails), not with a client's `OPERATION_CANCELED`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_action_killed_during_its_setup_completes_with_the_kill() {
    cancel_during_setup(krishiv_session::CancelReason::Admin, "KRISHIV_QUERY_KILLED").await;
}

/// The same for a time limit: the observation completes with `KRISHIV_QUERY_TIMEOUT`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_action_timed_out_during_its_setup_completes_with_the_timeout() {
    cancel_during_setup(
        krishiv_session::CancelReason::Timeout,
        "KRISHIV_QUERY_TIMEOUT",
    )
    .await;
}

/// Ends the session of an action queued for admission with `reason`: the action and its
/// observation both complete with `class`.
async fn cancel_during_setup(reason: krishiv_session::CancelReason, class: &str) {
    let s = one_slot_session(1);
    let rows = s
        .sql("SELECT session_id FROM system.sessions WHERE state = 'active'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let id = rows[0].column(0).as_string::<i32>().value(0).to_owned();
    let held = s.range(0, 1, 1, None).execute_stream().await.unwrap();
    let obs = Observation::new();
    let df = s
        .range(0, 10, 1, None)
        .observe(&obs, [count(lit(1)).alias("n")])
        .unwrap();
    let action = tokio::spawn(async move { df.count().await });
    // Queued behind `held`, so the kill finds the action in its setup.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!action.is_finished(), "admitted past a full slot");
    let rt = Arc::clone(s.runtime());
    rt.kill_session(&id, reason).unwrap();
    let err = action.await.unwrap().unwrap_err();
    assert_eq!(err.error_class().name(), class, "{err}");
    assert_eq!(failed_with(&obs), class);
    drop(held);
}

/// A query whose only read of an observed temporary view is inside a SQL function's body
/// completes the observation: the body's view keeps reach the query's observers.
#[test]
fn a_view_read_only_inside_a_function_body_completes_the_observation() {
    let s = Session::new().unwrap();
    let obs = Observation::new();
    let df = base(&s).observe(&obs, [count(lit(1)).alias("n")]).unwrap();
    block_on(df.create_or_replace_temp_view("fv")).unwrap();
    block_on(async {
        s.sql(
            "CREATE TEMPORARY FUNCTION from_fv() RETURNS TABLE (k STRING) RETURN SELECT k FROM fv",
        )
        .await
        .unwrap()
        .collect()
        .await
    })
    .unwrap();
    let n = block_on(async {
        s.sql("SELECT * FROM from_fv()")
            .await
            .unwrap()
            .count()
            .await
    })
    .unwrap();
    assert_eq!(n, 3);
    assert_eq!(
        metrics(&obs.get_timeout(Duration::from_secs(10)).unwrap().unwrap())["n"],
        3
    );
}
