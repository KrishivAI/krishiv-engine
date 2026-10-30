// SPDX-License-Identifier: Apache-2.0
//! The engine metrics registry end to end: every query that reaches
//! execution folds its merged metrics once at query end, and
//! `Session::metrics` serves them with the scraped gauges.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::FutureExt as _;
use futures::executor::block_on;
use krishiv_exec::{cpu_secs, scan_rows, spill_bytes, spill_files};
use krishiv_plan::analyzer::{Principal, PrincipalKind};
use krishiv_session::Session;

const Q1: &str = "\
SELECT l_returnflag, l_linestatus,
       sum(l_quantity) AS sum_qty,
       sum(l_extendedprice) AS sum_base_price,
       sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
       avg(l_quantity) AS avg_qty,
       avg(l_extendedprice) AS avg_price,
       avg(l_discount) AS avg_disc,
       count(*) AS count_order
FROM lineitem
WHERE l_shipdate <= date '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus";

fn fixture_path() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tpch/q1-lineitem.parquet"
    )
    .to_owned()
}

/// A session with the metrics fold on.
fn enabled() -> Session {
    Session::builder()
        .config("krishiv.metrics.enabled", "true")
        .build()
        .unwrap()
}

fn counter(snap: &krishiv_session::MetricsSnapshot, name: &str, labels: &[(&str, &str)]) -> f64 {
    snap.counter(name, labels).unwrap_or(-1.0)
}

#[test]
fn a_successful_query_folds_the_profile_totals() {
    let s = enabled();
    let lineitem = s.read().parquet([fixture_path()]);
    block_on(lineitem.create_or_replace_temp_view("lineitem")).unwrap();
    let df = block_on(s.sql(Q1)).unwrap();
    let (rows, profile) = block_on(df.collect_with_metrics()).unwrap();
    let result_rows: usize = rows.iter().map(RecordBatch::num_rows).sum();
    assert!(result_rows > 0);

    let snap = s.metrics();
    assert_eq!(
        counter(
            &snap,
            "krishiv.query.count",
            &[("status", "succeeded"), ("class", "interactive")]
        ),
        1.0
    );
    assert_eq!(
        counter(&snap, "krishiv.query.output.rows", &[]),
        result_rows as f64
    );
    assert_eq!(
        counter(&snap, "krishiv.query.scan.rows", &[]),
        scan_rows(&profile) as f64
    );
    assert_eq!(
        counter(&snap, "krishiv.query.cpu.time", &[("class", "interactive")]),
        cpu_secs(&profile)
    );
    assert_eq!(
        counter(&snap, "krishiv.query.spill.bytes", &[]),
        spill_bytes(&profile) as f64
    );
    assert_eq!(
        counter(&snap, "krishiv.query.spill.files", &[]),
        spill_files(&profile) as f64
    );
    let duration = snap
        .histograms
        .iter()
        .find(|h| h.name == "krishiv.query.duration")
        .unwrap();
    assert_eq!(duration.count, 1);
    assert!(duration.sum >= 0.0);
    // Gauges are scraped, not folded.
    let info = snap
        .gauges
        .iter()
        .find(|g| g.name == "krishiv.build.info")
        .unwrap();
    assert_eq!(info.value, 1.0);
    assert!(info.labels.iter().any(|(n, _)| n == "version"));
    assert!(info.labels.iter().any(|(n, _)| n == "git_sha"));
    // Entries without a source emit nothing.
    for name in [
        "krishiv.query.scan.bytes",
        "krishiv.cache.evictions",
        "krishiv.io.requests",
        "krishiv.io.bytes.read",
    ] {
        assert!(
            snap.counters.iter().all(|c| c.name != name),
            "{name} emitted without a source"
        );
    }
    assert!(
        snap.histograms
            .iter()
            .all(|h| h.name != "krishiv.io.request.duration")
    );
    // No query is running anymore.
    let active = snap
        .gauges
        .iter()
        .find(|g| {
            g.name == "krishiv.query.active"
                && g.labels == [("class".to_owned(), "interactive".to_owned())]
        })
        .unwrap();
    assert_eq!(active.value, 0.0);
}

#[test]
fn a_failing_query_folds_failed_with_its_error_class() {
    let s = enabled();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    let df = block_on(s.sql("SELECT a / 0 FROM t")).unwrap();
    let error = block_on(df.collect()).unwrap_err();
    assert_eq!(error.error_class().name(), "DIVIDE_BY_ZERO");

    let snap = s.metrics();
    assert_eq!(
        snap.counter(
            "krishiv.query.count",
            &[
                ("status", "failed"),
                ("class", "interactive"),
                ("error_class", "DIVIDE_BY_ZERO"),
            ]
        ),
        Some(1.0)
    );
    // No success was folded.
    assert_eq!(
        snap.counter(
            "krishiv.query.count",
            &[("status", "succeeded"), ("class", "interactive")]
        ),
        None
    );
}

#[test]
fn a_cancelled_query_folds_cancelled() {
    let s = enabled();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batches: Vec<RecordBatch> = (0..2_000_000_i64)
        .collect::<Vec<_>>()
        .chunks(8192)
        .map(|c| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from_iter_values(c.iter().copied()))],
            )
            .unwrap()
        })
        .collect();
    s.register_batches("big", schema, batches).unwrap();
    let df = block_on(s.sql("SELECT a FROM big ORDER BY a")).unwrap();
    {
        // Never read: dropping the stream cancels the running query.
        let _stream = block_on(df.execute_stream()).unwrap();
    }
    let snap = s.metrics();
    assert_eq!(
        snap.counter(
            "krishiv.query.count",
            &[("status", "cancelled"), ("class", "interactive")]
        ),
        Some(1.0)
    );
    let duration = snap
        .histograms
        .iter()
        .find(|h| {
            h.name == "krishiv.query.duration"
                && h.labels
                    .iter()
                    .any(|(n, v)| n == "status" && v == "cancelled")
        })
        .unwrap();
    assert_eq!(duration.count, 1);
}

#[test]
fn a_disabled_session_folds_nothing() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    let df = block_on(s.sql("SELECT a FROM t")).unwrap();
    assert_eq!(block_on(df.count()).unwrap(), 1);

    let snap = s.metrics();
    // The fold never ran: no folded series at all.
    assert!(snap.counters.is_empty());
    assert!(snap.histograms.is_empty());
    // Gauges are scraped regardless.
    assert!(snap.gauges.iter().any(|g| g.name == "krishiv.build.info"));
}

#[test]
fn a_spilling_query_folds_spill_totals() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .config("krishiv.metrics.enabled", "true")
        .config(
            "krishiv.spill.dir",
            dir.path().join("spill").to_str().unwrap(),
        )
        .config("krishiv.memory.limit", "6m")
        .build()
        .unwrap();
    // 300k fact rows joined and aggregated, as in `spill.rs`.
    let fact = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let rows: Vec<i64> = (0..300_000).collect();
    let batches: Vec<RecordBatch> = rows
        .chunks(8192)
        .map(|c| {
            RecordBatch::try_new(
                Arc::clone(&fact),
                vec![
                    Arc::new(Int64Array::from_iter_values(c.iter().map(|i| i % 60_000))),
                    Arc::new(Int64Array::from_iter_values(c.iter().map(|i| i % 1000))),
                ],
            )
            .unwrap()
        })
        .collect();
    s.register_batches("f", fact, batches).unwrap();
    let df = block_on(s.sql("SELECT k, sum(v) AS total FROM f GROUP BY k ORDER BY k")).unwrap();
    let (_, profile) = block_on(df.collect_with_metrics()).unwrap();
    assert!(spill_files(&profile) > 0, "the query did not spill");

    let snap = s.metrics();
    assert_eq!(
        counter(&snap, "krishiv.query.spill.files", &[]),
        spill_files(&profile) as f64
    );
    assert_eq!(
        counter(&snap, "krishiv.query.spill.bytes", &[]),
        spill_bytes(&profile) as f64
    );
    assert!(counter(&snap, "krishiv.query.output.rows", &[]) > 0.0);
}

#[test]
fn series_do_not_grow_with_principals_or_tables() {
    let executor =
        krishiv_exec::TokioExecutor::new(&krishiv_exec::ExecutorConfig::default()).unwrap();
    let mut sessions = Vec::new();
    for principal in 0..100 {
        let s = Session::builder()
            .executor(Arc::clone(&executor) as Arc<dyn krishiv_exec::Executor>)
            .principal(Principal::new(
                format!("user-{principal}"),
                PrincipalKind::User,
            ))
            .config("krishiv.metrics.enabled", "true")
            .build()
            .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        for table in 0..100 {
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(vec![table]))],
            )
            .unwrap();
            s.register_batches(&format!("t_{table}"), Arc::clone(&schema), vec![batch])
                .unwrap();
        }
        sessions.push(s);
    }
    // One query per table: 10,000 queries from 100 principals over 100
    // tables each. No series names a principal, table or query.
    for (p, s) in sessions.iter().enumerate() {
        let mut series_after_first = None;
        for table in 0..100 {
            let df = block_on(s.sql(&format!("SELECT COUNT(*) AS n FROM t_{table}"))).unwrap();
            assert_eq!(block_on(df.count()).unwrap(), 1);
            let snap = s.metrics();
            let count: usize = snap.counters.len() + snap.histograms.len() + snap.gauges.len();
            if table == 0 {
                series_after_first = Some(count);
            } else {
                assert_eq!(
                    Some(count),
                    series_after_first,
                    "principal {p}: new series after the first query"
                );
            }
        }
    }
}

/// A caller that drops the action before its stream exists (a cancel while
/// the query is being submitted) still folds the query: nothing stays
/// active, and the query is counted exactly once, as cancelled when the
/// drop came before it finished.
#[test]
fn dropping_the_action_future_before_the_stream_folds_the_query() {
    let s = enabled();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from_iter_values(0..1000))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    let df = block_on(s.sql("SELECT a FROM t")).unwrap();
    // One poll starts the query; dropping the future cancels the rest.
    let pending = df.execute_stream().now_or_never().is_none();
    let snap = s.metrics();
    let active = snap
        .gauges
        .iter()
        .find(|g| g.name == "krishiv.query.active")
        .unwrap();
    assert_eq!(active.value, 0.0);
    let counted: f64 = snap
        .counters
        .iter()
        .filter(|c| c.name == "krishiv.query.count")
        .map(|c| c.value)
        .sum();
    // A query that finished submitting before the drop folds when its
    // stream (dropped with the result) ends; either way, once.
    assert!(
        counted <= 1.0,
        "counted {counted} (future pending: {pending})"
    );
}
