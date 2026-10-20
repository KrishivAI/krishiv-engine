// SPDX-License-Identifier: Apache-2.0
//! `EXPLAIN ANALYZE` and `EXPLAIN VERBOSE` (Krishiv extensions, ADR 0023):
//! golden text with run-dependent values masked, the JSON round trip,
//! metrics that agree with the rows the query returns, and the optimizer
//! steps of `EXPLAIN VERBOSE`.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_exec::{ExecutorConfig, Metric, OperatorRole, TokioExecutor};
use krishiv_session::{ExplainVerbose, QueryProfile, Session, TextOptions};

const QUERY: &str = "\
SELECT c.name, count(*) AS n, sum(o.amount) AS total
FROM orders o JOIN customers c ON o.cust = c.id
WHERE o.amount > 10
GROUP BY c.name
ORDER BY c.name";

/// A session on `threads` compute threads with `customers` (4 rows) and
/// `orders` (2,000 rows).
fn session(threads: usize) -> Session {
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: threads,
        io_threads: 1,
        blocking_threads: threads,
    })
    .unwrap();
    let s = Session::builder().executor(executor).build().unwrap();
    let customers = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let c = RecordBatch::try_new(
        Arc::clone(&customers),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["ann", "bob", "cy", "dee"])),
        ],
    )
    .unwrap();
    s.register_batches("customers", customers, vec![c]).unwrap();
    let orders = Arc::new(Schema::new(vec![
        Field::new("cust", DataType::Int64, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let batches = (0..4)
        .map(|b| {
            let ids: Vec<i64> = (0..500).map(|i| (b * 500 + i) % 5 + 1).collect();
            let amounts: Vec<i64> = (0..500).map(|i| (b * 500 + i) % 40).collect();
            RecordBatch::try_new(
                Arc::clone(&orders),
                vec![
                    Arc::new(Int64Array::from(ids)),
                    Arc::new(Int64Array::from(amounts)),
                ],
            )
            .unwrap()
        })
        .collect();
    s.register_batches("orders", orders, batches).unwrap();
    s
}

fn rows(batches: &[RecordBatch]) -> u64 {
    batches.iter().map(|b| b.num_rows() as u64).sum()
}

#[test]
fn analyze_text_is_stable() {
    let s = session(1);
    let df = block_on(s.sql(QUERY)).unwrap();
    let profile = block_on(df.explain_analyze()).unwrap();
    insta::assert_snapshot!(
        "analyze_join_aggregate",
        profile.to_text(TextOptions::masked())
    );
}

#[test]
fn analyze_counts_the_rows_the_query_returns() {
    for threads in [1, 4] {
        let s = session(threads);
        let df = block_on(s.sql(QUERY)).unwrap();
        let plain = block_on(df.collect()).unwrap();
        let (got, profile) = block_on(df.collect_with_profile()).unwrap();
        assert_eq!(got, plain, "EXPLAIN ANALYZE's run returns the query's rows");
        let m = &profile.metrics;
        let result = m
            .operators
            .values()
            .find(|e| e.node.is_none() && e.role == OperatorRole::Sink)
            .unwrap();
        assert_eq!(result.metrics.rows_in, rows(&plain));
        // Every operator but the result sink is placed under a plan node.
        let placed = m.operators.values().filter(|e| e.node.is_some()).count();
        assert_eq!(placed, m.operators.len() - 1);
        // The scans' rows are the join's input: 4 customers, and the
        // orders over 10 of a customer (the pushed filter and the runtime
        // filter drop the rest in the scan).
        let scanned: u64 = m
            .operators
            .values()
            .filter(|e| e.role == OperatorRole::Source && e.name.contains("Scan"))
            .map(|e| e.metrics.rows_out)
            .sum();
        let want = (0..2000_i64).filter(|i| i % 40 > 10 && i % 5 < 4).count() as u64;
        assert_eq!(scanned, want + 4);
        // Drivers merge: a pipeline's operators count the pipeline's drivers.
        for (id, p) in &m.pipelines {
            assert!(p.drivers >= 1 && p.threads >= 1, "pipeline {id}");
            assert!(p.threads <= threads as u64, "pipeline {id}");
            for e in m.operators.values().filter(|e| e.pipelines == [*id]) {
                assert_eq!(e.metrics.drivers, p.drivers, "{} in p{id}", e.name);
                assert!(e.metrics.max_driver_compute_time <= e.metrics.compute_time);
            }
        }
        let build = m
            .operators
            .values()
            .find(|e| e.name == "HashJoinBuild")
            .unwrap();
        assert_eq!(build.extra("build_rows"), Some(&Metric::Count(4)));
        assert_eq!(build.extra("hash_entries"), Some(&Metric::Count(4)));
        assert!(build.extra("hash_buckets").and_then(Metric::value) >= Some(4));
        let probe = m
            .operators
            .values()
            .find(|e| e.name == "HashJoinProbe")
            .unwrap();
        let lookups = probe.extra("probe_lookups").and_then(Metric::value);
        assert!(lookups.is_some_and(|l| l > 0), "{probe:?}");
        assert!(probe.extra("max_probe_length").is_some());
        // The JSON document round-trips.
        let back = QueryProfile::from_json(&profile.to_json()).unwrap();
        assert_eq!(back, profile);
    }
}

#[test]
fn plain_runs_do_not_count_probe_lengths() {
    let s = session(2);
    let df = block_on(s.sql(QUERY)).unwrap();
    let (_, m) = block_on(df.collect_with_metrics()).unwrap();
    let probe = m
        .operators
        .values()
        .find(|e| e.name == "HashJoinProbe")
        .unwrap();
    assert_eq!(probe.extra("probe_lookups"), None);
    assert!(probe.extra("probe_rows").is_some());
}

#[test]
fn sql_explain_analyze_and_verbose() {
    let s = session(2);
    let one = |sql: &str| {
        let batches = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
        assert_eq!(rows(&batches), 1, "{sql}");
        assert_eq!(batches[0].schema().field(0).name(), "plan");
        let col = batches[0].column(0);
        match col.data_type() {
            DataType::Utf8View => col.as_string_view().value(0).to_owned(),
            _ => col.as_string::<i32>().value(0).to_owned(),
        }
    };
    let text = one(&format!("EXPLAIN ANALYZE {QUERY}"));
    assert!(
        text.starts_with("== Physical Plan (analyzed) ==\n"),
        "{text}"
    );
    assert!(text.contains("HashJoinProbe"), "{text}");
    assert!(text.contains("Pipelines:"), "{text}");
    let text = one(&format!("explain verbose {QUERY}"));
    assert!(text.starts_with("== Parsed Logical Plan ==\n"), "{text}");
    assert!(
        text.contains("== Optimizer: Rewrite round 1, PushDownFilter ==\n"),
        "{text}"
    );
    assert!(text.contains("\n== Physical Plan ==\n"), "{text}");
    // Spark's modes are unchanged.
    assert!(one(&format!("EXPLAIN {QUERY}")).starts_with("== Physical Plan ==\n"));
    // A failure is an error here, not part of the text.
    let e = block_on(s.sql("EXPLAIN ANALYZE SELECT nope FROM orders"))
        .err()
        .unwrap();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::UnresolvedColumnWithSuggestion
    );
}

#[test]
fn verbose_steps_lead_to_the_optimized_plan() {
    let s = session(1);
    let df = block_on(s.sql(QUERY)).unwrap();
    let v = block_on(df.explain_verbose()).unwrap();
    assert!(!v.steps.is_empty());
    assert_eq!(
        v.steps.last().map(|s| s.plan.as_str()),
        Some(v.optimized.as_str())
    );
    for step in &v.steps {
        assert!(!step.diff.is_empty(), "{step:?}");
        assert!(
            step.diff
                .lines()
                .all(|l| l.starts_with("- ") || l.starts_with("+ "))
        );
    }
    assert_eq!(ExplainVerbose::from_json(&v.to_json()).unwrap(), v);
}

#[test]
fn analyze_tpch_q1_on_the_fixture() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tpch/q1-lineitem.parquet"
    );
    let s = session(2);
    block_on(
        s.read()
            .parquet([path])
            .create_or_replace_temp_view("lineitem"),
    )
    .unwrap();
    let df = block_on(s.sql(
        "SELECT l_returnflag, count(*) FROM lineitem \
         WHERE l_shipdate <= date '1998-09-02' GROUP BY l_returnflag",
    ))
    .unwrap();
    let (got, profile) = block_on(df.collect_with_profile()).unwrap();
    assert_eq!(rows(&got), 3);
    // The aggregate counted every row it received.
    let counted: i64 = got
        .iter()
        .flat_map(|b| {
            b.column(1)
                .as_primitive::<arrow_array::types::Int64Type>()
                .values()
                .to_vec()
        })
        .sum();
    let agg_in: u64 = profile
        .metrics
        .operators
        .values()
        .filter(|e| e.name == "HashAggregate" && e.role == OperatorRole::Sink)
        .map(|e| e.metrics.rows_in)
        .sum();
    assert_eq!(agg_in, u64::try_from(counted).unwrap());
    let text = profile.to_text(TextOptions::default());
    assert!(text.contains("groups=3"), "{text}");
}
