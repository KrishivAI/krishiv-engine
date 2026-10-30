// SPDX-License-Identifier: Apache-2.0
//! `EXPLAIN ANALYZE` on TPC-H Q1, Q9 and Q18 over a small deterministic data
//! set (P2-T093a): snapshots of the text with timings and memory redacted,
//! per-operator row counts that equal the counts of independent queries, and
//! spill bytes under a tight memory limit.
#![cfg(test)]

mod common;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_exec::{ExecutorConfig, OperatorEntry, OperatorRole, QueryMetrics, TokioExecutor};
use krishiv_session::{Session, TextOptions};

use common::query;

/// Orders in the data set: 160 lineitems, 5 orders above Q18's threshold.
const ORDERS: usize = 40;

/// One compute thread, so the per-operator counts do not depend on how drivers
/// interleave.
fn session() -> Session {
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: 1,
        io_threads: 1,
        blocking_threads: 1,
        io_blocking_threads: 2,
    })
    .unwrap();
    let s = Session::builder().executor(executor).build().unwrap();
    common::register(&s, ORDERS);
    s
}

fn rows(batches: &[RecordBatch]) -> u64 {
    batches.iter().map(|b| b.num_rows() as u64).sum()
}

/// The single `count(*)` `sql` returns.
fn count(s: &Session, sql: &str) -> u64 {
    let batches = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    u64::try_from(batches[0].column(0).as_primitive::<Int64Type>().value(0)).unwrap()
}

#[test]
fn analyze_text_snapshots() {
    let s = session();
    for q in [1, 9, 18] {
        let df = block_on(s.sql(&query(q))).unwrap();
        let profile = block_on(df.explain_analyze()).unwrap();
        let text = profile.to_text(TextOptions::masked());
        // The data set exercises the query: the snapshot is not of an empty run.
        assert!(!text.contains("out 0 rows (0 batches)"), "q{q}: {text}");
        match q {
            1 => insta::assert_snapshot!("analyze_q01", text),
            9 => insta::assert_snapshot!("analyze_q09", text),
            _ => insta::assert_snapshot!("analyze_q18", text),
        }
    }
}

fn operators<'a>(
    m: &'a QueryMetrics,
    role: OperatorRole,
    name: &'a str,
) -> impl Iterator<Item = &'a OperatorEntry> {
    m.operators
        .values()
        .filter(move |e| e.role == role && e.name == name)
}

fn rows_out(m: &QueryMetrics, role: OperatorRole, name: &str) -> u64 {
    operators(m, role, name).map(|e| e.metrics.rows_out).sum()
}

fn rows_in(m: &QueryMetrics, role: OperatorRole, name: &str) -> u64 {
    operators(m, role, name).map(|e| e.metrics.rows_in).sum()
}

/// What the operators report equals what the queries return, and what
/// independent queries count.
#[test]
fn per_operator_row_counts_equal_the_actual_counts() {
    let s = session();
    for q in [1, 9, 18] {
        let df = block_on(s.sql(&query(q))).unwrap();
        let plain = block_on(df.collect()).unwrap();
        let (got, profile) = block_on(df.collect_with_profile()).unwrap();
        assert_eq!(got, plain, "q{q}: the profiled run returns the same rows");
        let m = &profile.metrics;
        let actual = rows(&plain);
        assert!(actual > 0, "q{q} returns rows on this data");

        // The result sink received exactly the rows returned.
        let result = m
            .operators
            .values()
            .find(|e| e.node.is_none() && e.role == OperatorRole::Sink)
            .unwrap();
        assert_eq!(result.metrics.rows_in, actual, "q{q} result sink");

        // A projection neither adds nor drops rows, and a sort reads as many
        // as it writes.
        for e in m.operators.values().filter(|e| e.name == "Project") {
            assert_eq!(e.metrics.rows_in, e.metrics.rows_out, "q{q} {e:?}");
        }
        assert_eq!(
            rows_in(m, OperatorRole::Sink, "Sort"),
            rows_out(m, OperatorRole::Source, "Sort"),
            "q{q} sort"
        );
    }
}

#[test]
fn q1_scan_and_aggregate_counts_match_independent_queries() {
    let s = session();
    let df = block_on(s.sql(&query(1))).unwrap();
    let (got, profile) = block_on(df.collect_with_profile()).unwrap();
    let m = &profile.metrics;
    let qualifying = count(
        &s,
        "SELECT count(*) FROM lineitem WHERE l_shipdate <= date '1998-09-02'",
    );
    let total = count(&s, "SELECT count(*) FROM lineitem");
    assert_eq!(total, (4 * ORDERS) as u64);
    assert!(qualifying > 0);
    // The fixture straddles the cutoff: the filter must remove rows, so a
    // scan that returned everything would fail the count below.
    assert!(
        qualifying < total,
        "qualifying {qualifying} of {total} lineitems"
    );
    // The scan's filter is pushed down: it emits the qualifying rows, which
    // the aggregate then consumes in full.
    let scanned: u64 = m
        .operators
        .values()
        .filter(|e| e.role == OperatorRole::Source && e.name.contains("Scan"))
        .map(|e| e.metrics.rows_out)
        .sum();
    assert_eq!(scanned, qualifying);
    assert_eq!(rows_in(m, OperatorRole::Sink, "HashAggregate"), qualifying);
    // One group per (returnflag, linestatus) is one returned row.
    let groups = count(
        &s,
        "SELECT count(*) FROM (SELECT DISTINCT l_returnflag, l_linestatus \
         FROM lineitem WHERE l_shipdate <= date '1998-09-02')",
    );
    assert_eq!(rows_out(m, OperatorRole::Source, "HashAggregate"), groups);
    assert_eq!(rows(&got), groups);
}

#[test]
fn q9_result_count_matches_independent_query() {
    let s = session();
    let df = block_on(s.sql(&query(9))).unwrap();
    let (got, _) = block_on(df.collect_with_profile()).unwrap();
    let actual = rows(&got);
    // One row per (nation, year) group: the distinct groups of the same
    // joined rows Q9 aggregates, counted without the aggregation.
    let expected = count(
        &s,
        "SELECT count(*) FROM (SELECT DISTINCT n_name AS nation, \
         extract(year FROM o_orderdate) AS o_year FROM part, supplier, \
         lineitem, partsupp, orders, nation WHERE s_suppkey = l_suppkey \
         AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey \
         AND p_partkey = l_partkey AND o_orderkey = l_orderkey \
         AND s_nationkey = n_nationkey AND p_name LIKE '%green%')",
    );
    assert_eq!(expected, 16, "the fixture yields 16 (nation, year) groups");
    assert_eq!(actual, expected, "q9 returns one row per group");
}

#[test]
fn q18_counts_match_independent_queries() {
    let s = session();
    let df = block_on(s.sql(&query(18))).unwrap();
    let (got, profile) = block_on(df.collect_with_profile()).unwrap();
    let big_orders = count(
        &s,
        "SELECT count(*) FROM (SELECT l_orderkey FROM lineitem \
         GROUP BY l_orderkey HAVING sum(l_quantity) > 300)",
    );
    assert_eq!(big_orders, (ORDERS / 8) as u64);
    // One row per big order, each with its 4 lineitems summed.
    assert_eq!(rows(&got), big_orders);
    let m = &profile.metrics;
    let sunk: u64 = m
        .operators
        .values()
        .filter(|e| e.node.is_none() && e.role == OperatorRole::Sink)
        .map(|e| e.metrics.rows_in)
        .sum();
    assert_eq!(sunk, big_orders);
}

/// Under a tight limit the profile reports spill bytes, in the metrics and in
/// the text; without a limit it reports none.
#[test]
fn spill_bytes_are_positive_under_a_tight_limit_and_zero_without() {
    let dir = tempfile::tempdir().unwrap();
    // The session creates the spill root (mode 0700) below the temp dir.
    let spill = dir.path().join("spill");
    let run = |limit: Option<&str>| {
        let mut b = Session::builder().config("krishiv.spill.dir", spill.to_str().unwrap());
        if let Some(limit) = limit {
            b = b.config("krishiv.memory.limit", limit);
        }
        let s = b.build().unwrap();
        // Enough data to overflow a 2 MiB pool: Q18's grouped lineitem.
        common::register(&s, 16_000);
        let df = block_on(s.sql(&query(18))).unwrap();
        block_on(df.collect_with_profile()).unwrap()
    };
    let spilled = |m: &QueryMetrics| -> u64 {
        m.operators
            .values()
            .map(|e| e.metrics.spill_write_bytes)
            .sum()
    };
    let (got_tight, tight) = run(Some("2m"));
    let (got_free, free) = run(None);
    assert_eq!(got_tight, got_free, "spilling does not change the result");
    assert!(spilled(&tight.metrics) > 0, "{:?}", tight.metrics);
    assert_eq!(spilled(&free.metrics), 0);
    let text = tight.to_text(TextOptions::default());
    assert!(text.contains("spill files "), "{text}");
}
