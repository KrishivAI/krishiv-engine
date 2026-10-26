// SPDX-License-Identifier: Apache-2.0
//! `EXPLAIN` as graphs: Mermaid and Graphviz renderings of the physical plan
//! and the profile DAG, and the `DataFrame` formats that select them.
#![cfg(test)]

mod common;

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_exec::{ExecutorConfig, TokioExecutor};
use krishiv_session::{
    ExplainMode, ExplainVerbose, GraphFormat, QueryProfile, Session, dot_of_profile,
    mermaid_of_profile,
};

use common::query;

const QUERY: &str = "\
SELECT c.name, count(*) AS n, sum(o.amount) AS total
FROM orders o JOIN customers c ON o.cust = c.id
WHERE o.amount > 10
GROUP BY c.name
ORDER BY c.name";

/// One compute thread, so the per-operator counts do not depend on how
/// drivers interleave; `customers` (4 rows) and `orders` (2,000 rows).
fn session() -> Session {
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: 1,
        io_threads: 1,
        blocking_threads: 1,
        io_blocking_threads: 2,
    })
    .unwrap();
    let s = Session::builder().executor(executor).build().unwrap();
    let customers = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&customers),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["ann", "bob", "cy", "dee"])),
        ],
    )
    .unwrap();
    s.register_batches("customers", customers, vec![batch])
        .unwrap();
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

/// A session over empty TPC-H tables (plans do not depend on data).
fn tpch() -> Session {
    let s = Session::new().unwrap();
    common::register(&s, 0);
    s
}

/// The node ids and edges of a Mermaid `flowchart TD` rendering.
fn mermaid_shape(text: &str) -> (Vec<usize>, Vec<(usize, usize)>) {
    assert!(text.starts_with("flowchart TD\n"), "{text}");
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for line in text.lines().skip(1) {
        let line = line
            .strip_prefix("    ")
            .unwrap_or_else(|| panic!("indent: {line}"));
        if line.contains("[\"") {
            let label = line
                .strip_prefix('n')
                .unwrap_or_else(|| panic!("node: {line}"));
            let end = label.find('[').unwrap_or_else(|| panic!("node: {line}"));
            let id: usize = label[..end].parse().unwrap_or_else(|e| panic!("{e}"));
            assert!(label[end..].starts_with("[\""), "{line}");
            assert!(label[end..].ends_with("\"]"), "{line}");
            nodes.push(id);
        } else {
            let mut parts = line.split(' ');
            let child = parts.next().unwrap_or_else(|| panic!("edge: {line}"));
            assert_eq!(parts.next(), Some("-->"), "{line}");
            let parent = parts.next().unwrap_or_else(|| panic!("edge: {line}"));
            assert_eq!(parts.next(), None, "{line}");
            let id = |n: &str| {
                n.strip_prefix('n')
                    .unwrap_or_else(|| panic!("edge: {line}"))
                    .parse()
                    .unwrap_or_else(|e| panic!("{e}"))
            };
            edges.push((id(child), id(parent)));
        }
    }
    (nodes, edges)
}

/// The node ids and edges of a Graphviz `digraph` rendering.
fn dot_shape(text: &str) -> (Vec<usize>, Vec<(usize, usize)>) {
    assert!(text.starts_with("digraph {\n"), "{text}");
    assert!(text.ends_with("}\n"), "{text}");
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for line in text.lines().skip(1).filter(|l| *l != "}") {
        let line = line
            .strip_prefix("    ")
            .unwrap_or_else(|| panic!("indent: {line}"));
        if line.contains("[label=") {
            let (head, label) = line
                .split_once(" [label=\"")
                .unwrap_or_else(|| panic!("{line}"));
            let id: usize = head
                .strip_prefix('n')
                .unwrap_or_else(|| panic!("node: {line}"))
                .parse()
                .unwrap_or_else(|e| panic!("{e}"));
            assert!(label.ends_with("\"];"), "{line}");
            nodes.push(id);
        } else {
            let line = line
                .strip_suffix(';')
                .unwrap_or_else(|| panic!("edge: {line}"));
            let mut parts = line.split(' ');
            let child = parts.next().unwrap_or_else(|| panic!("edge: {line}"));
            assert_eq!(parts.next(), Some("->"), "{line}");
            let parent = parts.next().unwrap_or_else(|| panic!("edge: {line}"));
            assert_eq!(parts.next(), None, "{line}");
            let id = |n: &str| {
                n.strip_prefix('n')
                    .unwrap_or_else(|| panic!("edge: {line}"))
                    .parse()
                    .unwrap_or_else(|e| panic!("{e}"))
            };
            edges.push((id(child), id(parent)));
        }
    }
    (nodes, edges)
}

/// A rendering is one tree: ids `n0..nN` in pre-order, one edge from each
/// parent to its inputs (the root sits at the top).
fn check_tree(nodes: &[usize], edges: &[(usize, usize)]) {
    assert_eq!(nodes, &(0..nodes.len()).collect::<Vec<_>>(), "{nodes:?}");
    assert_eq!(edges.len(), nodes.len().saturating_sub(1), "{edges:?}");
    let mut targets: Vec<usize> = edges.iter().map(|(_, child)| *child).collect();
    targets.sort_unstable();
    assert_eq!(targets, (1..nodes.len()).collect::<Vec<_>>(), "{edges:?}");
    for (parent, child) in edges {
        assert_ne!(child, parent, "{edges:?}");
        assert!(*parent < nodes.len(), "{edges:?}");
    }
}

/// Timings, byte counts, hash-seed-dependent table statistics and the rows a
/// runtime filter sees before it arrives vary from run to run; rows, batches
/// and structure do not.
fn normalize_profile(text: &str) -> String {
    const VOLATILE: [&str; 7] = [
        "runtime_filter_rows_tested",
        "runtime_filter_rows_dropped",
        "hash_buckets",
        "grace_hash_buckets",
        "probe_lookups",
        "probe_key_compares",
        "max_probe_length",
    ];
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        let Some(eq) = text[i..].find('=') else {
            out.push_str(&text[i..]);
            break;
        };
        let eq = i + eq;
        let mut start = eq;
        while start > i && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_') {
            start -= 1;
        }
        // `=` may start an escape (`\"`, `\\`, `\n`) or a plan predicate
        // (`[a] = [b]`): only a `key=digits` run is a metric.
        let mut end = eq + 1;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        let key = &text[start..eq];
        let placeholder = if end > eq + 1 && key.ends_with("_ns") {
            Some("<t>")
        } else if end > eq + 1 && key.ends_with("_bytes") {
            Some("<mem>")
        } else if end > eq + 1 && VOLATILE.contains(&key) {
            Some("<n>")
        } else {
            None
        };
        out.push_str(&text[i..eq + 1]);
        if let Some(mask) = placeholder {
            out.push_str(mask);
        } else {
            out.push_str(&text[eq + 1..end]);
        }
        i = end.max(eq + 1);
    }
    mask_filter_counts(&out)
}

/// `tested 2000, dropped 400` in a runtime filter's summary: both counts
/// depend on how many rows were scanned before the filter arrived.
fn mask_filter_counts(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(", tested ") {
        let (head, tail) = rest.split_at(at);
        out.push_str(head);
        let after = &tail[", tested ".len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        let tail_after = &after[digits..];
        match tail_after.strip_prefix(", dropped ") {
            Some(dropped) if digits > 0 => {
                let n = dropped.bytes().take_while(u8::is_ascii_digit).count();
                out.push_str(", tested <n>, dropped <n>");
                rest = &dropped[n..];
            }
            _ => {
                out.push_str(", tested ");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Every operator's rows and batches, in operator-id order: what two runs
/// of one query agree on.
fn operator_rows(profile: &QueryProfile) -> Vec<(u64, u64, u64, u64)> {
    profile
        .metrics
        .operators
        .values()
        .map(|e| {
            (
                e.metrics.rows_in,
                e.metrics.batches_in,
                e.metrics.rows_out,
                e.metrics.batches_out,
            )
        })
        .collect()
}

#[test]
fn normalize_masks_only_volatile_metrics() {
    let raw = "n1 [label=\"Scan t\\n#4 MemoryScan source [p3]: in_rows=0 in_batches=0 \
        out_rows=600 out_batches=4 compute_ns=12 max_compute_ns=7 io_ns=3 peak_bytes=64 \
        hash_buckets=8 probe_lookups=600 groups=4 drivers=1\"];";
    let normalized = normalize_profile(raw);
    assert!(normalized.contains("compute_ns=<t>"), "{normalized}");
    assert!(normalized.contains("peak_bytes=<mem>"), "{normalized}");
    assert!(normalized.contains("hash_buckets=<n>"), "{normalized}");
    assert!(normalized.contains("probe_lookups=<n>"), "{normalized}");
    assert!(normalized.contains("groups=4"), "{normalized}");
    assert_eq!(
        normalize_profile(
            "runtime_filter_rows_tested=2000 f=#1 [c]: range, tested 2000, dropped 400\""
        ),
        "runtime_filter_rows_tested=<n> f=#1 [c]: range, tested <n>, dropped <n>\""
    );
    assert_eq!(
        mask_filter_counts("x, tested y, tested 5 z, tested 7, dropped"),
        "x, tested y, tested 5 z, tested 7, dropped"
    );
    assert!(normalized.contains("out_rows=600"), "{normalized}");
    assert!(normalized.contains("drivers=1"), "{normalized}");
}

#[test]
fn plan_graph_snapshots_join_aggregate() {
    let s = session();
    let df = block_on(s.sql(QUERY)).unwrap();
    let mermaid = block_on(df.explain_with(ExplainMode::Simple, GraphFormat::Mermaid)).unwrap();
    let (nodes, edges) = mermaid_shape(&mermaid);
    check_tree(&nodes, &edges);
    assert!(mermaid.contains("HashJoin"), "{mermaid}");
    insta::assert_snapshot!("plan_join_aggregate_mermaid", mermaid);
    let dot = block_on(df.explain_with(ExplainMode::Simple, GraphFormat::Dot)).unwrap();
    let (nodes, edges) = dot_shape(&dot);
    check_tree(&nodes, &edges);
    insta::assert_snapshot!("plan_join_aggregate_dot", dot);
}

#[test]
fn plan_graph_snapshots_tpch_q9() {
    // Q3 cannot be graphed: planning it fails on main with
    // `INTERNAL_ERROR comparison of Utf8 with Utf8View` (the committed
    // `explain__q03` snapshot shows the error as its physical section), and
    // `explain_with` is strict like `DataFrame::explain`. Q9 is the
    // TPC-H-shaped query instead: a six-way join with an aggregate.
    let s = tpch();
    let df = block_on(s.sql(&query(9))).unwrap();
    let mermaid = block_on(df.explain_with(ExplainMode::Simple, GraphFormat::Mermaid)).unwrap();
    let (nodes, edges) = mermaid_shape(&mermaid);
    check_tree(&nodes, &edges);
    insta::assert_snapshot!("plan_q09_mermaid", mermaid);
    let dot = block_on(df.explain_with(ExplainMode::Simple, GraphFormat::Dot)).unwrap();
    let (nodes, edges) = dot_shape(&dot);
    check_tree(&nodes, &edges);
    insta::assert_snapshot!("plan_q09_dot", dot);
}

#[test]
fn profile_graph_snapshot_join_aggregate() {
    let s = session();
    let df = block_on(s.sql(QUERY)).unwrap();
    let profile = block_on(df.explain_analyze()).unwrap();
    let mermaid = normalize_profile(&mermaid_of_profile(&profile));
    let (nodes, edges) = mermaid_shape(&mermaid_of_profile(&profile));
    check_tree(&nodes, &edges);
    assert!(mermaid.contains("HashAggregate"), "{mermaid}");
    insta::assert_snapshot!("profile_join_aggregate_mermaid", mermaid);
    let dot = normalize_profile(&dot_of_profile(&profile));
    let (nodes, edges) = dot_shape(&dot_of_profile(&profile));
    check_tree(&nodes, &edges);
    insta::assert_snapshot!("profile_join_aggregate_dot", dot);
}

#[test]
fn explain_with_covers_every_mode() {
    let s = session();
    let df = block_on(s.sql(QUERY)).unwrap();
    // Every mode plans the same physical plan: all modes graph identically
    // to `Simple` (compared across modes, not against a second run of the
    // same mode).
    let baseline_mermaid =
        block_on(df.explain_with(ExplainMode::Simple, GraphFormat::Mermaid)).unwrap();
    let baseline_dot = block_on(df.explain_with(ExplainMode::Simple, GraphFormat::Dot)).unwrap();
    for mode in [
        ExplainMode::Simple,
        ExplainMode::Extended,
        ExplainMode::Formatted,
        ExplainMode::Cost,
        ExplainMode::Codegen,
    ] {
        let mermaid = block_on(df.explain_with(mode, GraphFormat::Mermaid)).unwrap();
        let (nodes, edges) = mermaid_shape(&mermaid);
        check_tree(&nodes, &edges);
        assert_eq!(mermaid, baseline_mermaid, "{mode:?}");
        let dot = block_on(df.explain_with(mode, GraphFormat::Dot)).unwrap();
        let (nodes, edges) = dot_shape(&dot);
        check_tree(&nodes, &edges);
        assert_eq!(dot, baseline_dot, "{mode:?}");
        // A plan mode has no JSON document (as `DataFrame::explain` has no
        // JSON form either): the error names the mode and the formats.
        let error = block_on(df.explain_with(mode, GraphFormat::Json)).unwrap_err();
        assert_eq!(error.error_class(), ErrorClass::InvalidParameterValueString);
        let name = match mode {
            ExplainMode::Simple => "simple",
            ExplainMode::Extended => "extended",
            ExplainMode::Formatted => "formatted",
            ExplainMode::Cost => "cost",
            ExplainMode::Codegen => "codegen",
            _ => panic!("a new ExplainMode variant needs a graph name here"),
        };
        let text = error.to_string();
        assert!(text.contains("INVALID_PARAMETER_VALUE"), "{text}");
        assert!(text.contains(name), "{text}");
        assert!(text.contains("mermaid"), "{text}");
        assert!(text.contains("dot"), "{text}");
    }
}

#[test]
fn verbose_and_analyze_with_all_formats() {
    let s = session();
    let df = block_on(s.sql(QUERY)).unwrap();
    let verbose = block_on(df.explain_verbose()).unwrap();
    // The fixture query rewrites, so the graph dropping the steps is visible.
    assert!(!verbose.steps.is_empty(), "QUERY should rewrite");
    let json = block_on(df.explain_verbose_with(GraphFormat::Json)).unwrap();
    assert_eq!(ExplainVerbose::from_json(&json).unwrap(), verbose);
    let mermaid = block_on(df.explain_verbose_with(GraphFormat::Mermaid)).unwrap();
    let (nodes, edges) = mermaid_shape(&mermaid);
    check_tree(&nodes, &edges);
    // The verbose graph is the physical plan only: the optimizer steps above
    // are dropped (see `explain_verbose_with` docs), matching `explain_with`.
    assert_eq!(
        mermaid,
        block_on(df.explain_with(ExplainMode::Simple, GraphFormat::Mermaid)).unwrap()
    );
    let dot = block_on(df.explain_verbose_with(GraphFormat::Dot)).unwrap();
    let (nodes, edges) = dot_shape(&dot);
    check_tree(&nodes, &edges);

    let analyzed = block_on(df.explain_analyze()).unwrap();
    let json = block_on(df.explain_analyze_with(GraphFormat::Json)).unwrap();
    // Two runs of one query: timings and hash-seed-dependent table
    // statistics differ, but the plan and every operator's rows do not.
    let round_tripped = QueryProfile::from_json(&json).unwrap();
    assert_eq!(round_tripped.plan, analyzed.plan);
    assert_eq!(
        operator_rows(&round_tripped),
        operator_rows(&analyzed),
        "{json}"
    );
    let mermaid = block_on(df.explain_analyze_with(GraphFormat::Mermaid)).unwrap();
    let (nodes, edges) = mermaid_shape(&mermaid);
    check_tree(&nodes, &edges);
    let dot = block_on(df.explain_analyze_with(GraphFormat::Dot)).unwrap();
    let (nodes, edges) = dot_shape(&dot);
    check_tree(&nodes, &edges);
}
