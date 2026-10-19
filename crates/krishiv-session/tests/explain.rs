// SPDX-License-Identifier: Apache-2.0
//! `EXPLAIN` and `DataFrame::explain`: Spark 4.2's section layout and mode
//! semantics (probed on 4.2.0), and snapshots of TPC-H Q1, Q3 and Q9.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_schema::{DataType as T, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::{ExplainMode, Session};

const TPCH: [(&str, &[(&str, T)]); 8] = [
    (
        "region",
        &[
            ("r_regionkey", T::Int64),
            ("r_name", T::Utf8),
            ("r_comment", T::Utf8),
        ],
    ),
    (
        "nation",
        &[
            ("n_nationkey", T::Int64),
            ("n_name", T::Utf8),
            ("n_regionkey", T::Int64),
            ("n_comment", T::Utf8),
        ],
    ),
    (
        "supplier",
        &[
            ("s_suppkey", T::Int64),
            ("s_name", T::Utf8),
            ("s_address", T::Utf8),
            ("s_nationkey", T::Int64),
            ("s_phone", T::Utf8),
            ("s_acctbal", T::Decimal128(15, 2)),
            ("s_comment", T::Utf8),
        ],
    ),
    (
        "customer",
        &[
            ("c_custkey", T::Int64),
            ("c_name", T::Utf8),
            ("c_address", T::Utf8),
            ("c_nationkey", T::Int64),
            ("c_phone", T::Utf8),
            ("c_acctbal", T::Decimal128(15, 2)),
            ("c_mktsegment", T::Utf8),
            ("c_comment", T::Utf8),
        ],
    ),
    (
        "part",
        &[
            ("p_partkey", T::Int64),
            ("p_name", T::Utf8),
            ("p_mfgr", T::Utf8),
            ("p_brand", T::Utf8),
            ("p_type", T::Utf8),
            ("p_size", T::Int32),
            ("p_container", T::Utf8),
            ("p_retailprice", T::Decimal128(15, 2)),
            ("p_comment", T::Utf8),
        ],
    ),
    (
        "partsupp",
        &[
            ("ps_partkey", T::Int64),
            ("ps_suppkey", T::Int64),
            ("ps_availqty", T::Int32),
            ("ps_supplycost", T::Decimal128(15, 2)),
            ("ps_comment", T::Utf8),
        ],
    ),
    (
        "orders",
        &[
            ("o_orderkey", T::Int64),
            ("o_custkey", T::Int64),
            ("o_orderstatus", T::Utf8),
            ("o_totalprice", T::Decimal128(15, 2)),
            ("o_orderdate", T::Date32),
            ("o_orderpriority", T::Utf8),
            ("o_clerk", T::Utf8),
            ("o_shippriority", T::Int32),
            ("o_comment", T::Utf8),
        ],
    ),
    (
        "lineitem",
        &[
            ("l_orderkey", T::Int64),
            ("l_partkey", T::Int64),
            ("l_suppkey", T::Int64),
            ("l_linenumber", T::Int32),
            ("l_quantity", T::Decimal128(15, 2)),
            ("l_extendedprice", T::Decimal128(15, 2)),
            ("l_discount", T::Decimal128(15, 2)),
            ("l_tax", T::Decimal128(15, 2)),
            ("l_returnflag", T::Utf8),
            ("l_linestatus", T::Utf8),
            ("l_shipdate", T::Date32),
            ("l_commitdate", T::Date32),
            ("l_receiptdate", T::Date32),
            ("l_shipinstruct", T::Utf8),
            ("l_shipmode", T::Utf8),
            ("l_comment", T::Utf8),
        ],
    ),
];

/// A session over empty TPC-H tables (plans do not depend on data).
fn tpch() -> Session {
    let s = Session::new().unwrap();
    for (name, cols) in TPCH {
        let fields: Vec<_> = cols
            .iter()
            .map(|(n, t)| Field::new(*n, t.clone(), true))
            .collect();
        s.register_batches(name, Arc::new(Schema::new(fields)), vec![])
            .unwrap();
    }
    s
}

fn query(q: usize) -> String {
    let path = format!(
        "{}/../../bench/queries/tpch/q{q:02}.sql",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(path).unwrap()
}

/// The single `plan` value `EXPLAIN` returns.
fn explain_sql(s: &Session, sql: &str) -> String {
    let df = block_on(s.sql(sql)).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(schema.fields().len(), 1);
    assert_eq!(schema.field(0).name(), "plan");
    assert_eq!(schema.field(0).data_type(), &T::Utf8View);
    let batches = block_on(df.collect()).unwrap();
    assert_eq!(
        batches
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        1
    );
    batches[0].column(0).as_string_view().value(0).to_owned()
}

fn headers(text: &str) -> Vec<&str> {
    text.lines().filter(|l| l.starts_with("== ")).collect()
}

const ALL: [&str; 4] = [
    "== Parsed Logical Plan ==",
    "== Analyzed Logical Plan ==",
    "== Optimized Logical Plan ==",
    "== Physical Plan ==",
];

#[test]
fn modes_print_spark_sections() {
    let s = tpch();
    let q = "SELECT n_name FROM nation WHERE n_regionkey > 1 ORDER BY n_name LIMIT 2";
    let simple = explain_sql(&s, &format!("EXPLAIN {q}"));
    assert_eq!(headers(&simple), ["== Physical Plan =="]);
    assert!(simple.ends_with("\n\n"), "{simple:?}");
    let extended = explain_sql(&s, &format!("EXPLAIN EXTENDED {q}"));
    assert_eq!(headers(&extended), ALL);
    assert!(
        extended.contains("== Analyzed Logical Plan ==\nn_name: string\n"),
        "{extended}"
    );
    assert!(
        extended.ends_with("\n") && !extended.ends_with("\n\n"),
        "{extended:?}"
    );
    let cost = explain_sql(&s, &format!("EXPLAIN COST {q}"));
    assert_eq!(headers(&cost), [ALL[2], ALL[3]]);
    assert!(
        cost.lines()
            .nth(1)
            .unwrap()
            .contains(", Statistics(rowCount="),
        "{cost}"
    );
    assert!(cost.ends_with("\n\n"), "{cost:?}");
    let formatted = explain_sql(&s, &format!("EXPLAIN FORMATTED {q}"));
    assert_eq!(headers(&formatted), ["== Physical Plan =="]);
    assert!(formatted.contains("\n(1) "), "{formatted}");
    assert!(formatted.ends_with("\n\n"), "{formatted:?}");
    let codegen = explain_sql(&s, &format!("EXPLAIN CODEGEN {q}"));
    assert_eq!(codegen, "Found 0 WholeStageCodegen subtrees.\n");
}

#[test]
fn formatted_numbers_children_first() {
    let s = tpch();
    let text = explain_sql(
        &s,
        "EXPLAIN FORMATTED SELECT n_name FROM nation WHERE n_regionkey > 1",
    );
    let outline: Vec<&str> = text.lines().skip(1).take_while(|l| !l.is_empty()).collect();
    let ids: Vec<usize> = outline
        .iter()
        .map(|l| {
            l.rsplit_once('(')
                .unwrap()
                .1
                .trim_end_matches(')')
                .parse()
                .unwrap()
        })
        .collect();
    // A chain: the root has the largest id, the leaf id 1.
    assert_eq!(ids, (1..=ids.len()).rev().collect::<Vec<_>>(), "{text}");
    for id in 1..=ids.len() {
        assert!(text.contains(&format!("\n({id}) ")), "{text}");
    }
}

#[test]
fn a_query_that_does_not_analyze_is_text_not_an_error() {
    let s = tpch();
    // Simple and formatted: Spark's fixed line.
    for mode in ["", "FORMATTED"] {
        let t = explain_sql(&s, &format!("EXPLAIN {mode} SELECT nope FROM nation"));
        assert_eq!(t, "Error occurred during query planning: ");
    }
    let t = explain_sql(&s, "EXPLAIN EXTENDED SELECT nope FROM nation");
    assert_eq!(headers(&t), [ALL[0], ALL[1]]);
    assert!(t.contains("UNRESOLVED_COLUMN"), "{t}");
    let t = explain_sql(&s, "EXPLAIN COST SELECT * FROM missing");
    assert!(t.contains("TABLE_OR_VIEW_NOT_FOUND"), "{t}");
    assert_eq!(headers(&t), [ALL[2], ALL[3]]);
    let t = explain_sql(&s, "EXPLAIN CODEGEN SELECT * FROM missing");
    assert!(t.contains("TABLE_OR_VIEW_NOT_FOUND"), "{t}");
}

#[test]
fn syntax_errors_and_unsupported_statements_are_errors() {
    let s = tpch();
    let e = block_on(s.sql("EXPLAIN SELECT 1 +")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::ParseSyntaxError);
    let e = block_on(s.sql("EXPLAIN LOGICAL SELECT 1")).unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_STATEMENT_OR_CLAUSE");
    let e = block_on(s.sql("EXPLAIN EXPLAIN SELECT 1")).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn dataframe_explain_raises_analysis_errors() {
    let s = tpch();
    let e = block_on(s.table("missing").explain(ExplainMode::Extended)).unwrap_err();
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    let df = s.table("nation");
    for (mode, first) in [
        (ExplainMode::Simple, "== Physical Plan =="),
        (ExplainMode::Formatted, "== Physical Plan =="),
        (ExplainMode::Cost, "== Optimized Logical Plan =="),
        (ExplainMode::Extended, "== Parsed Logical Plan =="),
    ] {
        let t = block_on(df.explain(mode)).unwrap();
        assert_eq!(t.lines().next(), Some(first));
    }
    let t = block_on(df.explain(ExplainMode::Codegen)).unwrap();
    assert_eq!(t, "Found 0 WholeStageCodegen subtrees.\n");
}

#[test]
fn explain_leaves_no_trace_in_the_catalog() {
    let s = tpch();
    let before = s.temp_table_names();
    explain_sql(&s, "EXPLAIN EXTENDED SELECT 1");
    assert_eq!(s.temp_table_names(), before);
}

fn logical_and_physical(s: &Session, q: usize) -> String {
    // The unresolved plan's expression ids and the physical operators move
    // with the planner; the snapshot is the whole EXTENDED text.
    explain_sql(s, &format!("EXPLAIN EXTENDED {}", query(q)))
}

#[test]
fn tpch_q1_q3_q9_snapshots() {
    let s = tpch();
    insta::assert_snapshot!("q01", logical_and_physical(&s, 1));
    insta::assert_snapshot!("q03", logical_and_physical(&s, 3));
    insta::assert_snapshot!("q09", logical_and_physical(&s, 9));
}

#[test]
fn cost_of_a_query_that_fails_after_analysis_repeats_the_error() {
    let s = tpch();
    // Analyzes, but the planner has no FULL OUTER JOIN yet.
    let q =
        "SELECT a.n_name FROM nation a FULL OUTER JOIN nation b ON a.n_nationkey = b.n_nationkey";
    // The query analyzes; only planning fails.
    block_on(block_on(s.sql(q)).unwrap().schema()).unwrap();
    let t = explain_sql(&s, &format!("EXPLAIN COST {q}"));
    // Spark's layout for a failed COST (oracle 4.2.0): the error, then it
    // again under the optimized and physical headers.
    let err = t.split("\n\n").next().unwrap();
    assert!(err.contains("FULL OUTER JOIN"), "{t}");
    assert_eq!(
        t,
        format!("{err}\n\n== Optimized Logical Plan ==\n{err}\n\n== Physical Plan ==\n{err}\n\n")
    );
}
