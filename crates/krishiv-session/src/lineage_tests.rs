// SPDX-License-Identifier: Apache-2.0
//! Column lineage over the session's analyzed plans (P3-T063 N1a): views read through the side table the
//! session records, and lineage reading the analyzed plan, never the optimized one.

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_plan::analyzer::AnalyzedPlan;
use krishiv_plan::analyzer::lineage::{Leaf, Lineage, lineage};

fn session() -> crate::Session {
    let s = crate::Session::new().unwrap();
    for sql in [
        "CREATE TABLE base (a INT, b INT)",
        "CREATE TABLE other (a INT, x INT)",
        "CREATE VIEW pv AS SELECT a + 1 AS k FROM base WHERE b > 0",
        "CREATE TEMPORARY VIEW tv AS SELECT b FROM base",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    s
}

fn analyzed(df: &crate::DataFrame) -> Arc<AnalyzedPlan> {
    Arc::clone(&block_on(df.analysis()).unwrap().analyzed)
}

/// A leaf as `base.a`.
fn leaf(l: &Leaf) -> String {
    match l {
        Leaf::Table { name, column } => format!("{}.{column}", name.last().unwrap()),
        other => format!("{other:?}"),
    }
}

/// Per output: `name: [leaf@via, ...] -> transform`.
fn columns(l: &Lineage) -> Vec<String> {
    l.columns
        .iter()
        .map(|c| {
            let sources: Vec<String> = c
                .sources
                .iter()
                .map(|s| match &s.via_view {
                    Some(v) => format!("{}@{}", leaf(&s.leaf), v.join(".")),
                    None => leaf(&s.leaf),
                })
                .collect();
            format!("{}: [{}] -> {}", c.name, sources.join(", "), c.transform)
        })
        .collect()
}

fn filters(l: &Lineage) -> Vec<String> {
    l.filters
        .iter()
        .map(|(k, l)| format!("{k} {}", leaf(l)))
        .collect()
}

#[test]
fn lineage_view_expanded() {
    let s = session();
    let l = lineage(&analyzed(
        &block_on(s.sql("SELECT k, b FROM pv, tv")).unwrap(),
    ))
    .unwrap();
    assert_eq!(
        columns(&l),
        [
            "k: [base.a@spark_catalog.default.pv] -> expression",
            "b: [base.b@tv] -> identity",
        ]
    );
    assert!(l.warnings.is_empty());
}

/// `df.union(df)` over one view: both expansions carry `via_view`.
#[test]
fn lineage_view_read_twice_through_shared_dataframe() {
    let s = session();
    let df = s.table("pv");
    for df in [df.union(&df), df.cross_join(&df)] {
        let l = lineage(&analyzed(&df)).unwrap();
        assert!(
            l.columns.iter().flat_map(|c| &c.sources).all(|s| s
                .via_view
                .as_ref()
                .is_some_and(|v| v.join(".") == "spark_catalog.default.pv")),
            "{:?}",
            columns(&l)
        );
        assert!(!l.columns.iter().all(|c| c.sources.is_empty()));
    }
}

#[test]
fn lineage_view_in_sql_function_body() {
    let s = session();
    block_on(s.sql("CREATE TEMPORARY FUNCTION tf() RETURNS INT RETURN SELECT max(k) FROM pv"))
        .unwrap();
    let l = lineage(&analyzed(&block_on(s.sql("SELECT tf() AS m")).unwrap())).unwrap();
    assert_eq!(
        columns(&l),
        ["m: [base.a@spark_catalog.default.pv] -> aggregate"]
    );
}

/// Lineage reads the analyzed plan: the optimizer's plan of the same query differs (it infers a filter on
/// the join's other side and folds `1 = 1 OR b > 0`), so a lineage of the optimized plan fails this test.
#[test]
fn lineage_reads_the_analyzed_plan_not_the_optimized_one() {
    let s = session();
    let mut compared = 0;
    for (sql, want) in [
        (
            "SELECT base.a FROM base JOIN other ON base.a = other.a WHERE base.a > 5",
            vec!["filter base.a", "join base.a", "join other.a"],
        ),
        (
            "SELECT a FROM base WHERE 1 = 1 OR b > 0",
            vec!["filter base.b"],
        ),
    ] {
        let a = analyzed(&block_on(s.sql(sql)).unwrap());
        let l = lineage(&a).unwrap();
        assert_eq!(filters(&l), want, "{sql}");
        // The mutation: the optimized plan in the analyzed plan's place.
        let optimized = krishiv_optimizer::optimize(&a.plan).unwrap();
        let swapped = AnalyzedPlan {
            plan: optimized,
            ..(*a).clone()
        };
        // The optimized plan may have no lineage (an `Err` is also a different answer), but at least one
        // case must have one, so the comparison runs.
        if let Ok(m) = lineage(&swapped) {
            assert_ne!(filters(&m), want, "{sql}: the optimizer changed nothing");
            compared += 1;
        }
    }
    assert!(compared > 0, "no optimized plan had a lineage to compare");
}
