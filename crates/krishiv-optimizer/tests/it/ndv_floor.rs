// SPDX-License-Identifier: Apache-2.0
//! The distinct-value floor (`AttrStats::ndv_floor`): a lower bound on a
//! column's distinct non-null values that the estimator never overstates.
//! One test per rule.
//!
//! `t(a, b, s)` has 10,000 rows; the table's statistics say `a` has at
//! least 2,000 distinct values, `s` at least 1,500 and `b` none known.
//! `u(k)` has 100 rows with at least 50 distinct `k`.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_optimizer::{AttrStats, ColumnStats, PlanStats, Statistics, TableStats};
use krishiv_plan::analyzer::{AnalysisContext, MemoryCatalog, TableColumn};
use krishiv_plan::{LogicalPlan, PlanNode, ReadSource, Scan};
use krishiv_types::DataType;

use crate::support;

fn column(name: &str, t: DataType) -> TableColumn {
    TableColumn {
        name: name.to_owned(),
        data_type: t,
        nullable: true,
        char_varchar: None,
    }
}

fn ctx() -> AnalysisContext {
    let catalog = MemoryCatalog::new("spark_catalog")
        .with_table(
            &["default"],
            "t",
            vec![
                column("a", DataType::Integer),
                column("b", DataType::Integer),
                column("s", DataType::String),
            ],
            None,
        )
        .with_table(
            &["default"],
            "u",
            vec![column("k", DataType::Integer)],
            None,
        );
    support::ctx_over(catalog)
}

fn floor(distinct: u64, at_least: u64) -> ColumnStats {
    ColumnStats {
        distinct: Some(distinct),
        nulls: Some(0),
        distinct_floor: Some(at_least),
        ..ColumnStats::default()
    }
}

#[derive(Debug)]
struct Fixture(HashMap<String, TableStats>);

impl Statistics for Fixture {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        self.0.get(name.last()?).cloned()
    }
}

fn fixture() -> Fixture {
    Fixture(HashMap::from([
        (
            "t".to_owned(),
            TableStats {
                rows: 10_000,
                columns: HashMap::from([
                    ("a".to_owned(), floor(3_000, 2_000)),
                    (
                        "b".to_owned(),
                        ColumnStats {
                            distinct: Some(10),
                            ..ColumnStats::default()
                        },
                    ),
                    ("s".to_owned(), floor(6_000, 1_500)),
                ]),
            },
        ),
        (
            "u".to_owned(),
            TableStats {
                rows: 100,
                columns: HashMap::from([("k".to_owned(), floor(60, 50))]),
            },
        ),
    ]))
}

/// The first relation of `plan` (parents first) that `is` accepts.
fn find(plan: &Arc<LogicalPlan>, is: &dyn Fn(&LogicalPlan) -> bool) -> Arc<LogicalPlan> {
    let mut stack = vec![Arc::clone(plan)];
    while let Some(p) = stack.pop() {
        if is(&p) {
            return p;
        }
        let mut children = Vec::new();
        p.for_each_child(&mut |c| children.push(Arc::clone(c)));
        stack.extend(children.into_iter().rev());
    }
    panic!("no such relation in\n{plan}");
}

/// The estimate of column `name` of the first relation of `sql` that `is`
/// accepts.
fn col_of(sql: &str, is: &dyn Fn(&LogicalPlan) -> bool, name: &str) -> AttrStats {
    let plan = support::resolve(sql, &ctx());
    let stats = PlanStats::compute(plan.root(), &fixture());
    let node = find(plan.root(), is);
    let id = node
        .resolved_output()
        .unwrap()
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no column {name} in\n{node}"))
        .id;
    stats.get(&node).unwrap().cols[&id].clone()
}

fn is_filter(p: &LogicalPlan) -> bool {
    matches!(p.node, PlanNode::Filter { .. })
}

fn is_project(p: &LogicalPlan) -> bool {
    matches!(p.node, PlanNode::Project { .. })
}

fn is_aggregate(p: &LogicalPlan) -> bool {
    matches!(p.node, PlanNode::Aggregate { .. })
}

fn filtered_floor(pred: &str, name: &str) -> f64 {
    col_of(
        &format!("SELECT a, b, s FROM t WHERE {pred}"),
        &is_filter,
        name,
    )
    .ndv_floor
}

#[test]
fn a_scan_takes_the_tables_floor_and_none_without_one() {
    let scan = |p: &LogicalPlan| matches!(p.node, PlanNode::Scan(_));
    assert_eq!(col_of("SELECT a FROM t", &scan, "a").ndv_floor, 2_000.0);
    assert_eq!(col_of("SELECT b FROM t", &scan, "b").ndv_floor, 0.0);
    assert_eq!(
        AttrStats::guess(10.0).ndv_floor,
        0.0,
        "a guess has no floor"
    );
}

#[test]
fn a_projection_passes_a_bare_column_and_its_alias_but_nothing_computed() {
    let sql = "SELECT a, a AS aa, a + 1 AS a1, CAST(a AS BIGINT) AS ca, s FROM t";
    assert_eq!(col_of(sql, &is_project, "a").ndv_floor, 2_000.0);
    assert_eq!(col_of(sql, &is_project, "aa").ndv_floor, 2_000.0);
    assert_eq!(col_of(sql, &is_project, "s").ndv_floor, 1_500.0);
    assert_eq!(col_of(sql, &is_project, "a1").ndv_floor, 0.0);
    assert_eq!(col_of(sql, &is_project, "ca").ndv_floor, 0.0);
}

#[test]
fn a_filter_keeps_the_floor_of_a_column_it_only_compares_unequal_or_not_null() {
    assert_eq!(filtered_floor("s <> ''", "s"), 1_499.0);
    assert_eq!(filtered_floor("s <> '' AND s <> 'x'", "s"), 1_498.0);
    assert_eq!(filtered_floor("s IS NOT NULL", "s"), 1_500.0);
    assert_eq!(filtered_floor("s IS NOT NULL AND s <> 'x'", "s"), 1_499.0);
    assert_eq!(filtered_floor("'' <> s", "s"), 1_499.0);
    assert_eq!(filtered_floor("NOT (s = 'x')", "s"), 1_499.0);
    // Every other column may lose any of its values.
    assert_eq!(filtered_floor("s <> ''", "a"), 0.0);
    assert_eq!(filtered_floor("a IS NOT NULL", "s"), 0.0);
}

#[test]
fn a_filter_drops_the_floor_for_any_other_predicate() {
    for pred in [
        "s = 'x'",
        "s > 'x'",
        "s <> '' AND a > 5",
        "s <> '' AND a <> 5",
        "s <> '' OR s <> 'x'",
        "s IN ('x', 'y')",
        "s <> a",
        "length(s) <> 3",
        "a IS NULL",
    ] {
        assert_eq!(filtered_floor(pred, "s"), 0.0, "{pred}");
        assert_eq!(filtered_floor(pred, "a"), 0.0, "{pred}");
    }
    // `<> NULL` keeps no row.
    assert_eq!(filtered_floor("s <> NULL", "s"), 0.0);
}

#[test]
fn an_aggregate_keeps_its_group_columns_floor_and_its_results_have_none() {
    let sql = "SELECT s, count(DISTINCT a) AS c, max(a) AS m FROM t GROUP BY s";
    assert_eq!(col_of(sql, &is_aggregate, "s").ndv_floor, 1_500.0);
    assert_eq!(col_of(sql, &is_aggregate, "c").ndv_floor, 0.0);
    assert_eq!(col_of(sql, &is_aggregate, "m").ndv_floor, 0.0);
    // Through the filter of q14's shape.
    let sql = "SELECT s, count(DISTINCT a) AS c FROM t WHERE s <> '' GROUP BY s";
    assert_eq!(col_of(sql, &is_aggregate, "s").ndv_floor, 1_499.0);
}

#[test]
fn joins_unions_limits_and_samples_drop_the_floor() {
    let join = |p: &LogicalPlan| matches!(p.node, PlanNode::Join { .. });
    let sql = "SELECT a, k FROM t JOIN u ON a = k";
    assert_eq!(col_of(sql, &join, "a").ndv_floor, 0.0);
    assert_eq!(col_of(sql, &join, "k").ndv_floor, 0.0);
    let sql = "SELECT a, k FROM t LEFT JOIN u ON a = k";
    assert_eq!(col_of(sql, &join, "a").ndv_floor, 0.0);
    let union = |p: &LogicalPlan| matches!(p.node, PlanNode::Union { .. });
    let sql = "SELECT a FROM t UNION ALL SELECT k FROM u";
    assert_eq!(col_of(sql, &union, "a").ndv_floor, 0.0);
    let limit = |p: &LogicalPlan| matches!(p.node, PlanNode::Limit { .. });
    assert_eq!(
        col_of("SELECT a FROM t LIMIT 10", &limit, "a").ndv_floor,
        0.0
    );
    let sample = |p: &LogicalPlan| matches!(p.node, PlanNode::Sample { .. });
    let sql = "SELECT a FROM t TABLESAMPLE (10 PERCENT)";
    assert_eq!(col_of(sql, &sample, "a").ndv_floor, 0.0);
}

#[test]
fn row_preserving_nodes_keep_the_floor() {
    let sort = |p: &LogicalPlan| matches!(p.node, PlanNode::Sort { .. });
    assert_eq!(
        col_of("SELECT a FROM t ORDER BY a", &sort, "a").ndv_floor,
        2_000.0
    );
    let alias = |p: &LogicalPlan| matches!(p.node, PlanNode::SubqueryAlias { .. });
    assert_eq!(
        col_of("SELECT q.a FROM (SELECT a FROM t) q", &alias, "a").ndv_floor,
        2_000.0
    );
}
