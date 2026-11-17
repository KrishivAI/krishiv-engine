// SPDX-License-Identifier: Apache-2.0
//! The physical planner's gate for key-routed pairs (design P3-A 1d): a
//! lone `count(DISTINCT)` in a `GROUP BY` gets `AggregatePlan::route` (and
//! `EXPLAIN` prints `pairs=routed`) only when a group key's distinct-value
//! floor, a lower bound the estimator never overstates, is at least 1,024.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_exec::planner::{ExplainNode, MemoryTable, MemoryTables};
use krishiv_optimizer::{ColumnStats, TableStats};
use krishiv_plan::analyzer::testing::context;
use krishiv_plan::analyzer::{MemoryCatalog, ResolutionConf, TableColumn};

use super::{Env, run, table};

/// `t` (the shared 10 rows) with statistics claiming 100,000 rows, `s` at
/// least `floor` distinct values, `a` at least 10 and `b` no floor.
fn env(floor: u64) -> Env {
    let (schema, batches) = table(10);
    let columns: Vec<TableColumn> = schema
        .fields()
        .iter()
        .map(|f| TableColumn {
            name: f.name().clone(),
            data_type: krishiv_types::DataType::from_arrow(f.data_type()).unwrap(),
            nullable: f.is_nullable(),
            char_varchar: None,
        })
        .collect();
    let catalog = MemoryCatalog::new("spark_catalog").with_table(&["default"], "t", columns, None);
    let at_least = |n: u64| ColumnStats {
        distinct_floor: Some(n),
        ..ColumnStats::default()
    };
    let t = MemoryTable::new(schema, batches).with_statistics(Some(TableStats {
        rows: 100_000,
        columns: HashMap::from([
            ("s".to_owned(), at_least(floor)),
            ("a".to_owned(), at_least(10)),
        ]),
    }));
    let tables = MemoryTables::new().with_table(&["spark_catalog", "default", "t"], Arc::new(t));
    Env {
        ctx: context(catalog, ResolutionConf::default()),
        tables: Arc::new(tables),
    }
}

fn routed_in(n: &ExplainNode) -> bool {
    n.detail.contains("pairs=routed") || n.children.iter().any(routed_in)
}

fn routed(floor: u64, sql: &str) -> bool {
    let plan = env(floor)
        .plan(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    routed_in(plan.explain())
}

const ALONE: &str = "SELECT s, count(DISTINCT b) FROM t GROUP BY s";

#[test]
fn the_gate_routes_from_a_floor_of_1024_keys() {
    assert!(!routed(1_023, ALONE));
    assert!(routed(1_024, ALONE));
    // Beside plain aggregates, and with a key of few values beside it.
    assert!(routed(
        1_024,
        "SELECT s, count(DISTINCT b), sum(b) FROM t GROUP BY s"
    ));
    assert!(routed(
        1_024,
        "SELECT a, s, count(DISTINCT b) FROM t GROUP BY a, s"
    ));
    assert!(!routed(
        1_024,
        "SELECT a, count(DISTINCT b) FROM t GROUP BY a"
    ));
    // No floor at all: never.
    assert!(!routed(0, ALONE));
}

#[test]
fn only_a_lone_unfiltered_distinct_count_is_routed() {
    for sql in [
        "SELECT s, sum(DISTINCT b) FROM t GROUP BY s",
        "SELECT s, count(DISTINCT b), count(DISTINCT a) FROM t GROUP BY s",
        "SELECT s, count(b) FROM t GROUP BY s",
        "SELECT count(DISTINCT b) FROM t",
    ] {
        assert!(!routed(100_000, sql), "{sql}");
    }
}

#[test]
fn a_filter_keeps_the_floor_only_through_unequal_and_not_null() {
    let q = |pred: &str| format!("SELECT s, count(DISTINCT b) FROM t WHERE {pred} GROUP BY s");
    // `s <> ''` removes one value at most (ClickBench q14's shape).
    assert!(routed(1_025, &q("s <> ''")));
    assert!(!routed(1_024, &q("s <> ''")));
    assert!(routed(1_024, &q("s IS NOT NULL")));
    for pred in ["b > 5", "s = 'x'", "s > 'x'", "s <> '' AND b > 5"] {
        assert!(!routed(100_000, &q(pred)), "{pred}");
    }
}

/// The routed plan returns the rows of the plan without routing
/// (`krishiv.exec.agg.distinctRouted` false).
#[test]
fn routed_and_unrouted_plans_return_the_same_rows() {
    let env = env(4_096);
    for sql in [
        ALONE,
        "SELECT s, count(DISTINCT a), max(b) FROM t GROUP BY s",
    ] {
        let on = env.plan(sql).unwrap();
        assert!(routed_in(on.explain()), "{sql}");
        let mut off = Config::new();
        off.set("krishiv.exec.agg.distinctRouted", "false").unwrap();
        let off = env.plan_with(sql, off).unwrap();
        let sorted = |b: Vec<arrow_array::RecordBatch>| {
            let mut r: Vec<String> = b
                .iter()
                .flat_map(|b| {
                    (0..b.num_rows()).map(move |i| format!("{:?}", b.slice(i, 1).columns()))
                })
                .collect();
            r.sort();
            r
        };
        let (a, b) = (sorted(run(&on).unwrap()), sorted(run(&off).unwrap()));
        assert_eq!(a, b, "{sql}");
        assert_eq!(a.len(), 10, "{sql}");
    }
}
