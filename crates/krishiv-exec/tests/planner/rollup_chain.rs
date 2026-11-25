// SPDX-License-Identifier: Apache-2.0
//! `RollupFromParent` (`krishiv-optimizer` `rollup_chain`, TPC-DS q67's shape): the grouping sets
//! of a pre-aggregated expansion are each merged from their smallest computed parent set, with
//! the rows of the aggregate over the expanded rows.

use std::collections::HashMap;

use krishiv_common::config::Config;
use krishiv_optimizer::{ColumnStats, Statistics, TableStats};
use krishiv_plan::{Literal, ReadSource, Scan};

use super::*;

/// `t` as a 2M-row table whose columns are nearly unique, so the pre-aggregate below the
/// expansion does not reduce, and whose integer ranges prove the sums cannot overflow (eager
/// aggregation moves an `int` sum or average only with them, never a `bigint` one); statistics
/// choose the plan, never its rows.
#[derive(Debug)]
struct T;

impl Statistics for T {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let column = |n: u64, range: Option<(Literal, Literal)>| {
            let (min, max) = range.unzip();
            ColumnStats {
                distinct: Some(n),
                nulls: Some(0),
                min,
                max,
                ..ColumnStats::default()
            }
        };
        (name.last()? == "t").then(|| TableStats {
            rows: 2_000_000,
            columns: HashMap::from([
                (
                    "a".to_owned(),
                    column(
                        1_500_000,
                        Some((Literal::Integer(0), Literal::Integer(2_000_000))),
                    ),
                ),
                (
                    "b".to_owned(),
                    column(
                        2_000_000,
                        Some((Literal::Long(0), Literal::Long(20_000_000))),
                    ),
                ),
                ("s".to_owned(), column(2_000_000, None)),
            ]),
        })
    }
}

const FIRES: [&str; 4] = [
    // Every merged call over a ROLLUP of the table's columns, the grand total included.
    "SELECT a, b, s, sum(a), count(*), count(a), min(s), max(b), CAST(avg(a) AS STRING) FROM t \
     GROUP BY ROLLUP(a, b, s)",
    // A grouping expression with repeated values, `grouping_id`, `grouping` and `HAVING`.
    "SELECT a % 5 AS p, b, s, grouping_id() AS g, CAST(grouping(s) AS INT) AS gs, sum(a), count(*) FROM t \
     GROUP BY CUBE(a % 5, b, s) HAVING count(*) > 0",
    // Equal sets keep their own rows.
    "SELECT a, b, s, sum(a) FROM t GROUP BY GROUPING SETS ((a, b, s), (a, b), (a, b), ())",
    // Sets with no common finest set read the pre-aggregate.
    "SELECT a, b, s, count(*), max(b) FROM t GROUP BY GROUPING SETS ((a, b), (s), (a))",
];

fn fired(plan: &Plan) -> bool {
    plan.root().to_string().contains("__shared_rollup_")
}

fn rows_under(env: &Env, sql: &str, on: bool) -> Vec<String> {
    let mut config = Config::new();
    config
        .set("krishiv.sql.optimizer.shareCteInstances", "true")
        .unwrap();
    config
        .set(
            "krishiv.optimizer.rollupFromParent",
            if on { "true" } else { "false" },
        )
        .unwrap();
    let plan = krishiv_optimizer::optimize_with_udfs(
        &env.analyze(sql),
        &T,
        &krishiv_udf_sdk::UdfLibrary::new(),
        &config,
    )
    .unwrap();
    assert_eq!(fired(&plan), on, "{sql}\n{}", plan.root());
    let cx = PlanningContext::new(Arc::clone(&env.tables) as _, Arc::new(config));
    let runnable = env.runnable(DefaultPlanner.plan(&plan, &cx).unwrap());
    let mut rows = Vec::new();
    for b in &run(&runnable).unwrap_or_else(|e| panic!("{sql}: {e}")) {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b.columns().iter().map(|c| cell(c.as_ref(), r)).collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    rows
}

/// On or off, the rows are the same; on, the sets read their parents.
#[test]
fn each_set_from_its_parent_keeps_the_rows() {
    let env = env_of(200);
    for q in FIRES {
        let off = rows_under(&env, q, false);
        assert!(!off.is_empty(), "{q}");
        assert_eq!(rows_under(&env, q, true), off, "{q}");
    }
}

/// An empty table: the grand total's row (or none) is what the expansion gives.
#[test]
fn an_empty_input_keeps_the_rows() {
    let env = env_of(0);
    for q in FIRES {
        assert_eq!(rows_under(&env, q, true), rows_under(&env, q, false), "{q}");
    }
}
