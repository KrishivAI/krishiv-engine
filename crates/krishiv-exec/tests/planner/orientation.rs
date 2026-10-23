// SPDX-License-Identifier: Apache-2.0
//! Which input a hash join prefers to build (design P2-T166): a semi, anti
//! or left outer join whose right input is much larger than its left
//! prefers to build the left one. Those joins do not run yet, so the plan
//! keeps the right build and `EXPLAIN` shows the preference.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_exec::planner::{MemoryTable, MemoryTables};
use krishiv_optimizer::TableStats;
use krishiv_plan::analyzer::testing::context;
use krishiv_plan::analyzer::{MemoryCatalog, ResolutionConf, TableColumn};

use super::{Env, table};

/// `t` (10 rows), `big` and `mid` (the same rows, statistics claiming 100
/// and 5 million), and `bare` (no statistics: a default estimate of one
/// million rows).
fn env() -> Env {
    let (schema, batches) = table(10);
    let columns: Vec<TableColumn> = schema
        .fields()
        .iter()
        .map(|f| TableColumn {
            name: f.name().clone(),
            data_type: krishiv_types::DataType::from_arrow(f.data_type()).unwrap(),
            nullable: f.is_nullable(),
        })
        .collect();
    let mut catalog = MemoryCatalog::new("spark_catalog");
    let mut tables = MemoryTables::new();
    for (name, rows) in [
        ("t", Some(10)),
        ("big", Some(100_000_000)),
        ("mid", Some(5_000_000)),
        ("bare", None),
    ] {
        catalog = catalog.with_table(&["default"], name, columns.clone(), None);
        let stats = rows.map(|rows| TableStats {
            rows,
            columns: HashMap::new(),
        });
        let table = MemoryTable::new(Arc::clone(&schema), batches.clone()).with_statistics(stats);
        tables = tables.with_table(&["spark_catalog", "default", name], Arc::new(table));
    }
    Env {
        ctx: context(catalog, ResolutionConf::default()),
        tables: Arc::new(tables),
    }
}

/// The `HashJoin` lines of the plan of `sql`.
fn joins(env: &Env, sql: &str) -> Vec<String> {
    let plan = env.plan(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    plan.to_string()
        .lines()
        .filter_map(|l| l.find("HashJoin").map(|i| l[i..].to_owned()))
        .collect()
}

#[test]
fn a_much_larger_right_input_is_shown_as_the_side_not_to_build() {
    let env = env();
    let cases = [
        (
            "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM big WHERE big.a = t.a)",
            "HashJoin LeftSemi, [a#0] = [a#3], RuntimeFilter #1: a#0 -> spark_catalog.default.t.a, \
             prefers RightSemi (probe: 10 rows / 0.0 MiB, statistics; build: 100000000 rows / 1144.4 MiB, statistics)",
        ),
        (
            "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM big WHERE big.a = t.a)",
            "HashJoin LeftAnti, [a#0] = [a#3], \
             prefers RightAnti (probe: 10 rows / 0.0 MiB, statistics; build: 100000000 rows / 1144.4 MiB, statistics)",
        ),
    ];
    for (sql, want) in cases {
        assert_eq!(joins(&env, sql), [want], "{sql}");
    }
    let outer = joins(
        &env,
        "SELECT t.a, big.b FROM t LEFT JOIN big ON t.a = big.a",
    );
    assert_eq!(outer.len(), 1);
    assert!(outer[0].starts_with("HashJoin LeftOuter"), "{outer:?}");
    assert!(
        outer[0].contains(", prefers RightOuter (probe: 10 rows"),
        "{outer:?}"
    );
}

#[test]
fn null_aware_inner_small_and_hinted_joins_show_no_preference() {
    let env = env();
    for (sql, want) in [
        // `NOT IN` reads whether the whole subquery side holds a NULL.
        (
            "SELECT a FROM t WHERE a NOT IN (SELECT a FROM big)",
            "HashJoin NullAwareAnti, [a#0] = [a#3]",
        ),
        // Inner joins keep the join orderer's choice.
        (
            "SELECT t.a FROM t JOIN big ON t.a = big.a",
            "HashJoin Inner, [a#0] = [a#3], RuntimeFilter #1: a#0 -> spark_catalog.default.t.a",
        ),
        // A build small enough to broadcast stays.
        (
            "SELECT a FROM big WHERE EXISTS (SELECT 1 FROM t WHERE big.a = t.a)",
            "HashJoin LeftSemi, [a#0] = [a#3], \
             RuntimeFilter #1: a#0 -> spark_catalog.default.big.a (bloom)",
        ),
        // A hint decides.
        (
            "SELECT /*+ SHUFFLE_HASH(big) */ t.a FROM t LEFT JOIN big ON t.a = big.a",
            "HashJoin LeftOuter, [a#0] = [a#3]",
        ),
    ] {
        assert_eq!(joins(&env, sql), [want], "{sql}");
    }
}

#[test]
fn a_default_estimate_needs_a_larger_ratio() {
    let env = env();
    // 5M rows against a default 1M: 5x is enough with statistics on both
    // sides (above 2x), not with a default estimate on one (8x).
    let sql = "SELECT a FROM bare WHERE EXISTS (SELECT 1 FROM mid WHERE mid.a = bare.a)";
    assert!(
        joins(&env, sql).iter().all(|j| !j.contains("prefers")),
        "{sql}"
    );
    let sql = "SELECT a FROM bare WHERE EXISTS (SELECT 1 FROM big WHERE big.a = bare.a)";
    let js = joins(&env, sql);
    assert!(
        js.iter().any(|j| j.contains(
            "prefers RightSemi (probe: 1000000 rows / 11.4 MiB, default; build: 100000000"
        )),
        "{js:?}"
    );
    let sql = "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM mid WHERE mid.a = t.a)";
    assert!(
        joins(&env, sql)
            .iter()
            .any(|j| j.contains("prefers RightSemi")),
        "{sql}"
    );
}

/// A `WITH` reference reads as a default-sized relation (the estimator
/// does not look through it), so it is not taken as resting on statistics.
#[test]
fn a_cte_reference_is_a_default_estimate() {
    let env = env();
    let sql = "WITH c AS (SELECT a FROM big) \
               SELECT a FROM t WHERE EXISTS (SELECT 1 FROM c WHERE c.a = t.a)";
    let js = joins(&env, sql);
    assert_eq!(js.len(), 1, "{js:?}");
    assert!(
        js[0].ends_with("prefers RightSemi (probe: 10 rows / 0.0 MiB, statistics; build: 1000000 rows / 11.4 MiB, default)"),
        "{js:?}"
    );
}
