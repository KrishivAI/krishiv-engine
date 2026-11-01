// SPDX-License-Identifier: Apache-2.0
//! Which input a hash join builds (design P2-T166): a semi, anti or left
//! outer join whose right input is much larger than its left builds the
//! left one, as a right semi, right anti or right outer join, and `EXPLAIN`
//! shows the statistics that chose it.

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
fn a_much_larger_right_input_builds_the_left_one() {
    let env = env();
    let stats = |mib: &str| {
        format!(
            "builds left (left: 10 rows / 0.0 MiB, statistics; right: 100000000 rows / {mib} MiB, statistics)"
        )
    };
    // The large side probes, and the small side's keys filter its scan.
    assert_eq!(
        joins(
            &env,
            "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM big WHERE big.a = t.a)"
        ),
        [format!(
            "HashJoin RightSemi, [a#3] = [a#0], RuntimeFilter #1: a#3 -> spark_catalog.default.big.a (bloom), {}",
            stats("1144.4")
        )]
    );
    assert_eq!(
        joins(
            &env,
            "SELECT b FROM t WHERE NOT EXISTS (SELECT 1 FROM big WHERE big.b = t.b)"
        ),
        [format!(
            "HashJoin RightAnti, [b#4] = [b#1], RuntimeFilter #1: b#4 -> spark_catalog.default.big.b (bloom), {}",
            stats("1525.9")
        )]
    );
    let outer = joins(
        &env,
        "SELECT t.b, big.a FROM t LEFT JOIN big ON t.b = big.b",
    );
    assert_eq!(outer.len(), 1, "{outer:?}");
    assert!(
        outer[0].starts_with("HashJoin RightOuter, [b#"),
        "{outer:?}"
    );
    assert!(outer[0].contains("builds left (left: 10 rows"), "{outer:?}");
    // A full outer join is symmetric: it builds the smaller input, and no
    // runtime filter reaches either side.
    let full = joins(
        &env,
        "SELECT t.b, big.a FROM t FULL JOIN big ON t.b = big.b",
    );
    assert_eq!(full.len(), 1, "{full:?}");
    assert!(full[0].starts_with("HashJoin FullOuter, [b#"), "{full:?}");
    assert!(full[0].contains("builds left (left: 10 rows"), "{full:?}");
    assert!(!full[0].contains("RuntimeFilter"), "{full:?}");
    // Nullable keys flip too: build rows with a NULL `=` key are stored
    // (and spill) with their partition, never indexed.
    let anti = joins(
        &env,
        "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM big WHERE big.a = t.a)",
    );
    assert!(
        anti.len() == 1 && anti[0].starts_with("HashJoin RightAnti"),
        "{anti:?}"
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
    // No preference: `bare` probes a `LeftSemi` that builds `mid`.
    assert_eq!(
        joins(&env, sql),
        ["HashJoin LeftSemi, [a#0] = [a#3], RuntimeFilter #1: a#0 -> spark_catalog.default.bare.a"],
        "{sql}"
    );
    let sql = "SELECT a FROM bare WHERE EXISTS (SELECT 1 FROM big WHERE big.a = bare.a)";
    let js = joins(&env, sql);
    assert!(
        js.iter().any(|j| j.starts_with("HashJoin RightSemi")
            && j.ends_with(
                "builds left (left: 1000000 rows / 11.4 MiB, default; \
                 right: 100000000 rows / 1144.4 MiB, statistics)"
            )),
        "{js:?}"
    );
    let sql = "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM mid WHERE mid.a = t.a)";
    assert!(
        joins(&env, sql)
            .iter()
            .any(|j| j.starts_with("HashJoin RightSemi")),
        "{sql}"
    );
}

/// A deterministic `WITH` definition is inlined (Spark's `InlineCTE`), so
/// its reference reads the statistics of what it scans, and so does a
/// nondeterministic one read twice, which stays a reference.
#[test]
fn inlined_and_kept_ctes_read_statistics() {
    let env = env();
    let sql = "WITH c AS (SELECT a FROM big) \
               SELECT a FROM t WHERE EXISTS (SELECT 1 FROM c WHERE c.a = t.a)";
    let js = joins(&env, sql);
    assert_eq!(js.len(), 1, "{js:?}");
    assert!(
        js[0].starts_with("HashJoin RightSemi")
            && js[0].ends_with(
                "builds left (left: 10 rows / 0.0 MiB, statistics; \
                 right: 100000000 rows / 1144.4 MiB, statistics)"
            ),
        "{js:?}"
    );
    let sql = "WITH c AS (SELECT a, rand() AS r FROM big) \
               SELECT a FROM t WHERE EXISTS (SELECT 1 FROM c WHERE c.a = t.a) \
               AND EXISTS (SELECT 1 FROM c WHERE c.a + 1 = t.a)";
    let js = joins(&env, sql);
    assert_eq!(js.len(), 2, "{js:?}");
    assert!(
        js.iter()
            .all(|j| j.contains("statistics") && !j.contains("default)")),
        "{js:?}"
    );
}

/// A shared `WITH` reference is sized by its definition: the semi join builds
/// its smaller side, so a runtime filter reaches the copy and the group keeps
/// its copies (TPC-DS q95). Read as the default size, the reference looked
/// smaller than `mid` and the query ran with no filter into the body.
#[test]
fn a_shared_reference_sizes_the_join_that_decides_sharing() {
    let env = env();
    let sql = "WITH w AS (SELECT a, count(*) AS n FROM big GROUP BY a) \
               SELECT a FROM mid WHERE a IN (SELECT a FROM w) \
               UNION ALL SELECT a FROM w";
    let (shared, rows) = super::shared::sharing(&env, sql, &[]);
    assert_eq!(shared, 0, "a filter from the semi join reaches a copy");
    let (copies, same) = super::shared::sharing(
        &env,
        sql,
        &[("krishiv.sql.optimizer.shareCteInstances", "false")],
    );
    assert_eq!(copies, 0);
    assert_eq!(rows, same);
    // The body not reached by any filter stays shared under the same sizes.
    let sql = "WITH w AS (SELECT a, count(*) AS n FROM big GROUP BY a) \
               SELECT a FROM w UNION ALL SELECT a FROM w";
    assert_eq!(super::shared::sharing(&env, sql, &[]).0, 1);
}

/// `krishiv.exec.join.buildSide=false` keeps Spark's build side (the right
/// input) whatever the statistics say.
#[test]
fn the_build_side_setting_keeps_sparks_side() {
    use krishiv_exec::planner::{DefaultPlanner, PhysicalPlanner as _, PlanningContext};
    let env = env();
    let sql = "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM big WHERE big.a = t.a)";
    let mut config = krishiv_common::config::Config::new();
    config.set("krishiv.exec.join.buildSide", "false").unwrap();
    let cx = PlanningContext::new(Arc::clone(&env.tables) as _, Arc::new(config));
    let plan = krishiv_optimizer::optimize(&env.analyze(sql)).unwrap();
    let text = DefaultPlanner.plan(&plan, &cx).unwrap().to_string();
    let joins: Vec<&str> = text
        .lines()
        .filter_map(|l| l.find("HashJoin").map(|i| &l[i..]))
        .collect();
    assert_eq!(joins.len(), 1, "{text}");
    assert!(joins[0].starts_with("HashJoin LeftSemi"), "{text}");
    assert!(!joins[0].contains("builds left"), "{text}");
}
