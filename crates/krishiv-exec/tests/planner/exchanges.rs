// SPDX-License-Identifier: Apache-2.0
//! Exchanges placed by the property pass (ADR 0012): where a node's
//! required distribution is not met, and nowhere else.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_exec::planner::{ExplainNode, MemoryTable, MemoryTables};
use krishiv_optimizer::TableStats;
use krishiv_plan::analyzer::testing::context;
use krishiv_plan::analyzer::{MemoryCatalog, ResolutionConf, TableColumn};

use super::{Env, table};

/// `t` (10 rows, as [`super::env`]) and `big`, the same rows with
/// statistics claiming 100 million: a build of `big` is too large to
/// broadcast.
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
    let catalog = MemoryCatalog::new("spark_catalog")
        .with_table(&["default"], "t", columns.clone(), None)
        .with_table(&["default"], "big", columns, None);
    let big =
        MemoryTable::new(Arc::clone(&schema), batches.clone()).with_statistics(Some(TableStats {
            rows: 100_000_000,
            columns: HashMap::new(),
        }));
    let tables = MemoryTables::new()
        .with_table(
            &["spark_catalog", "default", "t"],
            Arc::new(MemoryTable::new(schema, batches)),
        )
        .with_table(&["spark_catalog", "default", "big"], Arc::new(big));
    Env {
        ctx: context(catalog, ResolutionConf::default()),
        tables: Arc::new(tables),
    }
}

/// The plan of `sql` as `EXPLAIN` prints it.
fn explain(env: &Env, sql: &str) -> String {
    env.plan(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .explain()
        .to_string()
}

/// Every exchange of `n`, depth first, as `name detail`.
fn exchanges(n: &ExplainNode, out: &mut Vec<String>) {
    if n.name.contains("Exchange") {
        out.push(format!("{} {}", n.name, n.detail).trim().to_owned());
    }
    for c in &n.children {
        exchanges(c, out);
    }
}

fn exchanges_of(env: &Env, sql: &str) -> Vec<String> {
    let plan = env.plan(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    exchanges(plan.explain(), &mut out);
    out
}

#[test]
fn a_large_build_is_partitioned_and_a_small_one_broadcast() {
    let env = env();
    insta::assert_snapshot!(explain(&env, "SELECT x.s, y.b FROM big x JOIN big y ON x.a = y.a"), @"
    Project [s#2, b#4]
    +- HashJoin Inner, [a#0] = [a#3], RuntimeFilter #1: a#0 -> spark_catalog.default.big.a
       :- Exchange hashpartitioning(a#0)
       :  +- Project [a#0, s#2]
       :     +- Scan spark_catalog.default.big [a#0, b#1, s#2]
       +- Exchange hashpartitioning(a#3)
          +- Project [a#3, b#4]
             +- Scan spark_catalog.default.big [a#0, b#1, s#2] (ids +3)
    ");
    assert_eq!(
        exchanges_of(&env, "SELECT x.s FROM big x JOIN t y ON x.a = y.a"),
        ["BroadcastExchange"]
    );
}

#[test]
fn hints_choose_the_distribution() {
    let env = env();
    assert_eq!(
        exchanges_of(
            &env,
            "SELECT /*+ SHUFFLE_HASH(y) */ x.s FROM t x JOIN t y ON x.a = y.a"
        ),
        [
            "Exchange hashpartitioning(a#0)",
            "Exchange hashpartitioning(a#3)"
        ]
    );
    assert_eq!(
        exchanges_of(
            &env,
            "SELECT /*+ MERGE(y) */ x.s FROM t x JOIN t y ON x.a = y.a"
        ),
        [
            "Exchange hashpartitioning(a#0)",
            "Exchange hashpartitioning(a#3)"
        ]
    );
    assert_eq!(
        exchanges_of(
            &env,
            "SELECT /*+ BROADCAST(y) */ x.s FROM big x JOIN big y ON x.a = y.a"
        ),
        ["BroadcastExchange"]
    );
}

#[test]
fn the_null_aware_anti_join_always_broadcasts() {
    let env = env();
    assert_eq!(
        exchanges_of(
            &env,
            "SELECT s FROM big WHERE a NOT IN (SELECT /*+ SHUFFLE_HASH */ a FROM big)"
        ),
        ["BroadcastExchange"]
    );
}

#[test]
fn an_aggregate_on_the_join_key_reuses_the_join_partitioning() {
    let env = env();
    // The join placed rows by `x.a` and, equally, by `y.a`: grouping by
    // either needs no exchange; grouping by another column does.
    for key in ["x.a", "y.a"] {
        let sql =
            format!("SELECT {key}, count(*) FROM big x JOIN big y ON x.a = y.a GROUP BY {key}");
        assert_eq!(
            exchanges_of(&env, &sql),
            [
                "Exchange hashpartitioning(a#0)",
                "Exchange hashpartitioning(a#3)"
            ],
            "{sql}"
        );
    }
    let sql = "SELECT x.b, count(*) FROM big x JOIN big y ON x.a = y.a GROUP BY x.b";
    assert_eq!(
        exchanges_of(&env, sql),
        [
            "Exchange hashpartitioning(b#1)",
            "Exchange hashpartitioning(a#0)",
            "Exchange hashpartitioning(a#3)"
        ]
    );
    // A left outer join's build columns are NULL on unmatched rows: only
    // the probe's keys place its output.
    let sql = "SELECT y.a, count(*) FROM big x LEFT JOIN big y ON x.a = y.a GROUP BY y.a";
    assert_eq!(exchanges_of(&env, sql).len(), 3, "{sql}");
}

#[test]
fn a_join_on_an_aggregated_key_exchanges_only_the_other_side() {
    let env = env();
    let sql = "SELECT x.s, g.n FROM big x JOIN \
               (SELECT a, count(*) AS n FROM big GROUP BY a) g ON x.a = g.a";
    insta::assert_snapshot!(explain(&env, sql), @"
    Project [s#2, n#7]
    +- HashJoin Inner, [a#0] = [a#4], RuntimeFilter #1: a#0 -> spark_catalog.default.big.a
       :- Exchange hashpartitioning(a#0)
       :  +- Scan spark_catalog.default.big [a#0, s#2]
       +- Project [a#0, count(1)#4611686018427387904 AS n#3] (ids +4)
          +- HashAggregate keys=[a#0], functions=[count(1)]
             +- Exchange hashpartitioning(a#0)
                +- Project [a#0]
                   +- Scan spark_catalog.default.big [a#0, s#2]
    ");
    // Through a CTE reference, which renames the definition's columns.
    let sql = "WITH g AS (SELECT a, count(*) AS n FROM big GROUP BY a) \
               SELECT x.s, g.n FROM big x JOIN g ON x.a = g.a";
    assert_eq!(
        exchanges_of(&env, sql),
        [
            "Exchange hashpartitioning(a#4)",
            "Exchange hashpartitioning(a#0)"
        ],
        "{sql}"
    );
    // Both sides aggregated on their keys: no exchange above the
    // aggregates.
    let sql = "SELECT g.n, h.n FROM \
               (SELECT a, count(*) AS n FROM big GROUP BY a) g JOIN \
               (SELECT a, count(*) AS n FROM big GROUP BY a) h ON g.a = h.a";
    assert_eq!(exchanges_of(&env, sql).len(), 2, "{sql}");
    // The probe aggregated on its key: only the build is exchanged.
    let sql = "SELECT g.n, y.s FROM (SELECT a, count(*) AS n FROM big GROUP BY a) g \
               JOIN big y ON g.a = y.a";
    let found = exchanges_of(&env, sql);
    assert_eq!(found.len(), 2, "{sql}: {found:?}");
    assert!(
        found.iter().all(|e| e.starts_with("Exchange hash")),
        "{found:?}"
    );
    // Two single partitions are placed alike already.
    let sql = "SELECT /*+ SHUFFLE_HASH(y) */ x.c FROM (SELECT count(*) AS c FROM t) x \
               JOIN (SELECT count(*) AS c FROM t) y ON x.c = y.c";
    assert_eq!(
        exchanges_of(&env, sql),
        ["Exchange SinglePartition", "Exchange SinglePartition"],
        "{sql}"
    );
}

#[test]
fn global_operators_gather_into_one_partition() {
    let env = env();
    assert_eq!(
        exchanges_of(&env, "SELECT count(*) FROM big"),
        ["Exchange SinglePartition"]
    );
    assert_eq!(
        exchanges_of(&env, "SELECT a FROM big LIMIT 3"),
        ["Exchange SinglePartition"]
    );
    assert_eq!(
        exchanges_of(&env, "SELECT a FROM big ORDER BY a DESC"),
        ["Exchange rangepartitioning(a#0 DESC NULLS LAST)"]
    );
    // One row is one partition already; a single partition is clustered.
    assert!(exchanges_of(&env, "SELECT count(*) FROM (SELECT 1 AS x) GROUP BY x").is_empty());
    // A global aggregate's result is one partition: the limit above needs
    // no second gather.
    assert_eq!(
        exchanges_of(&env, "SELECT count(*) FROM big LIMIT 1"),
        ["Exchange SinglePartition"]
    );
}

/// A subquery sees all its rows in one partition: a scalar one must fail
/// on a second row and `EXISTS` must not be false for a partition that
/// holds no match.
#[test]
fn a_subquery_is_gathered_to_one_partition() {
    let env = env();
    for sql in [
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM big WHERE a > 5)",
        "SELECT a FROM t WHERE b > (SELECT b FROM big WHERE a = 1)",
    ] {
        assert_eq!(
            exchanges_of(&env, sql),
            ["Exchange SinglePartition"],
            "{sql}"
        );
    }
}

/// A projection that renames a hash key keeps the placement: a grouping on
/// the alias needs no further exchange.
#[test]
fn an_alias_of_a_hash_key_keeps_the_placement() {
    let env = env();
    let join = "SELECT x.a AS k FROM big x JOIN big y ON x.a = y.a";
    let joined = exchanges_of(&env, join);
    assert_eq!(joined.len(), 2, "{joined:?}");
    assert_eq!(
        exchanges_of(
            &env,
            &format!("SELECT k, count(*) FROM ({join}) GROUP BY k")
        ),
        joined
    );
}

#[test]
fn a_nested_loop_join_broadcasts_its_build() {
    let env = env();
    assert_eq!(
        exchanges_of(&env, "SELECT x.a FROM big x JOIN big y ON x.a < y.a"),
        ["BroadcastExchange"]
    );
}

#[test]
fn exchanges_keep_the_rows() {
    // On one worker an exchange runs as its input: a partitioned join
    // returns the broadcast join's rows.
    let env = env();
    let rows = |hint: &str| {
        let (_, mut rows) = env.rows(&format!(
            "SELECT /*+ {hint}(y) */ x.a, y.s, count(*) FROM t x JOIN t y \
             ON x.a = y.a GROUP BY x.a, y.s"
        ));
        rows.sort();
        rows
    };
    let broadcast = rows("BROADCAST");
    // `a` is NULL for every fourth row (3 and 7), which an inner join drops;
    // each remaining key matches itself once.
    assert_eq!(
        broadcast,
        [
            "0|s0|1", "1|s1|1", "2|s2|1", "4|s4|1", "5|s5|1", "6|s6|1", "8|s8|1", "9|s9|1"
        ]
    );
    assert_eq!(rows("SHUFFLE_HASH"), broadcast);
}
