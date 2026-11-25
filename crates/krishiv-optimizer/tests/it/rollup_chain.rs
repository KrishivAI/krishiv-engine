// SPDX-License-Identifier: Apache-2.0
//! Grouping sets computed from their parent sets (`rollup_chain`, `krishiv.optimizer.rollupFromParent`,
//! perf lever F-C2): where it applies and where it refuses, that the output keeps its attributes,
//! that the rows never change and that the plan is a fixed point.

use krishiv_common::config::Config;
use krishiv_optimizer::{ColumnStats, Statistics, TableStats, optimize_with_udfs};
use krishiv_plan::analyzer::{MemoryCatalog, TableColumn};
use krishiv_plan::{Literal, Plan, ReadSource, Scan};
use krishiv_udf_sdk::UdfLibrary;

use crate::support::eval::{self, Db, V};
use crate::support::{self, Rng};

/// `r(a int, b int, c string, d int, w int)`.
const COLUMNS: &[(&str, &str)] = &[
    ("a", "int"),
    ("b", "int"),
    ("c", "string"),
    ("d", "int"),
    ("w", "int"),
];

fn catalog() -> MemoryCatalog {
    let columns = COLUMNS
        .iter()
        .map(|(c, t)| TableColumn {
            name: (*c).to_owned(),
            data_type: krishiv_types::parse_data_type(t).unwrap(),
            nullable: true,
            char_varchar: None,
        })
        .collect();
    MemoryCatalog::new("spark_catalog").with_table(&["default"], "r", columns, None)
}

/// A million rows of `r`, 100 distinct values per column, integers within `[-1000, 1000]`: four
/// keys are nearly unique together, two reduce the rows a hundredfold.
#[derive(Debug)]
struct Stats;

impl Statistics for Stats {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        if name.last()?.as_str() != "r" {
            return None;
        }
        let columns = scan.output.iter().map(|c| {
            let (min, max) = match c.name.as_str() {
                "c" => (None, None),
                _ => (Some(Literal::Integer(-1000)), Some(Literal::Integer(1000))),
            };
            let stats = ColumnStats {
                distinct: Some(100),
                nulls: Some(0),
                min,
                max,
                ..ColumnStats::default()
            };
            (c.name.clone(), stats)
        });
        Some(TableStats {
            rows: 1_000_000,
            columns: columns.collect(),
        })
    }
}

fn config(on: bool) -> Config {
    let mut c = Config::new();
    c.set("krishiv.sql.optimizer.shareCteInstances", "true")
        .unwrap();
    c.set(
        "krishiv.optimizer.rollupFromParent",
        if on { "true" } else { "false" },
    )
    .unwrap();
    c
}

/// Rows with NULL keys and arguments, and duplicate groups at every level.
fn db(seed: u64) -> Db {
    let mut r = Rng(seed | 1);
    let mut db = Db::default();
    let mut rows = Vec::new();
    for _ in 0..120 {
        let key = |r: &mut Rng, n: i64| match r.range(0, n) {
            0 => V::Null,
            k => V::int(k),
        };
        let a = key(&mut r, 3);
        let b = key(&mut r, 4);
        let c = V::Str(r.pick(&["x", "y"]).into());
        let d = key(&mut r, 3);
        let w = match r.range(0, 6) {
            0 => V::Null,
            n => V::int(n * 13 - 40),
        };
        rows.push(vec![a, b, c, d, w]);
    }
    db.tables.insert(
        "r".to_owned(),
        (COLUMNS.iter().map(|c| c.0.to_owned()).collect(), rows),
    );
    db
}

fn fired(p: &Plan) -> bool {
    p.root().to_string().contains("__shared_rollup_")
}

/// Optimizes `sql` with the rule on and off, checking the output's attributes, the fixed point,
/// and that both plans return the unoptimized plan's rows over small data.
fn run(sql: &str) -> (Plan, Plan) {
    let ctx = support::ctx_over(catalog());
    let plan = support::resolve(sql, &ctx);
    let udfs = UdfLibrary::new();
    let on = optimize_with_udfs(&plan, &Stats, &udfs, &config(true)).unwrap();
    assert_eq!(
        on.root().resolved_output(),
        plan.root().resolved_output(),
        "{sql}: the output keeps its attributes\n{}",
        on.root()
    );
    let again = optimize_with_udfs(&on, &Stats, &udfs, &config(true)).unwrap();
    assert!(
        again.structurally_eq(&on),
        "{sql}: a fixed point\n{}\nagain:\n{}",
        on.root(),
        again.root()
    );
    let off = optimize_with_udfs(&plan, &Stats, &udfs, &config(false)).unwrap();
    assert!(!fired(&off), "{sql}: switched off\n{}", off.root());
    for seed in [3, 11, 29, 41] {
        let db = db(seed);
        let want = eval::rows(&db, plan.root());
        assert_eq!(
            eval::rows(&db, on.root()),
            want,
            "{sql} (seed {seed})\n{}",
            on.root()
        );
        assert_eq!(eval::rows(&db, off.root()), want, "{sql} off (seed {seed})");
    }
    // An empty table: a grouped aggregate yields no row, the grand total's set included.
    let mut empty = Db::default();
    empty.tables.insert(
        "r".to_owned(),
        (COLUMNS.iter().map(|c| c.0.to_owned()).collect(), Vec::new()),
    );
    let want = eval::rows(&empty, plan.root());
    assert_eq!(
        eval::rows(&empty, on.root()),
        want,
        "{sql} (empty)\n{}",
        on.root()
    );
    (on, off)
}

fn assert_fires(sql: &str) -> Plan {
    let (on, _) = run(sql);
    assert!(fired(&on), "{sql}\n{}", on.root());
    assert!(
        !on.root().to_string().contains("Expand"),
        "{sql}\n{}",
        on.root()
    );
    on
}

fn assert_stays(sql: &str) {
    let (on, _) = run(sql);
    assert!(!fired(&on), "{sql}\n{}", on.root());
}

/// A `ROLLUP` of four nearly unique keys: each level reads the one above it, so the
/// partial aggregates nest, one per level below the finest.
#[test]
fn a_rollup_reads_each_level_from_the_one_above() {
    let on = assert_fires(
        "SELECT a, b, c, d, sum(w), count(*), count(w), min(c), max(w), avg(w) FROM r \
         GROUP BY ROLLUP(a, b, c, d)",
    );
    let text = on.root().to_string();
    // Five sets: the finest is the pre-aggregate, three partial merges below it, and the grand
    // total reads the one-key level as it is (a keyless partial aggregate would yield a row
    // over empty input).
    assert_eq!(text.matches("PartialAggregate").count(), 4, "{text}");
    assert!(text.contains("Union"), "{text}");
}

/// `CUBE` with `grouping`, `grouping_id`, `HAVING` and `ORDER BY`: each set reads its smallest
/// computed superset.
#[test]
fn a_cube_keeps_its_grouping_ids() {
    assert_fires(
        "SELECT a, b, c, d, grouping_id() AS g, grouping(a) AS ga, count(w) AS n, sum(w) AS t \
         FROM r GROUP BY CUBE(a, b, c, d) HAVING count(w) > 0 ORDER BY g, a, b, c, d",
    );
}

/// Equal sets and the grand total keep their own rows (the expansion numbers its sets).
#[test]
fn repeated_sets_and_the_grand_total_keep_their_rows() {
    assert_fires(
        "SELECT a, b, c, d, sum(w) AS t FROM r \
         GROUP BY GROUPING SETS ((a, b, c, d), (a, b), (a, b), ()) ORDER BY a, b, t",
    );
}

/// Sets with no common finest set: the pre-aggregate's keys are their union.
#[test]
fn disjoint_sets_read_the_pre_aggregate() {
    assert_fires("SELECT a, b, c, d, count(*) AS n FROM r GROUP BY GROUPING SETS ((a, b), (c, d))");
}

/// A grouping expression is a key of the pre-aggregate like a column.
#[test]
fn a_grouping_expression_is_a_key() {
    assert_fires("SELECT a % 2 AS p, b, c, d, sum(w) FROM r GROUP BY ROLLUP(a % 2, b, c, d)");
}

/// Two keys reduce the input a hundredfold: the expansion of the pre-aggregate's groups is
/// already small, and the plan keeps it.
#[test]
fn a_reducing_finest_set_keeps_the_expansion() {
    assert_stays("SELECT a, b, sum(w) FROM r GROUP BY ROLLUP(a, b)");
}

/// The multi-distinct rewrite is an expansion too; its regular call merges below it and each
/// distinct argument's projection reads the states merged by it.
#[test]
fn the_multi_distinct_rewrite_keeps_its_rows() {
    run("SELECT count(DISTINCT a), count(DISTINCT b), count(DISTINCT c), max(w) FROM r");
}

/// One plain `GROUP BY` has no expansion.
#[test]
fn a_plain_group_by_is_left_alone() {
    assert_stays("SELECT a, b, c, d, sum(w) FROM r GROUP BY a, b, c, d");
}
