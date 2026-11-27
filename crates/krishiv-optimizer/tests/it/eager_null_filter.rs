// SPDX-License-Identifier: Apache-2.0
//! Eager aggregation over a dimension whose only filter is an inferred `isnotnull` on its join
//! key (study 54, lever F-C1, TPC-DS q22): the dimension still counts as read whole, on the direct
//! and the rotated path; any other filter still refuses. Rows never change and the plan is a fixed
//! point.

use krishiv_common::config::Config;
use krishiv_optimizer::{
    ColumnStats, Statistics, TableStats, optimize_traced_with_udfs, optimize_with_udfs,
};
use krishiv_plan::analyzer::{MemoryCatalog, TableColumn};
use krishiv_plan::{Literal, Plan, ReadSource, Scan};
use krishiv_udf_sdk::UdfLibrary;

use crate::support::eval::{self, Db, V};
use crate::support::{self, Rng};

/// `f(k, d, v)` the fact, `item(k, a)` and `dt(d, y)` its dimensions; every column nullable.
const TABLES: &[(&str, &[(&str, &str)])] = &[
    ("f", &[("k", "int"), ("d", "int"), ("v", "int")]),
    ("item", &[("k", "int"), ("a", "string")]),
    ("dt", &[("d", "int"), ("y", "int")]),
];

fn catalog() -> MemoryCatalog {
    let mut c = MemoryCatalog::new("spark_catalog");
    for (name, cols) in TABLES {
        let columns = cols
            .iter()
            .map(|(c, t)| TableColumn {
                name: (*c).to_owned(),
                data_type: krishiv_types::parse_data_type(t).unwrap(),
                nullable: true,
                char_varchar: None,
            })
            .collect();
        c = c.with_table(&["default"], name, columns, None);
    }
    c
}

/// A million fact rows over 1,000 items and 100 days; every count known.
#[derive(Debug)]
struct Stats;

impl Statistics for Stats {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let table = name.last()?.as_str();
        let rows = match table {
            "f" => 1_000_000,
            "item" => 1_000,
            "dt" => 100,
            _ => return None,
        };
        let columns = scan.output.iter().map(|c| {
            let (distinct, lo, hi) = match (table, c.name.as_str()) {
                (_, "k") => (1_000, 1, 1_000),
                (_, "d") => (100, 1, 100),
                (_, "y") => (10, 1, 10),
                (_, "v") => (2_001, -1_000, 1_000),
                _ => (50, 0, 0),
            };
            let (min, max) = if lo < hi {
                (Some(Literal::Integer(lo)), Some(Literal::Integer(hi)))
            } else {
                (None, None)
            };
            let stats = ColumnStats {
                distinct: Some(distinct),
                nulls: Some(0),
                min,
                max,
                ..ColumnStats::default()
            };
            (c.name.clone(), stats)
        });
        Some(TableStats {
            rows,
            columns: columns.collect(),
        })
    }
}

/// Small tables with NULL keys on both sides and duplicate groups.
fn db(seed: u64) -> Db {
    let mut r = Rng(seed | 1);
    let mut db = Db::default();
    let maybe = |r: &mut Rng, lo: i64, hi: i64| match r.range(0, 5) {
        0 => V::Null,
        _ => V::int(r.range(lo, hi)),
    };
    let fact = (0..90)
        .map(|_| {
            vec![
                maybe(&mut r, 1, 8),
                maybe(&mut r, 1, 5),
                maybe(&mut r, -9, 9),
            ]
        })
        .collect();
    let item = (0..10)
        .map(|_| {
            let a = r.pick(&["x", "y", "z"]);
            vec![maybe(&mut r, 1, 8), V::Str(a.into())]
        })
        .collect();
    let dt = (0..6)
        .map(|_| vec![maybe(&mut r, 1, 5), maybe(&mut r, 1, 3)])
        .collect();
    for ((name, cols), rows) in TABLES.iter().zip([fact, item, dt]) {
        let names = cols.iter().map(|c| c.0.to_owned()).collect();
        db.tables.insert((*name).to_owned(), (names, rows));
    }
    db
}

struct Run {
    on: Plan,
    notes: Vec<String>,
}

impl Run {
    fn fired(&self) -> bool {
        self.on.root().to_string().contains("PartialAggregate")
    }
}

/// Optimizes `sql` with eager aggregation on and off; checks the output attributes, the fixed
/// point and the rows over small data.
fn run(sql: &str) -> Run {
    let ctx = support::ctx_over(catalog());
    let plan = support::resolve(sql, &ctx);
    let udfs = UdfLibrary::new();
    let (on, steps) = optimize_traced_with_udfs(&plan, &Stats, &udfs, &Config::new()).unwrap();
    assert_eq!(
        on.root().resolved_output(),
        plan.root().resolved_output(),
        "{sql}: the output keeps its attributes\n{}",
        on.root()
    );
    let again = optimize_with_udfs(&on, &Stats, &udfs, &Config::new()).unwrap();
    assert!(
        again.structurally_eq(&on),
        "{sql}: a fixed point\n{}\nagain:\n{}",
        on.root(),
        again.root()
    );
    let mut off_config = Config::new();
    off_config
        .set("krishiv.optimizer.eagerAggregation", "false")
        .unwrap();
    let off = optimize_with_udfs(&plan, &Stats, &udfs, &off_config).unwrap();
    for seed in [3, 11, 29, 41] {
        let db = db(seed);
        let want = eval::rows(&db, plan.root());
        assert_eq!(eval::rows(&db, on.root()), want, "{sql} (seed {seed})");
        assert_eq!(eval::rows(&db, off.root()), want, "{sql} off (seed {seed})");
    }
    let notes = steps
        .into_iter()
        .filter(|s| s.rule == "EagerAggregate" && !s.note.is_empty())
        .map(|s| s.note)
        .collect();
    Run { on, notes }
}

/// The dimension carries the `isnotnull(k)` inferred join filters add, and the calls still move.
#[test]
fn an_inferred_null_filter_on_the_join_key_reads_the_dimension_whole() {
    let r = run("SELECT a, sum(v), count(*) FROM f JOIN item ON f.k = item.k GROUP BY a");
    let text = r.on.root().to_string();
    assert!(
        text.contains("isnotnull("),
        "the inferred filter is there\n{text}"
    );
    assert!(r.fired(), "{:?}\n{text}", r.notes);
    assert!(
        r.notes.iter().any(|n| n.starts_with("pushed by [k]: ")),
        "{:?}",
        r.notes
    );
}

/// q22's shape: the calls read the fact, one dimension is really filtered, the other has only
/// its null filter. Whichever join ordering puts on top, the calls move (directly below `item`,
/// or after the rotation brings `item` to the top).
#[test]
fn a_filtered_dimension_beside_a_null_filtered_one_still_moves_the_calls() {
    let r = run(
        "SELECT a, sum(v), count(v) FROM f JOIN dt ON f.d = dt.d JOIN item ON f.k = item.k \
         WHERE dt.y = 3 GROUP BY a",
    );
    assert!(r.fired(), "{:?}\n{}", r.notes, r.on.root());
}

/// A filter that drops rows with a key, or tests a column the join does not compare, keeps the
/// refusal: its runtime filter can prune the fact.
#[test]
fn other_filters_on_the_dimension_still_refuse() {
    for sql in [
        "SELECT a, sum(v) FROM f JOIN item ON f.k = item.k WHERE item.a = 'x' GROUP BY a",
        "SELECT a, sum(v) FROM f JOIN item ON f.k = item.k WHERE item.a IS NOT NULL GROUP BY a",
        "SELECT a, sum(v) FROM f JOIN item ON f.k = item.k WHERE item.k > 3 GROUP BY a",
    ] {
        let r = run(sql);
        assert!(!r.fired(), "{sql}: {:?}\n{}", r.notes, r.on.root());
        assert!(
            r.notes
                .iter()
                .any(|n| n.contains("the other side is filtered or joined")),
            "{sql}: {:?}",
            r.notes
        );
    }
}
