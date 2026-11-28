// SPDX-License-Identifier: Apache-2.0
//! Eager aggregation's reduction gate reads the fewest groups the keys can make (the largest
//! distinct count of one key), not the product of the counts, which assumes the keys
//! independent (TPC-DS q38/q87: a customer and the dates of their tickets, 5.4M rows into 481k
//! groups where the product says 182M); a key whose own count fails the gate still refuses. A
//! dimension read through a `Shift` (a shared scan's second reader) or a projection renaming
//! its key is still read whole. Rows never change and the plan is a fixed point.

use krishiv_common::config::Config;
use krishiv_optimizer::{
    ColumnStats, Statistics, TableStats, optimize_traced_with_udfs, optimize_with_udfs,
};
use krishiv_plan::analyzer::{MemoryCatalog, TableColumn};
use krishiv_plan::{Literal, Plan, ReadSource, Scan};
use krishiv_udf_sdk::UdfLibrary;

use crate::support::eval::{self, Db, V};
use crate::support::{self, Rng};

/// `f(c, d, u, v)` and `g(c, d, u, v)` two facts, `cust(c, name)` their dimension; every
/// column nullable.
const TABLES: &[(&str, &[(&str, &str)])] = &[
    (
        "f",
        &[("c", "int"), ("d", "int"), ("u", "int"), ("v", "int")],
    ),
    (
        "g",
        &[("c", "int"), ("d", "int"), ("u", "int"), ("v", "int")],
    ),
    ("cust", &[("c", "int"), ("name", "string")]),
];

fn catalog() -> MemoryCatalog {
    catalog_of(TABLES)
}

fn catalog_of(tables: &[(&str, &[(&str, &str)])]) -> MemoryCatalog {
    let mut c = MemoryCatalog::new("spark_catalog");
    for (name, cols) in tables {
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

/// A million rows in each fact over 10,000 customers and 300 days (the product of the two
/// counts, 3M, is above the rows; each count alone is far below a quarter of them), and `u`
/// with 900,000 distinct values; every count known.
#[derive(Debug)]
struct Stats;

impl Statistics for Stats {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let table = name.last()?.as_str();
        let rows = match table {
            "f" | "g" => 1_000_000,
            "cust" => 10_000,
            _ => return None,
        };
        let columns = scan.output.iter().map(|c| {
            let (distinct, lo, hi) = match c.name.as_str() {
                "c" => (10_000, 1, 10_000),
                "d" => (300, 1, 300),
                "u" => (900_000, 1, 900_000),
                "v" => (2_001, -1_000, 1_000),
                _ => (5_000, 0, 0),
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

/// Small tables with NULL keys, repeated (customer, day) pairs and repeated names.
fn db(seed: u64) -> Db {
    let mut r = Rng(seed | 1);
    let mut db = Db::default();
    let maybe = |r: &mut Rng, lo: i64, hi: i64| match r.range(0, 6) {
        0 => V::Null,
        _ => V::int(r.range(lo, hi)),
    };
    let fact = |r: &mut Rng| -> Vec<Vec<V>> {
        (0..80)
            .map(|_| {
                vec![
                    maybe(r, 1, 8),
                    maybe(r, 1, 4),
                    maybe(r, 1, 40),
                    maybe(r, -9, 9),
                ]
            })
            .collect()
    };
    let f = fact(&mut r);
    let g = fact(&mut r);
    let cust = (0..10)
        .map(|_| {
            let name = r.pick(&["ann", "bo", "cy"]);
            vec![maybe(&mut r, 1, 8), V::Str(name.into())]
        })
        .collect();
    for ((name, cols), rows) in TABLES.iter().zip([f, g, cust]) {
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
    /// How many partial aggregates the plan has.
    fn partials(&self) -> usize {
        self.on
            .root()
            .to_string()
            .matches("PartialAggregate")
            .count()
    }
}

/// Optimizes `sql` over the facts and `cust` with eager aggregation on and off.
fn run(sql: &str) -> Run {
    run_over(sql, catalog(), &Stats, db)
}

/// Optimizes `sql` with eager aggregation on and off; checks the output attributes, the fixed
/// point and the rows over small data.
fn run_over(sql: &str, catalog: MemoryCatalog, stats: &dyn Statistics, db: fn(u64) -> Db) -> Run {
    run_with(sql, catalog, stats, db, &[])
}

/// As [`run_over`], with the session settings `conf` on both sides.
fn run_with(
    sql: &str,
    catalog: MemoryCatalog,
    stats: &dyn Statistics,
    db: fn(u64) -> Db,
    conf: &[(&str, &str)],
) -> Run {
    let ctx = support::ctx_over(catalog);
    let plan = support::resolve(sql, &ctx);
    let udfs = UdfLibrary::new();
    let mut config = Config::new();
    for (k, v) in conf {
        config.set(k, v).unwrap();
    }
    let (on, steps) = optimize_traced_with_udfs(&plan, stats, &udfs, &config).unwrap();
    assert_eq!(
        on.root().resolved_output(),
        plan.root().resolved_output(),
        "{sql}: the output keeps its attributes\n{}",
        on.root()
    );
    let again = optimize_with_udfs(&on, stats, &udfs, &config).unwrap();
    assert!(
        again.structurally_eq(&on),
        "{sql}: a fixed point\n{}\nagain:\n{}",
        on.root(),
        again.root()
    );
    let mut off_config = config.clone();
    off_config
        .set("krishiv.optimizer.eagerAggregation", "false")
        .unwrap();
    let off = optimize_with_udfs(&plan, stats, &udfs, &off_config).unwrap();
    for seed in [3, 11, 29, 41, 57] {
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

/// The product of the counts (3M) is above the rows, the largest count (10,000) far below a
/// quarter of them: the distinct and the calls move below the dimension's join.
#[test]
fn correlated_keys_move_below_the_join() {
    for sql in [
        "SELECT DISTINCT name, d FROM f JOIN cust ON f.c = cust.c",
        "SELECT name, d, sum(v), count(v) FROM f JOIN cust ON f.c = cust.c GROUP BY name, d",
    ] {
        let r = run(sql);
        assert_eq!(r.partials(), 1, "{sql}: {:?}\n{}", r.notes, r.on.root());
        assert!(
            r.notes.iter().any(|n| n.starts_with("pushed by ")),
            "{sql}: {:?}",
            r.notes
        );
    }
}

/// A key whose own count is above a quarter of the rows keeps the refusal, and says both
/// bounds.
#[test]
fn a_key_that_alone_fails_the_reduction_still_refuses() {
    let sql = "SELECT DISTINCT name, u FROM f JOIN cust ON f.c = cust.c";
    let r = run(sql);
    assert_eq!(r.partials(), 0, "{sql}: {:?}\n{}", r.notes, r.on.root());
    assert!(
        r.notes
            .iter()
            .any(|n| n.contains("reduction below 4") && n.contains("groups 900000 to")),
        "{sql}: {:?}",
        r.notes
    );
}

/// TPC-DS q38's tables, the columns it reads (`d_date` as an integer: the interpreter's rows
/// need no calendar).
const Q38_TABLES: &[(&str, &[(&str, &str)])] = &[
    (
        "store_sales",
        &[("ss_sold_date_sk", "int"), ("ss_customer_sk", "int")],
    ),
    (
        "catalog_sales",
        &[("cs_sold_date_sk", "int"), ("cs_bill_customer_sk", "int")],
    ),
    (
        "web_sales",
        &[("ws_sold_date_sk", "int"), ("ws_bill_customer_sk", "int")],
    ),
    (
        "date_dim",
        &[
            ("d_date_sk", "int"),
            ("d_date", "int"),
            ("d_month_seq", "int"),
        ],
    ),
    (
        "customer",
        &[
            ("c_customer_sk", "int"),
            ("c_last_name", "string"),
            ("c_first_name", "string"),
            ("c_birth_year", "int"),
        ],
    ),
];

/// SF10-like counts: millions of sales rows over 500,000 customers and 73,049 dates, 12
/// months of which q38 keeps.
#[derive(Debug)]
struct Q38Stats;

impl Statistics for Q38Stats {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let table = name.last()?.as_str();
        let rows = match table {
            "store_sales" => 28_800_000,
            "catalog_sales" => 14_400_000,
            "web_sales" => 7_200_000,
            "date_dim" => 73_049,
            "customer" => 500_000,
            _ => return None,
        };
        let columns = scan.output.iter().map(|c| {
            let n = c.name.as_str();
            // Sales fall on five years of dates (as TPC-DS's do); the dimension has 200 years.
            let (distinct, lo, hi) = if n.ends_with("sold_date_sk") {
                (1_823, 2_450_816, 2_452_642)
            } else if n == "d_date_sk" || n == "d_date" {
                (73_049, 2_415_022, 2_488_070)
            } else if n.ends_with("customer_sk") {
                (500_000, 1, 500_000)
            } else if n == "d_month_seq" {
                (2_400, 0, 2_400)
            } else if n == "c_birth_year" {
                (69, 1_924, 1_992)
            } else {
                (5_000, 0, 0)
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

/// Small q38 tables: sales on a few days in and out of q38's months, NULL keys, shared names.
fn q38_db(seed: u64) -> Db {
    let mut r = Rng(seed | 1);
    let mut db = Db::default();
    let maybe = |r: &mut Rng, lo: i64, hi: i64| match r.range(0, 7) {
        0 => V::Null,
        _ => V::int(r.range(lo, hi)),
    };
    let sales = |r: &mut Rng| -> Vec<Vec<V>> {
        (0..60)
            .map(|_| vec![maybe(r, 1, 6), maybe(r, 1, 7)])
            .collect()
    };
    let ss = sales(&mut r);
    let cs = sales(&mut r);
    let ws = sales(&mut r);
    let dates = (1..6)
        .map(|d| vec![V::int(d), V::int(d % 3), V::int(1_198 + 2 * d)])
        .collect();
    let customers = (1..7)
        .map(|c| {
            let last = r.pick(&["ann", "bo"]);
            let first = r.pick(&["cy", "di"]);
            vec![
                V::int(c),
                V::Str(last.into()),
                V::Str(first.into()),
                V::int(1_975 + 3 * c),
            ]
        })
        .collect();
    for ((name, cols), rows) in Q38_TABLES.iter().zip([ss, cs, ws, dates, customers]) {
        let names = cols.iter().map(|c| c.0.to_owned()).collect();
        db.tables.insert((*name).to_owned(), (names, rows));
    }
    db
}

/// TPC-DS q38 itself: `ShareCte` reads `customer` once, so the catalog and web channels join it
/// through a `Shift` and a projection renaming its key. It is still read whole, and the store
/// and catalog channels' distincts move below their customer joins (with the product of the
/// counts as the groups, 500,000 x 335 dates, neither would).
#[test]
fn tpcds_q38_moves_the_store_and_catalog_distincts_below_their_customer_joins() {
    let r = run_over(&q38(), catalog_of(Q38_TABLES), &Q38Stats, q38_db);
    let text = r.on.root().to_string();
    assert!(
        !r.notes
            .iter()
            .any(|n| n.contains("the other side is filtered or joined")),
        "{:?}\n{text}",
        r.notes
    );
    // Store and catalog: about 500,000 customers over 5.3M and 2.6M rows. Web: 473,000
    // customers over 1.3M rows cannot reduce 4 times even if every customer bought on one day.
    assert_eq!(r.partials(), 2, "{:?}\n{text}", r.notes);
    assert!(
        r.notes
            .iter()
            .any(|n| n.starts_with("not applied: reduction below 4")
                && n.contains("groups 473089 to")),
        "{:?}",
        r.notes
    );
}

/// The q38 text, without its final `;`.
fn q38() -> String {
    include_str!("../../../../bench/queries/tpcds/q38.sql")
        .trim_end_matches([';', '\n', ' '])
        .to_owned()
}

/// A dimension that is really filtered stays refused when it is read through a `Shift`: q38
/// with `c_birth_year > 1925` in every channel (it keeps about 97% of the customers, so the
/// join's retention gate passes), so the catalog and web channels read the filtered `customer`
/// through `Shift`s of one shared body, and only the read-whole check refuses.
#[test]
fn a_filtered_dimension_read_through_a_shift_is_still_refused() {
    let sql = q38().replace(
        "d_month_seq BETWEEN 1200 AND 1200 + 11",
        "d_month_seq BETWEEN 1200 AND 1200 + 11 AND c_birth_year > 1925",
    );
    let r = run_over(&sql, catalog_of(Q38_TABLES), &Q38Stats, q38_db);
    let text = r.on.root().to_string();
    assert!(text.contains("Shift"), "{text}");
    assert_eq!(r.partials(), 0, "{:?}\n{text}", r.notes);
    assert!(
        r.notes
            .iter()
            .any(|n| n.contains("the other side is filtered or joined"))
            && !r.notes.iter().any(|n| n.contains("retention below")),
        "{:?}\n{text}",
        r.notes
    );
}

/// A dimension whose key a projection renames is read under the input's id: a not-NULL filter
/// on another of its columns still refuses (it drops rows with a key).
#[test]
fn a_renamed_key_over_a_filter_on_another_column_is_still_refused() {
    let sql = "SELECT DISTINCT name, d FROM f JOIN \
               (SELECT c AS k, name FROM cust WHERE name IS NOT NULL) t ON f.c = t.k";
    let r = run(sql);
    assert_eq!(r.partials(), 0, "{sql}: {:?}\n{}", r.notes, r.on.root());
    assert!(
        r.notes
            .iter()
            .any(|n| n.contains("the other side is filtered or joined")),
        "{sql}: {:?}\n{}",
        r.notes,
        r.on.root()
    );
}

/// A side already grouped on the keys (a `GROUP BY` or `DISTINCT` subquery) is not grouped again
/// below the join: its rows are distinct on them already (review of abe68c926, finding 1).
#[test]
fn a_side_already_grouped_on_the_keys_is_refused() {
    for sql in [
        "SELECT name, t.d, count(*) FROM (SELECT c, d FROM f GROUP BY c, d) t \
         JOIN cust ON t.c = cust.c GROUP BY name, t.d",
        "SELECT DISTINCT name, t.d FROM (SELECT DISTINCT c, d FROM f) t JOIN cust ON t.c = cust.c",
    ] {
        let r = run(sql);
        assert_eq!(r.partials(), 0, "{sql}: {:?}\n{}", r.notes, r.on.root());
        assert!(
            r.notes
                .iter()
                .any(|n| n.contains("the side is already grouped on some of these keys")),
            "{sql}: {:?}",
            r.notes
        );
    }
}

/// Deterministic mode grows its thread-local tables, which never turn append-only, so the gate
/// reads the product of the counts there: q38's distincts stay above their joins.
#[test]
fn deterministic_mode_reads_the_product_of_the_counts() {
    let r = run_with(
        &q38(),
        catalog_of(Q38_TABLES),
        &Q38Stats,
        q38_db,
        &[("krishiv.deterministicOrder", "true")],
    );
    assert_eq!(r.partials(), 0, "{:?}\n{}", r.notes, r.on.root());
    assert!(
        r.notes
            .iter()
            .any(|n| n.starts_with("not applied: reduction below 4")),
        "{:?}",
        r.notes
    );
}

/// The other half of the rename: a dimension read through a projection that renames its key,
/// with only the inferred not-NULL filter on that key, is read whole and the distinct moves.
#[test]
fn a_renamed_key_with_only_its_null_filter_is_read_whole() {
    let sql = "SELECT DISTINCT name, d FROM f JOIN (SELECT c AS k, name FROM cust) t ON f.c = t.k";
    let r = run(sql);
    assert_eq!(r.partials(), 1, "{sql}: {:?}\n{}", r.notes, r.on.root());
}
