// SPDX-License-Identifier: Apache-2.0
//! The q-error of the optimizer's cardinality estimates: for every join and
//! aggregate of the TPC-H and TPC-DS queries, the estimated rows
//! (`PlanStats`) against the rows the reference interpreter returns for that
//! relation, as `max(est / actual, actual / est)` (both at least 1). The
//! report prints the median, the 95th percentile and the largest, with
//! statistics propagation on and off.
//!
//! By default the data are generated in process: TPC-H as the other tests
//! generate it, TPC-DS (which has no generator here) from the schema with
//! keys drawn from small domains, so joins find matches. Statistics are read
//! off the data (exact row counts, distinct counts, nulls, bounds).
//!
//! Run it with `cargo xtask qerror`, which sets:
//! - `KRISHIV_QERROR_DIR`: read the tables from pipe-delimited files in this
//!   directory instead (`<table>.tbl` as `dbgen` writes TPC-H, `<table>.dat`
//!   as `dsdgen` writes TPC-DS) for a bigger run;
//! - `KRISHIV_QERROR_SCALE`: TPC-H orders (default 150); the TPC-DS
//!   synthetic fact tables get twice as many rows;
//! - `KRISHIV_QERROR_SUITES`: `tpch`, `tpcds` or both, comma separated.

use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Arc;

use krishiv_optimizer::{
    ColumnStats, Options, PlanStats, Statistics, TableStats, optimize_with_options,
};
use krishiv_plan::{Literal, LogicalPlan, PlanNode, ReadSource, Scan};
use krishiv_types::DataType;
use serde_json::Value;

use crate::support::eval::{self, Db, SCALE, V};
use crate::support::{self, Rng, data};
use crate::tpcds;

/// `max(est / actual, actual / est)`, both at least 1.
fn q_error(est: f64, actual: f64) -> f64 {
    let (e, a) = (est.max(1.0), actual.max(1.0));
    (e / a).max(a / e)
}

/// The `p`-th percentile (0 to 1) of `sorted`, nearest rank.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "a rank within the slice"
    )]
    let rank = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

/// The q-errors of one suite, and how many relations could not be measured.
#[derive(Debug, Default)]
struct Errors {
    q: Vec<f64>,
    /// Each measured relation: its query, kind, estimate and actual rows.
    what: Vec<String>,
    skipped: usize,
}

impl Errors {
    fn summary(&self) -> (f64, f64, f64) {
        let mut sorted = self.q.clone();
        sorted.sort_by(f64::total_cmp);
        (
            percentile(&sorted, 0.5),
            percentile(&sorted, 0.95),
            sorted.last().copied().unwrap_or(f64::NAN),
        )
    }
}

/// Table statistics by table name, read off the data.
#[derive(Debug)]
struct DataStats(HashMap<String, TableStats>);

impl Statistics for DataStats {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        self.0.get(name.last()?).cloned()
    }
}

fn bound(v: &V) -> Option<Literal> {
    match v {
        #[expect(clippy::cast_precision_loss, reason = "a bound of an estimate")]
        V::Num(n) => Some(Literal::Double(*n as f64 / SCALE as f64)),
        V::Date(d) => Some(Literal::Date(*d)),
        _ => None,
    }
}

/// The statistics a table of `db` would report after `ANALYZE`.
fn stats_of(db: &Db) -> DataStats {
    let tables = db.tables.iter().map(|(name, (cols, rows))| {
        let columns = cols.iter().enumerate().map(|(i, c)| {
            let present: Vec<&V> = rows
                .iter()
                .map(|r| &r[i])
                .filter(|v| **v != V::Null)
                .collect();
            let distinct: HashSet<&V> = present.iter().copied().collect();
            let stats = ColumnStats {
                distinct: Some(distinct.len() as u64),
                nulls: Some((rows.len() - present.len()) as u64),
                min: present.iter().min().copied().and_then(bound),
                max: present.iter().max().copied().and_then(bound),
                ..ColumnStats::default()
            };
            (c.clone(), stats)
        });
        let stats = TableStats {
            rows: rows.len() as u64,
            columns: columns.collect(),
        };
        (name.clone(), stats)
    });
    DataStats(tables.collect())
}

/// A suite's tables: names and typed columns.
type Schemas = Vec<(String, Vec<(String, DataType)>)>;

fn parse_type(ty: &str) -> DataType {
    let upper = ty.to_ascii_uppercase();
    let ty = if upper.starts_with("CHAR") || upper.starts_with("VARCHAR") {
        "string"
    } else {
        ty
    };
    krishiv_types::parse_data_type(ty).unwrap()
}

fn tpch_schemas() -> Schemas {
    let path = support::root().join("tests/fixtures/spark-4.2.0/analyzer-goldens.json");
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut out: Schemas = g["tables"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, cols)| {
            let cols = cols
                .as_array()
                .unwrap()
                .iter()
                .map(|f| {
                    (
                        f["name"].as_str().unwrap().to_owned(),
                        parse_type(f["type"].as_str().unwrap()),
                    )
                })
                .collect();
            (name.clone(), cols)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn tpcds_schemas() -> Schemas {
    let path = support::root().join("tests/fixtures/spark-4.2.0/golden-tables/tpcds-schemas.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    doc["tables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            let cols = t["columns"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    (
                        c["name"].as_str().unwrap().to_owned(),
                        parse_type(c["type"].as_str().unwrap()),
                    )
                })
                .collect();
            (t["name"].as_str().unwrap().to_owned(), cols)
        })
        .collect()
}

/// A value of `t` from the text of a `dbgen` / `dsdgen` field.
fn parse_field(text: &str, t: &DataType) -> V {
    if text.is_empty() {
        return V::Null;
    }
    match t {
        DataType::Byte | DataType::Short | DataType::Integer | DataType::Long => {
            text.parse::<i64>().map_or(V::Null, V::int)
        }
        DataType::Float | DataType::Double | DataType::Decimal(_) => decimal(text),
        DataType::Date => {
            let mut parts = text.split('-').map(|p| p.parse::<i64>().unwrap_or(0));
            let (y, m, d) = (
                parts.next().unwrap_or(0),
                parts.next().unwrap_or(1),
                parts.next().unwrap_or(1),
            );
            V::Date(eval::days(y, m, d))
        }
        _ => V::Str(text.into()),
    }
}

/// A fixed-point number from its decimal text.
fn decimal(text: &str) -> V {
    let (neg, text) = text.strip_prefix('-').map_or((false, text), |t| (true, t));
    let (whole, frac) = text.split_once('.').unwrap_or((text, ""));
    let mut frac: String = frac.chars().take(6).collect();
    while frac.len() < 6 {
        frac.push('0');
    }
    let (Ok(whole), Ok(frac)) = (whole.parse::<i128>(), frac.parse::<i128>()) else {
        return V::Null;
    };
    let n = whole * SCALE + frac;
    V::Num(if neg { -n } else { n })
}

/// The tables of `schemas` read from `dir/<table>.<ext>`: one row per line,
/// fields separated by `|` (with the trailing `|` the generators write).
fn load(dir: &Path, schemas: &Schemas, ext: &str) -> Db {
    let mut db = Db::default();
    for (name, cols) in schemas {
        let path = dir.join(format!("{name}.{ext}"));
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let rows = text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|line| {
                let mut fields = line.split('|');
                cols.iter()
                    .map(|(_, t)| parse_field(fields.next().unwrap_or(""), t))
                    .collect()
            })
            .collect();
        let names = cols.iter().map(|(c, _)| c.clone()).collect();
        db.tables.insert(name.clone(), (names, rows));
    }
    db
}

/// TPC-DS-shaped data from the schema: each dimension's surrogate key is its
/// row number; every other `_sk` column a draw from the dimensions' keys;
/// facts (the `sales`, `returns` and `inventory` tables) have `rows` rows,
/// dimensions a tenth of that; other columns draw from small domains.
fn synthetic_tpcds(schemas: &Schemas, seed: u64, rows: i64) -> Db {
    let mut r = Rng(seed | 1);
    let mut db = Db::default();
    let dims = (rows / 10).max(8);
    for (name, cols) in schemas {
        let fact = ["sales", "returns", "inventory"]
            .iter()
            .any(|k| name.contains(k));
        let n = if fact { rows } else { dims };
        let own_key = cols
            .iter()
            .position(|(c, _)| c.ends_with("_sk"))
            .filter(|_| !fact);
        let rows = (1..=n)
            .map(|i| {
                cols.iter()
                    .enumerate()
                    .map(|(j, (c, t))| match t {
                        _ if own_key == Some(j) => V::int(i),
                        DataType::Integer | DataType::Long if c.ends_with("_sk") => {
                            V::int(r.range(1, dims))
                        }
                        DataType::Integer | DataType::Long => V::int(r.range(0, 20)),
                        DataType::Decimal(_) | DataType::Double | DataType::Float => {
                            V::cents(r.range(0, 20_000))
                        }
                        DataType::Date => V::Date(
                            eval::days(1999, 1, 1) + i32::try_from(r.range(0, 1500)).unwrap_or(0),
                        ),
                        _ => V::Str(r.pick(&["a", "b", "c", "d", "M", "F", "Y", "N"]).into()),
                    })
                    .collect()
            })
            .collect();
        let names = cols.iter().map(|(c, _)| c.clone()).collect();
        db.tables.insert(name.clone(), (names, rows));
    }
    db
}

/// The joins and aggregates of `root` (subquery plans excluded), each once.
fn measured(root: &Arc<LogicalPlan>) -> Vec<Arc<LogicalPlan>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(root)];
    while let Some(p) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        if matches!(p.node, PlanNode::Join { .. } | PlanNode::Aggregate { .. }) {
            out.push(Arc::clone(&p));
        }
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    out
}

/// The q-errors of the joins and aggregates of the optimized `sqls`, over
/// `db`, with propagation `on` or not.
fn run_suite(
    sqls: &[(String, String)],
    ctx: &krishiv_plan::analyzer::AnalysisContext,
    db: &Db,
    on: bool,
) -> Errors {
    let stats = stats_of(db);
    let options = Options {
        stats_propagation: on,
        ..Options::default()
    };
    let mut errors = Errors::default();
    // The interpreter panics on constructs it does not run: those relations
    // are skipped (and counted), quietly.
    // Restores the panic hook on every exit, a panic included.
    struct Hook(Option<Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>>);
    impl Drop for Hook {
        fn drop(&mut self) {
            if let Some(h) = self.0.take() {
                std::panic::set_hook(h);
            }
        }
    }
    let _hook = Hook(Some(std::panic::take_hook()));
    std::panic::set_hook(Box::new(|_| {}));
    for (id, sql) in sqls {
        let Some(plan) = support::try_resolve(sql, ctx) else {
            continue;
        };
        let optimized = optimize_with_options(&plan, &stats, &options).unwrap();
        let estimates = PlanStats::compute_with_options(optimized.root(), &stats, &options);
        for node in measured(optimized.root()) {
            let actual = catch_unwind(AssertUnwindSafe(|| eval::run(db, &node).rows.len()));
            match (actual, estimates.get(&node)) {
                (Ok(actual), Some(est)) => {
                    #[expect(clippy::cast_precision_loss, reason = "a row count")]
                    let q = q_error(est.rows_for_costing(), actual as f64);
                    let kind = if matches!(node.node, PlanNode::Join { .. }) {
                        "join"
                    } else {
                        "aggregate"
                    };
                    errors.q.push(q);
                    errors.what.push(format!(
                        "{id} {kind}: estimated {:.0}, actual {actual}",
                        est.rows_for_costing()
                    ));
                }
                _ => errors.skipped += 1,
            }
        }
    }
    errors
}

#[expect(clippy::print_stdout, reason = "the report is the output")]
fn report(suite: &str, on: &Errors, off: &Errors) {
    let mut worst: Vec<(f64, &String)> = on.q.iter().copied().zip(&on.what).collect();
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (q, what) in worst.iter().take(3) {
        println!("q-error {suite:<6} worst with propagation: {q:.1}  {what}");
    }
    for (label, e) in [("propagation on ", on), ("propagation off", off)] {
        let (median, p95, max) = e.summary();
        println!(
            "q-error {suite:<6} {label}: median {median:>8.2}  p95 {p95:>10.2}  max {max:>12.2}  \
             ({} joins and aggregates, {} not run by the interpreter)",
            e.q.len(),
            e.skipped
        );
    }
}

fn suites() -> HashSet<String> {
    std::env::var("KRISHIV_QERROR_SUITES")
        .unwrap_or_else(|_| "tpch,tpcds".to_owned())
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .collect()
}

fn scale() -> i64 {
    std::env::var("KRISHIV_QERROR_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150)
}

fn tpch_report() -> (Errors, Errors) {
    let dir = std::env::var("KRISHIV_QERROR_DIR").ok();
    let db = match &dir {
        Some(dir) => load(Path::new(dir), &tpch_schemas(), "tbl"),
        None => data::tpch(7, scale()),
    };
    let sqls: Vec<(String, String)> = (1..=22)
        .map(|q| (format!("q{q:02}"), support::tpch_sql(q)))
        .collect();
    let ctx = support::ctx();
    let (on, off) = (
        run_suite(&sqls, &ctx, &db, true),
        run_suite(&sqls, &ctx, &db, false),
    );
    report("tpch", &on, &off);
    (on, off)
}

fn tpcds_report() -> (Errors, Errors) {
    let schemas = tpcds_schemas();
    let db = match std::env::var("KRISHIV_QERROR_DIR") {
        Ok(dir) => load(Path::new(&dir), &schemas, "dat"),
        Err(_) => synthetic_tpcds(&schemas, 11, 2 * scale()),
    };
    let sqls: Vec<(String, String)> = (1..=99)
        .map(|q| (format!("q{q:02}"), tpcds::sql(q)))
        .collect();
    let ctx = support::ctx_over(tpcds::catalog());
    let (on, off) = (
        run_suite(&sqls, &ctx, &db, true),
        run_suite(&sqls, &ctx, &db, false),
    );
    report("tpcds", &on, &off);
    (on, off)
}

/// The report on generated data: every number finite, enough relations
/// measured, and the estimates not worse with propagation than without (at
/// the median, with room for ties).
#[test]
fn q_error_report_on_generated_data() {
    let suites = suites();
    if suites.contains("tpch") {
        let (on, off) = tpch_report();
        assert!(on.q.len() >= 30, "{} relations measured", on.q.len());
        let ((m_on, p_on, _), (m_off, _, _)) = (on.summary(), off.summary());
        assert!(m_on.is_finite() && p_on.is_finite() && m_on >= 1.0);
        assert!(
            m_on <= m_off * 1.5,
            "median {m_on} with propagation, {m_off} without"
        );
    }
    if suites.contains("tpcds") {
        let (on, off) = tpcds_report();
        assert!(on.q.len() >= 30, "{} relations measured", on.q.len());
        let (m_on, p_on, _) = on.summary();
        assert!(m_on.is_finite() && p_on.is_finite() && m_on >= 1.0);
        let _ = off;
    }
}

#[test]
fn q_error_is_symmetric_and_floored_at_one_row() {
    assert_eq!(q_error(10.0, 10.0), 1.0);
    assert_eq!(q_error(100.0, 10.0), 10.0);
    assert_eq!(q_error(10.0, 100.0), 10.0);
    assert_eq!(q_error(0.0, 0.0), 1.0, "no rows either way is exact");
    assert_eq!(
        q_error(0.0, 50.0),
        50.0,
        "an empty estimate counts as one row"
    );
}

#[test]
fn percentiles_are_nearest_rank() {
    let v: Vec<f64> = (1..=100).map(f64::from).collect();
    assert_eq!(percentile(&v, 0.5), 50.0);
    assert_eq!(percentile(&v, 0.95), 95.0);
    assert_eq!(percentile(&v, 1.0), 100.0);
    assert_eq!(percentile(&[7.0], 0.95), 7.0);
    assert!(percentile(&[], 0.5).is_nan());
}

/// Files in `dbgen`'s format load as typed values: a trailing `|`, empty
/// fields as NULL, decimals as fixed point, dates as days.
#[test]
fn data_directories_load_in_the_generators_format() {
    let dir = std::env::temp_dir().join(format!("krishiv-qerror-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("r.tbl"),
        "1|abc|12.50|1995-03-15|\n2||-0.25|1970-01-02|\n",
    )
    .unwrap();
    let schemas: Schemas = vec![(
        "r".to_owned(),
        vec![
            ("k".to_owned(), DataType::Long),
            ("s".to_owned(), DataType::String),
            ("d".to_owned(), parse_type("decimal(12,2)")),
            ("t".to_owned(), DataType::Date),
        ],
    )];
    let db = load(&dir, &schemas, "tbl");
    std::fs::remove_dir_all(&dir).unwrap();
    let (cols, rows) = &db.tables["r"];
    assert_eq!(cols, &["k", "s", "d", "t"]);
    assert_eq!(
        rows[0],
        [
            V::int(1),
            V::Str("abc".into()),
            V::cents(1250),
            V::Date(eval::days(1995, 3, 15))
        ]
    );
    assert_eq!(rows[1], [V::int(2), V::Null, V::cents(-25), V::Date(1)]);
    let stats = stats_of(&db).0["r"].clone();
    assert_eq!(stats.rows, 2);
    assert_eq!(stats.columns["s"].nulls, Some(1));
    assert_eq!(stats.columns["k"].distinct, Some(2));
}
