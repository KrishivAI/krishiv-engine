// SPDX-License-Identifier: Apache-2.0
//! The per-rule differential of design P2-T072b §6 item 2 through the
//! engine: each TPC-H and TPC-DS query is optimized with every rule on and
//! with one switched rule off, planned and run by the executor over Parquet
//! files, and the two results must be equal as multisets of rows, doubles
//! within 1e-9 relative (absolute below magnitude 1). The goldens compare
//! rendered text; a rule that reorders a double aggregation may change its
//! last digits, so this tolerance stands in for theirs. Only arms whose
//! optimized plan differs from the all-on plan run.
//!
//! The SF1 tests do not require every rule to change a plan in each suite:
//! eager aggregation and late payload joins change TPC-H plans only, group
//! key codes TPC-DS ones. On 2026-11-16 every rule changed at least one plan
//! across the two (logs in the row 19 notes).
//!
//! The optimizer crate's `rule_differential` runs the same arms through the
//! reference interpreter on small generated data; this one runs the design's
//! SF1 (`KRISHIV_TPCH_SF1_DIR`, `KRISHIV_TPCDS_SF1_DIR`: `<table>.parquet`
//! per table; ignored tests) and TPC-H SF 0.01 from the fixture by default.
//! `KRISHIV_RULE_DIFF_QUERIES` (e.g. `q03,q72`) limits a run to those queries.

use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use krishiv_common::config::Config;
use krishiv_exec::planner::ProviderStatistics;
use krishiv_optimizer::{Options, optimize_with_options, optimize_with_udfs};
use krishiv_plan::Plan;
use krishiv_udf_sdk::UdfLibrary;

use super::tpch::{self, Tpch};

/// How a rule is switched off.
#[derive(Clone, Copy)]
enum Off {
    /// An [`Options`] switch (`optimize_with_options`, no session configuration).
    Option(fn(&mut Options)),
    /// A session configuration key set to `false` (`optimize_with_udfs`; the
    /// configured all-on arm sets every such key to `true`).
    Config(&'static str),
}

/// The switched rules: the optimizer differential's list.
const RULES: [(&str, Off); 10] = [
    ("cse", Off::Option(|o| o.cse = false)),
    (
        "stats_propagation",
        Off::Option(|o| o.stats_propagation = false),
    ),
    (
        "inferred_join_filters",
        Off::Option(|o| o.inferred_join_filters = false),
    ),
    (
        "eager_aggregation",
        Off::Option(|o| o.eager_aggregation = false),
    ),
    (
        "late_payload_join",
        Off::Option(|o| o.late_payload_join = false),
    ),
    (
        "preserved_side_semi_join",
        Off::Option(|o| o.preserved_side_semi_join = false),
    ),
    ("group_key_code", Off::Option(|o| o.group_key_code = false)),
    (
        "subquery_as_window",
        Off::Config("krishiv.optimizer.subqueryAsWindow"),
    ),
    (
        "share_repeated_subtrees",
        Off::Config("krishiv.optimizer.shareRepeatedSubtrees"),
    ),
    (
        "share_filtered_scans",
        Off::Config("krishiv.optimizer.shareFilteredScans"),
    ),
];

/// A session configuration with every [`Off::Config`] rule on, then `off`
/// set to `false`.
fn config(off: Option<&str>) -> Config {
    let mut c = Config::new();
    for (_, rule) in RULES {
        if let Off::Config(key) = rule {
            c.set(key, if Some(key) == off { "false" } else { "true" })
                .unwrap();
        }
    }
    c
}

/// One value of a result row: a double compared within a tolerance, any
/// other value by its rendering.
#[derive(Clone, Debug)]
enum Cell {
    F(f64),
    S(String),
}

impl Cell {
    /// The sort key: doubles rounded to nine significant digits, so rows
    /// whose doubles differ in the last bits sort alike.
    fn key(&self) -> String {
        match self {
            // `-0.0` matches `0.0`, so it sorts with it.
            Self::F(v) if *v == 0.0 => format!("{:.8e}", 0.0),
            Self::F(v) if v.is_finite() => format!("{v:.8e}"),
            Self::F(v) => format!("{v}"),
            Self::S(s) => s.clone(),
        }
    }

    fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F(a), Self::F(b)) => {
                a == b
                    || (a.is_nan() && b.is_nan())
                    || (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
            }
            (Self::S(a), Self::S(b)) => a == b,
            _ => false,
        }
    }
}

/// The rows of `batches`, sorted by their cells' keys.
fn multiset(batches: &[RecordBatch]) -> Vec<Vec<Cell>> {
    let mut out = Vec::new();
    for b in batches {
        for row in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), row))
                    .collect::<Vec<_>>(),
            );
        }
    }
    out.sort_by(|a, b| a.iter().map(Cell::key).cmp(b.iter().map(Cell::key)));
    out
}

fn cell(a: &dyn Array, row: usize) -> Cell {
    if a.is_null(row) {
        return Cell::S("NULL".to_owned());
    }
    match a.data_type() {
        ArrowType::Float64 => Cell::F(a.as_primitive::<Float64Type>().value(row)),
        ArrowType::Float32 => Cell::F(f64::from(a.as_primitive::<Float32Type>().value(row))),
        _ => Cell::S(tpch::render(a, row)),
    }
}

/// The first difference between two sorted multisets, if any.
fn difference(want: &[Vec<Cell>], got: &[Vec<Cell>]) -> Option<String> {
    if want.len() != got.len() {
        return Some(format!("{} rows, all on {} rows", got.len(), want.len()));
    }
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        if w.len() != g.len() || !w.iter().zip(g).all(|(a, b)| a.matches(b)) {
            return Some(format!("sorted row {i}: {g:?}, all on {w:?}"));
        }
    }
    None
}

/// What one suite found.
#[derive(Debug, Default)]
struct Outcome {
    /// Queries whose all-on plan ran.
    compared: usize,
    /// Queries whose all-on plan did not plan or run (not compared).
    skipped: Vec<String>,
    /// Per rule: changed plans run and compared.
    changed: [usize; RULES.len()],
    /// `query rule: ...` for every difference or failure of a changed arm.
    differences: Vec<String>,
}

/// Plans and runs `plan`, as a sorted multiset.
fn run(t: &Tpch, plan: &Plan) -> Result<Vec<Vec<Cell>>, String> {
    let runnable = t.runnable(plan).map_err(|e| e.to_string())?;
    super::run(&runnable)
        .map(|b| multiset(&b))
        .map_err(|e| e.to_string())
}

fn run_suite(suite: &str, t: &Tpch, sqls: &[(String, String)]) -> Outcome {
    let only: Option<Vec<String>> = std::env::var("KRISHIV_RULE_DIFF_QUERIES")
        .ok()
        .map(|s| s.split(',').map(|q| q.trim().to_owned()).collect());
    let stats = ProviderStatistics(Arc::clone(&t.tables) as _);
    let library = UdfLibrary::new();
    let mut out = Outcome::default();
    for (id, sql) in sqls {
        if only.as_ref().is_some_and(|o| !o.contains(id)) {
            continue;
        }
        let start = std::time::Instant::now();
        let plan = match t.analyzed(sql) {
            Ok(p) => p,
            Err(e) => {
                out.skipped.push(format!("{id}: analysis: {e}"));
                continue;
            }
        };
        let Ok(base) = optimize_with_options(&plan, &stats, &Options::default()) else {
            out.skipped.push(format!("{id}: optimizer"));
            continue;
        };
        let want = match run(t, &base) {
            Ok(w) => w,
            Err(e) => {
                out.skipped.push(format!("{id}: {e}"));
                continue;
            }
        };
        out.compared += 1;
        let base_text = format!("{:?}", base.root());
        let configured = |c: &Config| optimize_with_udfs(&plan, &stats, &library, c);
        let mut arms = 0;
        // The configured all-on arm: what a configuration rule's arm differs
        // from; its rows must be the all-on rows too.
        let conf_text = match configured(&config(None)) {
            Ok(p) => {
                let text = format!("{:?}", p.root());
                if text != base_text {
                    arms += 1;
                    match run(t, &p) {
                        Ok(got) => {
                            if let Some(d) = difference(&want, &got) {
                                out.differences
                                    .push(format!("{suite} {id} configured all on: {d}"));
                            }
                        }
                        Err(e) => out
                            .differences
                            .push(format!("{suite} {id} configured all on: {e}")),
                    }
                }
                text
            }
            Err(e) => {
                out.differences
                    .push(format!("{suite} {id} configured all on: optimizer: {e}"));
                continue;
            }
        };
        for (i, (rule, off)) in RULES.iter().enumerate() {
            let (other, against) = match *off {
                Off::Option(f) => {
                    let mut o = Options::default();
                    f(&mut o);
                    (optimize_with_options(&plan, &stats, &o), &base_text)
                }
                Off::Config(key) => (configured(&config(Some(key))), &conf_text),
            };
            let other = match other {
                Ok(p) => p,
                Err(e) => {
                    out.differences
                        .push(format!("{suite} {id} with {rule} off: optimizer: {e}"));
                    continue;
                }
            };
            if format!("{:?}", other.root()) == *against {
                continue;
            }
            arms += 1;
            out.changed[i] += 1;
            match run(t, &other) {
                Ok(got) => {
                    if let Some(d) = difference(&want, &got) {
                        out.differences
                            .push(format!("{suite} {id} with {rule} off: {d}"));
                    }
                }
                Err(e) => out
                    .differences
                    .push(format!("{suite} {id} with {rule} off: {e}")),
            }
        }
        eprintln!(
            "rule-differential {suite} {id}: {} rows, {arms} changed arms, {:.1?}",
            want.len(),
            start.elapsed()
        );
    }
    out
}

fn report(suite: &str, out: &Outcome) {
    for (i, (rule, _)) in RULES.iter().enumerate() {
        eprintln!(
            "rule-differential {suite:<6} {rule:<24} plans changed and run {:>3} (of {} queries)",
            out.changed[i], out.compared
        );
    }
    for s in &out.skipped {
        eprintln!("rule-differential {suite} not compared: {s}");
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn tpch_suite(dir: &Path) -> Outcome {
    let rt = runtime();
    let t = tpch::parquet_tables(dir, &rt);
    let sqls: Vec<(String, String)> = (1..=22)
        .map(|q| (format!("q{q:02}"), tpch::sql(q)))
        .collect();
    let out = run_suite("tpch", &t, &sqls);
    report("tpch", &out);
    out
}

/// TPC-H SF 0.01 (the fixture): every query runs, every rule off returns
/// the all-on rows.
#[test]
fn tpch_sf0_01_each_rule_off_returns_the_rows_of_all_rules_on() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let out = tpch_suite(&dir);
    assert!(out.differences.is_empty(), "{}", out.differences.join("\n"));
    if std::env::var_os("KRISHIV_RULE_DIFF_QUERIES").is_none() {
        assert_eq!(out.compared, 22, "not compared: {:?}", out.skipped);
        assert!(
            out.changed.iter().filter(|&&n| n > 0).count() >= 3,
            "too few rules changed a plan: {:?}",
            out.changed
        );
    }
}

/// As the SF 0.01 test at TPC-H SF1 (`KRISHIV_TPCH_SF1_DIR`).
#[test]
#[ignore = "needs the TPC-H SF1 data set"]
fn tpch_sf1_each_rule_off_returns_the_rows_of_all_rules_on() {
    let dir = std::env::var("KRISHIV_TPCH_SF1_DIR").expect("KRISHIV_TPCH_SF1_DIR");
    let out = tpch_suite(Path::new(&dir));
    assert!(out.differences.is_empty(), "{}", out.differences.join("\n"));
    if std::env::var_os("KRISHIV_RULE_DIFF_QUERIES").is_none() {
        assert_eq!(out.compared, 22, "not compared: {:?}", out.skipped);
    }
}

/// The TPC-DS tables.
const TPCDS_TABLES: [&str; 24] = [
    "call_center",
    "catalog_page",
    "catalog_returns",
    "catalog_sales",
    "customer",
    "customer_address",
    "customer_demographics",
    "date_dim",
    "household_demographics",
    "income_band",
    "inventory",
    "item",
    "promotion",
    "reason",
    "ship_mode",
    "store",
    "store_returns",
    "store_sales",
    "time_dim",
    "warehouse",
    "web_page",
    "web_returns",
    "web_sales",
    "web_site",
];

/// Every TPC-DS query over SF1 (`KRISHIV_TPCDS_SF1_DIR`) that plans and
/// runs with every rule on returns the same rows with each rule off.
#[test]
#[ignore = "needs the TPC-DS SF1 data set"]
fn tpcds_sf1_each_rule_off_returns_the_rows_of_all_rules_on() {
    let dir = std::env::var("KRISHIV_TPCDS_SF1_DIR").expect("KRISHIV_TPCDS_SF1_DIR");
    let rt = runtime();
    let t = tpch::parquet_dir(Path::new(&dir), &TPCDS_TABLES, true, &rt);
    let sqls: Vec<(String, String)> = (1..=99)
        .map(|q| {
            let path = tpch::root().join(format!("bench/queries/tpcds/q{q:02}.sql"));
            // The harness's copies quote column aliases with double quotes
            // (`AS "order count"`), which Spark reads as strings; Spark's own
            // TPC-DS files backtick them.
            let text = std::fs::read_to_string(path).unwrap().replace('"', "`");
            (format!("q{q:02}"), text)
        })
        .collect();
    let out = run_suite("tpcds", &t, &sqls);
    report("tpcds", &out);
    assert!(out.differences.is_empty(), "{}", out.differences.join("\n"));
    // Every query plans and runs at SF1 (2026-11-16: all 99, 369 s in a debug build).
    if std::env::var_os("KRISHIV_RULE_DIFF_QUERIES").is_none() {
        assert_eq!(out.compared, 99, "not compared: {:?}", out.skipped);
    }
}
