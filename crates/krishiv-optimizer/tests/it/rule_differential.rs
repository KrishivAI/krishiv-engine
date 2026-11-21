// SPDX-License-Identifier: Apache-2.0
//! The per-rule differential of design P2-T072b §6 item 2: for each rule with
//! a test switch ([`Options`]), the rows of every TPC-H and TPC-DS query
//! optimized with that rule off equal the rows with every rule on, through
//! the reference interpreter (exact fixed-point numbers, rows as a sorted
//! multiset). Each switch must change at least one plan, so no rule passes
//! vacuously.
//!
//! The data are the q-error report's (`qerror.rs`): generated TPC-H and
//! synthetic TPC-DS by default, statistics read off the data, and the same
//! settings: `KRISHIV_QERROR_DIR` (`dbgen`/`dsdgen` files), `KRISHIV_QERROR_SCALE`
//! (TPC-H orders) and `KRISHIV_QERROR_SUITES`.
//!
//! **This is not the design's SF1 run.** The default (the CI run) is TPC-H at
//! 150 orders and synthetic TPC-DS, where only 17 of the 84 queries the
//! interpreter runs return rows. SF1 is a `KRISHIV_QERROR_DIR` run, and the
//! interpreter's nested loops make it slow: at 1,500 orders TPC-H alone took
//! more than 15 minutes (2026-11-16). The interpreter computes every number,
//! doubles included, as exact fixed point, so the float rounding a reordered
//! aggregation can change is not exercised here (the goldens' tolerance never
//! applies). The design's SF1 run is the engine's:
//! `krishiv-exec/tests/planner/rule_differential.rs` runs the same arms
//! through the executor over Parquet files (TPC-H SF 0.01 by default; TPC-H
//! and TPC-DS SF1 as ignored tests), doubles within a tolerance.
//!
//! Rules that run only under a session configuration (`subquery_as_window`,
//! `share_repeated_subtrees`) are switched through it: their arms optimize
//! with `optimize_with_udfs` and a configuration that turns both on, then
//! one off. Their rows are compared with the same all-on rows as the
//! `Options` switches'.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use krishiv_common::config::Config;
use krishiv_optimizer::{Options, optimize_with_options, optimize_with_udfs};
use krishiv_plan::Plan;

use crate::qerror::{self, DataStats};
use crate::support::eval::{self, Db, V};
use crate::support::{self, data};
use crate::tpcds;

/// How a rule is switched off.
#[derive(Clone, Copy)]
enum Off {
    /// An [`Options`] switch (`optimize_with_options`, no session configuration).
    Option(fn(&mut Options)),
    /// A session configuration key set to `false` (`optimize_with_udfs`; the
    /// configuration's all-on arm sets every such key to `true`).
    Config(&'static str),
}

/// A switched rule: its name and what turns it off.
type Rule = (&'static str, Off);

/// A panic hook, as `std::panic::take_hook` returns it.
type PanicHook = Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>;

/// The switched rules.
const RULES: [Rule; 9] = [
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
        "semi_below_outer_join",
        Off::Option(|o| o.semi_below_outer_join = false),
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
];

/// A session configuration with every [`Off::Config`] rule on, then `off`
/// set to `false`.
fn config(off: Option<&str>) -> Config {
    let mut c = Config::new();
    for (_, rule) in RULES {
        if let Off::Config(key) = rule {
            let on = if Some(key) == off { "false" } else { "true" };
            c.set(key, on).unwrap();
        }
    }
    c
}

/// Silences the panic hook while it lives (the interpreter panics on what it
/// does not run), restoring it on every exit, a panic included.
struct QuietPanics(Option<PanicHook>);

impl QuietPanics {
    fn new() -> Self {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        Self(Some(hook))
    }
}

impl Drop for QuietPanics {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            std::panic::set_hook(h);
        }
    }
}

/// The rows of `plan` over `db`, or `None` when the interpreter cannot run it.
fn rows(db: &Db, plan: &Plan) -> Option<Vec<Vec<V>>> {
    catch_unwind(AssertUnwindSafe(|| eval::rows(db, plan.root()))).ok()
}

/// What one suite found, per rule (in [`RULES`] order).
#[derive(Debug, Default)]
struct Outcome {
    /// Queries whose all-on plan the interpreter ran.
    compared: usize,
    /// Queries whose rows were non-empty.
    nonempty: usize,
    /// Per rule: plans that changed with the rule off.
    changed: [usize; RULES.len()],
    /// Per rule: changed plans the interpreter could not run with the rule off.
    not_run: [usize; RULES.len()],
    /// Per rule: changed plans compared on a query that returns rows (an
    /// empty result proves little).
    with_rows: [usize; RULES.len()],
    /// `query rule: ...` for every difference.
    differences: Vec<String>,
}

fn run_suite(
    suite: &str,
    sqls: &[(String, String)],
    ctx: &krishiv_plan::analyzer::AnalysisContext,
    db: &Db,
) -> Outcome {
    let stats: DataStats = qerror::stats_of(db);
    let all_on = Options::default();
    let library = krishiv_udf_sdk::UdfLibrary::new();
    let mut out = Outcome::default();
    let _quiet = QuietPanics::new();
    for (id, sql) in sqls {
        let Some(plan) = support::try_resolve(sql, ctx) else {
            continue;
        };
        let base = optimize_with_options(&plan, &stats, &all_on).unwrap();
        let Some(want) = rows(db, &base) else {
            continue;
        };
        out.compared += 1;
        out.nonempty += usize::from(!want.is_empty());
        let base_text = format!("{:?}", base.root());
        // The configured all-on arm: its plan is what a configuration rule's
        // arm differs from; its rows must be the all-on rows too.
        let configured = |c: &Config| optimize_with_udfs(&plan, &stats, &library, c).unwrap();
        let base_conf = configured(&config(None));
        let conf_text = format!("{:?}", base_conf.root());
        if conf_text != base_text {
            match rows(db, &base_conf) {
                Some(got) if got == want => {}
                got => out.differences.push(format!(
                    "{suite} {id} configured all on: {}, all on: {} rows",
                    got.map_or_else(|| "not run".to_owned(), |g| format!("{} rows", g.len())),
                    want.len()
                )),
            }
        }
        for (i, (rule, off)) in RULES.iter().enumerate() {
            let (other, base_text) = match *off {
                Off::Option(f) => (
                    optimize_with_options(&plan, &stats, &support::options(f)).unwrap(),
                    &base_text,
                ),
                Off::Config(key) => (configured(&config(Some(key))), &conf_text),
            };
            if format!("{:?}", other.root()) == *base_text {
                continue;
            }
            out.changed[i] += 1;
            match rows(db, &other) {
                // A changed plan the interpreter cannot run would shrink the coverage unseen.
                None => {
                    out.not_run[i] += 1;
                    out.differences.push(format!(
                        "{suite} {id} with {rule} off: the interpreter cannot run the plan"
                    ));
                }
                Some(got) if got != want => out.differences.push(format!(
                    "{suite} {id} with {rule} off: {} rows, all on: {} rows",
                    got.len(),
                    want.len()
                )),
                Some(_) => out.with_rows[i] += usize::from(!want.is_empty()),
            }
        }
    }
    out
}

fn tpch() -> Outcome {
    let db = match std::env::var("KRISHIV_QERROR_DIR") {
        Ok(dir) => qerror::load(Path::new(&dir), &qerror::tpch_schemas(), "tbl"),
        Err(_) => data::tpch(7, qerror::scale()),
    };
    let sqls: Vec<(String, String)> = (1..=22)
        .map(|q| (format!("q{q:02}"), support::tpch_sql(q)))
        .collect();
    run_suite("tpch", &sqls, &support::ctx(), &db)
}

fn tpcds() -> Outcome {
    let schemas = qerror::tpcds_schemas();
    let db = match std::env::var("KRISHIV_QERROR_DIR") {
        Ok(dir) => qerror::load(Path::new(&dir), &schemas, "dat"),
        Err(_) => qerror::synthetic_tpcds(&schemas, 11, 2 * qerror::scale()),
    };
    let sqls: Vec<(String, String)> = (1..=99)
        .map(|q| (format!("q{q:02}"), tpcds::sql(q)))
        .collect();
    run_suite("tpcds", &sqls, &support::ctx_over(tpcds::catalog()), &db)
}

/// Every switched rule off, one at a time, returns the all-on rows on every
/// TPC-H and TPC-DS query the interpreter runs.
#[test]
fn each_rule_off_returns_the_rows_of_all_rules_on() {
    let suites = qerror::suites();
    let mut changed = [0; RULES.len()];
    let mut differences = Vec::new();
    for (name, run) in [("tpch", tpch as fn() -> Outcome), ("tpcds", tpcds)] {
        if !suites.contains(name) {
            continue;
        }
        let out = run();
        for (i, (rule, _)) in RULES.iter().enumerate() {
            println!(
                "rule-differential {name:<6} {rule:<22} plans changed {:>3}, compared with rows \
                 {:>3}, not run {:>3} (of {} queries compared, {} with rows)",
                out.changed[i], out.with_rows[i], out.not_run[i], out.compared, out.nonempty
            );
            changed[i] += out.with_rows[i];
        }
        // On the default data (scale 150) TPC-H runs 22 with 20 returning rows, TPC-DS
        // 84 with 17: most TPC-DS filters name literal values the synthetic data lack.
        // Other data hold no such floors.
        let default_data = std::env::var_os("KRISHIV_QERROR_DIR").is_none()
            && std::env::var_os("KRISHIV_QERROR_SCALE").is_none();
        let (runs, with_rows) = if name == "tpch" { (22, 18) } else { (80, 15) };
        assert!(
            !default_data || (out.compared >= runs && out.nonempty >= with_rows),
            "{name}: {} queries ran in the interpreter, {} with rows",
            out.compared,
            out.nonempty
        );
        differences.extend(out.differences);
    }
    assert!(differences.is_empty(), "{}", differences.join("\n"));
    if suites.len() == 2 {
        for (i, (rule, _)) in RULES.iter().enumerate() {
            assert!(
                changed[i] > 0,
                "{rule} off changed no plan of a query with rows: the differential is vacuous"
            );
        }
    }
}
