// SPDX-License-Identifier: Apache-2.0
//! `krishiv.optimizer.joinPruneFirst` (perf lever D1a): in a tree whose largest input joins
//! another on a composite key with a column clustered in the largest input's files, the inputs on
//! the other side of that edge are planned first when that costs at most 4x the cheapest order
//! (TPC-DS q17's shape). Unclustered or single-column edges leave the plan alone, the choice is
//! a fixed point, and the setting turns it off.

use std::collections::HashMap;

use krishiv_common::config::Config;
use krishiv_optimizer::{
    ColumnStats, RuleStep, StatSource, Statistics, TableStats, optimize_traced_with_udfs,
};
use krishiv_plan::{LogicalPlan, Plan, PlanNode, ReadSource, Scan};
use krishiv_udf_sdk::UdfLibrary;

use crate::support;

/// A table: rows, and per column its type, distinct count and range.
type Table = (&'static str, u64, &'static [(&'static str, u64, f64, f64)]);

/// q17's inputs at SF10, by row counts and key domains: sales `f` (28.8M rows, sorted by
/// ticket), returns `r` (2.88M, a subset of the sales' (item, ticket) pairs), catalog sales `c`
/// (14.4M) and a date table `d` read twice.
const TABLES: &[Table] = &[
    (
        "f",
        28_800_000,
        &[
            ("f_item", 102_000, 1.0, 102_000.0),
            ("f_ticket", 2_400_000, 1.0, 2_400_000.0),
            ("f_cust", 500_000, 1.0, 500_000.0),
            ("f_date", 1_823, 2_450_816.0, 2_452_642.0),
            ("f_qty", 100, 1.0, 100.0),
        ],
    ),
    (
        "r",
        2_880_000,
        &[
            ("r_item", 102_000, 1.0, 102_000.0),
            ("r_ticket", 2_300_000, 1.0, 2_400_000.0),
            ("r_cust", 500_000, 1.0, 500_000.0),
            ("r_date", 2_003, 2_450_820.0, 2_452_822.0),
            ("r_qty", 100, 1.0, 100.0),
        ],
    ),
    (
        "c",
        14_400_000,
        &[
            ("c_item", 102_000, 1.0, 102_000.0),
            ("c_cust", 500_000, 1.0, 500_000.0),
            ("c_date", 1_836, 2_450_815.0, 2_452_654.0),
            ("c_qty", 100, 1.0, 100.0),
        ],
    ),
    (
        "d",
        73_049,
        &[
            ("d_sk", 73_049, 2_415_022.0, 2_488_070.0),
            ("d_year", 201, 1_900.0, 2_100.0),
            ("d_moy", 12, 1.0, 12.0),
        ],
    ),
];

/// [`TABLES`] with `clustered` columns reported as spanning `spread` of their range per unit.
#[derive(Debug)]
struct Stats {
    clustered: &'static [&'static str],
    spread: f64,
}

impl Statistics for Stats {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let table = name.last()?;
        let (_, rows, cols) = TABLES.iter().find(|t| t.0 == table.as_str())?;
        let columns: HashMap<String, ColumnStats> = cols
            .iter()
            .map(|&(c, ndv, lo, hi)| {
                let stats = ColumnStats {
                    source: StatSource::Provided,
                    distinct: Some(ndv),
                    nulls: Some(0),
                    min: Some(krishiv_plan::Literal::Double(lo)),
                    max: Some(krishiv_plan::Literal::Double(hi)),
                };
                (c.to_owned(), stats)
            })
            .collect();
        Some(TableStats {
            rows: *rows,
            columns,
        })
    }

    fn unit_spread(&self, _: &Scan, column: &str) -> Option<f64> {
        self.clustered.contains(&column).then_some(self.spread)
    }
}

fn ctx() -> krishiv_plan::analyzer::AnalysisContext {
    let mut cat = support::catalog();
    for (table, _, columns) in TABLES {
        let columns = columns
            .iter()
            .map(|(c, ..)| krishiv_plan::analyzer::TableColumn {
                name: (*c).to_owned(),
                data_type: krishiv_types::DataType::Long,
                nullable: true,
                char_varchar: None,
            })
            .collect();
        cat = cat.with_table(&["default"], table, columns, None);
    }
    support::ctx_over(cat)
}

/// q17's join graph: `f` to `r` on (item, ticket, customer), `r` to `c` on (item, customer),
/// a filtered date on each of `f`, `r`, `c`.
const Q17: &str = "SELECT count(f_qty), sum(r_qty), sum(c_qty) \
     FROM f, r, c, d d1, d d2, d d3 \
     WHERE d1.d_moy BETWEEN 1 AND 3 AND d1.d_year = 2001 AND d1.d_sk = f_date \
       AND f_cust = r_cust AND f_item = r_item AND f_ticket = r_ticket \
       AND r_date = d2.d_sk AND d2.d_moy BETWEEN 4 AND 10 AND d2.d_year = 2001 \
       AND r_cust = c_cust AND r_item = c_item \
       AND c_date = d3.d_sk AND d3.d_year IN (2001, 2002, 2003)";

/// The plan of `sql` and the join-order notes, with the setting given (`None`: the default).
fn plan(sql: &str, stats: &Stats, on: Option<bool>) -> (Plan, Vec<String>) {
    let mut config = Config::new();
    if let Some(on) = on {
        config
            .set("krishiv.optimizer.joinPruneFirst", &on.to_string())
            .unwrap();
    }
    let plan = support::resolve(sql, &ctx());
    let (optimized, steps) =
        optimize_traced_with_udfs(&plan, stats, &UdfLibrary::new(), &config).unwrap();
    (optimized, notes(&steps))
}

fn notes(steps: &[RuleStep]) -> Vec<String> {
    steps
        .iter()
        .filter(|s| s.rule == "JoinOrder" && s.note.starts_with("tree "))
        .map(|s| s.note.clone())
        .collect()
}

/// The table names of the scans under `p`.
fn tables(p: &LogicalPlan) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![p];
    while let Some(n) = stack.pop() {
        if let PlanNode::Scan(s) = &n.node
            && let ReadSource::Table { name, .. } = &s.source
        {
            out.push(name.last().unwrap().clone());
        }
        n.for_each_child(&mut |c| stack.push(c));
    }
    out.sort();
    out
}

/// The topmost inner join over the scan of `table` and whether that join's other input holds
/// exactly `others` (sorted table names).
fn joined_last(p: &LogicalPlan, table: &str, others: &[&str]) -> bool {
    let mut stack = vec![p];
    while let Some(n) = stack.pop() {
        if let PlanNode::Join { left, right, .. } = &n.node {
            for (a, b) in [(left, right), (right, left)] {
                if tables(a).iter().any(|t| t == table) && !tables(b).iter().any(|t| t == table) {
                    let mut b = tables(b);
                    b.dedup();
                    if tables(a).iter().filter(|t| *t == table).count() == 1
                        && b.iter().map(String::as_str).collect::<Vec<_>>() == others
                    {
                        return true;
                    }
                }
            }
        }
        n.for_each_child(&mut |c| stack.push(c));
    }
    false
}

const CLUSTERED: Stats = Stats {
    clustered: &["f_ticket"],
    spread: 0.01,
};

#[test]
fn a_clustered_composite_key_plans_the_other_side_first() {
    let (on, notes) = plan(Q17, &CLUSTERED, None);
    assert!(
        notes.iter().any(|n| n.contains("prune_first=taken")),
        "{notes:?}"
    );
    // `f` meets the block {r, c, d2, d3} (and nothing else of it) as one input.
    assert!(
        joined_last(on.root(), "f", &["c", "d", "r"]),
        "{}",
        on.root()
    );
    let (off, off_notes) = plan(Q17, &CLUSTERED, Some(false));
    assert!(off_notes.iter().all(|n| !n.contains("prune_first")));
    assert!(!on.structurally_eq(&off), "the choice changes this plan");
    // A fixed point: planned again, the plan stays.
    let (again, _) =
        optimize_traced_with_udfs(&on, &CLUSTERED, &UdfLibrary::new(), &Config::new()).unwrap();
    assert!(again.structurally_eq(&on), "{}", again.root());
}

#[test]
fn an_unclustered_or_widely_spread_key_leaves_the_plan_alone() {
    let (off, _) = plan(Q17, &CLUSTERED, Some(false));
    for stats in [
        Stats {
            clustered: &[],
            spread: 0.01,
        },
        // More than a sixteenth of the range per unit is not clustered.
        Stats {
            clustered: &["f_ticket"],
            spread: 0.07,
        },
    ] {
        let (p, notes) = plan(Q17, &stats, None);
        assert!(
            notes.iter().all(|n| !n.contains("prune_first")),
            "{notes:?}"
        );
        assert!(p.structurally_eq(&off), "{stats:?}");
    }
}

#[test]
fn a_single_column_edge_does_not_open_the_gate() {
    let sql = "SELECT count(f_qty), sum(c_qty) FROM f, r, c, d \
               WHERE f_ticket = r_ticket AND r_cust = c_cust AND c_date = d_sk AND d_year = 2001";
    let (_, notes) = plan(sql, &CLUSTERED, None);
    assert!(
        notes.iter().all(|n| !n.contains("prune_first")),
        "{notes:?}"
    );
}

/// With one month of `f` (TPC-DS q29's `d1`), joining `f` to the block before its date costs
/// more than 4x the cheapest order: the cheapest order stays.
#[test]
fn a_dearer_blocked_order_is_kept_out() {
    let sql = Q17.replace(
        "d1.d_moy BETWEEN 1 AND 3 AND d1.d_year = 2001",
        "d1.d_moy = 4 AND d1.d_year = 2001",
    );
    let (on, notes) = plan(&sql, &CLUSTERED, None);
    assert!(
        notes.iter().any(|n| n.contains("prune_first=kept")),
        "{notes:?}"
    );
    let (off, _) = plan(&sql, &CLUSTERED, Some(false));
    assert!(on.structurally_eq(&off));
}
