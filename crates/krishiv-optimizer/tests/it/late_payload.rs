// SPDX-License-Identifier: Apache-2.0
//! `LatePayloadJoin` (P2b S5 lever (2)): where a dimension's payload joins
//! above the semi and anti joins that never read it, where it stays, and that
//! the rows never change (any key multiplicity, NULL keys) and the plan is a
//! fixed point.

use std::collections::HashMap;

use krishiv_optimizer::{
    NoStatistics, Options, Statistics, TableStats, optimize_traced_with_options,
    optimize_with_options,
};
use krishiv_plan::{ReadSource, Scan};

use crate::join_order::fixture;
use crate::support::{self, data, eval};
use crate::tpcds;

/// The optimized plan, its text and the rule's notes, with the rule on or off.
fn run_with(
    sql: &str,
    stats: &dyn Statistics,
    on: bool,
) -> (krishiv_plan::Plan, String, Vec<String>) {
    let plan = support::resolve(sql, &support::ctx());
    let options = Options {
        late_payload_join: on,
        ..Options::default()
    };
    let (optimized, steps) = optimize_traced_with_options(&plan, stats, &options).unwrap();
    let notes = steps
        .into_iter()
        .filter(|s| s.batch == "LatePayloadJoin")
        .map(|s| s.note)
        .collect();
    let text = optimized.root().to_string();
    (optimized, text, notes)
}

fn run(sql: &str, on: bool) -> (krishiv_plan::Plan, String, Vec<String>) {
    run_with(sql, &fixture(), on)
}

/// The rule's result returns the unoptimised plan's rows, keeps its output and
/// is a fixed point.
fn check(sql: &str) -> (String, Vec<String>) {
    let plan = support::resolve(sql, &support::ctx());
    let (optimized, text, notes) = run(sql, true);
    assert_eq!(
        optimized.root().resolved_output(),
        plan.root().resolved_output(),
        "{text}"
    );
    let again = optimize_with_options(&optimized, &fixture(), &Options::default()).unwrap();
    assert!(
        again.structurally_eq(&optimized),
        "not a fixed point:\n{}",
        again.root()
    );
    let db = data::tpch(7, 150);
    assert_eq!(
        eval::rows(&db, optimized.root()),
        eval::rows(&db, plan.root()),
        "{text}"
    );
    (text, notes)
}

fn moved(notes: &[String]) -> Vec<&String> {
    notes.iter().filter(|n| n.starts_with("moved")).collect()
}

/// TPC-H q21's `s_name` joins above the `EXISTS` and `NOT EXISTS` joins, and
/// `lineitem` keeps the Saudi suppliers' reduction through a key-only semi
/// join; q18's `c_name` leaves the `orders` build of the join with
/// `lineitem`. No other TPC-H query changes.
#[test]
fn q21_and_q18_payloads_join_above_their_semi_and_anti_joins() {
    for q in 1..=22 {
        let (text, notes) = check(&support::tpch_sql(q));
        match q {
            21 => {
                assert_eq!(
                    moved(&notes),
                    [
                        "moved the join on [l_suppkey = s_suppkey] above 2 non-inner join(s), payload [s_name]"
                    ],
                    "{notes:?}"
                );
                let anti = text.find("Join LeftAnti").unwrap();
                let semi_on_supplier = text.find("Join LeftSemi, (s_suppkey#").unwrap();
                assert!(anti < semi_on_supplier, "{text}");
                // No build below the anti join carries the name.
                assert!(!text[anti..].contains("s_name#1,"), "{text}");
            }
            18 => assert_eq!(
                moved(&notes),
                [
                    "moved the join on [o_custkey = c_custkey] above 1 non-inner join(s), payload [c_name]"
                ],
                "{notes:?}"
            ),
            _ => assert!(moved(&notes).is_empty(), "q{q:02}: {notes:?}"),
        }
    }
}

/// A dimension key that repeats: each supplier meets the five nations of its
/// region, and every one of them still counts it. NULL keys meet nothing,
/// before and after.
#[test]
fn repeating_and_null_dimension_keys_keep_every_joined_row() {
    for sql in [
        // EXISTS over lineitem: a semi join building the suppliers.
        "SELECT n_name, count(*) FROM supplier, nation WHERE s_nationkey = n_regionkey \
         AND EXISTS (SELECT * FROM lineitem WHERE l_suppkey = s_suppkey) GROUP BY n_name",
        // NOT EXISTS: an anti join.
        "SELECT n_name, count(*), sum(s_acctbal) FROM supplier, nation \
         WHERE s_nationkey = n_regionkey \
         AND NOT EXISTS (SELECT * FROM lineitem WHERE l_suppkey = s_suppkey AND l_quantity > 49) \
         GROUP BY n_name",
        // A payload read by a projection, not an aggregate.
        "SELECT s_suppkey, upper(n_name) FROM supplier, nation WHERE s_nationkey = n_regionkey \
         AND EXISTS (SELECT * FROM lineitem WHERE l_suppkey = s_suppkey)",
        // A key that is NULL for a third of the suppliers.
        "SELECT n_name, count(*) FROM (SELECT s_suppkey, \
         CASE WHEN s_suppkey % 3 = 0 THEN NULL ELSE s_nationkey END AS nk FROM supplier) s, nation \
         WHERE nk = n_regionkey \
         AND EXISTS (SELECT * FROM lineitem WHERE l_suppkey = s_suppkey) GROUP BY n_name",
    ] {
        let (text, notes) = check(sql);
        assert_eq!(moved(&notes).len(), 1, "{sql}\n{notes:?}\n{text}");
    }
}

/// Two dimensions under one consumer both move in one pass.
#[test]
fn two_dimensions_under_one_consumer_both_move() {
    let sql = "SELECT p_name, s_name, count(*) FROM partsupp, part, supplier \
               WHERE ps_partkey = p_partkey AND ps_suppkey = s_suppkey \
               AND EXISTS (SELECT * FROM lineitem WHERE l_partkey = ps_partkey \
               AND l_suppkey = ps_suppkey) GROUP BY p_name, s_name";
    let (text, notes) = check(sql);
    assert_eq!(moved(&notes).len(), 2, "{notes:?}\n{text}");
}

/// A left outer join with the dimension's side preserved passes it too.
#[test]
fn an_outer_join_on_the_preserved_side_passes_the_payload() {
    let sql = "SELECT n_name, count(l_orderkey) FROM (SELECT * FROM supplier, nation \
               WHERE s_nationkey = n_regionkey) sn \
               LEFT JOIN lineitem ON l_suppkey = s_suppkey AND l_quantity > 49 GROUP BY n_name";
    let (text, notes) = check(sql);
    assert_eq!(moved(&notes).len(), 1, "{notes:?}\n{text}");
}

/// Where the move cannot help or is not exact, the plan stays as it was, with
/// the gate that said so where one did.
#[test]
fn the_rule_stays_out_where_it_does_not_apply() {
    for (sql, why) in [
        // The semi join reads the payload.
        (
            "SELECT n_name, count(*) FROM supplier, nation WHERE s_nationkey = n_regionkey \
             AND EXISTS (SELECT * FROM lineitem WHERE l_suppkey = s_suppkey AND l_comment <> n_name) \
             GROUP BY n_name",
            None,
        ),
        // The semi join reads the dimension's key.
        (
            "SELECT n_name, count(*) FROM supplier, nation WHERE s_nationkey = n_regionkey \
             AND EXISTS (SELECT * FROM lineitem WHERE l_suppkey = s_suppkey \
             AND l_linenumber <> n_regionkey) GROUP BY n_name",
            None,
        ),
        // A dimension that is not the same rows twice.
        (
            "SELECT n_name, count(*) FROM supplier, (SELECT * FROM nation WHERE rand() < 0.5) n \
             WHERE s_nationkey = n_regionkey \
             AND EXISTS (SELECT * FROM lineitem WHERE l_suppkey = s_suppkey) GROUP BY n_name",
            None,
        ),
        // Eight bytes of payload.
        (
            "SELECT n_nationkey, count(*) FROM supplier, nation WHERE s_nationkey = n_regionkey \
             AND EXISTS (SELECT * FROM lineitem WHERE l_suppkey = s_suppkey) GROUP BY n_nationkey",
            Some("a payload of 8 B a row is under 16 B"),
        ),
        // The dimension is the larger side.
        (
            "SELECT s_name, count(*) FROM nation, supplier WHERE s_nationkey = n_nationkey \
             AND EXISTS (SELECT * FROM lineitem WHERE l_suppkey = n_regionkey) GROUP BY s_name",
            Some("the dimension estimates more rows"),
        ),
        // The semi join's other side is small: it is broadcast and the path
        // only streams through it.
        (
            "SELECT n_name, count(*) FROM supplier, nation WHERE s_nationkey = n_regionkey \
             AND EXISTS (SELECT * FROM region WHERE r_regionkey = s_nationkey) GROUP BY n_name",
            Some("no join on the path builds it"),
        ),
        // An existence join (EXISTS under OR) never builds its left input.
        (
            "SELECT n_name, count(*) FROM supplier, nation WHERE s_nationkey = n_regionkey \
             AND (EXISTS (SELECT * FROM lineitem WHERE l_suppkey = s_suppkey) OR s_acctbal > 0) \
             GROUP BY n_name",
            Some("no join on the path builds it"),
        ),
        // A right outer join with the path on its right builds its left input.
        (
            "SELECT n_name, count(l_orderkey) FROM lineitem RIGHT JOIN (SELECT * FROM supplier, \
             nation WHERE s_nationkey = n_regionkey) sn ON l_suppkey = s_suppkey \
             AND l_quantity > 49 GROUP BY n_name",
            Some("no join on the path builds it"),
        ),
        // Only inner joins: join ordering's business.
        (
            "SELECT n_name, count(*) FROM supplier, nation, lineitem \
             WHERE s_nationkey = n_nationkey AND l_suppkey = s_suppkey GROUP BY n_name",
            None,
        ),
    ] {
        let (_, text, notes) = run(sql, true);
        let (_, off, _) = run(sql, false);
        assert_eq!(text, off, "{sql}\n{notes:?}");
        assert!(moved(&notes).is_empty(), "{sql}\n{notes:?}");
        if let Some(why) = why {
            assert!(notes.iter().any(|n| n.contains(why)), "{sql}\n{notes:?}");
        }
        if !sql.contains("rand()") {
            check(sql);
        }
    }
}

/// Default row counts never move a join.
#[test]
fn default_row_counts_move_nothing() {
    let (_, _, notes) = run_with(&support::tpch_sql(21), &NoStatistics, true);
    assert!(moved(&notes).is_empty(), "{notes:?}");
    assert!(
        notes
            .iter()
            .any(|n| n.contains("rest on default row counts")),
        "{notes:?}"
    );
}

/// Switched off, by option or by the session key, nothing moves.
#[test]
fn the_rule_can_be_switched_off() {
    let sql = support::tpch_sql(21);
    let (_, on, _) = run(&sql, true);
    let (_, off, notes) = run(&sql, false);
    assert_ne!(on, off);
    assert!(notes.is_empty(), "{notes:?}");

    let plan = support::resolve(&sql, &support::ctx());
    let library = krishiv_udf_sdk::UdfLibrary::new();
    let mut config = krishiv_common::config::Config::new();
    let with = krishiv_optimizer::optimize_with_udfs(&plan, &fixture(), &library, &config).unwrap();
    assert_eq!(with.root().to_string(), on);
    config
        .set("krishiv.optimizer.latePayloadJoin", "false")
        .unwrap();
    let without =
        krishiv_optimizer::optimize_with_udfs(&plan, &fixture(), &library, &config).unwrap();
    assert_eq!(without.root().to_string(), off);
}

/// TPC-DS SF1 row counts (no column statistics): enough for the rule's
/// `from_statistics` gate.
#[derive(Debug)]
struct TpcdsRows;

const TPCDS_SF1_ROWS: [(&str, u64); 24] = [
    ("call_center", 6),
    ("catalog_page", 11_718),
    ("catalog_returns", 144_067),
    ("catalog_sales", 1_441_548),
    ("customer", 100_000),
    ("customer_address", 50_000),
    ("customer_demographics", 1_920_800),
    ("date_dim", 73_049),
    ("household_demographics", 7_200),
    ("income_band", 20),
    ("inventory", 11_745_000),
    ("item", 18_000),
    ("promotion", 300),
    ("reason", 35),
    ("ship_mode", 20),
    ("store", 12),
    ("store_returns", 287_514),
    ("store_sales", 2_880_404),
    ("time_dim", 86_400),
    ("warehouse", 5),
    ("web_page", 60),
    ("web_returns", 71_763),
    ("web_sales", 719_384),
    ("web_site", 30),
];

impl Statistics for TpcdsRows {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let table = name.last()?;
        let (_, rows) = TPCDS_SF1_ROWS.iter().find(|t| t.0 == table)?;
        Some(TableStats {
            rows: *rows,
            columns: HashMap::new(),
        })
    }
}

/// Over TPC-DS with row counts, every query keeps its output and is a fixed
/// point (q35 moves two dimensions under one consumer); the queries the rule
/// changes are listed, so a change to the list is a reviewed decision.
#[test]
fn tpcds_with_row_counts_is_a_fixed_point() {
    let ctx = support::ctx_over(tpcds::catalog());
    let mut changed = Vec::new();
    for q in 1..=99 {
        let Some(plan) = support::try_resolve(&tpcds::sql(q), &ctx) else {
            continue;
        };
        let (optimized, steps) =
            optimize_traced_with_options(&plan, &TpcdsRows, &Options::default()).unwrap();
        assert_eq!(
            optimized.root().resolved_output(),
            plan.root().resolved_output(),
            "q{q:02}"
        );
        let again = optimize_with_options(&optimized, &TpcdsRows, &Options::default()).unwrap();
        assert!(
            again.structurally_eq(&optimized),
            "q{q:02} is not a fixed point:\n{}",
            again.root()
        );
        let moves = steps
            .iter()
            .filter(|s| s.batch == "LatePayloadJoin" && s.note.starts_with("moved"))
            .count();
        if moves > 0 {
            changed.push((q, moves));
        }
    }
    insta::assert_debug_snapshot!(changed);
}
