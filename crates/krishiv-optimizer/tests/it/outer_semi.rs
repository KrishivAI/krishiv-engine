// SPDX-License-Identifier: Apache-2.0
//! `SemiBelowOuterJoin` (TPC-DS q40, q80): where a filtered relation an inner
//! join above an outer join joins to the preserved side also reduces that
//! side below the outer join, where it does not, and that the rows never
//! change (repeating and NULL keys) and the plan is a fixed point.

use krishiv_optimizer::{
    NoStatistics, Options, Statistics, optimize_traced_with_options, optimize_with_options,
};

use crate::join_order::fixture;
use crate::late_payload::TpcdsRows;
use crate::support::{self, data, eval};
use crate::tpcds;

/// The optimizer's options with the group-key code (D2) off, as sessions run
/// it by default (`krishiv.optimizer.groupKeyCode` off).
fn options() -> Options {
    let mut options = Options::default();
    options.group_key_code = false;
    options
}

fn run_with(
    sql: &str,
    stats: &dyn Statistics,
    on: bool,
) -> (krishiv_plan::Plan, String, Vec<String>) {
    let plan = support::resolve(sql, &support::ctx());
    let mut options = options();
    options.semi_below_outer_join = on;
    let (optimized, steps) = optimize_traced_with_options(&plan, stats, &options).unwrap();
    let notes = steps
        .into_iter()
        .filter(|s| s.batch == "SemiBelowOuterJoin")
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
    let again = optimize_with_options(&optimized, &fixture(), &options()).unwrap();
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

fn added(notes: &[String]) -> Vec<&String> {
    notes.iter().filter(|n| n.starts_with("added")).collect()
}

/// The q40 shape over TPC-H: `lineitem LEFT JOIN partsupp`, joined to a
/// filtered `part` on the preserved side's key. The semi join on
/// `l_partkey` lands on `lineitem`, below the outer join.
#[test]
fn a_filtered_relation_reduces_the_preserved_side_below_the_outer_join() {
    let sql = "SELECT p_brand, count(ps_supplycost), sum(l_quantity) \
               FROM lineitem LEFT JOIN partsupp ON l_partkey = ps_partkey \
               AND l_suppkey = ps_suppkey, part \
               WHERE l_partkey = p_partkey AND p_size = 3 GROUP BY p_brand";
    let (text, notes) = check(sql);
    assert_eq!(
        added(&notes),
        ["added a semi join on [l_partkey = p_partkey] below a left outer join"],
        "{notes:?}\n{text}"
    );
    let outer = text.find("Join LeftOuter").unwrap();
    let semi = text.find("Join LeftSemi").unwrap();
    assert!(outer < semi, "{text}");
}

/// The rows stay the same where the other side's key repeats (each region
/// key is five nations' key), where the preserved key is NULL for a third of
/// the rows, through a right outer join, and with two filtered relations on
/// the preserved side.
#[test]
fn repeating_and_null_keys_keep_every_row() {
    for (sql, n) in [
        // A repeating key on the inner join's other side.
        (
            "SELECT n_name, count(*), count(c_custkey) FROM supplier \
             LEFT JOIN customer ON s_nationkey = c_nationkey AND c_acctbal > 9000, nation \
             WHERE s_nationkey = n_regionkey AND n_name = 'FRANCE' GROUP BY n_name",
            1,
        ),
        // A NULL preserved key for a third of the rows.
        (
            "SELECT p_brand, count(*), count(ps_supplycost) FROM (SELECT l_suppkey, \
             CASE WHEN l_orderkey % 3 = 0 THEN NULL ELSE l_partkey END AS pk FROM lineitem) l \
             LEFT JOIN partsupp ON pk = ps_partkey AND l_suppkey = ps_suppkey, part \
             WHERE pk = p_partkey AND p_size = 3 GROUP BY p_brand",
            1,
        ),
        // A right outer join, the preserved side on its right.
        (
            "SELECT p_brand, count(*), count(ps_supplycost) FROM partsupp \
             RIGHT JOIN lineitem ON l_partkey = ps_partkey AND l_suppkey = ps_suppkey, part \
             WHERE l_partkey = p_partkey AND p_size = 3 GROUP BY p_brand",
            1,
        ),
        // Two filtered relations, each on its own preserved key.
        (
            "SELECT p_brand, count(*), count(ps_supplycost) FROM lineitem \
             LEFT JOIN partsupp ON l_partkey = ps_partkey AND l_suppkey = ps_suppkey, part, orders \
             WHERE l_partkey = p_partkey AND p_size = 3 AND l_orderkey = o_orderkey \
             AND o_orderdate < DATE '1992-03-01' GROUP BY p_brand",
            2,
        ),
    ] {
        let (text, notes) = check(sql);
        assert_eq!(added(&notes).len(), n, "{sql}\n{notes:?}\n{text}");
    }
}

/// Where the semi join cannot help or is not exact, the plan stays as it
/// was, with the gate that said so where one did.
#[test]
fn the_rule_stays_out_where_it_does_not_apply() {
    for (sql, why) in [
        // The inner join reads the null-supplying side.
        (
            "SELECT p_brand, count(*) FROM lineitem LEFT JOIN partsupp \
             ON l_partkey = ps_partkey AND l_suppkey = ps_suppkey, part \
             WHERE ps_partkey = p_partkey AND p_size = 3 GROUP BY p_brand",
            None,
        ),
        // A relation that is not the same rows twice.
        (
            "SELECT p_brand, count(*) FROM lineitem LEFT JOIN partsupp \
             ON l_partkey = ps_partkey AND l_suppkey = ps_suppkey, \
             (SELECT * FROM part WHERE rand() < 0.1) p \
             WHERE l_partkey = p_partkey GROUP BY p_brand",
            None,
        ),
        // No filter: the semi join keeps every row.
        (
            "SELECT p_brand, count(*), count(ps_supplycost) FROM lineitem LEFT JOIN partsupp \
             ON l_partkey = ps_partkey AND l_suppkey = ps_suppkey, part \
             WHERE l_partkey = p_partkey GROUP BY p_brand",
            Some("over 0.25"),
        ),
        // The other side scans more rows than the preserved side holds.
        (
            "SELECT s_name, count(*), count(n_name) FROM supplier \
             LEFT JOIN nation ON s_nationkey = n_nationkey, lineitem \
             WHERE l_suppkey = s_suppkey AND l_suppkey < 100 GROUP BY s_name",
            Some("scans more rows"),
        ),
        // Only inner joins: join ordering's business.
        (
            "SELECT p_brand, count(*) FROM lineitem, part \
             WHERE l_partkey = p_partkey AND p_size = 3 GROUP BY p_brand",
            None,
        ),
    ] {
        let Some(_) = support::try_resolve(sql, &support::ctx()) else {
            continue;
        };
        let (_, text, notes) = run(sql, true);
        let (_, off, _) = run(sql, false);
        assert_eq!(text, off, "{sql}\n{notes:?}");
        assert!(added(&notes).is_empty(), "{sql}\n{notes:?}");
        if let Some(why) = why {
            assert!(notes.iter().any(|n| n.contains(why)), "{sql}\n{notes:?}");
        }
        if !sql.contains("rand()") {
            check(sql);
        }
    }
}

/// Default row counts add nothing.
#[test]
fn default_row_counts_add_nothing() {
    let sql = "SELECT p_brand, count(ps_supplycost) FROM lineitem LEFT JOIN partsupp \
               ON l_partkey = ps_partkey AND l_suppkey = ps_suppkey, part \
               WHERE l_partkey = p_partkey AND p_size = 3 GROUP BY p_brand";
    let (_, _, notes) = run_with(sql, &NoStatistics, true);
    assert!(added(&notes).is_empty(), "{notes:?}");
    assert!(
        notes
            .iter()
            .any(|n| n.contains("rest on default row counts")),
        "{notes:?}"
    );
}

/// Switched off, by option or by the session key, nothing is added.
#[test]
fn the_rule_can_be_switched_off() {
    let sql = "SELECT p_brand, count(ps_supplycost) FROM lineitem LEFT JOIN partsupp \
               ON l_partkey = ps_partkey AND l_suppkey = ps_suppkey, part \
               WHERE l_partkey = p_partkey AND p_size = 3 GROUP BY p_brand";
    let (_, on, _) = run(sql, true);
    let (_, off, notes) = run(sql, false);
    assert_ne!(on, off);
    assert!(notes.is_empty(), "{notes:?}");

    let plan = support::resolve(sql, &support::ctx());
    let library = krishiv_udf_sdk::UdfLibrary::new();
    let mut config = krishiv_common::config::Config::new();
    config
        .set("krishiv.optimizer.groupKeyCode", "false")
        .unwrap();
    let with = krishiv_optimizer::optimize_with_udfs(&plan, &fixture(), &library, &config).unwrap();
    assert!(
        with.root().to_string().contains("Join LeftSemi"),
        "{}",
        with.root()
    );
    config
        .set("krishiv.optimizer.semiBelowOuterJoin", "false")
        .unwrap();
    let without =
        krishiv_optimizer::optimize_with_udfs(&plan, &fixture(), &library, &config).unwrap();
    assert!(
        !without.root().to_string().contains("Join LeftSemi"),
        "{}",
        without.root()
    );
}

/// Over TPC-DS with row counts, every query keeps its output and is a fixed
/// point; the queries the rule changes are listed, so a change to the list
/// is a reviewed decision.
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
        let adds = steps
            .iter()
            .filter(|s| s.batch == "SemiBelowOuterJoin" && s.note.starts_with("added"))
            .count();
        if adds > 0 {
            changed.push((q, adds));
        }
    }
    insta::assert_debug_snapshot!(changed);
}
