// SPDX-License-Identifier: Apache-2.0
//! Preserved-side semi joins (`restrict`, TPC-DS q40/q80, perf lever K6): the
//! preserved side of an outer join is reduced by a key-only semi join with a
//! filtered input an inner join above joins it to, so the hash join can build
//! the reduced side. Where it applies, where it does not, that the rows never
//! change and that the plan is a fixed point.

use krishiv_optimizer::{Options, optimize_with_options};

use crate::join_order::fixture;
use crate::late_payload::TpcdsRows;
use crate::support::{self, data, eval};
use crate::tpcds;

/// The optimized plan's text with the rule on or off, after checking that the
/// plan keeps the output and the rows of the unoptimised plan and is a fixed
/// point.
fn check(sql: &str, on: bool) -> String {
    check_as(sql, on, true)
}

/// [`check`], comparing rows only with `rows` (the reference interpreter
/// runs no nondeterministic function).
fn check_as(sql: &str, on: bool, rows: bool) -> String {
    let plan = support::resolve(sql, &support::ctx());
    let mut options = Options::default();
    options.group_key_code = false;
    options.preserved_side_semi_join = on;
    let optimized = optimize_with_options(&plan, &fixture(), &options).unwrap();
    let text = optimized.root().to_string();
    assert_eq!(
        optimized.root().resolved_output(),
        plan.root().resolved_output(),
        "{text}"
    );
    let again = optimize_with_options(&optimized, &fixture(), &options).unwrap();
    assert!(
        again.structurally_eq(&optimized),
        "not a fixed point:\n{text}\nagain:\n{}",
        again.root()
    );
    if rows {
        let db = data::tpch(7, 150);
        assert_eq!(
            eval::rows(&db, optimized.root()),
            eval::rows(&db, plan.root()),
            "{text}"
        );
    }
    text
}

/// The text of the outer join's inputs in `text`: every line below the
/// `Join LeftOuter` line whose node starts further right than it.
fn below_outer(text: &str) -> String {
    let start = |l: &str| l.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(0);
    let mut lines = text.lines().skip_while(|l| !l.contains("Join LeftOuter"));
    let Some(head) = lines.next() else {
        return String::new();
    };
    let depth = start(head);
    lines
        .take_while(|l| start(l) > depth)
        .collect::<Vec<_>>()
        .join("\n")
}

/// `orders LEFT JOIN lineitem` under an inner join with the few rich customers
/// of one segment: the orders side gets a semi join with those customers'
/// keys, below the outer join; with the rule off it does not.
#[test]
fn a_filtered_dimension_above_reduces_the_preserved_side() {
    let sql = "SELECT o_orderkey, count(l_linenumber) FROM orders \
               LEFT JOIN lineitem ON o_orderkey = l_orderkey AND l_quantity > 10, customer \
               WHERE o_custkey = c_custkey AND c_mktsegment = 'BUILDING' AND c_acctbal > 9000 \
               GROUP BY o_orderkey";
    let on = check(sql, true);
    let below = below_outer(&on);
    assert!(below.contains("Join LeftSemi"), "{on}");
    assert!(below.contains("Shift"), "{on}");
    let off = check(sql, false);
    assert!(!below_outer(&off).contains("Join LeftSemi"), "{off}");
}

/// The q78 shape: only the null-extended rows of `orders LEFT JOIN lineitem`
/// (`l_orderkey IS NULL` above the outer join) under the inner join with a
/// filtered dimension. The preserved side is still reduced, and over 40
/// seeds of the generated data (30 customers each, so only some seeds hold
/// a rich `BUILDING` customer) the rows are the unoptimised plan's, and
/// some seeds have rows.
#[test]
fn null_extended_rows_under_a_filter_keep_their_rows() {
    let sql = "SELECT o_orderkey, o_totalprice FROM orders \
               LEFT JOIN lineitem ON o_orderkey = l_orderkey AND l_quantity > 49, customer \
               WHERE l_orderkey IS NULL AND o_custkey = c_custkey \
               AND c_mktsegment = 'BUILDING' AND c_acctbal > 9000";
    let on = check(sql, true);
    assert!(below_outer(&on).contains("Join LeftSemi"), "{on}");
    let plan = support::resolve(sql, &support::ctx());
    let mut options = Options::default();
    options.group_key_code = false;
    let optimized = optimize_with_options(&plan, &fixture(), &options).unwrap();
    let mut with_rows = 0;
    for seed in 1..=40 {
        let db = data::tpch(seed, 150);
        let want = eval::rows(&db, plan.root());
        assert_eq!(eval::rows(&db, optimized.root()), want, "seed {seed}");
        with_rows += usize::from(!want.is_empty());
    }
    assert!(
        with_rows > 0,
        "some seed holds null-extended rows of rich customers"
    );
}

/// The same shape written as a `RIGHT OUTER JOIN`: the preserved side is the
/// right input.
#[test]
fn a_right_outer_join_reduces_its_right_input() {
    let sql = "SELECT o_orderkey, count(l_linenumber) FROM lineitem \
               RIGHT JOIN orders ON o_orderkey = l_orderkey, customer \
               WHERE o_custkey = c_custkey AND c_mktsegment = 'BUILDING' AND c_acctbal > 9000 \
               GROUP BY o_orderkey";
    let on = check(sql, true);
    assert!(on.contains("Join LeftSemi"), "{on}");
    let off = check(sql, false);
    assert!(!off.contains("Join LeftSemi"), "{off}");
}

/// No semi join where it would not pay or not hold: an unfiltered dimension
/// (its keys are every order's), a join above that reads the null-supplying
/// side, a key that may raise, and a dimension filter that
/// keeps most rows.
#[test]
fn no_semi_join_where_it_restricts_nothing_or_does_not_apply() {
    for sql in [
        // Every customer: the domain keeps all it reads.
        "SELECT o_orderkey, count(l_linenumber) FROM orders \
         LEFT JOIN lineitem ON o_orderkey = l_orderkey, customer \
         WHERE o_custkey = c_custkey GROUP BY o_orderkey",
        // The inner join reads the null-supplying side.
        "SELECT o_orderkey, count(*) FROM orders \
         LEFT JOIN lineitem ON o_orderkey = l_orderkey, supplier \
         WHERE l_suppkey = s_suppkey AND s_acctbal > 9000 GROUP BY o_orderkey",
        // A key that may raise (ANSI `+` can overflow) on rows the join never reads.
        "SELECT o_orderkey, count(l_linenumber) FROM orders \
         LEFT JOIN lineitem ON o_orderkey = l_orderkey, customer \
         WHERE o_custkey + 1 = c_custkey AND c_acctbal > 9000 GROUP BY o_orderkey",
        // Most customers: the side keeps more than a quarter.
        "SELECT o_orderkey, count(l_linenumber) FROM orders \
         LEFT JOIN lineitem ON o_orderkey = l_orderkey, customer \
         WHERE o_custkey = c_custkey AND c_acctbal > -500 GROUP BY o_orderkey",
    ] {
        let text = check(sql, true);
        assert!(
            !below_outer(&text).contains("Join LeftSemi"),
            "{sql}\n{text}"
        );
    }
    // Nondeterminism (plans only: the interpreter runs no `rand`). A
    // nondeterministic filter between the outer join and the join above
    // stays on the path (it never moves), and a nondeterministic dimension
    // is computed twice by the rewrite. A nondeterministic outer-join
    // condition is refused at analysis, as in Spark.
    for sql in [
        "SELECT o_orderkey, count(l_linenumber) FROM (SELECT * FROM orders \
         LEFT JOIN lineitem ON o_orderkey = l_orderkey AND l_quantity > 10 \
         WHERE rand(1) < 2.0) t, customer \
         WHERE o_custkey = c_custkey AND c_mktsegment = 'BUILDING' AND c_acctbal > 9000 \
         GROUP BY o_orderkey",
        "SELECT o_orderkey, count(l_linenumber) FROM orders \
         LEFT JOIN lineitem ON o_orderkey = l_orderkey AND l_quantity > 10, \
         (SELECT * FROM customer WHERE c_mktsegment = 'BUILDING' AND c_acctbal > 9000 \
         AND rand(1) < 2.0) c \
         WHERE o_custkey = c_custkey GROUP BY o_orderkey",
    ] {
        let text = check_as(sql, true, false);
        assert!(
            !below_outer(&text).contains("Join LeftSemi"),
            "{sql}\n{text}"
        );
    }
}

/// Over TPC-DS with row counts only, every query keeps its output and is a
/// fixed point with the rule on; the queries whose plans it changes are
/// listed, so a change to the list is a reviewed decision.
#[test]
fn tpcds_with_row_counts_is_a_fixed_point() {
    let ctx = support::ctx_over(tpcds::catalog());
    let mut changed = Vec::new();
    for q in 1..=99 {
        let Some(plan) = support::try_resolve(&tpcds::sql(q), &ctx) else {
            continue;
        };
        let mut off = Options::default();
        off.preserved_side_semi_join = false;
        let on = optimize_with_options(&plan, &TpcdsRows, &Options::default()).unwrap();
        assert_eq!(
            on.root().resolved_output(),
            plan.root().resolved_output(),
            "q{q:02}"
        );
        let again = optimize_with_options(&on, &TpcdsRows, &Options::default()).unwrap();
        assert!(
            again.structurally_eq(&on),
            "q{q:02} is not a fixed point:\n{}",
            again.root()
        );
        let without = optimize_with_options(&plan, &TpcdsRows, &off).unwrap();
        if !without.structurally_eq(&on) {
            changed.push(q);
        }
    }
    insta::assert_debug_snapshot!(changed);
}
