// SPDX-License-Identifier: Apache-2.0
//! The join enumerators (P2-T113, `krishiv.optimizer.joinEnumerator` and
//! `joinDpLimit`): `greedy` plans return the same rows and never cost less
//! than the exact plans, a higher limit plans larger trees exactly, and each
//! tree planned is described in the trace (`EXPLAIN VERBOSE`).

use krishiv_common::config::Config;
use krishiv_optimizer::{NoStatistics, RuleStep, Statistics, optimize_traced_with_udfs};
use krishiv_plan::Plan;
use krishiv_udf_sdk::UdfLibrary;

use crate::join_order::fixture;
use crate::support::{self, data, eval};
use crate::tpcds;

/// One tree's note, parsed: `key=value` fields.
#[derive(Debug, Clone, PartialEq)]
struct Tree {
    inputs: usize,
    enumerator: String,
    fallback: String,
    pairs: u64,
    considered: u64,
    cost: f64,
    shape: String,
}

fn trees(steps: &[RuleStep]) -> Vec<Tree> {
    steps
        .iter()
        .filter(|s| s.rule == "JoinOrder" && s.note.starts_with("tree "))
        .map(|s| {
            let field = |k: &str| {
                s.note
                    .split(' ')
                    .find_map(|f| f.strip_prefix(&format!("{k}=")))
                    .unwrap()
                    .to_owned()
            };
            Tree {
                inputs: field("inputs").parse().unwrap(),
                enumerator: field("enumerator"),
                fallback: field("fallback"),
                pairs: field("pairs").parse().unwrap(),
                considered: field("considered").parse().unwrap(),
                cost: field("cost").parse().unwrap(),
                shape: s.note.split_once("shape=").unwrap().1.to_owned(),
            }
        })
        .collect()
}

/// `plan` optimised under the given settings, with its trees.
fn plan_with(plan: &Plan, stats: &dyn Statistics, conf: &[(&str, &str)]) -> (Plan, Vec<Tree>) {
    let mut config = Config::new();
    for (k, v) in conf {
        config.set(k, v).unwrap();
    }
    let (optimized, steps) =
        optimize_traced_with_udfs(plan, stats, &UdfLibrary::new(), &config).unwrap();
    (optimized, trees(&steps))
}

const GREEDY: &[(&str, &str)] = &[("krishiv.optimizer.joinEnumerator", "greedy")];

/// TPC-H's trees (at most 8 inputs) are all planned exactly; forced greedy
/// plans cost no less and return the same rows.
#[test]
fn tpch_greedy_plans_return_the_same_rows() {
    let ctx = support::ctx();
    let stats = fixture();
    let db = data::tpch(7, 150);
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &ctx);
        let (_, exact) = plan_with(&plan, &stats, &[]);
        let (greedy, greedy_trees) = plan_with(&plan, &stats, GREEDY);
        assert_eq!(exact.len(), greedy_trees.len(), "q{q:02}");
        for (e, g) in exact.iter().zip(&greedy_trees) {
            assert_eq!(
                (e.enumerator.as_str(), e.fallback.as_str()),
                ("dpsub", "none")
            );
            assert_eq!(
                (g.enumerator.as_str(), g.fallback.as_str()),
                ("greedy", "forced")
            );
            assert_eq!(g.pairs, 0);
            assert!(g.cost >= e.cost, "q{q:02}: {g:?} {e:?}");
        }
        assert_eq!(
            eval::rows(&db, greedy.root()),
            eval::rows(&db, plan.root()),
            "q{q:02}"
        );
    }
}

/// Every TPC-DS tree of at most 10 inputs is planned exactly, at a cost no
/// greedy plan beats, and the trace counts the work; larger ones (q64's
/// shared CTE joins 18) are planned greedily for their size.
#[test]
fn tpcds_trees_are_planned_exactly() {
    let ctx = support::ctx_over(tpcds::catalog());
    let (mut planned, mut large) = (0, 0);
    for q in 1..=99 {
        let Some(plan) = support::try_resolve(&tpcds::sql(q), &ctx) else {
            continue;
        };
        let (_, exact) = plan_with(&plan, &NoStatistics, &[]);
        let (_, greedy) = plan_with(&plan, &NoStatistics, GREEDY);
        assert_eq!(exact.len(), greedy.len(), "q{q:02}");
        for (e, g) in exact.iter().zip(&greedy) {
            if e.inputs > 10 {
                assert_eq!(e.fallback, "inputs", "q{q:02}");
                large += 1;
            } else {
                assert_eq!(e.fallback, "none", "q{q:02}");
                assert!(e.considered >= e.pairs, "q{q:02}");
                assert!(g.cost >= e.cost, "q{q:02}: {g:?} {e:?}");
            }
            planned += 1;
        }
    }
    assert!(planned > 99, "{planned} trees");
    assert!(large > 0, "q64's trees are larger than the limit");
}

/// `greedy` and a lower `joinDpLimit` plan a four-input tree greedily, and
/// say why.
#[test]
fn the_fallback_is_named() {
    let sql = "SELECT t.a FROM t JOIN u ON t.a = u.a JOIN t x ON x.c = u.a JOIN u y ON y.a = x.a";
    let plan = support::resolve(sql, &support::ctx());
    let (_, exact) = plan_with(&plan, &NoStatistics, &[]);
    assert_eq!(exact.len(), 1);
    assert_eq!((exact[0].inputs, exact[0].fallback.as_str()), (4, "none"));
    let (_, forced) = plan_with(&plan, &NoStatistics, GREEDY);
    assert_eq!(
        (
            forced[0].enumerator.as_str(),
            forced[0].fallback.as_str(),
            forced[0].pairs
        ),
        ("greedy", "forced", 0)
    );
    let (_, small) = plan_with(
        &plan,
        &NoStatistics,
        &[("krishiv.optimizer.joinDpLimit", "3")],
    );
    assert_eq!(small[0].fallback, "inputs");
    // A conjunct over all three inputs is all that holds them together.
    let sql = "SELECT t.a FROM t, u, t x WHERE t.a + u.a = x.a";
    let plan = support::resolve(sql, &support::ctx());
    let (_, held) = plan_with(&plan, &NoStatistics, &[]);
    assert_eq!(held[0].fallback, "hyperedge");
}

/// A higher `joinDpLimit` plans a chain past the default limit exactly, at
/// no higher cost and with the same rows; a limit above the cap of 14
/// counts as 14.
#[test]
fn a_higher_limit_plans_larger_chains_exactly() {
    let mut sql = "SELECT t1.b FROM t t1".to_owned();
    for i in 2..=12 {
        sql.push_str(&format!(" JOIN t t{i} ON t{i}.a = t{}.c", i - 1));
    }
    let plan = support::resolve(&sql, &support::ctx());
    let (_, default) = plan_with(&plan, &NoStatistics, &[]);
    let limit = [("krishiv.optimizer.joinDpLimit", "12")];
    let (exact_plan, exact) = plan_with(&plan, &NoStatistics, &limit);
    assert_eq!(
        (default[0].inputs, default[0].fallback.as_str()),
        (12, "inputs")
    );
    // A chain of 12 has (12³ − 12) / 6 = 286 connected pairs.
    assert_eq!((exact[0].fallback.as_str(), exact[0].pairs), ("none", 286));
    assert!(exact[0].cost <= default[0].cost);
    for seed in 1..=2 {
        let db = data::small(seed);
        assert_eq!(
            eval::rows(&db, exact_plan.root()),
            eval::rows(&db, plan.root())
        );
    }
    let (_, capped) = plan_with(
        &plan,
        &NoStatistics,
        &[("krishiv.optimizer.joinDpLimit", "99")],
    );
    assert_eq!(capped[0].fallback, "none");
}
