// SPDX-License-Identifier: Apache-2.0
//! The join enumerators (P2-T113, `krishiv.optimizer.joinEnumerator`): every
//! one plans the queries' trees with the same costs, `dphyp` finds `dpsub`'s
//! plans wherever `dpsub` is exact and plans no worse elsewhere, and each
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

const DPHYP: &[(&str, &str)] = &[("krishiv.optimizer.joinEnumerator", "dphyp")];
const GREEDY: &[(&str, &str)] = &[("krishiv.optimizer.joinEnumerator", "greedy")];

/// TPC-H's trees (at most 8 inputs) are all planned exactly: `dphyp` finds
/// the same plans as `dpsub`, looking at fewer candidate pairs on trees of
/// six or more inputs, and `greedy` returns the
/// same rows.
#[test]
fn tpch_plans_are_the_same_under_dphyp() {
    let ctx = support::ctx();
    let stats = fixture();
    let db = data::tpch(7, 150);
    let mut fewer = 0;
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &ctx);
        let (sub, sub_trees) = plan_with(&plan, &stats, &[]);
        let (hyp, hyp_trees) = plan_with(&plan, &stats, DPHYP);
        assert!(
            hyp.structurally_eq(&sub),
            "q{q:02}:\n{}\n{}",
            hyp.root(),
            sub.root()
        );
        assert_eq!(sub_trees.len(), hyp_trees.len(), "q{q:02}");
        for (s, h) in sub_trees.iter().zip(&hyp_trees) {
            assert_eq!(s.enumerator, "dpsub");
            assert_eq!(h.enumerator, "dphyp");
            assert_eq!(
                (&s.fallback, &h.fallback),
                (&"none".to_owned(), &"none".to_owned())
            );
            assert_eq!((s.cost, &s.shape), (h.cost, &h.shape), "q{q:02}");
            assert_eq!(h.pairs, s.pairs, "q{q:02}");
            if s.inputs >= 6 {
                assert!(h.considered < s.considered, "q{q:02}: {h:?} {s:?}");
                fewer += 1;
            }
        }
        let (greedy, greedy_trees) = plan_with(&plan, &stats, GREEDY);
        assert!(greedy_trees.iter().all(|t| t.fallback == "forced"));
        assert_eq!(
            eval::rows(&db, greedy.root()),
            eval::rows(&db, plan.root()),
            "q{q:02}"
        );
    }
    assert!(fewer > 0, "dphyp looks at fewer pairs on some tree");
}

/// On every TPC-DS query, `dphyp` plans each tree at most at `dpsub`'s
/// cost: the same plan where `dpsub` was exact, an exact one where it fell
/// back to greedy because the tree was too large.
#[test]
fn tpcds_trees_are_planned_no_worse_under_dphyp() {
    let ctx = support::ctx_over(tpcds::catalog());
    let mut planned = 0;
    for q in 1..=99 {
        let Some(plan) = support::try_resolve(&tpcds::sql(q), &ctx) else {
            continue;
        };
        let (_, sub) = plan_with(&plan, &NoStatistics, &[]);
        let (_, hyp) = plan_with(&plan, &NoStatistics, DPHYP);
        assert_eq!(sub.len(), hyp.len(), "q{q:02}");
        for (s, h) in sub.iter().zip(&hyp) {
            assert_eq!(s.inputs, h.inputs, "q{q:02}");
            if s.fallback == "none" {
                assert_eq!((s.cost, &s.shape), (h.cost, &h.shape), "q{q:02}");
            } else if h.fallback == "none" {
                assert!(h.cost <= s.cost, "q{q:02}: {} > {}", h.cost, s.cost);
            }
            planned += 1;
        }
    }
    assert!(planned > 99, "{planned} trees");
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
    for conf in [&[][..], DPHYP] {
        let (_, held) = plan_with(&plan, &NoStatistics, conf);
        assert_eq!(held[0].fallback, "hyperedge", "{conf:?}");
    }
}

/// `dphyp` plans a tree past `dpsub`'s limit exactly, within its budget.
#[test]
fn dphyp_plans_large_chains_exactly() {
    let mut sql = "SELECT t1.b FROM t t1".to_owned();
    for i in 2..=14 {
        sql.push_str(&format!(" JOIN t t{i} ON t{i}.a = t{}.c", i - 1));
    }
    let plan = support::resolve(&sql, &support::ctx());
    let (_, sub) = plan_with(&plan, &NoStatistics, &[]);
    let (hyp_plan, hyp) = plan_with(&plan, &NoStatistics, DPHYP);
    assert_eq!((sub[0].inputs, sub[0].fallback.as_str()), (14, "inputs"));
    // A chain of 14 has (14³ − 14) / 6 = 455 csg-cmp pairs.
    assert_eq!((hyp[0].fallback.as_str(), hyp[0].pairs), ("none", 455));
    assert!(hyp[0].cost <= sub[0].cost);
    for seed in 1..=2 {
        let db = data::small(seed);
        assert_eq!(
            eval::rows(&db, hyp_plan.root()),
            eval::rows(&db, plan.root())
        );
    }
}
