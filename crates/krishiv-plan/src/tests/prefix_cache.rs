// SPDX-License-Identifier: Apache-2.0
//! The analysis prefix cache (`CachingAnalyzer`, design P3-T013a): a warm analysis equals the cold one,
//! and anything the prefix read that moved is a miss.

use std::sync::Arc;

use crate::analyzer::prefix::WARM_STARTS;
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisContext, Analyzer, CachingAnalyzer, DefaultAnalyzer, DenyAll, MemoryCatalog, Principal,
    PrincipalKind, ResolutionConf, TableColumn,
};
use crate::{
    ColumnRef, Expr, ExprNode, JoinCondition, JoinKind, LogicalPlan, Origin, Plan, PlanId, PlanNode,
};

use super::dsl::{add, int};

fn table(cols: &[&str]) -> MemoryCatalog {
    let cols = cols
        .iter()
        .map(|n| TableColumn::new((*n).to_owned(), krishiv_types::DataType::Integer, true))
        .collect();
    MemoryCatalog::new("spark_catalog").with_table(&["default"], "t", cols, None)
}

fn ctx(cols: &[&str]) -> AnalysisContext {
    let cat = table(cols).with_table(
        &["default"],
        "u",
        vec![TableColumn::new(
            "x".to_owned(),
            krishiv_types::DataType::Integer,
            true,
        )],
        None,
    );
    context(cat, ResolutionConf::default())
}

fn col(name: &str, id: i64) -> Expr {
    Expr::new(ExprNode::Column(ColumnRef {
        plan_id: Some(PlanId(id)),
        ..ColumnRef::new(vec![name.to_owned()])
    }))
}

/// The chain after `steps` `withColumn` calls over `t`, as a client resends it.
fn chain(steps: i64) -> LogicalPlan {
    let mut plan = LogicalPlan::table(["t"]).with_origin(Origin::plan(PlanId(0)));
    for i in 1..=steps {
        let prev = if i == 1 {
            "a".to_owned()
        } else {
            format!("c{}", i - 1)
        };
        let expr = add(col(&prev, i - 1), int(1));
        plan = LogicalPlan::new(PlanNode::WithColumns {
            input: Arc::new(plan),
            columns: vec![(format!("c{i}"), Arc::new(expr))],
        })
        .with_origin(Origin::plan(PlanId(i)));
    }
    plan
}

fn starts() -> usize {
    WARM_STARTS.with(std::cell::Cell::get)
}

/// Analyses `plan` warm (through `cache`) and cold, and asserts the same result or the same error.
fn same(cache: &CachingAnalyzer, plan: LogicalPlan, c: &AnalysisContext) -> bool {
    let p = Plan::try_new(plan).unwrap();
    let before = starts();
    let warm = block_on(cache.analyze(&p, c));
    let hit = starts() > before;
    let cold = block_on(DefaultAnalyzer.analyze(&p, c));
    match (warm, cold) {
        (Ok(w), Ok(k)) => {
            assert_eq!(w, k);
            assert_eq!(w.output_declared, k.output_declared);
            assert_eq!(w.output_comments, k.output_comments);
        }
        (Err(w), Err(k)) => assert_eq!(w, k),
        (w, k) => panic!("warm {w:?} but cold {k:?}"),
    }
    hit
}

#[test]
fn a_growing_chain_analyses_warm_like_cold() {
    let c = ctx(&["a", "b", "c"]);
    let cache = CachingAnalyzer::new();
    assert!(!same(&cache, chain(1), &c), "the first call is cold");
    for depth in 2..=40 {
        assert!(
            same(&cache, chain(depth), &c),
            "depth {depth} starts from the last root"
        );
    }
    // The same plan again: the whole of it is the prefix.
    assert!(same(&cache, chain(40), &c));
    // A shorter plan is not the old root plus steps.
    assert!(!same(&cache, chain(10), &c));
}

#[test]
fn a_changed_table_is_a_miss() {
    let cache = CachingAnalyzer::new();
    assert!(!same(&cache, chain(5), &ctx(&["a", "b", "c"])));
    // Same fingerprint (no epoch moved), another schema: the metadata check refuses the prefix.
    assert!(!same(&cache, chain(6), &ctx(&["a", "b", "c", "d"])));
    // The entry now holds the new schema's root.
    assert!(same(&cache, chain(7), &ctx(&["a", "b", "c", "d"])));
}

#[test]
fn another_principal_is_analysed_on_its_own() {
    let allow = ctx(&["a", "b", "c"]);
    let mut deny = ctx(&["a", "b", "c"]);
    deny.policy = Arc::new(DenyAll);
    deny.principal = Arc::new(Principal::new("mallory", PrincipalKind::User));
    let cache = CachingAnalyzer::new();
    assert!(!same(&cache, chain(5), &allow));
    let p = Plan::try_new(chain(6)).unwrap();
    assert!(
        block_on(cache.analyze(&p, &deny)).is_err(),
        "a denial is never served warm"
    );
    // Back to the allowed principal.
    same(&cache, chain(7), &allow);
}

#[test]
fn steps_that_join_or_move_the_prefix_stay_exact() {
    let c = ctx(&["a", "b", "c"]);
    let cache = CachingAnalyzer::new();
    same(&cache, chain(4), &c);
    // A join with the old root on the left: a hit that adds a securable.
    let left = chain(4).join(
        LogicalPlan::table(["u"]),
        JoinKind::Inner,
        JoinCondition::None,
    );
    assert!(same(&cache, left, &c));
    // The old root on the right of a join: not on the first-child spine, a miss.
    same(&cache, chain(4), &c);
    let right = LogicalPlan::table(["u"]).join(chain(4), JoinKind::Inner, JoinCondition::None);
    assert!(!same(&cache, right, &c));
    // The old root joined with itself: an equal relation outside the prefix, a miss.
    same(&cache, chain(4), &c);
    let twice = chain(4).join(chain(4), JoinKind::Inner, JoinCondition::None);
    assert!(!same(&cache, twice, &c));
}
