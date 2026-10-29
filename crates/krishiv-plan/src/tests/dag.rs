// SPDX-License-Identifier: Apache-2.0
//! Shared subtrees (`Arc` DAGs): rewrites, `PartialEq` and `Debug` take
//! time linear in the distinct nodes. Each test uses a DAG of 31 distinct
//! nodes that is 2^31 - 1 nodes as a tree, so an expanding traversal would
//! not finish.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::depth::self_joins;
use crate::{
    Builtin, Expr, ExprNode, FunctionCall, LogicalPlan, Origin, PlanId, PlanNode, ReadSource,
};

/// `e = e AND e` 30 times over `leaf`.
fn and_chain(leaf: Expr) -> Arc<Expr> {
    (0..30).fold(Arc::new(leaf), |e, _| {
        Arc::new(Expr::new(ExprNode::Function(FunctionCall {
            name: Builtin::And.name().to_owned(),
            qualifier: Vec::new(),
            arg_names: Vec::new(),
            args: vec![Arc::clone(&e), e],
            distinct: false,
            filter: None,
        })))
    })
}

fn timed(what: &str, f: impl FnOnce()) {
    let start = Instant::now();
    f();
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{what}: {:?}",
        start.elapsed()
    );
}

fn is_table(p: &LogicalPlan) -> bool {
    matches!(&p.node, PlanNode::Read(ReadSource::Table { .. }))
}

/// Renames the leaf table, so every relation above it is rebuilt.
fn rename(p: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, ()> {
    Ok(if is_table(&p) {
        Arc::new(LogicalPlan::table(["u"]))
    } else {
        p
    })
}

#[test]
fn plan_rewrites_visit_each_shared_relation_once() {
    timed("transform", || {
        let plan = Arc::new(self_joins(30));
        for down in [false, true] {
            let mut calls = 0;
            let mut f = |p| {
                calls += 1;
                rename(p)
            };
            let out = if down {
                Arc::clone(&plan).transform_down(&mut f)
            } else {
                Arc::clone(&plan).transform_up(&mut f)
            }
            .unwrap();
            assert_eq!(calls, 31, "down: {down}");
            // The rewrite is shared like the input, and equal to the input
            // over the renamed table.
            let PlanNode::Join { left, right, .. } = &out.node else {
                panic!("{out:?}");
            };
            assert!(Arc::ptr_eq(left, right));
            assert_ne!(out, plan);
            let renamed = (0..30).fold(LogicalPlan::table(["u"]), |p, _| {
                let p = Arc::new(p);
                LogicalPlan::new(PlanNode::Join {
                    left: Arc::clone(&p),
                    right: p,
                    kind: crate::JoinKind::Inner,
                    condition: crate::JoinCondition::None,
                })
            });
            assert_eq!(*out, renamed);
        }
    });
}

#[test]
fn expression_rewrites_visit_each_shared_node_once() {
    timed("transform", || {
        let e = and_chain(Expr::column(["a"]));
        for down in [false, true] {
            let mut calls = 0;
            let mut f = |e: Arc<Expr>| {
                calls += 1;
                Ok::<_, ()>(if matches!(e.node, ExprNode::Column(_)) {
                    Arc::new(Expr::column(["b"]))
                } else {
                    e
                })
            };
            let out = if down {
                Arc::clone(&e).transform_down(&mut f)
            } else {
                Arc::clone(&e).transform_up(&mut f)
            }
            .unwrap();
            assert_eq!(calls, 31, "down: {down}");
            assert_eq!(out, and_chain(Expr::column(["b"])));
        }
    });
}

/// `p = p.join(p)` `n` times where each side is a separate copy of the
/// previous level: the same plan as [`self_joins`] as a tree, built small
/// (`n` levels of two copies would be 2^n nodes, so only a few levels).
fn copied_joins(n: usize) -> LogicalPlan {
    use crate::{JoinCondition, JoinKind};
    if n == 0 {
        return LogicalPlan::table(["t"]);
    }
    LogicalPlan::new(PlanNode::Join {
        left: Arc::new(copied_joins(n - 1)),
        right: Arc::new(copied_joins(n - 1)),
        kind: JoinKind::Inner,
        condition: JoinCondition::None,
    })
}

#[test]
fn normalized_hashes_of_shared_dags_are_linear_and_equal_to_copies() {
    use crate::{Plan, normalized_plan_hash, normalized_subtree_hash};
    timed("hash", || {
        let shared = Plan::try_new(self_joins(30)).unwrap();
        let again = Plan::try_new(self_joins(30)).unwrap();
        assert_eq!(normalized_plan_hash(&shared), normalized_plan_hash(&again));
        assert_ne!(
            normalized_plan_hash(&shared),
            normalized_plan_hash(&Plan::try_new(self_joins(29)).unwrap())
        );
        // Every distinct relation, as `RelationCache::substitute` hashes them.
        let mut node = Arc::clone(shared.root());
        for _ in 0..30 {
            let PlanNode::Join { left, .. } = &node.node else {
                panic!("{node:?}")
            };
            let next = Arc::clone(left);
            assert_ne!(normalized_subtree_hash(&node), 0);
            node = next;
        }
    });
    // Sharing is invisible to the hash: a plan whose two sides are separate
    // copies hashes equal to the one that shares them.
    for n in 0..=8 {
        assert_eq!(
            normalized_subtree_hash(&self_joins(n)),
            normalized_subtree_hash(&copied_joins(n)),
            "{n} levels"
        );
    }
}

#[test]
fn normalized_hashes_of_shared_expressions_are_linear() {
    use crate::normalized_subtree_hash;
    timed("expr hash", || {
        let plan = LogicalPlan::new(PlanNode::Filter {
            input: Arc::new(LogicalPlan::table(["t"])),
            predicate: and_chain(Expr::column(["a"])),
        });
        let other = LogicalPlan::new(PlanNode::Filter {
            input: Arc::new(LogicalPlan::table(["t"])),
            predicate: and_chain(Expr::column(["b"])),
        });
        assert_eq!(
            normalized_subtree_hash(&plan),
            normalized_subtree_hash(&plan)
        );
        assert_ne!(
            normalized_subtree_hash(&plan),
            normalized_subtree_hash(&other)
        );
    });
}

#[test]
fn equality_of_separately_built_dags_is_linear() {
    timed("eq", || {
        assert_eq!(self_joins(30), self_joins(30));
        assert_eq!(
            and_chain(Expr::column(["a"])),
            and_chain(Expr::column(["a"]))
        );
        assert_ne!(
            and_chain(Expr::column(["a"])),
            and_chain(Expr::column(["b"]))
        );
        // Origins take part in `PartialEq` (not in the structural key).
        let with_origin = Expr::column(["a"]).with_origin(Origin::plan(PlanId(1)));
        assert_ne!(and_chain(Expr::column(["a"])), and_chain(with_origin));
        let other_leaf = LogicalPlan::table(["t"]).with_origin(Origin::plan(PlanId(1)));
        let p = Arc::new(other_leaf);
        let other = LogicalPlan::new(PlanNode::Join {
            left: Arc::clone(&p),
            right: p,
            kind: crate::JoinKind::Inner,
            condition: crate::JoinCondition::None,
        });
        assert_ne!(self_joins(1), other);
    });
}

#[test]
fn debug_prints_shared_nodes_once() {
    timed("debug", || {
        let plan = format!("{:?}", self_joins(30));
        assert!(plan.len() < 16_384, "{}", plan.len());
        assert_eq!(plan.matches("(shared, printed above)").count(), 30);
        assert!(plan.contains("LogicalPlan #1 { node: Join"));
        let expr = format!("{:?}", and_chain(Expr::column(["a"])));
        assert!(expr.len() < 16_384, "{}", expr.len());
        assert!(expr.contains("Expr #30 (shared, printed above)"));
    });
    // A tree without sharing prints as the derived form would.
    let tree = LogicalPlan::table(["t"]).filter(Expr::column(["a"]));
    let text = format!("{tree:?}");
    assert!(text.starts_with("LogicalPlan { node: Filter { input: LogicalPlan { node: Read"));
    assert!(!text.contains('#'));
}
