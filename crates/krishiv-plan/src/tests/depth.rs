// SPDX-License-Identifier: Apache-2.0
//! The depth cap (owner decision 6): a plan at [`MAX_DEPTH`] survives every
//! traversal on a 2 MiB stack; a deeper one is rejected with a Spark error
//! class and dropped without overflowing the stack.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use krishiv_common::error::ErrorClass;

use crate::{
    Builtin, Expr, JoinCondition, JoinKind, LogicalPlan, MAX_DEPTH, Plan, PlanNode, check_depth,
};

/// Stack of a tokio worker.
const WORKER_STACK: usize = 2 << 20;

/// Instrumented builds (`cargo llvm-cov` sets `LLVM_PROFILE_FILE` for the test
/// binary) spill counters and keep more locals live, so each resolver frame is
/// several times larger. The depth limit is a property of the uninstrumented
/// frame size; scale the stack so the same nesting fits.
const INSTRUMENTED_STACK: usize = 16 << 20;

/// The stack size [`on_small_stack`] uses given the profile-file variable.
pub(super) fn small_stack_size(profile_file: Option<&std::ffi::OsStr>) -> usize {
    if profile_file.is_some() {
        INSTRUMENTED_STACK
    } else {
        WORKER_STACK
    }
}

/// A thread with a tokio worker's 2 MiB stack (larger under coverage
/// instrumentation, see [`INSTRUMENTED_STACK`]).
pub(super) fn on_small_stack(f: impl FnOnce() + Send + 'static) {
    thread::Builder::new()
        .stack_size(small_stack_size(
            std::env::var_os("LLVM_PROFILE_FILE").as_deref(),
        ))
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn instrumented_builds_get_a_larger_stack() {
    use std::ffi::OsStr;
    assert_eq!(small_stack_size(None), 2 << 20);
    assert_eq!(small_stack_size(Some(OsStr::new("x.profraw"))), 16 << 20);
}

/// A chain of `n` filters over a table: depth `n + 1`.
fn filters(n: usize) -> LogicalPlan {
    (0..n).fold(LogicalPlan::table(["t"]), |p, _| {
        p.filter(Expr::column(["a"]))
    })
}

/// A table filtered by an expression nested `n` deep: depth `n + 1`.
fn nested_expr(n: usize) -> LogicalPlan {
    let e = (1..n).fold(Expr::column(["a"]), |e, _| {
        Expr::builtin(Builtin::Not, vec![e])
    });
    LogicalPlan::table(["t"]).filter(e)
}

fn exercise(build: fn() -> LogicalPlan) {
    let plan = Plan::try_new(build()).unwrap();
    let copy = plan.clone();
    assert_eq!(copy, plan);
    // Built apart: `PartialEq` and `Debug` walk every level.
    let apart = Plan::try_new(build()).unwrap();
    assert_eq!(apart, plan);
    assert!(format!("{apart:?}").len() > MAX_DEPTH);
    assert!(copy.structurally_eq(&plan));
    assert_eq!(plan.subtree_hashes()[0].1, plan.structural_hash());
    assert!(!plan.to_string().is_empty());
    assert!(!plan.root().is_resolved());
    let root = Arc::clone(plan.root());
    let same = Arc::clone(&root)
        .transform_up(&mut |p: Arc<LogicalPlan>| {
            p.map_expressions(|e| e.transform_down(&mut Ok::<_, ()>))
        })
        .unwrap();
    assert!(Arc::ptr_eq(&same, &root));
}

#[test]
fn plans_at_the_cap_survive_every_traversal() {
    on_small_stack(|| exercise(|| filters(MAX_DEPTH - 1)));
    on_small_stack(|| exercise(|| nested_expr(MAX_DEPTH - 1)));
}

#[test]
fn deeper_plans_are_rejected_and_dropped() {
    for plan in [filters(MAX_DEPTH), nested_expr(MAX_DEPTH)] {
        let err = check_depth(&plan).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::FailedToParseTooComplex);
        assert_eq!(err.sql_state(), Some("54001"));
        assert!(Plan::try_new(plan.clone()).is_err());
        assert_eq!(plan.to_string(), "<nested deeper than the maximum depth>\n");
    }
    on_small_stack(|| {
        let deep = filters(200_000);
        assert!(check_depth(&deep).is_err());
        let wide_and_deep = nested_expr(200_000);
        assert!(!wide_and_deep.is_resolved());
        drop((deep, wide_and_deep));
    });
}

/// `p = p.join(p)` `n` times: `n + 1` distinct relations, `2^(n+1) - 1` as
/// a tree.
pub(super) fn self_joins(n: usize) -> LogicalPlan {
    (0..n).fold(LogicalPlan::table(["t"]), |p, _| {
        let p = Arc::new(p);
        LogicalPlan::new(PlanNode::Join {
            left: Arc::clone(&p),
            right: p,
            kind: JoinKind::Inner,
            condition: JoinCondition::None,
        })
    })
}

#[test]
fn shared_subtrees_are_walked_once() {
    let start = Instant::now();
    let plan = self_joins(30);
    let plan = Plan::try_new(plan).unwrap();
    assert!(!plan.root().is_resolved());
    let transformed = Arc::clone(plan.root()).map_children(Ok::<_, ()>).unwrap();
    assert!(Arc::ptr_eq(&transformed, plan.root()));
    drop(plan);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
}

/// A subtree reached at two depths counts at the deeper one, even when the
/// shallower path is walked first.
#[test]
fn shared_subtrees_count_at_their_deepest_place() {
    let shared = Arc::new(filters(MAX_DEPTH - 6));
    let join = |above: usize| {
        let right = (0..above).fold(LogicalPlan::clone(&shared), |p, _| {
            p.filter(Expr::column(["a"]))
        });
        LogicalPlan::new(PlanNode::Join {
            left: Arc::clone(&shared),
            right: Arc::new(right),
            kind: JoinKind::Inner,
            condition: JoinCondition::None,
        })
    };
    // Height of `shared` is MAX_DEPTH - 5; the join adds 1, the filters `above`.
    assert!(check_depth(&join(4)).is_ok());
    assert!(check_depth(&join(5)).is_err());
}
