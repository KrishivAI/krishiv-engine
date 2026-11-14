// SPDX-License-Identifier: Apache-2.0
//! Column lineage on plans built in-crate (P3-T063 N1a): what the SQL tests in `krishiv-sql` cannot reach.
//! The memo (a CTE referenced twice is computed once, a shared expression DAG once per node), the deep
//! plans, and the typed errors for nodes an analyzed plan never holds. The per-row tests are in
//! `crates/krishiv-sql/tests/lineage.rs`.

use std::sync::Arc;

use krishiv_common::error::ErrorClass;
use krishiv_types::DataType;

use super::dsl::{c, eq, int, op};
use crate::analyzer::lineage::{Transform, lineage, lineage_counting};
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalyzedPlan, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use crate::{Builtin, Cte, Expr, ExprNode, FunctionCall, LogicalPlan, Plan, PlanNode, ReadSource};

fn catalog() -> MemoryCatalog {
    let col = |n: &str| TableColumn {
        name: n.to_owned(),
        data_type: DataType::Integer,
        nullable: true,
        char_varchar: None,
    };
    MemoryCatalog::new("spark_catalog")
        .with_table(&["default"], "t", vec![col("x"), col("y")], None)
        .with_table(&["default"], "u", vec![col("z")], None)
}

fn analyze(plan: LogicalPlan) -> AnalyzedPlan {
    let ctx = context(catalog(), ResolutionConf::default());
    block_on(DefaultAnalyzer.analyze(&Plan::try_new(plan).unwrap(), &ctx)).unwrap()
}

/// `WITH q AS (t, filtered `depth` times) SELECT .. FROM q [JOIN q q2]`.
fn cte(depth: usize, twice: bool) -> LogicalPlan {
    let body = (0..depth).fold(LogicalPlan::table(["t"]), |p, i| {
        p.filter(eq(c("x"), int(i32::try_from(i).unwrap())))
    });
    let read = |alias: &str| LogicalPlan::table(["q"]).alias(alias);
    let input = if twice {
        read("q").join(
            read("q2"),
            crate::JoinKind::Cross,
            crate::JoinCondition::None,
        )
    } else {
        read("q")
    };
    LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(input),
        ctes: vec![Cte {
            name: "q".to_owned(),
            plan: Arc::new(body),
        }],
    })
}

#[test]
fn a_cte_referenced_twice_is_computed_once() {
    let (once, n1) = lineage_counting(&analyze(cte(40, false))).unwrap();
    let (twice, n2) = lineage_counting(&analyze(cte(40, true))).unwrap();
    let (n1, n2) = (n1.relations, n2.relations);
    assert_eq!(once.columns.len(), 2);
    assert_eq!(twice.columns.len(), 4);
    assert_eq!(twice.columns[2].sources, once.columns[0].sources);
    // The second reference adds its own few nodes, never the 40-filter definition again.
    assert!(n2 > n1 && n2 - n1 < 10, "{n1} then {n2}");
}

#[test]
fn shared_expression_dags_are_walked_once() {
    // `e AND e` 30 times: 2^30 nodes as a tree.
    let e = and_dag(eq(c("x"), int(1)), 30);
    let a = analyze(LogicalPlan::table(["t"]).filter(e));
    let (l, counts) = lineage_counting(&a).unwrap();
    assert_eq!(l.filters.len(), 1);
    // Each distinct node once: 30 `AND`s, the `=`, its column and literal (the analyzer may wrap a few more,
    // never the 2^30 of the tree).
    assert!(counts.expressions < 100, "{counts:?}");
}

/// `e AND e`, `n` times over, as one shared DAG (2^n nodes as a tree).
fn and_dag(e: Expr, n: usize) -> Arc<Expr> {
    (0..n).fold(Arc::new(e), |e, _| {
        Arc::new(Expr::new(ExprNode::Function(FunctionCall {
            within_group: Vec::new(),
            name: Builtin::And.name().to_owned(),
            qualifier: Vec::new(),
            arg_names: Vec::new(),
            args: vec![Arc::clone(&e), e],
            distinct: false,
            filter: None,
        })))
    })
}

#[test]
fn a_semi_join_on_a_shared_condition_dag_is_probed_in_bounded_time() {
    // `x <=> z`, then `e AND e` 30 times: about 30 nodes, 2^30 `<=>` pairs as a tree. The set-operation
    // probe stops once the pairs outnumber the columns.
    let condition = and_dag(op(Builtin::EqNullSafe, c("x"), c("z")), 30);
    let plan = LogicalPlan::table(["t"]).join(
        LogicalPlan::table(["u"]),
        crate::JoinKind::LeftSemi,
        crate::JoinCondition::On(condition),
    );
    let start = std::time::Instant::now();
    let l = lineage(&analyze(plan)).unwrap();
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
    // Not the set operations' rewrite: the left side's columns as they are.
    assert_eq!(l.columns.len(), 2);
    assert_eq!(l.columns[0].transform, Transform::Identity);
    assert_eq!(l.columns[0].sources.len(), 1);
}

#[test]
fn deep_plans_do_not_overflow_the_stack() {
    // Close to `MAX_DEPTH` relations, on a test thread's stack.
    let plan = (0..crate::MAX_DEPTH / 2 - 10).fold(LogicalPlan::table(["t"]), |p, i| {
        p.filter(eq(c("x"), int(i32::try_from(i).unwrap())))
    });
    let l = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || lineage(&analyze(plan)))
        .unwrap()
        .join()
        .unwrap()
        .unwrap();
    assert_eq!(l.columns[0].transform, Transform::Identity);
}

/// `a` with its root replaced by `f(root)`.
fn with_root(a: &AnalyzedPlan, f: impl FnOnce(Arc<LogicalPlan>) -> LogicalPlan) -> AnalyzedPlan {
    AnalyzedPlan {
        plan: Plan::try_new(f(Arc::clone(a.plan.root()))).unwrap(),
        ..a.clone()
    }
}

#[test]
fn nodes_an_analyzed_plan_never_holds_are_internal_errors() {
    let a = analyze(LogicalPlan::table(["t"]));
    let partial = with_root(&a, |input| {
        LogicalPlan::new(PlanNode::PartialAggregate {
            input,
            keys: Vec::new(),
            aggregates: Vec::new(),
            states: Vec::new(),
        })
    });
    let unresolved = with_root(&a, |_| {
        LogicalPlan::new(PlanNode::Read(ReadSource::Table {
            name: vec!["t".to_owned()],
            options: std::collections::BTreeMap::new(),
        }))
    });
    let unresolved_expr = with_root(&a, |input| {
        LogicalPlan::new(PlanNode::Project {
            input,
            exprs: vec![Arc::new(c("x"))],
        })
    });
    let loose_ref = with_root(&a, |_| {
        LogicalPlan::new(PlanNode::UnionLoopRef {
            id: 7,
            output: a.output.clone(),
        })
    });
    for (what, p) in [
        ("partial aggregate", partial),
        ("unresolved relation", unresolved),
        ("unresolved expression", unresolved_expr),
        ("recursive reference outside its loop", loose_ref),
    ] {
        let e = lineage(&p).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{what}");
    }
    // An output that does not match the analyzed one.
    let mut wrong = a;
    wrong.output.pop();
    assert_eq!(
        lineage(&wrong).unwrap_err().error_class(),
        ErrorClass::InternalError
    );
}
