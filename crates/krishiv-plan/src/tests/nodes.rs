// SPDX-License-Identifier: Apache-2.0
//! Node construction, traversal order, rewrites, resolution state, the
//! canonical operator table and literal equality.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;

use krishiv_types::DataType;

use crate::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, JoinCondition, JoinKind, Literal, LogicalPlan,
    Origin, PlanId, PlanNode, ReadSource, ResolvedExpr, Scan, SortOrder, SourceId, Span,
    SubqueryKind,
};

fn a() -> Expr {
    Expr::column(["a"])
}

fn bound(id: u64) -> Expr {
    Expr::new(ExprNode::Resolved(ResolvedExpr::Column(BoundColumn {
        id: AttrId(id),
        name: "a".to_owned(),
        data_type: DataType::Integer,
        nullable: true,
    })))
}

fn scan() -> LogicalPlan {
    LogicalPlan::new(PlanNode::Scan(Scan {
        source: ReadSource::Table {
            name: vec!["t".to_owned()],
            options: BTreeMap::new(),
        },
        output: Vec::new(),
        version: None,
    }))
}

#[test]
fn builders_nest_children_in_arcs() {
    let plan = LogicalPlan::table(["t"])
        .filter(a())
        .project(vec![a().alias("b")])
        .sort(vec![SortOrder::desc(a())])
        .limit(Expr::from(Literal::Integer(1)));
    let PlanNode::Limit { input, .. } = &plan.node else {
        panic!("{plan:?}")
    };
    let PlanNode::Sort { input, global, .. } = &input.node else {
        panic!("{input:?}")
    };
    assert!(global);
    assert!(matches!(input.node, PlanNode::Project { .. }));
    assert_eq!(plan.origin, Origin::UNKNOWN);
}

#[test]
fn offset_nests_under_limit_and_encodes_apart_from_it() {
    let n = || Expr::from(Literal::Integer(3));
    let plan = LogicalPlan::table(["t"]).offset(n()).limit(n());
    let PlanNode::Limit { input, .. } = &plan.node else {
        panic!("{plan:?}")
    };
    let PlanNode::Offset { offset, .. } = &input.node else {
        panic!("{input:?}")
    };
    assert_eq!(**offset, n());
    assert_eq!(input.to_string(), "Offset 3\n+- Read t\n");
    // The count is an expression of the node: visited, and part of the key.
    let mut seen = 0;
    input.for_each_expr(&mut |_| seen += 1);
    assert_eq!(seen, 1);
    let key = |p: &LogicalPlan| {
        crate::Plan::try_new(p.clone())
            .unwrap()
            .canonical_encoding()
    };
    assert_ne!(
        key(&LogicalPlan::table(["t"]).offset(n())),
        key(&LogicalPlan::table(["t"]).limit(n()))
    );
    assert_ne!(
        key(&LogicalPlan::table(["t"]).offset(n())),
        key(&LogicalPlan::table(["t"]).offset(Expr::from(Literal::Integer(4))))
    );
}

#[test]
fn traversal_visits_children_and_expressions_in_order() {
    let join = LogicalPlan::table(["l"]).join(
        LogicalPlan::table(["r"]),
        JoinKind::Inner,
        JoinCondition::On(Arc::new(a())),
    );
    let mut names = Vec::new();
    join.for_each_child(&mut |c| {
        if let PlanNode::Read(ReadSource::Table { name, .. }) = &c.node {
            names.extend(name.iter().cloned());
        }
    });
    assert_eq!(names, ["l", "r"]);
    let mut exprs = 0;
    join.for_each_expr(&mut |_| exprs += 1);
    assert_eq!(exprs, 1);
}

/// Renames every column `a` to `z`.
fn rename(e: Arc<Expr>) -> Result<Arc<Expr>, Infallible> {
    Ok(match &e.node {
        ExprNode::Column(c) if c.name == ["a"] => Arc::new(Expr::column(["z"])),
        _ => e,
    })
}

#[test]
fn transforms_share_unchanged_subtrees() {
    let b = Arc::new(Expr::column(["b"]));
    let call = Arc::new(Expr::new(ExprNode::Function(crate::FunctionCall {
        name: "f".to_owned(),
        args: vec![Arc::new(a()), Arc::clone(&b)],
        distinct: false,
        filter: None,
    })));
    let renamed = Arc::clone(&call)
        .transform_up(&mut rename)
        .unwrap_or_else(|e| match e {});
    let ExprNode::Function(f) = &renamed.node else {
        panic!("{renamed:?}")
    };
    assert_eq!(*f.args[0], Expr::column(["z"]));
    assert!(Arc::ptr_eq(&f.args[1], &b), "unchanged child shared");
    let same = Arc::clone(&b)
        .transform_down(&mut rename)
        .unwrap_or_else(|e| match e {});
    assert!(Arc::ptr_eq(&same, &b), "no change, same Arc");

    let leaf = Arc::new(LogicalPlan::table(["t"]));
    let plan = Arc::new(LogicalPlan::clone(&leaf).filter(a()).project(vec![a()]));
    let plan = Arc::clone(&plan)
        .transform_down(&mut |p: Arc<LogicalPlan>| {
            p.map_expressions(|e| e.transform_up(&mut rename))
        })
        .unwrap_or_else(|e| match e {});
    let expected = LogicalPlan::clone(&leaf)
        .filter(Expr::column(["z"]))
        .project(vec![Expr::column(["z"])]);
    assert_eq!(*plan, expected);
    let up = Arc::clone(&plan)
        .transform_up(&mut |p: Arc<LogicalPlan>| match &p.node {
            PlanNode::Read(_) => Ok::<_, Infallible>(Arc::new(LogicalPlan::table(["u"]))),
            _ => Ok(p),
        })
        .unwrap_or_else(|e| match e {});
    let expected = LogicalPlan::table(["u"])
        .filter(Expr::column(["z"]))
        .project(vec![Expr::column(["z"])]);
    assert_eq!(*up, expected);
    let err = plan.map_children(|_| Err("stop"));
    assert_eq!(err.err(), Some("stop"));
}

#[test]
fn resolution_state() {
    assert!(!LogicalPlan::table(["t"]).is_resolved());
    assert!(!scan().filter(a()).is_resolved(), "unresolved column");
    assert!(!scan().project(vec![bound(1).alias("x")]).is_resolved());
    assert!(scan().filter(bound(1)).is_resolved());
    let cast = bound(1).cast(DataType::Long);
    assert!(
        !scan().project(vec![cast]).is_resolved(),
        "the analyzer pins CastMode::Default"
    );
    let ansi = Expr::new(ExprNode::Cast {
        expr: Arc::new(bound(1)),
        to: DataType::Long,
        mode: crate::CastMode::Ansi,
    });
    assert!(scan().project(vec![ansi]).is_resolved());
    let sub = Expr::subquery(SubqueryKind::Exists, LogicalPlan::table(["u"]));
    assert!(
        !scan().filter(sub).is_resolved(),
        "unresolved subquery plan"
    );
    let sub = Expr::subquery(SubqueryKind::Exists, scan());
    assert!(scan().filter(sub).is_resolved());
    let using = scan().join(
        scan(),
        JoinKind::Inner,
        JoinCondition::Using(vec!["a".into()]),
    );
    assert!(!using.is_resolved(), "USING is rewritten by the analyzer");
}

#[test]
fn equality_includes_origins() {
    let spanned = a().with_origin(Origin::sql(SourceId(0), Span::new(0, 1)));
    assert_ne!(spanned, a());
    let plan = LogicalPlan::table(["t"]);
    assert_ne!(plan.clone().with_origin(Origin::plan(PlanId(1))), plan);
    let both = Origin::plan(PlanId(3)).with_sql(SourceId(2), Span::new(4, 9));
    assert_eq!(both.plan_id, Some(PlanId(3)));
    assert_eq!(both.sql, Some((SourceId(2), Span::new(4, 9))));
}

#[test]
fn canonical_operator_names() {
    for op in Builtin::ALL {
        assert_eq!(Builtin::from_name(op.name()), Some(op));
        assert_eq!(
            Builtin::from_name(&op.name().to_ascii_uppercase()),
            Some(op)
        );
    }
    assert_eq!(Builtin::from_name("isNull"), Some(Builtin::IsNull));
    for spelling in ["=", "<>", "!=", "&&", "||"] {
        assert_eq!(
            Builtin::from_name(spelling),
            None,
            "{spelling} is not canonical"
        );
    }
}

#[test]
fn literal_equality_is_bitwise() {
    assert_eq!(Literal::Double(f64::NAN), Literal::Double(f64::NAN));
    assert_ne!(Literal::Double(0.0), Literal::Double(-0.0));
    assert_ne!(Literal::Float(0.0), Literal::Float(-0.0));
    assert_ne!(Literal::Integer(1), Literal::Long(1));
    assert_ne!(Literal::Null(None), Literal::Null(Some(DataType::Integer)));
}
