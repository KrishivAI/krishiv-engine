// SPDX-License-Identifier: Apache-2.0
//! Node construction, traversal order, the canonical operator table and
//! literal equality.

use std::sync::Arc;

use krishiv_types::DataType;

use crate::{
    Builtin, Expr, JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PlanId, PlanNode,
    ReadSource, SortOrder, SourceId, Span,
};

fn a() -> Expr {
    Expr::column(["a"])
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
