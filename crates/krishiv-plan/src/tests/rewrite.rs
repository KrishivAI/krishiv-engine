// SPDX-License-Identifier: Apache-2.0
//! Rewrites of every node kind: `map_children` / `map_expressions` write
//! each slot back in the order the `for_each_*` visitors read them, keep
//! the other fields, and return the same `Arc` when nothing changes. Also
//! the builders, literal equality and origins.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_types::{DataType, DecimalType};

use crate::{
    AttrId, BoundColumn, CastMode, Cte, DayTimeField, Expr, ExprNode, FunctionId, FunctionKind,
    Grouping, JoinCondition, JoinKind, Literal, LogicalPlan, Origin, Plan, PlanId, PlanNode,
    ResolvedExpr, ResolvedFunction, SortOrder, SourceId, Span, SqlArgs, SubqueryKind,
    YearMonthField,
};

fn t(name: &str) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::table([name]))
}

fn col(name: &str) -> Arc<Expr> {
    Arc::new(Expr::column([name]))
}

fn int(i: i32) -> Arc<Expr> {
    Arc::new(Literal::Integer(i).into())
}

fn bound(name: &str) -> BoundColumn {
    BoundColumn {
        id: AttrId(7),
        name: name.to_owned(),
        data_type: DataType::Integer,
        nullable: false,
    }
}

/// One relation of every kind that has children or expressions, with the
/// number of each it holds.
fn plans() -> Vec<(LogicalPlan, usize, usize)> {
    let node = LogicalPlan::new;
    vec![
        (
            node(PlanNode::Sql {
                query: "select ?".to_owned(),
                args: SqlArgs::Positional(vec![col("a"), col("b")]),
            }),
            0,
            2,
        ),
        (
            node(PlanNode::Sql {
                query: "select :x".to_owned(),
                args: SqlArgs::Named(BTreeMap::from([("x".to_owned(), col("a"))])),
            }),
            0,
            1,
        ),
        (
            node(PlanNode::Project {
                input: t("t"),
                exprs: vec![col("a"), col("b")],
            }),
            1,
            2,
        ),
        (
            node(PlanNode::WithColumns {
                input: t("t"),
                columns: vec![("x".to_owned(), col("a"))],
            }),
            1,
            1,
        ),
        (LogicalPlan::table(["t"]).filter(Expr::column(["a"])), 1, 1),
        (LogicalPlan::table(["t"]).limit(int(0)), 1, 1),
        (
            LogicalPlan::table(["t"])
                .aggregate(vec![Expr::column(["a"])], vec![Expr::column(["b"])]),
            1,
            2,
        ),
        (aggregate(Grouping::Rollup(vec![col("a")])), 1, 2),
        (aggregate(Grouping::Cube(vec![col("a")])), 1, 2),
        (
            aggregate(Grouping::GroupingSets {
                exprs: vec![col("a"), col("b")],
                sets: vec![vec![col("a")], vec![]],
            }),
            1,
            4,
        ),
        (
            LogicalPlan::table(["l"]).join(
                LogicalPlan::table(["r"]),
                JoinKind::LeftOuter,
                JoinCondition::On(col("a")),
            ),
            2,
            1,
        ),
        (
            LogicalPlan::table(["l"]).join(
                LogicalPlan::table(["r"]),
                JoinKind::Inner,
                JoinCondition::Using(vec!["a".to_owned()]),
            ),
            2,
            0,
        ),
        (
            LogicalPlan::table(["t"])
                .sort(vec![SortOrder::asc(col("a")), SortOrder::desc(col("b"))]),
            1,
            2,
        ),
        (LogicalPlan::table(["t"]).alias("x"), 1, 0),
        (
            node(PlanNode::WithCte {
                input: t("main"),
                ctes: vec![
                    Cte {
                        name: "c1".to_owned(),
                        plan: t("c1"),
                    },
                    Cte {
                        name: "c2".to_owned(),
                        plan: t("c2"),
                    },
                ],
            }),
            3,
            0,
        ),
    ]
}

fn aggregate(grouping: Grouping) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Aggregate {
        input: t("t"),
        grouping,
        aggregates: vec![col("c")],
    })
}

fn children_of(p: &LogicalPlan) -> Vec<Arc<LogicalPlan>> {
    let mut out = Vec::new();
    p.for_each_child(&mut |c| out.push(Arc::clone(c)));
    out
}

fn exprs_of(p: &LogicalPlan) -> Vec<Arc<Expr>> {
    let mut out = Vec::new();
    p.for_each_expr(&mut |e| out.push(Arc::clone(e)));
    out
}

#[test]
fn every_plan_slot_is_rewritten_in_order() {
    for (plan, n_children, n_exprs) in plans() {
        let plan = Arc::new(plan);
        assert_eq!(children_of(&plan).len(), n_children, "{plan:?}");
        assert_eq!(exprs_of(&plan).len(), n_exprs, "{plan:?}");

        let same = Arc::clone(&plan).map_children(Ok::<_, ()>).unwrap();
        assert!(Arc::ptr_eq(&same, &plan));
        let same = Arc::clone(&plan).map_expressions(Ok::<_, ()>).unwrap();
        assert!(Arc::ptr_eq(&same, &plan));

        let mut i = 0;
        let renamed = Arc::clone(&plan)
            .map_children(|_| {
                i += 1;
                Ok::<_, ()>(t(&format!("n{i}")))
            })
            .unwrap();
        let want: Vec<_> = (1..=n_children).map(|i| t(&format!("n{i}"))).collect();
        assert_eq!(children_of(&renamed), want, "{plan:?}");
        assert_eq!(exprs_of(&renamed), exprs_of(&plan), "{plan:?}");

        let mut i = 0;
        let numbered = Arc::clone(&plan)
            .map_expressions(|_| {
                i += 1;
                Ok::<_, ()>(int(i))
            })
            .unwrap();
        let want: Vec<_> = (1..=n_exprs)
            .map(|i| int(i32::try_from(i).unwrap()))
            .collect();
        assert_eq!(exprs_of(&numbered), want, "{plan:?}");
        assert_eq!(children_of(&numbered), children_of(&plan), "{plan:?}");
        if n_exprs > 0 {
            assert_ne!(numbered, plan);
        }
    }
}

#[test]
fn rewrite_errors_stop_the_rewrite() {
    let plan = Arc::new(LogicalPlan::table(["t"]).filter(Expr::column(["a"])));
    assert_eq!(
        Arc::clone(&plan).map_children(|_| Err("child")),
        Err("child")
    );
    assert_eq!(
        Arc::clone(&plan).map_expressions(|_| Err("expr")),
        Err("expr")
    );
    assert_eq!(Arc::clone(&plan).transform_up(&mut |_| Err(1)), Err(1));
    assert_eq!(Arc::clone(&plan).transform_down(&mut |_| Err(2)), Err(2));
    let e = Arc::new(Expr::column(["a"]).alias("b"));
    assert_eq!(Arc::clone(&e).map_children(|_| Err(3)), Err(3));
    assert_eq!(Arc::clone(&e).transform_up(&mut |_| Err(4)), Err(4));
    assert_eq!(e.transform_down(&mut |_| Err(5)), Err(5));
}

fn resolved_call(args: Vec<Arc<Expr>>, filter: Option<Arc<Expr>>) -> Expr {
    Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Registry {
                name: "f".to_owned(),
                id: 1,
            },
            kind: FunctionKind::Scalar,
            args,
            distinct: false,
            filter,
            data_type: DataType::Integer,
            nullable: true,
            deterministic: true,
        },
    )))
}

/// One expression of every kind, with its number of child expressions.
fn exprs() -> Vec<(Expr, usize)> {
    let subquery = |kind| Expr::subquery(kind, LogicalPlan::table(["s"]));
    vec![
        (Expr::column(["a"]), 0),
        (Literal::Null(None).into(), 0),
        (
            Expr::new(ExprNode::Star(crate::Star {
                qualifier: Vec::new(),
                plan_id: None,
            })),
            0,
        ),
        (
            Expr::new(ExprNode::Function(crate::FunctionCall {
                name: "f".to_owned(),
                args: vec![col("a"), col("b")],
                distinct: true,
                filter: Some(col("c")),
            })),
            3,
        ),
        (Expr::column(["a"]).cast(DataType::Long), 1),
        (Expr::column(["a"]).alias("b"), 1),
        (subquery(SubqueryKind::Scalar), 0),
        (subquery(SubqueryKind::Exists), 0),
        (subquery(SubqueryKind::In(vec![col("a"), col("b")])), 2),
        (
            Expr::new(ExprNode::Resolved(ResolvedExpr::Column(bound("a")))),
            0,
        ),
        (
            Expr::new(ExprNode::Resolved(ResolvedExpr::OuterColumn(bound("a")))),
            0,
        ),
        (resolved_call(vec![col("a")], Some(col("b"))), 2),
        (
            Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                expr: col("a"),
                output: bound("x"),
            })),
            1,
        ),
    ]
}

fn expr_children(e: &Expr) -> Vec<Arc<Expr>> {
    let mut out = Vec::new();
    e.for_each_child(&mut |c| out.push(Arc::clone(c)));
    out
}

#[test]
fn every_expression_slot_is_rewritten_in_order() {
    for (e, n) in exprs() {
        let e = Arc::new(e);
        assert_eq!(expr_children(&e).len(), n, "{e:?}");
        assert!(Arc::ptr_eq(
            &Arc::clone(&e).map_children(Ok::<_, ()>).unwrap(),
            &e
        ));
        let mut i = 0;
        let numbered = Arc::clone(&e)
            .map_children(|_| {
                i += 1;
                Ok::<_, ()>(int(i))
            })
            .unwrap();
        let want: Vec<_> = (1..=n).map(|i| int(i32::try_from(i).unwrap())).collect();
        assert_eq!(expr_children(&numbered), want, "{e:?}");
        if n > 0 {
            assert_ne!(numbered, e);
            assert_eq!(
                std::mem::discriminant(&numbered.node),
                std::mem::discriminant(&e.node)
            );
        }
    }
}

#[test]
fn expression_resolution() {
    let resolved = [
        Expr::new(ExprNode::Resolved(ResolvedExpr::Column(bound("a")))),
        Literal::Integer(1).into(),
        Expr::new(ExprNode::Resolved(ResolvedExpr::Column(bound("a")))).cast(DataType::Long),
        resolved_call(vec![int(1)], None),
    ];
    let mut cast = resolved[2].clone();
    for (r, mode) in [(false, CastMode::Default), (true, CastMode::Legacy)] {
        if let ExprNode::Cast { mode: m, .. } = &mut cast.node {
            *m = mode;
        }
        assert_eq!(cast.is_resolved(), r, "{mode:?}");
    }
    for e in &resolved[..2] {
        assert!(e.is_resolved(), "{e:?}");
    }
    assert!(resolved[3].is_resolved());
    for e in [
        Expr::column(["a"]),
        resolved_call(vec![col("a")], None),
        Expr::column(["a"]).alias("b"),
    ] {
        assert!(!e.is_resolved(), "{e:?}");
    }
}

#[test]
fn builders_set_their_fields() {
    let agg = LogicalPlan::table(["t"]).aggregate(vec![Expr::column(["k"])], vec![]);
    let PlanNode::Aggregate {
        grouping: Grouping::GroupBy(keys),
        aggregates,
        ..
    } = &agg.node
    else {
        panic!("{agg:?}");
    };
    assert_eq!(keys.as_slice(), &[col("k")]);
    assert!(aggregates.is_empty());
    let alias = LogicalPlan::table(["t"]).alias("x");
    assert!(matches!(
        &alias.node,
        PlanNode::SubqueryAlias { alias, columns, .. } if alias == "x" && columns.is_empty()
    ));
}

#[test]
fn literals_compare_by_value_and_type() {
    let decimal = |unscaled, scale| Literal::Decimal {
        unscaled,
        data_type: DecimalType::try_new(10, scale).unwrap(),
    };
    let ym = |start| Literal::YearMonthInterval {
        months: 14,
        start,
        end: YearMonthField::Month,
    };
    let dt = |end| Literal::DayTimeInterval {
        micros: 1,
        start: DayTimeField::Day,
        end,
    };
    let values = [
        Literal::Null(None),
        Literal::Null(Some(DataType::Integer)),
        Literal::Boolean(true),
        Literal::Byte(1),
        Literal::Short(1),
        Literal::Integer(1),
        Literal::Long(1),
        Literal::Float(1.0),
        Literal::Double(1.0),
        decimal(1, 0),
        decimal(1, 1),
        Literal::String("1".to_owned()),
        Literal::Binary(vec![1]),
        Literal::Date(1),
        Literal::Timestamp(1),
        Literal::TimestampNtz(1),
        ym(YearMonthField::Year),
        ym(YearMonthField::Month),
        dt(DayTimeField::Hour),
        dt(DayTimeField::Second),
    ];
    for (i, a) in values.iter().enumerate() {
        for (j, b) in values.iter().enumerate() {
            assert_eq!(a == b, i == j, "{a:?} vs {b:?}");
        }
    }
    // By bits: NaN equals itself, 0.0 and -0.0 differ.
    assert_eq!(Literal::Float(f32::NAN), Literal::Float(f32::NAN));
    assert_ne!(Literal::Double(0.0), Literal::Double(-0.0));
}

#[test]
fn origins() {
    let span = Span::new(3, 9);
    assert_eq!(span.to_string(), "3..9");
    assert_eq!(PlanId(42).to_string(), "42");
    let o = Origin::sql(SourceId(1), span).with_plan_id(PlanId(2));
    assert_eq!(o.plan_id, Some(PlanId(2)));
    assert_eq!(o.sql, Some((SourceId(1), span)));
    assert_eq!(
        Origin::plan(PlanId(2)).with_sql(SourceId(1), span),
        o,
        "builders commute"
    );
}

#[test]
fn equality_is_reflexive_and_hashes_are_values() {
    let plan = LogicalPlan::table(["t"]).filter(Expr::column(["a"]));
    assert!(PartialEq::eq(&plan, &plan));
    let plan = Plan::try_new(plan).unwrap();
    let hash = plan.structural_hash();
    assert_eq!(format!("{:032x}", hash.as_u128()), hash.to_string());
    assert_eq!(plan.ir_version(), crate::IR_VERSION);
}
