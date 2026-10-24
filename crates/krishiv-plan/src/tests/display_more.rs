// SPDX-License-Identifier: Apache-2.0
//! Pretty-printer: every node and expression form, escaping, and the
//! depth and length caps.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_types::DataType;

use crate::{
    AttrId, BoundColumn, Builtin, CastMode, Cte, DayTimeField, Expr, ExprNode, FunctionCall,
    FunctionId, FunctionKind, Grouping, JoinCondition, JoinKind, Literal, LogicalPlan, MAX_DEPTH,
    PlanId, PlanNode, ResolvedExpr, ResolvedFunction, Scan, SqlArgs, Star, SubqueryKind,
};

fn col(name: &str) -> Arc<Expr> {
    Arc::new(Expr::column([name]))
}

fn t() -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::table(["t"]))
}

fn bound(name: &str, id: u64) -> BoundColumn {
    BoundColumn {
        id: AttrId(id),
        name: name.to_owned(),
        data_type: DataType::Integer,
        nullable: true,
    }
}

fn line(node: PlanNode) -> String {
    let text = LogicalPlan::new(node).to_string();
    text.lines().next().unwrap_or_default().to_owned()
}

#[test]
fn every_relation_prints_its_fields() {
    let cases = [
        (
            PlanNode::Sql {
                query: "select ?".to_owned(),
                args: SqlArgs::Positional(vec![col("a")]),
            },
            "Sql [8 bytes], [a]",
        ),
        (
            PlanNode::Sql {
                query: "select :x".to_owned(),
                args: SqlArgs::Named(BTreeMap::from([
                    ("x".to_owned(), col("a")),
                    ("y".to_owned(), col("b")),
                ])),
            },
            "Sql [9 bytes], {x = a, y = b}",
        ),
        (
            PlanNode::Scan(Scan {
                source: crate::ReadSource::Table {
                    name: vec!["db".to_owned(), "t".to_owned()],
                    options: BTreeMap::new(),
                },
                output: vec![bound("a", 1), bound("b", 2)],
                version: Some(crate::TableVersion(3)),
            }),
            "Scan db.t [a#1, b#2] @ version 3",
        ),
        (
            PlanNode::WithColumns {
                input: t(),
                columns: vec![("x".to_owned(), col("a")), ("y".to_owned(), col("b"))],
            },
            "WithColumns [x = a, y = b]",
        ),
        (
            PlanNode::Aggregate {
                input: t(),
                grouping: Grouping::Rollup(vec![col("a")]),
                aggregates: vec![col("b")],
            },
            "Aggregate ROLLUP [a], [b]",
        ),
        (
            PlanNode::Aggregate {
                input: t(),
                grouping: Grouping::Cube(vec![col("a"), col("b")]),
                aggregates: vec![],
            },
            "Aggregate CUBE [a, b], []",
        ),
        (
            PlanNode::Aggregate {
                input: t(),
                grouping: Grouping::GroupingSets {
                    exprs: vec![col("a"), col("b")],
                    sets: vec![vec![col("a")], vec![]],
                },
                aggregates: vec![col("a")],
            },
            "Aggregate [a, b] GROUPING SETS ((a), ()), [a]",
        ),
        (
            PlanNode::Join {
                left: t(),
                right: t(),
                kind: JoinKind::FullOuter,
                condition: JoinCondition::Using(vec!["a".to_owned(), "b".to_owned()]),
            },
            "Join FullOuter, USING (a, b)",
        ),
        (
            PlanNode::WithCte {
                input: t(),
                ctes: vec![Cte {
                    name: "c".to_owned(),
                    plan: t(),
                }],
            },
            "WithCte [c]",
        ),
        (
            PlanNode::Read(crate::ReadSource::Table {
                name: vec!["we`ird".to_owned(), "t".to_owned()],
                options: BTreeMap::new(),
            }),
            "Read `we``ird`.t",
        ),
    ];
    for (node, want) in cases {
        assert_eq!(line(node), want);
    }
}

fn resolved(id: FunctionId, args: Vec<Arc<Expr>>, filter: Option<Arc<Expr>>) -> Expr {
    Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id,
            kind: FunctionKind::Aggregate,
            args,
            distinct: filter.is_some(),
            filter,
            data_type: DataType::Long,
            nullable: false,
            deterministic: true,
        },
    )))
}

#[test]
fn every_expression_form_prints() {
    let cases: Vec<(Expr, &str)> = vec![
        (
            Expr::new(ExprNode::Star(Star {
                qualifier: vec!["t".to_owned()],
                plan_id: Some(PlanId(4)),
            })),
            "t.*@plan4",
        ),
        (
            Expr::new(ExprNode::Resolved(ResolvedExpr::Column(bound("a", 1)))),
            "a#1",
        ),
        (
            Expr::new(ExprNode::Resolved(ResolvedExpr::OuterColumn(bound("a", 1)))),
            "outer(a#1)",
        ),
        (
            Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                expr: col("a"),
                output: bound("x", 9),
            })),
            "a AS x#9",
        ),
        (
            resolved(
                FunctionId::Builtin(Builtin::Add),
                vec![col("a"), col("b")],
                None,
            ),
            "(a + b)",
        ),
        (
            resolved(
                FunctionId::Registry {
                    name: "count".to_owned(),
                    id: 3,
                },
                vec![col("a")],
                Some(col("p")),
            ),
            "count(DISTINCT a) FILTER (WHERE p)",
        ),
        (
            Expr::builtin(Builtin::Negative, vec![Expr::column(["a"])]),
            "(- a)",
        ),
        (
            Expr::new(ExprNode::Function(FunctionCall {
                name: "sum".to_owned(),
                qualifier: Vec::new(),
                arg_names: Vec::new(),
                args: vec![col("a")],
                distinct: false,
                filter: Some(col("p")),
            })),
            "sum(a) FILTER (WHERE p)",
        ),
        (
            Expr::new(ExprNode::Cast {
                expr: col("a"),
                to: DataType::Long,
                mode: CastMode::Try,
            }),
            "TRY_CAST(a AS BIGINT)",
        ),
        (Literal::Null(None).into(), "NULL"),
        (
            Literal::Null(Some(DataType::Integer)).into(),
            "CAST(NULL AS INT)",
        ),
        (Literal::Boolean(true).into(), "true"),
        (Literal::Byte(-1).into(), "-1Y"),
        (Literal::Short(2).into(), "2S"),
        (Literal::Long(3).into(), "3L"),
        (Literal::Timestamp(4).into(), "TIMESTAMP_MICROS(4)"),
        (Literal::TimestampNtz(5).into(), "TIMESTAMP_NTZ_MICROS(5)"),
        (
            Literal::Double(f64::INFINITY).into(),
            "CAST('Infinity' AS DOUBLE)",
        ),
        (Literal::String("\\ \r \t".to_owned()).into(), r"'\\ \r \t'"),
        (
            Literal::DayTimeInterval {
                micros: 61_000_000,
                start: DayTimeField::Minute,
                end: DayTimeField::Second,
            }
            .into(),
            "INTERVAL '1:01' MINUTE TO SECOND",
        ),
        (
            Literal::Decimal {
                unscaled: 12,
                data_type: krishiv_types::DecimalType::try_new(5, 0).unwrap(),
            }
            .into(),
            "12",
        ),
        (
            Expr::subquery(SubqueryKind::Scalar, LogicalPlan::table(["u"])),
            "scalar-subquery",
        ),
    ];
    for (e, want) in cases {
        assert_eq!(e.to_string(), want);
    }
}

/// A subquery used twice in one operator prints once under it.
#[test]
fn shared_subqueries_print_once() {
    let sub = Arc::new(Expr::subquery(
        SubqueryKind::Exists,
        LogicalPlan::table(["u"]),
    ));
    let both = Expr::new(ExprNode::Function(FunctionCall {
        name: Builtin::And.name().to_owned(),
        qualifier: Vec::new(),
        arg_names: Vec::new(),
        args: vec![Arc::clone(&sub), sub],
        distinct: false,
        filter: None,
    }));
    let plan = LogicalPlan::table(["t"]).filter(both);
    assert_eq!(
        plan.to_string(),
        "Filter (exists and exists)\n:  +- Read u\n+- Read t\n"
    );
}

#[test]
fn too_deep_trees_print_a_marker() {
    let deep = (0..=MAX_DEPTH).fold(LogicalPlan::table(["t"]), |p, _| {
        p.filter(Expr::column(["a"]))
    });
    assert_eq!(deep.to_string(), "<nested deeper than the maximum depth>\n");
    let e = (0..=MAX_DEPTH).fold(Expr::column(["a"]), |e, _| {
        Expr::builtin(Builtin::Not, vec![e])
    });
    assert_eq!(e.to_string(), "<nested deeper than the maximum depth>");
}

/// Text is cut on a character boundary.
#[test]
fn cuts_respect_character_boundaries() {
    let wide = "é".repeat(40 * 1024);
    let e = Expr::builtin(
        Builtin::Concat,
        vec![
            Literal::String(wide.clone()).into(),
            Literal::String(wide).into(),
        ],
    );
    let shown = e.to_string();
    assert!(shown.ends_with("...(truncated)"));
    let huge: Expr = Literal::String("é".repeat(40 * 1024)).into();
    let shown = huge.to_string();
    assert!(shown.ends_with("...(truncated)"), "{}", shown.len());
}
