// SPDX-License-Identifier: Apache-2.0
//! Tests of the normalized plan hash and the relation-cache key.

use super::*;
use std::collections::BTreeMap;

use krishiv_types::DataType;

use crate::{Builtin, FunctionId, FunctionKind, Literal, ResolvedFunction};

fn bound(id: u64, name: &str) -> BoundColumn {
    BoundColumn {
        id: AttrId(id),
        name: name.to_owned(),
        data_type: DataType::Integer,
        nullable: false,
    }
}

fn rcol(id: u64, name: &str) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(bound(
        id, name,
    )))))
}

fn scan(ids: &[u64], names: &[&str]) -> LogicalPlan {
    let output = ids
        .iter()
        .zip(names.iter())
        .map(|(id, name)| bound(*id, name))
        .collect();
    LogicalPlan::new(PlanNode::Scan(crate::Scan {
        source: ReadSource::Table {
            name: vec!["t".to_owned()],
            options: BTreeMap::new(),
        },
        output,
        version: None,
    }))
}

fn project(input: LogicalPlan, cols: &[Arc<Expr>]) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Project {
        input: Arc::new(input),
        exprs: cols.to_vec(),
    })
}

fn filter(input: LogicalPlan, predicate: Expr) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Filter {
        input: Arc::new(input),
        predicate: Arc::new(predicate),
    })
}

fn gt(left: Arc<Expr>, value: i32) -> Expr {
    Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            within_group: Vec::new(),
            id: FunctionId::Builtin(Builtin::Gt),
            kind: FunctionKind::Scalar,
            args: vec![left, Arc::new(Expr::literal(Literal::Integer(value)))],
            distinct: false,
            filter: None,
            data_type: DataType::Boolean,
            nullable: false,
            deterministic: true,
        },
    )))
}

fn hash(plan: &LogicalPlan) -> u128 {
    let plan = Plan::try_new(plan.clone()).unwrap();
    normalized_plan_hash(&plan)
}

#[test]
fn renumbered_attribute_ids_hash_equal() {
    let a = project(scan(&[1, 2], &["a", "b"]), &[rcol(1, "a"), rcol(2, "b")]);
    let b = project(
        scan(&[41, 42], &["a", "b"]),
        &[rcol(41, "a"), rcol(42, "b")],
    );
    assert_eq!(hash(&a), hash(&b));
}

/// Calls that differ only in their `WITHIN GROUP` ordering hash apart.
#[test]
fn within_group_orderings_enter_the_hash() {
    use crate::{NullOrdering, SortDirection, SortOrder};
    let call = |col: u64, dir: SortDirection, nulls: NullOrdering, ordered: bool| {
        let within_group = if ordered {
            vec![SortOrder {
                expr: rcol(col, "a"),
                direction: dir,
                nulls,
            }]
        } else {
            Vec::new()
        };
        let f = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                within_group,
                id: FunctionId::Builtin(Builtin::Gt),
                kind: FunctionKind::Aggregate,
                args: vec![rcol(1, "a")],
                distinct: false,
                filter: None,
                data_type: DataType::Boolean,
                nullable: true,
                deterministic: true,
            },
        ))));
        project(scan(&[1, 2], &["a", "b"]), &[f])
    };
    use NullOrdering::{NullsFirst, NullsLast};
    use SortDirection::{Ascending, Descending};
    let all = [
        call(1, Ascending, NullsFirst, true),
        call(1, Descending, NullsFirst, true),
        call(1, Ascending, NullsLast, true),
        call(2, Ascending, NullsFirst, true),
        call(1, Ascending, NullsFirst, false),
    ];
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            assert_ne!(hash(a), hash(b));
        }
    }
    assert_eq!(hash(&all[0]), hash(&call(1, Ascending, NullsFirst, true)));
}

#[test]
fn different_literals_hash_differently() {
    let a = filter(scan(&[1], &["a"]), gt(rcol(1, "a"), 1));
    let b = filter(scan(&[1], &["a"]), gt(rcol(1, "a"), 2));
    assert_ne!(hash(&a), hash(&b));
}

#[test]
fn different_columns_hash_differently() {
    let a = project(scan(&[1, 2], &["a", "b"]), &[rcol(1, "a")]);
    let b = project(scan(&[1, 2], &["a", "b"]), &[rcol(2, "b")]);
    assert_ne!(hash(&a), hash(&b));
    // ... as do different names on the same ids.
    let c = project(scan(&[1], &["a"]), &[rcol(1, "a")]);
    let d = project(scan(&[1], &["z"]), &[rcol(1, "z")]);
    assert_ne!(hash(&c), hash(&d));
}

fn semantic(plan: &LogicalPlan) -> u128 {
    semantic_plan_hash(&Plan::try_new(plan.clone()).unwrap())
}

fn alias(e: Arc<Expr>, id: u64, name: &str) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
        expr: e,
        output: bound(id, name),
    })))
}

#[test]
fn the_semantic_hash_ignores_names_and_subquery_aliases() {
    // Column and alias names (Spark: `AttributeReference` canonicalized
    // as "none", `Alias` names erased).
    let a = project(scan(&[1], &["a"]), &[alias(rcol(1, "a"), 2, "x")]);
    let b = project(scan(&[7], &["z"]), &[alias(rcol(7, "z"), 8, "y")]);
    assert_eq!(semantic(&a), semantic(&b));
    assert_ne!(hash(&a), hash(&b), "the cache key keeps the names");
    // A `SubqueryAlias` (Spark: `doCanonicalize` is the child's).
    let aliased = LogicalPlan::new(PlanNode::SubqueryAlias {
        input: Arc::new(scan(&[1], &["a"])),
        alias: "t".to_owned(),
        columns: vec!["c".to_owned()],
    });
    assert_eq!(semantic(&aliased), semantic(&scan(&[3], &["b"])));
    // Other columns, literals and shapes still differ.
    let c = project(scan(&[1, 2], &["a", "b"]), &[rcol(1, "a")]);
    let d = project(scan(&[1, 2], &["a", "b"]), &[rcol(2, "a")]);
    assert_ne!(semantic(&c), semantic(&d));
    let e = filter(scan(&[1], &["a"]), gt(rcol(1, "a"), 1));
    let f = filter(scan(&[1], &["a"]), gt(rcol(1, "a"), 2));
    assert_ne!(semantic(&e), semantic(&f));
}

#[test]
fn range_and_table_function_hash_every_field() {
    let range = |start: i64, end: i64, step: i64, slices: Option<u32>, id: u64| {
        LogicalPlan::new(PlanNode::Range {
            start,
            end,
            step,
            num_slices: slices,
            output: bound(id, "id"),
        })
    };
    let base = range(0, 10, 1, None, 1);
    assert_eq!(hash(&base), hash(&range(0, 10, 1, None, 1)));
    // Renumbered attribute ids normalise away.
    assert_eq!(hash(&base), hash(&range(0, 10, 1, None, 77)));
    for other in [
        range(1, 10, 1, None, 1),
        range(0, 11, 1, None, 1),
        range(0, 10, 2, None, 1),
        range(0, 10, 1, Some(4), 1),
    ] {
        assert_ne!(hash(&base), hash(&other));
    }
    assert_ne!(
        hash(&range(0, 10, 1, Some(4), 1)),
        hash(&range(0, 10, 1, Some(5), 1))
    );
    let call = |name: &str, args: Vec<Arc<Expr>>| {
        LogicalPlan::new(PlanNode::TableFunction {
            name: name.to_owned(),
            args,
        })
    };
    let one = Arc::new(Expr::literal(Literal::Integer(1)));
    assert_ne!(
        hash(&call("range", vec![])),
        hash(&call("sql_keywords", vec![]))
    );
    assert_ne!(
        hash(&call("range", vec![])),
        hash(&call("range", vec![one]))
    );
}

/// One plan per node form and one expression per expression form: each
/// hashes the same on a second encoding and differently from every
/// other, so no form shares a tag or drops a field.
#[test]
fn every_node_and_expression_form_hashes_distinctly() {
    use crate::{
        ColumnRef, Cte, CteRef, FrameBound, FrameUnits, FunctionCall, JoinKind, JoinStrategy,
        NullOrdering, PivotValue, PlanId, SetOperation, SortDirection, SortOrder, Star,
        SubqueryKind, TypedLiteralKind, UnpivotColumnSet, WindowExpr, WindowFrame, WindowSpec,
        WindowSpecRef,
    };

    let base = || Arc::new(scan(&[1], &["a"]));
    let a = || rcol(1, "a");
    let lit = |v: i32| Arc::new(Expr::literal(Literal::Integer(v)));
    let unresolved = |node: ExprNode| Arc::new(Expr::new(node));
    let call = |name: &str| FunctionCall {
        within_group: Vec::new(),
        name: name.to_owned(),
        qualifier: Vec::new(),
        args: vec![lit(1)],
        arg_names: vec![Some("x".to_owned())],
        distinct: true,
        filter: Some(lit(2)),
    };
    let project = |e: Arc<Expr>| {
        LogicalPlan::new(PlanNode::Project {
            input: base(),
            exprs: vec![e],
        })
    };
    let node = |n: PlanNode| LogicalPlan::new(n);
    let order = |dir: SortDirection| SortOrder {
        expr: a(),
        direction: dir,
        nulls: NullOrdering::NullsLast,
    };
    let spec = WindowSpec {
        partition_by: vec![a()],
        order_by: vec![order(SortDirection::Descending)],
        frame: Some(WindowFrame {
            units: FrameUnits::Rows,
            lower: FrameBound::UnboundedPreceding,
            upper: FrameBound::Offset(lit(2)),
        }),
    };
    let sub = |kind: SubqueryKind| ExprNode::Subquery { kind, plan: base() };
    let resolved_fn = |filter: Option<Arc<Expr>>| {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                within_group: Vec::new(),
                id: FunctionId::Builtin(Builtin::Gt),
                kind: FunctionKind::Scalar,
                args: vec![a(), lit(1)],
                distinct: true,
                filter,
                data_type: DataType::Boolean,
                nullable: true,
                deterministic: true,
            },
        ))))
    };
    let resolved = |r: ResolvedExpr| Arc::new(Expr::new(ExprNode::Resolved(r)));
    let grouped = |g: Grouping| {
        node(PlanNode::Aggregate {
            input: base(),
            grouping: g,
            aggregates: vec![a()],
        })
    };
    let join = |kind: JoinKind, condition: JoinCondition| {
        node(PlanNode::Join {
            left: base(),
            right: Arc::new(scan(&[2], &["b"])),
            kind,
            condition,
        })
    };

    let exprs: Vec<(&str, Arc<Expr>)> = vec![
        (
            "column",
            unresolved(ExprNode::Column(ColumnRef {
                name: vec!["t".into(), "a".into()],
                plan_id: Some(PlanId(3)),
                by_identifier: false,
            })),
        ),
        (
            "column-bare",
            unresolved(ExprNode::Column(ColumnRef::new(vec!["a".into()]))),
        ),
        ("literal", lit(7)),
        (
            "typed-literal",
            unresolved(ExprNode::TypedLiteral {
                kind: TypedLiteralKind::Date,
                value: "2024-01-02".into(),
            }),
        ),
        (
            "star",
            unresolved(ExprNode::Star(Star {
                regex: None,
                replace: Vec::new(),
                position: None,
                qualifier: vec!["t".into()],
                except: vec![vec!["a".into()]],
                plan_id: Some(PlanId(1)),
            })),
        ),
        ("function", unresolved(ExprNode::Function(call("f")))),
        ("function-other", unresolved(ExprNode::Function(call("g")))),
        (
            "lambda",
            unresolved(ExprNode::Lambda {
                params: vec![bound(9, "x")],
                body: lit(1),
            }),
        ),
        (
            "field",
            unresolved(ExprNode::Field {
                base: a(),
                name: "f".into(),
            }),
        ),
        (
            "cast",
            unresolved(ExprNode::Cast {
                expr: a(),
                to: DataType::Long,
                mode: crate::CastMode::Ansi,
            }),
        ),
        (
            "alias",
            unresolved(ExprNode::Alias {
                expr: a(),
                name: "z".into(),
            }),
        ),
        ("scalar-subquery", unresolved(sub(SubqueryKind::Scalar))),
        ("exists-subquery", unresolved(sub(SubqueryKind::Exists))),
        (
            "in-subquery",
            unresolved(sub(SubqueryKind::In(vec![lit(1)]))),
        ),
        (
            "window-spec",
            unresolved(ExprNode::Window(WindowExpr {
                function: unresolved(ExprNode::Function(call("sum"))),
                window: WindowSpecRef::Spec(spec),
            })),
        ),
        (
            "window-named",
            unresolved(ExprNode::Window(WindowExpr {
                function: unresolved(ExprNode::Function(call("sum"))),
                window: WindowSpecRef::Named("w".into()),
            })),
        ),
        ("r-column", a()),
        (
            "r-outer",
            resolved(ResolvedExpr::OuterColumn(bound(1, "a"))),
        ),
        (
            "r-lambda-var",
            resolved(ResolvedExpr::LambdaVar(bound(1, "a"))),
        ),
        ("r-function", resolved_fn(Some(lit(3)))),
        ("r-function-unfiltered", resolved_fn(None)),
        (
            "r-field",
            resolved(ResolvedExpr::Field {
                base: a(),
                ordinal: 2,
                name: "f".into(),
                data_type: DataType::Integer,
                nullable: true,
            }),
        ),
        (
            "r-alias",
            resolved(ResolvedExpr::Alias {
                expr: a(),
                output: bound(5, "al"),
            }),
        ),
    ];
    let mut plans: Vec<(String, LogicalPlan)> = exprs
        .into_iter()
        .map(|(name, e)| (format!("expr {name}"), project(e)))
        .collect();
    plans.extend(
        [
            ("one-row", node(PlanNode::OneRowRelation)),
            (
                "table-function",
                node(PlanNode::TableFunction {
                    name: "range".into(),
                    args: vec![lit(3)],
                }),
            ),
            (
                "range",
                node(PlanNode::Range {
                    start: 0,
                    end: 5,
                    step: 1,
                    num_slices: Some(2),
                    output: bound(4, "id"),
                }),
            ),
            (
                "with-columns",
                node(PlanNode::WithColumns {
                    input: base(),
                    columns: vec![("c".into(), a())],
                }),
            ),
            (
                "filter",
                node(PlanNode::Filter {
                    input: base(),
                    predicate: lit(1),
                }),
            ),
            ("group-by", grouped(Grouping::GroupBy(vec![a()]))),
            ("rollup", grouped(Grouping::Rollup(vec![a()]))),
            ("cube", grouped(Grouping::Cube(vec![a()]))),
            (
                "grouping-sets",
                grouped(Grouping::GroupingSets {
                    exprs: vec![a()],
                    sets: vec![vec![a()], vec![]],
                }),
            ),
            ("join-none", join(JoinKind::Cross, JoinCondition::None)),
            (
                "join-existence",
                join(JoinKind::Existence(AttrId(1)), JoinCondition::On(lit(1))),
            ),
            ("join-on", join(JoinKind::Inner, JoinCondition::On(lit(1)))),
            (
                "join-using",
                join(JoinKind::LeftOuter, JoinCondition::Using(vec!["a".into()])),
            ),
            (
                "join-natural",
                join(JoinKind::FullOuter, JoinCondition::Natural),
            ),
            (
                "sort",
                node(PlanNode::Sort {
                    input: base(),
                    order: vec![order(SortDirection::Ascending)],
                    global: true,
                }),
            ),
            (
                "limit",
                node(PlanNode::Limit {
                    input: base(),
                    limit: lit(5),
                }),
            ),
            (
                "offset",
                node(PlanNode::Offset {
                    input: base(),
                    offset: lit(5),
                }),
            ),
            (
                "subquery-alias",
                node(PlanNode::SubqueryAlias {
                    input: base(),
                    alias: "s".into(),
                    columns: vec!["c".into()],
                }),
            ),
            (
                "with-cte",
                node(PlanNode::WithCte {
                    input: base(),
                    ctes: vec![Cte {
                        name: "c".into(),
                        plan: base(),
                    }],
                }),
            ),
            (
                "cte-ref",
                node(PlanNode::CteRef(CteRef {
                    name: "c".into(),
                    source: vec![AttrId(1)],
                    output: vec![bound(2, "a")],
                })),
            ),
            (
                "shift",
                node(PlanNode::Shift {
                    input: base(),
                    offset: 10,
                }),
            ),
            (
                "hint",
                node(PlanNode::Hint {
                    input: base(),
                    name: "h".into(),
                    params: vec![lit(1)],
                }),
            ),
            (
                "pipe-operator",
                node(PlanNode::PipeOperator { input: base() }),
            ),
            (
                "repartition",
                node(PlanNode::Repartition {
                    input: base(),
                    exprs: vec![lit(1)],
                }),
            ),
            (
                "join-hint",
                node(PlanNode::JoinHint {
                    input: base(),
                    strategy: JoinStrategy::Broadcast,
                }),
            ),
            (
                "union",
                node(PlanNode::Union {
                    inputs: vec![base(), base()],
                    by_name: true,
                    allow_missing_columns: true,
                }),
            ),
            ("distinct", node(PlanNode::Distinct { input: base() })),
            (
                "set-op",
                node(PlanNode::SetOp {
                    left: base(),
                    right: Arc::new(scan(&[2], &["b"])),
                    op: SetOperation::Except,
                    all: true,
                }),
            ),
            (
                "window",
                node(PlanNode::Window {
                    input: base(),
                    exprs: vec![a()],
                    partition_by: vec![a()],
                    order_by: vec![order(SortDirection::Ascending)],
                }),
            ),
            (
                "replicate-rows",
                node(PlanNode::ReplicateRows {
                    input: base(),
                    count: lit(2),
                }),
            ),
            (
                "expand",
                node(PlanNode::Expand {
                    input: base(),
                    projections: vec![vec![a()], vec![lit(1)]],
                    output: vec![bound(6, "e")],
                }),
            ),
            (
                "pivot",
                node(PlanNode::Pivot {
                    input: base(),
                    pivot_columns: vec![a()],
                    values: vec![
                        PivotValue {
                            exprs: vec![lit(1)],
                            alias: Some("one".into()),
                        },
                        PivotValue {
                            exprs: vec![lit(2)],
                            alias: None,
                        },
                    ],
                    aggregates: vec![a()],
                }),
            ),
            (
                "unpivot",
                node(PlanNode::Unpivot {
                    input: base(),
                    value_columns: vec!["v".into()],
                    name_column: "k".into(),
                    column_sets: vec![
                        UnpivotColumnSet {
                            columns: vec![a()],
                            alias: Some("x".into()),
                        },
                        UnpivotColumnSet {
                            columns: vec![a()],
                            alias: None,
                        },
                    ],
                    include_nulls: true,
                }),
            ),
            (
                "inline-table",
                node(PlanNode::InlineTable {
                    rows: vec![vec![lit(1)]],
                    columns: vec!["c".into()],
                }),
            ),
            ("scan-versioned", {
                let mut s = scan(&[1], &["a"]);
                if let PlanNode::Scan(sc) = &mut s.node {
                    sc.version = Some(crate::TableVersion(4));
                }
                s
            }),
            (
                "sql",
                node(PlanNode::Sql {
                    query: "SELECT 1".into(),
                    args: SqlArgs::Positional(vec![lit(1)]),
                }),
            ),
            (
                "sql-named",
                node(PlanNode::Sql {
                    query: "SELECT 1".into(),
                    args: SqlArgs::Named(BTreeMap::from([("k".to_owned(), lit(1))])),
                }),
            ),
            (
                "sql-no-args",
                node(PlanNode::Sql {
                    query: "SELECT 1".into(),
                    args: SqlArgs::None,
                }),
            ),
            (
                "read-files",
                node(PlanNode::Read(ReadSource::Files {
                    format: "csv".into(),
                    paths: vec![
                        OptionValue::from_user("path", "/p"),
                        OptionValue::from_user("path", "s3://user:pw@host/k"),
                    ],
                    options: BTreeMap::from([
                        ("secret".to_owned(), OptionValue::reference("s", "k")),
                        (
                            "password".to_owned(),
                            OptionValue::from_user("password", "pw"),
                        ),
                    ]),
                    schema: None,
                })),
            ),
            (
                "map-partitions",
                node(PlanNode::MapPartitions {
                    input: base(),
                    udf: py(crate::PythonEvalType::MapArrowIter),
                    output: vec![bound(2, "x")],
                    barrier: true,
                }),
            ),
            (
                "group-map",
                node(PlanNode::GroupMap {
                    input: base(),
                    keys: vec![a()],
                    udf: py(crate::PythonEvalType::GroupedMapArrow),
                    output: vec![bound(2, "x")],
                }),
            ),
            (
                "cogroup-map",
                node(PlanNode::CoGroupMap {
                    left: base(),
                    right: base(),
                    left_keys: vec![a()],
                    right_keys: vec![a()],
                    udf: py(crate::PythonEvalType::CogroupedMapArrow),
                    output: vec![bound(2, "x")],
                }),
            ),
            (
                "python-table-function",
                node(PlanNode::PythonTableFunction {
                    args: vec![lit(1)],
                    table_arg: Some(crate::TableArg {
                        input: base(),
                        partition_by: vec![a()],
                        order_by: vec![crate::SortOrder::asc(a())],
                        single_partition: false,
                        select: Vec::new(),
                        position: 0,
                        name: None,
                    }),
                    udf: py(crate::PythonEvalType::ArrowUdtf),
                    output: vec![bound(2, "x")],
                    lateral: None,
                    analyzed: None,
                }),
            ),
            (
                "python-table-function-no-table",
                node(PlanNode::PythonTableFunction {
                    args: vec![lit(1)],
                    table_arg: None,
                    udf: py(crate::PythonEvalType::TableUdf),
                    output: vec![bound(2, "x")],
                    lateral: None,
                    analyzed: None,
                }),
            ),
            (
                "read-table-options",
                node(PlanNode::Read(ReadSource::Table {
                    name: vec!["t".into()],
                    options: BTreeMap::from([("k".to_owned(), OptionValue::from_user("k", "v"))]),
                })),
            ),
        ]
        .into_iter()
        .map(|(name, plan)| (name.to_owned(), plan)),
    );

    let mut seen: BTreeMap<u128, String> = BTreeMap::new();
    for (name, plan) in &plans {
        let h = normalized_subtree_hash(plan);
        assert_eq!(h, normalized_subtree_hash(plan), "{name} is stable");
        if let Some(other) = seen.insert(h, name.clone()) {
            panic!("{name} hashes like {other}");
        }
    }
}

/// A Python UDF reference of a relation node.
fn py(eval_type: crate::PythonEvalType) -> crate::PythonUdfRef {
    crate::PythonUdfRef {
        name: "f".into(),
        version: 1,
        eval_type,
        thread_safe: true,
        deterministic: true,
    }
}

#[test]
fn subtree_hash_is_stable_and_differs_from_whole() {
    let sub = filter(scan(&[3], &["a"]), gt(rcol(3, "a"), 1));
    let same = filter(scan(&[90], &["a"]), gt(rcol(90, "a"), 1));
    assert_eq!(
        normalized_subtree_hash(&sub),
        normalized_subtree_hash(&same)
    );
    let whole = project(sub.clone(), &[rcol(3, "a")]);
    assert_ne!(normalized_subtree_hash(&sub), hash(&whole));
}

#[test]
fn the_unversioned_hash_leaves_out_scan_versions_only() {
    let versioned = |v: Option<i64>| {
        let mut s = scan(&[1], &["a"]);
        if let PlanNode::Scan(sc) = &mut s.node {
            sc.version = v.map(crate::TableVersion);
        }
        filter(s, gt(rcol(1, "a"), 1))
    };
    let base = unversioned_subtree_hash(&versioned(Some(3)));
    assert_eq!(unversioned_subtree_hash(&versioned(Some(4))), base);
    assert_eq!(unversioned_subtree_hash(&versioned(None)), base);
    assert_ne!(
        normalized_subtree_hash(&versioned(Some(3))),
        normalized_subtree_hash(&versioned(Some(4)))
    );
    let mut other = scan(&[1], &["a"]);
    if let PlanNode::Scan(sc) = &mut other.node {
        sc.version = Some(crate::TableVersion(3));
    }
    assert_ne!(
        unversioned_subtree_hash(&filter(other, gt(rcol(1, "a"), 2))),
        base,
        "everything else still enters"
    );
}

#[test]
fn cache_keys_separate_principals_policies_and_pins() {
    let plan = Plan::try_new(project(scan(&[1], &["a"]), &[rcol(1, "a")])).unwrap();
    let context = || crate::analyzer::ContextFingerprint {
        profile: crate::analyzer::SparkProfile::default(),
        conf: crate::analyzer::ResolutionConf::default(),
        current: vec!["spark_catalog".to_owned(), "default".to_owned()],
        catalogs: 0,
        catalog_epochs: BTreeMap::new(),
        functions: 0,
        table_functions: 0,
        sql: None,
        principal: "a".to_owned(),
        policy: PolicyVersion(7),
        files: false,
        variables: 0,
        mode: crate::analyzer::AnalysisMode::Execute,
    };
    let analyzed = |principal: &str, policy: u64| crate::analyzer::AnalyzedPlan {
        plan: plan.clone(),
        output: vec![bound(1, "a")],
        output_declared: Vec::new(),
        output_comments: Vec::new(),
        pins: BTreeMap::new(),
        securables: Vec::new(),
        input_key: plan.structural_hash(),
        context: crate::analyzer::ContextFingerprint {
            principal: principal.to_owned(),
            policy: PolicyVersion(policy),
            ..context()
        },
        query_text: None,
        reads_files: false,
        timings: crate::analyzer::AnalysisTimings::default(),
        warnings: Vec::new(),
        views: Vec::new(),
    };
    let base = CacheKey::of(&analyzed("a", 7));
    assert_ne!(CacheKey::of(&analyzed("b", 7)), base, "principals");
    assert_ne!(CacheKey::of(&analyzed("a", 8)), base, "policy version");
    let mut pinned = analyzed("a", 7);
    pinned.pins.insert(
        TableIdent {
            catalog: "spark_catalog".to_owned(),
            namespace: vec!["default".to_owned()],
            name: "t".to_owned(),
        },
        TableVersion(3),
    );
    assert_ne!(CacheKey::of(&pinned), base, "pins");
    assert_eq!(CacheKey::of(&analyzed("a", 7)), base, "same inputs");
}

/// Every `ColumnNamesOp` is part of the plan's normalized hash (a change of any field changes
/// it) and prints as one line naming its fields.
#[test]
fn column_names_operations_hash_apart_and_display_their_fields() {
    use crate::ColumnNamesOp as Op;
    let names = |n: &[&str]| n.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    let lit = |v: i32| Arc::new(Expr::literal(Literal::Integer(v)));
    let ops = [
        (Op::ToDf(names(&["x", "y"])), "ToDF x, y"),
        (Op::ToDf(names(&["x", "z"])), "ToDF x, z"),
        (
            Op::Rename(vec![("a".to_owned(), "b".to_owned())]),
            "RenameColumns a -> b",
        ),
        (
            Op::Drop {
                names: names(&["a"]),
                columns: names(&["b"]),
            },
            "DropColumns a (resolved: b)",
        ),
        (
            Op::Drop {
                names: names(&["a"]),
                columns: Vec::new(),
            },
            "DropColumns a",
        ),
        (
            Op::DropNa {
                subset: names(&["a"]),
                min_non_nulls: Some(2),
            },
            "DropNa min 2 a",
        ),
        (
            Op::DropNa {
                subset: Vec::new(),
                min_non_nulls: None,
            },
            "DropNa",
        ),
        (
            Op::Replace {
                cols: names(&["a"]),
                pairs: vec![lit(1), lit(2)],
            },
            "Replace (1, 2) a",
        ),
        (
            Op::FillNa {
                cols: names(&["a"]),
                values: vec![lit(7)],
            },
            "FillNa (7) a",
        ),
        (
            Op::FillNa {
                cols: Vec::new(),
                values: vec![lit(8)],
            },
            "FillNa (8)",
        ),
    ];
    let mut hashes = Vec::new();
    for (op, want) in ops {
        let plan = LogicalPlan::new(PlanNode::ColumnNames {
            input: Arc::new(scan(&[1, 2], &["a", "b"])),
            op,
        });
        let text = plan.to_string();
        assert_eq!(text.lines().next(), Some(want), "{text}");
        hashes.push(normalized_subtree_hash(&plan));
    }
    let mut distinct = hashes.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), hashes.len(), "{hashes:?}");
}
