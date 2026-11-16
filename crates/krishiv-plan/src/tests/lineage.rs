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

/// A Python UDF of `eval_type` (P3-T085 relations).
fn python_udf(eval_type: crate::PythonEvalType) -> crate::PythonUdfRef {
    crate::PythonUdfRef {
        name: "f".to_owned(),
        version: 1,
        eval_type,
        thread_safe: true,
        deterministic: true,
    }
}

#[test]
fn python_udf_relations_are_opaque_over_every_input_column() {
    use crate::analyzer::lineage::{FilterKind, Leaf, LineageWarning};
    use crate::{BoundColumn, GeneratorKind, PythonEvalType, SortOrder, TableArg};

    let a = analyze(LogicalPlan::table(["t"]));
    // `t` and `u` from one analysis, so their ids are distinct: the cogroup's sides.
    let tu = analyze(LogicalPlan::table(["t"]).join(
        LogicalPlan::table(["u"]),
        crate::JoinKind::Cross,
        crate::JoinCondition::None,
    ));
    let PlanNode::Join {
        left: t_side,
        right: u_side,
        ..
    } = &tu.plan.root().node
    else {
        panic!("{:?}", tu.plan)
    };
    let leaf = |table: &str, column: &str| Leaf::Table {
        name: vec![
            "spark_catalog".to_owned(),
            "default".to_owned(),
            table.to_owned(),
        ],
        column: column.to_owned(),
    };
    let column = |b: &BoundColumn| {
        Arc::new(Expr::new(ExprNode::Resolved(crate::ResolvedExpr::Column(
            b.clone(),
        ))))
    };
    let x = column(&a.output[0]);
    let y = column(&a.output[1]);
    let t_x = column(&tu.output[0]);
    let u_z = column(&tu.output[2]);
    let out = |id: u64| BoundColumn {
        id: crate::AttrId(id),
        name: format!("o{id}"),
        data_type: DataType::Long,
        nullable: true,
    };
    let outputs = vec![out(100), out(101)];
    type Build<'a> = Box<dyn Fn(Arc<LogicalPlan>) -> LogicalPlan + 'a>;
    /// What a case is called, its node over `t`, its grouping keys, its window keys, its source count.
    type Case<'a> = (&'a str, Build<'a>, Vec<Leaf>, Vec<Leaf>, usize);
    let nodes: Vec<Case<'_>> = vec![
        (
            "map",
            Box::new(|input| {
                LogicalPlan::new(PlanNode::MapPartitions {
                    input,
                    udf: python_udf(PythonEvalType::MapArrowIter),
                    output: outputs.clone(),
                    barrier: false,
                })
            }),
            vec![],
            vec![],
            2,
        ),
        (
            "group map",
            Box::new(|input| {
                LogicalPlan::new(PlanNode::GroupMap {
                    input,
                    keys: vec![Arc::clone(&x)],
                    udf: python_udf(PythonEvalType::GroupedMapArrow),
                    output: outputs.clone(),
                })
            }),
            vec![leaf("t", "x")],
            vec![],
            2,
        ),
        (
            "cogroup map",
            // `t` cogrouped with `u`: both sides are sources, and both sides' keys.
            Box::new(|_| {
                LogicalPlan::new(PlanNode::CoGroupMap {
                    left: Arc::clone(t_side),
                    right: Arc::clone(u_side),
                    left_keys: vec![Arc::clone(&t_x)],
                    right_keys: vec![Arc::clone(&u_z)],
                    udf: python_udf(PythonEvalType::CogroupedMapArrow),
                    output: outputs.clone(),
                })
            }),
            vec![leaf("t", "x"), leaf("u", "z")],
            vec![],
            3,
        ),
        (
            "table function",
            Box::new(|input| {
                LogicalPlan::new(PlanNode::PythonTableFunction {
                    args: vec![Arc::new(Expr::from(crate::Literal::Integer(1)))],
                    table_arg: Some(TableArg {
                        input,
                        partition_by: vec![Arc::clone(&x)],
                        order_by: vec![SortOrder::asc(Arc::clone(&y))],
                        single_partition: false,
                        select: Vec::new(),
                        position: 0,
                        name: None,
                    }),
                    udf: python_udf(PythonEvalType::ArrowUdtf),
                    output: outputs.clone(),
                    lateral: None,
                    analyzed: None,
                })
            }),
            vec![],
            vec![leaf("t", "x"), leaf("t", "y")],
            2,
        ),
    ];
    for (what, build, group_by, window, sources) in &nodes {
        let p = AnalyzedPlan {
            output: outputs.clone(),
            ..with_root(&a, build)
        };
        let l = lineage(&p).unwrap();
        assert_eq!(l.columns.len(), 2, "{what}");
        for col in &l.columns {
            assert_eq!(col.transform, Transform::Opaque, "{what}");
            assert_eq!(col.sources.len(), *sources, "{what}: every input column");
        }
        assert!(
            l.warnings.contains(&LineageWarning::LineageOpaqueUdf),
            "{what}"
        );
        let of = |kind: FilterKind| -> Vec<Leaf> {
            l.filters
                .iter()
                .filter(|(k, _)| *k == kind)
                .map(|(_, leaf)| leaf.clone())
                .collect()
        };
        assert_eq!(&of(FilterKind::GroupBy), group_by, "{what}");
        assert_eq!(&of(FilterKind::Window), window, "{what}");
    }

    drop(nodes);

    // A Python relation under a `Shift` (a self-join of a shared subtree): the instance's outputs read
    // `id + offset` (ids 1100 and 1101 resolve). Its input and keys reach the same leaves, so only the outputs are pinned.
    let group_map = Arc::new(LogicalPlan::new(PlanNode::GroupMap {
        input: Arc::clone(a.plan.root()),
        keys: vec![Arc::clone(&x)],
        udf: python_udf(PythonEvalType::GroupedMapArrow),
        output: outputs.clone(),
    }));
    let self_join = with_root(&a, |_| {
        LogicalPlan::new(PlanNode::Join {
            left: Arc::clone(&group_map),
            right: Arc::new(LogicalPlan::new(PlanNode::Shift {
                input: Arc::clone(&group_map),
                offset: 1000,
            })),
            kind: crate::JoinKind::Cross,
            condition: crate::JoinCondition::None,
        })
    });
    let p = AnalyzedPlan {
        output: [outputs, vec![out(1100), out(1101)]].concat(),
        ..self_join
    };
    let l = lineage(&p).unwrap();
    assert_eq!(l.columns.len(), 4);
    for col in &l.columns {
        assert_eq!(col.transform, Transform::Opaque, "{}", col.name);
        assert_eq!(col.sources.len(), 2, "{}: both of t's columns", col.name);
    }
    assert_eq!(l.columns[2].sources, l.columns[0].sources);
    let group_by: Vec<_> = l
        .filters
        .iter()
        .filter(|(k, _)| *k == FilterKind::GroupBy)
        .collect();
    assert_eq!(
        group_by.len(),
        1,
        "both instances group by t.x, deduplicated"
    );

    // A Python UDTF under LATERAL: the call over `x` is the generator's argument.
    let call = Arc::new(Expr::new(ExprNode::Resolved(
        crate::ResolvedExpr::Function(crate::ResolvedFunction {
            id: python_udf(PythonEvalType::ArrowUdtf).function_id(),
            kind: crate::FunctionKind::Generator,
            args: vec![Arc::clone(&x)],
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: DataType::Long,
            nullable: true,
            deterministic: true,
        }),
    )));
    let udtf = with_root(&a, |input| {
        LogicalPlan::new(PlanNode::Generate {
            input,
            kind: GeneratorKind::PythonUdtf,
            outer: false,
            arg: call,
            output: vec![out(102)],
            names: vec!["o102".to_owned()],
            qualifier: None,
        })
    });
    let p = AnalyzedPlan {
        output: [a.output, vec![out(102)]].concat(),
        ..udtf
    };
    let l = lineage(&p).unwrap();
    let generated = &l.columns[2];
    assert_eq!(generated.transform, Transform::Opaque);
    assert_eq!(generated.sources.len(), 1, "the UDTF's argument x");
    assert!(l.warnings.contains(&LineageWarning::LineageOpaqueUdf));
}
