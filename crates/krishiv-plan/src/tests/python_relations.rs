// SPDX-License-Identifier: Apache-2.0
//! The Python UDF relation nodes and eval types (design P3-T085 §3.1):
//! codes, kinds, traversal, outputs, encoding, printing and the mixed
//! aggregate check.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::error::ErrorClass;
use krishiv_types::DataType;

use crate::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, GeneratorKind,
    LogicalPlan, Plan, PlanNode, PythonEvalType, PythonUdfKind, PythonUdfRef, ReadSource,
    ResolvedExpr, ResolvedFunction, Scan, SortOrder, TableArg, check_python_aggregates,
};

fn col(id: u64, name: &str) -> BoundColumn {
    BoundColumn {
        id: AttrId(id),
        name: name.to_owned(),
        data_type: DataType::Long,
        nullable: true,
    }
}

fn column(id: u64) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(col(
        id, "k",
    )))))
}

fn scan(first: u64) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Scan(Scan {
        source: ReadSource::Table {
            name: vec!["t".to_owned()],
            options: BTreeMap::new(),
        },
        output: vec![col(first, "k"), col(first + 1, "v")],
        version: None,
    })))
}

fn udf(eval_type: PythonEvalType) -> PythonUdfRef {
    PythonUdfRef {
        name: "f".to_owned(),
        version: 1,
        eval_type,
        thread_safe: true,
    }
}

fn call(id: FunctionId, kind: FunctionKind, args: Vec<Arc<Expr>>) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id,
            kind,
            args,
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: DataType::Double,
            nullable: true,
            deterministic: true,
        },
    ))))
}

fn nodes() -> Vec<LogicalPlan> {
    let out = vec![col(10, "x"), col(11, "y")];
    vec![
        LogicalPlan::new(PlanNode::MapPartitions {
            input: scan(1),
            udf: udf(PythonEvalType::MapPandasIter),
            output: out.clone(),
            barrier: false,
        }),
        LogicalPlan::new(PlanNode::GroupMap {
            input: scan(1),
            keys: vec![column(1)],
            udf: udf(PythonEvalType::GroupedMapArrow),
            output: out.clone(),
        }),
        LogicalPlan::new(PlanNode::CoGroupMap {
            left: scan(1),
            right: scan(3),
            left_keys: vec![column(1)],
            right_keys: vec![column(3)],
            udf: udf(PythonEvalType::CogroupedMapPandas),
            output: out.clone(),
        }),
        LogicalPlan::new(PlanNode::PythonTableFunction {
            args: vec![Arc::new(Expr::from(crate::Literal::Integer(1)))],
            table_arg: Some(TableArg {
                input: scan(1),
                partition_by: vec![column(1)],
                order_by: vec![SortOrder::asc(column(2))],
                single_partition: false,
            }),
            udf: udf(PythonEvalType::ArrowUdtf),
            output: out,
        }),
    ]
}

#[test]
fn eval_type_codes_are_sparks_and_round_trip() {
    let codes: Vec<i32> = PythonEvalType::ALL.iter().map(|t| t.code()).collect();
    assert_eq!(
        codes,
        [
            101, 200, 201, 202, 203, 204, 205, 206, 207, 209, 210, 215, 216, 217, 250, 251, 252,
            253, 254, 300, 301, 302
        ]
    );
    for t in PythonEvalType::ALL {
        assert_eq!(PythonEvalType::from_code(t.code()), Some(t));
    }
    // Streaming state (P4) and the plain batched UDF are not plan kinds.
    for code in [100, 208, 211, 212, 213, 214, 0, 999] {
        assert_eq!(PythonEvalType::from_code(code), None, "{code}");
    }
}

#[test]
fn eval_types_classify_by_what_the_udf_sees() {
    let kind = |c| PythonEvalType::from_code(c).map(PythonEvalType::kind);
    for c in [101, 200, 250] {
        assert_eq!(kind(c), Some(PythonUdfKind::Scalar));
        assert!(PythonEvalType::from_code(c).is_some_and(PythonEvalType::is_scalar));
    }
    for (codes, k) in [
        (&[204, 251][..], PythonUdfKind::ScalarIter),
        (&[205, 207], PythonUdfKind::Map),
        (&[201, 209, 215, 216], PythonUdfKind::GroupedMap),
        (&[206, 210], PythonUdfKind::CogroupedMap),
        (&[202, 217, 252, 254], PythonUdfKind::GroupedAgg),
        (&[203, 253], PythonUdfKind::WindowAgg),
        (&[300, 301, 302], PythonUdfKind::Udtf),
    ] {
        for &c in codes {
            assert_eq!(kind(c), Some(k), "{c}");
            assert!(!PythonEvalType::from_code(c).is_some_and(PythonEvalType::is_scalar));
        }
    }
}

#[test]
fn a_udf_reference_is_its_function_identity() {
    let r = udf(PythonEvalType::GroupedAggArrow);
    let id = r.function_id();
    assert_eq!(PythonUdfRef::of(&id), Some(r));
    assert_eq!(PythonUdfRef::of(&FunctionId::Builtin(Builtin::Add)), None);
}

#[test]
fn relation_nodes_walk_their_children_and_keys_and_output_their_declared_columns() {
    let expected_children = [1, 1, 2, 1];
    let expected_exprs = [0, 1, 2, 3];
    for ((plan, children), exprs) in nodes().iter().zip(expected_children).zip(expected_exprs) {
        let mut n = 0;
        plan.for_each_child(&mut |_| n += 1);
        assert_eq!(n, children, "{plan}");
        let mut e = 0;
        plan.for_each_expr(&mut |_| e += 1);
        assert_eq!(e, exprs, "{plan}");
        let out = plan.resolved_output().unwrap_or_default();
        assert_eq!(
            out.iter().map(|c| c.id.0).collect::<Vec<_>>(),
            [10, 11],
            "{plan}"
        );
        assert!(plan.is_resolved(), "{plan}");
        // A rewrite that changes nothing keeps the node.
        let same = Arc::new(plan.clone())
            .transform_up(&mut |p| Ok::<_, std::convert::Infallible>(p))
            .unwrap();
        assert_eq!(*same, *plan);
    }
}

#[test]
fn relation_nodes_encode_apart_by_kind_udf_and_barrier() {
    let plans: Vec<Plan> = nodes()
        .into_iter()
        .map(|p| Plan::try_new(p).unwrap())
        .collect();
    for (i, a) in plans.iter().enumerate() {
        for (j, b) in plans.iter().enumerate() {
            assert_eq!(a.structurally_eq(b), i == j);
        }
    }
    let with = |version: u32, barrier: bool| {
        let mut r = udf(PythonEvalType::MapArrowIter);
        r.version = version;
        Plan::try_new(LogicalPlan::new(PlanNode::MapPartitions {
            input: scan(1),
            udf: r,
            output: vec![col(10, "x")],
            barrier,
        }))
        .unwrap()
    };
    let base = with(1, false);
    assert_ne!(base.structural_hash(), with(2, false).structural_hash());
    assert_ne!(base.structural_hash(), with(1, true).structural_hash());
    assert_eq!(base.structural_hash(), with(1, false).structural_hash());
}

#[test]
fn relation_nodes_print_their_udf_and_columns() {
    let text: Vec<String> = nodes()
        .iter()
        .map(|p| p.to_string().lines().next().unwrap_or_default().to_owned())
        .collect();
    assert_eq!(
        text[0],
        "MapPartitions f#1(205), [x#10, y#11], barrier=false"
    );
    assert_eq!(text[1], "GroupMap [k#1], f#1(209), [x#10, y#11]");
    assert_eq!(text[2], "CoGroupMap [k#1], [k#3], f#1(206), [x#10, y#11]");
    assert!(
        text[3]
            .starts_with("PythonTableFunction f#1(302)(1), [x#10, y#11], TABLE PARTITION BY [k#1]"),
        "{}",
        text[3]
    );
}

#[test]
fn a_lateral_python_udtf_carries_its_call_and_keeps_the_input_columns() {
    let udtf = call(
        udf(PythonEvalType::TableUdf).function_id(),
        FunctionKind::Generator,
        vec![column(1)],
    );
    let plan = LogicalPlan::new(PlanNode::Generate {
        input: scan(1),
        kind: GeneratorKind::PythonUdtf,
        outer: false,
        arg: udtf,
        output: vec![col(10, "x")],
        names: Vec::new(),
        qualifier: None,
    });
    assert_eq!(GeneratorKind::PythonUdtf.name(), "python_udtf");
    let out = plan.resolved_output().unwrap_or_default();
    assert_eq!(out.iter().map(|c| c.id.0).collect::<Vec<_>>(), [1, 2, 10]);
    let mut explode = plan.clone();
    if let PlanNode::Generate { kind, .. } = &mut explode.node {
        *kind = GeneratorKind::Explode;
    }
    assert!(
        !Plan::try_new(plan)
            .unwrap()
            .structurally_eq(&Plan::try_new(explode).unwrap())
    );
}

#[test]
fn a_python_udaf_beside_a_builtin_aggregate_is_refused() {
    let udaf = |name: &str| {
        let mut r = udf(PythonEvalType::GroupedAggPandas);
        r.name = name.to_owned();
        call(r.function_id(), FunctionKind::Aggregate, vec![column(2)])
    };
    let builtin = call(
        FunctionId::Registry {
            name: "sum".to_owned(),
            id: 0,
        },
        FunctionKind::Aggregate,
        vec![column(2)],
    );
    let scalar_python = call(
        udf(PythonEvalType::ScalarPandas).function_id(),
        FunctionKind::Scalar,
        vec![column(2)],
    );
    assert!(check_python_aggregates(&[udaf("f"), udaf("g")]).is_ok());
    assert!(check_python_aggregates(&[Arc::clone(&builtin), scalar_python]).is_ok());
    let e =
        check_python_aggregates(&[udaf("mean_udf"), Arc::clone(&builtin), udaf("g")]).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidPandasUdfPlacement);
    assert_eq!(e.sql_state(), Some("0A000"));
    assert!(e.to_string().contains("`mean_udf`, `g`"), "{e}");
    // Nested in an expression too.
    let plus = call(
        FunctionId::Builtin(Builtin::Add),
        FunctionKind::Scalar,
        vec![udaf("f"), builtin],
    );
    assert!(check_python_aggregates(&[plus]).is_err());
}

#[test]
fn rewrites_reach_the_relation_nodes_children_and_expressions() {
    let one = Arc::new(Expr::from(crate::Literal::Integer(1)));
    for plan in nodes() {
        let plan = Arc::new(plan);
        let mut children = 0;
        let rebuilt = Arc::clone(&plan)
            .map_children(|c| {
                children += 1;
                Ok::<_, std::convert::Infallible>(Arc::new((*c).clone()))
            })
            .unwrap();
        assert!(children > 0, "{plan}");
        assert_eq!(*rebuilt, *plan);
        let mut exprs = 0;
        let rewritten = Arc::clone(&plan)
            .map_expressions(|_| {
                exprs += 1;
                Ok::<_, std::convert::Infallible>(Arc::clone(&one))
            })
            .unwrap();
        let mut after = Vec::new();
        rewritten.for_each_expr(&mut |e| after.push(Arc::clone(e)));
        assert_eq!(after.len(), exprs, "{plan}");
        assert!(after.iter().all(|e| *e == one), "{plan}");
    }
}
