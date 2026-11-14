// SPDX-License-Identifier: Apache-2.0
//! The Python UDF relation nodes are opaque to the optimizer (design P3-T085
//! §3.1, ADR 0022 §5): no filter moves through them and their inputs keep
//! every column (the UDF sees them all).

use std::sync::Arc;

use krishiv_optimizer::optimize;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Literal, LogicalPlan,
    Plan, PlanNode, PythonEvalType, PythonUdfRef, ResolvedExpr, ResolvedFunction,
};
use krishiv_types::DataType;

use super::support::{ctx, resolve};

fn udf(eval_type: PythonEvalType) -> PythonUdfRef {
    PythonUdfRef {
        name: "f".to_owned(),
        version: 1,
        eval_type,
        thread_safe: true,
        deterministic: true,
    }
}

fn column(c: &BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
        c.clone(),
    ))))
}

/// `c > 1` over a `BIGINT` column.
fn above_one(c: &BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Builtin(Builtin::Gt),
            kind: FunctionKind::Scalar,
            args: vec![column(c), Arc::new(Expr::from(Literal::Long(1)))],
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: DataType::Boolean,
            nullable: true,
            deterministic: true,
        },
    ))))
}

/// The scan below `p` (through any single-input nodes).
fn scan_width(p: &LogicalPlan) -> usize {
    let mut found = 0;
    p.for_each_child(&mut |c| found = found.max(scan_width(c)));
    match &p.node {
        PlanNode::Scan(s) => s.output.len(),
        _ => found,
    }
}

/// The relation nodes over `input` (columns `cols`), each declaring one
/// `BIGINT` column `x` with id 1000.
fn relations(input: &Arc<LogicalPlan>, cols: &[BoundColumn]) -> Vec<LogicalPlan> {
    let x = vec![BoundColumn {
        id: AttrId(1000),
        name: "x".to_owned(),
        data_type: DataType::Long,
        nullable: true,
    }];
    vec![
        LogicalPlan::new(PlanNode::MapPartitions {
            input: Arc::clone(input),
            udf: udf(PythonEvalType::MapArrowIter),
            output: x.clone(),
            barrier: false,
        }),
        LogicalPlan::new(PlanNode::GroupMap {
            input: Arc::clone(input),
            keys: vec![column(&cols[0])],
            udf: udf(PythonEvalType::GroupedMapPandas),
            output: x.clone(),
        }),
        LogicalPlan::new(PlanNode::CoGroupMap {
            left: Arc::clone(input),
            right: Arc::clone(input),
            left_keys: vec![column(&cols[0])],
            right_keys: vec![column(&cols[0])],
            udf: udf(PythonEvalType::CogroupedMapArrow),
            output: x,
        }),
    ]
}

#[test]
fn a_filter_stays_above_a_python_relation_whose_input_keeps_every_column() {
    let base = resolve("SELECT * FROM nation", &ctx());
    let input = Arc::clone(base.root());
    let cols = input.resolved_output().unwrap();
    let width = cols.len();
    assert!(width > 2, "nation has several columns");
    let x = BoundColumn {
        id: AttrId(1000),
        name: "x".to_owned(),
        data_type: DataType::Long,
        nullable: true,
    };
    for node in relations(&input, &cols) {
        let kind = node
            .to_string()
            .split_whitespace()
            .next()
            .unwrap()
            .to_owned();
        let plan = Plan::try_new(LogicalPlan::new(PlanNode::Filter {
            input: Arc::new(node),
            predicate: above_one(&x),
        }))
        .unwrap();
        let out = optimize(&plan).unwrap();
        let root = out.root();
        let PlanNode::Filter { input, .. } = &root.node else {
            panic!("{kind}: the filter moved:\n{root}");
        };
        assert!(
            input.to_string().starts_with(&kind),
            "{kind}: not directly above it:\n{root}"
        );
        assert_eq!(
            scan_width(root),
            width,
            "{kind}: an input column was pruned:\n{root}"
        );
    }
}

/// The `CteRef`s left in `p`.
fn cte_refs(p: &LogicalPlan) -> usize {
    let mut n = usize::from(matches!(p.node, PlanNode::CteRef(_)));
    p.for_each_child(&mut |c| n += cte_refs(c));
    n
}

/// `WITH c AS (<a map of nation over a UDF deterministic or not>) SELECT *
/// FROM c UNION ALL SELECT * FROM c`, optimized.
fn twice_referenced_map(deterministic: bool) -> Arc<LogicalPlan> {
    let base = resolve("SELECT * FROM nation", &ctx());
    let x = |id: u64| BoundColumn {
        id: AttrId(id),
        name: "x".to_owned(),
        data_type: DataType::Long,
        nullable: true,
    };
    let def = Arc::new(LogicalPlan::new(PlanNode::MapPartitions {
        input: Arc::clone(base.root()),
        udf: PythonUdfRef {
            deterministic,
            ..udf(PythonEvalType::MapArrowIter)
        },
        output: vec![x(1000)],
        barrier: false,
    }));
    let reference = |id: u64| {
        Arc::new(LogicalPlan::new(PlanNode::CteRef(krishiv_plan::CteRef {
            name: "c".to_owned(),
            source: vec![AttrId(1000)],
            output: vec![x(id)],
        })))
    };
    let plan = Plan::try_new(LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::new(PlanNode::Union {
            inputs: vec![reference(2000), reference(3000)],
            by_name: false,
            allow_missing_columns: false,
        })),
        ctes: vec![krishiv_plan::Cte {
            name: "c".to_owned(),
            plan: Arc::clone(&def),
        }],
    }))
    .unwrap();
    assert_eq!(
        krishiv_optimizer::plan_is_deterministic(&def),
        deterministic
    );
    Arc::clone(optimize(&plan).unwrap().root())
}

/// A map over a nondeterministic UDF (Spark's `MapInPandas` allows one,
/// `CheckAnalysis` at v4.2.0) makes its CTE nondeterministic: referenced
/// twice, it is computed once and shared, not inlined per reference
/// (Spark's `InlineCTE.shouldInline`, `cteDef.deterministic`, which sees
/// the node's `PythonUDF`). A deterministic one is inlined.
#[test]
fn a_cte_over_a_nondeterministic_python_map_is_shared_not_inlined() {
    let shared = twice_referenced_map(false);
    assert_eq!(cte_refs(&shared), 2, "{shared}");
    let inlined = twice_referenced_map(true);
    assert_eq!(cte_refs(&inlined), 0, "{inlined}");
}
