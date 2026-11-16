// SPDX-License-Identifier: Apache-2.0
//! The analyzer over the Python UDF relations a front end builds (Spark
//! Connect's `MapPartitions`, `GroupMap`, `CoGroupMap`; P3-T014): inputs and
//! keys resolve, the declared output gets ids unique in the plan, and a
//! grouped or cogrouped map over a nondeterministic UDF is refused at
//! analysis (Spark's `CheckAnalysis`, v4.3.0 L1123-1143).

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_types::DataType;

use super::dsl::{c, f};
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisContext, AnalysisError, AnalyzedPlan, Analyzer as _, CachingAnalyzer, DefaultAnalyzer,
    MemoryCatalog, ResolutionConf, TableColumn,
};
use crate::{
    AttrId, BoundColumn, Expr, ExprNode, JoinCondition, JoinKind, LogicalPlan, Origin, Plan,
    PlanNode, PythonEvalType, PythonUdfRef, ReadSource, ResolvedExpr,
};

fn ctx() -> AnalysisContext {
    let cat = MemoryCatalog::new("spark_catalog")
        .with_table(
            &["default"],
            "t",
            vec![
                TableColumn::new("k".to_owned(), DataType::Long, true),
                TableColumn::new("v".to_owned(), DataType::String, true),
            ],
            None,
        )
        .with_table(
            &["default"],
            "u",
            vec![TableColumn::new("k2".to_owned(), DataType::Long, false)],
            None,
        );
    context(cat, ResolutionConf::default())
}

fn read(name: &str) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Read(ReadSource::Table {
        name: vec![name.to_owned()],
        options: BTreeMap::new(),
    })))
}

fn udf(eval_type: PythonEvalType, deterministic: bool) -> PythonUdfRef {
    PythonUdfRef {
        name: "f".to_owned(),
        version: 1,
        eval_type,
        thread_safe: true,
        deterministic,
    }
}

/// The declared columns `x bigint, y string not null`, with the ids a
/// translator gives (0 and 1: they meet the table's own).
fn declared() -> Vec<BoundColumn> {
    vec![
        BoundColumn {
            id: AttrId(0),
            name: "x".to_owned(),
            data_type: DataType::Long,
            nullable: true,
        },
        BoundColumn {
            id: AttrId(1),
            name: "y".to_owned(),
            data_type: DataType::String,
            nullable: false,
        },
    ]
}

fn map(deterministic: bool) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::MapPartitions {
        input: read("t"),
        udf: udf(PythonEvalType::MapPandasIter, deterministic),
        output: declared(),
        barrier: false,
    }))
}

fn group_map(keys: Vec<Expr>, deterministic: bool) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::GroupMap {
        input: read("t"),
        keys: keys.into_iter().map(Arc::new).collect(),
        udf: udf(PythonEvalType::GroupedMapPandas, deterministic),
        output: declared(),
    }))
}

fn cogroup(deterministic: bool) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::CoGroupMap {
        left: read("t"),
        right: read("u"),
        left_keys: vec![Arc::new(c("k"))],
        right_keys: vec![Arc::new(c("k2"))],
        udf: udf(PythonEvalType::CogroupedMapArrow, deterministic),
        output: declared(),
    }))
}

fn analyze(root: Arc<LogicalPlan>) -> Result<AnalyzedPlan, AnalysisError> {
    let plan = Plan::try_new(root).map_err(|error| AnalysisError {
        error,
        origin: Origin::UNKNOWN,
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, &ctx()))
}

fn shape(a: &AnalyzedPlan) -> Vec<(String, DataType, bool)> {
    a.output
        .iter()
        .map(|c| (c.name.clone(), c.data_type.clone(), c.nullable))
        .collect()
}

fn class(e: &AnalysisError) -> &'static str {
    e.error.error_class().name()
}

/// Every attribute id a resolved node of `p` defines or reads, once each.
fn ids(p: &LogicalPlan) -> Vec<AttrId> {
    let mut out = p
        .resolved_output()
        .unwrap_or_default()
        .iter()
        .map(|c| c.id)
        .collect::<Vec<_>>();
    p.for_each_child(&mut |k| out.extend(ids(k)));
    out
}

#[test]
fn a_map_resolves_its_input_and_declares_its_output() {
    let a = analyze(map(true)).unwrap();
    assert_eq!(
        shape(&a),
        [
            ("x".to_owned(), DataType::Long, true),
            ("y".to_owned(), DataType::String, false),
        ]
    );
    let PlanNode::MapPartitions { input, output, .. } = &a.plan.root().node else {
        panic!("{}", a.plan)
    };
    assert!(input.is_resolved());
    // The table's columns and the declared ones never share an id.
    let mut all = ids(a.plan.root());
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), n, "{all:?}");
    assert_eq!(output.len(), 2);
    assert!(a.plan.root().is_resolved());
}

#[test]
fn a_nondeterministic_map_is_allowed() {
    // CheckAnalysis lets MapInPandas and MapInArrow hold one.
    assert!(analyze(map(false)).is_ok());
}

#[test]
fn a_projection_above_reads_the_declared_columns() {
    let root = Arc::new(LogicalPlan::new(PlanNode::Project {
        input: map(true),
        exprs: vec![Arc::new(c("y")), Arc::new(c("x"))],
    }));
    let a = analyze(root).unwrap();
    let names: Vec<&str> = a.output.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["y", "x"]);
    // Analysing the analysed plan again gives the same plan (a fixed point).
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx())).unwrap();
    assert_eq!(again.output, a.output);
}

#[test]
fn two_nodes_with_the_same_declared_ids_stay_apart() {
    let root = Arc::new(LogicalPlan::new(PlanNode::Join {
        left: map(true),
        right: group_map(vec![c("k")], true),
        kind: JoinKind::Cross,
        condition: JoinCondition::None,
    }));
    let a = analyze(root).unwrap();
    let mut out: Vec<AttrId> = a.output.iter().map(|c| c.id).collect();
    assert_eq!(out.len(), 4);
    out.sort_unstable();
    out.dedup();
    assert_eq!(out.len(), 4, "{:?}", a.output);
}

#[test]
fn a_grouped_map_binds_its_keys_over_its_input() {
    let a = analyze(group_map(vec![c("k")], true)).unwrap();
    let PlanNode::GroupMap { keys, input, .. } = &a.plan.root().node else {
        panic!("{}", a.plan)
    };
    let in_ids: Vec<AttrId> = input
        .resolved_output()
        .unwrap_or_default()
        .iter()
        .map(|c| c.id)
        .collect();
    let [key] = keys.as_slice() else {
        panic!("{keys:?}")
    };
    let ExprNode::Resolved(ResolvedExpr::Column(k)) = &key.node else {
        panic!("{key:?}")
    };
    assert_eq!(k.name, "k");
    assert!(in_ids.contains(&k.id));
    assert_eq!(shape(&a).len(), 2);
    // No keys: one group.
    assert!(analyze(group_map(Vec::new(), true)).is_ok());
}

#[test]
fn a_grouped_map_key_must_name_an_input_column() {
    let e = analyze(group_map(vec![c("nope")], true)).unwrap_err();
    assert!(class(&e).starts_with("UNRESOLVED_COLUMN"), "{}", e.error);
}

#[test]
fn a_grouped_map_key_cannot_aggregate() {
    let e = analyze(group_map(vec![f("max", vec![c("k")])], true)).unwrap_err();
    assert_eq!(class(&e), "KRISHIV_UNSUPPORTED_FEATURE", "{}", e.error);
    assert!(
        e.error
            .message()
            .contains("an aggregate in a grouping key of FlatMapGroupsInPandas"),
        "{}",
        e.error
    );
    let e = analyze(group_map(
        vec![f("explode", vec![f("array", vec![c("k")])])],
        true,
    ))
    .unwrap_err();
    assert_eq!(class(&e), "KRISHIV_UNSUPPORTED_FEATURE", "{}", e.error);
    assert!(e.error.message().contains("a generator"), "{}", e.error);
}

#[test]
fn a_nondeterministic_grouped_map_is_refused_at_analysis() {
    let e = analyze(group_map(vec![c("k")], false)).unwrap_err();
    assert_eq!(class(&e), "INVALID_NON_DETERMINISTIC_EXPRESSIONS");
    assert_eq!(e.error.sql_state(), Some("42K0E"));
    let text = e.error.to_string();
    for shown in ["\"k\"", "\"f(k, v)\"", "\"x\"", "\"y\""] {
        assert!(text.contains(shown), "{shown}: {text}");
    }
    // Also under a projection (`df.select(...)` over it): the node's own check.
    let root = Arc::new(LogicalPlan::new(PlanNode::Project {
        input: group_map(vec![c("k")], false),
        exprs: vec![Arc::new(c("x"))],
    }));
    assert_eq!(
        class(&analyze(root).unwrap_err()),
        "INVALID_NON_DETERMINISTIC_EXPRESSIONS"
    );
}

#[test]
fn a_cogroup_binds_each_side_and_refuses_a_nondeterministic_udf() {
    let a = analyze(cogroup(true)).unwrap();
    let PlanNode::CoGroupMap {
        left_keys,
        right_keys,
        ..
    } = &a.plan.root().node
    else {
        panic!("{}", a.plan)
    };
    assert_eq!((left_keys.len(), right_keys.len()), (1, 1));
    let e = analyze(cogroup(false)).unwrap_err();
    assert_eq!(class(&e), "INVALID_NON_DETERMINISTIC_EXPRESSIONS");
    let text = e.error.to_string();
    assert!(text.contains("\"f(k, v, k2)\""), "{text}");
}

#[test]
fn a_cogroup_key_binds_over_its_own_side_only() {
    // `k2` is the right side's: the left keys cannot see it.
    let root = Arc::new(LogicalPlan::new(PlanNode::CoGroupMap {
        left: read("t"),
        right: read("u"),
        left_keys: vec![Arc::new(c("k2"))],
        right_keys: vec![Arc::new(c("k2"))],
        udf: udf(PythonEvalType::CogroupedMapPandas, true),
        output: declared(),
    }));
    let e = analyze(root).unwrap_err();
    assert!(class(&e).starts_with("UNRESOLVED_COLUMN"), "{}", e.error);
}

fn distinct_ids(a: &AnalyzedPlan) -> usize {
    let mut out: Vec<AttrId> = a.output.iter().map(|c| c.id).collect();
    out.sort_unstable();
    out.dedup();
    out.len()
}

fn cross(left: Arc<LogicalPlan>, right: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Join {
        left,
        right,
        kind: JoinKind::Cross,
        condition: JoinCondition::None,
    }))
}

#[test]
fn a_warm_analysis_gives_the_python_relations_the_cold_ids() {
    // A client grows `m1` into `m1.crossJoin(m2)`: the prefix cache starts from `m1`, which kept the
    // translator's ids 0 and 1; `m2` carries the same ids and must get fresh ones, warm as cold.
    let cx = ctx();
    let cache = CachingAnalyzer::new();
    let first = Plan::try_new(map(true)).unwrap();
    block_on(cache.analyze(&first, &cx)).unwrap();
    // `m2` reads another table: a relation equal to one inside the prefix (`Read t`) would be a miss.
    let m2 = Arc::new(LogicalPlan::new(PlanNode::MapPartitions {
        input: read("u"),
        udf: udf(PythonEvalType::MapArrowIter, true),
        output: declared(),
        barrier: false,
    }));
    let grown = Plan::try_new(cross(map(true), m2)).unwrap();
    let before = cache.warm_starts();
    let warm = block_on(cache.analyze(&grown, &cx)).unwrap();
    assert!(cache.warm_starts() > before, "the grown plan starts warm");
    let cold = block_on(DefaultAnalyzer.analyze(&grown, &cx)).unwrap();
    assert_eq!(warm, cold);
    assert_eq!(distinct_ids(&warm), 4, "{:?}", warm.output);
}

#[test]
fn an_analysed_python_relation_reached_twice_stays_apart() {
    // A stored analysed plan (a temporary view's) read on both sides of a self-join.
    let stored = analyze(map(true)).unwrap();
    let view = Arc::clone(stored.plan.root());
    let a = analyze(cross(Arc::clone(&view), view)).unwrap();
    assert_eq!(a.output.len(), 4);
    assert_eq!(distinct_ids(&a), 4, "{:?}", a.output);
}
