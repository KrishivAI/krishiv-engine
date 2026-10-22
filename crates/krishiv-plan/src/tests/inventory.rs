// SPDX-License-Identifier: Apache-2.0
//! Checks the node inventory in the crate rustdoc: the TPC-H column against
//! the hand-built TPC-H plans, the DataFrame-slice column against plans
//! built as each slice operation builds them, and the rows against every
//! node kind.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use krishiv_types::DataType;

use super::dsl::{c, int};
use super::tpch::QUERIES;
use crate::{
    Builtin, ColumnRef, Expr, ExprNode, JoinCondition, JoinKind, LogicalPlan, PlanId, PlanNode,
    ReadSource, ResolvedExpr, SortOrder, SqlArgs, Star,
};

/// The node kinds a plan uses, named as in the inventory. No wildcard arm:
/// a new node does not compile until it is named here.
pub(super) fn kinds(plan: &LogicalPlan, out: &mut BTreeSet<&'static str>) {
    out.insert(match &plan.node {
        PlanNode::Read(_) => "PlanNode::Read",
        PlanNode::Sql { .. } => "PlanNode::Sql",
        PlanNode::OneRowRelation => "PlanNode::OneRowRelation",
        PlanNode::Scan(_) => "PlanNode::Scan",
        PlanNode::Project { .. } => "PlanNode::Project",
        PlanNode::WithColumns { .. } => "PlanNode::WithColumns",
        PlanNode::Filter { .. } => "PlanNode::Filter",
        PlanNode::Aggregate { .. } => "PlanNode::Aggregate",
        PlanNode::Join { .. } => "PlanNode::Join",
        PlanNode::Sort { .. } => "PlanNode::Sort",
        PlanNode::Limit { .. } => "PlanNode::Limit",
        PlanNode::SubqueryAlias { .. } => "PlanNode::SubqueryAlias",
        PlanNode::WithCte { .. } => "PlanNode::WithCte",
        PlanNode::CteRef(_) => "PlanNode::CteRef",
        PlanNode::Shift { .. } => "PlanNode::Shift",
        PlanNode::Hint { .. } => "PlanNode::Hint",
        PlanNode::JoinHint { .. } => "PlanNode::JoinHint",
        PlanNode::Union { .. } => "PlanNode::Union",
        PlanNode::Distinct { .. } => "PlanNode::Distinct",
        PlanNode::SetOp { .. } => "PlanNode::SetOp",
        PlanNode::ReplicateRows { .. } => "PlanNode::ReplicateRows",
        PlanNode::Window { .. } => "PlanNode::Window",
    });
    plan.for_each_expr(&mut |e| expr_kinds(e, out));
    plan.for_each_child(&mut |c| kinds(c, out));
}

fn expr_kinds(expr: &Expr, out: &mut BTreeSet<&'static str>) {
    out.insert(match &expr.node {
        ExprNode::Column(_) => "ExprNode::Column",
        ExprNode::Literal(_) => "ExprNode::Literal",
        ExprNode::Star(_) => "ExprNode::Star",
        ExprNode::Function(_) => "ExprNode::Function",
        ExprNode::Cast { .. } => "ExprNode::Cast",
        ExprNode::Alias { .. } => "ExprNode::Alias",
        ExprNode::Subquery { .. } => "ExprNode::Subquery",
        ExprNode::Window(_) => "ExprNode::Window",
        ExprNode::Resolved(ResolvedExpr::Column(_)) => "ResolvedExpr::Column",
        ExprNode::Resolved(ResolvedExpr::OuterColumn(_)) => "ResolvedExpr::OuterColumn",
        ExprNode::Resolved(ResolvedExpr::Function(_)) => "ResolvedExpr::Function",
        ExprNode::Resolved(ResolvedExpr::Alias { .. }) => "ResolvedExpr::Alias",
    });
    if let ExprNode::Subquery { plan, .. } = &expr.node {
        kinds(plan, out);
    }
    expr.for_each_child(&mut |c| expr_kinds(c, out));
}

/// The inventory rows: node, TPC-H cell, DataFrame-slice cell.
fn rows() -> Vec<(&'static str, &'static str, &'static str)> {
    include_str!("../lib.rs")
        .lines()
        .filter_map(|l| l.strip_prefix("//! | `"))
        .map(|row| {
            let cells: Vec<&str> = row.split(" | ").collect();
            (
                cells[0].trim_end_matches('`'),
                cells[2],
                cells[3].trim_end_matches(" |"),
            )
        })
        .collect()
}

/// Renders query numbers as the inventory writes them.
fn render_queries(qs: &BTreeSet<usize>) -> String {
    if qs.is_empty() {
        return "-".to_owned();
    }
    if qs.len() == QUERIES.len() {
        return "all".to_owned();
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &q in qs {
        match runs.last_mut() {
            Some((_, end)) if *end + 1 == q => *end = q,
            _ => runs.push((q, q)),
        }
    }
    runs.iter()
        .map(|&(a, b)| match b - a {
            0 => format!("Q{a}"),
            1 => format!("Q{a}, Q{b}"),
            _ => format!("Q{a}-Q{b}"),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[test]
fn tpch_column_matches_the_tpch_plans() {
    let mut used: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    for (i, q) in QUERIES.iter().enumerate() {
        let mut set = BTreeSet::new();
        kinds(&q(), &mut set);
        for k in set {
            used.entry(k).or_default().insert(i + 1);
        }
    }
    for (node, tpch, _) in rows() {
        let expected = render_queries(used.get(node).unwrap_or(&BTreeSet::new()));
        assert_eq!(tpch, expected, "TPC-H column of `{node}`");
    }
}

fn df(id: i64) -> LogicalPlan {
    LogicalPlan::table(["t"]).with_origin(crate::Origin::plan(PlanId(id)))
}

/// Each DataFrame-slice operation, as the API builds it.
fn slice_ops() -> Vec<(&'static str, LogicalPlan)> {
    let star = Expr::new(ExprNode::Star(Star {
        qualifier: Vec::new(),
        plan_id: None,
    }));
    let df_col = Expr::new(ExprNode::Column(ColumnRef {
        name: vec!["a".to_owned()],
        plan_id: Some(PlanId(1)),
    }));
    let parquet = LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: "parquet".to_owned(),
        paths: vec![krishiv_common::secret::OptionValue::from_user(
            "path", "/data",
        )],
        options: BTreeMap::new(),
    }));
    let sql = LogicalPlan::new(PlanNode::Sql {
        query: "SELECT 1".to_owned(),
        args: SqlArgs::None,
    });
    let with_column = LogicalPlan::new(PlanNode::WithColumns {
        input: Arc::new(df(1)),
        columns: vec![("b".to_owned(), Arc::new(c("a")))],
    });
    let sum = Expr::builtin(Builtin::Add, vec![c("a"), int(1)]);
    vec![
        ("read.table", df(1)),
        ("read.parquet", parquet),
        ("sql", sql),
        ("select", df(1).project(vec![c("a")])),
        ("withColumn", with_column),
        ("filter", df(1).filter(c("a"))),
        ("where", df(1).filter(c("a"))),
        ("groupBy.agg", df(1).aggregate(vec![c("a")], vec![c("a")])),
        (
            "join",
            df(1).join(df(2), JoinKind::Inner, JoinCondition::None),
        ),
        ("orderBy", df(1).sort(vec![SortOrder::asc(c("a"))])),
        ("sort", df(1).sort(vec![SortOrder::asc(c("a"))])),
        ("limit", df(1).limit(int(1))),
        ("df.alias", df(1).alias("x")),
        ("col", df(1).project(vec![c("a")])),
        ("df[\"a\"]", df(1).project(vec![df_col])),
        ("lit", df(1).project(vec![int(1)])),
        ("select(\"*\")", df(1).project(vec![star])),
        ("operators", df(1).project(vec![sum])),
        (
            "functions",
            df(1).project(vec![Expr::function("upper", vec![c("a")])]),
        ),
        ("cast", df(1).project(vec![c("a").cast(DataType::Long)])),
        ("col.alias", df(1).project(vec![c("a").alias("b")])),
    ]
}

#[test]
fn dataframe_column_matches_the_slice_operations() {
    let ops: BTreeMap<&str, LogicalPlan> = slice_ops().into_iter().collect();
    let mut listed = BTreeSet::new();
    for (node, _, slice) in rows() {
        for op in slice.split(", ").filter(|s| *s != "-") {
            let op = op.trim_matches('`');
            let plan = ops.get(op).unwrap_or_else(|| panic!("no plan for `{op}`"));
            let mut used = BTreeSet::new();
            kinds(plan, &mut used);
            assert!(used.contains(node), "`{op}` does not build `{node}`");
            listed.insert(op);
        }
    }
    let built: BTreeSet<&str> = ops.keys().copied().collect();
    assert_eq!(listed, built, "every slice operation is listed");
}

#[test]
fn rows_cover_every_node_kind_once() {
    let mut seen = BTreeSet::new();
    for (node, _, _) in rows() {
        assert!(seen.insert(node), "`{node}` listed twice");
    }
    let generated = generated_kinds();
    assert_eq!(seen, generated, "inventory rows = node kinds");
}

/// The node kinds the generator produces over many inputs; the inventory
/// test requires them to be every inventory row.
fn generated_kinds() -> BTreeSet<&'static str> {
    let mut out = BTreeSet::new();
    for seed in 0..4000_u32 {
        let bytes: Vec<u8> = (0..400_u32)
            .map(|j| u8::try_from((seed * 7919 + j * 104_729 + j * j) % 251).unwrap())
            .collect();
        kinds(super::props::build(&bytes, false).root(), &mut out);
    }
    out
}
