// SPDX-License-Identifier: Apache-2.0
//! Python UDF relation kinds in plans (design P3-T085 §3.1): the checks
//! shared by the analyzer and the physical planner.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::error::{
    InvalidNonDeterministicExpressions, InvalidPandasUdfPlacement, KrishivError,
};

use crate::{Expr, ExprNode, FunctionId, FunctionKind, PlanNode, PythonUdfKind, ResolvedExpr};

/// Whether `e` is a call of a grouped-aggregate Python UDF (202, 217, 252,
/// 254).
#[must_use]
pub fn is_python_udaf(e: &Expr) -> bool {
    matches!(
        &e.node,
        ExprNode::Resolved(ResolvedExpr::Function(f)) if matches!(
            &f.id,
            FunctionId::PythonUdf { eval_type, .. } if eval_type.kind() == PythonUdfKind::GroupedAgg
        )
    )
}

/// Whether the Python relation UDF `node` runs, if it runs one, is
/// deterministic. A relation node's UDF is not an expression of it, so a
/// plan's determinism check asks this beside its expressions' (Spark's
/// `QueryPlan.deterministic` sees the node's `functionExpr`, a `PythonUDF`,
/// at v4.2.0).
#[must_use]
pub fn python_relation_is_deterministic(node: &PlanNode) -> bool {
    match node {
        PlanNode::MapPartitions { udf, .. }
        | PlanNode::GroupMap { udf, .. }
        | PlanNode::CoGroupMap { udf, .. }
        | PlanNode::PythonTableFunction { udf, .. } => udf.deterministic,
        _ => true,
    }
}

/// Whether every expression of `plan` (subquery plans included) and every Python relation UDF it runs is
/// deterministic: the optimizer's test for inlining a `WITH` definition at each reference (`InlineCTE`), shared
/// with the analyzer's binding of `input_file_name()`, which instantiates a definition the same way.
#[must_use]
pub fn plan_is_deterministic(plan: &crate::LogicalPlan) -> bool {
    use crate::walk::{Node, visit_distinct};
    let mut ok = true;
    let _ = visit_distinct(Node::Plan(plan), |n| {
        ok &= match n {
            Node::Plan(p) => python_relation_is_deterministic(&p.node),
            Node::Expr(e) => !matches!(
                &e.node,
                crate::ExprNode::Resolved(crate::ResolvedExpr::Function(f)) if !f.deterministic
            ),
        };
        if ok {
            std::ops::ControlFlow::Continue(())
        } else {
            std::ops::ControlFlow::Break(())
        }
    });
    ok
}

/// Refuses a grouped or cogrouped map whose UDF is nondeterministic:
/// Spark's `CheckAnalysis` allows nondeterministic expressions only on some
/// operators, of the Python relations `MapInPandas` and `MapInArrow`
/// (CheckAnalysis.scala L1032-1052 at v4.2.0), and a node's `functionExpr`
/// is a `PythonUDF` whose `deterministic` is `udfDeterministic`. Spark
/// lists the node's expressions (`QueryPlan.expressions`), each as
/// `toSQLExpr` shows it: a grouped map's grouping attributes, its UDF call
/// over the input's columns (`PrettyPythonUDF`: `name(a, b)`) and its
/// output attributes; a cogroup's UDF call and output (its key counts are
/// not expressions). Spark raises it at analysis (`df.schema`, `explain`
/// and SQL fail); no front end builds these nodes yet (Connect's
/// `GroupMap`/`CoGroupMap` are not lowered, P3-T014), so for now only the
/// physical planner calls it, at a DataFrame's first action. The front end
/// that builds them must call it where it resolves the node.
///
/// Known differences: the UDF's name is the canonical lower-case one (a
/// [`crate::PythonUdfRef`] keeps no other), and a
/// computed key shows as `toPrettySQL` names it, where Spark aliases it by
/// the unresolved expression's `toString`. The text is derived from the
/// source; a Spark-run golden awaits a lane with Spark.
///
/// # Errors
/// `INVALID_NON_DETERMINISTIC_EXPRESSIONS` (42K0E).
pub fn check_python_relation_determinism(node: &PlanNode) -> Result<(), KrishivError> {
    let call = |udf: &crate::PythonUdfRef, inputs: &[&crate::LogicalPlan]| {
        let args: Vec<String> = inputs
            .iter()
            .flat_map(|p| p.resolved_output().unwrap_or_default())
            .map(|c| c.name)
            .collect();
        format!(
            "{}({})",
            crate::function::shown_call_name("", &udf.name),
            args.join(", ")
        )
    };
    let shown: Vec<String> = match node {
        PlanNode::GroupMap {
            input,
            keys,
            udf,
            output,
        } if !udf.deterministic => keys
            .iter()
            .map(|k| crate::analyzer::pretty_sql(k))
            .chain(std::iter::once(call(udf, &[input])))
            .chain(output.iter().map(|c| c.name.clone()))
            .collect(),
        PlanNode::CoGroupMap {
            left,
            right,
            udf,
            output,
            ..
        } if !udf.deterministic => std::iter::once(call(udf, &[left, right]))
            .chain(output.iter().map(|c| c.name.clone()))
            .collect(),
        _ => return Ok(()),
    };
    Err(InvalidNonDeterministicExpressions {
        sql_exprs: shown
            .iter()
            .map(|s| format!("\"{s}\""))
            .collect::<Vec<_>>()
            .join(", "),
    }
    .into())
}

/// Refuses an aggregate whose expressions call a grouped-aggregate Python UDF
/// beside any other aggregate function (Spark's planner,
/// `SparkStrategies.scala` L670-671 at v4.2.0: `INVALID_PANDAS_UDF_PLACEMENT`,
/// the UDFs' names quoted as `toSQLId` quotes them). Called by the physical
/// planner, where Spark raises it: analysis accepts such an aggregate, so a
/// DataFrame is refused at its first action, not at creation. Subqueries
/// are not looked into.
///
/// Known difference: the names are the canonical lower-case ones a plan
/// carries, where Spark prints the name as registered.
///
/// # Errors
/// `INVALID_PANDAS_UDF_PLACEMENT` (0A000).
pub fn check_python_aggregates<'a>(
    aggregates: impl IntoIterator<Item = &'a Arc<Expr>>,
) -> Result<(), KrishivError> {
    let mut udafs: Vec<String> = Vec::new();
    let mut other = false;
    let mut seen = HashSet::new();
    // Depth first, in written order: the names are listed as first seen.
    let mut stack: Vec<&Expr> = aggregates.into_iter().map(AsRef::as_ref).collect();
    stack.reverse();
    while let Some(e) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(e).addr()) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node {
            if is_python_udaf(e) {
                let name = crate::function::shown_call_name("", f.id.name());
                let quoted = format!("`{}`", name.replace('`', "``"));
                if !udafs.contains(&quoted) {
                    udafs.push(quoted);
                }
                continue;
            }
            if f.kind == FunctionKind::Aggregate {
                other = true;
            }
        }
        let at = stack.len();
        e.for_each_child(&mut |c| stack.push(c));
        stack[at..].reverse();
    }
    if other && !udafs.is_empty() {
        return Err(InvalidPandasUdfPlacement {
            function_list: udafs.join(", "),
        }
        .into());
    }
    Ok(())
}
