// SPDX-License-Identifier: Apache-2.0
//! Python UDF relation kinds in plans (design P3-T085 §3.1): the checks
//! shared by the analyzer and the physical planner.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::error::{InvalidPandasUdfPlacement, KrishivError};

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
