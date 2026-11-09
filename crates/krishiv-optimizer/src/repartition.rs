// SPDX-License-Identifier: Apache-2.0
//! Spark's `CollapseRepartition` for `DISTRIBUTE BY`: a repartition
//! directly over another one sets the partitions alone, so the inner one
//! is dropped and its expressions are never evaluated (checked on Spark
//! 4.2.0: `SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT))
//! DISTRIBUTE BY x` returns the row the inner cast fails on). A local sort
//! (`SORT BY`) between them keeps both, as in Spark. Spark also drops a
//! global sort directly below a repartition; that changes no result here
//! (the rows a query returns are unordered either way), so it is left to
//! the physical plan.

use std::sync::Arc;

use krishiv_plan::{LogicalPlan, PlanNode};

/// The rule at a [`PlanNode::Repartition`].
pub(crate) fn collapse(node: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let PlanNode::Repartition { input, exprs } = &node.node else {
        return node;
    };
    let PlanNode::Repartition { input: below, .. } = &input.node else {
        return node;
    };
    let mut out = LogicalPlan::clone(&node);
    out.node = PlanNode::Repartition {
        input: Arc::clone(below),
        exprs: exprs.clone(),
    };
    Arc::new(out)
}
