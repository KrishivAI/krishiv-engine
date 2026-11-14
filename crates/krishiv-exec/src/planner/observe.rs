// SPDX-License-Identifier: Apache-2.0
//! `Dataset.observe`: [`PhysNode::CollectMetrics`].
//!
//! The metrics are a global aggregate over the rows that pass through, so
//! they plan as one (`Planner::aggregate` over a placeholder relation
//! with the input's columns), and the node keeps the resulting
//! projections and aggregate as plain data (`measure`). The rows
//! themselves are planned as they are: nothing moves below the node (a
//! runtime filter would change what the metrics count) and nothing above
//! it sees a difference.

use std::sync::Arc;

use krishiv_plan::{Expr, Grouping, LogicalPlan};

use super::build::{Planned, Planner, internal};
use super::explain::ExplainNode;
use super::properties::Placement;
use crate::error::Result;
use crate::physical::PhysNode;

impl Planner<'_> {
    /// `input`'s rows, observed as `metrics` under `name`. Where the rows
    /// are never read (a `LIMIT 0` above, which Spark replaces with an
    /// empty relation) the node is not planned.
    pub(super) fn collect_metrics(
        &mut self,
        input: &Arc<LogicalPlan>,
        name: &str,
        metrics: &[Arc<Expr>],
    ) -> Result<Planned> {
        // Numbered before its input's nodes: Spark's plan order is pre-order.
        let instance = self.observers;
        self.observers += 1;
        let planned = self.plan(input)?;
        if self.unobserved > 0 {
            return Ok(planned);
        }
        let rows = Planned {
            node: PhysNode::OneRow,
            schema: Arc::clone(&planned.schema),
            layout: planned.layout.clone(),
            explain: ExplainNode::new("ObservedRows", String::new(), Vec::new()),
            sites: Vec::new(),
            placement: Placement::Single,
            ordering: Vec::new(),
        };
        let measured = self.aggregate(rows, &Grouping::GroupBy(Vec::new()), metrics)?;
        if !matches!(measured.node, PhysNode::Project { .. }) {
            return Err(internal("observed metrics did not plan as a projection"));
        }
        let (schema, layout) = (Arc::clone(&planned.schema), planned.layout.clone());
        let mut out = planned.wrap(
            |input| PhysNode::CollectMetrics {
                input,
                name: name.to_owned(),
                measure: Box::new(measured.node),
                instance,
                limit: None,
            },
            schema,
            layout,
            ("CollectMetrics", name.to_owned()),
        );
        // A filter below would change which rows the metrics count.
        out.sites.clear();
        Ok(out)
    }
}

/// Caps the rows the observing nodes of `node` fold at `rows`, through the operators that pass
/// every row on (a projection, another observing node): a `LIMIT` above takes no more than
/// that, and Spark's `CollectMetricsExec` counts the rows its consumer pulled. A filter, sort,
/// join or any operator that changes which rows arrive ends the search.
pub(super) fn cap_observed(node: &mut PhysNode, rows: u64) {
    match node {
        PhysNode::Project { input, .. } => cap_observed(input, rows),
        PhysNode::CollectMetrics { input, limit, .. } => {
            *limit = Some(limit.map_or(rows, |l| l.min(rows)));
            cap_observed(input, rows);
        }
        _ => {}
    }
}
