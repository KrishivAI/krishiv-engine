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
