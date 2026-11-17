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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_plan::{Expr, Grouping, LogicalPlan};

use super::build::{Planned, Planner, internal};
use super::explain::ExplainNode;
use super::properties::Placement;
use crate::error::Result;
use crate::physical::{PhysNode, SharedId, SharedPlan};

/// The reads of shared plans, to tell the shared plans something live reads from those only a
/// `LIMIT 0` reads (Spark replaces the zero-row read with an empty relation, so what only that
/// reads observes nothing). A shared definition is planned once, with its observed nodes, whatever
/// reads it; [`Self::strip_dead`] then removes the nodes of the plans no live read reaches.
#[derive(Default)]
pub(super) struct SharedReads {
    /// The reads of the plan itself: the plan read and whether the read is live (outside a
    /// `LIMIT 0`).
    root: Vec<(SharedId, bool)>,
    /// The reads made so far by each definition being planned, innermost last.
    open: Vec<Vec<(SharedId, bool)>>,
    /// The reads of each planned definition's body.
    of: HashMap<SharedId, Vec<(SharedId, bool)>>,
}

impl SharedReads {
    /// A definition's body starts planning.
    pub(super) fn begin(&mut self) {
        self.open.push(Vec::new());
    }

    /// The body that started last is planned: the reads it made, to [`Self::record`] under its
    /// shared plan's number.
    pub(super) fn finish(&mut self) -> Vec<(SharedId, bool)> {
        self.open.pop().unwrap_or_default()
    }

    /// The reads of the body of shared plan `id`.
    pub(super) fn record(&mut self, id: SharedId, reads: Vec<(SharedId, bool)>) {
        self.of.insert(id, reads);
    }

    /// The plan being planned reads shared plan `id`, `live` unless under a `LIMIT 0`.
    pub(super) fn read(&mut self, id: SharedId, live: bool) {
        match self.open.last_mut() {
            Some(reads) => reads.push((id, live)),
            None => self.root.push((id, live)),
        }
    }

    /// Removes the observing nodes of every shared plan in `shared` that no live read reaches: a
    /// live read of a plan makes live the live reads its body makes.
    pub(super) fn strip_dead(&self, shared: &mut [SharedPlan]) {
        let mut live: HashSet<SharedId> = HashSet::new();
        let mut work: Vec<SharedId> = self
            .root
            .iter()
            .filter(|(_, l)| *l)
            .map(|(id, _)| *id)
            .collect();
        while let Some(id) = work.pop() {
            if live.insert(id) {
                work.extend(
                    self.of
                        .get(&id)
                        .into_iter()
                        .flatten()
                        .filter(|(_, l)| *l)
                        .map(|(id, _)| *id),
                );
            }
        }
        for plan in shared.iter_mut().filter(|p| !live.contains(&p.id)) {
            strip_observed(&mut plan.plan);
        }
    }
}

/// `node` without its [`PhysNode::CollectMetrics`] nodes: their input in their place. The walk
/// follows `children_mut`, which reaches every node that can hold one (the subquery plans of Filter,
/// Project, Expand, ReplicateRows and Generate included); a node type that held a plan elsewhere
/// would have to be added there.
fn strip_observed(node: &mut PhysNode) {
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        while let PhysNode::CollectMetrics { input, .. } = n {
            let inner = std::mem::replace(&mut **input, PhysNode::OneRow);
            *n = inner;
        }
        stack.extend(n.children_mut());
    }
}

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
