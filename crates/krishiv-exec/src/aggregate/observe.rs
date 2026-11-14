// SPDX-License-Identifier: Apache-2.0
//! `Dataset.observe` at run time ([`PhysNode::CollectMetrics`]).
//!
//! [`CollectMetrics`] is a pass-through operator: each driver folds the
//! batches it is handed into the partial states of the metrics' aggregates
//! (the table of a global partial aggregate: one group, O(1) state, weights
//! respected) and forwards every batch unchanged, so what is counted is the
//! rows that flowed through it, as Spark's `CollectMetricsExec` counts the
//! rows its consumer pulled (a limit above stops the pull; here at batch
//! granularity). A driver's states go to the query when its local state
//! ends, whether the input ended, a limit above stopped the pull or the
//! query failed (`Drop`), as the task metric of that partition; the query
//! merges the states of an observation's drivers and a [`Finalizer`] turns
//! them into one row (`QueryContext::observed`).
//!
//! Two instances of one name in a plan (a self-join of an observed
//! relation) do not add up: the last one's states are the observation's
//! (Spark's `CollectMetricsExec.collect` keeps the last). The rank of an
//! instance is plan data (`PhysNode::CollectMetrics::instance`, set by the
//! planner), so workers that each instantiate a part of the plan agree on it.
//!
//! A `LIMIT` above (through row-for-row operators) caps what is folded at the
//! rows it takes (`limit`, one budget for the instance's drivers): Spark
//! counts the rows its consumer pulled, not the batches a pull read. The
//! states are charged to the query's memory pool through the driver's
//! consumer, so a `collect_list` metric over a large input fails with
//! `UNABLE_TO_ACQUIRE_MEMORY` instead of growing without bound.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use krishiv_memory::Reservation;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_types::batch::Batch;

use super::AggregateOutput;
use super::table::{AggCore, AggTable};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::hash::HashSeed;
use crate::operator::{LocalState, Operator, OperatorOutput};
use crate::operators::{Project, local_mut};

/// Turns the partial states of one observation into its row.
#[derive(Debug)]
pub(crate) struct Finalizer {
    name: String,
    /// This instance's place among the plan's.
    instance: u32,
    /// Merges the states of the drivers and gives the aggregates' results.
    core: Arc<AggCore>,
    /// The metrics over those results, one column each.
    post: Arc<Project>,
}

impl Finalizer {
    /// The observation's name.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// The instance's rank in its plan.
    pub(crate) fn instance(&self) -> u32 {
        self.instance
    }

    /// The metrics' row over `partials`, the states of every driver of the
    /// instance: the aggregates' zero state if there are none.
    ///
    /// The merge table is charged to `reservation` as it grows.
    pub(crate) fn finalize(
        &self,
        partials: &[RecordBatch],
        reservation: &mut Reservation,
    ) -> Result<RecordBatch> {
        let mut table = AggTable::new(&self.core, &HashSeed::new(None))?;
        for states in partials {
            let p = table.prepare(&Batch::new(states.clone()), true)?;
            let want = table.projected(&p);
            reservation.try_grow(want.saturating_sub(reservation.size()))?;
            table.apply(&p)?;
            let over = table.memory_size().saturating_sub(reservation.size());
            reservation.try_grow(over)?;
        }
        let results = table.output_batch(&[0])?;
        let out = self.post.project(&Batch::new(results), 0)?;
        Ok(out.data().clone())
    }
}

/// The operator of a [`PhysNode::CollectMetrics`](crate::physical::PhysNode).
#[derive(Debug)]
pub(crate) struct CollectMetrics {
    /// The input's schema: rows pass through as they are.
    schema: SchemaRef,
    /// Computes the aggregates' arguments from a batch, if they are not
    /// plain input columns.
    pre: Option<Arc<Project>>,
    /// The partial aggregate (no keys).
    core: Arc<AggCore>,
    finalizer: Arc<Finalizer>,
    /// The rows a `LIMIT` above takes, if one does.
    limit: Option<u64>,
    /// Rows folded so far, over the instance's drivers (counts against `limit`).
    taken: AtomicU64,
}

impl CollectMetrics {
    /// The operator over rows of `schema`: `pre` computes the aggregates'
    /// arguments (`pre_schema`: its output, or `schema` without it),
    /// `partial` is the global aggregate's plan, `post` computes the
    /// metrics from its results.
    #[expect(
        clippy::too_many_arguments,
        reason = "the operator's plan data and builders"
    )]
    pub(crate) fn try_new(
        name: &str,
        instance: u32,
        limit: Option<u64>,
        schema: SchemaRef,
        pre: Option<Arc<Project>>,
        partial: super::AggregateSpec,
        final_spec: impl FnOnce(&SchemaRef) -> Result<super::AggregateSpec>,
        post: impl FnOnce(&SchemaRef) -> Result<Arc<Project>>,
    ) -> Result<Arc<Self>> {
        if partial.output != AggregateOutput::Partial || !partial.group_by.is_empty() {
            return Err(internal(
                "observed metrics are a global partial aggregate".to_owned(),
            ));
        }
        let core = Arc::new(AggCore::new(partial)?);
        let merge = Arc::new(AggCore::new(final_spec(&core.state_schema)?)?);
        let post = post(&merge.output_schema)?;
        let finalizer = Arc::new(Finalizer {
            name: name.to_owned(),
            instance,
            core: merge,
            post,
        });
        Ok(Arc::new(Self {
            schema,
            pre,
            core,
            finalizer,
            limit,
            taken: AtomicU64::new(0),
        }))
    }
}

/// One driver's partial states, sent to the query when it ends.
struct Local {
    table: AggTable,
    query: Arc<crate::context::QueryContext>,
    finalizer: Arc<Finalizer>,
    partition: u32,
    /// The driver's lane among the logical partitions' (0 when it is not one).
    lane: usize,
    sent: bool,
    /// The table's bytes, charged to the driver's consumer.
    reservation: Reservation,
    /// The rows of the input batch being folded, when a retry after `NeedMemory` must not
    /// claim them from the budget again.
    claimed: Option<usize>,
}

impl Local {
    /// Sends the states to the query, once.
    fn send(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.sent, true) {
            return Ok(());
        }
        let states = self.table.state_batch(&[0])?;
        // The states move to the query's own charge: release the driver's first.
        self.reservation.shrink(self.reservation.size());
        self.query
            .observe_partial(&self.finalizer, (self.partition, self.lane), states)
    }
}

impl Drop for Local {
    fn drop(&mut self) {
        // The input ended, a limit above stopped the pull, or the query
        // failed: the rows seen so far count either way.
        if let Err(e) = self.send() {
            self.query.fail(e);
        }
    }
}

impl CollectMetrics {
    /// How many of `rows` just read count: all, or what is left of the limit above's budget.
    fn claim(&self, rows: usize) -> usize {
        let Some(cap) = self.limit else {
            return rows;
        };
        let mut got = 0;
        // `try_update` retries the closure on contention, so `got` is the last attempt's.
        let _ = self
            .taken
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |taken| {
                got = usize::try_from(cap.saturating_sub(taken))
                    .unwrap_or(usize::MAX)
                    .min(rows);
                Some(taken.saturating_add(got as u64))
            });
        got
    }
}

impl Operator for CollectMetrics {
    fn lane_safe(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "CollectMetrics"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(Local {
            table: AggTable::new(&self.core, &HashSeed::new(ctx.query().config().seed))?,
            query: Arc::clone(ctx.query()),
            finalizer: Arc::clone(&self.finalizer),
            partition: ctx.partition(),
            lane: ctx.lane().map_or(0, |(lane, _)| lane),
            sent: false,
            reservation: ctx.consumer().reservation(),
            claimed: None,
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        let l = local_mut::<Local>(local, "CollectMetrics")?;
        if input.num_rows() == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let take = match l.claimed {
            Some(take) => take,
            None => self.claim(input.num_rows()),
        };
        if take == 0 {
            // A limit above has all the rows it takes: these are not counted.
            return Ok(OperatorOutput::NeedMoreInput(Some(input.clone())));
        }
        l.claimed = Some(take);
        let counted = if take < input.num_rows() {
            Some(input.slice(0, take)?)
        } else {
            None
        };
        let seen = counted.as_ref().unwrap_or(input);
        let args;
        let folded = match &self.pre {
            Some(pre) => {
                args = pre.project(seen, l.partition)?;
                &args
            }
            None => seen,
        };
        let p = l.table.prepare(folded, false)?;
        let want = l.table.projected(&p);
        let need = want.saturating_sub(l.reservation.size());
        if need > 0 && l.reservation.try_grow(need).is_err() {
            return Ok(OperatorOutput::NeedMemory { bytes: need });
        }
        l.table.apply(&p)?;
        l.claimed = None;
        // A fold that outgrew its projection is charged now.
        let keep = l.table.memory_size();
        let over = keep.saturating_sub(l.reservation.size());
        if over > 0 {
            l.reservation.try_grow(over)?;
        }
        l.reservation
            .shrink(l.reservation.size().saturating_sub(keep));
        Ok(OperatorOutput::NeedMoreInput(Some(input.clone())))
    }

    fn finish(&self, local: &mut LocalState) -> Result<Option<Batch>> {
        local_mut::<Local>(local, "CollectMetrics")?.send()?;
        Ok(None)
    }
}
