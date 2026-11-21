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

use krishiv_common::error::ErrorClass;
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
use crate::operators::{PartitionState, Project, local_mut};

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

#[cfg(test)]
impl Finalizer {
    /// A `count(*)` observation named `name` over no columns, for tests of the partials'
    /// bookkeeping (which never finalize it).
    pub(crate) fn count_for_tests(name: &str) -> Result<Arc<Self>> {
        let spec = super::AggregateSpec {
            input: Arc::new(arrow_schema::Schema::empty()),
            group_by: Vec::new(),
            aggregates: vec![super::AggregateExpr::count_star("n")],
            output: AggregateOutput::Final,
            ansi: true,
            spill: None,
            require_unit_weights: false,
            legacy_statistical: false,
            udafs: None,
        };
        let core = Arc::new(AggCore::new(spec)?);
        let post = Project::try_new(&core.output_schema, Vec::new())?;
        Ok(Arc::new(Self {
            name: name.to_owned(),
            instance: 0,
            core,
            post,
        }))
    }
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

    /// The metrics' row over `partials`, the `(partition, states)` of every driver of the
    /// instance: the aggregates' zero state if there are none.
    ///
    /// Spark merges each task's buffer into the driver's in task-completion order, and a merge
    /// that overflows (`sum` of a long under ANSI) throws inside the scheduler, which logs it and
    /// drops *that task's whole buffer* (`DAGScheduler.updateAccumulators`); the action still
    /// succeeds. Here `partials` arrive in `(partition, lane)` order (deterministic where Spark's
    /// order is not) and a state whose merge raises `ARITHMETIC_OVERFLOW` is dropped the same way.
    /// Overflow in a metric's result (a decimal sum, `max(v) + 2`) raises here, at `get`.
    /// The merge table is charged to `reservation` as it grows.
    pub(crate) fn finalize(
        &self,
        partials: &[RecordBatch],
        reservation: &mut Reservation,
    ) -> Result<RecordBatch> {
        let mut kept: Vec<&RecordBatch> = Vec::new();
        let mut table = AggTable::new(&self.core, &HashSeed::new(None))?;
        for states in partials {
            match self.merge(&mut table, states, reservation) {
                Ok(()) => kept.push(states),
                Err(e) if e.error_class() == ErrorClass::ArithmeticOverflow => {
                    // The failed merge may have changed some of the buffer: rebuild it from
                    // the states that merged.
                    table = AggTable::new(&self.core, &HashSeed::new(None))?;
                    for k in &kept {
                        self.merge(&mut table, k, reservation)?;
                    }
                }
                Err(e) => return Err(e),
            }
        }
        let results = table.output_batch(&[0])?;
        let out = self.post.project(&Batch::new(results), 0)?;
        Ok(out.data().clone())
    }

    fn merge(
        &self,
        table: &mut AggTable,
        states: &RecordBatch,
        reservation: &mut Reservation,
    ) -> Result<()> {
        let p = table.prepare(&Batch::new(states.clone()), true)?;
        let want = table.projected(&p);
        reservation.try_grow(want.saturating_sub(reservation.size()))?;
        table.apply(&p)?;
        let over = table.memory_size().saturating_sub(reservation.size());
        reservation.try_grow(over)?;
        Ok(())
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
    /// The aggregate's plan and seed, to start the next split's table.
    core: Arc<AggCore>,
    seed: Option<u64>,
    /// The split (or, untagged, the driver's partition) the open `table` folds.
    open: Option<u32>,
    query: Arc<crate::context::QueryContext>,
    finalizer: Arc<Finalizer>,
    partition: u32,
    /// The partition index the metrics' row expressions see (`spark_partition_id()`): a batch's
    /// split tag over the numbering base, as a `Project` at the same place sees it.
    index: PartitionState,
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
        let last = self.table.state_batch(&[0])?;
        let key = (self.open.unwrap_or(self.partition), self.lane);
        // The states move to the query's own charge: release the driver's first.
        self.reservation.shrink(self.reservation.size());
        self.query.observe_partial(&self.finalizer, key, last)
    }

    /// Ends the open split's table when a batch of another split arrives: Spark folds a partition
    /// on its own, so which driver the morsels of two splits reached must not decide whether a
    /// fold overflows (it would fail the action instead of the merge dropping a buffer). The
    /// finished split's states go to the query at once, under its own source key, so they are
    /// charged to the query's pool and kept if the driver later fails.
    fn switch_to(&mut self, split: u32) -> Result<()> {
        match self.open {
            Some(open) if open != split => {
                let fresh = AggTable::new(&self.core, &HashSeed::new(self.seed))?;
                let done = std::mem::replace(&mut self.table, fresh);
                let states = done.state_batch(&[0])?;
                drop(done);
                // Release the driver's charge before the query takes the states', as `send` does:
                // the same bytes are never counted twice, so a pool that holds them once takes them.
                let keep = self.table.memory_size();
                self.reservation
                    .shrink(self.reservation.size().saturating_sub(keep));
                self.query
                    .observe_partial(&self.finalizer, (open, self.lane), states)?;
            }
            _ => {}
        }
        self.open = Some(split);
        Ok(())
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
            core: Arc::clone(&self.core),
            seed: ctx.query().config().seed,
            open: None,
            query: Arc::clone(ctx.query()),
            finalizer: Arc::clone(&self.finalizer),
            partition: ctx.partition(),
            index: PartitionState::of(ctx),
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
        // A retry after `NeedMemory` is the same batch: the split is already open.
        l.switch_to(input.split().unwrap_or(l.partition))?;
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
                args = pre.project(seen, l.index.index(input.split()))?;
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

#[cfg(test)]
mod tests {
    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use krishiv_memory::{MemoryConfig, MemoryPool, SpillOwner, TreePool};

    use super::*;
    use crate::aggregate::{AggregateExpr, AggregateFunction, AggregateSpec};
    use crate::{
        ExecConfig, ExecutorConfig, OperatorId, QueryContext, QueryId, TaskContext, TokioExecutor,
        WorkloadClass,
    };

    fn rows(n: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from_iter_values(0..n))]).unwrap()
    }

    fn collect_list(schema: &SchemaRef) -> Arc<CollectMetrics> {
        let spec = |input: &SchemaRef, aggregates, output| AggregateSpec {
            input: Arc::clone(input),
            group_by: Vec::new(),
            aggregates,
            output,
            ansi: true,
            spill: None,
            require_unit_weights: false,
            legacy_statistical: false,
            udafs: None,
        };
        let agg = AggregateExpr::new(
            AggregateFunction::CollectList,
            0,
            DataType::Int64,
            "l".to_owned(),
        );
        let merging = agg.merging(0);
        CollectMetrics::try_new(
            "m",
            0,
            None,
            Arc::clone(schema),
            None,
            spec(schema, vec![agg], AggregateOutput::Partial),
            |states| Ok(spec(states, vec![merging], AggregateOutput::Final)),
            |results| Project::try_new(results, Vec::new()),
        )
        .unwrap()
    }

    const POOL: usize = 10_000_000;

    /// One driver over a big split 0 and a one-row split 1. After the first split is folded,
    /// ballast leaves only `free` bytes of the pool (`None`: none taken). Returns the query and
    /// the pool's reservation once split 0 is folded.
    fn two_splits(free: Option<usize>) -> Result<(Arc<QueryContext>, usize)> {
        let root = TreePool::new(&MemoryConfig {
            limit: None,
            quantum: 1,
            spill_scratch_reserve: 0,
            ..MemoryConfig::default()
        });
        let query = QueryContext::new(
            QueryId(1),
            WorkloadClass::Batch,
            ExecConfig::default(),
            root.child("q", Some(POOL)),
        );
        let exec = TokioExecutor::new(&ExecutorConfig::default())?;
        let ctx = TaskContext::new(
            Arc::clone(&query),
            exec,
            SpillOwner::new(),
            OperatorId(0),
            "op",
        );
        let batch = rows(1);
        let op = collect_list(&batch.schema());
        let mut local = op.local_state(&ctx)?;
        let big = Batch::new(rows(20_000)).with_split(Some(0));
        let small = Batch::new(rows(1)).with_split(Some(1));
        let out = op.execute(&big, &mut local)?;
        assert!(
            matches!(out, OperatorOutput::NeedMoreInput(_)),
            "the first split folds"
        );
        let held = query.pool().reserved();
        let _ballast = free.map(|free| {
            let mut ballast =
                krishiv_memory::ConsumerHandle::register(Arc::clone(query.pool()), "ballast", None)
                    .reservation();
            ballast.try_grow(POOL - held - free).unwrap();
            ballast
        });
        let out = op.execute(&small, &mut local)?;
        assert!(
            matches!(out, OperatorOutput::NeedMoreInput(_)),
            "the second split folds"
        );
        op.finish(&mut local)?;
        drop(local);
        Ok((query, held))
    }

    /// A finished split's states move to the query's charge as the driver releases its own: a pool
    /// that holds them once (and the next split's small table) takes the switch, and the query
    /// holds one partial per split.
    #[test]
    fn a_split_switch_charges_the_states_once_and_keeps_both_partials() {
        let (unlimited, _) = two_splits(None).unwrap();
        let both = unlimited.pool().reserved();
        // The first split's open table is `held`, its states about `both`. The pool has room for
        // half the states beyond the table: the switch takes the states only because the driver
        // releases the table first, so charging them before the release (twice over) fails.
        let (query, _) = two_splits(Some(both / 2)).unwrap();
        // Both splits' states are the query's (the unlimited run's total), and they merge.
        assert_eq!(query.pool().reserved(), both);
        assert!(query.observed_each().get("m").unwrap().is_ok());
    }
}
