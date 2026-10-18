// SPDX-License-Identifier: Apache-2.0
//! The streaming first phase of a two-phase aggregation: a flushable
//! partial aggregate with adaptive bypass (after Velox's flushable partial
//! aggregation).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_schema::SchemaRef;
use krishiv_memory::{ConsumerHandle, Reservation};
use krishiv_types::batch::Batch;

use super::table::{AggCore, AggTable};
use super::{AggregateOutput, AggregateSpec};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::hash::HashSeed;
use crate::operator::{LocalState, Operator, OperatorOutput};

/// When a [`PartialAggregate`] gives up aggregating.
///
/// After a driver has seen at least [`min_rows`](Self::min_rows) input rows,
/// if its table holds at least [`max_group_percent`](Self::max_group_percent)
/// groups per 100 input rows (it reduces the data by less than
/// `100 − max_group_percent` percent), it flushes the table and passes every
/// later row through as its own partial state: the final phase does all the
/// work and the partial phase stops costing a hash probe per row. The
/// defaults (100,000 rows, 80%) are Velox's
/// `abandon_partial_aggregation_min_rows` / `_min_pct`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BypassPolicy {
    /// Input rows a driver sees before it may switch to pass-through.
    pub min_rows: u64,
    /// Groups per 100 input rows at or above which it switches.
    pub max_group_percent: u64,
    /// Table bytes at which the table is flushed downstream instead of
    /// growing (flushable partial aggregation; never spills). Each driver's
    /// table is also capped at 1/8 of the query pool's limit divided by the
    /// worker slots, so partial tables never hold the memory the final
    /// phase needs (they cannot be asked to spill).
    pub max_table_bytes: usize,
}

impl Default for BypassPolicy {
    fn default() -> Self {
        Self {
            min_rows: 100_000,
            max_group_percent: 80,
            max_table_bytes: 16 << 20,
        }
    }
}

/// The first phase of a two-phase aggregation: aggregates raw rows into
/// partial states per driver and streams them out (to an exchange, or to a
/// [`HashAggregate`](super::HashAggregate) that merges them). It flushes
/// when its table is full or memory is short, never spills, and bypasses
/// itself when it does not reduce its input ([`BypassPolicy`]).
#[derive(Debug)]
pub struct PartialAggregate {
    core: Arc<AggCore>,
    policy: BypassPolicy,
    seed: HashSeed,
    flushes: AtomicU64,
    bypassed_drivers: AtomicU64,
    bypassed_rows: AtomicU64,
}

struct Local {
    table: AggTable,
    reservation: Reservation,
    consumer: Arc<ConsumerHandle>,
    rows: u64,
    bypass: bool,
    /// `finish` has started.
    finished: bool,
    /// A full table being emitted, and the next group to emit.
    draining: Option<(AggTable, Reservation, usize)>,
    batch_size: usize,
    /// This driver's table cap (see [`BypassPolicy::max_table_bytes`]).
    cap: usize,
}

impl PartialAggregate {
    /// A partial aggregate; `spec.output` must be
    /// [`AggregateOutput::Partial`]. `seed` is the query's
    /// `ExecConfig::seed`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a final-output spec; typing errors.
    pub fn new(spec: AggregateSpec, policy: BypassPolicy, seed: Option<u64>) -> Result<Arc<Self>> {
        if spec.output != AggregateOutput::Partial {
            return Err(internal(
                "a partial aggregate emits partial states".to_owned(),
            ));
        }
        Ok(Arc::new(Self {
            core: Arc::new(AggCore::new(spec)?),
            policy,
            seed: HashSeed::new(seed),
            flushes: AtomicU64::new(0),
            bypassed_drivers: AtomicU64::new(0),
            bypassed_rows: AtomicU64::new(0),
        }))
    }

    /// Drivers that switched to pass-through.
    pub fn bypassed_drivers(&self) -> u64 {
        self.bypassed_drivers.load(Ordering::Relaxed)
    }

    /// Moves the table out to be emitted chunk by chunk.
    fn start_drain(&self, l: &mut Local) -> Result<()> {
        let table = std::mem::replace(&mut l.table, AggTable::new(&self.core, &self.seed)?);
        let reservation = std::mem::replace(&mut l.reservation, l.consumer.reservation());
        l.draining = Some((table, reservation, 0));
        // The reduction is judged per table: count rows from this flush on.
        l.rows = 0;
        self.flushes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// The next chunk of the table being emitted, if any.
    fn drain_one(l: &mut Local) -> Result<Option<Batch>> {
        let Some((table, _, next)) = l.draining.as_mut() else {
            return Ok(None);
        };
        let (start, len) = (*next, table.len());
        let end = (start + l.batch_size).min(len);
        *next = end;
        let ids: Vec<u32> = (start..end)
            .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
            .collect();
        let out = (start < end).then(|| table.state_batch(&ids)).transpose()?;
        if end >= len {
            l.draining = None;
        }
        Ok(out.map(Batch::new))
    }
}

fn downcast(l: &mut LocalState) -> Result<&mut Local> {
    l.downcast_mut::<Local>()
        .ok_or_else(|| internal("wrong local state for PartialAggregate".to_owned()))
}

impl Operator for PartialAggregate {
    fn name(&self) -> &'static str {
        "PartialAggregate"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.core.state_schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        let consumer = Arc::clone(ctx.consumer().consumer());
        let share = ctx
            .query()
            .pool()
            .limit()
            .map_or(usize::MAX, |l| l / 8 / ctx.executor().parallelism().max(1));
        Ok(Box::new(Local {
            cap: self.policy.max_table_bytes.min(share),
            table: AggTable::new(&self.core, &self.seed)?,
            reservation: consumer.reservation(),
            consumer,
            rows: 0,
            bypass: false,
            finished: false,
            draining: None,
            batch_size: ctx.query().config().batch_size.max(1),
        }))
    }

    fn execute(&self, input: &Batch, l: &mut LocalState) -> Result<OperatorOutput> {
        let l = downcast(l)?;
        if let Some(b) = Self::drain_one(l)? {
            return Ok(OperatorOutput::HaveMoreOutput(b));
        }
        if self.core.spec.require_unit_weights && !super::sink::unit_weights(input.weights()) {
            return Err(super::unsupported(
                "DISTINCT over rows with weights other than 1 (needs incremental view maintenance)"
                    .to_owned(),
            ));
        }
        if l.bypass {
            self.bypassed_rows
                .fetch_add(input.num_rows() as u64, Ordering::Relaxed);
            let out = AggTable::states_of_rows(&self.core, input)?;
            return Ok(OperatorOutput::NeedMoreInput(Some(Batch::new(out))));
        }
        let p = l.table.prepare(input, false)?;
        let want = l.table.projected(&p);
        let need = want.saturating_sub(l.reservation.size());
        let fits = want <= l.cap && (need == 0 || l.reservation.try_grow(need).is_ok());
        if !fits {
            if !l.table.not_worth_spilling() {
                // Flush instead of growing; the input is retried after.
                self.start_drain(l)?;
                if let Some(b) = Self::drain_one(l)? {
                    return Ok(OperatorOutput::HaveMoreOutput(b));
                }
            }
            if need > 0 && l.reservation.try_grow(need).is_err() {
                return Ok(OperatorOutput::NeedMemory { bytes: need });
            }
        }
        l.table.apply(&p)?;
        let keep = l.table.memory_size();
        l.reservation
            .shrink(l.reservation.size().saturating_sub(keep));
        l.rows += input.num_rows() as u64;
        let groups = l.table.len() as u64;
        if l.rows >= self.policy.min_rows
            && groups.saturating_mul(100) >= l.rows.saturating_mul(self.policy.max_group_percent)
        {
            l.bypass = true;
            self.bypassed_drivers.fetch_add(1, Ordering::Relaxed);
            self.start_drain(l)?;
        }
        Ok(OperatorOutput::NeedMoreInput(None))
    }

    fn finish(&self, l: &mut LocalState) -> Result<Option<Batch>> {
        let l = downcast(l)?;
        if let Some(b) = Self::drain_one(l)? {
            return Ok(Some(b));
        }
        if !std::mem::replace(&mut l.finished, true) && l.table.len() > 0 {
            self.start_drain(l)?;
            return Self::drain_one(l);
        }
        Ok(None)
    }

    fn extra_metrics(&self) -> Vec<(&'static str, u64)> {
        vec![
            ("flushes", self.flushes.load(Ordering::Relaxed)),
            ("bypassed_drivers", self.bypassed_drivers()),
            ("bypassed_rows", self.bypassed_rows.load(Ordering::Relaxed)),
        ]
    }
}
