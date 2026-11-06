// SPDX-License-Identifier: Apache-2.0
//! The nested-loop join's build sink: collects the build rows, spills them
//! under memory pressure, and at finalize numbers them for the build-match
//! bitmap ([`Ready`]).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::{Reservation, SpillFile, SpillScope, SpillSignal};
use krishiv_types::batch::Batch;

use super::{Marks, Shared, unit_weights};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus};
use crate::operators::{global_ref, local_mut, owned_rows};
use crate::output_spill::refused;
use crate::sync::{Mutex, lock};

/// A spill writer's working memory, lent from the rows it writes: more than
/// one bounded slice needs ([`SPILL_SCRATCH`](crate::join::grace::SPILL_SCRATCH)).
const SCRATCH_BYTES: usize = 1 << 20;
const _: () = assert!(SCRATCH_BYTES >= crate::join::grace::SPILL_SCRATCH);

/// A run of build batches: in memory, or in one spill file.
enum Segment {
    Memory {
        batches: Vec<RecordBatch>,
        reservation: Reservation,
    },
    File(Arc<SpillFile>),
}

/// The build's shared state.
pub(super) struct BuildGlobal {
    collect: Mutex<Vec<Segment>>,
    ready: OnceLock<Arc<Ready>>,
}

/// The finalized build. Rows are numbered for [`Marks`]: the in-memory
/// batches first, then each file's batches in file order.
pub(super) struct Ready {
    /// The build batches held in memory.
    pub(super) memory: Arc<[RecordBatch]>,
    /// The first row number of each `memory` batch.
    pub(super) memory_bases: Arc<[u64]>,
    /// The spilled batches: each file with the number of its first row.
    pub(super) files: Vec<(Arc<SpillFile>, u64)>,
    /// The build-match bitmap of a join that returns build rows.
    pub(super) marks: Option<Marks>,
    /// The next `memory` batch a flushing driver emits build rows of.
    pub(super) claim: AtomicUsize,
    /// Probe drivers whose input ended, and those done with every chunk
    /// (a spilled build: the last one emits the build rows).
    pub(super) drivers: AtomicUsize,
    pub(super) finished: AtomicUsize,
    /// The local half of a split join published the bitmap (a build in
    /// memory: the first driver to flush does).
    pub(super) published: AtomicBool,
    /// What `memory` and `marks` are charged to.
    _memory: Vec<Reservation>,
}

impl Ready {
    /// Whether part of the build is on disk (a block nested loop).
    pub(super) fn spilled(&self) -> bool {
        !self.files.is_empty()
    }
}

/// The finalized build of the probe's dependency.
pub(super) fn ready(global: &GlobalState) -> Result<Arc<Ready>> {
    global_ref::<BuildGlobal>(global, "NestedLoopJoinBuild")?
        .ready
        .get()
        .cloned()
        .ok_or_else(|| internal("nested-loop probe before its build finalized".to_owned()))
}

/// Writes `batches` to a new spill file in `scope`, lending the writer's
/// working memory from `reservation`, which holds them: a full pool cannot
/// grant it afresh. Another driver's reservation (a segment combined at
/// finalize) is first re-charged to this driver's consumer, whose credit the
/// writer draws. Batches are written in slices of a bounded size, so the
/// writer's scratch stays within the loan whatever their width. On success
/// the batches' bytes are released and the file returned; on failure nothing
/// was freed and the reservation holds what it held before.
pub(super) async fn write(
    ctx: &TaskContext,
    scope: &SpillScope,
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    reservation: &mut Reservation,
) -> Result<SpillFile> {
    if reservation.consumer() != ctx.consumer().id() {
        reservation
            .transfer_to(ctx.consumer().consumer())
            .map_err(KrishivError::from)?;
    }
    let lent = reservation.size().min(SCRATCH_BYTES);
    reservation.shrink_to_credit(lent);
    // The scratch returns to credit when dropped, so a refused write can
    // always take back what was lent.
    let scratch = ctx
        .consumer()
        .reservation()
        .for_spill_scratch()
        .releasing_to_credit();
    let restore = |reservation: &mut Reservation| {
        // Nothing was freed: the lent bytes go back, from credit only.
        let _ = reservation.try_grow(lent);
        ctx.consumer().consumer().release_credit();
    };
    let writer = match scope.writer(schema, scratch) {
        Ok(w) => w,
        Err(e) => {
            restore(reservation);
            return Err(KrishivError::from(e));
        }
    };
    let query = Arc::clone(ctx.query());
    let written = ctx
        .run_blocking(move || {
            let mut writer = writer;
            for b in batches {
                query.check_cancelled()?;
                crate::join::grace::write_sliced(&mut writer, &Batch::new(b))?;
            }
            writer.finish().map_err(KrishivError::from)
        })
        .await;
    match written {
        Ok(file) => {
            ctx.consumer().consumer().release_credit();
            let rest = reservation.size();
            reservation.shrink(rest);
            Ok(file)
        }
        Err(e) => {
            restore(reservation);
            Err(e)
        }
    }
}

pub(super) struct BuildSink(pub(super) Arc<Shared>);

struct BuildLocal {
    segments: Vec<Segment>,
    batches: Vec<RecordBatch>,
    reservation: Reservation,
    signal: Arc<SpillSignal>,
}

impl Sink for BuildSink {
    fn name(&self) -> &'static str {
        "NestedLoopJoinBuild"
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(BuildGlobal {
            collect: Mutex::new(Vec::new()),
            ready: OnceLock::new(),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        let memory = ctx.consumer();
        Ok(Box::new(BuildLocal {
            segments: Vec::new(),
            batches: Vec::new(),
            reservation: memory.reservation(),
            signal: Arc::clone(memory.signal()),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        if input.num_rows() == 0 {
            return Ok(SinkStatus::NeedMoreInput);
        }
        unit_weights(&input)?;
        let l = local_mut::<BuildLocal>(local, "NestedLoopJoinBuild")?;
        // A scan's morsel can be a slice of a much larger batch: held as is,
        // every driver would keep and be charged the whole buffers.
        let input = owned_rows(input)?;
        let bytes = input.data().get_array_memory_size();
        if l.reservation.try_grow(bytes).is_err() {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        l.batches.push(input.into_parts().0);
        if self.0.spec.spill.is_some() {
            l.signal.set_spillable_bytes(l.reservation.size());
        }
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let l = local
            .downcast::<BuildLocal>()
            .map_err(|_| internal("nested-loop build state".to_owned()))?;
        let BuildLocal {
            segments,
            batches,
            reservation,
            signal,
        } = *l;
        signal.set_spillable_bytes(0);
        let g = global_ref::<BuildGlobal>(global, "NestedLoopJoinBuild")?;
        let mut collect = lock(&g.collect);
        collect.extend(segments);
        if !batches.is_empty() {
            collect.push(Segment::Memory {
                batches,
                reservation,
            });
        }
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global_ref::<BuildGlobal>(global, "NestedLoopJoinBuild")?;
            let mut segments = std::mem::take(&mut *lock(&g.collect));
            // Once part of the build is on disk, so is the rest: batches kept
            // in memory would stay reserved through the whole flush phase,
            // beside every driver's chunk and spill readers.
            if let Some(scope) = &self.0.spec.spill
                && segments.iter().any(|s| matches!(s, Segment::File(_)))
            {
                for s in &mut segments {
                    if let Segment::Memory {
                        batches,
                        reservation,
                    } = s
                    {
                        let schema = Arc::clone(&self.0.spec.build_schema);
                        let batches = std::mem::take(batches);
                        let file = write(ctx, scope, schema, batches, reservation).await?;
                        self.0.counters.build_spills.fetch_add(1, Ordering::Relaxed);
                        *s = Segment::File(Arc::new(file));
                    }
                }
            }
            let mut memory = Vec::new();
            let mut reservations = Vec::new();
            let mut files = Vec::new();
            for s in segments {
                match s {
                    Segment::Memory {
                        batches,
                        reservation,
                    } => {
                        memory.extend(batches);
                        reservations.push(reservation);
                    }
                    Segment::File(f) => files.push(f),
                }
            }
            let mut rows = 0u64;
            let memory_bases: Vec<u64> = memory
                .iter()
                .map(|b| {
                    let base = rows;
                    rows += b.num_rows() as u64;
                    base
                })
                .collect();
            let files: Vec<(Arc<SpillFile>, u64)> = files
                .into_iter()
                .map(|f| {
                    let base = rows;
                    rows += f.num_rows();
                    (f, base)
                })
                .collect();
            let marks = if self.0.join_type().builds_preserved() {
                let mut r = ctx.consumer().reservation();
                r.grow(Marks::bytes(rows))
                    .await
                    .map_err(KrishivError::from)?;
                reservations.push(r);
                Some(Marks::new(rows))
            } else {
                None
            };
            let ready = Arc::new(Ready {
                memory: memory.into(),
                memory_bases: memory_bases.into(),
                files,
                marks,
                claim: AtomicUsize::new(0),
                drivers: AtomicUsize::new(0),
                finished: AtomicUsize::new(0),
                published: AtomicBool::new(false),
                _memory: reservations,
            });
            g.ready
                .set(ready)
                .map_err(|_| internal("a nested-loop build finalized twice".to_owned()))
        })
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        _target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let Some(scope) = &self.0.spec.spill else {
                return Ok(0);
            };
            let l = local_mut::<BuildLocal>(local, "NestedLoopJoinBuild")?;
            if l.batches.is_empty() {
                return Ok(0);
            }
            let freed = l.reservation.size();
            // The batches stay in the local state until the file is complete.
            let batches = l.batches.clone();
            let schema = Arc::clone(&self.0.spec.build_schema);
            let file = match write(ctx, scope, schema, batches, &mut l.reservation).await {
                Ok(f) => f,
                Err(e) => return refused(e),
            };
            l.batches.clear();
            l.signal.set_spillable_bytes(0);
            l.segments.push(Segment::File(Arc::new(file)));
            self.0.counters.build_spills.fetch_add(1, Ordering::Relaxed);
            Ok(freed)
        })
    }
}
