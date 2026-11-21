// SPDX-License-Identifier: Apache-2.0
//! `Dataset.tail(n)`: the last `n` rows of the input stream, in stream order.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivUnsupportedFeature;
use krishiv_memory::Reservation;
use krishiv_types::batch::Batch;

use super::{global_ref, local_mut};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::sync::{Mutex, lock};

/// Keeps the last `limit` rows of its input and emits them, in input order,
/// once the input ends (Spark's `CollectTailExec`). This is not a `Sort` and
/// a `Limit`: the rows are the stream's own last rows.
///
/// It is a pipeline breaker (sink and source), not a streaming operator: a
/// `UNION ALL` below fans its branches into several pipelines, and a
/// streaming operator would keep one tail per pipeline. The planner runs the
/// input pipelines with one driver each (`preserve_order`), so each driver
/// keeps its own last rows and `combine` appends them to the shared state in
/// the order the drivers end: the members of a fan-in run one after another,
/// in branch order, unless a branch needs a cross-driver flush (then the
/// order between branches is unspecified, as for any consumer of such a
/// union).
///
/// **Memory.** The retained batches are reserved with the driver's consumer
/// before they are kept (`NeedMemory` when the pool refuses, the state
/// unchanged). A batch longer than `limit` is cut to its last `limit` rows
/// and copied, so the buffers of a long input are not pinned; a driver's
/// state holds at most `limit` rows plus the part of one batch before them,
/// and the shared state at most `limit` rows per driver until the source is
/// done.
///
/// **Weights.** Defined for insert-only input; weighted (change stream)
/// input is rejected with `KRISHIV_UNSUPPORTED_FEATURE`.
#[derive(Debug)]
pub(crate) struct Tail {
    schema: SchemaRef,
    limit: u64,
}

/// A retained batch and the bytes reserved for it.
#[derive(Debug)]
struct Kept {
    batch: Batch,
    bytes: usize,
}

/// One driver's last rows.
#[derive(Debug)]
struct TailLocal {
    kept: VecDeque<Kept>,
    rows: usize,
    reservation: Reservation,
}

/// The last rows of the drivers combined so far, in combine order, with the
/// reservations that cover them.
#[derive(Debug, Default)]
struct TailGlobal(Mutex<Combined>);

#[derive(Debug, Default)]
struct Combined {
    /// Each batch with the index (in `reservations`) of the driver reservation that
    /// covers it and the bytes charged for it.
    batches: VecDeque<(Batch, usize, usize)>,
    rows: usize,
    reservations: Vec<Reservation>,
}

impl Tail {
    /// The last `limit` rows of batches of `schema`.
    pub(crate) fn new(schema: SchemaRef, limit: u64) -> Arc<Self> {
        Arc::new(Self { schema, limit })
    }

    /// `limit` as a row count; a bound past `usize` keeps every row.
    fn keep(&self) -> usize {
        usize::try_from(self.limit).unwrap_or(usize::MAX)
    }

    /// Adds `input` to a driver's rows; `Some(bytes)` when the pool refused
    /// the reservation (nothing changed).
    fn add(&self, input: &Batch, l: &mut TailLocal) -> Result<Option<usize>> {
        let keep = self.keep();
        let rows = input.num_rows();
        if rows == 0 {
            return Ok(None);
        }
        let batch = if rows > keep {
            // Copy the last rows (a `take`, not a zero-copy slice), so the kept
            // rows do not pin the long input's buffers.
            let indices =
                arrow_array::UInt64Array::from_iter_values((rows - keep) as u64..rows as u64);
            let copied = arrow_select::take::take_record_batch(input.data(), &indices)
                .map_err(|e| internal(format!("tail could not copy its rows: {e}")))?;
            Batch::new(copied)
        } else {
            input.clone()
        };
        let bytes = batch.data().get_array_memory_size();
        if l.reservation.try_grow(bytes).is_err() {
            return Ok(Some(bytes));
        }
        l.rows += batch.num_rows();
        l.kept.push_back(Kept { batch, bytes });
        // Drop the batches wholly before the last `keep` rows, then cut the
        // first kept one (zero-copy) to the rows still needed.
        while l.rows > keep {
            let excess = l.rows - keep;
            let Some(front) = l.kept.front_mut() else {
                break;
            };
            let front_rows = front.batch.num_rows();
            if front_rows <= excess {
                l.rows -= front_rows;
                l.reservation.shrink(front.bytes);
                l.kept.pop_front();
            } else {
                front.batch = front.batch.slice(excess, front_rows - excess)?;
                l.rows = keep;
            }
        }
        Ok(None)
    }
}

impl Sink for Tail {
    fn name(&self) -> &'static str {
        "Tail"
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(TailGlobal::default()))
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(TailLocal {
            kept: VecDeque::new(),
            rows: 0,
            reservation: ctx.consumer().reservation(),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        if !input.is_unweighted() {
            return Err(KrishivUnsupportedFeature {
                feature: "tail over weighted (change stream) input".to_owned(),
            }
            .into());
        }
        if self.keep() == 0 {
            // Nothing is ever kept: the upstream need not be read further.
            return Ok(SinkStatus::Finished);
        }
        let l = local_mut::<TailLocal>(local, "Tail")?;
        Ok(match self.add(&input, l)? {
            Some(bytes) => SinkStatus::NeedMemory { bytes },
            None => SinkStatus::NeedMoreInput,
        })
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = global_ref::<TailGlobal>(global, "Tail")?;
        let l = local
            .downcast::<TailLocal>()
            .map_err(|_| internal("wrong local state for Tail".to_owned()))?;
        let l = *l;
        let keep = self.keep();
        let mut shared = lock(&g.0);
        // The driver's reservation covers what it kept; a batch trimmed away
        // below gives its bytes back at once, so many union branches do not
        // hold memory for rows no longer kept.
        let owner = shared.reservations.len();
        shared.reservations.push(l.reservation);
        for k in l.kept {
            shared.rows += k.batch.num_rows();
            shared.batches.push_back((k.batch, owner, k.bytes));
        }
        while shared.rows > keep {
            let excess = shared.rows - keep;
            let Some(front) = shared.batches.front() else {
                break;
            };
            let front_rows = front.0.num_rows();
            if front_rows <= excess {
                shared.rows -= front_rows;
                if let Some((_, owner, bytes)) = shared.batches.pop_front() {
                    // A slice still retains its buffers, so it keeps the full
                    // charge; only a batch dropped whole is released.
                    shared.reservations[owner].shrink(bytes);
                }
            } else {
                let cut = front.0.slice(excess, front_rows - excess)?;
                shared.batches[0].0 = cut;
                shared.rows = keep;
            }
        }
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        _global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

/// The source side: the kept batches, in order.
#[derive(Debug, Default)]
struct TailSource {
    taken: Option<Combined>,
}

impl Source for Tail {
    fn name(&self) -> &'static str {
        "Tail"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let state = ctx
            .dependency()
            .cloned()
            .ok_or_else(|| internal("Tail read before it was built".to_owned()))?;
        if !state.is::<TailGlobal>() {
            return Err(internal("wrong global state for Tail".to_owned()));
        }
        Ok(state)
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(TailSource::default()))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        let next = (|| {
            let l = local_mut::<TailSource>(local, "Tail")?;
            if l.taken.is_none() {
                let g = global_ref::<TailGlobal>(global, "Tail")?;
                l.taken = Some(std::mem::take(&mut *lock(&g.0)));
            }
            let Some(c) = l.taken.as_mut() else {
                return Ok(None);
            };
            let out = c.batches.pop_front().map(|(b, _, _)| b);
            if out.is_none() {
                // Done: the reservations go with the rows.
                c.reservations.clear();
            }
            Ok(out)
        })();
        Box::pin(std::future::ready(next))
    }
}

#[cfg(test)]
mod tests;
