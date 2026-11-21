// SPDX-License-Identifier: Apache-2.0
//! `Dataset.tail(n)`: the last `n` rows of the input stream, in stream order.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivUnsupportedFeature;
use krishiv_memory::Reservation;
use krishiv_types::batch::Batch;

use super::local_mut;
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{LocalState, Operator, OperatorOutput};

/// Keeps the last `limit` rows of its input and emits them, in input order,
/// once the input ends ([`Operator::finish`]; Spark's `CollectTailExec`).
/// This is not a `Sort` and a `Limit`: the rows are the stream's own last
/// rows, so the operator is planned over one driver (the stream order is the
/// plan order) and nothing is emitted before the input ends.
///
/// **Memory.** The retained batches are reserved with the driver's consumer
/// before they are kept (`NeedMemory` when the pool refuses, the state
/// unchanged). A batch longer than `limit` is cut to its last `limit` rows
/// and copied, so the buffers of a long input are not pinned; the state
/// holds at most `limit` rows plus the part of one batch before them, and
/// never more than one input batch per call.
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

#[derive(Debug)]
struct TailLocal {
    kept: VecDeque<Kept>,
    rows: usize,
    reservation: Reservation,
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
}

impl Operator for Tail {
    fn name(&self) -> &'static str {
        "Tail"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(TailLocal {
            kept: VecDeque::new(),
            rows: 0,
            reservation: ctx.consumer().reservation(),
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        if !input.is_unweighted() {
            return Err(KrishivUnsupportedFeature {
                feature: "tail over weighted (change stream) input".to_owned(),
            }
            .into());
        }
        let keep = self.keep();
        let rows = input.num_rows();
        if keep == 0 {
            // Nothing is ever kept: the upstream need not be read further.
            return Ok(OperatorOutput::Finished(None));
        }
        if rows == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let l = local_mut::<TailLocal>(local, "Tail")?;
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
            return Ok(OperatorOutput::NeedMemory { bytes });
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
        Ok(OperatorOutput::NeedMoreInput(None))
    }

    fn finish(&self, local: &mut LocalState) -> Result<Option<Batch>> {
        let l = local_mut::<TailLocal>(local, "Tail")?;
        Ok(l.kept.pop_front().map(|k| {
            l.reservation.shrink(k.bytes);
            l.rows = l.rows.saturating_sub(k.batch.num_rows());
            k.batch
        }))
    }
}

#[cfg(test)]
mod tests;
