// SPDX-License-Identifier: Apache-2.0
//! `LIMIT n [OFFSET m]` as a streaming operator.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivUnsupportedFeature;
use krishiv_types::batch::Batch;

use super::no_state;
use crate::context::TaskContext;
use crate::error::Result;
use crate::operator::{LocalState, Operator, OperatorOutput};

/// Passes rows `offset..offset + limit` of the input stream and then
/// returns [`OperatorOutput::Finished`], which stops the driver from
/// reading its source. The row positions are counted over all drivers of
/// the pipeline, so with several drivers *which* rows pass depends on the
/// batch order (as in any unordered parallel `LIMIT`); the count is exact.
///
/// **Rows read.** Once the last row is passed, the driver that passed it
/// reads nothing more; each other driver finishes at most the batch it has
/// in flight. A single driver therefore reads at most
/// `offset + limit + one batch` rows. `LIMIT 0` reads one batch.
///
/// **Weights.** The operator defines `LIMIT` for insert-only input
/// (`Weights::Const(1)`, a batch query) and rejects weighted input with
/// `KRISHIV_UNSUPPORTED_FEATURE`: the first `n` rows of a change stream are
/// not a function of the delta (incremental top-N is a separate operator).
/// Sliced batches share buffers with the input.
#[derive(Debug)]
pub struct Limit {
    schema: SchemaRef,
    offset: u64,
    limit: u64,
    seen: AtomicU64,
}

impl Limit {
    /// The first `limit` rows of batches of `schema`.
    pub fn new(schema: SchemaRef, limit: u64) -> Arc<Self> {
        Self::with_offset(schema, 0, limit)
    }

    /// `limit` rows after skipping `offset`.
    pub fn with_offset(schema: SchemaRef, offset: u64, limit: u64) -> Arc<Self> {
        Arc::new(Self {
            schema,
            offset,
            limit,
            seen: AtomicU64::new(0),
        })
    }
}

impl Operator for Limit {
    fn name(&self) -> &'static str {
        "Limit"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(no_state())
    }

    fn execute(&self, input: &Batch, _local: &mut LocalState) -> Result<OperatorOutput> {
        if !input.is_unweighted() {
            return Err(KrishivUnsupportedFeature {
                feature: "LIMIT over weighted (change stream) input".to_owned(),
            }
            .into());
        }
        let rows = input.num_rows();
        let before = self.seen.fetch_add(rows as u64, Ordering::Relaxed);
        let end = self.offset.saturating_add(self.limit);
        if before >= end {
            return Ok(OperatorOutput::Finished(None));
        }
        // Positions of this batch are `before..before + rows`; keep those
        // in `offset..end` (bounded by `rows`, so they fit a usize).
        let to_local = |position: u64| {
            usize::try_from(position.saturating_sub(before)).map_or(rows, |p| p.min(rows))
        };
        let (start, stop) = (to_local(self.offset), to_local(end));
        let out = if start == 0 && stop == rows {
            Some(input.clone())
        } else if start < stop {
            Some(input.slice(start, stop - start)?)
        } else {
            None
        };
        let done = before.saturating_add(rows as u64) >= end;
        Ok(if done {
            OperatorOutput::Finished(out)
        } else {
            OperatorOutput::NeedMoreInput(out)
        })
    }
}
