// SPDX-License-Identifier: Apache-2.0
//! The memory a merge of sorted spill runs reserves: [`merge_budget`] (with
//! the intermediate merge's fan-in fallback) and [`merge_need`].

use std::sync::Arc;

use krishiv_memory::{ConsumerHandle, MemoryError, Reservation};

use super::merge::SpillRun;
use super::sort::headroom;
use crate::error::Result;

/// Reserves (with arbitration) the memory to merge up to `max_fan_in` of
/// `runs` at once. An intermediate merge (`writing` a run) halves the
/// fan-in down to 2 while the pool refuses and also reserves the spill
/// writer's scratch; the final merge needs every run. While a smaller
/// fan-in is left, a refusal must not abort the query: those requests
/// spill other consumers but never escalate to an abort.
pub(super) async fn merge_budget(
    runs: &[SpillRun],
    max_fan_in: usize,
    batch_rows: usize,
    handle: &Arc<ConsumerHandle>,
    writing: bool,
) -> Result<(Reservation, usize)> {
    let mut fan_in = max_fan_in.min(runs.len());
    let mut budget = handle.reservation();
    if writing && fan_in > 2 {
        match budget
            .grow_without_abort(merge_need(runs, fan_in, batch_rows, writing))
            .await
        {
            Ok(()) => return Ok((budget, fan_in)),
            // That round spilled what it could: smaller fan-ins only take
            // what is free now, and arbitration runs again at 2. A refusal
            // by the resident-set guard (not spill exhaustion) also lands
            // here, so it too is only re-arbitrated at fan-in 2: correct,
            // only slower.
            Err(MemoryError::Exhausted { .. }) => {}
            // An abort (this query was chosen by another request) ends the merge.
            Err(e) => return Err(e.into()),
        }
        while fan_in > 2 {
            fan_in = (fan_in / 2).max(2);
            if fan_in > 2
                && budget
                    .try_grow(merge_need(runs, fan_in, batch_rows, writing))
                    .is_ok()
            {
                return Ok((budget, fan_in));
            }
        }
    }
    budget
        .grow(merge_need(runs, fan_in, batch_rows, writing))
        .await?;
    Ok((budget, fan_in))
}

/// The memory to merge `fan_in` of `runs` at once (see [`merge_budget`]).
pub(super) fn merge_need(
    runs: &[SpillRun],
    fan_in: usize,
    batch_rows: usize,
    writing: bool,
) -> usize {
    let per_run = runs.iter().map(SpillRun::merge_bytes).max().unwrap_or(0);
    let row_bytes = runs.iter().map(|r| r.row_bytes).max().unwrap_or(1);
    let out = if writing {
        headroom(row_bytes, batch_rows)
    } else {
        row_bytes.saturating_mul(batch_rows)
    };
    // The merger's per-source `slot`/`stamp` metadata (see
    // [`super::merge::metadata_bytes`]): small next to the frames, but
    // reserved with them so the merge never grows past its budget.
    let meta = super::merge::metadata_bytes(fan_in);
    per_run
        .saturating_mul(fan_in)
        .saturating_add(out)
        .saturating_add(meta)
}
