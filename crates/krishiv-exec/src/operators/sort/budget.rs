// SPDX-License-Identifier: Apache-2.0
//! The memory a sort's merges and spills reserve: [`merge_budget`] (with
//! the intermediate merge's fan-in fallback, and the wait in the query's
//! [`MERGE_COHORT`]), [`merge_need`], a spill's [`headroom`], and lending a
//! reservation to a merge or write ([`lending`], [`lending_for_write`]).

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use krishiv_memory::{ConsumerHandle, MemoryError, Reservation, Seat, SeatState};

use super::merge::SpillRun;
use super::{BATCH_SLACK_BYTES, SPILL_BATCH_BYTES, SPILL_BUFFER_BYTES};
use crate::cohort::reserve_in_cohort;
use crate::error::Result;

/// The cohort of a query's sort merge budgets: memory no spill takes back
/// but that a merge frees on its own once it ends. Two sorts of one query
/// (two tasks) each size a merge against the whole pool, so the second's
/// budget can find the first's holding most of it: it waits for that merge
/// to end instead of failing or aborting the query. A merge join's bucket
/// readers stay out: one driver holds both sides' merges at once.
pub(super) const MERGE_COHORT: &str = "sort merges";

/// A seat's state reset to idle if the request is dropped while it asks.
struct Asking<'a>(Option<&'a mut Seat>);

impl Drop for Asking<'_> {
    fn drop(&mut self) {
        if let Some(seat) = &mut self.0
            && seat.state() == SeatState::Reserving
        {
            seat.set(SeatState::Idle);
        }
    }
}

/// Reserves (with arbitration) the memory to merge up to `max_fan_in` of
/// `runs` at once. An intermediate merge (`writing` a run) halves the
/// fan-in down to 2 while the pool refuses and also reserves the spill
/// writer's scratch; the final merge needs every run. While a smaller
/// fan-in is left, a refusal must not abort the query: those requests
/// spill other consumers but never escalate to an abort.
///
/// With a `seat` (idle, in [`MERGE_COHORT`]) the seat is reserving while
/// it asks and holding once granted; the last request (fan-in 2, or the
/// final merge's) waits while another member is active
/// ([`reserve_in_cohort`], counting waits in the seat's counter). The
/// caller sets the seat idle after dropping the budget.
pub(super) async fn merge_budget(
    runs: &[SpillRun],
    max_fan_in: usize,
    batch_rows: usize,
    handle: &Arc<ConsumerHandle>,
    writing: bool,
    seat: Option<(&mut Seat, &AtomicU64)>,
) -> Result<(Reservation, usize)> {
    let mut fan_in = max_fan_in.min(runs.len());
    let mut budget = handle.reservation();
    let (seat, waits) = seat.map_or((None, None), |(s, w)| (Some(s), Some(w)));
    let mut asking = Asking(seat);
    if writing && fan_in > 2 {
        if let Some(seat) = &mut asking.0 {
            seat.set(SeatState::Reserving);
        }
        if wider(runs, &mut fan_in, batch_rows, &mut budget).await? {
            if let Some(seat) = &mut asking.0 {
                seat.set(SeatState::Holding);
            }
            return Ok((budget, fan_in));
        }
        // Back off without waking the cohort (the last request asks again).
        if let Some(seat) = &mut asking.0 {
            seat.stand_by();
        }
    }
    let seat = asking.0.take().zip(waits);
    let need = merge_need(runs, fan_in, batch_rows, writing);
    reserve_in_cohort(&mut budget, need, seat, SeatState::Holding, MERGE_COHORT).await?;
    Ok((budget, fan_in))
}

/// An intermediate merge's budget at a fan-in above 2: `true` once one is
/// granted (`fan_in` is then its fan-in), `false` with `fan_in` at 2.
async fn wider(
    runs: &[SpillRun],
    fan_in: &mut usize,
    batch_rows: usize,
    budget: &mut Reservation,
) -> Result<bool> {
    match budget
        .grow_without_abort(merge_need(runs, *fan_in, batch_rows, true))
        .await
    {
        Ok(()) => return Ok(true),
        // That round spilled what it could: smaller fan-ins only take
        // what is free now, and arbitration runs again at 2. A refusal
        // by the resident-set guard (not spill exhaustion) also lands
        // here, so it too is only re-arbitrated at fan-in 2: correct,
        // only slower.
        Err(MemoryError::Exhausted { .. }) => {}
        // An abort (this query was chosen by another request) ends the merge.
        Err(e) => return Err(e.into()),
    }
    while *fan_in > 2 {
        *fan_in = (*fan_in / 2).max(2);
        if *fan_in > 2
            && budget
                .try_grow(merge_need(runs, *fan_in, batch_rows, true))
                .is_ok()
        {
            return Ok(true);
        }
    }
    Ok(false)
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

/// A seat holding a merge budget in the query's [`MERGE_COHORT`] (none
/// outside a query).
pub(super) fn holding(handle: &ConsumerHandle) -> Option<Seat> {
    let mut seat = handle.pool().cohort(MERGE_COHORT)?.join();
    seat.set(SeatState::Holding);
    Some(seat)
}

/// Lends `res` to the next growths of its consumer (as grant credit) while
/// `work` runs, so the scratch memory of a spill or merge is never taken by
/// another consumer; then takes the credit back and tops `res` up to
/// `target` if the pool allows (best effort: a later reservation asks for
/// the rest).
pub(super) async fn lending<T>(
    res: &mut Reservation,
    handle: &ConsumerHandle,
    target: usize,
    work: impl Future<Output = T>,
) -> T {
    let lent = res.size();
    res.shrink_to_credit(lent);
    let out = work.await;
    // Credit only: this never reaches the pool, so it cannot fail.
    let _ = res.try_grow(handle.credit());
    let _ = res.try_grow(target.saturating_sub(res.size()));
    out
}

/// [`lending`] for a spill write: takes back at most what it lent (or
/// `target`) and returns the rest of the consumer's credit to the pool. The
/// writer's scratch beyond the lent bytes is temporary, and may have come
/// from the pool's spill-scratch floor, which must not stay with one
/// consumer.
pub(super) async fn lending_for_write<T>(
    res: &mut Reservation,
    handle: &ConsumerHandle,
    target: usize,
    work: impl Future<Output = T>,
) -> T {
    let lent = res.size();
    res.shrink_to_credit(lent);
    let out = work.await;
    // Credit only: this never reaches the pool, so it cannot fail.
    let _ = res.try_grow(handle.credit().min(lent.max(target)));
    handle.release_credit();
    let _ = res.try_grow(target.saturating_sub(res.size()));
    out
}

/// Rows per spilled batch: about [`SPILL_BATCH_BYTES`], at most a batch.
pub(super) fn spill_rows(row_bytes: usize, batch_rows: usize) -> usize {
    (SPILL_BATCH_BYTES / row_bytes.max(1)).clamp(1, batch_rows.max(1))
}

/// Headroom to spill one buffer: one merged spill batch of `row_bytes`
/// data bytes per row (plus its buffers' alignment), the writer's scratch
/// (twice that) and its write buffer.
pub(super) fn headroom(row_bytes: usize, batch_rows: usize) -> usize {
    row_bytes
        .saturating_mul(spill_rows(row_bytes, batch_rows))
        .saturating_add(BATCH_SLACK_BYTES)
        .saturating_mul(3)
        .saturating_add(SPILL_BUFFER_BYTES)
}
