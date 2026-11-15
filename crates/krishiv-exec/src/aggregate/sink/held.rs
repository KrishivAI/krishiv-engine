// SPDX-License-Identifier: Apache-2.0
//! Reservation helpers of the aggregate's breaker: growing a table's
//! reservation to what it needs, handing bytes between reservations, and
//! the spill headroom lent to a writer and taken back.

use krishiv_memory::Reservation;

use super::HEADROOM;
use crate::aggregate::table::Table;
use crate::context::TaskContext;

/// Grows `r` to `want` bytes; `Err(missing)` if the pool refuses.
pub(super) fn reserve_to(r: &mut Reservation, want: usize) -> Result<(), usize> {
    let need = want.saturating_sub(r.size());
    if need > 0 && r.try_grow(need).is_err() {
        return Err(need);
    }
    Ok(())
}

/// Up to `bytes` taken from the end of `from`: whole reservations, then a
/// split of the next one.
pub(super) fn take_bytes(from: &mut Vec<Reservation>, mut bytes: usize) -> Vec<Reservation> {
    let mut out = Vec::new();
    while bytes > 0 {
        let Some(r) = from.last_mut() else { break };
        if r.size() <= bytes {
            bytes -= r.size();
            out.extend(from.pop());
        } else {
            out.push(r.split(bytes));
            bytes = 0;
        }
    }
    out
}

pub(super) fn shrink_to(r: &mut Reservation, keep: usize) {
    r.shrink(r.size().saturating_sub(keep));
}

/// Moves the combined tables' bytes to the spilling driver's consumer
/// (from the combined tables' own consumer, or with no spill scope another
/// driver's, without passing through the pool) and marks that consumer as writing until the returned
/// guard drops: until the tables are freed, their bytes count as about to
/// be released, so another driver's round that cannot spill waits for them
/// instead of failing (`ConsumerHandle::spilling`). Take it only for a
/// write with no arbitration inside (a blocking job). A table whose
/// transfer is refused keeps its consumer and is freed all the same.
pub(super) fn charged_to_writer(
    tables: &mut [(u32, Table, Reservation)],
    ctx: &TaskContext,
) -> krishiv_memory::SpillWriting {
    let own = ctx.consumer().consumer();
    for (_, _, r) in tables.iter_mut() {
        if r.consumer() != own.id() {
            let _ = r.transfer_to(own);
        }
    }
    own.spilling()
}

/// Releases the headroom of `r` for a spill file's scratch memory (see
/// [`HEADROOM`]); the writer or reader of the same consumer reserves it
/// right after.
pub(super) fn lend_headroom(r: &mut Reservation) -> usize {
    let lent = HEADROOM.min(r.size());
    r.shrink(lent);
    lent
}

/// [`lend_headroom`] of only what `r` holds above `used` bytes (0 once a
/// refused write could not take the headroom back): never bytes the table
/// still occupies.
pub(super) fn lend_headroom_above(r: &mut Reservation, used: usize) -> usize {
    let lent = HEADROOM.min(r.size().saturating_sub(used));
    r.shrink(lent);
    lent
}

/// Takes back `lent` bytes of headroom after spill I/O, if the pool still
/// has them (the next `reserve_to` asks again otherwise).
pub(super) fn regrow(r: &mut Reservation, lent: usize) {
    if r.try_grow(lent).is_err() {
        // Not fatal: the headroom is requested again with the next batch.
    }
}
