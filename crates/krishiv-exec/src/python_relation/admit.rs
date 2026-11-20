// SPDX-License-Identifier: Apache-2.0
//! A driver's admission (design P3-T085 §3.3a): the slots of a runtime whose
//! relation tasks are bounded ([`PreparedRelationUdf::capacity`]), taken for
//! every lane of the driver at once.
//!
//! A driver feeds its lanes together: while one lane's window is full no
//! lane gets a frame. A lane whose task waited for a slot while its peers
//! waited for frames would so hang the driver, and with more drivers than
//! slots, a driver holding some slots while it waits for the rest could
//! hang another doing the same. So a driver takes one slot per lane in one
//! step, before it opens any task, or holds none and waits (`Pending`, its
//! waker woken when slots free). A driver needing more lanes than the
//! runtime ever has slots fails at once with the runtime's classed error.
//!
//! [`PreparedRelationUdf::capacity`]: super::PreparedRelationUdf::capacity

use std::sync::Arc;
use std::task::{Context, Poll};

use super::{RelationLane, RelationSlot, RelationSlots, RelationTicket};
use crate::error::{Result, internal};

/// One slot per lane of `specs` whose call site is bounded (`None` for the
/// others), in lane order; `Pending` with no slot held while any group of
/// lanes sharing slots cannot get all of them. `tickets` are the driver's
/// places in the groups' queues, kept between polls (dropped with the
/// driver's state when it stops waiting). Arrival order holds for one group
/// only (review R4-6): with several, the groups before the one still waiting
/// give their grant back and ask again at the tail. A query has one group
/// today (one slot set per session).
pub(super) fn poll_admit(
    specs: &[RelationLane],
    tickets: &mut Vec<Option<RelationTicket>>,
    cx: &mut Context<'_>,
) -> Poll<Result<Vec<Option<RelationSlot>>>> {
    // Lanes grouped by the slots they share (in practice one group: a
    // query's UDFs are its session's).
    let mut groups: Vec<(Arc<dyn RelationSlots>, Vec<usize>)> = Vec::new();
    for (i, lane) in specs.iter().enumerate() {
        let Some(cap) = lane.udf.capacity() else {
            continue;
        };
        match groups
            .iter_mut()
            .find(|(g, _)| std::ptr::addr_eq(Arc::as_ptr(g), Arc::as_ptr(&cap)))
        {
            Some((_, lanes)) => lanes.push(i),
            None => groups.push((cap, vec![i])),
        }
    }
    let mut slots: Vec<Option<RelationSlot>> = specs.iter().map(|_| None).collect();
    if tickets.len() < groups.len() {
        tickets.resize_with(groups.len(), || None);
    }
    for ((cap, lanes), ticket) in groups.iter().zip(tickets.iter_mut()) {
        // On `Pending` or an error the slots taken so far drop here: none is
        // held while the driver waits.
        let taken = match cap.poll_take(lanes.len(), ticket, cx) {
            Poll::Ready(Ok(taken)) => taken,
            Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
            Poll::Pending => return Poll::Pending,
        };
        if taken.len() != lanes.len() {
            return Poll::Ready(Err(internal(format!(
                "a Python UDF runtime granted {} slots for {} tasks",
                taken.len(),
                lanes.len()
            ))));
        }
        for (&lane, slot) in lanes.iter().zip(taken) {
            if let Some(s) = slots.get_mut(lane) {
                *s = Some(slot);
            }
        }
    }
    tickets.clear();
    Poll::Ready(Ok(slots))
}
