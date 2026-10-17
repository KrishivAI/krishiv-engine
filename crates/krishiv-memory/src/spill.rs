// SPDX-License-Identifier: Apache-2.0
//! The cooperative spill protocol (ADR 0007 update, spec 13 §3).
//!
//! The arbitrator never touches another task's operator state. It calls
//! [`Spillable::request_spill`], which only sets a flag and hands back a
//! [`SpillTicket`]. The owning driver services the request on its own task:
//! it spills, shrinks its reservations and completes the ticket.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use crate::sync::{AtomicUsize, Mutex, Ordering, lock};

/// A memory consumer that can give memory back on request: hash join, hash
/// aggregate, sort and window, and external consumers such as a UDF worker
/// pool. Every method is callable from any task and must not block; the
/// spill itself runs later, on the owner's task.
pub trait Spillable: Send + Sync {
    /// Bytes the owner could free by spilling: an atomic snapshot the owner
    /// maintains.
    fn spillable_bytes(&self) -> usize;
    /// Asks the owner to free at least `target` bytes. Non-blocking.
    /// Requests that arrive before the owner services the first coalesce:
    /// the largest target wins and every caller's ticket completes together.
    fn request_spill(&self, target: usize) -> SpillTicket;
    /// The pending target, if any. Drivers call this at every batch
    /// boundary, so it must be cheap (one atomic load for [`SpillSignal`]).
    fn spill_requested(&self) -> Option<usize>;
}

#[derive(Debug, Default)]
struct TicketState {
    freed: Option<usize>,
    wakers: Vec<Waker>,
}

/// Completion slot shared by every ticket of one request generation.
#[derive(Debug, Default)]
struct TicketCell {
    state: Mutex<TicketState>,
}

impl TicketCell {
    /// Called once per cell: [`SpillSignal::complete`] takes the cell out of
    /// the signal, and [`SpillTicket::ready`] completes a fresh one.
    fn complete(&self, freed: usize) {
        let wakers = {
            let mut s = lock(&self.state);
            s.freed = Some(freed);
            std::mem::take(&mut s.wakers)
        };
        for w in wakers {
            w.wake();
        }
    }
}

/// Resolves to the bytes the owner freed for a spill request. Dropping it
/// withdraws nothing (the owner still spills); it only stops waiting.
#[derive(Clone)]
pub struct SpillTicket {
    cell: Arc<TicketCell>,
}

impl SpillTicket {
    /// A ticket that is already complete, for consumers that free memory
    /// synchronously inside `request_spill` or have nothing to free.
    pub fn ready(freed: usize) -> Self {
        let cell = Arc::new(TicketCell::default());
        cell.complete(freed);
        Self { cell }
    }

    /// The bytes freed, if the owner has completed the request.
    pub fn try_get(&self) -> Option<usize> {
        lock(&self.cell.state).freed
    }
}

impl fmt::Debug for SpillTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillTicket")
            .field("freed", &self.try_get())
            .finish()
    }
}

impl Future for SpillTicket {
    type Output = usize;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<usize> {
        let mut s = lock(&self.cell.state);
        if let Some(freed) = s.freed {
            return Poll::Ready(freed);
        }
        if !s.wakers.iter().any(|w| w.will_wake(cx.waker())) {
            s.wakers.push(cx.waker().clone());
        }
        Poll::Pending
    }
}

#[derive(Debug, Default)]
struct Pending {
    target: usize,
    cell: Option<Arc<TicketCell>>,
}

/// A ready-made [`Spillable`] for operators: the owner keeps
/// [`set_spillable_bytes`](Self::set_spillable_bytes) current and calls
/// [`complete`](Self::complete) after spilling.
///
/// The pool holds it weakly: once the operator drops it, the consumer is
/// never asked again, and an outstanding request completes with 0 bytes.
#[derive(Debug, Default)]
pub struct SpillSignal {
    spillable: AtomicUsize,
    /// Mirror of `pending.target` for the lock-free check; 0 = no request.
    requested: AtomicUsize,
    pending: Mutex<Pending>,
}

impl SpillSignal {
    /// A signal with nothing spillable and no request.
    pub fn new() -> Self {
        Self::default()
    }

    /// Owner side: publishes how many bytes a spill could free.
    pub fn set_spillable_bytes(&self, bytes: usize) {
        self.spillable.store(bytes, Ordering::Release);
    }

    /// Owner side: the pending request is done (`freed` may be 0 if nothing
    /// could be spilled). Clears the request and completes every ticket of
    /// it; a request that arrives afterwards starts a new generation.
    pub fn complete(&self, freed: usize) {
        let cell = {
            let mut p = lock(&self.pending);
            p.target = 0;
            self.requested.store(0, Ordering::Release);
            p.cell.take()
        };
        if let Some(cell) = cell {
            cell.complete(freed);
        }
    }
}

impl Spillable for SpillSignal {
    fn spillable_bytes(&self) -> usize {
        self.spillable.load(Ordering::Acquire)
    }

    fn request_spill(&self, target: usize) -> SpillTicket {
        if target == 0 {
            return SpillTicket::ready(0);
        }
        let mut p = lock(&self.pending);
        p.target = p.target.max(target);
        self.requested.store(p.target, Ordering::Release);
        let cell = Arc::clone(p.cell.get_or_insert_with(Arc::default));
        SpillTicket { cell }
    }

    fn spill_requested(&self) -> Option<usize> {
        let target = self.requested.load(Ordering::Acquire);
        (target != 0).then_some(target)
    }
}

impl Drop for SpillSignal {
    fn drop(&mut self) {
        self.complete(0);
    }
}
