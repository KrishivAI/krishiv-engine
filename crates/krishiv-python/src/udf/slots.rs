// SPDX-License-Identifier: Apache-2.0
//! The relation task slots of the process's Python pool (design P3-T085 §3.3).
//!
//! A relation task holds one pool thread for its whole life, so a driver whose tasks outnumber the pool's
//! threads would wait forever for the last of them while the first ones wait for its frames. The pool
//! therefore has as many **slots** as threads, less one thread kept for scalar UDF jobs when it has two or
//! more. A driver takes the slots of all its lanes together, or waits for them in arrival order, holding
//! none; a driver that needs more than the pool has is refused at once with
//! `KRISHIV_PYTHON_WORKER_LIMIT` naming `krishiv.python.udf.threads`. A slot is freed when its task ends.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

use krishiv_common::error::{KrishivError, KrishivPythonWorkerLimit};
use krishiv_exec::python_relation::{RelationSlot, RelationSlots, RelationTicket};

/// The slot counter shared by the pool and its tickets and slots.
pub(crate) struct Slots {
    limit: usize,
    state: Mutex<State>,
}

struct State {
    free: usize,
    next_id: u64,
    /// The waiters in arrival order, with the waker of each one's last poll.
    queue: VecDeque<(u64, Waker)>,
}

impl Slots {
    pub(crate) fn new(limit: usize) -> Arc<Self> {
        let limit = limit.max(1);
        Arc::new(Self {
            limit,
            state: Mutex::new(State {
                free: limit,
                next_id: 0,
                queue: VecDeque::new(),
            }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A waiter's place in the queue; dropping it leaves the queue and passes the turn on.
struct Ticket {
    slots: Arc<Slots>,
    id: u64,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let mut state = self.slots.lock();
        state.queue.retain(|(id, _)| *id != self.id);
        if let Some((_, waker)) = state.queue.front() {
            waker.wake_by_ref();
        }
    }
}

/// One granted slot.
struct Held(Arc<Slots>);

impl Drop for Held {
    fn drop(&mut self) {
        let mut state = self.0.lock();
        state.free += 1;
        if let Some((_, waker)) = state.queue.front() {
            waker.wake_by_ref();
        }
    }
}

/// The pool's slots as the operator sees them (a local type, for the trait). A pool makes one and hands out
/// clones of its `Arc`: the operator groups a driver's lanes by that pointer, so every call site of the pool
/// must return the same one.
pub(crate) struct SlotHandle {
    pub(crate) slots: Arc<Slots>,
    /// The configured `krishiv.python.udf.threads`, which a refusal reports (the slot limit may be one less).
    threads: usize,
}

impl SlotHandle {
    /// The slots of a pool of `threads` threads: all of them, but with two or more one thread stays free for
    /// the scalar UDF jobs that feed a relation task (they queue on the same threads and never take a slot).
    /// A one-thread pool starts a second, spare thread (`Pool::start`) that runs scalar jobs while the relation task holds the first; scalar jobs still run one at a time.
    pub(crate) fn new(threads: usize) -> Arc<Self> {
        let limit = if threads >= 2 { threads - 1 } else { threads };
        Arc::new(Self {
            slots: Slots::new(limit),
            threads: threads.max(1),
        })
    }
}

impl RelationSlots for SlotHandle {
    fn limit(&self) -> usize {
        self.slots.limit
    }

    fn poll_take(
        &self,
        n: usize,
        ticket: &mut Option<RelationTicket>,
        cx: &mut Context<'_>,
    ) -> Poll<krishiv_exec::Result<Vec<RelationSlot>>> {
        if n > self.slots.limit {
            return Poll::Ready(Err(KrishivError::from(KrishivPythonWorkerLimit {
                operation: "A Python UDF relation operator (one task per UDF of a driver)"
                    .to_owned(),
                needed: n.to_string(),
                available: self.slots.limit.to_string(),
                setting: "krishiv.python.udf.threads".to_owned(),
                value: self.threads.to_string(),
            })));
        }
        // The caller's own ticket, moved out of the box and back (never dropped while it waits).
        let own: Option<Ticket> = ticket
            .take()
            .and_then(|t| t.into_inner().downcast::<Ticket>().ok())
            .map(|t| *t);
        let mut state = self.slots.lock();
        let id = match &own {
            Some(t) => t.id,
            None => {
                let id = state.next_id;
                state.next_id += 1;
                id
            }
        };
        let at_front = state.queue.front().is_none_or(|(front, _)| *front == id);
        if at_front && state.free >= n {
            state.free -= n;
            // Leave the queue (a ticket that was never queued has nothing to leave) and let the next in line
            // look at what is left.
            state.queue.retain(|(q, _)| *q != id);
            if let Some((_, waker)) = state.queue.front() {
                waker.wake_by_ref();
            }
            drop(state);
            // The grant is made: the ticket (already out of the queue) drops here.
            drop(own);
            let slots = (0..n)
                .map(|_| RelationSlot::new(Held(Arc::clone(&self.slots))))
                .collect();
            return Poll::Ready(Ok(slots));
        }
        match state.queue.iter_mut().find(|(q, _)| *q == id) {
            Some((_, waker)) => waker.clone_from(cx.waker()),
            None => state.queue.push_back((id, cx.waker().clone())),
        }
        drop(state);
        let ticket_out = own.unwrap_or_else(|| Ticket {
            slots: Arc::clone(&self.slots),
            id,
        });
        *ticket = Some(RelationTicket::new(ticket_out));
        Poll::Pending
    }
}
