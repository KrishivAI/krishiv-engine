// SPDX-License-Identifier: Apache-2.0
//! Admission of arbitration rounds, per binding pool.
//!
//! A round works inside the subtree of its binding pool. Two rounds
//! *overlap* when one binding is the other or below it; overlapping rounds
//! run one at a time, in arrival order, while rounds in disjoint subtrees
//! (two queries hitting their own limits) run concurrently. A root round
//! overlaps every round.
//!
//! There is a single short sync lock and no lock ordering: a round is
//! admitted when it overlaps no active round and no round that arrived
//! before it, so it cannot deadlock and cannot be starved by later rounds.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::event::{Event, Listener};
use crate::pool::Node;
use crate::sync::{Mutex, lock};

#[derive(Default)]
struct State {
    next: u64,
    /// Admitted rounds.
    active: Vec<(u64, Arc<Node>)>,
    /// Waiting rounds, in arrival order.
    queued: Vec<(u64, Arc<Node>)>,
}

fn overlaps(a: &Node, b: &Node) -> bool {
    a.is_within(b) || b.is_within(a)
}

/// See the module docs.
#[derive(Default)]
pub(crate) struct Rounds {
    state: Mutex<State>,
    changed: Event,
}

impl Rounds {
    /// Waits until a round at `binding` may run; the guard ends it.
    pub(crate) fn enter(&self, binding: Arc<Node>) -> Enter<'_> {
        let mut s = lock(&self.state);
        let ticket = s.next;
        s.next += 1;
        s.queued.push((ticket, binding));
        drop(s);
        Enter {
            rounds: self,
            ticket,
            listener: None,
            admitted: false,
        }
    }

    fn try_admit(&self, ticket: u64) -> bool {
        let mut s = lock(&self.state);
        let Some(pos) = s.queued.iter().position(|(t, _)| *t == ticket) else {
            return false;
        };
        let binding = &s.queued[pos].1;
        let blocked = s.active.iter().any(|(_, n)| overlaps(n, binding))
            || s.queued[..pos].iter().any(|(_, n)| overlaps(n, binding));
        if blocked {
            return false;
        }
        let entry = s.queued.remove(pos);
        s.active.push(entry);
        true
    }

    fn remove(&self, ticket: u64, active: bool) {
        {
            let mut s = lock(&self.state);
            let list = if active { &mut s.active } else { &mut s.queued };
            list.retain(|(t, _)| *t != ticket);
        }
        self.changed.notify();
    }
}

/// Resolves to a [`RoundGuard`] once the round is admitted. Dropping it
/// before then leaves the queue.
pub(crate) struct Enter<'a> {
    rounds: &'a Rounds,
    ticket: u64,
    listener: Option<Listener<'a>>,
    admitted: bool,
}

impl<'a> Future for Enter<'a> {
    type Output = RoundGuard<'a>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<RoundGuard<'a>> {
        let this = &mut *self;
        loop {
            // Listen first, then check: a change in between still wakes us.
            let mut listener = this
                .listener
                .take()
                .unwrap_or_else(|| this.rounds.changed.listen());
            if this.rounds.try_admit(this.ticket) {
                this.admitted = true;
                return Poll::Ready(RoundGuard {
                    rounds: this.rounds,
                    ticket: this.ticket,
                });
            }
            match Pin::new(&mut listener).poll(cx) {
                Poll::Ready(()) => {} // changed since we listened: re-check
                Poll::Pending => {
                    this.listener = Some(listener);
                    return Poll::Pending;
                }
            }
        }
    }
}

impl Drop for Enter<'_> {
    fn drop(&mut self) {
        if !self.admitted {
            self.rounds.remove(self.ticket, false);
        }
    }
}

/// An admitted round; dropping it lets overlapping rounds in.
pub(crate) struct RoundGuard<'a> {
    rounds: &'a Rounds,
    ticket: u64,
}

impl Drop for RoundGuard<'_> {
    fn drop(&mut self) {
        self.rounds.remove(self.ticket, true);
    }
}
