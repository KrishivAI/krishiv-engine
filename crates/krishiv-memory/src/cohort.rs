// SPDX-License-Identifier: Apache-2.0
//! Cohorts: members of one query that size optional memory together.
//!
//! Some consumers hold memory that no spill can take back but that they
//! free on their own soon (a spilled nested loop's build chunk is dropped
//! at the end of each pass over its probe rows). Each instance sized alone
//! takes the share it would take if it ran by itself, so several of them in
//! one query fill the query's limit, and the last one's first reservation
//! fails although the others are about to free theirs. A [`Cohort`] lets
//! them see each other: it counts its members, so a share can be split
//! among them, and a member that cannot get memory can wait while another
//! member is *active* (holds such memory, or is reserving it and will hold
//! it or give up) instead of failing.
//!
//! A query keeps one cohort per name ([`MemoryPool::cohort`]); members join
//! with a [`Seat`]. To wait, take the listener ([`Cohort::changed`]) before
//! checking [`Cohort::others_active`]: a member going idle in between still
//! completes it.
//!
//! [`MemoryPool::cohort`]: crate::MemoryPool::cohort

use std::future::Future;
use std::sync::Arc;

use crate::event::Event;
use crate::sync::{AtomicUsize, Ordering};

/// See the module docs.
#[derive(Debug, Default)]
pub struct Cohort {
    /// Seats taken.
    members: AtomicUsize,
    /// Seats in [`SeatState::Reserving`] or [`SeatState::Holding`].
    active: AtomicUsize,
    changed: Event,
}

impl Cohort {
    /// A seat in the cohort, idle, until the seat drops.
    #[must_use]
    pub fn join(self: &Arc<Self>) -> Seat {
        self.members.fetch_add(1, Ordering::AcqRel);
        Seat {
            cohort: Arc::clone(self),
            state: SeatState::Idle,
        }
    }

    /// The members now (at least one while a seat exists).
    #[must_use]
    pub fn members(&self) -> usize {
        self.members.load(Ordering::Acquire)
    }

    /// Whether a member other than `seat` is active: it holds memory it
    /// will free without help, or is reserving some.
    #[must_use]
    pub fn others_active(&self, seat: &Seat) -> bool {
        let own = usize::from(seat.state != SeatState::Idle);
        self.active.load(Ordering::Acquire) > own
    }

    /// Completes after the next time a member goes idle or leaves. Take it
    /// before checking [`others_active`](Self::others_active), so a change
    /// in between still completes it.
    pub fn changed(&self) -> impl Future<Output = ()> + Send + '_ {
        self.changed.listen()
    }
}

/// What a [`Seat`]'s member is doing with the cohort's kind of memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeatState {
    /// Holds none and is not asking for any.
    Idle,
    /// Asking for some: it will hold it, or give up and go idle.
    Reserving,
    /// Holds some, which it frees without help.
    Holding,
}

/// A member of a [`Cohort`]; going idle or leaving (the drop) wakes the
/// cohort's waiters.
#[derive(Debug)]
pub struct Seat {
    cohort: Arc<Cohort>,
    state: SeatState,
}

impl Seat {
    /// The cohort this seat is in.
    #[must_use]
    pub fn cohort(&self) -> &Arc<Cohort> {
        &self.cohort
    }

    /// This member's state.
    #[must_use]
    pub fn state(&self) -> SeatState {
        self.state
    }

    /// Moves this member to `state`. Going idle from an active state wakes
    /// the cohort's waiters; release the memory before that.
    pub fn set(&mut self, state: SeatState) {
        let (was, now) = (self.state != SeatState::Idle, state != SeatState::Idle);
        self.state = state;
        match (was, now) {
            (false, true) => {
                self.cohort.active.fetch_add(1, Ordering::AcqRel);
            }
            (true, false) => {
                self.cohort.active.fetch_sub(1, Ordering::AcqRel);
                self.cohort.changed.notify();
            }
            _ => {}
        }
    }
}

impl Drop for Seat {
    fn drop(&mut self) {
        self.set(SeatState::Idle);
        self.cohort.members.fetch_sub(1, Ordering::AcqRel);
        self.cohort.changed.notify();
    }
}
