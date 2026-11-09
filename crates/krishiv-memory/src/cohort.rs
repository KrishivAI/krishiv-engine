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

use crate::consumer::GrantCredit;
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
    /// A seat in the cohort, idle, until the seat drops. Arbitration does
    /// not see it: a seat holding memory a round may wait for is taken
    /// through [`crate::ConsumerHandle::join_cohort`].
    #[must_use]
    pub fn join(self: &Arc<Self>) -> Seat {
        self.members.fetch_add(1, Ordering::AcqRel);
        Seat {
            cohort: Arc::clone(self),
            state: SeatState::Idle,
            credit: None,
        }
    }

    /// [`join`](Self::join) for the consumer whose grant credit is
    /// `credit` ([`crate::ConsumerHandle::join_cohort`]): while the seat
    /// holds, arbitration counts that consumer's memory as about to be
    /// released.
    pub(crate) fn join_for(self: &Arc<Self>, credit: Arc<GrantCredit>) -> Seat {
        let mut seat = self.join();
        seat.credit = Some(credit);
        seat
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
    /// The consumer's grant credit, for a seat taken through
    /// [`crate::ConsumerHandle::join_cohort`]: counts the seat while it holds.
    credit: Option<Arc<GrantCredit>>,
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
        let held = (
            self.state == SeatState::Holding,
            state == SeatState::Holding,
        );
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
        if let Some(credit) = &self.credit
            && held.0 != held.1
        {
            credit.seat_holding(held.1);
        }
    }
}

impl Seat {
    /// [`SeatState::Reserving`] to idle *without* waking the cohort: an
    /// asker backing off to check [`Cohort::others_active`] before it waits.
    /// Waking there would make waiters wake each other in turn while a
    /// holder keeps its memory. Any member waiting on this one is woken by
    /// this one's next move to idle (its last), as it then is active again or
    /// waits only while some other member is active. Other states: as
    /// [`set`](Self::set) to idle.
    pub fn stand_by(&mut self) {
        debug_assert_eq!(self.state, SeatState::Reserving, "only an asker stands by");
        if self.state == SeatState::Reserving {
            self.state = SeatState::Idle;
            self.cohort.active.fetch_sub(1, Ordering::AcqRel);
        } else {
            self.set(SeatState::Idle);
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
