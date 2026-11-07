// SPDX-License-Identifier: Apache-2.0
//! Cohorts: members of one query that size optional memory together.
//!
//! Some consumers hold memory that no spill can take back but that they
//! free on their own soon (a spilled nested loop's build chunk is dropped
//! at the end of each pass over its probe rows). Each instance sized alone
//! takes the share it would take if it ran by itself, so several of them in
//! one query fill the query's limit, and the last one's first reservation
//! fails although the others are about to free theirs. A [`Cohort`] lets
//! them see each other: its weight is the number of members, so a share
//! can be split among them, and a member that cannot get memory can wait
//! while another member *holds* memory it will free, instead of failing.
//!
//! A query keeps one cohort per name ([`MemoryPool::cohort`]); members join
//! with a [`Seat`]. Waiting follows the [`crate::event`] rule: take the
//! listener ([`Cohort::changed`]) before checking the condition, so a
//! release in between still wakes the waiter.
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
    /// Seats that hold memory they will free without help.
    holding: AtomicUsize,
    changed: Event,
}

impl Cohort {
    /// A seat in the cohort until the seat drops.
    #[must_use]
    pub fn join(self: &Arc<Self>) -> Seat {
        self.members.fetch_add(1, Ordering::AcqRel);
        Seat {
            cohort: Arc::clone(self),
            holding: false,
        }
    }

    /// The members now (at least one while a seat exists).
    #[must_use]
    pub fn members(&self) -> usize {
        self.members.load(Ordering::Acquire)
    }

    /// Whether a member other than `seat` holds memory it will free.
    #[must_use]
    pub fn others_holding(&self, seat: &Seat) -> bool {
        self.holding.load(Ordering::Acquire) > usize::from(seat.holding)
    }

    /// Completes after the next time a member stops holding or leaves.
    /// Created before the caller re-checks its condition, so a change in
    /// between still completes it.
    pub fn changed(&self) -> impl Future<Output = ()> + Send + '_ {
        self.changed.listen()
    }
}

/// A member of a [`Cohort`]; leaving (the drop) wakes its waiters.
#[derive(Debug)]
pub struct Seat {
    cohort: Arc<Cohort>,
    holding: bool,
}

impl Seat {
    /// The cohort this seat is in.
    #[must_use]
    pub fn cohort(&self) -> &Arc<Cohort> {
        &self.cohort
    }

    /// Marks whether this member holds memory it will free without help.
    /// Clearing it wakes the cohort's waiters.
    pub fn set_holding(&mut self, holding: bool) {
        if holding == self.holding {
            return;
        }
        self.holding = holding;
        if holding {
            self.cohort.holding.fetch_add(1, Ordering::AcqRel);
        } else {
            self.cohort.holding.fetch_sub(1, Ordering::AcqRel);
            self.cohort.changed.notify();
        }
    }
}

impl Drop for Seat {
    fn drop(&mut self) {
        if self.holding {
            self.cohort.holding.fetch_sub(1, Ordering::AcqRel);
        }
        self.cohort.members.fetch_sub(1, Ordering::AcqRel);
        self.cohort.changed.notify();
    }
}
