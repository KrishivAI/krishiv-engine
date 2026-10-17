// SPDX-License-Identifier: Apache-2.0
//! Consumer registrations that unregister themselves, with grant credit.
//!
//! A sync operator whose `try_grow` fails returns `NeedMemory { bytes }`
//! with its state unchanged; the driver reserves the bytes outside the
//! operator ([`crate::SpillOwner::reserve`]) and calls the operator again
//! with the same input. The bytes the driver obtained are parked on the
//! [`ConsumerHandle`] as **grant credit**, and the operator's next
//! `try_grow` on a reservation made from the handle takes them from the
//! credit instead of the pool, so the retry never reserves twice. The driver
//! returns credit the retry did not use ([`ConsumerHandle::release_credit`]).

use std::fmt;
use std::sync::Arc;

use crate::sync::{AtomicUsize, Ordering};
use crate::{ConsumerId, MemoryPool, Reservation, Spillable};

/// A registered consumer that unregisters (returning everything it holds)
/// once the handle and every [`Reservation`] made from it are dropped, so a
/// reservation that moves into shared state (a pipeline breaker's combine)
/// keeps its consumer alive.
pub struct ConsumerHandle {
    pool: Arc<dyn MemoryPool>,
    id: ConsumerId,
    /// Bytes reserved for this consumer and not yet held by a reservation.
    credit: AtomicUsize,
    /// Bytes held by reservations made from this handle, and their peak.
    held: AtomicUsize,
    peak: AtomicUsize,
}

impl ConsumerHandle {
    /// Registers a consumer of `pool` (see [`MemoryPool::register`]).
    pub fn register(
        pool: Arc<dyn MemoryPool>,
        name: &str,
        spill: Option<Arc<dyn Spillable>>,
    ) -> Arc<Self> {
        let id = pool.register(name, spill);
        Arc::new(Self {
            pool,
            id,
            credit: AtomicUsize::new(0),
            held: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        })
    }

    /// The consumer id.
    pub fn id(&self) -> ConsumerId {
        self.id
    }

    /// The pool the consumer is registered with.
    pub fn pool(&self) -> &Arc<dyn MemoryPool> {
        &self.pool
    }

    /// An empty reservation for this consumer that draws grant credit before
    /// the pool and keeps the consumer registered while it lives.
    pub fn reservation(self: &Arc<Self>) -> Reservation {
        Reservation::with_handle(Arc::clone(self))
    }

    /// Driver side: `bytes` were just reserved for this consumer in the pool
    /// (by [`crate::SpillOwner::reserve`]); park them for the next
    /// `try_grow`.
    pub fn add_credit(&self, bytes: usize) {
        self.credit.fetch_add(bytes, Ordering::AcqRel);
    }

    /// Unused grant credit.
    pub fn credit(&self) -> usize {
        self.credit.load(Ordering::Acquire)
    }

    /// Returns unused grant credit to the pool.
    pub fn release_credit(&self) {
        let unused = self.credit.swap(0, Ordering::AcqRel);
        if unused > 0 {
            self.pool.release(self.id, unused);
        }
    }

    /// The most bytes reservations made from this handle held at once
    /// (for `EXPLAIN ANALYZE`).
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }

    pub(crate) fn grew(&self, bytes: usize) {
        let held = self.held.fetch_add(bytes, Ordering::AcqRel) + bytes;
        self.peak.fetch_max(held, Ordering::AcqRel);
    }

    pub(crate) fn shrank(&self, bytes: usize) {
        self.held.fetch_sub(bytes, Ordering::AcqRel);
    }

    /// Takes up to `max` bytes of credit.
    pub(crate) fn take_credit(&self, max: usize) -> usize {
        let mut cur = self.credit.load(Ordering::Acquire);
        loop {
            let take = cur.min(max);
            if take == 0 {
                return 0;
            }
            match self
                .credit
                .compare_exchange(cur, cur - take, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return take,
                Err(actual) => cur = actual,
            }
        }
    }
}

impl fmt::Debug for ConsumerHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConsumerHandle")
            .field("id", &self.id)
            .field("credit", &self.credit())
            .field("peak", &self.peak())
            .finish_non_exhaustive()
    }
}

impl Drop for ConsumerHandle {
    fn drop(&mut self) {
        // Unregistering returns the credit with everything else.
        self.pool.unregister(self.id);
    }
}
