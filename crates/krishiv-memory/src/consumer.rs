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
//!
//! Credit is touched only by the owning driver's task (grant, retry,
//! release), so it needs no loom model of its own; the atomics only make the
//! handle `Sync`. Settling a grant wakes rounds through the pool's
//! `ReleaseSignal`, which is loom-modelled. A [`Reservation::new`] on a
//! handle's id neither draws credit nor keeps the consumer registered: use
//! [`ConsumerHandle::reservation`].
//!
//! The pool sees the grant too ([`GrantCredit`],
//! [`MemoryPool::register_with_credit`]): from the grant until the driver
//! settles its retry ([`ConsumerHandle::settle_grant`]) the bytes are about
//! to be used or returned, so an arbitration round that needs them waits for
//! the retry rather than failing (ADR 0007 update 2026-11-06).

use std::fmt;
use std::sync::{Arc, OnceLock, Weak};

use crate::pool::Shared;
use crate::sync::{AtomicU64, AtomicUsize, Ordering};
use crate::{ConsumerId, MemoryPool, Reservation, Spillable};

/// Spill files and bytes written by spill writers of one or more consumers
/// (one operator's consumers in every driver share one), for metrics.
/// Counted when a [`crate::SpillWriter`] made with a reservation from a
/// [`ConsumerHandle`] finishes its file.
#[derive(Debug, Default)]
pub struct SpillStats {
    files: AtomicU64,
    bytes: AtomicU64,
}

impl SpillStats {
    /// Spill files written.
    pub fn files(&self) -> u64 {
        self.files.load(Ordering::Relaxed)
    }

    /// Bytes written to spill files (as stored: compressed, with framing).
    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    pub(crate) fn record(&self, bytes: u64) {
        // Statistics only: no other memory is published through these.
        self.files.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// A consumer's grant credit: bytes reserved in the pool for it and not yet
/// held by a reservation, and the driver's grant not yet settled. Its
/// [`ConsumerHandle`] owns it; the pool reads it
/// ([`MemoryPool::register_with_credit`]) so arbitration can tell memory a
/// retry is about to use or return from memory in use.
#[derive(Default)]
pub struct GrantCredit {
    bytes: AtomicUsize,
    /// Bytes granted by [`crate::SpillOwner::reserve_for`] whose retry has
    /// not ended yet ([`ConsumerHandle::settle_grant`]).
    unsettled: AtomicUsize,
    /// Spill writes of this consumer in progress ([`SpillWriting`]): its
    /// memory is about to be released, so rounds wait for it before they
    /// fail a request.
    writing: AtomicUsize,
    /// The pool tree whose waiting rounds re-plan when a grant settles.
    shared: OnceLock<Weak<Shared>>,
}

impl GrantCredit {
    /// Unused credit, in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Acquire)
    }

    /// Granted bytes whose retry has not ended yet.
    pub fn unsettled(&self) -> usize {
        self.unsettled.load(Ordering::Acquire)
    }

    /// Whether a spill write of the consumer is in progress.
    pub(crate) fn writing(&self) -> bool {
        self.writing.load(Ordering::Acquire) > 0
    }

    /// Wakes `shared`'s release waiters when a grant settles. Set once, by
    /// the pool the consumer registers with.
    pub(crate) fn notify(&self, shared: Weak<Shared>) {
        let _ = self.shared.set(shared);
    }
}

impl fmt::Debug for GrantCredit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GrantCredit")
            .field("bytes", &self.bytes())
            .field("unsettled", &self.unsettled())
            .field("writing", &self.writing.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

/// A spill write in progress ([`ConsumerHandle::spilling`]); dropping it
/// ends the write's count and wakes the rounds waiting for it.
#[derive(Debug)]
pub struct SpillWriting(Arc<GrantCredit>);

impl Drop for SpillWriting {
    fn drop(&mut self) {
        self.0.writing.fetch_sub(1, Ordering::AcqRel);
        if let Some(shared) = self.0.shared.get().and_then(Weak::upgrade) {
            shared.release.released();
        }
    }
}

/// A registered consumer that unregisters (returning everything it holds)
/// once the handle and every [`Reservation`] made from it are dropped, so a
/// reservation that moves into shared state (a pipeline breaker's combine)
/// keeps its consumer alive.
pub struct ConsumerHandle {
    pool: Arc<dyn MemoryPool>,
    id: ConsumerId,
    /// Bytes reserved for this consumer and not yet held by a reservation.
    credit: Arc<GrantCredit>,
    /// Bytes held by reservations made from this handle, and their peak.
    held: AtomicUsize,
    peak: AtomicUsize,
    /// Where this consumer's spill writers count their files.
    spill_stats: Arc<SpillStats>,
}

impl ConsumerHandle {
    /// Registers a consumer of `pool` (see [`MemoryPool::register`]).
    pub fn register(
        pool: Arc<dyn MemoryPool>,
        name: &str,
        spill: Option<Arc<dyn Spillable>>,
    ) -> Arc<Self> {
        Self::register_counted(pool, name, spill, Arc::default())
    }

    /// [`register`](Self::register), counting spill files in `spill_stats`
    /// (shared by the consumers whose spills are reported together).
    pub fn register_counted(
        pool: Arc<dyn MemoryPool>,
        name: &str,
        spill: Option<Arc<dyn Spillable>>,
        spill_stats: Arc<SpillStats>,
    ) -> Arc<Self> {
        let credit = Arc::new(GrantCredit::default());
        let id = pool.register_with_credit(name, spill, &credit);
        Arc::new(Self {
            pool,
            id,
            credit,
            held: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            spill_stats,
        })
    }

    /// Where this consumer's spill files are counted.
    pub fn spill_stats(&self) -> &Arc<SpillStats> {
        &self.spill_stats
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

    /// `bytes` were just reserved for this consumer in the pool; park them
    /// for the next `try_grow`. Only [`crate::SpillOwner::reserve_for`]
    /// (and refunds) call this, so credit is always backed by pool bytes.
    pub(crate) fn add_credit(&self, bytes: usize) {
        self.credit.bytes.fetch_add(bytes, Ordering::AcqRel);
    }

    /// Unused grant credit.
    pub fn credit(&self) -> usize {
        self.credit.bytes()
    }

    /// The bytes reservations made from this handle hold now.
    pub fn held(&self) -> usize {
        self.held.load(Ordering::Relaxed)
    }

    /// `bytes` were granted to this consumer's driver for a retry
    /// ([`crate::SpillOwner::reserve_for`]): until
    /// [`settle_grant`](Self::settle_grant), rounds count them as about to
    /// be used or returned.
    pub(crate) fn add_unsettled(&self, bytes: usize) {
        self.credit.unsettled.fetch_add(bytes, Ordering::AcqRel);
    }

    /// Marks a spill write of this consumer as in progress until the guard
    /// drops: the memory the write frees (the rows lent to it as credit and
    /// the writer's scratch) counts as about to be released, so a round that
    /// cannot spill enough waits for it (re-planning on every release, up to
    /// `abortReleaseTimeout`) before it fails a request or aborts a query.
    ///
    /// Hold it only across a write that ends without help: the blocking
    /// write job and the release of the credit it freed, with no
    /// arbitration (no `grow`, no wait for memory or input) inside. A writer
    /// kept open between batches that arrive only after a reservation of the
    /// same driver (a merge join's group file) must not hold one: two such
    /// drivers would wait for each other until the deadline.
    #[must_use]
    pub fn spilling(&self) -> SpillWriting {
        self.credit.writing.fetch_add(1, Ordering::AcqRel);
        SpillWriting(Arc::clone(&self.credit))
    }

    /// The retry a grant was for has ended: returns unused credit to the
    /// pool, and rounds waiting for the grant re-plan.
    pub fn settle_grant(&self) {
        // Unused bytes go back before the grant stops counting as unsettled,
        // so a round planning in between counts them once or twice (it then
        // waits and re-plans on the wakeup below), never zero times.
        let unused = self.credit.bytes.swap(0, Ordering::AcqRel);
        if unused > 0 {
            self.pool.release(self.id, unused);
        }
        let unsettled = self.credit.unsettled.swap(0, Ordering::AcqRel);
        // Always, not only through the release: a sub-quantum remainder
        // returns nothing to the pools, and that release notifies no one.
        if unsettled > 0
            && let Some(shared) = self.credit.shared.get().and_then(Weak::upgrade)
        {
            shared.release.released();
        }
    }

    /// Returns unused grant credit to the pool. An unsettled grant stays
    /// unsettled: only the driver that asked for it settles it
    /// ([`settle_grant`](Self::settle_grant)); operators managing their own
    /// credit (flush steps) never hold one.
    pub fn release_credit(&self) {
        let unused = self.credit.bytes.swap(0, Ordering::AcqRel);
        if unused > 0 {
            self.pool.release(self.id, unused);
        }
    }

    /// The most bytes reservations made from this handle held at once
    /// (for `EXPLAIN ANALYZE`).
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    pub(crate) fn grew(&self, bytes: usize) {
        // Statistics only: no other memory is published through these.
        let held = self.held.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.peak.fetch_max(held, Ordering::Relaxed);
    }

    pub(crate) fn shrank(&self, bytes: usize) {
        self.held.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Takes up to `max` bytes of credit.
    pub(crate) fn take_credit(&self, max: usize) -> usize {
        let mut cur = self.credit.bytes.load(Ordering::Acquire);
        loop {
            let take = cur.min(max);
            if take == 0 {
                return 0;
            }
            match self.credit.bytes.compare_exchange(
                cur,
                cur - take,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
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
