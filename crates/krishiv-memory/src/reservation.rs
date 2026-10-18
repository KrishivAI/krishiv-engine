// SPDX-License-Identifier: Apache-2.0
//! RAII reservations.

use std::fmt;
use std::sync::Arc;

use crate::{ConsumerHandle, ConsumerId, MemoryError, MemoryPool};

/// Bytes reserved for one consumer, released when dropped. Operators hold
/// one per buffered structure; several may share a consumer. Its size is
/// exact: quantisation happens in the pool, per consumer.
pub struct Reservation {
    pool: Arc<dyn MemoryPool>,
    id: ConsumerId,
    bytes: usize,
    /// Set when made by [`ConsumerHandle::reservation`]: growth draws the
    /// handle's grant credit first, and the handle stays registered.
    handle: Option<Arc<ConsumerHandle>>,
    /// Shrinking parks the bytes as the handle's credit instead of
    /// returning them to the pool ([`Reservation::releasing_to_credit`]).
    to_credit: bool,
}

impl Reservation {
    /// An empty reservation for consumer `id` of `pool`.
    pub fn new(pool: Arc<dyn MemoryPool>, id: ConsumerId) -> Self {
        Self {
            pool,
            id,
            bytes: 0,
            handle: None,
            to_credit: false,
        }
    }

    pub(crate) fn with_handle(handle: Arc<ConsumerHandle>) -> Self {
        Self {
            pool: Arc::clone(handle.pool()),
            id: handle.id(),
            bytes: 0,
            handle: Some(handle),
            to_credit: false,
        }
    }

    /// Makes every later shrink, and the drop, park the bytes as the
    /// consumer's grant credit instead of returning them to the pool (see
    /// [`shrink_to_credit`](Self::shrink_to_credit)). For scratch memory
    /// that is reserved and freed repeatedly while its owner lends it bytes
    /// as credit (a spill writer's or reader's working memory): what it
    /// frees stays with the consumer for its next growth, so another
    /// consumer cannot take it in between. The owner takes the credit back
    /// afterwards (a `try_grow` draws it) or returns it with
    /// [`ConsumerHandle::release_credit`]. No effect without a handle.
    #[must_use]
    pub fn releasing_to_credit(mut self) -> Self {
        self.to_credit = self.handle.is_some();
        self
    }

    /// Takes grant credit for up to `bytes`; returns the bytes still needed
    /// from the pool and the credit taken.
    fn draw_credit(&self, bytes: usize) -> (usize, usize) {
        let credit = self.handle.as_ref().map_or(0, |h| h.take_credit(bytes));
        (bytes - credit, credit)
    }

    /// Puts back credit taken for a request the pool refused.
    fn refund(&self, credit: usize) {
        if let Some(h) = self.handle.as_ref().filter(|_| credit > 0) {
            h.add_credit(credit);
        }
    }

    /// The consumer this reservation belongs to.
    pub fn consumer(&self) -> ConsumerId {
        self.id
    }

    /// Grows by `bytes` without arbitration (the fast path for sync
    /// operator code; on error, return `NeedMemory` to the driver).
    ///
    /// # Errors
    /// [`MemoryError`] if the pool cannot grant the bytes now.
    pub fn try_grow(&mut self, bytes: usize) -> Result<(), MemoryError> {
        let (rest, credit) = self.draw_credit(bytes);
        if rest > 0
            && let Err(e) = self.pool.try_reserve(self.id, rest)
        {
            self.refund(credit);
            return Err(e);
        }
        self.added(bytes);
        Ok(())
    }

    /// Grows by `bytes`, running arbitration if needed. For drivers between
    /// batches and async code; never call with a sync lock held.
    ///
    /// # Errors
    /// [`MemoryError`] if arbitration could not make room or aborted the query.
    pub async fn grow(&mut self, bytes: usize) -> Result<(), MemoryError> {
        let (rest, credit) = self.draw_credit(bytes);
        // Cancel-safe: dropping this future mid-wait puts the credit back.
        let mut drawn = Drawn {
            handle: self.handle.as_ref(),
            credit,
        };
        if rest > 0 {
            self.pool.reserve(self.id, rest).await?;
        }
        drawn.credit = 0;
        drop(drawn);
        self.added(bytes);
        Ok(())
    }

    fn added(&mut self, bytes: usize) {
        self.bytes = self.bytes.saturating_add(bytes);
        if let Some(h) = &self.handle {
            h.grew(bytes);
        }
    }

    /// Returns up to `bytes` to the pool.
    pub fn shrink(&mut self, bytes: usize) {
        let bytes = bytes.min(self.bytes);
        self.bytes -= bytes;
        self.release(bytes, self.to_credit);
    }

    /// Moves up to `bytes` of this reservation to its consumer's grant
    /// credit without returning them to the pool, so the next `try_grow` of
    /// any reservation made from the same [`ConsumerHandle`] takes them
    /// without touching the pool: a race-free hand-over of reserved bytes,
    /// such as pre-reserved headroom lent to a spill writer's scratch
    /// reservation. Unused credit goes back with
    /// [`ConsumerHandle::release_credit`] or the consumer's unregistration.
    /// A reservation made by [`Reservation::new`] has no handle and shrinks
    /// instead.
    pub fn shrink_to_credit(&mut self, bytes: usize) {
        let bytes = bytes.min(self.bytes);
        self.bytes -= bytes;
        self.release(bytes, true);
    }

    /// Returns `bytes` the reservation no longer counts: to the handle's
    /// credit when asked to (and it has a handle), else to the pool.
    fn release(&self, bytes: usize, to_credit: bool) {
        match &self.handle {
            Some(h) if to_credit => {
                h.shrank(bytes);
                h.add_credit(bytes);
            }
            Some(h) => {
                self.pool.release(self.id, bytes);
                h.shrank(bytes);
            }
            None => self.pool.release(self.id, bytes),
        }
    }

    /// Bytes currently reserved.
    pub fn size(&self) -> usize {
        self.bytes
    }

    /// Re-charges this reservation to `to` (a consumer of the same pool),
    /// for state handed over to another operator instance (a breaker's
    /// combined tables): the pool's total is unchanged, and `to`'s spill
    /// hook now covers these bytes.
    ///
    /// # Errors
    /// [`MemoryError::UnknownConsumer`] if `to` is not registered with this
    /// reservation's pool; the reservation is unchanged.
    pub fn transfer_to(&mut self, to: &Arc<ConsumerHandle>) -> Result<(), MemoryError> {
        self.pool.transfer(self.id, to.id(), self.bytes)?;
        if let Some(h) = &self.handle {
            h.shrank(self.bytes);
        }
        to.grew(self.bytes);
        self.id = to.id();
        self.handle = Some(Arc::clone(to));
        Ok(())
    }
}

/// Credit taken by a pending [`Reservation::grow`]; refunded unless used.
struct Drawn<'a> {
    handle: Option<&'a Arc<ConsumerHandle>>,
    credit: usize,
}

impl Drop for Drawn<'_> {
    fn drop(&mut self) {
        if let Some(h) = self.handle.filter(|_| self.credit > 0) {
            h.add_credit(self.credit);
        }
    }
}

impl fmt::Debug for Reservation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reservation")
            .field("consumer", &self.id)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.release(self.bytes, self.to_credit);
        }
    }
}
