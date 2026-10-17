// SPDX-License-Identifier: Apache-2.0
//! RAII reservations.

use std::fmt;
use std::sync::Arc;

use crate::{ConsumerId, MemoryError, MemoryPool};

/// Bytes reserved for one consumer, released when dropped. Operators hold
/// one per buffered structure; several may share a consumer. Its size is
/// exact: quantisation happens in the pool, per consumer.
pub struct Reservation {
    pool: Arc<dyn MemoryPool>,
    id: ConsumerId,
    bytes: usize,
}

impl Reservation {
    /// An empty reservation for consumer `id` of `pool`.
    pub fn new(pool: Arc<dyn MemoryPool>, id: ConsumerId) -> Self {
        Self { pool, id, bytes: 0 }
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
        self.pool.try_reserve(self.id, bytes)?;
        self.bytes = self.bytes.saturating_add(bytes);
        Ok(())
    }

    /// Grows by `bytes`, running arbitration if needed. For drivers between
    /// batches and async code; never call with a sync lock held.
    ///
    /// # Errors
    /// [`MemoryError`] if arbitration could not make room or aborted the query.
    pub async fn grow(&mut self, bytes: usize) -> Result<(), MemoryError> {
        self.pool.reserve(self.id, bytes).await?;
        self.bytes = self.bytes.saturating_add(bytes);
        Ok(())
    }

    /// Returns up to `bytes` to the pool.
    pub fn shrink(&mut self, bytes: usize) {
        let bytes = bytes.min(self.bytes);
        self.pool.release(self.id, bytes);
        self.bytes -= bytes;
    }

    /// Bytes currently reserved.
    pub fn size(&self) -> usize {
        self.bytes
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
            self.pool.release(self.id, self.bytes);
        }
    }
}
