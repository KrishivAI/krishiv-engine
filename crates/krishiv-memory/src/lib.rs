// SPDX-License-Identifier: Apache-2.0
//! Memory management (layer L1): the pool tree, reservations and the
//! cooperative memory arbitrator (ADR 0007, spec 13 §3).
//!
//! # Model
//! - **Pool tree.** [`TreePool::new`] makes the process root. Its children
//!   are queries ([`MemoryPool::child`]), whose descendants are tasks,
//!   pipelines and operators. Every pool may have a limit; the root's defaults to
//!   [`default_limit`] (80% of the cgroup or physical memory limit).
//! - **Reserve, then allocate.** Consumers [`register`](MemoryPool::register)
//!   (optionally as [`Spillable`]) and hold [`Reservation`]s, which release on
//!   drop.
//! - **Quantum.** A consumer takes memory from its pools in steps of
//!   `krishiv.memory.reservationQuantum` (default 1 MiB, at most 1/64 of the
//!   smallest limit above it), so most growth is a local check and shared
//!   counters change at most once per step. The slack is bounded by one step
//!   per consumer and is reclaimed before anything spills. A request whose
//!   rounded size does not fit falls back to the exact size.
//! - **Spill protocol.** A consumer that can give memory back registers a
//!   [`Spillable`] (usually a [`SpillSignal`]); requests only set a flag that
//!   its owner services on its own task and completes with a [`SpillTicket`].
//!
//! ```
//! use std::sync::Arc;
//! use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, SpillSignal, Spillable, TreePool};
//!
//! let root = TreePool::new(&MemoryConfig { limit: Some(64 << 20), ..MemoryConfig::default() });
//! let query = root.child("query-1", None);
//! let signal = Arc::new(SpillSignal::new());
//! let id = query.register("sort", Some(signal.clone()));
//! let mut r = Reservation::new(query.clone(), id);
//! r.try_grow(1000).unwrap();
//! assert_eq!(root.reserved(), 1 << 20); // one quantum
//! assert!(r.try_grow(128 << 20).is_err()); // over the limit: never a panic
//! assert_eq!(signal.spill_requested(), None);
//! drop(r);
//! assert_eq!(root.reserved(), 0);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

mod error;
mod limit;
mod pool;
mod reservation;
mod spill;
mod sync;

pub use error::MemoryError;
pub use limit::{DEFAULT_LIMIT_PERCENT, FALLBACK_LIMIT, default_limit, effective_memory_limit};
pub use pool::{MemoryConfig, TreePool};
pub use reservation::Reservation;
pub use spill::{SpillSignal, SpillTicket, Spillable};

/// A boxed `Send` future (the same type as `futures::future::BoxFuture`).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Identifies a registered consumer. Ids are unique in the process, so an
/// id used with the wrong pool is reported, never charged to another
/// consumer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ConsumerId(u64);

/// A memory pool in the tree (spec 13 §3). [`TreePool`] is the
/// implementation; the trait is the seam operators and drivers depend on.
pub trait MemoryPool: Send + Sync + std::fmt::Debug {
    /// A child pool, optionally with its own limit. Children of the root
    /// are queries.
    fn child(&self, name: &str, limit: Option<usize>) -> Arc<dyn MemoryPool>;
    /// Registers a consumer of this pool; use the id with this pool only.
    /// `spill` is held weakly: `None`, or a dropped hook, means the consumer
    /// cannot be asked to spill.
    fn register(&self, name: &str, spill: Option<Arc<dyn Spillable>>) -> ConsumerId;
    /// Unregisters a consumer and returns everything it holds.
    fn unregister(&self, id: ConsumerId);
    /// Fast path, no arbitration and no waiting. Quantised.
    ///
    /// # Errors
    /// [`MemoryError`] if a pool limit refuses the bytes, the query was
    /// aborted, or `id` is not registered with this pool.
    fn try_reserve(&self, id: ConsumerId, bytes: usize) -> Result<(), MemoryError>;
    /// Slow path: may run arbitration. Called only by drivers between
    /// batches or from async code, never with a sync lock held.
    ///
    /// # Errors
    /// [`MemoryError`] if no room could be made.
    fn reserve<'a>(
        &'a self,
        id: ConsumerId,
        bytes: usize,
    ) -> BoxFuture<'a, Result<(), MemoryError>>;
    /// Returns up to `bytes` reserved by `id` (clamped to what it holds).
    fn release(&self, id: ConsumerId, bytes: usize);
    /// Bytes held by this pool's subtree, including quantisation slack.
    fn reserved(&self) -> usize;
    /// This pool's own limit.
    fn limit(&self) -> Option<usize>;
}

#[cfg(test)]
mod tests;
