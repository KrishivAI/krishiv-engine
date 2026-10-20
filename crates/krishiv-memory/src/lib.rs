// SPDX-License-Identifier: Apache-2.0
//! Memory management (layer L1): the pool tree, reservations and the
//! cooperative memory arbitrator (ADR 0007, spec 13 §3).
//!
//! # Model
//! - **Pool tree.** [`TreePool::new`] makes the process root. Its children
//!   are queries ([`MemoryPool::child`]), whose descendants are tasks,
//!   pipelines and operators, and capped pools for external consumers
//!   ([`TreePool::external`]: a UDF worker pool's RSS, scan prefetch).
//!   Every pool may have a limit; the root's defaults to
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
//! - **Cooperative arbitration.** [`MemoryPool::try_reserve`] never waits.
//!   [`MemoryPool::reserve`] runs a round when it fails (see the
//!   `arbitrate` module): reclaim slack, *request* spills from consumers of
//!   queries above their fair share and await their [`SpillTicket`]s, then
//!   abort the query furthest above its fair share, waiting for aborted
//!   queries to release before aborting another.
//!
//! # Driver protocol
//! Each pipeline driver has a [`SpillOwner`] and registers every consumer
//! it owns with a signal from [`SpillOwner::signal`]. At every batch
//! boundary it checks [`MemoryPool::check_aborted`] and
//! [`SpillOwner::pending`] (atomic loads) and services spill requests on its
//! own task. A sync operator whose `try_grow` fails returns `NeedMemory`
//! with its state unchanged; the driver then calls [`SpillOwner::reserve`],
//! which spills the driver's own consumers first (the arbitrator asks the
//! requester's driver only after every other victim), then arbitrates while still servicing requests
//! from other rounds. The bytes it obtained are parked as grant credit on
//! the consumer's [`ConsumerHandle`] and taken by the operator's retried
//! `try_grow`, so the retry never reserves twice.
//!
//! ```
//! use std::sync::Arc;
//! use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, SpillOwner, Spillable, TreePool};
//!
//! let root = TreePool::new(&MemoryConfig { limit: Some(64 << 20), ..MemoryConfig::default() });
//! let query = root.child("query-1", None);
//! let driver = SpillOwner::new();
//! let signal = driver.signal();
//! let id = query.register("sort", Some(signal.clone()));
//! let mut r = Reservation::new(query.clone(), id);
//! r.try_grow(1000).unwrap();
//! assert_eq!(root.reserved(), 1 << 20); // one quantum
//! assert!(r.try_grow(128 << 20).is_err()); // over the limit: never a panic
//! assert!(!driver.pending());
//! drop(r);
//! assert_eq!(root.reserved(), 0);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

mod arbitrate;
mod clock;
mod consumer;
mod error;
mod event;
mod limit;
mod pool;
mod release;
mod reservation;
mod rounds;
mod spill;
mod spillfile;
mod sync;

pub use clock::{Clock, SystemClock};
pub use consumer::{ConsumerHandle, SpillStats};
pub use error::MemoryError;
pub use limit::{DEFAULT_LIMIT_PERCENT, FALLBACK_LIMIT, default_limit, effective_memory_limit};
pub use pool::{ArbitrationStats, DEFAULT_SPILL_SCRATCH_RESERVE, MemoryConfig, TreePool};
pub use release::{
    EPOCH_INTERVAL, pressure_epoch, release_if_pressured, release_if_pressured_here,
    set_release_hook,
};
pub use reservation::Reservation;
pub use spill::{OwnerId, SpillOwner, SpillService, SpillSignal, SpillTicket, Spillable};
pub use spillfile::{
    SpillCodec, SpillConfig, SpillError, SpillFile, SpillManager, SpillReader, SpillScope,
    SpillWriter,
};

/// A boxed `Send` future (the same type as `futures::future::BoxFuture`).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Identifies a registered consumer. Ids are unique in the process, so an
/// id used with the wrong pool is reported, never charged to another
/// consumer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ConsumerId(u64);

/// A memory pool in the tree (spec 13 §3). [`TreePool`] is the
/// implementation; the trait is the seam operators and drivers depend on.
pub trait MemoryPool: Send + Sync + std::fmt::Debug {
    /// A child pool, optionally with its own limit. Children of the root
    /// are queries.
    fn child(&self, name: &str, limit: Option<usize>) -> Arc<dyn MemoryPool>;
    /// Registers a consumer of this pool; use the id with this pool only.
    /// `spill` is held weakly: `None`, or a dropped hook, means the consumer
    /// cannot be asked to spill. A driver's consumers pass a signal from
    /// their [`SpillOwner`].
    fn register(&self, name: &str, spill: Option<Arc<dyn Spillable>>) -> ConsumerId;
    /// Unregisters a consumer and returns everything it holds.
    fn unregister(&self, id: ConsumerId);
    /// Fast path, no arbitration and no waiting. Quantised.
    ///
    /// # Errors
    /// [`MemoryError`] if a pool limit refuses the bytes, the query was
    /// aborted, or `id` is not registered with this pool.
    fn try_reserve(&self, id: ConsumerId, bytes: usize) -> Result<(), MemoryError>;
    /// [`try_reserve`](Self::try_reserve) for a spill writer's scratch
    /// memory: it may also take the pools' spill-scratch floors, and it is
    /// granted exactly. Only for memory freed again once the spill's I/O
    /// ends ([`Reservation::for_spill_scratch`]).
    ///
    /// # Errors
    /// As [`try_reserve`](Self::try_reserve).
    fn try_reserve_scratch(&self, id: ConsumerId, bytes: usize) -> Result<(), MemoryError> {
        self.try_reserve(id, bytes)
    }
    /// Slow path: may run arbitration. Called only by drivers between
    /// batches or from async code, never with a sync lock held; any executor
    /// works (deadlines use the pool's [`Clock`]). The arbitrator never asks the requester
    /// to spill, and its driver's other consumers last (the driver services
    /// them while it waits): drivers use [`SpillOwner::reserve`], which
    /// spills the driver's own consumers first. Consumers outside a query
    /// (external pools, the root) get the fast path only.
    ///
    /// # Errors
    /// [`MemoryError`] if arbitration could not make room or the query was
    /// aborted.
    fn reserve<'a>(
        &'a self,
        id: ConsumerId,
        bytes: usize,
    ) -> BoxFuture<'a, Result<(), MemoryError>>;
    /// [`reserve`](Self::reserve) without the abort phase: when spills
    /// cannot make room, fails with [`MemoryError::Exhausted`] instead of
    /// aborting a query. For requests with a cheaper fallback than an abort
    /// (a grace hash join partition that can be repartitioned instead).
    ///
    /// # Errors
    /// [`MemoryError`] if spilling could not make room or the query was
    /// aborted.
    fn reserve_without_abort<'a>(
        &'a self,
        id: ConsumerId,
        bytes: usize,
    ) -> BoxFuture<'a, Result<(), MemoryError>> {
        Box::pin(std::future::ready(self.try_reserve(id, bytes)))
    }
    /// Returns up to `bytes` reserved by `id` (clamped to what it holds).
    fn release(&self, id: ConsumerId, bytes: usize);
    /// Moves `bytes` held by consumer `from` to consumer `to` of this pool
    /// (clamped to what `from` holds), without changing what the pool
    /// holds: state handed from one operator instance to another stays
    /// accounted, now charged to (and spillable through) `to`.
    ///
    /// # Errors
    /// [`MemoryError::UnknownConsumer`] if either is not registered here.
    fn transfer(&self, from: ConsumerId, to: ConsumerId, bytes: usize) -> Result<(), MemoryError>;
    /// Bytes held by this pool's subtree, including quantisation slack.
    fn reserved(&self) -> usize;
    /// The most bytes this pool's subtree has held at once (the high-water
    /// mark of [`reserved`](Self::reserved) after accepted grants), for
    /// `EXPLAIN ANALYZE` and memory gates. Defaults to the current value.
    fn peak_reserved(&self) -> usize {
        self.reserved()
    }
    /// This pool's own limit.
    fn limit(&self) -> Option<usize>;
    /// `Err` once the arbitrator has aborted this pool's query: drivers call
    /// it at every batch boundary (one atomic load while not aborted). From
    /// the abort on, no reservation of the query succeeds and its held
    /// bytes only go down.
    ///
    /// # Errors
    /// The abort error, with its Spark-style error class.
    fn check_aborted(&self) -> Result<(), MemoryError>;
    /// Completes with the abort error once the arbitrator aborts this
    /// pool's query; never completes outside a query. Drivers race it
    /// against their waits (source I/O, a blocked sink), so an aborted query
    /// releases its memory without waiting for its next batch boundary.
    fn aborted(&self) -> BoxFuture<'_, MemoryError> {
        Box::pin(std::future::pending())
    }
}

#[cfg(all(test, not(loom)))]
mod tests;

#[cfg(all(test, loom))]
mod loom_tests;
