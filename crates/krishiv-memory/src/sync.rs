// SPDX-License-Identifier: Apache-2.0
//! Synchronisation primitives and timers, swapped for loom's (and a timer
//! that never fires) under `--cfg loom`, so the model tests explore every
//! interleaving of the accounting, wake-up and arbitration code
//! (`RUSTFLAGS="--cfg loom" cargo test -p krishiv-memory --lib --release`).

#[cfg(all(test, loom))]
pub(crate) use loom::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, fence},
};
#[cfg(not(all(test, loom)))]
pub(crate) use std::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, fence},
};

/// Locks `m`, ignoring poisoning: every critical section in this crate
/// leaves its data consistent before any call that could unwind.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A point in time for arbitration deadlines (Tokio's clock, so tests can
/// pause it).
#[cfg(not(all(test, loom)))]
pub(crate) type Instant = tokio::time::Instant;
#[cfg(all(test, loom))]
pub(crate) type Instant = std::time::Instant;

/// Resolves at `deadline`. Under loom it never resolves: the models check
/// that progress never depends on a timeout.
#[cfg(not(all(test, loom)))]
pub(crate) fn sleep_until(deadline: Instant) -> tokio::time::Sleep {
    tokio::time::sleep_until(deadline)
}
#[cfg(all(test, loom))]
pub(crate) fn sleep_until(_deadline: Instant) -> std::future::Pending<()> {
    std::future::pending()
}

/// Whether arbitration can run here: it needs a Tokio runtime for its
/// timeouts. (Tokio cannot report whether that runtime's time driver is
/// enabled; `MemoryPool::reserve` documents the requirement.)
#[cfg(not(all(test, loom)))]
pub(crate) fn timers_available() -> bool {
    tokio::runtime::Handle::try_current().is_ok()
}
#[cfg(all(test, loom))]
pub(crate) fn timers_available() -> bool {
    true
}
