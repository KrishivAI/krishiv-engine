// SPDX-License-Identifier: Apache-2.0
//! Synchronisation primitives and timers, in one place so that
//! model-checking tests can swap them.

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
pub(crate) type Instant = tokio::time::Instant;

/// Resolves at `deadline`.
pub(crate) fn sleep_until(deadline: Instant) -> tokio::time::Sleep {
    tokio::time::sleep_until(deadline)
}

/// Whether arbitration can run here: it needs a Tokio runtime for its
/// timeouts. (Tokio cannot report whether that runtime's time driver is
/// enabled; `MemoryPool::reserve` documents the requirement.)
pub(crate) fn timers_available() -> bool {
    tokio::runtime::Handle::try_current().is_ok()
}
