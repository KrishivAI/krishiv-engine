// SPDX-License-Identifier: Apache-2.0
//! Synchronisation primitives, swapped for loom's under `--cfg loom`, so
//! the model tests explore every interleaving of the accounting, wake-up
//! and arbitration code (the models inject a clock that never fires)
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
