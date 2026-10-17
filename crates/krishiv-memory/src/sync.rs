// SPDX-License-Identifier: Apache-2.0
//! Synchronisation primitives, in one place so that
//! model-checking tests can swap them.

pub(crate) use std::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicUsize, Ordering},
};

/// Locks `m`, ignoring poisoning: every critical section in this crate
/// leaves its data consistent before any call that could unwind.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}
