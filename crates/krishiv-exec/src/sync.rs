// SPDX-License-Identifier: Apache-2.0
//! Synchronisation primitives, swapped for loom's under `--cfg loom` so the
//! models in `loom_tests.rs` explore every interleaving of the morsel
//! cursor, the finalize countdown, cancellation and the result channel.
//!
//! [`Event`] is a broadcast wake-up with no lost notifications: a listener
//! records the event's epoch when created and completes once it moves on.
//! Create the listener *before* checking the condition it waits for.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

#[cfg(all(test, loom))]
pub(crate) use loom::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[cfg(not(all(test, loom)))]
pub(crate) use std::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

/// Locks `m`, ignoring poisoning: no critical section in this crate can
/// unwind with its data half-updated.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Adds `waker` to `wakers` unless an equivalent one is there.
pub(crate) fn register(wakers: &mut Vec<Waker>, waker: &Waker) {
    if !wakers.iter().any(|w| w.will_wake(waker)) {
        wakers.push(waker.clone());
    }
}

/// Wakes every waker in `wakers` (call with no lock held).
pub(crate) fn wake_all(wakers: Vec<Waker>) {
    for w in wakers {
        w.wake();
    }
}

#[derive(Debug, Default)]
struct EventState {
    epoch: u64,
    wakers: Vec<Waker>,
}

/// See the module docs.
#[derive(Debug, Default)]
pub(crate) struct Event {
    state: Mutex<EventState>,
}

impl Event {
    /// Completes every listener created before this call.
    pub(crate) fn notify(&self) {
        let wakers = {
            let mut s = lock(&self.state);
            s.epoch = s.epoch.wrapping_add(1);
            std::mem::take(&mut s.wakers)
        };
        wake_all(wakers);
    }

    /// A listener for the next [`notify`](Self::notify).
    pub(crate) fn listen(&self) -> Listener<'_> {
        Listener {
            event: self,
            epoch: lock(&self.state).epoch,
        }
    }
}

/// Completes after the event is notified (see [`Event::listen`]).
#[derive(Debug)]
pub(crate) struct Listener<'a> {
    event: &'a Event,
    epoch: u64,
}

impl Future for Listener<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let mut s = lock(&self.event.state);
        if s.epoch != self.epoch {
            return Poll::Ready(());
        }
        register(&mut s.wakers, cx.waker());
        Poll::Pending
    }
}
