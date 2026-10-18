// SPDX-License-Identifier: Apache-2.0
//! A cancellation token: one atomic load to check, and a future that
//! completes on cancellation, for `select!` around every await.

use crate::sync::{AtomicBool, Event, Ordering};

/// Set once, never reset. Drivers check it at every batch boundary and race
/// every await against [`cancelled`](Self::cancelled).
#[derive(Debug, Default)]
pub struct CancellationToken {
    cancelled: AtomicBool,
    event: Event,
}

impl CancellationToken {
    /// A token that is not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancels and wakes every waiter.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.event.notify();
    }

    /// Whether [`cancel`](Self::cancel) was called (one atomic load).
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Completes once cancelled (at once if already cancelled).
    pub async fn cancelled(&self) {
        // Listen before checking: a cancel in between still completes it.
        let listener = self.event.listen();
        if self.is_cancelled() {
            return;
        }
        listener.await;
    }
}
