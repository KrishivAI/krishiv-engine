// SPDX-License-Identifier: Apache-2.0
//! A broadcast wake-up with no lost notifications, built on [`crate::sync`]
//! so loom can model it.
//!
//! A listener records the event's epoch when it is created; it completes
//! once the epoch has moved on. Create the listener *before* checking the
//! condition it waits for: a notification between the check and the await
//! then still completes it.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use crate::sync::{Mutex, lock};

#[derive(Debug, Default)]
struct State {
    epoch: u64,
    wakers: Vec<Waker>,
}

/// See the module docs.
#[derive(Debug, Default)]
pub(crate) struct Event {
    state: Mutex<State>,
}

impl Event {
    /// Completes every listener created before this call.
    pub(crate) fn notify(&self) {
        let wakers = {
            let mut s = lock(&self.state);
            s.epoch = s.epoch.wrapping_add(1);
            std::mem::take(&mut s.wakers)
        };
        for w in wakers {
            w.wake();
        }
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
        if !s.wakers.iter().any(|w| w.will_wake(cx.waker())) {
            s.wakers.push(cx.waker().clone());
        }
        Poll::Pending
    }
}
