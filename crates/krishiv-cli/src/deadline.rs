// SPDX-License-Identifier: Apache-2.0
//! A wall-clock deadline for reading a result stream: a read that never completes is abandoned at the deadline
//! instead of blocking the CLI forever (`--max-seconds`).

use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use futures::channel::oneshot;
use futures::future::{Either, select};
use futures::{Stream, StreamExt as _};

/// Fires once at `at`. The timer thread ends early when the deadline is dropped.
pub(crate) struct Deadline {
    fired: oneshot::Receiver<()>,
    /// Dropping it wakes the timer thread, which then exits without firing.
    _cancel: mpsc::Sender<()>,
}

impl Deadline {
    /// A deadline at `at` (already past: fires at once).
    pub(crate) fn at(at: Instant) -> Self {
        let (tx, fired) = oneshot::channel();
        let (cancel, cancelled) = mpsc::channel::<()>();
        // A failed spawn leaves the deadline unarmed: the caller's between-batches check still applies.
        let _ = thread::Builder::new()
            .name("krishiv-deadline".to_owned())
            .spawn(move || {
                let wait = at.saturating_duration_since(Instant::now());
                if matches!(
                    cancelled.recv_timeout(wait),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    let _ = tx.send(());
                }
            });
        Self {
            fired,
            _cancel: cancel,
        }
    }

    /// The next item of `stream`; `None` when the deadline passed first (the read is abandoned).
    pub(crate) fn next_item<S: Stream + Unpin>(
        &mut self,
        stream: &mut S,
    ) -> Option<Option<S::Item>> {
        match futures::executor::block_on(select(stream.next(), &mut self.fired)) {
            Either::Left((item, _)) => Some(item),
            Either::Right(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::stream;

    use super::*;

    #[test]
    fn a_stream_that_never_yields_is_abandoned_at_the_deadline() {
        let mut deadline = Deadline::at(Instant::now() + Duration::from_millis(50));
        let mut never = stream::pending::<u8>();
        let started = Instant::now();
        assert!(deadline.next_item(&mut never).is_none());
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn items_before_the_deadline_pass_through_and_the_end_is_reported() {
        let mut deadline = Deadline::at(Instant::now() + Duration::from_secs(60));
        let mut items = stream::iter([1_u8, 2]);
        assert_eq!(deadline.next_item(&mut items), Some(Some(1)));
        assert_eq!(deadline.next_item(&mut items), Some(Some(2)));
        assert_eq!(deadline.next_item(&mut items), Some(None));
    }

    #[test]
    fn a_deadline_already_past_fires_at_once() {
        let mut deadline = Deadline::at(Instant::now());
        let mut never = stream::pending::<u8>();
        assert!(deadline.next_item(&mut never).is_none());
    }
}
