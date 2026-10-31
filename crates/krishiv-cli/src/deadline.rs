// SPDX-License-Identifier: Apache-2.0
//! A wall-clock deadline for reading a result stream: a read that never completes is abandoned at the deadline
//! instead of blocking the CLI forever (`--max-seconds`).

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use futures::channel::oneshot;
use futures::future::{Either, select};
use futures::{Stream, StreamExt as _};

/// Fires once at `at`. The timer thread ends early when the deadline is dropped.
pub(crate) struct Deadline {
    fired: oneshot::Receiver<()>,
    /// Dropping it wakes the timer thread, which then exits without firing.
    _cancel: mpsc::Sender<()>,
    /// The timer could not be started (its sender was dropped unsent): reads are plain.
    unarmed: bool,
}

impl Deadline {
    /// A deadline `budget_s` seconds after `start`; `None` when the budget is too large to be a point in time
    /// (no cap).
    pub(crate) fn after(start: Instant, budget_s: f64) -> Option<Self> {
        let budget = Duration::try_from_secs_f64(budget_s).ok()?;
        Some(Self::at(start.checked_add(budget)?))
    }

    /// A deadline at `at` (already past: fires at once). If the timer thread cannot be started the deadline
    /// never fires, and only the between-batches check applies.
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
            unarmed: false,
        }
    }

    /// The next item of `stream`; `None` when the deadline passed first (the read is abandoned). A deadline whose
    /// timer never started reads without a cap.
    pub(crate) fn next_item<S: Stream + Unpin>(
        &mut self,
        stream: &mut S,
    ) -> Option<Option<S::Item>> {
        if self.unarmed {
            return Some(futures::executor::block_on(stream.next()));
        }
        match futures::executor::block_on(select(stream.next(), &mut self.fired)) {
            Either::Left((item, _)) => Some(item),
            Either::Right((Ok(()), _)) => None,
            Either::Right((Err(oneshot::Canceled), _)) => {
                // The timer thread never ran (its sender was dropped without firing): not a timeout.
                self.unarmed = true;
                Some(futures::executor::block_on(stream.next()))
            }
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
    fn a_timer_that_never_started_is_not_a_timeout() {
        let (tx, fired) = oneshot::channel();
        drop(tx);
        let (cancel, _kept) = mpsc::channel::<()>();
        let mut deadline = Deadline {
            fired,
            _cancel: cancel,
            unarmed: false,
        };
        let mut items = stream::iter([7_u8]);
        assert_eq!(deadline.next_item(&mut items), Some(Some(7)));
        assert_eq!(deadline.next_item(&mut items), Some(None));
    }

    #[test]
    fn a_budget_too_large_for_a_point_in_time_means_no_deadline() {
        assert!(Deadline::after(Instant::now(), 1e30).is_none());
        assert!(Deadline::after(Instant::now(), f64::MAX).is_none());
        assert!(Deadline::after(Instant::now(), 60.0).is_some());
    }

    #[test]
    fn a_deadline_already_past_fires_at_once() {
        let mut deadline = Deadline::at(Instant::now());
        let mut never = stream::pending::<u8>();
        assert!(deadline.next_item(&mut never).is_none());
    }
}
