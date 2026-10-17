// SPDX-License-Identifier: Apache-2.0
//! The arbitrator's clock: an explicit dependency of the pool tree, so
//! arbitration never needs a particular async runtime or its time driver.
//!
//! [`SystemClock`] (the default) is a small built-in timer: one lazily
//! started thread keeps a deadline heap and wakes the wakers of due sleeps.
//! It works with any executor, on a runtime without timers, or under a
//! plain `block_on`. Tests inject a clock driven by a paused Tokio clock.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError, Weak};
use std::task::{Context, Poll, Waker};
use std::time::Instant;

use crate::BoxFuture;

/// Time source and sleeps for arbitration deadlines.
pub trait Clock: Send + Sync + fmt::Debug {
    /// The current time.
    fn now(&self) -> Instant;
    /// A future that completes at or after `deadline`. Dropping it cancels
    /// the sleep.
    fn sleep_until(&self, deadline: Instant) -> BoxFuture<'static, ()>;
}

/// The default [`Clock`]: wall time and the built-in timer thread.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn sleep_until(&self, deadline: Instant) -> BoxFuture<'static, ()> {
        Box::pin(Sleep {
            deadline,
            state: None,
        })
    }
}

#[derive(Debug, Default)]
struct SleepState {
    fired: bool,
    waker: Option<Waker>,
}

/// One pending sleep; the timer holds it weakly, so a dropped sleep is
/// cancelled and skipped.
type Entry = Arc<Mutex<SleepState>>;

#[derive(Default)]
struct Heap {
    next: u64,
    /// Earliest deadline first; `next` breaks ties in registration order.
    due: BinaryHeap<Reverse<(Instant, u64)>>,
    entries: Vec<(u64, Weak<Mutex<SleepState>>)>,
}

struct Timer {
    heap: Mutex<Heap>,
    changed: Condvar,
}

/// The process-wide timer, its thread started on first use. `None` if the
/// thread could not be started: sleeps then complete at once, so deadlines
/// expire early (arbitration escalates sooner) instead of hanging.
fn timer() -> Option<&'static Timer> {
    static TIMER: OnceLock<Option<&'static Timer>> = OnceLock::new();
    *TIMER.get_or_init(|| {
        let timer: &'static Timer = Box::leak(Box::new(Timer {
            heap: Mutex::new(Heap::default()),
            changed: Condvar::new(),
        }));
        std::thread::Builder::new()
            .name("krishiv-memory-timer".to_owned())
            .spawn(move || timer.run())
            .ok()
            .map(|_| timer)
    })
}

impl Timer {
    fn register(&self, deadline: Instant, entry: &Entry) {
        let mut heap = self.heap.lock().unwrap_or_else(PoisonError::into_inner);
        let id = heap.next;
        heap.next += 1;
        heap.due.push(Reverse((deadline, id)));
        heap.entries.push((id, Arc::downgrade(entry)));
        drop(heap);
        self.changed.notify_one();
    }

    fn run(&self) {
        let mut heap = self.heap.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            let now = Instant::now();
            let mut due = Vec::new();
            while let Some(&Reverse((deadline, id))) = heap.due.peek() {
                if deadline > now {
                    break;
                }
                heap.due.pop();
                if let Some(pos) = heap.entries.iter().position(|(e, _)| *e == id) {
                    due.push(heap.entries.swap_remove(pos).1);
                }
            }
            if !due.is_empty() {
                drop(heap);
                for entry in due.iter().filter_map(Weak::upgrade) {
                    let waker = {
                        let mut s = entry.lock().unwrap_or_else(PoisonError::into_inner);
                        s.fired = true;
                        s.waker.take()
                    };
                    if let Some(w) = waker {
                        w.wake();
                    }
                }
                heap = self.heap.lock().unwrap_or_else(PoisonError::into_inner);
                continue;
            }
            heap = match heap.due.peek() {
                Some(&Reverse((deadline, _))) => {
                    let wait = deadline.saturating_duration_since(now);
                    self.changed
                        .wait_timeout(heap, wait)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
                None => self
                    .changed
                    .wait(heap)
                    .unwrap_or_else(PoisonError::into_inner),
            };
        }
    }
}

/// [`SystemClock::sleep_until`]'s future.
struct Sleep {
    deadline: Instant,
    /// Registered with the timer on the first pending poll.
    state: Option<Entry>,
}

impl Future for Sleep {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if Instant::now() >= self.deadline {
            return Poll::Ready(());
        }
        let Some(timer) = timer() else {
            return Poll::Ready(());
        };
        let deadline = self.deadline;
        let entry = self.state.get_or_insert_with(|| {
            let entry = Entry::default();
            timer.register(deadline, &entry);
            entry
        });
        let mut s = entry.lock().unwrap_or_else(PoisonError::into_inner);
        if s.fired {
            return Poll::Ready(());
        }
        s.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::mpsc;
    use std::task::Wake;
    use std::time::Duration;

    use super::*;

    struct ThreadWaker(std::thread::Thread);

    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }

    fn block_on(mut fut: BoxFuture<'static, ()>) {
        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        while fut
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
        {
            std::thread::park();
        }
    }

    #[test]
    fn sleeps_complete_in_deadline_order_and_not_early() {
        let start = Instant::now();
        let (tx, rx) = mpsc::channel();
        let threads: Vec<_> = [60u64, 10, 30]
            .into_iter()
            .map(|ms| {
                let tx = tx.clone();
                std::thread::spawn(move || {
                    let deadline = start + Duration::from_millis(ms);
                    block_on(SystemClock.sleep_until(deadline));
                    assert!(SystemClock.now() >= deadline);
                    tx.send(ms).unwrap();
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        drop(tx);
        assert_eq!(rx.iter().collect::<Vec<_>>(), [10, 30, 60]);
    }

    #[test]
    fn a_dropped_sleep_is_cancelled_and_others_still_fire() {
        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        let mut cancelled = SystemClock.sleep_until(Instant::now() + Duration::from_millis(5));
        assert!(
            cancelled
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        drop(cancelled);
        let start = Instant::now();
        block_on(SystemClock.sleep_until(start + Duration::from_millis(20)));
        assert!(start.elapsed() >= Duration::from_millis(20));
    }

    #[test]
    fn a_past_deadline_is_ready_at_once() {
        let mut past = SystemClock.sleep_until(Instant::now());
        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        assert!(
            past.as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_ready()
        );
        assert!(format!("{SystemClock:?}").contains("SystemClock"));
    }
}
