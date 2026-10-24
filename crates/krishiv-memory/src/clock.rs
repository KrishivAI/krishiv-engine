// SPDX-License-Identifier: Apache-2.0
//! The arbitrator's clock: an explicit dependency of the pool tree, so
//! arbitration never needs a particular async runtime or its time driver.
//!
//! [`SystemClock`] (the default) is a small built-in timer: a thread,
//! started on first use, keeps a deadline heap and wakes due sleeps. It
//! works with any executor, on a runtime without timers, or under a plain
//! `block_on`. If the thread cannot be started the clock says so
//! ([`MemoryError::TimerUnavailable`]) and the next use tries again; it
//! never lets deadlines expire early. Tests inject a paused-Tokio clock.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};
use std::task::{Context, Poll, Waker};
use std::time::Instant;

use crate::{BoxFuture, MemoryError};

/// Time source and sleeps for arbitration deadlines.
pub trait Clock: Send + Sync + fmt::Debug {
    /// The current time.
    fn now(&self) -> Instant;
    /// Makes sure sleeps can be served. Arbitration calls this before it
    /// spills or aborts anything, and fails the request if it errs.
    ///
    /// # Errors
    /// [`MemoryError::TimerUnavailable`] if the clock cannot time sleeps.
    fn prepare(&self) -> Result<(), MemoryError> {
        Ok(())
    }
    /// A future that completes at or after `deadline`. Dropping it cancels
    /// the sleep.
    ///
    /// # Errors
    /// [`MemoryError::TimerUnavailable`] if the clock cannot time sleeps.
    fn sleep_until(&self, deadline: Instant) -> Result<BoxFuture<'static, ()>, MemoryError>;
}

/// Starts the timer thread running the given closure.
type Spawn = fn(Box<dyn FnOnce() + Send>) -> std::io::Result<()>;

fn spawn_thread(run: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("krishiv-memory-timer".to_owned())
        .spawn(run)
        .map(drop)
}

/// The default [`Clock`]: wall time and a built-in timer thread, started
/// on first use and stopped once the clock and all its sleeps are dropped.
#[derive(Clone)]
pub struct SystemClock {
    inner: Arc<ClockInner>,
}

struct ClockInner {
    timer: Mutex<Option<Arc<Timer>>>,
    spawn: Spawn,
}

impl Drop for ClockInner {
    fn drop(&mut self) {
        let timer = self
            .timer
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(t) = timer {
            t.release();
        }
    }
}

impl fmt::Debug for SystemClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemClock").finish_non_exhaustive()
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemClock {
    /// A clock whose timer thread starts on first use.
    pub fn new() -> Self {
        Self::with_spawn(spawn_thread)
    }

    /// A clock that starts its timer thread with `spawn` (tests inject
    /// failures).
    pub(crate) fn with_spawn(spawn: Spawn) -> Self {
        Self {
            inner: Arc::new(ClockInner {
                timer: Mutex::new(None),
                spawn,
            }),
        }
    }

    /// The running timer, starting it if needed (again after a failure).
    fn timer(&self) -> Result<Arc<Timer>, MemoryError> {
        let mut slot = self
            .inner
            .timer
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(t) = slot.as_ref() {
            return Ok(Arc::clone(t));
        }
        let timer = Arc::new(Timer {
            heap: Mutex::new(Heap {
                users: 1, // this clock
                ..Heap::default()
            }),
            changed: Condvar::new(),
        });
        let run = Arc::clone(&timer);
        (self.inner.spawn)(Box::new(move || Timer::run(&run))).map_err(|e| {
            MemoryError::TimerUnavailable {
                reason: format!("cannot start the timer thread: {e}"),
            }
        })?;
        *slot = Some(Arc::clone(&timer));
        Ok(timer)
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn prepare(&self) -> Result<(), MemoryError> {
        self.timer().map(drop)
    }

    fn sleep_until(&self, deadline: Instant) -> Result<BoxFuture<'static, ()>, MemoryError> {
        let timer = self.timer()?;
        timer.acquire();
        Ok(Box::pin(Sleep {
            deadline,
            timer,
            state: None,
        }))
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
    /// Clocks and sleeps using the timer; the thread exits at zero.
    users: usize,
}

struct Timer {
    heap: Mutex<Heap>,
    changed: Condvar,
}

impl Timer {
    fn acquire(&self) {
        self.heap
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .users += 1;
    }

    /// One clock or sleep stopped using the timer; with none left the thread
    /// exits.
    fn release(&self) {
        self.heap
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .users -= 1;
        self.changed.notify_one();
    }

    fn register(&self, deadline: Instant, entry: &Entry) {
        let mut heap = self.heap.lock().unwrap_or_else(PoisonError::into_inner);
        let id = heap.next;
        heap.next += 1;
        heap.due.push(Reverse((deadline, id)));
        heap.entries.push((id, Arc::downgrade(entry)));
        drop(heap);
        self.changed.notify_one();
    }

    /// Serves sleeps until no clock or sleep uses this timer any more.
    fn run(this: &Arc<Self>) {
        let mut heap = this.heap.lock().unwrap_or_else(PoisonError::into_inner);
        while heap.users > 0 {
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
                heap = this.heap.lock().unwrap_or_else(PoisonError::into_inner);
                continue;
            }
            heap = match heap.due.peek() {
                Some(&Reverse((deadline, _))) => {
                    let wait = deadline.saturating_duration_since(now);
                    this.changed
                        .wait_timeout(heap, wait)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
                None => this
                    .changed
                    .wait(heap)
                    .unwrap_or_else(PoisonError::into_inner),
            };
        }
    }
}

/// [`SystemClock::sleep_until`]'s future; it keeps the timer alive.
struct Sleep {
    deadline: Instant,
    timer: Arc<Timer>,
    /// Registered with the timer on the first pending poll.
    state: Option<Entry>,
}

impl Future for Sleep {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if Instant::now() >= self.deadline {
            return Poll::Ready(());
        }
        let this = &mut *self;
        let entry = this.state.get_or_insert_with(|| {
            let entry = Entry::default();
            this.timer.register(this.deadline, &entry);
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

impl Drop for Sleep {
    fn drop(&mut self) {
        self.timer.release();
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

    /// Held by a test to keep a gated timer thread from starting.
    static GATE: Mutex<()> = Mutex::new(());

    /// A timer thread that serves sleeps only once [`GATE`] is free.
    fn gated_spawn(run: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
        spawn_thread(Box::new(move || {
            drop(GATE.lock().unwrap_or_else(PoisonError::into_inner));
            run();
        }))
    }

    /// Sends its sleep's id and the time it was woken.
    struct Recorder {
        id: u64,
        woken: Mutex<mpsc::Sender<(u64, Instant)>>,
    }

    impl Wake for Recorder {
        fn wake(self: Arc<Self>) {
            let tx = self.woken.lock().unwrap_or_else(PoisonError::into_inner);
            // A closed channel means the test already failed elsewhere.
            tx.send((self.id, Instant::now())).ok();
        }
    }

    /// Sleeps registered out of order are woken in deadline order and
    /// never before their deadline. The timer starts only once every
    /// deadline has passed, so the order does not depend on how the
    /// machine schedules the timer or the waiting threads.
    #[test]
    fn sleeps_complete_in_deadline_order_and_not_early() {
        let gate = GATE.lock().unwrap_or_else(PoisonError::into_inner);
        let clock = SystemClock::with_spawn(gated_spawn);
        let timer = clock.timer().unwrap();
        let start = clock.now();
        let (tx, rx) = mpsc::channel();
        let entries: Vec<Entry> = [60u64, 10, 30]
            .into_iter()
            .map(|ms| {
                let entry = Entry::default();
                entry.lock().unwrap().waker = Some(Waker::from(Arc::new(Recorder {
                    id: ms,
                    woken: Mutex::new(tx.clone()),
                })));
                timer.register(start + Duration::from_millis(ms), &entry);
                entry
            })
            .collect();
        drop(tx);
        let last = start + Duration::from_millis(60);
        while Instant::now() < last {
            std::thread::sleep(last.saturating_duration_since(Instant::now()));
        }
        drop(gate);
        // Bounded, so a sleep the timer never wakes fails the test instead of
        // hanging it (each waker holds a sender, so the channel stays open).
        let woken: Vec<(u64, Instant)> = (0..3)
            .map(|_| {
                rx.recv_timeout(Duration::from_secs(60))
                    .expect("a sleep was never woken")
            })
            .collect();
        assert_eq!(woken.iter().map(|w| w.0).collect::<Vec<_>>(), [10, 30, 60]);
        for (ms, at) in woken {
            assert!(
                at >= start + Duration::from_millis(ms),
                "{ms} ms woken early"
            );
        }
        assert!(entries.iter().all(|e| e.lock().unwrap().fired));
    }

    #[test]
    fn a_dropped_sleep_is_cancelled_and_others_still_fire() {
        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        // Far enough away that a loaded machine cannot reach it before the
        // first poll (it is dropped unfired).
        let mut cancelled = SystemClock::new()
            .sleep_until(Instant::now() + Duration::from_secs(60))
            .unwrap();
        assert!(
            cancelled
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        drop(cancelled);
        let start = Instant::now();
        block_on(
            SystemClock::new()
                .sleep_until(start + Duration::from_millis(20))
                .unwrap(),
        );
        assert!(start.elapsed() >= Duration::from_millis(20));
    }

    #[test]
    fn a_past_deadline_is_ready_at_once() {
        let mut past = SystemClock::new().sleep_until(Instant::now()).unwrap();
        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        assert!(
            past.as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_ready()
        );
        assert!(format!("{:?}", SystemClock::default()).contains("SystemClock"));
    }

    fn failing_spawn(_: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
        Err(std::io::Error::other("no threads left"))
    }

    #[test]
    fn a_timer_that_cannot_start_is_reported_never_expires_early() {
        let clock = SystemClock::with_spawn(failing_spawn);
        let e = clock.prepare().unwrap_err();
        assert!(matches!(e, MemoryError::TimerUnavailable { .. }));
        assert!(e.to_string().contains("arbitration timer unavailable"));
        assert!(clock.sleep_until(Instant::now()).is_err());
    }
}
