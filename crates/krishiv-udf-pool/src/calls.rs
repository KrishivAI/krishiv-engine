// SPDX-License-Identifier: Apache-2.0
//! The pool's call threads: a bounded set of threads that run UDF calls, which block on their workers, so
//! that no call runs on a compute worker and N calls do not make N threads (review 2026-11-13 n2).
//!
//! The design note (§3.4, §9) says only that calls run on the pool's threads, so the simplest bounded design
//! is taken: at most `max_total` threads (one per worker the pool may run, so every worker can be busy), and
//! at most `max_per_session` running calls of one owner (one per worker that owner may have). A call beyond
//! either waits in a FIFO queue instead of on a thread. With both bounds a running call never waits long in
//! `Pool::checkout`: every worker of its owner, and every worker in all, is held by a running call or idle,
//! so one session's calls cannot take every thread while another session's call waits.
//!
//! Threads start when a call finds none free and end after `idle` without work, or when the pool is dropped.
//! A queued call holds what its closure holds (the pool among them), so the pool outlives its queued calls;
//! the threads themselves hold only the queue.

use std::collections::{HashMap, VecDeque};
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use crate::pool::Owner;

/// One call, run once on a call thread.
pub(crate) type Job = Box<dyn FnOnce() + Send + 'static>;

/// The call threads of one pool. Dropping it lets idle threads end; busy ones end after their call.
pub(crate) struct CallThreads {
    shared: Arc<Shared>,
}

struct Shared {
    state: Mutex<Queue>,
    work: Condvar,
    max_threads: usize,
    max_per_owner: usize,
    idle: Duration,
}

#[derive(Default)]
struct Queue {
    calls: VecDeque<(Owner, Job)>,
    /// Running calls per owner.
    running: HashMap<Owner, usize>,
    /// Live call threads.
    threads: usize,
    /// Threads waiting for a call, or started and about to look for one (so further calls submitted
    /// meanwhile do not start a thread each: the started one takes a call and wakes the next thread).
    waiting: usize,
    shutdown: bool,
}

impl CallThreads {
    /// At most `max_threads` threads and `max_per_owner` running calls of one owner; a thread ends after
    /// `idle` without a call.
    pub(crate) fn new(max_threads: usize, max_per_owner: usize, idle: Duration) -> Self {
        CallThreads {
            shared: Arc::new(Shared {
                state: Mutex::new(Queue::default()),
                work: Condvar::new(),
                max_threads: max_threads.max(1),
                max_per_owner: max_per_owner.max(1),
                idle,
            }),
        }
    }

    /// Runs `job` for `owner` on a call thread, now or when a thread and the owner's share are free.
    ///
    /// # Errors
    /// When no call thread exists and none can be started: `job` is then dropped unrun.
    pub(crate) fn submit(&self, owner: Owner, job: Job) -> Result<(), String> {
        let shared = &self.shared;
        let mut queue = shared.lock();
        queue.calls.push_back((owner, job));
        let started = shared.wake(&mut queue);
        if queue.threads == 0 {
            // No thread runs it later either: it was the last one pushed.
            queue.calls.pop_back();
            return Err(started
                .err()
                .unwrap_or_else(|| "no call thread is running".to_owned()));
        }
        Ok(())
    }

    /// Live call threads.
    #[cfg(test)]
    pub(crate) fn threads(&self) -> usize {
        self.shared.lock().threads
    }
}

impl Drop for CallThreads {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.work.notify_all();
    }
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The first queued call whose owner is under its share, taken off the queue.
    fn take(&self, queue: &mut Queue) -> Option<(Owner, Job)> {
        let at = queue.calls.iter().position(|(owner, _)| {
            queue.running.get(owner).copied().unwrap_or(0) < self.max_per_owner
        })?;
        let (owner, job) = queue.calls.remove(at)?;
        *queue.running.entry(owner.clone()).or_default() += 1;
        Some((owner, job))
    }

    fn runnable(&self, queue: &Queue) -> bool {
        queue
            .calls
            .iter()
            .any(|(owner, _)| queue.running.get(owner).copied().unwrap_or(0) < self.max_per_owner)
    }

    /// Makes sure a thread will take a runnable queued call: wakes a waiting one, else starts one under the
    /// bound (else every thread is busy and the first to finish takes it). Each thread that takes a call
    /// calls this again, so several runnable calls wake several threads in turn.
    fn wake(self: &Arc<Self>, queue: &mut Queue) -> Result<(), String> {
        if !self.runnable(queue) {
            // Every queued call's owner is at its share: one of its running calls takes it when it ends.
            return Ok(());
        }
        if queue.waiting > 0 {
            self.work.notify_one();
            return Ok(());
        }
        if queue.threads >= self.max_threads {
            return Ok(());
        }
        let shared = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("krishiv-udf-call".to_owned())
            .spawn(move || shared.serve());
        match spawned {
            Ok(_) => {
                queue.threads += 1;
                queue.waiting += 1;
                Ok(())
            }
            Err(e) => {
                let why = format!("cannot start a Python UDF call thread: {e}");
                tracing::warn!("{why}");
                Err(why)
            }
        }
    }

    /// A call thread: takes runnable calls until it has waited `idle` for one, or the pool is gone.
    fn serve(self: Arc<Self>) {
        let mut queue = self.lock();
        // Counted as waiting since it was started.
        queue.waiting -= 1;
        loop {
            let Some((owner, job)) = self.take(&mut queue) else {
                if queue.shutdown {
                    break;
                }
                queue.waiting += 1;
                let (guard, waited) = self
                    .work
                    .wait_timeout(queue, self.idle)
                    .unwrap_or_else(PoisonError::into_inner);
                queue = guard;
                queue.waiting -= 1;
                if waited.timed_out() && !self.runnable(&queue) {
                    break;
                }
                continue;
            };
            let _ = self.wake(&mut queue);
            drop(queue);
            // A call that panics loses only its result (its future reports that it ended without one).
            let _ = std::panic::catch_unwind(AssertUnwindSafe(job));
            queue = self.lock();
            if let Some(n) = queue.running.get_mut(&owner) {
                *n -= 1;
                if *n == 0 {
                    queue.running.remove(&owner);
                }
            }
            // Another of this owner's calls may be runnable now; this thread takes one next, others are woken.
            let _ = self.wake(&mut queue);
        }
        queue.threads -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Instant;

    fn owner(session: &str) -> Owner {
        Owner {
            principal: "p".to_owned(),
            session: session.to_owned(),
        }
    }

    #[test]
    fn many_calls_share_a_few_threads() {
        // Review 2026-11-13 n2: N concurrent calls do not make N threads.
        let calls = CallThreads::new(3, 8, Duration::from_secs(60));
        let running = Arc::new(AtomicUsize::new(0));
        let most = Arc::new(AtomicUsize::new(0));
        let (done, finished) = mpsc::channel();
        for _ in 0..40 {
            let (running, most, done) = (Arc::clone(&running), Arc::clone(&most), done.clone());
            calls
                .submit(
                    owner("a"),
                    Box::new(move || {
                        let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                        most.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(5));
                        running.fetch_sub(1, Ordering::SeqCst);
                        let _ = done.send(());
                    }),
                )
                .expect("submitted");
        }
        for _ in 0..40 {
            finished
                .recv_timeout(Duration::from_secs(30))
                .expect("every call ran");
        }
        assert!(calls.threads() <= 3, "{} threads", calls.threads());
        assert!(most.load(Ordering::SeqCst) <= 3);
    }

    #[test]
    fn an_owner_at_its_share_leaves_threads_to_others() {
        // a's calls block until released; b's call must still run, on a thread a cannot take.
        let calls = CallThreads::new(4, 2, Duration::from_secs(60));
        let gate = Arc::new(Barrier::new(3));
        let (done, finished) = mpsc::channel();
        for _ in 0..6 {
            let (gate, done) = (Arc::clone(&gate), done.clone());
            calls
                .submit(
                    owner("a"),
                    Box::new(move || {
                        gate.wait();
                        let _ = done.send("a");
                    }),
                )
                .expect("submitted");
        }
        let b = done.clone();
        calls
            .submit(
                owner("b"),
                Box::new(move || {
                    let _ = b.send("b");
                }),
            )
            .expect("submitted");
        assert_eq!(
            finished.recv_timeout(Duration::from_secs(10)),
            Ok("b"),
            "b ran while a's two running calls blocked"
        );
        assert!(calls.threads() <= 3, "a holds two threads, b one");
        for _ in 0..3 {
            gate.wait();
        }
        for _ in 0..6 {
            assert_eq!(finished.recv_timeout(Duration::from_secs(10)), Ok("a"));
        }
    }

    #[test]
    fn idle_threads_end_and_a_panicking_call_loses_only_itself() {
        let calls = CallThreads::new(2, 2, Duration::from_millis(50));
        let (done, finished) = mpsc::channel();
        calls
            .submit(owner("a"), Box::new(|| panic!("a UDF call panicked")))
            .expect("submitted");
        calls
            .submit(
                owner("a"),
                Box::new(move || {
                    let _ = done.send(());
                }),
            )
            .expect("submitted");
        finished
            .recv_timeout(Duration::from_secs(10))
            .expect("the next call ran");
        let deadline = Instant::now() + Duration::from_secs(10);
        while calls.threads() > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(calls.threads(), 0, "idle threads end");
        let (done, finished) = mpsc::channel();
        calls
            .submit(
                owner("a"),
                Box::new(move || {
                    let _ = done.send(());
                }),
            )
            .expect("submitted");
        finished
            .recv_timeout(Duration::from_secs(10))
            .expect("a new thread starts");
    }

    #[test]
    fn dropping_the_threads_ends_them() {
        let calls = CallThreads::new(2, 2, Duration::from_secs(600));
        let shared = Arc::clone(&calls.shared);
        let (done, finished) = mpsc::channel();
        calls
            .submit(
                owner("a"),
                Box::new(move || {
                    let _ = done.send(());
                }),
            )
            .expect("submitted");
        finished.recv_timeout(Duration::from_secs(10)).expect("ran");
        drop(calls);
        let deadline = Instant::now() + Duration::from_secs(10);
        while shared.lock().threads > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(shared.lock().threads, 0);
    }
}
