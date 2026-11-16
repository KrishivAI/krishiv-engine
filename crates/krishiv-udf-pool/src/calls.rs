// SPDX-License-Identifier: Apache-2.0
//! The pool's call threads: a bounded set of threads that run UDF calls, which block on their workers, so
//! that no call runs on a compute worker and N calls do not make N threads (review 2026-11-13 n2).
//!
//! The design note (§3.4, §9) says only that calls run on the pool's threads, so the simplest bounded design
//! is taken: at most `max_total` threads (one per worker the pool may run, so every worker can be busy), and
//! at most `max_per_session` running calls of one owner (one per worker that owner may have). A call beyond
//! either waits in a FIFO queue instead of on a thread. With both bounds a running call never waits long in
//! `Pool::checkout`: every worker of its owner, and every worker in all, is held by a running call or idle.
//! While `max_per_session < max_total`, one session's calls cannot take every thread while another session's
//! call waits; otherwise one session may hold every thread, and another's call waits (in FIFO order, so it is
//! not starved) until one of those calls ends.
//!
//! Threads start when a call finds none free and end after `idle` without work, or when the pool is dropped.
//! A queued call holds what its closure holds (the pool among them), so the pool outlives its queued calls;
//! the threads themselves hold only the queue.
//!
//! **Relation tasks** (design `P3-T085-embedded-python-udf-kinds.md` §3.3a) hold a call and a worker for their
//! whole life, idle while they wait for frames, so they never queue behind their owner's own tasks: an
//! operator takes a **slot** per task first ([`CallThreads::take_slots`], all of a driver's at once, or none
//! and a waker woken when slots free), and a task with its slot goes to the front of the queue
//! ([`CallThreads::submit_task`]), where it waits only for a thread or for the owner's scalar calls, which end
//! on their own. An owner's tasks hold at most `max_per_session - 1` slots, so its scalar calls (a
//! `PythonEval` below the relation operator) always have a worker; with `max_per_session` 1 the one slot is
//! the task's and a scalar call beside an open task is refused ([`Refused::Reserve`]) rather than left to
//! wait for a task that waits for it.

use std::collections::{HashMap, VecDeque};
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::task::Waker;
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
    /// Relation task slots held per owner.
    slots: HashMap<Owner, usize>,
    /// Operators waiting for an owner's slots.
    slot_waiters: HashMap<Owner, Vec<Waker>>,
    shutdown: bool,
}

/// Why a call was not submitted.
#[derive(Debug)]
pub(crate) enum Refused {
    /// No call thread runs and none can be started.
    NoThread(String),
    /// A scalar call of an owner whose one call (`max_per_session` 1) a relation task holds.
    Reserve,
}

/// What [`CallThreads::take_slots`] did.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SlotTake {
    /// The slots are the caller's (one [`TaskSlot`] each).
    Taken,
    /// Too few are free: the waker is woken when one frees.
    Wait,
    /// More than the owner can ever hold.
    TooMany,
}

/// A relation task's slot: freed, and the owner's waiting operators woken, when it drops.
pub(crate) struct TaskSlot {
    shared: Arc<Shared>,
    owner: Owner,
}

impl TaskSlot {
    /// Whether this slot is `owner`'s, of the pool whose threads are `calls`.
    pub(crate) fn is_of(&self, calls: &CallThreads, owner: &Owner) -> bool {
        Arc::ptr_eq(&self.shared, &calls.shared) && &self.owner == owner
    }
}

impl Drop for TaskSlot {
    fn drop(&mut self) {
        let waiters = {
            let mut queue = self.shared.lock();
            if let Some(n) = queue.slots.get_mut(&self.owner) {
                *n = n.saturating_sub(1);
                if *n == 0 {
                    queue.slots.remove(&self.owner);
                }
            }
            queue.slot_waiters.remove(&self.owner).unwrap_or_default()
        };
        for w in waiters {
            w.wake();
        }
    }
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

    /// Runs `job`, a scalar call of `owner`, on a call thread, now or when a thread and the owner's share are
    /// free.
    ///
    /// # Errors
    /// When no call thread exists and none can be started, or the owner's only call is a relation task's
    /// ([`Refused::Reserve`]): `job` is then dropped unrun.
    pub(crate) fn submit(&self, owner: Owner, job: Job) -> Result<(), Refused> {
        let queue = self.shared.lock();
        if self.shared.max_per_owner == 1 && queue.slots.get(&owner).is_some_and(|n| *n > 0) {
            drop(queue);
            drop(job);
            return Err(Refused::Reserve);
        }
        self.push(queue, owner, job, false)
    }

    /// Runs `job`, a relation task of `owner` holding one of its slots, on a call thread: ahead of every
    /// queued call, so it waits only for a thread or for the owner's running scalar calls.
    ///
    /// # Errors
    /// As [`Self::submit`], without the reserve.
    pub(crate) fn submit_task(&self, owner: Owner, job: Job) -> Result<(), Refused> {
        let queue = self.shared.lock();
        self.push(queue, owner, job, true)
    }

    fn push(
        &self,
        mut queue: MutexGuard<'_, Queue>,
        owner: Owner,
        job: Job,
        first: bool,
    ) -> Result<(), Refused> {
        let shared = &self.shared;
        if first {
            queue.calls.push_front((owner, job));
        } else {
            queue.calls.push_back((owner, job));
        }
        let started = shared.wake(&mut queue);
        if queue.threads == 0 {
            // No thread runs it later either. It is dropped after the lock is released, since it may hold the
            // last reference to the pool, whose drop takes this lock.
            let unrun = if first {
                queue.calls.pop_front()
            } else {
                queue.calls.pop_back()
            };
            drop(queue);
            drop(unrun);
            return Err(Refused::NoThread(
                started
                    .err()
                    .unwrap_or_else(|| "no call thread is running".to_owned()),
            ));
        }
        Ok(())
    }

    /// The most relation task slots one owner holds: one less than its calls, so its scalar calls always
    /// have one, unless it has only one.
    pub(crate) fn slot_limit(&self) -> usize {
        self.shared.max_per_owner.saturating_sub(1).max(1)
    }

    /// `n` of `owner`'s slots at once, or none (`waker` is then woken when one frees).
    pub(crate) fn take_slots(
        &self,
        owner: &Owner,
        n: usize,
        waker: &Waker,
    ) -> (SlotTake, Vec<TaskSlot>) {
        if n > self.slot_limit() {
            return (SlotTake::TooMany, Vec::new());
        }
        let mut queue = self.shared.lock();
        let held = queue.slots.get(owner).copied().unwrap_or(0);
        if held + n > self.slot_limit() {
            let waiters = queue.slot_waiters.entry(owner.clone()).or_default();
            if !waiters.iter().any(|w| w.will_wake(waker)) {
                waiters.push(waker.clone());
            }
            return (SlotTake::Wait, Vec::new());
        }
        *queue.slots.entry(owner.clone()).or_default() += n;
        drop(queue);
        let slots = (0..n)
            .map(|_| TaskSlot {
                shared: Arc::clone(&self.shared),
                owner: owner.clone(),
            })
            .collect();
        (SlotTake::Taken, slots)
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

    /// A waker that counts its wakes.
    struct Count(AtomicUsize);

    impl std::task::Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn slots_are_taken_all_at_once_and_a_freed_one_wakes_the_waiter() {
        let calls = CallThreads::new(8, 3, Duration::from_secs(60));
        assert_eq!(calls.slot_limit(), 2, "one call is kept for scalar calls");
        let count = Arc::new(Count(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&count));
        let (take, _) = calls.take_slots(&owner("a"), 3, &waker);
        assert_eq!(take, SlotTake::TooMany);
        let (take, first) = calls.take_slots(&owner("a"), 1, &waker);
        assert_eq!((take, first.len()), (SlotTake::Taken, 1));
        let (take, none) = calls.take_slots(&owner("a"), 2, &waker);
        assert_eq!((take, none.len()), (SlotTake::Wait, 0));
        // Another owner's slots are its own.
        assert_eq!(calls.take_slots(&owner("b"), 2, &waker).0, SlotTake::Taken);
        drop(first);
        assert_eq!(count.0.load(Ordering::SeqCst), 1, "the waiter was woken");
        assert_eq!(calls.take_slots(&owner("a"), 2, &waker).0, SlotTake::Taken);
    }

    #[test]
    fn with_one_call_per_owner_a_scalar_call_beside_a_task_is_refused_not_queued() {
        let calls = CallThreads::new(8, 1, Duration::from_secs(60));
        assert_eq!(calls.slot_limit(), 1);
        let waker = futures::task::noop_waker();
        let (take, slots) = calls.take_slots(&owner("a"), 1, &waker);
        assert_eq!(take, SlotTake::Taken);
        let refused = calls.submit(owner("a"), Box::new(|| {}));
        assert!(matches!(refused, Err(Refused::Reserve)), "{refused:?}");
        // Another owner is not affected; nor is this one once the task ended.
        calls.submit(owner("b"), Box::new(|| {})).expect("b");
        drop(slots);
        calls
            .submit(owner("a"), Box::new(|| {}))
            .expect("a after its task");
    }

    #[test]
    fn a_task_waiting_for_a_scalar_call_of_its_owner_does_not_hang() {
        // maxPerSession 2: the task holds one call while it waits for a scalar call submitted after it (a
        // `PythonEval` below the relation operator); the reserve runs that call.
        let calls = Arc::new(CallThreads::new(8, 2, Duration::from_secs(60)));
        let waker = futures::task::noop_waker();
        let (take, mut slots) = calls.take_slots(&owner("a"), 1, &waker);
        assert_eq!(take, SlotTake::Taken);
        let slot = slots.pop().expect("one slot");
        let (scalar_done, scalar) = mpsc::channel();
        let (task_done, task) = mpsc::channel();
        calls
            .submit_task(
                owner("a"),
                Box::new(move || {
                    let _slot = slot;
                    let ran = scalar.recv_timeout(Duration::from_secs(10)).is_ok();
                    let _ = task_done.send(ran);
                }),
            )
            .expect("task");
        calls
            .submit(
                owner("a"),
                Box::new(move || {
                    let _ = scalar_done.send(());
                }),
            )
            .expect("scalar");
        assert_eq!(task.recv_timeout(Duration::from_secs(20)), Ok(true));
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
