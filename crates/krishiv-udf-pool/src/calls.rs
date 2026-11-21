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
//! ([`CallThreads::submit_task`]), where it waits only for scalar calls, which end on their own. Slots are
//! bounded twice: an owner's tasks hold at most `min(max_per_session, max_total) - 1`, so its scalar calls (a
//! `PythonEval` below the relation operator) always have a worker, and all owners' tasks together hold at most
//! `max_total - 1`, so the threads are never all held by idle tasks while a granted task waits for one (review
//! 2026-11-16 R2-1: the threads are shared across sessions). When an owner can run only one call the one slot
//! is the task's and a scalar call beside an open task of the same owner is refused ([`Refused::Reserve`])
//! rather than left to wait for a task that waits for it; another owner's call waits (its task never waits for
//! that call). Waiters are served in arrival order ([`TaskTicket`]): a grant never passes a waiter ahead of
//! it of the same owner, nor one whose own owner has room (it waits for room in all, which a later request
//! would take), so a driver needing many slots is not starved by smaller requests behind it (review R3-3);
//! a waiter held back only by its own owner's share does not hold other owners back. A waiter that goes
//! away drops its ticket and leaves the queue. With `max_total` 1 the one thread is shared by tasks and
//! scalar calls, so a slot is not granted while another owner's call that arrived before the request waits
//! for the thread: tickets and queued calls share one arrival sequence (review R4-5, R5-2).

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
    /// Most running calls (and slots, one fewer) of one tenant's owners together; `usize::MAX` for none.
    max_per_tenant: usize,
    idle: Duration,
}

#[derive(Default)]
struct Queue {
    calls: VecDeque<Queued>,
    /// Running calls per owner.
    running: HashMap<Owner, usize>,
    /// Live call threads.
    threads: usize,
    /// Threads waiting for a call, or started and about to look for one (so further calls submitted
    /// meanwhile do not start a thread each: the started one takes a call and wakes the next thread).
    waiting: usize,
    /// Relation task slots held per owner.
    slots: HashMap<Owner, usize>,
    /// Relation task slots held in all.
    slots_total: usize,
    /// Running calls per cap group ([`Owner::group`]: the tenant, or the principal without one), kept beside
    /// `running` so the cap check is O(1).
    running_by_group: HashMap<String, usize>,
    /// Relation task slots per cap group, beside `slots`.
    slots_by_group: HashMap<String, usize>,
    /// Operators waiting for slots, in arrival order (any owner's: a slot freed by one owner may be what
    /// another waits for).
    waiting_slots: VecDeque<Waiter>,
    /// The next arrival's number: a slot waiter's ticket id, or a queued call's `seq` (one sequence, so the two
    /// kinds are ordered by arrival).
    next_ticket: u64,
    shutdown: bool,
}

/// A call waiting for a thread.
struct Queued {
    owner: Owner,
    job: Job,
    /// Its arrival, in the sequence of slot tickets ([`Queue::next_ticket`]).
    seq: u64,
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
    /// More than the owner can ever hold (`limit` is [`CallThreads::slot_limit`]).
    TooMany,
}

/// One operator waiting for slots.
struct Waiter {
    id: u64,
    owner: Owner,
    /// The slots it waits for.
    n: usize,
    waker: Waker,
}

/// A waiter's place in the slot queue: it leaves the queue when dropped (its operator stopped waiting).
pub(crate) struct TaskTicket {
    shared: Arc<Shared>,
    id: u64,
}

impl Drop for TaskTicket {
    fn drop(&mut self) {
        let wakers = {
            let mut queue = self.shared.lock();
            let Some(at) = queue.waiting_slots.iter().position(|w| w.id == self.id) else {
                return;
            };
            queue.waiting_slots.remove(at);
            // The waiters behind it may be served now.
            queue.slot_wakers()
        };
        wakers.into_iter().for_each(Waker::wake);
    }
}

/// A relation task's slot: freed, and the waiting operators woken, when it drops.
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
            group_sub(&mut queue.slots_by_group, &self.owner);
            queue.slots_total = queue.slots_total.saturating_sub(1);
            queue.slot_wakers()
        };
        for w in waiters {
            w.wake();
        }
    }
}

impl Queue {
    /// The wakers of every operator waiting for slots.
    fn slot_wakers(&self) -> Vec<Waker> {
        self.waiting_slots.iter().map(|w| w.waker.clone()).collect()
    }
}

/// Adds `n` to `group`'s count.
fn group_add(counts: &mut HashMap<String, usize>, owner: &Owner, n: usize) {
    if let Some(g) = owner.group() {
        *counts.entry(g).or_default() += n;
    }
}

/// Takes one from `group`'s count, removing it at zero.
fn group_sub(counts: &mut HashMap<String, usize>, owner: &Owner) {
    let Some(g) = owner.group() else { return };
    if let Some(n) = counts.get_mut(&g) {
        *n = n.saturating_sub(1);
        if *n == 0 {
            counts.remove(&g);
        }
    }
}

fn group_total(counts: &HashMap<String, usize>, owner: &Owner) -> usize {
    owner
        .group()
        .map_or(0, |g| counts.get(&g).copied().unwrap_or(0))
}

impl CallThreads {
    /// At most `max_threads` threads and `max_per_owner` running calls of one owner; a thread ends after
    /// `idle` without a call.
    #[cfg(test)]
    pub(crate) fn new(max_threads: usize, max_per_owner: usize, idle: Duration) -> Self {
        Self::with_tenant_cap(max_threads, max_per_owner, usize::MAX, idle)
    }

    /// As [`Self::new`], and at most `max_per_tenant` running calls over all owners of one tenant (owners
    /// without a tenant are not capped by it): one tenant's sessions cannot take every call thread.
    pub(crate) fn with_tenant_cap(
        max_threads: usize,
        max_per_owner: usize,
        max_per_tenant: usize,
        idle: Duration,
    ) -> Self {
        CallThreads {
            shared: Arc::new(Shared {
                state: Mutex::new(Queue::default()),
                work: Condvar::new(),
                max_threads: max_threads.max(1),
                max_per_owner: max_per_owner.max(1),
                max_per_tenant: max_per_tenant.max(1),
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
        // Only the owner's own call is refused: its task may wait for it. Another owner's call waits for a
        // thread (the task never waits for it), even with `max_total` 1 (review R3-4).
        if self.owner_calls_for(&owner) == 1 && queue.slots.get(&owner).is_some_and(|n| *n > 0) {
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
        let seq = queue.next_ticket;
        queue.next_ticket += 1;
        let call = Queued { owner, job, seq };
        if first {
            queue.calls.push_front(call);
        } else {
            queue.calls.push_back(call);
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

    /// The calls one owner can run at once: its share, within the threads.
    fn owner_calls(&self) -> usize {
        self.shared.max_per_owner.min(self.shared.max_threads)
    }

    /// [`Self::owner_calls`] for `owner`: within its cap group's share when it is in one (a tenant that can
    /// run one call at once has no call left beside a relation task, whatever the session's share).
    fn owner_calls_for(&self, owner: &Owner) -> usize {
        if owner.group().is_some() {
            self.owner_calls().min(self.shared.max_per_tenant)
        } else {
            self.owner_calls()
        }
    }

    /// The most relation task slots one owner holds: one less than the calls it can run, so its scalar calls
    /// always have one, unless it can run only one.
    pub(crate) fn slot_limit(&self) -> usize {
        self.owner_calls().saturating_sub(1).max(1)
    }

    /// [`Self::slot_limit`] for `owner`: within its cap group's share when it is in one.
    pub(crate) fn slot_limit_for(&self, owner: &Owner) -> usize {
        if owner.group().is_some() {
            self.slot_limit().min(self.tenant_slot_limit())
        } else {
            self.slot_limit()
        }
    }

    /// Whether the group's cap, not the owner's share or the threads, bounds [`Self::slot_limit_for`].
    pub(crate) fn tenant_bound(&self, owner: &Owner) -> bool {
        owner.group().is_some() && self.tenant_slot_limit() < self.slot_limit()
    }

    /// The most relation task slots one tenant's owners hold together: one less than its calls, so a tenant's
    /// scalar calls always have one, unless it can run only one.
    fn tenant_slot_limit(&self) -> usize {
        self.shared
            .max_per_tenant
            .min(self.shared.max_threads)
            .saturating_sub(1)
            .max(1)
    }

    /// The most relation task slots all owners hold together: one less than the threads, unless there is
    /// only one.
    fn total_limit(&self) -> usize {
        self.shared.max_threads.saturating_sub(1).max(1)
    }

    /// Whether the threads (`max_total`), not the owner's share (`max_per_session`), bound [`Self::slot_limit`].
    pub(crate) fn threads_bound(&self) -> bool {
        self.shared.max_threads < self.shared.max_per_owner
    }

    /// `n` of `owner`'s slots at once, or none. A caller that waits gets a `ticket` (its place in the queue,
    /// kept between calls; `waker` is woken when the queue moves) and is served in arrival order: slots are
    /// granted only to the waiter at the head, or to a new caller when nobody waits. The grant clears the
    /// ticket.
    pub(crate) fn take_slots(
        &self,
        owner: &Owner,
        n: usize,
        ticket: &mut Option<TaskTicket>,
        waker: &Waker,
    ) -> (SlotTake, Vec<TaskSlot>) {
        if n > self.slot_limit_for(owner) {
            return (SlotTake::TooMany, Vec::new());
        }
        let mut queue = self.shared.lock();
        let at = ticket.as_ref().and_then(|t| {
            queue
                .waiting_slots
                .iter()
                .position(|w| w.id == t.id && Arc::ptr_eq(&t.shared, &self.shared))
        });
        let ahead = at.unwrap_or(queue.waiting_slots.len());
        // A waiter's arrival is its ticket's; a new request arrives now.
        let arrived = at
            .and_then(|i| queue.waiting_slots.get(i))
            .map_or(queue.next_ticket, |w| w.id);
        if !self.grantable(&queue, owner, n, ahead, arrived) {
            match at.and_then(|i| queue.waiting_slots.get_mut(i)) {
                Some(w) => {
                    if !w.waker.will_wake(waker) {
                        w.waker = waker.clone();
                    }
                }
                None => {
                    let id = queue.next_ticket;
                    queue.next_ticket += 1;
                    queue.waiting_slots.push_back(Waiter {
                        id,
                        owner: owner.clone(),
                        n,
                        waker: waker.clone(),
                    });
                    // A stale ticket (another pool's) is replaced; its drop takes this lock, so after it.
                    let stale = ticket.replace(TaskTicket {
                        shared: Arc::clone(&self.shared),
                        id,
                    });
                    drop(queue);
                    drop(stale);
                    return (SlotTake::Wait, Vec::new());
                }
            }
            return (SlotTake::Wait, Vec::new());
        }
        if let Some(i) = at {
            queue.waiting_slots.remove(i);
        }
        *queue.slots.entry(owner.clone()).or_default() += n;
        group_add(&mut queue.slots_by_group, owner, n);
        queue.slots_total += n;
        // The new head may be served now.
        let wakers = queue.slot_wakers();
        drop(queue);
        // Its entry is gone: the ticket's drop finds nothing to remove.
        drop(ticket.take());
        wakers.into_iter().for_each(Waker::wake);
        let slots = (0..n)
            .map(|_| TaskSlot {
                shared: Arc::clone(&self.shared),
                owner: owner.clone(),
            })
            .collect();
        (SlotTake::Taken, slots)
    }

    /// One of `owner`'s slots now, or none: for a caller that will not wait (it takes no place in the queue,
    /// so it needs no waker).
    pub(crate) fn try_take_now(&self, owner: &Owner) -> Option<TaskSlot> {
        let mut queue = self.shared.lock();
        if !self.grantable(
            &queue,
            owner,
            1,
            queue.waiting_slots.len(),
            queue.next_ticket,
        ) {
            return None;
        }
        *queue.slots.entry(owner.clone()).or_default() += 1;
        group_add(&mut queue.slots_by_group, owner, 1);
        queue.slots_total += 1;
        drop(queue);
        Some(TaskSlot {
            shared: Arc::clone(&self.shared),
            owner: owner.clone(),
        })
    }

    /// Whether `n` slots can go to `owner`, behind the first `ahead` waiters, for a request that arrived as
    /// number `arrived` ([`Queue::next_ticket`]).
    fn grantable(
        &self,
        queue: &Queue,
        owner: &Owner,
        n: usize,
        ahead: usize,
        arrived: u64,
    ) -> bool {
        let limit = self.slot_limit();
        let tenant_limit = self.tenant_slot_limit();
        let held_by = |o: &Owner| queue.slots.get(o).copied().unwrap_or(0);
        let tenant_room = |o: &Owner, n: usize| {
            o.group().is_none() || group_total(&queue.slots_by_group, o) + n <= tenant_limit
        };
        // A waiter ahead holds this request back if it is the same owner's, or if its owner has room (it waits
        // for room in all, which this request would take).
        let held_back = queue.waiting_slots.iter().take(ahead).any(|w| {
            &w.owner == owner
                || w.owner.same_group(owner)
                || (held_by(&w.owner) + w.n <= limit && tenant_room(&w.owner, w.n))
        });
        // With one thread, no task is granted while another owner's call that arrived before the request waits
        // for that thread: arrival order across both kinds, so neither a stream of one owner's tasks (review
        // R4-5) nor a stream of other owners' scalar calls (R5-2) keeps the other waiting.
        let thread_wanted = self.shared.max_threads == 1
            && queue
                .calls
                .iter()
                .any(|c| &c.owner != owner && c.seq < arrived);
        !held_back
            && !thread_wanted
            && held_by(owner) + n <= limit
            && tenant_room(owner, n)
            && queue.slots_total + n <= self.total_limit()
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
        let at = queue
            .calls
            .iter()
            .position(|c| self.may_run(queue, &c.owner))?;
        let Queued { owner, job, .. } = queue.calls.remove(at)?;
        *queue.running.entry(owner.clone()).or_default() += 1;
        group_add(&mut queue.running_by_group, &owner, 1);
        Some((owner, job))
    }

    fn runnable(&self, queue: &Queue) -> bool {
        queue.calls.iter().any(|c| self.may_run(queue, &c.owner))
    }

    /// Whether `owner` is under its share and its tenant under the tenant's.
    fn may_run(&self, queue: &Queue, owner: &Owner) -> bool {
        queue.running.get(owner).copied().unwrap_or(0) < self.max_per_owner
            && (owner.group().is_none()
                || group_total(&queue.running_by_group, owner) < self.max_per_tenant)
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
            // With one thread a slot request may wait for a queued call to start (`take_slots`): it can go now.
            let slot_wakers = if self.max_threads == 1 {
                queue.slot_wakers()
            } else {
                Vec::new()
            };
            drop(queue);
            slot_wakers.into_iter().for_each(Waker::wake);
            // A call that panics loses only its result (its future reports that it ended without one).
            let _ = std::panic::catch_unwind(AssertUnwindSafe(job));
            queue = self.lock();
            if let Some(n) = queue.running.get_mut(&owner) {
                *n -= 1;
                if *n == 0 {
                    queue.running.remove(&owner);
                }
            }
            group_sub(&mut queue.running_by_group, &owner);
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
            principal: format!("p-{session}"),
            session: session.to_owned(),
            tenant: None,
            capped: true,
        }
    }

    fn tenant_owner(tenant: &str, session: &str) -> Owner {
        Owner {
            tenant: Some(tenant.to_owned()),
            capped: true,
            ..owner(session)
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
    fn a_tenant_at_its_share_leaves_threads_to_other_tenants() {
        // Four sessions of tenant t1 submit calls that block; at most two run at once (the tenant's share),
        // so a call of t2 and one without a tenant still find threads.
        let calls = CallThreads::with_tenant_cap(8, 8, 2, Duration::from_secs(60));
        let running = Arc::new(AtomicUsize::new(0));
        let most = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Barrier::new(3));
        let (done, finished) = mpsc::channel();
        for i in 0..6 {
            let (running, most, gate, done) = (
                Arc::clone(&running),
                Arc::clone(&most),
                Arc::clone(&gate),
                done.clone(),
            );
            calls
                .submit(
                    tenant_owner("t1", &format!("s{i}")),
                    Box::new(move || {
                        let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                        most.fetch_max(now, Ordering::SeqCst);
                        gate.wait();
                        running.fetch_sub(1, Ordering::SeqCst);
                        let _ = done.send("t1");
                    }),
                )
                .expect("submitted");
        }
        for who in [tenant_owner("t2", "x"), owner("y")] {
            let d = done.clone();
            calls
                .submit(
                    who,
                    Box::new(move || {
                        let _ = d.send("other");
                    }),
                )
                .expect("submitted");
        }
        for _ in 0..2 {
            assert_eq!(
                finished.recv_timeout(Duration::from_secs(10)),
                Ok("other"),
                "another tenant's call ran while t1's two running calls blocked"
            );
        }
        // Release t1's calls in pairs: its share never passes two.
        for _ in 0..3 {
            gate.wait();
            for _ in 0..2 {
                assert_eq!(finished.recv_timeout(Duration::from_secs(10)), Ok("t1"));
            }
        }
        assert!(most.load(Ordering::SeqCst) <= 2);
    }

    #[test]
    fn a_tenants_relation_slots_stay_within_its_share() {
        let calls = CallThreads::with_tenant_cap(8, 8, 3, Duration::from_secs(60));
        let waker = futures::task::noop_waker();
        // Owners with a tenant: one less than the tenant's calls, over all its sessions.
        assert_eq!(calls.slot_limit_for(&tenant_owner("t", "a")), 2);
        // Without a tenant the principal is the cap group; with authentication off there is none.
        assert_eq!(calls.slot_limit_for(&owner("a")), 2);
        assert_eq!(calls.slot_limit_for(&uncapped_owner("a")), 7);
        assert!(calls.tenant_bound(&owner("a")));
        assert!(!calls.tenant_bound(&uncapped_owner("a")));
        let (take, first) = calls.take_slots(&tenant_owner("t", "a"), 2, &mut None, &waker);
        assert_eq!((take, first.len()), (SlotTake::Taken, 2));
        // The tenant's other session finds none, though its own share is free; another tenant does.
        let mut wait = None;
        let (take, _) = calls.take_slots(&tenant_owner("t", "b"), 1, &mut wait, &waker);
        assert_eq!(take, SlotTake::Wait);
        assert_eq!(
            calls
                .take_slots(&tenant_owner("u", "c"), 2, &mut None, &waker)
                .0,
            SlotTake::Taken
        );
        assert_eq!(
            calls
                .take_slots(&tenant_owner("t", "b"), 3, &mut None, &waker)
                .0,
            SlotTake::TooMany
        );
        drop(first);
        assert_eq!(
            calls
                .take_slots(&tenant_owner("t", "b"), 1, &mut wait, &waker)
                .0,
            SlotTake::Taken
        );
    }

    fn principal_owner(principal: &str, session: &str) -> Owner {
        Owner {
            principal: principal.to_owned(),
            ..owner(session)
        }
    }

    fn uncapped_owner(session: &str) -> Owner {
        Owner {
            capped: false,
            ..owner(session)
        }
    }

    #[test]
    fn with_authentication_off_the_sessions_of_the_one_principal_reach_the_thread_bound() {
        // maxPerTenant 2, but every session is the one local principal: eight run at once (the threads).
        let calls = CallThreads::with_tenant_cap(8, 8, 2, Duration::from_secs(60));
        let running = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(AtomicUsize::new(0));
        for i in 0..8 {
            let (running, release) = (Arc::clone(&running), Arc::clone(&release));
            calls
                .submit(
                    Owner {
                        principal: "local".to_owned(),
                        ..uncapped_owner(&format!("s{i}"))
                    },
                    Box::new(move || {
                        running.fetch_add(1, Ordering::SeqCst);
                        while release.load(Ordering::SeqCst) == 0 {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                    }),
                )
                .expect("submitted");
        }
        // All eight run together; with a cap of 2 only two would start. Fails after a deadline, never hangs.
        let deadline = Instant::now() + Duration::from_secs(10);
        while running.load(Ordering::SeqCst) < 8 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(running.load(Ordering::SeqCst), 8);
        release.store(1, Ordering::SeqCst);
        let waker = futures::task::noop_waker();
        let (take, held) = calls.take_slots(&uncapped_owner("x"), 7, &mut None, &waker);
        assert_eq!((take, held.len()), (SlotTake::Taken, 7));
    }

    #[test]
    fn a_principal_without_a_tenant_is_capped_and_cannot_take_every_slot() {
        // Single-tenant server: mallory's sessions together hold at most the cap's slots; alice still finds one.
        let calls = CallThreads::with_tenant_cap(8, 8, 3, Duration::from_secs(60));
        let waker = futures::task::noop_waker();
        let (take, held) =
            calls.take_slots(&principal_owner("mallory", "s1"), 2, &mut None, &waker);
        assert_eq!((take, held.len()), (SlotTake::Taken, 2));
        let (take, _) = calls.take_slots(&principal_owner("mallory", "s2"), 1, &mut None, &waker);
        assert_eq!(
            take,
            SlotTake::Wait,
            "mallory's other session is over her share"
        );
        let (take, _) = calls.take_slots(&principal_owner("alice", "s1"), 2, &mut None, &waker);
        assert_eq!(take, SlotTake::Taken);
    }

    #[test]
    fn a_principal_without_a_tenant_leaves_running_calls_to_others() {
        let calls = CallThreads::with_tenant_cap(8, 8, 2, Duration::from_secs(60));
        let gate = Arc::new(Barrier::new(3));
        let (done, finished) = mpsc::channel();
        for i in 0..4 {
            let (gate, done) = (Arc::clone(&gate), done.clone());
            calls
                .submit(
                    principal_owner("mallory", &format!("s{i}")),
                    Box::new(move || {
                        gate.wait();
                        let _ = done.send("mallory");
                    }),
                )
                .expect("submitted");
        }
        let d = done.clone();
        calls
            .submit(
                principal_owner("alice", "s"),
                Box::new(move || {
                    let _ = d.send("alice");
                }),
            )
            .expect("submitted");
        assert_eq!(finished.recv_timeout(Duration::from_secs(10)), Ok("alice"));
        for _ in 0..2 {
            gate.wait();
            for _ in 0..2 {
                assert_eq!(
                    finished.recv_timeout(Duration::from_secs(10)),
                    Ok("mallory")
                );
            }
        }
    }

    #[test]
    fn a_multi_slot_waiter_of_a_tenant_is_not_starved_by_its_other_sessions() {
        // Slot limit 2: a holds 1; b asks for 2 and waits; c's request for 1 must queue behind b.
        let calls = CallThreads::with_tenant_cap(8, 8, 3, Duration::from_secs(60));
        let waker = futures::task::noop_waker();
        let (take, held_a) = calls.take_slots(&tenant_owner("t", "a"), 1, &mut None, &waker);
        assert_eq!(take, SlotTake::Taken);
        let mut wait_b = None;
        let (take, _) = calls.take_slots(&tenant_owner("t", "b"), 2, &mut wait_b, &waker);
        assert_eq!(take, SlotTake::Wait);
        let mut wait_c = None;
        let (take, _) = calls.take_slots(&tenant_owner("t", "c"), 1, &mut wait_c, &waker);
        assert_eq!(
            take,
            SlotTake::Wait,
            "c waits behind b, though one slot is free"
        );
        drop(held_a);
        let (take, _) = calls.take_slots(&tenant_owner("t", "c"), 1, &mut wait_c, &waker);
        assert_eq!(take, SlotTake::Wait, "b is first");
        assert_eq!(
            calls
                .take_slots(&tenant_owner("t", "b"), 2, &mut wait_b, &waker)
                .0,
            SlotTake::Taken
        );
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
        let (take, _) = calls.take_slots(&owner("a"), 3, &mut None, &waker);
        assert_eq!(take, SlotTake::TooMany);
        let (take, first) = calls.take_slots(&owner("a"), 1, &mut None, &waker);
        assert_eq!((take, first.len()), (SlotTake::Taken, 1));
        let mut a_wait = None;
        let (take, none) = calls.take_slots(&owner("a"), 2, &mut a_wait, &waker);
        assert_eq!((take, none.len()), (SlotTake::Wait, 0));
        // Another owner's slots are its own: a waiter held back by its own share does not hold b back.
        assert_eq!(
            calls.take_slots(&owner("b"), 2, &mut None, &waker).0,
            SlotTake::Taken
        );
        let before = count.0.load(Ordering::SeqCst);
        drop(first);
        assert!(
            count.0.load(Ordering::SeqCst) > before,
            "the waiter was woken"
        );
        assert_eq!(
            calls.take_slots(&owner("a"), 2, &mut a_wait, &waker).0,
            SlotTake::Taken
        );
    }

    #[test]
    fn with_one_call_per_owner_a_scalar_call_beside_a_task_is_refused_not_queued() {
        let calls = CallThreads::new(8, 1, Duration::from_secs(60));
        assert_eq!(calls.slot_limit(), 1);
        let waker = futures::task::noop_waker();
        let (take, slots) = calls.take_slots(&owner("a"), 1, &mut None, &waker);
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
    fn with_one_call_per_tenant_a_scalar_call_beside_a_task_is_refused_not_queued() {
        // maxPerSession 8, maxPerTenant 1: the tenant can run one call, so a scalar call beside its task would
        // wait for the task, which waits for it.
        let calls = CallThreads::with_tenant_cap(8, 8, 1, Duration::from_secs(60));
        let waker = futures::task::noop_waker();
        let (take, slots) = calls.take_slots(&tenant_owner("acme", "a"), 1, &mut None, &waker);
        assert_eq!(take, SlotTake::Taken);
        let refused = calls.submit(tenant_owner("acme", "a"), Box::new(|| {}));
        assert!(matches!(refused, Err(Refused::Reserve)), "{refused:?}");
        drop(slots);
        calls
            .submit(tenant_owner("acme", "a"), Box::new(|| {}))
            .expect("after its task");
    }

    #[test]
    fn a_task_waiting_for_a_scalar_call_of_its_owner_does_not_hang() {
        // maxPerSession 2: the task holds one call while it waits for a scalar call submitted after it (a
        // `PythonEval` below the relation operator); the reserve runs that call.
        let calls = Arc::new(CallThreads::new(8, 2, Duration::from_secs(60)));
        let waker = futures::task::noop_waker();
        let (take, mut slots) = calls.take_slots(&owner("a"), 1, &mut None, &waker);
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
    fn slots_are_bounded_across_owners_by_the_threads() {
        // R2-1: three threads. Owner a holds two slots (every thread but one); b's take of two waits, and is
        // granted once a's slots free.
        let calls = CallThreads::new(3, 8, Duration::from_secs(60));
        assert_eq!(calls.slot_limit(), 2);
        assert!(calls.threads_bound());
        let count = Arc::new(Count(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&count));
        let (take, a) = calls.take_slots(&owner("a"), 2, &mut None, &waker);
        assert_eq!(take, SlotTake::Taken);
        let mut b_wait = None;
        assert_eq!(
            calls.take_slots(&owner("b"), 2, &mut b_wait, &waker).0,
            SlotTake::Wait
        );
        assert_eq!(
            calls.take_slots(&owner("b"), 1, &mut None, &waker).0,
            SlotTake::Wait
        );
        drop(a);
        assert!(count.0.load(Ordering::SeqCst) >= 1, "b's waiter was woken");
        assert_eq!(
            calls.take_slots(&owner("b"), 2, &mut b_wait, &waker).0,
            SlotTake::Taken
        );
    }

    #[test]
    fn a_session_share_over_the_threads_is_clamped_to_them() {
        // maxPerSession 16 over maxTotal 4: at most 3 slots, so a fourth task never waits for a thread its
        // own idle peers hold.
        let calls = CallThreads::new(4, 16, Duration::from_secs(60));
        assert_eq!(calls.slot_limit(), 3);
        let waker = futures::task::noop_waker();
        assert_eq!(
            calls.take_slots(&owner("a"), 4, &mut None, &waker).0,
            SlotTake::TooMany
        );
        assert_eq!(
            calls.take_slots(&owner("a"), 3, &mut None, &waker).0,
            SlotTake::Taken
        );
    }

    #[test]
    fn two_sessions_at_four_threads_never_strand_a_granted_task() {
        // maxTotal 4, maxPerSession 4: a's tasks hold the three slots and block (idle in `want`); b gets no
        // slot, and b's scalar call still runs on the fourth thread, within the deadline.
        let calls = CallThreads::new(4, 4, Duration::from_secs(60));
        let waker = futures::task::noop_waker();
        let (take, slots) = calls.take_slots(&owner("a"), 3, &mut None, &waker);
        assert_eq!(take, SlotTake::Taken);
        let gate = Arc::new(Barrier::new(4));
        for slot in slots {
            let gate = Arc::clone(&gate);
            calls
                .submit_task(
                    owner("a"),
                    Box::new(move || {
                        let _slot = slot;
                        gate.wait();
                    }),
                )
                .expect("task");
        }
        assert_eq!(
            calls.take_slots(&owner("b"), 1, &mut None, &waker).0,
            SlotTake::Wait
        );
        let (done, ran) = mpsc::channel();
        calls
            .submit(
                owner("b"),
                Box::new(move || {
                    let _ = done.send(());
                }),
            )
            .expect("scalar");
        ran.recv_timeout(Duration::from_secs(10))
            .expect("the scalar call ran on the thread kept for it");
        gate.wait();
        let deadline = Instant::now() + Duration::from_secs(10);
        while calls.take_slots(&owner("b"), 1, &mut None, &waker).0 != SlotTake::Taken {
            assert!(Instant::now() < deadline, "a's slots never freed");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn a_wide_waiter_is_not_passed_by_narrow_takes_behind_it() {
        // R3-3: seven slots in all. a holds six; b asks for seven and waits first; c's single slot would fit,
        // but it is behind b, so it waits until b is served.
        let calls = CallThreads::new(8, 8, Duration::from_secs(60));
        assert_eq!(calls.slot_limit(), 7);
        let waker = futures::task::noop_waker();
        let (take, a) = calls.take_slots(&owner("a"), 6, &mut None, &waker);
        assert_eq!(take, SlotTake::Taken);
        let mut b_ticket = None;
        assert_eq!(
            calls.take_slots(&owner("b"), 7, &mut b_ticket, &waker).0,
            SlotTake::Wait
        );
        assert!(b_ticket.is_some(), "b waits in the queue");
        for _ in 0..3 {
            let mut c_ticket = None;
            assert_eq!(
                calls.take_slots(&owner("c"), 1, &mut c_ticket, &waker).0,
                SlotTake::Wait,
                "c must not pass b"
            );
        }
        drop(a);
        let (take, b) = calls.take_slots(&owner("b"), 7, &mut b_ticket, &waker);
        assert_eq!(take, SlotTake::Taken);
        assert!(b_ticket.is_none(), "the grant clears the ticket");
        drop(b);
        // A waiter that goes away leaves the queue: nobody waits behind a dropped ticket.
        let (_, a) = calls.take_slots(&owner("a"), 7, &mut None, &waker);
        let mut gone = None;
        assert_eq!(
            calls.take_slots(&owner("b"), 7, &mut gone, &waker).0,
            SlotTake::Wait
        );
        drop(gone);
        drop(a);
        assert_eq!(
            calls.take_slots(&owner("c"), 1, &mut None, &waker).0,
            SlotTake::Taken
        );
    }

    #[test]
    fn with_one_thread_another_owner_s_scalar_call_waits_rather_than_fails() {
        // R3-4: maxTotal 1. a's task holds the one slot; a's own scalar call is refused (the task may wait for
        // it), b's is queued and runs.
        let calls = CallThreads::new(1, 4, Duration::from_secs(60));
        let waker = futures::task::noop_waker();
        let (take, _slot) = calls.take_slots(&owner("a"), 1, &mut None, &waker);
        assert_eq!(take, SlotTake::Taken);
        assert!(matches!(
            calls.submit(owner("a"), Box::new(|| {})),
            Err(Refused::Reserve)
        ));
        let (done, ran) = mpsc::channel();
        calls
            .submit(
                owner("b"),
                Box::new(move || {
                    let _ = done.send(());
                }),
            )
            .expect("b's call is queued, not refused");
        ran.recv_timeout(Duration::from_secs(10))
            .expect("b's call ran");
    }

    #[test]
    fn with_one_thread_a_queued_scalar_call_goes_before_the_next_task() {
        // R4-5: maxTotal 1. The thread is busy; b's scalar call is queued. a's next task is not granted while
        // b's call waits for the thread, so a stream of a's tasks cannot keep it waiting.
        let calls = CallThreads::new(1, 4, Duration::from_secs(60));
        let (release, gate) = mpsc::channel::<()>();
        calls
            .submit(
                owner("c"),
                Box::new(move || {
                    let _ = gate.recv_timeout(Duration::from_secs(10));
                }),
            )
            .expect("c");
        let (done, ran) = mpsc::channel();
        calls
            .submit(
                owner("b"),
                Box::new(move || {
                    let _ = done.send(());
                }),
            )
            .expect("b");
        let count = Arc::new(Count(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&count));
        let mut ticket = None;
        assert_eq!(
            calls.take_slots(&owner("a"), 1, &mut ticket, &waker).0,
            SlotTake::Wait
        );
        let _ = release.send(());
        ran.recv_timeout(Duration::from_secs(10))
            .expect("b's call ran first");
        let deadline = Instant::now() + Duration::from_secs(10);
        while count.0.load(Ordering::SeqCst) == 0 {
            assert!(Instant::now() < deadline, "a's waiter was never woken");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            calls.take_slots(&owner("a"), 1, &mut ticket, &waker).0,
            SlotTake::Taken
        );
    }

    #[test]
    fn with_one_thread_a_task_waiter_goes_before_a_later_scalar_call() {
        // R5-2: maxTotal 1, arrival order across both kinds. a waits for the slot d holds; b's scalar call is
        // queued after that. When d's slot frees, a is granted (it came first), so a stream of other owners'
        // scalar calls cannot keep a waiting; a request made after b's call, with no waiter left, still yields
        // to it.
        let calls = CallThreads::new(1, 4, Duration::from_secs(60));
        let (release, gate) = mpsc::channel::<()>();
        let (started, running) = mpsc::channel::<()>();
        calls
            .submit(
                owner("c"),
                Box::new(move || {
                    let _ = started.send(());
                    let _ = gate.recv_timeout(Duration::from_secs(10));
                }),
            )
            .expect("c");
        // c holds the thread (it is no longer queued, so it is not ahead of a).
        running
            .recv_timeout(Duration::from_secs(10))
            .expect("c started");
        let waker = futures::task::noop_waker();
        let (take, d) = calls.take_slots(&owner("d"), 1, &mut None, &waker);
        assert_eq!(take, SlotTake::Taken);
        let mut ticket = None;
        assert_eq!(
            calls.take_slots(&owner("a"), 1, &mut ticket, &waker).0,
            SlotTake::Wait
        );
        let (done, ran) = mpsc::channel();
        calls
            .submit(
                owner("b"),
                Box::new(move || {
                    let _ = done.send(());
                }),
            )
            .expect("b");
        drop(d);
        let (take, a) = calls.take_slots(&owner("a"), 1, &mut ticket, &waker);
        assert_eq!(take, SlotTake::Taken, "a arrived before b's call");
        drop(a);
        // No waiter is queued now and the slot is free, so only b's queued call (it arrived first) can hold e
        // back: `held_back` alone would grant e (review N6-1).
        assert_eq!(
            calls.try_take_now(&owner("e")).map(|_| ()),
            None,
            "a later request yields to b's queued call"
        );
        let _ = release.send(());
        ran.recv_timeout(Duration::from_secs(10))
            .expect("b's call ran");
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
