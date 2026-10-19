// SPDX-License-Identifier: Apache-2.0
//! Cooperative arbitration on a deterministic Tokio runtime (current thread,
//! paused clock): spills at batch boundaries, fairness, timeouts, aborts.
#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_memory::{
    ArbitrationStats, BoxFuture, Clock, MemoryConfig, MemoryError, MemoryPool, Reservation,
    SpillOwner, SpillSignal, SpillTicket, Spillable, SystemClock, TreePool,
};
use tokio::time::{Instant, sleep};

const MIB: usize = 1 << 20;
const TIMEOUT: Duration = Duration::from_secs(2);
const RELEASE: Duration = Duration::from_secs(30);

/// Arbitration deadlines on Tokio's paused clock, for deterministic tests.
#[derive(Debug)]
struct TokioClock;

impl Clock for TokioClock {
    fn now(&self) -> std::time::Instant {
        Instant::now().into_std()
    }

    fn sleep_until(
        &self,
        deadline: std::time::Instant,
    ) -> Result<BoxFuture<'static, ()>, MemoryError> {
        Ok(Box::pin(tokio::time::sleep_until(Instant::from_std(
            deadline,
        ))))
    }
}

fn new_root(limit: usize) -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit: Some(limit),
            quantum: MIB,
            spill_wait_timeout: TIMEOUT,
            abort_release_timeout: RELEASE,
            spill_scratch_reserve: 0,
        },
        Arc::new(TokioClock),
    )
}

fn stats(aborts: u64, spill_requests: u64, rounds: u64) -> ArbitrationStats {
    ArbitrationStats {
        rounds,
        spill_requests,
        aborts,
    }
}

/// How a test driver behaves at its batch boundaries.
#[derive(Clone, Copy)]
struct Behaviour {
    /// Service spill requests (else ignore them).
    spills: bool,
    /// Delay before acting on a spill request or an abort.
    react_after: Duration,
    /// How long a spill takes once begun (its I/O).
    spill_takes: Duration,
}

const COOPERATIVE: Behaviour = Behaviour {
    spills: true,
    react_after: Duration::ZERO,
    spill_takes: Duration::ZERO,
};
const NO_SPILL: Behaviour = Behaviour {
    spills: false,
    react_after: Duration::ZERO,
    spill_takes: Duration::ZERO,
};

/// One consumer with its own driver, which runs batches and services spill
/// requests and aborts at every batch boundary, as the pipeline driver will.
struct Spiller {
    pool: Arc<dyn MemoryPool>,
    signal: Arc<SpillSignal>,
    reservation: Reservation,
}

impl Spiller {
    fn new(pool: &Arc<dyn MemoryPool>, name: &str, bytes: usize) -> Self {
        let owner = SpillOwner::new();
        let signal = owner.signal();
        let id = pool.register(name, Some(Arc::clone(&signal) as Arc<dyn Spillable>));
        let mut reservation = Reservation::new(Arc::clone(pool), id);
        reservation.try_grow(bytes).unwrap();
        signal.set_spillable_bytes(bytes);
        Self {
            pool: Arc::clone(pool),
            signal,
            reservation,
        }
    }

    /// Runs batches until `stop`; returns the bytes spilled or the abort.
    async fn drive(mut self, stop: Arc<AtomicBool>, b: Behaviour) -> Result<usize, MemoryError> {
        let mut spilled = 0;
        while !stop.load(Ordering::Acquire) {
            if self.pool.check_aborted().is_err() {
                sleep(b.react_after).await;
                self.pool.check_aborted()?;
            }
            if b.spills && self.signal.spill_requested().is_some() {
                sleep(b.react_after).await;
                let service = self.signal.begin().unwrap();
                sleep(b.spill_takes).await; // "written to disk"
                let freed = service.target().min(self.reservation.size());
                self.reservation.shrink(freed);
                self.signal.set_spillable_bytes(self.reservation.size());
                service.complete(freed);
                spilled += freed;
            }
            sleep(Duration::from_millis(1)).await; // one batch of work
        }
        Ok(spilled)
    }

    fn spawn(
        self,
        b: Behaviour,
    ) -> (
        Arc<AtomicBool>,
        tokio::task::JoinHandle<Result<usize, MemoryError>>,
    ) {
        let stop = Arc::new(AtomicBool::new(false));
        let handle = tokio::spawn(self.drive(Arc::clone(&stop), b));
        (stop, handle)
    }
}

fn holder(pool: &Arc<dyn MemoryPool>, name: &str, bytes: usize) -> Reservation {
    let id = pool.register(name, None);
    let mut r = Reservation::new(Arc::clone(pool), id);
    r.try_grow(bytes).unwrap();
    r
}

#[tokio::test(start_paused = true)]
async fn two_queries_one_limit_one_spills_while_the_other_allocates() {
    let root = new_root(8 * MIB);
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let (stop, driver) = Spiller::new(&qa, "sort", 6 * MIB).spawn(COOPERATIVE);

    let mut b = holder(&qb, "hash-build", 0);
    let start = Instant::now();
    b.grow(4 * MIB).await.unwrap();
    assert!(
        start.elapsed() < TIMEOUT,
        "served by a spill, not a timeout"
    );
    assert_eq!((qa.reserved(), qb.reserved()), (4 * MIB, 4 * MIB));
    stop.store(true, Ordering::Release);
    assert_eq!(
        driver.await.unwrap().unwrap(),
        2 * MIB,
        "A spilled the shortfall"
    );
    assert!(qa.check_aborted().is_ok() && qb.check_aborted().is_ok());
    assert_eq!(root.stats(), stats(0, 1, 1));
    drop(b);
    assert_eq!(root.reserved(), 0, "pools are empty at query end");
}

/// A victim that has begun spilling is awaited to the end of its spill,
/// however long its I/O takes: the timeout only bounds the wait for it to
/// begin, so nothing is aborted.
#[tokio::test(start_paused = true)]
async fn a_spill_in_progress_is_awaited_past_the_timeout() {
    let root = new_root(8 * MIB);
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let slow_io = Behaviour {
        spill_takes: 3 * TIMEOUT,
        ..COOPERATIVE
    };
    let (stop, driver) = Spiller::new(&qa, "sort", 6 * MIB).spawn(slow_io);

    let mut b = holder(&qb, "hash-build", 0);
    let start = Instant::now();
    b.grow(4 * MIB).await.unwrap();
    assert!(start.elapsed() >= 3 * TIMEOUT, "{:?}", start.elapsed());
    stop.store(true, Ordering::Release);
    assert_eq!(driver.await.unwrap().unwrap(), 2 * MIB);
    assert!(qa.check_aborted().is_ok() && qb.check_aborted().is_ok());
    assert_eq!(root.stats(), stats(0, 1, 1));
}

/// A victim that begins only after the timeout has been given up on: its
/// query is aborted, as one that never spills.
#[tokio::test(start_paused = true)]
async fn a_spill_begun_after_the_timeout_does_not_save_the_victim() {
    let root = new_root(8 * MIB);
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let late = Behaviour {
        react_after: TIMEOUT + Duration::from_millis(500),
        spill_takes: TIMEOUT,
        ..COOPERATIVE
    };
    let (_stop, driver) = Spiller::new(&qa, "sort", 6 * MIB).spawn(late);
    let mut b = holder(&qb, "hash-build", 0);
    b.grow(4 * MIB).await.unwrap();
    assert!(matches!(
        driver.await.unwrap(),
        Err(MemoryError::Aborted { .. })
    ));
    assert_eq!(root.stats().aborts, 1);
}

/// `reserve_without_abort` spills like `reserve` but fails instead of
/// aborting a query when spills cannot make room.
#[tokio::test(start_paused = true)]
async fn a_spill_only_request_never_aborts() {
    let root = new_root(8 * MIB);
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let (_stop, stubborn) = Spiller::new(&qa, "sort", 6 * MIB).spawn(NO_SPILL);
    let id = qb.register("grace-load", None);
    let e = qb.reserve_without_abort(id, 4 * MIB).await.unwrap_err();
    assert!(matches!(e, MemoryError::Exhausted { .. }), "{e:?}");
    assert_eq!(root.stats().aborts, 0);
    assert!(qa.check_aborted().is_ok() && qb.check_aborted().is_ok());
    assert_eq!(qb.reserved(), 0);
    stubborn.abort();
    assert!(stubborn.await.unwrap_err().is_cancelled());

    // With a victim that spills, the same request succeeds.
    let qc = root.child("query-c", None);
    let (stop, driver) = Spiller::new(&qc, "agg", 6 * MIB).spawn(COOPERATIVE);
    qb.reserve_without_abort(id, 4 * MIB).await.unwrap();
    assert_eq!(qb.reserved(), 4 * MIB);
    stop.store(true, Ordering::Release);
    assert!(driver.await.unwrap().unwrap() > 0);
    assert_eq!(root.stats().aborts, 0);
    qb.unregister(id);
}

#[tokio::test(start_paused = true)]
async fn spill_timeout_aborts_the_largest_other_query() {
    let root = new_root(8 * MIB);
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let (_stop, driver) = Spiller::new(&qa, "sort", 6 * MIB).spawn(NO_SPILL);

    let mut b = holder(&qb, "hash-build", 0);
    let start = Instant::now();
    b.grow(4 * MIB).await.unwrap();
    let waited = start.elapsed();
    assert!(waited >= TIMEOUT && waited < 2 * TIMEOUT, "{waited:?}");

    let e = driver.await.unwrap().unwrap_err();
    assert_eq!(e, MemoryError::Aborted { held: 6 * MIB });
    let k = KrishivError::from(e);
    assert_eq!(k.error_class(), ErrorClass::KrishivQueryAbortedOutOfMemory);
    assert_eq!(k.sql_state(), Some("53200"));
    assert!(qa.check_aborted().is_err() && qb.check_aborted().is_ok());
    assert_eq!(qa.reserved(), 0);
    assert_eq!(root.stats(), stats(1, 1, 1));
}

/// An aborted victim that is slow to release is pending release: the
/// requester waits for it and nothing else is aborted or failed.
#[tokio::test(start_paused = true)]
async fn a_slow_releasing_victim_is_aborted_once_and_the_requester_succeeds() {
    let root = new_root(8 * MIB);
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let qc = root.child("query-c", None);
    let slow = Behaviour {
        spills: false,
        react_after: 3 * TIMEOUT,
        spill_takes: Duration::ZERO,
    };
    let (_stop, victim) = Spiller::new(&qa, "sort", 5 * MIB).spawn(slow);
    let small = holder(&qc, "agg", 2 * MIB);

    let start = Instant::now();
    let mut b = holder(&qb, "join", 0);
    let first = tokio::spawn(async move { b.grow(3 * MIB).await.map(|()| b) });
    sleep(TIMEOUT + Duration::from_millis(500)).await; // A is aborted now
    // A second requester while A's memory is still held: it waits too.
    let mut c = holder(&qc, "agg-2", 0);
    c.grow(MIB).await.unwrap();
    let b = first.await.unwrap().unwrap();
    assert!(start.elapsed() >= 3 * TIMEOUT && start.elapsed() < RELEASE);
    assert_eq!(
        victim.await.unwrap().unwrap_err(),
        MemoryError::Aborted { held: 5 * MIB }
    );
    assert_eq!(root.stats().aborts, 1, "exactly one query aborted");
    assert!(qb.check_aborted().is_ok() && qc.check_aborted().is_ok());
    assert_eq!((b.size(), c.size(), small.size()), (3 * MIB, MIB, 2 * MIB));
}

#[tokio::test(start_paused = true)]
async fn a_victim_waiting_for_memory_is_released_by_the_abort() {
    let root = new_root(8 * MIB);
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let mut a = Spiller::new(&qa, "sort", 6 * MIB);
    let mut b = holder(&qb, "hash-build", 0);
    let start = Instant::now();
    let requester = tokio::spawn(async move { b.grow(4 * MIB).await.map(|()| b) });
    tokio::task::yield_now().await; // B's round is running
    // A's driver blocks in a plain `grow` instead of servicing B's request.
    let e = a.reservation.grow(4 * MIB).await.unwrap_err();
    assert_eq!(e, MemoryError::Aborted { held: 6 * MIB });
    drop(a);
    let b = requester.await.unwrap().unwrap();
    assert_eq!(b.size(), 4 * MIB);
    assert!(start.elapsed() < 2 * TIMEOUT, "no second timeout");
}

#[tokio::test(start_paused = true)]
async fn the_requester_is_aborted_when_it_holds_the_most() {
    let root = new_root(8 * MIB);
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let a = holder(&qa, "prefetch", MIB);
    let mut b = holder(&qb, "agg", 6 * MIB);

    let e = b.grow(4 * MIB).await.unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnableToAcquireMemory,
        "the failing call"
    );
    // The query's other tasks see the abort class, not the request.
    let sticky = qb.check_aborted().unwrap_err();
    assert_eq!(sticky, MemoryError::Aborted { held: 6 * MIB });
    assert!(qa.check_aborted().is_ok());
    assert_eq!(b.try_grow(1), Err(sticky));
    assert_eq!((a.size(), b.size()), (MIB, 6 * MIB));
    assert_eq!(root.stats(), stats(1, 0, 1));
}

#[tokio::test(start_paused = true)]
async fn a_task_limit_fails_the_request_not_the_query() {
    let root = new_root(64 * MIB);
    let q = root.child("query", None);
    let task = q.child("task", Some(4 * MIB));
    let mut r = holder(&task, "agg", 3 * MIB);
    assert_eq!(
        r.grow(2 * MIB).await.unwrap_err().error_class(),
        ErrorClass::UnableToAcquireMemory
    );
    assert!(q.check_aborted().is_ok() && task.check_aborted().is_ok());
    assert_eq!(root.stats().aborts, 0);
}

#[tokio::test(start_paused = true)]
async fn a_query_limit_only_spills_its_own_consumers() {
    let root = new_root(64 * MIB);
    let qa = root.child("query-a", Some(4 * MIB));
    let qb = root.child("query-b", None);
    let a1 = Spiller::new(&qa, "sort", 3 * MIB);
    let other = Spiller::new(&qb, "join", 8 * MIB);
    let other_signal = Arc::clone(&other.signal);
    let (stop, driver) = a1.spawn(COOPERATIVE);

    let mut a2 = holder(&qa, "agg", 0);
    a2.grow(2 * MIB).await.unwrap();
    assert_eq!(qa.reserved(), 4 * MIB);
    stop.store(true, Ordering::Release);
    assert_eq!(driver.await.unwrap().unwrap(), MIB);
    assert_eq!(
        other_signal.spill_requested(),
        None,
        "query B was not asked"
    );
}

#[tokio::test(start_paused = true)]
async fn slack_is_reclaimed_before_anything_spills() {
    let root = new_root(64 * MIB);
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let a = Spiller::new(&qa, "sort", 1); // granted a whole quantum
    let _filler = holder(&qb, "filler", 62 * MIB);
    let mut b = holder(&qb, "agg", 0);
    assert!(b.try_grow(MIB + MIB / 2).is_err());
    b.grow(MIB + MIB / 2).await.unwrap();
    assert_eq!(root.reserved(), 62 * MIB + MIB + MIB / 2 + 1);
    assert_eq!(a.signal.spill_requested(), None);
    assert_eq!(root.stats(), stats(0, 0, 1));
}

/// A spill smaller than the quantum still counts: the victim is trimmed to
/// its exact usage when its ticket completes.
#[tokio::test(start_paused = true)]
async fn sub_quantum_spills_free_exactly_what_they_spilled() {
    let root = new_root(64 * MIB);
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let (stop, driver) = Spiller::new(&qa, "sort", 60 * MIB).spawn(COOPERATIVE);
    let mut b = holder(&qb, "agg", 0);
    b.grow(4 * MIB + MIB / 2).await.unwrap();
    stop.store(true, Ordering::Release);
    assert_eq!(driver.await.unwrap().unwrap(), MIB / 2);
    assert_eq!(root.stats(), stats(0, 1, 1));
}

/// A consumer whose operator dropped its spill hook is never asked, so no
/// round waits on it.
#[tokio::test(start_paused = true)]
async fn finished_consumers_are_not_asked_to_spill() {
    let root = new_root(8 * MIB);
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let finished = Spiller::new(&qa, "sort", 6 * MIB);
    let Spiller {
        signal,
        reservation,
        ..
    } = finished;
    drop(signal); // operator done, consumer still registered
    let released = tokio::spawn({
        let qa = Arc::clone(&qa);
        async move {
            while qa.check_aborted().is_ok() {
                sleep(Duration::from_millis(1)).await;
            }
            drop(reservation);
        }
    });
    let mut b = holder(&qb, "agg", 0);
    let start = Instant::now();
    b.grow(4 * MIB).await.unwrap();
    assert!(start.elapsed() < TIMEOUT, "no wait on a dead consumer");
    assert_eq!(root.stats(), stats(1, 0, 1));
    released.await.unwrap();
}

/// Rounds in disjoint subtrees (two queries at their own limits) run at the
/// same time.
#[tokio::test(start_paused = true)]
async fn rounds_in_disjoint_queries_run_concurrently() {
    let root = new_root(64 * MIB);
    let slow = Behaviour {
        spills: true,
        react_after: Duration::from_secs(1),
        spill_takes: Duration::ZERO,
    };
    let mut requests = Vec::new();
    for name in ["query-a", "query-b"] {
        let q = root.child(name, Some(64 * MIB / 2));
        let (_stop, _driver) = Spiller::new(&q, "sort", 30 * MIB).spawn(slow);
        let mut r = holder(&q, "agg", 0);
        requests.push(tokio::spawn(
            async move { r.grow(4 * MIB).await.map(|()| r) },
        ));
    }
    let start = Instant::now();
    for r in requests {
        r.await.unwrap().unwrap();
    }
    let took = start.elapsed();
    assert!(
        took >= Duration::from_secs(1) && took < Duration::from_millis(1500),
        "{took:?}"
    );
}

/// Tickets are awaited together, and any release is noticed at once.
#[tokio::test(start_paused = true)]
async fn tickets_are_awaited_together_and_releases_are_seen() {
    let root = new_root(16 * MIB);
    let slow = Behaviour {
        spills: true,
        react_after: Duration::from_secs(1),
        spill_takes: Duration::ZERO,
    };
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let qc = root.child("query-c", None);
    let (_s1, _d1) = Spiller::new(&qa, "sort", 6 * MIB).spawn(slow);
    let (_s2, _d2) = Spiller::new(&qb, "sort", 6 * MIB).spawn(slow);
    let _rest = holder(&qc, "filler", 4 * MIB);
    let mut r = holder(&qc, "agg", 0);
    let start = Instant::now();
    r.grow(10 * MIB).await.unwrap(); // needs both victims
    let took = start.elapsed();
    assert!(
        took >= Duration::from_secs(1) && took < Duration::from_millis(1500),
        "{took:?}"
    );

    // A victim that never spills, but another query finishing frees memory.
    let tree = new_root(8 * MIB);
    let qa = tree.child("query-a", None);
    let qb = tree.child("query-b", None);
    let qc = tree.child("query-c", None);
    let (_s, _d) = Spiller::new(&qa, "sort", 4 * MIB).spawn(NO_SPILL);
    let finishing = holder(&qc, "result", 4 * MIB);
    tokio::spawn(async move {
        sleep(Duration::from_millis(100)).await;
        drop(finishing);
    });
    let mut r = holder(&qb, "agg", 0);
    let start = Instant::now();
    r.grow(4 * MIB).await.unwrap();
    assert!(
        start.elapsed() < Duration::from_millis(200),
        "no stale wait"
    );
}

/// Victims are chosen by memory held above the fair share, so a small query
/// is not asked while a large one exceeds its share.
#[tokio::test(start_paused = true)]
async fn victims_are_chosen_above_their_fair_share() {
    let root = new_root(64 * MIB);
    let small = root.child("small", None);
    let large = root.child("large", None);
    let requester = root.child("requester", None);
    let a = Spiller::new(&small, "sort", 21 * MIB); // under its share
    let a_signal = Arc::clone(&a.signal);
    let _pinned = holder(&large, "pinned", 30 * MIB);
    let (stop, driver) = Spiller::new(&large, "sort", 13 * MIB).spawn(COOPERATIVE);
    let mut r = holder(&requester, "agg", 0);
    r.grow(MIB).await.unwrap();
    assert_eq!(
        a_signal.spill_requested(),
        None,
        "the small query was not asked"
    );
    stop.store(true, Ordering::Release);
    assert_eq!(driver.await.unwrap().unwrap(), MIB);
    drop(a);
}

/// The arbitrator never asks the requester's own driver; the driver spills
/// its own consumers first through `SpillOwner::reserve`.
#[tokio::test(start_paused = true)]
async fn a_driver_spills_its_own_consumers_before_arbitrating() {
    let tree = new_root(8 * MIB);
    let q = tree.child("query", None);
    let owner = SpillOwner::new();
    let sibling_signal = owner.signal();
    let sibling_id = q.register(
        "join",
        Some(Arc::clone(&sibling_signal) as Arc<dyn Spillable>),
    );
    let sibling = Arc::new(Mutex::new(Reservation::new(Arc::clone(&q), sibling_id)));
    sibling.lock().unwrap().try_grow(6 * MIB).unwrap();
    sibling_signal.set_spillable_bytes(6 * MIB);
    let requester_signal = owner.signal();
    let id = q.register(
        "agg",
        Some(Arc::clone(&requester_signal) as Arc<dyn Spillable>),
    );

    // Plain arbitration does not ask the sibling (same driver): the lone
    // query fails instead of waiting for a request nobody would service.
    let start = Instant::now();
    let e = q.reserve(id, 4 * MIB).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
    assert!(start.elapsed() < TIMEOUT);
    assert_eq!(sibling_signal.spill_requested(), None);

    // The driver protocol spills the sibling locally and succeeds.
    let tree = new_root(8 * MIB);
    let q = tree.child("query", None);
    let owner = SpillOwner::new();
    let sibling_signal = owner.signal();
    let sibling_id = q.register(
        "join",
        Some(Arc::clone(&sibling_signal) as Arc<dyn Spillable>),
    );
    let sibling = Arc::new(Mutex::new(Reservation::new(Arc::clone(&q), sibling_id)));
    sibling.lock().unwrap().try_grow(6 * MIB).unwrap();
    sibling_signal.set_spillable_bytes(6 * MIB);
    let requester_signal = owner.signal();
    let id = q.register(
        "agg",
        Some(Arc::clone(&requester_signal) as Arc<dyn Spillable>),
    );
    let service = || {
        let (signal, sibling) = (Arc::clone(&sibling_signal), Arc::clone(&sibling));
        async move {
            if let Some(target) = signal.spill_requested() {
                let mut r = sibling.lock().unwrap();
                let freed = target.min(r.size());
                r.shrink(freed);
                signal.set_spillable_bytes(r.size());
                signal.complete(freed);
            }
        }
    };
    owner.reserve(&*q, id, 4 * MIB, service).await.unwrap();
    assert_eq!(sibling.lock().unwrap().size(), 4 * MIB);
    assert_eq!(tree.stats(), stats(0, 0, 0), "no arbitration needed");
    q.release(id, 4 * MIB);
}

/// An external consumer: a UDF worker pool reporting its RSS, which sheds
/// memory synchronously (recycling idle workers) when asked.
#[derive(Debug)]
struct WorkerPoolRss {
    rss: Mutex<Option<Reservation>>,
}

impl Spillable for WorkerPoolRss {
    fn spillable_bytes(&self) -> usize {
        self.rss
            .lock()
            .unwrap()
            .as_ref()
            .map_or(0, Reservation::size)
    }
    fn request_spill(&self, target: usize) -> SpillTicket {
        let mut rss = self.rss.lock().unwrap();
        let r = rss.as_mut().unwrap();
        let freed = target.min(r.size());
        r.shrink(freed);
        SpillTicket::ready(freed)
    }
    fn spill_requested(&self) -> Option<usize> {
        None
    }
}

#[tokio::test(start_paused = true)]
async fn external_consumers_shed_but_never_cause_aborts() {
    let root = new_root(8 * MIB);
    let headroom = root.external("udf-workers", 8 * MIB);
    let workers = Arc::new(WorkerPoolRss {
        rss: Mutex::new(None),
    });
    let wid = headroom.register("rss", Some(Arc::clone(&workers) as Arc<dyn Spillable>));
    let mut rss = Reservation::new(Arc::clone(&headroom), wid);
    rss.try_grow(5 * MIB).unwrap();
    assert!(rss.try_grow(4 * MIB).is_err(), "capped by the headroom");
    *workers.rss.lock().unwrap() = Some(rss);
    let q = root.child("query", None);
    let mut prefetch = holder(&q, "scan-prefetch", 3 * MIB);

    // The external pool reserves on the fast path only, even via `reserve`.
    let pid = headroom.register("more-rss", None);
    let e = headroom.reserve(pid, MIB).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
    assert!(q.check_aborted().is_ok());
    assert_eq!(root.stats(), stats(0, 0, 0));

    // A query round asks the external consumer to shed.
    prefetch.grow(MIB).await.unwrap();
    assert_eq!(workers.spillable_bytes(), 4 * MIB);
    assert_eq!(root.stats(), stats(0, 1, 1));
}

#[tokio::test(start_paused = true)]
async fn requests_that_can_never_fit_fail_without_aborting() {
    let root = new_root(4 * MIB);
    let q = root.child("query", None);
    let mut r = holder(&q, "op", 0);
    let e = r.grow(8 * MIB).await.unwrap_err();
    assert_eq!(
        e,
        MemoryError::Exhausted {
            requested: 8 * MIB,
            available: 4 * MIB
        }
    );
    assert!(q.check_aborted().is_ok());
    // Root-level consumers belong to no query and never arbitrate.
    let root_pool: Arc<dyn MemoryPool> = Arc::<TreePool>::clone(&root);
    let _held = holder(&root_pool, "cache", 3 * MIB);
    let mut mine = holder(&root_pool, "buffer", 0);
    assert!(mine.grow(2 * MIB).await.is_err());
    assert_eq!(root.stats(), stats(0, 0, 0));
}

struct ThreadWaker(std::thread::Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Polls `fut` to completion on this thread, with no async runtime.
fn block_on<T>(mut fut: BoxFuture<'_, T>) -> T {
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    loop {
        match fut.as_mut().poll(&mut Context::from_waker(&waker)) {
            Poll::Ready(r) => return r,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// A request that only a query that never releases could satisfy: with the
/// built-in system clock it fails once the release timeout has passed.
fn timeout_scenario(
    run: impl FnOnce(BoxFuture<'_, Result<(), MemoryError>>) -> Result<(), MemoryError>,
) {
    let root = TreePool::with_clock(
        &MemoryConfig {
            limit: Some(8 * MIB),
            quantum: MIB,
            spill_wait_timeout: Duration::from_millis(50),
            abort_release_timeout: Duration::from_millis(100),
            spill_scratch_reserve: 0,
        },
        Arc::new(SystemClock::new()),
    );
    let qa = root.child("query-a", None);
    let qb = root.child("query-b", None);
    let _a = holder(&qa, "stuck", 6 * MIB);
    let id = qb.register("agg", None);
    let start = std::time::Instant::now();
    let result = run(qb.reserve(id, 4 * MIB));
    let waited = start.elapsed();
    assert_eq!(
        result.unwrap_err().error_class(),
        ErrorClass::UnableToAcquireMemory
    );
    assert!(waited >= Duration::from_millis(100), "{waited:?}");
    assert!(waited < Duration::from_secs(10), "{waited:?}");
    assert!(qa.check_aborted().is_err(), "the holder was aborted once");
    assert_eq!(root.stats().aborts, 1);
}

/// Arbitration needs no async runtime: its deadlines use the pool's clock.
#[test]
fn reserve_outside_any_runtime_times_out_correctly() {
    timeout_scenario(block_on);
}

/// A Tokio runtime without its time driver: no panic, correct timeout.
#[test]
fn reserve_on_a_runtime_without_timers_times_out_correctly() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    timeout_scenario(|fut| rt.block_on(fut));
}
