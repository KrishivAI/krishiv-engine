// SPDX-License-Identifier: Apache-2.0
//! Loom models of reserve/release accounting and the cooperative spill
//! protocol (request, ack, give-up). Bounded for CI:
//! `RUSTFLAGS="--cfg loom" cargo test -p krishiv-memory --lib --release`.

use std::sync::Arc;
use std::time::Duration;

use loom::future::block_on;
use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use loom::thread;

use crate::arbitrate::race_abort;
use crate::{
    BoxFuture, Clock, MemoryConfig, MemoryError, MemoryPool, SpillOwner, SpillSignal, Spillable,
    TreePool,
};

fn model(f: impl Fn() + Sync + Send + 'static) {
    let mut b = loom::model::Builder::new();
    b.preemption_bound = Some(3);
    b.check(f);
}

/// Timeouts never fire in the models: progress must never depend on one.
#[derive(Debug)]
struct NeverClock;

impl Clock for NeverClock {
    fn now(&self) -> std::time::Instant {
        std::time::Instant::now()
    }

    fn sleep_until(&self, _deadline: std::time::Instant) -> BoxFuture<'static, ()> {
        Box::pin(std::future::pending())
    }
}

fn pool(limit: usize, quantum: usize) -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit: Some(limit),
            quantum,
            spill_wait_timeout: Duration::from_secs(1),
            abort_release_timeout: Duration::from_secs(1),
        },
        Arc::new(NeverClock),
    )
}

/// Two queries race for the last quantum: exactly one wins, the loser's
/// partial grant is rolled back, and everything returns to zero.
#[test]
fn racing_reservations_respect_the_limit() {
    model(|| {
        let root = pool(4, 4);
        let a = root.child_pool("a", None);
        let b = root.child_pool("b", None);
        let (ia, ib) = (a.register("a", None), b.register("b", None));
        let t = {
            let a = Arc::clone(&a);
            thread::spawn(move || a.try_reserve(ia, 3).is_ok())
        };
        let ok_b = b.try_reserve(ib, 3).is_ok();
        let ok_a = t.join().unwrap();
        assert!(ok_a ^ ok_b, "exactly one fits");
        // A 4-byte limit caps the quantum at 1 byte: grants are exact.
        assert_eq!((root.reserved(), a.reserved() + b.reserved()), (3, 3));
        a.release(ia, 3);
        b.release(ib, 3);
        assert_eq!(root.reserved(), 0);
    });
}

/// Grow and release on one consumer from two threads: the quantised grant
/// ends identical in every interleaving.
#[test]
fn concurrent_grow_and_release_on_one_consumer() {
    model(|| {
        let root = pool(8, 4);
        let id = root.register("op", None);
        root.try_reserve(id, 3).unwrap();
        let t = {
            let root = Arc::clone(&root);
            thread::spawn(move || root.try_reserve(id, 3).unwrap())
        };
        root.release(id, 2);
        t.join().unwrap();
        assert_eq!(root.reserved(), 4);
        root.release(id, 4);
        assert_eq!(root.reserved(), 0);
    });
}

/// Two reservations race through three-level chains for the last bytes of
/// the root: the loser passes the read-only check, is refused by the root's
/// CAS and rolls back its task and query, leaving nothing behind.
#[test]
fn racing_deep_reservations_roll_back() {
    model(|| {
        let root = pool(4, 1);
        let ta = root.child_pool("qa", None).child_pool("task", Some(4));
        let tb = root.child_pool("qb", None).child_pool("task", Some(4));
        let (ia, ib) = (ta.register("a", None), tb.register("b", None));
        let t = {
            let ta = Arc::clone(&ta);
            thread::spawn(move || ta.try_reserve(ia, 3).is_ok())
        };
        let ok_b = tb.try_reserve(ib, 3).is_ok();
        let ok_a = t.join().unwrap();
        assert!(ok_a ^ ok_b, "exactly one fits");
        assert_eq!(root.reserved(), 3);
        assert_eq!(ta.reserved() + tb.reserved(), 3, "the loser rolled back");
    });
}

/// Request → owner observes the flag at a batch boundary → spills on its
/// own thread → completes the ticket the requester awaits.
#[test]
fn spill_request_is_serviced_by_the_owner() {
    model(|| {
        let root = pool(8, 1);
        let signal = Arc::new(SpillSignal::new());
        let id = root.register("op", Some(Arc::clone(&signal) as Arc<dyn Spillable>));
        root.try_reserve(id, 8).unwrap();
        signal.set_spillable_bytes(8);
        let owner = {
            let (root, signal) = (Arc::clone(&root), Arc::clone(&signal));
            thread::spawn(move || {
                loop {
                    if let Some(target) = signal.spill_requested() {
                        root.release(id, target);
                        signal.set_spillable_bytes(8 - target);
                        signal.complete(target);
                        return;
                    }
                    thread::yield_now();
                }
            })
        };
        let freed = loom::future::block_on(signal.request_spill(5));
        assert_eq!(freed, 5);
        owner.join().unwrap();
        assert_eq!((root.reserved(), signal.spillable_bytes()), (3, 3));
    });
}

/// A request racing a completion is never lost: each ticket is either
/// complete or its target is still flagged for the owner.
#[test]
fn concurrent_requests_coalesce_without_loss() {
    model(|| {
        let signal = Arc::new(SpillSignal::new());
        let requester = {
            let signal = Arc::clone(&signal);
            thread::spawn(move || signal.request_spill(3))
        };
        let owner = {
            let signal = Arc::clone(&signal);
            thread::spawn(move || {
                if let Some(t) = signal.spill_requested() {
                    signal.complete(t);
                }
            })
        };
        let mine = signal.request_spill(7);
        let theirs = requester.join().unwrap();
        owner.join().unwrap();
        for (ticket, target) in [(&mine, 7), (&theirs, 3)] {
            assert!(
                ticket.try_get().is_some() || signal.spill_requested().is_some_and(|t| t >= target)
            );
        }
        signal.complete(0);
        assert!(mine.try_get().is_some() && theirs.try_get().is_some());
    });
}

/// The arbitrator gives up (drops its ticket, as on timeout) while the owner
/// completes: no stale completion leaks into the next request.
#[test]
fn giving_up_races_completion() {
    model(|| {
        let signal = Arc::new(SpillSignal::new());
        let ticket = signal.request_spill(4);
        let owner = {
            let signal = Arc::clone(&signal);
            thread::spawn(move || signal.complete(4))
        };
        drop(ticket);
        owner.join().unwrap();
        assert_eq!(signal.spill_requested(), None);
        let next = signal.request_spill(1);
        assert_eq!(next.try_get(), None);
        signal.complete(1);
        assert_eq!(next.try_get(), Some(1));
    });
}

/// An abort racing reservations through a task pool: the abort bit lives
/// in the query's counter word, so once the abort is done the query's held
/// bytes never grow, even for a reservation already past the flag check.
#[test]
fn an_abort_is_a_fence_for_grants() {
    model(|| {
        let root = pool(64, 1);
        let q = root.child_pool("q", None);
        let task = q.child_pool("task", None);
        let id = task.register("op", None);
        let grower = {
            let task = Arc::clone(&task);
            thread::spawn(move || {
                let _ = task.try_reserve(id, 1);
                let _ = task.try_reserve(id, 1);
            })
        };
        q.node().abort(MemoryError::Aborted { held: 0 });
        let at_abort = q.reserved();
        grower.join().unwrap();
        assert_eq!(q.reserved(), at_abort, "no grant after the abort");
        assert_eq!(root.reserved(), at_abort);
        assert_eq!(task.reserved(), at_abort);
    });
}

/// A release racing a round that waits for memory always wakes it: the
/// waiter count, the two SeqCst fences and the event of `ReleaseSignal`.
/// The memory is an independent counter read with plain loads, as in
/// production (the node counters use RMW reads under loom, which would add
/// synchronisation and hide a missing fence).
#[test]
fn a_release_wakes_a_waiting_round() {
    model(|| {
        let signal = Arc::new(crate::pool::ReleaseSignal::default());
        let held = Arc::new(AtomicUsize::new(4));
        let releaser = {
            let (signal, held) = (Arc::clone(&signal), Arc::clone(&held));
            thread::spawn(move || {
                held.fetch_sub(4, Ordering::Release);
                signal.released();
            })
        };
        let waiter = signal.wait();
        loop {
            let listener = waiter.listen();
            if held.load(Ordering::Acquire) == 0 {
                break;
            }
            block_on(listener);
        }
        drop(waiter);
        releaser.join().unwrap();
    });
}

/// A task waiting for round admission (or anything) returns as soon as its
/// query is aborted.
#[test]
fn race_abort_never_misses_an_abort() {
    model(|| {
        let root = pool(4, 1);
        let q = root.child_pool("q", None);
        let aborter = {
            let q = Arc::clone(&q);
            thread::spawn(move || q.node().abort(MemoryError::Aborted { held: 3 }))
        };
        let r = block_on(race_abort(
            q.node().group.as_deref(),
            std::future::pending::<()>(),
        ));
        assert_eq!(r, Err(MemoryError::Aborted { held: 3 }));
        aborter.join().unwrap();
    });
}

/// Unregistering a consumer while it reserves never leaks a grant.
#[test]
fn unregister_races_a_reservation() {
    model(|| {
        let root = pool(8, 1);
        let q = root.child_pool("q", None);
        let id = q.register("op", None);
        let reserver = {
            let q = Arc::clone(&q);
            thread::spawn(move || {
                let _ = q.try_reserve(id, 4);
            })
        };
        q.unregister(id);
        reserver.join().unwrap();
        assert_eq!(root.reserved(), 0);
    });
}

/// Overlapping rounds (root and a query) never run together.
#[test]
fn overlapping_rounds_are_exclusive() {
    model(|| {
        let root = pool(8, 1);
        let q = root.child_pool("q", None);
        let active = Arc::new(AtomicUsize::new(0));
        let run = |pool: Arc<TreePool>, active: Arc<AtomicUsize>| {
            let node = Arc::clone(pool.node());
            let guard = block_on(node.shared.rounds.enter(Arc::clone(&node)));
            assert_eq!(
                active.fetch_add(1, Ordering::SeqCst),
                0,
                "two rounds at once"
            );
            active.fetch_sub(1, Ordering::SeqCst);
            drop(guard);
        };
        let other = {
            let (q, active) = (Arc::clone(&q), Arc::clone(&active));
            thread::spawn(move || run(q, active))
        };
        run(Arc::clone(&root), Arc::clone(&active));
        other.join().unwrap();
    });
}

/// Rounds in disjoint subtrees run at the same time: each waits, inside its
/// round, for the other to be inside its own.
#[test]
fn disjoint_rounds_run_concurrently() {
    model(|| {
        let root = pool(8, 1);
        let (q1, q2) = (root.child_pool("q1", None), root.child_pool("q2", None));
        let inside = Arc::new([AtomicBool::new(false), AtomicBool::new(false)]);
        let run = move |pool: Arc<TreePool>, inside: Arc<[AtomicBool; 2]>, me: usize| {
            let node = Arc::clone(pool.node());
            let guard = block_on(node.shared.rounds.enter(Arc::clone(&node)));
            inside[me].store(true, Ordering::SeqCst);
            while !inside[1 - me].load(Ordering::SeqCst) {
                thread::yield_now();
            }
            drop(guard);
        };
        let other = {
            let inside = Arc::clone(&inside);
            thread::spawn(move || run(q2, inside, 1))
        };
        run(q1, inside, 0);
        other.join().unwrap();
    });
}

/// A full arbitration round: the requester's `reserve` asks the other
/// query's consumer to spill, its driver services the request on its own
/// thread, and the round grants the memory (timeouts never fire in the
/// model, so success cannot come from one).
#[test]
fn an_arbitration_round_is_served_by_a_spill() {
    let mut b = loom::model::Builder::new();
    b.preemption_bound = Some(2);
    b.check(|| {
        let root = pool(8, 1);
        let q1 = root.child_pool("q1", None);
        let q2 = root.child_pool("q2", None);
        let owner = SpillOwner::new();
        let signal = owner.signal();
        let x = q1.register("x", Some(Arc::clone(&signal) as Arc<dyn Spillable>));
        thread::yield_now();
        q1.try_reserve(x, 8).unwrap();
        signal.set_spillable_bytes(8);
        let w = q2.register("w", None);
        let requester = {
            let q2 = Arc::clone(&q2);
            thread::spawn(move || block_on(q2.reserve(w, 4)))
        };
        // The victim's driver (this thread) sleeps on its owner's waker
        // until a request arrives, then spills on its own thread.
        loop {
            let woken = owner.requested();
            if let Some(target) = signal.spill_requested() {
                q1.release(x, target);
                signal.complete(target);
                break;
            }
            block_on(woken);
        }
        requester.join().unwrap().unwrap();
        assert_eq!((q1.reserved(), q2.reserved(), root.reserved()), (4, 4, 8));
        assert_eq!(root.stats().aborts, 0);
    });
}
