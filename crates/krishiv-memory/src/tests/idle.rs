// SPDX-License-Identifier: Apache-2.0
//! Idle finished outputs ([`Spillable::idle`]) are asked first: before the
//! requester's driver spills its own consumers, and before other victims in
//! a round (study 30, M2).

use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

/// A consumer that frees what it holds at once when asked.
struct Held {
    idle: bool,
    held: std::sync::Mutex<Option<Reservation>>,
    asked: AtomicUsize,
}

impl Held {
    fn register(q: &Arc<TreePool>, name: &str, idle: bool, bytes: usize) -> Arc<Self> {
        let me = Arc::new(Self {
            idle,
            held: std::sync::Mutex::new(None),
            asked: AtomicUsize::new(0),
        });
        let id = q.register(name, Some(Arc::clone(&me) as Arc<dyn Spillable>));
        let mut r = Reservation::new(Arc::clone(q) as Arc<dyn MemoryPool>, id);
        assert!(r.try_grow(bytes).is_ok());
        *me.lock() = Some(r);
        me
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Reservation>> {
        self.held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn size(&self) -> usize {
        self.lock().as_ref().map_or(0, Reservation::size)
    }
}

impl Spillable for Held {
    fn spillable_bytes(&self) -> usize {
        self.size()
    }

    fn request_spill(&self, target: usize) -> SpillTicket {
        self.asked.fetch_add(1, Ordering::Relaxed);
        let mut held = self.lock();
        let freed = held.as_ref().map_or(0, |r| target.min(r.size()));
        if let Some(r) = held.as_mut() {
            r.shrink(freed);
        }
        SpillTicket::ready(freed)
    }

    fn spill_requested(&self) -> Option<usize> {
        None
    }

    fn idle(&self) -> bool {
        self.idle
    }
}

/// The requester's driver holds spillable state, and an idle output of the
/// same query holds the rest of the pool: the idle output spills, the
/// driver's own state stays.
#[tokio::test(start_paused = true)]
async fn an_idle_output_spills_before_the_requesters_own_state() {
    for idle in [true, false] {
        let root = root(Some(8 * Q), Q);
        let q = root.child_pool("q", None);
        let output = Held::register(&q, "output", idle, 4 * Q);
        let owner = SpillOwner::new();
        let signal = owner.signal();
        let x_id = q.register("x", Some(Arc::clone(&signal) as Arc<dyn Spillable>));
        let x = Arc::new(std::sync::Mutex::new(Reservation::new(
            Arc::clone(&q) as Arc<dyn MemoryPool>,
            x_id,
        )));
        x.lock().unwrap().try_grow(4 * Q).unwrap();
        signal.set_spillable_bytes(4 * Q);
        let y_id = q.register("y", Some(owner.signal() as Arc<dyn Spillable>));
        owner_reserve(&owner, &q, y_id, 2 * Q, &signal, &x)
            .await
            .unwrap();
        let sizes = (output.size(), x.lock().unwrap().size());
        if idle {
            assert_eq!(sizes, (2 * Q, 4 * Q), "the idle output spilled first");
        } else {
            assert_eq!(sizes, (4 * Q, 2 * Q), "the driver spilled its own state");
        }
    }
}

/// Another query's idle output is not this query's to spill first.
#[tokio::test(start_paused = true)]
async fn only_the_requesters_query_has_its_idle_outputs_spilled_first() {
    let root = root(Some(8 * Q), Q);
    let (q, other) = (root.child_pool("q", None), root.child_pool("other", None));
    let output = Held::register(&other, "output", true, 4 * Q);
    let id = q.register("x", None);
    assert_eq!(q.spill_idle(id, 2 * Q).await, 0);
    assert_eq!(output.asked.load(Ordering::Relaxed), 0);
    // The requester's own query: asked for the shortfall only.
    let mine = Held::register(&q, "mine", true, 3 * Q);
    assert_eq!(q.spill_idle(id, 2 * Q).await, 2 * Q);
    assert_eq!(mine.size(), Q);
    // Nothing asked of anyone for nothing.
    assert_eq!(q.spill_idle(id, 0).await, 0);
    assert_eq!(mine.asked.load(Ordering::Relaxed), 1);
}

/// In a round, an idle output is asked before a larger running consumer of
/// the same query.
#[tokio::test(start_paused = true)]
async fn a_round_asks_an_idle_output_before_a_larger_running_consumer() {
    let root = root(Some(8 * Q), Q);
    let q = root.child_pool("q", None);
    let running = Held::register(&q, "running", false, 5 * Q);
    let output = Held::register(&q, "output", true, 3 * Q);
    let id = q.register("r", None);
    q.reserve(id, 2 * Q).await.unwrap();
    assert_eq!((output.size(), running.size()), (Q, 5 * Q));
    assert_eq!(running.asked.load(Ordering::Relaxed), 0);
}

/// An idle output whose spill write is still running, with nothing more to
/// give.
struct Writing {
    signal: SpillSignal,
}

impl Spillable for Writing {
    fn spillable_bytes(&self) -> usize {
        self.signal.spillable_bytes()
    }

    fn request_spill(&self, target: usize) -> SpillTicket {
        let ticket = self.signal.request_spill(target);
        // Its rows are taken for the write.
        self.signal.set_spillable_bytes(0);
        ticket
    }

    fn spill_requested(&self) -> Option<usize> {
        self.signal.spill_requested()
    }

    fn idle(&self) -> bool {
        true
    }
}

/// While a driver waits for an idle output's write, another driver's round
/// asks one of the waiting driver's consumers to spill: the waiting driver
/// services it, as it would in arbitration, so the round is not left to
/// time out.
#[tokio::test(start_paused = true)]
async fn a_driver_waiting_for_an_idle_spill_services_its_own_consumers() {
    let root = root(Some(8 * Q), Q);
    let q = root.child_pool("q", None);
    let output = Arc::new(Writing {
        signal: SpillSignal::new(),
    });
    let out_id = q.register("output", Some(Arc::clone(&output) as Arc<dyn Spillable>));
    let mut rows = Reservation::new(Arc::clone(&q) as Arc<dyn MemoryPool>, out_id);
    rows.try_grow(4 * Q).unwrap();
    output.signal.set_spillable_bytes(4 * Q);

    let a = Arc::new(SpillOwner::new());
    let signal = a.signal();
    let x_id = q.register("x", Some(Arc::clone(&signal) as Arc<dyn Spillable>));
    let x = Arc::new(std::sync::Mutex::new(Reservation::new(
        Arc::clone(&q) as Arc<dyn MemoryPool>,
        x_id,
    )));
    x.lock().unwrap().try_grow(3 * Q).unwrap();
    signal.set_spillable_bytes(3 * Q);
    let y_id = q.register("y", Some(a.signal() as Arc<dyn Spillable>));
    let waiting = tokio::spawn({
        let (a, q, signal, x) = (
            Arc::clone(&a),
            Arc::clone(&q),
            Arc::clone(&signal),
            Arc::clone(&x),
        );
        async move { owner_reserve(&a, &q, y_id, 2 * Q, &signal, &x).await }
    });
    while output.signal.spill_requested().is_none() {
        tokio::task::yield_now().await;
    }
    // Another driver needs 2 MiB of the 1 MiB free: only x can give it.
    let b = SpillOwner::new();
    let z_id = q.register("z", Some(b.signal() as Arc<dyn Spillable>));
    let started = tokio::time::Instant::now();
    q.reserve(z_id, 2 * Q).await.unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "the round waited for a driver that did not service its consumers"
    );
    assert!(x.lock().unwrap().size() < 3 * Q, "x spilled for the round");
    // The write ends: the waiting request goes on.
    rows.shrink(2 * Q);
    output.signal.complete(2 * Q);
    waiting.await.unwrap().unwrap();
}
