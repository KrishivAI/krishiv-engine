// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;
use std::time::Duration;

use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use proptest::prelude::*;

use super::*;
use crate::pool::{Usage, round_up};

const Q: usize = 1 << 20;

/// Arbitration deadlines on Tokio's (pausable) clock, for deterministic tests.
#[derive(Debug)]
struct TokioClock;

impl Clock for TokioClock {
    fn now(&self) -> std::time::Instant {
        tokio::time::Instant::now().into_std()
    }

    fn sleep_until(
        &self,
        deadline: std::time::Instant,
    ) -> Result<BoxFuture<'static, ()>, MemoryError> {
        Ok(Box::pin(tokio::time::sleep_until(
            tokio::time::Instant::from_std(deadline),
        )))
    }
}

fn root(limit: Option<usize>, quantum: usize) -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit,
            quantum,
            spill_wait_timeout: Duration::from_secs(2),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: 0,
        },
        Arc::new(TokioClock),
    )
}

fn usage(pool: &TreePool, id: ConsumerId) -> Usage {
    pool.node()
        .consumer(id)
        .map_or_else(|_| Usage::default(), |c| *sync::lock(&c.usage))
}

#[test]
fn config_entries_feed_memory_config() {
    let (c, _) = Config::from_pairs([
        ("krishiv.memory.limit", "64m"),
        ("krishiv.memory.reservationQuantum", "4k"),
        ("krishiv.memory.spillWaitTimeout", "250ms"),
    ])
    .unwrap();
    let m = MemoryConfig::from_config(&c);
    assert_eq!(m.limit, Some(64 << 20));
    assert_eq!(m.quantum, 4096);
    assert_eq!(m.spill_wait_timeout, Duration::from_millis(250));
    let d = MemoryConfig::default();
    assert_eq!((d.limit, d.quantum), (default_limit(), Q));
    assert!(
        d.limit.is_some(),
        "an unset limit is derived, never unbounded"
    );
    assert_eq!(d.spill_wait_timeout, Duration::from_secs(2));
    assert_eq!(d.abort_release_timeout, Duration::from_secs(30));
    assert_eq!(d.spill_scratch_reserve, DEFAULT_SPILL_SCRATCH_RESERVE);
    let (c, _) = Config::from_pairs([("krishiv.memory.spillScratchReserve", "0")]).unwrap();
    assert_eq!(MemoryConfig::from_config(&c).spill_scratch_reserve, 0);
}

fn floored(limit: usize, reserve: usize) -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit: Some(limit),
            quantum: 4096,
            spill_wait_timeout: Duration::from_secs(2),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: reserve,
        },
        Arc::new(TokioClock),
    )
}

/// Ordinary reservations stop at the limit minus the floor; spill scratch
/// may go up to the limit, exactly (no quantum), and the floor is free
/// again once the scratch is released.
#[test]
fn the_spill_scratch_floor_is_kept_for_spill_scratch() {
    const MIB: usize = 1 << 20;
    let root = floored(8 * MIB, 512 << 10);
    let q = root.child_pool("q", None);
    let (a, w) = (q.register("table", None), q.register("writer", None));
    let e = q.try_reserve(a, 8 * MIB).unwrap_err();
    assert_eq!(
        e,
        MemoryError::Exhausted {
            requested: 8 * MIB,
            available: 8 * MIB - (512 << 10),
        }
    );
    q.try_reserve(a, 8 * MIB - (512 << 10)).unwrap();
    assert!(
        q.try_reserve(a, 1).is_err(),
        "the floor is not ordinary memory"
    );
    q.try_reserve_scratch(w, 100).unwrap();
    assert_eq!(usage(&q, w).granted, 100, "scratch is granted exactly");
    let mut writer = Reservation::new(Arc::clone(&q) as Arc<dyn MemoryPool>, w).for_spill_scratch();
    writer.try_grow((512 << 10) - 100).unwrap();
    assert_eq!(root.reserved(), 8 * MIB);
    assert!(writer.try_grow(1).is_err(), "never past the limit");
    drop(writer);
    q.release(w, 100);
    assert_eq!(root.reserved(), 8 * MIB - (512 << 10));
    // A plain reservation of the same consumer stays below the floor.
    let mut plain = Reservation::new(Arc::clone(&q) as Arc<dyn MemoryPool>, w);
    assert!(plain.try_grow(1).is_err());
    q.release(a, 8 * MIB);
    assert_eq!(root.reserved(), 0);
}

/// The floor is at most an eighth of each pool's limit, applies to every
/// limited pool on the way up, and a request larger than what ordinary
/// reservations can ever hold fails at once, without arbitration.
#[tokio::test(start_paused = true)]
async fn the_floor_is_capped_and_applies_to_every_limited_pool() {
    let root = floored(1 << 20, 1 << 20);
    let q = root.child_pool("q", Some(1 << 19));
    let id = q.register("op", None);
    q.try_reserve(id, (1 << 19) - (1 << 16)).unwrap();
    assert!(q.try_reserve(id, 1).is_err(), "q's floor is 64 KiB");
    q.try_reserve_scratch(id, 1 << 16).unwrap();
    assert_eq!(q.reserved(), 1 << 19);
    q.release(id, 1 << 19);
    let e = q.reserve(id, 1 << 19).await.unwrap_err();
    assert!(matches!(e, MemoryError::Exhausted { .. }));
    assert_eq!(root.stats().rounds, 0, "never fits: no round");
    // No limit, no floor; a zero reserve disables it.
    let unbounded = TreePool::new(&MemoryConfig {
        limit: None,
        ..MemoryConfig::default()
    });
    let u = unbounded.register("op", None);
    unbounded.try_reserve(u, usize::MAX >> 2).unwrap();
    let off = floored(1 << 20, 0);
    let o = off.register("op", None);
    off.try_reserve(o, 1 << 20).unwrap();
}

/// `begin` marks a request's tickets as started; dropping the service
/// guard completes it with nothing freed, so no ticket is left pending.
#[test]
fn a_begun_spill_is_started_and_its_guard_always_completes() {
    let signal = SpillSignal::new();
    assert!(signal.begin().is_none(), "nothing requested");
    let ticket = signal.request_spill(10);
    assert!(!ticket.started());
    let service = signal.begin().unwrap();
    assert!(ticket.started());
    let joined = signal.request_spill(20);
    assert!(joined.started(), "a request during a spill joins it");
    assert_eq!(service.target(), 10);
    drop(service);
    assert_eq!((ticket.try_get(), joined.try_get()), (Some(0), Some(0)));
    assert_eq!(signal.spill_requested(), None);
    let next = signal.request_spill(5);
    assert!(!next.started(), "a new request starts unstarted");
    signal.begin().unwrap().complete(5);
    assert_eq!(next.try_get(), Some(5));
    assert!(next.started() && SpillTicket::ready(1).started());
}

/// The trait's defaults: scratch is an ordinary reservation and a
/// spill-only reserve is the fast path, for pools without arbitration.
#[tokio::test(start_paused = true)]
async fn memory_pool_defaults_for_scratch_and_spill_only() {
    #[derive(Debug)]
    struct Fixed(Arc<TreePool>);
    impl MemoryPool for Fixed {
        fn child(&self, name: &str, limit: Option<usize>) -> Arc<dyn MemoryPool> {
            self.0.child(name, limit)
        }
        fn register(&self, name: &str, spill: Option<Arc<dyn Spillable>>) -> ConsumerId {
            self.0.register(name, spill)
        }
        fn unregister(&self, id: ConsumerId) {
            self.0.unregister(id);
        }
        fn try_reserve(&self, id: ConsumerId, bytes: usize) -> Result<(), MemoryError> {
            self.0.try_reserve(id, bytes)
        }
        fn reserve<'a>(
            &'a self,
            id: ConsumerId,
            bytes: usize,
        ) -> BoxFuture<'a, Result<(), MemoryError>> {
            self.0.reserve(id, bytes)
        }
        fn release(&self, id: ConsumerId, bytes: usize) {
            self.0.release(id, bytes);
        }
        fn transfer(
            &self,
            from: ConsumerId,
            to: ConsumerId,
            bytes: usize,
        ) -> Result<(), MemoryError> {
            self.0.transfer(from, to, bytes)
        }
        fn reserved(&self) -> usize {
            self.0.reserved()
        }
        fn limit(&self) -> Option<usize> {
            self.0.limit()
        }
        fn check_aborted(&self) -> Result<(), MemoryError> {
            self.0.check_aborted()
        }
    }
    let pool = Fixed(floored(1 << 20, 1 << 17));
    let id = pool.register("op", None);
    pool.try_reserve_scratch(id, (1 << 20) - (1 << 17)).unwrap();
    assert!(
        pool.try_reserve_scratch(id, 1).is_err(),
        "no floor by default"
    );
    assert!(pool.reserve_without_abort(id, 1).await.is_err());
    pool.release(id, 1 << 20);
    pool.reserve_without_abort(id, 1).await.unwrap();
    assert_eq!(pool.reserved(), 4096);
}

#[test]
fn round_up_to_the_quantum() {
    assert_eq!(round_up(0, Q), 0);
    assert_eq!(round_up(1, Q), Q);
    assert_eq!(round_up(Q, Q), Q);
    assert_eq!(round_up(Q + 1, Q), 2 * Q);
    assert_eq!(round_up(7, 1), 7);
    assert_eq!(round_up(7, 0), 7);
    // Overflow keeps the exact size instead of wrapping.
    assert_eq!(round_up(usize::MAX - 1, Q), usize::MAX - 1);
}

#[test]
fn grants_whole_quanta_and_grows_locally_within_one() {
    let root = root(None, Q);
    let q = root.child_pool("q", None);
    let id = q.register("op", None);
    q.try_reserve(id, 1).unwrap();
    assert_eq!(root.reserved(), Q);
    assert_eq!((usage(&q, id).used, usage(&q, id).granted), (1, Q));
    q.try_reserve(id, Q - 1).unwrap(); // fills the quantum: no pool change
    assert_eq!(root.reserved(), Q);
    q.try_reserve(id, 1).unwrap(); // crosses it: one more quantum
    assert_eq!(root.reserved(), 2 * Q);
    assert_eq!(q.reserved(), 2 * Q);
}

#[test]
fn falls_back_to_the_exact_size_at_the_limit() {
    let limit = 64 * Q + Q / 2;
    let root = root(Some(limit), Q);
    let id = root.register("op", None);
    root.try_reserve(id, 64 * Q + 10).unwrap();
    assert_eq!(usage(&root, id).granted, 64 * Q + 10);
    assert_eq!(root.reserved(), 64 * Q + 10);
    assert!(root.try_reserve(id, Q).is_err());
    assert_eq!(
        root.reserved(),
        64 * Q + 10,
        "a failed request changes nothing"
    );
}

#[test]
fn release_keeps_at_most_one_quantum_of_slack() {
    let root = root(None, Q);
    let id = root.register("op", None);
    root.try_reserve(id, 3 * Q).unwrap();
    root.release(id, Q / 2);
    let u = usage(&root, id);
    assert_eq!((u.used, u.granted), (3 * Q - Q / 2, 3 * Q));
    root.release(id, 2 * Q);
    let u = usage(&root, id);
    assert_eq!((u.used, u.granted), (Q / 2, Q));
    root.release(id, usize::MAX); // clamped
    assert_eq!(root.reserved(), 0);
}

#[test]
fn trim_reclaims_the_slack() {
    let root = root(None, Q);
    let id = root.register("op", None);
    root.try_reserve(id, 10).unwrap();
    let c = root.node().consumer(id).unwrap();
    assert_eq!(c.trim(root.node()), Q - 10);
    assert_eq!(root.reserved(), 10);
    // A release never grows the grant back to a whole quantum.
    root.release(id, 5);
    assert_eq!(root.reserved(), 10);
    root.release(id, 5);
    assert_eq!(root.reserved(), 0);
}

#[test]
fn unregister_and_drop_return_everything() {
    let root = root(None, Q);
    let q = root.child("q", None);
    let task = q.child("task", Some(256 * Q));
    let a = task.register("a", None);
    let b = task.register("b", None);
    task.try_reserve(a, 10).unwrap();
    task.try_reserve(b, Q + 1).unwrap();
    assert_eq!(root.reserved(), 3 * Q);
    task.unregister(a);
    assert_eq!(root.reserved(), 2 * Q);
    let e = task.try_reserve(a, 1).unwrap_err();
    assert_eq!(e, MemoryError::UnknownConsumer { id: a });
    assert_eq!(e.error_class(), ErrorClass::InternalErrorMemory);
    task.release(a, 10); // ignored
    drop(task);
    drop(q);
    assert_eq!(
        root.reserved(),
        0,
        "dropping a pool returns its consumers' memory"
    );
}

#[test]
fn reservation_is_raii() {
    let root = root(Some(4 * Q), 1);
    let pool: Arc<dyn MemoryPool> = root.child("q", None);
    let id = pool.register("op", None);
    let mut r = Reservation::new(Arc::clone(&pool), id);
    r.try_grow(100).unwrap();
    r.try_grow(50).unwrap();
    r.shrink(30);
    assert_eq!((r.size(), r.consumer()), (120, id));
    assert_eq!(root.reserved(), 120);
    r.shrink(1000);
    assert_eq!((r.size(), root.reserved()), (0, 0));
    r.try_grow(10).unwrap();
    let mut r2 = Reservation::new(pool, id);
    r2.try_grow(5).unwrap();
    drop(r);
    assert_eq!(root.reserved(), 5);
    drop(r2);
    assert_eq!(root.reserved(), 0);
}

#[test]
fn tiny_limits_fail_with_unable_to_acquire_memory() {
    let root = root(Some(1024), 1);
    let q = root.child("q", None);
    let id = q.register("op", None);
    let e = q.try_reserve(id, 2048).unwrap_err();
    assert_eq!(
        e,
        MemoryError::Exhausted {
            requested: 2048,
            available: 1024
        }
    );
    let k = KrishivError::from(&e);
    assert_eq!(k.error_class(), ErrorClass::UnableToAcquireMemory);
    assert_eq!(k.sql_state(), Some("53200"));
    assert_eq!(
        e.to_string(),
        "[UNABLE_TO_ACQUIRE_MEMORY] Unable to acquire 2048 bytes of memory, got 0. SQLSTATE: 53200"
    );
    let mut r = Reservation::new(q, id);
    assert!(r.try_grow(usize::MAX).is_err());
    r.try_grow(1024).unwrap();
    assert!(r.try_grow(1).is_err());
    assert_eq!(r.size(), 1024);
}

#[test]
fn error_classes_and_messages() {
    let aborted = MemoryError::Aborted { held: 42 };
    assert_eq!(
        aborted.error_class(),
        ErrorClass::KrishivQueryAbortedOutOfMemory
    );
    assert_eq!(
        KrishivError::from(aborted.clone()).error_class().name(),
        "KRISHIV_QUERY_ABORTED_OUT_OF_MEMORY"
    );
    assert!(aborted.to_string().contains("it held 42 bytes"));
    let unknown = MemoryError::UnknownConsumer { id: ConsumerId(7) };
    assert_eq!(
        unknown.to_string(),
        "[INTERNAL_ERROR_MEMORY] memory consumer 7 is not registered with this pool SQLSTATE: XX000"
    );
}

#[test]
fn spill_requests_coalesce_and_complete_together() {
    let s = SpillSignal::new();
    assert_eq!(s.spill_requested(), None);
    s.set_spillable_bytes(100);
    assert_eq!(s.spillable_bytes(), 100);
    let t1 = s.request_spill(10);
    let t2 = s.request_spill(30);
    let t3 = s.request_spill(20);
    assert_eq!(s.spill_requested(), Some(30), "the largest target wins");
    assert_eq!(t1.try_get(), None);
    s.complete(40);
    assert_eq!(s.spill_requested(), None);
    assert_eq!([t1.try_get(), t2.try_get(), t3.try_get()], [Some(40); 3]);
    // A later request is a new generation.
    let t4 = s.request_spill(5);
    assert_eq!(t4.try_get(), None);
    drop(s);
    assert_eq!(t4.try_get(), Some(0), "a dropped owner completes with 0");
    assert_eq!(SpillTicket::ready(3).try_get(), Some(3));
    assert_eq!(SpillSignal::new().request_spill(0).try_get(), Some(0));
}

#[test]
fn quantum_is_capped_by_small_limits() {
    let root = root(Some(64 * Q), Q);
    let q = root.child_pool("q", Some(640 * 1024));
    let id = q.register("op", None);
    q.try_reserve(id, 1).unwrap();
    assert_eq!(usage(&q, id).granted, 10 * 1024, "1/64 of the query limit");
    let big = root.register("root-level", None);
    root.try_reserve(big, 1).unwrap();
    assert_eq!(usage(&root, big).granted, Q);
}

#[test]
fn ids_from_another_tree_are_unknown() {
    let (a, b) = (root(None, Q), root(None, Q));
    let id_a = a.register("a", None);
    let _id_b = b.register("b", None);
    assert_eq!(
        b.try_reserve(id_a, 1),
        Err(MemoryError::UnknownConsumer { id: id_a })
    );
    assert_eq!(b.reserved(), 0);
}

#[test]
fn zero_byte_requests_still_see_abort_and_unregister() {
    let root = root(None, Q);
    let q = root.child_pool("q", None);
    let id = q.register("op", None);
    q.try_reserve(id, 0).unwrap();
    q.node().abort(MemoryError::Aborted { held: 0 });
    assert_eq!(q.try_reserve(id, 0), Err(MemoryError::Aborted { held: 0 }));
    let other = root.child_pool("q2", None);
    let id2 = other.register("op", None);
    other.unregister(id2);
    assert_eq!(
        other.try_reserve(id2, 0),
        Err(MemoryError::UnknownConsumer { id: id2 })
    );
}

#[test]
fn an_abort_stops_every_grant_and_memory_only_goes_down() {
    let root = root(None, 1);
    let q = root.child_pool("q", None);
    let task = q.child_pool("task", None);
    let id = task.register("op", None);
    task.try_reserve(id, 100).unwrap();
    q.node().abort(MemoryError::Aborted { held: 100 });
    let e = MemoryError::Aborted { held: 100 };
    assert_eq!(task.try_reserve(id, 1), Err(e.clone()));
    assert_eq!(q.check_aborted(), Err(e.clone()));
    assert_eq!(task.check_aborted(), Err(e));
    assert_eq!(
        (task.reserved(), q.reserved(), root.reserved()),
        (100, 100, 100)
    );
    task.release(id, 40);
    assert_eq!(
        (task.reserved(), q.reserved(), root.reserved()),
        (60, 60, 60)
    );
    // The abort bit stops grants even when the group flag was not checked
    // yet (as for a reservation already past that check).
    let other = root.child_pool("other", None);
    let oid = other.register("op", None);
    other.node().abort(MemoryError::Aborted { held: 0 });
    assert!(other.try_reserve(oid, 1).is_err());
    assert_eq!(root.reserved(), 60);
}

#[test]
fn concurrent_reservations_keep_exact_accounting() {
    let root = root(Some(1 << 30), 4096);
    let queries: Vec<Arc<TreePool>> = (0..2)
        .map(|i| root.child_pool(&format!("q{i}"), Some(600 << 20)))
        .collect();
    std::thread::scope(|s| {
        for t in 0..4u64 {
            let q = Arc::clone(&queries[usize::try_from(t % 2).unwrap()]);
            s.spawn(move || {
                let id = q.register("op", None);
                let mut r = Reservation::new(q as Arc<dyn MemoryPool>, id);
                let mut x = t.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
                for _ in 0..2000 {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    let n = usize::try_from(x % 100_000).unwrap();
                    if x % 3 == 0 {
                        r.shrink(n);
                    } else {
                        let _ = r.try_grow(n);
                    }
                }
            });
        }
    });
    assert_eq!(root.reserved(), 0, "every thread's reservation was dropped");
    assert_eq!(queries[0].reserved() + queries[1].reserved(), 0);
}

#[derive(Debug, Clone, Copy)]
enum Op {
    New(usize),
    Grow(usize, usize),
    Shrink(usize, usize),
    Drop(usize),
    Trim(usize),
    Unregister(usize),
    DropQuery(usize),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0..4usize).prop_map(Op::New),
        (0..8usize, 0..300_000usize).prop_map(|(r, n)| Op::Grow(r, n)),
        (0..8usize, 0..300_000usize).prop_map(|(r, n)| Op::Shrink(r, n)),
        (0..8usize).prop_map(Op::Drop),
        (0..4usize).prop_map(Op::Trim),
        (0..4usize).prop_map(Op::Unregister),
        (0..2usize).prop_map(Op::DropQuery),
    ]
}

/// Consumers `c` belong to query `c % 2`; unregistered consumers are `None`.
struct Model {
    root: Arc<TreePool>,
    queries: Vec<Arc<TreePool>>,
    ids: Vec<Option<ConsumerId>>,
    live: Vec<(usize, Reservation)>,
    /// Reservations of unregistered consumers: their release is ignored.
    zombies: Vec<(usize, Reservation)>,
}

const QUERY_LIMITS: [Option<usize>; 2] = [Some(500_000), None];

impl Model {
    fn new(quantum: usize) -> Self {
        let root = root(Some(800_000), quantum);
        let queries = (0..2)
            .map(|q| root.child_pool("q", QUERY_LIMITS[q]))
            .collect();
        let mut m = Self {
            root,
            queries,
            ids: vec![None; 4],
            live: Vec::new(),
            zombies: Vec::new(),
        };
        for c in 0..4 {
            m.ids[c] = Some(m.queries[c % 2].register("c", None));
        }
        m
    }

    fn apply(&mut self, op: Op) {
        match op {
            Op::New(c) => {
                if let Some(id) = self.ids[c] {
                    let pool = Arc::clone(&self.queries[c % 2]) as Arc<dyn MemoryPool>;
                    self.live.push((c, Reservation::new(pool, id)));
                }
            }
            Op::Grow(r, n) => {
                if let Some((_, res)) = self.live.get_mut(r) {
                    let before = res.size();
                    if res.try_grow(n).is_ok() {
                        assert_eq!(res.size(), before + n);
                    }
                }
            }
            Op::Shrink(r, n) => {
                if let Some((_, res)) = self.live.get_mut(r) {
                    res.shrink(n);
                }
            }
            Op::Drop(r) => {
                if r < self.live.len() {
                    self.live.swap_remove(r);
                }
            }
            Op::Trim(c) => {
                let q = &self.queries[c % 2];
                if let Some(Ok(consumer)) = self.ids[c].map(|id| q.node().consumer(id)) {
                    consumer.trim(q.node());
                }
            }
            Op::Unregister(c) => {
                if let Some(id) = self.ids[c].take() {
                    self.queries[c % 2].unregister(id);
                    let (dead, live): (Vec<_>, Vec<_>) = std::mem::take(&mut self.live)
                        .into_iter()
                        .partition(|(rc, _)| *rc == c);
                    self.live = live;
                    self.zombies.extend(dead);
                }
            }
            Op::DropQuery(q) => {
                self.live.retain(|(c, _)| c % 2 != q);
                self.zombies.retain(|(c, _)| c % 2 != q);
                self.queries[q] = self.root.child_pool("q", QUERY_LIMITS[q]);
                for c in (q..4).step_by(2) {
                    self.ids[c] = Some(self.queries[q].register("c", None));
                }
            }
        }
    }
}

proptest! {
    /// Accounted bytes never go negative, never exceed a limit, and always
    /// equal the grants of live consumers; each consumer's `used` equals the
    /// sum of its live reservations (with quantum 1, pool totals equal the
    /// sum of live reservations exactly). Covers trim, unregister and
    /// dropping a query pool.
    #[test]
    fn accounting_matches_live_reservations(
        quantum in prop_oneof![Just(1usize), Just(64), Just(5000)],
        ops in proptest::collection::vec(op(), 1..80),
    ) {
        let mut m = Model::new(quantum);
        for op in ops {
            m.apply(op);
            let mut granted_by_query = [0usize; 2];
            let mut live_total = 0;
            for c in 0..4 {
                let q = &m.queries[c % 2];
                let Some(id) = m.ids[c] else { continue };
                let consumer = q.node().consumer(id).unwrap();
                let u = usage(q, id);
                let held: usize = m.live.iter().filter(|(rc, _)| *rc == c).map(|(_, r)| r.size()).sum();
                prop_assert_eq!(u.used, held);
                prop_assert!(u.used <= u.granted && u.granted <= round_up(u.used, consumer.quantum));
                prop_assert!(consumer.quantum <= quantum);
                granted_by_query[c % 2] += u.granted;
                live_total += held;
            }
            prop_assert_eq!(m.queries[0].reserved(), granted_by_query[0]);
            prop_assert_eq!(m.queries[1].reserved(), granted_by_query[1]);
            prop_assert_eq!(m.root.reserved(), granted_by_query[0] + granted_by_query[1]);
            prop_assert!(m.queries[0].reserved() <= 500_000 && m.root.reserved() <= 800_000);
            if quantum == 1 {
                prop_assert_eq!(m.root.reserved(), live_total);
            }
        }
        m.live.clear();
        m.zombies.clear();
        prop_assert_eq!(m.root.reserved(), 0);
    }
}

/// A driver queued for arbitration is woken by a spill request from a round
/// admitted after it queued, and services it, so that round never times out.
#[tokio::test(start_paused = true)]
async fn a_queued_driver_services_a_later_rounds_request() {
    let root = root(Some(8 * Q), Q);
    let q1 = root.child_pool("q1", None);
    let q2 = root.child_pool("q2", None);
    let d1 = SpillOwner::new();
    let x_signal = d1.signal();
    let x_id = q1.register("x", Some(Arc::clone(&x_signal) as Arc<dyn Spillable>));
    let x = Arc::new(std::sync::Mutex::new(Reservation::new(
        Arc::clone(&q1) as Arc<dyn MemoryPool>,
        x_id,
    )));
    x.lock().unwrap().try_grow(6 * Q).unwrap();
    let y_id = q1.register("y", Some(d1.signal() as Arc<dyn Spillable>));

    // An active round at the root makes the next rounds queue behind it.
    let slot = root
        .node()
        .shared
        .rounds
        .enter(Arc::clone(root.node()))
        .await;
    let w_id = q2.register("w", None);
    let q2c = Arc::clone(&q2);
    let d2 = tokio::spawn(async move { q2c.reserve(w_id, 4 * Q).await });
    tokio::task::yield_now().await;
    let service = {
        let (signal, x) = (Arc::clone(&x_signal), Arc::clone(&x));
        move || {
            let (signal, x) = (Arc::clone(&signal), Arc::clone(&x));
            async move {
                if let Some(target) = signal.spill_requested() {
                    let mut r = x.lock().unwrap();
                    let freed = target.min(r.size());
                    r.shrink(freed);
                    signal.set_spillable_bytes(r.size());
                    signal.complete(freed);
                }
            }
        }
    };
    let q1c = Arc::clone(&q1);
    let d1_task = tokio::spawn(async move { d1.reserve(&*q1c, y_id, 3 * Q, service).await });
    tokio::task::yield_now().await; // D1 found nothing to self-spill and queued
    x_signal.set_spillable_bytes(6 * Q);
    let start = tokio::time::Instant::now();
    drop(slot); // D2 is admitted and asks X (D1's) to spill
    d2.await.unwrap().unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "serviced, no timeout"
    );
    // D1's own round then asks X too (its own driver, asked last, which
    // it services while it waits) for the 3 Q it needs.
    d1_task.await.unwrap().unwrap();
    assert_eq!(x.lock().unwrap().size(), Q);
    assert_eq!(root.stats().aborts, 0);
}

#[test]
fn debug_output_names_the_pieces() {
    let root = root(Some(64 * Q), Q);
    let q = root.child_pool("q-debug", None);
    let id = q.register("op-debug", None);
    let r = Reservation::new(Arc::clone(&q) as Arc<dyn MemoryPool>, id);
    assert!(format!("{root:?}").contains("root"));
    assert!(format!("{q:?}").contains("q-debug"));
    assert!(format!("{r:?}").contains("Reservation"));
    assert!(format!("{:?}", q.node().consumer(id).unwrap()).contains("op-debug"));
    assert!(format!("{:?}", SpillTicket::ready(1)).contains("Some(1)"));
    let owner = SpillOwner::new();
    assert_ne!(owner.id(), SpillOwner::new().id());
    assert!(format!("{owner:?}").contains("SpillOwner"));
}

#[test]
fn dropping_an_empty_pool_and_unregistering_mid_reserve() {
    let root = root(None, Q);
    drop(root.child_pool("empty", None));
    assert_eq!(root.reserved(), 0);
    let q = root.child_pool("q", None);
    let id = q.register("op", None);
    let consumer = q.node().consumer(id).unwrap();
    q.unregister(id);
    // A reservation that looked the consumer up before it was unregistered.
    assert!(matches!(
        q.node()
            .try_reserve_for(&consumer, 1, crate::pool::Class::Ordinary),
        Err(crate::pool::Denied::Error(
            MemoryError::UnknownConsumer { .. }
        ))
    ));
    assert_eq!(root.reserved(), 0);
}

#[test]
fn external_pools_hang_off_the_root_wherever_they_are_made() {
    let root = root(Some(64 * Q), Q);
    let task = root.child_pool("q", None).child_pool("task", None);
    let ext = task.external("prefetch", 8 * Q);
    let id = ext.register("buffers", None);
    ext.try_reserve(id, Q).unwrap();
    assert_eq!((root.reserved(), task.reserved()), (Q, 0));
    assert!(ext.check_aborted().is_ok());
}

/// Runs `owner.reserve` with a service that spills `x` by the requested
/// amount.
async fn owner_reserve(
    owner: &SpillOwner,
    pool: &Arc<TreePool>,
    id: ConsumerId,
    bytes: usize,
    signal: &Arc<SpillSignal>,
    x: &Arc<std::sync::Mutex<Reservation>>,
) -> Result<(), MemoryError> {
    let service = || {
        let (signal, x) = (Arc::clone(signal), Arc::clone(x));
        async move {
            if let Some(target) = signal.spill_requested() {
                let mut r = x.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let freed = target.min(r.size());
                r.shrink(freed);
                signal.complete(freed);
            }
        }
    };
    owner.reserve(&**pool, id, bytes, service).await
}

#[tokio::test(start_paused = true)]
async fn spill_owner_reserve_paths() {
    let root = root(Some(64 * Q), Q);
    let q = root.child_pool("q", None);
    let owner = SpillOwner::new();
    let signal = owner.signal();
    let x_id = q.register("x", Some(Arc::clone(&signal) as Arc<dyn Spillable>));
    let x = Arc::new(std::sync::Mutex::new(Reservation::new(
        Arc::clone(&q) as Arc<dyn MemoryPool>,
        x_id,
    )));
    let y_id = q.register("y", Some(owner.signal() as Arc<dyn Spillable>));

    // Fast path.
    owner_reserve(&owner, &q, y_id, Q, &signal, &x)
        .await
        .unwrap();
    // A non-retryable error is returned as is.
    let other = root.child_pool("other", None);
    let foreign = other.register("z", None);
    assert!(matches!(
        owner_reserve(&owner, &q, foreign, Q, &signal, &x).await,
        Err(MemoryError::UnknownConsumer { .. })
    ));
    // A request already pending on an own consumer with no spillable bytes
    // advertised is serviced before arbitrating.
    x.lock().unwrap().try_grow(63 * Q).unwrap();
    let woken = owner.requested();
    drop(signal.request_spill(Q));
    woken.await; // the request woke the owner
    assert!(owner.pending());
    owner_reserve(&owner, &q, y_id, Q, &signal, &x)
        .await
        .unwrap();
    assert_eq!(x.lock().unwrap().size(), 62 * Q);
    assert!(!owner.pending());
}

/// Root-level consumers take part in a root round's plan without belonging
/// to a query.
#[tokio::test(start_paused = true)]
async fn root_level_consumers_are_counted_but_never_aborted() {
    let root = root(Some(64 * Q), Q);
    let cache = root.register("cache", None);
    root.try_reserve(cache, 32 * Q).unwrap();
    let a = root.child_pool("a", None);
    let ida = a.register("held", None);
    a.try_reserve(ida, 30 * Q).unwrap();
    let b = root.child_pool("b", None);
    let idb = b.register("agg", None);
    // `a` never releases, so the request fails after the release timeout.
    let e = b.reserve(idb, 4 * Q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
    assert!(
        a.check_aborted().is_err(),
        "the query, not the cache, was aborted"
    );
    assert_eq!(root.stats().aborts, 1);
}

static SPAWNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Fails the first attempt to start the timer thread, then starts it.
fn spawn_fails_once(run: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
    if SPAWNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
        return Err(std::io::Error::other("thread limit reached"));
    }
    std::thread::Builder::new().spawn(run).map(drop)
}

/// If the clock cannot time waits, the request fails with a typed error
/// before anyone is asked to spill or aborted, and the next request retries
/// starting the timer.
#[test]
fn an_unavailable_timer_fails_the_request_without_hurting_anyone() {
    let clock = crate::clock::SystemClock::with_spawn(spawn_fails_once);
    let root = TreePool::with_clock(
        &MemoryConfig {
            limit: Some(8 * Q),
            quantum: Q,
            spill_wait_timeout: Duration::from_millis(20),
            abort_release_timeout: Duration::from_millis(20),
            spill_scratch_reserve: 0,
        },
        Arc::new(clock),
    );
    let qa = root.child_pool("a", None);
    let signal = Arc::new(SpillSignal::new());
    let a = qa.register("sort", Some(Arc::clone(&signal) as Arc<dyn Spillable>));
    qa.try_reserve(a, 6 * Q).unwrap();
    signal.set_spillable_bytes(6 * Q);
    let qb = root.child_pool("b", None);
    let b = qb.register("agg", None);
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let e = rt.block_on(qb.reserve(b, 4 * Q)).unwrap_err();
    assert!(matches!(e, MemoryError::TimerUnavailable { .. }));
    assert_eq!(e.error_class(), ErrorClass::InternalErrorMemory);
    assert!(e.to_string().contains("arbitration timer unavailable"));
    assert_eq!(signal.spill_requested(), None, "nobody was asked to spill");
    assert!(qa.check_aborted().is_ok(), "nobody was aborted");
    assert_eq!(root.stats(), ArbitrationStats::default());

    // The next request starts the timer and arbitrates normally: A is
    // asked to spill, never answers, and is aborted at the deadline.
    let e = rt.block_on(qb.reserve(b, 4 * Q)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
    assert_eq!(signal.spill_requested(), Some(2 * Q));
    assert!(qa.check_aborted().is_err());
    assert!(SPAWNS.load(std::sync::atomic::Ordering::SeqCst) >= 2);
}

#[test]
fn consumer_handles_unregister_after_their_last_reservation() {
    let root = root(Some(4 * Q), 1);
    let q: Arc<dyn MemoryPool> = root.child("q", None);
    let h = ConsumerHandle::register(Arc::clone(&q), "op", None);
    let id = h.id();
    assert!(format!("{h:?}").contains("credit"));
    let mut r = h.reservation();
    r.try_grow(100).unwrap();
    drop(h);
    // The reservation keeps the consumer registered.
    r.try_grow(10).unwrap();
    assert_eq!(root.reserved(), 110);
    drop(r);
    assert_eq!(root.reserved(), 0);
    assert_eq!(
        q.try_reserve(id, 1),
        Err(MemoryError::UnknownConsumer { id })
    );
}

#[test]
fn shrink_to_credit_hands_bytes_over_without_the_pool() {
    let root = root(Some(1000), 1);
    let q: Arc<dyn MemoryPool> = root.child("q", None);
    let h = ConsumerHandle::register(Arc::clone(&q), "op", None);
    let mut headroom = h.reservation();
    headroom.try_grow(600).unwrap();
    headroom.shrink_to_credit(400);
    assert_eq!(
        (headroom.size(), h.credit(), root.reserved()),
        (200, 400, 600)
    );
    // Another reservation of the same consumer takes the credit first.
    let mut scratch = h.reservation();
    scratch.try_grow(500).unwrap(); // 400 credit + 100 from the pool
    assert_eq!((h.credit(), root.reserved()), (0, 700));
    // Clamped to the size; a plain reservation just shrinks.
    headroom.shrink_to_credit(10_000);
    assert_eq!((headroom.size(), h.credit()), (0, 200));
    h.release_credit();
    let mut plain = Reservation::new(Arc::clone(&q), h.id());
    plain.try_grow(50).unwrap();
    plain.shrink_to_credit(50);
    assert_eq!((plain.size(), h.credit(), root.reserved()), (0, 0, 500));
    drop((scratch, headroom, plain, h));
    assert_eq!(root.reserved(), 0);
}

#[test]
fn a_reservation_releasing_to_credit_keeps_its_bytes_with_the_consumer() {
    let root = root(Some(1000), 1);
    let q: Arc<dyn MemoryPool> = root.child("q", None);
    let h = ConsumerHandle::register(Arc::clone(&q), "op", None);
    let mut scratch = h.reservation().releasing_to_credit();
    scratch.try_grow(300).unwrap();
    scratch.shrink(100);
    assert_eq!(
        (scratch.size(), h.credit(), root.reserved()),
        (200, 100, 300)
    );
    // Another consumer cannot take what was freed; this one regrows from it.
    scratch.try_grow(100).unwrap();
    assert_eq!((h.credit(), root.reserved()), (0, 300));
    drop(scratch);
    assert_eq!((h.credit(), root.reserved()), (300, 300));
    h.release_credit();
    assert_eq!(root.reserved(), 0);
    // Without a handle it releases to the pool as usual.
    let mut plain = Reservation::new(Arc::clone(&q), h.id()).releasing_to_credit();
    plain.try_grow(10).unwrap();
    drop(plain);
    assert_eq!((h.credit(), root.reserved()), (0, 0));
}

#[test]
fn grant_credit_is_taken_once_and_refunded_on_failure() {
    let root = root(Some(1000), 1);
    let q: Arc<dyn MemoryPool> = root.child("q", None);
    let h = ConsumerHandle::register(Arc::clone(&q), "op", None);
    // The driver reserved 300 bytes for a NeedMemory retry.
    q.try_reserve(h.id(), 300).unwrap();
    h.add_credit(300);
    let mut r = h.reservation();
    r.try_grow(200).unwrap(); // all from credit
    assert_eq!((r.size(), h.credit(), root.reserved()), (200, 100, 300));
    r.try_grow(150).unwrap(); // 100 credit + 50 from the pool
    assert_eq!((h.credit(), root.reserved()), (0, 350));
    h.add_credit(0);
    q.try_reserve(h.id(), 100).unwrap();
    h.add_credit(100);
    // A refused request puts its credit back.
    assert!(r.try_grow(2000).is_err());
    assert_eq!((h.credit(), root.reserved()), (100, 450));
    h.release_credit();
    h.release_credit();
    assert_eq!((h.credit(), root.reserved()), (0, 350));
    assert_eq!(h.peak(), 350);
    drop(r);
    assert_eq!(root.reserved(), 0);
    let mut r2 = h.reservation();
    r2.try_grow(10).unwrap();
    assert_eq!(h.peak(), 350, "the peak is a high-water mark");
}

#[tokio::test(start_paused = true)]
async fn aborted_completes_when_the_query_is_aborted() {
    let root = root(Some(8 * Q), Q);
    let a = root.child_pool("a", None);
    let ida = a.register("held", None);
    a.try_reserve(ida, 6 * Q).unwrap();
    let b = root.child_pool("b", None);
    let idb = b.register("agg", None);
    let waiter = tokio::spawn({
        let a = Arc::clone(&a);
        async move { a.aborted().await }
    });
    // `a` holds the most and never spills, so the arbitrator aborts it.
    let request = tokio::spawn({
        let b = Arc::clone(&b);
        async move { b.reserve(idb, 4 * Q).await }
    });
    let e = waiter.await.unwrap();
    assert_eq!(e.error_class(), ErrorClass::KrishivQueryAbortedOutOfMemory);
    assert_eq!(a.aborted().await, e, "already aborted: completes at once");
    a.unregister(ida);
    request.await.unwrap().unwrap();
    // Pools outside a query never complete.
    let ext = root.external("prefetch", Q);
    let never = tokio::time::timeout(Duration::from_secs(60), ext.aborted()).await;
    assert!(never.is_err());
    let none = tokio::time::timeout(Duration::from_secs(60), root.aborted()).await;
    assert!(none.is_err());
}

#[tokio::test(start_paused = true)]
async fn grow_draws_credit_and_refunds_it_when_cancelled() {
    let root = root(Some(10 * Q), 1);
    let q = root.child_pool("q", None);
    let h = ConsumerHandle::register(Arc::clone(&q) as Arc<dyn MemoryPool>, "op", None);
    q.try_reserve(h.id(), Q).unwrap();
    h.add_credit(Q);
    let mut r = h.reservation();
    r.grow(Q).await.unwrap(); // credit only
    q.try_reserve(h.id(), Q).unwrap();
    h.add_credit(Q);
    r.grow(2 * Q).await.unwrap(); // credit + pool
    assert_eq!((r.size(), h.credit(), root.reserved()), (3 * Q, 0, 3 * Q));
    // Another query holds the rest and never releases, so `grow` waits.
    let other = root.child_pool("other", None);
    let held = other.register("held", None);
    other.try_reserve(held, 6 * Q).unwrap();
    q.try_reserve(h.id(), Q).unwrap();
    h.add_credit(Q);
    let pending = tokio::time::timeout(Duration::from_millis(10), r.grow(4 * Q)).await;
    assert!(pending.is_err(), "the request is still waiting");
    assert_eq!(h.credit(), Q, "dropping the wait refunded the credit");
    // `other` is aborted but never releases, so the request fails.
    let e = r.grow(4 * Q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
    assert_eq!(h.credit(), Q);
}

#[tokio::test(start_paused = true)]
async fn reserve_for_parks_the_grant_as_credit() {
    let root = root(Some(4 * Q), 1);
    let q = root.child_pool("q", None);
    let owner = SpillOwner::new();
    let h = ConsumerHandle::register(
        Arc::clone(&q) as Arc<dyn MemoryPool>,
        "op",
        Some(owner.signal() as Arc<dyn Spillable>),
    );
    owner.reserve_for(&h, Q, || async {}).await.unwrap();
    assert_eq!((h.credit(), root.reserved()), (Q, Q));
    let e = owner.reserve_for(&h, 8 * Q, || async {}).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
    assert_eq!(h.credit(), Q, "a refused request parks nothing");
    // Without abort: self-spill and spill requests, never an abort.
    let other = root.child_pool("other", None);
    let hog = other.register("hog", None);
    other.try_reserve(hog, 2 * Q).unwrap();
    let e = owner
        .reserve_for_without_abort(&h, 2 * Q, || async {})
        .await
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
    assert_eq!((h.credit(), root.stats().aborts), (Q, 0));
    assert!(other.check_aborted().is_ok() && q.check_aborted().is_ok());
    other.release(hog, 2 * Q);
    owner
        .reserve_for_without_abort(&h, 2 * Q, || async {})
        .await
        .unwrap();
    assert_eq!(h.credit(), 3 * Q);
}

#[test]
fn transfer_moves_bytes_between_consumers_of_one_pool() {
    let root = root(Some(1 << 20), 4096);
    let q = root.child("q", None);
    let a = ConsumerHandle::register(Arc::clone(&q), "a", None);
    let b = ConsumerHandle::register(Arc::clone(&q), "b", None);
    let mut r = a.reservation();
    r.try_grow(10_000).unwrap();
    let before = q.reserved();
    r.transfer_to(&b).unwrap();
    assert_eq!(
        (r.consumer(), r.size(), q.reserved()),
        (b.id(), 10_000, before)
    );
    assert_eq!(b.peak(), 10_000);
    drop(a);
    r.shrink(4_000);
    drop(r);
    assert_eq!(q.reserved(), 0);
    // A consumer of another pool is refused; the reservation is unchanged.
    let other = ConsumerHandle::register(root.child("q2", None), "c", None);
    let mut r = b.reservation();
    r.try_grow(100).unwrap();
    assert!(matches!(
        r.transfer_to(&other),
        Err(MemoryError::UnknownConsumer { .. })
    ));
    assert_eq!(r.consumer(), b.id());
}

/// Bytes transferred to a spillable consumer are what the arbitrator sees:
/// it asks that consumer to spill them (`min(spillable, used)`).
#[tokio::test(start_paused = true)]
async fn the_arbitrator_sees_transferred_bytes() {
    let root = root(Some(64 * Q), Q);
    let q = root.child_pool("q", Some(16 * Q));
    let pool = Arc::clone(&q) as Arc<dyn MemoryPool>;
    let gone = ConsumerHandle::register(Arc::clone(&pool), "finished-driver", None);
    let mut state = gone.reservation();
    state.try_grow(10 * Q).unwrap();
    let owner = SpillOwner::new();
    let signal = owner.signal();
    let live = ConsumerHandle::register(
        Arc::clone(&pool),
        "finalizing-driver",
        Some(Arc::clone(&signal) as Arc<dyn Spillable>),
    );
    state.transfer_to(&live).unwrap();
    drop(gone);
    signal.set_spillable_bytes(10 * Q);
    let w = q.register("w", None);
    let qc = Arc::clone(&q);
    let waiting = tokio::spawn(async move { qc.reserve(w, 10 * Q).await });
    tokio::task::yield_now().await;
    let target = signal.spill_requested().expect("asked to spill");
    assert!(target >= 4 * Q, "{target}");
    state.shrink(target);
    signal.complete(target);
    waiting.await.unwrap().unwrap();
}
