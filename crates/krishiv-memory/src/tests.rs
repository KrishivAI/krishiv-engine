// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;
use std::time::Duration;

use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use proptest::prelude::*;

use super::*;
use crate::pool::{Usage, round_up};

const Q: usize = 1 << 20;

fn root(limit: Option<usize>, quantum: usize) -> Arc<TreePool> {
    TreePool::new(&MemoryConfig {
        limit,
        quantum,
        spill_wait_timeout: Duration::from_secs(2),
        abort_release_timeout: Duration::from_secs(30),
    })
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
    assert!(
        MemoryError::NoRuntime
            .to_string()
            .starts_with("[INTERNAL_ERROR_MEMORY] memory arbitration must run on a Tokio runtime")
    );
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
fn zero_byte_requests_still_see_unregister() {
    let root = root(None, Q);
    let q = root.child_pool("q", None);
    let id = q.register("op", None);
    q.try_reserve(id, 0).unwrap();
    let other = root.child_pool("q2", None);
    let id2 = other.register("op", None);
    other.unregister(id2);
    assert_eq!(
        other.try_reserve(id2, 0),
        Err(MemoryError::UnknownConsumer { id: id2 })
    );
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
    Unregister(usize),
    DropQuery(usize),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0..4usize).prop_map(Op::New),
        (0..8usize, 0..300_000usize).prop_map(|(r, n)| Op::Grow(r, n)),
        (0..8usize, 0..300_000usize).prop_map(|(r, n)| Op::Shrink(r, n)),
        (0..8usize).prop_map(Op::Drop),
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
    /// sum of live reservations exactly). Covers unregister and dropping a
    /// query pool.
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
        q.node().try_reserve_for(&consumer, 1),
        Err(crate::pool::Denied::Error(
            MemoryError::UnknownConsumer { .. }
        ))
    ));
    assert_eq!(root.reserved(), 0);
}
