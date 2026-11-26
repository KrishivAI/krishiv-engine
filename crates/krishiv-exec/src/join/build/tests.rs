// SPDX-License-Identifier: Apache-2.0
//! Tests of the hash join build sink: the runtime filter partials (hashes shed,
//! "Bloom lost" flags, reservations), the spillable signal and idle writes.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::DataType;
use krishiv_io::{ColumnPath, FilterId, FilterSnapshot, KeyFilterOptions, RuntimeFilter};
use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};

use super::spilled::events;
use super::{Partials, Writing, publish};
use crate::join::grace::Grace;
use crate::join::{FilterPublisher, JoinRuntimeFilter};
use crate::sync::{Event, Mutex, lock};
use crate::values::{LocalValues, Producer};

/// A write of an idle build's partitions that ends any other way than
/// with its files (cancelled with its query) leaves the grace phase
/// failed, never short of rows; one that lands hands over its files.
#[test]
fn an_unfinished_write_of_an_idle_build_loses_its_partitions() {
    let grace = Arc::new(Mutex::new(Grace::new(2)));
    let changed = Arc::new(Event::default());
    let writing = |parts| Writing {
        grace: Arc::clone(&grace),
        changed: Arc::clone(&changed),
        parts,
        landed: false,
    };
    lock(&grace).writing = 2;
    writing(1).landed(Vec::new());
    assert_eq!(lock(&grace).writing, 1);
    assert!(lock(&grace).lost.is_none());
    drop(writing(1));
    let g = lock(&grace);
    assert_eq!(g.writing, 0);
    assert!(g.lost.is_some(), "the rows were lost");
}

fn reservation(limit: usize) -> Reservation {
    let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig {
        limit: Some(limit),
        quantum: 1,
        // The whole limit for the partials: the spill-scratch floor
        // would hide part of it from these exact-size cases.
        spill_scratch_reserve: 0,
        ..MemoryConfig::default()
    });
    let id = pool.register("partials", None);
    Reservation::new(pool, id)
}

/// Publishes into `f`'s cell, as its only producer.
fn publisher(f: &JoinRuntimeFilter) -> FilterPublisher {
    let values = LocalValues::new();
    assert!(values.adopt_filter(Arc::clone(&f.filter)).is_ok());
    FilterPublisher {
        values: Arc::new(values),
        producer: Producer::default(),
    }
}

fn filter() -> JoinRuntimeFilter {
    JoinRuntimeFilter {
        key: 0,
        filter: Arc::new(RuntimeFilter::new(
            FilterId(1),
            ColumnPath::new("k"),
            DataType::Int64,
        )),
        options: KeyFilterOptions {
            max_values: 0,
            ..KeyFilterOptions::default()
        },
        column: None,
    }
}

/// Memory for the distinct keys but not for their Bloom hashes too: the
/// hashes go and the exact set stays (`drop_hashes`, review M1), with
/// the reservation equal to what the partial holds. Were the hashes and
/// the keys dropped together (`drop_bloom`), only the range would be
/// published.
#[test]
fn without_room_for_the_hashes_the_exact_set_stays() {
    let mut f = filter();
    f.options.max_values = 100;
    let shared = vec![f.clone()];
    // Sparse (not a dense range, which would publish as the range
    // whatever is kept), and more than the `IN` list holds (100), so
    // `drop_bloom` would drop them where `drop_hashes` keeps them.
    let keys: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..2000).map(|i: i64| i * 1_000_003),
    ));
    let sizes = |hashes: bool| {
        let mut b = krishiv_io::KeyFilterBuilder::new(DataType::Int64, f.options).unwrap();
        b.add(keys.as_ref(), None).unwrap();
        if !hashes {
            b.drop_hashes();
        }
        b.heap_bytes()
    };
    let (keys_only, with_hashes) = (sizes(false), sizes(true));
    assert!(with_hashes > keys_only, "no hashes to drop");
    let mut p = Partials {
        builders: vec![krishiv_io::KeyFilterBuilder::new(
            DataType::Int64,
            f.options,
        )],
        // Room for the keys and half the hashes: never both.
        memory: vec![reservation(keys_only + (with_hashes - keys_only) / 2)],
        bloom_lost: Arc::default(),
        spilled: Arc::default(),
    };
    p.add_keys(&shared, &[Arc::clone(&keys)], None);
    assert_eq!(p.memory[0].size(), p.heap_bytes(), "reservation != heap");
    assert_eq!(p.heap_bytes(), keys_only, "the keys were dropped too");
    publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    let FilterSnapshot::Final(k) = f.filter.published().snapshot.clone() else {
        unreachable!("not published")
    };
    assert_eq!(
        k.form(),
        krishiv_io::FilterForm::HashSet(2000),
        "no exact set published"
    );
}

/// A build asked to spill sheds its Bloom hashes first: the
/// reservation shrinks to the distinct keys, and the exact set is still
/// published.
#[test]
fn shedding_the_hashes_frees_their_memory_and_keeps_the_keys() {
    let mut f = filter();
    f.options.max_values = 100;
    let shared = vec![f.clone()];
    let keys: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..2000).map(|i: i64| i * 1_000_003),
    ));
    let mut p = Partials {
        builders: vec![krishiv_io::KeyFilterBuilder::new(
            DataType::Int64,
            f.options,
        )],
        memory: vec![reservation(1 << 24)],
        bloom_lost: Arc::default(),
        spilled: Arc::default(),
    };
    p.add_keys(&shared, &[Arc::clone(&keys)], None);
    let before = p.memory[0].size();
    assert_eq!(before, p.heap_bytes(), "hashes not reserved");
    let (freed, moved) = p.shed_hashes();
    assert!(freed > 0 && moved, "nothing shed");
    assert_eq!(p.memory[0].size(), before - freed);
    assert_eq!(p.memory[0].size(), p.heap_bytes(), "reservation != heap");
    assert_eq!(p.shed_hashes(), (0, false), "shed twice");
    publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    let FilterSnapshot::Final(k) = f.filter.published().snapshot.clone() else {
        unreachable!("not published")
    };
    assert_eq!(k.form(), krishiv_io::FilterForm::HashSet(2000));
}

fn flag() -> Arc<[std::sync::atomic::AtomicBool]> {
    Arc::new([std::sync::atomic::AtomicBool::new(false)])
}

/// Two partials of one build: once one sheds its hashes the other stops
/// collecting them at its next batch, and holds only its keys.
#[test]
fn once_one_partial_sheds_the_others_stop_collecting_hashes() {
    let mut f = filter();
    f.options.max_values = 100;
    let shared = vec![f.clone()];
    let keys: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..2000).map(|i: i64| i * 1_000_003),
    ));
    let lost = flag();
    let new = |lost: &Arc<_>| Partials {
        builders: vec![krishiv_io::KeyFilterBuilder::new(
            DataType::Int64,
            f.options,
        )],
        memory: vec![reservation(1 << 24)],
        bloom_lost: Arc::clone(lost),
        spilled: Arc::default(),
    };
    let (mut a, mut b) = (new(&lost), new(&lost));
    let mut other = new(&flag());
    for p in [&mut a, &mut b, &mut other] {
        p.add_keys(&shared, &[Arc::clone(&keys)], None);
        assert!(p.hash_bytes() > 0, "no hashes collected");
    }
    let hashes = a.hash_bytes();
    assert_eq!(a.shed_hashes().0, hashes, "the shed is not the hash bytes");
    assert_eq!(a.hash_bytes(), 0);
    assert!(b.hash_bytes() > 0, "shed before its next batch");
    b.add_keys(&shared, &[Arc::clone(&keys)], None);
    assert_eq!(b.hash_bytes(), 0, "still collecting hashes");
    assert_eq!(b.memory[0].size(), b.heap_bytes(), "reservation != heap");
    // Another build's partial keeps its Bloom.
    other.add_keys(&shared, &[Arc::clone(&keys)], None);
    assert!(other.hash_bytes() > 0);
}

/// A partial holding merged drivers' reservations whose growth is
/// refused drops its hashes and releases the excess from every
/// reservation, not only the first (which once kept the rest held).
#[test]
fn a_refused_merged_partial_releases_from_every_reservation() {
    let mut f = filter();
    f.options.max_values = 100;
    let keys: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..2000).map(|i: i64| i * 1_000_003),
    ));
    let mut builder = krishiv_io::KeyFilterBuilder::new(DataType::Int64, f.options).unwrap();
    builder.add(keys.as_ref(), None).unwrap();
    let heap = builder.heap_bytes();
    let hashes = builder.hash_bytes();
    assert!(hashes > 64, "too few hash bytes");
    // The first reservation is full at 64 bytes; the merged one holds
    // the rest but one byte, so `fit` must grow and is refused.
    let mut first = reservation(64);
    assert!(first.try_grow(64).is_ok());
    let mut merged = reservation(1 << 24);
    assert!(merged.try_grow(heap - 64 - 1).is_ok());
    let mut p = Partials {
        builders: vec![Some(builder)],
        memory: vec![first, merged],
        bloom_lost: flag(),
        spilled: Arc::default(),
    };
    p.fit();
    assert_eq!(p.hash_bytes(), 0, "the hashes stayed");
    let held: usize = p.memory.iter().map(Reservation::size).sum();
    assert_eq!(held, p.heap_bytes(), "reservations != heap");
    assert!(
        p.bloom_lost[0].load(std::sync::atomic::Ordering::Relaxed),
        "the others were not told"
    );
}

/// Through the sink: a build without a spill scope offers its Bloom
/// hashes to the arbitrator (they were not counted before) and sheds
/// them when asked, locally and after `combine` (`spill_global`); the
/// signal follows each step, and a driver that sheds stops the others.
#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "test-only runtime via #[tokio::test]; never runs on a compute worker"
)]
async fn a_build_offers_its_hashes_and_sheds_them_when_asked() {
    use krishiv_memory::{SpillOwner, Spillable as _};

    use crate::join::{HashJoin, JoinSpec, JoinType};

    let mut f = filter();
    f.options.max_values = 100;
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "k",
        DataType::Int64,
        true,
    )]));
    let spec = JoinSpec {
        join_type: JoinType::Inner,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        single_match: false,
        spill: None,
    };
    let join = HashJoin::try_new(spec)
        .and_then(|j| j.with_runtime_filters(vec![f.clone()]))
        .unwrap_or_else(|e| panic!("{e}"));
    let sink = join.build_sink();
    let root = TreePool::new(&MemoryConfig::default());
    let query = crate::QueryContext::new(
        crate::QueryId(1),
        crate::WorkloadClass::Interactive,
        crate::ExecConfig::default(),
        root.child("q", None),
    );
    let exec: Arc<dyn crate::Executor> =
        crate::TokioExecutor::new(&crate::ExecutorConfig::default())
            .unwrap_or_else(|e| panic!("{e}"));
    let ctx = |op: u32| {
        crate::TaskContext::new(
            Arc::clone(&query),
            Arc::clone(&exec),
            SpillOwner::new(),
            crate::OperatorId(op),
            "b",
        )
    };
    let batch = |from: i64| {
        let keys: ArrayRef = Arc::new(Int64Array::from_iter_values(
            (from..from + 2000).map(|i| i * 1_000_003),
        ));
        krishiv_types::batch::Batch::new(
            arrow_array::RecordBatch::try_new(Arc::clone(&schema), vec![keys])
                .unwrap_or_else(|e| panic!("{e}")),
        )
    };
    let hashes = |l: &crate::operator::LocalState| {
        l.downcast_ref::<super::BuildLocal>()
            .map(|l| l.runtime.hash_bytes())
            .unwrap_or_else(|| panic!("not a build local"))
    };
    let offered = |c: &crate::TaskContext| c.consumer().signal().spillable_bytes();

    // Two drivers: the one asked to spill sheds, the other stops.
    let (c1, c2) = (ctx(1), ctx(1));
    let g = sink.global_state(&c1).unwrap_or_else(|e| panic!("{e}"));
    let mut l1 = sink.local_state(&c1, &g).unwrap_or_else(|e| panic!("{e}"));
    let mut l2 = sink.local_state(&c2, &g).unwrap_or_else(|e| panic!("{e}"));
    assert!(sink.sink(batch(0), &mut l1, &g).is_ok());
    assert!(sink.sink(batch(2000), &mut l2, &g).is_ok());
    assert!(hashes(&l1) > 0, "no hashes collected");
    assert_eq!(offered(&c1), hashes(&l1), "hashes not offered");
    let freed = sink
        .spill(&mut l1, 1, &c1)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(freed > 0, "nothing shed");
    assert_eq!(offered(&c1), 0, "offers the shed hashes");
    assert!(sink.sink(batch(4000), &mut l2, &g).is_ok());
    assert_eq!(hashes(&l2), 0, "the other driver still collects hashes");
    assert_eq!(offered(&c2), 0);
    assert!(sink.combine(l1, &g).is_ok());
    assert!(sink.combine(l2, &g).is_ok());
    // No hashes are left to shed; it may still free the merged
    // reservations' slack over the merged keys.
    assert!(sink.spill_global(&g, 1, &c1).await.is_ok());
    {
        let r = lock(&super::state(&g).unwrap_or_else(|e| panic!("{e}")).runtime);
        assert_eq!(r.hash_bytes(), 0);
        let held: usize = r.memory.iter().map(Reservation::size).sum();
        assert_eq!(held, r.heap_bytes(), "reservation != heap");
    }
    drop(g);

    // After `combine` the merged partials' hashes are offered and shed.
    let c3 = ctx(2);
    let g = sink.global_state(&c3).unwrap_or_else(|e| panic!("{e}"));
    let mut l3 = sink.local_state(&c3, &g).unwrap_or_else(|e| panic!("{e}"));
    assert!(sink.sink(batch(0), &mut l3, &g).is_ok());
    let before = hashes(&l3);
    assert!(sink.combine(l3, &g).is_ok());
    assert_eq!(offered(&c3), before, "merged hashes not offered");
    let freed = sink
        .spill_global(&g, 1, &c3)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(freed, before, "spill_global did not shed the hashes");
    assert_eq!(offered(&c3), 0);
    drop((g, c1, c2, c3, sink, join));
    assert_eq!(query.pool().reserved(), 0, "the query pool is not empty");
}

/// A merge past `bloom_max_items` drops the merged hashes, tells the
/// drivers still sinking, and releases what both partials reserved for
/// them (#800 review).
#[test]
fn a_merge_past_the_bloom_cap_raises_the_flag() {
    let mut f = filter();
    f.options.max_values = 100;
    f.options.bloom_max_items = 3000;
    let shared = vec![f.clone()];
    let lost = flag();
    let new = || Partials {
        builders: vec![krishiv_io::KeyFilterBuilder::new(
            DataType::Int64,
            f.options,
        )],
        memory: vec![reservation(1 << 24)],
        bloom_lost: Arc::clone(&lost),
        spilled: Arc::default(),
    };
    let keys = |from: i64| -> ArrayRef {
        Arc::new(Int64Array::from_iter_values(
            (from..from + 2000).map(|i| i * 1_000_003),
        ))
    };
    let (mut a, mut b) = (new(), new());
    a.add_keys(&shared, &[keys(0)], None);
    b.add_keys(&shared, &[keys(2000)], None);
    assert!(a.hash_bytes() > 0 && b.hash_bytes() > 0);
    assert!(!lost[0].load(std::sync::atomic::Ordering::Relaxed));
    a.merge(b);
    assert_eq!(a.hash_bytes(), 0, "4000 rows kept a Bloom capped at 3000");
    assert!(
        lost[0].load(std::sync::atomic::Ordering::Relaxed),
        "flag not raised"
    );
    let held: usize = a.memory.iter().map(Reservation::size).sum();
    assert_eq!(held, a.heap_bytes(), "the dropped hashes stay reserved");
}

/// A build with a spill scope asked for more than its hashes sheds them
/// and writes tables for the rest; it then offers only the tables left
/// in memory.
#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "test-only runtime via #[tokio::test]; never runs on a compute worker"
)]
async fn a_spilling_build_sheds_its_hashes_then_writes_tables() {
    use krishiv_memory::{SpillConfig, SpillManager, SpillOwner, Spillable as _};

    use crate::join::{HashJoin, JoinSpec, JoinType};

    let dir = std::env::temp_dir().join(format!("krishiv-build-shed-{}", std::process::id()));
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.clone(),
        ..SpillConfig::default()
    })
    .unwrap_or_else(|e| panic!("{e}"));
    let scope = manager.scope().unwrap_or_else(|e| panic!("{e}"));
    let mut f = filter();
    f.options.max_values = 100;
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "k",
        DataType::Int64,
        true,
    )]));
    let join = HashJoin::try_new(JoinSpec {
        join_type: JoinType::Inner,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        single_match: false,
        spill: Some(scope),
    })
    .and_then(|j| j.with_runtime_filters(vec![f.clone()]))
    .unwrap_or_else(|e| panic!("{e}"));
    let sink = join.build_sink();
    let root = TreePool::new(&MemoryConfig::default());
    let query = crate::QueryContext::new(
        crate::QueryId(1),
        crate::WorkloadClass::Interactive,
        crate::ExecConfig::default(),
        root.child("q", None),
    );
    let exec: Arc<dyn crate::Executor> =
        crate::TokioExecutor::new(&crate::ExecutorConfig::default())
            .unwrap_or_else(|e| panic!("{e}"));
    let ctx = crate::TaskContext::new(
        Arc::clone(&query),
        exec,
        SpillOwner::new(),
        crate::OperatorId(1),
        "b",
    );
    let keys: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..20_000).map(|i: i64| i * 1_000_003),
    ));
    let batch = krishiv_types::batch::Batch::new(
        arrow_array::RecordBatch::try_new(Arc::clone(&schema), vec![keys])
            .unwrap_or_else(|e| panic!("{e}")),
    );
    let g = sink.global_state(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let mut l = sink.local_state(&ctx, &g).unwrap_or_else(|e| panic!("{e}"));
    assert!(sink.sink(batch, &mut l, &g).is_ok());
    let local = |l: &crate::operator::LocalState| {
        l.downcast_ref::<super::BuildLocal>()
            .map(|l| {
                (
                    l.runtime.hash_bytes(),
                    super::mem_bytes(&l.parts),
                    l.file_bytes,
                )
            })
            .unwrap_or_else(|| panic!("not a build local"))
    };
    let (hashes, tables, _) = local(&l);
    assert!(hashes > 0 && tables > 0);
    let offered = || ctx.consumer().signal().spillable_bytes();
    assert_eq!(
        offered(),
        hashes + tables,
        "hashes and tables not both offered"
    );
    let freed = sink
        .spill(&mut l, hashes + 1, &ctx)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let (after, left, written) = local(&l);
    assert_eq!(after, 0, "hashes kept");
    assert!(written > 0, "no table written");
    assert!(freed > hashes, "freed only the hashes");
    assert!(left < tables);
    assert_eq!(offered(), left, "the signal is not the tables left");
    drop((l, g, sink, join, ctx));
    assert_eq!(query.pool().reserved(), 0, "the query pool is not empty");
    drop(manager);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Without memory for the hashes the Bloom is dropped; without memory
/// for the published filter only the range is kept; a dropped partial
/// disables its filter.
#[test]
fn memory_pressure_drops_forms_never_fails() {
    let f = filter();
    let shared = vec![f.clone()];
    let keys: ArrayRef = Arc::new(Int64Array::from_iter_values(0..2000));
    let mut p = Partials {
        builders: vec![krishiv_io::KeyFilterBuilder::new(
            DataType::Int64,
            f.options,
        )],
        memory: vec![reservation(1024)],
        bloom_lost: Arc::default(),
        spilled: Arc::default(),
    };
    let spec_keys = [Arc::clone(&keys)];
    p.add_keys(&shared, &spec_keys, None);
    assert!(p.memory[0].size() <= 1024);
    let mut held = reservation(1 << 20);
    publish(&publisher(&f), &shared, p, &mut held);
    let FilterSnapshot::Final(k) = f.filter.published().snapshot.clone() else {
        unreachable!("not published")
    };
    assert!(k.bloom_bytes().is_none() && k.has_range());

    // The hashes fit, the published Bloom does not. Sparse keys and no
    // exact set, so the filter is a Bloom when it fits (dense keys would
    // publish as the range alone, passing without the fallback).
    let sparse: ArrayRef = Arc::new(Int64Array::from_iter_values((0..2000).map(|i| i * 3)));
    let sparse_keys = [sparse];
    let published = |held: usize| {
        let mut f = filter();
        f.options.exact_max_values = 0;
        let shared = vec![f.clone()];
        let mut p = Partials {
            builders: vec![krishiv_io::KeyFilterBuilder::new(
                DataType::Int64,
                f.options,
            )],
            memory: vec![reservation(1 << 20)],
            bloom_lost: Arc::default(),
            spilled: Arc::default(),
        };
        p.add_keys(&shared, &sparse_keys, None);
        publish(&publisher(&f), &shared, p, &mut reservation(held));
        let FilterSnapshot::Final(k) = f.filter.published().snapshot.clone() else {
            unreachable!("not published")
        };
        k
    };
    assert!(
        published(1 << 20).bloom_bytes().is_some(),
        "no Bloom with room"
    );
    let k = published(16);
    assert!(k.bloom_bytes().is_none() && k.has_range());

    // A key of another type drops the partial; the filter is disabled.
    let f = filter();
    let shared = vec![f.clone()];
    let mut p = Partials {
        builders: vec![krishiv_io::KeyFilterBuilder::new(
            DataType::Int64,
            f.options,
        )],
        memory: Vec::new(),
        bloom_lost: Arc::default(),
        spilled: Arc::default(),
    };
    let wrong: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1]));
    p.add_keys(&shared, &[wrong], None);
    publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    assert!(matches!(
        f.filter.published().snapshot,
        FilterSnapshot::Disabled(_)
    ));
}

/// A filter of a guessed build: a fixed Bloom of 64 KiB, no exact set, a
/// key cap of 1000.
fn fixed_filter() -> JoinRuntimeFilter {
    let mut f = filter();
    f.options.exact_max_values = 0;
    f.options.bloom_max_items = 1000;
    f.options.bloom_spill_bytes = Some(64 << 10);
    f
}

/// A partial of `f` as the build makes one (its builder kept past the cap).
fn owned_partial(
    f: &JoinRuntimeFilter,
    lost: &Arc<[std::sync::atomic::AtomicBool]>,
    spilled: &Arc<[super::SpilledBloom]>,
    limit: usize,
) -> Partials {
    let mut b = krishiv_io::KeyFilterBuilder::new(DataType::Int64, f.options);
    if let Some(b) = b.as_mut() {
        b.keep_hashes_past_cap();
    }
    Partials {
        builders: vec![b],
        memory: vec![reservation(limit)],
        bloom_lost: Arc::clone(lost),
        spilled: Arc::clone(spilled),
    }
}

fn fixed() -> Arc<[super::SpilledBloom]> {
    Arc::new([super::SpilledBloom::default()])
}

fn spread(from: i64, n: i64) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(
        (from..from + n).map(|i| i * 1_000_003),
    ))
}

fn published_filter(f: &JoinRuntimeFilter) -> Arc<krishiv_io::KeyFilter> {
    let FilterSnapshot::Final(k) = f.filter.published().snapshot.clone() else {
        unreachable!("not published")
    };
    k
}

/// One partial past `bloom_max_items` creates and reserves the fixed
/// filter and drains into it; another partial then drains too, and the
/// published Bloom passes every key, on the filter's own reservation.
#[test]
fn partials_past_the_cap_drain_into_one_fixed_bloom() {
    use std::sync::atomic::Ordering;

    let f = fixed_filter();
    let shared = vec![f.clone()];
    let (lost, spilled) = (flag(), fixed());
    let mut a = owned_partial(&f, &lost, &spilled, 1 << 24);
    let mut b = owned_partial(&f, &lost, &spilled, 1 << 24);
    b.add_keys(&shared, &[spread(10_000, 500)], None);
    assert!(
        b.hash_bytes() > 0,
        "drained below the cap with no filter yet"
    );
    a.add_keys(&shared, &[spread(0, 2000)], None);
    assert!(
        spilled[0].active(),
        "past the cap the fixed filter was not made"
    );
    assert_eq!(a.hash_bytes(), 0, "hashes kept past the cap");
    assert_eq!(a.memory[0].size(), a.heap_bytes(), "reservation != heap");
    b.add_keys(&shared, &[spread(20_000, 100)], None);
    assert_eq!(b.hash_bytes(), 0, "the other partial does not drain");
    assert!(!lost[0].load(Ordering::Relaxed));
    a.merge(b);
    let fixed = publish(&publisher(&f), &shared, a, &mut reservation(1 << 20));
    assert_eq!(
        fixed.len(),
        1,
        "the fixed filter's reservation not handed over"
    );
    assert_eq!(fixed[0].size(), 64 << 10);
    let k = published_filter(&f);
    assert_eq!(k.bloom_bytes(), Some(64 << 10), "no fixed Bloom published");
    for (from, n) in [(0, 2000), (10_000, 500), (20_000, 100)] {
        let pass = k
            .evaluate(spread(from, n).as_ref())
            .unwrap_or_else(|| panic!());
        assert_eq!(pass.true_count(), usize::try_from(n).unwrap_or(0));
    }
}

/// A spill request (`shed_hashes`) creates the filter when the whole
/// build's keys are big enough, whichever partial is asked, and returns the
/// bytes freed net of the filter it reserved and whether hashes left; a
/// small build's hashes are shed as before and its Bloom lost.
#[test]
fn a_spill_drains_by_the_builds_size_and_reports_the_net() {
    use std::sync::atomic::Ordering;

    let mut f = fixed_filter();
    f.options.bloom_max_items = 100_000;
    let shared = vec![f.clone()];
    // A small partial asked first, in a build whose other partial holds
    // 1,000 keys: 1,100 keys as hashes (8,800 bytes) pass 64 KiB / 16.
    let (lost, spilled) = (flag(), fixed());
    let mut big = owned_partial(&f, &lost, &spilled, 1 << 24);
    let mut small = owned_partial(&f, &lost, &spilled, 1 << 24);
    big.add_keys(&shared, &[spread(0, 1000)], None);
    small.add_keys(&shared, &[spread(50_000, 100)], None);
    assert!(small.hash_bytes() > 0 && !spilled[0].active());
    // The big partial's batch reserved the filter's memory: the drain
    // takes nothing more from the pool, so the net is the hashes freed.
    assert_eq!(spilled[0].outcome().0, events::RESERVED);
    let freed = small.hash_bytes();
    assert_eq!(
        small.shed_hashes(),
        (freed, true),
        "reported the filter again or no move"
    );
    assert!(
        spilled[0].active(),
        "the small partial lost the build's Bloom"
    );
    assert_eq!(small.hash_bytes(), 0);
    assert!(!lost[0].load(Ordering::Relaxed), "the Bloom was lost");
    // Now the filter exists: the big partial's shed frees its hashes.
    let freed = big.hash_bytes();
    assert_eq!(big.shed_hashes(), (freed, true));
    // A build of 100 keys in all is small: shed as before, Bloom lost.
    let (lost, spilled) = (flag(), fixed());
    let mut tiny = owned_partial(&f, &lost, &spilled, 1 << 24);
    tiny.add_keys(&shared, &[spread(0, 100)], None);
    assert!(tiny.shed_hashes().0 > 0);
    assert!(
        !spilled[0].active(),
        "a small build reserved the fixed filter"
    );
    assert!(lost[0].load(Ordering::Relaxed));
}

/// More keys than the fixed filter holds within `bloom_max_fpp`: it is
/// freed at once and hashing stops, and the build
/// publishes a range.
#[test]
fn a_full_fixed_filter_is_freed_and_the_bloom_lost() {
    use std::sync::atomic::Ordering;

    let mut f = fixed_filter();
    f.options.bloom_spill_bytes = Some(1 << 10);
    let shared = vec![f.clone()];
    let (lost, spilled) = (flag(), fixed());
    let mut p = owned_partial(&f, &lost, &spilled, 1 << 24);
    p.add_keys(&shared, &[spread(0, 1001)], None);
    assert!(spilled[0].active(), "1,001 keys do not fit 1 KiB");
    p.add_keys(&shared, &[spread(10_000, 20_000)], None);
    assert!(!spilled[0].active(), "a full filter kept its memory");
    assert!(lost[0].load(Ordering::Relaxed));
    assert_eq!(p.hash_bytes(), 0, "still hashing");
    p.add_keys(&shared, &[spread(40_000, 100)], None);
    assert_eq!(p.hash_bytes(), 0, "hashing resumed");
    let fixed = publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    assert!(fixed.is_empty());
    let k = published_filter(&f);
    assert!(k.bloom_bytes().is_none() && k.has_range());
    let (bits, keys) = spilled[0].outcome();
    assert_eq!(bits, events::RESERVED | events::ACTIVE | events::FULL);
    assert_eq!(keys, 21_101);
}

/// A Bloom lost after the filter was made (another partial's refused
/// memory) frees the filter at the next batch of any partial (review
/// finding 6), and a disabled filter's fixed Bloom is freed at publish.
#[test]
fn a_lost_bloom_frees_the_fixed_filter() {
    use std::sync::atomic::Ordering;

    let f = fixed_filter();
    let shared = vec![f.clone()];
    let (lost, spilled) = (flag(), fixed());
    let mut p = owned_partial(&f, &lost, &spilled, 1 << 24);
    p.add_keys(&shared, &[spread(0, 2000)], None);
    assert!(spilled[0].active());
    lost[0].store(true, Ordering::Relaxed);
    p.add_keys(&shared, &[spread(10_000, 10)], None);
    assert!(
        !spilled[0].active(),
        "the lost Bloom's filter is still held"
    );
    let held = publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    assert!(held.is_empty());
    assert!(published_filter(&f).bloom_bytes().is_none());
    assert_eq!(
        spilled[0].outcome().0,
        events::RESERVED | events::ACTIVE | events::LOST
    );
    // A partial dropped (a key of another type) disables the filter and
    // frees its fixed Bloom.
    let f = fixed_filter();
    let shared = vec![f.clone()];
    let (lost, spilled) = (flag(), fixed());
    let mut p = owned_partial(&f, &lost, &spilled, 1 << 24);
    p.add_keys(&shared, &[spread(0, 2000)], None);
    assert!(spilled[0].active());
    let wrong: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1]));
    p.add_keys(&shared, &[wrong], None);
    assert!(
        !spilled[0].active(),
        "a dropped builder's fixed Bloom is held until publish"
    );
    let fixed = publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    assert!(fixed.is_empty());
    assert!(
        !spilled[0].active(),
        "a disabled filter's Bloom is still held"
    );
}

/// Without memory for the fixed filter, past the cap the hashes go and the
/// other partials are told, as without one; only the range is published.
#[test]
fn a_refused_fixed_filter_loses_the_bloom_as_before() {
    let mut f = fixed_filter();
    f.options.bloom_spill_bytes = Some(1 << 20);
    let shared = vec![f.clone()];
    let (lost, spilled) = (flag(), fixed());
    // Room for the hashes, not for the 1 MiB filter.
    let mut p = owned_partial(&f, &lost, &spilled, 256 << 10);
    p.add_keys(&shared, &[spread(0, 2000)], None);
    assert!(!spilled[0].active());
    assert_eq!(spilled[0].outcome().0, events::REFUSED);
    assert_eq!(p.hash_bytes(), 0, "hashes kept past the cap");
    assert!(lost[0].load(std::sync::atomic::Ordering::Relaxed));
    let fixed = publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    assert!(fixed.is_empty());
    let k = published_filter(&f);
    assert!(k.bloom_bytes().is_none() && k.has_range());
}

/// A partial of `f` holding `n` keys as hashes, its pool then full: the
/// rest of the limit taken by another reservation of its consumer. With
/// `early`, the filter's memory was reserved while the pool had room;
/// without, the partial is given a fresh fixed filter that saw the same
/// keys, as if that reservation had been refused.
fn full_pool_partial(
    f: &JoinRuntimeFilter,
    lost: &Arc<[std::sync::atomic::AtomicBool]>,
    n: i64,
    early: bool,
) -> (Partials, Reservation, Arc<[super::SpilledBloom]>) {
    let limit = 1 << 20;
    let mut p = owned_partial(f, lost, &fixed(), limit);
    p.add_keys(std::slice::from_ref(f), &[spread(0, n)], None);
    assert!(!p.spilled[0].active(), "made below the cap, unforced");
    assert_eq!(p.spilled[0].outcome().0, events::RESERVED, "not reserved");
    if !early {
        let fresh = fixed();
        fresh[0].saw(u64::try_from(n).unwrap_or(0));
        p.spilled = fresh;
    }
    // The early reservation holds the filter's bytes of the same pool.
    let filter = if early {
        f.options.bloom_spill_bytes.map_or(0, |b| b as usize)
    } else {
        0
    };
    let mut hog = p.memory[0].split(0);
    assert!(hog.try_grow(limit - p.memory[0].size() - filter).is_ok());
    assert!(p.memory[0].split(0).try_grow(1).is_err(), "pool not full");
    let spilled = Arc::clone(&p.spilled);
    (p, hog, spilled)
}

/// Review F1 of f574c368f: a spill request when the pool is full and the
/// partial's hashes (2,000 keys, 16 KB) are far below the filter (64 KiB),
/// as at q09 SF1000 (160 MB of hashes against 320 MiB) and SF100: the
/// filter's memory, reserved early, makes it, and the pool is not asked.
#[test]
fn a_full_pool_makes_the_reserved_fixed_filter() {
    use std::sync::atomic::Ordering;

    let mut f = fixed_filter();
    f.options.bloom_max_items = 100_000;
    let lost = flag();
    let (mut p, _hog, spilled) = full_pool_partial(&f, &lost, 2_000, true);
    let hashes = p.hash_bytes();
    assert!(hashes < 64 << 10, "the hashes cover the filter");
    assert_eq!(p.shed_hashes(), (hashes, true), "net freed");
    assert_eq!(
        spilled[0].outcome().0,
        events::RESERVED | events::ACTIVE,
        "refused"
    );
    assert!(!lost[0].load(Ordering::Relaxed), "the Bloom was lost");
    assert_eq!(p.memory[0].size(), p.heap_bytes(), "reservation != heap");
    let shared = vec![f.clone()];
    let fixed = publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    assert_eq!(fixed.len(), 1);
    let k = published_filter(&f);
    assert_eq!(k.bloom_bytes(), Some(64 << 10), "no fixed Bloom published");
    let pass = k
        .evaluate(spread(0, 2_000).as_ref())
        .unwrap_or_else(|| panic!());
    assert_eq!(pass.true_count(), 2_000);
}

/// A reserved filter no partial drained into is freed at publish, and the
/// build's own Bloom goes out (a guessed build that stayed small).
#[test]
fn an_unused_reservation_is_freed_at_publish() {
    let mut f = fixed_filter();
    f.options.bloom_max_items = 100_000;
    let (lost, spilled) = (flag(), fixed());
    let mut p = owned_partial(&f, &lost, &spilled, 1 << 24);
    p.add_keys(std::slice::from_ref(&f), &[spread(0, 2_000)], None);
    assert_eq!(spilled[0].outcome().0, events::RESERVED);
    let shared = vec![f.clone()];
    let fixed = publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    assert!(fixed.is_empty(), "the unused reservation was handed over");
    let k = published_filter(&f);
    assert!(
        k.bloom_bytes().is_some_and(|b| b < 64 << 10),
        "not its own Bloom"
    );
}

/// An early reservation the pool refuses is tried again: once memory is
/// freed, the next batch reserves the filter (review nit of 59e0f89df).
#[test]
fn a_refused_reservation_is_tried_again_on_the_next_batch() {
    let mut f = fixed_filter();
    f.options.bloom_max_items = 100_000;
    let (lost, spilled) = (flag(), fixed());
    let limit = 1 << 20;
    let mut p = owned_partial(&f, &lost, &spilled, limit);
    // Room for 2,000 keys' hashes, not for the 64 KiB filter.
    let mut hog = p.memory[0].split(0);
    assert!(hog.try_grow(limit - (48 << 10)).is_ok());
    p.add_keys(std::slice::from_ref(&f), &[spread(0, 2_000)], None);
    assert!(p.hash_bytes() > 0, "the hashes did not fit");
    assert_eq!(spilled[0].outcome().0, 0, "reserved without room");
    drop(hog);
    p.add_keys(std::slice::from_ref(&f), &[spread(10_000, 10)], None);
    assert_eq!(spilled[0].outcome().0, events::RESERVED, "not tried again");
    assert!(!spilled[0].active(), "made before any partial needed it");
}

/// Review F2 of f574c368f: a filter made from the hashes' reservation and
/// filled by the same drain (capacity below the partial's keys, full
/// pool) gives its bytes back, so the net is every hash byte freed.
#[test]
fn a_filter_made_and_filled_at_once_reports_the_whole_net() {
    use std::sync::atomic::Ordering;

    let mut f = fixed_filter();
    f.options.bloom_max_items = 100_000;
    f.options.bloom_spill_bytes = Some(1 << 10);
    let lost = flag();
    let (mut p, _hog, spilled) = full_pool_partial(&f, &lost, 20_000, false);
    let held = p.memory[0].size();
    let (net, moved) = p.shed_hashes();
    assert!(moved);
    assert_eq!(spilled[0].outcome().0, events::ACTIVE | events::FULL);
    assert!(lost[0].load(Ordering::Relaxed));
    assert_eq!(p.memory[0].size(), p.heap_bytes(), "reservation != heap");
    assert_eq!(net, held - p.heap_bytes(), "the filled filter's bytes lost");
}

/// A spill request when the pool is full and the early reservation was
/// refused: the hashes being drained hold more than the filter, so it is
/// paid from their reservation, the pool is not asked, and the net freed
/// is the hashes less the filter.
#[test]
fn a_full_pool_funds_the_fixed_filter_from_the_hashes() {
    use std::sync::atomic::Ordering;

    let mut f = fixed_filter();
    f.options.bloom_max_items = 100_000;
    let lost = flag();
    let (mut p, _hog, spilled) = full_pool_partial(&f, &lost, 10_000, false);
    let held = p.memory[0].size();
    let hashes = p.hash_bytes();
    assert!(hashes >= 64 << 10, "the hashes do not cover the filter");
    let (net, moved) = p.shed_hashes();
    assert!(moved, "no hashes left");
    assert_eq!(spilled[0].outcome().0, events::ACTIVE, "refused");
    assert!(spilled[0].active());
    assert!(!lost[0].load(Ordering::Relaxed), "the Bloom was lost");
    assert_eq!(p.hash_bytes(), 0);
    assert_eq!(p.memory[0].size(), p.heap_bytes(), "reservation != heap");
    assert_eq!(net, held - p.heap_bytes() - (64 << 10), "net freed");
    let shared = vec![f.clone()];
    let fixed = publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    assert_eq!(fixed.len(), 1);
    assert_eq!(fixed[0].size(), 64 << 10, "the filter's reservation");
    let k = published_filter(&f);
    assert_eq!(k.bloom_bytes(), Some(64 << 10), "no fixed Bloom published");
    let pass = k
        .evaluate(spread(0, 10_000).as_ref())
        .unwrap_or_else(|| panic!());
    assert_eq!(pass.true_count(), 10_000);
}

/// A batch the full pool refuses (`fit`) moves the hashes into a filter
/// they pay for, and the partial's reservation is shrunk to its heap
/// rather than kept at the size it had before the filter took its share.
#[test]
fn a_refused_batch_funds_the_fixed_filter_from_the_hashes() {
    use std::sync::atomic::Ordering;

    let mut f = fixed_filter();
    f.options.bloom_max_items = 100_000;
    let lost = flag();
    let (mut p, _hog, spilled) = full_pool_partial(&f, &lost, 10_000, false);
    p.add_keys(&[f], &[spread(20_000, 100)], None);
    assert_eq!(spilled[0].outcome().0, events::ACTIVE, "refused");
    assert!(!lost[0].load(Ordering::Relaxed), "the Bloom was lost");
    assert_eq!(p.hash_bytes(), 0);
    assert_eq!(p.memory[0].size(), p.heap_bytes(), "reservation != heap");
}

/// Through the sink: a build with a spill scope and a
/// fixed filter, asked to spill below its key cap, moves its hashes to the
/// filter, writes tables for the rest of the request, and after `combine`
/// and `finalize` publishes the fixed Bloom, which passes every key; the
/// query pool is empty at the end.
#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "test-only runtime via #[tokio::test]; never runs on a compute worker"
)]
async fn a_spilling_build_publishes_its_fixed_bloom() {
    use krishiv_memory::{SpillConfig, SpillManager, SpillOwner};

    use crate::join::{HashJoin, JoinSpec, JoinType};

    let dir = std::env::temp_dir().join(format!("krishiv-build-fixed-{}", std::process::id()));
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.clone(),
        ..SpillConfig::default()
    })
    .unwrap_or_else(|e| panic!("{e}"));
    let scope = manager.scope().unwrap_or_else(|e| panic!("{e}"));
    let mut f = fixed_filter();
    f.options.bloom_max_items = 1_000_000;
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "k",
        DataType::Int64,
        true,
    )]));
    let join = HashJoin::try_new(JoinSpec {
        join_type: JoinType::Inner,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        single_match: false,
        spill: Some(scope),
    })
    .and_then(|j| j.with_runtime_filters(vec![f.clone()]))
    .unwrap_or_else(|e| panic!("{e}"));
    let sink = join.build_sink();
    let root = TreePool::new(&MemoryConfig::default());
    let query = crate::QueryContext::new(
        crate::QueryId(1),
        crate::WorkloadClass::Interactive,
        crate::ExecConfig::default(),
        root.child("q", None),
    );
    let exec: Arc<dyn crate::Executor> =
        crate::TokioExecutor::new(&crate::ExecutorConfig::default())
            .unwrap_or_else(|e| panic!("{e}"));
    let ctx = crate::TaskContext::new(
        Arc::clone(&query),
        exec,
        SpillOwner::new(),
        crate::OperatorId(1),
        "b",
    );
    let keys = spread(0, 20_000);
    let batch = krishiv_types::batch::Batch::new(
        arrow_array::RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&keys)])
            .unwrap_or_else(|e| panic!("{e}")),
    );
    let g = sink.global_state(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let mut l = sink.local_state(&ctx, &g).unwrap_or_else(|e| panic!("{e}"));
    assert!(sink.sink(batch, &mut l, &g).is_ok());
    let hashes = l
        .downcast_ref::<super::BuildLocal>()
        .map(|l| l.runtime.hash_bytes())
        .unwrap_or_else(|| panic!("not a build local"));
    assert!(
        hashes > 64 << 10,
        "too few hashes to pass the filter's size"
    );
    // The batch reserved the 64 KiB filter (the build is big enough), so
    // the hashes free their whole size; a request past them writes tables.
    let freed = sink
        .spill(&mut l, hashes + 1, &ctx)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let written = l
        .downcast_ref::<super::BuildLocal>()
        .map_or(0, |l| l.file_bytes);
    assert!(written > 0, "no table written (freed {freed} of {hashes})");
    assert!(freed > hashes, "the hashes' bytes not all counted");
    assert!(sink.combine(l, &g).is_ok());
    sink.finalize(&g, &ctx)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let k = published_filter(&f);
    assert_eq!(
        k.bloom_bytes(),
        Some(64 << 10),
        "a spilled build lost its Bloom"
    );
    let pass = k.evaluate(keys.as_ref()).unwrap_or_else(|| panic!());
    assert_eq!(pass.true_count(), 20_000);
    // `EXPLAIN ANALYZE` says what the fixed Bloom did.
    let extra = sink.extra_metrics();
    let get = |name: &str| {
        extra
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, m)| m.clone())
    };
    assert_eq!(
        get("fixed_bloom"),
        Some(crate::metrics::Metric::Text(
            "planned; reserved; active; bloom".into()
        ))
    );
    assert_eq!(
        get("fixed_bloom_keys"),
        Some(crate::metrics::Metric::Count(20_000))
    );
    drop((g, sink, join, ctx));
    assert_eq!(query.pool().reserved(), 0, "the query pool is not empty");
    drop(manager);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A build without a spill scope, below its key cap, asked to spill after
/// `combine`: `spill_global` moves the merged hashes into the fixed filter
/// the batch reserved, so it returns their bytes; the filter is active and
/// the spillable-bytes signal no longer offers the hashes (`shed_hashes`'
/// `moved`, fd8d1dada review finding 2).
#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "test-only runtime via #[tokio::test]; never runs on a compute worker"
)]
async fn spill_global_moving_hashes_into_a_fixed_filter_updates_the_signal() {
    use krishiv_memory::{SpillOwner, Spillable as _};

    use crate::join::{HashJoin, JoinSpec, JoinType};

    let mut f = fixed_filter();
    f.options.bloom_max_items = 100_000;
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "k",
        DataType::Int64,
        true,
    )]));
    let join = HashJoin::try_new(JoinSpec {
        join_type: JoinType::Inner,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        single_match: false,
        spill: None,
    })
    .and_then(|j| j.with_runtime_filters(vec![f.clone()]))
    .unwrap_or_else(|e| panic!("{e}"));
    let sink = join.build_sink();
    let root = TreePool::new(&MemoryConfig::default());
    let query = crate::QueryContext::new(
        crate::QueryId(1),
        crate::WorkloadClass::Interactive,
        crate::ExecConfig::default(),
        root.child("q", None),
    );
    let exec: Arc<dyn crate::Executor> =
        crate::TokioExecutor::new(&crate::ExecutorConfig::default())
            .unwrap_or_else(|e| panic!("{e}"));
    let ctx = crate::TaskContext::new(
        Arc::clone(&query),
        exec,
        SpillOwner::new(),
        crate::OperatorId(1),
        "b",
    );
    // 2,000 keys as hashes pass a sixteenth of the 64 KiB filter, which
    // their batch reserves.
    let keys = spread(0, 2000);
    let batch = krishiv_types::batch::Batch::new(
        arrow_array::RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&keys)])
            .unwrap_or_else(|e| panic!("{e}")),
    );
    let g = sink.global_state(&ctx).unwrap_or_else(|e| panic!("{e}"));
    let mut l = sink.local_state(&ctx, &g).unwrap_or_else(|e| panic!("{e}"));
    assert!(sink.sink(batch, &mut l, &g).is_ok());
    let hashes = l
        .downcast_ref::<super::BuildLocal>()
        .map(|l| l.runtime.hash_bytes())
        .unwrap_or_else(|| panic!("not a build local"));
    assert!(hashes > 0 && hashes < 64 << 10, "{hashes} hash bytes");
    assert!(sink.combine(l, &g).is_ok());
    let offered = || ctx.consumer().signal().spillable_bytes();
    let before = offered();
    assert!(before >= hashes, "merged hashes not offered");
    let freed = sink
        .spill_global(&g, 1, &ctx)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        freed, hashes,
        "the reserved filter counted against the shed"
    );
    assert_eq!(
        lock(&super::state(&g).unwrap_or_else(|e| panic!("{e}")).runtime).hash_bytes(),
        0
    );
    assert_eq!(
        offered(),
        before - hashes,
        "the signal still offers hashes that sit in the filter"
    );
    sink.finalize(&g, &ctx)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    let extra = sink.extra_metrics();
    let state = extra
        .iter()
        .find(|(n, _)| n == "fixed_bloom")
        .map(|(_, m)| m.clone());
    assert_eq!(
        state,
        Some(crate::metrics::Metric::Text(
            "planned; reserved; active; bloom".into()
        )),
        "the fixed filter was not active"
    );
    let k = published_filter(&f);
    assert_eq!(k.bloom_bytes(), Some(64 << 10), "no fixed Bloom published");
    let pass = k.evaluate(keys.as_ref()).unwrap_or_else(|| panic!());
    assert_eq!(pass.true_count(), 2000);
    drop((g, sink, join, ctx));
    assert_eq!(query.pool().reserved(), 0, "the query pool is not empty");
}

/// A bound from a residual (design P3-0.3.1 D2) is collected from its
/// build column, not the key, over the rows the build stores (`valid`) and
/// only their non-NULL values: a NULL never lets NULL probe values pass.
#[test]
fn a_residual_bound_collects_the_non_null_values_of_its_column() {
    let options = KeyFilterOptions {
        max_values: 0,
        exact_max_values: 0,
        bloom: false,
        side: krishiv_io::RangeSide::Upper,
        ..KeyFilterOptions::default()
    };
    let f = JoinRuntimeFilter {
        key: 0,
        filter: Arc::new(RuntimeFilter::new(
            FilterId(1),
            ColumnPath::new("q"),
            DataType::Int64,
        )),
        options,
        column: Some(1),
    };
    let shared = vec![f.clone()];
    let mut p = Partials {
        builders: vec![krishiv_io::KeyFilterBuilder::new(DataType::Int64, options)],
        memory: vec![reservation(1 << 20)],
        bloom_lost: Arc::default(),
        spilled: Arc::default(),
    };
    // Keys far above the column's values: a filter fed the key would keep
    // them. The third row is not stored (a NULL `=` key), the fourth's
    // value is NULL.
    let keys: ArrayRef = Arc::new(Int64Array::from(vec![1_000, 2_000, 3_000, 4_000]));
    let columns: Vec<ArrayRef> = vec![
        Arc::clone(&keys),
        Arc::new(Int64Array::from(vec![Some(7), Some(9), Some(50), None])),
    ];
    let stored = arrow_buffer::NullBuffer::from(vec![true, true, false, true]);
    p.add_sources(&shared, (&[Arc::clone(&keys)], &columns), Some(&stored));
    publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    let FilterSnapshot::Final(k) = f.filter.published().snapshot.clone() else {
        unreachable!("not published")
    };
    assert!(!k.nulls_pass());
    let probe = Int64Array::from(vec![Some(i64::MIN), Some(9), Some(10), Some(50), None]);
    let pass: Vec<bool> = k
        .evaluate(&probe)
        .unwrap()
        .iter()
        .map(|b| b == Some(true))
        .collect();
    assert_eq!(pass, [true, true, false, false, false]);
}
