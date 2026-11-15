// SPDX-License-Identifier: Apache-2.0
//! Tests of the hash join build sink: the runtime filter partials (hashes shed,
//! "Bloom lost" flags, reservations), the spillable signal and idle writes.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::DataType;
use krishiv_io::{ColumnPath, FilterId, FilterSnapshot, KeyFilterOptions, RuntimeFilter};
use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};

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
    };
    p.add_keys(&shared, &[Arc::clone(&keys)], None);
    let before = p.memory[0].size();
    assert_eq!(before, p.heap_bytes(), "hashes not reserved");
    let freed = p.shed_hashes();
    assert!(freed > 0, "nothing shed");
    assert_eq!(p.memory[0].size(), before - freed);
    assert_eq!(p.memory[0].size(), p.heap_bytes(), "reservation != heap");
    assert_eq!(p.shed_hashes(), 0, "shed twice");
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
    };
    let (mut a, mut b) = (new(&lost), new(&lost));
    let mut other = new(&flag());
    for p in [&mut a, &mut b, &mut other] {
        p.add_keys(&shared, &[Arc::clone(&keys)], None);
        assert!(p.hash_bytes() > 0, "no hashes collected");
    }
    let hashes = a.hash_bytes();
    assert_eq!(a.shed_hashes(), hashes, "the shed is not the hash bytes");
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

/// A merge past `bloom_max_items` drops the merged hashes and tells the
/// drivers still sinking.
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

    // The hashes fit, the published Bloom does not.
    let f = filter();
    let shared = vec![f.clone()];
    let mut p = Partials {
        builders: vec![krishiv_io::KeyFilterBuilder::new(
            DataType::Int64,
            f.options,
        )],
        memory: vec![reservation(1 << 20)],
        bloom_lost: Arc::default(),
    };
    p.add_keys(&shared, &spec_keys, None);
    let mut tiny = reservation(16);
    publish(&publisher(&f), &shared, p, &mut tiny);
    let FilterSnapshot::Final(k) = f.filter.published().snapshot.clone() else {
        unreachable!("not published")
    };
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
    };
    let wrong: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1]));
    p.add_keys(&shared, &[wrong], None);
    publish(&publisher(&f), &shared, p, &mut reservation(1 << 20));
    assert!(matches!(
        f.filter.published().snapshot,
        FilterSnapshot::Disabled(_)
    ));
}
