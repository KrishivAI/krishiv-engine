// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the grace join with real pools and spill files (not under loom).

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillOwner, TreePool};

use super::*;
use crate::{ExecConfig, ExecutorConfig, OperatorId, QueryContext, QueryId, TokioExecutor};
use crate::{WorkloadClass, context::TaskContext};

/// A spill directory under the system temp dir, removed when dropped.
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let name = format!("krishiv-{name}-{}", std::process::id());
        Self(std::env::temp_dir().join(name))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Level 0 is queued only once every write of an idle build's
/// partitions landed, and nothing is claimed once one was lost.
#[test]
fn level_zero_waits_for_an_idle_builds_writes() {
    let mut g = Grace::new(4);
    g.probe[1].rows = 1;
    g.writing = 1;
    assert!(matches!(g.next(), Next::Wait));
    g.writing = 0;
    assert!(matches!(g.next(), Next::Task(_)));
    g.lost = Some(crate::error::cancelled());
    assert!(matches!(g.next(), Next::Lost(_)));
}

/// Coalescing is size-tiered: a merged chunk is not copied again by the
/// next run of small chunks (merging all chunks every 16 pushes was
/// quadratic), the chunk count stays bounded, and no row is lost.
#[test]
fn coalescing_does_not_recopy_merged_chunks() {
    let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let id = pool.register("coalesce", None);
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let chunk = |i: i64| {
        let k = Int64Array::from_iter_values(i * 300..(i + 1) * 300);
        let s = StringArray::from_iter_values((0..300).map(|j| format!("row-{i}-{j}")));
        Batch::new(
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(k), Arc::new(s)]).unwrap(),
        )
    };
    let mut parts = vec![Part::with_memory(krishiv_memory::Reservation::new(
        Arc::clone(&pool),
        id,
    ))];
    let push = |parts: &mut Vec<Part>, i: i64| {
        let c = chunk(i);
        reserve_chunks(parts, &[(0, c.clone())]).unwrap();
        parts[0].chunks.push(c);
        parts[0].coalesce(&schema).unwrap()
    };
    let sizes: usize = (0..16)
        .map(|i| chunk(i).data().get_array_memory_size())
        .sum();
    let mut merged_run = 0;
    for i in 0..16 {
        merged_run = push(&mut parts, i);
    }
    assert_eq!(parts[0].chunks.len(), 1, "16 small chunks merge");
    // It reports the run it merged: the 16 chunks' bytes.
    assert_eq!(merged_run, sizes);
    let merged = Arc::clone(parts[0].chunks[0].data().column(0));
    for i in 16..31 {
        assert_eq!(push(&mut parts, i), 0, "no merge below the fan-out");
    }
    assert_eq!(parts[0].chunks.len(), 16);
    let first = parts[0].chunks[0].data().column(0);
    assert!(
        Arc::ptr_eq(first, &merged),
        "the merged chunk was copied again"
    );
    let mut most = 0;
    for i in 31..1024 {
        push(&mut parts, i);
        most = most.max(parts[0].chunks.len());
    }
    assert!(most < 3 * 16, "{most} chunks held");
    assert_eq!(parts[0].mem_rows(), 1024 * 300);
    let mut keys: Vec<i64> = parts[0]
        .chunks
        .iter()
        .flat_map(|c| {
            let k = c
                .data()
                .column(0)
                .as_primitive::<arrow_array::types::Int64Type>();
            k.values().to_vec()
        })
        .collect();
    keys.sort_unstable();
    assert!(
        keys.into_iter().eq(0..1024 * 300),
        "rows lost or duplicated"
    );
    drop(parts);
    assert_eq!(pool.reserved(), 0);
}

/// Probe rows deferred in many small chunks are coalesced, spilled in
/// slices, and read back by the grace phase with exactly the headroom
/// [`load`] reserves: with the pool otherwise full, no read reaches it.
/// (The headroom used to be twice the largest chunk *before*
/// coalescing, which a coalesced chunk's slices outgrow: a read then
/// drew from a pool that other drivers kept full.)
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn spilled_probe_rows_read_back_within_the_load_headroom() {
    const LIMIT: usize = 64 << 20;
    let root = TreePool::new(&MemoryConfig {
        limit: Some(LIMIT),
        quantum: 1,
        spill_scratch_reserve: 0,
        resident_guard: true,
        ..MemoryConfig::default()
    });
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        root.child("q", None),
    );
    let exec = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let ctx = TaskContext::new(query, exec, SpillOwner::new(), OperatorId(1), "probe");
    let dir = TempDir::new("grace-headroom");
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.0.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let chunk = |i: i64| {
        let k = Int64Array::from_iter_values(i * 300..(i + 1) * 300);
        let s = StringArray::from_iter_values((0..300).map(|j| format!("row-{i}-{j}")));
        Batch::new(
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(k), Arc::new(s)]).unwrap(),
        )
    };
    let mut parts = vec![Part::with_memory(ctx.consumer().reservation())];
    let mut largest_chunk = 0;
    for i in 0..64 {
        let c = chunk(i);
        largest_chunk = largest_chunk.max(c.data().get_array_memory_size());
        reserve_chunks(&mut parts, &[(0, c.clone())]).unwrap();
        parts[0].chunks.push(c);
        parts[0].coalesce(&schema).unwrap();
    }
    assert!(parts[0].chunks.len() < 16, "coalesced");
    spill_parts(&mut parts, &[0], &schema, &scope, &ctx, false)
        .await
        .unwrap();
    let headroom = parts[0].read_headroom();
    assert!(headroom > 2 * largest_chunk, "the old estimate fell short");

    // The pool is full but for the headroom, lent as credit as `load` does.
    let mut lent = ctx.consumer().reservation();
    lent.try_grow(headroom).unwrap();
    lent.shrink_to_credit(headroom);
    let other: Arc<dyn MemoryPool> = root.child("other", None);
    let id = other.register("filler", None);
    let mut filler = krishiv_memory::Reservation::new(Arc::clone(&other), id);
    filler.try_grow(LIMIT - root.reserved()).unwrap();
    let mut rows = 0;
    for file in &parts[0].files {
        for b in file.reader(ctx.consumer().reservation()).unwrap() {
            rows += b.unwrap().num_rows();
        }
    }
    assert_eq!(rows, 64 * 300);
    ctx.consumer().consumer().release_credit();
    drop((filler, parts));
    assert_eq!(root.reserved(), 0);
}

/// A reversal whose probe rows (read back from spill files) do not fit
/// the pool's free memory declines, and hands the task back intact.
#[tokio::test]
async fn a_probe_side_that_does_not_fit_declines_the_reversal() {
    const LIMIT: usize = 64 << 20;
    let root = TreePool::new(&MemoryConfig {
        limit: Some(LIMIT),
        quantum: 1,
        spill_scratch_reserve: 0,
        resident_guard: true,
        ..MemoryConfig::default()
    });
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        root.child("q", None),
    );
    let exec = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let ctx = TaskContext::new(query, exec, SpillOwner::new(), OperatorId(1), "probe");
    let dir = TempDir::new("grace-reverse-decline");
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.0.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let join = super::super::HashJoin::try_new(super::super::JoinSpec {
        join_type: super::super::JoinType::LeftSemi,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        single_match: false,
        spill: Some(scope.clone()),
    })
    .unwrap();
    let shared = &join.0;
    let hashing = shared.hashing(&ExecConfig::default());
    let k = Int64Array::from_iter_values(0..3000);
    let v = StringArray::from_iter_values((0..3000).map(|j| format!("row-{j}")));
    let batch = Batch::new(
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(k), Arc::new(v)]).unwrap(),
    );
    let mut parts = vec![Part::with_memory(ctx.consumer().reservation())];
    reserve_chunks(&mut parts, &[(0, batch.clone())]).unwrap();
    parts[0].chunks.push(batch);
    spill_parts(&mut parts, &[0], &schema, &scope, &ctx, false)
        .await
        .unwrap();
    let files = parts[0].files.len();
    assert!(files > 0 && parts[0].chunks.is_empty());
    let task = Task {
        depth: 0,
        no_shrink: false,
        build: Part::default(),
        probe: parts.remove(0),
    };
    // Frame headroom lent as `reverse` does; nothing else is free.
    let headroom = task.probe.read_headroom();
    let mut lent = ctx.consumer().reservation();
    lent.try_grow(headroom).unwrap();
    lent.shrink_to_credit(headroom);
    let other: Arc<dyn MemoryPool> = root.child("other", None);
    let id = other.register("filler", None);
    let mut filler = krishiv_memory::Reservation::new(Arc::clone(&other), id);
    filler.try_grow(LIMIT - root.reserved()).unwrap();
    let handle = Arc::clone(ctx.consumer().consumer());
    let Ok(Reversed::Declined(task)) = reverse_blocking(
        shared,
        &hashing,
        task,
        &handle,
        ctx.query(),
        ctx.partition(),
    ) else {
        panic!("the probe rows cannot fit");
    };
    assert_eq!(task.probe.files.len(), files, "nothing was consumed");
    assert_eq!(task.probe.rows, 3000);
    // With room, the same task joins: a semi join with no build rows
    // emits nothing, and the probe files are gone with the task.
    drop(filler);
    let Ok(Reversed::Joined(out)) = reverse_blocking(
        shared,
        &hashing,
        task,
        &handle,
        ctx.query(),
        ctx.partition(),
    ) else {
        panic!("the probe rows fit now");
    };
    assert!(out.is_none());
    ctx.consumer().consumer().release_credit();
    assert_eq!(root.reserved(), 0);
    assert_eq!(manager.used_bytes(), 0);
}

/// A reversed semi join with a residual condition decodes the build
/// columns in the arena's payload format: payload rows when
/// `krishiv.exec.join.rowLayout` is on (it read the arrow-row payload,
/// empty in that layout, and panicked decoding it), arrow rows
/// otherwise. Both give the same rows.
#[tokio::test]
async fn a_reversal_with_a_condition_decodes_the_build_rows_of_either_layout() {
    use arrow_array::BooleanArray;
    use arrow_array::types::Int64Type;

    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let batch = |k: Vec<i64>, v: Vec<i64>| {
        Batch::new(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(k)), Arc::new(Int64Array::from(v))],
            )
            .unwrap(),
        )
    };
    // probe.k = build.k AND build.v > probe.v
    let filter: Arc<dyn crate::Predicate> = Arc::new(crate::FnPredicate(|b: &RecordBatch| {
        let probe = b.column(1).as_primitive::<Int64Type>();
        let build = b.column(3).as_primitive::<Int64Type>();
        Ok(probe
            .iter()
            .zip(build.iter())
            .map(|(p, b)| Some(b? > p?))
            .collect::<BooleanArray>())
    }));
    for row_layout in [false, true] {
        let root = TreePool::new(&MemoryConfig::default());
        let config = ExecConfig {
            join_row_layout: row_layout,
            ..ExecConfig::default()
        };
        let query = QueryContext::new(
            QueryId(1),
            WorkloadClass::Interactive,
            config.clone(),
            root.child("q", None),
        );
        let exec = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
        let ctx = TaskContext::new(query, exec, SpillOwner::new(), OperatorId(1), "probe");
        let join = super::super::HashJoin::try_new(super::super::JoinSpec {
            join_type: super::super::JoinType::LeftSemi,
            probe_schema: Arc::clone(&schema),
            build_schema: Arc::clone(&schema),
            probe_keys: vec![0],
            build_keys: vec![0],
            null_safe: vec![false],
            filter: Some(Arc::clone(&filter)),
            single_match: false,
            spill: None,
        })
        .unwrap();
        let shared = &join.0;
        let hashing = shared.hashing(&config);
        assert_eq!(hashing.rows(), row_layout);

        // Probe rows k = v = 0..100; build rows k = 0..100 with v = k + 1
        // for odd k and v = k for even k: odd probe rows match.
        let probe = batch((0..100).collect(), (0..100).collect());
        let build = batch((0..100).collect(), (0..100).map(|k| k + (k % 2)).collect());
        let key_cols = vec![Arc::clone(build.data().column(0))];
        let keys = shared.keys.encode(&key_cols, &hashing).unwrap();
        let columns = build.data().columns();
        let payload = match (shared.row_payload(hashing.rows()), &shared.payload) {
            (Some(layout), _) => {
                super::super::build::Payload::Rows(layout.encode(columns).unwrap())
            }
            (None, Some((conv, _))) => {
                super::super::build::Payload::Arrow(conv.convert_columns(columns).unwrap())
            }
            (None, None) => panic!("the condition reads build columns"),
        };
        let all: Vec<usize> = (0..100).collect();
        let chunk = super::super::build::arena_chunk(
            shared,
            &hashing,
            &keys,
            keys.hashes(),
            Some(&payload),
            build.weights(),
            &all,
        )
        .unwrap();
        let mut build_part = vec![Part::with_memory(ctx.consumer().reservation())];
        reserve_chunks(&mut build_part, &[(0, chunk.clone())]).unwrap();
        build_part[0].chunks.push(chunk);
        let mut probe_part = vec![Part::with_memory(ctx.consumer().reservation())];
        reserve_chunks(&mut probe_part, &[(0, probe.clone())]).unwrap();
        probe_part[0].chunks.push(probe);
        let task = Task {
            depth: 0,
            no_shrink: false,
            build: build_part.remove(0),
            probe: probe_part.remove(0),
        };
        let handle = Arc::clone(ctx.consumer().consumer());
        let Ok(Reversed::Joined(Some(out))) = reverse_blocking(
            shared,
            &hashing,
            task,
            &handle,
            ctx.query(),
            ctx.partition(),
        ) else {
            panic!("the reversal joins (row layout {row_layout})");
        };
        let mut got = out
            .data()
            .column(0)
            .as_primitive::<Int64Type>()
            .values()
            .to_vec();
        got.sort_unstable();
        let want: Vec<i64> = (0..100).filter(|k| k % 2 == 1).collect();
        assert_eq!(got, want, "row layout {row_layout}");
        drop(out);
        assert_eq!(root.reserved(), 0);
    }
}

/// A spill whose rows and the pool's free memory cannot fund its
/// writer's scratch is declined: nothing is written or lost, no credit
/// is left behind, and the query does not fail.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn an_unfundable_spill_is_declined() {
    const LIMIT: usize = 4 << 20;
    let root = TreePool::new(&MemoryConfig {
        limit: Some(LIMIT),
        quantum: 1,
        spill_scratch_reserve: 0,
        resident_guard: true,
        ..MemoryConfig::default()
    });
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        root.child("q", None),
    );
    let exec = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let ctx = TaskContext::new(query, exec, SpillOwner::new(), OperatorId(1), "probe");
    let dir = TempDir::new("grace-decline");
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.0.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let small = Batch::new(
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(0..100))],
        )
        .unwrap(),
    );
    let mut parts = vec![Part::with_memory(ctx.consumer().reservation())];
    reserve_chunks(&mut parts, &[(0, small.clone())]).unwrap();
    parts[0].chunks.push(small);
    let held = parts[0].bytes();
    let other: Arc<dyn MemoryPool> = root.child("other", None);
    let id = other.register("filler", None);
    let mut filler = krishiv_memory::Reservation::new(Arc::clone(&other), id);
    filler.try_grow(LIMIT - root.reserved()).unwrap();
    let out = spill_parts(&mut parts, &[0], &schema, &scope, &ctx, false).await;
    assert_eq!(out.unwrap(), (0, 0));
    assert_eq!((parts[0].chunks.len(), parts[0].bytes()), (1, held));
    assert!(parts[0].files.is_empty());
    assert_eq!(ctx.consumer().consumer().credit(), 0);
    // With room for the scratch, the same spill goes ahead.
    drop(filler);
    let (freed, written) = spill_parts(&mut parts, &[0], &schema, &scope, &ctx, false)
        .await
        .unwrap();
    assert_eq!((freed, parts[0].files.len()), (held, 1));
    assert!(written > 0);
    assert_eq!(ctx.consumer().consumer().credit(), 0);
    drop(parts);
    assert_eq!(root.reserved(), 0);
}

/// With the pool's spill-scratch floor, a spill of a pool that ordinary
/// reservations filled is funded from the floor instead of declined,
/// and the floor is free again afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn the_scratch_floor_funds_a_spill_of_a_full_pool() {
    const LIMIT: usize = 4 << 20;
    let root = TreePool::new(&MemoryConfig {
        limit: Some(LIMIT),
        quantum: 1,
        spill_scratch_reserve: SPILL_SCRATCH,
        resident_guard: true,
        ..MemoryConfig::default()
    });
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        root.child("q", None),
    );
    let exec = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let ctx = TaskContext::new(query, exec, SpillOwner::new(), OperatorId(1), "probe");
    let dir = TempDir::new("grace-floor");
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.0.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let small = Batch::new(
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(0..100))],
        )
        .unwrap(),
    );
    let mut parts = vec![Part::with_memory(ctx.consumer().reservation())];
    reserve_chunks(&mut parts, &[(0, small.clone())]).unwrap();
    parts[0].chunks.push(small);
    let held = parts[0].bytes();
    let other: Arc<dyn MemoryPool> = root.child("other", None);
    let id = other.register("filler", None);
    let mut filler = krishiv_memory::Reservation::new(Arc::clone(&other), id);
    filler
        .try_grow(LIMIT - SPILL_SCRATCH - root.reserved())
        .unwrap();
    assert!(filler.try_grow(1).is_err(), "full for ordinary memory");
    let (freed, written) = spill_parts(&mut parts, &[0], &schema, &scope, &ctx, false)
        .await
        .unwrap();
    assert_eq!((freed, parts[0].files.len()), (held, 1));
    assert!(written > 0);
    assert_eq!(ctx.consumer().consumer().credit(), 0);
    assert_eq!(root.reserved(), filler.size(), "the floor is free again");
    drop((parts, filler));
    assert_eq!(root.reserved(), 0);
}

/// A grace partition's spill write runs under the consumer's `spilling`
/// guard, so another driver's round that cannot spill waits for the
/// rows it frees.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn a_spill_write_holds_the_spilling_guard() {
    let root = TreePool::new(&MemoryConfig::default());
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        root.child("q", None),
    );
    let exec = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let ctx = TaskContext::new(query, exec, SpillOwner::new(), OperatorId(1), "probe");
    let dir = TempDir::new("grace-guard");
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.0.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let consumer = Arc::clone(ctx.consumer().consumer());
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    scope.set_test_writer_hook(Arc::new({
        let seen = Arc::clone(&seen);
        move || seen.lock().unwrap().push(consumer.is_spilling())
    }));
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let small = Batch::new(
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(0..100))],
        )
        .unwrap(),
    );
    let mut parts = vec![Part::with_memory(ctx.consumer().reservation())];
    reserve_chunks(&mut parts, &[(0, small.clone())]).unwrap();
    parts[0].chunks.push(small);
    let (freed, _) = spill_parts(&mut parts, &[0], &schema, &scope, &ctx, true)
        .await
        .unwrap();
    assert!(freed > 0);
    assert_eq!(*seen.lock().unwrap(), [true]);
    assert!(!ctx.consumer().consumer().is_spilling());
    drop(parts);
    assert_eq!(root.reserved(), 0);
}

/// An alone load asks for an estimate of the whole rest of the
/// partition only until the arbitrator refuses it once; then for the
/// bytes missing now, and a refused exact ask gives up.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn a_refused_estimate_is_not_asked_again() {
    let root = TreePool::new(&MemoryConfig {
        limit: Some(64 << 20),
        quantum: 1,
        ..MemoryConfig::default()
    });
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        root.child("q", None),
    );
    let exec = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let ctx = TaskContext::new(query, exec, SpillOwner::new(), OperatorId(1), "probe");
    let dir = TempDir::new("grace-estimate");
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.0.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let rows = Batch::new(
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(0..10_000))],
        )
        .unwrap(),
    );
    let mut parts = vec![Part::with_memory(ctx.consumer().reservation())];
    reserve_chunks(&mut parts, &[(0, rows.clone())]).unwrap();
    parts[0].chunks.push(rows.clone());
    spill_parts(&mut parts, &[0], &schema, &scope, &ctx, true)
        .await
        .unwrap();
    let task = Task {
        depth: 0,
        no_shrink: false,
        build: parts.pop().unwrap(),
        probe: Part::default(),
    };
    let mut loading = Loading::new(task, true, &ctx);
    // Some rows read: the estimate covers the other 9,000 too.
    loading.memory.try_grow(8_000).unwrap();
    loading.chunks.push(rows.slice(0, 1_000).unwrap());
    let estimate = loading.ask(100, 0);
    assert!(estimate > 9_000 * 8, "{estimate}");
    assert!(loading.check_grant(estimate).is_none(), "granted");
    assert_eq!(loading.ask(100, 0), estimate, "asked again after a grant");
    assert!(matches!(loading.check_grant(0), Some(Loaded::Need(100))));
    assert!(matches!(loading.check_grant(0), Some(Loaded::TooLarge)));
    assert_eq!(loading.ask(100, 0), 100, "never the estimate again");
    assert!(loading.check_grant(100).is_none());
    drop(loading);
    ctx.consumer().consumer().release_credit();
}

/// A load holds its seat in the table cohort from its creation, first
/// try or alone retry; the seat goes idle when the load gives its task
/// back.
#[tokio::test]
async fn a_load_holds_its_table_seat_until_it_gives_the_task_back() {
    let root = TreePool::new(&MemoryConfig::default());
    let q = root.child("q", None);
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        Arc::clone(&q),
    );
    let exec = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let ctx = TaskContext::new(query, exec, SpillOwner::new(), OperatorId(1), "probe");
    let peer = q.cohort(TABLE_COHORT).unwrap().join();
    let cohort = Arc::clone(peer.cohort());
    let task = || Task {
        depth: 0,
        no_shrink: false,
        build: Part::default(),
        probe: Part::default(),
    };
    let first = Loading::new(task(), false, &ctx);
    assert!(cohort.others_active(&peer), "a first load holds");
    drop(first.into_task());
    assert!(!cohort.others_active(&peer), "idle once given back");
    let alone = Loading::new(task(), true, &ctx);
    assert!(cohort.others_active(&peer), "an alone retry holds");
    assert_eq!(cohort.members(), 2);
    drop(alone);
    assert!(!cohort.others_active(&peer));
}

/// A spill picks at least `SPILL_MIN` bytes when it can, never a part
/// without rows in memory.
#[test]
fn spills_write_at_least_the_scratch_they_need() {
    assert!(choose(&[Part::default()], 0, 1).is_empty());
}
