// SPDX-License-Identifier: Apache-2.0
//! The nested-loop join against a brute-force reference: every join type,
//! empty and multi-batch build sides, NULL conditions, and probe batches
//! large enough to be split into several chunks.
#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    reason = "test helpers sit outside #[test] fns"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::ErrorClass;
use krishiv_exec::testing::Scan;
use krishiv_exec::{
    ExecConfig, ExecutorConfig, FnPredicate, JoinType, MemoryScan, Metric, NestedLoopJoin,
    NestedLoopJoinNode, NestedLoopSpec, Predicate, QueryContext, QueryId, QueryRunner,
    ScanPushdown as _, ScanPushdowns, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{
    MemoryConfig, MemoryPool as _, SpillConfig, SpillManager, SpillScope, TreePool,
};
use krishiv_types::batch::{Batch, Weights};

fn schema(name: &str) -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, true)]))
}

/// Batches of `values` (`None` is NULL), `per` rows each.
fn batches(name: &str, values: &[Option<i64>], per: usize) -> Vec<Batch> {
    values
        .chunks(per.max(1))
        .map(|c| {
            let a = Int64Array::from(c.to_vec());
            Batch::new(RecordBatch::try_new(schema(name), vec![Arc::new(a)]).unwrap())
        })
        .collect()
}

/// `p < b`, NULL when either is NULL.
fn less() -> Arc<dyn Predicate> {
    Arc::new(FnPredicate(|b: &RecordBatch| {
        let (p, q) = (
            b.column(0).as_primitive::<Int64Type>(),
            b.column(1).as_primitive::<Int64Type>(),
        );
        Ok(p.iter()
            .zip(q.iter())
            .map(|(x, y)| Some(x? < y?))
            .collect::<BooleanArray>())
    }))
}

fn run(
    join_type: JoinType,
    filter: Option<Arc<dyn Predicate>>,
    probe: Vec<Batch>,
    build: Vec<Batch>,
) -> Result<Vec<String>, krishiv_common::error::KrishivError> {
    run_with(join_type, filter, probe, build, None).map(|(rows, _)| rows)
}

/// Removes a test's spill directory when dropped.
struct SpillDir {
    manager: Option<SpillManager>,
    dir: std::path::PathBuf,
}

impl Drop for SpillDir {
    fn drop(&mut self) {
        drop(self.manager.take());
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn spill_scope() -> (SpillDir, SpillScope) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("nested-loop-{}-{n}", std::process::id()));
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.clone(),
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    (
        SpillDir {
            manager: Some(manager),
            dir,
        },
        scope,
    )
}

/// Operator counters: (name, value).
type Counters = Vec<(String, u64)>;

/// The sorted output rows, and the operator counters (name, value). With
/// `limit`, the query runs under that memory limit with a spill scope.
fn run_with(
    join_type: JoinType,
    filter: Option<Arc<dyn Predicate>>,
    probe: Vec<Batch>,
    build: Vec<Batch>,
    limit: Option<usize>,
) -> Result<(Vec<String>, Counters), krishiv_common::error::KrishivError> {
    run_on(
        join_type,
        filter,
        probe,
        build,
        limit,
        ExecutorConfig::default(),
    )
}

/// As [`run_with`], on an executor with `executor`'s threads.
fn run_on(
    join_type: JoinType,
    filter: Option<Arc<dyn Predicate>>,
    probe: Vec<Batch>,
    build: Vec<Batch>,
    limit: Option<usize>,
    executor: ExecutorConfig,
) -> Result<(Vec<String>, Counters), krishiv_common::error::KrishivError> {
    run_beside(join_type, filter, probe, build, limit, executor, 0, None)
}

/// Another member of the query's nested-loop chunk cohort for
/// `release_after`, idle (it holds nothing, so it only counts as a member).
#[derive(Clone, Copy)]
struct Peer {
    release_after: std::time::Duration,
}

/// As [`run_on`], while a consumer of another query under the same root
/// holds `hold` bytes for the whole run.
#[expect(
    clippy::too_many_arguments,
    reason = "a test helper: the inputs, limit, executor and both neighbours"
)]
fn run_beside(
    join_type: JoinType,
    filter: Option<Arc<dyn Predicate>>,
    probe: Vec<Batch>,
    build: Vec<Batch>,
    limit: Option<usize>,
    executor: ExecutorConfig,
    hold: usize,
    peer: Option<Peer>,
) -> Result<(Vec<String>, Counters), krishiv_common::error::KrishivError> {
    let (_dir, scope) = spill_scope();
    let source = |name: &str, b: Vec<Batch>| {
        MemoryScan::try_new(schema(name), b)
            .unwrap()
            .build(ScanPushdowns::default())
            .unwrap()
    };
    let join = NestedLoopJoin::try_new(NestedLoopSpec {
        join_type,
        probe_schema: schema("p"),
        build_schema: schema("b"),
        filter,
        spill: limit.map(|_| scope.clone()),
    })?;
    let plan = Arc::new(NestedLoopJoinNode {
        join,
        build: Arc::new(Scan(source("b", build))),
        probe: Arc::new(Scan(source("p", probe))),
    });
    let executor = TokioExecutor::new(&executor)?;
    let runner = QueryRunner::new(executor);
    let root = TreePool::new(&MemoryConfig {
        limit,
        quantum: 4 << 10,
        spill_scratch_reserve: 32 << 10,
        ..MemoryConfig::default()
    });
    let other = root.child("other", None);
    let held = other.register("held", None);
    other.try_reserve(held, hold).unwrap();
    // The query runs in a task pool below its query pool, as a stage's
    // tasks do, so a peer can share the query's cohort and limit.
    let q = root.child("q", None);
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        q.child("task", None),
    );
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    // Runs on `rt` beside the query (while `block_on` drives it); dropped
    // with `rt` if the query ends first.
    let _peer = peer.map(|p| {
        // Same name as the operator's cohort (`join/nested/probe.rs`).
        let seat = q.cohort("nested-loop chunks").unwrap().join();
        rt.spawn(async move {
            tokio::time::sleep(p.release_after).await;
            drop(seat);
        })
    });
    let out = rt.block_on(async {
        let mut stream = runner.execute(plan, Arc::clone(&query)).await?;
        let mut rows = Vec::new();
        while let Some(b) = stream.next().await {
            let b = b?.into_parts().0;
            for r in 0..b.num_rows() {
                let cells: Vec<String> = b
                    .columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(r) {
                            "N".to_owned()
                        } else if let Some(c) = c.as_boolean_opt() {
                            c.value(r).to_string()
                        } else {
                            c.as_primitive::<Int64Type>().value(r).to_string()
                        }
                    })
                    .collect();
                rows.push(cells.join(","));
            }
        }
        query.finished().await;
        Ok::<_, krishiv_common::error::KrishivError>(rows)
    });
    other.release(held, hold);
    other.unregister(held);
    assert_eq!(root.reserved(), 0, "memory is released");
    let mut out = out?;
    out.sort();
    let counters = query
        .metrics()
        .operators
        .values()
        .flat_map(|o| o.extra.iter())
        .filter_map(|(k, v)| match v {
            Metric::Count(n) => Some((k.to_string(), *n)),
            _ => None,
        })
        .collect();
    Ok((out, counters))
}

/// The reference: pairs by brute force.
fn expected(
    join_type: JoinType,
    cond: bool,
    probe: &[Option<i64>],
    build: &[Option<i64>],
) -> Vec<String> {
    let hit =
        |p: Option<i64>, b: Option<i64>| !cond || matches!((p, b), (Some(p), Some(b)) if p < b);
    expected_by(join_type, &hit, probe, build)
}

/// The reference for the condition `hit`.
fn expected_by(
    join_type: JoinType,
    hit: &dyn Fn(Option<i64>, Option<i64>) -> bool,
    probe: &[Option<i64>],
    build: &[Option<i64>],
) -> Vec<String> {
    let s = |v: Option<i64>| v.map_or_else(|| "N".to_owned(), |v| v.to_string());
    let mut out = Vec::new();
    for &p in probe {
        let matches: Vec<Option<i64>> = build.iter().copied().filter(|&b| hit(p, b)).collect();
        match join_type {
            JoinType::Inner | JoinType::RightOuter => {
                out.extend(matches.iter().map(|&b| format!("{},{}", s(p), s(b))));
            }
            JoinType::LeftOuter | JoinType::FullOuter if matches.is_empty() => {
                out.push(format!("{},N", s(p)));
            }
            JoinType::LeftOuter | JoinType::FullOuter => {
                out.extend(matches.iter().map(|&b| format!("{},{}", s(p), s(b))));
            }
            JoinType::LeftSemi if !matches.is_empty() => out.push(s(p)),
            JoinType::LeftAnti if matches.is_empty() => out.push(s(p)),
            JoinType::Existence => out.push(format!("{},{}", s(p), !matches.is_empty())),
            _ => {}
        }
    }
    for &b in build {
        let matched = probe.iter().any(|&p| hit(p, b));
        match join_type {
            JoinType::RightOuter | JoinType::FullOuter if !matched => {
                out.push(format!("N,{}", s(b)));
            }
            JoinType::RightSemi if matched => out.push(s(b)),
            JoinType::RightAnti if !matched => out.push(s(b)),
            _ => {}
        }
    }
    out.sort();
    out
}

/// Every join type a nested loop runs.
const ALL: [JoinType; 9] = [
    JoinType::Inner,
    JoinType::LeftOuter,
    JoinType::RightOuter,
    JoinType::FullOuter,
    JoinType::LeftSemi,
    JoinType::LeftAnti,
    JoinType::RightSemi,
    JoinType::RightAnti,
    JoinType::Existence,
];

#[test]
fn every_join_type_matches_the_reference() {
    let probe: Vec<Option<i64>> = (0..3000).map(|i| (i % 7 != 3).then_some(i % 50)).collect();
    let build: Vec<Option<i64>> = (0..40).map(|i| (i % 9 != 4).then_some(i)).collect();
    for join_type in ALL {
        for (cond, per_build) in [(true, 7), (true, 40), (false, 13)] {
            let got = run(
                join_type,
                cond.then(less),
                batches("p", &probe, 1024),
                batches("b", &build, per_build),
            )
            .unwrap();
            assert_eq!(
                got,
                expected(join_type, cond, &probe, &build),
                "{join_type:?} cond={cond} per_build={per_build}"
            );
        }
        // An empty build side.
        let got = run(
            join_type,
            Some(less()),
            batches("p", &probe[..10], 4),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            got,
            expected(join_type, true, &probe[..10], &[]),
            "{join_type:?} empty"
        );
        // An empty probe side: the build-side joins still return build rows.
        let got = run(join_type, Some(less()), Vec::new(), batches("b", &build, 7)).unwrap();
        assert_eq!(
            got,
            expected(join_type, true, &[], &build),
            "{join_type:?} no probe rows"
        );
    }
}

#[test]
fn null_aware_and_weighted_inputs_are_refused() {
    let e = NestedLoopJoin::try_new(NestedLoopSpec {
        join_type: JoinType::NullAwareAnti,
        probe_schema: schema("p"),
        build_schema: schema("b"),
        filter: None,
        spill: None,
    })
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    let weighted = |name: &str| {
        let b = &batches(name, &[Some(1), Some(2)], 2)[0];
        vec![Batch::with_weights(b.data().clone(), Weights::Const(2)).unwrap()]
    };
    for (probe, build) in [
        (weighted("p"), batches("b", &[Some(5)], 1)),
        (batches("p", &[Some(1)], 1), weighted("b")),
    ] {
        let e = run(JoinType::Inner, None, probe, build).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    }
    let j = NestedLoopJoin::try_new(NestedLoopSpec {
        join_type: JoinType::LeftOuter,
        probe_schema: schema("p"),
        build_schema: schema("b"),
        filter: None,
        spill: None,
    })
    .unwrap();
    assert!(j.schema().field(1).is_nullable());
    assert!(format!("{j:?}").contains("LeftOuter"));
}

/// A memory limit far below the build's 100,000 rows (about 800 KiB), with
/// room for the result stream's batches.
const LIMIT: usize = 512 << 10;

/// `p <= b < p + 3`, NULL when either is NULL: about three build rows per
/// probe row.
fn near() -> Arc<dyn Predicate> {
    Arc::new(FnPredicate(|b: &RecordBatch| {
        let (p, q) = (
            b.column(0).as_primitive::<Int64Type>(),
            b.column(1).as_primitive::<Int64Type>(),
        );
        Ok(p.iter()
            .zip(q.iter())
            .map(|(x, y)| Some((0..3).contains(&(y? - x?))))
            .collect::<BooleanArray>())
    }))
}

fn is_near(p: Option<i64>, b: Option<i64>) -> bool {
    matches!((p, b), (Some(p), Some(b)) if (0..3).contains(&(b - p)))
}

/// The sum of the counter `name` over the operators.
fn counter(counters: &[(String, u64)], name: &str) -> u64 {
    counters
        .iter()
        .filter(|(k, _)| k == name)
        .map(|(_, v)| *v)
        .sum()
}

/// A build far over the memory limit spills: the join runs build chunk by
/// build chunk over the spilled probe rows and still matches the reference
/// for every join type, NULLs included.
#[test]
fn a_spilled_build_runs_in_chunks() {
    let probe: Vec<Option<i64>> = (0..64)
        .map(|i| (i % 11 != 5).then_some((i * 499) % 100_000))
        .collect();
    let build: Vec<Option<i64>> = (0..100_000)
        .map(|i| (i % 13 != 6).then_some((i * 7) % 100_000))
        .collect();
    for join_type in ALL {
        let (got, counters) = run_with(
            join_type,
            Some(near()),
            batches("p", &probe, 16),
            batches("b", &build, 4_096),
            Some(LIMIT),
        )
        .unwrap();
        assert_eq!(
            got,
            expected_by(join_type, &is_near, &probe, &build),
            "{join_type:?}"
        );
        assert!(
            counter(&counters, "build_spills") > 0,
            "{join_type:?}: {counters:?}"
        );
        assert!(
            counter(&counters, "build_chunks") > 1,
            "{join_type:?}: {counters:?}"
        );
    }
}

/// A spilled build with an empty probe side still returns the build rows
/// of the build-side joins.
#[test]
fn a_spilled_build_with_no_probe_rows() {
    let build: Vec<Option<i64>> = (0..100_000).map(Some).collect();
    for join_type in ALL {
        let (got, counters) = run_with(
            join_type,
            Some(near()),
            Vec::new(),
            batches("b", &build, 4_096),
            Some(LIMIT),
        )
        .unwrap();
        assert_eq!(
            got,
            expected_by(join_type, &is_near, &[], &build),
            "{join_type:?}"
        );
        assert!(
            counter(&counters, "build_spills") > 0,
            "{join_type:?}: {counters:?}"
        );
    }
}

/// Build batches of about a third of the limit: what the build still held
/// in memory when it finished goes to disk too, so a driver's chunk and
/// its spill readers fit beside nothing else of the build.
#[test]
fn a_spilled_build_of_large_batches() {
    let probe: Vec<Option<i64>> = (0..32).map(|i| Some(i * 3_001)).collect();
    let build: Vec<Option<i64>> = (0..100_000)
        .map(|i| (i % 13 != 6).then_some((i * 7) % 100_000))
        .collect();
    for join_type in [JoinType::Inner, JoinType::FullOuter, JoinType::LeftAnti] {
        let (got, counters) = run_with(
            join_type,
            Some(near()),
            batches("p", &probe, 8),
            batches("b", &build, 20_000),
            Some(LIMIT),
        )
        .unwrap();
        assert_eq!(
            got,
            expected_by(join_type, &is_near, &probe, &build),
            "{join_type:?}"
        );
        assert!(
            counter(&counters, "build_spills") > 0,
            "{join_type:?}: {counters:?}"
        );
    }
}

/// Twelve drivers sinking a 24 MB build under a 16 MiB limit from batches
/// of 1.2 MB, which the scan hands out as slices: each slice is held (and
/// charged) for its own rows, not for the whole batch it shares, so the
/// build spills instead of filling the pool with shared buffers.
#[test]
fn many_drivers_spill_wide_build_batches() {
    let probe: Vec<Option<i64>> = (0..16).map(|i| Some(i * 70_001)).collect();
    let build: Vec<Option<i64>> = (0..3_000_000).map(Some).collect();
    let executor = ExecutorConfig {
        compute_threads: 12,
        blocking_threads: 12,
        ..ExecutorConfig::default()
    };
    // One join type: the build is the same for every type.
    let join_type = JoinType::FullOuter;
    let (got, counters) = run_on(
        join_type,
        Some(near()),
        batches("p", &probe, 16),
        batches("b", &build, 150_000),
        Some(16 << 20),
        executor,
    )
    .unwrap();
    assert_eq!(
        got,
        expected_by(join_type, &is_near, &probe, &build),
        "{join_type:?}"
    );
    assert!(
        counter(&counters, "build_spills") > 0,
        "{join_type:?}: {counters:?}"
    );
}

/// A spilled build's chunks leave a quarter of the limit free for the rest
/// of the pool: beside another query holding most of the limit, the loop
/// reads its build in more, smaller chunks than alone (each chunk's own
/// budget counts only its drivers, so without this the chunks of several
/// loops sharing a pool fill it), and still returns the same rows.
#[test]
fn a_spilled_build_leaves_room_for_the_rest_of_the_pool() {
    let probe: Vec<Option<i64>> = (0..64).map(|i| Some((i * 499) % 100_000)).collect();
    let build: Vec<Option<i64>> = (0..300_000).map(|i| Some((i * 7) % 100_000)).collect();
    // One driver, so a chunk's own budget is a quarter of the limit.
    let limit = 2 * LIMIT;
    let one = ExecutorConfig {
        compute_threads: 1,
        blocking_threads: 1,
        ..ExecutorConfig::default()
    };
    let chunks = |hold: usize| {
        let (got, counters) = run_beside(
            JoinType::FullOuter,
            Some(near()),
            batches("p", &probe, 16),
            batches("b", &build, 4_096),
            Some(limit),
            one,
            hold,
            None,
        )
        .unwrap();
        assert_eq!(
            got,
            expected_by(JoinType::FullOuter, &is_near, &probe, &build),
            "hold {hold}"
        );
        assert!(counter(&counters, "build_spills") > 0, "{counters:?}");
        (
            counter(&counters, "build_chunks"),
            counter(&counters, "build_chunks_capped"),
        )
    };
    let ((alone, alone_capped), (beside, capped)) = (chunks(0), chunks(limit * 6 / 10));
    assert!(beside > alone, "alone {alone} chunks, beside {beside}");
    assert_eq!(alone_capped, 0);
    assert!(capped > 0, "beside: {capped} chunks cut short");
}

/// The chunks of nested loops in one query split one quarter of its limit:
/// with another member in the cohort, a loop's chunks are half as large, so
/// it reads more of them than alone, and none is cut short.
#[test]
fn nested_loops_of_one_query_share_the_chunk_budget() {
    let probe: Vec<Option<i64>> = (0..64).map(|i| Some((i * 499) % 100_000)).collect();
    let build: Vec<Option<i64>> = (0..300_000).map(|i| Some((i * 7) % 100_000)).collect();
    let one = ExecutorConfig {
        compute_threads: 1,
        blocking_threads: 1,
        ..ExecutorConfig::default()
    };
    let chunks = |peer: Option<Peer>| {
        let (got, counters) = run_beside(
            JoinType::FullOuter,
            Some(near()),
            batches("p", &probe, 16),
            batches("b", &build, 4_096),
            Some(2 * LIMIT),
            one,
            0,
            peer,
        )
        .unwrap();
        assert_eq!(
            got,
            expected_by(JoinType::FullOuter, &is_near, &probe, &build)
        );
        assert!(counter(&counters, "build_spills") > 0, "{counters:?}");
        assert_eq!(counter(&counters, "build_chunks_capped"), 0, "{counters:?}");
        assert_eq!(counter(&counters, "build_chunk_waits"), 0, "{counters:?}");
        counter(&counters, "build_chunks")
    };
    // An idle peer seat for the whole run: it never makes the loop wait.
    let seat = Peer {
        release_after: std::time::Duration::from_secs(3_600),
    };
    let (alone, shared) = (chunks(None), chunks(Some(seat)));
    assert!(
        shared >= alone * 3 / 2,
        "alone {alone} chunks, sharing {shared}"
    );
}

/// Runs two copies of one spilled nested loop at once as two tasks of one
/// query (two task pools below one query pool under `limit`, as a stage's
/// split tasks run); the sorted rows and counters of each.
fn run_two_tasks(
    probe: &[Option<i64>],
    build: &[Option<i64>],
    limit: usize,
) -> [(Vec<String>, Counters); 2] {
    let (_dir, scope) = spill_scope();
    let root = TreePool::new(&MemoryConfig {
        limit: Some(limit),
        quantum: 4 << 10,
        spill_scratch_reserve: 32 << 10,
        ..MemoryConfig::default()
    });
    let q = root.child("q", None);
    let one = ExecutorConfig {
        compute_threads: 1,
        blocking_threads: 1,
        ..ExecutorConfig::default()
    };
    let task = |i: u64| {
        let source = |name: &str, b: Vec<Batch>| {
            MemoryScan::try_new(schema(name), b)
                .unwrap()
                .build(ScanPushdowns::default())
                .unwrap()
        };
        let join = NestedLoopJoin::try_new(NestedLoopSpec {
            join_type: JoinType::FullOuter,
            probe_schema: schema("p"),
            build_schema: schema("b"),
            filter: Some(near()),
            spill: Some(scope.clone()),
        })
        .unwrap();
        let plan = Arc::new(NestedLoopJoinNode {
            join,
            build: Arc::new(Scan(source("b", batches("b", build, 4_096)))),
            probe: Arc::new(Scan(source("p", batches("p", probe, 16)))),
        });
        let query = QueryContext::new(
            QueryId(i),
            WorkloadClass::Interactive,
            ExecConfig::default(),
            q.child(&format!("task{i}"), None),
        );
        let runner = QueryRunner::new(TokioExecutor::new(&one).unwrap());
        (plan, query, runner)
    };
    let (t1, t2) = (task(1), task(2));
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let run = |(plan, query, runner): (Arc<NestedLoopJoinNode>, Arc<QueryContext>, QueryRunner)| async move {
        let mut stream = runner.execute(plan, Arc::clone(&query)).await.unwrap();
        let mut rows = Vec::new();
        while let Some(b) = stream.next().await {
            let b = b.unwrap().into_parts().0;
            for r in 0..b.num_rows() {
                let cells: Vec<String> = b
                    .columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(r) {
                            "N".to_owned()
                        } else {
                            c.as_primitive::<Int64Type>().value(r).to_string()
                        }
                    })
                    .collect();
                rows.push(cells.join(","));
            }
        }
        query.finished().await;
        rows.sort();
        let counters: Counters = query
            .metrics()
            .operators
            .values()
            .flat_map(|o| o.extra.iter())
            .filter_map(|(k, v)| match v {
                Metric::Count(n) => Some((k.to_string(), *n)),
                _ => None,
            })
            .collect();
        (rows, counters)
    };
    let (a, b) = rt.block_on(async { tokio::join!(run(t1), run(t2)) });
    assert_eq!(root.reserved(), 0, "memory is released");
    [a, b]
}

/// Two real nested loops of one query sharing a small limit: both return
/// every row, and together they read more chunks than one alone (their
/// chunks split one budget). Under part 1 alone (no cohort) the second
/// loop's first batch could fail the query.
#[test]
fn two_tasks_of_one_query_share_their_chunks() {
    let probe: Vec<Option<i64>> = (0..64).map(|i| Some((i * 499) % 100_000)).collect();
    let build: Vec<Option<i64>> = (0..300_000).map(|i| Some((i * 7) % 100_000)).collect();
    let want = expected_by(JoinType::FullOuter, &is_near, &probe, &build);
    let (_, alone) = run_with(
        JoinType::FullOuter,
        Some(near()),
        batches("p", &probe, 16),
        batches("b", &build, 4_096),
        Some(2 * LIMIT),
    )
    .unwrap();
    let alone = counter(&alone, "build_chunks");
    let [(a, ca), (b, cb)] = run_two_tasks(&probe, &build, 2 * LIMIT);
    assert_eq!(a, want);
    assert_eq!(b, want);
    for c in [&ca, &cb] {
        assert!(counter(c, "build_spills") > 0, "{c:?}");
    }
    let together = counter(&ca, "build_chunks") + counter(&cb, "build_chunks");
    assert!(
        together > 2 * alone,
        "alone {alone} chunks, together {together}: {ca:?} {cb:?}"
    );
}
