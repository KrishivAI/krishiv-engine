// SPDX-License-Identifier: Apache-2.0
//! Hash join benchmarks, single thread (CodSpeed simulation mode counts
//! instructions; a multi-thread runtime would mostly measure parking).
//!
//! - `hash_join/build_1m_rows`: build 1 M unique `Int64` keys with a
//!   payload column, probe one batch (the build dominates).
//! - `hash_join/build_1m_rows_runtime_filter`: the same build publishing a
//!   runtime filter (min/max, value set, Bloom filter of 1 M keys): the
//!   producer's overhead (design P2-T163 targets at most 5%).
//! - `hash_join/probe_1m_rows`: build 64 Ki keys, probe 1 M rows of which
//!   half match (inner join, build columns decoded).
//! - `hash_join/probe_string_keys_1m_rows`: the same join on 12-byte `Utf8`
//!   keys.
//! - `hash_join/probe_zipf_1m_rows`: the same on `Int64` keys where the probe
//!   side is Zipf-skewed (exponent 1: a few build keys take most probes).
//! - `hash_join/grace_spill_256k_rows`: 256 Ki ⋈ 256 Ki rows under a 4 MiB
//!   budget: partitions spill, are reloaded and joined (the spill path).
//! - `…_row_layout`: the same joins with `krishiv.exec.join.rowLayout` on
//!   (P2-T158: row-layout keys and the size-adaptive index).

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use krishiv_exec::testing::{Batches, Scan};
use krishiv_exec::{
    DEFAULT_BATCH_SIZE, ExecConfig, ExecNode, HashJoin, HashJoinNode, JoinRuntimeFilter, JoinSpec,
    JoinType, QueryContext, QueryId, QueryRunner, TokioClock, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillScope, TreePool};
use krishiv_types::batch::Batch;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]))
}

/// `rows` rows with keys `i * stride`, in 4,096-row batches.
fn data(rows: i64, stride: i64) -> Vec<Batch> {
    let step = i64::try_from(DEFAULT_BATCH_SIZE).expect("batch size");
    (0..rows)
        .step_by(DEFAULT_BATCH_SIZE)
        .map(|start| {
            let end = (start + step).min(rows);
            let k = Int64Array::from_iter_values((start..end).map(|i| i * stride));
            let v = Int64Array::from_iter_values(start..end);
            let b = RecordBatch::try_new(schema(), vec![Arc::new(k), Arc::new(v)]);
            Batch::new(b.expect("batch"))
        })
        .collect()
}

fn string_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
    ]))
}

/// `rows` rows with key `key-<key(i)>` (12 bytes), in 4,096-row batches.
fn string_data(rows: i64, key: impl Fn(i64) -> i64) -> Vec<Batch> {
    let step = i64::try_from(DEFAULT_BATCH_SIZE).expect("batch size");
    (0..rows)
        .step_by(DEFAULT_BATCH_SIZE)
        .map(|start| {
            let end = (start + step).min(rows);
            let k =
                StringArray::from_iter_values((start..end).map(|i| format!("key-{:08}", key(i))));
            let v = Int64Array::from_iter_values(start..end);
            let cols: Vec<ArrayRef> = vec![Arc::new(k), Arc::new(v)];
            Batch::new(RecordBatch::try_new(string_schema(), cols).expect("batch"))
        })
        .collect()
}

/// A Zipf(1) draw over `0..n` for row `i`: the inverse CDF `n^u - 1` of a
/// deterministic uniform `u`, so key 0 takes about `1/ln n` of the rows.
fn zipf(i: i64, n: i64) -> i64 {
    let x = i.cast_unsigned().wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11;
    let u = x as f64 / (1u64 << 53) as f64;
    // `u < 1`, so the draw is below `n`; the cast truncates toward 0 on purpose.
    #[expect(clippy::cast_possible_truncation, reason = "a rank below n")]
    let rank = ((n as f64).powf(u) - 1.0) as i64;
    rank.min(n - 1)
}

fn plan(build: &[Batch], probe: &[Batch], spill: Option<SpillScope>) -> Arc<dyn ExecNode> {
    plan_with(build, probe, spill, false)
}

fn plan_with(
    build: &[Batch],
    probe: &[Batch],
    spill: Option<SpillScope>,
    runtime_filter: bool,
) -> Arc<dyn ExecNode> {
    let schema = build[0].data().schema();
    let spec = JoinSpec {
        join_type: JoinType::Inner,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        single_match: false,
        spill,
    };
    let scan = |b: &[Batch]| -> Arc<dyn ExecNode> {
        Arc::new(Scan(Batches::new(Arc::clone(&schema), b.to_vec())))
    };
    let mut join = HashJoin::try_new(spec).expect("join");
    if runtime_filter {
        let filter = krishiv_io::RuntimeFilter::new(
            krishiv_io::FilterId(1),
            krishiv_io::ColumnPath::new("k"),
            DataType::Int64,
        );
        join = join
            .with_runtime_filters(vec![JoinRuntimeFilter {
                key: 0,
                filter: Arc::new(filter),
                options: krishiv_io::KeyFilterOptions::default(),
            }])
            .expect("runtime filter");
    }
    Arc::new(HashJoinNode {
        join,
        build: scan(build),
        probe: scan(probe),
    })
}

fn run(
    runner: &QueryRunner,
    caller: &tokio::runtime::Runtime,
    plan: Arc<dyn ExecNode>,
    limit: Option<usize>,
    join_row_layout: bool,
) -> usize {
    // The pool's spill and abort waits on the caller's paused clock: they
    // pass only while every task waits, never because the machine is slow.
    let root = TreePool::with_clock(
        &MemoryConfig {
            limit,
            ..MemoryConfig::default()
        },
        Arc::new(TokioClock),
    );
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig {
            join_row_layout,
            ..ExecConfig::default()
        },
        root.child("bench", None),
    );
    caller.block_on(async {
        let mut stream = runner
            .execute(plan, Arc::clone(&query))
            .await
            .expect("plan");
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b.expect("batch").num_rows();
        }
        query.finished().await;
        rows
    })
}

fn bench(c: &mut Criterion) {
    let caller = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .build()
        .expect("caller runtime");
    let runner = {
        let _in_runtime = caller.enter();
        QueryRunner::new(TokioExecutor::deterministic(1).expect("executor"))
    };
    let dir = std::env::temp_dir().join(format!("krishiv-bench-join-{}", std::process::id()));
    let manager = SpillManager::open(&SpillConfig {
        dir: dir.clone(),
        ..SpillConfig::default()
    })
    .expect("spill dir");
    let mut g = c.benchmark_group("hash_join");
    g.sample_size(10);

    let (big, one) = (data(1 << 20, 1), data(1, 1));
    let (small, probe) = (data(1 << 16, 2), data(1 << 20, 1));
    let side = data(1 << 18, 1);
    // Build 64 Ki unique keys; the probe keys are `2 * key` (half match) in
    // the string case, and Zipf-drawn build keys in the skewed one.
    let small_str = string_data(1 << 16, |i| i * 2);
    let probe_str = string_data(1 << 20, |i| i);
    let zipf_build = data(1 << 16, 1);
    let zipf_probe = {
        let step = DEFAULT_BATCH_SIZE as i64;
        (0..1i64 << 20)
            .step_by(DEFAULT_BATCH_SIZE)
            .map(|start| {
                let end = (start + step).min(1 << 20);
                let k = Int64Array::from_iter_values((start..end).map(|i| zipf(i, 1 << 16)));
                let v = Int64Array::from_iter_values(start..end);
                let b = RecordBatch::try_new(schema(), vec![Arc::new(k), Arc::new(v)]);
                Batch::new(b.expect("batch"))
            })
            .collect::<Vec<_>>()
    };
    for (suffix, rows) in [("", false), ("_row_layout", true)] {
        g.throughput(Throughput::Elements(1 << 20));
        g.bench_function(format!("build_1m_rows{suffix}"), |b| {
            b.iter(|| black_box(run(&runner, &caller, plan(&big, &one, None), None, rows)));
        });
        if !rows {
            g.bench_function("build_1m_rows_runtime_filter", |b| {
                b.iter(|| {
                    let plan = plan_with(&big, &one, None, true);
                    black_box(run(&runner, &caller, plan, None, false))
                });
            });
        }
        g.bench_function(format!("probe_1m_rows{suffix}"), |b| {
            b.iter(|| {
                black_box(run(
                    &runner,
                    &caller,
                    plan(&small, &probe, None),
                    None,
                    rows,
                ))
            });
        });
        g.bench_function(format!("probe_string_keys_1m_rows{suffix}"), |b| {
            b.iter(|| {
                black_box(run(
                    &runner,
                    &caller,
                    plan(&small_str, &probe_str, None),
                    None,
                    rows,
                ))
            });
        });
        g.bench_function(format!("probe_zipf_1m_rows{suffix}"), |b| {
            b.iter(|| {
                black_box(run(
                    &runner,
                    &caller,
                    plan(&zipf_build, &zipf_probe, None),
                    None,
                    rows,
                ))
            });
        });
        g.throughput(Throughput::Elements(1 << 18));
        g.bench_function(format!("grace_spill_256k_rows{suffix}"), |b| {
            b.iter(|| {
                let scope = manager.scope().expect("scope");
                let plan = plan(&side, &side, Some(scope));
                black_box(run(&runner, &caller, plan, Some(4 << 20), rows))
            });
        });
    }
    g.finish();
    drop(manager);
    // Best effort: the scratch directory is empty once every file dropped.
    let _ = std::fs::remove_dir_all(dir);
}

criterion_group!(benches, bench);
criterion_main!(benches);
