// SPDX-License-Identifier: Apache-2.0
//! The sort breakers over a prebuilt 1M-row table, on one deterministic
//! worker: `in_memory_1m_rows/{int,string}` (scan → sort by a random
//! `Int64` or a 12-byte string key), `in_memory_1m_rows/two_int_keys` (a
//! 100-value key, then a random one), `topn_k100_1m_rows` (`LIMIT 100`),
//! `topn_k1000_float_1m_rows` and `topn_k100_string_1m_rows` (the same over a
//! `Float64` and a string key) and
//! `external_1m_rows/int` (under a 4 MiB cap, about a tenth of the data:
//! sorted runs spill and merge with fan-in 8 through spill files).

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::expect_used,
    reason = "a broken fixture or a failing query must fail the bench loudly"
)]

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use krishiv_exec::testing::{Break, Breaker, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, MemoryScan, QueryContext, QueryId, QueryRunner, Sort, SortKey, Source,
    TokioClock, TokioExecutor, TopN, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, TreePool};
use krishiv_types::batch::Batch;

const BATCHES: usize = 256;
const BATCH_ROWS: usize = 4096;
const ROWS: u64 = (BATCHES * BATCH_ROWS) as u64;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]))
}

/// `k`, a 100-value `g`, a float `f` and the string `s`: the keys of the
/// multi-key, float and string top-N cases.
fn wide_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("g", DataType::Int64, false),
        Field::new("f", DataType::Float64, false),
        Field::new("s", DataType::Utf8, false),
    ]))
}

fn mix(i: u64) -> u64 {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^ (z >> 31)
}

fn table() -> Vec<Batch> {
    (0..BATCHES)
        .map(|b| {
            let ids = (b * BATCH_ROWS) as u64..((b + 1) * BATCH_ROWS) as u64;
            let k = Int64Array::from_iter_values(ids.clone().map(|i| (mix(i) >> 1) as i64));
            let s = StringArray::from_iter_values(ids.map(|i| format!("{:012x}", mix(i) >> 16)));
            Batch::new(
                RecordBatch::try_new(schema(), vec![Arc::new(k), Arc::new(s)]).expect("batch"),
            )
        })
        .collect()
}

fn wide_table() -> Vec<Batch> {
    (0..BATCHES)
        .map(|b| {
            let ids = (b * BATCH_ROWS) as u64..((b + 1) * BATCH_ROWS) as u64;
            let k = Int64Array::from_iter_values(ids.clone().map(|i| (mix(i) >> 1) as i64));
            let g = Int64Array::from_iter_values(ids.clone().map(|i| (mix(i ^ 7) % 100) as i64));
            let f = Float64Array::from_iter_values(
                ids.clone().map(|i| (mix(i ^ 11) >> 11) as f64 / 1e6),
            );
            let s = StringArray::from_iter_values(ids.map(|i| format!("{:012x}", mix(i) >> 16)));
            let cols: Vec<ArrayRef> = vec![Arc::new(k), Arc::new(g), Arc::new(f), Arc::new(s)];
            Batch::new(RecordBatch::try_new(wide_schema(), cols).expect("batch"))
        })
        .collect()
}

fn plan<B: Breaker + 'static>(table: &[Batch], breaker: B) -> Arc<dyn ExecNode> {
    plan_of(schema(), table, breaker)
}

fn plan_of<B: Breaker + 'static>(
    schema: SchemaRef,
    table: &[Batch],
    breaker: B,
) -> Arc<dyn ExecNode> {
    let scan: Arc<dyn Source> =
        Arc::new(MemoryScan::try_new(schema, table.to_vec()).expect("scan"));
    Arc::new(Break(Arc::new(Scan(scan)), Arc::new(breaker), true))
}

fn run(
    runner: &QueryRunner,
    caller: &tokio::runtime::Runtime,
    plan: Arc<dyn ExecNode>,
    limit: Option<usize>,
) -> usize {
    let root = TreePool::with_clock(&MemoryConfig::default(), Arc::new(TokioClock));
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("bench", limit),
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
        .build()
        .expect("caller runtime");
    let executor = {
        let _in_runtime = caller.enter();
        TokioExecutor::deterministic(1).expect("executor")
    };
    let runner = QueryRunner::new(executor);
    let table = table();
    let spill = SpillManager::open(&SpillConfig {
        dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/bench-spill"),
        ..SpillConfig::default()
    })
    .expect("spill directory");
    let sort = |key: usize| Sort::new(schema(), &[SortKey::asc(key)], None).expect("sort");

    let mut g = c.benchmark_group("sort");
    g.throughput(Throughput::Elements(ROWS));
    g.sample_size(10);
    g.bench_function("in_memory_1m_rows/int", |b| {
        b.iter(|| black_box(run(&runner, &caller, plan(&table, sort(0)), None)));
    });
    g.bench_function("in_memory_1m_rows/string", |b| {
        b.iter(|| black_box(run(&runner, &caller, plan(&table, sort(1)), None)));
    });
    let wide = wide_table();
    g.bench_function("in_memory_1m_rows/two_int_keys", |b| {
        b.iter(|| {
            let keys = [SortKey::asc(1), SortKey::asc(0)];
            let sort = Sort::new(wide_schema(), &keys, None).expect("sort");
            black_box(run(
                &runner,
                &caller,
                plan_of(wide_schema(), &wide, sort),
                None,
            ))
        });
    });
    for (name, key, k) in [
        ("topn_k1000_float_1m_rows", 2, 1000),
        ("topn_k100_string_1m_rows", 3, 100),
    ] {
        g.bench_function(name, |b| {
            b.iter(|| {
                let topn = TopN::new(wide_schema(), &[SortKey::asc(key)], 0, k).expect("topn");
                black_box(run(
                    &runner,
                    &caller,
                    plan_of(wide_schema(), &wide, topn),
                    None,
                ))
            });
        });
    }
    g.bench_function("topn_k100_1m_rows", |b| {
        b.iter(|| {
            let topn = TopN::new(schema(), &[SortKey::asc(0)], 0, 100).expect("topn");
            black_box(run(&runner, &caller, plan(&table, topn), None))
        });
    });
    g.bench_function("external_1m_rows/int", |b| {
        b.iter(|| {
            let scope = spill.scope().expect("spill scope");
            let sort = Sort::new(schema(), &[SortKey::asc(0)], Some(scope))
                .expect("sort")
                .with_max_fan_in(8);
            black_box(run(&runner, &caller, plan(&table, sort), Some(4 << 20)))
        });
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
