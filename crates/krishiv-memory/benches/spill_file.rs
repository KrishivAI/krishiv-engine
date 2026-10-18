// SPDX-License-Identifier: Apache-2.0
//! Spill write and read-back throughput of 1M rows (an `Int64` key, a decimal,
//! a timestamp, a low-cardinality string and a float, with unit weights) for
//! LZ4, Zstandard and no compression; run under CodSpeed in CI.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Decimal128Array, Float64Array, Int64Array, RecordBatch, StringViewArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use krishiv_memory::{
    MemoryConfig, MemoryPool, Reservation, SpillCodec, SpillConfig, SpillManager, TreePool,
};
use krishiv_types::batch::Batch;

const ROWS: usize = 1_000_000;
const BATCH_ROWS: usize = 8192;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("amount", DataType::Decimal128(18, 2), true),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("tag", DataType::Utf8View, true),
        Field::new("x", DataType::Float64, false),
    ]))
}

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn batches() -> Vec<Batch> {
    let schema = schema();
    (0..ROWS)
        .step_by(BATCH_ROWS)
        .map(|start| {
            let r = start..(start + BATCH_ROWS).min(ROWS);
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter_values(r.clone().map(|i| i as i64))),
                Arc::new(
                    Decimal128Array::from_iter(
                        r.clone()
                            .map(|i| (i % 17 != 0).then_some((i as i128 % 100_000) * 100)),
                    )
                    .with_precision_and_scale(18, 2)
                    .expect("valid decimal"),
                ),
                Arc::new(TimestampMicrosecondArray::from_iter_values(
                    r.clone()
                        .map(|i| 1_700_000_000_000_000 + (i as i64) * 1_000),
                )),
                Arc::new(StringViewArray::from_iter(
                    r.clone()
                        .map(|i| (i % 11 != 0).then(|| format!("category-{}", i % 64))),
                )),
                Arc::new(Float64Array::from_iter_values(r.map(|i| (i as f64).sqrt()))),
            ];
            Batch::new(RecordBatch::try_new(Arc::clone(&schema), columns).expect("valid batch"))
        })
        .collect()
}

#[expect(
    clippy::expect_used,
    reason = "a bench that cannot spill must fail loudly"
)]
fn bench_spill(c: &mut Criterion) {
    let input = batches();
    let in_memory: usize = input.iter().map(|b| b.data().get_array_memory_size()).sum();
    // Removed when the bench returns or panics.
    let root = tempfile::Builder::new()
        .prefix("krishiv-bench-spill-")
        .tempdir()
        .expect("temporary directory");
    let pool = TreePool::new(&MemoryConfig::default()).child("bench", None);
    let id = pool.register("spill", None);

    let mut group = c.benchmark_group("spill_1m_rows");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(in_memory as u64));
    for (name, codec) in [
        ("lz4", SpillCodec::Lz4),
        ("zstd", SpillCodec::Zstd),
        ("none", SpillCodec::None),
    ] {
        let manager = SpillManager::open(&SpillConfig {
            dir: root.path().join("spill"),
            codec,
            ..SpillConfig::default()
        })
        .expect("spill dir");
        let scope = manager.scope().expect("scope");
        let spill = || {
            let mut writer = scope
                .writer(schema(), Reservation::new(Arc::clone(&pool), id))
                .expect("writer");
            for batch in &input {
                writer.write(batch).expect("write");
            }
            writer.finish().expect("finish")
        };
        group.bench_function(format!("write_{name}"), |b| {
            b.iter(|| black_box(spill().bytes()));
        });
        let file = spill();
        group.bench_function(format!("read_{name}"), |b| {
            b.iter(|| {
                let reader = file
                    .reader(Reservation::new(Arc::clone(&pool), id))
                    .expect("reader");
                black_box(
                    reader
                        .map(|batch| batch.expect("read").num_rows())
                        .sum::<usize>(),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_spill);
criterion_main!(benches);
