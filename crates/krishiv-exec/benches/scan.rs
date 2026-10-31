// SPDX-License-Identifier: Apache-2.0
//! Parquet scan throughput over one 1 Mi-row file in an in-memory object
//! store, on one deterministic worker: scan → count.
//!
//! - `scan/parquet_int_1m_rows`: a random `Int64` column.
//! - `scan/parquet_string_1m_rows`: a row number and a 12-byte `Utf8View` column (the engine's string type).

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::expect_used,
    reason = "a broken fixture or a failing query must fail the bench loudly"
)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use krishiv_exec::testing::{Break, Count, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, FileScan, QueryContext, QueryId, QueryRunner, ScanPushdowns,
    TokioExecutor, WorkloadClass,
};
use krishiv_io::io::IoScheduler;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::split::SplitConfig;
use krishiv_io::{FileFormat, FileRef, FormatOptions, IoConfig, IoContext, ScanOptions, StoreId};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};

const ROWS: usize = 1 << 20;
const BATCH_ROWS: usize = 8192;

fn mix(i: u64) -> u64 {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^ (z >> 31)
}

/// `ROWS` rows: a random `Int64` column `c`, and for the string case a
/// 12-byte string column `s` after it (`Count` sums the first column, which
/// must be an `Int64`).
fn batches(strings: bool) -> (SchemaRef, Vec<RecordBatch>) {
    let mut fields = vec![Field::new("c", DataType::Int64, false)];
    if strings {
        fields.push(Field::new("s", DataType::Utf8View, false));
    }
    let schema = Arc::new(Schema::new(fields));
    let out = (0..ROWS)
        .step_by(BATCH_ROWS)
        .map(|start| {
            let ids = start as u64..(start + BATCH_ROWS) as u64;
            let mut cols: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(
                ids.clone().map(|i| (mix(i) >> 1) as i64),
            ))];
            if strings {
                cols.push(Arc::new(StringViewArray::from_iter_values(
                    ids.map(|i| format!("{:012x}", mix(i) >> 16)),
                )));
            }
            RecordBatch::try_new(Arc::clone(&schema), cols).expect("batch")
        })
        .collect();
    (schema, out)
}

/// Writes the batches as one Parquet file `t/f` and lists it.
async fn write(
    io: &IoContext,
    store: &Arc<InMemory>,
    id: &StoreId,
    schema: &SchemaRef,
    data: &[RecordBatch],
) -> FileRef {
    let format = ParquetFormat;
    let options = FormatOptions::defaults(format.options());
    let mut enc = format
        .writer(io, Arc::clone(schema), &options)
        .await
        .expect("writer");
    let path = Path::from("t/f");
    let mut upload = io.upload(id, path.clone()).expect("upload");
    for b in data {
        enc.write(b).expect("encode");
        while let Some(bytes) = enc.take_bytes() {
            upload.write(bytes).await.expect("upload bytes");
        }
    }
    let done = enc.finish().expect("finish");
    upload.write(done.tail).await.expect("tail");
    upload.finish().await.expect("finish upload");
    let meta = store.head(&path).await.expect("head");
    FileRef::from_meta(id.clone(), &meta)
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
    let mut g = c.benchmark_group("scan");
    g.throughput(Throughput::Elements(ROWS as u64));
    g.sample_size(10);
    for (name, strings) in [
        ("parquet_int_1m_rows", false),
        ("parquet_string_1m_rows", true),
    ] {
        let store = Arc::new(InMemory::new());
        let id = StoreId::new("memory://bench");
        let sched = IoScheduler::new(IoConfig::default());
        sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
        let io = sched.context();
        let (schema, data) = batches(strings);
        let file = caller.block_on(write(&io, &store, &id, &schema, &data));
        let scan = FileScan::new(
            Arc::new(ParquetFormat),
            io,
            schema,
            vec![file],
            FormatOptions::defaults(ParquetFormat.options()),
            ScanOptions::default(),
            SplitConfig {
                max_partition_bytes: 1 << 30,
                open_cost_in_bytes: 0,
                min_partition_num: 1,
                max_partition_num: None,
            },
        );
        g.bench_function(name, |b| {
            b.iter(|| {
                let source = scan.source(ScanPushdowns::default()).expect("source");
                let plan: Arc<dyn ExecNode> =
                    Arc::new(Break(Arc::new(Scan(Arc::new(source))), Count::new(), false));
                let root = TreePool::new(&MemoryConfig::default());
                let query = QueryContext::new(
                    QueryId(1),
                    WorkloadClass::Batch,
                    ExecConfig::default(),
                    root.child("bench", None),
                );
                black_box(caller.block_on(async {
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
                }))
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
