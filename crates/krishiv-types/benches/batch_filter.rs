// SPDX-License-Identifier: Apache-2.0
//! `Batch::filter` with constant (`Const(1)`) versus Int64 column weights,
//! run under CodSpeed in CI. The constant case should cost the same as
//! filtering the bare `RecordBatch`.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{BooleanArray, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_types::batch::{Batch, Weights};

const ROWS: i32 = 4096;

fn inputs() -> Option<(Batch, Batch, BooleanArray)> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let ids: Vec<i64> = (0..ROWS).map(i64::from).collect();
    let values: Vec<f64> = (0..ROWS).map(f64::from).collect();
    let weights: Vec<i64> = (0..ROWS).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
    // About half the rows, in an irregular pattern.
    let mask: BooleanArray = (0..ROWS).map(|i| Some((i * 7919) % 13 < 6)).collect();
    let data = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .ok()?;
    let column =
        Batch::with_weights(data.clone(), Weights::Column(Int64Array::from(weights))).ok()?;
    Some((Batch::new(data), column, mask))
}

fn bench_filter(c: &mut Criterion) {
    let Some((constant, column, mask)) = inputs() else {
        return;
    };
    let mut group = c.benchmark_group("batch_filter");
    group.bench_function("const_one", |b| {
        b.iter(|| black_box(&constant).filter(black_box(&mask)));
    });
    group.bench_function("int64_weights", |b| {
        b.iter(|| black_box(&column).filter(black_box(&mask)));
    });
    group.finish();
}

criterion_group!(benches, bench_filter);
criterion_main!(benches);
