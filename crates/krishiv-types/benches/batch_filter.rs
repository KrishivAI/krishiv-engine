// SPDX-License-Identifier: Apache-2.0
//! `Batch::filter` with unit weights (`Const(1)`) and with an Int64 weight
//! column, against filtering the bare `RecordBatch`; run under CodSpeed in
//! CI. The `Const(1)` case should cost the same as the bare baseline.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{BooleanArray, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::filter::filter_record_batch;
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_types::batch::{Batch, Weights};

const ROWS: i32 = 4096;

struct Inputs {
    bare: RecordBatch,
    unweighted: Batch,
    weighted: Batch,
    mask: BooleanArray,
}

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn inputs() -> Inputs {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let ids: Vec<i64> = (0..ROWS).map(i64::from).collect();
    let values: Vec<f64> = (0..ROWS).map(f64::from).collect();
    let weights: Vec<i64> = (0..ROWS).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
    // About half the rows, in an irregular pattern.
    let mask: BooleanArray = (0..ROWS).map(|i| Some((i * 7919) % 13 < 6)).collect();
    let bare = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("bench fixture: record batch");
    let weighted = Batch::with_weights(bare.clone(), Weights::Column(Int64Array::from(weights)))
        .expect("bench fixture: weighted batch");
    Inputs {
        unweighted: Batch::new(bare.clone()),
        bare,
        weighted,
        mask,
    }
}

fn bench_filter(c: &mut Criterion) {
    let Inputs {
        bare,
        unweighted,
        weighted,
        mask,
    } = inputs();
    let mut group = c.benchmark_group("batch_filter");
    group.bench_function("bare_record_batch", |b| {
        b.iter(|| filter_record_batch(black_box(&bare), black_box(&mask)));
    });
    group.bench_function("const_one", |b| {
        b.iter(|| black_box(&unweighted).filter(black_box(&mask)));
    });
    group.bench_function("int64_weights", |b| {
        b.iter(|| black_box(&weighted).filter(black_box(&mask)));
    });
    group.finish();
}

criterion_group!(benches, bench_filter);
criterion_main!(benches);
