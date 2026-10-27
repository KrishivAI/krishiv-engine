// SPDX-License-Identifier: Apache-2.0
//! List kernel benchmarks (64K rows, one thread; CodSpeed in CI):
//! `array` interleaves its arguments into one list per row.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::unwrap_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array};
use criterion::{Criterion, criterion_group, criterion_main};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const ROWS: usize = 1 << 16;

fn list_benches(c: &mut Criterion) {
    let cols: Vec<ArrayRef> = (0..3)
        .map(|k| {
            let a: Int32Array = (0..ROWS as i64)
                .map(|i| (i % 10 != k).then(|| (i.wrapping_mul(7919 + k) % 1000) as i32))
                .collect();
            Arc::new(a) as ArrayRef
        })
        .collect();
    c.bench_function("array_i32_x3", |bench| {
        bench.iter(|| krishiv_kernels::list::create(black_box(&cols), ROWS, true).unwrap());
    });
}

criterion_group!(benches, list_benches);
criterion_main!(benches);
