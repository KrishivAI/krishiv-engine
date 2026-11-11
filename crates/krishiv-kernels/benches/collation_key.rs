// SPDX-License-Identifier: Apache-2.0
//! Collation key benchmarks (64K rows, one thread; CodSpeed in CI): the `UTF8_LCASE` key of an
//! ASCII column (the byte path) and of a mixed column (the per-code-point table), and the
//! `UTF8_LCASE_RTRIM` key of padded ASCII.

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

use arrow_array::{ArrayRef, StringArray};
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_types::{Collation, DataType};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const ROWS: usize = 1 << 16;

fn column(f: impl Fn(usize) -> String) -> ArrayRef {
    Arc::new((0..ROWS).map(|i| Some(f(i))).collect::<StringArray>())
}

fn collation_key_benches(c: &mut Criterion) {
    let lcase = DataType::collated_string(Collation::parse("UTF8_LCASE").unwrap());
    let rtrim = DataType::collated_string(Collation::parse("UTF8_LCASE_RTRIM").unwrap());
    let ascii = column(|i| format!("Customer#{i:09}"));
    let mixed = column(|i| format!("Straße-ΣΑΣ-{i}"));
    let padded = column(|i| format!("Name{i}      "));
    c.bench_function("collation_key_lcase_ascii", |b| {
        b.iter(|| krishiv_kernels::string::collation_key(&lcase, black_box(&ascii)).unwrap());
    });
    c.bench_function("collation_key_lcase_mixed", |b| {
        b.iter(|| krishiv_kernels::string::collation_key(&lcase, black_box(&mixed)).unwrap());
    });
    c.bench_function("collation_key_lcase_rtrim", |b| {
        b.iter(|| krishiv_kernels::string::collation_key(&rtrim, black_box(&padded)).unwrap());
    });
}

criterion_group!(benches, collation_key_benches);
criterion_main!(benches);
