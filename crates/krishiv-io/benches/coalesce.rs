// SPDX-License-Identifier: Apache-2.0
//! Coalescing 10,000 column-chunk ranges of one file into requests, run
//! under CodSpeed in CI in its own binary (single-threaded, no I/O) so
//! allocator state from the Parquet encode/scan benchmarks cannot swing it.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::ops::Range;

use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_io::io::coalesce;

/// Column chunks of 500 row groups × 20 columns, a few KiB to a few MiB
/// apart, listed in column-major order (as a projection asks for them).
fn ranges() -> Vec<Range<u64>> {
    let mut out = Vec::with_capacity(10_000);
    for col in 0..20_u64 {
        for rg in 0..500_u64 {
            let start = rg * (8 << 20) + col * (300 << 10) + (rg * col) % 4096;
            out.push(start..start + (200 << 10) + (col * 977) % 65_536);
        }
    }
    out
}

fn bench(c: &mut Criterion) {
    let ranges = ranges();
    let mut group = c.benchmark_group("io");
    group.bench_function("coalesce_10k_ranges", |b| {
        b.iter(|| coalesce(black_box(&ranges), 1 << 20, 16 << 20));
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
