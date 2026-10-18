// SPDX-License-Identifier: Apache-2.0
//! File-scan planning and I/O request shaping, run under CodSpeed in CI
//! (single-threaded, no I/O):
//! - Spark split planning of 10,000 listed files (sizes up to 300 MiB, a
//!   third not splittable);
//! - coalescing 10,000 column-chunk ranges of one file into requests.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::ops::Range;

use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_common::config::Config;
use krishiv_io::io::coalesce;
use krishiv_io::split::{SplitConfig, SplitFile, plan};

fn files() -> Vec<SplitFile> {
    (0..10_000_u64)
        .map(|i| SplitFile {
            size: (i.wrapping_mul(2_654_435_761)) % (300 << 20) + 1,
            splittable: i % 3 != 0,
        })
        .collect()
}

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
    let files = files();
    let cfg = SplitConfig::from_config(&Config::new(), 16);
    let ranges = ranges();
    let mut group = c.benchmark_group("io");
    group.bench_function("split_plan_10k_files", |b| {
        b.iter(|| plan(black_box(&cfg), black_box(&files)));
    });
    group.bench_function("coalesce_10k_ranges", |b| {
        b.iter(|| coalesce(black_box(&ranges), 1 << 20, 16 << 20));
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
