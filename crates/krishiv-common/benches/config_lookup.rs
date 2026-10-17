// SPDX-License-Identifier: Apache-2.0
//! Micro-benchmark for config registry lookups, run under CodSpeed in CI.

#![allow(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_common::config::lookup;

fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("config");
    group.bench_function("lookup_registered", |b| {
        b.iter(|| lookup(black_box("krishiv.sql.ansi.enabled")));
    });
    group.bench_function("lookup_missing", |b| {
        b.iter(|| lookup(black_box("krishiv.sql.does.not.exist")));
    });
    group.finish();
}

criterion_group!(benches, bench_lookup);
criterion_main!(benches);
