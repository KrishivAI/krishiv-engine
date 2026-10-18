// SPDX-License-Identifier: Apache-2.0
//! Config key lookup, run under CodSpeed in CI.
//!
//! The search is measured on a fixed synthetic index of 256 keys rather than
//! on the live registry, whose size grows whenever a config key is added: a
//! benchmark of the live registry would report every new key as a
//! regression and say nothing about the algorithm. A slower algorithm (for
//! example a linear scan) shows up here at once. One real lookup keeps the
//! production entry point covered.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_common::config::{lookup, search};

/// 256 dotted keys shaped like real ones, ordered by (length, name).
fn synthetic_index() -> Vec<(String, u32)> {
    let groups = [
        "sql", "exec", "memory", "spill", "network", "shuffle", "io", "ui",
    ];
    let mut keys: Vec<(String, u32)> = (0..256u32)
        .map(|i| {
            let g = groups[(i as usize) % groups.len()];
            (format!("krishiv.{g}.setting{i}Name"), i)
        })
        .collect();
    keys.sort_by(|a, b| a.0.len().cmp(&b.0.len()).then_with(|| a.0.cmp(&b.0)));
    keys
}

fn bench_lookup(c: &mut Criterion) {
    let owned = synthetic_index();
    let index: Vec<(&str, u32)> = owned.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let hit = owned[173].0.clone();
    let mut group = c.benchmark_group("config");
    group.bench_function("search_hit_256", |b| {
        b.iter(|| search(black_box(&index), black_box(&hit)));
    });
    group.bench_function("search_miss_256", |b| {
        b.iter(|| search(black_box(&index), black_box("krishiv.sql.does.not.exist")));
    });
    group.bench_function("lookup_registry", |b| {
        b.iter(|| lookup(black_box("krishiv.spill.compression")));
    });
    group.finish();
}

criterion_group!(benches, bench_lookup);
criterion_main!(benches);
