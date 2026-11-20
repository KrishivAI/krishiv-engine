// SPDX-License-Identifier: Apache-2.0
//! `vector_*` kernels at 768 dimensions (the P3-T076 acceptance: >= 1 GB/s/core of float32 for the inner
//! product), 4096 rows per iteration, one thread.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use krishiv_kernels::vector;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DIM: usize = 768;
const ROWS: usize = 4096;

/// Deterministic values in [-0.5, 0.5).
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[expect(clippy::cast_precision_loss, reason = "bench data")]
            let u = (s >> 8) as f32 / (1_u32 << 24) as f32;
            u - 0.5
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let (a, b) = (values(DIM * ROWS, 1), values(DIM * ROWS, 2));
    let bytes = u64::try_from(DIM * ROWS * 4).unwrap_or(u64::MAX);
    let mut g = c.benchmark_group("vector_768");
    g.throughput(Throughput::Bytes(bytes));
    g.bench_function("inner_product", |bch| {
        bch.iter(|| {
            let mut t = 0.0_f32;
            for (x, y) in a.as_chunks::<DIM>().0.iter().zip(b.as_chunks::<DIM>().0) {
                t += vector::inner_product(black_box(x), black_box(y));
            }
            t
        });
    });
    g.bench_function("l2_distance", |bch| {
        bch.iter(|| {
            let mut t = 0.0_f32;
            for (x, y) in a.as_chunks::<DIM>().0.iter().zip(b.as_chunks::<DIM>().0) {
                t += vector::l2_distance(black_box(x), black_box(y));
            }
            t
        });
    });
    g.bench_function("cosine_similarity", |bch| {
        bch.iter(|| {
            let mut t = 0.0_f32;
            for (x, y) in a.as_chunks::<DIM>().0.iter().zip(b.as_chunks::<DIM>().0) {
                t += vector::cosine_similarity(black_box(x), black_box(y)).unwrap_or(0.0);
            }
            t
        });
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
