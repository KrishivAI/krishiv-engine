// SPDX-License-Identifier: Apache-2.0
//! Per-row hash cost (P2-T175): the routing hash `H_kg` (XXH64, seed 0,
//! of the normalized key bytes, folded over several columns; plus the
//! table finalizer `g = fin(h ^ seed)`) against seeded foldhash, the
//! tables' hash before it, over 64 Ki rows of one batch.
//!
//! - `routing_hash/h_kg_<keys>`: `hash_columns` over the key columns.
//! - `routing_hash/h_kg_g_<keys>`: the same, then `finalize` per row.
//! - `routing_hash/foldhash_<keys>`: seeded foldhash of each row's values.
//!
//! `<keys>` is `i64`, `i64_i32` (two columns), `str10` (inline-view
//! strings) and `str24` (long strings).

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hash::{BuildHasher, Hash, Hasher};
use std::hint::black_box;

use arrow_array::{Array, Int32Array, Int64Array, StringArray};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use krishiv_exec::rows::finalize;
use krishiv_kernels::hash::routing::hash_columns;

const ROWS: usize = 1 << 16;
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// A bijective scramble so keys are not dense small integers.
fn scramble(x: u64) -> u64 {
    let x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn strings(len: usize) -> StringArray {
    StringArray::from_iter_values((0..ROWS as u64).map(|i| {
        let s = format!("{:016x}{:016x}", scramble(i), scramble(!i));
        s[..len].to_owned()
    }))
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("routing_hash");
    g.throughput(Throughput::Elements(ROWS as u64));
    let i64s = Int64Array::from_iter_values((0..ROWS as u64).map(|i| scramble(i).cast_signed()));
    let i32s = Int32Array::from_iter_values(
        (0..ROWS as u64).map(|i| i32::try_from(scramble(i) >> 33).expect("31 bits")),
    );
    let (s10, s24) = (strings(10), strings(24));
    let state = foldhash::fast::FixedState::with_seed(SEED);

    let cases: [(&str, Vec<&dyn Array>); 4] = [
        ("i64", vec![&i64s]),
        ("i64_i32", vec![&i64s, &i32s]),
        ("str10", vec![&s10]),
        ("str24", vec![&s24]),
    ];
    for (name, cols) in &cases {
        let mut out = Vec::with_capacity(ROWS);
        g.bench_function(format!("h_kg_{name}"), |b| {
            b.iter(|| {
                hash_columns(black_box(cols), &mut out).expect("hash");
                black_box(&out);
            });
        });
        g.bench_function(format!("h_kg_g_{name}"), |b| {
            b.iter(|| {
                hash_columns(black_box(cols), &mut out).expect("hash");
                for h in &mut out {
                    *h = finalize(*h, SEED);
                }
                black_box(&out);
            });
        });
    }

    let mut out = vec![0u64; ROWS];
    g.bench_function("foldhash_i64", |b| {
        b.iter(|| {
            for (h, v) in out.iter_mut().zip(black_box(&i64s).values()) {
                *h = state.hash_one(v);
            }
            black_box(&out);
        });
    });
    g.bench_function("foldhash_i64_i32", |b| {
        b.iter(|| {
            let (a, c) = (black_box(&i64s).values(), black_box(&i32s).values());
            for ((h, x), y) in out.iter_mut().zip(a).zip(c) {
                let mut s = state.build_hasher();
                x.hash(&mut s);
                y.hash(&mut s);
                *h = s.finish();
            }
            black_box(&out);
        });
    });
    for (name, a) in [("str10", &s10), ("str24", &s24)] {
        g.bench_function(format!("foldhash_{name}"), |b| {
            b.iter(|| {
                for (h, v) in out.iter_mut().zip(black_box(a).iter()) {
                    *h = state.hash_one(v.map(str::as_bytes).unwrap_or_default());
                }
                black_box(&out);
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
