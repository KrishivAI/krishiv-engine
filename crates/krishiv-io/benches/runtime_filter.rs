// SPDX-License-Identifier: Apache-2.0
//! Runtime filter kernels (design P2-T163), run under CodSpeed in CI:
//! - building a filter from 262,144 distinct `bigint` keys (min/max, the
//!   value set overflowing at once, XXH64 of every key, an exactly sized
//!   split-block Bloom filter);
//! - probing 262,144 `bigint` rows against a min/max range, a 50-key value
//!   set and that Bloom filter, and `int` rows widened to the `bigint` key;
//! - probing 262,144 strings against a Bloom filter of 65,536 keys.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;

use arrow_array::{Array, Int32Array, Int64Array, StringViewArray};
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_io::{KeyFilter, KeyFilterBuilder, KeyFilterOptions};

const ROWS: i64 = 1 << 18;

fn build(keys: &dyn Array, max_values: usize, bloom: bool) -> KeyFilter {
    let options = KeyFilterOptions {
        max_values,
        bloom,
        ..KeyFilterOptions::default()
    };
    let mut b = KeyFilterBuilder::new(keys.data_type().clone(), options).expect("key type");
    b.add(keys, None).expect("keys");
    b.finish()
}

/// Keys spread over a wide domain (a multiplicative hash of the index).
fn spread(n: i64) -> Int64Array {
    Int64Array::from_iter_values((0..n).map(|i| i.wrapping_mul(0x9E37_79B9) & 0xFFFF_FFFF))
}

fn bench(c: &mut Criterion) {
    let build_keys = spread(ROWS);
    let probe = Int64Array::from_iter_values((0..ROWS).map(|i| i.wrapping_mul(7919) & 0xFFFF_FFFF));
    let narrow = Int32Array::from_iter_values((0..ROWS).map(|i| i32::try_from(i).expect("small")));
    let bloom = build(&build_keys, 0, true);
    assert!(bloom.bloom_bytes().is_some());
    let range = build(&Int64Array::from(vec![1 << 20, 1 << 30]), 0, false);
    let values = build(&spread(50), 50, false);
    assert_eq!(values.value_count(), Some(50));
    let strings: Vec<String> = (0..ROWS).map(|i| format!("customer#{i:09}")).collect();
    let probe_strings = StringViewArray::from_iter_values(strings.iter());
    let string_bloom = build(
        &StringViewArray::from_iter_values(strings.iter().step_by(4)),
        0,
        true,
    );

    let mut group = c.benchmark_group("runtime_filter");
    group.bench_function("build_bloom_256k_i64", |b| {
        b.iter(|| black_box(build(black_box(&build_keys), 50, true)));
    });
    group.bench_function("probe_range_256k_i64", |b| {
        b.iter(|| black_box(range.evaluate(black_box(&probe))));
    });
    group.bench_function("probe_values50_256k_i64", |b| {
        b.iter(|| black_box(values.evaluate(black_box(&probe))));
    });
    group.bench_function("probe_bloom_256k_i64", |b| {
        b.iter(|| black_box(bloom.evaluate(black_box(&probe))));
    });
    group.bench_function("probe_bloom_256k_i32_widened", |b| {
        b.iter(|| black_box(bloom.evaluate(black_box(&narrow))));
    });
    group.bench_function("probe_bloom_256k_strings", |b| {
        b.iter(|| black_box(string_bloom.evaluate(black_box(&probe_strings))));
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
