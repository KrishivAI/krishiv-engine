// SPDX-License-Identifier: Apache-2.0
//! Runtime filter kernels (design P2-T163), run under CodSpeed in CI:
//! - building a filter from 262,144 distinct `bigint` keys (min/max, the
//!   value set overflowing at once, XXH64 of every key, an exactly sized
//!   split-block Bloom filter);
//! - probing 262,144 `bigint` rows (almost all misses, as a selective join
//!   filter sees them) against each row form: a min/max range, a small set
//!   (8 keys), and an exact hash set and a Bloom filter of 1,024, 4,096
//!   and 65,536 keys (the exact-set cutoff is chosen from these);
//! - `int` rows widened to the `bigint` key, and 262,144 strings against a
//!   Bloom filter of 65,536 keys;
//! - a bitmap form (one bit per value of the build's key range), against
//!   the exact hash set and the Bloom filter of the same keys: 4,096,
//!   40,960 and 409,600 distinct keys over ranges of 1e5, 1e6 and 1e7,
//!   262,144 probe rows inside the range. The bitmap is not a filter form
//!   yet; its arm is the bit probe alone, and `probe_rangeonly_*` (the
//!   key normalization and range pass every form pays) is measured beside
//!   it, so a bitmap form would cost about the sum of the two.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;

use arrow_array::{Array, BooleanArray, Int32Array, Int64Array, StringViewArray};
use arrow_buffer::{BooleanBuffer, Buffer};
use criterion::{Criterion, criterion_group, criterion_main};
use krishiv_io::{FilterForm, KeyFilter, KeyFilterBuilder, KeyFilterOptions};

const ROWS: i64 = 1 << 18;

fn build(keys: &dyn Array, max_values: usize, bloom: bool) -> KeyFilter {
    build_exact(keys, max_values, max_values, bloom)
}

fn build_exact(keys: &dyn Array, max_values: usize, exact: usize, bloom: bool) -> KeyFilter {
    let options = KeyFilterOptions {
        max_values,
        exact_max_values: exact,
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
    let small = build_exact(&spread(8), 0, 8, false);
    assert_eq!(small.form(), FilterForm::SmallSet(8));
    let sized: Vec<(i64, KeyFilter, KeyFilter)> = [1i64 << 10, 1 << 12, 1 << 16]
        .into_iter()
        .map(|n| {
            let keys = spread(n);
            let exact = build_exact(&keys, 0, usize::MAX, false);
            let bloom = build_exact(&keys, 0, 0, true);
            assert!(matches!(exact.form(), FilterForm::HashSet(_)));
            assert!(matches!(bloom.form(), FilterForm::Bloom(_)));
            (n, exact, bloom)
        })
        .collect();
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
    group.bench_function("probe_small8_256k_i64", |b| {
        b.iter(|| black_box(small.evaluate(black_box(&probe))));
    });
    for (n, exact, bloom) in &sized {
        group.bench_function(format!("probe_set{n}_256k_i64"), |b| {
            b.iter(|| black_box(exact.evaluate(black_box(&probe))));
        });
        group.bench_function(format!("probe_bloom{n}_256k_i64"), |b| {
            b.iter(|| black_box(bloom.evaluate(black_box(&probe))));
        });
    }
    group.bench_function("probe_bloom_256k_i32_widened", |b| {
        b.iter(|| black_box(bloom.evaluate(black_box(&narrow))));
    });
    group.bench_function("probe_bloom_256k_strings", |b| {
        b.iter(|| black_box(string_bloom.evaluate(black_box(&probe_strings))));
    });
    group.finish();
}

/// `n` distinct keys in `[0, range)` and `ROWS` probe values in the same
/// range, from a fixed linear congruential sequence.
fn in_range(n: usize, range: i64) -> (Int64Array, Int64Array) {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        i64::try_from((state >> 33) % u64::try_from(range).expect("positive")).expect("small")
    };
    let mut seen = std::collections::HashSet::with_capacity(n);
    let mut keys = Vec::with_capacity(n);
    while keys.len() < n {
        let k = next();
        if seen.insert(k) {
            keys.push(k);
        }
    }
    let probe = (0..ROWS).map(|_| next()).collect::<Vec<_>>();
    (Int64Array::from(keys), Int64Array::from(probe))
}

/// A bitmap of `keys` over their own `[min, max]`: one bit per value.
struct Bitmap {
    min: i64,
    span: u64,
    words: Vec<u64>,
}

impl Bitmap {
    fn new(keys: &Int64Array) -> Self {
        let values = keys.values();
        let min = values.iter().copied().min().expect("keys");
        let max = values.iter().copied().max().expect("keys");
        let span = max.abs_diff(min) + 1;
        let mut words = vec![0u64; usize::try_from(span.div_ceil(64)).expect("fits")];
        for &v in values {
            let o = v.abs_diff(min);
            words[usize::try_from(o >> 6).expect("fits")] |= 1 << (o & 63);
        }
        Self { min, span, words }
    }

    /// The rows of `probe` whose value is a key, packed 64 answers a word
    /// (the same packing as the hashed forms' probe loop).
    fn evaluate(&self, probe: &Int64Array) -> BooleanArray {
        let values = probe.values();
        let mut out = Vec::with_capacity(values.len().div_ceil(64));
        for chunk in values.chunks(64) {
            let mut w = 0u64;
            for (j, &v) in chunk.iter().enumerate() {
                let o = v.wrapping_sub(self.min).cast_unsigned();
                let hit = o < self.span && {
                    let i = usize::try_from(o >> 6).unwrap_or(usize::MAX);
                    self.words
                        .get(i)
                        .is_some_and(|word| word >> (o & 63) & 1 == 1)
                };
                w |= u64::from(hit) << j;
            }
            out.push(w);
        }
        BooleanArray::new(
            BooleanBuffer::new(Buffer::from_vec(out), 0, values.len()),
            None,
        )
    }
}

fn bench_bitmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_filter_bitmap");
    for n in [4096usize, 40_960, 409_600] {
        for range in [100_000i64, 1_000_000, 10_000_000] {
            if i64::try_from(n).expect("small") * 2 > range {
                continue;
            }
            let (keys, probe) = in_range(n, range);
            let exact = build_exact(&keys, 0, usize::MAX, false);
            let bloom = build_exact(&keys, 0, 0, true);
            let range_only = build_exact(&keys, 0, 0, false);
            assert!(matches!(exact.form(), FilterForm::HashSet(_)));
            assert!(matches!(bloom.form(), FilterForm::Bloom(_)));
            assert_eq!(range_only.form(), FilterForm::Range);
            let bitmap = Bitmap::new(&keys);
            let want = exact.evaluate(&probe).expect("bigint");
            assert_eq!(
                bitmap.evaluate(&probe),
                want,
                "bitmap answers as the exact set"
            );
            let tag = format!("{n}keys_{range}");
            group.bench_function(format!("probe_set_{tag}"), |b| {
                b.iter(|| black_box(exact.evaluate(black_box(&probe))));
            });
            group.bench_function(format!("probe_bitmap_{tag}"), |b| {
                b.iter(|| black_box(bitmap.evaluate(black_box(&probe))));
            });
            group.bench_function(format!("probe_bloom_{tag}"), |b| {
                b.iter(|| black_box(bloom.evaluate(black_box(&probe))));
            });
            group.bench_function(format!("probe_rangeonly_{tag}"), |b| {
                b.iter(|| black_box(range_only.evaluate(black_box(&probe))));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench, bench_bitmap);
criterion_main!(benches);
