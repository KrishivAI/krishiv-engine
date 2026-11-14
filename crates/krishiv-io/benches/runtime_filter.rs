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
use krishiv_io::runtime_filter::{RuntimeBloom, xxh64_fixed};
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

/// Rows ahead the prefetching probes request: one packed output word.
const AHEAD: usize = 64;

/// Asks for the cache line at `p` (a no-op off x86-64).
#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch is a hint and never dereferences `p`; any address is allowed.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p.cast::<i8>());
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

impl Bitmap {
    /// [`Self::evaluate`] with a prefetch of the word row `i + AHEAD` reads.
    fn evaluate_prefetch(&self, probe: &Int64Array) -> BooleanArray {
        let values = probe.values();
        let mut out = Vec::with_capacity(values.len().div_ceil(64));
        let base = self.words.as_ptr();
        let last = self.words.len().saturating_sub(1);
        for (c, chunk) in values.chunks(64).enumerate() {
            let mut w = 0u64;
            for (j, &v) in chunk.iter().enumerate() {
                if let Some(&a) = values.get(c * 64 + j + AHEAD) {
                    let o = a.wrapping_sub(self.min).cast_unsigned();
                    let i = usize::try_from(o >> 6).unwrap_or(usize::MAX).min(last);
                    prefetch(base.wrapping_add(i));
                }
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

const SALT: [u32; 8] = [
    0x47b6_137b,
    0x4497_4d91,
    0x8824_ad5b,
    0xa2b7_289d,
    0x7054_95c7,
    0x2df1_424b,
    0x9efc_4947,
    0x5c6b_fb31,
];

/// The library's split-block Bloom filter, copied out through `to_bytes`
/// so the probe loop can prefetch its blocks. Answers are asserted equal
/// to [`RuntimeBloom::check_batch`].
struct Sbbf {
    blocks: Vec<[u32; 8]>,
}

impl Sbbf {
    fn new(bloom: &RuntimeBloom) -> Self {
        let bytes = bloom.to_bytes();
        let blocks = bytes
            .as_chunks::<32>()
            .0
            .iter()
            .map(|b| {
                let mut w = [0u32; 8];
                for (x, c) in w.iter_mut().zip(b.as_chunks::<4>().0) {
                    *x = u32::from_le_bytes(*c);
                }
                w
            })
            .collect();
        Self { blocks }
    }

    #[inline]
    fn block(&self, h: u64) -> usize {
        usize::try_from(((h >> 32) * self.blocks.len() as u64) >> 32).expect("fits")
    }

    #[inline]
    fn check(&self, h: u64) -> bool {
        let b = &self.blocks[self.block(h)];
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the SBBF uses the low half"
        )]
        let key = h as u32;
        let mut miss = 0;
        for (w, salt) in b.iter().zip(SALT) {
            miss |= (1 << (key.wrapping_mul(salt) >> 27)) & !w;
        }
        miss == 0
    }

    fn check_batch(&self, hashes: &[u64], ahead: bool) -> BooleanBuffer {
        let mut out = Vec::with_capacity(hashes.len().div_ceil(64));
        let base = self.blocks.as_ptr();
        for (c, chunk) in hashes.chunks(64).enumerate() {
            let mut w = 0u64;
            for (j, &h) in chunk.iter().enumerate() {
                if ahead && let Some(&a) = hashes.get(c * 64 + j + AHEAD) {
                    prefetch(base.wrapping_add(self.block(a)));
                }
                w |= u64::from(self.check(h)) << j;
            }
            out.push(w);
        }
        BooleanBuffer::new(Buffer::from_vec(out), 0, hashes.len())
    }
}

/// Spans past L2 (2e7: 2.4 MiB of bitmap) and past this server's L3 per
/// core complex (2e8: 24 MiB), at 40 K, 400 K and 4 M keys, and 2^25 bits
/// (4 MiB) at 40 K keys (Blooms of
/// 48 KiB, 484 KiB and 4.7 MiB): the bitmap and
/// the Bloom's member pass, each with and without a prefetch `AHEAD` rows
/// early, plus the Bloom form end to end and the range-only pass.
fn bench_bitmap_wide(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_filter_bitmap_wide");
    // Plus 2^25 bits at 40 K keys: the candidate default cap (research-1,
    // gap ledger fifth revision; 2^24 would take customer's keys, not part's).
    let points = [40_960usize, 409_600, 4_096_000]
        .into_iter()
        .flat_map(|n| [20_000_000i64, 200_000_000].map(|r| (n, r)))
        .chain([(40_960, 33_554_432)]);
    for (n, range) in points {
        {
            let (keys, probe) = in_range(n, range);
            let options = KeyFilterOptions {
                max_values: 0,
                exact_max_values: 0,
                bloom: true,
                bloom_max_items: u64::MAX,
                bloom_max_bytes: 64 << 20,
                ..KeyFilterOptions::default()
            };
            let mut b = KeyFilterBuilder::new(keys.data_type().clone(), options).expect("key type");
            b.add(&keys, None).expect("keys");
            let bloom_form = b.finish();
            assert!(matches!(bloom_form.form(), FilterForm::Bloom(_)));
            let range_only = build_exact(&keys, 0, 0, false);
            let bitmap = Bitmap::new(&keys);
            let want = bitmap.evaluate(&probe);
            assert_eq!(bitmap.evaluate_prefetch(&probe), want);
            // The member pass alone over precomputed hashes, as the form's
            // probe runs after its hash pass.
            let hash = |v: i64| xxh64_fixed::<8>(u128::from(v.cast_unsigned()));
            let mut rb = RuntimeBloom::with_bytes(RuntimeBloom::bytes_for(n as u64, 0.01));
            for &k in keys.values() {
                rb.insert_hash(hash(k));
            }
            let sbbf = Sbbf::new(&rb);
            let hashes: Vec<u64> = probe.values().iter().map(|&v| hash(v)).collect();
            let lib = rb.check_batch(&hashes);
            assert_eq!(sbbf.check_batch(&hashes, false), lib);
            assert_eq!(sbbf.check_batch(&hashes, true), lib);
            assert_eq!(bloom_form.bloom_bytes(), Some(rb.num_bytes()));
            let tag = format!("{n}keys_{range}");
            group.bench_function(format!("probe_bitmap_{tag}"), |b| {
                b.iter(|| black_box(bitmap.evaluate(black_box(&probe))));
            });
            group.bench_function(format!("probe_bitmap_pf_{tag}"), |b| {
                b.iter(|| black_box(bitmap.evaluate_prefetch(black_box(&probe))));
            });
            group.bench_function(format!("check_bloom_{tag}"), |b| {
                b.iter(|| black_box(sbbf.check_batch(black_box(&hashes), false)));
            });
            group.bench_function(format!("check_bloom_pf_{tag}"), |b| {
                b.iter(|| black_box(sbbf.check_batch(black_box(&hashes), true)));
            });
            group.bench_function(format!("probe_bloom_{tag}"), |b| {
                b.iter(|| black_box(bloom_form.evaluate(black_box(&probe))));
            });
            group.bench_function(format!("probe_rangeonly_{tag}"), |b| {
                b.iter(|| black_box(range_only.evaluate(black_box(&probe))));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench, bench_bitmap, bench_bitmap_wide);
criterion_main!(benches);
