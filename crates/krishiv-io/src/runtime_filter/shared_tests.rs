// SPDX-License-Identifier: Apache-2.0
//! Builders over one shared Bloom array (design P2b-R1): every key any
//! partial stored passes the finished filter, whatever the partials'
//! refusals and wherever the array's allocation falls; a lost or unfunded
//! array gives the range, never a filter that misses keys.
#![cfg(not(loom))]

use std::sync::Arc;

use arrow_array::{Array, Int64Array};
use arrow_schema::DataType;
use proptest::prelude::*;

use super::*;

/// Options of a large build: a planned size above what the `Vec` path
/// collects for, exact sets off so the Bloom form is chosen.
fn large(bytes: u32) -> KeyFilterOptions {
    KeyFilterOptions {
        max_values: 0,
        exact_max_values: 0,
        bloom_max_items: 100,
        bloom_bytes: Some(bytes),
        ..KeyFilterOptions::default()
    }
}

fn shared(bloom: &Arc<SharedBloom>, options: KeyFilterOptions) -> KeyFilterBuilder {
    KeyFilterBuilder::with_shared_bloom(DataType::Int64, options, Arc::clone(bloom))
        .unwrap_or_else(|| panic!("int64 keys"))
}

fn add(b: &mut KeyFilterBuilder, keys: &[i64]) {
    b.add(&Int64Array::from(keys.to_vec()), None)
        .unwrap_or_else(|e| panic!("{e}"));
}

fn passes(f: &KeyFilter, keys: &[i64]) -> bool {
    let col = Int64Array::from(keys.to_vec());
    let out = f.evaluate(&col).unwrap_or_else(|| panic!("int64"));
    out.null_count() == 0 && out.true_count() == keys.len()
}

#[test]
fn shared_bloom_bytes_only_above_the_collected_size() {
    let o = large(1 << 20);
    assert_eq!(o.shared_bloom_bytes(), Some(1 << 20));
    let capped = KeyFilterOptions {
        bloom_max_bytes: 4096,
        ..o
    };
    assert_eq!(capped.shared_bloom_bytes(), Some(4096));
    // A partitioned build's size (at most what `bloom_max_items` keys
    // take) or none at all: the hashes are collected.
    let small = RuntimeBloom::bytes_for(o.bloom_max_items, o.bloom_fpp);
    let partitioned = large(u32::try_from(small).unwrap_or(u32::MAX));
    assert_eq!(partitioned.shared_bloom_bytes(), None);
    assert_eq!(KeyFilterOptions::default().shared_bloom_bytes(), None);
    let off = KeyFilterOptions { bloom: false, ..o };
    assert_eq!(off.shared_bloom_bytes(), None);
}

#[test]
fn partials_merge_into_one_filter_with_every_key() {
    let options = large(64 << 10);
    let bloom = Arc::new(SharedBloom::with_bytes(64 << 10));
    let mut a = shared(&bloom, options);
    let mut b = shared(&bloom, options);
    // Keys before the allocation wait in the partial, then go in.
    add(&mut a, &(0..500).collect::<Vec<_>>());
    assert!(
        a.heap_bytes() >= 500 * 8,
        "pending hashes are the partial's"
    );
    assert!(bloom.allocate());
    add(&mut b, &(500..1000).collect::<Vec<_>>());
    assert!(
        b.heap_bytes() < 1000,
        "inserted keys cost the partial nothing"
    );
    let mut global = shared(&bloom, options);
    global.merge(a);
    global.merge(b);
    drop(bloom);
    assert!(global.has_shared_bloom());
    let (f, outcome) = global.finish_reporting();
    assert_eq!(outcome, Some(SharedBloomOutcome::Published));
    assert!(matches!(f.form(), FilterForm::Bloom(_)));
    assert_eq!(f.bloom_bytes(), Some(64 << 10));
    assert!(passes(&f, &(0..1000).collect::<Vec<_>>()));
    let out: Vec<i64> = (1_000_000..1_010_000).collect();
    let col = Int64Array::from(out);
    let fp = f.evaluate(&col).map_or(0, |b| b.true_count());
    assert!(fp < 200, "{fp} false positives in 10000");
}

#[test]
fn an_outstanding_holder_or_no_allocation_gives_the_range() {
    let options = large(4096);
    let bloom = Arc::new(SharedBloom::with_bytes(4096));
    assert!(bloom.allocate());
    let mut a = shared(&bloom, options);
    add(&mut a, &[1, 2, 3]);
    // `bloom` is still held: no copy of the array is made.
    let (f, outcome) = a.finish_reporting();
    assert_eq!(outcome, Some(SharedBloomOutcome::StillShared));
    assert_eq!(f.form(), FilterForm::Range);
    assert!(passes(&f, &[1, 2, 3]));

    let abandoned = Arc::new(SharedBloom::with_bytes(4096));
    let mut b = shared(&abandoned, options);
    add(&mut b, &[5]);
    assert!(abandoned.abandon());
    add(&mut b, &[6]);
    assert_eq!(b.heap_bytes(), 0, "an abandoned array's pending hashes go");
    drop(abandoned);
    let (f, outcome) = b.finish_reporting();
    assert_eq!(outcome, Some(SharedBloomOutcome::Unfunded));
    assert!(passes(&f, &[5, 6]));
}

#[test]
fn an_undersized_array_gives_the_range() {
    let options = large(32);
    let bloom = Arc::new(SharedBloom::with_bytes(32));
    assert!(bloom.allocate());
    let mut a = shared(&bloom, options);
    add(&mut a, &(0..10_000).collect::<Vec<_>>());
    drop(bloom);
    let (f, outcome) = a.finish_reporting();
    assert_eq!(outcome, Some(SharedBloomOutcome::Undersized));
    assert_eq!(f.form(), FilterForm::Range);
}

#[test]
fn a_value_form_reports_nothing() {
    let options = KeyFilterOptions {
        bloom_max_items: 100,
        bloom_bytes: Some(1 << 20),
        ..KeyFilterOptions::default()
    };
    let bloom = Arc::new(SharedBloom::with_bytes(1 << 20));
    let mut a = shared(&bloom, options);
    add(&mut a, &[7, 9]);
    drop(bloom);
    let (f, outcome) = a.finish_reporting();
    assert_eq!(outcome, None);
    assert!(!matches!(f.form(), FilterForm::Bloom(_)));
}

#[test]
fn a_partial_without_the_array_loses_the_bloom_form() {
    let options = large(4096);
    let bloom = Arc::new(SharedBloom::with_bytes(4096));
    assert!(bloom.allocate());
    let mut a = shared(&bloom, options);
    add(&mut a, &[1]);
    let mut plain =
        KeyFilterBuilder::new(DataType::Int64, options).unwrap_or_else(|| panic!("int64"));
    add(&mut plain, &[2]);
    a.merge(plain);
    drop(bloom);
    let (f, _) = a.finish_reporting();
    assert_eq!(f.form(), FilterForm::Range);
    assert!(passes(&f, &[1, 2]));
}

/// A refusal once the array is allocated flushes the pending hashes into
/// it: the partial keeps its Bloom form (review finding 3).
#[test]
fn a_refusal_after_allocation_keeps_the_bloom_form() {
    let options = large(64 << 10);
    let bloom = Arc::new(SharedBloom::with_bytes(64 << 10));
    let mut a = shared(&bloom, options);
    add(&mut a, &(0..300).collect::<Vec<_>>());
    assert!(bloom.allocate());
    a.drop_hashes();
    assert_eq!(a.heap_bytes(), 0, "flushed, not held");
    let mut b = shared(&bloom, options);
    add(&mut b, &[1_000]);
    b.merge(a);
    drop(bloom);
    let (f, outcome) = b.finish_reporting();
    assert_eq!(outcome, Some(SharedBloomOutcome::Published));
    assert!(passes(&f, &(0..300).chain([1_000]).collect::<Vec<_>>()));
}

/// Drivers on threads, one of them allocating the array partway: every
/// key passes the merged filter.
#[test]
fn threads_and_a_late_allocation_miss_no_key() {
    let options = large(256 << 10);
    let bloom = Arc::new(SharedBloom::with_bytes(256 << 10));
    let partials: Vec<KeyFilterBuilder> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4i64)
            .map(|t| {
                let bloom = Arc::clone(&bloom);
                s.spawn(move || {
                    let mut p = shared(&bloom, options);
                    for batch in 0..50i64 {
                        if t == 0 && batch == 20 {
                            assert!(bloom.allocate());
                        }
                        let keys: Vec<i64> = (0..100).map(|i| (t * 50 + batch) * 100 + i).collect();
                        add(&mut p, &keys);
                    }
                    p
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| panic!("a driver panicked")))
            .collect()
    });
    let mut global = shared(&bloom, options);
    for p in partials {
        global.merge(p);
    }
    drop(bloom);
    let (f, outcome) = global.finish_reporting();
    assert_eq!(outcome, Some(SharedBloomOutcome::Published));
    assert!(passes(&f, &(0..20_000).collect::<Vec<_>>()));
}

#[test]
fn a_leftover_holder_is_reported_as_such() {
    let options = large(4096);
    let bloom = Arc::new(SharedBloom::with_bytes(4096));
    assert!(bloom.allocate());
    let mut a = shared(&bloom, options);
    add(&mut a, &[1, 2]);
    let (f, outcome) = a.finish_reporting();
    assert_eq!(outcome, Some(SharedBloomOutcome::StillShared));
    assert!(passes(&f, &[1, 2]));
}

/// Options of a shared array's size never build a Bloom of that size from
/// collected hashes, nor when partials of several producers merge (design
/// P2b-R1 M5).
#[test]
fn collected_partials_never_build_a_shared_size() {
    let options = KeyFilterOptions {
        bloom_max_bytes: 512 << 20,
        ..large(64 << 20)
    };
    let mut a = KeyFilterBuilder::new(DataType::Int64, options).unwrap_or_else(|| panic!("int64"));
    add(&mut a, &(0..50).collect::<Vec<_>>());
    let f = a.finish();
    assert_eq!(f.form(), FilterForm::Range);
    // Two exact sets whose union is past the exact cap: not a planned Bloom.
    let exact = KeyFilterOptions {
        exact_max_values: 40,
        ..options
    };
    let set = |keys: std::ops::Range<i64>| {
        let mut b =
            KeyFilterBuilder::new(DataType::Int64, exact).unwrap_or_else(|| panic!("int64"));
        add(&mut b, &keys.map(|k| k * 7).collect::<Vec<_>>());
        b.finish()
    };
    let merged = set(0..30)
        .merge_into(&set(30..60), &exact)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(merged.bloom_bytes(), None);
    assert!(passes(&merged, &(0..60).map(|k| k * 7).collect::<Vec<_>>()));
}

/// One partial's script: batches of keys, refusals after some of them.
#[derive(Debug, Clone)]
struct Script {
    batches: Vec<(Vec<i64>, u8)>,
}

fn script() -> impl Strategy<Value = Script> {
    prop::collection::vec(
        (prop::collection::vec(-5_000i64..5_000, 1..40), 0u8..6),
        1..8,
    )
    .prop_map(|batches| Script { batches })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// B1: every key any partial stored passes the published filter, under
    /// any pattern of refusals (`drop_hashes`, `drop_bloom`) and any point
    /// of allocation (or none: abandoned).
    #[test]
    fn no_stored_key_is_missed(
        scripts in prop::collection::vec(script(), 1..5),
        allocate_after in 0usize..12,
        abandon in any::<bool>(),
    ) {
        let options = large(16 << 10);
        let bloom = Arc::new(SharedBloom::with_bytes(16 << 10));
        let mut partials: Vec<KeyFilterBuilder> =
            scripts.iter().map(|_| shared(&bloom, options)).collect();
        let mut stored = Vec::new();
        let mut step = 0usize;
        let longest = scripts.iter().map(|s| s.batches.len()).max().unwrap_or(0);
        for round in 0..longest {
            for (p, s) in partials.iter_mut().zip(&scripts) {
                let Some((keys, act)) = s.batches.get(round) else { continue };
                if step == allocate_after {
                    if abandon { bloom.abandon(); } else { bloom.allocate(); }
                }
                step += 1;
                add(p, keys);
                stored.extend_from_slice(keys);
                match act {
                    0 => p.drop_hashes(),
                    1 => p.drop_bloom(),
                    _ => {}
                }
            }
        }
        let mut global = shared(&bloom, options);
        for p in partials {
            global.merge(p);
        }
        drop(bloom);
        let f = global.finish();
        prop_assert!(passes(&f, &stored));
    }
}
