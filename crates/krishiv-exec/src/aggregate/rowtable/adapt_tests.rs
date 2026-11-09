// SPDX-License-Identifier: Apache-2.0
//! Tests of a fixed index's adaptation: append, growth, rebuilds and
//! skipped lookups.

use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::Int64Array;

use super::tests::{apply, batch, core, core_str, long_keys, mix};
use super::*;

/// After `ADAPT_ROWS` rows, unique keys switch a fixed index to
/// append-only; keys that repeat beyond its capacity grow it to their
/// count as soon as it has filled with enough partial duplicates, long
/// before `ADAPT_ROWS`.
#[test]
fn fixed_index_adapts() {
    let core = core();
    let seed = HashSeed::new(Some(2));
    let mut unique =
        RowTable::new(&core, &seed, Mode::Fixed { limit: 1024 }, true, RADIX_BITS).unwrap();
    let mut repeat =
        RowTable::new(&core, &seed, Mode::Fixed { limit: 1024 }, true, RADIX_BITS).unwrap();
    let step = 8192;
    let mut grown_at = None;
    for start in (0..ADAPT_ROWS + step).step_by(step) {
        let s = i64::try_from(start).unwrap();
        apply(&mut unique, &batch(&core, s..s + 8192));
        apply(
            &mut repeat,
            &batch(&core, (s..s + 8192).map(|i| i % 10_000)),
        );
        if grown_at.is_none() && repeat.mode != (Mode::Fixed { limit: 1024 }) {
            grown_at = Some(start);
        }
    }
    assert_eq!(unique.mode, Mode::Append);
    assert_eq!(unique.adapt.as_ref().unwrap().skip, None);
    // 10,000 keys at load 2/3: 2^14 slots, in one step from 2^11.
    assert_eq!(
        repeat.mode,
        Mode::Fixed {
            limit: (1 << 14) * 2 / 3
        }
    );
    assert!(grown_at.is_some_and(|r| r < 64 * 1024), "{grown_at:?}");
    // Once grown, the 10,000 keys stay in the index: no more rows.
    let groups = repeat.groups;
    apply(&mut repeat, &batch(&core, (0..8192).map(|i| i % 10_000)));
    assert_eq!(repeat.groups, groups);
}

/// A full fixed index grows (DuckDB 2.0's rule) only once its partial
/// duplicates take `GROW_DUPLICATE_FACTOR` times the grown index's
/// bytes, at least 2x and at most to `MAX_SINK_SLOTS`; and it decides
/// only after it filled.
#[test]
fn fixed_index_grows_by_its_duplicates() {
    let core = core();
    let seed = HashSeed::new(Some(2));
    // A full index of `slots` with about 512 distinct keys and `dups`
    // rows beyond them.
    let table = |slots: usize, dups: usize| {
        let mut t = RowTable::new(
            &core,
            &seed,
            Mode::Fixed {
                limit: slots * 2 / 3,
            },
            true,
            RADIX_BITS,
        )
        .unwrap();
        let a = t.adapt.as_mut().unwrap();
        for h in 0..64 {
            a.sketch.add(mix(h));
        }
        a.filled = true;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a small positive estimate"
        )]
        let distinct = RowTable::distinct(a) as usize;
        t.groups = distinct + dups;
        t
    };
    // From 2^11 slots: the distinct keys fit, so 2x, 2^12 slots.
    let index = buckets_for((1 << 12) * 2 / 3) * BUCKET_BYTES;
    let stride = table(1 << 11, 0).stride;
    let enough = (GROW_DUPLICATE_FACTOR * index).div_ceil(stride);
    assert_eq!(
        table(1 << 11, enough - 1).next_mode(),
        Mode::Fixed {
            limit: (1 << 11) * 2 / 3
        }
    );
    let grown = Mode::Fixed {
        limit: (1 << 12) * 2 / 3,
    };
    assert_eq!(table(1 << 11, enough).next_mode(), grown);
    // Not before it filled.
    let mut t = table(1 << 11, enough);
    t.adapt.as_mut().unwrap().filled = false;
    assert_eq!(t.mode, t.next_mode());
    // The switch to it restarts the early-widening window.
    let mut t = table(1 << 11, enough);
    let a = t.adapt.as_mut().unwrap();
    (a.rows, a.pushed, a.early) = (5_000, 4_000, true);
    t.switch(t.next_mode());
    assert_eq!(t.mode, grown);
    let a = t.adapt.as_ref().unwrap();
    assert!(!a.filled && !a.early);
    assert_eq!(a.window, (5_000, 4_000));
    // At most MAX_SINK_SLOTS: from half of it, 2x lands on it; from
    // there it cannot double.
    let huge = 1 << 30;
    let full = Mode::Fixed {
        limit: MAX_SINK_SLOTS * 2 / 3,
    };
    assert_eq!(table(MAX_SINK_SLOTS / 2, huge).next_mode(), full);
    assert_eq!(table(MAX_SINK_SLOTS, huge).next_mode(), full);
    // Many distinct keys: straight to their capacity when they fit.
    let many = |hashes: u64| {
        let mut t = table(1 << 11, huge);
        let a = t.adapt.as_mut().unwrap();
        for h in 64..hashes {
            a.sketch.add(mix(h));
        }
        t.next_mode()
    };
    let Mode::Fixed { limit } = many(20_000) else {
        panic!("{:?}", many(20_000));
    };
    assert!(limit >= 100_000, "{limit}");
    // Keys that would not fit in MAX_SINK_SLOTS: no growth at all.
    assert_eq!(
        many(60_000),
        Mode::Fixed {
            limit: (1 << 11) * 2 / 3
        }
    );
}

/// A grown index is rebuilt over the rows, newest first in a
/// one-partition table, up to its limit: keys an earlier clear dropped
/// find their row again, for fixed-width and long string keys.
#[test]
fn a_grown_index_finds_rows_from_before_its_clears() {
    let seed = HashSeed::new(Some(4));
    let int = core();
    let string = core_str();
    type Keys = fn(&AggCore, std::ops::Range<i64>) -> Batch;
    let cases: [(&Arc<AggCore>, Keys); 2] = [
        (&int, |c, r| batch(c, r)),
        (&string, |c, r| long_keys(c, r)),
    ];
    for (core, keys) in cases {
        let table = || {
            // One partition, as a thread-local table starts: rows in
            // arrival order.
            let mut t = RowTable::new(core, &seed, Mode::Fixed { limit: 8 }, true, 0).unwrap();
            apply(&mut t, &keys(core, 0..100));
            assert_eq!((t.groups, t.unique), (100, false));
            assert!(t.index.len() <= 8);
            t
        };
        // Room for every key: none makes another row.
        let mut t = table();
        t.switch(Mode::Fixed { limit: 1_000 });
        assert_eq!(t.index.len(), 100);
        apply(&mut t, &keys(core, 0..100));
        assert_eq!(t.groups, 100);
        // Room for 50: the newest 50 keys are found, the oldest are not.
        let mut t = table();
        t.switch(Mode::Fixed { limit: 50 });
        assert_eq!(t.index.len(), 50);
        apply(&mut t, &keys(core, 50..100));
        assert_eq!(t.groups, 100);
        apply(&mut t, &keys(core, 0..50));
        assert_eq!(t.groups, 150);
        // The same output as one row per key.
        let all = t.ids(0..t.groups);
        let out = t.output_batch(&all).unwrap();
        let counts = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(counts.iter().flatten().sum::<i64>(), 200);
    }
}

/// A full fixed index grows through `apply` itself (sketch, clears and
/// all), within the bytes `projected` reserved, for fixed-width and long
/// string keys; the growth counts as a resize, and afterwards a repeated
/// key makes no new row.
#[test]
fn a_full_index_grows_inside_its_projection() {
    let seed = HashSeed::new(Some(6));
    let int = core();
    let string = core_str();
    type Keys = fn(&AggCore, std::ops::Range<i64>) -> Batch;
    let cases: [(&Arc<AggCore>, Keys); 2] = [
        (&int, |c, r| batch(c, r)),
        (&string, |c, r| long_keys(c, r)),
    ];
    for (core, keys) in cases {
        let resizes = Arc::new(AtomicU64::new(0));
        let mut t = RowTable::new(core, &seed, Mode::Fixed { limit: 8 }, true, RADIX_BITS)
            .unwrap()
            .with_resizes(Arc::clone(&resizes));
        // 100 keys, over and over: every pass past the first is all
        // partial duplicates until the index grows.
        for _ in 0..20 {
            apply(&mut t, &keys(core, 0..100));
        }
        let Mode::Fixed { limit } = t.mode else {
            panic!("{:?}", t.mode);
        };
        assert!(limit >= 100, "{limit}");
        assert!(resizes.load(Ordering::Relaxed) > 0);
        let groups = t.groups;
        apply(&mut t, &keys(core, 0..100));
        assert_eq!(t.groups, groups);
        let out = t.output_batch(&t.ids(0..t.groups)).unwrap();
        let counts = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(counts.iter().flatten().sum::<i64>(), 2_100);
    }
}

/// A rebuild over several partitions stops at its limit and keeps one
/// entry per key; rows it leaves out only make partial duplicates.
#[test]
fn a_rebuild_over_partitions_stops_at_its_limit() {
    let seed = HashSeed::new(Some(7));
    let core = core();
    let mut t = RowTable::new(&core, &seed, Mode::Fixed { limit: 8 }, true, RADIX_BITS).unwrap();
    apply(&mut t, &batch(&core, 0..100));
    assert!(!t.unique);
    assert!(t.part_count() > 1);
    t.switch(Mode::Fixed { limit: 50 });
    assert_eq!(t.index.len(), 50);
    // The indexed keys are 50 distinct keys, and each finds its row.
    let indexed: Vec<u32> = t.index.iter().copied().collect();
    let out = t.output_batch(&indexed).unwrap();
    let mut keys: Vec<i64> = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .iter()
        .flatten()
        .collect();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(keys.len(), 50);
    apply(&mut t, &batch(&core, keys.iter().copied()));
    assert_eq!(t.groups, 100);
    let out = t.output_batch(&t.ids(0..t.groups)).unwrap();
    let counts = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(counts.iter().flatten().sum::<i64>(), 150);
}

/// Keys that repeat only after far more rows than the index holds, and
/// too many to grow to (TPC-H q13's pre-aggregate): after an interval in
/// which nearly every row made a new row, the table skips its lookups for
/// the next interval, then looks up again over a rebuilt index, and skips
/// again while the lookups still miss; no row is lost.
#[test]
fn lookups_that_seldom_hit_are_skipped_for_an_interval() {
    let core = core();
    let seed = HashSeed::new(Some(3));
    let fixed = Mode::Fixed { limit: 1024 };
    let mut t = RowTable::new(&core, &seed, fixed, true, RADIX_BITS).unwrap();
    let fed = std::cell::Cell::new(0_i64);
    let feed = |t: &mut RowTable, rows: usize, key: fn(i64) -> i64| {
        for _ in (0..rows).step_by(8192) {
            let s = fed.get();
            apply(t, &batch(&core, (s..s + 8192).map(key)));
            fed.set(s + 8192);
        }
    };
    feed(&mut t, ADAPT_ROWS + 8192, |i| i % 700_000);
    assert_eq!(t.mode, Mode::Append);
    assert_eq!(t.adapt.as_ref().unwrap().skip, NonZeroU32::new(1024));
    // Not the input's verdict: the keys are not 95% distinct.
    assert!(!RowTable::all_unique(t.adapt.as_ref().unwrap()));
    // Another interval: lookups again, over the newest rows.
    feed(&mut t, ADAPT_ROWS, |i| i % 700_000);
    assert_eq!(t.mode, fixed);
    let a = t.adapt.as_ref().unwrap();
    assert_eq!(a.skip, None);
    assert_eq!(a.next, a.window.0 + ADAPT_ROWS);
    // The lookups work again: the newest keys find their rows.
    let (groups, last) = (t.groups, fed.get());
    apply(
        &mut t,
        &batch(&core, (last - 512..last).map(|i| i % 700_000)),
    );
    assert_eq!(t.groups, groups);
    fed.set(last + 512);
    // A resumed interval that still misses skips again (q13's steady state).
    feed(&mut t, ADAPT_ROWS, |i| i % 700_000);
    assert_eq!(t.mode, Mode::Append);
    assert_eq!(t.adapt.as_ref().unwrap().skip, NonZeroU32::new(1024));
    feed(&mut t, ADAPT_ROWS, |i| i % 700_000);
    assert_eq!(t.mode, fixed);
    // Keys that now repeat within a fill: the lookups pay, no new skip.
    feed(&mut t, ADAPT_ROWS + 8192, |i| i / 2);
    assert_eq!(t.mode, fixed);
    let out = t.output_batch(&t.ids(0..t.groups)).unwrap();
    let counts = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let rows: i64 = counts.iter().flatten().sum();
    assert_eq!(rows, fed.get());
}

/// Each key on a few adjacent rows (TPC-H q18's lines of an order): the
/// lookups find three rows in four, so the table keeps its fixed index.
#[test]
fn lookups_that_hit_are_not_skipped() {
    let core = core();
    let seed = HashSeed::new(Some(5));
    let fixed = Mode::Fixed { limit: 1024 };
    let mut t = RowTable::new(&core, &seed, fixed, true, RADIX_BITS).unwrap();
    for start in (0..2 * ADAPT_ROWS + 8192).step_by(8192) {
        let s = i64::try_from(start).unwrap();
        apply(&mut t, &batch(&core, (s..s + 8192).map(|i| i / 4)));
    }
    assert_eq!(t.mode, fixed);
    assert_eq!(t.adapt.as_ref().unwrap().skip, None);
}
