// SPDX-License-Identifier: Apache-2.0
//! Tests of the row table: modes, the salted merge index, widening, partitions and byte accounting.

use arrow_array::{Int64Array, StringArray};
use arrow_schema::{DataType, Field, Schema};

use super::*;
use crate::aggregate::{AggregateExpr, AggregateFunction, AggregateSpec};

fn core() -> Arc<AggCore> {
    let s = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
    ]));
    let spec = AggregateSpec {
        input: Arc::clone(&s),
        group_by: vec![0],
        aggregates: vec![
            AggregateExpr::count_star("c"),
            AggregateExpr::new(AggregateFunction::Sum, 1, DataType::Int64, "s"),
        ],
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
        legacy_statistical: false,
        udafs: None,
    };
    Arc::new(AggCore::new(spec).unwrap())
}

fn batch(core: &AggCore, keys: impl Iterator<Item = i64>) -> Batch {
    let k: Vec<i64> = keys.collect();
    let v = Int64Array::from(vec![1; k.len()]);
    let cols: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(k)), Arc::new(v)];
    Batch::new(RecordBatch::try_new(Arc::clone(&core.spec.input), cols).unwrap())
}

/// Applies after reserving: the projection bounds the bytes after.
fn apply(t: &mut RowTable, b: &Batch) {
    let p = t.prepare(b, false).unwrap();
    let want = t.projected(&p);
    t.apply(&p).unwrap();
    assert!(t.memory_size() <= want, "{} > {want}", t.memory_size());
}

/// A table grouping by a `Utf8` key (`v` an `Int64`).
fn core_str() -> Arc<AggCore> {
    let s = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Utf8, true),
        Field::new("v", DataType::Int64, true),
    ]));
    // As `core`'s, over a string key and with `count(*)` only.
    let spec = AggregateSpec {
        input: Arc::clone(&s),
        aggregates: vec![AggregateExpr::count_star("c")],
        ..core().spec.clone()
    };
    Arc::new(AggCore::new(spec).unwrap())
}

/// Keys of 200 bytes (heap values) for [`core_str`].
fn long_keys(core: &AggCore, keys: impl Iterator<Item = i64>) -> Batch {
    let k: Vec<String> = keys.map(|i| format!("{i:0>200}")).collect();
    let v = Int64Array::from(vec![1; k.len()]);
    let k = arrow_array::StringArray::from(k);
    let cols: Vec<ArrayRef> = vec![Arc::new(k), Arc::new(v)];
    Batch::new(RecordBatch::try_new(Arc::clone(&core.spec.input), cols).unwrap())
}

/// Long string keys live in the partitions' heaps: the table counts
/// them, and the projections of an apply and of a merge bound them.
#[test]
fn heap_bytes_are_accounted() {
    let core = core_str();
    let seed = HashSeed::new(Some(9));
    for mode in [Mode::Grow, Mode::Fixed { limit: 64 }, Mode::Append] {
        let mut t = RowTable::new(&core, &seed, mode, true, RADIX_BITS).unwrap();
        for start in (0..2_000).step_by(500) {
            apply(&mut t, &long_keys(&core, start..start + 500));
        }
        // Every key is new: 2,000 values of 200 bytes in the heaps.
        assert!(
            t.memory_size() >= 2_000 * 200,
            "{mode:?}: {}",
            t.memory_size()
        );
        let mut merged = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
        for p in 0..PARTS {
            let c = t.merge_chunk(p, 0..t.part_len(p)).unwrap();
            let want = merged.projected_merge(&c);
            merged.apply_merge(&t, &c).unwrap();
            assert!(
                merged.memory_size() <= want,
                "{} > {want}",
                merged.memory_size()
            );
        }
        assert!(merged.memory_size() >= 2_000 * 200);
        let out = merged.output_batch(&merged.ids(0..merged.len())).unwrap();
        assert_eq!(out.num_rows(), 2_000);
    }
}

/// Merges `inputs` chunk by chunk into a new table presized for `presize`
/// groups, checking the projections hold; returns the table.
fn merged(core: &Arc<AggCore>, seed: &HashSeed, inputs: &[RowTable], presize: usize) -> RowTable {
    let mut out = RowTable::new(core, seed, Mode::Grow, false, RADIX_BITS).unwrap();
    let want = out.projected_presize(presize);
    out.presize(presize).unwrap();
    assert!(out.memory_size() <= want, "{} > {want}", out.memory_size());
    for t in inputs {
        for p in 0..PARTS {
            for start in (0..t.part_len(p)).step_by(1024) {
                let end = (start + 1024).min(t.part_len(p));
                let c = t.merge_chunk(p, start..end).unwrap();
                let want = out.projected_merge(&c);
                out.apply_merge(t, &c).unwrap();
                assert!(out.memory_size() <= want, "{} > {want}", out.memory_size());
            }
        }
    }
    out
}

/// A merge through the salted index finds the same groups, in the same
/// order, with the same states, as one through `hashbrown`; a low
/// estimate only makes it grow.
#[test]
fn a_large_merge_table_takes_the_salted_index() {
    let core = core();
    let seed = HashSeed::new(Some(5));
    let input = |keys: Range<i64>| {
        let mut t = RowTable::new(&core, &seed, Mode::Append, true, RADIX_BITS).unwrap();
        for s in keys.clone().step_by(4096) {
            apply(&mut t, &batch(&core, s..(s + 4096).min(keys.end)));
        }
        t
    };
    // Overlapping: keys of the second are hits for 10,000 of its rows.
    let inputs = [input(0..30_000), input(20_000..40_000)];
    let flat = merged(&core, &seed, &inputs, 1_000);
    assert!(flat.merge_index.is_none());
    let large = merged(&core, &seed, &inputs, 40_000);
    assert!(large.merge_index.is_some());
    // 8,000 keys presize 2^14 slots: too few, so this one grows.
    let grown = merged(&core, &seed, &inputs, 8_000);
    assert!(grown.buckets() > 1 << 14);
    let ids = flat.ids(0..flat.len());
    let want = flat.output_batch(&ids).unwrap();
    assert_eq!(want.num_rows(), 40_000);
    for t in [&large, &grown] {
        assert_eq!(t.len(), 40_000);
        assert_eq!(t.output_batch(&t.ids(0..t.len())).unwrap(), want);
    }
}

/// A detailed profile counts the salted merge's lookups and key compares.
#[test]
fn a_salted_merge_counts_its_probes() {
    let core = core();
    let seed = HashSeed::new(Some(5));
    let mut t = RowTable::new(&core, &seed, Mode::Append, true, RADIX_BITS).unwrap();
    apply(&mut t, &batch(&core, 0..4_000));
    let totals = Arc::new(ProbeTotals::default());
    let mut out = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS)
        .unwrap()
        .with_probes(Some(Arc::clone(&totals)));
    out.presize(10_000).unwrap();
    // Twice: 4,000 new keys, then 4,000 hits with a compare each.
    for _ in 0..2 {
        for p in 0..PARTS {
            let c = t.merge_chunk(p, 0..t.part_len(p)).unwrap();
            out.apply_merge(&t, &c).unwrap();
        }
    }
    assert_eq!(out.len(), 4_000);
    let count = totals.get();
    assert_eq!(count.lookups, 8_000);
    assert!(count.compares >= 4_000, "{}", count.compares);
}

/// Different keys with one forged hash meet in one slot's walk, with
/// equal salts: each is compared, and each stays its own group.
#[test]
fn a_salted_merge_tells_apart_keys_of_one_salt() {
    const H: u64 = 0x1234;
    let core = core();
    let seed = HashSeed::new(Some(5));
    let mut real = RowTable::new(&core, &seed, Mode::Append, true, RADIX_BITS).unwrap();
    apply(&mut real, &batch(&core, 0..50));
    let mut src = RowTable::new(&core, &seed, Mode::Append, false, RADIX_BITS).unwrap();
    let refs: Vec<u32> = real
        .ids(0..50)
        .iter()
        .map(|&r| {
            let mut row = row_at(&real.parts, real.stride, real.refs, r).to_vec();
            row[..8].copy_from_slice(&H.to_le_bytes());
            src.push(&row, &[], H).unwrap()
        })
        .collect();
    let c = MergeChunk {
        part: 0,
        refs,
        states: Vec::new(),
        heap: 0,
    };
    let mut out = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
    out.presize(10_000).unwrap();
    let (mut first, mut again) = (Vec::new(), Vec::new());
    let mut count = ProbeCount::default();
    out.merge_rows_salted(&src, &c, &mut first, &mut count)
        .unwrap();
    out.merge_rows_salted(&src, &c, &mut again, &mut count)
        .unwrap();
    assert_eq!(out.len(), 50);
    assert_eq!(first, again);
    let mut distinct = first.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), 50);
}

/// A table with a merge index refuses raw input, and splitting a table
/// drops the index.
#[test]
fn a_merge_table_only_takes_merges() {
    let core = core();
    let seed = HashSeed::new(Some(5));
    let mut out = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
    out.presize(10_000).unwrap();
    let held = out.memory_size();
    let b = batch(&core, 0..10);
    let prepared = out.prepare(&b, false).unwrap();
    assert!(out.apply(&prepared).is_err());
    out.widen_to = RADIX_BITS;
    out.bits = 0;
    assert!(out.widen().is_err());
    out.bits = RADIX_BITS;
    out.split_parts(&[]);
    assert!(out.merge_index.is_none() && out.memory_size() < held);
}

/// Long string keys compare through the layout and their bytes are
/// counted, as with `hashbrown`.
#[test]
fn a_salted_merge_compares_long_string_keys() {
    let core = core_str();
    let seed = HashSeed::new(Some(9));
    let input = |keys: Range<i64>| {
        let mut t = RowTable::new(&core, &seed, Mode::Append, true, RADIX_BITS).unwrap();
        apply(&mut t, &long_keys(&core, keys));
        t
    };
    let inputs = [input(0..1_500), input(1_000..2_000)];
    let flat = merged(&core, &seed, &inputs, 100);
    let salted = merged(&core, &seed, &inputs, 10_000);
    assert!(salted.merge_index.is_some());
    assert_eq!(salted.len(), 2_000);
    assert!(salted.memory_size() >= 2_000 * 200);
    assert_eq!(
        salted.output_batch(&salted.ids(0..salted.len())).unwrap(),
        flat.output_batch(&flat.ids(0..flat.len())).unwrap()
    );
}

/// A seedless 64-bit mix (splitmix64's finalizer) for test hashes.
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[test]
fn hll_estimates_and_merges() {
    for n in [0u64, 10, 1_000, 100_000] {
        let mut h = Hll::new(PART_HLL_BITS);
        for i in 0..n {
            h.add(mix(i));
            h.add(mix(i));
        }
        #[expect(clippy::cast_precision_loss, reason = "test counts")]
        let n = n as f64;
        let e = h.estimate();
        assert!((e - n).abs() <= n * 0.1 + 1.0, "{e} for {n}");
    }
    let (mut a, mut b) = (Hll::new(PART_HLL_BITS), Hll::new(PART_HLL_BITS));
    (0..50_000).for_each(|i| a.add(mix(i)));
    (25_000..75_000).for_each(|i| b.add(mix(i)));
    a.merge(&b);
    assert!(
        (a.estimate() - 75_000.0).abs() < 7_500.0,
        "{}",
        a.estimate()
    );
}

/// A fixed index is cleared when full: a key seen again gets a second
/// row, so the table is no longer unique; growing and append-only
/// tables behave as their modes say.
#[test]
fn modes_and_partial_duplicates() {
    let core = core();
    let seed = HashSeed::new(Some(1));
    let mut grow = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
    let mut fixed =
        RowTable::new(&core, &seed, Mode::Fixed { limit: 8 }, true, RADIX_BITS).unwrap();
    let mut append = RowTable::new(&core, &seed, Mode::Append, true, RADIX_BITS).unwrap();
    for t in [&mut grow, &mut fixed, &mut append] {
        apply(t, &batch(&core, 0..20));
        apply(t, &batch(&core, (0..20).map(|i| i % 4)));
    }
    assert_eq!((grow.len(), grow.is_unique()), (20, true));
    // Cleared at 8 keys: 16..19 are left, 0..3 come again.
    assert_eq!((fixed.len(), fixed.is_unique()), (24, false));
    assert_eq!((append.len(), append.is_unique()), (40, false));
    // Every row is in its partition of `h`, and positions cover them.
    let all = append.ids(0..append.len());
    assert_eq!(all.len(), 40);
    // Spilling from a position skips exactly the rows before it.
    for from in [0, 1, 17, 39, 40, 41] {
        let mut got: Vec<u32> = append.ids_by_partition(0, from).concat();
        let mut want = append.ids(from..append.len());
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "from {from}");
    }
    let per: usize = (0..PARTS).map(|p| append.part_len(p)).sum();
    assert_eq!(per, 40);
    let out = grow.output_batch(&grow.ids(0..grow.len())).unwrap();
    assert_eq!(out.num_rows(), 20);
    // A merge table takes another table's rows and merges duplicates.
    let mut merged = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
    // Different register counts do not merge.
    let mut one = Hll::new(PART_HLL_BITS);
    one.add(1);
    one.merge(&Hll::new(ADAPT_HLL_BITS));
    assert_eq!(one.used(), 1);
    for p in 0..PARTS {
        let c = append.merge_chunk(p, 0..append.part_len(p)).unwrap();
        let want = merged.projected_merge(&c);
        merged.apply_merge(&append, &c).unwrap();
        assert!(merged.memory_size() <= want);
    }
    assert_eq!((merged.len(), merged.is_unique()), (20, true));
    let out = merged.output_batch(&merged.ids(0..20)).unwrap();
    let keys = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let counts = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    for (k, c) in keys.iter().zip(counts) {
        assert_eq!(c, Some(if k < Some(4) { 6 } else { 1 }), "key {k:?}");
    }
    // Only growing tables merge.
    let c = append.merge_chunk(0, 0..0).unwrap();
    let e = fixed.apply_merge(&append, &c).unwrap_err();
    assert!(e.to_string().contains("not growing"), "{e}");
}

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

/// Widening a fixed index that was cleared rebuilds it over the rows,
/// up to its limit and one entry per key: the indexed keys make no new
/// rows afterwards, and the index is no larger than before.
#[test]
fn widening_a_cleared_fixed_index_rebuilds_it() {
    let seed = HashSeed::new(Some(7));
    let core = core();
    let mut t = RowTable::new(&core, &seed, Mode::Fixed { limit: 64 }, true, 0)
        .unwrap()
        .widening(RADIX_BITS, 100);
    apply(&mut t, &batch(&core, (0..300).map(|i| i % 200)));
    assert!(!t.unique && t.wants_widen());
    let (groups, buckets) = (t.groups, t.buckets());
    t.widen().unwrap();
    assert_eq!(t.part_count(), PARTS);
    assert_eq!(t.index.len(), 64);
    assert!(t.buckets() <= buckets);
    let indexed: Vec<u32> = t.index.iter().copied().collect();
    let out = t.output_batch(&indexed).unwrap();
    let keys: Vec<i64> = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .iter()
        .flatten()
        .collect();
    apply(&mut t, &batch(&core, keys.iter().copied()));
    assert_eq!(t.groups, groups);
}

/// Deterministic, one-thread and two-thread defaults.
#[test]
fn local_modes() {
    assert_eq!(Mode::local(true, 12), Mode::Grow);
    assert_eq!(Mode::local(false, 1), Mode::Grow);
    assert_eq!(
        Mode::local(false, 2),
        Mode::Fixed {
            limit: (1 << 18) * 2 / 3
        }
    );
    assert_eq!(
        Mode::local(false, 3),
        Mode::Fixed {
            limit: (1 << 15) * 2 / 3
        }
    );
}

/// The final phase merges small results in one task and splits large
/// ones over every partition; empty partitions get no task.
#[test]
fn final_tasks_follow_the_estimate() {
    let core = core();
    let seed = HashSeed::new(Some(3));
    let mut small =
        RowTable::new(&core, &seed, Mode::Fixed { limit: 64 }, true, RADIX_BITS).unwrap();
    apply(&mut small, &batch(&core, 0..1_000));
    let tasks = final_tasks(&[&small]);
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].parts, 0..PARTS);
    assert!(
        tasks[0].groups >= 1_000 && tasks[0].groups <= 1_200,
        "{:?}",
        tasks[0]
    );
    let mut big = RowTable::new(&core, &seed, Mode::Append, true, RADIX_BITS).unwrap();
    for s in (0..600_000).step_by(8192) {
        apply(&mut big, &batch(&core, s..(s + 8192).min(600_000)));
    }
    let tasks = final_tasks(&[&big, &small]);
    assert_eq!(tasks.len(), PARTS);
    assert!(tasks.iter().all(|t| t.parts.len() == 1));
    assert!(final_tasks(&[]).is_empty());
}

/// The sorted (key, count, sum) rows of a final output batch.
fn sorted(b: &RecordBatch) -> Vec<(Option<i64>, Option<i64>, Option<i64>)> {
    let col = |i: usize| b.column(i).as_any().downcast_ref::<Int64Array>().unwrap();
    let mut v: Vec<_> = (0..b.num_rows())
        .map(|i| {
            let get = |c: &Int64Array| c.is_valid(i).then(|| c.value(i));
            (get(col(0)), get(col(1)), get(col(2)))
        })
        .collect();
    v.sort_unstable();
    v
}

/// A fixed index cleared inside a prefetched chunk of rows: the rows of
/// the chunk after the clear still find (or make) their own key's row,
/// so each key's partial rows add up to its input rows.
#[test]
fn a_clear_inside_a_prefetched_chunk_keeps_every_count() {
    let core = core();
    let seed = HashSeed::new(Some(6));
    // 100 keys of index, 150 keys cycling: a clear every 100 new keys,
    // at varying positions within each 64-row chunk.
    let mut t = RowTable::new(&core, &seed, Mode::Fixed { limit: 100 }, false, 0).unwrap();
    // The per-row probe, not the batch-local key map (which a batch of 150 keys would take).
    t.local_skip = 1;
    apply(&mut t, &batch(&core, (0..5_000).map(|i| i % 150)));
    assert!(!t.is_unique());
    let mut sums = [0_i64; 150];
    for (k, c, s) in sorted(&t.output_batch(&t.ids(0..t.len())).unwrap()) {
        let k = usize::try_from(k.unwrap()).unwrap();
        assert_eq!(c, s);
        sums[k] += c.unwrap();
    }
    let want: Vec<i64> = (0..150).map(|k| (5_000 - k + 149) / 150).collect();
    assert_eq!(sums.to_vec(), want);
}

/// One partition (one thread) and sixteen fold the same groups; output
/// across partitions concatenates each partition's states in the order
/// of the references.
#[test]
fn one_partition_equals_sixteen() {
    let core = core();
    let seed = HashSeed::new(Some(4));
    let mut one = RowTable::new(&core, &seed, Mode::Grow, false, 0).unwrap();
    let mut many = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
    assert_eq!((one.part_count(), many.part_count()), (1, PARTS));
    for t in [&mut one, &mut many] {
        apply(t, &batch(&core, (0..3_000).map(|i| i % 700)));
        apply(t, &batch(&core, 0..10));
    }
    let all = |t: &RowTable| t.output_batch(&t.ids(0..t.len())).unwrap();
    assert_eq!(sorted(&all(&one)), sorted(&all(&many)));
    assert_eq!(one.len(), 700);
    // References out of partition order still line up with their keys.
    let mut refs = many.ids(0..many.len());
    refs.reverse();
    let b = many.output_batch(&refs).unwrap();
    assert_eq!(sorted(&b), sorted(&all(&one)));
    let states = many.state_batch(&refs[..5]).unwrap();
    assert_eq!(states.num_rows(), 5);
    // No rows: empty columns of the right types.
    assert_eq!(many.output_batch(&[]).unwrap().num_rows(), 0);
    // A reference past a partition's rows is an internal error.
    let bad = many.refs.make(0, many.part_len(0) + 1);
    assert!(many.output_batch(&[bad]).is_err());
    assert!(many.merge_chunk(0, 0..many.part_len(0) + 1).is_err());
    // A one-partition table merges every partition of a wider one
    // (the final phase's merge of inputs of mixed widths)...
    let mut grow = RowTable::new(&core, &seed, Mode::Grow, false, 0).unwrap();
    for p in 0..PARTS {
        let c = many.merge_chunk(p, 0..many.part_len(p)).unwrap();
        assert!(grow.projected_merge(&c) >= grow.memory_size());
        grow.apply_merge(&many, &c).unwrap();
    }
    assert_eq!(sorted(&all(&grow)), sorted(&all(&one)));
    // ... but a partitioned one needs the same radix bits.
    let c = many.merge_chunk(0, 0..1).unwrap();
    let mut two = RowTable::new(&core, &seed, Mode::Grow, false, 1).unwrap();
    let e = two.apply_merge(&many, &c).unwrap_err();
    assert!(e.to_string().contains("radix bits"), "{e}");
}

/// A thread-local table starts its partition sketches at
/// `SKETCH_ROWS` rows, from the rows it already has: smaller tables
/// hold none (their row counts are the final phase's estimate).
#[test]
fn sketches_start_with_enough_rows() {
    let core = core();
    let seed = HashSeed::new(Some(6));
    let mut t = RowTable::new(&core, &seed, Mode::Grow, true, RADIX_BITS).unwrap();
    apply(&mut t, &batch(&core, 0..1_000));
    assert!(t.sketches().is_none());
    let before = t.memory_size();
    let n = i64::try_from(SKETCH_ROWS).unwrap();
    apply(&mut t, &batch(&core, 1_000..n));
    assert!(t.memory_size() > before);
    let s = t.sketches().expect("sketches from SKETCH_ROWS rows");
    let mut all = Hll::new(PART_HLL_BITS);
    s.iter().for_each(|h| all.merge(h));
    #[expect(clippy::cast_precision_loss, reason = "test counts")]
    let rows = SKETCH_ROWS as f64;
    assert!(
        (all.estimate() - rows).abs() < rows * 0.1,
        "{}",
        all.estimate()
    );
    // One partition, or a merge table: never.
    let mut one = RowTable::new(&core, &seed, Mode::Grow, true, 0).unwrap();
    apply(&mut one, &batch(&core, 0..n));
    assert!(one.sketches().is_none());
    // A small table among sketched ones counts its rows.
    let mut small = RowTable::new(&core, &seed, Mode::Grow, true, RADIX_BITS).unwrap();
    apply(&mut small, &batch(&core, n..n + 100));
    let groups: usize = final_tasks(&[&t, &small]).iter().map(|f| f.groups).sum();
    assert!(
        (SKETCH_ROWS + 100..=(SKETCH_ROWS + 100) * 12 / 10).contains(&groups),
        "{groups}"
    );
}

/// Widening spreads a narrow table's rows, long key bytes and states
/// over the radix partitions without changing any group: a growing
/// or fixed index still finds every key, and sketches start.
#[test]
fn widening_keeps_every_group() {
    let core = core();
    let seed = HashSeed::new(Some(8));
    for mode in [Mode::Grow, Mode::Fixed { limit: 5_000 }, Mode::Append] {
        let mut t = RowTable::new(&core, &seed, mode, true, 0)
            .unwrap()
            .widening(RADIX_BITS, 4_000);
        let mut narrow = RowTable::new(&core, &seed, mode, true, 0).unwrap();
        for tb in [&mut t, &mut narrow] {
            apply(tb, &batch(&core, (0..6_000).map(|i| i % 4_500)));
        }
        assert!(t.wants_widen() && !narrow.wants_widen());
        let all = |t: &RowTable| sorted(&t.output_batch(&t.ids(0..t.len())).unwrap());
        let want = all(&t);
        let bytes = t.memory_size();
        // The reservation covers the copy, the state arrays and the
        // row lists the widening holds while it runs.
        let accs: usize = t
            .parts
            .iter()
            .flat_map(|p| &p.accs)
            .map(Accumulator::size)
            .sum();
        let lists = 2 * t.len() * size_of::<u32>();
        assert!(t.widen_bytes() >= bytes + 2 * accs + lists);
        t.widen().unwrap();
        assert_eq!((t.part_count(), t.wants_widen()), (PARTS, false));
        assert_eq!(all(&t), want, "{mode:?}");
        // The copy grows each partition once, exactly: no more bytes
        // than before but for the sketches it starts.
        let sketches = PARTS * Hll::new(PART_HLL_BITS).bytes();
        assert!(t.memory_size() <= bytes + sketches, "{mode:?}");
        assert!(t.sketches().is_some());
        let groups = t.len();
        apply(&mut t, &batch(&core, 0..4_500));
        match mode {
            // Every key is found again: no new rows, still unique.
            Mode::Grow | Mode::Fixed { .. } => {
                assert_eq!((t.len(), t.is_unique()), (groups, true), "{mode:?}");
            }
            Mode::Append => assert!(t.len() > groups && !t.is_unique()),
        }
        // Widening an already wide table changes nothing.
        let n = t.len();
        t.widen().unwrap();
        assert_eq!(t.len(), n);
    }
}

/// A widening that fails part-way leaves the table as it was: every
/// group with its states, its width and an index that still finds
/// every key; a later widening succeeds.
#[test]
fn failed_widening_keeps_the_table() {
    let core = core();
    let seed = HashSeed::new(Some(8));
    for (bits, copied) in [(0, 0), (2, 1)] {
        let mut t = RowTable::new(&core, &seed, Mode::Grow, true, bits)
            .unwrap()
            .widening(RADIX_BITS, 64);
        apply(&mut t, &batch(&core, (0..3_000).map(|i| i % 2_500)));
        assert!(t.wants_widen());
        let all = |t: &RowTable| sorted(&t.output_batch(&t.ids(0..t.len())).unwrap());
        let (want, groups, parts) = (all(&t), t.len(), t.part_count());
        WIDEN_FAULT.set(Some(copied));
        let err = t.widen().unwrap_err();
        WIDEN_FAULT.set(None);
        assert!(err.to_string().contains("injected widening fault"), "{err}");
        assert_eq!((t.len(), t.part_count()), (groups, parts), "bits {bits}");
        assert_eq!(all(&t), want, "bits {bits}");
        // The index still finds every key: no new groups.
        apply(&mut t, &batch(&core, 0..2_500));
        assert_eq!((t.len(), t.is_unique()), (groups, true), "bits {bits}");
        t.widen().unwrap();
        assert_eq!(t.part_count(), PARTS);
        let mut twice = want.clone();
        for row in &mut twice {
            row.1 = row.1.map(|c| c + 1);
            row.2 = row.2.map(|v| v + 1);
        }
        assert_eq!(all(&t), twice, "bits {bits}");
    }
}

/// Row references split into partition and row by the table's radix
/// bits: one partition keeps all 32 bits for rows (2^32 groups).
#[test]
fn refs_split_by_radix_bits() {
    let one = Refs::new(0);
    assert_eq!(
        (one.mask(), one.part(u32::MAX), one.row(u32::MAX)),
        (u32::MAX, 0, u32::MAX)
    );
    assert_eq!(one.make(0, 5), 5);
    let many = Refs::new(RADIX_BITS);
    assert_eq!(many.mask(), (1 << 28) - 1);
    let r = many.make(PARTS - 1, 7);
    assert_eq!((many.part(r), many.row(r)), (PARTS - 1, 7));
    // Never more than RADIX_BITS.
    assert_eq!(Refs::new(9), many);
}

/// A partition with few rows can hold most of a variable-width
/// argument column's bytes, so the gather bound charges the full
/// column for types without a primitive width.
#[test]
fn gather_bytes_covers_skewed_var_width() {
    let s = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Utf8, true),
    ]));
    let spec = AggregateSpec {
        input: Arc::clone(&s),
        group_by: vec![0],
        aggregates: vec![AggregateExpr::new(
            AggregateFunction::Min,
            1,
            DataType::Utf8,
            "m",
        )],
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
        legacy_statistical: false,
        udafs: None,
    };
    let core = Arc::new(AggCore::new(spec).unwrap());
    let seed = HashSeed::new(Some(6));
    let t = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
    // Exact-capacity values: a grown `Vec` can leave slack in the
    // values buffer, which would hide the skew.
    let mut values = Vec::with_capacity(4103);
    values.extend(std::iter::repeat_n(b'x', 4096));
    values.extend(std::iter::repeat_n(b'y', 7));
    let mut offsets = Vec::with_capacity(9);
    offsets.push(0i32);
    offsets.push(4096);
    for i in 1..=7 {
        offsets.push(4096 + i);
    }
    let strings: ArrayRef = Arc::new(StringArray::new(
        arrow_buffer::OffsetBuffer::new(offsets.into()),
        arrow_buffer::Buffer::from_vec(values),
        None,
    ));
    let keys: ArrayRef = Arc::new(Int64Array::from(vec![0; 8]));
    let cols = vec![keys, Arc::clone(&strings)];
    let mut counts = vec![0; PARTS];
    counts[0] = 1;
    counts[1] = 7;
    let got = t.gather_bytes(&cols, Fold::Raw, &Weights::Const(1), &counts);
    let take = |idx: &[u32]| {
        arrow_select::take::take(strings.as_ref(), &UInt32Array::from(idx.to_vec()), None)
            .unwrap()
            .get_array_memory_size()
    };
    let small = take(&[0]);
    let large = take(&[1, 2, 3, 4, 5, 6, 7]);
    assert!(got >= small, "{got} < {small}");
    assert!(got >= large, "{got} < {large}");
}

/// Partitions split out of a table carry their rows and states; the
/// rest stays; a released partition frees its memory.
#[test]
fn split_and_release_parts() {
    let core = core();
    let seed = HashSeed::new(Some(5));
    let mut t = RowTable::new(&core, &seed, Mode::Fixed { limit: 100 }, true, RADIX_BITS).unwrap();
    apply(&mut t, &batch(&core, (0..5_000).map(|i| i % 1_000)));
    let total = t.len();
    let want = sorted(&t.output_batch(&t.ids(0..total)).unwrap());
    let mut slices = t.split_parts(&[0..4, 4..PARTS]);
    assert_eq!(t.len(), 0);
    assert_eq!(t.buckets(), 0);
    assert_eq!(slices.iter().map(RowTable::len).sum::<usize>(), total);
    let mut got = Vec::new();
    for s in &slices {
        got.extend(sorted(&s.output_batch(&s.ids(0..s.len())).unwrap()));
    }
    got.sort_unstable();
    assert_eq!(got, want);
    let s = &mut slices[1];
    let before = s.memory_size();
    let n4 = s.part_len(4);
    s.release_part(4);
    assert_eq!(s.part_len(4), 0);
    assert!(s.memory_size() < before || n4 == 0);
    for p in 5..PARTS {
        s.release_part(p);
    }
    assert_eq!((s.len(), s.memory_size()), (0, 0));
}
