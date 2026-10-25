// SPDX-License-Identifier: Apache-2.0
//! The routing hash `H_kg` (P2-T175): golden vectors, agreement of the
//! unrolled widths with XXH64, carrier independence, and key-group
//! routing.

#![expect(clippy::panic, reason = "test helpers outside #[test] fns fail loudly")]

use std::sync::Arc;

use arrow_array::types::{Int8Type, Int16Type, Int32Type};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Decimal64Array,
    Decimal128Array, DictionaryArray, Float32Array, Float64Array, Int8Array, Int32Array,
    Int64Array, LargeBinaryArray, LargeStringArray, NullArray, StringArray, StringViewArray,
};
use krishiv_kernels::hash::routing::{
    FOLD_INIT, KEY_GROUPS, NULL_HASH, fold_column, hash_columns, key_group, key_group_range,
    partition_of, radix, xxh64, xxh64_fixed,
};
use proptest::prelude::*;

fn h_of(columns: &[&dyn Array]) -> Vec<u64> {
    let mut out = Vec::new();
    hash_columns(columns, &mut out).unwrap_or_else(|e| panic!("{e}"));
    out
}

/// XXH64 seed 0 on the reference implementation's published vectors.
#[test]
fn xxh64_matches_the_reference_vectors() {
    assert_eq!(xxh64(b""), 0xEF46_DB37_51D8_E999);
    assert_eq!(xxh64(b"a"), 0xD24E_C4F1_A98C_6E5B);
    assert_eq!(xxh64(b"abc"), 0x44BC_2CF5_AD77_0999);
}

/// Golden vectors pinning `H_kg` version 1. The inputs are spelled as
/// explicit little-endian bytes and as typed columns: both must give the
/// pinned value on every platform, whatever its byte order. A change here
/// is a new routing-hash version (a new plan feature), never an edit.
#[test]
fn golden_vectors_pin_h_kg_v1() {
    // One column: XXH64 of the normalized little-endian bytes.
    let int64 = Int64Array::from(vec![Some(0), Some(1), Some(-1), Some(1 << 40), None]);
    let le: [&[u8]; 4] = [
        &[0, 0, 0, 0, 0, 0, 0, 0],
        &[1, 0, 0, 0, 0, 0, 0, 0],
        &[0xFF; 8],
        &[0, 0, 0, 0, 0, 1, 0, 0],
    ];
    let h = h_of(&[&int64]);
    for (i, bytes) in le.iter().enumerate() {
        assert_eq!(h[i], xxh64(bytes), "row {i}");
    }
    assert_eq!(h[4], NULL_HASH);
    assert_eq!(
        h,
        [
            GOLDEN_I64[0],
            GOLDEN_I64[1],
            GOLDEN_I64[2],
            GOLDEN_I64[3],
            NULL_HASH
        ]
    );

    let int32 = Int32Array::from(vec![7, -7]);
    assert_eq!(
        h_of(&[&int32]),
        [xxh64(&[7, 0, 0, 0]), xxh64(&[0xF9, 0xFF, 0xFF, 0xFF])]
    );
    assert_eq!(h_of(&[&int32]), GOLDEN_I32);
    let date = Date32Array::from(vec![7]);
    assert_eq!(
        h_of(&[&date])[0],
        GOLDEN_I32[0],
        "a date hashes as its days"
    );
    let int8 = Int8Array::from(vec![-1]);
    assert_eq!(h_of(&[&int8]), [xxh64(&[0xFF])]);
    let bools = BooleanArray::from(vec![false, true]);
    assert_eq!(h_of(&[&bools]), [xxh64(&[0]), xxh64(&[1])]);

    // Floats: -0.0 is 0.0, every NaN is one NaN.
    let f64s = Float64Array::from(vec![0.0, -0.0, f64::NAN, -f64::NAN, 1.5]);
    let h = h_of(&[&f64s]);
    assert_eq!(h[0], xxh64(&0.0f64.to_bits().to_le_bytes()));
    assert_eq!((h[0], h[2]), (h[1], h[3]));
    assert_eq!(h[4], xxh64(&[0, 0, 0, 0, 0, 0, 0xF8, 0x3F]));
    assert_eq!(h, GOLDEN_F64);
    let f32s = Float32Array::from(vec![-0.0f32, 1.5]);
    assert_eq!(h_of(&[&f32s]), [xxh64(&[0; 4]), xxh64(&[0, 0, 0xC0, 0x3F])]);

    // Decimals: the unscaled value as 16 little-endian bytes.
    let dec = Decimal128Array::from(vec![12_345_i128, -1])
        .with_precision_and_scale(20, 2)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut unscaled = [0u8; 16];
    unscaled[..2].copy_from_slice(&[0x39, 0x30]);
    assert_eq!(h_of(&[&dec]), [xxh64(&unscaled), xxh64(&[0xFF; 16])]);
    assert_eq!(h_of(&[&dec]), GOLDEN_DEC);

    // Strings and binary: the bytes.
    let s = StringArray::from(vec![Some(""), Some("a"), Some("krishiv key group"), None]);
    assert_eq!(
        h_of(&[&s]),
        [
            0xEF46_DB37_51D8_E999,
            0xD24E_C4F1_A98C_6E5B,
            GOLDEN_STR,
            NULL_HASH
        ]
    );

    // Several columns: FOLD_INIT, then mix(h ^ h_col) per column.
    let a = Int64Array::from(vec![1, 2]);
    let b = StringArray::from(vec!["x", "y"]);
    let h = h_of(&[&a, &b]);
    let ha = h_of(&[&a]);
    let hb = h_of(&[&b]);
    for i in 0..2 {
        assert_eq!(h[i], fold_column(fold_column(FOLD_INIT, ha[i]), hb[i]));
    }
    assert_eq!(h, GOLDEN_TWO);
    assert_eq!(fold_column(0, 0), 0);
    assert_eq!(fold_column(FOLD_INIT, 0), GOLDEN_FOLD);
}

const GOLDEN_I64: [u64; 4] = [
    3_803_688_792_395_291_579,
    11_468_921_228_449_061_269,
    9_642_548_396_912_002_761,
    11_618_905_265_722_610_771,
];
const GOLDEN_I32: [u64; 2] = [13_243_476_881_813_409_038, 7_562_510_126_331_999_253];
const GOLDEN_F64: [u64; 5] = [
    3_803_688_792_395_291_579,
    3_803_688_792_395_291_579,
    16_838_308_782_748_609_196,
    16_838_308_782_748_609_196,
    5_329_932_555_030_153_977,
];
const GOLDEN_DEC: [u64; 2] = [13_438_840_123_374_532_013, 12_042_982_046_572_201_028];
const GOLDEN_STR: u64 = 807_839_895_623_716_763;
const GOLDEN_TWO: [u64; 2] = [17_888_868_210_822_470_624, 17_970_421_014_506_146_441];
const GOLDEN_FOLD: u64 = 79_260_605_269_340_283;

/// The fold is order-dependent and does not collapse equal columns.
#[test]
fn several_columns_are_ordered_and_equal_columns_do_not_collapse() {
    let a = Int64Array::from_iter_values(0..1000);
    let b = Int64Array::from_iter_values((0..1000).rev());
    let ab = h_of(&[&a, &b]);
    let ba = h_of(&[&b, &a]);
    assert!(ab.iter().zip(&ba).filter(|(x, y)| x == y).count() <= 1);
    let aa = h_of(&[&a, &a]);
    let mut distinct = aa.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), aa.len(), "(a, a) keys stay distinct");
    // A NULL column differs from a missing one and from a zero.
    let nulls = NullArray::new(1000);
    assert_ne!(h_of(&[&a, &nulls]), h_of(&[&a]));
    let zeros = Int64Array::from(vec![0; 1000]);
    assert_ne!(h_of(&[&a, &nulls]), h_of(&[&a, &zeros]));
}

/// Top key-group bits of sequential and composite keys are spread evenly:
/// every key group's count is within a few standard deviations.
#[test]
fn key_groups_are_spread_evenly() {
    let n = 1 << 20;
    let a = Int64Array::from_iter_values(0..n);
    let b = Int32Array::from_iter_values((0..n).map(|i| i32::try_from(i % 7).unwrap_or(0)));
    for h in [h_of(&[&a]), h_of(&[&a, &b])] {
        let mut counts = vec![0u32; KEY_GROUPS as usize];
        for &x in &h {
            counts[key_group(x) as usize] += 1;
        }
        let mean = f64::from(u32::try_from(n).unwrap_or(0)) / f64::from(KEY_GROUPS);
        let worst = counts
            .iter()
            .map(|&c| (f64::from(c) - mean).abs())
            .fold(0.0, f64::max);
        assert!(worst < 6.0 * mean.sqrt(), "worst deviation {worst}");
    }
}

#[test]
fn unsupported_types_and_bad_lengths_are_errors() {
    let ints: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    let list =
        arrow_array::ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![Some(1)])]);
    let mut out = Vec::new();
    assert!(hash_columns(&[&list], &mut out).is_err());
    let two = Int32Array::from(vec![1, 2]);
    assert!(hash_columns(&[ints.as_ref(), &two], &mut out).is_err());
    assert!(hash_columns(&[], &mut out).is_ok() && out.is_empty());
}

/// A dictionary without values (all rows NULL) hashes as NULLs.
#[test]
fn a_dictionary_without_values_is_all_null() {
    let keys = arrow_array::Int32Array::from(vec![None, None]);
    let empty = Arc::new(StringArray::from(Vec::<&str>::new()));
    let dict = DictionaryArray::<Int32Type>::try_new(keys, empty).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(h_of(&[&dict]), [NULL_HASH; 2]);
    let none: DictionaryArray<Int32Type> = std::iter::empty::<&str>().collect();
    assert!(h_of(&[&none]).is_empty());
}

fn strings() -> impl Strategy<Value = Vec<Option<String>>> {
    prop::collection::vec(prop::option::of("[a-c]{0,20}"), 0..60)
}

fn config() -> ProptestConfig {
    // Miri interprets each case, so it runs two; no failure file (Miri's
    // isolation has no file system, so persistence would call `getcwd`).
    let default = ProptestConfig::default();
    ProptestConfig {
        cases: if cfg!(miri) { 2 } else { default.cases },
        failure_persistence: None,
        ..default
    }
}

proptest! {
    #![proptest_config(config())]

    /// The unrolled widths agree with the general XXH64 bit for bit.
    #[test]
    fn fixed_widths_match_xxh64(v in any::<u128>()) {
        let b = v.to_le_bytes();
        prop_assert_eq!(xxh64_fixed::<1>(v & 0xFF), xxh64(&b[..1]));
        prop_assert_eq!(xxh64_fixed::<2>(v & 0xFFFF), xxh64(&b[..2]));
        prop_assert_eq!(xxh64_fixed::<4>(v & 0xFFFF_FFFF), xxh64(&b[..4]));
        prop_assert_eq!(xxh64_fixed::<8>(v & u128::from(u64::MAX)), xxh64(&b[..8]));
        prop_assert_eq!(xxh64_fixed::<16>(v), xxh64(&b));
    }

    /// Plain, large, view and dictionary carriers of one string column
    /// hash alike, alone and next to another column.
    #[test]
    fn string_carriers_agree(values in strings()) {
        let refs: Vec<Option<&str>> = values.iter().map(Option::as_deref).collect();
        let plain = StringArray::from(refs.clone());
        let large = LargeStringArray::from(refs.clone());
        let view = StringViewArray::from(refs.clone());
        let dict: DictionaryArray<Int32Type> = refs.iter().copied().collect();
        let dict8: DictionaryArray<Int8Type> = refs.iter().copied().collect();
        let bin = BinaryArray::from(refs.iter().map(|v| v.map(str::as_bytes)).collect::<Vec<_>>());
        let large_bin =
            LargeBinaryArray::from(refs.iter().map(|v| v.map(str::as_bytes)).collect::<Vec<_>>());
        let bin_view =
            BinaryViewArray::from(refs.iter().map(|v| v.map(str::as_bytes)).collect::<Vec<_>>());
        let expected = h_of(&[&plain]);
        for other in [&large as &dyn Array, &view, &dict, &dict8, &bin, &large_bin, &bin_view] {
            prop_assert_eq!(&h_of(&[other]), &expected);
        }
        let other = Int64Array::from_iter_values((0..i64::try_from(refs.len()).unwrap_or(0)).map(|i| i * 3));
        let with = h_of(&[&other, &plain]);
        prop_assert_eq!(&h_of(&[&other, &view]), &with);
        prop_assert_eq!(&h_of(&[&other, &dict]), &with);
    }

    /// Integer dictionaries and decimal carriers hash as their values.
    #[test]
    fn value_carriers_agree(values in prop::collection::vec(prop::option::of(-5i64..5), 0..60)) {
        let plain = Int64Array::from(values.clone());
        let keys = arrow_array::Int16Array::from(
            (0..i16::try_from(values.len()).unwrap_or(0)).map(Some).collect::<Vec<_>>(),
        );
        let dict = DictionaryArray::<Int16Type>::try_new(keys, Arc::new(plain.clone()))
            .unwrap_or_else(|e| panic!("{e}"));
        prop_assert_eq!(h_of(&[&dict]), h_of(&[&plain]));
        let d64 = Decimal64Array::from(values.clone()).with_precision_and_scale(18, 2)
            .unwrap_or_else(|e| panic!("{e}"));
        let d128 = Decimal128Array::from(values.iter().map(|v| v.map(i128::from)).collect::<Vec<_>>())
            .with_precision_and_scale(18, 2)
            .unwrap_or_else(|e| panic!("{e}"));
        prop_assert_eq!(h_of(&[&d64]), h_of(&[&d128]));
    }

    /// A row's partition owns its key group; partitions own contiguous,
    /// covering key-group ranges; `2^k` partitions are the top `k` bits
    /// and nest inside radix partitions of fewer bits.
    #[test]
    fn key_group_ranges_route(h in any::<u64>(), n in 1u32..=4096, k in 0u32..=20, r0 in 0u32..=8) {
        let p = partition_of(h, n);
        prop_assert!(p < n);
        prop_assert!(key_group_range(p, n).contains(&key_group(h)));
        let pow = 1u32 << k;
        prop_assert_eq!(partition_of(h, pow), radix(h, k));
        if r0 <= k {
            prop_assert_eq!(radix(h, r0), partition_of(h, pow) >> (k - r0));
        }
        prop_assert!(partition_of(h, 1 << 20) < 1 << 20);
        prop_assert!(key_group_range(partition_of(h, 1 << 20), 1 << 20).contains(&key_group(h)));
    }
}

/// Ranges of every count tile `[0, 4096)` in order, and merging adjacent
/// partitions (AQE coalescing) gives contiguous ranges.
#[test]
fn key_group_ranges_tile() {
    for n in (1..=64).chain([100, 1000, 4095, 4096]) {
        let mut at = 0;
        for p in 0..n {
            let r = key_group_range(p, n);
            assert_eq!(r.start, at, "n = {n}, p = {p}");
            assert!(!r.is_empty(), "n = {n}, p = {p}");
            at = r.end;
        }
        assert_eq!(at, KEY_GROUPS);
        assert!(key_group_range(n, n).is_empty());
    }
    assert_eq!(key_group_range(0, 0), 0..KEY_GROUPS);
    assert_eq!(partition_of(u64::MAX, 0), 0);
    assert_eq!(radix(u64::MAX, 0), 0);
    assert_eq!(key_group(u64::MAX), KEY_GROUPS - 1);
}
