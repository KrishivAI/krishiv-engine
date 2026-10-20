// SPDX-License-Identifier: Apache-2.0
//! Row forms chosen by build size (P2-T163 part 2): every form keeps every
//! matching row, the exact forms keep nothing else, evaluation over a
//! selection and over all rows agree, hashes passed by a producer are
//! reused, and filters survive their stable encoding.
#![cfg(not(loom))]

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, StringArray,
    StringViewArray, TimestampMicrosecondArray,
};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use proptest::prelude::*;

use super::*;
use crate::ScalarValue;

/// `y` as a float (the test domain fits `i16`).
fn f32_of(y: i32) -> f32 {
    f32::from(i16::try_from(y).unwrap_or_else(|e| panic!("{e}")))
}

fn filter(keys: &dyn Array, options: KeyFilterOptions) -> KeyFilter {
    let mut b = KeyFilterBuilder::new(keys.data_type().clone(), options)
        .unwrap_or_else(|| panic!("unsupported {}", keys.data_type()));
    b.add(keys, None).unwrap_or_else(|e| panic!("{e}"));
    b.finish()
}

fn forced(max_values: usize, exact: usize, bloom: bool) -> KeyFilterOptions {
    KeyFilterOptions {
        max_values,
        exact_max_values: exact,
        bloom,
        ..KeyFilterOptions::default()
    }
}

fn pass(f: &KeyFilter, column: &dyn Array) -> Vec<bool> {
    f.evaluate(column)
        .unwrap_or_else(|| panic!("{} does not widen", column.data_type()))
        .iter()
        .map(|b| b == Some(true))
        .collect()
}

fn round_trip(f: &KeyFilter) -> KeyFilter {
    let bytes = f.encode();
    let back = KeyFilter::decode(&bytes).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(&back, f, "decode(encode(f)) == f");
    assert_eq!(back.encode().len(), bytes.len());
    back
}

/// Build keys of a key type, and the same values in the probe's
/// representation (a narrower or other one where the type has one).
fn keyed(kind: u8, v: &[Option<i32>]) -> (ArrayRef, ArrayRef) {
    let ints = |f: fn(i32) -> i64| v.iter().map(|x| x.map(f)).collect::<Int64Array>();
    match kind {
        0 => (
            Arc::new(ints(i64::from)),
            Arc::new(v.iter().copied().collect::<Int32Array>()),
        ),
        1 => (
            Arc::new(
                v.iter()
                    .map(|x| x.map(|y| i128::from(y) * 100))
                    .collect::<Decimal128Array>()
                    .with_precision_and_scale(12, 2)
                    .unwrap_or_else(|e| panic!("{e}")),
            ),
            Arc::new(
                v.iter()
                    .map(|x| x.map(|y| i128::from(y) * 10))
                    .collect::<Decimal128Array>()
                    .with_precision_and_scale(10, 1)
                    .unwrap_or_else(|e| panic!("{e}")),
            ),
        ),
        2 => (
            Arc::new(
                v.iter()
                    .map(|x| x.map(|y| format!("k{y}")))
                    .collect::<StringViewArray>(),
            ),
            Arc::new(
                v.iter()
                    .map(|x| x.map(|y| format!("k{y}")))
                    .collect::<StringArray>(),
            ),
        ),
        3 => (
            Arc::new(
                v.iter()
                    .map(|x| x.map(|y| if y == 0 { -0.0 } else { f64::from(y) / 4.0 }))
                    .collect::<Float64Array>(),
            ),
            Arc::new(
                v.iter()
                    .map(|x| x.map(|y| f32_of(y) / 4.0))
                    .collect::<Float32Array>(),
            ),
        ),
        4 => (
            Arc::new(
                v.iter()
                    .map(|x| x.map(|y| if y == 0 { -0.0 } else { f32_of(y) * 0.5 }))
                    .collect::<Float32Array>(),
            ),
            Arc::new(
                v.iter()
                    .map(|x| x.map(|y| f32_of(y) * 0.5))
                    .collect::<Float32Array>(),
            ),
        ),
        _ => (
            Arc::new(v.iter().copied().collect::<Int32Array>()),
            Arc::new(
                v.iter()
                    .map(|x| x.and_then(|y| i16::try_from(y).ok()))
                    .collect::<Int16Array>(),
            ),
        ),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(160))]
    /// Every form keeps every probe row whose key a producer stored; the
    /// exact forms (range of dense integers, small set, hash set) keep no
    /// other; NULL rows pass exactly when a NULL key was kept; and the
    /// filter decodes to one that answers the same.
    #[test]
    fn forms_are_supersets_and_exact_forms_are_exact(
        kind in 0u8..6,
        build_keys in proptest::collection::vec(proptest::option::weighted(0.9, -60i32..60), 0..120),
        probe_keys in proptest::collection::vec(proptest::option::weighted(0.9, -80i32..80), 0..300),
        options in prop_oneof![
            Just(forced(0, 0, false)),
            Just(forced(0, SMALL_SET_MAX, false)),
            Just(forced(5, 1000, false)),
            Just(forced(0, 0, true)),
            Just(forced(50, EXACT_SET_MAX, true)),
        ],
        null_safe in any::<bool>(),
        split in 0usize..120,
    ) {
        let split = split.min(build_keys.len());
        let mut parts = Vec::new();
        for keys in [&build_keys[..split], &build_keys[split..]] {
            let (b, _) = keyed(kind, keys);
            let mut x = KeyFilterBuilder::new(b.data_type().clone(), options).unwrap_or_else(|| panic!());
            // An `=` join does not store rows with a NULL key.
            let include = (!null_safe).then(|| b.logical_nulls()).flatten();
            x.add(b.as_ref(), include.as_ref()).unwrap_or_else(|e| panic!("{e}"));
            parts.push(x);
        }
        let mut b = parts.remove(0);
        b.merge(parts.remove(0));
        let f = b.finish();
        let back = round_trip(&f);
        let (_, probe) = keyed(kind, &probe_keys);
        let out = pass(&f, probe.as_ref());
        prop_assert_eq!(&pass(&back, probe.as_ref()), &out);
        let kept_null = null_safe && build_keys.iter().any(Option::is_none);
        let stored = |k: i32| {
            // The probe of kind 5 is int16: a key that does not fit is NULL.
            build_keys.contains(&Some(k))
        };
        let dense = match (kind, f.range_scalars()) {
            (0, Some((ScalarValue::Int64(lo), ScalarValue::Int64(hi)))) => {
                let mut d: Vec<i32> = build_keys.iter().flatten().copied().collect();
                d.sort_unstable();
                d.dedup();
                i64::try_from(d.len()).ok() == Some(hi - lo + 1)
            }
            _ => false,
        };
        let exact = matches!(f.form(), FilterForm::SmallSet(_) | FilterForm::HashSet(_))
            || (dense && f.form() == FilterForm::Range);
        for (i, k) in probe_keys.iter().enumerate() {
            let k = if kind == 5 { k.filter(|y| i16::try_from(*y).is_ok()) } else { *k };
            match k {
                None => prop_assert_eq!(out[i], kept_null, "NULL row {}", i),
                Some(k) if stored(k) => prop_assert!(out[i], "key {} of kind {} dropped ({:?})", k, kind, f.form()),
                Some(k) => {
                    if exact {
                        prop_assert!(!out[i], "key {} of kind {} kept by {:?}", k, kind, f.form());
                    }
                }
            }
        }
    }
}

/// The form follows the distinct key count: the range alone for dense
/// integers, a small set, a hash set, then a Bloom filter (or the range
/// when no Bloom was planned).
#[test]
fn form_follows_build_size() {
    let o = KeyFilterOptions::default();
    let dense = filter(&Int64Array::from_iter_values(0..1000), o);
    assert_eq!(dense.form(), FilterForm::Range);
    assert_eq!(
        pass(&dense, &Int64Array::from(vec![-1, 0, 999, 1000])),
        [false, true, true, false]
    );
    let spread = |n: i64| Int64Array::from_iter_values((0..n).map(|i| i * 7));
    assert_eq!(filter(&spread(8), o).form(), FilterForm::SmallSet(8));
    assert_eq!(filter(&spread(9), o).form(), FilterForm::HashSet(9));
    let at_cap = i64::try_from(EXACT_SET_MAX).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        filter(&spread(at_cap), o).form(),
        FilterForm::HashSet(EXACT_SET_MAX)
    );
    let over = filter(&spread(at_cap + 1), o);
    assert!(
        matches!(over.form(), FilterForm::Bloom(_)),
        "{:?}",
        over.form()
    );
    assert_eq!(over.bloom_bytes().map(|b| b > 0), Some(true));
    assert_eq!(
        filter(&spread(at_cap + 1), forced(50, EXACT_SET_MAX, false)).form(),
        FilterForm::Range
    );
    // The exact forms cost less than a Bloom; the range least.
    let costs: Vec<f64> = [dense, filter(&spread(8), o), filter(&spread(9), o)]
        .iter()
        .map(KeyFilter::row_cost_ns)
        .collect();
    assert!(costs.windows(2).all(|w| w[0] <= w[1]), "{costs:?}");
    // A failed reservation keeps only the IN-list sized distinct keys.
    let mut b = KeyFilterBuilder::new(DataType::Int64, o).unwrap_or_else(|| panic!());
    b.add(&spread(100), None).unwrap_or_else(|e| panic!("{e}"));
    let before = b.heap_bytes();
    b.drop_bloom();
    assert!(b.heap_bytes() < before);
    assert_eq!(b.finish().form(), FilterForm::Range);
}

/// Hashes a producer already computed with the same row hash are used as
/// given; any other hashes are ignored and the keys hashed again.
#[test]
fn producer_hashes_are_reused() {
    let keys = Int64Array::from(vec![Some(3), None, Some(70_000), Some(3)]);
    let o = forced(0, 0, true);
    let norm = normalize_key_bytes(&keys, &DataType::Int64).unwrap_or_else(|| panic!());
    let h = norm.hashes();
    let mut given = KeyFilterBuilder::new(DataType::Int64, o).unwrap_or_else(|| panic!());
    given
        .add_hashed(&keys, None, Some((RowHash::CURRENT, &h)))
        .unwrap_or_else(|e| panic!("{e}"));
    let mut own = KeyFilterBuilder::new(DataType::Int64, o).unwrap_or_else(|| panic!());
    own.add(&keys, None).unwrap_or_else(|e| panic!("{e}"));
    let mut short = KeyFilterBuilder::new(DataType::Int64, o).unwrap_or_else(|| panic!());
    short
        .add_hashed(&keys, None, Some((RowHash::CURRENT, &h[..1])))
        .unwrap_or_else(|e| panic!("{e}"));
    let (given, own, short) = (given.finish(), own.finish(), short.finish());
    assert!(matches!(own.form(), FilterForm::Bloom(_)));
    assert_eq!(given, own);
    assert_eq!(short, own, "a wrong-length hash vector is not used");
    // The specialised widths are XXH64 of the key type's bytes.
    assert_eq!(h[0], xxh64(&3i64.to_le_bytes()));
    assert_eq!(xxh64_fixed::<16>(7), xxh64(&7u128.to_le_bytes()));
}

/// Evaluation over a selection (most rows out of range) and over every
/// row give the same answers for each hashed form.
#[test]
fn selected_and_full_evaluation_agree() {
    let build = Int64Array::from_iter_values((0..200).map(|i| i * 13));
    // 90% of the probe is below the range: the hashed forms probe only the
    // rest.
    let probe =
        Int64Array::from_iter_values((0..4000).map(|i| if i % 10 == 0 { i } else { -i - 1 }));
    let wide = Int64Array::from_iter_values((0..4000).map(|i| i % 2600));
    for o in [forced(0, 1000, false), forced(0, 0, true)] {
        let f = filter(&build, o);
        let got = pass(&f, &probe);
        for (i, v) in probe.values().iter().enumerate() {
            let member = *v >= 0 && v % 13 == 0 && *v < 2600;
            assert!(!member || got[i], "{v} dropped by {:?}", f.form());
            if matches!(f.form(), FilterForm::HashSet(_)) {
                assert_eq!(got[i], member, "{v}");
            }
        }
        let got = pass(&f, &wide);
        for (i, v) in wide.values().iter().enumerate() {
            assert!(v % 13 != 0 || got[i], "{v} dropped by {:?}", f.form());
        }
    }
}

/// Every key type survives the encoding, for every form; the decoded
/// filter answers the same.
#[test]
fn every_key_type_round_trips() {
    let ts =
        TimestampMicrosecondArray::from(vec![Some(7), None, Some(9)]).with_timezone("Asia/Kolkata");
    let columns: Vec<ArrayRef> = vec![
        Arc::new(BooleanArray::from(vec![Some(true), None])),
        Arc::new(Int8Array::from(vec![-3, 4, 5])),
        Arc::new(Int16Array::from(vec![-3, 400])),
        Arc::new(Int32Array::from(vec![Some(1), None, Some(-9)])),
        Arc::new(Date32Array::from(vec![10, 20])),
        Arc::new(ts),
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2])),
        Arc::new(Float32Array::from(vec![1.5, -0.0, f32::NAN])),
        Arc::new(Float64Array::from(vec![2.5, -0.0])),
        Arc::new(
            Decimal128Array::from(vec![12_345, -7])
                .with_precision_and_scale(20, 3)
                .unwrap_or_else(|e| panic!("{e}")),
        ),
        Arc::new(StringArray::from(vec![Some("a"), None, Some("zz")])),
        Arc::new(LargeStringArray::from(vec!["x", "y"])),
        Arc::new(StringViewArray::from(vec![
            "long enough to not be inlined",
            "s",
        ])),
        Arc::new(BinaryArray::from(vec![b"\x00\x01".as_ref(), b"\xff"])),
        Arc::new(arrow_array::LargeBinaryArray::from(vec![b"q".as_ref()])),
        Arc::new(arrow_array::BinaryViewArray::from(vec![
            b"q".as_ref(),
            b"r",
        ])),
    ];
    for c in &columns {
        let include = NullBuffer::from(vec![true; c.len()]);
        for o in [
            forced(0, 0, false),
            forced(50, 1000, false),
            forced(0, 0, true),
        ] {
            let mut b = KeyFilterBuilder::new(c.data_type().clone(), o).unwrap_or_else(|| panic!());
            b.add(c.as_ref(), Some(&include))
                .unwrap_or_else(|e| panic!("{e}"));
            let f = b.finish();
            let back = round_trip(&f);
            assert_eq!(
                pass(&back, c.as_ref()),
                pass(&f, c.as_ref()),
                "{}",
                c.data_type()
            );
            assert!(
                pass(&f, c.as_ref())
                    .iter()
                    .zip(
                        c.logical_nulls()
                            .map_or(vec![true; c.len()], |n| n.iter().collect())
                    )
                    .all(|(p, valid)| !valid || *p)
            );
            let range = round_trip(&f.range_only());
            assert_eq!(range.form(), FilterForm::Range);
        }
    }
    // Empty and NULL-only builds.
    let mut e = KeyFilterBuilder::new(DataType::Int64, KeyFilterOptions::default())
        .unwrap_or_else(|| panic!());
    e.add(&Int64Array::from(vec![None::<i64>]), None)
        .unwrap_or_else(|e| panic!("{e}"));
    let e = round_trip(&e.finish());
    assert!(e.is_empty() && e.nulls_pass());
}

/// Decoding untrusted bytes never panics: every truncation, every single
/// corrupted byte and other versions are refused or decode to some filter.
#[test]
fn corrupt_encodings_are_refused() {
    let forms = [
        filter(
            &StringArray::from(vec!["a", "bb", "ccc"]),
            forced(50, 1000, false),
        ),
        filter(
            &Int64Array::from_iter_values((0..40).map(|i| i * 3)),
            forced(50, 1000, false),
        ),
        filter(
            &Int64Array::from_iter_values((0..40).map(|i| i * 3)),
            forced(0, 0, true),
        ),
        filter(
            &Decimal128Array::from(vec![1, 2, 30])
                .with_precision_and_scale(9, 2)
                .unwrap_or_else(|e| panic!("{e}")),
            forced(0, 1000, false),
        ),
    ];
    for f in &forms {
        let bytes = f.encode();
        for n in 0..bytes.len() {
            assert!(KeyFilter::decode(&bytes[..n]).is_err(), "truncated at {n}");
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(KeyFilter::decode(&longer).is_err(), "trailing bytes");
        for i in 0..bytes.len() {
            for flip in [0x01u8, 0x80, 0xff] {
                let mut b = bytes.clone();
                b[i] ^= flip;
                // Either refused or some well-formed filter: never a panic.
                let _ = KeyFilter::decode(&b);
            }
        }
        for (at, v) in [
            (0, FORMAT_VERSION + 1),
            (1, 0),
            (2, NORMALIZATION_VERSION + 1),
        ] {
            let mut b = bytes.clone();
            b[at] = v;
            assert!(KeyFilter::decode(&b).is_err(), "header byte {at}");
        }
    }
    assert!(KeyFilter::decode(&[]).is_err());
    // A small set beyond its cap (each probe row compares every member) is
    // refused: re-encode an 8-member set with a ninth member.
    let small = filter(
        &Int64Array::from_iter_values((0..8).map(|i| i * 3)),
        forced(0, 8, false),
    );
    assert_eq!(small.form(), FilterForm::SmallSet(8));
    let mut bytes = small.encode();
    let tail = bytes.len() - 8 * 8;
    let count = tail - 4;
    assert_eq!(bytes[count..tail], 8u32.to_le_bytes());
    bytes[count..tail].copy_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&99i64.to_le_bytes());
    assert!(KeyFilter::decode(&bytes).is_err());
}
