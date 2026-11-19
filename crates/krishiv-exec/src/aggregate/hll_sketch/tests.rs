// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the HLL sketch accumulators.

use super::*;
use krishiv_common::error::ErrorClass;

fn longs(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn strs(v: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(v.to_vec()))
}

fn binaries(v: &[Option<&[u8]>]) -> ArrayRef {
    Arc::new(BinaryArray::from_iter(v.iter().copied()))
}

fn estimate(a: &HllSketchAcc, g: u32) -> i64 {
    let out = a.evaluate(&[g]).unwrap();
    let b = out.as_any().downcast_ref::<BinaryViewArray>().unwrap();
    hll::estimate(&hll::from_bytes(b.value(0), "t").unwrap())
}

#[test]
fn sketch_counts_distinct_values_per_group_and_skips_nulls_and_empties() {
    let mut a = HllSketchAcc::new(HllSketchOp::Sketch(12)).unwrap();
    a.resize(3);
    a.update(
        &[0, 0, 0, 0, 1, 1],
        &longs(&[Some(1), Some(1), Some(2), None, Some(7), Some(8)]),
        |_| true,
    )
    .unwrap();
    assert_eq!(
        (estimate(&a, 0), estimate(&a, 1), estimate(&a, 2)),
        (2, 2, 0)
    );
    let mut s = HllSketchAcc::new(HllSketchOp::Sketch(12)).unwrap();
    s.resize(1);
    s.update(
        &[0; 4],
        &strs(&[Some(""), Some("a"), Some("a"), None]),
        |_| true,
    )
    .unwrap();
    assert_eq!(estimate(&s, 0), 1);
    let mut b = HllSketchAcc::new(HllSketchOp::Sketch(12)).unwrap();
    b.resize(1);
    b.update(
        &[0; 3],
        &binaries(&[Some(b""), Some(b"a"), Some(b"b")]),
        |_| true,
    )
    .unwrap();
    assert_eq!(estimate(&b, 0), 2);
}

#[test]
fn the_filter_mask_drops_rows() {
    let mut a = HllSketchAcc::new(HllSketchOp::Sketch(12)).unwrap();
    a.resize(1);
    a.update(&[0, 0], &longs(&[Some(1), Some(2)]), |i| i == 0)
        .unwrap();
    assert_eq!(estimate(&a, 0), 1);
}

#[test]
fn a_bad_lg_k_is_refused() {
    for k in [0_u8, 3, 22] {
        let e = HllSketchAcc::new(HllSketchOp::Sketch(k)).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::HllInvalidLgK);
    }
}

#[test]
fn partial_states_merge_like_one_pass() {
    let mut p1 = HllSketchAcc::new(HllSketchOp::Sketch(10)).unwrap();
    let mut p2 = HllSketchAcc::new(HllSketchOp::Sketch(10)).unwrap();
    p1.resize(1);
    p2.resize(1);
    p1.update(&[0; 3], &longs(&[Some(1), Some(2), Some(3)]), |_| true)
        .unwrap();
    p2.update(&[0; 3], &longs(&[Some(3), Some(4), Some(5)]), |_| true)
        .unwrap();
    let mut f = HllSketchAcc::new(HllSketchOp::Sketch(10)).unwrap();
    f.resize(1);
    for p in [&p1, &p2] {
        let st = p.state(&[0]).unwrap();
        f.merge(&[0], st[0].as_any().downcast_ref::<BinaryArray>().unwrap())
            .unwrap();
    }
    assert_eq!(estimate(&f, 0), 5);
}

#[test]
fn union_takes_the_first_lg_k_and_checks_the_rest() {
    let sketch = |k: u8, v: i64| {
        let mut a = HllSketchAcc::new(HllSketchOp::Sketch(k)).unwrap();
        a.resize(1);
        a.update(&[0], &longs(&[Some(v)]), |_| true).unwrap();
        let st = a.state(&[0]).unwrap();
        st[0]
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .value(0)
            .to_vec()
    };
    let (s12, s20, s12b) = (sketch(12, 1), sketch(20, 1), sketch(12, 2));
    let input = binaries(&[Some(&s12), None, Some(&s12b)]);
    let mut u = HllSketchAcc::new(HllSketchOp::Union(false)).unwrap();
    u.resize(2);
    u.update(&[0, 0, 0], &input, |_| true).unwrap();
    assert_eq!((estimate(&u, 0), estimate(&u, 1)), (2, 0));
    let mixed = binaries(&[Some(&s12), Some(&s20)]);
    let e = u.update(&[1, 1], &mixed, |_| true).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::HllUnionDifferentLgK);
    let mut ok = HllSketchAcc::new(HllSketchOp::Union(true)).unwrap();
    ok.resize(1);
    ok.update(&[0, 0], &mixed, |_| true).unwrap();
    assert_eq!(estimate(&ok, 0), 1);
}

#[test]
fn union_of_garbage_is_the_buffer_error() {
    let mut u = HllSketchAcc::new(HllSketchOp::Union(false)).unwrap();
    u.resize(1);
    let e = u
        .update(&[0], &binaries(&[Some(b"abc")]), |_| true)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::HllInvalidInputSketchBuffer);
}

#[test]
fn size_follows_the_coupons_not_the_full_registers() {
    let mut a = HllSketchAcc::new(HllSketchOp::Sketch(12)).unwrap();
    a.resize(1000);
    let before = a.size();
    // A thousand empty groups are far below one full sketch each.
    assert!(before < 1000 * 4096 / 4, "{before}");
    a.update(&[0; 3], &longs(&[Some(1), Some(2), Some(3)]), |_| true)
        .unwrap();
    assert!(a.size() >= before);
    assert!(a.size() < before + 1024);
    // A new group is projected at a header, and a row at what one update can add to the largest group.
    assert!(a.projected(5, 10) >= a.size() + 5 * 128 + 10 * 128);
    assert!(a.projected(5, 10) < a.size() + 5 * 4096);
    // Many values in one group reach the full registers and stop there.
    let many: Vec<Option<i64>> = (0..20_000).map(Some).collect();
    a.update(&vec![1; many.len()], &longs(&many), |_| true)
        .unwrap();
    let full = a.size();
    a.update(&vec![1; many.len()], &longs(&many), |_| true)
        .unwrap();
    assert_eq!(a.size(), full);
    // The group went from a list to the 4,096 registers (less the little it started with).
    assert!(full >= before + 3800, "{full} vs {before}");
}

/// Review 552b06e8b finding 1: a set that has doubled, and a sketch just promoted to registers, are charged
/// what they hold (`estimated_size`), at every step of a group's growth.
#[test]
fn a_group_is_charged_at_least_what_its_sketch_holds_in_every_mode() {
    let mut a = HllSketchAcc::new(HllSketchOp::Sketch(12)).unwrap();
    a.resize(2);
    let mut next = 0_i64;
    for step in [3_i64, 100, 90, 100, 100, 100, 1000] {
        let vals: Vec<Option<i64>> = (next..next + step).map(Some).collect();
        next += step;
        a.update(&vec![1; vals.len()], &longs(&vals), |_| true)
            .unwrap();
        let Groups::Sketches(v, _, total, _) = &a.groups else {
            panic!("sketches");
        };
        assert!(
            *total
                >= v.iter()
                    .map(|s| OVERHEAD + s.estimated_size())
                    .sum::<usize>(),
            "after {next} values"
        );
        // And the charge never exceeds the registers plus the header.
        assert!(v[1].estimated_size() <= (1 << 12) + 64);
    }
}

#[test]
fn every_value_carrier_is_counted_and_others_are_refused() {
    use arrow_array::{LargeStringArray, StringViewArray};
    let carriers: Vec<ArrayRef> = vec![
        longs(&[Some(1), Some(2), Some(2)]),
        strs(&[Some("a"), Some("b"), Some("b")]),
        Arc::new(LargeStringArray::from(vec![
            Some("a"),
            Some("b"),
            Some("b"),
        ])),
        Arc::new(StringViewArray::from(vec![Some("a"), Some("b"), Some("b")])),
        binaries(&[Some(b"a"), Some(b"b"), Some(b"b")]),
        Arc::new(BinaryViewArray::from_iter(vec![
            Some(&b"a"[..]),
            Some(&b"b"[..]),
            Some(&b"b"[..]),
        ])),
    ];
    for arg in carriers {
        let mut a = HllSketchAcc::new(HllSketchOp::Sketch(12)).unwrap();
        a.resize(1);
        a.update(&[0, 0, 0], &arg, |_| true).unwrap();
        assert_eq!(estimate(&a, 0), 2, "{:?}", arg.data_type());
    }
    let mut a = HllSketchAcc::new(HllSketchOp::Sketch(12)).unwrap();
    a.resize(1);
    let bad: ArrayRef = Arc::new(arrow_array::BooleanArray::from(vec![true]));
    assert!(a.update(&[0], &bad, |_| true).is_err());
}

#[test]
fn malformed_inputs_are_internal_errors() {
    let mut a = HllSketchAcc::new(HllSketchOp::Sketch(12)).unwrap();
    a.resize(1);
    // A column and group ids of different lengths.
    assert!(a.update(&[0, 0], &longs(&[Some(1)]), |_| true).is_err());
    // A group past the accumulator's.
    assert!(a.update(&[7], &longs(&[Some(1)]), |_| true).is_err());
    assert!(a.state(&[7]).is_err());
    // A state of another length than its groups.
    let one = binaries(&[Some(b"x")]);
    let states = one.as_any().downcast_ref::<BinaryArray>().unwrap();
    assert!(a.merge(&[0, 0], states).is_err());

    let mut u = HllSketchAcc::new(HllSketchOp::Union(false)).unwrap();
    u.resize(1);
    // A union column that is not binary.
    assert!(u.update(&[0], &longs(&[Some(1)]), |_| true).is_err());
    // A group past the accumulator's.
    assert!(u.update(&[7], &binaries(&[Some(&[])]), |_| true).is_err());
    assert!(u.state(&[7]).is_err());
}

#[test]
fn reserve_shrink_and_empty_union_groups() {
    for op in [HllSketchOp::Sketch(12), HllSketchOp::Union(false)] {
        let mut a = HllSketchAcc::new(op).unwrap();
        a.reserve(4);
        a.resize(3);
        let three = a.size();
        a.resize(1);
        // A union group with no sketch yet holds nothing; a sketch group holds its empty sketch.
        if matches!(op, HllSketchOp::Sketch(_)) {
            assert!(a.size() < three, "{} vs {three}", a.size());
        }
        assert!(a.size() <= a.projected(0, 0));
        // A union group that saw no sketch has an empty state.
        if matches!(op, HllSketchOp::Union(_)) {
            let st = a.state(&[0]).unwrap();
            let b = st[0].as_any().downcast_ref::<BinaryArray>().unwrap();
            assert!(b.value(0).is_empty());
        }
    }
}

#[test]
fn partial_states_merge_into_a_union_and_a_view_input_is_read() {
    let mut s = HllSketchAcc::new(HllSketchOp::Sketch(12)).unwrap();
    s.resize(1);
    s.update(&[0, 0], &longs(&[Some(1), Some(2)]), |_| true)
        .unwrap();
    let st = s.state(&[0]).unwrap();
    let bytes = st[0].as_any().downcast_ref::<BinaryArray>().unwrap();

    let mut u = HllSketchAcc::new(HllSketchOp::Union(false)).unwrap();
    u.resize(1);
    u.merge(&[0], bytes).unwrap();
    assert_eq!(estimate(&u, 0), 2);

    let view: ArrayRef = Arc::new(BinaryViewArray::from_iter(vec![Some(bytes.value(0))]));
    let mut v = HllSketchAcc::new(HllSketchOp::Union(false)).unwrap();
    v.resize(1);
    v.update(&[0], &view, |_| true).unwrap();
    assert_eq!(estimate(&v, 0), 2);
}
