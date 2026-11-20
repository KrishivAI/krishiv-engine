// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the bitmap accumulator.

use super::*;

fn longs(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn binaries(v: &[Option<&[u8]>]) -> ArrayRef {
    Arc::new(BinaryArray::from_iter(v.iter().copied()))
}

fn out(a: &BitmapAcc, g: u32) -> Vec<u8> {
    let s = a.state(&[g]).unwrap();
    let b = s[0].as_any().downcast_ref::<BinaryArray>().unwrap();
    b.value(0).to_vec()
}

#[test]
fn construct_sets_low_bit_first_and_skips_null() {
    let mut a = BitmapAcc::new(BitmapOp::Construct);
    a.resize(2);
    let arg = longs(&[Some(0), Some(9), Some(32767), None, Some(1)]);
    a.update(&[0, 0, 0, 0, 1], &arg, |_| true).unwrap();
    let g0 = out(&a, 0);
    assert_eq!(g0.len(), NUM_BYTES);
    assert_eq!((g0[0], g0[1], g0[4095]), (1, 2, 0x80));
    assert_eq!(g0.iter().map(|b| b.count_ones()).sum::<u32>(), 3);
    assert_eq!(out(&a, 1)[0], 2);
}

#[test]
fn construct_refuses_a_position_outside_the_buffer() {
    for bad in [-1_i64, 32768, i64::MAX] {
        let mut a = BitmapAcc::new(BitmapOp::Construct);
        a.resize(1);
        let err = a.update(&[0], &longs(&[Some(bad)]), |_| true).unwrap_err();
        assert_eq!(
            err.error_class(),
            krishiv_common::error::ErrorClass::InvalidBitmapPosition
        );
    }
}

#[test]
fn empty_groups_are_the_start_value() {
    for (op, byte) in [
        (BitmapOp::Construct, 0),
        (BitmapOp::Or, 0),
        (BitmapOp::And, 0xFF),
    ] {
        let mut a = BitmapAcc::new(op);
        a.resize(1);
        assert_eq!(out(&a, 0), vec![byte; NUM_BYTES]);
    }
}

#[test]
fn or_and_and_over_shorter_inputs() {
    let mut or = BitmapAcc::new(BitmapOp::Or);
    or.resize(1);
    let arg = binaries(&[Some(&[0x10]), Some(&[0x20, 0x01]), None, Some(&[0x40])]);
    or.update(&[0; 4], &arg, |_| true).unwrap();
    let o = out(&or, 0);
    assert_eq!((o[0], o[1], o[2]), (0x70, 0x01, 0));

    let mut and = BitmapAcc::new(BitmapOp::And);
    and.resize(1);
    let arg = binaries(&[
        Some(&[0xF0, 0xFF]),
        Some(&[0x70]),
        None,
        Some(&[0x30, 0x0F]),
    ]);
    and.update(&[0; 4], &arg, |_| true).unwrap();
    let a = out(&and, 0);
    // Bytes past a shorter input are zero; the second input had one byte only.
    assert_eq!((a[0], a[1], a[2], a[4095]), (0x30, 0, 0, 0));
}

#[test]
fn merging_partial_states_equals_one_pass() {
    for op in [BitmapOp::Or, BitmapOp::And] {
        let rows = [vec![0xF0_u8; NUM_BYTES], vec![0x3C; NUM_BYTES]];
        let mut whole = BitmapAcc::new(op);
        whole.resize(1);
        let arg = binaries(&[Some(&rows[0]), Some(&rows[1])]);
        whole.update(&[0, 0], &arg, |_| true).unwrap();
        let mut merged = BitmapAcc::new(op);
        merged.resize(1);
        for r in &rows {
            let mut part = BitmapAcc::new(op);
            part.resize(1);
            part.update(&[0], &binaries(&[Some(r)]), |_| true).unwrap();
            let s = part.state(&[0]).unwrap();
            let s = s[0].as_any().downcast_ref::<BinaryArray>().unwrap();
            merged.merge(&[0], s).unwrap();
        }
        assert_eq!(out(&whole, 0), out(&merged, 0));
    }
}

#[test]
fn the_mask_drops_rows_and_memory_is_charged_per_group() {
    let mut a = BitmapAcc::new(BitmapOp::Construct);
    a.resize(1);
    a.update(&[0, 0], &longs(&[Some(1), Some(2)]), |i| i == 1)
        .unwrap();
    assert_eq!(out(&a, 0)[0], 4);
    assert!(a.projected(3) >= 4 * NUM_BYTES);
    assert!(a.size() >= NUM_BYTES);
}

#[test]
fn reserve_and_evaluate_return_the_bitmaps() {
    let mut a = BitmapAcc::new(BitmapOp::Construct);
    a.reserve(3);
    a.resize(2);
    a.update(&[1], &longs(&[Some(3)]), |_| true).unwrap();
    let r = a.evaluate(&[1, 0]).unwrap();
    let r = r.as_any().downcast_ref::<BinaryViewArray>().unwrap();
    assert_eq!((r.value(0)[0], r.value(1)[0]), (8, 0));
}

#[test]
fn a_view_binary_input_merges_like_a_binary_one() {
    let mut a = BitmapAcc::new(BitmapOp::Or);
    a.resize(1);
    let arg: ArrayRef = Arc::new(BinaryViewArray::from_iter(vec![
        Some(&[0x01_u8][..]),
        None,
        Some(&[0x02][..]),
    ]));
    a.update(&[0, 0, 0], &arg, |_| true).unwrap();
    assert_eq!(out(&a, 0)[0], 3);
}

#[test]
fn malformed_inputs_are_internal_errors() {
    let mut a = BitmapAcc::new(BitmapOp::Construct);
    a.resize(1);
    // Column and group ids of different lengths.
    assert!(a.update(&[0, 0], &longs(&[Some(1)]), |_| true).is_err());
    // Positions of another type.
    let text: ArrayRef = Arc::new(arrow_array::StringArray::from(vec!["x"]));
    assert!(a.update(&[0], &text, |_| true).is_err());

    let mut o = BitmapAcc::new(BitmapOp::Or);
    o.resize(1);
    // A bitmap column of another type.
    assert!(o.update(&[0], &longs(&[Some(1)]), |_| true).is_err());
    // A state of another length than its groups.
    let states = BinaryArray::from_iter_values([&[1_u8][..]]);
    assert!(o.merge(&[0, 0], &states).is_err());
    // A group past the accumulator's groups.
    assert!(o.state(&[5]).is_err());
}
