// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, BinaryArray, BinaryViewArray, Int64Array, ListArray};
use krishiv_functions::sketch::theta;

use super::*;

fn estimate_of(a: &ArrayRef, row: usize) -> i64 {
    let v = a.as_any().downcast_ref::<BinaryViewArray>().expect("view");
    theta::estimate(&theta::from_bytes(v.value(row), "t").expect("sketch"))
}

fn longs(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

#[test]
fn a_sketch_counts_distinct_values_per_group_and_skips_nulls() {
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    acc.resize(2);
    let arg = longs(&[Some(1), Some(1), Some(2), None, Some(5), Some(6)]);
    acc.update(&[0, 0, 0, 0, 1, 1], &arg, |i| i != 5).unwrap();
    let out = acc.evaluate(&[0, 1]).unwrap();
    assert_eq!(estimate_of(&out, 0), 2);
    assert_eq!(estimate_of(&out, 1), 1, "the filtered row is dropped");
}

#[test]
fn array_values_hash_as_one_value_and_an_empty_array_is_skipped() {
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    acc.resize(1);
    let list = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(1), Some(2)]),
        Some(vec![Some(1), Some(2)]),
        Some(vec![]),
        None,
        Some(vec![Some(3)]),
    ]);
    acc.update(&[0; 5], &(Arc::new(list) as ArrayRef), |_| true)
        .unwrap();
    assert_eq!(estimate_of(&acc.evaluate(&[0]).unwrap(), 0), 2);
}

#[test]
fn partials_merge_through_their_bytes() {
    let mut a = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    let mut b = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    a.resize(1);
    b.resize(1);
    a.update(&[0, 0, 0], &longs(&[Some(1), Some(2), Some(3)]), |_| true)
        .unwrap();
    b.update(&[0, 0], &longs(&[Some(3), Some(4)]), |_| true)
        .unwrap();
    let mut fin = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    fin.resize(1);
    for p in [&a, &b] {
        let state = p.state(&[0]).unwrap().remove(0);
        let state = state
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .clone();
        fin.merge(&[0], &state).unwrap();
    }
    assert_eq!(estimate_of(&fin.evaluate(&[0]).unwrap(), 0), 4);
}

#[test]
fn a_raw_update_after_a_merge_goes_into_the_union_the_merge_made() {
    // A table that widens while input arrives merges its sketches into each other (`RowTable::widen`), which leaves
    // every group a union; the next raw rows of that group must still count (and array rows too).
    let mut a = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    a.resize(1);
    a.update(&[0, 0], &longs(&[Some(1), Some(2)]), |_| true)
        .unwrap();
    let state = a.state(&[0]).unwrap().remove(0);
    let state = state.as_any().downcast_ref::<BinaryArray>().unwrap();
    let mut b = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    b.resize(1);
    b.merge(&[0], state).unwrap();
    b.update(&[0, 0, 0], &longs(&[Some(2), Some(3), None]), |_| true)
        .unwrap();
    let list =
        ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![Some(7), Some(8)])]);
    b.update(&[0], &(Arc::new(list) as ArrayRef), |_| true)
        .unwrap();
    // {1, 2} merged, then 2 (again), 3 and the array {7, 8} as one value: 4 distinct.
    assert_eq!(estimate_of(&b.evaluate(&[0]).unwrap(), 0), 4);
    // A second batch after the first post-merge update: a pending sketch folded twice or dropped shows here.
    b.update(&[0, 0], &longs(&[Some(3), Some(9)]), |_| true)
        .unwrap();
    assert_eq!(estimate_of(&b.evaluate(&[0]).unwrap(), 0), 5);
    // The state taken while raw rows are pending holds them, and a merge after them folds them in first.
    let state = b.state(&[0]).unwrap().remove(0);
    let state = state.as_any().downcast_ref::<BinaryArray>().unwrap();
    assert_eq!(
        theta::estimate(&theta::from_bytes(state.value(0), "t").unwrap()),
        5
    );
    let mut other = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    other.resize(1);
    other
        .update(&[0, 0], &longs(&[Some(100), Some(101)]), |_| true)
        .unwrap();
    let other_state = other.state(&[0]).unwrap().remove(0);
    let other_state = other_state.as_any().downcast_ref::<BinaryArray>().unwrap();
    b.merge(&[0], other_state).unwrap();
    b.update(&[0], &longs(&[Some(102)]), |_| true).unwrap();
    assert_eq!(estimate_of(&b.evaluate(&[0]).unwrap(), 0), 8);
    // Two groups in one batch keep separate pending sketches.
    let mut c = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    c.resize(2);
    c.update(&[0, 1], &longs(&[Some(1), Some(1)]), |_| true)
        .unwrap();
    let state = c.state(&[0, 1]).unwrap().remove(0);
    let state = state.as_any().downcast_ref::<BinaryArray>().unwrap();
    let mut d = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    d.resize(2);
    d.merge(&[0, 1], state).unwrap();
    d.update(
        &[0, 1, 1, 1],
        &longs(&[Some(5), Some(5), Some(6), Some(7)]),
        |_| true,
    )
    .unwrap();
    assert_eq!(estimate_of(&d.evaluate(&[0]).unwrap(), 0), 2);
    assert_eq!(estimate_of(&d.evaluate(&[1]).unwrap(), 0), 4);
}

#[test]
fn an_intersection_with_no_input_is_the_empty_sketch_and_an_empty_state() {
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Intersection).unwrap();
    acc.resize(1);
    let state = acc.state(&[0]).unwrap().remove(0);
    let state = state.as_any().downcast_ref::<BinaryArray>().unwrap();
    assert!(state.value(0).is_empty());
    assert_eq!(estimate_of(&acc.evaluate(&[0]).unwrap(), 0), 0);
    // A merge of that empty state changes nothing.
    acc.merge(&[0], state).unwrap();
    assert!(
        acc.state(&[0]).unwrap()[0]
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .value(0)
            .is_empty()
    );
}

#[test]
fn union_and_intersection_aggregates_fold_sketches() {
    let sketch = |values: &[i64]| {
        let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
        acc.resize(1);
        let arg = longs(&values.iter().map(|v| Some(*v)).collect::<Vec<_>>());
        acc.update(&vec![0; values.len()], &arg, |_| true).unwrap();
        let out = acc.evaluate(&[0]).unwrap();
        out.as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap()
            .value(0)
            .to_vec()
    };
    let col: ArrayRef = Arc::new(BinaryArray::from_iter(vec![
        Some(sketch(&[1, 2, 3])),
        Some(sketch(&[2, 3, 4])),
        None,
    ]));
    let mut u = ThetaSketchAcc::new(ThetaSketchOp::Union(12)).unwrap();
    u.resize(1);
    u.update(&[0, 0, 0], &col, |_| true).unwrap();
    assert_eq!(estimate_of(&u.evaluate(&[0]).unwrap(), 0), 4);
    let mut i = ThetaSketchAcc::new(ThetaSketchOp::Intersection).unwrap();
    i.resize(1);
    i.update(&[0, 0, 0], &col, |_| true).unwrap();
    assert_eq!(estimate_of(&i.evaluate(&[0]).unwrap(), 0), 2);
}

#[test]
fn a_bad_buffer_is_the_theta_error_and_an_out_of_range_size_is_refused() {
    let mut u = ThetaSketchAcc::new(ThetaSketchOp::Union(12)).unwrap();
    u.resize(1);
    let col: ArrayRef = Arc::new(BinaryArray::from_iter_values([b"junk".to_vec()]));
    let e = u.update(&[0], &col, |_| true).unwrap_err().to_string();
    assert!(e.contains("THETA_INVALID_INPUT_SKETCH_BUFFER"), "{e}");
    for op in [ThetaSketchOp::Sketch(3), ThetaSketchOp::Union(27)] {
        let e = ThetaSketchAcc::new(op).err().map(|e| e.to_string());
        assert!(
            e.as_deref()
                .is_some_and(|e| e.contains("SKETCH_INVALID_LG_NOM_ENTRIES")),
            "{e:?}"
        );
    }
}

/// Feeds `n` values to group 0, one row per batch, and checks every batch against its projection.
fn one_row_batches_stay_under_the_projection(lg: u8, groups: u32, n: i64) {
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(lg)).unwrap();
    let width = groups as usize;
    acc.resize(width);
    for v in 0..n {
        let arg = longs(&vec![Some(v); width]);
        let ids: Vec<u32> = (0..groups).collect();
        let projected = acc.projected(0, width, false, 0);
        acc.update(&ids, &arg, |_| true).unwrap();
        assert!(
            acc.size() <= projected,
            "lg {lg} row {v}: {} > {projected}",
            acc.size()
        );
    }
}

#[test]
fn a_row_that_crosses_the_resize_edge_is_projected() {
    one_row_batches_stay_under_the_projection(12, 1, 3_000);
    one_row_batches_stay_under_the_projection(5, 3, 200);
}

#[test]
fn new_groups_with_many_rows_are_projected() {
    for rows_per_group in [1_u32, 5, 65, 300] {
        let groups = 7_u32;
        let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
        let before = acc.size();
        let n = groups * rows_per_group;
        let projected = acc.projected(groups as usize, n as usize, false, 0);
        assert!(projected >= before);
        acc.resize(groups as usize);
        let ids: Vec<u32> = (0..n).map(|i| i % groups).collect();
        let arg = longs(&(0..i64::from(n)).map(Some).collect::<Vec<_>>());
        acc.update(&ids, &arg, |_| true).unwrap();
        assert!(
            acc.size() <= projected,
            "{rows_per_group}: {} > {projected}",
            acc.size()
        );
    }
}

#[test]
fn an_intersection_is_projected_by_its_input_not_by_the_default_size() {
    // Sketches built at lgNomEntries 16 in estimation mode: an intersection's table follows its first input.
    let mut big = theta::new_sketch(16, "theta_sketch_agg").unwrap();
    for v in 0..300_000_i64 {
        theta::update_long(&mut big, v);
    }
    let bytes = theta::to_bytes(&theta::compact(&mut big));
    let col: ArrayRef = Arc::new(BinaryArray::from_iter_values((0..4).map(|_| bytes.clone())));
    let state_bytes = bytes.len() * 4;
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Intersection).unwrap();
    acc.resize(4);
    let projected = acc.projected(0, 4, true, state_bytes);
    acc.update(&[0, 1, 2, 3], &col, |_| true).unwrap();
    assert!(acc.size() <= projected, "{} > {projected}", acc.size());
    assert!(
        projected > 4 * (8 << 17),
        "the 64 KiB default table is not the bound"
    );
}

#[test]
fn state_batches_over_two_gib_are_an_error_not_a_wrapped_array() {
    assert!(binary_array(vec![vec![0_u8; 8], vec![1_u8; 8]]).is_ok());
    let max = i32::MAX as usize;
    // Lengths only: nothing of that size is allocated.
    assert!(check_total_len([max].into_iter()).is_ok());
    assert!(check_total_len([max - 1, 1].into_iter()).is_ok());
    assert!(check_total_len([max, 1].into_iter()).is_err());
    assert!(check_total_len([usize::MAX, 1].into_iter()).is_err());
}

/// A raw sketch column (the first stage of `theta_union_agg(s)` / `theta_intersection_agg(s)`, not partial
/// states) is charged by the bytes of its sketches, through the `Accumulator` the row table calls.
#[test]
fn a_raw_sketch_column_is_projected_by_its_bytes_for_union_and_intersection() {
    use super::super::accumulator::Accumulator;
    use super::super::{AggregateExpr, AggregateFunction};
    use arrow_schema::DataType;

    let mut big = theta::new_sketch(20, "theta_sketch_agg").unwrap();
    for v in 0..1_000_000_i64 {
        theta::update_long(&mut big, v);
    }
    let bytes = theta::to_bytes(&theta::compact(&mut big));
    let col: ArrayRef = Arc::new(BinaryArray::from_iter_values((0..4).map(|_| bytes.clone())));
    for function in [
        AggregateFunction::ThetaUnionAgg { lg_nom: 12 },
        AggregateFunction::ThetaIntersectionAgg,
    ] {
        let expr = AggregateExpr::new(function.clone(), 0, DataType::Binary, "t".to_owned());
        let mut acc = Accumulator::new(&expr, false).unwrap();
        acc.resize(1);
        let projected = acc.projected(0, Some(&col), None, false);
        let overhead_only = acc.projected(0, None, None, false);
        assert!(
            projected > 10 * overhead_only,
            "{function:?}: the sketch bytes are not charged ({projected} vs {overhead_only})"
        );
        if matches!(function, AggregateFunction::ThetaIntersectionAgg) {
            // Not capped by lgNomEntries: follows the input.
            assert!(projected > 4 * bytes.len(), "{function:?}: {projected}");
        }
    }
}

/// A union's fold decodes its whole input (`theta::from_bytes`, eight bytes a retained hash) whatever the union's own
/// lgNomEntries: the projection covers the decoded input, not a table capped at the union's size.
#[test]
fn a_union_fold_is_projected_by_the_decoded_input_not_by_its_own_table() {
    let mut big = theta::new_sketch(20, "theta_sketch_agg").unwrap();
    for v in 0..1_000_000_i64 {
        theta::update_long(&mut big, v);
    }
    let bytes = theta::to_bytes(&theta::compact(&mut big));
    let decoded = theta::from_bytes(&bytes, "theta_union_agg")
        .unwrap()
        .num_retained()
        * 8;
    for (op, states) in [
        (ThetaSketchOp::Union(12), false),
        (ThetaSketchOp::Union(12), true),
        (ThetaSketchOp::Sketch(12), true),
    ] {
        let mut acc = ThetaSketchAcc::new(op).unwrap();
        acc.resize(1);
        let room = acc.projected(0, 1, states, bytes.len()) - acc.size();
        assert!(
            room >= decoded,
            "{op:?} states {states}: {room} < {decoded} decoded bytes"
        );
    }
}

/// A sketch at its largest table holding `2^lgNom` hashes rebuilds on its next distinct hash (in `recharge`'s trim):
/// the rebuild holds the retained entries, up to one more table, beside the table. The projection reserves that
/// scratch whatever room the table has left.
#[test]
fn a_raw_batch_that_can_rebuild_a_full_sketch_reserves_the_rebuild_scratch() {
    let lg = 10_u8;
    let table = 8_usize << (lg + 1);
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(lg)).unwrap();
    acc.resize(1);
    let values: Vec<Option<i64>> = (0..20_000).map(Some).collect();
    acc.update(&vec![0; values.len()], &longs(&values), |_| true)
        .unwrap();
    let Slot::Updatable(s) = &acc.slots[0] else {
        panic!("a sketch group")
    };
    assert_eq!(s.num_retained(), 1 << lg, "trimmed to its nominal size");
    assert!(s.estimated_size() >= table, "at its largest table");
    let room = acc.projected(0, 1, false, 0) - acc.size();
    assert!(room >= table, "{room} < {table} rebuild bytes");
}

/// A batch that cannot take any sketch past `2^lgNom` hashes cannot rebuild one, so it reserves no rebuild scratch.
#[test]
fn a_raw_batch_that_cannot_rebuild_reserves_no_rebuild_scratch() {
    let lg = 20_u8;
    let table = 8_usize << (lg + 1);
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(lg)).unwrap();
    acc.resize(1);
    acc.update(&[0, 0], &longs(&[Some(1), Some(2)]), |_| true)
        .unwrap();
    let room = acc.projected(0, 1, false, 0) - acc.size();
    assert!(
        room < table / 8,
        "{room} bytes for one row of a small sketch"
    );
}

/// `state_bound` bounds the serialized state of every slot kind, so a batch cut by it never overflows its column.
#[test]
fn the_state_bound_covers_the_serialized_state() {
    let values: Vec<Option<i64>> = (0..50_000).map(Some).collect();
    let mut a = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    a.resize(3);
    let ids: Vec<u32> = (0..values.len()).map(|i| u32::from(i % 7 == 0)).collect();
    a.update(&ids, &longs(&values), |_| true).unwrap();
    let state = a.state(&[0]).unwrap().remove(0);
    let state = state.as_any().downcast_ref::<BinaryArray>().unwrap();
    // Group 2 becomes a union with pending raw rows.
    a.merge(&[2], state).unwrap();
    a.update(&[2, 2], &longs(&[Some(-1), Some(-2)]), |_| true)
        .unwrap();
    // The union beside raw rows is rebuilt while the state is read: its bound adds that scratch (three charges),
    // and a group without pending rows stays at one.
    assert!(matches!(a.slots[2], Slot::Union(_, Some(_))));
    assert!(
        a.state_bound(2) >= 3 * a.charges[2],
        "scratch of the pending union"
    );
    assert!(
        a.state_bound(0) < 2 * a.charges[0] + 1024,
        "no scratch without pending rows"
    );
    let mut i = ThetaSketchAcc::new(ThetaSketchOp::Intersection).unwrap();
    i.resize(2);
    i.merge(&[0], state).unwrap();
    for (acc, groups) in [(&a, 3_u32), (&i, 2)] {
        for g in 0..groups {
            let bytes = acc.bytes(g).unwrap().len();
            assert!(
                acc.state_bound(g) >= bytes,
                "group {g}: {} < {bytes}",
                acc.state_bound(g)
            );
        }
    }
}

/// A union at its largest table can rebuild on any fold, however small the input: the projection of a state batch
/// reserves one table of rebuild scratch beside the decoded input.
#[test]
fn a_fold_into_a_full_union_reserves_the_rebuild_scratch() {
    let lg = 10_u8;
    let table = 8_usize << (lg + 1);
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Union(lg)).unwrap();
    acc.resize(1);
    for part in 0..3_i64 {
        let mut s = theta::new_sketch(16, "theta_sketch_agg").unwrap();
        for v in 0..20_000_i64 {
            theta::update_long(&mut s, part * 1_000_000 + v);
        }
        let bytes = theta::to_bytes(&theta::compact(&mut s));
        let col: ArrayRef = Arc::new(BinaryArray::from_iter_values([bytes]));
        acc.update(&[0], &col, |_| true).unwrap();
    }
    assert!(acc.peak >= table, "the union is at its largest table");
    let mut one = theta::new_sketch(lg.into(), "theta_sketch_agg").unwrap();
    theta::update_long(&mut one, -1);
    let small = theta::to_bytes(&theta::compact(&mut one)).len();
    let room = acc.projected(0, 1, false, small) - acc.size();
    assert!(room >= table, "{room} < {table} rebuild bytes");
}

/// A batch of partial states is cut by the states' bound bytes (at least one group), and an aggregate with no
/// large state takes every group.
#[test]
fn a_state_batch_is_cut_by_its_bound_bytes() {
    use super::super::accumulator::{Accumulator, state_prefix};
    use super::super::{AggregateExpr, AggregateFunction};
    use arrow_schema::DataType;

    let expr = AggregateExpr::new(
        AggregateFunction::ThetaSketchAgg { lg_nom: 12 },
        0,
        DataType::Int64,
        "t".to_owned(),
    );
    let mut acc = Accumulator::new(&expr, false).unwrap();
    acc.resize(4);
    let values: Vec<Option<i64>> = (0..40_000).map(Some).collect();
    let groups: Vec<u32> = (0..values.len())
        .map(|i| u32::try_from(i % 4).unwrap_or(0))
        .collect();
    let arg = longs(&values);
    acc.update(&groups, &[arg], &krishiv_types::batch::Weights::Const(1))
        .unwrap();
    let each = acc.state_bound(0);
    assert!(each > 8 << 12, "a full sketch: {each}");
    let ids = [0, 1, 2, 3];
    let mut used = 0;
    assert_eq!(
        state_prefix(
            std::slice::from_ref(&acc),
            &ids,
            &mut used,
            2 * each + each / 2
        ),
        2
    );
    assert_eq!(used, 2 * each);
    assert_eq!(
        state_prefix(std::slice::from_ref(&acc), &ids, &mut 0, 0),
        1,
        "at least one"
    );
    assert_eq!(
        state_prefix(std::slice::from_ref(&acc), &ids, &mut 1, 0),
        0,
        "a batch already full"
    );
    let count = AggregateExpr::new(AggregateFunction::Count, 0, DataType::Int64, "c".to_owned());
    let count = Accumulator::new(&count, false).unwrap();
    assert_eq!(state_prefix(&[count], &ids, &mut 0, 0), 4);
}
