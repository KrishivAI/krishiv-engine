// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch, RecordBatchOptions, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::{ErrorClass, KrishivError};
use proptest::prelude::*;

use super::{Batch, ChangeBatch, Weights};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("name", DataType::Utf8, true),
    ]))
}

/// Rows `ids` with names `n<id>`.
fn data(ids: &[i64]) -> RecordBatch {
    let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap()
}

fn column(w: &[i64]) -> Weights {
    Weights::Column(Int64Array::from(w.to_vec()))
}

fn ids(b: &Batch) -> Vec<i64> {
    b.data()
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

fn assert_internal(r: Result<Batch, KrishivError>) {
    let e = r.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert_eq!(e.sql_state(), Some("XX000"));
}

#[test]
fn new_is_unweighted() {
    let b = Batch::new(data(&[1, 2, 3]));
    assert_eq!(b.num_rows(), 3);
    assert_eq!(b.weights(), &Weights::Const(1));
    assert!(b.is_unweighted());
    assert_eq!(b.schema(), schema());
}

#[test]
fn with_weights_checks_column_length_and_nulls() {
    let b = Batch::with_weights(data(&[1, 2]), column(&[1, -1])).unwrap();
    assert_eq!(b.weights(), &column(&[1, -1]));
    assert!(!b.is_unweighted());

    let neg = Batch::with_weights(data(&[1, 2]), Weights::Const(-1)).unwrap();
    assert!(!neg.is_unweighted());

    assert_internal(Batch::with_weights(data(&[1, 2]), column(&[1])));
    assert_internal(Batch::with_weights(data(&[1, 2]), column(&[1, 1, 1])));
    assert_internal(Batch::with_weights(
        data(&[1, 2]),
        Weights::Column(Int64Array::from(vec![Some(1), None])),
    ));
}

#[test]
fn into_parts_returns_data_and_weights() {
    let (d, w) = Batch::with_weights(data(&[4]), Weights::Const(3))
        .unwrap()
        .into_parts();
    assert_eq!(d, data(&[4]));
    assert_eq!(w, Weights::Const(3));
}

#[test]
fn zero_column_batch_keeps_row_count() {
    let empty = Arc::new(Schema::empty());
    let d = RecordBatch::try_new_with_options(
        Arc::clone(&empty),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(4)),
    )
    .unwrap();
    let b = Batch::with_weights(d, column(&[1, 2, 3, 4])).unwrap();
    assert_eq!(b.num_rows(), 4);

    let f = b
        .filter(&BooleanArray::from(vec![true, false, true, true]))
        .unwrap();
    assert_eq!(f.num_rows(), 3);
    assert_eq!(f.weights(), &column(&[1, 3, 4]));

    let s = b.slice(1, 2).unwrap();
    assert_eq!(s.num_rows(), 2);
    assert_eq!(s.weights(), &column(&[2, 3]));

    let c = Batch::concat(&empty, &[b.clone(), Batch::new(b.data().clone())]).unwrap();
    assert_eq!(c.num_rows(), 8);
    assert_eq!(c.weights(), &column(&[1, 2, 3, 4, 1, 1, 1, 1]));

    let p = Batch::new(data(&[1, 2])).project(&[]).unwrap();
    assert_eq!((p.num_rows(), p.data().num_columns()), (2, 0));
}

#[test]
fn filter_const_weights_stay_const() {
    let b = Batch::with_weights(data(&[1, 2, 3, 4]), Weights::Const(5)).unwrap();
    let f = b
        .filter(&BooleanArray::from(vec![true, false, false, true]))
        .unwrap();
    assert_eq!(ids(&f), [1, 4]);
    assert_eq!(f.weights(), &Weights::Const(5));
}

#[test]
fn filter_column_weights_alongside_rows() {
    let b = Batch::with_weights(data(&[1, 2, 3, 4]), column(&[10, -20, 30, -40])).unwrap();
    let f = b
        .filter(&BooleanArray::from(vec![false, true, true, false]))
        .unwrap();
    assert_eq!(ids(&f), [2, 3]);
    assert_eq!(f.weights(), &column(&[-20, 30]));
}

#[test]
fn filter_treats_null_mask_as_false() {
    let b = Batch::with_weights(data(&[1, 2, 3]), column(&[1, 2, 3])).unwrap();
    let f = b
        .filter(&BooleanArray::from(vec![Some(true), None, Some(true)]))
        .unwrap();
    assert_eq!(ids(&f), [1, 3]);
    assert_eq!(f.weights(), &column(&[1, 3]));
}

#[test]
fn filter_all_and_none() {
    let b = Batch::with_weights(data(&[1, 2]), column(&[7, 8])).unwrap();
    let all = b.filter(&BooleanArray::from(vec![true, true])).unwrap();
    assert_eq!(all, b);
    // Selecting every row shares the input's buffers.
    let (Weights::Column(src), Weights::Column(out)) = (b.weights(), all.weights()) else {
        panic!("expected weight columns");
    };
    assert_eq!(src.values().as_ptr(), out.values().as_ptr());

    let none = b.filter(&BooleanArray::from(vec![false, false])).unwrap();
    assert_eq!(none.num_rows(), 0);
    assert_eq!(none.weights(), &column(&[]));
}

#[test]
fn filter_rejects_mask_of_wrong_length() {
    let b = Batch::new(data(&[1, 2, 3]));
    assert_internal(b.filter(&BooleanArray::from(vec![true, false])));
    assert_internal(b.filter(&BooleanArray::from(vec![true; 4])));
}

#[test]
fn project_keeps_rows_and_weights_without_copying() {
    let b = Batch::with_weights(data(&[1, 2]), column(&[3, -3])).unwrap();
    let p = b.project(&[1, 0, 1]).unwrap();
    assert_eq!(p.data().num_columns(), 3);
    assert_eq!(p.schema().field(0).name(), "name");
    assert_eq!(p.weights(), b.weights());
    assert!(Arc::ptr_eq(p.data().column(1), b.data().column(0)));

    let c = Batch::new(data(&[1])).project(&[0]).unwrap();
    assert!(c.is_unweighted());

    assert_internal(b.project(&[2]));
}

#[test]
fn slice_keeps_weights_aligned() {
    let b = Batch::with_weights(data(&[1, 2, 3, 4]), column(&[1, 2, 3, 4])).unwrap();
    let s = b.slice(1, 2).unwrap();
    assert_eq!(ids(&s), [2, 3]);
    assert_eq!(s.weights(), &column(&[2, 3]));

    let c = Batch::with_weights(data(&[1, 2, 3]), Weights::Const(-1)).unwrap();
    assert_eq!(c.slice(3, 0).unwrap().weights(), &Weights::Const(-1));

    assert_internal(b.slice(3, 2));
    assert_internal(b.slice(5, 0));
    assert_internal(b.slice(usize::MAX, 2));
}

#[test]
fn concat_equal_consts_stay_const() {
    let a = Batch::new(data(&[1, 2]));
    let b = Batch::new(data(&[3]));
    let c = Batch::concat(&schema(), &[a, b]).unwrap();
    assert_eq!(ids(&c), [1, 2, 3]);
    assert!(c.is_unweighted());

    let neg = Batch::with_weights(data(&[1]), Weights::Const(-2)).unwrap();
    let c = Batch::concat(&schema(), &[neg.clone(), neg]).unwrap();
    assert_eq!(c.weights(), &Weights::Const(-2));
}

#[test]
fn concat_mixed_weights_become_a_column() {
    let one = Batch::new(data(&[1, 2]));
    let minus = Batch::with_weights(data(&[3]), Weights::Const(-1)).unwrap();
    let col = Batch::with_weights(data(&[4, 5]), column(&[7, -7])).unwrap();
    let c = Batch::concat(&schema(), &[one, minus, col]).unwrap();
    assert_eq!(ids(&c), [1, 2, 3, 4, 5]);
    assert_eq!(c.weights(), &column(&[1, 1, -1, 7, -7]));

    let col2 = Batch::with_weights(data(&[6]), column(&[1])).unwrap();
    let c = Batch::concat(&schema(), &[col2.clone(), col2]).unwrap();
    // Equal column values are not re-detected as a constant.
    assert_eq!(c.weights(), &column(&[1, 1]));
}

#[test]
fn concat_ignores_empty_batches_when_keeping_a_constant() {
    let empty_col = Batch::with_weights(data(&[]), column(&[])).unwrap();
    let c = Batch::concat(
        &schema(),
        &[Batch::new(data(&[1])), empty_col, Batch::new(data(&[2, 3]))],
    )
    .unwrap();
    assert_eq!(ids(&c), [1, 2, 3]);
    assert!(c.is_unweighted());

    let empty_neg = Batch::with_weights(data(&[]), Weights::Const(-1)).unwrap();
    let c = Batch::concat(&schema(), &[empty_neg.clone(), Batch::new(data(&[4]))]).unwrap();
    assert!(c.is_unweighted());

    // No rows at all: the first batch's constant.
    let c = Batch::concat(&schema(), &[empty_neg.clone(), empty_neg]).unwrap();
    assert_eq!((c.num_rows(), c.weights()), (0, &Weights::Const(-1)));
}

#[test]
fn concat_empty_and_single() {
    let e = Batch::concat(&schema(), &[]).unwrap();
    assert_eq!(e.num_rows(), 0);
    assert!(e.is_unweighted());
    assert_eq!(e.schema(), schema());

    let b = Batch::with_weights(data(&[1, 2]), column(&[2, 3])).unwrap();
    let c = Batch::concat(&schema(), std::slice::from_ref(&b)).unwrap();
    assert_eq!(c, b);
    assert!(Arc::ptr_eq(c.data().column(0), b.data().column(0)));
}

#[test]
fn concat_rejects_other_fields() {
    let other = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let d = RecordBatch::try_new(
        Arc::clone(&other),
        vec![Arc::new(Int64Array::from(vec![1]))],
    )
    .unwrap();
    assert_internal(Batch::concat(
        &schema(),
        &[Batch::new(data(&[1])), Batch::new(d.clone())],
    ));
    assert_internal(Batch::concat(&schema(), &[Batch::new(d)]));
}

#[test]
fn change_batch_is_batch() {
    let c: ChangeBatch = Batch::new(data(&[1]));
    assert!(c.is_unweighted());
}

/// Row ids with a parallel mask (nulls included).
fn rows_and_mask() -> impl Strategy<Value = (Vec<i64>, Vec<Option<bool>>)> {
    (0..300usize).prop_flat_map(|n| {
        (
            proptest::collection::vec(any::<i64>(), n),
            proptest::collection::vec(proptest::option::of(any::<bool>()), n),
        )
    })
}

proptest! {
    /// Filtering never materialises a weight column for constant weights,
    /// and `Const(1)` stays on the insert-only fast path.
    #[test]
    fn filter_const_never_materialises_weights(
        (rows, mask) in rows_and_mask(),
        c in prop_oneof![Just(1i64), any::<i64>()],
    ) {
        let b = Batch::with_weights(data(&rows), Weights::Const(c)).unwrap();
        let mask = BooleanArray::from(mask);
        let f = b.filter(&mask).unwrap();
        prop_assert_eq!(f.weights(), &Weights::Const(c));
        prop_assert_eq!(f.is_unweighted(), c == 1);
        prop_assert_eq!(f.num_rows(), mask.true_count());
        // Chained ops on the result keep the constant too.
        let half = f.num_rows() / 2;
        let g = f.slice(0, half).unwrap().project(&[0]).unwrap();
        let h = Batch::concat(&g.schema(), &[g.clone(), g]).unwrap();
        prop_assert_eq!(h.weights(), &Weights::Const(c));
    }

    /// A weight column is filtered with the same mask as the rows.
    #[test]
    fn filter_column_matches_row_filter((rows, mask) in rows_and_mask()) {
        let w: Vec<i64> = rows.iter().map(|r| r.wrapping_mul(3)).collect();
        let b = Batch::with_weights(data(&rows), column(&w)).unwrap();
        let f = b.filter(&BooleanArray::from(mask.clone())).unwrap();
        let keep = |v: &[i64]| -> Vec<i64> {
            v.iter().zip(&mask).filter(|(_, m)| **m == Some(true)).map(|(x, _)| *x).collect()
        };
        prop_assert_eq!(ids(&f), keep(&rows));
        prop_assert_eq!(f.weights(), &column(&keep(&w)));
    }
}
