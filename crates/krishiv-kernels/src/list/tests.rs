// SPDX-License-Identifier: Apache-2.0
//! Unit tests for the list kernels: nulls, empty lists, nested lists,
//! large offsets and precondition failures.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, BooleanArray, Int32Array, ListArray};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};

use super::{create, exists_list, filter_list, forall_list, transform_list};

fn field() -> Arc<Field> {
    Arc::new(Field::new("element", DataType::Int32, true))
}

fn list(values: &[Option<Vec<Option<i32>>>]) -> ListArray {
    let mut ints = Vec::new();
    let mut offsets = vec![0_i32];
    let mut nulls = Vec::new();
    for row in values {
        match row {
            None => {
                nulls.push(false);
                offsets.push(i32::try_from(ints.len()).unwrap());
            }
            Some(items) => {
                nulls.push(true);
                for v in items {
                    ints.push(*v);
                }
                offsets.push(i32::try_from(ints.len()).unwrap());
            }
        }
    }
    ListArray::new(
        field(),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(Int32Array::from(ints)),
        Some(NullBuffer::from(nulls)),
    )
}

fn marks(values: Vec<Option<bool>>) -> BooleanArray {
    values.into_iter().collect()
}

fn flat(list: &ListArray) -> Vec<Option<i32>> {
    list.values().as_primitive::<Int32Type>().iter().collect()
}

fn rows(list: &ListArray) -> Vec<Option<Vec<Option<i32>>>> {
    let values = flat(list);
    let offsets = list.value_offsets();
    (0..list.len())
        .map(|i| {
            if list.is_null(i) {
                return None;
            }
            let next = i.checked_add(1).unwrap();
            let (start, stop) = (
                usize::try_from(offsets[i]).unwrap(),
                usize::try_from(offsets[next]).unwrap(),
            );
            Some(values[start..stop].to_vec())
        })
        .collect()
}

#[test]
fn create_interleaves_columns() {
    let a: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]));
    let b: ArrayRef = Arc::new(Int32Array::from(vec![Some(4), Some(5), None]));
    let out = create(&[a, b], 3, true).unwrap();
    assert_eq!(
        rows(&out),
        vec![
            Some(vec![Some(1), Some(4)]),
            Some(vec![None, Some(5)]),
            Some(vec![Some(3), None]),
        ]
    );
}

#[test]
fn create_no_arguments_is_empty() {
    let out = create(&[], 2, false).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out.value_offsets(), &[0, 0, 0]);
}

#[test]
fn create_rejects_a_short_argument() {
    let a: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    assert!(create(&[a], 2, true).is_err());
}

#[test]
fn transform_rebuilds_over_new_values() {
    // `[1, 2], NULL, [], [3]`, doubled to `[2, 4], NULL, [], [6]`.
    let input = list(&[
        Some(vec![Some(1), Some(2)]),
        None,
        Some(vec![]),
        Some(vec![Some(3)]),
    ]);
    let values: ArrayRef = Arc::new(Int32Array::from(vec![2, 4, 6]));
    let out = transform_list(&input, &values, true).unwrap();
    assert_eq!(
        rows(&out),
        vec![
            Some(vec![Some(2), Some(4)]),
            None,
            Some(vec![]),
            Some(vec![Some(6)]),
        ]
    );
    assert!(out.is_null(1));
}

#[test]
fn transform_rejects_short_values() {
    let input = list(&[Some(vec![Some(1)])]);
    let values: ArrayRef = Arc::new(Int32Array::new_null(0));
    assert!(transform_list(&input, &values, true).is_err());
}

#[test]
fn filter_drops_unmarked_elements() {
    // `[1, NULL, 3], NULL, [], [4]`, keeping the first of each pair.
    let input = list(&[
        Some(vec![Some(1), None, Some(3)]),
        None,
        Some(vec![]),
        Some(vec![Some(4)]),
    ]);
    let keep = marks(vec![Some(true), None, Some(false), Some(true)]);
    let out = filter_list(&input, &keep).unwrap();
    assert_eq!(
        rows(&out),
        vec![Some(vec![Some(1)]), None, Some(vec![]), Some(vec![Some(4)]),]
    );
}

#[test]
fn filter_null_marks_drop() {
    let input = list(&[Some(vec![Some(1), Some(2)])]);
    let keep = marks(vec![None, None]);
    let out = filter_list(&input, &keep).unwrap();
    assert_eq!(rows(&out), vec![Some(vec![])]);
}

#[test]
fn filter_rejects_short_marks() {
    let input = list(&[Some(vec![Some(1), Some(2)])]);
    assert!(filter_list(&input, &marks(vec![Some(true)])).is_err());
}

#[test]
fn exists_folds_any_true() {
    let input = list(&[
        Some(vec![Some(1), None]),
        Some(vec![None, None]),
        None,
        Some(vec![]),
    ]);
    // Marks: row 0 has a true, row 1 only nulls, row 3 is empty.
    let keep = marks(vec![Some(false), None, None, None]);
    let out = exists_list(&input, &keep).unwrap();
    let got: Vec<Option<bool>> = out.iter().collect();
    assert_eq!(got, vec![Some(false), Some(false), None, Some(false)]);
    let keep = marks(vec![Some(true), None, None, None]);
    let out = exists_list(&input, &keep).unwrap();
    let got: Vec<Option<bool>> = out.iter().collect();
    assert_eq!(got, vec![Some(true), Some(false), None, Some(false)]);
}

#[test]
fn forall_folds_no_false() {
    let input = list(&[
        Some(vec![Some(1), None]),
        Some(vec![None]),
        None,
        Some(vec![]),
    ]);
    // NULL marks decide nothing: row 0 is all-true, row 1 vacuous.
    let keep = marks(vec![Some(true), None, None]);
    let out = forall_list(&input, &keep).unwrap();
    let got: Vec<Option<bool>> = out.iter().collect();
    assert_eq!(got, vec![Some(true), Some(true), None, Some(true)]);
    let keep = marks(vec![Some(true), Some(false), None]);
    let out = forall_list(&input, &keep).unwrap();
    let got: Vec<Option<bool>> = out.iter().collect();
    assert_eq!(got, vec![Some(false), Some(true), None, Some(true)]);
}

#[test]
fn fold_rejects_short_marks() {
    let input = list(&[Some(vec![Some(1)])]);
    let keep = marks(vec![]);
    assert!(exists_list(&input, &keep).is_err());
    assert!(forall_list(&input, &keep).is_err());
}

#[test]
fn nested_lists_reshape() {
    // `[[1, 2], [3]]`: filtering the single outer row's elements keeps
    // whole inner lists, like any other element.
    let inner = list(&[Some(vec![Some(1), Some(2)]), Some(vec![Some(3)])]);
    let outer = ListArray::new(
        Arc::new(Field::new("element", DataType::List(field()), true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2])),
        Arc::new(inner),
        None,
    );
    let keep = marks(vec![Some(false), Some(true)]);
    let out = filter_list(&outer, &keep).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out.value_offsets(), &[0, 1]);
    let kept: &ListArray = out.values().as_list::<i32>();
    assert_eq!(kept.len(), 1);
    assert_eq!(kept.value_offsets(), &[0, 1]);
    assert_eq!(flat(kept), vec![Some(3)]);
}

#[test]
fn large_offsets_reshape() {
    // 100,000 single-element lists: the reshape is offset arithmetic.
    let n = 100_000usize;
    let values: ArrayRef = Arc::new(Int32Array::from(vec![1; n]));
    let input = ListArray::new(
        field(),
        OffsetBuffer::new(ScalarBuffer::from_iter(0..=i32::try_from(n).unwrap())),
        values,
        None,
    );
    let doubled: ArrayRef = Arc::new(Int32Array::from(vec![2; n]));
    let out = transform_list(&input, &doubled, false).unwrap();
    assert_eq!(out.len(), n);
    assert_eq!(usize::try_from(out.value_offsets()[n]).unwrap(), n);
    let keep = BooleanArray::from(vec![true; n]);
    let out = filter_list(&input, &keep).unwrap();
    assert_eq!(usize::try_from(out.value_offsets()[n]).unwrap(), n);
    let out = exists_list(&input, &keep).unwrap();
    assert!(out.iter().all(|v| v == Some(true)));
    let out = forall_list(&input, &keep).unwrap();
    assert!(out.iter().all(|v| v == Some(true)));
}
