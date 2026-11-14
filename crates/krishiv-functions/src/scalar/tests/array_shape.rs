// SPDX-License-Identifier: Apache-2.0
//! `sequence` over narrow integers and the argument shapes `flatten`,
//! `slice` and `sequence` refuse.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int16Type};
use arrow_array::{Array, ArrayRef, Int8Array, Int16Array};
use arrow_schema::DataType;

use super::{BindOptions, bind_scalar, bind_with, col};

fn values<T: arrow_array::types::ArrowPrimitiveType>(a: &ArrayRef, row: usize) -> Vec<T::Native> {
    a.as_list::<i32>()
        .value(row)
        .as_primitive::<T>()
        .values()
        .to_vec()
}

#[test]
fn sequence_over_tinyint_and_smallint_keeps_the_element_type() {
    let opts = BindOptions::new();
    let a: ArrayRef = Arc::new(Int8Array::from(vec![Some(1), Some(5), None]));
    let b: ArrayRef = Arc::new(Int8Array::from(vec![Some(4), Some(3), Some(2)]));
    let f = bind_with("sequence", &[&a, &b], &opts);
    let out = f.invoke(&[col(&a), col(&b)], 3).unwrap();
    assert_eq!(values::<Int8Type>(&out, 0), vec![1, 2, 3, 4]);
    assert_eq!(values::<Int8Type>(&out, 1), vec![5, 4, 3]);
    assert!(out.is_null(2));

    let a: ArrayRef = Arc::new(Int16Array::from(vec![Some(-2)]));
    let b: ArrayRef = Arc::new(Int16Array::from(vec![Some(2)]));
    let s: ArrayRef = Arc::new(Int16Array::from(vec![Some(2)]));
    let f = bind_with("sequence", &[&a, &b, &s], &opts);
    let out = f.invoke(&[col(&a), col(&b), col(&s)], 1).unwrap();
    assert_eq!(values::<Int16Type>(&out, 0), vec![-2, 0, 2]);
}

#[test]
fn unsupported_shapes_do_not_bind() {
    let opts = BindOptions::new();
    let int = DataType::Int32;
    let list = DataType::new_list(int.clone(), true);
    assert!(bind_scalar("flatten", &[list.into()], &opts).is_err());
    assert!(bind_scalar("flatten", &[int.clone().into()], &opts).is_err());
    assert!(
        bind_scalar(
            "sequence",
            &[DataType::Utf8.into(), DataType::Utf8.into()],
            &opts
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "slice",
            &[int.clone().into(), int.clone().into(), int.into()],
            &opts
        )
        .is_err()
    );
}

#[test]
fn reverse_of_an_array_reverses_the_elements_and_keeps_nulls_and_empty_arrays() {
    use arrow_array::builder::{Int32Builder, ListBuilder};
    let mut b = ListBuilder::new(Int32Builder::new());
    for row in [
        Some(vec![Some(2), Some(1), Some(4), Some(3)]),
        Some(vec![Some(1), None, Some(3)]),
        Some(vec![]),
        None,
    ] {
        match row {
            Some(v) => {
                for x in v {
                    b.values().append_option(x);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    let list: ArrayRef = Arc::new(b.finish());
    let f = bind_with("reverse", &[&list], &BindOptions::new());
    assert_eq!(f.return_type(), list.data_type());
    let out = f.invoke(&[col(&list)], 4).unwrap();
    let out = out.as_list::<i32>();
    let ints = |row: usize| -> Vec<Option<i32>> {
        out.value(row)
            .as_primitive::<arrow_array::types::Int32Type>()
            .iter()
            .collect()
    };
    assert_eq!(ints(0), [Some(3), Some(4), Some(1), Some(2)]);
    assert_eq!(ints(1), [Some(3), None, Some(1)]);
    assert!(ints(2).is_empty());
    assert!(out.is_null(3));
}
