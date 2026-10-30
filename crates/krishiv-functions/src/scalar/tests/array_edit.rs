// SPDX-License-Identifier: Apache-2.0
//! `array_append`, `array_prepend`, `array_insert` and `array_compact`,
//! checked against Spark 4.2's `array.sql` goldens and a Spark 4.2.0 run
//! (the session golden run covers the SQL forms).

use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, Int32Array};
use arrow_schema::{DataType, Field};

use super::{BindArg, BindOptions, bind_scalar, bind_with, col};

const ABC: &[Option<i32>] = &[Some(2), Some(3), None, Some(4)];

fn lists(rows: &[Option<&[Option<i32>]>]) -> ArrayRef {
    let mut b = ListBuilder::new(Int32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        DataType::Int32,
        true,
    )));
    for row in rows {
        match row {
            Some(values) => {
                for v in *values {
                    b.values().append_option(*v);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

fn ints(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(v.to_vec()))
}

fn run(name: &str, args: &[&ArrayRef]) -> Result<ArrayRef, String> {
    let f = bind_with(name, args, &BindOptions::new());
    f.invoke(
        &args.iter().map(|a| col(a)).collect::<Vec<_>>(),
        args[0].len(),
    )
    .map_err(|e| format!("{} {e}", e.error_class().name()))
}

/// The rows of a list column, `None` for a NULL array.
fn rows(a: &ArrayRef) -> Vec<Option<Vec<Option<i32>>>> {
    let l = a.as_list::<i32>();
    (0..l.len())
        .map(|r| (!l.is_null(r)).then(|| l.value(r).as_primitive::<Int32Type>().iter().collect()))
        .collect()
}

#[test]
fn append_and_prepend_add_the_item_even_when_null() {
    let a = lists(&[Some(&[Some(1), Some(2)]), None, Some(&[])]);
    let item = ints(&[Some(9), Some(9), None]);
    let out = run("array_append", &[&a, &item]).unwrap();
    assert_eq!(
        rows(&out),
        [
            Some(vec![Some(1), Some(2), Some(9)]),
            None,
            Some(vec![None])
        ]
    );
    let out = run("array_prepend", &[&a, &item]).unwrap();
    assert_eq!(
        rows(&out),
        [
            Some(vec![Some(9), Some(1), Some(2)]),
            None,
            Some(vec![None])
        ]
    );
}

#[test]
fn the_result_element_may_hold_null_when_the_array_or_item_may() {
    let non_null = DataType::List(Arc::new(Field::new("element", DataType::Int32, false)));
    let ty = |item: BindArg| {
        bind_scalar(
            "array_append",
            &[non_null.clone().into(), item],
            &BindOptions::new(),
        )
        .unwrap()
        .return_type()
        .clone()
    };
    let element_nullable = |t: DataType| matches!(t, DataType::List(f) if f.is_nullable());
    assert!(!element_nullable(ty(BindArg::Column {
        ty: DataType::Int32,
        nullable: false
    })));
    assert!(element_nullable(ty(BindArg::from(DataType::Int32))));
}

#[test]
fn insert_positions_follow_spark() {
    let a = lists(&[Some(ABC); 8]);
    let item = ints(&[Some(1); 8]);
    let pos = ints(&[
        Some(1),
        Some(3),
        Some(5),
        Some(7),
        Some(-1),
        Some(-2),
        Some(-5),
        Some(-6),
    ]);
    let out = run("array_insert", &[&a, &pos, &item]).unwrap();
    let n = None;
    let want = [
        vec![Some(1), Some(2), Some(3), n, Some(4)],
        vec![Some(2), Some(3), Some(1), n, Some(4)],
        vec![Some(2), Some(3), n, Some(4), Some(1)],
        vec![Some(2), Some(3), n, Some(4), n, n, Some(1)],
        vec![Some(2), Some(3), n, Some(4), Some(1)],
        vec![Some(2), Some(3), n, Some(1), Some(4)],
        vec![Some(1), Some(2), Some(3), n, Some(4)],
        vec![Some(1), n, Some(2), Some(3), n, Some(4)],
    ];
    assert_eq!(rows(&out), want.map(Some));
}

#[test]
fn insert_with_null_array_or_position_is_null_and_zero_is_refused() {
    let a = lists(&[None, Some(ABC)]);
    let item = ints(&[Some(1), Some(1)]);
    let out = run("array_insert", &[&a, &ints(&[Some(1), None]), &item]).unwrap();
    assert_eq!(rows(&out), [None, None]);
    let err = run("array_insert", &[&a, &ints(&[Some(0), Some(0)]), &item]).unwrap_err();
    assert!(err.starts_with("INVALID_INDEX_OF_ZERO"), "{err}");
}

#[test]
fn compact_drops_nulls_and_keeps_null_arrays() {
    let a = lists(&[Some(ABC), None, Some(&[None, None]), Some(&[])]);
    let out = run("array_compact", &[&a]).unwrap();
    assert_eq!(
        rows(&out),
        [
            Some(vec![Some(2), Some(3), Some(4)]),
            None,
            Some(vec![]),
            Some(vec![])
        ]
    );
    assert!(matches!(out.data_type(), DataType::List(f) if !f.is_nullable()));
    // An all-NULL untyped child (`array(NULL)`) compacts to empty.
    let nulls: ArrayRef = arrow_array::new_null_array(&DataType::Null, 2);
    let offsets = arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![0, 1, 2]));
    let list: ArrayRef = Arc::new(arrow_array::ListArray::new(
        Arc::new(Field::new("element", DataType::Null, true)),
        offsets,
        nulls,
        None,
    ));
    let out = run("array_compact", &[&list]).unwrap();
    assert_eq!(out.as_list::<i32>().value_length(0), 0);
}

#[test]
fn only_matching_element_types_bind() {
    let ints_t = DataType::List(Arc::new(Field::new("element", DataType::Int32, true)));
    let opts = BindOptions::new();
    assert!(
        bind_scalar(
            "array_append",
            &[ints_t.clone().into(), DataType::Utf8.into()],
            &opts
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "array_append",
            &[DataType::Int32.into(), DataType::Int32.into()],
            &opts
        )
        .is_err()
    );
    assert!(
        bind_scalar(
            "array_insert",
            &[
                ints_t.into(),
                DataType::Int64.into(),
                DataType::Int32.into()
            ],
            &opts
        )
        .is_err()
    );
    assert!(bind_scalar("array_compact", &[DataType::Int32.into()], &opts).is_err());
}

/// A position that would pad past the size limit fails before any padding is
/// built (a 2^31 position would otherwise allocate gigabytes).
#[test]
fn insert_past_the_size_limit_is_refused_without_allocating() {
    let a = lists(&[Some(&[Some(1)])]);
    for position in [i32::MAX, i32::MIN, 2_147_483_633] {
        let err = run(
            "array_insert",
            &[&a, &ints(&[Some(position)]), &ints(&[Some(0)])],
        )
        .expect_err("past the limit");
        assert!(
            err.starts_with("COLLECTION_SIZE_LIMIT_EXCEEDED.FUNCTION"),
            "{position}: {err}"
        );
    }
    // The largest position that still fits is not refused by the size check,
    // so only the cheap boundary just under the limit is exercised here.
    let ok = run("array_insert", &[&a, &ints(&[Some(3)]), &ints(&[Some(0)])]).expect("fits");
    assert_eq!(rows(&ok), vec![Some(vec![Some(1), None, Some(0)])]);
}
