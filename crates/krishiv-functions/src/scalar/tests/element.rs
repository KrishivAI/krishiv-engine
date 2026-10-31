// SPDX-License-Identifier: Apache-2.0
//! `array_size`, `get`, `element_at` and `try_element_at`, checked against
//! Spark 4.2's `array.sql`, `map.sql` and `try_element_at.sql` goldens (the
//! session golden run covers the SQL forms).

use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder, MapBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, Int32Array};
use arrow_schema::{DataType, Field};

use super::{BindArg, BindOptions, bind_scalar, bind_with, col};

const ONE_TWO_THREE: &[Option<i32>] = &[Some(1), Some(2), Some(3)];

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

fn idx(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(v.to_vec()))
}

fn run(name: &str, opts: &BindOptions, args: &[&ArrayRef]) -> Result<ArrayRef, String> {
    let f = bind_with(name, args, opts);
    f.invoke(
        &args.iter().map(|a| col(a)).collect::<Vec<_>>(),
        args[0].len(),
    )
    .map_err(|e| format!("{} {e}", e.error_class().name()))
}

fn ints(a: &ArrayRef) -> Vec<Option<i32>> {
    a.as_primitive::<Int32Type>().iter().collect()
}

#[test]
fn array_size_counts_elements() {
    let a = lists(&[Some(&[]), Some(&[Some(2), None]), None]);
    let out = run("array_size", &BindOptions::new(), &[&a]).unwrap();
    assert_eq!(ints(&out), [Some(0), Some(2), None]);
    // An untyped NULL column is NULL; a map does not bind.
    let nulls: ArrayRef = arrow_array::new_null_array(&DataType::Null, 2);
    let out = run("array_size", &BindOptions::new(), &[&nulls]).unwrap();
    assert_eq!(ints(&out), [None, None]);
    assert!(bind_scalar("array_size", &[DataType::Int32.into()], &BindOptions::new()).is_err());
}

#[test]
fn get_counts_from_zero_and_is_null_out_of_range() {
    let a = lists(&[Some(&[Some(1), Some(2), Some(3)]), None, Some(&[Some(1)])]);
    let i = idx(&[Some(0), Some(0), None]);
    let out = run("get", &BindOptions::new(), &[&a, &i]).unwrap();
    assert_eq!(ints(&out), [Some(1), None, None]);
    let a = lists(&[Some(ONE_TWO_THREE); 2]);
    let out = run(
        "get",
        &BindOptions::new(),
        &[&a, &idx(&[Some(3), Some(-1)])],
    )
    .unwrap();
    assert_eq!(ints(&out), [None, None]);
}

#[test]
fn element_at_counts_from_one_or_the_end() {
    let a = lists(&[Some(ONE_TWO_THREE); 4]);
    let i = idx(&[Some(1), Some(3), Some(-1), Some(-3)]);
    let out = run("element_at", &BindOptions::new(), &[&a, &i]).unwrap();
    assert_eq!(ints(&out), [Some(1), Some(3), Some(3), Some(1)]);
    // NULL array, NULL index and a NULL element are NULL, even for index 0.
    let a = lists(&[None, Some(&[None]), Some(&[Some(1)])]);
    let out = run(
        "element_at",
        &BindOptions::new(),
        &[&a, &idx(&[Some(0), Some(1), None])],
    )
    .unwrap();
    assert_eq!(ints(&out), [None, None, None]);
}

#[test]
fn element_at_out_of_range_depends_on_ansi() {
    let a = lists(&[Some(&[Some(1), Some(2), Some(3)])]);
    for bad in [5, -5] {
        let err = run("element_at", &BindOptions::new(), &[&a, &idx(&[Some(bad)])]).unwrap_err();
        assert!(
            err.starts_with("INVALID_ARRAY_INDEX_IN_ELEMENT_AT"),
            "{err}"
        );
        assert!(err.contains(&bad.to_string()) && err.contains('3'), "{err}");
        let legacy = BindOptions::new().with_ansi(false);
        let out = run("element_at", &legacy, &[&a, &idx(&[Some(bad)])]).unwrap();
        assert_eq!(ints(&out), [None]);
        let out = run(
            "try_element_at",
            &BindOptions::new(),
            &[&a, &idx(&[Some(bad)])],
        )
        .unwrap();
        assert_eq!(ints(&out), [None]);
    }
    // Index 0 is an error for both, whatever ANSI says.
    for name in ["element_at", "try_element_at"] {
        for ansi in [true, false] {
            let opts = BindOptions::new().with_ansi(ansi);
            let err = run(name, &opts, &[&a, &idx(&[Some(0)])]).unwrap_err();
            assert!(err.starts_with("INVALID_INDEX_OF_ZERO"), "{name} {err}");
        }
    }
}

#[test]
fn element_at_reads_a_map_by_key() {
    let mut b = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    b.keys().append_value("a");
    b.values().append_value("b");
    b.append(true).unwrap();
    b.keys().append_value("a");
    b.values().append_value("b");
    b.append(true).unwrap();
    b.append(false).unwrap();
    let map: ArrayRef = Arc::new(b.finish());
    let key: ArrayRef = Arc::new(arrow_array::StringArray::from(vec![
        Some("a"),
        Some("abc"),
        Some("a"),
    ]));
    for name in ["element_at", "try_element_at"] {
        let out = run(name, &BindOptions::new(), &[&map, &key]).unwrap();
        let out = out.as_string::<i32>();
        assert_eq!(
            [out.is_null(0), out.is_null(1), out.is_null(2)],
            [false, true, true],
            "{name}"
        );
        assert_eq!(out.value(0), "b");
    }
    // A key of another type than the map's does not bind.
    let wrong = [
        map.data_type().clone().into(),
        BindArg::from(DataType::Int32),
    ];
    assert!(bind_scalar("element_at", &wrong, &BindOptions::new()).is_err());
    assert!(
        bind_scalar(
            "element_at",
            &[DataType::Int32.into(), DataType::Int32.into()],
            &BindOptions::new()
        )
        .is_err()
    );
}

#[test]
fn subscript_counts_from_zero_and_errors_out_of_range_under_ansi() {
    let a = lists(&[Some(ONE_TWO_THREE), None, Some(ONE_TWO_THREE)]);
    let i = idx(&[Some(0), Some(0), None]);
    let out = run("__subscript", &BindOptions::new(), &[&a, &i]).unwrap();
    assert_eq!(ints(&out), [Some(1), None, None]);
    // Past the end or negative: INVALID_ARRAY_INDEX with ANSI, NULL without.
    for bad in [3, -1] {
        let a = lists(&[Some(ONE_TWO_THREE)]);
        let err = run(
            "__subscript",
            &BindOptions::new(),
            &[&a, &idx(&[Some(bad)])],
        )
        .unwrap_err();
        assert!(err.starts_with("INVALID_ARRAY_INDEX"), "{err}");
        assert!(
            !err.starts_with("INVALID_ARRAY_INDEX_IN_ELEMENT_AT"),
            "{err}"
        );
        let legacy = BindOptions::new().with_ansi(false);
        let out = run("__subscript", &legacy, &[&a, &idx(&[Some(bad)])]).unwrap();
        assert_eq!(ints(&out), [None]);
    }
}

#[test]
fn subscript_reads_a_map_by_key_and_misses_are_null() {
    let mut b = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    b.keys().append_value("a");
    b.values().append_value("b");
    b.append(true).unwrap();
    b.keys().append_value("a");
    b.values().append_value("b");
    b.append(true).unwrap();
    let map: ArrayRef = Arc::new(b.finish());
    let key: ArrayRef = Arc::new(arrow_array::StringArray::from(vec![Some("a"), Some("z")]));
    for ansi in [true, false] {
        let opts = BindOptions::new().with_ansi(ansi);
        let out = run("__subscript", &opts, &[&map, &key]).unwrap();
        let out = out.as_string::<i32>();
        assert_eq!([out.is_null(0), out.is_null(1)], [false, true]);
        assert_eq!(out.value(0), "b");
    }
    // A key of another type than the map's does not bind.
    let wrong = [
        map.data_type().clone().into(),
        BindArg::from(DataType::Int32),
    ];
    assert!(bind_scalar("__subscript", &wrong, &BindOptions::new()).is_err());
}
