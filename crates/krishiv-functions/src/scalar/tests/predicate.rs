// SPDX-License-Identifier: Apache-2.0
//! The predicate bindings: `equal_null`, `isnan`, `isnull`/`isnotnull`
//! and `assert_true`/`raise_error` over nulls, NaNs and both ANSI modes.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int64Array,
};
use arrow_schema::DataType;
use krishiv_common::error::ErrorClass;

use super::{BindArg, BindOptions, bind, bind_scalar, bind_with, class, col, ints, run, strs};

fn bools(v: &[Option<bool>]) -> ArrayRef {
    Arc::new(BooleanArray::from(v.to_vec()))
}

fn doubles(v: &[Option<f64>]) -> ArrayRef {
    Arc::new(Float64Array::from(v.to_vec()))
}

fn floats(v: &[Option<f32>]) -> ArrayRef {
    Arc::new(Float32Array::from(v.to_vec()))
}

fn bool_values(a: &ArrayRef) -> Vec<Option<bool>> {
    a.as_boolean().iter().collect()
}

fn legacy() -> BindOptions {
    BindOptions::new().with_ansi(false)
}

#[test]
fn equal_null_matches_the_null_handling_golden() {
    // `select b, c, equal_null(b, c), equal_null(c, b) from t1` in
    // `null-handling.sql`.
    let (b, c) = (
        ints(&[Some(0), Some(0), Some(1), Some(1), None, None, None]),
        ints(&[Some(0), Some(1), Some(0), Some(1), Some(0), Some(1), None]),
    );
    let want = [true, false, false, true, false, false, true].map(Some);
    let f = bind("equal_null", &[&b, &c]);
    assert_eq!(f.return_type(), &DataType::Boolean);
    assert_eq!(
        bool_values(&f.invoke(&[col(&b), col(&c)], 7).unwrap()),
        want
    );
    assert_eq!(
        bool_values(&run("equal_null", &[&c, &b])),
        [true, false, false, true, false, false, true].map(Some)
    );
    // Never NULL, in both ANSI modes.
    let g = bind_with("equal_null", &[&b, &c], &legacy());
    assert_eq!(
        bool_values(&g.invoke(&[col(&b), col(&c)], 7).unwrap()),
        want
    );
}

#[test]
fn equal_null_compares_every_type_null_safely() {
    // Doubles: `NaN` equals `NaN`, `-0.0` equals `0.0` (as `<=>` does).
    let (a, b) = (
        doubles(&[Some(f64::NAN), Some(-0.0), None]),
        doubles(&[Some(f64::NAN), Some(0.0), Some(1.0)]),
    );
    assert_eq!(
        bool_values(&run("equal_null", &[&a, &b])),
        [Some(true), Some(true), Some(false)]
    );
    // Floats, booleans, strings, binaries, dates, timestamps, decimals.
    let (a, b) = (floats(&[Some(1.5), None]), floats(&[Some(1.5), Some(1.5)]));
    assert_eq!(
        bool_values(&run("equal_null", &[&a, &b])),
        [Some(true), Some(false)]
    );
    let (a, b) = (
        bools(&[Some(true), None]),
        bools(&[Some(true), Some(false)]),
    );
    assert_eq!(
        bool_values(&run("equal_null", &[&a, &b])),
        [Some(true), Some(false)]
    );
    let (a, b) = (strs(&[Some("x"), None]), strs(&[Some("x"), Some("y")]));
    assert_eq!(
        bool_values(&run("equal_null", &[&a, &b])),
        [Some(true), Some(false)]
    );
    // Two untyped NULLs are equal.
    let (n, m): (ArrayRef, ArrayRef) = (
        Arc::new(arrow_array::NullArray::new(2)),
        Arc::new(arrow_array::NullArray::new(2)),
    );
    let f = bind("equal_null", &[&n, &m]);
    assert_eq!(
        bool_values(&f.invoke(&[col(&n), col(&m)], 2).unwrap()),
        [Some(true), Some(true)]
    );
}

#[test]
fn isnan_is_never_null() {
    let a = doubles(&[Some(f64::NAN), Some(1.0), Some(f64::INFINITY), None]);
    let f = bind("isnan", &[&a]);
    assert_eq!(f.return_type(), &DataType::Boolean);
    assert_eq!(
        bool_values(&f.invoke(&[col(&a)], 4).unwrap()),
        [Some(true), Some(false), Some(false), Some(false)]
    );
    let b = floats(&[Some(f32::NAN), Some(-0.0)]);
    assert_eq!(bool_values(&run("isnan", &[&b])), [Some(true), Some(false)]);
}

#[test]
fn isnull_and_isnotnull() {
    let a = ints(&[Some(1), None]);
    assert_eq!(
        bool_values(&run("isnull", &[&a])),
        [Some(false), Some(true)]
    );
    assert_eq!(
        bool_values(&run("isnotnull", &[&a])),
        [Some(true), Some(false)]
    );
    for name in ["isnull", "ISNOTNULL"] {
        assert_eq!(bind(name, &[&a]).return_type(), &DataType::Boolean);
    }
}

#[test]
fn assert_true_passes_null_and_raises_otherwise() {
    // `SELECT assert_true(true), assert_true(boolean(1))` passes void NULLs.
    let (t, m) = (bools(&[Some(true)]), strs(&[Some("'true' is not true!")]));
    let f = bind("assert_true", &[&t, &m]);
    assert_eq!(f.return_type(), &DataType::Null);
    assert_eq!(f.invoke(&[col(&t), col(&m)], 1).unwrap().len(), 1);
    // The `misc-functions.sql` failures.
    for (cond, msg, want) in [
        (Some(false), "'false' is not true!", "'false' is not true!"),
        (None, "'NULL' is not true!", "'NULL' is not true!"),
    ] {
        let (c, msg) = (bools(&[cond]), strs(&[Some(msg)]));
        let err = bind("assert_true", &[&c, &msg])
            .invoke(&[col(&c), col(&msg)], 1)
            .unwrap_err();
        assert_eq!(class(&err), ErrorClass::UserRaisedException);
        assert_eq!(
            err.to_string(),
            format!("[USER_RAISED_EXCEPTION] {want} SQLSTATE: P0001")
        );
    }
    // A custom message passes through.
    let (c, msg) = (bools(&[Some(false)]), strs(&[Some("custom error message")]));
    let err = bind("assert_true", &[&c, &msg])
        .invoke(&[col(&c), col(&msg)], 1)
        .unwrap_err();
    assert!(err.to_string().contains("custom error message"), "{err}");
}

#[test]
fn raise_error_always_raises() {
    // `SELECT raise_error('error message')` in `misc-functions.sql`.
    let m = strs(&[Some("error message"), None]);
    let f = bind("raise_error", &[&m]);
    assert_eq!(f.return_type(), &DataType::Null);
    let err = f.invoke(&[col(&m)], 2).unwrap_err();
    assert_eq!(class(&err), ErrorClass::UserRaisedException);
    assert_eq!(
        err.to_string(),
        "[USER_RAISED_EXCEPTION] error message SQLSTATE: P0001"
    );
    // A NULL message raises `"null"`.
    let n = strs(&[None]);
    let err = bind("raise_error", &[&n])
        .invoke(&[col(&n)], 1)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "[USER_RAISED_EXCEPTION] null SQLSTATE: P0001"
    );
}

#[test]
fn predicates_reject_other_types() {
    let s = strs(&[Some("a")]);
    let err = bind_scalar(
        "isnan",
        &[s.data_type().clone().into()],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
    let b = bools(&[Some(true)]);
    let bound: Vec<BindArg> = [b.data_type().clone().into()].to_vec();
    let err = bind_scalar("assert_true", &bound, &BindOptions::new()).unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn equal_null_covers_widths_dates_and_bytes() {
    use arrow_array::{Date32Array, Decimal128Array, TimestampMicrosecondArray};
    // Narrow integrals keep their width.
    let (a, b) = (
        Arc::new(Int8Array::from(vec![Some(1_i8), None])) as ArrayRef,
        Arc::new(Int8Array::from(vec![Some(1_i8), Some(2_i8)])) as ArrayRef,
    );
    assert_eq!(
        bool_values(&run("equal_null", &[&a, &b])),
        [Some(true), Some(false)]
    );
    let (a, b) = (
        Arc::new(Int16Array::from(vec![Some(1_i16)])) as ArrayRef,
        Arc::new(Int16Array::from(vec![Some(2_i16)])) as ArrayRef,
    );
    assert_eq!(bool_values(&run("equal_null", &[&a, &b])), [Some(false)]);
    let (a, b) = (
        Arc::new(Int64Array::from(vec![Some(1_i64)])) as ArrayRef,
        Arc::new(Int64Array::from(vec![Some(1_i64)])) as ArrayRef,
    );
    assert_eq!(bool_values(&run("equal_null", &[&a, &b])), [Some(true)]);
    // Decimals, dates, timestamps and binaries compare by value.
    let dec = |v: i128| {
        Arc::new(
            Decimal128Array::from(vec![Some(v)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ) as ArrayRef
    };
    assert_eq!(
        bool_values(&run("equal_null", &[&dec(150), &dec(150)])),
        [Some(true)]
    );
    let (a, b) = (
        Arc::new(Date32Array::from(vec![Some(1)])) as ArrayRef,
        Arc::new(Date32Array::from(vec![None])) as ArrayRef,
    );
    assert_eq!(bool_values(&run("equal_null", &[&a, &b])), [Some(false)]);
    let (a, b) = (
        Arc::new(TimestampMicrosecondArray::from(vec![Some(1_i64)])) as ArrayRef,
        Arc::new(TimestampMicrosecondArray::from(vec![Some(1_i64)])) as ArrayRef,
    );
    assert_eq!(bool_values(&run("equal_null", &[&a, &b])), [Some(true)]);
    let (a, b) = (
        Arc::new(arrow_array::BinaryArray::from(vec![Some(b"x".as_slice())])) as ArrayRef,
        Arc::new(arrow_array::BinaryArray::from(vec![Some(b"y".as_slice())])) as ArrayRef,
    );
    assert_eq!(bool_values(&run("equal_null", &[&a, &b])), [Some(false)]);
    // Mixed types (which the analyzer casts first) are refused at bind time.
    let (x, y) = (ints(&[Some(1)]), strs(&[Some("1")]));
    let bad: Vec<BindArg> = vec![x.data_type().clone().into(), y.data_type().clone().into()];
    let err = bind_scalar("equal_null", &bad, &BindOptions::new()).unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
}
