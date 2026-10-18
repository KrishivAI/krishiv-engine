// SPDX-License-Identifier: Apache-2.0
//! The math, arithmetic and hash bindings.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Float64Type, Int64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, Decimal128Array, Float64Array, Int64Array};
use arrow_schema::DataType;
use krishiv_common::error::ErrorClass;

use super::{
    Arg, BindArg, BindOptions, bind, bind_scalar, bind_with, bytes, class, col, int_values, ints,
    lit, owned, run, strs, text,
};

fn doubles(v: &[Option<f64>]) -> ArrayRef {
    Arc::new(Float64Array::from(v.to_vec()))
}

fn longs(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn decimals(v: &[Option<i128>], p: u8, s: i8) -> ArrayRef {
    Arc::new(
        Decimal128Array::from(v.to_vec())
            .with_precision_and_scale(p, s)
            .unwrap(),
    )
}

fn f64s(a: &ArrayRef) -> Vec<Option<f64>> {
    a.as_primitive::<Float64Type>().iter().collect()
}

fn i64s(a: &ArrayRef) -> Vec<Option<i64>> {
    a.as_primitive::<Int64Type>().iter().collect()
}

fn dec_values(a: &ArrayRef) -> Vec<Option<i128>> {
    a.as_primitive::<Decimal128Type>().iter().collect()
}

fn legacy() -> BindOptions {
    BindOptions::new().with_ansi(false)
}

/// `name` over columns `args` where the last argument is a literal.
fn run_lit(name: &str, args: &[&ArrayRef], literal: &ArrayRef) -> ArrayRef {
    let rows = args.first().map_or(1, |a| Array::len(a.as_ref()));
    let mut bound: Vec<BindArg> = args.iter().map(|a| a.data_type().clone().into()).collect();
    bound.push(BindArg::Literal(Arc::clone(literal)));
    let f = bind_scalar(name, &bound, &BindOptions::new()).unwrap();
    let mut given: Vec<Arg> = args.iter().map(|a| col(a)).collect();
    given.push(lit(literal));
    f.invoke(&given, rows).unwrap()
}

#[test]
fn double_functions() {
    let x = doubles(&[Some(4.0), None, Some(-1.0), Some(0.0)]);
    let sqrt = run("sqrt", &[&x]);
    assert_eq!(sqrt.data_type(), &DataType::Float64);
    let v = f64s(&sqrt);
    assert_eq!((v[0], v[1], v[3]), (Some(2.0), None, Some(0.0)));
    assert!(v[2].unwrap().is_nan());
    // `ln` and `log` are NULL outside their domain.
    assert_eq!(f64s(&run("ln", &[&x]))[2..], [None, None]);
    let (base, v) = (doubles(&[Some(2.0)]), doubles(&[Some(8.0)]));
    let f = f64s(&run("log", &[&base, &v]));
    assert!((f[0].unwrap() - 3.0).abs() < 1e-12);
    assert_eq!(f64s(&run("log", &[&x]))[0], f64s(&run("ln", &[&x]))[0]);
    let (a, b) = (
        doubles(&[Some(2.0), None]),
        doubles(&[Some(10.0), Some(1.0)]),
    );
    assert_eq!(f64s(&run("pow", &[&a, &b])), [Some(1024.0), None]);
    assert_eq!(f64s(&run("power", &[&a, &b]))[0], Some(1024.0));
    let hyp = f64s(&run("hypot", &[&a, &a]))[0].unwrap();
    assert!((hyp * hyp - 8.0).abs() < 1e-9);
    assert_eq!(f64s(&run("signum", &[&x]))[2], Some(-1.0));
    let pi = bind("pi", &[]).invoke(&[], 3).unwrap();
    assert_eq!(f64s(&pi), [Some(std::f64::consts::PI); 3]);
    assert_eq!(
        f64s(&bind("e", &[]).invoke(&[], 1).unwrap()),
        [Some(std::f64::consts::E)]
    );
}

#[test]
fn rounding_takes_its_scale_from_a_literal() {
    let scale = |s| ints(&[Some(s)]);
    let d = doubles(&[Some(2.5), Some(-2.5), Some(1.15), None]);
    let half_up = run_lit("round", &[&d], &scale(0));
    assert_eq!(f64s(&half_up), [Some(3.0), Some(-3.0), Some(1.0), None]);
    let half_even = run_lit("bround", &[&d], &scale(0));
    assert_eq!(f64s(&half_even), [Some(2.0), Some(-2.0), Some(1.0), None]);
    let i = ints(&[Some(15), Some(25), None]);
    assert_eq!(
        int_values(&run_lit("round", &[&i], &scale(-1))),
        [Some(20), Some(30), None]
    );
    assert_eq!(
        int_values(&run_lit("bround", &[&i], &scale(-1))),
        [Some(20), Some(20), None]
    );
    // A decimal's type follows the scale: 2.5 is decimal(2,1).
    let dec = decimals(&[Some(25), Some(-25)], 2, 1);
    let r = run_lit("round", &[&dec], &scale(0));
    assert_eq!(r.data_type(), &DataType::Decimal128(2, 0));
    assert_eq!(dec_values(&r), [Some(3), Some(-3)]);
    // A NULL scale gives NULLs of the same type.
    let null = arrow_array::new_null_array(&DataType::Int32, 1);
    let r = run_lit("round", &[&d], &null);
    assert_eq!(r.null_count(), 4);
}

#[test]
fn rounding_rejects_a_scale_that_varies() {
    let err = bind_scalar(
        "round",
        &[
            BindArg::from(DataType::Float64),
            BindArg::from(DataType::Int32),
        ],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn ceil_and_floor() {
    let d = doubles(&[Some(1.2), Some(-1.2), None]);
    assert_eq!(i64s(&run("ceil", &[&d])), [Some(2), Some(-1), None]);
    assert_eq!(i64s(&run("floor", &[&d])), [Some(1), Some(-2), None]);
    assert_eq!(i64s(&run("ceiling", &[&d]))[0], Some(2));
    // `bigint` stays as it is.
    let l = longs(&[Some(7), None]);
    assert_eq!(i64s(&run("ceil", &[&l])), [Some(7), None]);
    // A decimal rounds to its scale: 3.1411 (decimal(5,4)) to 3 digits.
    let dec = decimals(&[Some(31411), Some(-31411)], 5, 4);
    let up = run_lit("ceil", &[&dec], &ints(&[Some(3)]));
    assert_eq!(up.data_type(), &DataType::Decimal128(5, 3));
    assert_eq!(dec_values(&up), [Some(3142), Some(-3141)]);
    let down = run_lit("floor", &[&dec], &ints(&[Some(3)]));
    assert_eq!(dec_values(&down), [Some(3141), Some(-3142)]);
    let whole = run("floor", &[&dec]);
    assert_eq!(whole.data_type(), &DataType::Decimal128(2, 0));
    assert_eq!(dec_values(&whole), [Some(3), Some(-4)]);
}

#[test]
fn pmod_is_never_negative() {
    let (a, b) = (
        ints(&[Some(-7), Some(7), Some(-7), None]),
        ints(&[Some(3), Some(-3), Some(0), Some(3)]),
    );
    let err = bind("pmod", &[&a, &b])
        .invoke(&[col(&a), col(&b)], 4)
        .unwrap_err();
    assert_eq!(class(&err), ErrorClass::RemainderByZero);
    let f = bind_with("pmod", &[&a, &b], &legacy());
    let r = f.invoke(&[col(&a), col(&b)], 4).unwrap();
    assert_eq!(int_values(&r), [Some(2), Some(1), None, None]);
    let (x, y) = (doubles(&[Some(-7.5)]), doubles(&[Some(2.0)]));
    assert_eq!(f64s(&run("pmod", &[&x, &y])), [Some(0.5)]);
    let (p, q) = (decimals(&[Some(-75)], 3, 1), decimals(&[Some(20)], 2, 1));
    let r = run("pmod", &[&p, &q]);
    assert_eq!(r.data_type(), &DataType::Decimal128(2, 1));
    assert_eq!(dec_values(&r), [Some(5)]);
}

#[test]
fn greatest_and_least_skip_nulls() {
    let (a, b, c) = (
        ints(&[Some(1), None, None]),
        ints(&[None, Some(5), None]),
        ints(&[Some(3), Some(2), None]),
    );
    assert_eq!(
        int_values(&run("greatest", &[&a, &b, &c])),
        [Some(3), Some(5), None]
    );
    assert_eq!(
        int_values(&run("least", &[&a, &b, &c])),
        [Some(1), Some(2), None]
    );
    let (s, t) = (strs(&[Some("b"), Some("a")]), strs(&[Some("a"), None]));
    assert_eq!(
        text(&run("greatest", &[&s, &t])),
        owned(&[Some("b"), Some("a")])
    );
    let (f, g) = (doubles(&[Some(f64::NAN)]), doubles(&[Some(1.0)]));
    assert!(f64s(&run("greatest", &[&f, &g]))[0].unwrap().is_nan());
    assert_eq!(f64s(&run("least", &[&f, &g])), [Some(1.0)]);
    let bools = |v: &[Option<bool>]| -> ArrayRef { Arc::new(BooleanArray::from(v.to_vec())) };
    let r = run("greatest", &[&bools(&[Some(false)]), &bools(&[Some(true)])]);
    assert_eq!(r.as_boolean().iter().collect::<Vec<_>>(), [Some(true)]);
    let (x, y) = (bytes(&[Some(b"\x01")]), bytes(&[Some(b"\x02")]));
    assert_eq!(
        run("greatest", &[&x, &y]).as_binary::<i32>().value(0),
        b"\x02"
    );
}

#[test]
fn abs_and_positive() {
    let a = ints(&[Some(-3), Some(i32::MIN), None]);
    let err = bind("abs", &[&a]).invoke(&[col(&a)], 3).unwrap_err();
    assert_eq!(class(&err), ErrorClass::ArithmeticOverflow);
    let r = bind_with("abs", &[&a], &legacy())
        .invoke(&[col(&a)], 3)
        .unwrap();
    assert_eq!(int_values(&r), [Some(3), Some(i32::MIN), None]);
    let d = doubles(&[Some(-0.0), Some(-1.5)]);
    assert_eq!(f64s(&run("abs", &[&d])), [Some(0.0), Some(1.5)]);
    let dec = decimals(&[Some(-150)], 3, 2);
    assert_eq!(dec_values(&run("abs", &[&dec])), [Some(150)]);
    let p = run("positive", &[&a]);
    assert_eq!(int_values(&p), [Some(-3), Some(i32::MIN), None]);
    let s = strs(&[Some("a")]);
    let err = bind_scalar("abs", &[s.data_type().clone().into()], &BindOptions::new()).unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn try_arithmetic_is_null_where_the_operator_raises() {
    let (a, b) = (
        ints(&[Some(i32::MAX), Some(1), None, Some(i32::MIN)]),
        ints(&[Some(1), Some(2), Some(1), Some(-1)]),
    );
    assert_eq!(
        int_values(&run("try_add", &[&a, &b])),
        [None, Some(3), None, None]
    );
    assert_eq!(
        int_values(&run("try_subtract", &[&a, &b])),
        [Some(i32::MAX - 1), Some(-1), None, Some(i32::MIN + 1)]
    );
    assert_eq!(
        int_values(&run("try_multiply", &[&a, &b]))[..2],
        [Some(i32::MAX), Some(2)]
    );
    let zero = ints(&[Some(0), Some(0)]);
    let n = ints(&[Some(5), Some(7)]);
    assert_eq!(int_values(&run("try_mod", &[&n, &zero])), [None, None]);
    let (x, y) = (
        doubles(&[Some(1.0), Some(1.0)]),
        doubles(&[Some(0.0), Some(4.0)]),
    );
    assert_eq!(f64s(&run("try_divide", &[&x, &y])), [None, Some(0.25)]);
    // Decimals take the result type of the operator, NULL on overflow.
    let big = decimals(
        &[Some(99_999_999_999_999_999_999_999_999_999_999_999_999)],
        38,
        0,
    );
    let two = decimals(&[Some(2)], 1, 0);
    let r = run("try_multiply", &[&big, &two]);
    assert_eq!(r.data_type(), &DataType::Decimal128(38, 0));
    assert_eq!(r.null_count(), 1);
    let s = decimals(&[Some(150)], 3, 2);
    let r = run("try_add", &[&s, &s]);
    assert_eq!(r.data_type(), &DataType::Decimal128(4, 2));
    assert_eq!(dec_values(&r), [Some(300)]);
}

#[test]
fn mod_follows_the_ansi_mode() {
    let (a, z) = (ints(&[Some(7), Some(-7)]), ints(&[Some(0), Some(3)]));
    let err = bind("mod", &[&a, &z])
        .invoke(&[col(&a), col(&z)], 2)
        .unwrap_err();
    assert_eq!(class(&err), ErrorClass::RemainderByZero);
    let r = bind_with("mod", &[&a, &z], &legacy())
        .invoke(&[col(&a), col(&z)], 2)
        .unwrap();
    assert_eq!(int_values(&r), [None, Some(-1)]);
}

#[test]
fn hashes_match_spark() {
    let one = ints(&[Some(1), None]);
    let h = run("hash", &[&one]);
    assert_eq!(h.data_type(), &DataType::Int32);
    assert_eq!(int_values(&h), [Some(-559_580_957), Some(42)]);
    let x = run("xxhash64", &[&one]);
    assert_eq!(i64s(&x), [Some(-6_698_625_589_789_238_999), Some(42)]);
    let (s, n) = (strs(&[Some("abc"), None]), ints(&[Some(1), Some(1)]));
    assert_eq!(int_values(&run("hash", &[&s, &n])).len(), 2);
    let abc = bytes(&[Some(b"abc"), None]);
    assert_eq!(
        text(&run("md5", &[&abc])),
        owned(&[Some("900150983cd24fb0d6963f7d28e17f72"), None])
    );
    for name in ["sha", "sha1"] {
        assert_eq!(
            text(&run(name, &[&abc])),
            owned(&[Some("a9993e364706816aba3e25717850c26c9cd0d89d"), None])
        );
    }
    assert_eq!(i64s(&run("crc32", &[&abc])), [Some(891_568_578), None]);
    let bits = ints(&[Some(256), Some(0)]);
    let both = bytes(&[Some(b"abc"), Some(b"abc")]);
    let sha256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert_eq!(
        text(&run("sha2", &[&both, &bits])),
        owned(&[Some(sha256), Some(sha256)])
    );
    let odd = ints(&[Some(100), Some(256)]);
    assert_eq!(text(&run("sha2", &[&both, &odd]))[0], None);
    // A string argument is read as its bytes.
    let text_abc = strs(&[Some("abc")]);
    let f = bind_scalar(
        "md5",
        &[text_abc.data_type().clone().into()],
        &BindOptions::new(),
    );
    assert_eq!(
        text(&f.unwrap().invoke(&[col(&text_abc)], 1).unwrap()),
        owned(&[Some("900150983cd24fb0d6963f7d28e17f72")])
    );
}

#[test]
fn hash_arguments_are_checked() {
    let err = bind_scalar("hash", &[], &BindOptions::new()).unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
    let err = bind_scalar(
        "sha2",
        &[
            BindArg::from(DataType::Binary),
            BindArg::from(DataType::Int64),
        ],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
}
