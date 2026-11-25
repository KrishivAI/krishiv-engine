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
fn ceil_and_floor_overflow_raises_only_on_the_4_3_profile() {
    use krishiv_kernels::{KernelCtx, SparkVersion};
    let d = doubles(&[
        Some(1e308),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
        Some(f64::NAN),
    ]);
    for name in ["ceil", "floor"] {
        let v42 =
            BindOptions::new().with_kernel(KernelCtx::new().with_spark_version(SparkVersion::V4_2));
        let f = bind_with(name, &[&d], &v42);
        assert!(f.invoke(&[col(&d)], 4).is_ok(), "{name} saturates on 4.2");
        let v43 =
            BindOptions::new().with_kernel(KernelCtx::new().with_spark_version(SparkVersion::V4_3));
        let f = bind_with(name, &[&d], &v43);
        let e = f.invoke(&[col(&d)], 4).unwrap_err();
        assert_eq!(class(&e), ErrorClass::ArithmeticOverflow);
    }
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

/// `hmac` against Spark 4.3.0 (`misc-functions.sql` and a run of the same build).
#[test]
fn hmac_matches_spark() {
    fn hex(a: &ArrayRef) -> Vec<Option<String>> {
        a.as_binary::<i32>()
            .iter()
            .map(|b| b.map(|b| b.iter().map(|x| format!("{x:02X}")).collect()))
            .collect()
    }
    let (key, msg) = (
        bytes(&[Some(b"key"), None, Some(b"key")]),
        bytes(&[Some(b"message"), Some(b"m"), None]),
    );
    assert_eq!(
        hex(&run("hmac", &[&key, &msg])),
        owned(&[
            Some("6E9EF29B75FFFC5B7ABAE527D58FDADB2FE42E7219011976917343065F58ED4A"),
            None,
            None
        ])
    );
    let (key, msg) = (bytes(&[Some(b"key")]), bytes(&[Some(b"message")]));
    for (algorithm, want) in [
        (
            "SHA-256",
            "6E9EF29B75FFFC5B7ABAE527D58FDADB2FE42E7219011976917343065F58ED4A",
        ),
        (
            "sha256",
            "6E9EF29B75FFFC5B7ABAE527D58FDADB2FE42E7219011976917343065F58ED4A",
        ),
        (
            "SHA-512",
            "E477384D7CA229DD1426E64B63EBF2D36EBD6D7E669A6735424E72EA6C01D3F8B56EB39C36D8232F5427999B8D1A3F9CD1128FC69F4D75B434216810FA367E98",
        ),
        ("md5", "4E4748E62B463521F6775FBF921234B5"),
        ("SHA1", "2088DF74D5F2146B48146CAF4965377E9D0BE3A4"),
        (
            "sha-224",
            "A0B5EECAE3F74F0561A8DA6F389F78F1A3895C8C183C31C1756D7925",
        ),
    ] {
        let a = strs(&[Some(algorithm)]);
        assert_eq!(
            hex(&run("hmac", &[&key, &msg, &a])),
            owned(&[Some(want)]),
            "{algorithm}"
        );
    }
    // A key longer than the block is hashed first (200 bytes: past SHA-512's 128, 100 past SHA-256's 64).
    let long = |n: usize| bytes(&[Some(&vec![b'k'; n])]);
    let a = strs(&[Some("SHA-512")]);
    assert_eq!(
        hex(&run("hmac", &[&long(200), &msg, &a]))[0].as_deref(),
        Some(
            "A478FECF5C141A239F904428CCEFD58B93140691B2111D79148CB120C51A5E169AC7CF8FE0C8DA81B16F86FD479E4118B882A04EF1725BF16BE494BFFD564A54"
        )
    );
    let a = strs(&[Some("SHA-256")]);
    assert_eq!(
        hex(&run("hmac", &[&long(100), &msg, &a]))[0].as_deref(),
        Some("1C28735416D320163F56F81BDBB83D651EED508D184E6B8B03662740A533293E")
    );
    // A NULL algorithm is NULL.
    let a = strs(&[None]);
    assert_eq!(hex(&run("hmac", &[&key, &msg, &a])), [None]);
}

#[test]
fn hmac_refuses_what_spark_refuses() {
    let (key, msg) = (bytes(&[Some(b"key")]), bytes(&[Some(b"message")]));
    for name in ["SHA-3", "", " SHA-256"] {
        let a = strs(&[Some(name)]);
        let f = bind("hmac", &[&key, &msg, &a]);
        let e = f.invoke(&[col(&key), col(&msg), col(&a)], 1).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "INVALID_PARAMETER_VALUE.HMAC_ALGORITHM",
            "{name:?}"
        );
        assert!(e.to_string().contains(&format!("'{name}'")), "{e}");
    }
    // The algorithm is checked before the key, an empty key second.
    let (empty, a) = (bytes(&[Some(b"")]), strs(&[Some("MD5")]));
    let e = bind("hmac", &[&empty, &msg, &a])
        .invoke(&[col(&empty), col(&msg), col(&a)], 1)
        .unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "INVALID_PARAMETER_VALUE.HMAC_CRYPTO_ERROR"
    );
    let bad = strs(&[Some("nope")]);
    let e = bind("hmac", &[&empty, &msg, &bad])
        .invoke(&[col(&empty), col(&msg), col(&bad)], 1)
        .unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "INVALID_PARAMETER_VALUE.HMAC_ALGORITHM"
    );
}

fn floats(v: &[Option<f32>]) -> ArrayRef {
    Arc::new(arrow_array::Float32Array::from(v.to_vec()))
}

fn small(v: &[Option<i8>]) -> ArrayRef {
    Arc::new(arrow_array::Int8Array::from(v.to_vec()))
}

fn shorts(v: &[Option<i16>]) -> ArrayRef {
    Arc::new(arrow_array::Int16Array::from(v.to_vec()))
}

#[test]
fn float_arithmetic() {
    let (a, b) = (
        doubles(&[Some(1.5), Some(7.0)]),
        doubles(&[Some(0.5), Some(0.0)]),
    );
    assert_eq!(f64s(&run("try_add", &[&a, &b])), [Some(2.0), Some(7.0)]);
    assert_eq!(
        f64s(&run("try_subtract", &[&a, &b])),
        [Some(1.0), Some(7.0)]
    );
    assert_eq!(
        f64s(&run("try_multiply", &[&a, &b])),
        [Some(0.75), Some(0.0)]
    );
    assert_eq!(f64s(&run("try_divide", &[&a, &b])), [Some(3.0), None]);
    assert_eq!(f64s(&run("try_mod", &[&a, &b])), [Some(0.0), None]);
    // `mod` raises under ANSI and is NULL without.
    let err = bind("mod", &[&a, &b])
        .invoke(&[col(&a), col(&b)], 2)
        .unwrap_err();
    assert_eq!(class(&err), ErrorClass::RemainderByZero);
    let r = bind_with("mod", &[&a, &b], &legacy())
        .invoke(&[col(&a), col(&b)], 2)
        .unwrap();
    assert_eq!(f64s(&r), [Some(0.0), None]);
    let (x, y) = (floats(&[Some(1.5)]), floats(&[Some(0.5)]));
    let r = run("try_add", &[&x, &y]);
    assert_eq!(r.data_type(), &DataType::Float32);
    assert_eq!(
        r.as_primitive::<arrow_array::types::Float32Type>().value(0),
        2.0
    );
}

#[test]
fn other_integral_widths() {
    let (a, b) = (small(&[Some(100), Some(3)]), small(&[Some(100), Some(2)]));
    let r = run("try_add", &[&a, &b]);
    assert_eq!(
        r.as_primitive::<arrow_array::types::Int8Type>()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some(5)]
    );
    let (a, b) = (
        shorts(&[Some(i16::MAX), Some(3)]),
        shorts(&[Some(1), Some(2)]),
    );
    let r = run("try_add", &[&a, &b]);
    assert_eq!(
        r.as_primitive::<arrow_array::types::Int16Type>()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some(5)]
    );
    let (a, b) = (
        longs(&[Some(i64::MAX), Some(3)]),
        longs(&[Some(1), Some(2)]),
    );
    assert_eq!(i64s(&run("try_add", &[&a, &b])), [None, Some(5)]);
    assert_eq!(i64s(&run("try_mod", &[&a, &b])), [Some(0), Some(1)]);
    // abs and pmod on every width.
    assert_eq!(
        i64s(&run("abs", &[&longs(&[Some(-4), None])])),
        [Some(4), None]
    );
    let s = run("abs", &[&small(&[Some(-4)])]);
    assert_eq!(s.as_primitive::<arrow_array::types::Int8Type>().value(0), 4);
    let s = run("abs", &[&shorts(&[Some(-4)])]);
    assert_eq!(
        s.as_primitive::<arrow_array::types::Int16Type>().value(0),
        4
    );
    let f = run("abs", &[&floats(&[Some(-1.5)])]);
    assert_eq!(
        f.as_primitive::<arrow_array::types::Float32Type>().value(0),
        1.5
    );
    let p = run("pmod", &[&longs(&[Some(-7)]), &longs(&[Some(3)])]);
    assert_eq!(i64s(&p), [Some(2)]);
    let p = run("pmod", &[&small(&[Some(-7)]), &small(&[Some(3)])]);
    assert_eq!(p.as_primitive::<arrow_array::types::Int8Type>().value(0), 2);
    let p = run("pmod", &[&shorts(&[Some(-7)]), &shorts(&[Some(3)])]);
    assert_eq!(
        p.as_primitive::<arrow_array::types::Int16Type>().value(0),
        2
    );
    let p = run("pmod", &[&floats(&[Some(-7.5)]), &floats(&[Some(2.0)])]);
    assert_eq!(
        p.as_primitive::<arrow_array::types::Float32Type>().value(0),
        0.5
    );
}

#[test]
fn rounding_every_type() {
    let scale = |s| ints(&[Some(s)]);
    let null = arrow_array::new_null_array(&DataType::Int32, 1);
    let r = run_lit("round", &[&small(&[Some(127)])], &scale(0));
    assert_eq!(
        r.as_primitive::<arrow_array::types::Int8Type>().value(0),
        127
    );
    let r = run_lit("round", &[&shorts(&[Some(15)])], &scale(-1));
    assert_eq!(
        r.as_primitive::<arrow_array::types::Int16Type>().value(0),
        20
    );
    assert_eq!(
        i64s(&run_lit("bround", &[&longs(&[Some(25)])], &scale(-1))),
        [Some(20)]
    );
    let r = run_lit("round", &[&floats(&[Some(2.5), None])], &scale(0));
    assert_eq!(
        r.as_primitive::<arrow_array::types::Float32Type>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(3.0), None]
    );
    // A NULL scale gives NULLs of every type.
    for a in [
        small(&[Some(1)]),
        shorts(&[Some(1)]),
        ints(&[Some(1)]),
        longs(&[Some(1)]),
        floats(&[Some(1.0)]),
        doubles(&[Some(1.0)]),
        decimals(&[Some(15)], 2, 1),
    ] {
        let r = run_lit("round", &[&a], &null);
        assert_eq!((r.len(), r.null_count()), (1, 1), "{}", a.data_type());
    }
    let r = run_lit("ceil", &[&decimals(&[Some(15)], 2, 1)], &null);
    assert_eq!(r.null_count(), 1);
    // `bigint` literals pass through `ceil` untouched.
    let r = bind("ceil", &[&longs(&[Some(3)])])
        .invoke(&[lit(&longs(&[Some(3)]))], 2)
        .unwrap();
    assert_eq!(i64s(&r), [Some(3), Some(3)]);
}

/// Spark's one-argument `floor`/`ceil` keep the type of a decimal whose scale is 0
/// (`case dt @ DecimalType.Fixed(_, 0) => dt`) and add one integral digit to any other
/// (found by the nightly expression differential: `floor(decimal(9,0))` was `decimal(10,0)`).
#[test]
fn one_argument_floor_and_ceil_keep_a_scale_zero_decimal_type() {
    for f in ["floor", "ceil"] {
        let b = bind(f, &[&decimals(&[Some(5)], 9, 0)]);
        assert_eq!(
            b.return_type(),
            &DataType::Decimal128(9, 0),
            "{f}(decimal(9,0))"
        );
        let b = bind(f, &[&decimals(&[Some(5)], 10, 2)]);
        assert_eq!(
            b.return_type(),
            &DataType::Decimal128(9, 0),
            "{f}(decimal(10,2))"
        );
        let r = run(f, &[&decimals(&[Some(5), None, Some(-7)], 9, 0)]);
        assert_eq!(r.data_type(), &DataType::Decimal128(9, 0), "{f} result");
        assert_eq!(dec_values(&r), [Some(5), None, Some(-7)], "{f} values");
    }
}

#[test]
fn greatest_and_least_of_every_type() {
    let both = |name: &str, a: &ArrayRef, b: &ArrayRef| run(name, &[a, b]);
    let r = both("greatest", &small(&[Some(1)]), &small(&[Some(2)]));
    assert_eq!(r.as_primitive::<arrow_array::types::Int8Type>().value(0), 2);
    let r = both("least", &shorts(&[Some(1)]), &shorts(&[Some(2)]));
    assert_eq!(
        r.as_primitive::<arrow_array::types::Int16Type>().value(0),
        1
    );
    assert_eq!(
        i64s(&both("greatest", &longs(&[Some(1)]), &longs(&[Some(2)]))),
        [Some(2)]
    );
    let r = both("greatest", &floats(&[Some(1.0)]), &floats(&[Some(2.0)]));
    assert_eq!(
        r.as_primitive::<arrow_array::types::Float32Type>().value(0),
        2.0
    );
    let r = both(
        "least",
        &decimals(&[Some(15)], 2, 1),
        &decimals(&[Some(12)], 2, 1),
    );
    assert_eq!(dec_values(&r), [Some(12)]);
    let d = |v: i32| -> ArrayRef { Arc::new(arrow_array::Date32Array::from(vec![Some(v)])) };
    let r = both("greatest", &d(1), &d(2));
    assert_eq!(
        r.as_primitive::<arrow_array::types::Date32Type>().value(0),
        2
    );
    let t = |v: i64| -> ArrayRef {
        Arc::new(arrow_array::TimestampMicrosecondArray::from(vec![Some(v)]).with_timezone("UTC"))
    };
    let r = both("least", &t(5), &t(3));
    assert_eq!(
        r.as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .value(0),
        3
    );
    // Arguments of different types do not bind.
    let err = bind_scalar(
        "greatest",
        &[
            BindArg::from(DataType::Int32),
            BindArg::from(DataType::Int64),
        ],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn unsupported_shapes_are_reported() {
    for (name, args) in [
        ("sqrt", vec![DataType::Int32]),
        ("pmod", vec![DataType::Int32, DataType::Int64]),
        ("try_add", vec![DataType::Int32, DataType::Int64]),
        ("try_divide", vec![DataType::Int32, DataType::Int32]),
        ("positive", vec![DataType::Utf8]),
        ("round", vec![DataType::Utf8]),
        ("ceil", vec![DataType::Int32]),
        ("greatest", vec![DataType::Int32]),
        ("md5", vec![DataType::Int32]),
    ] {
        let bound: Vec<BindArg> = args.into_iter().map(BindArg::from).collect();
        let err = bind_scalar(name, &bound, &BindOptions::new()).unwrap_err();
        assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature, "{name}");
    }
    // A scale that is not an `int` literal is an internal error.
    let bad = BindArg::Literal(longs(&[Some(1)]));
    let err = bind_scalar(
        "round",
        &[BindArg::from(DataType::Float64), bad],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(class(&err), ErrorClass::InternalError);
}
