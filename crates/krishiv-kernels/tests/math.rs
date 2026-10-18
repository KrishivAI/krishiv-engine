// SPDX-License-Identifier: Apache-2.0
//! Math kernels: values and errors recorded from Spark 4.0-4.2 (the shapes
//! the differential fixture does not reach), and property tests.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use arrow_array::{
    Array, Decimal128Array, Float32Array, Float64Array, Int8Array, Int32Array, Int64Array,
};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::math::{
    BinaryFn, DecimalRounding, RoundMode, UnaryFn, binary, ceil_floor_to_long, round_decimal,
    round_decimal_type, round_f32, round_f64, round_integral, unary,
};
use krishiv_kernels::{KernelCtx, Operand, OverflowMode, SparkVersion};
use proptest::prelude::*;

fn ctx(v: SparkVersion) -> KernelCtx {
    KernelCtx::new().with_spark_version(v)
}

fn f64s(v: &[Option<f64>]) -> Float64Array {
    Float64Array::from(v.to_vec())
}

fn rf64(x: &Float64Array, scale: i32, mode: RoundMode) -> Float64Array {
    round_f64(x, scale, mode).unwrap()
}

fn rf32(x: &Float32Array, scale: i32, mode: RoundMode) -> Float32Array {
    round_f32(x, scale, mode).unwrap()
}

fn dec(values: &[Option<i128>], p: u8, s: i8) -> Decimal128Array {
    Decimal128Array::from(values.to_vec())
        .with_precision_and_scale(p, s)
        .unwrap()
}

#[test]
#[cfg_attr(
    miri,
    ignore = "libm calls inline assembly (sqrtsd), which Miri does not run; the math kernels contain no unsafe code"
)]
fn double_functions_follow_the_domain_rules() {
    let c = ctx(SparkVersion::V4_2);
    let x = f64s(&[
        Some(-1.0),
        Some(0.0),
        Some(-0.0),
        Some(1.0),
        Some(f64::NAN),
        None,
    ]);
    let ln = unary(UnaryFn::Ln, &x, &c);
    assert_eq!(
        ln.iter().map(|v| v.map(f64::is_nan)).collect::<Vec<_>>(),
        [None, None, None, Some(false), Some(true), None]
    );
    assert_eq!(unary(UnaryFn::Log2, &x, &c).null_count(), 4);
    assert_eq!(unary(UnaryFn::Log10, &x, &c).null_count(), 4);
    // log1p is defined above -1.
    let l1p = unary(UnaryFn::Log1p, &f64s(&[Some(-1.0), Some(-0.5)]), &c);
    assert_eq!(
        (l1p.is_null(0), l1p.value(1)),
        (true, -std::f64::consts::LN_2)
    );
    assert!(unary(UnaryFn::Sqrt, &x, &c).value(0).is_nan());
    assert_eq!(
        unary(UnaryFn::Sqrt, &x, &c).value(2).to_bits(),
        (-0.0_f64).to_bits()
    );
    let sign = unary(UnaryFn::Signum, &x, &c);
    assert_eq!(
        (
            sign.value(0),
            sign.value(2).to_bits(),
            sign.value(3),
            sign.value(4).is_nan()
        ),
        (-1.0, (-0.0_f64).to_bits(), 1.0, true)
    );
    assert_eq!(
        unary(
            UnaryFn::Rint,
            &f64s(&[Some(2.5), Some(3.5), Some(-2.5)]),
            &c
        )
        .values()
        .to_vec(),
        [2.0, 4.0, -2.0]
    );
    assert_eq!(
        unary(UnaryFn::Cot, &f64s(&[Some(0.0)]), &c).value(0),
        f64::INFINITY
    );
    assert_eq!(
        unary(UnaryFn::Degrees, &f64s(&[Some(1.0)]), &c).value(0),
        57.295_779_513_082_32
    );
    assert_eq!(
        unary(UnaryFn::Radians, &f64s(&[Some(1.0)]), &c).value(0),
        0.017_453_292_519_943_295
    );
    // Recorded from Spark.
    assert_eq!(
        unary(UnaryFn::Atanh, &f64s(&[Some(1.0), Some(2.0)]), &c).value(0),
        f64::INFINITY
    );
    assert!(
        unary(UnaryFn::Acosh, &f64s(&[Some(0.5)]), &c)
            .value(0)
            .is_nan()
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "libm calls inline assembly (sqrtsd), which Miri does not run; the math kernels contain no unsafe code"
)]
fn asinh_and_acosh_follow_the_spark_version() {
    let x = f64s(&[Some(-1.0e17), Some(f64::NEG_INFINITY), Some(2.0)]);
    for v in [SparkVersion::V4_0, SparkVersion::V4_1] {
        let a = unary(UnaryFn::Asinh, &x, &ctx(v));
        assert_eq!(
            (a.value(0), a.value(1)),
            (f64::NEG_INFINITY, f64::NEG_INFINITY)
        );
    }
    let a = unary(UnaryFn::Asinh, &x, &ctx(SparkVersion::V4_2));
    assert!((a.value(0) + 40.0).abs() < 1.0);
    // acosh of a large argument: the naive formula (4.0, 4.1) and the accurate one differ.
    let big = f64s(&[Some(8_607_221_536.235_134)]);
    let old = unary(UnaryFn::Acosh, &big, &ctx(SparkVersion::V4_0)).value(0);
    let new = unary(UnaryFn::Acosh, &big, &ctx(SparkVersion::V4_2)).value(0);
    assert_eq!((old, new), (23.569_014_581_919_38, 23.569_014_581_919_376));
}

#[test]
#[cfg_attr(
    miri,
    ignore = "libm calls inline assembly (sqrtsd), which Miri does not run; the math kernels contain no unsafe code"
)]
fn binary_functions_and_shapes() {
    let a = f64s(&[Some(2.0), None, Some(0.0)]);
    let b = f64s(&[Some(10.0), Some(1.0), Some(-1.0)]);
    let p = binary(BinaryFn::Pow, Operand::Array(&a), Operand::Array(&b)).unwrap();
    assert_eq!(
        (p.value(0), p.is_null(1), p.value(2)),
        (1024.0, true, f64::INFINITY)
    );
    // A scalar operand broadcasts; two scalars make one row.
    let q = binary(
        BinaryFn::Hypot,
        Operand::Scalar(Some(3.0)),
        Operand::Array(&f64s(&[Some(4.0), None])),
    )
    .unwrap();
    assert_eq!((q.value(0), q.is_null(1)), (5.0, true));
    let one = binary(
        BinaryFn::Atan2,
        Operand::Scalar(Some(0.0)),
        Operand::Scalar(Some(-1.0)),
    )
    .unwrap();
    assert_eq!((one.len(), one.value(0)), (1, std::f64::consts::PI));
    let none = binary(
        BinaryFn::Pow,
        Operand::Scalar(None),
        Operand::Scalar(Some(1.0)),
    )
    .unwrap();
    assert!(none.is_null(0));
    let short = f64s(&[Some(1.0)]);
    let e = binary(BinaryFn::Pow, Operand::Array(&a), Operand::Array(&short)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // Java's pow: a NaN exponent or |1| ** infinity is NaN, ** 0 is 1.
    let pow = |x: f64, y: f64| BinaryFn::Pow.apply(x, y).unwrap();
    assert!(
        pow(1.0, f64::NAN).is_nan()
            && pow(1.0, f64::INFINITY).is_nan()
            && pow(-1.0, f64::NEG_INFINITY).is_nan()
    );
    assert_eq!((pow(f64::NAN, 0.0), pow(0.0, -1.0)), (1.0, f64::INFINITY));
    // log(base, x): NULL unless both are positive; log(1, x) divides by zero.
    let log = |b: f64, x: f64| BinaryFn::Log.apply(b, x);
    assert_eq!(
        (log(0.0, 10.0), log(-2.0, 8.0), log(2.0, -0.0)),
        (None, None, None)
    );
    assert_eq!(
        (log(2.0, 8.0), log(1.0, 10.0)),
        (Some(3.0), Some(f64::INFINITY))
    );
    assert!(log(1.0, 1.0).unwrap().is_nan());
}

#[test]
fn integral_rounding() {
    let ansi = OverflowMode::AnsiError;
    let wrap = OverflowMode::JavaWrap;
    let up = RoundMode::HalfUp;
    let even = RoundMode::HalfEven;
    let a = Int32Array::from(vec![
        Some(5),
        Some(15),
        Some(25),
        Some(-5),
        Some(-15),
        Some(125),
        None,
    ]);
    let r = |m, s| {
        round_integral(&a, s, m, wrap)
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
    };
    assert_eq!(
        r(up, -1),
        [
            Some(10),
            Some(20),
            Some(30),
            Some(-10),
            Some(-20),
            Some(130),
            None
        ]
    );
    assert_eq!(
        r(even, -1),
        [
            Some(0),
            Some(20),
            Some(20),
            Some(0),
            Some(-20),
            Some(120),
            None
        ]
    );
    // A non-negative scale is the identity; so is any scale for 0.
    assert_eq!(r(up, 3), a.iter().collect::<Vec<_>>());
    assert_eq!(
        round_integral(&Int8Array::from(vec![-128, 5]), -40, up, ansi)
            .unwrap()
            .values()
            .to_vec(),
        [0, 0]
    );
    // Overflow: Spark raises ARITHMETIC_OVERFLOW ("Overflow") under ANSI and wraps otherwise.
    let big = Int32Array::from(vec![2_147_483_647, 1]);
    let e = round_integral(&big, -1, up, ansi).unwrap_err();
    assert_eq!(
        (e.error_class(), e.row()),
        (ErrorClass::ArithmeticOverflow, Some(0))
    );
    assert!(e.to_string().contains("Overflow"), "{e}");
    assert_eq!(
        round_integral(&big, -1, up, wrap)
            .unwrap()
            .values()
            .to_vec(),
        [-2_147_483_646, 0]
    );
    // A NULL row never raises.
    let nulled = Int32Array::from(vec![None, Some(1)]);
    assert!(round_integral(&nulled, -1, up, ansi).is_ok());
    let wide = Int64Array::from(vec![i64::MAX, i64::MIN]);
    assert_eq!(
        round_integral(&wide, -1, up, wrap)
            .unwrap()
            .values()
            .to_vec(),
        [-9_223_372_036_854_775_806, 9_223_372_036_854_775_806]
    );
    assert_eq!(
        round_integral(&Int8Array::from(vec![-128]), -2, up, ansi)
            .unwrap()
            .value(0),
        -100
    );
}

#[test]
fn floating_point_rounding_rounds_the_decimal_text() {
    let x = f64s(&[
        Some(2.675),
        Some(1.005),
        Some(8.345),
        Some(0.285),
        Some(-0.5),
        Some(0.5),
        Some(2.5),
        Some(-2.5),
        None,
    ]);
    let up = rf64(&x, 2, RoundMode::HalfUp);
    assert_eq!(
        up.iter().take(4).collect::<Vec<_>>(),
        [Some(2.68), Some(1.01), Some(8.35), Some(0.29)]
    );
    let up0 = rf64(&x, 0, RoundMode::HalfUp);
    assert_eq!(
        up0.iter().skip(4).take(4).collect::<Vec<_>>(),
        [Some(-1.0), Some(1.0), Some(3.0), Some(-3.0)]
    );
    let even0 = rf64(&x, 0, RoundMode::HalfEven);
    assert_eq!(
        even0.iter().skip(4).take(4).collect::<Vec<_>>(),
        [Some(0.0), Some(0.0), Some(2.0), Some(-2.0)]
    );
    assert_eq!(rf64(&x, 2, RoundMode::HalfEven).value(2), 8.34);
    assert!(up0.is_null(8));
    // A zero result is +0.0 whatever the sign of the input (BigDecimal).
    let z = rf64(&f64s(&[Some(-0.4), Some(-0.0)]), 0, RoundMode::HalfUp);
    assert!(z.values().iter().all(|v| v.to_bits() == 0));
    let odd = f64s(&[
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(1e300),
        Some(1.5),
        Some(123.456),
        Some(1_234_567.891),
        Some(1e-10),
        Some(9.99e22),
    ]);
    let r = rf64(&odd, 2, RoundMode::HalfUp);
    assert!(r.value(0).is_nan() && r.value(1) == f64::INFINITY);
    assert_eq!(
        (r.value(2), r.value(3), r.value(4), r.value(6)),
        (1e300, 1.5, 123.46, 0.0)
    );
    assert_eq!(rf64(&odd, 400, RoundMode::HalfUp).value(3), 1.5);
    assert_eq!(rf64(&odd, -1, RoundMode::HalfUp).value(4), 120.0);
    assert_eq!(rf64(&odd, -3, RoundMode::HalfUp).value(5), 1_235_000.0);
    // Carrying: 9.99 rounds to 10.0 at one digit.
    assert_eq!(
        rf64(
            &f64s(&[Some(9.99), Some(99.5), Some(0.96)]),
            1,
            RoundMode::HalfUp
        )
        .values()
        .to_vec(),
        [10.0, 99.5, 1.0]
    );
    assert_eq!(
        rf64(&f64s(&[Some(0.04), Some(5e-324)]), 1, RoundMode::HalfUp)
            .values()
            .to_vec(),
        [0.0, 0.0]
    );
    // A float is rounded through its exact double value.
    let f = Float32Array::from(vec![2.675_f32, 8.345, f32::NAN]);
    let r = rf32(&f, 2, RoundMode::HalfUp);
    assert_eq!((r.value(0), r.value(1)), (2.67_f32, 8.35_f32));
    assert!(r.value(2).is_nan());
}

#[test]
fn a_scale_far_below_the_value_fails_like_big_decimal() {
    let up = RoundMode::HalfUp;
    let msg = |e: krishiv_kernels::KernelError| (e.error_class(), e.row(), e.to_string());
    let overflow = ErrorClass::ArithmeticOverflow;
    // Recorded from Spark 4.0-4.2. An integral value: scale difference past int, or past BigInteger.
    let ints = Int32Array::from(vec![Some(25), None, Some(0)]);
    for (scale, text) in [
        (i32::MIN, "Underflow"),
        (-1_000_000_000, "BigInteger would overflow supported range"),
    ] {
        let (class, row, shown) =
            msg(round_integral(&ints, scale, up, OverflowMode::JavaWrap).unwrap_err());
        assert_eq!((class, row), (overflow, Some(0)));
        assert!(shown.contains(text), "{shown}");
    }
    // Zeros and NULLs never raise; a scale just inside the limit is a plain 0.
    let zeros = Int32Array::from(vec![Some(0), None]);
    assert_eq!(
        round_integral(&zeros, i32::MIN, up, OverflowMode::AnsiError)
            .unwrap()
            .value(0),
        0
    );
    assert_eq!(
        round_integral(&ints, -536_870_919, up, OverflowMode::JavaWrap)
            .unwrap()
            .value(0),
        0
    );
    // Doubles: BigDecimal.valueOf(2.5) has scale 1, so INT_MIN + 1 already underflows.
    let x = f64s(&[
        Some(2.5),
        None,
        Some(0.0),
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(1e10),
    ]);
    let e = round_f64(&x, i32::MIN, up).unwrap_err();
    assert_eq!(msg(e).1, Some(0));
    let only_special = f64s(&[Some(0.0), Some(f64::NAN), Some(f64::INFINITY), None]);
    assert!(round_f64(&only_special, i32::MIN, up).is_ok());
    // 1e10 prints as 1.0E10: scale -9, so the difference still fits an int.
    let big = f64s(&[Some(1e10)]);
    let e = msg(round_f64(&big, i32::MIN, up).unwrap_err());
    assert!(e.2.contains("BigInteger would overflow"), "{}", e.2);
    let tiny = f64s(&[Some(1e-300)]);
    assert!(
        msg(round_f64(&tiny, -5, up)
            .and(round_f64(&tiny, i32::MIN, up))
            .unwrap_err())
        .2
        .contains("Underflow")
    );
    let floats = Float32Array::from(vec![1.5_f32]);
    assert!(round_f32(&floats, i32::MIN, up).is_err());
    // Decimals: the scale of the column counts.
    let d = dec(&[Some(125), None], 5, 2);
    for (scale, text) in [
        (-2_147_483_646, "Underflow"),
        (-2_147_483_645, "BigInteger would overflow supported range"),
        (-600_000_000, "BigInteger would overflow supported range"),
    ] {
        let shown = msg(round_decimal(&d, scale, DecimalRounding::HalfUp).unwrap_err()).2;
        assert!(shown.contains(text), "{scale}: {shown}");
    }
    let whole = dec(&[Some(125)], 5, 0);
    assert!(
        msg(round_decimal(&whole, -2_147_483_647, DecimalRounding::Floor).unwrap_err())
            .2
            .contains("BigInteger")
    );
    assert!(round_decimal(&dec(&[Some(0)], 5, 2), i32::MIN, DecimalRounding::Ceiling).is_ok());
    assert_eq!(
        round_decimal(&whole, -300, DecimalRounding::HalfUp)
            .unwrap()
            .value(0),
        0
    );
}

#[test]
fn decimal_result_types_match_spark() {
    // Recorded from Spark: decimal(p,s) -> the result "p.s" for the scales below.
    let scales = [-3, -1, 0, 1, 2, 5, 20, 40];
    let table = [
        ((10, 2), "9.0 9.0 9.0 10.1 11.2 11.2 11.2 11.2"),
        ((10, 0), "11.0 11.0 11.0 11.0 11.0 11.0 11.0 11.0"),
        ((5, 5), "4.0 2.0 1.0 2.1 3.2 6.5 6.5 6.5"),
        ((38, 18), "21.0 21.0 21.0 22.1 23.2 26.5 38.18 38.18"),
        ((38, 0), "38.0 38.0 38.0 38.0 38.0 38.0 38.0 38.0"),
        ((3, 1), "4.0 3.0 3.0 4.1 4.1 4.1 4.1 4.1"),
        ((20, 10), "11.0 11.0 11.0 12.1 13.2 16.5 21.10 21.10"),
        ((1, 0), "4.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0"),
        ((38, 38), "4.0 2.0 1.0 2.1 3.2 6.5 21.20 38.38"),
    ];
    for ((p, s), want) in table {
        for (scale, w) in scales.iter().zip(want.split(' ')) {
            let (wp, ws) = w.split_once('.').unwrap();
            let w = (wp.parse::<u8>().unwrap(), ws.parse::<i8>().unwrap());
            assert_eq!(
                round_decimal_type(p, s, *scale),
                w,
                "decimal({p},{s}) scale {scale}"
            );
        }
    }
    // The smallest scale: `-scale + 1` wraps in Spark's Int arithmetic.
    assert_eq!(round_decimal_type(2, 1, i32::MIN), (2, 0));
}

#[test]
fn decimal_rounding_modes() {
    let a = dec(
        &[
            Some(250),
            Some(-250),
            Some(235),
            Some(-235),
            Some(149),
            None,
        ],
        3,
        2,
    );
    let to = |m, d| round_decimal(&a, d, m).unwrap().iter().collect::<Vec<_>>();
    assert_eq!(
        to(DecimalRounding::HalfUp, 1),
        [Some(25), Some(-25), Some(24), Some(-24), Some(15), None]
    );
    assert_eq!(
        to(DecimalRounding::HalfEven, 1),
        [Some(25), Some(-25), Some(24), Some(-24), Some(15), None]
    );
    assert_eq!(
        to(DecimalRounding::HalfUp, 0),
        [Some(3), Some(-3), Some(2), Some(-2), Some(1), None]
    );
    assert_eq!(
        to(DecimalRounding::HalfEven, 0),
        [Some(2), Some(-2), Some(2), Some(-2), Some(1), None]
    );
    assert_eq!(
        to(DecimalRounding::Ceiling, 0),
        [Some(3), Some(-2), Some(3), Some(-2), Some(2), None]
    );
    assert_eq!(
        to(DecimalRounding::Floor, 0),
        [Some(2), Some(-3), Some(2), Some(-3), Some(1), None]
    );
    // A scale at or above the input's keeps the digits (only the type changes).
    assert_eq!(to(DecimalRounding::HalfUp, 5), a.iter().collect::<Vec<_>>());
    let wide = round_decimal(&dec(&[Some(12_345)], 5, 0), -2, DecimalRounding::Ceiling).unwrap();
    assert_eq!(
        (wide.value(0), wide.precision(), wide.scale()),
        (12_400, 6, 0)
    );
    let tiny = dec(&[Some(7), Some(-7)], 1, 0);
    let huge = round_decimal(&tiny, -40, DecimalRounding::HalfUp).unwrap();
    assert_eq!(huge.values().to_vec(), [0, 0]);
    // Beyond 10^38 a ceiling still moves one unit: 10^40 does not fit decimal(38, 0).
    let e = round_decimal(&tiny, -40, DecimalRounding::Ceiling).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::NumericValueOutOfRangeWithoutSuggestion
    );
    assert!(
        e.to_string()
            .contains("rounded half up from 1E+40 cannot be represented as Decimal(38, 0)"),
        "{e}"
    );
    assert_eq!(
        round_decimal(&tiny, -40, DecimalRounding::Floor)
            .unwrap_err()
            .row(),
        Some(1)
    );
}

#[test]
fn decimal_rounding_overflow_matches_spark() {
    let nines = "9".repeat(38).parse::<i128>().unwrap();
    let a = dec(&[Some(nines), Some(-nines)], 38, 0);
    for (mode, d) in [
        (DecimalRounding::HalfUp, -1),
        (DecimalRounding::Ceiling, -1),
    ] {
        let e = round_decimal(&a, d, mode).unwrap_err();
        assert_eq!(
            (e.error_class(), e.row()),
            (ErrorClass::NumericValueOutOfRangeWithoutSuggestion, Some(0))
        );
        // Spark: "The 1000...0 rounded half up from 1.000...E+38 cannot be represented as Decimal(38, 0)."
        assert!(
            e.to_string().contains(
                "The 100000000000000000000000000000000000000 rounded half up from 1.0000000000000000000000000000000000000E+38 cannot be represented as Decimal(38, 0)."
            ),
            "{e}"
        );
    }
    let e = round_decimal(&dec(&[Some(-nines)], 38, 0), -1, DecimalRounding::Floor).unwrap_err();
    assert!(e.to_string().contains("The -100000000000000000000000000000000000000 rounded half up from -1.0000000000000000000000000000000000000E+38"), "{e}");
    // With a scale: 9999.99 -> 10000.0 needs six digits; the type has them.
    let ok = round_decimal(&dec(&[Some(999_999)], 6, 2), 1, DecimalRounding::HalfUp).unwrap();
    assert_eq!((ok.value(0), ok.precision(), ok.scale()), (100_000, 6, 1));
    // Not a decimal column / negative scale input types.
    let bad = Decimal128Array::from(vec![1_i128])
        .with_precision_and_scale(5, -2)
        .unwrap();
    assert!(round_decimal(&bad, 0, DecimalRounding::HalfUp).is_err());
}

#[test]
fn ceil_and_floor_of_doubles_saturate() {
    let x = f64s(&[
        Some(1.1),
        Some(f64::NAN),
        Some(-1e30),
        Some(f64::INFINITY),
        Some(-0.5),
        None,
    ]);
    let c = ceil_floor_to_long(&x, true);
    let want = [
        Some(2),
        Some(0),
        Some(i64::MIN),
        Some(i64::MAX),
        Some(0),
        None,
    ];
    assert_eq!(c.iter().collect::<Vec<_>>(), want);
}

/// An independent oracle: rounds `u / 10^drop` using floor division.
fn oracle(u: i128, drop: u32, mode: DecimalRounding) -> i128 {
    let unit = 10_i128.pow(drop);
    let (q, r) = (u.div_euclid(unit), u.rem_euclid(unit));
    match mode {
        DecimalRounding::Floor => q,
        DecimalRounding::Ceiling => q + i128::from(r > 0),
        DecimalRounding::HalfUp => {
            // Ties go away from zero: up for positive values, down (the floor) for negative.
            let tie_up = u >= 0;
            q + i128::from(2 * r > unit || (2 * r == unit && tie_up))
        }
        DecimalRounding::HalfEven => q + i128::from(2 * r > unit || (2 * r == unit && q % 2 != 0)),
    }
}

fn config() -> ProptestConfig {
    // Miri interprets each case, so it runs two; no failure file (Miri's
    // isolation has no file system).
    let default = ProptestConfig::default();
    ProptestConfig {
        cases: if cfg!(miri) { 2 } else { default.cases },
        failure_persistence: None,
        ..default
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn decimal_rounding_agrees_with_floor_arithmetic(units in any::<i64>(), s in 1_i8..=18, d in 0_i32..=18) {
        let a = dec(&[Some(i128::from(units))], 38, s);
        for mode in [DecimalRounding::HalfUp, DecimalRounding::HalfEven, DecimalRounding::Ceiling, DecimalRounding::Floor] {
            let got = round_decimal(&a, d, mode).unwrap().value(0);
            let want = if d >= i32::from(s) { i128::from(units) } else { oracle(i128::from(units), u32::try_from(i32::from(s) - d).unwrap(), mode) };
            prop_assert_eq!(got, want, "{:?} decimal(38,{}) {} to {}", mode, s, units, d);
        }
    }

    #[test]
    fn decimal_rounding_to_tens_agrees_with_floor_arithmetic(units in any::<i64>(), s in 0_i8..=10, d in 1_i32..=12) {
        let a = dec(&[Some(i128::from(units))], 38, s);
        for mode in [DecimalRounding::HalfUp, DecimalRounding::HalfEven, DecimalRounding::Ceiling, DecimalRounding::Floor] {
            let got = round_decimal(&a, -d, mode).unwrap().value(0);
            let drop = u32::try_from(i32::from(s) + d).unwrap();
            let want = oracle(i128::from(units), drop, mode) * 10_i128.pow(u32::try_from(d).unwrap());
            prop_assert_eq!(got, want);
        }
    }

    #[test]
    fn rounding_a_double_to_its_own_digits_is_the_identity(v in -1.0e15_f64..1.0e15, d in 0_i32..=15) {
        let x = f64s(&[Some(v)]);
        let r = rf64(&x, d, RoundMode::HalfUp).value(0);
        // Rounding to d digits twice changes nothing, and moves the value by at most half a unit.
        let again = rf64(&f64s(&[Some(r)]), d, RoundMode::HalfUp).value(0);
        prop_assert_eq!(r.to_bits(), again.to_bits());
        prop_assert!((r - v).abs() <= 0.5 * 10_f64.powi(-d) * (1.0 + 1e-9) + v.abs() * 1e-15);
    }
}
