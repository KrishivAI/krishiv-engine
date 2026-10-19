// SPDX-License-Identifier: Apache-2.0
//! Targeted Spark-semantics and dispatch tests for the kernel framework.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{
    Array, BooleanArray, Decimal128Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array,
};
use arrow_buffer::NullBuffer;
use krishiv_common::error::ErrorClass;
use krishiv_kernels::compare::{Equal, Less, compare};
use krishiv_kernels::decimal::{DecimalPlan, DecimalStorage, decimal_binary};
use krishiv_kernels::integer::{UnaryOp, binary, integral_divide, remainder, unary};
use krishiv_kernels::{
    Add, AnsiError, ArithOp, JavaWrap, KernelCtx, Mul, NullOnOverflow, Operand, SparkVersion, Sub,
    Target, Unchecked,
};

fn ctx() -> KernelCtx {
    KernelCtx::new()
}

#[test]
fn null_rows_never_raise() {
    // Row 1 would overflow but is NULL; the values under it are arbitrary.
    let a = Int64Array::from(vec![Some(1), None, Some(2)]);
    let b = Int64Array::from(vec![1, i64::MAX, 3]);
    let out = binary::<Int64Type, Add, AnsiError>(Operand::Array(&a), Operand::Array(&b), &ctx());
    assert_eq!(out.unwrap(), Int64Array::from(vec![Some(2), None, Some(5)]));
    // A NULL slot whose hidden value overflows, in a sliced array.
    let hidden = Int64Array::new(
        vec![0, i64::MAX, 0].into(),
        Some(vec![true, false, true].into()),
    );
    let s = hidden.slice(1, 2);
    let out =
        binary::<Int64Type, Add, AnsiError>(Operand::Array(&s), Operand::Scalar(Some(1)), &ctx());
    assert_eq!(out.unwrap(), Int64Array::from(vec![None, Some(1)]));
}

#[test]
fn ansi_reports_the_first_failing_row() {
    let mut v = vec![1_i32; 200];
    v[70] = i32::MAX;
    v[150] = i32::MAX;
    let a = Int32Array::from(v);
    for t in Target::available() {
        let c = KernelCtx::with_target(t).unwrap();
        let e =
            binary::<Int32Type, Add, AnsiError>(Operand::Array(&a), Operand::Scalar(Some(1)), &c)
                .unwrap_err();
        assert_eq!(e.row(), Some(70));
        assert_eq!(e.error_class(), ErrorClass::ArithmeticOverflow);
        assert!(e.to_string().contains("try_add"));
        let e =
            binary::<Int32Type, Sub, AnsiError>(Operand::Scalar(Some(-2)), Operand::Array(&a), &c)
                .unwrap_err();
        assert_eq!(e.row(), Some(70));
        assert!(e.to_string().contains("try_subtract"));
    }
}

#[test]
fn policies_differ_only_on_overflow() {
    let a = Int64Array::from(vec![i64::MAX, 2]);
    let two = Operand::Scalar(Some(2));
    let wrap = binary::<Int64Type, Mul, JavaWrap>(Operand::Array(&a), two, &ctx()).unwrap();
    assert_eq!(wrap, Int64Array::from(vec![-2, 4]));
    let unchecked = binary::<Int64Type, Mul, Unchecked>(Operand::Array(&a), two, &ctx()).unwrap();
    assert_eq!(unchecked, wrap);
    let null = binary::<Int64Type, Mul, NullOnOverflow>(Operand::Array(&a), two, &ctx()).unwrap();
    assert_eq!(null, Int64Array::from(vec![None, Some(4)]));
}

#[test]
fn nulls_path_matches_the_no_null_path() {
    use krishiv_kernels::integer::binary_nulls_path;
    let a = Int64Array::from(vec![1, i64::MAX, 3]);
    let one = Operand::Scalar(Some(1));
    let fast = binary::<Int64Type, Add, NullOnOverflow>(Operand::Array(&a), one, &ctx()).unwrap();
    let nulls =
        binary_nulls_path::<Int64Type, Add, NullOnOverflow>(Operand::Array(&a), one, &ctx());
    assert_eq!(nulls.unwrap(), fast);
    let e = binary_nulls_path::<Int64Type, Add, AnsiError>(Operand::Array(&a), one, &ctx());
    assert_eq!(e.unwrap_err().row(), Some(1));
    let all_null = binary_nulls_path::<Int64Type, Add, AnsiError>(
        Operand::Array(&a),
        Operand::Scalar(None),
        &ctx(),
    );
    assert_eq!(all_null.unwrap().null_count(), 3);
}

#[test]
fn null_scalar_and_engine_errors() {
    let a = Int64Array::from(vec![1, 2, 3]);
    let out =
        binary::<Int64Type, Add, AnsiError>(Operand::Array(&a), Operand::Scalar(None), &ctx())
            .unwrap();
    assert_eq!(out.null_count(), 3);
    let short = Int64Array::from(vec![1]);
    let e = binary::<Int64Type, Add, AnsiError>(Operand::Array(&a), Operand::Array(&short), &ctx())
        .unwrap_err();
    assert_eq!(
        (e.error_class(), e.row()),
        (ErrorClass::InternalError, None)
    );
    let e = binary::<Int64Type, Add, AnsiError>(
        Operand::Scalar(Some(1)),
        Operand::Scalar(Some(1)),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let f = Float64Array::from(vec![1.0]);
    let n = compare::<_, Equal>(Operand::Array(&f), Operand::Scalar(None), &ctx()).unwrap();
    assert_eq!(n, BooleanArray::new_null(1));
}

#[test]
fn spark_float_semantics_differ_from_ieee() {
    let a = Float64Array::from(vec![f64::NAN, -0.0, 1.0, f64::NAN, f64::INFINITY]);
    let b = Float64Array::from(vec![f64::NAN, 0.0, f64::NAN, 1.0, f64::NAN]);
    let eq = compare::<_, Equal>(Operand::Array(&a), Operand::Array(&b), &ctx()).unwrap();
    assert_eq!(
        eq,
        BooleanArray::from(vec![true, true, false, false, false])
    );
    let lt = compare::<_, Less>(Operand::Array(&a), Operand::Array(&b), &ctx()).unwrap();
    assert_eq!(
        lt,
        BooleanArray::from(vec![false, false, true, false, true])
    );
}

fn dec(v: Vec<Option<i128>>, p: u8, s: i8) -> Decimal128Array {
    Decimal128Array::from(v)
        .with_precision_and_scale(p, s)
        .unwrap()
}

#[test]
fn decimal_out_of_range_matches_spark() {
    // 99999999999999999999999999999999999.999 + 0.001 at Decimal(38, 3).
    let max = 10_i128.pow(38) - 1;
    let a = dec(vec![Some(1), None, Some(max)], 38, 3);
    let b = dec(vec![Some(1), Some(1), Some(1)], 38, 3);
    let plan = DecimalPlan::new(ArithOp::Add, (38, 3), (38, 3), (38, 3)).unwrap();
    let e = decimal_binary::<Add, AnsiError>(Operand::Array(&a), Operand::Array(&b), &plan, &ctx())
        .unwrap_err();
    assert_eq!(e.row(), Some(2));
    assert_eq!(
        e.error_class(),
        ErrorClass::NumericValueOutOfRangeWithSuggestion
    );
    assert_eq!(
        e.to_string(),
        "[NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION]  100000000000000000000000000000000000.000 \
         cannot be represented as Decimal(38, 3). If necessary set \"spark.sql.ansi.enabled\" to \
         \"false\" to bypass this error, and return NULL instead. SQLSTATE: 22003"
    );
    let n = decimal_binary::<Add, NullOnOverflow>(
        Operand::Array(&a),
        Operand::Array(&b),
        &plan,
        &ctx(),
    )
    .unwrap();
    assert_eq!(n, dec(vec![Some(2), None, None], 38, 3));
}

#[test]
fn decimal_i64_hook_matches_i128_and_rescales() {
    // 1.5 (10,1) * 2.25 (10,2) = 3.375 (21,3); 1.5 + 2.25 = 3.75 (12,2).
    let a = dec(vec![Some(15), Some(-15)], 10, 1);
    let b = dec(vec![Some(225), None], 10, 2);
    let add = DecimalPlan::new(ArithOp::Add, (10, 1), (10, 2), (12, 2)).unwrap();
    assert_eq!(add.storage(), DecimalStorage::I64);
    for p in [add, add.with_storage(DecimalStorage::I128).unwrap()] {
        let r =
            decimal_binary::<Add, Unchecked>(Operand::Array(&a), Operand::Array(&b), &p, &ctx())
                .unwrap();
        assert_eq!(r, dec(vec![Some(375), None], 12, 2));
    }
    let sub = DecimalPlan::new(ArithOp::Subtract, (10, 1), (10, 2), (12, 2)).unwrap();
    let r = decimal_binary::<Sub, AnsiError>(
        Operand::Array(&a),
        Operand::Scalar(Some(5)),
        &sub,
        &ctx(),
    )
    .unwrap();
    assert_eq!(r, dec(vec![Some(145), Some(-155)], 12, 2));
    let mul = DecimalPlan::new(ArithOp::Multiply, (10, 1), (10, 2), (21, 3)).unwrap();
    let r = decimal_binary::<Mul, AnsiError>(Operand::Array(&a), Operand::Array(&b), &mul, &ctx())
        .unwrap();
    assert_eq!(r, dec(vec![Some(3375), None], 21, 3));
    let none =
        decimal_binary::<Mul, AnsiError>(Operand::Scalar(None), Operand::Array(&b), &mul, &ctx())
            .unwrap();
    assert_eq!(none.null_count(), 2);
    assert_eq!(none.precision(), 21);
    // An input outside its declared precision is settled exactly, not by
    // the 64-bit fast path.
    let wide = Decimal128Array::from(vec![i128::from(i64::MAX) + 1])
        .with_precision_and_scale(10, 1)
        .unwrap();
    let e = decimal_binary::<Add, AnsiError>(
        Operand::Array(&wide),
        Operand::Scalar(Some(0)),
        &add,
        &ctx(),
    );
    assert_eq!(
        e.unwrap_err().error_class(),
        ErrorClass::NumericValueOutOfRangeWithSuggestion
    );
}

#[test]
fn decimal_flagged_but_in_range_is_not_an_error() {
    // 18 at scale 0 rescaled to scale 37 overflows i128 (1.8e38), yet
    // 18 + (-9.9...) = 8.1 fits Decimal(38, 37): the slow path decides.
    let plan = DecimalPlan::new(ArithOp::Add, (2, 0), (38, 37), (38, 37)).unwrap();
    let a = dec(vec![Some(18)], 2, 0);
    let b_val = -99 * 10_i128.pow(36);
    let b = dec(vec![Some(b_val)], 38, 37);
    let r = decimal_binary::<Add, AnsiError>(Operand::Array(&a), Operand::Array(&b), &plan, &ctx())
        .unwrap();
    assert_eq!(r.value(0), 81 * 10_i128.pow(36));
}

#[test]
fn dispatch_baseline_and_best_agree() {
    let required = std::env::var("KRISHIV_REQUIRE_TARGETS").unwrap_or_default();
    Target::check_required(&required).unwrap();
    let n = if cfg!(miri) { 300 } else { 100_000 };
    let a: Int64Array = (0..n).map(|i| (i % 7 != 0).then_some(i * 7919)).collect();
    let b: Int64Array = (0..n).map(|i| Some(i64::MAX - i * 13)).collect();
    let run = |t: Target| {
        let c = KernelCtx::with_target(t).unwrap();
        (
            binary::<Int64Type, Add, NullOnOverflow>(Operand::Array(&a), Operand::Array(&b), &c)
                .unwrap(),
            binary::<Int64Type, Sub, JavaWrap>(Operand::Array(&a), Operand::Array(&b), &c).unwrap(),
        )
    };
    let base = run(Target::Baseline);
    let best = run(Target::best());
    assert_eq!(base, best);
    assert!(base.0.null_count() > a.null_count());
}

/// Spark's ANSI suffix of every `ARITHMETIC_OVERFLOW` message.
const BYPASS: &str = "If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error. SQLSTATE: 22003";

/// Values and messages recorded from Spark 4.0, 4.1 and 4.2.
#[test]
fn integral_arithmetic_matches_spark() {
    let v40 = ctx().with_spark_version(SparkVersion::V4_0);
    let v41 = ctx().with_spark_version(SparkVersion::V4_1);
    // tinyint/smallint raise BINARY_ARITHMETIC_OVERFLOW in every version.
    let b = Int8Array::from(vec![127_i8, -128]);
    let e = binary::<Int8Type, Add, AnsiError>(Operand::Array(&b), Operand::Scalar(Some(1)), &v40);
    assert_eq!(
        e.unwrap_err().to_string(),
        "[BINARY_ARITHMETIC_OVERFLOW] 127S + 1S caused overflow. Use `try_add` to ignore overflow \
         problem and return NULL. SQLSTATE: 22003"
    );
    let e =
        binary::<Int8Type, Mul, AnsiError>(Operand::Array(&b), Operand::Scalar(Some(-1)), &ctx());
    assert!(
        e.unwrap_err()
            .to_string()
            .contains("-128S * -1S caused overflow")
    );
    let s = Int16Array::from(vec![32767_i16]);
    let w =
        binary::<Int16Type, Add, JavaWrap>(Operand::Array(&s), Operand::Scalar(Some(1)), &ctx());
    assert_eq!(w.unwrap(), Int16Array::from(vec![i16::MIN]));
    // int/bigint: the JDK's text before 4.2.
    let i = Int32Array::from(vec![i32::MAX]);
    let add = |c: &KernelCtx| {
        binary::<Int32Type, Add, AnsiError>(Operand::Array(&i), Operand::Scalar(Some(1)), c)
            .unwrap_err()
            .to_string()
    };
    let tail = format!("Use 'try_add' to tolerate overflow and return NULL instead. {BYPASS}");
    assert_eq!(
        add(&v40),
        format!("[ARITHMETIC_OVERFLOW] integer overflow. {tail}")
    );
    assert_eq!(
        add(&ctx()),
        format!("[ARITHMETIC_OVERFLOW] overflow. {tail}")
    );
    let l = Int64Array::from(vec![i64::MAX]);
    let e = binary::<Int64Type, Mul, AnsiError>(Operand::Array(&l), Operand::Scalar(Some(2)), &v41);
    assert!(
        e.unwrap_err()
            .to_string()
            .starts_with("[ARITHMETIC_OVERFLOW] long overflow. Use 'try_multiply'")
    );
}

#[test]
fn remainder_and_div_match_spark() {
    let a = Int32Array::from(vec![-7, 7, i32::MIN, 1]);
    let b = Int32Array::from(vec![3, -3, -1, 0]);
    let wrap = remainder::<Int32Type, JavaWrap>(Operand::Array(&a), Operand::Array(&b), &ctx());
    assert_eq!(
        wrap.unwrap(),
        Int32Array::from(vec![Some(-1), Some(1), Some(0), None])
    );
    let e = remainder::<Int32Type, AnsiError>(Operand::Array(&a), Operand::Array(&b), &ctx())
        .unwrap_err();
    assert_eq!(
        (e.row(), e.error_class()),
        (Some(3), ErrorClass::RemainderByZero)
    );
    assert!(
        e.to_string()
            .contains("Use `try_mod` to tolerate divisor being 0")
    );
    let v40 = ctx().with_spark_version(SparkVersion::V4_0);
    let e = remainder::<Int32Type, AnsiError>(Operand::Array(&a), Operand::Array(&b), &v40);
    assert_eq!(e.unwrap_err().error_class(), ErrorClass::DivideByZero);
    let x = Int64Array::from(vec![7, -7, i64::MIN, 1]);
    let y = Int64Array::from(vec![-2, 2, -1, 0]);
    let wrap = integral_divide::<JavaWrap>(Operand::Array(&x), Operand::Array(&y), &ctx());
    assert_eq!(
        wrap.unwrap(),
        Int64Array::from(vec![Some(-3), Some(-3), Some(i64::MIN), None])
    );
    let e =
        integral_divide::<AnsiError>(Operand::Array(&x), Operand::Array(&y), &ctx()).unwrap_err();
    assert_eq!(
        e.to_string(),
        format!(
            "[ARITHMETIC_OVERFLOW] Overflow in integral divide. Use 'try_divide' to tolerate \
             overflow and return NULL instead. {BYPASS}"
        )
    );
    let e = integral_divide::<AnsiError>(Operand::Array(&x), Operand::Scalar(Some(0)), &ctx());
    assert_eq!(e.unwrap_err().error_class(), ErrorClass::DivideByZero);
    let n = integral_divide::<NullOnOverflow>(Operand::Scalar(None), Operand::Array(&y), &ctx());
    assert_eq!(n.unwrap().null_count(), 4);
    let n = remainder::<Int32Type, AnsiError>(Operand::Array(&a), Operand::Scalar(None), &ctx());
    assert_eq!(n.unwrap().null_count(), 4);
}

#[test]
fn unary_minus_and_abs_match_spark() {
    let b = Int8Array::from(vec![Some(5), None, Some(i8::MIN)]);
    let v40 = ctx().with_spark_version(SparkVersion::V4_0);
    let e = unary::<Int8Type, AnsiError>(UnaryOp::Negate, &b, &v40).unwrap_err();
    assert_eq!(e.row(), Some(2));
    assert_eq!(
        e.to_string(),
        format!("[ARITHMETIC_OVERFLOW] byte overflow. {BYPASS}")
    );
    let e = unary::<Int8Type, AnsiError>(UnaryOp::Abs, &b, &ctx()).unwrap_err();
    assert_eq!(
        e.to_string(),
        format!("[ARITHMETIC_OVERFLOW] overflow. {BYPASS}")
    );
    let w = unary::<Int8Type, JavaWrap>(UnaryOp::Negate, &b, &ctx()).unwrap();
    assert_eq!(w, Int8Array::from(vec![Some(-5), None, Some(i8::MIN)]));
    let s = Int16Array::from(vec![i16::MIN]);
    let e = unary::<Int16Type, AnsiError>(UnaryOp::Abs, &s, &v40).unwrap_err();
    assert!(
        e.to_string()
            .starts_with("[ARITHMETIC_OVERFLOW] short overflow.")
    );
    let l = Int64Array::from(vec![-3, 4]);
    let a = unary::<Int64Type, AnsiError>(UnaryOp::Abs, &l, &ctx()).unwrap();
    assert_eq!(a, Int64Array::from(vec![3, 4]));
}

/// Values recorded from Spark 4.2.
#[test]
fn comparisons_match_spark() {
    use arrow_array::{BinaryArray, StringArray};
    use krishiv_kernels::compare::{
        BoolOperand, BytesOperand, Greater, NotEqual, Validity, compare_bool, compare_bytes,
        null_safe,
    };
    // 'abc' < 'abe', 'a' < 'ab', 'é' > 'z', '' < 'a', 'ab' = 'ab ' is false, 'A' < 'a'.
    let l = StringArray::from(vec!["abc", "a", "é", "", "ab", "A"]);
    let r = StringArray::from(vec!["abe", "ab", "z", "a", "ab ", "a"]);
    let lt = compare_bytes::<_, Less>(BytesOperand::Array(&l), BytesOperand::Array(&r)).unwrap();
    assert_eq!(
        lt,
        BooleanArray::from(vec![true, true, false, true, true, true])
    );
    let ne =
        compare_bytes::<_, NotEqual>(BytesOperand::Array(&l), BytesOperand::Scalar(Some(b"ab")));
    assert!(!ne.unwrap().value(4));
    // X'00' < X'FF', X'7F' < X'80': unsigned bytes.
    let x = BinaryArray::from(vec![&[0x00_u8][..], &[0x7F]]);
    let y = BinaryArray::from(vec![&[0xFF_u8][..], &[0x80]]);
    let lt = compare_bytes::<_, Less>(BytesOperand::Array(&x), BytesOperand::Array(&y)).unwrap();
    assert_eq!(lt, BooleanArray::from(vec![true, true]));
    let none = compare_bytes::<_, Less>(BytesOperand::Array(&x), BytesOperand::Scalar(None));
    assert_eq!(none.unwrap().null_count(), 2);
    // true > false; true <=> NULL is false; NULL <=> NULL is true.
    let t = BooleanArray::from(vec![Some(true), None]);
    let gt = compare_bool::<Greater>(BoolOperand::Array(&t), BoolOperand::Scalar(Some(false)));
    assert_eq!(gt.unwrap(), BooleanArray::from(vec![Some(true), None]));
    let eq = compare_bool::<Equal>(BoolOperand::Array(&t), BoolOperand::Scalar(None)).unwrap();
    let ns = null_safe(
        &eq,
        Validity::Array(t.nulls()),
        Validity::Scalar { null: true },
    );
    assert_eq!(ns.unwrap(), BooleanArray::from(vec![false, true]));
    // -0.0 <=> 0.0 and NaN = NaN.
    let f = Float64Array::from(vec![-0.0, f64::NAN]);
    let g = Float64Array::from(vec![0.0, f64::NAN]);
    let eq = compare::<_, Equal>(Operand::Array(&f), Operand::Array(&g), &ctx()).unwrap();
    let ns = null_safe(&eq, Validity::Array(None), Validity::Array(None));
    assert_eq!(ns.unwrap(), BooleanArray::from(vec![true, true]));
    let short = NullBuffer::new_valid(1);
    let e = null_safe(&eq, Validity::Array(Some(&short)), Validity::Array(None));
    assert_eq!(e.unwrap_err().error_class(), ErrorClass::InternalError);
    let e = compare_bool::<Equal>(BoolOperand::Scalar(Some(true)), BoolOperand::Scalar(None));
    assert_eq!(e.unwrap_err().error_class(), ErrorClass::InternalError);
}

/// Values and errors recorded from Spark 4.2.
#[test]
fn float_arithmetic_matches_spark() {
    use arrow_array::types::Float64Type;
    use krishiv_kernels::float;
    let a = Float64Array::from(vec![Some(1.0), Some(f64::NAN), None, Some(-7.5)]);
    let b = Float64Array::from(vec![Some(0.0), Some(1.0), Some(0.0), Some(2.0)]);
    let legacy =
        float::divide::<Float64Type, JavaWrap>(Operand::Array(&a), Operand::Array(&b), &ctx());
    let legacy = legacy.unwrap();
    assert!(legacy.is_null(0) && legacy.value(1).is_nan() && legacy.is_null(2));
    let e = float::divide::<Float64Type, AnsiError>(Operand::Array(&a), Operand::Array(&b), &ctx());
    assert_eq!(e.unwrap_err().error_class(), ErrorClass::DivideByZero);
    let e = float::remainder::<Float64Type, AnsiError>(
        Operand::Array(&a),
        Operand::Scalar(Some(-0.0)),
        &ctx(),
    );
    assert_eq!(e.unwrap_err().error_class(), ErrorClass::RemainderByZero);
    let r =
        float::remainder::<Float64Type, AnsiError>(Operand::Array(&a), Operand::Array(&b), &ctx());
    assert_eq!(r.unwrap_err().row(), Some(0));
    let r = float::remainder::<Float64Type, NullOnOverflow>(
        Operand::Array(&a),
        Operand::Array(&b),
        &ctx(),
    );
    assert_eq!(r.unwrap().value(3), -1.5);
    // Infinity * 0 is NaN; nothing overflows.
    let inf = Float64Array::from(vec![f64::INFINITY, f64::MAX]);
    let m = float::binary(
        ArithOp::Multiply,
        Operand::Array(&inf),
        Operand::Scalar(Some(0.0)),
        &ctx(),
    );
    assert!(m.unwrap().value(0).is_nan());
    let s = float::binary(
        ArithOp::Add,
        Operand::Array(&inf),
        Operand::Scalar(None),
        &ctx(),
    );
    assert_eq!(s.unwrap().null_count(), 2);
    let d = float::binary(
        ArithOp::Subtract,
        Operand::Array(&inf),
        Operand::Array(&inf),
        &ctx(),
    );
    assert_eq!(d.unwrap().value(1), 0.0);
    let n =
        float::divide::<Float64Type, AnsiError>(Operand::Scalar(None), Operand::Array(&b), &ctx());
    assert_eq!(n.unwrap().null_count(), 4);
    let n = float::remainder::<Float64Type, AnsiError>(
        Operand::Scalar(None),
        Operand::Array(&b),
        &ctx(),
    );
    assert_eq!(n.unwrap().null_count(), 4);
    let neg = float::unary(UnaryOp::Abs, &Float64Array::from(vec![Some(-0.0), None])).unwrap();
    assert_eq!((neg.value(0).to_bits(), neg.null_count()), (0, 1));
    let neg = float::unary(UnaryOp::Negate, &Float64Array::from(vec![2.5])).unwrap();
    assert_eq!(neg.value(0), -2.5);
}

#[test]
fn comparison_engine_errors() {
    use arrow_array::StringArray;
    use krishiv_kernels::compare::{BytesOperand, compare_bytes};
    let a = dec(vec![Some(1)], 5, 0);
    let b = dec(vec![Some(1)], 6, 1);
    let e = compare::<_, Equal>(Operand::Array(&a), Operand::Array(&b), &ctx()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let (x, y) = (
        StringArray::from(vec!["a"]),
        StringArray::from(vec!["a", "b"]),
    );
    let op = BytesOperand::Array(&x);
    #[expect(clippy::clone_on_copy, reason = "exercises the manual Clone impl")]
    let copy = op.clone();
    let e = compare_bytes::<_, Equal>(copy, BytesOperand::Array(&y)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let e = compare_bytes::<_, Equal>(
        BytesOperand::<arrow_array::types::Utf8Type>::Scalar(None),
        BytesOperand::Scalar(None),
    );
    assert!(e.is_err());
}

/// `l op r` under Spark 4.2's default result type (precision loss allowed).
fn dec_op(
    op: krishiv_kernels::decimal::DecimalOp,
    l: (i128, (u8, i8)),
    r: (i128, (u8, i8)),
) -> Result<Decimal128Array, krishiv_kernels::KernelError> {
    use krishiv_kernels::decimal::{decimal_arith, result_type};
    let out = result_type(op, l.1, r.1, true)?;
    let plan = DecimalPlan::new(op, l.1, r.1, out)?;
    let (a, b) = (
        dec(vec![Some(l.0)], l.1.0, l.1.1),
        dec(vec![Some(r.0)], r.1.0, r.1.1),
    );
    decimal_arith::<AnsiError>(Operand::Array(&a), Operand::Array(&b), &plan, &ctx())
}

/// Values, types and messages recorded from Spark 4.2.
#[test]
fn decimal_division_and_rounding_match_spark() {
    use krishiv_kernels::decimal::DecimalOp::{Divide, Multiply, Remainder};
    let check = |r: Result<Decimal128Array, _>, v: i128, p: u8, s: i8| {
        let r = r.unwrap();
        assert_eq!((r.value(0), r.precision(), r.scale()), (v, p, s));
    };
    check(dec_op(Divide, (1, (38, 0)), (3, (38, 0))), 333_333, 38, 6);
    let two_thirds = 6_666_666_666_666_666_666_666_666_666_666_666_667;
    check(
        dec_op(Divide, (2 * 10_i128.pow(37), (38, 37)), (3, (1, 0))),
        two_thirds,
        38,
        37,
    );
    check(
        dec_op(Divide, (10_i128.pow(9), (10, 9)), (8, (1, 0))),
        12_500_000_000,
        12,
        11,
    );
    check(dec_op(Divide, (5, (5, 0)), (9, (5, 0))), 555_556, 11, 6);
    check(dec_op(Divide, (-1, (5, 0)), (16, (5, 0))), -62_500, 11, 6);
    check(dec_op(Divide, (1, (1, 0)), (2_000_000, (7, 0))), 50, 9, 8);
    check(dec_op(Divide, (-1, (1, 0)), (3_000_000, (7, 0))), -33, 9, 8);
    // 0.000005 * 0.5 at decimal(38,6) rounds half up to 0.000003.
    check(dec_op(Multiply, (5, (38, 6)), (5, (38, 1))), 3, 38, 6);
    check(dec_op(Multiply, (-5, (38, 6)), (5, (38, 1))), -3, 38, 6);
    check(dec_op(Remainder, (-75, (3, 1)), (20, (3, 1))), -15, 3, 1);
    check(dec_op(Remainder, (-7, (1, 0)), (3, (1, 1))), -1, 1, 1);
    check(dec_op(Remainder, (7, (38, 0)), (3, (38, 38))), 1, 38, 38);
    // Spark's "Division impossible": the integral quotient needs 40 digits.
    let e = dec_op(Remainder, (-100, (3, 0)), (3, (38, 38))).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivDecimalRemainderImpossible
    );
    assert!(
        e.to_string()
            .contains("-100 divided by 0.00000000000000000000000000000000000003"),
        "{e}"
    );
    // Java cuts the quotient to 39 significant digits and fails only when a
    // dropped digit is not zero: Spark 4.0-4.2 return 0E-38 for these, in
    // every mode.
    let big = 12_345_678_901_234_567_890_123_456_789_012_345_678;
    for (a, b) in [
        (10_i128.pow(37), 1),
        (10_i128.pow(37), 2),
        (big, 3),
        (big, 1),
    ] {
        check(dec_op(Remainder, (a, (38, 0)), (b, (38, 38))), 0, 38, 38);
        let plan = DecimalPlan::new(Remainder, (38, 0), (38, 38), (38, 38)).unwrap();
        let (l, r) = (dec(vec![Some(a)], 38, 0), dec(vec![Some(b)], 38, 38));
        let legacy = krishiv_kernels::decimal::decimal_arith::<NullOnOverflow>(
            Operand::Array(&l),
            Operand::Array(&r),
            &plan,
            &ctx(),
        )
        .unwrap();
        assert_eq!((legacy.is_valid(0), legacy.value(0)), (true, 0));
    }
    // Still impossible: a non-zero digit is dropped.
    for (a, b) in [(10_i128.pow(37), (3, (38, 38))), (-big, (-7, (3, 3)))] {
        let e = dec_op(Remainder, (a, (38, 0)), b).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivDecimalRemainderImpossible
        );
    }
    let msg = |r: Result<Decimal128Array, krishiv_kernels::KernelError>| r.unwrap_err().to_string();
    let tail = " cannot be represented as Decimal(38, 6). If necessary set \"spark.sql.ansi.enabled\" \
                to \"false\" to bypass this error, and return NULL instead. SQLSTATE: 22003";
    assert_eq!(
        msg(dec_op(Divide, (1, (38, 0)), (1, (38, 38)))),
        format!(
            "[NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION]  1{}.{}{tail}",
            "0".repeat(38),
            "0".repeat(39)
        )
    );
    assert_eq!(
        msg(dec_op(Divide, (-2, (38, 0)), (3, (38, 38)))),
        format!(
            "[NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION]  -{}.{}{tail}",
            "6".repeat(38),
            "6".repeat(39)
        )
    );
    // 99999999999999999999999999999.999999999 * 99999999.9999999999.
    let a = 99_999_999_999_999_999_999_999_999_999_999_999_999;
    assert_eq!(
        msg(dec_op(
            Multiply,
            (a, (38, 9)),
            (999_999_999_999_999_999, (38, 10))
        )),
        format!(
            "[NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION]  9999999999999999989999999999999999999.90{tail}"
        )
    );
    let e = dec_op(Divide, (1, (2, 1)), (0, (1, 0))).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
    let e = dec_op(Remainder, (1, (2, 1)), (0, (1, 0))).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::RemainderByZero);
}

#[test]
fn decimal_div_and_unary_match_spark() {
    use krishiv_kernels::decimal::{decimal_integral_divide, decimal_unary};
    let a = dec(vec![Some(999), Some(-999), Some(-75), None], 3, 2);
    let b = dec(vec![Some(1), Some(200), Some(20), Some(1)], 3, 2);
    let q = decimal_integral_divide::<AnsiError>(
        Operand::Array(&a),
        Operand::Array(&b),
        (3, 2),
        (3, 2),
        &ctx(),
    );
    assert_eq!(
        q.unwrap(),
        Int64Array::from(vec![Some(999), Some(-4), Some(-3), None])
    );
    // 12345678901234567890123456789012345678 div 1 keeps the low 64 bits.
    let big = dec(
        vec![Some(12_345_678_901_234_567_890_123_456_789_012_345_678)],
        38,
        0,
    );
    let one = Operand::Scalar(Some(1));
    let q =
        decimal_integral_divide::<AnsiError>(Operand::Array(&big), one, (38, 0), (1, 0), &ctx());
    assert_eq!(q.unwrap().value(0), -4_302_749_291_975_740_594);
    // The quotient must fit decimal(38, 0) before it becomes a bigint.
    let tiny = dec(vec![Some(1)], 38, 38);
    let e = decimal_integral_divide::<AnsiError>(
        Operand::Scalar(Some(1)),
        Operand::Array(&tiny),
        (1, 0),
        (38, 38),
        &ctx(),
    );
    assert_eq!(
        e.unwrap_err().to_string(),
        format!(
            "[NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION]  1{} cannot be represented as Decimal(38, 0). If \
             necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error, and return NULL \
             instead. SQLSTATE: 22003",
            "0".repeat(38)
        )
    );
    let n = decimal_integral_divide::<NullOnOverflow>(
        Operand::Scalar(Some(1)),
        Operand::Array(&tiny),
        (1, 0),
        (38, 38),
        &ctx(),
    );
    assert_eq!(n.unwrap().null_count(), 1);
    let z = decimal_integral_divide::<NullOnOverflow>(
        Operand::Array(&a),
        Operand::Scalar(Some(0)),
        (3, 2),
        (1, 0),
        &ctx(),
    );
    assert_eq!(z.unwrap().null_count(), 4);
    let e = decimal_integral_divide::<AnsiError>(
        Operand::Array(&a),
        Operand::Scalar(Some(0)),
        (3, 2),
        (1, 0),
        &ctx(),
    );
    assert_eq!(e.unwrap_err().error_class(), ErrorClass::DivideByZero);
    let n = decimal_integral_divide::<AnsiError>(
        Operand::Scalar(None),
        Operand::Array(&b),
        (3, 2),
        (3, 2),
        &ctx(),
    );
    assert_eq!(n.unwrap().null_count(), 4);
    let neg = decimal_unary(UnaryOp::Negate, &dec(vec![Some(150), None], 10, 2)).unwrap();
    assert_eq!(neg, dec(vec![Some(-150), None], 10, 2));
    let abs = decimal_unary(UnaryOp::Abs, &dec(vec![Some(-150)], 10, 2)).unwrap();
    assert_eq!(abs, dec(vec![Some(150)], 10, 2));
}

/// The carrier API of the decimal kernels (P2-T160): `Decimal64` operands,
/// mixed carriers, a result on either carrier, and the checks.
#[test]
fn decimal_carriers() {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Decimal64Type, Decimal128Type};
    use arrow_array::{ArrayRef, Decimal64Array};
    use krishiv_kernels::decimal::{
        DecimalCarrier, DecimalOperand, decimal_arith_any, decimal_unary,
    };
    use krishiv_kernels::integer::UnaryOp;
    let d64 = |v: Vec<Option<i64>>, p, s| {
        Decimal64Array::from(v)
            .with_precision_and_scale(p, s)
            .unwrap()
    };
    let neg = decimal_unary(UnaryOp::Negate, &d64(vec![Some(150), None], 10, 2)).unwrap();
    assert_eq!(neg, d64(vec![Some(-150), None], 10, 2));
    let abs = decimal_unary(UnaryOp::Abs, &d64(vec![Some(-150)], 10, 2)).unwrap();
    assert_eq!(abs.value(0), 150);
    // `of_array` takes either carrier and nothing else.
    let a = d64(vec![Some(125), None, Some(-1)], 10, 2);
    let b = dec(vec![Some(250), Some(1), Some(1)], 20, 2);
    assert!(matches!(
        DecimalOperand::of_array(&a).unwrap(),
        DecimalOperand::Decimal64(_)
    ));
    assert!(matches!(
        DecimalOperand::of_array(&b).unwrap(),
        DecimalOperand::Decimal128(_)
    ));
    assert!(DecimalOperand::of_array(&arrow_array::Int64Array::from(vec![1])).is_err());
    // Mixed carriers: decimal(10,2) + decimal(20,2) -> decimal(21,2).
    let plan = DecimalPlan::new(ArithOp::Add, (10, 2), (20, 2), (21, 2)).unwrap();
    let (l, r) = (
        DecimalOperand::Decimal64(&a),
        DecimalOperand::Decimal128(&b),
    );
    let sum =
        decimal_arith_any::<AnsiError>(l, r, &plan, DecimalCarrier::Decimal128, &ctx()).unwrap();
    assert_eq!(
        sum.as_primitive::<Decimal128Type>(),
        &dec(vec![Some(375), None, Some(0)], 21, 2)
    );
    // A `Decimal64` result above precision 18 is an engine bug.
    let e =
        decimal_arith_any::<AnsiError>(l, r, &plan, DecimalCarrier::Decimal64, &ctx()).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // An operand of another type than the plan's.
    let wrong = DecimalPlan::new(ArithOp::Add, (11, 2), (20, 2), (21, 2)).unwrap();
    assert!(
        decimal_arith_any::<AnsiError>(l, r, &wrong, DecimalCarrier::Decimal128, &ctx()).is_err()
    );
    // A NULL literal: every row NULL, on the requested carrier.
    let small = DecimalPlan::new(ArithOp::Add, (10, 2), (10, 2), (11, 2)).unwrap();
    for (carrier, t) in [
        (
            DecimalCarrier::Decimal64,
            arrow_schema::DataType::Decimal64(11, 2),
        ),
        (
            DecimalCarrier::Decimal128,
            arrow_schema::DataType::Decimal128(11, 2),
        ),
    ] {
        let n: ArrayRef = decimal_arith_any::<AnsiError>(
            DecimalOperand::Decimal64(&a),
            DecimalOperand::Scalar(None),
            &small,
            carrier,
            &ctx(),
        )
        .unwrap();
        assert_eq!((n.data_type(), n.null_count()), (&t, 3));
    }
    // A literal beyond `i64` next to a `Decimal64` column takes the 128-bit
    // lanes; a rounding plan (the 256-bit path) narrows its result.
    let wide_lit = DecimalPlan::new(ArithOp::Add, (10, 2), (20, 2), (21, 2)).unwrap();
    let big = 10_i128.pow(19);
    let s = decimal_arith_any::<AnsiError>(
        DecimalOperand::Decimal64(&a),
        DecimalOperand::Scalar(Some(big)),
        &wide_lit,
        DecimalCarrier::Decimal128,
        &ctx(),
    )
    .unwrap();
    assert_eq!(s.as_primitive::<Decimal128Type>().value(0), big + 125);
    let div = DecimalPlan::new(
        krishiv_kernels::decimal::DecimalOp::Divide,
        (10, 2),
        (10, 2),
        (18, 6),
    )
    .unwrap();
    let q = decimal_arith_any::<AnsiError>(
        DecimalOperand::Decimal64(&a),
        DecimalOperand::Scalar(Some(300)),
        &div,
        DecimalCarrier::Decimal64,
        &ctx(),
    )
    .unwrap();
    // 1.25 / 3.00 = 0.416667 (HALF_UP).
    assert_eq!(q.as_primitive::<Decimal64Type>().value(0), 416_667);
    assert_eq!(q.data_type(), &arrow_schema::DataType::Decimal64(18, 6));
}
