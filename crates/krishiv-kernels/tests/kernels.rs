// SPDX-License-Identifier: Apache-2.0
//! Targeted Spark-semantics and dispatch tests for the kernel framework.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, BooleanArray, Decimal128Array, Float64Array, Int32Array, Int64Array};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::decimal::{DecimalPlan, DecimalStorage, decimal_binary};
use krishiv_kernels::float::{Equal, Less, compare};
use krishiv_kernels::integer::binary;
use krishiv_kernels::{
    Add, AnsiError, ArithOp, JavaWrap, KernelCtx, Mul, NullOnOverflow, Operand, Sub, Target,
    Unchecked,
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
