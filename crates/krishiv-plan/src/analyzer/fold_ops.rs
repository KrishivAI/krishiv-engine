// SPDX-License-Identifier: Apache-2.0
//! The operators a foldable expression may contain, evaluated on one-row
//! arrays through the same kernels the executor runs (so overflow,
//! division by zero and ANSI/legacy behaviour are the kernels').
//!
//! Every function answers `None` for an operand shape it does not
//! evaluate: the expression is then not folded (a constant-only argument
//! site reports it as non-foldable, see `fold.rs`).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
};
use arrow_array::{Array, ArrayRef, BooleanArray, StringArray};
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::KrishivError;
use krishiv_kernels::compare::{self, Equal, Greater, GreaterEq, Less, LessEq};
use krishiv_kernels::decimal::{
    DecimalOp, DecimalPlan, decimal_arith, decimal_integral_divide, decimal_unary,
};
use krishiv_kernels::integer::{self, SparkIntegral, UnaryOp};
use krishiv_kernels::{
    Add, AnsiError, ArithOp, JavaWrap, KernelCtx, KernelError, Mul, NullOnOverflow, Operand, Sub,
    float,
};

use crate::Builtin;

/// An evaluation: `None` when the shape is not evaluated here.
pub(super) type Eval = Option<Result<ArrayRef, KrishivError>>;

fn done<A: Array + 'static>(r: Result<A, KernelError>) -> Eval {
    Some(
        r.map(|a| Arc::new(a) as ArrayRef)
            .map_err(KrishivError::from),
    )
}

fn operand<T: arrow_array::ArrowPrimitiveType>(a: &ArrayRef) -> Operand<'_, T> {
    Operand::Array(a.as_primitive::<T>())
}

/// `l op r` for `+ - * / %` over operands of the result type `out`.
pub(super) fn arithmetic(
    op: Builtin,
    l: &ArrayRef,
    r: &ArrayRef,
    out: &ArrowType,
    ansi: bool,
    ctx: &KernelCtx,
) -> Eval {
    if let (
        ArrowType::Decimal128(lp, ls),
        ArrowType::Decimal128(rp, rs),
        ArrowType::Decimal128(p, s),
    ) = (l.data_type(), r.data_type(), out)
    {
        let dop = match op {
            Builtin::Add => DecimalOp::Add,
            Builtin::Subtract => DecimalOp::Subtract,
            Builtin::Multiply => DecimalOp::Multiply,
            Builtin::Divide => DecimalOp::Divide,
            Builtin::Remainder => DecimalOp::Remainder,
            _ => return None,
        };
        let plan = match DecimalPlan::new(dop, (*lp, *ls), (*rp, *rs), (*p, *s)) {
            Ok(plan) => plan,
            Err(e) => return Some(Err(e.into())),
        };
        let (a, b) = (operand::<Decimal128Type>(l), operand::<Decimal128Type>(r));
        return if ansi {
            done(decimal_arith::<AnsiError>(a, b, &plan, ctx))
        } else {
            done(decimal_arith::<NullOnOverflow>(a, b, &plan, ctx))
        };
    }
    if l.data_type() != out || r.data_type() != out {
        return None;
    }
    match out {
        ArrowType::Int8 => integral::<Int8Type>(op, l, r, ansi, ctx),
        ArrowType::Int16 => integral::<Int16Type>(op, l, r, ansi, ctx),
        ArrowType::Int32 => integral::<Int32Type>(op, l, r, ansi, ctx),
        ArrowType::Int64 => integral::<Int64Type>(op, l, r, ansi, ctx),
        ArrowType::Float32 => floating::<Float32Type>(op, l, r, ansi, ctx),
        ArrowType::Float64 => floating::<Float64Type>(op, l, r, ansi, ctx),
        _ => None,
    }
}

/// Integral `+ - * %` (`/` of integrals is typed `double` by the analyzer).
fn integral<T: SparkIntegral>(
    op: Builtin,
    l: &ArrayRef,
    r: &ArrayRef,
    ansi: bool,
    ctx: &KernelCtx,
) -> Eval {
    let (a, b) = (operand::<T>(l), operand::<T>(r));
    macro_rules! run {
        ($p:ty) => {
            match op {
                Builtin::Add => done(integer::binary::<T, Add, $p>(a, b, ctx)),
                Builtin::Subtract => done(integer::binary::<T, Sub, $p>(a, b, ctx)),
                Builtin::Multiply => done(integer::binary::<T, Mul, $p>(a, b, ctx)),
                Builtin::Remainder => done(integer::remainder::<T, $p>(a, b, ctx)),
                _ => None,
            }
        };
    }
    if ansi {
        run!(AnsiError)
    } else {
        run!(JavaWrap)
    }
}

fn floating<T>(op: Builtin, l: &ArrayRef, r: &ArrayRef, ansi: bool, ctx: &KernelCtx) -> Eval
where
    T: arrow_array::ArrowPrimitiveType<Native: float::SparkFloat>,
{
    let (a, b) = (operand::<T>(l), operand::<T>(r));
    match (op, ansi) {
        (Builtin::Add, _) => done(float::binary(ArithOp::Add, a, b, ctx)),
        (Builtin::Subtract, _) => done(float::binary(ArithOp::Subtract, a, b, ctx)),
        (Builtin::Multiply, _) => done(float::binary(ArithOp::Multiply, a, b, ctx)),
        (Builtin::Divide, true) => done(float::divide::<T, AnsiError>(a, b, ctx)),
        (Builtin::Divide, false) => done(float::divide::<T, JavaWrap>(a, b, ctx)),
        (Builtin::Remainder, true) => done(float::remainder::<T, AnsiError>(a, b, ctx)),
        (Builtin::Remainder, false) => done(float::remainder::<T, JavaWrap>(a, b, ctx)),
        _ => None,
    }
}

/// `l div r` over `bigint`s or decimals.
pub(super) fn integral_divide(l: &ArrayRef, r: &ArrayRef, ansi: bool, ctx: &KernelCtx) -> Eval {
    match (l.data_type(), r.data_type()) {
        (ArrowType::Int64, ArrowType::Int64) => {
            let (a, b) = (operand::<Int64Type>(l), operand::<Int64Type>(r));
            if ansi {
                done(integer::integral_divide::<AnsiError>(a, b, ctx))
            } else {
                done(integer::integral_divide::<JavaWrap>(a, b, ctx))
            }
        }
        (ArrowType::Decimal128(lp, ls), ArrowType::Decimal128(rp, rs)) => {
            let (lt, rt) = ((*lp, *ls), (*rp, *rs));
            let (a, b) = (operand::<Decimal128Type>(l), operand::<Decimal128Type>(r));
            if ansi {
                done(decimal_integral_divide::<AnsiError>(a, b, lt, rt, ctx))
            } else {
                done(decimal_integral_divide::<NullOnOverflow>(a, b, lt, rt, ctx))
            }
        }
        _ => None,
    }
}

/// Unary `-`.
pub(super) fn negate(a: &ArrayRef, ansi: bool, ctx: &KernelCtx) -> Eval {
    macro_rules! int {
        ($t:ty) => {
            if ansi {
                done(integer::unary::<$t, AnsiError>(
                    UnaryOp::Negate,
                    a.as_primitive::<$t>(),
                    ctx,
                ))
            } else {
                done(integer::unary::<$t, JavaWrap>(
                    UnaryOp::Negate,
                    a.as_primitive::<$t>(),
                    ctx,
                ))
            }
        };
    }
    match a.data_type() {
        ArrowType::Int8 => int!(Int8Type),
        ArrowType::Int16 => int!(Int16Type),
        ArrowType::Int32 => int!(Int32Type),
        ArrowType::Int64 => int!(Int64Type),
        ArrowType::Float32 => done(float::unary(
            UnaryOp::Negate,
            a.as_primitive::<Float32Type>(),
        )),
        ArrowType::Float64 => done(float::unary(
            UnaryOp::Negate,
            a.as_primitive::<Float64Type>(),
        )),
        ArrowType::Decimal128(..) => done(decimal_unary(
            UnaryOp::Negate,
            a.as_primitive::<Decimal128Type>(),
        )),
        _ => None,
    }
}

/// `l op r` for `= < <= > >=` over two operands of one numeric type, or
/// two strings (compared by bytes, as Spark's `UTF8String` is).
pub(super) fn comparison(op: Builtin, l: &ArrayRef, r: &ArrayRef, ctx: &KernelCtx) -> Eval {
    if l.data_type() != r.data_type() {
        return None;
    }
    macro_rules! prim {
        ($t:ty) => {{
            let (a, b) = (operand::<$t>(l), operand::<$t>(r));
            match op {
                Builtin::Eq => done(compare::compare::<$t, Equal>(a, b, ctx)),
                Builtin::Lt => done(compare::compare::<$t, Less>(a, b, ctx)),
                Builtin::LtEq => done(compare::compare::<$t, LessEq>(a, b, ctx)),
                Builtin::Gt => done(compare::compare::<$t, Greater>(a, b, ctx)),
                Builtin::GtEq => done(compare::compare::<$t, GreaterEq>(a, b, ctx)),
                _ => None,
            }
        }};
    }
    match l.data_type() {
        ArrowType::Int8 => prim!(Int8Type),
        ArrowType::Int16 => prim!(Int16Type),
        ArrowType::Int32 => prim!(Int32Type),
        ArrowType::Int64 => prim!(Int64Type),
        ArrowType::Float32 => prim!(Float32Type),
        ArrowType::Float64 => prim!(Float64Type),
        ArrowType::Decimal128(..) => prim!(Decimal128Type),
        ArrowType::Utf8 => {
            let (a, b) = (l.as_string::<i32>(), r.as_string::<i32>());
            let ord = (a.is_valid(0) && b.is_valid(0))
                .then(|| a.value(0).as_bytes().cmp(b.value(0).as_bytes()));
            let holds = |o: std::cmp::Ordering| match op {
                Builtin::Eq => Some(o.is_eq()),
                Builtin::Lt => Some(o.is_lt()),
                Builtin::LtEq => Some(o.is_le()),
                Builtin::Gt => Some(o.is_gt()),
                Builtin::GtEq => Some(o.is_ge()),
                _ => None,
            };
            let v = match ord {
                Some(o) => Some(holds(o)?),
                None => None,
            };
            Some(Ok(Arc::new(BooleanArray::from(vec![v])) as ArrayRef))
        }
        _ => None,
    }
}

/// The boolean value of a one-row array: `Some(None)` for NULL.
pub(super) fn boolean(a: &ArrayRef) -> Option<Option<bool>> {
    if a.data_type() == &ArrowType::Null {
        return Some(None);
    }
    let b = a.as_boolean_opt()?;
    Some(b.is_valid(0).then(|| b.value(0)))
}

/// A one-row boolean array.
pub(super) fn bool_array(v: Option<bool>) -> ArrayRef {
    Arc::new(BooleanArray::from(vec![v]))
}

/// `concat` of strings: NULL when an argument is.
pub(super) fn concat(parts: &[ArrayRef]) -> Eval {
    let mut out = String::new();
    let mut null = false;
    for p in parts {
        let s = p.as_string_opt::<i32>()?;
        if s.is_valid(0) {
            out.push_str(s.value(0));
        } else {
            null = true;
        }
    }
    let v: Option<&str> = (!null).then_some(out.as_str());
    Some(Ok(Arc::new(StringArray::from(vec![v])) as ArrayRef))
}

/// A one-row array is NULL.
pub(super) fn is_null(a: &ArrayRef) -> bool {
    a.data_type() == &ArrowType::Null || a.null_count() > 0
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        Decimal128Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, NullArray,
    };
    use krishiv_common::error::ErrorClass;

    use super::*;

    // The session's current Spark line, as the tests that do not choose
    // one use it.
    fn arithmetic(op: Builtin, l: &ArrayRef, r: &ArrayRef, out: &ArrowType, ansi: bool) -> Eval {
        super::arithmetic(op, l, r, out, ansi, &KernelCtx::new())
    }

    fn integral_divide(l: &ArrayRef, r: &ArrayRef, ansi: bool) -> Eval {
        super::integral_divide(l, r, ansi, &KernelCtx::new())
    }

    fn negate(a: &ArrayRef, ansi: bool) -> Eval {
        super::negate(a, ansi, &KernelCtx::new())
    }

    fn comparison(op: Builtin, l: &ArrayRef, r: &ArrayRef) -> Eval {
        super::comparison(op, l, r, &KernelCtx::new())
    }

    fn i32s(v: Option<i32>) -> ArrayRef {
        Arc::new(Int32Array::from(vec![v]))
    }

    fn i64s(v: Option<i64>) -> ArrayRef {
        Arc::new(Int64Array::from(vec![v]))
    }

    fn f64s(v: Option<f64>) -> ArrayRef {
        Arc::new(Float64Array::from(vec![v]))
    }

    fn dec(v: Option<i128>, p: u8, s: i8) -> ArrayRef {
        Arc::new(
            Decimal128Array::from(vec![v])
                .with_precision_and_scale(p, s)
                .unwrap(),
        )
    }

    fn strs(v: Option<&str>) -> ArrayRef {
        Arc::new(StringArray::from(vec![v]))
    }

    fn ok(e: Eval) -> ArrayRef {
        e.expect("evaluated").expect("no error")
    }

    fn err(e: Eval) -> ErrorClass {
        e.expect("evaluated").expect_err("an error").error_class()
    }

    #[test]
    fn integral_arithmetic_follows_the_overflow_mode() {
        let int = ArrowType::Int32;
        let sum = |a, b, ansi| arithmetic(Builtin::Add, &i32s(a), &i32s(b), &int, ansi);
        assert_eq!(
            ok(sum(Some(2), Some(3), true))
                .as_primitive::<Int32Type>()
                .value(0),
            5
        );
        assert_eq!(
            err(sum(Some(i32::MAX), Some(1), true)),
            ErrorClass::ArithmeticOverflow
        );
        // Legacy mode wraps.
        let wrapped = ok(sum(Some(i32::MAX), Some(1), false));
        assert_eq!(wrapped.as_primitive::<Int32Type>().value(0), i32::MIN);
        // NULL propagates.
        assert!(ok(sum(None, Some(1), true)).is_null(0));
        let sub = arithmetic(
            Builtin::Subtract,
            &i32s(Some(1)),
            &i32s(Some(5)),
            &int,
            true,
        );
        assert_eq!(ok(sub).as_primitive::<Int32Type>().value(0), -4);
        let mul = arithmetic(
            Builtin::Multiply,
            &i32s(Some(6)),
            &i32s(Some(7)),
            &int,
            true,
        );
        assert_eq!(ok(mul).as_primitive::<Int32Type>().value(0), 42);
        let rem = |b, ansi| {
            arithmetic(
                Builtin::Remainder,
                &i32s(Some(7)),
                &i32s(Some(b)),
                &int,
                ansi,
            )
        };
        assert_eq!(ok(rem(4, true)).as_primitive::<Int32Type>().value(0), 3);
        assert_eq!(err(rem(0, true)), ErrorClass::RemainderByZero);
        assert!(ok(rem(0, false)).is_null(0));
        // Every integral width.
        for (a, t) in [
            (
                Arc::new(Int8Array::from(vec![1_i8])) as ArrayRef,
                ArrowType::Int8,
            ),
            (Arc::new(Int16Array::from(vec![1_i16])), ArrowType::Int16),
            (i64s(Some(1)), ArrowType::Int64),
        ] {
            let r = ok(arithmetic(Builtin::Add, &a, &a, &t, true));
            assert_eq!(r.data_type(), &t);
        }
    }

    #[test]
    fn arithmetic_shapes_it_does_not_evaluate_are_none() {
        let int = ArrowType::Int32;
        // `/` of integrals (the analyzer types it `double`), another
        // operator, operands of the wrong type.
        assert!(arithmetic(Builtin::Divide, &i32s(Some(1)), &i32s(Some(1)), &int, true).is_none());
        assert!(arithmetic(Builtin::Eq, &i32s(Some(1)), &i32s(Some(1)), &int, true).is_none());
        assert!(arithmetic(Builtin::Add, &i32s(Some(1)), &i64s(Some(1)), &int, true).is_none());
        let s = strs(Some("1"));
        assert!(arithmetic(Builtin::Add, &s, &s, &ArrowType::Utf8, true).is_none());
        let d = dec(Some(1), 5, 0);
        let dt = ArrowType::Decimal128(6, 0);
        assert!(arithmetic(Builtin::Eq, &d, &d, &dt, true).is_none());
    }

    #[test]
    fn floating_arithmetic() {
        let t = ArrowType::Float64;
        let run = |op, a, b, ansi| arithmetic(op, &f64s(Some(a)), &f64s(Some(b)), &t, ansi);
        let v = |e: Eval| ok(e).as_primitive::<Float64Type>().value(0);
        assert!((v(run(Builtin::Add, 1.5, 2.0, true)) - 3.5).abs() < f64::EPSILON);
        assert!((v(run(Builtin::Subtract, 1.5, 2.0, true)) + 0.5).abs() < f64::EPSILON);
        assert!((v(run(Builtin::Multiply, 1.5, 2.0, true)) - 3.0).abs() < f64::EPSILON);
        assert!((v(run(Builtin::Divide, 3.0, 2.0, true)) - 1.5).abs() < f64::EPSILON);
        assert!((v(run(Builtin::Remainder, 5.5, 2.0, false)) - 1.5).abs() < f64::EPSILON);
        assert_eq!(
            err(run(Builtin::Divide, 1.0, 0.0, true)),
            ErrorClass::DivideByZero
        );
        assert!(ok(run(Builtin::Divide, 1.0, 0.0, false)).is_null(0));
        assert_eq!(
            err(run(Builtin::Remainder, 1.0, 0.0, true)),
            ErrorClass::RemainderByZero
        );
        assert!(ok(run(Builtin::Remainder, 1.0, 0.0, false)).is_null(0));
        let f32s: ArrayRef = Arc::new(arrow_array::Float32Array::from(vec![1.5_f32]));
        let r = arithmetic(Builtin::Add, &f32s, &f32s, &ArrowType::Float32, true);
        assert_eq!(ok(r).as_primitive::<Float32Type>().value(0), 3.0);
        // `div` is not floating-point arithmetic.
        assert!(run(Builtin::IntegralDivide, 1.0, 1.0, true).is_none());
    }

    #[test]
    fn the_session_spark_version_reaches_the_kernels() {
        use krishiv_kernels::SparkVersion;
        let int = ArrowType::Int32;
        let rem = |ctx: &KernelCtx| {
            super::arithmetic(
                Builtin::Remainder,
                &i32s(Some(1)),
                &i32s(Some(0)),
                &int,
                true,
                ctx,
            )
        };
        let v40 = KernelCtx::new().with_spark_version(SparkVersion::V4_0);
        assert_eq!(err(rem(&v40)), ErrorClass::DivideByZero);
        assert_eq!(err(rem(&KernelCtx::new())), ErrorClass::RemainderByZero);
    }

    #[test]
    fn decimal_arithmetic() {
        let out = ArrowType::Decimal128(7, 2);
        let a = dec(Some(150), 5, 2);
        let b = dec(Some(25), 5, 2);
        let run = |op, ansi| arithmetic(op, &a, &b, &out, ansi);
        let v = |e: Eval| ok(e).as_primitive::<Decimal128Type>().value(0);
        assert_eq!(v(run(Builtin::Add, true)), 175);
        assert_eq!(v(run(Builtin::Subtract, false)), 125);
        // Overflow of the result type: an error, or NULL in legacy mode.
        let tight = ArrowType::Decimal128(3, 2);
        let big = dec(Some(999), 3, 2);
        let over = arithmetic(Builtin::Add, &big, &big, &tight, true);
        assert!(over.expect("evaluated").is_err());
        let wrapped = arithmetic(Builtin::Add, &big, &big, &tight, false);
        assert!(ok(wrapped).is_null(0));
        // A plan the kernels refuse is the kernels' error, not a panic.
        let bad = arithmetic(
            Builtin::Add,
            &dec(Some(1), 5, 2),
            &dec(Some(1), 5, 2),
            &ArrowType::Decimal128(38, 38),
            true,
        );
        assert!(matches!(bad, Some(Err(_))), "{bad:?}");
    }

    #[test]
    fn integral_division() {
        let q = |a: i64, b: i64, ansi| integral_divide(&i64s(Some(a)), &i64s(Some(b)), ansi);
        assert_eq!(ok(q(7, 2, true)).as_primitive::<Int64Type>().value(0), 3);
        assert_eq!(err(q(1, 0, true)), ErrorClass::DivideByZero);
        assert!(ok(q(1, 0, false)).is_null(0));
        let d = |ansi| integral_divide(&dec(Some(750), 5, 2), &dec(Some(200), 5, 2), ansi);
        assert_eq!(ok(d(true)).as_primitive::<Int64Type>().value(0), 3);
        assert_eq!(ok(d(false)).as_primitive::<Int64Type>().value(0), 3);
        assert!(integral_divide(&i32s(Some(1)), &i32s(Some(1)), true).is_none());
    }

    #[test]
    fn negation() {
        let n = |a: &ArrayRef, ansi| negate(a, ansi);
        assert_eq!(
            ok(n(&i32s(Some(5)), true))
                .as_primitive::<Int32Type>()
                .value(0),
            -5
        );
        assert_eq!(
            err(n(&i32s(Some(i32::MIN)), true)),
            ErrorClass::ArithmeticOverflow
        );
        let wrapped = ok(n(&i32s(Some(i32::MIN)), false));
        assert_eq!(wrapped.as_primitive::<Int32Type>().value(0), i32::MIN);
        for a in [
            Arc::new(Int8Array::from(vec![1_i8])) as ArrayRef,
            Arc::new(Int16Array::from(vec![1_i16])),
            i64s(Some(1)),
            f64s(Some(1.5)),
            Arc::new(arrow_array::Float32Array::from(vec![1.5_f32])),
            dec(Some(15), 3, 1),
        ] {
            assert_eq!(ok(n(&a, true)).data_type(), a.data_type());
        }
        assert!(n(&strs(Some("x")), true).is_none());
    }

    #[test]
    fn comparisons() {
        let t = |e: Eval| ok(e).as_boolean().value(0);
        assert!(t(comparison(Builtin::Lt, &i32s(Some(1)), &i32s(Some(2)))));
        assert!(t(comparison(Builtin::LtEq, &i32s(Some(2)), &i32s(Some(2)))));
        assert!(t(comparison(Builtin::Eq, &i64s(Some(2)), &i64s(Some(2)))));
        assert!(t(comparison(
            Builtin::Gt,
            &f64s(Some(2.5)),
            &f64s(Some(2.0))
        )));
        assert!(!t(comparison(
            Builtin::GtEq,
            &dec(Some(1), 3, 1),
            &dec(Some(2), 3, 1)
        )));
        assert!(t(comparison(
            Builtin::Lt,
            &strs(Some("a")),
            &strs(Some("b"))
        )));
        assert!(t(comparison(
            Builtin::GtEq,
            &strs(Some("b")),
            &strs(Some("b"))
        )));
        assert!(t(comparison(
            Builtin::Gt,
            &strs(Some("b")),
            &strs(Some("a"))
        )));
        assert!(t(comparison(
            Builtin::LtEq,
            &strs(Some("a")),
            &strs(Some("a"))
        )));
        assert!(t(comparison(
            Builtin::Eq,
            &strs(Some("a")),
            &strs(Some("a"))
        )));
        // A NULL string compares to NULL.
        let null = comparison(Builtin::Eq, &strs(None), &strs(Some("a")));
        assert!(ok(null).is_null(0));
        // Another operator, another type, mismatched types.
        assert!(comparison(Builtin::Add, &strs(Some("a")), &strs(Some("a"))).is_none());
        assert!(comparison(Builtin::Add, &i32s(Some(1)), &i32s(Some(1))).is_none());
        assert!(comparison(Builtin::Eq, &i32s(Some(1)), &i64s(Some(1))).is_none());
        let b: ArrayRef = bool_array(Some(true));
        assert!(comparison(Builtin::Eq, &b, &b).is_none());
        for t in [ArrowType::Int8, ArrowType::Int16] {
            let a = arrow_array::new_null_array(&t, 1);
            assert!(comparison(Builtin::Eq, &a, &a).is_some());
        }
    }

    #[test]
    fn booleans_nulls_and_concat() {
        assert_eq!(boolean(&bool_array(Some(true))), Some(Some(true)));
        assert_eq!(boolean(&bool_array(None)), Some(None));
        let null: ArrayRef = Arc::new(NullArray::new(1));
        assert_eq!(boolean(&null), Some(None));
        assert_eq!(boolean(&i32s(Some(1))), None);
        assert!(is_null(&null));
        assert!(is_null(&i32s(None)));
        assert!(!is_null(&i32s(Some(1))));
        let joined = ok(concat(&[strs(Some("a")), strs(Some("b"))]));
        assert_eq!(joined.as_string::<i32>().value(0), "ab");
        let with_null = ok(concat(&[strs(Some("a")), strs(None)]));
        assert!(with_null.is_null(0));
        assert!(concat(&[strs(Some("a")), i32s(Some(1))]).is_none());
    }
}
