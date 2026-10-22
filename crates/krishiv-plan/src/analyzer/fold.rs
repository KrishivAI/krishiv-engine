// SPDX-License-Identifier: Apache-2.0
//! The value of a constant argument, for the functions whose typing reads
//! it (`round`'s scale, `extract`'s field, a `LIMIT`).
//!
//! Spark folds a foldable argument before it types the call, so
//! `round(x, CAST(1.5 AS FLOAT))` has scale 1 and so does
//! `round(x, 0.5 + 0.5)`. Foldable here means what Spark's `foldable` means
//! for the expressions this folder evaluates: literals; casts of foldable
//! values; `+ - * / % div` and unary `-`; comparisons, `AND`/`OR`/`NOT`,
//! `IS [NOT] NULL`; `CASE`/`if`/`coalesce`; `concat`; and the deterministic
//! string, math and arithmetic functions (`abs`, `length`, `upper`, ...) of
//! foldable arguments. Anything else (a column, a nondeterministic call, a
//! datetime function, which reads the session time zone) is not a constant.
//!
//! Evaluation runs through the same kernels the executor uses (so overflow,
//! division by zero, malformed input and ANSI/legacy behaviour are the
//! kernels'). A call that fails folds to [`Constant::Error`]: the function
//! that reads the value raises it, any other ignores it, so an unselected
//! branch never raises. `CASE`, `if` and `coalesce` evaluate lazily, as
//! Spark's do.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Decimal128Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, StringArray, new_null_array,
};
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_functions::scalar::{Arg, BindArg, BindOptions, bind_scalar, is_argument_pure};
use krishiv_functions::signature::FunctionKind;
use krishiv_kernels::cast::{CastMode as KernelMode, CastOptions, FixedOffset, cast};
use krishiv_types::DataType;

use super::Constant;
use super::fold_ops;
use crate::{
    Builtin, CastMode, Expr, ExprNode, FunctionId, Literal, ResolvedExpr, ResolvedFunction,
};

/// The session settings folding reads.
#[derive(Debug, Clone, Copy)]
pub(super) struct Conf {
    /// `spark.sql.ansi.enabled`.
    pub ansi: bool,
    /// `spark.sql.decimalOperations.allowPrecisionLoss`.
    pub decimal_loss: bool,
}

/// One folded subexpression: `None` when it is not foldable, else its
/// one-row value or the error evaluating it raises.
type Folded = Option<Result<ArrayRef, KrishivError>>;

/// Whether `e` is a plain literal (what `first(x, true)` requires of its
/// flag, where Spark matches `Literal` and does not fold).
pub(super) const fn is_literal(e: &Expr) -> bool {
    matches!(e.node, ExprNode::Literal(_))
}

/// The value of `e` as a constant, when it is foldable.
pub(super) fn constant(e: &Expr, conf: Conf) -> Option<Constant> {
    match &e.node {
        ExprNode::Literal(Literal::Null(_)) => return Some(Constant::Null),
        ExprNode::Literal(Literal::Boolean(v)) => return Some(Constant::Boolean(*v)),
        ExprNode::Literal(Literal::String(v)) => return Some(Constant::String(v.clone())),
        _ => {}
    }
    let folded = fold(e, conf)?;
    Some(match folded.and_then(|a| value(&a, conf.ansi)) {
        Ok(c) => c,
        Err(e) => Constant::Error(e),
    })
}

/// A folded array as a constant. A fractional value is what the implicit
/// cast to `INT` of an integral argument gives (the scale of `round`).
fn value(a: &ArrayRef, ansi: bool) -> Result<Constant, KrishivError> {
    if a.null_count() > 0 || a.data_type() == &ArrowType::Null {
        return Ok(Constant::Null);
    }
    Ok(match a.data_type() {
        ArrowType::Boolean => Constant::Boolean(a.as_boolean().value(0)),
        ArrowType::Int8 => Constant::Integral(i64::from(a.as_primitive::<Int8Type>().value(0))),
        ArrowType::Int16 => Constant::Integral(i64::from(a.as_primitive::<Int16Type>().value(0))),
        ArrowType::Int32 => Constant::Integral(i64::from(a.as_primitive::<Int32Type>().value(0))),
        ArrowType::Int64 => Constant::Integral(a.as_primitive::<Int64Type>().value(0)),
        ArrowType::Utf8 => Constant::String(a.as_string::<i32>().value(0).to_owned()),
        ArrowType::Float32 | ArrowType::Float64 | ArrowType::Decimal128(..) => {
            let int = cast(
                a.as_ref(),
                &ArrowType::Int32,
                &options(mode(ansi), &FixedOffset(0)),
            )
            .map_err(KrishivError::from)?;
            if int.null_count() > 0 {
                Constant::Null
            } else {
                Constant::Fractional(i64::from(int.as_primitive::<Int32Type>().value(0)))
            }
        }
        _ => return Err(unsupported()),
    })
}

fn mode(ansi: bool) -> KernelMode {
    if ansi {
        KernelMode::Ansi
    } else {
        KernelMode::Legacy
    }
}

fn options<'a>(mode: KernelMode, zone: &'a FixedOffset) -> CastOptions<'a> {
    CastOptions {
        mode,
        time_zone: zone,
    }
}

fn unsupported() -> KrishivError {
    krishiv_common::error::KrishivUnsupportedFeature {
        feature: "folding this constant".to_owned(),
    }
    .into()
}

/// `e` as a one-row array: `None` when it is not foldable here.
fn fold(e: &Expr, conf: Conf) -> Folded {
    match &e.node {
        ExprNode::Literal(l) => literal(l).map(Ok),
        ExprNode::Cast { expr, to, mode: m } => fold_cast(expr, to, *m, conf),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => call(f, conf),
        _ => None,
    }
}

fn fold_cast(expr: &Expr, to: &DataType, m: CastMode, conf: Conf) -> Folded {
    let inner = match fold(expr, conf)? {
        Ok(a) => a,
        Err(e) => return Some(Err(e)),
    };
    // A NULL of any type casts to a NULL of the target type.
    if inner.null_count() > 0 || inner.data_type() == &ArrowType::Null {
        return Some(Ok(new_null_array(&to.to_arrow(), 1)));
    }
    // The kernels' targets: numbers (a string or a date is not folded).
    if !matches!(
        to,
        DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Double
            | DataType::Decimal(_)
    ) {
        return None;
    }
    let kernel_mode = match m {
        CastMode::Try => KernelMode::Try,
        CastMode::Legacy => KernelMode::Legacy,
        CastMode::Ansi => KernelMode::Ansi,
        _ => mode(conf.ansi),
    };
    match cast(
        inner.as_ref(),
        &to.to_arrow(),
        &options(kernel_mode, &FixedOffset(0)),
    ) {
        Ok(a) => Some(Ok(a)),
        Err(err) => {
            let err = KrishivError::from(err);
            (err.error_class() != ErrorClass::KrishivUnsupportedFeature).then_some(Err(err))
        }
    }
}

/// Every child of a call, folded; `None` when any is not foldable (a call
/// is foldable only if all its arguments are, whichever it evaluates).
fn kids(f: &ResolvedFunction, conf: Conf) -> Option<Vec<Result<ArrayRef, KrishivError>>> {
    // Stops at the first argument that is not foldable (a column), so a
    // deep non-constant tree costs one descent, not an evaluation.
    f.args.iter().map(|a| fold(a, conf)).collect()
}

fn call(f: &ResolvedFunction, conf: Conf) -> Folded {
    if !f.deterministic || f.distinct || f.filter.is_some() || f.kind != FunctionKind::Scalar {
        return None;
    }
    let kids = kids(f, conf)?;
    match &f.id {
        FunctionId::Builtin(b) => operator(*b, f, kids, conf),
        FunctionId::Registry { name, .. } => registry(name, f, kids, conf),
    }
}

/// Splits the children into values, or the first error (evaluation order).
fn values(kids: Vec<Result<ArrayRef, KrishivError>>) -> Result<Vec<ArrayRef>, KrishivError> {
    kids.into_iter().collect()
}

fn operator(
    op: Builtin,
    f: &ResolvedFunction,
    kids: Vec<Result<ArrayRef, KrishivError>>,
    conf: Conf,
) -> Folded {
    match op {
        Builtin::And | Builtin::Or => return logic(op == Builtin::And, kids),
        Builtin::When => return when(f, kids),
        _ => {}
    }
    let args = match values(kids) {
        Ok(a) => a,
        Err(e) => return Some(Err(e)),
    };
    let out = f.data_type.to_arrow();
    match (op, args.as_slice()) {
        (
            Builtin::Add
            | Builtin::Subtract
            | Builtin::Multiply
            | Builtin::Divide
            | Builtin::Remainder,
            [l, r],
        ) => fold_ops::arithmetic(op, l, r, &out, conf.ansi),
        (Builtin::IntegralDivide, [l, r]) => fold_ops::integral_divide(l, r, conf.ansi),
        (Builtin::Negative, [a]) => fold_ops::negate(a, conf.ansi),
        (Builtin::Eq | Builtin::Lt | Builtin::LtEq | Builtin::Gt | Builtin::GtEq, [l, r]) => {
            if fold_ops::is_null(l) || fold_ops::is_null(r) {
                return Some(Ok(fold_ops::bool_array(None)));
            }
            fold_ops::comparison(op, l, r)
        }
        (Builtin::Not, [a]) => {
            let v = fold_ops::boolean(a)?;
            Some(Ok(fold_ops::bool_array(v.map(|b| !b))))
        }
        (Builtin::IsNull, [a]) => Some(Ok(fold_ops::bool_array(Some(fold_ops::is_null(a))))),
        (Builtin::IsNotNull, [a]) => Some(Ok(fold_ops::bool_array(Some(!fold_ops::is_null(a))))),
        (Builtin::Concat, parts) => {
            if parts.iter().any(|p| p.data_type() == &ArrowType::Null) {
                return Some(Ok(new_null_array(&ArrowType::Utf8, 1)));
            }
            fold_ops::concat(parts)
        }
        _ => None,
    }
}

/// Kleene `AND`/`OR`, left operand first: a decided left operand skips the
/// right one (its error is not raised).
fn logic(and: bool, kids: Vec<Result<ArrayRef, KrishivError>>) -> Folded {
    let [l, r] = <[_; 2]>::try_from(kids).ok()?;
    let l = match l {
        Ok(l) => fold_ops::boolean(&l)?,
        Err(e) => return Some(Err(e)),
    };
    // `false AND x` and `true OR x` are decided by the left operand.
    if l == Some(!and) {
        return Some(Ok(fold_ops::bool_array(l)));
    }
    let r = match r {
        Ok(r) => fold_ops::boolean(&r)?,
        Err(e) => return Some(Err(e)),
    };
    let v = match (l, r) {
        (_, d) if d == Some(!and) => d,
        (None, _) | (_, None) => None,
        _ => Some(and),
    };
    Some(Ok(fold_ops::bool_array(v)))
}

/// `CASE WHEN c1 THEN v1 ... [ELSE e]`: the first true condition's value.
fn when(f: &ResolvedFunction, kids: Vec<Result<ArrayRef, KrishivError>>) -> Folded {
    let mut it = kids.into_iter();
    loop {
        let Some(c) = it.next() else {
            return Some(Ok(new_null_array(&f.data_type.to_arrow(), 1)));
        };
        let Some(v) = it.next() else {
            return Some(c);
        };
        let c = match c {
            Ok(c) => fold_ops::boolean(&c)?,
            Err(e) => return Some(Err(e)),
        };
        if c == Some(true) {
            return Some(v);
        }
    }
}

fn registry(
    name: &str,
    f: &ResolvedFunction,
    kids: Vec<Result<ArrayRef, KrishivError>>,
    conf: Conf,
) -> Folded {
    match name {
        "coalesce" | "ifnull" | "nvl" => {
            for k in kids {
                match k {
                    Ok(a) if fold_ops::is_null(&a) => {}
                    other => return Some(other),
                }
            }
            return Some(Ok(new_null_array(&f.data_type.to_arrow(), 1)));
        }
        "if" => {
            let [c, a, b] = <[_; 3]>::try_from(kids).ok()?;
            return match c.map(|c| fold_ops::boolean(&c)) {
                Ok(Some(Some(true))) => Some(a),
                Ok(Some(_)) => Some(b),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            };
        }
        _ => {}
    }
    if !is_argument_pure(name) {
        return None;
    }
    let args = match values(kids) {
        Ok(a) => a,
        Err(e) => return Some(Err(e)),
    };
    let opts = BindOptions::new()
        .with_ansi(conf.ansi)
        .with_decimal_precision_loss(conf.decimal_loss);
    let bind: Vec<BindArg> = args.iter().cloned().map(BindArg::Literal).collect();
    let run = bind_scalar(name, &bind, &opts).and_then(|func| {
        let call: Vec<Arg> = args.into_iter().map(Arg::Scalar).collect();
        func.invoke(&call, 1)
    });
    match run {
        Ok(a) => Some(Ok(normalized(a))),
        // A binding the kernels lack is not a constant here, not an error.
        Err(e) if e.error_class() == ErrorClass::KrishivUnsupportedFeature => None,
        Err(e) => Some(Err(e)),
    }
}

/// Strings as `Utf8`, the carrier folding works on.
fn normalized(a: ArrayRef) -> ArrayRef {
    match a.data_type() {
        ArrowType::Utf8View => {
            let v = a.as_string_view();
            Arc::new(StringArray::from_iter(v.iter()))
        }
        ArrowType::LargeUtf8 => {
            let v = a.as_string::<i64>();
            Arc::new(StringArray::from_iter(v.iter()))
        }
        _ => a,
    }
}

/// A literal as a one-row array of its Spark type's carrier.
fn literal(l: &Literal) -> Option<ArrayRef> {
    Some(match l {
        Literal::Null(Some(t)) => new_null_array(&t.to_arrow(), 1),
        Literal::Null(None) => new_null_array(&ArrowType::Null, 1),
        Literal::Boolean(v) => Arc::new(BooleanArray::from(vec![*v])),
        Literal::Byte(v) => Arc::new(Int8Array::from(vec![*v])),
        Literal::Short(v) => Arc::new(Int16Array::from(vec![*v])),
        Literal::Integer(v) => Arc::new(Int32Array::from(vec![*v])),
        Literal::Long(v) => Arc::new(Int64Array::from(vec![*v])),
        Literal::Float(v) => Arc::new(Float32Array::from(vec![*v])),
        Literal::Double(v) => Arc::new(Float64Array::from(vec![*v])),
        Literal::Decimal {
            unscaled,
            data_type,
        } => Arc::new(
            Decimal128Array::from(vec![*unscaled])
                .with_precision_and_scale(
                    data_type.precision(),
                    i8::try_from(data_type.scale()).ok()?,
                )
                .ok()?,
        ),
        Literal::String(v) => Arc::new(StringArray::from(vec![v.as_str()])),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;
    use krishiv_types::DecimalType;

    use super::*;

    fn constant(e: &Expr, ansi: bool) -> Option<Constant> {
        super::constant(
            e,
            Conf {
                ansi,
                decimal_loss: true,
            },
        )
    }

    fn lit(v: Literal) -> Expr {
        Expr::new(ExprNode::Literal(v))
    }

    fn cast(v: Literal, to: DataType) -> Expr {
        cast_of(lit(v), to)
    }

    fn cast_of(inner: Expr, to: DataType) -> Expr {
        Expr::new(ExprNode::Cast {
            expr: Arc::new(inner),
            to,
            mode: CastMode::Default,
        })
    }

    fn dec(unscaled: i128, p: u32, s: u32) -> Literal {
        Literal::Decimal {
            unscaled,
            data_type: DecimalType::try_new(p, s).unwrap(),
        }
    }

    fn class(c: Option<Constant>) -> Option<ErrorClass> {
        match c {
            Some(Constant::Error(e)) => Some(e.error_class()),
            _ => None,
        }
    }

    /// Spark 4.2: `round(x, CAST(1.5 AS FLOAT))` has scale 1.
    #[test]
    fn casts_of_literals_to_numbers_fold() {
        let n = |v: i64| Some(Constant::Integral(v));
        let f = |v: i64| Some(Constant::Fractional(v));
        let ansi = true;
        assert_eq!(
            constant(&cast(Literal::Integer(7), DataType::Byte), ansi),
            n(7)
        );
        assert_eq!(
            constant(&cast(Literal::Byte(-3), DataType::Long), ansi),
            n(-3)
        );
        assert_eq!(constant(&cast(dec(15, 2, 1), DataType::Float), ansi), f(1));
        assert_eq!(
            constant(&cast(dec(-15, 2, 1), DataType::Double), ansi),
            f(-1)
        );
        assert_eq!(
            constant(&cast(Literal::Double(2.9), DataType::Float), ansi),
            f(2)
        );
        // A decimal target rounds half up first: 1.5 is 2 at scale 0.
        assert_eq!(
            constant(
                &cast(
                    dec(15, 2, 1),
                    DataType::Decimal(DecimalType::try_new(10, 0).unwrap())
                ),
                ansi
            ),
            f(2)
        );
        assert_eq!(
            constant(
                &cast(
                    dec(15, 2, 1),
                    DataType::Decimal(DecimalType::try_new(10, 2).unwrap())
                ),
                ansi
            ),
            f(1)
        );
        // Strings parse as the cast kernel parses them.
        assert_eq!(
            constant(
                &cast(Literal::String("2.7".to_owned()), DataType::Float),
                ansi
            ),
            f(2)
        );
        assert_eq!(
            constant(
                &cast(Literal::String(" 12 ".to_owned()), DataType::Integer),
                ansi
            ),
            n(12)
        );
        // Casts nest.
        let inner = cast(Literal::Integer(3), DataType::Float);
        assert_eq!(constant(&cast_of(inner, DataType::Double), ansi), f(3));
        // A NULL stays NULL through any cast.
        assert_eq!(
            constant(&cast(Literal::Null(None), DataType::Float), ansi),
            Some(Constant::Null)
        );
        // Literals alone: a fractional one is the INT it casts to.
        assert_eq!(constant(&lit(Literal::Double(-2.5)), ansi), f(-2));
        assert_eq!(constant(&lit(dec(15, 2, 1)), ansi), f(1));
        assert_eq!(constant(&lit(Literal::Long(5)), ansi), n(5));
    }

    /// A cast Spark raises on is an error value, raised by the function
    /// that reads it; legacy mode follows the kernels' legacy rules.
    #[test]
    fn failing_casts_fold_to_errors() {
        let ansi = true;
        assert_eq!(
            class(constant(&cast(Literal::Integer(300), DataType::Byte), ansi)),
            Some(ErrorClass::CastOverflow)
        );
        assert_eq!(
            class(constant(
                &cast(Literal::Double(1e30), DataType::Integer),
                ansi
            )),
            Some(ErrorClass::CastOverflow)
        );
        assert_eq!(
            class(constant(&lit(Literal::Double(1e30)), ansi)),
            Some(ErrorClass::CastOverflow)
        );
        assert_eq!(
            class(constant(
                &cast(Literal::String("x".to_owned()), DataType::Float),
                ansi
            )),
            Some(ErrorClass::CastInvalidInput)
        );
        assert_eq!(
            class(constant(
                &cast(
                    dec(995, 3, 1),
                    DataType::Decimal(DecimalType::try_new(2, 0).unwrap())
                ),
                ansi
            )),
            Some(ErrorClass::NumericValueOutOfRangeWithSuggestion)
        );
        // Not ANSI: the malformed string is NULL, the wrapped byte is 44.
        assert_eq!(
            constant(
                &cast(Literal::String("x".to_owned()), DataType::Float),
                false
            ),
            Some(Constant::Null)
        );
        assert_eq!(
            constant(&cast(Literal::Integer(300), DataType::Byte), false),
            Some(Constant::Integral(44))
        );
    }

    /// What is neither foldable nor a literal is not a constant.
    #[test]
    fn other_expressions_are_not_constants() {
        let ansi = true;
        let col = Expr::new(ExprNode::Column(crate::ColumnRef {
            name: vec!["a".to_owned()],
            plan_id: None,
        }));
        assert_eq!(constant(&col, ansi), None);
        assert_eq!(constant(&cast_of(col, DataType::Float), ansi), None);
    }

    fn call(id: FunctionId, args: Vec<Expr>, data_type: DataType) -> Expr {
        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id,
                kind: FunctionKind::Scalar,
                args: args.into_iter().map(Arc::new).collect(),
                distinct: false,
                filter: None,
                data_type,
                nullable: true,
                deterministic: true,
            },
        )))
    }

    fn op(b: Builtin, args: Vec<Expr>, data_type: DataType) -> Expr {
        call(FunctionId::Builtin(b), args, data_type)
    }

    fn named(name: &str, args: Vec<Expr>, data_type: DataType) -> Expr {
        call(
            FunctionId::Registry {
                name: name.to_owned(),
                id: 0,
            },
            args,
            data_type,
        )
    }

    fn int(v: i32) -> Expr {
        lit(Literal::Integer(v))
    }

    fn text(v: &str) -> Expr {
        lit(Literal::String(v.to_owned()))
    }

    fn truth(v: bool) -> Expr {
        lit(Literal::Boolean(v))
    }

    fn string(v: &str) -> Option<Constant> {
        Some(Constant::String(v.to_owned()))
    }

    /// An expression whose folding raises under ANSI: `1 / 0` over doubles.
    fn failing() -> Expr {
        op(
            Builtin::Divide,
            vec![lit(Literal::Double(1.0)), lit(Literal::Double(0.0))],
            DataType::Double,
        )
    }

    #[test]
    fn arithmetic_folds_through_the_kernels() {
        let sum = |a, b| op(Builtin::Add, vec![int(a), int(b)], DataType::Integer);
        let n = |v: i64| Some(Constant::Integral(v));
        assert_eq!(constant(&sum(1, 2), true), n(3));
        let nested = op(
            Builtin::Multiply,
            vec![sum(1, 2), int(3)],
            DataType::Integer,
        );
        assert_eq!(constant(&nested, true), n(9));
        // Overflow raises under ANSI and wraps in legacy mode.
        assert_eq!(
            class(constant(&sum(i32::MAX, 1), true)),
            Some(ErrorClass::ArithmeticOverflow)
        );
        assert_eq!(constant(&sum(i32::MAX, 1), false), n(i64::from(i32::MIN)));
        let sub = op(Builtin::Subtract, vec![int(1), int(5)], DataType::Integer);
        assert_eq!(constant(&sub, true), n(-4));
        let neg = op(Builtin::Negative, vec![int(5)], DataType::Integer);
        assert_eq!(constant(&neg, true), n(-5));
        let rem = op(Builtin::Remainder, vec![int(7), int(4)], DataType::Integer);
        assert_eq!(constant(&rem, true), n(3));
        let long = |v: i64| lit(Literal::Long(v));
        let div = op(
            Builtin::IntegralDivide,
            vec![long(7), long(2)],
            DataType::Long,
        );
        assert_eq!(constant(&div, true), n(3));
        // Division by zero: an error, or NULL.
        assert_eq!(
            class(constant(&failing(), true)),
            Some(ErrorClass::DivideByZero)
        );
        assert_eq!(constant(&failing(), false), Some(Constant::Null));
        // A double is the integer it truncates to.
        let half = op(
            Builtin::Add,
            vec![lit(Literal::Double(1.5)), lit(Literal::Double(1.5))],
            DataType::Double,
        );
        assert_eq!(constant(&half, true), Some(Constant::Fractional(3)));
        // NULL operands give NULL.
        let null = op(
            Builtin::Add,
            vec![cast(Literal::Null(None), DataType::Integer), int(1)],
            DataType::Integer,
        );
        assert_eq!(constant(&null, true), Some(Constant::Null));
        let mixed = op(
            Builtin::Add,
            vec![int(1), lit(Literal::Long(1))],
            DataType::Long,
        );
        assert_eq!(constant(&mixed, true), None);
    }

    #[test]
    fn comparisons_and_logic_fold() {
        let b = |v: bool| Some(Constant::Boolean(v));
        let lt = |x, y| op(Builtin::Lt, vec![int(x), int(y)], DataType::Boolean);
        assert_eq!(constant(&lt(1, 2), true), b(true));
        assert_eq!(constant(&lt(2, 1), true), b(false));
        let with_null = op(
            Builtin::Eq,
            vec![lit(Literal::Null(None)), int(1)],
            DataType::Boolean,
        );
        assert_eq!(constant(&with_null, true), Some(Constant::Null));
        assert_eq!(
            constant(
                &op(Builtin::Not, vec![truth(true)], DataType::Boolean),
                true
            ),
            b(false)
        );
        assert_eq!(
            constant(&op(Builtin::IsNull, vec![int(1)], DataType::Boolean), true),
            b(false)
        );
        assert_eq!(
            constant(
                &op(
                    Builtin::IsNotNull,
                    vec![lit(Literal::Null(None))],
                    DataType::Boolean
                ),
                true
            ),
            b(false)
        );
        let null = || lit(Literal::Null(Some(DataType::Boolean)));
        let and = |x, y| op(Builtin::And, vec![x, y], DataType::Boolean);
        let or = |x, y| op(Builtin::Or, vec![x, y], DataType::Boolean);
        assert_eq!(constant(&and(truth(true), truth(true)), true), b(true));
        assert_eq!(constant(&and(truth(true), truth(false)), true), b(false));
        assert_eq!(
            constant(&and(truth(true), null()), true),
            Some(Constant::Null)
        );
        assert_eq!(constant(&and(null(), truth(false)), true), b(false));
        assert_eq!(constant(&or(truth(false), truth(false)), true), b(false));
        assert_eq!(constant(&or(null(), truth(true)), true), b(true));
        assert_eq!(
            constant(&or(truth(false), null()), true),
            Some(Constant::Null)
        );
        // The left operand decides, as in Spark: its error is raised, and
        // a decided left operand hides the right one's.
        let boom = || {
            op(
                Builtin::Eq,
                vec![failing(), lit(Literal::Double(0.0))],
                DataType::Boolean,
            )
        };
        assert_eq!(constant(&and(truth(false), boom()), true), b(false));
        assert_eq!(constant(&or(truth(true), boom()), true), b(true));
        assert_eq!(
            class(constant(&and(boom(), truth(false)), true)),
            Some(ErrorClass::DivideByZero)
        );
        assert_eq!(
            class(constant(&and(truth(true), boom()), true)),
            Some(ErrorClass::DivideByZero)
        );
    }

    #[test]
    fn conditionals_evaluate_lazily() {
        let n = |v: i64| Some(Constant::Integral(v));
        let boom = || op(Builtin::Add, vec![int(i32::MAX), int(1)], DataType::Integer);
        let when = |args| op(Builtin::When, args, DataType::Integer);
        assert_eq!(
            constant(&when(vec![truth(true), int(1), boom()]), true),
            n(1)
        );
        assert_eq!(
            constant(&when(vec![truth(false), boom(), int(2)]), true),
            n(2)
        );
        assert_eq!(
            constant(&when(vec![truth(false), int(1), truth(true), int(3)]), true),
            n(3)
        );
        // No branch taken and no ELSE: NULL.
        assert_eq!(
            constant(&when(vec![truth(false), int(1)]), true),
            Some(Constant::Null)
        );
        // The taken branch's error is raised.
        assert_eq!(
            class(constant(&when(vec![truth(true), boom(), int(1)]), true)),
            Some(ErrorClass::ArithmeticOverflow)
        );
        let coalesce = |args| named("coalesce", args, DataType::Integer);
        assert_eq!(
            constant(
                &coalesce(vec![lit(Literal::Null(None)), int(2), boom()]),
                true
            ),
            n(2)
        );
        assert_eq!(
            constant(&coalesce(vec![lit(Literal::Null(None))]), true),
            Some(Constant::Null)
        );
        let iff = |c, a, b| named("if", vec![c, a, b], DataType::Integer);
        assert_eq!(constant(&iff(truth(true), int(1), boom()), true), n(1));
        assert_eq!(constant(&iff(truth(false), boom(), int(2)), true), n(2));
        assert_eq!(
            constant(&iff(lit(Literal::Null(None)), boom(), int(2)), true),
            n(2)
        );
        // One argument that is not foldable keeps the call from folding.
        let col = Expr::new(ExprNode::Column(crate::ColumnRef {
            name: vec!["a".to_owned()],
            plan_id: None,
        }));
        assert_eq!(constant(&iff(truth(true), int(1), col), true), None);
    }

    #[test]
    fn strings_and_registry_functions_fold() {
        let concat = |parts: Vec<Expr>| op(Builtin::Concat, parts, DataType::String);
        assert_eq!(
            constant(&concat(vec![text("a"), text("b")]), true),
            string("ab")
        );
        let null = lit(Literal::Null(None));
        assert_eq!(
            constant(&concat(vec![text("a"), null]), true),
            Some(Constant::Null)
        );
        assert_eq!(
            constant(
                &concat(vec![text("a"), lit(Literal::String(String::new()))]),
                true
            ),
            string("a")
        );
        let cmp = op(Builtin::Lt, vec![text("a"), text("b")], DataType::Boolean);
        assert_eq!(constant(&cmp, true), Some(Constant::Boolean(true)));
        // The cast kernels give no string of a number: not folded.
        assert_eq!(
            constant(&cast(Literal::Integer(12), DataType::String), true),
            None
        );
        // Deterministic string, math and arithmetic functions.
        assert_eq!(
            constant(&named("upper", vec![text("ab")], DataType::String), true),
            string("AB")
        );
        assert_eq!(
            constant(&named("length", vec![text("abc")], DataType::Integer), true),
            Some(Constant::Integral(3))
        );
        let abs = named("abs", vec![int(-2)], DataType::Integer);
        assert_eq!(constant(&abs, true), Some(Constant::Integral(2)));
        // Datetime, session and unknown functions are not evaluated.
        let date = lit(Literal::Date(0));
        assert_eq!(
            constant(&named("date_add", vec![date, int(1)], DataType::Date), true),
            None
        );
        assert_eq!(
            constant(&named("nosuchfn", vec![int(1)], DataType::Integer), true),
            None
        );
        // A function the kernels lack for the argument types is not folded.
        assert_eq!(
            constant(&named("abs", vec![text("x")], DataType::Integer), true),
            None
        );
        // Nondeterministic, `DISTINCT` and filtered calls are not constants.
        let mut f = ResolvedFunction {
            id: FunctionId::Registry {
                name: "abs".to_owned(),
                id: 0,
            },
            kind: FunctionKind::Scalar,
            args: vec![Arc::new(int(1))],
            distinct: false,
            filter: None,
            data_type: DataType::Integer,
            nullable: false,
            deterministic: false,
        };
        let expr =
            |f: &ResolvedFunction| Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f.clone())));
        assert_eq!(constant(&expr(&f), true), None);
        f.deterministic = true;
        assert_eq!(constant(&expr(&f), true), Some(Constant::Integral(1)));
        f.distinct = true;
        assert_eq!(constant(&expr(&f), true), None);
        f.distinct = false;
        f.filter = Some(Arc::new(truth(true)));
        assert_eq!(constant(&expr(&f), true), None);
        f.filter = None;
        f.kind = FunctionKind::Aggregate;
        assert_eq!(constant(&expr(&f), true), None);
    }

    #[test]
    fn a_literal_is_not_a_folded_expression() {
        assert!(is_literal(&truth(true)));
        assert!(!is_literal(&op(
            Builtin::Not,
            vec![truth(false)],
            DataType::Boolean
        )));
        assert!(!is_literal(&cast(Literal::Integer(1), DataType::Long)));
        // `constant` folds the second; the flag of `first` needs the first.
        assert_eq!(
            constant(
                &op(Builtin::Not, vec![truth(false)], DataType::Boolean),
                true
            ),
            Some(Constant::Boolean(true))
        );
    }
}
