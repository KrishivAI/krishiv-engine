// SPDX-License-Identifier: Apache-2.0
//! Constant folding and boolean simplification of expressions.
//!
//! Folding never changes what a query returns or raises: a call is folded
//! only where its result does not depend on session state outside the plan.
//! Arithmetic folds only when no overflow happens (so ANSI and legacy agree)
//! and never divides; a cast folds through the cast kernels in the cast's
//! own mode, except casts involving `TIMESTAMP` (they read the session time
//! zone); a call that would raise stays in the plan, so the error surfaces
//! only if a row reaches it, as in Spark.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType as ArrowType, TimeUnit};
use krishiv_kernels::cast::{CastMode as KernelMode, CastOptions, FixedOffset, cast};
use krishiv_plan::{Builtin, CastMode, Expr, ExprNode, Literal, ResolvedFunction};
use krishiv_types::{DataType, DecimalType};

use crate::expr::{builtin, data_type, is_deterministic, lit, nullable, recall, same};

/// `e` with every foldable subexpression folded; `e` itself when nothing
/// changes.
pub(crate) fn fold(e: &Arc<Expr>) -> Arc<Expr> {
    let out = Arc::clone(e).transform_up(&mut |x| Ok::<_, std::convert::Infallible>(simplify(x)));
    match out {
        Ok(v) => v,
        Err(never) => match never {},
    }
}

fn literal(e: &Expr) -> Option<&Literal> {
    match &e.node {
        ExprNode::Literal(l) => Some(l),
        _ => None,
    }
}

fn boolean(e: &Expr) -> Option<Option<bool>> {
    match literal(e)? {
        Literal::Boolean(b) => Some(Some(*b)),
        Literal::Null(_) => Some(None),
        _ => None,
    }
}

fn bool_lit(v: Option<bool>) -> Arc<Expr> {
    lit(v.map_or(Literal::Null(Some(DataType::Boolean)), Literal::Boolean))
}

/// Builtins whose result is NULL whenever an argument is NULL.
fn null_intolerant(op: Builtin) -> bool {
    matches!(
        op,
        Builtin::Eq
            | Builtin::Lt
            | Builtin::LtEq
            | Builtin::Gt
            | Builtin::GtEq
            | Builtin::Add
            | Builtin::Subtract
            | Builtin::Multiply
            | Builtin::Divide
            | Builtin::Remainder
            | Builtin::IntegralDivide
            | Builtin::Negative
            | Builtin::Like
            | Builtin::ILike
            | Builtin::RLike
            | Builtin::Concat
    )
}

/// One folding step at the root of `e`, whose children are folded.
fn simplify(e: Arc<Expr>) -> Arc<Expr> {
    if let ExprNode::Cast { expr, to, mode } = &e.node {
        if data_type(expr).as_ref() == Some(to) {
            return Arc::clone(expr);
        }
        return literal(expr)
            .and_then(|l| fold_cast(l, to, *mode))
            .map_or(e, lit);
    }
    let Some((op, args, f)) = builtin(&e) else {
        return e;
    };
    if null_intolerant(op)
        && args
            .iter()
            .any(|a| matches!(literal(a), Some(Literal::Null(_))))
    {
        return lit(Literal::Null(Some(f.data_type.clone())));
    }
    let folded = match (op, args) {
        (Builtin::And, [a, b]) => match (boolean(a), boolean(b)) {
            (Some(Some(false)), _) | (_, Some(Some(false))) => Some(bool_lit(Some(false))),
            (Some(Some(true)), _) => Some(Arc::clone(b)),
            (_, Some(Some(true))) => Some(Arc::clone(a)),
            (Some(None), Some(None)) => Some(bool_lit(None)),
            _ if idempotent(a, b) => Some(Arc::clone(a)),
            _ => None,
        },
        (Builtin::Or, [a, b]) => match (boolean(a), boolean(b)) {
            (Some(Some(true)), _) | (_, Some(Some(true))) => Some(bool_lit(Some(true))),
            (Some(Some(false)), _) => Some(Arc::clone(b)),
            (_, Some(Some(false))) => Some(Arc::clone(a)),
            (Some(None), Some(None)) => Some(bool_lit(None)),
            _ if idempotent(a, b) => Some(Arc::clone(a)),
            _ => None,
        },
        (Builtin::Not, [a]) => not(a),
        (Builtin::IsNull | Builtin::IsNotNull, [a]) => {
            let is_null = match literal(a) {
                Some(l) => Some(matches!(l, Literal::Null(_))),
                None => (!nullable(a)).then_some(false),
            };
            is_null.map(|n| bool_lit(Some(n == (op == Builtin::IsNull))))
        }
        (Builtin::Eq | Builtin::Lt | Builtin::LtEq | Builtin::Gt | Builtin::GtEq, [a, b]) => {
            compare(a, b).map(|o| bool_lit(Some(test(op, o))))
        }
        (Builtin::EqNullSafe, [a, b]) => match (literal(a), literal(b)) {
            (Some(Literal::Null(_)), Some(Literal::Null(_))) => Some(bool_lit(Some(true))),
            (Some(Literal::Null(_)), Some(_)) | (Some(_), Some(Literal::Null(_))) => {
                Some(bool_lit(Some(false)))
            }
            _ => compare(a, b).map(|o| bool_lit(Some(o == Ordering::Equal))),
        },
        (Builtin::In, [v, list @ ..]) => fold_in(v, list),
        (Builtin::Between, [v, lo, hi]) => {
            let ge = compare_lit(v, lo).map(|o| o.map(|o| o != Ordering::Less));
            let le = compare_lit(v, hi).map(|o| o.map(|o| o != Ordering::Greater));
            match (ge, le) {
                (Some(Some(false)), _) | (_, Some(Some(false))) => Some(bool_lit(Some(false))),
                (Some(x), Some(y)) => Some(bool_lit(x.and(y))),
                _ => None,
            }
        }
        (Builtin::Add | Builtin::Subtract | Builtin::Multiply, [a, b]) => {
            arith(op, a, b, &f.data_type)
        }
        (Builtin::Negative, [a]) => negate(a, &f.data_type),
        (Builtin::When, branches) => when(branches, f),
        _ => None,
    };
    folded.unwrap_or(e)
}

/// `not(a)`: a literal, a double negation or an inverted comparison.
fn not(a: &Arc<Expr>) -> Option<Arc<Expr>> {
    if let Some(v) = boolean(a) {
        return Some(bool_lit(v.map(|b| !b)));
    }
    let (op, args, f) = builtin(a)?;
    let inverse = match op {
        Builtin::Not => return args.first().cloned(),
        Builtin::Lt => Builtin::GtEq,
        Builtin::LtEq => Builtin::Gt,
        Builtin::Gt => Builtin::LtEq,
        Builtin::GtEq => Builtin::Lt,
        _ => return None,
    };
    Some(recall(f, inverse, args.to_vec()))
}

fn test(op: Builtin, o: Ordering) -> bool {
    match op {
        Builtin::Eq => o == Ordering::Equal,
        Builtin::Lt => o == Ordering::Less,
        Builtin::LtEq => o != Ordering::Greater,
        Builtin::Gt => o == Ordering::Greater,
        _ => o != Ordering::Less,
    }
}

/// How two non-null literals compare, when Spark's comparison of them is
/// known here.
fn compare(a: &Expr, b: &Expr) -> Option<Ordering> {
    compare_lit(a, b).flatten()
}

/// `Some(None)` when either side is NULL.
fn compare_lit(a: &Expr, b: &Expr) -> Option<Option<Ordering>> {
    let (x, y) = (literal(a)?, literal(b)?);
    if matches!(x, Literal::Null(_)) || matches!(y, Literal::Null(_)) {
        return Some(None);
    }
    compare_values(x, y).map(Some)
}

/// Spark's ordering of two non-null literals of one type: NaN is equal to
/// itself and above every other value, `-0.0` equals `0.0`, strings and
/// binaries compare by bytes.
pub(crate) fn compare_values(x: &Literal, y: &Literal) -> Option<Ordering> {
    fn float(a: f64, b: f64) -> Ordering {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
        }
    }
    Some(match (x, y) {
        (Literal::Boolean(a), Literal::Boolean(b)) => a.cmp(b),
        (Literal::Byte(a), Literal::Byte(b)) => a.cmp(b),
        (Literal::Short(a), Literal::Short(b)) => a.cmp(b),
        (Literal::Integer(a), Literal::Integer(b)) | (Literal::Date(a), Literal::Date(b)) => {
            a.cmp(b)
        }
        (Literal::Long(a), Literal::Long(b))
        | (Literal::Timestamp(a), Literal::Timestamp(b))
        | (Literal::TimestampNtz(a), Literal::TimestampNtz(b)) => a.cmp(b),
        (Literal::Float(a), Literal::Float(b)) => float(f64::from(*a), f64::from(*b)),
        (Literal::Double(a), Literal::Double(b)) => float(*a, *b),
        (Literal::String(a), Literal::String(b)) => a.as_bytes().cmp(b.as_bytes()),
        (Literal::Binary(a), Literal::Binary(b)) => a.cmp(b),
        (
            Literal::Decimal {
                unscaled: a,
                data_type: s,
            },
            Literal::Decimal {
                unscaled: b,
                data_type: t,
            },
        ) => {
            let scale = s.scale().max(t.scale());
            let up =
                |v: i128, from: u8| v.checked_mul(10_i128.checked_pow(u32::from(scale - from))?);
            up(*a, s.scale())?.cmp(&up(*b, t.scale())?)
        }
        _ => return None,
    })
}

/// `v IN (list)` over literals: true on a match, else NULL if any side is
/// NULL, else false.
fn fold_in(v: &Expr, list: &[Arc<Expr>]) -> Option<Arc<Expr>> {
    // `x IN ()` is false, even for a NULL `x` (Spark 3.5+).
    if list.is_empty() {
        return Some(bool_lit(Some(false)));
    }
    if matches!(literal(v)?, Literal::Null(_)) {
        return Some(bool_lit(None));
    }
    let mut saw_null = false;
    for item in list {
        match compare_lit(v, item)? {
            Some(Ordering::Equal) => return Some(bool_lit(Some(true))),
            Some(_) => {}
            None => saw_null = true,
        }
    }
    Some(bool_lit((!saw_null).then_some(false)))
}

/// Integral `+ - *` of two literals of the result type, when it does not
/// overflow.
fn arith(op: Builtin, a: &Expr, b: &Expr, to: &DataType) -> Option<Arc<Expr>> {
    let (x, y) = (int_value(literal(a)?, to)?, int_value(literal(b)?, to)?);
    let v = match op {
        Builtin::Add => x.checked_add(y),
        Builtin::Subtract => x.checked_sub(y),
        _ => x.checked_mul(y),
    }?;
    int_literal(v, to).map(lit)
}

fn negate(a: &Expr, to: &DataType) -> Option<Arc<Expr>> {
    int_literal(int_value(literal(a)?, to)?.checked_neg()?, to).map(lit)
}

/// The value of an integral literal of type `to`.
fn int_value(l: &Literal, to: &DataType) -> Option<i64> {
    match (l, to) {
        (Literal::Byte(v), DataType::Byte) => Some(i64::from(*v)),
        (Literal::Short(v), DataType::Short) => Some(i64::from(*v)),
        (Literal::Integer(v), DataType::Integer) => Some(i64::from(*v)),
        (Literal::Long(v), DataType::Long) => Some(*v),
        _ => None,
    }
}

fn int_literal(v: i64, to: &DataType) -> Option<Literal> {
    Some(match to {
        DataType::Byte => Literal::Byte(i8::try_from(v).ok()?),
        DataType::Short => Literal::Short(i16::try_from(v).ok()?),
        DataType::Integer => Literal::Integer(i32::try_from(v).ok()?),
        _ => Literal::Long(v),
    })
}

/// `CASE`: drops branches whose condition is a false or NULL literal and
/// ends at the first true one. Folds to a single value only when it has
/// the call's type.
fn when(args: &[Arc<Expr>], f: &ResolvedFunction) -> Option<Arc<Expr>> {
    let to = &f.data_type;
    let mut kept = Vec::new();
    let mut changed = false;
    let mut result = None;
    for chunk in args.chunks(2) {
        match chunk {
            [c, v] => match boolean(c) {
                Some(Some(true)) => {
                    result = Some(Arc::clone(v));
                    changed = true;
                    break;
                }
                Some(_) => changed = true,
                None => kept.extend([Arc::clone(c), Arc::clone(v)]),
            },
            [otherwise] => result = Some(Arc::clone(otherwise)),
            _ => {}
        }
    }
    if kept.is_empty() {
        let value = result.unwrap_or_else(|| lit(Literal::Null(Some(to.clone()))));
        return (data_type(&value).as_ref() == Some(to)).then_some(value);
    }
    if !changed {
        return None;
    }
    kept.extend(result);
    // Dropping branches never makes the result null where it was not.
    Some(recall(f, Builtin::When, kept))
}

/// `a and a` / `a or a` is `a` when `a` is deterministic.
fn idempotent(a: &Expr, b: &Expr) -> bool {
    same(a, b) && is_deterministic(a)
}

/// The type of a literal.
pub(crate) fn literal_type(l: &Literal) -> Option<DataType> {
    Some(match l {
        Literal::Null(t) => return t.clone(),
        Literal::Boolean(_) => DataType::Boolean,
        Literal::Byte(_) => DataType::Byte,
        Literal::Short(_) => DataType::Short,
        Literal::Integer(_) => DataType::Integer,
        Literal::Long(_) => DataType::Long,
        Literal::Float(_) => DataType::Float,
        Literal::Double(_) => DataType::Double,
        Literal::Decimal { data_type, .. } => DataType::Decimal(*data_type),
        Literal::String(_) => DataType::String,
        Literal::Binary(_) => DataType::Binary,
        Literal::Date(_) => DataType::Date,
        Literal::Timestamp(_) => DataType::Timestamp,
        Literal::TimestampNtz(_) => DataType::TimestampNtz,
        _ => return None,
    })
}

/// The Arrow type a cast to `t` folds through; `None` for targets the cast
/// kernels lack (boolean, binary) or that read the session time zone.
fn arrow_type(t: &DataType) -> Option<ArrowType> {
    Some(match t {
        DataType::String => ArrowType::Utf8,
        DataType::Binary | DataType::Boolean | DataType::Timestamp => return None,
        other => other.to_arrow(),
    })
}

fn to_array(l: &Literal) -> Option<ArrayRef> {
    Some(match l {
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
        Literal::Date(v) => Arc::new(Date32Array::from(vec![*v])),
        Literal::TimestampNtz(v) => Arc::new(TimestampMicrosecondArray::from(vec![*v])),
        _ => return None,
    })
}

fn from_array(a: &dyn Array, t: &DataType) -> Option<Literal> {
    if a.len() != 1 {
        return None;
    }
    if a.is_null(0) {
        return Some(Literal::Null(Some(t.clone())));
    }
    Some(match (t, a.data_type()) {
        (DataType::Byte, _) => Literal::Byte(a.as_primitive_opt::<Int8Type>()?.value(0)),
        (DataType::Short, _) => Literal::Short(a.as_primitive_opt::<Int16Type>()?.value(0)),
        (DataType::Integer, _) => Literal::Integer(a.as_primitive_opt::<Int32Type>()?.value(0)),
        (DataType::Long, _) => Literal::Long(a.as_primitive_opt::<Int64Type>()?.value(0)),
        (DataType::Float, _) => Literal::Float(a.as_primitive_opt::<Float32Type>()?.value(0)),
        (DataType::Double, _) => Literal::Double(a.as_primitive_opt::<Float64Type>()?.value(0)),
        (DataType::Decimal(d), ArrowType::Decimal128(p, s))
            if *p == d.precision() && i8::try_from(d.scale()).ok() == Some(*s) =>
        {
            Literal::Decimal {
                unscaled: a.as_primitive_opt::<Decimal128Type>()?.value(0),
                data_type: DecimalType::try_new(u32::from(*p), u32::from(d.scale())).ok()?,
            }
        }
        (DataType::String, ArrowType::Utf8) => {
            Literal::String(a.as_string_opt::<i32>()?.value(0).to_owned())
        }
        (DataType::Date, _) => Literal::Date(a.as_primitive_opt::<Date32Type>()?.value(0)),
        (DataType::TimestampNtz, ArrowType::Timestamp(TimeUnit::Microsecond, None)) => {
            Literal::TimestampNtz(a.as_primitive_opt::<TimestampMicrosecondType>()?.value(0))
        }
        _ => return None,
    })
}

/// `CAST(l AS to)` through the cast kernels; `None` when the cast reads
/// the session time zone, is unsupported here, or raises.
fn fold_cast(l: &Literal, to: &DataType, mode: CastMode) -> Option<Literal> {
    let mode = match mode {
        CastMode::Ansi => KernelMode::Ansi,
        CastMode::Legacy => KernelMode::Legacy,
        CastMode::Try => KernelMode::Try,
        _ => return None,
    };
    if let Literal::Null(_) = l {
        return Some(Literal::Null(Some(to.clone())));
    }
    let target = arrow_type(to)?;
    let input = to_array(l)?;
    let opts = CastOptions {
        mode,
        time_zone: &FixedOffset(0),
    };
    let out = cast(input.as_ref(), &target, &opts).ok()?;
    from_array(out.as_ref(), to)
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use krishiv_plan::{Builtin, CastMode, Literal};
    use krishiv_types::{DataType, DecimalType};

    use super::{arith, compare_values, fold_cast, lit, negate};

    fn dec(unscaled: i128, p: u32, s: u32) -> Literal {
        Literal::Decimal {
            unscaled,
            data_type: DecimalType::try_new(p, s).unwrap(),
        }
    }

    #[test]
    fn casts_fold_through_the_kernels() {
        let d52 = DataType::Decimal(DecimalType::try_new(5, 2).unwrap());
        let ansi = CastMode::Ansi;
        let cases = [
            (
                Literal::Integer(7),
                DataType::Long,
                ansi,
                Some(Literal::Long(7)),
            ),
            (
                Literal::Long(7),
                DataType::Short,
                ansi,
                Some(Literal::Short(7)),
            ),
            (
                Literal::Byte(-3),
                DataType::Integer,
                ansi,
                Some(Literal::Integer(-3)),
            ),
            (Literal::Short(300), DataType::Byte, ansi, None),
            (
                Literal::Short(300),
                DataType::Byte,
                CastMode::Legacy,
                Some(Literal::Byte(44)),
            ),
            (
                Literal::Short(300),
                DataType::Byte,
                CastMode::Try,
                Some(Literal::Null(Some(DataType::Byte))),
            ),
            (
                Literal::Integer(7),
                DataType::Double,
                ansi,
                Some(Literal::Double(7.0)),
            ),
            (
                Literal::Float(1.5),
                DataType::Double,
                ansi,
                Some(Literal::Double(1.5)),
            ),
            (
                Literal::Double(2.5),
                DataType::Float,
                ansi,
                Some(Literal::Float(2.5)),
            ),
            (Literal::Integer(7), d52.clone(), ansi, Some(dec(700, 5, 2))),
            (
                dec(150, 5, 2),
                DataType::Integer,
                ansi,
                Some(Literal::Integer(1)),
            ),
            (
                Literal::String("1.25".into()),
                d52,
                ansi,
                Some(dec(125, 5, 2)),
            ),
            (
                Literal::Boolean(true),
                DataType::Integer,
                ansi,
                Some(Literal::Integer(1)),
            ),
            (Literal::Integer(0), DataType::Boolean, ansi, None),
            (Literal::String("ab".into()), DataType::Binary, ansi, None),
            (
                Literal::Date(1),
                DataType::TimestampNtz,
                ansi,
                Some(Literal::TimestampNtz(86_400_000_000)),
            ),
            (
                Literal::TimestampNtz(86_400_000_000),
                DataType::Date,
                ansi,
                Some(Literal::Date(1)),
            ),
            (
                Literal::Date(0),
                DataType::String,
                ansi,
                Some(Literal::String("1970-01-01".into())),
            ),
            (Literal::Integer(1), DataType::Timestamp, ansi, None),
            (Literal::Timestamp(0), DataType::Date, ansi, None),
            (
                Literal::Null(None),
                DataType::Integer,
                ansi,
                Some(Literal::Null(Some(DataType::Integer))),
            ),
            (Literal::Integer(1), DataType::Long, CastMode::Default, None),
        ];
        let wrong: Vec<_> = cases
            .into_iter()
            .filter(|(from, to, mode, want)| fold_cast(from, to, *mode) != *want)
            .map(|(from, to, mode, _)| (fold_cast(&from, &to, mode), from, to, mode))
            .collect();
        assert!(wrong.is_empty(), "{wrong:?}");
    }

    #[test]
    fn literals_compare_as_spark_orders_them() {
        let nan = f64::NAN;
        let cases = [
            (
                Literal::Boolean(false),
                Literal::Boolean(true),
                Some(Ordering::Less),
            ),
            (Literal::Byte(2), Literal::Byte(1), Some(Ordering::Greater)),
            (Literal::Short(1), Literal::Short(1), Some(Ordering::Equal)),
            (
                Literal::Timestamp(5),
                Literal::Timestamp(6),
                Some(Ordering::Less),
            ),
            (
                Literal::Binary(vec![1]),
                Literal::Binary(vec![1, 0]),
                Some(Ordering::Less),
            ),
            (
                Literal::Double(nan),
                Literal::Double(1.0),
                Some(Ordering::Greater),
            ),
            (
                Literal::Double(1.0),
                Literal::Double(nan),
                Some(Ordering::Less),
            ),
            (
                Literal::Float(-0.0),
                Literal::Float(0.0),
                Some(Ordering::Equal),
            ),
            (dec(150, 5, 2), dec(15, 3, 1), Some(Ordering::Equal)),
            (Literal::Integer(1), Literal::Long(1), None),
        ];
        for (a, b, want) in cases {
            assert_eq!(compare_values(&a, &b), want, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn small_integers_fold_without_overflow() {
        let (a, b) = (lit(Literal::Byte(100)), lit(Literal::Byte(27)));
        let sum = arith(Builtin::Add, &a, &b, &DataType::Byte).unwrap();
        assert_eq!(sum.node, lit(Literal::Byte(127)).node);
        let c = lit(Literal::Byte(28));
        assert!(arith(Builtin::Add, &a, &c, &DataType::Byte).is_none());
        let s = lit(Literal::Short(i16::MIN));
        assert!(negate(&s, &DataType::Short).is_none());
        let s = lit(Literal::Short(5));
        assert_eq!(
            negate(&s, &DataType::Short).unwrap().node,
            lit(Literal::Short(-5)).node
        );
    }
}
