// SPDX-License-Identifier: Apache-2.0
//! The math and arithmetic functions' analysis signatures: the `double`
//! functions, `round` and its relatives, `pmod`, `greatest`/`least`, `abs`
//! and the `try_*` arithmetic. The arithmetic ones are typed as the
//! operators are ([`Inputs::Operator`]).

use krishiv_common::error::{DatatypeMismatchNonFoldableInput, KrishivError, NonFoldableArgument};
use krishiv_kernels::decimal::{DecimalOp, result_type};
use krishiv_kernels::math::round_decimal_type;
use krishiv_types::{DataType, DecimalType};

use super::defs::{Spec, exact, scalar, typed};
use crate::signature::{ArgType, CallTypes, Constant, InputBuilder, InputType, Inputs, Operator};

// ---- the `double` functions ----

/// A function of one `double`: always nullable. Spark 4.2's
/// `UnaryMathExpression` declares `nullable = true` for every one of them
/// (not only the logarithms), so `sqrt(id)` over a non-nullable column is
/// still nullable in the result schema.
fn unary(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Double, true))
}

/// A function of two `double`s: NULL when an argument is.
fn binary(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Double, c)
}

fn constant(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Double, false))
}

// ---- rounding ----

/// The constant scale of `round`/`ceil`/`floor` (argument `i`): Spark folds
/// the argument, a column is `NON_FOLDABLE_ARGUMENT`.
fn scale_of(c: &CallTypes<'_>, function: &str, i: usize) -> Result<i32, KrishivError> {
    let clamp = |n: i64| i32::try_from(n).unwrap_or(if n < 0 { i32::MIN } else { i32::MAX });
    match c.constant(i) {
        Some(Constant::Integral(n) | Constant::Fractional(n)) => Ok(clamp(*n)),
        Some(Constant::String(s)) => Ok(s.trim().parse::<f64>().map_or(0, |v| {
            // Spark casts the folded value to `int`.
            #[expect(clippy::cast_possible_truncation, reason = "a saturating cast")]
            clamp(v as i64)
        })),
        Some(Constant::Null) => Ok(0),
        Some(Constant::Error(e)) => Err(e.clone()),
        _ => Err(non_foldable_input(c, function, i)),
    }
}

/// `DATATYPE_MISMATCH.NON_FOLDABLE_INPUT`: `round`/`bround` check their
/// scale as an input type (`ceil`/`floor` raise `NON_FOLDABLE_ARGUMENT`).
fn non_foldable_input(c: &CallTypes<'_>, function: &str, i: usize) -> KrishivError {
    let args: Vec<String> = (0..c.args().len())
        .map(|n| c.sql(n).unwrap_or_default())
        .collect();
    DatatypeMismatchNonFoldableInput {
        sql_expr: format!("\"{function}({})\"", args.join(", ")),
        input_name: "`scale`".to_owned(),
        input_type: "\"INT\"".to_owned(),
        input_expr: format!("\"{}\"", c.sql(i).unwrap_or_default()),
    }
    .into()
}

/// The scale of `ceil`/`floor`: an `int` literal.
fn int_scale(c: &CallTypes<'_>, function: &str) -> Result<i32, KrishivError> {
    match (c.data_type(1), c.constant(1)) {
        (Some(DataType::Integer), Some(Constant::Error(e))) => Err(e.clone()),
        (Some(DataType::Integer), Some(Constant::Integral(n))) => {
            Ok(i32::try_from(*n).unwrap_or(0))
        }
        _ => Err(non_foldable(function)),
    }
}

fn non_foldable(function: &str) -> KrishivError {
    NonFoldableArgument {
        func_name: format!("`{function}`"),
        param_name: "`scale`".to_owned(),
        param_type: "\"INT\"".to_owned(),
    }
    .into()
}

/// `decimal(p, s)` rounded to `scale` digits.
fn rounded(d: DecimalType, scale: i32) -> Result<DataType, KrishivError> {
    let s = i8::try_from(d.scale()).unwrap_or(i8::MAX);
    let (p, s) = round_decimal_type(d.precision(), s, scale);
    Ok(DataType::Decimal(DecimalType::try_new(
        u32::from(p),
        u32::try_from(s).unwrap_or(0),
    )?))
}

/// `round(x, d)`, `bround(x, d)`: the type of `x`; a decimal's precision
/// follows the scale. NULL for a NULL scale or argument: always nullable.
fn round_type(function: &str, c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    // The scale is checked whatever the type of `x`.
    let scale = if c.args().len() > 1 {
        scale_of(c, function, 1)?
    } else {
        0
    };
    let t = match c.data_type(0) {
        Some(DataType::Decimal(d)) => rounded(*d, scale)?,
        Some(t) => t.clone(),
        None => DataType::Double,
    };
    Ok(ArgType::new(t, true))
}

fn round_fn(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    round_type("round", c)
}

fn bround_fn(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    round_type("bround", c)
}

/// `DecimalType.forType` of an integral type.
fn integral_decimal(t: &DataType) -> Option<DataType> {
    let (p, s) = match t {
        DataType::Byte => (3, 0),
        DataType::Short => (5, 0),
        DataType::Integer => (10, 0),
        DataType::Long => (20, 0),
        _ => return None,
    };
    DecimalType::try_new(p, s).ok().map(DataType::Decimal)
}

fn decimal(p: u32, s: u32) -> InputType {
    DecimalType::try_new(p, s).map_or(InputType::Any, |d| exact(DataType::Decimal(d)))
}

/// `ceil(x)` / `floor(x)`: a decimal stays, an integral type is `bigint`,
/// everything else `double`. With a scale, `x` is a decimal: a float is
/// `decimal(14,7)`, a double `decimal(30,15)`, a string `decimal(38,18)`.
fn ceil_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let t = c.data_type(0);
    let first = if c.args().len() == 1 {
        match t {
            Some(DataType::Decimal(_)) => InputType::Any,
            Some(DataType::Byte | DataType::Short | DataType::Integer | DataType::Long) => {
                exact(DataType::Long)
            }
            _ => exact(DataType::Double),
        }
    } else {
        match t {
            Some(DataType::Decimal(_)) => InputType::Any,
            Some(DataType::Float) => decimal(14, 7),
            Some(DataType::Double) => decimal(30, 15),
            Some(other) => integral_decimal(other).map_or_else(|| decimal(38, 18), exact),
            None => decimal(38, 18),
        }
    };
    // Spark folds only an `int` literal for the scale (checked after the
    // arguments' types, by the result rule).
    let int_literal = c.data_type(1) == Some(&DataType::Integer)
        && matches!(c.constant(1), Some(Constant::Integral(_)));
    vec![
        first,
        if int_literal {
            exact(DataType::Integer)
        } else {
            InputType::Any
        },
    ]
}

fn ceil_type(function: &str, c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let t = match c.data_type(0) {
        Some(DataType::Decimal(d)) => {
            let scale = if c.args().len() > 1 {
                int_scale(c, function)?
            } else {
                0
            };
            rounded(*d, scale)?
        }
        _ => DataType::Long,
    };
    Ok(ArgType::new(t, true))
}

fn ceil_fn(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    ceil_type("ceil", c)
}

fn floor_fn(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    ceil_type("floor", c)
}

// ---- greatest, least, abs ----

/// `greatest`/`least`: the arguments' common type; NULL only when all are.
fn extremum(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: c.args().iter().find_map(|t| t.data_type.clone()),
        nullable: c.args().iter().all(|t| t.nullable),
    })
}

/// `abs`, `positive`, `negative`: the argument's type (a string or NULL is `double`).
fn same_type(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType {
        data_type: Some(c.data_type(0).cloned().unwrap_or(DataType::Double)),
        nullable: c.any_nullable(),
    })
}

// ---- arithmetic ----

/// The result type of `a op b` for operands already cast to one type: the
/// decimal rules for decimals, `double` for `/`, else the operand type.
fn arithmetic_type(op: Operator, c: &CallTypes<'_>) -> Result<DataType, KrishivError> {
    let dop = match op {
        Operator::Add => DecimalOp::Add,
        Operator::Subtract => DecimalOp::Subtract,
        Operator::Multiply => DecimalOp::Multiply,
        Operator::Divide => DecimalOp::Divide,
        Operator::Remainder => DecimalOp::Remainder,
    };
    let pair = |d: &DecimalType| (d.precision(), i8::try_from(d.scale()).unwrap_or(i8::MAX));
    Ok(match (c.data_type(0), c.data_type(1)) {
        (Some(DataType::Decimal(a)), Some(DataType::Decimal(b))) => {
            let (p, s) = result_type(dop, pair(a), pair(b), c.decimal_loss())?;
            DataType::Decimal(DecimalType::try_new(
                u32::from(p),
                u32::try_from(s).unwrap_or(0),
            )?)
        }
        _ if op == Operator::Divide => DataType::Double,
        (Some(t), _) => t.clone(),
        (None, _) => DataType::Double,
    })
}

/// `try_*`, `mod`, `pmod`: NULL on overflow or a zero divisor.
macro_rules! arithmetic {
    ($name:ident, $op:expr) => {
        fn $name(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
            Ok(ArgType::new(arithmetic_type($op, c)?, true))
        }
    };
}
arithmetic!(try_add, Operator::Add);
arithmetic!(try_subtract, Operator::Subtract);
arithmetic!(try_multiply, Operator::Multiply);
arithmetic!(try_divide, Operator::Divide);
arithmetic!(remainder, Operator::Remainder);

pub(super) fn specs() -> Vec<Spec> {
    let double = Inputs::Expects(vec![exact(DataType::Double)]);
    let mut out = scalar(
        &[
            "sqrt", "cbrt", "exp", "expm1", "ln", "log10", "log2", "log1p", "sin", "cos", "tan",
            "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "cot",
            "csc", "sec", "degrees", "radians", "rint", "sign", "signum", "log",
        ],
        &double,
        unary,
    );
    // `log(base, x)` is NULL outside its domain too.
    out.extend(scalar(&["pow", "power", "atan2", "hypot"], &double, binary));
    out.extend(scalar(&["pi", "e"], &Inputs::Any, constant));
    let numeric = || InputType::Numeric { intervals: false };
    let rounding = Inputs::Expects(vec![numeric(), exact(DataType::Integer)]);
    out.extend(scalar(&["round"], &rounding, round_fn));
    out.extend(scalar(&["bround"], &rounding, bround_fn));
    let ceil = Inputs::Built(InputBuilder(ceil_inputs));
    out.extend(scalar(&["ceil", "ceiling"], &ceil, ceil_fn));
    out.extend(scalar(&["floor"], &ceil, floor_fn));
    out.extend(scalar(
        &["greatest", "least"],
        &Inputs::CommonNoStrings,
        extremum,
    ));
    let numeric_or_interval = Inputs::Expects(vec![InputType::Numeric { intervals: true }]);
    out.extend(scalar(
        &["abs", "positive", "negative"],
        &numeric_or_interval,
        same_type,
    ));
    let op = Inputs::Operator;
    let try_op = Inputs::TryOperator;
    out.extend(scalar(&["try_add"], &try_op(Operator::Add), try_add));
    out.extend(scalar(
        &["try_subtract"],
        &try_op(Operator::Subtract),
        try_subtract,
    ));
    out.extend(scalar(
        &["try_multiply"],
        &try_op(Operator::Multiply),
        try_multiply,
    ));
    out.extend(scalar(
        &["try_divide"],
        &try_op(Operator::Divide),
        try_divide,
    ));
    out.extend(scalar(
        &["try_mod"],
        &try_op(Operator::Remainder),
        remainder,
    ));
    out.extend(scalar(
        &["mod", "pmod"],
        &op(Operator::Remainder),
        remainder,
    ));
    out
}
