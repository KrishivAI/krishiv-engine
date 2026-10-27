// SPDX-License-Identifier: Apache-2.0
//! The value of a constant argument, for the functions whose typing reads
//! it (`round`'s scale, `extract`'s field, a `LIMIT`).
//!
//! Spark folds a foldable argument before it types the call, so
//! `round(x, CAST(1.5 AS FLOAT))` has scale 1 and so does
//! `round(x, 0.5 + 0.5)`. Foldable here means what Spark's `foldable` means
//! for the expressions this folder evaluates: literals (dates included);
//! casts of foldable values to a number, a string or a date; `+ - * / % div`
//! and unary `-`; comparisons, `AND`/`OR`/`NOT`, `IS [NOT] NULL`;
//! `CASE`/`if`/`coalesce`; `concat`; the deterministic string, math and
//! arithmetic functions (`abs`, `length`, `upper`, ...) of foldable
//! arguments; and the day and field datetime functions (`date_add`, `year`,
//! `extract`, ...) of foldable date arguments. Anything else (a column, a
//! nondeterministic call, a datetime function over a timestamp, which reads
//! the session time zone, or the clock and session functions, which read
//! the query start or the user) is not a constant.
//!
//! A folded date has no [`Constant`]: it is not a value the typing rules
//! read, so it stays unfolded to them (not an error), while the calls that
//! read a folded date (`year`, `date_add`, a `LIMIT` over one) still
//! evaluate. Datetime functions over timestamps stay unfolded too: the
//! folder has no session time zone, so a timestamp result could differ from
//! the executor's.
//!
//! Evaluation runs through the same kernels the executor uses (so overflow,
//! division by zero, malformed input and ANSI/legacy behaviour are the
//! kernels'). A call that fails folds to [`Constant::Error`]: the function
//! that reads the value raises it, any other ignores it, so an unselected
//! branch never raises. `CASE`, `if` and `coalesce` evaluate lazily, as
//! Spark's do.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type,
};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, StringArray, new_null_array,
};
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::{ErrorClass, InternalError, KrishivError};
use krishiv_common::stack::{STEP, on_big_stack};
use krishiv_functions::scalar::{Arg, BindArg, BindOptions, bind_scalar, is_argument_pure};
use krishiv_functions::signature::FunctionKind;
use krishiv_kernels::cast::{CastMode as KernelMode, CastOptions, FixedOffset, cast};
use krishiv_kernels::{KernelCtx, SparkVersion};
use krishiv_types::DataType;

use super::fold_ops;
use super::{Constant, SparkProfile};
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
    /// The kernels' context: the session's Spark line, whose error classes
    /// and messages the kernels raise.
    pub kernel: KernelCtx,
}

impl Conf {
    /// The settings of a session on `profile`.
    pub(super) fn new(ansi: bool, decimal_loss: bool, profile: SparkProfile) -> Self {
        // The kernels model 4.0 as their oldest line; 3.5 behaves as it.
        let version = match profile {
            SparkProfile::V3_5 | SparkProfile::V4_0 => SparkVersion::V4_0,
            SparkProfile::V4_1 => SparkVersion::V4_1,
            _ => SparkVersion::V4_2,
        };
        Self {
            ansi,
            decimal_loss,
            kernel: KernelCtx::new().with_spark_version(version),
        }
    }
}

/// One folded subexpression: `None` when it is not foldable, else its
/// one-row value or the error evaluating it raises.
type Folded = Option<Result<ArrayRef, KrishivError>>;

/// What a folded node's value depends on besides the node: its address
/// and the settings it was folded under.
type Key = (usize, bool, bool, SparkVersion);

/// The values of the expressions folded so far in one resolver pass, by
/// node address. Folding a call folds its arguments, and `Resolver::expr`
/// shares subtrees, so without a memo `e = f(e', e')` folds `e'` twice at
/// every level (exponential in the depth), and each enclosing call folds
/// its arguments again. Each entry keeps its node alive, so an address is
/// never reused for another node while the memo lives.
#[derive(Debug, Default)]
pub(super) struct Memo {
    map: HashMap<Key, (Arc<Expr>, Folded)>,
    depth: usize,
}

/// Whether `e` is a plain literal (what `first(x, true)` requires of its
/// flag, where Spark matches `Literal` and does not fold).
pub(super) const fn is_literal(e: &Expr) -> bool {
    matches!(e.node, ExprNode::Literal(_))
}

/// The value of `e` as a constant, when it is foldable. A folded date (or
/// timestamp, which never folds here) has no [`Constant`]: it is not a
/// value the typing rules read, so it is not a constant rather than an
/// error, while the calls evaluated from it still fold.
pub(super) fn constant(e: &Arc<Expr>, conf: Conf, memo: &mut Memo) -> Option<Constant> {
    match &e.node {
        ExprNode::Literal(Literal::Null(_)) => return Some(Constant::Null),
        ExprNode::Literal(Literal::Boolean(v)) => return Some(Constant::Boolean(*v)),
        ExprNode::Literal(Literal::String(v)) => return Some(Constant::String(v.clone())),
        _ => {}
    }
    let folded = fold(e, conf, memo)?;
    let array = match folded {
        Err(e) => return Some(Constant::Error(e)),
        Ok(a) => a,
    };
    if array.null_count() > 0 || array.data_type() == &ArrowType::Null {
        return Some(Constant::Null);
    }
    if !matches!(
        array.data_type(),
        ArrowType::Null
            | ArrowType::Boolean
            | ArrowType::Int8
            | ArrowType::Int16
            | ArrowType::Int32
            | ArrowType::Int64
            | ArrowType::Utf8
            | ArrowType::Float32
            | ArrowType::Float64
            | ArrowType::Decimal128(..)
    ) {
        return None;
    }
    Some(match value(&array, conf.ansi) {
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

/// `e` folded to doubles, for a quantile percentage or an array of
/// them: `None` when `e` is not foldable here, else its values (`None`
/// for a NULL) or the error evaluating it raises. Strings are never
/// parsed (the caller folds the cast to `DOUBLE`, whose kernels raise
/// `CAST_INVALID_INPUT` for them); any other non-numeric type is an
/// internal error (the caller type-checks first).
pub(super) fn doubles(
    e: &Arc<Expr>,
    conf: Conf,
    memo: &mut Memo,
) -> Option<Result<Vec<Option<f64>>, KrishivError>> {
    let array = match fold(e, conf, memo)? {
        Err(e) => return Some(Err(e)),
        Ok(a) => a,
    };
    Some(doubles_of(&array))
}

/// The values of the one-row array `a` as doubles (`None` for a NULL),
/// descending into a one-row list for an array of percentages.
fn doubles_of(a: &ArrayRef) -> Result<Vec<Option<f64>>, KrishivError> {
    match a.data_type() {
        ArrowType::List(_) | ArrowType::LargeList(_) | ArrowType::FixedSizeList(..) => {
            let list = arrow_array::cast::as_list_array(a);
            if list.null_count() > 0 || list.len() != 1 {
                return scalar_double(a).map(|v| vec![v]);
            }
            let values = list.value(0);
            (0..values.len())
                .map(|i| scalar_double(&values.slice(i, 1)))
                .collect::<Result<Vec<_>, _>>()
                .map(|v| v.into_iter().collect())
        }
        _ => scalar_double(a).map(|v| vec![v]),
    }
}

/// The one-row array `a` as a double (`None` for a NULL).
fn scalar_double(a: &ArrayRef) -> Result<Option<f64>, KrishivError> {
    if a.is_empty() || a.null_count() > 0 || a.data_type() == &ArrowType::Null {
        return Ok(None);
    }
    Ok(Some(match a.data_type() {
        ArrowType::Int8 => f64::from(a.as_primitive::<Int8Type>().value(0)),
        ArrowType::Int16 => f64::from(a.as_primitive::<Int16Type>().value(0)),
        ArrowType::Int32 => f64::from(a.as_primitive::<Int32Type>().value(0)),
        ArrowType::Int64 => a.as_primitive::<Int64Type>().value(0) as f64,
        ArrowType::Float32 => f64::from(a.as_primitive::<Float32Type>().value(0)),
        ArrowType::Float64 => a.as_primitive::<Float64Type>().value(0),
        ArrowType::Decimal128(_, s) => {
            decimal_to_double(a.as_primitive::<Decimal128Type>().value(0), *s)?
        }
        ArrowType::Decimal64(_, s) => {
            decimal_to_double(i128::from(a.as_primitive::<Decimal64Type>().value(0)), *s)?
        }
        _ => {
            return Err(InternalError {
                message: format!("a quantile percentage of type {}", a.data_type()),
            }
            .into());
        }
    }))
}

/// The unscaled decimal `unscaled` at `scale` as the nearest `f64`, as
/// Java's `BigDecimal.doubleValue` gives it (correctly rounded, through
/// the decimal text).
fn decimal_to_double(unscaled: i128, scale: i8) -> Result<f64, KrishivError> {
    let digits = unscaled.unsigned_abs().to_string();
    let scale = usize::try_from(scale).unwrap_or(0);
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (int, frac) = padded.split_at(padded.len() - scale);
    let text = format!("{}{int}.{frac}", if unscaled < 0 { "-" } else { "" });
    text.parse::<f64>().map_err(|_| {
        InternalError {
            message: format!("a quantile percentage of {text}"),
        }
        .into()
    })
}

/// `e` as a one-row array: `None` when it is not foldable here. Every
/// `STEP` levels run on a fresh stack (see `Resolver::rel`).
fn fold(e: &Arc<Expr>, conf: Conf, memo: &mut Memo) -> Folded {
    memo.depth += 1;
    let r = if memo.depth.is_multiple_of(STEP) {
        on_big_stack(|| fold_on_stack(e, conf, memo)).unwrap_or(None)
    } else {
        fold_on_stack(e, conf, memo)
    };
    memo.depth -= 1;
    r
}

fn fold_on_stack(e: &Arc<Expr>, conf: Conf, memo: &mut Memo) -> Folded {
    if let ExprNode::Literal(l) = &e.node {
        return literal(l).map(Ok);
    }
    let key = (
        Arc::as_ptr(e) as usize,
        conf.ansi,
        conf.decimal_loss,
        conf.kernel.spark_version(),
    );
    if let Some((_, hit)) = memo.map.get(&key) {
        return hit.clone();
    }
    let folded = match &e.node {
        ExprNode::Cast { expr, to, mode: m } => fold_cast(expr, to, *m, conf, memo),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => call(f, conf, memo),
        _ => None,
    };
    memo.map.insert(key, (Arc::clone(e), folded.clone()));
    folded
}

fn fold_cast(expr: &Arc<Expr>, to: &DataType, m: CastMode, conf: Conf, memo: &mut Memo) -> Folded {
    let inner = match fold(expr, conf, memo)? {
        // The executor reads strings through their `Utf8` carrier too.
        Ok(a) => normalized(a),
        Err(e) => return Some(Err(e)),
    };
    // A NULL of any type casts to a NULL of the target type.
    if inner.null_count() > 0 || inner.data_type() == &ArrowType::Null {
        return Some(Ok(new_null_array(&to.to_arrow(), 1)));
    }
    // A cast to the value's own type is the value (the kernels below do
    // not cover every target, e.g. arrays of literals).
    if inner.data_type() == &to.to_arrow() {
        return Some(Ok(inner));
    }
    // The kernels' targets: numbers, strings and dates (no timestamp: it
    // reads the session time zone, so a cast to one stays unfolded).
    if !matches!(
        to,
        DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Double
            | DataType::Decimal(_)
            | DataType::String
            | DataType::Date
    ) {
        return None;
    }
    let kernel_mode = match m {
        CastMode::Try => KernelMode::Try,
        CastMode::Legacy => KernelMode::Legacy,
        CastMode::Ansi => KernelMode::Ansi,
        _ => mode(conf.ansi),
    };
    // The kernels' string carrier is `Utf8`: what the executor maps
    // `Utf8View` to around the same call. The result stays `Utf8`, which
    // the value reader and the function bindings both take as a string.
    let target = to.to_arrow();
    let kernel_target = match &target {
        ArrowType::Utf8View => ArrowType::Utf8,
        _ => target,
    };
    match cast(
        inner.as_ref(),
        &kernel_target,
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
fn kids(
    f: &ResolvedFunction,
    conf: Conf,
    memo: &mut Memo,
) -> Option<Vec<Result<ArrayRef, KrishivError>>> {
    // Stops at the first argument that is not foldable (a column), so a
    // deep non-constant tree costs one descent, not an evaluation.
    f.args.iter().map(|a| fold(a, conf, memo)).collect()
}

fn call(f: &ResolvedFunction, conf: Conf, memo: &mut Memo) -> Folded {
    if !f.deterministic || f.distinct || f.filter.is_some() || f.kind != FunctionKind::Scalar {
        return None;
    }
    let kids = kids(f, conf, memo)?;
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
        ) => fold_ops::arithmetic(op, l, r, &out, conf.ansi, &conf.kernel.baseline()),
        (Builtin::IntegralDivide, [l, r]) => {
            fold_ops::integral_divide(l, r, conf.ansi, &conf.kernel.baseline())
        }
        (Builtin::Negative, [a]) => fold_ops::negate(a, conf.ansi, &conf.kernel.baseline()),
        (Builtin::Eq | Builtin::Lt | Builtin::LtEq | Builtin::Gt | Builtin::GtEq, [l, r]) => {
            if fold_ops::is_null(l) || fold_ops::is_null(r) {
                return Some(Ok(fold_ops::bool_array(None)));
            }
            fold_ops::comparison(op, l, r, &conf.kernel.baseline())
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

/// The datetime functions the folder evaluates: the day arithmetic, the
/// field functions and `extract`/`date_part`. Their results over dates
/// never read the session time zone; over timestamps they do, so those
/// stay unfolded (checked at the call). The clock and session functions
/// (`current_date`, `current_timestamp`, `localtimestamp`, `current_user`,
/// ...) read the query start or the user and never fold here: the folder
/// has neither, so Spark's foldable call stays unfolded for the executor.
fn is_foldable_datetime(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "date_add"
            | "dateadd"
            | "date_sub"
            | "datediff"
            | "date_diff"
            | "year"
            | "month"
            | "day"
            | "dayofmonth"
            | "extract"
            | "date_part"
    )
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
    if !is_argument_pure(name) && !is_foldable_datetime(name) {
        return None;
    }
    let args = match values(kids) {
        Ok(a) => a,
        Err(e) => return Some(Err(e)),
    };
    // The folder has no session time zone, so a datetime function over a
    // timestamp does not fold here (dates never read the zone). It stays
    // unfolded, for the executor to evaluate once.
    if is_foldable_datetime(name)
        && args
            .iter()
            .any(|a| matches!(a.data_type(), ArrowType::Timestamp(..)))
    {
        return None;
    }
    let opts = BindOptions::new()
        .with_kernel(conf.kernel)
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
        Literal::Date(v) => Arc::new(Date32Array::from(vec![*v])),
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
            &Arc::new(e.clone()),
            Conf::new(ansi, true, SparkProfile::V4_2),
            &mut Memo::default(),
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

    /// Casts to `STRING` and `DATE` fold through the cast kernels: a number
    /// or a date has a string, a well-formed string has a date. A folded
    /// date is not a value the typing rules read, but the calls reading it
    /// still fold.
    #[test]
    fn casts_to_string_and_date_fold() {
        assert_eq!(
            constant(&cast(Literal::Date(0), DataType::String), true),
            string("1970-01-01")
        );
        assert_eq!(
            constant(
                &cast(Literal::String("2024-01-01".to_owned()), DataType::Date),
                true
            ),
            None
        );
        assert_eq!(
            constant(
                &named(
                    "year",
                    vec![cast(
                        Literal::String("2024-01-01".to_owned()),
                        DataType::Date
                    )],
                    DataType::Integer
                ),
                true
            ),
            Some(Constant::Integral(2024))
        );
        assert_eq!(
            constant(
                &named(
                    "length",
                    vec![cast(Literal::Integer(12), DataType::String)],
                    DataType::Integer
                ),
                true
            ),
            Some(Constant::Integral(2))
        );
        // Malformed input fails as the cast kernels fail it.
        assert_eq!(
            class(constant(
                &cast(Literal::String("x".to_owned()), DataType::Date),
                true
            )),
            Some(ErrorClass::CastInvalidInput)
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

    /// `e_n = e_(n-1) + e_(n-1)` over one shared node each: 2^n evaluations
    /// without a memo, one per node with it, and a second fold of the same
    /// expression (an enclosing call's) evaluates nothing again.
    #[test]
    fn a_shared_subtree_folds_once() {
        let long = || lit(Literal::Long(1));
        let mut e = Arc::new(op(Builtin::Add, vec![long(), long()], DataType::Long));
        let depth = 200;
        for _ in 0..depth {
            e = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    id: FunctionId::Builtin(Builtin::Add),
                    kind: FunctionKind::Scalar,
                    args: vec![Arc::clone(&e), Arc::clone(&e)],
                    distinct: false,
                    filter: None,
                    data_type: DataType::Long,
                    nullable: false,
                    deterministic: true,
                },
            ))));
        }
        let conf = Conf::new(false, true, SparkProfile::V4_2);
        let mut memo = Memo::default();
        // 2 * 2^200 wraps to 0 in legacy mode: the value is the kernels'.
        let first = super::constant(&e, conf, &mut memo);
        assert!(first.is_some());
        assert_eq!(memo.map.len(), depth + 1, "one entry per distinct node");
        assert_eq!(super::constant(&e, conf, &mut memo), first);
        assert_eq!(memo.map.len(), depth + 1);
        // Another setting folds under its own entries.
        let ansi = Conf::new(true, true, SparkProfile::V4_2);
        let _ = super::constant(&e, ansi, &mut memo);
        assert_eq!(memo.map.len(), 2 * (depth + 1));
    }

    /// The kernels raise the session's Spark line's error: `%` by zero is
    /// `DIVIDE_BY_ZERO` before 4.1 and `REMAINDER_BY_ZERO` after it.
    #[test]
    fn folding_follows_the_session_spark_line() {
        let rem = Arc::new(op(
            Builtin::Remainder,
            vec![int(1), int(0)],
            DataType::Integer,
        ));
        let class = |profile| match super::constant(
            &rem,
            Conf::new(true, true, profile),
            &mut Memo::default(),
        ) {
            Some(Constant::Error(e)) => e.error_class(),
            other => panic!("{other:?}"),
        };
        assert_eq!(class(SparkProfile::V4_0), ErrorClass::DivideByZero);
        assert_eq!(class(SparkProfile::V3_5), ErrorClass::DivideByZero);
        assert_eq!(class(SparkProfile::V4_1), ErrorClass::RemainderByZero);
        assert_eq!(class(SparkProfile::V4_2), ErrorClass::RemainderByZero);
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
        // Casts to a string fold through the cast kernels, so `concat` of
        // mixed literals folds too.
        assert_eq!(
            constant(&cast(Literal::Integer(12), DataType::String), true),
            string("12")
        );
        let mixed = |parts: Vec<Expr>| op(Builtin::Concat, parts, DataType::String);
        assert_eq!(
            constant(
                &named(
                    "length",
                    vec![mixed(vec![
                        text("a"),
                        cast(Literal::Integer(1), DataType::String)
                    ])],
                    DataType::Integer
                ),
                true
            ),
            Some(Constant::Integral(2))
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
        // The day and field datetime functions fold over dates; a folded
        // date has no constant (the typing rules do not read dates), so
        // `date_add` of literals is still not a constant, while `year` of
        // one is. Clock, session and unknown functions never fold.
        let date = lit(Literal::Date(0));
        assert_eq!(
            constant(&named("date_add", vec![date, int(1)], DataType::Date), true),
            None
        );
        assert_eq!(
            constant(
                &named("year", vec![lit(Literal::Date(0))], DataType::Integer),
                true
            ),
            Some(Constant::Integral(1970))
        );
        assert_eq!(
            constant(
                &named(
                    "datediff",
                    vec![lit(Literal::Date(1)), lit(Literal::Date(0))],
                    DataType::Integer
                ),
                true
            ),
            Some(Constant::Integral(1))
        );
        // Over timestamps the same functions read the session time zone,
        // which the folder has not: they stay unfolded.
        assert_eq!(
            constant(
                &named(
                    "year",
                    vec![cast(
                        Literal::String("2024-01-01".to_owned()),
                        DataType::Timestamp
                    )],
                    DataType::Integer
                ),
                true
            ),
            None
        );
        for clock in ["current_date", "current_timestamp", "localtimestamp"] {
            assert_eq!(
                constant(&named(clock, vec![], DataType::Date), true),
                None,
                "{clock}"
            );
        }
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

    /// A foldable expression near [`crate::MAX_DEPTH`] folds on a 2 MiB
    /// thread: every `STEP` levels hop to a fresh stack, so without the hop
    /// this overflows the thread's stack.
    #[test]
    fn deep_foldable_expression_survives_small_stack() {
        let mut e = int(0);
        for _ in 0..crate::MAX_DEPTH - 1 {
            e = op(Builtin::Add, vec![e, int(1)], DataType::Integer);
        }
        let e = Arc::new(e);
        std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(move || {
                let conf = Conf::new(false, true, SparkProfile::V4_2);
                let mut memo = Memo::default();
                assert_eq!(
                    super::constant(&e, conf, &mut memo),
                    Some(Constant::Integral(crate::MAX_DEPTH as i64 - 1))
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
