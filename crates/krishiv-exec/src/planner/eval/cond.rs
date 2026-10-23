// SPDX-License-Identifier: Apache-2.0
//! `IN (list)`, `CASE WHEN` and the lazy conditional functions (`coalesce`,
//! `if`, `nvl`, `nvl2`, `nullif`, `ifnull`, `zeroifnull`, `nullifzero`), which
//! lower to `CASE WHEN` so untaken branches are never evaluated.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, new_null_array};
use arrow_schema::DataType as ArrowType;
use arrow_select::filter::filter_record_batch;
use arrow_select::interleave::interleave;
use krishiv_plan::{Builtin, CastMode, Literal, ResolvedFunction};
use krishiv_types::{DataType, DecimalType};

use super::cast;
use super::logic::{and_or, compare, is_null};
use super::{EvalEnv, Input, PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::internal;

/// `v IN (a, b, ...)` as `v = a OR v = b OR ...`: true on a match, NULL when
/// `v` is NULL or nothing matched and an element is NULL, else false.
pub(super) fn in_list(
    mut args: Vec<Arc<dyn PhysExpr>>,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let list = args.split_off(1);
    let v = args.pop().ok_or_else(|| internal("IN arity"))?;
    let mut out: Option<Arc<dyn PhysExpr>> = None;
    for item in list {
        let eq = compare(Builtin::Eq, Arc::clone(&v), item, env)?;
        out = Some(match out {
            None => eq,
            Some(prev) => and_or(false, prev, eq),
        });
    }
    out.ok_or_else(|| internal("IN without a list"))
}

/// `CASE WHEN c1 THEN v1 ... [ELSE e] END`, lazily: each condition sees only
/// the rows no earlier branch took, each value only the rows it takes.
#[derive(Debug)]
struct When {
    branches: Vec<(Arc<dyn PhysExpr>, Arc<dyn PhysExpr>)>,
    otherwise: Option<Arc<dyn PhysExpr>>,
    out: ArrowType,
}

pub(super) fn when(mut args: Vec<Arc<dyn PhysExpr>>, out: ArrowType) -> Result<Arc<dyn PhysExpr>> {
    let otherwise = if args.len() % 2 == 1 {
        args.pop()
    } else {
        None
    };
    let mut branches = Vec::with_capacity(args.len() / 2);
    let mut it = args.into_iter();
    while let (Some(c), Some(v)) = (it.next(), it.next()) {
        if c.data_type() != &ArrowType::Boolean {
            return Err(internal("a CASE condition is not boolean"));
        }
        branches.push((c, v));
    }
    for v in branches.iter().map(|(_, v)| v).chain(&otherwise) {
        if v.data_type() != &out && v.data_type() != &ArrowType::Null {
            return Err(internal(&format!(
                "a CASE value of type {} in a CASE of type {out}",
                v.data_type()
            )));
        }
    }
    Ok(Arc::new(When {
        branches,
        otherwise,
        out,
    }))
}

impl PhysExpr for When {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.otherwise.as_ref().is_none_or(|e| e.nullable())
            || self.branches.iter().any(|(_, v)| v.nullable())
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let rows = batch.num_rows();
        // `rest`: the rows no branch took yet, as a batch and their row
        // numbers in `batch`.
        let mut rest = batch.clone();
        let mut rest_rows: Vec<usize> = (0..rows).collect();
        let mut parts: Vec<ArrayRef> = Vec::new();
        // Per output row: (part, row within the part).
        let mut pick = vec![(0_usize, 0_usize); rows];
        for (cond, value) in &self.branches {
            if rest_rows.is_empty() {
                break;
            }
            let c = cond.eval(&rest)?.into_array(rest.num_rows())?;
            let c = c
                .as_boolean_opt()
                .ok_or_else(|| internal("CASE condition"))?;
            // NULL is not true.
            let taken = match c.nulls() {
                Some(n) => c.values() & n.inner(),
                None => c.values().clone(),
            };
            if taken.count_set_bits() == 0 {
                continue;
            }
            let mask = BooleanArray::new(taken.clone(), None);
            let chosen = filter_record_batch(&rest, &mask).map_err(|e| internal(&e.to_string()))?;
            let v = cast_null(
                value.eval(&chosen)?.into_array(chosen.num_rows())?,
                &self.out,
            );
            let part = parts.len();
            parts.push(v);
            let mut k = 0;
            for (i, r) in rest_rows.iter().enumerate() {
                if taken.value(i) {
                    pick[*r] = (part, k);
                    k += 1;
                }
            }
            let left = BooleanArray::new(!&taken, None);
            rest = filter_record_batch(&rest, &left).map_err(|e| internal(&e.to_string()))?;
            rest_rows = rest_rows
                .iter()
                .zip(taken.iter())
                .filter_map(|(r, t)| (!t).then_some(*r))
                .collect();
        }
        if !rest_rows.is_empty() {
            let v = match &self.otherwise {
                Some(e) => cast_null(e.eval(&rest)?.into_array(rest.num_rows())?, &self.out),
                None => new_null_array(&self.out, rest_rows.len()),
            };
            let part = parts.len();
            parts.push(v);
            for (k, r) in rest_rows.iter().enumerate() {
                pick[*r] = (part, k);
            }
        }
        if parts.len() == 1 {
            return Ok(Value::Array(parts.swap_remove(0)));
        }
        let refs: Vec<&dyn Array> = parts.iter().map(AsRef::as_ref).collect();
        if refs.is_empty() {
            return Ok(Value::Array(new_null_array(&self.out, 0)));
        }
        interleave(&refs, &pick)
            .map(Value::Array)
            .map_err(|e| internal(&e.to_string()))
    }
}

/// An untyped NULL value as a NULL of `out`.
fn cast_null(a: ArrayRef, out: &ArrowType) -> ArrayRef {
    if a.data_type() == &ArrowType::Null {
        new_null_array(out, a.len())
    } else {
        a
    }
}

/// Whether `name` is a lazy conditional function, bound here rather than by
/// the registry's scalar bindings: `coalesce`, `if`, `nvl`, `nvl2`,
/// `nullif`, `ifnull`, `zeroifnull` and `nullifzero` (the registry's
/// lower-case names, as [`krishiv_plan::FunctionId::Registry`] carries
/// them).
pub(super) fn is_conditional(name: &str) -> bool {
    matches!(
        name,
        "coalesce" | "if" | "nvl" | "nvl2" | "nullif" | "ifnull" | "zeroifnull" | "nullifzero"
    )
}

/// Binds a lazy conditional function call to `CASE WHEN` machinery, so an
/// untaken branch is never evaluated: `coalesce(1, 1 / 0)` answers `1`
/// without raising, as Spark does. The analyzer has already cast every
/// argument but [`Inputs::Comparable`](krishiv_functions::signature::Inputs::Comparable)
/// ones (`nullif`, `nullifzero`) to their common type; `nullif` compares
/// its uncast arguments as `=` would and returns its first argument's
/// value.
///
/// `f` is the analyzed call (for the result type), `args` its compiled
/// arguments. Names outside [`is_conditional`] and argument counts the
/// analyzer never produces are engine bugs (`INTERNAL_ERROR`).
pub(super) fn conditional(
    name: &str,
    f: &ResolvedFunction,
    args: Vec<Arc<dyn PhysExpr>>,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    let out = f.data_type.to_arrow();
    match name {
        // `coalesce(a, ...)` (`ifnull`, `nvl`): the first non-null argument.
        // Each condition sees only the rows no earlier argument took, each
        // value only the rows it takes.
        "coalesce" | "ifnull" | "nvl" => {
            let mut acc: Option<Arc<dyn PhysExpr>> = None;
            for a in args.into_iter().rev() {
                acc = Some(match acc {
                    None => a,
                    Some(taken) => {
                        let cond = is_null(false, Arc::clone(&a));
                        when(vec![cond, a, taken], out.clone())?
                    }
                });
            }
            acc.ok_or_else(|| internal("coalesce without arguments"))
        }
        // `if(c, t, e)`.
        "if" => {
            let [c, t, e] = <[Arc<dyn PhysExpr>; 3]>::try_from(args)
                .map_err(|_| internal("if without three arguments"))?;
            when(vec![c, t, e], out)
        }
        // `nvl2(x, t, e)`: `t` where `x` is not null, else `e`.
        "nvl2" => {
            let [x, t, e] = <[Arc<dyn PhysExpr>; 3]>::try_from(args)
                .map_err(|_| internal("nvl2 without three arguments"))?;
            when(vec![is_null(false, x), t, e], out)
        }
        // `nullif(a, b)`: `NULL` where `a = b`, else `a`.
        "nullif" => {
            let [a, b] = <[Arc<dyn PhysExpr>; 2]>::try_from(args)
                .map_err(|_| internal("nullif without two arguments"))?;
            let cond = equal(Arc::clone(&a), b, input)?;
            when(vec![cond, null()?, a], out)
        }
        // `nullifzero(x)`: `NULL` where `x = 0`, else `x`. The hidden `0`
        // the analyzer checked comparability against is an `INT`, as in
        // [`Inputs::Comparable`](krishiv_functions::signature::Inputs::Comparable).
        "nullifzero" => {
            let [a] = <[Arc<dyn PhysExpr>; 1]>::try_from(args)
                .map_err(|_| internal("nullifzero without one argument"))?;
            let cond = equal(Arc::clone(&a), int_zero()?, input)?;
            when(vec![cond, null()?, a], out)
        }
        // `zeroifnull(x)`: `0` (of the result type the analyzer met `x`
        // with) where `x` is null, else `x`.
        "zeroifnull" => {
            let [x] = <[Arc<dyn PhysExpr>; 1]>::try_from(args)
                .map_err(|_| internal("zeroifnull without one argument"))?;
            when(vec![is_null(false, Arc::clone(&x)), x, zero(&out)?], out)
        }
        _ => Err(internal(&format!("{name} is not a conditional function"))),
    }
}

/// An untyped `NULL` literal.
fn null() -> Result<Arc<dyn PhysExpr>> {
    Ok(Arc::new(super::Lit::new(&Literal::Null(None))?))
}

/// The literal `0` (`INT`) the analyzer checks `nullifzero` against.
fn int_zero() -> Result<Arc<dyn PhysExpr>> {
    Ok(Arc::new(super::Lit::new(&Literal::Integer(0))?))
}

/// The literal `0` of `out`, the result type `zeroifnull` met its argument
/// with: `0` of a numeric type, `'0'` where the common type is `STRING`
/// (legacy mode).
fn zero(out: &ArrowType) -> Result<Arc<dyn PhysExpr>> {
    let spark = DataType::from_arrow(out).map_err(|_| internal(&format!("zeroifnull of {out}")))?;
    let literal = match &spark {
        DataType::Byte => Literal::Byte(0),
        DataType::Short => Literal::Short(0),
        DataType::Integer => Literal::Integer(0),
        DataType::Long => Literal::Long(0),
        DataType::Float => Literal::Float(0.0),
        DataType::Double => Literal::Double(0.0),
        DataType::Decimal(d) => Literal::Decimal {
            unscaled: 0,
            data_type: *d,
        },
        DataType::String => Literal::String("0".to_owned()),
        _ => return Err(internal(&format!("zeroifnull of {out}"))),
    };
    Ok(Arc::new(super::Lit::new(&literal)?))
}

/// `l = r`, where `l` and `r` may have different types: the analyzer checks
/// [`Inputs::Comparable`](krishiv_functions::signature::Inputs::Comparable)
/// arguments are comparable with `=` but does not cast them, so both sides
/// are cast here to the type `=` would compare them at, in the session's
/// cast mode. Wrong or malformed input casts raise as casts do
/// (`nullifzero('abc')` is `CAST_INVALID_INPUT` with ANSI on, `NULL`
/// without).
fn equal(
    l: Arc<dyn PhysExpr>,
    r: Arc<dyn PhysExpr>,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    let (lt, rt) = (l.data_type().clone(), r.data_type().clone());
    if lt == rt || matches!((&lt, &rt), (ArrowType::Null, _) | (_, ArrowType::Null)) {
        return compare(Builtin::Eq, l, r, input.env);
    }
    let common = comparison_type(&lt, &rt, input.env.ansi)
        .ok_or_else(|| internal(&format!("comparison of {lt} with {rt}")))?;
    let mode = if input.env.ansi {
        CastMode::Ansi
    } else {
        CastMode::Legacy
    };
    let target = common.to_arrow();
    let l = if l.data_type() == &target {
        l
    } else {
        cast::compile(l, &common, mode, input)?
    };
    let r = if r.data_type() == &target {
        r
    } else {
        cast::compile(r, &common, mode, input)?
    };
    compare(Builtin::Eq, l, r, input.env)
}

/// The type `=` casts `l` and `r` to before comparing them: the analyzer's
/// `common` for binary comparisons, over the physical Arrow types. Integral
/// arguments always meet at their full width (the analyzer sizes an
/// integral *literal* by its digits against a decimal; equality of exact
/// values is the same at any width that fits both sides).
///
/// Without ANSI a boolean compares with a number as booleans: the analyzer
/// accepts the comparison but casts nothing, and a string meets a boolean
/// at the boolean, so the number casts the same way.
fn comparison_type(l: &ArrowType, r: &ArrowType, ansi: bool) -> Option<DataType> {
    let l = DataType::from_arrow(l).ok()?;
    let r = DataType::from_arrow(r).ok()?;
    if l == r {
        return Some(l);
    }
    Some(match (&l, &r) {
        (DataType::Decimal(a), DataType::Decimal(b)) => DataType::Decimal(wider_decimal(*a, *b)),
        (DataType::Decimal(a), n) | (n, DataType::Decimal(a)) if is_integral(n) => {
            DataType::Decimal(wider_decimal(*a, as_decimal(n)))
        }
        (a, b) if is_integral(a) && is_integral(b) => {
            if integral_digits(a) >= integral_digits(b) {
                a.clone()
            } else {
                b.clone()
            }
        }
        (DataType::Float, n) | (n, DataType::Float) if !ansi && is_integral(n) => DataType::Float,
        (a, b) if is_numeric(a) && is_numeric(b) => DataType::Double,
        (DataType::String, n) | (n, DataType::String) if is_integral(n) => DataType::Long,
        (DataType::String, n) | (n, DataType::String) if is_numeric(n) => DataType::Double,
        (DataType::String, t) | (t, DataType::String)
            if matches!(
                t,
                DataType::Boolean
                    | DataType::Date
                    | DataType::Timestamp
                    | DataType::TimestampNtz
                    | DataType::Binary
            ) =>
        {
            t.clone()
        }
        (DataType::Boolean, n) | (n, DataType::Boolean) if !ansi && is_numeric(n) => {
            DataType::Boolean
        }
        (DataType::Date, t @ (DataType::Timestamp | DataType::TimestampNtz))
        | (t @ (DataType::Timestamp | DataType::TimestampNtz), DataType::Date) => t.clone(),
        (DataType::Timestamp, DataType::TimestampNtz)
        | (DataType::TimestampNtz, DataType::Timestamp) => DataType::Timestamp,
        _ => return None,
    })
}

/// The digits of an integral type, as the analyzer counts them.
fn integral_digits(t: &DataType) -> Option<u8> {
    match t {
        DataType::Byte => Some(3),
        DataType::Short => Some(5),
        DataType::Integer => Some(10),
        DataType::Long => Some(20),
        _ => None,
    }
}

/// Whether `t` is an integral type.
fn is_integral(t: &DataType) -> bool {
    integral_digits(t).is_some()
}

/// Whether `t` is a numeric type.
fn is_numeric(t: &DataType) -> bool {
    is_integral(t) || matches!(t, DataType::Float | DataType::Double | DataType::Decimal(_))
}

/// The decimal an integral argument meets a decimal as: itself at its full
/// width. Callers only pass integrals (a decimal meets another decimal
/// with no conversion); anything else meets at the widest decimal.
fn as_decimal(t: &DataType) -> DecimalType {
    decimal(
        u32::from(integral_digits(t).unwrap_or(DecimalType::MAX_PRECISION)),
        0,
    )
}

/// `DecimalPrecision.widerDecimalType`: the larger integer range and scale;
/// past 38 digits the fraction gives way and the range stays.
fn wider_decimal(a: DecimalType, b: DecimalType) -> DecimalType {
    let scale = u32::from(a.scale().max(b.scale()));
    let range = u32::from((a.precision() - a.scale()).max(b.precision() - b.scale()));
    let max = u32::from(DecimalType::MAX_PRECISION);
    if range + scale <= max {
        decimal(range + scale, scale)
    } else {
        decimal(max, max.saturating_sub(range))
    }
}

/// `decimal(precision, scale)` within Spark's bounds.
fn decimal(precision: u32, scale: u32) -> DecimalType {
    let precision = precision.min(u32::from(DecimalType::MAX_PRECISION));
    let scale = scale.min(precision);
    DecimalType::try_new(precision.max(1), scale).unwrap_or(DecimalType::SYSTEM_DEFAULT)
}

#[cfg(test)]
mod tests {
    use arrow_array::types::{Decimal128Type, Float64Type, Int32Type, Int64Type};
    use arrow_array::{
        BooleanArray, Decimal128Array, Float64Array, Int32Array, Int64Array, NullArray,
        StringViewArray,
    };
    use arrow_schema::{Field, Schema};
    use krishiv_common::config::Config;
    use krishiv_common::error::ErrorClass;
    use krishiv_plan::{AttrId, FunctionId, FunctionKind};
    use krishiv_types::DataType;

    use super::*;

    /// [`EvalEnv`] with `spark.sql.ansi.enabled = ansi`.
    fn env(ansi: bool) -> EvalEnv {
        let mut config = Config::new();
        config
            .set(
                "spark.sql.ansi.enabled",
                if ansi { "true" } else { "false" },
            )
            .unwrap();
        EvalEnv::new(&config, Arc::from("test"))
    }

    /// [`Input`] over `schema` with `layout` columns.
    fn input<'a>(
        env: &'a EvalEnv,
        layout: &'a [AttrId],
        schema: &'a std::sync::Arc<Schema>,
    ) -> Input<'a> {
        Input {
            layout,
            schema,
            env,
            subqueries: &[],
        }
    }

    /// An analyzed call of `name` with result type `data_type`.
    fn resolved(name: &str, data_type: DataType) -> ResolvedFunction {
        ResolvedFunction {
            id: FunctionId::Registry {
                name: name.to_owned(),
                id: 0,
            },
            kind: FunctionKind::Scalar,
            args: Vec::new(),
            distinct: false,
            filter: None,
            data_type,
            nullable: true,
            deterministic: true,
        }
    }

    /// A literal expression.
    fn lit(l: &Literal) -> Arc<dyn PhysExpr> {
        Arc::new(super::super::Lit::new(l).unwrap())
    }

    /// An input-column expression.
    fn column(index: usize, data_type: ArrowType, nullable: bool) -> Arc<dyn PhysExpr> {
        Arc::new(super::super::Column {
            index,
            data_type,
            nullable,
        })
    }

    /// A `rows`-row batch with no usable columns, for literal-only plans.
    fn rows_batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("d", ArrowType::Null, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(NullArray::new(rows))]).unwrap()
    }

    /// Evaluates `e` over `batch` as a column of `batch.num_rows()` rows.
    fn values(e: &Arc<dyn PhysExpr>, batch: &RecordBatch) -> ArrayRef {
        e.eval(batch).unwrap().into_array(batch.num_rows()).unwrap()
    }

    /// `1 / 0`: `NULL` without ANSI, `DIVIDE_BY_ZERO` with it.
    fn one_over_zero(env: &EvalEnv) -> Arc<dyn PhysExpr> {
        super::super::arith::binary(
            Builtin::Divide,
            lit(&Literal::Double(1.0)),
            lit(&Literal::Double(0.0)),
            ArrowType::Float64,
            true,
            env,
        )
        .unwrap()
    }

    /// `e` cast to `to`, as the analyzer's implicit casts do.
    fn casted(
        e: Arc<dyn PhysExpr>,
        to: &DataType,
        ansi: bool,
        input: Input<'_>,
    ) -> Arc<dyn PhysExpr> {
        super::super::cast::compile(
            e,
            to,
            if ansi {
                CastMode::Ansi
            } else {
                CastMode::Legacy
            },
            input,
        )
        .unwrap()
    }

    /// The conditionals lower here, and nothing else does.
    #[test]
    fn only_the_conditionals_lower_here() {
        for name in [
            "coalesce",
            "if",
            "nvl",
            "nvl2",
            "nullif",
            "ifnull",
            "zeroifnull",
            "nullifzero",
        ] {
            assert!(is_conditional(name), "{name}");
        }
        for name in ["upper", "when", "nanvl", "COALESCE", ""] {
            assert!(!is_conditional(name), "{name}");
        }
        let env = env(true);
        let schema = Arc::new(Schema::empty());
        let input = input(&env, &[], &schema);
        let f = resolved("upper", DataType::String);
        assert!(conditional("upper", &f, Vec::new(), input).is_err());
    }

    /// Wrong argument counts are engine bugs: the analyzer never produces
    /// them.
    #[test]
    fn bad_arities_are_internal_errors() {
        let env = env(true);
        let schema = Arc::new(Schema::empty());
        let input = input(&env, &[], &schema);
        let one = || lit(&Literal::Integer(1));
        assert!(
            conditional(
                "coalesce",
                &resolved("coalesce", DataType::Integer),
                vec![],
                input
            )
            .is_err()
        );
        assert!(
            conditional(
                "if",
                &resolved("if", DataType::Integer),
                vec![one(), one()],
                input
            )
            .is_err()
        );
        assert!(
            conditional(
                "nullif",
                &resolved("nullif", DataType::Integer),
                vec![one()],
                input
            )
            .is_err()
        );
        assert!(
            conditional(
                "nullifzero",
                &resolved("nullifzero", DataType::Integer),
                vec![],
                input
            )
            .is_err()
        );
        assert!(
            conditional(
                "zeroifnull",
                &resolved("zeroifnull", DataType::Integer),
                vec![one(), one()],
                input
            )
            .is_err()
        );
    }

    /// `coalesce` answers the first non-null argument of each row.
    #[test]
    fn coalesce_takes_the_first_non_null_argument_per_row() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", ArrowType::Int32, true),
            Field::new("b", ArrowType::Int32, true),
            Field::new("c", ArrowType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, None, None])),
                Arc::new(Int32Array::from(vec![Some(2), Some(3), None, None])),
                Arc::new(Int32Array::from(vec![Some(4), Some(5), Some(6), None])),
            ],
        )
        .unwrap();
        let layout = [AttrId(0), AttrId(1), AttrId(2)];
        let f = resolved("coalesce", DataType::Integer);
        for (name, want) in [
            (
                "coalesce",
                Int32Array::from(vec![Some(1), Some(3), Some(6), None]),
            ),
            (
                "ifnull",
                Int32Array::from(vec![Some(1), Some(3), Some(6), None]),
            ),
            (
                "nvl",
                Int32Array::from(vec![Some(1), Some(3), Some(6), None]),
            ),
        ] {
            let e = conditional(
                name,
                &f,
                vec![
                    column(0, ArrowType::Int32, true),
                    column(1, ArrowType::Int32, true),
                ],
                input(&env, &layout, &schema),
            )
            .unwrap();
            assert_eq!(
                values(&e, &batch).as_primitive::<Int32Type>(),
                &Int32Array::from(vec![Some(1), Some(3), None, None]),
                "{name} of two"
            );
            // A single argument is itself.
            let one = conditional(
                name,
                &f,
                vec![column(0, ArrowType::Int32, true)],
                input(&env, &layout, &schema),
            )
            .unwrap();
            assert_eq!(
                values(&one, &batch).as_primitive::<Int32Type>(),
                &Int32Array::from(vec![Some(1), None, None, None]),
                "{name} of one"
            );
            let three = conditional(
                name,
                &f,
                vec![
                    column(0, ArrowType::Int32, true),
                    column(1, ArrowType::Int32, true),
                    column(2, ArrowType::Int32, true),
                ],
                input(&env, &layout, &schema),
            )
            .unwrap();
            assert_eq!(
                values(&three, &batch).as_primitive::<Int32Type>(),
                &want,
                "{name} of three"
            );
        }
    }

    /// `coalesce(1, 1 / 0)` answers without raising; `coalesce(1 / 0, 1)`
    /// raises with ANSI on and answers `1` without.
    #[test]
    fn coalesce_is_lazy_in_both_modes() {
        for ansi in [true, false] {
            let env = env(ansi);
            let batch = rows_batch(2);
            let schema = Arc::new(Schema::empty());
            let f = resolved("coalesce", DataType::Double);
            let one = || {
                casted(
                    lit(&Literal::Integer(1)),
                    &DataType::Double,
                    ansi,
                    input(&env, &[], &schema),
                )
            };
            let e = conditional(
                "coalesce",
                &f,
                vec![one(), one_over_zero(&env)],
                input(&env, &[], &schema),
            )
            .unwrap();
            assert_eq!(
                values(&e, &batch).as_primitive::<Float64Type>(),
                &Float64Array::from(vec![Some(1.0), Some(1.0)]),
                "ansi = {ansi}"
            );
            let e = conditional(
                "coalesce",
                &f,
                vec![one_over_zero(&env), one()],
                input(&env, &[], &schema),
            )
            .unwrap();
            if ansi {
                let err = e.eval(&batch).unwrap_err();
                assert_eq!(err.error_class(), ErrorClass::DivideByZero);
            } else {
                assert_eq!(
                    values(&e, &batch).as_primitive::<Float64Type>(),
                    &Float64Array::from(vec![Some(1.0), Some(1.0)]),
                );
            }
        }
    }

    /// `if` takes its branches lazily; a null condition takes the `ELSE`.
    #[test]
    fn if_takes_branches_lazily() {
        for ansi in [true, false] {
            let env = env(ansi);
            let batch = rows_batch(1);
            let schema = Arc::new(Schema::empty());
            let f = resolved("if", DataType::Double);
            let one = || {
                casted(
                    lit(&Literal::Integer(1)),
                    &DataType::Double,
                    ansi,
                    input(&env, &[], &schema),
                )
            };
            let two = || {
                casted(
                    lit(&Literal::Integer(2)),
                    &DataType::Double,
                    ansi,
                    input(&env, &[], &schema),
                )
            };
            let e = conditional(
                "if",
                &f,
                vec![lit(&Literal::Boolean(true)), one(), one_over_zero(&env)],
                input(&env, &[], &schema),
            )
            .unwrap();
            assert_eq!(
                values(&e, &batch).as_primitive::<Float64Type>(),
                &Float64Array::from(vec![Some(1.0)]),
            );
            let e = conditional(
                "if",
                &f,
                vec![lit(&Literal::Boolean(false)), one_over_zero(&env), two()],
                input(&env, &[], &schema),
            )
            .unwrap();
            assert_eq!(
                values(&e, &batch).as_primitive::<Float64Type>(),
                &Float64Array::from(vec![Some(2.0)]),
            );
            // A null condition takes the `ELSE`: a null `BOOLEAN` column.
            let nulls = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("c", ArrowType::Boolean, true)])),
                vec![Arc::new(BooleanArray::from(vec![None]))],
            )
            .unwrap();
            let null_layout = [AttrId(0)];
            let null_schema = nulls.schema();
            let e = conditional(
                "if",
                &f,
                vec![column(0, ArrowType::Boolean, true), one(), two()],
                input(&env, &null_layout, &null_schema),
            )
            .unwrap();
            assert_eq!(
                values(&e, &nulls).as_primitive::<Float64Type>(),
                &Float64Array::from(vec![Some(2.0)]),
            );
        }
    }

    /// `nvl2(x, t, e)` answers `t` where `x` is not null, else `e`.
    #[test]
    fn nvl2_tests_for_null() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            ArrowType::Utf8View,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringViewArray::from(vec![Some("x"), None]))],
        )
        .unwrap();
        let layout = [AttrId(0)];
        let f = resolved("nvl2", DataType::String);
        let e = conditional(
            "nvl2",
            &f,
            vec![
                column(0, ArrowType::Utf8View, true),
                lit(&Literal::String("t".to_owned())),
                lit(&Literal::String("e".to_owned())),
            ],
            input(&env, &layout, &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch)
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            [Some("t"), Some("e")]
        );
    }

    /// `nullif` of one type: `NULL` where equal, the value elsewhere.
    #[test]
    fn nullif_same_type() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            ArrowType::Utf8View,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringViewArray::from(vec![
                Some("x"),
                Some("y"),
                None,
            ]))],
        )
        .unwrap();
        let layout = [AttrId(0)];
        let f = resolved("nullif", DataType::String);
        let e = conditional(
            "nullif",
            &f,
            vec![
                column(0, ArrowType::Utf8View, true),
                lit(&Literal::String("x".to_owned())),
            ],
            input(&env, &layout, &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch)
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            [None, Some("y"), None]
        );
    }

    /// `nullif(NULL, 1)` is `NULL` and `nullif(1, NULL)` is `1`: an untyped
    /// `NULL` side compares as `NULL`, so the `ELSE` keeps the first
    /// argument's value.
    #[test]
    fn nullif_with_an_untyped_null_side() {
        let env = env(true);
        let batch = rows_batch(1);
        let schema = Arc::new(Schema::empty());
        let f = resolved("nullif", DataType::Integer);
        let e = conditional(
            "nullif",
            &f,
            vec![lit(&Literal::Null(None)), lit(&Literal::Integer(1))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![None]),
        );
        let e = conditional(
            "nullif",
            &f,
            vec![lit(&Literal::Integer(1)), lit(&Literal::Null(None))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(1)]),
        );
    }

    /// `nullif(1, 2.1)` is `1` and `nullif(1, 1.0)` is `NULL`: the `INT` and
    /// `DOUBLE` sides meet at `DOUBLE` before comparing.
    #[test]
    fn nullif_compares_across_numeric_types() {
        let env = env(true);
        let batch = rows_batch(1);
        let schema = Arc::new(Schema::empty());
        let f = resolved("nullif", DataType::Integer);
        let e = conditional(
            "nullif",
            &f,
            vec![lit(&Literal::Integer(1)), lit(&Literal::Double(2.1))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(1)]),
        );
        let e = conditional(
            "nullif",
            &f,
            vec![lit(&Literal::Integer(1)), lit(&Literal::Double(1.0))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![None]),
        );
        // A side that is already the common type is not cast again: a
        // `BIGINT` meets an `INT` at `BIGINT`.
        let g = resolved("nullif", DataType::Long);
        let e = conditional(
            "nullif",
            &g,
            vec![lit(&Literal::Long(1)), lit(&Literal::Integer(1))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int64Type>(),
            &Int64Array::from(vec![None]),
        );
    }

    /// `nullifzero(0.0)` on a `decimal(1,1)` is `NULL`. The comparison widens
    /// to `decimal(11,1)`: the analyzer meets the integral *literal* `0` by
    /// its digits at `decimal(2,1)`, while the physical comparison widens by
    /// the `INT` range; equality of exact values is the same either way.
    #[test]
    fn nullifzero_compares_decimals_at_the_wider_type() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            ArrowType::Decimal128(1, 1),
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(
                Decimal128Array::from(vec![Some(0)])
                    .with_precision_and_scale(1, 1)
                    .unwrap(),
            )],
        )
        .unwrap();
        let layout = [AttrId(0)];
        let f = resolved(
            "nullifzero",
            DataType::Decimal(DecimalType::try_new(1, 1).unwrap()),
        );
        let e = conditional(
            "nullifzero",
            &f,
            vec![column(0, ArrowType::Decimal128(1, 1), true)],
            input(&env, &layout, &schema),
        )
        .unwrap();
        let got = values(&e, &batch);
        assert_eq!(got.data_type(), &ArrowType::Decimal128(1, 1));
        assert_eq!(
            got.as_primitive::<Decimal128Type>(),
            &Decimal128Array::from(vec![None as Option<i128>])
                .with_precision_and_scale(1, 1)
                .unwrap(),
        );
        assert_eq!(
            comparison_type(&ArrowType::Decimal128(1, 1), &ArrowType::Int32, true),
            Some(DataType::Decimal(DecimalType::try_new(11, 1).unwrap())),
        );
    }

    /// `nullifzero('0')` is `NULL`; `nullifzero('abc')` raises an ANSI cast
    /// error and answers the argument without.
    #[test]
    fn nullifzero_casts_strings_to_bigint() {
        for (text, ansi, want) in [
            ("0", true, None),
            ("abc", false, Some("abc")),
            ("0", false, None),
        ] {
            let env = env(ansi);
            let batch = rows_batch(1);
            let schema = Arc::new(Schema::empty());
            let f = resolved("nullifzero", DataType::String);
            let e = conditional(
                "nullifzero",
                &f,
                vec![lit(&Literal::String(text.to_owned()))],
                input(&env, &[], &schema),
            )
            .unwrap();
            match want {
                None => assert_eq!(
                    values(&e, &batch)
                        .as_string_view()
                        .iter()
                        .collect::<Vec<_>>(),
                    [None],
                    "{text} ansi = {ansi}"
                ),
                Some(w) => assert_eq!(
                    values(&e, &batch)
                        .as_string_view()
                        .iter()
                        .collect::<Vec<_>>(),
                    [Some(w)],
                    "{text} ansi = {ansi}"
                ),
            }
        }
        let env = env(true);
        let batch = rows_batch(1);
        let schema = Arc::new(Schema::empty());
        let f = resolved("nullifzero", DataType::String);
        let e = conditional(
            "nullifzero",
            &f,
            vec![lit(&Literal::String("abc".to_owned()))],
            input(&env, &[], &schema),
        )
        .unwrap();
        let err = e.eval(&batch).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::CastInvalidInput);
        assert_eq!(err.sql_state(), Some("22018"));
    }

    /// `zeroifnull` answers `0` of the result type: an `INT` for an `INT`, a
    /// `decimal(12,2)` zero for a `decimal(10,2)`, `'0'` for a legacy
    /// `STRING`.
    #[test]
    fn zeroifnull_answers_zero_of_the_result_type() {
        let ansi = env(true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", ArrowType::Int32, true),
            Field::new("d", ArrowType::Decimal128(10, 2), true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![None, Some(5)])),
                Arc::new(
                    Decimal128Array::from(vec![None, Some(150)])
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        let layout = [AttrId(0), AttrId(1)];
        let f = resolved("zeroifnull", DataType::Integer);
        let e = conditional(
            "zeroifnull",
            &f,
            vec![column(0, ArrowType::Int32, true)],
            input(&ansi, &layout, &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(0), Some(5)]),
        );
        let g = resolved(
            "zeroifnull",
            DataType::Decimal(DecimalType::try_new(12, 2).unwrap()),
        );
        // The analyzer casts the argument to the common type first.
        let widened = casted(
            column(1, ArrowType::Decimal128(10, 2), true),
            &DataType::Decimal(DecimalType::try_new(12, 2).unwrap()),
            true,
            input(&ansi, &layout, &schema),
        );
        let e = conditional(
            "zeroifnull",
            &g,
            vec![widened],
            input(&ansi, &layout, &schema),
        )
        .unwrap();
        let got = values(&e, &batch);
        assert_eq!(got.data_type(), &ArrowType::Decimal128(12, 2));
        assert_eq!(
            got.as_primitive::<Decimal128Type>(),
            &Decimal128Array::from(vec![Some(0), Some(150)])
                .with_precision_and_scale(12, 2)
                .unwrap(),
        );
        // Legacy strings meet the hidden `0` at `STRING`.
        let legacy = env(false);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            ArrowType::Utf8View,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringViewArray::from(vec![None, Some("a")]))],
        )
        .unwrap();
        let layout = [AttrId(0)];
        let h = resolved("zeroifnull", DataType::String);
        let e = conditional(
            "zeroifnull",
            &h,
            vec![column(0, ArrowType::Utf8View, true)],
            input(&legacy, &layout, &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch)
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            [Some("0"), Some("a")]
        );
    }

    /// `zero` only exists for the types an argument meets the hidden `0` at.
    #[test]
    fn zero_of_an_unmeetable_type_is_an_internal_error() {
        for out in [
            ArrowType::Boolean,
            ArrowType::Null,
            ArrowType::Date32,
            ArrowType::BinaryView,
        ] {
            assert!(zero(&out).is_err(), "{out}");
        }
    }

    /// `zero` builds the `0` of every meetable type.
    #[test]
    fn zero_builds_for_every_meetable_type() {
        use arrow_array::types::{Float32Type, Int8Type, Int16Type};
        let batch = rows_batch(1);
        let cases: Vec<(ArrowType, ArrowType)> = vec![
            (ArrowType::Int8, ArrowType::Int8),
            (ArrowType::Int16, ArrowType::Int16),
            (ArrowType::Int32, ArrowType::Int32),
            (ArrowType::Int64, ArrowType::Int64),
            (ArrowType::Float32, ArrowType::Float32),
            (ArrowType::Float64, ArrowType::Float64),
            (ArrowType::Decimal128(4, 2), ArrowType::Decimal128(4, 2)),
            (ArrowType::Utf8View, ArrowType::Utf8View),
        ];
        for (out, want) in cases {
            let e = zero(&out).unwrap();
            assert_eq!(e.data_type(), &want, "{out}");
            let got = values(&e, &batch);
            assert_eq!(got.data_type(), &want, "{out}");
            assert_eq!(got.len(), 1, "{out}");
            assert!(got.is_valid(0), "{out}");
            match &want {
                ArrowType::Int8 => assert_eq!(got.as_primitive::<Int8Type>().value(0), 0),
                ArrowType::Int16 => assert_eq!(got.as_primitive::<Int16Type>().value(0), 0),
                ArrowType::Int32 => assert_eq!(got.as_primitive::<Int32Type>().value(0), 0),
                ArrowType::Int64 => assert_eq!(got.as_primitive::<Int64Type>().value(0), 0),
                ArrowType::Float32 => assert_eq!(got.as_primitive::<Float32Type>().value(0), 0.0),
                ArrowType::Float64 => assert_eq!(got.as_primitive::<Float64Type>().value(0), 0.0),
                ArrowType::Decimal128(_, _) => {
                    assert_eq!(got.as_primitive::<Decimal128Type>().value(0), 0);
                }
                ArrowType::Utf8View => {
                    assert_eq!(got.as_string_view().value(0), "0");
                }
                _ => unreachable!("unmeetable test type"),
            }
        }
    }

    /// The comparison type is the type `=` would compare at.
    #[test]
    fn comparison_types_match_spark_widening() {
        use ArrowType as A;
        let dec = |p: u8, s: i8| A::Decimal128(p, s);
        let cases: Vec<(A, A, bool, Option<DataType>)> = vec![
            (A::Int32, A::Int32, true, Some(DataType::Integer)),
            (A::Int8, A::Int64, true, Some(DataType::Long)),
            (A::Int16, A::Int32, true, Some(DataType::Integer)),
            (dec(10, 2), dec(5, 1), true, {
                let d = DecimalType::try_new(10, 2).unwrap();
                Some(DataType::Decimal(d))
            }),
            (dec(10, 2), A::Int64, true, {
                let d = DecimalType::try_new(22, 2).unwrap();
                Some(DataType::Decimal(d))
            }),
            (A::Float32, A::Int32, true, Some(DataType::Double)),
            (A::Float32, A::Int32, false, Some(DataType::Float)),
            (A::Float64, A::Int16, true, Some(DataType::Double)),
            (A::Float32, A::Float64, true, Some(DataType::Double)),
            (A::Utf8View, A::Int32, true, Some(DataType::Long)),
            (A::Utf8View, A::Float64, true, Some(DataType::Double)),
            (A::Utf8View, dec(4, 1), true, Some(DataType::Double)),
            (A::Utf8View, A::Boolean, true, Some(DataType::Boolean)),
            (A::Utf8View, A::Date32, true, Some(DataType::Date)),
            (A::Utf8View, A::BinaryView, true, Some(DataType::Binary)),
            (
                A::Utf8View,
                A::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
                true,
                Some(DataType::TimestampNtz),
            ),
            (
                A::Utf8View,
                A::Timestamp(arrow_schema::TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                true,
                Some(DataType::Timestamp),
            ),
            (
                A::Date32,
                A::Timestamp(arrow_schema::TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                true,
                Some(DataType::Timestamp),
            ),
            (
                A::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
                A::Timestamp(arrow_schema::TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                true,
                Some(DataType::Timestamp),
            ),
            (A::Boolean, A::Int32, true, None),
            (A::Boolean, A::Int32, false, Some(DataType::Boolean)),
            (A::Int32, A::Date32, true, None),
            (A::BinaryView, A::Int32, true, None),
            // No earlier arm meets a `STRING` with an untyped `NULL`.
            (A::Utf8View, A::Null, true, None),
        ];
        for (l, r, ansi, want) in cases {
            assert_eq!(comparison_type(&l, &r, ansi), want, "{l} {r} {ansi}");
            // The relation is symmetric.
            assert_eq!(comparison_type(&r, &l, ansi), want, "{r} {l} {ansi}");
        }
        // Past 38 digits the fraction gives way: `decimal(38,0)` meets
        // `decimal(10,5)` at `decimal(38,0)`.
        assert_eq!(
            comparison_type(&dec(38, 0), &dec(10, 5), true),
            Some(DataType::Decimal(DecimalType::try_new(38, 0).unwrap())),
        );
    }
}
