// SPDX-License-Identifier: Apache-2.0
//! `IN (list)`, `CASE WHEN`, `coalesce` (lowered to `CASE WHEN` so untaken
//! branches are never evaluated) and the single-evaluation conditionals
//! `nullif` / `nullifzero` (`NullIf`) and `nanvl` (`Nanvl`). Every other
//! conditional (`if`, `nvl`, `nvl2`, `ifnull`, `zeroifnull`) is rewritten to
//! these in the analyzer, which owns all type coercion: the nodes here take
//! the analyzer's casts and derive no types.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, new_null_array};
use arrow_buffer::BooleanBuffer;
use arrow_schema::DataType as ArrowType;
use arrow_select::filter::filter_record_batch;
use arrow_select::interleave::interleave;
use arrow_select::zip::zip;
use krishiv_kernels::KernelCtx;
use krishiv_kernels::compare::CompareKind;
use krishiv_plan::{Builtin, CastMode, ResolvedFunction};

use super::cast::{CastParts, apply, canonical_carrier, parts};
use super::logic::{and_or, cmp, compare, is_null};
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

pub(super) fn when(
    mut args: Vec<Arc<dyn PhysExpr>>,
    out: ArrowType,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let mut otherwise = if args.len() % 2 == 1 {
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
    for v in branches.iter_mut().map(|(_, v)| v).chain(&mut otherwise) {
        coerce_branch(v, &out, env)?;
    }
    Ok(Arc::new(When {
        branches,
        otherwise,
        out,
    }))
}

/// `v` as a value of the `CASE` result type `out`: branches of one Spark
/// `STRING`/`BINARY` in different Arrow encodings meet in the canonical
/// carrier; an untyped `NULL` stays one. Anything else is an engine bug:
/// the analyzer has cast every branch to the common type.
fn coerce_branch(v: &mut Arc<dyn PhysExpr>, out: &ArrowType, env: &EvalEnv) -> Result<()> {
    let t = v.data_type().clone();
    if &t == out || t == ArrowType::Null {
        return Ok(());
    }
    if canonical_carrier(&t) == Some(out.clone()) {
        *v = super::cast::coerce_to(Arc::clone(v), out, env)?;
        return Ok(());
    }
    Err(internal(&format!(
        "a CASE value of type {t} in a CASE of type {out}"
    )))
}

impl PhysExpr for When {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.otherwise.as_ref().is_none_or(|e| e.nullable())
            || self.branches.iter().any(|(_, v)| v.nullable())
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
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
            let c = cond.eval(&rest, partition)?.into_array(rest.num_rows())?;
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
                value
                    .eval(&chosen, partition)?
                    .into_array(chosen.num_rows())?,
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
                Some(e) => cast_null(
                    e.eval(&rest, partition)?.into_array(rest.num_rows())?,
                    &self.out,
                ),
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

/// Binds `coalesce` to `CASE WHEN` machinery, so an untaken branch is never
/// evaluated: `coalesce(1, 1 / 0)` answers `1` without raising, as Spark
/// does. The analyzer has cast every argument to their common type.
///
/// `f` is the analyzed call (for the result type), `args` its compiled
/// arguments. An argument count the analyzer never produces is an engine bug
/// (`INTERNAL_ERROR`).
pub(super) fn coalesce(
    f: &ResolvedFunction,
    args: Vec<Arc<dyn PhysExpr>>,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let out = f.data_type.to_arrow();
    // The first non-null argument: each condition sees only the rows no
    // earlier argument took, each value only the rows it takes.
    let mut acc: Option<Arc<dyn PhysExpr>> = None;
    for a in args.into_iter().rev() {
        acc = Some(match acc {
            None => a,
            Some(taken) => {
                let cond = is_null(false, Arc::clone(&a));
                when(vec![cond, a, taken], out.clone(), env)?
            }
        });
    }
    acc.ok_or_else(|| internal("coalesce without arguments"))
}

/// `nullif(a, b)` / `nullifzero(x)`: NULL where `a = b`, else `a`, with `a`
/// evaluated once. The analyzer has cast `b` to the type `=` compares at
/// (`other`'s type); operands of one Spark `STRING`/`BINARY` in different
/// encodings meet in the canonical carrier instead. The casts are for the
/// comparison only; the result keeps `a`'s values at the result's carrier.
#[derive(Debug)]
pub(super) struct NullIf {
    value: Arc<dyn PhysExpr>,
    other: Arc<dyn PhysExpr>,
    /// Each side to the comparison type, for the comparison only; `None`
    /// when already there (no copy).
    value_cast: Option<CastParts>,
    other_cast: Option<CastParts>,
    /// `value` to the result type: the analyzer types the result at the
    /// value's Spark type, whose canonical carrier may differ from the
    /// value's own (a `Utf8` column in a `Utf8View` result). `None` when
    /// already there; a different Spark type is left for `zip` to reject
    /// as before.
    result_cast: Option<CastParts>,
    kernels: KernelCtx,
    out: ArrowType,
}

/// Binds `nullif` / `nullifzero`: `value` at its own type, `other` at the
/// type `=` compares at. `f` is the analyzed call (for the result type),
/// `args` its compiled arguments (`value`, then `other`). Anything but two
/// arguments is an engine bug: the analyzer derives the comparison type.
pub(super) fn null_if(
    f: &ResolvedFunction,
    args: Vec<Arc<dyn PhysExpr>>,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    let [value, other] = <[Arc<dyn PhysExpr>; 2]>::try_from(args)
        .map_err(|_| internal("nullif without two arguments"))?;
    let mode = if input.env.ansi {
        CastMode::Ansi
    } else {
        CastMode::Legacy
    };
    // Same-typed comparisons need no cast: `apply` would copy the column.
    let (value_ty, other_ty) = (value.data_type().clone(), other.data_type().clone());
    let out = f.data_type.to_arrow();
    let (value_cast, other_cast) =
        match (canonical_carrier(&value_ty), canonical_carrier(&other_ty)) {
            // One Spark `STRING`/`BINARY` in different encodings: compare in
            // the canonical carrier, keeping the side already there.
            (Some(a), Some(b)) if a == b && value_ty != other_ty => {
                let at = |t: &ArrowType| {
                    if t == &a {
                        Ok(None)
                    } else {
                        parts(t, a.clone(), mode, input.env).map(Some)
                    }
                };
                (at(&value_ty)?, at(&other_ty)?)
            }
            _ if value_ty == other_ty => (None, None),
            _ => (
                Some(parts(&value_ty, other_ty.clone(), mode, input.env)?),
                None,
            ),
        };
    // The result keeps the value's Spark type at the result's carrier.
    let result_cast = if value_ty == out {
        None
    } else if canonical_carrier(&value_ty) == Some(out.clone()) {
        Some(parts(&value_ty, out.clone(), mode, input.env)?)
    } else {
        None
    };
    Ok(Arc::new(NullIf {
        value,
        other,
        value_cast,
        other_cast,
        result_cast,
        kernels: input.env.kernels,
        out,
    }))
}

impl PhysExpr for NullIf {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        true
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        // `value` is evaluated once: the comparison casts its array, the
        // result keeps its values.
        let a = self.value.eval(batch, partition)?;
        let b = self.other.eval(batch, partition)?;
        let scalar = a.is_scalar() && b.is_scalar();
        let n = if scalar { 1 } else { rows };
        let a_full = a.into_array(n)?;
        let b_full = b.into_array(n)?;
        // An untyped NULL never equals (the kernels cannot cast a `Null`
        // array either); otherwise each side meets the comparison type.
        let null_side =
            a_full.data_type() == &ArrowType::Null || b_full.data_type() == &ArrowType::Null;
        let ab = if null_side {
            Arc::clone(&a_full)
        } else if let Some(cast) = &self.value_cast {
            apply(cast, &a_full)?
        } else {
            Arc::clone(&a_full)
        };
        let bb = if null_side {
            Arc::clone(&b_full)
        } else if let Some(cast) = &self.other_cast {
            apply(cast, &b_full)?
        } else {
            Arc::clone(&b_full)
        };
        let eq = if null_side {
            BooleanArray::new_null(n)
        } else {
            cmp(
                CompareKind::Equal,
                &Value::Array(ab),
                &Value::Array(bb),
                &self.kernels,
            )?
        };
        // NULL is not equal: keep `value` where the comparison is not true.
        let valid = eq
            .nulls()
            .map(|ns| ns.inner().clone())
            .unwrap_or_else(|| BooleanBuffer::new_set(n));
        let mask = BooleanArray::new(eq.values() & &valid, None);
        let kept = match &self.result_cast {
            Some(cast) => apply(cast, &a_full)?,
            None => Arc::clone(&a_full),
        };
        let out = zip(&mask, &new_null_array(&self.out, n), &kept)
            .map_err(|e| internal(&e.to_string()))?;
        Ok(if scalar {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}

/// `nanvl(a, b)`: `b` where `a` is NaN, else `a`, with `a` evaluated once and
/// `b` only where `a` is NaN. Both sides are already the analyzer's
/// `float`/`double`.
#[derive(Debug)]
struct Nanvl {
    a: Arc<dyn PhysExpr>,
    b: Arc<dyn PhysExpr>,
    out: ArrowType,
}

/// Binds `nanvl` over two same-typed `float`/`double` children. Anything
/// else is an engine bug: the analyzer casts both sides.
pub(super) fn nan_vl(
    f: &ResolvedFunction,
    args: Vec<Arc<dyn PhysExpr>>,
    _input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    let [a, b] = <[Arc<dyn PhysExpr>; 2]>::try_from(args)
        .map_err(|_| internal("nanvl without two arguments"))?;
    let t = a.data_type();
    if t != b.data_type() || !matches!(t, ArrowType::Float32 | ArrowType::Float64) {
        return Err(internal("nanvl of non-float arguments"));
    }
    Ok(Arc::new(Nanvl {
        a,
        b,
        out: f.data_type.to_arrow(),
    }))
}

impl PhysExpr for Nanvl {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.a.nullable() || self.b.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        // `a` is evaluated once; NULL is not NaN.
        let a = self.a.eval(batch, partition)?.into_array(rows)?;
        let nan = match a.data_type() {
            ArrowType::Float32 => BooleanBuffer::from_iter(
                a.as_primitive::<Float32Type>()
                    .iter()
                    .map(|v| v.is_some_and(f32::is_nan)),
            ),
            ArrowType::Float64 => BooleanBuffer::from_iter(
                a.as_primitive::<Float64Type>()
                    .iter()
                    .map(|v| v.is_some_and(f64::is_nan)),
            ),
            t => return Err(internal(&format!("nanvl of {t}"))),
        };
        if nan.count_set_bits() == 0 {
            return Ok(Value::Array(a));
        }
        // `b` sees only the NaN rows, so `nanvl(1, 1 / 0)` never raises.
        let b = super::eval_where(self.b.as_ref(), batch, &nan, partition)?;
        let out =
            zip(&BooleanArray::new(nan, None), &b, &a).map_err(|e| internal(&e.to_string()))?;
        Ok(Value::Array(out))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::types::{Decimal128Type, Float32Type, Float64Type, Int32Type, Int64Type};
    use arrow_array::{
        Decimal128Array, Float64Array, Int32Array, Int64Array, NullArray, StringViewArray,
    };
    use arrow_schema::{Field, Schema};
    use krishiv_common::config::Config;
    use krishiv_common::error::ErrorClass;
    use krishiv_plan::{AttrId, FunctionId, FunctionKind, Literal};
    use krishiv_types::{DataType, DecimalType};

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
        e.eval(batch, 0)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap()
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

    /// `coalesce` answers the first non-null argument of each row.
    #[test]
    fn coalesce_takes_the_first_non_null_argument_per_row() {
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
        let f = resolved("coalesce", DataType::Integer);
        let env = env(true);
        let e = coalesce(
            &f,
            vec![
                column(0, ArrowType::Int32, true),
                column(1, ArrowType::Int32, true),
            ],
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(1), Some(3), None, None]),
        );
        // A single argument is itself.
        let one = coalesce(&f, vec![column(0, ArrowType::Int32, true)], &env).unwrap();
        assert_eq!(
            values(&one, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(1), None, None, None]),
        );
        // No arguments is an engine bug: the analyzer always has some.
        assert!(coalesce(&f, Vec::new(), &env).is_err());
    }

    /// `coalesce(1, 1 / 0)` answers without raising; `coalesce(1 / 0, 1)`
    /// raises with ANSI on and answers `1` without.
    #[test]
    fn coalesce_is_lazy_in_both_modes() {
        for ansi in [true, false] {
            let env = env(ansi);
            let batch = rows_batch(2);
            let f = resolved("coalesce", DataType::Double);
            let e = coalesce(
                &f,
                vec![lit(&Literal::Double(1.0)), one_over_zero(&env)],
                &env,
            )
            .unwrap();
            assert_eq!(
                values(&e, &batch).as_primitive::<Float64Type>(),
                &Float64Array::from(vec![Some(1.0), Some(1.0)]),
                "ansi = {ansi}"
            );
            let e = coalesce(
                &f,
                vec![one_over_zero(&env), lit(&Literal::Double(1.0))],
                &env,
            )
            .unwrap();
            if ansi {
                let err = e.eval(&batch, 0).unwrap_err();
                assert_eq!(err.error_class(), ErrorClass::DivideByZero);
            } else {
                assert_eq!(
                    values(&e, &batch).as_primitive::<Float64Type>(),
                    &Float64Array::from(vec![Some(1.0), Some(1.0)]),
                );
            }
        }
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
        let e = null_if(
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
        // Two arguments are required: the analyzer always passes both.
        assert!(
            null_if(
                &f,
                vec![lit(&Literal::Integer(1))],
                input(&env, &layout, &schema),
            )
            .is_err()
        );
    }

    /// `nullif(1, 2.1)`: the `INT` value meets the `DOUBLE` other at
    /// `DOUBLE` for the comparison, but the result keeps the `INT` value.
    #[test]
    fn nullif_compares_at_the_other_type_and_keeps_its_own() {
        let env = env(true);
        let batch = rows_batch(1);
        let schema = Arc::new(Schema::empty());
        let f = resolved("nullif", DataType::Integer);
        let e = null_if(
            &f,
            vec![lit(&Literal::Integer(1)), lit(&Literal::Double(2.1))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(1)]),
        );
        let e = null_if(
            &f,
            vec![lit(&Literal::Integer(1)), lit(&Literal::Double(1.0))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![None]),
        );
    }

    /// `nullif(NULL, 1)` is `NULL` and `nullif(1, NULL)` is `1`: an untyped
    /// `NULL` side never equals, so the value is kept.
    #[test]
    fn nullif_with_an_untyped_null_side() {
        let env = env(true);
        let batch = rows_batch(1);
        let schema = Arc::new(Schema::empty());
        let f = resolved("nullif", DataType::Null);
        let e = null_if(
            &f,
            vec![lit(&Literal::Null(None)), lit(&Literal::Integer(1))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(values(&e, &batch).data_type(), &ArrowType::Null);
        let g = resolved("nullif", DataType::Integer);
        let e = null_if(
            &g,
            vec![lit(&Literal::Integer(1)), lit(&Literal::Null(None))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(1)]),
        );
    }

    /// `nullif` casts its value for the comparison in the session's mode:
    /// `nullifzero('abc')` raises an ANSI cast error and answers the
    /// argument without.
    #[test]
    fn nullif_casts_for_the_comparison_only() {
        for (ansi, want) in [(true, None), (false, Some("abc"))] {
            let env = env(ansi);
            let batch = rows_batch(1);
            let schema = Arc::new(Schema::empty());
            let f = resolved("nullif", DataType::String);
            let e = null_if(
                &f,
                vec![
                    lit(&Literal::String("abc".to_owned())),
                    lit(&Literal::Long(0)),
                ],
                input(&env, &[], &schema),
            )
            .unwrap();
            match want {
                None => {
                    let err = e.eval(&batch, 0).unwrap_err();
                    assert_eq!(err.error_class(), ErrorClass::CastInvalidInput);
                    assert_eq!(err.sql_state(), Some("22018"));
                }
                Some(w) => assert_eq!(
                    values(&e, &batch)
                        .as_string_view()
                        .iter()
                        .collect::<Vec<_>>(),
                    [Some(w)],
                ),
            }
        }
        // A `NULL` value casts to nothing: the comparison is NULL.
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
            "nullif",
            DataType::Decimal(DecimalType::try_new(1, 1).unwrap()),
        );
        let e = null_if(
            &f,
            vec![
                column(0, ArrowType::Decimal128(1, 1), true),
                lit(&Literal::Integer(0)),
            ],
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
    }

    /// An expression counting its evaluations: `nullif` evaluates its first
    /// argument once per batch, for the comparison and the value alike.
    #[derive(Debug)]
    struct Count {
        inner: Arc<dyn PhysExpr>,
        n: Arc<AtomicUsize>,
    }

    impl PhysExpr for Count {
        fn data_type(&self) -> &ArrowType {
            self.inner.data_type()
        }

        fn nullable(&self) -> bool {
            self.inner.nullable()
        }

        fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
            self.n.fetch_add(1, Ordering::SeqCst);
            self.inner.eval(batch, partition)
        }
    }

    /// The value is computed once even though it is both compared and
    /// returned.
    #[test]
    fn nullif_evaluates_its_first_argument_once() {
        let env = env(true);
        let batch = rows_batch(3);
        let schema = Arc::new(Schema::empty());
        let n = Arc::new(AtomicUsize::new(0));
        let value: Arc<dyn PhysExpr> = Arc::new(Count {
            inner: lit(&Literal::Long(1)),
            n: Arc::clone(&n),
        });
        let f = resolved("nullif", DataType::Long);
        let e = null_if(
            &f,
            vec![value, lit(&Literal::Long(2))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int64Type>(),
            &Int64Array::from(vec![Some(1), Some(1), Some(1)]),
        );
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    /// `nanvl(a, b)`: `b` where `a` is NaN, else `a`; NULL is not NaN.
    #[test]
    fn nanvl_takes_the_second_argument_for_nan_only() {
        let env = env(true);
        let batch = rows_batch(4);
        let schema = Arc::new(Schema::empty());
        for (ty, a, b) in [
            (
                DataType::Float,
                Literal::Float(f32::NAN),
                Literal::Float(1.0),
            ),
            (
                DataType::Double,
                Literal::Double(f64::NAN),
                Literal::Double(1.0),
            ),
        ] {
            let f = resolved("nanvl", ty);
            let e = nan_vl(&f, vec![lit(&a), lit(&b)], input(&env, &[], &schema)).unwrap();
            assert_eq!(values(&e, &batch).len(), 4);
        }
        let f = resolved("nanvl", DataType::Double);
        for (a, want) in [
            (f64::NAN, 1.0),
            (2.5, 2.5),
            (-0.0, -0.0),
            (f64::INFINITY, f64::INFINITY),
        ] {
            let e = nan_vl(
                &f,
                vec![lit(&Literal::Double(a)), lit(&Literal::Double(1.0))],
                input(&env, &[], &schema),
            )
            .unwrap();
            let got = values(&e, &batch).as_primitive::<Float64Type>().value(0);
            assert_eq!(got.to_bits(), want.to_bits(), "{a} -> {got}, want {want}");
        }
        let g = resolved("nanvl", DataType::Float);
        for (a, want) in [
            (f32::NAN, 1.0),
            (2.5, 2.5),
            (-0.0, -0.0),
            (f32::INFINITY, f32::INFINITY),
        ] {
            let e = nan_vl(
                &g,
                vec![lit(&Literal::Float(a)), lit(&Literal::Float(1.0))],
                input(&env, &[], &schema),
            )
            .unwrap();
            let got = values(&e, &batch).as_primitive::<Float32Type>().value(0);
            assert_eq!(got.to_bits(), want.to_bits(), "{a} -> {got}, want {want}");
        }
        // Anything but two same-typed floats is an engine bug.
        assert!(
            nan_vl(
                &f,
                vec![lit(&Literal::Double(1.0)), lit(&Literal::Float(1.0)),],
                input(&env, &[], &schema),
            )
            .is_err()
        );
        assert!(
            nan_vl(
                &f,
                vec![lit(&Literal::Double(1.0))],
                input(&env, &[], &schema),
            )
            .is_err()
        );
    }

    /// `nanvl` evaluates `b` only where `a` is NaN: `nanvl(1, 1 / 0)`
    /// answers `1` without raising.
    #[test]
    fn nanvl_is_lazy_in_its_second_argument() {
        for ansi in [true, false] {
            let env = env(ansi);
            let batch = rows_batch(2);
            let schema = Arc::new(Schema::empty());
            let f = resolved("nanvl", DataType::Double);
            let e = nan_vl(
                &f,
                vec![lit(&Literal::Double(1.0)), one_over_zero(&env)],
                input(&env, &[], &schema),
            )
            .unwrap();
            assert_eq!(
                values(&e, &batch).as_primitive::<Float64Type>(),
                &Float64Array::from(vec![Some(1.0), Some(1.0)]),
                "ansi = {ansi}"
            );
            let e = nan_vl(
                &f,
                vec![lit(&Literal::Double(f64::NAN)), one_over_zero(&env)],
                input(&env, &[], &schema),
            )
            .unwrap();
            if ansi {
                let err = e.eval(&batch, 0).unwrap_err();
                assert_eq!(err.error_class(), ErrorClass::DivideByZero);
            } else {
                assert!(values(&e, &batch).as_primitive::<Float64Type>().is_null(0));
            }
        }
    }

    /// `coalesce` of one Spark `STRING` in `Utf8` and `Utf8View` carriers:
    /// every branch meets the `CASE` type, keeping its values.
    #[test]
    fn coalesce_unifies_string_encodings() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("u", ArrowType::Utf8, true),
            Field::new("v", ArrowType::Utf8View, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow_array::StringArray::from(vec![None, Some("b")])) as ArrayRef,
                Arc::new(StringViewArray::from(vec![Some("v"), Some("w")])),
            ],
        )
        .unwrap();
        let f = resolved("coalesce", DataType::String);
        for (args, want) in [
            (
                vec![
                    column(0, ArrowType::Utf8, true),
                    column(1, ArrowType::Utf8View, true),
                ],
                [Some("v"), Some("b")],
            ),
            (
                vec![
                    column(1, ArrowType::Utf8View, true),
                    column(0, ArrowType::Utf8, true),
                ],
                [Some("v"), Some("w")],
            ),
        ] {
            let e = coalesce(&f, args, &env).unwrap();
            assert_eq!(e.data_type(), &ArrowType::Utf8View);
            assert_eq!(
                values(&e, &batch)
                    .as_string_view()
                    .iter()
                    .collect::<Vec<_>>(),
                want
            );
        }
    }

    /// `CASE WHEN` with `Utf8` branches in a `STRING` (`Utf8View`) case:
    /// the branches are cast; an `INT` branch still fails.
    #[test]
    fn when_unifies_string_encodings() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("u", ArrowType::Utf8, true),
            Field::new("i", ArrowType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow_array::StringArray::from(vec![Some("a"), None])) as ArrayRef,
                Arc::new(Int32Array::from(vec![Some(1), Some(2)])),
            ],
        )
        .unwrap();
        let e = when(
            vec![
                lit(&Literal::Boolean(true)),
                column(0, ArrowType::Utf8, true),
                lit(&Literal::String("d".to_owned())),
            ],
            ArrowType::Utf8View,
            &env,
        )
        .unwrap();
        assert_eq!(e.data_type(), &ArrowType::Utf8View);
        assert_eq!(
            values(&e, &batch)
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            [Some("a"), None]
        );
        // A branch of another Spark type still fails as before.
        let err = when(
            vec![
                lit(&Literal::Boolean(true)),
                column(1, ArrowType::Int32, true),
            ],
            ArrowType::Utf8View,
            &env,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("a CASE value of type Int32 in a CASE of type Utf8View"),
            "{err}"
        );
    }

    /// `nullif` of `Utf8` and `Utf8View` operands compares in the view and
    /// returns the `STRING` carrier with the value's contents.
    #[test]
    fn nullif_unifies_string_encodings() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("u", ArrowType::Utf8, true),
            Field::new("v", ArrowType::Utf8View, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow_array::StringArray::from(vec![
                    Some("x"),
                    Some("y"),
                    None,
                ])) as ArrayRef,
                Arc::new(StringViewArray::from(vec![Some("x"), Some("z"), Some("q")])),
            ],
        )
        .unwrap();
        let layout = [AttrId(0), AttrId(1)];
        let input = input(&env, &layout, &schema);
        // A `Utf8` value against a `Utf8View` literal: the result is the
        // `STRING` carrier with the value's contents.
        let f = resolved("nullif", DataType::String);
        let e = null_if(
            &f,
            vec![
                column(0, ArrowType::Utf8, true),
                lit(&Literal::String("x".to_owned())),
            ],
            input,
        )
        .unwrap();
        assert_eq!(e.data_type(), &ArrowType::Utf8View);
        assert_eq!(
            values(&e, &batch)
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            [None, Some("y"), None]
        );
        // A `Utf8View` value against a `Utf8` other: the comparison stays in
        // the view, the result keeps the view.
        let e = null_if(
            &f,
            vec![
                column(1, ArrowType::Utf8View, true),
                column(0, ArrowType::Utf8, true),
            ],
            input,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch)
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            [None, Some("z"), Some("q")]
        );
    }
}
