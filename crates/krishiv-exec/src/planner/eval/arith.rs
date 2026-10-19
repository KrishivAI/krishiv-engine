// SPDX-License-Identifier: Apache-2.0
//! Arithmetic: `+ - * / %` and unary `-`, with Spark's overflow policies.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
};
use arrow_array::{ArrayRef, PrimitiveArray, RecordBatch};
use arrow_schema::DataType as ArrowType;
use krishiv_kernels::decimal::{
    DecimalOp, DecimalPlan, decimal_arith, decimal_integral_divide, decimal_unary,
};
use krishiv_kernels::integer::{self, SparkIntegral, UnaryOp};
use krishiv_kernels::{
    Add, AnsiError, ArithOp, JavaWrap, KernelCtx, Mul, NullOnOverflow, Sub, float,
};
use krishiv_plan::Builtin;

use super::logic::prim;
use super::{EvalEnv, PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::{internal, unsupported};

/// `l op r`.
#[derive(Debug)]
struct Binary {
    op: Builtin,
    l: Arc<dyn PhysExpr>,
    r: Arc<dyn PhysExpr>,
    out: ArrowType,
    nullable: bool,
    ansi: bool,
    /// Decimal operands: the kernel plan.
    decimal: Option<DecimalPlan>,
    kernels: KernelCtx,
}

pub(super) fn binary(
    op: Builtin,
    l: Arc<dyn PhysExpr>,
    r: Arc<dyn PhysExpr>,
    out: ArrowType,
    nullable: bool,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let decimal = match (l.data_type(), r.data_type(), &out) {
        (
            ArrowType::Decimal128(lp, ls),
            ArrowType::Decimal128(rp, rs),
            ArrowType::Decimal128(p, s),
        ) => {
            let dop = match op {
                Builtin::Add => DecimalOp::Add,
                Builtin::Subtract => DecimalOp::Subtract,
                Builtin::Multiply => DecimalOp::Multiply,
                Builtin::Divide => DecimalOp::Divide,
                _ => DecimalOp::Remainder,
            };
            Some(DecimalPlan::new(dop, (*lp, *ls), (*rp, *rs), (*p, *s))?)
        }
        (lt, rt, out) if lt == rt && lt == out && is_arith_type(lt) => {
            if op == Builtin::Divide && !matches!(lt, ArrowType::Float32 | ArrowType::Float64) {
                return Err(internal(&format!("`/` on {lt}")));
            }
            None
        }
        (lt, rt, out) => {
            return Err(unsupported(&format!(
                "arithmetic {} on {lt} and {rt} giving {out}",
                op.name()
            )));
        }
    };
    Ok(Arc::new(Binary {
        op,
        l,
        r,
        out,
        nullable,
        ansi: env.ansi,
        decimal,
        kernels: env.kernels,
    }))
}

fn is_arith_type(t: &ArrowType) -> bool {
    matches!(
        t,
        ArrowType::Int8
            | ArrowType::Int16
            | ArrowType::Int32
            | ArrowType::Int64
            | ArrowType::Float32
            | ArrowType::Float64
    )
}

/// Integral `+ - * %`: ANSI raises, legacy wraps (`%` by zero is NULL).
fn integral<T: SparkIntegral>(
    op: Builtin,
    l: &Value,
    r: &Value,
    ansi: bool,
    ctx: &KernelCtx,
) -> Result<ArrayRef> {
    macro_rules! pol {
        ($p:ty) => {
            match op {
                Builtin::Add => integer::binary::<T, Add, $p>(prim(l), prim(r), ctx),
                Builtin::Subtract => integer::binary::<T, Sub, $p>(prim(l), prim(r), ctx),
                Builtin::Multiply => integer::binary::<T, Mul, $p>(prim(l), prim(r), ctx),
                _ => integer::remainder::<T, $p>(prim(l), prim(r), ctx),
            }
        };
    }
    let out = if ansi {
        pol!(AnsiError)
    } else {
        pol!(JavaWrap)
    }?;
    Ok(Arc::new(out))
}

/// Floating-point arithmetic: `/` and `%` by zero raise under ANSI and are
/// NULL in legacy mode.
fn floating<T>(op: Builtin, l: &Value, r: &Value, ansi: bool, ctx: &KernelCtx) -> Result<ArrayRef>
where
    T: arrow_array::ArrowPrimitiveType<Native: krishiv_kernels::float::SparkFloat>,
{
    let (a, b) = (prim::<T>(l), prim::<T>(r));
    let out: PrimitiveArray<T> = match (op, ansi) {
        (Builtin::Add, _) => float::binary(ArithOp::Add, a, b, ctx),
        (Builtin::Subtract, _) => float::binary(ArithOp::Subtract, a, b, ctx),
        (Builtin::Multiply, _) => float::binary(ArithOp::Multiply, a, b, ctx),
        (Builtin::Divide, true) => float::divide::<T, AnsiError>(a, b, ctx),
        (Builtin::Divide, false) => float::divide::<T, JavaWrap>(a, b, ctx),
        (_, true) => float::remainder::<T, AnsiError>(a, b, ctx),
        (_, false) => float::remainder::<T, JavaWrap>(a, b, ctx),
    }?;
    Ok(Arc::new(out))
}

impl PhysExpr for Binary {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let (mut l, r) = (self.l.eval(batch)?, self.r.eval(batch)?);
        let scalar = l.is_scalar() && r.is_scalar();
        if scalar {
            l = Value::Array(Arc::clone(l.array()));
        }
        let (op, ansi, ctx) = (self.op, self.ansi, &self.kernels);
        let out = if let Some(plan) = &self.decimal {
            let (a, b) = (prim::<Decimal128Type>(&l), prim::<Decimal128Type>(&r));
            let out = if ansi {
                decimal_arith::<AnsiError>(a, b, plan, ctx)
            } else {
                decimal_arith::<NullOnOverflow>(a, b, plan, ctx)
            }?;
            Arc::new(out) as ArrayRef
        } else {
            match &self.out {
                ArrowType::Int8 => integral::<Int8Type>(op, &l, &r, ansi, ctx)?,
                ArrowType::Int16 => integral::<Int16Type>(op, &l, &r, ansi, ctx)?,
                ArrowType::Int32 => integral::<Int32Type>(op, &l, &r, ansi, ctx)?,
                ArrowType::Int64 => integral::<Int64Type>(op, &l, &r, ansi, ctx)?,
                ArrowType::Float32 => floating::<Float32Type>(op, &l, &r, ansi, ctx)?,
                _ => floating::<Float64Type>(op, &l, &r, ansi, ctx)?,
            }
        };
        Ok(if scalar {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}

/// Unary `-`.
#[derive(Debug)]
struct Negative {
    e: Arc<dyn PhysExpr>,
    ansi: bool,
    kernels: KernelCtx,
}

pub(super) fn negative(e: Arc<dyn PhysExpr>, env: &EvalEnv) -> Result<Arc<dyn PhysExpr>> {
    let t = e.data_type();
    if !is_arith_type(t) && !matches!(t, ArrowType::Decimal128(..)) {
        return Err(unsupported(&format!("unary minus on {t}")));
    }
    Ok(Arc::new(Negative {
        e,
        ansi: env.ansi,
        kernels: env.kernels,
    }))
}

impl PhysExpr for Negative {
    fn data_type(&self) -> &ArrowType {
        self.e.data_type()
    }

    fn nullable(&self) -> bool {
        self.e.nullable()
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let v = self.e.eval(batch)?;
        let a = v.array();
        let ctx = &self.kernels;
        macro_rules! int {
            ($t:ty) => {{
                let a = a.as_primitive::<$t>();
                Arc::new(if self.ansi {
                    integer::unary::<$t, AnsiError>(UnaryOp::Negate, a, ctx)?
                } else {
                    integer::unary::<$t, JavaWrap>(UnaryOp::Negate, a, ctx)?
                }) as ArrayRef
            }};
        }
        let out: ArrayRef = match a.data_type() {
            ArrowType::Int8 => int!(Int8Type),
            ArrowType::Int16 => int!(Int16Type),
            ArrowType::Int32 => int!(Int32Type),
            ArrowType::Int64 => int!(Int64Type),
            ArrowType::Float32 => Arc::new(float::unary(
                UnaryOp::Negate,
                a.as_primitive::<Float32Type>(),
            )?),
            ArrowType::Float64 => Arc::new(float::unary(
                UnaryOp::Negate,
                a.as_primitive::<Float64Type>(),
            )?),
            _ => Arc::new(decimal_unary(
                UnaryOp::Negate,
                a.as_primitive::<Decimal128Type>(),
            )?),
        };
        Ok(match v {
            Value::Scalar(_) => Value::Scalar(out),
            Value::Array(_) => Value::Array(out),
        })
    }
}

/// `l div r` (`IntegralDivide`): a `bigint` truncated towards zero, from
/// `bigint` or decimal operands (the analyzer cast them).
#[derive(Debug)]
struct IntegralDivide {
    l: Arc<dyn PhysExpr>,
    r: Arc<dyn PhysExpr>,
    /// Decimal operands' types.
    decimal: Option<((u8, i8), (u8, i8))>,
    ansi: bool,
    kernels: KernelCtx,
}

pub(super) fn integral_divide(
    l: Arc<dyn PhysExpr>,
    r: Arc<dyn PhysExpr>,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let decimal = match (l.data_type(), r.data_type()) {
        (ArrowType::Int64, ArrowType::Int64) => None,
        (ArrowType::Decimal128(lp, ls), ArrowType::Decimal128(rp, rs)) => {
            Some(((*lp, *ls), (*rp, *rs)))
        }
        (lt, rt) => return Err(unsupported(&format!("`div` on {lt} and {rt}"))),
    };
    Ok(Arc::new(IntegralDivide {
        l,
        r,
        decimal,
        ansi: env.ansi,
        kernels: env.kernels,
    }))
}

impl PhysExpr for IntegralDivide {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Int64
    }

    fn nullable(&self) -> bool {
        true
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let (mut l, r) = (self.l.eval(batch)?, self.r.eval(batch)?);
        let scalar = l.is_scalar() && r.is_scalar();
        if scalar {
            l = Value::Array(Arc::clone(l.array()));
        }
        let ctx = &self.kernels;
        let out: ArrayRef = match self.decimal {
            None => {
                let (a, b) = (prim::<Int64Type>(&l), prim::<Int64Type>(&r));
                Arc::new(if self.ansi {
                    integer::integral_divide::<AnsiError>(a, b, ctx)?
                } else {
                    integer::integral_divide::<JavaWrap>(a, b, ctx)?
                })
            }
            Some((lt, rt)) => {
                let (a, b) = (prim::<Decimal128Type>(&l), prim::<Decimal128Type>(&r));
                Arc::new(if self.ansi {
                    decimal_integral_divide::<AnsiError>(a, b, lt, rt, ctx)?
                } else {
                    decimal_integral_divide::<NullOnOverflow>(a, b, lt, rt, ctx)?
                })
            }
        };
        Ok(if scalar {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}
