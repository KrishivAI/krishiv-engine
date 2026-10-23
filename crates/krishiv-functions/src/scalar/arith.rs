// SPDX-License-Identifier: Apache-2.0
//! Arithmetic functions (P2-T021a kernels): `abs`, `positive`, `mod` and
//! the `try_*` family (`try_add`, `try_subtract`, `try_multiply`,
//! `try_divide`, `try_mod`).
//!
//! The analyzer gives both operands one type, as for the operators: the
//! wider integral type, `double` (always for `/`), or `decimal` operands the
//! result type of the operation is computed from. A `try_*` function returns
//! NULL where the operator would raise (overflow, a zero divisor); `mod`
//! follows the operator `%` (an error under ANSI, NULL otherwise for a zero
//! divisor).

use std::sync::Arc;

use arrow_array::types::{
    Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
};
use arrow_array::{ArrayRef, PrimitiveArray};
use arrow_schema::DataType;
use krishiv_kernels::decimal::{DecimalOp, DecimalPlan, decimal_arith, decimal_unary, result_type};
use krishiv_kernels::integer::{self, SparkIntegral, UnaryOp};
use krishiv_kernels::{
    Add, AnsiError, ArithOp, JavaWrap, KernelCtx, KernelError, Mul, NullOnOverflow, Operand, Sub,
    float,
};

use super::args::Inputs;
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "abs",
    "positive",
    "negative",
    "mod",
    "try_add",
    "try_subtract",
    "try_multiply",
    "try_divide",
    "try_mod",
];

/// A binary arithmetic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

/// The operations on integral operands (`/` divides `double`s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntOp {
    Add,
    Sub,
    Mul,
    Rem,
}

/// What an overflow or a zero divisor does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    /// NULL (`try_*`).
    Null,
    /// An error (ANSI).
    Raise,
    /// Wrap; a zero divisor is NULL (legacy).
    Wrap,
}

fn int_op<T: SparkIntegral>(
    op: IntOp,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ctx: &KernelCtx,
    policy: Policy,
) -> Result<PrimitiveArray<T>, KernelError> {
    macro_rules! with {
        ($p:ty) => {
            match op {
                IntOp::Add => integer::binary::<T, Add, $p>(l, r, ctx),
                IntOp::Sub => integer::binary::<T, Sub, $p>(l, r, ctx),
                IntOp::Mul => integer::binary::<T, Mul, $p>(l, r, ctx),
                IntOp::Rem => integer::remainder::<T, $p>(l, r, ctx),
            }
        };
    }
    match policy {
        Policy::Null => with!(NullOnOverflow),
        Policy::Raise => with!(AnsiError),
        Policy::Wrap => with!(JavaWrap),
    }
}

fn float_op<T>(
    op: Op,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ctx: &KernelCtx,
    raise: bool,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: arrow_array::ArrowPrimitiveType,
    T::Native: float::SparkFloat,
{
    match op {
        Op::Add => float::binary(ArithOp::Add, l, r, ctx),
        Op::Sub => float::binary(ArithOp::Subtract, l, r, ctx),
        Op::Mul => float::binary(ArithOp::Multiply, l, r, ctx),
        // Only `try_divide` divides: a zero divisor is NULL.
        Op::Div => float::divide::<T, JavaWrap>(l, r, ctx),
        Op::Rem if raise => float::remainder::<T, AnsiError>(l, r, ctx),
        Op::Rem => float::remainder::<T, JavaWrap>(l, r, ctx),
    }
}

/// `l op r` for two columns of one type.
fn binary(op: Op, try_: bool, types: &[DataType], opts: &BindOptions) -> Option<Binding> {
    let [l, r] = types else { return None };
    let ctx = opts.kernel;
    let policy = match (try_, opts.ansi) {
        (true, _) => Policy::Null,
        (false, true) => Policy::Raise,
        (false, false) => Policy::Wrap,
    };
    let int_kind = match op {
        Op::Add => Some(IntOp::Add),
        Op::Sub => Some(IntOp::Sub),
        Op::Mul => Some(IntOp::Mul),
        Op::Rem => Some(IntOp::Rem),
        Op::Div => None,
    };
    macro_rules! int {
        ($t:ty) => {{
            let kind = int_kind?;
            Binding::new(l.clone(), move |x: &Inputs<'_>| {
                let (a, b) = (x.prim::<$t>(0)?, x.prim::<$t>(1)?);
                Ok(
                    Arc::new(int_op::<$t>(kind, a.operand(), b.operand(), &ctx, policy)?)
                        as ArrayRef,
                )
            })
        }};
    }
    macro_rules! flt {
        ($t:ty) => {
            Binding::new(l.clone(), move |x: &Inputs<'_>| {
                let (a, b) = (x.prim::<$t>(0)?, x.prim::<$t>(1)?);
                let raise = policy == Policy::Raise;
                Ok(
                    Arc::new(float_op::<$t>(op, a.operand(), b.operand(), &ctx, raise)?)
                        as ArrayRef,
                )
            })
        };
    }
    Some(match (l, r) {
        (DataType::Int8, DataType::Int8) => int!(Int8Type),
        (DataType::Int16, DataType::Int16) => int!(Int16Type),
        (DataType::Int32, DataType::Int32) => int!(Int32Type),
        (DataType::Int64, DataType::Int64) => int!(Int64Type),
        (DataType::Float32, DataType::Float32) => flt!(Float32Type),
        (DataType::Float64, DataType::Float64) => flt!(Float64Type),
        (DataType::Decimal128(p1, s1), DataType::Decimal128(p2, s2)) => {
            let dop = match op {
                Op::Add => DecimalOp::Add,
                Op::Sub => DecimalOp::Subtract,
                Op::Mul => DecimalOp::Multiply,
                Op::Div => DecimalOp::Divide,
                Op::Rem => DecimalOp::Remainder,
            };
            let (lt, rt) = ((*p1, *s1), (*p2, *s2));
            let out = result_type(dop, lt, rt, opts.decimal_precision_loss).ok()?;
            let plan = DecimalPlan::new(dop, lt, rt, out).ok()?;
            let raise = policy == Policy::Raise;
            Binding::new(DataType::Decimal128(out.0, out.1), move |x: &Inputs<'_>| {
                let (a, b) = (x.prim::<Decimal128Type>(0)?, x.prim::<Decimal128Type>(1)?);
                let out = if raise {
                    decimal_arith::<AnsiError>(a.operand(), b.operand(), &plan, &ctx)
                } else {
                    decimal_arith::<NullOnOverflow>(a.operand(), b.operand(), &plan, &ctx)
                }?;
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        _ => return None,
    })
}

/// `abs` / `negative` of one column.
fn unary_arith(op: UnaryOp, t: &DataType, opts: &BindOptions) -> Option<Binding> {
    let ctx = opts.kernel;
    let ansi = opts.ansi;
    macro_rules! int {
        ($t:ty) => {
            Binding::new(t.clone(), move |x: &Inputs<'_>| {
                let a = x.array::<$t>(0)?;
                Ok(Arc::new(if ansi {
                    integer::unary::<$t, AnsiError>(op, &a, &ctx)?
                } else {
                    integer::unary::<$t, JavaWrap>(op, &a, &ctx)?
                }) as ArrayRef)
            })
        };
    }
    Some(match t {
        DataType::Int8 => int!(Int8Type),
        DataType::Int16 => int!(Int16Type),
        DataType::Int32 => int!(Int32Type),
        DataType::Int64 => int!(Int64Type),
        DataType::Float32 => Binding::new(t.clone(), move |x: &Inputs<'_>| {
            Ok(Arc::new(float::unary(op, &x.array::<Float32Type>(0)?)?) as ArrayRef)
        }),
        DataType::Float64 => Binding::new(t.clone(), move |x: &Inputs<'_>| {
            Ok(Arc::new(float::unary(op, &x.array::<Float64Type>(0)?)?) as ArrayRef)
        }),
        DataType::Decimal128(..) => Binding::new(t.clone(), move |x: &Inputs<'_>| {
            Ok(Arc::new(decimal_unary(op, &x.array::<Decimal128Type>(0)?)?) as ArrayRef)
        }),
        _ => return None,
    })
}

fn abs(t: &DataType, opts: &BindOptions) -> Option<Binding> {
    unary_arith(UnaryOp::Abs, t, opts)
}

fn negative(t: &DataType, opts: &BindOptions) -> Option<Binding> {
    unary_arith(UnaryOp::Negate, t, opts)
}

/// Binds the arithmetic function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("positive", [t])
            if matches!(
                t,
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::Float32
                    | DataType::Float64
                    | DataType::Decimal128(..)
            ) =>
        {
            Some(Binding::new(t.clone(), |x: &Inputs<'_>| x.column(0)))
        }
        ("abs", [t]) => abs(t, opts),
        ("negative", [t]) => negative(t, opts),
        _ => {
            let (op, try_) = match listed {
                "mod" => (Op::Rem, false),
                "try_add" => (Op::Add, true),
                "try_subtract" => (Op::Sub, true),
                "try_multiply" => (Op::Mul, true),
                "try_divide" => (Op::Div, true),
                "try_mod" => (Op::Rem, true),
                _ => return Some((listed, Err(bad_types(listed, args)))),
            };
            binary(op, try_, &types, opts)
        }
    };
    Some(match binding {
        Some(b) => (listed, Ok(b)),
        None => (listed, Err(bad_types(listed, args))),
    })
}
