// SPDX-License-Identifier: Apache-2.0
//! The math functions of the kernel evaluator (P2-T026): the `double`
//! functions, `round`/`bround`/`ceil`/`floor`, `sign` and the constants
//! `pi()`/`e()`, typed with Spark 4.2's
//! implicit casts (a string, NULL or any other numeric type is a `double`
//! where the function takes one).

use std::sync::Arc;

use arrow_array::types::{
    Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
};
use arrow_array::{ArrayRef, Float64Array, cast::AsArray};
use arrow_schema::DataType;
use krishiv_kernels::OverflowMode;
use krishiv_kernels::math::{
    self, BinaryFn, DecimalRounding, RoundMode, UnaryFn, round_decimal_type,
};
use krishiv_sql::ast;

use super::{Binder, EvalError, Evaluator, Node, Skip, is_integral, is_numeric, simple_string};

/// A math function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum MathFn {
    Unary(UnaryFn),
    Binary(BinaryFn),
    /// `round` / `bround` to a constant scale.
    Round(RoundMode, i32),
    /// `ceil` / `floor` of a `double` as a `bigint`.
    CeilFloorLong {
        ceil: bool,
    },
    /// `ceil` / `floor` of a decimal to a constant scale.
    CeilFloorDecimal {
        ceil: bool,
        scale: i32,
    },
}

fn unary_fn(name: &str) -> Option<UnaryFn> {
    Some(match name {
        "sqrt" => UnaryFn::Sqrt,
        "cbrt" => UnaryFn::Cbrt,
        "exp" => UnaryFn::Exp,
        "expm1" => UnaryFn::Expm1,
        "ln" => UnaryFn::Ln,
        "log10" => UnaryFn::Log10,
        "log2" => UnaryFn::Log2,
        "log1p" => UnaryFn::Log1p,
        "sin" => UnaryFn::Sin,
        "cos" => UnaryFn::Cos,
        "tan" => UnaryFn::Tan,
        "asin" => UnaryFn::Asin,
        "acos" => UnaryFn::Acos,
        "atan" => UnaryFn::Atan,
        "sinh" => UnaryFn::Sinh,
        "cosh" => UnaryFn::Cosh,
        "tanh" => UnaryFn::Tanh,
        "asinh" => UnaryFn::Asinh,
        "acosh" => UnaryFn::Acosh,
        "atanh" => UnaryFn::Atanh,
        "cot" => UnaryFn::Cot,
        "csc" => UnaryFn::Csc,
        "sec" => UnaryFn::Sec,
        "degrees" => UnaryFn::Degrees,
        "radians" => UnaryFn::Radians,
        "rint" => UnaryFn::Rint,
        "sign" | "signum" => UnaryFn::Signum,
        _ => return None,
    })
}

/// `DecimalType.forType` of an integral type.
fn integral_decimal(t: &DataType) -> Option<DataType> {
    Some(match t {
        DataType::Int8 => DataType::Decimal128(3, 0),
        DataType::Int16 => DataType::Decimal128(5, 0),
        DataType::Int32 => DataType::Decimal128(10, 0),
        DataType::Int64 => DataType::Decimal128(20, 0),
        _ => return None,
    })
}

/// The value of a constant numeric expression (a literal, or casts of one).
fn constant(n: &Node) -> Option<f64> {
    match n {
        Node::Lit(a) if a.len() == 1 && a.is_valid(0) => match a.data_type() {
            DataType::Int8 => Some(f64::from(a.as_primitive::<Int8Type>().value(0))),
            DataType::Int16 => Some(f64::from(a.as_primitive::<Int16Type>().value(0))),
            DataType::Int32 => Some(f64::from(a.as_primitive::<Int32Type>().value(0))),
            #[expect(clippy::cast_precision_loss, reason = "scales are small")]
            DataType::Int64 => Some(a.as_primitive::<Int64Type>().value(0) as f64),
            DataType::Float32 => Some(f64::from(a.as_primitive::<Float32Type>().value(0))),
            DataType::Float64 => Some(a.as_primitive::<Float64Type>().value(0)),
            DataType::Decimal128(_, s) => {
                #[expect(clippy::cast_precision_loss, reason = "scales are small")]
                let v = a.as_primitive::<Decimal128Type>().value(0) as f64;
                Some(v / 10_f64.powi(i32::from(*s)))
            }
            DataType::Utf8 => a.as_string::<i32>().value(0).trim().parse().ok(),
            _ => None,
        },
        Node::Cast(c, ..) => constant(c),
        _ => None,
    }
}

impl Binder<'_> {
    /// A numeric operand as a `double`.
    fn double_arg(self, e: &ast::Expr) -> Result<Node, Skip> {
        let n = self.bind(e)?;
        match n.ty() {
            DataType::Float64 => Ok(n),
            DataType::Null | DataType::Utf8 => Ok(self.implicit(n, &DataType::Float64)),
            t if is_numeric(&t) => Ok(self.implicit(n, &DataType::Float64)),
            t => Err(format!(
                "analysis: {} as a double argument",
                simple_string(&t)
            )),
        }
    }

    /// The constant integer scale of `round`/`ceil`/`floor`.
    fn scale_arg(self, e: &ast::Expr) -> Result<i32, Skip> {
        let n = self.bind(e)?;
        let v = constant(&n).ok_or_else(|| "scale that is not a numeric constant".to_owned())?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Spark truncates the scale to int"
        )]
        Ok(v.trunc() as i32)
    }

    /// The math function `name` applied to `args`, or `None` for another name.
    pub(super) fn math_function(
        self,
        name: &str,
        args: &[&ast::Expr],
    ) -> Option<Result<Node, Skip>> {
        let known = unary_fn(name).is_some()
            || matches!(
                name,
                "pow"
                    | "power"
                    | "atan2"
                    | "hypot"
                    | "log"
                    | "round"
                    | "bround"
                    | "ceil"
                    | "ceiling"
                    | "floor"
                    | "pi"
                    | "e"
            );
        known.then(|| self.math_call(name, args))
    }

    fn math_call(self, name: &str, args: &[&ast::Expr]) -> Result<Node, Skip> {
        let arity = |ok: bool| -> Result<(), Skip> {
            ok.then_some(())
                .ok_or_else(|| format!("analysis: {name} with {} arguments", args.len()))
        };
        let double = DataType::Float64;
        if let Some(f) = unary_fn(name) {
            arity(args.len() == 1)?;
            return Ok(Node::Math(
                MathFn::Unary(f),
                vec![self.double_arg(args[0])?],
                double,
            ));
        }
        match name {
            "pi" | "e" => {
                arity(args.is_empty())?;
                let v = if name == "pi" {
                    std::f64::consts::PI
                } else {
                    std::f64::consts::E
                };
                Ok(Node::Lit(Arc::new(Float64Array::from(vec![v]))))
            }
            "log" if args.len() == 1 => Ok(Node::Math(
                MathFn::Unary(UnaryFn::Ln),
                vec![self.double_arg(args[0])?],
                double,
            )),
            "pow" | "power" | "atan2" | "hypot" | "log" => {
                arity(args.len() == 2)?;
                let f = match name {
                    "atan2" => BinaryFn::Atan2,
                    "hypot" => BinaryFn::Hypot,
                    "log" => BinaryFn::Log,
                    _ => BinaryFn::Pow,
                };
                let (l, r) = (self.double_arg(args[0])?, self.double_arg(args[1])?);
                Ok(Node::Math(MathFn::Binary(f), vec![l, r], double))
            }
            "round" | "bround" => {
                arity((1..=2).contains(&args.len()))?;
                let mode = if name == "round" {
                    RoundMode::HalfUp
                } else {
                    RoundMode::HalfEven
                };
                let scale = args.get(1).map_or(Ok(0), |e| self.scale_arg(e))?;
                let x = self.bind(args[0])?;
                let x = match x.ty() {
                    DataType::Null | DataType::Utf8 => self.implicit(x, &double),
                    t if is_numeric(&t) => x,
                    t => return Err(format!("analysis: {name} of {}", simple_string(&t))),
                };
                let ty = match x.ty() {
                    DataType::Decimal128(p, s) => {
                        let (p, s) = round_decimal_type(p, s, scale);
                        DataType::Decimal128(p, s)
                    }
                    t => t,
                };
                Ok(Node::Math(MathFn::Round(mode, scale), vec![x], ty))
            }
            "ceil" | "ceiling" | "floor" => {
                arity((1..=2).contains(&args.len()))?;
                let ceil = name != "floor";
                let x = self.bind(args[0])?;
                if args.len() == 1 {
                    return match x.ty() {
                        t if is_integral(&t) => Ok(self.implicit(x, &DataType::Int64)),
                        DataType::Null | DataType::Utf8 | DataType::Float32 | DataType::Float64 => {
                            let x = self.implicit(x, &double);
                            Ok(Node::Math(
                                MathFn::CeilFloorLong { ceil },
                                vec![x],
                                DataType::Int64,
                            ))
                        }
                        DataType::Decimal128(p, s) => {
                            let (op, os) = round_decimal_type(p, s, 0);
                            Ok(Node::Math(
                                MathFn::CeilFloorDecimal { ceil, scale: 0 },
                                vec![x],
                                DataType::Decimal128(op, os),
                            ))
                        }
                        t => Err(format!("analysis: {name} of {}", simple_string(&t))),
                    };
                }
                let scale = self.scale_arg(args[1])?;
                let as_decimal = match x.ty() {
                    DataType::Decimal128(..) => x.ty(),
                    DataType::Float32 => DataType::Decimal128(14, 7),
                    DataType::Float64 => DataType::Decimal128(30, 15),
                    DataType::Utf8 => DataType::Decimal128(38, 18),
                    t => integral_decimal(&t)
                        .ok_or_else(|| format!("analysis: {name} of {}", simple_string(&t)))?,
                };
                let x = self.implicit(x, &as_decimal);
                let DataType::Decimal128(p, s) = as_decimal else {
                    return Err("decimal".to_owned());
                };
                let (op, os) = round_decimal_type(p, s, scale);
                Ok(Node::Math(
                    MathFn::CeilFloorDecimal { ceil, scale },
                    vec![x],
                    DataType::Decimal128(op, os),
                ))
            }
            other => Err(format!("math function `{other}`")),
        }
    }
}

impl Evaluator<'_> {
    fn overflow_mode(&self) -> OverflowMode {
        if self.ansi {
            OverflowMode::AnsiError
        } else {
            OverflowMode::JavaWrap
        }
    }

    /// Evaluates math function `f` over `args`.
    pub(super) fn math_fn(&self, f: MathFn, args: &[Node]) -> Result<ArrayRef, EvalError> {
        let v = args
            .iter()
            .map(|a| self.eval(a))
            .collect::<Result<Vec<_>, _>>()?;
        let a = &v[0];
        Ok(match f {
            MathFn::Unary(u) => {
                Arc::new(math::unary(u, a.as_primitive::<Float64Type>(), &self.ctx))
            }
            MathFn::Binary(b) => {
                let (l, r) = self.operands::<Float64Type>(a.as_ref(), v[1].as_ref());
                Arc::new(math::binary(b, l, r)?)
            }
            MathFn::Round(mode, scale) => {
                macro_rules! int {
                    ($t:ty) => {
                        Arc::new(math::round_integral(
                            a.as_primitive::<$t>(),
                            scale,
                            mode,
                            self.overflow_mode(),
                        )?)
                    };
                }
                match a.data_type() {
                    DataType::Int8 => int!(Int8Type),
                    DataType::Int16 => int!(Int16Type),
                    DataType::Int32 => int!(Int32Type),
                    DataType::Int64 => int!(Int64Type),
                    DataType::Float32 => Arc::new(math::round_f32(
                        a.as_primitive::<Float32Type>(),
                        scale,
                        mode,
                    )?),
                    DataType::Float64 => Arc::new(math::round_f64(
                        a.as_primitive::<Float64Type>(),
                        scale,
                        mode,
                    )?),
                    _ => Arc::new(math::round_decimal(
                        a.as_primitive::<Decimal128Type>(),
                        scale,
                        mode.into(),
                    )?),
                }
            }
            MathFn::CeilFloorLong { ceil } => Arc::new(math::ceil_floor_to_long(
                a.as_primitive::<Float64Type>(),
                ceil,
            )),
            MathFn::CeilFloorDecimal { ceil, scale } => {
                let rounding = if ceil {
                    DecimalRounding::Ceiling
                } else {
                    DecimalRounding::Floor
                };
                Arc::new(math::round_decimal(
                    a.as_primitive::<Decimal128Type>(),
                    scale,
                    rounding,
                )?)
            }
        })
    }
}
