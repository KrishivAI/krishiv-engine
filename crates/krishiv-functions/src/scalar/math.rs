// SPDX-License-Identifier: Apache-2.0
//! Math functions (P2-T026 kernels): the `double` functions, `pow`-style
//! binary functions, `round`/`bround`/`ceil`/`floor`, `pmod`,
//! `greatest`/`least` and the constants `pi()` and `e()`.
//!
//! The analyzer casts a string, `NULL` or any other numeric type to the
//! `double` a function takes, so a `double` function binds `double`
//! arguments only. `round`, `ceil` and `floor` take their scale from a
//! literal argument.

use std::sync::Arc;

use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, new_null_array};
use arrow_schema::DataType;
use krishiv_common::error::KrishivError;
use krishiv_kernels::OverflowMode;
use krishiv_kernels::math::{
    self, BinaryFn, DecimalRounding, Extremum, RoundMode, UnaryFn, round_decimal_type,
};

use super::args::{Inputs, internal, is_binary, is_string, unsupported};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "sqrt",
    "cbrt",
    "exp",
    "expm1",
    "ln",
    "log10",
    "log2",
    "log1p",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "cot",
    "csc",
    "sec",
    "degrees",
    "radians",
    "rint",
    "sign",
    "signum",
    "log",
    "pow",
    "power",
    "atan2",
    "hypot",
    "pi",
    "e",
    "round",
    "bround",
    "ceil",
    "ceiling",
    "floor",
    "pmod",
    "greatest",
    "least",
    "width_bucket",
];

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

/// The overflow policy of integral math under the session's ANSI mode.
pub(super) fn overflow_mode(opts: &BindOptions) -> OverflowMode {
    if opts.ansi {
        OverflowMode::AnsiError
    } else {
        OverflowMode::JavaWrap
    }
}

/// The scale argument of `round`, `ceil` and `floor`: a literal `int`
/// (`None` for a NULL scale). Anything else is not a binding Spark's
/// analyzer lets through.
fn scale_of(arg: Option<&BindArg>, function: &str) -> Result<Option<i32>, KrishivError> {
    let Some(arg) = arg else { return Ok(Some(0)) };
    let BindArg::Literal(a) = arg else {
        return Err(unsupported(format!("{function} with a scale that varies")));
    };
    let Some(a) = arrow_array::cast::AsArray::as_primitive_opt::<Int32Type>(a) else {
        return Err(internal(format!(
            "{function}: a scale of type {}",
            a.data_type()
        )));
    };
    Ok(arrow_array::Array::is_valid(a, 0).then(|| a.value(0)))
}

/// `Math.pi` / `Math.e` as a column.
fn constant(v: f64) -> Binding {
    Binding::new(DataType::Float64, move |x: &Inputs<'_>| {
        Ok(Arc::new(Float64Array::from(vec![v; x.rows()])) as ArrayRef)
    })
}

/// Spark's `WidthBucket.computeBucketNumber`: NULL for a bad bucket count,
/// a NaN value or a NaN, infinite or empty range.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Spark converts the bucket count and the position with JVM casts"
)]
fn width_bucket(value: f64, min: f64, max: f64, buckets: i64) -> Option<i64> {
    if buckets <= 0
        || buckets == i64::MAX
        || value.is_nan()
        || min == max
        || !min.is_finite()
        || !max.is_finite()
    {
        return None;
    }
    let (lower, upper) = (min.min(max), min.max(max));
    // `as i64` saturates like the JVM's `toLong`.
    let scaled = |num: f64| (buckets as f64 * num / (upper - lower)) as i64 + 1;
    Some(if min < max {
        if value < lower {
            0
        } else if value >= upper {
            buckets + 1
        } else {
            scaled(value - lower)
        }
    } else if value > upper {
        0
    } else if value <= lower {
        buckets + 1
    } else {
        scaled(upper - value)
    })
}

/// `round`/`bround` of column 0 to `scale`.
fn round(t: &DataType, scale: Option<i32>, mode: RoundMode, opts: &BindOptions) -> Option<Binding> {
    let overflow = overflow_mode(opts);
    macro_rules! int {
        ($t:ty) => {
            Binding::new(t.clone(), move |x: &Inputs<'_>| {
                let Some(scale) = scale else {
                    return Ok(new_null_array(
                        &<$t as arrow_array::ArrowPrimitiveType>::DATA_TYPE,
                        x.rows(),
                    ));
                };
                Ok(Arc::new(math::round_integral(
                    &x.array::<$t>(0)?,
                    scale,
                    mode,
                    overflow,
                )?) as ArrayRef)
            })
        };
    }
    Some(match t {
        DataType::Int8 => int!(Int8Type),
        DataType::Int16 => int!(Int16Type),
        DataType::Int32 => int!(Int32Type),
        DataType::Int64 => int!(Int64Type),
        DataType::Float32 => Binding::new(DataType::Float32, move |x: &Inputs<'_>| {
            let Some(scale) = scale else {
                return Ok(new_null_array(&DataType::Float32, x.rows()));
            };
            Ok(Arc::new(math::round_f32(&x.array::<Float32Type>(0)?, scale, mode)?) as ArrayRef)
        }),
        DataType::Float64 => Binding::new(DataType::Float64, move |x: &Inputs<'_>| {
            let Some(scale) = scale else {
                return Ok(new_null_array(&DataType::Float64, x.rows()));
            };
            Ok(Arc::new(math::round_f64(&x.array::<Float64Type>(0)?, scale, mode)?) as ArrayRef)
        }),
        DataType::Decimal128(p, s) => {
            let (rp, rs) = round_decimal_type(*p, *s, scale.unwrap_or(0));
            let out = DataType::Decimal128(rp, rs);
            let empty = out.clone();
            Binding::new(out, move |x: &Inputs<'_>| {
                let Some(scale) = scale else {
                    return Ok(new_null_array(&empty, x.rows()));
                };
                Ok(Arc::new(math::round_decimal(
                    &x.array::<Decimal128Type>(0)?,
                    scale,
                    mode.into(),
                )?) as ArrayRef)
            })
        }
        _ => return None,
    })
}

/// `ceil`/`floor` of a `double` as a `bigint`, of a `bigint` as it is, or of
/// a decimal to `scale`.
fn ceil_floor(t: &DataType, scale: Option<i32>, ceil: bool, one_arg: bool) -> Option<Binding> {
    let rounding = if ceil {
        DecimalRounding::Ceiling
    } else {
        DecimalRounding::Floor
    };
    Some(match t {
        DataType::Int64 if one_arg => Binding::new(DataType::Int64, |x: &Inputs<'_>| x.column(0)),
        DataType::Float64 if one_arg => Binding::new(DataType::Int64, move |x: &Inputs<'_>| {
            Ok(Arc::new(math::ceil_floor_to_long(&x.array::<Float64Type>(0)?, ceil)) as ArrayRef)
        }),
        DataType::Decimal128(p, s) => {
            let (rp, rs) = round_decimal_type(*p, *s, scale.unwrap_or(0));
            let out = DataType::Decimal128(rp, rs);
            let empty = out.clone();
            Binding::new(out, move |x: &Inputs<'_>| {
                let Some(scale) = scale else {
                    return Ok(new_null_array(&empty, x.rows()));
                };
                Ok(Arc::new(math::round_decimal(
                    &x.array::<Decimal128Type>(0)?,
                    scale,
                    rounding,
                )?) as ArrayRef)
            })
        }
        _ => return None,
    })
}

/// `greatest`/`least` of columns of one type.
fn extremum(t: &DataType, kind: Extremum, n: usize) -> Option<Binding> {
    macro_rules! prim {
        ($t:ty) => {
            Binding::new(t.clone(), move |x: &Inputs<'_>| {
                let cols = (0..n)
                    .map(|i| x.array::<$t>(i))
                    .collect::<Result<Vec<_>, _>>()?;
                let refs: Vec<_> = cols.iter().collect();
                Ok(Arc::new(math::extremum_primitive(kind, &refs, |a, b| a.cmp(b))?) as ArrayRef)
            })
        };
    }
    macro_rules! flt {
        ($t:ty) => {
            Binding::new(t.clone(), move |x: &Inputs<'_>| {
                let cols = (0..n)
                    .map(|i| x.array::<$t>(i))
                    .collect::<Result<Vec<_>, _>>()?;
                let refs: Vec<_> = cols.iter().collect();
                Ok(Arc::new(math::extremum_float(kind, &refs)?) as ArrayRef)
            })
        };
    }
    Some(match t {
        DataType::Int8 => prim!(Int8Type),
        DataType::Int16 => prim!(Int16Type),
        DataType::Int32 => prim!(Int32Type),
        DataType::Int64 => prim!(Int64Type),
        DataType::Date32 => prim!(Date32Type),
        DataType::Timestamp(..) => prim!(TimestampMicrosecondType),
        DataType::Decimal128(..) => prim!(Decimal128Type),
        DataType::Float32 => flt!(Float32Type),
        DataType::Float64 => flt!(Float64Type),
        DataType::Boolean => Binding::new(DataType::Boolean, move |x: &Inputs<'_>| {
            let cols = (0..n)
                .map(|i| x.booleans(i))
                .collect::<Result<Vec<_>, _>>()?;
            let refs: Vec<_> = cols.iter().collect();
            Ok(Arc::new(math::extremum_bool(kind, &refs)?) as ArrayRef)
        }),
        t if is_string(t) => Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
            let cols = (0..n)
                .map(|i| x.strings(i))
                .collect::<Result<Vec<_>, _>>()?;
            let refs: Vec<_> = cols.iter().collect();
            Ok(Arc::new(math::extremum_bytes(kind, &refs)?) as ArrayRef)
        }),
        t if is_binary(t) => Binding::new(DataType::Binary, move |x: &Inputs<'_>| {
            let cols = (0..n)
                .map(|i| x.binaries(i))
                .collect::<Result<Vec<_>, _>>()?;
            let refs: Vec<_> = cols.iter().collect();
            Ok(Arc::new(math::extremum_bytes(kind, &refs)?) as ArrayRef)
        }),
        _ => return None,
    })
}

/// `pmod` of two columns of one type.
fn pmod(l: &DataType, r: &DataType, opts: &BindOptions) -> Option<Binding> {
    let mode = overflow_mode(opts);
    let ctx = opts.kernel;
    macro_rules! int {
        ($t:ty) => {
            Binding::new(l.clone(), move |x: &Inputs<'_>| {
                let (a, b) = (x.prim::<$t>(0)?, x.prim::<$t>(1)?);
                Ok(Arc::new(math::pmod_integral::<$t>(
                    a.operand(),
                    b.operand(),
                    mode,
                    &ctx,
                )?) as ArrayRef)
            })
        };
    }
    macro_rules! flt {
        ($t:ty) => {
            Binding::new(l.clone(), move |x: &Inputs<'_>| {
                let (a, b) = (x.prim::<$t>(0)?, x.prim::<$t>(1)?);
                Ok(Arc::new(math::pmod_float::<$t>(
                    a.operand(),
                    b.operand(),
                    mode,
                    &ctx,
                )?) as ArrayRef)
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
            let (lt, rt) = ((*p1, *s1), (*p2, *s2));
            let loss = opts.decimal_precision_loss;
            let (p, s) = krishiv_kernels::decimal::result_type(
                krishiv_kernels::decimal::DecimalOp::Remainder,
                lt,
                rt,
                loss,
            )
            .ok()?;
            Binding::new(DataType::Decimal128(p, s), move |x: &Inputs<'_>| {
                let (a, b) = (x.prim::<Decimal128Type>(0)?, x.prim::<Decimal128Type>(1)?);
                Ok(Arc::new(math::pmod_decimal(
                    a.operand(),
                    b.operand(),
                    lt,
                    rt,
                    loss,
                    mode,
                    &ctx,
                )?) as ArrayRef)
            })
        }
        _ => return None,
    })
}

/// Binds the math function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let ctx = opts.kernel;
    let double = DataType::Float64;
    let binding = match (listed, types.as_slice()) {
        (n, [DataType::Float64]) if unary_fn(n).is_some() || n == "log" => {
            let f = unary_fn(n).unwrap_or(UnaryFn::Ln);
            Binding::new(double, move |x: &Inputs<'_>| {
                Ok(Arc::new(math::unary(f, &x.array::<Float64Type>(0)?, &ctx)) as ArrayRef)
            })
        }
        ("pow" | "power" | "atan2" | "hypot" | "log", [DataType::Float64, DataType::Float64]) => {
            let f = match listed {
                "atan2" => BinaryFn::Atan2,
                "hypot" => BinaryFn::Hypot,
                "log" => BinaryFn::Log,
                _ => BinaryFn::Pow,
            };
            Binding::new(double, move |x: &Inputs<'_>| {
                let (l, r) = (x.prim::<Float64Type>(0)?, x.prim::<Float64Type>(1)?);
                Ok(Arc::new(math::binary(f, l.operand(), r.operand())?) as ArrayRef)
            })
        }
        (
            "width_bucket",
            [
                DataType::Float64,
                DataType::Float64,
                DataType::Float64,
                DataType::Int64,
            ],
        ) => Binding::new(DataType::Int64, move |x: &Inputs<'_>| {
            let v = x.array::<Float64Type>(0)?;
            let lo = x.array::<Float64Type>(1)?;
            let hi = x.array::<Float64Type>(2)?;
            let n = x.array::<Int64Type>(3)?;
            let out: Int64Array = (0..x.rows())
                .map(|i| {
                    if v.is_null(i) || lo.is_null(i) || hi.is_null(i) || n.is_null(i) {
                        return None;
                    }
                    width_bucket(v.value(i), lo.value(i), hi.value(i), n.value(i))
                })
                .collect();
            Ok(Arc::new(out) as ArrayRef)
        }),
        ("pi", []) => constant(std::f64::consts::PI),
        ("e", []) => constant(std::f64::consts::E),
        ("round" | "bround", [t] | [t, _]) => {
            let mode = if listed == "round" {
                RoundMode::HalfUp
            } else {
                RoundMode::HalfEven
            };
            let scale = match scale_of(args.get(1), listed) {
                Ok(s) => s,
                Err(e) => return Some((listed, Err(e))),
            };
            match round(t, scale, mode, opts) {
                Some(b) => b,
                None => return Some((listed, Err(bad_types(listed, args)))),
            }
        }
        ("ceil" | "ceiling" | "floor", [t] | [t, _]) => {
            let scale = match scale_of(args.get(1), listed) {
                Ok(s) => s,
                Err(e) => return Some((listed, Err(e))),
            };
            match ceil_floor(t, scale, listed != "floor", args.len() == 1) {
                Some(b) => b,
                None => return Some((listed, Err(bad_types(listed, args)))),
            }
        }
        ("pmod", [l, r]) => match pmod(l, r, opts) {
            Some(b) => b,
            None => return Some((listed, Err(bad_types(listed, args)))),
        },
        ("greatest" | "least", [first, rest @ ..])
            if !rest.is_empty() && rest.iter().all(|t| t == first) =>
        {
            let kind = if listed == "greatest" {
                Extremum::Greatest
            } else {
                Extremum::Least
            };
            match extremum(first, kind, types.len()) {
                Some(b) => b,
                None => return Some((listed, Err(bad_types(listed, args)))),
            }
        }
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
