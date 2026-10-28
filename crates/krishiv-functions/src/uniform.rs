// SPDX-License-Identifier: Apache-2.0
//! The `uniform(min, max)` result-type rule, shared by the analyzer rule
//! (`registry::misc`, over [`krishiv_types::DataType`]) and the executor
//! binding (`scalar::random`, over Arrow types): Spark's `Uniform.dataType`.
//!
//! Both layers map their own type to a [`UniformKind`]; [`uniform_result`]
//! then picks the result kind, so the rank order, the `double`/`float`
//! priority and the decimal rule live once. The layers only map kinds back
//! to their own type. `uniform_result_agrees_with_the_executor_bind` (in
//! `registry::tests`) pins every pair of kinds end to end across both
//! layers, so a mapper cannot drift from the rule.

/// The numeric kinds `uniform`'s result rule distinguishes: the four
/// integral widths in Spark's `UpCastRule` order, the two floating widths, a
/// decimal width, and everything else (strings, booleans, timestamps, ...),
/// which reads as `DOUBLE` — as does an untyped `NULL` bound, which Spark
/// casts to `DOUBLE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UniformKind {
    Byte,
    Short,
    Int,
    Long,
    Float,
    Double,
    Decimal { precision: u8, scale: i8 },
    Other,
}

/// Whether `kind` is one of the four integral widths.
fn is_integral(kind: UniformKind) -> bool {
    matches!(
        kind,
        UniformKind::Byte | UniformKind::Short | UniformKind::Int | UniformKind::Long
    )
}

/// The result kind of `uniform(min, max)` for bound kinds `min`/`max`.
pub(crate) fn uniform_result(min: UniformKind, max: UniformKind) -> UniformKind {
    let rank = |kind: UniformKind| match kind {
        UniformKind::Byte => 0,
        UniformKind::Short => 1,
        UniformKind::Int => 2,
        _ => 3,
    };
    match (min, max) {
        (a, b) if is_integral(a) && is_integral(b) => {
            if rank(a) >= rank(b) {
                a
            } else {
                b
            }
        }
        (UniformKind::Double, _) | (_, UniformKind::Double) => UniformKind::Double,
        (UniformKind::Float, _) | (_, UniformKind::Float) => UniformKind::Float,
        (
            UniformKind::Decimal {
                precision: p1,
                scale: s1,
            },
            UniformKind::Decimal {
                precision: p2,
                scale: s2,
            },
        ) => {
            if is_wider_decimal((p1, s1), (p2, s2)) {
                min
            } else {
                max
            }
        }
        (d @ UniformKind::Decimal { .. }, _) | (_, d @ UniformKind::Decimal { .. }) => d,
        _ => UniformKind::Double,
    }
}

/// Whether the decimal `(p1, s1)` is wider than `(p2, s2)` (Spark's
/// `DecimalType.isWiderThan`: every part at least as wide).
pub(crate) fn is_wider_decimal(left: (u8, i8), right: (u8, i8)) -> bool {
    let (p1, s1) = (i32::from(left.0), i32::from(left.1));
    let (p2, s2) = (i32::from(right.0), i32::from(right.1));
    p1 - s1 >= p2 - s2 && s1 >= s2 && p1 >= p2
}
