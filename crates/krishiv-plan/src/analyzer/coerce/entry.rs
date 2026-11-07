// SPDX-License-Identifier: Apache-2.0
//! The entry points of type unification for plan-level inputs (set operations, `IN (subquery)`,
//! `UNPIVOT`, inline tables): they work on the plain type the display-only `UTF8_BINARY` marker
//! equals (ADR 0043, update 2026-11-07).

use krishiv_types::DataType;

use super::{Arg, Nested, Rules, common, common_all, common_in};

/// The type a value and a subquery column of `(values) IN (subquery)`
/// meet at.
pub(in crate::analyzer) fn common_type(
    a: &Arg<'_>,
    b: &Arg<'_>,
    case_sensitive: bool,
) -> Option<Option<DataType>> {
    let (x, y) = (plain(a), plain(b));
    common(&x.arg(a), &y.arg(b), false, case_sensitive)
}

/// An argument's type with the display-only `UTF8_BINARY` marker replaced by `STRING` (ADR 0043,
/// update 2026-11-07): unification works on the plain type the marker equals.
struct Plain(Option<DataType>);

impl Plain {
    fn arg<'a>(&'a self, of: &Arg<'_>) -> Arg<'a> {
        Arg {
            ty: self.0.as_ref(),
            nullable: of.nullable,
            literal: of.literal,
        }
    }
}

fn plain(a: &Arg<'_>) -> Plain {
    Plain(a.ty.map(DataType::without_binary_marker))
}

/// The type two set-operation columns meet at (`findWiderTypeForTwo`, the
/// legacy rules with ANSI off).
pub(in crate::analyzer) fn wider_type(
    a: &Arg<'_>,
    b: &Arg<'_>,
    rules: Rules,
) -> Option<Option<DataType>> {
    let (x, y) = (plain(a), plain(b));
    common_all(&[x.arg(a), y.arg(b)], rules)
}

/// The type two inline-table values meet at without ANSI
/// (`findWiderTypeWithoutStringPromotion`): the set-operation rules, except
/// that a string never meets another type.
pub(in crate::analyzer) fn wider_type_without_string_promotion(
    a: &Arg<'_>,
    b: &Arg<'_>,
    case_sensitive: bool,
) -> Option<Option<DataType>> {
    let (px, py) = (plain(a), plain(b));
    let (a, b) = (&px.arg(a), &py.arg(b));
    match (a.ty, b.ty) {
        (Some(x), Some(y)) if (x == &DataType::String) != (y == &DataType::String) => None,
        _ => {
            let rules = Rules {
                ansi: false,
                case_sensitive,
                concat_binary_as_string: false,
            };
            common_in(a, b, false, rules, Nested::Wider)
        }
    }
}
