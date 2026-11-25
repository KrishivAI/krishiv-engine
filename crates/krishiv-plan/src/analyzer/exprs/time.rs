// SPDX-License-Identifier: Apache-2.0
//! `TIME` literals and casts to `TIME(p)` by profile (P3 R3): Spark 4.3 keeps up to nine
//! fraction digits in a literal and cuts a string cast to `TIME(p)` to `p` digits; Spark 4.2 keeps
//! six in a literal and every digit a string cast reads.

use super::*;

impl Resolver<'_> {
    /// `TIME '...'`: a string cast to `TIME(p)`. Kept out of the recursive
    /// `expr` match so its locals do not widen every level's frame.
    /// Returns the cast, the literal as shown and the precision `p`.
    #[inline(never)]
    pub(super) fn time_literal(
        &self,
        e: &Arc<Expr>,
        value: &str,
    ) -> Result<(Arc<Expr>, Arc<Expr>, u8), AnalysisError> {
        // Spark evaluates a `TIME '..'` literal with `spark.sql.timeType.enabled` off too (4.2.0
        // probe); the functions that take one refuse it (`unexecutable`).
        // A time is a string cast to `TIME(6)` (design P2-T181): the
        // value is checked again here, in case the plan came from a
        // front end other than the SQL parser. Spark 4.3 keeps up to
        // nine digits and types the literal `TIME(max(6, digits))`
        // (more than nine: `INVALID_TIME_LITERAL_PRECISION`); Spark 4.2
        // drops the digits past the sixth, in its name too.
        let nanos_literal = self.ctx.profile >= crate::analyzer::SparkProfile::V4_3;
        let digits = krishiv_kernels::cast::time_fraction_digits(value);
        if nanos_literal && digits > usize::from(DataType::TIME_MAX_PRECISION) {
            return Err(AnalysisError::new(
                krishiv_common::error::InvalidTimeLiteralPrecision {
                    value: format!("'{value}'"),
                },
                e.origin,
            ));
        }
        let (parsed, precision) = if nanos_literal {
            let p = u8::try_from(digits.max(6)).unwrap_or(DataType::TIME_MAX_PRECISION);
            (krishiv_kernels::cast::parse_time_nanos(value), p)
        } else {
            (krishiv_kernels::cast::parse_time(value), 6)
        };
        let Some(nanos) = parsed else {
            return Err(AnalysisError::new(
                krishiv_common::error::InvalidTypedLiteral {
                    value: format!("'{value}'"),
                    value_type: "\"TIME\"".to_owned(),
                },
                e.origin,
            ));
        };
        let mut text = krishiv_kernels::cast::format_time(nanos);
        // Spark 4.3 names a `TIME(7..=9)` literal with exactly `precision` digits (its
        // `Literal.sql` pads back the zeros the formatter drops: `TIME '12:00:00.123456700'`);
        // a `TIME(6)` keeps the formatter's text.
        if precision > 6 {
            let have = text.rsplit_once('.').map_or(0, |(_, f)| f.len());
            if have == 0 {
                text.push('.');
            }
            text.extend(std::iter::repeat_n(
                '0',
                usize::from(precision).saturating_sub(have),
            ));
        }
        // Spark names a time by its value as read: the shown literal
        // holds that text, so the name follows the profile's digits.
        let shown = Expr::new(ExprNode::TypedLiteral {
            kind: TypedLiteralKind::Time,
            value: text.clone(),
        })
        .with_origin(e.origin);
        let text = Arc::new(Expr::from(Literal::String(text)).with_origin(e.origin));
        // A string cast to `TIME(7|8)` reads six digits (Spark 4.2's string cast): the literal
        // reads the text as a `TIME(9)` and cuts that to its precision, which keeps every digit
        // it holds (`TIME '12:00:00.1234567'` is `12:00:00.1234567`).
        let read_as = if precision > 6 {
            DataType::Time {
                precision: DataType::TIME_MAX_PRECISION,
            }
        } else {
            DataType::Time { precision }
        };
        let mut cast = Expr::new(ExprNode::Cast {
            expr: text,
            to: read_as.clone(),
            mode: CastMode::Ansi,
        })
        .with_origin(e.origin);
        if read_as != (DataType::Time { precision }) {
            cast = Expr::new(ExprNode::Cast {
                expr: Arc::new(cast),
                to: DataType::Time { precision },
                mode: CastMode::Ansi,
            })
            .with_origin(e.origin);
        }
        Ok((Arc::new(cast), Arc::new(shown), precision))
    }

    /// The trailing literals a built-in TIME call gets by profile, after its `n` arguments.
    /// `to_time(s)` keeps nine fraction digits from Spark 4.3 on and six before
    /// (`stringToTime`): the older grammar follows as a trailing `false`. `extract(SECOND FROM t)`
    /// reads a `time(p)` to `p` digits whatever digits the value holds (a string cast keeps all of
    /// them under 4.2, and Arrow's `Time64` does not carry `p`): a `p` other than 6 follows as a
    /// literal, and under Spark 4.3 a `true` after it, since 4.3 types the seconds of every
    /// `time(p)` `decimal(p + 2, p)` (4.2: `decimal(8, 6)`).
    pub(super) fn time_trailing_args(
        &self,
        name: &str,
        n: usize,
        kids: &[Bound],
        origin: Origin,
        out: &mut Vec<Arc<Expr>>,
    ) {
        let spark_4_3 = self.ctx.profile >= crate::analyzer::SparkProfile::V4_3;
        let literal = |l: Literal| Arc::new(Expr::new(ExprNode::Literal(l)).with_origin(origin));
        if matches!(name, "to_time" | "try_to_time") && n == 1 && !spark_4_3 {
            out.push(literal(Literal::Boolean(false)));
        }
        if matches!(name, "extract" | "date_part" | "datepart")
            && n == 2
            && let Some(DataType::Time { precision }) =
                kids.get(1).and_then(|k| k.ty.data_type.as_ref())
            && *precision != 6
        {
            out.push(literal(Literal::Integer(i32::from(*precision))));
            if spark_4_3 {
                out.push(literal(Literal::Boolean(true)));
            }
        }
    }
}
