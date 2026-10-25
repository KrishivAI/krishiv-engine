// SPDX-License-Identifier: Apache-2.0
//! The checks Spark's `AstBuilder.withSample` makes on a `TABLESAMPLE`.

use krishiv_common::error::{
    KrishivError, LegacyErrorTemp0014, LegacyErrorTemp0015, LegacyErrorTemp0016,
    LegacyErrorTemp0064, UnsupportedFeatureTablesampleSystemRepeatable,
    UnsupportedFeatureTablesampleSystemSampleMethod,
};

use super::{Checker, R, at};
use crate::ast::{BucketOn, SampleMethodKind, SampleType, TableSample};

/// `RandomSampler.roundingEpsilon`: a fraction may overshoot `[0, 1]` by it.
const ROUNDING_EPSILON: f64 = 1e-6;

impl Checker<'_> {
    /// A sample's checks, in `withSample`'s order.
    pub(super) fn sample(&self, s: &TableSample) -> R {
        let Some(method) = &s.method else {
            return Err(at(LegacyErrorTemp0014 {}, s.span));
        };
        let system = s.sample_type == Some(SampleType::System);
        if system && s.seed.is_some() {
            return Err(at(UnsupportedFeatureTablesampleSystemRepeatable {}, s.span));
        }
        let system_method = |name: &str| -> R {
            if system {
                let sample_method = name.to_owned();
                return Err(at(
                    UnsupportedFeatureTablesampleSystemSampleMethod { sample_method },
                    method.span,
                ));
            }
            Ok(())
        };
        let unsupported = |msg: &str| -> KrishivError {
            LegacyErrorTemp0015 {
                msg: msg.to_owned(),
            }
            .into()
        };
        match &method.kind {
            SampleMethodKind::Rows(e) => {
                system_method("ROWS")?;
                self.expr(e)
            }
            SampleMethodKind::Percent { negative, value } => {
                let fraction: f64 = value.parse().unwrap_or(f64::NAN);
                let sign = if *negative { -1.0 } else { 1.0 };
                self.fraction(sign * fraction / 100.0, s)
            }
            SampleMethodKind::Bytes(e) => {
                system_method("BYTES")?;
                self.expr(e)?;
                let text = e.span.text(self.sql);
                let literal = text.as_bytes().split_last().is_some_and(|(last, digits)| {
                    !digits.is_empty()
                        && digits.iter().all(u8::is_ascii_digit)
                        && b"bBkKmMgG".contains(last)
                });
                let error = if literal {
                    unsupported("byteLengthLiteral")
                } else {
                    LegacyErrorTemp0016 {
                        bytes_str: text.to_owned(),
                    }
                    .into()
                };
                Err(at(error, method.span))
            }
            SampleMethodKind::Bucket {
                numerator,
                denominator,
                on,
            } => {
                system_method("BUCKET")?;
                let target = match on {
                    Some(BucketOn::Column(_)) => "BUCKET x OUT OF y ON colname",
                    Some(BucketOn::Function(_)) => "BUCKET x OUT OF y ON function",
                    None => {
                        let n: f64 = numerator.parse().unwrap_or(f64::NAN);
                        let d: f64 = denominator.parse().unwrap_or(f64::NAN);
                        return self.fraction(n / d, s);
                    }
                };
                Err(at(unsupported(target), method.span))
            }
        }
    }

    fn fraction(&self, fraction: f64, s: &TableSample) -> R {
        if (-ROUNDING_EPSILON..=1.0 + ROUNDING_EPSILON).contains(&fraction) {
            return Ok(());
        }
        Err(at(
            LegacyErrorTemp0064 {
                msg: format!(
                    "Sampling fraction ({}) must be on interval [0, 1]",
                    java_double(fraction)
                ),
            },
            s.span,
        ))
    }
}

/// `Double.toString`.
fn java_double(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    let magnitude = x.abs();
    if magnitude != 0.0 && !(1e-3..1e7).contains(&magnitude) {
        let sci = format!("{x:e}");
        let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
        let mantissa = if mantissa.contains('.') {
            mantissa.to_owned()
        } else {
            format!("{mantissa}.0")
        };
        return format!("{mantissa}E{exponent}");
    }
    let plain = x.to_string();
    if plain.contains('.') {
        plain
    } else {
        format!("{plain}.0")
    }
}

#[cfg(test)]
mod tests {
    use super::java_double;

    #[test]
    fn doubles_print_like_java() {
        for (x, shown) in [
            (-1.0, "-1.0"),
            (-0.01, "-0.01"),
            (1.01, "1.01"),
            (2.0, "2.0"),
            (0.0, "0.0"),
            (1e-4, "1.0E-4"),
            (1.5e10, "1.5E10"),
            (123_456.0, "123456.0"),
            (f64::NAN, "NaN"),
        ] {
            assert_eq!(java_double(x), shown);
        }
    }
}
