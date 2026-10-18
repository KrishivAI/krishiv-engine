// SPDX-License-Identifier: Apache-2.0
//! Spark's decimal result types (`DecimalPrecision` / the arithmetic
//! expressions' `resultDecimalType`), with and without
//! `spark.sql.decimalOperations.allowPrecisionLoss` (default `true`).
//! Checked against Spark 4.2's `typeof` for both settings.

#![expect(
    clippy::arithmetic_side_effects,
    reason = "precision/scale arithmetic on values in 0..=77 cannot overflow an i32"
)]

use super::{DecimalOp, DecimalType, MAX_PRECISION};
use crate::error::KernelError;

/// Spark's `DecimalType.MINIMUM_ADJUSTED_SCALE`.
const MIN_ADJUSTED_SCALE: i32 = 6;
const MAX: i32 = MAX_PRECISION as i32;

/// `DecimalType.adjustPrecisionScale`: keep the integral digits and cut
/// the scale (not below 6) when the precision exceeds 38.
fn adjust(p: i32, s: i32) -> (i32, i32) {
    if p <= MAX {
        return (p, s);
    }
    let int_digits = p.saturating_sub(s);
    let min_scale = s.min(MIN_ADJUSTED_SCALE);
    (MAX, MAX.saturating_sub(int_digits).max(min_scale))
}

/// `DecimalType.bounded`: clamp both to 38.
fn bounded(p: i32, s: i32) -> (i32, i32) {
    (p.min(MAX), s.min(MAX))
}

/// The result type of `left op right` (`div` is `bigint` and has none).
///
/// # Errors
/// `INTERNAL_ERROR` for an invalid input type or `op` = `div`;
/// `KRISHIV_UNSUPPORTED_FEATURE` for a negative scale.
pub fn result_type(
    op: DecimalOp,
    left: DecimalType,
    right: DecimalType,
    allow_precision_loss: bool,
) -> Result<DecimalType, KernelError> {
    let (p1, s1) = (i32::from(left.0), i32::from(super::check_type(left)?));
    let (p2, s2) = (i32::from(right.0), i32::from(super::check_type(right)?));
    let fit = |(p, s): (i32, i32)| {
        if allow_precision_loss {
            adjust(p, s)
        } else {
            bounded(p, s)
        }
    };
    let (p, s) = match op {
        DecimalOp::Add | DecimalOp::Subtract => {
            let s = s1.max(s2);
            fit(((p1 - s1).max(p2 - s2) + s + 1, s))
        }
        DecimalOp::Multiply => fit((p1 + p2 + 1, s1 + s2)),
        DecimalOp::Divide if allow_precision_loss => {
            let scale = MIN_ADJUSTED_SCALE.max(s1 + p2 + 1);
            adjust(p1 - s1 + s2 + scale, scale)
        }
        DecimalOp::Divide => {
            let mut int_digits = MAX.min(p1 - s1 + s2);
            let mut dec_digits = MAX.min(MIN_ADJUSTED_SCALE.max(s1 + p2 + 1));
            let diff = int_digits + dec_digits - MAX;
            if diff > 0 {
                dec_digits -= diff / 2 + 1;
                int_digits = MAX - dec_digits;
            }
            bounded(int_digits + dec_digits, dec_digits)
        }
        DecimalOp::Remainder => {
            let s = s1.max(s2);
            fit(((p1 - s1).min(p2 - s2) + s, s))
        }
        DecimalOp::IntegralDivide => {
            return Err(KernelError::internal("decimal div has a bigint result"));
        }
    };
    let narrow =
        |v: i32| u8::try_from(v).map_err(|_| KernelError::internal("decimal type out of range"));
    // Spark's precision is at least 1 (`decimal(0, 0)` does not exist).
    let p = narrow(p.max(1))?;
    let s = i8::try_from(narrow(s)?).map_err(|_| KernelError::internal("decimal scale"))?;
    Ok((p, s))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(op, left, right, with loss, without loss)` from Spark 4.2 `typeof`.
    #[test]
    fn spark_result_types() {
        use DecimalOp::{Add, Divide, Multiply, Remainder};
        let cases = [
            (Multiply, (38, 10), (38, 10), (38, 6), (38, 20)),
            (Divide, (38, 10), (38, 10), (38, 6), (38, 18)),
            (Remainder, (38, 10), (5, 2), (13, 10), (13, 10)),
            (Add, (38, 10), (38, 2), (38, 6), (38, 10)),
            (Multiply, (10, 2), (30, 20), (38, 19), (38, 22)),
            (Add, (30, 25), (30, 0), (38, 7), (38, 25)),
            (Divide, (30, 25), (30, 25), (38, 8), (38, 22)),
            (Divide, (5, 2), (30, 25), (38, 10), (38, 21)),
            (Remainder, (38, 38), (38, 0), (38, 38), (38, 38)),
            (Divide, (10, 0), (10, 0), (21, 11), (21, 11)),
            (Divide, (2, 1), (1, 0), (7, 6), (7, 6)),
            (Divide, (38, 37), (1, 0), (38, 37), (38, 37)),
            (Add, (10, 2), (1, 0), (11, 2), (11, 2)),
            (Remainder, (2, 1), (1, 0), (2, 1), (2, 1)),
        ];
        for (op, l, r, loss, no_loss) in cases {
            assert_eq!(
                result_type(op, l, r, true).unwrap(),
                loss,
                "{op:?} {l:?} {r:?}"
            );
            assert_eq!(
                result_type(op, l, r, false).unwrap(),
                no_loss,
                "{op:?} {l:?} {r:?}"
            );
        }
        assert!(result_type(DecimalOp::IntegralDivide, (1, 0), (1, 0), true).is_err());
        assert!(result_type(Add, (1, -1), (1, 0), true).is_err());
    }
}
