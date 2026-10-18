// SPDX-License-Identifier: Apache-2.0
//! Exact 256-bit evaluation for decimal plans the 128/64-bit lanes cannot
//! take: results that round (Spark's precision-loss adjustment), `/`, `%`
//! and `div`. Every operand is below `10^38`, so exact sums, products and
//! rescaled dividends stay below `10^76 < 2^255`.

use arrow_buffer::i256;

/// `10^k` for `k <= 76`.
const POW10: [i256; 77] = {
    let mut t = [i256::ONE; 77];
    let ten = i256::from_i128(10);
    let mut i = 1;
    while i < 77 {
        t[i] = t[i - 1].wrapping_mul(ten);
        i += 1;
    }
    t
};

/// `10^k`, saturating at `10^76`.
pub(super) fn pow10(k: u32) -> i256 {
    POW10[usize::try_from(k).map_or(76, |k| k.min(76))]
}

/// Whether cutting `q` (`0 <= q < 10^76`) to `digits` significant digits
/// drops a non-zero digit.
pub(super) fn drops_nonzero_digits(q: i256, digits: u32) -> bool {
    // The number of digits of q, when above `digits`.
    let Some(len) = (digits.saturating_add(1)..=76).find(|&k| q < pow10(k)) else {
        return false;
    };
    q >= pow10(digits) && q.wrapping_rem(pow10(len.saturating_sub(digits))) != i256::ZERO
}

/// `v / 10^drop`, rounded half away from zero (Java's `HALF_UP`).
pub(super) fn round_half_up(v: i256, drop: u32) -> i256 {
    if drop == 0 {
        return v;
    }
    if drop > 76 {
        // |v| < 10^76: the quotient rounds to zero.
        return i256::ZERO;
    }
    let d = pow10(drop);
    half_up_div(v, d)
}

/// `n / d` (`d != 0`), rounded half away from zero. `|n|, |d| < 10^77`.
pub(super) fn half_up_div(n: i256, d: i256) -> i256 {
    let (q, r) = (n.wrapping_div(d), n.wrapping_rem(d));
    // |r| < |d| <= 10^76, so 2|r| < 2^256 as an unsigned value: compare
    // unsigned to stay exact.
    let twice = r.wrapping_abs().wrapping_mul(i256::from_i128(2));
    if !unsigned_lt(twice, d.wrapping_abs()) {
        let away = if n.is_negative() == d.is_negative() {
            i256::ONE
        } else {
            i256::MINUS_ONE
        };
        q.wrapping_add(away)
    } else {
        q
    }
}

/// `a < b` for the unsigned values of two 256-bit patterns.
fn unsigned_lt(a: i256, b: i256) -> bool {
    (a ^ i256::MIN) < (b ^ i256::MIN)
}

/// The plain decimal string of `n / d` truncated to `scale` fractional
/// digits (Java `BigDecimal.divide(d, scale, DOWN).toPlainString()`), for
/// `d != 0`, `|n|, |d| < 10^77`.
pub(super) fn quotient_string(n: i256, d: i256, scale: u32) -> String {
    let negative = n.is_negative() != d.is_negative() && n != i256::ZERO;
    let (n, d) = (n.wrapping_abs(), d.wrapping_abs());
    let int = n.wrapping_div(d);
    let mut rem = n.wrapping_rem(d);
    let ten = i256::from_i128(10);
    let mut frac = String::with_capacity(usize::try_from(scale).unwrap_or(0));
    let mut any = int != i256::ZERO;
    for _ in 0..scale {
        // rem < d < 10^77, so 10 * rem < 2^256: exact as unsigned.
        rem = rem.wrapping_mul(ten);
        let mut digit = 0_u8;
        while !unsigned_lt(rem, d) {
            rem = rem.wrapping_sub(d);
            digit = digit.wrapping_add(1);
        }
        any |= digit != 0;
        frac.push(char::from(b'0'.wrapping_add(digit)));
    }
    let sign = if negative && any { "-" } else { "" };
    if scale == 0 {
        format!("{sign}{int}")
    } else {
        format!("{sign}{int}.{frac}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: i128) -> i256 {
        i256::from_i128(x)
    }

    #[test]
    fn half_up_rounds_away_from_zero() {
        assert_eq!(round_half_up(v(25), 1), v(3));
        assert_eq!(round_half_up(v(-25), 1), v(-3));
        assert_eq!(round_half_up(v(24), 1), v(2));
        assert_eq!(round_half_up(v(-249), 2), v(-2));
        assert_eq!(round_half_up(v(7), 0), v(7));
        assert_eq!(round_half_up(v(7), 80), i256::ZERO);
        assert_eq!(half_up_div(v(2), v(-3)), v(-1));
        assert_eq!(half_up_div(v(-1), v(2)), v(-1));
        assert_eq!(half_up_div(v(1), v(3)), v(0));
        // Near the top of the range the doubled remainder needs all 256 bits.
        let big = pow10(76);
        let r = half_up_div(big.wrapping_sub(i256::ONE), big);
        assert_eq!(r, i256::ONE);
        assert_eq!(pow10(80), pow10(76));
    }

    #[test]
    fn quotient_strings() {
        assert_eq!(quotient_string(v(2), v(3), 5), "0.66666");
        assert_eq!(quotient_string(v(-2), v(3), 3), "-0.666");
        assert_eq!(quotient_string(v(1), v(-8), 4), "-0.1250");
        assert_eq!(quotient_string(v(-1), v(3000), 2), "0.00");
        assert_eq!(quotient_string(v(10), v(4), 0), "2");
        let big = pow10(76).wrapping_sub(i256::ONE);
        let s = quotient_string(big, big.wrapping_sub(i256::ONE), 39);
        assert!(s.starts_with("1.000000000000000000000000000000000000000"));
    }
}
