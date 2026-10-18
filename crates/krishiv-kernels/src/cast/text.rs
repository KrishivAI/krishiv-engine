// SPDX-License-Identifier: Apache-2.0
//! Spark's string parsers (`UTF8String.toLong`, Java `BigDecimal(String)`,
//! `Double.parseDouble`, `DateTimeUtils.stringToDate`) and renderings
//! (`Double.toString`, `BigDecimal.toPlainString`). Every rule here was
//! checked against Spark 4.2 (JDK 21).
#![expect(
    clippy::arithmetic_side_effects,
    reason = "digit and exponent arithmetic is bounded: digit counts by the input length, \
              exponents saturate"
)]

use arrow_buffer::i256;

use super::datetime::days_from_civil;

/// Strips the bytes `<= 0x20` (ASCII space and controls) from both ends,
/// as Java's `String.trim` does (the decimal and floating-point parsers).
pub(crate) fn trim(s: &[u8]) -> &[u8] {
    trim_by(s, |b| b <= b' ')
}

/// Spark's `UTF8String.trimAll` (the integral and date parsers): also
/// strips DEL (`0x7F`), an ISO control character `String.trim` keeps.
pub(crate) fn trim_all(s: &[u8]) -> &[u8] {
    trim_by(s, |b| b <= b' ' || b == 0x7F)
}

fn trim_by(s: &[u8], strip: impl Fn(u8) -> bool) -> &[u8] {
    let start = s.iter().position(|&b| !strip(b)).unwrap_or(s.len());
    let end = s.iter().rposition(|&b| !strip(b)).map_or(start, |e| e + 1);
    s.get(start..end).unwrap_or_default()
}

/// `UTF8String.toLong` (`exact == false`, legacy cast: a fraction after
/// `.` is dropped) or `toLongExact` (`exact`, ANSI cast: digits only), on
/// trimmed input. `None` is malformed or out of `i64` range.
pub(crate) fn parse_integral(s: &[u8], exact: bool) -> Option<i64> {
    let s = trim_all(s);
    let (neg, digits) = match s.split_first() {
        Some((b'-', rest)) => (true, rest),
        Some((b'+', rest)) => (false, rest),
        Some(_) => (false, s),
        None => return None,
    };
    if digits.is_empty() {
        return None;
    }
    let (int, frac) = match digits.iter().position(|&b| b == b'.') {
        Some(_) if exact => return None,
        Some(dot) => (digits.get(..dot)?, digits.get(dot + 1..)?),
        None => (digits, &[][..]),
    };
    if !frac.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut v: i128 = 0;
    for &b in int {
        if !b.is_ascii_digit() {
            return None;
        }
        v = v * 10 + i128::from(b - b'0');
        if v > i128::from(i64::MAX) + 1 {
            return None;
        }
    }
    i64::try_from(if neg { -v } else { v }).ok()
}

/// The zero of every Unicode `Nd` run Java 21 knows (`Character.digit(c,
/// 10) == 0`); each run is ten consecutive digits. Java's `BigDecimal`
/// accepts them all.
const DIGIT_ZEROS: [u32; 68] = [
    0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66,
    0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90,
    0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10,
    0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650,
    0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0,
    0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950,
    0x1fbf0,
];

/// Java's `Character.digit(c, 10)`.
fn java_digit(c: char) -> Option<u8> {
    let c = u32::from(c);
    let i = DIGIT_ZEROS.partition_point(|&z| z <= c).checked_sub(1)?;
    let d = c - DIGIT_ZEROS.get(i)?;
    u8::try_from(d).ok().filter(|&d| d < 10)
}

/// A parsed decimal: `digits × 10^exp` (digits without leading zeros;
/// empty for zero).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BigDec {
    pub(crate) negative: bool,
    pub(crate) digits: Vec<u8>,
    /// Java's `-scale`.
    pub(crate) exp: i64,
}

impl BigDec {
    /// Spark's `numDigitsInIntegralPart`: `precision - scale` (precision 1
    /// for zero).
    pub(crate) fn integral_digits(&self) -> i64 {
        i64::try_from(self.digits.len().max(1))
            .unwrap_or(i64::MAX)
            .saturating_add(self.exp)
    }

    /// The value rounded `HALF_UP` to `scale` decimals as an unscaled
    /// integer, or `None` when it needs more than 76 digits.
    pub(crate) fn to_scale(&self, scale: u8) -> Option<i256> {
        let shift = self.exp.saturating_add(i64::from(scale));
        let ten = i256::from_i128(10);
        let mut v = i256::ZERO;
        let (keep, round) = if shift >= 0 {
            (self.digits.len(), false)
        } else {
            let drop = usize::try_from(shift.unsigned_abs()).unwrap_or(usize::MAX);
            let keep = self.digits.len().saturating_sub(drop);
            let first_dropped = if drop <= self.digits.len() {
                self.digits.get(keep)
            } else {
                None
            };
            (keep, first_dropped.is_some_and(|&d| d >= 5))
        };
        if keep.saturating_add(usize::try_from(shift.max(0)).unwrap_or(usize::MAX)) > 76 {
            return None;
        }
        for &d in self.digits.get(..keep)? {
            v = v
                .wrapping_mul(ten)
                .wrapping_add(i256::from_i128(i128::from(d)));
        }
        for _ in 0..shift.max(0) {
            v = v.wrapping_mul(ten);
        }
        if round {
            v = v.wrapping_add(i256::ONE);
        }
        Some(if self.negative { v.wrapping_neg() } else { v })
    }

    /// `BigDecimal.toPlainString`.
    pub(crate) fn plain(&self) -> String {
        let digits: String = if self.digits.is_empty() {
            "0".to_owned()
        } else {
            self.digits.iter().map(|&d| char::from(b'0' + d)).collect()
        };
        let sign = if self.negative && !self.digits.is_empty() {
            "-"
        } else {
            ""
        };
        if self.exp >= 0 {
            let zeros = if self.digits.is_empty() { 0 } else { self.exp };
            return format!(
                "{sign}{digits}{}",
                "0".repeat(usize::try_from(zeros).unwrap_or(0))
            );
        }
        let scale = usize::try_from(self.exp.unsigned_abs()).unwrap_or(usize::MAX);
        if digits.len() > scale {
            let (int, frac) = digits.split_at(digits.len() - scale);
            format!("{sign}{int}.{frac}")
        } else {
            format!("{sign}0.{}{digits}", "0".repeat(scale - digits.len()))
        }
    }
}

/// The common plain form `[+-]digits[.digits]` (ASCII, at most 38 digits)
/// at `scale`, rounded `HALF_UP`: the same value as [`parse_decimal`] and
/// [`BigDec::to_scale`], without allocating. `None` means "use the general
/// parser" (not "malformed").
pub(crate) fn parse_decimal_fast(s: &[u8], scale: u8) -> Option<i128> {
    let s = trim(s);
    let (neg, body) = match s.split_first() {
        Some((b'-', r)) => (true, r),
        Some((b'+', r)) => (false, r),
        _ => (false, s),
    };
    let (int, frac) = match body.iter().position(|&b| b == b'.') {
        Some(dot) => (body.get(..dot)?, body.get(dot + 1..)?),
        None => (body, &[][..]),
    };
    if int.len() + frac.len() == 0 || int.len() + frac.len() > 38 {
        return None;
    }
    let keep = frac.len().min(usize::from(scale));
    let mut v: i128 = 0;
    for &b in int.iter().chain(frac.get(..keep)?) {
        if !b.is_ascii_digit() {
            return None;
        }
        v = v * 10 + i128::from(b - b'0');
    }
    let rest = frac.get(keep..)?;
    if !rest.iter().all(u8::is_ascii_digit) {
        return None;
    }
    if rest.first().is_some_and(|&d| d >= b'5') {
        v += 1;
    }
    for _ in keep..usize::from(scale) {
        v = v.checked_mul(10)?;
    }
    Some(if neg { -v } else { v })
}

/// Java's `new BigDecimal(s.trim())`; `None` where it throws.
pub(crate) fn parse_decimal(s: &str) -> Option<BigDec> {
    let s = std::str::from_utf8(trim(s.as_bytes())).ok()?;
    let (negative, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let mut digits = Vec::new();
    let (mut seen_digit, mut seen_dot, mut frac) = (false, false, 0_i64);
    let mut chars = rest.chars().peekable();
    let mut exp = 0_i64;
    while let Some(c) = chars.next() {
        if let Some(d) = java_digit(c) {
            seen_digit = true;
            if !digits.is_empty() || d != 0 {
                digits.push(d);
            }
            if seen_dot {
                frac += 1;
            }
        } else if c == '.' && !seen_dot {
            seen_dot = true;
        } else if (c == 'e' || c == 'E') && seen_digit {
            let neg = match chars.peek() {
                Some('-') => {
                    chars.next();
                    true
                }
                Some('+') => {
                    chars.next();
                    false
                }
                _ => false,
            };
            let mut e: i64 = 0;
            let mut any = false;
            for c in chars.by_ref() {
                e = e * 10 + i64::from(java_digit(c)?);
                any = true;
                if e > i64::from(i32::MAX) + 1 {
                    return None;
                }
            }
            if !any || (!neg && e > i64::from(i32::MAX)) {
                return None;
            }
            exp = if neg { -e } else { e };
        } else {
            return None;
        }
    }
    if !seen_digit {
        return None;
    }
    // Java's scale (`frac - exp`) must fit an int.
    let scale = frac - exp;
    if i32::try_from(scale).is_err() {
        return None;
    }
    Some(BigDec {
        negative,
        digits,
        exp: -scale,
    })
}

/// Java's `Double.parseDouble` (`F = f64`) or `Float.parseFloat` (`f32`) on
/// trimmed input, then Spark's special literals (`inf`, `-infinity`,
/// `nan`, ... in any case). `None` is malformed.
pub(crate) fn parse_float<F: std::str::FromStr + HexFloat>(s: &[u8]) -> Option<F> {
    let s = std::str::from_utf8(trim(s)).ok()?;
    let (sign, body) = match s.strip_prefix('-') {
        Some(b) => ("-", b),
        None => ("", s.strip_prefix('+').unwrap_or(s)),
    };
    if body == "NaN" || body == "Infinity" {
        return format!("{sign}{}", if body == "NaN" { "NaN" } else { "inf" })
            .parse()
            .ok();
    }
    if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        return F::from_hex(sign == "-", hex);
    }
    // A Java float literal: digits, an optional fraction, an optional
    // exponent, an optional type suffix. Rust parses the rest exactly.
    let core = body.strip_suffix(['d', 'D', 'f', 'F']).unwrap_or(body);
    let valid = !core.is_empty()
        && core
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'))
        && core
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_digit() || b == b'.');
    if valid && let Ok(v) = format!("{sign}{core}").parse::<F>() {
        return Some(v);
    }
    match s.to_ascii_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => "inf".parse().ok(),
        "-inf" | "-infinity" => "-inf".parse().ok(),
        "nan" => "NaN".parse().ok(),
        _ => None,
    }
}

/// Java hexadecimal floating-point literals (`0x1.8p1`).
pub(crate) trait HexFloat: Sized {
    /// The value of `[sign] 0x hex`, or `None` if malformed.
    fn from_hex(negative: bool, hex: &str) -> Option<Self>;
}

/// Parses `h[.h]p[+-]d[suffix]` into a mantissa (up to 120 bits plus a
/// sticky bit) and a binary exponent.
fn hex_parts(hex: &str) -> Option<(u128, i64, bool)> {
    let hex = hex.strip_suffix(['d', 'D', 'f', 'F']).unwrap_or(hex);
    let (mant, exp) = hex.split_once(['p', 'P'])?;
    let exp: i64 = {
        let digits = exp.strip_prefix(['+', '-']).unwrap_or(exp);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        // Any |exponent| beyond the float range behaves the same; the
        // bound keeps every later exponent sum small.
        let v = digits.bytes().fold(0_i64, |v, b| {
            v.saturating_mul(10)
                .saturating_add(i64::from(b - b'0'))
                .min(1_000_000)
        });
        if exp.starts_with('-') { -v } else { v }
    };
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    if int.is_empty() && frac.is_empty() {
        return None;
    }
    let (mut m, mut e, mut sticky) = (0_u128, exp, false);
    for (i, c) in int.chars().chain(frac.chars()).enumerate() {
        let d = u128::from(c.to_digit(16)?);
        if m >> 120 == 0 {
            m = (m << 4) | d;
            if i >= int.len() {
                e = e.saturating_sub(4);
            }
        } else {
            sticky |= d != 0;
            if i < int.len() {
                e = e.saturating_add(4);
            }
        }
    }
    Some((m, e, sticky))
}

/// `m × 2^e` (plus a sticky bit below `m`) rounded half-even to a float
/// with `bits` significand bits and minimum normal exponent `emin`.
fn compose(m: u128, e: i64, sticky: bool, bits: u32, emin: i64, emax: i64) -> (u128, i64, bool) {
    if m == 0 {
        return (0, 0, false);
    }
    let n = i64::from(128 - m.leading_zeros());
    let top = n - 1 + e;
    // Significand bits to keep: fewer for subnormals.
    let keep = if top >= emin {
        i64::from(bits)
    } else {
        i64::from(bits) - (emin - top)
    };
    let drop = n - keep;
    let (mut q, mut exp) = if drop <= 0 {
        (m << u32::try_from(-drop).unwrap_or(0), e + drop)
    } else if drop >= 128 {
        // Below half the smallest subnormal unless m is at least 2^127.
        let half = 1_u128 << 127;
        let up = drop == 128 && (m > half || (m == half && sticky));
        (u128::from(up), e + drop)
    } else {
        let d = u32::try_from(drop).unwrap_or(0);
        let rem = m & ((1_u128 << d) - 1);
        let half = 1_u128 << (d - 1);
        let mut q = m >> d;
        if rem > half || (rem == half && (sticky || q & 1 == 1)) {
            q += 1;
        }
        (q, e + drop)
    };
    if q >> bits != 0 {
        q >>= 1;
        exp += 1;
    }
    let overflow = q != 0 && i64::from(128 - q.leading_zeros()) - 1 + exp > emax;
    (q, exp, overflow)
}

/// Exact powers of two by construction.
trait Pow2 {
    /// `2^e` for `e` in the normal exponent range.
    fn pow2(e: i32) -> Self;
}

impl Pow2 for f64 {
    fn pow2(e: i32) -> Self {
        let biased = u64::try_from(e.clamp(-1022, 1023) + 1023).unwrap_or(1023);
        Self::from_bits(biased << 52)
    }
}

impl Pow2 for f32 {
    fn pow2(e: i32) -> Self {
        let biased = u32::try_from(e.clamp(-126, 127) + 127).unwrap_or(127);
        Self::from_bits(biased << 23)
    }
}

macro_rules! hex_float {
    ($t:ty, $bits:expr, $emin:expr, $emax:expr) => {
        impl HexFloat for $t {
            fn from_hex(negative: bool, hex: &str) -> Option<Self> {
                let (m, e, sticky) = hex_parts(hex)?;
                let (q, exp, overflow) = compose(m, e, sticky, $bits, $emin, $emax);
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "q has at most $bits bits: exact"
                )]
                let mut v = if overflow {
                    <$t>::INFINITY
                } else {
                    // q < 2^bits is exact; scale in two exact steps (each
                    // power of two is normal; `powi` is not guaranteed exact).
                    let e = i32::try_from(exp).unwrap_or(0);
                    let half = e / 2;
                    (q as $t) * <$t>::pow2(half) * <$t>::pow2(e - half)
                };
                if negative {
                    v = -v;
                }
                Some(v)
            }
        }
    };
}
hex_float!(f64, 53, -1022, 1023);
hex_float!(f32, 24, -126, 127);

/// `DateTimeUtils.stringToDate` on trimmed input: `[+-]yyyy[y{0,3}]`,
/// `-[m]m`, `-[d]d`, then anything after a ` ` or `T`. `None` is malformed
/// or outside the `date` range.
pub(crate) fn parse_date(s: &[u8]) -> Option<i32> {
    let s = trim_all(s);
    let (sign, mut rest) = match s.split_first() {
        Some((b'-', r)) => (-1_i64, r),
        Some((b'+', r)) => (1, r),
        _ => (1, s),
    };
    let mut segments = [1_i64; 3];
    let mut i = 0;
    let (mut value, mut count) = (0_i64, 0_u32);
    let valid = |i: usize, n: u32| {
        if i == 0 {
            (4..=7).contains(&n)
        } else {
            (1..=2).contains(&n)
        }
    };
    while let Some((&b, tail)) = rest.split_first() {
        if b == b' ' || b == b'T' {
            break;
        }
        if i < 2 && b == b'-' {
            if !valid(i, count) {
                return None;
            }
            *segments.get_mut(i)? = value;
            (value, count) = (0, 0);
            i += 1;
        } else if b.is_ascii_digit() {
            // A segment longer than its limit is malformed; stopping here
            // also keeps the value far from overflow.
            if count >= if i == 0 { 7 } else { 2 } {
                return None;
            }
            value = value * 10 + i64::from(b - b'0');
            count += 1;
        } else {
            return None;
        }
        rest = tail;
    }
    if !valid(i, count) || (i < 2 && !rest.is_empty()) {
        return None;
    }
    *segments.get_mut(i)? = value;
    let [y, m, d] = segments;
    let days = days_from_civil(sign * y, u32::try_from(m).ok()?, u32::try_from(d).ok()?)?;
    i32::try_from(days).ok()
}

/// Of the two shortest decimals around a tie — the truncation `head` and
/// `head + 1` in the last digit — the one with an even last digit, given the
/// odd one `odd` a formatter picked; `true` when rounding up carried into a
/// new leading digit.
fn even_neighbour(head: &str, odd: &str) -> (String, bool) {
    if head != odd {
        return (head.to_owned(), false);
    }
    let mut up: Vec<u8> = head.bytes().collect();
    for d in up.iter_mut().rev() {
        if *d == b'9' {
            *d = b'0';
        } else {
            *d += 1;
            return (String::from_utf8(up).unwrap_or_default(), false);
        }
    }
    up.insert(0, b'1');
    (String::from_utf8(up).unwrap_or_default(), true)
}

/// Java's `Double.toString` / `Float.toString` (JDK 19+: shortest
/// digits): plain for `1e-3 <= |v| < 1e7`, else `d.dddE[-]n`.
pub(crate) fn java_float_string<F: std::fmt::LowerExp + Copy>(
    v: F,
    nan: bool,
    inf: Option<bool>,
) -> String {
    if nan {
        return "NaN".to_owned();
    }
    if let Some(neg) = inf {
        return if neg { "-Infinity" } else { "Infinity" }.to_owned();
    }
    // Java picks the closest decimal of at least two digits: when the
    // shortest has one (5e-324), the correctly rounded two-digit one
    // (4.9e-324) is at least as close, so it also round-trips.
    let mut e = format!("{v:e}");
    if e.split_once('e')
        .is_some_and(|(m, _)| m.trim_start_matches('-').len() == 1)
    {
        e = format!("{v:.1e}");
    }
    let (mant, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let exp: i64 = exp.parse().unwrap_or(0);
    let (sign, mant) = mant.strip_prefix('-').map_or(("", mant), |m| ("-", m));
    let mut digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    if digits.bytes().all(|b| b == b'0') {
        return format!("{sign}0.0");
    }
    let mut exp = exp;
    // Of two equally close shortest decimals Java takes the even one; the
    // shortest-digits formatter may pick the odd one. A tie means the exact
    // value ends in a single 5 right after the last digit.
    if digits.bytes().last().is_some_and(|d| d % 2 == 1) {
        let exact = format!("{v:.766e}");
        let exact: String = exact
            .split('e')
            .next()
            .unwrap_or_default()
            .chars()
            .filter(char::is_ascii_digit)
            .collect();
        let n = digits.len();
        let (head, tail) = exact.split_at(n.min(exact.len()));
        if tail.starts_with('5') && tail.bytes().skip(1).all(|b| b == b'0') {
            let (even, carry) = even_neighbour(head, &digits);
            digits = even;
            exp += i64::from(carry);
        }
    }
    let digits = digits.trim_end_matches('0');
    if (-3..7).contains(&exp) {
        let point = exp + 1;
        if point <= 0 {
            let zeros = "0".repeat(usize::try_from(-point).unwrap_or(0));
            return format!("{sign}0.{zeros}{digits}");
        }
        let p = usize::try_from(point).unwrap_or(0);
        let int: String = digits
            .chars()
            .chain(std::iter::repeat('0'))
            .take(p)
            .collect();
        let frac = digits.get(p..).filter(|f| !f.is_empty()).unwrap_or("0");
        return format!("{sign}{int}.{frac}");
    }
    let (first, rest) = digits.split_at(1);
    format!(
        "{sign}{first}.{}E{exp}",
        if rest.is_empty() { "0" } else { rest }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_strings_match_spark() {
        let ansi = |s: &str| parse_integral(s.as_bytes(), true);
        let legacy = |s: &str| parse_integral(s.as_bytes(), false);
        assert_eq!(ansi(" 12 "), Some(12));
        assert_eq!(ansi("\t12\n"), Some(12));
        assert_eq!(ansi("\u{1}12\u{0}"), Some(12));
        assert_eq!(ansi("+12"), Some(12));
        assert_eq!(ansi("-0"), Some(0));
        assert_eq!(ansi("0012"), Some(12));
        assert_eq!(ansi("-00000000000000000000000001"), Some(-1));
        assert_eq!(ansi("-9223372036854775808"), Some(i64::MIN));
        assert_eq!(ansi("99999999999999999999"), None);
        for bad in [
            "12.0",
            "12.7",
            "",
            "-",
            "+",
            "\u{3000}12",
            "12\u{a0}",
            "1e3",
            ".5",
            "9223372036854775808",
        ] {
            assert_eq!(ansi(bad), None, "{bad:?}");
        }
        assert_eq!(legacy("12.7"), Some(12));
        assert_eq!(legacy("-12.7"), Some(-12));
        assert_eq!(legacy("12."), Some(12));
        assert_eq!(legacy(".5"), Some(0));
        assert_eq!(legacy("."), Some(0));
        assert_eq!(legacy("-.5"), Some(0));
        assert_eq!(legacy("1. "), Some(1));
        for bad in ["1e3", "-", "+", "1.a", " 1 .5", "\u{661}\u{662}", "1.2.3"] {
            assert_eq!(legacy(bad), None, "{bad:?}");
        }
    }

    fn dec(s: &str, scale: u8) -> Option<(String, i128, i64)> {
        let d = parse_decimal(s)?;
        Some((
            d.plain(),
            d.to_scale(scale)?.to_i128()?,
            d.integral_digits(),
        ))
    }

    #[test]
    fn decimal_strings_match_java() {
        assert_eq!(dec("1e3", 2), Some(("1000".into(), 100_000, 4)));
        assert_eq!(dec(" 1.235 ", 2), Some(("1.235".into(), 124, 1)));
        assert_eq!(dec("-1.235", 2), Some(("-1.235".into(), -124, 1)));
        assert_eq!(dec(".5", 1), Some(("0.5".into(), 5, 0)));
        assert_eq!(dec("5.", 1), Some(("5".into(), 50, 1)));
        assert_eq!(dec("+1.5", 1), Some(("1.5".into(), 15, 1)));
        assert_eq!(dec("-0.005", 2), Some(("-0.005".into(), -1, -2)));
        assert_eq!(dec("1.e2", 1), Some(("100".into(), 1000, 3)));
        assert_eq!(dec("1E+2", 1), Some(("100".into(), 1000, 3)));
        assert_eq!(dec("1.5e-3", 4), Some(("0.0015".into(), 15, -2)));
        assert_eq!(dec("007.50", 2), Some(("7.50".into(), 750, 1)));
        assert_eq!(dec("-0.0", 1), Some(("0.0".into(), 0, 0)));
        assert_eq!(dec("\u{661}", 1), Some(("1".into(), 10, 1)));
        assert_eq!(dec("1e-400", 1).map(|d| d.1), Some(0));
        let long = format!("1.{}", "0".repeat(44));
        assert_eq!(dec(&long, 0).map(|d| d.1), Some(1));
        let p = parse_decimal("123456789012345678901234567890123456789e-10").unwrap();
        assert_eq!(p.integral_digits(), 29);
        assert_eq!(
            p.to_scale(0).unwrap().to_string(),
            "12345678901234567890123456789"
        );
        assert_eq!(parse_decimal("1e400").unwrap().integral_digits(), 401);
        assert_eq!(parse_decimal("1e400").unwrap().to_scale(0), None);
        for bad in [
            "1d",
            "abc",
            "1.2.3",
            "e5",
            "1e",
            "Infinity",
            "",
            ".",
            "-",
            "1_000",
            "0x10",
            "1,5",
            "1e2147483648",
        ] {
            assert_eq!(parse_decimal(bad), None, "{bad:?}");
        }
        assert!(parse_decimal("1e-2147483648").is_none());
        assert!(parse_decimal("1e99999999999").is_none());
    }

    #[test]
    fn fast_decimal_path_agrees() {
        for t in [
            "1.235",
            "-1.235",
            " 12 ",
            ".5",
            "5.",
            "+0.005",
            "99999999999999999999.999999999999999999",
            "1",
            "-0",
        ] {
            for scale in [0, 2, 3, 10] {
                let general = parse_decimal(t)
                    .and_then(|d| d.to_scale(scale))
                    .and_then(arrow_buffer::i256::to_i128);
                assert_eq!(
                    parse_decimal_fast(t.as_bytes(), scale),
                    general,
                    "{t} {scale}"
                );
            }
        }
        for t in ["", ".", "-", "1e3", "1.2.3", "\u{661}", "x", "1.23x"] {
            assert_eq!(parse_decimal_fast(t.as_bytes(), 2), None, "{t}");
        }
    }

    #[test]
    fn float_strings_match_spark() {
        let f = |s: &str| parse_float::<f64>(s.as_bytes());
        for (s, v) in [
            ("  1.5  ", 1.5),
            ("1.5d", 1.5),
            ("1.5F", 1.5),
            ("+.5", 0.5),
            ("5.", 5.0),
            ("1.5e+3", 1500.0),
            ("1.5E-3D", 0.0015),
            ("0x1p3", 8.0),
            ("0x1.8p1", 3.0),
            ("-0x.8P-1", -0.25),
            ("1e400", f64::INFINITY),
            ("1e-400", 0.0),
            ("inf", f64::INFINITY),
            ("INF", f64::INFINITY),
            (" -inf ", f64::NEG_INFINITY),
            ("+infinity", f64::INFINITY),
            ("-Infinity", f64::NEG_INFINITY),
        ] {
            assert_eq!(f(s), Some(v), "{s:?}");
        }
        assert!(f("nan").is_some_and(f64::is_nan) && f("+NaN").is_some_and(f64::is_nan));
        assert_eq!(f("-0").map(f64::to_bits), Some((-0.0_f64).to_bits()));
        for bad in [
            ".",
            "e5",
            "1e",
            "\u{ff11}",
            "NaNd",
            "+nan",
            "-nan",
            "Infinityx",
            "1_0",
            "0x1",
            "0xp1",
            "abc",
        ] {
            assert_eq!(f(bad), None, "{bad:?}");
        }
        assert_eq!(parse_float::<f32>(b"3.4028236e38"), Some(f32::INFINITY));
        assert_eq!(parse_float::<f32>(b"0x1p-149"), Some(f32::from_bits(1)));
        assert_eq!(parse_float::<f64>(b"0x1p-1074"), Some(f64::from_bits(1)));
        assert_eq!(parse_float::<f64>(b"0x1p1024"), Some(f64::INFINITY));
        assert_eq!(parse_float::<f64>(b"0x1.00000000000008p0"), Some(1.0));
        assert_eq!(
            parse_float::<f64>(b"0x1.00000000000018p0"),
            Some(1.0 + 2.0 * f64::EPSILON)
        );
        assert_eq!(parse_float::<f64>(b"0x0p0"), Some(0.0));
        assert_eq!(parse_float::<f64>(b"0x1p"), None);
        assert_eq!(
            parse_float::<f64>(b"0x10p9223372036854775807"),
            Some(f64::INFINITY)
        );
        assert_eq!(parse_float::<f64>(b"0x1p-9223372036854775807"), Some(0.0));
        assert_eq!(parse_float::<f64>(b"0x1p-1200"), Some(0.0));
        assert_eq!(parse_float::<f64>(b"0x1p-1300"), Some(0.0));
        assert_eq!(parse_float::<f64>(b"0x1.fffffffffffff8p0"), Some(2.0));
        // Digits past 120 bits only feed the sticky bit.
        let long = format!("0x1{}1p0", "0".repeat(40));
        assert_eq!(
            parse_float::<f64>(long.as_bytes()),
            Some(f64::from_bits((164 + 1023) << 52))
        );
        let frac = format!("0x1.{}1p0", "0".repeat(40));
        assert_eq!(parse_float::<f64>(frac.as_bytes()), Some(1.0));
    }

    #[test]
    fn dates_match_spark() {
        let d = |s: &str| parse_date(s.as_bytes());
        for (s, v) in [
            (" 2024-02-29 ", 19_782),
            ("2024-2-9", 19_762),
            ("2024", 19_723),
            ("2024-02", 19_754),
            ("2024-02-29T10:00", 19_782),
            ("2024-02-29 garbage", 19_782),
            ("2024-02-29T", 19_782),
            ("+2024-02-29", 19_782),
            ("02024-02-29", 19_782),
            ("-0001-01-01", -719_893),
            ("0000-01-01", -719_528),
            ("5881580-07-11", i32::MAX),
            ("-5877641-06-23", i32::MIN),
            ("\u{b}2024-02-29\u{0}", 19_782),
            ("12345-01-01", 3_789_391),
            ("-2024", -1_458_779),
        ] {
            assert_eq!(d(s), Some(v), "{s:?}");
        }
        for bad in [
            "2024-02-30",
            "2024-13-01",
            "24-02-01",
            "2024-02-29garbage",
            "2024/02/29",
            "5881580-07-12",
            "-5877641-06-22",
            "2024-001-01",
            "2024-1-001",
            "2024-02-",
            "2024-",
            "2024-02-29\tx",
            "\u{3000}2024-02-29",
            "12345678-01-01",
            "99999999999999999999-01-01",
            "2024-123",
            "2024-01-99999999999999999999",
            "123",
            "1.5",
            "2024 x",
            "2024-02 x",
            "",
        ] {
            assert_eq!(d(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn tie_breaks_to_the_even_neighbour() {
        assert_eq!(even_neighbour("1219", "1219"), ("1220".to_owned(), false));
        assert_eq!(even_neighbour("1218", "1219"), ("1218".to_owned(), false));
        assert_eq!(even_neighbour("99", "99"), ("100".to_owned(), true));
    }

    #[test]
    fn float_renderings_match_java() {
        let s = |v: f64| java_float_string(v, v.is_nan(), v.is_infinite().then_some(v < 0.0));
        for (v, t) in [
            (1.5, "1.5"),
            (1e7, "1.0E7"),
            (1e-3, "0.001"),
            (9.999_999e-4, "9.999999E-4"),
            (123_456_789.0, "1.23456789E8"),
            (-0.0, "-0.0"),
            (1e21, "1.0E21"),
            (1e-10, "1.0E-10"),
            (9_999_999.0, "9999999.0"),
            (100.0, "100.0"),
            (4.9e-324, "4.9E-324"),
            (f64::NAN, "NaN"),
            (f64::NEG_INFINITY, "-Infinity"),
            (1_709_210_096.123_456, "1.709210096123456E9"),
        ] {
            assert_eq!(s(v), t);
        }
        assert_eq!(java_float_string(1.1_f32, false, None), "1.1");
        // Ties between two shortest decimals go to the even digit (checked
        // against JDK 21 on 300,000 random values).
        assert_eq!(
            s(f64::from_bits(0x4311_5417_2198_6fdd)),
            "1.2193832321750952E15"
        );
        assert_eq!(
            s(f64::from_bits(0xc30a_3881_8007_86aa)),
            "-9.225597805488212E14"
        );
        let f = f32::from_bits(0xca05_c3f5);
        assert_eq!(java_float_string(f, false, None), "-2191613.2");
        assert_eq!(java_float_string(3.4e38_f32, false, None), "3.4E38");
    }
}
