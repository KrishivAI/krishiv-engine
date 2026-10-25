// SPDX-License-Identifier: Apache-2.0
//! Spark-compatible literal text for output names and plan printing.
//!
//! The analyzer names unaliased expressions as Spark's `toPrettySQL` does,
//! and the tree printer shows literals as Spark's `Literal.sql` does. Both
//! need Java's number formatting, not Rust's `Debug`:
//!
//! - `Double.toString` / `Float.toString`: the shortest round-trip digits,
//!   with a decimal point, in plain notation for magnitudes in
//!   `[1e-3, 1e7)` and scientific (`E`, no `+`) outside it. Examples from
//!   the Spark 4.2.0 goldens: `1.0`, `1.2`, `1.0E10`, `150000.0`, `0.1`,
//!   `1.2345678901234568E48`, `2.35E10`, `1.2345678E-8`.
//! - `BigDecimal.toString`: plain notation when the adjusted exponent is
//!   `>= -6`, scientific otherwise (`1.230E-26`, `1.2345678E-8`).
//! - `DATE` / `TIMESTAMP` literals with a `+` before years past 9999
//!   (`DATE '+999999-03-18'`, `TIMESTAMP '+99999-03-18 12:03:17'`).
//!
//! The formatters take the physical values Spark stores (doubles, decimal
//! unscaled/scale, days since the epoch, micros since the epoch) and return
//! the text Spark prints. They never fail and never allocate more than a
//! short string.

/// Java's `Double.toString` for a finite `v`.
#[must_use]
pub(crate) fn double(v: f64) -> String {
    java_float(v, false)
}

/// Java's `Float.toString` for a finite `v`.
#[must_use]
pub(crate) fn float(v: f32) -> String {
    java_float(f64::from(v), true)
}

/// Java's `Double.toString` / `Float.toString`.
///
/// `is_float` only decides how the shortest digits are obtained: Rust's
/// `{:e}` for `f64` / `f32` already gives each type's shortest round-trip
/// digits in scientific form, which is what Java formats. The layout rules
/// (plain in `[1e-3, 1e7)`, scientific outside, `E`, a decimal point) are
/// the same for both.
fn java_float(v: f64, is_float: bool) -> String {
    debug_assert!(v.is_finite());
    // Preserve negative zero: `{:e}` keeps the sign (`-0e0`).
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0.0".to_owned()
        } else {
            "0.0".to_owned()
        };
    }
    // `v` holds an `f32` value exactly when `is_float`; format the `f32` so
    // the digits are the float's shortest, not the double's.
    #[expect(clippy::cast_possible_truncation, reason = "v holds an f32 value")]
    let as_f32 = v as f32;
    let shortest = if is_float {
        format!("{as_f32:e}")
    } else {
        format!("{v:e}")
    };
    // Java never prints a single digit: when the shortest round-trip
    // decimal has one digit, it picks the two-digit decimal closest to the
    // value (`Float.MIN_VALUE` is `1.4E-45`, not `1.0E-45`; `4.9E-324` for
    // `Double.MIN_VALUE`). That decimal is Rust's correctly rounded
    // `{:.1e}`; it only differs from `d.0` for values far from a
    // one-digit decimal, which in practice means subnormals.
    let text = if split_scientific(&shortest).0.contains('.') {
        shortest
    } else {
        let two = if is_float {
            format!("{as_f32:.1e}")
        } else {
            format!("{v:.1e}")
        };
        if split_scientific(&two).0.ends_with(".0") {
            shortest
        } else {
            two
        }
    };
    let negative = text.starts_with('-');
    let (mant, exp) = split_scientific(&text);
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-3..7).contains(&exp) {
        plain_with_sign(&mant, exp, negative)
    } else {
        scientific_with_sign(&mant, exp, negative)
    }
}

/// Java's `BigDecimal.toString` for `unscaled * 10^-scale`.
///
/// `scale` is never negative here: the parser folds positive exponents
/// into zeros of the unscaled value, so decimals only carry `scale >= 0`.
/// Trailing zeros of the unscaled value are kept (`1.230E-26`).
#[must_use]
pub(crate) fn decimal(unscaled: i128, scale: u32) -> String {
    let scale_i64 = i64::from(scale);
    if unscaled == 0 {
        if scale == 0 {
            return "0".to_owned();
        }
        // Java's adjusted exponent for zero is `-scale` (`ulength` is 1),
        // so scales above six use scientific notation (`0E-7`).
        let adjusted = -scale_i64;
        if adjusted < -6 {
            return format!("0E{adjusted}");
        }
        let scale_usize = usize::try_from(scale).unwrap_or(0);
        let mut out = String::with_capacity(2 + scale_usize);
        out.push_str("0.");
        for _ in 0..scale_usize {
            out.push('0');
        }
        return out;
    }
    let sign = if unscaled < 0 { "-" } else { "" };
    let digits = unscaled.unsigned_abs().to_string();
    if scale == 0 {
        return format!("{sign}{digits}");
    }
    // The adjusted exponent decides plain vs scientific, as Java does.
    let len = i64::try_from(digits.len()).unwrap_or(0);
    let adjusted = len - scale_i64 - 1;
    if adjusted >= -6 {
        let scale_usize = usize::try_from(scale).unwrap_or(0);
        if digits.len() > scale_usize {
            let (int, frac) = digits.split_at(digits.len() - scale_usize);
            format!("{sign}{int}.{frac}")
        } else {
            let mut out = String::with_capacity(2 + scale_usize);
            out.push_str(sign);
            out.push_str("0.");
            for _ in 0..scale_usize.saturating_sub(digits.len()) {
                out.push('0');
            }
            out.push_str(&digits);
            out
        }
    } else {
        let (first, rest) = digits.split_at(1);
        // Java inserts a point only when more than one digit remains
        // (`1E-7`, not `1.0E-7`; `1.230E-26` keeps its trailing zero).
        let mant = if rest.is_empty() {
            first.to_owned()
        } else {
            format!("{first}.{rest}")
        };
        format!("{sign}{mant}E{adjusted}")
    }
}

/// `DATE 'yyyy-MM-dd'` with a `+` before years past 9999.
#[must_use]
pub(crate) fn date(days: i32) -> String {
    format!("DATE '{}'", date_value(days))
}

/// `yyyy-MM-dd`, as a cast to string renders a date.
#[must_use]
pub(crate) fn date_value(days: i32) -> String {
    let (y, m, d) = civil_from_days(i64::from(days));
    year_month_day(y, m, d)
}

/// `TIMESTAMP 'yyyy-MM-dd HH:mm:ss[.ffffff]'`, or the `_NTZ` spelling.
#[must_use]
pub(crate) fn timestamp(micros: i64, ntz: bool) -> String {
    let keyword = if ntz { "TIMESTAMP_NTZ" } else { "TIMESTAMP" };
    format!("{keyword} '{}'", timestamp_value(micros))
}

/// `yyyy-MM-dd HH:mm:ss[.ffffff]`, as a cast to string renders a
/// timestamp.
#[must_use]
pub(crate) fn timestamp_value(micros: i64) -> String {
    const DAY: i64 = 86_400_000_000;
    let days = micros.div_euclid(DAY);
    let mut rest = micros.rem_euclid(DAY);
    let (y, m, d) = civil_from_days(days);
    let hour = rest / 3_600_000_000;
    rest %= 3_600_000_000;
    let min = rest / 60_000_000;
    rest %= 60_000_000;
    let sec = rest / 1_000_000;
    let frac = rest % 1_000_000;
    let mut out = String::new();
    out.push_str(year_month_day(y, m, d).as_str());
    out.push(' ');
    push2(&mut out, hour);
    out.push(':');
    push2(&mut out, min);
    out.push(':');
    push2(&mut out, sec);
    if frac != 0 {
        let mut digits = format!("{frac:06}");
        while digits.ends_with('0') {
            digits.pop();
        }
        out.push('.');
        out.push_str(&digits);
    }
    out
}

/// `yyyy-MM-dd` with a `+` before years past 9999 (`-` keeps its sign).
fn year_month_day(y: i64, m: i64, d: i64) -> String {
    let sign = if y < 0 {
        "-"
    } else if y > 9999 {
        "+"
    } else {
        ""
    };
    // `{:04}` pads short years; long years print whole.
    format!("{sign}{:04}-{m:02}-{d:02}", y.unsigned_abs())
}

fn push2(out: &mut String, v: i64) {
    use std::fmt::Write as _;
    let _ = write!(out, "{v:02}");
}

/// Proleptic Gregorian `(year, month, day)` of a day count since 1970-01-01
/// (H. Hinnant's `civil_from_days`).
pub(crate) const fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    (y, m, d)
}

/// Splits Rust's `{:e}` output (`-d[.ddd]e[+-]exp`) into mantissa digits
/// (`d[.]ddd`, sign removed) and exponent text.
fn split_scientific(text: &str) -> (String, &str) {
    let text = text.strip_prefix('-').unwrap_or(text);
    let (mant, exp) = text.split_once('e').unwrap_or((text, "0"));
    (mant.to_owned(), exp)
}

/// The sign of Rust's `{:e}` output (`-` kept).
#[cfg(test)]
fn is_negative(text: &str) -> bool {
    text.starts_with('-')
}

/// Plain notation for scientific mantissa `mant` (`d[.]ddd`) with exponent
/// `exp` (one digit before the point): `exp + 1` integer digits.
#[cfg(test)]
fn plain(mant: &str, exp: i32) -> String {
    plain_with_sign(mant, exp, false)
}

/// [`plain`] keeping a `-` from `{:e}` output.
#[cfg(test)]
fn plain_neg(mant: &str, exp: i32) -> String {
    plain_with_sign(mant, exp, true)
}

fn plain_with_sign(mant: &str, exp: i32, negative: bool) -> String {
    let sign = if negative { "-" } else { "" };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let ints = i64::from(exp) + 1;
    let len = i64::try_from(digits.len()).unwrap_or(0);
    if ints <= 0 {
        let zeros = usize::try_from(-ints).unwrap_or(0);
        let mut out = String::with_capacity(digits.len() + 2 + zeros);
        out.push_str(sign);
        out.push_str("0.");
        for _ in 0..zeros {
            out.push('0');
        }
        out.push_str(&digits);
        out
    } else if ints >= len {
        let ints_usize = usize::try_from(ints).unwrap_or(0);
        let mut out = String::with_capacity(ints_usize + 2);
        out.push_str(sign);
        out.push_str(&digits);
        for _ in 0..ints_usize.saturating_sub(digits.len()) {
            out.push('0');
        }
        out.push_str(".0");
        out
    } else {
        let at = usize::try_from(ints).unwrap_or(0);
        let (int, frac) = digits.split_at(at);
        format!("{sign}{int}.{frac}")
    }
}

/// Scientific notation for mantissa `mant` with exponent `exp`.
#[cfg(test)]
fn scientific(mant: &str, exp: i32) -> String {
    scientific_with_sign(mant, exp, false)
}

#[cfg(test)]
fn scientific_neg(mant: &str, exp: i32) -> String {
    scientific_with_sign(mant, exp, true)
}

fn scientific_with_sign(mant: &str, exp: i32, negative: bool) -> String {
    let sign = if negative { "-" } else { "" };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let (first, rest) = digits.split_at(1);
    let mant = if rest.is_empty() {
        format!("{first}.0")
    } else {
        format!("{first}.{rest}")
    };
    format!("{sign}{mant}E{exp}")
}

/// Java formatting from a full `{:e}` string (keeps its sign).
#[cfg(test)]
fn from_scientific(text: &str) -> String {
    let negative = is_negative(text);
    let (mant, exp) = split_scientific(text);
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-3..7).contains(&exp) {
        if negative {
            plain_neg(&mant, exp)
        } else {
            plain(&mant, exp)
        }
    } else if negative {
        scientific_neg(&mant, exp)
    } else {
        scientific(&mant, exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_match_java() {
        // Every double literal of the Spark 4.2.0 goldens, with the name
        // Spark records for it.
        let cases: &[(f64, &str)] = &[
            (1.0, "1.0"),
            (1.2, "1.2"),
            (1e10, "1.0E10"),
            (1.5e5, "150000.0"),
            (0.1, "0.1"),
            (10000.0, "10000.0"),
            (90.0, "90.0"),
            (900e-1, "90.0"),
            (-1.0, "-1.0"),
            (-1.2, "-1.2"),
            (-1e10, "-1.0E10"),
            (-1.5e5, "-150000.0"),
            (-0.1, "-0.1"),
            (-10000.0, "-10000.0"),
            (1.2345678901234568e48, "1.2345678901234568E48"),
            (2.35e10, "2.35E10"),
            (1.2345678e-8, "1.2345678E-8"),
            (3.14e8, "3.14E8"),
            (3.14e-8, "3.14E-8"),
            (-3.14e8, "-3.14E8"),
            (-3.14e-8, "-3.14E-8"),
            (6.8, "6.8"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1e-3, "0.001"),
            (1e-4, "1.0E-4"),
            (1e6, "1000000.0"),
            (1e7, "1.0E7"),
            (9999999.0, "9999999.0"),
            (2.7182818284590455, "2.7182818284590455"),
            (2.1442904349215556e14, "2.1442904349215556E14"),
            (4.663547361468238e-15, "4.663547361468238E-15"),
            (1.8508157176809255, "1.8508157176809255"),
            (1.1883951057781212, "1.1883951057781212"),
            (0.6420926159343306, "0.6420926159343306"),
            (1e23, "1.0E23"),
            (0.001, "0.001"),
            // `Double.MIN_VALUE`: one shortest digit (`5e-324`), so Java
            // prints the closest two-digit decimal, as Spark 4.2 names
            // `SELECT 4.9E-324D`.
            (f64::from_bits(1), "4.9E-324"),
        ];
        for (v, want) in cases {
            assert_eq!(&double(*v), want, "{v}");
        }
    }

    #[test]
    fn floats_match_java() {
        let cases: &[(f32, &str)] = &[
            (1.0, "1.0"),
            (1.2, "1.2"),
            (0.1, "0.1"),
            (-1.0, "-1.0"),
            (-1.2, "-1.2"),
            (-0.1, "-0.1"),
            (6.8, "6.8"),
            (3.4e38, "3.4E38"),
            (1.0e-10, "1.0E-10"),
            // Subnormals whose shortest decimal has one digit print the
            // closest two-digit decimal, as Spark 4.2 names
            // `SELECT 1.4E-45F` and `SELECT 9E-45F`.
            (f32::from_bits(1), "1.4E-45"),
            (f32::from_bits(6), "8.4E-45"),
            (-f32::from_bits(1), "-1.4E-45"),
        ];
        for (v, want) in cases {
            assert_eq!(&float(*v), want, "{v}");
        }
        // `3.4028235E39f` overflows to infinity (out of range literal).
        assert!("3.4028235e39".parse::<f32>().unwrap().is_infinite());
    }

    #[test]
    fn decimals_match_java_big_decimal_to_string() {
        // `(unscaled, scale, want)` from the goldens.
        let cases: &[(i128, u32, &str)] = &[
            (
                90912830918230182310293801923652346786,
                0,
                "90912830918230182310293801923652346786",
            ),
            (1230, 29, "1.230E-26"),
            (12308, 2, "123.08"),
            (314, 2, "3.14"),
            (-314, 2, "-3.14"),
            (12345678, 15, "1.2345678E-8"),
            (
                12345678912345123456789123,
                12,
                "12345678912345.123456789123",
            ),
            (700, 1, "70.0"),
            (1, 2, "0.01"),
            (10, 2, "0.10"),
            (-10, 2, "-0.10"),
            (1, 5, "0.00001"),
            (24, 3, "0.024"),
            (535, 2, "5.35"),
            (0, 0, "0"),
            (0, 3, "0.000"),
            (0, 6, "0.000000"),
            (0, 7, "0E-7"),
            (0, 8, "0E-8"),
            (1, 7, "1E-7"),
            (-1, 7, "-1E-7"),
            (5, 0, "5"),
        ];
        for (unscaled, scale, want) in cases {
            assert_eq!(&decimal(*unscaled, *scale), want, "{unscaled}E-{scale}");
        }
    }

    #[test]
    fn dates_add_plus_before_long_years() {
        // `DATE '+999999-03-18'` etc. from `datetime-special`.
        assert_eq!(date(364_522_683), "DATE '+999999-03-18'");
        assert_eq!(date(-719_866), "DATE '-0001-01-28'");
        assert_eq!(date(-714_049), "DATE '0015-01-01'");
        assert_eq!(date(2_145_042_905), "DATE '+5874897-12-31'");
        assert_eq!(date(10957), "DATE '2000-01-01'");
    }

    #[test]
    fn timestamps_print_as_spark_sql() {
        assert_eq!(timestamp(0, false), "TIMESTAMP '1970-01-01 00:00:00'");
        assert_eq!(
            timestamp(-124_271_308_800_000_000, false),
            "TIMESTAMP '-1969-12-31 16:00:00'"
        );
        assert_eq!(
            timestamp(3_093_503_054_597_000_000, false),
            "TIMESTAMP '+99999-03-18 12:03:17'"
        );
        assert_eq!(
            timestamp(1_457_729_640_123_456, false),
            "TIMESTAMP '2016-03-11 20:54:00.123456'"
        );
        assert_eq!(
            timestamp(1_457_729_640_123_000, false),
            "TIMESTAMP '2016-03-11 20:54:00.123'"
        );
        assert_eq!(
            timestamp(1_457_729_640_000_000, true),
            "TIMESTAMP_NTZ '2016-03-11 20:54:00'"
        );
    }

    #[test]
    fn layout_follows_magnitude() {
        // Plain in `[1e-3, 1e7)`, scientific outside, `E` without `+`.
        for (e, want) in [
            ("1e-3", "0.001"),
            ("1e-4", "1.0E-4"),
            ("1e6", "1000000.0"),
            ("1e7", "1.0E7"),
            ("-1e-3", "-0.001"),
            ("-1e7", "-1.0E7"),
        ] {
            assert_eq!(from_scientific(e), want, "{e}");
        }
    }
}
