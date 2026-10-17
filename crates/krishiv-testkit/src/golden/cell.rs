// SPDX-License-Identifier: Apache-2.0
//! A testkit-owned formatter for result cells in the style of Spark's `HiveResult`
//! (`toHiveString`), so engines do not each re-implement Spark's text rendering.
//!
//! Top-level NULL renders as `NULL`, nested NULL as `null`; nested strings are quoted; arrays
//! are `[a,b]`, maps and structs `{"k":v}`; doubles and floats follow Java's `toString`
//! (`1.0`, `1.0E10`, `NaN`, `Infinity`); decimals keep their scale; timestamps are rendered
//! from local-time microseconds (the engine applies the session time zone).

use std::fmt::Write as _;

/// How `spark.sql.binaryOutputStyle` renders binary values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BinaryStyle {
    /// UTF-8 text (Spark's default when the config is unset).
    #[default]
    Utf8,
    /// Upper-case hex, `DEADBEEF`.
    Hex,
    /// Upper-case hex pairs, `[DE AD BE EF]`.
    HexDiscrete,
    /// Signed byte values, `[-34, -83]`.
    Basic,
    /// Base64.
    Base64,
}

/// One result value.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    /// SQL NULL.
    Null,
    /// BOOLEAN.
    Bool(bool),
    /// TINYINT/SMALLINT/INT/BIGINT.
    Int(i64),
    /// FLOAT.
    Float(f32),
    /// DOUBLE.
    Double(f64),
    /// DECIMAL: unscaled value and scale.
    Decimal(i128, u32),
    /// STRING.
    Str(String),
    /// BINARY.
    Binary(Vec<u8>),
    /// DATE as days since the epoch.
    Date(i32),
    /// TIMESTAMP as local-time microseconds since the epoch.
    Timestamp(i64),
    /// ARRAY.
    Array(Vec<Cell>),
    /// MAP, in entry order.
    Map(Vec<(Cell, Cell)>),
    /// STRUCT: field names and values.
    Struct(Vec<(String, Cell)>),
}

/// Render a top-level row as Spark does (`NULL` for nulls).
#[must_use]
pub fn render_row(cells: &[Cell], binary: BinaryStyle) -> Vec<String> {
    cells.iter().map(|c| render(c, binary, true)).collect()
}

fn render(cell: &Cell, binary: BinaryStyle, top: bool) -> String {
    let nested = |c: &Cell| render(c, binary, false);
    match cell {
        Cell::Null => if top { "NULL" } else { "null" }.to_owned(),
        Cell::Bool(b) => b.to_string(),
        Cell::Int(i) => i.to_string(),
        Cell::Float(f) => java_double(f64::from(*f), Some(*f)),
        Cell::Double(d) => java_double(*d, None),
        Cell::Decimal(u, scale) => decimal(*u, *scale),
        Cell::Str(s) if top => s.clone(),
        Cell::Str(s) => format!("\"{s}\""),
        Cell::Binary(b) => binary_text(b, binary),
        Cell::Date(d) => date(i64::from(*d)),
        Cell::Timestamp(us) => timestamp(*us),
        Cell::Array(v) => format!("[{}]", v.iter().map(nested).collect::<Vec<_>>().join(",")),
        Cell::Map(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}:{}", nested(k), nested(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Cell::Struct(f) => format!(
            "{{{}}}",
            f.iter()
                .map(|(k, v)| format!("\"{k}\":{}", nested(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

/// Java's `Double.toString` / `Float.toString` (JDK 19+ shortest-digits behaviour).
fn java_double(d: f64, as_f32: Option<f32>) -> String {
    if d.is_nan() {
        return "NaN".to_owned();
    }
    if d.is_infinite() {
        return if d > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    if d == 0.0 {
        return if d.is_sign_negative() { "-0.0" } else { "0.0" }.to_owned();
    }
    // Shortest round-trip digits in scientific form: `d.ddde<exp>`.
    let sci = as_f32.map_or_else(|| format!("{d:e}"), |f| format!("{f:e}"));
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (-3..7).contains(&exp) {
        let point = usize::try_from(exp + 1).unwrap_or(0);
        let plain = if exp >= 0 {
            let int_len = point.min(digits.len());
            let mut int_part = digits[..int_len].to_owned();
            int_part.push_str(&"0".repeat(point - int_len));
            let frac = &digits[int_len..];
            format!("{int_part}.{}", if frac.is_empty() { "0" } else { frac })
        } else {
            format!(
                "0.{}{digits}",
                "0".repeat(usize::try_from(-exp - 1).unwrap_or(0))
            )
        };
        format!("{sign}{plain}")
    } else {
        let frac = &digits[1..];
        format!(
            "{sign}{}.{}E{exp}",
            &digits[..1],
            if frac.is_empty() { "0" } else { frac }
        )
    }
}

fn decimal(unscaled: i128, scale: u32) -> String {
    let neg = unscaled < 0;
    let mut digits = unscaled.unsigned_abs().to_string();
    let scale = scale as usize;
    if scale > 0 {
        if digits.len() <= scale {
            digits = format!("{}{digits}", "0".repeat(scale + 1 - digits.len()));
        }
        digits.insert(digits.len() - scale, '.');
    }
    if neg { format!("-{digits}") } else { digits }
}

/// Civil date from days since 1970-01-01 (proleptic Gregorian).
fn civil(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn date(days: i64) -> String {
    let (y, m, d) = civil(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn timestamp(us: i64) -> String {
    let secs = us.div_euclid(1_000_000);
    let micros = us.rem_euclid(1_000_000);
    let tod = secs.rem_euclid(86_400);
    let mut s = format!(
        "{} {:02}:{:02}:{:02}",
        date(secs.div_euclid(86_400)),
        tod / 3600,
        tod % 3600 / 60,
        tod % 60
    );
    if micros != 0 {
        let frac = format!("{micros:06}");
        let _ = write!(s, ".{}", frac.trim_end_matches('0'));
    }
    s
}

fn binary_text(b: &[u8], style: BinaryStyle) -> String {
    match style {
        BinaryStyle::Utf8 => String::from_utf8_lossy(b).into_owned(),
        BinaryStyle::Hex => b.iter().map(|x| format!("{x:02X}")).collect(),
        BinaryStyle::HexDiscrete => format!(
            "[{}]",
            b.iter()
                .map(|x| format!("{x:02X}"))
                .collect::<Vec<_>>()
                .join(" ")
        ),
        BinaryStyle::Basic => format!(
            "[{}]",
            b.iter()
                .map(|x| (*x as i8).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        BinaryStyle::Base64 => base64(b),
    }
}

fn base64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in b.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0_u32, |a, (i, x)| a | (u32::from(*x) << (16 - 8 * i)));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(char::from(T[((n >> (18 - 6 * i)) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(c: &Cell) -> String {
        render(c, BinaryStyle::Utf8, true)
    }

    #[test]
    fn doubles_follow_java() {
        let d = |x: f64| one(&Cell::Double(x));
        assert_eq!(d(1.0), "1.0");
        assert_eq!(d(0.001), "0.001");
        assert_eq!(d(0.000_123), "1.23E-4");
        assert_eq!(d(1234.5), "1234.5");
        assert_eq!(d(1.0e7), "1.0E7");
        assert_eq!(d(2.35e10), "2.35E10");
        assert_eq!(d(-0.0), "-0.0");
        assert_eq!(d(f64::NAN), "NaN");
        assert_eq!(d(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(one(&Cell::Float(0.1)), "0.1");
    }

    #[test]
    fn decimals_dates_timestamps() {
        assert_eq!(one(&Cell::Decimal(-5, 3)), "-0.005");
        assert_eq!(one(&Cell::Decimal(12345, 2)), "123.45");
        assert_eq!(one(&Cell::Decimal(7, 0)), "7");
        assert_eq!(one(&Cell::Date(0)), "1970-01-01");
        assert_eq!(one(&Cell::Date(-1)), "1969-12-31");
        assert_eq!(
            one(&Cell::Timestamp(1_609_459_200_000_000)),
            "2021-01-01 00:00:00"
        );
        assert_eq!(one(&Cell::Timestamp(1_500_000)), "1970-01-01 00:00:01.5");
    }

    #[test]
    fn nesting_nulls_and_binary() {
        let a = Cell::Array(vec![Cell::Null, Cell::Str("b".into())]);
        assert_eq!(one(&a), "[null,\"b\"]");
        assert_eq!(one(&Cell::Null), "NULL");
        let s = Cell::Struct(vec![(
            "a".into(),
            Cell::Map(vec![(Cell::Str("k".into()), Cell::Int(1))]),
        )]);
        assert_eq!(one(&s), r#"{"a":{"k":1}}"#);
        let b = vec![0xDE, 0xAD];
        assert_eq!(binary_text(&b, BinaryStyle::Hex), "DEAD");
        assert_eq!(binary_text(&b, BinaryStyle::HexDiscrete), "[DE AD]");
        assert_eq!(binary_text(&b, BinaryStyle::Basic), "[-34, -83]");
        assert_eq!(binary_text(b"Man", BinaryStyle::Base64), "TWFu");
        assert_eq!(binary_text(b"Ma", BinaryStyle::Base64), "TWE=");
    }
}
