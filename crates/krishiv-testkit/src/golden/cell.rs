// SPDX-License-Identifier: Apache-2.0
//! A testkit-owned formatter for result cells, a port of the rules of Spark 4.2.0's `HiveResult`
//! (`toHiveString`) as `SQLQueryTestSuite` runs them, so engines do not each re-implement
//! Spark's text rendering. The rules are checked against Spark itself: `tests/fixtures/spark-4.2.0/
//! hive-result.json` holds queries run through `HiveResult.hiveResultString` with the typed
//! values they returned.
//!
//! * NULL is `NULL` at the top level and `null` inside arrays, maps and structs; nested strings
//!   are quoted (without escaping); binary is never quoted.
//! * Arrays are `[a,b]`, structs `{"name":v}`, maps `{k:v}` with the rendered entries sorted as
//!   strings in UTF-16 order (Spark sorts the entry texts, not the keys).
//! * Integers, booleans and decimals (`toPlainString`, so the scale is kept and there is never an
//!   exponent) print as their value; floats and doubles as Java's `toString` of the JDK 19+
//!   shortest-digits algorithm (`1.0E10`, `1.0E-4`, `NaN`, `Infinity`; plain notation for
//!   magnitudes in `[1e-3, 1e7)`). The older JDK 17 algorithm is not reproduced; use
//!   [`super::Jdk::Java21`] results.
//! * Dates and timestamps print as `yyyy-MM-dd[ HH:mm:ss[.fraction]]` with the year of the
//!   proleptic calendar: four digits, `+` before a fifth one, `-` when negative. A TIMESTAMP is
//!   an instant shown in the session time zone (the suite collects `java.time` values, so the
//!   zone's historical offsets, local mean time included, apply); TIMESTAMP_NTZ is shown as is.
//! * Intervals print Hive style, whatever their fields: `1-2`, `-1 02:03:04.500000000`; the
//!   multi-unit `interval` type prints as `CalendarInterval.toString`.
//! * Not covered: the `time`, `variant`, geospatial and user-defined types; `TIME` is off by
//!   default in Spark 4.2.

use std::fmt::LowerExp;
use std::str::FromStr;

use jiff::Timestamp;
use jiff::tz::{Offset, TimeZone};

/// The session time zone of `SQLQueryTestSuite` (its test JVM runs in this zone).
pub const SUITE_TIME_ZONE: &str = "America/Los_Angeles";

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
    /// Base64 without padding.
    Base64,
}

impl BinaryStyle {
    /// The style a `spark.sql.binaryOutputStyle` value names (case-insensitive).
    #[must_use]
    pub fn from_conf(value: &str) -> Option<Self> {
        match value.to_ascii_uppercase().as_str() {
            "UTF-8" => Some(Self::Utf8),
            "HEX" => Some(Self::Hex),
            "HEX_DISCRETE" => Some(Self::HexDiscrete),
            "BASIC" => Some(Self::Basic),
            "BASE64" => Some(Self::Base64),
            _ => None,
        }
    }
}

/// A [`CellFormat`] that cannot be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CellFormatError {
    /// Not a time zone Spark accepts.
    #[error("unknown time zone `{0}`")]
    TimeZone(String),
    /// Not a value of `spark.sql.binaryOutputStyle`.
    #[error("unknown binary output style `{0}`")]
    BinaryStyle(String),
}

/// The session settings that decide how cells print.
#[derive(Debug, Clone)]
pub struct CellFormat {
    zone: TimeZone,
    binary: BinaryStyle,
}

impl Default for CellFormat {
    /// The suite's session zone ([`SUITE_TIME_ZONE`], which the goldens were recorded in; its
    /// test JVM runs there) and UTF-8 binary. Falls back to UTC only if the zone database lacks
    /// the zone.
    fn default() -> Self {
        Self {
            zone: parse_zone(SUITE_TIME_ZONE).unwrap_or(TimeZone::UTC),
            binary: BinaryStyle::default(),
        }
    }
}

/// `+h`, `+hh`, `+hh:mm`, `+hhmm`, `+hh:mm:ss` or `+hhmmss` (or with `-`) as the seconds east
/// of UTC, as Java's `ZoneOffset.of` takes them.
fn offset_seconds(s: &str) -> Option<i32> {
    let (sign, rest) = match s.split_at_checked(1)? {
        ("+", rest) => (1, rest),
        ("-", rest) => (-1, rest),
        _ => return None,
    };
    let colons = rest.matches(':').count();
    let bytes = rest.as_bytes();
    let shaped = colons == 0
        || (bytes.get(2) == Some(&b':')
            && (colons == 1 && rest.len() == 5
                || colons == 2 && rest.len() == 8 && bytes.get(5) == Some(&b':')));
    let digits: String = rest.chars().filter(|c| *c != ':').collect();
    if !shaped || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let part = |i: usize| -> Option<i32> { digits.get(i..i + 2)?.parse().ok() };
    let (h, m, sec) = match digits.len() {
        1 => (digits.parse().ok()?, 0, 0),
        2 => (part(0)?, 0, 0),
        4 => (part(0)?, part(2)?, 0),
        6 => (part(0)?, part(2)?, part(4)?),
        _ => return None,
    };
    let total = h * 3600 + m * 60 + sec;
    (m < 60 && sec < 60 && total <= 18 * 3600).then_some(sign * total)
}

/// Java's `ZoneId.SHORT_IDS`: the three-letter IDs `ZoneId.of` accepts when given them.
const SHORT_IDS: [(&str, &str); 28] = [
    ("ACT", "Australia/Darwin"),
    ("AET", "Australia/Sydney"),
    ("AGT", "America/Argentina/Buenos_Aires"),
    ("ART", "Africa/Cairo"),
    ("AST", "America/Anchorage"),
    ("BET", "America/Sao_Paulo"),
    ("BST", "Asia/Dhaka"),
    ("CAT", "Africa/Harare"),
    ("CNT", "America/St_Johns"),
    ("CST", "America/Chicago"),
    ("CTT", "Asia/Shanghai"),
    ("EAT", "Africa/Addis_Ababa"),
    ("ECT", "Europe/Paris"),
    ("IET", "America/Indiana/Indianapolis"),
    ("IST", "Asia/Kolkata"),
    ("JST", "Asia/Tokyo"),
    ("MIT", "Pacific/Apia"),
    ("NET", "Asia/Yerevan"),
    ("NST", "Pacific/Auckland"),
    ("PLT", "Asia/Karachi"),
    ("PNT", "America/Phoenix"),
    ("PRT", "America/Puerto_Rico"),
    ("PST", "America/Los_Angeles"),
    ("SST", "Pacific/Guadalcanal"),
    ("VST", "Asia/Ho_Chi_Minh"),
    ("EST", "-05:00"),
    ("MST", "-07:00"),
    ("HST", "-10:00"),
];

/// Spark's two `getZoneId` normalizations (`SparkDateTimeUtils`), in order:
/// `singleHourTz` `(\+|\-)(\d):` → `$10$2:` (`+5:30` is `+05:30`), then
/// `singleMinuteTz` `(\+|\-)(\d\d):(\d)$` → `$1$2:0$3` (`+10:7` is
/// `+10:07`). Each replaces the first match only.
fn pad_hour(name: &str) -> String {
    let b = name.as_bytes();
    let hour = (0..b.len().saturating_sub(2))
        .find(|&i| matches!(b[i], b'+' | b'-') && b[i + 1].is_ascii_digit() && b[i + 2] == b':');
    let name = match hour {
        Some(i) => format!("{}0{}", &name[..=i], &name[i + 1..]),
        None => name.to_owned(),
    };
    // `(\+|\-)(\d\d):(\d)$`: the first sign followed by two digits, a colon
    // and exactly one final digit.
    let b = name.as_bytes();
    let minute = (0..b.len().saturating_sub(4)).find(|&i| {
        matches!(b[i], b'+' | b'-')
            && b[i + 1].is_ascii_digit()
            && b[i + 2].is_ascii_digit()
            && b[i + 3] == b':'
            && b[i + 4].is_ascii_digit()
            && i + 5 == b.len()
    });
    match minute {
        Some(i) => format!("{}0{}", &name[..i + 4], &name[i + 4..]),
        None => name,
    }
}

/// A zone ID as Spark's `DateTimeUtils.getZoneId` takes it (`ZoneId.of` with the short IDs):
/// `Z`, an offset, `UTC`/`GMT`/`UT` with an optional offset, a short ID or a region name.
fn parse_zone(name: &str) -> Option<TimeZone> {
    let fixed = |secs: i32| Offset::from_seconds(secs).ok().map(TimeZone::fixed);
    let padded = pad_hour(name);
    let name = SHORT_IDS
        .iter()
        .find(|(short, _)| *short == padded)
        .map_or(padded.as_str(), |(_, id)| id);
    if matches!(name, "Z" | "UTC" | "GMT" | "UT") {
        return Some(TimeZone::UTC);
    }
    if let Some(secs) = offset_seconds(name) {
        return fixed(secs);
    }
    for prefix in ["UTC", "GMT", "UT"] {
        if let Some(secs) = name.strip_prefix(prefix).and_then(offset_seconds) {
            return fixed(secs);
        }
    }
    TimeZone::get(name).ok()
}

impl CellFormat {
    /// Format for session time zone `zone` and binary `binary` style.
    pub fn new(zone: &str, binary: BinaryStyle) -> Result<Self, CellFormatError> {
        Ok(Self {
            zone: parse_zone(zone).ok_or_else(|| CellFormatError::TimeZone(zone.to_owned()))?,
            binary,
        })
    }

    /// Format for the session `configs` ([`super::setup::SessionSetup::configs`]): the last
    /// `spark.sql.session.timeZone` ([`SUITE_TIME_ZONE`] when absent) and
    /// `spark.sql.binaryOutputStyle` (UTF-8 when absent).
    pub fn for_session(configs: &[(String, String)]) -> Result<Self, CellFormatError> {
        let last = |key: &str| {
            configs
                .iter()
                .rev()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
        };
        let binary = match last("spark.sql.binaryOutputStyle") {
            None => BinaryStyle::default(),
            Some(v) => BinaryStyle::from_conf(v)
                .ok_or_else(|| CellFormatError::BinaryStyle(v.to_owned()))?,
        };
        Self::new(
            last("spark.sql.session.timeZone").unwrap_or(SUITE_TIME_ZONE),
            binary,
        )
    }

    /// Render a top-level row as Spark does, one string per cell.
    #[must_use]
    pub fn render_row(&self, cells: &[Cell]) -> Vec<String> {
        cells.iter().map(|c| self.render(c, true)).collect()
    }

    /// Spark's `toHiveString`; `top` is false inside arrays, maps and structs.
    #[must_use]
    pub fn render(&self, cell: &Cell, top: bool) -> String {
        let nested = |c: &Cell| self.render(c, false);
        match cell {
            Cell::Null => if top { "NULL" } else { "null" }.to_owned(),
            Cell::Bool(b) => b.to_string(),
            Cell::Int(i) => i.to_string(),
            Cell::Float(f) => java_float(*f),
            Cell::Double(d) => java_double(*d),
            Cell::Decimal(u, scale) => decimal(*u, *scale),
            Cell::Str(s) if top => s.clone(),
            Cell::Str(s) => format!("\"{s}\""),
            Cell::Binary(b) => binary_text(b, self.binary),
            Cell::Date(d) => date(i64::from(*d)),
            Cell::Time(ns) => krishiv_kernels::cast::format_time(*ns),
            Cell::Timestamp(us) => {
                let secs = us.div_euclid(1_000_000);
                let offset = zone_offset(&self.zone, secs);
                datetime(secs + offset, us.rem_euclid(1_000_000))
            }
            Cell::TimestampNtz(us) => datetime(us.div_euclid(1_000_000), us.rem_euclid(1_000_000)),
            Cell::YearMonth(months) => year_month(*months),
            Cell::DayTime(micros) => day_time(*micros),
            Cell::Calendar {
                months,
                days,
                micros,
            } => calendar(*months, *days, *micros),
            Cell::Array(v) => format!("[{}]", v.iter().map(nested).collect::<Vec<_>>().join(",")),
            Cell::Map(m) => {
                let mut entries: Vec<String> = m
                    .iter()
                    .map(|(k, v)| format!("{}:{}", nested(k), nested(v)))
                    .collect();
                // Scala `String` order is UTF-16 code-unit order.
                entries.sort_unstable_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
                format!("{{{}}}", entries.join(","))
            }
            Cell::Struct(f) => format!(
                "{{{}}}",
                f.iter()
                    .map(|(k, v)| format!("\"{k}\":{}", nested(v)))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
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
    /// STRING, CHAR, VARCHAR.
    Str(String),
    /// BINARY.
    Binary(Vec<u8>),
    /// DATE as days since the epoch.
    Date(i32),
    /// TIME as nanoseconds since midnight.
    Time(i64),
    /// TIMESTAMP as microseconds since the epoch (an instant; shown in the session time zone).
    Timestamp(i64),
    /// TIMESTAMP_NTZ as microseconds of local time since the epoch.
    TimestampNtz(i64),
    /// INTERVAL YEAR TO MONTH (any fields) as months.
    YearMonth(i32),
    /// INTERVAL DAY TO SECOND (any fields) as microseconds.
    DayTime(i64),
    /// INTERVAL (calendar): months, days and microseconds.
    Calendar {
        /// Months.
        months: i32,
        /// Days.
        days: i32,
        /// Microseconds.
        micros: i64,
    },
    /// ARRAY.
    Array(Vec<Cell>),
    /// MAP, in storage order.
    Map(Vec<(Cell, Cell)>),
    /// STRUCT: field names and values.
    Struct(Vec<(String, Cell)>),
}

/// Render a top-level row as Spark does (`NULL` for nulls).
#[must_use]
pub fn render_row(cells: &[Cell], format: &CellFormat) -> Vec<String> {
    format.render_row(cells)
}

/// The shortest decimal that round-trips, as `d.ddde<exp>`. When that is a single digit, Java's
/// `toString` takes the closest two-digit decimal in the rounding interval instead, which differs
/// from `d.0` only for subnormals (`4.9E-324`).
fn shortest_sci<T>(x: T) -> String
where
    T: LowerExp + FromStr + PartialEq + Copy,
{
    let sci = format!("{x:e}");
    if sci
        .split('e')
        .next()
        .is_some_and(|m| m.trim_start_matches('-').len() == 1)
    {
        let two = format!("{x:.1e}");
        if two.parse::<T>().is_ok_and(|back| back == x) {
            return two;
        }
    }
    sci
}

/// `Double.toString` / `Float.toString` of JDK 19+, from the shortest digits `sci`: plain
/// notation for magnitudes in `[1e-3, 1e7)`, `d.dddE<exp>` outside, at least one fraction digit.
fn java_float_text(sci: &str) -> String {
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    // The two-digit form of a one-digit value is `d.0`.
    let digits = digits.trim_end_matches('0');
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

/// `NaN`, `Infinity`, `-Infinity` and signed zero.
fn special(d: f64) -> Option<String> {
    if d.is_nan() {
        Some("NaN".to_owned())
    } else if d.is_infinite() {
        Some(if d > 0.0 { "Infinity" } else { "-Infinity" }.to_owned())
    } else if d == 0.0 {
        Some(if d.is_sign_negative() { "-0.0" } else { "0.0" }.to_owned())
    } else {
        None
    }
}

fn java_double(d: f64) -> String {
    special(d).unwrap_or_else(|| java_float_text(&shortest_sci(d)))
}

fn java_float(f: f32) -> String {
    special(f64::from(f)).unwrap_or_else(|| java_float_text(&shortest_sci(f)))
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

/// `yyyy-MM-dd` with Spark's year: at least four digits, `+` before a fifth, `-` when negative.
fn date(days: i64) -> String {
    let (y, m, d) = civil(days);
    let year = match y {
        10_000.. => format!("+{y}"),
        ..0 => format!("-{:04}", -y),
        _ => format!("{y:04}"),
    };
    format!("{year}-{m:02}-{d:02}")
}

/// `yyyy-MM-dd HH:mm:ss` of local `secs` since the epoch, plus the fraction without trailing
/// zeros.
fn datetime(secs: i64, micros: i64) -> String {
    let tod = secs.rem_euclid(86_400);
    let mut s = format!(
        "{} {:02}:{:02}:{:02}",
        date(secs.div_euclid(86_400)),
        tod / 3600,
        tod % 3600 / 60,
        tod % 60
    );
    if micros != 0 {
        s.push('.');
        s.push_str(format!("{micros:06}").trim_end_matches('0'));
    }
    s
}

/// Seconds east of UTC that `zone` has at the instant `secs`. `jiff` covers the years
/// -9999..=9999; later and earlier instants take the offset of the same moment of the 400-year
/// Gregorian cycle, which repeats the weekdays and so the zone's rules.
fn zone_offset(zone: &TimeZone, secs: i64) -> i64 {
    const CYCLE: i64 = 146_097 * 86_400;
    let (min, max) = (Timestamp::MIN.as_second(), Timestamp::MAX.as_second());
    let folded = if secs > max {
        secs - (secs - max + CYCLE - 1) / CYCLE * CYCLE
    } else if secs < min {
        secs + (min - secs + CYCLE - 1) / CYCLE * CYCLE
    } else {
        secs
    };
    Timestamp::from_second(folded).map_or(0, |ts| i64::from(zone.to_offset(ts).seconds()))
}

/// Hive-style year-month interval: `y-m`, with a leading `-` when negative.
fn year_month(months: i32) -> String {
    let abs = i64::from(months).abs();
    format!(
        "{}{}-{}",
        if months < 0 { "-" } else { "" },
        abs / 12,
        abs % 12
    )
}

/// Hive-style day-time interval: `d HH:mm:ss.nnnnnnnnn`, with a leading `-` when negative.
fn day_time(micros: i64) -> String {
    let abs = micros.unsigned_abs();
    let (secs, frac) = (abs / 1_000_000, abs % 1_000_000);
    format!(
        "{}{} {:02}:{:02}:{:02}.{:09}",
        if micros < 0 { "-" } else { "" },
        secs / 86_400,
        secs % 86_400 / 3600,
        secs % 3600 / 60,
        secs % 60,
        frac * 1000
    )
}

/// `CalendarInterval.toString`: the non-zero units, `0 seconds` for none.
fn calendar(months: i32, days: i32, micros: i64) -> String {
    if months == 0 && days == 0 && micros == 0 {
        return "0 seconds".to_owned();
    }
    let mut parts = Vec::new();
    let mut unit = |n: i64, name: &str| {
        if n != 0 {
            parts.push(format!("{n} {name}"));
        }
    };
    unit(i64::from(months / 12), "years");
    unit(i64::from(months % 12), "months");
    unit(i64::from(days), "days");
    const HOUR: i64 = 3_600_000_000;
    const MINUTE: i64 = 60_000_000;
    unit(micros / HOUR, "hours");
    unit(micros % HOUR / MINUTE, "minutes");
    let rest = micros % MINUTE;
    if rest != 0 {
        let frac = format!("{:06}", rest.unsigned_abs() % 1_000_000);
        let frac = frac.trim_end_matches('0');
        let sign = if rest < 0 { "-" } else { "" };
        let whole = rest.unsigned_abs() / 1_000_000;
        let point = if frac.is_empty() {
            String::new()
        } else {
            format!(".{frac}")
        };
        parts.push(format!("{sign}{whole}{point} seconds"));
    }
    parts.join(" ")
}

fn binary_text(b: &[u8], style: BinaryStyle) -> String {
    match style {
        BinaryStyle::Utf8 => java_utf8(b),
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
                .map(|x| i8::from_le_bytes([*x]).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        BinaryStyle::Base64 => base64(b),
    }
}

/// `new String(bytes, UTF_8)` of JDK 21: malformed input becomes U+FFFD, one per malformed
/// sequence as the JDK counts them (an encoded surrogate is one, three bytes long).
fn java_utf8(b: &[u8]) -> String {
    const REPL: char = '\u{FFFD}';
    let at = |i: usize| b.get(i).copied().unwrap_or(0);
    let cont = |x: u8| x & 0xc0 == 0x80;
    let mut out = String::new();
    let mut sp = 0;
    while sp < b.len() {
        let (start, b1) = (sp, b[sp]);
        sp += 1;
        match b1 {
            0..=0x7f => out.push(char::from(b1)),
            0xc2..=0xdf => {
                if sp >= b.len() {
                    out.push(REPL);
                    break;
                }
                let b2 = at(sp);
                if cont(b2) {
                    sp += 1;
                    out.extend(char::from_u32(
                        u32::from(b1 & 0x1f) << 6 | u32::from(b2 & 0x3f),
                    ));
                } else {
                    out.push(REPL);
                }
            }
            0xe0..=0xef if sp + 1 < b.len() => {
                let (b2, b3) = (at(sp), at(sp + 1));
                if (b1 == 0xe0 && b2 & 0xe0 == 0x80) || !cont(b2) || !cont(b3) {
                    out.push(REPL);
                    sp = start
                        + if (b1 == 0xe0 && b2 & 0xe0 == 0x80) || !cont(b2) {
                            1
                        } else {
                            2
                        };
                } else {
                    let c = u32::from(b1 & 0x0f) << 12
                        | u32::from(b2 & 0x3f) << 6
                        | u32::from(b3 & 0x3f);
                    out.push(char::from_u32(c).unwrap_or(REPL));
                    sp += 2;
                }
            }
            0xe0..=0xef => {
                // Truncated: fewer than two bytes follow.
                out.push(REPL);
                if !(sp < b.len() && ((b1 == 0xe0 && at(sp) & 0xe0 == 0x80) || !cont(at(sp)))) {
                    break;
                }
            }
            0xf0..=0xf7 if sp + 2 < b.len() => {
                let (b2, b3, b4) = (at(sp), at(sp + 1), at(sp + 2));
                let uc = u32::from(b1 & 0x07) << 18
                    | u32::from(b2 & 0x3f) << 12
                    | u32::from(b3 & 0x3f) << 6
                    | u32::from(b4 & 0x3f);
                match char::from_u32(uc) {
                    Some(c) if uc >= 0x1_0000 && cont(b2) && cont(b3) && cont(b4) => {
                        out.push(c);
                        sp += 3;
                    }
                    _ => {
                        out.push(REPL);
                        let bad2 = b1 > 0xf4
                            || (b1 == 0xf0 && !(0x90..=0xbf).contains(&b2))
                            || (b1 == 0xf4 && b2 & 0xf0 != 0x80)
                            || !cont(b2);
                        sp = start
                            + if bad2 {
                                1
                            } else if !cont(b3) {
                                2
                            } else {
                                3
                            };
                    }
                }
            }
            0xf0..=0xf7 => {
                // Truncated: fewer than three bytes follow.
                out.push(REPL);
                if b1 > 0xf4
                    || (sp < b.len()
                        && ((b1 == 0xf0 && !(0x90..=0xbf).contains(&at(sp)))
                            || (b1 == 0xf4 && at(sp) & 0xf0 != 0x80)
                            || !cont(at(sp))))
                {
                    continue;
                }
                sp += 1;
                if !(sp < b.len() && !cont(at(sp))) {
                    break;
                }
            }
            _ => out.push(REPL),
        }
    }
    out
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
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::golden::value::SparkType;

    type R = Result<(), Box<dyn std::error::Error>>;

    fn fmt() -> CellFormat {
        CellFormat::new("UTC", BinaryStyle::Utf8).unwrap_or_default()
    }

    #[test]
    fn default_format_is_the_suite_zone() {
        let f = CellFormat::default();
        assert_eq!(f.render(&Cell::Timestamp(0), true), "1969-12-31 16:00:00");
        assert_eq!(
            fmt().render(&Cell::Timestamp(0), true),
            "1970-01-01 00:00:00"
        );
    }

    fn one(c: &Cell) -> String {
        fmt().render(c, true)
    }

    /// Every query of `hive-result.json`: Spark's own text against the formatter's.
    #[test]
    fn matches_spark_hive_result_on_every_probe() -> R {
        let path = crate::tests_dir().join("fixtures/spark-4.2.0/hive-result.json");
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let probes = doc["probes"].as_array().ok_or("probes")?;
        assert!(probes.len() > 200, "{} probes", probes.len());
        for p in probes {
            let sql = p["sql"].as_str().ok_or("sql")?;
            let style = BinaryStyle::from_conf(p["binary_style"].as_str().ok_or("style")?)
                .ok_or("binary style")?;
            let format = CellFormat::new(p["time_zone"].as_str().ok_or("zone")?, style)?;
            let types = p["types"]
                .as_array()
                .ok_or("types")?
                .iter()
                .map(|t| SparkType::parse(t.as_str().unwrap_or_default()))
                .collect::<Result<Vec<_>, _>>()?;
            let expected: Vec<&str> = p["hive"]
                .as_array()
                .ok_or("hive")?
                .iter()
                .filter_map(Value::as_str)
                .collect();
            let rows = p["rows"].as_array().ok_or("rows")?;
            assert_eq!(rows.len(), expected.len(), "{sql}");
            for (row, want) in rows.iter().zip(expected) {
                let cells = row
                    .as_array()
                    .ok_or("row")?
                    .iter()
                    .zip(&types)
                    .map(|(v, t)| Cell::from_json(v, t))
                    .collect::<Result<Vec<_>, _>>()?;
                let got = render_row(&cells, &format).join("\t");
                assert_eq!(got, want, "{sql} [{} {:?}]", p["time_zone"], style);
            }
        }
        Ok(())
    }

    #[test]
    fn doubles_and_floats_follow_java() {
        let d = |x: f64| one(&Cell::Double(x));
        let f = |x: f32| one(&Cell::Float(x));
        assert_eq!(d(1.0), "1.0");
        assert_eq!(d(0.001), "0.001");
        assert_eq!(d(0.000_123), "1.23E-4");
        assert_eq!(d(1234.5), "1234.5");
        assert_eq!(d(9_999_999.5), "9999999.5");
        assert_eq!(d(1.0e7), "1.0E7");
        assert_eq!(d(2.35e10), "2.35E10");
        assert_eq!(d(-0.0), "-0.0");
        assert_eq!(d(f64::NAN), "NaN");
        assert_eq!(d(f64::INFINITY), "Infinity");
        assert_eq!(d(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(d(5e-324), "4.9E-324");
        assert_eq!(d(-1e-323), "-9.9E-324");
        assert_eq!(f(0.1), "0.1");
        assert_eq!(f(1.0e-45), "1.4E-45");
        assert_eq!(f(16_777_216.0), "1.6777216E7");
        assert_eq!(f(f32::NAN), "NaN");
    }

    #[test]
    fn decimals_dates_timestamps() {
        assert_eq!(one(&Cell::Decimal(-5, 3)), "-0.005");
        assert_eq!(one(&Cell::Decimal(12345, 2)), "123.45");
        assert_eq!(one(&Cell::Decimal(7, 0)), "7");
        assert_eq!(one(&Cell::Decimal(0, 2)), "0.00");
        assert_eq!(one(&Cell::Date(0)), "1970-01-01");
        assert_eq!(one(&Cell::Date(-1)), "1969-12-31");
        assert_eq!(one(&Cell::Date(-719_529)), "-0001-12-31");
        assert_eq!(one(&Cell::Date(2_932_897)), "+10000-01-01");
        assert_eq!(
            one(&Cell::Timestamp(1_609_459_200_000_000)),
            "2021-01-01 00:00:00"
        );
        assert_eq!(one(&Cell::Timestamp(1_500_000)), "1970-01-01 00:00:01.5");
        assert_eq!(one(&Cell::TimestampNtz(-1)), "1969-12-31 23:59:59.999999");
    }

    #[test]
    fn intervals_hive_style() {
        assert_eq!(one(&Cell::YearMonth(14)), "1-2");
        assert_eq!(one(&Cell::YearMonth(i32::MIN)), "-178956970-8");
        assert_eq!(one(&Cell::DayTime(-1)), "-0 00:00:00.000001000");
        assert_eq!(one(&Cell::DayTime(93_784_005_000)), "1 02:03:04.005000000");
        let cal = |months, days, micros| {
            one(&Cell::Calendar {
                months,
                days,
                micros,
            })
        };
        assert_eq!(cal(0, 0, 0), "0 seconds");
        assert_eq!(
            cal(14, 25, 18_367_500_000),
            "1 years 2 months 25 days 5 hours 6 minutes 7.5 seconds"
        );
        assert_eq!(cal(-13, 0, -500_000), "-1 years -1 months -0.5 seconds");
        assert_eq!(cal(0, 0, 5_000_000), "5 seconds");
    }

    #[test]
    fn nesting_nulls_and_map_order() {
        let a = Cell::Array(vec![Cell::Null, Cell::Str("b".into())]);
        assert_eq!(one(&a), "[null,\"b\"]");
        assert_eq!(one(&Cell::Null), "NULL");
        let s = Cell::Struct(vec![(
            "a".into(),
            Cell::Map(vec![(Cell::Str("k".into()), Cell::Int(1))]),
        )]);
        assert_eq!(one(&s), r#"{"a":{"k":1}}"#);
        // The rendered entries are sorted as strings: `10:` sorts before `2:`.
        let m = Cell::Map(vec![
            (Cell::Int(2), Cell::Bool(true)),
            (Cell::Int(10), Cell::Bool(false)),
        ]);
        assert_eq!(one(&m), "{10:false,2:true}");
    }

    #[test]
    fn binary_styles() {
        let b = vec![0xDE, 0xAD];
        assert_eq!(binary_text(&b, BinaryStyle::Hex), "DEAD");
        assert_eq!(binary_text(&b, BinaryStyle::HexDiscrete), "[DE AD]");
        assert_eq!(binary_text(&b, BinaryStyle::Basic), "[-34, -83]");
        assert_eq!(binary_text(b"Man", BinaryStyle::Base64), "TWFu");
        assert_eq!(binary_text(b"Ma", BinaryStyle::Base64), "TWE");
        assert_eq!(binary_text(b"M", BinaryStyle::Base64), "TQ");
        assert_eq!(binary_text(b"", BinaryStyle::Base64), "");
        assert_eq!(binary_text("héllo".as_bytes(), BinaryStyle::Utf8), "héllo");
    }

    /// Where `new String(bytes, UTF_8)` and Rust's lossy decoding differ, and the tails the JDK
    /// handles on their own.
    #[test]
    fn utf8_replacement_counts_follow_the_jdk() {
        let rep = |b: &[u8]| java_utf8(b).chars().filter(|c| *c == '\u{FFFD}').count();
        assert_eq!(rep(&[0xED, 0xA0, 0x80]), 1); // an encoded surrogate: one, not three
        assert_eq!(rep(&[0xF4, 0x90, 0x80, 0x80]), 4);
        assert_eq!(rep(&[0xC0, 0xAF]), 2);
        assert_eq!(rep(&[0xE0, 0x80, 0x80]), 3);
        assert_eq!(rep(&[0xE0, 0xA0]), 1); // truncated three-byte sequence
        assert_eq!(rep(&[0xE0, 0x41]), 1);
        assert_eq!(rep(&[0xE2, 0x82]), 1);
        assert_eq!(rep(&[0xF0, 0x90, 0x8D]), 1); // truncated four-byte sequence
        assert_eq!(rep(&[0xF0, 0x41]), 1);
        assert_eq!(rep(&[0xF0, 0x90, 0x41]), 1);
        assert_eq!(rep(&[0xF0]), 1);
        assert_eq!(rep(&[0xF5, 0x80, 0x80, 0x80]), 4);
        assert_eq!(rep(&[0xF8, 0x80]), 2);
        assert_eq!(rep(&[0xC3]), 1);
        assert_eq!(rep(&[0xC3, 0x28]), 1);
        assert_eq!(rep(&[0xF0, 0x41, 0x80, 0x80, 0x80]), 4);
        assert_eq!(rep(&[0xF0, 0x90, 0x80, 0x41, 0x41]), 1);
        assert_eq!(java_utf8(&[0xF0, 0x9F, 0x98, 0x80]), "\u{1F600}");
        assert_eq!(java_utf8(&[0xE2, 0x82, 0xAC]), "\u{20AC}");
    }

    #[test]
    fn session_formats() -> R {
        let cfg = |kv: &[(&str, &str)]| -> Vec<(String, String)> {
            kv.iter().map(|(k, v)| ((*k).into(), (*v).into())).collect()
        };
        // The default is the suite's zone: midnight UTC is the previous afternoon.
        let ts = Cell::Timestamp(0);
        let la = CellFormat::for_session(&[])?;
        assert_eq!(la.render(&ts, true), "1969-12-31 16:00:00");
        let later = cfg(&[
            ("spark.sql.session.timeZone", "UTC"),
            ("spark.sql.session.timeZone", "+05:30"),
            ("spark.sql.binaryOutputStyle", "hex"),
        ]);
        let f = CellFormat::for_session(&later)?;
        assert_eq!(f.render(&ts, true), "1970-01-01 05:30:00");
        assert_eq!(f.render(&Cell::Binary(vec![1, 255]), true), "01FF");
        assert!(
            CellFormat::for_session(&cfg(&[("spark.sql.session.timeZone", "Mars/Base")])).is_err()
        );
        let err = CellFormat::for_session(&cfg(&[("spark.sql.binaryOutputStyle", "rot13")]));
        assert_eq!(
            err.err(),
            Some(CellFormatError::BinaryStyle("rot13".into()))
        );
        Ok(())
    }

    #[test]
    fn zone_ids_follow_java() -> R {
        let at = |zone: &str| -> Result<String, CellFormatError> {
            Ok(CellFormat::new(zone, BinaryStyle::Utf8)?.render(&Cell::Timestamp(0), true))
        };
        assert_eq!(at("Z")?, "1970-01-01 00:00:00");
        assert_eq!(at("UTC")?, "1970-01-01 00:00:00");
        assert_eq!(at("GMT+1")?, "1970-01-01 01:00:00");
        assert_eq!(at("UTC-08:00")?, "1969-12-31 16:00:00");
        assert_eq!(at("UT+0530")?, "1970-01-01 05:30:00");
        assert_eq!(at("+9")?, "1970-01-01 09:00:00");
        assert_eq!(at("-01")?, "1969-12-31 23:00:00");
        assert_eq!(at("+10:00:30")?, "1970-01-01 10:00:30");
        assert_eq!(at("-103045")?, "1969-12-31 13:29:15");
        assert_eq!(at("Asia/Kolkata")?, "1970-01-01 05:30:00");
        assert_eq!(at("+18:00")?, "1970-01-01 18:00:00");
        assert_eq!(at("+5:30")?, "1970-01-01 05:30:00");
        assert_eq!(at("GMT-8:15")?, "1969-12-31 15:45:00");
        assert_eq!(at("PST")?, "1969-12-31 16:00:00");
        assert_eq!(at("EST")?, "1969-12-31 19:00:00");
        assert_eq!(at("IST")?, "1970-01-01 05:30:00");
        // Spark pads a one-digit hour and then a one-digit final minute.
        assert_eq!(at("+10:7")?, at("+10:07")?);
        assert_eq!(at("+1:3")?, at("+01:03")?);
        for bad in [
            "",
            "+",
            "+19",
            "+18:30",
            "+18:00:01",
            "+10::00",
            "+aa",
            "+10:60",
            "+1000000",
            "GMT+x",
            "Nowhere",
        ] {
            assert!(at(bad).is_err(), "{bad}");
        }
        Ok(())
    }

    #[test]
    fn instants_beyond_the_tz_database_fold_by_gregorian_cycles() -> R {
        let la = CellFormat::new("America/Los_Angeles", BinaryStyle::Utf8)?;
        let render = |us: i64| la.render(&Cell::Timestamp(us), true);
        // June is daylight time and January standard time, far before and after 9999.
        assert_eq!(render(2_000_000_000_000_000_000), "+65347-06-23 20:33:20");
        assert_eq!(render(-9_000_000_000_000_000_000), "-283229-05-10 00:07:02");
        Ok(())
    }

    #[test]
    fn zone_ids_get_both_spark_paddings() {
        assert_eq!(pad_hour("+5:30"), "+05:30");
        assert_eq!(pad_hour("+10:7"), "+10:07");
        assert_eq!(pad_hour("-5:7"), "-05:07");
        assert_eq!(pad_hour("GMT+10:7"), "GMT+10:07");
        assert_eq!(pad_hour("+10:07"), "+10:07");
        assert_eq!(
            pad_hour("+10:7:00"),
            "+10:7:00",
            "only a final single minute digit"
        );
        assert_eq!(pad_hour("America/Los_Angeles"), "America/Los_Angeles");
    }
}
