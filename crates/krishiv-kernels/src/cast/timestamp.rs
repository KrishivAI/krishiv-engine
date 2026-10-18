// SPDX-License-Identifier: Apache-2.0
//! Spark's `DateTimeUtils.stringToTimestamp` / `stringToTimestampWithoutTimeZone`
//! grammar: `[+-]yyyy[y{0,3}]`, `-[m]m`, `-[d]d`, then ` ` or `T` and `[h]h`,
//! `:[m]m`, `:[s]s`, `.fraction` (6 digits kept), and a zone right
//! after the seconds: `Z`, an offset (`+hh:mm`, `+hhmm`, `+hh`, `-h`),
//! `UTC`/`GMT`/`UT` with an optional offset, or a region id. Checked against
//! Spark 4.2 by the `cast_timestamp_differential` test.
#![expect(
    clippy::arithmetic_side_effects,
    reason = "digit accumulation is bounded by the digit-count limits; the microsecond \
              composition is checked"
)]

use super::datetime::{MICROS_PER_DAY, days_from_civil};
use super::text::trim_all;

/// The zone a timestamp string names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Zone {
    /// None: the session zone applies.
    Session,
    /// A fixed offset in seconds east of UTC (`Z` is 0).
    Offset(i32),
    /// A region id (`America/Los_Angeles`): needs the tz database.
    Region,
}

/// The outcome of parsing a string as a timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Parsed {
    /// The local date-time as microseconds since 1970-01-01 00:00:00, and
    /// the zone it is in.
    At(i64, Zone),
    /// A time of day without a date (`T10:00`, `10:00:00`): today's date in
    /// the session zone, which depends on the query clock. (`now`, `today`,
    /// `epoch` and the like are not timestamps in a column; Spark converts them
    /// in string literals only, which is the analyzer's job.)
    NeedsClock,
    /// Not a timestamp.
    Malformed,
}

const MICROS_PER_SECOND: i64 = 1_000_000;

/// Parses one string.
pub(crate) fn parse(s: &[u8]) -> Parsed {
    let s = trim_all(s);
    // A time of day without a date (`T10:00`, `10:00:00`) is today's.
    let first_non_digit = s.iter().position(|b| !b.is_ascii_digit());
    if s.first() == Some(&b'T') || first_non_digit.is_some_and(|i| i > 0 && s.get(i) == Some(&b':'))
    {
        return Parsed::NeedsClock;
    }
    datetime(s).unwrap_or(Parsed::Malformed)
}

/// A read position in the text.
struct Cursor<'a> {
    s: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn done(&self) -> bool {
        self.at >= self.s.len()
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.at).copied()
    }

    fn eat(&mut self, b: u8) -> bool {
        let hit = self.peek() == Some(b);
        self.at += usize::from(hit);
        hit
    }

    /// A run of `min..=max` digits, or `None` (also for a longer run).
    fn number(&mut self, min: usize, max: usize) -> Option<i64> {
        let run = self.s.get(self.at..)?;
        let n = run.iter().take_while(|b| b.is_ascii_digit()).count();
        if !(min..=max).contains(&n) {
            return None;
        }
        self.at += n;
        run.get(..n)?
            .iter()
            .try_fold(0_i64, |v, b| Some(v * 10 + i64::from(b - b'0')))
    }
}

fn datetime(s: &[u8]) -> Option<Parsed> {
    let mut c = Cursor { s, at: 0 };
    let sign = if c.eat(b'-') {
        -1
    } else {
        c.eat(b'+');
        1
    };
    let year = c.number(4, 7)?;
    let (mut month, mut day, mut parts) = (1, 1, 1);
    if c.eat(b'-') {
        month = c.number(1, 2)?;
        parts = 2;
        if c.eat(b'-') {
            day = c.number(1, 2)?;
            parts = 3;
        }
    }
    let days = days_from_civil(
        sign * year,
        u32::try_from(month).ok()?,
        u32::try_from(day).ok()?,
    )?;
    // i128: the date part alone may lie outside the range that date plus
    // time of day fits (`-290308-12-21 19:59:05.224192` is `i64::MIN`).
    let date = i128::from(days) * i128::from(MICROS_PER_DAY);
    let at_micros = |time: i128| i64::try_from(date + time).ok();
    if c.done() {
        return Some(Parsed::At(at_micros(0)?, Zone::Session));
    }
    // A time needs the whole date and a separator.
    if parts < 3 || !(c.eat(b' ') || c.eat(b'T')) {
        return None;
    }
    let mut time = i128::from(c.number(1, 2).filter(|h| *h <= 23)? * 3600 * MICROS_PER_SECOND);
    if !c.done() {
        if !c.eat(b':') {
            return None;
        }
        time += i128::from(c.number(1, 2).filter(|m| *m <= 59)? * 60 * MICROS_PER_SECOND);
        if !c.done() {
            if !c.eat(b':') {
                return None;
            }
            time += i128::from(c.number(1, 2).filter(|s| *s <= 59)? * MICROS_PER_SECOND);
            if c.eat(b'.') {
                time += i128::from(fraction(&mut c)?);
            }
        }
    }
    let zone = if c.done() {
        Zone::Session
    } else {
        // Only the seconds (or their fraction) may be followed by a zone.
        zone(c.s.get(c.at..)?)?
    };
    Some(Parsed::At(at_micros(time)?, zone))
}

/// The digits after `.` (any number; the first 6 are kept) as microseconds.
fn fraction(c: &mut Cursor<'_>) -> Option<i64> {
    let run = c.s.get(c.at..)?;
    let n = run.iter().take_while(|b| b.is_ascii_digit()).count();
    c.at += n;
    let kept = run.get(..n.min(6))?;
    let micros = kept.iter().fold(0_i64, |v, b| v * 10 + i64::from(b - b'0'));
    Some(micros * 10_i64.pow(u32::try_from(6 - kept.len()).ok()?))
}

/// The zone text after the seconds: `Z`, ` +05:30`, `UTC+8`, a region id.
fn zone(z: &[u8]) -> Option<Zone> {
    let z = z.trim_ascii();
    match z {
        [] => None,
        b"Z" | b"UTC" | b"GMT" | b"UT" => Some(Zone::Offset(0)),
        [b'+' | b'-', ..] => offset(z).map(Zone::Offset),
        _ => {
            for prefix in [&b"UTC"[..], b"GMT", b"UT"] {
                if let Some(rest) = z.strip_prefix(prefix)
                    && matches!(rest.first(), Some(b'+' | b'-'))
                {
                    return offset(rest).map(Zone::Offset);
                }
            }
            // Java region ids: a letter, then letters, digits and `~/._+-`.
            let region = z.first().is_some_and(u8::is_ascii_alphabetic)
                && z.len() > 1
                && z.iter().all(|b| {
                    b.is_ascii_alphanumeric()
                        || matches!(b, b'~' | b'/' | b'.' | b'_' | b'+' | b'-')
                });
            region.then_some(Zone::Region)
        }
    }
}

/// Java's `ZoneOffset.of`: `+h`, `+hh`, `+hh:mm`, `+hhmm`, `+hh:mm:ss`,
/// `+hhmmss`, within +-18:00. Seconds east of UTC.
fn offset(z: &[u8]) -> Option<i32> {
    let (sign, rest) = match z.split_first()? {
        (b'+', r) => (1_i32, r),
        (b'-', r) => (-1, r),
        _ => return None,
    };
    // Exactly `n` digits, or one when `lenient` (Spark pads `+h:` to `+0h:`
    // and a final `:m` to `:0m` before `ZoneOffset.of`).
    let digits = |d: &[u8], lenient: bool| -> Option<i32> {
        let ok = d.iter().all(u8::is_ascii_digit);
        match d {
            [a, b] if ok => Some(i32::from(a - b'0') * 10 + i32::from(b - b'0')),
            [a] if ok && lenient => Some(i32::from(a - b'0')),
            _ => None,
        }
    };
    let (h, m, s) = match rest {
        [h] if h.is_ascii_digit() => (i32::from(h - b'0'), 0, 0),
        [h @ .., b':', m1, m2, b':', s1, s2] => (
            digits(h, true)?,
            digits(&[*m1, *m2], false)?,
            digits(&[*s1, *s2], false)?,
        ),
        [h @ .., b':', m] if h.len() <= 2 && !h.contains(&b':') => {
            (digits(h, true)?, digits(&[*m], true)?, 0)
        }
        [h @ .., b':', m1, m2] => (digits(h, true)?, digits(&[*m1, *m2], false)?, 0),
        [h1, h2] => (digits(&[*h1, *h2], false)?, 0, 0),
        [h1, h2, m1, m2] => (digits(&[*h1, *h2], false)?, digits(&[*m1, *m2], false)?, 0),
        [h1, h2, m1, m2, s1, s2] => (
            digits(&[*h1, *h2], false)?,
            digits(&[*m1, *m2], false)?,
            digits(&[*s1, *s2], false)?,
        ),
        _ => return None,
    };
    let total = h * 3600 + m * 60 + s;
    (h <= 18 && m <= 59 && s <= 59 && total <= 18 * 3600).then_some(sign * total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(s: &str) -> Option<(i64, Zone)> {
        match parse(s.as_bytes()) {
            Parsed::At(m, z) => Some((m, z)),
            _ => None,
        }
    }

    #[test]
    fn grammar() {
        let day = 86_400 * MICROS_PER_SECOND;
        assert_eq!(at("1970-01-02"), Some((day, Zone::Session)));
        assert_eq!(
            at("1970-01-01 01:02:03.5Z"),
            Some((3_723_500_000, Zone::Offset(0)))
        );
        assert_eq!(at("1970-01-01T01"), Some((3_600_000_000, Zone::Session)));
        assert_eq!(
            at("1970-01-01 0:0:0 +05:30"),
            Some((0, Zone::Offset(19_800)))
        );
        assert_eq!(
            at("1970-01-01 00:00:00 America/Los_Angeles"),
            Some((0, Zone::Region))
        );
        assert_eq!(
            at("1970-01-01 00:00:00.123456789"),
            Some((123_456, Zone::Session))
        );
        assert_eq!(at("1970-1"), Some((0, Zone::Session)));
        for bad in [
            "",
            "abc",
            "1970-01 00:00",
            "1970-01-01 24:00:00",
            "1970-01-01 00:60",
            "1970-01-01 00:00:60",
            "1970-01-01 00:",
            "1970-01-01 00:00 Z",
            "1970-01-01 00:00:00 +19:00",
            "1970-01-01 00:00:00 z",
        ] {
            assert_eq!(at(bad), None, "{bad}");
        }
        assert_eq!(parse(b"T10:00"), Parsed::NeedsClock);
        assert_eq!(parse(b"10:00:00"), Parsed::NeedsClock);
        assert_eq!(parse(b"3:4"), Parsed::NeedsClock);
        for special in ["now", "today", "epoch", "yesterday", "tomorrow", "t10:00"] {
            assert_eq!(parse(special.as_bytes()), Parsed::Malformed, "{special}");
        }
    }

    #[test]
    fn offsets() {
        for (text, secs) in [
            ("+8", 8 * 3600),
            ("-8", -8 * 3600),
            ("+08", 8 * 3600),
            ("+0800", 8 * 3600),
            ("+08:30", 8 * 3600 + 1800),
            ("+08:30:15", 8 * 3600 + 1815),
            ("+083015", 8 * 3600 + 1815),
            ("-18:00", -18 * 3600),
            ("+8:00", 8 * 3600),
            ("+01:3", 3600 + 180),
            ("-1:3", -(3600 + 180)),
            ("+8:5", 8 * 3600 + 300),
            ("+8:30:15", 8 * 3600 + 1815),
        ] {
            assert_eq!(offset(text.as_bytes()), Some(secs), "{text}");
        }
        for bad in [
            "+19:00", "+18:01", "+8:5:6", "+08:30:5", "+08:60", "08:00", "+", "+080", "+12345",
        ] {
            assert_eq!(offset(bad.as_bytes()), None, "{bad}");
        }
        assert_eq!(zone(b" GMT+8"), Some(Zone::Offset(8 * 3600)));
        assert_eq!(zone(b"UTC-01:30"), Some(Zone::Offset(-5400)));
        assert_eq!(zone(b"Europe/Berlin"), Some(Zone::Region));
        assert_eq!(zone(b"1x"), None);
    }
}
