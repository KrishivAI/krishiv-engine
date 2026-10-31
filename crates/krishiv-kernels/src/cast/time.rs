// SPDX-License-Identifier: Apache-2.0
//! `TIME` text: Spark's `string -> TIME` grammar and `TIME -> string`.
//!
//! A `TIME` is carried as `Time64(Nanosecond)`, nanoseconds since midnight
//! (design P2-T181 section 2). Behaviour here is from Spark 4.2.0 probes:
//! surrounding whitespace and a leading `T` are accepted; hour and minute
//! are 1 or 2 digits, seconds are optional, a fraction of any length is
//! truncated to nanoseconds (`12:34:56.` is valid); an `AM`/`PM` suffix is
//! read (`01:02:03 PM` is 13:02:03); `24:00` and `60` minutes or seconds
//! are malformed.

/// Nanoseconds in a second.
pub(crate) const NANOS_PER_SECOND: i64 = 1_000_000_000;

/// Nanoseconds since midnight for `text`, or `None` when malformed.
pub(crate) fn parse_time(text: &str) -> Option<i64> {
    let s = text.trim_matches(|c: char| c.is_ascii_whitespace());
    let s = s.strip_prefix('T').unwrap_or(s);
    let (clock, meridiem) = match s.rsplit_once(' ') {
        Some((clock, m)) if m.eq_ignore_ascii_case("AM") => (clock.trim_end(), Some(false)),
        Some((clock, m)) if m.eq_ignore_ascii_case("PM") => (clock.trim_end(), Some(true)),
        _ => (s, None),
    };
    let field = |p: &str| -> Option<i64> {
        ((1..=2).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_digit()))
            .then(|| p.parse().ok())
            .flatten()
    };
    let mut parts = clock.split(':');
    let hour = parts.next().and_then(field)?;
    let minute = parts.next().and_then(field)?;
    let (second, nanos) = match parts.next() {
        None => (0, 0),
        Some(sec) => {
            let (whole, fraction) = sec.split_once('.').unwrap_or((sec, ""));
            if !fraction.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let mut nanos = 0_i64;
            for (i, d) in fraction.bytes().take(9).enumerate() {
                let scale = 10_i64.checked_pow(u32::try_from(8_usize.checked_sub(i)?).ok()?)?;
                nanos = nanos.checked_add(i64::from(d - b'0').checked_mul(scale)?)?;
            }
            (field(whole)?, nanos)
        }
    };
    if parts.next().is_some() || minute > 59 || second > 59 {
        return None;
    }
    let hour = match meridiem {
        None if hour < 24 => hour,
        Some(pm) if (1..=12).contains(&hour) => hour % 12 + if pm { 12 } else { 0 },
        _ => return None,
    };
    Some(((hour * 60 + minute) * 60 + second) * NANOS_PER_SECOND + nanos)
}

/// `HH:mm:ss` with the fraction's trailing zeros dropped, as Spark shows a
/// `TIME`.
pub(crate) fn format_time(nanos: i64) -> String {
    let secs = nanos.div_euclid(NANOS_PER_SECOND);
    let frac = nanos.rem_euclid(NANOS_PER_SECOND);
    let mut out = format!("{:02}:{:02}:{:02}", secs / 3600, secs / 60 % 60, secs % 60);
    if frac != 0 {
        let digits = format!("{frac:09}");
        out.push('.');
        out.push_str(digits.trim_end_matches('0'));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: i64 = NANOS_PER_SECOND;

    #[test]
    fn reads_what_spark_reads() {
        for (text, want) in [
            ("12:34", Some((12 * 60 + 34) * 60 * S)),
            (" 12:34 ", Some((12 * 60 + 34) * 60 * S)),
            ("T12:34", Some((12 * 60 + 34) * 60 * S)),
            ("1:2:3", Some(3723 * S)),
            ("12:34:56.", Some(45296 * S)),
            ("12:34:56.1234567", Some(45296 * S + 123_456_700)),
            ("12:34:56.1234567890", Some(45296 * S + 123_456_789)),
            ("01:02:03 PM", Some((13 * 3600 + 2 * 60 + 3) * S)),
            ("12:00 AM", Some(0)),
            ("12:34 PM", Some((12 * 60 + 34) * 60 * S)),
            ("12", None),
            ("-12:34", None),
            ("24:00", None),
            ("00:60", None),
            ("00:00:60", None),
            ("13:00 PM", None),
            ("00:00 AM", None),
            ("bad", None),
        ] {
            assert_eq!(parse_time(text), want, "{text:?}");
        }
    }

    #[test]
    fn shows_like_spark() {
        assert_eq!(format_time(0), "00:00:00");
        assert_eq!(format_time(45296 * S + 500_000_000), "12:34:56.5");
        assert_eq!(format_time(45296 * S + 789_123_000), "12:34:56.789123");
    }
}
