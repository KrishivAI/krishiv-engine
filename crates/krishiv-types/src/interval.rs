// SPDX-License-Identifier: Apache-2.0
//! Spark's text forms of interval values (`IntervalUtils.toYearMonthIntervalString`,
//! `toDayTimeIntervalString` in `ANSI_STYLE`, and `CalendarInterval.toString`).

use std::fmt::Write as _;

use crate::{DayTimeField, YearMonthField};

const MICROS_PER_SECOND: i64 = 1_000_000;
const MICROS_PER_MINUTE: i64 = 60 * MICROS_PER_SECOND;
const MICROS_PER_HOUR: i64 = 60 * MICROS_PER_MINUTE;

/// `START` or `START TO END`, upper-case, appended to `out`.
fn write_qualifier(out: &mut String, start: &str, end: &str) {
    out.extend(start.chars().map(|c| c.to_ascii_uppercase()));
    if start != end {
        out.push_str(" TO ");
        out.extend(end.chars().map(|c| c.to_ascii_uppercase()));
    }
}

/// `INTERVAL '-1-2' YEAR TO MONTH`: a year-month value in Spark's ANSI form.
pub fn year_month_string(months: i32, start: YearMonthField, end: YearMonthField) -> String {
    let mut out = String::new();
    write_year_month(&mut out, months, start, end);
    out
}

/// [`year_month_string`], appended to `out`.
pub fn write_year_month(out: &mut String, months: i32, start: YearMonthField, end: YearMonthField) {
    out.push_str("INTERVAL '");
    if months < 0 {
        out.push('-');
    }
    let m = months.unsigned_abs();
    match (start, end) {
        (YearMonthField::Year, YearMonthField::Month) => write!(out, "{}-{}", m / 12, m % 12),
        (YearMonthField::Year, YearMonthField::Year) => write!(out, "{}", m / 12),
        (YearMonthField::Month, _) => write!(out, "{m}"),
    }
    .ok();
    out.push_str("' ");
    write_qualifier(out, start.name(), end.name());
}

/// `INTERVAL '-1 02:03:04.5' DAY TO SECOND`: a day-time value in Spark's ANSI
/// form. The leading field is unbounded and, unless it is DAY, at least two
/// digits; the others stay in their range.
pub fn day_time_string(micros: i64, start: DayTimeField, end: DayTimeField) -> String {
    let mut out = String::new();
    write_day_time(&mut out, micros, start, end);
    out
}

/// [`day_time_string`], appended to `out`.
pub fn write_day_time(out: &mut String, micros: i64, start: DayTimeField, end: DayTimeField) {
    let units = [86_400, 3_600, 60, 1].map(|s: u64| s * MICROS_PER_SECOND.unsigned_abs());
    let (first, last) = (start as usize, end as usize);
    let mut rest = micros.unsigned_abs();
    out.push_str("INTERVAL '");
    if micros < 0 {
        out.push('-');
    }
    for (i, unit) in units.iter().enumerate().take(last + 1).skip(first) {
        let whole = rest / unit;
        rest %= unit;
        match i {
            0 => write!(out, "{whole}"),
            _ if i == first => write!(out, "{whole:02}"),
            1 => write!(out, " {whole:02}"),
            _ => write!(out, ":{whole:02}"),
        }
        .ok();
    }
    if end == DayTimeField::Second && rest > 0 {
        write!(out, ".{rest:06}").ok();
        out.truncate(out.trim_end_matches('0').len());
    }
    out.push_str("' ");
    write_qualifier(out, start.name(), end.name());
}

/// `CalendarInterval.toString`: the non-zero units, `0 seconds` for none,
/// e.g. `1 years 2 months 3 days 4.5 seconds`.
pub fn calendar_string(months: i32, days: i32, micros: i64) -> String {
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
    unit(micros / MICROS_PER_HOUR, "hours");
    unit(micros % MICROS_PER_HOUR / MICROS_PER_MINUTE, "minutes");
    let rest = micros % MICROS_PER_MINUTE;
    if rest != 0 {
        let sign = if rest < 0 { "-" } else { "" };
        let abs = rest.unsigned_abs();
        let (whole, fraction) = (
            abs / MICROS_PER_SECOND.unsigned_abs(),
            abs % MICROS_PER_SECOND.unsigned_abs(),
        );
        let fraction = format!("{fraction:06}");
        let fraction = fraction.trim_end_matches('0');
        let dot = if fraction.is_empty() { "" } else { "." };
        parts.push(format!("{sign}{whole}{dot}{fraction} seconds"));
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use DayTimeField::{Day, Hour, Minute, Second};
    use YearMonthField::{Month, Year};

    #[test]
    fn year_month_values_read_as_in_spark() {
        assert_eq!(year_month_string(12, Year, Year), "INTERVAL '1' YEAR");
        assert_eq!(year_month_string(12, Month, Month), "INTERVAL '12' MONTH");
        assert_eq!(
            year_month_string(-14, Year, Month),
            "INTERVAL '-1-2' YEAR TO MONTH"
        );
        assert_eq!(
            year_month_string(i32::MIN, Year, Month),
            "INTERVAL '-178956970-8' YEAR TO MONTH"
        );
    }

    #[test]
    fn a_leading_day_time_field_is_padded_unless_it_is_day() {
        let hour = MICROS_PER_HOUR;
        let minute = MICROS_PER_MINUTE;
        let second = MICROS_PER_SECOND;
        assert_eq!(day_time_string(hour, Hour, Hour), "INTERVAL '01' HOUR");
        assert_eq!(
            day_time_string(100 * hour, Hour, Hour),
            "INTERVAL '100' HOUR"
        );
        assert_eq!(
            day_time_string(hour + 2 * minute, Hour, Minute),
            "INTERVAL '01:02' HOUR TO MINUTE"
        );
        assert_eq!(
            day_time_string(minute + 2 * second + 300_000, Minute, Second),
            "INTERVAL '01:02.3' MINUTE TO SECOND"
        );
        assert_eq!(
            day_time_string(5 * second, Second, Second),
            "INTERVAL '05' SECOND"
        );
        assert_eq!(
            day_time_string(1_000, Second, Second),
            "INTERVAL '00.001' SECOND"
        );
        assert_eq!(
            day_time_string(-hour, Day, Hour),
            "INTERVAL '-0 01' DAY TO HOUR"
        );
        assert_eq!(day_time_string(0, Day, Day), "INTERVAL '0' DAY");
    }

    #[test]
    fn calendar_values_list_their_non_zero_units() {
        assert_eq!(calendar_string(0, 0, 0), "0 seconds");
        assert_eq!(calendar_string(14, 3, 0), "1 years 2 months 3 days");
        assert_eq!(calendar_string(0, 1, 0), "1 days");
        assert_eq!(
            calendar_string(0, 0, 3_600_000_000 + 60_000_000 + 1_500_000),
            "1 hours 1 minutes 1.5 seconds"
        );
        assert_eq!(calendar_string(0, 0, -500_000), "-0.5 seconds");
        assert_eq!(calendar_string(0, 0, 4_000_000), "4 seconds");
    }
}
