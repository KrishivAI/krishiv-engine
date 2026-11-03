// SPDX-License-Identifier: Apache-2.0
//! The value of the interval in `SET TIME ZONE INTERVAL ...`, as Spark's `visitSetTimeZone` reads
//! it (`parseIntervalLiteral`, then the checks on months, days and microseconds).

use crate::ast::{Interval, IntervalValue};

const MICROS_PER_SECOND: i128 = 1_000_000;
const MICROS_PER_HOUR: i128 = 3_600_000_000;

/// An interval's parts, as Spark's `CalendarInterval` keeps them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Parts {
    pub(super) months: i128,
    pub(super) days: i128,
    pub(super) micros: i128,
}

/// What `SET TIME ZONE INTERVAL ...` does with an interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Outcome {
    /// The offset zone, as `ZoneOffset.toString` writes it.
    Zone(String),
    /// `INVALID_INTERVAL_FORMAT.TIMEZONE_INTERVAL_OUT_OF_RANGE` with this `input`.
    OutOfRange(String),
}

/// The parts of `interval`, or `None` for a form not read here (a trailing interval, a range or
/// unit this does not know, a malformed value).
pub(super) fn parts(interval: &Interval) -> Option<Parts> {
    if interval.trailing.is_some() {
        return None;
    }
    let mut out = Parts {
        months: 0,
        days: 0,
        micros: 0,
    };
    if let Some((value, from, to)) = &interval.range {
        add_range(&mut out, value, from, to)?;
    }
    for (value, unit) in &interval.units {
        let micros = decimal_micros(value)?;
        let whole = |m: i128| (m % MICROS_PER_SECOND == 0).then_some(m / MICROS_PER_SECOND);
        match unit.trim_end_matches('s') {
            "year" => out.months += 12 * whole(micros)?,
            "month" => out.months += whole(micros)?,
            "week" => out.days += 7 * whole(micros)?,
            "day" => out.days += whole(micros)?,
            "hour" => out.micros += whole(micros)? * MICROS_PER_HOUR,
            "minute" => out.micros += whole(micros)? * 60 * MICROS_PER_SECOND,
            "second" => out.micros += micros,
            "millisecond" => out.micros += whole(micros)? * 1000,
            "microsecond" => out.micros += whole(micros)?,
            _ => return None,
        }
    }
    Some(out)
}

/// The signed value of `v` times a million: `1`, `-2`, `1.5`.
fn decimal_micros(v: &IntervalValue) -> Option<i128> {
    let text = v.value.trim();
    let (whole, frac) = text.split_once('.').unwrap_or((text, ""));
    if whole.is_empty() && frac.is_empty()
        || !whole.bytes().all(|b| b.is_ascii_digit())
        || !frac.bytes().all(|b| b.is_ascii_digit())
        || frac.len() > 6
    {
        return None;
    }
    let w: i128 = if whole.is_empty() {
        0
    } else {
        whole.parse().ok()?
    };
    let f: i128 = if frac.is_empty() {
        0
    } else {
        format!("{frac:0<6}").parse().ok()?
    };
    let n = w.checked_mul(MICROS_PER_SECOND)?.checked_add(f)?;
    Some(if v.negative { -n } else { n })
}

/// `'text' FROM TO to` for the day-time ranges (`'19:40:32' HOUR TO SECOND`).
fn add_range(out: &mut Parts, v: &IntervalValue, from: &str, to: &str) -> Option<()> {
    if !v.is_string {
        return None;
    }
    let (negative, text) = match v.value.trim().strip_prefix('-') {
        Some(rest) => (!v.negative, rest),
        None => (v.negative, v.value.trim().trim_start_matches('+')),
    };
    let int = |s: &str| -> Option<i128> {
        (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse().ok())?
    };
    let secs = |s: &str| -> Option<i128> {
        let (w, f) = s.split_once('.').unwrap_or((s, ""));
        if f.len() > 6 || !f.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let frac: i128 = if f.is_empty() {
            0
        } else {
            format!("{f:0<6}").parse().ok()?
        };
        Some(int(w)? * MICROS_PER_SECOND + frac)
    };
    let micros = match (from, to) {
        ("hour", "second") => {
            let p: Vec<&str> = text.split(':').collect();
            let [h, m, s] = p.as_slice() else { return None };
            int(h)? * MICROS_PER_HOUR + int(m)? * 60 * MICROS_PER_SECOND + secs(s)?
        }
        ("hour", "minute") => {
            let (h, m) = text.split_once(':')?;
            int(h)? * MICROS_PER_HOUR + int(m)? * 60 * MICROS_PER_SECOND
        }
        ("minute", "second") => {
            let (m, s) = text.split_once(':')?;
            int(m)? * 60 * MICROS_PER_SECOND + secs(s)?
        }
        _ => return None,
    };
    out.micros += if negative { -micros } else { micros };
    Some(())
}

/// Spark's checks on an interval of `SET TIME ZONE INTERVAL`, in its order.
pub(super) fn outcome(p: Parts) -> Outcome {
    if p.months != 0 {
        return Outcome::OutOfRange(p.months.to_string());
    }
    if p.days != 0 {
        return Outcome::OutOfRange(p.days.to_string());
    }
    if p.micros.abs() > 18 * MICROS_PER_HOUR {
        return Outcome::OutOfRange((p.micros.abs() / MICROS_PER_HOUR).to_string());
    }
    if p.micros % MICROS_PER_SECOND != 0 {
        return Outcome::OutOfRange((p.micros / MICROS_PER_SECOND).to_string());
    }
    Outcome::Zone(offset_text(
        i64::try_from(p.micros / MICROS_PER_SECOND).unwrap_or(0),
    ))
}

/// `ZoneOffset.ofTotalSeconds(seconds).toString`: `Z`, `+hh:mm` or `+hh:mm:ss`.
fn offset_text(seconds: i64) -> String {
    if seconds == 0 {
        return "Z".to_owned();
    }
    let sign = if seconds < 0 { '-' } else { '+' };
    let a = seconds.abs();
    let (h, m, s) = (a / 3600, a / 60 % 60, a % 60);
    if s == 0 {
        format!("{sign}{h:02}:{m:02}")
    } else {
        format!("{sign}{h:02}:{m:02}:{s:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_print_as_java_prints_them() {
        assert_eq!(offset_text(0), "Z");
        assert_eq!(offset_text(36_000), "+10:00");
        assert_eq!(offset_text(-12_345), "-03:25:45");
    }

    #[test]
    fn spark_checks_run_in_order() {
        let p = |months, days, micros| {
            outcome(Parts {
                months,
                days,
                micros,
            })
        };
        assert_eq!(p(1, 1, 0), Outcome::OutOfRange("1".into()));
        assert_eq!(p(0, 3, 0), Outcome::OutOfRange("3".into()));
        assert_eq!(
            p(0, 0, 24 * MICROS_PER_HOUR),
            Outcome::OutOfRange("24".into())
        );
        assert_eq!(
            p(0, 0, 10 * MICROS_PER_HOUR + 1000),
            Outcome::OutOfRange("36000".into())
        );
        assert_eq!(
            p(0, 0, 18 * MICROS_PER_HOUR),
            Outcome::Zone("+18:00".into())
        );
    }
}
