// SPDX-License-Identifier: Apache-2.0
//! Dates and timestamps as text for the text formats: the session time
//! zone, Spark's default grammars and the `dateFormat` / `timestampFormat` /
//! `timestampNTZFormat` / `timeFormat` patterns.

use krishiv_common::error::KrishivError;
use krishiv_kernels::cast::{format_time, parse_time};
use krishiv_kernels::datetime::{ZoneRef, resolve_zone, zone_id};
use krishiv_kernels::datetime_pattern::{DateTimePattern, Purpose, Zone};
use krishiv_kernels::text::{
    spark_string_to_date, spark_string_to_timestamp, spark_string_to_timestamp_ntz, today_in,
};

use crate::options::FormatOptions;

/// Spark's default write patterns.
const DATE_PATTERN: &str = "yyyy-MM-dd";
const TIMESTAMP_PATTERN: &str = "yyyy-MM-dd'T'HH:mm:ss[.SSS][XXX]";
const NTZ_PATTERN: &str = "yyyy-MM-dd'T'HH:mm:ss[.SSS]";

/// Microseconds in a day, and nanoseconds in a microsecond.
const MICROS_PER_DAY: i64 = 86_400_000_000;
const NANOS_PER_MICRO: i64 = 1_000;

/// Dates and timestamps as text in one session time zone.
#[derive(Debug)]
pub(crate) struct Datetimes {
    zone_id: String,
    /// The session zone; an unsupported one fails only the conversions
    /// that need it, so a read without timestamps still works.
    rules: Result<ZoneRef, KrishivError>,
    date: Option<DateTimePattern>,
    timestamp: Option<DateTimePattern>,
    ntz: Option<DateTimePattern>,
    /// `timeFormat`: unset reads Spark's `TIME` grammar and writes the value's
    /// own digits (Spark 4.3.0, `tests/fixtures/spark-4.3.0/time-file-reads.json`).
    /// Spark builds the formatter lazily, so a pattern that does not compile
    /// fails each value it meets (a malformed record on read, the write on
    /// writing), never the reader itself; the error is kept for that.
    time: Option<Result<DateTimePattern, KrishivError>>,
    /// Today in the session zone, for a bare time of day.
    today: i32,
    fallback_date: bool,
    fallback_timestamp: bool,
}

/// Spark's `cleanLegacyTimestampStr`: the legacy fallback drops `GMT`.
fn clean(s: &str) -> std::borrow::Cow<'_, str> {
    if s.contains("GMT") {
        std::borrow::Cow::Owned(s.replace("GMT", ""))
    } else {
        std::borrow::Cow::Borrowed(s)
    }
}

/// `compiled` as a `TIME` pattern. Writing: java.time's `LocalTime` has only
/// the time-of-day fields, so a pattern with any other letter (a year, a day,
/// a zone) fails to format; reading resolves a date-time to its time of day.
/// Known difference: a pattern with a field Spark's timestamp checks refuse
/// (`Q`, `E`, `F`, ...) fails here as a whole, where java.time could still
/// parse a time from some texts (the fixture's `qqq HH` parses nothing).
fn time_pattern(
    text: &str,
    compiled: DateTimePattern,
    purpose: Purpose,
) -> Result<DateTimePattern, KrishivError> {
    if matches!(purpose, Purpose::Format)
        && let Some(c) = unquoted_letters(text).find(|c| !"HhKkmsSnNAa".contains(*c))
    {
        return Err(crate::error::unsupported(format!(
            "a TIME written with the timeFormat letter '{c}'"
        )));
    }
    Ok(compiled)
}

/// The pattern letters of `text` outside quotes.
fn unquoted_letters(text: &str) -> impl Iterator<Item = char> + '_ {
    let mut quoted = false;
    text.chars().filter(move |c| {
        if *c == '\'' {
            quoted = !quoted;
            return false;
        }
        !quoted && c.is_ascii_alphabetic()
    })
}

impl Datetimes {
    /// The conversions of `o`: its `timeZone` and patterns. For reading
    /// ([`Purpose::Parse`]) an unset pattern means Spark's lenient default
    /// grammar; for writing the default patterns are used.
    ///
    /// # Errors
    /// An invalid pattern or time zone.
    pub(crate) fn new(o: &FormatOptions, purpose: Purpose) -> Result<Self, KrishivError> {
        let configured = o.str("timeZone").unwrap_or("UTC");
        let rules = resolve_zone(configured);
        let zone_id = zone_id(configured);
        let pattern = |opt: &str, default: &str| -> Result<Option<DateTimePattern>, KrishivError> {
            match (o.str(opt), purpose) {
                (Some(p), _) => DateTimePattern::new(p, purpose).map(Some),
                (None, Purpose::Format) => DateTimePattern::new(default, purpose).map(Some),
                (None, Purpose::Parse | Purpose::ParseNtz) => Ok(None),
            }
        };
        let date = pattern("dateFormat", DATE_PATTERN)?;
        let timestamp = pattern("timestampFormat", TIMESTAMP_PATTERN)?;
        let ntz = pattern("timestampNTZFormat", NTZ_PATTERN)?;
        let time = o
            .str("timeFormat")
            .map(|p| DateTimePattern::new(p, purpose).and_then(|c| time_pattern(p, c, purpose)));
        let explicit = o.bool("enableDateTimeParsingFallback");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_micros()).ok())
            .unwrap_or(0);
        Ok(Self {
            fallback_date: explicit.unwrap_or(date.is_none()),
            fallback_timestamp: explicit.unwrap_or(timestamp.is_none() && ntz.is_none()),
            today: rules.as_ref().map_or(0, |r| today_in(now, r)),
            zone_id,
            rules,
            date,
            timestamp,
            ntz,
            time,
        })
    }

    fn rules(&self) -> Result<&ZoneRef, KrishivError> {
        self.rules.as_ref().map_err(Clone::clone)
    }

    fn zone(&self) -> Result<Zone<'_>, KrishivError> {
        Ok(Zone {
            id: &self.zone_id,
            rules: self.rules()?,
        })
    }

    /// A `DATE` from text; `Ok(None)` when it does not parse.
    ///
    /// # Errors
    /// A region zone id in the text.
    pub(crate) fn parse_date(&self, s: &str) -> Result<Option<i32>, KrishivError> {
        let parsed = match &self.date {
            Some(p) => p.parse_date(s)?,
            None => spark_string_to_date(&clean(s)),
        };
        match parsed {
            None if self.fallback_date && self.date.is_some() => {
                Ok(spark_string_to_date(&clean(s)))
            }
            other => Ok(other),
        }
    }

    /// A `TIMESTAMP` (microseconds since the epoch) from text.
    ///
    /// # Errors
    /// A region zone id in the text.
    pub(crate) fn parse_timestamp(&self, s: &str) -> Result<Option<i64>, KrishivError> {
        let rules = self.rules()?;
        let default = || spark_string_to_timestamp(&clean(s), rules, self.today);
        let parsed = match &self.timestamp {
            Some(p) => p.parse_timestamp(s, rules)?,
            None => default()?,
        };
        match parsed {
            None if self.fallback_timestamp && self.timestamp.is_some() => default(),
            other => Ok(other),
        }
    }

    /// A `TIMESTAMP_NTZ` (local microseconds) from text.
    ///
    /// # Errors
    /// A region zone id in the text.
    pub(crate) fn parse_ntz(&self, s: &str) -> Result<Option<i64>, KrishivError> {
        let parsed = match &self.ntz {
            Some(p) => p.parse_local(s, false)?,
            None => spark_string_to_timestamp_ntz(s),
        };
        match parsed {
            None if self.fallback_timestamp && self.ntz.is_some() => {
                Ok(spark_string_to_timestamp_ntz(s))
            }
            other => Ok(other),
        }
    }

    /// A `TIME` (nanoseconds since midnight) from text: `timeFormat` exactly
    /// when set (no fallback to the default grammar), else Spark's `TIME`
    /// grammar, as `CAST(text AS TIME)`.
    ///
    /// # Errors
    /// A text the pattern refuses with an error of its own.
    pub(crate) fn parse_time(&self, s: &str) -> Result<Option<i64>, KrishivError> {
        match &self.time {
            // Spark's formatter fails to build on this value: not a time.
            Some(Err(_)) => Ok(None),
            Some(Ok(p)) => Ok(p
                .parse_local(s, false)?
                .map(|m| m.rem_euclid(MICROS_PER_DAY).wrapping_mul(NANOS_PER_MICRO))),
            None => Ok(parse_time(s)),
        }
    }

    /// `nanos` (a `TIME`) as text: `timeFormat` when set, else `HH:mm:ss` with
    /// the fraction's own digits.
    ///
    /// # Errors
    /// A `timeFormat` that does not compile or names a field a time of day
    /// has not (a date, a zone): Spark fails the write at the first non-NULL
    /// `TIME` (`TASK_WRITE_FAILED`).
    pub(crate) fn format_time(&self, nanos: i64) -> Result<String, KrishivError> {
        match &self.time {
            Some(Err(e)) => Err(e.clone()),
            Some(Ok(p)) => p.format_local(nanos.div_euclid(NANOS_PER_MICRO)),
            None => Ok(format_time(nanos)),
        }
    }

    /// `days` as text.
    ///
    /// # Errors
    /// A value the pattern cannot show.
    pub(crate) fn format_date(&self, days: i32) -> Result<String, KrishivError> {
        match &self.date {
            Some(p) => p.format_date(days),
            None => Err(crate::error::internal("a date pattern for writing")),
        }
    }

    /// `micros` (an instant) as text in the session zone.
    ///
    /// # Errors
    /// A value the pattern cannot show.
    pub(crate) fn format_timestamp(&self, micros: i64) -> Result<String, KrishivError> {
        match &self.timestamp {
            Some(p) => p.format_timestamp(micros, self.zone()?),
            None => Err(crate::error::internal("a timestamp pattern for writing")),
        }
    }

    /// A local date-time as text.
    ///
    /// # Errors
    /// A value the pattern cannot show.
    pub(crate) fn format_ntz(&self, local: i64) -> Result<String, KrishivError> {
        match &self.ntz {
            Some(p) => p.format_local(local),
            None => Err(crate::error::internal("a timestamp pattern for writing")),
        }
    }
}
