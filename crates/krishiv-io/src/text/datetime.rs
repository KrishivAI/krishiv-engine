// SPDX-License-Identifier: Apache-2.0
//! Dates and timestamps as text for the text formats: the session time
//! zone, Spark's default grammars and the `dateFormat` / `timestampFormat` /
//! `timestampNTZFormat` patterns.

use krishiv_common::error::KrishivError;
use krishiv_kernels::datetime::{ZoneRef, resolve_zone, zone_id};
use krishiv_kernels::datetime_pattern::{DateTimePattern, Purpose, Zone};
use krishiv_kernels::text::{
    spark_string_to_date, spark_string_to_timestamp, spark_string_to_timestamp_ntz,
    split_timestamp_nanos, timestamp_nanos_of, timestamp_string_sub_micros, today_in,
};

use crate::options::FormatOptions;

/// Spark's default write patterns.
const DATE_PATTERN: &str = "yyyy-MM-dd";
const TIMESTAMP_PATTERN: &str = "yyyy-MM-dd'T'HH:mm:ss[.SSS][XXX]";
const NTZ_PATTERN: &str = "yyyy-MM-dd'T'HH:mm:ss[.SSS]";

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
                (None, Purpose::Format | Purpose::FormatTime) => {
                    DateTimePattern::new(default, purpose).map(Some)
                }
                (None, Purpose::Parse | Purpose::ParseNtz) => Ok(None),
            }
        };
        let date = pattern("dateFormat", DATE_PATTERN)?;
        let timestamp = pattern("timestampFormat", TIMESTAMP_PATTERN)?;
        let ntz = pattern("timestampNTZFormat", NTZ_PATTERN)?;
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

    /// A `TIMESTAMP_LTZ(precision)` (nanoseconds since the epoch) from text, as Spark 4.3's
    /// `parseNanos`: the pattern, else the default grammar; no fallback to the default grammar
    /// after a pattern fails. Digits past `precision` are floored.
    ///
    /// # Errors
    /// A region zone id in the text; `DATETIME_OVERFLOW` for an instant past the nanosecond
    /// carrier (1677-09-21 to 2262-04-11).
    pub(crate) fn parse_timestamp_nanos(
        &self,
        s: &str,
        precision: u8,
    ) -> Result<Option<i64>, KrishivError> {
        let rules = self.rules()?;
        let parsed = match &self.timestamp {
            Some(p) => p.parse_timestamp_nanos(s, rules)?,
            None => spark_string_to_timestamp(&clean(s), rules, self.today)?
                .map(|m| (m, timestamp_string_sub_micros(s))),
        };
        parsed.map(|v| nanos_of(v, precision)).transpose()
    }

    /// A `TIMESTAMP_NTZ(precision)` (local nanoseconds) from text, as Spark 4.3's
    /// `parseWithoutTimeZoneNanos(s, precision, false)`; digits past `precision` are floored.
    ///
    /// # Errors
    /// A region zone id in the text; `DATETIME_OVERFLOW` past the nanosecond carrier.
    pub(crate) fn parse_ntz_nanos(
        &self,
        s: &str,
        precision: u8,
    ) -> Result<Option<i64>, KrishivError> {
        let parsed = match &self.ntz {
            Some(p) => p.parse_local_nanos(s, false)?,
            None => spark_string_to_timestamp_ntz(s).map(|m| (m, timestamp_string_sub_micros(s))),
        };
        parsed.map(|v| nanos_of(v, precision)).transpose()
    }

    /// `nanos` (a `TIMESTAMP_LTZ(precision)` instant) as text in the session zone, as Spark 4.3's
    /// `formatNanos`: floored to `precision`, then the pattern prints as many digits as it has.
    ///
    /// # Errors
    /// A value the pattern cannot show.
    pub(crate) fn format_timestamp_nanos(
        &self,
        nanos: i64,
        precision: u8,
    ) -> Result<String, KrishivError> {
        let (micros, rest) = split_timestamp_nanos(nanos, precision);
        match &self.timestamp {
            Some(p) => p.format_timestamp_nanos(micros, rest, self.zone()?),
            None => Err(crate::error::internal("a timestamp pattern for writing")),
        }
    }

    /// `nanos` (a `TIMESTAMP_NTZ(precision)` local date-time) as text, as Spark 4.3's
    /// `formatWithoutTimeZoneNanos`.
    ///
    /// # Errors
    /// A value the pattern cannot show.
    pub(crate) fn format_ntz_nanos(
        &self,
        nanos: i64,
        precision: u8,
    ) -> Result<String, KrishivError> {
        let (micros, rest) = split_timestamp_nanos(nanos, precision);
        match &self.ntz {
            Some(p) => p.format_local_nanos(micros, rest),
            None => Err(crate::error::internal("a timestamp pattern for writing")),
        }
    }
}

/// Microseconds and the nanoseconds within them as one nanosecond value floored to `precision`.
///
/// # Errors
/// `DATETIME_OVERFLOW` past the `i64` nanosecond carrier.
fn nanos_of((micros, rest): (i64, u16), precision: u8) -> Result<i64, KrishivError> {
    timestamp_nanos_of(micros, rest, precision).ok_or_else(|| {
        krishiv_common::error::DatetimeOverflow {
            operation: "read a nanosecond timestamp from text".to_owned(),
        }
        .into()
    })
}
