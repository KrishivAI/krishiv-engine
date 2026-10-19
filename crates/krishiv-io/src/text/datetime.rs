// SPDX-License-Identifier: Apache-2.0
//! Dates and timestamps as text for the text formats: the session time
//! zone, Spark's default grammars and the `dateFormat` / `timestampFormat` /
//! `timestampNTZFormat` patterns.

use krishiv_common::error::KrishivError;
use krishiv_kernels::cast::FixedOffset;
use krishiv_kernels::datetime_pattern::{DateTimePattern, Purpose, Zone, fixed_zone_offset};
use krishiv_kernels::text::{
    spark_string_to_date, spark_string_to_timestamp, spark_string_to_timestamp_ntz, today_in,
};

use crate::error::unsupported;
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
    rules: Result<FixedOffset, KrishivError>,
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

/// The offset of the zone `id`.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a region id (needs the tz database).
pub(crate) fn zone_rules(id: &str) -> Result<FixedOffset, KrishivError> {
    fixed_zone_offset(id).map(FixedOffset).ok_or_else(|| {
        unsupported(format!(
            "the time zone {id} (only UTC and fixed offsets are known)"
        ))
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
        let zone_id = o.str("timeZone").unwrap_or("UTC").to_owned();
        let rules = zone_rules(&zone_id);
        let pattern = |opt: &str, default: &str| -> Result<Option<DateTimePattern>, KrishivError> {
            match (o.str(opt), purpose) {
                (Some(p), _) => DateTimePattern::new(p, purpose).map(Some),
                (None, Purpose::Format) => DateTimePattern::new(default, purpose).map(Some),
                (None, Purpose::Parse) => Ok(None),
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

    fn rules(&self) -> Result<&FixedOffset, KrishivError> {
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
}
