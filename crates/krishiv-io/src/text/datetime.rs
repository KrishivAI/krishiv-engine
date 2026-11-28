// SPDX-License-Identifier: Apache-2.0
//! Dates and timestamps as text for the text formats: the session time
//! zone, Spark's default grammars and the `dateFormat` / `timestampFormat` /
//! `timestampNTZFormat` / `timeFormat` patterns.

use krishiv_common::error::KrishivError;
use krishiv_kernels::cast::{format_time, parse_time, parse_time_nanos};
use krishiv_kernels::datetime::{ZoneRef, resolve_zone, zone_id};
use krishiv_kernels::datetime_pattern::{DateTimePattern, Purpose, Zone};
use krishiv_kernels::text::{
    spark_string_to_date, spark_string_to_timestamp, spark_string_to_timestamp_ntz, today_in,
};

use crate::options::FormatOptions;

/// Spark's default write patterns.
const DATE_PATTERN: &str = "yyyy-MM-dd";
const TIMESTAMP_PATTERN: &str = "yyyy-MM-dd'T'HH:mm:ss[.SSS][XXXXX]";
/// [`TIMESTAMP_PATTERN`] without an offset's seconds (Spark 3.5, or
/// `spark.sql.files.supportSecondOffsetFormat` off).
const TIMESTAMP_PATTERN_MINUTES: &str = "yyyy-MM-dd'T'HH:mm:ss[.SSS][XXX]";
const NTZ_PATTERN: &str = "yyyy-MM-dd'T'HH:mm:ss[.SSS]";
/// The `FastDateFormat` patterns of dates and timestamps under LEGACY (read
/// and write defaults alike).
const LEGACY_DATE: &str = "yyyy-MM-dd";
const LEGACY_TIMESTAMP: &str = "yyyy-MM-dd'T'HH:mm:ss.SSSXXX";

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
    /// A pattern with a field Spark's timestamp checks refuse (`Q`, `E`,
    /// `F`, ...) fails every value, as in Spark (4.3.0 run: `E HH:mm:ss`,
    /// `QQ HH:mm:ss`, `F HH:mm:ss` read `Mon 12:34:56` and the like as NULL;
    /// the fixture's `qqq HH` parses nothing).
    time: Option<Result<DateTimePattern, KrishivError>>,
    /// A `TIME` read in the default grammar keeps six fraction digits, not
    /// nine (the Spark 4.2 profile, `krishiv.io.text.timeNanos` off).
    time_micros_only: bool,
    /// Today in the session zone, for a bare time of day.
    today: i32,
    fallback_date: bool,
    fallback_timestamp: bool,
    /// The fallback of `TIMESTAMP_NTZ` reads (the parser policy does not
    /// turn it on).
    fallback_ntz: bool,
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
    /// The conversions of `o` for the JSON reader and writer: its `timeZone`
    /// and patterns. For reading ([`Purpose::Parse`]) an unset pattern means
    /// Spark's lenient default grammar (`JSONOptions.dateFormatInRead` is the
    /// option alone); for writing the default patterns are used.
    ///
    /// # Errors
    /// An invalid pattern or time zone.
    pub(crate) fn new(o: &FormatOptions, purpose: Purpose) -> Result<Self, KrishivError> {
        Self::build(o, purpose, false)
    }

    /// [`Datetimes::new`] for the CSV reader and writer. Under `preferDate`
    /// (default on, off under LEGACY) a read without `dateFormat` takes the
    /// strict `yyyy-MM-dd` (`CSVOptions.dateFormatInRead`), so with the
    /// fallback turned off `2024-01-02 03:04:05` is NULL; with `preferDate`
    /// off dates read leniently as in JSON (Spark 4.2.1 and 4.3.0 runs,
    /// `time-text-profiles.json`).
    ///
    /// # Errors
    /// An invalid pattern or time zone.
    pub(crate) fn csv(o: &FormatOptions, purpose: Purpose) -> Result<Self, KrishivError> {
        Self::build(o, purpose, crate::text::prefer_date(o)?)
    }

    fn build(
        o: &FormatOptions,
        purpose: Purpose,
        strict_date_default: bool,
    ) -> Result<Self, KrishivError> {
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
        // Under LEGACY (the CSV and JSON options carry the policy) dates and
        // timestamps take `FastDateFormat` patterns, Spark's legacy defaults
        // when none is named, for reading and writing alike
        // (`JSONOptions`/`CSVOptions` `dateFormatInRead`, `timestampFormatInRead`,
        // `commonTimestampFormat`; the generators build `FAST_DATE_FORMAT`
        // formatters). `TIMESTAMP_NTZ` keeps the `java.time` pattern. The
        // generators build their formatters eagerly, so a pattern that does
        // not compile fails the write, never one value.
        let legacy = crate::text::legacy_parser(o);
        let reading = matches!(purpose, Purpose::Parse | Purpose::ParseNtz);
        let (date, timestamp) = if legacy {
            let fast = |opt: &str, default: &str| -> Result<DateTimePattern, KrishivError> {
                let p = DateTimePattern::new_fast(o.str(opt).unwrap_or(default));
                match p.build_error() {
                    Some(e) if !reading => Err(e),
                    _ => Ok(p),
                }
            };
            (
                Some(fast("dateFormat", LEGACY_DATE)?),
                Some(fast("timestampFormat", LEGACY_TIMESTAMP)?),
            )
        } else {
            (
                pattern("dateFormat", DATE_PATTERN)?,
                pattern(
                    "timestampFormat",
                    if o.bool(crate::text::SECOND_OFFSETS_CONF).unwrap_or(true) {
                        TIMESTAMP_PATTERN
                    } else {
                        TIMESTAMP_PATTERN_MINUTES
                    },
                )?,
            )
        };
        let ntz = pattern("timestampNTZFormat", NTZ_PATTERN)?;
        // Writing, `HH:mm:ss` is Spark's fraction formatter (UnivocityGenerator,
        // JacksonGenerator), as with no `timeFormat`; reading keeps it exact.
        let time = o
            .str("timeFormat")
            .filter(|p| {
                !(matches!(purpose, Purpose::Format | Purpose::FormatTime)
                    && *p == krishiv_kernels::datetime_pattern::DEFAULT_TIME_PATTERN)
            })
            .map(|p| DateTimePattern::new(p, purpose));
        let explicit = o.bool("enableDateTimeParsingFallback");
        // CSV under `preferDate` reads `yyyy-MM-dd` when no `dateFormat` is
        // named; the fallback (on unless named) makes that lenient again.
        let date = match (date, purpose) {
            (None, Purpose::Parse | Purpose::ParseNtz) if strict_date_default => {
                Some(DateTimePattern::new(DATE_PATTERN, purpose)?)
            }
            (date, _) => date,
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_micros()).ok())
            .unwrap_or(0);
        Ok(Self {
            // `JacksonParser` and `UnivocityParser`: the fallback is on under
            // LEGACY or without a `dateFormat` option, unless named.
            fallback_date: explicit.unwrap_or(legacy || o.str("dateFormat").is_none()),
            fallback_timestamp: explicit
                .unwrap_or(legacy || (timestamp.is_none() && ntz.is_none())),
            fallback_ntz: explicit.unwrap_or(timestamp.is_none() && ntz.is_none()),
            today: rules.as_ref().map_or(0, |r| today_in(now, r)),
            zone_id,
            rules,
            date,
            timestamp,
            ntz,
            time,
            time_micros_only: !o.bool(crate::text::TIME_NANOS_CONF).unwrap_or(true),
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

    /// A `TIMESTAMP` under `timestampFormat` alone (the default grammar
    /// when unset), without the fallback: Spark's inference
    /// (`timestampFormatter.parseOptional`).
    ///
    /// # Errors
    /// A region zone id in the text.
    pub(crate) fn parse_timestamp_exact(&self, s: &str) -> Result<Option<i64>, KrishivError> {
        let rules = self.rules()?;
        match &self.timestamp {
            Some(p) => p.parse_timestamp(s, rules),
            None => spark_string_to_timestamp(&clean(s), rules, self.today),
        }
    }

    /// A `TIMESTAMP_NTZ` under `timestampNTZFormat` alone (the zone-less
    /// grammar when unset), without the fallback: Spark's inference
    /// (`parseWithoutTimeZoneOptional(text, false)`).
    ///
    /// # Errors
    /// A text the pattern refuses with an error of its own.
    pub(crate) fn parse_ntz_exact(&self, s: &str) -> Result<Option<i64>, KrishivError> {
        match &self.ntz {
            Some(p) => p.parse_local(s, false),
            None => Ok(spark_string_to_timestamp_ntz(s)),
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
            None if self.fallback_ntz && self.ntz.is_some() => Ok(spark_string_to_timestamp_ntz(s)),
            other => Ok(other),
        }
    }

    /// A `TIME` (nanoseconds since midnight) from text: `timeFormat` exactly
    /// when set (no fallback to the default grammar), else Spark's `TIME`
    /// grammar. Up to nine fraction digits are kept whatever the column's
    /// precision (Spark 4.3.0: a `TIME(3)` column reads `12:34:56.123456789`);
    /// the default grammar keeps six under the Spark 4.2 profile
    /// (`12:34:56.123456`), a pattern still nine (Spark 4.2.1 run).
    ///
    /// # Errors
    /// A text the pattern refuses with an error of its own.
    pub(crate) fn parse_time(&self, s: &str) -> Result<Option<i64>, KrishivError> {
        match &self.time {
            // Spark's formatter fails to build on this value: not a time.
            Some(Err(_)) => Ok(None),
            // Every digit the pattern reads, under the Spark 4.2 profile too
            // (Spark 4.2.1 run, `tests/fixtures/spark-4.2.1/time-text-profiles.json`:
            // `HH:mm:ss.SSSSSSSSS` reads nine digits into `TIME(6)`).
            Some(Ok(p)) => p.parse_time_of_day_nanos(s),
            None if self.time_micros_only => Ok(parse_time(s)),
            None => Ok(parse_time_nanos(s)),
        }
    }

    /// `nanos` (a `TIME`) as text: `timeFormat` when set, else `HH:mm:ss` with
    /// the fraction's own digits.
    ///
    /// # Errors
    /// A `timeFormat` that does not compile, or names a field a time of day
    /// has not (a date, a zone) outside an optional section: Spark fails the
    /// write at the first non-NULL `TIME` (`TASK_WRITE_FAILED`). Every
    /// fraction digit is kept (`SSSSSSSSS` prints nine).
    pub(crate) fn format_time(&self, nanos: i64) -> Result<String, KrishivError> {
        match &self.time {
            Some(Err(e)) => Err(e.clone()),
            Some(Ok(p)) => p.format_time_of_day(nanos)?.ok_or_else(|| {
                crate::error::unsupported(
                    "a TIME written with a timeFormat field a time of day has not (a date or a zone)",
                )
            }),
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
