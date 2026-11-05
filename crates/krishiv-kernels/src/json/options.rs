// SPDX-License-Identifier: Apache-2.0
//! The JSON options of `from_json`, `to_json` and `schema_of_json`: the
//! option names of Spark's JSON data source (`JSONOptions`).
//!
//! Only the options a scalar function can observe are honoured: the three
//! date-time patterns, the inference flags (`primitivesAsString`,
//! `prefersDecimal`), the leniency flags (`allowComments`,
//! `allowSingleQuotes`, `allowNumericLeadingZeros`,
//! `allowBackslashEscapingAnyCharacter`, `allowUnquotedFieldNames`,
//! `allowUnquotedControlChars`, `allowNonNumericNumbers`), `sortKeys` and
//! `ignoreNullFields` (two `to_json` writer flags), `dropFieldIfAllNull` (a
//! `schema_of_json` rendering flag) and `mode` (`PERMISSIVE` or `FAILFAST`
//! for `from_json`; `DROPMALFORMED` is rejected with
//! `PARSE_MODE_UNSUPPORTED`). Unknown options are ignored, as Spark's
//! reader ignores them, and the file options (`multiLine`,
//! `columnNameOfCorruptRecord`, `samplingRatio`, `locale`, `timeFormat`,
//! ...) have no scalar meaning and are ignored.

use std::collections::BTreeMap;

use krishiv_common::error::KrishivError;

use crate::datetime_pattern::{DateTimePattern, Purpose};

/// How `from_json` handles malformed records (Spark's `mode` option).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParseMode {
    /// Return `NULL` for a malformed record (Spark's default).
    #[default]
    Permissive,
    /// Fail the query with `MALFORMED_RECORD_IN_PARSING`.
    FailFast,
    /// Drop malformed records (files only): `from_json` rejects this with
    /// `PARSE_MODE_UNSUPPORTED`.
    DropMalformed,
}

/// The parsed options of one JSON function call.
#[derive(Debug, Clone, Default)]
pub struct JsonOptions {
    /// `dateFormat`, compiled for parsing (a `from_json` reader flag).
    pub date_format: Option<DateTimePattern>,
    /// `timestampFormat`, compiled for parsing (a `from_json` reader flag).
    pub timestamp_format: Option<DateTimePattern>,
    /// `timestampNTZFormat`, compiled for parsing (a `from_json` reader flag).
    pub timestamp_ntz_format: Option<DateTimePattern>,
    /// `timeFormat`, compiled for parsing (a `from_json` reader flag).
    pub time_format: Option<DateTimePattern>,
    /// `timeFormat`, compiled for formatting (a `to_json` writer flag).
    pub time_output: Option<DateTimePattern>,
    /// `dateFormat`, compiled for formatting (a `to_json` writer flag).
    pub date_output: Option<DateTimePattern>,
    /// `timestampFormat`, compiled for formatting (a `to_json` writer flag).
    pub timestamp_output: Option<DateTimePattern>,
    /// `timestampNTZFormat`, compiled for formatting (a `to_json` writer flag).
    pub timestamp_ntz_output: Option<DateTimePattern>,
    /// `primitivesAsString` (a `schema_of_json` inference flag).
    pub primitives_as_string: bool,
    /// `prefersDecimal` (a `schema_of_json` inference flag).
    pub prefers_decimal: bool,
    /// `allowComments`.
    pub allow_comments: bool,
    /// `allowSingleQuotes`.
    pub allow_single_quotes: bool,
    /// `allowNumericLeadingZeros`.
    pub allow_numeric_leading_zeros: bool,
    /// `allowBackslashEscapingAnyCharacter`.
    pub allow_backslash_escaping_any: bool,
    /// `allowUnquotedFieldNames`.
    pub allow_unquoted_field_names: bool,
    /// `allowUnquotedControlChars`.
    pub allow_unquoted_control_chars: bool,
    /// `allowNonNumericNumbers` (Spark's default): the `NaN`, `Infinity`
    /// and `INF` spellings, signed, read as floating-point numbers.
    pub allow_non_numeric_numbers: bool,
    /// `sortKeys` (a `to_json` writer flag).
    pub sort_keys: bool,
    /// `ignoreNullFields` (a `to_json` writer flag, Spark's default):
    /// struct fields that are `NULL` are not written.
    pub ignore_null_fields: bool,
    /// `dropFieldIfAllNull` (a `schema_of_json` rendering flag): struct
    /// fields that infer as `VOID` are dropped instead of rendering as
    /// `STRING`.
    pub drop_field_if_all_null: bool,
    /// `mode` (a `from_json` reader flag).
    pub mode: ParseMode,
    /// `enableDateTimeParsingFallback`: a date or timestamp the pattern does
    /// not read is read again by the lenient default grammar. Spark's default
    /// is on under `timeParserPolicy=LEGACY` and when the option names no
    /// pattern, off otherwise.
    pub datetime_fallback: bool,
}

impl JsonOptions {
    /// The defaults (no options argument).
    #[must_use]
    pub fn new() -> Self {
        Self {
            allow_non_numeric_numbers: true,
            ignore_null_fields: true,
            ..Self::default()
        }
    }

    /// Parses `options` (option name to value, as Spark writes a
    /// `MAP<STRING, STRING>` literal).
    ///
    /// # Errors
    /// Spark's pattern error for a `dateFormat`, `timestampFormat` or
    /// `timestampNTZFormat` value Spark rejects. Unknown options and the
    /// file-only options are ignored.
    pub fn from_map(options: &BTreeMap<String, String>) -> Result<Self, KrishivError> {
        Self::from_map_with(options, false)
    }

    /// [`Self::from_map`], with the date and zoned timestamp patterns
    /// compiled as Spark's `FastDateFormat` ones (and its default patterns,
    /// and the lenient fallback) when `legacy`
    /// (`spark.sql.legacy.timeParserPolicy=LEGACY`).
    ///
    /// # Errors
    /// As [`Self::from_map`].
    pub fn from_map_with(
        options: &BTreeMap<String, String>,
        legacy: bool,
    ) -> Result<Self, KrishivError> {
        let mut out = Self::new();
        out.datetime_fallback = legacy;
        if legacy {
            // Under LEGACY the options carry explicit `FastDateFormat`
            // patterns even when none is named (`JSONOptions`).
            out.date_format = Some(DateTimePattern::new_fast(LEGACY_DATE));
            out.date_output = Some(DateTimePattern::new_fast(LEGACY_DATE));
            out.timestamp_format = Some(DateTimePattern::new_fast(LEGACY_TIMESTAMP));
            out.timestamp_output = Some(DateTimePattern::new_fast(LEGACY_TIMESTAMP));
        }
        for (name, value) in options {
            match name.as_str() {
                "enableDateTimeParsingFallback" => out.datetime_fallback = is_true(value),
                "dateFormat" if legacy => {
                    out.date_format = Some(DateTimePattern::new_fast(value));
                    out.date_output = Some(DateTimePattern::new_fast(value));
                }
                "timestampFormat" if legacy => {
                    out.timestamp_format = Some(DateTimePattern::new_fast(value));
                    out.timestamp_output = Some(DateTimePattern::new_fast(value));
                }
                "dateFormat" => {
                    let pattern = DateTimePattern::new(value, Purpose::Parse)?;
                    out.date_format = Some(pattern);
                    out.date_output = Some(DateTimePattern::new(value, Purpose::Format)?);
                }
                "timeFormat" => {
                    let pattern = DateTimePattern::new(value, Purpose::Parse)?;
                    out.time_format = Some(pattern);
                    out.time_output = Some(DateTimePattern::new(value, Purpose::Format)?);
                }
                "timestampFormat" => {
                    let pattern = DateTimePattern::new(value, Purpose::Parse)?;
                    out.timestamp_format = Some(pattern);
                    out.timestamp_output = Some(DateTimePattern::new(value, Purpose::Format)?);
                }
                // Spark builds the NTZ formatter with `forTimestampNTZ`, which
                // never takes the legacy one, whatever the parser policy.
                "timestampNTZFormat" => {
                    let pattern = DateTimePattern::new(value, Purpose::Parse)?;
                    out.timestamp_ntz_format = Some(pattern);
                    out.timestamp_ntz_output = Some(DateTimePattern::new(value, Purpose::Format)?);
                }
                "primitivesAsString" => out.primitives_as_string = is_true(value),
                "prefersDecimal" => out.prefers_decimal = is_true(value),
                "allowComments" => out.allow_comments = is_true(value),
                "allowSingleQuotes" => out.allow_single_quotes = is_true(value),
                "allowNumericLeadingZeros" => out.allow_numeric_leading_zeros = is_true(value),
                "allowBackslashEscapingAnyCharacter" => {
                    out.allow_backslash_escaping_any = is_true(value);
                }
                "allowUnquotedFieldNames" => out.allow_unquoted_field_names = is_true(value),
                "allowUnquotedControlChars" => out.allow_unquoted_control_chars = is_true(value),
                "allowNonNumericNumbers" => out.allow_non_numeric_numbers = is_true(value),
                "sortKeys" => out.sort_keys = is_true(value),
                "ignoreNullFields" => out.ignore_null_fields = is_true(value),
                "dropFieldIfAllNull" => out.drop_field_if_all_null = is_true(value),
                "mode" => {
                    out.mode = match value.trim().to_ascii_uppercase().as_str() {
                        "PERMISSIVE" => ParseMode::Permissive,
                        "FAILFAST" => ParseMode::FailFast,
                        "DROPMALFORMED" => ParseMode::DropMalformed,
                        _ => out.mode,
                    };
                }
                // The file-only options (`multiLine`,
                // `columnNameOfCorruptRecord`, ...), `timeFormat` (no `TIME`
                // type yet) and anything unknown.
                _ => {}
            }
        }
        Ok(out)
    }

    /// Fails for a `timestampFormat` Spark cannot build: its generator
    /// builds the timestamp formatter eagerly (the date one only when it
    /// formats a date), so `to_json` fails at once.
    ///
    /// # Errors
    /// Spark's pattern error.
    pub fn validate_for_write(&self) -> Result<(), KrishivError> {
        match self
            .timestamp_output
            .as_ref()
            .and_then(DateTimePattern::build_error)
        {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The lexer flags of these options.
    #[must_use]
    pub(crate) const fn lexer(&self) -> super::parse::LexerFlags {
        super::parse::LexerFlags {
            allow_comments: self.allow_comments,
            allow_single_quotes: self.allow_single_quotes,
            allow_numeric_leading_zeros: self.allow_numeric_leading_zeros,
            allow_backslash_escaping_any: self.allow_backslash_escaping_any,
            allow_unquoted_field_names: self.allow_unquoted_field_names,
            allow_unquoted_control_chars: self.allow_unquoted_control_chars,
            allow_non_numeric_numbers: self.allow_non_numeric_numbers,
        }
    }
}

/// Spark's defaults for the date and timestamp patterns under LEGACY.
const LEGACY_DATE: &str = "yyyy-MM-dd";
const LEGACY_TIMESTAMP: &str = "yyyy-MM-dd'T'HH:mm:ss.SSSXXX";

/// Spark's `toBoolean`: only `"true"` (any case, trimmed) is true.
fn is_true(value: &str) -> bool {
    value.trim().eq_ignore_ascii_case("true")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ntz_pattern_stays_on_the_corrected_formatter_under_legacy() {
        let options = BTreeMap::from([(
            "timestampNTZFormat".to_owned(),
            "yyyy-MM-dd HH:mm:ss.SSSSSS".to_owned(),
        )]);
        let o = JsonOptions::from_map_with(&options, true).unwrap();
        let text = "2024-01-02 03:04:05.000123";
        let ntz = o.timestamp_ntz_format.as_ref().unwrap();
        let micros = ntz.parse_local(text, false).unwrap().unwrap();
        // `.000123` is 123 microseconds, not the legacy 123 milliseconds.
        assert_eq!(micros.rem_euclid(1_000_000), 123);
        let output = o.timestamp_ntz_output.as_ref().unwrap();
        assert_eq!(output.format_local(micros).unwrap(), text);
    }
}
