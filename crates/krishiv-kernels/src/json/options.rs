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
//! `PARSE_MODE_UNSUPPORTED`). A `locale` other than `en-US` is refused
//! ([`check_locale`]). Unknown options are ignored, as Spark's reader
//! ignores them, and the file options (`multiLine`,
//! `columnNameOfCorruptRecord`, `samplingRatio`, `timeFormat`, ...) have no
//! scalar meaning and are ignored. Every boolean option, the ignored
//! `multiLine`, `pretty`, `writeNonAsciiCharacterAsCodePoint` and
//! `useUnsafeRow` included, must be `true` or `false` in any case
//! ([`crate::text::option_to_boolean`]), as Spark builds `JSONOptions`.

use std::collections::BTreeMap;

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};

use crate::datetime_pattern::{DateTimePattern, Purpose};

/// Fails unless the JSON option `locale` names the US locale.
///
/// Spark reads the tag with `Locale.forLanguageTag` (so `en-US` in any
/// case is `Locale.US`, and `en_US` is not a tag at all). Only `Locale.US`
/// parses a decimal string with the US rule (`ExprUtils.getDecimalParser`:
/// commas dropped, then `BigDecimal`); any other locale parses it with
/// Java's `DecimalFormat` and that locale's symbols (`1,5` is 1.5 under
/// `de-DE`), in reads, `from_json` and `prefersDecimal` inference, and its
/// date patterns use that locale's month and day names. Krishiv has no
/// locale data, so it refuses rather than read such values with the US rule.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for any other tag.
pub fn check_locale(tag: &str) -> Result<(), KrishivError> {
    if tag.eq_ignore_ascii_case("en-US") {
        Ok(())
    } else {
        Err(KrishivUnsupportedFeature {
            feature: format!("the JSON option locale {tag:?} (only en-US is supported)"),
        }
        .into())
    }
}

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
    /// `inferTimestamp` (a `schema_of_json` inference flag): strings may
    /// infer `TIME`, `TIMESTAMP_NTZ` or `TIMESTAMP`.
    pub infer_timestamp: bool,
    /// `spark.sql.legacy.timeParserPolicy=LEGACY` (the options were built by
    /// [`Self::from_map_with`] with `legacy`): inference reads a zone-less
    /// timestamp text as `TIMESTAMP` (SPARK-46769).
    pub legacy_parser: bool,
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
    /// Spark 4.2 reads six fraction digits of a `TIME` in the default grammar, a `timeFormat`
    /// every digit (a `from_json` reader flag set from the
    /// profile); 4.3 (`false`, the default) keeps nine.
    pub time_micros_only: bool,
}

impl JsonOptions {
    /// The defaults (no options argument).
    #[must_use]
    pub fn new() -> Self {
        Self {
            // `JSONOptions`' defaults: both on (Spark 4.2.1 / 4.3.0 runs read
            // `{'a': 'x'}` without options).
            allow_single_quotes: true,
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
        Self::build(options, legacy, false)
    }

    /// [`Self::from_map_with`] for `to_json`: the patterns are compiled for
    /// formatting only, as Spark's `JacksonGenerator` builds only formatters
    /// (`EEE, d MMM yyyy` writes, though no parser takes it).
    ///
    /// # Errors
    /// As [`Self::from_map`], for the formatting patterns.
    pub fn from_map_for_write(
        options: &BTreeMap<String, String>,
        legacy: bool,
    ) -> Result<Self, KrishivError> {
        Self::build(options, legacy, true)
    }

    fn build(
        options: &BTreeMap<String, String>,
        legacy: bool,
        writing: bool,
    ) -> Result<Self, KrishivError> {
        // A pattern compiled for parsing, or none for a write (never read).
        let parse = |value: &str| -> Result<Option<DateTimePattern>, KrishivError> {
            if writing {
                Ok(None)
            } else {
                DateTimePattern::new(value, Purpose::Parse).map(Some)
            }
        };
        let mut out = Self::new();
        out.datetime_fallback = legacy;
        out.legacy_parser = legacy;
        if legacy {
            // Under LEGACY the options carry explicit `FastDateFormat`
            // patterns even when none is named (`JSONOptions`).
            out.date_format = Some(DateTimePattern::new_fast(LEGACY_DATE));
            out.date_output = Some(DateTimePattern::new_fast(LEGACY_DATE));
            out.timestamp_format = Some(DateTimePattern::new_fast(LEGACY_TIMESTAMP));
            out.timestamp_output = Some(DateTimePattern::new_fast(LEGACY_TIMESTAMP));
        }
        for (name, value) in options {
            match name.to_ascii_lowercase().as_str() {
                "enabledatetimeparsingfallback" => out.datetime_fallback = flag(name, value)?,
                "dateformat" if legacy => {
                    out.date_format = Some(DateTimePattern::new_fast(value));
                    out.date_output = Some(DateTimePattern::new_fast(value));
                }
                "timestampformat" if legacy => {
                    out.timestamp_format = Some(DateTimePattern::new_fast(value));
                    out.timestamp_output = Some(DateTimePattern::new_fast(value));
                }
                "dateformat" => {
                    out.date_format = parse(value)?;
                    out.date_output = Some(DateTimePattern::new(value, Purpose::Format)?);
                }
                "timeformat" => {
                    out.time_format = parse(value)?;
                    // `HH:mm:ss` writes through Spark's fraction formatter.
                    let output = DateTimePattern::new(value, Purpose::Format)?;
                    out.time_output = (!output.is_default_time_pattern()).then_some(output);
                }
                "timestampformat" => {
                    out.timestamp_format = parse(value)?;
                    out.timestamp_output = Some(DateTimePattern::new(value, Purpose::Format)?);
                }
                // Spark builds the NTZ formatter with `forTimestampNTZ`, which
                // never takes the legacy one, whatever the parser policy.
                "timestampntzformat" => {
                    out.timestamp_ntz_format = parse(value)?;
                    out.timestamp_ntz_output = Some(DateTimePattern::new(value, Purpose::Format)?);
                }
                "locale" => check_locale(value)?,
                "primitivesasstring" => out.primitives_as_string = flag(name, value)?,
                "prefersdecimal" => out.prefers_decimal = flag(name, value)?,
                "infertimestamp" => out.infer_timestamp = flag(name, value)?,
                "allowcomments" => out.allow_comments = flag(name, value)?,
                "allowsinglequotes" => out.allow_single_quotes = flag(name, value)?,
                "allownumericleadingzeros" => out.allow_numeric_leading_zeros = flag(name, value)?,
                "allowbackslashescapinganycharacter" => {
                    out.allow_backslash_escaping_any = flag(name, value)?;
                }
                "allowunquotedfieldnames" => out.allow_unquoted_field_names = flag(name, value)?,
                "allowunquotedcontrolchars" => {
                    out.allow_unquoted_control_chars = flag(name, value)?;
                }
                "allownonnumericnumbers" => out.allow_non_numeric_numbers = flag(name, value)?,
                "sortkeys" => out.sort_keys = flag(name, value)?,
                "ignorenullfields" => out.ignore_null_fields = flag(name, value)?,
                "dropfieldifallnull" => out.drop_field_if_all_null = flag(name, value)?,
                // Boolean options with no scalar meaning: Spark still reads
                // each one with `toBoolean` when it builds `JSONOptions`.
                "multiline" | "pretty" | "writenonasciicharacterascodepoint" | "useunsaferow" => {
                    flag(name, value)?;
                }
                "mode" => {
                    out.mode = match value.trim().to_ascii_uppercase().as_str() {
                        "PERMISSIVE" => ParseMode::Permissive,
                        "FAILFAST" => ParseMode::FailFast,
                        "DROPMALFORMED" => ParseMode::DropMalformed,
                        _ => out.mode,
                    };
                }
                // The file-only options (`columnNameOfCorruptRecord`,
                // `samplingRatio`, ...) and anything unknown.
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

/// A boolean JSON option as Spark's `JSONOptions` reads every one: Scala's
/// `toBoolean` ([`crate::text::option_to_boolean`]), in every JSON entry point.
fn flag(name: &str, value: &str) -> Result<bool, KrishivError> {
    crate::text::option_to_boolean("JSON", name, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_us_locale_tag_is_accepted() {
        // `Locale.forLanguageTag` folds case; `en_US` is no tag (the root
        // locale in Spark, a `DecimalFormat` parser).
        for tag in ["en-US", "en-us", "EN-US"] {
            check_locale(tag).unwrap();
        }
        for tag in ["de-DE", "en_US", "en", " en-US", "fr-FR"] {
            let e = check_locale(tag).unwrap_err();
            assert_eq!(
                e.error_class(),
                krishiv_common::error::ErrorClass::KrishivUnsupportedFeature
            );
        }
        let named = BTreeMap::from([("Locale".to_owned(), "de-DE".to_owned())]);
        assert!(JsonOptions::from_map(&named).is_err());
        let us = BTreeMap::from([("locale".to_owned(), "en-US".to_owned())]);
        JsonOptions::from_map(&us).unwrap();
    }

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
