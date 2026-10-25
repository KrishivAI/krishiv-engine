// SPDX-License-Identifier: Apache-2.0
//! The JSON options of `from_json`, `to_json` and `schema_of_json`: the
//! option names of Spark's JSON data source (`JSONOptions`).
//!
//! Only the options a scalar function can observe are honoured: the three
//! date-time patterns, the inference flags (`primitivesAsString`,
//! `prefersDecimal`), the leniency flags (`allowComments`,
//! `allowSingleQuotes`, `allowNumericLeadingZeros`,
//! `allowBackslashEscapingAnyCharacter`, `allowUnquotedFieldNames`,
//! `allowUnquotedControlChars`) and `sortKeys` (a `to_json` writer flag).
//! Unknown options are ignored, as Spark's reader ignores them; `mode` is
//! accepted and always reads `PERMISSIVE` (Spark's default), and the file
//! options (`multiLine`, `columnNameOfCorruptRecord`, `samplingRatio`,
//! `dropFieldIfAllNull`, `locale`, `timeFormat`, ...) have no scalar
//! meaning and are ignored.

use std::collections::BTreeMap;

use krishiv_common::error::KrishivError;

use crate::datetime_pattern::{DateTimePattern, Purpose};

/// The parsed options of one JSON function call.
#[derive(Debug, Clone, Default)]
pub struct JsonOptions {
    /// `dateFormat`, compiled for parsing (a `from_json` reader flag).
    pub date_format: Option<DateTimePattern>,
    /// `timestampFormat`, compiled for parsing (a `from_json` reader flag).
    pub timestamp_format: Option<DateTimePattern>,
    /// `timestampNTZFormat`, compiled for parsing (a `from_json` reader flag).
    pub timestamp_ntz_format: Option<DateTimePattern>,
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
    /// `sortKeys` (a `to_json` writer flag).
    pub sort_keys: bool,
}

impl JsonOptions {
    /// The defaults (no options argument).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses `options` (option name to value, as Spark writes a
    /// `MAP<STRING, STRING>` literal).
    ///
    /// # Errors
    /// Spark's pattern error for a `dateFormat`, `timestampFormat` or
    /// `timestampNTZFormat` value Spark rejects. Unknown options and the
    /// file-only options are ignored.
    pub fn from_map(options: &BTreeMap<String, String>) -> Result<Self, KrishivError> {
        let mut out = Self::new();
        for (name, value) in options {
            match name.as_str() {
                "dateFormat" => {
                    let pattern = DateTimePattern::new(value, Purpose::Parse)?;
                    out.date_format = Some(pattern);
                    out.date_output = Some(DateTimePattern::new(value, Purpose::Format)?);
                }
                "timestampFormat" => {
                    let pattern = DateTimePattern::new(value, Purpose::Parse)?;
                    out.timestamp_format = Some(pattern);
                    out.timestamp_output = Some(DateTimePattern::new(value, Purpose::Format)?);
                }
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
                "sortKeys" => out.sort_keys = is_true(value),
                // `mode` (always `PERMISSIVE` here), the file-only options
                // (`multiLine`, `columnNameOfCorruptRecord`, ...),
                // `timeFormat` (no `TIME` type yet) and anything unknown.
                _ => {}
            }
        }
        Ok(out)
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
        }
    }
}

/// Spark's `toBoolean`: only `"true"` (any case, trimmed) is true.
fn is_true(value: &str) -> bool {
    value.trim().eq_ignore_ascii_case("true")
}
