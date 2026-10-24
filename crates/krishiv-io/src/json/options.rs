// SPDX-License-Identifier: Apache-2.0
//! The options of the JSON format (Spark's names and defaults) and the
//! parsed [`JsonOptions`] the tokenizer, decoder and encoder use.

use std::sync::LazyLock;

use crate::error::Result;
use crate::options::{FormatOptions, OptionDefault, OptionKind, OptionSchema, OptionSpec};
use crate::text::{Charset, LineSep, check_write_codec, invalid_option};

fn string(name: &'static str, doc: &'static str) -> OptionSpec {
    OptionSpec {
        name,
        kind: OptionKind::String,
        default: OptionDefault::None,
        conf_fallback: None,
        metadata_key: false,
        doc,
    }
}

fn flag(name: &'static str, default: bool, doc: &'static str) -> OptionSpec {
    OptionSpec {
        name,
        kind: OptionKind::Bool,
        default: OptionDefault::Bool(default),
        conf_fallback: None,
        metadata_key: false,
        doc,
    }
}

/// An option read from the session conf `conf` when not given.
fn conf(
    name: &'static str,
    kind: OptionKind,
    default: OptionDefault,
    conf: &'static str,
    doc: &'static str,
) -> OptionSpec {
    OptionSpec {
        name,
        kind,
        default,
        conf_fallback: Some(conf),
        metadata_key: false,
        doc,
    }
}

/// The parse modes.
const MODES: &[&str] = &["PERMISSIVE", "DROPMALFORMED", "FAILFAST"];

/// The options as one list (a brace-delimited macro: `rustfmt` leaves the
/// table one option per line).
macro_rules! options {
    ($($spec:expr;)*) => { vec![$($spec),*] };
}

static SPECS: LazyLock<Vec<OptionSpec>> = LazyLock::new(|| {
    options! {
        flag("multiLine", false, "Records may span lines (the file is read whole).");
        flag("primitivesAsString", false, "Read every primitive value as a string.");
        flag("prefersDecimal", false, "Infer decimals instead of doubles.");
        flag("allowComments", false, "Allow `//` and `/* */` comments in records.");
        flag("allowUnquotedFieldNames", false, "Allow unquoted field names.");
        flag("allowSingleQuotes", false, "Allow single-quoted strings.");
        flag("allowNumericLeadingZeros", false, "Allow numbers like `007`.");
        flag("allowBackslashEscapingAnyCharacter", false, "Allow `\\` before any character.");
        flag("allowUnquotedControlChars", false, "Allow raw control characters in strings.");
        mode_spec();
        conf("columnNameOfCorruptRecord", OptionKind::String, OptionDefault::Str("_corrupt_record"), "spark.sql.columnNameOfCorruptRecord", "The column that receives malformed records in PERMISSIVE mode.");
        string("dateFormat", "The pattern of DATE values (default: Spark's lenient grammar).");
        string("timestampFormat", "The pattern of TIMESTAMP values.");
        string("timestampNTZFormat", "The pattern of TIMESTAMP_NTZ values.");
        conf("timeZone", OptionKind::String, OptionDefault::Str("UTC"), "spark.sql.session.timeZone", "The time zone of timestamps without an offset.");
        string("encoding", "The charset: UTF-8, ISO-8859-1 or US-ASCII.");
        string("charset", "Alias of `encoding`.");
        string("lineSep", "The line separator (read: \\r, \\r\\n, \\n; write: \\n).");
        string("samplingRatio", "The fraction of records inference reads (default 1.0).");
        flag("dropFieldIfAllNull", false, "Inference drops columns that are null in every record.");
        flag("ignoreNullFields", false, "Write: omit fields whose value is null.");
        flag("enableDateTimeParsingFallback", false, "Parse with the lenient grammar when the pattern fails.");
        string("compression", "Write codec: only none and uncompressed so far.");
        string("codec", "Alias of `compression`.");
        conf("inferenceMaxPartitionBytes", OptionKind::String, OptionDefault::None, "spark.sql.files.maxPartitionBytes", "Sampling: the split planning conf.");
        conf("inferenceOpenCostInBytes", OptionKind::String, OptionDefault::None, "spark.sql.files.openCostInBytes", "Sampling: the split planning conf.");
        conf("inferenceMinPartitionNum", OptionKind::String, OptionDefault::None, "spark.sql.files.minPartitionNum", "Sampling: the split planning conf.");
        conf("inferenceMaxPartitionNum", OptionKind::String, OptionDefault::None, "spark.sql.files.maxPartitionNum", "Sampling: the split planning conf.");
        conf("inferenceParallelism", OptionKind::String, OptionDefault::None, "spark.sql.leafNodeDefaultParallelism", "Sampling: the split planning conf.");
    }
});

pub(super) static OPTIONS: LazyLock<OptionSchema> = LazyLock::new(|| OptionSchema {
    format: "json",
    options: SPECS.as_slice(),
});

fn mode_spec() -> OptionSpec {
    OptionSpec {
        name: "mode",
        kind: OptionKind::Enum(MODES),
        default: OptionDefault::Str("PERMISSIVE"),
        conf_fallback: None,
        metadata_key: false,
        doc: "How malformed records are handled.",
    }
}

/// How malformed records are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Permissive,
    DropMalformed,
    FailFast,
}

/// The syntax switches of the JSON tokenizer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Syntax {
    /// `//` and `/* */` comments are skipped.
    pub comments: bool,
    /// Bare `name:` field names are accepted.
    pub unquoted_names: bool,
    /// `'...'` strings are accepted.
    pub single_quotes: bool,
    /// `007` numbers are accepted.
    pub leading_zeros: bool,
    /// `\<any>` is the character itself.
    pub backslash_any: bool,
    /// Raw bytes below `0x20` are accepted in strings.
    pub unquoted_controls: bool,
}

impl From<&JsonOptions> for Syntax {
    fn from(o: &JsonOptions) -> Self {
        Self {
            comments: o.syntax.comments,
            unquoted_names: o.syntax.unquoted_names,
            single_quotes: o.syntax.single_quotes,
            leading_zeros: o.syntax.leading_zeros,
            backslash_any: o.syntax.backslash_any,
            unquoted_controls: o.syntax.unquoted_controls,
        }
    }
}

/// The parsed options of a read or write.
#[derive(Debug, Clone)]
pub(crate) struct JsonOptions {
    pub multi_line: bool,
    pub primitives_as_string: bool,
    pub prefers_decimal: bool,
    pub syntax: Syntax,
    pub mode: Mode,
    pub corrupt_column: String,
    pub charset: Charset,
    pub line_sep: LineSep,
    pub drop_field_if_all_null: bool,
    pub ignore_null_fields: bool,
}

impl JsonOptions {
    /// The options of a read (`writing` false) or a write.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_FORMAT_OPTION_VALUE` / `KRISHIV_UNSUPPORTED_FEATURE`
    /// for a value that cannot be honoured.
    pub(crate) fn new(o: &FormatOptions, writing: bool) -> Result<Self> {
        if writing {
            check_write_codec(o)?;
        }
        let charset = Charset::parse(o.str("encoding").or_else(|| o.str("charset")))?;
        let line_sep = LineSep::parse(o.str("lineSep"), charset, "json")?;
        let mode = match o.str("mode") {
            Some("DROPMALFORMED") => Mode::DropMalformed,
            Some("FAILFAST") => Mode::FailFast,
            _ => Mode::Permissive,
        };
        Ok(Self {
            multi_line: o.bool("multiLine").unwrap_or(false),
            primitives_as_string: o.bool("primitivesAsString").unwrap_or(false),
            prefers_decimal: o.bool("prefersDecimal").unwrap_or(false),
            syntax: Syntax {
                comments: o.bool("allowComments").unwrap_or(false),
                unquoted_names: o.bool("allowUnquotedFieldNames").unwrap_or(false),
                single_quotes: o.bool("allowSingleQuotes").unwrap_or(false),
                leading_zeros: o.bool("allowNumericLeadingZeros").unwrap_or(false),
                backslash_any: o
                    .bool("allowBackslashEscapingAnyCharacter")
                    .unwrap_or(false),
                unquoted_controls: o.bool("allowUnquotedControlChars").unwrap_or(false),
            },
            mode,
            corrupt_column: o
                .str("columnNameOfCorruptRecord")
                .unwrap_or("_corrupt_record")
                .to_owned(),
            charset,
            line_sep,
            drop_field_if_all_null: o.bool("dropFieldIfAllNull").unwrap_or(false),
            ignore_null_fields: o.bool("ignoreNullFields").unwrap_or(false),
        })
    }
}

/// A sampling ratio: a number greater than 0.
pub(crate) fn sampling_ratio(o: &FormatOptions) -> crate::error::FormatResult<f64> {
    let Some(v) = o.str("samplingRatio") else {
        return Ok(1.0);
    };
    v.trim()
        .parse::<f64>()
        .ok()
        .filter(|r| *r > 0.0 && r.is_finite())
        .ok_or_else(|| {
            crate::error::FormatError::Other(invalid_option(
                "json",
                "samplingRatio",
                v,
                "expected a number greater than 0",
            ))
        })
}
