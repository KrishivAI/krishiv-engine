// SPDX-License-Identifier: Apache-2.0
//! The options of the CSV format (Spark's names and defaults) and the
//! parsed [`CsvOptions`] the tokenizer, decoder and encoder use.

use std::sync::LazyLock;

use crate::error::Result;
use crate::options::{FormatOptions, OptionDefault, OptionKind, OptionSchema, OptionSpec};
use crate::text::{Charset, LineSep, check_write_codec, invalid_option};
use krishiv_kernels::csv::{Eol, Unescaped};
pub(crate) use krishiv_kernels::csv::{ParseMode as Mode, Syntax};

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

fn flag(name: &'static str, default: Option<bool>, doc: &'static str) -> OptionSpec {
    OptionSpec {
        name,
        kind: OptionKind::Bool,
        default: match default {
            Some(b) => OptionDefault::Bool(b),
            None => OptionDefault::None,
        },
        conf_fallback: None,
        metadata_key: false,
        doc,
    }
}

/// The parse modes.
const MODES: &[&str] = &["PERMISSIVE", "DROPMALFORMED", "FAILFAST"];

/// The `unescapedQuoteHandling` values.
const UNESCAPED: &[&str] = &[
    "STOP_AT_CLOSING_QUOTE",
    "BACK_TO_DELIMITER",
    "STOP_AT_DELIMITER",
    "SKIP_VALUE",
    "RAISE_ERROR",
];

fn int(name: &'static str, min: i64, max: i64, default: i64, doc: &'static str) -> OptionSpec {
    OptionSpec {
        name,
        kind: OptionKind::Int { min, max },
        default: OptionDefault::Int(default),
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

const NONE: OptionDefault = OptionDefault::None;

/// The options as one list (a brace-delimited macro: `rustfmt` leaves the
/// table one option per line).
macro_rules! options {
    ($($spec:expr;)*) => { vec![$($spec),*] };
}

static SPECS: LazyLock<Vec<OptionSpec>> = LazyLock::new(|| {
    options! {
        flag("header", Some(false), "Read: the first line names the columns; write: a header line.");
        string("sep", "The field separator (default `,`, may be several characters).");
        string("delimiter", "Alias of `sep`.");
        string("quote", "The quote character (default `\"`, empty disables quoting).");
        string("escape", "The escape character in quoted values (default `\\`).");
        string("charToEscapeQuoteEscaping", "The character that escapes the escape character.");
        string("comment", "Lines starting with this character are skipped.");
        flag("multiLine", Some(false), "Records may span lines (the file is read whole).");
        mode_spec();
        conf("columnNameOfCorruptRecord", OptionKind::String, OptionDefault::Str("_corrupt_record"), "spark.sql.columnNameOfCorruptRecord", "The column that receives malformed records in PERMISSIVE mode.");
        string("nullValue", "The text that reads and writes as NULL (default empty).");
        string("nanValue", "The text of NaN (default `NaN`).");
        string("positiveInf", "The text of +Infinity (default `Inf`).");
        string("negativeInf", "The text of -Infinity (default `-Inf`).");
        string("emptyValue", "The text of an empty string (read: empty, write: `\"\"`).");
        string("dateFormat", "The pattern of DATE values (default: Spark's lenient grammar).");
        string("timestampFormat", "The pattern of TIMESTAMP values.");
        string("timestampNTZFormat", "The pattern of TIMESTAMP_NTZ values.");
        conf("timeZone", OptionKind::String, OptionDefault::Str("UTC"), "spark.sql.session.timeZone", "The time zone of timestamps without an offset.");
        string("encoding", "The charset: UTF-8, ISO-8859-1 or US-ASCII.");
        string("charset", "Alias of `encoding`.");
        flag("inferSchema", Some(false), "Infer column types from the data.");
        string("samplingRatio", "The fraction of lines inference reads (default 1.0).");
        flag("prefersDate", Some(true), "Accepted, no effect: Spark 4.2 infers DATE.");
        flag("prefersDecimal", Some(false), "Accepted, no effect: Spark 4.2 infers DOUBLE.");
        flag("enforceSchema", Some(true), "Use the given schema by position; the header is not checked.");
        flag("ignoreLeadingWhiteSpace", None, "Trim leading white space (read: false, write: true).");
        flag("ignoreTrailingWhiteSpace", None, "Trim trailing white space (read: false, write: true).");
        flag("quoteAll", Some(false), "Write: quote every value.");
        flag("escapeQuotes", Some(true), "Write: quote values that hold a quote.");
        string("lineSep", "The line separator (read: \\r, \\r\\n, \\n; write: \\n).");
        int("maxColumns", 1, i64::from(i32::MAX), 20480, "Most columns of a record.");
        int("maxCharsPerColumn", -1, i64::from(i32::MAX), -1, "Most characters of a value (-1: any).");
        unescaped_spec();
        flag("enableDateTimeParsingFallback", None, "Parse with the lenient grammar when the pattern fails.");
        conf("columnPruning", OptionKind::Bool, OptionDefault::Bool(true), "spark.sql.csv.parser.columnPruning.enabled", "Parse only the requested columns.");
        string("compression", "Write codec: only none and uncompressed so far.");
        string("codec", "Alias of `compression`.");
        string("locale", "The locale of numbers (only en-US).");
        conf("inferenceMaxPartitionBytes", OptionKind::String, NONE, "spark.sql.files.maxPartitionBytes", "Sampling: the split planning conf.");
        conf("inferenceOpenCostInBytes", OptionKind::String, NONE, "spark.sql.files.openCostInBytes", "Sampling: the split planning conf.");
        conf("inferenceMinPartitionNum", OptionKind::String, NONE, "spark.sql.files.minPartitionNum", "Sampling: the split planning conf.");
        conf("inferenceMaxPartitionNum", OptionKind::String, NONE, "spark.sql.files.maxPartitionNum", "Sampling: the split planning conf.");
        conf("inferenceParallelism", OptionKind::String, NONE, "spark.sql.leafNodeDefaultParallelism", "Sampling: the split planning conf.");
    }
});

pub(super) static OPTIONS: LazyLock<OptionSchema> = LazyLock::new(|| OptionSchema {
    format: "csv",
    options: SPECS.as_slice(),
});

fn unescaped_spec() -> OptionSpec {
    OptionSpec {
        name: "unescapedQuoteHandling",
        kind: OptionKind::Enum(UNESCAPED),
        default: OptionDefault::Str("STOP_AT_DELIMITER"),
        conf_fallback: None,
        metadata_key: false,
        doc: "How a quote inside a value is handled.",
    }
}

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

/// The parsed options of a read or write.
#[derive(Debug, Clone)]
pub(crate) struct CsvOptions {
    pub header: bool,
    pub syntax: Syntax,
    pub multi_line: bool,
    pub mode: Mode,
    pub corrupt_column: String,
    pub null_value: String,
    pub nan: String,
    pub pos_inf: String,
    pub neg_inf: String,
    pub charset: Charset,
    pub line_sep: LineSep,
    pub enforce_schema: bool,
    pub quote_all: bool,
    pub escape_quotes: bool,
    pub column_pruning: bool,
}

/// A single character option (`quote`, `escape`, `comment`): `None` when
/// empty (disabled).
fn one_char(o: &FormatOptions, name: &str) -> Result<Option<Option<char>>> {
    let Some(v) = o.str(name) else {
        return Ok(None);
    };
    let mut chars = v.chars();
    match (chars.next(), chars.next()) {
        (None, _) => Ok(Some(None)),
        (Some(c), None) => Ok(Some(Some(c))),
        _ => Err(invalid_option(
            "csv",
            name,
            v,
            "the value must be a single character",
        )),
    }
}

/// Spark's `CSVExprUtils.toDelimiterStr`: backslash escapes of the
/// separator.
fn separator(raw: &str) -> Result<String> {
    if raw.is_empty() {
        return Err(invalid_option(
            "csv",
            "sep",
            raw,
            "the separator cannot be empty",
        ));
    }
    let mut out = String::new();
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        let bad = || invalid_option("csv", "sep", raw, "unsupported escape sequence");
        match chars.next() {
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('b') => out.push('\u{8}'),
            Some('f') => out.push('\u{c}'),
            Some('"') => out.push('"'),
            Some('\'') => out.push('\''),
            Some('\\') => out.push('\\'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                let c = u32::from_str_radix(&hex, 16)
                    .ok()
                    .filter(|_| hex.len() == 4)
                    .and_then(char::from_u32)
                    .ok_or_else(bad)?;
                out.push(c);
            }
            _ => return Err(bad()),
        }
    }
    Ok(out)
}

impl CsvOptions {
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
        if let Some(l) = o.str("locale")
            && !l.trim().eq_ignore_ascii_case("en-US")
            && !l.trim().eq_ignore_ascii_case("en_US")
        {
            return Err(crate::error::unsupported(format!(
                "the locale {l} (only en-US is supported)"
            )));
        }
        let line_sep = LineSep::parse(o.str("lineSep"), charset, "csv")?;
        let multi_line = o.bool("multiLine").unwrap_or(false);
        let quote = one_char(o, "quote")?.unwrap_or(Some('"'));
        let escape = one_char(o, "escape")?.unwrap_or(Some('\\'));
        let escape_escape = match one_char(o, "charToEscapeQuoteEscaping")? {
            Some(c) => c,
            None if escape != quote => escape,
            None => None,
        };
        let sep = separator(o.str("sep").or_else(|| o.str("delimiter")).unwrap_or(","))?;
        let trim_default = writing;
        let eol = match (&line_sep, multi_line) {
            (LineSep::Custom(_), true) => Eol::Custom(o.str("lineSep").unwrap_or("\n").to_owned()),
            (LineSep::Default, true) => Eol::Default,
            (_, false) => Eol::EndOfText,
        };
        let max_chars = o
            .int("maxCharsPerColumn")
            .filter(|n| *n >= 0)
            .and_then(|n| usize::try_from(n).ok());
        let empty_default = if writing { "\"\"" } else { "" };
        let mode = match o.str("mode") {
            Some("DROPMALFORMED") => Mode::DropMalformed,
            Some("FAILFAST") => Mode::FailFast,
            _ => Mode::Permissive,
        };
        Ok(Self {
            header: o.bool("header").unwrap_or(false),
            syntax: Syntax {
                sep,
                quote,
                escape,
                escape_escape,
                comment: one_char(o, "comment")?.flatten(),
                ltrim: o.bool("ignoreLeadingWhiteSpace").unwrap_or(trim_default),
                rtrim: o.bool("ignoreTrailingWhiteSpace").unwrap_or(trim_default),
                empty_value: o.str("emptyValue").unwrap_or(empty_default).to_owned(),
                unescaped: match o.str("unescapedQuoteHandling") {
                    Some("STOP_AT_CLOSING_QUOTE") => Unescaped::StopAtClosingQuote,
                    Some("BACK_TO_DELIMITER") => Unescaped::BackToDelimiter,
                    Some("SKIP_VALUE") => Unescaped::SkipValue,
                    Some("RAISE_ERROR") => Unescaped::RaiseError,
                    _ => Unescaped::StopAtDelimiter,
                },
                max_columns: o
                    .int("maxColumns")
                    .and_then(|n| usize::try_from(n).ok())
                    .unwrap_or(20480),
                max_chars,
                eol,
            },
            multi_line,
            mode,
            corrupt_column: o
                .str("columnNameOfCorruptRecord")
                .unwrap_or("_corrupt_record")
                .to_owned(),
            null_value: o.str("nullValue").unwrap_or("").to_owned(),
            nan: o.str("nanValue").unwrap_or("NaN").to_owned(),
            pos_inf: o.str("positiveInf").unwrap_or("Inf").to_owned(),
            neg_inf: o.str("negativeInf").unwrap_or("-Inf").to_owned(),
            charset,
            line_sep,
            enforce_schema: o.bool("enforceSchema").unwrap_or(true),
            quote_all: o.bool("quoteAll").unwrap_or(false),
            escape_quotes: o.bool("escapeQuotes").unwrap_or(true),
            column_pruning: o.bool("columnPruning").unwrap_or(true),
        })
    }
}
