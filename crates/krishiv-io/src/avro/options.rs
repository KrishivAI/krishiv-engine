// SPDX-License-Identifier: Apache-2.0
//! The options of the Avro format (Spark's names and defaults) and the
//! parsed [`AvroOptions`] the reader and writer use.
//!
//! The names, defaults and value spellings follow the Spark 4.2 Avro data
//! source documentation: `avroSchema`, `compression`, `recordName`,
//! `recordNamespace`, `ignoreExtension`, `positionalFieldMatching`, `mode`
//! and `datetimeRebaseMode`. Unlike the CSV format there are no session-conf
//! fallbacks: Spark's Avro options have no documented `spark.sql.avro.*`
//! conf keys, so every option is read as given, else its default.

use std::sync::LazyLock;

use crate::error::Result;
use crate::options::{FormatOptions, OptionDefault, OptionKind, OptionSchema, OptionSpec};
use crate::text::invalid_option;

/// How malformed records are handled.
const MODES: &[&str] = &["PERMISSIVE", "DROPMALFORMED", "FAILFAST"];

/// How dates and timestamps before the calendar switch are handled.
const REBASE_MODES: &[&str] = &["EXCEPTION", "CORRECTED", "LEGACY"];

/// Spark's compression names, stored in this spelling: `uncompressed` and
/// `null` both mean no codec.
const COMPRESSIONS: &[&str] = &[
    "uncompressed",
    "null",
    "snappy",
    "deflate",
    "bzip2",
    "xz",
    "zstandard",
    "zstd",
];

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

fn spec(name: &'static str, value: &'static str, doc: &'static str) -> OptionSpec {
    OptionSpec {
        name,
        kind: OptionKind::String,
        default: OptionDefault::Str(value),
        conf_fallback: None,
        metadata_key: false,
        doc,
    }
}

/// An option as one list (a brace-delimited macro: `rustfmt` leaves the
/// table one option per line).
macro_rules! options {
    ($($spec:expr;)*) => { vec![$($spec),*] };
}

static SPECS: LazyLock<Vec<OptionSpec>> = LazyLock::new(|| {
    options! {
        OptionSpec { name: "avroSchema", kind: OptionKind::String, default: OptionDefault::None, conf_fallback: None, metadata_key: true, doc: "The Avro schema (JSON) to read or write with: the reader schema for a read, the output schema for a write. When unset, a read uses each file's own schema and a write builds one from the data." };
        OptionSpec { name: "compression", kind: OptionKind::Enum(COMPRESSIONS), default: OptionDefault::Str("snappy"), conf_fallback: None, metadata_key: false, doc: "The block codec: uncompressed (null), snappy (default), deflate, bzip2, xz, zstandard.", };
        spec("recordName", "topLevelRecord", "The record name of a written schema built from the data (without `avroSchema`).");
        spec("recordNamespace", "", "The record namespace of a written schema built from the data (without `avroSchema`).");
        flag("ignoreExtension", true, "Accepted: the framework lists files without filtering by extension, so extensions are always ignored.");
        flag("positionalFieldMatching", false, "Match a file's fields to the read schema by position instead of by name.");
        OptionSpec { name: "mode", kind: OptionKind::Enum(MODES), default: OptionDefault::Str("PERMISSIVE"), conf_fallback: None, metadata_key: false, doc: "How corrupt records are handled.", };
        OptionSpec { name: "datetimeRebaseMode", kind: OptionKind::Enum(REBASE_MODES), default: OptionDefault::Str("EXCEPTION"), conf_fallback: None, metadata_key: false, doc: "How dates and timestamps before 1582-10-15 are handled (Spark's Avro default is EXCEPTION).", };
    }
});

pub(super) static OPTIONS: LazyLock<OptionSchema> = LazyLock::new(|| OptionSchema {
    format: "avro",
    options: SPECS.as_slice(),
});

/// How malformed records are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Permissive,
    DropMalformed,
    FailFast,
}

/// How dates and timestamps before the calendar switch are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rebase {
    Exception,
    Corrected,
    Legacy,
}

/// The block codec of a read or write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Codec {
    Null,
    Snappy,
    Deflate,
    Bzip2,
    Xz,
    Zstandard,
}

impl Codec {
    /// The codec named by the `compression` option (validated by the
    /// option schema, matched case-insensitively like Spark's).
    pub(crate) fn of(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "snappy" => Self::Snappy,
            "deflate" => Self::Deflate,
            "bzip2" => Self::Bzip2,
            "xz" => Self::Xz,
            "zstandard" | "zstd" => Self::Zstandard,
            _ => Self::Null,
        }
    }

    /// The name stored in an Avro file header (`avro.codec`).
    pub(crate) fn header_name(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Snappy => "snappy",
            Self::Deflate => "deflate",
            Self::Bzip2 => "bzip2",
            Self::Xz => "xz",
            Self::Zstandard => "zstandard",
        }
    }
}

/// The parsed options of a read or write.
#[derive(Debug, Clone)]
pub(crate) struct AvroOptions {
    pub avro_schema: Option<String>,
    pub codec: Codec,
    pub record_name: String,
    pub record_namespace: String,
    pub positional: bool,
    pub mode: Mode,
    pub rebase: Rebase,
}

impl AvroOptions {
    /// The options of a read (`writing` false) or a write.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_FORMAT_OPTION_VALUE` for a value that cannot be
    /// honoured.
    pub(crate) fn new(o: &FormatOptions, writing: bool) -> Result<Self> {
        let _ = writing;
        let record_name = o.str("recordName").unwrap_or("topLevelRecord").to_owned();
        if record_name.trim().is_empty() {
            return Err(invalid_option(
                "avro",
                "recordName",
                &record_name,
                "the record name cannot be empty",
            ));
        }
        let record_namespace = o.str("recordNamespace").unwrap_or("").to_owned();
        Ok(Self {
            avro_schema: o.str("avroSchema").map(str::to_owned),
            codec: Codec::of(o.str("compression").unwrap_or("snappy")),
            record_name,
            record_namespace,
            positional: o.bool("positionalFieldMatching").unwrap_or(false),
            mode: match o.str("mode") {
                Some("DROPMALFORMED") => Mode::DropMalformed,
                Some("FAILFAST") => Mode::FailFast,
                _ => Mode::Permissive,
            },
            rebase: match o.str("datetimeRebaseMode") {
                Some("CORRECTED") => Rebase::Corrected,
                Some("LEGACY") => Rebase::Legacy,
                _ => Rebase::Exception,
            },
        })
    }
}
