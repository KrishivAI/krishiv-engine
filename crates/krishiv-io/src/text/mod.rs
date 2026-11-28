// SPDX-License-Identifier: Apache-2.0
//! What the line-oriented text formats share (CSV, JSON lines, `text`):
//!
//! - `LineReader`: a sans-IO reader of the lines of one split, with
//!   Hadoop's `LineRecordReader` split rule (a split that does not start at
//!   byte 0 skips its first, partial line; a split reads every line that
//!   starts at or before its end), so any split boundaries read each line
//!   exactly once.
//! - `WholeReader`: a whole file, for formats whose records may span lines
//!   (`multiLine`) or files that cannot be split (compressed).
//! - `Charset`: the `encoding` option; decoding replaces invalid bytes
//!   with U+FFFD, as Java's decoders do in Spark.
//! - `Datetimes`: dates and timestamps as text, with the session time
//!   zone, Spark's default grammars and the `dateFormat` / `timestampFormat`
//!   / `timestampNTZFormat` patterns.
//! - [`TextMetadata`]: these formats have no footer; their metadata is the
//!   file size.

pub(crate) mod codec;
mod datetime;
pub(crate) mod reader;
#[cfg(test)]
pub(crate) mod time_golden;

use std::borrow::Cow;
use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};

pub(crate) use self::datetime::Datetimes;

/// The session's `TIME` gate, carried to the CSV and JSON readers as an
/// option named as the conf (`OptionSchema::with_conf_fallbacks`): schema
/// inference gives `TIME` only while it is on, as Spark 4.3.0's
/// `CSVInferSchema` / `JsonInferSchema` read `SQLConf.isTimeTypeEnabled`.
pub(crate) const TIME_TYPE_CONF: &str = krishiv_common::config::TIME_TYPE_ENABLED;

/// The session's `krishiv.io.text.timeNanos`, read as an option named as the
/// conf: a `TIME` read keeps nine fraction digits (Spark 4.3.0) or six (Spark
/// 4.2, whose default `TIME` grammar reads microseconds; a `timeFormat` pattern
/// keeps every digit in both).
pub(crate) const TIME_NANOS_CONF: &str = "krishiv.io.text.timeNanos";

/// The session's `krishiv.io.text.inferTime`, read as an option named as the
/// conf: schema inference may give `TIME` (with [`TIME_TYPE_CONF`] on) only
/// under the Spark 4.3 profile; Spark 4.2's `CSVInferSchema` /
/// `JsonInferSchema` never give `TIME` (Spark 4.2.1 run, gate on:
/// `tests/fixtures/spark-4.2.1/time-text-profiles.json`).
pub(crate) const TIME_INFER_CONF: &str = "krishiv.io.text.inferTime";

/// The session's `spark.sql.timestampType`, read as a session-only conf
/// option (JSON): inference gives `TIMESTAMP_NTZ` only when it is
/// `TIMESTAMP_NTZ` (Spark's `JsonInferSchema.isDefaultNTZ`).
pub(crate) const TIMESTAMP_TYPE_CONF: &str = "spark.sql.timestampType";

/// The session's `spark.sql.legacy.timeParserPolicy`, read as a
/// session-only conf option (JSON): `LEGACY` reads dates and timestamps
/// with `FastDateFormat` patterns and the lenient fallback (`JSONOptions`,
/// `JacksonParser`), and inference takes a zone-less timestamp text as
/// `TIMESTAMP` (SPARK-46769).
pub(crate) const PARSER_POLICY_CONF: &str = "spark.sql.legacy.timeParserPolicy";

/// Whether the session's default timestamp type is `TIMESTAMP_NTZ`
/// ([`TIMESTAMP_TYPE_CONF`]).
pub(crate) fn default_ntz(o: &crate::options::FormatOptions) -> bool {
    o.str(TIMESTAMP_TYPE_CONF)
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("TIMESTAMP_NTZ"))
}

/// Whether the session's parser policy is `LEGACY` ([`PARSER_POLICY_CONF`]).
pub(crate) fn legacy_parser(o: &crate::options::FormatOptions) -> bool {
    o.str(PARSER_POLICY_CONF)
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("LEGACY"))
}

/// Whether schema inference may give `TIME`: the session's gate and its
/// Spark 4.3 profile ([`TIME_INFER_CONF`]).
pub(crate) fn infers_time(o: &crate::options::FormatOptions) -> bool {
    o.bool(TIME_TYPE_CONF).unwrap_or(false) && o.bool(TIME_INFER_CONF).unwrap_or(true)
}

/// Spark 4.3.0 refuses to write a `TIME` column (at any depth) to a text
/// file while [`TIME_TYPE_CONF`] is off: `UNSUPPORTED_TIME_TYPE`.
///
/// # Errors
/// `UNSUPPORTED_TIME_TYPE`.
pub(crate) fn refuse_time_without_gate(
    schema: &arrow_schema::Schema,
    o: &crate::options::FormatOptions,
) -> Result<(), krishiv_common::error::KrishivError> {
    fn has_time(t: &arrow_schema::DataType) -> bool {
        use arrow_schema::DataType as A;
        match t {
            A::Time32(_) | A::Time64(_) => true,
            A::List(f)
            | A::LargeList(f)
            | A::ListView(f)
            | A::FixedSizeList(f, _)
            | A::Map(f, _) => has_time(f.data_type()),
            A::Struct(fs) => fs.iter().any(|f| has_time(f.data_type())),
            _ => false,
        }
    }
    if !o.bool(TIME_TYPE_CONF).unwrap_or(false)
        && schema.fields().iter().any(|f| has_time(f.data_type()))
    {
        return Err(krishiv_common::error::UnsupportedTimeType {}.into());
    }
    Ok(())
}
use crate::error::unsupported;
use crate::format::FileMetadata;
use crate::stats::{FileStatistics, Precision};

/// Bytes a text decoder asks for at a time.
pub(crate) const CHUNK_BYTES: u64 = 1 << 20;

/// Rows per batch a text decoder returns.
pub(crate) const BATCH_ROWS: usize = 4096;

/// The metadata of a file without a footer: its size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextMetadata {
    /// The file size (compressed, for a compressed file).
    pub size: u64,
    /// The file's compression codec, by its suffix.
    pub(crate) codec: Option<codec::Codec>,
}

impl FileMetadata for TextMetadata {
    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::empty())
    }

    fn statistics(&self) -> FileStatistics {
        FileStatistics {
            bytes: Precision::Exact(self.size),
            ..FileStatistics::default()
        }
    }

    fn heap_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl TextMetadata {
    /// The metadata of the listed `file` under the read's suffix rules (the
    /// metadata cache keys on them: [`compression_suffixes_spec`]).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a zstd file under the rules of a
    /// Spark line before 4.1.
    pub(crate) fn of(
        file: &crate::io::FileRef,
        o: &crate::options::FormatOptions,
    ) -> crate::error::Result<Self> {
        let rules = codec::CompressionSuffixes::of(o);
        check_plain(file.path.as_ref(), rules)?;
        Ok(Self {
            size: file.size,
            codec: codec::Codec::of_path(file.path.as_ref(), rules),
        })
    }

    /// The unit a split reads: the split's bytes, if it owns any line.
    /// `whole` files (`multiLine`) belong to the
    /// split that starts at byte 0.
    pub(crate) fn plan(
        &self,
        split: &crate::io::ByteRange,
        whole: bool,
    ) -> crate::format::UnitPlan {
        // A compressed file is read whole by the split at byte 0.
        let whole = whole || self.codec.is_some();
        let owns = if whole {
            split.start == 0
        } else {
            split.start < self.size
        };
        let mut plan = crate::format::UnitPlan::default();
        if owns && self.size > 0 {
            plan.owned = 1;
            plan.units.push(crate::format::ScanUnit {
                index: 0,
                first_row: 0,
                rows: 0,
                bytes: split.start..split.end.min(self.size),
            });
        }
        plan
    }
}

/// The metadata of `m` as a text file's.
pub(crate) fn downcast(
    m: &std::sync::Arc<dyn FileMetadata>,
) -> crate::error::FormatResult<TextMetadata> {
    let any: &dyn std::any::Any = m.as_ref();
    any.downcast_ref::<TextMetadata>()
        .cloned()
        .ok_or_else(|| crate::error::FormatError::Corrupt("not text metadata".to_owned()))
}

/// The `encoding` (`charset`) of a text file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Charset {
    Utf8,
    /// ISO-8859-1: one byte per character.
    Latin1,
    /// US-ASCII: bytes above 0x7F are malformed.
    Ascii,
}

impl Charset {
    /// The charset named by the option (Java names and aliases,
    /// case-insensitive).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for any other charset.
    pub(crate) fn parse(name: Option<&str>) -> crate::error::Result<Self> {
        let Some(name) = name else {
            return Ok(Self::Utf8);
        };
        let norm = name.trim().to_ascii_uppercase().replace('_', "-");
        match norm.as_str() {
            "UTF-8" | "UTF8" => Ok(Self::Utf8),
            "ISO-8859-1" | "ISO8859-1" | "LATIN1" | "ISO-LATIN-1" | "L1" | "8859-1" | "CP819"
            | "IBM819" => Ok(Self::Latin1),
            "US-ASCII" | "ASCII" | "ASCII7" | "646" | "ISO646-US" => Ok(Self::Ascii),
            _ => Err(unsupported(format!(
                "the file encoding {name} (UTF-8, ISO-8859-1 and US-ASCII are supported)"
            ))),
        }
    }

    /// `bytes` as text; malformed input becomes U+FFFD.
    pub(crate) fn decode<'a>(self, bytes: &'a [u8]) -> Cow<'a, str> {
        match self {
            Self::Utf8 => String::from_utf8_lossy(bytes),
            Self::Latin1 => match std::str::from_utf8(bytes) {
                Ok(s) if s.is_ascii() => Cow::Borrowed(s),
                _ => Cow::Owned(bytes.iter().map(|&b| char::from(b)).collect()),
            },
            Self::Ascii => match std::str::from_utf8(bytes) {
                Ok(s) if s.is_ascii() => Cow::Borrowed(s),
                _ => Cow::Owned(
                    bytes
                        .iter()
                        .map(|&b| {
                            if b.is_ascii() {
                                char::from(b)
                            } else {
                                '\u{fffd}'
                            }
                        })
                        .collect(),
                ),
            },
        }
    }

    /// `text` in this charset; characters it cannot hold become `?`, as
    /// Java's encoders write them.
    pub(crate) fn encode(self, text: &str, out: &mut Vec<u8>) {
        match self {
            Self::Utf8 => out.extend_from_slice(text.as_bytes()),
            Self::Latin1 => out.extend(text.chars().map(|c| u8::try_from(c).unwrap_or(b'?'))),
            Self::Ascii => out.extend(
                text.chars()
                    .map(|c| u8::try_from(c).ok().filter(u8::is_ascii).unwrap_or(b'?')),
            ),
        }
    }
}

/// How lines end: `\n`, `\r` or `\r\n` (the default), or the `lineSep`
/// option's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LineSep {
    Default,
    Custom(Vec<u8>),
}

impl LineSep {
    /// The separator of the `lineSep` option.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_FORMAT_OPTION_VALUE` for an empty separator.
    pub(crate) fn parse(
        sep: Option<&str>,
        charset: Charset,
        format: &str,
    ) -> crate::error::Result<Self> {
        match sep {
            None => Ok(Self::Default),
            Some("") => Err(invalid_option(
                format,
                "lineSep",
                "",
                "the line separator cannot be empty",
            )),
            Some(s) => {
                let mut bytes = Vec::new();
                charset.encode(s, &mut bytes);
                Ok(Self::Custom(bytes))
            }
        }
    }
}

/// `KRISHIV_INVALID_FORMAT_OPTION_VALUE`.
pub(crate) fn invalid_option(
    format: &str,
    option: &str,
    value: &str,
    reason: &str,
) -> krishiv_common::error::KrishivError {
    krishiv_common::error::KrishivInvalidFormatOptionValue {
        value: format!("'{value}'"),
        option: format!("\"{option}\""),
        format: format!("\"{format}\""),
        reason: reason.to_owned(),
    }
    .into()
}

/// A type as Spark prints it (`int`, `array<int>`).
pub(crate) fn spark_type_name(dt: &arrow_schema::DataType) -> String {
    krishiv_types::DataType::from_arrow(dt).map_or_else(|_| dt.to_string(), |t| t.simple_string())
}

#[cfg(test)]
mod codec_tests;
#[cfg(test)]
mod format_codec_tests;
#[cfg(test)]
mod reader_tests;
#[cfg(test)]
mod tests;

/// Zstandard's suffixes, which the rules of Spark 3.5 and 4.0 do not name.
const ZSTD_SUFFIXES: &[&str] = &[".zst", ".zstd"];

/// Checks that the file at `path` is not a zstd file the `rules` have no codec
/// for: stock Spark 3.5 and 4.0 fail on `.zst`, and reading either suffix as
/// text would give the compressed bytes.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for such a file.
pub(crate) fn check_plain(
    path: &str,
    rules: codec::CompressionSuffixes,
) -> crate::error::Result<()> {
    if rules != codec::CompressionSuffixes::Extended
        && ZSTD_SUFFIXES.iter().any(|s| rules.matches(path, s))
    {
        return Err(unsupported(format!(
            "reading zstd-compressed files such as {path} under this Spark profile (Spark reads \
             them from 4.1: {})",
            codec::SUFFIXES_CONF
        )));
    }
    Ok(())
}

/// The session's [`codec::SUFFIXES_CONF`] as a conf option of the line
/// formats. The codec it gives a file is part of the file's metadata, so it
/// joins the metadata cache key.
pub(crate) fn compression_suffixes_spec() -> crate::options::OptionSpec {
    crate::options::OptionSpec {
        name: codec::SUFFIXES_CONF,
        kind: crate::options::OptionKind::Enum(codec::SUFFIXES_VALUES),
        default: crate::options::OptionDefault::Str("extended"),
        conf_fallback: Some(codec::SUFFIXES_CONF),
        metadata_key: true,
        doc: "The session's `krishiv.io.text.compressionSuffixes` (from its Spark profile): which file suffixes name a compression codec.",
    }
}

/// The codec the write options ask for.
///
/// # Errors
/// `CODEC_NOT_AVAILABLE.WITH_AVAILABLE_CODECS_SUGGESTION` for an unknown name.
pub(crate) fn write_codec(
    o: &crate::options::FormatOptions,
) -> crate::error::Result<Option<codec::Codec>> {
    match o.str("compression").or_else(|| o.str("codec")) {
        None => Ok(None),
        Some(c) => codec::Codec::parse_write(c).ok_or_else(|| {
            krishiv_common::error::CodecNotAvailableWithAvailableCodecsSuggestion {
                codec_name: c.to_ascii_lowercase(),
                available_codecs: codec::WRITE_NAMES.join(", "),
            }
            .into()
        }),
    }
}

/// `encoder`, compressing its output if the write options name a codec.
///
/// # Errors
/// `CODEC_NOT_AVAILABLE.WITH_AVAILABLE_CODECS_SUGGESTION` for an unknown codec.
pub(crate) fn compress_output(
    encoder: Box<dyn crate::format::Encoder>,
    o: &crate::options::FormatOptions,
) -> crate::error::FormatResult<Box<dyn crate::format::Encoder>> {
    Ok(match write_codec(o)? {
        Some(c) => Box::new(codec::CompressedEncoder::new(encoder, c)),
        None => encoder,
    })
}

/// `base` (`.csv`) with the suffix of the write codec; the base alone for an
/// unknown codec (the writer reports it).
pub(crate) fn output_suffix(base: &str, o: &crate::options::FormatOptions) -> String {
    match write_codec(o) {
        Ok(Some(c)) => format!("{base}{}", c.suffix()),
        _ => base.to_owned(),
    }
}
