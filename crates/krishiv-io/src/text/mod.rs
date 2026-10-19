// SPDX-License-Identifier: Apache-2.0
//! What the line-oriented text formats share (CSV, JSON lines, `text`):
//!
//! - [`LineReader`]: a sans-IO reader of the lines of one split, with
//!   Hadoop's `LineRecordReader` split rule (a split that does not start at
//!   byte 0 skips its first, partial line; a split reads every line that
//!   starts at or before its end), so any split boundaries read each line
//!   exactly once.
//! - [`WholeReader`]: a whole file, for formats whose records may span lines
//!   (`multiLine`) or files that cannot be split (compressed).
//! - [`Charset`]: the `encoding` option; decoding replaces invalid bytes
//!   with U+FFFD, as Java's decoders do in Spark.
//! - [`Datetimes`]: dates and timestamps as text, with the session time
//!   zone, Spark's default grammars and the `dateFormat` / `timestampFormat`
//!   / `timestampNTZFormat` patterns.
//! - [`TextMetadata`]: these formats have no footer; their metadata is the
//!   file size.

mod datetime;
pub(crate) mod reader;

use std::borrow::Cow;
use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};

pub(crate) use self::datetime::Datetimes;
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
    /// The file size.
    pub size: u64,
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
    /// The metadata of the listed `file`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a compressed file.
    pub(crate) fn of(file: &crate::io::FileRef) -> crate::error::Result<Self> {
        check_plain(file.path.as_ref())?;
        Ok(Self { size: file.size })
    }

    /// The unit a split reads: the split's bytes, if it owns any line.
    /// `whole` files (`multiLine`) belong to the
    /// split that starts at byte 0.
    pub(crate) fn plan(
        &self,
        split: &crate::io::ByteRange,
        whole: bool,
    ) -> crate::format::UnitPlan {
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
mod tests;

/// Hadoop's compression suffixes: compressed files need a codec this build
/// does not have.
const COMPRESSED: &[&str] = &[".gz", ".deflate", ".bz2", ".lz4", ".snappy", ".zst"];

/// Checks that the file at `path` is not compressed.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a compressed file.
pub(crate) fn check_plain(path: &str) -> crate::error::Result<()> {
    if COMPRESSED.iter().any(|s| path.ends_with(s)) {
        return Err(unsupported(format!(
            "reading compressed files such as {path} (compression codecs are not implemented yet)"
        )));
    }
    Ok(())
}

/// Checks that the write options ask for no compression.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a codec.
pub(crate) fn check_write_codec(o: &crate::options::FormatOptions) -> crate::error::Result<()> {
    match o.str("compression").or_else(|| o.str("codec")) {
        None => Ok(()),
        Some(c)
            if matches!(
                c.trim().to_ascii_lowercase().as_str(),
                "none" | "uncompressed"
            ) =>
        {
            Ok(())
        }
        Some(c) => Err(unsupported(format!(
            "the compression codec {c} (writing compressed files is not implemented yet)"
        ))),
    }
}
