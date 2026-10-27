// SPDX-License-Identifier: Apache-2.0
//! Spark's Avro format: [`AvroFormat`], a container-file reader and writer
//! with Spark's options (see `options`) and Spark's type mapping (see
//! `schema`).
//!
//! Files are self-describing (the header carries the writer schema) and
//! splittable at block (sync marker) boundaries: the metadata pass finds
//! every block's bytes, first row and record count from the block headers
//! and sync markers alone (it never reads a block's records), and a split
//! owns the blocks whose midpoint it holds. The decoder reads one owned
//! block at a time (see `decode`); the writer frames compressed blocks (see
//! `encode`). Whole-file row and byte counts are reported, but there are no
//! column statistics, page pruning or late materialisation;
//! `_metadata.row_index` and file-relative row selections are honoured from
//! the block index.
//!
//! Typed refusals: the `zstandard` block codec (reads and writes) is
//! `CODEC_NOT_AVAILABLE.WITH_AVAILABLE_CODECS_SUGGESTION` (the `apache-avro`
//! codec needs zstd 0.13, which carries RUSTSEC-2026-0391, so it is not
//! built); the `bzip2` and `xz` block codecs and `LEGACY` rebase of dates
//! and timestamps before 1582-10-15 are `KRISHIV_UNSUPPORTED_FEATURE`.

mod decode;
mod encode;
mod options;
mod schema;

#[cfg(test)]
mod tests;

use std::ops::Range;
use std::sync::Arc;

use apache_avro::Schema as AvroSchema;
use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::CodecNotAvailableWithAvailableCodecsSuggestion;

use self::decode::{AvroDecoder, Parsed, parse_header, parse_long};
use self::options::{AvroOptions, OPTIONS, Rebase};
use self::schema::{fields_of_record, schema_kind};
use crate::error::{FormatError, FormatResult, unsupported};
use crate::format::{
    Capabilities, Decoder, Encoder, FileFormat, FileMetadata, MetadataNeeds, ScanUnit, Splittable,
    UnitPlan,
};
use crate::io::{ByteRange, FileRef, IoContext};
use crate::options::FormatOptions;
use crate::request::ScanRequest;
use crate::stats::{FileStatistics, Precision};

pub(crate) use self::options::Codec;

/// Bytes fetched per read while parsing the header.
const HEADER_CHUNK: u64 = 64 << 10;

/// The most bytes of a block's count and size (two zigzag longs).
const BLOCK_HEAD: usize = 20;

/// Spark's `org.apache.spark.version` header key.
const SPARK_VERSION_KEY: &[u8] = b"org.apache.spark.version";

/// Spark's `org.apache.spark.legacyDateTime` header key (set by writers in
/// `LEGACY` rebase mode).
const SPARK_LEGACY_DATETIME_KEY: &[u8] = b"org.apache.spark.legacyDateTime";

/// A corrupt-header error.
fn corrupt_footer(reason: impl Into<String>) -> FormatError {
    FormatError::CorruptFooter(reason.into())
}

/// The `apache-avro` codec of `codec`.
///
/// # Errors
/// `CODEC_NOT_AVAILABLE.WITH_AVAILABLE_CODECS_SUGGESTION` for zstandard;
/// `KRISHIV_UNSUPPORTED_FEATURE` for `bzip2` and `xz`.
fn block_codec(codec: Codec) -> FormatResult<apache_avro::Codec> {
    use apache_avro::Codec as AvCodec;
    match codec {
        Codec::Null => Ok(AvCodec::Null),
        Codec::Deflate => Ok(AvCodec::Deflate(apache_avro::DeflateSettings::default())),
        Codec::Snappy => Ok(AvCodec::Snappy),
        Codec::Zstandard => Err(FormatError::Other(
            CodecNotAvailableWithAvailableCodecsSuggestion {
                codec_name: codec.header_name().to_owned(),
                available_codecs: "uncompressed, deflate, snappy".to_owned(),
            }
            .into(),
        )),
        Codec::Bzip2 | Codec::Xz => Err(FormatError::Other(unsupported(format!(
            "the Avro {} codec (only uncompressed, snappy and deflate are supported)",
            codec.header_name()
        )))),
    }
}

/// The parsed metadata of one Avro file.
#[derive(Debug)]
pub(crate) struct AvroMetadata {
    /// The file's size.
    pub size: u64,
    /// The file's (writer) schema.
    pub writer: AvroSchema,
    /// The schema records resolve against (`avroSchema`, else the writer).
    pub reader: AvroSchema,
    /// The Arrow schema of the reader.
    pub arrow: SchemaRef,
    /// The file's block codec.
    pub codec: Codec,
    /// The file's sync marker.
    pub sync: [u8; 16],
    /// How the writer says its dates and timestamps were written (Spark's
    /// `datetimeRebaseSpec`): `Some` when the header names a Spark version,
    /// else the `datetimeRebaseMode` option applies.
    pub writer_rebase: Option<Rebase>,
    /// Every block: its bytes, its records' bytes, rows before it, count.
    pub blocks: Vec<ScannedBlock>,
    /// Total records.
    pub rows: u64,
}

/// One scanned block.
#[derive(Debug, Clone)]
pub(crate) struct ScannedBlock {
    /// The block's bytes in the file (count and size varints, data, sync).
    pub bytes: Range<u64>,
    /// The records' bytes within `bytes`.
    pub data: Range<u64>,
    /// Rows before this block.
    pub first_row: u64,
    /// Records the block holds.
    pub count: u64,
}

impl FileMetadata for AvroMetadata {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.arrow)
    }

    fn statistics(&self) -> FileStatistics {
        FileStatistics {
            rows: Precision::Exact(self.rows),
            bytes: Precision::Exact(self.size),
            ..FileStatistics::default()
        }
    }

    fn heap_size(&self) -> usize {
        let schemas = self.writer.canonical_form().len()
            + self.reader.canonical_form().len()
            + self
                .arrow
                .fields()
                .iter()
                .map(|f| f.name().len() + 64)
                .sum::<usize>();
        size_of::<Self>()
            + self.blocks.len() * size_of::<ScannedBlock>()
            + self.sync.len()
            + schemas
    }
}

/// The reader schema and Arrow schema of `avro_schema` JSON.
///
/// # Errors
/// `KRISHIV_INVALID_FORMAT_OPTION_VALUE` for a schema that does not parse
/// or is not a record; `KRISHIV_UNSUPPORTED_FEATURE` for a schema with no
/// Arrow mapping.
fn reader_of(json: &str) -> FormatResult<(AvroSchema, SchemaRef)> {
    use crate::text::invalid_option;
    let parsed = AvroSchema::parse_str(json).map_err(|e| {
        FormatError::Other(invalid_option(
            "avro",
            "avroSchema",
            json,
            &format!("the schema does not parse: {e}"),
        ))
    })?;
    if !matches!(parsed, AvroSchema::Record(_)) {
        return Err(FormatError::Other(invalid_option(
            "avro",
            "avroSchema",
            json,
            "the schema is not a record",
        )));
    }
    let fields = fields_of_record(&parsed)?;
    Ok((parsed, Arc::new(arrow_schema::Schema::new(fields))))
}

/// The bytes `start..end` of `file` (`end` is clamped to its size).
///
/// # Errors
/// What the scheduler reports (storage failures, never corruption);
/// `FAILED_READ_FILE.CANNOT_READ_FILE_FOOTER` for a short read.
async fn fetch(io: &IoContext, file: &FileRef, start: u64, end: u64) -> FormatResult<Vec<u8>> {
    let end = end.min(file.size);
    if start >= end {
        return Ok(Vec::new());
    }
    let chunks = io
        .get_ranges(file, std::slice::from_ref(&(start..end)))
        .await
        .map_err(FormatError::Other)?;
    let mut out = Vec::with_capacity(usize::try_from(end - start).unwrap_or(0));
    for bytes in chunks {
        out.extend_from_slice(&bytes);
    }
    if (out.len() as u64) != end - start {
        return Err(corrupt_footer("a short read of the block index"));
    }
    Ok(out)
}

/// The metadata of `file`: its header, then every block header.
///
/// # Errors
/// `FAILED_READ_FILE.CANNOT_READ_FILE_FOOTER` for a truncated or invalid
/// header or block; `KRISHIV_UNSUPPORTED_FEATURE` for a codec this build
/// cannot read.
async fn read_meta(io: &IoContext, file: &FileRef, o: &AvroOptions) -> FormatResult<AvroMetadata> {
    if file.size == 0 {
        return Err(corrupt_footer("an empty file"));
    }
    // The header: fetch from the start until it parses.
    let mut buf: Vec<u8> = Vec::new();
    let (meta, header_len) = loop {
        let from = buf.len() as u64;
        buf.extend_from_slice(&fetch(io, file, from, from + HEADER_CHUNK).await?);
        match parse_header(&buf) {
            Parsed::Need => {
                if (buf.len() as u64) >= file.size {
                    return Err(corrupt_footer("a truncated header"));
                }
            }
            Parsed::BadMagic => {
                return Err(corrupt_footer("not an Avro container file (bad magic)"));
            }
            Parsed::Done((meta, len)) => break (meta, len),
        }
    };
    let schema_json = meta
        .get(b"avro.schema".as_slice())
        .ok_or_else(|| corrupt_footer("a header without avro.schema"))?;
    let schema_json = std::str::from_utf8(schema_json)
        .map_err(|_| corrupt_footer("an avro.schema that is not UTF-8"))?;
    let writer = AvroSchema::parse_str(schema_json)
        .map_err(|e| corrupt_footer(format!("an unreadable avro.schema: {e}")))?;
    if !matches!(writer, AvroSchema::Record(_)) {
        return Err(corrupt_footer(format!(
            "a top-level {} schema (only records are supported)",
            schema_kind(&writer)
        )));
    }
    let (reader, arrow) = match &o.avro_schema {
        Some(json) => reader_of(json)?,
        None => {
            let fields = fields_of_record(&writer)?;
            (writer.clone(), Arc::new(arrow_schema::Schema::new(fields)))
        }
    };
    let codec_name = meta
        .get(b"avro.codec".as_slice())
        .map(|c| {
            std::str::from_utf8(c).map_err(|_| corrupt_footer("an avro.codec that is not UTF-8"))
        })
        .transpose()?
        .unwrap_or("null");
    let codec = match codec_name {
        "null" | "uncompressed" => Codec::Null,
        "deflate" => Codec::Deflate,
        "snappy" => Codec::Snappy,
        "zstandard" | "zstd" => Codec::Zstandard,
        "bzip2" => Codec::Bzip2,
        "xz" => Codec::Xz,
        other => return Err(corrupt_footer(format!("the unknown codec {other}"))),
    };
    block_codec(codec)?;
    if buf.len() < header_len + 16 {
        return Err(corrupt_footer("a truncated header (no sync marker)"));
    }
    let sync: [u8; 16] = buf[header_len..header_len + 16]
        .try_into()
        .map_err(|_| corrupt_footer("a truncated header (no sync marker)"))?;
    // Spark names its version in the header and marks files written in the
    // LEGACY rebase mode; such files carry their own rebase mode (as Spark's
    // `datetimeRebaseSpec`), else the option applies.
    let writer_rebase = meta.get(SPARK_VERSION_KEY).map(|version| {
        if version.as_slice() < b"3.0.0".as_slice() || meta.contains_key(SPARK_LEGACY_DATETIME_KEY)
        {
            Rebase::Legacy
        } else {
            Rebase::Corrected
        }
    });
    // The blocks: each header, then only the sync marker that ends the
    // block and the next block's header, never the block's records.
    let mut blocks = Vec::new();
    let mut rows = 0_u64;
    let mut pos = u64::try_from(header_len)
        .map_err(|_| corrupt_footer("a header too long"))?
        .checked_add(16)
        .ok_or_else(|| corrupt_footer("a header too long"))?;
    let mut head = match buf.get(usize::try_from(pos).unwrap_or(usize::MAX)..) {
        Some(rest) if rest.len() as u64 >= (BLOCK_HEAD as u64).min(file.size - pos) => {
            rest[..rest.len().min(BLOCK_HEAD)].to_vec()
        }
        _ => fetch(io, file, pos, pos + BLOCK_HEAD as u64).await?,
    };
    drop(buf);
    while pos < file.size {
        let Some((count, n)) = parse_long(&head, 0) else {
            return Err(corrupt_footer("a truncated block header"));
        };
        let count =
            u64::try_from(count).map_err(|_| corrupt_footer("a block with a negative count"))?;
        let Some((size, m)) = parse_long(&head, n) else {
            return Err(corrupt_footer("a truncated block header"));
        };
        let size =
            u64::try_from(size).map_err(|_| corrupt_footer("a block with a negative size"))?;
        let (n, m) = (
            u64::try_from(n).map_err(|_| corrupt_footer("a block header too long"))?,
            u64::try_from(m).map_err(|_| corrupt_footer("a block header too long"))?,
        );
        let total = n
            .checked_add(m)
            .and_then(|v| v.checked_add(size))
            .and_then(|v| v.checked_add(16))
            .ok_or_else(|| corrupt_footer("a block too large"))?;
        let end = pos
            .checked_add(total)
            .ok_or_else(|| corrupt_footer("a block too large"))?;
        if end > file.size {
            return Err(corrupt_footer("a truncated block"));
        }
        let tail = fetch(io, file, end - 16, end + BLOCK_HEAD as u64).await?;
        if tail.get(..16) != Some(sync.as_slice()) {
            return Err(corrupt_footer("a block with a bad sync marker"));
        }
        let data_start = end - 16 - size;
        blocks.push(ScannedBlock {
            bytes: pos..end,
            data: data_start..data_start + size,
            first_row: rows,
            count,
        });
        rows += count;
        pos = end;
        head = tail[16..].to_vec();
    }
    Ok(AvroMetadata {
        size: file.size,
        writer,
        reader,
        arrow,
        codec,
        sync,
        writer_rebase,
        blocks,
        rows,
    })
}

/// Spark's Avro format (see the module docs).
#[derive(Debug, Default, Clone, Copy)]
pub struct AvroFormat;

impl FileFormat for AvroFormat {
    fn name(&self) -> &'static str {
        "avro"
    }

    fn options(&self) -> &'static crate::options::OptionSchema {
        &OPTIONS
    }

    fn capabilities(&self, _o: &FormatOptions) -> Capabilities {
        let mut c = Capabilities::new(Splittable::Yes);
        c.self_describing = true;
        c.row_index = true;
        c.selection = true;
        c.write = true;
        c
    }

    fn file_suffix(&self, _o: &FormatOptions) -> String {
        ".avro".to_owned()
    }

    fn infer_schema<'a>(
        &'a self,
        io: &'a IoContext,
        files: &'a [FileRef],
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<SchemaRef>> {
        Box::pin(async move {
            let opts = AvroOptions::new(o, false)?;
            if let Some(json) = &opts.avro_schema {
                return Ok(reader_of(json)?.1);
            }
            let Some(first) = files.first() else {
                return Err(FormatError::Other(crate::error::internal(
                    "schema inference without files".to_owned(),
                )));
            };
            let m = io
                .read_metadata(self, first, MetadataNeeds::default(), o)
                .await?;
            Ok(m.schema())
        })
    }

    fn read_metadata<'a>(
        &'a self,
        io: &'a IoContext,
        file: &'a FileRef,
        _needs: MetadataNeeds,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Arc<dyn FileMetadata>>> {
        Box::pin(async move {
            let opts = AvroOptions::new(o, false)?;
            let m = read_meta(io, file, &opts).await?;
            Ok(Arc::new(m) as Arc<dyn FileMetadata>)
        })
    }

    fn plan_units(
        &self,
        m: &dyn FileMetadata,
        split: &ByteRange,
        _req: &ScanRequest,
    ) -> FormatResult<UnitPlan> {
        let any: &dyn std::any::Any = m;
        let meta = any
            .downcast_ref::<AvroMetadata>()
            .ok_or_else(|| FormatError::Corrupt("not Avro metadata".to_owned()))?;
        let mut plan = UnitPlan::default();
        for (index, block) in meta.blocks.iter().enumerate() {
            if !split.owns(&block.bytes) {
                continue;
            }
            plan.owned += 1;
            plan.units.push(ScanUnit {
                index,
                first_row: block.first_row,
                rows: block.count,
                bytes: block.bytes.clone(),
            });
        }
        Ok(plan)
    }

    fn open<'a>(
        &'a self,
        _io: &'a IoContext,
        m: Arc<dyn FileMetadata>,
        units: UnitPlan,
        req: &'a ScanRequest,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Decoder>>> {
        Box::pin(async move {
            let any: Arc<dyn std::any::Any + Send + Sync> = m;
            let meta = any
                .downcast::<AvroMetadata>()
                .map_err(|_| FormatError::Corrupt("not Avro metadata".to_owned()))?;
            let opts = AvroOptions::new(o, false)?;
            let d = AvroDecoder::new(&meta, &units, req, &opts)?;
            Ok(Box::new(d) as Box<dyn Decoder>)
        })
    }

    fn writer<'a>(
        &'a self,
        _io: &'a IoContext,
        schema: SchemaRef,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Encoder>>> {
        Box::pin(std::future::ready(
            encode::AvroEncoder::new(&schema, o).map(|e| Box::new(e) as Box<dyn Encoder>),
        ))
    }

    fn check_write(&self, schema: &SchemaRef, o: &FormatOptions) -> FormatResult<()> {
        encode::AvroEncoder::new(schema, o).map(drop)
    }
}
