// SPDX-License-Identifier: Apache-2.0
//! The Parquet reader (P2-T080a): [`ParquetFormat`] over arrow-rs's
//! sans-IO `ParquetPushDecoder` (ADR 0014, E08). No custom decoders.
//!
//! - **Footer** (`meta.rs`): one speculative tail read through the
//!   scheduler, decoded by arrow-rs's metadata push decoder, checked against
//!   the file size, cached by the framework.
//! - **Pruning, stage 1** ([`FileFormat::plan_units`]): a split owns the
//!   row groups whose byte midpoint it holds; their min/max/null-count
//!   statistics prune with the (re-typed, file-level) predicate and the
//!   current runtime filters, which the decoder re-reads at each row-group
//!   boundary and prunes the row groups left with when a filter changed
//!   (a filter published while the scan runs). Only exact bounds prune
//!   (see `meta.rs`).
//! - **Decoding** (`decode.rs`): arrow-rs decodes the projected leaves of the
//!   surviving row groups, with the row selection mapped into each row
//!   group, strings as `Utf8View`; fetches are the framework's.
//! - **Corrupt data is an error.** The footer is checked before arrow-rs
//!   uses it, and the few arrow-rs assertions corrupt pages can still hit
//!   are contained at the decoder boundary (`contain`): a typed error,
//!   never a crashed task.
//! - **Strings are valid UTF-8.** A `STRING` value with invalid UTF-8 fails
//!   the scan with `INVALID_UTF8_STRING` (Spark 4.2 passes such bytes
//!   through as a string; Krishiv's strings are always valid UTF-8). It is
//!   a data error, not a corrupt file: `ignoreCorruptFiles` does not skip it.
//! - **Calendars** (`calendar.rs`): Spark's rebase rules, with the writer's
//!   Spark version deciding before the read mode. Values a rebase would
//!   change fail the scan (`KRISHIV_UNSUPPORTED_FEATURE`; the rebase is not
//!   implemented) and ambiguous ones fail as Spark's `EXCEPTION` mode does;
//!   every other value is read as stored.
//! - **Writing** (`write.rs`, P2-T082a): arrow-rs's `ArrowWriter` as a
//!   sans-IO encoder, with Spark's codecs, file names and calendar markers,
//!   page indexes always written.
//! - Page indexes, bloom filters and dictionary pruning on read are
//!   P2-T080b; runtime filters reach those stages through
//!   [`ScanFilter::predicate`](crate::ScanFilter::predicate) (statistics
//!   comparisons for page indexes, `Membership` for dictionary values and
//!   stored Bloom filters) once the reader exposes them.

mod calendar;
mod decode;
mod meta;
mod row_filter;
mod thrift_guard;
mod write;

use std::sync::Arc;

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use parquet::file::properties::WriterPropertiesBuilder;

pub use self::meta::ParquetMetadata;
use self::meta::SchemaOptions;
pub use self::write::SPARK_FILE_VERSION;
use crate::error::{FormatError, FormatResult};
use crate::format::{
    Capabilities, Decoder, Encoder, FileFormat, FileMetadata, MetadataNeeds, ScanUnit, Splittable,
    UnitPlan,
};
use crate::io::{ByteRange, FileRef, IoContext};
use crate::options::{FormatOptions, OptionDefault, OptionKind, OptionSchema, OptionSpec};
use crate::request::ScanRequest;
use crate::stats::{TriState, prune};

static OPTIONS: OptionSchema = OptionSchema {
    format: "parquet",
    options: &[
        OptionSpec {
            name: "binaryAsString",
            kind: OptionKind::Bool,
            default: OptionDefault::Bool(false),
            conf_fallback: Some("spark.sql.parquet.binaryAsString"),
            metadata_key: true,
            doc: "Read BINARY columns without a STRING annotation as strings.",
        },
        OptionSpec {
            name: "int96AsTimestamp",
            kind: OptionKind::Bool,
            default: OptionDefault::Bool(true),
            conf_fallback: Some("spark.sql.parquet.int96AsTimestamp"),
            metadata_key: true,
            doc: "Read INT96 columns as timestamps.",
        },
        OptionSpec {
            name: "datetimeRebaseMode",
            kind: OptionKind::Enum(REBASE_MODES),
            default: OptionDefault::Str("CORRECTED"),
            conf_fallback: Some("spark.sql.parquet.datetimeRebaseModeInRead"),
            metadata_key: false,
            doc: "Calendar of DATE and TIMESTAMP values (only CORRECTED rebases nothing).",
        },
        OptionSpec {
            name: "int96RebaseMode",
            kind: OptionKind::Enum(REBASE_MODES),
            default: OptionDefault::Str("CORRECTED"),
            conf_fallback: Some("spark.sql.parquet.int96RebaseModeInRead"),
            metadata_key: false,
            doc: "Calendar of INT96 timestamps (only CORRECTED rebases nothing).",
        },
        OptionSpec {
            name: "lateMaterialization",
            kind: OptionKind::Enum(&["auto", "on", "off"]),
            default: OptionDefault::Str("auto"),
            conf_fallback: Some("krishiv.io.lateMaterialization"),
            metadata_key: false,
            doc: "Read: evaluate the pushed filter before decoding the other columns \
                  (`auto` decides per row group).",
        },
        OptionSpec {
            name: "compression",
            kind: OptionKind::String,
            default: OptionDefault::None,
            conf_fallback: None,
            metadata_key: false,
            doc: "Write codec: none, uncompressed, snappy, gzip, lz4, lz4_raw, zstd, brotli \
                  (before parquet.compression).",
        },
        OptionSpec {
            name: "parquet.compression",
            kind: OptionKind::String,
            default: OptionDefault::Str("snappy"),
            conf_fallback: Some("spark.sql.parquet.compression.codec"),
            metadata_key: false,
            doc: "Write codec when `compression` is not set.",
        },
        OptionSpec {
            name: "parquet.block.size",
            kind: OptionKind::Int {
                min: 1,
                max: i64::MAX,
            },
            default: OptionDefault::Int(128 << 20),
            conf_fallback: None,
            metadata_key: false,
            doc: "Write: encoded bytes of a row group.",
        },
        OptionSpec {
            name: "parquet.page.size",
            kind: OptionKind::Int {
                min: 1,
                max: i64::MAX,
            },
            default: OptionDefault::Int(1 << 20),
            conf_fallback: None,
            metadata_key: false,
            doc: "Write: encoded bytes of a data page.",
        },
        OptionSpec {
            name: "parquet.enable.dictionary",
            kind: OptionKind::Bool,
            default: OptionDefault::Bool(true),
            conf_fallback: None,
            metadata_key: false,
            doc: "Write: dictionary-encode columns.",
        },
        OptionSpec {
            name: "parquet.writer.version",
            kind: OptionKind::Enum(&["v1", "v2", "PARQUET_1_0", "PARQUET_2_0"]),
            default: OptionDefault::Str("v1"),
            conf_fallback: None,
            metadata_key: false,
            doc: "Write: data page version (v1 or v2).",
        },
        OptionSpec {
            name: "outputTimestampType",
            kind: OptionKind::Enum(&["INT96", "TIMESTAMP_MICROS", "TIMESTAMP_MILLIS"]),
            default: OptionDefault::Str("TIMESTAMP_MICROS"),
            conf_fallback: Some("spark.sql.parquet.outputTimestampType"),
            metadata_key: false,
            doc: "Write: physical type of timestamps (INT96 cannot be written).",
        },
        OptionSpec {
            name: "datetimeRebaseModeInWrite",
            kind: OptionKind::Enum(REBASE_MODES),
            default: OptionDefault::Str("CORRECTED"),
            conf_fallback: Some("spark.sql.parquet.datetimeRebaseModeInWrite"),
            metadata_key: false,
            doc: "Write: calendar of DATE and TIMESTAMP values (only CORRECTED writes ancient values).",
        },
    ],
};

const REBASE_MODES: &[&str] = &["EXCEPTION", "CORRECTED", "LEGACY"];

/// The Parquet format (see the module docs).
#[derive(Debug, Default, Clone, Copy)]
pub struct ParquetFormat;

impl ParquetFormat {
    fn schema_options(o: &FormatOptions) -> SchemaOptions {
        SchemaOptions {
            binary_as_string: o.bool("binaryAsString").unwrap_or(false),
            int96_as_timestamp: o.bool("int96AsTimestamp").unwrap_or(true),
        }
    }

    /// An encoder of `schema` with `o`, its arrow-rs writer properties
    /// adjusted by `tweak` (tests cut row groups or mimic other writers
    /// this way); [`FileFormat::writer`] is this without a tweak.
    ///
    /// # Errors
    /// As [`FileFormat::check_write`].
    pub fn encoder_with(
        schema: &SchemaRef,
        o: &FormatOptions,
        tweak: impl FnOnce(WriterPropertiesBuilder) -> WriterPropertiesBuilder,
    ) -> FormatResult<Box<dyn Encoder>> {
        Ok(Box::new(write::ParquetEncoder::new(schema, o, tweak)?))
    }

    /// Describes a footer already decoded (a writer's, for its statistics).
    ///
    /// # Errors
    /// [`FormatError::CorruptFooter`] if the footer does not fit a
    /// `size`-byte file.
    pub fn describe(
        meta: &Arc<parquet::file::metadata::ParquetMetaData>,
        size: u64,
        o: &FormatOptions,
    ) -> FormatResult<ParquetMetadata> {
        meta::describe(meta, size, Self::schema_options(o))
    }
}

/// Runs arrow-rs code over untrusted bytes. arrow-rs asserts on some
/// corrupt inputs (an out-of-range dictionary index slices out of bounds):
/// such a panic is contained here and reported as the data error it is
/// (`what` says which step), so a corrupt file fails its scan with a typed
/// error instead of taking the task down. Callers drop the arrow-rs state
/// the panic interrupted.
pub(crate) fn contain<T>(
    what: &str,
    error: fn(String) -> FormatError,
    f: impl FnOnce() -> FormatResult<T>,
) -> FormatResult<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        tracing::warn!(what, message, "arrow-rs panicked on corrupt Parquet data");
        Err(error(format!("{what} failed on corrupt data: {message}")))
    })
}

fn downcast(m: Arc<dyn FileMetadata>) -> FormatResult<Arc<ParquetMetadata>> {
    let any: Arc<dyn std::any::Any + Send + Sync> = m;
    any.downcast::<ParquetMetadata>()
        .map_err(|_| FormatError::Corrupt("not Parquet metadata".to_owned()))
}

impl FileFormat for ParquetFormat {
    fn name(&self) -> &'static str {
        "parquet"
    }

    fn options(&self) -> &'static OptionSchema {
        &OPTIONS
    }

    fn capabilities(&self, _o: &FormatOptions) -> Capabilities {
        let mut c = Capabilities::new(Splittable::Yes);
        c.self_describing = true;
        c.statistics = true;
        c.row_index = true;
        c.selection = true;
        c.write = true;
        c
    }

    fn file_suffix(&self, o: &FormatOptions) -> String {
        // An invalid codec fails `check_write` before a file is named.
        let codec = write::Settings::new(&arrow_schema::Schema::empty(), o)
            .map(|s| s.extension())
            .unwrap_or_default();
        format!("{codec}.parquet")
    }

    fn read_metadata<'a>(
        &'a self,
        io: &'a IoContext,
        file: &'a FileRef,
        _needs: MetadataNeeds,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Arc<dyn FileMetadata>>> {
        Box::pin(async move {
            let m = meta::read(io, file, Self::schema_options(o)).await?;
            Ok(Arc::new(m) as Arc<dyn FileMetadata>)
        })
    }

    fn plan_units(
        &self,
        m: &dyn FileMetadata,
        split: &ByteRange,
        req: &ScanRequest,
    ) -> FormatResult<UnitPlan> {
        let any: &dyn std::any::Any = m;
        let meta = any
            .downcast_ref::<ParquetMetadata>()
            .ok_or_else(|| FormatError::Corrupt("not Parquet metadata".to_owned()))?;
        let runtime: Vec<_> = req
            .dynamic
            .iter()
            .filter_map(|d| Some((d.predicate()?, d.filter())))
            .collect();
        let mut plan = UnitPlan::default();
        for (index, rg) in meta.row_groups.iter().enumerate() {
            if !split.owns(&rg.bytes) {
                continue;
            }
            plan.owned += 1;
            if req
                .predicate
                .as_ref()
                .is_some_and(|p| prune(p, &rg.statistics) == TriState::False)
            {
                plan.pruned_by_statistics += 1;
                continue;
            }
            if let Some((_, f)) = runtime
                .iter()
                .find(|(p, _)| prune(p, &rg.statistics) == TriState::False)
            {
                f.stats().add_unit(rg.bytes.end - rg.bytes.start);
                plan.pruned_by_statistics += 1;
                continue;
            }
            plan.units.push(ScanUnit {
                index,
                first_row: rg.first_row,
                rows: rg.rows,
                bytes: rg.bytes.clone(),
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
            let meta = downcast(m)?;
            let d = decode::ParquetDecoder::new(&meta, &units, req, o)?;
            Ok(Box::new(d) as Box<dyn Decoder>)
        })
    }

    fn writer<'a>(
        &'a self,
        _io: &'a IoContext,
        schema: SchemaRef,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Encoder>>> {
        Box::pin(std::future::ready(Self::encoder_with(&schema, o, |p| p)))
    }

    fn check_write(&self, schema: &SchemaRef, o: &FormatOptions) -> FormatResult<()> {
        write::Settings::new(schema, o).map(drop)
    }
}

#[cfg(test)]
mod late_tests;
#[cfg(test)]
mod tests;
