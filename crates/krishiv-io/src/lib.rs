// SPDX-License-Identifier: Apache-2.0
//! Object store and file format I/O (layer L2, ADR 0014, P2-T117).
//!
//! # Model
//! - **One [`FileFormat`] per file format**, implemented once: schema
//!   inference, metadata (footer) reading, unit planning with statistics
//!   pruning, and a **sans-IO** [`Decoder`] / [`Encoder`] pair. A format never
//!   touches `object_store`: its decoder asks for byte ranges
//!   ([`Step::NeedData`]) and is pushed the bytes; its encoder hands out bytes.
//! - **All I/O goes through one [`IoScheduler`]** (per process) and the
//!   per-query [`IoContext`]: range coalescing, an in-flight cap per store,
//!   jittered retries honouring `Retry-After`, per-attempt timeouts, request
//!   and byte counters, `if_match` / `if_unmodified_since` version checks on
//!   every read, one speculative tail read per footer, the file-metadata
//!   cache, and multipart uploads that abort on cancel or error.
//! - **The framework owns everything multi-file and cross-format**: Spark's
//!   split planning ([`split`]), schema evolution and case rules
//!   ([`SchemaMapper`], applied to predicates before pruning and to batches
//!   after decode), typed options ([`OptionSchema`]), and the scan and write
//!   drivers ([`ScanStream`], [`FileWriter`]) that the executor's file source
//!   and sink wrap.
//!
//! # Stability
//! Public but **unstable** (outside the semver-checked API) until a second
//! columnar format is built on it; then frozen and re-exported by the
//! connector SDK (P2-T117 decision 11).

#![forbid(unsafe_code)]

mod commit;
pub mod csv;
mod error;
mod format;
pub mod io;
pub mod json;
mod options;
pub mod parquet;
mod registry;
mod request;
pub mod runtime_filter;
mod scan;
mod scan_gate;
mod schema_map;
pub mod split;
mod stats;
pub mod text;
pub mod text_format;
mod write;

pub use commit::{DirectoryWrite, SUCCESS_MARKER, SaveMode};
pub use csv::CsvFormat;
pub use error::{FormatError, FormatResult, Result};
pub use format::{
    AsyncDecoder, Capabilities, Decoder, Encoder, FileFormat, FileMetadata, MetadataNeeds,
    PushReadAt, ReadAt, ScanUnit, ScanUnitMetrics, Splittable, Step, UnitPlan, WrittenFile,
};
pub use io::{
    ByteRange, FileRef, Held, IoConfig, IoContext, IoCounters, IoScheduler, SecretResolver,
    StoreId, StorePolicy, Tail, VersionToken,
};
pub use json::JsonFormat;
pub use options::{
    FILE_SOURCE_OPTIONS, FormatOptions, OptionDefault, OptionKind, OptionSchema, OptionSpec,
    OptionWarning,
};
pub use registry::{FormatRegistry, SPARK_FILE_SOURCES, is_spark_file_source};
pub use request::{
    CmpOp, Collation, ColumnPath, FieldMapping, MembershipFilter, MetadataCols, PushdownExpr,
    ROW_INDEX_COLUMN, RowFilterPolicy, RowLevel, RowSelection, ScalarValue, ScanRequest,
    project_batch, project_schema,
};
pub use runtime_filter::{
    DisabledReason, FilterCounts, FilterForm, FilterId, FilterSnapshot, KeyFilter,
    KeyFilterBuilder, KeyFilterOptions, Published, RowHash, RuntimeFilter, ScanFilter,
};
pub use scan::{ScanOptions, ScanStream};
pub use scan_gate::ScanGate;
pub use schema_map::{FIELD_ID_KEY, SchemaMapper};
pub use stats::{ColumnStatistics, FileStatistics, Precision, TriState, prune};
pub use text_format::TextFormat;
pub use write::{FileWriter, WriteOptions, WrittenFileInfo, spark_file_name};

#[cfg(test)]
mod tests;
