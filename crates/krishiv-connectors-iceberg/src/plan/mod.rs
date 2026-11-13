// SPDX-License-Identifier: Apache-2.0
//! Iceberg scan planning (P3-T026, PR-25a; `design/P3-T026-T027-iceberg-scan-and-commit.md`
//! §2): the snapshot a read sees ([`VersionSpec`], [`TableSnapshot`]), the manifest
//! list → manifests → entries pipeline with Krishiv's inheritance ([`inherit`]), and
//! the might-match evaluator ([`Pruner`]) over Krishiv-owned structs.
//!
//! Every type here is Krishiv's: iceberg-rust's types stay in the private
//! `iceberg_v0` module, which reads and parses the files and builds these values.
//! The evaluator does no I/O, so T027's conflict validation reuses it
//! ([`PruneMode::Conservative`]).

pub(crate) mod bounds;
pub(crate) mod deletes;
pub(crate) mod dv;
pub(crate) mod entry;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the write path of PR-36a (next commit) calls it")
)]
pub(crate) mod file_metrics;
pub(crate) mod metrics;
pub(crate) mod prune;
pub(crate) mod snapshot;
pub(crate) mod transform;
pub(crate) mod types;

#[cfg(test)]
mod eq_narrow_tests;
#[cfg(test)]
mod file_metrics_tests;
#[cfg(test)]
mod prop_tests;
#[cfg(test)]
pub(crate) mod tests;

pub use bounds::{decode, encode};
pub use deletes::{DELETE_FILE_PATH_ID, DELETE_FILE_POS_ID, DeleteIndex, TaskDeletes};
pub use dv::{decode_dv, encode_dv};
pub use entry::{
    ContentFile, ContentKind, DataFileEntry, DeleteFileEntry, DeleteKind, EntryStatus,
    FileFormatKind, FileScanTask, LiveEntry, ManifestContent, ManifestFileEntry, ManifestFilter,
    ParsedEntry, PartitionFieldSummary, PartitionValue, ScanPlan, inherit,
};
pub use metrics::{METRICS_COLUMN_PREFIX, METRICS_DEFAULT, MetricsMode, WriterMetrics};
pub use prune::{DEFAULT_TRUNCATE_LENGTH, PruneMode, PruneOptions, Pruner};
pub use snapshot::{RefInfo, RefKind, TableSnapshot, VersionSpec};
pub use transform::apply as apply_transform;
pub use types::{
    IcebergType, PartitionFieldInfo, PartitionSpecInfo, PartitionSpecs, SchemaField, Transform,
};
