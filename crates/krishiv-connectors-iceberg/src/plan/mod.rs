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
pub(crate) mod entry;
pub(crate) mod prune;
pub(crate) mod snapshot;
pub(crate) mod transform;
pub(crate) mod types;

#[cfg(test)]
mod prop_tests;
#[cfg(test)]
mod tests;

pub use bounds::{decode, encode};
pub use entry::{
    ContentFile, ContentKind, DataFileEntry, DeleteFileEntry, DeleteKind, EntryStatus,
    FileFormatKind, FileScanTask, LiveEntry, ManifestContent, ManifestFileEntry, ManifestFilter,
    ParsedEntry, PartitionFieldSummary, PartitionValue, ScanPlan, inherit,
};
pub use prune::{DEFAULT_TRUNCATE_LENGTH, PruneMode, PruneOptions, Pruner};
pub use snapshot::{RefInfo, RefKind, TableSnapshot, VersionSpec};
pub use transform::apply as apply_transform;
pub use types::{
    IcebergType, PartitionFieldInfo, PartitionSpecInfo, PartitionSpecs, SchemaField, Transform,
};
