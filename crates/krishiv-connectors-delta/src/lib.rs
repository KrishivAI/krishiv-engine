// SPDX-License-Identifier: Apache-2.0
//! Delta Lake connector (layer L5) on `delta_kernel` 0.30.1 with Krishiv's own `Engine`
//! (design note `P3-T021-delta.md`): the kernel reads the log, Krishiv reads the data.
//!
//! # Kernel features
//! `delta_kernel` is built with `default-features = false` and only `internal-api` (its Arrow
//! features pull `chrono-tz`, banned by ADR 0028). The `internal-api` items this crate uses are
//! listed here so a kernel bump reviews exactly those: `Snapshot::table_configuration`,
//! `TableConfiguration::{protocol, metadata, column_mapping_mode}`,
//! `Protocol::{min_reader_version, reader_features, writer_features}`,
//! `Metadata::{configuration, partition_columns}`, `StructField::physical_name` and the
//! `TableFeature` and `ColumnMappingMode` types ([`features`], [`table`]),
//! `Protocol::min_writer_version` ([`features::WriteProtocol`]); PR-37 adds the
//! create-table builder and `update_deletion_vectors`. [`features`] also reads the text of the
//! kernel's `"Feature '<name>' is not supported"` refusal (pinned by a test).
//!
//! # Error classes
//! Delta's own classes are vendored from delta-spark 4.4.1 into `krishiv-common`
//! (`vendored/delta-error-classes.json`), so a refusal reads as it does under Spark + Delta:
//! - feature gates: `DELTA_UNSUPPORTED_FEATURES_FOR_READ`, `DELTA_UNSUPPORTED_FEATURES_FOR_WRITE`,
//!   `DELTA_INVALID_PROTOCOL_VERSION`;
//! - writes: `DELTA_CANNOT_MODIFY_APPEND_ONLY`, `DELTA_METADATA_MISMATCH`,
//!   `DELTA_FAILED_TO_MERGE_FIELDS`, `DELTA_INVALID_ISOLATION_LEVEL`,
//!   `DELTA_REPLACE_WHERE_MISMATCH`, `DELTA_PATH_EXISTS`,
//!   `DELTA_CREATE_TABLE_WITH_NON_EMPTY_LOCATION`;
//! - time travel: `DELTA_VERSION_NOT_FOUND`, `DELTA_TIMESTAMP_EARLIER_THAN_COMMIT_RETENTION`,
//!   `DELTA_TIMESTAMP_GREATER_THAN_COMMIT`, `DELTA_TIMESTAMP_INVALID`;
//! - conflicts: `DELTA_CONCURRENT_APPEND`, `DELTA_CONCURRENT_DELETE_READ`,
//!   `DELTA_CONCURRENT_DELETE_DELETE`, `DELTA_PROTOCOL_CHANGED`, `DELTA_METADATA_CHANGED`,
//!   `DELTA_CONCURRENT_TRANSACTION`, `DELTA_MAX_COMMIT_RETRIES_EXCEEDED`.
//!
//! Kernel errors map to Krishiv's classes in [`error`]. Krishiv's own: `KRISHIV_CATALOG_LIMIT_EXCEEDED`
//! (a parse limit), `KRISHIV_UNSUPPORTED_FEATURE` (a commit on a store without conditional put).
//!
//! # Engine
//! [`engine::DeltaEngine`] is the kernel's `Engine`: Krishiv's storage, JSON and Parquet handlers
//! over `IoContext` and an evaluator over Arrow compute. Every kernel call runs on the process's
//! bounded log pool ([`engine::pool::LogPool`]) through [`engine::DeltaEngine::delta_blocking`].
//!
//! # Reading a table
//! [`table::DeltaTable`] is the `TableProvider` of one table at one version for one query: the
//! snapshot, Krishiv's reader allow-list ([`features`]), the live files, and the scan that reads
//! them through krishiv-io (column mapping, partition values, type widening, deletion vectors).
//!
//! # Committing
//! [`commit::commit`] commits what a write staged on the snapshot it read: Krishiv's writer
//! allow-list ([`features::check_write`]), one kernel transaction per attempt carrying the write's
//! id, the conflict checker against every winning commit ([`commit::conflict`], a port of
//! delta-spark's), the rebuild at the last checked version, and the landed check after a lost
//! answer (design note sections 4.2 and 5).
//!
//! # Writing
//! [`write`](mod@write) is the write path behind [`table::DeltaTable`]'s `TableProvider::writer` and
//! [`write::NewDeltaTable`]: data files with physical names and field ids, Krishiv's own
//! statistics, the overwrite modes' removes and Delta's `WRITE` commit (design note section 4).

#![forbid(unsafe_code)]

pub mod commit;
pub mod engine;
pub mod error;
pub mod features;
pub mod table;
pub mod write;

/// The local store the tests read through: the settled listing store on Unix, the plain one
/// elsewhere (as the session registers it; Windows port).
#[cfg(all(test, unix))]
pub(crate) use krishiv_io::SettledLocalFileSystem as TestLocalStore;
/// The local store the tests read through: the settled listing store on Unix, the plain one
/// elsewhere (as the session registers it; Windows port).
#[cfg(all(test, not(unix)))]
pub(crate) use object_store::local::LocalFileSystem as TestLocalStore;
