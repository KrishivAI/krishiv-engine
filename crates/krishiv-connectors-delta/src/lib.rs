// SPDX-License-Identifier: Apache-2.0
//! Delta Lake connector (layer L5) on `delta_kernel` 0.30.1 with Krishiv's own `Engine`
//! (design note `P3-T021-delta.md`): the kernel reads the log, Krishiv reads the data.
//!
//! # Kernel features
//! `delta_kernel` is built with `default-features = false` and only `internal-api` (its Arrow
//! features pull `chrono-tz`, banned by ADR 0028). The `internal-api` items this crate uses are
//! listed here so a kernel bump reviews exactly those: none yet (PR-08b brings
//! `Snapshot::table_configuration`, PR-37 the create-table builder and `update_deletion_vectors`).
//!
//! # Error classes
//! Delta's own classes are vendored from delta-spark 4.4.1 into `krishiv-common`
//! (`vendored/delta-error-classes.json`), so a refusal reads as it does under Spark + Delta:
//! - feature gates: `DELTA_UNSUPPORTED_FEATURES_FOR_READ`, `DELTA_UNSUPPORTED_FEATURES_FOR_WRITE`,
//!   `DELTA_INVALID_PROTOCOL_VERSION`;
//! - writes: `DELTA_CANNOT_MODIFY_APPEND_ONLY`, `DELTA_METADATA_MISMATCH`,
//!   `DELTA_FAILED_TO_MERGE_FIELDS`, `DELTA_INVALID_ISOLATION_LEVEL`;
//! - time travel: `DELTA_VERSION_NOT_FOUND`, `DELTA_TIMESTAMP_EARLIER_THAN_COMMIT_RETENTION`,
//!   `DELTA_TIMESTAMP_GREATER_THAN_COMMIT`, `DELTA_TIMESTAMP_INVALID`;
//! - conflicts: `DELTA_CONCURRENT_APPEND`, `DELTA_CONCURRENT_DELETE_READ`,
//!   `DELTA_CONCURRENT_DELETE_DELETE`, `DELTA_PROTOCOL_CHANGED`, `DELTA_METADATA_CHANGED`,
//!   `DELTA_CONCURRENT_TRANSACTION`, `DELTA_MAX_COMMIT_RETRIES_EXCEEDED`.
//!
//! Kernel errors map to Krishiv's classes in [`error`].

#![forbid(unsafe_code)]

pub mod error;
