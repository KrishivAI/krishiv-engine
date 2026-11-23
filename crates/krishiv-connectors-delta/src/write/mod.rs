// SPDX-License-Identifier: Apache-2.0
//! The Delta write path (design note P3-T021 section 4, PR-37a's second half):
//! [`DeltaWriteSink`], the sink a [`TableProvider::writer`](krishiv_exec::planner::TableProvider::writer)
//! returns for a Delta table, and [`NewDeltaTable`], the provider of a table a `CREATE TABLE ..
//! USING delta [AS SELECT]` creates.
//!
//! A write streams: each driver writes its rows as they arrive (`files`): one open Parquet
//! file per partition it has seen, rolled over at `maxRecordsPerFile` and at a target size,
//! buffering at most a row group and the upload's parts, all held in the driver's reservation
//! (the sink spills by encoding and shipping the buffered row groups). The first driver to need
//! it builds the write's kernel transaction on the snapshot the statement read (for a create,
//! the kernel's `create_table` with the table-features protocol) and takes its write state.
//! When the sink finalizes it:
//! 1. ends every driver's open files, with Krishiv's statistics ([`stats`]);
//! 2. selects the files the mode removes from the kernel's scan metadata and checks each
//!    exactly against the predicate (only TRUE removes): every live file (truncate), the
//!    files of the partitions the filter selects (`replaceWhere`, a static partition
//!    overwrite) or of the partitions written (dynamic partition overwrite);
//! 3. commits through the commit loop ([`crate::commit::commit`]) with the write's id, Delta's
//!    `WRITE` operation, its parameters and metrics; a create commits version 0 in its one
//!    transaction, and a lost race or a failure reads version 0: this write's id there means it
//!    landed;
//! 4. starts a checkpoint when the commits since the last one reach `delta.checkpointInterval`
//!    (default 10, delta-spark's `DeltaConfig.scala:480-485`), detached from the statement: a
//!    failed or cut-short checkpoint is logged and counted ([`checkpoint_failures`]), never
//!    raised. At most one runs per table root ([`checkpoints_skipped`] counts the others),
//!    within a time budget, and it is cancelled when the runtime drops it.
//!
//! Data files are invisible until the commit. A write that fails before its commit attempt
//! (or whose create surely did not land) deletes its files, best effort; a cancelled write's
//! files (during its batches or its `finalize`, up to the commit attempt) are deleted in the
//! background. From the commit attempt on, every failure of an existing table's commit
//! (conflicts and exhausted retries included) keeps the files, as delta-spark's are kept: the
//! commit loop does not tell a surely-lost commit from an unknown outcome. A file left behind
//! is an orphan no snapshot reads.
//!
//! # Refusals (before any file is written)
//! - Krishiv's writer allow-list ([`crate::features::check_write`]).
//! - A removing mode on an append-only table: `DELTA_CANNOT_MODIFY_APPEND_ONLY`.
//! - A removing mode on a table with the change data feed enabled (the kernel refuses adds with
//!   removes there, design section 6.2): `UNSUPPORTED_FEATURE.TABLE_OPERATION`.
//! - A filter on a data column: `UNSUPPORTED_FEATURE.TABLE_OPERATION` ([`filter`]).
//!
//! # Known differences
//! - Krishiv-created tables carry the table-features protocol (writer 7) where delta-spark
//!   writes (1, 2) for a featureless table (design section 2.1).
//! - A CTAS commit's operation is the kernel's `CREATE TABLE`, not `CREATE TABLE AS SELECT`;
//!   its parameters are the kernel's.
//! - `isDynamicPartitionOverwrite` is the string `"true"` (the kernel's parameters are strings).
//! - Files are named `part-00000-<uuid>.c000.snappy.parquet` for every task.
//! - The target file size is a constant 128 MiB (a `krishiv.delta.*` key is a follow-up).
//! - Checkpoints are written without the `.crc` and V2 checkpoints without sidecars.
//! - `DELTA_REPLACE_WHERE_MISMATCH`'s nested message names the constraint by the filter's text,
//!   where delta-spark prints its expression tree.

pub mod dml;
mod files;
pub mod filter;
pub mod merge;
pub(crate) mod sink;
pub mod stats;
#[cfg(test)]
pub(crate) mod test_hooks;

pub use self::sink::{DeltaWriteSink, NewDeltaTable, checkpoint_failures, checkpoints_skipped};

/// The internal write option a session sets when the statement reads the table it writes
/// (`INSERT INTO t SELECT .. FROM t`): the commit then is not a blind append and is checked as
/// a whole-table read.
pub const READS_TARGET_OPTION: &str = "__krishiv_delta_reads_target";

/// The write option naming the filter's text for `commitInfo.operationParameters.predicate`.
pub const PREDICATE_TEXT_OPTION: &str = "__krishiv_delta_predicate";

#[cfg(test)]
mod tests;
