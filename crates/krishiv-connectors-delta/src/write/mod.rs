// SPDX-License-Identifier: Apache-2.0
//! The Delta write path (design note P3-T021 section 4, PR-37a's second half):
//! [`DeltaWriteSink`], the sink a [`TableProvider::writer`](krishiv_exec::planner::TableProvider::writer)
//! returns for a Delta table, and [`NewDeltaTable`], the provider of a table a `CREATE TABLE ..
//! USING delta [AS SELECT]` creates.
//!
//! A write buffers its rows under the query's memory (the sink is not spillable: a write larger
//! than the query's memory fails with the memory class), and when the sink finalizes:
//! 1. builds a kernel transaction on the snapshot the statement read (for a create, the
//!    kernel's `create_table` with the table-features protocol) and takes its write state;
//! 2. writes one Parquet file per partition ([`files`]), with Krishiv's statistics ([`stats`]);
//! 3. selects the files the mode removes from the kernel's scan metadata: every live file
//!    (truncate), the files of the partitions the filter selects (`replaceWhere`, a static
//!    partition overwrite) or of the partitions written (dynamic partition overwrite);
//! 4. commits through the commit loop ([`crate::commit::commit`]) with the write's id, Delta's
//!    `WRITE` operation, its parameters and metrics; a create commits version 0 in one
//!    transaction (the loop does not apply: a lost race means the table now exists);
//! 5. writes a checkpoint when the commits since the last one reach `delta.checkpointInterval`
//!    (default 10, delta-spark's `DeltaConfig.scala:480-485`); a failed checkpoint never fails
//!    the statement.
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
//! - `DELTA_REPLACE_WHERE_MISMATCH`'s nested message names the constraint by the filter's text,
//!   where delta-spark prints its expression tree.

mod files;
pub mod filter;
pub(crate) mod sink;
pub mod stats;

pub use self::sink::{DeltaWriteSink, NewDeltaTable};

/// The internal write option a session sets when the statement reads the table it writes
/// (`INSERT INTO t SELECT .. FROM t`): the commit then is not a blind append and is checked as
/// a whole-table read.
pub const READS_TARGET_OPTION: &str = "__krishiv_delta_reads_target";

/// The write option naming the filter's text for `commitInfo.operationParameters.predicate`.
pub const PREDICATE_TEXT_OPTION: &str = "__krishiv_delta_predicate";

#[cfg(test)]
mod tests;
