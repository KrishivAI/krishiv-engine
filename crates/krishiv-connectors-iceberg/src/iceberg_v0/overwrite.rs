// SPDX-License-Identifier: Apache-2.0
//! `INSERT OVERWRITE` into an Iceberg table (P3-T027, PR-36b part 2b): the
//! sink [`IcebergTableProvider::writer`](krishiv_exec::planner::TableProvider::writer)
//! returns for a truncate, an overwrite by filter and a dynamic overwrite,
//! as Spark's `SparkWrite` commits them (iceberg-spark 1.12.0; notes 36b2
//! §1, §2, §6):
//!
//! - **Truncate** is an overwrite by `alwaysTrue` (Spark's
//!   `SupportsOverwrite.truncate`), and becomes a dynamic overwrite when the
//!   write option `overwrite-mode` is `dynamic` (`SparkWriteBuilder.java:103-118`).
//! - **Overwrite by filter** (`table.newOverwrite().overwriteByRowFilter(expr)`,
//!   `SparkWrite.java:381-390`): the request's filter translated exactly
//!   ([`WriteRequest::overwrite_filter_pushdown`]); one that does not
//!   translate is refused (`KRISHIV_UNSUPPORTED_FEATURE`), never widened,
//!   and so is one holding a part the strict evaluator cannot answer
//!   (`Pred::unsupported`: `LIKE '%a%'`, `x < NaN`, `IS NULL` on a struct;
//!   review 36b2b B1).
//!   `filter.rs` removes every file whose rows all match, on every attempt's
//!   base; a data file some of whose rows match fails with Java's "Cannot
//!   delete file where some, but not all, rows match filter %s: %s"
//!   (`KRISHIV_ICEBERG_OVERWRITE_PARTIAL_FILE`, SQLSTATE 22000: Spark
//!   surfaces this `ValidationException` with no class, so the class is
//!   Krishiv's, with Java's text; a data exception, since the files do not
//!   line up with the filter and a retry fails the same way, where the
//!   commit conflict's 40001 invites a retry). Operation `overwrite`, or
//!   `delete` when no row is written.
//! - **Dynamic overwrite** (`table.newReplacePartitions()`,
//!   `SparkWrite.java:338-365`): the partitions (spec id and tuple) of the
//!   written files replaced, every live data and delete file in them removed;
//!   a table whose spec has no field is replaced whole (`deleteByRowFilter(
//!   alwaysTrue)`, `BaseReplacePartitions.java:113-117`); operation
//!   `overwrite`, summary `replace-partitions=true`; a write of no rows skips
//!   the commit ("Dynamic overwrite is empty, skipping commit",
//!   `SparkWrite.java:341-344`), where a static overwrite commits.
//! - **Validation** only when the write option `isolation-level` is set
//!   (`serializable` or `snapshot`, any case; Spark reads no table property
//!   for INSERT OVERWRITE, `SparkWriteConf.java:484-488`), from the write
//!   option `validate-from-snapshot-id`, on every attempt. As Java: without
//!   that option the starting snapshot is null and the **whole history** is
//!   validated (which fails on nearly every table with data, in Spark too);
//!   Krishiv then names the option in the conflict's message. `serializable`
//!   checks added data files, both levels the added delete files and the
//!   removed data files, over the row filter (`BaseOverwriteFiles.validate`,
//!   `:136-179`, no conflict filter set, no files deleted by path) or the
//!   replaced partitions (`BaseReplacePartitions.validate`, `:89-110`; every
//!   row of an unpartitioned table). With no `isolation-level` nothing is
//!   validated: a rebase re-applies the filter or the partitions to the new
//!   base (`filter.rs`), so a concurrent append into an overwritten
//!   partition is removed with it.
//!
//! Divergences: the filter and partition texts in messages are Krishiv's
//! renderings (Java's `Expression` and `PartitionSet` `toString` were not
//! read); the checks run in `validate.rs`'s fixed order (Java's
//! `ReplacePartitions` checks removed files before added deletes); a spec of
//! only `void` fields is unpartitioned, as `PartitionSpec.isUnpartitioned` (apache-iceberg
//! 1.12.0): a dynamic overwrite under it replaces every data file, whatever spec it
//! was written under, and validates over `alwaysTrue`.
//! Partition
//! tuples are compared by value at the spec's current type
//! (`filter::partition_key`), so a promoted column's old 4-byte tuples match
//! new 8-byte ones; a rewritten manifest writes them at the promoted type.

use std::sync::Arc;

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::planner::{WriteMode, WriteRequest};
use krishiv_io::PushdownExpr;

use super::filter::Removals;
use super::scan::schema_fields;
use super::sink::{IcebergWriteSink, Overwrite, WriteTarget};
use super::validate::{Checks, Isolation, Validation};
use crate::plan::strict::Pred;

/// The write option naming the isolation level (Java's
/// `SparkWriteOptions.ISOLATION_LEVEL`).
pub(crate) const ISOLATION_LEVEL: &str = "isolation-level";
/// The write option naming the snapshot validation starts from.
pub(crate) const VALIDATE_FROM_SNAPSHOT_ID: &str = "validate-from-snapshot-id";
/// The write option that turns a truncate into a dynamic overwrite.
pub(crate) const OVERWRITE_MODE: &str = "overwrite-mode";

/// A write option, its key in any case (Spark's
/// `CaseInsensitiveStringMap`).
fn option<'r>(request: &'r WriteRequest, key: &str) -> Option<&'r str> {
    request
        .options
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(key))
        .map(|(_, v)| v.as_str())
}

fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

/// The sink of the overwrite `request` (module docs).
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a filter that does not translate
/// exactly or holds a part the strict evaluator cannot answer (a substring
/// match, a range comparison with NaN, a non-primitive column), an unknown isolation level or a `validate-from-snapshot-id`
/// that is not a number; [`IcebergWriteSink::try_rewrite`]'s errors.
pub(crate) fn sink(
    target: Arc<WriteTarget>,
    request: &WriteRequest,
) -> Result<IcebergWriteSink, KrishivError> {
    let dynamic_mode =
        option(request, OVERWRITE_MODE).is_some_and(|m| m.eq_ignore_ascii_case("dynamic"));
    let filter = match &request.mode {
        WriteMode::Truncate => Some(PushdownExpr::Constant(true)),
        WriteMode::OverwriteByFilter(_) => Some(request.overwrite_filter_pushdown().ok_or_else(|| {
            unsupported(format!(
                "an overwrite of the Iceberg table {} by a filter that does not translate to an \
                 Iceberg expression exactly",
                request.table_name
            ))
        })?),
        WriteMode::OverwriteDynamic => None,
        other => {
            return Err(unsupported(format!(
                "the write mode {other:?} as an Iceberg overwrite"
            )));
        }
    };
    // `SparkWriteBuilder`: an `alwaysTrue` overwrite under
    // `overwrite-mode=dynamic` is a dynamic one.
    let filter = filter.filter(|f| !(dynamic_mode && *f == PushdownExpr::Constant(true)));
    let isolation = match option(request, ISOLATION_LEVEL) {
        None => None,
        Some(v) if v.eq_ignore_ascii_case("serializable") => Some(Isolation::Serializable),
        Some(v) if v.eq_ignore_ascii_case("snapshot") => Some(Isolation::Snapshot),
        Some(v) => {
            return Err(unsupported(format!(
                "the Iceberg isolation level {v:?} (write option {ISOLATION_LEVEL}); the levels \
                 are serializable and snapshot"
            )));
        }
    };
    let start = match option(request, VALIDATE_FROM_SNAPSHOT_ID) {
        None => None,
        Some(v) => Some(v.parse::<i64>().map_err(|_| {
            unsupported(format!(
                "the write option {VALIDATE_FROM_SNAPSHOT_ID} {v:?}, which is not a snapshot id"
            ))
        })?),
    };
    let mut validation = Validation {
        max_intervening: target.max_intervening,
        ..Validation::none()
    };
    if let Some(isolation) = isolation {
        validation.isolation = isolation;
        validation.checks = Checks {
            added_data_files: isolation == Isolation::Serializable,
            added_delete_files: true,
            deleted_data_files: true,
            ..Checks::default()
        };
        validation.start = Some(start);
        if start.is_none() {
            validation.hint = Some(format!(
                "the write option {ISOLATION_LEVEL} is set without {VALIDATE_FROM_SNAPSHOT_ID}, so \
                 the whole table history was validated, as Iceberg does; set \
                 {VALIDATE_FROM_SNAPSHOT_ID} to the snapshot the write read"
            ));
        }
    }
    let removals = match &filter {
        Some(f) => {
            let pred = Pred::from_spark(f);
            // Every part must be one the strict side evaluates (review 36b2b
            // B1); Java refuses such a filter when it converts or binds it.
            let fields = schema_fields(target.base.current_schema());
            if let Some(part) = pred.unsupported(&fields) {
                let text = match &request.mode {
                    WriteMode::OverwriteByFilter(e) => e.to_string(),
                    _ => pred.to_string(),
                };
                return Err(unsupported(format!(
                    "an overwrite of the Iceberg table {} by the filter {text}: it holds {part}",
                    request.table_name
                )));
            }
            validation.filter_text = pred.to_string();
            validation.filter = Some(f.clone());
            Removals {
                row_filter: Some(f.clone()),
                ..Removals::default()
            }
        }
        // The partitions are the written files' (`sink.rs` at finalize).
        None => Removals {
            replace_partitions: true,
            ..Removals::default()
        },
    };
    let overwrite = Overwrite {
        removals,
        validation,
        lineage: false,
    };
    IcebergWriteSink::try_rewrite(
        target,
        &request.input_schema,
        &request.table_name,
        request.write_id.clone(),
        overwrite,
    )
}
