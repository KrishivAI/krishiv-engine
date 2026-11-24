// SPDX-License-Identifier: Apache-2.0
//! Copy-on-write `DELETE`, `UPDATE` and `MERGE` on an Iceberg table (P3-T027,
//! PR-36b part 2a; design note §8 item 2), as Spark's `SparkWrite`
//! copy-on-write operation (`table.newOverwrite()`, iceberg-spark 1.12.0
//! `SparkWrite.java:416-537`): [`IcebergRowLevel`], the
//! [`RowLevelTable`] an [`IcebergTableProvider`] with a committer, opened at
//! the current version, offers to the session's row-level planner.
//!
//! - **Scans** ([`RowLevelTable::scan`], [`RowLevelTable::scan_files`]): a
//!   [`RowIdScan`] over the provider's scan (deletes applied, so every row id
//!   is a live row), restricted to the touched files for the rewrite. On a
//!   format v3 table both read through the lineage scan, so the table's
//!   columns are followed by `_row_id` and `_last_updated_sequence_number`.
//! - **Rewriter**: [`IcebergWriteSink`] committing an `OverwriteFiles`
//!   (`filter.rs`, `commit.rs`) with `request.write_id`: the touched data
//!   files removed (`overwrittenFiles`), with the DVs the scan assigned them
//!   (`danglingDVs`, `:435-443`) and any other DV of a removed file
//!   (`removeDanglingDeletesFor`); the rows written added. An empty change
//!   still commits (Spark commits an empty copy-on-write snapshot, operation
//!   `overwrite`). On v3 the rows must carry the lineage columns (the rows
//!   moved keep `_row_id` and, when unchanged, `_last_updated_sequence_number`;
//!   changed rows a null sequence number, inserted rows nulls), written under
//!   their reserved ids; a rewrite without them is refused
//!   (`KRISHIV_UNSUPPORTED_FEATURE`), never new ids for untouched rows (§8
//!   item 3).
//! - **Validation** on every rebase (`SparkWrite.java:495-536`, Java's
//!   `BaseOverwriteFiles.validate` with no row filter, `:136-179`): from S0
//!   (the snapshot the provider pinned), with the conflict filter the
//!   statement's read filter ([`RowLevelWrite::read_filter_pushdown`]; none or
//!   untranslatable: every row); `serializable` (the default of
//!   `write.{delete,update,merge}.isolation-level`) checks added data files
//!   and new deletes for the replaced files, `snapshot` the deletes only. A
//!   removed file no longer live is "Missing required files to delete"
//!   (`failMissingDeletePaths`, which Java's `validateNoConflicting*` turn on),
//!   from the filter on every attempt.
//! - **Output**: none (Spark returns no rows for a v2 table).
//!
//! Divergences: the conflict message prints the filter as Krishiv's typed
//! predicate, not Java's `Expression`; an isolation level other than
//! `serializable`/`snapshot` (any case) is `KRISHIV_UNSUPPORTED_FEATURE`
//! (Java's `IsolationLevel.fromName`; its text was not read); an empty
//! change is still validated, where Spark commits with no validation when
//! the optimizer replaced the scan with an empty relation (`scan == null`,
//! `SparkWrite.java:473-492`; notes 36b2 §6): an empty `serializable`
//! `DELETE` fails against a concurrent append where Spark commits. This
//! refuses more, never less; whether Krishiv's planner ever produces Spark's
//! `scan == null` shape is open. The removed DVs are keyed per blob
//! ([`DeleteKey`]): one Puffin file holds the DVs of many data files.

use std::collections::BTreeSet;
use std::sync::Arc;

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::planner::{
    CardinalityRule, RowIdScan, RowLevelCapabilities, RowLevelOperation, RowLevelTable,
    RowLevelWrite,
};
use krishiv_exec::{ScanPushdown, Sink};

use super::filter::{DeleteKey, Removals};
use super::provider::IcebergTableProvider;
use super::sink::{IcebergWriteSink, Overwrite};
use super::validate::{Checks, Isolation, Validation};

/// `write.delete.isolation-level` (Java's `TableProperties.DELETE_ISOLATION_LEVEL`).
pub(crate) const DELETE_ISOLATION_LEVEL: &str = "write.delete.isolation-level";
/// `write.update.isolation-level`.
pub(crate) const UPDATE_ISOLATION_LEVEL: &str = "write.update.isolation-level";
/// `write.merge.isolation-level`.
pub(crate) const MERGE_ISOLATION_LEVEL: &str = "write.merge.isolation-level";

/// The row-level table of an Iceberg provider (module docs).
#[derive(Debug)]
pub(crate) struct IcebergRowLevel {
    provider: IcebergTableProvider,
    op: RowLevelOperation,
    /// A format v3 table: the scans and the rewrite carry row lineage.
    lineage: bool,
}

impl IcebergRowLevel {
    /// The row-level table of `provider` for `op`.
    pub(crate) fn new(provider: IcebergTableProvider, op: RowLevelOperation) -> Self {
        let lineage = provider.snapshot().format_version >= 3;
        Self {
            provider,
            op,
            lineage,
        }
    }

    /// The isolation level `op` runs at (module docs).
    fn isolation(
        &self,
        props: &std::collections::HashMap<String, String>,
    ) -> Result<Isolation, KrishivError> {
        isolation(props, self.op)
    }
}

/// The isolation level `op` runs at on a table with `props`
/// (`write.{delete,update,merge}.isolation-level`, default `serializable`;
/// copy-on-write and merge-on-read alike, `SparkRowLevelOperationBuilder`).
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a level other than `serializable` or
/// `snapshot` (any case).
pub(crate) fn isolation(
    props: &std::collections::HashMap<String, String>,
    op: RowLevelOperation,
) -> Result<Isolation, KrishivError> {
    let key = match op {
        RowLevelOperation::Update => UPDATE_ISOLATION_LEVEL,
        RowLevelOperation::Merge => MERGE_ISOLATION_LEVEL,
        _ => DELETE_ISOLATION_LEVEL,
    };
    match props.get(key) {
        None => Ok(Isolation::Serializable),
        Some(v) if v.eq_ignore_ascii_case("serializable") => Ok(Isolation::Serializable),
        Some(v) if v.eq_ignore_ascii_case("snapshot") => Ok(Isolation::Snapshot),
        Some(v) => Err(KrishivUnsupportedFeature {
            feature: format!(
                "the Iceberg isolation level {v:?} ({key}); the levels are serializable and \
                     snapshot"
            ),
        }
        .into()),
    }
}

impl IcebergRowLevel {
    /// Spark fails a statement whose `write.{delete,update,merge}.mode` is not a known mode
    /// (`Unknown row-level operation mode`), before it reads anything.
    fn check_mode(&self) -> krishiv_exec::Result<()> {
        if let Some(t) = self.provider.write_target() {
            super::row_delta::row_level_mode(t.base.properties(), self.op)?;
        }
        Ok(())
    }

    /// Refuses a v3 table's rewrite whose rows lack the row lineage columns.
    fn require_lineage(&self, request: &RowLevelWrite) -> krishiv_exec::Result<()> {
        let s = &request.input_schema;
        let has = |n: &str| s.fields().iter().any(|f| f.name() == n);
        if self.lineage && !(has(super::lineage::ROW_ID) && has(super::lineage::LAST_UPDATED)) {
            return Err(KrishivUnsupportedFeature {
                feature: format!(
                    "{} of the Iceberg format version 3 table {} whose rewrite does not carry \
                     the row lineage columns {} and {}",
                    self.op.name(),
                    request.table_name,
                    super::lineage::ROW_ID,
                    super::lineage::LAST_UPDATED
                ),
            }
            .into());
        }
        Ok(())
    }
}

impl RowLevelTable for IcebergRowLevel {
    fn capabilities(&self) -> RowLevelCapabilities {
        let mut c = RowLevelCapabilities::copy_on_write(CardinalityRule::SparkMergeCardinality);
        // A mode that cannot be read is the statement's error, raised by `scan` (`check_mode`).
        c.idempotent_commit = true;
        c.merge_on_read = self.provider.write_target().is_some_and(|t| {
            matches!(
                super::row_delta::row_level_mode(t.base.properties(), self.op),
                Ok(super::row_delta::RowLevelMode::MergeOnRead)
            )
        });
        c
    }

    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.check_mode()?;
        let inner = self.provider.row_level_scan(self.lineage, None)?;
        Ok(Arc::new(RowIdScan::new(inner)?))
    }

    fn scan_files(&self, files: &[Vec<u8>]) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        let inner = self.provider.row_level_scan(self.lineage, Some(files))?;
        Ok(Arc::new(RowIdScan::new(inner)?))
    }

    fn rewriter(&self, request: &RowLevelWrite) -> krishiv_exec::Result<Arc<dyn Sink>> {
        // A merge-on-read statement's rows end with the row id (`mor_sink.rs`).
        if super::mor_sink::is_merge_on_read_input(&request.input_schema) {
            // Before anything runs, as copy-on-write does: rows without the lineage columns
            // fail only when a batch has some, so an empty UPDATE would commit.
            if self.op != RowLevelOperation::Delete {
                self.require_lineage(request)?;
            }
            let write = super::row_delta::MergeOnReadWrite::new(
                &self.provider,
                self.op,
                request.read_filter_pushdown(),
                request.write_id.clone(),
                request.idempotency_key(),
            )?;
            return Ok(Arc::new(super::mor_sink::MergeOnReadSink::new(
                write, self.op,
            )));
        }
        let target = Arc::clone(self.provider.write_target().ok_or_else(|| {
            KrishivError::from(krishiv_common::error::InternalError {
                message: "a row-level Iceberg table without a write target".to_owned(),
            })
        })?);
        self.require_lineage(request)?;
        let isolation = self.isolation(target.base.properties())?;
        let touched = self.provider.touched(&request.touched)?;
        let data_files: BTreeSet<String> =
            touched.iter().map(|(e, _)| e.file.path.clone()).collect();
        let delete_files: BTreeSet<DeleteKey> = touched
            .iter()
            .flat_map(|(_, d)| d.deletion_vector.iter())
            .map(|d| DeleteKey {
                path: d.entry.file.path.clone(),
                referenced_data_file: d.entry.file.referenced_data_file.clone(),
                content_offset: d.entry.file.content_offset,
                content_size: d.entry.file.content_size_in_bytes,
            })
            .collect();
        let filter = request.read_filter_pushdown();
        let filter_text = filter
            .as_ref()
            .map_or_else(|| "true".to_owned(), |f| format!("{f:?}"));
        let validation = Validation {
            checks: Checks {
                added_data_files: true,
                new_deletes_for_replaced: true,
                ..Checks::default()
            },
            isolation,
            filter,
            filter_text,
            // `removes` stays empty: it is `RewriteFiles`' set for the
            // removed-data-files check, which CoW's `OverwriteFiles` does not
            // run (notes 36b2 §1 item 3); the removed files' existence is
            // the filter's missing-files check.
            replaced: touched.iter().map(|(e, _)| (**e).clone()).collect(),
            max_intervening: target.max_intervening,
            ..Validation::none()
        };
        let overwrite = Overwrite {
            removals: Removals {
                data_files,
                delete_files,
                ..Removals::default()
            },
            validation,
            lineage: self.lineage,
        };
        Ok(Arc::new(
            IcebergWriteSink::try_rewrite(
                target,
                &request.input_schema,
                &request.table_name,
                request.write_id.clone(),
                overwrite,
            )?
            .with_idempotency_key(request.idempotency_key()),
        ))
    }
}
