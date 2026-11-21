// SPDX-License-Identifier: Apache-2.0
//! Merge-on-read `DELETE`, `UPDATE` and `MERGE` on an Iceberg table (P3-T027,
//! PR-36c part 1; design note §8 item 4), as Spark's `SparkPositionDeltaWrite`
//! commits a `RowDelta` (apache-iceberg 1.12.0; notes 36c):
//! [`MergeOnReadWrite`], the internal entry point the session's merge-on-read
//! rewrite (PR-37d's contract, not wired yet) adapts to: per task, the deleted
//! row ids `(file reference, position)` ([`MorTask::delete`]: values of
//! `_krishiv_row_file` and `_krishiv_row_index`) and, for `UPDATE`/`MERGE`, the
//! new rows ([`MorTask::insert`]); then one commit of every task's output.
//!
//! - **Mode** ([`row_level_mode`]): `write.{delete,update,merge}.mode`,
//!   `copy-on-write` when unset (`TableProperties.java:437-454`), either name in
//!   any case, else Java's "Unknown row-level operation mode: %s"
//!   (`RowLevelOperationMode.fromName`) as `KRISHIV_UNSUPPORTED_FEATURE`. A
//!   format v1 table refuses merge-on-read with Java's "Deletes are supported
//!   in V2 and above" (`MergingSnapshotProducer.java:295-316`).
//! - **The delete kind follows the format version S0 had**
//!   (`SparkWriteConf.deleteFileFormat`): v3 writes DVs, one Puffin file per
//!   task (`PartitioningDVWriter`), its blobs sorted by data file path (Java:
//!   `HashMap` order); v2 one position delete file per data file
//!   (`FanoutPositionOnlyDeleteWriter` at Spark's `FILE` granularity), in data
//!   file path order (`delete_write.rs`). A task with no deletes writes no
//!   delete file.
//! - **Absorbing** (`BaseDVFileWriter.close`, `:134-149`;
//!   `SparkBatchQueryScan.rewritableDeletes`): on v3 every position delete
//!   the scan assigned the data file (its DV, else its v2 position delete
//!   files, partition-scoped ones included) is read and unioned into the new
//!   DV; on v2 only its file-scoped position delete files. The file-scoped
//!   ones (a DV, a position delete file whose `referenced_data_file` or equal
//!   `file_path` bounds name the data file) are removed in the same commit
//!   (`removeDeletes`, through `filter.rs` on every attempt's base); a
//!   partition-scoped file stays (`spec.md:1947`).
//! - **Commit** ([`MergeOnReadWrite::commit`]): DVs of one data file from two
//!   tasks are merged into one Puffin file `merged-dvs-<snapshot id>-1.puffin`
//!   (Java's `mergeDVs`, `MergingSnapshotProducer.java:1209-1235`), the
//!   Puffin files left with no live blob deleted; then the data files and
//!   delete files are added in one snapshot with the absorbed deletes removed,
//!   through `commit.rs` with `request.write_id`. Operation: `delete` when
//!   only deletes are added, `append` when only data, else `overwrite` (an
//!   empty row delta too; `BaseRowDelta.java:50-60`). Summary as Java's
//!   (`added-dvs`, `added-position-delete-files`, `added-files-size` counting
//!   a DV's blob, the removed counterparts).
//! - **Validation** on every rebase from S0 (the provider's pinned snapshot),
//!   with the read filter as the conflict filter (`None`: every row), per
//!   notes 36c §1.5: the data files the deletes reference must not have been
//!   removed by an `overwrite` or `replace` (`DELETE`) or also a `delete`
//!   snapshot (`UPDATE`/`MERGE`); under `serializable` no data file matching
//!   the filter may have been added; for `UPDATE`/`MERGE` no delete file
//!   matching the filter may have been added, and an absorbed delete file no
//!   longer live fails ("Missing required files to delete",
//!   `validateDeletedFiles`), where a `DELETE` skips it; whenever DVs are
//!   added, at either level, a DV another commit added for the same data
//!   file fails ("Found concurrently added DV", `validateAddedDVs`). The
//!   isolation level is `write.{delete,update,merge}.isolation-level`. A
//!   rebase onto a table upgraded to v3 refuses v2 position delete files
//!   (`commit.rs`, Java's "Must use DVs for position deletes in V3").
//! - **v3 row lineage**: `UPDATE`/`MERGE` rows carry `_row_id` and
//!   `_last_updated_sequence_number` after the table's columns (an updated
//!   row its id and a null sequence number, an inserted row nulls), as the
//!   copy-on-write rewrite does; rows without them are refused.
//!
//! Divergences: validation always runs (Spark skips it when the optimizer
//! replaced the scan with an empty relation, `scan == null`); the checks run
//! in `validate.rs`'s fixed order; Puffin blob order and file counters follow
//! data file paths, not Java's hash order; a position delete file's
//! `split_offsets` are not written; the memory charge is an estimate: 8 bytes per position a
//! task collects (and per absorbed one), held until the statement's commit drops the outputs,
//! and each delete file's bytes while it is built and uploaded.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use super::types::arrow_schema;
use arrow_array::RecordBatch;
use iceberg::spec::{DataFile, FormatVersion, TableMetadata};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::planner::RowLevelOperation;
use krishiv_io::PushdownExpr;
use krishiv_memory::{ConsumerHandle, Reservation};
use roaring::RoaringTreemap;

use super::append::{WriteIds, write_new_manifests};
use super::commit::{CommitClock, CommitContext, commit_validated};
use super::data_file::{DataLocations, WrittenDataFile, data_file, delete_file_name};
use super::delete_write::{
    DeleteTarget, created_by, dv_file, dv_puffin, partition_struct, put, write_position_deletes,
};
use super::filter::{DeleteKey, Removals};
use super::provider::IcebergTableProvider;
use super::row_level::isolation;
use super::sink::{WriteTarget, Written, io_clock, random_uuid, snapshot_id, unique_u64};
use super::validate::{Checks, Isolation, Validation};
use super::write::{DataFileWriter, TaskIds};
use crate::plan::deletes::referenced_data_file;
use crate::plan::{DataFileEntry, DeleteFileEntry, DeleteKind, TaskDeletes};

/// `write.delete.mode` (Java's `TableProperties.DELETE_MODE`).
pub(crate) const DELETE_MODE: &str = "write.delete.mode";
/// `write.update.mode`.
pub(crate) const UPDATE_MODE: &str = "write.update.mode";
/// `write.merge.mode`.
pub(crate) const MERGE_MODE: &str = "write.merge.mode";

/// The branch a row delta commits to.
const BRANCH: &str = "main";

/// Bytes charged per deleted position (uncompressed).
const POSITION_BYTES: usize = 8;

/// How a row-level statement changes a table (Java's `RowLevelOperationMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowLevelMode {
    /// Rewrite the touched files (`row_level.rs`).
    CopyOnWrite,
    /// Record deletes (this module).
    MergeOnRead,
}

/// The mode of `op` on a table with `props` (module docs).
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for an unknown mode name, with Java's text.
pub(crate) fn row_level_mode(
    props: &HashMap<String, String>,
    op: RowLevelOperation,
) -> Result<RowLevelMode, KrishivError> {
    let key = match op {
        RowLevelOperation::Update => UPDATE_MODE,
        RowLevelOperation::Merge => MERGE_MODE,
        _ => DELETE_MODE,
    };
    match props.get(key) {
        None => Ok(RowLevelMode::CopyOnWrite),
        Some(v) if v.eq_ignore_ascii_case("copy-on-write") => Ok(RowLevelMode::CopyOnWrite),
        Some(v) if v.eq_ignore_ascii_case("merge-on-read") => Ok(RowLevelMode::MergeOnRead),
        Some(v) => Err(unsupported(format!(
            "the Iceberg table property {key}: Unknown row-level operation mode: {v}"
        ))),
    }
}

/// One merge-on-read statement against the table as a provider pinned it
/// (module docs).
pub(crate) struct MergeOnReadWrite {
    provider: IcebergTableProvider,
    target: Arc<WriteTarget>,
    op: RowLevelOperation,
    isolation: Isolation,
    filter: Option<PushdownExpr>,
    filter_text: String,
    write_id: String,
    /// S0 is format v3: DVs, and row lineage on inserted rows.
    dvs: bool,
    clock: Arc<dyn CommitClock>,
    created_by: String,
}

impl std::fmt::Debug for MergeOnReadWrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MergeOnReadWrite")
            .field("target", &self.target)
            .field("op", &self.op)
            .field("write_id", &self.write_id)
            .finish_non_exhaustive()
    }
}

impl MergeOnReadWrite {
    /// The statement `op` of `provider`'s table, conflict-checked against
    /// `read_filter` (`None`: every row), committed with `write_id` (a fresh
    /// UUID when `None`).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a provider without a committer or
    /// not opened at the current version, a format v1 table, an unknown
    /// isolation level; `INTERNAL_ERROR` without an I/O runtime.
    pub(crate) fn new(
        provider: &IcebergTableProvider,
        op: RowLevelOperation,
        read_filter: Option<PushdownExpr>,
        write_id: Option<String>,
    ) -> Result<Arc<Self>, KrishivError> {
        let target = Arc::clone(provider.write_target().ok_or_else(|| {
            unsupported(
                "a merge-on-read change of an Iceberg table not opened for writing at its \
                 current version"
                    .to_owned(),
            )
        })?);
        let version = target.base.format_version();
        if version == FormatVersion::V1 {
            return Err(unsupported(format!(
                "merge-on-read {} of the format version 1 Iceberg table {}: Deletes are \
                 supported in V2 and above",
                op.name(),
                target.committer.name()
            )));
        }
        let isolation = isolation(target.base.properties(), op)?;
        let clock = io_clock(&target.registered)?;
        let filter_text = read_filter
            .as_ref()
            .map_or_else(|| "true".to_owned(), |f| format!("{f:?}"));
        Ok(Arc::new(Self {
            provider: provider.clone(),
            dvs: version == FormatVersion::V3,
            target,
            op,
            isolation,
            filter: read_filter,
            filter_text,
            write_id: write_id.unwrap_or_else(random_uuid),
            clock,
            created_by: created_by(),
        }))
    }

    /// Fails on a NULL in a required column of the replacement rows of an `UPDATE`.
    pub(crate) fn check_nulls(&self, rows: &RecordBatch) -> Result<(), KrishivError> {
        let table = arrow_schema(self.target.base.current_schema())?;
        super::sink::check_required(&table, rows)
    }

    /// The writer of one task (`partition_id`, Java's partition id in file
    /// names), its data files' and loaded deletes' memory drawn from `mem`.
    #[must_use]
    pub(crate) fn task(self: &Arc<Self>, partition_id: u32, mem: Arc<ConsumerHandle>) -> MorTask {
        MorTask {
            write: Arc::clone(self),
            partition_id,
            attempt_id: unique_u64(),
            charge: mem.reservation(),
            pending: 0,
            mem,
            by_ref: HashMap::new(),
            by_path: HashMap::new(),
            files: Vec::new(),
            data: None,
        }
    }

    /// Commits every task's output as one `RowDelta` (module docs) and
    /// returns the table's new metadata. Before the commit loop starts every
    /// file the tasks wrote is deleted on failure; after, `commit.rs`'s §7.4
    /// rules decide.
    ///
    /// # Errors
    /// `commit_validated`'s (`KRISHIV_ICEBERG_COMMIT_CONFLICT`, `_FAILED`,
    /// `_STATE_UNKNOWN`, ...); the merged DV's write errors; the manifests'.
    pub(crate) async fn commit(
        &self,
        tasks: Vec<TaskOutput>,
        query_id: Option<&str>,
    ) -> Result<TableMetadata, KrishivError> {
        let mut written = self.written_by(&tasks);
        match self.commit_inner(tasks, &mut written, query_id).await {
            Ok(m) => Ok(m),
            Err(e) => {
                if written.armed {
                    written.delete().await;
                }
                Err(e)
            }
        }
    }

    /// Deletes the files of finished tasks' outputs `tasks` that the
    /// statement will not commit (another task failed, or the statement was
    /// cancelled; Java's `PositionDeltaBatchWrite.abort`). Cancellation-safe:
    /// dropped midway, the files not deleted yet are deleted in the
    /// background.
    #[cfg(test)]
    pub(crate) async fn abort(&self, tasks: Vec<TaskOutput>) {
        self.written_by(&tasks).delete().await;
    }

    /// An armed guard over every data and delete file of `tasks`.
    pub(crate) fn written_by(&self, tasks: &[TaskOutput]) -> Written {
        let t = &self.target;
        let mut written = Written::new(t.file_io.clone(), &t.registered);
        for task in tasks {
            written
                .data
                .extend(task.data.iter().map(|f| f.location.clone()));
            written.data.extend(task.files.iter().cloned());
        }
        written
    }

    async fn commit_inner(
        &self,
        tasks: Vec<TaskOutput>,
        written: &mut Written,
        query_id: Option<&str>,
    ) -> Result<TableMetadata, KrishivError> {
        let t = &self.target;
        let base = &t.base;
        let snapshot_id = snapshot_id(base);
        let mut data = Vec::new();
        let mut deletes = Vec::new();
        let mut rewritten: BTreeSet<DeleteKey> = BTreeSet::new();
        let mut task_files = Vec::new();
        for task in tasks {
            data.extend(task.data);
            deletes.extend(task.deletes);
            task_files.extend(task.files);
            for d in &task.rewritten {
                rewritten.insert(delete_key(d));
            }
        }
        let deletes = self
            .merge_duplicate_dvs(deletes, snapshot_id, written, &task_files)
            .await?;
        let references: BTreeSet<String> = deletes.iter().map(|d| d.data.clone()).collect();
        let dvs_for: BTreeSet<String> = deletes
            .iter()
            .filter(|d| d.positions.is_some())
            .map(|d| d.data.clone())
            .collect();
        let schema = base.current_schema();
        let mut entries = data
            .iter()
            .map(|f| data_file(f, schema).map(|d| (f.spec_id, d)))
            .collect::<Result<Vec<_>, _>>()?;
        entries.extend(deletes.into_iter().map(|d| (d.spec_id, d.file)));
        let commit_uuid = random_uuid();
        let mut new = write_new_manifests(
            &t.file_io,
            base,
            entries,
            snapshot_id,
            &commit_uuid,
            &mut written.manifests,
        )
        .await?;
        new.row_delta = true;
        let row_change = self.op != RowLevelOperation::Delete;
        new.removals = Some(Removals {
            delete_files: rewritten,
            ignore_missing_deletes: !row_change,
            ..Removals::default()
        });
        let validation = Validation {
            checks: Checks {
                added_data_files: true,
                added_delete_files: row_change,
                added_dvs: true,
                removed_data_files: true,
                ..Checks::default()
            },
            isolation: self.isolation,
            filter: self.filter.clone(),
            filter_text: self.filter_text.clone(),
            references,
            references_skip_deletes: !row_change,
            dvs_for,
            max_intervening: t.max_intervening,
            ..Validation::none()
        };
        // From here the commit loop owns cleanup (design note §7.4).
        written.armed = false;
        let cx = CommitContext {
            io: &t.file_io,
            clock: self.clock.as_ref(),
            max_metadata_bytes: t.max_metadata_bytes,
        };
        let ids = WriteIds {
            branch: BRANCH,
            write_id: &self.write_id,
            query_id,
        };
        commit_validated(t.committer.as_ref(), cx, base, &mut new, ids, &validation).await
    }

    /// The deletes with the DVs of a data file written by more than one task
    /// merged into one, in one Puffin file (Java's `mergeDVs`); a task's
    /// Puffin file left with no DV is deleted.
    async fn merge_duplicate_dvs(
        &self,
        deletes: Vec<WrittenDelete>,
        snapshot_id: i64,
        written: &mut Written,
        task_files: &[String],
    ) -> Result<Vec<WrittenDelete>, KrishivError> {
        let mut count: HashMap<&str, usize> = HashMap::new();
        for d in deletes.iter().filter(|d| d.positions.is_some()) {
            *count.entry(d.data.as_str()).or_default() += 1;
        }
        if count.values().all(|n| *n < 2) {
            return Ok(deletes);
        }
        let dup: BTreeSet<String> = count
            .into_iter()
            .filter(|(_, n)| *n > 1)
            .map(|(p, _)| p.to_owned())
            .collect();
        let mut kept = Vec::new();
        let mut merged: BTreeMap<String, (i32, iceberg::spec::Struct, RoaringTreemap)> =
            BTreeMap::new();
        for d in deletes {
            match (&d.positions, dup.contains(&d.data)) {
                (Some(p), true) => {
                    let at = merged.entry(d.data.clone()).or_insert_with(|| {
                        (d.spec_id, d.file.partition().clone(), RoaringTreemap::new())
                    });
                    at.2 |= p;
                }
                _ => kept.push(d),
            }
        }
        let t = &self.target;
        let location = DataLocations::for_table(&t.base)?
            .unpartitioned(&format!("merged-dvs-{snapshot_id}-1.puffin"));
        let list: Vec<(&str, &RoaringTreemap)> = merged
            .iter()
            .map(|(p, (_, _, b))| (p.as_str(), b))
            .collect();
        let (bytes, blobs) = dv_puffin(&list, &self.created_by)?;
        written.data.push(location.clone());
        let size = put(
            &t.file_io,
            t.registered.io.scheduler().runtime(),
            &location,
            bytes,
        )
        .await?;
        for ((path, (spec_id, partition, positions)), blob) in merged.into_iter().zip(blobs) {
            let file = dv_file(&location, size, blob, &path, spec_id, partition)?;
            kept.push(WrittenDelete {
                file,
                spec_id,
                data: path,
                positions: Some(positions),
            });
        }
        // A task's Puffin file whose every DV was merged is in no manifest.
        let live: BTreeSet<&str> = kept.iter().map(|d| d.file.file_path()).collect();
        for f in task_files {
            if f.ends_with(".puffin") && !live.contains(f.as_str()) {
                // A failed delete leaves the path on the list, so the commit's cleanup retries it, but only
                // while `written` is armed (before the commit loop); after a commit, or a failure inside
                // the loop, the file stays an orphan.
                if t.file_io.delete(f).await.is_ok() {
                    written.data.retain(|p| p != f);
                }
            }
        }
        Ok(kept)
    }
}

/// A delete file's removal key, as `filter.rs` matches manifest entries.
fn delete_key(d: &DeleteFileEntry) -> DeleteKey {
    let f = &d.entry.file;
    DeleteKey {
        path: f.path.clone(),
        referenced_data_file: f.referenced_data_file.clone(),
        content_offset: f.content_offset,
        content_size: f.content_size_in_bytes,
    }
}

/// One delete file entry a task wrote.
#[derive(Debug, Clone)]
pub(crate) struct WrittenDelete {
    /// Its manifest entry.
    pub(crate) file: DataFile,
    /// Its data file's spec id.
    pub(crate) spec_id: i32,
    /// The data file it deletes from.
    pub(crate) data: String,
    /// A DV's positions (for merging two tasks' DVs of one file).
    pub(crate) positions: Option<RoaringTreemap>,
}

/// What one task wrote.
#[derive(Debug, Default)]
pub(crate) struct TaskOutput {
    /// The charge of the positions the output holds (`deletes`' bitmaps on v3), kept until the
    /// statement's commit or abort drops the output.
    pub(crate) held: Option<Reservation>,
    /// The data files (`UPDATE`/`MERGE` rows).
    pub(crate) data: Vec<WrittenDataFile>,
    /// The delete file entries.
    pub(crate) deletes: Vec<WrittenDelete>,
    /// The delete files absorbed (removed by the commit).
    pub(crate) rewritten: Vec<Arc<DeleteFileEntry>>,
    /// The delete files' locations (a Puffin file once).
    pub(crate) files: Vec<String>,
}

/// A data file a task deletes from.
struct Touched {
    entry: Arc<DataFileEntry>,
    deletes: TaskDeletes,
    positions: RoaringTreemap,
}

/// One task's writer of a [`MergeOnReadWrite`] (module docs).
pub(crate) struct MorTask {
    write: Arc<MergeOnReadWrite>,
    partition_id: u32,
    attempt_id: u64,
    mem: Arc<ConsumerHandle>,
    /// What the positions and the delete files held in memory are charged to.
    charge: Reservation,
    /// Bytes of positions added since the last [`settle`](Self::settle).
    pending: usize,
    /// Each file reference seen, by its index in `files`.
    by_ref: HashMap<Vec<u8>, usize>,
    /// The same slots by the data file's path (two encodings of one file share a slot).
    by_path: HashMap<String, usize>,
    files: Vec<Touched>,
    data: Option<DataFileWriter>,
}

impl std::fmt::Debug for MorTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MorTask")
            .field("partition_id", &self.partition_id)
            .field("files", &self.files.len())
            .finish_non_exhaustive()
    }
}

impl MorTask {
    /// Deletes the row at `pos` of the data file `file` (a value of
    /// `_krishiv_row_file` of the provider's scan).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a reference that does not decode or a file the
    /// snapshot lacks; `KRISHIV_ICEBERG_INVALID_DELETES` for a negative
    /// position; the delete assignment's errors.
    pub(crate) fn delete(&mut self, file: &[u8], pos: i64) -> Result<(), KrishivError> {
        let p = u64::try_from(pos)
            .map_err(|_| crate::plan::deletes::invalid(format!("Invalid position: {pos}")))?;
        let i = match self.by_ref.get(file) {
            Some(i) => *i,
            None => {
                let (entry, deletes) = self
                    .write
                    .provider
                    .touched(&[file.to_vec()])?
                    .pop()
                    .ok_or_else(|| internal("a row file reference of no file".to_owned()))?;
                let i = match self.by_path.get(&entry.file.path) {
                    Some(i) => *i,
                    None => {
                        self.by_path
                            .insert(entry.file.path.clone(), self.files.len());
                        self.files.push(Touched {
                            entry,
                            deletes,
                            positions: RoaringTreemap::new(),
                        });
                        self.files.len() - 1
                    }
                };
                self.by_ref.insert(file.to_vec(), i);
                i
            }
        };
        if let Some(t) = self.files.get_mut(i)
            && t.positions.insert(p)
        {
            self.pending = self.pending.saturating_add(POSITION_BYTES);
        }
        Ok(())
    }

    /// Charges the positions added by [`delete`](Self::delete) since the last call to the task's
    /// consumer (an uncompressed 8 bytes each, an upper bound of the bitmaps').
    ///
    /// # Errors
    /// The memory error class when the pool cannot grant it.
    pub(crate) async fn settle(&mut self) -> Result<(), KrishivError> {
        let n = std::mem::take(&mut self.pending);
        self.charge.grow(n).await?;
        Ok(())
    }

    /// Writes the new rows `batch` (`UPDATE`/`MERGE`): the table's columns,
    /// then on a v3 table `_row_id` and `_last_updated_sequence_number`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a `DELETE`; `KRISHIV_UNSUPPORTED_FEATURE` for rows
    /// of a v3 table without the lineage columns; the data writer's errors.
    pub(crate) async fn insert(&mut self, batch: &RecordBatch) -> Result<(), KrishivError> {
        let w = &self.write;
        if w.op == RowLevelOperation::Delete {
            return Err(internal("a DELETE writes no rows".to_owned()));
        }
        if w.dvs {
            let names = batch.schema();
            let has = |n: &str| names.fields().iter().any(|f| f.name() == n);
            if !(has(super::lineage::ROW_ID) && has(super::lineage::LAST_UPDATED)) {
                return Err(unsupported(format!(
                    "merge-on-read {} of the Iceberg format version 3 table {} whose rows do \
                     not carry the row lineage columns {} and {}",
                    w.op.name(),
                    w.target.committer.name(),
                    super::lineage::ROW_ID,
                    super::lineage::LAST_UPDATED
                )));
            }
        }
        if self.data.is_none() {
            let ids = TaskIds {
                partition_id: self.partition_id,
                attempt_id: self.attempt_id,
                write_id: &w.write_id,
            };
            let mut writer = DataFileWriter::with_registered(
                &w.target.base,
                w.target.registered.clone(),
                ids,
                w.target.max_open_files,
                self.mem.reservation(),
            )?;
            if w.dvs {
                writer = writer.with_lineage()?;
            }
            self.data = Some(writer);
        }
        match self.data.as_mut() {
            Some(d) => d.write(batch).await,
            None => Err(internal("no data writer".to_owned())),
        }
    }

    /// Ends the task: its data files finished and its delete files written,
    /// absorbing the deletes they replace (module docs). On failure nothing
    /// it wrote is left; nor when the future is dropped midway (the data
    /// writer's guard, then one over the finished data files and each delete
    /// file from before its upload starts, deleting in the background). A
    /// finished output the statement does not commit goes to
    /// [`MergeOnReadWrite::abort`].
    ///
    /// # Errors
    /// The data writer's errors; the absorbed deletes' read errors; the
    /// delete files' write errors.
    pub(crate) async fn finish(mut self) -> Result<TaskOutput, KrishivError> {
        let t = Arc::clone(&self.write.target);
        let mut guard = Written::new(t.file_io.clone(), &t.registered);
        let mut out = TaskOutput::default();
        if let Some(writer) = self.data.take() {
            out.data = writer.finish().await?;
        }
        guard
            .data
            .extend(out.data.iter().map(|f| f.location.clone()));
        if let Err(e) = self.write_deletes(&mut out, &mut guard).await {
            guard.delete().await;
            return Err(e);
        }
        guard.armed = false;
        let held = self.charge.size();
        out.held = Some(self.charge.split(held));
        Ok(out)
    }

    /// Drops the task's data files (a statement that fails before its
    /// commit); a task never finished leaves nothing either.
    pub(crate) async fn abort(mut self) {
        if let Some(writer) = self.data.take() {
            writer.abort().await;
        }
    }

    /// Writes the delete files into `out`, each location pushed onto `guard`
    /// before its upload starts.
    async fn write_deletes(
        &mut self,
        out: &mut TaskOutput,
        guard: &mut Written,
    ) -> Result<(), KrishivError> {
        let w = Arc::clone(&self.write);
        let t = &w.target;
        let (io, locate) = w.provider.delete_io();
        self.files
            .sort_by(|a, b| a.entry.file.path.cmp(&b.entry.file.path));
        for touched in &mut self.files {
            // What the new delete absorbs: on v3 every position delete of
            // the file, on v2 its file-scoped position delete files.
            let previous: Vec<Arc<DeleteFileEntry>> = touched
                .deletes
                .deletion_vector
                .iter()
                .chain(&touched.deletes.position_deletes)
                .filter(|d| w.dvs || file_scoped(d))
                .cloned()
                .collect();
            if previous.is_empty() || touched.positions.is_empty() {
                continue;
            }
            let path = &touched.entry.file.path;
            let old =
                super::deletes::positions_of(&io, &*locate, &self.mem, path, &previous).await?;
            let before = touched.positions.len();
            touched.positions |= old;
            let added = touched.positions.len().saturating_sub(before);
            self.charge
                .grow(
                    usize::try_from(added)
                        .unwrap_or(usize::MAX)
                        .saturating_mul(POSITION_BYTES),
                )
                .await?;
            out.rewritten
                .extend(previous.into_iter().filter(|d| file_scoped(d)));
        }
        let touched: Vec<&Touched> = self
            .files
            .iter()
            .filter(|t| !t.positions.is_empty())
            .collect();
        if touched.is_empty() {
            return Ok(());
        }
        let base = &t.base;
        let locations = DataLocations::for_table(base)?;
        if w.dvs {
            let name =
                delete_file_name(self.partition_id, self.attempt_id, &w.write_id, 1, "puffin");
            let location = locations.unpartitioned(&name);
            let list: Vec<(&str, &RoaringTreemap)> = touched
                .iter()
                .map(|t| (t.entry.file.path.as_str(), &t.positions))
                .collect();
            let (bytes, blobs) = dv_puffin(&list, &w.created_by)?;
            let file_bytes = bytes.len();
            self.charge.grow(file_bytes).await?;
            out.files.push(location.clone());
            guard.data.push(location.clone());
            let size = put(
                &t.file_io,
                t.registered.io.scheduler().runtime(),
                &location,
                bytes,
            )
            .await?;
            // The bytes are gone; the positions stay charged while the output holds them.
            self.charge.shrink(file_bytes);
            for (d, blob) in touched.iter().zip(blobs) {
                let f = &d.entry.file;
                let partition = partition_struct(base, f.spec_id, &f.partition)?;
                out.deletes.push(WrittenDelete {
                    file: dv_file(&location, size, blob, &f.path, f.spec_id, partition)?,
                    spec_id: f.spec_id,
                    data: f.path.clone(),
                    positions: Some(d.positions.clone()),
                });
            }
            return Ok(());
        }
        let target = DeleteTarget {
            io: &io,
            file_io: &t.file_io,
            metadata: base,
        };
        let schema = base.current_schema();
        for (n, d) in touched.iter().enumerate() {
            let f = &d.entry.file;
            let spec = base.partition_spec_by_id(f.spec_id).ok_or_else(|| {
                internal(format!("a data file of the unknown spec {}", f.spec_id))
            })?;
            let partition = partition_struct(base, f.spec_id, &f.partition)?;
            let n = u32::try_from(n + 1)
                .map_err(|_| internal("too many delete files in one task".to_owned()))?;
            let name = delete_file_name(
                self.partition_id,
                self.attempt_id,
                &w.write_id,
                n,
                "parquet",
            );
            let location = locations.location(spec, schema, &partition, &name)?;
            // The file is built whole in memory: its positions, sorted, are the bound.
            let built = usize::try_from(d.positions.len())
                .unwrap_or(usize::MAX)
                .saturating_mul(POSITION_BYTES);
            self.charge.grow(built).await?;
            out.files.push(location.clone());
            guard.data.push(location.clone());
            let file = write_position_deletes(
                &target,
                &location,
                &f.path,
                f.spec_id,
                partition,
                &d.positions,
            )
            .await?;
            self.charge.shrink(built);
            out.deletes.push(WrittenDelete {
                file,
                spec_id: f.spec_id,
                data: f.path.clone(),
                positions: None,
            });
        }
        Ok(())
    }
}

/// Java's `ContentFileUtil.isFileScoped`: a DV, or a position delete file
/// naming one data file (explicitly or by equal `file_path` bounds).
fn file_scoped(d: &DeleteFileEntry) -> bool {
    d.kind == DeleteKind::DeletionVector
        || (d.kind == DeleteKind::PositionDelete && referenced_data_file(d).is_some())
}

fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}
