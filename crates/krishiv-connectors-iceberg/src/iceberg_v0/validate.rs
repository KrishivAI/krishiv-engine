// SPDX-License-Identifier: Apache-2.0
//! Commit validation (P3-T027, PR-36b; design note
//! `P3-T026-T027-iceberg-scan-and-commit.md` §7.3 steps 2 and 3), as
//! apache-iceberg 1.12.0's `MergingSnapshotProducer` validates: run by the
//! commit loop (`commit.rs`) on every rebase, after the own-key search and
//! before the next attempt is built.
//!
//! - **The walk** ([`walk`], step 2): from the refreshed head Sn of the
//!   written branch back along parent ids to S0, the snapshot the write
//!   started from (`SnapshotUtil.ancestorsBetween`, `SnapshotUtil.java:210-223`).
//!   A walk that does not end at S0's child (a rollback, a replaced branch, an
//!   expired S0) is a conflict with Java's text ("Cannot determine history
//!   between starting snapshot ...", `MergingSnapshotProducer.java:992-996`);
//!   S0 = none (an empty table when the write started) walks the whole chain.
//!   More than `krishiv.iceberg.commit.maxIntervening` snapshots (default
//!   1,000) is a conflict too, never a partial validation.
//! - **History** ([`read_history`], Java's `validationHistory`,
//!   `MergingSnapshotProducer.java:957-1000`): for each intervening snapshot
//!   whose `operation` some enabled check looks at, only the manifests it added
//!   (`added_snapshot_id` = that snapshot) of the content the checks need, read
//!   with the spec's inheritance; within them only `ADDED` and `DELETED`
//!   entries. `EXISTING` entries (a foreign merged manifest) are never changes
//!   (R:S4); each check keeps only entries whose snapshot id is one of the
//!   intervening snapshots of its own operations.
//! - **The checks** ([`check`], step 3), each over the conflict filter (`None`
//!   matches everything) through [`Pruner`] in [`PruneOptions::conservative`]:
//!   every unknown is a match (a missing or undecodable bound, an unknown
//!   transform, a spec or column the filter cannot be projected onto, missing
//!   counts). Here pruning is the correctness mechanism: a wrong "cannot
//!   match" is a lost update (R:S11).
//!
//! | Check | Operations (`MergingSnapshotProducer.java:73-85`) | Java |
//! |---|---|---|
//! | added data files matching the filter (serializable only) | `append`, `overwrite` | `validateAddedDataFiles`, `:363-440` |
//! | added delete files matching the filter, data sequence number above S0's | `overwrite`, `delete` | `validateNoNewDeleteFiles`, `:534-640`; `DeleteFileIndex.java:489-504` |
//! | an added DV for a data file this write writes a DV for | `overwrite`, `delete`, `replace` | `validateAddedDVs`, `:869-955` |
//! | a data file this write removes or references removed | `overwrite`, `replace`, `delete` (`delete` skipped for references when asked) | `validateDataFilesExist`, `:817-860` |
//!
//! A check that fires names the intervening snapshot that held the file, the
//! check and Java's message text; the loop turns it into
//! `KRISHIV_ICEBERG_COMMIT_CONFLICT`. The checks run in the table's order and
//! the first that fires is reported, as Java throws at the first failure.
//!
//! Divergences, all toward refusal or equivalent: the path checks ignore the
//! conflict filter (Java's `validateDataFilesExist` also applies it, `:842`),
//! as a file this write depends on conflicts whatever its rows; the added
//! delete files check counts only `ADDED` entries of the intervening
//! snapshots (Java's `DeleteFileIndex` also counts a live `EXISTING` entry
//! whose data sequence number is above S0's, which a foreign rewrite of
//! delete manifests carries over unchanged). The DV description in the
//! "concurrently added DV" message follows `ContentFileUtil.dvDesc`'s shape,
//! **unverified** (the source was not read).

use std::collections::{BTreeSet, HashSet};

use iceberg::io::FileIO;
use iceberg::spec::{FormatVersion, Operation, TableMetadata};
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_io::PushdownExpr;

use super::scan::{
    parse_manifest, partition_specs, read_capped, read_manifest_list, schema_fields,
};
use crate::plan::entry::{
    ContentFile, DataFileEntry, DeleteKind, EntryStatus, LiveEntry, ManifestContent,
    ManifestFileEntry, inherit_changes,
};
use crate::plan::metrics::WriterMetrics;
use crate::plan::prune::{PruneOptions, Pruner};
use crate::plan::types::{PartitionSpecs, SchemaField};

/// `krishiv.iceberg.commit.maxIntervening`'s default.
pub(crate) const DEFAULT_MAX_INTERVENING: usize = 1_000;

/// Check names, as the conflict error names them.
pub(crate) const CHECK_HISTORY: &str = "history";
/// See [`CHECK_HISTORY`].
pub(crate) const CHECK_ADDED_DATA: &str = "added data files";
/// See [`CHECK_HISTORY`].
pub(crate) const CHECK_ADDED_DELETES: &str = "added delete files";
/// See [`CHECK_HISTORY`].
pub(crate) const CHECK_ADDED_DVS: &str = "added DVs";
/// See [`CHECK_HISTORY`].
pub(crate) const CHECK_REMOVED: &str = "removed data files";

/// `data is only added in "append" and "overwrite" operations`
/// (`VALIDATE_ADDED_FILES_OPERATIONS`).
const ADDED_FILES_OPS: &[Operation] = &[Operation::Append, Operation::Overwrite];
/// `VALIDATE_DATA_FILES_EXIST_OPERATIONS`.
const FILES_EXIST_OPS: &[Operation] =
    &[Operation::Overwrite, Operation::Replace, Operation::Delete];
/// `VALIDATE_DATA_FILES_EXIST_SKIP_DELETE_OPERATIONS`.
const FILES_EXIST_SKIP_DELETE_OPS: &[Operation] = &[Operation::Overwrite, Operation::Replace];
/// `VALIDATE_ADDED_DELETE_FILES_OPERATIONS`.
const ADDED_DELETES_OPS: &[Operation] = &[Operation::Overwrite, Operation::Delete];
/// `VALIDATE_ADDED_DVS_OPERATIONS`.
const ADDED_DVS_OPS: &[Operation] = &[Operation::Overwrite, Operation::Delete, Operation::Replace];

/// The isolation level of a write (`write.{delete,update,merge}.isolation-level`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Isolation {
    /// Every check the operation asks for.
    #[default]
    Serializable,
    /// Concurrently added data files are not a conflict (design note §7.3's
    /// table: no "added data files" check).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "chosen by the operations of PR-36b unit 2")
    )]
    Snapshot,
}

/// Which checks an operation asks for (design note §7.3's statement table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "four independent checks, as Java's four validate calls"
)]
pub(crate) struct Checks {
    /// Data files added matching the filter (serializable only).
    pub(crate) added_data_files: bool,
    /// Delete files added matching the filter.
    pub(crate) added_delete_files: bool,
    /// A DV added for a data file in [`Validation::dvs_for`].
    pub(crate) added_dvs: bool,
    /// A file of [`Validation::removes`] or [`Validation::references`]
    /// removed.
    pub(crate) removed_data_files: bool,
}

/// What one write validates on a rebase.
#[derive(Debug, Clone)]
pub(crate) struct Validation {
    /// The checks asked for.
    pub(crate) checks: Checks,
    /// The isolation level.
    pub(crate) isolation: Isolation,
    /// The conflict filter, in the table schema's names (`None`: every row).
    pub(crate) filter: Option<PushdownExpr>,
    /// The filter as the error message shows it (Java prints its
    /// `Expression`; `true` for none).
    pub(crate) filter_text: String,
    /// Data files this write removes (CoW's rewritten files, `RewriteFiles`).
    pub(crate) removes: BTreeSet<String>,
    /// Data files this write's deletes reference (position deletes, DVs).
    pub(crate) references: BTreeSet<String>,
    /// `validateDataFilesExist(skipDeletes)`: snapshots of operation `delete`
    /// do not count for [`Self::references`] (MoR DELETE).
    pub(crate) references_skip_deletes: bool,
    /// The data files this write writes a DV for.
    pub(crate) dvs_for: BTreeSet<String>,
    /// `krishiv.iceberg.commit.maxIntervening`.
    pub(crate) max_intervening: usize,
}

impl Default for Validation {
    fn default() -> Self {
        Self::none()
    }
}

impl Validation {
    /// Nothing to validate (an append): the loop skips the walk.
    pub(crate) fn none() -> Self {
        Self {
            checks: Checks::default(),
            isolation: Isolation::default(),
            filter: None,
            filter_text: "true".to_owned(),
            removes: BTreeSet::new(),
            references: BTreeSet::new(),
            references_skip_deletes: false,
            dvs_for: BTreeSet::new(),
            max_intervening: DEFAULT_MAX_INTERVENING,
        }
    }

    fn added_data(&self) -> bool {
        self.checks.added_data_files && self.isolation == Isolation::Serializable
    }

    fn added_dvs(&self) -> bool {
        self.checks.added_dvs && !self.dvs_for.is_empty()
    }

    fn removed(&self) -> bool {
        self.checks.removed_data_files && !(self.removes.is_empty() && self.references.is_empty())
    }

    /// Whether any check runs (else the walk is skipped, as for an append).
    pub(crate) fn runs(&self) -> bool {
        self.added_data() || self.checks.added_delete_files || self.added_dvs() || self.removed()
    }

    /// Whether an intervening snapshot of `op` must be read for `content`.
    fn reads(&self, op: &Operation, content: ManifestContent) -> bool {
        let any = |on: bool, ops: &[Operation]| on && ops.contains(op);
        match content {
            ManifestContent::Data => {
                any(self.added_data(), ADDED_FILES_OPS) || any(self.removed(), FILES_EXIST_OPS)
            }
            ManifestContent::Deletes => {
                any(self.checks.added_delete_files, ADDED_DELETES_OPS)
                    || any(self.added_dvs(), ADDED_DVS_OPS)
            }
        }
    }
}

/// A check that fired (or the walk): the loop's conflict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Fired {
    /// The check's name.
    pub(crate) check: &'static str,
    /// The snapshot that conflicts (`None`: none known).
    pub(crate) snapshot_id: Option<i64>,
    /// Java's message text.
    pub(crate) message: String,
}

/// Java prints a null id as `null`.
fn id_text(id: Option<i64>) -> String {
    id.map_or_else(|| "null".to_owned(), |id| id.to_string())
}

/// The intervening snapshots from `head` back to (not including) `s0`, head
/// first, through `parent_of` (`None`: the snapshot is not in the table;
/// `Some(None)`: it has no parent). Java's `ancestorsBetween` then
/// `validationHistory`'s check (module docs).
///
/// # Errors
/// A [`Fired`] [`CHECK_HISTORY`] when the walk does not end at `s0` or holds
/// more than `max` snapshots.
pub(crate) fn walk(
    head: Option<i64>,
    s0: Option<i64>,
    max: usize,
    parent_of: impl Fn(i64) -> Option<Option<i64>>,
) -> Result<Vec<i64>, Fired> {
    let mut out = Vec::new();
    // No current state: nothing was added (`addedDataFiles`' `parent == null`).
    let Some(head) = head else {
        return Ok(out);
    };
    let mut at = head;
    // The parent of the last snapshot walked (`Some(None)`: none).
    let mut last_parent: Option<Option<i64>> = None;
    while Some(at) != s0 {
        let Some(parent) = parent_of(at) else {
            break;
        };
        if out.len() == max {
            return Err(Fired {
                check: CHECK_HISTORY,
                snapshot_id: Some(head),
                message: format!(
                    "more than {max} snapshots were committed to the branch since snapshot \
                     {}; \"krishiv.iceberg.commit.maxIntervening\" bounds the history a \
                     commit validates.",
                    id_text(s0)
                ),
            });
        }
        out.push(at);
        last_parent = Some(parent);
        match parent {
            Some(p) => at = p,
            None => break,
        }
    }
    match last_parent {
        // Sn = S0 (or Sn absent: Java's lookup fails the same way below).
        None if Some(head) == s0 => Ok(out),
        Some(parent) if parent == s0 => Ok(out),
        _ => Err(Fired {
            check: CHECK_HISTORY,
            snapshot_id: Some(head),
            message: format!(
                "Cannot determine history between starting snapshot {} and the last known \
                 ancestor {}",
                id_text(s0),
                id_text(out.last().copied())
            ),
        }),
    }
}

/// The manifests one intervening snapshot added, with their `ADDED` and
/// `DELETED` entries.
#[derive(Debug, Clone)]
pub(crate) struct Change {
    /// The intervening snapshot.
    pub(crate) snapshot_id: i64,
    /// Its operation.
    pub(crate) operation: Operation,
    /// One manifest it added.
    pub(crate) manifest: ManifestFileEntry,
    /// Its `ADDED` and `DELETED` entries, inherited.
    pub(crate) entries: Vec<LiveEntry>,
}

/// What the checks read: the changes of the intervening snapshots and the
/// table they are judged against.
#[derive(Debug, Clone)]
pub(crate) struct History {
    /// Every intervening snapshot and its operation, head first.
    pub(crate) snapshots: Vec<(i64, Operation)>,
    /// The manifests they added, of the content and operations the enabled
    /// checks read, in walk order.
    pub(crate) changes: Vec<Change>,
    /// S0's sequence number (0 when S0 is none or not in the table, as
    /// Java's `startingSequenceNumber`, `MergingSnapshotProducer.java:775-782`).
    pub(crate) start_sequence: i64,
    /// The table's format version is 1: no delete files.
    pub(crate) v1: bool,
    /// The table's default spec id.
    pub(crate) default_spec: i32,
}

impl History {
    /// The intervening snapshots of `ops`.
    fn ids(&self, ops: &[Operation]) -> HashSet<i64> {
        self.snapshots
            .iter()
            .filter(|(_, op)| ops.contains(op))
            .map(|(id, _)| *id)
            .collect()
    }

    /// The entries of `status` of the `content` manifests the snapshots of
    /// `ops` added, whose snapshot id is one of them, with their change.
    fn entries<'h>(
        &'h self,
        ops: &[Operation],
        content: ManifestContent,
        status: EntryStatus,
    ) -> impl Iterator<Item = (&'h Change, &'h LiveEntry)> {
        let ids = self.ids(ops);
        self.changes
            .iter()
            .filter(move |c| ops.contains(&c.operation) && c.manifest.content == content)
            .flat_map(|c| c.entries.iter().map(move |e| (c, e)))
            .filter(move |(_, e)| {
                let d = data_entry(e);
                d.status == status && ids.contains(&d.snapshot_id)
            })
    }
}

fn data_entry(e: &LiveEntry) -> &DataFileEntry {
    match e {
        LiveEntry::Data(d) => d,
        LiveEntry::Delete(d) => &d.entry,
    }
}

/// The table the checks judge files against: Sn's current schema, every
/// partition spec, and the writers' metrics modes.
pub(crate) struct Judge {
    fields: Vec<SchemaField>,
    specs: PartitionSpecs,
    metrics: WriterMetrics,
}

impl Judge {
    /// From the refreshed table `m`.
    pub(crate) fn of(m: &TableMetadata) -> Self {
        let schema = m.current_schema();
        let fields = schema_fields(schema);
        let specs = partition_specs(m, schema);
        let metrics = WriterMetrics::from_properties(
            m.properties().iter().map(|(k, v)| (k.as_str(), v.as_str())),
            &fields,
        );
        Self {
            fields,
            specs,
            metrics,
        }
    }

    /// From its parts (tests).
    #[cfg(test)]
    pub(crate) fn new(fields: Vec<SchemaField>, specs: PartitionSpecs) -> Self {
        Self {
            fields,
            specs,
            metrics: WriterMetrics::default(),
        }
    }

    fn pruner<'j>(&'j self, filter: &'j PushdownExpr) -> Pruner<'j> {
        Pruner::new(
            filter,
            &self.fields,
            &self.specs,
            PruneOptions::conservative(),
        )
        .with_metrics(&self.metrics)
    }
}

/// Java's `Iterators.toString` / a `Set`'s `toString`: `[a, b]`.
fn listed<'a>(paths: impl IntoIterator<Item = &'a str>) -> String {
    format!("[{}]", paths.into_iter().collect::<Vec<_>>().join(", "))
}

/// The four checks over `history` (module docs); the first that fires.
pub(crate) fn check(history: &History, v: &Validation, judge: &Judge) -> Option<Fired> {
    let always = PushdownExpr::Constant(true);
    let pruner = judge.pruner(v.filter.as_ref().unwrap_or(&always));
    let matches = |c: &Change, f: &ContentFile| {
        pruner.manifest_may_match(&c.manifest) && pruner.file_may_match(f)
    };
    if v.added_data() {
        let found: Vec<_> = history
            .entries(ADDED_FILES_OPS, ManifestContent::Data, EntryStatus::Added)
            .filter(|(c, e)| matches(c, &data_entry(e).file))
            .collect();
        let text = |p: String| {
            format!(
                "Found conflicting files that can contain records matching {}: {p}",
                v.filter_text
            )
        };
        if let Some(f) = fired(CHECK_ADDED_DATA, &found, text) {
            return Some(f);
        }
    }
    if v.checks.added_delete_files && !history.v1 {
        let mut seen = HashSet::new();
        let found: Vec<_> = history
            .entries(
                ADDED_DELETES_OPS,
                ManifestContent::Deletes,
                EntryStatus::Added,
            )
            .filter(|(c, e)| {
                let d = data_entry(e);
                // `afterSequenceNumber(startingSequenceNumber)`; a missing one
                // counts.
                d.data_sequence_number
                    .is_none_or(|s| s > history.start_sequence)
                    && matches(c, &d.file)
            })
            // A `LinkedHashSet` of locations.
            .filter(|(_, e)| seen.insert(data_entry(e).file.path.clone()))
            .collect();
        let text = |p: String| {
            format!(
                "Found new conflicting delete files that can apply to records matching {}: {p}",
                v.filter_text
            )
        };
        if let Some(f) = fired(CHECK_ADDED_DELETES, &found, text) {
            return Some(f);
        }
    }
    if v.added_dvs()
        && let Some(f) = added_dvs(history, v, &pruner)
    {
        return Some(f);
    }
    if v.removed() {
        return removed(history, v);
    }
    None
}

/// The conflict of a check that found the files `found` (none: no
/// conflict), naming the snapshot of the first.
fn fired(
    check: &'static str,
    found: &[(&Change, &LiveEntry)],
    message: impl Fn(String) -> String,
) -> Option<Fired> {
    let first = found.first()?.0.snapshot_id;
    let paths = listed(found.iter().map(|(_, e)| data_entry(e).file.path.as_str()));
    Some(Fired {
        check,
        snapshot_id: Some(first),
        message: message(paths),
    })
}

/// `validateAddedDVs`: the first DV an intervening `overwrite`, `delete` or
/// `replace` added for a data file of [`Validation::dvs_for`].
fn added_dvs(history: &History, v: &Validation, pruner: &Pruner<'_>) -> Option<Fired> {
    let ops = ADDED_DVS_OPS;
    // `filterManifestsByPartition`: no manifest pruning without a filter, or
    // when some manifest has another spec than the table's default (a spec
    // change during the validation window).
    let considered: Vec<&Change> = history
        .changes
        .iter()
        .filter(|c| ops.contains(&c.operation) && c.manifest.content == ManifestContent::Deletes)
        .collect();
    let prune_manifests = v.filter.is_some()
        && considered
            .iter()
            .all(|c| c.manifest.spec_id == history.default_spec);
    let ids = history.ids(ops);
    for c in considered {
        // `ManifestFile::hasAddedFiles` (an unknown count may have some).
        if c.manifest.added_files == Some(0)
            || (prune_manifests && !pruner.manifest_may_match(&c.manifest))
        {
            continue;
        }
        for e in &c.entries {
            let LiveEntry::Delete(d) = e else { continue };
            let f = &d.entry.file;
            let Some(referenced) = &f.referenced_data_file else {
                continue;
            };
            if d.kind != DeleteKind::DeletionVector
                || d.entry.status != EntryStatus::Added
                || !ids.contains(&d.entry.snapshot_id)
                || !v.dvs_for.contains(referenced)
                || !pruner.file_may_match(f)
            {
                continue;
            }
            return Some(Fired {
                check: CHECK_ADDED_DVS,
                snapshot_id: Some(c.snapshot_id),
                message: format!(
                    "Found concurrently added DV for {referenced}: DV{{location={}, offset={}, \
                     length={}, referencedDataFile={referenced}}}",
                    f.path,
                    id_text(f.content_offset),
                    id_text(f.content_size_in_bytes),
                ),
            });
        }
    }
    None
}

/// The files of [`Validation::removes`] and [`Validation::references`] an
/// intervening snapshot removed (`DELETED` entries, by path).
fn removed(history: &History, v: &Validation) -> Option<Fired> {
    let gone = |ops: &[Operation], wanted: &BTreeSet<String>| -> Vec<(i64, String)> {
        history
            .entries(ops, ManifestContent::Data, EntryStatus::Deleted)
            .filter(|(_, e)| wanted.contains(&data_entry(e).file.path))
            .map(|(c, e)| (c.snapshot_id, data_entry(e).file.path.clone()))
            .collect()
    };
    let removes = gone(FILES_EXIST_OPS, &v.removes);
    if let Some((id, _)) = removes.first() {
        // `ManifestFilterManager`'s structural check (unverified text).
        return Some(Fired {
            check: CHECK_REMOVED,
            snapshot_id: Some(*id),
            message: format!(
                "Missing required files to delete: {}",
                removes
                    .iter()
                    .map(|(_, p)| p.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        });
    }
    let ops = if v.references_skip_deletes {
        FILES_EXIST_SKIP_DELETE_OPS
    } else {
        FILES_EXIST_OPS
    };
    let references = gone(ops, &v.references);
    let (id, _) = references.first()?;
    Some(Fired {
        check: CHECK_REMOVED,
        snapshot_id: Some(*id),
        message: format!(
            "Cannot commit, missing data files: {}",
            listed(references.iter().map(|(_, p)| p.as_str()))
        ),
    })
}

/// Reads the history of the intervening snapshots `walked` (head first) of
/// the table `m` for the checks of `v`.
///
/// # Errors
/// The read's I/O error; `KRISHIV_CATALOG_LIMIT_EXCEEDED` for a list or
/// manifest over `max_bytes`; `KRISHIV_UNSUPPORTED_FEATURE` for an encrypted
/// manifest; an internal error for one that does not parse; never a partial
/// history.
pub(crate) async fn read_history(
    io: &FileIO,
    m: &TableMetadata,
    walked: &[i64],
    s0: Option<i64>,
    v: &Validation,
    max_bytes: u64,
) -> Result<History, KrishivError> {
    let mut snapshots = Vec::with_capacity(walked.len());
    let mut changes = Vec::new();
    for id in walked {
        let Some(s) = m.snapshot_by_id(*id) else {
            continue;
        };
        let op = s.summary().operation.clone();
        snapshots.push((*id, op.clone()));
        let wanted = |c: ManifestContent| v.reads(&op, c);
        if !wanted(ManifestContent::Data) && !wanted(ManifestContent::Deletes) {
            continue;
        }
        // A snapshot from before an upgrade to v2 has a v1 list when its
        // header does not say (as `plan_files`).
        let fallback = if s.sequence_number() == 0 {
            1
        } else {
            m.format_version() as u8
        };
        let list = read_manifest_list(io, s.manifest_list(), fallback, max_bytes).await?;
        for manifest in list {
            if manifest.added_snapshot_id != *id || !wanted(manifest.content) {
                continue;
            }
            let entries = read_changes(io, &manifest, max_bytes).await?;
            changes.push(Change {
                snapshot_id: *id,
                operation: op.clone(),
                manifest,
                entries,
            });
        }
    }
    let start_sequence = s0
        .and_then(|id| m.snapshot_by_id(id))
        .map_or(0, |s| s.sequence_number());
    Ok(History {
        snapshots,
        changes,
        start_sequence,
        v1: m.format_version() == FormatVersion::V1,
        default_spec: m.default_partition_spec_id(),
    })
}

/// A manifest's `ADDED` and `DELETED` entries, inherited.
async fn read_changes(
    io: &FileIO,
    manifest: &ManifestFileEntry,
    max_bytes: u64,
) -> Result<Vec<LiveEntry>, KrishivError> {
    if manifest.encrypted {
        return Err(KrishivUnsupportedFeature {
            feature: "encrypted Iceberg manifests".to_owned(),
        }
        .into());
    }
    let declared = u64::try_from(manifest.length).ok().filter(|n| *n > 0);
    let bytes = read_capped(io, "manifest", &manifest.path, declared, max_bytes).await?;
    let mut entries = inherit_changes(manifest, parse_manifest(&bytes, manifest.spec_id)?)?;
    entries.retain(|e| data_entry(e).status != EntryStatus::Existing);
    Ok(entries)
}

/// Steps 2 and 3 against the refreshed table `m` whose branch head is
/// `head`, for a write that started at `s0`: `Ok(None)` when nothing
/// conflicts.
///
/// # Errors
/// As [`read_history`].
pub(crate) async fn validate(
    io: &FileIO,
    m: &TableMetadata,
    head: Option<i64>,
    s0: Option<i64>,
    v: &Validation,
    max_bytes: u64,
) -> Result<Option<Fired>, KrishivError> {
    if !v.runs() {
        return Ok(None);
    }
    let parent_of = |id: i64| m.snapshot_by_id(id).map(|s| s.parent_snapshot_id());
    let walked = match walk(head, s0, v.max_intervening, parent_of) {
        Ok(w) => w,
        Err(f) => return Ok(Some(f)),
    };
    if walked.is_empty() {
        return Ok(None);
    }
    let history = read_history(io, m, &walked, s0, v, max_bytes).await?;
    Ok(check(&history, v, &Judge::of(m)))
}
