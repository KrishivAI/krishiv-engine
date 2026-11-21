// SPDX-License-Identifier: Apache-2.0
//! An append's metadata (P3-T027, PR-36a; design note §6.2, §6.3, §7.2): the
//! manifests of the files a write added, written once per write
//! ([`write_new_manifests`]), and per commit attempt the manifest list, the
//! snapshot, and the catalog request's updates and requirements
//! ([`prepare_attempt`]). Nothing here sends the request: the commit loop
//! (§7) does, and on a conflict calls [`prepare_attempt`] again on the
//! refreshed metadata with the same manifests and snapshot id.
//!
//! As Java's `SnapshotProducer`/`MergeAppend` (apache-iceberg 1.12.0):
//! - the manifests of each partition spec (in spec id order) grouped and
//!   rolled at `commit.manifest.target-size-bytes` as Java's
//!   `writeDataManifests` (`rolling.rs`), at
//!   `<metadata dir>/<commit uuid>-m<n>.avro`, `n` from 0 in list order;
//!   entries `ADDED` with the write's snapshot id and null (inherited)
//!   sequence numbers, so the manifests are valid on any base;
//! - the list at `<metadata dir>/snap-<snapshot id>-<attempt>-<commit uuid>.avro`,
//!   the new manifests first, then the base snapshot's data manifests, then
//!   its delete manifests, merged on the first attempt (`merge.rs`, design
//!   note §6.2); the new manifests reset to an unassigned sequence number and
//!   no `first_row_id` before each attempt (§7.3 step 1), so the list writer
//!   assigns them;
//! - the sequence number is the table's `last-sequence-number` + 1 (v1: 0);
//!   v3 takes `first-row-id` from the table's `next-row-id` and `added-rows`
//!   from what the list writer assigned;
//! - the summary: `operation` `append`, iceberg-rust's added-* counts and
//!   partition summaries (`write.summary.partition-limit`), the `total-*` values
//!   from the parent's (each left out when the parent lacks it; from 0 without
//!   a parent), Java's `manifests-created`/`manifests-kept`/`manifests-replaced`
//!   (`buildManifestCountSummary`), `engine-name`/`engine-version`,
//!   `krishiv.write-id` and `krishiv.query-id`;
//! - requirements `assert-table-uuid` and `assert-ref-snapshot-id` (the base,
//!   or null for a ref that does not exist yet); updates `add-snapshot` and
//!   `set-snapshot-ref`, the branch keeping its retention, then (PR-54b)
//!   `set-statistics` for the attempt's own Puffin statistics file
//!   (`statistics_file.rs`), when the write has sketches and they describe
//!   the whole snapshot on this attempt's parent.
//!
//! Where a manifest rolls follows Java's rule over iceberg-rust's
//! (uncompressed) encoded length, not Java's flushed, compressed length, and
//! the group count fixes Java's host-dependent pool at 4 (`rolling.rs`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use iceberg::io::FileIO;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, ManifestContentType, ManifestFile,
    ManifestListWriter, Operation, Snapshot, SnapshotReference, SnapshotRetention,
    SnapshotSummaryCollector, StatisticsFile, Summary, TableMetadata, UNASSIGNED_SEQUENCE_NUMBER,
};
use iceberg::{TableRequirement, TableUpdate};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};

use super::data_file::check_partition;
use super::filter::{
    FilterContext, FilterError, Removals, add_delete_files, filter_manifests, merge_summary,
    overwrite_summary,
};
use super::merge::{MergeConfig, MergeContext, arrange};
use super::rolling::{ManifestNames, SpecManifests};
use super::scan::{read_manifest_files, snapshot_refs};
use super::statistics_file::{AttemptSnapshot, WriteSketches, attempt_statistics, skipped};
use super::storage::from_iceberg;

/// `write.metadata.path` (Java's `TableProperties.WRITE_METADATA_LOCATION`).
pub(crate) const WRITE_METADATA_PATH: &str = "write.metadata.path";

/// `write.summary.partition-limit`.
const PARTITION_SUMMARY_LIMIT: &str = "write.summary.partition-limit";

/// Java's `SnapshotSummary.CHANGED_PARTITION_COUNT_PROP`.
const CHANGED_PARTITION_COUNT: &str = "changed-partition-count";

/// Java's `SnapshotSummary.PARTITION_SUMMARY_PROP`.
const PARTITION_SUMMARIES_INCLUDED: &str = "partition-summaries-included";

/// Java's `SnapshotSummary.CREATED_MANIFESTS_COUNT`.
pub(crate) const MANIFESTS_CREATED: &str = "manifests-created";

/// Java's `SnapshotSummary.KEPT_MANIFESTS_COUNT`.
pub(crate) const MANIFESTS_KEPT: &str = "manifests-kept";

/// Java's `SnapshotSummary.REPLACED_MANIFESTS_COUNT`.
pub(crate) const MANIFESTS_REPLACED: &str = "manifests-replaced";

/// The summary key of a write's idempotency key (ADR 0014 item 5).
pub(crate) const WRITE_ID_PROP: &str = "krishiv.write-id";

/// The summary key of the query that wrote a snapshot.
pub(crate) const QUERY_ID_PROP: &str = "krishiv.query-id";

/// The `total-*` summary values and the added and removed values each one
/// moves by (Java's `SnapshotProducer.summary`).
const TOTALS: [(&str, &str, &str); 6] = [
    ("total-records", "added-records", "deleted-records"),
    ("total-files-size", "added-files-size", "removed-files-size"),
    ("total-data-files", "added-data-files", "deleted-data-files"),
    (
        "total-delete-files",
        "added-delete-files",
        "removed-delete-files",
    ),
    (
        "total-position-deletes",
        "added-position-deletes",
        "removed-position-deletes",
    ),
    (
        "total-equality-deletes",
        "added-equality-deletes",
        "removed-equality-deletes",
    ),
];

/// The table's metadata directory (`write.metadata.path`, else
/// `<location>/metadata`), without a trailing slash.
pub(crate) fn metadata_dir(metadata: &TableMetadata) -> String {
    let dir = match metadata.properties().get(WRITE_METADATA_PATH) {
        Some(p) => p.clone(),
        None => format!("{}/metadata", metadata.location().trim_end_matches('/')),
    };
    dir.trim_end_matches('/').to_owned()
}

/// `write.summary.partition-limit` (default 0: no partition summaries).
fn partition_summary_limit(metadata: &TableMetadata) -> u64 {
    metadata
        .properties()
        .get(PARTITION_SUMMARY_LIMIT)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

/// What one write adds, once for all its attempts. Not `Clone`: it owns the
/// attempt counter, so no two attempts share a manifest-list path.
#[derive(Debug)]
pub(crate) struct NewManifests {
    /// The snapshot id chosen for the write (§6.2: one per write).
    pub(crate) snapshot_id: i64,
    /// Names the write's metadata files.
    pub(crate) commit_uuid: String,
    /// The manifests, as the manifest writer returned them.
    pub(crate) manifests: Vec<ManifestFile>,
    /// The summary of the added files (added-* and partition summaries).
    pub(crate) added: HashMap<String, String>,
    /// The locations of the data files the manifests list, deleted when the
    /// write fails with every outcome known (design note §7.4).
    pub(crate) data_files: Vec<String>,
    /// The locations of the delete files the manifests list (a row delta's,
    /// PR-36c; a Puffin file once), cleaned up as `data_files`.
    pub(crate) delete_files: Vec<String>,
    /// A position delete file (not a DV) among them: a rebase onto a v3
    /// table refuses it (Java's "Must use DVs for position deletes in V3").
    pub(crate) position_delete_file: Option<String>,
    /// A `RowDelta` (merge-on-read, `row_delta.rs`): its operation is
    /// Java's `BaseRowDelta.operation`.
    pub(crate) row_delta: bool,
    /// The merged manifests written by any attempt (only the first merges,
    /// `merge.rs`) and the manifests an overwrite's filter rewrote
    /// (`filter.rs`, every attempt): kept when the committed list names
    /// them, else deleted with the attempt's other files (design note §7.4).
    pub(crate) merged: Vec<String>,
    /// The write's column sketches (PR-54b, `statistics_file.rs`): each attempt
    /// writes a statistics file of its snapshot from them and adds
    /// `set-statistics`; `None` when the write has none or dropped them.
    /// Appends only: rule 2 unions with the parent's sketch, so a write that
    /// removes rows (an overwrite or delete through `commit_validated`) must
    /// leave it `None` until it decides what it carries forward (review S3).
    pub(crate) statistics: Option<WriteSketches>,
    /// Every statistics file an attempt started writing: kept when the
    /// committed metadata names it, else deleted with the attempt's other
    /// files (design note §7.4).
    pub(crate) statistics_files: Vec<String>,
    /// The partition paths of the added files (Java's `partitionToPath`,
    /// `""` unpartitioned), for an overwrite's `changed-partition-count`.
    pub(crate) changed: HashSet<String>,
    /// The files an overwrite removes (`filter.rs`), re-applied to every
    /// attempt's base; `None`: an append.
    pub(crate) removals: Option<Removals>,
    /// Attempts prepared so far (Java's `SnapshotProducer.attempt`).
    attempts: u32,
}

/// Writes the manifests of `files` (data files, each with the id of the spec of
/// `metadata` it was written under: iceberg-rust's `DataFile` does not expose it)
/// under the snapshot id `snapshot_id` (module docs), each manifest's path
/// pushed onto `written` before its file is created, so a caller can delete
/// what a failed or dropped write left.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` when `commit.manifest.target-size-bytes`,
/// `commit.manifest.min-count-to-merge` or `commit.manifest-merge.enabled`
/// does not parse (`MergeConfig`, read here before any manifest, as Java's
/// `SnapshotProducer`/`MergeAppend` constructors parse them); an internal error for a file of a spec the table lacks or whose partition
/// tuple does not fit its spec ([`check_partition`], checked before any
/// iceberg-rust call, which would panic); iceberg-rust's write failures, with
/// their Krishiv class when the storage raised one.
pub(crate) async fn write_new_manifests(
    io: &FileIO,
    metadata: &TableMetadata,
    files: Vec<(i32, DataFile)>,
    snapshot_id: i64,
    commit_uuid: &str,
    written: &mut Vec<String>,
) -> Result<NewManifests, KrishivError> {
    let schema = Arc::clone(metadata.current_schema());
    let dir = metadata_dir(metadata);
    // A row delta's delete files (PR-36c) go into delete manifests after
    // the data manifests.
    let (files, deletes): (Vec<_>, Vec<_>) = files
        .into_iter()
        .partition(|(_, f)| f.content_type() == DataContentType::Data);
    let data_files = files
        .iter()
        .map(|(_, f)| f.file_path().to_owned())
        .collect();
    let mut delete_files: Vec<String> = Vec::new();
    let mut position_delete_file = None;
    for (_, f) in &deletes {
        if !delete_files.iter().any(|p| p == f.file_path()) {
            delete_files.push(f.file_path().to_owned());
        }
        if f.file_format() != DataFileFormat::Puffin && position_delete_file.is_none() {
            position_delete_file = Some(f.file_path().to_owned());
        }
    }
    let group = |files: Vec<(i32, DataFile)>| {
        let mut by_spec: BTreeMap<i32, Vec<DataFile>> = BTreeMap::new();
        for (spec_id, f) in files {
            by_spec.entry(spec_id).or_default().push(f);
        }
        by_spec
    };
    let by_spec = group(files);
    let deletes_by_spec = group(deletes);
    let mut summary = SnapshotSummaryCollector::default();
    summary.set_partition_summary_limit(partition_summary_limit(metadata));
    // Java's `changed-partition-count` counts every partition path, the
    // unpartitioned "" included; iceberg-rust's collector skips unpartitioned files.
    let mut changed: HashSet<String> = HashSet::new();
    // Every file is checked before any manifest is written, so a bad file
    // leaves no manifest behind (review 36a N9).
    let mut delete_paths: Vec<(String, &DataFile)> = Vec::new();
    for (spec_id, files) in by_spec.iter().chain(&deletes_by_spec) {
        let spec = metadata.partition_spec_by_id(*spec_id).ok_or_else(|| {
            internal(format!(
                "a data file of partition spec {spec_id}, which the table lacks"
            ))
        })?;
        for f in files {
            check_partition(spec, &schema, f.partition())?;
            let path = spec.partition_to_path(f.partition(), Arc::clone(&schema));
            if f.content_type() != DataContentType::Data {
                delete_paths.push((path.clone(), f));
            }
            changed.insert(path);
        }
    }
    let target_size_bytes = MergeConfig::from_properties(metadata.properties())?.target_size_bytes;
    let mut names = ManifestNames {
        prefix: format!("{dir}/{commit_uuid}-m"),
        next: 0,
        written,
    };
    let mut manifests = Vec::new();
    let mut added_deletes = HashMap::new();
    add_delete_files(&mut added_deletes, &delete_paths);
    for (content, groups) in [
        (ManifestContentType::Data, by_spec),
        (ManifestContentType::Deletes, deletes_by_spec),
    ] {
        for (spec_id, files) in groups {
            let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
                internal(format!(
                    "a data file of partition spec {spec_id}, which the table lacks"
                ))
            })?;
            if content == ManifestContentType::Data {
                for f in &files {
                    summary.add_file(f, Arc::clone(&schema), Arc::clone(spec));
                }
            }
            let writer = SpecManifests {
                io,
                version: metadata.format_version(),
                schema: &schema,
                spec,
                snapshot_id,
                target_size_bytes,
                content,
            };
            manifests.extend(writer.write(&mut names, files).await?);
        }
    }
    let mut added = summary.build();
    // The delete files' values (Java's `SnapshotSummary.addedFile`: a DV
    // sized by its blob), which iceberg-rust's collector would count wrong.
    merge_summary(&mut added, &added_deletes);
    // Java's `SnapshotSummary.Builder.build`: the count always, and
    // `partition-summaries-included` when partition summaries were written.
    added.insert(
        CHANGED_PARTITION_COUNT.to_owned(),
        changed.len().to_string(),
    );
    let limit = partition_summary_limit(metadata);
    if !changed.is_empty() && u64::try_from(changed.len()).unwrap_or(u64::MAX) <= limit {
        added.insert(PARTITION_SUMMARIES_INCLUDED.to_owned(), "true".to_owned());
    }
    Ok(NewManifests {
        snapshot_id,
        commit_uuid: commit_uuid.to_owned(),
        manifests,
        added,
        data_files,
        delete_files,
        position_delete_file,
        row_delta: false,
        merged: Vec::new(),
        statistics: None,
        statistics_files: Vec::new(),
        changed,
        removals: None,
        attempts: 0,
    })
}

/// The branch a write commits to and who wrote (for the summary).
#[derive(Debug, Clone, Copy)]
pub(crate) struct WriteIds<'a> {
    /// The branch.
    pub(crate) branch: &'a str,
    /// The write's idempotency key (`krishiv.write-id`). Not unique per
    /// snapshot: a re-executed or replayed write keeps it, so the commit
    /// loop's own-key search takes a snapshot of this write's id first and a
    /// key match under another id only after (`commit.rs`, R:S1); a key
    /// found under another id means "this write is in the table", and this
    /// execution's files are then orphans.
    ///
    /// The key is matched over every snapshot of the table, not only the
    /// branch's ancestry, so a key is one write to one branch: a caller must
    /// never reuse it for a write to another branch (re-review nit 2; only
    /// `main` is written today, and branch writes, PR-36d, must either keep
    /// this or restrict the match to the target branch's ancestry).
    pub(crate) write_id: &'a str,
    /// The query that wrote, when known.
    pub(crate) query_id: Option<&'a str>,
}

/// One commit attempt: the request's body and the files written for it.
#[derive(Debug, Clone)]
pub(crate) struct Attempt {
    /// The snapshot the attempt adds.
    pub(crate) snapshot: Snapshot,
    /// `assert-table-uuid`, `assert-ref-snapshot-id`.
    pub(crate) requirements: Vec<TableRequirement>,
    /// `add-snapshot`, `set-snapshot-ref`, then `set-statistics` when the
    /// attempt has a statistics file (PR-54b).
    pub(crate) updates: Vec<TableUpdate>,
}

/// The next attempt (1, 2, ..., counted in `new`, so a list path is never
/// reused: a request of an earlier attempt may still land, §7.2.1) of the
/// append `new` to the branch `ids.branch` of `metadata`, at `timestamp_ms`
/// (module docs). A branch that does not exist yet starts from the current
/// snapshot, as Java's `SnapshotUtil.latestSnapshot`, with its `total-*` from
/// 0 (Java's summary reads the branch's own previous snapshot, which it lacks).
/// A resend of the same attempt reuses the returned [`Attempt`].
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` when `branch` names a tag (Java's text); an
/// internal error when the metadata's refs cannot be read or the branch points
/// at a snapshot the metadata lacks; the base manifest list's read errors
/// (`KRISHIV_CATALOG_LIMIT_EXCEEDED` past `max_metadata_bytes`); iceberg-rust's
/// write failures.
#[cfg(test)]
pub(crate) async fn prepare_attempt(
    io: &FileIO,
    metadata: &TableMetadata,
    new: &mut NewManifests,
    ids: WriteIds<'_>,
    timestamp_ms: i64,
    max_metadata_bytes: u64,
) -> Result<Attempt, KrishivError> {
    prepare(io, metadata, new, ids, timestamp_ms, max_metadata_bytes)
        .await
        .map_err(|e| match e {
            FilterError::Error(e) => e,
            FilterError::Missing(m) => internal(m),
            FilterError::Partial { filter, file } => {
                internal(FilterError::partial_text(&filter, &file))
            }
        })
}

/// [`prepare_attempt`] for an append or an overwrite (`new.removals`, an
/// `OverwriteFiles` as Java's `BaseOverwriteFiles`): the removals are
/// applied to this attempt's base (`filter.rs`) before the list is arranged,
/// the operation is `delete` when files are only removed, `append` when only
/// added, else `overwrite` (`BaseOverwriteFiles.java:49-60`; an empty
/// overwrite is `overwrite`), and the summary carries the removed files.
///
/// # Errors
/// [`FilterError::Missing`] when a file to remove is not live in the base
/// (the loop's conflict); else as [`prepare_attempt`].
pub(crate) async fn prepare(
    io: &FileIO,
    metadata: &TableMetadata,
    new: &mut NewManifests,
    ids: WriteIds<'_>,
    timestamp_ms: i64,
    max_metadata_bytes: u64,
) -> Result<Attempt, FilterError> {
    let branch = ids.branch;
    let refs = snapshot_refs(metadata)?;
    let mut base_ref = refs.get(branch).cloned();
    if base_ref.is_none()
        && branch == "main"
        && let Some(current) = metadata.current_snapshot_id()
    {
        // A v1 table serialises no refs; its `main` is the current snapshot.
        base_ref = Some(SnapshotReference::new(current, main_retention()));
    }
    if let Some(r) = &base_ref
        && !r.is_branch()
    {
        return Err(KrishivError::from(KrishivUnsupportedFeature {
            feature: format!(
                "{branch} is a tag, not a branch. Tags cannot be targets for producing snapshots"
            ),
        })
        .into());
    }
    // A dangling ref is refused: read as "no base", the list would drop every
    // file the branch holds.
    let parent = match &base_ref {
        Some(r) => Some(metadata.snapshot_by_id(r.snapshot_id).ok_or_else(|| {
            internal(format!(
                "branch {branch} points at snapshot {}, which the table metadata lacks",
                r.snapshot_id
            ))
        })?),
        None => metadata.current_snapshot(),
    };
    // The totals continue the branch's own previous snapshot only.
    let previous = base_ref
        .is_some()
        .then(|| parent.map(|p| &p.summary().additional_properties))
        .flatten();
    let version = metadata.format_version();
    let base_manifests = match parent {
        Some(p) => {
            read_manifest_files(io, p.manifest_list(), version as u8, max_metadata_bytes).await?
        }
        None => Vec::new(),
    };

    let snapshot_id = new.snapshot_id;
    let parent_id = parent.map(|p| p.snapshot_id());
    let sequence_number = match version {
        FormatVersion::V1 => 0,
        FormatVersion::V2 | FormatVersion::V3 => metadata.last_sequence_number() + 1,
    };
    let first_row_id = metadata.next_row_id();
    // Merging runs on the first attempt only (`merge.rs`); its properties are
    // read then and never again.
    let merge = if new.attempts == 0 {
        Some(MergeConfig::from_properties(metadata.properties())?)
    } else {
        None
    };
    let fresh: Vec<ManifestFile> = new
        .manifests
        .iter()
        .cloned()
        .map(|mut m| {
            m.sequence_number = UNASSIGNED_SEQUENCE_NUMBER;
            m.min_sequence_number = UNASSIGNED_SEQUENCE_NUMBER;
            m.first_row_id = None;
            m
        })
        .collect();
    let dir = metadata_dir(metadata);
    let prefix = format!("{dir}/{}-m", new.commit_uuid);
    let mut filtered = None;
    let base_manifests = match &new.removals {
        None => base_manifests,
        Some(removals) => {
            let next = new.manifests.len() + new.merged.len();
            let f = filter_manifests(
                &mut FilterContext {
                    io,
                    metadata,
                    snapshot_id,
                    prefix: prefix.clone(),
                    next,
                    written: &mut new.merged,
                    max_metadata_bytes,
                },
                base_manifests,
                removals,
            )
            .await?;
            let manifests = f.manifests.clone();
            filtered = Some(f);
            manifests
        }
    };
    let next = new.manifests.len() + new.merged.len();
    let arranged = arrange(
        &mut MergeContext {
            io,
            metadata,
            snapshot_id,
            prefix,
            next,
            merged: &mut new.merged,
            max_metadata_bytes,
        },
        fresh,
        base_manifests,
        merge,
    )
    .await?;
    new.attempts += 1;
    let attempt = new.attempts;
    let path = format!(
        "{}/snap-{snapshot_id}-{attempt}-{}.avro",
        dir, new.commit_uuid
    );
    let file = io
        .new_output(&path)
        .map_err(from_iceberg)?
        .writer()
        .await
        .map_err(from_iceberg)?;
    let mut list = match version {
        FormatVersion::V1 => ManifestListWriter::v1(file, snapshot_id, parent_id),
        FormatVersion::V2 => ManifestListWriter::v2(file, snapshot_id, parent_id, sequence_number),
        FormatVersion::V3 => ManifestListWriter::v3(
            file,
            snapshot_id,
            parent_id,
            sequence_number,
            Some(first_row_id),
        ),
    };
    // Java's `buildManifestCountSummary`: a manifest of this snapshot is
    // created, one of another snapshot kept (an unassigned id is neither).
    let created = arranged
        .manifests
        .iter()
        .filter(|m| m.added_snapshot_id == snapshot_id)
        .count();
    let kept = arranged
        .manifests
        .iter()
        .filter(|m| m.added_snapshot_id != snapshot_id && m.added_snapshot_id != -1)
        .count();
    list.add_manifests(arranged.manifests.into_iter())
        .map_err(from_iceberg)?;
    let next_row_id = list.next_row_id();
    list.close().await.map_err(from_iceberg)?;

    let mut props = match &filtered {
        None => new.added.clone(),
        Some(f) => overwrite_summary(
            &new.added,
            &new.changed,
            &f.removed,
            partition_summary_limit(metadata),
        ),
    };
    for (total, added, removed) in TOTALS {
        // Java's `updateTotal`: a total the previous snapshot lacks, an
        // unparseable value or a negative result leaves the key out.
        let before: u64 = match previous {
            Some(prev) => match prev.get(total).and_then(|v| v.parse::<u64>().ok()) {
                Some(v) => v,
                None => continue,
            },
            None => 0,
        };
        let get = |k: &str| match props.get(k) {
            Some(v) => v.parse::<u64>().ok(),
            None => Some(0),
        };
        let (Some(add), Some(remove)) = (get(added), get(removed)) else {
            continue;
        };
        let Some(after) = before.checked_add(add).and_then(|v| v.checked_sub(remove)) else {
            continue;
        };
        props.insert(total.to_owned(), after.to_string());
    }
    props.insert(MANIFESTS_CREATED.to_owned(), created.to_string());
    props.insert(MANIFESTS_KEPT.to_owned(), kept.to_string());
    let replaced = arranged.replaced + filtered.as_ref().map_or(0, |f| f.replaced);
    props.insert(MANIFESTS_REPLACED.to_owned(), replaced.to_string());
    if new.removals.as_ref().is_some_and(|r| r.replace_partitions) {
        // `BaseReplacePartitions` (`:36`).
        props.insert("replace-partitions".to_owned(), "true".to_owned());
    }
    props.insert("engine-name".to_owned(), "krishiv".to_owned());
    props.insert(
        "engine-version".to_owned(),
        env!("CARGO_PKG_VERSION").to_owned(),
    );
    props.insert(WRITE_ID_PROP.to_owned(), ids.write_id.to_owned());
    if let Some(q) = ids.query_id {
        props.insert(QUERY_ID_PROP.to_owned(), q.to_owned());
    }

    let builder = Snapshot::builder()
        .with_snapshot_id(snapshot_id)
        .with_parent_snapshot_id(parent_id)
        .with_sequence_number(sequence_number)
        .with_timestamp_ms(timestamp_ms)
        .with_manifest_list(path)
        .with_schema_id(metadata.current_schema_id())
        .with_summary(Summary {
            operation: operation(new),
            additional_properties: props,
        });
    let snapshot = match (version, next_row_id) {
        (FormatVersion::V3, Some(next)) => builder
            .with_row_range(first_row_id, next.saturating_sub(first_row_id))
            .build(),
        _ => builder.build(),
    };
    let retention = base_ref
        .as_ref()
        .map_or_else(main_retention, |r| r.retention.clone());
    let mut updates = vec![
        TableUpdate::AddSnapshot {
            snapshot: snapshot.clone(),
        },
        TableUpdate::SetSnapshotRef {
            ref_name: branch.to_owned(),
            reference: SnapshotReference::new(snapshot_id, retention),
        },
    ];
    if let Some(statistics) = attempt_statistics_file(
        io,
        AttemptSnapshot {
            metadata,
            parent: parent.map(AsRef::as_ref),
            snapshot_id,
            sequence_number,
        },
        new,
        &format!("{dir}/{snapshot_id}-{}-{attempt}.stats", new.commit_uuid),
    )
    .await
    {
        updates.push(TableUpdate::SetStatistics { statistics });
    }
    Ok(Attempt {
        requirements: vec![
            TableRequirement::UuidMatch {
                uuid: metadata.uuid(),
            },
            TableRequirement::RefSnapshotIdMatch {
                r#ref: branch.to_owned(),
                snapshot_id: base_ref.as_ref().map(|r| r.snapshot_id),
            },
        ],
        updates,
        snapshot,
    })
}

/// The statistics file of an attempt (PR-54b, `statistics_file.rs`), written
/// at `path` from `new`'s sketches; `None` when the write has none, no column
/// describes the whole snapshot (rule 2), or writing failed (rule 3: counted,
/// logged, the statistics dropped for the later attempts too, and the path
/// left in `new.statistics_files` for cleanup).
async fn attempt_statistics_file(
    io: &FileIO,
    at: AttemptSnapshot<'_>,
    new: &mut NewManifests,
    path: &str,
) -> Option<StatisticsFile> {
    let sketches = new.statistics.as_ref()?;
    // Listed before the file exists, so a failed write's file is deleted.
    new.statistics_files.push(path.to_owned());
    match attempt_statistics(io, at, sketches, path).await {
        Ok(Some(file)) => Some(file),
        Ok(None) => {
            new.statistics_files.pop();
            None
        }
        Err(e) => {
            // Dropped for the rest of the write, counted once.
            new.statistics = None;
            skipped(&format!("writing the statistics file {path}: {e}"));
            None
        }
    }
}

/// The operation of `new`'s snapshot: `append` for an append; for an
/// overwrite Java's `BaseOverwriteFiles.operation` (module docs of
/// [`prepare`]).
fn operation(new: &NewManifests) -> Operation {
    if new.row_delta {
        // `BaseRowDelta.operation` (`BaseRowDelta.java:50-60`): removed
        // delete files do not count; an empty row delta is `overwrite`.
        let adds_data = !new.data_files.is_empty();
        let adds_deletes = !new.delete_files.is_empty();
        let deletes_data = new
            .removals
            .as_ref()
            .is_some_and(Removals::deletes_data_files);
        return match (adds_data, adds_deletes, deletes_data) {
            (true, false, false) => Operation::Append,
            (false, true, _) => Operation::Delete,
            _ => Operation::Overwrite,
        };
    }
    let Some(removals) = &new.removals else {
        return Operation::Append;
    };
    // `BaseReplacePartitions.operation` (`:45-48`).
    if removals.replace_partitions {
        return Operation::Overwrite;
    }
    let adds = !new.data_files.is_empty();
    let deletes = removals.deletes_data_files();
    match (adds, deletes) {
        (false, true) => Operation::Delete,
        (true, false) => Operation::Append,
        _ => Operation::Overwrite,
    }
}

/// A branch with no retention of its own (the table's defaults apply).
fn main_retention() -> SnapshotRetention {
    SnapshotRetention::Branch {
        min_snapshots_to_keep: None,
        max_snapshot_age_ms: None,
        max_ref_age_ms: None,
    }
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}
