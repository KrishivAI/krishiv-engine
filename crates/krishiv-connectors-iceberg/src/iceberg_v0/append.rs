// SPDX-License-Identifier: Apache-2.0
//! An append's metadata (P3-T027, PR-36a; design note §6.2, §6.3, §7.2): the
//! manifests of the files a write added, written once per write
//! ([`write_new_manifests`]), and per commit attempt the manifest list, the
//! snapshot, and the catalog request's updates and requirements
//! ([`prepare_attempt`]). Nothing here sends the request: the commit loop
//! (§7) does, and on a conflict calls [`prepare_attempt`] again on the
//! refreshed metadata with the same manifests and snapshot id.
//!
//! As Java's `SnapshotProducer`/`FastAppend` (apache-iceberg 1.12.0):
//! - one manifest per partition spec, at `<metadata dir>/<commit uuid>-m<n>.avro`;
//!   entries `ADDED` with the write's snapshot id and null (inherited)
//!   sequence numbers, so the manifests are valid on any base;
//! - the list at `<metadata dir>/snap-<snapshot id>-<attempt>-<commit uuid>.avro`,
//!   the new manifests first, then the base snapshot's, unchanged; the new
//!   manifests reset to an unassigned sequence number and no `first_row_id`
//!   before each attempt (§7.3 step 1), so the list writer assigns them;
//! - the sequence number is the table's `last-sequence-number` + 1 (v1: 0);
//!   v3 takes `first-row-id` from the table's `next-row-id` and `added-rows`
//!   from what the list writer assigned;
//! - the summary: `operation` `append`, iceberg-rust's added-* counts and
//!   partition summaries (`write.summary.partition-limit`), the `total-*` values
//!   from the parent's (each left out when the parent lacks it; from 0 without
//!   a parent), `engine-name`/`engine-version`, `krishiv.write-id` and
//!   `krishiv.query-id`;
//! - requirements `assert-table-uuid` and `assert-ref-snapshot-id` (the base,
//!   or null for a ref that does not exist yet); updates `add-snapshot` and
//!   `set-snapshot-ref`, the branch keeping its retention.
//!
//! Manifests are not split at `commit.manifest.target-size-bytes` and not
//! merged (`commit.manifest-merge.enabled`) yet: PR-36a's later commit.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use iceberg::io::FileIO;
use iceberg::spec::{
    DataFile, FormatVersion, ManifestFile, ManifestListWriter, ManifestWriterBuilder, Operation,
    Snapshot, SnapshotReference, SnapshotRetention, SnapshotSummaryCollector, Summary,
    TableMetadata, UNASSIGNED_SEQUENCE_NUMBER,
};
use iceberg::{TableRequirement, TableUpdate};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};

use super::data_file::check_partition;
use super::scan::{read_manifest_files, snapshot_refs};
use super::storage::from_iceberg;

/// `write.metadata.path` (Java's `TableProperties.WRITE_METADATA_LOCATION`).
pub(crate) const WRITE_METADATA_PATH: &str = "write.metadata.path";

/// `write.summary.partition-limit`.
const PARTITION_SUMMARY_LIMIT: &str = "write.summary.partition-limit";

/// Java's `SnapshotSummary.CHANGED_PARTITION_COUNT_PROP`.
const CHANGED_PARTITION_COUNT: &str = "changed-partition-count";

/// Java's `SnapshotSummary.PARTITION_SUMMARY_PROP`.
const PARTITION_SUMMARIES_INCLUDED: &str = "partition-summaries-included";

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
    /// Attempts prepared so far (Java's `SnapshotProducer.attempt`).
    attempts: u32,
}

/// Writes the manifests of `files` (data files, each with the id of the spec of
/// `metadata` it was written under: iceberg-rust's `DataFile` does not expose it)
/// under the snapshot id `snapshot_id` (module docs).
///
/// # Errors
/// An internal error for a file of a spec the table lacks or whose partition
/// tuple does not fit its spec ([`check_partition`], checked before any
/// iceberg-rust call, which would panic); iceberg-rust's write failures, with
/// their Krishiv class when the storage raised one.
pub(crate) async fn write_new_manifests(
    io: &FileIO,
    metadata: &TableMetadata,
    files: Vec<(i32, DataFile)>,
    snapshot_id: i64,
    commit_uuid: &str,
) -> Result<NewManifests, KrishivError> {
    let schema = Arc::clone(metadata.current_schema());
    let dir = metadata_dir(metadata);
    let mut by_spec: BTreeMap<i32, Vec<DataFile>> = BTreeMap::new();
    for (spec_id, f) in files {
        by_spec.entry(spec_id).or_default().push(f);
    }
    let mut summary = SnapshotSummaryCollector::default();
    summary.set_partition_summary_limit(partition_summary_limit(metadata));
    // Java's `changed-partition-count` counts every partition path, the
    // unpartitioned "" included; iceberg-rust's collector skips unpartitioned files.
    let mut changed: HashSet<String> = HashSet::new();
    // Every file is checked before any manifest is written, so a bad file
    // leaves no manifest behind (review 36a N9).
    let mut specs = Vec::with_capacity(by_spec.len());
    for (spec_id, files) in &by_spec {
        let spec = metadata.partition_spec_by_id(*spec_id).ok_or_else(|| {
            internal(format!(
                "a data file of partition spec {spec_id}, which the table lacks"
            ))
        })?;
        for f in files {
            check_partition(spec, &schema, f.partition())?;
            changed.insert(spec.partition_to_path(f.partition(), Arc::clone(&schema)));
        }
        specs.push(Arc::clone(spec));
    }
    let mut manifests = Vec::with_capacity(by_spec.len());
    for (n, (spec, files)) in specs.iter().zip(by_spec.into_values()).enumerate() {
        let path = format!("{dir}/{commit_uuid}-m{n}.avro");
        let output = io.new_output(&path).map_err(from_iceberg)?;
        let builder = ManifestWriterBuilder::new(
            output,
            Some(snapshot_id),
            Arc::clone(&schema),
            spec.as_ref().clone(),
        );
        let mut writer = match metadata.format_version() {
            FormatVersion::V1 => builder.build_v1(),
            FormatVersion::V2 => builder.build_v2_data(),
            FormatVersion::V3 => builder.build_v3_data(),
        };
        for f in files {
            summary.add_file(&f, Arc::clone(&schema), Arc::clone(spec));
            writer
                .add_file(f, UNASSIGNED_SEQUENCE_NUMBER)
                .map_err(from_iceberg)?;
        }
        manifests.push(writer.write_manifest_file().await.map_err(from_iceberg)?);
    }
    let mut added = summary.build();
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
        attempts: 0,
    })
}

/// The branch a write commits to and who wrote (for the summary).
#[derive(Debug, Clone, Copy)]
pub(crate) struct WriteIds<'a> {
    /// The branch.
    pub(crate) branch: &'a str,
    /// The write's idempotency key.
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
    /// `add-snapshot`, `set-snapshot-ref`.
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
pub(crate) async fn prepare_attempt(
    io: &FileIO,
    metadata: &TableMetadata,
    new: &mut NewManifests,
    ids: WriteIds<'_>,
    timestamp_ms: i64,
    max_metadata_bytes: u64,
) -> Result<Attempt, KrishivError> {
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
        return Err(KrishivUnsupportedFeature {
            feature: format!(
                "{branch} is a tag, not a branch. Tags cannot be targets for producing snapshots"
            ),
        }
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
    new.attempts += 1;
    let attempt = new.attempts;
    let path = format!(
        "{}/snap-{snapshot_id}-{attempt}-{}.avro",
        metadata_dir(metadata),
        new.commit_uuid
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
    let fresh = new.manifests.iter().cloned().map(|mut m| {
        m.sequence_number = UNASSIGNED_SEQUENCE_NUMBER;
        m.min_sequence_number = UNASSIGNED_SEQUENCE_NUMBER;
        m.first_row_id = None;
        m
    });
    list.add_manifests(fresh.chain(base_manifests))
        .map_err(from_iceberg)?;
    let next_row_id = list.next_row_id();
    list.close().await.map_err(from_iceberg)?;

    let mut props = new.added.clone();
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
            operation: Operation::Append,
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
        updates: vec![
            TableUpdate::AddSnapshot {
                snapshot: snapshot.clone(),
            },
            TableUpdate::SetSnapshotRef {
                ref_name: branch.to_owned(),
                reference: SnapshotReference::new(snapshot_id, retention),
            },
        ],
        snapshot,
    })
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
