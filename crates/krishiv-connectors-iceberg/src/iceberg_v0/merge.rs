// SPDX-License-Identifier: Apache-2.0
//! Manifest merging on append (P3-T027, PR-36a; design note §6.2 "Size and
//! merging"), as Java's `MergeAppend` (apache-iceberg 1.12.0
//! `MergingSnapshotProducer.apply`, `ManifestMergeManager.mergeManifests`,
//! `BinPacking.ListPacker.packEnd`); Spark's Iceberg append is
//! `table.newAppend()`, a `MergeAppend`, so this is the layout Spark writes.
//!
//! - **The list** ([`arrange`]): the data manifests (this write's new ones, then
//!   the base snapshot's), then the base's delete manifests; a base manifest
//!   with neither added nor existing files is dropped (Java's `shouldKeep`; a
//!   null count counts as having files, as `hasAddedFiles`). The dropped
//!   manifest is only left out of the list, never deleted.
//! - **Merging** ([`MergeConfig`], table properties
//!   `commit.manifest.target-size-bytes` (8 MiB),
//!   `commit.manifest.min-count-to-merge` (100) and
//!   `commit.manifest-merge.enabled` (true); a value that does not parse is
//!   `KRISHIV_UNSUPPORTED_FEATURE`, as the `commit.retry.*` budget; a boolean is
//!   `true` or `false` in any case, where Java's `Boolean.parseBoolean` reads any
//!   other text as false, a divergence toward refusal): data and delete
//!   manifests separately; `first` is the group's first input manifest; groups
//!   by partition spec id, highest first; within a group [`pack_end`] by
//!   manifest length with a lookback of 1, so the under-filled bin is the first
//!   one; a bin of one is kept, the bin holding `first` is merged only with at
//!   least `min-count-to-merge` manifests, any other bin of two or more is
//!   merged. A merged manifest is written at `<metadata dir>/<commit uuid>-m<n>.avro`
//!   (`n` continuing after the new manifests) under this write's snapshot id;
//!   its entries: a `DELETED` entry of another snapshot dropped (review-1 pass
//!   2, P2: only live entries carry over), an `ADDED` entry of this write
//!   stays `ADDED` with null (inherited) sequence numbers, every other entry is
//!   `EXISTING` with explicit data and file sequence numbers and the snapshot id
//!   read with inheritance (the rules of `plan::entry::inherit`), and on v3 an
//!   explicit `first_row_id` on every `EXISTING` data entry (null stays null),
//!   so the files keep their `_row_id`s.
//! - **First attempt only** (design note §6.2): merging runs only on a write's
//!   first attempt. A later attempt (a rebase or a new attempt after a
//!   conflict) lists the unmerged manifests, the drop and the data-then-delete
//!   order still applied, and never reuses a merged manifest of a failed
//!   attempt (it is derived from the old base). A divergence: Java re-runs
//!   `apply` and its merge on every attempt and caches merged bins across
//!   attempts. The summary's `manifests-replaced` therefore counts the base
//!   manifests merged by this attempt (0 on later ones), where Java's counter
//!   accumulates over its attempts until the commit.
//!
//! Manifests are not rolled at the target size while written (Java's
//! `RollingManifestWriter`): `write_new_manifests` writes one per spec.

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use iceberg::io::FileIO;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, FormatVersion, Manifest, ManifestContentType,
    ManifestFile, ManifestStatus, ManifestWriter, ManifestWriterBuilder, TableMetadata,
    UNASSIGNED_SEQUENCE_NUMBER,
};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};

use super::scan::read_capped;
use super::storage::from_iceberg;

/// Java's `TableProperties.MANIFEST_TARGET_SIZE_BYTES`.
pub(crate) const MANIFEST_TARGET_SIZE_BYTES: &str = "commit.manifest.target-size-bytes";
/// Java's `TableProperties.MANIFEST_MIN_MERGE_COUNT`.
pub(crate) const MANIFEST_MIN_MERGE_COUNT: &str = "commit.manifest.min-count-to-merge";
/// Java's `TableProperties.MANIFEST_MERGE_ENABLED`.
pub(crate) const MANIFEST_MERGE_ENABLED: &str = "commit.manifest-merge.enabled";

/// The merge settings of a table (module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MergeConfig {
    /// `commit.manifest.target-size-bytes` (Java's `long`).
    pub(crate) target_size_bytes: i64,
    /// `commit.manifest.min-count-to-merge` (Java's `int`).
    pub(crate) min_count_to_merge: i32,
    /// `commit.manifest-merge.enabled`.
    pub(crate) enabled: bool,
}

impl MergeConfig {
    /// The settings in `props`, Java's defaults for the unset ones.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` naming a value that does not parse.
    pub(crate) fn from_properties(props: &HashMap<String, String>) -> Result<Self, KrishivError> {
        // No trim: Java's `Long.parseLong`/`Integer.parseInt` refuse " 4".
        let target_size_bytes = match props.get(MANIFEST_TARGET_SIZE_BYTES) {
            None => 8 * 1024 * 1024,
            Some(v) => v
                .parse::<i64>()
                .map_err(|_| not_a(MANIFEST_TARGET_SIZE_BYTES, v, "a number"))?,
        };
        let min_count_to_merge = match props.get(MANIFEST_MIN_MERGE_COUNT) {
            None => 100,
            Some(v) => v
                .parse::<i32>()
                .map_err(|_| not_a(MANIFEST_MIN_MERGE_COUNT, v, "a number"))?,
        };
        let enabled = match props.get(MANIFEST_MERGE_ENABLED) {
            None => true,
            Some(v) if v.eq_ignore_ascii_case("true") => true,
            Some(v) if v.eq_ignore_ascii_case("false") => false,
            Some(v) => return Err(not_a(MANIFEST_MERGE_ENABLED, v, "true or false")),
        };
        Ok(Self {
            target_size_bytes,
            min_count_to_merge,
            enabled,
        })
    }
}

fn not_a(key: &str, value: &str, what: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: format!("committing to an Iceberg table whose {key} is {value:?}, not {what}"),
    }
    .into()
}

/// Java's `ListPacker(target, 1, false).packEnd(items, weight)`: the indices of
/// `weights` in bins, in order, packed from the end with one open bin, so the
/// first bin is the under-filled one. An item heavier than `target` is alone.
pub(crate) fn pack_end(weights: &[i64], target: i64) -> Vec<Vec<usize>> {
    let target = i128::from(target);
    let mut bins: Vec<Vec<usize>> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    let mut weight: i128 = 0;
    for (i, w) in weights.iter().enumerate().rev() {
        let w = i128::from(*w);
        if open.is_empty() {
            weight = w;
            open.push(i);
        } else if weight + w <= target {
            weight += w;
            open.push(i);
        } else {
            bins.push(std::mem::replace(&mut open, vec![i]));
            weight = w;
        }
    }
    if !open.is_empty() {
        bins.push(open);
    }
    bins.reverse();
    for b in &mut bins {
        b.reverse();
    }
    bins
}

/// What a merge writes through: the table, the write's names and the
/// merged manifests it has written so far.
pub(crate) struct MergeContext<'a> {
    /// The table's file access.
    pub(crate) io: &'a FileIO,
    /// The metadata of the attempt.
    pub(crate) metadata: &'a TableMetadata,
    /// The write's snapshot id.
    pub(crate) snapshot_id: i64,
    /// `<metadata dir>/<commit uuid>-m`, the merged manifests' path prefix.
    pub(crate) prefix: String,
    /// The number of the next manifest written.
    pub(crate) next: usize,
    /// Every merged manifest path written (pushed before the write, so a
    /// failed write's partial file is cleaned up with the others).
    pub(crate) merged: &'a mut Vec<String>,
    /// `krishiv.catalog.maxMetadataBytes` for each manifest read.
    pub(crate) max_metadata_bytes: u64,
}

/// An attempt's manifests in list order, and how many base manifests were
/// merged (`manifests-replaced`).
#[derive(Debug)]
pub(crate) struct Arranged {
    /// The manifests of the list.
    pub(crate) manifests: Vec<ManifestFile>,
    /// Base manifests merged into a new manifest.
    pub(crate) replaced: usize,
}

/// The list of an append's attempt (module docs): `fresh` (this write's new
/// data manifests), then the base's data manifests of `base`, then its delete
/// manifests, each kind merged when `merge` is `Some`.
///
/// # Errors
/// [`merge_manifests`]'s.
pub(crate) async fn arrange(
    cx: &mut MergeContext<'_>,
    fresh: Vec<ManifestFile>,
    base: Vec<ManifestFile>,
    merge: Option<MergeConfig>,
) -> Result<Arranged, KrishivError> {
    let ours = cx.snapshot_id;
    let keep = |m: &ManifestFile| {
        m.has_added_files() || m.has_existing_files() || m.added_snapshot_id == ours
    };
    let (base_data, base_deletes): (Vec<_>, Vec<_>) = base
        .into_iter()
        .filter(keep)
        .partition(|m| m.content == ManifestContentType::Data);
    let data: Vec<ManifestFile> = fresh.into_iter().filter(keep).chain(base_data).collect();
    let mut replaced = 0;
    let mut manifests = Vec::new();
    for group in [data, base_deletes] {
        match merge {
            Some(config) if config.enabled => {
                let (merged, n) = merge_manifests(cx, group, config).await?;
                manifests.extend(merged);
                replaced += n;
            }
            _ => manifests.extend(group),
        }
    }
    Ok(Arranged {
        manifests,
        replaced,
    })
}

/// Java's `ManifestMergeManager.mergeManifests` over `manifests` (one content
/// kind): the output manifests and the number of manifests of other
/// snapshots merged.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for an encrypted manifest to merge;
/// `KRISHIV_CATALOG_LIMIT_EXCEEDED` for a manifest over the parse limit; an
/// internal error for an entry without the sequence numbers a rewrite needs;
/// iceberg-rust's read, parse and write failures.
pub(crate) async fn merge_manifests(
    cx: &mut MergeContext<'_>,
    manifests: Vec<ManifestFile>,
    config: MergeConfig,
) -> Result<(Vec<ManifestFile>, usize), KrishivError> {
    if manifests.is_empty() {
        return Ok((manifests, 0));
    }
    // `first` is index 0; groups by spec id, highest first.
    let mut groups: BTreeMap<Reverse<i32>, Vec<(usize, ManifestFile)>> = BTreeMap::new();
    for (i, m) in manifests.into_iter().enumerate() {
        groups
            .entry(Reverse(m.partition_spec_id))
            .or_default()
            .push((i, m));
    }
    let mut out = Vec::new();
    let mut replaced = 0;
    for (Reverse(spec_id), group) in groups {
        let weights: Vec<i64> = group.iter().map(|(_, m)| m.manifest_length).collect();
        for bin in pack_end(&weights, config.target_size_bytes) {
            let members: Vec<&(usize, ManifestFile)> = bin.iter().map(|&j| &group[j]).collect();
            let has_first = members.iter().any(|(i, _)| *i == 0);
            let below = i64::try_from(members.len()).unwrap_or(i64::MAX)
                < i64::from(config.min_count_to_merge);
            if members.len() == 1 || (has_first && below) {
                out.extend(members.iter().map(|(_, m)| m.clone()));
                continue;
            }
            let bin: Vec<&ManifestFile> = members.iter().map(|(_, m)| m).collect();
            out.push(write_merged(cx, spec_id, &bin).await?);
            replaced += bin
                .iter()
                .filter(|m| m.added_snapshot_id != cx.snapshot_id)
                .count();
        }
    }
    Ok((out, replaced))
}

/// One merged manifest of `bin` (all of spec `spec_id` and one content kind),
/// as Java's `ManifestMergeManager.createManifest` (module docs).
async fn write_merged(
    cx: &mut MergeContext<'_>,
    spec_id: i32,
    bin: &[&ManifestFile],
) -> Result<ManifestFile, KrishivError> {
    let metadata = cx.metadata;
    let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
        internal(format!(
            "a manifest of partition spec {spec_id}, which the table lacks"
        ))
    })?;
    let content = bin.first().map_or(ManifestContentType::Data, |m| m.content);
    let version = metadata.format_version();
    let path = format!("{}{}.avro", cx.prefix, cx.next);
    cx.next += 1;
    cx.merged.push(path.clone());
    let output = cx.io.new_output(&path).map_err(from_iceberg)?;
    let builder = ManifestWriterBuilder::new(
        output,
        Some(cx.snapshot_id),
        Arc::clone(metadata.current_schema()),
        spec.as_ref().clone(),
    );
    let mut writer = match (version, content) {
        (FormatVersion::V1, ManifestContentType::Data) => builder.build_v1(),
        (FormatVersion::V2, ManifestContentType::Data) => builder.build_v2_data(),
        (FormatVersion::V2, ManifestContentType::Deletes) => builder.build_v2_deletes(),
        (FormatVersion::V3, ManifestContentType::Data) => builder.build_v3_data(),
        (FormatVersion::V3, ManifestContentType::Deletes) => builder.build_v3_deletes(),
        (FormatVersion::V1, ManifestContentType::Deletes) => {
            return Err(internal(
                "a delete manifest in a format version 1 table".to_owned(),
            ));
        }
    };
    for m in bin {
        copy_entries(cx, m, version, &mut writer).await?;
    }
    writer.write_manifest_file().await.map_err(from_iceberg)
}

/// The entries of `m` into `writer` (module docs).
async fn copy_entries(
    cx: &MergeContext<'_>,
    m: &ManifestFile,
    version: FormatVersion,
    writer: &mut ManifestWriter,
) -> Result<(), KrishivError> {
    if m.key_metadata.is_some() {
        return Err(KrishivUnsupportedFeature {
            feature: "merging encrypted Iceberg manifests".to_owned(),
        }
        .into());
    }
    let declared = u64::try_from(m.manifest_length).ok().filter(|n| *n > 0);
    let bytes = read_capped(
        cx.io,
        "manifest",
        &m.manifest_path,
        declared,
        cx.max_metadata_bytes,
    )
    .await?;
    let (entries, _) = Manifest::parse_avro(&bytes)
        .map_err(from_iceberg)?
        .into_parts();
    let ours = cx.snapshot_id;
    let lineage = version == FormatVersion::V3 && m.content == ManifestContentType::Data;
    // `plan::entry::inherit`'s row-id rule: assigned in order to the live
    // entries without one, from the manifest's `first_row_id`; none without it.
    let mut next_row_id = match (lineage, m.first_row_id) {
        (true, Some(id)) => Some(i64::try_from(id).map_err(|_| {
            internal(format!(
                "first_row_id {id} of manifest {} exceeds the long range",
                m.manifest_path
            ))
        })?),
        _ => None,
    };
    for e in &entries {
        let status = e.status();
        let snapshot_id = e.snapshot_id().unwrap_or(m.added_snapshot_id);
        let inherits = status == ManifestStatus::Added || m.sequence_number == 0;
        let seq = |v: Option<i64>| v.or(inherits.then_some(m.sequence_number));
        let data_seq = seq(e.sequence_number());
        let file_seq = seq(e.file_sequence_number);
        let file = e.data_file();
        let row_id = match &mut next_row_id {
            Some(next) if status != ManifestStatus::Deleted && file.first_row_id().is_none() => {
                let id = *next;
                *next = next
                    .checked_add_unsigned(file.record_count())
                    .ok_or_else(|| {
                        internal(format!("row id overflow in manifest {}", m.manifest_path))
                    })?;
                Some(id)
            }
            Some(_) => file.first_row_id(),
            None => None,
        };
        let explicit = |v: Option<i64>, what: &str| {
            v.filter(|n| *n >= 0).ok_or_else(|| {
                internal(format!(
                    "the {what} sequence number of {} in manifest {} is unknown, so the \
                     manifest cannot be merged",
                    file.file_path(),
                    m.manifest_path
                ))
            })
        };
        let result = match status {
            ManifestStatus::Deleted if snapshot_id == ours => writer.add_delete_file(
                file.clone(),
                explicit(data_seq, "data")?,
                Some(explicit(file_seq, "file")?),
            ),
            // A removal by another snapshot: not live, never carried over.
            ManifestStatus::Deleted => continue,
            ManifestStatus::Added if snapshot_id == ours => writer.add_file(
                file.clone(),
                e.sequence_number().unwrap_or(UNASSIGNED_SEQUENCE_NUMBER),
            ),
            ManifestStatus::Added | ManifestStatus::Existing => {
                let file = if lineage && file.content_type() == DataContentType::Data {
                    with_first_row_id(file, row_id, m.partition_spec_id)?
                } else {
                    file.clone()
                };
                writer.add_existing_file(
                    file,
                    snapshot_id,
                    explicit(data_seq, "data")?,
                    Some(explicit(file_seq, "file")?),
                )
            }
        };
        result.map_err(from_iceberg)?;
    }
    Ok(())
}

/// `f` with `first_row_id` `id` (iceberg-rust has no setter: every field is
/// copied through its builder).
pub(crate) fn with_first_row_id(
    f: &DataFile,
    id: Option<i64>,
    spec_id: i32,
) -> Result<DataFile, KrishivError> {
    if f.first_row_id() == id {
        return Ok(f.clone());
    }
    let mut b = DataFileBuilder::default();
    b.content(f.content_type())
        .file_path(f.file_path().to_owned())
        .file_format(f.file_format())
        .partition(f.partition().clone())
        .record_count(f.record_count())
        .file_size_in_bytes(f.file_size_in_bytes())
        .column_sizes(f.column_sizes().clone())
        .value_counts(f.value_counts().clone())
        .null_value_counts(f.null_value_counts().clone())
        .nan_value_counts(f.nan_value_counts().clone())
        .lower_bounds(f.lower_bounds().clone())
        .upper_bounds(f.upper_bounds().clone())
        .key_metadata(f.key_metadata().map(<[u8]>::to_vec))
        .split_offsets(f.split_offsets().map(<[i64]>::to_vec))
        .equality_ids(f.equality_ids())
        .first_row_id(id)
        .partition_spec_id(spec_id)
        .referenced_data_file(f.referenced_data_file())
        .content_offset(f.content_offset())
        .content_size_in_bytes(f.content_size_in_bytes());
    if let Some(order) = f.sort_order_id() {
        b.sort_order_id(order);
    }
    b.build()
        .map_err(|e| internal(format!("data file {}: {e}", f.file_path())))
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}
