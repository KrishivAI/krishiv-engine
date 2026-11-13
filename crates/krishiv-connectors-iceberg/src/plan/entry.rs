// SPDX-License-Identifier: Apache-2.0
//! Manifest-list entries, manifest entries and the files they describe, as
//! Krishiv-owned values, and the inheritance of the fields a writer may leave
//! null (table spec "Sequence Number Inheritance", "First Row ID Inheritance";
//! FC B2: iceberg-rust's public `Manifest::parse_avro` does not inherit, so the
//! planner does it here).
//!
//! Bounds, partition values and partition summaries stay in the spec's
//! single-value serialisation; [`super::prune`] decodes them by width against
//! the read schema ([`super::bounds`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_io::PushdownExpr;

/// What a manifest holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ManifestContent {
    /// Data files.
    Data,
    /// Delete files (position, equality, deletion vectors).
    Deletes,
}

/// One partition field's summary in a manifest-list entry.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PartitionFieldSummary {
    /// Whether some file has a null value for the field.
    pub contains_null: bool,
    /// Whether some file has a NaN value (`None`: not written).
    pub contains_nan: Option<bool>,
    /// The smallest non-null, non-NaN value (serialised).
    pub lower: Option<Vec<u8>>,
    /// The largest non-null, non-NaN value (serialised).
    pub upper: Option<Vec<u8>>,
}

/// An entry of a snapshot's manifest list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestFileEntry {
    /// The manifest's location.
    pub path: String,
    /// Its length in bytes.
    pub length: i64,
    /// The partition spec its files were written with.
    pub spec_id: i32,
    /// Data or deletes.
    pub content: ManifestContent,
    /// The sequence number of the commit that added it (0 for format v1).
    pub sequence_number: i64,
    /// The smallest data sequence number of its live files.
    pub min_sequence_number: i64,
    /// The snapshot that added it.
    pub added_snapshot_id: i64,
    /// Files added, existing and deleted (`None`: not written).
    pub added_files: Option<u32>,
    /// See `added_files`.
    pub existing_files: Option<u32>,
    /// See `added_files`.
    pub deleted_files: Option<u32>,
    /// The first row id of its files (format v3).
    pub first_row_id: Option<u64>,
    /// Partition summaries, one per field of the spec.
    pub partitions: Option<Vec<PartitionFieldSummary>>,
    /// Whether the manifest is encrypted (it carries key metadata).
    pub encrypted: bool,
}

impl ManifestFileEntry {
    /// Whether the manifest may hold live files: a manifest that says it has
    /// neither added nor existing files holds only deleted entries.
    #[must_use]
    pub fn may_have_live_files(&self) -> bool {
        !matches!((self.added_files, self.existing_files), (Some(0), Some(0)))
    }
}

/// A manifest entry's status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntryStatus {
    /// Carried over from an earlier snapshot.
    Existing,
    /// Added by the manifest's snapshot.
    Added,
    /// Removed by the manifest's snapshot (not part of the table).
    Deleted,
}

/// What a content file holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentKind {
    /// Rows.
    Data,
    /// Position deletes (`file_path`, `pos`) or, in a Puffin file, a deletion vector.
    PositionDeletes,
    /// Equality deletes.
    EqualityDeletes,
}

/// A content file's format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileFormatKind {
    /// Parquet.
    Parquet,
    /// ORC.
    Orc,
    /// Avro.
    Avro,
    /// Puffin (deletion vectors).
    Puffin,
}

/// A partition value of a content file, in the spec's serialisation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PartitionValue {
    /// Null.
    Null,
    /// The serialised value (decoded with the partition field's result type).
    Value(Vec<u8>),
    /// A value the reader could not serialise: never pruned on.
    Unreadable,
}

/// A manifest entry as parsed, before inheritance.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedEntry {
    /// The status.
    pub status: EntryStatus,
    /// The snapshot that added the file (null: the manifest's).
    pub snapshot_id: Option<i64>,
    /// Data sequence number (null: inherited).
    pub sequence_number: Option<i64>,
    /// File sequence number (null: inherited).
    pub file_sequence_number: Option<i64>,
    /// The file, with its `first_row_id` as written.
    pub file: ContentFile,
}

/// The fields every content file has (data and delete files alike).
#[derive(Debug, Clone, PartialEq)]
pub struct ContentFile {
    /// Data, position or equality deletes.
    pub content: ContentKind,
    /// The file's location.
    pub path: String,
    /// Its format.
    pub format: FileFormatKind,
    /// The partition spec it was written with.
    pub spec_id: i32,
    /// Its partition tuple, in the spec's field order.
    pub partition: Vec<PartitionValue>,
    /// Number of records (rows, or deletes).
    pub record_count: u64,
    /// Size in bytes.
    pub size: u64,
    /// Bytes per column, by field id.
    pub column_sizes: BTreeMap<i32, u64>,
    /// Values per column (nulls and NaNs included), by field id.
    pub value_counts: BTreeMap<i32, u64>,
    /// Nulls per column, by field id.
    pub null_counts: BTreeMap<i32, u64>,
    /// NaNs per column, by field id.
    pub nan_counts: BTreeMap<i32, u64>,
    /// Lower bounds (serialised; strings and binaries may be truncated).
    pub lower_bounds: BTreeMap<i32, Vec<u8>>,
    /// Upper bounds (serialised; a truncated string or binary is rounded up by
    /// conforming writers).
    pub upper_bounds: BTreeMap<i32, Vec<u8>>,
    /// Split offsets, ascending.
    pub split_offsets: Option<Vec<i64>>,
    /// The sort order id.
    pub sort_order_id: Option<i32>,
    /// The first row id (format v3).
    pub first_row_id: Option<i64>,
    /// Equality field ids (equality deletes).
    pub equality_ids: Option<Vec<i32>>,
    /// The data file a deletion vector or file-scoped position delete applies to.
    pub referenced_data_file: Option<String>,
    /// A deletion vector's blob offset in its Puffin file.
    pub content_offset: Option<i64>,
    /// A deletion vector's blob length.
    pub content_size_in_bytes: Option<i64>,
}

/// A live data file of a snapshot, with its sequence numbers and snapshot id
/// inherited.
#[derive(Debug, Clone, PartialEq)]
pub struct DataFileEntry {
    /// The file (with `first_row_id` inherited in format v3).
    pub file: ContentFile,
    /// Existing or added.
    pub status: EntryStatus,
    /// The snapshot that added it.
    pub snapshot_id: i64,
    /// Data sequence number.
    pub data_sequence_number: i64,
    /// File sequence number.
    pub file_sequence_number: i64,
}

/// What a delete file deletes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeleteKind {
    /// A v2 position delete file.
    PositionDelete,
    /// An equality delete file.
    EqualityDelete,
    /// A v3 deletion vector (a Puffin blob).
    DeletionVector,
}

/// A live delete file of a snapshot (parsed and carried; applied by PR-25b).
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteFileEntry {
    /// The file and its inherited numbers.
    pub entry: DataFileEntry,
    /// What it deletes.
    pub kind: DeleteKind,
}

/// A live entry of either kind.
#[derive(Debug, Clone, PartialEq)]
pub enum LiveEntry {
    /// A data file.
    Data(DataFileEntry),
    /// A delete file.
    Delete(DeleteFileEntry),
}

/// The live entries of `manifest` after inheritance (DELETED entries skipped):
/// - a null snapshot id takes the manifest's `added_snapshot_id`;
/// - a null data or file sequence number takes the manifest's sequence number
///   when the entry is ADDED, or when the manifest's is 0 (format v1, where every
///   entry has sequence number 0);
/// - in a data manifest with a `first_row_id`, each live file without one is
///   assigned the running id, which then advances by its record count (format v3);
///   a data manifest without one leaves every file's `first_row_id` null.
///
/// # Errors
/// `INTERNAL_ERROR` for an EXISTING entry with no sequence number in a manifest
/// whose own is not 0 (the spec requires it to be written), and for a row id
/// that overflows.
pub fn inherit(
    manifest: &ManifestFileEntry,
    entries: Vec<ParsedEntry>,
) -> Result<Vec<LiveEntry>, KrishivError> {
    let mut next_row_id = match (manifest.content, manifest.first_row_id) {
        (ManifestContent::Data, Some(id)) => Some(i64::try_from(id).map_err(|_| {
            internal(format!(
                "first_row_id {id} of manifest {} exceeds the long range",
                manifest.path
            ))
        })?),
        _ => None,
    };
    let mut out = Vec::with_capacity(entries.len());
    for e in entries {
        if e.status == EntryStatus::Deleted {
            continue;
        }
        let inherits = e.status == EntryStatus::Added || manifest.sequence_number == 0;
        let seq = |v: Option<i64>, what: &str| match v {
            Some(v) => Ok(v),
            None if inherits => Ok(manifest.sequence_number),
            None => Err(internal(format!(
                "an existing entry for {} in manifest {} has no {what} sequence number",
                e.file.path, manifest.path
            ))),
        };
        let data_sequence_number = seq(e.sequence_number, "data")?;
        let file_sequence_number = seq(e.file_sequence_number, "file")?;
        let mut file = e.file;
        if manifest.content == ManifestContent::Data {
            match &mut next_row_id {
                None => file.first_row_id = None,
                Some(next) if file.first_row_id.is_none() => {
                    file.first_row_id = Some(*next);
                    *next = next
                        .checked_add_unsigned(file.record_count)
                        .ok_or_else(|| {
                            internal(format!("row id overflow in manifest {}", manifest.path))
                        })?;
                }
                Some(_) => {}
            }
        }
        let entry = DataFileEntry {
            status: e.status,
            snapshot_id: e.snapshot_id.unwrap_or(manifest.added_snapshot_id),
            data_sequence_number,
            file_sequence_number,
            file,
        };
        out.push(match entry.file.content {
            ContentKind::Data => LiveEntry::Data(entry),
            ContentKind::EqualityDeletes => LiveEntry::Delete(DeleteFileEntry {
                entry,
                kind: DeleteKind::EqualityDelete,
            }),
            ContentKind::PositionDeletes => {
                let kind = if entry.file.format == FileFormatKind::Puffin {
                    DeleteKind::DeletionVector
                } else {
                    DeleteKind::PositionDelete
                };
                LiveEntry::Delete(DeleteFileEntry { entry, kind })
            }
        });
    }
    Ok(out)
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// Which manifests a plan reads. Planning "files added between two snapshots"
/// (P4-T025 changelog reads, T027's conflict validation) restricts the
/// manifests by `added_snapshot_id`; that variant comes with its first user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ManifestFilter {
    /// Every manifest of the snapshot.
    #[default]
    All,
}

/// One data file to read: the whole file (iceberg-rust's tasks do not split
/// either; splitting by `split_offsets` is the scan's), with the predicate the
/// rows must still pass.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FileScanTask {
    /// The data file.
    pub file: Arc<DataFileEntry>,
    /// The predicate evaluated on the rows (pruning keeps a superset).
    pub residual: Option<PushdownExpr>,
}

/// The files of a snapshot that may hold matching rows.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct ScanPlan {
    /// The snapshot planned (`None`: the table has no snapshot, so no rows).
    pub snapshot_id: Option<i64>,
    /// The data files to read, in manifest-list then manifest order.
    pub tasks: Vec<FileScanTask>,
    /// The live delete files of the surviving delete manifests (assigned to tasks
    /// by PR-25b).
    pub delete_files: Vec<Arc<DeleteFileEntry>>,
    /// Manifests in the list.
    pub manifests_total: usize,
    /// Manifests skipped by their partition summaries or counts.
    pub manifests_pruned: usize,
    /// Live data files skipped by partition values or metrics.
    pub files_pruned: usize,
}
