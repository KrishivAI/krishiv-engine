// SPDX-License-Identifier: Apache-2.0
//! The removals of an `OverwriteFiles` commit applied to a base snapshot's
//! manifests (P3-T027, PR-36b part 2a; design note §7.3 step 4a.2), as
//! apache-iceberg 1.12.0's `ManifestFilterManager` and the part of
//! `MergingSnapshotProducer.apply` that drives it
//! (`MergingSnapshotProducer.java:1018-1063`). Run on **every attempt**
//! against that attempt's base, so a rebased commit removes the files from
//! the manifests the branch holds then (a foreign merge may have moved them).
//!
//! - **Data manifests** ([`filter_manifests`]): a live entry whose path is
//!   one of [`Removals::data_files`] is removed (Java's `deleteFile(DataFile)`,
//!   a `DataFileSet` by location). A manifest without live files, or whose
//!   live entries hit nothing, is kept as it is, never rewritten
//!   (`ManifestFilterManager.java:377-380`, `:393-396`). A manifest with a hit
//!   is rewritten at `<metadata dir>/<commit uuid>-m<n>.avro` under the
//!   write's snapshot id, from its **live entries only** (review-1 P2: an
//!   older `DELETED` entry copied as `EXISTING` would bring a removed file
//!   back): a hit `DELETED` (the write's snapshot id, its sequence numbers),
//!   the others `EXISTING` with their own snapshot id, explicit data and file
//!   sequence numbers and, in a v3 data manifest, an explicit `first_row_id`
//!   (inherited as `plan::entry::inherit` does), so the files keep their
//!   sequence numbers and row ids. A path live twice in one manifest is
//!   removed twice and counted once, the second as a duplicate
//!   (`deleted-duplicate-files`, `:542-552`).
//! - **Delete manifests**, after the data manifests: a live delete file is
//!   removed when its [`DeleteKey`] (location, referenced data file and
//!   content offset: one Puffin file holds the DVs of many data files) is one
//!   of [`Removals::delete_files`], when it is a
//!   DV (Puffin) of a data file removed above (`removeDanglingDeletesFor`,
//!   `isDanglingDV`, `:499-501`), or when its data sequence number is above 0
//!   and below the smallest data sequence number of the filtered data
//!   manifests (`dropDeleteFilesOlderThan`: the manifests' assigned
//!   `min_sequence_number`, from the table's `last-sequence-number`,
//!   `MergingSnapshotProducer.java:1025-1034`): such a delete file applies
//!   to no live data file.
//! - **Missing files** (`failMissingDeletePaths`, `:282-304`): every data
//!   file path and every delete file key of [`Removals`] must have been
//!   removed from some manifest (a DV replaced concurrently is missing even
//!   when its Puffin file still holds other live DVs), else
//!   [`FilterError::Missing`] with Java's "Missing required files to delete:
//!   %s"; the commit loop turns it into `KRISHIV_ICEBERG_COMMIT_CONFLICT`.
//! - **Summary** ([`Removed`]): Java's `SnapshotSummary.UpdateMetrics` for
//!   each removed file (a DV's size is its blob's, `ScanTaskUtil
//!   .contentSizeInBytes`), per manifest; the partitions of removed files
//!   count as changed; [`overwrite_summary`] merges them with the added
//!   files' values.
//!
//! - **Row filter and dropped partitions** (PR-36b part 2b, INSERT
//!   OVERWRITE): [`Removals::row_filter`] (`deleteByRowFilter`) removes a
//!   live file, data or delete, when every row matches (`plan::strict`'s
//!   residual and strict metrics evaluator, bound to the attempt's schema);
//!   a data file some, but not all, of whose rows may match is
//!   [`FilterError::Partial`] with Java's "Cannot delete file where some, but
//!   not all, rows match filter %s: %s" (`:478-480`; the commit loop raises
//!   it as `KRISHIV_ICEBERG_OVERWRITE_PARTIAL_FILE`), a delete file so
//!   matched is kept (`:554`). [`Removals::drop_partitions`]
//!   (`dropPartition`) removes every live file of a spec id and tuple.
//!
//! Divergences: every manifest with live files is read (Java first prunes by
//! the partitions of the files to delete, `canContainDroppedFiles`, and by
//! the filter's manifest evaluator; the result is the same, only the reads
//! differ); the separator of the missing-files list (`, `) is unverified, and
//! a missing DV is named with the data file it applies to (Krishiv's text:
//! one Puffin file holds many DVs); Java prints its `Expression`, Krishiv
//! `plan::strict::Pred`'s recalled rendering of it.

use std::collections::{BTreeSet, HashMap, HashSet};

use iceberg::io::FileIO;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Manifest, ManifestContentType,
    ManifestFile, ManifestStatus, TableMetadata, UNASSIGNED_SEQUENCE_NUMBER,
};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_io::PushdownExpr;

use super::merge::{manifest_writer, rebuild, with_first_row_id};
use super::scan::{content_file, partition_specs, read_capped, schema_fields};
use super::storage::from_iceberg;
use crate::plan::PartitionValue;
use crate::plan::bounds::decode;
use crate::plan::metrics::WriterMetrics;
use crate::plan::strict::{Pred, RowFilter};
use crate::plan::types::{IcebergType, PartitionSpecs};

/// The files an `OverwriteFiles` or `ReplacePartitions` removes: data files
/// by location, delete files per entry ([`DeleteKey`]), every file whose
/// rows all match a row filter, every file of a dropped partition.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Removals {
    /// Data files (Java's `deleteFile(DataFile)`).
    pub(crate) data_files: BTreeSet<String>,
    /// Delete files (Java's `deleteFiles(_, DeleteFileSet)`: CoW's DVs of
    /// the files it rewrites).
    pub(crate) delete_files: BTreeSet<DeleteKey>,
    /// Java's `deleteByRowFilter` (an `overwriteByRowFilter`), in Spark's
    /// semantics (`plan::strict::Pred::from_spark` turns it into Iceberg's):
    /// a file is removed when every row matches; a data file where some, but
    /// not all, rows may match fails the commit. `Constant(false)` is none.
    pub(crate) row_filter: Option<PushdownExpr>,
    /// Java's `dropPartition`: every live file (data or delete) whose spec id
    /// and partition tuple are one of these (`ReplacePartitions`).
    pub(crate) drop_partitions: HashSet<PartitionKey>,
    /// A `ReplacePartitions` (operation `overwrite` always, summary
    /// `replace-partitions=true`).
    pub(crate) replace_partitions: bool,
    /// A delete file of [`Self::delete_files`] no longer live is skipped, not
    /// "Missing required files to delete": a merge-on-read `DELETE`'s
    /// `RowDelta` without `validateDeletedFiles` (Java checks
    /// `failMissingDeletePaths` only when it is set, `BaseRowDelta.java:151-153`).
    pub(crate) ignore_missing_deletes: bool,
}

/// A partition of a spec: its id and the tuple, each value as
/// [`partition_key`] normalises it.
pub(crate) type PartitionKey = (i32, Vec<PartitionValue>);

/// The key of the partition `partition` of the spec `spec_id`, its values
/// compared by value as Java's `PartitionSet` (`StructLikeWrapper` at the
/// spec's current partition type) does, not by stored bytes (review 36b2b
/// M3): after `int -> long` or `float -> double` an old manifest holds a
/// 4-byte tuple and a new file an 8-byte one for the same partition. Each
/// value is decoded with the field's result type in `specs` (by stored
/// width) and written at the widest width of its family (ints and longs as
/// 8-byte longs, floats and doubles as 8-byte doubles); a value the type
/// does not decode keeps its bytes (an 8-byte long under a type still
/// `int` is already in that form). `None` when a value is unreadable: such
/// a key never matches.
pub(crate) fn partition_key(
    specs: &PartitionSpecs,
    spec_id: i32,
    partition: &[PartitionValue],
) -> Option<PartitionKey> {
    use krishiv_io::ScalarValue as S;
    let fields = specs.get(&spec_id).map(|s| s.fields.as_slice());
    let values = partition
        .iter()
        .enumerate()
        .map(|(i, v)| match v {
            PartitionValue::Null => Some(PartitionValue::Null),
            PartitionValue::Unreadable => None,
            PartitionValue::Value(b) => {
                let ty = fields
                    .and_then(|f| f.get(i))
                    .and_then(|f| f.result_type.as_ref());
                let wide = match ty.and_then(|t| decode(b, t)) {
                    Some(S::Int32(x)) if ty == Some(&IcebergType::Int) => {
                        i64::from(x).to_le_bytes().to_vec()
                    }
                    Some(S::Int64(x) | S::TimestampMicros(x)) => x.to_le_bytes().to_vec(),
                    Some(S::Float32(x)) => f64::from(x).to_le_bytes().to_vec(),
                    Some(S::Float64(x)) => x.to_le_bytes().to_vec(),
                    _ => b.clone(),
                };
                Some(PartitionValue::Value(wide))
            }
        })
        .collect::<Option<Vec<_>>>()?;
    Some((spec_id, values))
}

impl Removals {
    /// Whether any data file is removed (Java's `deletesDataFiles`,
    /// `ManifestFilterManager.containsDeletes`: paths, a row filter other
    /// than `alwaysFalse`, or dropped partitions).
    pub(crate) fn deletes_data_files(&self) -> bool {
        !self.data_files.is_empty() || self.filter().is_some() || !self.drop_partitions.is_empty()
    }

    /// The row filter, `None` for none or `alwaysFalse`.
    fn filter(&self) -> Option<&PushdownExpr> {
        self.row_filter
            .as_ref()
            .filter(|f| **f != PushdownExpr::Constant(false))
    }
}

/// One delete file to remove, as Java's `DeleteFileSet` tells them apart:
/// many DVs share one Puffin file (one entry per blob, each with its own
/// referenced data file and content offset), so a location alone would
/// remove the DVs of files the write never touched (review 36b2a B1).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct DeleteKey {
    /// The file's location.
    pub(crate) path: String,
    /// The data file a DV or file-scoped position delete applies to.
    pub(crate) referenced_data_file: Option<String>,
    /// A DV's blob offset in its Puffin file.
    pub(crate) content_offset: Option<i64>,
    /// A DV's blob length (Java's `DeleteFileSet` tells DVs apart by location,
    /// offset and length; recalled, unverified).
    pub(crate) content_size: Option<i64>,
}

impl DeleteKey {
    /// The key of the manifest entry `f`.
    pub(crate) fn of(f: &DataFile) -> Self {
        Self {
            path: f.file_path().to_owned(),
            referenced_data_file: f.referenced_data_file(),
            content_offset: f.content_offset(),
            content_size: f.content_size_in_bytes(),
        }
    }

    /// The key as the missing-files message names it: a DV with the data
    /// file it applies to (one Puffin file holds many).
    fn describe(&self) -> String {
        match (&self.referenced_data_file, self.content_offset) {
            (Some(r), Some(_)) => format!("{} (the deletion vector of {r})", self.path),
            _ => self.path.clone(),
        }
    }
}

/// What a filter writes through: the table, the write's names, and the
/// rewritten manifests written so far.
pub(crate) struct FilterContext<'a> {
    /// The table's file access.
    pub(crate) io: &'a FileIO,
    /// The metadata of the attempt (the base the removals apply to).
    pub(crate) metadata: &'a TableMetadata,
    /// The write's snapshot id.
    pub(crate) snapshot_id: i64,
    /// `<metadata dir>/<commit uuid>-m`, the rewritten manifests' prefix.
    pub(crate) prefix: String,
    /// The number of the next manifest written.
    pub(crate) next: usize,
    /// Every manifest path written (pushed before the write, so a failed
    /// write's partial file is cleaned up with the others).
    pub(crate) written: &'a mut Vec<String>,
    /// `krishiv.catalog.maxMetadataBytes` for each manifest read.
    pub(crate) max_metadata_bytes: u64,
}

/// `f` with its partition tuple at the spec's current partition type
/// (review 36b2b M3): a manifest written before `int -> long` or `float ->
/// double` holds the old literal, which iceberg-rust refuses to write under
/// the promoted type when the manifest is rewritten or merged.
pub(super) fn widen_partition(
    f: DataFile,
    metadata: &TableMetadata,
    spec_id: i32,
) -> Result<DataFile, KrishivError> {
    use iceberg::spec::{Literal, PrimitiveLiteral, PrimitiveType, Type};
    let Some(spec) = metadata.partition_spec_by_id(spec_id) else {
        return Ok(f);
    };
    let Ok(ty) = spec.partition_type(metadata.current_schema()) else {
        return Ok(f);
    };
    if ty.fields().len() != f.partition().fields().len() {
        return Ok(f);
    }
    let mut changed = false;
    let values: Vec<Option<Literal>> = f
        .partition()
        .iter()
        .zip(ty.fields())
        .map(|(v, field)| match (v, field.field_type.as_ref()) {
            (
                Some(Literal::Primitive(PrimitiveLiteral::Int(x))),
                Type::Primitive(PrimitiveType::Long),
            ) => {
                changed = true;
                Some(Literal::long(i64::from(*x)))
            }
            (
                Some(Literal::Primitive(PrimitiveLiteral::Float(x))),
                Type::Primitive(PrimitiveType::Double),
            ) => {
                changed = true;
                Some(Literal::double(f64::from(x.0)))
            }
            (v, _) => v.cloned(),
        })
        .collect();
    if !changed {
        return Ok(f);
    }
    rebuild(&f, f.first_row_id(), values.into_iter().collect(), spec_id)
}

/// Why a filter failed.
#[derive(Debug)]
pub(crate) enum FilterError {
    /// Files to remove that no manifest holds live: Java's message.
    Missing(String),
    /// A data file some, but not all, of whose rows match the row filter
    /// (Java's `ValidationException`): the filter's text and the file.
    Partial {
        /// The filter as Java prints it.
        filter: String,
        /// The file's location.
        file: String,
    },
    /// Anything else.
    Error(KrishivError),
}

impl FilterError {
    /// Java's text of a [`FilterError::Partial`]
    /// (`ManifestFilterManager.java:478-480`).
    pub(crate) fn partial_text(filter: &str, file: &str) -> String {
        format!("Cannot delete file where some, but not all, rows match filter {filter}: {file}")
    }
}

impl From<KrishivError> for FilterError {
    fn from(e: KrishivError) -> Self {
        Self::Error(e)
    }
}

/// The base's manifests with the removals applied, and what was removed.
#[derive(Debug)]
pub(crate) struct Filtered {
    /// The base's manifests in their order, each hit one replaced by its
    /// rewrite.
    pub(crate) manifests: Vec<ManifestFile>,
    /// Manifests rewritten (`manifests-replaced`).
    pub(crate) replaced: usize,
    /// The summary of the removed files.
    pub(crate) removed: Removed,
}

/// Java's `SnapshotSummary.UpdateMetrics`, both sides.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Metrics {
    added_size: u64,
    removed_size: u64,
    added_files: u64,
    removed_files: u64,
    added_eq_delete_files: u64,
    removed_eq_delete_files: u64,
    added_pos_delete_files: u64,
    removed_pos_delete_files: u64,
    added_dvs: u64,
    removed_dvs: u64,
    added_delete_files: u64,
    removed_delete_files: u64,
    added_records: u64,
    deleted_records: u64,
    added_pos_deletes: u64,
    removed_pos_deletes: u64,
    added_eq_deletes: u64,
    removed_eq_deletes: u64,
}

impl Metrics {
    /// Java's `addedFile` for a delete file (`SnapshotSummary.java:290-310`:
    /// a DV is `added-dvs`, sized by its blob, `ScanTaskUtil
    /// .contentSizeInBytes`).
    fn add_delete(&mut self, f: &DataFile) {
        let dv = is_dv(f);
        self.added_size += if dv {
            f.content_size_in_bytes()
                .and_then(|n| u64::try_from(n).ok())
                .unwrap_or(0)
        } else {
            f.file_size_in_bytes()
        };
        match f.content_type() {
            DataContentType::Data => {
                self.added_files += 1;
                self.added_records += f.record_count();
            }
            DataContentType::PositionDeletes => {
                if dv {
                    self.added_dvs += 1;
                } else {
                    self.added_pos_delete_files += 1;
                }
                self.added_delete_files += 1;
                self.added_pos_deletes += f.record_count();
            }
            DataContentType::EqualityDeletes => {
                self.added_delete_files += 1;
                self.added_eq_delete_files += 1;
                self.added_eq_deletes += f.record_count();
            }
        }
    }

    /// Java's `removedFile`.
    fn remove(&mut self, f: &DataFile) {
        let dv = is_dv(f);
        let size = if dv {
            f.content_size_in_bytes()
                .and_then(|n| u64::try_from(n).ok())
                .unwrap_or(0)
        } else {
            f.file_size_in_bytes()
        };
        self.removed_size += size;
        match f.content_type() {
            DataContentType::Data => {
                self.removed_files += 1;
                self.deleted_records += f.record_count();
            }
            DataContentType::PositionDeletes => {
                if dv {
                    self.removed_dvs += 1;
                } else {
                    self.removed_pos_delete_files += 1;
                }
                self.removed_delete_files += 1;
                self.removed_pos_deletes += f.record_count();
            }
            DataContentType::EqualityDeletes => {
                self.removed_delete_files += 1;
                self.removed_eq_delete_files += 1;
                self.removed_eq_deletes += f.record_count();
            }
        }
    }

    fn merge(&mut self, o: &Self) {
        self.added_size += o.added_size;
        self.removed_size += o.removed_size;
        self.added_files += o.added_files;
        self.removed_files += o.removed_files;
        self.added_eq_delete_files += o.added_eq_delete_files;
        self.removed_eq_delete_files += o.removed_eq_delete_files;
        self.added_pos_delete_files += o.added_pos_delete_files;
        self.removed_pos_delete_files += o.removed_pos_delete_files;
        self.added_dvs += o.added_dvs;
        self.removed_dvs += o.removed_dvs;
        self.added_delete_files += o.added_delete_files;
        self.removed_delete_files += o.removed_delete_files;
        self.added_records += o.added_records;
        self.deleted_records += o.deleted_records;
        self.added_pos_deletes += o.added_pos_deletes;
        self.removed_pos_deletes += o.removed_pos_deletes;
        self.added_eq_deletes += o.added_eq_deletes;
        self.removed_eq_deletes += o.removed_eq_deletes;
    }

    /// Java's `addTo` order: each key and its value.
    fn pairs(&self) -> [(&'static str, u64); 18] {
        [
            ("added-data-files", self.added_files),
            ("deleted-data-files", self.removed_files),
            ("added-equality-delete-files", self.added_eq_delete_files),
            (
                "removed-equality-delete-files",
                self.removed_eq_delete_files,
            ),
            ("added-position-delete-files", self.added_pos_delete_files),
            (
                "removed-position-delete-files",
                self.removed_pos_delete_files,
            ),
            ("added-delete-files", self.added_delete_files),
            ("removed-delete-files", self.removed_delete_files),
            ("added-dvs", self.added_dvs),
            ("removed-dvs", self.removed_dvs),
            ("added-records", self.added_records),
            ("deleted-records", self.deleted_records),
            ("added-files-size", self.added_size),
            ("removed-files-size", self.removed_size),
            ("added-position-deletes", self.added_pos_deletes),
            ("removed-position-deletes", self.removed_pos_deletes),
            ("added-equality-deletes", self.added_eq_deletes),
            ("removed-equality-deletes", self.removed_eq_deletes),
        ]
    }

    /// The values of `text` (`k=v,k=v`, a partition summary) added in.
    fn add_text(&mut self, text: &str) {
        for pair in text.split(',') {
            let Some((k, v)) = pair.split_once('=') else {
                continue;
            };
            let Ok(v) = v.parse::<u64>() else { continue };
            let slot = match k {
                "added-data-files" => &mut self.added_files,
                "deleted-data-files" => &mut self.removed_files,
                "added-equality-delete-files" => &mut self.added_eq_delete_files,
                "removed-equality-delete-files" => &mut self.removed_eq_delete_files,
                "added-position-delete-files" => &mut self.added_pos_delete_files,
                "removed-position-delete-files" => &mut self.removed_pos_delete_files,
                "added-delete-files" => &mut self.added_delete_files,
                "removed-delete-files" => &mut self.removed_delete_files,
                "added-dvs" => &mut self.added_dvs,
                "removed-dvs" => &mut self.removed_dvs,
                "added-records" => &mut self.added_records,
                "deleted-records" => &mut self.deleted_records,
                "added-files-size" => &mut self.added_size,
                "removed-files-size" => &mut self.removed_size,
                "added-position-deletes" => &mut self.added_pos_deletes,
                "removed-position-deletes" => &mut self.removed_pos_deletes,
                "added-equality-deletes" => &mut self.added_eq_deletes,
                "removed-equality-deletes" => &mut self.removed_eq_deletes,
                _ => continue,
            };
            *slot += v;
        }
    }

    /// Java's `partitionSummary`: the positive values, `k=v` joined by `,`.
    fn text(&self) -> String {
        self.pairs()
            .iter()
            .filter(|(_, v)| *v > 0)
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// The files a filter removed, for the snapshot summary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Removed {
    /// The table-wide values.
    pub(crate) metrics: Metrics,
    /// Per partition path (Java's `partitionToPath`; `""` unpartitioned).
    pub(crate) partitions: HashMap<String, Metrics>,
    /// Paths removed a second time from one manifest.
    pub(crate) duplicates: u64,
}

/// Whether `f` is a deletion vector (Java's `ContentFileUtil.isDV`: a
/// Puffin file).
fn is_dv(f: &DataFile) -> bool {
    f.content_type() == DataContentType::PositionDeletes
        && f.file_format() == DataFileFormat::Puffin
}

/// What one manifest's filter removed.
struct Hit {
    /// The rewritten manifest.
    manifest: ManifestFile,
    /// The files removed, each once.
    removed: Vec<DataFile>,
    /// Paths removed a second time.
    duplicates: u64,
}

/// The removals `removals` applied to `base` (an attempt's base snapshot's
/// manifests, in list order; module docs).
///
/// # Errors
/// [`FilterError::Missing`] for a path no manifest holds live; otherwise
/// `KRISHIV_UNSUPPORTED_FEATURE` for an encrypted manifest to rewrite,
/// `KRISHIV_CATALOG_LIMIT_EXCEEDED` for one over the parse limit, an internal
/// error for an entry whose sequence numbers a rewrite needs are unknown,
/// iceberg-rust's read, parse and write failures.
pub(crate) async fn filter_manifests(
    cx: &mut FilterContext<'_>,
    base: Vec<ManifestFile>,
    removals: &Removals,
) -> Result<Filtered, FilterError> {
    let mut manifests = base;
    let mut replaced = 0;
    let mut removed = Removed::default();
    let mut found: HashSet<String> = HashSet::new();
    let mut found_deletes: HashSet<DeleteKey> = HashSet::new();
    let schema = std::sync::Arc::clone(cx.metadata.current_schema());
    let record = |spec_id: i32, hit: &Hit, removed: &mut Removed| {
        let spec = cx.metadata.partition_spec_by_id(spec_id);
        for f in &hit.removed {
            removed.metrics.remove(f);
            let path = spec.map_or_else(String::new, |s| {
                s.partition_to_path(f.partition(), std::sync::Arc::clone(&schema))
            });
            removed.partitions.entry(path).or_default().remove(f);
        }
        removed.duplicates += hit.duplicates;
    };
    // The row filter, bound to this attempt's schema and specs (Java binds
    // it on every `apply`).
    let fields = schema_fields(&schema);
    let specs = partition_specs(cx.metadata, &schema);
    let metrics = WriterMetrics::from_properties(
        cx.metadata
            .properties()
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str())),
        &fields,
    );
    let pred = removals.filter().map(Pred::from_spark);
    let text = pred.as_ref().map(ToString::to_string).unwrap_or_default();
    let row_filter = pred.map(|p| RowFilter::new(p, &fields, &specs).with_metrics(&metrics));
    // Java's per-file decision (`ManifestFilterManager.java:461-490`): marked
    // by path or partition, or every row matching the filter; some rows
    // matching is an error in a data manifest and a keep in a delete one.
    let by_filter = |f: &DataFile, spec_id: i32, data: bool| -> Result<bool, FilterError> {
        if removals.drop_partitions.is_empty() && row_filter.is_none() {
            return Ok(false);
        }
        // One file's metrics, built once for both decisions.
        let file = content_file(f, spec_id);
        let marked = !removals.drop_partitions.is_empty()
            && partition_key(&specs, spec_id, &file.partition)
                .is_some_and(|k| removals.drop_partitions.contains(&k));
        if marked {
            return Ok(true);
        }
        let Some(filter) = &row_filter else {
            return Ok(false);
        };
        if !filter.rows_might_match(&file) {
            return Ok(false);
        }
        if filter.rows_must_match(&file) {
            return Ok(true);
        }
        if data {
            return Err(FilterError::Partial {
                filter: text.clone(),
                file: f.file_path().to_owned(),
            });
        }
        Ok(false)
    };
    let by_rows = row_filter.is_some() || !removals.drop_partitions.is_empty();
    // Data manifests first: the data files removed decide the dangling DVs.
    let mut removed_data: HashSet<String> = HashSet::new();
    for m in &mut manifests {
        if m.content != ManifestContentType::Data || !(by_rows || !removals.data_files.is_empty()) {
            continue;
        }
        let spec_id = m.partition_spec_id;
        let hit = rewrite(cx, m, |f, _| {
            Ok(removals.data_files.contains(f.file_path()) || by_filter(f, spec_id, true)?)
        })
        .await?;
        if let Some(hit) = hit {
            for f in &hit.removed {
                found.insert(f.file_path().to_owned());
                removed_data.insert(f.file_path().to_owned());
            }
            record(m.partition_spec_id, &hit, &mut removed);
            *m = hit.manifest;
            replaced += 1;
        }
    }
    // `dropDeleteFilesOlderThan`: the smallest assigned data sequence number
    // of the filtered data manifests, from the table's last.
    let min_sequence = manifests
        .iter()
        .filter(|m| m.content == ManifestContentType::Data)
        .map(|m| m.min_sequence_number)
        .filter(|s| *s != UNASSIGNED_SEQUENCE_NUMBER)
        .fold(cx.metadata.last_sequence_number(), i64::min);
    if cx.metadata.format_version() != FormatVersion::V1 {
        for m in &mut manifests {
            if m.content != ManifestContentType::Deletes {
                continue;
            }
            let spec_id = m.partition_spec_id;
            let hit = rewrite(cx, m, |f, data_sequence| {
                Ok(removals.delete_files.contains(&DeleteKey::of(f))
                    || (data_sequence > 0 && data_sequence < min_sequence)
                    || (is_dv(f)
                        && f.referenced_data_file()
                            .is_some_and(|r| removed_data.contains(&r)))
                    || (by_rows && by_filter(f, spec_id, false)?))
            })
            .await?;
            if let Some(hit) = hit {
                for f in &hit.removed {
                    found_deletes.insert(DeleteKey::of(f));
                }
                record(m.partition_spec_id, &hit, &mut removed);
                *m = hit.manifest;
                replaced += 1;
            }
        }
    }
    let missing: Vec<String> = removals
        .data_files
        .iter()
        .filter(|p| !found.contains(p.as_str()))
        .cloned()
        .chain(
            removals
                .delete_files
                .iter()
                .filter(|k| !removals.ignore_missing_deletes && !found_deletes.contains(*k))
                .map(DeleteKey::describe),
        )
        .collect();
    if !missing.is_empty() {
        return Err(FilterError::Missing(format!(
            "Missing required files to delete: {}",
            missing.join(", ")
        )));
    }
    Ok(Filtered {
        manifests,
        replaced,
        removed,
    })
}

/// `m` rewritten without the live entries `hit` (given the file and its
/// data sequence number) marks, or `None` when none is marked (module docs).
async fn rewrite(
    cx: &mut FilterContext<'_>,
    m: &ManifestFile,
    hit: impl Fn(&DataFile, i64) -> Result<bool, FilterError>,
) -> Result<Option<Hit>, FilterError> {
    if !m.has_added_files() && !m.has_existing_files() {
        return Ok(None);
    }
    if m.key_metadata.is_some() {
        return Err(KrishivError::from(KrishivUnsupportedFeature {
            feature: "removing files from encrypted Iceberg manifests".to_owned(),
        })
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
    let lineage =
        cx.metadata.format_version() == FormatVersion::V3 && m.content == ManifestContentType::Data;
    // `plan::entry::inherit`'s rules: sequence numbers inherited by `ADDED`
    // entries (and every entry of a sequence-number-0 manifest), row ids
    // assigned in order to the live data entries without one.
    let mut next_row_id = match (lineage, m.first_row_id) {
        (true, Some(id)) => Some(i64::try_from(id).map_err(|_| {
            internal(format!(
                "first_row_id {id} of manifest {} exceeds the long range",
                m.manifest_path
            ))
        })?),
        _ => None,
    };
    struct Live {
        index: usize,
        snapshot_id: i64,
        data_seq: Option<i64>,
        file_seq: Option<i64>,
        row_id: Option<i64>,
        marked: bool,
    }
    let mut live = Vec::new();
    for (index, e) in entries.iter().enumerate() {
        let status = e.status();
        if status == ManifestStatus::Deleted {
            continue;
        }
        let inherits = status == ManifestStatus::Added || m.sequence_number == 0;
        let seq = |v: Option<i64>| v.or(inherits.then_some(m.sequence_number));
        let data_seq = seq(e.sequence_number());
        let file = e.data_file();
        let row_id = match &mut next_row_id {
            Some(next) if file.first_row_id().is_none() => {
                let id = *next;
                *next = next
                    .checked_add_unsigned(file.record_count())
                    .ok_or_else(|| {
                        internal(format!("row id overflow in manifest {}", m.manifest_path))
                    })?;
                Some(id)
            }
            // An explicit id always wins, also in a manifest without a
            // `first_row_id`.
            _ => file.first_row_id(),
        };
        live.push(Live {
            index,
            snapshot_id: e.snapshot_id().unwrap_or(m.added_snapshot_id),
            data_seq,
            file_seq: seq(e.file_sequence_number),
            row_id,
            marked: hit(file, data_seq.unwrap_or(-1))?,
        });
    }
    if !live.iter().any(|l| l.marked) {
        return Ok(None);
    }
    let path = format!("{}{}.avro", cx.prefix, cx.next);
    cx.next += 1;
    cx.written.push(path.clone());
    let mut writer = manifest_writer(
        cx.io,
        &path,
        cx.metadata,
        cx.snapshot_id,
        m.partition_spec_id,
        m.content,
    )?;
    let mut removed: Vec<DataFile> = Vec::new();
    // A DV's identity is its blob (one Puffin file holds many), as `DeleteKey`; a data file's is
    // its path (its other fields are `None`).
    let mut seen: HashSet<DeleteKey> = HashSet::new();
    let mut duplicates = 0;
    for l in &live {
        let Some(e) = entries.get(l.index) else {
            continue;
        };
        let file = e.data_file();
        let explicit = |v: Option<i64>, what: &str| {
            v.filter(|n| *n >= 0).ok_or_else(|| {
                internal(format!(
                    "the {what} sequence number of {} in manifest {} is unknown, so the \
                     manifest cannot be rewritten",
                    file.file_path(),
                    m.manifest_path
                ))
            })
        };
        let file = if lineage && file.content_type() == DataContentType::Data {
            with_first_row_id(file, l.row_id, m.partition_spec_id)?
        } else {
            file.clone()
        };
        let file = widen_partition(file, cx.metadata, m.partition_spec_id)?;
        let data_seq = explicit(l.data_seq, "data")?;
        let file_seq = explicit(l.file_seq, "file")?;
        if l.marked {
            if seen.insert(DeleteKey::of(&file)) {
                removed.push(file.clone());
            } else {
                duplicates += 1;
            }
            writer
                .add_delete_file(file, data_seq, Some(file_seq))
                .map_err(from_iceberg)?;
        } else {
            writer
                .add_existing_file(file, l.snapshot_id, data_seq, Some(file_seq))
                .map_err(from_iceberg)?;
        }
    }
    let manifest = writer.write_manifest_file().await.map_err(from_iceberg)?;
    Ok(Some(Hit {
        manifest,
        removed,
        duplicates,
    }))
}

/// The summary of an overwrite: `added` (the added files' values, as
/// `write_new_manifests` built them, whose changed partitions are
/// `changed`) with `removed`'s values, the changed partitions of both
/// counted once, and Java's partition summaries when they fit
/// `limit` (`write.summary.partition-limit`, `SnapshotSummary.Builder.build`).
pub(crate) fn overwrite_summary(
    added: &HashMap<String, String>,
    changed: &HashSet<String>,
    removed: &Removed,
    limit: u64,
) -> HashMap<String, String> {
    let mut props: HashMap<String, String> = added
        .iter()
        .filter(|(k, _)| !k.starts_with("partitions.") && *k != "partition-summaries-included")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (k, v) in removed.metrics.pairs() {
        if v > 0 && !k.starts_with("added-") {
            props.insert(k.to_owned(), v.to_string());
        }
    }
    if removed.duplicates > 0 {
        props.insert(
            "deleted-duplicate-files".to_owned(),
            removed.duplicates.to_string(),
        );
    }
    let all: BTreeSet<&String> = changed.iter().chain(removed.partitions.keys()).collect();
    props.insert("changed-partition-count".to_owned(), all.len().to_string());
    if u64::try_from(all.len()).unwrap_or(u64::MAX) <= limit {
        if !all.is_empty() {
            props.insert("partition-summaries-included".to_owned(), "true".to_owned());
        }
        for key in all {
            if key.is_empty() {
                continue;
            }
            let mut m = Metrics::default();
            if let Some(text) = added.get(&format!("partitions.{key}")) {
                m.add_text(text);
            }
            if let Some(r) = removed.partitions.get(key) {
                m.merge(r);
            }
            props.insert(format!("partitions.{key}"), m.text());
        }
    }
    props
}

/// The summary values of added delete files (`files`, each with its
/// partition path, `""` unpartitioned): the totals and each partition's
/// `partitions.<path>` text, as Java's `SnapshotSummary.addedFile` counts
/// them (a row delta's, PR-36c).
pub(crate) fn add_delete_files(out: &mut HashMap<String, String>, files: &[(String, &DataFile)]) {
    let mut total = Metrics::default();
    let mut parts: HashMap<&str, Metrics> = HashMap::new();
    for (path, f) in files {
        total.add_delete(f);
        if !path.is_empty() {
            parts.entry(path.as_str()).or_default().add_delete(f);
        }
    }
    for (k, v) in total.pairs() {
        if v > 0 {
            out.insert(k.to_owned(), v.to_string());
        }
    }
    for (path, m) in parts {
        out.insert(format!("partitions.{path}"), m.text());
    }
}

/// `extra`'s values added into `added`: counters summed, partition texts
/// merged (their keys in Java's order).
pub(crate) fn merge_summary(added: &mut HashMap<String, String>, extra: &HashMap<String, String>) {
    for (k, v) in extra {
        if k.starts_with("partitions.") {
            let mut m = Metrics::default();
            if let Some(old) = added.get(k) {
                m.add_text(old);
            }
            m.add_text(v);
            added.insert(k.clone(), m.text());
            continue;
        }
        let sum = added
            .get(k)
            .and_then(|a| a.parse::<u64>().ok())
            .unwrap_or(0)
            .saturating_add(v.parse::<u64>().unwrap_or(0));
        added.insert(k.clone(), sum.to_string());
    }
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}
