// SPDX-License-Identifier: Apache-2.0
//! Which delete files apply to which data file (P3-T026, PR-25b; design note
//! §3.2 with the fact check's S1 and review-1's M1), as Java's
//! `DeleteFileIndex` (apache-iceberg 1.12.0) decides it:
//!
//! | Delete | Applies to a data file when |
//! |---|---|
//! | Deletion vector | its `referenced_data_file` is the data file's path; its data sequence number must be `>=` the file's (else the table is corrupt) |
//! | Position delete, file-scoped | `referenced_data_file`, or equal `file_path` lower and upper bounds, name the data file, and `delete.seq >= data.seq` |
//! | Position delete, partition-scoped | same spec id and partition tuple, and `delete.seq >= data.seq` (never table-wide: M1) |
//! | Equality delete | `delete.seq > data.seq`, and the delete's spec is unpartitioned (global) or it has the data file's spec and tuple |
//!
//! A data file with a deletion vector takes no position delete file: the DV
//! replaces them (FC S1). Two DVs for one data file are an error. Equality
//! deletes are assigned here and applied by PR-25c.
//!
//! Errors carry Java's `ValidationException` texts (class
//! `KRISHIV_ICEBERG_INVALID_DELETES`).

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::{
    KrishivError, KrishivIcebergInvalidDeletes, KrishivUnsupportedFeature,
};

use super::entry::{DataFileEntry, DeleteFileEntry, DeleteKind, PartitionValue};
use super::types::PartitionSpecs;

/// The field id of a position delete file's `file_path` column (reserved).
pub const DELETE_FILE_PATH_ID: i32 = i32::MAX - 101;
/// The field id of a position delete file's `pos` column (reserved).
pub const DELETE_FILE_POS_ID: i32 = i32::MAX - 102;

/// The error of an Iceberg table whose deletes cannot be applied, with Java's text.
pub(crate) fn invalid(message: String) -> KrishivError {
    KrishivIcebergInvalidDeletes { message }.into()
}

/// The delete files that apply to one data file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskDeletes {
    /// The file's deletion vector (when set, `position_deletes` is empty).
    pub deletion_vector: Option<Arc<DeleteFileEntry>>,
    /// Position delete files, in manifest order.
    pub position_deletes: Vec<Arc<DeleteFileEntry>>,
    /// Equality delete files, in manifest order.
    pub equality_deletes: Vec<Arc<DeleteFileEntry>>,
}

impl TaskDeletes {
    /// Whether nothing applies.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.deletion_vector.is_none()
            && self.position_deletes.is_empty()
            && self.equality_deletes.is_empty()
    }
}

/// A delete file with its data sequence number.
type Seqd = (i64, Arc<DeleteFileEntry>);

/// Deletes keyed by partition tuple within one spec. A tuple holding a value
/// the reader could not serialise cannot be keyed: such deletes are kept per
/// spec and offered to every data file of the spec (a superset; see
/// [`DeleteIndex::for_data_file`]).
#[derive(Debug, Default)]
struct ByPartition {
    tuples: HashMap<Vec<PartitionValue>, Vec<Seqd>>,
    unkeyed: Vec<Seqd>,
}

impl ByPartition {
    fn add(&mut self, partition: &[PartitionValue], d: Seqd) {
        if readable(partition) {
            self.tuples.entry(partition.to_vec()).or_default().push(d);
        } else {
            self.unkeyed.push(d);
        }
    }

    /// The deletes whose tuple may equal `partition`.
    fn candidates<'a>(&'a self, partition: &'a [PartitionValue]) -> Vec<&'a Seqd> {
        let mut out: Vec<&Seqd> = if readable(partition) {
            self.tuples.get(partition).into_iter().flatten().collect()
        } else {
            self.tuples.values().flatten().collect()
        };
        out.extend(self.unkeyed.iter());
        out
    }
}

fn readable(partition: &[PartitionValue]) -> bool {
    !partition.contains(&PartitionValue::Unreadable)
}

/// The live delete files of a snapshot, indexed for assignment.
#[derive(Debug, Default)]
pub struct DeleteIndex {
    dvs: HashMap<String, Seqd>,
    position_by_path: HashMap<String, Vec<Seqd>>,
    position_by_partition: HashMap<i32, ByPartition>,
    equality_global: Vec<Seqd>,
    equality_by_partition: HashMap<i32, ByPartition>,
    empty: bool,
}

/// Java's `ContentFileUtil.dvDesc`.
fn dv_desc(d: &DeleteFileEntry) -> String {
    let f = &d.entry.file;
    let opt = |v: Option<i64>| v.map_or_else(|| "null".to_owned(), |v| v.to_string());
    format!(
        "DV{{location={}, offset={}, length={}, referencedDataFile={}}}",
        f.path,
        opt(f.content_offset),
        opt(f.content_size_in_bytes),
        f.referenced_data_file.as_deref().unwrap_or("null")
    )
}

/// The data file a position delete file names: `referenced_data_file`, or
/// its `file_path` bounds when they are equal (Java's
/// `ContentFileUtil.referencedDataFile`).
fn referenced_data_file(d: &DeleteFileEntry) -> Option<String> {
    let f = &d.entry.file;
    if let Some(p) = &f.referenced_data_file {
        return Some(p.clone());
    }
    let lower = f.lower_bounds.get(&DELETE_FILE_PATH_ID)?;
    let upper = f.upper_bounds.get(&DELETE_FILE_PATH_ID)?;
    (lower == upper)
        .then(|| String::from_utf8(lower.clone()).ok())
        .flatten()
}

impl DeleteIndex {
    /// The index of `files` (a snapshot's live delete files).
    ///
    /// # Errors
    /// `KRISHIV_ICEBERG_INVALID_DELETES` for two DVs of one data file (Java's
    /// text), a DV without a referenced data file, a delete file without a
    /// data sequence number or with a spec id the table does not have.
    pub fn new(
        files: &[Arc<DeleteFileEntry>],
        specs: &PartitionSpecs,
    ) -> Result<Self, KrishivError> {
        let mut index = Self {
            empty: files.is_empty(),
            ..Self::default()
        };
        for d in files {
            let f = &d.entry.file;
            let seq = d.entry.data_sequence_number.ok_or_else(|| {
                invalid(format!(
                    "Delete file {} has no data sequence number: the table metadata is corrupted",
                    f.path
                ))
            })?;
            let entry = (seq, Arc::clone(d));
            match d.kind {
                DeleteKind::DeletionVector => {
                    let path = f.referenced_data_file.clone().ok_or_else(|| {
                        invalid(format!(
                            "Deletion vector {} has no referenced data file",
                            f.path
                        ))
                    })?;
                    if let Some((_, existing)) = index.dvs.get(&path) {
                        return Err(invalid(format!(
                            "Can't index multiple DVs for {path}: {} and {}",
                            dv_desc(d),
                            dv_desc(existing)
                        )));
                    }
                    index.dvs.insert(path, entry);
                }
                DeleteKind::PositionDelete => match referenced_data_file(d) {
                    Some(path) => index.position_by_path.entry(path).or_default().push(entry),
                    None => index
                        .position_by_partition
                        .entry(f.spec_id)
                        .or_default()
                        .add(&f.partition, entry),
                },
                DeleteKind::EqualityDelete => {
                    let spec = specs.get(&f.spec_id).ok_or_else(|| {
                        invalid(format!(
                            "Delete file {} has the unknown partition spec {}",
                            f.path, f.spec_id
                        ))
                    })?;
                    if spec.is_unpartitioned() {
                        index.equality_global.push(entry);
                    } else {
                        index
                            .equality_by_partition
                            .entry(f.spec_id)
                            .or_default()
                            .add(&f.partition, entry);
                    }
                }
            }
        }
        Ok(index)
    }

    /// Whether the snapshot has no live delete file.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.empty
    }

    /// Every live equality delete file.
    #[cfg(test)]
    pub(crate) fn equality_files(&self) -> impl Iterator<Item = &Arc<DeleteFileEntry>> {
        self.equality_global
            .iter()
            .chain(
                self.equality_by_partition
                    .values()
                    .flat_map(|p| p.tuples.values().flatten().chain(p.unkeyed.iter())),
            )
            .map(|(_, d)| d)
    }

    /// Whether the snapshot has a live equality delete file.
    #[must_use]
    pub fn has_equality_deletes(&self) -> bool {
        !self.equality_global.is_empty() || !self.equality_by_partition.is_empty()
    }

    /// The deletes that apply to `data` (see the module docs).
    ///
    /// # Errors
    /// `KRISHIV_ICEBERG_INVALID_DELETES` (Java's texts) for a DV whose
    /// sequence number is below the data file's, a file-scoped delete or DV
    /// whose spec or partition differs from the data file's, or a data file
    /// without a data sequence number in a snapshot with deletes;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for an equality delete of the data
    /// file's spec when either partition tuple holds an unreadable value.
    pub fn for_data_file(&self, data: &DataFileEntry) -> Result<TaskDeletes, KrishivError> {
        if self.empty {
            return Ok(TaskDeletes::default());
        }
        let file = &data.file;
        let seq = data.data_sequence_number.ok_or_else(|| {
            invalid(format!(
                "Data file {} has no data sequence number: deletes cannot be assigned to it",
                file.path
            ))
        })?;
        let mut out = TaskDeletes::default();
        // Equality deletes: strictly newer than the data.
        for (s, d) in &self.equality_global {
            if *s > seq {
                out.equality_deletes.push(Arc::clone(d));
            }
        }
        if let Some(p) = self.equality_by_partition.get(&file.spec_id) {
            for (s, d) in p.candidates(&file.partition) {
                if *s > seq && same_partition(d, data) {
                    // A tuple that could not be read may equal any other:
                    // offering such a delete widens position deletes only
                    // by files they never name, but an equality delete
                    // would remove rows of another partition (review N4).
                    if !readable(&d.entry.file.partition) || !readable(&file.partition) {
                        return Err(KrishivUnsupportedFeature {
                            feature: format!(
                                "applying the Iceberg equality delete file {} to the data file \
                                 {}: a partition value of one of them cannot be read",
                                d.entry.file.path, file.path
                            ),
                        }
                        .into());
                    }
                    out.equality_deletes.push(Arc::clone(d));
                }
            }
        }
        if let Some((s, dv)) = self.dvs.get(&file.path) {
            if *s < seq {
                return Err(invalid(format!(
                    "DV data sequence number ({s}) must be greater than or equal to data file \
                     sequence number ({seq})"
                )));
            }
            validate_partition_match(dv, data)?;
            out.deletion_vector = Some(Arc::clone(dv));
            return Ok(out);
        }
        if let Some(p) = self.position_by_partition.get(&file.spec_id) {
            for (s, d) in p.candidates(&file.partition) {
                if *s >= seq && same_partition(d, data) {
                    out.position_deletes.push(Arc::clone(d));
                }
            }
        }
        if let Some(by_path) = self.position_by_path.get(&file.path) {
            for (s, d) in by_path {
                if *s >= seq {
                    validate_partition_match(d, data)?;
                    out.position_deletes.push(Arc::clone(d));
                }
            }
        }
        Ok(out)
    }
}

/// Whether the delete's partition tuple may equal the data file's (same
/// spec; a tuple with an unreadable value may equal any).
fn same_partition(d: &DeleteFileEntry, data: &DataFileEntry) -> bool {
    let (a, b) = (&d.entry.file, &data.file);
    a.spec_id == b.spec_id
        && (!readable(&a.partition) || !readable(&b.partition) || a.partition == b.partition)
}

/// Java's `DeleteFileIndex.validatePartitionMatch`.
fn validate_partition_match(d: &DeleteFileEntry, data: &DataFileEntry) -> Result<(), KrishivError> {
    let (a, b) = (&d.entry.file, &data.file);
    if a.spec_id != b.spec_id {
        return Err(invalid(format!(
            "Mismatched partition specs ({}, {}) for delete file {} and data file {}: metadata is \
             corrupted",
            a.spec_id, b.spec_id, a.path, b.path
        )));
    }
    if !same_partition(d, data) {
        return Err(invalid(format!(
            "Mismatched partition tuples ({:?}, {:?}) for delete file {} and data file {}: \
             metadata is corrupted",
            a.partition, b.partition, a.path, b.path
        )));
    }
    Ok(())
}
