// SPDX-License-Identifier: Apache-2.0
//! Deletion vectors of a merge-on-read `DELETE`, `UPDATE` or `MERGE` (design note P3-T021
//! section 6.1 step 4, PR-37d).
//!
//! A merge-on-read rewrite's rows carry, after the table's columns, the target row each one
//! deletes or replaces (`_krishiv_row_file`, `_krishiv_row_index`) and whether its columns are
//! written as a new row (`__krishiv_row_write`; see `RowLevelTable::merge_on_read`). Each
//! driver marks its rows' target rows per file ([`Marks`], a kernel
//! [`KernelDeletionVector`] per file, reserved at an upper bound of its size before a row is
//! marked); the written rows go through the write path as any rewrite's.
//!
//! At finalize ([`write`]) every file's marks are combined with the file's existing deletion
//! vector (delta-spark's `DeletionVectorWriter.storeBitmapAndGenerateResult`,
//! `DMLWithDeletionVectorsHelper.scala:721-753`) and the file is classified as delta-spark's
//! `TouchedFileWithDV` does (`DMLWithDeletionVectorsHelper.scala:568-598`):
//! - unchanged (no new row marked): left alone;
//! - fully replaced (`numRecords` less the old vector's cardinality equals the rows newly
//!   marked): removed whole, with no new vector;
//! - otherwise a new vector: serialized by the kernel's [`StreamingDeletionVectorWriter`] into
//!   one file per statement under the table root (`deletion_vector_<uuid>.bin`, the random
//!   prefix of `delta.randomizeFilePrefixes`; delta-spark's `DeletionVectorStore`), described
//!   as a persisted-relative descriptor, and committed through the kernel's
//!   `update_deletion_vectors` (a remove of the file with its old vector and an add with the
//!   new one, `tightBounds: false`).
//!
//! Counts as delta-spark's `processUnmodifiedData` (`DMLWithDeletionVectorsHelper.scala:
//! 192-230`): `numDeletionVectorsAdded` (a new vector on a file without one),
//! `numDeletionVectorsUpdated` (a file whose vector is replaced), `numDeletionVectorsRemoved`
//! (fully replaced files that had one), the rows newly marked.
//!
//! The vector file is invisible until the commit names it; a failed statement deletes it with
//! its data files, and a cancelled one leaves an orphan no snapshot reads.

use std::collections::{HashMap, HashSet};

use arrow_array::cast::AsArray;
use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use bytes::Bytes;
use delta_kernel::actions::deletion_vector::{DeletionVectorDescriptor, DeletionVectorStorageType};
use delta_kernel::actions::deletion_vector_writer::{
    DeletionVector, KernelDeletionVector, StreamingDeletionVectorWriter,
};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::planner::ROW_WRITE_COLUMN;
use krishiv_exec::{ROW_FILE_COLUMN, ROW_POSITION_COLUMN};
use krishiv_memory::Reservation;
use object_store::path::Path;

use crate::engine::DeltaEngine;
use crate::error::from_kernel;
use crate::table::{DeltaTable, ScanUnit};

/// The marks of one file.
#[derive(Debug)]
struct Marked {
    dv: KernelDeletionVector,
    /// Rows marked, counting a row marked twice twice (an upper bound of the cardinality).
    rows: u64,
    /// The largest row index marked.
    max: u64,
    /// The bytes reserved for `dv`.
    bound: usize,
}

/// An upper bound of the bytes a roaring bitmap of `rows` row indexes up to `max` holds: per
/// 2^16-row container at most 2 bytes a row and at most 8 KiB, plus its header, and the
/// structure's own.
fn bitmap_bound(rows: u64, max: u64) -> usize {
    let containers = rows.min(max / 65_536 + 1);
    let data = rows.saturating_mul(2).min(containers.saturating_mul(8_192));
    usize::try_from(
        data.saturating_add(containers.saturating_mul(32))
            .saturating_add(128),
    )
    .unwrap_or(usize::MAX)
}

/// One driver's marked rows, by the path the log names (see the module docs).
#[derive(Debug)]
pub(crate) struct Marks {
    files: HashMap<String, Marked>,
    /// The reservation marks grow; after [`Self::absorb`] also the absorbed drivers' ones.
    mem: Vec<Reservation>,
}

/// A merge-on-read batch split: the rows to write (the table's columns) and the rows marked,
/// by file.
#[derive(Debug)]
pub(crate) struct Split {
    /// The written rows; `None` when none is written.
    pub(crate) data: Option<RecordBatch>,
    /// The row indexes marked, by file.
    pub(crate) marks: Vec<(String, Vec<u64>)>,
}

/// Splits a merge-on-read rewrite's batch (see the module docs).
pub(crate) fn split(batch: &RecordBatch) -> Result<Split, KrishivError> {
    let schema = batch.schema();
    let index_of = |name: &str| {
        schema
            .index_of(name)
            .map_err(|_| internal(&format!("a merge-on-read row without {name}")))
    };
    let (file_i, pos_i, write_i) = (
        index_of(ROW_FILE_COLUMN)?,
        index_of(ROW_POSITION_COLUMN)?,
        index_of(ROW_WRITE_COLUMN)?,
    );
    let write = batch
        .column(write_i)
        .as_boolean_opt()
        .ok_or_else(|| internal("a merge-on-read write flag that is not a boolean"))?;
    let positions = arrow_cast::cast(batch.column(pos_i), &ArrowType::Int64)
        .map_err(|e| internal(&format!("a merge-on-read row index: {e}")))?;
    let positions: &Int64Array = positions
        .as_primitive_opt()
        .ok_or_else(|| internal("a merge-on-read row index that is not a long"))?;
    let files = arrow_cast::cast(batch.column(file_i), &ArrowType::Binary)
        .map_err(|e| internal(&format!("a merge-on-read row file: {e}")))?;
    let files = files.as_binary::<i32>();
    let mut by_file: HashMap<&[u8], Vec<u64>> = HashMap::new();
    for i in 0..batch.num_rows() {
        if files.is_null(i) {
            continue;
        }
        let index = (!positions.is_null(i))
            .then(|| u64::try_from(positions.value(i)).ok())
            .flatten()
            .ok_or_else(|| internal("a merge-on-read row of a file without its row index"))?;
        by_file.entry(files.value(i)).or_default().push(index);
    }
    let mut marks = Vec::with_capacity(by_file.len());
    for (f, rows) in by_file {
        let path = std::str::from_utf8(f)
            .map_err(|_| internal("a Delta row id file that is not UTF-8"))?
            .to_owned();
        marks.push((path, rows));
    }
    let keep: BooleanArray = write
        .iter()
        .map(|v| Some(v == Some(true)))
        .collect::<BooleanArray>();
    let data = if keep.true_count() == 0 {
        None
    } else {
        let columns: Vec<usize> = (0..batch.num_columns())
            .filter(|i| ![file_i, pos_i, write_i].contains(i))
            .collect();
        let projected = batch
            .project(&columns)
            .map_err(|e| internal(&format!("a merge-on-read row's columns: {e}")))?;
        Some(
            arrow_select::filter::filter_record_batch(&projected, &keep)
                .map_err(|e| internal(&format!("a merge-on-read row's columns: {e}")))?,
        )
    };
    Ok(Split { data, marks })
}

impl Marks {
    /// No marks, reserving from `mem`.
    pub(crate) fn new(mem: Reservation) -> Self {
        Self {
            files: HashMap::new(),
            mem: vec![mem],
        }
    }

    /// Gives back up to `bytes`.
    fn release(&mut self, mut bytes: usize) {
        for r in &mut self.mem {
            let n = bytes.min(r.size());
            r.shrink(n);
            bytes -= n;
        }
    }

    /// Marks `marks`, after reserving their bound; `Some(bytes)` (nothing marked) when the pool
    /// refuses the bytes.
    pub(crate) fn mark(&mut self, marks: Vec<(String, Vec<u64>)>) -> Option<usize> {
        let mut need = 0_usize;
        for (path, rows) in &marks {
            let max = rows.iter().copied().max().unwrap_or(0);
            let n = u64::try_from(rows.len()).unwrap_or(u64::MAX);
            let (old_rows, old_max, old_bound) = self
                .files
                .get(path)
                .map_or((0, 0, 0), |m| (m.rows, m.max, m.bound));
            let bound = bitmap_bound(old_rows.saturating_add(n), old_max.max(max));
            need = need.saturating_add(bound.saturating_sub(old_bound));
        }
        if need > 0 {
            let Some(mem) = self.mem.first_mut() else {
                return Some(need);
            };
            if mem.try_grow(need).is_err() {
                return Some(need);
            }
        }
        for (path, rows) in marks {
            let max = rows.iter().copied().max().unwrap_or(0);
            let n = u64::try_from(rows.len()).unwrap_or(u64::MAX);
            let m = self.files.entry(path).or_insert_with(|| Marked {
                dv: KernelDeletionVector::new(),
                rows: 0,
                max: 0,
                bound: 0,
            });
            m.dv.add_deleted_row_indexes(rows);
            m.rows = m.rows.saturating_add(n);
            m.max = m.max.max(max);
            m.bound = m.bound.max(bitmap_bound(m.rows, m.max));
        }
        None
    }

    /// Takes `other`'s marks and reservations in (the bound of a union is at most the sum of
    /// the bounds; a file both marked holds both bounds until it is written).
    pub(crate) fn absorb(&mut self, other: Self) {
        self.mem.extend(other.mem);
        for (path, theirs) in other.files {
            match self.files.get_mut(&path) {
                Some(m) => {
                    m.dv.add_deleted_row_indexes(DeletionVector::into_iter(theirs.dv));
                    m.rows = m.rows.saturating_add(theirs.rows);
                    m.max = m.max.max(theirs.max);
                    m.bound = m.bound.saturating_add(theirs.bound);
                }
                None => {
                    self.files.insert(path, theirs);
                }
            }
        }
    }
}

/// What the vectors of one statement did.
#[derive(Debug, Default)]
pub(crate) struct DvOutcome {
    /// The new vectors of the files that keep a live row, by path.
    pub(crate) descriptors: HashMap<String, DeletionVectorDescriptor>,
    /// The files fully replaced: removed whole.
    pub(crate) removed: HashSet<String>,
    /// Rows newly marked (delta-spark's `numModifiedRows`).
    pub(crate) modified_rows: u64,
    /// `numDeletionVectorsAdded`.
    pub(crate) added: u64,
    /// `numDeletionVectorsUpdated`.
    pub(crate) updated: u64,
    /// `numDeletionVectorsRemoved`: fully replaced files that had a vector.
    pub(crate) removed_vectors: u64,
    /// The bytes of the fully replaced files.
    pub(crate) removed_bytes: u64,
}

/// z85 (ZeroMQ's base 85, delta-spark's `Base85Codec`) of a UUID's 16 bytes.
fn z85(bytes: &[u8; 16]) -> String {
    const ALPHABET: &[u8; 85] =
        b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";
    let mut out = String::with_capacity(20);
    for chunk in bytes.as_chunks::<4>().0 {
        let mut v = chunk
            .iter()
            .fold(0_u32, |acc, b| (acc << 8) | u32::from(*b));
        let mut digits = [0_u8; 5];
        for d in digits.iter_mut().rev() {
            // `v % 85` is below 85: a valid index.
            *d = ALPHABET[usize::try_from(v % 85).unwrap_or(0)];
            v /= 85;
        }
        out.extend(digits.iter().map(|b| char::from(*b)));
    }
    out
}

/// A UUID's standard text form.
fn uuid_text(b: &[u8; 16]) -> String {
    let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// The table's random file prefix (`delta.randomizeFilePrefixes`, `delta.randomPrefixLength`
/// default 2; delta-spark's `Utils.getRandomPrefixLength`), empty without one.
fn random_prefix(table: &DeltaTable) -> String {
    let props = table.snapshot().table_properties();
    if props.randomize_file_prefixes != Some(true) {
        return String::new();
    }
    let len = props.random_prefix_length.map_or(2, std::num::NonZero::get);
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let bytes = super::files::uuid_bytes();
    (0..len)
        .map(|i| {
            let k = usize::try_from(i).unwrap_or(0) % bytes.len();
            char::from(CHARS[usize::from(bytes[k]) % CHARS.len()])
        })
        .collect()
}

/// One statement's vector file while it is written.
struct DvFile {
    upload: krishiv_io::io::Upload,
    /// Bytes written so far (the next vector's offset).
    offset: usize,
    /// The descriptor's path field (`<prefix><z85 uuid>`).
    encoded: String,
    /// Held while the upload buffers its parts.
    mem: Reservation,
}

/// Writes the vectors of `marks` (every driver's) for `table` (see the module docs). The
/// vector file's path is pushed to `files` before its first byte, so a failure (here or later,
/// before the commit) deletes it with the data files.
///
/// # Errors
/// A store failure, the kernel's serialization, a refused reservation
/// (`UNABLE_TO_ACQUIRE_MEMORY`), a marked file not live in the snapshot.
pub(crate) async fn write(
    engine: &DeltaEngine,
    table: &DeltaTable,
    marks: Vec<Marks>,
    files: &std::sync::Mutex<Vec<Path>>,
) -> Result<DvOutcome, KrishivError> {
    let mut all: Option<Marks> = None;
    for m in marks {
        match &mut all {
            Some(a) => a.absorb(m),
            None => all = Some(m),
        }
    }
    let mut out = DvOutcome::default();
    let Some(mut all) = all else {
        return Ok(out);
    };
    let units: HashMap<&str, &ScanUnit> =
        table.units().iter().map(|u| (u.path.as_str(), u)).collect();
    let mut paths: Vec<String> = all.files.keys().cloned().collect();
    paths.sort_unstable();
    let name = table.name().to_owned();
    let mut extra = match all.mem.first_mut() {
        Some(r) => r.split(0).unguarded(),
        None => return Err(internal("deletion vector marks without a reservation")),
    };
    let mut file: Option<DvFile> = None;
    let result = async {
        for path in paths {
            let Some(mut m) = all.files.remove(&path) else {
                continue;
            };
            let unit = units.get(path.as_str()).ok_or_else(|| {
                internal(&format!(
                    "the Delta row id file {path} is not a live file of the snapshot"
                ))
            })?;
            let existing = if unit.dv.has_vector() {
                unit.dv
                    .cardinality()
                    .map_err(|e| from_kernel(&e, &name))?
                    .unwrap_or(0)
            } else {
                0
            };
            if existing > 0 {
                // The old vector's rows, 8 bytes each while they are added.
                let bytes = usize::try_from(existing)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(8);
                extra.try_grow(bytes)?;
                let (dv, root, n) = (unit.dv.clone(), engine.table_root().clone(), name.clone());
                let old = engine
                    .delta_blocking(move |e| {
                        dv.get_row_indexes(e, &root)
                            .map_err(|err| from_kernel(&err, &n))
                    })
                    .await??
                    .unwrap_or_default();
                m.dv.add_deleted_row_indexes(old);
                extra.shrink(bytes);
            }
            let cardinality = m.dv.cardinality();
            let modified = cardinality.saturating_sub(existing);
            if modified == 0 {
                all.release(m.bound);
                continue;
            }
            out.modified_rows = out.modified_rows.saturating_add(modified);
            if unit.records == Some(cardinality) {
                // Every live row goes: the file is removed whole.
                out.removed.insert(path.clone());
                out.removed_bytes = out.removed_bytes.saturating_add(unit.size);
                if existing > 0 {
                    out.removed_vectors += 1;
                }
                all.release(m.bound);
                continue;
            }
            let bound = m.bound;
            let (dv, n) = (m.dv, name.clone());
            let (framed, written) = engine
                .delta_blocking(move |_| {
                    let mut buf: Vec<u8> = Vec::new();
                    let mut w = StreamingDeletionVectorWriter::new(&mut buf);
                    let r = w
                        .write_deletion_vector(dv)
                        .map_err(|err| from_kernel(&err, &n))?;
                    w.finalize().map_err(|err| from_kernel(&err, &n))?;
                    Ok::<_, KrishivError>((buf, r))
                })
                .await??;
            all.release(bound);
            if file.is_none() {
                let uuid = super::files::uuid_bytes();
                let prefix = random_prefix(table);
                let relative = if prefix.is_empty() {
                    format!("deletion_vector_{}.bin", uuid_text(&uuid))
                } else {
                    format!("{prefix}/deletion_vector_{}.bin", uuid_text(&uuid))
                };
                let url = engine
                    .table_root()
                    .join(&relative)
                    .map_err(|e| internal(&format!("the deletion vector file {relative}: {e}")))?;
                let p = engine.root().confine(&url)?;
                files
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(p.clone());
                let mut upload = engine.io().upload(engine.store(), p)?;
                // The file's version byte (the writer's first byte).
                upload.write(Bytes::from_static(&[1_u8])).await?;
                file = Some(DvFile {
                    upload,
                    offset: 1,
                    encoded: format!("{prefix}{}", z85(&uuid)),
                    mem: extra.split(0),
                });
            }
            let Some(f) = file.as_mut() else {
                return Err(internal("a deletion vector file that is not open"));
            };
            // The kernel's writer frames each vector after the file's version byte: one fresh
            // writer per vector, its version byte dropped, places it at the file's offset.
            let frame = framed
                .get(1..)
                .ok_or_else(|| internal("an empty deletion vector frame"))?;
            let descriptor = DeletionVectorDescriptor::try_new(
                DeletionVectorStorageType::PersistedRelative,
                f.encoded.clone(),
                Some(
                    i32::try_from(f.offset)
                        .map_err(|_| internal("a deletion vector file past 2 GiB"))?,
                ),
                written.size_in_bytes,
                written.cardinality,
            )
            .map_err(|err| from_kernel(&err, &name))?;
            f.offset = f.offset.saturating_add(frame.len());
            f.upload.write(Bytes::copy_from_slice(frame)).await?;
            let held = usize::try_from(f.upload.held_bytes()).unwrap_or(usize::MAX);
            let have = f.mem.size();
            if held > have {
                f.mem.try_grow(held - have)?;
            } else {
                f.mem.shrink(have - held);
            }
            if existing > 0 {
                out.updated += 1;
            } else {
                out.added += 1;
            }
            out.descriptors.insert(path, descriptor);
        }
        if let Some(f) = file.take() {
            let DvFile {
                upload, mut mem, ..
            } = f;
            upload.finish().await?;
            let size = mem.size();
            mem.shrink(size);
        }
        Ok::<(), KrishivError>(())
    }
    .await;
    if let Err(e) = result {
        if let Some(f) = file.take() {
            f.upload.abort().await;
        }
        return Err(e);
    }
    Ok(out)
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn z85_names_the_file_the_kernel_reads() {
        let uuid = [
            0x6f, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x4f, 0x60, 0x81, 0x92, 0xa3, 0xb4, 0xc5, 0xd6,
            0xe7, 0xf8,
        ];
        let d = DeletionVectorDescriptor::try_new(
            DeletionVectorStorageType::PersistedRelative,
            format!("ab{}", z85(&uuid)),
            Some(1),
            10,
            2,
        )
        .unwrap();
        let root = url::Url::parse("file:///tmp/t/").unwrap();
        let path = d.absolute_path(&root).unwrap().unwrap();
        assert_eq!(
            path.as_str(),
            format!("file:///tmp/t/ab/deletion_vector_{}.bin", uuid_text(&uuid))
        );
    }

    #[test]
    fn the_bound_covers_sparse_and_dense_vectors() {
        // Sparse: 2 bytes a row and a header per container.
        assert!(bitmap_bound(10, 10_000_000) >= 20);
        // Dense: one 8 KiB container per 2^16 rows, not 2 bytes a row.
        let dense = bitmap_bound(10_000_000, 10_000_000);
        assert!(dense < 20_000_000 / 10, "{dense}");
        assert!(dense >= 153 * 8_192);
        // A real bitmap fits its bound.
        for (rows, step) in [(1_000_u64, 1_u64), (1_000, 70_000), (200_000, 1)] {
            let mut dv = KernelDeletionVector::new();
            dv.add_deleted_row_indexes((0..rows).map(|i| i * step));
            let bytes = DeletionVector::serialize(dv).unwrap().len();
            assert!(
                bytes <= bitmap_bound(rows, (rows - 1) * step),
                "{rows} {step}"
            );
        }
    }

    #[test]
    fn split_keeps_written_rows_and_marks_by_file() {
        use arrow_array::{BinaryViewArray, StringArray};
        use arrow_schema::{Field, Schema};
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", ArrowType::Utf8, true),
            Field::new(ROW_FILE_COLUMN, ArrowType::BinaryView, true),
            Field::new(ROW_POSITION_COLUMN, ArrowType::Int64, true),
            Field::new(ROW_WRITE_COLUMN, ArrowType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
                Arc::new(BinaryViewArray::from(vec![
                    Some(b"f1".as_slice()),
                    Some(b"f1".as_slice()),
                    None,
                ])),
                Arc::new(Int64Array::from(vec![Some(3), Some(5), None])),
                Arc::new(BooleanArray::from(vec![true, false, true])),
            ],
        )
        .unwrap();
        let s = split(&batch).unwrap();
        let data = s.data.unwrap();
        assert_eq!(data.num_columns(), 1);
        assert_eq!(data.num_rows(), 2);
        assert_eq!(s.marks, vec![("f1".to_owned(), vec![3, 5])]);
    }
}
