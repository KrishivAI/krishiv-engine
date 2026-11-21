// SPDX-License-Identifier: Apache-2.0
//! Loading the position deletes and deletion vectors a scan applies (P3-T026,
//! PR-25b; design note §3.3), into one [`RowSelection`] per data file that the
//! file scan reads through [`FileSelections`]: deleted rows are never decoded.
//!
//! - **Position delete files** (Parquet, columns `file_path` and `pos`) are
//!   read once per scan however many data files they apply to (review-1 S12),
//!   with row groups pruned by `file_path IN (<the data files>)`, and only the
//!   rows naming one of those data files kept. Their order is not trusted:
//!   positions are collected and sorted.
//! - **Deletion vectors** are read by range (`content_offset`,
//!   `content_size_in_bytes`) from their Puffin file and decoded by
//!   [`decode_dv`], which checks what Java checks.
//! - A position past the end of the data file deletes nothing, as in Java
//!   (the selection excludes positions; it does not bound the file).
//! - A delete file that is missing or unreadable fails the query with the
//!   store's error naming it; it is never skipped.
//!
//! The positions are charged to the query's memory (the scan's consumer,
//! [`super::lazy`]) from their read until their data files' selections are
//! built (one data file at a time), and the selections while they live; a
//! load that does not fit fails with the memory class (PR-25d).

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_schema::{DataType, Field, Schema};
use futures::{StreamExt, TryStreamExt};
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::FileSelections;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    ByteRange, ColumnPath, FieldMapping, FileFormat as _, FileRef, FormatOptions, IoContext,
    PushdownExpr, RowSelection, ScalarValue, ScanOptions, ScanRequest, ScanStream,
};
use krishiv_memory::{ConsumerHandle, Reservation};
use roaring::RoaringTreemap;

use crate::plan::deletes::invalid;
use crate::plan::{DataFileEntry, DeleteFileEntry, FileFormatKind, TaskDeletes, decode_dv};

/// Delete files read at once.
const DELETE_FILE_CONCURRENCY: usize = 8;
/// The bytes a deleted position is charged: it is collected as a `u64`
/// before it becomes a selection.
const POSITION_BYTES: usize = 8;
/// The bytes a range of a selection is charged (`Range<u64>`).
const RANGE_BYTES: usize = std::mem::size_of::<std::ops::Range<u64>>();
/// At most this many data paths are pushed as the `file_path IN (...)`
/// row-group filter of one position delete file (pruning only).
const MAX_PUSHED_PATHS: usize = 256;

/// The rows to read of each data file with deletes, by store and path.
#[derive(Default)]
pub(super) struct DeleteSelections {
    by_file: HashMap<(String, String), RowSelection>,
}

impl fmt::Debug for DeleteSelections {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeleteSelections")
            .field("files", &self.by_file.len())
            .finish()
    }
}

fn key(f: &FileRef) -> (String, String) {
    (f.store.key().to_owned(), f.path.as_ref().to_owned())
}

#[cfg(test)]
impl DeleteSelections {
    /// How many data files have a selection.
    pub(super) fn files(&self) -> usize {
        self.by_file.len()
    }
}

impl FileSelections for DeleteSelections {
    fn selection(&self, file: &FileRef) -> Option<RowSelection> {
        self.by_file.get(&key(file)).cloned()
    }
}

/// The selections of `tasks` (each data file with its `FileRef` and the
/// deletes that apply to it), reading every delete file once, and the
/// reservations of `mem` that pay for them (held as long as the selections).
///
/// # Errors
/// The store's error for a delete file that is missing or unreadable;
/// `KRISHIV_ICEBERG_INVALID_DELETES` for a corrupt DV or a negative or null
/// position; `KRISHIV_UNSUPPORTED_FEATURE` for an Avro or ORC position
/// delete file; `UNABLE_TO_ACQUIRE_MEMORY` when the positions do not fit.
pub(super) async fn load(
    io: &IoContext,
    locate: &(dyn Fn(&str) -> Result<FileRef, KrishivError> + Send + Sync),
    mem: &Arc<ConsumerHandle>,
    tasks: &[(&DataFileEntry, &FileRef, &TaskDeletes)],
) -> Result<(DeleteSelections, Vec<Reservation>), KrishivError> {
    // Each position delete file once, with the data files it applies to.
    let mut position: Vec<(Arc<DeleteFileEntry>, BTreeSet<String>)> = Vec::new();
    let mut slot: HashMap<String, usize> = HashMap::new();
    let mut dvs = Vec::new();
    for (data, _, deletes) in tasks {
        if let Some(dv) = &deletes.deletion_vector {
            dvs.push((data.file.path.clone(), Arc::clone(dv)));
        }
        for d in &deletes.position_deletes {
            let i = *slot.entry(d.entry.file.path.clone()).or_insert_with(|| {
                position.push((Arc::clone(d), BTreeSet::new()));
                position.len() - 1
            });
            position[i].1.insert(data.file.path.clone());
        }
    }
    // The read futures are built before their streams: a closure that
    // borrows into `stream::iter` makes the future lose `Send` (its lifetime
    // is not provably general enough), and the scan's source must be `Send`.
    let position_reads: Vec<_> = position
        .iter()
        .map(|(d, paths)| read_position_file(io, locate, mem, d, paths))
        .collect();
    let dv_reads: Vec<_> = dvs
        .iter()
        .map(|(data, dv)| async move {
            read_dv(io, locate, mem, dv)
                .await
                .map(|(p, charge)| (data.clone(), p, charge))
        })
        .collect();
    let read_positions = futures::stream::iter(position_reads)
        .buffered(DELETE_FILE_CONCURRENCY)
        .try_collect::<Vec<_>>();
    let read_dvs = futures::stream::iter(dv_reads)
        .buffered(DELETE_FILE_CONCURRENCY)
        .try_collect::<Vec<_>>();
    let (from_files, from_dvs) = futures::try_join!(read_positions, read_dvs)?;
    // Each read's charge, with how many data files' positions it still
    // pays for: it goes once they are all converted (a position delete file
    // can cover several data files).
    let mut deleted: HashMap<String, (Vec<u64>, Vec<usize>)> = HashMap::new();
    let mut charges: Vec<(Option<Reservation>, usize)> =
        Vec::with_capacity(from_files.len() + from_dvs.len());
    for (per_file, charge) in from_files {
        let c = charges.len();
        charges.push((Some(charge), per_file.len()));
        for (path, mut p) in per_file {
            let at = deleted.entry(path).or_default();
            at.0.append(&mut p);
            at.1.push(c);
        }
    }
    for (path, p, charge) in from_dvs {
        let c = charges.len();
        charges.push((Some(charge), 1));
        let at = deleted.entry(path).or_default();
        at.0.extend(p);
        at.1.push(c);
    }
    // One data file at a time: its positions are sorted in a copy and
    // become ranges (one per position, plus one), both charged while it
    // converts; then the copy's charge goes and the ranges' stays as long as
    // the selections.
    let mut held = mem.reservation();
    let mut out = DeleteSelections::default();
    for (data, file, _) in tasks {
        let Some((p, paid_by)) = deleted.remove(&data.file.path) else {
            continue;
        };
        let n = p.len().saturating_add(1);
        held.grow(n.saturating_mul(POSITION_BYTES + RANGE_BYTES))
            .await?;
        // `excluding` sorts and dedups; the selection is unbounded so a
        // wrong record count in the manifest never drops a row.
        out.by_file
            .insert(key(file), RowSelection::excluding(u64::MAX, p));
        // A selection's ranges are allocated for its positions plus one
        // (its capacity, which `RowSelection` does not expose; dedup only
        // lowers it).
        held.shrink(n.saturating_mul(POSITION_BYTES));
        for c in paid_by {
            if let Some((charge, left)) = charges.get_mut(c) {
                *left = left.saturating_sub(1);
                if *left == 0 {
                    drop(charge.take());
                }
            }
        }
    }
    drop((deleted, charges));
    Ok((out, vec![held]))
}

/// The positions the delete files `files` (a DV or v2 position delete
/// files) delete from the data file `data`: what a merge-on-read write
/// absorbs into its new delete (PR-36c, Java's `BaseDeleteLoader
/// .loadPositionDeletes` through `PreviousDeleteLoader`). Read with the
/// scan's readers and checks; the charge of `mem` is released once they are
/// unioned (the result is the caller's to account).
///
/// # Errors
/// As [`load`].
pub(super) async fn positions_of(
    io: &IoContext,
    locate: &(dyn Fn(&str) -> Result<FileRef, KrishivError> + Send + Sync),
    mem: &Arc<ConsumerHandle>,
    data: &str,
    files: &[Arc<DeleteFileEntry>],
) -> Result<RoaringTreemap, KrishivError> {
    let mut out = RoaringTreemap::new();
    let paths = BTreeSet::from([data.to_owned()]);
    for d in files {
        match d.kind {
            crate::plan::DeleteKind::DeletionVector => {
                let (p, charge) = read_dv(io, locate, mem, d).await?;
                out |= p;
                drop(charge);
            }
            crate::plan::DeleteKind::PositionDelete => {
                let (mut by_path, charge) = read_position_file(io, locate, mem, d, &paths).await?;
                out.extend(by_path.remove(data).unwrap_or_default());
                drop(charge);
            }
            crate::plan::DeleteKind::EqualityDelete => {}
        }
    }
    Ok(out)
}

/// The positions of a v2 position delete file `d` that name one of `paths`,
/// by data path, and the reservation of `mem` paying for them.
async fn read_position_file(
    io: &IoContext,
    locate: &(dyn Fn(&str) -> Result<FileRef, KrishivError> + Send + Sync),
    mem: &Arc<ConsumerHandle>,
    d: &DeleteFileEntry,
    paths: &BTreeSet<String>,
) -> Result<(HashMap<String, Vec<u64>>, Reservation), KrishivError> {
    let f = &d.entry.file;
    if f.format != FileFormatKind::Parquet {
        return Err(KrishivUnsupportedFeature {
            feature: format!(
                "reading Iceberg {:?} position delete files ({})",
                f.format, f.path
            ),
        }
        .into());
    }
    let mut file = locate(&f.path)?;
    file.size = f.size;
    let schema = Arc::new(Schema::new(vec![
        Field::new("file_path", DataType::Utf8, true),
        Field::new("pos", DataType::Int64, true),
    ]));
    let mut req = ScanRequest::new(
        Arc::clone(&schema),
        vec![ColumnPath::new("file_path"), ColumnPath::new("pos")],
    );
    req.mapping = FieldMapping::ByName {
        case_sensitive: true,
    };
    if paths.len() <= MAX_PUSHED_PATHS {
        req.predicate = Some(PushdownExpr::In {
            column: ColumnPath::new("file_path"),
            values: paths.iter().map(|p| ScalarValue::Utf8(p.clone())).collect(),
            collation: krishiv_io::Collation::default(),
        });
    }
    let mut charge = mem.reservation();
    let mut stream = ScanStream::open(
        io,
        &ParquetFormat,
        file.clone(),
        ByteRange::whole(file.size),
        &req,
        &FormatOptions::defaults(ParquetFormat.options()),
        mem.reservation(),
        ScanOptions::default(),
    )
    .await?;
    let mut out: HashMap<String, Vec<u64>> = HashMap::new();
    while let Some(batch) = stream.next_batch().await? {
        let bad = |what: &str| {
            invalid(format!(
                "Cannot read position delete file {}: {what}",
                f.path
            ))
        };
        let names = batch
            .column(0)
            .as_string_opt::<i32>()
            .ok_or_else(|| bad("file_path is not a string"))?;
        let pos = batch
            .column(1)
            .as_primitive_opt::<Int64Type>()
            .ok_or_else(|| bad("pos is not a long"))?;
        let mut kept = 0_usize;
        for row in 0..batch.num_rows() {
            if names.is_null(row) {
                continue;
            }
            let name = names.value(row);
            if !paths.contains(name) {
                continue;
            }
            if pos.is_null(row) {
                return Err(bad("a null position"));
            }
            let p = u64::try_from(pos.value(row))
                .map_err(|_| invalid(format!("Invalid position: {}", pos.value(row))))?;
            // One allocation per data file, not per delete row.
            match out.get_mut(name) {
                Some(v) => v.push(p),
                None => {
                    out.insert(name.to_owned(), vec![p]);
                }
            }
            kept += 1;
        }
        // Paid per batch, before the next one is read: a delete file too
        // large for the query fails with the memory class.
        charge.grow(kept.saturating_mul(POSITION_BYTES)).await?;
    }
    Ok((out, charge))
}

/// The positions of the deletion vector `dv`, and the reservation of `mem`
/// paying for them.
async fn read_dv(
    io: &IoContext,
    locate: &(dyn Fn(&str) -> Result<FileRef, KrishivError> + Send + Sync),
    mem: &Arc<ConsumerHandle>,
    dv: &DeleteFileEntry,
) -> Result<(RoaringTreemap, Reservation), KrishivError> {
    let f = &dv.entry.file;
    let (Some(offset), Some(len)) = (f.content_offset, f.content_size_in_bytes) else {
        return Err(invalid(format!(
            "Deletion vector {} has no content offset or size",
            f.path
        )));
    };
    let (Ok(start), Ok(size)) = (u64::try_from(offset), u64::try_from(len)) else {
        return Err(invalid(format!(
            "Deletion vector {} has the invalid range {offset} + {len}",
            f.path
        )));
    };
    // A range past the end of the file is invalid, not a memory request
    // (review 25d N5): checked before it is charged.
    if start.checked_add(size).is_none_or(|end| end > f.size) {
        return Err(invalid(format!(
            "Deletion vector {} has the invalid range {offset} + {len}",
            f.path
        )));
    }
    let mut file = locate(&f.path)?;
    file.size = f.size;
    // The blob while it is decoded; the positions once they are.
    let mut charge = mem.reservation();
    charge
        .grow(usize::try_from(size).unwrap_or(usize::MAX))
        .await?;
    let bytes = io
        .get_ranges(
            &file,
            std::slice::from_ref(&(start..start.saturating_add(size))),
        )
        .await?;
    let blob = match bytes.as_slice() {
        [one] => one.clone(),
        many => bytes::Bytes::from(many.concat()),
    };
    let positions = decode_dv(&blob, len, f.record_count, &f.path)?;
    drop((blob, bytes));
    let held = usize::try_from(positions.len())
        .unwrap_or(usize::MAX)
        .saturating_mul(POSITION_BYTES);
    match held.checked_sub(charge.size()) {
        Some(more) => charge.grow(more).await?,
        None => charge.shrink(charge.size() - held),
    }
    Ok((positions, charge))
}
