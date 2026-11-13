// SPDX-License-Identifier: Apache-2.0
//! Equality deletes (P3-T026, PR-25c; design note §3.3): each applicable
//! equality delete file is read once per scan into a set of row keys, and a
//! wrapper around the file scan's source drops the decoded rows whose key is
//! in a set that applies to the row's data file.
//!
//! - **Keys.** A delete file is read projected to its `equality_ids` columns
//!   by field id, through the table's current schema, so a key written as
//!   `int` and a column read as `long` compare as `long` (type promotion).
//!   Keys are compared in Arrow's row format (one buffer per set, a
//!   `hashbrown` table of indices hashed with foldhash): `null` equals `null` (Iceberg
//!   deletes, unlike SQL), every `NaN` equals every `NaN` and `-0.0` differs
//!   from `0.0` (Java's `Comparators` over `Float`/`Double`, the order the
//!   delete set `StructLikeSet` uses). A field of a null struct is null, as
//!   Java's `StructProjection` reads it.
//! - **Which file a row is from.** The scan reads one more per-file constant
//!   column, [`FILE_ORDINAL`], the data file's index in the scan (through
//!   the same `FileConstants` hook as identity partitions), so the wrapper
//!   needs no change to the file scan. The column, and the equality columns
//!   the query did not ask for, are dropped before the batch leaves.
//! - **Scope.** A set is shared by every data file its delete file applies
//!   to (sequence numbers and partitions were decided by `DeleteIndex`).
//! - The sets are held for the scan and charged to the query's memory as
//!   they are read (PR-25d); a set that does not fit fails the query with
//!   the memory class (no spilling: design note §11 item 2).
//! - A delete file that is missing or unreadable fails the query; an
//!   equality id that is not a primitive column of the read schema is
//!   refused (`KRISHIV_UNSUPPORTED_FEATURE`), never ignored.

use std::collections::HashMap;
use std::fmt;
use std::hash::BuildHasher as _;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int32Type};
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions};
use arrow_row::{RowConverter, Rows, SortField};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use foldhash::fast::FixedState;
use futures::{StreamExt, TryStreamExt};
use hashbrown::HashTable;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    ByteRange, ColumnPath, FieldMapping, FileFormat as _, FileRef, FormatOptions, IoContext,
    MetadataCols, ScanOptions, ScanRequest, ScanStream, WithoutFieldIds,
};
use krishiv_memory::{ConsumerHandle, Reservation};

use crate::plan::{DeleteFileEntry, FileFormatKind, SchemaField};

/// The per-file constant column holding the data file's index in the scan.
pub(super) const FILE_ORDINAL: &str = "_krishiv_iceberg_file";

/// Delete files read at once.
const DELETE_FILE_CONCURRENCY: usize = 8;

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

/// The [`FILE_ORDINAL`] field.
pub(super) fn ordinal_field() -> Field {
    Field::new(FILE_ORDINAL, DataType::Int32, false)
}

/// The keys of one equality delete file.
struct KeySet {
    /// The equality columns, in the delete file's `equality_ids` order.
    columns: Vec<ColumnPath>,
    /// The key columns' types, which `converter` encodes.
    types: Vec<DataType>,
    converter: RowConverter,
    keys: Keys,
    /// Pays for `keys` while the set lives.
    _charge: Reservation,
}

/// Distinct keys in Arrow's row format: every key read in one [`Rows`]
/// buffer and a table of the indices of the distinct ones, hashed with
/// foldhash (review 25c N3: no allocation per key, no SipHash per probe).
/// A duplicate key stays in the buffer and is not indexed.
struct Keys {
    rows: Option<Rows>,
    index: HashTable<usize>,
    hasher: FixedState,
}

impl Keys {
    fn new() -> Self {
        Self {
            rows: None,
            index: HashTable::new(),
            hasher: FixedState::with_seed(0),
        }
    }

    /// The bytes held.
    fn size(&self) -> usize {
        self.rows.as_ref().map_or(0, Rows::size)
            + self.index.capacity() * (std::mem::size_of::<usize>() + 1)
    }

    /// Appends the keys of `cols`, encoded by `converter`.
    fn append(&mut self, converter: &RowConverter, cols: &[ArrayRef]) -> Result<(), ArrowError> {
        let rows = self.rows.get_or_insert_with(|| converter.empty_rows(0, 0));
        let from = rows.num_rows();
        converter.append(rows, cols)?;
        let hasher = &self.hasher;
        for i in from..rows.num_rows() {
            let row = rows.row(i);
            let h = hasher.hash_one(row.as_ref());
            if self.index.find(h, |&j| rows.row(j) == row).is_none() {
                self.index
                    .insert_unique(h, i, |&j| hasher.hash_one(rows.row(j).as_ref()));
            }
        }
        Ok(())
    }

    /// Whether `key` is one of the keys.
    fn contains(&self, key: &[u8]) -> bool {
        let Some(rows) = &self.rows else {
            return false;
        };
        let h = self.hasher.hash_one(key);
        self.index
            .find(h, |&j| rows.row(j).as_ref() == key)
            .is_some()
    }
}

/// The equality deletes of a scan: for each data file (by its index in the
/// scan), the key sets that apply to it.
pub(super) struct EqualityDeletes {
    by_file: Vec<Vec<Arc<KeySet>>>,
}

impl fmt::Debug for EqualityDeletes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EqualityDeletes")
            .field(
                "files",
                &self.by_file.iter().filter(|s| !s.is_empty()).count(),
            )
            .finish()
    }
}

/// The equality deletes of the scan's files: `(i, d)` in `deletes` are the
/// equality delete files `d` of data file `i` (of `files`). Every delete file
/// is read once.
///
/// # Errors
/// The store's error for a delete file that is missing or unreadable;
/// `KRISHIV_UNSUPPORTED_FEATURE` for a delete file that is not Parquet or
/// whose equality ids are missing, empty, or not primitive columns of the
/// read schema (`fields`); `UNABLE_TO_ACQUIRE_MEMORY` when the keys do not
/// fit in the query's memory (`mem`, which pays for them while they live).
pub(super) async fn load(
    io: &IoContext,
    locate: &(dyn Fn(&str) -> Result<FileRef, KrishivError> + Send + Sync),
    mem: &Arc<ConsumerHandle>,
    table: &SchemaRef,
    fields: &[SchemaField],
    files: usize,
    deletes: &[(usize, &[Arc<DeleteFileEntry>])],
) -> Result<EqualityDeletes, KrishivError> {
    let mut unique: Vec<Arc<DeleteFileEntry>> = Vec::new();
    let mut slot: HashMap<String, usize> = HashMap::new();
    for d in deletes.iter().flat_map(|(_, d)| d.iter()) {
        slot.entry(d.entry.file.path.clone()).or_insert_with(|| {
            unique.push(Arc::clone(d));
            unique.len() - 1
        });
    }
    // Futures built before the stream, so the load stays `Send` (see
    // `deletes::load`).
    // Java's delete loader reads delete files without the name mapping
    // (writers give them ids): a file without ids takes the positional
    // fallback.
    let mapping = &FieldMapping::FieldIdOr(WithoutFieldIds::Position);
    let reads: Vec<_> = unique
        .iter()
        .map(|d| read_keys(io, locate, mem, table, mapping, fields, d))
        .collect();
    let sets: Vec<Arc<KeySet>> = futures::stream::iter(reads)
        .buffered(DELETE_FILE_CONCURRENCY)
        .map_ok(Arc::new)
        .try_collect()
        .await?;
    let mut by_file: Vec<Vec<Arc<KeySet>>> = vec![Vec::new(); files];
    for (i, ds) in deletes {
        let at = by_file
            .get_mut(*i)
            .ok_or_else(|| internal(format!("no data file {i} in the scan")))?;
        at.extend(
            ds.iter()
                .filter_map(|d| slot.get(&d.entry.file.path).map(|k| Arc::clone(&sets[*k]))),
        );
    }
    Ok(EqualityDeletes { by_file })
}

/// The equality columns of the delete file `d`, in its `equality_ids` order.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for missing or empty equality ids, or an id
/// that is not a primitive column of the read schema (`fields`).
fn key_paths(fields: &[SchemaField], d: &DeleteFileEntry) -> Result<Vec<ColumnPath>, KrishivError> {
    let f = &d.entry.file;
    let ids = f.equality_ids.as_deref().unwrap_or_default();
    if ids.is_empty() {
        return Err(unsupported(format!(
            "reading the Iceberg equality delete file {} without equality field ids",
            f.path
        )));
    }
    ids.iter()
        .map(|id| {
            fields
                .iter()
                .find(|c| c.id == *id)
                .map(|c| c.path.clone())
                .ok_or_else(|| {
                    unsupported(format!(
                        "applying the Iceberg equality delete file {}: the equality field id {id} \
                         is not a primitive column of the read schema",
                        f.path
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()
}

/// Every equality column of `deletes`, once each, in first-seen order: what
/// a scan reading files with these deletes must read besides the query's.
///
/// # Errors
/// As [`key_paths`].
pub(super) fn equality_columns<'a>(
    fields: &[SchemaField],
    deletes: impl IntoIterator<Item = &'a Arc<DeleteFileEntry>>,
) -> Result<Vec<ColumnPath>, KrishivError> {
    let mut out: Vec<ColumnPath> = Vec::new();
    for d in deletes {
        for c in key_paths(fields, d)? {
            if !out.contains(&c) {
                out.push(c);
            }
        }
    }
    Ok(out)
}

/// The key set of the equality delete file `d`.
async fn read_keys(
    io: &IoContext,
    locate: &(dyn Fn(&str) -> Result<FileRef, KrishivError> + Send + Sync),
    mem: &Arc<ConsumerHandle>,
    table: &SchemaRef,
    mapping: &FieldMapping,
    fields: &[SchemaField],
    d: &DeleteFileEntry,
) -> Result<KeySet, KrishivError> {
    let f = &d.entry.file;
    if f.format != FileFormatKind::Parquet {
        return Err(unsupported(format!(
            "reading Iceberg {:?} equality delete files ({})",
            f.format, f.path
        )));
    }
    let columns = key_paths(fields, d)?;
    let mut file = locate(&f.path)?;
    file.size = f.size;
    let mut req = ScanRequest::new(Arc::clone(table), columns.clone());
    // Columns by id (the caller's rule for a file without ids).
    req.mapping = mapping.clone();
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
    let mut converter: Option<RowConverter> = None;
    let mut types: Vec<DataType> = Vec::new();
    let mut keys = Keys::new();
    while let Some(batch) = stream.next_batch().await? {
        let cols = key_columns(&batch, &columns)?;
        let conv = match &mut converter {
            Some(c) => c,
            None => {
                types = cols.iter().map(|c| c.data_type().clone()).collect();
                converter.insert(converter_for(&cols)?)
            }
        };
        keys.append(conv, &cols)
            .map_err(|e| internal(format!("equality delete keys of {}: {e}", f.path)))?;
        // Paid per batch: a key set too large for the query fails with
        // the memory class.
        let held = keys.size();
        if let Some(more) = held.checked_sub(charge.size()) {
            charge.grow(more).await?;
        }
    }
    let converter = match converter {
        Some(c) => c,
        // No rows: the key types come from the read schema.
        None => {
            let schema = krishiv_io::project_schema(table, &columns)?;
            let empty = RecordBatch::new_empty(schema);
            let cols = key_columns(&empty, &columns)?;
            types = cols.iter().map(|c| c.data_type().clone()).collect();
            converter_for(&cols)?
        }
    };
    Ok(KeySet {
        columns,
        types,
        converter,
        keys,
        _charge: charge,
    })
}

fn converter_for(cols: &[ArrayRef]) -> Result<RowConverter, KrishivError> {
    RowConverter::new(
        cols.iter()
            .map(|c| SortField::new(c.data_type().clone()))
            .collect(),
    )
    .map_err(|e| {
        unsupported(format!(
            "Iceberg equality deletes on these column types: {e}"
        ))
    })
}

/// The columns of `batch` at `paths`, as keys: a field of a null struct is
/// null and every float `NaN` is the same `NaN`.
fn key_columns(batch: &RecordBatch, paths: &[ColumnPath]) -> Result<Vec<ArrayRef>, KrishivError> {
    paths.iter().map(|p| key_column(batch, p)).collect()
}

fn key_column(batch: &RecordBatch, path: &ColumnPath) -> Result<ArrayRef, KrishivError> {
    let missing = || internal(format!("the equality column {path} is not in the batch"));
    let mut col = Arc::clone(batch.column_by_name(path.root()).ok_or_else(missing)?);
    for name in &path.0[1..] {
        let parent = col.as_struct_opt().ok_or_else(missing)?;
        let child = Arc::clone(parent.column_by_name(name).ok_or_else(missing)?);
        col = match parent.logical_nulls() {
            Some(nulls) if nulls.null_count() > 0 => {
                let mask: BooleanArray = nulls.iter().map(|valid| Some(!valid)).collect();
                arrow_select::nullif::nullif(&child, &mask)
                    .map_err(|e| internal(format!("equality column {path}: {e}")))?
            }
            _ => child,
        };
    }
    Ok(match col.data_type() {
        DataType::Float32 => Arc::new(
            col.as_primitive::<Float32Type>()
                .unary::<_, Float32Type>(|v| if v.is_nan() { f32::NAN } else { v }),
        ),
        DataType::Float64 => Arc::new(
            col.as_primitive::<Float64Type>()
                .unary::<_, Float64Type>(|v| if v.is_nan() { f64::NAN } else { v }),
        ),
        _ => col,
    })
}

/// Whether any `_metadata` column is asked for.
pub(super) fn any_metadata(c: &MetadataCols) -> bool {
    c.file_path
        || c.file_name
        || c.file_size
        || c.file_block_start
        || c.file_block_length
        || c.file_modification_time
        || c.row_index
}

/// How the batches of a scan with equality deletes become its output: the
/// inner file scan reads the query's columns, the equality columns and
/// [`FILE_ORDINAL`]; the output holds the query's columns (`output` paths)
/// only. Applied by [`super::lazy::DeleteSource`].
pub(super) struct EqProjection {
    /// The column of [`FILE_ORDINAL`] in the inner batches.
    ordinal: usize,
    /// The output's columns, as paths into the inner batches.
    output: Vec<ColumnPath>,
    /// With only top-level output columns, their indices in the inner
    /// batches, worked out once (review 25c N2): the projection per batch is
    /// then a column pick.
    top: Option<Vec<usize>>,
    schema: SchemaRef,
}

impl EqProjection {
    /// The projection of batches of `inner` (the inner source's schema) to
    /// its `output` columns.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when `inner` has no [`FILE_ORDINAL`] or a path of
    /// `output` is not in it.
    pub(super) fn new(inner: &SchemaRef, output: Vec<ColumnPath>) -> Result<Self, KrishivError> {
        let ordinal = inner
            .index_of(FILE_ORDINAL)
            .map_err(|_| internal("the scan does not read the file ordinal".to_owned()))?;
        let schema = krishiv_io::project_schema(inner, &output)?;
        let top = output
            .iter()
            .map(|p| match p.0.as_slice() {
                [name] => inner.index_of(name).ok(),
                _ => None,
            })
            .collect();
        Ok(Self {
            ordinal,
            output,
            top,
            schema,
        })
    }

    /// The output schema.
    pub(super) fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// `batch` without the rows `deletes` delete, projected to the output.
    pub(super) fn apply(
        &self,
        deletes: &EqualityDeletes,
        batch: &RecordBatch,
    ) -> Result<RecordBatch, KrishivError> {
        let files = batch
            .column(self.ordinal)
            .as_primitive_opt::<Int32Type>()
            .ok_or_else(|| internal("the file ordinal is not an int".to_owned()))?;
        let n = batch.num_rows();
        let mut keep: Option<Vec<bool>> = None;
        let mut start = 0;
        while start < n {
            let file = files.value(start);
            let mut end = start + 1;
            while end < n && files.value(end) == file {
                end += 1;
            }
            let sets = usize::try_from(file)
                .ok()
                .and_then(|i| deletes.by_file.get(i))
                .ok_or_else(|| internal(format!("no data file {file} in the scan")))?;
            if !sets.is_empty() {
                let run = batch.slice(start, end - start);
                // The run's keys are encoded once per distinct equality
                // column list and key types (Flink writes one delete file per
                // checkpoint, all on the same columns): such sets share an
                // encoding, as Java shares one set per `equality_ids` group.
                let mut encoded: Vec<(&KeySet, arrow_row::Rows)> = Vec::new();
                for set in sets {
                    let at = encoded
                        .iter()
                        .position(|(k, _)| k.columns == set.columns && k.types == set.types);
                    let at = if let Some(i) = at {
                        i
                    } else {
                        let cols = key_columns(&run, &set.columns)?;
                        let rows = set
                            .converter
                            .convert_columns(&cols)
                            .map_err(|e| internal(format!("equality delete keys: {e}")))?;
                        encoded.push((set, rows));
                        encoded.len() - 1
                    };
                    let rows = &encoded[at].1;
                    for (r, row) in rows.iter().enumerate() {
                        if set.keys.contains(row.as_ref()) {
                            keep.get_or_insert_with(|| vec![true; n])[start + r] = false;
                        }
                    }
                }
            }
            start = end;
        }
        let kept = match keep {
            Some(k) => arrow_select::filter::filter_record_batch(batch, &BooleanArray::from(k))
                .map_err(|e| internal(format!("dropping equality-deleted rows: {e}")))?,
            None => batch.clone(),
        };
        let columns = match &self.top {
            Some(top) => top.iter().map(|&i| Arc::clone(kept.column(i))).collect(),
            None => krishiv_io::project_batch(&kept, &self.output)?
                .columns()
                .to_vec(),
        };
        // The projection keeps the inner fields; the schema is the one the
        // planner was given (same fields). The row count is explicit for an
        // output of no columns.
        RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(kept.num_rows())),
        )
        .map_err(|e| internal(format!("equality-deleted batch: {e}")))
    }
}

/// The schema `table` with [`FILE_ORDINAL`] appended.
pub(super) fn with_ordinal(table: &SchemaRef) -> SchemaRef {
    let mut fields: Vec<_> = table.fields().iter().cloned().collect();
    fields.push(Arc::new(ordinal_field()));
    Arc::new(Schema::new_with_metadata(fields, table.metadata().clone()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StructArray};
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field, Fields, Schema};
    use krishiv_io::ColumnPath;

    use super::{Keys, converter_for, key_columns};

    /// The key set indexes each distinct key once across batches and finds
    /// exactly the keys it holds (review 25c N3).
    #[test]
    fn keys_index_distinct_rows_across_batches() {
        let a: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(2)]));
        let b: ArrayRef = Arc::new(Int64Array::from(vec![Some(3), Some(1)]));
        let conv = converter_for(std::slice::from_ref(&a)).unwrap();
        let mut keys = Keys::new();
        let probe = |k: &Keys, v: Option<i64>| {
            let col: ArrayRef = Arc::new(Int64Array::from(vec![v]));
            let rows = conv.convert_columns(&[col]).unwrap();
            k.contains(rows.row(0).as_ref())
        };
        assert!(!probe(&keys, Some(1)), "an empty set holds nothing");
        keys.append(&conv, &[a]).unwrap();
        keys.append(&conv, &[b]).unwrap();
        assert_eq!(keys.index.len(), 4);
        for v in [Some(1), Some(2), Some(3), None] {
            assert!(probe(&keys, v), "{v:?}");
        }
        assert!(!probe(&keys, Some(4)));
        assert!(keys.size() > 0);
    }

    /// Java's delete set compares floats as `Double.compare`: every NaN is
    /// one value, `-0.0` is not `0.0`; a field of a null struct is null.
    #[test]
    fn keys_follow_javas_equality() {
        let nan2 = f64::from_bits(f64::NAN.to_bits() | 1);
        let x: ArrayRef = Arc::new(Float64Array::from(vec![f64::NAN, nan2, -0.0, 0.0]));
        let a: ArrayRef = Arc::new(Int64Array::from(vec![5, 5, 7, 7]));
        let s_fields = Fields::from(vec![Field::new("a", DataType::Int64, true)]);
        let s = StructArray::new(
            s_fields.clone(),
            vec![a],
            Some(NullBuffer::from(vec![true, false, true, true])),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("s", DataType::Struct(s_fields), true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![x, Arc::new(s)]).unwrap();
        let paths = [
            ColumnPath::new("x"),
            ColumnPath(vec!["s".to_owned(), "a".to_owned()]),
        ];
        let cols = key_columns(&batch, &paths[..1]).unwrap();
        let rows = converter_for(&cols)
            .unwrap()
            .convert_columns(&cols)
            .unwrap();
        assert_eq!(rows.row(0), rows.row(1));
        assert_ne!(rows.row(2), rows.row(3));
        let cols = key_columns(&batch, &paths[1..]).unwrap();
        assert!(cols[0].is_valid(0) && cols[0].is_null(1));
        let rows = converter_for(&cols)
            .unwrap()
            .convert_columns(&cols)
            .unwrap();
        assert_ne!(rows.row(0), rows.row(1));
        assert_eq!(rows.row(2), rows.row(3));
    }
}
