// SPDX-License-Identifier: Apache-2.0
//! Equality deletes (P3-T026, PR-25c; design note §3.3): each applicable
//! equality delete file is read once per scan into a set of row keys, and a
//! wrapper around the file scan's source drops the decoded rows whose key is
//! in a set that applies to the row's data file.
//!
//! - **Keys.** A delete file is read projected to its `equality_ids` columns
//!   by field id, through the table's current schema, so a key written as
//!   `int` and a column read as `long` compare as `long` (type promotion).
//!   Keys are compared in Arrow's row format: `null` equals `null` (Iceberg
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
//! - The sets are held for the scan and not charged to a query pool yet
//!   (PR-25d: planning memory, the per-worker delete cache, spilling).
//! - A delete file that is missing or unreadable fails the query; an
//!   equality id that is not a primitive column of the read schema is
//!   refused (`KRISHIV_UNSUPPORTED_FEATURE`), never ignored.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int32Type};
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use futures::{StreamExt, TryStreamExt};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::{GlobalState, LocalState, Source, TaskContext};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    ByteRange, ColumnPath, FieldMapping, FileFormat as _, FileRef, FilterId, FormatOptions,
    IoContext, MetadataCols, ScanOptions, ScanRequest, ScanStream, WithoutFieldIds,
};
use krishiv_memory::{MemoryConfig, Reservation, TreePool};
use krishiv_types::batch::Batch;

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
    converter: RowConverter,
    keys: HashSet<Box<[u8]>>,
}

/// The equality deletes of a scan: for each data file (by its index in the
/// scan), the key sets that apply to it.
pub(super) struct EqualityDeletes {
    by_file: Vec<Vec<Arc<KeySet>>>,
    /// Every equality column of every set, once each.
    columns: Vec<ColumnPath>,
}

impl fmt::Debug for EqualityDeletes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EqualityDeletes")
            .field(
                "files",
                &self.by_file.iter().filter(|s| !s.is_empty()).count(),
            )
            .field("columns", &self.columns)
            .finish()
    }
}

impl EqualityDeletes {
    /// The equality columns the scan must read.
    pub(super) fn columns(&self) -> &[ColumnPath] {
        &self.columns
    }
}

/// The equality deletes of the scan's files: `deletes[i]` are the equality
/// delete files of data file `i`. Every delete file is read once.
///
/// # Errors
/// The store's error for a delete file that is missing or unreadable;
/// `KRISHIV_UNSUPPORTED_FEATURE` for a delete file that is not Parquet or
/// whose equality ids are missing, empty, or not primitive columns of the
/// read schema (`fields`).
pub(super) async fn load(
    io: &IoContext,
    locate: &dyn Fn(&str) -> Result<FileRef, KrishivError>,
    table: &SchemaRef,
    fields: &[SchemaField],
    deletes: &[&[Arc<DeleteFileEntry>]],
) -> Result<EqualityDeletes, KrishivError> {
    let mut unique: Vec<Arc<DeleteFileEntry>> = Vec::new();
    let mut slot: HashMap<String, usize> = HashMap::new();
    for d in deletes.iter().flat_map(|d| d.iter()) {
        slot.entry(d.entry.file.path.clone()).or_insert_with(|| {
            unique.push(Arc::clone(d));
            unique.len() - 1
        });
    }
    let pool: Arc<dyn krishiv_memory::MemoryPool> = TreePool::new(&MemoryConfig::default());
    let sets: Vec<Arc<KeySet>> = futures::stream::iter(unique.iter())
        .map(|d| read_keys(io, locate, &pool, table, fields, d))
        .buffered(DELETE_FILE_CONCURRENCY)
        .map_ok(Arc::new)
        .try_collect()
        .await?;
    let mut columns: Vec<ColumnPath> = Vec::new();
    for s in &sets {
        for c in &s.columns {
            if !columns.contains(c) {
                columns.push(c.clone());
            }
        }
    }
    let by_file = deletes
        .iter()
        .map(|ds| {
            ds.iter()
                .filter_map(|d| slot.get(&d.entry.file.path).map(|i| Arc::clone(&sets[*i])))
                .collect()
        })
        .collect();
    Ok(EqualityDeletes { by_file, columns })
}

/// The key set of the equality delete file `d`.
async fn read_keys(
    io: &IoContext,
    locate: &dyn Fn(&str) -> Result<FileRef, KrishivError>,
    pool: &Arc<dyn krishiv_memory::MemoryPool>,
    table: &SchemaRef,
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
    let ids = f.equality_ids.as_deref().unwrap_or_default();
    if ids.is_empty() {
        return Err(unsupported(format!(
            "reading the Iceberg equality delete file {} without equality field ids",
            f.path
        )));
    }
    let columns = ids
        .iter()
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
        .collect::<Result<Vec<_>, _>>()?;
    let mut file = locate(&f.path)?;
    file.size = f.size;
    let mut req = ScanRequest::new(Arc::clone(table), columns.clone());
    // Writers put field ids on every column of a delete file.
    req.mapping = FieldMapping::FieldIdOr(WithoutFieldIds::Refuse);
    let id = pool.register("iceberg equality deletes", None);
    let result = async {
        let mut stream = ScanStream::open(
            io,
            &ParquetFormat,
            file.clone(),
            ByteRange::whole(file.size),
            &req,
            &FormatOptions::defaults(ParquetFormat.options()),
            Reservation::new(Arc::clone(pool), id),
            ScanOptions::default(),
        )
        .await?;
        let mut converter: Option<RowConverter> = None;
        let mut keys = HashSet::new();
        while let Some(batch) = stream.next_batch().await? {
            let cols = key_columns(&batch, &columns)?;
            let conv = match &mut converter {
                Some(c) => c,
                None => converter.insert(converter_for(&cols)?),
            };
            let rows = conv
                .convert_columns(&cols)
                .map_err(|e| internal(format!("equality delete keys of {}: {e}", f.path)))?;
            for r in &rows {
                keys.insert(Box::from(r.as_ref()));
            }
        }
        let converter = match converter {
            Some(c) => c,
            // No rows: the key types come from the read schema.
            None => {
                let schema = krishiv_io::project_schema(table, &columns)?;
                let empty = RecordBatch::new_empty(schema);
                converter_for(&key_columns(&empty, &columns)?)?
            }
        };
        Ok(KeySet {
            columns,
            converter,
            keys,
        })
    }
    .await;
    pool.unregister(id);
    result
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

/// A file scan's source with the equality-deleted rows dropped (see the
/// module docs). The inner source reads the query's columns, the equality
/// columns and [`FILE_ORDINAL`]; the output holds the query's columns
/// (`output` paths) only.
pub(super) struct EqDeleteSource {
    inner: Arc<dyn Source>,
    deletes: Arc<EqualityDeletes>,
    /// The column of [`FILE_ORDINAL`] in the inner batches.
    ordinal: usize,
    /// The output's columns, as paths into the inner batches.
    output: Vec<ColumnPath>,
    schema: SchemaRef,
}

impl EqDeleteSource {
    /// The source over `inner` whose output is `output` of its columns.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when the inner source has no [`FILE_ORDINAL`] or a
    /// path of `output` is not in it.
    pub(super) fn new(
        inner: Arc<dyn Source>,
        deletes: Arc<EqualityDeletes>,
        output: Vec<ColumnPath>,
    ) -> Result<Self, KrishivError> {
        let in_schema = inner.schema();
        let ordinal = in_schema
            .index_of(FILE_ORDINAL)
            .map_err(|_| internal("the scan does not read the file ordinal".to_owned()))?;
        let schema = krishiv_io::project_schema(&in_schema, &output)?;
        Ok(Self {
            inner,
            deletes,
            ordinal,
            output,
            schema,
        })
    }

    /// `batch` without its deleted rows, projected to the output.
    fn apply(&self, batch: &RecordBatch) -> Result<RecordBatch, KrishivError> {
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
                .and_then(|i| self.deletes.by_file.get(i))
                .ok_or_else(|| internal(format!("no data file {file} in the scan")))?;
            if !sets.is_empty() {
                let run = batch.slice(start, end - start);
                for set in sets {
                    let cols = key_columns(&run, &set.columns)?;
                    let rows = set
                        .converter
                        .convert_columns(&cols)
                        .map_err(|e| internal(format!("equality delete keys: {e}")))?;
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
        let out = krishiv_io::project_batch(&kept, &self.output)?;
        // The projection keeps the inner fields; the schema is the one the
        // planner was given (same fields).
        RecordBatch::try_new(Arc::clone(&self.schema), out.columns().to_vec())
            .map_err(|e| internal(format!("equality-deleted batch: {e}")))
    }
}

impl Source for EqDeleteSource {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        self.inner.global_state(ctx)
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        self.inner.max_parallelism(global)
    }

    fn deterministic_split(&self, global: &GlobalState) -> Option<usize> {
        self.inner.deterministic_split(global)
    }

    fn partitions(&self, global: &GlobalState) -> usize {
        self.inner.partitions(global)
    }

    fn local_state(
        &self,
        ctx: &TaskContext,
        global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        self.inner.local_state(ctx, global)
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, krishiv_exec::Result<Option<Batch>>> {
        Box::pin(async move {
            loop {
                let Some(batch) = self.inner.get_data(local, global).await? else {
                    return Ok(None);
                };
                let out = self.apply(batch.data())?;
                // A batch whose rows were all deleted is skipped, not sent.
                if out.num_rows() > 0 || batch.num_rows() == 0 {
                    return Ok(Some(Batch::new(out).with_split(batch.split())));
                }
            }
        })
    }

    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        self.inner.morsels_claimed(local)
    }

    fn reports_positions(&self) -> bool {
        self.inner.reports_positions()
    }

    fn position(&self, local: &LocalState) -> Option<u64> {
        self.inner.position(local)
    }

    fn runtime_filters(&self) -> Vec<FilterId> {
        self.inner.runtime_filters()
    }

    fn extra_metrics(&self) -> krishiv_exec::ExtraMetrics {
        self.inner.extra_metrics()
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

    use super::{converter_for, key_columns};

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
