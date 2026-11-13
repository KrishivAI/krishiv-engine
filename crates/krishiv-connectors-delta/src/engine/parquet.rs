// SPDX-License-Identifier: Apache-2.0
//! The kernel's `ParquetHandler` over krishiv-io's Parquet reader and writer (design note
//! P3-T021 section 3.2). The kernel uses it for checkpoints, sidecars, `stats_parsed` and CDF
//! reads; scan data files never pass through it.
//!
//! `read_parquet_files`:
//! - files are read lazily, in order, one batch (at most a row group) per pull, through
//!   `ScanStream` and `IoContext`: the query's memory pool when the context has one, and the
//!   query's cancellation checked before every pull (so between row groups);
//! - every path is confined to the table root; the footer's length is checked against
//!   `krishiv.catalog.maxMetadataBytes` before the footer is decoded
//!   (`KRISHIV_CATALOG_LIMIT_EXCEEDED`);
//! - columns are matched by field id first (`parquet.field.id` against the file's
//!   `PARQUET:field_id`), then by name, at every struct level; a missing nullable column is
//!   NULL and a missing non-nullable one an error; the reader's widenings apply (an integer to
//!   a wider one, a float to a double, an upcastable decimal, `Date32` to `timestamp_ntz`,
//!   `Int32` to `date`, `Int64` to `timestamp`; millisecond timestamps are read as
//!   microseconds by krishiv-io);
//! - the metadata columns `row_index` (0-based within the file) and `_file` (the file's URL)
//!   are filled; the row-tracking ones are unsupported;
//! - output is exactly `physical_schema`, in its order; the predicate is ignored (allowed by
//!   the contract).
//!
//! `write_parquet_file` encodes with krishiv-io's encoder (field ids from the batches' Arrow
//! metadata) and writes the whole file with one overwriting put, returning its exact size.
//! `read_parquet_footer` gives the footer's schema with its field ids.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, Int64Array, ListArray, MapArray, RecordBatch, RecordBatchOptions, StringArray,
    StructArray, new_null_array,
};
use arrow_schema::{Field, Fields, Schema};
use bytes::{Bytes, BytesMut};
use delta_kernel::expressions::PredicateRef;
use delta_kernel::schema::{
    ColumnMetadataKey, DataType, MetadataColumnSpec, MetadataValue, SchemaRef, StructField,
    StructType,
};
use delta_kernel::{
    EngineData, FileDataReadResultIterator, FileMeta, FileSize, KernelError, KernelResult,
    KernelResultIteratorStatic, ParquetFooter, ParquetHandler,
};
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    ByteRange, ColumnPath, FIELD_ID_KEY, FieldMapping, FileFormat, FileRef, FormatOptions,
    MetadataNeeds, ScanOptions, ScanRequest, ScanStream, VersionToken,
};
use url::Url;

use super::Shared;
use super::data::KrishivEngineData;
use super::eval::arrow_err;
use super::eval::types::conform;
use super::json::check_size;
use super::relabel::relabel_batch;
use super::schema::{to_arrow_schema, to_arrow_type, to_kernel_schema};
use super::storage::KrishivStorageHandler;
use crate::error::to_kernel;

/// The reserved field id of the `_file` metadata column.
const FILE_NAME_FIELD_ID: i64 = 2_147_483_646;

/// The Parquet trailer: a 4-byte footer length and `PAR1`.
const TRAILER: u64 = 8;

/// The Parquet handler of a [`super::DeltaEngine`].
#[derive(Debug)]
pub struct KrishivParquetHandler {
    shared: Arc<Shared>,
}

/// The Parquet format options (all defaults).
pub(crate) fn format_options() -> Result<FormatOptions, KrishivError> {
    Ok(ParquetFormat
        .options()
        .parse(&BTreeMap::new(), &Config::default())?
        .0)
}

/// The field id a kernel field asks for, if any.
fn kernel_field_id(f: &StructField) -> Option<i64> {
    match f.get_config_value(&ColumnMetadataKey::ParquetFieldId)? {
        MetadataValue::Number(n) => Some(*n),
        MetadataValue::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// The index in `fields` of the column for `want`: by field id, then by name.
fn find(fields: &Fields, want: &StructField) -> Option<usize> {
    if let Some(id) = kernel_field_id(want) {
        let id = id.to_string();
        if let Some(i) = fields
            .iter()
            .position(|f| f.metadata().get(FIELD_ID_KEY) == Some(&id))
        {
            return Some(i);
        }
    }
    fields.iter().position(|f| f.name() == want.name())
}

/// What a physical-schema field is filled from.
#[derive(Debug, Clone, Copy)]
enum Source {
    /// The file column at this index of the projection.
    Column(usize),
    /// Absent from the file: NULL.
    Missing,
    RowIndex,
    FilePath,
}

fn metadata_spec(f: &StructField) -> KernelResult<Option<MetadataColumnSpec>> {
    match f.get_metadata_column_spec() {
        Some(MetadataColumnSpec::RowIndex) => Ok(Some(MetadataColumnSpec::RowIndex)),
        Some(MetadataColumnSpec::FilePath) => Ok(Some(MetadataColumnSpec::FilePath)),
        Some(other) => Err(KernelError::unsupported(format!(
            "the metadata column {} in a Parquet read",
            other.text_value()
        ))),
        None if kernel_field_id(f) == Some(FILE_NAME_FIELD_ID) => {
            Ok(Some(MetadataColumnSpec::FilePath))
        }
        None => Ok(None),
    }
}

/// `array` (a file column) shaped as the kernel type `want`: struct children matched by id
/// then name, missing nullable children NULL. The exact Arrow type is set by [`conform`].
fn reconcile(array: &ArrayRef, want: &DataType, path: &str) -> KernelResult<ArrayRef> {
    match want {
        DataType::Struct(s) | DataType::Variant(s) => {
            let got = array.as_struct_opt().ok_or_else(|| {
                KernelError::generic(format!(
                    "{path}: expected a struct, got {}",
                    array.data_type()
                ))
            })?;
            let mut fields = Vec::with_capacity(s.num_fields());
            let mut columns = Vec::with_capacity(s.num_fields());
            for f in s.fields() {
                let child_path = format!("{path}.{}", f.name());
                let column = match find(got.fields(), f) {
                    Some(i) => reconcile(got.column(i), f.data_type(), &child_path)?,
                    None if f.is_nullable() => {
                        new_null_array(&to_arrow_type(f.data_type())?, got.len())
                    }
                    None => return Err(missing(&child_path)),
                };
                fields.push(Field::new(f.name(), column.data_type().clone(), true));
                columns.push(column);
            }
            Ok(Arc::new(if columns.is_empty() {
                StructArray::new_empty_fields(got.len(), got.nulls().cloned())
            } else {
                StructArray::try_new(fields.into(), columns, got.nulls().cloned())
                    .map_err(arrow_err)?
            }))
        }
        DataType::Array(a) => {
            let Some(l) = array.as_list_opt::<i32>() else {
                return conform(array, &to_arrow_type(want)?);
            };
            let values = reconcile(l.values(), a.element_type(), path)?;
            let field = Field::new("element", values.data_type().clone(), true);
            Ok(Arc::new(
                ListArray::try_new(
                    Arc::new(field),
                    l.offsets().clone(),
                    values,
                    l.nulls().cloned(),
                )
                .map_err(arrow_err)?,
            ))
        }
        DataType::Map(m) => {
            let got = array.as_map_opt().ok_or_else(|| {
                KernelError::generic(format!("{path}: expected a map, got {}", array.data_type()))
            })?;
            let keys = reconcile(got.keys(), m.key_type(), path)?;
            let values = reconcile(got.values(), m.value_type(), path)?;
            let entries = StructArray::try_new(
                vec![
                    Field::new("key", keys.data_type().clone(), false),
                    Field::new("value", values.data_type().clone(), true),
                ]
                .into(),
                vec![keys, values],
                None,
            )
            .map_err(arrow_err)?;
            let field = Field::new("key_value", entries.data_type().clone(), false);
            Ok(Arc::new(
                MapArray::try_new(
                    Arc::new(field),
                    got.offsets().clone(),
                    entries,
                    got.nulls().cloned(),
                    false,
                )
                .map_err(arrow_err)?,
            ))
        }
        DataType::Primitive(_) => conform(array, &to_arrow_type(want)?),
    }
}

fn missing(path: &str) -> KernelError {
    KernelError::generic(format!(
        "the non-nullable column {path} is missing from the Parquet file"
    ))
}

/// How one read fills `physical_schema` from a file of schema `file`.
struct Plan {
    physical: Arc<StructType>,
    output: Arc<Schema>,
    /// The file columns to read, by name.
    projection: Vec<String>,
    sources: Vec<Source>,
}

impl Plan {
    fn new(physical: &SchemaRef, file: &Schema) -> KernelResult<Self> {
        let output = Arc::new(to_arrow_schema(physical)?);
        let mut projection: Vec<String> = Vec::new();
        let mut sources = Vec::with_capacity(physical.num_fields());
        for f in physical.fields() {
            let source = match metadata_spec(f)? {
                Some(MetadataColumnSpec::RowIndex) => Source::RowIndex,
                Some(_) => Source::FilePath,
                None => match find(file.fields(), f) {
                    Some(i) => {
                        let name = file.field(i).name();
                        let at = projection
                            .iter()
                            .position(|p| p == name)
                            .unwrap_or_else(|| {
                                projection.push(name.clone());
                                projection.len() - 1
                            });
                        Source::Column(at)
                    }
                    None if f.is_nullable() => Source::Missing,
                    None => return Err(missing(f.name())),
                },
            };
            sources.push(source);
        }
        Ok(Self {
            physical: Arc::clone(physical),
            output,
            projection,
            sources,
        })
    }

    /// One output batch from the decoded `batch` (the projection, in order), whose first row
    /// is row `first_row` of the file `location`.
    fn assemble(
        &self,
        batch: &RecordBatch,
        first_row: i64,
        location: &str,
    ) -> KernelResult<RecordBatch> {
        let rows = batch.num_rows();
        let mut columns = Vec::with_capacity(self.sources.len());
        for ((source, field), target) in self
            .sources
            .iter()
            .zip(self.physical.fields())
            .zip(self.output.fields())
        {
            let column: ArrayRef = match source {
                Source::Column(i) => {
                    let got = batch.columns().get(*i).ok_or_else(|| {
                        KernelError::internal_error("a projected column is missing")
                    })?;
                    reconcile(got, field.data_type(), field.name())?
                }
                Source::Missing => new_null_array(target.data_type(), rows),
                Source::RowIndex => {
                    let end = first_row.saturating_add(i64::try_from(rows).unwrap_or(i64::MAX));
                    Arc::new(Int64Array::from_iter_values(first_row..end))
                }
                Source::FilePath => Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    location, rows,
                ))),
            };
            columns.push(conform(&column, target.data_type())?);
        }
        RecordBatch::try_new_with_options(
            Arc::clone(&self.output),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map_err(arrow_err)
    }
}

/// One file being read.
struct Open {
    stream: ScanStream,
    plan: Plan,
    location: String,
    next_row: i64,
}

/// The lazy read of a list of files.
struct Reads {
    shared: Arc<Shared>,
    physical: SchemaRef,
    files: VecDeque<FileMeta>,
    open: Option<Open>,
    done: bool,
}

/// The file `meta` names, its format options and its schema: the path confined, the footer's
/// length checked against the limit, then the footer read (through the metadata cache).
fn locate(
    shared: &Shared,
    meta: &FileMeta,
) -> KernelResult<(FileRef, FormatOptions, arrow_schema::SchemaRef)> {
    let name = meta.location.as_str();
    let path = shared.root.confine(&meta.location).map_err(to_kernel)?;
    let io = &shared.io;
    let store = &shared.store;
    let size = if meta.size > 0 {
        meta.size
    } else {
        shared
            .block_on(io.head(store, &path))?
            .ok_or_else(|| KernelError::file_not_found(name))?
            .size
    };
    if size < TRAILER {
        return Err(KernelError::generic(format!(
            "{name} is not a Parquet file ({size} bytes)"
        )));
    }
    let file = FileRef {
        store: store.clone(),
        path,
        size,
        version: VersionToken::Immutable,
        modified_micros: meta.last_modified.checked_mul(1000),
    };
    let tail = shared.block_on(io.get_tail(&file, TRAILER))?;
    let bytes = &tail.bytes;
    let footer = bytes
        .len()
        .checked_sub(8)
        .and_then(|at| bytes.get(at..at + 4))
        .and_then(|b| <[u8; 4]>::try_from(b).ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| KernelError::generic(format!("{name}: no Parquet trailer")))?;
    drop(tail);
    check_size(&shared.limits, name, u64::from(footer) + TRAILER).map_err(to_kernel)?;
    let options = format_options().map_err(to_kernel)?;
    let metadata = shared.block_on(io.read_metadata(
        &ParquetFormat,
        &file,
        MetadataNeeds::default(),
        &options,
    ))?;
    Ok((file, options, metadata.schema()))
}

impl Reads {
    fn open(&self, meta: &FileMeta) -> KernelResult<Open> {
        let shared = &self.shared;
        let io = &shared.io;
        let (file, options, schema) = locate(shared, meta)?;
        let size = file.size;
        let plan = Plan::new(&self.physical, &schema)?;
        let mut request = ScanRequest::new(
            Arc::clone(&schema),
            plan.projection.iter().map(ColumnPath::new).collect(),
        );
        request.mapping = FieldMapping::ByName {
            case_sensitive: true,
        };
        let pool: Arc<dyn krishiv_memory::MemoryPool> = match io.memory_pool() {
            Some(p) => Arc::clone(p),
            None => krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default()),
        };
        let consumer = pool.register("delta parquet read", None);
        let stream = shared.block_on(ScanStream::open(
            io,
            &ParquetFormat,
            file,
            ByteRange::whole(size),
            &request,
            &options,
            krishiv_memory::Reservation::new(pool, consumer),
            ScanOptions::default(),
        ))?;
        Ok(Open {
            stream,
            plan,
            location: meta.location.to_string(),
            next_row: 0,
        })
    }

    fn pull(&mut self) -> KernelResult<Option<RecordBatch>> {
        loop {
            if self.open.is_none() {
                let Some(meta) = self.files.pop_front() else {
                    return Ok(None);
                };
                self.open = Some(self.open(&meta)?);
            }
            let Some(open) = self.open.as_mut() else {
                continue;
            };
            // Checks the query's cancellation before every pull (a row group at most).
            match self.shared.block_on(open.stream.next_batch())? {
                Some(batch) => {
                    let out = open.plan.assemble(&batch, open.next_row, &open.location)?;
                    let rows = i64::try_from(batch.num_rows()).unwrap_or(i64::MAX);
                    open.next_row = open.next_row.saturating_add(rows);
                    return Ok(Some(out));
                }
                None => self.open = None,
            }
        }
    }
}

impl Iterator for Reads {
    type Item = KernelResult<Box<dyn EngineData>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.pull() {
            Ok(Some(batch)) => Some(Ok(Box::new(KrishivEngineData::new(batch)))),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

impl KrishivParquetHandler {
    pub(crate) fn new(shared: Arc<Shared>) -> Self {
        Self { shared }
    }

    /// The bytes of one Parquet file of `data`.
    fn encode(data: KernelResultIteratorStatic<Box<dyn EngineData>>) -> KernelResult<Bytes> {
        let mut encoder = None;
        let mut schema = None;
        let mut out = BytesMut::new();
        for chunk in data {
            let batch = KrishivEngineData::from_engine_data(chunk?)?.into_batch();
            let schema = schema.get_or_insert_with(|| batch.schema());
            let batch = relabel_batch(&batch, Arc::clone(schema)).map_err(arrow_err)?;
            let encoder = match &mut encoder {
                Some(e) => e,
                None => encoder.insert(
                    ParquetFormat::encoder_with(
                        schema,
                        &format_options().map_err(to_kernel)?,
                        |p| p,
                    )
                    .map_err(|e| KernelError::generic(e.to_string()))?,
                ),
            };
            encoder
                .write(&batch)
                .map_err(|e| KernelError::generic(e.to_string()))?;
            while let Some(b) = encoder.take_bytes() {
                out.extend_from_slice(&b);
            }
        }
        let Some(mut encoder) = encoder else {
            return Err(KernelError::generic("write_parquet_file was given no data"));
        };
        while let Some(b) = encoder.take_bytes() {
            out.extend_from_slice(&b);
        }
        let done = encoder
            .finish()
            .map_err(|e| KernelError::generic(e.to_string()))?;
        out.extend_from_slice(&done.tail);
        Ok(out.freeze())
    }
}

impl ParquetHandler for KrishivParquetHandler {
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        _predicate: Option<PredicateRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        self.shared.check_cancelled()?;
        // Refuse an unsupported metadata column before any I/O.
        for f in physical_schema.fields() {
            metadata_spec(f)?;
        }
        for f in files {
            self.shared.root.confine(&f.location).map_err(to_kernel)?;
        }
        Ok(Box::new(Reads {
            shared: Arc::clone(&self.shared),
            physical: physical_schema,
            files: files.iter().cloned().collect(),
            open: None,
            done: false,
        }))
    }

    fn write_parquet_file(
        &self,
        location: Url,
        data: KernelResultIteratorStatic<Box<dyn EngineData>>,
    ) -> KernelResult<FileSize> {
        self.shared.check_cancelled()?;
        self.shared.root.confine(&location).map_err(to_kernel)?;
        let bytes = Self::encode(data)?;
        let size = bytes.len() as u64;
        KrishivStorageHandler::new(Arc::clone(&self.shared)).put_bytes(&location, bytes, true)?;
        Ok(size)
    }

    fn read_parquet_footer(&self, file: &FileMeta) -> KernelResult<ParquetFooter> {
        self.shared.check_cancelled()?;
        let (_, _, schema) = locate(&self.shared, file)?;
        Ok(ParquetFooter {
            schema: Arc::new(to_kernel_schema(&schema)?),
        })
    }
}
