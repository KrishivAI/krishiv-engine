// SPDX-License-Identifier: Apache-2.0
//! The kernel's `JsonHandler` over Arrow 60 and `krishiv-io` (design `P3-T021-delta.md` §3.2).
//!
//! Behaviour follows the kernel's default engine (`delta_kernel` 0.30.1, Apache-2.0,
//! `engine/arrow_utils`): an unparseable timestamp, date or decimal cell becomes NULL instead of failing the
//! batch, a missing string cell decodes as `{}`, and the commit file is written through `put` with
//! `overwrite = false`, which is the atomic commit primitive (`FileAlreadyExists` is a lost race).
//!
//! Every byte parsed here is bounded by [`ParseLimits`]: file size, JSON nesting depth and schema width.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_cast::cast::{CastOptions, cast_with_options};
use arrow_json::ReaderBuilder;
use arrow_json::writer::{
    Encoder, EncoderFactory, EncoderOptions, LineDelimited, NullableEncoder, WriterBuilder,
    make_encoder,
};
use arrow_schema::{
    ArrowError, DataType as ArrowType, Field, FieldRef, Schema, SchemaRef as ArrowSchemaRef,
};
use bytes::Bytes;
use delta_kernel::engine_data::FilteredEngineData;
use delta_kernel::expressions::PredicateRef;
use delta_kernel::schema::SchemaRef;
use delta_kernel::{
    EngineData, FileDataReadResultIterator, FileMeta, FileSize, JsonHandler, KernelError,
    KernelResult, StorageHandler,
};
use url::Url;

use super::data::KrishivEngineData;
use super::schema::arrow_schema;
use super::storage::KrishivStorageHandler;

/// Bounds on what the log parsers accept (design §3.2, T-C6): `krishiv.catalog.maxMetadataBytes`,
/// a nesting depth of 64 and `krishiv.catalog.maxColumns`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseLimits {
    /// Largest JSON commit file or string cell, in bytes.
    pub max_bytes: u64,
    /// Deepest `{`/`[` nesting a document may have.
    pub max_depth: usize,
    /// Most columns a schema the handler is asked to decode may have, counting nested fields.
    pub max_columns: usize,
}

impl Default for ParseLimits {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024,
            max_depth: 64,
            max_columns: 10_000,
        }
    }
}

fn limit_exceeded(
    what: &str,
    value: impl std::fmt::Display,
    limit: impl std::fmt::Display,
) -> KernelError {
    KernelError::generic(format!(
        "KRISHIV_CATALOG_LIMIT_EXCEEDED: {what} {value} exceeds the limit {limit}"
    ))
}

/// Nesting depth of the JSON text `s`, ignoring braces inside strings. A cheap pre-scan: it does not
/// validate, it only bounds recursion before the real decoder sees the text.
fn json_depth_exceeds(s: &[u8], max: usize) -> bool {
    let (mut depth, mut in_string, mut escaped) = (0usize, false, false);
    for &b in s {
        if in_string {
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > max {
                    return true;
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    false
}

fn field_count(t: &ArrowType) -> usize {
    match t {
        ArrowType::Struct(fields) => fields.iter().map(|f| 1 + field_count(f.data_type())).sum(),
        ArrowType::List(f) | ArrowType::LargeList(f) | ArrowType::FixedSizeList(f, _) => {
            1 + field_count(f.data_type())
        }
        ArrowType::Map(f, _) => 1 + field_count(f.data_type()),
        _ => 0,
    }
}

fn check_width(schema: &ArrowSchemaRef, limits: &ParseLimits) -> KernelResult<()> {
    let width: usize = schema
        .fields()
        .iter()
        .map(|f| 1 + field_count(f.data_type()))
        .sum();
    if width > limits.max_columns {
        return Err(limit_exceeded("schema width", width, limits.max_columns));
    }
    Ok(())
}

fn arrow_error(e: ArrowError) -> KernelError {
    KernelError::generic_err(e)
}

/// Leaves arrow-json's typed decoder fails the whole batch on; they decode as strings and are safe-cast
/// back so a bad cell becomes NULL.
fn is_failure_prone(t: &ArrowType) -> bool {
    matches!(
        t,
        ArrowType::Timestamp(..)
            | ArrowType::Date32
            | ArrowType::Decimal128(..)
            | ArrowType::Decimal256(..)
    )
}

/// `t` with failure-prone leaves inside structs replaced by `Utf8`; `None` when nothing changed.
/// Arrays, maps and variants are not visited (Delta keeps no min/max stats for them).
fn stringify(t: &ArrowType) -> Option<ArrowType> {
    if is_failure_prone(t) {
        return Some(ArrowType::Utf8);
    }
    let ArrowType::Struct(fields) = t else {
        return None;
    };
    let mut changed = false;
    let new: Vec<FieldRef> = fields
        .iter()
        .map(|f| match stringify(f.data_type()) {
            Some(dt) => {
                changed = true;
                Arc::new(
                    Field::new(f.name(), dt, f.is_nullable()).with_metadata(f.metadata().clone()),
                )
            }
            None => Arc::clone(f),
        })
        .collect();
    changed.then(|| ArrowType::Struct(new.into()))
}

fn relaxed_schema(target: &ArrowSchemaRef) -> Option<ArrowSchemaRef> {
    let mut changed = false;
    let fields: Vec<FieldRef> = target
        .fields()
        .iter()
        .map(|f| match stringify(f.data_type()) {
            Some(dt) => {
                changed = true;
                Arc::new(
                    Field::new(f.name(), dt, f.is_nullable()).with_metadata(f.metadata().clone()),
                )
            }
            None => Arc::clone(f),
        })
        .collect();
    changed.then(|| Arc::new(Schema::new(fields)))
}

fn safe_cast_back(decoded: &RecordBatch, target: &ArrowSchemaRef) -> KernelResult<RecordBatch> {
    let opts = CastOptions {
        safe: true,
        ..CastOptions::default()
    };
    let rows = decoded.num_rows();
    let columns = decoded
        .columns()
        .iter()
        .zip(target.fields())
        .map(|(array, field)| {
            cast_with_options(array, field.data_type(), &opts).map_err(arrow_error)
        })
        .collect::<KernelResult<Vec<ArrayRef>>>()?;
    RecordBatch::try_new_with_options(
        Arc::clone(target),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .map_err(arrow_error)
}

/// Decode one JSON object per cell; `None` decodes as `{}` so the row stays present, all NULL.
fn decode_cells<'a>(
    cells: impl Iterator<Item = Option<&'a str>>,
    rows: usize,
    schema: ArrowSchemaRef,
) -> KernelResult<RecordBatch> {
    if rows == 0 {
        return Ok(RecordBatch::new_empty(schema));
    }
    let mut decoder = ReaderBuilder::new(schema)
        .with_batch_size(rows)
        .with_coerce_primitive(true)
        .build_decoder()
        .map_err(arrow_error)?;
    for (cell, row) in cells.zip(1usize..) {
        let line = cell.unwrap_or("{}");
        let consumed = decoder.decode(line.as_bytes()).map_err(arrow_error)?;
        if consumed != line.len() || decoder.has_partial_record() || decoder.len() != row {
            return Err(KernelError::generic(format!(
                "Malformed JSON: multiple, partial, or 0 JSON objects on row {row}"
            )));
        }
    }
    match decoder.flush().map_err(arrow_error)? {
        Some(batch) if batch.num_rows() == rows => Ok(batch),
        Some(batch) => Err(KernelError::generic(format!(
            "unexpected number of rows decoded: got {}, expected {rows}",
            batch.num_rows()
        ))),
        None => Err(KernelError::generic("Malformed JSON: nothing was decoded")),
    }
}

/// Parses the string column `strings` into `target`, one object per row.
fn parse_strings(
    strings: &dyn Array,
    target: ArrowSchemaRef,
    limits: &ParseLimits,
) -> KernelResult<RecordBatch> {
    check_width(&target, limits)?;
    let rows = strings.len();
    let cells: Vec<Option<&str>> = match strings.data_type() {
        ArrowType::Utf8 => strings.as_string::<i32>().iter().collect(),
        ArrowType::LargeUtf8 => strings.as_string::<i64>().iter().collect(),
        ArrowType::Utf8View => strings.as_string_view().iter().collect(),
        other => {
            return Err(KernelError::generic(format!(
                "expected a string column for JSON parsing, got {other}"
            )));
        }
    };
    for cell in cells.iter().flatten() {
        if cell.len() as u64 > limits.max_bytes {
            return Err(limit_exceeded("JSON value", cell.len(), limits.max_bytes));
        }
        if json_depth_exceeds(cell.as_bytes(), limits.max_depth) {
            return Err(limit_exceeded(
                "JSON nesting depth",
                "over",
                limits.max_depth,
            ));
        }
    }
    match relaxed_schema(&target) {
        None => decode_cells(cells.into_iter(), rows, target),
        Some(relaxed) => {
            let decoded = decode_cells(cells.into_iter(), rows, relaxed)?;
            safe_cast_back(&decoded, &target)
        }
    }
}

/// Keeps `null` map values (`partitionValues: {"k": null}` is what Delta logs expect); everything else uses the
/// default encoder, which omits null fields.
#[derive(Debug)]
struct NullValueMapEncoderFactory;

struct NullMapPlaceholder;

impl Encoder for NullMapPlaceholder {
    fn encode(&mut self, _idx: usize, out: &mut Vec<u8>) {
        out.extend_from_slice(b"null");
    }
}

impl EncoderFactory for NullValueMapEncoderFactory {
    fn make_default_encoder<'a>(
        &self,
        field: &'a FieldRef,
        array: &'a dyn Array,
        _options: &'a EncoderOptions,
    ) -> Result<Option<NullableEncoder<'a>>, ArrowError> {
        if !matches!(array.data_type(), ArrowType::Map(..)) {
            return Ok(None);
        }
        if array.null_count() == array.len() {
            // Every row is null: the writer uses the null buffer and never calls the placeholder.
            return Ok(Some(NullableEncoder::new(
                Box::new(NullMapPlaceholder),
                array.nulls().cloned(),
            )));
        }
        static OPTIONS: std::sync::LazyLock<EncoderOptions> =
            std::sync::LazyLock::new(|| EncoderOptions::default().with_explicit_nulls(true));
        make_encoder(field, array, &OPTIONS).map(Some)
    }
}

/// Newline-delimited JSON for the selected rows of `data`.
fn to_json_bytes(
    data: impl Iterator<Item = KernelResult<FilteredEngineData>>,
) -> KernelResult<Vec<u8>> {
    let mut writer = WriterBuilder::new()
        .with_encoder_factory(Arc::new(NullValueMapEncoderFactory))
        .build::<_, LineDelimited>(Vec::new());
    for chunk in data {
        let selected = chunk?.apply_selection_vector()?;
        writer
            .write(KrishivEngineData::extract(selected.as_ref())?)
            .map_err(arrow_error)?;
    }
    writer.finish().map_err(arrow_error)?;
    Ok(writer.into_inner())
}

/// A `JsonHandler` for one table.
#[derive(Debug, Clone)]
pub struct KrishivJsonHandler {
    storage: KrishivStorageHandler,
    limits: ParseLimits,
}

impl KrishivJsonHandler {
    /// A handler reading and committing through `storage`, bounded by `limits`.
    #[must_use]
    pub fn new(storage: KrishivStorageHandler, limits: ParseLimits) -> Self {
        Self { storage, limits }
    }

    fn read_file(&self, file: &FileMeta) -> KernelResult<Bytes> {
        if file.size > self.limits.max_bytes {
            return Err(limit_exceeded(
                "JSON file",
                file.size,
                self.limits.max_bytes,
            ));
        }
        let mut parts = self
            .storage
            .read_files(vec![(file.location.clone(), None)])?;
        let bytes = parts
            .next()
            .ok_or_else(|| KernelError::file_not_found(file.location.as_str()))??;
        // `size` comes from a listing and may be stale or zero: check what was read too.
        if bytes.len() as u64 > self.limits.max_bytes {
            return Err(limit_exceeded(
                "JSON file",
                bytes.len(),
                self.limits.max_bytes,
            ));
        }
        Ok(bytes)
    }
}

impl JsonHandler for KrishivJsonHandler {
    fn parse_json(
        &self,
        json_strings: Box<dyn EngineData>,
        output_schema: SchemaRef,
    ) -> KernelResult<Box<dyn EngineData>> {
        let batch = KrishivEngineData::extract(json_strings.as_ref())?;
        if batch.num_columns() != 1 {
            return Err(KernelError::generic(format!(
                "parse_json takes a single string column, got {}",
                batch.num_columns()
            )));
        }
        let target = Arc::new(arrow_schema(&output_schema)?);
        let parsed = parse_strings(batch.column(0).as_ref(), target, &self.limits)?;
        Ok(Box::new(KrishivEngineData::new(parsed)))
    }

    fn read_json_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        _predicate: Option<PredicateRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        if physical_schema
            .fields()
            .any(|f| f.get_metadata_column_spec().is_some())
        {
            return Err(KernelError::unsupported(
                "metadata columns when reading a JSON file",
            ));
        }
        let target = Arc::new(arrow_schema(&physical_schema)?);
        check_width(&target, &self.limits)?;
        let handler = self.clone();
        let files = files.to_vec();
        // One file at a time, in the order given; a file's batches are never merged with another's.
        let iter = files.into_iter().map(move |file| {
            let bytes = handler.read_file(&file)?;
            if json_depth_exceeds(&bytes, handler.limits.max_depth) {
                return Err(limit_exceeded(
                    "JSON nesting depth",
                    "over",
                    handler.limits.max_depth,
                ));
            }
            let lines: Vec<&str> = std::str::from_utf8(&bytes)
                .map_err(KernelError::generic_err)?
                .lines()
                .filter(|l| !l.trim().is_empty())
                .collect();
            let batch = match relaxed_schema(&target) {
                None => decode_cells(
                    lines.iter().map(|l| Some(*l)),
                    lines.len(),
                    Arc::clone(&target),
                )?,
                Some(relaxed) => {
                    let decoded =
                        decode_cells(lines.iter().map(|l| Some(*l)), lines.len(), relaxed)?;
                    safe_cast_back(&decoded, &target)?
                }
            };
            Ok(Box::new(KrishivEngineData::new(batch)) as Box<dyn EngineData>)
        });
        Ok(Box::new(iter))
    }

    fn write_json_file(
        &self,
        path: &Url,
        data: Box<dyn Iterator<Item = KernelResult<FilteredEngineData>> + Send + '_>,
        overwrite: bool,
    ) -> KernelResult<FileSize> {
        let bytes = to_json_bytes(data)?;
        let len = bytes.len() as FileSize;
        if len > self.limits.max_bytes {
            return Err(limit_exceeded("JSON file", len, self.limits.max_bytes));
        }
        self.storage.put(path, Bytes::from(bytes), overwrite)?;
        Ok(len)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int64Array, StringArray, TimestampMicrosecondArray};
    use delta_kernel::schema::{DataType, PrimitiveType, StructField, StructType};
    use krishiv_io::{IoConfig, IoScheduler, StoreId};
    use object_store::local::LocalFileSystem;
    use tokio::runtime::Handle;

    use super::*;

    fn handler(dir: &std::path::Path, handle: Handle, limits: ParseLimits) -> KrishivJsonHandler {
        let io = IoScheduler::new(IoConfig::default());
        let store = StoreId::new("local");
        io.register_store(store.clone(), Arc::new(LocalFileSystem::new()));
        let root = Url::from_directory_path(dir).unwrap();
        KrishivJsonHandler::new(
            KrishivStorageHandler::new(io.context(), store, root, handle),
            limits,
        )
    }

    fn on_pool<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::spawn(f).join().unwrap()
    }

    fn strings(rows: &[Option<&str>]) -> Box<dyn EngineData> {
        let schema = Arc::new(Schema::new(vec![Field::new("s", ArrowType::Utf8, true)]));
        let col: ArrayRef = Arc::new(StringArray::from(rows.to_vec()));
        Box::new(KrishivEngineData::new(
            RecordBatch::try_new(schema, vec![col]).unwrap(),
        ))
    }

    fn stats_schema() -> SchemaRef {
        Arc::new(
            StructType::try_new([
                StructField::nullable("n", DataType::LONG),
                StructField::nullable("ts", DataType::Primitive(PrimitiveType::Timestamp)),
            ])
            .unwrap(),
        )
    }

    #[test]
    fn a_bad_timestamp_cell_is_null_and_a_missing_cell_is_an_empty_object() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let h = handler(dir.path(), rt.handle().clone(), ParseLimits::default());
        let out = h
            .parse_json(
                strings(&[
                    Some(r#"{"n":1,"ts":"2024-01-02T03:04:05.000Z"}"#),
                    Some(r#"{"n":2,"ts":"not a time"}"#),
                    None,
                ]),
                stats_schema(),
            )
            .unwrap();
        let batch = KrishivEngineData::extract(out.as_ref()).unwrap();
        let n = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let ts = batch
            .column(1)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!((n.value(0), n.value(1), n.is_null(2)), (1, 2, true));
        assert!(ts.is_valid(0) && ts.is_null(1) && ts.is_null(2));
        assert_eq!(ts.value(0), 1_704_164_645_000_000);
    }

    #[test]
    fn two_objects_in_one_cell_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let h = handler(dir.path(), rt.handle().clone(), ParseLimits::default());
        let err = h
            .parse_json(strings(&[Some(r#"{"n":1}{"n":2}"#)]), stats_schema())
            .err()
            .unwrap();
        assert!(err.to_string().contains("Malformed JSON"), "{err}");
    }

    #[test]
    fn nesting_size_and_width_limits_raise_the_catalog_limit() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let limits = ParseLimits {
            max_bytes: 40,
            max_depth: 3,
            max_columns: 1,
        };
        let h = handler(dir.path(), rt.handle().clone(), limits);
        let deep = r#"{"n":[[[[1]]]]}"#;
        let err = h
            .parse_json(
                strings(&[Some(deep)]),
                Arc::new(
                    StructType::try_new([StructField::nullable("n", DataType::LONG)]).unwrap(),
                ),
            )
            .err()
            .unwrap();
        assert!(
            err.to_string().contains("KRISHIV_CATALOG_LIMIT_EXCEEDED"),
            "{err}"
        );
        // A brace inside a string does not count as nesting.
        let quoted = r#"{"n":"[[[[[[["}"#;
        let schema =
            Arc::new(StructType::try_new([StructField::nullable("n", DataType::STRING)]).unwrap());
        assert!(h.parse_json(strings(&[Some(quoted)]), schema).is_ok());
        let wide = h
            .parse_json(strings(&[Some("{}")]), stats_schema())
            .err()
            .unwrap();
        assert!(wide.to_string().contains("schema width"), "{wide}");
        let big = "{\"n\":1,\"pad\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"}";
        let err = h
            .parse_json(
                strings(&[Some(big)]),
                Arc::new(
                    StructType::try_new([StructField::nullable("n", DataType::LONG)]).unwrap(),
                ),
            )
            .err()
            .unwrap();
        assert!(err.to_string().contains("JSON value"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_commit_is_written_once_omits_nulls_and_reads_back_in_file_order() {
        let dir = tempfile::tempdir().unwrap();
        let h = handler(dir.path(), Handle::current(), ParseLimits::default());
        on_pool(move || {
            let schema = stats_schema();
            let batch = |ns: Vec<Option<i64>>| {
                let arrow = Arc::new(arrow_schema(&schema).unwrap());
                let n: ArrayRef = Arc::new(Int64Array::from(ns.clone()));
                let ts: ArrayRef = Arc::new(
                    TimestampMicrosecondArray::from(vec![None; ns.len()]).with_timezone("+00:00"),
                );
                Box::new(KrishivEngineData::new(
                    RecordBatch::try_new(arrow, vec![n, ts]).unwrap(),
                )) as Box<dyn EngineData>
            };
            let a = h.storage.root_join("_delta_log/00000000000000000000.json");
            let rows = FilteredEngineData::try_new(
                batch(vec![Some(1), Some(2), None]),
                vec![true, false, true],
            )
            .unwrap();
            let len = h
                .write_json_file(&a, Box::new(std::iter::once(Ok(rows))), false)
                .unwrap();
            let text =
                std::fs::read_to_string(dir.path().join("_delta_log/00000000000000000000.json"))
                    .unwrap();
            assert_eq!(text, "{\"n\":1}\n{}\n");
            assert_eq!(len, text.len() as u64);
            let again = h.write_json_file(
                &a,
                Box::new(std::iter::once(Ok(
                    FilteredEngineData::with_all_rows_selected(batch(vec![Some(9)])),
                ))),
                false,
            );
            assert!(matches!(again, Err(KernelError::FileAlreadyExists(_))));

            let b = h.storage.root_join("_delta_log/00000000000000000001.json");
            h.write_json_file(
                &b,
                Box::new(std::iter::once(Ok(
                    FilteredEngineData::with_all_rows_selected(batch(vec![Some(7)])),
                ))),
                false,
            )
            .unwrap();
            let files: Vec<FileMeta> = [&a, &b]
                .iter()
                .map(|u| h.storage.head(u).unwrap())
                .collect();
            let got: Vec<i64> = h
                .read_json_files(&files, Arc::clone(&schema), None)
                .unwrap()
                .flat_map(|d| {
                    let d = d.unwrap();
                    let b = KrishivEngineData::extract(d.as_ref()).unwrap();
                    let n = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                    (0..n.len())
                        .map(|i| if n.is_null(i) { -1 } else { n.value(i) })
                        .collect::<Vec<_>>()
                })
                .collect();
            assert_eq!(got, vec![1, -1, 7]);
        });
    }
}
