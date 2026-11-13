// SPDX-License-Identifier: Apache-2.0
//! The kernel's `JsonHandler` over arrow-json and `IoContext` (design note P3-T021 section 3.2),
//! and the parse limits every byte the kernel parses goes through.
//!
//! - `parse_json` and `read_json_files` decode with arrow-json into the requested schema,
//!   keeping file and row order; a missing (null) string decodes as `{}`, a row of nulls, as in
//!   the kernel's default engine (`delta_kernel` 0.30.1 `engine/arrow_utils`, Apache-2.0). A
//!   malformed value fails the call: "unparseable becomes NULL" is the `ParseJson` expression's
//!   rule (the evaluator), not the handler's.
//! - `write_json_file` writes newline-delimited JSON of the selected rows, nulls omitted except
//!   map values (`{"partitionValues":{"p":null}}`, as the kernel writes them), and with
//!   `overwrite = false` it is the commit primitive: `put_if_absent`, an existing file the
//!   kernel's `FileAlreadyExists`, a store without conditional put
//!   `KRISHIV_UNSUPPORTED_FEATURE`.
//! - Limits ([`super::DeltaLimits`], `KRISHIV_CATALOG_LIMIT_EXCEEDED`): a commit file, a parsed
//!   string or a `_last_checkpoint`/`.crc` file over `krishiv.catalog.maxMetadataBytes`; JSON
//!   nested deeper than 64 (a pre-scan, before any parser); a `metaData.schemaString` with more
//!   columns than `krishiv.catalog.maxColumns`.
//!
//! arrow-json is built without `chrono-tz` (ADR 0028), so it cannot read the zone label `UTC`; a
//! `timestamp` crosses arrow-json labelled `+00:00` (the same instants) and is relabelled
//! without a copy (the crate's `relabel` module).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch, StringArray};
use arrow_json::ReaderBuilder;
use arrow_json::writer::{
    Encoder, EncoderFactory, EncoderOptions, LineDelimited, NullableEncoder, WriterBuilder,
    make_encoder,
};
use arrow_schema::{
    ArrowError, DataType as ArrowType, FieldRef, Schema, SchemaRef as ArrowSchemaRef,
};
use bytes::Bytes;
use delta_kernel::engine_data::FilteredEngineData;
use delta_kernel::expressions::PredicateRef;
use delta_kernel::schema::{MetadataColumnSpec, SchemaRef, StructType};
use delta_kernel::{
    EngineData, FileDataReadResultIterator, FileMeta, FileSize, JsonHandler, KernelError,
    KernelResult, KernelResultIterator,
};
use krishiv_common::error::{KrishivCatalogLimitExceeded, KrishivError};
use url::Url;

use super::data::KrishivEngineData;
use super::relabel::{ARROW_UTC, relabel_batch, schema_with_zone};
use super::schema::to_arrow_schema;
use super::storage::KrishivStorageHandler;
use super::{DeltaLimits, Shared};
use crate::error::to_kernel;

/// Rows per decoded batch of a commit file.
const BATCH_ROWS: usize = 8192;

/// The zone label timestamps carry across arrow-json.
const JSON_UTC: &str = ARROW_UTC;

// ---------------------------------------------------------------------------------------------
// Limits.

fn limit(source: &str, limit: &str, value: String, config: &str) -> KrishivError {
    KrishivCatalogLimitExceeded {
        source: source.to_owned(),
        limit: limit.to_owned(),
        value,
        config: config.to_owned(),
    }
    .into()
}

/// Fails when `bytes` exceeds the metadata size limit.
///
/// # Errors
/// `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
pub fn check_size(limits: &DeltaLimits, source: &str, bytes: u64) -> Result<(), KrishivError> {
    if bytes > limits.max_metadata_bytes {
        return Err(limit(
            source,
            &format!(
                "the metadata size limit of {} bytes",
                limits.max_metadata_bytes
            ),
            format!("{bytes} bytes"),
            "\"krishiv.catalog.maxMetadataBytes\" raised",
        ));
    }
    Ok(())
}

/// Fails when the JSON text `text` nests objects and arrays deeper than the limit. A pre-scan
/// that only counts brackets outside strings; it does not validate.
///
/// # Errors
/// `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
pub fn check_depth(limits: &DeltaLimits, source: &str, text: &[u8]) -> Result<(), KrishivError> {
    let (mut depth, mut in_string, mut escaped) = (0usize, false, false);
    for &b in text {
        if in_string {
            match b {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > limits.max_depth {
                    return Err(limit(
                        source,
                        &format!("the JSON nesting depth limit of {}", limits.max_depth),
                        format!("deeper than {}", limits.max_depth),
                        "a document nested less deeply (the depth limit is fixed)",
                    ));
                }
            }
            // No reset at a newline: a pretty-printed document would undercount. Each line of
            // a well-formed commit file closes what it opens, so the depth is 0 between lines.
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

/// Columns of a Delta schema string's type (a struct's fields, recursively, nested ones
/// counted), or 0 when it is not a Delta type.
fn columns_of(t: &serde_json::Value) -> usize {
    let Some(obj) = t.as_object() else {
        return 0;
    };
    match obj.get("type").and_then(serde_json::Value::as_str) {
        Some("struct") => obj
            .get("fields")
            .and_then(serde_json::Value::as_array)
            .map_or(0, |fields| {
                fields
                    .iter()
                    .map(|f| 1 + f.get("type").map_or(0, columns_of))
                    .sum()
            }),
        Some("array") => obj.get("elementType").map_or(0, columns_of),
        Some("map") => {
            obj.get("keyType").map_or(0, columns_of) + obj.get("valueType").map_or(0, columns_of)
        }
        _ => 0,
    }
}

/// Fails when the Delta schema string `schema` (a `metaData.schemaString`) is deeper or wider
/// than the limits. A string that is not valid JSON is left to the kernel's own error.
///
/// # Errors
/// `KRISHIV_CATALOG_LIMIT_EXCEEDED`.
pub fn check_schema_string(
    limits: &DeltaLimits,
    source: &str,
    schema: &str,
) -> Result<(), KrishivError> {
    check_size(limits, source, schema.len() as u64)?;
    check_depth(limits, source, schema.as_bytes())?;
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(schema) else {
        return Ok(());
    };
    let columns = columns_of(&parsed);
    if columns > limits.max_columns {
        return Err(limit(
            source,
            &format!("the schema width limit of {} columns", limits.max_columns),
            format!("{columns} columns"),
            "\"krishiv.catalog.maxColumns\" raised",
        ));
    }
    Ok(())
}

/// Checks every `metaData.schemaString` of a decoded batch.
fn check_batch(
    limits: &DeltaLimits,
    source: &str,
    batch: &RecordBatch,
) -> Result<(), KrishivError> {
    let Some(meta) = batch
        .column_by_name("metaData")
        .and_then(|c| c.as_struct_opt())
    else {
        return Ok(());
    };
    let Some(schemas) = meta.column_by_name("schemaString") else {
        return Ok(());
    };
    let text = |row: usize| -> Option<&str> {
        if schemas.is_null(row) {
            return None;
        }
        schemas
            .as_string_opt::<i32>()
            .map(|s| s.value(row))
            .or_else(|| schemas.as_string_opt::<i64>().map(|s| s.value(row)))
            .or_else(|| schemas.as_string_view_opt().map(|s| s.value(row)))
    };
    for row in 0..schemas.len() {
        if let Some(schema) = text(row) {
            check_schema_string(limits, &format!("the schema string of {source}"), schema)?;
        }
    }
    Ok(())
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "used as `map_err(arrow_err)`"
)]
fn arrow_err(e: ArrowError) -> KernelError {
    KernelError::generic(e.to_string())
}

// ---------------------------------------------------------------------------------------------
// Decoding.

/// The decode target of `schema`: the Arrow schema without metadata columns, and where each
/// `_file` column goes.
struct Target {
    arrow: ArrowSchemaRef,
    json: ArrowSchemaRef,
    /// `(position in the output, field)` of each `_file` column.
    file_columns: Vec<(usize, FieldRef)>,
    output: ArrowSchemaRef,
}

impl Target {
    fn new(schema: &StructType) -> KernelResult<Self> {
        let output = Arc::new(to_arrow_schema(schema)?);
        let mut data_fields = Vec::new();
        let mut file_columns = Vec::new();
        for (i, (kernel, arrow)) in schema.fields().zip(output.fields()).enumerate() {
            match kernel.get_metadata_column_spec() {
                None => data_fields.push(Arc::clone(arrow)),
                Some(MetadataColumnSpec::FilePath) => file_columns.push((i, Arc::clone(arrow))),
                Some(other) => {
                    return Err(KernelError::unsupported(format!(
                        "the metadata column {} in a JSON read",
                        other.text_value()
                    )));
                }
            }
        }
        let arrow = Arc::new(Schema::new(data_fields));
        let json = Arc::new(schema_with_zone(&arrow, JSON_UTC));
        Ok(Self {
            arrow,
            json,
            file_columns,
            output,
        })
    }

    /// `batch` (decoded with [`Self::json`]) as the output schema, `_file` columns filled.
    fn finish(&self, batch: &RecordBatch, file: &str) -> KernelResult<RecordBatch> {
        let batch = relabel_batch(batch, Arc::clone(&self.arrow)).map_err(arrow_err)?;
        if self.file_columns.is_empty() {
            return Ok(batch);
        }
        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        for (at, _) in &self.file_columns {
            let names: ArrayRef = Arc::new(StringArray::from(vec![file; batch.num_rows()]));
            columns.insert(*at, names);
        }
        RecordBatch::try_new(Arc::clone(&self.output), columns).map_err(arrow_err)
    }
}

/// Decodes newline-delimited JSON `text`, one batch per [`BATCH_ROWS`] rows.
fn decode_lines(target: &Target, text: &[u8], file: &str) -> KernelResult<Vec<RecordBatch>> {
    let mut decoder = ReaderBuilder::new(Arc::clone(&target.json))
        .with_coerce_primitive(true)
        .with_batch_size(BATCH_ROWS)
        .build_decoder()
        .map_err(arrow_err)?;
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let read = decoder.decode(rest).map_err(arrow_err)?;
        if read == 0 {
            break;
        }
        rest = &rest[read..];
        if decoder.len() >= BATCH_ROWS
            && let Some(batch) = decoder.flush().map_err(arrow_err)?
        {
            out.push(target.finish(&batch, file)?);
        }
    }
    if decoder.has_partial_record() {
        return Err(KernelError::generic(format!(
            "Malformed JSON: a partial record at the end of {file}"
        )));
    }
    if let Some(batch) = decoder.flush().map_err(arrow_err)? {
        out.push(target.finish(&batch, file)?);
    }
    Ok(out)
}

/// A commit file that does not decode as Delta actions, as Spark reports it: the file's
/// `FAILED_READ_FILE.NO_HINT` (KD001), with the decoder's message as the cause. A Krishiv error
/// already named (a parse limit) passes through.
fn corrupt_commit(file: &str, error: KernelError) -> KernelError {
    let named = crate::error::from_kernel(&error, file);
    if named.error_class() != krishiv_common::error::ErrorClass::InternalError {
        return error;
    }
    to_kernel(
        KrishivError::from(krishiv_common::error::FailedReadFileNoHint {
            path: file.to_owned(),
        })
        .with_cause(named),
    )
}

/// Decodes one JSON object per string (a null string is `{}`), one output row each.
fn decode_rows<'s>(
    target: &Target,
    rows: impl Iterator<Item = Option<&'s str>>,
    count: usize,
) -> KernelResult<RecordBatch> {
    if count == 0 {
        return Ok(RecordBatch::new_empty(Arc::clone(&target.output)));
    }
    let mut decoder = ReaderBuilder::new(Arc::clone(&target.json))
        .with_coerce_primitive(true)
        .with_batch_size(count)
        .build_decoder()
        .map_err(arrow_err)?;
    for (text, row) in rows.zip(1..) {
        let line = text.unwrap_or("{}");
        let read = decoder.decode(line.as_bytes()).map_err(arrow_err)?;
        if read != line.len() || decoder.has_partial_record() || decoder.len() != row {
            return Err(KernelError::generic(format!(
                "Malformed JSON: Multiple, partial, or 0 JSON objects on row {row}"
            )));
        }
    }
    let batch = decoder
        .flush()
        .map_err(arrow_err)?
        .ok_or_else(|| KernelError::generic("Malformed JSON: nothing decoded"))?;
    target.finish(&batch, "")
}

// ---------------------------------------------------------------------------------------------
// Encoding.

/// Map values keep their nulls (`{"p":null}`); everything else uses arrow-json's encoders.
#[derive(Debug)]
struct NullValueMaps;

static MAP_OPTIONS: std::sync::LazyLock<EncoderOptions> =
    std::sync::LazyLock::new(|| EncoderOptions::default().with_explicit_nulls(true));

impl EncoderFactory for NullValueMaps {
    fn make_default_encoder<'a>(
        &self,
        field: &'a FieldRef,
        array: &'a dyn Array,
        _options: &'a EncoderOptions,
    ) -> Result<Option<NullableEncoder<'a>>, ArrowError> {
        match array.data_type() {
            ArrowType::Map(..) if array.null_count() == array.len() => Ok(Some(
                NullableEncoder::new(Box::new(NullPlaceholder), array.nulls().cloned()),
            )),
            ArrowType::Map(..) => make_encoder(field, array, &MAP_OPTIONS).map(Some),
            _ => Ok(None),
        }
    }
}

/// Every row is null, so the writer never calls it.
struct NullPlaceholder;

impl Encoder for NullPlaceholder {
    fn encode(&mut self, _idx: usize, out: &mut Vec<u8>) {
        out.extend_from_slice(b"null");
    }
}

/// The selected rows of `data` as newline-delimited JSON.
fn encode(data: KernelResultIterator<'_, FilteredEngineData>) -> KernelResult<Vec<u8>> {
    let mut writer = WriterBuilder::new()
        .with_encoder_factory(Arc::new(NullValueMaps))
        .build::<_, LineDelimited>(Vec::new());
    for chunk in data {
        let selected = chunk?.apply_selection_vector()?;
        let batch = KrishivEngineData::from_engine_data(selected)?.into_batch();
        let zoned = Arc::new(schema_with_zone(batch.schema_ref(), JSON_UTC));
        let batch = relabel_batch(&batch, zoned).map_err(arrow_err)?;
        writer.write(&batch).map_err(arrow_err)?;
    }
    writer.finish().map_err(arrow_err)?;
    Ok(writer.into_inner())
}

// ---------------------------------------------------------------------------------------------
// The handler.

/// The JSON handler of a [`super::DeltaEngine`].
#[derive(Debug)]
pub struct KrishivJsonHandler {
    shared: Arc<Shared>,
}

impl KrishivJsonHandler {
    pub(crate) fn new(shared: Arc<Shared>) -> Self {
        Self { shared }
    }

    fn storage(&self) -> KrishivStorageHandler {
        KrishivStorageHandler::new(Arc::clone(&self.shared))
    }

    /// Reads one commit file whole, with its limits.
    fn read_one(&self, file: &FileMeta, target: &Target) -> KernelResult<Vec<RecordBatch>> {
        self.shared.check_cancelled()?;
        let limits = &self.shared.limits;
        let name = file.location.as_str();
        check_size(limits, name, file.size).map_err(to_kernel)?;
        let path = self
            .shared
            .root
            .confine(&file.location)
            .map_err(to_kernel)?;
        let bytes = self
            .shared
            .block_on(self.shared.io.get_object(&self.shared.store, &path))?
            .ok_or_else(|| KernelError::file_not_found(name))?;
        check_size(limits, name, bytes.len() as u64).map_err(to_kernel)?;
        check_depth(limits, name, &bytes).map_err(to_kernel)?;
        let batches = decode_lines(target, &bytes, name).map_err(|e| corrupt_commit(name, e))?;
        for b in &batches {
            check_batch(limits, name, b).map_err(to_kernel)?;
        }
        Ok(batches)
    }
}

impl JsonHandler for KrishivJsonHandler {
    fn parse_json(
        &self,
        json_strings: Box<dyn EngineData>,
        output_schema: SchemaRef,
    ) -> KernelResult<Box<dyn EngineData>> {
        self.shared.check_cancelled()?;
        let limits = &self.shared.limits;
        let batch = KrishivEngineData::batch_of(json_strings.as_ref())?;
        if batch.num_columns() != 1 {
            return Err(KernelError::generic(format!(
                "parse_json expects one string column, got {}",
                batch.num_columns()
            )));
        }
        let column = batch.column(0);
        let strings: Vec<Option<&str>> = if let Some(s) = column.as_string_opt::<i32>() {
            s.iter().collect()
        } else if let Some(s) = column.as_string_opt::<i64>() {
            s.iter().collect()
        } else if let Some(s) = column.as_string_view_opt() {
            s.iter().collect()
        } else {
            return Err(KernelError::generic(format!(
                "Expected string array for JSON parsing, got {}",
                column.data_type()
            )));
        };
        for text in strings.iter().flatten() {
            check_size(limits, "a JSON value of the Delta log", text.len() as u64)
                .map_err(to_kernel)?;
            check_depth(limits, "a JSON value of the Delta log", text.as_bytes())
                .map_err(to_kernel)?;
        }
        let target = Target::new(&output_schema)?;
        let parsed = decode_rows(&target, strings.iter().copied(), strings.len())?;
        check_batch(limits, "a JSON value of the Delta log", &parsed).map_err(to_kernel)?;
        Ok(Box::new(KrishivEngineData::new(parsed)))
    }

    fn read_json_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        _predicate: Option<PredicateRef>,
    ) -> KernelResult<FileDataReadResultIterator> {
        self.shared.check_cancelled()?;
        let target = Target::new(&physical_schema)?;
        // Read lazily, file by file and in order, so a later file is not read when the kernel
        // stops early.
        let this = Self::new(Arc::clone(&self.shared));
        let files = files.to_vec();
        let iter = files
            .into_iter()
            .flat_map(move |file| match this.read_one(&file, &target) {
                Ok(batches) => batches
                    .into_iter()
                    .map(|b| Ok(Box::new(KrishivEngineData::new(b)) as Box<dyn EngineData>))
                    .collect::<Vec<_>>(),
                Err(e) => vec![Err(e)],
            });
        Ok(Box::new(iter))
    }

    fn write_json_file(
        &self,
        path: &Url,
        data: KernelResultIterator<'_, FilteredEngineData>,
        overwrite: bool,
    ) -> KernelResult<FileSize> {
        self.shared.check_cancelled()?;
        let bytes = encode(data)?;
        let size = bytes.len() as u64;
        self.storage()
            .put_bytes(path, Bytes::from(bytes), overwrite)?;
        Ok(size)
    }
}
