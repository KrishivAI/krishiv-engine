// SPDX-License-Identifier: Apache-2.0
//! The decimal carrier at the session's edges (design P2-T160): queries run on
//! the canonical carrier of each decimal's precision, what a caller hands in may
//! be on any, and what a caller reads is on `Decimal128`, as Spark sends it.
//! A calendar interval runs as its struct of microseconds and is read as Arrow's
//! `Interval(MonthDayNano)` (ADR 0009 update 2026-11-06).
//! A registered table's strings and binaries run on the view carriers the analyzer types them
//! with, at every depth, as a file scan's do.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::KrishivError;
use krishiv_exec::carrier::recarried_array;
use krishiv_types::arrow::{canonical_arrow_type, edge_arrow_type};

use crate::session::{RecordBatchStream, internal};

/// `schema` with each field's type mapped by `to`.
fn mapped(schema: &Schema, to: fn(&DataType) -> DataType) -> SchemaRef {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone().with_data_type(to(f.data_type())))
        .collect();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// The view carrier of `t` for encoding-only differences: `Utf8`/`LargeUtf8` as `Utf8View` and
/// `Binary`/`LargeBinary` as `BinaryView`, at any depth.
pub(crate) fn viewed(t: &DataType) -> DataType {
    match t {
        DataType::Utf8 | DataType::LargeUtf8 => DataType::Utf8View,
        DataType::Binary | DataType::LargeBinary => DataType::BinaryView,
        other => nested_viewed(other),
    }
}

/// `t` with every string and binary below its top level on the view carrier ([`viewed`]) and
/// every list element and map entry under the field names the analyzer types them with
/// (`element`; `entries`, `key`, `value`): pyarrow names a list's element `item`, and arrow-rs's
/// `MapBuilder` its map fields `keys` and `values`. A top-level string or binary keeps its carrier.
fn nested_viewed(t: &DataType) -> DataType {
    let inner = |f: &Field| match f.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
            viewed(f.data_type())
        }
        other => nested_viewed(other),
    };
    let named = |f: &Field, name: &str| f.clone().with_name(name).with_data_type(inner(f));
    match t {
        DataType::List(f) => DataType::List(Arc::new(named(f, "element"))),
        DataType::LargeList(f) => DataType::LargeList(Arc::new(named(f, "element"))),
        DataType::FixedSizeList(f, n) => DataType::FixedSizeList(Arc::new(named(f, "element")), *n),
        DataType::Map(entries, sorted) => {
            let DataType::Struct(kv) = entries.data_type() else {
                return t.clone();
            };
            let kv: Vec<Field> = kv
                .iter()
                .zip(["key", "value"])
                .map(|(f, name)| named(f, name))
                .collect();
            DataType::Map(
                Arc::new(
                    entries
                        .as_ref()
                        .clone()
                        .with_name("entries")
                        .with_data_type(DataType::Struct(kv.into())),
                ),
                *sorted,
            )
        }
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|f| f.as_ref().clone().with_data_type(inner(f)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `batches` (of `schema`) with every string and binary on the view carrier and nested fields
/// under the analyzer's names ([`viewed`]): a `createDataFrame` table (pandas, pyarrow, rows) or
/// one a Rust caller registers holds `Utf8`/`LargeUtf8`, lists of `item` and maps of `keys`, where
/// kernels that build nested values read the analyzer's types (R1-D-13). A vector carrier is
/// kept as it is.
pub(crate) fn as_views(
    schema: &SchemaRef,
    batches: Vec<RecordBatch>,
) -> Result<(SchemaRef, Vec<RecordBatch>), KrishivError> {
    let viewed = mapped(schema, |t| {
        if krishiv_types::arrow::is_vector_carrier(t) {
            t.clone()
        } else {
            viewed(t)
        }
    });
    if &viewed == schema {
        return Ok((Arc::clone(schema), batches));
    }
    let batches = batches
        .iter()
        .map(|b| {
            let columns = b
                .columns()
                .iter()
                .zip(viewed.fields())
                .map(|(c, f)| {
                    if c.data_type() == f.data_type() {
                        Ok(Arc::clone(c))
                    } else {
                        arrow_cast::cast(c, f.data_type()).map_err(|e| internal(e.to_string()))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            RecordBatch::try_new_with_options(
                Arc::clone(&viewed),
                columns,
                &arrow_array::RecordBatchOptions::new().with_row_count(Some(b.num_rows())),
            )
            .map_err(|e| internal(e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((viewed, batches))
}

/// `schema` as a caller reads it.
pub(crate) fn client_schema(schema: &Schema) -> SchemaRef {
    mapped(schema, edge_arrow_type)
}

/// `schema` as queries read it.
pub(crate) fn canonical_schema(schema: SchemaRef) -> SchemaRef {
    let canonical = mapped(&schema, canonical_arrow_type);
    if canonical == schema {
        schema
    } else {
        canonical
    }
}

/// `batch` under `schema`, which differs at most in decimal carriers.
pub(crate) fn recarried(
    batch: &RecordBatch,
    schema: &SchemaRef,
) -> Result<RecordBatch, KrishivError> {
    if batch.schema_ref() == schema {
        return Ok(batch.clone());
    }
    let columns = batch
        .columns()
        .iter()
        .zip(schema.fields())
        .map(|(c, f)| recarried_array(c, f.data_type()))
        .collect::<Result<Vec<_>, _>>()?;
    let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
    RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options)
        .map_err(|e| internal(e.to_string()))
}

/// `batch` as a caller reads it.
pub(crate) fn client_batch(batch: &RecordBatch) -> Result<RecordBatch, KrishivError> {
    recarried(batch, &client_schema(batch.schema_ref()))
}

/// `stream`'s batches as a caller reads them.
pub(crate) fn client_stream(stream: RecordBatchStream) -> RecordBatchStream {
    stream
        .map(|batch| batch.and_then(|b| client_batch(&b)))
        .boxed()
}
