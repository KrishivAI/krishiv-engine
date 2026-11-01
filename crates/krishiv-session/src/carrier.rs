// SPDX-License-Identifier: Apache-2.0
//! The decimal carrier at the session's edges (design P2-T160): queries run on
//! the canonical carrier of each decimal's precision, what a caller hands in may
//! be on any, and what a caller reads is on `Decimal128`, as Spark sends it.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::KrishivError;
use krishiv_exec::carrier::recarried_array;
use krishiv_types::arrow::{canonical_arrow_type, client_arrow_type};

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

/// `schema` as a caller reads it.
pub(crate) fn client_schema(schema: &Schema) -> SchemaRef {
    mapped(schema, client_arrow_type)
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
    RecordBatch::try_new(Arc::clone(schema), columns).map_err(|e| internal(e.to_string()))
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
