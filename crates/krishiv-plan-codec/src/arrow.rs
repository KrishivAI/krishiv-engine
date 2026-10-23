// SPDX-License-Identifier: Apache-2.0
//! Arrow schemas and types as bytes: the Arrow IPC `Schema` flatbuffer,
//! which is Arrow's own stable, cross-version encoding of a schema. The
//! flatbuffer is verified on decode.

use std::sync::Arc;

use arrow_ipc::convert::{IpcSchemaEncoder, try_fb_to_schema};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::error::{Result, corrupt};

/// The IPC encoding of `schema`.
pub(crate) fn schema_bytes(schema: &Schema) -> Vec<u8> {
    IpcSchemaEncoder::new()
        .schema_to_fb(schema)
        .finished_data()
        .to_vec()
}

/// The schema `bytes` encode.
pub(crate) fn schema_from(bytes: &[u8]) -> Result<SchemaRef> {
    let fb = arrow_ipc::root_as_schema(bytes).map_err(|_| corrupt("an invalid Arrow schema"))?;
    let schema = try_fb_to_schema(fb).map_err(|_| corrupt("an invalid Arrow schema"))?;
    Ok(Arc::new(schema))
}

/// The field name a lone type is wrapped in.
const FIELD: &str = "t";

/// The IPC encoding of a schema of the one nullable field of type `t`.
pub(crate) fn type_bytes(t: &DataType) -> Vec<u8> {
    schema_bytes(&Schema::new(vec![Field::new(FIELD, t.clone(), true)]))
}

/// The type `bytes` encode.
pub(crate) fn type_from(bytes: &[u8]) -> Result<DataType> {
    let schema = schema_from(bytes)?;
    match schema.fields().iter().as_slice() {
        [field] if field.name() == FIELD => Ok(field.data_type().clone()),
        _ => Err(corrupt("an Arrow type that is not one field")),
    }
}
