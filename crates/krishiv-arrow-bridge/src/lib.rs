// SPDX-License-Identifier: Apache-2.0
//! Arrow major bridge (layer L1, a temporary FFI crate; ADR 0029 update
//! 2026-11-13): moves Arrow values from the Arrow major iceberg-rust is built
//! on (58) to the workspace's (60) through the Arrow C Data Interface.
//!
//! # Why a crate
//! iceberg-rust 0.11 depends on `arrow ^58.4`; the workspace is on Arrow 60.
//! Only the values iceberg-rust *returns as Arrow* cross: its Arrow schema
//! conversion (byte-identical field metadata such as `PARQUET:field_id`) and
//! the metadata tables (`inspect`). Data batches never cross; Krishiv reads
//! data files itself (`design/P3-lakehouse.md` §5.2). The move needs `unsafe`,
//! so it lives here and `krishiv-connectors-iceberg` stays
//! `forbid(unsafe_code)`.
//!
//! # How a value crosses
//! The Arrow 58 side exports an `FFI_ArrowSchema` / `FFI_ArrowArray`; the
//! Arrow 60 side takes it over with `from_raw`, the C Data Interface's
//! [move](https://arrow.apache.org/docs/format/CDataInterface.html#moving-an-array),
//! which leaves the exported struct released (no double free), and imports it
//! with `from_ffi`. No buffer is copied: the Arrow 60 value owns the export and
//! calls Arrow 58's release callback when dropped. Both majors' structs are
//! `#[repr(C)]` copies of the specification's, which a compile-time check and
//! the `ffi_layout` test hold them to.
//!
//! The crate is deleted once iceberg-rust and the workspace share an Arrow
//! major (apache/iceberg-rust#3257).

mod unsafe_move;

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchOptions, StructArray};
use arrow_schema::Schema;
use krishiv_common::error::{InternalError, KrishivError};

/// Arrow 58, the major iceberg-rust is built on, for callers that hold its
/// values (`krishiv-connectors-iceberg`).
pub mod arrow58 {
    pub use arrow_array_58::{RecordBatch, StructArray};
    pub use arrow_schema_58::{DataType, Field, Fields, Schema, SchemaRef};
}

/// The result of a bridge call.
pub type Result<T> = std::result::Result<T, KrishivError>;

fn arrow_error(step: &str, e: impl std::fmt::Display) -> KrishivError {
    InternalError {
        message: format!("Arrow 58 to 60 bridge, {step}: {e}"),
    }
    .into()
}

/// Moves an Arrow 58 schema to Arrow 60: fields, nesting, nullability, field
/// metadata and the schema's own metadata, unchanged.
///
/// # Errors
/// `INTERNAL_ERROR` if either side refuses the schema (a type the C Data
/// Interface cannot carry).
pub fn schema_58_to_60(schema: &arrow_schema_58::Schema) -> Result<Schema> {
    let exported = arrow_schema_58::ffi::FFI_ArrowSchema::try_from(schema)
        .map_err(|e| arrow_error("export schema", e))?;
    let moved = unsafe_move::schema(exported);
    Schema::try_from(&moved).map_err(|e| arrow_error("import schema", e))
}

/// Moves an Arrow 58 record batch to Arrow 60 without copying its buffers.
/// The schema, including its metadata, and the row count are kept (also for
/// a batch with no columns).
///
/// # Errors
/// `INTERNAL_ERROR` if the export, the import or the imported data's full
/// validation fails.
pub fn record_batch_58_to_60(batch: &arrow_array_58::RecordBatch) -> Result<RecordBatch> {
    let schema = Arc::new(schema_58_to_60(batch.schema_ref())?);
    let data = arrow_array_58::Array::into_data(arrow_array_58::StructArray::from(batch.clone()));
    let (array, c_schema) =
        arrow_array_58::ffi::to_ffi(&data).map_err(|e| arrow_error("export batch", e))?;
    let imported = unsafe_move::array(array, c_schema)?;
    // Full validation: the import trusts offsets and lengths, and these
    // batches are metadata-sized, so the check is cheap.
    imported
        .validate_full()
        .map_err(|e| arrow_error("validate batch", e))?;
    let (_, columns, _) = StructArray::from(imported).into_parts();
    RecordBatch::try_new_with_options(
        schema,
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| arrow_error("rebuild batch", e))
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;

    #[test]
    fn a_refused_value_is_an_internal_error_naming_the_step() {
        let e = super::arrow_error("import schema", "unsupported format string");
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        let text = e.to_string();
        assert!(
            text.contains("Arrow 58 to 60 bridge, import schema"),
            "{text}"
        );
        assert!(text.contains("unsupported format string"), "{text}");
    }
}
