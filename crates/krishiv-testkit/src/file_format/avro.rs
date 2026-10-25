// SPDX-License-Identifier: Apache-2.0
//! The Avro format's fixture for the conformance suite.
//!
//! Avro is self-describing with block-level splits, row indexes and
//! selections, but stores no column statistics (so the pruning-vacuity
//! check does not apply) and no field ids (so the field-id check does not
//! apply). Narrow integer types widen on the next read (`byte`/`short` to
//! `int`) and the `null` type has no writable column, so the round trip
//! skips them; the Variant placeholder struct would lose its Parquet
//! extension tag, so it is skipped too.

use std::sync::Arc;

use arrow_schema::DataType;
use krishiv_io::FileFormat;
use krishiv_io::avro::AvroFormat;

use super::FormatFixture;

/// The Avro format's fixture (see the module docs).
#[derive(Debug, Clone, Copy, Default)]
pub struct AvroFixture;

impl FormatFixture for AvroFixture {
    fn format(&self) -> Arc<dyn FileFormat> {
        Arc::new(AvroFormat)
    }

    fn unit_rows(&self, _rows: usize) -> Vec<(String, String)> {
        Vec::new()
    }

    fn footerless(&self) -> bool {
        // Avro has no footer or column statistics: the header and block
        // index are read with range reads, not a tail read, and no unit is
        // pruned. The suite's tail-read, cache-hit and pruning-vacuity
        // checks do not apply.
        true
    }

    fn has_statistics(&self) -> bool {
        false
    }

    fn supports_field_ids(&self) -> bool {
        false
    }

    fn supports(&self, dt: &DataType) -> bool {
        match dt {
            // Narrow integers widen, `null` has no writable column, and maps
            // are unordered (the writer sorts keys, so key order does not
            // round-trip).
            DataType::Int8 | DataType::Int16 | DataType::Null | DataType::Map(..) => false,
            // The Variant placeholder (`struct<metadata: binary, value:
            // binary>` tagged with the Parquet extension name): Avro would
            // lose the extension tag on the round trip.
            DataType::Struct(fields)
                if fields.len() == 2
                    && fields.iter().any(|f| {
                        f.name() == "metadata" && matches!(f.data_type(), DataType::BinaryView)
                    })
                    && fields.iter().any(|f| {
                        f.name() == "value" && matches!(f.data_type(), DataType::BinaryView)
                    }) =>
            {
                false
            }
            _ => true,
        }
    }
}
