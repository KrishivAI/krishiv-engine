// SPDX-License-Identifier: Apache-2.0
//! Spark's CSV format: [`CsvFormat`], a line-oriented tokenizer and decoder
//! with Spark's options, schema inference with Spark's `samplingRatio`
//! semantics, and an encoder with Spark's quoting rules. Compressed files
//! are `KRISHIV_UNSUPPORTED_FEATURE` for now.
//!
//! The tokenizer reproduces Univocity's handling of quotes, escapes,
//! `charToEscapeQuoteEscaping`, white space and unescaped quotes; the
//! decoder Spark's parse modes (`PERMISSIVE`, `DROPMALFORMED`, `FAILFAST`),
//! `columnNameOfCorruptRecord`, header checks and column pruning. Both are
//! checked against Spark 4.2 (`tests/fixtures/spark-4.2.0/csv.json`).

mod convert;
mod decode;
mod encode;
mod infer;
mod options;
mod sampling;
mod tokenizer;

use std::sync::Arc;

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;

use self::options::OPTIONS;
use crate::error::{FormatError, FormatResult};
use crate::format::{
    Capabilities, Decoder, Encoder, FileFormat, FileMetadata, MetadataNeeds, Splittable, UnitPlan,
};
use crate::io::{ByteRange, FileRef, IoContext};
use crate::options::{FormatOptions, OptionSchema};
use crate::request::ScanRequest;
use crate::text::TextMetadata;

/// Spark's CSV format (see the module docs).
#[derive(Debug, Default, Clone, Copy)]
pub struct CsvFormat;

/// The CSV reader's options, for callers that parse literals (see
/// [`ddl_of_string`]).
#[must_use]
pub fn option_schema() -> &'static OptionSchema {
    &OPTIONS
}

/// The DDL of a CSV literal (Spark's `schema_of_csv`; see
/// [`ddl_of_string`] for the contract).
pub use self::infer::ddl_of_string;

impl FileFormat for CsvFormat {
    fn name(&self) -> &'static str {
        "csv"
    }

    fn options(&self) -> &'static OptionSchema {
        &OPTIONS
    }

    fn capabilities(&self, o: &FormatOptions) -> Capabilities {
        let mut c = Capabilities::new(if o.bool("multiLine").unwrap_or(false) {
            Splittable::No
        } else {
            Splittable::ByCodec
        });
        c.write = true;
        // The row index of a split that starts after byte 0 counts the rows
        // before it by reading them.
        c.row_index = true;
        c
    }

    fn file_suffix(&self, _o: &FormatOptions) -> String {
        ".csv".to_owned()
    }

    fn infer_schema<'a>(
        &'a self,
        io: &'a IoContext,
        files: &'a [FileRef],
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<SchemaRef>> {
        infer::infer(io, files, o)
    }

    fn read_metadata<'a>(
        &'a self,
        _io: &'a IoContext,
        file: &'a FileRef,
        _needs: MetadataNeeds,
        _o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Arc<dyn FileMetadata>>> {
        Box::pin(std::future::ready(
            TextMetadata::of(file)
                .map(|m| Arc::new(m) as Arc<dyn FileMetadata>)
                .map_err(Into::into),
        ))
    }

    fn plan_units(
        &self,
        m: &dyn FileMetadata,
        split: &ByteRange,
        _req: &ScanRequest,
    ) -> FormatResult<UnitPlan> {
        let any: &dyn std::any::Any = m;
        let m = any
            .downcast_ref::<TextMetadata>()
            .ok_or_else(|| FormatError::Corrupt("not text metadata".to_owned()))?;
        Ok(m.plan(split, false))
    }

    fn open<'a>(
        &'a self,
        _io: &'a IoContext,
        m: Arc<dyn FileMetadata>,
        units: UnitPlan,
        req: &'a ScanRequest,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Decoder>>> {
        Box::pin(std::future::ready(
            decode::CsvDecoder::new(&m, &units, req, o).map(|d| Box::new(d) as Box<dyn Decoder>),
        ))
    }

    fn writer<'a>(
        &'a self,
        _io: &'a IoContext,
        schema: SchemaRef,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Encoder>>> {
        Box::pin(std::future::ready(
            encode::CsvEncoder::new(&schema, o).map(|e| Box::new(e) as Box<dyn Encoder>),
        ))
    }

    fn check_write(&self, schema: &SchemaRef, o: &FormatOptions) -> FormatResult<()> {
        encode::CsvEncoder::new(schema, o).map(drop)
    }
}

#[cfg(test)]
mod tests;
