// SPDX-License-Identifier: Apache-2.0
//! Spark's JSON format: [`JsonFormat`], a flat-object tokenizer and decoder
//! with Spark's options and parse modes, schema inference with Spark's
//! `samplingRatio` semantics, and an encoder with Spark's value renderings.
//!
//! Only flat schemas are supported: a nested object or array in the
//! requested schema is `KRISHIV_UNSUPPORTED_FEATURE`. A nested value in the
//! *data* is a malformed record (nulls plus the raw record in PERMISSIVE,
//! dropped in DROPMALFORMED, failed in FAILFAST). Compressed files are
//! `KRISHIV_UNSUPPORTED_FEATURE` for now.

mod convert;
mod decode;
mod encode;
mod infer;
mod options;

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

/// Spark's JSON format for flat records (see the module docs).
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonFormat;

impl FileFormat for JsonFormat {
    fn name(&self) -> &'static str {
        "json"
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
        ".json".to_owned()
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
            decode::JsonDecoder::new(&m, &units, req, o).map(|d| Box::new(d) as Box<dyn Decoder>),
        ))
    }

    fn writer<'a>(
        &'a self,
        _io: &'a IoContext,
        schema: SchemaRef,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Encoder>>> {
        Box::pin(std::future::ready(
            encode::JsonEncoder::new(&schema, o).map(|e| Box::new(e) as Box<dyn Encoder>),
        ))
    }

    fn check_write(&self, schema: &SchemaRef, o: &FormatOptions) -> FormatResult<()> {
        encode::JsonEncoder::new(schema, o).map(drop)
    }
}

#[cfg(test)]
mod tests;
