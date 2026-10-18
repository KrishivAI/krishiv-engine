// SPDX-License-Identifier: Apache-2.0
//! The Parquet reader's fixture for the conformance suite.
//!
//! The reader under test is `krishiv_io::parquet::ParquetFormat`. The suite
//! also writes files, and the product writer is a later task, so the fixture
//! wraps the reader with a **test-only** encoder: arrow-rs's `ArrowWriter`
//! over a drained in-memory buffer, with options to cut row groups
//! (`rowGroupRows`), truncate statistics (`statsTruncateLength`) and claim
//! an old writer (`createdBy`, for legacy-statistics files). Every read goes
//! through the real format.

use std::sync::{Arc, LazyLock};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, SchemaRef};
use bytes::Bytes;
use futures::future::BoxFuture;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    ByteRange, Capabilities, Decoder, Encoder, FileFormat, FileMetadata, FileRef, FormatError,
    FormatOptions, FormatResult, IoContext, MetadataNeeds, OptionDefault, OptionKind, OptionSchema,
    OptionSpec, ScanRequest, UnitPlan, WrittenFile,
};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use super::FormatFixture;

static SPECS: LazyLock<Vec<OptionSpec>> = LazyLock::new(|| {
    let mut specs = ParquetFormat.options().options.to_vec();
    specs.extend([
        OptionSpec {
            name: "rowGroupRows",
            kind: OptionKind::Int {
                min: 1,
                max: 1 << 30,
            },
            default: OptionDefault::Int(1 << 20),
            conf_fallback: None,
            metadata_key: false,
            doc: "Test writer: rows per row group.",
        },
        OptionSpec {
            name: "statsTruncateLength",
            kind: OptionKind::Int {
                min: 0,
                max: 1 << 20,
            },
            default: OptionDefault::Int(0),
            conf_fallback: None,
            metadata_key: false,
            doc: "Test writer: truncate statistics to this many bytes (0: never).",
        },
        OptionSpec {
            name: "createdBy",
            kind: OptionKind::String,
            default: OptionDefault::None,
            conf_fallback: None,
            metadata_key: false,
            doc: "Test writer: the writer name recorded in the footer.",
        },
    ]);
    specs
});

static SCHEMA: LazyLock<OptionSchema> = LazyLock::new(|| OptionSchema {
    format: "parquet",
    options: SPECS.as_slice(),
});

/// Parquet reads through [`ParquetFormat`], writes through a test encoder.
#[derive(Debug, Default, Clone, Copy)]
pub struct TestParquet;

impl FileFormat for TestParquet {
    fn name(&self) -> &'static str {
        ParquetFormat.name()
    }
    fn options(&self) -> &'static OptionSchema {
        &SCHEMA
    }
    fn capabilities(&self, o: &FormatOptions) -> Capabilities {
        let mut c = ParquetFormat.capabilities(o);
        c.write = true;
        c
    }
    fn file_suffix(&self, o: &FormatOptions) -> String {
        ParquetFormat.file_suffix(o)
    }
    fn read_metadata<'a>(
        &'a self,
        io: &'a IoContext,
        file: &'a FileRef,
        needs: MetadataNeeds,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Arc<dyn FileMetadata>>> {
        ParquetFormat.read_metadata(io, file, needs, o)
    }
    fn plan_units(
        &self,
        m: &dyn FileMetadata,
        split: &ByteRange,
        req: &ScanRequest,
    ) -> FormatResult<UnitPlan> {
        ParquetFormat.plan_units(m, split, req)
    }
    fn open<'a>(
        &'a self,
        io: &'a IoContext,
        m: Arc<dyn FileMetadata>,
        units: UnitPlan,
        req: &'a ScanRequest,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Decoder>>> {
        ParquetFormat.open(io, m, units, req, o)
    }
    fn writer<'a>(
        &'a self,
        _io: &'a IoContext,
        schema: SchemaRef,
        o: &'a FormatOptions,
    ) -> BoxFuture<'a, FormatResult<Box<dyn Encoder>>> {
        let made = (|| {
            let mut props = WriterProperties::builder().set_max_row_group_row_count(Some(
                usize::try_from(o.int("rowGroupRows").unwrap_or(1 << 20)).unwrap_or(1 << 20),
            ));
            if let Some(n) = o.int("statsTruncateLength").filter(|n| *n > 0) {
                props = props.set_statistics_truncate_length(usize::try_from(n).ok());
            }
            if let Some(by) = o.str("createdBy") {
                props = props.set_created_by(by.to_owned());
            }
            let writer = ArrowWriter::try_new(Vec::new(), schema, Some(props.build()))
                .map_err(|e| FormatError::Corrupt(e.to_string()))?;
            Ok(Box::new(TestEncoder {
                writer,
                options: o.clone(),
            }) as Box<dyn Encoder>)
        })();
        Box::pin(std::future::ready(made))
    }
}

struct TestEncoder {
    writer: ArrowWriter<Vec<u8>>,
    options: FormatOptions,
}

fn bad(e: impl std::fmt::Display) -> FormatError {
    FormatError::Corrupt(e.to_string())
}

impl Encoder for TestEncoder {
    fn write(&mut self, batch: &RecordBatch) -> FormatResult<()> {
        self.writer.write(batch).map_err(bad)
    }
    fn flush(&mut self) -> FormatResult<()> {
        self.writer.flush().map_err(bad)
    }
    fn take_bytes(&mut self) -> Option<Bytes> {
        let buf = std::mem::take(self.writer.inner_mut());
        (!buf.is_empty()).then(|| Bytes::from(buf))
    }
    fn finish(mut self: Box<Self>) -> FormatResult<WrittenFile> {
        let meta = self.writer.finish().map_err(bad)?;
        let size = self.writer.bytes_written() as u64;
        let tail = Bytes::from(std::mem::take(self.writer.inner_mut()));
        let described = ParquetFormat::describe(&Arc::new(meta), size, &self.options)?;
        Ok(WrittenFile::new(tail, described.statistics()))
    }
    fn buffered_bytes(&self) -> usize {
        self.writer.memory_size() + self.writer.inner().len()
    }
    fn written_bytes(&self) -> u64 {
        self.writer.bytes_written() as u64
    }
}

/// The Parquet reader's fixture (see the module docs).
#[derive(Debug, Clone, Copy, Default)]
pub struct ParquetFixture;

fn opt(k: &str, v: &dyn std::fmt::Display) -> (String, String) {
    (k.to_owned(), v.to_string())
}

impl FormatFixture for ParquetFixture {
    fn format(&self) -> Arc<dyn FileFormat> {
        Arc::new(TestParquet)
    }
    fn unit_rows(&self, rows: usize) -> Vec<(String, String)> {
        vec![opt("rowGroupRows", &rows)]
    }
    fn supports(&self, dt: &DataType) -> bool {
        // Parquet has no column of the null type of its own.
        *dt != DataType::Null
    }
    fn statistics_edge_cases(&self) -> Vec<(String, Vec<(String, String)>)> {
        vec![
            (
                "truncated string bounds".into(),
                vec![opt("statsTruncateLength", &1)],
            ),
            (
                "legacy-writer statistics".into(),
                vec![opt("createdBy", &"parquet-mr version 1.8.1 (build 0)")],
            ),
        ]
    }
}
