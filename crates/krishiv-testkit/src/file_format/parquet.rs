// SPDX-License-Identifier: Apache-2.0
//! The Parquet format's fixture for the conformance suite.
//!
//! The format under test is `krishiv_io::parquet::ParquetFormat`, reads and
//! writes. The fixture adds **test-only** write options on top of the
//! product writer (`ParquetFormat::encoder_with`): cut row groups by rows
//! (`rowGroupRows`), truncate statistics (`statsTruncateLength`) and claim
//! an old writer (`createdBy`, for legacy-statistics files).

use std::sync::{Arc, LazyLock};

use arrow_schema::{DataType, SchemaRef};
use futures::future::BoxFuture;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    ByteRange, Capabilities, Decoder, Encoder, FileFormat, FileMetadata, FileRef, FormatOptions,
    FormatResult, IoContext, MetadataNeeds, OptionDefault, OptionKind, OptionSchema, OptionSpec,
    ScanRequest, UnitPlan,
};

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
            doc: "Test writer: truncate statistics to this many bytes (0: the product default).",
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

/// [`ParquetFormat`] with the fixture's test-only write options.
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
        ParquetFormat.capabilities(o)
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
        let rows = usize::try_from(o.int("rowGroupRows").unwrap_or(1 << 20)).unwrap_or(1 << 20);
        // 0 keeps the product writer's truncation; a positive length
        // overrides it (truncated-bound edge cases).
        let truncate = usize::try_from(o.int("statsTruncateLength").unwrap_or(0))
            .ok()
            .filter(|n| *n > 0);
        let created_by = o.str("createdBy").map(str::to_owned);
        Box::pin(std::future::ready(ParquetFormat::encoder_with(
            &schema,
            o,
            |p| {
                let p = p.set_max_row_group_row_count(Some(rows));
                let p = match truncate {
                    Some(n) => p.set_statistics_truncate_length(Some(n)),
                    None => p,
                };
                match created_by {
                    Some(by) => p.set_created_by(by),
                    None => p,
                }
            },
        )))
    }

    fn check_write(&self, schema: &SchemaRef, o: &FormatOptions) -> FormatResult<()> {
        ParquetFormat.check_write(schema, o)
    }
}

/// The Parquet format's fixture (see the module docs).
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
