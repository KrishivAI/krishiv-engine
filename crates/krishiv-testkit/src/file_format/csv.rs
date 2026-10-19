// SPDX-License-Identifier: Apache-2.0
//! The CSV format's fixture for the conformance suite.
//!
//! CSV is read with the table's schema (it carries none), so the suite's
//! evolution check does not apply. The fixture sets the options that make a
//! round trip exact: timestamps with microseconds, no white-space trimming,
//! and a `nullValue` that cannot collide with a string (Spark reads a
//! quoted empty string as `NULL` by default).

use std::sync::Arc;

use arrow_schema::DataType;
use krishiv_io::FileFormat;
use krishiv_io::csv::CsvFormat;

use super::FormatFixture;

/// The CSV format's fixture (see the module docs).
#[derive(Debug, Clone, Copy, Default)]
pub struct CsvFixture;

fn opt(k: &str, v: &str) -> (String, String) {
    (k.to_owned(), v.to_owned())
}

impl FormatFixture for CsvFixture {
    fn format(&self) -> Arc<dyn FileFormat> {
        Arc::new(CsvFormat)
    }

    fn options(&self) -> Vec<(String, String)> {
        vec![
            opt("timestampFormat", "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX"),
            opt("timestampNTZFormat", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"),
            opt("ignoreLeadingWhiteSpace", "false"),
            opt("ignoreTrailingWhiteSpace", "false"),
            opt("nullValue", "\u{1}null\u{1}"),
            opt("multiLine", "false"),
        ]
    }

    fn unit_rows(&self, _rows: usize) -> Vec<(String, String)> {
        Vec::new()
    }

    fn footerless(&self) -> bool {
        true
    }

    fn supports(&self, dt: &DataType) -> bool {
        // Nested values and binary (written as `[AB 01]`) do not round trip.
        !matches!(
            dt,
            DataType::Struct(_)
                | DataType::List(_)
                | DataType::Map(..)
                | DataType::Binary
                | DataType::BinaryView
                | DataType::LargeBinary
        )
    }
}
