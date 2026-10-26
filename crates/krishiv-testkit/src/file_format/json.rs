// SPDX-License-Identifier: Apache-2.0
//! The JSON format's fixture for the conformance suite.
//!
//! JSON is read with the table's schema (it carries none), so the suite's
//! evolution check applies only to the flat renames it supports. The fixture
//! sets the options that make a round trip exact: timestamps with
//! microseconds.

use std::sync::Arc;

use arrow_schema::DataType;
use krishiv_io::FileFormat;
use krishiv_io::json::JsonFormat;

use super::FormatFixture;

/// The JSON format's fixture (see the module docs).
#[derive(Debug, Clone, Copy, Default)]
pub struct JsonFixture;

fn opt(k: &str, v: &str) -> (String, String) {
    (k.to_owned(), v.to_owned())
}

impl FormatFixture for JsonFixture {
    fn format(&self) -> Arc<dyn FileFormat> {
        Arc::new(JsonFormat)
    }

    fn options(&self) -> Vec<(String, String)> {
        vec![
            opt("timestampFormat", "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX"),
            opt("timestampNTZFormat", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"),
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
        // Nested values have no flat representation.
        !matches!(
            dt,
            DataType::Struct(_) | DataType::List(_) | DataType::Map(..)
        )
    }
}
