// SPDX-License-Identifier: Apache-2.0
//! The CSV format passes the generic `FileFormat` conformance suite,
//! through the framework's scan loop and end to end through the executor.

use krishiv_testkit::file_format::csv::CsvFixture;
use krishiv_testkit::file_format::drive::ThroughExecutor;

krishiv_testkit::file_format_conformance!(csv, CsvFixture);
krishiv_testkit::file_format_conformance!(csv_exec, ThroughExecutor(CsvFixture));
