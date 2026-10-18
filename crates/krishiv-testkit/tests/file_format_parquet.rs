// SPDX-License-Identifier: Apache-2.0
//! The Parquet reader passes the generic `FileFormat` conformance suite,
//! through the framework's scan loop and end to end through the executor.

use krishiv_testkit::file_format::drive::ThroughExecutor;
use krishiv_testkit::file_format::parquet::ParquetFixture;

krishiv_testkit::file_format_conformance!(parquet, ParquetFixture);
krishiv_testkit::file_format_conformance!(parquet_exec, ThroughExecutor(ParquetFixture));
