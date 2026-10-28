// SPDX-License-Identifier: Apache-2.0
//! The Avro format passes the generic `FileFormat` conformance suite,
//! through the framework's scan loop and end to end through the executor.

use krishiv_testkit::file_format::avro::AvroFixture;
use krishiv_testkit::file_format::drive::ThroughExecutor;

krishiv_testkit::file_format_conformance!(avro, AvroFixture);
krishiv_testkit::file_format_conformance!(avro_exec, ThroughExecutor(AvroFixture));
