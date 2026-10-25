// SPDX-License-Identifier: Apache-2.0
//! The JSON format passes the generic `FileFormat` conformance suite,
//! through the framework's scan loop and end to end through the executor.

use krishiv_testkit::file_format::drive::ThroughExecutor;
use krishiv_testkit::file_format::json::JsonFixture;

krishiv_testkit::file_format_conformance!(json, JsonFixture);
krishiv_testkit::file_format_conformance!(json_exec, ThroughExecutor(JsonFixture));
