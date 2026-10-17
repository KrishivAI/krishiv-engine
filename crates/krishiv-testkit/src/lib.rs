// SPDX-License-Identifier: Apache-2.0
//! Conformance test harness for Krishiv (P0-T20..T23).
//!
//! * [`hash_fixture`]: Spark `hash()`/`xxhash64()` golden fixtures (ADR 0010).
//! * [`batch_stream`]: batch == bounded-stream invariant harness (ADR 0015/0017).
//! * [`golden`]: Spark `SQLQueryTestSuite` golden-file enumeration.
//! * [`slt`]: sqllogictest runner stub.
#![forbid(unsafe_code)]

pub mod batch_stream;
pub mod golden;
pub mod hash_fixture;
pub mod slt;

use std::path::PathBuf;

/// Absolute path of the repo-level `tests/` directory.
#[must_use]
pub fn tests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests")
}
