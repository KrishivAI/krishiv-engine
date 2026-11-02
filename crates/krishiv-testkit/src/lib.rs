// SPDX-License-Identifier: Apache-2.0
//! Conformance test harness for Krishiv (P0-T20..T23).
//!
//! * [`hash_fixture`]: Spark `hash()`/`xxhash64()` golden fixtures (ADR 0010).
//! * [`fn_matrix`]: Spark function-matrix fixtures and the coverage-matrix report (P2-T106a).
//! * [`file_format`]: the generic `FileFormat` conformance suite and a mock format (P2-T117).
//! * [`kernel_eval`]: runs the arithmetic/comparison/cast matrix cells on `krishiv-kernels` (P2-T021a).
//! * [`batch_stream`]: batch == bounded-stream invariant harness (ADR 0015/0017).
//! * [`golden`]: Spark `SQLQueryTestSuite` golden-file runner, exclusion manifest and three-number report.
//! * [`manifest_dir`]: split test manifests, one small JSON file per key.
//! * [`slt`]: sqllogictest runner stub.
#![forbid(unsafe_code)]

pub mod batch_stream;
pub mod expr_diff;
pub mod file_format;
pub mod fn_matrix;
pub mod golden;
pub mod hash_fixture;
pub mod kernel_eval;
pub mod manifest_dir;
pub mod metamorphic;
pub mod query_oracle;
pub mod slt;

use std::path::PathBuf;

/// Absolute path of the repo-level `tests/` directory.
#[must_use]
pub fn tests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests")
}
