// SPDX-License-Identifier: Apache-2.0
//! The Spark golden-file ratchet: the whole vendored Spark 4.2 `SQLQueryTestSuite` runs through
//! `krishiv-session`, and the statements that do not pass must be exactly those of the frozen
//! baseline `tests/spark-golden/known-failures.json`. A statement that newly fails is a
//! regression; a listed statement that passes must leave the list (it only shrinks). Refresh
//! the list with `cargo run -p krishiv-testkit --bin golden-report -- --engine session
//! --ratchet update`.
#![forbid(unsafe_code)]

use krishiv_testkit::golden::known_failures::KnownFailures;
use krishiv_testkit::golden::manifest::ExclusionManifest;
use krishiv_testkit::golden::run::run;
use krishiv_testkit::golden::session_engine::SessionEngine;
use krishiv_testkit::golden::{DEFAULT_TAG, GoldenSet};

type R = Result<(), Box<dyn std::error::Error>>;

#[test]
fn session_engine_matches_the_frozen_baseline() -> R {
    let known = KnownFailures::load(&KnownFailures::default_path())?;
    assert_eq!(known.spark_tag, DEFAULT_TAG);
    let unused = known.unused_categories();
    assert!(unused.is_empty(), "categories without cases: {unused:?}");
    let manifest = ExclusionManifest::load(&ExclusionManifest::default_path())?;
    assert!(manifest.frozen, "the exclusion manifest is frozen");
    let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
    let mut engine = SessionEngine::new().map_err(|e| e.message)?;
    let result = run(&mut engine, &set, &manifest)?;
    assert!(result.failed_files.is_empty(), "{:?}", result.failed_files);
    let timeouts: Vec<String> = result
        .files
        .iter()
        .flat_map(|f| {
            f.timeouts
                .iter()
                .map(move |c| format!("{}#{}: {}", f.name, c.index, c.detail))
        })
        .collect();
    assert!(
        timeouts.is_empty(),
        "harness timeouts, not baseline failures: {timeouts:?}"
    );
    let ratchet = known.check(&result);
    assert!(ratchet.is_clean(), "{}", ratchet.describe());
    Ok(())
}
