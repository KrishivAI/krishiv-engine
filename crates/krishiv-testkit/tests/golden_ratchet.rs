// SPDX-License-Identifier: Apache-2.0
//! The Spark golden-file ratchet: the whole vendored Spark 4.2 `SQLQueryTestSuite` runs through
//! `krishiv-session`, and the statements that do not pass must be exactly those of the frozen
//! baseline `tests/spark-golden/known-failures.json` (with its per-category and per-file
//! directories). A statement that newly fails is a
//! regression; a listed statement that passes must leave the list (it only shrinks). Refresh
//! the list with `cargo run -p krishiv-testkit --bin golden-report -- --engine session
//! --ratchet update`. The nightly serialize-through job runs the same ratchet with every plan
//! encoded to `krishiv.plan.v1` and decoded before it runs (P2-T173).
#![forbid(unsafe_code)]
// The baseline no longer lists the ICU rows of `collations-basic` (ICU collations are default-on,
// ADR 0043 D5), so the ratchet runs with `collations-icu`, which is a default feature here. Without
// it those rows fail and the ratchet is compiled out; `tests/golden_icu.rs` covers the ICU rows.
#![cfg(feature = "collations-icu")]

use krishiv_testkit::golden::known_failures::KnownFailures;
use krishiv_testkit::golden::manifest::ExclusionManifest;
use krishiv_testkit::golden::run::run;
use krishiv_testkit::golden::session_engine::SessionEngine;
use krishiv_testkit::golden::{DEFAULT_TAG, GoldenSet};

type R = Result<(), Box<dyn std::error::Error>>;

#[test]
fn session_engine_matches_the_frozen_baseline() -> R {
    matches_the_frozen_baseline(SessionEngine::new().map_err(|e| e.message)?)
}

/// The same baseline with `krishiv.exec.debug.planRoundTrip` on: the wire format carries every
/// plan the goldens run, and the results are those of the plan as planned.
#[test]
#[ignore = "nightly serialize-through job (P2-T173); run with --ignored"]
fn serialize_through_matches_the_frozen_baseline() -> R {
    matches_the_frozen_baseline(
        SessionEngine::new()
            .map_err(|e| e.message)?
            .with_plan_round_trip(),
    )
}

fn matches_the_frozen_baseline(mut engine: SessionEngine) -> R {
    let known = KnownFailures::load(&KnownFailures::default_path())?;
    assert_eq!(known.spark_tag, DEFAULT_TAG);
    let unused = known.unused_categories();
    assert!(unused.is_empty(), "categories without cases: {unused:?}");
    let manifest = ExclusionManifest::load(&ExclusionManifest::default_path())?;
    assert!(manifest.frozen, "the exclusion manifest is frozen");
    let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
    manifest.validate(&set)?;
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
