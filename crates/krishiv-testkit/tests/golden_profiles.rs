// SPDX-License-Identifier: Apache-2.0
//! The Spark golden ratchet for versions other than the committed 4.2 set (P3-T090, owner 2026-11-06).
//!
//! The upstream goldens of another Spark tag are not committed: the nightly `golden-profiles` job fetches the
//! tag with `tests/scripts/import_spark_golden.sh <tag>` and runs this test with `--ignored`. What is
//! committed is the baseline of failing statements, `tests/spark-golden/profiles/<tag>/` (the same layout as
//! `tests/spark-golden/known-failures*`): a statement that newly fails is a regression, a listed statement
//! that passes is only reported, because these baselines are not refreshed by every PR that improves the
//! engine. Refresh one, any time, with
//! `cargo run --release -p krishiv-testkit --bin golden-report -- --engine session --tag <tag> --ratchet update
//!  --known-failures tests/spark-golden/profiles/<tag>/known-failures.json`.
#![forbid(unsafe_code)]

use krishiv_testkit::golden::GoldenSet;
use krishiv_testkit::golden::known_failures::KnownFailures;
use krishiv_testkit::golden::manifest::ExclusionManifest;
use krishiv_testkit::golden::run::run;
use krishiv_testkit::golden::session_engine::SessionEngine;

type R = Result<(), Box<dyn std::error::Error>>;

fn baseline_path(tag: &str) -> std::path::PathBuf {
    krishiv_testkit::tests_dir()
        .join("spark-golden")
        .join("profiles")
        .join(tag)
        .join("known-failures.json")
}

fn matches_the_baseline(tag: &str) -> R {
    let dir = GoldenSet::tag_dir(tag);
    assert!(
        dir.is_dir(),
        "{} is not fetched: run `tests/scripts/import_spark_golden.sh {tag}` first",
        dir.display()
    );
    let known = KnownFailures::load(&baseline_path(tag))?;
    assert_eq!(known.spark_tag, tag);
    let unused = known.unused_categories();
    assert!(unused.is_empty(), "categories without cases: {unused:?}");
    let manifest = ExclusionManifest::load(&ExclusionManifest::default_path())?;
    assert!(manifest.frozen, "the exclusion manifest is frozen");
    let set = GoldenSet::scan(&dir)?;
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
    // Another version's baseline is not refreshed by every PR that improves the engine (the 4.2 baseline is:
    // its ratchet runs in PR CI). So a listed statement that now passes is a note here, not a failure; a
    // statement that newly fails, or a listed file that did not run, is a regression and fails.
    if !ratchet.newly_passing.is_empty() {
        eprintln!(
            "note: {} listed statement(s) of {tag} now pass; refresh the baseline with `golden-report --engine session \
             --tag {tag} --ratchet update --known-failures tests/spark-golden/profiles/{tag}/known-failures.json`",
            ratchet.newly_passing.len()
        );
    }
    assert!(
        ratchet.newly_failing.is_empty() && ratchet.missing_files.is_empty(),
        "{}",
        ratchet.describe()
    );
    Ok(())
}

#[test]
#[ignore = "needs the fetched v4.3.0 goldens: the nightly golden-profiles job runs it with --ignored"]
fn spark_4_3_0_matches_its_baseline() -> R {
    matches_the_baseline("v4.3.0")
}

#[test]
#[ignore = "needs the fetched v4.1.4 goldens: the nightly golden-profiles job runs it with --ignored"]
fn spark_4_1_4_matches_its_baseline() -> R {
    matches_the_baseline("v4.1.4")
}

#[test]
#[ignore = "needs the fetched v4.0.5 goldens: the nightly golden-profiles job runs it with --ignored"]
fn spark_4_0_5_matches_its_baseline() -> R {
    matches_the_baseline("v4.0.5")
}

#[test]
#[ignore = "needs the fetched v3.5.9 goldens: the nightly golden-profiles job runs it with --ignored"]
fn spark_3_5_9_matches_its_baseline() -> R {
    matches_the_baseline("v3.5.9")
}
