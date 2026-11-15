// SPDX-License-Identifier: Apache-2.0
//! The Spark golden ratchet for versions other than the committed 4.2 set (P3-T090, owner 2026-11-06).
//!
//! The upstream goldens of another Spark tag are not committed: the nightly `golden-profiles` job fetches the
//! tag with `tests/scripts/import_spark_golden.sh <tag>` and runs this test with `--ignored`. What is
//! committed is the baseline of failing statements, `tests/spark-golden/profiles/<tag>/` (the same layout as
//! `tests/spark-golden/known-failures*`): a statement that newly fails is a regression, a listed statement
//! that passes is only reported, because these baselines are not refreshed by every PR that improves the
//! engine. Each profile also has `gate.json`: the phase gate's result rate for that version
//! (`result_target_percent`: 4.3 and 4.2.1 90, 4.0 and 4.1 85, 3.5 80) and a floor, the rate the last refresh
//! measured (`result_floor_per_mille`, rounded down): the run must not fall below the floor, and the job
//! summary shows each version's rate against its target, so the golden tail (plan row GT) is sized from data.
//! Refresh one, any time, with
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

/// A passing test's output is hidden in CI, so the improvement is written where a person looks: the job summary
/// (`$GITHUB_STEP_SUMMARY`, when set) with the statements, and stderr for a local run.
#[expect(
    clippy::print_stderr,
    reason = "a local run's note; CI reads the job summary"
)]
fn report_newly_passing(tag: &str, ratchet: &krishiv_testkit::golden::known_failures::Ratchet) {
    use std::io::Write;
    let refresh = format!(
        "golden-report --engine session --tag {tag} --ratchet update --known-failures \
         tests/spark-golden/profiles/{tag}/known-failures.json"
    );
    let n = ratchet.newly_passing.len();
    eprintln!(
        "note: {n} listed statement(s) of {tag} now pass; refresh the baseline with `{refresh}`"
    );
    if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY")
        && let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(path)
    {
        let _ = writeln!(
            f,
            "### Spark {tag}: {n} baseline statement(s) now pass\n\nRefresh: `{refresh}`\n"
        );
        for d in ratchet.newly_passing.iter().take(60) {
            let _ = writeln!(f, "- `{}#{}`", d.file, d.index);
        }
        if n > 60 {
            let _ = writeln!(f, "- ... and {} more", n - 60);
        }
    }
}

/// The gate file of a profile: the target result rate of the phase gate, and the floor the ratchet enforces.
struct Gate {
    target_percent: u64,
    floor_per_mille: u64,
}

fn load_gate(tag: &str) -> Result<Gate, Box<dyn std::error::Error>> {
    let path = baseline_path(tag).with_file_name("gate.json");
    let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    assert_eq!(doc["format_version"], 1, "{}", path.display());
    let field = |name: &str| {
        doc[name]
            .as_u64()
            .ok_or_else(|| format!("{}: `{name}` must be a number", path.display()))
    };
    Ok(Gate {
        target_percent: field("result_target_percent")?,
        floor_per_mille: field("result_floor_per_mille")?,
    })
}

/// The rate against the gate, to the job summary (`$GITHUB_STEP_SUMMARY`) and stderr; a passing test's output is
/// hidden in CI.
#[expect(
    clippy::print_stderr,
    reason = "a local run's note; CI reads the job summary"
)]
fn report_rate(tag: &str, passed: u64, total: u64, gate: &Gate) {
    use std::io::Write;
    let per_mille = passed * 1000 / total.max(1);
    let line = format!(
        "{tag}: result {passed} / {total} = {}.{}% (gate {}%, floor {}.{}%; {} statements to the gate)",
        per_mille / 10,
        per_mille % 10,
        gate.target_percent,
        gate.floor_per_mille / 10,
        gate.floor_per_mille % 10,
        (gate.target_percent * total)
            .div_ceil(100)
            .saturating_sub(passed),
    );
    eprintln!("{line}");
    if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY")
        && let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(path)
    {
        let _ = writeln!(f, "- {line}");
    }
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
    let manifest = manifest.restricted_to(&set);
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
    let gate = load_gate(tag)?;
    let (_, _, rate) = krishiv_testkit::golden::report::totals(&result);
    report_rate(tag, rate.passed, rate.total, &gate);
    assert!(
        rate.passed * 1000 >= gate.floor_per_mille * rate.total,
        "{tag}: the result rate {} / {} fell below the floor of {} per mille in gate.json",
        rate.passed,
        rate.total,
        gate.floor_per_mille
    );
    let ratchet = known.check(&result);
    // Another version's baseline is not refreshed by every PR that improves the engine (the 4.2 baseline is:
    // its ratchet runs in PR CI). So a listed statement that now passes is a note here, not a failure; a
    // statement that newly fails, or a listed file that did not run, is a regression and fails.
    if !ratchet.newly_passing.is_empty() {
        report_newly_passing(tag, &ratchet);
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
#[ignore = "needs the fetched v4.2.1 goldens: the nightly golden-profiles job runs it with --ignored"]
fn spark_4_2_1_matches_its_baseline() -> R {
    matches_the_baseline("v4.2.1")
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
