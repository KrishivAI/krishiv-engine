// SPDX-License-Identifier: Apache-2.0
//! The ICU rows of the `collations-basic` golden (ADR 0043): the statements listed in
//! `tests/spark-golden/icu-cases/collations-basic.json` fail with default features (the baseline
//! lists them) and must pass with `collations-icu`. Run with
//! `cargo test -p krishiv-testkit --features collations-icu --test golden_icu`.
#![forbid(unsafe_code)]
#![cfg(feature = "collations-icu")]

use krishiv_testkit::golden::manifest::ExclusionManifest;
use krishiv_testkit::golden::run::run;
use krishiv_testkit::golden::session_engine::SessionEngine;
use krishiv_testkit::golden::{DEFAULT_TAG, GoldenSet};

type R = Result<(), Box<dyn std::error::Error>>;

#[test]
fn the_icu_rows_of_collations_basic_pass() -> R {
    let list: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        krishiv_testkit::tests_dir().join("spark-golden/icu-cases/collations-basic.json"),
    )?)?;
    let wanted: Vec<u32> = list["statements"]
        .as_array()
        .ok_or("statements")?
        .iter()
        .filter_map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()))
        .collect();
    assert!(!wanted.is_empty());
    let mut set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
    set.pairs.retain(|p| p.name == "collations-basic");
    set.unpaired_inputs.clear();
    set.unpaired_outputs.clear();
    assert_eq!(set.pairs.len(), 1, "collations-basic is vendored");
    let manifest = ExclusionManifest::default();
    let mut engine = SessionEngine::new().map_err(|e| e.message)?;
    let result = run(&mut engine, &set, &manifest)?;
    let file = result.files.first().ok_or("no file result")?;
    let failing: Vec<u32> = wanted
        .iter()
        .copied()
        .filter(|i| file.cases.iter().any(|c| c.index == *i && !c.passed()))
        .collect();
    let detail: Vec<_> = file
        .cases
        .iter()
        .filter(|c| failing.contains(&c.index))
        .map(|c| format!("{}: {}", c.index, c.detail))
        .collect();
    assert!(
        failing.is_empty(),
        "ICU statements still failing: {detail:#?}"
    );
    let seen: Vec<u32> = wanted
        .iter()
        .copied()
        .filter(|i| file.cases.iter().any(|c| c.index == *i))
        .collect();
    assert_eq!(seen, wanted, "every listed statement ran");
    Ok(())
}
