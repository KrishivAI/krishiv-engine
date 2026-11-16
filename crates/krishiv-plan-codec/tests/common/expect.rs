// SPDX-License-Identifier: Apache-2.0
//! What each inventory case's header must say, one file per case: `tests/expect/<case>.txt`
//! holds the features the encoder lists and the legacy `min_reader_version` it writes, so a new
//! case adds a file and edits no shared list. Format (`#` lines are comments, kept by hand):
//!
//! ```text
//! # Why the case lists what it lists, when it is not obvious.
//! features = nested_literals, nested_types
//! legacy_reader_version = 6
//! ```
//!
//! `KRISHIV_BLESS_GOLDEN=1` writes the file of every case that has none from the encoder's
//! output (as the goldens are blessed); an existing file is never rewritten, so a change in what
//! a case lists fails until the file is edited by hand.
#![expect(clippy::panic, reason = "test code")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use krishiv_exec::physical::PhysicalPlan;
use krishiv_plan_codec::to_message;

/// One case's expected header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Expected {
    pub(crate) features: Vec<String>,
    pub(crate) legacy_reader_version: u32,
}

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/expect")
}

fn actual(plan: &PhysicalPlan) -> Expected {
    let m = to_message(plan).unwrap_or_else(|e| panic!("{e}"));
    Expected {
        features: m.features,
        legacy_reader_version: m.min_reader_version,
    }
}

fn render(e: &Expected) -> String {
    let features = if e.features.is_empty() {
        String::new()
    } else {
        format!(" {}", e.features.join(", "))
    };
    format!(
        "features ={features}\nlegacy_reader_version = {}\n",
        e.legacy_reader_version
    )
}

fn parse(name: &str, text: &str) -> Expected {
    let (mut features, mut level) = (None, None);
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            panic!("{name}.txt: `{line}` is not `key = value`");
        };
        let value = value.trim();
        match key.trim() {
            "features" => {
                features = Some(
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|f| !f.is_empty())
                        .map(str::to_owned)
                        .collect(),
                );
            }
            "legacy_reader_version" => {
                level = Some(
                    value
                        .parse()
                        .unwrap_or_else(|e| panic!("{name}.txt: legacy_reader_version: {e}")),
                );
            }
            other => panic!("{name}.txt: unknown key `{other}`"),
        }
    }
    Expected {
        features: features.unwrap_or_else(|| panic!("{name}.txt: no `features` line")),
        legacy_reader_version: level
            .unwrap_or_else(|| panic!("{name}.txt: no `legacy_reader_version` line")),
    }
}

/// Every case's expected header and what the encoder writes for it, by case name. Fails when a
/// case has no file (after a bless, when blessing) or a file names no case.
pub(crate) fn expected_and_actual(
    cases: &[(String, PhysicalPlan)],
) -> BTreeMap<String, (Expected, Expected)> {
    let dir = dir();
    let bless = std::env::var_os("KRISHIV_BLESS_GOLDEN").is_some();
    let mut out = BTreeMap::new();
    for (name, plan) in cases {
        let path = dir.join(format!("{name}.txt"));
        let got = actual(plan);
        if bless && !path.exists() {
            std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
            std::fs::write(&path, render(&got)).unwrap_or_else(|e| panic!("{e}"));
        }
        let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
            panic!(
                "{}: missing; run KRISHIV_BLESS_GOLDEN=1 cargo nextest run -p krishiv-plan-codec \
                 --test features, then check the file it writes",
                path.display()
            )
        });
        out.insert(name.clone(), (parse(name, &text), got));
    }
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.unwrap_or_else(|e| panic!("{e}")).path();
        let stem = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".txt"))
            .unwrap_or_else(|| panic!("{}: not a <case>.txt file", path.display()));
        assert!(
            out.contains_key(stem),
            "{}: no inventory case is named {stem}",
            path.display()
        );
    }
    out
}
