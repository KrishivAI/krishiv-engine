// SPDX-License-Identifier: Apache-2.0
//! The statement classifier (`krishiv_sql::classify`) over every statement
//! of the vendored Spark `SQLQueryTestSuite` inputs: one golden per input
//! file (`tests/fixtures/statement-classes/<input>.txt`), one line per
//! statement with its class, or `PARSE_ERROR` when it does not parse,
//! then the statement's first line. The statements are parsed with the
//! options of the file's first session, `SET`s applied in order, as the
//! parse corpus test does.
//!
//! A new or changed input adds or rewrites only its own golden: run
//! `KRISHIV_BLESS_GOLDEN=1 cargo nextest run -p krishiv-testkit --test
//! statement_classes` and review the diff.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::PathBuf;

use krishiv_sql::{classify, parse_statement};
use krishiv_testkit::golden::run::{Engine as _, load_statements};
use krishiv_testkit::golden::setup::FileSetup;
use krishiv_testkit::golden::sql_parse::ParseOnlyEngine;
use krishiv_testkit::golden::{DEFAULT_TAG, GoldenSet};

type R = Result<(), Box<dyn std::error::Error>>;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/statement-classes")
}

/// The golden file of the input `name` (`a/b.sql` → `a__b.sql.txt`).
fn golden_path(name: &str) -> PathBuf {
    golden_dir().join(format!("{}.txt", name.replace(['/', '\\'], "__")))
}

/// The first line of `sql`, at most 120 characters, as the golden shows it.
fn head(sql: &str) -> String {
    let line = sql.trim().lines().next().unwrap_or_default().trim_end();
    line.chars().take(120).collect()
}

#[test]
fn every_corpus_statement_has_its_golden_class() -> R {
    let bless = std::env::var_os("KRISHIV_BLESS_GOLDEN").is_some();
    let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
    let mut engine = ParseOnlyEngine::default();
    let mut mismatched = Vec::new();
    let mut expected_files = BTreeSet::new();
    let (mut classified, mut total) = (0_usize, 0_usize);
    for pair in &set.pairs {
        let statements = load_statements(pair)?;
        let setup = FileSetup::parse(&pair.name, &std::fs::read_to_string(&pair.sql)?);
        if let Some(session) = setup.sessions().first() {
            engine.reset(session).map_err(|e| e.message)?;
        }
        let mut out = String::new();
        for s in &statements {
            total += 1;
            let class = match parse_statement(&s.query, &engine.options()) {
                Ok(statement) => {
                    classified += 1;
                    classify(&statement).name()
                }
                Err(_) => "PARSE_ERROR",
            };
            engine.apply_set(&s.query);
            writeln!(out, "{class}\t{}", head(&s.query))?;
        }
        let path = golden_path(&pair.name);
        expected_files.insert(path.clone());
        let current = std::fs::read_to_string(&path).ok();
        if current.as_deref() != Some(out.as_str()) {
            if bless {
                std::fs::create_dir_all(golden_dir())?;
                std::fs::write(&path, &out)?;
            } else {
                mismatched.push(pair.name.clone());
            }
        }
    }
    // No golden outlives its input.
    let mut stale = Vec::new();
    std::fs::create_dir_all(golden_dir())?;
    for entry in std::fs::read_dir(golden_dir())? {
        let path = entry?.path();
        if !expected_files.contains(&path) {
            if bless {
                std::fs::remove_file(&path)?;
            } else {
                stale.push(path.display().to_string());
            }
        }
    }
    println!("statement classes: {classified} / {total} corpus statements parse and classify");
    assert!(
        mismatched.is_empty(),
        "classes differ from the goldens for {mismatched:?} (bless with KRISHIV_BLESS_GOLDEN=1 and review)"
    );
    assert!(stale.is_empty(), "goldens without an input: {stale:?}");
    assert!(classified > total / 2, "{classified} / {total}");
    Ok(())
}
