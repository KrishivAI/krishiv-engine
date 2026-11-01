// SPDX-License-Identifier: Apache-2.0
//! Random TPC-H queries against DuckDB (P2-T110a): `tests/fixtures/query-oracle/queries.json`
//! (`tests/scripts/gen_query_oracle.py`) replayed on `krishiv-session` over the SF 0.01 Parquet
//! files. Queries that disagree are listed in `known-failures.json` beside the fixture: a new one
//! fails the test, and so does a listed one that now agrees (`KRISHIV_UPDATE_MANIFEST=1` rewrites
//! the list). The nightly runs a larger fixture (`KRISHIV_QUERY_ORACLE`, one hour of queries).
#![forbid(unsafe_code)]

use krishiv_testkit::expr_diff::{parse_known, ratchet_keys, render_known_keys};
use krishiv_testkit::query_oracle::{Query, QueryRunner, SessionRunner, normalise, parse, run};

type R = Result<(), Box<dyn std::error::Error>>;

fn dir() -> std::path::PathBuf {
    krishiv_testkit::tests_dir().join("fixtures/query-oracle")
}

fn data() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../krishiv-exec/tests/fixtures/tpch-sf0.01-parquet")
}

fn queries() -> Result<Vec<Query>, Box<dyn std::error::Error>> {
    let path = std::env::var_os("KRISHIV_QUERY_ORACLE")
        .map_or_else(|| dir().join("queries.json"), Into::into);
    Ok(parse(&std::fs::read_to_string(path)?)?)
}

#[test]
fn matches_duckdb_up_to_known_failures() -> R {
    let qs = queries()?;
    let mut runner = SessionRunner::new(&data())?;
    let found = run(&qs, &mut runner);
    let keys: Vec<String> = found.iter().map(|d| d.sql.clone()).collect();
    let known_path = dir().join("known-failures.json");
    if std::env::var_os("KRISHIV_UPDATE_MANIFEST").is_some() {
        std::fs::write(&known_path, render_known_keys(&keys))?;
        return Ok(());
    }
    let r = ratchet_keys(&keys, &parse_known(&std::fs::read_to_string(&known_path)?)?);
    let detail: Vec<String> = found
        .iter()
        .filter(|d| r.new.contains(&d.sql))
        .map(|d| format!("{}\n  {}", d.sql, d.detail))
        .collect();
    assert!(
        r.new.is_empty(),
        "new disagreements:\n{}",
        detail.join("\n")
    );
    assert!(
        r.fixed.is_empty(),
        "listed queries that now agree, remove them:\n{}",
        r.fixed.join("\n")
    );
    Ok(())
}

/// DuckDB's own rows, played back.
struct Replay<'a>(&'a [Query]);

impl QueryRunner for Replay<'_> {
    fn run(&mut self, sql: &str) -> Result<Vec<Vec<String>>, String> {
        self.0
            .iter()
            .find(|q| q.sql == sql)
            .map(|q| q.rows.clone())
            .ok_or_else(|| "unknown query".to_owned())
    }
}

/// A planted optimizer bug: `LEFT JOIN` is planned as an inner join.
struct PlantedLeftJoin(SessionRunner);

impl QueryRunner for PlantedLeftJoin {
    fn run(&mut self, sql: &str) -> Result<Vec<Vec<String>>, String> {
        self.0.run(&sql.replace("LEFT JOIN", "JOIN"))
    }
}

#[test]
fn replaying_duckdb_agrees() -> R {
    let qs = queries()?;
    assert!(run(&qs, &mut Replay(&qs)).is_empty());
    Ok(())
}

/// Acceptance: a planted optimizer bug is found by the fixture's queries.
#[test]
fn planted_left_join_bug_is_found() -> R {
    let qs = queries()?;
    let with_left: Vec<Query> = qs
        .into_iter()
        .filter(|q| q.sql.contains("LEFT JOIN"))
        .collect();
    assert!(!with_left.is_empty(), "the fixture has no LEFT JOIN query");
    let mut honest = SessionRunner::new(&data())?;
    let honest_bad: Vec<String> = run(&with_left, &mut honest)
        .into_iter()
        .map(|d| d.sql)
        .collect();
    let mut planted = PlantedLeftJoin(SessionRunner::new(&data())?);
    let found = run(&with_left, &mut planted);
    assert!(
        found.iter().any(|d| !honest_bad.contains(&d.sql)),
        "the planted bug went unnoticed in {} LEFT JOIN queries",
        with_left.len()
    );
    Ok(())
}

#[test]
fn numbers_compare_as_numbers() {
    assert_eq!(normalise("12.50"), normalise("12.5"));
    assert_eq!(normalise("7"), normalise("7.000000"));
    assert_eq!(normalise("NULL"), "NULL");
    assert_ne!(normalise("abc"), normalise("abz"));
    assert_ne!(normalise("1.0"), normalise("1.1"));
}
