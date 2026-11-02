// SPDX-License-Identifier: Apache-2.0
//! Random TPC-H queries against a DuckDB oracle (P2-T110a).
//!
//! The fixture (`tests/fixtures/query-oracle/queries.json`, written by
//! `tests/scripts/gen_query_oracle.py`) holds seeded SQLancer-style queries over the TPC-H SF 0.01
//! tables with DuckDB's rows for each. [`run`] answers each on a [`QueryRunner`] and compares the
//! rows as multisets after [`normalise`]: numbers are compared as numbers to nine significant
//! digits (DuckDB and Spark print the same sum with different scales), everything else as text.

use std::path::Path;

use serde_json::Value;

use crate::golden::run::Engine as _;
use crate::golden::session_engine::SessionEngine;
use crate::golden::setup::{SessionSetup, TestKind};

/// The TPC-H tables, as the Parquet files of the planner tests are named.
pub const TABLES: [&str; 8] = [
    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
];

/// One generated query and DuckDB's answer.
#[derive(Debug, Clone)]
pub struct Query {
    /// The SQL, valid in DuckDB and Spark alike.
    pub sql: String,
    /// DuckDB's rows as text cells (`NULL` for SQL NULL).
    pub rows: Vec<Vec<String>>,
}

/// Parses the fixture.
///
/// # Errors
/// When the text is not the generator's JSON.
pub fn parse(text: &str) -> Result<Vec<Query>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let raw = v
        .get("queries")
        .and_then(Value::as_array)
        .ok_or("missing `queries`")?;
    raw.iter()
        .map(|q| {
            let sql = q
                .get("sql")
                .and_then(Value::as_str)
                .ok_or("query without sql")?;
            let rows = q
                .get("rows")
                .and_then(Value::as_array)
                .ok_or("query without rows")?
                .iter()
                .map(|r| {
                    r.as_array()
                        .ok_or("row is not a list")?
                        .iter()
                        .map(|c| c.as_str().map(str::to_owned).ok_or("cell is not text"))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Query {
                sql: sql.to_owned(),
                rows,
            })
        })
        .collect::<Result<Vec<_>, &str>>()
        .map_err(str::to_owned)
}

/// A cell in the form both engines' answers are compared in: numbers to nine significant digits.
#[must_use]
pub fn normalise(cell: &str) -> String {
    match cell.parse::<f64>() {
        Ok(v) if v.is_finite() => format!("{v:.9e}"),
        _ => cell.to_owned(),
    }
}

fn canonical(rows: &[Vec<String>]) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = rows
        .iter()
        .map(|r| r.iter().map(|c| normalise(c)).collect())
        .collect();
    out.sort();
    out
}

/// Answers a query on the engine under test.
pub trait QueryRunner {
    /// The rows (text cells) or the failure's error class and message.
    ///
    /// # Errors
    /// The engine's failure, as text.
    fn run(&mut self, sql: &str) -> Result<Vec<Vec<String>>, String>;
}

/// A query whose answer differs from DuckDB's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disagreement {
    /// The SQL.
    pub sql: String,
    /// What differs, in a line.
    pub detail: String,
}

/// Runs every query; returns the ones that disagree.
pub fn run(queries: &[Query], runner: &mut dyn QueryRunner) -> Vec<Disagreement> {
    queries
        .iter()
        .filter_map(|q| {
            let detail = match runner.run(&q.sql) {
                Err(e) => format!("engine failed: {e}"),
                Ok(got) => {
                    let (want, got) = (canonical(&q.rows), canonical(&got));
                    if want == got {
                        return None;
                    }
                    let first = want.iter().zip(&got).find(|(a, b)| a != b);
                    format!(
                        "DuckDB {} rows, engine {} rows; first difference {first:?}",
                        want.len(),
                        got.len()
                    )
                }
            };
            Some(Disagreement {
                sql: q.sql.clone(),
                detail,
            })
        })
        .collect()
}

/// `krishiv-session` over the TPC-H Parquet files.
#[derive(Debug)]
pub struct SessionRunner {
    engine: SessionEngine,
}

impl SessionRunner {
    /// A session with a temporary view per TPC-H table over `dir`.
    ///
    /// # Errors
    /// When the session or a view cannot be created.
    pub fn new(dir: &Path) -> Result<Self, String> {
        let mut engine = SessionEngine::new().map_err(|e| e.message)?;
        engine
            .reset(&SessionSetup::plain(TestKind::Regular))
            .map_err(|e| e.message)?;
        for t in TABLES {
            let path = dir.join(format!("{t}.parquet"));
            engine
                .execute(&format!(
                    "CREATE TEMPORARY VIEW {t} AS SELECT * FROM parquet.`{}`",
                    path.display()
                ))
                .map_err(|e| format!("{t}: {}", e.message))?;
        }
        Ok(Self { engine })
    }
}

impl QueryRunner for SessionRunner {
    fn run(&mut self, sql: &str) -> Result<Vec<Vec<String>>, String> {
        self.engine
            .execute(sql)
            .map(|o| o.rows)
            .map_err(|e| format!("{} {}", e.class.unwrap_or_default(), e.message))
    }
}
