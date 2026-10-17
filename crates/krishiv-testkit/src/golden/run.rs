// SPDX-License-Identifier: Apache-2.0
//! The golden-file runner: drives an [`Engine`] over the statements of each golden file and
//! tallies three separate numbers per file.
//!
//! 1. **parse**: of the statements Spark parsed, how many the engine parses.
//! 2. **negative**: of the statements Spark rejected with an error class, how many the engine
//!    rejects with the same class (and the same SQLSTATE when Spark recorded one).
//! 3. **result**: of the statements Spark executed, how many the engine answers with the same
//!    schema and rows. Like `SQLQueryTestSuite`, rows are sorted on both sides unless the query
//!    has an `ORDER BY`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use krishiv_common::error::ErrorClass;

use super::manifest::{ExclusionManifest, ManifestError};
use super::parse::{Expected, ExpectedError, ParseError, Statement, parse_results};
use super::{GoldenPair, GoldenSet};

/// An engine failure on one statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError {
    /// Spark error class, when the engine raised a classified error.
    pub class: Option<String>,
    /// SQLSTATE, when known.
    pub sql_state: Option<String>,
    /// Human-readable message.
    pub message: String,
}

/// A successful query answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryOutput {
    /// Schema as Spark's `struct<name:type,...>` string.
    pub schema: String,
    /// Rows; every cell rendered as in Spark's golden files (`NULL` for SQL NULL).
    pub rows: Vec<Vec<String>>,
}

/// The engine under test. One session per golden file: [`Engine::reset`] starts it.
pub trait Engine {
    /// Engine name for the report.
    fn name(&self) -> &str;
    /// Start a fresh session (called before every file).
    fn reset(&mut self);
    /// Parse `sql` without running it.
    fn parse(&mut self, sql: &str) -> Result<(), EngineError>;
    /// Run `sql` in the current session.
    fn execute(&mut self, sql: &str) -> Result<QueryOutput, EngineError>;
}

/// An engine that rejects every statement, without an error class.
#[derive(Debug, Default, Clone, Copy)]
pub struct RejectAllEngine;

impl RejectAllEngine {
    fn reject() -> EngineError {
        EngineError {
            class: None,
            sql_state: None,
            message: "RejectAllEngine rejects every statement".to_owned(),
        }
    }
}

impl Engine for RejectAllEngine {
    fn name(&self) -> &str {
        "reject-all"
    }

    fn reset(&mut self) {}

    fn parse(&mut self, _sql: &str) -> Result<(), EngineError> {
        Err(Self::reject())
    }

    fn execute(&mut self, _sql: &str) -> Result<QueryOutput, EngineError> {
        Err(Self::reject())
    }
}

/// `passed` of `total`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tally {
    /// Statements that matched.
    pub passed: u64,
    /// Statements that count towards this number.
    pub total: u64,
}

impl Tally {
    fn record(&mut self, passed: bool) {
        self.total += 1;
        self.passed += u64::from(passed);
    }

    pub(super) fn add(&mut self, other: Self) {
        self.passed += other.passed;
        self.total += other.total;
    }
}

/// Results of one golden file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileResult {
    /// Golden name.
    pub name: String,
    /// Statements in the file (0 when it has no `.sql.out`).
    pub statements: u64,
    /// Number 1: parse.
    pub parse: Tally,
    /// Number 2: negative / error-class match.
    pub negative: Tally,
    /// Number 3: statement-level result match.
    pub result: Tally,
    /// Spark errors without an error class (outside the negative denominator).
    pub unclassified_errors: u64,
}

/// A skipped file and what it would have counted for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedFile {
    /// Golden name.
    pub name: String,
    /// Statements the file has.
    pub statements: u64,
}

/// Results of a whole run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunResult {
    /// Engine name.
    pub engine: String,
    /// Per-file results of the files that ran.
    pub files: Vec<FileResult>,
    /// Excluded files (manifest order).
    pub excluded: Vec<SkippedFile>,
    /// `.sql` inputs without a `.sql.out`; nothing to compare against.
    pub without_expected: Vec<String>,
    /// Error classes Spark expects that Krishiv's registry does not know.
    pub unregistered_classes: BTreeSet<String>,
}

/// A run failure.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    /// Reading a golden file failed.
    #[error("reading {path}: {source}")]
    Io {
        /// File path.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// A golden file is malformed.
    #[error("{file}: {source}")]
    Parse {
        /// Golden name.
        file: String,
        /// The problem.
        source: ParseError,
    },
    /// The exclusion manifest is invalid.
    #[error(transparent)]
    Manifest(#[from] ManifestError),
}

fn read_lossy(path: &Path) -> Result<String, RunError> {
    let bytes = std::fs::read(path).map_err(|source| RunError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Parse the statements of one golden pair.
pub fn load_statements(pair: &GoldenPair) -> Result<Vec<Statement>, RunError> {
    parse_results(&read_lossy(&pair.out)?).map_err(|source| RunError::Parse {
        file: pair.name.clone(),
        source,
    })
}

/// Run `engine` over `set`, skipping the files of `manifest`.
pub fn run(
    engine: &mut dyn Engine,
    set: &GoldenSet,
    manifest: &ExclusionManifest,
) -> Result<RunResult, RunError> {
    manifest.validate(set)?;
    let mut result = RunResult {
        engine: engine.name().to_owned(),
        ..RunResult::default()
    };
    for pair in &set.pairs {
        let statements = load_statements(pair)?;
        note_unregistered(&statements, &mut result.unregistered_classes);
        if manifest.get(&pair.name).is_some() {
            result.excluded.push(SkippedFile {
                name: pair.name.clone(),
                statements: statements.len() as u64,
            });
        } else {
            engine.reset();
            result.files.push(run_file(engine, &pair.name, &statements));
        }
    }
    result.without_expected = set
        .unpaired_inputs
        .iter()
        .filter(|n| manifest.get(n).is_none())
        .cloned()
        .collect();
    for e in &manifest.entries {
        if set.unpaired_inputs.contains(&e.file) {
            result.excluded.push(SkippedFile {
                name: e.file.clone(),
                statements: 0,
            });
        }
    }
    Ok(result)
}

fn note_unregistered(statements: &[Statement], out: &mut BTreeSet<String>) {
    for s in statements {
        if let Expected::Error(e) = &s.expected
            && let Some(class) = &e.error_class
            && ErrorClass::from_name(class).is_none()
        {
            out.insert(class.clone());
        }
    }
}

/// Run one file's statements in the engine's current session.
pub fn run_file(engine: &mut dyn Engine, name: &str, statements: &[Statement]) -> FileResult {
    let mut file = FileResult {
        name: name.to_owned(),
        statements: statements.len() as u64,
        ..FileResult::default()
    };
    for s in statements {
        let parsed = engine.parse(&s.query);
        let executed = engine.execute(&s.query);
        match &s.expected {
            Expected::Rows(output) => {
                file.parse.record(parsed.is_ok());
                file.result.record(
                    executed
                        .as_ref()
                        .is_ok_and(|got| result_matches(s, output, got)),
                );
            }
            Expected::Error(e) => {
                if !e.is_parse_error() {
                    file.parse.record(parsed.is_ok());
                }
                match &e.error_class {
                    None => file.unclassified_errors += 1,
                    Some(class) => {
                        let ok = executed
                            .as_ref()
                            .err()
                            .is_some_and(|got| error_matches(e, class, got));
                        file.negative.record(ok);
                    }
                }
            }
        }
    }
    file
}

fn error_matches(expected: &ExpectedError, class: &str, got: &EngineError) -> bool {
    got.class.as_deref() == Some(class)
        && expected
            .sql_state
            .as_ref()
            .is_none_or(|st| got.sql_state.as_ref() == Some(st))
}

fn result_matches(s: &Statement, expected_output: &str, got: &QueryOutput) -> bool {
    if got.schema != s.schema {
        return false;
    }
    let rendered = got
        .rows
        .iter()
        .map(|r| r.join("\t"))
        .collect::<Vec<_>>()
        .join("\n");
    normalise(expected_output, s.is_ordered()) == normalise(&rendered, s.is_ordered())
}

/// Spark's comparison form: output lines, sorted when the query has no `ORDER BY`.
fn normalise(output: &str, ordered: bool) -> Vec<&str> {
    let mut lines: Vec<&str> = output.trim_end().split('\n').collect();
    if !ordered {
        lines.sort_unstable();
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    type R = Result<(), Box<dyn std::error::Error>>;

    /// Engine answering with canned results keyed by query text.
    struct Canned(Vec<(&'static str, Result<QueryOutput, EngineError>)>);

    impl Engine for Canned {
        fn name(&self) -> &str {
            "canned"
        }
        fn reset(&mut self) {}
        fn parse(&mut self, sql: &str) -> Result<(), EngineError> {
            if self.0.iter().any(|(q, _)| *q == sql) {
                Ok(())
            } else {
                Err(RejectAllEngine::reject())
            }
        }
        fn execute(&mut self, sql: &str) -> Result<QueryOutput, EngineError> {
            self.0
                .iter()
                .find(|(q, _)| *q == sql)
                .map(|(_, r)| r.clone())
                .unwrap_or_else(|| Err(RejectAllEngine::reject()))
        }
    }

    const OUT: &str = "-- !query\nselect a from t\n-- !query schema\nstruct<a:int>\n-- !query output\n1\n2\n\n\n\
-- !query\nselect a from t order by a desc\n-- !query schema\nstruct<a:int>\n-- !query output\n2\n1\n\n\n\
-- !query\nselect 1/0\n-- !query schema\nstruct<>\n-- !query output\norg.apache.spark.SparkArithmeticException\n\
{\n  \"errorClass\" : \"DIVIDE_BY_ZERO\",\n  \"sqlState\" : \"22012\"\n}\n";

    fn out(rows: &[&str]) -> Result<QueryOutput, EngineError> {
        Ok(QueryOutput {
            schema: "struct<a:int>".into(),
            rows: rows.iter().map(|r| vec![(*r).to_owned()]).collect(),
        })
    }

    #[test]
    fn hand_made_matching_result_passes() -> R {
        let statements = parse_results(OUT)?;
        let mut engine = Canned(vec![
            ("select a from t", out(&["2", "1"])), // unordered: row order is normalised
            ("select a from t order by a desc", out(&["2", "1"])),
            (
                "select 1/0",
                Err(EngineError {
                    class: Some("DIVIDE_BY_ZERO".into()),
                    sql_state: Some("22012".into()),
                    message: String::new(),
                }),
            ),
        ]);
        let f = run_file(&mut engine, "t", &statements);
        assert_eq!(
            f.parse,
            Tally {
                passed: 3,
                total: 3
            }
        );
        assert_eq!(
            f.negative,
            Tally {
                passed: 1,
                total: 1
            }
        );
        assert_eq!(
            f.result,
            Tally {
                passed: 2,
                total: 2
            }
        );
        Ok(())
    }

    #[test]
    fn order_by_keeps_order_and_wrong_class_fails() -> R {
        let statements = parse_results(OUT)?;
        let mut engine = Canned(vec![
            ("select a from t", out(&["1", "3"])),
            ("select a from t order by a desc", out(&["1", "2"])),
            (
                "select 1/0",
                Err(EngineError {
                    class: Some("OTHER".into()),
                    sql_state: None,
                    message: String::new(),
                }),
            ),
        ]);
        let f = run_file(&mut engine, "t", &statements);
        assert_eq!(
            f.result,
            Tally {
                passed: 0,
                total: 2
            }
        );
        assert_eq!(
            f.negative,
            Tally {
                passed: 0,
                total: 1
            }
        );
        Ok(())
    }

    #[test]
    fn reject_all_scores_zero_with_denominators() -> R {
        let statements = parse_results(OUT)?;
        let f = run_file(&mut RejectAllEngine, "t", &statements);
        assert_eq!(f.statements, 3);
        assert_eq!(
            f.parse,
            Tally {
                passed: 0,
                total: 3
            }
        );
        assert_eq!(
            f.negative,
            Tally {
                passed: 0,
                total: 1
            }
        );
        assert_eq!(
            f.result,
            Tally {
                passed: 0,
                total: 2
            }
        );
        Ok(())
    }
}
