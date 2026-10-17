// SPDX-License-Identifier: Apache-2.0
//! The golden-file runner: drives an [`Engine`] over the statements of each golden file and
//! tallies three separate numbers per file.
//!
//! 1. **parse**: of the statements Spark parsed, how many the engine parses.
//! 2. **negative**: of the statements Spark rejected with an error class, how many the engine
//!    rejects with the same class (and the same SQLSTATE when Spark recorded one). Spark errors
//!    without a class are counted separately as `unclassified_errors` and are outside this
//!    denominator. Only class and SQLSTATE are compared, not message parameters.
//! 3. **result**: of the statements Spark executed, how many the engine answers with the same
//!    schema and rows, after Spark's output masking ([`super::normalise`]). The comparison is exact text, as in Spark (no float tolerance). Like
//!    `SQLQueryTestSuite`, rows are sorted (per row, in UTF-16 order) unless the result is
//!    sorted, which the engine reports in [`QueryOutput::sorted`].
//!
//! A file runs once per `--CONFIG_DIM` combination (see [`super::setup`]); a statement counts as
//! passed only if it passes in every run, and counts once in the denominators. Excluded files
//! are never parsed or run; a file that cannot be read or parsed is recorded in
//! [`RunResult::failed_files`] and the run continues.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use krishiv_common::error::ErrorClass;

use super::manifest::{ExclusionManifest, ManifestError};
use super::normalise::Masker;
use super::parse::{Expected, ExpectedError, ParseError, Statement, parse_results};
use super::setup::{FileSetup, SessionSetup};
use super::{GoldenPair, GoldenSet, Jdk};

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
    /// Rows; every cell rendered as Spark's `HiveResult.hiveResultString` does (`NULL` for SQL
    /// NULL), cells joined by a tab.
    pub rows: Vec<Vec<String>>,
    /// Whether the result order is significant, as Spark's `isSemanticallySorted` decides on the
    /// analyzed plan. An engine fills this from its own plan; `None` falls back to
    /// [`Statement::text_sorted`], an approximation meant for stubs.
    pub sorted: Option<bool>,
}

/// The engine under test.
pub trait Engine {
    /// Engine name for the report.
    fn name(&self) -> &str;
    /// The session's warehouse directory, masked out of rows like Spark does (empty: none).
    fn warehouse_dir(&self) -> &str {
        ""
    }
    /// Start a fresh session configured as `setup` (called before every run of every file).
    fn reset(&mut self, setup: &SessionSetup) -> Result<(), EngineError>;
    /// Parse `sql` without running it, under the current session configs.
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

    fn reset(&mut self, _setup: &SessionSetup) -> Result<(), EngineError> {
        Ok(())
    }

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
    /// Statements in the file.
    pub statements: u64,
    /// Runs of the file (`--CONFIG_DIM` combinations; at least 1).
    pub runs: u64,
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
    /// Statements the file has (0 for a file excluded without being parsed).
    pub statements: u64,
}

/// A file that could not be run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedFile {
    /// Golden name.
    pub name: String,
    /// Why.
    pub error: String,
}

/// Results of a whole run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunResult {
    /// Engine name.
    pub engine: String,
    /// Per-file results of the files that ran.
    pub files: Vec<FileResult>,
    /// Excluded files, in scan order (statement counts are 0: excluded files are not parsed).
    pub excluded: Vec<SkippedFile>,
    /// Files that could not be read or parsed; the run continued without them.
    pub failed_files: Vec<FailedFile>,
    /// `.sql` inputs without a `.sql.out`; nothing to compare against.
    pub without_expected: Vec<String>,
    /// JDK the results were chosen for.
    pub jdk: Jdk,
    /// Files compared against their JDK-specific alternate results.
    pub variant_files: Vec<String>,
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
        jdk: set.jdk,
        ..RunResult::default()
    };
    for pair in &set.pairs {
        if manifest.get(&pair.name).is_some() {
            result.excluded.push(SkippedFile {
                name: pair.name.clone(),
                statements: 0,
            });
            continue;
        }
        if pair.variant {
            result.variant_files.push(pair.name.clone());
        }
        let loaded = load_statements(pair)
            .and_then(|statements| read_lossy(&pair.sql).map(|sql| (statements, sql)));
        match loaded {
            Ok((statements, sql)) => {
                note_unregistered(&statements, &mut result.unregistered_classes);
                let setup = FileSetup::parse(&pair.name, &sql);
                result
                    .files
                    .push(run_file(engine, &pair.name, &statements, &setup));
            }
            Err(e) => result.failed_files.push(FailedFile {
                name: pair.name.clone(),
                error: e.to_string(),
            }),
        }
    }
    for name in &set.unpaired_inputs {
        if manifest.get(name).is_some() {
            result.excluded.push(SkippedFile {
                name: name.clone(),
                statements: 0,
            });
        } else {
            result.without_expected.push(name.clone());
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

/// What one statement scored in one run; `None` means it does not count for that number.
#[derive(Debug, Clone, Copy, Default)]
struct Outcome {
    parse: Option<bool>,
    negative: Option<bool>,
    result: Option<bool>,
    unclassified: bool,
}

impl Outcome {
    /// Pass only if passed in both runs.
    fn and(self, other: Self) -> Self {
        let both = |a: Option<bool>, b: Option<bool>| match (a, b) {
            (Some(a), Some(b)) => Some(a && b),
            (a, b) => a.or(b),
        };
        Self {
            parse: both(self.parse, other.parse),
            negative: both(self.negative, other.negative),
            result: both(self.result, other.result),
            unclassified: self.unclassified,
        }
    }
}

/// Run one file's statements once per session of `setup`.
pub fn run_file(
    engine: &mut dyn Engine,
    name: &str,
    statements: &[Statement],
    setup: &FileSetup,
) -> FileResult {
    let sessions = setup.sessions();
    let mut merged: Option<Vec<Outcome>> = None;
    for session in &sessions {
        let outcomes = match engine.reset(session) {
            Ok(()) => run_session(engine, statements),
            // The session could not be set up: every statement fails.
            Err(_) => statements.iter().map(failed).collect(),
        };
        merged = Some(match merged {
            None => outcomes,
            Some(prev) => prev
                .into_iter()
                .zip(outcomes)
                .map(|(a, b)| a.and(b))
                .collect(),
        });
    }
    let mut file = FileResult {
        name: name.to_owned(),
        statements: statements.len() as u64,
        runs: sessions.len() as u64,
        ..FileResult::default()
    };
    for o in merged.unwrap_or_default() {
        for (tally, v) in [
            (&mut file.parse, o.parse),
            (&mut file.negative, o.negative),
            (&mut file.result, o.result),
        ] {
            if let Some(v) = v {
                tally.record(v);
            }
        }
        file.unclassified_errors += u64::from(o.unclassified);
    }
    file
}

fn failed(s: &Statement) -> Outcome {
    outcome(s, false, None, None)
}

fn run_session(engine: &mut dyn Engine, statements: &[Statement]) -> Vec<Outcome> {
    let Ok(masker) = Masker::new(engine.warehouse_dir()) else {
        return statements.iter().map(failed).collect();
    };
    statements
        .iter()
        .map(|s| {
            let parsed = engine.parse(&s.query).is_ok();
            outcome(s, parsed, Some(engine.execute(&s.query)), Some(&masker))
        })
        .collect()
}

fn outcome(
    s: &Statement,
    parsed: bool,
    executed: Option<Result<QueryOutput, EngineError>>,
    masker: Option<&Masker>,
) -> Outcome {
    match &s.expected {
        Expected::Rows(output) => Outcome {
            parse: Some(parsed),
            result: Some(
                executed
                    .and_then(Result::ok)
                    .zip(masker)
                    .is_some_and(|(got, m)| result_matches(s, output, &got, m)),
            ),
            ..Outcome::default()
        },
        Expected::Error(e) => Outcome {
            parse: (!e.is_parse_error()).then_some(parsed),
            negative: e.error_class.as_deref().map(|class| {
                executed
                    .and_then(Result::err)
                    .is_some_and(|got| error_matches(e, class, &got))
            }),
            unclassified: e.error_class.is_none(),
            ..Outcome::default()
        },
    }
}

fn error_matches(expected: &ExpectedError, class: &str, got: &EngineError) -> bool {
    got.class.as_deref() == Some(class)
        && expected
            .sql_state
            .as_ref()
            .is_none_or(|st| got.sql_state.as_ref() == Some(st))
}

fn result_matches(
    s: &Statement,
    expected_output: &str,
    got: &QueryOutput,
    masker: &Masker,
) -> bool {
    if got.schema != s.schema {
        return false;
    }
    // `getNormalizedQueryExecutionResult`: mask each row, then sort unless the plan is sorted.
    let mut rows: Vec<String> = got
        .rows
        .iter()
        .map(|r| masker.mask(&r.join("\t")))
        .collect();
    if !got.sorted.unwrap_or_else(|| s.text_sorted()) {
        // Scala `String` order is UTF-16 code-unit order.
        rows.sort_unstable_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    }
    rows.join("\n").trim_end() == expected_output.trim_end()
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
        fn reset(&mut self, _setup: &SessionSetup) -> Result<(), EngineError> {
            Ok(())
        }
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

    fn plain() -> FileSetup {
        FileSetup::parse("t", "")
    }

    fn out(rows: &[&str]) -> Result<QueryOutput, EngineError> {
        Ok(QueryOutput {
            schema: "struct<a:int>".into(),
            rows: rows.iter().map(|r| vec![(*r).to_owned()]).collect(),
            sorted: None,
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
        let f = run_file(&mut engine, "t", &statements, &plain());
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
        let f = run_file(&mut engine, "t", &statements, &plain());
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
        let f = run_file(&mut RejectAllEngine, "t", &statements, &plain());
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
