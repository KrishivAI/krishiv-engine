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

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use krishiv_common::error::ErrorClass;

use super::manifest::{ExclusionManifest, ManifestError};
use super::normalise::{Masker, normalize_test_results};
use super::parse::{Expected, ExpectedError, ParseError, Statement, parse_results};
use super::setup::{FileSetup, SessionSetup};
use super::{GoldenPair, GoldenSet, Jdk};

/// The error class a statement that exceeds the runner's time limit fails
/// with. It is a harness failure, never a baseline failure: [`FileResult`]
/// keeps timeouts out of [`Case`] so [`crate::golden::known_failures`] never
/// lists them, and the ratchet never asks for them to be removed.
pub const TIMEOUT_CLASS: &str = "KRISHIV_GOLDEN_TIMEOUT";

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
    /// analyzed plan. An engine fills this from its own resolved plan with
    /// [`super::sorted::is_semantically_sorted`]; `None` falls back to
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

/// The negative score of one expected error class.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassTally {
    /// Statements of the class whose error the engine matched, and all of them.
    pub tally: Tally,
    /// Spark raises the class while parsing (a `ParseException`), so no query was planned.
    pub parse_phase: bool,
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
    /// Number 2 per expected Spark error class.
    pub classes: BTreeMap<String, ClassTally>,
    /// Per-statement outcomes, in file order (the counted statements only).
    pub cases: Vec<Case>,
    /// Statements that timed out, kept out of the baseline comparison and
    /// reported as harness failures instead.
    pub timeouts: Vec<Case>,
}

/// What one statement of a file scored, merged over the file's runs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Case {
    /// Position of the statement in the file (0-based, over all statements).
    pub index: u32,
    /// Number 1, when the statement counts for it.
    pub parse: Option<bool>,
    /// Number 2, when the statement counts for it.
    pub negative: Option<bool>,
    /// Number 3, when the statement counts for it.
    pub result: Option<bool>,
    /// Why the statement failed in the first run it failed in (empty when it passed).
    pub detail: String,
}

impl Case {
    /// Whether every number the statement counts for passed.
    #[must_use]
    pub fn passed(&self) -> bool {
        [self.parse, self.negative, self.result]
            .into_iter()
            .flatten()
            .all(|p| p)
    }
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
    run_with(engine, set, manifest, &|_| true)
}

/// [`run`], counting only the statements whose SQL satisfies `keep` (the
/// others still run, so that `SET` statements take effect).
pub fn run_with(
    engine: &mut dyn Engine,
    set: &GoldenSet,
    manifest: &ExclusionManifest,
    keep: &dyn Fn(&str) -> bool,
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
                    .push(run_file_with(engine, &pair.name, &statements, &setup, keep));
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
#[derive(Debug, Clone, Default)]
struct Outcome {
    parse: Option<bool>,
    negative: Option<bool>,
    result: Option<bool>,
    unclassified: bool,
    /// The expected error class and whether Spark raises it while parsing, for a counted negative.
    class: Option<(String, bool)>,
    detail: String,
    timeout: bool,
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
            class: self.class.or(other.class),
            detail: if self.detail.is_empty() {
                other.detail
            } else {
                self.detail
            },
            timeout: self.timeout || other.timeout,
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
    run_file_with(engine, name, statements, setup, &|_| true)
}

/// [`run_file`], counting only the statements whose SQL satisfies `keep`.
pub fn run_file_with(
    engine: &mut dyn Engine,
    name: &str,
    statements: &[Statement],
    setup: &FileSetup,
    keep: &dyn Fn(&str) -> bool,
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
    let kept: Vec<bool> = statements.iter().map(|s| keep(&s.query)).collect();
    let mut file = FileResult {
        name: name.to_owned(),
        statements: kept.iter().filter(|k| **k).count() as u64,
        runs: sessions.len() as u64,
        ..FileResult::default()
    };
    let merged = merged.unwrap_or_default().into_iter().zip(kept).enumerate();
    for (index, o) in merged.filter_map(|(i, (o, k))| k.then_some((i, o))) {
        let case = Case {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            parse: o.parse,
            negative: o.negative,
            result: o.result,
            detail: o.detail.clone(),
        };
        if o.timeout {
            file.timeouts.push(case);
            continue;
        }
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
        if let (Some(v), Some((class, parse_phase))) = (o.negative, o.class.clone()) {
            let entry = file.classes.entry(class).or_default();
            entry.parse_phase = parse_phase;
            entry.tally.record(v);
        }
        let mut case = case;
        if case.passed() {
            case.detail.clear();
        }
        file.cases.push(case);
    }
    file
}

fn failed(s: &Statement) -> Outcome {
    outcome(s, false, None)
}

fn run_session(engine: &mut dyn Engine, statements: &[Statement]) -> Vec<Outcome> {
    let Ok(masker) = Masker::new(engine.warehouse_dir()) else {
        return statements.iter().map(failed).collect();
    };
    statements
        .iter()
        .map(|s| {
            let parsed = engine.parse(&s.query).is_ok();
            outcome(s, parsed, Some((engine.execute(&s.query), &masker)))
        })
        .collect()
}

/// A one-line description of an engine error, for [`Case::detail`].
fn error_detail(e: &EngineError) -> String {
    let message: String = e
        .message
        .lines()
        .next()
        .unwrap_or_default()
        .chars()
        .take(160)
        .collect();
    format!("error {}: {message}", e.class.as_deref().unwrap_or("-"))
}

/// What `s` scored: `executed` is the engine's answer with the run's masker, `None` when the
/// statement could not run (the session could not be set up).
fn outcome(
    s: &Statement,
    parsed: bool,
    executed: Option<(Result<QueryOutput, EngineError>, &Masker)>,
) -> Outcome {
    let is_timeout = matches!(
        &executed,
        Some((Err(e), _)) if e.class.as_deref() == Some(TIMEOUT_CLASS)
    );
    let parse_detail = || {
        if parsed {
            String::new()
        } else {
            "parse failed; ".to_owned()
        }
    };
    match &s.expected {
        Expected::Rows(output) => {
            let (passed, why) = match executed {
                Some((Ok(got), m)) => match result_mismatch(s, output, &got, m) {
                    None => (true, String::new()),
                    Some(why) => (false, why),
                },
                Some((Err(e), _)) => (false, error_detail(&e)),
                None => (false, "not executed".to_owned()),
            };
            Outcome {
                parse: Some(parsed),
                result: Some(passed),
                detail: if passed && parsed {
                    String::new()
                } else {
                    format!("{}{why}", parse_detail())
                },
                timeout: is_timeout,
                ..Outcome::default()
            }
        }
        Expected::Error(e) => {
            let counts_parse = !e.is_parse_error();
            let (negative, why) = match e.error_class.as_deref() {
                None => (None, String::new()),
                Some(class) => match executed {
                    Some((Err(got), _)) if error_matches(e, class, &got) => {
                        (Some(true), String::new())
                    }
                    Some((Err(got), _)) => (Some(false), error_detail(&got)),
                    Some((Ok(_), _)) => (Some(false), "no error".to_owned()),
                    None => (Some(false), "not executed".to_owned()),
                },
            };
            let parse_failed = counts_parse && !parsed;
            Outcome {
                parse: counts_parse.then_some(parsed),
                negative,
                class: e
                    .error_class
                    .clone()
                    .map(|class| (class, e.is_parse_error())),
                unclassified: e.error_class.is_none(),
                detail: if parse_failed {
                    format!("{}{why}", parse_detail())
                } else {
                    why
                },
                timeout: is_timeout,
                ..Outcome::default()
            }
        }
    }
}

fn error_matches(expected: &ExpectedError, class: &str, got: &EngineError) -> bool {
    got.class.as_deref() == Some(class)
        && expected
            .sql_state
            .as_ref()
            .is_none_or(|st| got.sql_state.as_ref() == Some(st))
}

/// Why `got` does not match the recorded result, or `None` when it does.
fn result_mismatch(
    s: &Statement,
    expected_output: &str,
    got: &QueryOutput,
    masker: &Masker,
) -> Option<String> {
    if got.schema != s.schema {
        return Some(format!("schema {} (want {})", got.schema, s.schema));
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
    // `normalizeTestResults` on both sides.
    let (got, want) = (
        normalize_test_results(&rows.join("\n")),
        normalize_test_results(expected_output),
    );
    if got == want {
        return None;
    }
    let clip = |s: Option<&&str>| -> String {
        s.map_or_else(|| "<none>".to_owned(), |s| s.chars().take(80).collect())
    };
    // `split` keeps empty segments, so two different texts differ at some segment.
    let (g, w): (Vec<&str>, Vec<&str>) = (got.split('\n').collect(), want.split('\n').collect());
    let at = (0..g.len().max(w.len()))
        .find(|&i| g.get(i) != w.get(i))
        .unwrap_or_default();
    Some(format!(
        "rows differ at row {at}: got `{}`, want `{}`",
        clip(g.get(at)),
        clip(w.get(at))
    ))
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
    fn python_tracebacks_do_not_count() -> R {
        let golden = "-- !query\nselect a from t\n-- !query schema\nstruct<a:int>\n-- !query output\n1\nboom\n";
        let statements = parse_results(golden)?;
        let rows = [
            "1",
            "Traceback (most recent call last):",
            "  File \"u.py\", line 2",
            "boom",
        ];
        let mut canned = out(&rows);
        if let Ok(o) = canned.as_mut() {
            o.sorted = Some(true);
        }
        let mut engine = Canned(vec![("select a from t", canned)]);
        let f = run_file(&mut engine, "t", &statements, &plain());
        assert_eq!(
            f.result,
            Tally {
                passed: 1,
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

    #[test]
    fn failed_cases_say_why() -> R {
        let statements = parse_results(OUT)?;
        let mut engine = Canned(vec![
            (
                "select a from t",
                Ok(QueryOutput {
                    schema: "struct<b:int>".into(),
                    rows: vec![],
                    sorted: None,
                }),
            ),
            ("select a from t order by a desc", out(&["2", "3"])),
            ("select 1/0", out(&["1"])),
        ]);
        let f = run_file(&mut engine, "t", &statements, &plain());
        let details: Vec<&str> = f.cases.iter().map(|c| c.detail.as_str()).collect();
        assert_eq!(
            details,
            [
                "schema struct<b:int> (want struct<a:int>)",
                "rows differ at row 1: got `3`, want `1`",
                "no error",
            ]
        );
        assert!(f.cases.iter().all(|c| !c.passed()));
        assert_eq!(
            f.cases.iter().map(|c| c.index).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        let f = run_file(&mut RejectAllEngine, "t", &statements, &plain());
        assert_eq!(
            f.cases[0].detail,
            "parse failed; error -: RejectAllEngine rejects every statement"
        );
        assert_eq!(
            f.cases[2].detail,
            "parse failed; error -: RejectAllEngine rejects every statement"
        );
        Ok(())
    }

    #[test]
    fn a_short_result_is_reported_at_its_first_missing_row() -> R {
        let statements = parse_results(OUT)?;
        let mut engine = Canned(vec![("select a from t order by a desc", out(&["2"]))]);
        let f = run_file(&mut engine, "t", &statements, &plain());
        assert_eq!(
            f.cases[1].detail,
            "rows differ at row 1: got `<none>`, want `1`"
        );
        Ok(())
    }

    /// An engine whose sessions cannot be set up.
    struct NoSession;

    impl Engine for NoSession {
        fn name(&self) -> &str {
            "no-session"
        }
        fn reset(&mut self, _setup: &SessionSetup) -> Result<(), EngineError> {
            Err(RejectAllEngine::reject())
        }
        fn parse(&mut self, _sql: &str) -> Result<(), EngineError> {
            Ok(())
        }
        fn execute(&mut self, _sql: &str) -> Result<QueryOutput, EngineError> {
            Err(RejectAllEngine::reject())
        }
    }

    #[test]
    fn a_session_that_cannot_start_fails_every_statement_unrun() -> R {
        let statements = parse_results(OUT)?;
        let f = run_file(&mut NoSession, "t", &statements, &plain());
        let details: Vec<&str> = f.cases.iter().map(|c| c.detail.as_str()).collect();
        assert_eq!(
            details,
            [
                "parse failed; not executed",
                "parse failed; not executed",
                "parse failed; not executed"
            ]
        );
        assert_eq!(f.result.passed + f.negative.passed + f.parse.passed, 0);
        let mut e = NoSession;
        assert_eq!(e.name(), "no-session");
        assert!(e.parse("x").is_ok() && e.execute("x").is_err());
        Ok(())
    }

    /// An engine that times out on every statement.
    struct TimeoutEngine;

    impl Engine for TimeoutEngine {
        fn name(&self) -> &str {
            "timeout"
        }
        fn reset(&mut self, _setup: &SessionSetup) -> Result<(), EngineError> {
            Ok(())
        }
        fn parse(&mut self, _sql: &str) -> Result<(), EngineError> {
            Ok(())
        }
        fn execute(&mut self, sql: &str) -> Result<QueryOutput, EngineError> {
            let _ = sql;
            Err(EngineError {
                class: Some(TIMEOUT_CLASS.to_owned()),
                sql_state: None,
                message: "no answer within 0ns".to_owned(),
            })
        }
    }

    #[test]
    fn timeouts_stay_out_of_the_baseline_comparison() -> R {
        let statements = parse_results(OUT)?;
        let f = run_file(&mut TimeoutEngine, "t", &statements, &plain());
        // Timeouts are harness failures: no tallies, no baseline cases.
        assert!(f.timeouts.len() == 3, "{f:?}");
        assert!(f.cases.is_empty(), "{f:?}");
        assert_eq!((f.parse.total, f.negative.total, f.result.total), (0, 0, 0));
        assert_eq!(f.unclassified_errors, 0);
        let detail = &f.timeouts[0].detail;
        assert!(detail.contains(TIMEOUT_CLASS), "{detail}");
        assert_eq!(f.timeouts[0].index, 0);
        Ok(())
    }
}
