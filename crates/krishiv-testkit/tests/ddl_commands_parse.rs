// SPDX-License-Identifier: Apache-2.0
//! The `krishiv-sql` parser on the DDL/DML slice 1 statements of the
//! vendored Spark `SQLQueryTestSuite` goldens: `CREATE TABLE` (including
//! `LIKE`), `DROP TABLE`, `CREATE`/`DROP` `DATABASE`|`SCHEMA`|`NAMESPACE`
//! and `USE`.
//!
//! A statement Spark parsed (its `.sql.out` shows a result or an analysis
//! or runtime error, not a `ParseException`) must parse here, with spans
//! ([`check_statement_spans`]) and differential `==` equality under
//! re-spacing; a statement Spark rejected while parsing must fail here
//! with Spark's error class and SQLSTATE. Statements outside slice 1
//! (queries, views, functions, DML, ...) are skipped: later slices cover
//! them.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use krishiv_sql::lexer::{TokenKind, tokenize};
use krishiv_sql::{ParserOptions, parse_statement};
use krishiv_testkit::golden::run::{Engine as _, load_statements};
use krishiv_testkit::golden::setup::FileSetup;
use krishiv_testkit::golden::sql_parse::ParseOnlyEngine;
use krishiv_testkit::golden::sql_spans::{check_statement_spans, respaced};
use krishiv_testkit::golden::{DEFAULT_TAG, GoldenSet};

type R = Result<(), Box<dyn std::error::Error>>;

/// Whether `sql` is a slice 1 statement: `CREATE TABLE` (or `CREATE [OR
/// REPLACE] [TEMPORARY] [EXTERNAL] TABLE`, `CREATE TABLE ... LIKE`),
/// `DROP TABLE`, `CREATE`/`DROP` `DATABASE`|`SCHEMA`|`NAMESPACE`, or `USE`.
fn in_slice_scope(sql: &str) -> bool {
    let tokens: Vec<_> = tokenize(sql)
        .into_iter()
        .filter(|t| !t.kind.is_trivia())
        .collect();
    let word = |i: usize, kw: &str| {
        tokens
            .get(i)
            .is_some_and(|t| t.kind == TokenKind::Word && t.span.text(sql).eq_ignore_ascii_case(kw))
    };
    if word(0, "USE") {
        return true;
    }
    if word(0, "CREATE") {
        let mut i = 1;
        if word(i, "OR") && word(i + 1, "REPLACE") {
            i += 2;
        }
        if word(i, "TEMP") || word(i, "TEMPORARY") {
            i += 1;
        }
        if word(i, "EXTERNAL") {
            i += 1;
        }
        return word(i, "TABLE")
            || ["DATABASE", "SCHEMA", "NAMESPACE"]
                .iter()
                .any(|k| word(i, k));
    }
    if word(0, "DROP") {
        return word(1, "TABLE")
            || ["DATABASE", "SCHEMA", "NAMESPACE"]
                .iter()
                .any(|k| word(1, k));
    }
    false
}

/// One statement's outcome, merged across the file's session runs (a
/// statement passes only if it passes in every run).
#[derive(Debug, Clone, Default)]
struct Outcome {
    parse: Option<bool>,
    negative: Option<bool>,
    note: Option<String>,
}

impl Outcome {
    fn and(self, other: Self) -> Self {
        let both = |a: Option<bool>, b: Option<bool>| match (a, b) {
            (Some(a), Some(b)) => Some(a && b),
            (a, b) => a.or(b),
        };
        Self {
            parse: both(self.parse, other.parse),
            negative: both(self.negative, other.negative),
            note: self.note.or(other.note),
        }
    }
}

#[test]
fn ddl_slice_parses_with_spans_and_differential_equality() -> R {
    let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
    let mut engine = ParseOnlyEngine::default();
    let (mut total, mut parsed) = (0_u64, 0_u64);
    let mut rejected: BTreeMap<String, u64> = BTreeMap::new();
    let mut failures = String::new();
    for pair in &set.pairs {
        let statements = load_statements(pair)?;
        let setup = FileSetup::parse(&pair.name, &std::fs::read_to_string(&pair.sql)?);
        let mut merged: Option<Vec<(bool, Outcome)>> = None;
        for session in setup.sessions() {
            engine.reset(&session).map_err(|e| e.message)?;
            let outcomes: Vec<(bool, Outcome)> = statements
                .iter()
                .map(|s| {
                    if engine.apply_set(&s.query) || !in_slice_scope(&s.query) {
                        return (false, Outcome::default());
                    }
                    let options = engine.options();
                    (true, outcome(&s.query, &s.expected, &options))
                })
                .collect();
            merged = Some(match merged {
                None => outcomes,
                Some(prev) => prev
                    .into_iter()
                    .zip(outcomes)
                    .map(|((k, a), (_, b))| (k, a.and(b)))
                    .collect(),
            });
        }
        for (s, (kept, o)) in statements.iter().zip(merged.unwrap_or_default()) {
            if !kept {
                continue;
            }
            total += 1;
            let ok = o.parse.unwrap_or(true) && o.negative.unwrap_or(true);
            if ok {
                parsed += 1;
                if o.negative == Some(true) {
                    let class = match &s.expected {
                        krishiv_testkit::golden::parse::Expected::Error(e) => {
                            e.error_class.clone().unwrap_or_default()
                        }
                        _ => String::new(),
                    };
                    *rejected.entry(class).or_default() += 1;
                }
            } else {
                writeln!(
                    failures,
                    "{}: {}{}\n{}",
                    pair.name,
                    match &s.expected {
                        krishiv_testkit::golden::parse::Expected::Error(e) =>
                            format!("expected {}", e.error_class.clone().unwrap_or_default()),
                        _ => "expected parse".to_owned(),
                    },
                    o.note.map_or_else(String::new, |n| format!(" ({n})")),
                    s.query.replace('\n', " ")
                )?;
            }
        }
    }
    println!(
        "DDL slice 1: {parsed} / {total} parse ({:.3}%); Spark parse-time errors rejected: \
         {rejected:?}\n{failures}",
        100.0 * parsed as f64 / total as f64
    );
    assert!(total > 600, "too few statements: {total}");
    assert!(failures.is_empty(), "DDL slice 1 mismatches:\n{failures}");
    Ok(())
}

/// Parses `sql` (Spark's outcome is `expected`) with `options`: `parse`
/// is whether a statement Spark parsed parses here (with spans and
/// differential equality), `negative` whether a statement Spark rejected
/// while parsing fails here with its class and SQLSTATE.
fn outcome(
    sql: &str,
    expected: &krishiv_testkit::golden::parse::Expected,
    options: &ParserOptions,
) -> Outcome {
    use krishiv_testkit::golden::parse::Expected;
    match expected {
        Expected::Rows(_) => match parse_statement(sql, options) {
            Ok(s) => {
                if let Err(e) = check_statement_spans(sql, &s, options) {
                    return Outcome {
                        parse: Some(false),
                        negative: None,
                        note: Some(format!("span check failed: {e}")),
                    };
                }
                let again = parse_statement(&respaced(sql), options);
                Outcome {
                    parse: Some(matches!(again, Ok(r) if r == s)),
                    negative: None,
                    note: None,
                }
            }
            Err(_) => Outcome {
                parse: Some(false),
                negative: None,
                note: None,
            },
        },
        Expected::Error(e) if e.is_parse_error() => {
            let class = e.error_class.clone().unwrap_or_default();
            match parse_statement(sql, options) {
                Err(ours) => {
                    let class_ok = ours.error.error_class().name() == class;
                    let state_ok = e
                        .sql_state
                        .as_ref()
                        .is_none_or(|st| ours.error.sql_state() == Some(st.as_str()));
                    Outcome {
                        parse: None,
                        negative: Some(class_ok && state_ok),
                        note: None,
                    }
                }
                Ok(_) => Outcome {
                    parse: None,
                    negative: Some(false),
                    note: None,
                },
            }
        }
        // Spark parsed the statement (an analysis or runtime error): it
        // must parse here.
        Expected::Error(_) => Outcome {
            parse: Some(parse_statement(sql, options).is_ok()),
            negative: None,
            note: None,
        },
    }
}
