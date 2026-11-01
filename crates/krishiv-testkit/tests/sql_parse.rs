// SPDX-License-Identifier: Apache-2.0
//! The `krishiv-sql` parser on the DQL statements of the vendored Spark
//! `SQLQueryTestSuite` goldens:
//!
//! * acceptance against Spark: statements Spark parsed must parse (≥ 99.9%,
//!   every failure listed); every statement Spark rejects while parsing must
//!   be rejected with Spark's error class; so is a statement Spark rejects
//!   when building the plan (`AstBuilder`), by the post-parse checks
//!   (P2-T079), except the classes they do not raise;
//! * the differential `==` check: each statement re-lexed with comments
//!   between all tokens parses to an `==` tree (equality ignores spans);
//! * spans (`sql_spans::check_spans`): every node's span lies on token
//!   boundaries, inside its parent's span, and every expression's text
//!   parses to that expression.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use krishiv_sql::parse_query;
use krishiv_testkit::golden::parse::Expected;
use krishiv_testkit::golden::run::{Engine as _, load_statements};
use krishiv_testkit::golden::setup::FileSetup;
use krishiv_testkit::golden::sql_parse::{ParseOnlyEngine, in_dql_scope};
use krishiv_testkit::golden::sql_spans::{check_spans, respaced};
use krishiv_testkit::golden::{DEFAULT_TAG, GoldenSet};

type R = Result<(), Box<dyn std::error::Error>>;

/// Error classes Spark raises while parsing (the parser must raise them).
const PARSER_CLASSES: &[&str] = &[
    "PARSE_SYNTAX_ERROR",
    "PARSE_EMPTY_STATEMENT",
    "UNCLOSED_BRACKETED_COMMENT",
    "INVALID_IDENTIFIER",
    "SYNTAX_DISCONTINUED.BANG_EQUALS_NOT",
];

/// Error classes Spark raises in `AstBuilder` that the post-parse checks
/// (P2-T079) do not raise, with the number of corpus statements (per session
/// run): an inline table whose expressions fail when Spark evaluates them
/// while parsing needs expression evaluation. Every other `AstBuilder`
/// error is raised with Spark's class.
const OWED_TO_VALIDATION: &[(&str, u64)] =
    &[("INVALID_INLINE_TABLE.FAILED_SQL_EXPRESSION_EVALUATION", 3)];

#[test]
fn dql_corpus_parses_with_spans_and_differential_equality() -> R {
    let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
    let mut engine = ParseOnlyEngine::default();
    let (mut total, mut parsed) = (0_u64, 0_u64);
    let mut rejected: BTreeMap<String, u64> = BTreeMap::new();
    let mut accepted: BTreeMap<String, u64> = BTreeMap::new();
    let mut failures = String::new();
    for pair in &set.pairs {
        let statements = load_statements(pair)?;
        let setup = FileSetup::parse(&pair.name, &std::fs::read_to_string(&pair.sql)?);
        for session in setup.sessions() {
            engine.reset(&session).map_err(|e| e.message)?;
            for s in &statements {
                if engine.apply_set(&s.query) || !in_dql_scope(&s.query) {
                    continue;
                }
                let options = engine.options();
                let result = parse_query(&s.query, &options);
                if let Expected::Error(e) = &s.expected
                    && e.is_parse_error()
                {
                    let class = e.error_class.clone().unwrap_or_default();
                    match &result {
                        Err(ours) => {
                            assert_eq!(
                                ours.error.error_class().name(),
                                class,
                                "{}: {}",
                                pair.name,
                                s.query
                            );
                            *rejected.entry(class).or_default() += 1;
                        }
                        Ok(_) => {
                            assert!(
                                !PARSER_CLASSES.contains(&class.as_str()),
                                "{}: Spark raises {class} while parsing\n{}",
                                pair.name,
                                s.query
                            );
                            *accepted.entry(class).or_default() += 1;
                        }
                    }
                    continue;
                }
                total += 1;
                let q = match result {
                    Ok(q) => q,
                    Err(e) => {
                        writeln!(
                            failures,
                            "{}: {}\n  {}",
                            pair.name,
                            e.error,
                            s.query.replace('\n', " ")
                        )?;
                        continue;
                    }
                };
                parsed += 1;
                check_spans(&s.query, &q, &options)
                    .map_err(|e| format!("{}: {e}\n{}", pair.name, s.query))?;
                let again = parse_query(&respaced(&s.query), &options);
                assert!(
                    matches!(&again, Ok(r) if *r == q),
                    "{}: differential mismatch\n{}",
                    pair.name,
                    s.query
                );
            }
        }
    }
    println!(
        "DQL corpus: {parsed} / {total} parse ({:.3}%); Spark parse-time errors rejected: \
         {rejected:?}; accepted (owed to post-parse validation): {accepted:?}\n{failures}",
        100.0 * parsed as f64 / total as f64
    );
    assert!(total > 13_000, "too few statements: {total}");
    assert!(
        parsed * 1000 >= total * 999,
        "parse rate below 99.9%:\n{failures}"
    );
    let owed: BTreeMap<String, u64> = OWED_TO_VALIDATION
        .iter()
        .map(|(c, n)| ((*c).to_owned(), *n))
        .collect();
    assert_eq!(accepted, owed);
    assert_eq!(rejected.get("PARSE_SYNTAX_ERROR").copied(), Some(90));
    assert_eq!(rejected.get("UNCLOSED_BRACKETED_COMMENT").copied(), Some(2));
    Ok(())
}

/// Fuzz finding (weekly `sql_parse`, input `fuzz/regressions/sql_parse/crash-41592bf8…`): in operand
/// position `NOT` is a column name when no operand follows as a boolean (`1 = NOT + 1`: Spark 4.2
/// answers `UNRESOLVED_COLUMN` for `NOT`, checked on a real run), so the expression's text on its
/// own (`NOT + 1`, a logical NOT of `+ 1`) reads differently. The span check accepts a text that
/// parses to the node in any of its read-back contexts, the operand one included.
#[test]
fn a_keyword_read_as_a_name_in_operand_position_passes_the_span_check() -> R {
    let options = krishiv_sql::ParserOptions::default();
    for sql in [
        "SELECT 1 + 11++ 1= NOT+ +   + 1 AS ten",
        "SELECT 1 = NOT + 1",
        "SELECT 1 = NOT+ + 1",
        "SELECT NOT+ +  + 1",
    ] {
        let q = parse_query(sql, &options).map_err(|e| format!("{sql}: {e:?}"))?;
        check_spans(sql, &q, &options).map_err(|e| format!("{sql}: {e}"))?;
    }
    Ok(())
}
