// SPDX-License-Identifier: Apache-2.0
//! The `krishiv-sql` parser on the DQL statements of the vendored Spark
//! `SQLQueryTestSuite` goldens:
//!
//! * acceptance against Spark: statements Spark parsed must parse (≥ 99.9%,
//!   every failure listed), and the syntax errors Spark raised are counted;
//! * the differential `==` check: each statement re-lexed with comments
//!   between all tokens parses to an `==` tree (equality ignores spans);
//! * spans: every node's span lies on token boundaries and inside its
//!   parent's span, and every struct node has one.

use std::fmt::Write as _;

use krishiv_sql::ast::Query;
use krishiv_sql::lexer::{Token, tokenize};
use krishiv_sql::parse_query;
use krishiv_testkit::golden::parse::Expected;
use krishiv_testkit::golden::run::{Engine as _, load_statements};
use krishiv_testkit::golden::setup::FileSetup;
use krishiv_testkit::golden::sql_parse::{ParseOnlyEngine, in_dql_scope};
use krishiv_testkit::golden::{DEFAULT_TAG, GoldenSet};

type R = Result<(), Box<dyn std::error::Error>>;

/// Enum struct variants: written `Name { .. }` in `Debug`, but not nodes.
const SPANLESS: &[&str] = &[
    "SetOp",
    "Wildcard",
    "IsNull",
    "IsTruth",
    "IsDistinctFrom",
    "Between",
    "InList",
    "InSubquery",
    "Like",
    "LikeQuantified",
    "RLike",
    "Cast",
    "Case",
    "Lambda",
    "Number",
    "Typed",
    "Primitive",
    "Function",
];

struct Frame {
    name: String,
    span: Option<(usize, usize)>,
    children: Vec<(usize, usize)>,
}

/// Checks the spans of `q` parsed from `sql` (see the module docs).
fn check_spans(sql: &str, q: &Query) -> Result<(), String> {
    let tokens: Vec<Token> = tokenize(sql).map_err(|e| e.to_string())?;
    let starts: Vec<usize> = tokens.iter().map(|t| t.span.start).collect();
    let ends: Vec<usize> = tokens.iter().map(|t| t.span.end).collect();
    let debug = format!("{q:#?}");
    let mut stack = vec![Frame {
        name: "root".into(),
        span: None,
        children: vec![],
    }];
    for line in debug.lines().map(str::trim) {
        if let Some(range) = line.strip_prefix("span: ").map(|r| r.trim_end_matches(',')) {
            let (a, b) = range.split_once("..").ok_or("bad span")?;
            let span = (a.parse().map_err(|_| "bad")?, b.parse().map_err(|_| "bad")?);
            if let Some(top) = stack.last_mut() {
                top.span = Some(span);
            }
        } else if line.ends_with('{') || line.ends_with('(') || line.ends_with('[') {
            let name = line.trim_end_matches(['{', '(', '[']).trim();
            let name = name.rsplit(": ").next().unwrap_or(name).to_owned();
            let struct_like = line.ends_with('{');
            stack.push(Frame {
                name: if struct_like { name } else { String::new() },
                span: None,
                children: vec![],
            });
        } else if line.starts_with(['}', ')', ']']) {
            let f = stack.pop().ok_or("unbalanced")?;
            let parent = stack.last_mut().ok_or("unbalanced")?;
            match f.span {
                Some((a, b)) => {
                    if !(a < b && starts.contains(&a) && ends.contains(&b)) {
                        return Err(format!("{} span {a}..{b} not on tokens", f.name));
                    }
                    if let Some(c) = f.children.iter().find(|(ca, cb)| *ca < a || *cb > b) {
                        return Err(format!(
                            "{} span {a}..{b} does not cover child {c:?}",
                            f.name
                        ));
                    }
                    parent.children.push((a, b));
                }
                None if !f.name.is_empty() && !SPANLESS.contains(&f.name.as_str()) => {
                    return Err(format!("struct {} has no span", f.name));
                }
                None => parent.children.extend(f.children),
            }
        }
    }
    let (a, b) = (q.span.start, q.span.end);
    if a != starts.first().copied().unwrap_or(0) || !ends.contains(&b) {
        return Err(format!(
            "query span {a}..{b} does not start at the first token"
        ));
    }
    Ok(())
}

/// `sql` with a comment between every two tokens.
fn respaced(sql: &str) -> Option<String> {
    let tokens = tokenize(sql).ok()?;
    let parts: Vec<&str> = tokens.iter().map(|t| t.span.text(sql)).collect();
    Some(parts.join(" /* x */\n"))
}

#[test]
fn dql_corpus_parses_with_spans_and_differential_equality() -> R {
    let set = GoldenSet::scan(&GoldenSet::tag_dir(DEFAULT_TAG))?;
    let mut engine = ParseOnlyEngine::default();
    let (mut total, mut parsed, mut negatives, mut rejected) = (0_u64, 0_u64, 0_u64, 0_u64);
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
                    negatives += 1;
                    rejected += u64::from(result.is_err());
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
                check_spans(&s.query, &q)
                    .map_err(|e| format!("{}: {e}\n{}", pair.name, s.query))?;
                let again = respaced(&s.query).map(|r| parse_query(&r, &options));
                assert!(
                    matches!(&again, Some(Ok(r)) if *r == q),
                    "{}: differential mismatch\n{}",
                    pair.name,
                    s.query
                );
            }
        }
    }
    println!(
        "DQL corpus: {parsed} / {total} parse ({:.3}%); Spark parse errors rejected: {rejected} / {negatives}\n{failures}",
        100.0 * parsed as f64 / total as f64
    );
    assert!(total > 13_000, "too few statements: {total}");
    assert!(
        parsed * 1000 >= total * 999,
        "parse rate below 99.9%:\n{failures}"
    );
    Ok(())
}

#[test]
fn span_checker_rejects_bad_spans() -> R {
    let sql = "SELECT a + 1 AS x FROM t";
    let mut q = parse_query(sql, &Default::default())?;
    check_spans(sql, &q)?;
    q.span.end = 3;
    assert!(check_spans(sql, &q).is_err());
    assert_eq!(respaced("SELECT 1").as_deref(), Some("SELECT /* x */\n1"));
    Ok(())
}
