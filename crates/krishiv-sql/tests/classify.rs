// SPDX-License-Identifier: Apache-2.0
//! The statement classifier against its goldens
//! (`tests/fixtures/statement-classes.tsv`): every line's statement parses,
//! has the branch its label names and the class it expects; and every
//! `Statement` variant and every parser branch the classifier
//! distinguishes (the variants of the sub-kinds it looks at, and the
//! `TEMPORARY`/`GLOBAL` flags) has at least one golden, read from
//! `src/ast.rs` so a new variant fails here until it has one.
#![expect(clippy::panic, reason = "test code")]

use std::collections::BTreeSet;

use krishiv_sql::ast::Statement;
use krishiv_sql::{
    ExplainKind, ParserOptions, StatementClass, classify, explain_kind, parse_statement,
    statement_queries,
};

const GOLDENS: &str = include_str!("fixtures/statement-classes.tsv");
const AST: &str = include_str!("../src/ast.rs");

/// The sub-enums the classifier (or the read listing) branches on, by the
/// statement variant that holds them.
const BRANCHES: &[(&str, &str)] = &[
    ("Explain", "ExplainMode"),
    ("Set", "SetKind"),
    ("SetVariable", "SetVariableKind"),
    ("Show", "ShowKind"),
    ("Describe", "DescribeKind"),
    ("Cache", "CacheKind"),
    ("Refresh", "RefreshKind"),
    ("AlterTable", "AlterTableAction"),
    ("AlterView", "AlterViewAction"),
    ("AlterNamespace", "AlterNamespaceAction"),
    ("Merge", "MergeSource"),
];

/// Flag branches: the label and the field that must be `true`.
const FLAGS: &[&str] = &[
    "CreateTable::temporary",
    "CreatePermanentView::temporary",
    "CreatePermanentView::global",
];

/// Branches `parse_statement` never returns, with the statement that
/// reaches them and the error class it fails with instead: they stay
/// classified (the match is exhaustive), and the test checks they are still
/// rejected, so one that becomes reachable needs a golden.
const UNREACHABLE: &[(&str, &str, &str)] = &[(
    "AlterNamespace::DefaultCollation",
    "ALTER SCHEMA d DEFAULT COLLATION UTF8_LCASE",
    "UNSUPPORTED_FEATURE.SCHEMA_LEVEL_COLLATIONS",
)];

/// The variant names of `pub enum name` in `src/ast.rs`.
fn variants(name: &str) -> Vec<String> {
    let head = format!("pub enum {name} {{");
    let start = AST.find(&head).unwrap_or_else(|| panic!("no enum {name}"));
    let mut out = Vec::new();
    for line in AST[start + head.len()..].lines() {
        if line == "}" {
            break;
        }
        // Variants sit at one indent level; fields and docs do not start
        // with a capital at that level.
        let Some(rest) = line.strip_prefix("    ") else {
            continue;
        };
        if rest.starts_with(' ') {
            continue;
        }
        let ident: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if ident.starts_with(|c: char| c.is_ascii_uppercase()) {
            out.push(ident);
        }
    }
    assert!(!out.is_empty(), "enum {name} has no variants");
    out
}

fn parse_class(name: &str) -> StatementClass {
    const ALL: [StatementClass; 5] = [
        StatementClass::Read,
        StatementClass::Write,
        StatementClass::Ddl,
        StatementClass::Session,
        StatementClass::Admin,
    ];
    ALL.into_iter()
        .find(|c| c.name() == name)
        .unwrap_or_else(|| panic!("unknown class {name}"))
}

/// Whether the parsed `debug` text has the branch `label` names.
fn has_branch(debug: &str, label: &str) -> bool {
    let (top, sub) = label
        .split_once("::")
        .map_or((label, None), |(t, s)| (t, Some(s)));
    if !debug.starts_with(&format!("{top}(")) {
        return false;
    }
    let Some(sub) = sub else {
        return true;
    };
    if FLAGS.contains(&label) {
        return debug.contains(&format!("{sub}: true"));
    }
    // The sub-kind is the value of the statement's field that holds it.
    let field = match top {
        "Explain" => "mode: ",
        "Merge" => "source: ",
        "AlterTable" | "AlterView" | "AlterNamespace" => "action: ",
        _ => "kind: ",
    };
    let at = debug
        .find(field)
        .unwrap_or_else(|| panic!("{label}: no {field} in {debug}"));
    let rest = &debug[at + field.len()..];
    rest.starts_with(sub)
        && !rest[sub.len()..].starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
}

fn goldens() -> Vec<(&'static str, StatementClass, &'static str)> {
    GOLDENS
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut f = l.split('\t');
            let (Some(label), Some(class), Some(sql), None) =
                (f.next(), f.next(), f.next(), f.next())
            else {
                panic!("bad golden line: {l}");
            };
            (label, parse_class(class), sql)
        })
        .collect()
}

#[test]
fn every_golden_has_its_branch_and_class() {
    let options = ParserOptions::default();
    let mut failures = Vec::new();
    for (label, class, sql) in goldens() {
        let statement = match parse_statement(sql, &options) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{label}: {sql}: {}", e.error));
                continue;
            }
        };
        let debug = format!("{statement:?}");
        if !has_branch(&debug, label) {
            failures.push(format!("{label}: {sql} parsed as {debug}"));
        }
        let got = classify(&statement);
        if got != class {
            failures.push(format!("{label}: {sql}: {got}, expected {class}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn every_statement_variant_and_branch_has_a_golden() {
    let labels: BTreeSet<&str> = goldens().iter().map(|(l, _, _)| *l).collect();
    let tops: BTreeSet<&str> = labels
        .iter()
        .map(|l| l.split_once("::").map_or(*l, |(t, _)| t))
        .collect();
    let mut missing = Vec::new();
    for v in variants("Statement") {
        if !tops.contains(v.as_str()) {
            missing.push(v);
        }
    }
    for (top, sub_enum) in BRANCHES {
        for v in variants(sub_enum) {
            let label = format!("{top}::{v}");
            if !labels.contains(label.as_str()) {
                missing.push(label);
            }
        }
    }
    let options = ParserOptions::default();
    for (label, sql, class) in UNREACHABLE {
        let e = parse_statement(sql, &options).unwrap_err();
        assert_eq!(e.error.error_class().name(), *class, "{label}: {sql}");
        assert!(
            !labels.contains(label),
            "{label} has a golden: drop it from UNREACHABLE"
        );
    }
    let unreachable: BTreeSet<&str> = UNREACHABLE.iter().map(|(l, _, _)| *l).collect();
    missing.retain(|l| !unreachable.contains(l.as_str()));
    for flag in FLAGS {
        if !labels.contains(flag) {
            missing.push((*flag).to_owned());
        }
    }
    assert!(missing.is_empty(), "branches without a golden: {missing:?}");
}

#[test]
fn every_class_is_reachable_and_named() {
    let classes: BTreeSet<StatementClass> = goldens().iter().map(|(_, c, _)| *c).collect();
    assert_eq!(classes.len(), 5);
    for c in classes {
        assert_eq!(parse_class(c.name()), c);
        assert_eq!(c.to_string(), c.name());
    }
}

#[test]
fn several_statements_are_a_parse_error_never_a_partial_run() {
    let options = ParserOptions::default();
    for sql in [
        "SELECT 1; DROP TABLE t",
        "SELECT 1 ; INSERT INTO t SELECT 1",
        "SET a = b; DROP DATABASE d",
    ] {
        let e = parse_statement(sql, &options).unwrap_err();
        assert_eq!(e.error.error_class().name(), "PARSE_SYNTAX_ERROR", "{sql}");
    }
}

#[test]
fn explain_kind_and_statement_queries() {
    let options = ParserOptions::default();
    let parse = |sql: &str| parse_statement(sql, &options).unwrap();
    assert_eq!(explain_kind(&parse("SELECT 1")), None);
    assert_eq!(
        explain_kind(&parse("EXPLAIN SELECT 1")),
        Some(ExplainKind::Plan)
    );
    assert_eq!(
        explain_kind(&parse("EXPLAIN ANALYZE SELECT 1")),
        Some(ExplainKind::Analyze)
    );
    let count = |sql: &str| statement_queries(&parse(sql)).map(|q| q.len());
    for (sql, n) in [
        ("SELECT 1", Some(1)),
        ("EXPLAIN SELECT 1", Some(1)),
        ("CREATE TEMPORARY VIEW v AS SELECT 1", Some(1)),
        ("CREATE TABLE t AS SELECT 1", Some(1)),
        ("CREATE TABLE t (a INT)", Some(0)),
        ("CREATE VIEW db.v AS SELECT 1", Some(1)),
        ("SET VAR (v) = (SELECT 1)", Some(1)),
        ("SET VAR v = 1", None),
        ("DECLARE v INT", Some(0)),
        ("DECLARE v INT DEFAULT 1", None),
        ("INSERT INTO t SELECT 1", Some(1)),
        ("FROM s INSERT INTO t SELECT a", None),
        (
            "MERGE INTO t USING (SELECT 1 k) s ON t.k = s.k WHEN MATCHED THEN DELETE",
            Some(1),
        ),
        (
            "MERGE INTO t USING s ON t.k = s.k WHEN MATCHED THEN DELETE",
            Some(0),
        ),
        ("CACHE TABLE c AS SELECT 1", Some(1)),
        ("CACHE TABLE c", Some(0)),
        ("UNCACHE TABLE c", Some(0)),
        ("CLEAR CACHE", Some(0)),
        ("DESCRIBE QUERY SELECT 1", Some(1)),
        ("DESCRIBE TABLE t", Some(0)),
        ("DESCRIBE FUNCTION abs", Some(0)),
        ("DESCRIBE DATABASE d", Some(0)),
        ("ALTER VIEW v AS SELECT 1", Some(1)),
        ("ALTER VIEW v RENAME TO w", Some(0)),
        ("DELETE FROM t", Some(0)),
        ("DELETE FROM t WHERE a = 1", None),
        ("UPDATE t SET a = 1", None),
        ("DROP TABLE t", Some(0)),
    ] {
        assert_eq!(count(sql), n, "{sql}");
    }
    // `EXPLAIN` lists the queries of the statement it explains.
    let Statement::Explain(e) = parse("EXPLAIN INSERT INTO t SELECT 1") else {
        panic!("not an EXPLAIN");
    };
    assert_eq!(statement_queries(&e.statement).map(|q| q.len()), Some(1));
}
