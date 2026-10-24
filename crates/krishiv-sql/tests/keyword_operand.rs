// SPDX-License-Identifier: Apache-2.0
//! A keyword where an operand is expected reads as a column, and `NOT`
//! where nothing can follow it is a column too.
//!
//! Ground truth is a Spark 4.2.0 run (the Connect oracle, default
//! settings). `SELECT a/NOT FROM VALUES (1) t(a)`, `SELECT 1/not`,
//! `SELECT NOT`, `SELECT NOT = 1`, `SELECT NOT.x` and `SELECT 1 + <word>`
//! for every word below all parse and fail only in analysis
//! (`UNRESOLVED_COLUMN`, the word as the column name, error position
//! `line 1 pos` at the word), while `SELECT NOT TRUE`, `SELECT NOT + 1`
//! (`NOT (+1)`) and `SELECT NOT NOT TRUE` keep `NOT` an operator. `SELECT
//! NOT )`, `SELECT !` and `SELECT 1/NOT 2` are `PARSE_SYNTAX_ERROR`s.
//! This file pins the parse shape of each.
use krishiv_sql::ParserOptions;
use krishiv_sql::ast::{Expr, ExprKind, SetExprKind};
use krishiv_sql::parse_query;

/// The first select item, as a short s-expression.
fn shape(sql: &str) -> Result<String, String> {
    let q = parse_query(sql, &ParserOptions::default()).map_err(|e| e.to_string())?;
    let SetExprKind::Select(s) = q.body.kind else {
        return Err("not a select".into());
    };
    Ok(render(&s.projection[0].expr))
}

fn render(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Column(parts) => {
            let names: Vec<&str> = parts.iter().map(|p| p.value.as_str()).collect();
            format!("col({})", names.join("."))
        }
        ExprKind::Literal(_) => "lit".into(),
        ExprKind::Not(x) => format!("not({})", render(x)),
        ExprKind::Unary(op, x) => format!("{op:?}({})", render(x)),
        ExprKind::Binary(l, op, r) => format!("({} {op:?} {})", render(l), render(r)),
        ExprKind::Nested(x) => render(x),
        other => format!("{other:?}").chars().take(24).collect(),
    }
}

#[test]
fn not_is_an_operator_when_an_operand_follows() {
    for (sql, want) in [
        ("SELECT NOT TRUE", "not(lit)"),
        ("SELECT NOT NOT TRUE", "not(not(lit))"),
        ("SELECT NOT x", "not(col(x))"),
        ("SELECT NOT (TRUE)", "not(lit)"),
        ("SELECT NOT(1)", "not(lit)"),
        ("SELECT NOT NULL", "not(lit)"),
        ("SELECT NOT + 1", "not(Plus(lit))"),
        ("SELECT NOT - 1", "not(lit)"),
        ("SELECT ! TRUE", "not(lit)"),
        ("SELECT NOT NOT NOT", "not(not(col(NOT)))"),
    ] {
        assert_eq!(shape(sql).as_deref(), Ok(want), "{sql}");
    }
}

#[test]
fn not_is_a_column_when_no_operand_follows() {
    for (sql, want) in [
        ("SELECT NOT", "col(NOT)"),
        ("SELECT NOT,1", "col(NOT)"),
        ("SELECT NOT FROM VALUES (1) t(a)", "col(NOT)"),
        ("SELECT (NOT)", "col(NOT)"),
        ("SELECT NOT.x", "col(NOT.x)"),
        ("SELECT NOT = 1", "(col(NOT) Eq lit)"),
        ("SELECT (NOT) + 1", "(col(NOT) Plus lit)"),
        ("SELECT a/NOT", "(col(a) Divide col(NOT))"),
        ("SELECT a/NoT", "(col(a) Divide col(NoT))"),
        ("SELECT 1/not", "(lit Divide col(not))"),
        ("SELECT 1 + NOT", "(lit Plus col(NOT))"),
        ("SELECT 1 + (NOT)", "(lit Plus col(NOT))"),
        ("SELECT a AND NOT", "(col(a) And col(NOT))"),
        ("SELECT a OR NOT", "(col(a) Or col(NOT))"),
        ("SELECT -NOT", "Minus(col(NOT))"),
    ] {
        assert_eq!(shape(sql).as_deref(), Ok(want), "{sql}");
    }
}

#[test]
fn not_followed_by_nothing_valid_is_a_syntax_error() {
    for sql in ["SELECT NOT )", "SELECT !", "SELECT 1/NOT 2"] {
        let e = shape(sql).unwrap_err();
        assert!(e.contains("PARSE_SYNTAX_ERROR"), "{sql}: {e}");
    }
}

/// Every word Spark accepts as a column after a binary operator.
#[test]
fn keywords_in_operand_position_are_columns() {
    for word in [
        "NOT", "AND", "OR", "IS", "IN", "LIKE", "ILIKE", "RLIKE", "BETWEEN", "EXISTS", "CASE",
        "ANTI", "LEFT", "FROM", "WHERE", "SELECT", "ESCAPE", "DIV", "ALL", "NATURAL", "CROSS",
        "UNION", "ASC", "AS", "ANY", "SOME", "OVER",
    ] {
        let sql = format!("SELECT 1 + {word}");
        assert_eq!(shape(&sql), Ok(format!("(lit Plus col({word}))")), "{sql}");
        let lower = format!("SELECT 1 + {}", word.to_lowercase());
        assert_eq!(
            shape(&lower),
            Ok(format!("(lit Plus col({}))", word.to_lowercase())),
            "{lower}"
        );
    }
}
