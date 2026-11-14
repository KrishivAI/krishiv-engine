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
        ExprKind::Cast { expr, .. } => format!("cast({})", render(expr)),
        ExprKind::Subscript(base, index) => {
            format!("subscript({}, {})", render(base), render(index))
        }
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
        ("SELECT NOT::INT", "cast(col(NOT))"),
        ("SELECT NOT[0]", "subscript(col(NOT), lit)"),
    ] {
        assert_eq!(shape(sql).as_deref(), Ok(want), "{sql}");
    }
}

#[test]
fn not_followed_by_nothing_valid_is_a_syntax_error() {
    for (sql, want) in [
        (
            "SELECT NOT )",
            "[PARSE_SYNTAX_ERROR] Syntax error at or near ')': extra input ')'. SQLSTATE: 42601",
        ),
        (
            "SELECT !",
            "[PARSE_SYNTAX_ERROR] Syntax error at or near end of input. SQLSTATE: 42601",
        ),
        (
            "SELECT 1/NOT 2",
            "[PARSE_SYNTAX_ERROR] Syntax error at or near '2': extra input '2'. SQLSTATE: 42601",
        ),
    ] {
        assert_eq!(shape(sql).unwrap_err(), want, "{sql}");
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

/// A keyword before `-`, `.`, `[`, `]`, an infix keyword or a `CASE`
/// branch keyword is a column; `NOT` before IS, AND, OR, THEN, ELSE or END
/// is a column unless the operator can end a select item (`SELECT NOT
/// END`, `SELECT NOT AND x`); `CASE` that may be a column is the `CASE`
/// expression when it parses. Each lowered plan is the shape Spark 4.2.0's
/// parser gives (SQL-3; the review rows of
/// `ops/reviews/2026-11-14-opus-3-sql3-lambda-keywords-diff.md`).
#[test]
fn keywords_read_as_columns_as_spark_reads_them() {
    use krishiv_plan::SourceId;
    for (sql, want) in [
        (
            r#"SELECT NOT END"#,
            r#"Project [(NOT END)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT NOT THEN"#,
            r#"Project [(NOT THEN)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT NOT ELSE"#,
            r#"Project [(NOT ELSE)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT NOT AND"#,
            r#"Project [(NOT AND)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT NOT OR"#,
            r#"Project [(NOT OR)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT NOT AND x"#,
            r#"Project [(NOT AND) AS x] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT NOT AND x FROM t"#,
            r#"Project [(NOT AND) AS x] | +- Read t"#,
        ),
        (
            r#"SELECT a, NOT AND x FROM t"#,
            r#"Project [a, (NOT AND) AS x] | +- Read t"#,
        ),
        (
            r#"SELECT NOT OR x"#,
            r#"Project [(NOT OR) AS x] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT NOT IS NULL"#,
            r#"Project [(NOT IS) AS NULL] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT NOT AND x AS y"#,
            r#"Project [(NOT and x) AS y] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT NOT IS NULL AS y"#,
            r#"Project [isnull(NOT) AS y] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT NOT IS NOT NULL"#,
            r#"Project [isnotnull(NOT)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT a FROM t WHERE NOT AND x"#,
            r#"Project [a] | +- Filter (NOT and x) |    +- Read t"#,
        ),
        (
            r#"SELECT a FROM t ORDER BY NOT AND x"#,
            r#"Sort [(NOT and x) ASC NULLS FIRST], true | +- Project [a] |    +- Read t"#,
        ),
        (
            r#"SELECT a FROM t GROUP BY NOT AND x"#,
            r#"Aggregate [(NOT and x)], [a] | +- Read t"#,
        ),
        (
            r#"SELECT CASE WHEN NOT IS NULL THEN 1 END"#,
            r#"Project [CASE WHEN isnull(NOT) THEN 1 END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT CASE - x WHEN 1 THEN 2 END"#,
            r#"Project [CASE WHEN ((- x) == 1) THEN 2 END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT CASE NOT x WHEN true THEN 2 END"#,
            r#"Project [CASE WHEN ((NOT x) == true) THEN 2 END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT CASE - 1"#,
            r#"Project [(CASE - 1)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT count(DISTINCT, x)"#,
            r#"Project [count(DISTINCT, x)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT count(ALL, x)"#,
            r#"Project [count(ALL, x)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT count(DISTINCT)"#,
            r#"Project [count(DISTINCT)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT count(DISTINCT x)"#,
            r#"Project [count(DISTINCT x)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT case.a, from - 1, left - right, on[0], a[from], from IS NULL"#,
            r#"Project [case.a, (from - 1), (left - right), __subscript(on, 0), __subscript(a, from), isnull(from)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT INTERVAL - 1"#,
            r#"Project [(INTERVAL - 1)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT INTERVAL -1 DAY"#,
            r#"Project [INTERVAL '-1' DAY] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT x FROM t WHERE case - 1 > 0"#,
            r#"Project [x] | +- Filter ((case - 1) > 0) |    +- Read t"#,
        ),
        (
            r#"SELECT CASE WHEN a THEN CASE - 1 WHEN 0 THEN 1 END END"#,
            r#"Project [CASE WHEN a THEN CASE WHEN (-1 == 0) THEN 1 END END] | +- OneRowRelation"#,
        ),
    ] {
        let plan = krishiv_sql::parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        let got = plan.root().to_string().replace('\n', " | ");
        assert_eq!(got.trim_end_matches([' ', '|']), want, "{sql}");
    }
    // A simple `CASE`'s operand in `MERGE ... ON` and a lambda.
    for sql in [
        "MERGE INTO t USING s ON t.id = case - 1 WHEN MATCHED THEN DELETE",
        "SELECT transform(array(1), (case, y) -> CASE case - 1 WHEN 0 THEN 1 END)",
    ] {
        krishiv_sql::parse_statement(sql, &ParserOptions::default())
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    // `INTERVAL` is not reserved when keywords are enforced.
    let ansi = ParserOptions {
        enforce_reserved_keywords: true,
        ..ParserOptions::default()
    };
    krishiv_sql::parse_query("SELECT INTERVAL - 1", &ansi).unwrap();
}
