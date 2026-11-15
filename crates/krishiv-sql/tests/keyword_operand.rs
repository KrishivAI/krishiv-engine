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

/// `NOT <word>` outside a select item, and a `CASE` that starts a simple
/// `CASE`'s operand: the operator wherever an expression may end after the
/// word (`WHERE NOT END`, `x -> NOT AND`, `WHEN NOT END THEN`), the column
/// when the word closes the `CASE` part or an operand follows (`ELSE NOT
/// END`, `WHERE NOT IS NULL`). Each plan is the shape Spark 4.2.0's parser
/// gives (the re-review rows of
/// `ops/reviews/2026-11-14-opus-3-sql3-lambda-keywords-diff.md`).
#[test]
fn not_takes_a_keyword_operand_where_spark_does() {
    use krishiv_plan::SourceId;
    for (sql, want) in [
        (
            r#"SELECT 1 FROM t WHERE NOT AND (x)"#,
            r#"Project [1] | +- Filter (NOT AND(x)) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT OR (x)"#,
            r#"Project [1] | +- Filter (NOT OR(x)) |    +- Read t"#,
        ),
        (
            r#"SELECT NOT AND (x)"#,
            r#"Project [(NOT AND(x))] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT transform(array(1), x -> NOT AND (x))"#,
            r#"Project [transform(array(1), (x) -> (NOT AND(x)))] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT IS (x)"#,
            r#"Project [1] | +- Filter (NOT IS(x)) |    +- Read t"#,
        ),
        (
            r#"SELECT CASE WHEN a THEN NOT END (x) END"#,
            r#"Project [CASE WHEN a THEN (NOT END(x)) END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT IS IN (1)"#,
            r#"Project [1] | +- Filter (NOT (IS IN (1))) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT AND IS NULL"#,
            r#"Project [1] | +- Filter (NOT isnull(AND)) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT END BETWEEN 1 AND 2"#,
            r#"Project [1] | +- Filter (NOT (END BETWEEN 1 AND 2)) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT IS.a"#,
            r#"Project [1] | +- Filter (NOT IS.a) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT END[0]"#,
            r#"Project [1] | +- Filter (NOT __subscript(END, 0)) |    +- Read t"#,
        ),
        (
            r#"SELECT transform(array(1), x -> NOT END IS NULL)"#,
            r#"Project [transform(array(1), (x) -> (NOT isnull(END)))] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT OR LIKE 'a'"#,
            r#"Project [1] | +- Filter (NOT (OR like 'a')) |    +- Read t"#,
        ),
        (
            r#"SELECT CASE WHEN a THEN NOT END IS NULL END"#,
            r#"Project [CASE WHEN a THEN (NOT isnull(END)) END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT IS NOT NULL"#,
            r#"Project [1] | +- Filter isnotnull(NOT) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT END NOT IN (1)"#,
            r#"Project [1] | +- Filter (NOT (NOT (END IN (1)))) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT AND DIV 2"#,
            r#"Project [1] | +- Filter (NOT (AND div 2)) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT END::int"#,
            r#"Project [1] | +- Filter (NOT CAST(END AS INT)) |    +- Read t"#,
        ),
        (
            r#"SELECT CASE WHEN a THEN 1 ELSE NOT END.x END"#,
            r#"Project [CASE WHEN a THEN 1 ELSE (NOT END.x) END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT END"#,
            r#"Project [1] | +- Filter (NOT END) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT THEN"#,
            r#"Project [1] | +- Filter (NOT THEN) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT AND"#,
            r#"Project [1] | +- Filter (NOT AND) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT OR"#,
            r#"Project [1] | +- Filter (NOT OR) |    +- Read t"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT IS"#,
            r#"Project [1] | +- Filter (NOT IS) |    +- Read t"#,
        ),
        (
            r#"SELECT transform(array(1), x -> NOT END)"#,
            r#"Project [transform(array(1), (x) -> (NOT END))] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT transform(array(1), x -> NOT AND)"#,
            r#"Project [transform(array(1), (x) -> (NOT AND))] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT CASE WHEN NOT END THEN 1 END"#,
            r#"Project [CASE WHEN (NOT END) THEN 1 END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT CASE WHEN NOT AND THEN 1 END"#,
            r#"Project [CASE WHEN (NOT AND) THEN 1 END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT 1 FROM t ORDER BY NOT END"#,
            r#"Sort [(NOT END) ASC NULLS FIRST], true | +- Project [1] |    +- Read t"#,
        ),
        (
            r#"SELECT (NOT END)"#,
            r#"Project [(NOT END)] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT IS NULL"#,
            r#"Project [1] | +- Filter isnull(NOT) |    +- Read t"#,
        ),
        (
            r#"SELECT transform(array(1), x -> NOT IS NULL)"#,
            r#"Project [transform(array(1), (x) -> isnull(NOT))] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT AND x"#,
            r#"Project [1] | +- Filter (NOT and x) |    +- Read t"#,
        ),
        (
            r#"SELECT CASE WHEN NOT THEN x END"#,
            r#"Project [CASE WHEN NOT THEN x END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT CASE WHEN a THEN NOT END END"#,
            r#"Project [CASE WHEN a THEN (NOT END) END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT CASE CASE - 1 WHEN 0 THEN 1 END WHEN 1 THEN 'a' END"#,
            r#"Project [CASE WHEN (CASE WHEN (-1 == 0) THEN 1 END == 1) THEN 'a' END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT 1 FROM t WHERE NOT END AND x"#,
            r#"Project [1] | +- Filter ((NOT END) and x) |    +- Read t"#,
        ),
        (
            r#"SELECT CASE WHEN a THEN 1 ELSE NOT END END"#,
            r#"Project [CASE WHEN a THEN 1 ELSE (NOT END) END] | +- OneRowRelation"#,
        ),
        (
            r#"SELECT CASE WHEN a THEN 1 ELSE NOT END"#,
            r#"Project [CASE WHEN a THEN 1 ELSE NOT END] | +- OneRowRelation"#,
        ),
    ] {
        let plan = krishiv_sql::parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        let got = plan.root().to_string().replace('\n', " | ");
        assert_eq!(got.trim_end_matches([' ', '|']), want, "{sql}");
    }
    // `NOT and 'x'` is the operator over a typed literal of the unknown
    // type `and`: an error (Spark's class is UNSUPPORTED_TYPED_LITERAL).
    assert!(
        parse_query(
            "SELECT 1 FROM t WHERE NOT AND 'x'",
            &ParserOptions::default()
        )
        .is_err()
    );
    // After a failed `CASE` reading, the error is at the furthest token
    // any reading reached, as in Spark.
    let sql = "SELECT transform(array(1), case -> case - 1 WHEN 0 THEN 1)";
    let e = parse_query(sql, &ParserOptions::default()).unwrap_err();
    assert_eq!((e.span.start, e.span.end), (57, 58), "{sql}");
}

/// A `CASE` that may be a column is read once per position: a long chain of
/// them parses in linear time (each try of the `CASE` reading used to
/// re-read every later one, doubling the time per term).
#[test]
fn chains_of_case_columns_parse_in_linear_time() {
    let terms = vec!["case"; 256].join(" - ");
    for sql in [
        format!("SELECT transform(array(1), case -> {terms})"),
        format!("SELECT x FROM t WHERE {terms} > 0"),
        format!("SELECT {}1", "CASE WHEN - ".repeat(64)),
    ] {
        let started = std::time::Instant::now();
        let _ = parse_query(&sql, &ParserOptions::default());
        // Generous: linear parsing takes milliseconds; the old doubling
        // would never finish.
        assert!(started.elapsed().as_secs() < 30, "{}", &sql[..40]);
    }
}

/// Whether a `WHEN` follows a `case` comes from a table built once per
/// statement: a wide projection of `case` columns parses in linear time
/// (Codex on #761: each column rescanned the rest of the statement), and the
/// table keeps the bracket levels apart.
#[test]
fn wide_projections_of_case_columns_parse_in_linear_time() {
    let cols = vec!["case"; 100_000].join(", ");
    let sql = format!("SELECT {cols} FROM t");
    let started = std::time::Instant::now();
    let parsed = parse_query(&sql, &ParserOptions::default());
    assert!(parsed.is_ok(), "{:?}", parsed.err());
    // Generous: linear parsing takes well under a second; the rescan was
    // 5 * 10^9 token visits.
    assert!(started.elapsed().as_secs() < 10);
    for (sql, want) in [
        // A `WHEN` outside the brackets of the `case` does not count.
        ("SELECT f(case), CASE WHEN a THEN 1 END FROM t", true),
        ("SELECT f(case - 1) + CASE WHEN a THEN 1 END FROM t", true),
        // One inside nested brackets does not either.
        (
            "SELECT case FROM t WHERE x IN (SELECT CASE WHEN a THEN 1 END)",
            true,
        ),
        ("SELECT CASE (case) WHEN 1 THEN 2 END FROM t", true),
    ] {
        assert_eq!(shape(sql).is_ok(), want, "{sql}: {:?}", shape(sql));
    }
}

/// `NOT <closer> <binary operator>` inside a `CASE` part: the operator over
/// the operation when a closer of the part follows, the column when none is
/// left (`ELSE not END AND x FROM t`). Codex on #761: `THEN NOT end = 1 END`
/// failed. Each plan is the shape Spark 4.2.0's `parsePlan` gives.
#[test]
fn not_before_a_closer_and_an_operator_follows_spark() {
    use krishiv_plan::SourceId;
    let t = " | +- Read t";
    for (sql, want) in [
        (
            "SELECT CASE WHEN true THEN NOT end = 1 END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == 1)) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end + 1 END FROM t",
            "Project [CASE WHEN true THEN (NOT (end + 1)) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end AND x END FROM t",
            "Project [CASE WHEN true THEN ((NOT end) and x) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end < 1 END FROM t",
            "Project [CASE WHEN true THEN (NOT (end < 1)) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end || 'a' END FROM t",
            "Project [CASE WHEN true THEN (NOT concat(end, 'a')) END]",
        ),
        (
            "SELECT CASE x WHEN 1 THEN NOT else = 2 ELSE 3 END FROM t",
            "Project [CASE WHEN (x == 1) THEN (NOT (else == 2)) ELSE 3 END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT when = 1 END FROM t",
            "Project [CASE WHEN true THEN (NOT (when == 1)) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end = 1 ELSE NOT else = 2 END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == 1)) ELSE (NOT (else == 2)) END]",
        ),
        (
            "SELECT CASE WHEN a THEN 1 ELSE not END AND x FROM t",
            "Project [(CASE WHEN a THEN 1 ELSE not END and x)]",
        ),
        (
            "SELECT CASE WHEN a THEN NOT end = CASE WHEN b THEN 2 END END FROM t",
            "Project [CASE WHEN a THEN (NOT (end == CASE WHEN b THEN 2 END)) END]",
        ),
        // Codex's query (the same CASE inside a call).
        (
            "SELECT coalesce(CASE WHEN true THEN NOT end = 1 END, false) FROM t",
            "Project [coalesce(CASE WHEN true THEN (NOT (end == 1)) END, false)]",
        ),
    ] {
        let plan = krishiv_sql::parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        let got = plan.root().to_string().replace('\n', " | ");
        assert_eq!(
            got.trim_end_matches([' ', '|']),
            format!("{want}{t}"),
            "{sql}"
        );
    }
}

/// Codex on #761, round 2: a qualified `t.case` is a column, not a nested
/// `CASE`, for the closer look-ahead of `NOT`; and a `CASE` reading that
/// runs into the depth cap while speculating falls back to the column (a
/// flat `OR` chain of `case + 1 > 0` is shallow).
#[test]
fn case_columns_do_not_count_as_nested_case() {
    use krishiv_plan::SourceId;
    let sql = "SELECT CASE WHEN true THEN NOT end = t.case END FROM t";
    let plan = krishiv_sql::parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let got = plan.root().to_string().replace('\n', " | ");
    assert_eq!(
        got.trim_end_matches([' ', '|']),
        "Project [CASE WHEN true THEN (NOT (end == t.case)) END] | +- Read t"
    );
    let chain = vec!["case + 1 > 0"; 2000].join(" OR ");
    let sql = format!("SELECT 1 FROM t WHERE {chain}");
    if let Err(e) = parse_query(&sql, &ParserOptions::default()) {
        panic!("{e}");
    }
    // With a `WHEN` ahead each `case` is read speculatively; a read that
    // reaches the depth cap falls back to the column.
    let chain = vec!["case + 1 > 0"; 300].join(" OR ");
    let sql = format!("SELECT 1 FROM t WHERE {chain} OR CASE WHEN a THEN true END");
    let started = std::time::Instant::now();
    if let Err(e) = parse_query(&sql, &ParserOptions::default()) {
        panic!("{e}");
    }
    assert!(started.elapsed().as_secs() < 30);
}

/// Codex on #761, round 3: a `case` column in arithmetic is not a nested
/// `CASE` for the closer look-ahead of `NOT`; one with no `WHEN` after it,
/// or one the column rule takes (`case +`, `case *`, `case ||`, `case /`,
/// `case =`, `case.x`), leaves the real `END` to close the part. Ground
/// truth: Spark 4.2.0 `parsePlan` (`'Project [unresolvedalias(CASE WHEN
/// true THEN NOT ('end = ('case + 1)) END)]` and so on).
#[test]
fn case_columns_in_arithmetic_do_not_count_as_nested_case() {
    use krishiv_plan::SourceId;
    let t = " | +- Read t";
    for (sql, want) in [
        (
            "SELECT CASE WHEN true THEN NOT end = case + 1 END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == (case + 1))) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end = case - 1 END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == (case - 1))) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end = case * case END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == (case * case))) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end = case || 'x' END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == concat(case, 'x'))) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end = case.x END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == case.x)) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end = case / 2 END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == (case / 2))) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end = case = 1 END FROM t",
            "Project [CASE WHEN true THEN (NOT ((end == case) == 1)) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end = case + 1 ELSE 0 END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == (case + 1))) ELSE 0 END]",
        ),
        // Real nested `CASE`s still count.
        (
            "SELECT CASE WHEN true THEN NOT end = CASE WHEN a THEN 1 END END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == CASE WHEN a THEN 1 END)) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end = CASE x WHEN 1 THEN 2 END END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == CASE WHEN (x == 1) THEN 2 END)) END]",
        ),
        (
            "SELECT CASE WHEN true THEN NOT end = case - x WHEN 1 THEN 2 END END FROM t",
            "Project [CASE WHEN true THEN (NOT (end == CASE WHEN ((- x) == 1) THEN 2 END)) END]",
        ),
    ] {
        let plan = krishiv_sql::parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        let got = plan.root().to_string().replace('\n', " | ");
        assert_eq!(
            got.trim_end_matches([' ', '|']),
            format!("{want}{t}"),
            "{sql}"
        );
    }
}

/// Codex on #761, round 3: whether a `NOT <closer>` is the operator reads
/// the bracket depth and the closers ahead from tables built once per
/// statement, so a long `OR` chain of `NOT end = 1` inside a `CASE` parses in
/// linear time (each `NOT` rescanned the part from its start and the rest of
/// the statement), and keeps Spark's tree (`((NOT ('end = 1) OR NOT ('end =
/// 1)) OR NOT ('end = 1))`, Spark 4.2.0 `parsePlan`).
#[test]
fn not_closer_chains_parse_in_linear_time() {
    use krishiv_plan::SourceId;
    let sql = "SELECT CASE WHEN true THEN NOT end = 1 OR NOT end = 1 OR NOT end = 1 END FROM t";
    let plan = krishiv_sql::parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let got = plan.root().to_string().replace('\n', " | ");
    assert_eq!(
        got.trim_end_matches([' ', '|']),
        "Project [CASE WHEN true THEN (((NOT (end == 1)) or (NOT (end == 1))) or (NOT (end == 1))) END] | +- Read t"
    );
    let chain = vec!["NOT end = 1"; 60_000].join(" OR ");
    let sql = format!("SELECT CASE WHEN true THEN {chain} END FROM t");
    let started = std::time::Instant::now();
    if let Err(e) = parse_query(&sql, &ParserOptions::default()) {
        panic!("{e}");
    }
    // Generous: linear parsing takes well under a second; the rescans were
    // about 10^10 token visits.
    assert!(started.elapsed().as_secs() < 10);
}

/// A `CASE` that may be a column inside a `CASE` branch is the nested `CASE`
/// expression only when the enclosing `CASE` can still close after it; when
/// the nested reading would take the enclosing branches and `END`, it is the
/// column (Codex on #761, round 4). Each lowered plan is the shape Spark
/// 4.2.0's `parsePlan` gives (run 2026-11-14,
/// `~/krishiv-data/tmp/opus-3/r4/p5.out`).
#[test]
fn a_case_column_inside_a_case_branch_leaves_the_enclosing_case_whole() {
    use krishiv_plan::SourceId;
    for (sql, want) in [
        (
            "SELECT CASE WHEN true THEN case + 1 WHEN false THEN 2 END FROM t",
            "Project [CASE WHEN true THEN (case + 1) WHEN false THEN 2 END] | +- Read t",
        ),
        (
            "SELECT CASE WHEN true THEN case + 1 WHEN false THEN 2 ELSE 3 END FROM t",
            "Project [CASE WHEN true THEN (case + 1) WHEN false THEN 2 ELSE 3 END] | +- Read t",
        ),
        (
            "SELECT CASE WHEN true THEN case + 1 ELSE 3 END FROM t",
            "Project [CASE WHEN true THEN (case + 1) ELSE 3 END] | +- Read t",
        ),
        (
            "SELECT CASE WHEN true THEN case + 1 WHEN false THEN 2 END END FROM t",
            "Project [CASE WHEN true THEN CASE WHEN (positive(1) == false) THEN 2 END END] | +- Read t",
        ),
        // Spark 4.2.0: `CASE WHEN 'a THEN ((CASE WHEN (-'x = 1) THEN 2 END +
        // 'case) - 'y) END`, the second `case` a column.
        (
            "SELECT CASE WHEN a THEN case - x WHEN 1 THEN 2 END + case - y END FROM t",
            "Project [CASE WHEN a THEN ((CASE WHEN ((- x) == 1) THEN 2 END + case) - y) END] | +- Read t",
        ),
        // Spark 4.2.0 parsePlan, sibling CASEs after the branch (Codex on #761,
        // round 5): `'CASE` is a column in the first.
        (
            "SELECT CASE WHEN true THEN case + 1 WHEN false THEN 2 END + CASE WHEN true THEN 3 END FROM t",
            "Project [CASE WHEN true THEN (CASE WHEN (positive(1) == false) THEN 2 END + CASE) WHEN true THEN 3 END] | +- Read t",
        ),
        (
            "SELECT CASE WHEN true THEN case + 1 WHEN false THEN 2 END, CASE WHEN true THEN 3 END FROM t",
            "Project [CASE WHEN true THEN (case + 1) WHEN false THEN 2 END, CASE WHEN true THEN 3 END] | +- Read t",
        ),
        (
            "SELECT CASE WHEN true THEN case + 1 WHEN false THEN 2 END + CASE x WHEN 1 THEN 3 END FROM t",
            "Project [(CASE WHEN true THEN (case + 1) WHEN false THEN 2 END + CASE WHEN (x == 1) THEN 3 END)] | +- Read t",
        ),
        (
            "SELECT CASE WHEN true THEN case + 1 WHEN false THEN 2 END END + CASE WHEN true THEN 3 END FROM t",
            "Project [(CASE WHEN true THEN CASE WHEN (positive(1) == false) THEN 2 END END + CASE WHEN true THEN 3 END)] | +- Read t",
        ),
        // Spark 4.2.0: a searched CASE in a branch is the column `'CASE` when
        // its END would leave the enclosing CASE none.
        (
            "SELECT CASE WHEN a THEN CASE WHEN b THEN 1 END FROM t",
            "Project [CASE WHEN a THEN CASE WHEN b THEN 1 END] | +- Read t",
        ),
        (
            "SELECT CASE WHEN a THEN CASE WHEN b THEN 1 END WHEN c THEN 2 END FROM t",
            "Project [CASE WHEN a THEN CASE WHEN b THEN 1 END WHEN c THEN 2 END] | +- Read t",
        ),
        (
            "SELECT CASE WHEN a THEN 1 + CASE WHEN b THEN 2 END END FROM t",
            "Project [CASE WHEN a THEN (1 + CASE WHEN b THEN 2 END) END] | +- Read t",
        ),
        (
            "SELECT case + 1 WHEN 2 THEN 3 END FROM t",
            "Project [CASE WHEN (positive(1) == 2) THEN 3 END] | +- Read t",
        ),
    ] {
        let plan = krishiv_sql::parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        let got = plan.root().to_string().replace('\n', " | ");
        assert_eq!(got.trim_end_matches([' ', '|']), want, "{sql}");
    }
}

/// Review of round 4, finding 3 (Spark 4.2.0 parsePlan, `local[1]`): a later
/// select item and `MERGE` clauses after a `CASE` whose branch holds a `case`
/// column.
#[test]
fn a_case_column_before_later_items_and_merge_clauses() {
    use krishiv_plan::SourceId;
    for (sql, want) in [
        (
            "SELECT CASE WHEN true THEN case + 1 WHEN false THEN 2 END, case + 1 WHEN 2 THEN 3 END FROM t",
            "Project [CASE WHEN true THEN (case + 1) WHEN false THEN 2 END, CASE WHEN (positive(1) == 2) THEN 3 END] | +- Read t",
        ),
        (
            "SELECT CASE WHEN true THEN case + 1 WHEN false THEN 2 END, case + 1 FROM t",
            "Project [CASE WHEN true THEN (case + 1) WHEN false THEN 2 END, (case + 1)] | +- Read t",
        ),
    ] {
        let plan = krishiv_sql::parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        let got = plan.root().to_string().replace('\n', " | ");
        assert_eq!(got.trim_end_matches([' ', '|']), want, "{sql}");
    }
    let opts = ParserOptions::default();
    // Spark: `deleteaction(Some((CASE WHEN true THEN ('case + 1) WHEN false
    // THEN 2 END > 0)))`, then the insert.
    for sql in [
        "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED AND CASE WHEN true THEN case + 1 WHEN false THEN 2 END > 0 THEN DELETE WHEN NOT MATCHED THEN INSERT *",
        "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET x = CASE WHEN true THEN case + 1 WHEN false THEN 2 END",
    ] {
        krishiv_sql::parse_statement(sql, &opts).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    // Spark reads the nested `CASE` here and fails at the end of the input.
    let sql = "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET x = CASE WHEN true THEN case + 1 WHEN false THEN 2 END WHEN NOT MATCHED THEN INSERT *";
    let err = krishiv_sql::parse_statement(sql, &opts).expect_err(sql);
    assert!(err.to_string().contains("end of input"), "{err}");
}
