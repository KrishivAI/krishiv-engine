// SPDX-License-Identifier: Apache-2.0
//! Parser unit tests. Tree shapes (spans included) are `insta` snapshots,
//! one per feature group; errors, options, depth and lexing are asserted
//! directly.

use std::fmt::Write as _;

use krishiv_common::error::ErrorClass;

use super::ast::{BinaryOp, ExprKind, SetExprKind, Span};
use super::lexer::{TokenKind, tokenize, unescape_string};
use super::*;

fn ok(sql: &str) -> ast::Query {
    match parse_query(sql, &ParserOptions::default()) {
        Ok(q) => q,
        Err(e) => panic!("{sql}: {}", e.error),
    }
}

fn err(sql: &str, options: &ParserOptions) -> ParseError {
    match parse_query(sql, options) {
        Ok(q) => panic!("{sql} parsed: {q:?}"),
        Err(e) => e,
    }
}

fn snapshot(statements: &[&str]) -> String {
    let mut out = String::new();
    for sql in statements {
        writeln!(out, "-- {sql}\n{:#?}", ok(sql)).unwrap();
    }
    out
}

#[test]
fn clauses() {
    insta::assert_snapshot!(snapshot(&[
        "WITH RECURSIVE r (n) MAX RECURSION LEVEL 10 AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT * FROM r",
        "SELECT 1 UNION SELECT 2 INTERSECT DISTINCT SELECT 3 EXCEPT SELECT 4 MINUS SELECT 5",
        "VALUES (1, 'a'), (2, 'b') AS t(x, y)",
        "TABLE db.t",
        "(SELECT 1) ORDER BY 1 DESC NULLS LAST, 2 ASC NULLS FIRST LIMIT ALL OFFSET 3",
        "SELECT a FROM t CLUSTER BY a DISTRIBUTE BY b SORT BY c WINDOW w AS (ORDER BY a) LIMIT 1",
        "FROM t SELECT a WHERE a > 1 SELECT b ORDER BY b",
        "FROM t LATERAL VIEW explode(x) e AS c",
        "SELECT /*+ BROADCAST(t), REPARTITION(3, a) */ /*+ COALESCE(1) */ DISTINCT a b, c AS d, e AS (f, g), h (i) FROM t",
        "SELECT all, distinct FROM t",
        "SELECT a FROM t GROUP BY ALL HAVING count(*) > 1 QUALIFY row_number() OVER w = 1 WINDOW w AS (PARTITION BY a)",
        "SELECT a FROM t GROUP BY a, ROLLUP (b, (c, d)), CUBE (e), GROUPING SETS ((a), (), ROLLUP (b), GROUPING SETS (c))",
        "SELECT a FROM t GROUP BY a, b WITH ROLLUP",
        "SELECT a FROM t GROUP BY a WITH CUBE",
        "SELECT a FROM t GROUP BY a, b GROUPING SETS ((a), (a + b) * 2)",
        "SELECT a FROM t ORDER BY ALL",
        "SELECT a FROM, b AS, c NOT",
    ]));
}

#[test]
fn relations() {
    insta::assert_snapshot!(snapshot(&[
        "SELECT * FROM a JOIN b ON a.x = b.x INNER JOIN c USING (x) CROSS JOIN d LEFT OUTER JOIN e ON true RIGHT JOIN f FULL OUTER JOIN g",
        "SELECT * FROM a LEFT SEMI JOIN b SEMI JOIN c LEFT ANTI JOIN d ANTI JOIN e NATURAL JOIN f NATURAL LEFT JOIN g JOIN LATERAL (SELECT 1) h",
        "SELECT * FROM t, LATERAL (SELECT t.a) AS s (b), (a JOIN b) AS j, (VALUES 1) v",
        "SELECT * FROM range(1, end => 10) r, ((SELECT 1) UNION (SELECT 2)) u",
        "SELECT * FROM t PIVOT (sum(a) AS s, avg(b) FOR (c, d) IN ((1, 2) AS one, (3, 4) two)) p",
        "SELECT * FROM t PIVOT (sum(a) FOR c IN (1, 2))",
        "SELECT * FROM t UNPIVOT INCLUDE NULLS (v FOR n IN (a AS x, b)) AS u",
        "SELECT * FROM t UNPIVOT EXCLUDE NULLS ((v1, v2) FOR n IN ((a, b) AS ab, (c, d)))",
        "SELECT * FROM t LATERAL VIEW OUTER explode(m) e k, v LATERAL VIEW stack() s PIVOT (count(*) FOR k IN ('a')) UNPIVOT (x FOR y IN (z))",
        "SELECT * FROM from t, values, u AS, v WHERE",
        "FROM t",
        "SELECT * FROM t LATERAL VIEW explode(a) AS c",
    ]));
}

#[test]
fn expressions() {
    insta::assert_snapshot!(snapshot(&[
        "SELECT a OR b AND NOT c = d + e * - f | g ^ h & i << j >> k >>> l || m DIV n % o / p - ~q",
        "SELECT a AND b AND c AND d AND e, NOT NOT !x, a <=> b, a <> b, a != b, a <= b, a >= b, a !> b, a !< b, a == b",
        "SELECT a BETWEEN 1 AND 2, a NOT BETWEEN b AND c, a IN (), a NOT IN (1, 2), a IN (SELECT 1), a NOT IN (VALUES 1)",
        "SELECT a LIKE 'x' ESCAPE '/', a NOT ILIKE 'y', a LIKE ANY ('x', 'y'), a NOT ILIKE ALL (), a LIKE SOME ('z')",
        "SELECT a RLIKE 'x', a NOT REGEXP 'y', a IS NULL, a IS NOT NULL, a IS TRUE, a IS NOT FALSE, a IS UNKNOWN",
        "SELECT a IS DISTINCT FROM b, a IS NOT DISTINCT FROM b, NOT a IN (1) AND b OR c",
        "SELECT EXISTS (SELECT 1), (SELECT 1), ((SELECT 1)), (1, 2 AS b), (a), struct(), struct(1 AS a, b), exists(1)",
        "SELECT CASE WHEN a THEN 1 WHEN b THEN 2 ELSE 3 END, CASE a WHEN 1 THEN 'x' END",
        "SELECT a[1][b].c.d, f(x).y, t.*, t.* EXCEPT (a, b.c), * EXCEPT (d), ?",
        "SELECT x -> x + 1, (x, y) -> x * y, transform(a, all -> all * all)",
        "SELECT count(DISTINCT a), count(ALL b), count(*), count(ALL), sum(c) FILTER (WHERE d), first(e IGNORE NULLS), last(f) RESPECT NULLS",
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY a DESC), rank() OVER (PARTITION BY a ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)",
        "SELECT sum(a) OVER w, sum(a) OVER (w), sum(a) OVER (DISTRIBUTE BY a SORT BY b RANGE 1 PRECEDING), sum(a) OVER (CLUSTER BY a ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)",
        "SELECT left(a, 1), right(b, 2), db.f(1), a.left(1), if(a, 1, 2), all, all + 1",
        "SELECT current_date, current_timestamp, user, session_user, current_path, current_date()",
    ]));
}

#[test]
fn literals_and_types() {
    insta::assert_snapshot!(snapshot(&[
        "SELECT 1, -1, 1L, 2S, 3Y, 1.5, .5, 1., 1e3, 1.5E-3, 2D, 3F, 4BD, -1.5BD, - 2",
        "SELECT 'a' 'b' \"c\", R'\\n', DATE '2020-01-01', TIMESTAMP_NTZ '2020-01-01', X'1F', TIME '10:00', geometry 'POINT'",
        "SELECT NULL, TRUE, FALSE, INTERVAL '1 day', INTERVAL 1 DAY 2 HOURS, INTERVAL -'1-2' YEAR TO MONTH, INTERVAL '1' 'x' SECOND",
        "SELECT INTERVAL + 1.5 MINUTES - 3 SECOND, interval, interval.a, INTERVAL 1 DAY TO",
        "SELECT CAST(a AS DECIMAL(10, 2)), TRY_CAST(b AS ARRAY<MAP<STRING, STRUCT<x: INT NOT NULL COMMENT 'c', `y z` ARRAY<INT>>>>), c::STRUCT<>, d::array",
        "SELECT CAST(a AS TIMESTAMP WITH LOCAL TIME ZONE), CAST(b AS TIMESTAMP WITHOUT TIME ZONE), CAST(c AS TIME(6) WITHOUT TIME ZONE)",
        "SELECT CAST(a AS INTERVAL YEAR TO MONTH), CAST(b AS INTERVAL DAY TO SECOND), CAST(c AS INTERVAL HOUR), CAST(d AS INTERVAL), CAST(e AS geometry(ANY)), CAST(f AS map), CAST(g AS struct)",
    ]));
}

#[test]
fn special_forms() {
    insta::assert_snapshot!(snapshot(&[
        "SELECT extract(YEAR FROM d), extract('year', d), substring(s FROM 2 FOR 3), substr(s, 2), substring(s, 1, 2)",
        "SELECT trim(BOTH 'x' FROM s), trim(LEADING FROM s), trim(TRAILING 'y' FROM s), trim(s), trim(s, 'z')",
        "SELECT position('a' IN s), position('a', s), overlay(s PLACING 'x' FROM 2 FOR 1), overlay(s PLACING 'y' FROM 1)",
        "SELECT timestampadd(DAY, 1, t), dateadd(HOUR, 2, t), date_add(d, 1), timestampdiff(MONTH, a, b), datediff(year, 1)",
    ]));
}

#[test]
fn and_or_chains_are_balanced_like_spark() {
    let q = ok("SELECT a AND b AND c AND d AND e");
    let SetExprKind::Select(s) = &q.body.kind else {
        panic!("not a select");
    };
    // Spark's `reduceToExpressionTree(0, 4)`: ((a AND b) AND c) AND (d AND e).
    let ExprKind::Binary(l, BinaryOp::And, r) = &s.projection[0].expr.kind else {
        panic!("not AND");
    };
    let column = |e: &ast::Expr| matches!(e.kind, ExprKind::Column(_));
    assert!(matches!(&l.kind, ExprKind::Binary(ab, BinaryOp::And, c)
        if matches!(&ab.kind, ExprKind::Binary(a, BinaryOp::And, b) if column(a) && column(b)) && column(c)));
    assert!(matches!(&r.kind, ExprKind::Binary(d, BinaryOp::And, e) if column(d) && column(e)));
}

#[test]
fn equality_ignores_spans() {
    let a = ok("SELECT a+1 FROM t");
    let b = ok("SELECT  a /* x */ + 1\nFROM t;");
    assert_eq!(a, b);
    assert_ne!(a.span.range(), b.span.range());
    assert_ne!(a, ok("SELECT a+2 FROM t"));
    assert_eq!(Span::new(1, 2), Span::new(3, 4));
    let hash = |q: &ast::Query| {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        q.hash(&mut h);
        h.finish()
    };
    assert_eq!(hash(&a), hash(&b));
    assert_eq!(format!("{:?}", Span::new(1, 2)), "1..2");
    assert_eq!(Span::new(5, 9).to(Span::new(2, 6)).range(), 2..9);
    assert_eq!(Span::new(9, 12).text("abc"), "");
}

#[test]
fn spans_cover_source_text() {
    let sql = "SELECT  t.a AS x, f(b) FROM t";
    let q = ok(sql);
    let SetExprKind::Select(s) = &q.body.kind else {
        panic!("not a select");
    };
    assert_eq!(q.span.text(sql), sql);
    assert_eq!(s.projection[0].span.text(sql), "t.a AS x");
    assert_eq!(s.projection[0].expr.span.text(sql), "t.a");
    assert_eq!(s.projection[1].span.text(sql), "f(b)");
    assert_eq!(s.from.as_ref().unwrap().relations[0].span.text(sql), "t");
}

#[test]
fn options_follow_spark_confs() {
    let conf = |pairs: &'static [(&'static str, &'static str)]| {
        ParserOptions::from_conf(|k| pairs.iter().find(|(key, _)| *key == k).map(|(_, v)| *v))
    };
    assert_eq!(conf(&[]), ParserOptions::default());
    let all = conf(&[
        ("spark.sql.ansi.enforceReservedKeywords", "true"),
        ("spark.sql.ansi.doubleQuotedIdentifiers", " TRUE "),
        ("spark.sql.legacy.setopsPrecedence.enabled", "true"),
        ("spark.sql.parser.escapedStringLiterals", "true"),
    ]);
    assert!(all.enforce_reserved_keywords && all.double_quoted_identifiers);
    assert!(all.legacy_setops_precedence && all.escaped_string_literals);
    let off = conf(&[
        ("spark.sql.ansi.enabled", "false"),
        ("spark.sql.ansi.enforceReservedKeywords", "true"),
        ("spark.sql.ansi.doubleQuotedIdentifiers", "true"),
    ]);
    assert!(!off.enforce_reserved_keywords && !off.double_quoted_identifiers);

    let q = parse_query("SELECT \"a\"\"b\" FROM t WHERE c = 'x\\n'", &all).unwrap();
    let text = format!("{q:?}");
    assert!(text.contains("value: \"a\\\"b\", quoted: true"), "{text}");
    assert!(text.contains("x\\\\n"), "{text}");
    for sql in [
        "SELECT 1 AS select",
        "SELECT all FROM t",
        "SELECT * FROM from t",
        "SELECT all -> 1",
        "SELECT a FROM",
    ] {
        assert!(parse_query(sql, &all).is_err(), "{sql}");
    }
    let q = parse_query("SELECT current_date", &all).unwrap();
    assert!(format!("{q:?}").contains("Function("));
    // Legacy precedence: `1 UNION 2 INTERSECT 3` is `(1 UNION 2) INTERSECT 3`.
    let legacy = parse_query("SELECT 1 UNION SELECT 2 INTERSECT SELECT 3", &all).unwrap();
    assert!(matches!(
        legacy.body.kind,
        SetExprKind::SetOp {
            op: ast::SetOperator::Intersect,
            ..
        }
    ));
}

#[test]
fn syntax_errors_match_spark() {
    let d = ParserOptions::default();
    let cases = [
        ("SELECT (1", "end of input", "", 9..9),
        ("SELECT 1 abc def", "'def'", ": extra input 'def'", 13..16),
        ("SELECT 'a\nb' c d e", "'d'", "", 15..16),
        (
            "SELECT 1; SELECT 2",
            "'SELECT'",
            ": extra input 'SELECT'",
            10..16,
        ),
        ("SELECT 1 +", "end of input", "", 10..10),
        ("SELECT\t'unterminated", "'''", "", 7..8),
        ("SELECT (1 AS a)", "')'", "", 14..15),
        ("SELECT a.b.c d.e", "'.'", "", 14..15),
        ("-- only a comment", "end of input", "", 17..17),
    ];
    for (sql, found, hint, span) in cases {
        let e = err(sql, &d);
        assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError, "{sql}");
        assert_eq!(
            e.error.message_parameters(),
            [("error", found.to_owned()), ("hint", hint.to_owned())],
            "{sql}"
        );
        assert_eq!(e.span.range(), span, "{sql}");
    }
    let e = err("SELECT 1 abc\tdef\n", &d);
    assert_eq!(
        e.error.message(),
        "Syntax error at or near 'def': extra input 'def'."
    );
}

#[test]
fn other_error_classes() {
    let d = ParserOptions::default();
    let class = |sql: &str| err(sql, &d).error.error_class();
    assert_eq!(class(" \t\n"), ErrorClass::ParseEmptyStatement);
    assert_eq!(
        class("SELECT 1 /* a /* b */"),
        ErrorClass::UnclosedBracketedComment
    );
    // Spark checks the unclosed comment only after the statement parsed, so a
    // syntax error before it wins.
    assert_eq!(
        class("SELECT FROM WHERE /* never closed"),
        ErrorClass::ParseSyntaxError
    );
    assert_eq!(
        class("SELECT ab\u{e9} FROM t"),
        ErrorClass::InvalidIdentifier
    );
    assert_eq!(
        class("SELECT * FROM test-table"),
        ErrorClass::InvalidIdentifier
    );
    let e = err("SELECT * FROM a.test-table-x", &d);
    assert_eq!(e.error.message_parameters()[0].1, "test-table-x");
    // A fatal error inside an alternative that is retried is still reported.
    assert_eq!(
        class("SELECT * FROM (SELECT ab\u{e9})"),
        ErrorClass::InvalidIdentifier
    );
    assert_eq!(
        class("SELECT ((SELECT ab\u{e9}))"),
        ErrorClass::InvalidIdentifier
    );
    assert_eq!(
        class("SELECT a FROM t GROUP BY GROUPING SETS ((ab\u{e9}))"),
        ErrorClass::InvalidIdentifier
    );
    assert_eq!(
        class("SELECT trim(ab\u{e9})"),
        ErrorClass::InvalidIdentifier
    );
    let e: krishiv_common::error::KrishivError = err("SELECT", &d).into();
    assert_eq!(e.sql_state(), Some("42601"));
    assert_eq!(err("SELECT", &d).to_string(), e.to_string());
}

/// Runs `f` on a thread with a 2 MiB stack, the default of spawned threads
/// and of tokio workers.
fn on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn depth_is_capped_without_stack_overflow() {
    assert_eq!(MAX_DEPTH, krishiv_plan::MAX_DEPTH);
    let shapes = [
        ("SELECT ", "(", "1", ")", ""),
        ("SELECT ", "f(", "1", ")", ""),
        ("SELECT ", "CASE WHEN ", "x", " THEN 1 END", ""),
        ("SELECT ", "a[", "1", "]", ""),
        ("SELECT ", "x -> ", "1", "", ""),
        ("SELECT * FROM ", "(", "t", ")", ""),
        ("SELECT * FROM ", "(SELECT * FROM ", "t", ")", ""),
        ("SELECT CAST(x AS ", "ARRAY<", "INT", ">", ")"),
        ("SELECT a FROM t GROUP BY ", "GROUPING SETS (", "a", ")", ""),
    ];
    for (pre, open, mid, close, post) in shapes {
        let deep = |n: usize| format!("{pre}{}{mid}{}{post}", open.repeat(n), close.repeat(n));
        let (inline, shallow, too_deep) = (deep(16), deep(400), deep(5000));
        // The deepest statement parsed on the caller's stack fits in 1 MiB.
        let i = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(move || parse_query(&inline, &ParserOptions::default()).map(drop))
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(i, Ok(()), "{pre}{open}");
        let (a, b) = on_small_stack(move || {
            let d = ParserOptions::default();
            (
                parse_query(&shallow, &d).map(drop),
                parse_query(&too_deep, &d).map(drop),
            )
        });
        assert_eq!(a, Ok(()), "{pre}{open}");
        assert_eq!(
            b.unwrap_err().error.error_class(),
            ErrorClass::FailedToParseTooComplex
        );
    }
    // Operator chains and prefix runs do not recurse; the trees they build
    // count towards the cap.
    for (sql, within) in [
        (format!("SELECT {}x", "NOT ".repeat(990)), true),
        (format!("SELECT {}x", "- ".repeat(5000)), false),
        (format!("SELECT 1{}", " + 1".repeat(990)), true),
        (format!("SELECT 1{}", " + 1".repeat(5000)), false),
        (format!("SELECT x{}", "[0]".repeat(5000)), false),
        (format!("SELECT x{}", "::INT".repeat(5000)), false),
        (format!("SELECT 1{}", " AND 1".repeat(100_000)), true),
        (format!("SELECT 1{}", " UNION SELECT 1".repeat(5000)), false),
        (format!("SELECT 1 FROM t{}", " JOIN t".repeat(5000)), true),
    ] {
        let r = on_small_stack(move || parse_query(&sql, &ParserOptions::default()).map(drop));
        match r {
            Ok(()) => assert!(within),
            Err(e) => {
                assert!(!within);
                assert_eq!(e.error.error_class(), ErrorClass::FailedToParseTooComplex);
            }
        }
    }
    // A tree at the cap is dropped, cloned and compared on a small stack.
    on_small_stack(|| {
        let q = ok(&format!("SELECT {}1{}", "(".repeat(990), ")".repeat(990)));
        assert_eq!(q.clone(), q);
    });
}

#[test]
fn data_types_resolve_through_the_ddl_parser() {
    let cast = |t: &str| {
        let q = ok(&format!("SELECT x::{t}"));
        let SetExprKind::Select(s) = q.body.kind else {
            panic!("not a select");
        };
        let ExprKind::Cast { data_type, .. } = s.projection[0].expr.kind.clone() else {
            panic!("not a cast");
        };
        *data_type
    };
    for t in [
        "INT",
        "decimal(10, 2)",
        "DECIMAL",
        "timestamp_ntz",
        "TIMESTAMP WITH LOCAL TIME ZONE",
        "string",
        "binary",
    ] {
        assert_eq!(
            cast(t).to_data_type(),
            krishiv_types::parse_data_type(t),
            "{t}"
        );
    }
    assert_eq!(
        cast("ARRAY<STRUCT<a: INT NOT NULL, `b``c` MAP<STRING, ARRAY>>>").to_string(),
        "array<struct<`a`:int not null,`b``c`:map<string,array>>>"
    );
    assert_eq!(
        cast("TIME(6) WITHOUT TIME ZONE").to_string(),
        "time(6) without time zone"
    );
    assert_eq!(cast("STRUCT<>").to_string(), "struct<>");
    assert_eq!(cast("STRUCT").to_string(), "struct");
    assert_eq!(cast("MAP").to_string(), "map");
    let class = |t: &str| cast(t).to_data_type().map_err(|e| e.error_class());
    assert_eq!(
        class("decimal(39, 0)"),
        Err(ErrorClass::DecimalPrecisionExceedsMaxPrecision)
    );
    assert_eq!(
        class("ARRAY<INT>"),
        Err(ErrorClass::KrishivUnsupportedFeature)
    );
}

#[test]
fn lexer_follows_spark() {
    let texts = |sql: &str| {
        tokenize(sql)
            .iter()
            .filter(|t| !t.kind.is_trivia())
            .map(|t| t.span.text(sql).to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        texts("1a 10L 10Lx 1e5 1e5L 1.5a .5 1_0 a.1"),
        [
            "1a", "10L", "10Lx", "1e5", "1e5L", "1", ".", "5a", ".5", "1_0", "a", ".1"
        ]
    );
    assert_eq!(
        texts("1.5e 1.e2D 2.0BDx .5x"),
        ["1", ".", "5e", "1.e2D", "2", ".", "0BDx", ".", "5x"]
    );
    assert_eq!(
        texts("file://b/k-1.x?y=1 a:// s3://"),
        [
            "file://b/k-1.x?y=1",
            "a",
            ":",
            "/",
            "/",
            "s3",
            ":",
            "/",
            "/"
        ]
    );
    assert_eq!(
        texts("x -- c \\\n still comment\ny /* a /* b */ c */ z --"),
        ["x", "y", "z"]
    );
    assert_eq!(texts("a\u{2028}b\u{3000}c"), ["a", "b", "c"]);
    let kinds = tokenize("\u{2029}");
    assert_eq!(kinds[0].kind, TokenKind::Unrecognized);
    assert_eq!(texts("/*+ x */ a*/b"), ["/*+", "x", "*/", "a", "*/", "b"]);
    assert_eq!(
        texts("ARRAY<ARRAY<INT>> a >> b >>> c STRUCT<> d >> e"),
        [
            "ARRAY", "<", "ARRAY", "<", "INT", ">", ">", "a", ">>", "b", ">>>", "c", "STRUCT",
            "<>", "d", ">>", "e"
        ]
    );
    assert_eq!(
        texts("'a''b' \"c\"\"d\" `e``f` r'g\\' R\"h\" 'i\\'j'"),
        [
            "'a''b'",
            "\"c\"\"d\"",
            "`e``f`",
            "r'g\\'",
            "R\"h\"",
            "'i\\'j'"
        ]
    );
    assert_eq!(texts("`x 'y \"z $ @"), ["`x 'y \"z $ @"]);
    assert_eq!(
        texts("a $ @ $$ $x$ b $x$ $y$ c"),
        ["a", "$", "@", "$$ $x$ b $x$ $y$ c"]
    );
    assert_eq!(texts("$x$ b $x$ $1"), ["$x$ b $x$", "$", "1"]);
    let kinds = |sql: &str| tokenize(sql).iter().map(|t| t.kind).collect::<Vec<_>>();
    assert_eq!(
        kinds("a -- c\n/* d */ 'x' r'y"),
        [
            TokenKind::Word,
            TokenKind::Comment,
            TokenKind::Comment,
            TokenKind::String,
            TokenKind::Word,
            TokenKind::Unterminated
        ]
    );
    assert_eq!(
        kinds("a /* b"),
        [TokenKind::Word, TokenKind::UnclosedComment]
    );
    assert_eq!(
        kinds("\u{216B} \u{e9}"),
        [TokenKind::Unrecognized, TokenKind::Word]
    );
    assert_eq!(
        texts(
            "( ) [ ] , . ; * + - / % ~ & | || |> ^ = == <=> <> != < <= > >= !> !< << : :: -> => ? !"
        ),
        [
            "(", ")", "[", "]", ",", ".", ";", "*", "+", "-", "/", "%", "~", "&", "|", "||", "|>",
            "^", "=", "==", "<=>", "<>", "!=", "<", "<=", ">", ">=", "!>", "!<", "<<", ":", "::",
            "->", "=>", "?", "!"
        ]
    );
}

#[test]
fn strings_unescape_like_spark() {
    let u = |raw: &str| unescape_string(raw, true);
    assert_eq!(u(r"'a\nb\tc\rd\0e\bf\Zg'"), "a\nb\tc\rd\0e\u{8}f\u{1a}g");
    assert_eq!(u(r"'\%\_\q\\'"), "\\%\\_q\\");
    assert_eq!(u(r"'\u0041\U0001F600\101\777'"), "A\u{1F600}A777");
    assert_eq!(
        u(r"'\uD83D\uDE00 \uD800 \UFFFFFFFF \u12'"),
        "\u{1F600} \u{FFFD} \u{FFFD} u12"
    );
    assert_eq!(u("'it''s'"), "it's");
    assert_eq!(u("\"say \"\"hi\"\"\""), "say \"hi\"");
    assert_eq!(u(r"'trailing\'"), "trailing\\");
    assert_eq!(u(r"R'\n'"), "\\n");
    assert_eq!(unescape_string(r"'\n'", false), "\\n");
    assert_eq!(lexer::unquote_ident("`a``b`"), "a`b");
}

/// Statements where this parser and Spark 4.2 disagree, with the reason.
const SPARK_DEVIATIONS: &[(&str, &str)] = &[
    (
        "SELECT (SELECT 1",
        "ANTLR adds the hint `: missing ')'` (error quality, P2-T077)",
    ),
    (
        "SELECT a[1",
        "ANTLR adds the hint `: missing ']'` (error quality, P2-T077)",
    ),
    (
        "SELECT a FROM t ORDER BY",
        "ANTLR drops the trailing `BY` as extra input (P2-T077)",
    ),
    (
        "SELECT a, count(*) FROM t GROUP BY a HAVING",
        "ANTLR drops the trailing `HAVING` as extra input (P2-T077)",
    ),
    (
        "SELECT a FROM",
        "with enforced keywords ANTLR reports `FROM`, not the end of input (P2-T077)",
    ),
    (
        "SELECT 1 'secret'",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT * FROM t end",
        "with enforced keywords ANTLR adds no `extra input` hint here (P2-T077)",
    ),
];

/// Error classes Spark raises while parsing a statement (not in
/// `AstBuilder`): the parser must raise the same class and parameters.
const PARSER_CLASSES: &[&str] = &[
    "PARSE_SYNTAX_ERROR",
    "PARSE_EMPTY_STATEMENT",
    "UNCLOSED_BRACKETED_COMMENT",
    "INVALID_IDENTIFIER",
    "SYNTAX_DISCONTINUED.BANG_EQUALS_NOT",
    "FAILED_TO_PARSE_TOO_COMPLEX",
];

/// The probe recorded from a real Spark 4.2 run: statements Spark parses
/// must parse; errors Spark raises while parsing must match in class,
/// message parameters and (where Spark records one) position; errors Spark
/// raises when building the plan must parse here (the post-parse
/// validation, P2-T079, raises them).
#[test]
fn agrees_with_spark_on_the_parse_probe() {
    let path = format!(
        "{}/../../tests/fixtures/spark-4.2.0/sql-parse-probe.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut deviations = vec![];
    let mut owed_to_validation = 0;
    for case in doc["cases"].as_array().unwrap() {
        let sql = case["sql"].as_str().unwrap();
        let conf = case["conf"].as_object().unwrap();
        let options = ParserOptions::from_conf(|k| conf.get(k).and_then(|v| v.as_str()));
        let ours = parse_query(sql, &options);
        let agrees = if case["parses"].as_bool().unwrap() {
            ours.is_ok()
        } else {
            let class = case["error_class"].as_str().unwrap();
            if PARSER_CLASSES.contains(&class) {
                ours.as_ref().is_err_and(|e| {
                    let params: serde_json::Map<String, serde_json::Value> = e
                        .error
                        .message_parameters()
                        .iter()
                        .map(|(k, v)| ((*k).to_owned(), v.clone().into()))
                        .collect();
                    let position = case["start"].as_u64().is_none_or(|start| {
                        sql.get(..e.span.start).map(|p| p.chars().count() as u64) == Some(start)
                    });
                    e.error.error_class().name() == class
                        && Some(&params) == case["message_parameters"].as_object()
                        && position
                })
            } else {
                owed_to_validation += 1;
                ours.is_ok()
            }
        };
        if !agrees {
            deviations.push(sql);
        }
    }
    let mut known: Vec<&str> = SPARK_DEVIATIONS.iter().map(|(sql, _)| *sql).collect();
    known.sort_unstable();
    deviations.sort_unstable();
    assert_eq!(deviations, known);
    assert_eq!(owed_to_validation, 8);
}

#[test]
fn lexing_and_lookahead_stay_linear() {
    let started = std::time::Instant::now();
    // Unterminated quotes and long runs of parentheses used to rescan the
    // rest of the input from every position.
    for sql in [
        format!("SELECT {}", "'\\".repeat(100_000)),
        format!("SELECT {}", "\"\\".repeat(100_000)),
        format!("SELECT {}", "`x".repeat(100_000)),
        format!("SELECT {}", "r'".repeat(100_000)),
        format!("SELECT {}SELECT", "(".repeat(200_000)),
    ] {
        let _ = parse_query(&sql, &ParserOptions::default());
    }
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
    let texts = |sql: &str| {
        tokenize(sql)
            .iter()
            .filter(|t| !t.kind.is_trivia())
            .map(|t| t.span.text(sql).to_owned())
            .collect::<Vec<_>>()
    };
    // An unterminated quote runs to the end of the input.
    assert_eq!(texts("'a'' x"), ["'a'' x"]);
    assert_eq!(texts(r"'\' x \'"), [r"'\' x \'"]);
}

#[test]
fn invalid_identifiers_in_aliases_are_reported() {
    let d = ParserOptions::default();
    for sql in [
        "SELECT a ab\u{e9}",
        "SELECT * FROM t ab\u{e9}",
        "SELECT * FROM t LATERAL VIEW f() v ab\u{e9}",
    ] {
        assert_eq!(
            err(sql, &d).error.error_class(),
            ErrorClass::InvalidIdentifier,
            "{sql}"
        );
    }
}

#[test]
fn less_common_shapes() {
    let d = ParserOptions::default();
    let dq = ParserOptions {
        double_quoted_identifiers: true,
        ..d
    };
    for sql in ["SELECT /*+ NO_PARAMS */ a", "SELECT CAST(x AS STRUCT< >)"] {
        assert!(parse_query(sql, &d).is_ok(), "{sql}");
    }
    let q = parse_query("SELECT 'a' \"b\"", &dq).unwrap();
    assert!(format!("{q:?}").contains("value: \"b\", quoted: true"));
    for sql in [
        "WITH r MAX RECURSION LEVEL x AS (SELECT 1) SELECT 1",
        "SELECT * FROM t NATURAL x",
        "SELECT a.'s'",
        "SELECT CASE x END",
        "SELECT f(a x NULLS)",
        "SELECT CAST(x AS STRUCT<a: INT COMMENT x>)",
        "SELECT CAST(x AS DECIMAL(10, x))",
    ] {
        assert_eq!(
            err(sql, &d).error.error_class(),
            ErrorClass::ParseSyntaxError,
            "{sql}"
        );
    }
}

#[test]
fn review_forms() {
    insta::assert_snapshot!(snapshot(&[
        "SELECT * FROM t VERSION AS OF 1, u FOR SYSTEM_TIME AS OF '2020-01-01' AS x",
        "SELECT * FROM t WITH ('k' = 'v', a.b 1, c = true, d) x",
        "SELECT a IN ((SELECT 1), 2), a IN ((SELECT 1) UNION (SELECT 2)), a IN ((SELECT 1) + 1)",
        "SELECT interval '1-2' year to month 1 day, interval 1 day 2 hours '3:4' hour to minute",
        "SELECT `date` '2020-01-01', geo 'x', dateadd(MICROSECOND, 1, t), extract(dow FROM d)",
        "SELECT sum(a) OVER (CLUSTER BY a), sum(a) OVER (DISTRIBUTE BY a SORT BY b)",
        "SELECT a LIKE ANY ('x') ESCAPE '/', ?, ?, CASE END",
        "FROM t SELECT 1 SELECT 2 ORDER BY 1",
        "WITH order AS (SELECT 1) SELECT * FROM order AS u",
    ]));
}

#[test]
fn bang_for_not_in_predicates_follows_the_legacy_flag() {
    let d = ParserOptions::default();
    for (sql, at) in [
        ("SELECT a ! IN (1)", 9..10),
        ("SELECT a IS ! NULL", 12..13),
        ("SELECT a ! LIKE 'x'", 9..10),
    ] {
        let e = err(sql, &d);
        assert_eq!(
            e.error.error_class(),
            ErrorClass::SyntaxDiscontinuedBangEqualsNot
        );
        assert_eq!(e.span.range(), at, "{sql}");
    }
    // A syntax error later in the statement is reported first, as in Spark.
    let e = err("SELECT a ! IN (1) +", &d);
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let legacy = ParserOptions {
        legacy_bang_equals_not: true,
        ..d
    };
    assert!(parse_query("SELECT a ! BETWEEN 1 AND 2, !b", &legacy).is_ok());
    assert!(parse_query("SELECT !b AND ! c", &d).is_ok());
}

#[test]
fn parse_errors_compare_positions() {
    let d = ParserOptions::default();
    let (a, b) = (err("SELECT 1 x y", &d), err("SELECT 1 x  y", &d));
    assert_eq!(a.error, b.error);
    assert_ne!(a, b);
    assert_eq!(a, err("SELECT 1 x y", &d));
    let t = tokenize("a a");
    assert_ne!(t[0], t[1]);
    assert!(lexer::is_keyword("\u{131}n", "IN") && !lexer::is_keyword("in", "INTO"));
}

#[test]
fn errors_never_echo_string_literals() {
    let d = ParserOptions::default();
    let dq = ParserOptions {
        double_quoted_identifiers: true,
        ..d
    };
    for (sql, options) in [
        ("SELECT 's3cr3t' x y", &d),
        ("SELECT 1 's3cr3t'", &d),
        ("SELECT f('s3cr3t' 'x'", &d),
        ("SELECT 's3cr3t", &d),
        ("SELECT a, 's3cr3t", &d),
        ("SELECT \"s3cr3t\" x y", &d),
        ("SELECT 1 \"x\" 's3cr3t'", &dq),
        ("SELECT $x$s3cr3t$x$", &d),
        ("SELECT $x$s3cr3t", &d),
        ("SELECT * FROM t WITH (k = 's3cr3t' x)", &d),
        ("SELECT X's3cr3t' z z", &d),
        ("SELECT R's3cr3t' x y", &d),
        ("SELECT 1 /* s3cr3t", &d),
    ] {
        let e = err(sql, options);
        let shown = format!("{e} {e:?}");
        assert!(!shown.contains("s3cr3t"), "{sql}: {shown}");
    }
    // The secret is still in the tokens, so a redactor can find it.
    assert!(
        tokenize("SELECT 's3cr3t")
            .iter()
            .any(|t| t.kind == TokenKind::Unterminated)
    );
}

#[test]
fn from_statement_shares_its_from_clause() {
    // Every body reads the one `FROM` clause: the tree grows linearly with
    // tables x bodies, not with their product.
    let statement = |n: usize| {
        let tables: Vec<String> = (0..n).map(|i| format!("t{i}")).collect();
        format!("FROM {} {}", tables.join(", "), "SELECT 1 ".repeat(n))
    };
    let size = |n: usize| format!("{:?}", ok(&statement(n))).len();
    let (small, large) = (size(500), size(2000));
    assert!(large < small * 5, "{small} -> {large}");
    let started = std::time::Instant::now();
    let _ = ok(&statement(10_000));
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
}
