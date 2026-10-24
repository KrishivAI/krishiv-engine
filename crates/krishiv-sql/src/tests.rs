// SPDX-License-Identifier: Apache-2.0
//! Parser unit tests. Tree shapes (spans included) are `insta` snapshots,
//! one per feature group; errors, options, depth and lexing are asserted
//! directly.

use std::fmt::Write as _;

use krishiv_common::error::ErrorClass;

use super::ast::{BinaryOp, ExprKind, SetExprKind, Span};
use super::lexer::{TokenKind, tokenize, unescape_string};
use super::*;

mod probe;

/// The tree of `sql`, without Spark's post-parse checks (tree shapes).
fn ok(sql: &str) -> ast::Query {
    match parser::parse_query_unchecked(sql, ParserOptions::default()) {
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
        cast("ARRAY<STRUCT<a: INT NOT NULL, `b``c` MAP<STRING, ARRAY<INT>>>>").to_string(),
        "array<struct<`a`:int not null,`b``c`:map<string,array<int>>>>"
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

/// Modes, nesting and spans of `EXPLAIN`.
#[test]
fn explain_statements() {
    use ast::{ExplainMode, Statement};
    let d = ParserOptions::default();
    let explain = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::Explain(e) => e,
        Statement::Query(_) => panic!("{sql} is a query"),
        other => panic!("{sql} is {other:?}"),
    };
    for (sql, mode) in [
        ("EXPLAIN SELECT 1", ExplainMode::Simple),
        ("explain Extended SELECT 1", ExplainMode::Extended),
        ("EXPLAIN FORMATTED SELECT 1", ExplainMode::Formatted),
        ("EXPLAIN CODEGEN SELECT 1", ExplainMode::Codegen),
        ("EXPLAIN COST SELECT 1;", ExplainMode::Cost),
    ] {
        let e = explain(sql);
        assert_eq!(e.mode, mode, "{sql}");
        assert_eq!(e.span.text(sql), sql.trim_end_matches(';'), "{sql}");
        let Statement::Query(q) = *e.statement else {
            panic!("{sql}")
        };
        assert_eq!(*q, ok("SELECT 1"), "{sql}");
    }
    // An EXPLAIN of an EXPLAIN: each spans from its keyword to the end.
    let sql = "EXPLAIN EXTENDED EXPLAIN SELECT a FROM t";
    let outer = explain(sql);
    let Statement::Explain(inner) = *outer.statement else {
        panic!("not nested")
    };
    assert_eq!(inner.mode, ExplainMode::Simple);
    assert_eq!(inner.span.text(sql), "EXPLAIN SELECT a FROM t");
    // A query is a statement too; EXPLAIN is still a name elsewhere.
    assert!(matches!(
        parse_statement("SELECT explain FROM explain", &d),
        Ok(Statement::Query(_))
    ));
    // `parse_query` takes queries only.
    assert_eq!(
        err("EXPLAIN SELECT 1", &d).error.error_class(),
        ErrorClass::ParseSyntaxError
    );
}

/// `EXPLAIN` errors, positions and precedence as Spark 4.2 raises them.
#[test]
fn explain_errors_follow_spark() {
    let d = ParserOptions::default();
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    // LOGICAL parses, then the outermost one is rejected over its statement.
    let sql = "EXPLAIN EXPLAIN LOGICAL SELECT a ! IN (1);";
    let e = fail(sql);
    assert_eq!(e.error.error_class(), ErrorClass::InvalidStatementOrClause);
    assert_eq!(
        e.error.message_parameters(),
        [("operation", "EXPLAIN LOGICAL".to_owned())]
    );
    assert_eq!(e.span.range(), 8..41);
    // A syntax error or an unclosed comment is raised before it.
    let e = fail("EXPLAIN LOGICAL SELECT (");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail("EXPLAIN LOGICAL SELECT 1 /* x");
    assert_eq!(e.error.error_class(), ErrorClass::UnclosedBracketedComment);
    // `EXPLAIN ANALYZE TABLE` reads as `ANALYZE TABLE`: the error is
    // after `ANALYZE`, as in Spark.
    let e = fail("EXPLAIN ANALYZE TABLE t COMPUTE STATISTICS");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 16..21);
    assert_eq!(fail("EXPLAIN ANALYZE").span.range(), 15..15);
    assert_eq!(fail("EXPLAIN VERBOSE").span.range(), 15..15);
    // Krishiv's extensions, syntax errors in Spark (ADR 0023).
    use ast::{ExplainMode, Statement};
    for (sql, mode) in [
        ("EXPLAIN ANALYZE SELECT 1", ExplainMode::Analyze),
        (
            "explain analyze WITH x AS (SELECT 1) SELECT * FROM x",
            ExplainMode::Analyze,
        ),
        ("EXPLAIN VERBOSE SELECT 1", ExplainMode::Verbose),
        ("EXPLAIN VERBOSE FROM t SELECT a", ExplainMode::Verbose),
    ] {
        let Ok(Statement::Explain(e)) = parse_statement(sql, &ParserOptions::default()) else {
            panic!("{sql} does not parse as EXPLAIN");
        };
        assert_eq!(e.mode, mode, "{sql}");
        assert!(matches!(*e.statement, Statement::Query(_)), "{sql}");
    }
    let e = fail("EXPLAIN EXTENDED EXTENDED SELECT 1");
    assert_eq!(e.span.range(), 17..25);
}

/// A long `EXPLAIN` chain is capped without deep recursion.
#[test]
fn explain_chains_are_capped_without_recursion() {
    let d = ParserOptions::default();
    let deepest = format!("{}SELECT 1", "EXPLAIN ".repeat(MAX_DEPTH));
    let mut s = parse_statement(&deepest, &d).unwrap();
    let mut levels = 0;
    while let ast::Statement::Explain(e) = s {
        levels += 1;
        s = *e.statement;
    }
    assert_eq!(levels, MAX_DEPTH);
    let deeper = format!("{}SELECT 1", "EXPLAIN ".repeat(100_000));
    let len = deeper.len();
    let e =
        on_small_stack(move || parse_statement(&deeper, &ParserOptions::default())).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::FailedToParseTooComplex);
    assert_eq!(e.span.range(), 0..len);
}

/// The cap is exact: for each shape, the deepest statement within
/// [`MAX_DEPTH`] tree levels parses, and one level more fails with
/// `FAILED_TO_PARSE_TOO_COMPLEX` reported, as Spark does, at the whole
/// statement. `limit` counts the repeated unit; the query and its select
/// item (or relation) take the remaining levels.
#[test]
fn depth_limit_is_exact_and_reported_at_the_statement() {
    let shapes: [(&str, &str, &str, &str, &str, usize); 15] = [
        ("SELECT ", "(", "1", ")", "", MAX_DEPTH - 2),
        ("SELECT ", "f(", "1", ")", "", MAX_DEPTH - 2),
        (
            "SELECT ",
            "CASE WHEN ",
            "x",
            " THEN 1 END",
            "",
            MAX_DEPTH - 2,
        ),
        // An element access is the access and its index.
        ("SELECT ", "a[", "1", "]", "", (MAX_DEPTH - 2) / 2),
        ("SELECT ", "x -> ", "1", "", "", MAX_DEPTH - 2),
        ("SELECT * FROM ", "(", "t", ")", "", MAX_DEPTH - 2),
        // A subquery relation is a relation and a query.
        (
            "SELECT * FROM ",
            "(SELECT * FROM ",
            "t",
            ")",
            "",
            (MAX_DEPTH - 2) / 2,
        ),
        (
            "SELECT CAST(x AS ",
            "ARRAY<",
            "INT",
            ">",
            ")",
            MAX_DEPTH - 3,
        ),
        (
            "SELECT a FROM t GROUP BY ",
            "GROUPING SETS (",
            "a",
            ")",
            "",
            MAX_DEPTH - 1,
        ),
        ("SELECT ", "NOT ", "x", "", "", MAX_DEPTH - 2),
        ("SELECT ", "- ", "x", "", "", MAX_DEPTH - 2),
        ("SELECT 1", " + 1", "", "", "", MAX_DEPTH - 2),
        ("SELECT 1", " UNION SELECT 1", "", "", "", MAX_DEPTH - 2),
        ("SELECT x", "[0]", "", "", "", MAX_DEPTH - 3),
        ("SELECT x", "::INT", "", "", "", MAX_DEPTH - 3),
    ];
    for (pre, open, mid, close, post, limit) in shapes {
        let deep = move |n: usize| format!("{pre}{}{mid}{}{post}", open.repeat(n), close.repeat(n));
        let (at, over) = (deep(limit), deep(limit + 1));
        let len = over.len();
        let (a, b) = on_small_stack(move || {
            let d = ParserOptions::default();
            (parse_query(&at, &d).map(drop), parse_query(&over, &d))
        });
        assert_eq!(a, Ok(()), "{pre}{open} x{limit}");
        let e = b.unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::FailedToParseTooComplex,
            "{pre}{open} x{}",
            limit + 1
        );
        assert_eq!(e.span.range(), 0..len, "{pre}{open}");
    }
    // Leading and trailing trivia are outside the statement.
    let sql = format!(
        "  /* c */ SELECT {}1{} ; -- x",
        "(".repeat(MAX_DEPTH),
        ")".repeat(MAX_DEPTH)
    );
    let end = sql.find(" -- x").unwrap();
    let e = on_small_stack(move || parse_query(&sql, &ParserOptions::default())).unwrap_err();
    assert_eq!(e.span.range(), 10..end);
}

#[test]
fn multipart_identifiers_parse_like_spark() {
    // Spark 4.2.0, `spark.table(name)` (`parseMultipartIdentifier`).
    let d = ParserOptions::default();
    let parts = |name: &str| -> Vec<String> {
        parse_multipart_identifier(name, &d)
            .unwrap()
            .parts
            .into_iter()
            .map(|p| p.value)
            .collect()
    };
    assert_eq!(parts("`x.y`"), ["x.y"]);
    assert_eq!(parts("x.y"), ["x", "y"]);
    assert_eq!(parts("`a``b`"), ["a`b"]);
    assert_eq!(parts(" default.`my.table` "), ["default", "my.table"]);
    for (name, message, at) in [
        ("`unclosed", "Syntax error at or near '`'", 0),
        ("a..b", "Syntax error at or near '.'", 1),
        ("a b", "Syntax error at or near 'b': extra input 'b'", 2),
        ("`x.y`.", "Syntax error at or near '.': extra input '.'", 5),
    ] {
        let e = parse_multipart_identifier(name, &d).unwrap_err();
        assert_eq!(
            e.error.to_string(),
            format!("[PARSE_SYNTAX_ERROR] {message}. SQLSTATE: 42601"),
            "{name}"
        );
        assert_eq!(e.span.start, at, "{name}");
    }
    let e = parse_multipart_identifier("", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseEmptyStatement);
}

/// Typed datetime literals lower to an unresolved node (the type keyword
/// and the raw string), with no time zone involved: the analyzer resolves
/// them with the session zone.
#[test]
fn typed_datetime_literals_lower_unresolved() {
    use crate::ast::{ExprKind, SetExprKind};
    use krishiv_plan::SourceId;

    // Each kind keeps its keyword and raw string, with the literal's span,
    // and lowers without a time zone.
    for (sql, keyword) in [
        ("SELECT DATE '2016-03-12'", "DATE"),
        ("SELECT TIMESTAMP '2024-01-01 10:00:00'", "TIMESTAMP"),
        (
            "SELECT TIMESTAMP_NTZ '2024-01-01 10:00:00'",
            "TIMESTAMP_NTZ",
        ),
        (
            "SELECT TIMESTAMP_LTZ '2024-01-01 10:00:00'",
            "TIMESTAMP_LTZ",
        ),
    ] {
        let q = crate::parse_query(sql, &ParserOptions::default()).unwrap();
        let SetExprKind::Select(s) = q.body.kind else {
            panic!("{sql} is not a select");
        };
        let ExprKind::Literal(crate::ast::Literal::Typed { type_name, value }) =
            s.projection[0].expr.kind.clone()
        else {
            panic!("{sql} is not a typed literal");
        };
        assert_eq!(type_name.value.to_ascii_uppercase(), keyword, "{sql}");
        assert!(!value.is_empty(), "{sql}");
        assert_eq!(s.projection[0].expr.span.range(), 7..sql.len(), "{sql}");
        // Lowering succeeds with the default options (no zone) and keeps
        // the raw text: the plan holds a `TypedLiteral`, never micros.
        let plan = crate::parse_and_lower(sql, &ParserOptions::default(), SourceId(0)).unwrap();
        let text = plan.root().to_string();
        assert!(text.contains(&format!("{keyword} '")), "{sql}: {text}");
    }
    // The wall-clock text is preserved exactly, including an explicit zone.
    let q = parser::parse_query_unchecked(
        "SELECT TIMESTAMP '2024-01-01 10:00:00+05:30'",
        ParserOptions::default(),
    )
    .unwrap();
    let SetExprKind::Select(s) = q.body.kind else {
        panic!("not a select");
    };
    let ExprKind::Literal(crate::ast::Literal::Typed { value, .. }) =
        s.projection[0].expr.kind.clone()
    else {
        panic!("not a typed literal");
    };
    assert_eq!(value, "2024-01-01 10:00:00+05:30");
    let plan = crate::parse_and_lower(
        "SELECT TIMESTAMP '2024-01-01 10:00:00'",
        &ParserOptions::default(),
        SourceId(0),
    )
    .unwrap();
    let text = plan.root().to_string();
    assert!(text.contains("TIMESTAMP '2024-01-01 10:00:00'"), "{text}");
    assert!(!text.contains("1704103200000000"), "{text}");
}

/// `parse_named_expression`: one named expression (`F.expr`,
/// `selectExpr` once per argument string, `filter`/`where` strings).
#[test]
fn expression_strings() {
    use super::ast::Alias;
    let d = ParserOptions::default();
    // One expression with an optional alias keeps its spans.
    let item = parse_named_expression("a + 1", &d).unwrap();
    assert_eq!(item.expr.span.text("a + 1"), "a + 1");
    assert!(item.alias.is_none());
    assert!(parse_named_expression("count(*) OVER (PARTITION BY a ORDER BY b)", &d).is_ok());
    assert!(parse_named_expression("*", &d).is_ok());
    assert!(parse_named_expression("CASE WHEN a THEN 1 ELSE 0 END", &d).is_ok());
    // Explicit and implicit aliases parse, as in Spark's `parseExpression`.
    for sql in ["a AS b", "a b", "count(*) AS n", "a > 0 AS c", "v + 1 as w"] {
        let item = parse_named_expression(sql, &d).unwrap_or_else(|e| panic!("{sql}: {}", e.error));
        assert!(item.alias.is_some(), "{sql}");
        if sql == "a b" {
            let Alias::Name(name) = item.alias.as_ref().unwrap() else {
                panic!("not a name: {:?}", item.alias);
            };
            assert_eq!(name.value, "b");
        }
    }
    assert!(parse_named_expression("t.*", &d).unwrap().alias.is_none());
    // A statement is no expression.
    let e = parse_named_expression("SELECT 1", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 0..6);
    // Each argument string is one expression: a comma is trailing input.
    let e = parse_named_expression("a, b", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(
        e.error.message_parameters(),
        [("error", "','".to_owned()), ("hint", String::new())]
    );
    assert_eq!(e.span.range(), 1..2);
    let e = parse_named_expression("a;", &d).unwrap_err();
    assert_eq!(
        e.error.message_parameters(),
        [
            ("error", "';'".to_owned()),
            ("hint", ": extra input ';'".to_owned())
        ]
    );
    assert_eq!(e.span.range(), 1..2);
    // An unterminated quote with text after it is several tokens to Spark,
    // so no "extra input" hint (as for statements: `SELECT a 'bc` has none,
    // `SELECT a '` keeps it for the lone quote).
    let e = parse_named_expression("a 'bc", &d).unwrap_err();
    assert_eq!(
        e.error.message_parameters(),
        [("error", "'''".to_owned()), ("hint", String::new())]
    );
    assert_eq!(e.span.range(), 2..3);
    let e = parse_named_expression("a '", &d).unwrap_err();
    assert_eq!(
        e.error.message_parameters(),
        [
            ("error", "'''".to_owned()),
            ("hint", ": extra input '''".to_owned())
        ]
    );
    assert_eq!(e.span.range(), 2..3);
    // An unfinished expression ends at the end of the input.
    let e = parse_named_expression("a +", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 3..3);
    // Empty strings are empty statements.
    for sql in ["", "  "] {
        assert_eq!(
            parse_named_expression(sql, &d)
                .unwrap_err()
                .error
                .error_class(),
            ErrorClass::ParseEmptyStatement,
            "{sql:?}"
        );
    }
    // Post-parse checks run on the expression: an empty `IN ()` is an error.
    let e = parse_named_expression("a IN ()", &d).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "INVALID_SQL_SYNTAX.EMPTY_IN_PREDICATE"
    );
}

/// Expression strings lower as `SELECT` items do.
#[test]
fn expression_strings_lower_like_select_items() {
    use krishiv_plan::SourceId;
    let d = ParserOptions::default();
    let lowered = parse_and_lower_named_expression("a + 1", &d, SourceId(0)).unwrap();
    let krishiv_plan::ExprNode::Function(call) = &lowered.node else {
        panic!("not a call: {lowered:?}");
    };
    assert_eq!(call.name, "+");
    assert_eq!(call.args.len(), 2);
    let lowered = parse_and_lower_named_expression("b AS c", &d, SourceId(0)).unwrap();
    let krishiv_plan::ExprNode::Alias { name, .. } = &lowered.node else {
        panic!("not an alias: {lowered:?}");
    };
    assert_eq!(name, "c");
    // Constructs the IR lacks fail while lowering.
    let e = parse_and_lower_named_expression("a[0]", &d, SourceId(0)).unwrap_err();
    assert_eq!(e.error.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
}

/// `lower_expression` lowers one parsed expression as `SELECT` items lower
/// it, and `lower_select_item` keeps its alias.
#[test]
fn lower_expression_and_select_item() {
    use super::lower::{lower_expression, lower_select_item};
    use krishiv_plan::SourceId;
    let d = ParserOptions::default();
    let item = parse_named_expression("a + 1", &d).unwrap();
    let lowered = lower_expression(&item.expr, SourceId(0)).unwrap();
    let krishiv_plan::ExprNode::Function(call) = &lowered.node else {
        panic!("not a call: {lowered:?}");
    };
    assert_eq!(call.name, "+");
    let item = parse_named_expression("b AS c", &d).unwrap();
    let lowered = lower_select_item(&item, SourceId(0)).unwrap();
    let krishiv_plan::ExprNode::Alias { name, .. } = &lowered.node else {
        panic!("not an alias: {lowered:?}");
    };
    assert_eq!(name, "c");
}

/// Deep expression strings fail with `FAILED_TO_PARSE_TOO_COMPLEX` at the
/// whole string instead of overflowing the stack. The cap is exact: one
/// nested level per parenthesis, so `MAX_DEPTH - 1` pairs parse and one
/// more fails with the full-input span.
#[test]
fn expression_depth_is_capped() {
    let d = ParserOptions::default();
    let (shallow, deep) = on_small_stack(move || {
        let shallow = format!(
            "{}1{}",
            "(".repeat(MAX_DEPTH - 1),
            ")".repeat(MAX_DEPTH - 1)
        );
        let deep = format!("{}1{}", "(".repeat(MAX_DEPTH), ")".repeat(MAX_DEPTH));
        (
            parse_named_expression(&shallow, &d).map(drop),
            parse_named_expression(&deep, &d),
        )
    });
    assert_eq!(shallow, Ok(()));
    let e = deep.unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::FailedToParseTooComplex);
    assert_eq!(e.span.range(), 0..2 * MAX_DEPTH + 1);
}

/// `SET` statements: every form, with spans and decoded keys/values.
#[test]
fn set_statements() {
    use ast::{SetKind, Statement};
    let d = ParserOptions::default();
    let set = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::Set(s) => s,
        other => panic!("{sql} is {other:?}"),
    };
    // `SET` lists every explicitly set key; a trailing `;` is not part
    // of the span.
    for sql in ["SET", "SET;", "SET ; ;", "set"] {
        let s = set(sql);
        assert!(matches!(s.kind, SetKind::All), "{sql}");
        assert_eq!(
            s.span.text(sql),
            sql.trim_end_matches([';', ' ', '\t']),
            "{sql}"
        );
    }
    // `SET -v`.
    for sql in ["SET -v", "SET -v;", "set -V"] {
        let s = set(sql);
        assert!(matches!(s.kind, SetKind::Verbose), "{sql}");
    }
    // `SET key`: the key is the trimmed rest of the statement.
    let s = set("SET spark.sql.ansi.enabled");
    let SetKind::Show { key, key_span } = &s.kind else {
        panic!("{:?}", s.kind)
    };
    assert_eq!(key, "spark.sql.ansi.enabled");
    assert_eq!(
        key_span.text("SET spark.sql.ansi.enabled"),
        "spark.sql.ansi.enabled"
    );
    assert_eq!(
        s.span.text("SET spark.sql.ansi.enabled"),
        "SET spark.sql.ansi.enabled"
    );
    // A trailing `;` ends the key without joining it.
    let s = set("SET my.key;");
    let SetKind::Show { key, .. } = &s.kind else {
        panic!("{:?}", s.kind)
    };
    assert_eq!(key, "my.key");
    assert_eq!(s.span.text("SET my.key;"), "SET my.key");
    // `SET key = value`, with and without spaces: the value is the raw
    // rest, trimmed.
    let s = set("SET a=1");
    let SetKind::Assign {
        key,
        key_span,
        value,
        value_span,
    } = &s.kind
    else {
        panic!("{:?}", s.kind)
    };
    assert_eq!((key.as_str(), value.as_str()), ("a", "1"));
    assert_eq!(key_span.text("SET a=1"), "a");
    assert_eq!(value_span.text("SET a=1"), "1");
    let sql = "SET my.key = a=b=c ;";
    let s = set(sql);
    let SetKind::Assign {
        key,
        value,
        value_span,
        ..
    } = &s.kind
    else {
        panic!("{:?}", s.kind)
    };
    assert_eq!((key.as_str(), value.as_str()), ("my.key", "a=b=c"));
    assert_eq!(value_span.text(sql), "a=b=c");
    assert_eq!(s.span.text(sql), "SET my.key = a=b=c");
    // Quoted keys and values: only backquotes are special, as in Spark.
    for (sql, key, value) in [
        ("SET `my.key` = 1", "my.key", "1"),
        ("SET `a``b` = `x``y`", "a`b", "x`y"),
        ("SET k = 'a;b'", "k", "'a;b'"),
        ("SET `k` = 'v'", "k", "'v'"),
    ] {
        let s = set(sql);
        match &s.kind {
            SetKind::Assign {
                key: k, value: v, ..
            } => assert_eq!((k.as_str(), v.as_str()), (key, value), "{sql}"),
            SetKind::Show { key: k, .. } => assert_eq!((k.as_str(), ""), (key, value), "{sql}"),
            kind => panic!("{sql} is {kind:?}"),
        }
    }
    // A backquoted key with no `=` is a lookup.
    let s = set("SET `a=b`");
    let SetKind::Show { key, .. } = &s.kind else {
        panic!("{:?}", s.kind)
    };
    assert_eq!(key, "a=b");
    // Double-quoted keys are syntax errors; single-quoted keys fail the
    // key check (asserted in `set_errors_follow_spark`).
    for sql in ["SET \"k\" = \"v\"", "SET \"k\""] {
        assert_eq!(
            parse_statement(sql, &d).unwrap_err().error.error_class(),
            ErrorClass::ParseSyntaxError,
            "{sql}"
        );
    }
    // `SET TIME ZONE ...`.
    let s = set("SET TIME ZONE 'UTC'");
    let SetKind::TimeZone { zone, .. } = &s.kind else {
        panic!("{:?}", s.kind)
    };
    assert_eq!(zone, &ast::TimeZone::Named("UTC".to_owned()));
    let s = set("set time zone local");
    let SetKind::TimeZone { zone, .. } = &s.kind else {
        panic!("{:?}", s.kind)
    };
    assert_eq!(zone, &ast::TimeZone::Local);
    // `parse_query` takes queries only.
    assert_eq!(
        err("SET a = 1", &d).error.error_class(),
        ErrorClass::ParseSyntaxError
    );
    // A query is a statement too; `SET` is still a name elsewhere.
    assert!(matches!(
        parse_statement("SELECT set FROM set", &d),
        Ok(Statement::Query(_))
    ));
}

/// `SET` errors, positions and precedence as Spark 4.2 raises them.
#[test]
fn set_errors_follow_spark() {
    let d = ParserOptions::default();
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    // Anything but `SET`, `SET -v`, a plain key or `key = value` is
    // `INVALID_SET_SYNTAX`.
    for sql in [
        "SET a b",
        "SET -x",
        "SET = 1",
        "SET -v x",
        "SET -v x y",
        "SET 'k'=1",
        "SET 'k'",
    ] {
        let e = fail(sql);
        assert_eq!(e.error.error_class(), ErrorClass::InvalidSetSyntax, "{sql}");
        assert_eq!(e.error.sql_state(), Some("42000"), "{sql}");
    }
    assert!(
        fail("SET a b")
            .error
            .to_string()
            .contains("Expected format is 'SET'"),
    );
    // An empty key fails at the `=`.
    let e = fail("SET = 1");
    assert_eq!(e.span.range(), 0..3);
    // An unterminated quote fails at the quote.
    let e = fail("SET x = 'abc");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 8..9);
    let e = fail("SET 'abc = 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 4..5);
    // A quote later in a bare key fails at the quote too.
    let e = fail("SET k'x = 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 5..6);
    // An `INTERVAL` time zone is a supported grammar form the engine does
    // not implement.
    let e = fail("SET TIME ZONE INTERVAL '1' DAY");
    assert_eq!(e.error.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.error.to_string().contains("INTERVAL"), "{e:?}");
    // A missing zone is a syntax error.
    let e = fail("SET TIME ZONE");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // Anything after the value's `;` is extra input, as for queries.
    let e = fail("SET x = 1; SELECT 2");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(
        e.error.to_string(),
        "[PARSE_SYNTAX_ERROR] Syntax error at or near 'SELECT': extra input 'SELECT'. \
         SQLSTATE: 42601"
    );
    assert_eq!(e.span.range(), 11..17);
    // Under `EXPLAIN`, a command is the explained statement.
    let ast::Statement::Explain(e) = parse_statement("EXPLAIN SET x = 1", &d).unwrap() else {
        panic!("not an EXPLAIN")
    };
    let ast::Statement::Set(_) = *e.statement else {
        panic!("not a SET")
    };
}

/// `RESET` statements and their errors.
#[test]
fn reset_statements() {
    use ast::Statement;
    let d = ParserOptions::default();
    let reset = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::Reset(r) => r,
        other => panic!("{sql} is {other:?}"),
    };
    for sql in ["RESET", "RESET;", "reset"] {
        let r = reset(sql);
        assert_eq!(r.key, None, "{sql}");
        assert_eq!(r.key_span, None, "{sql}");
    }
    let sql = "RESET spark.sql.ansi.enabled;";
    let r = reset(sql);
    assert_eq!(r.key.as_deref(), Some("spark.sql.ansi.enabled"));
    assert_eq!(r.key_span.unwrap().text(sql), "spark.sql.ansi.enabled");
    assert_eq!(r.span.text(sql), "RESET spark.sql.ansi.enabled");
    let r = reset("RESET `my key`");
    assert_eq!(r.key.as_deref(), Some("my key"));
    // Anything but one plain key is `INVALID_RESET_COMMAND_FORMAT`.
    for sql in ["RESET a b", "RESET a=b", "RESET 'k'", "RESET \"k\""] {
        let e = parse_statement(sql, &d).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidResetCommandFormat,
            "{sql}"
        );
        assert_eq!(e.error.sql_state(), Some("42000"), "{sql}");
    }
    assert!(
        parse_statement("RESET a b", &d)
            .unwrap_err()
            .error
            .to_string()
            .contains("Expected format is 'RESET'"),
    );
    // An unterminated quote fails at the quote.
    let e = parse_statement("RESET 'abc", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 6..7);
    let e = parse_statement("RESET ab'cd", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 8..9);
}

/// `CREATE TEMPORARY VIEW` statements and their errors.
#[test]
fn create_view_statements() {
    use ast::Statement;
    let d = ParserOptions::default();
    let create = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateView(v) => v,
        other => panic!("{sql} is {other:?}"),
    };
    let v = create("CREATE TEMPORARY VIEW v AS SELECT 1");
    assert!(!v.or_replace && !v.if_not_exists);
    assert_eq!(v.name.value, "v");
    assert!(v.columns.is_empty());
    assert_eq!(*v.query, ok("SELECT 1"));
    assert_eq!(
        v.span.text("CREATE TEMPORARY VIEW v AS SELECT 1"),
        "CREATE TEMPORARY VIEW v AS SELECT 1"
    );
    // `TEMP`, `OR REPLACE` and the column list.
    let sql = "CREATE OR REPLACE TEMP VIEW `v` (a COMMENT 'the a', b) AS SELECT 1, 2;";
    let v = create(sql);
    assert!(v.or_replace && !v.if_not_exists);
    assert_eq!(v.name.value, "v");
    assert_eq!(
        v.columns
            .iter()
            .map(|c| c.name.value.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(v.columns[0].comment.as_deref(), Some("the a"));
    assert_eq!(v.columns[1].comment, None);
    assert_eq!(v.columns[0].span.text(sql), "a COMMENT 'the a'");
    assert_eq!(*v.query, ok("SELECT 1, 2"));
    assert_eq!(
        v.span.text(sql),
        "CREATE OR REPLACE TEMP VIEW `v` (a COMMENT 'the a', b) AS SELECT 1, 2"
    );
    // `OR REPLACE` with `IF NOT EXISTS` parses and fails with
    // `CREATE_VIEW_WITH_IF_NOT_EXISTS_AND_REPLACE`, as in Spark, before
    // the view's query is checked.
    let e = parse_statement(
        "CREATE OR REPLACE TEMP VIEW IF NOT EXISTS `v` (a COMMENT 'the a', b) AS SELECT DATE '2020-02-30'",
        &d,
    )
    .unwrap_err();
    assert_eq!(
        e.error.error_class(),
        ErrorClass::CreateViewWithIfNotExistsAndReplace
    );
    assert_eq!(e.error.sql_state(), Some("42601"));
    assert_eq!(
        e.error.message_parameters(),
        [("viewName", "`v`".to_owned())]
    );
    // The view's query gets Spark's post-parse checks.
    let e = parse_statement("CREATE TEMP VIEW v AS SELECT DATE '2020-02-30'", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::InvalidTypedLiteral);
    // `IF NOT EXISTS` without `OR REPLACE` parses and fails with
    // `_LEGACY_ERROR_TEMP_0053`, as in Spark.
    let e = parse_statement("CREATE TEMP VIEW IF NOT EXISTS v AS SELECT 1", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::LegacyErrorTemp0053);
    assert_eq!(
        e.error.message(),
        "It is not allowed to define a TEMPORARY view with IF NOT EXISTS."
    );
    // A qualified name parses and fails with
    // `INVALID_TEMP_OBJ_QUALIFIER`, as in Spark.
    let e = parse_statement("CREATE TEMP VIEW a.b AS SELECT 1", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::InvalidTempObjQualifier);
    assert_eq!(e.error.sql_state(), Some("42602"));
    assert_eq!(
        e.error.message_parameters(),
        [
            ("objectType", "view".to_owned()),
            ("objectName", "`a`.`b`".to_owned()),
            ("qualifier", "`a`".to_owned()),
        ]
    );
    // Only temporary views: anything else fails where it stands.
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    let e = fail("CREATE VIEW v AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 7..11);
    let e = fail("CREATE GLOBAL TEMPORARY VIEW v AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 7..13);
    // `AS` is required, and the query follows it.
    let e = fail("CREATE TEMP VIEW v SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 19..25);
    let e = fail("CREATE TEMP VIEW v AS");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 21..21);
    // The column list is names (with comments), not types.
    let e = fail("CREATE TEMP VIEW v (a INT) AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 22..25);
    let e = fail("CREATE TEMP VIEW v () AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail("CREATE TEMP VIEW v (a COMMENT 1) AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 30..31);
}

/// `DROP VIEW` statements and their errors.
#[test]
fn drop_view_statements() {
    use ast::Statement;
    let d = ParserOptions::default();
    let drop = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::DropView(v) => v,
        other => panic!("{sql} is {other:?}"),
    };
    let v = drop("DROP VIEW v;");
    assert!(!v.if_exists);
    assert_eq!(v.name.value, "v");
    assert_eq!(v.span.text("DROP VIEW v;"), "DROP VIEW v");
    let v = drop("drop view if exists `my view`");
    assert!(v.if_exists);
    assert_eq!(v.name.value, "my view");
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    // Only views drop here: anything else fails where it stands.
    let e = fail("DROP VIEWPOINT v");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 5..14);
    // The name is required.
    let e = fail("DROP VIEW");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 9..9);
    let e = fail("DROP VIEW IF EXISTS");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 19..19);
}

/// `CREATE TABLE` statements: flags, columns and the table clauses.
#[test]
fn create_table_statements() {
    use ast::{DdlName, Statement, TableElement};
    let d = ParserOptions::default();
    let create = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateTable(t) => t,
        other => panic!("{sql} is {other:?}"),
    };
    // The minimal form.
    let t = create("CREATE TABLE t (a INT)");
    assert!(!t.or_replace && !t.temporary && !t.external && !t.if_not_exists);
    let DdlName::Parts(name) = &t.name else {
        panic!("{t:?}");
    };
    assert_eq!(
        name.parts
            .iter()
            .map(|p| p.value.as_str())
            .collect::<Vec<_>>(),
        ["t"]
    );
    assert_eq!(t.elements.len(), 1);
    let TableElement::Column(c) = &t.elements[0] else {
        panic!("{:?}", t.elements[0]);
    };
    assert!(matches!(&c.name, ast::ColumnName::Ident(id) if id.value == "a"));
    assert_eq!(c.data_type.to_string(), "int");
    assert!(!c.not_null && c.default.is_none() && c.generated.is_none());
    assert_eq!(c.comment, None);
    assert!(c.constraints.is_empty());
    assert!(t.using.is_none() && t.as_query.is_none());
    assert_eq!(
        t.span.text("CREATE TABLE t (a INT)"),
        "CREATE TABLE t (a INT)"
    );
    // Flags, a qualified name and a trailing `;`.
    let t = create("CREATE TEMPORARY EXTERNAL TABLE IF NOT EXISTS a.b (c INT);");
    assert!(!t.or_replace && t.temporary && t.external && t.if_not_exists);
    let DdlName::Parts(name) = &t.name else {
        panic!("{t:?}");
    };
    assert_eq!(
        name.parts
            .iter()
            .map(|p| p.value.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    // `IF NOT EXISTS` is not an `OR REPLACE` form: the name is `IF`.
    let t = create("CREATE OR REPLACE TABLE IF (a INT)");
    let DdlName::Parts(name) = &t.name else {
        panic!("{t:?}");
    };
    assert_eq!(
        name.parts
            .iter()
            .map(|p| p.value.as_str())
            .collect::<Vec<_>>(),
        ["IF"]
    );
    // Every column option, every clause.
    let sql = "CREATE TABLE t (a INT NOT NULL DEFAULT 42 COMMENT 'the a' GENERATED ALWAYS AS (a + 1), \
        b STRING, c STRUCT<x: INT>) USING parquet OPTIONS ('a' 1, password = 'pw') \
        PARTITIONED BY (c, d, years(ts), e INT NOT NULL COMMENT 'p') CLUSTER BY (a, b) \
        LOCATION '/tmp/t' COMMENT 'table' DEFAULT COLLATION UTF8_LCASE \
        TBLPROPERTIES ('k' = 'v', flag) AS SELECT 1";
    // Spark builds this tree and then rejects a CTAS with a schema.
    let e = parse_statement(sql, &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::LegacyErrorTemp0035);
    let Statement::CreateTable(t) = crate::parser::parse_statement_unchecked(sql, d).unwrap()
    else {
        panic!("{sql}");
    };
    assert_eq!(t.elements.len(), 3);
    let TableElement::Column(a) = &t.elements[0] else {
        panic!("{:?}", t.elements[0]);
    };
    assert!(a.not_null);
    assert_eq!(a.default.as_ref().map(|e| e.span.text(sql)), Some("42"));
    assert_eq!(a.comment.as_deref(), Some("the a"));
    assert!(matches!(a.generated, Some(ref g) if matches!(g.kind, ast::GeneratedKind::Expr(_))));
    assert_eq!(
        a.span.text(sql),
        "a INT NOT NULL DEFAULT 42 COMMENT 'the a' GENERATED ALWAYS AS (a + 1)"
    );
    let TableElement::Column(c) = &t.elements[2] else {
        panic!("{:?}", t.elements[2]);
    };
    assert_eq!(c.data_type.to_string(), "struct<`x`:int>");
    assert_eq!(t.using.as_ref().map(|u| u.span.text(sql)), Some("parquet"));
    assert_eq!(t.options.len(), 2);
    assert_eq!(
        t.options[0].value.as_ref().map(|e| e.span.text(sql)),
        Some("1")
    );
    assert_eq!(t.options[1].key.span.text(sql), "password");
    assert_eq!(t.partitioned_by.len(), 4);
    assert!(matches!(
        &t.partitioned_by[2].kind,
        ast::PartitionFieldKind::Transform(tr)
            if tr.args.len() == 1
                && matches!(&tr.name, ast::DdlName::Parts(n) if n.span.text(sql) == "years")
    ));
    assert!(matches!(
        t.partitioned_by[3].kind,
        ast::PartitionFieldKind::Column(ref col) if col.not_null && col.comment.as_deref() == Some("p")
    ));
    assert_eq!(t.cluster_by.len(), 2);
    assert_eq!(t.location.as_deref(), Some("/tmp/t"));
    assert_eq!(t.comment.as_deref(), Some("table"));
    assert_eq!(
        t.collation.as_ref().map(|c| c.span.text(sql)),
        Some("UTF8_LCASE")
    );
    assert_eq!(t.tblproperties.len(), 2);
    assert!(t.tblproperties[1].value.is_none());
    assert!(t.as_query.is_some());
    assert_eq!(t.span.text(sql), sql);
    // A CTAS without `AS`, and one without columns.
    let t = create("CREATE TABLE t USING parquet SELECT a FROM s");
    assert!(t.as_query.is_some() && t.elements.is_empty());
    let t = create("CREATE TABLE t AS SELECT 1");
    assert!(t.as_query.is_some());
    // A bare `VALUES` is a query too.
    let t = create("CREATE TABLE t AS VALUES (1), (2)");
    assert!(t.as_query.is_some());
    // `parse_query` takes queries only.
    assert_eq!(
        err("CREATE TABLE t (a INT)", &d).error.error_class(),
        ErrorClass::ParseSyntaxError
    );
    // Under `EXPLAIN`, a command is the explained statement.
    let ast::Statement::Explain(e) = parse_statement("EXPLAIN CREATE TABLE t (a INT)", &d).unwrap()
    else {
        panic!("not an EXPLAIN")
    };
    assert!(matches!(*e.statement, ast::Statement::CreateTable(_)));
}

/// `CREATE TABLE` bucketing, skew, row formats and file formats.
#[test]
fn create_table_storage_clauses() {
    use ast::{FileFormatKind, RowFormatKind, Statement};
    let d = ParserOptions::default();
    let create = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateTable(t) => t,
        other => panic!("{sql} is {other:?}"),
    };
    let t =
        create("CREATE TABLE t (a INT) CLUSTERED BY (a) SORTED BY (b ASC, c DESC) INTO 2 BUCKETS");
    let b = t.bucket.as_ref().unwrap();
    assert_eq!(
        b.clustered_by
            .iter()
            .map(|i| i.value.as_str())
            .collect::<Vec<_>>(),
        ["a"]
    );
    assert_eq!(b.sorted_by.len(), 2);
    assert_eq!(b.sorted_by[0].asc, Some(true));
    assert_eq!(b.sorted_by[1].asc, Some(false));
    assert_eq!(b.buckets, 2);
    // `SKEWED BY` parses, and Spark then rejects it in `CREATE TABLE`.
    let skewed = |sql: &str| {
        let e = parse_statement(sql, &d).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidStatementOrClause,
            "{sql}"
        );
        assert!(
            e.error.to_string().contains("CREATE TABLE ... SKEWED BY"),
            "{sql}: {}",
            e.error
        );
        assert_eq!(e.span.range(), 0..sql.len());
        match crate::parser::parse_statement_unchecked(sql, d).unwrap() {
            Statement::CreateTable(t) => t,
            other => panic!("{sql} is {other:?}"),
        }
    };
    let t = skewed("CREATE TABLE t (a INT) SKEWED BY (a) ON (1, 'x') STORED AS DIRECTORIES");
    let s = t.skew.as_ref().unwrap();
    assert!(s.stored_as_directories && s.values.len() == 1 && s.values[0].len() == 2);
    let t = skewed("CREATE TABLE t (a INT) SKEWED BY (a, b) ON ((1, 2), (3, 4))");
    let s = t.skew.as_ref().unwrap();
    assert!(!s.stored_as_directories && s.values.len() == 2);
    let t = create(
        "CREATE TABLE t (a INT) ROW FORMAT SERDE 'com.example.SerDe' WITH SERDEPROPERTIES ('k' = 'v')",
    );
    let ast::RowFormatKind::Serde(serde) = &t.row_format.as_ref().unwrap().kind else {
        panic!("{:?}", t.row_format);
    };
    assert_eq!(serde.class, "com.example.SerDe");
    assert_eq!(serde.properties.len(), 1);
    let sql = "CREATE TABLE t (a INT) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' ESCAPED BY '\\\\' \
        COLLECTION ITEMS TERMINATED BY '|' MAP KEYS TERMINATED BY ':' LINES TERMINATED BY '\n' NULL DEFINED AS 'NULL'";
    let t = create(sql);
    let RowFormatKind::Delimited(del) = &t.row_format.as_ref().unwrap().kind else {
        panic!("{:?}", t.row_format);
    };
    assert_eq!(del.span.text(sql).lines().count(), 2);
    let delimited = del.span.text(sql);
    assert!(delimited.starts_with("DELIMITED"));
    assert!(delimited.ends_with("'NULL'"));
    assert_eq!(del.fields_terminated_by.as_deref(), Some(","));
    assert_eq!(del.escaped_by.as_deref(), Some("\\"));
    assert_eq!(del.collection_items_terminated_by.as_deref(), Some("|"));
    assert_eq!(del.map_keys_terminated_by.as_deref(), Some(":"));
    assert_eq!(del.lines_terminated_by.as_deref(), Some("\n"));
    assert_eq!(del.null_defined_as.as_deref(), Some("NULL"));
    let t = create("CREATE TABLE t (a INT) ROW FORMAT DELIMITED");
    assert!(matches!(
        t.row_format.as_ref().unwrap().kind,
        RowFormatKind::Delimited(_)
    ));
    let t = create("CREATE TABLE t (a INT) STORED AS parquet");
    assert!(matches!(
        t.file_format.as_ref().unwrap().kind,
        FileFormatKind::StoredAs(_)
    ));
    let t = create("CREATE TABLE t (a INT) STORED BY 'com.example.Storage'");
    let FileFormatKind::StoredBy(h) = &t.file_format.as_ref().unwrap().kind else {
        panic!("{:?}", t.file_format);
    };
    assert_eq!(h, "com.example.Storage");
}

/// `CREATE TABLE ... LIKE ...`.
#[test]
fn create_table_like_statements() {
    use ast::{DdlName, Statement};
    let d = ParserOptions::default();
    let like = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateTableLike(t) => t,
        other => panic!("{sql} is {other:?}"),
    };
    let t = like("CREATE TABLE IF NOT EXISTS a.t LIKE s");
    assert!(t.if_not_exists);
    assert!(matches!(t.name, DdlName::Parts(_)));
    assert!(matches!(t.source, DdlName::Parts(_)));
    assert!(t.using.is_none() && t.location.is_none() && t.tblproperties.is_empty());
    let sql = "CREATE TABLE t LIKE s USING parquet ROW FORMAT SERDE 'x' STORED AS parquet \
        LOCATION '/tmp/t' TBLPROPERTIES ('a' = '1')";
    let t = like(sql);
    assert!(t.using.is_some() && t.row_format.is_some() && t.file_format.is_some());
    assert_eq!(t.location.as_deref(), Some("/tmp/t"));
    assert_eq!(t.tblproperties.len(), 1);
    assert_eq!(t.span.text(sql), sql);
    // `OR REPLACE` takes no `LIKE`: the name is followed by `LIKE`.
    let e = parse_statement("CREATE OR REPLACE TABLE t LIKE s", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 26..30);
}

/// `CREATE TABLE` column constraints, identity columns and `IDENTIFIER` names.
#[test]
fn create_table_columns_constraints_and_names() {
    use ast::{ColumnName, ConstraintCharacteristic, DdlName, Statement, TableConstraintKind};
    let d = ParserOptions::default();
    let create = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateTable(t) => t,
        other => panic!("{sql} is {other:?}"),
    };
    let t = create(
        "CREATE TABLE t (a INT PRIMARY KEY, b STRING UNIQUE, c INT REFERENCES r (id), \
        d INT CHECK (d > 0), CONSTRAINT pk PRIMARY KEY (a, b) ENFORCED, \
        FOREIGN KEY (c) REFERENCES r, CONSTRAINT u UNIQUE (b) NOT ENFORCED NORELY, \
        CONSTRAINT ck CHECK (a > 0) RELY, e BIGINT GENERATED ALWAYS AS IDENTITY \
        (START WITH 10 INCREMENT BY 5), f BIGINT GENERATED BY DEFAULT AS IDENTITY, \
        g INT GENERATED ALWAYS AS IDENTITY)",
    );
    assert_eq!(t.elements.len(), 11);
    let ast::TableElement::Column(a) = &t.elements[0] else {
        panic!("{:?}", t.elements[0]);
    };
    assert_eq!(a.constraints.len(), 1);
    assert!(matches!(
        a.constraints[0].kind,
        ast::ColumnConstraintKind::PrimaryKey
    ));
    let ast::TableElement::Constraint(pk) = &t.elements[4] else {
        panic!("{:?}", t.elements[4]);
    };
    assert_eq!(pk.name.as_ref().map(|n| n.value.as_str()), Some("pk"));
    assert!(matches!(&pk.kind, TableConstraintKind::PrimaryKey(cols) if cols.len() == 2));
    assert_eq!(
        pk.characteristics,
        [ConstraintCharacteristic::Enforced(true)]
    );
    let ast::TableElement::Constraint(u) = &t.elements[6] else {
        panic!("{:?}", t.elements[6]);
    };
    assert_eq!(
        u.characteristics,
        [
            ConstraintCharacteristic::Enforced(false),
            ConstraintCharacteristic::Rely(false)
        ]
    );
    let ast::TableElement::Column(e) = &t.elements[8] else {
        panic!("{:?}", t.elements[8]);
    };
    let Some(ast::GeneratedColumn {
        kind: ast::GeneratedKind::Identity(id),
        ..
    }) = &e.generated
    else {
        panic!("{:?}", e.generated);
    };
    assert_eq!(id.how, ast::IdentityHow::Always);
    assert_eq!((id.start, id.step), (Some(10), Some(5)));
    // `IDENTIFIER(...)` names: a whole table name, and a column name.
    let t =
        create("CREATE TABLE IDENTIFIER('ta' || 'b') (c1 INT, IDENTIFIER('c2') STRING) USING CSV");
    let DdlName::Dynamic(dyn_name) = &t.name else {
        panic!("{t:?}");
    };
    assert_eq!(
        dyn_name.span.text(
            "CREATE TABLE IDENTIFIER('ta' || 'b') (c1 INT, IDENTIFIER('c2') STRING) USING CSV"
        ),
        "IDENTIFIER('ta' || 'b')"
    );
    let ast::TableElement::Column(c2) = &t.elements[1] else {
        panic!("{:?}", t.elements[1]);
    };
    assert!(matches!(c2.name, ColumnName::Dynamic(_)));
    // In legacy identifier mode a column `IDENTIFIER(...)` fails at the
    // `(`, as in Spark, while a table one still parses.
    let legacy = ParserOptions {
        legacy_identifier_clause: true,
        ..d
    };
    let e =
        parse_statement("CREATE TABLE tab(IDENTIFIER('c1') INT) USING CSV", &legacy).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 27..28);
    let Statement::CreateTable(t) =
        parse_statement("CREATE TABLE IDENTIFIER('tab')(c1 INT) USING CSV", &legacy).unwrap()
    else {
        panic!("legacy table name")
    };
    assert!(matches!(t.name, DdlName::Dynamic(_)));
}

/// `CREATE TABLE` errors: positions and classes.
#[test]
fn create_table_errors() {
    let d = ParserOptions::default();
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    // The name is required.
    let e = fail("CREATE TABLE");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 12..12);
    // A column needs a type.
    let e = fail("CREATE TABLE t (a)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 17..18);
    // `NOT` needs `NULL`.
    let e = fail("CREATE TABLE t (a INT NOT)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 25..26);
    // A `DEFAULT` needs its expression.
    let e = fail("CREATE TABLE t (a INT DEFAULT)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // `PARTITIONED BY` needs its list.
    let e = fail("CREATE TABLE t (a INT) PARTITIONED BY");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // `INTO` needs its bucket count.
    let e = fail("CREATE TABLE t (a INT) CLUSTERED BY (a) INTO BUCKETS");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 45..52);
    // `TBLPROPERTIES` takes literals only.
    let e = fail("CREATE TABLE t (a INT) TBLPROPERTIES ('a' = 1L)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 44..46);
    // A qualified column type is rejected in `PARTITIONED BY`.
    let e = fail("CREATE TABLE t (a INT) PARTITIONED BY (a INT COLLATE x)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // `LIKE` needs its source.
    let e = fail("CREATE TABLE t LIKE");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // Anything else fails where it stands.
    let e = fail("CREATE TABLE");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // An unknown type fails with Spark's class, at the type.
    let e = fail("create table foo (with baz)");
    assert_eq!(e.error.error_class(), ErrorClass::UnsupportedDatatype);
    // `STRING COLLATE` without a name leaves `COLLATE` as extra input.
    let e = parse_data_type("string collate", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 7..14);
    // Doubly-quoted names fail without the option, as in Spark.
    let e = fail("DROP TABLE \"myview\"");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 11..19);
}

/// `DROP TABLE [IF EXISTS] [PURGE]`.
#[test]
fn drop_table_statements() {
    use ast::{DdlName, Statement};
    let d = ParserOptions::default();
    let drop = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::DropTable(t) => t,
        other => panic!("{sql} is {other:?}"),
    };
    let t = drop("DROP TABLE t;");
    assert!(!t.if_exists && !t.purge);
    assert_eq!(t.span.text("DROP TABLE t;"), "DROP TABLE t");
    let t = drop("DROP TABLE IF EXISTS a.b.c PURGE");
    assert!(t.if_exists && t.purge);
    let DdlName::Parts(name) = &t.name else {
        panic!("{t:?}");
    };
    assert_eq!(name.parts.len(), 3);
    let t = drop("DROP TABLE IF EXISTS IDENTIFIER('ta' || 'b')");
    assert!(matches!(t.name, DdlName::Dynamic(_)));
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    let e = fail("DROP TABLE");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 10..10);
    let e = fail("DROP TABLE IF EXISTS");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 20..20);
}

/// `CREATE`/`DROP` `DATABASE`|`SCHEMA`|`NAMESPACE` and `USE`.
#[test]
fn namespace_and_use_statements() {
    use ast::{DdlName, DropBehavior, NamespaceKind, Statement};
    let d = ParserOptions::default();
    let create = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateNamespace(n) => n,
        other => panic!("{sql} is {other:?}"),
    };
    let n = create(
        "CREATE DATABASE IF NOT EXISTS db COMMENT 'c' LOCATION '/tmp/db' \
        DEFAULT COLLATION UTF8_LCASE WITH DBPROPERTIES ('k' = 'v', flag)",
    );
    assert_eq!(n.kind, NamespaceKind::Database);
    assert!(n.if_not_exists);
    assert_eq!(n.comment.as_deref(), Some("c"));
    assert_eq!(n.location.as_deref(), Some("/tmp/db"));
    assert!(n.collation.is_some());
    assert_eq!(n.properties.len(), 2);
    assert!(n.properties[1].value.is_none());
    let n = create("CREATE SCHEMA s WITH PROPERTIES ('a' = 1)");
    assert_eq!(n.kind, NamespaceKind::Schema);
    assert_eq!(n.properties.len(), 1);
    let n = create("CREATE NAMESPACE n");
    assert_eq!(n.kind, NamespaceKind::Namespace);
    let drop = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::DropNamespace(n) => n,
        other => panic!("{sql} is {other:?}"),
    };
    let n = drop("DROP DATABASE IF EXISTS db CASCADE");
    assert_eq!(n.kind, NamespaceKind::Database);
    assert!(n.if_exists);
    assert_eq!(n.behavior, Some(DropBehavior::Cascade));
    let n = drop("DROP SCHEMA s RESTRICT");
    assert_eq!(n.behavior, Some(DropBehavior::Restrict));
    let n = drop("DROP NAMESPACE IDENTIFIER('id' || 'ent')");
    assert!(matches!(n.name, DdlName::Dynamic(_)));
    let use_ = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::Use(u) => u,
        other => panic!("{sql} is {other:?}"),
    };
    let u = use_("USE mydb1");
    assert!(u.kind.is_none());
    let u = use_("USE SCHEMA IDENTIFIER('id' || 'ent')");
    assert_eq!(u.kind, Some(NamespaceKind::Schema));
    assert!(matches!(u.name, DdlName::Dynamic(_)));
    let u = use_("USE NAMESPACE a.b");
    assert_eq!(u.kind, Some(NamespaceKind::Namespace));
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    let e = fail("CREATE SCHEMA");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail("DROP DATABASE db CASCADE RESTRICT");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail("USE");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 3..3);
    // Doubly-quoted names fail without the option, as in Spark.
    let e = fail("USE SCHEMA \"not_exist\"");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 11..22);
}

/// `COLLATE` in types, and `AS (columns)` aliases without a table name.
#[test]
fn collate_types_and_nameless_aliases() {
    use ast::Statement;
    let d = ParserOptions::default();
    // `STRING COLLATE ...`, including in nested types.
    let t = parse_data_type("STRING COLLATE UTF8_LCASE", &d).unwrap();
    assert_eq!(t.to_string(), "string collate UTF8_LCASE");
    assert_eq!(
        t.to_data_type().unwrap_err().error_class().name(),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    let t = parse_data_type("STRUCT<a: STRING COLLATE x, b: INT>", &d).unwrap();
    assert_eq!(t.to_string(), "struct<`a`:string collate x,`b`:int>");
    // `CHAR` and `VARCHAR` take one too, with any quoting of the name.
    let t = parse_data_type("CHAR(10) COLLATE `a``b`", &d).unwrap();
    assert_eq!(t.to_string(), "char(10) collate `a``b`");
    let t = parse_data_type("VARCHAR(10) COLLATE x", &d).unwrap();
    assert_eq!(t.to_string(), "varchar(10) collate x");
    // A non-ASCII collation name is no identifier, as in Spark.
    let e = parse_data_type("STRING COLLATE \u{e9}", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::InvalidIdentifier);
    // `COLLATE` on any other type is a syntax error.
    let e = parse_data_type("INT COLLATE x", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // A cast to a collated type parses, then fails to lower as
    // unsupported, as in Spark (which analyzes it).
    let q = ok("SELECT CAST(a AS STRING COLLATE UTF8_LCASE) FROM t");
    let e = crate::lower(&q, krishiv_plan::SourceId(0)).unwrap_err();
    assert_eq!(e.error.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    // `VALUES ... AS (columns)`: a column list without a table name.
    let q = ok("SELECT * FROM (VALUES (1), (2), (1) AS (my_col)) GROUP BY 1");
    let ast::SetExprKind::Select(s) = &q.body.kind else {
        panic!("{q:?}");
    };
    let ast::TableFactorKind::Subquery(inner) = &s.from.as_ref().unwrap().relations[0].primary.kind
    else {
        panic!("{q:?}");
    };
    let ast::SetExprKind::Values(v) = &inner.body.kind else {
        panic!("{inner:?}");
    };
    assert_eq!(v.rows.len(), 3);
    let alias = v.alias.as_ref().unwrap();
    assert!(alias.name.is_none());
    assert_eq!(alias.columns.len(), 1);
    assert_eq!(
        alias
            .span
            .text("SELECT * FROM (VALUES (1), (2), (1) AS (my_col)) GROUP BY 1"),
        "AS (my_col)"
    );
    // A missing column list is still an error.
    let e = err("SELECT * FROM (VALUES (1)) AS ()", &d);
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // A nameless alias lowers like an unnamed one, keeping the columns.
    let q = ok("SELECT * FROM (SELECT 1) AS (a)");
    let ast::SetExprKind::Select(s) = &q.body.kind else {
        panic!("{q:?}");
    };
    let rel = &s.from.as_ref().unwrap().relations[0];
    assert!(rel.primary.alias.as_ref().unwrap().name.is_none());
    crate::lower(&q, krishiv_plan::SourceId(0)).unwrap();
    let Statement::CreateTable(t) =
        parse_statement("CREATE TABLE t AS SELECT * FROM (VALUES (1)) AS (c)", &d).unwrap()
    else {
        panic!("not a CREATE TABLE")
    };
    assert!(t.as_query.is_some());
}

/// New commands enforce the recursion limit, reported at the whole
/// statement.
#[test]
fn ddl_commands_enforce_the_depth_limit() {
    use crate::MAX_DEPTH;
    let d = ParserOptions::default();
    // A CTAS nested past the limit fails; one level less parses.
    let deep = |n: usize| {
        format!(
            "CREATE TABLE t AS SELECT {}1{}",
            "(".repeat(n),
            ")".repeat(n)
        )
    };
    let (mut ok, mut bad) = (0, MAX_DEPTH + 10);
    while ok + 1 < bad {
        let mid = (ok + bad) / 2;
        if parse_statement(&deep(mid), &d).is_ok() {
            ok = mid;
        } else {
            bad = mid;
        }
    }
    assert!(ok > 0);
    let sql = deep(bad);
    let e = parse_statement(&sql, &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::FailedToParseTooComplex);
    assert_eq!(e.span.range(), 0..sql.len());
    // A deeply nested `DEFAULT` fails the same way.
    let sql = format!(
        "CREATE TABLE t (a INT DEFAULT {}1{})",
        "(".repeat(MAX_DEPTH),
        ")".repeat(MAX_DEPTH)
    );
    let e = parse_statement(&sql, &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::FailedToParseTooComplex);
    assert_eq!(e.span.range(), 0..sql.len());
    // So does a deeply nested parenthesized query.
    let sql = format!(
        "CREATE TABLE t (SELECT {}1{})",
        "(".repeat(MAX_DEPTH),
        ")".repeat(MAX_DEPTH)
    );
    let e = parse_statement(&sql, &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::FailedToParseTooComplex);
    assert_eq!(e.span.range(), 0..sql.len());
}

/// `CREATE TABLE` errors, continued: every clause's malformed forms fail
/// where they stand.
#[test]
fn create_table_clause_errors() {
    let d = ParserOptions::default();
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    let syntax = |sql: &str| {
        let e = fail(sql);
        assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError, "{sql}");
        e
    };
    // `GENERATED` needs its form.
    syntax("CREATE TABLE t (a INT GENERATED)");
    syntax("CREATE TABLE t (a INT GENERATED ALWAYS)");
    syntax("CREATE TABLE t (a INT GENERATED ALWAYS AS 1)");
    // A `COMMENT` needs its string.
    let e = syntax("CREATE TABLE t (a INT COMMENT 1)");
    assert_eq!(e.span.range(), 30..31);
    // Constraints need their shape.
    syntax("CREATE TABLE t (a INT, CONSTRAINT c FOO)");
    syntax("CREATE TABLE t (a INT, PRIMARY FOO)");
    // `OPTIONS` needs its value after `=`.
    syntax("CREATE TABLE t (a INT) OPTIONS (a = )");
    // `COMMENT`, `LOCATION` and `COLLATION` need their strings.
    syntax("CREATE TABLE t (a INT) COMMENT 1");
    syntax("CREATE TABLE t (a INT) DEFAULT COLLATION");
    // `CLUSTER BY` needs its list.
    syntax("CREATE TABLE t (a INT) CLUSTER BY a");
    // `STORED` needs `AS` or `BY`.
    syntax("CREATE TABLE t (a INT) STORED");
    // `ROW FORMAT` needs `SERDE` or `DELIMITED`.
    syntax("CREATE TABLE t (a INT) ROW FORMAT");
    syntax("CREATE TABLE t (a INT) ROW FORMAT FOO");
    // `SKEWED ... STORED` needs `AS DIRECTORIES`.
    syntax("CREATE TABLE t (a INT) SKEWED BY (a) ON (1) STORED");
    syntax("CREATE TABLE t (a INT) SKEWED BY (a) ON (1) STORED AS");
    // `AS` needs its query.
    syntax("CREATE TABLE t (a INT) AS");
    // An unclosed parenthesized query fails inside it.
    let e = syntax("CREATE TABLE t (SELECT 1");
    assert_eq!(e.span.range(), 23..24);
    // An unclosed `IDENTIFIER(` fails naming the missing `)`.
    let e = syntax("CREATE TABLE IDENTIFIER('a'");
    assert_eq!(e.span.range(), 27..27);
    // Anything else fails where it stands.
    let e = syntax("CREATE FOO");
    assert_eq!(e.span.range(), 7..10);
    let e = syntax("DROP FOO");
    assert_eq!(e.span.range(), 5..8);
    // `WITH` needs its properties.
    syntax("CREATE SCHEMA s WITH");
    // `USE` needs its name.
    let e = syntax("USE SCHEMA");
    assert_eq!(e.span.range(), 10..10);
}

/// `PARTITIONED BY` field forms.
#[test]
fn partition_field_forms() {
    use ast::{DdlName, PartitionFieldKind, Statement};
    let d = ParserOptions::default();
    let fields = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateTable(t) => t.partitioned_by,
        other => panic!("{sql} is {other:?}"),
    };
    // A dotted name is an identity transform.
    let f = fields("CREATE TABLE t (a INT) PARTITIONED BY (a.b)");
    assert!(matches!(
        &f[0].kind,
        PartitionFieldKind::Transform(tr)
            if matches!(&tr.name, DdlName::Parts(n) if n.parts.len() == 2) && tr.args.is_empty()
    ));
    // A dynamic name without a type is one too.
    let f = fields("CREATE TABLE t (a INT) PARTITIONED BY (IDENTIFIER('c'))");
    assert!(matches!(
        &f[0].kind,
        PartitionFieldKind::Transform(tr)
            if matches!(&tr.name, DdlName::Dynamic(_)) && tr.args.is_empty()
    ));
    // `NOT NULL` without `NULL`, a `COMMENT` without a string, and a
    // broken type fail inside the column.
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    let e = fail("CREATE TABLE t (a INT) PARTITIONED BY (a INT NOT)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 48..49);
    let e = fail("CREATE TABLE t (a INT) PARTITIONED BY (a INT COMMENT 1)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail("CREATE TABLE t (a INT) PARTITIONED BY (a array<int)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail("CREATE TABLE t (a INT) PARTITIONED BY (a 1)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 41..42);
}

/// `TBLPROPERTIES` literal values, `OPTIONS` expressions and storage
/// defaults.
#[test]
fn property_values_and_storage_defaults() {
    use ast::{RowFormatKind, Statement};
    let d = ParserOptions::default();
    let create = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateTable(t) => t,
        other => panic!("{sql} is {other:?}"),
    };
    let t = create("CREATE TABLE t (a INT) TBLPROPERTIES ('a' = TRUE, 'b' = 1.5, 'c' = \"x\")");
    assert_eq!(t.tblproperties.len(), 3);
    assert!(t.tblproperties.iter().all(|p| p.value.is_some()));
    // An `OPTIONS` value without `=` reads as the next expression.
    let t = create("CREATE TABLE t (a INT) OPTIONS (a b)");
    assert_eq!(t.options.len(), 1);
    // `SORTED BY` without a direction, `ROW FORMAT SERDE` without
    // properties, and an `IDENTITY` without options.
    let t = create("CREATE TABLE t (a INT) CLUSTERED BY (a) SORTED BY (b) INTO 2 BUCKETS");
    assert_eq!(t.bucket.as_ref().unwrap().sorted_by[0].asc, None);
    let t = create("CREATE TABLE t (a INT) ROW FORMAT SERDE 'x'");
    assert!(matches!(
        t.row_format.as_ref().unwrap().kind,
        RowFormatKind::Serde(_)
    ));
    // A bucket count that is no integer fails.
    let e = parse_statement(
        "CREATE TABLE t (a INT) CLUSTERED BY (a) INTO 99999999999999999999999 BUCKETS",
        &d,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
}

/// A parenthesized query is the CTAS query: no columns, no clauses.
#[test]
fn parenthesized_ctas_forms() {
    use ast::Statement;
    let d = ParserOptions::default();
    let t = match parse_statement("CREATE TABLE t (SELECT 1)", &d).unwrap() {
        Statement::CreateTable(t) => t,
        other => panic!("{other:?}"),
    };
    assert!(t.elements.is_empty() && t.using.is_none());
    let q = t.as_query.as_ref().unwrap();
    assert_eq!(q.span.text("CREATE TABLE t (SELECT 1)"), "(SELECT 1)");
    // Clauses do not follow one: it ends the statement.
    let e = parse_statement("CREATE TABLE t (SELECT 1) USING parquet", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // `OR REPLACE` with anything but a table fails where it stands.
    let e = parse_statement("CREATE OR REPLACE VIEW v AS SELECT 1", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 18..22);
    // So does anything else.
    let e = parse_statement("CREATE TEMPORARY FOO", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 17..20);
}

/// `NOT NULL` without `NULL` before a `COMMENT`, and identity column
/// counts that are no integers.
#[test]
fn column_option_edge_cases() {
    let d = ParserOptions::default();
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    let e = fail("CREATE TABLE t (a INT NOT COMMENT 'x')");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail("CREATE TABLE t (a INT GENERATED ALWAYS AS IDENTITY (START WITH x))");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail(
        "CREATE TABLE t (a INT GENERATED ALWAYS AS IDENTITY (START WITH 99999999999999999999999))",
    );
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // Inline constraint names and references.
    let t = match parse_statement(
        "CREATE TABLE t (a INT CONSTRAINT c PRIMARY KEY, b INT REFERENCES r, c INT CONSTRAINT d CHECK (c > 0))",
        &d,
    )
    .unwrap()
    {
        ast::Statement::CreateTable(t) => t,
        other => panic!("{other:?}"),
    };
    assert_eq!(t.elements.len(), 3);
    let ast::TableElement::Column(a) = &t.elements[0] else {
        panic!("{:?}", t.elements[0]);
    };
    assert_eq!(a.constraints.len(), 1);
    assert_eq!(a.constraints[0].name.as_ref().unwrap().value, "c");
    let e = fail("CREATE TABLE t (a INT PRIMARY FOO)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
}

/// `PARTITIONED BY` restores and reports.
#[test]
fn partition_field_edge_cases() {
    use ast::{PartitionFieldKind, Statement};
    let d = ParserOptions::default();
    let fields = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::CreateTable(t) => t.partitioned_by,
        other => panic!("{sql} is {other:?}"),
    };
    // A `COLLATE`d string reads as an identity transform, which fails at
    // the `COLLATE`.
    let f = fields("CREATE TABLE t (a INT) PARTITIONED BY (a STRING)");
    assert!(matches!(&f[0].kind, PartitionFieldKind::Column(_)));
    let e = parse_statement(
        "CREATE TABLE t (a INT) PARTITIONED BY (a STRING COLLATE x)",
        &d,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // A broken type rewinds to an identity transform, which fails after
    // the name.
    let e = parse_statement("CREATE TABLE t (a INT) PARTITIONED BY (a array<int)", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // A bad name fails as one.
    let e = parse_statement("CREATE TABLE t (a INT) PARTITIONED BY (a \u{e9})", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::InvalidIdentifier);
    // An unclosed type fails at the end of the input.
    let e =
        parse_statement("CREATE TABLE t (x INT) PARTITIONED BY (a decimal(10)", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 52..52);
    // `NOT NULL` without `NULL` before a `COMMENT` fails at the `COMMENT`.
    let e = parse_statement(
        "CREATE TABLE t (a INT) PARTITIONED BY (a INT NOT COMMENT 'x')",
        &d,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // A table constraint with referenced columns.
    let t = match parse_statement(
        "CREATE TABLE t (a INT, b INT, CONSTRAINT fk FOREIGN KEY (a) REFERENCES r (id))",
        &d,
    )
    .unwrap()
    {
        Statement::CreateTable(t) => t,
        other => panic!("{other:?}"),
    };
    let ast::TableElement::Constraint(c) = &t.elements[2] else {
        panic!("{:?}", t.elements[2]);
    };
    let ast::TableConstraintKind::ForeignKey(fk) = &c.kind else {
        panic!("{c:?}");
    };
    assert_eq!(fk.ref_columns.len(), 1);
    assert_eq!(fk.table.parts.len(), 1);
}

/// The checks Spark's builder makes on a parsed `CREATE TABLE`, with the
/// error classes and order of Spark 4.2 (each case checked on Spark).
#[test]
fn create_table_builder_checks() {
    use ast::{GeneratedKind, Statement, TableElement};
    let d = ParserOptions::default();
    let fail = |sql: &str| {
        let e = parse_statement(sql, &d).unwrap_err();
        (e.error.error_class(), e.error.to_string(), e.span.range())
    };
    // A clause written twice; the first in Spark's check order wins over
    // the order in the text.
    for (clause, text) in [
        ("TBLPROPERTIES", "TBLPROPERTIES ('a' = 'b')"),
        ("OPTIONS", "OPTIONS (k 1)"),
        ("PARTITIONED BY", "PARTITIONED BY (a)"),
        ("STORED AS/BY", "STORED AS parquet"),
        ("ROW FORMAT", "ROW FORMAT DELIMITED"),
        ("COMMENT", "COMMENT 'x'"),
        ("DEFAULT COLLATION", "DEFAULT COLLATION UTF8_BINARY"),
        ("CLUSTERED BY", "CLUSTERED BY (a) INTO 2 BUCKETS"),
        ("CLUSTER BY", "CLUSTER BY (a)"),
        ("LOCATION", "LOCATION '/a'"),
    ] {
        let sql = format!("CREATE TABLE t (a INT) {text} {text}");
        let (class, message, span) = fail(&sql);
        assert_eq!(class, ErrorClass::DuplicateClauses, "{sql}");
        assert!(
            message.contains(&format!("Found duplicate clauses: {clause}.")),
            "{sql}: {message}"
        );
        assert_eq!(span, 0..sql.len());
    }
    let (_, message, _) =
        fail("CREATE TABLE t (a INT) LOCATION '/a' LOCATION '/b' COMMENT 'x' COMMENT 'y'");
    assert!(message.contains("clauses: COMMENT."), "{message}");
    let (_, message, _) = fail(
        "CREATE TABLE t (a INT) COMMENT 'x' TBLPROPERTIES ('a' = 'b') COMMENT 'y' TBLPROPERTIES ('c' = 'd')",
    );
    assert!(message.contains("clauses: TBLPROPERTIES."), "{message}");

    // A column descriptor written twice: the first repeat in the text.
    for (sql, descriptor) in [
        ("CREATE TABLE t (a INT COMMENT 'x' COMMENT 'y')", "COMMENT"),
        ("CREATE TABLE t (a INT NOT NULL NOT NULL)", "NOT NULL"),
        ("CREATE TABLE t (a INT DEFAULT 1 DEFAULT 2)", "DEFAULT"),
        (
            "CREATE TABLE t (a INT DEFAULT 1 COMMENT 'x' COMMENT 'y' DEFAULT 2)",
            "COMMENT",
        ),
        (
            "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS (1) GENERATED ALWAYS AS IDENTITY)",
            "GENERATED ALWAYS AS",
        ),
        (
            "CREATE TABLE t (a INT CHECK (a > 0) PRIMARY KEY)",
            "CONSTRAINT",
        ),
        (
            "CREATE TABLE t (IDENTIFIER('a') INT COMMENT 'x' COMMENT 'y')",
            "COMMENT",
        ),
    ] {
        let (class, message, span) = fail(sql);
        assert_eq!(
            class,
            ErrorClass::CreateTableColumnDescriptorDuplicate,
            "{sql}"
        );
        assert!(
            message.contains(&format!("column a specifies descriptor \"{descriptor}\"")),
            "{sql}: {message}"
        );
        assert_eq!(span, 0..sql.len());
    }

    // Identity options: negative counts parse; a repeated option and a
    // zero step fail, after the column's descriptors.
    let sql =
        "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY (START WITH -1 INCREMENT BY - 2))";
    let Statement::CreateTable(t) = parse_statement(sql, &d).unwrap() else {
        panic!("{sql}");
    };
    let TableElement::Column(c) = &t.elements[0] else {
        panic!("{t:?}");
    };
    let Some(GeneratedKind::Identity(i)) = c.generated.as_ref().map(|g| &g.kind) else {
        panic!("{c:?}");
    };
    assert_eq!((i.start, i.step), (Some(-1), Some(-2)));
    for (sql, option) in [
        (
            "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 1 START WITH 2))",
            "START",
        ),
        (
            "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY \
             (INCREMENT BY 1 START WITH 1 INCREMENT BY 2 START WITH 3))",
            "STEP",
        ),
    ] {
        let (class, message, _) = fail(sql);
        assert_eq!(
            class,
            ErrorClass::IdentityColumnsDuplicatedSequenceGeneratorOption,
            "{sql}"
        );
        assert!(
            message.contains(&format!("generator option: {option}.")),
            "{sql}: {message}"
        );
    }
    let (class, _, _) =
        fail("CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY (INCREMENT BY 0))");
    assert_eq!(class, ErrorClass::IdentityColumnsIllegalStep);
    let (class, _, _) = fail(
        "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY (START WITH 1 START WITH 2) \
         COMMENT 'x' COMMENT 'y')",
    );
    assert_eq!(class, ErrorClass::CreateTableColumnDescriptorDuplicate);
    // A count out of range is a syntax error, as is `10L`.
    let (class, _, _) = fail(
        "CREATE TABLE t (a BIGINT GENERATED ALWAYS AS IDENTITY (START WITH -99999999999999999999))",
    );
    assert_eq!(class, ErrorClass::ParseSyntaxError);

    // `SKEWED BY` and CTAS restrictions.
    let (class, message, _) = fail("CREATE TABLE t (a INT) AS SELECT 1");
    assert_eq!(class, ErrorClass::LegacyErrorTemp0035);
    assert!(
        message
            .contains("Schema may not be specified in a Create Table As Select (CTAS) statement"),
        "{message}"
    );
    let (class, message, _) = fail("CREATE TABLE t PARTITIONED BY (a INT) AS SELECT 1");
    assert_eq!(class, ErrorClass::LegacyErrorTemp0035);
    assert!(
        message.contains(
            "Partition column types may not be specified in Create Table As Select (CTAS)"
        ),
        "{message}"
    );
    let (_, message, _) = fail("CREATE TABLE t (a INT) PARTITIONED BY (b INT) AS SELECT 1");
    assert!(message.contains("Schema may not be specified"), "{message}");
    assert!(parse_statement("CREATE TABLE t PARTITIONED BY (a) AS SELECT 1 AS a", &d).is_ok());

    // Literals anywhere in the statement get the query checks.
    let bad = "DATE '2020-13-45'";
    for sql in [
        format!("CREATE TABLE t (a DATE DEFAULT {bad})"),
        format!("CREATE TABLE t (a DATE GENERATED ALWAYS AS ({bad}))"),
        format!("CREATE TABLE t (a INT CHECK ({bad} IS NULL))"),
        format!("CREATE TABLE t (a INT, CONSTRAINT c CHECK ({bad} IS NULL))"),
        format!("CREATE TABLE t (a INT) OPTIONS (k {bad})"),
        format!("CREATE TABLE t (a INT) PARTITIONED BY (bucket(2, {bad}))"),
        format!("CREATE TABLE IDENTIFIER({bad} || 'x') (a INT)"),
        format!("CREATE TABLE t (IDENTIFIER({bad} || 'x') INT)"),
        format!("CREATE TABLE t (a INT) AS SELECT {bad}"),
        format!("CREATE TABLE t (a INT DEFAULT {bad}) AS SELECT 1"),
    ] {
        assert_eq!(fail(&sql).0, ErrorClass::InvalidTypedLiteral, "{sql}");
    }

    // The order of the checks.
    for (sql, class) in [
        (
            "CREATE TABLE t (a INT COMMENT 'x' COMMENT 'y') COMMENT 'x' COMMENT 'y'",
            ErrorClass::CreateTableColumnDescriptorDuplicate,
        ),
        (
            "CREATE TABLE t (a INT) COMMENT 'x' COMMENT 'y' SKEWED BY (a) ON (1)",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE TABLE t (a INT) COMMENT 'x' COMMENT 'y' AS SELECT 1",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE TABLE t (a INT) OPTIONS (k DATE '2020-13-45') COMMENT 'x' COMMENT 'y'",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE TABLE t SKEWED BY (a) ON (1) AS SELECT DATE '2020-13-45'",
            ErrorClass::InvalidStatementOrClause,
        ),
    ] {
        assert_eq!(fail(sql).0, class, "{sql}");
    }
}

/// With `double_quoted_identifiers`, a double-quoted word after a
/// partition column name reads as its type, as in a column definition.
#[test]
fn create_table_double_quoted_partition_type() {
    let dq = ParserOptions {
        double_quoted_identifiers: true,
        ..ParserOptions::default()
    };
    let column = parse_statement("CREATE TABLE t (a \"int\")", &dq).unwrap_err();
    let partition =
        parse_statement("CREATE TABLE t (a INT) PARTITIONED BY (b \"int\")", &dq).unwrap_err();
    assert_eq!(column.error.error_class(), ErrorClass::UnsupportedDatatype);
    assert_eq!(
        partition.error.error_class(),
        ErrorClass::UnsupportedDatatype
    );
}

/// An error deferred before a speculative parse that then succeeds is
/// still the one reported.
#[test]
fn deferred_error_survives_a_speculative_parse() {
    let d = ParserOptions::default();
    for sql in [
        "SELECT a ! IN (1), CAST(b AS STRING COLLATE c) FROM t",
        "CREATE TABLE IDENTIFIER(CASE WHEN x ! IN (1) THEN 't' END) (SELECT 1)",
    ] {
        let e = parse_statement(sql, &d).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::SyntaxDiscontinuedBangEqualsNot,
            "{sql}"
        );
        let bang = sql.find('!').unwrap();
        assert_eq!(e.span.range(), bang..bang + 1, "{sql}");
    }
}
