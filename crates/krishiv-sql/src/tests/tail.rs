// SPDX-License-Identifier: Apache-2.0
//! Spark 4.2's query syntax tail: `COLLATE` expressions and view
//! collations, `|>` pipe operators, `TABLESAMPLE`, `NEAREST BY` joins,
//! `TABLE(...)` arguments, `IDENTIFIER(...)`, variant paths and `TRANSFORM`.

use super::*;
use crate::ast::{Expr, Statement};

fn defaults() -> ParserOptions {
    ParserOptions::default()
}

/// The error class of a query Spark rejects.
fn class(sql: &str) -> String {
    err(sql, &defaults()).error.error_class().name().to_owned()
}

/// The error class of a statement Spark rejects.
fn stmt_class(sql: &str) -> String {
    parse_statement(sql, &defaults())
        .unwrap_err()
        .error
        .error_class()
        .name()
        .to_owned()
}

/// The first select item's expression of `sql`.
fn first_expr(sql: &str) -> Expr {
    let q = ok(sql);
    let SetExprKind::Select(s) = q.body.kind else {
        panic!("{sql} is not a SELECT");
    };
    s.projection.into_iter().next().unwrap().expr
}

fn unsupported(sql: &str) -> String {
    let q = ok(sql);
    lower(&q, krishiv_plan::SourceId(0))
        .unwrap_err()
        .error
        .to_string()
}

#[test]
fn collate_expressions() {
    let sql = "SELECT 'a' COLLATE UTF8_LCASE, b COLLATE system.builtin.UTF8_BINARY";
    let q = ok(sql);
    let SetExprKind::Select(s) = &q.body.kind else {
        panic!()
    };
    let ExprKind::Collate(inner, name) = &s.projection[0].expr.kind else {
        panic!("{:?}", s.projection[0]);
    };
    assert!(matches!(inner.kind, ExprKind::Literal(_)));
    assert_eq!(name.span.text(sql), "UTF8_LCASE");
    let ExprKind::Collate(_, name) = &s.projection[1].expr.kind else {
        panic!()
    };
    assert_eq!(name.parts.len(), 3);
    assert_eq!(
        s.projection[1].expr.span.text(sql),
        "b COLLATE system.builtin.UTF8_BINARY"
    );
    // It binds tighter than binary operators, and stacks.
    let ExprKind::Binary(_, BinaryOp::Concat, right) = first_expr("SELECT a || b COLLATE x").kind
    else {
        panic!()
    };
    assert!(matches!(right.kind, ExprKind::Collate(..)));
    let ExprKind::Collate(inner, _) = first_expr("SELECT a::INT COLLATE x COLLATE y").kind else {
        panic!()
    };
    assert!(matches!(inner.kind, ExprKind::Collate(..)));
    // Inside a collated cast type the clause belongs to the type.
    let ExprKind::Cast { data_type, .. } = first_expr("SELECT CAST(a AS STRING COLLATE x)").kind
    else {
        panic!()
    };
    assert!(data_type.collation.is_some());
    assert_eq!(class("SELECT a COLLATE"), "PARSE_SYNTAX_ERROR");
    assert_eq!(class("SELECT a COLLATE 1"), "PARSE_SYNTAX_ERROR");
    assert!(unsupported("SELECT a COLLATE x FROM t").contains("COLLATE"));
}

#[test]
fn view_default_collation() {
    let create = |sql: &str| match parse_statement(sql, &defaults()).unwrap() {
        Statement::CreateView(v) => v,
        other => panic!("{other:?}"),
    };
    let v = create("CREATE TEMPORARY VIEW v DEFAULT COLLATION UNICODE AS SELECT 'a'");
    assert_eq!(v.collation.unwrap().value, "UNICODE");
    let v = create("CREATE OR REPLACE TEMP VIEW v (c) DEFAULT COLLATION sr_ci_ai AS SELECT 1");
    assert_eq!(
        (v.columns.len(), v.collation.unwrap().value.as_str()),
        (1, "sr_ci_ai")
    );
    assert!(
        create("CREATE TEMPORARY VIEW v AS SELECT 1")
            .collation
            .is_none()
    );
    assert_eq!(
        stmt_class("CREATE TEMPORARY VIEW v DEFAULT COLLATION AS SELECT 1"),
        "PARSE_SYNTAX_ERROR"
    );
    assert_eq!(
        stmt_class("CREATE TEMPORARY VIEW v AS SELECT 1 DEFAULT COLLATION x"),
        "PARSE_SYNTAX_ERROR"
    );
}

/// The `FROM` clause's first relation primary of a `SELECT * FROM ...`.
fn first_factor(sql: &str) -> ast::TableFactor {
    let q = ok(sql);
    let SetExprKind::Select(s) = q.body.kind else {
        panic!("{sql} is not a SELECT");
    };
    s.from
        .unwrap()
        .relations
        .into_iter()
        .next()
        .unwrap()
        .primary
}

#[test]
fn tablesample_forms() {
    use ast::{SampleMethodKind as M, SampleType};
    let sample = |sql: &str| first_factor(sql).sample.unwrap();
    let s = sample("SELECT * FROM t TABLESAMPLE (10 PERCENT)");
    let M::Percent { negative, value } = s.method.unwrap().kind else {
        panic!()
    };
    assert!(!negative && value == "10" && s.sample_type.is_none() && s.seed.is_none());
    let s = sample("SELECT * FROM t TABLESAMPLE BERNOULLI (0.5 PERCENT) REPEATABLE (42)");
    assert_eq!(
        (s.sample_type, s.seed.as_deref()),
        (Some(SampleType::Bernoulli), Some("42"))
    );
    let s = sample("SELECT * FROM t TABLESAMPLE SYSTEM (-1 PERCENT)");
    let M::Percent { negative, .. } = s.method.unwrap().kind else {
        panic!()
    };
    assert!(negative && s.sample_type == Some(SampleType::System));
    assert!(matches!(
        sample("SELECT * FROM t TABLESAMPLE (5 + 5 ROWS)")
            .method
            .unwrap()
            .kind,
        M::Rows(_)
    ));
    let M::Bucket {
        numerator,
        denominator,
        on,
    } = sample("SELECT * FROM t TABLESAMPLE (BUCKET 1 OUT OF 4)")
        .method
        .unwrap()
        .kind
    else {
        panic!()
    };
    assert_eq!(
        (numerator.as_str(), denominator.as_str(), on.is_none()),
        ("1", "4", true)
    );
    let sql = "SELECT * FROM t TABLESAMPLE (BUCKET 1 OUT OF 4 ON f())";
    assert!(matches!(
        sample(sql).method.unwrap().kind,
        M::Bucket {
            on: Some(ast::BucketOn::Function(_)),
            ..
        }
    ));
    assert!(matches!(
        sample("SELECT * FROM t TABLESAMPLE (BUCKET 1 OUT OF 4 ON c)")
            .method
            .unwrap()
            .kind,
        M::Bucket {
            on: Some(ast::BucketOn::Column(_)),
            ..
        }
    ));
    assert!(matches!(
        sample("SELECT * FROM t TABLESAMPLE (200b)")
            .method
            .unwrap()
            .kind,
        M::Bytes(_)
    ));
    assert!(sample("SELECT * FROM t TABLESAMPLE ()").method.is_none());
}

#[test]
fn tablesample_positions() {
    // The sample comes before the alias and after time travel and options.
    let sql = "SELECT * FROM t WITH ('k' = 'v') TABLESAMPLE (1 ROWS) AS x";
    let f = first_factor(sql);
    assert!(f.sample.is_some() && f.alias.is_some());
    assert!(
        first_factor("SELECT * FROM (SELECT 1) TABLESAMPLE (1 ROWS) s")
            .sample
            .is_some()
    );
    assert!(
        first_factor("SELECT * FROM (t) TABLESAMPLE (1 ROWS) s")
            .sample
            .is_some()
    );
    let q = ok("SELECT * FROM a TABLESAMPLE (1 ROWS) JOIN b TABLESAMPLE (2 ROWS) ON a.x = b.x");
    let SetExprKind::Select(s) = q.body.kind else {
        panic!()
    };
    let rel = &s.from.unwrap().relations[0];
    assert!(rel.primary.sample.is_some());
    let ast::RelationExt::Join(j) = &rel.extensions[0] else {
        panic!()
    };
    assert!(j.right.sample.is_some());
    // Table functions and inline tables take none, and neither does a
    // sample name a column type.
    for sql in [
        "SELECT * FROM range(1) TABLESAMPLE (1 ROWS)",
        "SELECT * FROM VALUES (1) TABLESAMPLE (1 ROWS)",
        "SELECT * FROM t TABLESAMPLE 10 PERCENT",
        "SELECT * FROM t TABLESAMPLE (x PERCENT)",
        "SELECT * FROM t TABLESAMPLE (10 PERCENT) REPEATABLE (1.5)",
        "SELECT * FROM t TABLESAMPLE (BUCKET 1 OUT OF 4 ON a.b)",
    ] {
        assert_eq!(class(sql), "PARSE_SYNTAX_ERROR", "{sql}");
    }
    assert!(unsupported("SELECT * FROM t TABLESAMPLE (1 ROWS)").contains("TABLESAMPLE"));
}

#[test]
fn tablesample_checks() {
    let check = |sample: &str| class(&format!("SELECT * FROM t TABLESAMPLE {sample}"));
    for (sample, expected) in [
        ("()", "_LEGACY_ERROR_TEMP_0014"),
        ("(-1 PERCENT)", "_LEGACY_ERROR_TEMP_0064"),
        ("(101 PERCENT)", "_LEGACY_ERROR_TEMP_0064"),
        ("(BUCKET 2 OUT OF 1)", "_LEGACY_ERROR_TEMP_0064"),
        ("(200b)", "_LEGACY_ERROR_TEMP_0015"),
        ("(200)", "_LEGACY_ERROR_TEMP_0016"),
        ("(BUCKET 1 OUT OF 2 ON c)", "_LEGACY_ERROR_TEMP_0015"),
        ("(BUCKET 1 OUT OF 2 ON f())", "_LEGACY_ERROR_TEMP_0015"),
        (
            "SYSTEM (10 PERCENT) REPEATABLE (1)",
            "UNSUPPORTED_FEATURE.TABLESAMPLE_SYSTEM_REPEATABLE",
        ),
        (
            "SYSTEM (10 ROWS)",
            "UNSUPPORTED_FEATURE.TABLESAMPLE_SYSTEM_SAMPLE_METHOD",
        ),
        (
            "SYSTEM (BUCKET 1 OUT OF 2)",
            "UNSUPPORTED_FEATURE.TABLESAMPLE_SYSTEM_SAMPLE_METHOD",
        ),
        (
            "SYSTEM (200b)",
            "UNSUPPORTED_FEATURE.TABLESAMPLE_SYSTEM_SAMPLE_METHOD",
        ),
    ] {
        assert_eq!(check(sample), expected, "{sample}");
    }
    // The bounds and the usual forms are fine.
    for sample in [
        "(0 PERCENT)",
        "(100 PERCENT)",
        "(BUCKET 1 OUT OF 1)",
        "(7 ROWS)",
        "SYSTEM (50 PERCENT)",
    ] {
        parse_query(
            &format!("SELECT * FROM t TABLESAMPLE {sample}"),
            &defaults(),
        )
        .unwrap();
    }
    let e = err("SELECT * FROM t TABLESAMPLE (-1 PERCENT)", &defaults());
    assert!(
        e.error
            .to_string()
            .contains("Sampling fraction (-0.01) must be on interval [0, 1].")
    );
    let sql = "SELECT * FROM t TABLESAMPLE (101 PERCENT)";
    assert_eq!(
        err(sql, &defaults()).span.text(sql),
        "TABLESAMPLE (101 PERCENT)"
    );
}

fn pipe_ops(sql: &str) -> Vec<ast::PipeOpKind> {
    let q = ok(sql);
    let SetExprKind::Pipe(p) = q.body.kind else {
        panic!("{sql} is not a pipeline: {:?}", q.body.kind);
    };
    p.ops.into_iter().map(|o| o.kind).collect()
}

#[test]
fn pipe_operators_parse_into_a_flat_list() {
    use ast::PipeOpKind as K;
    let sql = "TABLE t |> SELECT a, b AS c GROUP BY a WINDOW w AS (ORDER BY a) \
        |> EXTEND a + 1 AS d |> SET a = 2, b = 3 |> DROP a, t.b |> AS u \
        |> WHERE a > 1 |> TABLESAMPLE (5 ROWS) |> JOIN v ON u.a = v.a \
        |> LEFT SEMI JOIN w USING (a) |> UNION ALL TABLE x |> ORDER BY a LIMIT 3 \
        |> AGGREGATE sum(a) AS s GROUP BY b AS g |> PIVOT (sum(a) FOR b IN (1)) \
        |> UNPIVOT (v FOR n IN (a, b))";
    let ops = pipe_ops(sql);
    assert_eq!(ops.len(), 14);
    let K::Select(s) = &ops[0] else { panic!() };
    assert_eq!(
        (s.projection.len(), s.group_by.is_some(), s.windows.len()),
        (2, true, 1)
    );
    assert!(matches!(&ops[1], K::Extend(items) if items.len() == 1));
    assert!(matches!(&ops[2], K::Set(a) if a.len() == 2 && a[0].key.len() == 1));
    assert!(matches!(&ops[3], K::Drop(names) if names[1].parts.len() == 2));
    assert!(matches!(&ops[4], K::As(name) if name.value == "u"));
    assert!(matches!(&ops[5], K::Where(_)) && matches!(&ops[6], K::Sample(_)));
    assert!(matches!(&ops[7], K::Join(j) if matches!(j.constraint, ast::JoinConstraint::On(_))));
    assert!(matches!(&ops[8], K::Join(j) if j.kind == ast::JoinKind::LeftSemi));
    assert!(matches!(
        &ops[9],
        K::SetOp {
            op: ast::SetOperator::Union,
            quantifier: Some(ast::SetQuantifier::All),
            ..
        }
    ));
    let K::Organization(o) = &ops[10] else {
        panic!()
    };
    assert!(o.limit.is_some() && o.order_by.len() == 1);
    let K::Aggregate(a) = &ops[11] else { panic!() };
    let items = &a.group_by.as_ref().unwrap().items;
    assert!(matches!(items[0].kind, ast::GroupItemKind::Named(_)) && a.aggregates.len() == 1);
    assert!(matches!(&ops[12], K::Pivot(_, None)) && matches!(&ops[13], K::Unpivot(_, None)));
}

#[test]
fn pipes_bind_below_set_operators() {
    // `a UNION b |> x` pipes the union; `a |> x UNION b` unions the pipe.
    let q = ok("TABLE a UNION TABLE b |> WHERE x");
    let SetExprKind::Pipe(p) = &q.body.kind else {
        panic!()
    };
    assert!(matches!(p.input.kind, SetExprKind::SetOp { .. }));
    let q = ok("TABLE a |> WHERE x UNION TABLE b");
    let SetExprKind::SetOp { left, .. } = &q.body.kind else {
        panic!()
    };
    assert!(matches!(left.kind, SetExprKind::Pipe(_)));
    // A parenthesized pipeline is an operand; trailing clauses organize
    // the whole query; a `WITH` clause scopes over the pipeline.
    let q = ok("WITH c AS (TABLE t |> WHERE x) (TABLE c |> LIMIT 1) |> WHERE y ORDER BY z");
    assert!(q.with.is_some() && q.order_by.len() == 1);
    let SetExprKind::Pipe(p) = &q.body.kind else {
        panic!()
    };
    assert!(matches!(p.input.kind, SetExprKind::Query(_)));
    // `FROM` statements and `VALUES` start pipelines too.
    pipe_ops("FROM t |> SELECT a");
    pipe_ops("VALUES (1) AS t(a) |> WHERE a > 0");
    // The union's right side is a query primary, not a pipeline.
    let q = ok("TABLE a |> UNION (TABLE b |> WHERE x) |> LIMIT 1");
    assert_eq!(
        pipe_ops("TABLE a |> UNION (TABLE b |> WHERE x) |> LIMIT 1").len(),
        2
    );
    drop(q);
}

#[test]
fn single_character_pipe_and_bitwise_or() {
    use ast::PipeOpKind as K;
    assert_eq!(
        pipe_ops("TABLE t | SELECT x |> WHERE x | ORDER BY x").len(),
        3
    );
    assert!(matches!(
        &pipe_ops("TABLE t | LEFT JOIN u ON a")[0],
        K::Join(_)
    ));
    // `|` before one of the operator words is never a bitwise or, ...
    assert!(matches!(
        first_expr("SELECT a | b").kind,
        ExprKind::Binary(_, BinaryOp::BitOr, _)
    ));
    assert_eq!(pipe_ops("SELECT 1 | EXTEND y").len(), 1);
    // ... unless the option is off, where it is one.
    let off = ParserOptions::from_conf(|k| {
        (k == "spark.sql.parser.singleCharacterPipeOperator.enabled").then_some("false")
    });
    let q = parse_query("SELECT 1 | extend y", &off).unwrap();
    let SetExprKind::Select(s) = q.body.kind else {
        panic!()
    };
    assert!(matches!(
        s.projection[0].expr.kind,
        ExprKind::Binary(_, BinaryOp::BitOr, _)
    ));
    assert!(parse_query("TABLE t | SELECT x", &off).is_err());
    assert!(parse_query("TABLE t |> SELECT x", &off).is_ok());
    assert!(ParserOptions::default().single_character_pipe_operator);
}

#[test]
fn pipe_syntax_errors() {
    for sql in [
        "FROM t |> FROM t",
        "TABLE t |> EXTEND DISTINCT x AS z",
        "TABLE t |> SET x = 1 AS z",
        "TABLE t |> AS u, v",
        "TABLE t |> AS 1 + 2",
        "TABLE t |> AS u@v",
        "TABLE t |> PIVOT (sum(a) FOR b IN (1)) PIVOT (sum(a) FOR b IN (1))",
        "TABLE t |> INNER JOIN u INNER JOIN v",
        "TABLE t jt |> CROSS JOIN (SELECT * FROM jt)",
        "TABLE t |> ORDER BY x DESC ORDER BY x + y ORDER BY y",
        "SELECT 1 AS x |> AGGREGATE GROUP BY ()",
        "TABLE t |> GROUP BY a",
        "TABLE t |> SELECT",
        "TABLE t |> WHERE",
        "TABLE t |> LATERAL",
        "TABLE t |>",
        "|> SELECT 1",
    ] {
        let sql = if sql == "TABLE t |>" {
            "TABLE t |> )"
        } else {
            sql
        };
        assert_eq!(class(sql), "PARSE_SYNTAX_ERROR", "{sql}");
    }
    assert_eq!(class("TABLE t |> AS u-v"), "INVALID_IDENTIFIER");
}

#[test]
fn pipe_operator_checks() {
    for (sql, expected) in [
        ("TABLE t |> SET a.b = 1", "_LEGACY_ERROR_TEMP_0035"),
        ("TABLE t |> AGGREGATE", "_LEGACY_ERROR_TEMP_0035"),
        (
            "TABLE t |> WHERE sum(x) OVER w = 1 WINDOW w AS (PARTITION BY y)",
            "NOT_ALLOWED_IN_PIPE_OPERATOR_WHERE.WINDOW_CLAUSE",
        ),
        (
            "TABLE t |> PIVOT (sum(a) FOR b IN (1)) UNPIVOT (v FOR n IN (a))",
            "NOT_ALLOWED_IN_FROM.UNPIVOT_WITH_PIVOT",
        ),
        (
            "TABLE t |> UNPIVOT (v FOR n IN (a)) PIVOT (sum(a) FOR b IN (1))",
            "NOT_ALLOWED_IN_FROM.UNPIVOT_WITH_PIVOT",
        ),
        ("TABLE t |> TABLESAMPLE ()", "_LEGACY_ERROR_TEMP_0014"),
        (
            "TABLE t |> TABLESAMPLE (101 PERCENT)",
            "_LEGACY_ERROR_TEMP_0064",
        ),
        (
            "TABLE t |> ORDER BY x LIMIT 1",
            "MULTIPLE_QUERY_RESULT_CLAUSES_WITH_PIPE_OPERATORS",
        ),
        ("TABLE t |> OFFSET 1 LIMIT 2", ""),
        (
            "TABLE t |> ORDER BY x OFFSET 1",
            "MULTIPLE_QUERY_RESULT_CLAUSES_WITH_PIPE_OPERATORS",
        ),
        ("TABLE t |> LIMIT ALL", ""),
        (
            "TABLE t |> ORDER BY x SORT BY x",
            "UNSUPPORTED_FEATURE.COMBINATION_QUERY_RESULT_CLAUSES",
        ),
        (
            "TABLE t |> WINDOW w AS (ORDER BY x)",
            "UNSUPPORTED_FEATURE.CLAUSE_WITH_PIPE_OPERATORS",
        ),
        (
            "TABLE t |> AGGREGATE GROUP BY ALL",
            "UNSUPPORTED_FEATURE.PIPE_OPERATOR_AGGREGATE_UNSUPPORTED_CASE",
        ),
        (
            "TABLE t |> AGGREGATE sum(a) GROUP BY ROLLUP (a)",
            "UNSUPPORTED_FEATURE.PIPE_OPERATOR_AGGREGATE_UNSUPPORTED_CASE",
        ),
        (
            "TABLE t |> AGGREGATE sum(a) GROUP BY a WITH CUBE",
            "UNSUPPORTED_FEATURE.PIPE_OPERATOR_AGGREGATE_UNSUPPORTED_CASE",
        ),
        (
            "TABLE t |> AGGREGATE sum(a) GROUP BY a GROUPING SETS (a)",
            "UNSUPPORTED_FEATURE.PIPE_OPERATOR_AGGREGATE_UNSUPPORTED_CASE",
        ),
        (
            "TABLE t |> AGGREGATE grouping(a) + 1 GROUP BY a",
            "UNSUPPORTED_FEATURE.PIPE_OPERATOR_AGGREGATE_UNSUPPORTED_CASE",
        ),
        (
            "TABLE t |> AGGREGATE sum(a) OVER () GROUP BY b",
            "UNSUPPORTED_FEATURE.PIPE_OPERATOR_AGGREGATE_UNSUPPORTED_CASE",
        ),
        // Without `GROUP BY`, an aggregate with `OVER` is left to analysis.
        ("TABLE t |> AGGREGATE sum(a) OVER ()", ""),
        // Plain queries share the combination check.
        (
            "TABLE t ORDER BY x SORT BY y",
            "UNSUPPORTED_FEATURE.COMBINATION_QUERY_RESULT_CLAUSES",
        ),
        ("TABLE t DISTRIBUTE BY y SORT BY x", ""),
    ] {
        match parse_query(sql, &defaults()) {
            Ok(_) => assert_eq!(expected, "", "{sql}"),
            Err(e) => assert_eq!(e.error.error_class().name(), expected, "{sql}"),
        }
    }
}

#[test]
fn pipelines_lower_as_unsupported_and_cap_their_depth() {
    assert!(unsupported("TABLE t |> WHERE a").contains("pipe operators"));
    // Operators of one pipeline are a list: a long chain costs no depth.
    let chain = format!("TABLE t{}", " |> WHERE a".repeat(5000));
    assert_eq!(pipe_ops(&chain).len(), 5000);
    // Nesting is capped like any other: the deepest statement parses and
    // one level more fails at the whole statement.
    let deep = |n: usize| {
        format!(
            "TABLE t |> UNION {}TABLE t{}",
            "(TABLE t |> UNION ".repeat(n),
            ")".repeat(n)
        )
    };
    let (at, over) = (deep(499), deep(500));
    let len = over.len();
    let (a, b) = super::on_small_stack(move || {
        (
            parse_query(&at, &defaults()).map(drop),
            parse_query(&over, &defaults()),
        )
    });
    assert_eq!(a, Ok(()));
    let e = b.unwrap_err();
    assert_eq!(e.error.error_class().name(), "FAILED_TO_PARSE_TOO_COMPLEX");
    assert_eq!(e.span.range(), 0..len);
}

#[test]
fn pipe_trees_are_snapshotted() {
    insta::assert_snapshot!(super::snapshot(&["TABLE t |> WHERE a | LIMIT 2"]));
}
