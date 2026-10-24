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

fn first_join(sql: &str) -> ast::Join {
    let q = ok(sql);
    let SetExprKind::Select(s) = q.body.kind else {
        panic!()
    };
    let rel = s.from.unwrap().relations.into_iter().next().unwrap();
    match rel.extensions.into_iter().next() {
        Some(ast::RelationExt::Join(j)) => j,
        other => panic!("{sql}: {other:?}"),
    }
}

#[test]
fn nearest_by_joins() {
    use ast::{JoinConstraint, NearestDirection};
    let sql = "SELECT * FROM u JOIN p APPROX NEAREST 2 BY DISTANCE abs(u.s - p.s) WHERE true";
    let j = first_join(sql);
    let JoinConstraint::NearestBy(n) = &j.constraint else {
        panic!("{j:?}")
    };
    assert!(n.approx && n.num_results.as_deref() == Some("2"));
    assert_eq!(n.direction, NearestDirection::Distance);
    assert_eq!(n.ranking.span.text(sql), "abs(u.s - p.s)");
    assert!(j.right.alias.is_none(), "APPROX is no alias");
    let j = first_join("SELECT * FROM u LEFT OUTER JOIN p q EXACT NEAREST BY SIMILARITY -abs(x)");
    let JoinConstraint::NearestBy(n) = &j.constraint else {
        panic!()
    };
    assert!(!n.approx && n.num_results.is_none() && j.right.alias.is_some());
    assert_eq!(n.direction, NearestDirection::Similarity);
    // `approx` and `exact` stay usable as names.
    ok("SELECT approx, exact FROM t approx JOIN u exact ON true");
    for sql in [
        "SELECT * FROM u JOIN p APPROX NEAREST BY x",
        "SELECT * FROM u JOIN p NEAREST BY DISTANCE x",
        "SELECT * FROM u NATURAL JOIN p APPROX NEAREST BY DISTANCE x",
        "SELECT * FROM u JOIN p APPROX NEAREST 1.5 BY DISTANCE x",
    ] {
        assert_eq!(class(sql), "PARSE_SYNTAX_ERROR", "{sql}");
    }
    for (join, expected) in [
        (
            "RIGHT OUTER JOIN p",
            "NEAREST_BY_JOIN.UNSUPPORTED_JOIN_TYPE",
        ),
        ("FULL JOIN p", "NEAREST_BY_JOIN.UNSUPPORTED_JOIN_TYPE"),
        ("LEFT SEMI JOIN p", "NEAREST_BY_JOIN.UNSUPPORTED_JOIN_TYPE"),
        (
            "JOIN LATERAL (SELECT 1) p",
            "UNSUPPORTED_FEATURE.LATERAL_JOIN_NEAREST_BY",
        ),
    ] {
        let sql = format!("SELECT * FROM u {join} APPROX NEAREST BY DISTANCE x");
        assert_eq!(class(&sql), expected, "{sql}");
    }
    for k in ["0", "100001", "99999999999999999999"] {
        let sql = format!("SELECT * FROM u JOIN p APPROX NEAREST {k} BY DISTANCE x");
        assert_eq!(
            class(&sql),
            "NEAREST_BY_JOIN.NUM_RESULTS_OUT_OF_RANGE",
            "{k}"
        );
    }
    parse_query(
        "SELECT * FROM u JOIN p EXACT NEAREST 100000 BY DISTANCE x",
        &defaults(),
    )
    .unwrap();
    parse_query(
        "SELECT * FROM u INNER JOIN p EXACT NEAREST 1 BY DISTANCE x",
        &defaults(),
    )
    .unwrap();
    assert!(
        unsupported("SELECT * FROM u JOIN p APPROX NEAREST BY DISTANCE x").contains("NEAREST BY")
    );
}

fn table_function_args(sql: &str) -> Vec<ast::FunctionArg> {
    match first_factor(sql).kind {
        ast::TableFactorKind::TableFunction { args, .. } => args,
        other => panic!("{sql}: {other:?}"),
    }
}

#[test]
fn table_arguments() {
    use ast::TableArgumentSource as S;
    let table_arg = |arg: &ast::FunctionArg| match &arg.expr.kind {
        ExprKind::TableArgument(t) => (**t).clone(),
        other => panic!("{other:?}"),
    };
    let args = table_function_args("SELECT * FROM f(TABLE(t), TABLE(SELECT 1), 3)");
    assert!(matches!(table_arg(&args[0]).source, S::Table(_)));
    assert!(matches!(table_arg(&args[1]).source, S::Query(_)));
    assert!(matches!(args[2].expr.kind, ExprKind::Literal(_)));
    let sql = "SELECT * FROM f(k => TABLE(a.b) PARTITION BY (x, y + 1) ORDER BY z DESC, w)";
    let args = table_function_args(sql);
    assert_eq!(args[0].name.as_ref().unwrap().value, "k");
    let p = table_arg(&args[0]).partitioning.unwrap();
    assert_eq!(
        (p.partition_by.len(), p.order_by.len(), p.single),
        (2, 2, false)
    );
    assert!(p.extra_partition.is_none() && !p.distribute && !p.sort);
    let sql =
        "SELECT * FROM f(TABLE(t) DISTRIBUTE BY x SORT BY y), g(TABLE(t) WITH SINGLE PARTITION)";
    let q = ok(sql);
    let SetExprKind::Select(s) = q.body.kind else {
        panic!()
    };
    let rels = s.from.unwrap().relations;
    let ast::TableFactorKind::TableFunction { args, .. } = &rels[0].primary.kind else {
        panic!()
    };
    let p = table_arg(&args[0]).partitioning.unwrap();
    assert!(p.distribute && p.sort && p.partition_by.len() == 1);
    let ast::TableFactorKind::TableFunction { args, .. } = &rels[1].primary.kind else {
        panic!()
    };
    assert!(table_arg(&args[0]).partitioning.unwrap().single);
    // A parenthesized prefix of a bigger expression is that expression.
    let args = table_function_args("SELECT * FROM f(TABLE(t) PARTITION BY (a + b) * 2)");
    let p = table_arg(&args[0]).partitioning.unwrap();
    assert!(matches!(p.partition_by[0].kind, ExprKind::Binary(..)));
    // Only the table-valued functions of a FROM clause take them; in an
    // expression `TABLE(t)` calls a function named `TABLE`, as Spark reads it.
    assert!(matches!(
        first_expr("SELECT f(TABLE(t))").kind,
        ExprKind::Function(_)
    ));
    for sql in ["SELECT * FROM f(TABLE(t)", "SELECT * FROM f(TABLE)"] {
        assert_eq!(class(sql), "PARSE_SYNTAX_ERROR", "{sql}");
    }
    for (sql, expected) in [
        (
            "SELECT * FROM f(TABLE t)",
            "INVALID_SQL_SYNTAX.INVALID_TABLE_FUNCTION_IDENTIFIER_ARGUMENT_MISSING_PARENTHESES",
        ),
        (
            "SELECT * FROM f(TABLE(t) PARTITION BY a, b)",
            "_LEGACY_ERROR_TEMP_0064",
        ),
        (
            "SELECT * FROM f(TABLE(t) PARTITION BY a ORDER BY b, c)",
            "_LEGACY_ERROR_TEMP_0064",
        ),
        (
            "SELECT * FROM f(TABLE(SELECT DATE 'x'))",
            "INVALID_TYPED_LITERAL",
        ),
    ] {
        let found = parse_query(sql, &defaults()).map(drop);
        assert_eq!(
            found.unwrap_err().error.error_class().name(),
            expected,
            "{sql}"
        );
    }
    assert!(unsupported("SELECT * FROM f(TABLE(t))").contains("table-valued functions"));
}

#[test]
fn identifier_clauses() {
    use ast::TableFactorKind as F;
    let table = |sql: &str| first_factor(sql).kind;
    assert!(matches!(
        table("SELECT * FROM IDENTIFIER('a' || 'b')"),
        F::DynamicTable { .. }
    ));
    assert!(matches!(
        table("SELECT * FROM IDENTIFIER('tab')"),
        F::DynamicTable { .. }
    ));
    for sql in [
        "SELECT * FROM IDENTIFIER('s').tab",
        "SELECT * FROM s.IDENTIFIER('tab')",
    ] {
        let F::Table { name, .. } = table(sql) else {
            panic!("{sql}")
        };
        assert_eq!(
            name.parts
                .iter()
                .map(|p| p.value.as_str())
                .collect::<Vec<_>>(),
            ["s", "tab"]
        );
    }
    let q = ok("TABLE IDENTIFIER('t')");
    assert!(matches!(q.body.kind, SetExprKind::DynamicTable(_)));
    // A table-valued function and a scalar function by computed name.
    let F::TableFunction {
        name, dynamic_name, ..
    } = table("SELECT * FROM IDENTIFIER('ra' || 'nge')(0, 1)")
    else {
        panic!()
    };
    assert!(dynamic_name.is_some() && name.parts[0].value == "IDENTIFIER");
    let ExprKind::Function(f) = first_expr("SELECT IDENTIFIER('abs')(c1)").kind else {
        panic!()
    };
    assert!(f.dynamic_name.is_some() && f.args.len() == 1);
    // Without a second argument list it is the function `identifier`.
    let ExprKind::Function(f) = first_expr("SELECT IDENTIFIER('c' || '1')").kind else {
        panic!()
    };
    assert!(f.dynamic_name.is_none() && f.name.parts[0].value == "IDENTIFIER" && f.args.len() == 1);
    let ExprKind::Field(inner, field) = first_expr("SELECT IDENTIFIER('t').c1").kind else {
        panic!()
    };
    assert!(matches!(inner.kind, ExprKind::Function(_)) && field.value == "c1");
    // An identifier literal names one part wherever an identifier fits.
    let sql = "SELECT 1 AS IDENTIFIER('`a b`') FROM t AS IDENTIFIER('x') (IDENTIFIER('y'))";
    let q = ok(sql);
    let SetExprKind::Select(s) = q.body.kind else {
        panic!()
    };
    let Some(ast::Alias::Name(alias)) = &s.projection[0].alias else {
        panic!()
    };
    assert_eq!((alias.value.as_str(), alias.quoted), ("a b", true));
    assert_eq!(alias.span.text(sql), "IDENTIFIER('`a b`')");
    let table_alias = s.from.unwrap().relations[0].primary.alias.clone().unwrap();
    assert_eq!(table_alias.columns[0].value, "y");
    for (sql, expected) in [
        (
            "SELECT 1 AS IDENTIFIER('a.b')",
            "IDENTIFIER_TOO_MANY_NAME_PARTS",
        ),
        (
            "SELECT * FROM VALUES (1) AS IDENTIFIER('s.t')",
            "IDENTIFIER_TOO_MANY_NAME_PARTS",
        ),
        ("SELECT IDENTIFIER('c 1')", "PARSE_SYNTAX_ERROR"),
        ("SELECT IDENTIFIER('')", "PARSE_EMPTY_STATEMENT"),
        (
            "SELECT IDENTIFIER((SELECT 'a', 'b'))",
            "UNSUPPORTED_TYPED_LITERAL",
        ),
    ] {
        let found = parse_query(sql, &defaults()).map(drop);
        assert_eq!(
            found.unwrap_err().error.error_class().name(),
            expected,
            "{sql}"
        );
    }
    parse_query(
        "SELECT IDENTIFIER('t.c1'), IDENTIFIER((SELECT 'c'))",
        &defaults(),
    )
    .unwrap();
    // Not where only a plain word fits: hints, `EXTRACT` fields.
    for sql in [
        "SELECT /*+ IDENTIFIER('BROADCAST')(t) */ * FROM t",
        "SELECT EXTRACT(IDENTIFIER('YEAR') FROM d)",
    ] {
        assert_eq!(class(sql), "PARSE_SYNTAX_ERROR", "{sql}");
    }
    assert!(unsupported("SELECT * FROM IDENTIFIER('t')").contains("IDENTIFIER"));
    assert!(unsupported("SELECT IDENTIFIER('abs')(c1) FROM t").contains("IDENTIFIER"));
}

#[test]
fn legacy_identifier_clause_takes_only_the_function_syntax() {
    let legacy =
        ParserOptions::from_conf(|k| (k == "spark.sql.legacy.identifierClause").then_some("true"));
    for sql in [
        "SELECT 1 AS IDENTIFIER('c')",
        "SELECT * FROM s.IDENTIFIER('tab')",
        "SELECT * FROM IDENTIFIER('s').IDENTIFIER('tab')",
        "SELECT * FROM IDENTIFIER('s').tab",
        "SELECT map('a', 1).IDENTIFIER('a')",
        "SELECT * FROM t JOIN u USING (IDENTIFIER('c'))",
    ] {
        assert!(parse_query(sql, &legacy).is_err(), "{sql}");
        if !sql.contains("map(") {
            assert!(parse_query(sql, &defaults()).is_ok(), "{sql}");
        }
    }
    for sql in [
        "SELECT * FROM IDENTIFIER('t')",
        "SELECT IDENTIFIER('abs')(c1)",
        "SELECT IDENTIFIER('c')",
    ] {
        parse_query(sql, &legacy).unwrap();
    }
}

#[test]
fn variant_paths() {
    use ast::VariantPathPartKind as K;
    let path = |sql: &str| {
        let ExprKind::VariantGet(_, p) = first_expr(sql).kind else {
            panic!("{sql}")
        };
        p.parts.into_iter().map(|p| p.kind).collect::<Vec<_>>()
    };
    let parts = path("SELECT v:a.b[0]['k'][c].d");
    assert!(matches!(&parts[0], K::Field(i) if i.value == "a"));
    assert!(matches!(&parts[1], K::Field(i) if i.value == "b"));
    assert!(matches!(&parts[2], K::Index(i) if i == "0"));
    assert!(matches!(&parts[3], K::Key(k) if k == "k"));
    assert!(matches!(&parts[4], K::Name(i) if i.value == "c"));
    assert!(matches!(&parts[5], K::Field(i) if i.value == "d"));
    assert!(matches!(&path("SELECT v:['name']")[0], K::Key(_)));
    assert!(matches!(&path("SELECT v:[3].id")[0], K::Index(_)));
    assert!(matches!(&path("SELECT v:`field-name`")[0], K::Field(i) if i.quoted));
    // A cast applies to the whole extraction; extractions chain.
    let ExprKind::Cast { expr, .. } = first_expr("SELECT v:a.b::DECIMAL(5, 2)").kind else {
        panic!()
    };
    assert!(matches!(expr.kind, ExprKind::VariantGet(..)));
    let ExprKind::VariantGet(inner, _) = first_expr("SELECT v:a:b").kind else {
        panic!()
    };
    assert!(matches!(inner.kind, ExprKind::VariantGet(..)));
    let sql = "SELECT v:a, v:b::INT + w:c[1] AS d FROM t";
    let q = ok(sql);
    let SetExprKind::Select(s) = q.body.kind else {
        panic!()
    };
    assert_eq!(s.projection[1].expr.span.text(sql), "v:b::INT + w:c[1]");
    for sql in [
        "SELECT v:*",
        "SELECT v:",
        "SELECT v:a.",
        "SELECT v:[a]",
        "SELECT v:['a'",
        "SELECT v:[1.5]",
        "SELECT v:a[]",
    ] {
        assert_eq!(class(sql), "PARSE_SYNTAX_ERROR", "{sql}");
    }
    // Colons of struct types are still types.
    ok("SELECT CAST(x AS STRUCT<a: INT, b:STRING>)");
    assert!(unsupported("SELECT v:a FROM t").contains("variant paths"));
}

fn transform_of(sql: &str) -> ast::TransformSelect {
    let q = ok(sql);
    match q.body.kind {
        SetExprKind::Transform(t) => *t,
        other => panic!("{sql}: {other:?}"),
    }
}

#[test]
fn transform_clauses() {
    use ast::{TransformKind, TransformOutput as O};
    let sql = "SELECT TRANSFORM(a, b + 1) USING 'cat' AS (x, y) FROM t WHERE a > 1";
    let t = transform_of(sql);
    assert_eq!(
        (t.kind, t.script.as_str()),
        (TransformKind::SelectTransform, "cat")
    );
    assert_eq!(t.select.projection.len(), 2);
    assert!(t.select.selection.is_some() && t.select.from.is_some());
    assert!(matches!(&t.output, Some(O::Names(n)) if n.len() == 2));
    assert_eq!(t.span.text(sql), sql);
    let t = transform_of(
        "SELECT TRANSFORM(a) USING 'cat' AS (x int, y array<string> NOT NULL COMMENT 'c') FROM t",
    );
    let Some(O::Columns(columns)) = &t.output else {
        panic!()
    };
    assert!(columns.len() == 2 && columns[1].not_null && columns[1].comment.is_some());
    let t = transform_of("SELECT TRANSFORM(a) USING 'cat' AS x, y FROM t GROUP BY a HAVING true");
    assert!(matches!(&t.output, Some(O::Names(n)) if n.len() == 2));
    assert!(t.select.group_by.is_some() && t.select.having.is_some());
    let t = transform_of(
        "SELECT TRANSFORM(a) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' NULL DEFINED AS 'x' \
         RECORDWRITER 'w' USING 'cat' AS (a) ROW FORMAT DELIMITED LINES TERMINATED BY '\\n' \
         RECORDREADER 'r' FROM t",
    );
    assert!(t.input_format.is_some() && t.output_format.is_some());
    assert_eq!(
        (t.record_writer.as_deref(), t.record_reader.as_deref()),
        (Some("w"), Some("r"))
    );
    assert!(
        transform_of("SELECT TRANSFORM(a) USING 'cat'")
            .output
            .is_none()
    );
    assert_eq!(
        transform_of("MAP a, b USING 'cat' AS (a, b) FROM t").kind,
        TransformKind::Map
    );
    assert_eq!(
        transform_of("REDUCE a USING 'cat' FROM t").kind,
        TransformKind::Reduce
    );
    // After `FROM`, in a body of the statement.
    let q = ok("FROM t SELECT TRANSFORM(a) USING 'cat' WHERE a > 1 SELECT b");
    let SetExprKind::FromStatement(f) = q.body.kind else {
        panic!()
    };
    assert!(matches!(f.bodies[0].body.kind, SetExprKind::Transform(_)));
    assert!(matches!(f.bodies[1].body.kind, SetExprKind::Select(_)));
    // Without `USING` and friends it is the function `transform`.
    for sql in [
        "SELECT TRANSFORM(a) FROM t",
        "SELECT TRANSFORM(arr, x -> x + 1) FROM t",
        "SELECT TRANSFORM(a)",
    ] {
        ok(sql);
        assert!(matches!(ok(sql).body.kind, SetExprKind::Select(_)), "{sql}");
    }
    assert!(unsupported("SELECT TRANSFORM(a) USING 'cat' FROM t").contains("TRANSFORM"));
}

#[test]
fn transform_errors() {
    for sql in [
        "SELECT TRANSFORM(b AS c) USING 'cat'",
        "SELECT TRANSFORM(b c) USING 'cat'",
        "SELECT TRANSFORM(a) ROW FORMAT DELIMITED FROM t",
        "SELECT TRANSFORM(a) USING cat",
        "SELECT TRANSFORM(a) USING 'cat' AS ()",
        "SELECT TRANSFORM(a) USING 'cat' AS (x, y int)",
        "SELECT TRANSFORM() USING 'cat'",
        "SELECT TRANSFORM(a) USING 'cat' QUALIFY true",
    ] {
        assert_eq!(class(sql), "PARSE_SYNTAX_ERROR", "{sql}");
    }
    for (sql, expected) in [
        (
            "SELECT TRANSFORM(DISTINCT a) USING 'cat'",
            "UNSUPPORTED_FEATURE.TRANSFORM_DISTINCT_ALL",
        ),
        (
            "SELECT TRANSFORM(ALL a) USING 'cat'",
            "UNSUPPORTED_FEATURE.TRANSFORM_DISTINCT_ALL",
        ),
        (
            "MAP DISTINCT a USING 'cat'",
            "UNSUPPORTED_FEATURE.TRANSFORM_DISTINCT_ALL",
        ),
        (
            "SELECT TRANSFORM(a) ROW FORMAT SERDE 'x.Y' USING 'cat'",
            "UNSUPPORTED_FEATURE.TRANSFORM_NON_HIVE",
        ),
        (
            "SELECT TRANSFORM(a) USING 'cat' AS (a) ROW FORMAT SERDE 'x.Y'",
            "UNSUPPORTED_FEATURE.TRANSFORM_NON_HIVE",
        ),
        (
            "SELECT TRANSFORM(a) ROW FORMAT DELIMITED LINES TERMINATED BY '@' USING 'cat'",
            "_LEGACY_ERROR_TEMP_0064",
        ),
        (
            "SELECT TRANSFORM(a) USING 'cat' AS (a nosuchtype)",
            "UNSUPPORTED_DATATYPE",
        ),
    ] {
        assert_eq!(class(sql), expected, "{sql}");
    }
}
