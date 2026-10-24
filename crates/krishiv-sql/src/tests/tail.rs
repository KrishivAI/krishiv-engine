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
