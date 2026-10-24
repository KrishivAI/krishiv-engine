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
