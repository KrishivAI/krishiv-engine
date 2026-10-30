// SPDX-License-Identifier: Apache-2.0
//! `CREATE FUNCTION` and `DROP FUNCTION` statements: every form, the
//! errors Spark's parser raises (the class, parameters and position), and
//! the depth cap on the new paths. Statements marked `golden` are from
//! Spark 4.2.0's `sql-udf.sql`.

use krishiv_common::error::ErrorClass;

use super::super::ast::{
    CreateFunction, DropFunction, FunctionBodyKind, FunctionReturnsKind, SqlDataAccess,
    SqlSecurity, Statement,
};
use super::super::{MAX_DEPTH, ParserOptions, parse_statement};

fn create(sql: &str) -> CreateFunction {
    match parse_statement(sql, &ParserOptions::default()) {
        Ok(Statement::CreateFunction(f)) => *f,
        other => panic!("{sql}: {other:?}"),
    }
}

fn drop_fn(sql: &str) -> DropFunction {
    match parse_statement(sql, &ParserOptions::default()) {
        Ok(Statement::DropFunction(f)) => f,
        other => panic!("{sql}: {other:?}"),
    }
}

fn fail(sql: &str) -> super::super::ParseError {
    match parse_statement(sql, &ParserOptions::default()) {
        Ok(s) => panic!("{sql} parsed: {s:?}"),
        Err(e) => e,
    }
}

#[test]
fn scalar_function_without_parameters() {
    let sql = "CREATE FUNCTION foo1a0() RETURNS INT RETURN 1";
    let f = create(sql);
    assert!(!f.or_replace && !f.temporary && !f.if_not_exists);
    assert_eq!(f.name.parts.len(), 1);
    assert_eq!(f.name.parts[0].value, "foo1a0");
    assert!(f.params.is_empty());
    assert_eq!(f.params_span.range(), 23..23);
    let returns = f.returns.as_ref().unwrap();
    assert!(matches!(returns.kind, FunctionReturnsKind::Type(_)));
    assert_eq!(returns.span.text(sql), "RETURNS INT");
    assert!(matches!(f.body.kind, FunctionBodyKind::Expr(_)));
    assert_eq!(f.body.span.text(sql), "1");
    assert_eq!(f.span.text(sql), sql);
    // Equality ignores positions: the same statement spaced out is equal.
    assert_eq!(
        create("CREATE  FUNCTION  foo1a0 ( )  RETURNS  INT  RETURN  1 ;"),
        f
    );
}

#[test]
fn temporary_or_replace_and_if_not_exists() {
    let f = create("CREATE OR REPLACE TEMPORARY FUNCTION f(a INT) RETURNS INT RETURN a");
    assert!(f.or_replace && f.temporary && !f.if_not_exists);
    let f = create("create temp function f() return 1");
    assert!(!f.or_replace && f.temporary);
    assert!(f.returns.is_none());
    let f = create("CREATE OR REPLACE FUNCTION f() RETURN 1");
    assert!(f.or_replace && !f.temporary);
    let f = create("CREATE FUNCTION IF NOT EXISTS db.f() RETURNS INT RETURN 1");
    assert!(f.if_not_exists && !f.temporary);
    assert_eq!(f.name.parts.len(), 2);
}

#[test]
fn parameters_defaults_and_comments() {
    // golden: foo1d2
    let sql = "CREATE OR REPLACE FUNCTION foo1d2(a INT, b INT DEFAULT 7, c INT DEFAULT 8, \
               d INT DEFAULT 9 COMMENT 'test') RETURNS STRING RETURN a || ' ' || b";
    let f = create(sql);
    assert_eq!(f.params.len(), 4);
    assert!(f.params[0].default.is_none());
    assert!(f.params[1].default.is_some());
    assert_eq!(f.params[3].comment.as_deref(), Some("test"));
    assert_eq!(
        f.params_span.text(sql),
        "a INT, b INT DEFAULT 7, c INT DEFAULT 8, d INT DEFAULT 9 COMMENT 'test'"
    );
    // A default may be any expression; a parameter name may be quoted.
    let f = create("CREATE FUNCTION f(` a` INT DEFAULT length('x') + 1, `a b` STRING) RETURN 1");
    assert_eq!(f.params.len(), 2);
    // NOT NULL parses (Spark rejects it when it analyzes the statement).
    let f = create("CREATE FUNCTION foo1e1(x INT NOT NULL, y INT) RETURNS INT RETURN 1");
    assert!(f.params[0].not_null);
}

#[test]
fn table_function_returns() {
    // golden: foo1b2, foo2a2
    let f = create(
        "CREATE FUNCTION f(a INT) RETURNS TABLE (c1 INT, c2 STRING COMMENT 'two') RETURN SELECT 1, 'a'",
    );
    let Some(FunctionReturnsKind::Table(Some(columns))) = f.returns.map(|r| r.kind) else {
        panic!("not a table function");
    };
    assert_eq!(columns.len(), 2);
    assert_eq!(columns[1].name.value, "c2");
    assert_eq!(columns[1].comment.as_deref(), Some("two"));
    assert!(matches!(f.body.kind, FunctionBodyKind::Query(_)));
    let f = create("CREATE FUNCTION f() RETURNS TABLE(x INT NOT NULL) RETURN SELECT 1");
    let Some(FunctionReturnsKind::Table(Some(columns))) = f.returns.map(|r| r.kind) else {
        panic!("not a table function");
    };
    assert!(columns[0].not_null);
    // golden: foo3_4e: no column list.
    let f = create("CREATE OR REPLACE FUNCTION f(x INT) RETURNS TABLE RETURN SELECT * FROM g(x)");
    assert!(matches!(
        f.returns.map(|r| r.kind),
        Some(FunctionReturnsKind::Table(None))
    ));
    for body in [
        "VALUES (1)",
        "WITH t AS (SELECT 1) SELECT * FROM t",
        "FROM t SELECT a",
    ] {
        let f = create(&format!(
            "CREATE FUNCTION f() RETURNS TABLE(a INT) RETURN {body}"
        ));
        assert!(matches!(f.body.kind, FunctionBodyKind::Query(_)), "{body}");
    }
}

#[test]
fn a_parenthesised_subquery_is_an_expression() {
    // golden: foo31, foo51
    let sql = "CREATE FUNCTION foo51() RETURNS INT RETURN (SELECT a FROM VALUES(1), (2) AS T(a))";
    let f = create(sql);
    assert!(matches!(f.body.kind, FunctionBodyKind::Expr(_)));
    assert_eq!(
        f.body.span.text(sql),
        "(SELECT a FROM VALUES(1), (2) AS T(a))"
    );
    // golden: foo41: a bare query is a query, whatever the return type.
    let f = create("CREATE FUNCTION foo41() RETURNS INT RETURN SELECT 1");
    assert!(matches!(f.body.kind, FunctionBodyKind::Query(_)));
}

#[test]
fn characteristics_in_any_order() {
    let sql = "CREATE FUNCTION f() RETURNS INT COMMENT 'c' LANGUAGE SQL NOT DETERMINISTIC \
               READS SQL DATA SQL SECURITY DEFINER RETURN 1";
    let c = create(sql).characteristics;
    assert_eq!(c.comment.as_deref(), Some("c"));
    assert_eq!(c.language.as_ref().map(|l| l.value.as_str()), Some("SQL"));
    assert_eq!(c.deterministic, Some(false));
    assert_eq!(c.data_access, Some(SqlDataAccess::ReadsSqlData));
    assert_eq!(c.security, Some(SqlSecurity::Definer));
    assert!(c.duplicate.is_none());
    assert_eq!(
        c.span.text(sql),
        "COMMENT 'c' LANGUAGE SQL NOT DETERMINISTIC READS SQL DATA SQL SECURITY DEFINER"
    );
    let c = create("CREATE FUNCTION f() SQL SECURITY INVOKER CONTAINS SQL DETERMINISTIC LANGUAGE python RETURN 1")
        .characteristics;
    assert_eq!(c.security, Some(SqlSecurity::Invoker));
    assert_eq!(c.data_access, Some(SqlDataAccess::ContainsSql));
    assert_eq!(c.deterministic, Some(true));
    assert_eq!(
        c.language.as_ref().map(|l| l.value.as_str()),
        Some("python")
    );
    // None written: an empty span.
    let c = create("CREATE FUNCTION f() RETURN 1").characteristics;
    assert_eq!(c, super::super::ast::RoutineCharacteristics::default());
    assert_eq!(c.span.range(), 20..20);
}

#[test]
fn a_repeated_characteristic_is_duplicate_clauses() {
    for (clause, sql) in [
        ("COMMENT", "COMMENT 'a' COMMENT 'b'"),
        ("LANGUAGE", "LANGUAGE SQL LANGUAGE SQL"),
        ("DETERMINISTIC", "DETERMINISTIC NOT DETERMINISTIC"),
        ("SQL DATA ACCESS", "CONTAINS SQL READS SQL DATA"),
        (
            "SQL SECURITY RIGHTS",
            "SQL SECURITY INVOKER SQL SECURITY DEFINER",
        ),
    ] {
        let text = format!("CREATE FUNCTION f() RETURNS INT {sql} RETURN 1");
        let e = fail(&text);
        assert_eq!(
            e.error.error_class(),
            ErrorClass::DuplicateClauses,
            "{text}"
        );
        assert_eq!(
            e.error.message_parameters(),
            [("clauseName", clause.to_owned())]
        );
        assert_eq!(e.span.range(), 0..text.len());
    }
}

#[test]
fn temporary_function_rules() {
    let sql = "CREATE TEMPORARY FUNCTION db.f() RETURN 1";
    let e = fail(sql);
    assert_eq!(
        e.error.error_class(),
        ErrorClass::InvalidSqlSyntaxCreateTempFuncWithDatabase
    );
    assert_eq!(
        e.error.message_parameters(),
        [("database", "db".to_owned())]
    );
    assert_eq!(e.span.range(), 0..sql.len());
    let e = fail("CREATE TEMPORARY FUNCTION IF NOT EXISTS f() RETURN 1");
    assert_eq!(
        e.error.error_class(),
        ErrorClass::InvalidSqlSyntaxCreateTempFuncWithIfNotExists
    );
    let e = fail("CREATE OR REPLACE FUNCTION IF NOT EXISTS f() RETURN 1");
    assert_eq!(
        e.error.error_class(),
        ErrorClass::InvalidSqlSyntaxCreateRoutineWithIfNotExistsAndReplace
    );
}

#[test]
fn parameters_that_functions_do_not_take() {
    // golden: foo1f1, foo1f2, foo1g1, foo1g2: ParseExceptions at the
    // whole statement.
    for (class, sql) in [
        (
            ErrorClass::InvalidSqlSyntaxCreateFuncWithGeneratedColumnsAsParameters,
            "CREATE FUNCTION foo1f1(x INT, y INT GENERATED ALWAYS AS (x + 10)) RETURNS INT RETURN y + 1",
        ),
        (
            ErrorClass::InvalidSqlSyntaxCreateFuncWithGeneratedColumnsAsParameters,
            "CREATE FUNCTION foo1f2(id BIGINT GENERATED ALWAYS AS IDENTITY) RETURNS BIGINT RETURN id + 1",
        ),
        (
            ErrorClass::InvalidSqlSyntaxCreateFuncWithColumnConstraints,
            "CREATE FUNCTION foo1g1(x INT, y INT UNIQUE) RETURNS INT RETURN y + 1",
        ),
        (
            ErrorClass::InvalidSqlSyntaxCreateFuncWithColumnConstraints,
            "CREATE FUNCTION foo1g2(id BIGINT CHECK (true)) RETURNS BIGINT RETURN id + 1",
        ),
    ] {
        let e = fail(sql);
        assert_eq!(e.error.error_class(), class, "{sql}");
        assert_eq!(e.error.sql_state(), Some("42000"), "{sql}");
        assert_eq!(e.span.range(), 0..sql.len(), "{sql}");
    }
}

#[test]
fn syntax_errors_match_spark() {
    // golden: foo2a0: an empty `RETURNS TABLE ()` fails at the `)`.
    let sql = "CREATE FUNCTION foo2a0() RETURNS TABLE() RETURN SELECT 1";
    let e = fail(sql);
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(
        e.error.message_parameters(),
        [("error", "')'".to_owned()), ("hint", String::new())]
    );
    assert_eq!(e.span.text(sql), ")");
    // golden: foo2c1: a returned column takes no DEFAULT.
    let sql = "CREATE FUNCTION foo2c1() RETURNS TABLE(c1 INT DEFAULT 5) RETURN SELECT 1, 2";
    let e = fail(sql);
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(
        e.error.message_parameters(),
        [("error", "'DEFAULT'".to_owned()), ("hint", String::new())]
    );
    assert_eq!(e.span.text(sql), "DEFAULT");
    // `RETURN` is required, and so is a body.
    let sql = "CREATE FUNCTION f() RETURNS INT 1";
    assert_eq!(fail(sql).span.text(sql), "1");
    let sql = "CREATE FUNCTION f() RETURNS INT RETURN";
    let e = fail(sql);
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), sql.len()..sql.len());
    // The parameter list is required.
    let sql = "CREATE FUNCTION f RETURN 1";
    assert_eq!(fail(sql).span.text(sql), "RETURN");
    // A parameter needs a type; a type needs to be complete.
    let sql = "CREATE FUNCTION f(a) RETURN 1";
    assert_eq!(fail(sql).span.text(sql), ")");
    let sql = "CREATE FUNCTION f(a INT,) RETURN 1";
    assert_eq!(fail(sql).span.text(sql), ")");
    // A characteristic's argument is required.
    for text in [
        "CREATE FUNCTION f() CONTAINS RETURN 1",
        "CREATE FUNCTION f() READS SQL RETURN 1",
        "CREATE FUNCTION f() SQL SECURITY RETURN 1",
        "CREATE FUNCTION f() COMMENT RETURN 1",
    ] {
        let e = fail(text);
        assert_eq!(
            e.error.error_class(),
            ErrorClass::ParseSyntaxError,
            "{text}"
        );
        assert_eq!(e.span.text(text), "RETURN", "{text}");
    }
    // `RETURN` after `LANGUAGE` is the language's name.
    let text = "CREATE FUNCTION f() LANGUAGE RETURN 1";
    assert_eq!(fail(text).span.text(text), "1");
    let text = "CREATE FUNCTION f() NOT RETURN 1";
    assert_eq!(fail(text).span.text(text), "NOT");
    // Anything after the statement but `;`.
    let e = fail("CREATE FUNCTION f() RETURN 1 2");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // The Hive form (`AS 'class'`) is not a SQL function.
    let sql = "CREATE FUNCTION f AS 'com.example.F'";
    assert_eq!(fail(sql).span.text(sql), "AS");
}

#[test]
fn post_parse_checks_reach_every_part() {
    // A type, a default, a returned column and the body are checked as in
    // any other statement.
    for sql in [
        "CREATE FUNCTION f(a DECIMAL(40, 2)) RETURN 1",
        "CREATE FUNCTION f(a INT DEFAULT DATE '2020-02-30') RETURN 1",
        "CREATE FUNCTION f() RETURNS DECIMAL(40, 2) RETURN 1",
        "CREATE FUNCTION f() RETURNS TABLE(a DECIMAL(40, 2)) RETURN SELECT 1",
        "CREATE FUNCTION f() RETURN DATE '2020-02-30'",
        "CREATE FUNCTION f() RETURNS TABLE(a INT) RETURN SELECT DATE '2020-02-30'",
    ] {
        let e = fail(sql);
        assert!(
            matches!(
                e.error.error_class(),
                ErrorClass::DecimalPrecisionExceedsMaxPrecision | ErrorClass::InvalidTypedLiteral
            ),
            "{sql}: {:?}",
            e.error.error_class()
        );
    }
}

#[test]
fn drop_function_forms() {
    let f = drop_fn("DROP FUNCTION f");
    assert!(!f.temporary && !f.if_exists);
    assert_eq!(f.name.parts[0].value, "f");
    let sql = "drop temporary function if exists `my fn`;";
    let f = drop_fn(sql);
    assert!(f.temporary && f.if_exists);
    assert_eq!(f.name.parts[0].value, "my fn");
    assert_eq!(
        f.span.text(sql),
        "drop temporary function if exists `my fn`"
    );
    let f = drop_fn("DROP TEMP FUNCTION a.b");
    assert!(f.temporary);
    assert_eq!(f.name.parts.len(), 2);
    // The name is required; other things fail where they stand.
    let sql = "DROP TEMPORARY FUNCTION";
    assert_eq!(fail(sql).span.range(), sql.len()..sql.len());
    let sql = "DROP FUNCTION IF EXISTS";
    assert_eq!(fail(sql).span.range(), sql.len()..sql.len());
    let sql = "DROP TEMPORARY VIEW v";
    assert_eq!(fail(sql).span.text(sql), "VIEW");
    let sql = "DROP FUNCTIONS f";
    assert_eq!(fail(sql).span.text(sql), "FUNCTIONS");
}

#[test]
fn explain_takes_function_statements() {
    let s = parse_statement(
        "EXPLAIN CREATE FUNCTION f() RETURN 1",
        &ParserOptions::default(),
    )
    .unwrap();
    let Statement::Explain(e) = s else {
        panic!("not an explain");
    };
    assert!(matches!(*e.statement, Statement::CreateFunction(_)));
    let s = parse_statement("EXPLAIN DROP FUNCTION f", &ParserOptions::default()).unwrap();
    let Statement::Explain(e) = s else {
        panic!("not an explain");
    };
    assert!(matches!(*e.statement, Statement::DropFunction(_)));
}

/// The depth cap holds on the new paths: the deepest statement within
/// [`MAX_DEPTH`] levels parses and one more fails with
/// `FAILED_TO_PARSE_TOO_COMPLEX` at the whole statement. The command takes
/// a level, and the body's expression or query the rest.
#[test]
fn depth_limit_is_exact_in_bodies_defaults_and_columns() {
    let shapes: [(&str, &str, &str, &str, &str, usize); 5] = [
        (
            "CREATE FUNCTION f() RETURN ",
            "(",
            "1",
            ")",
            "",
            MAX_DEPTH - 2,
        ),
        (
            "CREATE FUNCTION f(a INT DEFAULT ",
            "(",
            "1",
            ")",
            ") RETURN 1",
            MAX_DEPTH - 2,
        ),
        (
            "CREATE FUNCTION f() RETURNS TABLE(a INT) RETURN SELECT * FROM ",
            "(",
            "t",
            ")",
            "",
            MAX_DEPTH - 3,
        ),
        (
            "CREATE FUNCTION f() RETURNS ",
            "ARRAY<",
            "INT",
            ">",
            " RETURN 1",
            MAX_DEPTH - 2,
        ),
        (
            "CREATE FUNCTION f() RETURNS TABLE(a ",
            "ARRAY<",
            "INT",
            ">",
            ") RETURN SELECT 1",
            MAX_DEPTH - 2,
        ),
    ];
    for (pre, open, mid, close, post, limit) in shapes {
        let deep = move |n: usize| format!("{pre}{}{mid}{}{post}", open.repeat(n), close.repeat(n));
        let (at, over) = (deep(limit), deep(limit + 1));
        let len = over.len();
        let (a, b) = super::on_small_stack(move || {
            let d = ParserOptions::default();
            (
                parse_statement(&at, &d).map(drop),
                parse_statement(&over, &d),
            )
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
}

/// Every prefix and every single-token deletion of function statements
/// parses or fails with an error whose span lies in the input: no panic,
/// whatever the cut.
#[test]
fn truncated_and_mutilated_statements_never_panic() {
    let statements = [
        "CREATE OR REPLACE TEMPORARY FUNCTION f(a INT DEFAULT 1 COMMENT 'x', b STRING) \
         RETURNS TABLE (c INT NOT NULL COMMENT 'c', d ARRAY<INT>) COMMENT 'f' LANGUAGE SQL \
         NOT DETERMINISTIC READS SQL DATA SQL SECURITY INVOKER RETURN SELECT (SELECT a), b",
        "CREATE FUNCTION IF NOT EXISTS db.f(`a b` MAP<STRING, INT>) RETURNS INT CONTAINS SQL \
         RETURN CASE WHEN a THEN 1 END",
        "DROP TEMPORARY FUNCTION IF EXISTS a.`b c`",
    ];
    let options = ParserOptions::default();
    let check = |sql: &str| {
        if let Err(e) = parse_statement(sql, &options) {
            assert!(
                e.span.start <= e.span.end && e.span.end <= sql.len(),
                "{sql}: {:?}",
                e.span
            );
        }
    };
    for sql in statements {
        for end in (0..=sql.len()).filter(|&i| sql.is_char_boundary(i)) {
            check(&sql[..end]);
        }
        let words: Vec<&str> = sql.split(' ').collect();
        for skip in 0..words.len() {
            let mut kept = words.clone();
            kept.remove(skip);
            check(&kept.join(" "));
        }
    }
}
