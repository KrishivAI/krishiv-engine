// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

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

/// `SET TIME ZONE` as Spark 4.2's `visitSetTimeZone` reads it: an interval within [-18, 18] hours
/// is an offset zone; the others are `TIMEZONE_INTERVAL_OUT_OF_RANGE` (the first of months, days,
/// whole hours, truncated seconds); anything that is no zone is `_LEGACY_ERROR_TEMP_0045`.
#[test]
fn set_time_zone_follows_spark() {
    let d = ParserOptions::default();
    let zone = |sql: &str| match parse_statement(sql, &d).unwrap() {
        Statement::Set(s) => match s.kind {
            SetKind::TimeZone { zone, .. } => zone,
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    };
    for (sql, want) in [
        ("SET TIME ZONE INTERVAL 10 HOURS", "+10:00"),
        ("SET TIME ZONE INTERVAL -8 HOURS", "-08:00"),
        ("SET TIME ZONE INTERVAL 0 HOURS", "Z"),
        ("SET TIME ZONE INTERVAL 5 HOURS 30 MINUTES", "+05:30"),
        ("SET TIME ZONE INTERVAL '18:00' HOUR TO MINUTE", "+18:00"),
        (
            "SET TIME ZONE INTERVAL 1 HOURS 2 MINUTES 3 SECONDS",
            "+01:02:03",
        ),
    ] {
        assert_eq!(zone(sql), ast::TimeZone::Named(want.to_owned()), "{sql}");
    }
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    for (sql, input) in [
        ("SET TIME ZONE INTERVAL 3 DAYS", "3"),
        ("SET TIME ZONE INTERVAL 1 MONTH", "1"),
        ("SET TIME ZONE INTERVAL 24 HOURS", "24"),
        ("SET TIME ZONE INTERVAL '19:40:32' HOUR TO SECOND", "19"),
        ("SET TIME ZONE INTERVAL 10 HOURS 1 MILLISECOND", "36000"),
    ] {
        let e = fail(sql);
        assert_eq!(
            e.error.error_class().name(),
            "INVALID_INTERVAL_FORMAT.TIMEZONE_INTERVAL_OUT_OF_RANGE",
            "{sql}"
        );
        assert!(
            e.error.to_string().contains(&format!("\"{input}\""))
                || e.error.to_string().contains(input),
            "{sql}: {}",
            e.error
        );
        // The whole statement is the query context.
        assert_eq!(e.span.start, 0, "{sql}");
        assert_eq!(e.span.end, sql.len(), "{sql}");
    }
    for sql in ["SET TIME ZONE", "SET TIME ZONE INTERVAL 10 HOURS 'GMT+1'"] {
        let e = fail(sql);
        assert_eq!(
            e.error.error_class().name(),
            "_LEGACY_ERROR_TEMP_0045",
            "{sql}"
        );
        assert_eq!((e.span.start, e.span.end), (0, sql.len()), "{sql}");
    }
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
    // An `INTERVAL` time zone holds hours, minutes and seconds only.
    let e = fail("SET TIME ZONE INTERVAL '1' DAY");
    assert_eq!(
        e.error.error_class(),
        ErrorClass::InvalidIntervalFormatTimezoneIntervalOutOfRange
    );
    assert!(e.error.to_string().contains("'1'"), "{e:?}");
    // A captured second range is the literal's own error, not a zone.
    let e = fail("SET TIME ZONE INTERVAL '01:00' HOUR TO MINUTE '1' DAY");
    assert_eq!(e.error.error_class(), ErrorClass::LegacyErrorTemp0024);
    // The largest magnitude does not overflow: it is out of range.
    let e = fail("SET TIME ZONE INTERVAL '-9223372036854775807 microseconds -1 microsecond'");
    assert_eq!(
        e.error.error_class(),
        ErrorClass::InvalidIntervalFormatTimezoneIntervalOutOfRange
    );
    // A temporary function takes no database, with or without a class.
    let e = fail("CREATE TEMPORARY FUNCTION a.f AS 'x.Y'");
    assert_eq!(e.error.error_class(), ErrorClass::InvalidTempObjQualifier);
    let e = fail("CREATE TEMPORARY FUNCTION f AS 'x.Y' USING JAR 'a.jar'");
    assert_eq!(e.error.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // A missing zone, or text that is no zone, is the catch-all alternative.
    let e = fail("SET TIME ZONE");
    assert_eq!(e.error.error_class(), ErrorClass::LegacyErrorTemp0045);
    assert_eq!(e.span.range(), 0..13);
    let e = fail("SET TIME ZONE INTERVAL 10 HOURS 'GMT+1'");
    assert_eq!(e.error.error_class(), ErrorClass::LegacyErrorTemp0045);
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
    // Spark names the view by its unquoted parts.
    assert_eq!(e.error.message_parameters(), [("viewName", "v".to_owned())]);
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
    // `INVALID_TEMP_OBJ_QUALIFIER`, naming the last part and its
    // qualifier, as in Spark 4.2.0.
    let e = parse_statement("CREATE TEMP VIEW a.b AS SELECT 1", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::InvalidTempObjQualifier);
    assert_eq!(e.error.sql_state(), Some("42602"));
    assert_eq!(
        e.error.message_parameters(),
        [
            ("objectType", "VIEW".to_owned()),
            ("objectName", "`b`".to_owned()),
            ("qualifier", "`a`".to_owned()),
        ]
    );
    // A permanent view without `AS` fails where the query would start.
    let fail = |sql: &str| parse_statement(sql, &d).unwrap_err();
    let e = fail("CREATE VIEW v SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 14..20);
    let e = fail("CREATE GLOBAL TEMPORARY VIEW v AS SELECT");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = fail("CREATE TEMP VIEW v SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 19..25);
    let e = fail("CREATE TEMP VIEW v AS");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 21..21);
    // A typed column list is a datasource view's, which needs `USING`
    // (Spark 4.2.0: at `AS`, "missing 'USING'").
    let e = fail("CREATE TEMP VIEW v (a INT) AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 27..29);
    assert_eq!(
        e.error.message_parameters(),
        [
            ("error", "'AS'".to_owned()),
            ("hint", ": missing 'USING'".to_owned())
        ]
    );
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
    let sql = "CREATE TABLE t LIKE s ROW FORMAT SERDE 'x' STORED AS textfile \
        LOCATION '/tmp/t' TBLPROPERTIES ('a' = '1')";
    let t = like(sql);
    assert!(t.using.is_none() && t.row_format.is_some() && t.file_format.is_some());
    assert_eq!(t.location.as_deref(), Some("/tmp/t"));
    assert_eq!(t.tblproperties.len(), 1);
    assert_eq!(t.span.text(sql), sql);
    assert_eq!(t.duplicate_clause, None);
    let sql = "CREATE TABLE t LIKE s USING parquet LOCATION '/tmp/t'";
    assert!(like(sql).using.is_some());
    // `OR REPLACE` takes no `LIKE`: the name is followed by `LIKE`.
    let e = parse_statement("CREATE OR REPLACE TABLE t LIKE s", &d).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 26..30);
}

/// A clause written twice in `CREATE TABLE ... LIKE` is `DUPLICATE_CLAUSES`,
/// the first in Spark's check order (`USING` is `PROVIDER` there).
#[test]
fn create_table_like_duplicate_clauses() {
    use ast::Statement;
    let d = ParserOptions::default();
    for (clause, text) in [
        ("PROVIDER", "USING parquet"),
        ("STORED AS/BY", "STORED AS parquet"),
        ("ROW FORMAT", "ROW FORMAT DELIMITED"),
        ("LOCATION", "LOCATION '/a'"),
        ("TBLPROPERTIES", "TBLPROPERTIES ('a' = 'b')"),
    ] {
        let sql = format!("CREATE TABLE t LIKE s {text} {text}");
        let e = parse_statement(&sql, &d).unwrap_err();
        assert_eq!(e.error.error_class(), ErrorClass::DuplicateClauses, "{sql}");
        assert_eq!(
            e.error.to_string(),
            format!(
                "[DUPLICATE_CLAUSES] Found duplicate clauses: {clause}. Please, remove one of \
                 them. SQLSTATE: 42614"
            ),
            "{sql}"
        );
        assert_eq!(
            e.error.message_parameters(),
            [("clauseName", clause.to_owned())],
            "{sql}"
        );
        assert_eq!(e.span.range(), 0..sql.len(), "{sql}");
        // The AST records it; the statement is otherwise as written.
        let Statement::CreateTableLike(t) =
            crate::parser::parse_statement_unchecked(&sql, d).unwrap()
        else {
            panic!("{sql}");
        };
        assert_eq!(t.duplicate_clause, Some(clause), "{sql}");
    }
    // The first in Spark's order wins over the order in the text, and a
    // different clause twice does not count.
    let e = parse_statement(
        "CREATE TABLE t LIKE s LOCATION '/a' LOCATION '/b' STORED AS orc STORED AS orc",
        &d,
    )
    .unwrap_err();
    assert_eq!(
        e.error.message_parameters(),
        [("clauseName", "STORED AS/BY".to_owned())]
    );
    assert!(
        parse_statement(
            "CREATE TABLE t LIKE s LOCATION '/a' TBLPROPERTIES ('a' = 'b') ROW FORMAT DELIMITED",
            &d
        )
        .is_ok()
    );
}

/// `USING` beside `STORED AS` / `ROW FORMAT` is `INVALID_STATEMENT_OR_CLAUSE`
/// (Spark's `invalidStatement`), in `CREATE TABLE` and `CREATE TABLE ... LIKE`.
/// The text follows `SerdeInfo.describe` of Spark 4.2: `STORED AS` keeps a
/// space before an empty serde clause.
#[test]
fn using_with_a_serde_clause_is_invalid() {
    let d = ParserOptions::default();
    for (sql, operation) in [
        (
            "CREATE TABLE t (a INT) USING parquet STORED AS orc",
            "CREATE TABLE ... USING ... STORED AS orc ",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet ROW FORMAT SERDE 'com.example.S'",
            "CREATE TABLE ... USING ... ROW FORMAT SERDE com.example.S",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet ROW FORMAT DELIMITED FIELDS TERMINATED BY ','",
            "CREATE TABLE ... USING ... ROW FORMAT DELIMITED",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet ROW FORMAT DELIMITED",
            "CREATE TABLE ... USING ... ",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet STORED AS textfile ROW FORMAT DELIMITED \
             LINES TERMINATED BY '\n'",
            "CREATE TABLE ... USING ... STORED AS textfile ROW FORMAT DELIMITED",
        ),
        (
            "CREATE TABLE t USING parquet STORED AS `orc` AS SELECT 1",
            "CREATE TABLE ... USING ... STORED AS `orc` ",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet STORED BY 'com.example.H'",
            "STORED BY",
        ),
        (
            "CREATE TABLE t LIKE s USING parquet STORED AS orc",
            "CREATE TABLE LIKE ... USING ... STORED AS orc ",
        ),
        (
            "CREATE TABLE t LIKE s ROW FORMAT SERDE 'x' USING parquet",
            "CREATE TABLE LIKE ... USING ... ROW FORMAT SERDE x",
        ),
    ] {
        let e = parse_statement(sql, &d).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidStatementOrClause,
            "{sql}"
        );
        assert_eq!(
            e.error.message_parameters(),
            [("operation", operation.to_owned())],
            "{sql}"
        );
        assert_eq!(e.span.range(), 0..sql.len(), "{sql}");
    }
    // Either clause alone, a serde clause without `USING`, and a CTAS
    // query over `USING` are fine.
    for sql in [
        "CREATE TABLE t (a INT) USING parquet",
        "CREATE TABLE t (a INT) STORED AS orc",
        "CREATE TABLE t (a INT) ROW FORMAT DELIMITED STORED AS textfile",
        "CREATE TABLE t USING parquet AS SELECT 1",
        "CREATE TABLE t LIKE s STORED AS orc",
        "CREATE TABLE t LIKE s USING parquet",
    ] {
        assert!(parse_statement(sql, &d).is_ok(), "{sql}");
    }
    // The order of the checks: a repeated clause comes before it, and it
    // comes before the CTAS query's own errors.
    let e = parse_statement(
        "CREATE TABLE t (a INT) USING parquet STORED AS orc STORED AS orc",
        &d,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::DuplicateClauses);
    let e = parse_statement(
        "CREATE TABLE t USING parquet STORED AS orc AS SELECT DATE '2020-13-45'",
        &d,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::InvalidStatementOrClause);
}

/// `CLUSTERED BY ... INTO n BUCKETS` needs `0 < n <= 100000`
/// (`INVALID_BUCKET_COUNT`, reported at the clause), after the repeated and
/// `SKEWED BY` checks and before `USING` with a serde clause.
#[test]
fn bucket_count_must_be_in_range() {
    let d = ParserOptions::default();
    let prefix = "CREATE TABLE t (a INT) ";
    for (count, valid) in [
        ("1", true),
        ("100000", true),
        ("0", false),
        ("100001", false),
        ("4294967296", false),
    ] {
        let clause = format!("CLUSTERED BY (a) SORTED BY (a DESC) INTO {count} BUCKETS");
        let sql = format!("{prefix}{clause} COMMENT 'x'");
        let parsed = parse_statement(&sql, &d);
        if valid {
            assert!(parsed.is_ok(), "{sql}");
            continue;
        }
        let e = parsed.unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidBucketCount,
            "{sql}"
        );
        assert_eq!(
            e.error.message_parameters(),
            [
                ("bucketingMaxBuckets", "100000".to_owned()),
                ("numBuckets", count.to_owned())
            ],
            "{sql}"
        );
        assert_eq!(
            e.error.to_string(),
            format!(
                "[INVALID_BUCKET_COUNT] Number of buckets should be greater than 0 but less \
                 than or equal to bucketing.maxBuckets (`100000`). Got `{count}`. SQLSTATE: 22003"
            ),
            "{sql}"
        );
        assert_eq!(e.span.text(&sql), clause, "{sql}");
    }
    for (sql, class) in [
        (
            "CREATE TABLE t (a INT) CLUSTERED BY (a) INTO 0 BUCKETS COMMENT 'x' COMMENT 'y'",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE TABLE t (a INT) CLUSTERED BY (a) INTO 0 BUCKETS SKEWED BY (a) ON (1)",
            ErrorClass::InvalidStatementOrClause,
        ),
        (
            "CREATE TABLE t (a INT) CLUSTERED BY (a) INTO 0 BUCKETS OPTIONS (k DATE '2020-13-45')",
            ErrorClass::InvalidBucketCount,
        ),
        (
            "CREATE TABLE t (a INT) USING parquet STORED AS orc CLUSTERED BY (a) INTO 0 BUCKETS",
            ErrorClass::InvalidBucketCount,
        ),
    ] {
        assert_eq!(
            parse_statement(sql, &d).unwrap_err().error.error_class(),
            class,
            "{sql}"
        );
    }
    // The session's `spark.sql.sources.bucketing.maxBuckets` moves the limit.
    let two = ParserOptions::from_conf(|k| {
        (k == "spark.sql.sources.bucketing.maxBuckets").then_some("2")
    });
    assert_eq!(two.max_buckets, 2);
    assert!(
        parse_statement(
            "CREATE TABLE t (a INT) CLUSTERED BY (a) INTO 2 BUCKETS",
            &two
        )
        .is_ok()
    );
    let e = parse_statement(
        "CREATE TABLE t (a INT) CLUSTERED BY (a) INTO 3 BUCKETS",
        &two,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::InvalidBucketCount);
    assert_eq!(
        e.error.message_parameters(),
        [
            ("bucketingMaxBuckets", "2".to_owned()),
            ("numBuckets", "3".to_owned())
        ]
    );
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
        DEFAULT COLLATION UTF8_LCASE WITH DBPROPERTIES ('k' = 'v', flag true)",
    );
    assert_eq!(n.kind, NamespaceKind::Database);
    assert!(n.if_not_exists);
    assert_eq!(n.comment.as_deref(), Some("c"));
    assert_eq!(n.location.as_deref(), Some("/tmp/db"));
    assert!(n.collation.is_some());
    assert_eq!(n.properties.len(), 2);
    assert!(n.properties[1].value.is_some());
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
    let lcase = krishiv_types::Collation::parse("UTF8_LCASE").unwrap();
    assert_eq!(
        t.to_data_type().unwrap(),
        krishiv_types::DataType::collated_string(lcase)
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
    // A cast to a collated type lowers.
    let q = ok("SELECT CAST(a AS STRING COLLATE UTF8_LCASE) FROM t");
    let plan = crate::lower(&q, krishiv_plan::SourceId(0)).unwrap();
    assert!(format!("{plan:?}").contains("CollatedString(Collation"));
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
