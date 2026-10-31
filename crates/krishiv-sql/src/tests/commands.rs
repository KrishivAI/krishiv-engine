// SPDX-License-Identifier: Apache-2.0
//! Parser unit tests for the command statements: `ALTER`, `SHOW`,
//! `DESCRIBE`, `INSERT`, `DELETE`, `UPDATE`, `MERGE`, `TRUNCATE`, `CACHE`,
//! `REFRESH`, `ANALYZE`, `REPAIR`, `ADD`/`LIST` and the full `CREATE`/`DROP
//! VIEW` grammar. Tree shapes (spans included) are `insta` snapshots, one
//! per statement family; errors, spans and re-spacing equality are asserted
//! directly.

use super::*;
use krishiv_common::error::ErrorClass;

/// The tree of `sql`, without Spark's post-parse checks (tree shapes).
fn tree(sql: &str) -> ast::Statement {
    match crate::parser::parse_statement_unchecked(sql, ParserOptions::default()) {
        Ok(s) => s,
        Err(e) => panic!("{sql}: {}", e.error),
    }
}

/// The tree of `sql`, with Spark's post-parse checks.
fn checked(sql: &str) -> ast::Statement {
    match parse_statement(sql, &ParserOptions::default()) {
        Ok(s) => s,
        Err(e) => panic!("{sql}: {}", e.error),
    }
}

fn err(sql: &str) -> ParseError {
    parse_statement(sql, &ParserOptions::default()).unwrap_err()
}

/// `sql`'s tokens (comments dropped) joined by a comment and a newline.
fn respaced(sql: &str) -> String {
    tokenize(sql)
        .iter()
        .filter(|t| !t.kind.is_trivia())
        .map(|t| t.span.text(sql))
        .collect::<Vec<_>>()
        .join(" /* x */\n")
}

/// Snapshots of `statements`: each parses, re-parses to an `==` tree with
/// other whitespace and comments, and renders with spans.
fn snapshot_stmts(statements: &[&str]) -> String {
    let mut out = String::new();
    for sql in statements {
        let parsed = tree(sql);
        let again = tree(&respaced(sql));
        assert_eq!(parsed, again, "{sql}");
        use std::fmt::Write as _;
        writeln!(out, "-- {sql}\n{parsed:#?}").unwrap();
    }
    out
}

/// `ALTER TABLE` statements: every action.
#[test]
fn alter_table_statements() {
    insta::assert_snapshot!(snapshot_stmts(&[
        "ALTER TABLE t ADD COLUMN c INT",
        "ALTER TABLE t ADD COLUMNS (a INT COMMENT 'the a', b STRING)",
        "ALTER TABLE t DROP COLUMN c",
        "ALTER TABLE t DROP COLUMNS (a, b)",
        "ALTER TABLE t RENAME COLUMN a TO b",
        "ALTER TABLE t RENAME TO t2",
        "ALTER TABLE t CHANGE a TYPE STRING",
        "ALTER TABLE t CHANGE COLUMN a COMMENT 'the a'",
        "ALTER TABLE t CHANGE a FIRST",
        "ALTER TABLE t CHANGE a AFTER b",
        "ALTER TABLE t CHANGE a SET NOT NULL",
        "ALTER TABLE t CHANGE a DROP NOT NULL",
        "ALTER TABLE t ALTER COLUMN a COMMENT 'the a'",
        "ALTER TABLE t ALTER a SET NOT NULL",
        "ALTER TABLE t ALTER a DROP NOT NULL",
        "ALTER TABLE t ALTER a SET DEFAULT 1",
        "ALTER TABLE t ALTER a DROP DEFAULT",
        "ALTER TABLE t SET TBLPROPERTIES ('yes' = 'no')",
        "ALTER TABLE t UNSET TBLPROPERTIES ('yes')",
        "ALTER TABLE t UNSET TBLPROPERTIES IF EXISTS ('a', 'b')",
        "ALTER TABLE t SET SERDEPROPERTIES ('yes' = 'no')",
        "ALTER TABLE t SET LOCATION 'loc'",
        "ALTER TABLE t ADD PARTITION (c = 'Us', d = 1)",
        "ALTER TABLE t ADD IF NOT EXISTS PARTITION (c = 'Us') LOCATION 'loc1' PARTITION (d = 2)",
        "ALTER TABLE t DROP PARTITION (c = 'Us')",
        "ALTER TABLE t DROP IF EXISTS PARTITION (c = 'Us'), PARTITION (d = 2) PURGE",
        "ALTER TABLE t PARTITION (c = 'Us') SET SERDEPROPERTIES ('yes' = 'no')",
        "ALTER TABLE t PARTITION (c = 'Us') RENAME TO PARTITION (c = 'Them')",
        "ALTER TABLE t PARTITION (c = 'Us', d = 1) SET LOCATION 'loc2'",
        "ALTER TABLE t RECOVER PARTITIONS",
        "ALTER TABLE IDENTIFIER('ta' || 'b') ADD COLUMN c2 INT",
        "ALTER TABLE IDENTIFIER('tab') RENAME TO IDENTIFIER('tab_renamed')",
    ]));
}

/// `ALTER TABLE` errors, positions and precedence as Spark 4.2 raises them.
#[test]
fn alter_table_errors() {
    let d = ParserOptions::default();
    // A bare `CHANGE` names what the change needs, over the statement.
    let sql = "ALTER TABLE test_change CHANGE a";
    let e = err(sql);
    assert_eq!(e.error.error_class(), ErrorClass::LegacyErrorTemp0035);
    assert_eq!(
        e.error.message_parameters(),
        [(
            "message",
            "ALTER TABLE table CHANGE COLUMN requires a TYPE, a SET/DROP, a COMMENT, or a FIRST/AFTER"
                .to_owned()
        )]
    );
    assert_eq!(e.span.range(), 0..sql.len());
    // A computed column name is rejected at the `(` in the legacy
    // identifier mode, as in Spark.
    let legacy = ParserOptions {
        legacy_identifier_clause: true,
        ..d
    };
    for (sql, at) in [
        ("ALTER TABLE t ADD COLUMN IDENTIFIER('c') INT", 35..36),
        ("ALTER TABLE t DROP COLUMN IDENTIFIER('c')", 36..37),
        ("ALTER TABLE t RENAME COLUMN IDENTIFIER('a') TO b", 38..39),
        (
            "ALTER TABLE t ALTER COLUMN IDENTIFIER('c') COMMENT 'x'",
            37..38,
        ),
    ] {
        let e = parse_statement(sql, &legacy).unwrap_err();
        assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError, "{sql}");
        assert_eq!(e.span.range(), at, "{sql}");
    }
    // So is a computed rename target.
    let e = parse_statement("ALTER TABLE t RENAME TO IDENTIFIER('u')", &legacy).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // The same statements parse outside the legacy mode.
    for sql in [
        "ALTER TABLE t ADD COLUMN IDENTIFIER('c') INT",
        "ALTER TABLE t RENAME TO IDENTIFIER('u')",
    ] {
        assert!(parse_statement(sql, &d).is_ok(), "{sql}");
    }
    // Unknown actions fail where they stand (positions from Spark 4.2.0).
    let e = err("ALTER TABLE t SET FOO 'x'");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 18..21);
    let e = err("ALTER TABLE t PARTITION (c = 1) SET FOO 'x'");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 36..39);
    // `RENAME` needs `COLUMN` or `TO`: Spark fails at the column.
    let e = err("ALTER TABLE t RENAME a TO b");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 21..22);
    // Partition specs to drop are comma-separated.
    let e = err("ALTER TABLE t DROP IF EXISTS PARTITION (c = 'Us') PARTITION (d = 2) PURGE");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 50..59);
}

/// Property lists: `SET` needs a value for every key and `UNSET` takes
/// none, with Spark 4.2.0's `_LEGACY_ERROR_TEMP_0035` over the statement.
#[test]
fn property_list_shapes() {
    for (sql, message) in [
        (
            "ALTER TABLE t SET TBLPROPERTIES ('a')",
            "Values must be specified for key(s): [a]",
        ),
        (
            "ALTER TABLE t SET TBLPROPERTIES ('a', 'b')",
            "Values must be specified for key(s): [a,b]",
        ),
        (
            "ALTER TABLE t SET TBLPROPERTIES (a.b)",
            "Values must be specified for key(s): [a.b]",
        ),
        (
            "ALTER TABLE t UNSET TBLPROPERTIES ('a', 'b' = 'c')",
            "Values should not be specified for key(s): [b]",
        ),
        (
            "ALTER TABLE t UNSET TBLPROPERTIES IF EXISTS ('a' = '1')",
            "Values should not be specified for key(s): [a]",
        ),
        (
            "ALTER TABLE t SET SERDEPROPERTIES ('a')",
            "Values must be specified for key(s): [a]",
        ),
        (
            "ALTER TABLE t PARTITION (p = 1) SET SERDEPROPERTIES ('a')",
            "Values must be specified for key(s): [a]",
        ),
        (
            "ALTER VIEW v SET TBLPROPERTIES ('a')",
            "Values must be specified for key(s): [a]",
        ),
        (
            "ALTER VIEW v UNSET TBLPROPERTIES ('a' = 'b')",
            "Values should not be specified for key(s): [a]",
        ),
        (
            "ALTER NAMESPACE n SET PROPERTIES ('a')",
            "Values must be specified for key(s): [a]",
        ),
        (
            "CREATE VIEW v TBLPROPERTIES ('a') AS SELECT 1",
            "Values must be specified for key(s): [a]",
        ),
        (
            "CREATE TEMPORARY VIEW v USING parquet OPTIONS ('a')",
            "Values must be specified for key(s): [a]",
        ),
        (
            "CACHE TABLE t OPTIONS ('a') AS SELECT 1",
            "Values must be specified for key(s): [a]",
        ),
    ] {
        let e = err(sql);
        assert_eq!(
            e.error.error_class(),
            ErrorClass::LegacyErrorTemp0035,
            "{sql}"
        );
        assert_eq!(
            e.error.message_parameters(),
            [("message", message.to_owned())],
            "{sql}"
        );
        assert_eq!(e.span.range(), 0..sql.len(), "{sql}");
    }
    // `INSERT` target options: Spark 4.2.0 spans the target name.
    let sql = "INSERT INTO t WITH ('a') SELECT 1";
    let e = err(sql);
    assert_eq!(
        e.error.error_class(),
        ErrorClass::LegacyErrorTemp0035,
        "{sql}"
    );
    assert_eq!(
        e.error.message_parameters(),
        [(
            "message",
            "Values must be specified for key(s): [a]".to_owned()
        )],
        "{sql}"
    );
    assert_eq!(e.span.range(), 12..13, "{sql}");
    // Schema-level collations are off in Spark 4.2.
    let sql = "ALTER SCHEMA s DEFAULT COLLATION UTF8_LCASE";
    let e = err(sql);
    assert_eq!(
        e.error.error_class(),
        ErrorClass::UnsupportedFeatureSchemaLevelCollations
    );
}

/// `ALTER VIEW` and `ALTER DATABASE|SCHEMA|NAMESPACE` statements.
#[test]
fn alter_view_and_namespace_statements() {
    insta::assert_snapshot!(snapshot_stmts(&[
        "ALTER VIEW v AS SELECT * FROM t",
        "ALTER VIEW v WITH SCHEMA BINDING",
        "ALTER VIEW v WITH SCHEMA EVOLUTION",
        "ALTER VIEW v WITH SCHEMA COMPENSATION",
        "ALTER VIEW v WITH SCHEMA TYPE EVOLUTION",
        "ALTER VIEW v SET TBLPROPERTIES ('yes' = 'no')",
        "ALTER VIEW v UNSET TBLPROPERTIES IF EXISTS ('yes')",
        "ALTER VIEW v RENAME TO v2",
        "ALTER VIEW IDENTIFIER('v') AS VALUES (2)",
        "ALTER SCHEMA s SET PROPERTIES (k = 'v')",
        "ALTER SCHEMA s SET LOCATION 'loc'",
        "ALTER SCHEMA s DEFAULT COLLATION UTF8_LCASE",
        "ALTER DATABASE d SET PROPERTIES (k = 'v')",
        "ALTER NAMESPACE n SET LOCATION 'loc'",
    ]));
    // Unknown actions fail where they stand.
    let e = err("ALTER VIEW v SET FOO 'x'");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = err("ALTER SCHEMA s SET FOO 'x'");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
}

/// The full `CREATE VIEW` grammar: permanent views, and the temporary
/// views the legacy node cannot express.
#[test]
fn create_full_view_statements() {
    insta::assert_snapshot!(snapshot_stmts(&[
        "CREATE VIEW v AS SELECT * FROM t",
        "CREATE OR REPLACE VIEW v AS SELECT 1",
        "CREATE VIEW IF NOT EXISTS v AS SELECT 1",
        "CREATE VIEW v (a COMMENT 'the a', b) AS SELECT 1, 2",
        "CREATE VIEW v COMMENT 'the view' TBLPROPERTIES ('p' = 'v') AS SELECT 1",
        "CREATE VIEW v WITH SCHEMA BINDING AS SELECT 1",
        "CREATE VIEW v WITH SCHEMA EVOLUTION AS SELECT 1",
        "CREATE VIEW v WITH SCHEMA COMPENSATION AS SELECT 1",
        "CREATE VIEW v WITH SCHEMA TYPE EVOLUTION AS SELECT 1",
        "CREATE GLOBAL TEMPORARY VIEW v AS SELECT 1",
        "CREATE OR REPLACE GLOBAL TEMP VIEW v AS SELECT 1",
        "CREATE TEMPORARY VIEW v USING parquet",
        "CREATE TEMPORARY VIEW v (a INT) USING parquet OPTIONS (k = 'v')",
        "CREATE TEMPORARY VIEW v (e INT) USING parquet",
        "CREATE OR REPLACE TEMPORARY VIEW v (a INT NOT NULL) USING parquet",
        "CREATE VIEW v TBLPROPERTIES ('a' = 'b') COMMENT 'c' AS SELECT 1",
        "CREATE TEMPORARY VIEW v USING org.apache.spark.sql.sources.DDLScanSource OPTIONS (From '1', To '10', Table 'test1')",
        "CREATE TEMPORARY VIEW v DEFAULT COLLATION UNICODE AS SELECT CAST(c1 AS STRING)",
        "CREATE VIEW IDENTIFIER('v') (c1) AS VALUES (1)",
        "CREATE OR REPLACE VIEW IDENTIFIER('v') (c1) AS VALUES (1)",
        "CREATE TEMPORARY VIEW IDENTIFIER('v') (c1) AS VALUES (1)",
        "CREATE VIEW v (IDENTIFIER('c1')) AS VALUES (1)",
        "CREATE VIEW db.v AS SELECT * FROM t",
    ]));
    // Flags and spans.
    let ast::Statement::CreatePermanentView(v) = checked("CREATE VIEW v AS SELECT 1") else {
        panic!("not a full view")
    };
    assert!(!v.or_replace && !v.temporary && !v.global && !v.if_not_exists);
    assert_eq!(
        v.span.text("CREATE VIEW v AS SELECT 1"),
        "CREATE VIEW v AS SELECT 1"
    );
    let ast::Statement::CreatePermanentView(v) =
        checked("CREATE OR REPLACE GLOBAL TEMPORARY VIEW v AS SELECT 1")
    else {
        panic!("not a full view")
    };
    assert!(v.or_replace && v.temporary && v.global);
}

/// Full `CREATE VIEW` errors as Spark 4.2.0 raises them (positions and
/// messages from a Spark 4.2.0 run).
#[test]
fn create_full_view_errors() {
    // Syntax errors: (statement, error text, hint, position).
    for (sql, near, hint, at) in [
        // A datasource view is always temporary.
        (
            "CREATE VIEW v USING parquet AS SELECT 1",
            "'USING'",
            "",
            14..19,
        ),
        ("CREATE VIEW v USING parquet", "'USING'", "", 14..19),
        // `GLOBAL` needs `TEMPORARY`.
        ("CREATE GLOBAL VIEW v AS SELECT 1", "'VIEW'", "", 14..18),
        // A datasource view's name has at most two parts.
        (
            "CREATE TEMPORARY VIEW system.session.v USING parquet",
            "'USING'",
            "",
            39..44,
        ),
        // No query and no provider.
        ("CREATE VIEW v", "end of input", "", 13..13),
        // A column type reads only in a temporary view's typed list: at
        // the type in a permanent view, and as "extra input" in a
        // temporary view whose first column has none.
        ("CREATE VIEW v (a INT) AS SELECT 1", "'INT'", "", 17..20),
        (
            "CREATE VIEW v (a DECIMAL(10, 2)) AS SELECT 1",
            "'DECIMAL'",
            "",
            17..24,
        ),
        (
            "CREATE VIEW v (b, a INT COMMENT 'x') AS SELECT 1, 2",
            "'INT'",
            "",
            20..23,
        ),
        (
            "CREATE TEMPORARY VIEW v (a, b INT) USING parquet",
            "'INT'",
            ": extra input 'INT'",
            30..33,
        ),
        (
            "CREATE TEMPORARY VIEW v (a, b DECIMAL(10, 2)) USING parquet",
            "'DECIMAL'",
            ": extra input 'DECIMAL'",
            30..37,
        ),
        // A typed list needs a type on every column and then `USING`.
        (
            "CREATE TEMPORARY VIEW v (a INT, b) USING parquet",
            "')'",
            "",
            33..34,
        ),
        (
            "CREATE TEMPORARY VIEW v (a INT, b, c INT) USING parquet",
            "','",
            "",
            33..34,
        ),
        (
            "CREATE TEMPORARY VIEW v (a INT) AS SELECT 1",
            "'AS'",
            ": missing 'USING'",
            32..34,
        ),
        (
            "CREATE TEMPORARY VIEW v (a INT) COMMENT 'x' AS SELECT 1",
            "'COMMENT'",
            ": missing 'USING'",
            32..39,
        ),
        // An untyped list is `createView`'s: no `USING`.
        (
            "CREATE TEMPORARY VIEW v (a) USING parquet OPTIONS (path 'x')",
            "'USING'",
            "",
            28..33,
        ),
        // A datasource view takes only `OPTIONS` after its provider.
        (
            "CREATE TEMPORARY VIEW v USING parquet OPTIONS (path 'x') AS SELECT 1",
            "'AS'",
            "",
            57..59,
        ),
        (
            "CREATE TEMPORARY VIEW v USING parquet COMMENT 'c'",
            "'COMMENT'",
            "",
            38..45,
        ),
        (
            "CREATE TEMPORARY VIEW v USING parquet TBLPROPERTIES ('a' = 'b')",
            "'TBLPROPERTIES'",
            "",
            38..51,
        ),
        (
            "CREATE TEMPORARY VIEW v USING parquet WITH SCHEMA BINDING",
            "'WITH'",
            "",
            38..42,
        ),
        (
            "CREATE TEMPORARY VIEW IF NOT EXISTS v USING parquet",
            "'USING'",
            "",
            38..43,
        ),
    ] {
        let e = err(sql);
        assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError, "{sql}");
        assert_eq!(
            e.error.message_parameters(),
            [("error", near.to_owned()), ("hint", hint.to_owned())],
            "{sql}"
        );
        assert_eq!(e.span.range(), at, "{sql}");
    }
    // Checks after parsing, over the statement, in Spark's order: a
    // qualified temporary name names its last part and its qualifier.
    let e = err("CREATE TEMPORARY VIEW a.b AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::InvalidTempObjQualifier);
    assert_eq!(
        e.error.message_parameters(),
        [
            ("objectType", "VIEW".to_owned()),
            ("objectName", "`b`".to_owned()),
            ("qualifier", "`a`".to_owned()),
        ]
    );
    let e = err("CREATE TEMPORARY VIEW db.v USING parquet");
    assert_eq!(e.error.error_class(), ErrorClass::InvalidTempObjQualifier);
    // A global datasource view keeps its database, which Spark rejects.
    let e = err("CREATE GLOBAL TEMPORARY VIEW db.v USING parquet");
    assert_eq!(e.error.error_class(), ErrorClass::LegacyErrorTemp1292);
    assert_eq!(
        e.error.message_parameters(),
        [("tableIdent", "`db`.`v`".to_owned())]
    );
    // `session` and `system.session` are a temporary view's own.
    assert!(matches!(
        checked("CREATE TEMPORARY VIEW session.v AS SELECT 1"),
        ast::Statement::CreatePermanentView(_)
    ));
    checked("CREATE TEMPORARY VIEW system.session.v AS SELECT 1");
    checked("CREATE TEMPORARY VIEW session.v USING parquet");
    // `OR REPLACE` with `IF NOT EXISTS` fails on every view, naming the
    // view as written, before the query is checked.
    for (sql, name) in [
        ("CREATE OR REPLACE VIEW IF NOT EXISTS v AS SELECT 1", "v"),
        (
            "CREATE OR REPLACE TEMPORARY VIEW IF NOT EXISTS db.v AS SELECT 1",
            "db.v",
        ),
        (
            "CREATE OR REPLACE VIEW IF NOT EXISTS `a b` . v AS SELECT 1",
            "a b.v",
        ),
        (
            "CREATE OR REPLACE VIEW IF NOT EXISTS IDENTIFIER('v') AS SELECT 1",
            "IDENTIFIER('v')",
        ),
        (
            "CREATE OR REPLACE VIEW IF NOT EXISTS v AS SELECT DATE '2020-02-30'",
            "v",
        ),
        (
            "CREATE OR REPLACE TEMPORARY VIEW IF NOT EXISTS v TBLPROPERTIES ('a' = 'b') AS SELECT 1",
            "v",
        ),
    ] {
        let e = err(sql);
        assert_eq!(
            e.error.error_class(),
            ErrorClass::CreateViewWithIfNotExistsAndReplace,
            "{sql}"
        );
        assert_eq!(
            e.error.message_parameters(),
            [("viewName", name.to_owned())],
            "{sql}"
        );
        assert_eq!(e.span.range(), 0..sql.len(), "{sql}");
    }
    // `IF NOT EXISTS` on a temporary view, and a temporary qualifier, fail
    // after the query's own checks.
    let e = err("CREATE TEMPORARY VIEW IF NOT EXISTS v AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::LegacyErrorTemp0053);
    for sql in [
        "CREATE TEMPORARY VIEW IF NOT EXISTS v AS SELECT DATE '2020-02-30'",
        "CREATE TEMPORARY VIEW db.v AS SELECT DATE '2020-02-30'",
    ] {
        assert_eq!(
            err(sql).error.error_class(),
            ErrorClass::InvalidTypedLiteral,
            "{sql}"
        );
    }
    // The other builder checks, over the statement.
    for (sql, class) in [
        (
            "CREATE VIEW v PARTITIONED ON (a) COMMENT 'x' COMMENT 'y' AS SELECT 1",
            ErrorClass::InvalidStatementOrClause,
        ),
        (
            "CREATE OR REPLACE VIEW IF NOT EXISTS v COMMENT 'x' COMMENT 'y' AS SELECT 1",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE VIEW v WITH SCHEMA BINDING WITH SCHEMA EVOLUTION AS SELECT 1",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE TEMPORARY VIEW v TBLPROPERTIES ('a' = 'b') AS SELECT 1",
            ErrorClass::LegacyErrorTemp0035,
        ),
        (
            "CREATE TEMPORARY VIEW v WITH SCHEMA BINDING AS SELECT 1",
            ErrorClass::UnsupportedFeatureTemporaryViewWithSchemaBindingMode,
        ),
    ] {
        let e = err(sql);
        assert_eq!(e.error.error_class(), class, "{sql}");
        assert_eq!(e.span.range(), 0..sql.len(), "{sql}");
    }
    let e = err(
        "CREATE VIEW v DEFAULT COLLATION UTF8_BINARY DEFAULT COLLATION UTF8_BINARY AS SELECT 1",
    );
    assert_eq!(
        e.error.message_parameters(),
        [("clauseName", "DEFAULT COLLATION".to_owned())]
    );
}

/// `WITH SCHEMA ...` with `spark.sql.legacy.viewSchemaBindingMode` off: `FEATURE_NOT_ENABLED` at the clause
/// (golden `view-schema-binding-config`), on `CREATE VIEW` after the query and on `ALTER VIEW`; a temporary view
/// fails on its own check first, and a view without the clause parses.
#[test]
fn schema_binding_needs_the_binding_mode_config() {
    let off = ParserOptions {
        view_schema_binding: false,
        ..ParserOptions::default()
    };
    for (sql, fragment) in [
        (
            "CREATE OR REPLACE VIEW v WITH SCHEMA BINDING AS SELECT 1",
            "WITH SCHEMA BINDING",
        ),
        (
            "CREATE VIEW v WITH SCHEMA TYPE EVOLUTION AS SELECT 1",
            "WITH SCHEMA TYPE EVOLUTION",
        ),
        (
            "CREATE VIEW v COMMENT 'c' WITH SCHEMA COMPENSATION AS SELECT 1",
            "WITH SCHEMA COMPENSATION",
        ),
        (
            "ALTER VIEW v WITH SCHEMA EVOLUTION",
            "WITH SCHEMA EVOLUTION",
        ),
    ] {
        let e = parse_statement(sql, &off).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::FeatureNotEnabled,
            "{sql}"
        );
        assert_eq!(e.span.text(sql), fragment, "{sql}");
        assert_eq!(
            e.error.message_parameters(),
            [
                ("featureName", "VIEW ... WITH SCHEMA ...".to_owned()),
                (
                    "configKey",
                    "spark.sql.legacy.viewSchemaBindingMode".to_owned()
                ),
                ("configValue", "true".to_owned()),
            ]
        );
        assert!(
            parse_statement(sql, &ParserOptions::default()).is_ok(),
            "{sql}"
        );
    }
    // The query's own errors come first.
    let e = parse_statement(
        "CREATE VIEW v WITH SCHEMA BINDING AS SELECT DATE '2020-02-30'",
        &off,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::InvalidTypedLiteral);
    let e = parse_statement(
        "CREATE TEMPORARY VIEW v WITH SCHEMA BINDING AS SELECT 1",
        &off,
    )
    .unwrap_err();
    assert_eq!(
        e.error.error_class(),
        ErrorClass::UnsupportedFeatureTemporaryViewWithSchemaBindingMode
    );
    assert!(parse_statement("CREATE VIEW v AS SELECT 1", &off).is_ok());
}

/// Full `DROP VIEW` statements: qualified or computed names.
#[test]
fn drop_full_view_statements() {
    insta::assert_snapshot!(snapshot_stmts(&[
        "DROP VIEW a.b",
        "DROP VIEW IF EXISTS a.b.c",
        "DROP VIEW IDENTIFIER('v')",
        "DROP VIEW global_temp.view_3",
    ]));
    let ast::Statement::DropPermanentView(v) = checked("DROP VIEW IF EXISTS a.b") else {
        panic!("not a full drop")
    };
    assert!(v.if_exists);
    assert_eq!(
        v.span.text("DROP VIEW IF EXISTS a.b"),
        "DROP VIEW IF EXISTS a.b"
    );
    // A missing name fails at the end of the input.
    let e = err("DROP VIEW");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 9..9);
}

/// Statement dispatch: the parser reads each ambiguous shape as exactly
/// one statement kind.
#[test]
fn statement_dispatch_disambiguation() {
    // `CREATE OR REPLACE VIEW` is a permanent view, not a table; `CREATE
    // OR REPLACE TABLE` is a table, not a view.
    let ast::Statement::CreatePermanentView(v) = checked("CREATE OR REPLACE VIEW v AS SELECT 1")
    else {
        panic!("not a permanent view")
    };
    assert!(v.or_replace && !v.temporary);
    let ast::Statement::CreateTable(_) = checked("CREATE OR REPLACE TABLE t (a INT)") else {
        panic!("not a table")
    };
    assert!(!matches!(
        checked("CREATE OR REPLACE VIEW v AS SELECT 1"),
        ast::Statement::CreateTable(_)
    ));
    assert!(!matches!(
        checked("CREATE OR REPLACE TABLE t (a INT)"),
        ast::Statement::CreatePermanentView(_)
    ));
    // `CREATE OR REPLACE TEMP VIEW` stays the legacy temporary view, not
    // the full grammar.
    assert!(matches!(
        checked("CREATE OR REPLACE TEMP VIEW v AS SELECT 1"),
        ast::Statement::CreateView(_)
    ));
    // A bare `DROP VIEW v` is the legacy drop; a qualified name is the
    // full grammar.
    assert!(matches!(
        checked("DROP VIEW v"),
        ast::Statement::DropView(_)
    ));
    assert!(!matches!(
        checked("DROP VIEW v"),
        ast::Statement::DropPermanentView(_)
    ));
    assert!(matches!(
        checked("DROP VIEW a.b"),
        ast::Statement::DropPermanentView(_)
    ));
    assert!(!matches!(
        checked("DROP VIEW a.b"),
        ast::Statement::DropView(_)
    ));
    // A `FROM`-led query is a query, not a multi-table insert; a
    // `FROM`-led insert is an insert, not a query.
    assert!(matches!(
        checked("FROM t SELECT a"),
        ast::Statement::Query(_)
    ));
    assert!(!matches!(
        checked("FROM t SELECT a"),
        ast::Statement::Insert(_)
    ));
    assert!(matches!(
        checked("FROM s INSERT INTO a SELECT *"),
        ast::Statement::Insert(_)
    ));
    assert!(!matches!(
        checked("FROM s INSERT INTO a SELECT *"),
        ast::Statement::Query(_)
    ));
}

/// `INSERT` statements: single-table and multi-table.
#[test]
fn insert_statements() {
    insta::assert_snapshot!(snapshot_stmts(&[
        "INSERT INTO t SELECT * FROM s",
        "INSERT INTO t VALUES (1, 'a'), (2, 'b')",
        "INSERT INTO TABLE t SELECT * FROM s",
        "INSERT OVERWRITE TABLE t SELECT * FROM s",
        "INSERT INTO t (a, b) SELECT c, d FROM s",
        "INSERT INTO t PARTITION (c = 'Us', d = 1) SELECT * FROM s",
        "INSERT INTO a PARTITION (a = TIMESTAMP_NTZ '2018-11-17 13:33:33') VALUES (1)",
        "INSERT INTO t BY NAME SELECT * FROM s",
        "INSERT INTO t REPLACE WHERE c > 0 SELECT * FROM s",
        "INSERT INTO t PARTITION (d) BY NAME SELECT * FROM s",
        "INSERT OVERWRITE t PARTITION (d = 1) SELECT * FROM s",
        "INSERT INTO tab (IDENTIFIER('c1')) VALUES (1)",
        "INSERT INTO IDENTIFIER('ta' || 'b') VALUES (1)",
        "INSERT INTO t WITH s AS (SELECT 44 AS col) SELECT * FROM s",
        "INSERT INTO t (SELECT * FROM s)",
        "INSERT INTO t PARTITION (c = 'Us') (a, b) SELECT 1, 2",
        "INSERT OVERWRITE TABLE t PARTITION (c = 'Us') IF NOT EXISTS SELECT 1",
        "INSERT OVERWRITE t PARTITION (c = 'Us') IF NOT EXISTS BY NAME SELECT 1",
        "INSERT INTO t BY NAME REPLACE WHERE c > 0 SELECT 1",
        "INSERT WITH SCHEMA EVOLUTION INTO t SELECT 1",
        "INSERT WITH SCHEMA EVOLUTION OVERWRITE TABLE t SELECT 1",
        "INSERT WITH SCHEMA EVOLUTION INTO t WITH ('a' = '1') SELECT 1",
        "INSERT INTO t WITH ('a' = '1') SELECT 1",
        "INSERT INTO t AS a REPLACE ON a.c1 = s.c1 SELECT * FROM s",
        "INSERT INTO t REPLACE ON c1 = 1 SELECT 1",
        "INSERT INTO t BY NAME REPLACE ON c1 = 1 SELECT 1",
        "INSERT INTO t REPLACE USING (c1) SELECT 1",
        "INSERT INTO t AS a BY NAME REPLACE USING (c1) SELECT 1",
        "FROM s INSERT INTO a SELECT * INSERT INTO b SELECT *",
        "FROM s INSERT OVERWRITE TABLE a SELECT * INSERT INTO b PARTITION (d = 1) SELECT *",
        "FROM s INSERT INTO a SELECT * WHERE x > 1 GROUP BY x INSERT INTO b SELECT 1 ORDER BY 1",
        "FROM s INSERT INTO a SELECT *",
    ]));
    let ast::Statement::Insert(i) = checked("INSERT OVERWRITE TABLE t SELECT * FROM s") else {
        panic!("not an insert")
    };
    assert!(i.source.is_none() && i.clauses.len() == 1 && i.clauses[0].overwrite);
    assert_eq!(
        i.span.text("INSERT OVERWRITE TABLE t SELECT * FROM s"),
        "INSERT OVERWRITE TABLE t SELECT * FROM s"
    );
    let ast::Statement::Insert(i) = checked("FROM s INSERT INTO a SELECT * INSERT INTO b SELECT *")
    else {
        panic!("not an insert")
    };
    assert!(i.source.is_some() && i.clauses.len() == 2 && !i.clauses[1].overwrite);
    // The clause head ends before the query.
    let sql = "INSERT OVERWRITE t PARTITION (c = 'Us') IF NOT EXISTS (a) SELECT 1";
    let ast::Statement::Insert(i) = checked(sql) else {
        panic!("not an insert")
    };
    assert!(i.clauses[0].if_not_exists);
    assert_eq!(
        i.clauses[0].head_span.text(sql),
        "INSERT OVERWRITE t PARTITION (c = 'Us') IF NOT EXISTS (a)"
    );
}

/// `INSERT` errors as Spark 4.2 raises them.
#[test]
fn insert_errors() {
    // Syntax errors at Spark 4.2.0's positions: the target is required;
    // the column list or `BY NAME` follows the partition (and not both);
    // `BY NAME` precedes `REPLACE WHERE`; a multi-table insert body has no
    // `FROM`, set operator or parentheses of its own.
    for (sql, near, at) in [
        ("INSERT INTO SELECT * FROM s", "'*'", 19..20),
        (
            "INSERT INTO t (a) PARTITION (p = 1) SELECT 1",
            "'PARTITION'",
            18..27,
        ),
        ("INSERT INTO t (a, b) BY NAME SELECT 1, 2", "'BY'", 21..23),
        (
            "INSERT OVERWRITE t PARTITION (p=1) (a) IF NOT EXISTS SELECT 1",
            "'IF'",
            39..41,
        ),
        (
            "INSERT INTO t REPLACE WHERE a = 1 BY NAME SELECT 1",
            "'WHERE'",
            22..27,
        ),
        (
            "INSERT INTO t PARTITION (p=1) REPLACE WHERE a = 1 SELECT 1",
            "'REPLACE'",
            30..37,
        ),
        (
            "INSERT INTO t (a) REPLACE WHERE a = 1 SELECT 1",
            "'REPLACE'",
            18..25,
        ),
        ("FROM s SELECT * FROM s", "'FROM'", 16..20),
        (
            "FROM s INSERT INTO t SELECT a FROM x INSERT INTO u SELECT b",
            "'FROM'",
            30..34,
        ),
        ("FROM s INSERT INTO t SELECT a FROM x", "'FROM'", 30..34),
        ("FROM s INSERT INTO t SELECT a SELECT b", "'SELECT'", 30..36),
        (
            "FROM s INSERT INTO t (SELECT a) INSERT INTO u SELECT b",
            "'('",
            21..22,
        ),
    ] {
        let e = err(sql);
        assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError, "{sql}");
        assert_eq!(
            e.error.message_parameters(),
            [("error", near.to_owned()), ("hint", String::new())],
            "{sql}"
        );
        assert_eq!(e.span.range(), at, "{sql}");
    }
    // `IF NOT EXISTS` fails after parsing, over the clause head: on
    // `INTO`, and on `OVERWRITE` with a dynamic partition.
    let sql = "INSERT INTO t PARTITION (p=1) IF NOT EXISTS (a, b) SELECT 1";
    let e = err(sql);
    assert_eq!(e.error.error_class(), ErrorClass::InvalidStatementOrClause);
    assert_eq!(
        e.error.message_parameters(),
        [("operation", "INSERT INTO ... IF NOT EXISTS".to_owned())]
    );
    assert_eq!(
        e.span.text(sql),
        "INSERT INTO t PARTITION (p=1) IF NOT EXISTS (a, b)"
    );
    let sql = "INSERT INTO t PARTITION (p=1) IF NOT EXISTS SELECT 1";
    assert_eq!(
        err(sql).span.text(sql),
        "INSERT INTO t PARTITION (p=1) IF NOT EXISTS"
    );
    let sql = "INSERT OVERWRITE t PARTITION (p) IF NOT EXISTS SELECT 1";
    let e = err(sql);
    assert_eq!(e.error.error_class(), ErrorClass::LegacyErrorTemp0035);
    assert_eq!(
        e.error.message_parameters(),
        [(
            "message",
            "IF NOT EXISTS with dynamic partitions: p".to_owned()
        )]
    );
    assert_eq!(
        e.span.text(sql),
        "INSERT OVERWRITE t PARTITION (p) IF NOT EXISTS"
    );
    // A computed column is rejected at the inner `(` in the legacy mode,
    // as in Spark (whose `: missing ')'` hint this parser does not
    // reproduce; the class and position match).
    let legacy = ParserOptions {
        legacy_identifier_clause: true,
        ..ParserOptions::default()
    };
    let e = parse_statement("INSERT INTO tab (IDENTIFIER('c1')) VALUES (1)", &legacy).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 27..28);
    // `BY NAME REPLACE WHERE` parses (Spark's order).
    let ast::Statement::Insert(i) =
        checked("INSERT INTO t BY NAME REPLACE WHERE c > 0 SELECT * FROM s")
    else {
        panic!("not an insert")
    };
    assert!(i.clauses[0].replace_where.is_some() && i.clauses[0].by_name);
    // `REPLACE ON`, `REPLACE USING`, `WITH SCHEMA EVOLUTION`, aliases and
    // target options parse, keeping `WHERE` apart from `ON`.
    let ast::Statement::Insert(i) = checked("INSERT WITH SCHEMA EVOLUTION INTO t SELECT 1") else {
        panic!("not an insert")
    };
    assert!(i.clauses[0].with_schema_evolution);
    let ast::Statement::Insert(i) =
        checked("INSERT INTO t AS a WITH ('k' = 'v') BY NAME REPLACE ON a.c1 = 1 SELECT 1")
    else {
        panic!("not an insert")
    };
    let c = &i.clauses[0];
    assert!(c.replace_on.is_some() && c.replace_where.is_none());
    assert!(c.by_name && c.target_alias.is_some() && !c.target_options.is_empty());
    let ast::Statement::Insert(i) = checked("INSERT INTO t REPLACE USING (c1) SELECT 1") else {
        panic!("not an insert")
    };
    assert!(i.clauses[0].replace_using.is_some());
    // `REPLACE WHERE` rejects a target alias, as Spark does.
    // The alias is rejected before the options are read.
    for sql in [
        "INSERT INTO t AS a REPLACE WHERE c > 0 SELECT 1",
        "INSERT INTO t a WITH ('k') REPLACE WHERE c > 0 SELECT 1",
    ] {
        let e = err(sql);
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InsertReplaceWhereTableAliasNotAllowed,
            "{sql}"
        );
    }
}

/// `DELETE`, `UPDATE`, `MERGE` and `TRUNCATE` statements.
#[test]
fn delete_update_merge_truncate_statements() {
    insta::assert_snapshot!(snapshot_stmts(&[
        "DELETE FROM t WHERE 1 = 0",
        "DELETE FROM t",
        "DELETE FROM IDENTIFIER('ta' || 'b') WHERE 1 = 0",
        "UPDATE t SET c1 = 2",
        "UPDATE t SET c1 = 2, c2 = c2 + 1 WHERE c1 > 0",
        "UPDATE t AS x SET x.c1 = 2",
        "UPDATE t x SET x.c1 = 2",
        "UPDATE IDENTIFIER('ta' || 'b') SET c1 = 2",
        "MERGE INTO t AS x USING s AS y ON x.c1 = y.c1 WHEN MATCHED THEN UPDATE SET c1 = 3",
        "MERGE INTO t USING s ON t.c1 = s.c1 WHEN MATCHED THEN DELETE",
        "MERGE INTO t USING (SELECT * FROM s) AS y ON t.c1 = y.c1 WHEN NOT MATCHED THEN INSERT *",
        "MERGE INTO t USING s ON t.c1 = s.c1 WHEN NOT MATCHED BY TARGET AND s.c2 > 0 THEN INSERT (c1, c2) VALUES (1, 2)",
        "MERGE INTO t USING s ON t.c1 = s.c1 WHEN NOT MATCHED BY SOURCE THEN UPDATE SET c1 = 0",
        "MERGE INTO IDENTIFIER('ta' || 'b') AS t USING IDENTIFIER('ta' || 'b') AS s ON s.c1 = t.c1 WHEN MATCHED THEN UPDATE SET c1 = 3",
        "TRUNCATE TABLE t",
        "TRUNCATE TABLE IDENTIFIER('ta' || 'b')",
        "TRUNCATE TABLE t PARTITION (c = 'Us')",
    ]));
    let ast::Statement::Update(u) = checked("UPDATE t SET c1 = 2 WHERE c1 > 0") else {
        panic!("not an update")
    };
    assert_eq!(u.assignments.len(), 1);
    assert!(u.where_clause.is_some());
    assert_eq!(
        u.span.text("UPDATE t SET c1 = 2 WHERE c1 > 0"),
        "UPDATE t SET c1 = 2 WHERE c1 > 0"
    );
}

/// `DELETE`, `UPDATE`, `MERGE` and `TRUNCATE` errors.
#[test]
fn delete_update_merge_truncate_errors() {
    // `MERGE` needs at least one `WHEN` clause: Spark's builder rejects it
    // over the statement.
    let sql = "MERGE INTO t USING s ON t.c1 = s.c1";
    let e = err(sql);
    assert_eq!(e.error.error_class(), ErrorClass::MergeWithoutWhen);
    assert_eq!(e.span.range(), 0..sql.len());
    // Syntax errors at Spark 4.2.0's positions, with its "missing" hints.
    for (sql, near, hint, at) in [
        (
            "MERGE INTO t USING s ON true WHEN SOMETIMES THEN DELETE",
            "'SOMETIMES'",
            "",
            34..43,
        ),
        (
            "MERGE INTO t USING s ON true WHEN MATCHED THEN UPSERT",
            "'UPSERT'",
            "",
            47..53,
        ),
        // `UPDATE SET *` only after `WHEN MATCHED`, and alone.
        (
            "MERGE INTO t USING s ON t.a = s.a WHEN NOT MATCHED BY SOURCE THEN UPDATE SET *",
            "'*'",
            "",
            77..78,
        ),
        (
            "MERGE INTO t USING s ON t.a = s.a WHEN MATCHED THEN UPDATE SET *, a = 1",
            "','",
            "",
            64..65,
        ),
        // `INSERT ... VALUES` needs its column list.
        (
            "MERGE INTO t USING s ON t.a = s.a WHEN NOT MATCHED THEN INSERT VALUES (s.a)",
            "'VALUES'",
            "",
            63..69,
        ),
        (
            "MERGE INTO t USING s ON t.a = s.a WHEN NOT MATCHED THEN INSERT () VALUES ()",
            "')'",
            "",
            64..65,
        ),
        ("DELETE t", "'t'", ": missing 'FROM'", 7..8),
        ("UPDATE t c1 = 2", "'='", "", 12..13),
        ("TRUNCATE t", "'t'", ": missing 'TABLE'", 9..10),
    ] {
        let e = err(sql);
        assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError, "{sql}");
        assert_eq!(
            e.error.message_parameters(),
            [("error", near.to_owned()), ("hint", hint.to_owned())],
            "{sql}"
        );
        assert_eq!(e.span.range(), at, "{sql}");
    }
    // A `MERGE` action Spark's grammar rejects for the `WHEN` kind fails
    // at the action.
    for sql in [
        "MERGE INTO t USING s ON true WHEN MATCHED THEN INSERT *",
        "MERGE INTO t USING s ON true WHEN MATCHED THEN INSERT (c1) VALUES (1)",
        "MERGE INTO t USING s ON true WHEN NOT MATCHED THEN UPDATE SET c1 = 0",
        "MERGE INTO t USING s ON true WHEN NOT MATCHED THEN DELETE",
        "MERGE INTO t USING s ON true WHEN NOT MATCHED BY TARGET THEN UPDATE SET c1 = 0",
        "MERGE INTO t USING s ON true WHEN NOT MATCHED BY TARGET THEN DELETE",
        "MERGE INTO t USING s ON true WHEN NOT MATCHED BY SOURCE THEN INSERT *",
        "MERGE INTO t USING s ON true WHEN NOT MATCHED BY SOURCE THEN INSERT (c1) VALUES (1)",
    ] {
        let e = err(sql);
        assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError, "{sql}");
        let action = if sql.contains("UPDATE") {
            "UPDATE"
        } else if sql.contains("DELETE") {
            "DELETE"
        } else {
            "INSERT"
        };
        assert_eq!(e.span.text(sql), action, "{sql}");
    }
}

/// Every `WHEN` kind with every action Spark allows for it.
#[test]
fn merge_action_pairings() {
    use ast::{MergeAction, MergeMatched};
    let action_of = |sql: &str| {
        let ast::Statement::Merge(m) = checked(sql) else {
            panic!("{sql} is not a merge")
        };
        assert_eq!(m.clauses.len(), 1, "{sql}");
        m.clauses.into_iter().next().unwrap()
    };
    let clause = action_of("MERGE INTO t USING s ON true WHEN MATCHED THEN UPDATE SET c1 = 0");
    assert!(matches!(clause.matched, MergeMatched::Matched));
    assert!(matches!(clause.action, MergeAction::Update(_)));
    let clause = action_of("MERGE INTO t USING s ON true WHEN MATCHED THEN DELETE");
    assert!(matches!(clause.matched, MergeMatched::Matched));
    assert!(matches!(clause.action, MergeAction::Delete));
    let clause =
        action_of("MERGE INTO t USING s ON t.a = s.a WHEN MATCHED AND s.b > 1 THEN UPDATE SET *");
    assert!(matches!(clause.matched, MergeMatched::Matched));
    assert!(matches!(clause.action, MergeAction::UpdateStar));
    assert!(clause.condition.is_some());
    // `INSERT (columns) VALUES (...)` takes dotted column names.
    let clause = action_of(
        "MERGE INTO t USING s ON t.a = s.a WHEN NOT MATCHED THEN INSERT (t.a, b.c) VALUES (s.a, 1)",
    );
    let MergeAction::Insert(ast::MergeInsert::Values {
        columns, values, ..
    }) = clause.action
    else {
        panic!("not an insert")
    };
    assert_eq!(
        columns
            .iter()
            .map(|c| c
                .parts
                .iter()
                .map(|p| p.value.as_str())
                .collect::<Vec<_>>()
                .join("."))
            .collect::<Vec<_>>(),
        ["t.a", "b.c"]
    );
    assert_eq!(values.len(), 2);
    let clause = action_of("MERGE INTO t USING s ON true WHEN NOT MATCHED THEN INSERT *");
    assert!(matches!(clause.matched, MergeMatched::NotMatched));
    assert!(matches!(clause.action, MergeAction::Insert(_)));
    let clause = action_of(
        "MERGE INTO t USING s ON true WHEN NOT MATCHED BY TARGET THEN INSERT (c1) VALUES (1)",
    );
    assert!(matches!(clause.matched, MergeMatched::NotMatchedByTarget));
    assert!(matches!(clause.action, MergeAction::Insert(_)));
    let clause =
        action_of("MERGE INTO t USING s ON true WHEN NOT MATCHED BY SOURCE THEN UPDATE SET c1 = 0");
    assert!(matches!(clause.matched, MergeMatched::NotMatchedBySource));
    assert!(matches!(clause.action, MergeAction::Update(_)));
    let clause = action_of("MERGE INTO t USING s ON true WHEN NOT MATCHED BY SOURCE THEN DELETE");
    assert!(matches!(clause.matched, MergeMatched::NotMatchedBySource));
    assert!(matches!(clause.action, MergeAction::Delete));
}

/// `SHOW` statements: every shape.
#[test]
fn show_statements() {
    insta::assert_snapshot!(snapshot_stmts(&[
        "SHOW TABLES",
        "SHOW TABLES IN showdb",
        "SHOW TABLES 'show_t*'",
        "SHOW TABLES LIKE 'show_t*'",
        "SHOW TABLES IN showdb LIKE 'show_t*'",
        "SHOW TABLES IN IDENTIFIER('id' || 'ent')",
        "SHOW TABLE EXTENDED LIKE 'show_t*'",
        "SHOW TABLE EXTENDED IN showdb LIKE 'show_t*'",
        "SHOW TABLE EXTENDED LIKE 'show_t1' PARTITION (c = 'Us', d = 1)",
        "SHOW VIEWS",
        "SHOW VIEWS FROM showdb",
        "SHOW VIEWS IN showdb LIKE 'view_*'",
        "SHOW VIEWS 'view_*'",
        "SHOW COLUMNS FROM t",
        "SHOW COLUMNS IN showdb.showcolumn1 FROM baddb",
        "SHOW COLUMNS FROM IDENTIFIER('ta' || 'b')",
        "SHOW FUNCTIONS",
        "SHOW USER FUNCTIONS",
        "SHOW SYSTEM FUNCTIONS LIKE 'a*'",
        "SHOW FUNCTIONS FROM db LIKE 'a*'",
        "SHOW FUNCTIONS IN db 'a*'",
        "SHOW USER FUNCTIONS FROM db LIKE 'x'",
        "SHOW FUNCTIONS a.b",
        "SHOW FUNCTIONS LIKE a",
        "SHOW DATABASES",
        "SHOW SCHEMAS LIKE 's*'",
        "SHOW NAMESPACES",
        "SHOW CREATE TABLE t",
        "SHOW CREATE TABLE v AS SERDE",
        "SHOW CREATE TABLE IDENTIFIER('test_show')",
        "SHOW TBLPROPERTIES t",
        "SHOW TBLPROPERTIES t (k)",
        "SHOW TBLPROPERTIES tbl (\"p1\")",
        "SHOW TBLPROPERTIES IDENTIFIER('ta' || 'b')",
        "SHOW PARTITIONS t",
        "SHOW PARTITIONS t PARTITION (c = 'Us')",
        "SHOW PARTITIONS IDENTIFIER('test_show')",
        "SHOW CURRENT SCHEMA",
        "SHOW CURRENT NAMESPACE",
        "SHOW CURRENT DATABASE",
    ]));
}

/// `SHOW` errors as Spark 4.2 raises them.
#[test]
fn show_errors() {
    // `SHOW TABLE EXTENDED` names one table: the pattern is required, and
    // a partition needs one too.
    let e = err("SHOW TABLE EXTENDED");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 19..19);
    let e = err("SHOW TABLE EXTENDED PARTITION (c = 'Us', d = 1)");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 20..29);
    // A computed scope is rejected at the `(`.
    let e = err("SHOW IDENTIFIER('USER') FUNCTIONS");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 15..16);
    // Unknown shapes fail where they stand.
    let e = err("SHOW TABLE");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = err("SHOW CURRENT TABLE");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // `SHOW COLUMNS` names its table; `SHOW FUNCTIONS` reads the namespace
    // before the pattern (positions from Spark 4.2.0).
    let e = err("SHOW COLUMNS");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 12..12);
    let e = err("SHOW FUNCTIONS LIKE 'a*' FROM db");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 25..29);
    // A legacy (name) pattern beside a namespace fails over the statement.
    for sql in ["SHOW FUNCTIONS IN db a.b", "SHOW FUNCTIONS IN db LIKE a.b"] {
        let e = err(sql);
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidSqlSyntaxShowFunctionsInvalidPattern,
            "{sql}"
        );
        assert_eq!(
            e.error.message_parameters(),
            [("pattern", "`a`.`b`".to_owned())],
            "{sql}"
        );
        assert_eq!(e.span.range(), 0..sql.len(), "{sql}");
    }
}

/// Every name and partition value a command holds is checked after
/// parsing: an invalid typed literal fails as Spark 4.2.0 fails it.
#[test]
fn command_expressions_are_checked() {
    for sql in [
        "DROP VIEW IDENTIFIER(DATE '2020-02-30')",
        "SHOW COLUMNS IN IDENTIFIER(DATE '2020-02-30')",
        "CACHE TABLE IDENTIFIER(DATE '2020-02-30')",
        "UNCACHE TABLE IDENTIFIER(DATE '2020-02-30')",
        "ANALYZE TABLE IDENTIFIER(DATE '2020-02-30') COMPUTE STATISTICS",
        "SHOW PARTITIONS t PARTITION (d = DATE '2020-02-30')",
        "SHOW FUNCTIONS IN IDENTIFIER(DATE '2020-02-30')",
        "REFRESH TABLE IDENTIFIER(DATE '2020-02-30')",
        "MSCK REPAIR TABLE IDENTIFIER(DATE '2020-02-30')",
        "ALTER NAMESPACE IDENTIFIER(DATE '2020-02-30') SET PROPERTIES ('a' = 'b')",
        "SHOW TABLE EXTENDED LIKE 'x' PARTITION (d = DATE '2020-02-30')",
    ] {
        let e = err(sql);
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidTypedLiteral,
            "{sql}"
        );
        assert_eq!(e.span.text(sql), "DATE '2020-02-30'", "{sql}");
    }
}

/// `DESCRIBE` statements: every shape.
#[test]
fn describe_statements() {
    insta::assert_snapshot!(snapshot_stmts(&[
        "DESCRIBE t",
        "DESC t",
        "DESC TABLE t",
        "DESC EXTENDED t",
        "DESC FORMATTED t",
        "DESC t PARTITION (c = 'Us', d = 1)",
        "DESC FORMATTED t PARTITION (c = 'Us', d = 1)",
        "DESC TABLE EXTENDED f PARTITION (B = 'SPARK', C = TIMESTAMP '2018-11-17 13:33:33') AS JSON",
        "DESCRIBE EXTENDED t AS JSON",
        "DESCRIBE t b",
        "DESC FORMATTED t c",
        "DESC t s.field",
        "DESCRIBE EXTENDED t a.`b c`.d",
        "DESC IDENTIFIER('test_desc')",
        "DESCRIBE FUNCTION f",
        "DESCRIBE FUNCTION EXTENDED f",
        "DESCRIBE FUNCTION IDENTIFIER('ident.' || 'myDoubleAvg')",
        "DESCRIBE SCHEMA s",
        "DESCRIBE SCHEMA EXTENDED test_schema",
        "DESCRIBE DATABASE d",
        "DESCRIBE NAMESPACE n",
        "DESCRIBE QUERY SELECT * FROM t",
        "DESC SELECT * FROM t",
        "DESC WITH s AS (SELECT 'hello' AS col1) SELECT * FROM s",
        "DESCRIBE QUERY WITH s AS (SELECT * FROM t) SELECT * FROM s",
    ]));
    let ast::Statement::Describe(d) = checked("DESC FORMATTED t PARTITION (c = 'Us')") else {
        panic!("not a describe")
    };
    assert_eq!(
        d.span.text("DESC FORMATTED t PARTITION (c = 'Us')"),
        "DESC FORMATTED t PARTITION (c = 'Us')"
    );
}

/// `DESCRIBE` errors as Spark 4.2 raises them.
#[test]
fn describe_errors() {
    // `AS JSON` needs `EXTENDED` or `FORMATTED`.
    let e = err("DESCRIBE t AS JSON");
    assert_eq!(e.error.error_class(), ErrorClass::DescribeJsonNotExtended);
    assert_eq!(e.error.sql_state(), Some("0A000"));
    assert_eq!(
        e.error.message_parameters(),
        [("tableName", "t".to_owned())]
    );
    // No column with `AS JSON`.
    let e = err("DESC FORMATTED t a AS JSON");
    assert_eq!(
        e.error.error_class(),
        ErrorClass::UnsupportedFeatureDescTableColumnJson
    );
    assert_eq!(e.error.sql_state(), Some("0A000"));
    // Every partition key sets its value.
    let e = err("DESC t PARTITION (c = 'Us', d)");
    assert_eq!(
        e.error.error_class(),
        ErrorClass::InvalidSqlSyntaxEmptyPartitionValue
    );
    assert_eq!(e.error.sql_state(), Some("42000"));
    assert_eq!(
        e.error.message_parameters(),
        [("partKey", "`d`".to_owned())]
    );
    // No column with a partition.
    let e = err("DESC FORMATTED t PARTITION (x = 1) c");
    assert_eq!(
        e.error.error_class(),
        ErrorClass::UnsupportedFeatureDescTableColumnPartition
    );
    assert_eq!(e.error.sql_state(), Some("0A000"));
    // An insert is no query to describe.
    let e = err("DESCRIBE INSERT INTO t VALUES (1, 'a')");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 21..22);
    let e = err("DESCRIBE FROM t a INSERT INTO t SELECT * INSERT INTO u SELECT *");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 18..24);
    // A computed column is rejected at the `(` in the legacy mode.
    let legacy = ParserOptions {
        legacy_identifier_clause: true,
        ..ParserOptions::default()
    };
    let e = parse_statement("DESCRIBE t IDENTIFIER('c')", &legacy).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
}

/// `CACHE`, `REFRESH`, `ANALYZE`, `REPAIR`, `ADD` and `LIST` statements.
#[test]
fn cache_refresh_analyze_repair_resource_statements() {
    insta::assert_snapshot!(snapshot_stmts(&[
        "CACHE TABLE t",
        "CACHE LAZY TABLE t",
        "CACHE TABLE t AS SELECT * FROM s",
        "CACHE TABLE t OPTIONS (storageLevel 'MEMORY_AND_DISK') AS SELECT * FROM s",
        "CACHE TABLE IDENTIFIER('ta' || 'b')",
        "UNCACHE TABLE t",
        "UNCACHE TABLE IDENTIFIER('ta' || 'b')",
        "UNCACHE TABLE IF EXISTS t",
        "CLEAR CACHE",
        "REFRESH TABLE t",
        "REFRESH FUNCTION f",
        "REFRESH FUNCTION IDENTIFIER('ident.' || 'myDoubleAvg')",
        "REFRESH '/path/to/data'",
        "ANALYZE TABLE t COMPUTE STATISTICS",
        "ANALYZE TABLE t PARTITION (ds = '2017-08-01', hr = 10) COMPUTE STATISTICS",
        "ANALYZE TABLE t PARTITION (ds, hr) COMPUTE STATISTICS",
        "ANALYZE TABLE t COMPUTE STATISTICS NOSCAN",
        "ANALYZE TABLE t COMPUTE STATISTICS FOR COLUMNS c1, c2",
        "ANALYZE TABLE t COMPUTE STATISTICS FOR ALL COLUMNS",
        "MSCK REPAIR TABLE t",
        "REPAIR TABLE t",
        "REPAIR TABLE t ADD PARTITIONS",
        "REPAIR TABLE t DROP PARTITIONS",
        "MSCK REPAIR TABLE t SYNC PARTITIONS",
        "LIST JAR",
        "LIST FILE",
        "LIST FILES",
    ]));
    // Resource paths are the raw rest of the statement (comments and
    // spacing included, as Spark's `remainder` takes them), so they are
    // asserted here rather than re-spaced: split as Spark splits them,
    // quoted (unescaped) or unquoted.
    for (sql, kind, paths) in [
        (
            "ADD JAR '/path/to.jar'",
            ast::ResourceType::Jar,
            vec!["/path/to.jar"],
        ),
        (
            "ADD FILE '/path/to/file'",
            ast::ResourceType::File,
            vec!["/path/to/file"],
        ),
        (
            "ADD ARCHIVE '/path/to.tgz'",
            ast::ResourceType::Archive,
            vec!["/path/to.tgz"],
        ),
        (
            "ADD ARCHIVE /a.tgz",
            ast::ResourceType::Archive,
            vec!["/a.tgz"],
        ),
        (
            "ADD JARS '/a.jar' \"/b c.jar\" /d.jar",
            ast::ResourceType::Jar,
            vec!["/a.jar", "/b c.jar", "/d.jar"],
        ),
    ] {
        let ast::Statement::AddResource(r) = checked(sql) else {
            panic!("{sql} is not ADD")
        };
        assert_eq!(
            (r.kind, r.paths),
            (kind, paths.iter().map(|p| (*p).to_owned()).collect()),
            "{sql}"
        );
    }
    for (sql, paths) in [
        ("ADD JAR /tmp/a.jar", vec!["/tmp/a.jar"]),
        ("ADD JAR 'a' b \"c d\"", vec!["a", "b", "c d"]),
        ("ADD JAR 'a\\'b'", vec!["a'b"]),
        ("ADD JAR a;", vec!["a;"]),
        ("ADD JAR ivy://org:name:1.0", vec!["ivy://org:name:1.0"]),
        ("ADD JAR", vec![]),
    ] {
        let ast::Statement::AddResource(r) = checked(sql) else {
            panic!("{sql} is not ADD")
        };
        assert_eq!(r.paths, paths, "{sql}");
    }
    for (sql, paths) in [
        ("LIST JAR a 'b c'", vec!["a", "b c"]),
        ("LIST JAR '/path'", vec!["/path"]),
    ] {
        let ast::Statement::ListResource(r) = checked(sql) else {
            panic!("{sql} is not LIST")
        };
        assert_eq!(r.paths, paths, "{sql}");
    }
    for (sql, path) in [
        ("REFRESH /a/b;", "/a/b;"),
        ("REFRESH \"a b\"", "a b"),
        ("REFRESH '/tmp/path'", "/tmp/path"),
    ] {
        let ast::Statement::Refresh(r) = checked(sql) else {
            panic!("{sql} is not REFRESH")
        };
        assert!(
            matches!(&r.kind, ast::RefreshKind::Path { path: p, .. } if p == path),
            "{sql}"
        );
    }
    // Syntax errors at Spark 4.2.0's positions, with its "missing" hints.
    for (sql, near, hint, at) in [
        ("CACHE t", "'t'", "", 6..7),
        ("UNCACHE t", "'t'", ": missing 'TABLE'", 8..9),
        ("CLEAR TABLES", "'TABLES'", "", 6..12),
        (
            "ANALYZE TABLE t COMPUTE",
            "end of input",
            ": missing 'STATISTICS'",
            23..23,
        ),
        ("REPAIR t", "'t'", ": missing 'TABLE'", 7..8),
    ] {
        let e = err(sql);
        assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError, "{sql}");
        assert_eq!(
            e.error.message_parameters(),
            [("error", near.to_owned()), ("hint", hint.to_owned())],
            "{sql}"
        );
        assert_eq!(e.span.range(), at, "{sql}");
    }
    // Errors Spark's builder raises over the statement: an unknown
    // resource kind, and an empty or spaced unquoted `REFRESH` path (a
    // `REFRESH TABLE` with more after the name is such a path).
    for (sql, class, message) in [
        (
            "ADD FOLDER '/x'",
            ErrorClass::LegacyErrorTemp0035,
            "ADD with resource type 'folder'",
        ),
        (
            "LIST FOLDER",
            ErrorClass::LegacyErrorTemp0035,
            "LIST with resource type 'folder'",
        ),
        (
            "REFRESH",
            ErrorClass::LegacyErrorTemp0064,
            "Resource paths cannot be empty in REFRESH statements. Use / to match everything",
        ),
        (
            "REFRESH TABLE t ASYNC",
            ErrorClass::LegacyErrorTemp0064,
            "REFRESH statements cannot contain ' ', '\\n', '\\r', '\\t' inside unquoted resource paths",
        ),
        (
            "REFRESH RESOURCE 'x'",
            ErrorClass::LegacyErrorTemp0064,
            "REFRESH statements cannot contain ' ', '\\n', '\\r', '\\t' inside unquoted resource paths",
        ),
    ] {
        let e = err(sql);
        assert_eq!(e.error.error_class(), class, "{sql}");
        let key = if class == ErrorClass::LegacyErrorTemp0035 {
            "message"
        } else {
            "msg"
        };
        assert_eq!(
            e.error.message_parameters(),
            [(key, message.to_owned())],
            "{sql}"
        );
        assert_eq!(e.span.range(), 0..sql.len(), "{sql}");
    }
}

/// Commands under `EXPLAIN` parse as explained commands.
#[test]
fn explained_commands() {
    use ast::Statement;
    for sql in [
        "EXPLAIN INSERT INTO t SELECT * FROM s",
        "EXPLAIN DESCRIBE t",
        "EXPLAIN DESC QUERY SELECT * FROM t WHERE key > 0",
        "EXPLAIN SHOW TABLES",
        "EXPLAIN ALTER TABLE t ADD COLUMN c INT",
        "EXPLAIN TRUNCATE TABLE t",
        "EXPLAIN CACHE TABLE t AS SELECT * FROM s",
    ] {
        let Statement::Explain(e) = checked(sql) else {
            panic!("{sql} does not parse as EXPLAIN")
        };
        assert!(!matches!(*e.statement, Statement::Query(_)), "{sql}");
        assert_eq!(e.span.text(sql), sql, "{sql}");
    }
    // A two-table insert explains as one statement.
    let Statement::Explain(e) =
        checked("EXPLAIN FROM s INSERT INTO a SELECT * INSERT INTO b SELECT *")
    else {
        panic!("not explained")
    };
    assert!(matches!(*e.statement, Statement::Insert(_)));
}
