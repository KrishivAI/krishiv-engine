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
        "ALTER TABLE t RENAME a TO b",
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
        "ALTER TABLE t DROP IF EXISTS PARTITION (c = 'Us') PARTITION (d = 2) PURGE",
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
    // Unknown actions fail where they stand.
    let e = err("ALTER TABLE t SET FOO 'x'");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = err("ALTER TABLE t PARTITION (c = 1) SET FOO 'x'");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
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
        "CREATE TEMPORARY VIEW v (a) USING parquet OPTIONS (k = 'v')",
        "CREATE TEMPORARY VIEW v (e INT) USING parquet",
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

/// Full `CREATE VIEW` errors as Spark 4.2 raises them.
#[test]
fn create_full_view_errors() {
    // A datasource view is always temporary.
    let e = err("CREATE VIEW v USING parquet AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // `GLOBAL` needs `TEMPORARY`.
    let e = err("CREATE GLOBAL VIEW v AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // No query and no provider.
    let e = err("CREATE VIEW v");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // A column type reads only for datasource (`USING`) views.
    let e = err("CREATE VIEW v (a INT) AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.range(), 17..20);
    // A qualified temporary name is rejected with its qualifier.
    let e = err("CREATE TEMPORARY VIEW a.b AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::InvalidTempObjQualifier);
    assert_eq!(
        e.error.message_parameters(),
        [
            ("objectType", "view".to_owned()),
            ("objectName", "`a`.`b`".to_owned()),
            ("qualifier", "`a`".to_owned()),
        ]
    );
    // `OR REPLACE` with `IF NOT EXISTS`, and `IF NOT EXISTS` alone, fail
    // on a temporary view as they do on the legacy one.
    let e = err("CREATE OR REPLACE TEMPORARY VIEW IF NOT EXISTS v AS SELECT 1");
    assert_eq!(
        e.error.error_class(),
        ErrorClass::CreateViewWithIfNotExistsAndReplace
    );
    let e = err("CREATE TEMPORARY VIEW IF NOT EXISTS v AS SELECT 1");
    assert_eq!(e.error.error_class(), ErrorClass::LegacyErrorTemp0053);
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
        checked("FROM s INSERT INTO a SELECT * FROM s"),
        ast::Statement::Insert(_)
    ));
    assert!(!matches!(
        checked("FROM s INSERT INTO a SELECT * FROM s"),
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
        "FROM s INSERT INTO a SELECT * FROM s INSERT INTO b SELECT * FROM s",
        "FROM s INSERT OVERWRITE TABLE a SELECT * FROM s INSERT INTO b PARTITION (d = 1) SELECT * FROM s",
        "FROM s INSERT INTO a SELECT * FROM s UNION ALL SELECT * FROM t INSERT INTO b SELECT 1",
        "FROM s INSERT INTO a SELECT * FROM x JOIN y INSERT INTO b SELECT 2",
    ]));
    let ast::Statement::Insert(i) = checked("INSERT OVERWRITE TABLE t SELECT * FROM s") else {
        panic!("not an insert")
    };
    assert!(i.source.is_none() && i.clauses.len() == 1 && i.clauses[0].overwrite);
    assert_eq!(
        i.span.text("INSERT OVERWRITE TABLE t SELECT * FROM s"),
        "INSERT OVERWRITE TABLE t SELECT * FROM s"
    );
    let ast::Statement::Insert(i) =
        checked("FROM s INSERT INTO a SELECT * FROM s INSERT INTO b SELECT * FROM s")
    else {
        panic!("not an insert")
    };
    assert!(i.source.is_some() && i.clauses.len() == 2 && !i.clauses[1].overwrite);
}

/// `INSERT` errors as Spark 4.2 raises them.
#[test]
fn insert_errors() {
    // The target is required.
    let e = err("INSERT INTO SELECT * FROM s");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
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
    // A multi-table insert needs at least one `INSERT` clause.
    let e = err("FROM s SELECT * FROM s");
    assert!(matches!(
        e.error.error_class(),
        ErrorClass::ParseSyntaxError
    ));
    // `REPLACE WHERE` comes before `BY NAME`: the reverse order fails at
    // `REPLACE`.
    let sql = "INSERT INTO t BY NAME REPLACE WHERE c > 0 SELECT * FROM s";
    let e = err(sql);
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(e.span.text(sql), "REPLACE");
    // The documented order parses.
    let ast::Statement::Insert(i) =
        checked("INSERT INTO t REPLACE WHERE c > 0 BY NAME SELECT * FROM s")
    else {
        panic!("not an insert")
    };
    assert!(i.clauses[0].replace_where.is_some() && i.clauses[0].by_name);
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
    // `MERGE` needs at least one `WHEN` clause.
    let e = err("MERGE INTO t USING s ON t.c1 = s.c1");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    // Unknown `WHEN` shapes fail where they stand.
    let e = err("MERGE INTO t USING s ON true WHEN SOMETIMES THEN DELETE");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = err("MERGE INTO t USING s ON true WHEN MATCHED THEN UPSERT");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
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
    // Missing pieces fail where they stand.
    let e = err("DELETE t");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = err("UPDATE t c1 = 2");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
    let e = err("TRUNCATE t");
    assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError);
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
        "SHOW FUNCTIONS LIKE 'a*' FROM db",
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
        "CLEAR CACHE",
        "REFRESH TABLE t",
        "REFRESH TABLE t ASYNC",
        "REFRESH FUNCTION f",
        "REFRESH FUNCTION IDENTIFIER('ident.' || 'myDoubleAvg')",
        "REFRESH RESOURCE '/path/to.jar'",
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
        "ADD JAR '/path/to.jar'",
        "ADD FILE '/path/to/file'",
        "ADD ARCHIVE '/path/to.tgz'",
        "LIST JAR",
        "LIST JAR '/path'",
        "LIST FILE",
    ]));
    // Unknown shapes fail where they stand.
    for sql in [
        "CACHE t",
        "UNCACHE t",
        "CLEAR TABLES",
        "REFRESH",
        "ANALYZE TABLE t COMPUTE",
        "REPAIR t",
        "ADD FOLDER '/x'",
        "LIST FOLDER",
    ] {
        let e = err(sql);
        assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError, "{sql}");
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
        checked("EXPLAIN FROM s INSERT INTO a SELECT * FROM s INSERT INTO b SELECT * FROM s")
    else {
        panic!("not explained")
    };
    assert!(matches!(*e.statement, Statement::Insert(_)));
}
