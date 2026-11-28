// SPDX-License-Identifier: Apache-2.0
//! Names with more parts than the session catalog holds, and `SHOW COLUMNS` with a namespace.
//! Expected errors are those of Spark 4.2.0 and 4.3.0 runs of the same statements.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn session(dir: &std::path::Path) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.to_str().unwrap())
        .unwrap();
    s
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

/// The error class and message of `sql`, which must fail.
fn error(s: &Session, sql: &str) -> (String, String) {
    let e = block_on(async { s.sql(sql).await?.collect().await })
        .map(|_| ())
        .expect_err(sql);
    (e.error_class().name().to_owned(), e.to_string())
}

/// Asserts `sql` is `REQUIRES_SINGLE_PART_NAMESPACE` naming exactly `identifier`.
fn nested(s: &Session, sql: &str, identifier: &str) {
    let (class, message) = error(s, sql);
    assert_eq!(class, "REQUIRES_SINGLE_PART_NAMESPACE", "{sql}: {message}");
    let want = format!("but got identifier {identifier}.");
    assert!(message.contains(&want), "{sql}: {message}");
}

#[test]
fn a_table_statement_names_the_parts_after_the_session_catalog() {
    let s = Session::new().unwrap();
    for statement in [
        "DESCRIBE TABLE {}",
        "SELECT * FROM {}",
        "DROP TABLE {}",
        "DROP TABLE IF EXISTS {}",
        "DROP VIEW IF EXISTS {}",
        "SHOW COLUMNS IN {}",
        "SHOW CREATE TABLE {}",
        "SHOW PARTITIONS {}",
        "SHOW TBLPROPERTIES {}",
        "TRUNCATE TABLE {}",
        "ANALYZE TABLE {} COMPUTE STATISTICS",
        "REFRESH TABLE {}",
        "CACHE TABLE {}",
        "UNCACHE TABLE {}",
        "UNCACHE TABLE IF EXISTS {}",
        "MSCK REPAIR TABLE {}",
        "ALTER TABLE {} SET TBLPROPERTIES (x = '1')",
        "INSERT INTO {} VALUES (1)",
        "CREATE TABLE {} (x INT) USING parquet",
        "CREATE VIEW {} AS SELECT 1",
    ] {
        for name in ["a.b.Tt", "spark_catalog.a.b.Tt", "Spark_Catalog.a.b.Tt"] {
            nested(&s, &statement.replace("{}", name), "`a`.`b`.`Tt`");
        }
    }
}

#[test]
fn a_schema_name_of_two_parts_is_refused() {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE SCHEMA Aa.Bb",
        "CREATE SCHEMA IF NOT EXISTS Aa.Bb",
        "CREATE DATABASE spark_catalog.Aa.Bb",
    ] {
        nested(&s, sql, "`Aa`.`Bb`");
    }
    nested(&s, "CREATE SCHEMA a.b.c", "`a`.`b`.`c`");
    nested(&s, "CREATE SCHEMA spark_catalog.a.b", "`a`.`b`");
    // `system` is a catalog, and a reserved one.
    assert_eq!(
        error(&s, "CREATE SCHEMA system.x").0,
        "KRISHIV_RESERVED_NAME"
    );
}

#[test]
fn a_function_statement_names_the_parts_after_the_session_catalog() {
    let s = Session::new().unwrap();
    for statement in [
        "CREATE FUNCTION {}() RETURNS INT RETURN 1",
        "CREATE FUNCTION IF NOT EXISTS {}() RETURNS INT RETURN 1",
        "CREATE OR REPLACE FUNCTION {}() RETURNS INT RETURN 1",
        "DROP FUNCTION {}",
        "DROP FUNCTION IF EXISTS {}",
        "DESCRIBE FUNCTION {}",
        "REFRESH FUNCTION {}",
        "SELECT {}()",
    ] {
        for name in ["a.b.f", "spark_catalog.a.b.f"] {
            nested(&s, &statement.replace("{}", name), "`a`.`b`.`f`");
        }
    }
    // A built-in's name under another namespace is that namespace's function.
    for statement in [
        "DESCRIBE FUNCTION a.b.abs",
        "DROP FUNCTION a.b.abs",
        "REFRESH FUNCTION a.b.abs",
    ] {
        nested(&s, statement, "`a`.`b`.`abs`");
    }
    // The built-in and session qualifiers still name functions.
    run(&s, "DESCRIBE FUNCTION system.builtin.abs");
    run(&s, "REFRESH FUNCTION system.builtin.abs");
    run(&s, "REFRESH FUNCTION default.abs");
}

#[test]
fn show_columns_names_the_table_in_the_from_namespace() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE tt (x INT) USING parquet");
    nested(&s, "SHOW COLUMNS IN Tt FROM a.b", "`a`.`b`.`Tt`");
    nested(&s, "SHOW COLUMNS IN Tt FROM a.b.c", "`a`.`b`.`c`.`Tt`");
    nested(
        &s,
        "SHOW COLUMNS IN Tt FROM spark_catalog.a.b",
        "`a`.`b`.`Tt`",
    );
    nested(&s, "SHOW COLUMNS IN Tt FROM spark_catalog", "`Tt`");
    nested(&s, "SHOW COLUMNS IN a.b.Tt FROM Dummy", "`a`.`b`.`Tt`");
    // A missing table is named as written, with the namespace.
    for (sql, relation) in [
        ("SHOW COLUMNS IN Tt FROM Dummy", "`Dummy`.`Tt`"),
        (
            "SHOW COLUMNS IN Tt FROM spark_catalog.Dummy",
            "`spark_catalog`.`Dummy`.`Tt`",
        ),
        ("SHOW COLUMNS IN Dummy.Tt FROM a.b", "`Dummy`.`Tt`"),
        (
            "SHOW COLUMNS IN Dummy.Tt FROM spark_catalog",
            "`Dummy`.`Tt`",
        ),
    ] {
        let (class, message) = error(&s, sql);
        assert_eq!(class, "TABLE_OR_VIEW_NOT_FOUND", "{sql}: {message}");
        assert!(
            message.contains(&format!("view {relation} cannot")),
            "{sql}: {message}"
        );
    }
}

#[test]
fn show_columns_refuses_a_namespace_that_is_not_the_tables_database() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE tt (x INT) USING parquet");
    run(&s, "CREATE SCHEMA other");
    for (sql, named) in [
        ("SHOW COLUMNS IN default.tt FROM a.b", "`a`"),
        ("SHOW COLUMNS IN default.tt FROM other", "`other`"),
        ("SHOW COLUMNS IN default.tt FROM Dummy", "`Dummy`"),
        (
            "SHOW COLUMNS IN default.tt FROM spark_catalog",
            "`spark_catalog`",
        ),
        (
            "SHOW COLUMNS IN default.tt FROM spark_catalog.default",
            "`spark_catalog`",
        ),
        (
            "SHOW COLUMNS IN tt FROM spark_catalog.default",
            "`spark_catalog`",
        ),
    ] {
        let (class, message) = error(&s, sql);
        assert_eq!(
            class, "SHOW_COLUMNS_WITH_CONFLICT_NAMESPACE",
            "{sql}: {message}"
        );
        let want = format!("{named} != `default`.");
        assert!(message.contains(&want), "{sql}: {message}");
    }
    for sql in [
        "SHOW COLUMNS IN tt FROM default",
        "SHOW COLUMNS IN default.tt FROM DEFAULT",
        "SHOW COLUMNS IN spark_catalog.default.tt FROM default",
        // Only the first part of the namespace is compared (Spark 3.5.9 and 4.3.0 runs).
        "SHOW COLUMNS IN default.tt FROM default.x",
        "SHOW COLUMNS IN spark_catalog.default.tt FROM default.x",
    ] {
        run(&s, sql);
    }
}

#[test]
fn show_columns_compares_the_namespace_by_the_sessions_case_rule() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE SCHEMA Db");
    run(&s, "CREATE TABLE Db.t (c INT) USING parquet");
    run(&s, "CREATE TABLE tt (x INT) USING parquet");
    run(&s, "SHOW COLUMNS IN Db.t FROM db");
    run(&s, "SET spark.sql.caseSensitive = true");
    run(&s, "SHOW COLUMNS IN db.t FROM db");
    for (sql, want) in [
        ("SHOW COLUMNS IN db.t FROM Db", "`Db` != `db`."),
        (
            "SHOW COLUMNS IN default.tt FROM DEFAULT",
            "`DEFAULT` != `default`.",
        ),
    ] {
        let (class, message) = error(&s, sql);
        assert_eq!(
            class, "SHOW_COLUMNS_WITH_CONFLICT_NAMESPACE",
            "{sql}: {message}"
        );
        assert!(message.contains(want), "{sql}: {message}");
    }
}
