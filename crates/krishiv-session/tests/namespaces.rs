// SPDX-License-Identifier: Apache-2.0
//! `CREATE|DROP DATABASE|SCHEMA|NAMESPACE`, `USE` and `SHOW CURRENT NAMESPACE` through `Session::sql`.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
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

fn class(s: &Session, sql: &str) -> String {
    block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_err()
        .error_class()
        .name()
        .to_owned()
}

/// Every cell of `sql` as strings, row by row.
fn rows(s: &Session, sql: &str) -> Vec<Vec<String>> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(i) {
                            "~".to_owned()
                        } else {
                            match c.data_type() {
                                ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
                                ArrowType::Boolean => c.as_boolean().value(i).to_string(),
                                ArrowType::Int32 => c
                                    .as_primitive::<arrow_array::types::Int32Type>()
                                    .value(i)
                                    .to_string(),
                                _ => c.as_string::<i32>().value(i).to_owned(),
                            }
                        }
                    })
                    .collect(),
            );
        }
    }
    out
}

#[test]
fn use_changes_where_bare_names_resolve() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    assert_eq!(
        rows(&s, "SHOW CURRENT NAMESPACE"),
        [["spark_catalog", "default"]]
    );
    run(&s, "CREATE DATABASE db1");
    run(&s, "USE db1");
    assert_eq!(rows(&s, "SHOW CURRENT SCHEMA"), [["spark_catalog", "db1"]]);
    run(&s, "CREATE TABLE t (a INT)");
    assert!(dir.path().join("db1.db").join("t").is_dir());
    run(&s, "INSERT INTO t VALUES (1)");
    assert_eq!(rows(&s, "SELECT CAST(a AS STRING) FROM db1.t"), [["1"]]);
    assert_eq!(rows(&s, "SELECT CAST(a AS STRING) FROM t"), [["1"]]);
    run(&s, "USE NAMESPACE default");
    assert_eq!(class(&s, "SELECT * FROM t"), "TABLE_OR_VIEW_NOT_FOUND");
    assert_eq!(
        rows(&s, "SELECT CAST(a AS STRING) FROM spark_catalog.db1.t"),
        [["1"]]
    );
    run(&s, "USE DATABASE db1");
    run(&s, "DROP TABLE t");
    assert!(!dir.path().join("db1.db").join("t").exists());
}

#[test]
fn namespace_errors() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE SCHEMA IF NOT EXISTS s1 COMMENT 'c' WITH DBPROPERTIES (k = 'v')",
    );
    assert_eq!(class(&s, "CREATE NAMESPACE s1"), "SCHEMA_ALREADY_EXISTS");
    run(&s, "CREATE NAMESPACE IF NOT EXISTS s1");
    assert_eq!(class(&s, "USE missing"), "SCHEMA_NOT_FOUND");
    assert_eq!(class(&s, "DROP DATABASE missing"), "SCHEMA_NOT_FOUND");
    run(&s, "DROP DATABASE IF EXISTS missing");
    assert_eq!(
        class(&s, "CREATE TABLE missing.t (a INT)"),
        "SCHEMA_NOT_FOUND"
    );
    assert_eq!(
        class(&s, "DROP DATABASE default"),
        "UNSUPPORTED_FEATURE.DROP_DATABASE"
    );
    assert_eq!(class(&s, "USE a.b.c"), "REQUIRES_SINGLE_PART_NAMESPACE");
    assert_eq!(
        class(&s, "USE other_catalog.s1"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    run(&s, "USE spark_catalog.s1");
    assert_eq!(
        rows(&s, "SHOW CURRENT NAMESPACE"),
        [["spark_catalog", "s1"]]
    );
}

#[test]
fn dropping_a_namespace_with_tables() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE DATABASE d");
    run(&s, "CREATE TABLE d.t (a INT)");
    run(&s, "INSERT INTO d.t VALUES (1)");
    assert_eq!(class(&s, "DROP DATABASE d"), "SCHEMA_NOT_EMPTY");
    assert_eq!(class(&s, "DROP DATABASE d RESTRICT"), "SCHEMA_NOT_EMPTY");
    run(&s, "USE d");
    run(&s, "DROP DATABASE d CASCADE");
    assert!(!dir.path().join("d.db").join("t").exists());
    // The dropped namespace was current: back in `default`.
    assert_eq!(
        rows(&s, "SHOW CURRENT NAMESPACE"),
        [["spark_catalog", "default"]]
    );
    assert_eq!(class(&s, "SELECT * FROM d.t"), "TABLE_OR_VIEW_NOT_FOUND");
    run(&s, "CREATE DATABASE d");
    run(&s, "DROP SCHEMA d");
}

#[test]
fn namespace_names_that_could_be_paths_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    assert_eq!(
        class(&s, "CREATE DATABASE `a/b`"),
        "INVALID_SCHEMA_OR_RELATION_NAME"
    );
    assert_eq!(
        class(&s, "CREATE SCHEMA `..`"),
        "INVALID_ATTRIBUTE_NAME_SYNTAX"
    );
    assert_eq!(
        class(&s, "CREATE NAMESPACE `x y`"),
        "INVALID_SCHEMA_OR_RELATION_NAME"
    );
}

#[cfg(unix)]
#[test]
fn drop_cascade_never_deletes_outside_the_warehouse() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("keep.txt"), "x").unwrap();
    let s = session(dir.path());
    run(&s, "CREATE DATABASE d");
    run(&s, "CREATE TABLE d.a (a INT)");
    run(&s, "CREATE TABLE d.b (a INT)");
    // One table's directory is swapped for a link that leaves the warehouse.
    let b = dir.path().join("d.db").join("b");
    std::fs::remove_dir_all(&b).unwrap();
    std::os::unix::fs::symlink(outside.path(), &b).unwrap();
    assert_eq!(class(&s, "DROP DATABASE d CASCADE"), "INTERNAL_ERROR");
    // Nothing changed: both tables are still there, and so is the other directory.
    assert!(dir.path().join("d.db").join("a").is_dir());
    assert!(outside.path().join("keep.txt").is_file());
    run(&s, "INSERT INTO d.a VALUES (1)");
}

#[test]
fn a_global_temporary_view_lives_in_global_temp() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE GLOBAL TEMP VIEW gv(a, b) AS SELECT 1, 'x'");
    assert_eq!(
        rows(&s, "SELECT b, a FROM global_temp.gv"),
        vec![vec!["x".to_owned(), "1".to_owned()]]
    );
    // Not a temporary view, and not listed with them.
    assert_eq!(class(&s, "SELECT * FROM gv"), "TABLE_OR_VIEW_NOT_FOUND");
    assert!(rows(&s, "SHOW TABLES").is_empty());
    assert_eq!(
        rows(&s, "SHOW VIEWS IN global_temp"),
        vec![vec![
            "global_temp".to_owned(),
            "gv".to_owned(),
            "true".to_owned()
        ]]
    );
    assert_eq!(
        rows(&s, "SHOW COLUMNS IN gv FROM global_temp"),
        vec![vec!["a".to_owned()], vec!["b".to_owned()]]
    );
    assert_eq!(
        class(&s, "CREATE GLOBAL TEMP VIEW gv AS SELECT 2"),
        "TEMP_TABLE_OR_VIEW_ALREADY_EXISTS"
    );
    run(&s, "CREATE OR REPLACE GLOBAL TEMP VIEW gv AS SELECT 2 AS c");
    assert_eq!(
        rows(&s, "SELECT c FROM global_temp.gv"),
        vec![vec!["2".to_owned()]]
    );
    assert_eq!(
        class(&s, "CREATE VIEW pv AS SELECT * FROM global_temp.gv"),
        "INVALID_TEMP_OBJ_REFERENCE"
    );
    run(&s, "DROP VIEW global_temp.gv");
    assert_eq!(
        class(&s, "SELECT * FROM global_temp.gv"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    assert_eq!(
        class(&s, "DROP VIEW global_temp.gv"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    run(&s, "DROP VIEW IF EXISTS global_temp.gv");
}

/// A global view is read through `global_temp.v.col` as well as `v.col`, and a view that reads itself
/// (directly or through another global view) is `RECURSIVE_VIEW` before the catalog changes.
#[test]
fn global_views_qualify_columns_and_refuse_cycles() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE GLOBAL TEMP VIEW v AS SELECT 1 AS x");
    for q in [
        "SELECT global_temp.v.x FROM global_temp.v",
        "SELECT v.x FROM global_temp.v",
        "SELECT x FROM global_temp.v",
    ] {
        assert_eq!(rows(&s, q), vec![vec!["1".to_owned()]], "{q}");
    }
    assert_eq!(
        class(
            &s,
            "CREATE OR REPLACE GLOBAL TEMP VIEW v AS SELECT * FROM global_temp.v"
        ),
        "RECURSIVE_VIEW"
    );
    // The old definition survived.
    assert_eq!(
        rows(&s, "SELECT x FROM global_temp.v"),
        vec![vec!["1".to_owned()]]
    );
    run(
        &s,
        "CREATE GLOBAL TEMP VIEW w AS SELECT x FROM global_temp.v",
    );
    assert_eq!(
        class(
            &s,
            "CREATE OR REPLACE GLOBAL TEMP VIEW v AS SELECT x FROM global_temp.w"
        ),
        "RECURSIVE_VIEW"
    );
}

/// `global_temp` is reserved, a global view describes like a table, and dropping a cached global view
/// evicts what was cached from it.
#[test]
fn global_temp_is_reserved_and_global_views_describe_and_uncache() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    assert_eq!(
        class(&s, "CREATE DATABASE global_temp"),
        "SCHEMA_ALREADY_EXISTS"
    );
    run(&s, "CREATE DATABASE IF NOT EXISTS global_temp");
    run(&s, "CREATE GLOBAL TEMP VIEW gv AS SELECT 1 AS x");
    assert_eq!(
        rows(&s, "DESCRIBE global_temp.gv")[0][..2],
        ["x".to_owned(), "int".to_owned()]
    );
    run(&s, "CACHE TABLE c AS SELECT x FROM global_temp.gv");
    run(&s, "DROP VIEW global_temp.gv");
    run(&s, "CREATE GLOBAL TEMP VIEW gv AS SELECT 2 AS x");
    assert_eq!(
        rows(&s, "SELECT x FROM global_temp.gv"),
        vec![vec!["2".to_owned()]]
    );
}

/// Statement introspection reads the statement the session would run, after `${name}` substitution.
#[test]
fn statement_info_substitutes_variables() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "SET value = 5");
    let info = block_on(s.statement_info("SELECT ${value}"));
    assert!(info.is_ok(), "{info:?}");
}
