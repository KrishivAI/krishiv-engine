// SPDX-License-Identifier: Apache-2.0
//! `Session::catalog()` (Spark's `spark.catalog`, T082): the PySpark `Catalog` doc examples, as Rust.
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

#[test]
fn current_database_and_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let c = s.catalog();
    assert_eq!(c.current_database(), "default");
    assert_eq!(c.current_catalog(), "spark_catalog");
    run(&s, "CREATE DATABASE db1");
    block_on(c.set_current_database("db1")).unwrap();
    assert_eq!(c.current_database(), "db1");
    let e = block_on(c.set_current_database("nope")).unwrap_err();
    assert_eq!(e.error_class().name(), "SCHEMA_NOT_FOUND");
    assert_eq!(c.list_catalogs(), ["spark_catalog", "system"]);
}

#[test]
fn databases_list_get_exists() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE DATABASE db1 COMMENT 'first'");
    let c = s.catalog();
    let names: Vec<_> = c.list_databases(None).into_iter().map(|d| d.name).collect();
    assert_eq!(names, ["db1", "default"]);
    let d = c.get_database("DB1").unwrap();
    assert_eq!(d.name, "db1");
    assert_eq!(d.catalog, "spark_catalog");
    assert_eq!(d.description.as_deref(), Some("first"));
    assert!(d.location_uri.ends_with("db1.db"));
    assert!(c.database_exists("default"));
    assert!(!c.database_exists("zzz"));
    assert_eq!(c.list_databases(Some("d*")).len(), 2);
    assert_eq!(c.list_databases(Some("db*")).len(), 1);
}

#[test]
fn tables_list_get_exists() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE tbl1 (name STRING, age INT) USING parquet");
    run(&s, "CREATE VIEW pv AS SELECT 1 AS x");
    run(&s, "CREATE TEMPORARY VIEW tab1 AS SELECT 1 AS x");
    run(&s, "CREATE GLOBAL TEMPORARY VIEW gv AS SELECT 1 AS x");
    let c = s.catalog();
    let t = c.list_tables(None, None).unwrap();
    let got: Vec<_> = t
        .iter()
        .map(|t| (t.name.as_str(), t.table_type.as_str(), t.is_temporary))
        .collect();
    assert_eq!(
        got,
        [
            ("pv", "VIEW", false),
            ("tab1", "TEMPORARY", true),
            ("tbl1", "MANAGED", false)
        ]
    );
    let tbl1 = c.get_table("tbl1").unwrap();
    assert_eq!(tbl1.catalog.as_deref(), Some("spark_catalog"));
    assert_eq!(tbl1.namespace, Some(vec!["default".to_owned()]));
    assert_eq!(tbl1.description, None);
    let tab1 = c.get_table("tab1").unwrap();
    assert_eq!((tab1.catalog, tab1.namespace), (None, Some(vec![])));
    let gv = c.get_table("global_temp.gv").unwrap();
    assert_eq!(gv.namespace, Some(vec!["global_temp".to_owned()]));
    assert_eq!(c.list_tables(Some("global_temp"), None).unwrap().len(), 1);
    assert_eq!(c.list_tables(None, Some("tb*")).unwrap().len(), 1);
    assert!(c.table_exists("default.tbl1"));
    assert!(c.table_exists("spark_catalog.default.tbl1"));
    assert!(!c.table_exists("missing"));
    assert_eq!(
        c.get_table("missing").unwrap_err().error_class().name(),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    assert_eq!(
        c.list_tables(Some("nope"), None)
            .unwrap_err()
            .error_class()
            .name(),
        "SCHEMA_NOT_FOUND"
    );
}

#[test]
fn columns_carry_partition_and_bucket_flags() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (id INT COMMENT 'key', name STRING, p STRING) USING parquet \
         PARTITIONED BY (p)",
    );
    run(&s, "CREATE TEMPORARY VIEW v AS SELECT 1 AS x, 'a' AS y");
    let c = s.catalog();
    let cols = block_on(c.list_columns("t")).unwrap();
    let got: Vec<_> = cols
        .iter()
        .map(|c| {
            (
                c.name.as_str(),
                c.data_type.as_str(),
                c.nullable,
                c.is_partition,
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("id", "int", true, false),
            ("name", "string", true, false),
            ("p", "string", true, true)
        ]
    );
    assert_eq!(cols[0].description.as_deref(), Some("key"));
    let v = block_on(c.list_columns("v")).unwrap();
    assert_eq!(
        v.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["x", "y"]
    );
    assert_eq!(
        block_on(c.list_columns("nope"))
            .unwrap_err()
            .error_class()
            .name(),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn functions_list_get_exists() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TEMPORARY FUNCTION twice(x INT) RETURNS INT RETURN x * 2",
    );
    let c = s.catalog();
    assert!(c.function_exists("abs"));
    assert!(c.function_exists("TWICE"));
    assert!(!c.function_exists("no_such_fn"));
    let f = c.get_function("abs").unwrap();
    assert_eq!((f.catalog, f.namespace, f.is_temporary), (None, None, true));
    assert!(
        c.list_functions(Some("tw*"))
            .iter()
            .any(|f| f.name == "twice")
    );
    assert_eq!(
        c.get_function("no_such_fn")
            .unwrap_err()
            .error_class()
            .name(),
        "UNRESOLVED_ROUTINE"
    );
}

#[test]
fn create_table_managed_with_a_schema_and_external_from_files() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let c = s.catalog();
    block_on(c.create_table(
        "m",
        Some("parquet"),
        Some("a INT, b STRING"),
        Some("a managed table"),
        &[],
    ))
    .unwrap();
    let t = c.get_table("m").unwrap();
    assert_eq!(t.table_type, "MANAGED");
    assert_eq!(t.description.as_deref(), Some("a managed table"));
    let cols = block_on(c.list_columns("m")).unwrap();
    assert_eq!(
        cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    // a second create is the statement's error
    assert_eq!(
        block_on(c.create_table("m", Some("parquet"), Some("a INT"), None, &[]))
            .unwrap_err()
            .error_class()
            .name(),
        "TABLE_OR_VIEW_ALREADY_EXISTS"
    );
    // an external table over files written by another table: its schema comes from the files
    run(&s, "INSERT INTO m VALUES (1, 'x'), (2, 'y')");
    let location = dir.path().join("m");
    let path = location.to_str().unwrap();
    block_on(c.create_table("ext", Some("parquet"), None, None, &[("path", path)])).unwrap();
    assert_eq!(c.get_table("ext").unwrap().table_type, "EXTERNAL");
    let rows = block_on(async {
        s.sql("SELECT concat_ws('|', a, b) FROM ext ORDER BY a")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert_eq!(
        rows.iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );
    let cols = block_on(c.list_columns("ext")).unwrap();
    assert_eq!(
        cols.iter()
            .map(|c| (c.name.as_str(), c.data_type.as_str()))
            .collect::<Vec<_>>(),
        [("a", "int"), ("b", "string")]
    );
}

#[test]
fn drop_views_and_refresh() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TEMPORARY VIEW tab1 AS SELECT 1 AS x");
    run(&s, "CREATE GLOBAL TEMPORARY VIEW gv AS SELECT 1 AS x");
    run(&s, "CREATE TABLE t (a INT) USING parquet");
    let c = s.catalog();
    assert!(c.drop_temp_view("tab1"));
    assert!(!c.drop_temp_view("tab1"));
    assert!(c.drop_global_temp_view("gv"));
    assert!(!c.drop_global_temp_view("gv"));
    assert!(!block_on(c.is_cached("t")).unwrap());
    block_on(c.cache_table("t")).unwrap();
    assert!(block_on(c.is_cached("t")).unwrap(), "cacheTable caches it");
    block_on(c.refresh_table("t")).unwrap();
    block_on(c.uncache_table("t")).unwrap();
    assert!(
        !block_on(c.is_cached("t")).unwrap(),
        "uncacheTable drops it"
    );
    assert_eq!(
        block_on(c.is_cached("missing"))
            .unwrap_err()
            .error_class()
            .name(),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    c.clear_cache();
    assert_eq!(
        block_on(c.refresh_table("missing"))
            .unwrap_err()
            .error_class()
            .name(),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}
