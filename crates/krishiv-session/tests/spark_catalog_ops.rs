// SPDX-License-Identifier: Apache-2.0
//! The rest of `Session::catalog()` (PR-46, CM group 9): the drops, `createDatabase`, truncate,
//! analyze, partitions, views, table properties, the create statement and the catalog pattern.
#![cfg(test)]

use std::collections::BTreeMap;

use futures::executor::block_on;
use krishiv_session::Session;
use krishiv_session::spark_catalog::Catalog;

fn session(dir: &std::path::Path) -> Session {
    Session::builder()
        .config("spark.sql.warehouse.dir", dir.to_str().unwrap())
        .build()
        .unwrap()
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

fn count(s: &Session, table: &str) -> i64 {
    use arrow_array::cast::AsArray;
    let batches = block_on(async {
        s.sql(&format!("SELECT count(*) AS n FROM {table}"))
            .await?
            .collect()
            .await
    })
    .unwrap();
    batches[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0)
}

#[test]
fn qualified_quotes_each_part() {
    assert_eq!(Catalog::qualified("db", "t"), "`db`.`t`");
    assert_eq!(Catalog::qualified("a.b", "c`d"), "`a`.`b`.`c``d`");
}

#[test]
fn drop_table_honours_if_exists() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT) USING parquet");
    let c = s.catalog();
    block_on(c.drop_table("t", false, false)).unwrap();
    assert!(!block_on(c.table_exists("t")).unwrap());
    let e = block_on(c.drop_table("t", false, false)).unwrap_err();
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    block_on(c.drop_table("t", true, false)).unwrap();
    run(&s, "CREATE TABLE u (a INT) USING parquet");
    block_on(c.drop_table("u", false, true)).unwrap();
    assert!(!block_on(c.table_exists("u")).unwrap());
}

#[test]
fn drop_view_takes_temporary_and_permanent_views() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE VIEW pv AS SELECT 1 AS a");
    run(&s, "CREATE TEMPORARY VIEW tv AS SELECT 1 AS a");
    let c = s.catalog();
    block_on(c.drop_view("pv", false)).unwrap();
    block_on(c.drop_view("tv", false)).unwrap();
    let e = block_on(c.drop_view("pv", false)).unwrap_err();
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    block_on(c.drop_view("pv", true)).unwrap();
}

#[test]
fn databases_are_created_with_properties_and_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let c = s.catalog();
    let props = BTreeMap::from([
        ("team".to_owned(), "it's".to_owned()),
        ("lead".to_owned(), "x".to_owned()),
    ]);
    block_on(c.create_database("db1", false, &props)).unwrap();
    assert!(block_on(c.database_exists("db1")).unwrap());
    let e = block_on(c.create_database("db1", false, &BTreeMap::new())).unwrap_err();
    assert_eq!(e.error_class().name(), "SCHEMA_ALREADY_EXISTS");
    block_on(c.create_database("db1", true, &BTreeMap::new())).unwrap();
    run(&s, "CREATE TABLE db1.t (a INT) USING parquet");
    let e = block_on(c.drop_database("db1", false, false)).unwrap_err();
    assert_eq!(e.error_class().name(), "SCHEMA_NOT_EMPTY");
    block_on(c.drop_database("db1", false, true)).unwrap();
    assert!(!block_on(c.database_exists("db1")).unwrap());
    block_on(c.drop_database("db1", true, false)).unwrap();
    let e = block_on(c.drop_database("db1", false, false)).unwrap_err();
    assert_eq!(e.error_class().name(), "SCHEMA_NOT_FOUND");
}

#[test]
fn truncate_empties_the_table_and_analyze_runs() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT) USING parquet");
    run(&s, "INSERT INTO t VALUES (1), (2), (3)");
    let c = s.catalog();
    block_on(c.analyze_table("t", true)).unwrap();
    block_on(c.analyze_table("t", false)).unwrap();
    assert_eq!(count(&s, "t"), 3);
    block_on(c.truncate_table("t")).unwrap();
    assert_eq!(count(&s, "t"), 0);
    let e = block_on(c.truncate_table("nope")).unwrap_err();
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn partitions_are_listed_as_the_statement_shows_them() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE p (a INT, k STRING) USING parquet PARTITIONED BY (k)",
    );
    let c = s.catalog();
    assert!(block_on(c.list_partitions("p")).unwrap().is_empty());
    run(&s, "ALTER TABLE p ADD PARTITION (k = 'x')");
    assert_eq!(block_on(c.list_partitions("p")).unwrap(), ["k=x"]);
    let e = block_on(c.list_partitions("nope")).unwrap_err();
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn views_are_tables_kept_to_views_and_temporary_views() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT) USING parquet");
    run(&s, "CREATE VIEW pv AS SELECT 1 AS a");
    run(&s, "CREATE TEMPORARY VIEW tv AS SELECT 1 AS a");
    let c = s.catalog();
    let names: Vec<_> = block_on(c.list_views(None, None))
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(names, ["pv", "tv"]);
    let names: Vec<_> = block_on(c.list_views(Some("default"), Some("p*")))
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(names, ["pv"]);
    let e = block_on(c.list_views(Some("nope"), None)).unwrap_err();
    assert_eq!(e.error_class().name(), "SCHEMA_NOT_FOUND");
}

#[test]
fn table_properties_and_the_create_statement() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES ('k' = 'v')",
    );
    let c = s.catalog();
    let props = block_on(c.get_table_properties("t")).unwrap();
    assert_eq!(props.get("k").map(String::as_str), Some("v"));
    let text = block_on(c.get_create_table_string("t", false)).unwrap();
    assert!(text.starts_with("CREATE TABLE"), "{text}");
    let e = block_on(c.get_table_properties("nope")).unwrap_err();
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    let e = block_on(c.get_create_table_string("nope", false)).unwrap_err();
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn catalogs_match_a_pattern() {
    let s = Session::new().unwrap();
    let c = s.catalog();
    assert_eq!(c.list_catalogs_matching(None), ["spark_catalog"]);
    assert_eq!(c.list_catalogs_matching(Some("spark*")), ["spark_catalog"]);
    assert_eq!(
        c.list_catalogs_matching(Some("a|spark_catalog")),
        ["spark_catalog"]
    );
    assert!(c.list_catalogs_matching(Some("zz*")).is_empty());
}
