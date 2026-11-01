// SPDX-License-Identifier: Apache-2.0
//! `SET PATH`: the stored path, its errors, and lookup of bare routines,
//! tables and views in path order (Spark 4.2 `sql-path.sql`).
#![cfg(test)]

use arrow_array::cast::AsArray;
use krishiv_session::Session;

fn block_on<T>(f: impl Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(f)
}

fn session() -> Session {
    let s = Session::new().unwrap();
    s.conf().set("spark.sql.path.enabled", "true").unwrap();
    s
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await.unwrap().collect().await.unwrap() });
}

fn first(s: &Session, sql: &str) -> String {
    let batches = block_on(async { s.sql(sql).await.unwrap().collect().await.unwrap() });
    let b = &batches[0];
    let c = b.column(0);
    match c.data_type() {
        arrow_schema::DataType::Utf8 => c.as_string::<i32>().value(0).to_owned(),
        arrow_schema::DataType::Utf8View => c.as_string_view().value(0).to_owned(),
        arrow_schema::DataType::Int32 => c
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0)
            .to_string(),
        other => panic!("unexpected {other}"),
    }
}

fn error_class(s: &Session, sql: &str) -> String {
    let e = block_on(async {
        match s.sql(sql).await {
            Err(e) => e,
            Ok(df) => df.collect().await.unwrap_err(),
        }
    });
    e.error_class().name().to_owned()
}

#[test]
fn the_default_path_and_set_path_show_in_current_path() {
    let s = session();
    assert_eq!(
        first(&s, "SELECT current_path()"),
        "system.builtin,system.session,spark_catalog.default"
    );
    run(&s, "SET PATH = Spark_Catalog.Default, System.Builtin");
    assert_eq!(
        first(&s, "SELECT current_path()"),
        "Spark_Catalog.Default,System.Builtin"
    );
    run(&s, "SET PATH = PATH, system.session");
    assert_eq!(
        first(&s, "SELECT current_path()"),
        "Spark_Catalog.Default,System.Builtin,system.session"
    );
    run(&s, "SET PATH = DEFAULT_PATH");
    run(&s, "SET PATH = SYSTEM_PATH, CURRENT_SCHEMA");
    assert_eq!(
        first(&s, "SELECT current_path()"),
        "system.builtin,spark_catalog.default"
    );
}

#[test]
fn set_path_errors() {
    let s = session();
    assert_eq!(
        error_class(
            &s,
            "SET PATH = spark_catalog.default, SPARK_CATALOG.DEFAULT"
        ),
        "DUPLICATE_SQL_PATH_ENTRY"
    );
    assert_eq!(
        error_class(&s, "SET PATH = DEFAULT_PATH, system.builtin"),
        "DUPLICATE_SQL_PATH_ENTRY"
    );
    assert_eq!(
        error_class(&s, "SET PATH = only_one_part"),
        "INVALID_SQL_PATH_SCHEMA_REFERENCE"
    );
    assert_eq!(
        error_class(&s, "SET spark.sql.defaultPath = this is not a path"),
        "INVALID_CONF_VALUE.REQUIREMENT"
    );
    s.conf().set("spark.sql.path.enabled", "false").unwrap();
    assert_eq!(
        error_class(&s, "SET PATH = spark_catalog.default"),
        "UNSUPPORTED_FEATURE.SET_PATH_WHEN_DISABLED"
    );
}

#[test]
fn bare_routines_tables_and_views_follow_the_path_in_order() {
    let s = session();
    for sql in [
        "CREATE SCHEMA pa",
        "CREATE SCHEMA pb",
        "CREATE FUNCTION pa.pick() RETURNS INT RETURN 7",
        "CREATE FUNCTION pb.pick() RETURNS INT RETURN 11",
        "CREATE TABLE pa.t USING parquet AS SELECT 1 AS id",
        "CREATE TABLE pb.t USING parquet AS SELECT 2 AS id",
        "CREATE VIEW pb.only_b AS SELECT 9 AS id",
        // A view in the current schema must not hide the table that comes first in the path.
        "CREATE VIEW default.t AS SELECT 3 AS id",
    ] {
        run(&s, sql);
    }
    run(
        &s,
        "SET PATH = spark_catalog.pa, spark_catalog.pb, system.builtin",
    );
    assert_eq!(first(&s, "SELECT pick()"), "7");
    assert_eq!(first(&s, "SELECT id FROM t"), "1");
    assert_eq!(first(&s, "SELECT id FROM only_b"), "9");
    run(
        &s,
        "SET PATH = spark_catalog.pb, spark_catalog.pa, system.builtin",
    );
    assert_eq!(first(&s, "SELECT pick()"), "11");
    assert_eq!(first(&s, "SELECT id FROM t"), "2");
    run(&s, "SET PATH = spark_catalog.default, system.builtin");
    assert_eq!(error_class(&s, "SELECT pick()"), "UNRESOLVED_ROUTINE");
    assert_eq!(
        error_class(&s, "SELECT id FROM only_b"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    // A path naming no schema of the catalog finds no bare table.
    run(&s, "SET PATH = SYSTEM_PATH");
    assert_eq!(
        error_class(&s, "SELECT id FROM t"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn canonical_conf_keys_drive_the_path() {
    let s = Session::new().unwrap();
    s.conf().set("krishiv.sql.path.enabled", "true").unwrap();
    s.conf()
        .set(
            "krishiv.sql.defaultPath",
            "spark_catalog.default, system.builtin",
        )
        .unwrap();
    run(&s, "SET PATH = SYSTEM_PATH");
    assert!(!first(&s, "SELECT current_path()").is_empty());
}
