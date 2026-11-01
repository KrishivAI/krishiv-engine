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

#[test]
fn a_view_keeps_the_path_it_was_created_under() {
    let s = session();
    for sql in [
        "CREATE SCHEMA fa",
        "CREATE SCHEMA fb",
        "CREATE TABLE fa.frozen_t USING parquet AS SELECT 1 AS id",
        "CREATE TABLE fb.frozen_t USING parquet AS SELECT 2 AS id",
        "SET PATH = spark_catalog.fa, system.builtin",
        "CREATE VIEW default.v_frozen AS SELECT id FROM frozen_t",
        "SET PATH = spark_catalog.fb, system.builtin",
    ] {
        run(&s, sql);
    }
    assert_eq!(first(&s, "SELECT id FROM frozen_t"), "2");
    assert_eq!(first(&s, "SELECT id FROM default.v_frozen"), "1");
}

#[test]
fn a_stored_function_keeps_the_path_it_was_created_under() {
    let s = session();
    for sql in [
        "CREATE SCHEMA ga",
        "CREATE SCHEMA gb",
        "CREATE TABLE ga.frozen_t USING parquet AS SELECT 10 AS id",
        "CREATE TABLE gb.frozen_t USING parquet AS SELECT 20 AS id",
        "SET PATH = spark_catalog.ga, system.builtin",
        "CREATE FUNCTION default.frozen_fn() RETURNS INT RETURN (SELECT MAX(id) FROM frozen_t)",
        "CREATE FUNCTION default.frozen_tvf() RETURNS TABLE (id INT) RETURN SELECT id FROM frozen_t",
        "SET PATH = spark_catalog.gb, system.builtin",
    ] {
        run(&s, sql);
    }
    assert_eq!(first(&s, "SELECT MAX(id) FROM frozen_t"), "20");
    assert_eq!(first(&s, "SELECT default.frozen_fn()"), "10");
    assert_eq!(first(&s, "SELECT * FROM default.frozen_tvf()"), "10");
}

#[test]
fn alter_view_as_takes_the_path_current_at_the_alter() {
    let s = session();
    for sql in [
        "CREATE SCHEMA va",
        "CREATE SCHEMA vb",
        "CREATE TABLE va.alter_t USING parquet AS SELECT 1 AS id",
        "CREATE TABLE vb.alter_t USING parquet AS SELECT 2 AS id",
        "SET PATH = spark_catalog.va, system.builtin",
        "CREATE VIEW default.v_alter AS SELECT id FROM alter_t",
        "SET PATH = spark_catalog.vb, system.builtin",
    ] {
        run(&s, sql);
    }
    assert_eq!(first(&s, "SELECT id FROM default.v_alter"), "1");
    run(&s, "ALTER VIEW default.v_alter AS SELECT id FROM alter_t");
    run(&s, "SET PATH = spark_catalog.va, system.builtin");
    assert_eq!(first(&s, "SELECT id FROM default.v_alter"), "2");
}

#[test]
fn a_view_does_not_read_a_namespace_its_path_left_out() {
    let s = session();
    for sql in [
        "CREATE SCHEMA ma",
        "CREATE TABLE ma.gone_t USING parquet AS SELECT 1 AS id",
        "SET PATH = spark_catalog.ma, system.builtin",
        "CREATE VIEW default.v_gone AS SELECT id FROM gone_t",
        "DROP TABLE ma.gone_t",
        "CREATE TABLE default.gone_t USING parquet AS SELECT 2 AS id",
    ] {
        run(&s, sql);
    }
    let err = block_on(async {
        s.sql("SELECT id FROM default.v_gone")
            .await?
            .collect()
            .await
    })
    .unwrap_err();
    assert!(
        format!("{err:?}").contains("TableOrViewNotFound"),
        "{err:?}"
    );
}

#[test]
fn a_stored_function_calls_routines_through_the_path_it_was_created_under() {
    let s = session();
    for sql in [
        "CREATE SCHEMA fa",
        "CREATE SCHEMA fb",
        "CREATE FUNCTION fa.pick() RETURNS INT RETURN 1",
        "CREATE FUNCTION fb.pick() RETURNS INT RETURN 2",
        "SET PATH = spark_catalog.fa, system.builtin",
        "CREATE FUNCTION default.outer_fn() RETURNS INT RETURN pick()",
        "SET PATH = spark_catalog.fb, system.builtin",
    ] {
        run(&s, sql);
    }
    assert_eq!(first(&s, "SELECT pick()"), "2");
    assert_eq!(first(&s, "SELECT default.outer_fn()"), "1");
}

#[test]
fn a_view_calls_routines_through_the_path_it_was_created_under() {
    let s = session();
    for sql in [
        "CREATE SCHEMA va",
        "CREATE SCHEMA vb",
        "CREATE FUNCTION va.pick() RETURNS INT RETURN 1",
        "CREATE FUNCTION vb.pick() RETURNS INT RETURN 2",
        "SET PATH = spark_catalog.va, system.builtin",
        "CREATE VIEW default.pick_view AS SELECT pick() AS v",
        "SET PATH = spark_catalog.vb, system.builtin",
    ] {
        run(&s, sql);
    }
    assert_eq!(first(&s, "SELECT pick()"), "2");
    assert_eq!(first(&s, "SELECT v FROM default.pick_view"), "1");
}

#[test]
fn a_session_only_path_reaches_temporary_views_not_stored_tables() {
    let s = session();
    for sql in [
        "CREATE TABLE default.only_stored USING parquet AS SELECT 1 AS id",
        "CREATE TEMPORARY VIEW only_temp AS SELECT 2 AS id",
        "SET PATH = system.session, system.builtin",
    ] {
        run(&s, sql);
    }
    assert_eq!(first(&s, "SELECT id FROM only_temp"), "2");
    let err =
        block_on(async { s.sql("SELECT id FROM only_stored").await?.collect().await }).unwrap_err();
    assert!(
        format!("{err:?}").contains("TableOrViewNotFound"),
        "{err:?}"
    );
}

#[test]
fn a_view_does_not_fall_back_to_its_namespace_for_a_routine_gone_from_its_path() {
    let s = session();
    for sql in [
        "CREATE SCHEMA gfa",
        "CREATE FUNCTION gfa.pick() RETURNS INT RETURN 1",
        "CREATE FUNCTION default.pick() RETURNS INT RETURN 2",
        "SET PATH = spark_catalog.gfa, system.builtin",
        "CREATE VIEW default.gone_view AS SELECT pick() AS x",
        "DROP FUNCTION gfa.pick",
    ] {
        run(&s, sql);
    }
    let read = block_on(async {
        s.sql("SELECT x FROM default.gone_view")
            .await?
            .collect()
            .await
    });
    assert!(read.is_err(), "{read:?}");
}

#[test]
fn a_view_on_a_session_only_path_does_not_fall_back_to_its_namespace() {
    let s = session();
    for sql in [
        "CREATE TEMPORARY FUNCTION pick() RETURNS INT RETURN 1",
        "CREATE FUNCTION default.pick() RETURNS INT RETURN 2",
        "SET PATH = system.session, system.builtin",
        "CREATE VIEW default.so_view AS SELECT pick() AS x",
        "DROP TEMPORARY FUNCTION pick",
    ] {
        run(&s, sql);
    }
    let read = block_on(async {
        s.sql("SELECT x FROM default.so_view")
            .await?
            .collect()
            .await
    });
    assert!(read.is_err(), "{read:?}");
}

#[test]
fn a_view_over_a_temporary_function_keeps_calling_it() {
    let s = session();
    for sql in [
        "CREATE TEMPORARY FUNCTION tmp_pick() RETURNS INT RETURN 7",
        "CREATE VIEW default.tmp_view AS SELECT tmp_pick() AS x",
    ] {
        run(&s, sql);
    }
    assert_eq!(first(&s, "SELECT x FROM default.tmp_view"), "7");
}

#[test]
fn a_view_with_qualify_reads_back() {
    let s = session();
    run(
        &s,
        "CREATE VIEW default.q_view AS SELECT row_number() OVER (ORDER BY id) AS rn FROM range(3) QUALIFY rn = 1",
    );
    assert_eq!(first(&s, "SELECT rn FROM default.q_view"), "1");
}

#[test]
fn a_stored_function_falls_back_to_its_own_namespace() {
    let s = session();
    for sql in [
        "CREATE SCHEMA fns",
        "CREATE FUNCTION fns.inner_f() RETURNS INT RETURN 1",
        "CREATE FUNCTION fns.outer_f() RETURNS INT RETURN inner_f()",
    ] {
        run(&s, sql);
    }
    assert_eq!(first(&s, "SELECT fns.outer_f()"), "1");
}
