// SPDX-License-Identifier: Apache-2.0
//! Permanent views `WITH SCHEMA BINDING | COMPENSATION | TYPE EVOLUTION | EVOLUTION`, `ALTER VIEW ... WITH SCHEMA`
//! and the legacy configs `spark.sql.legacy.viewSchemaBindingMode` / `viewSchemaCompensation`. Every case is a
//! statement sequence of Spark 4.2's goldens `view-schema-binding`, `view-schema-binding-config`,
//! `view-schema-compensation`, `view-schema-type-evolution` and `view-schema-evolution`
//! (`tests/spark-golden/v4.2.0/results/`).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use krishiv_types::DataType;

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

fn err(s: &Session, sql: &str) -> KrishivError {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql}: no error"),
        Err(e) => e,
    }
}

fn param(e: &KrishivError, name: &str) -> String {
    e.message_parameters()
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| panic!("{e}: no parameter {name}"))
}

/// The `name:type` of each column of `sql`'s result.
fn schema(s: &Session, sql: &str) -> Vec<String> {
    let schema = block_on(async { s.sql(sql).await?.schema().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    schema
        .fields()
        .iter()
        .map(|f| {
            let t = DataType::from_arrow_field(f).unwrap();
            format!("{}:{}", f.name(), t.catalog_string())
        })
        .collect()
}

/// Every row of `sql`, cells as strings joined by `|` (`~` for NULL).
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(i) {
                        return "~".to_owned();
                    }
                    match c.data_type() {
                        ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
                        ArrowType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                        _ => arrow_cast_string(c.as_ref(), i),
                    }
                })
                .collect();
            out.push(cells.join("|"));
        }
    }
    out
}

fn arrow_cast_string(c: &dyn Array, i: usize) -> String {
    use arrow_array::types::{Float64Type, Int32Type, Int64Type};
    match c.data_type() {
        ArrowType::Int32 => c.as_primitive::<Int32Type>().value(i).to_string(),
        ArrowType::Int64 => c.as_primitive::<Int64Type>().value(i).to_string(),
        ArrowType::Float64 => c.as_primitive::<Float64Type>().value(i).to_string(),
        other => panic!("unexpected type {other}"),
    }
}

/// The `DESCRIBE EXTENDED v` rows as `col_name|data_type|comment`.
fn describe(s: &Session, view: &str) -> Vec<String> {
    rows(s, &format!("DESCRIBE EXTENDED {view}"))
}

/// The `data_type` of the `DESCRIBE EXTENDED` row named `key`, if any.
fn info(s: &Session, view: &str, key: &str) -> Option<String> {
    describe(s, view).into_iter().find_map(|r| {
        let mut parts = r.splitn(3, '|');
        (parts.next() == Some(key)).then(|| parts.next().unwrap_or_default().to_owned())
    })
}

/// The column rows of `DESCRIBE EXTENDED v`, before its blank row.
fn columns(s: &Session, view: &str) -> Vec<String> {
    describe(s, view)
        .into_iter()
        .take_while(|r| r != "||")
        .collect()
}

fn show_create(s: &Session, view: &str) -> String {
    rows(s, &format!("SHOW CREATE TABLE {view}")).join("\n")
}

#[test]
fn binding_refuses_a_type_that_does_not_up_cast() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t(c1 INT NOT NULL) USING PARQUET");
    run(
        &s,
        "CREATE OR REPLACE VIEW v WITH SCHEMA BINDING AS SELECT * FROM t",
    );
    assert_eq!(schema(&s, "SELECT * FROM v"), ["c1:int"]);
    assert_eq!(
        info(&s, "v", "View Schema Mode").as_deref(),
        Some("BINDING")
    );
    assert_eq!(
        info(&s, "v", "View Query Output Columns").as_deref(),
        Some("[`c1`]")
    );
    run(&s, "DROP TABLE t");
    run(&s, "CREATE TABLE t(c1 BIGINT NOT NULL) USING PARQUET");
    let e = err(&s, "SELECT * FROM v");
    assert_eq!(e.error_class().name(), "CANNOT_UP_CAST_DATATYPE", "{e}");
    assert_eq!(param(&e, "expression"), "spark_catalog.default.t.c1");
    assert_eq!(param(&e, "sourceType"), "\"BIGINT\"");
    assert_eq!(param(&e, "targetType"), "\"INT\"");
    assert_eq!(
        param(&e, "details"),
        "The type path of the target object is:\n\nYou can either add an explicit cast to the input \
         data or choose a higher precision type of the field in the target object"
    );
    // The stored schema did not change.
    assert_eq!(columns(&s, "v"), ["c1|int|"]);
}

#[test]
fn binding_widens_a_type_that_up_casts() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t(c1 INT) USING PARQUET");
    run(&s, "CREATE VIEW v WITH SCHEMA BINDING AS SELECT * FROM t");
    run(&s, "DROP TABLE t");
    run(&s, "CREATE TABLE t(c1 SMALLINT) USING PARQUET");
    run(&s, "INSERT INTO t VALUES (7)");
    assert_eq!(schema(&s, "SELECT * FROM v"), ["c1:int"]);
    assert_eq!(rows(&s, "SELECT * FROM v"), ["7"]);
}

#[test]
fn a_lost_column_is_an_incompatible_schema_change() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t(c1 INT, c2 INT) USING PARQUET");
    run(
        &s,
        "CREATE OR REPLACE VIEW v WITH SCHEMA BINDING AS SELECT * FROM t",
    );
    assert_eq!(schema(&s, "SELECT * FROM v"), ["c1:int", "c2:int"]);
    run(&s, "DROP TABLE t");
    run(&s, "CREATE TABLE t(c1 INT) USING PARQUET");
    let e = err(&s, "SELECT * FROM v");
    assert_eq!(
        e.error_class().name(),
        "INCOMPATIBLE_VIEW_SCHEMA_CHANGE",
        "{e}"
    );
    assert_eq!(param(&e, "viewName"), "`spark_catalog`.`default`.`v`");
    assert_eq!(param(&e, "colName"), "c2");
    assert_eq!(param(&e, "expectedNum"), "1");
    assert_eq!(param(&e, "actualCols"), "[]");
    assert_eq!(
        param(&e, "suggestion"),
        "CREATE OR REPLACE VIEW spark_catalog.default.v AS SELECT * FROM t"
    );
    assert_eq!(e.sql_state(), Some("51024"));
}

#[test]
fn the_default_mode_follows_the_compensation_config() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "SET spark.sql.legacy.viewSchemaCompensation=false");
    run(&s, "CREATE TABLE t(c1 INT NOT NULL) USING PARQUET");
    run(&s, "CREATE OR REPLACE VIEW v AS SELECT * FROM t");
    assert_eq!(
        info(&s, "v", "View Schema Mode").as_deref(),
        Some("BINDING")
    );
    run(&s, "ALTER VIEW v WITH SCHEMA BINDING");
    assert_eq!(
        info(&s, "v", "View Schema Mode").as_deref(),
        Some("BINDING")
    );
    run(&s, "DROP TABLE t");
    run(&s, "CREATE TABLE t(c1 BIGINT NOT NULL) USING PARQUET");
    assert_eq!(
        err(&s, "SELECT * FROM v").error_class().name(),
        "CANNOT_UP_CAST_DATATYPE"
    );
    run(&s, "SET spark.sql.legacy.viewSchemaCompensation=true");
    run(&s, "CREATE OR REPLACE VIEW v AS SELECT * FROM t");
    assert_eq!(
        info(&s, "v", "View Schema Mode").as_deref(),
        Some("COMPENSATION")
    );
    assert!(show_create(&s, "v").contains("\nWITH SCHEMA COMPENSATION\nAS SELECT * FROM t"));
}

#[test]
fn compensation_casts_with_ansi_casts() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "SET spark.sql.ansi.enabled = false");
    run(&s, "CREATE TABLE t(c1 INT NOT NULL) USING PARQUET");
    run(
        &s,
        "CREATE OR REPLACE VIEW v WITH SCHEMA COMPENSATION AS SELECT * FROM t",
    );
    run(&s, "DROP TABLE t");
    run(&s, "CREATE TABLE t(c1 BIGINT NOT NULL) USING PARQUET");
    run(&s, "INSERT INTO t VALUES (1)");
    assert_eq!(schema(&s, "SELECT * FROM v"), ["c1:int"]);
    assert_eq!(rows(&s, "SELECT * FROM v"), ["1"]);
    run(&s, "DROP TABLE IF EXISTS t");
    run(
        &s,
        "CREATE TABLE t(c1 STRING NOT NULL, c2 INT) USING PARQUET",
    );
    run(&s, "INSERT INTO t VALUES ('1', 2)");
    assert_eq!(rows(&s, "SELECT * FROM v"), ["1"]);
    run(&s, "INSERT INTO t VALUES ('a', 2)");
    assert_eq!(
        err(&s, "SELECT * FROM v").error_class().name(),
        "CAST_INVALID_INPUT"
    );
    run(&s, "DROP TABLE IF EXISTS t");
    run(
        &s,
        "CREATE TABLE t(c1 MAP<STRING, STRING>, c2 INT) USING PARQUET",
    );
    assert_eq!(
        err(&s, "SELECT * FROM v").error_class().name(),
        "DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION"
    );
    assert_eq!(columns(&s, "v"), ["c1|int|"]);
    // A missing column.
    run(&s, "DROP TABLE IF EXISTS t");
    run(&s, "CREATE TABLE t(c3 INT NOT NULL, c2 INT) USING PARQUET");
    let e = err(&s, "SELECT * FROM v");
    assert_eq!(
        e.error_class().name(),
        "INCOMPATIBLE_VIEW_SCHEMA_CHANGE",
        "{e}"
    );
    assert_eq!(param(&e, "colName"), "c1");
}

#[test]
fn alter_view_changes_the_mode() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t(c1 INT) USING PARQUET");
    run(&s, "INSERT INTO t VALUES(1)");
    run(
        &s,
        "CREATE OR REPLACE VIEW v WITH SCHEMA BINDING AS SELECT * FROM t",
    );
    run(&s, "DROP TABLE IF EXISTS t");
    run(&s, "CREATE TABLE t(c1 STRING) USING PARQUET");
    run(&s, "INSERT INTO t VALUES('1')");
    let e = err(&s, "SELECT * FROM v");
    assert_eq!(e.error_class().name(), "CANNOT_UP_CAST_DATATYPE", "{e}");
    assert_eq!(param(&e, "sourceType"), "\"STRING\"");
    run(&s, "ALTER VIEW v WITH SCHEMA COMPENSATION");
    assert_eq!(
        info(&s, "v", "View Schema Mode").as_deref(),
        Some("COMPENSATION")
    );
    assert_eq!(rows(&s, "SELECT * FROM v"), ["1"]);
    run(&s, "ALTER VIEW v WITH SCHEMA TYPE EVOLUTION");
    assert_eq!(schema(&s, "SELECT * FROM v"), ["c1:string"]);
    assert_eq!(columns(&s, "v"), ["c1|string|"]);
    assert_eq!(
        info(&s, "v", "View Schema Mode").as_deref(),
        Some("TYPE EVOLUTION")
    );
    // A temporary view has no binding mode.
    run(&s, "CREATE TEMPORARY VIEW tv AS SELECT 1 AS x");
    assert_eq!(
        err(&s, "ALTER VIEW tv WITH SCHEMA BINDING")
            .error_class()
            .name(),
        "UNSUPPORTED_FEATURE.TEMPORARY_VIEW_WITH_SCHEMA_BINDING_MODE"
    );
}

#[test]
fn type_evolution_keeps_the_names_and_takes_the_types() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t(c1 INT NOT NULL, c2 INT) USING PARQUET");
    run(&s, "INSERT INTO t VALUES (1, 2)");
    run(
        &s,
        "CREATE OR REPLACE VIEW v WITH SCHEMA TYPE EVOLUTION AS SELECT * FROM t",
    );
    assert_eq!(rows(&s, "SELECT * FROM v"), ["1|2"]);
    run(&s, "DROP TABLE IF EXISTS t");
    run(
        &s,
        "CREATE TABLE t(c1 STRING NOT NULL, c2 DOUBLE, c3 DATE) USING PARQUET",
    );
    run(&s, "INSERT INTO t VALUES ('1', 2.0, DATE'2022-01-01')");
    assert_eq!(schema(&s, "SELECT * FROM v"), ["c1:string", "c2:double"]);
    assert_eq!(rows(&s, "SELECT * FROM v"), ["1|2"]);
    // The read stored the new types.
    assert_eq!(columns(&s, "v"), ["c1|string|", "c2|double|"]);
    assert_eq!(
        info(&s, "v", "View Query Output Columns").as_deref(),
        Some("[`c1`, `c2`]")
    );
    run(&s, "DROP TABLE IF EXISTS t");
    run(&s, "CREATE TABLE t(c1 INT) USING PARQUET");
    let e = err(&s, "SELECT * FROM v");
    assert_eq!(
        e.error_class().name(),
        "INCOMPATIBLE_VIEW_SCHEMA_CHANGE",
        "{e}"
    );
    assert_eq!(param(&e, "colName"), "c2");
}

#[test]
fn a_column_list_turns_evolution_into_type_evolution() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t(c1 INT NOT NULL, c2 INT) USING PARQUET");
    run(
        &s,
        "CREATE OR REPLACE VIEW v(a1, a2) WITH SCHEMA EVOLUTION AS SELECT * FROM t",
    );
    assert_eq!(
        info(&s, "v", "View Schema Mode").as_deref(),
        Some("TYPE EVOLUTION")
    );
    assert_eq!(
        info(&s, "v", "View Query Output Columns").as_deref(),
        Some("[`c1`, `c2`]")
    );
    run(&s, "DROP TABLE IF EXISTS t");
    run(
        &s,
        "CREATE TABLE t(c1 STRING NOT NULL, c2 DOUBLE) USING PARQUET",
    );
    assert_eq!(schema(&s, "SELECT * FROM v"), ["a1:string", "a2:double"]);
    assert_eq!(columns(&s, "v"), ["a1|string|", "a2|double|"]);
    run(&s, "DROP TABLE IF EXISTS t");
    run(&s, "CREATE TABLE t(c3 INT, c2 INT) USING PARQUET");
    let e = err(&s, "SELECT * FROM v");
    assert_eq!(
        e.error_class().name(),
        "INCOMPATIBLE_VIEW_SCHEMA_CHANGE",
        "{e}"
    );
    assert_eq!(param(&e, "colName"), "c1");
    assert_eq!(
        param(&e, "suggestion"),
        "CREATE OR REPLACE VIEW spark_catalog.default.v (a1, a2) AS SELECT * FROM t"
    );
}

#[test]
fn evolution_takes_the_query_schema() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t(c1 INT NOT NULL, c2 INT) USING PARQUET");
    run(
        &s,
        "CREATE OR REPLACE VIEW v WITH SCHEMA EVOLUTION AS SELECT * FROM t",
    );
    assert_eq!(
        info(&s, "v", "View Schema Mode").as_deref(),
        Some("EVOLUTION")
    );
    run(&s, "DROP TABLE IF EXISTS t");
    run(
        &s,
        "CREATE TABLE t(c4 STRING NOT NULL, c5 DOUBLE) USING PARQUET",
    );
    run(&s, "INSERT INTO t VALUES ('1', 2.0)");
    assert_eq!(schema(&s, "SELECT * FROM v"), ["c4:string", "c5:double"]);
    assert_eq!(rows(&s, "SELECT * FROM v"), ["1|2"]);
    assert_eq!(columns(&s, "v"), ["c4|string|", "c5|double|"]);
    assert_eq!(
        info(&s, "v", "View Query Output Columns").as_deref(),
        Some("[`c4`, `c5`]")
    );
    run(&s, "DROP TABLE IF EXISTS t");
    run(&s, "CREATE TABLE t(c1 INT) USING PARQUET");
    assert_eq!(schema(&s, "SELECT * FROM v"), ["c1:int"]);
    assert_eq!(columns(&s, "v"), ["c1|int|"]);
}

#[test]
fn evolution_refuses_duplicate_names() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t1(c1 INT) USING PARQUET");
    run(&s, "CREATE TABLE t2(c2 INT) USING PARQUET");
    run(
        &s,
        "CREATE OR REPLACE VIEW v WITH SCHEMA EVOLUTION AS SELECT * FROM t1, t2",
    );
    assert_eq!(schema(&s, "SELECT * FROM v"), ["c1:int", "c2:int"]);
    run(&s, "DROP TABLE IF EXISTS t2");
    run(&s, "CREATE TABLE t2(c1 INT) USING PARQUET");
    let e = err(&s, "SELECT * FROM v");
    assert_eq!(e.error_class().name(), "COLUMN_ALREADY_EXISTS", "{e}");
    assert_eq!(param(&e, "columnName"), "`c1`");
}

#[test]
fn alter_view_to_evolution_follows_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t(c1 INT) USING PARQUET");
    run(&s, "CREATE OR REPLACE VIEW v AS SELECT * FROM t");
    run(&s, "ALTER VIEW v WITH SCHEMA EVOLUTION");
    run(&s, "DROP TABLE IF EXISTS t");
    run(&s, "CREATE TABLE t(c1 STRING, c2 INT) USING PARQUET");
    assert_eq!(schema(&s, "SELECT * FROM v"), ["c1:string", "c2:int"]);
    assert_eq!(columns(&s, "v"), ["c1|string|", "c2|int|"]);
    assert_eq!(
        info(&s, "v", "View Query Output Columns").as_deref(),
        Some("[`c1`, `c2`]")
    );
}

#[test]
fn the_binding_mode_config_off_hides_and_refuses_the_modes() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    assert_eq!(
        rows(&s, "SET spark.sql.legacy.viewSchemaBindingMode"),
        ["spark.sql.legacy.viewSchemaBindingMode|true"]
    );
    run(&s, "SET spark.sql.legacy.viewSchemaBindingMode = false");
    for mode in ["BINDING", "COMPENSATION", "TYPE EVOLUTION", "EVOLUTION"] {
        let sql = format!("CREATE OR REPLACE VIEW v WITH SCHEMA {mode} AS SELECT 1");
        let e = err(&s, &sql);
        assert_eq!(e.error_class().name(), "FEATURE_NOT_ENABLED", "{e}");
        assert_eq!(param(&e, "featureName"), "VIEW ... WITH SCHEMA ...");
        assert_eq!(
            param(&e, "configKey"),
            "spark.sql.legacy.viewSchemaBindingMode"
        );
        assert_eq!(param(&e, "configValue"), "true");
        let context = &e.query_contexts()[0];
        assert_eq!(context.fragment, format!("WITH SCHEMA {mode}"));
    }
    run(&s, "CREATE OR REPLACE VIEW v AS SELECT 1");
    assert_eq!(info(&s, "v", "View Schema Mode"), None);
    assert_eq!(
        show_create(&s, "v"),
        "CREATE VIEW default.v (\n  `1`)\nAS SELECT 1\n"
    );
    // Created without a mode, the view reads as BINDING once the feature is on.
    run(&s, "SET spark.sql.legacy.viewSchemaBindingMode = true");
    assert_eq!(
        info(&s, "v", "View Schema Mode").as_deref(),
        Some("BINDING")
    );
    assert!(show_create(&s, "v").contains("\nWITH SCHEMA BINDING\nAS SELECT 1"));
}

#[test]
fn without_the_feature_a_view_compensates_unless_configured_not_to() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "SET spark.sql.legacy.viewSchemaBindingMode = false");
    run(&s, "SET spark.sql.ansi.enabled = false");
    run(&s, "CREATE TABLE t(c1 INT NOT NULL) USING PARQUET");
    run(&s, "CREATE OR REPLACE VIEW v AS SELECT * FROM t");
    run(&s, "DROP TABLE t");
    run(&s, "CREATE TABLE t(c1 BIGINT NOT NULL) USING PARQUET");
    run(&s, "INSERT INTO t VALUES (1)");
    assert_eq!(schema(&s, "SELECT * FROM v"), ["c1:int"]);
    assert_eq!(rows(&s, "SELECT * FROM v"), ["1"]);
    // The Krishiv key is the same setting.
    run(&s, "SET krishiv.sql.legacy.viewSchemaCompensation = false");
    assert_eq!(
        err(&s, "SELECT * FROM v").error_class().name(),
        "CANNOT_UP_CAST_DATATYPE"
    );
    run(&s, "SET spark.sql.legacy.viewSchemaCompensation = true");
    assert_eq!(rows(&s, "SELECT * FROM v"), ["1"]);
}

#[test]
fn the_krishiv_key_turns_the_binding_mode_off_too() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "SET krishiv.sql.legacy.viewSchemaBindingMode = false");
    assert_eq!(
        err(&s, "CREATE VIEW v WITH SCHEMA BINDING AS SELECT 1")
            .error_class()
            .name(),
        "FEATURE_NOT_ENABLED"
    );
    run(&s, "CREATE VIEW v AS SELECT 1");
    assert_eq!(info(&s, "v", "View Schema Mode"), None);
}
