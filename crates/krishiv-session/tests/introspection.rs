// SPDX-License-Identifier: Apache-2.0
//! `DESCRIBE` and `SHOW` on the catalog through `Session::sql`; the expected rows are Spark 4.2.0's.
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

/// Every cell of `sql` as strings (`NULL` for a null), row by row, with the column names first.
fn table(s: &Session, sql: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let names = batches[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(i) {
                            "NULL".to_owned()
                        } else {
                            match c.data_type() {
                                ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
                                ArrowType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                                ArrowType::Boolean => c.as_boolean().value(i).to_string(),
                                other => panic!("unexpected {other}"),
                            }
                        }
                    })
                    .collect(),
            );
        }
    }
    (names, out)
}

fn rows(s: &Session, sql: &str) -> Vec<Vec<String>> {
    table(s, sql).1
}

fn setup(s: &Session) {
    run(s, "CREATE DATABASE d1 COMMENT 'dc'");
    run(
        s,
        "CREATE TABLE t (a INT NOT NULL COMMENT 'ca', b STRING) USING parquet COMMENT 'tc' TBLPROPERTIES ('k'='v')",
    );
    run(s, "CREATE VIEW v AS SELECT a FROM t");
    run(s, "CREATE TEMPORARY VIEW tv AS SELECT 1 AS x");
}

#[test]
fn show_tables_and_views() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    setup(&s);
    let (names, all) = table(&s, "SHOW TABLES");
    assert_eq!(names, ["namespace", "tableName", "isTemporary"]);
    assert_eq!(
        all,
        [
            ["default", "t", "false"],
            ["default", "v", "false"],
            ["", "tv", "true"]
        ]
    );
    assert_eq!(
        rows(&s, "SHOW TABLES LIKE 't*'"),
        [["default", "t", "false"], ["", "tv", "true"]]
    );
    assert_eq!(rows(&s, "SHOW TABLES IN d1"), [["", "tv", "true"]]);
    let (names, views) = table(&s, "SHOW VIEWS");
    assert_eq!(names, ["namespace", "viewName", "isTemporary"]);
    assert_eq!(views, [["default", "v", "false"], ["", "tv", "true"]]);
    assert_eq!(class(&s, "SHOW TABLES IN nope"), "SCHEMA_NOT_FOUND");
}

#[test]
fn show_columns_namespaces_and_properties() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    setup(&s);
    assert_eq!(rows(&s, "SHOW COLUMNS IN t"), [["a"], ["b"]]);
    assert_eq!(rows(&s, "SHOW COLUMNS IN tv"), [["x"]]);
    assert_eq!(rows(&s, "SHOW DATABASES"), [["d1"], ["default"]]);
    assert_eq!(rows(&s, "SHOW NAMESPACES LIKE 'd*'"), [["d1"], ["default"]]);
    assert_eq!(rows(&s, "SHOW TBLPROPERTIES t"), [["k", "v"]]);
    assert_eq!(rows(&s, "SHOW TBLPROPERTIES t ('k')"), [["k", "v"]]);
    assert_eq!(
        rows(&s, "SHOW TBLPROPERTIES t ('zz')"),
        [[
            "zz",
            "Table spark_catalog.default.t does not have property: zz"
        ]]
    );
    assert_eq!(
        class(&s, "SHOW COLUMNS IN missing"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn show_create_table() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    setup(&s);
    run(
        &s,
        "CREATE TABLE u USING csv OPTIONS (sep ';') AS SELECT 1 AS id",
    );
    let (names, t) = table(&s, "SHOW CREATE TABLE t");
    assert_eq!(names, ["createtab_stmt"]);
    assert_eq!(
        t[0][0],
        "CREATE TABLE spark_catalog.default.t (\n  a INT COMMENT 'ca',\n  b STRING COLLATE UTF8_BINARY)\nUSING parquet\nCOMMENT 'tc'\nTBLPROPERTIES (\n  'k' = 'v')\n"
    );
    assert_eq!(
        rows(&s, "SHOW CREATE TABLE u")[0][0],
        "CREATE TABLE spark_catalog.default.u (\n  id INT)\nUSING csv\nOPTIONS (\n  'sep' = ';')\n"
    );
    assert_eq!(
        rows(&s, "SHOW CREATE TABLE v")[0][0],
        "CREATE VIEW default.v (\n  a)\nWITH SCHEMA COMPENSATION\nAS SELECT a FROM t\n"
    );
}

#[test]
fn describe_table_query_namespace_and_column() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    setup(&s);
    let (names, d) = table(&s, "DESCRIBE t");
    assert_eq!(names, ["col_name", "data_type", "comment"]);
    assert_eq!(d, [["a", "int", "ca"], ["b", "string", ""]]);
    let ext = rows(&s, "DESCRIBE TABLE EXTENDED t");
    assert_eq!(ext[2], ["", "", ""]);
    assert_eq!(ext[3][0], "# Detailed Table Information");
    let keys: Vec<&str> = ext[4..].iter().map(|r| r[0].as_str()).collect();
    assert_eq!(
        keys,
        [
            "Catalog",
            "Database",
            "Table",
            "Created Time",
            "Last Access",
            "Created By",
            "Type",
            "Provider",
            "Comment",
            "Table Properties",
            "Location"
        ]
    );
    assert_eq!(ext[10][1], "MANAGED");
    assert_eq!(ext[13][1], "[k=v]");
    assert_eq!(
        rows(&s, "DESCRIBE QUERY SELECT a, b FROM t"),
        [["a", "int", "NULL"], ["b", "string", "NULL"]]
    );
    let (names, c) = table(&s, "DESCRIBE t a");
    assert_eq!(names, ["info_name", "info_value"]);
    assert_eq!(
        c,
        [["col_name", "a"], ["data_type", "int"], ["comment", "ca"]]
    );
    assert_eq!(
        class(&s, "DESCRIBE t zz"),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let ns = rows(&s, "DESCRIBE DATABASE EXTENDED d1");
    assert_eq!(ns[0], ["Catalog Name", "spark_catalog"]);
    assert_eq!(ns[1], ["Namespace Name", "d1"]);
    assert_eq!(ns[2], ["Comment", "dc"]);
    assert_eq!(ns[5], ["Properties", "()"]);
    assert_eq!(rows(&s, "DESCRIBE tv"), [["x", "int", ""]]);
    assert_eq!(class(&s, "DESCRIBE missing"), "TABLE_OR_VIEW_NOT_FOUND");
}
