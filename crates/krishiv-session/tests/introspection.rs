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
    if batches.is_empty() {
        return (Vec::new(), Vec::new());
    }
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

#[test]
fn temporary_views_and_unsupported_layouts() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    setup(&s);
    // A temporary view has columns only, and no properties.
    assert_eq!(rows(&s, "DESCRIBE EXTENDED tv"), [["x", "int", ""]]);
    assert!(rows(&s, "SHOW TBLPROPERTIES tv").is_empty());
    assert_eq!(
        class(&s, "SHOW CREATE TABLE tv"),
        "EXPECT_PERMANENT_VIEW_NOT_TEMP"
    );
    // `FORMATTED` is `EXTENDED` (Spark's `DescribeTableCommand` takes either as `isExtended`).
    assert_eq!(
        rows(&s, "DESCRIBE FORMATTED t"),
        rows(&s, "DESCRIBE EXTENDED t")
    );
    // The `IN`/`FROM` namespace must agree with a qualified table name.
    assert_eq!(
        class(&s, "SHOW COLUMNS IN default.t FROM d1"),
        "SHOW_COLUMNS_WITH_CONFLICT_NAMESPACE"
    );
    assert_eq!(rows(&s, "SHOW COLUMNS IN t FROM default"), [["a"], ["b"]]);
    assert_eq!(class(&s, "DESCRIBE NAMESPACE nope"), "SCHEMA_NOT_FOUND");
}

#[test]
fn create_table_text_round_trips_quotes_and_hides_secrets() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE q (a INT) USING parquet TBLPROPERTIES ('it''s' = 'a\\b', 'my.password' = 'hunter2')",
    );
    let text = &rows(&s, "SHOW CREATE TABLE q")[0][0];
    assert!(text.contains("'it\\'s' = 'a\\\\b'"), "{text}");
    assert!(
        text.contains("'my.password' = '*********(redacted)'"),
        "{text}"
    );
    assert!(!text.contains("hunter2"));
    assert_eq!(
        rows(&s, "SHOW TBLPROPERTIES q ('my.password')"),
        [["my.password", "*********(redacted)"]]
    );
    let ext = rows(&s, "DESCRIBE EXTENDED q");
    assert!(ext.iter().all(|r| !r[1].contains("hunter2")));
    run(&s, "CREATE TABLE n (s STRUCT<a: INT>) USING parquet");
    assert_eq!(
        class(&s, "SHOW CREATE TABLE n"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn many_stars_in_a_pattern_stay_fast() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab (a INT) USING parquet",
    );
    assert!(rows(&s, "SHOW TABLES LIKE '*a*a*a*a*a*a*a*a*a*a*a*a*a*a*a*c'").is_empty());
}

/// Review follow-ups: column statistics, namespace secrets, names and locations in `SHOW CREATE TABLE`.
#[test]
fn describe_modes_secrets_and_quoting() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (`a b` INT, c STRING) USING parquet");
    assert_eq!(
        class(&s, "DESCRIBE EXTENDED t c"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(
        class(&s, "DESCRIBE FORMATTED t c"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    let text = table(&s, "SHOW CREATE TABLE t").1[0][0].clone();
    assert!(text.contains("  `a b` INT"), "{text}");
    assert!(text.contains("\n  c STRING"), "{text}");
    run(
        &s,
        "CREATE DATABASE d WITH DBPROPERTIES ('db.password'='hunter2', 'k'='v')",
    );
    let shown = table(&s, "DESCRIBE NAMESPACE EXTENDED d").1;
    let props = shown
        .iter()
        .find(|r| r[0] == "Properties")
        .map(|r| r[1].clone())
        .unwrap();
    assert!(
        !props.contains("hunter2") && props.contains("k=v"),
        "{props}"
    );
    // An external location with a quote reads back as one literal.
    let odd = dir.path().join("it's");
    run(
        &s,
        &format!(
            "CREATE TABLE e (x INT) USING parquet LOCATION '{}'",
            odd.display().to_string().replace('\'', "\\'")
        ),
    );
    let text = table(&s, "SHOW CREATE TABLE e").1[0][0].clone();
    assert!(text.contains("it\\'s'"), "{text}");
}
