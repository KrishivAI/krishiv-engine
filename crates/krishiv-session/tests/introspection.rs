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
    // A header option is shown like any other, whichever way it was set.
    run(
        &s,
        "CREATE TABLE h (a INT) USING csv OPTIONS (header 'true')",
    );
    assert_eq!(
        rows(&s, "SHOW CREATE TABLE h")[0][0],
        "CREATE TABLE spark_catalog.default.h (\n  a INT)\nUSING csv\nOPTIONS (\n  'header' = 'true')\n"
    );
    assert!(
        rows(&s, "DESCRIBE EXTENDED h")
            .iter()
            .any(|r| r[0] == "Storage Properties" && r[1] == "[header=true]")
    );
    run(
        &s,
        "CREATE TABLE f (a INT) USING csv OPTIONS (header 'false')",
    );
    assert_eq!(
        rows(&s, "SHOW CREATE TABLE f")[0][0],
        "CREATE TABLE spark_catalog.default.f (\n  a INT)\nUSING csv\nOPTIONS (\n  'header' = 'false')\n"
    );
    run(&s, "CREATE TABLE p (a INT) USING csv");
    assert_eq!(
        rows(&s, "SHOW CREATE TABLE p")[0][0],
        "CREATE TABLE spark_catalog.default.p (\n  a INT)\nUSING csv\n"
    );
    assert_eq!(
        rows(&s, "SHOW CREATE TABLE v")[0][0],
        // The view's column keeps `t.a`'s comment (Spark 4.2.0: `column-comments/probe2.out`).
        "CREATE VIEW default.v (\n  a COMMENT 'ca')\nWITH SCHEMA COMPENSATION\nAS SELECT a FROM t\n"
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
        [["a", "int", "ca"], ["b", "string", "NULL"]]
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
    // The catalog keeps no column statistics: the rows Spark adds for `EXTENDED | FORMATTED` are NULL.
    for mode in ["EXTENDED", "FORMATTED"] {
        let rows = table(&s, &format!("DESCRIBE {mode} t c")).1;
        assert_eq!(rows.len(), 10, "{mode}");
        assert_eq!(rows[3], ["min", "NULL"], "{mode}");
        assert_eq!(rows[9], ["histogram", "NULL"], "{mode}");
    }
    assert_eq!(table(&s, "DESCRIBE t c").1.len(), 3);
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

/// A column's comment follows it through a query as Spark 4.2.0's field metadata does
/// (`research/spark/column-comments/probe1.out` in krishiv-internal): through aliases, joins, subqueries,
/// set operations (the left side's), filters, sorts, grouping keys and windows, into CTAS tables and views;
/// any other expression, a cast to the same type included, has none.
#[test]
fn column_comments_follow_columns_through_queries() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (a INT COMMENT 'ca', b STRING COMMENT 'cb', c INT) USING parquet",
    );
    run(
        &s,
        "CREATE TABLE u (a INT COMMENT 'ua', d INT COMMENT 'ud') USING parquet",
    );
    let comments = |q: &str| -> Vec<(String, String)> {
        rows(&s, &format!("DESC QUERY {q}"))
            .into_iter()
            .map(|r| (r[0].clone(), r[2].clone()))
            .collect()
    };
    let pairs = |p: &[(&str, &str)]| -> Vec<(String, String)> {
        p.iter()
            .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
            .collect()
    };
    let cases: [(&str, &[(&str, &str)]); 18] = [
        (
            "SELECT a, b FROM t GROUP BY ROLLUP(a, b)",
            &[("a", "ca"), ("b", "cb")],
        ),
        (
            "SELECT a, count(*) FROM t GROUP BY CUBE(a)",
            &[("a", "ca"), ("count(1)", "NULL")],
        ),
        (
            "SELECT a, b FROM t GROUP BY GROUPING SETS ((a), (b))",
            &[("a", "ca"), ("b", "cb")],
        ),
        (
            "SELECT a, a FROM t UNION ALL SELECT a, a FROM t",
            &[("a", "ca"), ("a", "ca")],
        ),
        ("SELECT a AS x, b FROM t", &[("x", "ca"), ("b", "cb")]),
        ("SELECT t.a FROM t JOIN u ON t.a = u.a", &[("a", "ca")]),
        (
            "SELECT u.a, d FROM t JOIN u ON t.a = u.a",
            &[("a", "ua"), ("d", "ud")],
        ),
        (
            "SELECT * FROM (SELECT a, b FROM t) s",
            &[("a", "ca"), ("b", "cb")],
        ),
        (
            "SELECT * FROM (SELECT a, b FROM t) s(x, y)",
            &[("x", "ca"), ("y", "cb")],
        ),
        ("SELECT CAST(a AS BIGINT) FROM t", &[("a", "NULL")]),
        ("SELECT CAST(a AS INT) AS a FROM t", &[("a", "NULL")]),
        ("SELECT a FROM t UNION ALL SELECT a FROM u", &[("a", "ca")]),
        ("SELECT a FROM t WHERE a > 1 ORDER BY a", &[("a", "ca")]),
        (
            "SELECT max(a), a FROM t GROUP BY a",
            &[("max(a)", "NULL"), ("a", "ca")],
        ),
        (
            "SELECT a, row_number() OVER (ORDER BY a) r FROM t",
            &[("a", "ca"), ("r", "NULL")],
        ),
        (
            "SELECT x FROM t LATERAL VIEW explode(array(a)) e AS x",
            &[("x", "NULL")],
        ),
        ("SELECT a FROM t INTERSECT SELECT a FROM u", &[("a", "ca")]),
        ("SELECT coalesce(a) FROM t", &[("coalesce(a)", "NULL")]),
    ];
    let mut wrong = Vec::new();
    for (q, want) in cases {
        let got = comments(q);
        if got != pairs(want) {
            wrong.push(format!("{q}: got {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    let desc = |name: &str| -> Vec<(String, String)> {
        rows(&s, &format!("DESC {name}"))
            .into_iter()
            .map(|r| (r[0].clone(), r[2].clone()))
            .collect()
    };
    run(&s, "CREATE TABLE ct AS SELECT a, b AS bb, c FROM t");
    assert_eq!(
        desc("ct"),
        // DESC prints an empty comment for a table column without one.
        pairs(&[("a", "ca"), ("bb", "cb"), ("c", "")])
    );
    run(&s, "CREATE VIEW v1 AS SELECT a, b FROM t");
    assert_eq!(desc("v1"), pairs(&[("a", "ca"), ("b", "cb")]));
    run(
        &s,
        "CREATE VIEW v2 (x, y COMMENT 'cy') AS SELECT a, b FROM t",
    );
    assert_eq!(desc("v2"), pairs(&[("x", "ca"), ("y", "cy")]));
    assert_eq!(
        comments("SELECT * FROM v1"),
        pairs(&[("a", "ca"), ("b", "cb")])
    );
    // A read of a view gives each column the view's stored comment, not the one of the column under it
    // (Spark 4.2.0: `column-comments/probe3.out`).
    assert_eq!(
        comments("SELECT * FROM v2"),
        pairs(&[("x", "ca"), ("y", "cy")])
    );
    // A view column stored without a comment has none, whatever the column under it has now.
    run(&s, "CREATE TABLE w (a INT COMMENT 'wa', b INT)");
    run(&s, "CREATE VIEW v3 AS SELECT a, b FROM w");
    run(&s, "ALTER TABLE w ALTER COLUMN b COMMENT 'later'");
    assert_eq!(
        comments("SELECT * FROM v3"),
        pairs(&[("a", "wa"), ("b", "NULL")])
    );
    // A DataFrame's temporary view holds its analyzed plan: a read of it keeps the scan's comments.
    block_on(s.table("t").create_or_replace_temp_view("dv")).unwrap();
    assert_eq!(
        comments("SELECT a, c FROM dv"),
        pairs(&[("a", "ca"), ("c", "NULL")])
    );
    // A changed comment is not served from a cached analysis (the catalog epoch is in its fingerprint).
    assert_eq!(comments("SELECT a FROM t"), pairs(&[("a", "ca")]));
    run(&s, "ALTER TABLE t ALTER COLUMN a COMMENT 'new'");
    assert_eq!(comments("SELECT a FROM t"), pairs(&[("a", "new")]));
}
