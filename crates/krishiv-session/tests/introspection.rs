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
    // Spark 4.2.0 prints nothing for no properties (`schema-collation/probe1.out`).
    assert_eq!(ns[5], ["Properties", ""]);
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
        !props.contains("hunter2") && props.contains("(k,v)"),
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
    // A stored view keeps the comments it stored, or none, whatever the columns under it have now; an
    // `EVOLUTION` view reads its query's (Spark 4.2.0: `column-comments/probe4.out`).
    run(
        &s,
        "CREATE TABLE w (a INT COMMENT 'wa', b INT) USING parquet",
    );
    run(&s, "CREATE VIEW v3 AS SELECT a, b FROM w");
    run(
        &s,
        "CREATE VIEW v3b WITH SCHEMA BINDING AS SELECT a, b FROM w",
    );
    run(
        &s,
        "CREATE VIEW v3c WITH SCHEMA COMPENSATION AS SELECT a, b FROM w",
    );
    run(
        &s,
        "CREATE VIEW v3e WITH SCHEMA EVOLUTION AS SELECT a, b FROM w",
    );
    run(&s, "ALTER TABLE w ALTER COLUMN b COMMENT 'later'");
    run(&s, "ALTER TABLE w ALTER COLUMN a COMMENT 'wa2'");
    for v in ["v3", "v3b", "v3c"] {
        assert_eq!(
            comments(&format!("SELECT * FROM {v}")),
            pairs(&[("a", "wa"), ("b", "NULL")]),
            "{v}"
        );
    }
    assert_eq!(
        comments("SELECT * FROM v3e"),
        pairs(&[("a", "wa2"), ("b", "later")])
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

/// Field metadata (the comment, the declared `CHAR`/`VARCHAR` type) where a column is renamed by position, copied
/// as a grouping key or repeated in a set operation, and on temporary views with a column list of their own
/// (Spark 4.2.0: `column-comments/probe5.out`).
#[test]
fn column_metadata_follows_renames_keys_and_repeated_columns() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (a INT COMMENT 'ca', b STRING COMMENT 'cb') USING parquet",
    );
    run(
        &s,
        "CREATE TABLE c (k CHAR(3) COMMENT 'ck', n VARCHAR(4) COMMENT 'cn', v INT) USING parquet",
    );
    run(&s, "INSERT INTO c VALUES ('a', 'x', 1), ('bb', 'yy', 2)");
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
    let mut wrong = Vec::new();
    // A view renamed by position over repeated names, and temporary views with their own column list.
    for (ddl, q, want) in [
        (
            "CREATE VIEW vr (x, y COMMENT 'cy') AS SELECT a, a FROM t",
            "SELECT * FROM vr",
            &[("x", "ca"), ("y", "cy")][..],
        ),
        (
            "CREATE VIEW vr2 (x COMMENT 'cx', y) AS SELECT a, a FROM t",
            "SELECT * FROM vr2",
            &[("x", "cx"), ("y", "ca")],
        ),
        (
            "CREATE VIEW vr3 AS SELECT a, a AS a2, b FROM t",
            "SELECT * FROM vr3",
            &[("a", "ca"), ("a2", "ca"), ("b", "cb")],
        ),
        (
            "CREATE TEMPORARY VIEW tv (x COMMENT 'tx', y) AS SELECT a, b FROM t",
            "SELECT * FROM tv",
            &[("x", "tx"), ("y", "cb")],
        ),
        (
            "CREATE TEMPORARY VIEW tv2 (x, y COMMENT 'ty') AS SELECT a, a FROM t",
            "SELECT * FROM tv2",
            &[("x", "ca"), ("y", "ty")],
        ),
        (
            "CREATE GLOBAL TEMPORARY VIEW gv (x COMMENT 'gx', y) AS SELECT a, b FROM t",
            "SELECT * FROM global_temp.gv",
            &[("x", "gx"), ("y", "cb")],
        ),
        (
            "CREATE TEMPORARY VIEW tvc (kk COMMENT 'tk', nn) AS SELECT k, n FROM c",
            "SELECT * FROM tvc",
            &[("kk", "tk"), ("nn", "cn")],
        ),
    ] {
        run(&s, ddl);
        let got = comments(q);
        if got != pairs(want) {
            wrong.push(format!("{q}: got {got:?}"));
        }
    }
    // The declared CHAR type after a grouping key copy or a repeated set-operation column: the comparison
    // pads the literal to the column's length.
    for (q, want) in [
        (
            "SELECT k FROM c GROUP BY ROLLUP(k) HAVING k = 'a'",
            &["a  "][..],
        ),
        (
            "SELECT k FROM (SELECT k, sum(v) FROM c GROUP BY ROLLUP(k)) WHERE k = 'a'",
            &["a  "],
        ),
        (
            "SELECT k FROM (SELECT k, sum(v) FROM c GROUP BY CUBE(k)) WHERE k = 'a'",
            &["a  "],
        ),
        (
            "SELECT k FROM (SELECT k, n FROM c GROUP BY GROUPING SETS ((k), (n))) WHERE k = 'a'",
            &["a  "],
        ),
        (
            "SELECT q FROM (SELECT k, k FROM c UNION SELECT k, k FROM c) s(p, q) WHERE q = 'a'",
            &["a  "],
        ),
        (
            "SELECT q FROM (SELECT k, k FROM c UNION ALL SELECT k, k FROM c) s(p, q) WHERE q = 'a'",
            &["a  ", "a  "],
        ),
        (
            "SELECT q FROM (SELECT k, k FROM c INTERSECT SELECT k, k FROM c) s(p, q) WHERE q = 'a'",
            &["a  "],
        ),
        (
            "SELECT q FROM (SELECT k, k FROM c EXCEPT SELECT k, n FROM c) s(p, q) WHERE q = 'a'",
            &["a  "],
        ),
    ] {
        let got: Vec<String> = rows(&s, q).into_iter().map(|r| r[0].clone()).collect();
        if got != want {
            wrong.push(format!("{q}: got {got:?}"));
        }
    }
    // CTAS keeps the declared types and comments of the key copies and repeated union columns.
    let desc = |name: &str| -> Vec<(String, String, String)> {
        rows(&s, &format!("DESC {name}"))
            .into_iter()
            .map(|r| (r[0].clone(), r[1].clone(), r[2].clone()))
            .collect()
    };
    for (name, q, want) in [
        (
            "x1",
            "SELECT k, sum(v) s FROM c GROUP BY ROLLUP(k)",
            &[("k", "char(3)", "ck"), ("s", "bigint", "")][..],
        ),
        (
            "x2",
            "SELECT k, n, sum(v) s FROM c GROUP BY CUBE(k, n)",
            &[
                ("k", "char(3)", "ck"),
                ("n", "varchar(4)", "cn"),
                ("s", "bigint", ""),
            ],
        ),
        (
            "x3",
            "SELECT k AS p, k AS q FROM c UNION SELECT k, k FROM c",
            &[("p", "char(3)", "ck"), ("q", "char(3)", "ck")],
        ),
        (
            "x4",
            "SELECT k AS p, k AS q FROM c UNION ALL SELECT k, k FROM c",
            &[("p", "char(3)", "ck"), ("q", "char(3)", "ck")],
        ),
    ] {
        run(&s, &format!("CREATE TABLE {name} USING parquet AS {q}"));
        let got = desc(name);
        let want: Vec<(String, String, String)> = want
            .iter()
            .map(|(a, b, c)| ((*a).to_owned(), (*b).to_owned(), (*c).to_owned()))
            .collect();
        if got != want {
            wrong.push(format!("{name}: got {got:?}"));
        }
    }
    // A DataFrame temporary view over a commented CHAR column keeps the comment and the padding.
    block_on(s.table("c").create_or_replace_temp_view("dv")).unwrap();
    let got = comments("SELECT * FROM dv");
    if got != pairs(&[("k", "ck"), ("n", "cn"), ("v", "NULL")]) {
        wrong.push(format!("dv: got {got:?}"));
    }
    let got: Vec<String> = rows(&s, "SELECT k FROM dv WHERE k = 'a'")
        .into_iter()
        .map(|r| r[0].clone())
        .collect();
    if got != ["a  "] {
        wrong.push(format!("dv k = 'a': got {got:?}"));
    }
    // A view over repeated names reads its stored types (and comments) after the column under it changes type.
    run(&s, "CREATE TABLE r (a INT COMMENT 'ra') USING parquet");
    for (v, mode) in [
        ("vrc", "COMPENSATION"),
        ("vrb", "BINDING"),
        ("vrt", "TYPE EVOLUTION"),
    ] {
        run(
            &s,
            &format!(
                "CREATE VIEW {v} (x, y COMMENT 'vy') WITH SCHEMA {mode} AS SELECT a, a FROM r"
            ),
        );
    }
    run(&s, "DROP TABLE r");
    run(&s, "CREATE TABLE r (a BIGINT COMMENT 'rb') USING parquet");
    let typed = |q: &str| -> Vec<String> {
        rows(&s, &format!("DESC QUERY {q}"))
            .into_iter()
            .map(|r| format!("{} {} {}", r[0], r[1], r[2]))
            .collect()
    };
    for (v, want) in [
        ("vrc", ["x int ra", "y int vy"]),
        ("vrt", ["x bigint ra", "y bigint vy"]),
    ] {
        let got = typed(&format!("SELECT * FROM {v}"));
        if got != want {
            wrong.push(format!("{v}: got {got:?}"));
        }
    }
    let got = class(&s, "SELECT * FROM vrb");
    if got != "CANNOT_UP_CAST_DATATYPE" {
        wrong.push(format!("vrb: got {got}"));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// `spark.sql.collation.schemaLevel.enabled` (owner decision 2026-11-07; Spark 4.2.0
/// `research/spark/schema-collation/probe2.out`): off (the default) a schema's `DEFAULT COLLATION` is refused; on (a
/// session `SET` too), the schema stores and shows it after its comment, `ALTER` changes it, and a table, view or
/// function in it is refused, since Krishiv does not apply the default to their strings yet.
#[test]
fn schema_level_collations_are_stored_only_when_enabled() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    assert_eq!(
        class(&s, "CREATE SCHEMA c DEFAULT COLLATION UNICODE"),
        "UNSUPPORTED_FEATURE.SCHEMA_LEVEL_COLLATIONS"
    );
    run(&s, "SET spark.sql.collation.schemaLevel.enabled = true");
    run(&s, "CREATE SCHEMA c DEFAULT COLLATION unicode");
    let shown = rows(&s, "DESCRIBE SCHEMA c");
    assert_eq!(shown[2][0], "Comment", "{shown:?}");
    assert_eq!(shown[3], ["Collation", "UNICODE"], "{shown:?}");
    run(&s, "ALTER SCHEMA c DEFAULT COLLATION UTF8_LCASE");
    let shown = rows(&s, "DESCRIBE SCHEMA EXTENDED c");
    assert_eq!(shown[3], ["Collation", "UTF8_LCASE"], "{shown:?}");
    run(&s, "CREATE TABLE default.src (a STRING) USING parquet");
    for sql in [
        "CREATE TABLE c.t (a STRING) USING parquet",
        "CREATE TABLE c.t2 USING parquet AS SELECT 'x' AS a",
        "CREATE VIEW c.v AS SELECT 'x' AS a",
        "CREATE TABLE c.t3 LIKE default.src",
        "CREATE FUNCTION c.f(x STRING) RETURNS STRING RETURN x",
    ] {
        assert_eq!(class(&s, sql), "KRISHIV_UNSUPPORTED_FEATURE", "{sql}");
    }
    assert_eq!(
        class(&s, "CREATE SCHEMA d DEFAULT COLLATION nope"),
        "COLLATION_INVALID_NAME"
    );
}

/// `SHOW TABLE EXTENDED` of a temporary view made by `CREATE TEMPORARY VIEW .. AS <query>` prints the query text,
/// the catalog and namespace and the output columns (Spark 4.2.0 `view-schema-binding-config`); a DataFrame view
/// has no text and is refused.
#[test]
fn show_table_extended_of_a_temporary_view_prints_its_text() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TEMPORARY VIEW tv AS SELECT 1");
    let shown = rows(&s, "SHOW TABLE EXTENDED LIKE 'tv'");
    let info = shown[0][3].clone();
    for want in [
        "Table: tv",
        "Type: VIEW",
        "View Text: SELECT 1",
        "View Catalog and Namespace: spark_catalog.default",
        "View Query Output Columns: [`1`]",
        " |-- 1: integer (nullable = false)",
    ] {
        assert!(info.contains(want), "{want} not in {info}");
    }
    assert_eq!(shown[0][2], "true");
}
