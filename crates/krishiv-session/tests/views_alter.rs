// SPDX-License-Identifier: Apache-2.0
//! Permanent views and `ALTER TABLE` / `ALTER VIEW` through `Session::sql`.
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

fn col0(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        let c = b.column(0);
        for i in 0..b.num_rows() {
            out.push(if c.is_null(i) {
                "~".to_owned()
            } else {
                match c.data_type() {
                    ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
                    _ => c.as_string::<i32>().value(i).to_owned(),
                }
            });
        }
    }
    out
}

#[test]
fn a_permanent_view_follows_its_tables() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT, b STRING)");
    run(&s, "INSERT INTO t VALUES (1, 'x'), (2, 'y')");
    run(
        &s,
        "CREATE VIEW v (id, name) AS SELECT a, b FROM t WHERE a > 0",
    );
    assert_eq!(
        col0(
            &s,
            "SELECT concat_ws('|', CAST(id AS STRING), name) FROM v ORDER BY id"
        ),
        ["1|x", "2|y"]
    );
    run(&s, "INSERT INTO t VALUES (3, 'z')");
    assert_eq!(
        col0(&s, "SELECT CAST(count(*) AS STRING) FROM default.v"),
        ["3"]
    );
    // A view over a view.
    run(&s, "CREATE VIEW w AS SELECT name FROM v WHERE id > 1");
    assert_eq!(col0(&s, "SELECT name FROM w ORDER BY name"), ["y", "z"]);
    run(
        &s,
        "CREATE OR REPLACE VIEW v AS SELECT a AS id, b AS name FROM t WHERE a > 2",
    );
    assert_eq!(col0(&s, "SELECT name FROM w"), ["z"]);
    run(&s, "DROP VIEW w");
    run(&s, "DROP VIEW IF EXISTS w");
    assert_eq!(class(&s, "SELECT * FROM w"), "TABLE_OR_VIEW_NOT_FOUND");
    assert_eq!(class(&s, "DROP VIEW w"), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn view_names_resolve_in_the_namespace_it_was_created_in() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE DATABASE d");
    run(&s, "USE d");
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (5)");
    run(&s, "CREATE VIEW v AS SELECT a FROM t");
    run(&s, "USE default");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM d.v"), ["5"]);
    // Another table `t` in the namespace now current does not capture the view's bare name.
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (6)");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM d.v"), ["5"]);
}

#[test]
fn view_errors() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "CREATE TEMPORARY VIEW tv AS SELECT 1 AS a");
    assert_eq!(
        class(&s, "CREATE VIEW p AS SELECT * FROM tv"),
        "INVALID_TEMP_OBJ_REFERENCE"
    );
    run(&s, "CREATE VIEW v AS SELECT * FROM t");
    assert_eq!(
        class(&s, "CREATE VIEW v AS SELECT 1"),
        "TABLE_OR_VIEW_ALREADY_EXISTS"
    );
    run(&s, "CREATE VIEW IF NOT EXISTS v AS SELECT 2");
    assert_eq!(
        class(&s, "INSERT INTO v VALUES (1)"),
        "EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"
    );
    assert_eq!(
        class(&s, "TRUNCATE TABLE v"),
        "EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"
    );
    assert_eq!(class(&s, "DROP VIEW t"), "WRONG_COMMAND_FOR_OBJECT_TYPE");
    assert_eq!(
        class(&s, "CREATE OR REPLACE VIEW t AS SELECT 1"),
        "EXPECT_VIEW_NOT_TABLE.NO_ALTERNATIVE"
    );
    assert_eq!(
        class(&s, "ALTER TABLE v SET TBLPROPERTIES ('a' = 'b')"),
        "EXPECT_TABLE_NOT_VIEW.USE_ALTER_VIEW"
    );
    // DROP TABLE refuses a permanent view, as Spark's DropTableCommand does.
    let e = block_on(s.sql("DROP TABLE v")).unwrap_err();
    assert!(
        e.to_string().contains("WRONG_COMMAND_FOR_OBJECT_TYPE"),
        "{e}"
    );
    assert!(e.to_string().contains("default.v"), "{e}");
    run(&s, "DROP VIEW v");
    assert_eq!(class(&s, "SELECT * FROM v"), "TABLE_OR_VIEW_NOT_FOUND");
    assert_eq!(
        class(&s, "CREATE VIEW nodb.x AS SELECT 1"),
        "SCHEMA_NOT_FOUND"
    );
    assert_eq!(
        class(&s, "CREATE VIEW x AS SELECT * FROM nope"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    assert_eq!(
        class(&s, "CREATE GLOBAL TEMPORARY VIEW g (a INT) USING parquet"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(
        class(&s, "CREATE VIEW c (a, b) AS SELECT 1"),
        "CREATE_VIEW_COLUMN_ARITY_MISMATCH.NOT_ENOUGH_DATA_COLUMNS"
    );
    assert_eq!(
        class(&s, "CREATE VIEW c (a) AS SELECT 1, 2"),
        "CREATE_VIEW_COLUMN_ARITY_MISMATCH.TOO_MANY_DATA_COLUMNS"
    );
    assert_eq!(
        class(&s, "CREATE VIEW c AS SELECT 1 AS a, 2 AS A"),
        "COLUMN_ALREADY_EXISTS"
    );
    assert_eq!(
        class(&s, "CREATE VIEW c (x, X) AS SELECT 1, 2"),
        "COLUMN_ALREADY_EXISTS"
    );
    // Schema binding modes run (`tests/views_schema_binding.rs`).
    run(&s, "CREATE VIEW sb WITH SCHEMA BINDING AS SELECT 1");
    assert_eq!(
        class(&s, "CREATE OR REPLACE VIEW IF NOT EXISTS z AS SELECT 1"),
        "CREATE_VIEW_WITH_IF_NOT_EXISTS_AND_REPLACE"
    );
}

#[test]
fn alter_view() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1), (2)");
    run(&s, "CREATE VIEW v AS SELECT a FROM t WHERE a = 1");
    run(&s, "ALTER VIEW v AS SELECT a FROM t WHERE a = 2");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM v"), ["2"]);
    run(&s, "ALTER VIEW v SET TBLPROPERTIES ('k' = 'v')");
    assert_eq!(
        class(&s, "ALTER VIEW t AS SELECT 1"),
        "EXPECT_VIEW_NOT_TABLE.USE_ALTER_TABLE"
    );
    assert_eq!(
        class(&s, "ALTER VIEW nope AS SELECT 1"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    run(&s, "CREATE TEMPORARY VIEW tv AS SELECT 1 AS a");
    assert_eq!(
        class(&s, "ALTER VIEW v AS SELECT * FROM tv"),
        "INVALID_TEMP_OBJ_REFERENCE"
    );
    assert_eq!(
        class(&s, "ALTER VIEW v RENAME TO v2"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn alter_table() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1)");
    run(&s, "ALTER TABLE t ADD COLUMNS (b STRING, c INT)");
    assert_eq!(
        col0(
            &s,
            "SELECT concat_ws('|', CAST(a AS STRING), b, CAST(c AS STRING)) FROM t"
        ),
        ["1"]
    );
    run(&s, "INSERT INTO t VALUES (2, 'x', 3)");
    assert_eq!(
        col0(
            &s,
            "SELECT concat_ws('|', CAST(a AS STRING), b, CAST(c AS STRING)) FROM t ORDER BY a"
        ),
        ["1", "2|x|3"]
    );
    assert_eq!(
        class(&s, "ALTER TABLE t ADD COLUMNS (A INT)"),
        "COLUMN_ALREADY_EXISTS"
    );
    run(&s, "ALTER TABLE t SET TBLPROPERTIES ('p' = 'q', 'r' = 's')");
    run(&s, "ALTER TABLE t UNSET TBLPROPERTIES ('p')");
    assert_eq!(
        class(&s, "ALTER TABLE nope SET TBLPROPERTIES ('a' = 'b')"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    assert_eq!(
        class(&s, "ALTER TABLE t DROP COLUMNS (b)"),
        "UNSUPPORTED_FEATURE.TABLE_OPERATION"
    );
    run(&s, "CREATE VIEW v AS SELECT 1 AS x");
    assert_eq!(
        class(&s, "ALTER TABLE v SET TBLPROPERTIES ('a' = 'b')"),
        "EXPECT_TABLE_NOT_VIEW.USE_ALTER_VIEW"
    );
}

#[test]
fn rename_moves_a_managed_table() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE DATABASE d");
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (9)");
    run(&s, "ALTER TABLE t RENAME TO u");
    assert!(!dir.path().join("t").exists());
    assert!(dir.path().join("u").is_dir());
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM u"), ["9"]);
    assert_eq!(class(&s, "SELECT * FROM t"), "TABLE_OR_VIEW_NOT_FOUND");
    // A table never changes namespace.
    assert_eq!(
        class(&s, "ALTER TABLE u RENAME TO d.w"),
        "_LEGACY_ERROR_TEMP_1073"
    );
    assert!(dir.path().join("u").is_dir());
    run(&s, "CREATE TABLE x (a INT)");
    assert_eq!(
        class(&s, "ALTER TABLE u RENAME TO x"),
        "TABLE_OR_VIEW_ALREADY_EXISTS"
    );
    assert_eq!(
        class(&s, "ALTER TABLE u RENAME TO nodb.y"),
        "SCHEMA_NOT_FOUND"
    );
    // A bare target stays in the source's namespace, whatever is current.
    run(&s, "CREATE TABLE d.m (a INT)");
    run(&s, "ALTER TABLE d.m RENAME TO m2");
    assert!(dir.path().join("d.db").join("m2").is_dir());
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM u"), ["9"]);
}

#[test]
fn view_and_rename_target_names_are_validated() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    assert_eq!(
        class(&s, "CREATE VIEW `a/b` AS SELECT 1"),
        "INVALID_SCHEMA_OR_RELATION_NAME"
    );
    assert_eq!(
        class(&s, "CREATE VIEW `..` AS SELECT 1"),
        "INVALID_ATTRIBUTE_NAME_SYNTAX"
    );
    assert_eq!(
        class(&s, "ALTER TABLE t RENAME TO `../x`"),
        "INVALID_SCHEMA_OR_RELATION_NAME"
    );
    assert!(dir.path().join("t").is_dir());
}

#[test]
fn a_view_keeps_its_columns_when_its_base_table_gains_one() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT, b STRING)");
    run(&s, "INSERT INTO t VALUES (1, 'x')");
    run(&s, "CREATE VIEW v AS SELECT * FROM t");
    run(&s, "CREATE VIEW w (p, q) AS SELECT a, b FROM t");
    run(&s, "ALTER TABLE t ADD COLUMNS (c INT)");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM v"), ["1"]);
    assert_eq!(
        col0(&s, "SELECT concat(CAST(p AS STRING), q) FROM w"),
        ["1x"]
    );
    // `SELECT *` over the view is still the stored two columns.
    let n = block_on(async { s.sql("SELECT * FROM v").await?.collect().await }).unwrap()[0]
        .num_columns();
    assert_eq!(n, 2);
}

#[test]
fn a_view_column_named_with_a_dot_or_star_reads_back() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE VIEW dv AS SELECT 1 AS `a.b`, 2 AS `x*`");
    assert_eq!(
        col0(
            &s,
            "SELECT concat(CAST(`a.b` AS STRING), '|', CAST(`x*` AS STRING)) FROM dv"
        ),
        ["1|2"]
    );
}

#[test]
fn a_temporary_view_shadows_a_permanent_one_of_the_same_bare_name() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE VIEW pv AS SELECT 7 AS x");
    run(&s, "CREATE TEMPORARY VIEW pv AS SELECT 5 AS x");
    assert_eq!(col0(&s, "SELECT CAST(x AS STRING) FROM pv"), ["5"]);
    assert_eq!(col0(&s, "SELECT CAST(x AS STRING) FROM default.pv"), ["7"]);
}

#[test]
fn a_dropped_namespace_is_left_even_when_a_directory_cannot_go() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE DATABASE gone");
    run(&s, "USE gone");
    run(&s, "DROP DATABASE gone");
    assert_eq!(col0(&s, "SELECT current_database()"), ["default"]);
}
