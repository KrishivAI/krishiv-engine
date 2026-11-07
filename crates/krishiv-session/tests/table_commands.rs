// SPDX-License-Identifier: Apache-2.0
//! `CREATE TABLE`, `DROP TABLE`, `INSERT` and `TRUNCATE TABLE` on the
//! permanent catalog through `Session::sql`: what the tables read back, the
//! files they leave, and Spark's error classes.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

/// A session whose warehouse is `dir`.
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
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_err()
}

fn class(s: &Session, sql: &str) -> String {
    err(s, sql).error_class().name().to_owned()
}

/// The first column of `sql` as strings (`~` for NULL).
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
                    ArrowType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                    other => panic!("{sql}: unexpected type {other}"),
                }
            });
        }
    }
    out
}

#[test]
fn create_insert_select_and_drop_a_parquet_table() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT, b STRING) USING parquet");
    assert!(dir.path().join("t").is_dir());
    run(&s, "INSERT INTO t VALUES (1, 'x'), (2, 'y')");
    run(&s, "INSERT INTO t VALUES (3, NULL)");
    assert_eq!(
        col0(
            &s,
            "SELECT concat_ws('|', CAST(a AS STRING), b) FROM t ORDER BY a"
        ),
        ["1|x", "2|y", "3"]
    );
    // Qualified spellings find the same table.
    assert_eq!(
        col0(&s, "SELECT CAST(count(*) AS STRING) FROM default.t"),
        ["3"]
    );
    assert_eq!(
        col0(
            &s,
            "SELECT CAST(count(*) AS STRING) FROM spark_catalog.default.t"
        ),
        ["3"]
    );
    run(&s, "INSERT OVERWRITE t VALUES (9, 'z')");
    assert_eq!(
        col0(&s, "SELECT concat_ws('|', CAST(a AS STRING), b) FROM t"),
        ["9|z"]
    );
    run(&s, "DROP TABLE t");
    assert!(!dir.path().join("t").exists());
    assert_eq!(class(&s, "SELECT * FROM t"), "TABLE_OR_VIEW_NOT_FOUND");
    run(&s, "DROP TABLE IF EXISTS t");
    assert_eq!(class(&s, "DROP TABLE t"), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn csv_tables_read_back_what_they_were_given() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE c (id BIGINT, name STRING) USING csv");
    run(&s, "INSERT INTO c VALUES (1, 'a'), (2, 'b')");
    assert_eq!(
        col0(
            &s,
            "SELECT concat_ws('|', CAST(id AS STRING), name) FROM c ORDER BY id"
        ),
        ["1|a", "2|b"]
    );
}

/// Spark reads a CSV table without a header unless the table's options ask for one, so a directory with one row
/// per file reads every row, not none.
#[test]
fn csv_tables_read_files_without_a_header_line() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    std::fs::create_dir(&data).unwrap();
    std::fs::write(data.join("part-00000.csv"), "1\n").unwrap();
    std::fs::write(data.join("part-00001.csv"), "2\n").unwrap();
    let s = session(&dir.path().join("wh"));
    run(
        &s,
        &format!(
            "CREATE TABLE e (d INT) USING csv LOCATION '{}'",
            data.display()
        ),
    );
    assert_eq!(
        col0(&s, "SELECT CAST(d AS STRING) FROM e ORDER BY d"),
        ["1", "2"]
    );
}

/// The text of the `part-*` files in `dir`.
fn part_text(dir: &std::path::Path) -> String {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("part-"))
        })
        .map(|p| std::fs::read_to_string(p).unwrap())
        .collect()
}

/// What a CSV table writes is Spark's text: no header line by default, and one when the table asks for it.
#[test]
fn csv_tables_write_a_header_line_only_when_asked() {
    let dir = tempfile::tempdir().unwrap();
    let wh = dir.path().join("wh");
    let s = session(&wh);
    run(&s, "CREATE TABLE plain (a INT, b INT) USING csv");
    run(&s, "INSERT INTO plain VALUES (1, 2)");
    assert_eq!(part_text(&wh.join("plain")), "1,2\n");
    run(
        &s,
        "CREATE TABLE headed (a INT, b INT) USING csv OPTIONS (header 'true')",
    );
    run(&s, "INSERT INTO headed VALUES (1, 2)");
    assert_eq!(part_text(&wh.join("headed")), "a,b\n1,2\n");
    assert_eq!(col0(&s, "SELECT concat_ws('|', a, b) FROM headed"), ["1|2"]);
    run(&s, "CREATE TABLE ctas USING csv AS SELECT 1 AS x");
    assert_eq!(part_text(&wh.join("ctas")), "1\n");
    assert_eq!(col0(&s, "SELECT CAST(x AS STRING) FROM ctas"), ["1"]);
    run(&s, "CREATE TABLE liked LIKE plain");
    run(&s, "INSERT INTO liked VALUES (3, 4)");
    assert_eq!(part_text(&wh.join("liked")), "3,4\n");
    // LIKE copies the options, as Spark 4.2.0 does.
    run(
        &s,
        "CREATE TABLE src (a INT, b INT) USING csv OPTIONS (header 'true', sep ';')",
    );
    run(&s, "CREATE TABLE liked_options LIKE src");
    run(&s, "INSERT INTO liked_options VALUES (1, 2)");
    assert_eq!(part_text(&wh.join("liked_options")), "a;b\n1;2\n");
}

#[test]
fn create_table_errors() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    assert_eq!(
        class(&s, "CREATE TABLE t (a INT)"),
        "TABLE_OR_VIEW_ALREADY_EXISTS"
    );
    run(&s, "CREATE TABLE IF NOT EXISTS t (a INT, b INT)");
    assert_eq!(col0(&s, "SELECT CAST(count(*) AS STRING) FROM t"), ["0"]);
    assert_eq!(
        class(&s, "CREATE TABLE d (a INT) USING delta"),
        "DATA_SOURCE_NOT_FOUND"
    );
    assert_eq!(
        class(&s, "CREATE TABLE h (a INT) STORED AS parquet"),
        "NOT_SUPPORTED_COMMAND_WITHOUT_HIVE_SUPPORT"
    );
    assert_eq!(class(&s, "CREATE TABLE nodb.x (a INT)"), "SCHEMA_NOT_FOUND");
    assert_eq!(
        class(&s, "CREATE TABLE dup (a INT, A INT)"),
        "COLUMN_ALREADY_EXISTS"
    );
    // File-source tables cannot be replaced, as in Spark; nothing is written or lost.
    run(&s, "INSERT INTO t VALUES (1)");
    assert_eq!(
        class(&s, "CREATE OR REPLACE TABLE t (x STRING)"),
        "UNSUPPORTED_FEATURE.TABLE_OPERATION"
    );
    assert_eq!(
        class(&s, "CREATE OR REPLACE TABLE t AS SELECT 'x' AS x"),
        "UNSUPPORTED_FEATURE.TABLE_OPERATION"
    );
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM t"), ["1"]);
}

#[test]
fn insert_matches_columns_and_applies_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (a INT, b STRING, c DECIMAL(5,2) DEFAULT 1.5) USING parquet",
    );
    run(&s, "INSERT INTO t VALUES (1, 'x', 2.25)");
    run(&s, "INSERT INTO t (b, a) VALUES ('y', 2)");
    run(
        &s,
        "INSERT INTO t BY NAME SELECT 'z' AS b, 3 AS a, 4.5 AS c",
    );
    assert_eq!(
        col0(
            &s,
            "SELECT concat_ws('|', CAST(a AS STRING), b, CAST(c AS STRING)) FROM t ORDER BY a"
        ),
        ["1|x|2.25", "2|y|1.50", "3|z|4.50"]
    );
    assert_eq!(
        class(&s, "INSERT INTO t VALUES (1, 'x')"),
        "INSERT_COLUMN_ARITY_MISMATCH.NOT_ENOUGH_DATA_COLUMNS"
    );
    assert_eq!(
        class(&s, "INSERT INTO t VALUES (1, 'x', 1, 4)"),
        "INSERT_COLUMN_ARITY_MISMATCH.TOO_MANY_DATA_COLUMNS"
    );
    assert_eq!(
        class(&s, "INSERT INTO t VALUES ('abc', 'o', 1)"),
        "INCOMPATIBLE_DATA_FOR_TABLE.CANNOT_SAFELY_CAST"
    );
    assert_eq!(
        class(&s, "INSERT INTO t (a, nope) VALUES (1, 2)"),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    assert_eq!(
        class(&s, "INSERT INTO missing VALUES (1)"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn insert_into_a_view_and_other_wrong_targets() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TEMPORARY VIEW v AS SELECT 1 AS a");
    assert_eq!(
        class(&s, "INSERT INTO v VALUES (1)"),
        "EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"
    );
    // `DROP TABLE` drops a temporary view, as in Spark.
    run(&s, "DROP TABLE v");
    assert_eq!(class(&s, "SELECT * FROM v"), "TABLE_OR_VIEW_NOT_FOUND");
    run(&s, "CREATE TABLE t (a INT)");
    assert_eq!(
        class(&s, "INSERT INTO t PARTITION (a = 1) VALUES (1)"),
        "PARTITIONS_NOT_FOUND"
    );
}

#[test]
fn ctas_writes_then_adds_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE u USING parquet AS SELECT 1 AS id, 'q' AS s",
    );
    assert_eq!(
        col0(&s, "SELECT concat_ws('|', CAST(id AS STRING), s) FROM u"),
        ["1|q"]
    );
    // A failing query leaves no table and no files.
    assert_eq!(
        class(&s, "CREATE TABLE bad AS SELECT * FROM nope"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    assert!(!dir.path().join("bad").exists());
    assert_eq!(class(&s, "SELECT * FROM bad"), "TABLE_OR_VIEW_NOT_FOUND");
}

/// A CTAS that fails while writing removes the directory it made, so the
/// same statement can be retried (and then succeeds).
#[test]
fn a_failed_ctas_leaves_no_directory_and_can_be_retried() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "SET spark.sql.ansi.enabled = true");
    let _ = err(
        &s,
        "CREATE TABLE w USING parquet AS SELECT CAST('x' AS INT) AS a",
    );
    assert!(!dir.path().join("w").exists());
    run(&s, "CREATE TABLE w USING parquet AS SELECT 1 AS a");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM w"), ["1"]);
}

#[test]
fn create_table_like_copies_columns() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE a (x INT, y STRING)");
    run(&s, "CREATE TABLE b LIKE a");
    run(&s, "INSERT INTO b VALUES (1, 'k')");
    assert_eq!(col0(&s, "SELECT y FROM b"), ["k"]);
    run(&s, "CREATE TEMPORARY VIEW v AS SELECT 1 AS p, 'q' AS r");
    run(&s, "CREATE TABLE c LIKE v");
    run(&s, "INSERT INTO c VALUES (2, 'r')");
    assert_eq!(col0(&s, "SELECT r FROM c"), ["r"]);
    assert_eq!(
        class(&s, "CREATE TABLE d LIKE nope"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn create_table_like_a_view_takes_the_default_source() {
    let dir = tempfile::tempdir().unwrap();
    let wh = dir.path().join("wh");
    let s = session(&wh);
    // A view has no provider: Spark 4.2.0's `CreateTableLikeCommand` uses
    // `spark.sql.sources.default` for a view-like source.
    run(&s, "SET spark.sql.sources.default=csv");
    run(&s, "CREATE TEMPORARY VIEW v AS SELECT 1 AS p, 'q' AS r");
    run(&s, "CREATE TABLE c LIKE v");
    run(&s, "INSERT INTO c VALUES (2, 'r')");
    assert_eq!(part_text(&wh.join("c")), "2,r\n");
    run(&s, "CREATE VIEW pv AS SELECT 1 AS p, 'q' AS r");
    run(&s, "CREATE TABLE d LIKE pv");
    run(&s, "INSERT INTO d VALUES (3, 's')");
    assert_eq!(part_text(&wh.join("d")), "3,s\n");
    // An explicit USING never consults the default.
    run(&s, "SET spark.sql.sources.default=nonexistent");
    run(&s, "CREATE TABLE e LIKE v USING csv");
    run(&s, "INSERT INTO e VALUES (5, 't')");
    assert_eq!(part_text(&wh.join("e")), "5,t\n");
    run(&s, "SET spark.sql.sources.default=csv");
    // A table keeps its own provider.
    run(&s, "CREATE TABLE j (a INT) USING json");
    run(&s, "CREATE TABLE k LIKE j");
    run(&s, "INSERT INTO k VALUES (4)");
    assert_eq!(part_text(&wh.join("k")), "{\"a\":4}\n");
}

#[test]
fn truncate_keeps_the_directory() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1), (2)");
    run(&s, "TRUNCATE TABLE t");
    assert!(dir.path().join("t").is_dir());
    assert_eq!(col0(&s, "SELECT CAST(count(*) AS STRING) FROM t"), ["0"]);
    run(&s, "INSERT INTO t VALUES (5)");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM t"), ["5"]);
    assert_eq!(class(&s, "TRUNCATE TABLE nope"), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn external_tables_keep_their_files_when_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let loc = dir.path().join("ext");
    run(
        &s,
        &format!(
            "CREATE EXTERNAL TABLE e (a INT) USING parquet LOCATION '{}'",
            loc.display()
        ),
    );
    run(&s, "INSERT INTO e VALUES (7)");
    run(&s, "DROP TABLE e");
    assert!(loc.is_dir());
    // Re-created over the same files, the rows are still there.
    run(
        &s,
        &format!(
            "CREATE EXTERNAL TABLE e (a INT) USING parquet LOCATION '{}'",
            loc.display()
        ),
    );
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM e"), ["7"]);
}

#[test]
fn temporary_views_shadow_tables_of_the_same_bare_name() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1)");
    run(&s, "CREATE TEMPORARY VIEW t AS SELECT 99 AS a");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM t"), ["99"]);
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM default.t"), ["1"]);
    assert_eq!(
        class(&s, "INSERT INTO t VALUES (2)"),
        "EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"
    );
}

#[test]
fn a_session_without_a_warehouse_uses_its_own_directory() {
    let s = Session::new().unwrap();
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (3)");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM t"), ["3"]);
    let _ = ErrorClass::DivideByZero;
}

#[test]
fn a_failed_create_leaves_no_metadata() {
    let dir = tempfile::tempdir().unwrap();
    // The warehouse is below a regular file, so the table directory cannot be created.
    let blocker = dir.path().join("file");
    std::fs::write(&blocker, "x").unwrap();
    let s = session(&blocker.join("wh"));
    assert_eq!(class(&s, "CREATE TABLE t (a INT)"), "INTERNAL_ERROR");
    assert_eq!(class(&s, "SELECT * FROM t"), "TABLE_OR_VIEW_NOT_FOUND");
    // A table of the same name can be created once the warehouse is usable.
    s.conf()
        .set(
            "spark.sql.warehouse.dir",
            dir.path().join("ok").to_str().unwrap(),
        )
        .unwrap();
    run(&s, "CREATE TABLE t (a INT)");
}

#[test]
fn store_overflow_is_an_error_even_with_ansi_off() {
    // Spark 4.2 verified: the store assignment is strict whatever `spark.sql.ansi.enabled` says.
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    s.conf().set("spark.sql.ansi.enabled", "false").unwrap();
    run(&s, "CREATE TABLE t (s SMALLINT)");
    let e = err(&s, "INSERT INTO t SELECT CAST(100000 AS BIGINT)");
    assert_eq!(
        e.error_class().name(),
        "CAST_OVERFLOW_IN_TABLE_INSERT",
        "{e}"
    );
    assert!(col0(&s, "SELECT CAST(s AS STRING) FROM t").is_empty());
}

#[test]
fn dropping_a_table_after_the_warehouse_moved_removes_its_files() {
    let old = tempfile::tempdir().unwrap();
    let new = tempfile::tempdir().unwrap();
    let s = session(old.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1)");
    s.conf()
        .set("spark.sql.warehouse.dir", new.path().to_str().unwrap())
        .unwrap();
    run(&s, "DROP TABLE t");
    assert!(!old.path().join("t").exists());
    assert_eq!(class(&s, "DROP TABLE t"), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn not_null_is_dropped_for_file_tables_as_in_spark() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT NOT NULL, b INT)");
    run(&s, "INSERT INTO t VALUES (NULL, 2)");
    run(&s, "INSERT INTO t (b) VALUES (3)");
    assert_eq!(
        col0(&s, "SELECT CAST(count(*) AS STRING) FROM t WHERE a IS NULL"),
        ["2"]
    );
}

#[test]
fn a_query_overflow_is_reported_as_its_own() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a BIGINT)");
    let e = err(&s, "INSERT INTO t SELECT CAST(3000000000 AS INT)");
    assert_eq!(e.error_class().name(), "CAST_OVERFLOW", "{e}");
}

#[test]
fn writes_and_drops_clear_cached_reads() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1)");
    let df = s.table("t").cache();
    let rows = |df: &krishiv_session::DataFrame| {
        block_on(async { df.collect().await }).map(|b| {
            b.iter()
                .map(arrow_array::RecordBatch::num_rows)
                .sum::<usize>()
        })
    };
    assert_eq!(rows(&df).unwrap(), 1);
    run(&s, "INSERT INTO t VALUES (2)");
    assert_eq!(rows(&df).unwrap(), 2);
    run(&s, "TRUNCATE TABLE t");
    assert_eq!(rows(&df).unwrap(), 0);
    run(&s, "INSERT INTO t VALUES (5)");
    assert_eq!(rows(&df).unwrap(), 1);
    run(&s, "DROP TABLE t");
    assert!(rows(&df).is_err());
}

#[test]
fn a_failed_insert_writes_no_files() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    // The cast overflows while the rows are written.
    let e = err(&s, "INSERT INTO t SELECT CAST(2147483648 AS BIGINT)");
    assert_eq!(
        e.error_class().name(),
        "CAST_OVERFLOW_IN_TABLE_INSERT",
        "{e}"
    );
    let files: Vec<_> = std::fs::read_dir(dir.path().join("t"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        files
            .iter()
            .all(|f| f.starts_with('_') || f.starts_with('.')),
        "{files:?}"
    );
    assert_eq!(col0(&s, "SELECT CAST(count(*) AS STRING) FROM t"), ["0"]);
}

#[test]
fn a_query_run_before_an_insert_still_sees_what_it_analysed_and_a_new_one_sees_the_rows() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1)");
    let before = block_on(s.sql("SELECT CAST(count(*) AS STRING) FROM t")).unwrap();
    run(&s, "INSERT INTO t VALUES (2)");
    // A DataFrame is a plan: it lists the table's files when it runs, so it sees both rows.
    let batches = block_on(before.collect()).unwrap();
    let c = batches[0].column(0);
    let text = match c.data_type() {
        ArrowType::Utf8View => c.as_string_view().value(0).to_owned(),
        _ => c.as_string::<i32>().value(0).to_owned(),
    };
    assert_eq!(text, "2");
    run(&s, "DROP TABLE t");
    run(&s, "CREATE TABLE t (a INT, b INT)");
    assert_eq!(col0(&s, "SELECT CAST(count(*) AS STRING) FROM t"), ["0"]);
}

#[test]
fn insert_overwrite_never_loses_the_old_rows_before_the_new_ones_are_complete() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1), (2), (3)");
    let reader = {
        let s = s.clone();
        std::thread::spawn(move || {
            let mut seen = std::collections::BTreeMap::<String, usize>::new();
            let mut errors = Vec::new();
            for _ in 0..200 {
                match block_on(async {
                    s.sql("SELECT CAST(count(*) AS STRING) FROM t")
                        .await?
                        .collect()
                        .await
                }) {
                    Ok(batches) => {
                        let c = batches[0].column(0);
                        let v = match c.data_type() {
                            ArrowType::Utf8View => c.as_string_view().value(0).to_owned(),
                            _ => c.as_string::<i32>().value(0).to_owned(),
                        };
                        *seen.entry(v).or_default() += 1;
                    }
                    Err(e) => errors.push(e.to_string()),
                }
            }
            (seen, errors)
        })
    };
    for _ in 0..20 {
        run(&s, "INSERT OVERWRITE t VALUES (7), (8), (9), (10), (11)");
        run(&s, "INSERT OVERWRITE t VALUES (1), (2), (3)");
    }
    let (seen, errors) = reader.join().unwrap();
    // A reader sees the old rows, the new rows, or (between the two steps of the writer's commit, until the
    // manifest commit of P2-T082b) both: never fewer rows than the smaller side, and never none. The directory
    // listing re-reads a directory in which an entry vanished, so a file replaced mid-listing is replaced in
    // it. Until that commit a reader that listed the old files can also lose one to the overwrite's cleanup:
    // that is the only error allowed (with the commit log, which is the default, the read says
    // `KRISHIV_CONCURRENT_COMMIT` instead: retry it).
    assert!(
        errors
            .iter()
            .all(|e| e.contains("FILE_NOT_EXIST") || e.contains("KRISHIV_CONCURRENT_COMMIT")),
        "{errors:?}"
    );
    let allowed: std::collections::BTreeSet<&str> = ["3", "5", "8"].into();
    assert!(
        seen.keys().all(|k| allowed.contains(k.as_str())),
        "unexpected counts: {seen:?}"
    );
    assert!(!seen.is_empty());
}

#[test]
fn overwriting_a_table_from_itself_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1)");
    assert_eq!(
        class(&s, "INSERT OVERWRITE t SELECT a + 1 FROM t"),
        "UNSUPPORTED_OVERWRITE.TABLE"
    );
    assert_eq!(
        class(
            &s,
            "INSERT OVERWRITE default.t SELECT * FROM (SELECT a FROM spark_catalog.default.t)"
        ),
        "UNSUPPORTED_OVERWRITE.TABLE"
    );
    // The data is untouched, and INSERT INTO from itself is fine.
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM t"), ["1"]);
    run(&s, "INSERT INTO t SELECT a + 1 FROM t");
    run(&s, "CREATE TABLE u (a INT)");
    run(&s, "INSERT OVERWRITE u SELECT a FROM t");
}

#[test]
fn names_that_could_be_paths_are_refused_and_nothing_outside_the_warehouse_is_touched() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("keep.txt"), "x").unwrap();
    let s = session(&dir.path().join("wh"));
    for sql in [
        "CREATE TABLE `/etc` (a INT)".to_owned(),
        format!("CREATE TABLE `{}` (a INT)", outside.path().display()),
        "CREATE TABLE `a/b` (a INT)".to_owned(),
        "CREATE TABLE `a b` (a INT)".to_owned(),
        "CREATE TABLE `a-b` (a INT)".to_owned(),
        "CREATE TABLE `x.y` (a INT)".to_owned(),
        "CREATE TABLE `` (a INT)".to_owned(),
        "CREATE TABLE default.`a/b` AS SELECT 1 AS a".to_owned(),
        "CREATE TABLE `/etc` LIKE default.nope".to_owned(),
    ] {
        let c = class(&s, &sql);
        assert!(
            matches!(
                c.as_str(),
                "INVALID_SCHEMA_OR_RELATION_NAME"
                    | "PARSE_SYNTAX_ERROR"
                    | "TABLE_OR_VIEW_NOT_FOUND"
            ),
            "{sql}: {c}"
        );
    }
    assert_eq!(
        class(&s, "CREATE TABLE `..` (a INT)"),
        "INVALID_ATTRIBUTE_NAME_SYNTAX"
    );
    assert_eq!(
        class(&s, "CREATE TABLE `.` (a INT)"),
        "INVALID_ATTRIBUTE_NAME_SYNTAX"
    );
    assert_eq!(
        class(&s, "CREATE TABLE `/etc` (a INT)"),
        "INVALID_SCHEMA_OR_RELATION_NAME"
    );
    assert_eq!(
        class(&s, "CREATE TABLE d.`..` (a INT)"),
        "INVALID_ATTRIBUTE_NAME_SYNTAX"
    );
    // No table was created, and neither the warehouse nor the other directory was touched.
    assert_eq!(class(&s, "DROP TABLE `/etc`"), "TABLE_OR_VIEW_NOT_FOUND");
    assert_eq!(class(&s, "TRUNCATE TABLE `..`"), "TABLE_OR_VIEW_NOT_FOUND");
    assert!(outside.path().join("keep.txt").is_file());
    assert!(dir.path().exists());
}

#[cfg(unix)]
#[test]
fn a_link_in_the_warehouse_is_never_followed_by_a_delete() {
    let dir = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("keep.txt"), "x").unwrap();
    let wh = dir.path().join("wh");
    let s = session(&wh);
    run(&s, "CREATE TABLE t (a INT)");
    run(&s, "INSERT INTO t VALUES (1)");
    // The table's directory is swapped for a link to a directory outside the warehouse.
    std::fs::remove_dir_all(wh.join("t")).unwrap();
    std::os::unix::fs::symlink(outside.path(), wh.join("t")).unwrap();
    assert_eq!(class(&s, "TRUNCATE TABLE t"), "INTERNAL_ERROR");
    assert_eq!(class(&s, "DROP TABLE t"), "INTERNAL_ERROR");
    assert!(outside.path().join("keep.txt").is_file());
}

#[test]
fn provider_names_and_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE p (a INT) USING PARQUET");
    s.conf().set("spark.sql.sources.default", "csv").unwrap();
    run(&s, "CREATE TABLE q (a INT)");
    run(&s, "INSERT INTO q VALUES (4)");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM q"), ["4"]);
    run(&s, "CREATE TABLE j (a INT) USING json");
    run(&s, "INSERT INTO j VALUES (1)");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM j"), ["1"]);
    run(&s, "CREATE TABLE o (a INT) USING orc");
    assert_eq!(
        class(&s, "INSERT INTO o VALUES (1)"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(
        class(&s, "CREATE TABLE i (a INT) USING iceberg"),
        "DATA_SOURCE_NOT_FOUND"
    );
    assert_eq!(
        class(&s, "CREATE TABLE e (a INT) USING parquet LOCATION ''"),
        "INVALID_EMPTY_LOCATION"
    );
    // A bucketed table is created, but the engine writes no bucketed files.
    run(
        &s,
        "CREATE TABLE b (a INT) USING parquet CLUSTERED BY (a) INTO 2 BUCKETS",
    );
    assert_eq!(
        class(&s, "INSERT INTO b VALUES (1)"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(
        class(&s, "CREATE TEMPORARY TABLE tt (a INT) USING parquet"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(
        class(&s, "CREATE TABLE nocols USING parquet"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn existing_locations_and_names() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    std::fs::create_dir(dir.path().join("taken")).unwrap();
    assert_eq!(
        class(&s, "CREATE TABLE taken (a INT)"),
        "LOCATION_ALREADY_EXISTS"
    );
    assert_eq!(
        class(&s, "CREATE TABLE c.d.e.f (a INT)"),
        "REQUIRES_SINGLE_PART_NAMESPACE"
    );
    assert_eq!(
        class(&s, "CREATE TABLE other_catalog.default.t (a INT)"),
        "REQUIRES_SINGLE_PART_NAMESPACE"
    );
    run(&s, "CREATE TABLE spark_catalog.default.q (a INT)");
    run(&s, "INSERT INTO spark_catalog.default.q VALUES (1)");
    assert_eq!(col0(&s, "SELECT CAST(a AS STRING) FROM q"), ["1"]);
    assert_eq!(
        class(&s, "TRUNCATE TABLE q PARTITION (a = 1)"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    let loc = dir.path().join("ext2");
    run(
        &s,
        &format!(
            "CREATE TABLE ex (a INT) USING parquet LOCATION '{}'",
            loc.display()
        ),
    );
    assert_eq!(
        class(&s, "TRUNCATE TABLE ex"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn insert_column_lists_by_name_and_conflicts() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT, b STRING)");
    run(&s, "INSERT INTO t (b) VALUES ('only b')");
    assert_eq!(
        col0(&s, "SELECT concat_ws('|', CAST(a AS STRING), b) FROM t"),
        ["only b"]
    );
    assert_eq!(
        class(&s, "INSERT INTO t (a, b) VALUES (1)"),
        "INSERT_COLUMN_ARITY_MISMATCH.NOT_ENOUGH_DATA_COLUMNS"
    );
    assert_eq!(
        class(&s, "INSERT INTO t (a) VALUES (1, 2)"),
        "INSERT_COLUMN_ARITY_MISMATCH.TOO_MANY_DATA_COLUMNS"
    );
    assert_eq!(
        class(&s, "INSERT INTO t BY NAME SELECT 1 AS zzz"),
        "INSERT_COLUMN_ARITY_MISMATCH.TOO_MANY_DATA_COLUMNS"
    );
    assert_eq!(
        class(&s, "INSERT INTO t REPLACE WHERE a = 1 VALUES (1, 'x')"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(
        class(
            &s,
            "FROM (SELECT 1 AS a, 'x' AS b) s INSERT INTO t SELECT a, b INSERT INTO t SELECT a, b"
        ),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    // A string literal cannot be stored into an INT column under ANSI store assignment; a wider integer can.
    run(&s, "INSERT INTO t VALUES (CAST(5 AS BIGINT), 7)");
}

/// `DELETE`, `UPDATE` and `MERGE` need the table to exist and then fail as Spark's v1 tables do.
#[test]
fn row_level_changes_are_unsupported_on_file_tables() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT, b INT) USING parquet");
    for (sql, operation) in [
        ("DELETE FROM t WHERE a = 1", "DELETE"),
        ("UPDATE t SET b = 1 WHERE a = 1", "UPDATE TABLE"),
        (
            "MERGE INTO t USING t AS s ON t.a = s.a WHEN MATCHED THEN DELETE",
            "MERGE INTO TABLE",
        ),
    ] {
        let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "UNSUPPORTED_FEATURE.TABLE_OPERATION",
            "{sql}"
        );
        assert_eq!(
            e.message_parameters(),
            [
                ("tableName", "`spark_catalog`.`default`.`t`".to_owned()),
                ("operation", operation.to_owned()),
            ],
            "{sql}"
        );
    }
    assert_eq!(
        class(&s, "DELETE FROM missing WHERE a = 1"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    run(&s, "CREATE TEMPORARY VIEW v AS SELECT 1 AS a");
    assert_eq!(
        class(&s, "DELETE FROM v WHERE a = 1"),
        "EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"
    );
}

/// `ANALYZE TABLE` validates its target and stores nothing; resource commands are refused.
#[test]
fn analyze_validates_and_resource_commands_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT, b INT) USING parquet");
    // A successful ANALYZE returns no rows.
    for sql in [
        "ANALYZE TABLE t COMPUTE STATISTICS",
        "ANALYZE TABLE t COMPUTE STATISTICS NOSCAN",
        "ANALYZE TABLE t COMPUTE STATISTICS FOR ALL COLUMNS",
        "ANALYZE TABLE default.t COMPUTE STATISTICS FOR COLUMNS A, b",
    ] {
        let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
        assert_eq!(
            batches
                .iter()
                .map(arrow_array::RecordBatch::num_rows)
                .sum::<usize>(),
            0,
            "{sql}"
        );
    }
    // Spark gathers statistics for scalar types only, for FOR COLUMNS and FOR ALL COLUMNS alike.
    run(
        &s,
        "CREATE TABLE nested (n INT, arr ARRAY<INT>) USING parquet",
    );
    for sql in [
        "ANALYZE TABLE nested COMPUTE STATISTICS FOR COLUMNS arr",
        "ANALYZE TABLE nested COMPUTE STATISTICS FOR ALL COLUMNS",
    ] {
        let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "UNSUPPORTED_FEATURE.ANALYZE_UNSUPPORTED_COLUMN_TYPE",
            "{sql}"
        );
        assert_eq!(
            e.message_parameters(),
            [
                ("columnType", "\"ARRAY<INT>\"".to_owned()),
                ("columnName", "`arr`".to_owned()),
                ("tableName", "`default`.`nested`".to_owned()),
            ],
            "{sql}"
        );
    }
    run(&s, "ANALYZE TABLE nested COMPUTE STATISTICS FOR COLUMNS n");
    assert_eq!(
        class(&s, "ANALYZE TABLE missing COMPUTE STATISTICS"),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    assert_eq!(
        class(&s, "ANALYZE TABLE t COMPUTE STATISTICS FOR COLUMNS nope"),
        "COLUMN_NOT_FOUND"
    );
    assert_eq!(
        class(&s, "ANALYZE TABLE t PARTITION (a = 1) COMPUTE STATISTICS"),
        "PARTITIONS_NOT_FOUND"
    );
    for sql in ["ADD JAR '/tmp/x.jar'", "LIST FILES", "ADD FILE '/tmp/f'"] {
        assert_eq!(class(&s, sql), "KRISHIV_UNSUPPORTED_FEATURE", "{sql}");
    }
}

/// `UPDATE` and `DELETE` resolve their columns and condition before Spark finds that the table cannot
/// change; an unknown column is the error then, with the analyzer's suggestions.
#[test]
fn row_level_changes_resolve_columns_first() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT, b INT) USING parquet");
    assert_eq!(
        class(&s, "UPDATE t SET nope = 1"),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    assert_eq!(
        class(&s, "UPDATE t SET a = nope"),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    assert_eq!(
        class(&s, "UPDATE t SET a = 1 WHERE nope = 2"),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    assert_eq!(
        class(&s, "DELETE FROM t WHERE nope = 2"),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let e = block_on(async { s.sql("UPDATE t SET zzz = 1").await?.collect().await }).unwrap_err();
    assert!(
        e.message_parameters()
            .iter()
            .any(|(k, v)| *k == "proposal" && v.contains("`a`") && v.contains("`b`")),
        "{e}"
    );
    // A valid statement, `DEFAULT` included, reaches the table-operation error.
    for sql in [
        "UPDATE t SET a = b + 1 WHERE a > 0",
        "UPDATE t SET a = DEFAULT",
        "DELETE FROM t WHERE b IS NULL",
    ] {
        assert_eq!(
            class(&s, sql),
            "UNSUPPORTED_FEATURE.TABLE_OPERATION",
            "{sql}"
        );
    }
}

/// A scalar subquery in `INSERT ... VALUES` is refused after it is resolved: an unknown column inside it
/// is the error first.
#[test]
fn insert_values_refuse_scalar_subqueries() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT) USING parquet");
    assert_eq!(
        class(&s, "INSERT INTO t VALUES ((SELECT 1))"),
        "UNSUPPORTED_SUBQUERY_EXPRESSION_CATEGORY.SCALAR_SUBQUERY_IN_VALUES"
    );
    assert_eq!(
        class(&s, "INSERT INTO t VALUES (1 + (SELECT 2))"),
        "UNSUPPORTED_SUBQUERY_EXPRESSION_CATEGORY.SCALAR_SUBQUERY_IN_VALUES"
    );
    assert_eq!(
        class(&s, "INSERT INTO t VALUES ((SELECT nope))"),
        "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    );
    // Without a subquery nothing changes.
    run(&s, "INSERT INTO t VALUES (1), (2)");
}

/// A CTAS owns the directory it makes: a concurrent or earlier one at the same location is never
/// touched by a failure, and a dropped (cancelled) statement removes only what it made.
#[test]
fn a_failed_ctas_never_removes_a_directory_it_did_not_create() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "SET spark.sql.ansi.enabled = true");
    // Another statement's table directory with data in it.
    let other = dir.path().join("w");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(other.join("keep.txt"), b"data").unwrap();
    let e = err(
        &s,
        "CREATE TABLE w USING parquet AS SELECT CAST('x' AS INT) AS a",
    );
    assert_eq!(e.error_class().name(), "LOCATION_ALREADY_EXISTS", "{e}");
    assert_eq!(std::fs::read(other.join("keep.txt")).unwrap(), b"data");
}

/// The statements of one duplicate-column shape for a name and its repeat (the last one fails).
type Statements = fn(&str, &str) -> Vec<String>;

/// Spark 4.2.0 (local run, 2026-11-07) quotes a repeated column name with `toSQLId(String)`: split at its
/// dots, each part in backticks. A repeated name that is not an attribute name fails with
/// `INVALID_ATTRIBUTE_NAME_SYNTAX`, raised while Spark builds the message. Same at every check.
#[test]
fn a_repeated_column_is_named_by_its_parts() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE base (x INT)");
    let shapes: [(&str, Statements); 8] = [
        ("ctas", |n, m| {
            vec![format!(
                "CREATE TABLE t1 USING parquet AS SELECT 1 AS {n}, 2 AS {m}"
            )]
        }),
        ("columns", |n, m| {
            vec![format!("CREATE TABLE t2 ({n} INT, {m} INT) USING parquet")]
        }),
        ("view", |n, m| {
            vec![format!("CREATE VIEW v1 AS SELECT 1 AS {n}, 2 AS {m}")]
        }),
        ("view columns", |n, m| {
            vec![format!("CREATE VIEW v2 ({n}, {m}) AS SELECT 1, 2")]
        }),
        ("temp view", |n, m| {
            vec![format!("CREATE TEMP VIEW v3 AS SELECT 1 AS {n}, 2 AS {m}")]
        }),
        ("partitioned by", |n, m| {
            vec![format!(
                "CREATE TABLE t3 ({n} INT, y INT) USING parquet PARTITIONED BY ({n}, {m})"
            )]
        }),
        ("typed partition", |n, m| {
            vec![format!(
                "CREATE TABLE t4 ({n} INT) USING parquet PARTITIONED BY ({m} INT)"
            )]
        }),
        ("add columns", |n, m| {
            vec![
                "DROP TABLE IF EXISTS t5".to_owned(),
                format!("CREATE TABLE t5 ({n} INT) USING parquet"),
                format!("ALTER TABLE t5 ADD COLUMNS ({m} INT)"),
            ]
        }),
    ];
    for (n, m, want) in [
        (
            "`a.b`",
            "`a.b`",
            "COLUMN_ALREADY_EXISTS: The column `a`.`b` already exists",
        ),
        (
            "`a.b.c`",
            "`a.b.c`",
            "COLUMN_ALREADY_EXISTS: The column `a`.`b`.`c` already exists",
        ),
        (
            "`A.B`",
            "`a.b`",
            "COLUMN_ALREADY_EXISTS: The column `a`.`b` already exists",
        ),
        (
            "`a b`",
            "`a b`",
            "COLUMN_ALREADY_EXISTS: The column `a b` already exists",
        ),
        (
            "`ab`",
            "`ab`",
            "COLUMN_ALREADY_EXISTS: The column `ab` already exists",
        ),
        (
            "`a``b`",
            "`a``b`",
            "INVALID_ATTRIBUTE_NAME_SYNTAX: Syntax error in the attribute name: a`b.",
        ),
        (
            "`a..b`",
            "`a..b`",
            "INVALID_ATTRIBUTE_NAME_SYNTAX: Syntax error in the attribute name: a..b.",
        ),
        (
            "`a.``b`",
            "`a.``b`",
            "INVALID_ATTRIBUTE_NAME_SYNTAX: Syntax error in the attribute name: a.`b.",
        ),
    ] {
        for (shape, sqls) in &shapes {
            let sqls = sqls(n, m);
            let (last, setup) = sqls.split_last().unwrap();
            for sql in setup {
                run(&s, sql);
            }
            let e = err(&s, last);
            let got = format!("{}: {}", e.error_class().name(), e.message());
            assert!(got.starts_with(want), "{shape} {n}: {got}");
        }
    }
}

/// The DataFrame paths and a view's schema evolution quote a repeated name the same way (Spark 4.2.0
/// local run, 2026-11-07: `saveAsTable`, `partitionBy`, `unionByName` and an `EVOLUTION` view).
#[test]
fn a_repeated_column_is_named_by_its_parts_on_dataframe_paths() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let message = |r: Result<(), KrishivError>| {
        let e = r.unwrap_err();
        format!("{}: {}", e.error_class().name(), e.message())
    };
    let want = "COLUMN_ALREADY_EXISTS: The column `a`.`b` already exists";
    let dup = block_on(s.sql("SELECT 1 AS `a.b`, 2 AS `A.B`")).unwrap();
    let got = message(block_on(dup.write().save_as_table("t6")));
    assert!(got.starts_with(want), "saveAsTable: {got}");
    let out = dir.path().join("dup");
    let got = message(block_on(dup.write().parquet(out.to_str().unwrap())));
    assert!(got.starts_with(want), "parquet: {got}");
    let one = block_on(s.sql("SELECT 1 AS `A.B`, 2 AS y")).unwrap();
    let got = message(block_on(
        one.write()
            .partition_by(["A.B", "a.b"])
            .parquet(out.to_str().unwrap()),
    ));
    assert!(got.starts_with(want), "partitionBy: {got}");
    let got = message(block_on(dup.union_by_name(&dup, false).collect()).map(|_| ()));
    assert!(got.starts_with(want), "unionByName: {got}");
    let bad = block_on(s.sql("SELECT 1 AS `a``b`, 2 AS `a``b`")).unwrap();
    let got = message(block_on(bad.union_by_name(&bad, false).collect()).map(|_| ()));
    assert!(
        got.starts_with("INVALID_ATTRIBUTE_NAME_SYNTAX: Syntax error in the attribute name: a`b."),
        "unionByName: {got}"
    );

    run(&s, "CREATE TABLE e1 (`a.b` INT) USING parquet");
    run(&s, "CREATE TABLE e2 (c2 INT) USING parquet");
    run(
        &s,
        "CREATE VIEW vev WITH SCHEMA EVOLUTION AS SELECT * FROM e1, e2",
    );
    run(&s, "SELECT * FROM vev");
    run(&s, "DROP TABLE e2");
    run(&s, "CREATE TABLE e2 (`A.B` INT) USING parquet");
    let e = err(&s, "SELECT * FROM vev");
    let got = format!("{}: {}", e.error_class().name(), e.message());
    assert!(got.starts_with(want), "evolution: {got}");
}

/// Spark 4.2.0 (local run, 2026-11-07) reports the smallest repeated name, not the first one met, and folds
/// case only without `spark.sql.caseSensitive`, at every check.
#[test]
fn the_smallest_repeated_column_is_reported_and_case_follows_the_session() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let shown = |sql: &str| {
        let e = err(&s, sql);
        format!("{}: {}", e.error_class().name(), e.message())
    };
    let a = "COLUMN_ALREADY_EXISTS: The column `a` already exists";
    for sql in [
        "CREATE TABLE o1 USING parquet AS SELECT 1 AS z, 2 AS z, 3 AS a, 4 AS a",
        "CREATE TABLE o3 USING parquet AS SELECT 1 AS `z``b`, 2 AS `z``b`, 3 AS a, 4 AS a",
        "CREATE TABLE o4 (z INT, z INT, a INT, a INT) USING parquet",
        "CREATE VIEW o5 AS SELECT 1 AS z, 2 AS z, 3 AS a, 4 AS a",
        "CREATE TABLE o6 (z INT, a INT, y INT) USING parquet PARTITIONED BY (z, z, a, a)",
        "CREATE TABLE o7 (z INT) USING parquet PARTITIONED BY (a INT, z INT, A INT)",
    ] {
        assert!(shown(sql).starts_with(a), "{sql}: {}", shown(sql));
    }
    let got =
        shown("CREATE TABLE o2 USING parquet AS SELECT 1 AS z, 2 AS z, 3 AS `a``b`, 4 AS `a``b`");
    assert!(
        got.starts_with("INVALID_ATTRIBUTE_NAME_SYNTAX: Syntax error in the attribute name: a`b."),
        "{got}"
    );
    for (sql, name) in [
        ("CREATE TABLE d1 (`.a` INT, `.a` INT) USING parquet", ".a"),
        (
            "CREATE TABLE d2 USING parquet AS SELECT 1 AS `a.`, 2 AS `a.`",
            "a.",
        ),
    ] {
        let got = shown(sql);
        let want =
            format!("INVALID_ATTRIBUTE_NAME_SYNTAX: Syntax error in the attribute name: {name}.");
        assert!(got.starts_with(&want), "{sql}: {got}");
    }
    // A repeat in `partitionBy` is found before a missing column is.
    let df = block_on(s.sql("SELECT 1 AS a, 2 AS z, 3 AS y")).unwrap();
    let out = dir.path().join("out");
    for cols in [["missing", "a", "a"], ["a", "a", "missing"]] {
        let e = block_on(df.write().partition_by(cols).parquet(out.to_str().unwrap())).unwrap_err();
        let got = format!("{}: {}", e.error_class().name(), e.message());
        assert!(got.starts_with(a), "{cols:?}: {got}");
    }

    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    run(&s, "CREATE TABLE c1 (A INT, a INT) USING parquet");
    run(&s, "CREATE TABLE c3 (A INT) USING parquet");
    run(&s, "ALTER TABLE c3 ADD COLUMNS (a INT)");
    run(
        &s,
        "CREATE TABLE c4 (A INT) USING parquet PARTITIONED BY (a INT)",
    );
    let upper = "COLUMN_ALREADY_EXISTS: The column `A` already exists";
    for sql in [
        "CREATE TABLE c2 (A INT, A INT) USING parquet",
        "ALTER TABLE c3 ADD COLUMNS (A INT)",
        "CREATE TABLE c5 (A INT) USING parquet PARTITIONED BY (A INT)",
    ] {
        assert!(shown(sql).starts_with(upper), "{sql}: {}", shown(sql));
    }
}
