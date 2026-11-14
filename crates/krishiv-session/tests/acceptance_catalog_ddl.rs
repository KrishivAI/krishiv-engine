// SPDX-License-Identifier: Apache-2.0
//! Acceptance fixes of batch B7 (`research/qa/2026-11-14-acceptance-issues.md`): SQL-16 (`DESCRIBE` prints NULL for a column with no
//! comment), SQL-19 and RS-8 (`CREATE TABLE ... USING fmt LOCATION` and `CREATE TEMPORARY VIEW ... USING fmt OPTIONS (path ...)` without
//! a column list read the schema from the files).
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
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

/// Every row of `sql` as optional strings, NULL as `None`.
fn rows(s: &Session, sql: &str) -> Vec<Vec<Option<String>>> {
    let batches: Vec<RecordBatch> = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        c.is_valid(i).then(|| {
                            if let Some(a) = c.as_string_opt::<i32>() {
                                a.value(i).to_owned()
                            } else if let Some(a) = c.as_string_opt::<i64>() {
                                a.value(i).to_owned()
                            } else if let Some(a) = c.as_string_view_opt() {
                                a.value(i).to_owned()
                            } else if let Some(a) =
                                c.as_primitive_opt::<arrow_array::types::Int32Type>()
                            {
                                a.value(i).to_string()
                            } else if let Some(a) =
                                c.as_primitive_opt::<arrow_array::types::Int64Type>()
                            {
                                a.value(i).to_string()
                            } else {
                                panic!(
                                    "a column type this helper does not print: {:?}",
                                    c.data_type()
                                )
                            }
                        })
                    })
                    .collect(),
            );
        }
    }
    out
}

fn some(v: &str) -> Option<String> {
    Some(v.to_owned())
}

#[test]
fn sql_16_describe_prints_null_for_a_column_without_a_comment() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t1 (id INT, name STRING COMMENT 'the name', e STRING COMMENT '') USING parquet",
    );
    assert_eq!(
        rows(&s, "DESCRIBE t1"),
        [
            vec![some("id"), some("int"), None],
            vec![some("name"), some("string"), some("the name")],
            vec![some("e"), some("string"), some("")],
        ]
    );
    // a partition column with no comment is NULL too
    run(
        &s,
        "CREATE TABLE t2 (id INT, p STRING) USING parquet PARTITIONED BY (p)",
    );
    let all = rows(&s, "DESCRIBE t2");
    assert_eq!(all[0], vec![some("id"), some("int"), None]);
    assert_eq!(all.last().unwrap(), &vec![some("p"), some("string"), None]);
    // so a user filtering on `comment IS NULL` finds the columns without one
    let without: Vec<_> = rows(&s, "DESCRIBE t1")
        .into_iter()
        .filter(|r| r[2].is_none())
        .collect();
    assert_eq!(without.len(), 1);
}

#[test]
fn sql_19_create_table_over_partitioned_files_reads_the_schema_and_the_partition_columns() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE src (id INT, p STRING) USING parquet PARTITIONED BY (p)",
    );
    run(&s, "INSERT INTO src VALUES (1, 'x'), (2, 'y')");
    let location = dir.path().join("src");
    let l = location.to_str().unwrap();
    run(&s, &format!("CREATE TABLE pt USING parquet LOCATION '{l}'"));
    // `p` is the table's partition column: it is listed after the data columns, and the partition commands work
    let described = rows(&s, "DESCRIBE pt");
    assert_eq!(described[0], vec![some("id"), some("int"), None]);
    assert!(
        described
            .iter()
            .any(|r| r[0].as_deref() == Some("# Partition Information")),
        "{described:?}"
    );
    // the partitions are registered by MSCK REPAIR TABLE, then the table reads them
    run(&s, "MSCK REPAIR TABLE pt");
    assert_eq!(rows(&s, "SHOW PARTITIONS pt").len(), 2);
    assert_eq!(
        rows(&s, "SELECT id, p FROM pt ORDER BY id"),
        [vec![some("1"), some("x")], vec![some("2"), some("y")]]
    );
    // dropping a partition of an external table forgets it and keeps its files, as Spark does
    run(&s, "ALTER TABLE pt DROP PARTITION (p = 'x')");
    assert_eq!(rows(&s, "SHOW PARTITIONS pt").len(), 1);
    assert_eq!(rows(&s, "SELECT id FROM pt ORDER BY id"), [vec![some("2")]]);
    assert!(
        std::fs::read_dir(location.join("p=x"))
            .unwrap()
            .any(|f| f.is_ok_and(|f| f.path().extension().is_some_and(|e| e == "parquet"))),
        "the data files of an external table stay"
    );
}

#[test]
fn sql_19_a_managed_table_without_columns_is_still_refused() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let e = block_on(s.sql("CREATE TABLE nothing USING parquet")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToInferSchema);
}

#[test]
fn rs_8_a_view_over_a_csv_file_infers_its_schema() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let f = dir.path().join("zones.csv");
    std::fs::write(&f, "LocationID,Borough\n1,Queens\n2,Manhattan\n3,Queens\n").unwrap();
    let p = f.to_str().unwrap();
    run(
        &s,
        &format!(
            "CREATE TEMPORARY VIEW zones USING csv OPTIONS (path '{p}', header 'true', inferSchema 'true')"
        ),
    );
    assert_eq!(
        rows(&s, "DESCRIBE zones"),
        [
            vec![some("LocationID"), some("int"), None],
            vec![some("Borough"), some("string"), None],
        ]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT Borough, sum(LocationID) FROM zones GROUP BY Borough ORDER BY 1"
        ),
        [
            vec![some("Manhattan"), some("2")],
            vec![some("Queens"), some("4")]
        ]
    );
    // without inferSchema every column is a string, as in Spark
    run(
        &s,
        &format!("CREATE TEMPORARY VIEW raw USING csv OPTIONS (path '{p}', header 'true')"),
    );
    assert_eq!(
        rows(&s, "DESCRIBE raw")[0],
        vec![some("LocationID"), some("string"), None]
    );
}

#[test]
fn rs_8_a_view_over_parquet_files_infers_its_schema_and_a_missing_path_is_still_reported() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE src (id INT, name STRING) USING parquet");
    run(&s, "INSERT INTO src VALUES (1, 'a'), (2, 'b')");
    let l = dir.path().join("src");
    run(
        &s,
        &format!(
            "CREATE TEMPORARY VIEW pv USING parquet OPTIONS (path '{}')",
            l.to_str().unwrap()
        ),
    );
    assert_eq!(rows(&s, "SELECT count(*) FROM pv"), [vec![some("2")]]);
    let e = block_on(
        s.sql("CREATE TEMPORARY VIEW gone USING parquet OPTIONS (path '/nonexistent/krishiv')"),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::PathNotFound);
    // a view with neither columns nor a path has nothing to infer from
    let e = block_on(s.sql("CREATE TEMPORARY VIEW nopath USING parquet")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn rs_8_a_view_over_a_partitioned_directory_reads_every_row_with_its_partition_column() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE src (id INT, p INT) USING parquet PARTITIONED BY (p)",
    );
    run(&s, "INSERT INTO src VALUES (1, 10), (2, 20)");
    let l = dir.path().join("src");
    run(
        &s,
        &format!(
            "CREATE TEMPORARY VIEW pv USING parquet OPTIONS (path '{}')",
            l.to_str().unwrap()
        ),
    );
    assert_eq!(rows(&s, "SELECT count(*) FROM pv"), [vec![some("2")]]);
    assert_eq!(
        rows(&s, "SELECT id, p FROM pv ORDER BY id"),
        [vec![some("1"), some("10")], vec![some("2"), some("20")]]
    );
}

#[test]
fn sql_19_a_table_without_columns_does_not_take_partitioning_or_bucketing_and_a_missing_location_cannot_infer()
 {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE src (id INT, p INT) USING parquet PARTITIONED BY (p)",
    );
    run(&s, "INSERT INTO src VALUES (1, 10)");
    let l = dir.path().join("src");
    let l = l.to_str().unwrap();
    let e = block_on(s.sql(&format!(
        "CREATE TABLE a USING parquet PARTITIONED BY (p) LOCATION '{l}'"
    )))
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::SpecifyPartitionIsNotAllowed);
    let e = block_on(s.sql(&format!(
        "CREATE TABLE b USING parquet CLUSTERED BY (id) INTO 2 BUCKETS LOCATION '{l}'"
    )))
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::SpecifyBucketingIsNotAllowed);
    let e = block_on(s.sql("CREATE TABLE c USING parquet LOCATION '/nonexistent/krishiv'"))
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToInferSchema);
}

#[test]
fn sql_19_hidden_and_underscore_directories_are_not_partition_columns() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE src (id INT, p INT) USING parquet PARTITIONED BY (p)",
    );
    run(&s, "INSERT INTO src VALUES (1, 10)");
    let l = dir.path().join("src");
    std::fs::create_dir_all(l.join(".hidden=1")).unwrap();
    std::fs::create_dir_all(l.join("_temporary=1")).unwrap();
    run(
        &s,
        &format!(
            "CREATE TABLE pt USING parquet LOCATION '{}'",
            l.to_str().unwrap()
        ),
    );
    run(&s, "MSCK REPAIR TABLE pt");
    assert_eq!(rows(&s, "SHOW PARTITIONS pt").len(), 1);
}

#[test]
fn rs_8_a_view_with_declared_columns_over_a_partitioned_directory_reads_every_row() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE src (id INT, p INT) USING parquet PARTITIONED BY (p)",
    );
    run(&s, "INSERT INTO src VALUES (1, 10), (2, 20)");
    let l = dir.path().join("src");
    run(
        &s,
        &format!(
            "CREATE TEMPORARY VIEW dv (id BIGINT, p INT) USING parquet OPTIONS (path '{}')",
            l.to_str().unwrap()
        ),
    );
    assert_eq!(
        rows(&s, "SELECT id, p FROM dv ORDER BY id"),
        [vec![some("1"), some("10")], vec![some("2"), some("20")]]
    );
    // the declared type is the view's type, not the files' (`INT` in the parquet files)
    assert_eq!(rows(&s, "DESCRIBE dv")[0][1], some("bigint"));
}
