// SPDX-License-Identifier: Apache-2.0
//! Partition discovery for reads by path (P2-T082b, design §2.5): the
//! `name=value` directories the partitioned writer makes read back as
//! columns, typed as Spark infers them.
#![expect(clippy::unwrap_used, reason = "test code")]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

/// Rows of `sql` with every column cast to a string by the query, sorted.
fn rows(s: &Session, sql: &str) -> Vec<Vec<String>> {
    let df = block_on(s.sql(sql)).unwrap();
    let mut out = Vec::new();
    for b in block_on(df.collect()).unwrap() {
        for r in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(r) {
                            "NULL".to_owned()
                        } else {
                            match c.data_type() {
                                DataType::Utf8View => c.as_string_view().value(r).to_owned(),
                                _ => c.as_string::<i32>().value(r).to_owned(),
                            }
                        }
                    })
                    .collect(),
            );
        }
    }
    out.sort();
    out
}

/// A table `t(id, year INT, region STRING)` written partitioned by
/// `(year, region)`; returns the root.
fn written(s: &Session, dir: &std::path::Path, years: &[i32], regions: &[Option<&str>]) -> String {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("year", DataType::Int32, false),
        Field::new("region", DataType::Utf8, true),
    ]));
    let ids: Vec<i32> = (1..).take(years.len()).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Int32Array::from(years.to_vec())),
            Arc::new(StringArray::from(regions.to_vec())),
        ],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    let root = dir.join("t");
    let root = root.to_str().unwrap().to_owned();
    block_on(
        s.table("t")
            .write()
            .partition_by(["year", "region"])
            .parquet(&root),
    )
    .unwrap();
    root
}

#[test]
fn partition_columns_read_back_typed_and_after_the_data_columns() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let root = written(
        &s,
        dir.path(),
        &[2020, 2020, 2021, 2021],
        &[Some("eu"), Some("us"), Some("eu"), None],
    );
    let schema = block_on(s.read().parquet([root.as_str()]).schema()).unwrap();
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().to_string()))
        .collect();
    assert_eq!(
        fields,
        [
            ("id".to_owned(), "Int32".to_owned()),
            ("year".to_owned(), "Int32".to_owned()),
            ("region".to_owned(), "Utf8View".to_owned()),
        ]
    );
    let all = rows(
        &s,
        &format!("SELECT CAST(id AS STRING), CAST(year AS STRING), region FROM parquet.`{root}`"),
    );
    assert_eq!(
        all,
        [
            ["1", "2020", "eu"],
            ["2", "2020", "us"],
            ["3", "2021", "eu"],
            ["4", "2021", "NULL"],
        ]
    );
}

#[test]
fn a_filter_on_a_partition_column_keeps_the_matching_rows() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let root = written(
        &s,
        dir.path(),
        &[2020, 2020, 2021, 2021],
        &[Some("eu"), Some("us"), Some("eu"), None],
    );
    let q = |pred: &str| {
        rows(
            &s,
            &format!("SELECT CAST(id AS STRING) FROM parquet.`{root}` WHERE {pred}"),
        )
    };
    assert_eq!(q("year = 2021"), [["3"], ["4"]]);
    assert_eq!(q("region = 'eu' AND id > 1"), [["3"]]);
    assert_eq!(q("region IS NULL"), [["4"]]);
    assert_eq!(q("year = 2020 OR region IS NULL"), [["1"], ["2"], ["4"]]);
    // Only partition columns are read.
    assert_eq!(
        rows(
            &s,
            &format!("SELECT CAST(count(*) AS STRING) FROM parquet.`{root}` WHERE year = 2020")
        ),
        [["2"]]
    );
    assert_eq!(
        rows(
            &s,
            &format!("SELECT DISTINCT CAST(year AS STRING) FROM parquet.`{root}`")
        ),
        [["2020"], ["2021"]]
    );
}

#[test]
fn files_at_different_depths_conflict() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let root = written(&s, dir.path(), &[2020, 2021], &[Some("eu"), Some("us")]);
    std::fs::copy(
        std::fs::read_dir(format!("{root}/year=2020/region=eu"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "parquet"))
            .unwrap(),
        format!("{root}/year=2020/loose.parquet"),
    )
    .unwrap();
    let e = block_on(s.read().parquet([root.as_str()]).schema()).unwrap_err();
    assert_eq!(e.error_class().name(), "CONFLICTING_DIRECTORY_STRUCTURES");
}

#[test]
fn a_directory_that_is_not_name_value_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let root = written(&s, dir.path(), &[2020], &[Some("eu")]);
    std::fs::create_dir(format!("{root}/plain")).unwrap();
    std::fs::copy(
        std::fs::read_dir(format!("{root}/year=2020/region=eu"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "parquet"))
            .unwrap(),
        format!("{root}/plain/f.parquet"),
    )
    .unwrap();
    let e = block_on(s.read().parquet([root.as_str()]).schema()).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn escaped_values_read_back_unescaped() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let root = written(
        &s,
        dir.path(),
        &[2020, 2021],
        &[Some("a/b=c%d"), Some("a%2Fb")],
    );
    assert!(std::path::Path::new(&format!("{root}/year=2020/region=a%2Fb%3Dc%25d")).is_dir());
    let all = rows(&s, &format!("SELECT region FROM parquet.`{root}`"));
    assert_eq!(all, [["a%2Fb"], ["a/b=c%d"]]);
}

#[test]
fn a_csv_read_gets_the_partition_columns_from_the_path() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let root = written(&s, dir.path(), &[2020, 2021], &[Some("eu"), Some("us")]);
    let csv = dir.path().join("c");
    let csv = csv.to_str().unwrap().to_owned();
    block_on(
        s.table("t")
            .write()
            .partition_by(["year"])
            .format("csv")
            .save(&csv),
    )
    .unwrap();
    let all = rows(
        &s,
        &format!("SELECT _c0, _c1, CAST(year AS STRING) FROM csv.`{csv}`"),
    );
    assert_eq!(all, [["1", "eu", "2020"], ["2", "us", "2021"]]);
    drop(root);
}

#[test]
fn recursive_file_lookup_reads_every_file_without_partition_columns() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let root = written(&s, dir.path(), &[2020, 2021], &[Some("eu"), Some("us")]);
    let df = s
        .read()
        .option("recursiveFileLookup", "true")
        .parquet([root.as_str()]);
    let names: Vec<String> = block_on(df.schema())
        .unwrap()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert!(!names.contains(&"year".to_owned()), "{names:?}");
    assert!(names.contains(&"id".to_owned()), "{names:?}");
}

#[test]
fn base_path_keeps_partition_columns_of_a_subdirectory() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let root = written(&s, dir.path(), &[2020, 2021], &[Some("eu"), Some("us")]);
    let sub = format!("{root}/year=2020");
    let df = s
        .read()
        .option("basePath", root.as_str())
        .parquet([sub.as_str()]);
    let names: Vec<String> = block_on(df.schema())
        .unwrap()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert!(names.contains(&"year".to_owned()), "{names:?}");
    assert!(names.contains(&"region".to_owned()), "{names:?}");
    // A basePath that is not a parent of the path is refused.
    let other = tempfile::tempdir().unwrap();
    let df = s
        .read()
        .option("basePath", other.path().to_str().unwrap())
        .parquet([sub.as_str()]);
    assert!(block_on(df.schema()).is_err());
}

#[test]
fn nested_and_repeated_projections_keep_the_column_layout() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    written(&s, dir.path(), &[2020, 2021], &[Some("eu"), Some("us")]);
    let root = dir.path().join("nested");
    let root = root.to_str().unwrap().to_owned();
    let df = block_on(s.sql("SELECT id, year, named_struct('a', id, 'b', id * 10) AS s FROM t"))
        .unwrap();
    block_on(df.write().partition_by(["year"]).parquet(&root)).unwrap();
    let q = |select: &str| rows(&s, &format!("SELECT {select} FROM parquet.`{root}`"));
    // Two paths of one struct, with the partition column between them.
    assert_eq!(
        q("CAST(s.a AS STRING), CAST(year AS STRING), CAST(s.b AS STRING)"),
        [["1", "2020", "10"], ["2", "2021", "20"]]
    );
    assert_eq!(
        q("CAST(id AS STRING), CAST(id AS STRING), CAST(year AS STRING), CAST(year AS STRING)"),
        [["1", "1", "2020", "2020"], ["2", "2", "2021", "2021"]]
    );
}

#[test]
fn a_glob_reads_each_match_as_its_own_root() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let root = written(
        &s,
        dir.path(),
        &[2020, 2020, 2021],
        &[Some("eu"), Some("us"), Some("eu")],
    );
    // A matched directory is a root of its own, so `year` is not a column.
    let ids = rows(
        &s,
        &format!("SELECT CAST(id AS STRING), region FROM parquet.`{root}/year=2020`"),
    );
    assert_eq!(ids, [["1", "eu"], ["2", "us"]]);
    let globbed = rows(
        &s,
        &format!("SELECT CAST(id AS STRING), region FROM parquet.`{root}/year=202{{0,1}}`"),
    );
    assert_eq!(globbed, [["1", "eu"], ["2", "us"], ["3", "eu"]]);
    let star = rows(
        &s,
        &format!("SELECT CAST(id AS STRING) FROM parquet.`{root}/year=2021/region=e?`"),
    );
    assert_eq!(star, [["3"]]);
    let e = block_on(s.sql(&format!("SELECT * FROM parquet.`{root}/year=1999*`")));
    let e = match e {
        Ok(df) => block_on(df.collect()).unwrap_err(),
        Err(e) => e,
    };
    assert_eq!(e.error_class().name(), "PATH_NOT_FOUND");
}
