// SPDX-License-Identifier: Apache-2.0
//! `DataFrame::count` is Spark's `groupBy().count()`: an aggregate over the
//! plan, so a scan reads no column for it (PY-013 follow-up).

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

/// With column pruning on, Spark 4.2.0 counts every non-empty, non-comment
/// line of a CSV without parsing it: a value over `maxCharsPerColumn` and a
/// record with too many or too few values are counted (5 here, from a Spark
/// 4.2.0 run of the same file). Parsing the lines would refuse the first.
#[test]
fn count_of_a_csv_parses_no_value_as_spark() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.csv");
    std::fs::write(
        &path,
        "a,b\n1,x\n\n#c\n2,yyyyyyyy\n3,\"q\"w\"\n4\n5,6,7\n   \n",
    )
    .unwrap();
    let path = path.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    let read = |mode: &str| {
        s.read()
            .format("csv")
            .option("header", "true")
            .option("comment", "#")
            .option("maxCharsPerColumn", "3")
            .option("mode", mode)
            .load([path.as_str()])
    };
    for mode in ["PERMISSIVE", "DROPMALFORMED", "FAILFAST"] {
        assert_eq!(block_on(read(mode).count()).unwrap(), 5, "{mode}");
    }
}

#[test]
fn count_of_an_empty_and_a_filtered_frame() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]))],
    )
    .unwrap();
    s.register_batches("t", Arc::clone(&schema), vec![batch])
        .unwrap();
    s.register_batches("e", schema, vec![]).unwrap();
    assert_eq!(block_on(s.table("t").count()).unwrap(), 3);
    assert_eq!(block_on(s.table("e").count()).unwrap(), 0);
    let some = s.table("t").filter_expr("v > 1").unwrap();
    assert_eq!(block_on(some.count()).unwrap(), 1);
    let grouped = block_on(s.sql("SELECT v FROM t GROUP BY v")).unwrap();
    assert_eq!(block_on(grouped.count()).unwrap(), 3);
}

/// Every file format reads with no column for `count()` (and a partitioned
/// directory read for its partition column only).
#[test]
fn every_file_format_counts_with_no_column() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int32, true),
        Field::new("p", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), None, Some(3), Some(4)])),
            Arc::new(Int32Array::from(vec![1, 1, 2, 2])),
        ],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    for format in ["parquet", "csv", "json", "avro", "text", "orc"] {
        let path = dir.path().join(format);
        let path = path.to_str().unwrap().to_owned();
        let frame = if format == "text" {
            s.table("t")
                .select_expr(["CAST(v AS STRING) AS v"])
                .unwrap()
        } else {
            s.table("t")
        };
        if let Err(e) = block_on(frame.write().format(format).save(&path)) {
            // The engine does not write ORC; nothing else may fail.
            assert_eq!(format, "orc", "{e:?}");
            continue;
        }
        let n = block_on(s.read().format(format).load([path.as_str()]).count());
        assert_eq!(n.unwrap(), 4, "{format}");
    }
    let parted = dir.path().join("parted");
    let parted = parted.to_str().unwrap().to_owned();
    block_on(
        s.table("t")
            .write()
            .partition_by(["p"])
            .format("parquet")
            .save(&parted),
    )
    .unwrap();
    let df = s.read().format("parquet").load([parted.as_str()]);
    assert_eq!(block_on(df.count()).unwrap(), 4);
    let only_p = df.filter_expr("p = 2").unwrap();
    assert_eq!(block_on(only_p.count()).unwrap(), 2);
    let sql = format!("SELECT count(*), count(p) FROM parquet.`{parted}`");
    let b = block_on(block_on(s.sql(&sql)).unwrap().collect()).unwrap();
    let col = |i: usize| {
        b[0].column(i)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap()
            .value(0)
    };
    assert_eq!((col(0), col(1)), (4, 4));
}

/// A JSON `count()` parses each record and converts no value: Spark 4.2.0
/// drops only the records that are not JSON objects under `DROPMALFORMED`
/// (7 of 9 here; the inferred schema converts every other record, so
/// `collect()` keeps 7 too) and counts every one under `PERMISSIVE`;
/// `FAILFAST` fails on the line that is not JSON. A `multiLine` file is its
/// first value (1 row). With a declared schema the count and the rows differ
/// (`krishiv-io` `json/count_tests.rs`).
#[test]
fn count_of_a_json_converts_no_value_as_spark() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.json");
    std::fs::write(
        &path,
        "{\"id\":1,\"name\":\"a\"}\n{\"id\":\"x\",\"name\":\"b\"}\nnot json\n\
         {\"id\":3,\"name\":4}\n{\"id\":4}\n[{\"id\":5},{\"id\":6}]\n\n7\n{\"id\":{\"z\":1}}\n",
    )
    .unwrap();
    let path = path.to_str().unwrap().to_owned();
    let s = Session::new().unwrap();
    let read = |mode: &str, multi: &str| {
        s.read()
            .format("json")
            .option("mode", mode)
            .option("multiLine", multi)
            .load([path.as_str()])
    };
    let rows = |df: krishiv_session::DataFrame| -> usize {
        block_on(df.collect())
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum()
    };
    assert_eq!(block_on(read("PERMISSIVE", "false").count()).unwrap(), 9);
    assert_eq!(block_on(read("DROPMALFORMED", "false").count()).unwrap(), 7);
    assert_eq!(rows(read("DROPMALFORMED", "false")), 7);
    assert!(block_on(read("FAILFAST", "false").count()).is_err());
    for mode in ["PERMISSIVE", "DROPMALFORMED", "FAILFAST"] {
        assert_eq!(block_on(read(mode, "true").count()).unwrap(), 1, "{mode}");
        assert_eq!(rows(read(mode, "true")), 1, "{mode}");
    }
}

/// A partitioned directory of gzip CSV and JSON files counts with no data
/// column: `count()`, `count(*)` with `count(p)`, and a count under a filter
/// on the partition column.
#[test]
fn partitioned_gzip_text_files_count_with_no_column() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int32, true),
        Field::new("p", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            // No NULL: a CSV row whose only data column is NULL is a blank
            // line, which no reader counts.
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 2])),
        ],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    for format in ["csv", "json"] {
        let path = dir.path().join(format);
        let path = path.to_str().unwrap().to_owned();
        block_on(
            s.table("t")
                .write()
                .partition_by(["p"])
                .format(format)
                .option("compression", "gzip")
                .save(&path),
        )
        .unwrap();
        let gz = walk(dir.path().join(format).as_path())
            .iter()
            .filter(|f| f.ends_with(".gz"))
            .count();
        assert!(gz >= 2, "{format}: {gz} gzip files");
        let df = s.read().format(format).load([path.as_str()]);
        assert_eq!(block_on(df.count()).unwrap(), 5, "{format}");
        let only_p = df.filter_expr("p = 2").unwrap();
        assert_eq!(block_on(only_p.count()).unwrap(), 3, "{format}");
        let sql = format!("SELECT count(*), count(p) FROM {format}.`{path}`");
        let b = block_on(block_on(s.sql(&sql)).unwrap().collect()).unwrap();
        let col = |i: usize| {
            b[0].column(i)
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .unwrap()
                .value(0)
        };
        assert_eq!((col(0), col(1)), (5, 5), "{format}");
    }
}

/// Every file name under `dir`.
fn walk(dir: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    // An unreadable entry is left out: the caller's assertion counts files.
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p.to_string_lossy().into_owned());
        }
    }
    out
}
