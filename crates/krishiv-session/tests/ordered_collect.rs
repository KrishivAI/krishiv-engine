// SPDX-License-Identifier: Apache-2.0
//! Without `ORDER BY`, a query's rows come back partition by partition (a
//! `UNION ALL`'s children in order), as Spark returns them, however many
//! drivers read the partitions and however they race.
#![cfg(test)]

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_exec::{ExecutorConfig, TokioExecutor};
use krishiv_io::split::{SplitConfig, SplitFile, plan};
use krishiv_session::Session;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader as _, SerializedFileReader};

fn session(threads: usize) -> Session {
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: threads,
        io_threads: 1,
        blocking_threads: threads,
        io_blocking_threads: 2,
    })
    .unwrap();
    Session::builder().executor(executor).build().unwrap()
}

/// The first column's values in the order the query returned them.
fn collect_in_order(session: &Session, sql: &str) -> Vec<i64> {
    let df = block_on(session.sql(sql)).unwrap();
    let batches = block_on(df.collect()).unwrap();
    batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
        .collect()
}

/// Runs `sql` 25 times on each of four threads of one session.
fn repeat_under_load(session: &Session, sql: &str, want: &[i64]) {
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let session = session.clone();
            scope.spawn(move || {
                for _ in 0..25 {
                    assert_eq!(collect_in_order(&session, sql), want, "{sql}");
                }
            });
        }
    });
}

fn ids(range: std::ops::Range<i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from_iter_values(range))]).unwrap()
}

#[test]
fn union_of_ranges_returns_child_by_child() {
    let want: Vec<i64> = (0..128).collect();
    for threads in [2, 8] {
        let s = session(threads);
        repeat_under_load(
            &s,
            "SELECT * FROM range(0, 64, 1, 8) UNION ALL SELECT * FROM range(64, 128, 1, 8)",
            &want,
        );
    }
}

/// A memory table of five morsels (128K rows each) read by several
/// drivers, through a filter and a projection.
#[test]
fn a_large_memory_table_returns_morsel_by_morsel() {
    let rows: i64 = 600_000;
    let batches: Vec<RecordBatch> = (0..rows)
        .step_by(8192)
        .map(|start| ids(start..(start + 8192).min(rows)))
        .collect();
    let want: Vec<i64> = (0..rows).filter(|i| i % 7 != 0).map(|i| i + 1).collect();
    for threads in [2, 8] {
        let s = session(threads);
        s.register_batches("t", batches[0].schema(), batches.clone())
            .unwrap();
        repeat_under_load(&s, "SELECT id + 1 FROM t WHERE id % 7 != 0", &want);
    }
}

/// Several Parquet files, each its own split: every run returns the rows
/// in the order a single driver reads the splits.
#[test]
fn a_multi_split_parquet_scan_returns_split_by_split() {
    let dir = tempfile::tempdir().unwrap();
    for f in 0..8_i64 {
        let path = dir.path().join(format!("part-{f}.parquet"));
        let file = std::fs::File::create(path).unwrap();
        let batch = ids(f * 10_000..(f + 1) * 10_000);
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2_500))
            .build();
        let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }
    let path = dir.path().to_str().unwrap().to_owned();
    let sql = format!("SELECT id FROM parquet.`{path}` WHERE id % 3 != 1");
    // The files hold 0..10_000, 10_000..20_000, ... and are listed by name:
    // the rows come back ascending, split by split.
    let want: Vec<i64> = (0..80_000).filter(|i| i % 3 != 1).collect();
    assert_eq!(collect_in_order(&session(1), &sql), want);
    for threads in [2, 8] {
        let s = session(threads);
        repeat_under_load(&s, &sql, &want);
    }
}

/// An empty file among the splits (a partition with no rows) does not
/// hold the order back.
#[test]
fn an_empty_split_keeps_the_order() {
    let dir = tempfile::tempdir().unwrap();
    for f in 0..6_i64 {
        let path = dir.path().join(format!("part-{f}.parquet"));
        let file = std::fs::File::create(path).unwrap();
        let batch = if f == 2 {
            ids(0..0)
        } else {
            ids(f * 5_000..(f + 1) * 5_000)
        };
        let mut w = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }
    let path = dir.path().to_str().unwrap().to_owned();
    let sql = format!("SELECT id FROM parquet.`{path}`");
    // Files 0, 1, 3, 4, 5 hold 5_000 rows each; file 2 is empty.
    let want: Vec<i64> = (0..6_i64)
        .filter(|f| *f != 2)
        .flat_map(|f| f * 5_000..(f + 1) * 5_000)
        .collect();
    assert_eq!(want.len(), 25_000);
    assert_eq!(collect_in_order(&session(1), &sql), want);
    repeat_under_load(&session(8), &sql, &want);
}

/// A `UNION ALL` whose middle branch joins (an operator that flushes after
/// its input, so the branches run concurrently): the ordered branches
/// still come back first child first, each in slice order; the join's
/// rows may land anywhere.
#[test]
fn a_concurrent_union_keeps_its_ordered_children_in_order() {
    let sql = "SELECT id FROM range(0, 64, 1, 8) \
               UNION ALL SELECT a.id FROM range(1000, 1100) a JOIN range(1000, 1100) b ON a.id = b.id \
               UNION ALL SELECT id FROM range(64, 128, 1, 8)";
    let s = session(8);
    for _ in 0..40 {
        let got = collect_in_order(&s, sql);
        assert_eq!(got.len(), 228);
        let ranges: Vec<i64> = got.into_iter().filter(|&v| v < 1000).collect();
        assert_eq!(ranges, (0..128).collect::<Vec<i64>>());
    }
}

/// An error in the first partition fails the query while later drivers
/// hold batches back, and dropping the stream mid-way ends the query: no
/// driver waits for a turn that never comes.
#[test]
fn errors_and_a_dropped_stream_end_an_ordered_query() {
    use futures::StreamExt as _;
    let s = session(8);
    let df = block_on(s.sql(
        "SELECT CASE WHEN id = 0 THEN raise_error('boom') ELSE CAST(id AS STRING) END \
         FROM range(0, 400000, 1, 8)",
    ))
    .unwrap();
    let e = block_on(df.collect()).unwrap_err();
    assert!(e.to_string().contains("boom"), "{e}");
    for _ in 0..10 {
        let df = block_on(s.sql("SELECT id FROM range(0, 4000000, 1, 8)")).unwrap();
        let mut stream = block_on(df.execute_stream()).unwrap();
        let first = block_on(stream.next()).unwrap().unwrap();
        assert_eq!(first.column(0).as_primitive::<Int64Type>().value(0), 0);
        drop(stream);
    }
    // The session still answers.
    assert_eq!(collect_in_order(&s, "SELECT id FROM range(3)"), [0, 1, 2]);
}

/// The `EXPLAIN ANALYZE` text of `sql`.
fn analyzed(session: &Session, sql: &str) -> String {
    let df = block_on(session.sql(&format!("EXPLAIN ANALYZE {sql}"))).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let col = batches[0].column(0);
    match col.data_type() {
        DataType::Utf8View => col.as_string_view().value(0).to_owned(),
        _ => col.as_string::<i32>().value(0).to_owned(),
    }
}

/// One Parquet file of 32 row groups cut into finer morsels than its
/// splits (a small `openCostInBytes` lowers the morsel floor): every run
/// returns the rows in file order, also for a filter that keeps one
/// stretch of the file.
#[test]
fn finer_morsels_keep_the_file_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sorted.parquet");
    let file = std::fs::File::create(&path).unwrap();
    let batch = ids(0..80_000);
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2_500))
        .build();
    let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    let path = path.to_str().unwrap().to_owned();
    let all = format!("SELECT id FROM parquet.`{path}` WHERE id % 3 != 1");
    let stretch = format!("SELECT id FROM parquet.`{path}` WHERE id >= 50000 AND id < 61000");
    let want_all: Vec<i64> = (0..80_000).filter(|i| i % 3 != 1).collect();
    let want_stretch: Vec<i64> = (50_000..61_000).collect();
    for threads in [2, 8] {
        let s = session(threads);
        s.conf()
            .set("spark.sql.files.openCostInBytes", "1024")
            .unwrap();
        // Four morsels per driver, each a piece of one of the `threads`
        // planned splits.
        let text = analyzed(&s, &all);
        let pieces: usize = text
            .split("scan_splits=")
            .nth(1)
            .and_then(|t| t.split(' ').next())
            .and_then(|n| n.parse().ok())
            .unwrap();
        assert!(pieces >= 4 * threads, "{text}");
        repeat_under_load(&s, &all, &want_all);
        repeat_under_load(&s, &stretch, &want_stretch);
    }
}

/// Writes `groups` row groups of 2,500 rows, ids from `first`, and returns
/// the row groups' `(ids, byte range)` as the reader plans them.
fn write_groups(path: &Path, first: i64, groups: i64) -> Vec<(Range<i64>, Range<u64>)> {
    let batch = ids(first..first + groups * 2_500);
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2_500))
        .build();
    let file = std::fs::File::create(path).unwrap();
    let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    let meta = SerializedFileReader::new(std::fs::File::open(path).unwrap())
        .unwrap()
        .metadata()
        .clone();
    let mut start = first;
    meta.row_groups()
        .iter()
        .map(|rg| {
            let (at, len) = rg.column(0).byte_range();
            let rows = start..start + rg.num_rows();
            start = rows.end;
            (rows, at..at + len)
        })
        .collect()
}

/// Files of different sizes, cut into finer morsels than their splits:
/// the rows come back in the order of the splits Spark's planner gives
/// (the longest split first, whatever the file), however the drivers race.
#[test]
fn finer_morsels_keep_the_planned_split_order() {
    let dir = tempfile::tempdir().unwrap();
    // Neither the names nor the ids follow the sizes.
    let shapes = [
        ("a.parquet", 0, 8),
        ("b.parquet", 20_000, 40),
        ("c.parquet", 120_000, 20),
    ];
    let mut groups = Vec::new();
    let mut files = Vec::new();
    for (name, first, count) in shapes {
        let path = dir.path().join(name);
        groups.push(write_groups(&path, first, count));
        files.push(SplitFile {
            size: std::fs::metadata(&path).unwrap().len(),
            splittable: true,
        });
    }
    let sql = format!("SELECT id FROM parquet.`{}`", dir.path().display());
    for threads in [2_usize, 8] {
        let s = session(threads);
        s.conf()
            .set("spark.sql.files.openCostInBytes", "1024")
            .unwrap();
        let planned = plan(
            &SplitConfig {
                max_partition_bytes: 128 << 20,
                open_cost_in_bytes: 1024,
                min_partition_num: threads as u64,
                max_partition_num: None,
            },
            &files,
        );
        let splits: Vec<_> = planned.iter().flat_map(|p| &p.splits).collect();
        let want: Vec<i64> = splits
            .iter()
            .flat_map(|split| {
                groups[split.file]
                    .iter()
                    .filter(|(_, bytes)| split.range.owns(bytes))
                    .flat_map(|(rows, _)| rows.clone())
            })
            .collect();
        assert_eq!(want.len(), 170_000);
        assert!(!want.is_sorted(), "the plan must not follow the ids");
        // The scan really read more pieces than the plan has splits.
        let text = analyzed(&s, &sql);
        let pieces: usize = text
            .split("scan_splits=")
            .nth(1)
            .and_then(|t| t.split(' ').next())
            .and_then(|n| n.parse().ok())
            .unwrap();
        assert!(pieces > splits.len(), "{pieces} pieces: {text}");
        repeat_under_load(&s, &sql, &want);
    }
}

/// Under a tight limit the drivers ahead stop at their shares and the rows
/// still come back in order; finer morsels only move a driver on sooner.
#[test]
fn finer_morsels_keep_the_order_under_a_tight_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sorted.parquet");
    write_groups(&path, 0, 400);
    let sql = format!("SELECT id FROM parquet.`{}`", path.display());
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: 8,
        io_threads: 1,
        blocking_threads: 8,
        io_blocking_threads: 2,
    })
    .unwrap();
    let s = Session::builder()
        .executor(executor)
        .config("krishiv.memory.limit", "32m")
        .build()
        .unwrap();
    s.conf()
        .set("spark.sql.files.openCostInBytes", "1024")
        .unwrap();
    let want: Vec<i64> = (0..1_000_000).collect();
    for _ in 0..5 {
        assert_eq!(collect_in_order(&s, &sql), want);
    }
}

/// A predicate that keeps half the rows, over a file cut into about one
/// morsel per row group: each driver guesses the row filter on once, not
/// once per morsel, and every matching row comes back in file order.
#[test]
fn drivers_keep_the_row_filter_measurement_across_morsels() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("half.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Utf8, false),
    ]));
    let rows = 0..80_000_i64;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(rows.clone())),
            Arc::new(Int64Array::from_iter_values(rows.clone().map(|i| i % 2))),
            Arc::new(StringArray::from_iter_values(
                rows.map(|i| format!("a value long enough to outweigh the key {i}")),
            )),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2_500))
        .build();
    let file = std::fs::File::create(&path).unwrap();
    let mut w = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    let sql = format!("SELECT id, v FROM parquet.`{}` WHERE k = 0", path.display());
    let threads = 4;
    let s = session(threads);
    s.conf()
        .set("spark.sql.files.openCostInBytes", "1024")
        .unwrap();
    let want: Vec<i64> = (0..80_000).step_by(2).collect();
    assert_eq!(collect_in_order(&s, &sql), want);
    let text = analyzed(&s, &sql);
    let counter = |name: &str| -> usize {
        text.split(&format!("{name}="))
            .nth(1)
            .and_then(|t| t.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("no {name}: {text}"))
    };
    assert!(counter("scan_splits") >= 4 * threads, "{text}");
    let on = counter("row_filter_units");
    assert!((1..=threads).contains(&on), "{on} guesses: {text}");
}

/// The `scan_splits` counter of `sql`'s `EXPLAIN ANALYZE`.
fn scan_splits(session: &Session, sql: &str) -> usize {
    let text = analyzed(session, sql);
    text.split("scan_splits=")
        .nth(1)
        .and_then(|t| t.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no scan_splits: {text}"))
}

/// A text file cut into morsels mid-line: every line is read once, by the
/// piece holding its start, in file order.
#[test]
fn finer_morsels_of_a_text_file_keep_its_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lines.csv");
    let lines: String = (0..20_000)
        .map(|i| format!("{i},row number {i}\n"))
        .collect();
    std::fs::write(&path, lines).unwrap();
    let sql = format!(
        "SELECT CAST(_c0 AS BIGINT) AS id FROM csv.`{}`",
        path.display()
    );
    let want: Vec<i64> = (0..20_000).collect();
    for threads in [2, 8] {
        let s = session(threads);
        s.conf()
            .set("spark.sql.files.openCostInBytes", "1024")
            .unwrap();
        assert!(scan_splits(&s, &sql) >= 4 * threads);
        repeat_under_load(&s, &sql, &want);
    }
}
