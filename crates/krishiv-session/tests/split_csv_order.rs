// SPDX-License-Identifier: Apache-2.0
//! The rows `first`/`head`/`limit` return from a CSV file that is read in several splits (PY-067), against Spark
//! 4.2.0 (`py067.*` probes of `tests/fixtures/spark-4.2.0/csv-json-corrupt-record/goldens.json`; the file is
//! 100000 rows of 80 bytes, `spark.sql.files.maxPartitionBytes` 1 MiB). Spark promises no order without `ORDER BY`,
//! but a single file reads in file order and `LIMIT` takes the first rows; a `LIMIT` over a file scan runs on one
//! driver so that the first split is the one read.
#![cfg(test)]

use std::fmt::Write as _;

use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

fn goldens() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/csv-json-corrupt-record/goldens.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Row `i` of the generator: `"<prefix><i>,name<i>,"` padded with `x` to 79 bytes, then a newline.
fn write_rows(path: &std::path::Path, prefix: &str, rows: usize) {
    let mut out = String::with_capacity(rows * 80);
    for i in 0..rows {
        let mut line = format!("{prefix}{i},name{i},");
        while line.len() < 79 {
            line.push('x');
        }
        writeln!(out, "{line}").unwrap();
    }
    std::fs::write(path, out).unwrap();
}

fn render(batches: &[RecordBatch]) -> Vec<String> {
    let mut rows = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(i) {
                        "null".to_owned()
                    } else {
                        c.as_string_view_opt().map_or_else(
                            || c.as_string::<i32>().value(i).to_owned(),
                            |v| v.value(i).to_owned(),
                        )
                    }
                })
                .collect();
            rows.push(format!("[{}]", cells.join(",")));
        }
    }
    rows
}

fn session() -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.files.maxPartitionBytes", "1048576")
        .unwrap();
    s
}

fn query(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    render(&block_on(df.collect()).unwrap())
}

#[test]
fn limit_over_a_split_csv_file_returns_the_first_rows_every_time() {
    let g = goldens();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.csv");
    write_rows(
        &path,
        "",
        usize::try_from(g["big_file"]["rows"].as_u64().unwrap()).unwrap(),
    );
    let s = session();
    let from = format!("csv.`{}`", path.display());
    let head3: Vec<String> = g["probes"]["py067.head3.run1"]["value"]
        .as_str()
        .unwrap()
        .split(" | ")
        .map(str::to_owned)
        .collect();
    let want_first = &g["probes"]["py067.first.run1"]["value"];
    for run in 0..10 {
        let got = query(&s, &format!("SELECT * FROM {from} LIMIT 3"));
        assert_eq!(got, head3, "LIMIT 3, run {run}");
        let got = query(&s, &format!("SELECT * FROM {from} LIMIT 1"));
        assert_eq!(&got[0], want_first.as_str().unwrap(), "first, run {run}");
        let got = query(&s, &format!("SELECT * FROM {from} LIMIT 1000"));
        assert_eq!(got.len(), 1000);
        let ids: Vec<String> = (0..1000).map(|i| format!("[{i},name{i},")).collect();
        assert!(
            got.iter().zip(&ids).all(|(r, id)| r.starts_with(id)),
            "LIMIT 1000, run {run}: rows 0..999 in order, got {:?}..{:?}",
            &got[0][..12],
            &got[999][..16]
        );
    }
    // The whole file is still in file order.
    let all = query(&s, &format!("SELECT _c0 FROM {from}"));
    assert_eq!(all.len(), 100_000);
    assert!(
        all.iter().enumerate().all(|(i, r)| *r == format!("[{i}]")),
        "a full read is in file order"
    );
}

/// Several files: Spark orders the splits by length, longest first, ties by its listing order, and `first()` is
/// the first row of the first split (`b0`: the 1 MiB split of `f2.csv`, listed before `f1.csv`). The engine's
/// own listing order decides the ties, so this is not asserted; it must be one file's first row, every time.
#[test]
fn limit_over_two_split_csv_files_returns_the_first_row_of_a_split_start() {
    let g = goldens();
    let dir = tempfile::tempdir().unwrap();
    let rows = g["big_file"]["two_file_rows"].as_array().unwrap();
    let prefixes = g["big_file"]["two_file_prefixes"].as_array().unwrap();
    for (k, name) in ["f1.csv", "f2.csv"].iter().enumerate() {
        write_rows(
            &dir.path().join(name),
            prefixes[k].as_str().unwrap(),
            usize::try_from(rows[k].as_u64().unwrap()).unwrap(),
        );
    }
    let s = session();
    let from = format!("csv.`{}`", dir.path().display());
    let first = query(&s, &format!("SELECT * FROM {from} LIMIT 1"));
    let want = g["probes"]["py067.two_files_first"]["value"]
        .as_str()
        .unwrap();
    let a0 = want.replacen('b', "a", 1);
    assert!(
        first[0] == want || first[0] == a0,
        "the first row is the start of f1 or f2, got {}",
        &first[0][..20]
    );
    for _ in 0..10 {
        assert_eq!(query(&s, &format!("SELECT * FROM {from} LIMIT 1")), first);
    }
}
