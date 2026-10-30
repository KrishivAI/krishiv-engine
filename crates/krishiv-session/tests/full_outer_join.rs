// SPDX-License-Identifier: Apache-2.0
//! `FULL OUTER JOIN` through the whole pipeline (the join of TPC-DS q51 and
//! q97). Expected rows from Spark 4.2.0 on the same data.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;

/// `l(k, v)` and `r(k, w)`, each with a NULL key and duplicates of key 2.
fn session() -> Session {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE TEMPORARY VIEW l AS SELECT * FROM VALUES \
         (1,'a'),(2,'b'),(2,'c'),(NULL,'n') AS l(k,v)",
        "CREATE TEMPORARY VIEW r AS SELECT * FROM VALUES \
         (2,'x'),(2,'y'),(3,'z'),(NULL,'m') AS r(k,w)",
        "CREATE TEMPORARY VIEW e AS SELECT * FROM r WHERE false",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    s
}

fn text(b: &RecordBatch, c: usize, r: usize) -> String {
    let a = b.column(c);
    if a.is_null(r) {
        return "-".to_owned();
    }
    match a.data_type() {
        DataType::Int32 => a.as_primitive::<Int32Type>().value(r).to_string(),
        DataType::Int64 => a
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(r)
            .to_string(),
        DataType::Utf8 => a.as_string::<i32>().value(r).to_owned(),
        DataType::Utf8View => a.as_string_view().value(r).to_owned(),
        other => panic!("unexpected type {other}"),
    }
}

/// The rows, each as its values joined by spaces; NULL is `-`.
fn run(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let row: Vec<String> = (0..b.num_columns()).map(|c| text(b, c, r)).collect();
            out.push(row.join(" "));
        }
    }
    out
}

#[test]
fn full_joins_match_spark() {
    let s = session();
    let both = "SELECT l.k, v, r.k, w FROM l FULL JOIN r";
    let padded_pairs = ["2 b 2 x", "2 b 2 y", "2 c 2 x", "2 c 2 y"];
    let cases: Vec<(String, Vec<&str>)> = vec![
        (
            format!("{both} ON l.k = r.k ORDER BY v, w"),
            [
                &["- - - m", "- - 3 z", "1 a - -"],
                &padded_pairs[..],
                &["- n - -"],
            ]
            .concat(),
        ),
        (
            format!("{both} ON l.k = r.k AND v <> 'b' ORDER BY v, w"),
            vec![
                "- - - m", "- - 3 z", "1 a - -", "2 b - -", "2 c 2 x", "2 c 2 y", "- n - -",
            ],
        ),
        (
            format!("{both} ON l.k <=> r.k ORDER BY v, w"),
            [&["- - 3 z", "1 a - -"], &padded_pairs[..], &["- n - m"]].concat(),
        ),
        (
            "SELECT l.k, v, e.k, w FROM l FULL JOIN e ON l.k = e.k ORDER BY v, w".to_owned(),
            vec!["1 a - -", "2 b - -", "2 c - -", "- n - -"],
        ),
        (
            "SELECT e.k, r.k, r.w FROM e FULL JOIN r ON e.k = r.k ORDER BY r.w".to_owned(),
            vec!["- - m", "- 2 x", "- 2 y", "- 3 z"],
        ),
        (
            "SELECT k, v, w FROM l FULL JOIN r USING (k) ORDER BY v, w".to_owned(),
            vec![
                "- - m", "3 - z", "1 a -", "2 b x", "2 b y", "2 c x", "2 c y", "- n -",
            ],
        ),
        (
            "SELECT * FROM l NATURAL FULL JOIN r ORDER BY v, w".to_owned(),
            vec![
                "- - m", "3 - z", "1 a -", "2 b x", "2 b y", "2 c x", "2 c y", "- n -",
            ],
        ),
        (
            "SELECT coalesce(l.k, r.k) c, count(*), count(v), count(w) \
             FROM l FULL JOIN r ON l.k = r.k GROUP BY 1 ORDER BY 1"
                .to_owned(),
            vec!["- 2 1 1", "1 1 1 0", "2 4 4 4", "3 1 0 1"],
        ),
        (
            format!("{both} ON l.k = r.k WHERE l.k IS NOT NULL ORDER BY v, w"),
            [&["1 a - -"], &padded_pairs[..]].concat(),
        ),
        (
            format!("{both} ON l.k = r.k WHERE r.k IS NOT NULL ORDER BY v, w"),
            [&["- - 3 z"], &padded_pairs[..]].concat(),
        ),
        (
            format!("{both} ON l.k = r.k WHERE r.k = 2 AND l.k = 2 ORDER BY v, w"),
            padded_pairs.to_vec(),
        ),
        (
            format!("{both} ON l.k = r.k ORDER BY v NULLS LAST, w NULLS LAST LIMIT 2"),
            vec!["1 a - -", "2 b 2 x"],
        ),
        // Spark ignores the hint: a side the join preserves is not broadcast.
        (
            "SELECT /*+ BROADCAST(r) */ l.k, v, r.k, w FROM l FULL JOIN r ON l.k = r.k \
             ORDER BY v, w"
                .to_owned(),
            [
                &["- - - m", "- - 3 z", "1 a - -"],
                &padded_pairs[..],
                &["- n - -"],
            ]
            .concat(),
        ),
    ];
    for (sql, want) in cases {
        assert_eq!(run(&s, &sql), want, "{sql}");
    }
}

/// Every output column can be NULL.
#[test]
fn every_column_of_a_full_join_is_nullable() {
    let s = session();
    let df =
        block_on(s.sql("SELECT * FROM (SELECT 1 a) x FULL JOIN (SELECT 1 b) y ON a = b")).unwrap();
    let batches = block_on(df.collect()).unwrap();
    assert!(
        batches[0].schema().fields().iter().all(|f| f.is_nullable()),
        "{:?}",
        batches[0].schema()
    );
}
