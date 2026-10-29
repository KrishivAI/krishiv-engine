// SPDX-License-Identifier: Apache-2.0
//! Spark semantics that benchmark queries depend on, at small scale.
//!
//! Each case is the shape of a TPC-DS or ClickBench query whose result once
//! differed from the benchmark's DuckDB reference: the difference was the
//! reference's dialect, and these goldens pin Spark's answer. Every expected
//! value was produced by Spark 4.2.0 on the same rows.
//!
//! * TPC-DS q12: `ORDER BY` ascending puts NULL keys first (Spark's default
//!   null order; DuckDB's is NULLS LAST), and a ratio over
//!   `sum(sum(x)) OVER (PARTITION BY k)` groups the NULL key as one partition.
//! * ClickBench q29: in a Spark SQL string literal `'\1'` is the one-character
//!   string `1` and `'\.'` is `.`; the regex `.` does not match `\r` (Java's
//!   line terminators) unless `(?s)` is set; `$1` is the group reference.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Decimal128Type;
use arrow_array::{Array, Decimal128Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

/// `w(cls, item, price)`: two `arts` items (one sold twice), one `books` item
/// and two items whose class is NULL. `u(s)`: URLs, one with a `\r`.
fn session() -> Session {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("cls", ArrowType::Utf8, true),
        Field::new("item", ArrowType::Utf8, false),
        Field::new("price", ArrowType::Decimal128(7, 2), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                Some("arts"),
                Some("arts"),
                None,
                None,
                Some("books"),
                Some("arts"),
            ])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "a"])),
            Arc::new(
                Decimal128Array::from(vec![1000, 3000, 500, 1500, 2000, 250])
                    .with_precision_and_scale(7, 2)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    s.register_batches("w", schema, vec![batch]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("s", ArrowType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec![
            "https://www.example.com/a/b",
            "http://wwwxfoo.org/x",
            "plain",
            "http://nopath.com",
            "http://cr.net/a\rb",
        ]))],
    )
    .unwrap();
    s.register_batches("u", schema, vec![batch]).unwrap();
    s
}

/// Each row as text: strings as they are, decimals with their scale, NULL as
/// `NULL`.
fn rows(s: &Session, sql: &str) -> Vec<Vec<String>> {
    let df = block_on(s.sql(sql)).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let row = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(r) {
                        return "NULL".to_owned();
                    }
                    match c.data_type() {
                        ArrowType::Utf8 => c.as_string::<i32>().value(r).to_owned(),
                        ArrowType::LargeUtf8 => c.as_string::<i64>().value(r).to_owned(),
                        ArrowType::Utf8View => c.as_string_view().value(r).to_owned(),
                        ArrowType::Decimal128(..) => {
                            c.as_primitive::<Decimal128Type>().value_as_string(r)
                        }
                        other => panic!("unexpected column type {other} in {sql}"),
                    }
                })
                .collect();
            out.push(row);
        }
    }
    out
}

fn expect(s: &Session, sql: &str, want: &[&[&str]]) {
    let got = rows(s, sql);
    let want: Vec<Vec<String>> = want
        .iter()
        .map(|r| r.iter().map(|v| (*v).to_owned()).collect())
        .collect();
    assert_eq!(got, want, "{sql}");
}

#[test]
fn tpcds_q12_shape_null_partition_sorts_first() {
    let s = session();
    expect(
        &s,
        "SELECT item, cls, sum(price) AS rev, \
         sum(price)*100.0000/sum(sum(price)) OVER (PARTITION BY cls) AS ratio \
         FROM w GROUP BY item, cls ORDER BY cls, item, ratio LIMIT 4",
        &[
            &["c", "NULL", "5.00", "25.00000000000000000"],
            &["d", "NULL", "15.00", "75.00000000000000000"],
            &["a", "arts", "12.50", "29.41176470588235294"],
            &["b", "arts", "30.00", "70.58823529411764706"],
        ],
    );
}

#[test]
fn descending_order_puts_nulls_last() {
    let s = session();
    expect(
        &s,
        "SELECT cls FROM w GROUP BY cls ORDER BY cls DESC",
        &[&["books"], &["arts"], &["NULL"]],
    );
}

#[test]
fn clickbench_q29_shape_backslash_one_is_the_literal_one() {
    let s = session();
    expect(
        &s,
        r"SELECT s, regexp_replace(s, '^https?://(?:www\.)?([^/]+)/.*$', '\1') FROM u",
        &[
            &["https://www.example.com/a/b", "1"],
            &["http://wwwxfoo.org/x", "1"],
            &["plain", "plain"],
            &["http://nopath.com", "http://nopath.com"],
            &["http://cr.net/a\rb", "http://cr.net/a\rb"],
        ],
    );
}

#[test]
fn clickbench_q29_shape_dollar_group_with_dotall() {
    let s = session();
    expect(
        &s,
        r"SELECT s, regexp_replace(s, '(?s)^https?://(?:www\\.)?([^/]+)/.*$', '$1') FROM u",
        &[
            &["https://www.example.com/a/b", "example.com"],
            &["http://wwwxfoo.org/x", "wwwxfoo.org"],
            &["plain", "plain"],
            &["http://nopath.com", "http://nopath.com"],
            &["http://cr.net/a\rb", "cr.net"],
        ],
    );
}
