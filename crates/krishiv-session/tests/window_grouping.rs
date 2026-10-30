// SPDX-License-Identifier: Apache-2.0
//! Windows over a `ROLLUP` whose specification reads `grouping(..)` (the shape
//! of TPC-DS q70 and q86): `rank() OVER (PARTITION BY grouping(a) ...)`.
//! Expected rows and types from Spark 4.2.0 on the same data.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int32Type, Int64Type};
use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

/// `t(a, b, c)`: `('x','p',1), ('x','q',2), ('y','p',3)`.
fn session() -> Session {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, true),
        Field::new("b", DataType::Utf8, true),
        Field::new("c", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["x", "x", "y"])),
            Arc::new(StringArray::from(vec!["p", "q", "p"])),
            Arc::new(Int32Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    s
}

/// The rows as text, and each column's Arrow type.
fn run(s: &Session, sql: &str) -> (Vec<DataType>, Vec<Vec<String>>) {
    let df = block_on(s.sql(sql)).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let mut types = Vec::new();
    let mut rows = Vec::new();
    for b in &batches {
        types = b
            .schema()
            .fields()
            .iter()
            .map(|f| f.data_type().clone())
            .collect();
        for r in 0..b.num_rows() {
            let row = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(r) {
                        return "NULL".to_owned();
                    }
                    match c.data_type() {
                        DataType::Utf8 => c.as_string::<i32>().value(r).to_owned(),
                        DataType::Utf8View => c.as_string_view().value(r).to_owned(),
                        DataType::Int8 => c.as_primitive::<Int8Type>().value(r).to_string(),
                        DataType::Int32 => c.as_primitive::<Int32Type>().value(r).to_string(),
                        DataType::Int64 => c.as_primitive::<Int64Type>().value(r).to_string(),
                        other => panic!("unexpected type {other} in {sql}"),
                    }
                })
                .collect();
            rows.push(row);
        }
    }
    (types, rows)
}

fn rows(want: &[&[&str]]) -> Vec<Vec<String>> {
    want.iter()
        .map(|r| r.iter().map(|v| (*v).to_owned()).collect())
        .collect()
}

#[test]
fn a_window_partitioned_by_grouping() {
    let s = session();
    let (types, got) = run(
        &s,
        "SELECT a, sum(c) s, rank() OVER (PARTITION BY grouping(a) ORDER BY sum(c)) r \
         FROM t GROUP BY rollup(a) ORDER BY a",
    );
    // Spark: struct<a:string,s:bigint,r:int>.
    assert_eq!(types[1..], [DataType::Int64, DataType::Int32]);
    assert_eq!(
        got,
        rows(&[&["NULL", "6", "1"], &["x", "3", "1"], &["y", "3", "1"]])
    );
}

#[test]
fn q86_shape_partition_by_grouping_level_and_parent() {
    let s = session();
    let (types, got) = run(
        &s,
        "SELECT a, b, grouping(a)+grouping(b) l, \
         rank() OVER (PARTITION BY grouping(a)+grouping(b), \
                      CASE WHEN grouping(b) = 0 THEN a END ORDER BY sum(c) DESC) r \
         FROM t GROUP BY rollup(a, b) ORDER BY l DESC, a, b",
    );
    // Spark: struct<a:string,b:string,l:tinyint,r:int>.
    assert_eq!(types[2..], [DataType::Int8, DataType::Int32]);
    assert_eq!(
        got,
        rows(&[
            &["NULL", "NULL", "2", "1"],
            &["x", "NULL", "1", "1"],
            &["y", "NULL", "1", "1"],
            &["x", "p", "0", "2"],
            &["x", "q", "0", "1"],
            &["y", "p", "0", "1"],
        ])
    );
}

/// `ORDER BY grouping(..)` over a windowed aggregate (the sort of q70 and q86):
/// the id comes up through the windows, also for a key the select list drops.
#[test]
fn order_by_grouping_over_windows() {
    let s = session();
    let select = "SELECT a, rank() OVER (PARTITION BY grouping(a)+grouping(b) \
                  ORDER BY sum(c) DESC) r FROM t GROUP BY rollup(a,b)";
    for (order, want) in [
        (
            "grouping(a)+grouping(b) DESC, CASE WHEN grouping(a)+grouping(b)=0 THEN a END, r, a",
            rows(&[
                &["NULL", "1"],
                &["x", "1"],
                &["y", "1"],
                &["x", "2"],
                &["x", "3"],
                &["y", "1"],
            ]),
        ),
        (
            "grouping(b), a, r",
            rows(&[
                &["x", "2"],
                &["x", "3"],
                &["y", "1"],
                &["NULL", "1"],
                &["x", "1"],
                &["y", "1"],
            ]),
        ),
        (
            "grouping_id(), a, r",
            rows(&[
                &["x", "2"],
                &["x", "3"],
                &["y", "1"],
                &["x", "1"],
                &["y", "1"],
                &["NULL", "1"],
            ]),
        ),
        (
            "grouping_id(a,b), a, r",
            rows(&[
                &["x", "2"],
                &["x", "3"],
                &["y", "1"],
                &["x", "1"],
                &["y", "1"],
                &["NULL", "1"],
            ]),
        ),
    ] {
        let (types, got) = run(&s, &format!("{select} ORDER BY {order}"));
        assert_eq!(
            types.len(),
            2,
            "{order}: the id does not stay in the output"
        );
        assert_eq!(got, want, "{order}");
    }
}

/// What Spark 4.2.0 refuses in that position.
#[test]
fn order_by_grouping_over_windows_errors() {
    let s = session();
    let rollup = "SELECT a, rank() OVER (ORDER BY sum(c)) r FROM t GROUP BY rollup(a,b)";
    for (sql, class) in [
        (
            "SELECT a, rank() OVER (ORDER BY sum(c)) r FROM t GROUP BY a ORDER BY grouping(a)"
                .to_owned(),
            "UNSUPPORTED_GROUPING_EXPRESSION",
        ),
        (
            format!("{rollup} ORDER BY grouping(c)"),
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
        ),
        (
            format!("{rollup} ORDER BY grouping_id(b,a)"),
            "GROUPING_ID_COLUMN_MISMATCH",
        ),
        (
            format!("{rollup} ORDER BY sum(c)"),
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
        ),
    ] {
        let err = block_on(async { s.sql(&sql).await?.collect().await })
            .err()
            .unwrap_or_else(|| panic!("{sql} ran"));
        assert_eq!(err.error_class().name(), class, "{sql}: {err}");
    }
}
