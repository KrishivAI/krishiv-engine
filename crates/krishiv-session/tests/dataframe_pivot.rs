// SPDX-License-Identifier: Apache-2.0
//! `GroupedData::pivot` builds the plan SQL's `PIVOT` builds: the same columns, names and cells.
#![forbid(unsafe_code)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;
use krishiv_session::functions::{col, count, lit, max, sum};

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// `(year, course, earnings)`: 2012 has two courses, 2013 one.
fn session() -> R<Session> {
    let s = Session::new()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("year", DataType::Int32, true),
        Field::new("course", DataType::Utf8, true),
        Field::new("earnings", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![2012, 2012, 2012, 2013])),
            Arc::new(StringArray::from(vec!["Java", "dotNET", "dotNET", "Java"])),
            Arc::new(Int32Array::from(vec![20, 10, 5, 30])),
        ],
    )?;
    s.register_batches("t", schema, vec![batch])?;
    Ok(s)
}

/// The column names and `year|cells` rows of a frame, sorted.
fn shown(batches: &[RecordBatch]) -> (Vec<String>, Vec<String>) {
    let names = batches
        .first()
        .map(|b| {
            b.schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        })
        .unwrap_or_default();
    let mut rows = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(r) {
                        "NULL".to_owned()
                    } else if let DataType::Int32 = c.data_type() {
                        c.as_primitive::<arrow_array::types::Int32Type>()
                            .value(r)
                            .to_string()
                    } else {
                        c.as_primitive::<Int64Type>().value(r).to_string()
                    }
                })
                .collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    (names, rows)
}

#[test]
fn pivot_matches_the_sql_clause() -> R {
    let s = session()?;
    let df = s
        .table("t")
        .group_by(["year"])
        .pivot(col("course"), [lit("Java"), lit("dotNET")])
        .agg([sum(col("earnings"))]);
    let got = shown(&block_on(df.collect())?);
    let sql = block_on(s.sql(
        "SELECT * FROM (SELECT year, course, earnings FROM t) \
         PIVOT (sum(earnings) FOR course IN ('Java', 'dotNET'))",
    ))?;
    let want = shown(&block_on(sql.collect())?);
    assert_eq!(got, want);
    assert_eq!(got.0, ["year", "Java", "dotNET"]);
    assert_eq!(got.1, ["2012|20|15", "2013|30|NULL"]);
    Ok(())
}

#[test]
fn an_aggregate_may_read_the_pivot_column_and_names_use_the_aggregate() -> R {
    let s = session()?;
    let df = s
        .table("t")
        .group_by(["year"])
        .pivot(col("course"), [lit("Java")])
        .agg([count(col("course")), max(col("earnings"))]);
    let got = shown(&block_on(df.collect())?);
    assert_eq!(
        got.0,
        ["year", "Java_count(course)", "Java_max(earnings)"]
    );
    assert_eq!(got.1, ["2012|1|20", "2013|1|30"]);
    Ok(())
}

#[test]
fn an_aggregate_reading_a_grouping_column_is_refused() -> R {
    let s = session()?;
    let df = s
        .table("t")
        .group_by(["year"])
        .pivot(col("course"), [lit("Java")])
        .agg([max(col("year"))]);
    assert!(block_on(df.collect()).is_err());
    Ok(())
}
