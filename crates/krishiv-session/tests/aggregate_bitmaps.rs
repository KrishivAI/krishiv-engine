// SPDX-License-Identifier: Apache-2.0
//! `bitmap_construct_agg`, `bitmap_or_agg` and `bitmap_and_agg` through SQL (P3-T086). Expected values
//! follow Spark 4.2's `bitmapExpressions.scala` and the `function-matrix` golden for the three
//! aggregates: a bitmap is 4,096 bytes, an empty group is 4,096 zero bytes (`0xFF` for the AND), NULLs
//! are skipped, a position outside `0..32768` is `INVALID_BITMAP_POSITION`, and the result does not
//! depend on the number of partitions.
#![forbid(unsafe_code)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn cell(b: &RecordBatch, c: usize, row: usize) -> String {
    let a = b.column(c);
    if a.is_null(row) {
        return "NULL".into();
    }
    if let Some(s) = a.as_string_opt::<i32>() {
        return s.value(row).to_owned();
    }
    if let Some(s) = a.as_string_view_opt() {
        return s.value(row).to_owned();
    }
    if let Some(s) = a.as_primitive_opt::<Int64Type>() {
        return s.value(row).to_string();
    }
    if let Some(s) = a.as_primitive_opt::<arrow_array::types::Int32Type>() {
        return s.value(row).to_string();
    }
    format!("<{}>", a.data_type())
}

fn rows(s: &Session, sql: &str) -> R<Vec<String>> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let mut out = Vec::new();
        for b in &batches {
            for r in 0..b.num_rows() {
                out.push(
                    (0..b.num_columns())
                        .map(|c| cell(b, c, r))
                        .collect::<Vec<_>>()
                        .join("|"),
                );
            }
        }
        out.sort();
        Ok(out)
    })
}

fn class(s: &Session, sql: &str) -> R<String> {
    match rows(s, sql) {
        Ok(_) => Err(format!("{sql} did not fail").into()),
        Err(e) => match e.downcast::<KrishivError>() {
            Ok(e) => Ok(e.error_class().name().to_owned()),
            Err(_) => Err("not a KrishivError".into()),
        },
    }
}

#[test]
fn spark_documented_examples() -> R {
    let s = Session::new()?;
    let head = |f: &str, vals: &str| {
        format!("SELECT substring(hex({f}(col)), 0, 6) FROM VALUES {vals} AS tab(col)")
    };
    assert_eq!(
        rows(&s, &head("bitmap_or_agg", "(X '10'), (X '20'), (X '40')"))?,
        ["700000"]
    );
    assert_eq!(
        rows(&s, &head("bitmap_and_agg", "(X 'F0'), (X '70'), (X '30')"))?,
        ["300000"]
    );
    assert_eq!(
        rows(&s, &head("bitmap_and_agg", "(X 'FF'), (X 'FF'), (X 'FF')"))?,
        ["FF0000"]
    );
    let construct = |vals: &str| {
        format!(
            "SELECT substring(hex(bitmap_construct_agg(bitmap_bit_position(col))), 0, 6) \
             FROM VALUES {vals} AS tab(col)"
        )
    };
    assert_eq!(rows(&s, &construct("(1), (2), (3)"))?, ["070000"]);
    assert_eq!(rows(&s, &construct("(1), (1), (1)"))?, ["010000"]);
    Ok(())
}

#[test]
fn empty_and_all_null_groups_are_the_start_value() -> R {
    let s = Session::new()?;
    let q = "SELECT length(bitmap_construct_agg(bitmap_bit_position(c))), \
             hex(substring(bitmap_construct_agg(bitmap_bit_position(c)), 1, 2)), \
             hex(substring(bitmap_or_agg(b), 4095, 2)), hex(substring(bitmap_and_agg(b), 4095, 2)) \
             FROM VALUES (CAST(NULL AS BIGINT), CAST(NULL AS BINARY)) AS t(c, b)";
    assert_eq!(rows(&s, q)?, ["4096|0000|0000|FFFF"]);
    let empty = "SELECT length(bitmap_or_agg(b)), hex(substring(bitmap_and_agg(b), 1, 2)) \
                 FROM VALUES (X '00') AS t(b) WHERE false";
    assert_eq!(rows(&s, empty)?, ["4096|FFFF"]);
    Ok(())
}

#[test]
fn groups_nulls_and_short_inputs() -> R {
    let s = Session::new()?;
    let q = "SELECT g, bitmap_count(bitmap_construct_agg(p)), \
             hex(substring(bitmap_or_agg(b), 1, 3)), hex(substring(bitmap_and_agg(b), 1, 3)) FROM VALUES \
             (1, 0L, X '0F'), (1, 9L, X 'F0AA'), (1, CAST(NULL AS BIGINT), CAST(NULL AS BINARY)), \
             (2, 32767L, X 'FF00FF') AS t(g, p, b) GROUP BY g";
    assert_eq!(rows(&s, q)?, ["1|2|FFAA00|000000", "2|1|FF00FF|FF00FF"]);
    Ok(())
}

#[test]
fn one_and_eight_partitions_agree() -> R {
    let s = Session::new()?;
    let one = |slices: u32| {
        format!(
            "SELECT id % 5 AS g, bitmap_count(bitmap_construct_agg(bitmap_bit_position(id))), \
             hex(sha1(bitmap_construct_agg(bitmap_bit_position(id)))) \
             FROM range(0, 100000, 1, {slices}) GROUP BY id % 5"
        )
    };
    let a = rows(&s, &one(1))?;
    assert_eq!(a.len(), 5);
    assert_eq!(a, rows(&s, &one(8))?);
    // The OR and the AND of per-row bitmaps, built by a first aggregate in a subquery.
    let two = |slices: u32| {
        format!(
            "SELECT hex(sha1(bitmap_or_agg(m))), hex(sha1(bitmap_and_agg(m))) FROM (\
             SELECT id % 100 AS k, bitmap_construct_agg(bitmap_bit_position(id)) AS m \
             FROM range(0, 100000, 1, {slices}) GROUP BY id % 100)"
        )
    };
    let b = rows(&s, &two(1))?;
    assert_eq!(b.len(), 1);
    assert_eq!(b, rows(&s, &two(8))?);
    Ok(())
}

#[test]
fn a_position_outside_the_bitmap_is_refused() -> R {
    let s = Session::new()?;
    for v in ["-1", "32768"] {
        let q = format!("SELECT bitmap_construct_agg(p) FROM VALUES ({v}L) AS t(p)");
        assert_eq!(class(&s, &q)?, "INVALID_BITMAP_POSITION", "{v}");
    }
    Ok(())
}

#[test]
fn a_window_over_a_bitmap_aggregate_is_unsupported_and_a_string_is_refused() -> R {
    let s = Session::new()?;
    assert!(
        rows(
            &s,
            "SELECT bitmap_or_agg(b) OVER () FROM VALUES (X '01') AS t(b)"
        )
        .is_err()
    );
    assert_eq!(
        class(&s, "SELECT bitmap_or_agg(b) FROM VALUES ('a') AS t(b)")?,
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
    Ok(())
}
