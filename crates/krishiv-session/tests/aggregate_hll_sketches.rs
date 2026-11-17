// SPDX-License-Identifier: Apache-2.0
//! `hll_sketch_agg`, `hll_union_agg`, `hll_sketch_estimate` and `hll_union` through SQL (P3-T086). Expected
//! values and error classes are Spark 4.2.0's `hll.sql` golden (`datasketchesAggregates.scala`): small inputs
//! count exactly, NULLs and empty strings and binaries are skipped, an empty group is the empty sketch, a bad
//! lgConfigK is `HLL_INVALID_LG_K`, a non-constant one `HLL_K_MUST_BE_CONSTANT`, mixed lgConfigK values
//! `HLL_UNION_DIFFERENT_LG_K` and a buffer that is no sketch `HLL_INVALID_INPUT_SKETCH_BUFFER`.
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
fn spark_golden_values() -> R {
    let s = Session::new()?;
    let one = |sql: &str| rows(&s, sql);
    assert_eq!(
        one(
            "SELECT hll_sketch_estimate(hll_sketch_agg(col)) FROM VALUES (0), (1), (2), (2), (2), (3), (4) AS tab(col)"
        )?,
        ["5"]
    );
    assert_eq!(
        one(
            "SELECT hll_sketch_estimate(hll_sketch_agg(col, 12)) FROM VALUES (50), (60), (60), (60), (75), (100) AS tab(col)"
        )?,
        ["4"]
    );
    assert_eq!(
        one(
            "SELECT hll_sketch_estimate(hll_sketch_agg(col)) FROM VALUES ('abc'), ('def'), ('abc'), ('ghi'), ('abc') AS tab(col)"
        )?,
        ["3"]
    );
    // Empty binary is skipped: X'' and the two-space binary differ, so 4 of the 5.
    assert_eq!(
        one(
            "SELECT hll_sketch_estimate(hll_sketch_agg(bytes)) FROM VALUES (X''), (CAST('  ' AS BINARY)), (X'e280'), (X'c1'), (X'c120') AS t(bytes)"
        )?,
        ["4"]
    );
    assert_eq!(
        one(
            "SELECT hll_sketch_estimate(hll_union(hll_sketch_agg(col1), hll_sketch_agg(col2))) FROM VALUES (1, 4), (1, 4), (2, 5), (2, 5), (3, 6) AS tab(col1, col2)"
        )?,
        ["6"]
    );
    assert_eq!(
        one(
            "SELECT hll_sketch_estimate(hll_union_agg(sketch, true)) FROM (SELECT hll_sketch_agg(col) AS sketch FROM VALUES (1) AS tab(col) UNION ALL SELECT hll_sketch_agg(col, 20) AS sketch FROM VALUES (1) AS tab(col))"
        )?,
        ["1"]
    );
    Ok(())
}

#[test]
fn groups_nulls_and_empty_inputs() -> R {
    let s = Session::new()?;
    assert_eq!(
        rows(
            &s,
            "SELECT k, hll_sketch_estimate(hll_sketch_agg(v)) FROM VALUES (1, 10), (1, 10), (1, 11), (2, NULL), (3, 5) AS t(k, v) GROUP BY k"
        )?,
        ["1|2", "2|0", "3|1"]
    );
    // No rows: the empty sketch, never NULL.
    assert_eq!(
        rows(
            &s,
            "SELECT hll_sketch_estimate(hll_sketch_agg(v)) FROM VALUES (1) AS t(v) WHERE v > 5"
        )?,
        ["0"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT hll_sketch_estimate(hll_union_agg(b)) FROM VALUES (CAST(NULL AS BINARY)) AS t(b)"
        )?,
        ["0"]
    );
    Ok(())
}

#[test]
fn the_goldens_errors() -> R {
    let s = Session::new()?;
    let c = |sql: &str| class(&s, sql);
    assert_eq!(
        c("SELECT hll_sketch_agg(col, 2) FROM VALUES (50), (60) AS tab(col)")?,
        "HLL_INVALID_LG_K"
    );
    assert_eq!(
        c("SELECT hll_sketch_agg(col, 40) FROM VALUES (50), (60) AS tab(col)")?,
        "HLL_INVALID_LG_K"
    );
    assert_eq!(
        c("SELECT hll_sketch_agg(col, CAST(NULL AS INT)) FROM VALUES (15) AS tab(col)")?,
        "HLL_INVALID_LG_K"
    );
    assert_eq!(
        c("SELECT hll_sketch_agg(col, CAST(col AS INT)) FROM VALUES (15), (16) AS tab(col)")?,
        "HLL_K_MUST_BE_CONSTANT"
    );
    assert_eq!(
        c(
            "SELECT hll_union(hll_sketch_agg(col1, 12), hll_sketch_agg(col2, 13)) FROM VALUES (1, 4) AS tab(col1, col2)"
        )?,
        "HLL_UNION_DIFFERENT_LG_K"
    );
    assert_eq!(
        c(
            "SELECT hll_union_agg(sketch, false) FROM (SELECT hll_sketch_agg(col, 12) AS sketch FROM VALUES (1) AS tab(col) UNION ALL SELECT hll_sketch_agg(col, 20) AS sketch FROM VALUES (1) AS tab(col))"
        )?,
        "HLL_UNION_DIFFERENT_LG_K"
    );
    assert_eq!(
        c("SELECT hll_sketch_estimate(CAST('abc' AS BINARY))")?,
        "HLL_INVALID_INPUT_SKETCH_BUFFER"
    );
    assert_eq!(
        c("SELECT hll_union(CAST('abc' AS BINARY), CAST('def' AS BINARY))")?,
        "HLL_INVALID_INPUT_SKETCH_BUFFER"
    );
    assert_eq!(
        c("SELECT hll_union_agg(buffer, false) FROM (SELECT CAST('abc' AS BINARY) AS buffer)")?,
        "HLL_INVALID_INPUT_SKETCH_BUFFER"
    );
    assert_eq!(
        c("SELECT hll_sketch_agg(col) FROM VALUES (ARRAY(1, 2)) AS tab(col)")?,
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
    assert_eq!(
        c("SELECT hll_sketch_agg(col, '15') FROM VALUES (50) AS tab(col)")?,
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
    assert_eq!(
        c("SELECT hll_union(1, 2)")?,
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
    Ok(())
}

#[test]
fn the_result_does_not_depend_on_partitions() -> R {
    let s = Session::new()?;
    let q = |n: u32| {
        format!(
            "SELECT hll_sketch_estimate(hll_sketch_agg(id % 997)) FROM (SELECT /*+ REPARTITION({n}) */ id FROM range(0, 20000))"
        )
    };
    let (a, b) = (rows(&s, &q(1))?, rows(&s, &q(8))?);
    assert_eq!(a, b);
    let est: i64 = a[0].parse()?;
    assert!((est - 997).abs() <= 997 / 50, "{est}");
    Ok(())
}
