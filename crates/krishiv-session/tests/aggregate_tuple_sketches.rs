// SPDX-License-Identifier: Apache-2.0
//! `tuple_sketch_agg_*`, `tuple_union_agg_*` and `tuple_intersection_agg_*` through SQL (P3 goldens, sketches unit 3):
//! estimates and summaries do not depend on the partition count, empty groups give the empty sketch, and bad constants
//! are Spark's analysis errors.
#![forbid(unsafe_code)]

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn cell(b: &RecordBatch, c: usize, row: usize) -> String {
    let a = b.column(c);
    if a.is_null(row) {
        return "NULL".into();
    }
    if let Some(s) = a.as_primitive_opt::<Int64Type>() {
        return s.value(row).to_string();
    }
    if let Some(s) = a.as_primitive_opt::<Float64Type>() {
        return s.value(row).to_string();
    }
    if let Some(s) = a.as_string_opt::<i32>() {
        return s.value(row).to_owned();
    }
    if let Some(s) = a.as_string_view_opt() {
        return s.value(row).to_owned();
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

#[test]
fn estimates_and_summaries_do_not_depend_on_partitions() -> R {
    let s = Session::new()?;
    for n in [1_u32, 8] {
        let from = format!("(SELECT /*+ REPARTITION({n}) */ id FROM range(0, 100))");
        let got = rows(
            &s,
            &format!(
                "SELECT tuple_sketch_estimate_double(tuple_sketch_agg_double(id % 10, CAST(1 AS DOUBLE))), \
                 tuple_sketch_summary_double(tuple_sketch_agg_double(id % 10, CAST(1 AS DOUBLE)), 'sum'), \
                 tuple_sketch_summary_integer(tuple_sketch_agg_integer(id % 10, 2, 12, 'max'), 'max') FROM {from}"
            ),
        )?;
        assert_eq!(got, ["10|100|2"]);
        let merged = rows(
            &s,
            &format!(
                "SELECT tuple_sketch_estimate_double(tuple_union_agg_double(sk)), \
                 tuple_sketch_summary_double(tuple_union_agg_double(sk), 'sum') FROM \
                 (SELECT tuple_sketch_agg_double(id, CAST(1 AS DOUBLE)) AS sk FROM {from} GROUP BY id % 4)"
            ),
        )?;
        assert_eq!(merged, ["100|100"]);
    }
    Ok(())
}

#[test]
fn intersection_keeps_the_shared_keys_and_an_empty_group_is_the_empty_sketch() -> R {
    let s = Session::new()?;
    let got = rows(
        &s,
        "SELECT tuple_sketch_estimate_double(tuple_intersection_agg_double(sk)) FROM \
         (SELECT tuple_sketch_agg_double(id, CAST(1 AS DOUBLE)) AS sk FROM range(0, 10) \
          UNION ALL SELECT tuple_sketch_agg_double(id, CAST(1 AS DOUBLE)) FROM range(0, 4))",
    )?;
    assert_eq!(got, ["4"]);
    let none = rows(
        &s,
        "SELECT tuple_sketch_estimate_double(tuple_intersection_agg_integer(sk)), \
         tuple_sketch_estimate_double(tuple_union_agg_integer(sk)) FROM \
         (SELECT tuple_sketch_agg_integer(id, 1) AS sk FROM range(0, 5)) WHERE false",
    )?;
    assert_eq!(none, ["0|0"]);
    let skipped = rows(
        &s,
        "SELECT tuple_sketch_estimate_double(tuple_sketch_agg_double(k, v)) \
         FROM VALUES (1, 1.0D), (NULL, 2.0D), (3, CAST(NULL AS DOUBLE)) AS t(k, v)",
    )?;
    assert_eq!(skipped, ["1"]);
    Ok(())
}

#[test]
fn bad_constants_are_analysis_errors() -> R {
    let s = Session::new()?;
    for (call, class) in [
        (
            "tuple_sketch_agg_double(id, 1.0D, 3)",
            "SKETCH_INVALID_LG_NOM_ENTRIES",
        ),
        (
            "tuple_sketch_agg_double(id, 1.0D, 12, 'median')",
            "TUPLE_INVALID_SKETCH_MODE",
        ),
        (
            "tuple_sketch_agg_double(id, 1.0D, CAST(NULL AS INT))",
            "UNEXPECTED_NULL",
        ),
    ] {
        let e = block_on(async {
            s.sql(&format!("SELECT {call} FROM range(0, 3)"))
                .await?
                .collect()
                .await
        })
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(e.contains(class), "{call}: {e}");
    }
    Ok(())
}

#[test]
fn a_bad_buffer_and_a_theta_buffer_are_runtime_errors() -> R {
    let s = Session::new()?;
    for (call, class) in [
        (
            "tuple_sketch_estimate_double(X'616263')",
            "TUPLE_INVALID_INPUT_SKETCH_BUFFER",
        ),
        (
            "tuple_sketch_estimate_double(theta_sketch_agg(id))",
            "TUPLE_INVALID_INPUT_SKETCH_BUFFER_FAMILY",
        ),
    ] {
        let e = block_on(async {
            s.sql(&format!("SELECT {call} FROM range(0, 3)"))
                .await?
                .collect()
                .await
        })
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(e.contains(class), "{call}: {e}");
    }
    Ok(())
}

#[test]
fn folded_constant_size_and_mode_lower_like_their_literals() -> R {
    let s = Session::new()?;
    let folded = rows(
        &s,
        "SELECT tuple_sketch_summary_integer(tuple_sketch_agg_integer(id % 10, 2, 6 + 6, concat('m', 'ax')), 'max') \
         FROM range(0, 100)",
    )?;
    let literal = rows(
        &s,
        "SELECT tuple_sketch_summary_integer(tuple_sketch_agg_integer(id % 10, 2, 12, 'max'), 'max') \
         FROM range(0, 100)",
    )?;
    assert_eq!(folded, literal);
    Ok(())
}

/// Spark 4.3.0: the MIN and MAX summaries of an empty sketch are zero, not an infinity.
#[test]
fn the_min_and_max_summaries_of_an_empty_sketch_are_zero() -> R {
    let s = Session::new()?;
    let got = rows(
        &s,
        "SELECT tuple_sketch_summary_double(tuple_sketch_agg_double(id, 1.0D, 12), 'max'), \
         tuple_sketch_summary_integer(tuple_sketch_agg_integer(id, 1, 12), 'min') \
         FROM range(5) WHERE id > 10",
    )?;
    assert_eq!(got, ["0|0"]);
    Ok(())
}

/// Spark 4.3.0 `local[2]` (and `REPARTITION(4)`): over several partitions the merged sketch is cut to
/// 2^lgNomEntries entries, which is what Krishiv's final stage always does.
#[test]
fn a_merged_sketch_is_cut_to_the_requested_size() -> R {
    let s = Session::new()?;
    let got = rows(
        &s,
        "SELECT CAST(length(s) AS STRING), tuple_sketch_estimate_double(s), tuple_sketch_theta_double(s) FROM \
         (SELECT tuple_sketch_agg_double(id, 1.0D, 5) s FROM range(0, 100))",
    )?;
    assert_eq!(got, ["568|130.20954598259996|0.2457577112224644"]);
    Ok(())
}
