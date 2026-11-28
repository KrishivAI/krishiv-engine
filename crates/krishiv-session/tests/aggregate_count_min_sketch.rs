// SPDX-License-Identifier: Apache-2.0
//! `count_min_sketch` through SQL against what Spark 4.3.0 answered
//! (`tests/spark-golden/functions/count_min_sketch/`): the serialized sketch byte for byte over every input type,
//! the constants' checks with their error classes, and partial states merged across partitions.
#![forbid(unsafe_code)]

use std::fmt::Write as _;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

const FIXTURE: &str =
    include_str!("../../../tests/spark-golden/functions/count_min_sketch/fixture.json");

fn cell(a: &dyn Array, row: usize) -> String {
    if a.is_null(row) {
        return "None".into();
    }
    if let Some(s) = a.as_string_opt::<i32>() {
        return s.value(row).to_owned();
    }
    if let Some(s) = a.as_string_view_opt() {
        return s.value(row).to_owned();
    }
    if let Some(s) = a.as_primitive_opt::<Int32Type>() {
        return s.value(row).to_string();
    }
    if let Some(s) = a.as_primitive_opt::<Int64Type>() {
        return s.value(row).to_string();
    }
    format!("<{}>", a.data_type())
}

/// The column names and rows, or the error's text.
fn run(s: &Session, sql: &str) -> Result<(Vec<String>, Vec<String>), String> {
    block_on(async {
        let frame = s.sql(sql).await.map_err(|e| e.to_string())?;
        let batches = frame.collect().await.map_err(|e| e.to_string())?;
        let names = batches.first().map_or_else(Vec::new, |b| {
            b.schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        });
        let mut rows = Vec::new();
        for b in &batches {
            for r in 0..b.num_rows() {
                rows.push(
                    b.columns()
                        .iter()
                        .map(|c| cell(c.as_ref(), r))
                        .collect::<Vec<_>>()
                        .join("|"),
                );
            }
        }
        Ok((names, rows))
    })
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| s.as_str().map(str::to_owned))
        .collect()
}

#[test]
fn every_statement_answers_as_spark() -> R {
    let s = Session::new()?;
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let cases = fixture["cases"].as_array().ok_or("no cases")?;
    let mut wrong = String::new();
    for case in cases {
        let sql = case["sql"].as_str().ok_or("no sql")?;
        let got = run(&s, sql);
        if let Some(class) = case["error"].as_str() {
            let state = case["sqlstate"].as_str().ok_or("no sqlstate")?;
            match got {
                Err(e)
                    if e.contains(&format!("[{class}]"))
                        && e.contains(&format!("SQLSTATE: {state}")) => {}
                other => writeln!(wrong, "{sql}\n  want {class} {state}\n  got {other:?}")?,
            }
        } else {
            let want = (strings(&case["columns"]), strings(&case["rows"]));
            if got.as_ref() != Ok(&want) {
                writeln!(wrong, "{sql}\n  want {want:?}\n  got {got:?}")?;
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong}");
    Ok(())
}

/// Spark's serialization: version (4 bytes), total count (8), depth (4), width (4), then the counters, so the
/// header says what the sketch of `n` values at `eps = 0.1`, `confidence = 0.9` must hold:
/// depth `ceil(-ln(1 - 0.9) / ln 2) = 4`, width `ceil(2 / 0.1) = 20`.
fn header(hex: &str) -> (u64, u32, u32) {
    let at = |from: usize, len: usize| {
        u64::from_str_radix(&hex[from..from + len], 16).unwrap_or(u64::MAX)
    };
    (
        at(8, 16),
        u32::try_from(at(24, 8)).unwrap_or(u32::MAX),
        u32::try_from(at(32, 8)).unwrap_or(u32::MAX),
    )
}

#[test]
fn the_sketch_does_not_depend_on_partitions_or_batches() -> R {
    let s = Session::new()?;
    let q = |partitions: u32| {
        format!(
            "SELECT id % 7 AS g, hex(count_min_sketch(id % 100, 0.1d, 0.9d, 1)) AS h \
             FROM range(0, 20000, 1, {partitions}) GROUP BY id % 7 ORDER BY g"
        )
    };
    let one = run(&s, &q(1))?.1;
    assert_eq!(one.len(), 7);
    // Each group holds the rows of its residue: 0..20000 has 2858 of residue 0 and 2857 of the others.
    for (g, row) in one.iter().enumerate() {
        let (group, hex) = row.split_once('|').ok_or("no hex")?;
        assert_eq!(group, g.to_string());
        let want = if g == 0 { 2858 } else { 2857 };
        assert_eq!(header(hex), (want, 4, 20), "group {g}");
    }
    assert_eq!(run(&s, &q(8))?.1, one);
    Ok(())
}

#[test]
fn a_collated_value_is_spark_s_type_mismatch() -> R {
    let s = Session::new()?;
    let got = run(
        &s,
        "SELECT count_min_sketch('a' COLLATE UTF8_LCASE, 0.5d, 0.5d, 1)",
    );
    assert!(
        matches!(&got, Err(e) if e.contains("DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE")),
        "{got:?}"
    );
    Ok(())
}

#[test]
fn a_sketch_past_what_a_binary_holds_or_with_empty_rows_is_refused() -> R {
    let s = Session::new()?;
    for (sql, why) in [
        (
            "SELECT count_min_sketch(id, 1e-12d, 0.5d, 1) FROM range(3)",
            "more than 2 GiB",
        ),
        (
            "SELECT count_min_sketch(id, double('Infinity'), 0.5d, 1) FROM range(3)",
            "infinite or NaN eps",
        ),
        (
            "SELECT count_min_sketch(id, double('NaN'), 0.5d, 1) FROM range(3)",
            "infinite or NaN eps",
        ),
        (
            "SELECT count_min_sketch(id, 0.5d, 0.5d, 1) OVER (ORDER BY id) FROM range(3)",
            "the window aggregate count_min_sketch",
        ),
    ] {
        let got = run(&s, sql);
        assert!(
            matches!(&got, Err(e) if e.contains("KRISHIV_UNSUPPORTED_FEATURE") && e.contains(why)),
            "{sql}: {got:?}"
        );
    }
    Ok(())
}
