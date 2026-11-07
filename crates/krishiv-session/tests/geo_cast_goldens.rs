// SPDX-License-Identifier: Apache-2.0
//! Casts to and between `GEOMETRY` and `GEOGRAPHY`, `try_cast` included,
//! against a real Spark 4.2.0 run (`tests/fixtures/spark-4.2.0/geo-cast-goldens.json`,
//! from `research/2026-11-06-spark-geo-wkt-and-geography-cast.md`): the
//! type check (equal SRIDs only), the runtime checks (a geographic SRID,
//! coordinates in range), and `try_cast` turning only the runtime errors
//! into NULL. Spark's results are the same with ANSI on and off.
//!
//! Not matched yet (`is_known_gap`): a cast from `GEOMETRY(ANY)` to
//! `GEOGRAPHY(ANY)`. Spark accepts it at analysis and checks the value's SRID
//! at run time against its registry of geographic systems
//! (`sql/api/.../srs_registry.csv`, 1,406 geographic SRIDs, generated from
//! PROJ); the engine refuses it at analysis, and its geographic check knows
//! only 4326. The gap cases must still differ, so the list can only shrink.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

/// Whether `sql` is one of the cases the engine does not match yet.
fn is_known_gap(sql: &str) -> bool {
    sql.contains("GEOMETRY(ANY)") && sql.contains("GEOGRAPHY(ANY)")
}

/// What the first cell of `sql` prints as: EWKT for a spatial value,
/// `true`/`false` for a boolean, `NULL` for a NULL.
fn first_cell(s: &Session, sql: &str) -> Result<String, String> {
    let df = block_on(s.sql(sql)).map_err(|e| e.to_string())?;
    let batches = block_on(df.collect()).map_err(|e| e.to_string())?;
    let column = &batches[0].columns()[0];
    if column.is_null(0) {
        return Ok("NULL".to_owned());
    }
    if let Some(b) = column.as_boolean_opt() {
        return Ok(b.value(0).to_string());
    }
    let st = column
        .as_struct_opt()
        .ok_or_else(|| format!("a {} column", column.data_type()))?;
    let srid = st.column(0).as_primitive::<Int32Type>().value(0);
    let wkb = st.column(1).as_binary_view().value(0);
    krishiv_types::geo::to_ewkt(srid, wkb).map_err(|e| e.to_string())
}

#[test]
fn every_spark_geography_cast_case_agrees() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.2.0/geo-cast-goldens.json"
    ))
    .unwrap_or_else(|e| panic!("{e}"));
    let doc: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}"));
    let cases = doc["cases"].as_array().expect("cases");
    assert!(cases.len() >= 40);
    let mut wrong = Vec::new();
    for ansi in ["true", "false"] {
        let s = Session::new().unwrap();
        block_on(s.sql(&format!("SET spark.sql.ansi.enabled = {ansi}"))).unwrap();
        for c in cases {
            let sql = c["sql"].as_str().expect("sql");
            let got = first_cell(&s, sql);
            let ok = match (c["error_class"].as_str(), c["text"].as_str()) {
                (Some(class), _) => got.as_ref().err().is_some_and(|e| e.contains(class)),
                (None, Some(want)) => got.as_deref() == Ok(want),
                _ => false,
            };
            match (ok, is_known_gap(sql)) {
                (false, false) => wrong.push(format!(
                    "ansi={ansi}: {sql}\n    want {c}\n    got  {got:?}"
                )),
                (true, true) => wrong.push(format!(
                    "ansi={ansi}: now matches Spark, remove it from is_known_gap: {sql}"
                )),
                _ => {}
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
