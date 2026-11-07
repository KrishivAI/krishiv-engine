// SPDX-License-Identifier: Apache-2.0
//! Casts to and between `GEOMETRY` and `GEOGRAPHY`, `try_cast` included,
//! against a real Spark 4.2.0 run (`tests/fixtures/spark-4.2.0/geo-cast-goldens.json`,
//! from `research/2026-11-06-spark-geo-wkt-and-geography-cast.md`): the
//! type check (equal SRIDs only), the runtime checks (a geographic SRID,
//! coordinates in range), and `try_cast` turning only the runtime errors
//! into NULL. Spark's results are the same with ANSI on and off.
//!
//! Every case of the fixture matches: a cast from `GEOMETRY(ANY)` to `GEOGRAPHY(ANY)` is
//! accepted at analysis (equal `ANY` SRIDs) and checks each value's SRID against Spark's
//! registry of geographic systems at run time (`ST_INVALID_SRID_VALUE`), then its coordinates.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

/// What the first column of `sql` prints as, rows joined by `, `: EWKT for a spatial value,
/// `true`/`false` for a boolean, `NULL` for a NULL.
fn first_cell(s: &Session, sql: &str) -> Result<String, String> {
    let df = block_on(s.sql(sql)).map_err(|e| e.to_string())?;
    let batches = block_on(df.collect()).map_err(|e| e.to_string())?;
    let mut cells = Vec::new();
    for batch in &batches {
        let column = &batch.columns()[0];
        for row in 0..batch.num_rows() {
            if column.is_null(row) {
                cells.push("NULL".to_owned());
                continue;
            }
            if let Some(b) = column.as_boolean_opt() {
                cells.push(b.value(row).to_string());
                continue;
            }
            let st = column
                .as_struct_opt()
                .ok_or_else(|| format!("a {} column", column.data_type()))?;
            let srid = st.column(0).as_primitive::<Int32Type>().value(row);
            let wkb = st.column(1).as_binary_view().value(row);
            cells.push(krishiv_types::geo::to_ewkt(srid, wkb).map_err(|e| e.to_string())?);
        }
    }
    Ok(cells.join(", "))
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
            if !ok {
                wrong.push(format!(
                    "ansi={ansi}: {sql}\n    want {c}\n    got  {got:?}"
                ));
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
