// SPDX-License-Identifier: Apache-2.0
//! A string (or cast string) `RANGE` frame bound is cast to the key's type: a FLOAT or DOUBLE key reads `'inf'` and
//! `'NaN'` as values, an integer key raises the cast's error. Expected rows are Spark 4.2.0's, from the
//! `postgreSQL/window_part2` golden (`numerics` table).
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;

fn session(dir: &std::path::Path) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.to_str().unwrap())
        .unwrap();
    for sql in [
        "create table numerics (id int, f_float4 float, f_float8 float, f_numeric int) using parquet",
        "insert into numerics values (1,-3,-3,-3),(2,-1,-1,-1),(3,0,0,0),(4,1.1,1.1,1.1),(5,1.12,1.12,1.12),(6,2,2,2),(7,100,100,100)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    s
}

/// `first(id)-last(id)` of every row (ordered by id), or the error text.
fn frames(s: &Session, key: &str, bounds: &str) -> Result<String, String> {
    let sql = format!(
        "select id, first(id) over w, last(id) over w from numerics \
         window w as (order by {key} range between {bounds}) order by id"
    );
    let batches =
        block_on(async { s.sql(&sql).await?.collect().await }).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for b in &batches {
        let (f, l) = (
            b.column(1).as_primitive::<Int32Type>(),
            b.column(2).as_primitive::<Int32Type>(),
        );
        out.extend((0..b.num_rows()).map(|i| format!("{}-{}", f.value(i), l.value(i))));
    }
    Ok(out.join(" "))
}

#[test]
fn infinity_and_nan_bounds_on_float_keys_match_spark() {
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    for key in ["f_float4", "f_float8"] {
        for bounds in [
            "'inf' preceding and 'inf' following",
            "cast('inf' as float) preceding and cast('inf' as float) following",
        ] {
            assert_eq!(
                frames(&s, key, bounds).unwrap(),
                "1-7 1-7 1-7 1-7 1-7 1-7 1-7",
                "{key} {bounds}"
            );
        }
        // NaN sorts last, so `key + NaN` reaches the end of the partition.
        assert_eq!(
            frames(&s, key, "1.1 preceding and 'NaN' following").unwrap(),
            "1-7 2-7 2-7 3-7 4-7 4-7 7-7",
            "{key}"
        );
    }
}

#[test]
fn a_nan_string_bound_on_an_integer_key_is_a_cast_error() {
    let d = tempfile::tempdir().unwrap();
    let s = session(d.path());
    let err = frames(&s, "f_numeric", "1.1 preceding and 'NaN' following").unwrap_err();
    assert!(err.contains("CAST_INVALID_INPUT"), "{err}");
}
