// SPDX-License-Identifier: Apache-2.0
//! `current_time([p])` and `hash`/`xxhash64` of TIME, checked against a
//! Spark 4.2.0 run (`spark.sql.timeType.enabled`).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(sql: &str) -> Result<Option<String>, String> {
    let s = Session::new().map_err(|e| e.to_string())?;
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).map_err(|e| e.to_string())?;
    let df =
        block_on(s.sql(&format!("SELECT CAST(({sql}) AS STRING)"))).map_err(|e| e.to_string())?;
    let b = block_on(df.collect()).map_err(|e| e.to_string())?;
    let c = &b[0].columns()[0];
    Ok((!c.is_null(0)).then(|| c.as_string_view().value(0).to_owned()))
}

fn text(sql: &str) -> String {
    run(sql).unwrap().unwrap()
}

#[test]
fn the_precision_types_the_result_and_cuts_the_digits() {
    assert_eq!(
        text("typeof(current_time()) || typeof(current_time(0)) || typeof(current_time(3))"),
        "time(6)time(0)time(3)"
    );
    assert_eq!(text("current_time(0) = current_time(0)"), "true");
    assert_eq!(text("current_time(1+1) = current_time(2)"), "true");
    assert_eq!(
        text("CAST(current_time() AS STRING) LIKE '__:__:__%'"),
        "true"
    );
    // A fresh `current_time(p)` has no more than p fractional digits.
    assert_eq!(
        text("CAST(current_time(0) AS STRING) LIKE '__:__:__'"),
        "true"
    );
    assert_eq!(text("current_time() IS NULL"), "false");
}

#[test]
fn a_precision_outside_zero_to_six_is_refused() {
    for p in ["7", "-1"] {
        let e = run(&format!("current_time({p})")).unwrap_err();
        assert!(e.contains("DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE"), "{e}");
    }
    let e = run("current_time(1, 2)").unwrap_err();
    assert!(e.contains("WRONG_NUM_ARGS"), "{e}");
}

#[test]
fn time_hashes_as_spark() {
    assert_eq!(text("hash(time'10:00')"), "-1113572777");
    assert_eq!(text("xxhash64(time'10:00')"), "-8977161405312030080");
}
