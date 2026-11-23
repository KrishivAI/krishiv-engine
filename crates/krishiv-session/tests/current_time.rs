// SPDX-License-Identifier: Apache-2.0
//! `current_time([p])` and `hash`/`xxhash64` of TIME, checked against a
//! Spark 4.2.0 run (`spark.sql.timeType.enabled`).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn run(sql: &str) -> Result<Option<String>, String> {
    run_on(SparkProfile::default(), sql)
}

fn run_on(profile: SparkProfile, sql: &str) -> Result<Option<String>, String> {
    let s = Session::builder()
        .profile(profile)
        .build()
        .map_err(|e| e.to_string())?;
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
fn a_precision_outside_the_profiles_range_is_refused() {
    // Spark 4.3 (the default profile): TIME precision is [0, 9].
    for p in ["10", "-1"] {
        let e = run(&format!("current_time({p})")).unwrap_err();
        assert!(e.contains("DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE"), "{e}");
    }
    assert_eq!(text("typeof(current_time(7))"), "time(7)");
    // Spark 4.2: [0, 6].
    for p in ["7", "-1"] {
        let e = run_on(SparkProfile::V4_2, &format!("current_time({p})")).unwrap_err();
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

#[test]
fn bare_current_time_is_the_function() {
    assert_eq!(text("typeof(current_time)"), "time(6)");
}

#[test]
fn localtime_with_a_column_list_is_a_multi_name_alias_under_spark_4_3() {
    // Spark 4.3.0 (`SqlBaseParser.g4` v4.3.0 `#currentLike` has `LOCALTIME`):
    // `SELECT localtime(n) FROM VALUES (6) t(n)` is `localtime AS (n)`, so
    // `MULTI_ALIAS_WITHOUT_GENERATOR` ("... with \"current_time(6)\"").
    let sql = "SELECT localtime(n) FROM VALUES (6) t(n)";
    let s = Session::builder().build().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let e = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_err()
        .to_string();
    assert!(e.contains("MULTI_ALIAS_WITHOUT_GENERATOR"), "{e}");
    assert!(e.contains("current_time"), "{e}");
    // The other `currentLike` words take the same path before `FROM` (probe2-430:
    // `current_date(n) FROM VALUES (6) t(n)` is `MULTI_ALIAS_WITHOUT_GENERATOR`).
    for f in ["current_time", "current_date", "current_timestamp"] {
        let e = block_on(async {
            s.sql(&format!("SELECT {f}(n) FROM VALUES (6) t(n)"))
                .await?
                .collect()
                .await
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("MULTI_ALIAS_WITHOUT_GENERATOR"), "{f}: {e}");
    }
    // A bare `localtime` is `current_time()`, and `localtime(3)` the call.
    assert_eq!(text("typeof(localtime)"), "time(6)");
    assert_eq!(text("typeof(localtime(3))"), "time(3)");
}
