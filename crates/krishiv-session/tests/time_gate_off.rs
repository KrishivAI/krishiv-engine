// SPDX-License-Identifier: Apache-2.0
//! With `spark.sql.timeType.enabled` off (the default) Spark 4.2.0 still evaluates a `TIME` that is produced,
//! compared, shown or projected, and refuses the written type and the time functions (SQL-26, probe on a 4.2.0
//! container).
#![forbid(unsafe_code)]

use arrow_array::Array;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Result<String, String> {
    let df = block_on(s.sql(sql)).map_err(|e| e.error_class().name().to_owned())?;
    let b = block_on(df.collect()).map_err(|e| e.error_class().name().to_owned())?;
    let c = &b[0].columns()[0];
    Ok(if c.is_null(0) {
        "NULL".to_owned()
    } else {
        arrow_cast::display::array_value_to_string(c, 0).map_err(|e| e.to_string())?
    })
}

#[test]
fn evaluates_what_spark_evaluates_with_the_key_off() {
    let s = Session::new().unwrap();
    for q in [
        "SELECT current_time(3)",
        "SELECT current_time(0) IS NOT NULL",
        "SELECT typeof(current_time(3))",
        "SELECT TIME'12:00:00'",
        "SELECT TIME'12:00:00' < TIME'13:00:00'",
        "SELECT typeof(TIME'12:00:00')",
        "SELECT current_time(3) < current_time(3)",
        "SELECT CAST(current_time() AS STRING)",
        "SELECT * FROM (SELECT current_time(3) AS t)",
    ] {
        assert!(run(&s, q).is_ok(), "{q}: {:?}", run(&s, q));
    }
    assert_eq!(
        run(&s, "SELECT current_time(0) IS NOT NULL").unwrap(),
        "true"
    );
    assert_eq!(
        run(&s, "SELECT current_time(3) < current_time(3)").unwrap(),
        "false"
    );
    assert_eq!(run(&s, "SELECT typeof(TIME'12:00:00')").unwrap(), "time(6)");
    assert_eq!(
        run(&s, "SELECT typeof(current_time(3))").unwrap(),
        "time(3)"
    );
    assert_eq!(
        run(&s, "SELECT TIME'12:00:00' < TIME'13:00:00'").unwrap(),
        "true"
    );
}

#[test]
fn refuses_the_written_type_and_the_time_functions_with_the_key_off() {
    let s = Session::new().unwrap();
    for q in [
        "SELECT hour(current_time())",
        "SELECT CAST('12:00' AS TIME)",
        "SELECT hour(TIME'12:00:00')",
        "SELECT make_time(1, 2, 3)",
        "SELECT to_time('12:00')",
        "SELECT CAST(NULL AS TIME)",
        "SELECT time_trunc('HOUR', TIME'12:34:56')",
    ] {
        assert_eq!(run(&s, q).unwrap_err(), "UNSUPPORTED_TIME_TYPE", "{q}");
    }
}
