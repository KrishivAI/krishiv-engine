// SPDX-License-Identifier: Apache-2.0
//! `TIMESTAMP` <-> `TIMESTAMP_NTZ` casts move between an instant and the
//! wall clock of the session zone. Every row is what a Spark 4.2.0 run
//! returned (the cast-matrix fixture, `America/Los_Angeles`).
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn one(s: &Session, sql: &str) -> String {
    let df = block_on(s.sql(sql)).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let c = batches[0].column(0);
    c.as_string_opt::<i32>()
        .map_or_else(|| c.as_string_view().value(0), |a| a.value(0))
        .to_owned()
}

#[test]
fn casts_between_instants_and_wall_clock_read_the_session_zone() {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.session.timeZone", "America/Los_Angeles")
        .unwrap();
    for (sql, want) in [
        (
            "CAST(CAST(TIMESTAMP'0001-01-01 00:00:00' AS TIMESTAMP_NTZ) AS STRING)",
            "0001-01-01 00:00:00",
        ),
        (
            "CAST(CAST(TIMESTAMP'1969-12-31 23:59:59.999999' AS TIMESTAMP_NTZ) AS STRING)",
            "1969-12-31 23:59:59.999999",
        ),
        (
            "CAST(CAST(TIMESTAMP_NTZ'1970-01-01 00:00:00' AS TIMESTAMP) AS STRING)",
            "1970-01-01 00:00:00",
        ),
        (
            "CAST(CAST(TIMESTAMP_NTZ'0001-01-01 00:00:00' AS TIMESTAMP) AS STRING)",
            "0001-01-01 00:00:00",
        ),
        (
            "CAST(CAST(timestamp_micros(1500000) AS TIMESTAMP_NTZ) AS STRING)",
            "1969-12-31 16:00:01.5",
        ),
        (
            "CAST(CAST(timestamp_micros(1500000) AS TIMESTAMP_NTZ) AS DATE) || ''",
            "1969-12-31",
        ),
    ] {
        assert_eq!(one(&s, &format!("SELECT {sql}")), want, "{sql}");
    }
}
