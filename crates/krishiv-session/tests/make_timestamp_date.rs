// SPDX-License-Identifier: Apache-2.0
//! `make_timestamp(date[, time[, zone]])`, checked against a Spark 4.2.0 run
//! (session zone `America/Los_Angeles`) beyond the `timestamp` goldens.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn text(sql: &str) -> Option<String> {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    block_on(s.sql("SET spark.sql.session.timeZone = America/Los_Angeles")).unwrap();
    let df = block_on(s.sql(&format!("SELECT CAST(({sql}) AS STRING)"))).unwrap();
    let b = block_on(df.collect()).unwrap();
    let c = &b[0].columns()[0];
    (!c.is_null(0)).then(|| c.as_string_view().value(0).to_owned())
}

#[test]
fn a_date_and_a_time_are_read_in_the_session_zone() {
    assert_eq!(
        text("make_timestamp(date'2021-03-04', time'10:11:12.5')").as_deref(),
        Some("2021-03-04 10:11:12.5")
    );
    assert_eq!(
        text("make_timestamp(date'2021-03-04')").as_deref(),
        Some("2021-03-04 00:00:00")
    );
}

#[test]
fn a_zone_argument_shifts_the_instant() {
    // 10:11:12.5 in Tokyo is 18:11:12.5 the day before in Los Angeles.
    assert_eq!(
        text("make_timestamp(date'2021-03-04', time'10:11:12.5', 'Asia/Tokyo')").as_deref(),
        Some("2021-03-03 17:11:12.5")
    );
    assert_eq!(
        text("make_timestamp_ltz(date'2021-03-04', time'10:11:12', 'UTC')").as_deref(),
        Some("2021-03-04 02:11:12")
    );
}

#[test]
fn ntz_is_the_wall_clock_and_a_null_argument_is_null() {
    assert_eq!(
        text("make_timestamp_ntz(date'2021-03-04', time'10:11:12.123456')").as_deref(),
        Some("2021-03-04 10:11:12.123456")
    );
    assert_eq!(text("make_timestamp(NULL, time'10:11:12')"), None);
    assert_eq!(text("make_timestamp(date'2021-03-04', NULL)"), None);
    assert_eq!(
        text("make_timestamp(date'2021-03-04', time'10:11:12', NULL)"),
        None
    );
    assert_eq!(
        text("try_make_timestamp(date'2021-03-04', time'10:11:12')").as_deref(),
        Some("2021-03-04 10:11:12")
    );
}
