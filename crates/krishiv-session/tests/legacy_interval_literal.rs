// SPDX-License-Identifier: Apache-2.0
//! With `spark.sql.legacy.interval.enabled` an interval literal is a calendar interval: days kept
//! apart from the time of day, and year-month with day-time fields mixed in one literal.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn shown(s: &Session, sql: &str) -> String {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let col = batches[0].column(0);
    arrow_cast::display::array_value_to_string(col.as_ref(), 0).unwrap()
}

#[test]
fn legacy_interval_literals_are_calendar_intervals() {
    let s = Session::builder()
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.sql.legacy.interval.enabled", "true")
        .build()
        .unwrap();
    for (sql, want) in [
        ("SELECT typeof(interval '999' minute)", "interval"),
        ("SELECT cast(interval '999' hour AS string)", "999 hours"),
        ("SELECT cast(interval '999' day AS string)", "999 days"),
        (
            "SELECT cast(interval '1 2:03' day to minute AS string)",
            "1 days 2 hours 3 minutes",
        ),
        (
            "SELECT cast(interval 1 year 2 days AS string)",
            "1 years 2 days",
        ),
    ] {
        assert_eq!(shown(&s, sql), want, "{sql}");
    }
}
