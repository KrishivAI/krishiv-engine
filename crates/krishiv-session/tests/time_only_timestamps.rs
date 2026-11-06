// SPDX-License-Identifier: Apache-2.0
//! A time of day alone as a timestamp literal, and inline-table cells that fail to cast (found by
//! the parser-negatives gate, `postgreSQL/window_part3`).
#![expect(clippy::unwrap_used, reason = "test code")]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

fn one(s: &Session, sql: &str) -> String {
    let b = block_on(
        block_on(s.sql(&format!("SELECT CAST(({sql}) AS STRING)")))
            .unwrap()
            .collect(),
    )
    .unwrap();
    let c = &b[0].columns()[0];
    c.as_string_opt::<i32>().map_or_else(
        || c.as_string_view().value(0).to_owned(),
        |v| v.value(0).to_owned(),
    )
}

#[test]
fn a_bare_time_is_that_time_today_in_the_session_zone() {
    let s = Session::new().unwrap();
    s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
    assert_eq!(
        one(
            &s,
            "SELECT hour(TIMESTAMP '11:45:30') * 10000 + minute(TIMESTAMP 'T11:45:30') * 100 + second(TIMESTAMP '11:45:30')"
        ),
        "114530"
    );
    assert_eq!(
        one(
            &s,
            "SELECT CAST(TIMESTAMP '11:00' AS DATE) = current_date()"
        ),
        "true"
    );
    assert_eq!(
        one(
            &s,
            "SELECT TIMESTAMP_LTZ '23:59:59.5' > TIMESTAMP_LTZ '23:59:59'"
        ),
        "true"
    );
}

#[test]
fn a_malformed_timestamp_cast_in_an_inline_table_is_a_failed_cell() {
    let s = Session::new().unwrap();
    for ansi in ["true", "false"] {
        s.conf().set("spark.sql.ansi.enabled", ansi).unwrap();
        let sql = "SELECT * FROM VALUES (1, CAST('1 year' AS TIMESTAMP)) AS t(a, b)";
        let r = block_on(s.sql(sql)).and_then(|df| block_on(df.collect()));
        if ansi == "true" {
            let e = r.unwrap_err();
            assert_eq!(
                e.error_class(),
                ErrorClass::InvalidInlineTableFailedSqlExpressionEvaluation
            );
            assert!(e.to_string().contains("CAST(1 year AS TIMESTAMP)"), "{e}");
        } else {
            // Not ANSI: the malformed string is NULL.
            assert!(r.is_ok(), "{r:?}");
        }
    }
}
