// SPDX-License-Identifier: Apache-2.0
//! `TIME ± day-time interval` and `TIME - TIME` (Spark 4.2.0 `TimeAddInterval`, `SubtractTimes`). Expected values:
//! `tests/spark-golden/v4.2.0/results/time.sql.out` (0-based statement numbers in the comments).
#![expect(clippy::unwrap_used, reason = "test code")]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(async {
        s.sql("SET spark.sql.timeType.enabled = true")
            .await?
            .collect()
            .await
    })
    .unwrap();
    s
}

/// The column name, Spark type and first value (as a string) of `sql`.
fn answer(s: &Session, sql: &str) -> (String, String, String) {
    let df = block_on(s.sql(sql)).unwrap();
    let name = block_on(df.schema()).unwrap().fields()[0].name().clone();
    let typed = block_on(async {
        s.sql(&format!(
            "SELECT typeof(x), CAST(x AS STRING) FROM ({sql}) q(x)"
        ))
        .await?
        .collect()
        .await
    })
    .unwrap();
    let text = |c: usize| {
        let a = typed[0].column(c);
        if a.is_null(0) {
            return "NULL".to_owned();
        }
        a.as_string_opt::<i32>().map_or_else(
            || a.as_string_view().value(0).to_owned(),
            |x| x.value(0).to_owned(),
        )
    };
    (name, text(0), text(1))
}

#[test]
fn time_plus_and_minus_an_interval() {
    let s = session();
    for (sql, name, ty, value) in [
        // #151
        (
            "SELECT '12:43:33.1234' :: TIME(4) + INTERVAL '01:04:05.56' HOUR TO SECOND",
            "CAST(12:43:33.1234 AS TIME(4)) + INTERVAL '01:04:05.56' HOUR TO SECOND",
            "time(6)",
            "13:47:38.6834",
        ),
        // #152, #153: a NULL is a day-time interval, and the time is shown first.
        (
            "SELECT TIME'08:30' + NULL",
            "TIME '08:30:00' + NULL",
            "time(6)",
            "NULL",
        ),
        (
            "SELECT NULL + TIME'08:30'",
            "TIME '08:30:00' + NULL",
            "time(6)",
            "NULL",
        ),
        // #155-#158
        (
            "SELECT TIME'12:30' - INTERVAL '12:29:59.000001' HOUR TO SECOND",
            "TIME '12:30:00' - INTERVAL '12:29:59.000001' HOUR TO SECOND",
            "time(6)",
            "00:00:00.999999",
        ),
        (
            "SELECT '23:59:59.999999' :: TIME - INTERVAL '23:59:59.999999' HOUR TO SECOND",
            "CAST(23:59:59.999999 AS TIME(6)) - INTERVAL '23:59:59.999999' HOUR TO SECOND",
            "time(6)",
            "00:00:00",
        ),
        (
            "SELECT '08:30' :: TIME(0) - INTERVAL '6' HOUR",
            "CAST(08:30 AS TIME(0)) - INTERVAL '06' HOUR",
            "time(0)",
            "02:30:00",
        ),
    ] {
        assert_eq!(
            answer(&s, sql),
            (name.to_owned(), ty.to_owned(), value.to_owned()),
            "{sql}"
        );
    }
}

#[test]
fn time_minus_time() {
    let s = session();
    for (sql, name, value) in [
        // #229-#235
        (
            "SELECT TIME'12:30:41' - TIME'10:00'",
            "TIME '12:30:41' - TIME '10:00:00'",
            "INTERVAL '02:30:41' HOUR TO SECOND",
        ),
        (
            "SELECT TIME'08:30' - NULL",
            "TIME '08:30:00' - NULL",
            "NULL",
        ),
        (
            "SELECT NULL - TIME'10:32'",
            "NULL - TIME '10:32:00'",
            "NULL",
        ),
        (
            "SELECT '12:30:41.123' - TIME'10:00:01'",
            "12:30:41.123 - TIME '10:00:01'",
            "INTERVAL '02:30:40.123' HOUR TO SECOND",
        ),
        (
            "SELECT '00:00:00.1234' :: TIME(4) - TIME'23:59:59'",
            "CAST(00:00:00.1234 AS TIME(4)) - TIME '23:59:59'",
            "INTERVAL '-23:59:58.8766' HOUR TO SECOND",
        ),
    ] {
        assert_eq!(
            answer(&s, sql),
            (
                name.to_owned(),
                "interval hour to second".to_owned(),
                value.to_owned()
            ),
            "{sql}"
        );
    }
}

#[test]
fn a_result_outside_the_day_overflows() {
    let s = session();
    let e = block_on(async {
        s.sql("SELECT TIME'23:00' + INTERVAL '2' HOUR")
            .await?
            .collect()
            .await
    })
    .unwrap_err();
    assert_eq!(e.error_class().name(), "DATETIME_OVERFLOW");
}
