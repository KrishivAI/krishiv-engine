// SPDX-License-Identifier: Apache-2.0
//! `time_bucket` (Spark 4.2.0 `TimeBucket`). Expected values: the Spark 4.2.0 golden
//! `tests/spark-golden/v4.2.0/results/time-bucket.sql.out` (statement numbers in the comments).
#![expect(clippy::unwrap_used, reason = "test code")]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn session(zone: &str) -> Session {
    let s = Session::new().unwrap();
    block_on(async {
        s.sql(&format!("SET TIME ZONE '{zone}'"))
            .await?
            .collect()
            .await
    })
    .unwrap();
    s
}

/// The rows of `sql`'s first column cast to string (`NULL` for NULL).
fn texts(s: &Session, sql: &str) -> Vec<String> {
    let q = format!("SELECT CAST(b AS STRING) FROM ({sql}) q(b)");
    let batches = block_on(async { s.sql(&q).await?.collect().await }).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let a = b.column(0);
        for i in 0..b.num_rows() {
            out.push(if a.is_null(i) {
                "NULL".to_owned()
            } else {
                a.as_string_opt::<i32>().map_or_else(
                    || a.as_string_view().value(i).to_owned(),
                    |x| x.value(i).to_owned(),
                )
            });
        }
    }
    out
}

fn one(s: &Session, sql: &str) -> String {
    texts(s, sql).remove(0)
}

fn error(s: &Session, sql: &str) -> (String, Vec<(String, String)>) {
    let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
    let p = e
        .message_parameters()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    (e.error_class().name().to_owned(), p)
}

fn param(p: &[(String, String)], key: &str) -> String {
    p.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.clone())
        .unwrap_or_default()
}

#[test]
fn buckets_follow_the_golden() {
    let s = session("UTC");
    for (sql, expected) in [
        // #34-#44: day-time buckets, the default and explicit origins, `TIMESTAMP_NTZ`.
        (
            "SELECT time_bucket(INTERVAL '15' MINUTE, TIMESTAMP '2024-01-01 11:27:00')",
            "2024-01-01 11:15:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '7' DAY, TIMESTAMP '2024-01-10 11:27:00')",
            "2024-01-04 00:00:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '1 00:30' DAY TO MINUTE, TIMESTAMP '2024-06-20 10:00:00')",
            "2024-06-20 00:00:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '0.000001' SECOND, TIMESTAMP '2024-06-20 10:00:00.123456')",
            "2024-06-20 10:00:00.123456",
        ),
        (
            "SELECT time_bucket(INTERVAL '15' MINUTE, TIMESTAMP_NTZ '2024-01-01 11:27:00')",
            "2024-01-01 11:15:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '1' HOUR, TIMESTAMP '2024-01-01 11:27:00', TIMESTAMP '2025-01-01 00:30:00')",
            "2024-01-01 10:30:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '1' HOUR, TIMESTAMP_NTZ '2024-01-15 10:23:00', TIMESTAMP_NTZ '2024-01-15 00:30:00')",
            "2024-01-15 09:30:00",
        ),
        // #45-#56: year-month buckets, month-end clamping.
        (
            "SELECT time_bucket(INTERVAL '3' MONTH, TIMESTAMP '2024-05-15 10:00:00')",
            "2024-04-01 00:00:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '1-3' YEAR TO MONTH, TIMESTAMP '2024-06-20 10:00:00')",
            "2023-10-01 00:00:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '1' MONTH, TIMESTAMP '2024-03-20 09:00:00', TIMESTAMP '1970-01-15 00:00:00')",
            "2024-03-15 00:00:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '1' MONTH, TIMESTAMP '2024-03-01 12:00:00', TIMESTAMP '1970-01-31 00:00:00')",
            "2024-02-29 00:00:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '1' YEAR, TIMESTAMP '2025-03-01 00:00:00', TIMESTAMP '2024-02-29 00:00:00')",
            "2025-02-28 00:00:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '3' MONTH, TIMESTAMP '2024-02-15 10:00:00', TIMESTAMP '2024-08-01 00:00:00')",
            "2024-02-01 00:00:00",
        ),
        // #57-#60: before the epoch.
        (
            "SELECT time_bucket(INTERVAL '1' DAY, TIMESTAMP '1969-12-31 23:30:00')",
            "1969-12-31 00:00:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '1' HOUR, TIMESTAMP '2024-01-01 11:27:00', TIMESTAMP '1960-06-15 00:30:00')",
            "2024-01-01 10:30:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '1' MONTH, TIMESTAMP '1968-07-15 10:00:00')",
            "1968-07-01 00:00:00",
        ),
        // #61-#65: folded sizes and origins.
        (
            "SELECT time_bucket(INTERVAL '1' HOUR - INTERVAL '15' MINUTE, TIMESTAMP '2024-06-20 09:47:00')",
            "2024-06-20 09:45:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '2' MONTH + INTERVAL '1' MONTH, TIMESTAMP '2024-06-20 09:47:00')",
            "2024-04-01 00:00:00",
        ),
        (
            "SELECT time_bucket(INTERVAL '1' HOUR, TIMESTAMP '2024-06-20 09:47:00', TIMESTAMP '2024-01-01 00:00:00' + INTERVAL '5' MINUTE)",
            "2024-06-20 09:05:00",
        ),
        // #28-#33: NULLs.
        (
            "SELECT time_bucket(NULL, TIMESTAMP '2024-01-01 11:27:00')",
            "NULL",
        ),
        (
            "SELECT time_bucket(INTERVAL '1' HOUR, NULL, TIMESTAMP_NTZ '2024-01-01 00:00:00')",
            "NULL",
        ),
    ] {
        assert_eq!(one(&s, sql), expected, "{sql}");
    }
    assert_eq!(
        texts(
            &s,
            "SELECT time_bucket(INTERVAL '1' HOUR, t) FROM VALUES (TIMESTAMP '2024-01-15 10:23:00'), (CAST(NULL AS TIMESTAMP)) tab(t)"
        ),
        ["2024-01-15 10:00:00", "NULL"]
    );
}

#[test]
fn a_region_zone_buckets_on_its_calendar() {
    // #70-#74 under America/Los_Angeles: month and day buckets start at local midnight across both DST changes;
    // a 36-hour bucket adds its day part as calendar days.
    let s = session("America/Los_Angeles");
    for (t, expected) in [
        ("2024-03-15 10:00:00", "2024-03-01 00:00:00"),
        ("2024-04-15 10:00:00", "2024-04-01 00:00:00"),
    ] {
        assert_eq!(
            one(
                &s,
                &format!("SELECT time_bucket(INTERVAL '1' MONTH, TIMESTAMP '{t}')")
            ),
            expected
        );
    }
    for (t, expected) in [
        ("2024-03-10 12:00:00", "2024-03-10 00:00:00"),
        ("2024-03-11 12:00:00", "2024-03-11 00:00:00"),
        ("2024-11-03 12:00:00", "2024-11-03 00:00:00"),
        ("2024-11-04 12:00:00", "2024-11-04 00:00:00"),
    ] {
        assert_eq!(
            one(
                &s,
                &format!("SELECT time_bucket(INTERVAL '1' DAY, TIMESTAMP '{t}')")
            ),
            expected
        );
    }
    assert_eq!(
        one(
            &s,
            "SELECT time_bucket(INTERVAL '36' HOUR, TIMESTAMP '2024-11-05 11:30:00', TIMESTAMP '2024-11-01 00:00:00')"
        ),
        "2024-11-04 00:00:00"
    );
}

#[test]
fn errors_follow_the_golden() {
    let s = session("UTC");
    // #2, #3, #8: a bucket size that is not positive.
    let (class, p) = error(
        &s,
        "SELECT time_bucket(INTERVAL '0' SECOND, TIMESTAMP '2024-01-01 11:00:00')",
    );
    assert_eq!(class, "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE");
    assert_eq!(param(&p, "currentValue"), "INTERVAL '00' SECOND");
    assert_eq!(
        param(&p, "sqlExpr"),
        "\"time_bucket(INTERVAL '00' SECOND, TIMESTAMP '2024-01-01 11:00:00', TIMESTAMP '1970-01-01 00:00:00')\""
    );
    let (_, p) = error(
        &s,
        "SELECT time_bucket(INTERVAL '0' MONTH, TIMESTAMP '2024-01-01 11:00:00')",
    );
    assert_eq!(param(&p, "currentValue"), "INTERVAL '0' MONTH");
    let (c8, p) = error(
        &s,
        "SELECT time_bucket(INTERVAL '15' MINUTE - INTERVAL '15' MINUTE, TIMESTAMP '2024-06-20 09:47:00')",
    );
    assert_eq!(c8, "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE", "{p:?}");
    assert_eq!(param(&p, "currentValue"), "INTERVAL '00' MINUTE");
    // #12, #14-#18: types, no implicit casts.
    for (sql, index, required) in [
        (
            "SELECT time_bucket(INTERVAL '1' HOUR, TIMESTAMP '2024-01-01 11:27:00', TIMESTAMP_NTZ '2024-01-01 00:00:00')",
            "third",
            "\"TIMESTAMP\"",
        ),
        (
            "SELECT time_bucket('15 minutes', TIMESTAMP '2024-01-15 10:23:00')",
            "first",
            "(\"INTERVAL DAY TO SECOND\" or \"INTERVAL YEAR TO MONTH\")",
        ),
        (
            "SELECT time_bucket(INTERVAL '15' MINUTE, DATE '2024-01-15')",
            "second",
            "\"(TIMESTAMP OR TIMESTAMP WITHOUT TIME ZONE)\"",
        ),
        (
            "SELECT time_bucket(INTERVAL '15' MINUTE, '2024-01-15 10:23:00')",
            "second",
            "\"(TIMESTAMP OR TIMESTAMP WITHOUT TIME ZONE)\"",
        ),
        (
            "SELECT time_bucket(INTERVAL '15' MINUTE, TIMESTAMP '2024-01-15 10:23:00', '2024-01-01 00:00:00')",
            "third",
            "\"(TIMESTAMP OR TIMESTAMP WITHOUT TIME ZONE)\"",
        ),
    ] {
        let (class, p) = error(&s, sql);
        assert_eq!(class, "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE", "{sql}");
        assert_eq!(
            (param(&p, "paramIndex"), param(&p, "requiredType")),
            (index.to_owned(), required.to_owned()),
            "{sql}"
        );
    }
    // #19, #21, #23: not foldable.
    for (sql, name, ty) in [
        (
            "SELECT time_bucket(bs, TIMESTAMP '2024-06-20 09:47:00') FROM VALUES (INTERVAL '1' HOUR) tab(bs)",
            "`bucketSize`",
            "\"INTERVAL HOUR\"",
        ),
        (
            "SELECT time_bucket((SELECT INTERVAL '1' HOUR), TIMESTAMP '2024-06-20 09:47:00')",
            "`bucketSize`",
            "\"INTERVAL HOUR\"",
        ),
        (
            "SELECT time_bucket(INTERVAL '1' HOUR, TIMESTAMP '2024-06-20 09:47:00', o) FROM VALUES (TIMESTAMP '2024-01-01 00:00:00') tab(o)",
            "`origin`",
            "\"TIMESTAMP\"",
        ),
    ] {
        let (class, p) = error(&s, sql);
        assert_eq!(class, "DATATYPE_MISMATCH.NON_FOLDABLE_INPUT", "{sql}");
        assert_eq!(
            (param(&p, "inputName"), param(&p, "inputType")),
            (name.to_owned(), ty.to_owned()),
            "{sql}"
        );
    }
    // #26, #27: two or three arguments.
    let (class, p) = error(&s, "SELECT time_bucket(INTERVAL '1' HOUR)");
    assert_eq!(class, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    assert_eq!(param(&p, "expectedNum"), "[2, 3]");
}

#[test]
fn a_session_zone_changed_by_set_applies_to_the_bucket() {
    // The golden runner starts in America/Los_Angeles (Spark's test default) and the file sets UTC first.
    let s = session("America/Los_Angeles");
    block_on(async { s.sql("SET TIME ZONE 'UTC'").await?.collect().await }).unwrap();
    assert_eq!(
        one(
            &s,
            "SELECT time_bucket(INTERVAL '15' MINUTE, TIMESTAMP '2024-01-01 11:27:00')"
        ),
        "2024-01-01 11:15:00"
    );
}
