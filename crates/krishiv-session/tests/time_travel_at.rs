// SPDX-License-Identifier: Apache-2.0
//! The `@` time-travel suffix of a table name (Spark 4.3) and the clause forms: the error class of each statement
//! is what Spark 4.3.0 raised for it (real run). A table of the session catalog does not support time travel.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn class(s: &Session, sql: &str) -> String {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => "ok".to_owned(),
        Err(e) => e.error_class().name().to_owned(),
    }
}

#[test]
fn at_suffix_and_clauses_match_spark() {
    let s = Session::new().unwrap();
    block_on(async {
        s.sql("create table tbl (id int, name string) using parquet")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let unsupported = "UNSUPPORTED_FEATURE.TIME_TRAVEL";
    for (sql, want) in [
        ("select * from tbl version as of 1", unsupported),
        (
            "select * from tbl timestamp as of '2020-01-01'",
            unsupported,
        ),
        ("select * from tbl@v1", unsupported),
        ("select * from tbl@V1", unsupported),
        ("select * from tbl @v1", unsupported),
        ("select * from tbl@20240101000000000", unsupported),
        ("select * from tbl@v1 as t", unsupported),
        ("select * from tbl@v1 with ('k'='v')", unsupported),
        (
            "select * from tbl@v1 version as of 1",
            "MULTIPLE_TIME_TRAVEL_SPEC",
        ),
        (
            "select * from tbl@v1 timestamp as of '2020-01-01'",
            "MULTIPLE_TIME_TRAVEL_SPEC",
        ),
        (
            "select * from tbl@2024",
            "INVALID_TIME_TRAVEL_TIMESTAMP_FORMAT",
        ),
        ("select * from tbl@abc", "PARSE_SYNTAX_ERROR"),
        ("select * from tbl@ v1", "PARSE_SYNTAX_ERROR"),
        ("select * from tbl@v1@v2", "PARSE_SYNTAX_ERROR"),
        (
            "select * from tbl version as of 1 timestamp as of '2020-01-01'",
            "PARSE_SYNTAX_ERROR",
        ),
    ] {
        assert_eq!(class(&s, sql), want, "{sql}");
    }
}
