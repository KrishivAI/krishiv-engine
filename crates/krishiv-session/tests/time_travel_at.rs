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

/// The class and the message (without the SQLSTATE and position suffix) of a failing statement.
fn error(s: &Session, sql: &str) -> (String, String) {
    let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
    (e.error_class().name().to_owned(), e.message().to_owned())
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
        // The digits must make a real timestamp; the whitespace and string forms Spark's lexer allows.
        (
            "select * from tbl@20241301000000000",
            "INVALID_TIME_TRAVEL_TIMESTAMP_FORMAT",
        ),
        (
            "select * from tbl@20230229000000000",
            "INVALID_TIME_TRAVEL_TIMESTAMP_FORMAT",
        ),
        (
            "select * from tbl@20240101250000000",
            "INVALID_TIME_TRAVEL_TIMESTAMP_FORMAT",
        ),
        (
            "select * from tbl@20240101006000000",
            "INVALID_TIME_TRAVEL_TIMESTAMP_FORMAT",
        ),
        ("select * from tbl@20240229000000000", unsupported),
        ("select * from tbl@00000101000000000", unsupported),
        ("select * from tbl@v 1", unsupported),
        ("select * from tbl@ 20240101000000000", unsupported),
        ("select * from tbl@v'1'", unsupported),
        ("select * from tbl@v 'x'", unsupported),
        ("select * from tbl@v", "PARSE_SYNTAX_ERROR"),
        ("select * from tbl@'1'", "PARSE_SYNTAX_ERROR"),
        ("select * from nope@v1", "TABLE_OR_VIEW_NOT_FOUND"),
        (
            "select * from nope version as of 1",
            "TABLE_OR_VIEW_NOT_FOUND",
        ),
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

#[test]
fn time_travel_names_the_resolved_relation() {
    let s = Session::new().unwrap();
    for q in [
        "create table tbl (id int) using parquet",
        "create database db2",
        "create table db2.t2 (id int) using parquet",
        "create temp view tv as select 1",
    ] {
        block_on(async { s.sql(q).await?.collect().await }).unwrap();
    }
    let travel = |relation: &str| {
        (
            "UNSUPPORTED_FEATURE.TIME_TRAVEL".to_owned(),
            format!("The feature is not supported: Time travel on the relation: {relation}."),
        )
    };
    // A temporary view is named as written.
    assert_eq!(error(&s, "select * from tv@v1"), travel("`tv`"));
    assert_eq!(
        error(&s, "select * from tv version as of 1"),
        travel("`tv`")
    );
    assert_eq!(
        error(&s, "select * from db2.t2@v1"),
        travel("`spark_catalog`.`db2`.`t2`")
    );
    assert_eq!(
        error(&s, "select * from spark_catalog.db2.t2@v1"),
        travel("`spark_catalog`.`db2`.`t2`")
    );
    // The current namespace qualifies a bare name.
    block_on(async { s.sql("use db2").await?.collect().await }).unwrap();
    assert_eq!(
        error(&s, "select * from t2@v1"),
        travel("`spark_catalog`.`db2`.`t2`")
    );
    assert_eq!(
        error(&s, "select * from default.tbl@v1"),
        travel("`spark_catalog`.`default`.`tbl`")
    );
}
