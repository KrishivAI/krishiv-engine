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
        "create view pv as select 1 as a",
        "create global temp view gv as select 1 as a",
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
    // A permanent view is a `V1Table` to Spark, which refuses it by its qualified name.
    assert_eq!(
        error(&s, "select * from pv@v1"),
        travel("`spark_catalog`.`default`.`pv`")
    );
    // The planning entry points (Flight SQL's `GetFlightInfo` reads `statement_info`) name it too.
    let e = block_on(s.statement_info("select * from tbl@v1")).unwrap_err();
    assert_eq!(e.error_class().name(), "UNSUPPORTED_FEATURE.TIME_TRAVEL");
    // A global temporary view and a CTE are named as written.
    assert_eq!(
        error(&s, "select * from global_temp.gv@v1"),
        travel("`global_temp`.`gv`")
    );
    assert_eq!(
        error(&s, "with c as (select 1) select * from c@v1"),
        travel("`c`")
    );
    assert_eq!(
        error(
            &s,
            "with c as (select 1), d as (select * from c version as of 1) select * from d"
        ),
        travel("`c`")
    );
    // The other statements that lower a query give the same errors.
    assert_eq!(
        error(&s, "create table x2 as select * from tbl@v1"),
        travel("`spark_catalog`.`default`.`tbl`")
    );
    assert_eq!(
        error(&s, "insert into tbl select * from tbl@v1"),
        travel("`spark_catalog`.`default`.`tbl`")
    );
    assert_eq!(
        error(&s, "create view v9 as select * from tbl@v1"),
        travel("`spark_catalog`.`default`.`tbl`")
    );
    assert_eq!(
        error(&s, "insert into tbl select * from nope@v1").0,
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    assert_eq!(
        error(&s, "explain select * from tbl@v1"),
        travel("`spark_catalog`.`default`.`tbl`")
    );
    // An embedded backtick is doubled in the name as written.
    block_on(async {
        s.sql("create temp view `a``b` as select 1")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert_eq!(error(&s, "select * from `a``b`@v1"), travel("`a``b`"));
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

/// A session with an access policy other than `AllowAll` gives one error for an existing and a missing relation:
/// the catalog is not read before the policy is asked, so a denied principal learns nothing about names.
#[test]
fn time_travel_under_a_policy_is_no_existence_oracle() {
    use krishiv_session::{AllowAll, ReadOnlyPolicy};
    use std::sync::Arc;
    let s = Session::builder()
        .access_policy(Arc::new(ReadOnlyPolicy::new(Arc::new(AllowAll))))
        .build()
        .unwrap();
    assert_eq!(
        error(&s, "select * from nope@v1"),
        error(&s, "select * from nope2 version as of 1")
    );
    assert_eq!(
        error(&s, "select * from nope@v1").0,
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

/// An existing table answers as a missing one does for a session of a principal (`new_session_as`,
/// which never reads the catalog for the error), so the error is no existence oracle.
#[test]
fn an_existing_table_answers_as_a_missing_one_for_a_principal() {
    use krishiv_common::principal::{Principal, PrincipalKind};
    use krishiv_session::AllowAll;
    use std::sync::Arc;
    let base = Session::new().unwrap();
    block_on(async {
        base.sql("create table tbl (id int) using parquet")
            .await?
            .collect()
            .await
    })
    .unwrap();
    // The open session names the relation; the principal's session must not.
    assert_eq!(
        error(&base, "select * from tbl@v1").0,
        "UNSUPPORTED_FEATURE.TIME_TRAVEL"
    );
    let s = base.new_session_as(Principal::new("u", PrincipalKind::User), Arc::new(AllowAll));
    let existing = error(&s, "select * from tbl@v1");
    assert_eq!(existing, error(&s, "select * from nope@v1"));
    assert_eq!(existing.0, "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn a_non_ascii_word_after_the_at_sign_is_a_syntax_error() {
    let s = Session::new().unwrap();
    assert_eq!(class(&s, "select * from tbl@é"), "PARSE_SYNTAX_ERROR");
    assert_eq!(class(&s, "select * from tbl@vé"), "PARSE_SYNTAX_ERROR");
}
