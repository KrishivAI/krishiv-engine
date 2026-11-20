// SPDX-License-Identifier: Apache-2.0
//! The `@` time travel suffix of a table name (Spark 4.3, `spark.sql.timeTravel.atSyntax.enabled`) and
//! `COMMENT ON COLUMN`, beside the cases `spark43_coverage.rs` takes from a real Spark 4.3.0 run.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn run(session: &Session, sql: &str) -> Result<(), String> {
    block_on(async { session.sql(sql).await?.collect().await })
        .map(|_| ())
        .map_err(|e| e.error_class().name().to_owned())
}

fn session_with_table() -> Session {
    let s = Session::new().expect("session");
    run(&s, "create table tbl (id int, name string) using parquet").expect("create");
    s
}

#[test]
fn a_suffix_reads_like_the_clause_it_stands_for() {
    let s = session_with_table();
    for sql in [
        "select * from tbl@v1",
        "select * from tbl@V12",
        "select * from tbl@20240101000000000",
        "select * from tbl version as of 1",
    ] {
        assert_eq!(
            run(&s, sql),
            Err("UNSUPPORTED_FEATURE.TIME_TRAVEL".to_owned()),
            "{sql}"
        );
    }
}

#[test]
fn a_suffix_and_a_clause_are_one_too_many() {
    let s = session_with_table();
    assert_eq!(
        run(&s, "select * from tbl@v1 timestamp as of '2024-01-01'"),
        Err("MULTIPLE_TIME_TRAVEL_SPEC".to_owned())
    );
}

#[test]
fn a_malformed_suffix_is_a_syntax_error() {
    let s = session_with_table();
    for sql in [
        "select * from tbl@",
        "select * from tbl@x1",
        "select * from tbl@v",
        "select * from tbl@2024",
        "select * from tbl@ v1",
    ] {
        assert_eq!(run(&s, sql), Err("PARSE_SYNTAX_ERROR".to_owned()), "{sql}");
    }
}

#[test]
fn the_suffix_is_off_with_its_config() {
    let s = session_with_table();
    run(&s, "set spark.sql.timeTravel.atSyntax.enabled=false").expect("set");
    assert_eq!(
        run(&s, "select * from tbl@v1"),
        Err("UNSUPPORTED_FEATURE.TIME_TRAVEL_AT_SYNTAX".to_owned())
    );
    // The clause is not the suffix: it still parses.
    assert_eq!(
        run(&s, "select * from tbl version as of 1"),
        Err("UNSUPPORTED_FEATURE.TIME_TRAVEL".to_owned())
    );
}

#[test]
fn comment_on_column_sets_and_clears_the_comment() {
    let s = session_with_table();
    run(&s, "comment on column tbl.name is 'the name'").expect("comment");
    let rows = block_on(async { s.sql("describe tbl").await?.collect().await }).expect("describe");
    let text = format!("{rows:?}");
    assert!(text.contains("the name"), "{text}");
    // `IS NULL` is Spark's empty comment; a bare name has no table.
    run(&s, "comment on column tbl.name is null").expect("clear");
    assert!(run(&s, "comment on column name is 'x'").is_err());
    assert_eq!(
        run(&s, "comment on column tbl.nope is 'x'"),
        Err("UNRESOLVED_COLUMN.WITH_SUGGESTION".to_owned())
    );
}

#[test]
fn replace_where_and_target_options_are_refused_by_the_table() {
    let s = session_with_table();
    for sql in [
        "insert into tbl (id, name) replace where id = 1 values (1, 'a')",
        "insert into tbl replace on tbl.id = 1 values (1, 'a')",
        "update tbl with ('k' = 'v') set name = 'x'",
        "delete from tbl with ('k' = 'v')",
    ] {
        assert_eq!(
            run(&s, sql),
            Err("UNSUPPORTED_FEATURE.TABLE_OPERATION".to_owned()),
            "{sql}"
        );
    }
    // A table that does not exist is the first error.
    assert_eq!(
        run(&s, "delete from nope with ('k' = 'v')"),
        Err("TABLE_OR_VIEW_NOT_FOUND".to_owned())
    );
}
