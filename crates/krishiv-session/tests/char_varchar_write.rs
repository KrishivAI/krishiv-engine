// SPDX-License-Identifier: Apache-2.0
//! The write side of `CHAR(n)` / `VARCHAR(n)` columns: trailing spaces beyond the limit are
//! trimmed, anything else longer fails with `EXCEED_LIMIT_LENGTH`
//! (Spark 4.2.0 `CharVarcharUtils.stringLengthCheck`); the legacy flag turns the check off.
#![expect(clippy::unwrap_used, reason = "test code")]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Result<(), krishiv_common::error::KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await }).map(|_| ())
}

fn lengths(s: &Session) -> Vec<String> {
    let b = block_on(async {
        s.sql("select c, v from t order by 1")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let mut out = Vec::new();
    for batch in &b {
        let (c, v) = (batch.column(0), batch.column(1));
        let text = |a: &arrow_array::ArrayRef, i: usize| {
            a.as_string_opt::<i32>().map_or_else(
                || a.as_string_view().value(i).to_owned(),
                |s| s.value(i).to_owned(),
            )
        };
        for i in 0..batch.num_rows() {
            out.push(format!("{}|{}", text(c, i), text(v, i)));
        }
    }
    out
}

fn fresh() -> Session {
    let s = Session::new().unwrap();
    run(&s, "drop table if exists t").unwrap();
    run(&s, "create table t(c char(3), v varchar(2)) using parquet").unwrap();
    s
}

#[test]
fn trailing_spaces_over_the_limit_are_trimmed_and_other_overflow_fails() {
    let s = fresh();
    run(&s, "insert into t values ('ab   ', 'x   ')").unwrap();
    // The char column reads back padded to its length; the varchar column is cut to its limit (two characters), not padded.
    assert_eq!(lengths(&s), vec!["ab |x "]);
    let e = run(&s, "insert into t values ('abcd', 'x')").unwrap_err();
    assert_eq!(e.error_class().name(), "EXCEED_LIMIT_LENGTH");
    let e = run(&s, "insert into t values ('abc', 'xyz')").unwrap_err();
    assert_eq!(e.error_class().name(), "EXCEED_LIMIT_LENGTH");
}

#[test]
fn the_legacy_flag_turns_the_check_off() {
    let s = fresh();
    run(&s, "set spark.sql.legacy.charVarcharAsString=true").unwrap();
    run(&s, "insert into t values ('abcdef', 'xyz')").unwrap();
    assert_eq!(lengths(&s), vec!["abcdef|xyz"]);
}
