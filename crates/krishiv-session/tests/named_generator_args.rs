// SPDX-License-Identifier: Apache-2.0
//! A generator in `FROM` takes its one parameter by name (`explode(collection => ...)`,
//! `inline(input => ...)`), as Spark 4.2.0's `named-function-arguments.sql` golden does.
#![cfg(test)]

use arrow_array::RecordBatch;
use futures::executor::block_on;
use krishiv_session::Session;

fn rows(s: &Session, sql: &str) -> usize {
    let out = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    out.iter().map(RecordBatch::num_rows).sum()
}

#[test]
fn a_named_argument_names_the_generators_one_parameter() {
    let s = Session::new().unwrap();
    for (sql, n) in [
        ("SELECT * FROM explode(collection => array(1, 2))", 2),
        (
            "SELECT * FROM explode_outer(collection => map('a', 1, 'b', 2))",
            2,
        ),
        ("SELECT * FROM posexplode(collection => array(1, 2))", 2),
        (
            "SELECT * FROM posexplode_outer(collection => map('a', 1, 'b', 2))",
            2,
        ),
        (
            "SELECT * FROM inline(input => array(struct(1, 'a'), struct(2, 'b')))",
            2,
        ),
        (
            "SELECT * FROM inline_outer(input => array(struct(1, 'a'), struct(2, 'b')))",
            2,
        ),
    ] {
        assert_eq!(rows(&s, sql), n, "{sql}");
    }
}

#[test]
fn another_name_is_still_refused() {
    let s = Session::new().unwrap();
    let r = block_on(async {
        s.sql("SELECT * FROM explode(input => array(1))")
            .await?
            .collect()
            .await
    });
    assert!(r.is_err());
}
