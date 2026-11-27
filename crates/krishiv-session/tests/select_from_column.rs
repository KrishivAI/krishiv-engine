// SPDX-License-Identifier: Apache-2.0
//! `SELECT FROM t` outside ANSI mode: `FROM` is the first column (aliased `t`),
//! so the analyzer reports it as an unresolved column, as Spark does.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

#[test]
fn a_from_that_starts_the_select_list_is_an_unresolved_column() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TEMPORARY VIEW t AS SELECT 1 AS key, 'x' AS data")).unwrap();
    let err = match block_on(s.sql("SELECT FROM t GROUP BY key HAVING length(data) > 0")) {
        Ok(df) => block_on(df.collect()).expect_err("must not run"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("UNRESOLVED_COLUMN"), "{err}");
}
