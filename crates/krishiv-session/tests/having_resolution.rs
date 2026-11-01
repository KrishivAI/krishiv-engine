// SPDX-License-Identifier: Apache-2.0
//! `HAVING` names: a grouping column wins over a select-list alias of the same name (Spark 4.2,
//! `having.sql`), while an alias nothing else holds is still the alias.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_session::Session;

/// The first column of `sql`, as numbers, sorted.
fn firsts(s: &Session, sql: &str) -> Vec<i64> {
    let df = block_on(s.sql(sql)).unwrap();
    let mut out = Vec::new();
    for b in block_on(df.collect()).unwrap() {
        out.extend(
            b.column(0)
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .copied(),
        );
    }
    out.sort_unstable();
    out
}

const T: &str = "FROM VALUES (1, 10), (2, 20) AS T(a, b)";

#[test]
fn a_grouping_column_wins_over_an_alias() {
    let s = Session::new().unwrap();
    assert_eq!(
        firsts(
            &s,
            &format!("SELECT SUM(a) AS b {T} GROUP BY b HAVING b > 10")
        ),
        [2]
    );
    assert_eq!(
        firsts(
            &s,
            &format!("SELECT SUM(a) AS b {T} GROUP BY GROUPING SETS ((b), (a, b)) HAVING b > 10")
        ),
        [2, 2]
    );
    assert_eq!(
        firsts(
            &s,
            &format!("SELECT SUM(a) AS b {T} GROUP BY ROLLUP(a, b) HAVING b > 10")
        ),
        [2]
    );
}

#[test]
fn an_alias_nothing_else_holds_is_the_alias() {
    let s = Session::new().unwrap();
    assert_eq!(
        firsts(
            &s,
            &format!("SELECT SUM(a) AS s {T} GROUP BY b HAVING s > 1")
        ),
        [2]
    );
}
