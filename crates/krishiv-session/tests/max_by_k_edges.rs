// SPDX-License-Identifier: Apache-2.0
//! Edges of `max_by(x, y, k)` and `aggregate` merge casts found in review.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn text(sql: &str) -> String {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(&format!("SELECT CAST(({sql}) AS STRING)"))).unwrap();
    let b = block_on(df.collect()).unwrap();
    arrow_array::cast::AsArray::as_string_view(b[0].column(0))
        .iter()
        .map(|v| v.unwrap_or("NULL").to_owned())
        .collect::<Vec<_>>()
        .join("|")
}

#[test]
fn the_value_may_hold_a_map_because_only_the_ordering_argument_orders() {
    assert_eq!(text("SELECT max_by(map('a', 1), 10, 1)"), "[{a -> 1}]");
    assert_eq!(
        text(
            "SELECT min_by(m, y, 2) FROM VALUES (map('a', 1), 3), (map('b', 2), 1), (map('c', 3), 2) AS t(m, y)"
        ),
        "[{b -> 2}, {c -> 3}]"
    );
    assert_eq!(
        text(
            "SELECT max_by(m, y, 2) FROM VALUES (map('a', 1), 3), (map('b', 2), 1), (map('c', 3), 2) AS t(m, y)"
        ),
        "[{a -> 1}, {c -> 3}]"
    );
}

#[test]
fn a_nullable_merge_field_over_a_non_null_seed_field_keeps_its_nulls() {
    assert_eq!(
        text(
            "aggregate(array(1, CAST(NULL AS INT)), named_struct('s', 0), (acc, x) -> named_struct('s', x), acc -> 42)"
        ),
        "42"
    );
}
