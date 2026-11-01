// SPDX-License-Identifier: Apache-2.0
//! `any_value` over strings in a window frame picks the frame's first row,
//! as `first` does.
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn column(sql: &str) -> Vec<Option<String>> {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(sql)).unwrap();
    let b = block_on(df.collect()).unwrap();
    let mut out = Vec::new();
    for batch in &b {
        let c = batch.column(0).as_string_view();
        out.extend(c.iter().map(|v| v.map(str::to_owned)));
    }
    out
}

#[test]
fn any_value_of_strings_follows_first_over_a_frame() {
    let t = "FROM VALUES (1, NULL), (2, 'b'), (3, 'c') AS t(id, s)";
    let frame = "ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW";
    for ignore in ["", ", true"] {
        let any = column(&format!("SELECT any_value(s{ignore}) OVER ({frame}) {t}"));
        let first = column(&format!(
            "SELECT first(s{}) OVER ({frame}) {t}",
            if ignore.is_empty() { "" } else { ", true" }
        ));
        assert_eq!(any, first, "ignore nulls: {ignore:?}");
    }
}
