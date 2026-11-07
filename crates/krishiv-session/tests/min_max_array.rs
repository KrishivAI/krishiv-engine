// SPDX-License-Identifier: Apache-2.0
//! `min`/`max` over an array column: Spark orders arrays element by element, a NULL element
//! before any value, NaN after every number, a shorter array before a longer one it starts.
//! The group, merge and spill paths must agree with the in-memory run.
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;

/// The first column of `sql` rendered as `[a, b]` with `NULL` for a missing cell.
fn lists(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    let mut out = Vec::new();
    for batch in block_on(df.collect()).unwrap() {
        let col = batch.column(0);
        for row in 0..batch.num_rows() {
            if col.is_null(row) {
                out.push("NULL".to_owned());
                continue;
            }
            let items = col.as_list::<i32>().value(row);
            let cells: Vec<String> = (0..items.len())
                .map(|i| {
                    if items.is_null(i) {
                        "NULL".to_owned()
                    } else {
                        items.as_primitive::<Int32Type>().value(i).to_string()
                    }
                })
                .collect();
            out.push(format!("[{}]", cells.join(", ")));
        }
    }
    out
}

const ROWS: &str = "VALUES (1, array(1, 2)), (1, array(1, 2, 0)), (1, array(1, NULL)), \
    (2, array()), (2, array(0)), (3, cast(NULL as array<int>)), (3, array(NULL)), \
    (4, array(2)), (4, array(10)), (4, array(1, 99)) t(g, a)";

#[test]
fn min_and_max_order_arrays_element_by_element() {
    let s = Session::new().unwrap();
    for (f, want) in [
        ("min", ["[1, NULL]", "[]", "[NULL]", "[1, 99]"]),
        ("max", ["[1, 2, 0]", "[0]", "[NULL]", "[10]"]),
    ] {
        let got = lists(
            &s,
            &format!("SELECT {f}(a) FROM ({ROWS}) GROUP BY g ORDER BY g"),
        );
        assert_eq!(got, want, "{f} per group");
    }
    assert_eq!(lists(&s, &format!("SELECT min(a) FROM ({ROWS})")), ["[]"]);
    assert_eq!(lists(&s, &format!("SELECT max(a) FROM ({ROWS})")), ["[10]"]);
}

#[test]
fn min_and_max_of_an_empty_or_all_null_input_are_null() {
    let s = Session::new().unwrap();
    for f in ["min", "max"] {
        let got = lists(
            &s,
            &format!("SELECT {f}(a) FROM (SELECT cast(NULL as array<int>) AS a) WHERE a IS NULL"),
        );
        assert_eq!(got, ["NULL"], "{f}");
        let got = lists(&s, &format!("SELECT {f}(a) FROM ({ROWS}) WHERE g = -1"));
        assert_eq!(got, ["NULL"], "{f} over no rows");
    }
}

#[test]
fn an_array_of_intervals_is_refused_in_sql_types() {
    let s = Session::new().unwrap();
    let err = block_on(s.sql("SELECT min(a) FROM VALUES (array(INTERVAL '1' YEAR)) t(a)"))
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("KRISHIV_UNSUPPORTED_FEATURE") || err.contains("not supported"),
        "{err}"
    );
    assert!(
        !err.contains("Interval("),
        "a raw Arrow type in the message: {err}"
    );
}
