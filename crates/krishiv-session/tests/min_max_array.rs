// SPDX-License-Identifier: Apache-2.0
//! `min`/`max` over an array column: Spark orders arrays element by element, a NULL element
//! before any value, NaN after every number, a shorter array before a longer one it starts.
//! The expected rows follow Spark's array ordering rules (`interpretedOrdering`); no Spark run
//! backs them yet.
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
fn an_array_with_an_element_the_comparator_does_not_take_is_refused_by_name() {
    // Spark orders these (year-month and day-time intervals are atomic types); the engine's
    // comparator does not take them yet, so the refusal is explicit and names the SQL type.
    let s = Session::new().unwrap();
    for v in [
        "array(INTERVAL '1' YEAR)",
        "array(INTERVAL '1' DAY)",
        "array(struct(INTERVAL '1' YEAR))",
    ] {
        let err = match block_on(s.sql(&format!("SELECT min(a) FROM VALUES ({v}) t(a)"))) {
            Err(e) => e.to_string(),
            Ok(df) => block_on(df.collect()).unwrap_err().to_string(),
        };
        assert!(err.contains("KRISHIV_UNSUPPORTED_FEATURE"), "{v}: {err}");
        assert!(
            !err.contains("Interval("),
            "{v}: a raw Arrow type in the message: {err}"
        );
    }
}
