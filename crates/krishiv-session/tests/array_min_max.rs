// SPDX-License-Identifier: Apache-2.0
//! `min` / `max` over arrays and `<=>` / `=` / `<` over structs holding arrays, in Spark's order:
//! element by element, a NULL element before every value, a proper prefix before its extension,
//! a NULL array ignored by the aggregates.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn rows(sql: &str) -> Vec<Vec<String>> {
    let s = Session::new().unwrap();
    let batches = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(r) {
                            "NULL".to_owned()
                        } else if let Some(v) = c.as_string_opt::<i32>() {
                            v.value(r).to_owned()
                        } else {
                            c.as_string_view().value(r).to_owned()
                        }
                    })
                    .collect(),
            );
        }
    }
    out
}

fn one(sql: &str) -> Vec<String> {
    rows(sql).remove(0)
}

const ARRAYS: &str = "VALUES (array(1, 2)), (array(1)), (array(2)), (array(1, 3)), \
                      (CAST(NULL AS ARRAY<INT>)) AS t(a)";

#[test]
fn min_and_max_of_arrays_follow_sparks_order() {
    assert_eq!(
        one(&format!(
            "SELECT CAST(min(a) AS STRING), CAST(max(a) AS STRING) FROM ({ARRAYS})"
        )),
        ["[1]", "[2]"],
        "[1] < [1, 2] < [1, 3] < [2]; the NULL array is ignored"
    );
    // A NULL element sorts before every value.
    assert_eq!(
        one(
            "SELECT CAST(min(a) AS STRING), CAST(max(a) AS STRING) FROM \
             (VALUES (array(1)), (array(CAST(NULL AS INT), 5)), (array(0)) AS t(a))"
        ),
        ["[null, 5]", "[1]"]
    );
    // Nothing but NULL arrays: NULL.
    assert_eq!(
        one("SELECT CAST(max(a) AS STRING) FROM (VALUES (CAST(NULL AS ARRAY<INT>)) AS t(a))"),
        ["NULL"]
    );
}

#[test]
fn array_extrema_by_group_and_over_other_element_types() {
    let got = rows(
        "SELECT CAST(k AS STRING), CAST(max(a) AS STRING) FROM \
         (VALUES (1, array('b')), (1, array('a', 'z')), (2, array('a')), (2, array('a', 'b')) \
          AS t(k, a)) GROUP BY k ORDER BY k",
    );
    assert_eq!(
        got,
        vec![
            vec!["1".to_owned(), "[b]".to_owned()],
            vec!["2".to_owned(), "[a, b]".to_owned()],
        ]
    );
    // Arrays of structs order by their fields.
    assert_eq!(
        one("SELECT CAST(min(a) AS STRING) FROM \
             (VALUES (array(named_struct('x', 2))), (array(named_struct('x', 1), named_struct('x', 9))) AS t(a))"),
        ["[{1}, {9}]"]
    );
}

#[test]
fn structs_holding_arrays_compare() {
    let t = |sql: &str| one(&format!("SELECT CAST(({sql}) AS STRING)"))[0].clone();
    for (sql, want) in [
        (
            "named_struct('a', array(1, 2)) = named_struct('a', array(1, 2))",
            "true",
        ),
        (
            "named_struct('a', array(1, 2)) <=> named_struct('a', array(1, 2))",
            "true",
        ),
        (
            "named_struct('a', array(1, 2)) = named_struct('a', array(1, 3))",
            "false",
        ),
        (
            "named_struct('a', array(1)) < named_struct('a', array(1, 2))",
            "true",
        ),
        (
            "named_struct('a', array(2)) < named_struct('a', array(1, 2))",
            "false",
        ),
        (
            "named_struct('a', 1, 'b', array(2)) >= named_struct('a', 1, 'b', array(1))",
            "true",
        ),
    ] {
        assert_eq!(t(sql), want, "{sql}");
    }
}

/// Arrays inside structs compare whatever the element nullability (Spark 4.2.0: false, true, true).
#[test]
fn structs_holding_arrays_of_differing_element_nullability_compare() {
    let r = one(
        "SELECT cast(named_struct('a', array(1)) = named_struct('a', array(CAST(NULL AS INT))) AS STRING), \
                cast(named_struct('a', array(1, 2)) = named_struct('a', array(1, 2)) AS STRING), \
                cast(named_struct('a', array(1)) < named_struct('a', array(1, 2)) AS STRING)",
    );
    assert_eq!(r, ["false", "true", "true"]);
}
