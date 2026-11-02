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

#[test]
fn k_is_validated_as_the_int_it_casts_to() {
    let s = Session::new().unwrap();
    for k in [
        "CAST(100001 AS DOUBLE)",
        "0.5",
        "'0'",
        "CAST(100001 AS DECIMAL(10, 2))",
    ] {
        let df = block_on(s.sql(&format!("SELECT max_by('a', 1, {k})")));
        let err = match df {
            Ok(df) => block_on(df.collect()).expect_err(k),
            Err(e) => e,
        };
        assert!(err.to_string().contains("VALUE_OUT_OF_RANGE"), "{k}: {err}");
    }
    assert_eq!(text("SELECT max_by('a', 1, CAST(1.9 AS DOUBLE))"), "[a]");
}

#[test]
fn a_merge_may_put_nulls_in_a_non_null_seed_collection() {
    assert_eq!(
        text("aggregate(array(1), array(0), (acc, x) -> array(CAST(NULL AS INT)), acc -> 42)"),
        "42"
    );
    assert_eq!(
        text(
            "aggregate(array(1), map('a', 1), (acc, x) -> map('a', CAST(NULL AS INT)), acc -> 42)"
        ),
        "42"
    );
}

#[test]
fn a_count_that_is_not_a_constant_is_refused_as_spark_does() {
    let s = Session::new().unwrap();
    // Spark 4.2.0: DATATYPE_MISMATCH.NON_FOLDABLE_INPUT, "the input k should be a foldable int
    // expression; however, got t.k" (a column, also under GROUP BY; Krishiv's error text does not
    // print the table qualifier) and "got CAST(rand() AS INT)".
    for (sql, got) in [
        (
            "SELECT k, max_by(x, y, k) FROM VALUES ('a', 1, 0), ('b', 2, 0) AS t(x, y, k) GROUP BY k",
            "k",
        ),
        (
            "SELECT min_by(x, y, k) FROM VALUES ('a', 1, 2) AS t(x, y, k)",
            "k",
        ),
        (
            "SELECT max_by(x, y, rand()) FROM VALUES ('a', 1, 2) AS t(x, y, k)",
            "CAST(rand() AS INT)",
        ),
    ] {
        let err = match block_on(s.sql(sql)) {
            Ok(df) => block_on(df.collect()).expect_err(sql),
            Err(e) => e,
        };
        let text = err.to_string();
        assert!(text.contains("NON_FOLDABLE_INPUT"), "{sql}: {text}");
        assert!(
            text.contains(&format!(
                "the input k should be a foldable int expression; however, got {got}."
            )),
            "{sql}: {text}"
        );
    }
}
