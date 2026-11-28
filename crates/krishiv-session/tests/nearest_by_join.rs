// SPDX-License-Identifier: Apache-2.0
//! Spark 4.3's `JOIN ... (APPROX | EXACT) NEAREST [k] BY (DISTANCE | SIMILARITY) ranking` end to
//! end: the k nearest right rows per left row, both directions, `LEFT OUTER` with and without a
//! match, rows with a NULL ranking, duplicate left rows, and the schema (no internal columns).
//! The Spark goldens are `join-nearest-by.sql`; these cases pin the rewrite's corners.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE TEMPORARY VIEW u AS SELECT * FROM VALUES (1, 10), (2, 20), (2, 20), (3, 30) AS u(id, s)",
        "CREATE TEMPORARY VIEW p AS SELECT * FROM VALUES ('A', 11), ('B', 22), ('C', 5), ('D', NULL) AS p(name, s)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await })
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    s
}

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        other => format!("<{other}>"),
    }
}

fn run(s: &Session, sql: &str) -> (Vec<String>, Vec<String>) {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let names = batches
        .first()
        .map(|b| {
            b.schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        })
        .unwrap_or_default();
    let mut out: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(|r| {
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), r))
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect();
    out.sort();
    (names, out)
}

#[test]
fn the_nearest_k_by_distance_and_by_similarity() {
    let s = session();
    let (_, d) = run(
        &s,
        "SELECT u.id, p.name FROM u JOIN p APPROX NEAREST 2 BY DISTANCE abs(u.s - p.s) WHERE u.id = 1",
    );
    assert_eq!(d, ["1|A", "1|C"]);
    // SIMILARITY ranks the largest first: the two largest sums.
    let (_, m) = run(
        &s,
        "SELECT u.id, p.name FROM u JOIN p EXACT NEAREST 2 BY SIMILARITY u.s + p.s WHERE u.id = 3",
    );
    assert_eq!(m, ["3|A", "3|B"]);
}

#[test]
fn k_defaults_to_one_and_a_repeated_left_row_gets_its_own_matches() {
    let s = session();
    let (_, rows) = run(
        &s,
        "SELECT u.id, p.name FROM u JOIN p EXACT NEAREST BY DISTANCE abs(u.s - p.s)",
    );
    assert_eq!(rows, ["1|A", "2|B", "2|B", "3|B"]);
}

#[test]
fn a_null_ranking_never_matches_and_a_left_join_keeps_a_row_without_matches() {
    let s = session();
    // k above the number of right rows: every right row but the one ranked NULL (Spark skips those).
    let (_, rows) = run(
        &s,
        "SELECT p.name FROM u JOIN p APPROX NEAREST 9 BY DISTANCE abs(u.s - p.s) WHERE u.id = 1",
    );
    assert_eq!(rows, ["A", "B", "C"]);
    let (_, left) = run(
        &s,
        "SELECT u.id, q.name FROM u LEFT OUTER JOIN (SELECT * FROM p WHERE false) q \
         APPROX NEAREST 1 BY DISTANCE abs(u.s - q.s) WHERE u.id = 3",
    );
    assert_eq!(left, ["3|NULL"]);
    // Every candidate ranked NULL: an inner join drops the left row, a left outer join pads it.
    let (_, inner) = run(
        &s,
        "SELECT u.id, q.name FROM u JOIN (SELECT * FROM p WHERE name = 'D') q \
         APPROX NEAREST 1 BY DISTANCE abs(u.s - q.s)",
    );
    assert!(inner.is_empty(), "{inner:?}");
    let (_, outer) = run(
        &s,
        "SELECT u.id, q.name FROM u LEFT OUTER JOIN (SELECT * FROM p WHERE name = 'D') q \
         APPROX NEAREST 1 BY DISTANCE abs(u.s - q.s) WHERE u.id = 3",
    );
    assert_eq!(outer, ["3|NULL"]);
}

#[test]
fn select_star_has_only_the_two_sides_columns() {
    let s = session();
    let (names, _) = run(
        &s,
        "SELECT * FROM u JOIN p APPROX NEAREST 1 BY DISTANCE abs(u.s - p.s)",
    );
    assert_eq!(names, ["id", "s", "name", "s"]);
}

fn error_of(s: &Session, sql: &str) -> String {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql} should fail"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn a_ranking_that_does_not_order_is_the_joins_own_error() {
    let s = session();
    let msg = error_of(
        &s,
        "SELECT u.id FROM u JOIN p APPROX NEAREST 1 BY SIMILARITY map(u.s, p.s)",
    );
    assert!(
        msg.contains("NEAREST_BY_JOIN.NON_ORDERABLE_RANKING_EXPRESSION"),
        "{msg}"
    );
    assert!(msg.contains("MAP<INT, INT>"), "{msg}");
}

#[test]
fn nearest_by_needs_cross_joins_enabled() {
    let s = session();
    block_on(s.sql("SET spark.sql.crossJoin.enabled = false")).unwrap();
    let msg = error_of(
        &s,
        "SELECT u.id FROM u JOIN p APPROX NEAREST 1 BY DISTANCE abs(u.s - p.s)",
    );
    assert!(
        msg.contains("NEAREST_BY_JOIN.CROSS_JOIN_NOT_ENABLED"),
        "{msg}"
    );
    block_on(s.sql("SET spark.sql.crossJoin.enabled = true")).unwrap();
    let (_, rows) = run(
        &s,
        "SELECT u.id FROM u JOIN p APPROX NEAREST 1 BY DISTANCE abs(u.s - p.s) WHERE u.id = 1",
    );
    assert_eq!(rows.len(), 1);
}

#[test]
fn a_window_ordered_by_a_map_is_invalid_ordering_type() {
    let s = session();
    let msg = error_of(
        &s,
        "SELECT row_number() OVER (ORDER BY map(u.s, u.s)) FROM u",
    );
    assert!(msg.contains("INVALID_ORDERING_TYPE"), "{msg}");
    assert!(msg.contains("sortorder"), "{msg}");
}
