// SPDX-License-Identifier: Apache-2.0
//! Spark 4.3's `ASOF JOIN ... MATCH_CONDITION (...)` end to end: the match directions, strict and
//! exact comparisons, left joins, NULL values and keys, a residual condition, no equality key, `USING`,
//! a spilled group, and the flag. The Spark goldens are the upstream `join-asof-*.sql` files (the
//! 4.3.0 golden profile); these cases pin the executor's paths.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    for sql in [
        "SET spark.sql.join.asofJoin.enabled=true",
        "CREATE TEMPORARY VIEW t AS SELECT * FROM VALUES \
         (1, 'a', 10), (2, 'a', 20), (3, 'a', 30), (4, 'b', 15), (5, 'c', 5), (6, 'a', NULL), \
         (7, NULL, 20) AS t(id, k, ts)",
        "CREATE TEMPORARY VIEW q AS SELECT * FROM VALUES \
         (100, 'a', 10), (101, 'a', 18), (102, 'a', 19), (103, 'a', 25), (104, 'b', 20), \
         (105, 'a', NULL), (106, NULL, 21) AS q(qid, k, ts)",
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
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        other => format!("<{other}>"),
    }
}

/// The rows (`a|b`) of `sql`, sorted.
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
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
    out
}

fn error(s: &Session, sql: &str) -> String {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql} ran"),
        Err(e) => e.error_class().name().to_owned(),
    }
}

fn join(s: &Session, kind: &str, cond: &str) -> Vec<String> {
    rows(
        s,
        &format!(
            "SELECT t.id, q.qid FROM t {kind} ASOF JOIN q MATCH_CONDITION ({cond}) ON t.k = q.k"
        ),
    )
}

#[test]
fn directions_and_exact_matches() {
    let s = session();
    // Backward: the greatest q.ts at most t.ts (no ties here: Spark leaves a tie's pick open).
    assert_eq!(join(&s, "", "t.ts >= q.ts"), ["1|100", "2|102", "3|103"]);
    // Strictly before.
    assert_eq!(join(&s, "", "t.ts > q.ts"), ["2|102", "3|103"]);
    // Forward: the least q.ts at least t.ts, the first of ties.
    assert_eq!(join(&s, "", "t.ts <= q.ts"), ["1|100", "2|103", "4|104"]);
    assert_eq!(join(&s, "", "t.ts < q.ts"), ["1|101", "2|103", "4|104"]);
    // The operands either way round: `q.ts <= t.ts` is `t.ts >= q.ts`.
    assert_eq!(join(&s, "", "q.ts <= t.ts"), join(&s, "", "t.ts >= q.ts"));
}

#[test]
fn left_joins_keep_unmatched_and_null_rows() {
    let s = session();
    assert_eq!(
        join(&s, "LEFT", "t.ts >= q.ts"),
        [
            "1|100", "2|102", "3|103", "4|NULL", "5|NULL", "6|NULL", "7|NULL"
        ]
    );
    assert_eq!(
        join(&s, "LEFT OUTER", "t.ts < q.ts"),
        [
            "1|101", "2|103", "3|NULL", "4|104", "5|NULL", "6|NULL", "7|NULL"
        ]
    );
}

#[test]
fn residual_filters_before_the_nearest_row() {
    let s = session();
    // q 102 and 103 fail the residual: t 2 and t 3 take 101.
    let sql = "SELECT t.id, q.qid FROM t ASOF JOIN q MATCH_CONDITION (t.ts >= q.ts) \
               ON t.k = q.k AND q.qid NOT IN (102, 103)";
    assert_eq!(rows(&s, sql), ["1|100", "2|101", "3|101"]);
    let sql = "SELECT t.id, q.qid FROM t LEFT ASOF JOIN q MATCH_CONDITION (t.ts <= q.ts) \
               ON t.k = q.k AND q.qid > t.id + 100";
    assert_eq!(
        rows(&s, sql),
        [
            "1|102", "2|103", "3|NULL", "4|NULL", "5|NULL", "6|NULL", "7|NULL"
        ]
    );
}

#[test]
fn no_key_and_using() {
    let s = session();
    // Every right row is a candidate.
    let sql =
        "SELECT t.id, q.qid FROM t ASOF JOIN q MATCH_CONDITION (t.ts >= q.ts) WHERE t.id <= 4";
    assert_eq!(rows(&s, sql), ["1|100", "2|104", "3|103", "4|100"]);
    // `USING (k)`: one `k` column, then the rest.
    let sql = "SELECT * FROM t ASOF JOIN q USING (k) MATCH_CONDITION (t.ts >= q.ts)";
    assert_eq!(error(&s, sql), "PARSE_SYNTAX_ERROR");
    let sql = "SELECT * FROM t ASOF JOIN q MATCH_CONDITION (t.ts >= q.ts) USING (k)";
    assert_eq!(
        rows(&s, sql),
        ["a|1|10|100|10", "a|2|20|102|19", "a|3|30|103|25"]
    );
}

#[test]
fn a_spilled_group_gives_the_same_rows() {
    let s = session();
    let sql = "SELECT count(*), sum(q.id) FROM range(0, 20000, 1, 4) t \
               ASOF JOIN (SELECT id, id * 2 AS v FROM range(0, 20000, 1, 4)) q \
               MATCH_CONDITION (t.id * 2 >= q.v + 1) ON t.id % 3 = q.id % 3";
    let expected = rows(&s, sql);
    block_on(async {
        s.sql("SET krishiv.exec.join.smj.bufferBytes=1024")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert_eq!(rows(&s, sql), expected);
}

#[test]
fn the_flag_gates_the_syntax() {
    let s = Session::new().unwrap();
    let sql = "SELECT * FROM range(3) t ASOF JOIN range(3) q MATCH_CONDITION (t.id >= q.id)";
    assert_eq!(error(&s, sql), "UNSUPPORTED_FEATURE.ASOF_JOIN");
}
