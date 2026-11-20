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
    // Each `t.id` from 3 takes the greatest `q.id < t.id` of its residue class: `t.id - 3`.
    let expected = ["19997|199930006"];
    assert_eq!(rows(&s, sql), expected);
    block_on(async {
        s.sql("SET krishiv.exec.join.smj.bufferBytes=1024")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let df = block_on(s.sql(sql)).unwrap();
    let (batches, profile) = block_on(df.collect_with_profile()).unwrap();
    let text = profile.to_text(krishiv_session::TextOptions::default());
    let spilled = text
        .lines()
        .filter_map(|l| l.split("groups_spilled=").nth(1))
        .filter_map(|v| {
            v.split(|c: char| !c.is_ascii_digit())
                .next()?
                .parse::<u64>()
                .ok()
        })
        .sum::<u64>();
    assert!(spilled > 0, "a group spilled:\n{text}");
    let got: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(move |r| {
                (0..b.num_columns())
                    .map(|c| cell(b.column(c).as_ref(), r))
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn the_flag_gates_the_syntax() {
    let s = Session::new().unwrap();
    let sql = "SELECT * FROM range(3) t ASOF JOIN range(3) q MATCH_CONDITION (t.id >= q.id)";
    assert_eq!(error(&s, sql), "UNSUPPORTED_FEATURE.ASOF_JOIN");
}

/// Spark 4.3.0 (real run): a forward match evaluates its distance `R - L` on the matched row and
/// on the next passing row, so an ANSI overflow there raises; a later row's does not, and
/// backward evaluates none.
#[test]
fn the_forward_distance_raises_where_spark_raises() {
    let s = session();
    let sql = |left: &str, rights: &str, op: &str| {
        format!(
            "SELECT * FROM VALUES ({left}) AS t(i) ASOF JOIN VALUES {rights} AS r(i) \
             MATCH_CONDITION (t.i {op} r.i)"
        )
    };
    assert_eq!(
        error(&s, &sql("-2", "(2147483647)", "<=")),
        "ARITHMETIC_OVERFLOW"
    );
    // The next passing row after the match (0) overflows.
    assert_eq!(
        error(&s, &sql("-2", "(0), (2147483647)", "<=")),
        "ARITHMETIC_OVERFLOW"
    );
    // The next passing row is 1; 2147483647 is never evaluated.
    assert_eq!(
        rows(&s, &sql("-2", "(0), (1), (2147483647)", "<=")),
        ["-2|0"]
    );
    assert_eq!(
        rows(&s, &sql("2147483647", "(-2)", ">=")),
        ["2147483647|-2"]
    );
}

/// The optimizer moves an as-of join's one-sided conjuncts into its inputs (right-only `ON`
/// conjuncts, left-only ones for an inner join, left-only filters above): the rows are those of
/// the same filters written below by hand.
#[test]
fn one_sided_conditions_filter_the_inputs_with_the_same_rows() {
    let s = session();
    for kind in ["", "LEFT"] {
        let pushed = format!(
            "SELECT t.id, q.qid FROM t {kind} ASOF JOIN q MATCH_CONDITION (t.ts >= q.ts) \
             ON t.k = q.k AND q.qid <> 102"
        );
        let by_hand = format!(
            "SELECT t.id, q.qid FROM t {kind} ASOF JOIN (SELECT * FROM q WHERE qid <> 102) q \
             MATCH_CONDITION (t.ts >= q.ts) ON t.k = q.k"
        );
        assert_eq!(rows(&s, &pushed), rows(&s, &by_hand), "{kind}");
        let above = format!(
            "SELECT t.id, q.qid FROM t {kind} ASOF JOIN q MATCH_CONDITION (t.ts >= q.ts) \
             ON t.k = q.k WHERE t.id <> 2"
        );
        let by_hand = format!(
            "SELECT t.id, q.qid FROM (SELECT * FROM t WHERE id <> 2) t {kind} ASOF JOIN q \
             MATCH_CONDITION (t.ts >= q.ts) ON t.k = q.k"
        );
        assert_eq!(rows(&s, &above), rows(&s, &by_hand), "{kind}");
    }
    // A left-only `ON` conjunct: an inner join drops the rows it rejects, a left join keeps them
    // unmatched.
    let on_left = |kind: &str| {
        format!(
            "SELECT t.id, q.qid FROM t {kind} ASOF JOIN q MATCH_CONDITION (t.ts >= q.ts) \
             ON t.k = q.k AND t.id <> 2"
        )
    };
    assert_eq!(
        rows(&s, &on_left("")),
        rows(
            &s,
            "SELECT t.id, q.qid FROM (SELECT * FROM t WHERE id <> 2) t ASOF JOIN q \
             MATCH_CONDITION (t.ts >= q.ts) ON t.k = q.k"
        )
    );
    assert!(rows(&s, &on_left("LEFT")).contains(&"2|NULL".to_owned()));
}
