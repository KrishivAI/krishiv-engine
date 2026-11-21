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

/// Without ANSI an integral distance wraps, and Spark's forward scan compares the distances of the
/// match and the next passing row (left `-1`, right `0` and `9223372036854775807`: the second
/// distance wraps to the smallest `BIGINT`), so the forward join refuses there; backward Spark
/// evaluates no distance, and a timestamp distance raises with ANSI off too, so those run.
#[test]
fn a_forward_distance_that_can_wrap_without_ansi_is_refused() {
    let s = session();
    block_on(async {
        s.sql("SET spark.sql.ansi.enabled=false")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let sql = |left: &str, rights: &str, op: &str| {
        format!(
            "SELECT * FROM VALUES ({left}) AS t(i) ASOF JOIN VALUES {rights} AS r(i) \
             MATCH_CONDITION (t.i {op} r.i)"
        )
    };
    assert_eq!(
        error(&s, &sql("-1L", "(0L), (9223372036854775807L)", "<=")),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(rows(&s, &sql("5L", "(1L), (3L), (9L)", ">=")), ["5|3"]);
    let ts = |sec: u32| format!("TIMESTAMP'2024-01-01 00:00:0{sec}'");
    let got = rows(
        &s,
        &format!(
            "SELECT count(*) FROM VALUES ({}) AS t(i) ASOF JOIN VALUES ({}), ({}) AS r(i) \
             MATCH_CONDITION (t.i <= r.i)",
            ts(2),
            ts(4),
            ts(9)
        ),
    );
    assert_eq!(got, ["1"]);
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
    assert_eq!(
        rows(&s, &on_left("LEFT")),
        [
            "1|100", "2|NULL", "3|103", "4|NULL", "5|NULL", "6|NULL", "7|NULL"
        ]
    );
}

/// The class and message parameters of the failure of `sql`.
fn failure(s: &Session, sql: &str) -> (String, Vec<(String, String)>) {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql} ran"),
        Err(e) => (
            e.error_class().name().to_owned(),
            e.message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
        ),
    }
}

/// The analysis errors of a match condition, with Spark's parameters: the cases and answers of
/// Spark 4.3.0's `sql-tests/results/join-asof-errors.sql.out` (the 4.3.0 golden profile), over
/// its `trades` and `quotes` views.
#[test]
fn match_condition_errors_are_sparks() {
    let s = session();
    for sql in [
        "CREATE OR REPLACE TEMP VIEW trades(trade_time, symbol, quantity) AS VALUES \
         (TIMESTAMP '2026-06-29 10:00:05', 'AAPL', 100), \
         (TIMESTAMP '2026-06-29 10:00:11', 'AAPL', 200), \
         (TIMESTAMP '2026-06-29 10:00:12', 'MSFT', 50), \
         (TIMESTAMP '2026-06-29 09:59:59', 'GOOG', 30)",
        "CREATE OR REPLACE TEMP VIEW quotes(quote_time, symbol, bid_price) AS VALUES \
         (TIMESTAMP '2026-06-29 10:00:00', 'AAPL', 180.10), \
         (TIMESTAMP '2026-06-29 10:00:07', 'AAPL', 180.15), \
         (TIMESTAMP '2026-06-29 10:00:10', 'AAPL', 180.20), \
         (TIMESTAMP '2026-06-29 10:00:08', 'MSFT', 420.50)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await })
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    let p = |pairs: &[(&str, &str)]| -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    };
    let cases = [
        (
            "SELECT * FROM trades t ASOF JOIN quotes q \
             MATCH_CONDITION (t.trade_time >= t.trade_time - INTERVAL 1 HOUR) \
             ON t.symbol = q.symbol",
            "ASOF_JOIN_MATCH_CONDITION_TABLE_REFERENCE",
            p(&[
                ("refs1", "\"trade_time\""),
                ("refs2", "\"trade_time - INTERVAL '01' HOUR\""),
            ]),
        ),
        (
            "SELECT * FROM trades t ASOF JOIN quotes q \
             MATCH_CONDITION (t.trade_time >= max(q.quote_time)) ON t.symbol = q.symbol",
            "ASOF_JOIN_MATCH_CONDITION_INVALID_EXPRESSION",
            p(&[("expr", "\"max(quote_time)\"")]),
        ),
        (
            "SELECT * FROM VALUES (MAP('a', 1)) AS t(m) ASOF JOIN VALUES (MAP('a', 1)) AS r(m) \
             MATCH_CONDITION (t.m >= r.m)",
            "ASOF_JOIN_MATCH_CONDITION_INVALID_TYPE",
            p(&[
                ("type1", "\"MAP<STRING, INT>\""),
                ("type2", "\"MAP<STRING, INT>\""),
            ]),
        ),
    ];
    for (sql, class, params) in cases {
        assert_eq!(failure(&s, sql), (class.to_owned(), params), "{sql}");
    }
    // An operand reading both inputs. Spark names it `"(trade_time + (quote_time -
    // trade_time))"`: its outer `+` is not yet rewritten to the timestamp addition when the
    // check runs. Krishiv names the rewritten form, without the outer parentheses.
    let sql = "SELECT * FROM trades t ASOF JOIN quotes q \
               MATCH_CONDITION (t.trade_time + (q.quote_time - t.trade_time) >= q.quote_time) \
               ON t.symbol = q.symbol";
    let (class, params) = failure(&s, sql);
    assert_eq!(class, "ASOF_JOIN_MATCH_CONDITION_TABLE_REFERENCE");
    assert_eq!(params[1], ("refs2".to_owned(), "\"quote_time\"".to_owned()));
}

/// A floating-point distance ranks unlike the values at the infinities: Spark's forward scan over
/// left `-Infinity` and right `-Infinity`, `0`, `5` sees the distances NaN then +Infinity, and
/// +Infinity is smaller than NaN in Spark's order, so Spark returns `-Infinity|0.0` where the merge
/// takes the first right value at or above the left one. The SQL forward join on `DOUBLE` values
/// is refused, ANSI or not; backward Spark evaluates no distance, so that runs.
#[test]
fn a_forward_as_of_join_on_floating_point_values_is_refused() {
    let s = session();
    let sql = |left: &str, rights: &str, op: &str| {
        format!(
            "SELECT * FROM VALUES ({left}) AS t(v) ASOF JOIN VALUES {rights} AS r(v) \
             MATCH_CONDITION (t.v {op} r.v)"
        )
    };
    for ansi in ["true", "false"] {
        block_on(async {
            s.sql(&format!("SET spark.sql.ansi.enabled={ansi}"))
                .await?
                .collect()
                .await
        })
        .unwrap();
        assert_eq!(
            error(
                &s,
                &sql(
                    "CAST('-Infinity' AS DOUBLE)",
                    "(CAST('-Infinity' AS DOUBLE)), (0D), (5D)",
                    "<="
                )
            ),
            "KRISHIV_UNSUPPORTED_FEATURE"
        );
        let backward = sql("5D", "(1D), (3D), (9D)", ">=").replacen(
            "SELECT *",
            "SELECT CAST(t.v AS STRING), CAST(r.v AS STRING)",
            1,
        );
        assert_eq!(rows(&s, &backward), ["5.0|3.0"]);
    }
}

/// A key an earlier `USING` join merged away stays reachable by qualified name, and it is the left
/// input's column (Spark's `AddMetadataColumns` puts it in the left side's output before the match
/// operands are checked): `b.k` is a left value, not a reference to both sides.
#[test]
fn a_hidden_using_key_is_a_left_as_of_value() {
    let s = session();
    let got = rows(
        &s,
        "SELECT k, v FROM VALUES (1), (5) AS a(k) JOIN VALUES (1), (5) AS b(k) USING (k) \
         ASOF JOIN VALUES (0), (3), (9) AS c(v) MATCH_CONDITION (b.k >= c.v) ORDER BY k",
    );
    assert_eq!(got, ["1|0", "5|3"]);
}

/// A file table's `_metadata` is reachable from an as-of operand: Spark's `AddMetadataColumns` adds
/// it to the side's output for the match condition too, so the join input is widened with it.
#[test]
fn an_as_of_operand_may_read_the_metadata_column() {
    let s = session();
    for q in [
        "CREATE TABLE pm(x INT) USING PARQUET",
        "INSERT INTO pm VALUES (1)",
    ] {
        block_on(async { s.sql(q).await?.collect().await }).unwrap();
    }
    let got = rows(
        &s,
        "SELECT x, v FROM pm ASOF JOIN VALUES (0), (3) AS c(v) \
         MATCH_CONDITION (pm._metadata.file_size >= c.v)",
    );
    assert_eq!(got, ["1|3"]);
}
