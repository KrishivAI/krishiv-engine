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
    // An operand reading both inputs, named as Spark's check sees it: before the time zone of a
    // datetime node the arithmetic rewrite made, so an operator over one is still `(l op r)`
    // (Spark 4.3.0 runs; one-level `trade_time - INTERVAL '01' HOUR` is in the table above).
    let both = [
        (
            "t.trade_time + (q.quote_time - t.trade_time)",
            "(trade_time + (quote_time - trade_time))",
        ),
        (
            "(t.trade_time + INTERVAL '1' HOUR) + (q.quote_time - t.trade_time)",
            "(trade_time + INTERVAL '01' HOUR + (quote_time - trade_time))",
        ),
        (
            "(t.trade_time + INTERVAL '1' MONTH) + (q.quote_time - t.trade_time)",
            "(trade_time + INTERVAL '1' MONTH + (quote_time - trade_time))",
        ),
        (
            "(t.trade_time - INTERVAL '1' HOUR) + (q.quote_time - t.trade_time)",
            "(trade_time - INTERVAL '01' HOUR + (quote_time - trade_time))",
        ),
        (
            "t.trade_time - (q.quote_time - t.trade_time)",
            "(trade_time - (quote_time - trade_time))",
        ),
        (
            "t.trade_time + (q.quote_time - t.trade_time) + INTERVAL '1' HOUR",
            "((trade_time + (quote_time - trade_time)) + INTERVAL '01' HOUR)",
        ),
        // Explicit casts have their zone, so the date arithmetic is rewritten.
        (
            "CAST(t.trade_time AS DATE) + (CAST(q.quote_time AS DATE) - CAST(t.trade_time AS DATE))",
            "date_add(CAST(trade_time AS DATE), extractansiintervaldays((CAST(quote_time AS DATE) - CAST(trade_time AS DATE))))",
        ),
    ];
    for (operand, named) in both {
        let sql = format!(
            "SELECT * FROM trades t ASOF JOIN quotes q \
             MATCH_CONDITION ({operand} >= q.quote_time) ON t.symbol = q.symbol"
        );
        let (class, params) = failure(&s, &sql);
        assert_eq!(class, "ASOF_JOIN_MATCH_CONDITION_TABLE_REFERENCE", "{sql}");
        assert_eq!(
            params,
            vec![
                ("refs1".to_owned(), format!("\"{named}\"")),
                ("refs2".to_owned(), "\"quote_time\"".to_owned()),
            ],
            "{sql}"
        );
    }
    // Spark's check waits for the `ON` condition to resolve (`condition.forall(_.resolved)`), so a
    // condition that needs a later pass (a zoned datetime node or implicit cast) lets the operand's
    // next level be rewritten first (Spark 4.3.0 runs, `rev-f2-asof-ref`, `opus1-f5`).
    for sql in [
        "CREATE OR REPLACE TEMP VIEW trades_on AS SELECT * FROM VALUES \
         (1, TIMESTAMP'2024-01-01 10:00:00', '2024-01-01 10:00:00') AS t(k, trade_time, s)",
        "CREATE OR REPLACE TEMP VIEW quotes_on AS SELECT * FROM VALUES \
         (1, TIMESTAMP'2024-01-01 09:00:00', '2024-01-01 09:00:00') AS q(k, quote_time, s)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    let one = "t.trade_time + (q.quote_time - t.trade_time)";
    let two = "t.trade_time + (q.quote_time - t.trade_time) + INTERVAL '1' HOUR";
    let date = "CAST(t.trade_time AS DATE) - INTERVAL '1' MONTH + (q.quote_time - t.trade_time)";
    let delayed = [
        // Pass 1.
        (one, "USING (k)", "(trade_time + (quote_time - trade_time))"),
        (one, "", "(trade_time + (quote_time - trade_time))"),
        (
            one,
            "ON t.k = q.k AND CAST(t.s AS TIMESTAMP) > q.quote_time",
            "(trade_time + (quote_time - trade_time))",
        ),
        (
            two,
            "ON t.k = q.k",
            "((trade_time + (quote_time - trade_time)) + INTERVAL '01' HOUR)",
        ),
        (
            date,
            "ON t.k = q.k",
            "(CAST(trade_time AS DATE) - INTERVAL '1' MONTH + (quote_time - trade_time))",
        ),
        // Pass 2.
        (
            one,
            "ON t.k = q.k AND t.trade_time + INTERVAL '1' HOUR > q.quote_time",
            "trade_time + (quote_time - trade_time)",
        ),
        (
            one,
            "ON t.trade_time = q.s",
            "trade_time + (quote_time - trade_time)",
        ),
        (
            one,
            "ON t.k = q.k AND t.trade_time - q.quote_time > INTERVAL '1' HOUR",
            "trade_time + (quote_time - trade_time)",
        ),
        (
            two,
            "ON t.k = q.k AND t.trade_time + INTERVAL '1' HOUR > q.quote_time",
            "(trade_time + (quote_time - trade_time) + INTERVAL '01' HOUR)",
        ),
        (
            two,
            "ON t.trade_time = q.s",
            "(trade_time + (quote_time - trade_time) + INTERVAL '01' HOUR)",
        ),
        (
            date,
            "ON t.trade_time = q.s",
            "CAST(trade_time AS DATE) - INTERVAL '1' MONTH + (quote_time - trade_time)",
        ),
        // Pass 3.
        (
            two,
            "ON t.k = q.k AND (t.trade_time + INTERVAL '1' HOUR) + INTERVAL '1' HOUR > q.quote_time",
            "trade_time + (quote_time - trade_time) + INTERVAL '01' HOUR",
        ),
    ];
    for (operand, on, named) in delayed {
        let sql = format!(
            "SELECT * FROM trades_on t ASOF JOIN quotes_on q \
             MATCH_CONDITION ({operand} >= q.quote_time) {on}"
        );
        let (class, params) = failure(&s, &sql);
        assert_eq!(class, "ASOF_JOIN_MATCH_CONDITION_TABLE_REFERENCE", "{sql}");
        assert_eq!(
            params,
            vec![
                ("refs1".to_owned(), format!("\"{named}\"")),
                ("refs2".to_owned(), "\"quote_time\"".to_owned()),
            ],
            "{sql}"
        );
    }
    // `date - <year-month interval>` is Spark's `DateAddYMInterval`, which has no zone: it
    // resolves at once, so an operator over it is rewritten too (no outer parentheses); the
    // timestamp form is zoned and stays `(l op r)` (Spark 4.3.0 runs, review of dad55024c F3).
    for sql in [
        "CREATE OR REPLACE TEMP VIEW trades_ym AS SELECT * FROM VALUES \
         ('A', TIMESTAMP'2024-01-01 10:00:00', INTERVAL '1' MONTH) AS t(symbol, trade_time, ym)",
        "CREATE OR REPLACE TEMP VIEW quotes_ym AS SELECT * FROM VALUES \
         ('A', TIMESTAMP'2024-01-01 09:00:00', INTERVAL '2' MONTH) AS q(symbol, quote_time, ym)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    let year_month = [
        (
            "(CAST(t.trade_time AS DATE) - INTERVAL '1' MONTH) + q.ym >= CAST(q.quote_time AS DATE)",
            "CAST(trade_time AS DATE) - INTERVAL '1' MONTH + ym",
            "CAST(quote_time AS DATE)",
        ),
        (
            "(t.trade_time - INTERVAL '1' MONTH) + q.ym >= q.quote_time",
            "(trade_time - INTERVAL '1' MONTH + ym)",
            "quote_time",
        ),
        (
            "(CAST(t.trade_time AS DATE) + INTERVAL '1' MONTH) + q.ym >= CAST(q.quote_time AS DATE)",
            "CAST(trade_time AS DATE) + INTERVAL '1' MONTH + ym",
            "CAST(quote_time AS DATE)",
        ),
    ];
    for (condition, refs1, refs2) in year_month {
        let sql = format!(
            "SELECT * FROM trades_ym t ASOF JOIN quotes_ym q \
             MATCH_CONDITION ({condition}) ON t.symbol = q.symbol"
        );
        let (class, params) = failure(&s, &sql);
        assert_eq!(class, "ASOF_JOIN_MATCH_CONDITION_TABLE_REFERENCE", "{sql}");
        assert_eq!(
            params,
            vec![
                ("refs1".to_owned(), format!("\"{refs1}\"")),
                ("refs2".to_owned(), format!("\"{refs2}\"")),
            ],
            "{sql}"
        );
    }
    on_condition_passes_and_errors(&s);
}

/// Any implicit cast that needs a time zone delays the `ON` condition one pass, whatever node adds
/// it (`IN`, `BETWEEN`, `coalesce`, `CASE`, a function's argument), and an `ON` condition that does
/// not resolve is reported before any as-of check (Spark 4.3.0 runs, `rev-f1-spark/delta8c9`
/// p11-p22, `opus1-r3` r3 and r4; review of 8c9ab3fba findings 3 and 4).
fn on_condition_passes_and_errors(s: &Session) {
    for sql in [
        "CREATE OR REPLACE TEMP VIEW trades_d AS SELECT * FROM VALUES (1, \
         TIMESTAMP'2024-01-01 10:00:00', '2024-01-01 10:00:00', DATE'2024-01-01', \
         TIMESTAMP_NTZ'2024-01-01 10:00:00') AS t(k, trade_time, s, d, n)",
        "CREATE OR REPLACE TEMP VIEW quotes_d AS SELECT * FROM VALUES (1, \
         TIMESTAMP'2024-01-01 09:00:00', '2024-01-01 09:00:00', DATE'2024-01-01', \
         TIMESTAMP_NTZ'2024-01-01 09:00:00') AS q(k, quote_time, s, d, n)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    }
    let rewritten = "trade_time + (quote_time - trade_time)";
    let written = "(trade_time + (quote_time - trade_time))";
    for (on, named) in [
        ("t.k = q.k AND hour(q.s) > 3", rewritten),
        ("t.k = q.k AND t.trade_time IN (q.s)", rewritten),
        (
            "t.k = q.k AND coalesce(t.trade_time, q.s) > q.quote_time",
            rewritten,
        ),
        (
            "t.k = q.k AND coalesce(t.trade_time, q.s) IS NOT NULL",
            rewritten,
        ),
        ("t.k = q.k AND t.trade_time BETWEEN q.s AND q.s", rewritten),
        (
            "t.k = q.k AND datediff(t.trade_time, q.quote_time) = 0",
            rewritten,
        ),
        ("t.k = q.k AND date_add(t.trade_time, 1) > q.d", rewritten),
        ("t.k = q.k AND hour(t.s) = 1", rewritten),
        ("t.k = q.k AND date_format(t.d, 'yyyy') = q.s", rewritten),
        (
            "t.k = q.k AND CASE WHEN t.k = 1 THEN t.trade_time ELSE q.d END IS NOT NULL",
            rewritten,
        ),
        // An NTZ argument Spark casts to TIMESTAMP (`MonthsBetween`, `UTCTimestamp`,
        // `TimestampDiff`) and Krishiv's kernel reads as written (Spark 4.3.0 runs).
        (
            "t.k = q.k AND months_between(t.n, q.quote_time) > 0",
            rewritten,
        ),
        (
            "t.k = q.k AND from_utc_timestamp(t.n, 'UTC') > q.quote_time",
            rewritten,
        ),
        (
            "t.k = q.k AND to_utc_timestamp(q.n, 'UTC') > t.trade_time",
            rewritten,
        ),
        (
            "t.k = q.k AND timestampdiff(HOUR, t.n, q.quote_time) > 0",
            rewritten,
        ),
        ("t.k = q.k AND timestampdiff(HOUR, t.n, q.n) > 0", rewritten),
        // Controls: no implicit zoned cast, so the check runs in pass 1.
        ("t.k = q.k AND hour(q.quote_time) > 3", written),
        (
            "t.k = q.k AND months_between(t.trade_time, q.quote_time) > 0",
            written,
        ),
        // `timestampadd` takes any timestamp: the NTZ stays one in Spark too.
        ("t.k = q.k AND timestampadd(HOUR, 1, t.n) > q.n", written),
        // A condition that resolves and is not a boolean fails a later check, after the as-of one.
        ("t.k + q.k", written),
    ] {
        let sql = format!(
            "SELECT * FROM trades_d t ASOF JOIN quotes_d q \
             MATCH_CONDITION (t.trade_time + (q.quote_time - t.trade_time) >= q.quote_time) ON {on}"
        );
        let (class, params) = failure(s, &sql);
        assert_eq!(class, "ASOF_JOIN_MATCH_CONDITION_TABLE_REFERENCE", "{sql}");
        assert_eq!(
            params,
            vec![
                ("refs1".to_owned(), format!("\"{named}\"")),
                ("refs2".to_owned(), "\"quote_time\"".to_owned()),
            ],
            "{sql}"
        );
    }
    // An `ON` condition that does not resolve is the error, before the as-of checks.
    for (condition, on, class, sql_expr) in [
        (
            "t.trade_time >= q.s",
            "t.k = q.nope",
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            None,
        ),
        (
            "t.trade_time + (q.quote_time - t.trade_time) >= q.quote_time",
            "t.k = q.k AND t.k + t.trade_time > 1",
            "DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES",
            Some("(k + trade_time)"),
        ),
        (
            "t.trade_time >= q.quote_time",
            "t.k = array(q.k)",
            "DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES",
            Some("(k = array(k))"),
        ),
        (
            "t.trade_time >= q.quote_time",
            "t.k = q.k AND t.d - q.d > 0",
            "DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES",
            Some("((d - d) > 0)"),
        ),
        // The match condition alone would be `ASOF_JOIN_MATCH_CONDITION_INVALID_TYPE`.
        (
            "t.trade_time >= q.s",
            "t.k = q.k AND t.k + t.trade_time > 1",
            "DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES",
            Some("(k + trade_time)"),
        ),
        (
            "t.k >= q.d",
            "t.k = q.nope",
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            None,
        ),
    ] {
        let sql = format!(
            "SELECT * FROM trades_d t ASOF JOIN quotes_d q MATCH_CONDITION ({condition}) ON {on}"
        );
        let (got, params) = failure(s, &sql);
        assert_eq!(got, class, "{sql}");
        if let Some(e) = sql_expr {
            let shown = params
                .iter()
                .find(|(k, _)| k == "sqlExpr")
                .map(|(_, v)| v.as_str());
            assert_eq!(shown, Some(format!("\"{e}\"").as_str()), "{sql}");
        }
    }
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

/// Spark subtracts `TIMESTAMP`s as local date-times in the session zone, so across New York's
/// 2026-11-01 fallback the later right row (01:15 EST, 06:15Z) is 45 minutes from the left
/// (00:30 EDT) and the earlier one (01:45 EDT, 05:45Z) 75: Spark's forward scan takes the later.
/// The merge takes the first by value, so a forward join there is refused; backward evaluates no
/// distance, and a fixed-offset zone or `TIMESTAMP_NTZ` values keep the order.
#[test]
fn a_forward_timestamp_as_of_join_across_daylight_saving_is_refused() {
    let s = session();
    let set = |sql: &str| {
        block_on(async { s.sql(sql).await?.collect().await })
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    };
    let sql = |ty: &str, op: &str| {
        format!(
            "SELECT r.id FROM VALUES (CAST('2026-11-01 04:30:00Z' AS {ty})) AS t(v) ASOF JOIN \
             VALUES (1, CAST('2026-11-01 05:45:00Z' AS {ty})), \
             (2, CAST('2026-11-01 06:15:00Z' AS {ty})) AS r(id, v) MATCH_CONDITION (t.v {op} r.v)"
        )
    };
    set("SET spark.sql.session.timeZone=America/New_York");
    assert_eq!(
        error(&s, &sql("TIMESTAMP", "<=")),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(
        rows(&s, &sql("TIMESTAMP", ">=").replace("04:30", "07:00")),
        ["2"]
    );
    assert_eq!(
        rows(&s, &sql("TIMESTAMP_NTZ", "<=").replace("Z'", "'")),
        ["1"]
    );
    set("SET spark.sql.session.timeZone=UTC");
    assert_eq!(rows(&s, &sql("TIMESTAMP", "<=")), ["1"]);
}

/// Spark builds the distance in every form, so `CheckAnalysis` checks its subtraction even where
/// it is never evaluated (SQL backward); for the operand pairs below, a sample of those the
/// comparison accepts, the subtraction resolves too (a pair whose subtraction fails to resolve
/// needs the Connect form, unit 2). Spark 4.3.0 runs (`opus1-asof-c`): each query below analyzes, and
/// the backward ones find their row (`a.dec >= b.s` none); a forward match finds none in this data.
/// A distance that does not resolve would now fail the query, where it was dropped before.
#[test]
fn the_distance_resolves_for_every_comparable_pair() {
    let s = session();
    for sql in [
        "CREATE TEMPORARY VIEW a AS SELECT * FROM VALUES (1, 5, DATE'2024-01-02', \
         TIMESTAMP'2024-01-01 10:00:00', TIMESTAMP_NTZ'2024-01-01 10:00:00', 1.5D, 2.5BD, '7') \
         AS a(k, i, d, ts, ntz, db, dec, s)",
        "CREATE TEMPORARY VIEW b AS SELECT * FROM VALUES (1, 4, DATE'2024-01-01', \
         TIMESTAMP'2024-01-01 09:00:00', TIMESTAMP_NTZ'2024-01-01 09:00:00', 0.5D, 1.5BD, '3') \
         AS b(k, i, d, ts, ntz, db, dec, s)",
    ] {
        rows(&s, sql);
    }
    for (cond, n) in [
        ("a.i <= b.s", 0),
        ("a.d <= b.ts", 0),
        ("a.d <= b.ntz", 0),
        ("a.ts <= b.ntz", 0),
        ("a.ntz <= b.ts", 0),
        ("a.ts <= b.d", 0),
        ("a.s <= b.i", 0),
        ("a.d >= b.ts", 1),
        ("a.i >= b.s", 1),
        ("a.s >= b.i", 1),
        ("a.dec >= b.s", 0),
        // An untyped `NULL` meets any type (Spark's `NullType`); nothing compares to it.
        ("a.i >= NULL", 0),
        ("a.i <= NULL", 0),
        ("NULL <= b.i", 0),
        ("a.d <= NULL", 0),
        ("a.ts >= NULL", 0),
        ("a.ts <= NULL", 0),
    ] {
        let sql = format!("SELECT a.k FROM a ASOF JOIN b MATCH_CONDITION ({cond}) ON a.k = b.k");
        assert_eq!(rows(&s, &sql).len(), n, "{cond}");
    }
    // A left as-of join keeps the row, with no match, scalar or composite (Spark 4.3.0,
    // `rev-f2-asof-ref`, `rev-f2-asof-ref2` k.out).
    for cond in [
        "a.i >= NULL",
        "array(a.i) >= NULL",
        "array(a.i) <= NULL",
        "(a.i, a.k) >= NULL",
    ] {
        let sql = format!(
            "SELECT a.k, b.k FROM a LEFT ASOF JOIN b MATCH_CONDITION ({cond}) ON a.k = b.k"
        );
        assert_eq!(rows(&s, &sql), ["1|NULL"], "{cond}");
    }
    // An interval meets a STRING at STRING, so the as-of rule accepts the pair and the comparison
    // as written refuses it (Spark 4.3.0, `rev-f2-asof-ref` `outc.txt`).
    rows(
        &s,
        "CREATE TEMPORARY VIEW c AS SELECT * FROM VALUES \
         (1, INTERVAL '3' HOUR, INTERVAL '1' MONTH, '7', '0-2') AS c(k, dti, ymi, s, sy)",
    );
    // Nothing compares to an untyped `NULL` (Spark 4.3.0, `rev-f2-asof-ref2` k.out).
    for cond in ["NULL >= NULL", "c.dti >= NULL", "NULL >= e.dti"] {
        let sql =
            format!("SELECT c.k FROM c ASOF JOIN c AS e MATCH_CONDITION ({cond}) ON c.k = e.k");
        assert!(rows(&s, &sql).is_empty(), "{cond}");
    }
    // The comparison is the normalized one: the left input's operand first, the operator flipped
    // with it (Spark 4.3.0, `rev-f1-spark/delta8c9` p1, p2).
    for (cond, shown, types) in [
        ("c.dti <= e.s", "(dti <= s)", ("INTERVAL HOUR", "STRING")),
        ("c.dti >= e.s", "(dti >= s)", ("INTERVAL HOUR", "STRING")),
        ("c.s >= e.dti", "(s >= dti)", ("STRING", "INTERVAL HOUR")),
        ("c.ymi <= e.sy", "(ymi <= sy)", ("INTERVAL MONTH", "STRING")),
        ("c.ymi >= e.sy", "(ymi >= sy)", ("INTERVAL MONTH", "STRING")),
        ("e.s >= c.dti", "(dti <= s)", ("INTERVAL HOUR", "STRING")),
        ("e.s < c.dti", "(dti > s)", ("INTERVAL HOUR", "STRING")),
        // Structs are compared rebuilt from their fields, named as `CreateStruct` names them
        // (Spark 4.3.0, `rev-f2-asof-ref2` k.out, `opus1-r3` r4.out).
        (
            "named_struct('x', c.dti) >= named_struct('x', e.s)",
            "(named_struct(dti, dti) >= named_struct(s, s))",
            (
                "STRUCT<dti: INTERVAL HOUR NOT NULL>",
                "STRUCT<s: STRING NOT NULL>",
            ),
        ),
        (
            "named_struct('x', e.s) <= named_struct('y', c.dti)",
            "(named_struct(dti, dti) >= named_struct(s, s))",
            (
                "STRUCT<dti: INTERVAL HOUR NOT NULL>",
                "STRUCT<s: STRING NOT NULL>",
            ),
        ),
        (
            "(c.dti, c.k) >= (e.s, e.k)",
            "(named_struct(dti, dti, k, k) >= named_struct(s, s, k, k))",
            (
                "STRUCT<dti: INTERVAL HOUR NOT NULL, k: INT NOT NULL>",
                "STRUCT<s: STRING NOT NULL, k: INT NOT NULL>",
            ),
        ),
        (
            "(c.dti, c.k + 1) >= (e.s, e.k)",
            "(named_struct(dti, dti, col2, (k + 1)) >= named_struct(s, s, k, k))",
            (
                "STRUCT<dti: INTERVAL HOUR NOT NULL, col2: INT NOT NULL>",
                "STRUCT<s: STRING NOT NULL, k: INT NOT NULL>",
            ),
        ),
        (
            "named_struct('x', c.k, 'y', named_struct('z', c.dti)) >= \
             named_struct('x', e.k, 'y', named_struct('w', e.s))",
            "(named_struct(k, k, col2, named_struct(dti, dti)) >= \
             named_struct(k, k, col2, named_struct(s, s)))",
            (
                "STRUCT<k: INT NOT NULL, col2: STRUCT<dti: INTERVAL HOUR NOT NULL> NOT NULL>",
                "STRUCT<k: INT NOT NULL, col2: STRUCT<s: STRING NOT NULL> NOT NULL>",
            ),
        ),
    ] {
        let sql =
            format!("SELECT c.k FROM c ASOF JOIN c AS e MATCH_CONDITION ({cond}) ON c.k = e.k");
        let (class, params) = failure(&s, &sql);
        assert_eq!(class, "DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES", "{cond}");
        let get = |k: &str| params.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        assert_eq!(get("sqlExpr"), Some(format!("\"{shown}\"")), "{cond}");
        assert_eq!(get("left"), Some(format!("\"{}\"", types.0)), "{cond}");
        assert_eq!(get("right"), Some(format!("\"{}\"", types.1)), "{cond}");
    }
    // A forward double distance resolves too; Krishiv refuses to run it (its order differs from
    // the values' at the infinities), which Spark does not.
    for cond in ["a.i <= b.db", "a.dec <= b.db"] {
        let sql = format!("SELECT a.k FROM a ASOF JOIN b MATCH_CONDITION ({cond}) ON a.k = b.k");
        assert_eq!(error(&s, &sql), "KRISHIV_UNSUPPORTED_FEATURE", "{cond}");
    }
}

/// A nested struct field's wider type matches names as the session does: `A` meets `a` unless
/// `spark.sql.caseSensitive` is on (Spark 4.3.0, `rev-f1-spark/delta8c9` p3 and p5; review of
/// 8c9ab3fba finding 2).
#[test]
fn nested_struct_fields_match_names_as_the_session_does() {
    let s = session();
    for sql in [
        "CREATE TEMPORARY VIEW ca AS SELECT 1 AS k, \
         array(named_struct('f', named_struct('A', '2024-01-02'))) AS x",
        "CREATE TEMPORARY VIEW cb AS SELECT 1 AS k, \
         array(named_struct('f', named_struct('a', DATE'2024-01-01'))) AS y",
    ] {
        rows(&s, sql);
    }
    let sql = "SELECT ca.k FROM ca ASOF JOIN cb MATCH_CONDITION (ca.x >= cb.y) ON ca.k = cb.k";
    let types = (
        "\"ARRAY<STRUCT<f: STRUCT<A: STRING NOT NULL> NOT NULL>>\"",
        "\"ARRAY<STRUCT<f: STRUCT<a: DATE NOT NULL> NOT NULL>>\"",
    );
    let (class, params) = failure(&s, sql);
    assert_eq!(class, "DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES");
    let get = |k: &str| params.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
    assert_eq!(get("sqlExpr").as_deref(), Some("\"(x >= y)\""));
    assert_eq!(get("left").as_deref(), Some(types.0));
    assert_eq!(get("right").as_deref(), Some(types.1));
    rows(&s, "SET spark.sql.caseSensitive=true");
    let (class, params) = failure(&s, sql);
    assert_eq!(class, "ASOF_JOIN_MATCH_CONDITION_INVALID_TYPE");
    assert_eq!(
        params,
        vec![
            ("type1".to_owned(), types.0.to_owned()),
            ("type2".to_owned(), types.1.to_owned()),
        ]
    );
}
