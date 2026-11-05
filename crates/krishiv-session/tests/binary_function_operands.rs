// SPDX-License-Identifier: Apache-2.0
//! Two-argument functions that are Spark `BinaryExpression`s return NULL on a left value that is NULL at run time
//! without evaluating the right operand, so a failing right operand on such a row never surfaces (the nightly
//! differential, 2026-11-05: `power`, `add_months`). The control of each function has a left value that is set: the
//! right operand then raises. Every expected answer is that of a real Spark 4.2.0 run of the same statement, ANSI on;
//! the left value comes from `CASE WHEN k > 5 THEN ... END`, so Spark cannot fold it to a NULL literal first.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use krishiv_session::Session;

/// `(name.form, statement, Spark's answer)`.
#[rustfmt::skip]
const SPARK: &[(&str, &str, &str)] = &[
    ("power.null_left", "SELECT CAST(power((CASE WHEN k > 5 THEN 1.0D END), (1 / (k - k))) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("power.set_left", "SELECT CAST(power((CASE WHEN k > -1 THEN 1.0D END), (1 / (k - k))) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("pow.null_left", "SELECT CAST(pow((CASE WHEN k > 5 THEN 1.0D END), (1 / (k - k))) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("pow.set_left", "SELECT CAST(pow((CASE WHEN k > -1 THEN 1.0D END), (1 / (k - k))) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("atan2.null_left", "SELECT CAST(atan2((CASE WHEN k > 5 THEN 1.0D END), (1 / (k - k))) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("atan2.set_left", "SELECT CAST(atan2((CASE WHEN k > -1 THEN 1.0D END), (1 / (k - k))) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("hypot.null_left", "SELECT CAST(hypot((CASE WHEN k > 5 THEN 1.0D END), (1 / (k - k))) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("hypot.set_left", "SELECT CAST(hypot((CASE WHEN k > -1 THEN 1.0D END), (1 / (k - k))) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("log2.null_left", "SELECT CAST(log((CASE WHEN k > 5 THEN 1.0D END), (1 / (k - k))) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("log2.set_left", "SELECT CAST(log((CASE WHEN k > -1 THEN 1.0D END), (1 / (k - k))) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("add_months.null_left", "SELECT CAST(add_months((CASE WHEN k > 5 THEN DATE'2020-01-01' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("add_months.set_left", "SELECT CAST(add_months((CASE WHEN k > -1 THEN DATE'2020-01-01' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("date_add.null_left", "SELECT CAST(date_add((CASE WHEN k > 5 THEN DATE'2020-01-01' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("date_add.set_left", "SELECT CAST(date_add((CASE WHEN k > -1 THEN DATE'2020-01-01' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("date_sub.null_left", "SELECT CAST(date_sub((CASE WHEN k > 5 THEN DATE'2020-01-01' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("date_sub.set_left", "SELECT CAST(date_sub((CASE WHEN k > -1 THEN DATE'2020-01-01' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("bit_and.null_left", "SELECT CAST(((CASE WHEN k > 5 THEN 1L END) & CAST((1 / (k - k)) AS BIGINT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("bit_and.set_left", "SELECT CAST(((CASE WHEN k > -1 THEN 1L END) & CAST((1 / (k - k)) AS BIGINT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("bit_or.null_left", "SELECT CAST(((CASE WHEN k > 5 THEN 1L END) | CAST((1 / (k - k)) AS BIGINT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("bit_or.set_left", "SELECT CAST(((CASE WHEN k > -1 THEN 1L END) | CAST((1 / (k - k)) AS BIGINT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("bit_xor.null_left", "SELECT CAST(((CASE WHEN k > 5 THEN 1L END) ^ CAST((1 / (k - k)) AS BIGINT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("bit_xor.set_left", "SELECT CAST(((CASE WHEN k > -1 THEN 1L END) ^ CAST((1 / (k - k)) AS BIGINT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("shiftleft.null_left", "SELECT CAST(shiftleft((CASE WHEN k > 5 THEN 1L END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("shiftleft.set_left", "SELECT CAST(shiftleft((CASE WHEN k > -1 THEN 1L END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("shiftright.null_left", "SELECT CAST(shiftright((CASE WHEN k > 5 THEN 1L END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("shiftright.set_left", "SELECT CAST(shiftright((CASE WHEN k > -1 THEN 1L END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("startswith.null_left", "SELECT CAST(startswith((CASE WHEN k > 5 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("startswith.set_left", "SELECT CAST(startswith((CASE WHEN k > -1 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("endswith.null_left", "SELECT CAST(endswith((CASE WHEN k > 5 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("endswith.set_left", "SELECT CAST(endswith((CASE WHEN k > -1 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("contains.null_left", "SELECT CAST(contains((CASE WHEN k > 5 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("contains.set_left", "SELECT CAST(contains((CASE WHEN k > -1 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("repeat.null_left", "SELECT CAST(repeat((CASE WHEN k > 5 THEN 'x' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("repeat.set_left", "SELECT CAST(repeat((CASE WHEN k > -1 THEN 'x' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("left.null_left", "SELECT CAST(left((CASE WHEN k > 5 THEN 'x' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("left.set_left", "SELECT CAST(left((CASE WHEN k > -1 THEN 'x' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("right.null_left", "SELECT CAST(right((CASE WHEN k > 5 THEN 'x' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("right.set_left", "SELECT CAST(right((CASE WHEN k > -1 THEN 'x' END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("instr.null_left", "SELECT CAST(instr((CASE WHEN k > 5 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("instr.set_left", "SELECT CAST(instr((CASE WHEN k > -1 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("levenshtein.null_left", "SELECT CAST(levenshtein((CASE WHEN k > 5 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("levenshtein.set_left", "SELECT CAST(levenshtein((CASE WHEN k > -1 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("like.null_left", "SELECT CAST(((CASE WHEN k > 5 THEN 'x' END) LIKE CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("like.set_left", "SELECT CAST(((CASE WHEN k > -1 THEN 'x' END) LIKE CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("find_in_set.null_left", "SELECT CAST(find_in_set((CASE WHEN k > 5 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("find_in_set.set_left", "SELECT CAST(find_in_set((CASE WHEN k > -1 THEN 'x' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("array_contains.null_left", "SELECT CAST(array_contains((CASE WHEN k > 5 THEN array(1) END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("array_contains.set_left", "SELECT CAST(array_contains((CASE WHEN k > -1 THEN array(1) END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("element_at.null_left", "SELECT CAST(element_at((CASE WHEN k > 5 THEN array(1) END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("element_at.set_left", "SELECT CAST(element_at((CASE WHEN k > -1 THEN array(1) END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("next_day.null_left", "SELECT CAST(next_day((CASE WHEN k > 5 THEN DATE'2020-01-01' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("next_day.set_left", "SELECT CAST(next_day((CASE WHEN k > -1 THEN DATE'2020-01-01' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("date_format.null_left", "SELECT CAST(date_format((CASE WHEN k > 5 THEN TIMESTAMP'2020-01-01 00:00:00' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("date_format.set_left", "SELECT CAST(date_format((CASE WHEN k > -1 THEN TIMESTAMP'2020-01-01 00:00:00' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("trunc.null_left", "SELECT CAST(trunc((CASE WHEN k > 5 THEN DATE'2020-01-01' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("trunc.set_left", "SELECT CAST(trunc((CASE WHEN k > -1 THEN DATE'2020-01-01' END), CAST((1 / (k - k)) AS STRING)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("nullif.null_left", "SELECT CAST(nullif((CASE WHEN k > 5 THEN 1 END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("nullif.set_left", "SELECT CAST(nullif((CASE WHEN k > -1 THEN 1 END), CAST((1 / (k - k)) AS INT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("add.null_left", "SELECT CAST(((CASE WHEN k > 5 THEN 1L END) + CAST((1 / (k - k)) AS BIGINT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("add.set_left", "SELECT CAST(((CASE WHEN k > -1 THEN 1L END) + CAST((1 / (k - k)) AS BIGINT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
    ("lt.null_left", "SELECT CAST(((CASE WHEN k > 5 THEN 1L END) < CAST((1 / (k - k)) AS BIGINT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "NULL"),
    ("lt.set_left", "SELECT CAST(((CASE WHEN k > -1 THEN 1L END) < CAST((1 / (k - k)) AS BIGINT)) AS STRING) FROM (SELECT id AS k FROM range(1)) AS t", "ERR:DIVIDE_BY_ZERO"),
];

#[test]
fn binary_functions_skip_a_failing_right_operand_where_the_left_is_null() {
    let s = Session::new().unwrap();
    let mut wrong = Vec::new();
    for (name, sql, want) in SPARK {
        let got = match futures::executor::block_on(async { s.sql(sql).await?.collect().await }) {
            Ok(batches) => {
                let a = batches[0].column(0).as_string_view();
                if a.is_null(0) {
                    "NULL".to_owned()
                } else {
                    a.value(0).to_owned()
                }
            }
            Err(e) => format!("ERR:{}", e.error_class().name()),
        };
        if got != *want {
            wrong.push(format!("{name}: Spark {want}, got {got}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(SPARK.len(), 62);
}
