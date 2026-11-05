// SPDX-License-Identifier: Apache-2.0
//! The operand order of Spark's `DivModLike.eval` (`/`, `%`, `div`, `pmod`): the divisor is evaluated first, and the
//! left operand only where the divisor is set and, without ANSI mode, not zero (Spark 4.2.0,
//! `arithmetic.scala`). So an error in the left operand never surfaces where the divisor is NULL; a zero divisor over
//! a NULL left value is NULL (no error); and a zero divisor over a failing left operand raises the left error, not
//! the division by zero. The expectations are the answers of a real Spark 4.2.0 run of the same expressions over
//! the same rows (`s`, `l`), one row per query, ANSI on and off.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_session::Session;

/// The rows `k`: `(s, l)`.
const ROWS: [(Option<&str>, Option<i64>); 6] = [
    (Some("x"), None),    // 0: the divisor is NULL, the left cast would fail
    (Some("x"), Some(2)), // 1: the left cast fails
    (None, Some(0)),      // 2: the left is NULL, the divisor is zero
    (Some("x"), Some(0)), // 3: the left cast fails, the divisor is zero
    (Some("7"), Some(0)), // 4: a good left, the divisor is zero
    (Some("7"), Some(2)), // 5: a good row
];

const OPS: [(&str, &str); 4] = [
    ("/", "CAST(s AS BIGINT) / l"),
    ("%", "CAST(s AS BIGINT) % l"),
    ("div", "CAST(s AS BIGINT) div l"),
    ("pmod", "pmod(CAST(s AS BIGINT), l)"),
];

/// `(ansi, operator, k, Spark's answer)`: a value, `NULL`, or `ERR:<error class>`.
#[rustfmt::skip]
const SPARK: &[(bool, &str, i32, &str)] = &[
        (true, "/", 0, "NULL"),
        (true, "/", 1, "ERR:CAST_INVALID_INPUT"),
        (true, "/", 2, "NULL"),
        (true, "/", 3, "ERR:CAST_INVALID_INPUT"),
        (true, "/", 4, "ERR:DIVIDE_BY_ZERO"),
        (true, "/", 5, "3.5"),
        (true, "%", 0, "NULL"),
        (true, "%", 1, "ERR:CAST_INVALID_INPUT"),
        (true, "%", 2, "NULL"),
        (true, "%", 3, "ERR:CAST_INVALID_INPUT"),
        (true, "%", 4, "ERR:REMAINDER_BY_ZERO"),
        (true, "%", 5, "1"),
        (true, "div", 0, "NULL"),
        (true, "div", 1, "ERR:CAST_INVALID_INPUT"),
        (true, "div", 2, "NULL"),
        (true, "div", 3, "ERR:CAST_INVALID_INPUT"),
        (true, "div", 4, "ERR:DIVIDE_BY_ZERO"),
        (true, "div", 5, "3"),
        (true, "pmod", 0, "NULL"),
        (true, "pmod", 1, "ERR:CAST_INVALID_INPUT"),
        (true, "pmod", 2, "NULL"),
        (true, "pmod", 3, "ERR:CAST_INVALID_INPUT"),
        (true, "pmod", 4, "ERR:REMAINDER_BY_ZERO"),
        (true, "pmod", 5, "1"),
        (false, "/", 0, "NULL"),
        (false, "/", 1, "NULL"),
        (false, "/", 2, "NULL"),
        (false, "/", 3, "NULL"),
        (false, "/", 4, "NULL"),
        (false, "/", 5, "3.5"),
        (false, "%", 0, "NULL"),
        (false, "%", 1, "NULL"),
        (false, "%", 2, "NULL"),
        (false, "%", 3, "NULL"),
        (false, "%", 4, "NULL"),
        (false, "%", 5, "1"),
        (false, "div", 0, "NULL"),
        (false, "div", 1, "NULL"),
        (false, "div", 2, "NULL"),
        (false, "div", 3, "NULL"),
        (false, "div", 4, "NULL"),
        (false, "div", 5, "3"),
        (false, "pmod", 0, "NULL"),
        (false, "pmod", 1, "NULL"),
        (false, "pmod", 2, "NULL"),
        (false, "pmod", 3, "NULL"),
        (false, "pmod", 4, "NULL"),
        (false, "pmod", 5, "1"),
];

fn session(ansi: bool) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set(
            "spark.sql.ansi.enabled",
            if ansi { "true" } else { "false" },
        )
        .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("s", DataType::Utf8, true),
        Field::new("l", DataType::Int64, true),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(0..6)),
        Arc::new(StringArray::from_iter(ROWS.iter().map(|r| r.0))),
        Arc::new(Int64Array::from_iter(ROWS.iter().map(|r| r.1))),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    s
}

/// The answer of `expr` over row `k`, in the notation of [`SPARK`].
fn answer(s: &Session, expr: &str, k: i32) -> String {
    let sql = format!("SELECT CAST({expr} AS STRING) FROM t WHERE k = {k}");
    match futures::executor::block_on(async { s.sql(&sql).await?.collect().await }) {
        Ok(batches) => {
            let a = batches[0].column(0);
            let a = a.as_string_view();
            if a.is_null(0) {
                "NULL".to_owned()
            } else {
                a.value(0).to_owned()
            }
        }
        Err(e) => format!("ERR:{}", e.error_class().name()),
    }
}

#[test]
fn division_follows_sparks_operand_order() {
    let mut wrong = Vec::new();
    for ansi in [true, false] {
        let s = session(ansi);
        for (op, expr) in OPS {
            for k in 0..6 {
                let want = SPARK
                    .iter()
                    .find(|(a, o, kk, _)| *a == ansi && *o == op && *kk == k)
                    .map(|r| r.3)
                    .unwrap();
                let got = answer(&s, expr, k);
                if got != want {
                    wrong.push(format!("ansi={ansi} {op} row {k}: Spark {want}, got {got}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(SPARK.len(), 2 * OPS.len() * ROWS.len());
}
