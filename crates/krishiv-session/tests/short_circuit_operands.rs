// SPDX-License-Identifier: Apache-2.0
//! The compiler's choice of how `AND`, `OR` and `BETWEEN` run their last
//! operand, seen through SQL: an operand that may raise runs only on the rows
//! the first one leaves open (Spark's `And`/`Or` short circuit; `BETWEEN` is
//! `v >= lo AND v <= hi`), and one that cannot raise (ANSI off) runs on the
//! whole batch with the same answer. `range(0, 6, 1, 1)` is one batch that mixes
//! decided and open rows; `10 / k` raises `DIVIDE_BY_ZERO` at `k = 0` with
//! ANSI on and is NULL with it off.
#![cfg(test)]

use arrow_array::cast::AsArray;
use krishiv_session::Session;

/// `(predicate, its values for k = 0..5)`, the same with ANSI on and off.
/// The predicates are projected, not filtered on: a `WHERE` splits its
/// conjuncts before they reach the compiled `AND`.
const CASES: [(&str, &str); 4] = [
    ("k <> 0 AND 10 / k >= 2", "f t t t t t"),
    ("k = 0 OR 10 / k >= 3", "t t t t f f"),
    ("k BETWEEN 1 AND 10 / k", "f t t t f f"),
    ("NOT (k = 0 OR 10 / k < 3)", "f t t t f f"),
];

/// The values of `predicate` over one batch of `k = 0..5`, or
/// `ERR:<error class>`.
fn run(ansi: bool, predicate: &str) -> String {
    let s = Session::new().unwrap();
    s.conf()
        .set(
            "spark.sql.ansi.enabled",
            if ansi { "true" } else { "false" },
        )
        .unwrap();
    let sql = format!("SELECT {predicate} FROM (SELECT id AS k FROM range(0, 6, 1, 1)) ORDER BY k");
    match futures::executor::block_on(async { s.sql(&sql).await?.collect().await }) {
        Ok(batches) => {
            let mut out = Vec::new();
            for b in &batches {
                let a = b.column(0).as_boolean();
                out.extend((0..a.len()).map(|i| if a.value(i) { "t" } else { "f" }));
            }
            out.join(" ")
        }
        Err(e) => format!("ERR:{}", e.error_class().name()),
    }
}

#[test]
fn a_raising_last_operand_runs_only_on_open_rows() {
    for (predicate, want) in CASES {
        assert_eq!(run(true, predicate), want, "{predicate}");
    }
}

#[test]
fn a_last_operand_that_cannot_raise_gives_the_same_rows() {
    for (predicate, want) in CASES {
        assert_eq!(run(false, predicate), want, "{predicate}");
    }
}

/// The control: unguarded, the operand does raise on these rows.
#[test]
fn the_unguarded_operand_raises() {
    assert_eq!(run(true, "10 / k >= 2"), "ERR:DIVIDE_BY_ZERO");
}
