// SPDX-License-Identifier: Apache-2.0
//! Behaviour Spark 4.3 fixed as bugs, held in every profile (research/2026-11-03-spark-4.3-migration.md items 11 and
//! 18). The expectations follow the migration notes; they were not run against a Spark 4.3 oracle (none was up).
//! Items 22 (failing partition predicates) and 15 (per-reference `WITH (...)` options) cannot be tested yet: see
//! ops/queue-sonnet.md.
#![cfg(test)]

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// A non-deterministic predicate is evaluated once per row, wherever it sits among the conjuncts: a scan that also
/// evaluated it would square the pass rate (0.5 -> 0.25).
#[test]
fn a_nondeterministic_predicate_is_evaluated_once_per_row() {
    let d = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", d.path().to_str().unwrap())
        .unwrap();
    run(
        &s,
        "CREATE TABLE r USING parquet AS SELECT id AS v FROM range(20000)",
    );
    for sql in [
        "SELECT count(*) FROM r WHERE rand() < 0.5",
        "SELECT count(*) FROM r WHERE rand() < 0.5 AND v >= 0",
        "SELECT count(*) FROM r WHERE v >= 0 AND rand() < 0.5",
    ] {
        let n = run(&s, sql)[0]
            .column(0)
            .as_primitive::<Int64Type>()
            .value(0);
        assert!((9_000..=11_000).contains(&n), "{sql}: {n} of 20000 rows");
    }
}

/// With a generator in the SELECT list, HAVING removes groups before the window function sees them: the window's
/// sum covers the groups HAVING kept (20 + 30), not all three.
#[test]
fn having_applies_before_window_functions_with_a_generator() {
    let s = Session::new().unwrap();
    let b = run(
        &s,
        "SELECT k, explode(array(1, 2)) AS e, sum(sum(v)) OVER () AS w \
         FROM VALUES (1, 10), (2, 20), (3, 30) AS t(k, v) GROUP BY k HAVING sum(v) > 10 ORDER BY k, e",
    );
    let b = &b[0];
    let k: Vec<i32> = b.column(0).as_primitive::<Int32Type>().values().to_vec();
    let w: Vec<i64> = b.column(2).as_primitive::<Int64Type>().values().to_vec();
    assert_eq!(k, [2, 2, 3, 3]);
    assert_eq!(w, [50, 50, 50, 50]);
}
