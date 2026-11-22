// SPDX-License-Identifier: Apache-2.0
//! `FROM UNNEST(arrays) [WITH ORDINALITY]` (Spark 4.3): rows, schema nullability and one evaluation per array.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn run(sql: &str) -> Vec<arrow_array::RecordBatch> {
    let s = Session::builder().build().expect("session");
    block_on(async { s.sql(sql).await?.collect().await }).expect("runs")
}

fn rows(sql: &str) -> usize {
    run(sql)
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum()
}

#[test]
fn ordinality_is_not_nullable_and_col_follows_the_arrays_own_nullability() {
    let batches = run("select * from unnest(array(1, 2)) with ordinality");
    let schema = batches[0].schema();
    assert_eq!(schema.field(0).name(), "col");
    assert!(!schema.field(0).is_nullable(), "{schema:?}");
    assert_eq!(schema.field(1).name(), "ordinality");
    assert!(!schema.field(1).is_nullable(), "{schema:?}");
    let with_null = run("select * from unnest(array(1, null)) with ordinality");
    assert!(with_null[0].schema().field(0).is_nullable());
}

#[test]
fn a_nondeterministic_argument_is_evaluated_once() {
    // Two evaluations of the slice length would give a values array and a positions array of different
    // lengths, so some rows would hold a NULL `col` or a NULL `ordinality`.
    for _ in 0..20 {
        let sql = "select count(*) c, count(col) v, count(ordinality) o from \
                   unnest(slice(array(1, 2, 3), 1, cast(rand() * 3 as int) + 1)) with ordinality";
        let b = run(sql);
        let col = |i: usize| {
            arrow_array::cast::as_primitive_array::<arrow_array::types::Int64Type>(b[0].column(i))
                .value(0)
        };
        assert_eq!((col(0), col(0)), (col(1), col(2)), "{b:?}");
    }
    assert_eq!(
        rows("select * from unnest(array(1, 2), array('a')) with ordinality"),
        2
    );
}
