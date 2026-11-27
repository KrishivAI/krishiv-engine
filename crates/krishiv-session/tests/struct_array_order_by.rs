// SPDX-License-Identifier: Apache-2.0
//! `ORDER BY` over a struct or an array, as Spark orders them: a struct field by field, an array
//! element by element then by length, a NULL field or element the smallest, NaN the largest float.
//! `NULLS FIRST/LAST` that differs from the direction's default moves only the key's own NULL in
//! Spark, which the row encoding cannot say for a nested key: refused, not guessed.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;

fn ids(s: &Session, sql: &str) -> Vec<i32> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_primitive::<Int32Type>();
            (0..c.len()).map(|i| c.value(i)).collect::<Vec<_>>()
        })
        .collect()
}

fn session() -> Session {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE TEMPORARY VIEW st AS SELECT id, a FROM VALUES \
         (1, named_struct('x', 2, 'y', 'b')), (2, named_struct('x', 1, 'y', 'z')), \
         (3, named_struct('x', 1, 'y', 'a')), (4, named_struct('x', CAST(NULL AS INT), 'y', 'a')), \
         (5, CAST(NULL AS STRUCT<x: INT, y: STRING>)) AS st(id, a)",
        "CREATE TEMPORARY VIEW ar AS SELECT id, a FROM VALUES \
         (1, array(2, 1)), (2, array(1, 5)), (3, array(1)), (4, CAST(NULL AS ARRAY<INT>)), \
         (5, array(CAST(NULL AS INT), 9)), (6, CAST(array() AS ARRAY<INT>)) AS ar(id, a)",
        "CREATE TEMPORARY VIEW fl AS SELECT id, a FROM VALUES \
         (1, named_struct('d', CAST('NaN' AS DOUBLE))), (2, named_struct('d', 1.0D)), \
         (3, named_struct('d', -0.0D)), (4, named_struct('d', 0.0D)), \
         (5, named_struct('d', -CAST('NaN' AS DOUBLE))) AS fl(id, a)",
    ] {
        block_on(async { s.sql(sql).await?.collect().await })
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    s
}

#[test]
fn a_struct_orders_field_by_field_with_a_null_field_smallest() {
    let s = session();
    assert_eq!(ids(&s, "SELECT id FROM st ORDER BY a, id"), [5, 4, 3, 2, 1]);
    assert_eq!(
        ids(&s, "SELECT id FROM st ORDER BY a DESC, id"),
        [1, 2, 3, 4, 5]
    );
}

#[test]
fn an_array_orders_element_by_element_then_by_length() {
    let s = session();
    assert_eq!(
        ids(&s, "SELECT id FROM ar ORDER BY a, id"),
        [4, 6, 5, 3, 2, 1]
    );
    assert_eq!(
        ids(&s, "SELECT id FROM ar ORDER BY a DESC, id"),
        [1, 2, 3, 5, 6, 4]
    );
}

#[test]
fn a_float_in_a_struct_orders_nan_last_and_negative_zero_as_zero() {
    let s = session();
    // 1.0 is 2; the zeros (3, 4) tie and so keep id order; both NaNs (1, 5) are the largest.
    assert_eq!(ids(&s, "SELECT id FROM fl ORDER BY a, id"), [3, 4, 2, 1, 5]);
}

#[test]
fn a_nested_key_with_a_non_default_null_placement_is_refused() {
    let s = session();
    let err = block_on(async {
        s.sql("SELECT id FROM st ORDER BY a NULLS LAST")
            .await?
            .collect()
            .await
    })
    .unwrap_err()
    .to_string();
    assert!(err.contains("KRISHIV_UNSUPPORTED_FEATURE"), "{err}");
}
