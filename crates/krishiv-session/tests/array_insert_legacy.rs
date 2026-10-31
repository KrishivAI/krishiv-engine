// SPDX-License-Identifier: Apache-2.0
//! `array_insert` under `spark.sql.legacy.negativeIndexInArrayInsert`: every
//! row below is what a Spark 4.2.0 run returned with the switch on and off.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;

fn insert(s: &Session, call: &str) -> Option<Vec<Option<i32>>> {
    let df = block_on(s.sql(&format!("SELECT {call}"))).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let col = batches[0].column(0);
    if col.is_null(0) {
        return None;
    }
    let items = col.as_list::<i32>().value(0);
    let items = items.as_primitive::<Int32Type>();
    Some(items.iter().collect())
}

fn set(s: &Session, on: bool) {
    block_on(s.sql(&format!(
        "SET spark.sql.legacy.negativeIndexInArrayInsert={on}"
    )))
    .unwrap();
}

#[test]
fn a_negative_position_follows_the_switch() {
    let s = Session::new().unwrap();
    type Items = &'static [Option<i32>];
    let cases: [(&str, Items, Items); 8] = [
        (
            "array_insert(array(1,3,4), -1, 9)",
            &[Some(1), Some(3), Some(9), Some(4)],
            &[Some(1), Some(3), Some(4), Some(9)],
        ),
        (
            "array_insert(array(1,3,4), -2, 2)",
            &[Some(1), Some(2), Some(3), Some(4)],
            &[Some(1), Some(3), Some(2), Some(4)],
        ),
        (
            "array_insert(array(1,3,4), -3, 9)",
            &[Some(9), Some(1), Some(3), Some(4)],
            &[Some(1), Some(9), Some(3), Some(4)],
        ),
        (
            "array_insert(array(1,3,4), -4, 9)",
            &[Some(9), None, Some(1), Some(3), Some(4)],
            &[Some(9), Some(1), Some(3), Some(4)],
        ),
        (
            "array_insert(array(2,3,NULL,4), -5, 1)",
            &[Some(1), None, Some(2), Some(3), None, Some(4)],
            &[Some(1), Some(2), Some(3), None, Some(4)],
        ),
        ("array_insert(array(), -1, 9)", &[Some(9), None], &[Some(9)]),
        (
            "array_insert(array(), -3, 9)",
            &[Some(9), None, None, None],
            &[Some(9), None, None],
        ),
        (
            "array_insert(array(1), -3, 9)",
            &[Some(9), None, None, Some(1)],
            &[Some(9), None, Some(1)],
        ),
    ];
    for on in [true, false] {
        set(&s, on);
        for (call, legacy, default) in cases {
            let want = if on { legacy } else { default };
            assert_eq!(
                insert(&s, call).as_deref(),
                Some(want),
                "{call} legacy={on}"
            );
        }
    }
    // Positive positions, a NULL array and the shown name do not change.
    set(&s, true);
    assert_eq!(
        insert(&s, "array_insert(array(1,3,4), 5, 9)").as_deref(),
        Some(&[Some(1), Some(3), Some(4), None, Some(9)][..])
    );
    assert_eq!(
        insert(&s, "array_insert(cast(NULL as array<int>), -1, 9)"),
        None
    );
    let df = block_on(s.sql("SELECT array_insert(array(1,3,4), -2, 2)")).unwrap();
    assert_eq!(
        block_on(df.schema()).unwrap().field(0).name(),
        "array_insert(array(1, 3, 4), -2, 2)"
    );
    let df = block_on(s.sql("SELECT array_insert(array(1), 0, 9)")).unwrap();
    assert_eq!(
        block_on(df.collect()).unwrap_err().error_class().name(),
        "INVALID_INDEX_OF_ZERO"
    );
}
