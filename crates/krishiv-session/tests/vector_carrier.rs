// SPDX-License-Identifier: Apache-2.0
//! A column handed in as a `FixedSizeList` of floats runs as a vector carrier (design P3-T075): expressions and the
//! caller see Spark's `array<float>` (a plain list), a bare column passes through, and a fixed-size list that is not
//! a vector (other elements, too long) is an ordinary list from the start.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int32Type};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

fn vectors() -> (Arc<Schema>, RecordBatch) {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let values = Float32Array::from(vec![
        1.0, 2.0, 3.0, //
        0.0, 0.0, 0.0, // the null row's slot
        7.0, 8.0, 9.0,
    ]);
    let v: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(
            Arc::clone(&item),
            3,
            Arc::new(values),
            Some(NullBuffer::from(vec![true, false, true])),
        )
        .unwrap(),
    );
    let id: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("v", DataType::FixedSizeList(item, 3), true),
    ]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![id, v]).unwrap();
    (schema, batch)
}

fn session() -> Session {
    let s = Session::new().unwrap();
    let (schema, batch) = vectors();
    s.register_batches("t", schema, vec![batch]).unwrap();
    s
}

fn rows(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap()
}

/// What a caller reads is Spark's plain list, with the same values and the null row.
#[test]
fn a_caller_reads_a_vector_column_as_a_plain_list() {
    let s = session();
    let df = block_on(s.sql("SELECT id, v FROM t ORDER BY id")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert!(
        matches!(schema.field(1).data_type(), DataType::List(f) if f.data_type() == &DataType::Float32),
        "{schema:?}"
    );
    let batches = block_on(df.collect()).unwrap();
    let v = batches[0].column(1).as_list::<i32>();
    assert!(v.is_valid(0) && v.is_null(1) && v.is_valid(2));
    assert_eq!(
        v.value(2).as_primitive::<Float32Type>().values(),
        &[7.0, 8.0, 9.0]
    );
}

/// Functions over arrays read a vector column as they read a list: the result is the one a list column gives.
#[test]
fn array_functions_over_a_vector_column_match_a_list_column() {
    let s = session();
    let (_, batch) = vectors();
    let plain = krishiv_exec::carrier::recarried_array(
        batch.column(1),
        &DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("v", plain.data_type().clone(), true),
    ]));
    let as_list = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::clone(batch.column(0)), plain],
    )
    .unwrap();
    s.register_batches("l", schema, vec![as_list]).unwrap();
    for sql in [
        "SELECT size(v) FROM {} ORDER BY id",
        "SELECT array_max(v) FROM {} ORDER BY id",
        "SELECT v[1] FROM {} ORDER BY id",
        "SELECT id FROM {} WHERE array_contains(v, 8.0) ORDER BY id",
        "SELECT id, explode(v) FROM {} ORDER BY id, 2",
        "SELECT count(v), max(size(v)) FROM {}",
        "SELECT id, v FROM {} ORDER BY id DESC",
    ] {
        let render = |table: &str| {
            let batches = rows(&s, &sql.replace("{}", table));
            batches
                .iter()
                .map(|b| format!("{:?}", b.columns()))
                .collect::<Vec<_>>()
        };
        assert_eq!(render("t"), render("l"), "{sql}");
    }
}

/// A fixed-size list of anything but floats, or too long to be a vector, enters as a plain list.
#[test]
fn a_fixed_size_list_that_is_not_a_vector_is_an_ordinary_list() {
    let s = Session::new().unwrap();
    let item = Arc::new(Field::new("item", DataType::Int32, true));
    let v: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(
            Arc::clone(&item),
            2,
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            None,
        )
        .unwrap(),
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::FixedSizeList(item, 2),
        false,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![v]).unwrap();
    s.register_batches("ints", schema, vec![batch]).unwrap();
    let out = rows(&s, "SELECT v FROM ints");
    let list = out[0].column(0).as_list::<i32>();
    assert_eq!(list.value(1).as_primitive::<Int32Type>().values(), &[3, 4]);
}
