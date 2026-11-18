// SPDX-License-Identifier: Apache-2.0
//! A literal query vector is read once, not copied once per row (P3-T077 section 8.4): the heap a distance
//! query adds on top of its table stays far below one broadcast of the query (`rows * dim * 4` bytes).
#![cfg(test)]

use std::alloc::System;
use std::sync::Arc;

use arrow_array::builder::{Float32Builder, ListBuilder};
use arrow_array::{ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_alloc::heap::{self, Counting};
use krishiv_session::Session;

mod vector_support;
use vector_support::random_vector;

#[global_allocator]
static HEAP: Counting<System> = Counting(System);

const ROWS: i32 = 8_192;
const DIM: usize = 768;

/// Fails today: `array(a, b, ...)` (`krishiv-functions/src/scalar/array.rs`) broadcasts each literal element to the
/// batch's rows, so a written-out `array(...)` query vector costs about half a broadcast (13 MB of 25 MB here) per
/// batch before the distance kernel sees it. A column or parameter query vector does not (the kernel reads a literal
/// at row 0). Fix: fold `array(literals)` to one literal row; then drop the `ignore`.
#[test]
#[ignore = "array(literals) is rebuilt per batch; see the doc comment"]
fn a_literal_query_vector_is_not_copied_per_row() {
    let field = Arc::new(Field::new("element", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("v", DataType::List(Arc::clone(&field)), true),
    ]));
    let mut vectors = ListBuilder::new(Float32Builder::new()).with_field(field);
    for i in 0..ROWS {
        for x in random_vector(DIM, 7000 + u64::try_from(i).unwrap(), 0, 1.0) {
            vectors.values().append_option(x);
        }
        vectors.append(true);
    }
    let ids: ArrayRef = Arc::new(Int32Array::from_iter_values(0..ROWS));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![ids, Arc::new(vectors.finish())]).unwrap();
    let s = Session::new().unwrap();
    s.register_batches("vt", schema, vec![batch]).unwrap();
    let items: Vec<String> = random_vector(DIM, 8000, 1, 1.0)
        .iter()
        .map(|x| format!("CAST({:?} AS FLOAT)", f64::from(x.unwrap())))
        .collect();
    let sql = format!(
        "SELECT sum(vector_l2_distance(v, array({}))) FROM vt",
        items.join(", ")
    );
    // Warm up (plan caches, thread pools), then measure the high-water mark above the live heap.
    block_on(async { s.sql(&sql).await?.collect().await }).unwrap();
    let base = heap::live();
    heap::reset_peak();
    block_on(async { s.sql(&sql).await?.collect().await }).unwrap();
    let added = heap::peak().saturating_sub(base);
    let broadcast = usize::try_from(ROWS).unwrap() * DIM * 4;
    assert!(
        added < broadcast / 4,
        "the query added {added} bytes at its peak; one broadcast of the query is {broadcast}"
    );
}
