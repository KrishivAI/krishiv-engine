// SPDX-License-Identifier: Apache-2.0
//! A user-defined type column through the engine (P3-T079): it keeps its type in the result schema, in a Parquet
//! write and read, and `ORDER BY`, `GROUP BY`, `=` and a join key run on its storage type (as they do in Spark).

#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;
use krishiv_types::{DataType, UdtType};

/// `ExamplePoint`: a UDT over `int` (its storage keeps the test small; the type is what is under test).
fn point() -> DataType {
    DataType::Udt(Arc::new(UdtType::jvm(
        DataType::Integer,
        "org.apache.spark.sql.test.ExamplePointUDT".to_owned(),
        Some("pyspark.testing.objects.ExamplePointUDT".to_owned()),
    )))
}

fn session() -> Session {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", arrow_schema::DataType::Int32, false),
        point().to_arrow_field("p", true),
    ]));
    let ids: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
    let ps: ArrayRef = Arc::new(Int32Array::from(vec![Some(30), Some(10), None, Some(10)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, ps]).unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    s
}

fn collect(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn ids(batches: &[RecordBatch]) -> Vec<i32> {
    batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect()
}

#[test]
fn a_udt_column_keeps_its_type_in_the_result_schema() {
    let s = session();
    for sql in [
        "select * from t",
        "select id, p from t where id > 1 order by id",
    ] {
        let batches = collect(&s, sql);
        let field = batches[0].schema().field_with_name("p").unwrap().clone();
        assert_eq!(
            DataType::from_arrow_field(&field).unwrap(),
            point(),
            "{sql}"
        );
    }
}

#[test]
fn order_by_group_by_equality_and_a_join_key_run_on_the_storage_type() {
    let s = session();
    assert_eq!(
        ids(&collect(&s, "select id from t order by p, id")),
        [3, 2, 4, 1]
    );
    assert_eq!(
        ids(&collect(
            &s,
            "select id from t where p = (select max(p) from t) order by id"
        )),
        [1]
    );
    let groups = collect(&s, "select p, count(*) c from t group by p order by p");
    let counts: Vec<i64> = groups
        .iter()
        .flat_map(|b| {
            b.column(1)
                .as_primitive::<arrow_array::types::Int64Type>()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(counts, [1, 2, 1]);
    let joined = collect(
        &s,
        "select a.id from t a join t b on a.p = b.p where a.id < b.id order by a.id",
    );
    assert_eq!(ids(&joined), [2]);
}

#[test]
fn a_udt_column_keeps_its_type_through_a_parquet_write_and_read() {
    let s = session();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("p.parquet");
    block_on(async {
        s.sql("select * from t")
            .await?
            .write()
            .mode("overwrite")
            .parquet(path.to_str().unwrap())
            .await
    })
    .unwrap();
    let back =
        block_on(async { s.read().parquet([path.to_str().unwrap()]).collect().await }).unwrap();
    let field = back[0].schema().field_with_name("p").unwrap().clone();
    assert_eq!(DataType::from_arrow_field(&field).unwrap(), point());
    assert_eq!(back.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);
}
