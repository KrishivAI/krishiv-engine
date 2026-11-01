// SPDX-License-Identifier: Apache-2.0
//! A decimal of precision 18 or less runs on `Decimal64`, wider ones on
//! `Decimal128` (design P2-T160). Callers hand in either and always read
//! `Decimal128`, and queries that mix the two carriers give Spark's answers.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Decimal128Type;
use arrow_array::{ArrayRef, Decimal64Array, Decimal128Array, ListArray, RecordBatch};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

fn dec128(p: u8, s: i8, values: &[i128]) -> ArrayRef {
    Arc::new(
        Decimal128Array::from_iter_values(values.iter().copied())
            .with_precision_and_scale(p, s)
            .unwrap(),
    )
}

fn dec64(p: u8, s: i8, values: &[i64]) -> ArrayRef {
    Arc::new(
        Decimal64Array::from_iter_values(values.iter().copied())
            .with_precision_and_scale(p, s)
            .unwrap(),
    )
}

fn register(s: &Session, name: &str, column: ArrayRef) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "d",
        column.data_type().clone(),
        false,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column]).unwrap();
    s.register_batches(name, schema, vec![batch]).unwrap();
}

/// The first column of `sql`'s rows, unscaled, with its type.
fn unscaled(s: &Session, sql: &str) -> (DataType, Vec<i128>) {
    let df = block_on(s.sql(sql)).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let t = schema.field(0).data_type().clone();
    let mut values = Vec::new();
    for b in &batches {
        assert_eq!(b.column(0).data_type(), &t, "{sql}");
        values.extend(b.column(0).as_primitive::<Decimal128Type>().values());
    }
    values.sort_unstable();
    (t, values)
}

#[test]
fn callers_read_decimal128_whichever_carrier_they_handed_in() {
    let s = Session::new().unwrap();
    register(&s, "narrow128", dec128(10, 2, &[150, 275]));
    register(&s, "narrow64", dec64(10, 2, &[150, 275]));
    register(&s, "wide", dec128(20, 2, &[150, 275]));
    for table in ["narrow128", "narrow64"] {
        assert_eq!(
            unscaled(&s, &format!("SELECT d FROM {table}")),
            (DataType::Decimal128(10, 2), vec![150, 275]),
        );
    }
    assert_eq!(
        unscaled(&s, "SELECT d FROM wide"),
        (DataType::Decimal128(20, 2), vec![150, 275]),
    );
}

#[test]
fn an_empty_result_has_the_client_type_too() {
    let s = Session::new().unwrap();
    register(&s, "t", dec64(10, 2, &[150]));
    assert_eq!(
        unscaled(&s, "SELECT d FROM t WHERE d < 0"),
        (DataType::Decimal128(10, 2), vec![]),
    );
}

#[test]
fn a_nested_decimal_is_read_on_decimal128() {
    let s = Session::new().unwrap();
    let item = Arc::new(Field::new("item", DataType::Decimal128(10, 2), true));
    let list = ListArray::new(
        Arc::clone(&item),
        OffsetBuffer::from_lengths([2]),
        dec128(10, 2, &[150, 275]),
        None,
    );
    register(&s, "t", Arc::new(list));
    let df = block_on(s.sql("SELECT d, d[0] AS first FROM t")).unwrap();
    let batches = block_on(df.collect()).unwrap();
    assert_eq!(batches[0].column(0).data_type(), &DataType::List(item));
    assert_eq!(
        batches[0].column(1).data_type(),
        &DataType::Decimal128(10, 2)
    );
}

#[test]
fn queries_that_mix_the_carriers() {
    let s = Session::new().unwrap();
    register(&s, "a", dec64(10, 2, &[150, 275, 300]));
    register(&s, "b", dec128(20, 2, &[150, 300, 400]));
    let wide = DataType::Decimal128(20, 2);
    // Join and existence keys.
    assert_eq!(
        unscaled(&s, "SELECT a.d FROM a JOIN b ON a.d = b.d"),
        (DataType::Decimal128(10, 2), vec![150, 300]),
    );
    assert_eq!(
        unscaled(&s, "SELECT d FROM a WHERE d IN (SELECT d FROM b)"),
        (DataType::Decimal128(10, 2), vec![150, 300]),
    );
    // Set operations.
    assert_eq!(
        unscaled(&s, "SELECT d FROM a UNION ALL SELECT d FROM b"),
        (wide.clone(), vec![150, 150, 275, 300, 300, 400]),
    );
    assert_eq!(
        unscaled(&s, "SELECT d FROM a UNION SELECT d FROM b"),
        (wide.clone(), vec![150, 275, 300, 400]),
    );
    assert_eq!(
        unscaled(&s, "SELECT d FROM a INTERSECT SELECT d FROM b"),
        (wide.clone(), vec![150, 300]),
    );
    assert_eq!(
        unscaled(&s, "SELECT d FROM a EXCEPT SELECT d FROM b"),
        (wide.clone(), vec![275]),
    );
    // Expressions with a common type.
    assert_eq!(
        unscaled(
            &s,
            "SELECT coalesce(a.d, b.d) FROM a FULL JOIN b ON a.d = b.d"
        ),
        (wide.clone(), vec![150, 275, 300, 400]),
    );
    assert_eq!(
        unscaled(
            &s,
            "SELECT CASE WHEN a.d > 200 THEN a.d ELSE b.d END FROM a JOIN b ON a.d = b.d"
        ),
        (wide, vec![150, 300]),
    );
    assert_eq!(
        unscaled(&s, "SELECT d FROM a WHERE d IN (1.50, 3.000)"),
        (DataType::Decimal128(10, 2), vec![150, 300]),
    );
    // Arithmetic whose result moves from one carrier to the other.
    assert_eq!(
        unscaled(&s, "SELECT d * d FROM a WHERE d = 1.50"),
        (DataType::Decimal128(21, 4), vec![22_500]),
    );
    assert_eq!(
        unscaled(&s, "SELECT sum(d) FROM a"),
        (DataType::Decimal128(20, 2), vec![725]),
    );
    assert_eq!(
        unscaled(&s, "SELECT avg(d) FROM a"),
        (DataType::Decimal128(14, 6), vec![2_416_667],),
    );
}

/// `hash` follows the Spark type: a long for precision 18 or less, its bytes
/// above, whichever carrier the value came in on.
#[test]
fn hash_follows_the_precision_not_the_carrier() {
    let s = Session::new().unwrap();
    register(&s, "a", dec64(10, 2, &[150]));
    register(&s, "b", dec128(10, 2, &[150]));
    let hash = |sql: &str| {
        let df = block_on(s.sql(sql)).unwrap();
        let b = block_on(df.collect()).unwrap();
        b[0].column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0)
    };
    let narrow = hash("SELECT hash(d) FROM a");
    assert_eq!(narrow, hash("SELECT hash(d) FROM b"));
    assert_eq!(narrow, hash("SELECT hash(CAST(1.50 AS DECIMAL(10,2)))"));
    assert_eq!(
        hash("SELECT hash(CAST(d AS DECIMAL(20,2))) FROM a"),
        hash("SELECT hash(CAST(1.50 AS DECIMAL(20,2)))"),
    );
    assert_ne!(narrow, hash("SELECT hash(CAST(d AS DECIMAL(20,2))) FROM a"));
}

/// Every file format writes and reads decimals of both carriers.
#[test]
fn file_formats_round_trip_both_carriers() {
    let s = Session::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    for format in ["parquet", "csv"] {
        for (name, ty) in [("narrow", "DECIMAL(10,2)"), ("wide", "DECIMAL(20,2)")] {
            let path = dir.path().join(format!("{format}_{name}"));
            let path = path.to_str().unwrap();
            let rows =
                format!("SELECT CAST(1.50 AS {ty}) AS d UNION ALL SELECT CAST(-2.75 AS {ty})");
            let df = block_on(s.sql(&rows)).unwrap();
            block_on(df.write().format(format).save(path))
                .unwrap_or_else(|e| panic!("write {format} {ty}: {e}"));
            if format == "csv" {
                // The reader's own tests read CSV on both carriers.
                let mut text = String::new();
                for file in std::fs::read_dir(path).unwrap() {
                    text += &std::fs::read_to_string(file.unwrap().path()).unwrap();
                }
                assert!(text.contains("1.50") && text.contains("-2.75"), "{text}");
                continue;
            }
            let table = format!("{format}_{name}");
            let batches = block_on(async {
                s.sql(&format!(
                    "CREATE TABLE {table} (d {ty}) USING {format} LOCATION '{path}'"
                ))
                .await?
                .collect()
                .await?;
                s.sql(&format!("SELECT d FROM {table}"))
                    .await?
                    .collect()
                    .await
            })
            .unwrap_or_else(|e| panic!("read {format} {ty}: {e}"));
            let mut got: Vec<i128> = batches
                .iter()
                .flat_map(|b| {
                    b.column(0)
                        .as_primitive::<Decimal128Type>()
                        .values()
                        .to_vec()
                })
                .collect();
            got.sort_unstable();
            assert_eq!(got, [-275, 150], "{format} {ty}");
        }
    }
}

/// A join on a narrow key publishes its runtime filter, and map higher-order
/// functions run on the query's carrier.
#[test]
fn narrow_join_keys_and_map_functions() {
    let s = Session::new().unwrap();
    register(&s, "a", dec64(10, 2, &[150, 275, 300]));
    register(&s, "b", dec64(10, 2, &[150, 300, 400]));
    assert_eq!(
        unscaled(&s, "SELECT a.d FROM a JOIN b ON a.d = b.d"),
        (DataType::Decimal128(10, 2), vec![150, 300]),
    );
    assert_eq!(
        unscaled(
            &s,
            "SELECT m['k'] FROM (SELECT transform_values(map('k', d), (k, v) -> v) AS m FROM a)"
        ),
        (DataType::Decimal128(10, 2), vec![150, 275, 300]),
    );
    assert_eq!(
        unscaled(
            &s,
            "SELECT m[d] FROM (SELECT d, map_filter(map(d, d), (k, v) -> k > 2.00) AS m FROM a) \
             WHERE d > 2.00"
        ),
        (DataType::Decimal128(10, 2), vec![275, 300]),
    );
    assert_eq!(
        unscaled(
            &s,
            "SELECT m[d] FROM (SELECT d, map_zip_with(map(d, d), map(d, d), (k, x, y) -> x + y) \
             AS m FROM a)"
        ),
        (DataType::Decimal128(11, 2), vec![300, 550, 600]),
    );
}
