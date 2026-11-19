// SPDX-License-Identifier: Apache-2.0
//! `Session::register_batch_source`: a table whose rows are asked for again for every plan
//! (the Python Data Source's reader, design P3-T040 §3.3), also through a view over it.
#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::{BatchSource, Session};

fn count(session: &Session, sql: &str) -> i64 {
    let rows = block_on(async { session.sql(sql).await?.collect().await }).unwrap();
    let batch = &rows[0];
    arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(batch.column(0))
        .value(0)
}

#[test]
fn rows_are_asked_for_again_on_every_plan() {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    let calls = Arc::new(AtomicI32::new(0));
    let (s, c) = (Arc::clone(&schema), Arc::clone(&calls));
    let source: BatchSource = Arc::new(move || {
        let n = c.fetch_add(1, Ordering::SeqCst) + 1;
        let batch = RecordBatch::try_new(
            Arc::clone(&s),
            vec![Arc::new(Int32Array::from((0..n).collect::<Vec<_>>()))],
        )
        .map_err(|e| {
            KrishivError::from(krishiv_common::error::InternalError {
                message: e.to_string(),
            })
        })?;
        Ok(vec![batch])
    });
    session
        .register_batch_source("grows", schema, source)
        .unwrap();
    let first = count(&session, "SELECT count(*) FROM grows");
    let second = count(&session, "SELECT count(*) FROM grows");
    assert!(
        second > first,
        "a second action saw {second} rows after {first}"
    );
    // a view over it keeps reading it
    block_on(async {
        session
            .sql("CREATE TEMPORARY VIEW v AS SELECT * FROM grows")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert!(count(&session, "SELECT count(*) FROM v") > 0);
}

#[test]
fn a_failing_source_fails_the_query() {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    let source: BatchSource = Arc::new(|| {
        Err(krishiv_common::error::InternalError {
            message: "no rows today".to_owned(),
        }
        .into())
    });
    session
        .register_batch_source("boom", schema, source)
        .unwrap();
    let err =
        block_on(async { session.sql("SELECT * FROM boom").await?.collect().await }).unwrap_err();
    assert!(err.to_string().contains("no rows today"), "{err}");
}

#[test]
fn a_batch_of_another_schema_is_refused() {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    let other = Arc::new(Schema::new(vec![Field::new("z", DataType::Int32, true)]));
    let source: BatchSource = Arc::new(move || {
        Ok(vec![
            RecordBatch::try_new(
                Arc::clone(&other),
                vec![Arc::new(Int32Array::from(vec![1]))],
            )
            .unwrap(),
        ])
    });
    session
        .register_batch_source("odd", schema, source)
        .unwrap();
    assert!(block_on(async { session.sql("SELECT * FROM odd").await?.collect().await }).is_err());
}

#[test]
fn a_reserved_name_is_refused() {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
    let source: BatchSource = Arc::new(|| Ok(Vec::new()));
    assert!(
        session
            .register_batch_source("__krishiv_snapshot_x", schema, source)
            .is_err()
    );
}
