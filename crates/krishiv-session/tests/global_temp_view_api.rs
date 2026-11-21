// SPDX-License-Identifier: Apache-2.0
//! `DataFrame::create_global_temp_view` / `create_or_replace_global_temp_view` (PySpark's
//! `createGlobalTempView`) and the request tags of `Session::with_request_tags`.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use krishiv_session::Session;

async fn count(s: &Session, from: &str) -> i64 {
    let batches = s
        .sql(&format!("SELECT count(*) AS n FROM {from}"))
        .await
        .expect("sql")
        .collect()
        .await
        .expect("collect");
    batches[0].column(0).as_primitive::<Int64Type>().value(0)
}

#[tokio::test]
async fn a_global_view_is_read_as_global_temp_name() {
    let s = Session::new().expect("session");
    let df = s.sql("SELECT * FROM range(3)").await.expect("sql");
    df.create_global_temp_view("gv").await.expect("create");
    assert_eq!(count(&s, "global_temp.gv").await, 3);
    // Not a local view.
    assert!(s.sql("SELECT * FROM gv").await.is_err());
}

#[tokio::test]
async fn creating_an_existing_global_view_fails_and_replace_replaces() {
    let s = Session::new().expect("session");
    let three = s.sql("SELECT * FROM range(3)").await.expect("sql");
    let five = s.sql("SELECT * FROM range(5)").await.expect("sql");
    three.create_global_temp_view("gv").await.expect("create");
    let err = five
        .create_global_temp_view("gv")
        .await
        .expect_err("exists");
    assert_eq!(
        err.error_class().name(),
        "TEMP_TABLE_OR_VIEW_ALREADY_EXISTS"
    );
    assert!(err.to_string().contains("`global_temp`.`gv`"), "{err}");
    five.create_or_replace_global_temp_view("gv")
        .await
        .expect("replace");
    assert_eq!(count(&s, "global_temp.gv").await, 5);
}

#[tokio::test]
async fn a_tagged_handle_runs_queries_and_a_closed_session_refuses() {
    let s = Session::new().expect("session");
    let tagged = s.with_request_tags(["etl"]).expect("handle");
    assert_eq!(count(&tagged, "range(4)").await, 4);
    assert!(s.interrupt_tag("etl").is_empty());
    s.close();
    assert!(s.with_request_tags(["etl"]).is_err());
}
