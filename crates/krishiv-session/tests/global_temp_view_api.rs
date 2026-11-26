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

/// Spark's `Dataset.createTempView` parses the name as a multipart identifier (Spark v4.3.0
/// `classic/Dataset.scala:1679-1712`): backquotes are stripped, two parts are `TEMP_VIEW_NAME_TOO_MANY_NAME_PARTS`,
/// an unparsable name is `_LEGACY_ERROR_TEMP_1321`.
#[tokio::test]
async fn a_view_name_is_parsed_as_spark_does() {
    let s = Session::new().expect("session");
    let df = s.sql("SELECT * FROM range(3)").await.expect("sql");
    df.create_or_replace_temp_view("`qv`")
        .await
        .expect("quoted");
    assert_eq!(count(&s, "qv").await, 3);
    df.create_or_replace_global_temp_view("`g`")
        .await
        .expect("quoted global");
    assert_eq!(count(&s, "global_temp.g").await, 3);
    for name in ["db.v", "global_temp.v"] {
        let e = df.create_temp_view(name).await.expect_err(name);
        assert_eq!(e.error_class().name(), "TEMP_VIEW_NAME_TOO_MANY_NAME_PARTS");
        assert!(e.to_string().contains(name), "{e}");
    }
    let e = df.create_temp_view("a..b").await.expect_err("bad");
    assert_eq!(e.error_class().name(), "_LEGACY_ERROR_TEMP_1321");
}

/// QA2-8: a global view over a read by path is read as `global_temp.name` (it was stored as a
/// file table that only a bare name found).
#[tokio::test]
async fn a_global_view_over_a_file_read_is_read_as_global_temp_name() {
    let dir = tempfile::tempdir().expect("dir");
    let path = dir.path().join("people.json");
    std::fs::write(
        &path,
        "{\"age\":30,\"name\":\"Andy\"}\n{\"age\":19,\"name\":\"Justin\"}\n",
    )
    .expect("write");
    let s = Session::new().expect("session");
    let df = s.read().format("json").load([path.to_str().expect("utf8")]);
    df.create_global_temp_view("people").await.expect("create");
    assert_eq!(count(&s, "global_temp.people").await, 2);
    assert!(s.sql("SELECT * FROM people").await.is_err());
}
