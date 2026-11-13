// SPDX-License-Identifier: Apache-2.0
//! Without the `delta` feature, `delta` is not a data source: every Delta read and DDL fails as
//! before P3-T021 PR-08b part 2: Spark's `DATA_SOURCE_NOT_FOUND` (42K02) naming it, or the
//! refusal that came first before (a data-source view without declared columns).
#![cfg(all(test, not(feature = "delta")))]

use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn not_found(e: &KrishivError) {
    assert_eq!(e.error_class(), ErrorClass::DataSourceNotFound, "{e}");
    assert_eq!(e.sql_state(), Some("42K02"));
}

#[test]
fn delta_is_not_a_data_source_without_the_feature() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    let e = block_on(s.read().format("delta").load([&path]).collect()).unwrap_err();
    not_found(&e);
    for sql in [
        format!("CREATE TABLE d USING delta LOCATION '{path}'"),
        "CREATE TABLE c USING delta AS SELECT 1 AS id".to_owned(),
    ] {
        let e = block_on(async { s.sql(&sql).await?.collect().await }).unwrap_err();
        not_found(&e);
    }
    // A data-source view without declared columns is refused before its source is looked up,
    // as it was before the feature existed.
    let e = block_on(async {
        s.sql(&format!(
            "CREATE TEMPORARY VIEW v USING delta OPTIONS (path '{path}')"
        ))
        .await?
        .collect()
        .await
    })
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    // `delta`.`path` stays a table name: no such table.
    let e = block_on(async {
        s.sql(&format!("SELECT * FROM delta.`{path}`"))
            .await?
            .collect()
            .await
    })
    .unwrap_err();
    assert_ne!(e.error_class(), ErrorClass::InternalError, "{e}");
}
