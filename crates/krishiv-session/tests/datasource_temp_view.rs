// SPDX-License-Identifier: Apache-2.0
//! `CREATE TEMPORARY VIEW name (cols) USING format [OPTIONS (...)]`: a lazy read of files, typed by
//! the declared columns, listed by `SHOW TABLES` and `DESCRIBE`.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn count(s: &Session, sql: &str) -> usize {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.collect())
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum()
}

#[test]
fn a_datasource_temp_view_is_created_listed_and_described() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TEMPORARY VIEW show_t3(e int) USING parquet")).unwrap();
    assert_eq!(count(&s, "DESCRIBE show_t3"), 1);
    assert_eq!(count(&s, "SHOW TABLES LIKE 'show_t3'"), 1);
}

#[test]
fn a_second_create_needs_or_replace() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TEMPORARY VIEW v(e int) USING parquet")).unwrap();
    assert!(block_on(s.sql("CREATE TEMPORARY VIEW v(e int) USING parquet")).is_err());
    block_on(s.sql("CREATE OR REPLACE TEMPORARY VIEW v(f bigint) USING parquet")).unwrap();
    assert_eq!(count(&s, "DESCRIBE v"), 1);
}

#[test]
fn the_format_and_path_are_checked_when_the_view_is_created() {
    let s = Session::new().unwrap();
    let e = block_on(s.sql("CREATE TEMPORARY VIEW v(e int) USING nosuchformat")).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::DataSourceNotFound
    );
    let e = block_on(
        s.sql("CREATE TEMPORARY VIEW v(e int) USING parquet OPTIONS (path '/nonexistent/krishiv')"),
    )
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::PathNotFound
    );
    block_on(s.sql("CREATE TEMPORARY VIEW q(e int) USING org.apache.spark.sql.parquet")).unwrap();
}
