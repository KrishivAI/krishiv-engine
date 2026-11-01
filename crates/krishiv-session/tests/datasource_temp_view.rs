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
