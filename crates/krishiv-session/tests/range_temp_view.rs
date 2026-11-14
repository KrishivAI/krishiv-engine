// SPDX-License-Identifier: Apache-2.0
//! A temporary view over `range()` can be queried (SQL-25: it failed with
//! `INTERNAL_ERROR attribute #0 is not an input column`).
#![forbid(unsafe_code)]

use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

#[test]
fn a_temp_view_over_range_is_queryable() -> R {
    let s = Session::new()?;
    block_on(s.sql("CREATE TEMPORARY VIEW v AS SELECT id * 2 AS a FROM range(5)"))?;
    let batches = block_on(block_on(s.sql("SELECT a FROM v ORDER BY a"))?.collect())?;
    let rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
    assert_eq!(rows, 5);
    Ok(())
}

#[test]
fn a_dataframe_temp_view_over_range_is_queryable() -> R {
    let s = Session::new()?;
    block_on(
        block_on(s.sql("SELECT id * 2 AS a FROM range(5)"))?.create_or_replace_temp_view("v"),
    )?;
    let batches = block_on(block_on(s.sql("SELECT a FROM v ORDER BY a"))?.collect())?;
    let rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
    assert_eq!(rows, 5);
    Ok(())
}
