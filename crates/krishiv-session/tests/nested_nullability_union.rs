// SPDX-License-Identifier: Apache-2.0
//! R1-D-16: rows that differ only in nested nullability under ANSI off are two rows in `VALUES` and
//! `UNION ALL`, not an `INTERNAL_ERROR`.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

fn rows(s: &Session, sql: &str) -> Result<usize, KrishivError> {
    let batches = block_on(block_on(s.sql(sql))?.collect())?;
    Ok(batches.iter().map(arrow_array::RecordBatch::num_rows).sum())
}

#[test]
fn nested_nullability_differences_give_rows_with_ansi_off() -> Result<(), KrishivError> {
    let s = Session::new()?;
    s.conf().set("spark.sql.ansi.enabled", "false")?;
    for sql in [
        "select v from values (array(1,2)),(array(cast(null as int),1)) t(v)",
        "select array(1,2) v union all select array(cast(null as int),1)",
        "select v from values (named_struct('a',1)),(named_struct('a',cast(null as int))) t(v)",
        "select named_struct('a',1) v union all select named_struct('a',cast(null as int))",
    ] {
        assert_eq!(rows(&s, sql)?, 2, "{sql}");
    }
    Ok(())
}
