// SPDX-License-Identifier: Apache-2.0
//! `current_time()` names its column `current_time(6)`, as Spark does (SQL-26).
#![forbid(unsafe_code)]

use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn column_name(s: &Session, sql: &str) -> R<String> {
    let batches = block_on(block_on(s.sql(sql))?.collect())?;
    Ok(batches[0].schema().field(0).name().clone())
}

#[test]
fn the_default_precision_shows_in_the_name() -> R {
    let s = Session::new()?;
    block_on(s.sql("SET spark.sql.timeType.enabled = true"))?;
    assert_eq!(column_name(&s, "SELECT current_time()")?, "current_time(6)");
    assert_eq!(
        column_name(&s, "SELECT current_time(3)")?,
        "current_time(3)"
    );
    Ok(())
}
