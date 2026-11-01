// SPDX-License-Identifier: Apache-2.0
//! `NULL IN (subquery)`: false over no rows, NULL over some (Spark 4.2.0's
//! in-null-semantics golden); NULL either way under
//! `spark.sql.legacy.nullInEmptyListBehavior`.
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn one(s: &Session, sql: &str) -> R<String> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let c = batches[0].column(0);
        Ok(if c.is_null(0) {
            "NULL".to_owned()
        } else {
            c.as_boolean().value(0).to_string()
        })
    })
}

#[test]
fn null_in_a_subquery_depends_on_whether_it_has_rows() -> R {
    let s = Session::new()?;
    block_on(s.sql("create temporary view v(e) as values (1), (null)"))?;
    block_on(s.sql("create temporary view v_empty(e) as select e from v where false"))?;
    assert_eq!(one(&s, "select null in (select e from v_empty)")?, "false");
    assert_eq!(
        one(&s, "select null not in (select e from v_empty)")?,
        "true"
    );
    assert_eq!(one(&s, "select null in (select e from v)")?, "NULL");
    assert_eq!(one(&s, "select null not in (select e from v)")?, "NULL");
    block_on(s.sql("set spark.sql.legacy.nullInEmptyListBehavior = true"))?;
    assert_eq!(one(&s, "select null in (select e from v_empty)")?, "NULL");
    assert_eq!(
        one(&s, "select null not in (select e from v_empty)")?,
        "NULL"
    );
    Ok(())
}
