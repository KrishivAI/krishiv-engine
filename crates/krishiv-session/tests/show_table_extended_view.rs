// SPDX-License-Identifier: Apache-2.0
//! `SHOW TABLE EXTENDED ... PARTITION (...)` on a view. Spark 4.2.0 builds
//! `UnresolvedTable(names, "SHOW TABLE EXTENDED ... PARTITION ...")` for this form
//! (`AstBuilder.visitShowTableExtended`), so a permanent, temporary view is `EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE` with that operation,
//! not `FORBIDDEN_OPERATION`. The expected text is the class and operation of the
//! vendored golden for the same class (`ALTER TABLE ... CHANGE COLUMN`), applied to
//! this operation; a Spark run of these statements is requested.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn failure(s: &Session, sql: &str) -> String {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => "answered".to_owned(),
        Err(e) => e.to_string(),
    }
}

fn session() -> Session {
    let s = Session::new().unwrap();
    for q in [
        "CREATE TABLE p (a INT, b INT) USING parquet PARTITIONED BY (b)",
        "CREATE VIEW v AS SELECT * FROM p",
        "CREATE TEMPORARY VIEW tv AS SELECT * FROM p",
    ] {
        block_on(async { s.sql(q).await?.collect().await }).unwrap();
    }
    s
}

#[test]
fn a_view_is_a_table_that_is_not_one() {
    let s = session();
    for q in [
        "SHOW TABLE EXTENDED LIKE 'v' PARTITION (b = 1)",
        "SHOW TABLE EXTENDED LIKE 'tv' PARTITION (b = 1)",
    ] {
        let e = failure(&s, q);
        assert!(
            e.contains("EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"),
            "{q}: {e}"
        );
        assert!(
            e.contains("'SHOW TABLE EXTENDED ... PARTITION ...' expects a table but"),
            "{q}: {e}"
        );
    }
}

#[test]
fn a_table_with_no_such_partition_is_still_partitions_not_found() {
    let s = session();
    let q = "SHOW TABLE EXTENDED LIKE 'p' PARTITION (b = 1)";
    let e = failure(&s, q);
    assert!(e.contains("PARTITIONS_NOT_FOUND"), "{q}: {e}");
}
