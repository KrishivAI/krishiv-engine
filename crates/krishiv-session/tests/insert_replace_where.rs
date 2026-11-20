// SPDX-License-Identifier: Apache-2.0
//! `INSERT INTO t [(columns)] REPLACE WHERE cond` on a table that has no overwrite by filter: Spark 4.3 refuses
//! the operation on the table (`UNSUPPORTED_FEATURE.TABLE_OPERATION`), Spark 4.2 refuses the V1 write.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn class_of(profile: SparkProfile, sql: &str) -> String {
    let s = Session::builder()
        .profile(profile)
        .build()
        .expect("session");
    block_on(async {
        s.sql("create table tbl (id int, name string) using parquet")
            .await?
            .collect()
            .await
    })
    .expect("create");
    let err = block_on(async { s.sql(sql).await?.collect().await }).expect_err("refused");
    err.to_string()
}

#[test]
fn replace_where_on_a_file_table_is_refused_by_profile() {
    for sql in [
        "insert into tbl replace where id = 1 values (1, 'a')",
        "insert into tbl (id, name) replace where id = 1 values (1, 'a')",
    ] {
        let new = class_of(SparkProfile::V4_3, sql);
        assert!(new.contains("UNSUPPORTED_FEATURE.TABLE_OPERATION"), "{new}");
        let old = class_of(SparkProfile::V4_2, sql);
        assert!(old.contains("Cannot write into v1 table"), "{old}");
    }
}

#[test]
fn replace_on_takes_no_column_list() {
    let err = class_of(
        SparkProfile::V4_3,
        "insert into tbl (id) replace on tbl.id = 1 values (1)",
    );
    assert!(
        err.contains("INSERT_REPLACE_ON_COLUMN_LIST_NOT_ALLOWED"),
        "{err}"
    );
}
