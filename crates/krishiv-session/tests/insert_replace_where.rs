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
    let sql = "insert into tbl replace where id = 1 values (1, 'a')";
    let new = class_of(SparkProfile::V4_3, sql);
    assert!(new.contains("UNSUPPORTED_FEATURE.TABLE_OPERATION"), "{new}");
    let old = class_of(SparkProfile::V4_2, sql);
    assert!(old.contains("Cannot write into v1 table"), "{old}");
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

#[test]
fn a_column_list_before_replace_where_is_4_3_grammar() {
    let sql = "insert into tbl (id, name) replace where id = 1 values (1, 'a')";
    let new = class_of(SparkProfile::V4_3, sql);
    assert!(new.contains("UNSUPPORTED_FEATURE.TABLE_OPERATION"), "{new}");
    // Spark 4.2 reads `(id, name)` as the insert's own list and fails at `REPLACE`.
    let old = class_of(SparkProfile::V4_2, sql);
    assert!(old.contains("PARSE_SYNTAX_ERROR"), "{old}");
    assert!(old.to_lowercase().contains("'replace'"), "{old}");
    let on = class_of(
        SparkProfile::V4_2,
        "insert into tbl (id) replace on tbl.id = 1 values (1)",
    );
    assert!(on.contains("PARSE_SYNTAX_ERROR"), "{on}");
}

#[test]
fn unnest_in_from_is_4_3_grammar() {
    let s = Session::builder()
        .profile(SparkProfile::V4_2)
        .build()
        .expect("session");
    let err = block_on(async {
        s.sql("select * from unnest(array(1, 2))")
            .await?
            .collect()
            .await
    })
    .expect_err("refused");
    assert!(
        err.to_string()
            .contains("UNRESOLVABLE_TABLE_VALUED_FUNCTION"),
        "{err}"
    );
}
