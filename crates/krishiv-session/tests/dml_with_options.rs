// SPDX-License-Identifier: Apache-2.0
//! `WITH (key = value, ...)` after the target of `UPDATE`, `DELETE` and `MERGE` (Spark 4.3), after the alias and
//! never before it. The classes are what Spark 4.3.0 raised: a session-catalog parquet table supports none of
//! the three, so a statement that parses is refused as `UNSUPPORTED_FEATURE.TABLE_OPERATION`.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn class(s: &Session, sql: &str) -> String {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => "ok".to_owned(),
        Err(e) => e.error_class().name().to_owned(),
    }
}

#[test]
fn with_options_parse_after_the_alias_and_match_spark() {
    let s = Session::new().unwrap();
    for q in [
        "create table tbl (id int, name string) using parquet",
        "create temporary view src as select 1 as id, 'z' as name",
    ] {
        block_on(async { s.sql(q).await?.collect().await }).unwrap();
    }
    let table_op = "UNSUPPORTED_FEATURE.TABLE_OPERATION";
    for (sql, want) in [
        (
            "update tbl with ('k'='v') set name='x' where id=1",
            table_op,
        ),
        ("update tbl as t with ('k'='v') set name='x'", table_op),
        ("update tbl t with ('k'='v') set name='x'", table_op),
        (
            "update tbl with ('k'='v') as t set name='x'",
            "PARSE_SYNTAX_ERROR",
        ),
        ("delete from tbl with ('k'='v') where id=1", table_op),
        ("delete from tbl t with ('k'='v')", table_op),
        ("delete from tbl with ('k'='v')", table_op),
        (
            "delete from tbl with ('k'='v') t where t.id=1",
            "PARSE_SYNTAX_ERROR",
        ),
        (
            "merge into tbl with ('k'='v') using src on tbl.id=src.id when matched then update set name=src.name",
            table_op,
        ),
        (
            "merge into tbl using src with ('k'='v') s on tbl.id=s.id when matched then delete",
            table_op,
        ),
        (
            "merge into tbl using src s with ('k'='v') on tbl.id=s.id when matched then delete",
            "PARSE_SYNTAX_ERROR",
        ),
        (
            "merge into tbl with ('k'='v') t using src s on t.id=s.id when matched then delete",
            "PARSE_SYNTAX_ERROR",
        ),
        ("update tbl with () set name='x'", "PARSE_SYNTAX_ERROR"),
        (
            "update nope with ('k'='v') set name='x'",
            "TABLE_OR_VIEW_NOT_FOUND",
        ),
        ("delete from nope with ('k'='v')", "TABLE_OR_VIEW_NOT_FOUND"),
    ] {
        assert_eq!(class(&s, sql), want, "{sql}");
    }
}

/// The options of a `MERGE` source are checked as the target's are (Spark reads both through
/// `createUnresolvedRelation`): a duplicate key and a key without a value.
#[test]
fn merge_source_options_are_validated_like_the_targets() {
    let s = Session::new().unwrap();
    for q in [
        "create table tbl (id int, name string) using parquet",
        "create temporary view src as select 1 as id, 'z' as name",
    ] {
        block_on(async { s.sql(q).await?.collect().await }).unwrap();
    }
    let tail = "on tbl.id = s.id when matched then delete";
    for options in ["('k'='v', 'k'='w')", "('k')"] {
        let target = class(
            &s,
            &format!("merge into tbl with {options} using src s {tail}"),
        );
        let source = class(
            &s,
            &format!("merge into tbl using src with {options} s {tail}"),
        );
        assert_ne!(target, "UNSUPPORTED_FEATURE.TABLE_OPERATION", "{options}");
        assert_eq!(source, target, "{options}");
    }
}
