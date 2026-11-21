// SPDX-License-Identifier: Apache-2.0
//! Spark 4.3 syntax with a session behind it: the `@` time travel suffix of a table name
//! (`spark.sql.timeTravel.atSyntax.enabled`), `COMMENT ON COLUMN`, `REPLACE` and `WITH` options on DML
//! targets, beside the cases `spark43_coverage.rs` takes from a real Spark 4.3.0 run.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn run(session: &Session, sql: &str) -> Result<(), String> {
    block_on(async { session.sql(sql).await?.collect().await })
        .map(|_| ())
        .map_err(|e| e.error_class().name().to_owned())
}

/// The error's message, for what the class alone does not say.
fn message(session: &Session, sql: &str) -> String {
    match block_on(async { session.sql(sql).await?.collect().await }) {
        Ok(_) => "no error".to_owned(),
        Err(e) => e.to_string(),
    }
}

/// The comment of each column of `tbl` (`None` for no comment), by column name, from `DESCRIBE`.
fn comments(session: &Session) -> Vec<(String, Option<String>)> {
    use arrow_array::{Array, StringArray};
    let batches =
        block_on(async { session.sql("describe tbl").await?.collect().await }).expect("describe");
    let mut out = Vec::new();
    for b in &batches {
        let as_text = |i: usize| {
            arrow_cast::cast(b.column(i), &arrow_schema::DataType::Utf8).expect("cast to text")
        };
        let (names, comment) = (as_text(0), as_text(2));
        let names = names.as_any().downcast_ref::<StringArray>().expect("names");
        let comment = comment
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("comments");
        for i in 0..b.num_rows() {
            out.push((
                names.value(i).to_owned(),
                (!comment.is_null(i)).then(|| comment.value(i).to_owned()),
            ));
        }
    }
    out
}

/// Whether `sql` returns exactly one row of one column and that cell is NULL.
fn is_one_null(session: &Session, sql: &str) -> bool {
    use arrow_array::Array;
    let batches = block_on(async { session.sql(sql).await?.collect().await }).expect(sql);
    let rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
    rows == 1
        && batches
            .iter()
            .find(|b| b.num_rows() == 1)
            .is_some_and(|b| b.num_columns() == 1 && b.column(0).is_null(0))
}

fn comment_of(session: &Session, column: &str) -> Option<String> {
    comments(session)
        .into_iter()
        .find(|(n, _)| n == column)
        .unwrap_or_else(|| panic!("no column {column}"))
        .1
}

fn session_with_table() -> Session {
    let s = Session::new().expect("session");
    run(&s, "create table tbl (id int, name string) using parquet").expect("create");
    s
}

#[test]
fn a_suffix_reads_like_the_clause_it_stands_for() {
    let s = session_with_table();
    for sql in [
        "select * from tbl@v1",
        "select * from tbl@V12",
        "select * from tbl@20240101000000000",
        "select * from tbl version as of 1",
        "select * from tbl@v 1",
        "select * from tbl@v'abc'",
        "select * from tbl@ 20240101000000000",
    ] {
        assert_eq!(
            run(&s, sql),
            Err("UNSUPPORTED_FEATURE.TIME_TRAVEL".to_owned()),
            "{sql}"
        );
    }
}

#[test]
fn the_error_names_the_resolved_table_and_a_missing_table_comes_first() {
    let s = session_with_table();
    let text = message(&s, "select * from tbl@v1");
    assert!(text.contains("`spark_catalog`.`default`.`tbl`"), "{text}");
    assert_eq!(
        run(&s, "select * from nope@v1"),
        Err("TABLE_OR_VIEW_NOT_FOUND".to_owned())
    );
    assert_eq!(
        run(&s, "select * from nope version as of 1"),
        Err("TABLE_OR_VIEW_NOT_FOUND".to_owned())
    );
}

#[test]
fn a_suffix_and_a_clause_are_one_too_many() {
    let s = session_with_table();
    assert_eq!(
        run(&s, "select * from tbl@v1 timestamp as of '2024-01-01'"),
        Err("MULTIPLE_TIME_TRAVEL_SPEC".to_owned())
    );
}

#[test]
fn a_malformed_suffix_is_a_syntax_error() {
    let s = session_with_table();
    for sql in [
        "select * from tbl@",
        "select * from tbl@x1",
        "select * from tbl@v",
        "select * from tbl@ v1",
    ] {
        assert_eq!(run(&s, sql), Err("PARSE_SYNTAX_ERROR".to_owned()), "{sql}");
    }
}

#[test]
fn a_timestamp_suffix_is_checked_as_spark_checks_it() {
    let s = session_with_table();
    // Any digits parse; the length and the date-time are checked (`parseAtSyntaxTimestamp`).
    for sql in [
        "select * from tbl@123",
        "select * from tbl@2024",
        "select * from tbl@20190129",
        "select * from tbl@20190129003758",
        "select * from tbl@20191301000000000",
        "select * from tbl@20190230000000000",
    ] {
        assert_eq!(
            run(&s, sql),
            Err("INVALID_TIME_TRAVEL_TIMESTAMP_FORMAT".to_owned()),
            "{sql}"
        );
    }
    // The format error comes before the clause conflict.
    assert_eq!(
        run(&s, "select * from tbl@123 version as of 2"),
        Err("INVALID_TIME_TRAVEL_TIMESTAMP_FORMAT".to_owned())
    );
}

#[test]
fn the_suffix_is_off_with_the_krishiv_key() {
    let s = session_with_table();
    run(&s, "set krishiv.sql.timeTravel.atSyntax.enabled=false").expect("set");
    assert_eq!(
        run(&s, "select * from tbl@v1"),
        Err("UNSUPPORTED_FEATURE.TIME_TRAVEL_AT_SYNTAX".to_owned())
    );
}

#[test]
fn the_suffix_is_off_with_its_config() {
    let s = session_with_table();
    run(&s, "set spark.sql.timeTravel.atSyntax.enabled=false").expect("set");
    assert_eq!(
        run(&s, "select * from tbl@v1"),
        Err("UNSUPPORTED_FEATURE.TIME_TRAVEL_AT_SYNTAX".to_owned())
    );
    assert!(
        message(&s, "select * from tbl@v1")
            .contains("Set \"spark.sql.timeTravel.atSyntax.enabled\" to true to enable it."),
        "{}",
        message(&s, "select * from tbl@v1")
    );
    // The clause is not the suffix: it still parses.
    assert_eq!(
        run(&s, "select * from tbl version as of 1"),
        Err("UNSUPPORTED_FEATURE.TIME_TRAVEL".to_owned())
    );
}

#[test]
fn comment_on_column_sets_and_removes_the_comment() {
    let s = session_with_table();
    run(&s, "comment on column tbl.name is 'the name'").expect("comment");
    assert_eq!(comment_of(&s, "name"), Some("the name".to_owned()));
    assert_eq!(comment_of(&s, "id"), None);
    // `IS NULL` removes it (the DESCRIBE cell is NULL); `IS ''` sets the empty one.
    run(&s, "comment on column tbl.name is null").expect("clear");
    assert_eq!(comment_of(&s, "name"), None);
    run(&s, "comment on column tbl.name is ''").expect("empty");
    assert_eq!(
        comment_of(&s, "name"),
        Some(String::new()),
        "an empty comment is not a missing one"
    );
    assert_eq!(
        run(&s, "comment on column name is 'x'"),
        Err("INVALID_SQL_SYNTAX.COMMENT_ON_COLUMN_INCORRECT_IDENTIFIER".to_owned())
    );
    assert_eq!(
        run(&s, "comment on column tbl.nope is 'x'"),
        Err("UNRESOLVED_COLUMN.WITH_SUGGESTION".to_owned())
    );
}

#[test]
fn comment_on_table_column_changes_several_columns() {
    let s = session_with_table();
    run(&s, "comment on table tbl column (id is 'a', name is 'b')").expect("comments");
    assert_eq!(comment_of(&s, "id"), Some("a".to_owned()));
    assert_eq!(comment_of(&s, "name"), Some("b".to_owned()));
    run(&s, "comment on table tbl column (id is null)").expect("clear");
    assert_eq!(comment_of(&s, "id"), None);
    assert_eq!(comment_of(&s, "name"), Some("b".to_owned()));
    assert_eq!(
        run(&s, "comment on table tbl column (id is 'a', ID is 'b')"),
        Err("NOT_SUPPORTED_CHANGE_SAME_COLUMN".to_owned())
    );
}

#[test]
fn comment_on_column_names_its_own_command_for_a_view() {
    let s = session_with_table();
    run(&s, "create view v as select 1 as a").expect("view");
    run(&s, "create temporary view tv as select 1 as a").expect("temp view");
    for sql in [
        "comment on column v.a is 'x'",
        "comment on column tv.a is 'x'",
    ] {
        assert_eq!(
            run(&s, sql),
            Err("EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE".to_owned()),
            "{sql}"
        );
    }
    let text = |sql: &str| message(&s, sql);
    assert!(
        text("comment on column v.a is 'x'").contains(
            "'COMMENT ON COLUMN' expects a table but `spark_catalog`.`default`.`v` is a view"
        ),
        "{}",
        text("comment on column v.a is 'x'")
    );
    assert!(
        text("comment on column tv.a is 'x'")
            .contains("'COMMENT ON COLUMN' expects a table but `tv` is a view"),
        "{}",
        text("comment on column tv.a is 'x'")
    );
    for view in ["v", "tv"] {
        let sql = format!("comment on table {view} column (a is 'x')");
        assert!(
            message(&s, &sql).contains("'COMMENT ON TABLE ... COLUMN' expects a table"),
            "{}",
            message(&s, &sql)
        );
    }
}

#[test]
fn comment_on_columns_resolves_every_column_before_looking_for_a_repeat() {
    let s = session_with_table();
    assert_eq!(
        run(
            &s,
            "comment on table tbl column (id is 'a', id is 'b', nope is 'c')"
        ),
        Err("UNRESOLVED_COLUMN.WITH_SUGGESTION".to_owned())
    );
    assert_eq!(
        run(&s, "comment on table tbl column (id is 'a', id is 'b')"),
        Err("NOT_SUPPORTED_CHANGE_SAME_COLUMN".to_owned())
    );
}

#[test]
fn replace_where_and_target_options_are_refused_by_the_table() {
    let s = session_with_table();
    for sql in [
        "insert into tbl (id, name) replace where id = 1 values (1, 'a')",
        "insert into tbl replace on tbl.id = 1 values (1, 'a')",
        "insert into tbl as x with ('k' = 'v') (id, name) replace on x.id = 1 values (1, 'a')",
        "update tbl with ('k' = 'v') set name = 'x'",
        "update tbl t with ('k' = 'v') set name = 'x'",
        "delete from tbl with ('k' = 'v')",
        "delete from tbl as t with ('k' = 'v') where t.id = 1",
        "merge into tbl t with ('k' = 'v') using tbl s on t.id = s.id when matched then delete",
        "merge into tbl using tbl with ('k' = 'v') s on tbl.id = s.id when matched then delete",
    ] {
        assert_eq!(
            run(&s, sql),
            Err("UNSUPPORTED_FEATURE.TABLE_OPERATION".to_owned()),
            "{sql}"
        );
    }
    for (sql, operation) in [
        (
            "insert into tbl (id, name) replace where id = 1 values (1, 'a')",
            "INSERT INTO ... REPLACE WHERE",
        ),
        (
            "insert into tbl replace on tbl.id = 1 values (1, 'a')",
            "INSERT INTO ... REPLACE ON/USING",
        ),
    ] {
        assert!(
            message(&s, sql).contains(operation),
            "{sql}: {}",
            message(&s, sql)
        );
    }
    // The operation is the statement's, as without options.
    for (sql, operation) in [
        ("update tbl with ('k' = 'v') set name = 'x'", "UPDATE TABLE"),
        ("delete from tbl with ('k' = 'v')", "DELETE"),
        (
            "merge into tbl with ('k' = 'v') using tbl s on tbl.id = s.id when matched then delete",
            "MERGE INTO TABLE",
        ),
    ] {
        let text = message(&s, sql);
        assert!(
            text.contains(operation) && !text.contains("WITH options"),
            "{text}"
        );
    }
    // The table is resolved, and the assignments analysed, before the table refuses.
    assert_eq!(
        run(&s, "delete from nope with ('k' = 'v')"),
        Err("TABLE_OR_VIEW_NOT_FOUND".to_owned())
    );
    assert_eq!(
        run(&s, "update tbl with ('k' = 'v') set nope = 1"),
        Err("UNRESOLVED_COLUMN.WITH_SUGGESTION".to_owned())
    );
}

/// The clause is refused where each kind of relation resolves, with the relation as Spark names it.
#[test]
fn the_clause_is_refused_for_every_kind_of_relation() {
    let s = session_with_table();
    run(&s, "create view pv as select 1 as a").expect("view");
    run(&s, "create temporary view tv as select 1 as a").expect("temp view");
    let holder = tempfile::tempdir().expect("temp dir");
    let dir = holder.path().to_path_buf();
    for (sql, relation) in [
        ("with c as (select 1 a) select * from c@v1", "`c`"),
        (
            "with c as (select 1 a) select * from c timestamp as of '2024-01-01'",
            "`c`",
        ),
        ("select * from tv@v1", "`tv`"),
        (
            "describe query select * from tbl version as of 1",
            "`spark_catalog`.`default`.`tbl`",
        ),
        ("select * from pv@v1", "`spark_catalog`.`default`.`pv`"),
        (
            "create view v2 as select * from tbl@v1",
            "`spark_catalog`.`default`.`tbl`",
        ),
        (
            "delete from tbl where id in (select id from tbl@v1)",
            "`spark_catalog`.`default`.`tbl`",
        ),
        (
            "update tbl set name = (select max(name) from tbl@v1)",
            "`spark_catalog`.`default`.`tbl`",
        ),
        (
            "create or replace temporary view tv3 as select * from tbl@v1",
            "`spark_catalog`.`default`.`tbl`",
        ),
        (
            "create temporary view tv2 as select * from tbl@v1",
            "`spark_catalog`.`default`.`tbl`",
        ),
        (
            "create global temporary view g2 as select * from tbl@v1",
            "`spark_catalog`.`default`.`tbl`",
        ),
        (
            "alter view pv as select * from tbl@v1",
            "`spark_catalog`.`default`.`tbl`",
        ),
        (
            "merge into tbl t using (select * from tbl@v1) s on t.id = s.id when matched then delete",
            "`spark_catalog`.`default`.`tbl`",
        ),
        (
            "merge into tbl t using (select * from pv@v1) s on t.id = s.a when matched then delete",
            "`spark_catalog`.`default`.`pv`",
        ),
        (
            "merge into tbl t using tbl s on t.id = s.id when matched and t.id in (select id from tbl@v1) then delete",
            "`spark_catalog`.`default`.`tbl`",
        ),
        (
            "select * from pv version as of 1 with ('a' = 'b')",
            "`spark_catalog`.`default`.`pv`",
        ),
        ("create view v4 as select * from tv@v1", "`tv`"),
        ("alter view pv as select * from tv@v1", "`tv`"),
        (
            "delete from tv where a in (select id from tbl@v1)",
            "`spark_catalog`.`default`.`tbl`",
        ),
        (
            &format!("select * from parquet.`{}` version as of 1", dir.display()),
            &format!("`parquet`.`{}`", dir.display()),
        ),
        (
            &format!("select * from parquet.`{}`@v1", dir.display()),
            &format!("`parquet`.`{}`", dir.display()),
        ),
    ] {
        assert_eq!(
            run(&s, sql),
            Err("UNSUPPORTED_FEATURE.TIME_TRAVEL".to_owned()),
            "{sql}: {}",
            message(&s, sql)
        );
        assert!(
            message(&s, sql).contains(relation),
            "{sql}: {}",
            message(&s, sql)
        );
    }
    // A view that was refused at creation does not exist.
    assert_eq!(
        run(&s, "select * from v2"),
        Err("TABLE_OR_VIEW_NOT_FOUND".to_owned())
    );
}

/// A variant edit whose argument folds to NULL is NULL before its literal path is parsed, as in Spark
/// (`NullIntolerant` after constant folding); a call that does not fold still raises.
#[test]
fn a_variant_edit_over_an_argument_that_folds_to_null_is_null() {
    let s = Session::new().expect("session");
    for sql in [
        "select variant_insert(parse_json(cast(null as string)), 'bad', 1)",
        "select variant_set(parse_json(cast(null as string)), 'bad', 1)",
        "select variant_array_append(parse_json(cast(null as string)), 'bad', 1)",
        "select variant_insert(parse_json('{}'), 'bad', parse_json(cast(null as string)))",
        "select try_variant_insert(to_variant_object(cast(null as struct<a:int>)), 'bad', 1)",
    ] {
        assert!(is_one_null(&s, sql), "{sql}");
    }
    assert_eq!(
        run(&s, "select variant_insert(parse_json('{}'), 'bad', 1)"),
        Err("INVALID_VARIANT_PATH".to_owned())
    );
}

/// `JSON_TABLE` checks every path first, then each column's cast in the session's mode; a refusal names the
/// whole call (`JsonTable.checkInputDataTypes`).
#[test]
fn a_json_table_column_type_no_cast_reaches_names_the_call() {
    let s = Session::new().expect("session");
    let struct_col =
        "select * from json_table('{\"a\":1}', '$' columns (v struct<x: int> path '$.a'))";
    assert_eq!(
        run(&s, struct_col),
        Err("DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION".to_owned())
    );
    let text = message(&s, struct_col);
    assert!(
        text.contains("JSON_TABLE(")
            && text.contains("COLUMNS (v STRUCT<x: INT> PATH '$.a') NULL ON ERROR)"),
        "{text}"
    );
    assert!(
        text.contains("cannot cast \"STRING\" to \"STRUCT<x: INT>\""),
        "{text}"
    );
    // An `EXISTS` column (BOOLEAN) cast to TIMESTAMP is refused under ANSI, with the call's text, not the cast.
    let exists =
        "select * from json_table('{\"a\":1}', '$' columns (v timestamp exists path '$.a'))";
    assert_eq!(
        run(&s, exists),
        Err("DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION".to_owned()),
        "{}",
        message(&s, exists)
    );
    let text = message(&s, exists);
    assert!(
        text.contains("JSON_TABLE(") && !text.contains("__json_table"),
        "{text}"
    );
    // Every path is checked before any column's cast.
    assert_eq!(
        run(
            &s,
            "select * from json_table('{\"a\":1}', '$' columns (v struct<x: int> path '$.a', w int path 'bad'))"
        ),
        Err("DATATYPE_MISMATCH.INVALID_JSON_TABLE_PATH".to_owned())
    );
    // With ANSI off the cast of an EXISTS column to TIMESTAMP is allowed.
    run(&s, "set spark.sql.ansi.enabled = false").expect("set");
    assert_eq!(run(&s, exists), Ok(()));
}

/// Temporary views are named as written (upper case kept, a global one without a catalog); a permanent
/// view refuses `REPLACE WHERE` as a view write; the repeated column of a comment list is quoted.
#[test]
fn view_names_as_written_and_view_replace_where() {
    let s = session_with_table();
    run(&s, "create temporary view TV as select 1 as a").expect("temp view");
    run(&s, "create global temporary view gv as select 1 as a").expect("global view");
    run(&s, "create view pv as select 1 as a").expect("view");
    assert!(
        message(&s, "comment on column TV.a is 'x'").contains("`TV` is a view"),
        "{}",
        message(&s, "comment on column TV.a is 'x'")
    );
    let global = message(&s, "comment on column global_temp.gv.a is 'x'");
    assert!(
        global.contains("`global_temp`.`gv` is a view") && !global.contains("spark_catalog"),
        "{global}"
    );
    assert_eq!(
        run(&s, "insert into pv replace where a = 1 select 1"),
        Err("VIEW_WRITE_NOT_ALLOWED".to_owned())
    );
    let same = message(&s, "comment on table tbl column (id is 'a', ID is 'b')");
    assert!(same.contains("`id`"), "{same}");
}

/// Review threads of #937: where each refusal sits relative to the others.
#[test]
fn refusals_keep_their_order() {
    let s = session_with_table();
    run(&s, "create view pv as select 1 as a").expect("view");
    // MERGE: a `VERSION AS OF` in an action's value is the time travel refusal, not TABLE_OPERATION.
    for sql in [
        "merge into tbl t using tbl s on t.id = s.id when matched then update set name = (select max(name) from tbl@v1)",
        "merge into tbl t using tbl s on t.id = s.id when not matched then insert (id, name) values (1, (select max(name) from tbl@v1))",
    ] {
        assert_eq!(
            run(&s, sql),
            Err("UNSUPPORTED_FEATURE.TIME_TRAVEL".to_owned()),
            "{sql}"
        );
    }
    // A missing target is reported before a time travel clause in a subquery.
    assert_eq!(
        run(
            &s,
            "delete from missing where exists (select 1 from tbl version as of 1)"
        ),
        Err("TABLE_OR_VIEW_NOT_FOUND".to_owned())
    );
    // A hidden name read with a time travel clause is a missing table, as without one.
    for sql in [
        "select * from __krishiv_snapshot_x@v1",
        "select * from __krishiv_snapshot_x version as of 1",
    ] {
        assert_eq!(
            run(&s, sql),
            Err("TABLE_OR_VIEW_NOT_FOUND".to_owned()),
            "{sql}"
        );
    }
    // A multi-table insert refuses `REPLACE WHERE` into a view as a single one does.
    assert_eq!(
        run(
            &s,
            "from (select 1 as a) insert into pv replace where a = 1 select a insert into tbl select a, 'x'"
        ),
        Err("VIEW_WRITE_NOT_ALLOWED".to_owned())
    );
    // The validator visits the columns of COMMENT ON TABLE ... COLUMN.
    assert_eq!(
        run(
            &s,
            "comment on table tbl column (identifier(1 in ()) is 'x')"
        ),
        Err("INVALID_SQL_SYNTAX.EMPTY_IN_PREDICATE".to_owned())
    );
}

/// The `@` suffix faults are Spark's `AstBuilder` checks: a syntax error later in the statement comes first.
#[test]
fn a_syntax_error_comes_before_a_time_travel_fault() {
    let s = session_with_table();
    for sql in [
        "select * from tbl@20240230000000000 where (",
        "select * from tbl@v1 version as of 1 where (",
    ] {
        assert_eq!(run(&s, sql), Err("PARSE_SYNTAX_ERROR".to_owned()), "{sql}");
    }
    assert_eq!(
        run(&s, "select * from tbl@20240230000000000"),
        Err("INVALID_TIME_TRAVEL_TIMESTAMP_FORMAT".to_owned())
    );
    assert_eq!(
        run(&s, "select * from tbl@v1 version as of 1"),
        Err("MULTIPLE_TIME_TRAVEL_SPEC".to_owned())
    );
}
