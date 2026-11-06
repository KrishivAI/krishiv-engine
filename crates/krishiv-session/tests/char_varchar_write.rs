// SPDX-License-Identifier: Apache-2.0
//! The write side of `CHAR(n)` / `VARCHAR(n)` columns: trailing spaces beyond the limit are
//! trimmed, anything else longer fails with `EXCEED_LIMIT_LENGTH`
//! (Spark 4.2.0 `CharVarcharUtils.stringLengthCheck`); the legacy flag turns the check off.
#![expect(clippy::unwrap_used, reason = "test code")]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Result<(), krishiv_common::error::KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await }).map(|_| ())
}

fn lengths(s: &Session) -> Vec<String> {
    let b = block_on(async {
        s.sql("select c, v from t order by 1")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let mut out = Vec::new();
    for batch in &b {
        let (c, v) = (batch.column(0), batch.column(1));
        let text = |a: &arrow_array::ArrayRef, i: usize| {
            a.as_string_opt::<i32>().map_or_else(
                || a.as_string_view().value(i).to_owned(),
                |s| s.value(i).to_owned(),
            )
        };
        for i in 0..batch.num_rows() {
            out.push(format!("{}|{}", text(c, i), text(v, i)));
        }
    }
    out
}

fn fresh() -> Session {
    let s = Session::new().unwrap();
    run(&s, "drop table if exists t").unwrap();
    run(&s, "create table t(c char(3), v varchar(2)) using parquet").unwrap();
    s
}

#[test]
fn trailing_spaces_over_the_limit_are_trimmed_and_other_overflow_fails() {
    let s = fresh();
    run(&s, "insert into t values ('ab   ', 'x   ')").unwrap();
    // The char column reads back padded to its length; the varchar column is cut to its limit (two characters), not padded.
    assert_eq!(lengths(&s), vec!["ab |x "]);
    let e = run(&s, "insert into t values ('abcd', 'x')").unwrap_err();
    assert_eq!(e.error_class().name(), "EXCEED_LIMIT_LENGTH");
    let e = run(&s, "insert into t values ('abc', 'xyz')").unwrap_err();
    assert_eq!(e.error_class().name(), "EXCEED_LIMIT_LENGTH");
}

#[test]
fn the_legacy_flag_turns_the_check_off() {
    let s = fresh();
    run(&s, "set spark.sql.legacy.charVarcharAsString=true").unwrap();
    run(&s, "insert into t values ('abcdef', 'xyz')").unwrap();
    assert_eq!(lengths(&s), vec!["abcdef|xyz"]);
}

fn first_text(s: &Session, sql: &str) -> String {
    text_at(s, sql, 0)
}

fn text_at(s: &Session, sql: &str, column: usize) -> String {
    let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let a = b[0].column(column);
    a.as_string_opt::<i32>().map_or_else(
        || a.as_string_view().value(0).to_owned(),
        |x| x.value(0).to_owned(),
    )
}

#[test]
fn a_table_or_view_made_from_a_query_keeps_the_declared_type() {
    let s = fresh();
    run(&s, "drop table if exists t2").unwrap();
    run(&s, "drop view if exists tv").unwrap();
    run(&s, "create table t2 using parquet as select * from t").unwrap();
    let ddl = first_text(&s, "show create table t2");
    assert!(
        ddl.contains("CHAR(3)") && ddl.contains("VARCHAR(2)"),
        "{ddl}"
    );
    run(&s, "create view tv as select * from t").unwrap();
    let described = text_at(&s, "describe table tv", 1);
    assert_eq!(described, "char(3)");
    // An expression over the column is a plain string.
    run(&s, "drop table if exists t3").unwrap();
    run(
        &s,
        "create table t3 using parquet as select concat(c, 'x') as c from t",
    )
    .unwrap();
    assert!(!first_text(&s, "show create table t3").contains("CHAR"));
}

#[test]
fn a_cast_to_char_or_varchar_is_a_string_cast() {
    let s = fresh();
    run(&s, "insert into t values ('ab', 'x')").unwrap();
    // No truncation and no padding beyond what the char column's read already gave.
    assert_eq!(first_text(&s, "select cast(c as char(1)) from t"), "ab ");
    assert_eq!(
        first_text(&s, "select cast('abcdef' as varchar(2))"),
        "abcdef"
    );
}

#[test]
fn describe_formatted_of_a_column_shows_its_declared_type_and_statistics() {
    let s = fresh();
    let rows = |sql: &str| {
        let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
        b.iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>()
    };
    assert_eq!(rows("describe t c"), 3);
    assert_eq!(rows("describe formatted t c"), 10);
    assert_eq!(text_at(&s, "describe formatted t c", 1), "c");
}

#[test]
fn view_properties_serde_properties_and_partition_locations() {
    let s = fresh();
    run(&s, "create view tv as select * from t").unwrap();
    run(&s, "alter view tv set tblproperties('yes'='no')").unwrap();
    run(&s, "alter view tv unset tblproperties('yes')").unwrap();
    run(&s, "alter table t set serdeproperties('yes'='no')").unwrap();
    assert!(all_text(&s, "describe table extended t").contains("[yes=no]"));
    run(&s, "drop table if exists tp").unwrap();
    run(
        &s,
        "create table tp(a string, v varchar(2), c char(2)) using parquet partitioned by (v, c)",
    )
    .unwrap();
    // A relative location is made absolute; a partition that is not there is PARTITIONS_NOT_FOUND.
    run(
        &s,
        "alter table tp add partition (v='ke', c='nt') location 'loc1'",
    )
    .unwrap();
    run(
        &s,
        "alter table tp partition (v='ke', c='nt') set location 'loc2'",
    )
    .unwrap();
    let e = run(
        &s,
        "alter table tp partition (v='no', c='pe') set location 'loc3'",
    )
    .unwrap_err();
    assert_eq!(e.error_class().name(), "PARTITIONS_NOT_FOUND");
    // Renaming needs every partition column in both specs (_LEGACY_ERROR_TEMP_1232).
    let e = run(
        &s,
        "alter table tp partition (v='ke') rename to partition (v='nt')",
    )
    .unwrap_err();
    assert_eq!(e.error_class().name(), "_LEGACY_ERROR_TEMP_1232");
}

#[test]
fn msck_repair_registers_the_partition_directories_it_finds() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    run(&s, "drop table if exists pr").unwrap();
    run(
        &s,
        &format!(
            "create table pr(a string, v varchar(2), c char(2)) using parquet partitioned by (v, c) location '{}'",
            dir.path().display()
        ),
    )
    .unwrap();
    std::fs::create_dir_all(dir.path().join("v=ke").join("c=nt")).unwrap();
    std::fs::create_dir_all(dir.path().join("_hidden").join("c=x")).unwrap();
    std::fs::create_dir_all(dir.path().join("v=zz")).unwrap();
    let rows = |sql: &str| {
        let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
        b.iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>()
    };
    assert_eq!(rows("show partitions pr"), 0);
    run(&s, "msck repair table pr").unwrap();
    assert_eq!(first_text(&s, "show partitions pr"), "v=ke/c=nt");
    assert_eq!(rows("show partitions pr"), 1);
    let e = run(&s, "msck repair table pr sync partitions").unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
}

fn all_text(s: &Session, sql: &str) -> String {
    let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let mut out = String::new();
    for batch in &b {
        for c in 0..batch.num_columns() {
            let a = batch.column(c);
            for i in 0..batch.num_rows() {
                let v = a.as_string_opt::<i32>().map_or_else(
                    || a.as_string_view().value(i).to_owned(),
                    |x| x.value(i).to_owned(),
                );
                out.push_str(&v);
                out.push('\t');
            }
        }
    }
    out
}

fn count(s: &Session, sql: &str) -> usize {
    let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    b.iter().map(arrow_array::RecordBatch::num_rows).sum()
}

#[test]
fn read_side_padding_only_pads() {
    let s = fresh();
    run(&s, "set spark.sql.legacy.charVarcharAsString=true").unwrap();
    run(&s, "insert into t values ('abcdef', 'x')").unwrap();
    run(&s, "set spark.sql.legacy.charVarcharAsString=false").unwrap();
    // A value longer than the column (written under the legacy flag) is read as it is, not cut.
    assert_eq!(first_text(&s, "select c from t"), "abcdef");
}

#[test]
fn the_declared_type_follows_aliases_instances_and_unions() {
    let s = fresh();
    run(&s, "insert into t values ('a', 'x')").unwrap();
    // The padded column still compares equal to an unpadded literal after an alias.
    assert_eq!(
        count(&s, "select * from (select c as x from t) where x = 'a'"),
        1
    );
    // A second instance of a CTE (self-join) pads its side too.
    assert_eq!(
        count(
            &s,
            "with w as (select * from t) select 1 from w a join w b on a.v = b.v where b.c = 'a'"
        ),
        1
    );
    // A permanent view over the column pads too.
    run(&s, "drop view if exists vv").unwrap();
    run(&s, "create view vv as select c as y from t").unwrap();
    assert_eq!(count(&s, "select * from vv where y = 'a'"), 1);
    for (name, query) in [
        ("ta", "select c as x from t"),
        ("tu", "select c from t union all select c from t"),
    ] {
        run(&s, &format!("drop table if exists {name}")).unwrap();
        run(&s, &format!("create table {name} using parquet as {query}")).unwrap();
        let ddl = first_text(&s, &format!("show create table {name}"));
        assert!(ddl.contains("CHAR(3)"), "{name}: {ddl}");
    }
}

#[test]
fn in_over_a_char_column_and_nested_char_writes_are_refused() {
    let s = fresh();
    let e = block_on(async {
        s.sql("select * from t where c in ('a', 'b')")
            .await?
            .collect()
            .await
    })
    .unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    // A varchar column and the legacy flag are not affected.
    run(&s, "select * from t where v in ('a', 'b')").unwrap();
    run(&s, "drop table if exists tn").unwrap();
    run(&s, "create table tn(a array<varchar(2)>) using parquet").unwrap();
    let e = run(&s, "insert into tn values (array('abc'))").unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn lengths_count_code_points_and_nulls_pass() {
    let s = fresh();
    // Four-byte characters: three of them overflow char(3)? No: exactly three fit, four do not.
    run(
        &s,
        "insert into t values ('\u{1F600}\u{1F600}\u{1F600}', null)",
    )
    .unwrap();
    let e = run(
        &s,
        "insert into t values ('\u{1F600}\u{1F600}\u{1F600}\u{1F600}', null)",
    )
    .unwrap_err();
    assert_eq!(e.error_class().name(), "EXCEED_LIMIT_LENGTH");
    run(&s, "insert into t values (null, null)").unwrap();
    assert_eq!(count(&s, "select * from t where c is null"), 1);
    // Rows that fit and rows that do not in one statement: the failing one fails the statement, and a
    // statement of rows that all fit is not affected by the check.
    let e = run(&s, "insert into t values ('a', 'x'), ('abcd', 'y')").unwrap_err();
    assert_eq!(e.error_class().name(), "EXCEED_LIMIT_LENGTH");
    run(&s, "insert into t values ('a', 'x'), ('b', 'y')").unwrap();
}

#[test]
fn review_cases_static_partitions_windows_missing_sizes_metadata_and_nested_reads() {
    let s = fresh();
    run(&s, "insert into t values ('a', 'x')").unwrap();
    // A window in the select list keeps the declared type of an aliased column.
    assert_eq!(
        count(
            &s,
            "select * from (select c as x, count(*) over () as n from t) q where x = 'a'"
        ),
        1
    );
    // A missing length is DATATYPE_MISSING_SIZE in a cast and in a column definition.
    for sql in [
        "select cast('a' as char)",
        "create table bad(c varchar) using parquet",
    ] {
        let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
        assert_eq!(e.error_class().name(), "DATATYPE_MISSING_SIZE", "{sql}");
    }
    // A static partition value is checked like a dynamic one.
    run(&s, "drop table if exists sp").unwrap();
    run(
        &s,
        "create table sp(a int, p varchar(2)) using parquet partitioned by (p)",
    )
    .unwrap();
    let e = run(&s, "insert into sp partition (p='abc') select 1").unwrap_err();
    assert_eq!(e.error_class().name(), "EXCEED_LIMIT_LENGTH");
    run(&s, "insert into sp partition (p='ab  ') select 1").unwrap();
    assert_eq!(first_text(&s, "show partitions sp"), "p=ab");
    // Reads of a nested CHAR column are refused, as the matching writes are.
    run(&s, "drop table if exists tn").unwrap();
    run(&s, "create table tn(a array<char(2)>) using parquet").unwrap();
    let e = block_on(async { s.sql("select * from tn").await?.collect().await }).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn hidden_file_metadata_survives_the_padding_projection() {
    let s = fresh();
    run(&s, "insert into t values ('a', 'x')").unwrap();
    let name = first_text(&s, "select _metadata.file_name from t");
    assert!(name.ends_with(".parquet"), "{name}");
    assert_eq!(first_text(&s, "select c from t"), "a  ");
}
