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
    // Spark 4.2.0 `charvarchar.sql` (`desc formatted char_tbl c`): the declared type, then the statistics rows,
    // all NULL before `ANALYZE`.
    let s = fresh();
    let rows = |sql: &str| -> Vec<String> {
        let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
        let text = |a: &arrow_array::ArrayRef, i: usize| {
            if a.is_null(i) {
                return "NULL".to_owned();
            }
            a.as_string_opt::<i32>().map_or_else(
                || a.as_string_view().value(i).to_owned(),
                |x| x.value(i).to_owned(),
            )
        };
        b.iter()
            .flat_map(|batch| {
                (0..batch.num_rows())
                    .map(|i| format!("{}\t{}", text(batch.column(0), i), text(batch.column(1), i)))
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    assert_eq!(
        rows("describe t c"),
        ["col_name\tc", "data_type\tchar(3)", "comment\tNULL"]
    );
    assert_eq!(
        rows("describe formatted t c"),
        [
            "col_name\tc",
            "data_type\tchar(3)",
            "comment\tNULL",
            "min\tNULL",
            "max\tNULL",
            "num_nulls\tNULL",
            "distinct_count\tNULL",
            "avg_col_len\tNULL",
            "max_col_len\tNULL",
            "histogram\tNULL",
        ]
    );
}

#[test]
fn view_properties_serde_properties_and_partition_locations() {
    let s = fresh();
    run(&s, "create view tv as select * from t").unwrap();
    run(
        &s,
        "alter view tv set tblproperties('yes'='no', 'keep'='1')",
    )
    .unwrap();
    // `all_text` lists column by column: keys, then values.
    assert_eq!(all_text(&s, "show tblproperties tv"), "keep\tyes\t1\tno\t");
    run(&s, "alter view tv unset tblproperties('yes')").unwrap();
    assert_eq!(all_text(&s, "show tblproperties tv"), "keep\t1\t");
    // A key that is not set is no error, with or without `IF EXISTS` (Spark 4.2.0).
    run(&s, "alter view tv unset tblproperties('nope')").unwrap();
    run(&s, "alter table t set serdeproperties('yes'='no')").unwrap();
    assert!(all_text(&s, "describe table extended t").contains("[yes=no]"));
    run(&s, "drop table if exists tp").unwrap();
    run(
        &s,
        "create table tp(a string, v varchar(2), c char(2)) using parquet partitioned by (v, c)",
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let (loc1, loc2) = (dir.path().join("loc1"), dir.path().join("loc2"));
    // `all_text` lists column by column: the value of `Location` sits one column's length after its key.
    let location = || {
        let b = block_on(async {
            s.sql("describe table extended tp partition (v='ke', c='nt')")
                .await?
                .collect()
                .await
        })
        .unwrap();
        let text = |a: &arrow_array::ArrayRef, i: usize| {
            a.as_string_opt::<i32>().map_or_else(
                || a.as_string_view().value(i).to_owned(),
                |x| x.value(i).to_owned(),
            )
        };
        b.iter()
            .find_map(|batch| {
                (0..batch.num_rows())
                    .find(|&i| text(batch.column(0), i) == "Location")
                    .map(|i| text(batch.column(1), i))
            })
            .unwrap()
    };
    run(
        &s,
        &format!(
            "alter table tp add partition (v='ke', c='nt') location '{}'",
            loc1.display()
        ),
    )
    .unwrap();
    assert_eq!(location(), format!("file:{}", loc1.display()));
    run(
        &s,
        &format!(
            "alter table tp partition (v='ke', c='nt') set location '{}'",
            loc2.display()
        ),
    )
    .unwrap();
    assert_eq!(location(), format!("file:{}", loc2.display()));
    // An empty location is refused (Spark: Hadoop cannot make a path of it).
    for sql in [
        "alter table tp partition (v='ke', c='nt') set location ''",
        "alter table tp add partition (v='xx', c='yy') location ''",
    ] {
        let e = run(&s, sql).unwrap_err();
        assert_eq!(e.error_class().name(), "INVALID_EMPTY_LOCATION", "{sql}");
    }
    // A location already holding data, even only below a subdirectory, or a file, would be ignored by reads.
    let nested = dir.path().join("nested");
    std::fs::create_dir_all(nested.join("x=1")).unwrap();
    std::fs::write(nested.join("x=1").join("part-0.parquet"), b"").unwrap();
    let file = dir.path().join("plain");
    std::fs::write(&file, b"x").unwrap();
    for target in [&nested, &file] {
        let e = run(
            &s,
            &format!(
                "alter table tp partition (v='ke', c='nt') set location '{}'",
                target.display()
            ),
        )
        .unwrap_err();
        assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    }
    assert_eq!(location(), format!("file:{}", loc2.display()));
    let e = run(
        &s,
        &format!(
            "alter table tp partition (v='no', c='pe') set location '{}'",
            dir.path().join("loc3").display()
        ),
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
fn change_column_to_an_equivalent_character_type_is_allowed() {
    // Spark 4.2.0: `CHARACTER(3)` is `CHAR(3)`, so the change is no change; `DESCRIBE` still shows `char(3)`.
    let s = fresh();
    run(&s, "alter table t change column c type character(3)").unwrap();
    assert_eq!(text_at(&s, "describe table t", 1), "char(3)");
    let e = run(&s, "alter table t change column c type char(4)").unwrap_err();
    assert_eq!(e.error_class().name(), "NOT_SUPPORTED_CHANGE_COLUMN");
}

#[test]
fn msck_repair_of_a_missing_directory_fails() {
    // Spark 4.2.0 throws an unclassified `FileNotFoundException` ("File file:<dir> does not exist"); Krishiv
    // maps that failure to `PATH_NOT_FOUND`, its error class for a missing path (not a Spark-observable class here).
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("gone");
    let s = Session::new().unwrap();
    run(&s, "drop table if exists pm").unwrap();
    run(
        &s,
        &format!(
            "create table pm(a string, k string) using parquet partitioned by (k) location '{}'",
            root.display()
        ),
    )
    .unwrap();
    let _ = std::fs::remove_dir_all(&root);
    let e = run(&s, "msck repair table pm").unwrap_err();
    assert_eq!(e.error_class().name(), "PATH_NOT_FOUND");
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
    run(&s, "msck repair table pr sync partitions").unwrap();
    assert_eq!(rows("show partitions pr"), 1);
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
}

/// The answers below are a real Spark 4.2.0 run's (`ApplyCharTypePadding` on `IN`).
#[test]
fn in_over_a_char_column_pads_as_spark_does() {
    let s = fresh();
    run(&s, "drop table if exists c1").unwrap();
    run(
        &s,
        "create table c1(id int, c char(3), d char(5), v varchar(3)) using parquet",
    )
    .unwrap();
    run(
        &s,
        "insert into c1 values (1,'a','b','a'),(2,'ab','xyz','a'),(3,null,null,null),(4,'abc','abcde','abc')",
    )
    .unwrap();
    let ids = |s: &Session, cond: &str| -> Vec<String> {
        block_on(async {
            s.sql(&format!("select id from c1 where {cond} order by id"))
                .await?
                .collect()
                .await
        })
        .unwrap()
        .iter()
        .flat_map(|b| {
            let a = b
                .column(0)
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .unwrap()
                .clone();
            (0..a.len())
                .map(move |i| a.value(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect()
    };
    assert_eq!(ids(&s, "c in ('a')"), ["1"]);
    assert_eq!(ids(&s, "c in ('a  ')"), ["1"]);
    assert_eq!(ids(&s, "c in ('a','ab')"), ["1", "2"]);
    assert_eq!(ids(&s, "c in ('a','abcd')"), ["1"]);
    assert!(ids(&s, "c in ('a  x')").is_empty());
    assert_eq!(ids(&s, "c not in ('a')"), ["2", "4"]);
    // A `NULL` in the list: no padding at all.
    assert!(ids(&s, "c in ('a', null)").is_empty());
    assert!(ids(&s, "c in (null, 'a', 'abcd')").is_empty());
    // A constant string folds, then pads; `upper('a')` is `A`.
    assert_eq!(ids(&s, "c in (concat('a', ''))"), ["1"]);
    assert!(ids(&s, "c in (upper('a'))").is_empty());
    // Only an all-`char` list of attributes pads.
    assert!(ids(&s, "c in (d)").is_empty());
    assert_eq!(ids(&s, "c in (d, c)"), ["1", "2", "4"]);
    assert_eq!(ids(&s, "c in (d, v)"), ["4"]);
    assert!(ids(&s, "c in ('a', d)").is_empty());
    let names = block_on(async {
        s.sql("select c in ('a','abcd') from c1")
            .await
            .unwrap()
            .schema()
            .await
    })
    .unwrap();
    assert_eq!(names.fields()[0].name(), "(c IN (a, abcd))");
}

/// The answers below are a real Spark 4.2.0 run's (`CharVarcharUtils.stringLengthCheck`, `addPaddingForScan`).
#[test]
fn nested_char_and_varchar_columns_are_checked_on_write_and_padded_on_read() {
    let s = fresh();
    run(&s, "drop table if exists n1").unwrap();
    run(&s, "drop table if exists n2").unwrap();
    run(
        &s,
        "create table n1(a array<char(2)>, s struct<x:char(2),y:varchar(3)>, m map<char(2),varchar(2)>) using parquet",
    )
    .unwrap();
    run(
        &s,
        "insert into n1 values (array('a','bb'), named_struct('x','a','y','b'), map('k','v'))",
    )
    .unwrap();
    assert_eq!(
        first_text(
            &s,
            "select concat_ws('|', a[0], a[1], s.x, s.y, m['k ']) from n1"
        ),
        "a |bb|a |b|v"
    );
    assert_eq!(
        first_text(&s, "select cast(length(a[0]) as string) from n1"),
        "2"
    );
    // Trailing spaces go down to the limit.
    run(
        &s,
        "insert into n1 values (array('a  '), named_struct('x','a ','y','bcd '), map('k ','v '))",
    )
    .unwrap();
    assert_eq!(
        first_text(
            &s,
            "select concat_ws('|', a[0], s.x, s.y, m['k ']) from n1 where s.y = 'bcd'"
        ),
        "a |a |bcd|v "
    );
    // NULLs at every level stay NULL.
    run(&s, "insert into n1 select null,null,null").unwrap();
    run(
        &s,
        "insert into n1 values (array(null,'b'), null, map('k',null))",
    )
    .unwrap();
    assert_eq!(
        first_text(
            &s,
            "select cast(count(*) as string) from n1 where s is null and a[0] is null"
        ),
        "2"
    );
    for (sql, limit) in [
        (
            "insert into n1 values (array('abc'), named_struct('x','a','y','b'), map('k','v'))",
            "2",
        ),
        (
            "insert into n1 values (array('a'), named_struct('x','a','y','bcde'), map('k','v'))",
            "3",
        ),
        (
            "insert into n1 values (array('a'), named_struct('x','a','y','b'), map('kkk','v'))",
            "2",
        ),
        (
            "insert into n1 values (array('a'), named_struct('x','a','y','b'), map('k','vvv'))",
            "2",
        ),
    ] {
        let e = run(&s, sql).unwrap_err();
        assert_eq!(e.error_class().name(), "EXCEED_LIMIT_LENGTH", "{sql}");
        assert!(e.to_string().contains(limit), "{sql}: {e}");
    }
    // Structs inside arrays inside structs.
    run(
        &s,
        "create table n2(s struct<x:char(4),a:array<struct<y:char(2)>>>) using parquet",
    )
    .unwrap();
    run(
        &s,
        "insert into n2 values (named_struct('x','a','a',array(named_struct('y','b'))))",
    )
    .unwrap();
    assert_eq!(
        first_text(&s, "select concat(s.x, s.a[0].y) from n2"),
        "a   b "
    );
    let e = run(
        &s,
        "insert into n2 values (named_struct('x','a','a',array(named_struct('y','bcd'))))",
    )
    .unwrap_err();
    assert_eq!(e.error_class().name(), "EXCEED_LIMIT_LENGTH");
}

#[test]
fn hidden_file_metadata_survives_the_padding_projection() {
    let s = fresh();
    run(&s, "insert into t values ('a', 'x')").unwrap();
    let name = first_text(&s, "select _metadata.file_name from t");
    assert!(name.ends_with(".parquet"), "{name}");
    assert_eq!(first_text(&s, "select c from t"), "a  ");
}

#[test]
fn an_explicit_cast_drops_the_declared_type() {
    let s = fresh();
    run(&s, "insert into t values ('a', 'x')").unwrap();
    // `CAST(c AS STRING)` is a plain string: the padded `'a  '` no longer equals `'a'` (Spark 4.2.0: no row).
    assert_eq!(
        count(
            &s,
            "select * from (select cast(c as string) as x from t) q where x = 'a'"
        ),
        0
    );
    assert_eq!(
        count(
            &s,
            "select * from (select try_cast(c as string) as x from t) q where x = 'a'"
        ),
        0
    );
    // A view over the cast stores `string` (`v_view_cast_desc` in the char/varchar goldens).
    run(&s, "drop view if exists vc").unwrap();
    run(&s, "create view vc as select cast(c as char(5)) c5 from t").unwrap();
    assert_eq!(first_text(&s, "describe table vc"), "c5");
    assert_eq!(text_at(&s, "describe table vc", 1), "string");
    // A permanent view of the bare column keeps it: its read projection's cast is the engine's own.
    run(&s, "drop view if exists vb").unwrap();
    run(&s, "create view vb as select c from t").unwrap();
    assert_eq!(count(&s, "select * from vb where c = 'a'"), 1);
    assert_eq!(text_at(&s, "describe table vb", 1), "char(3)");
}

#[test]
fn a_dataframe_temp_view_keeps_the_declared_type() {
    let s = fresh();
    run(&s, "insert into t values ('a', 'x')").unwrap();
    block_on(async { s.table("t").create_or_replace_temp_view("dv").await }).unwrap();
    assert_eq!(count(&s, "select * from dv where c = 'a'"), 1);
    // Through an alias and a filter in the stored plan too.
    block_on(async {
        s.sql("select c as z from t where v = 'x'")
            .await?
            .create_or_replace_temp_view("dv2")
            .await
    })
    .unwrap();
    assert_eq!(count(&s, "select * from dv2 where z = 'a'"), 1);
    // A cast in the stored plan drops it, as in a query.
    block_on(async {
        s.sql("select cast(c as string) as z from t")
            .await?
            .create_or_replace_temp_view("dv3")
            .await
    })
    .unwrap();
    assert_eq!(count(&s, "select * from dv3 where z = 'a'"), 0);
}

/// Spark 4.2.0 (local run 2026-11-07): a DataFrame temp view over a permanent view over `CHAR(3)` keeps the
/// declared type, in `DESCRIBE` and in comparison padding.
#[test]
fn a_dataframe_temp_view_over_a_permanent_view_keeps_the_declared_type() {
    let s = fresh();
    run(&s, "insert into t values ('a', 'x'), ('ab', 'y')").unwrap();
    run(&s, "drop view if exists pv").unwrap();
    run(&s, "create view pv as select c, v from t").unwrap();
    block_on(async { s.table("pv").create_or_replace_temp_view("tv").await }).unwrap();
    assert_eq!(text_at(&s, "describe table tv", 1), "char(3)");
    assert_eq!(count(&s, "select * from tv where c = 'a'"), 1);
    block_on(async {
        s.sql("select c from pv")
            .await?
            .create_or_replace_temp_view("tv2")
            .await
    })
    .unwrap();
    assert_eq!(text_at(&s, "describe table tv2", 1), "char(3)");
    assert_eq!(count(&s, "select * from tv2 where c = 'a'"), 1);
}

#[test]
fn between_over_a_char_column_does_not_pad_its_bounds() {
    // Spark 4.x rewrites `BETWEEN` before the padding rule sees a comparison, so the bounds stay unpadded
    // (`cmp_default_c3_between` in the char/varchar goldens: 4.0.0-4.2.0 drop the `'ab'` row, 3.5.7 keeps it).
    let s = fresh();
    run(&s, "insert into t values ('a', 'x'), ('ab', 'y')").unwrap();
    assert_eq!(
        all_text(
            &s,
            "select c from t where c between 'a' and 'ab' order by c"
        ),
        "a  \t"
    );
    assert_eq!(count(&s, "select * from t where c between 'a' and 'a'"), 0);
}

#[test]
fn review_cases_grouped_aliases_user_rpad_and_oversized_lengths() {
    let s = fresh();
    run(&s, "insert into t values ('a', 'x')").unwrap();
    // An explicit GROUP BY keeps the declared type on its select-list alias.
    assert_eq!(
        count(
            &s,
            "select * from (select c as x from t group by c) q where x = 'a'"
        ),
        1
    );
    // A user's rpad over the column is a plain string in a DataFrame temp view.
    block_on(async {
        s.sql("select rpad(c, greatest(length(v), 1), '*') as z from t")
            .await?
            .create_or_replace_temp_view("rp")
            .await
    })
    .unwrap();
    // `rpad('a  ', 1, '*')` is `'a'`; as a string it equals `'a'` unpadded.
    assert_eq!(count(&s, "select * from rp where z = 'a'"), 1);
    // A length past `INT` is refused (Spark: an unclassified NumberFormatException).
    let e = run(&s, "create table tbig(c varchar(2147483648)) using parquet").unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn equal_null_and_lateral_aliases_pad_char_comparisons() {
    // `z_default_eq_funcs` (Spark 4.2.0): `equal_null` pads as `<=>` does.
    let s = fresh();
    run(&s, "insert into t values ('a', 'x')").unwrap();
    assert_eq!(count(&s, "select * from t where equal_null(c, 'a')"), 1);
    // A later select item reading an earlier CHAR alias compares it padded.
    assert_eq!(
        count(
            &s,
            "select * from (select c as x, x = 'a' as matched from t) q where matched"
        ),
        1
    );
}

#[test]
fn stored_cte_references_wide_chars_and_linked_partitions() {
    let s = fresh();
    run(&s, "insert into t values ('a', 'x')").unwrap();
    // A DataFrame temp view over a CTE keeps the declared type through the stored CTE reference.
    block_on(async {
        s.sql("with w as (select c from t) select c from w")
            .await?
            .create_or_replace_temp_view("cv")
            .await
    })
    .unwrap();
    assert_eq!(count(&s, "select * from cv where c = 'a'"), 1);
    // A comparison over the widest CHAR analyzes without building the padded literal.
    run(&s, "drop table if exists wide").unwrap();
    run(&s, "create table wide(c char(2147483647)) using parquet").unwrap();
    assert_eq!(count(&s, "select * from wide where c = 'a'"), 0);
    // MSCK REPAIR follows a symbolic link to a partition directory, as Spark's Hadoop listing does.
    let dir = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    run(&s, "drop table if exists pl").unwrap();
    run(
        &s,
        &format!(
            "create table pl(a string, k string) using parquet partitioned by (k) location '{}'",
            dir.path().display()
        ),
    )
    .unwrap();
    std::os::unix::fs::symlink(elsewhere.path(), dir.path().join("k=x")).unwrap();
    run(&s, "msck repair table pl").unwrap();
    assert_eq!(first_text(&s, "show partitions pl"), "k=x");
}

/// The CHAR/VARCHAR switches Krishiv does not implement are refused, not ignored (design
/// `P2-T001b-char-varchar.md` point (b), ADR 0008), at `SET` and on the session builder; their
/// defaults are fine.
#[test]
fn unimplemented_char_varchar_switches_are_refused() {
    let s = fresh();
    for (key, refused, default) in [
        ("spark.sql.charAsVarchar", "true", "false"),
        ("spark.sql.preserveCharVarcharTypeInfo", "true", "false"),
        ("spark.sql.readSideCharPadding", "false", "true"),
        ("spark.sql.legacy.noCharPaddingInPredicate", "true", "false"),
    ] {
        let e = run(&s, &format!("SET {key} = {refused}")).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{key}"
        );
        run(&s, &format!("SET {key} = {default}")).unwrap();
        let built = Session::builder().config(key, refused).build();
        assert!(built.is_err(), "{key} on the builder");
    }
}

/// The rows below are a real Spark 4.2.0 run's (`AnalyzeColumnCommand`, `DescribeColumnCommand`).
#[test]
fn column_statistics_are_stored_and_described() {
    let s = fresh();
    run(&s, "drop table if exists st").unwrap();
    run(
        &s,
        "create table st(b boolean, i int, l bigint, f float, db double, dc decimal(10,2), x string, bi binary, dt date, ts timestamp, tn timestamp_ntz, c char(3)) using parquet",
    )
    .unwrap();
    run(
        &s,
        "insert into st values (true,1,1,1.5,2.5,1.25,'abc',X'01',date'2020-01-02',timestamp'2020-01-02 03:04:05',timestamp_ntz'2020-01-02 03:04:05.5','a'),(false,5,50000000000,-1.0,1e10,-3.50,'héllo',X'0203',date'1969-12-31',timestamp'1969-12-31 23:59:59.123456',timestamp_ntz'2021-01-02 03:04:05','bcd'),(null,null,null,null,null,null,null,null,null,null,null,null)",
    )
    .unwrap();
    run(&s, "set spark.sql.session.timeZone=UTC").unwrap();
    run(
        &s,
        "analyze table st compute statistics for columns b,i,l,f,db,dc,x,bi,dt,ts,tn,c",
    )
    .unwrap();
    let stats = |s: &Session, sql: &str| -> Vec<String> {
        let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
        b.iter()
            .flat_map(|batch| {
                (3..batch.num_rows())
                    .map(|i| {
                        let a = batch.column(1);
                        a.as_string_opt::<i32>().map_or_else(
                            || a.as_string_view().value(i).to_owned(),
                            |x| x.value(i).to_owned(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let row = |v: [&str; 6]| {
        let mut out: Vec<String> = v.iter().map(ToString::to_string).collect();
        out.push("NULL".to_owned());
        out
    };
    let e = |c: &str| format!("describe extended st {c}");
    assert_eq!(
        stats(&s, &e("b")),
        row(["false", "true", "1", "2", "1", "1"])
    );
    assert_eq!(stats(&s, &e("i")), row(["1", "5", "1", "2", "4", "4"]));
    assert_eq!(
        stats(&s, &e("l")),
        row(["1", "50000000000", "1", "2", "8", "8"])
    );
    assert_eq!(stats(&s, &e("f")), row(["-1.0", "1.5", "1", "2", "4", "4"]));
    assert_eq!(
        stats(&s, &e("db")),
        row(["2.5", "1.0E10", "1", "2", "8", "8"])
    );
    assert_eq!(
        stats(&s, &e("dc")),
        row(["-3.50", "1.25", "1", "2", "8", "8"])
    );
    assert_eq!(
        stats(&s, &e("x")),
        row(["NULL", "NULL", "1", "2", "4", "5"])
    );
    assert_eq!(
        stats(&s, "describe formatted st bi"),
        row(["NULL", "NULL", "1", "2", "2", "2"])
    );
    assert_eq!(
        stats(&s, &e("dt")),
        row(["1969-12-31", "2020-01-02", "1", "2", "4", "4"])
    );
    assert_eq!(
        stats(&s, &e("ts")),
        row([
            "1969-12-31 23:59:59.123456 +0000",
            "2020-01-02 03:04:05.000000 +0000",
            "1",
            "2",
            "8",
            "8"
        ])
    );
    assert_eq!(
        stats(&s, &e("tn")),
        row([
            "2020-01-02 03:04:05.500000",
            "2021-01-02 03:04:05.000000",
            "1",
            "2",
            "8",
            "8"
        ])
    );
    run(&s, "set spark.sql.session.timeZone=Asia/Tokyo").unwrap();
    assert_eq!(
        stats(&s, &e("ts")),
        row([
            "1970-01-01 08:59:59.123456 +0900",
            "2020-01-02 12:04:05.000000 +0900",
            "1",
            "2",
            "8",
            "8"
        ])
    );
    run(&s, "set spark.sql.session.timeZone=UTC").unwrap();
    // Plain `DESCRIBE` has no statistics rows.
    assert!(stats(&s, "describe st i").is_empty());

    // An empty table, then NaN and NULL-only columns.
    run(&s, "drop table if exists e1").unwrap();
    run(
        &s,
        "create table e1(i int, x string, f double) using parquet",
    )
    .unwrap();
    run(&s, "analyze table e1 compute statistics for all columns").unwrap();
    assert_eq!(
        stats(&s, "describe extended e1 i"),
        row(["NULL", "NULL", "0", "0", "4", "4"])
    );
    assert_eq!(
        stats(&s, "describe extended e1 x"),
        row(["NULL", "NULL", "0", "0", "20", "20"])
    );
    run(
        &s,
        "insert into e1 values (null,null,double('NaN')),(null,null,1.0)",
    )
    .unwrap();
    run(&s, "analyze table e1 compute statistics for columns i,x,f").unwrap();
    assert_eq!(
        stats(&s, "describe extended e1 i"),
        row(["NULL", "NULL", "2", "0", "4", "4"])
    );
    assert_eq!(
        stats(&s, "describe extended e1 x"),
        row(["NULL", "NULL", "2", "0", "20", "20"])
    );
    assert_eq!(
        stats(&s, "describe extended e1 f"),
        row(["1.0", "NaN", "0", "2", "8", "8"])
    );

    // A write drops all of them; a later ANALYZE restores only what it names.
    let none = || row(["NULL", "NULL", "NULL", "NULL", "NULL", "NULL"]);
    run(&s, "insert into e1 values (7,'x',2)").unwrap();
    assert_eq!(stats(&s, "describe extended e1 i"), none());
    run(&s, "analyze table e1 compute statistics for columns x").unwrap();
    assert_eq!(stats(&s, "describe extended e1 i"), none());
    assert_eq!(
        stats(&s, "describe extended e1 x"),
        row(["NULL", "NULL", "2", "1", "1", "1"])
    );
    run(&s, "truncate table e1").unwrap();
    assert_eq!(stats(&s, "describe extended e1 x"), none());

    // The HyperLogLog estimate is Spark's; a rename and a new column keep the statistics.
    run(&s, "drop table if exists e3").unwrap();
    run(&s, "drop table if exists e4").unwrap();
    run(&s, "create table e3(x string, i int) using parquet").unwrap();
    run(
        &s,
        "insert into e3 select cast(id % 5 as string), cast(id as int) from range(100)",
    )
    .unwrap();
    run(&s, "analyze table e3 compute statistics for columns x, i").unwrap();
    assert_eq!(
        stats(&s, "describe extended e3 i"),
        row(["0", "99", "0", "99", "4", "4"])
    );
    assert_eq!(
        stats(&s, "describe extended e3 x"),
        row(["NULL", "NULL", "0", "5", "1", "1"])
    );
    run(&s, "alter table e3 rename to e4").unwrap();
    run(&s, "alter table e4 add columns (z int)").unwrap();
    assert_eq!(
        stats(&s, "describe extended e4 i"),
        row(["0", "99", "0", "99", "4", "4"])
    );

    // Partition columns are columns too.
    run(&s, "drop table if exists p9").unwrap();
    run(
        &s,
        "create table p9(a int, p int) using parquet partitioned by (p)",
    )
    .unwrap();
    run(&s, "insert into p9 values (1,1),(2,2)").unwrap();
    run(&s, "analyze table p9 compute statistics for columns a,p").unwrap();
    assert_eq!(
        stats(&s, "describe extended p9 p"),
        row(["1", "2", "0", "2", "4", "4"])
    );
}

/// Fixture `ddl_temp_view_desc` (Spark 4.2.0): `DESCRIBE` of a temporary view over a table with
/// `CHAR`/`VARCHAR` columns prints their declared types (3.5.7 to 4.1.0 print `string`; review of #496 M1).
#[test]
fn a_temporary_view_describes_its_char_varchar_columns() {
    let s = Session::new().unwrap();
    run(
        &s,
        "CREATE TABLE t_cmp(c3 char(3), c5 char(5), v3 varchar(3), v5 varchar(5), s string) USING parquet",
    )
    .unwrap();
    run(&s, "CREATE TEMPORARY VIEW tv_cv AS SELECT * FROM t_cmp").unwrap();
    let df = block_on(s.sql("DESCRIBE TABLE tv_cv")).unwrap();
    let mut got = Vec::new();
    for b in block_on(df.collect()).unwrap() {
        let (n, t) = (b.column(0).as_string_view(), b.column(1).as_string_view());
        for i in 0..b.num_rows() {
            got.push(format!("{} {}", n.value(i), t.value(i)));
        }
    }
    assert_eq!(
        got,
        [
            "c3 char(3)",
            "c5 char(5)",
            "v3 varchar(3)",
            "v5 varchar(5)",
            "s string"
        ]
    );
}

/// The rows below are a real Spark 4.2.0 run's: struct fields whose names differ only in case compare, and the
/// equal lengths need no padding.
#[test]
fn nested_char_comparison_ignores_field_name_case() {
    let s = fresh();
    run(&s, "drop table if exists nc").unwrap();
    run(
        &s,
        "create table nc(i int, a struct<X:char(3)>, b struct<x:char(3)>) using parquet",
    )
    .unwrap();
    run(
        &s,
        "insert into nc values (1, named_struct('X','a'), named_struct('x','a')), (2, named_struct('X','ab'), named_struct('x','b'))",
    )
    .unwrap();
    assert_eq!(count(&s, "select i from nc where a = b"), 1);
    assert_eq!(
        first_text(&s, "select cast(i as string) from nc where a = b"),
        "1"
    );
}
