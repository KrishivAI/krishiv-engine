// SPDX-License-Identifier: Apache-2.0
//! Partitioned and bucketed tables, their partitions (`ALTER TABLE ... ADD PARTITION`, `INSERT ... PARTITION`,
//! `CREATE TABLE ... PARTITIONED BY ... AS SELECT`, `SHOW PARTITIONS`, `ANALYZE TABLE ... PARTITION`,
//! `DESCRIBE ... PARTITION`) and the `ALTER TABLE ... CHANGE | ALTER COLUMN` forms, against the Spark 4.2 goldens
//! `describe`, `describe-part-after-analyze`, `change-column`, `explain`, `identifier-clause` and
//! `show-create-table` (rows as `SQLQueryTestSuite` prints them, cells joined by a tab and trimmed).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

/// A session whose warehouse is `dir`.
fn session(dir: &std::path::Path) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.to_str().unwrap())
        .unwrap();
    s
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

fn err(s: &Session, sql: &str) -> KrishivError {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql}: no error"),
        Err(e) => e,
    }
}

fn class(s: &Session, sql: &str) -> String {
    err(s, sql).error_class().name().to_owned()
}

/// The message parameters of the error `sql` raises, as `name=value` in order.
fn params(s: &Session, sql: &str) -> Vec<String> {
    err(s, sql)
        .message_parameters()
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect()
}

fn cell(c: &dyn Array, i: usize) -> String {
    if c.is_null(i) {
        return String::new();
    }
    match c.data_type() {
        ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
        ArrowType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
        other => panic!("unexpected type {other}"),
    }
}

/// Every row of `sql`, its cells joined by a tab and the trailing blanks trimmed.
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let cells: Vec<String> = (0..b.num_columns())
                .map(|c| cell(b.column(c).as_ref(), i))
                .collect();
            out.push(cells.join("\t").trim_end().to_owned());
        }
    }
    out
}

/// The `(key, value)` rows of a `DESCRIBE`, with the values Spark's golden masks dropped.
fn described(s: &Session, sql: &str) -> Vec<String> {
    rows(s, sql)
        .into_iter()
        .map(|r| {
            for masked in [
                "Created Time",
                "Last Access",
                "Created By",
                "Location",
                "Partition Statistics",
            ] {
                if r.starts_with(masked) {
                    return masked.to_owned();
                }
            }
            r
        })
        .collect()
}

const PARTITION_INFO: [&str; 4] = [
    "# Partition Information",
    "# col_name\tdata_type\tcomment",
    "c\tstring",
    "d\tstring",
];

#[test]
fn partition_columns_go_last_and_describe_lists_them() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (c STRING, a STRING, d STRING, b INT) USING parquet PARTITIONED BY (C, d)",
    );
    let mut want = vec!["a\tstring", "b\tint", "c\tstring", "d\tstring"];
    want.extend(PARTITION_INFO);
    assert_eq!(rows(&s, "DESCRIBE t"), want);
    assert_eq!(rows(&s, "DESC TABLE default.t"), want);
    // A typed partition column is added after the columns (explain.sql's `explain_temp5`).
    run(
        &s,
        "CREATE table  explain_temp5 (key int) USING PARQUET PARTITIONED BY(val string)",
    );
    assert_eq!(
        rows(&s, "DESC explain_temp5"),
        [
            "key\tint",
            "val\tstring",
            "# Partition Information",
            "# col_name\tdata_type\tcomment",
            "val\tstring"
        ]
    );
    run(&s, "DROP TABLE explain_temp5");
    assert!(!dir.path().join("explain_temp5").exists());
}

#[test]
fn partition_columns_are_checked() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    assert_eq!(
        class(
            &s,
            "CREATE TABLE t (a INT, b INT) USING parquet PARTITIONED BY (x)"
        ),
        "COLUMN_NOT_DEFINED_IN_TABLE"
    );
    assert_eq!(
        class(
            &s,
            "CREATE TABLE t (a INT, b INT, c INT) USING parquet PARTITIONED BY (a) \
             CLUSTERED BY (b) SORTED BY (a) INTO 2 BUCKETS"
        ),
        "_LEGACY_ERROR_TEMP_1167"
    );
    assert_eq!(
        params(
            &s,
            "CREATE TABLE t (a INT, b INT) USING parquet PARTITIONED BY (x)"
        ),
        [
            "colType=partition",
            "colName=`x`",
            "tableName=`spark_catalog`.`default`.`t`",
            "tableCols=`a`, `b`"
        ]
    );
    assert_eq!(
        class(
            &s,
            "CREATE TABLE t (a INT, b INT) USING parquet PARTITIONED BY (a, b)"
        ),
        "ALL_PARTITION_COLUMNS_NOT_ALLOWED"
    );
    assert_eq!(
        class(
            &s,
            "CREATE TABLE t (a INT, b ARRAY<INT>) USING parquet PARTITIONED BY (b)"
        ),
        "INVALID_PARTITION_COLUMN_DATA_TYPE"
    );
    assert_eq!(
        class(
            &s,
            "CREATE TABLE t (a INT, b INT) USING parquet PARTITIONED BY (a, A)"
        ),
        "COLUMN_ALREADY_EXISTS"
    );
    assert_eq!(
        class(
            &s,
            "CREATE TABLE t (a INT, b INT, c INT) USING parquet PARTITIONED BY (a) \
             CLUSTERED BY (a) INTO 2 BUCKETS"
        ),
        "_LEGACY_ERROR_TEMP_1166"
    );
    // Nothing was created.
    assert!(!dir.path().join("t").exists());
}

/// describe.sql: a partitioned, bucketed table with options, a comment and properties.
fn describe_table(s: &Session) {
    run(
        s,
        "CREATE TABLE t (a STRING, b INT, c STRING, d STRING) USING parquet
  OPTIONS (a '1', b '2', password 'password')
  PARTITIONED BY (c, d) CLUSTERED BY (a) SORTED BY (b ASC) INTO 2 BUCKETS
  COMMENT 'table_comment'
  TBLPROPERTIES (t 'test', password 'password')",
    );
    run(s, "CREATE TEMPORARY VIEW temp_v AS SELECT * FROM t");
    run(s, "CREATE VIEW v AS SELECT * FROM t");
    run(s, "ALTER TABLE t ADD PARTITION (c='Us', d=1)");
}

fn columns_and_partitions() -> Vec<&'static str> {
    let mut want = vec!["a\tstring", "b\tint", "c\tstring", "d\tstring"];
    want.extend(PARTITION_INFO);
    want
}

#[test]
fn describe_extended_shows_buckets_storage_and_partition_provider() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    describe_table(&s);
    let mut want = columns_and_partitions();
    want.extend([
        "",
        "# Detailed Table Information",
        "Catalog\tspark_catalog",
        "Database\tdefault",
        "Table\tt",
        "Created Time",
        "Last Access",
        "Created By",
        "Type\tMANAGED",
        "Provider\tparquet",
        "Num Buckets\t2",
        "Bucket Columns\t[`a`]",
        "Sort Columns\t[`b`]",
        "Comment\ttable_comment",
        "Table Properties\t[password=*********(redacted), t=test]",
        "Location",
        "Storage Properties\t[a=1, b=2, password=*********(redacted)]",
        "Partition Provider\tCatalog",
    ]);
    assert_eq!(described(&s, "DESC EXTENDED t"), want);
    assert_eq!(described(&s, "DESC FORMATTED t"), want);
    // The partition directory was created.
    assert!(dir.path().join("t").join("c=Us").join("d=1").is_dir());
}

#[test]
fn describe_partition() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    describe_table(&s);
    assert_eq!(
        rows(&s, "DESC t PARTITION (c='Us', d=1)"),
        columns_and_partitions()
    );
    let mut want = columns_and_partitions();
    want.extend([
        "",
        "# Detailed Partition Information",
        "Database\tdefault",
        "Table\tt",
        "Partition Values\t[c=Us, d=1]",
        "Location",
        "Storage Properties\t[a=1, b=2, password=*********(redacted)]",
        "Created Time",
        "Last Access",
        "",
        "# Storage Information",
        "Num Buckets\t2",
        "Bucket Columns\t[`a`]",
        "Sort Columns\t[`b`]",
        "Location",
        "Storage Properties\t[a=1, b=2, password=*********(redacted)]",
    ]);
    assert_eq!(
        described(&s, "DESC EXTENDED t PARTITION (c='Us', d=1)"),
        want
    );
    assert_eq!(
        described(&s, "DESC FORMATTED t PARTITION (c='Us', d=1)"),
        want
    );
    assert_eq!(
        described(&s, "DESC EXTENDED t PARTITION (C='Us', D=1)"),
        want
    );
    let location = rows(&s, "DESC EXTENDED t PARTITION (c='Us', d=1)")
        .into_iter()
        .find(|r| r.starts_with("Location"))
        .unwrap();
    assert!(location.ends_with("/t/c=Us/d=1"), "{location}");
    assert_eq!(
        params(&s, "DESC t PARTITION (c='Us', d=2)"),
        [
            "partitionList=PARTITION (`c` = Us, `d` = 2)",
            "tableName=`default`.`t`"
        ]
    );
    assert_eq!(
        params(&s, "DESC t PARTITION (c='Us')"),
        [
            "specKeys=c",
            "partitionColumnNames=c, d",
            "tableName=`spark_catalog`.`default`.`t`"
        ]
    );
    assert_eq!(
        class(&s, "DESC t PARTITION (c='Us', d)"),
        "INVALID_SQL_SYNTAX.EMPTY_PARTITION_VALUE"
    );
    assert_eq!(
        class(&s, "DESC temp_v PARTITION (c='Us', d=1)"),
        "FORBIDDEN_OPERATION"
    );
    assert_eq!(
        class(&s, "DESC t PARTITION (c='Us')"),
        "_LEGACY_ERROR_TEMP_1232"
    );
    assert_eq!(
        class(&s, "ALTER TABLE t ADD PARTITION (c='Us', d=1)"),
        "PARTITIONS_ALREADY_EXIST"
    );
    assert_eq!(
        params(&s, "DESC temp_v PARTITION (c='Us', d=1)"),
        [
            "statement=DESC PARTITION",
            "objectType=TEMPORARY VIEW",
            "objectName=`temp_v`"
        ]
    );
    assert_eq!(
        params(&s, "DESC v PARTITION (c='Us', d=1)"),
        [
            "statement=DESC PARTITION",
            "objectType=VIEW",
            "objectName=`v`"
        ]
    );
    // An unknown key.
    assert_eq!(
        class(&s, "DESC t PARTITION (x='Us', d=1)"),
        "PARTITIONS_NOT_FOUND"
    );
    // Adding it again.
    assert_eq!(
        params(&s, "ALTER TABLE t ADD PARTITION (c='Us', d=1)"),
        [
            "partitionList=PARTITION (`c` = Us, `d` = 1)",
            "tableName=`default`.`t`"
        ]
    );
    run(
        &s,
        "ALTER TABLE t ADD IF NOT EXISTS PARTITION (c='Us', d=1)",
    );
}

#[test]
fn bucketed_and_partitioned_tables_refuse_partitions_where_spark_does() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE p (a INT, b INT) USING parquet");
    assert_eq!(
        class(&s, "SHOW PARTITIONS p"),
        "INVALID_PARTITION_OPERATION.PARTITION_SCHEMA_IS_EMPTY"
    );
    assert_eq!(
        params(&s, "SHOW PARTITIONS p"),
        ["name=`spark_catalog`.`default`.`p`"]
    );
    assert_eq!(
        class(&s, "DESC p PARTITION (a=1)"),
        "_LEGACY_ERROR_TEMP_1251"
    );
}

#[test]
fn show_create_table_prints_partitioning_and_buckets() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE tbl (a INT, b STRING, c INT) USING parquet\nPARTITIONED BY (a)",
    );
    assert_eq!(
        rows(&s, "SHOW CREATE TABLE tbl"),
        [
            "CREATE TABLE spark_catalog.default.tbl (\n  b STRING COLLATE UTF8_BINARY,\n  c INT,\n  a INT)\nUSING parquet\nPARTITIONED BY (a)"
        ]
    );
    run(&s, "DROP TABLE tbl");
    run(
        &s,
        "CREATE TABLE tbl (a INT, b STRING, c INT) USING parquet\nCLUSTERED BY (a) SORTED BY (b ASC) INTO 2 BUCKETS",
    );
    assert_eq!(
        rows(&s, "SHOW CREATE TABLE tbl"),
        [
            "CREATE TABLE spark_catalog.default.tbl (\n  a INT,\n  b STRING COLLATE UTF8_BINARY,\n  c INT)\nUSING parquet\nCLUSTERED BY (a)\nSORTED BY (b)\nINTO 2 BUCKETS"
        ]
    );
    // Bucketed writes are not supported: refused before anything is written.
    assert_eq!(
        class(&s, "INSERT INTO tbl VALUES (1, 'x', 2)"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

/// describe-part-after-analyze.sql.
#[test]
fn insert_into_partitions_and_analyze_them() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (key STRING, value STRING, ds STRING, hr INT) USING parquet
    PARTITIONED BY (ds, hr)",
    );
    run(
        &s,
        "INSERT INTO TABLE t PARTITION (ds='2017-08-01', hr=10)
VALUES ('k1', 100), ('k2', 200), ('k3', 300)",
    );
    run(
        &s,
        "INSERT INTO TABLE t PARTITION (ds='2017-08-01', hr=11)
VALUES ('k1', 101), ('k2', 201), ('k3', 301), ('k4', 401)",
    );
    run(
        &s,
        "INSERT INTO TABLE t PARTITION (ds='2017-09-01', hr=5)
VALUES ('k1', 102), ('k2', 202)",
    );
    assert!(dir.path().join("t/ds=2017-08-01/hr=10").is_dir());
    assert_eq!(
        rows(&s, "SHOW PARTITIONS t"),
        [
            "ds=2017-08-01/hr=10",
            "ds=2017-08-01/hr=11",
            "ds=2017-09-01/hr=5"
        ]
    );
    let detail = |hr: &str, ds: &str, stats: bool| {
        let mut want = vec![
            "key\tstring",
            "value\tstring",
            "ds\tstring",
            "hr\tint",
            "# Partition Information",
            "# col_name\tdata_type\tcomment",
            "ds\tstring",
            "hr\tint",
            "",
            "# Detailed Partition Information",
            "Database\tdefault",
            "Table\tt",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        want.push(format!("Partition Values\t[ds={ds}, hr={hr}]"));
        want.extend(["Location", "Created Time", "Last Access"].map(str::to_owned));
        if stats {
            want.push("Partition Statistics".to_owned());
        }
        want.extend(["", "# Storage Information", "Location"].map(str::to_owned));
        want
    };
    let stats_of = |sql: &str| {
        rows(&s, sql)
            .into_iter()
            .filter(|r| r.starts_with("Partition Statistics"))
            .map(|r| {
                let (_, rest) = r.split_once(" bytes, ").unwrap();
                format!("Partition Statistics {}", rest.trim())
            })
            .collect::<Vec<_>>()
    };
    let p10 = "DESC EXTENDED t PARTITION (ds='2017-08-01', hr=10)";
    let p11 = "DESC EXTENDED t PARTITION (ds='2017-08-01', hr=11)";
    let p5 = "DESC EXTENDED t PARTITION (ds='2017-09-01', hr=5)";
    assert_eq!(described(&s, p10), detail("10", "2017-08-01", false));
    run(
        &s,
        "ANALYZE TABLE t PARTITION (ds='2017-08-01', hr=10) COMPUTE STATISTICS",
    );
    assert_eq!(
        described(&s, p10),
        detail("10", "2017-08-01", true),
        "the statistics row follows Last Access"
    );
    assert_eq!(stats_of(p10), ["Partition Statistics 3 rows"]);
    assert!(stats_of(p11).is_empty());
    run(
        &s,
        "ANALYZE TABLE t PARTITION (ds='2017-08-01') COMPUTE STATISTICS",
    );
    assert_eq!(stats_of(p11), ["Partition Statistics 4 rows"]);
    assert!(stats_of(p5).is_empty());
    run(&s, "ANALYZE TABLE t PARTITION (ds, hr) COMPUTE STATISTICS");
    assert_eq!(stats_of(p5), ["Partition Statistics 2 rows"]);
    assert_eq!(
        class(
            &s,
            "ANALYZE TABLE t PARTITION (ds='2017-10-01') COMPUTE STATISTICS"
        ),
        "PARTITIONS_NOT_FOUND"
    );
    assert_eq!(
        class(&s, "ANALYZE TABLE t PARTITION (hr=5) COMPUTE STATISTICS"),
        "_LEGACY_ERROR_TEMP_1237"
    );
    run(&s, "DROP TABLE t");
    assert!(!dir.path().join("t").exists());
}

/// describe.sql's `f` (in Parquet and with a string column: the engine writes no JSON and cannot render a
/// binary partition value yet): a CTAS registers the partitions it wrote with their file counts and sizes.
#[test]
fn ctas_registers_partitions_with_file_statistics() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    // Timestamp partition values render in a fixed-offset session time zone only.
    s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
    run(
        &s,
        "CREATE TABLE f USING parquet PARTITIONED BY (B, C) AS SELECT 'APACHE' A, 'SPARK' B, \
         TIMESTAMP'2018-11-17 13:33:33' C",
    );
    let got = rows(
        &s,
        "DESC FORMATTED f PARTITION (B='SPARK', C=TIMESTAMP'2018-11-17 13:33:33')",
    );
    let at = |key: &str| {
        got.iter()
            .find(|r| r.starts_with(key))
            .unwrap_or_else(|| panic!("{key} in {got:?}"))
            .clone()
    };
    assert_eq!(&got[..3], ["A\tstring", "B\tstring", "C\ttimestamp"]);
    assert_eq!(
        at("Partition Values"),
        "Partition Values\t[B=SPARK, C=2018-11-17 13:33:33]"
    );
    assert!(
        at("Location").ends_with("/f/B=SPARK/C=2018-11-17 13%3A33%3A33"),
        "{got:?}"
    );
    let params = at("Partition Parameters");
    let size = std::fs::read_dir(dir.path().join("f/B=SPARK/C=2018-11-17 13%3A33%3A33"))
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| !e.file_name().to_string_lossy().starts_with(['.', '_']))
        .map(|e| e.metadata().unwrap().len())
        .sum::<u64>();
    assert!(
        params.starts_with(&format!(
            "Partition Parameters\t[numFiles=1, totalSize={size}, transient_lastDdlTime="
        )),
        "{params}"
    );
    assert_eq!(
        rows(&s, "SHOW PARTITIONS f"),
        ["B=SPARK/C=2018-11-17 13%3A33%3A33"]
    );
}

/// change-column.sql and identifier-clause.sql.
#[test]
fn alter_table_change_column() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE test_change(a INT, b STRING, c INT) using parquet",
    );
    let unsupported = |sql: &str, operation: &str| {
        assert_eq!(
            params(&s, sql),
            [
                "tableName=`spark_catalog`.`default`.`test_change`".to_owned(),
                format!("operation={operation}"),
            ],
            "{sql}"
        );
    };
    unsupported(
        "ALTER TABLE test_change RENAME COLUMN a TO a1",
        "RENAME COLUMN",
    );
    unsupported("ALTER TABLE test_change DROP COLUMN a", "DROP COLUMN");
    unsupported(
        "ALTER TABLE test_change CHANGE a AFTER b",
        "ALTER COLUMN ... FIRST | AFTER",
    );
    unsupported(
        "ALTER TABLE test_change CHANGE b FIRST",
        "ALTER COLUMN ... FIRST | AFTER",
    );
    unsupported(
        "ALTER TABLE test_change ALTER COLUMN b SET NOT NULL",
        "ALTER COLUMN ... SET NOT NULL",
    );
    assert_eq!(
        class(&s, "ALTER TABLE test_change CHANGE a TYPE STRING"),
        "NOT_SUPPORTED_CHANGE_COLUMN"
    );
    assert_eq!(
        params(&s, "ALTER TABLE test_change CHANGE a TYPE STRING"),
        [
            "table=`spark_catalog`.`default`.`test_change`",
            "originName=`a`",
            "originType=\"INT\"",
            "newName=`a`",
            "newType=\"STRING\""
        ]
    );
    run(
        &s,
        "ALTER TABLE test_change CHANGE a COMMENT 'this is column a'",
    );
    run(&s, "ALTER TABLE test_change CHANGE b COMMENT '#*02?`'");
    run(&s, "ALTER TABLE test_change CHANGE c COMMENT ''");
    let want = ["a\tint\tthis is column a", "b\tstring\t#*02?`", "c\tint"];
    assert_eq!(rows(&s, "DESC test_change"), want);
    run(&s, "ALTER TABLE test_change CHANGE a TYPE INT");
    assert_eq!(rows(&s, "DESC test_change"), want);
    assert_eq!(
        params(&s, "ALTER TABLE test_change CHANGE invalid_col TYPE INT"),
        ["objectName=`invalid_col`", "proposal=`a`, `b`, `c`"]
    );
    run(
        &s,
        "ALTER TABLE test_change CHANGE A COMMENT 'case insensitivity'",
    );
    run(
        &s,
        "ALTER TABLE test_change ALTER COLUMN IDENTIFIER('c') COMMENT 'column comment'",
    );
    assert_eq!(
        rows(&s, "DESC test_change"),
        [
            "a\tint\tcase insensitivity",
            "b\tstring\t#*02?`",
            "c\tint\tcolumn comment"
        ]
    );
    run(
        &s,
        "CREATE TEMPORARY VIEW temp_view(a, b) AS SELECT 1, \"one\"",
    );
    assert_eq!(
        class(&s, "ALTER TABLE temp_view CHANGE a TYPE INT"),
        "EXPECT_TABLE_NOT_VIEW.NO_ALTERNATIVE"
    );
    // Partition columns cannot change.
    run(
        &s,
        "CREATE TABLE pt (a INT, b STRING) USING parquet PARTITIONED BY (b)",
    );
    assert_eq!(
        class(&s, "ALTER TABLE pt CHANGE b COMMENT 'x'"),
        "CANNOT_ALTER_PARTITION_COLUMN"
    );
    // New columns go before the partition columns.
    run(&s, "ALTER TABLE pt ADD COLUMNS (c INT)");
    assert_eq!(rows(&s, "DESC pt")[..3], ["a\tint", "c\tint", "b\tstring"]);
}

/// A renamed table keeps its partitions under its new directory; `LIKE` copies the layout; unsetting the
/// `comment` property drops the table comment (describe.sql).
#[test]
fn rename_like_and_comment() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (a INT, c STRING) USING parquet PARTITIONED BY (c) COMMENT 'x'",
    );
    run(&s, "ALTER TABLE t ADD PARTITION (c='Us')");
    run(&s, "ALTER TABLE t RENAME TO u");
    let location = rows(&s, "DESC EXTENDED u PARTITION (c='Us')")
        .into_iter()
        .find(|r| r.starts_with("Location"))
        .unwrap();
    assert!(location.ends_with("/u/c=Us"), "{location}");
    assert!(dir.path().join("u/c=Us").is_dir());
    run(&s, "CREATE TABLE v LIKE u");
    assert_eq!(
        rows(&s, "DESC v")[2..],
        [
            "# Partition Information",
            "# col_name\tdata_type\tcomment",
            "c\tstring"
        ]
    );
    assert_eq!(rows(&s, "SHOW PARTITIONS v"), Vec::<String>::new());
    assert!(rows(&s, "DESC EXTENDED u").contains(&"Comment\tx".to_owned()));
    run(&s, "ALTER TABLE u UNSET TBLPROPERTIES (comment)");
    assert!(
        !rows(&s, "DESC EXTENDED u")
            .iter()
            .any(|r| r.starts_with("Comment"))
    );
}

/// `partitionVal: identifier (EQ constant)? | identifier EQ DEFAULT` (SqlBaseParser.g4, v4.2.0): a value is a
/// literal, a negative number, a typed literal, NULL or a parameter marker; anything else is a syntax error.
#[test]
fn partition_values_are_constants() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (a INT, p INT, q STRING) USING parquet PARTITIONED BY (p, q)",
    );
    for (sql, near) in [
        ("ALTER TABLE t ADD PARTITION (p = 1 + 1, q = 'x')", "'+'"),
        (
            "ALTER TABLE t ADD PARTITION (p = rand(), q = 'x')",
            "'rand'",
        ),
        (
            "ALTER TABLE t ADD PARTITION (p = 1, q = current_date())",
            "'current_date'",
        ),
        (
            "ALTER TABLE t ADD PARTITION (p = (SELECT 1), q = 'x')",
            "'('",
        ),
        ("DESC t PARTITION (p = a, q = 'x')", "'a'"),
        (
            "INSERT INTO t PARTITION (p = 1, q = 'a' || 'b') VALUES (1)",
            "'||'",
        ),
    ] {
        let e = err(&s, sql);
        assert_eq!(e.error_class().name(), "PARSE_SYNTAX_ERROR", "{sql}");
        assert!(
            e.to_string().contains(&format!("near {near}")),
            "{sql}: {e}"
        );
    }
    // Constants of every kind.
    run(
        &s,
        "ALTER TABLE t ADD PARTITION (p = -1, q = DATE'2020-01-02')",
    );
    run(&s, "ALTER TABLE t ADD PARTITION (p = 2, q = NULL)");
    run(&s, "ALTER TABLE t ADD PARTITION (p = 3, q = true)");
    assert_eq!(
        rows(&s, "SHOW PARTITIONS t"),
        [
            "p=-1/q=2020-01-02",
            "p=2/q=__HIVE_DEFAULT_PARTITION__",
            "p=3/q=true"
        ]
    );
}

#[test]
fn add_partition_locations() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (a INT, p STRING) USING parquet PARTITIONED BY (p)",
    );
    let elsewhere = dir.path().join("elsewhere");
    run(
        &s,
        &format!(
            "ALTER TABLE t ADD PARTITION (p = 'x') LOCATION '{}'",
            elsewhere.display()
        ),
    );
    assert!(elsewhere.is_dir());
    let other = dir.path().join("other");
    run(
        &s,
        &format!(
            "ALTER TABLE t ADD PARTITION (p = 'y') LOCATION 'file:{}'",
            other.display()
        ),
    );
    assert!(other.is_dir());
    let location = |spec: &str| {
        rows(&s, &format!("DESC EXTENDED t PARTITION ({spec})"))
            .into_iter()
            .find(|r| r.starts_with("Location"))
            .unwrap()
    };
    assert_eq!(
        location("p = 'x'"),
        format!("Location\tfile:{}", elsewhere.display())
    );
    assert_eq!(
        location("p = 'y'"),
        format!("Location\tfile:{}", other.display())
    );
    // A relative location is made absolute against the working directory (Spark's `makeQualified`). The
    // directory `ADD PARTITION` creates there has a name of its own and is removed even if an assertion fails.
    struct RemoveOnDrop(std::path::PathBuf);
    impl Drop for RemoveOnDrop {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let rel = format!("krishiv-test-rel-{}", std::process::id());
    let created = RemoveOnDrop(std::env::current_dir().unwrap().join(&rel));
    run(
        &s,
        &format!("ALTER TABLE t ADD PARTITION (p = 'z') LOCATION '{rel}'"),
    );
    assert_eq!(
        location("p = 'z'"),
        format!("Location\tfile:{}", created.0.display())
    );
    assert!(created.0.is_dir());
    assert_eq!(rows(&s, "SHOW PARTITIONS t"), ["p=x", "p=y", "p=z"]);
}

/// Denies `CREATE` on any path with `outside` in it.
struct NoOutside;

impl krishiv_session::AccessPolicy for NoOutside {
    fn version(&self) -> krishiv_plan::analyzer::PolicyVersion {
        krishiv_plan::analyzer::PolicyVersion(0)
    }

    fn check<'a>(
        &'a self,
        _: &'a krishiv_session::Principal,
        securable: &'a krishiv_session::Securable,
        privilege: krishiv_session::Privilege,
    ) -> krishiv_plan::analyzer::BoxFuture<'a, Result<krishiv_session::AccessDecision, KrishivError>>
    {
        let denied = privilege == krishiv_session::Privilege::Create
            && matches!(securable, krishiv_session::Securable::Path(p)
                if p.as_plain().is_some_and(|t| t.contains("outside")));
        Box::pin(std::future::ready(Ok(if denied {
            krishiv_session::AccessDecision::Deny {
                reason: "outside".to_owned(),
            }
        } else {
            krishiv_session::AccessDecision::Allow
        })))
    }
}

#[test]
fn add_partition_outside_the_policy_is_denied() {
    use std::sync::Arc;
    let dir = tempfile::tempdir().unwrap();
    let warehouse = dir.path().join("wh");
    std::fs::create_dir_all(&warehouse).unwrap();
    let s = Session::builder()
        .access_policy(Arc::new(NoOutside))
        .config("spark.sql.warehouse.dir", warehouse.to_str().unwrap())
        .build()
        .unwrap();
    run(
        &s,
        "CREATE TABLE t (a INT, p STRING) USING parquet PARTITIONED BY (p)",
    );
    let outside = dir.path().join("outside");
    assert_eq!(
        class(
            &s,
            &format!(
                "ALTER TABLE t ADD PARTITION (p = 'x') LOCATION '{}'",
                outside.display()
            )
        ),
        "KRISHIV_INSUFFICIENT_PERMISSIONS"
    );
    assert!(!outside.exists());
    assert!(rows(&s, "SHOW PARTITIONS t").is_empty());
}

/// Denies `MODIFY` on any path with `p=locked` in it; allows everything else (table DDL, `CREATE` on paths).
struct LockedPartition;

impl krishiv_session::AccessPolicy for LockedPartition {
    fn version(&self) -> krishiv_plan::analyzer::PolicyVersion {
        krishiv_plan::analyzer::PolicyVersion(0)
    }

    fn check<'a>(
        &'a self,
        _: &'a krishiv_session::Principal,
        securable: &'a krishiv_session::Securable,
        privilege: krishiv_session::Privilege,
    ) -> krishiv_plan::analyzer::BoxFuture<'a, Result<krishiv_session::AccessDecision, KrishivError>>
    {
        let denied = privilege == krishiv_session::Privilege::Modify
            && matches!(securable, krishiv_session::Securable::Path(p)
                if p.as_plain().is_some_and(|t| t.contains("p=locked")));
        Box::pin(std::future::ready(Ok(if denied {
            krishiv_session::AccessDecision::Deny {
                reason: "locked".to_owned(),
            }
        } else {
            krishiv_session::AccessDecision::Allow
        })))
    }
}

/// DROP PARTITION deletes the partition's files and RENAME PARTITION moves them out of its directory: both need
/// `MODIFY` on that directory, as TRUNCATE and INSERT need it on the paths they change. A denial changes nothing.
#[test]
fn drop_or_rename_of_a_partition_whose_path_is_not_modifiable_is_denied() {
    use std::sync::Arc;
    let dir = tempfile::tempdir().unwrap();
    let warehouse = dir.path().join("wh");
    std::fs::create_dir_all(&warehouse).unwrap();
    let s = Session::builder()
        .access_policy(Arc::new(LockedPartition))
        .config("spark.sql.warehouse.dir", warehouse.to_str().unwrap())
        .build()
        .unwrap();
    run(
        &s,
        "CREATE TABLE t (a INT, p STRING) USING parquet PARTITIONED BY (p)",
    );
    run(&s, "ALTER TABLE t ADD PARTITION (p = 'locked')");
    run(&s, "ALTER TABLE t ADD PARTITION (p = 'free')");
    let locked = warehouse.join("t").join("p=locked");
    std::fs::write(locked.join("keep.txt"), "x").unwrap();
    for sql in [
        "ALTER TABLE t DROP PARTITION (p = 'locked')",
        "ALTER TABLE t PARTITION (p = 'locked') RENAME TO PARTITION (p = 'moved')",
    ] {
        assert_eq!(class(&s, sql), "KRISHIV_INSUFFICIENT_PERMISSIONS", "{sql}");
        assert!(locked.join("keep.txt").is_file(), "{sql}");
        assert_eq!(
            rows(&s, "SHOW PARTITIONS t"),
            ["p=free", "p=locked"],
            "{sql}"
        );
    }
    // Other partitions are not affected.
    run(&s, "ALTER TABLE t DROP PARTITION (p = 'free')");
    assert_eq!(rows(&s, "SHOW PARTITIONS t"), ["p=locked"]);
}

#[test]
fn msck_repair_of_a_denied_partition_directory_is_denied() {
    // MSCK REPAIR registers each partition at its directory, so it asks the policy as ADD PARTITION does.
    use std::sync::Arc;
    let dir = tempfile::tempdir().unwrap();
    let warehouse = dir.path().join("wh");
    std::fs::create_dir_all(&warehouse).unwrap();
    let s = Session::builder()
        .access_policy(Arc::new(NoOutside))
        .config("spark.sql.warehouse.dir", warehouse.to_str().unwrap())
        .build()
        .unwrap();
    run(
        &s,
        "CREATE TABLE t (a INT, p STRING) USING parquet PARTITIONED BY (p)",
    );
    let root = warehouse.join("t");
    std::fs::create_dir_all(root.join("p=inside")).unwrap();
    run(&s, "MSCK REPAIR TABLE t");
    assert_eq!(rows(&s, "SHOW PARTITIONS t"), ["p=inside"]);
    std::fs::create_dir_all(root.join("p=outside")).unwrap();
    assert_eq!(
        class(&s, "MSCK REPAIR TABLE t"),
        "KRISHIV_INSUFFICIENT_PERMISSIONS"
    );
    assert_eq!(rows(&s, "SHOW PARTITIONS t"), ["p=inside"]);
}

#[test]
fn partition_values_are_escaped_and_filtered() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (a INT, p STRING, q INT) USING parquet PARTITIONED BY (p, q)",
    );
    run(
        &s,
        "INSERT INTO t PARTITION (p = 'a/b=c%d', q = 1) VALUES (1)",
    );
    run(&s, "INSERT INTO t PARTITION (p = 'x', q = 2) VALUES (2)");
    run(&s, "INSERT INTO t PARTITION (p = NULL, q = 3) VALUES (3)");
    // Dynamic partition values come from the query.
    run(&s, "INSERT INTO t PARTITION (p = 'x', q) VALUES (4, 5)");
    assert!(dir.path().join("t/p=a%2Fb%3Dc%25d/q=1").is_dir());
    assert!(
        dir.path()
            .join("t/p=__HIVE_DEFAULT_PARTITION__/q=3")
            .is_dir()
    );
    assert_eq!(
        rows(&s, "SHOW PARTITIONS t"),
        [
            "p=__HIVE_DEFAULT_PARTITION__/q=3",
            "p=a%2Fb%3Dc%25d/q=1",
            "p=x/q=2",
            "p=x/q=5"
        ]
    );
    assert_eq!(
        rows(&s, "SHOW PARTITIONS t PARTITION (p = 'x')"),
        ["p=x/q=2", "p=x/q=5"]
    );
    assert_eq!(
        rows(&s, "DESC EXTENDED t PARTITION (p = 'a/b=c%d', q = 1)")
            .into_iter()
            .find(|r| r.starts_with("Partition Values"))
            .unwrap(),
        "Partition Values\t[p=a/b=c%d, q=1]"
    );
    assert_eq!(
        class(
            &s,
            "INSERT OVERWRITE t PARTITION (p = 'x', q = 2) VALUES (1)"
        ),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    // A failed insert registers nothing.
    assert_eq!(
        class(&s, "INSERT INTO t PARTITION (p = 'y', q = 9) SELECT 1 / 0"),
        "DIVIDE_BY_ZERO"
    );
    // A bare `INTERVAL` is no constant.
    assert_eq!(
        class(&s, "INSERT INTO t PARTITION (p = INTERVAL, q = 1) SELECT 1"),
        "PARSE_SYNTAX_ERROR"
    );
    // A written spec names every partition column (`PreprocessTableInsertion`).
    assert_eq!(
        class(&s, "INSERT INTO t PARTITION (p = 'y') SELECT 1"),
        "_LEGACY_ERROR_TEMP_1169"
    );
    assert!(
        !rows(&s, "SHOW PARTITIONS t")
            .iter()
            .any(|r| r.contains("q=9"))
    );
    // A catalog partition whose directory is gone stays.
    std::fs::remove_dir_all(dir.path().join("t/p=x/q=5")).unwrap();
    run(&s, "INSERT INTO t PARTITION (p = 'x', q = 2) VALUES (6)");
    assert_eq!(rows(&s, "SHOW PARTITIONS t PARTITION (p = 'x')").len(), 2);
}

#[test]
fn analyze_noscan_records_size_only() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (a INT, p STRING) USING parquet PARTITIONED BY (p)",
    );
    run(&s, "INSERT INTO t PARTITION (p = 'x') VALUES (1), (2)");
    run(
        &s,
        "ANALYZE TABLE t PARTITION (p = 'x') COMPUTE STATISTICS NOSCAN",
    );
    let stats = rows(&s, "DESC EXTENDED t PARTITION (p = 'x')")
        .into_iter()
        .find(|r| r.starts_with("Partition Statistics"))
        .unwrap();
    assert!(stats.ends_with(" bytes"), "{stats}");
    run(&s, "ANALYZE TABLE t PARTITION (p = 'x') COMPUTE STATISTICS");
    let stats = rows(&s, "DESC EXTENDED t PARTITION (p = 'x')")
        .into_iter()
        .find(|r| r.starts_with("Partition Statistics"))
        .unwrap();
    assert!(stats.ends_with(" bytes, 2 rows"), "{stats}");
}

#[test]
fn layouts_spark_refuses_or_copies() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    assert_eq!(
        class(
            &s,
            "CREATE TABLE c USING parquet PARTITIONED BY (p STRING) AS SELECT 1 AS a, 'x' AS p"
        ),
        // `Partition column types may not be specified in Create Table As Select (CTAS)`.
        "_LEGACY_ERROR_TEMP_0035"
    );
    // `PARTITION BY: Cannot mix partition expressions and partition columns` (`operationNotAllowed`).
    assert_eq!(
        class(
            &s,
            "CREATE TABLE m (a INT, b INT) USING parquet PARTITIONED BY (a, c STRING)"
        ),
        "_LEGACY_ERROR_TEMP_0035"
    );
    run(
        &s,
        "CREATE TABLE b (a INT, b STRING, c INT) USING parquet CLUSTERED BY (a) SORTED BY (b) INTO 4 BUCKETS",
    );
    run(&s, "CREATE TABLE b2 LIKE b");
    let buckets: Vec<String> = rows(&s, "DESC EXTENDED b2")
        .into_iter()
        .filter(|r| r.contains("Bucket") || r.starts_with("Sort Columns"))
        .collect();
    assert_eq!(
        buckets,
        [
            "Num Buckets\t4",
            "Bucket Columns\t[`a`]",
            "Sort Columns\t[`b`]"
        ]
    );
}

#[test]
fn alter_column_drop_default_and_comment_property() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE t (a INT DEFAULT 42, b INT) USING parquet");
    run(&s, "INSERT INTO t (b) VALUES (1)");
    run(&s, "ALTER TABLE t ALTER COLUMN a DROP DEFAULT");
    run(&s, "INSERT INTO t (b) VALUES (2)");
    assert_eq!(
        rows(
            &s,
            "SELECT coalesce(CAST(a AS STRING), 'null') FROM t ORDER BY b"
        ),
        ["42", "null"]
    );
    // describe-table-after-alter-table: the `comment` property is the table comment.
    run(
        &s,
        "ALTER TABLE t SET TBLPROPERTIES (comment = \"added comment\")",
    );
    let extended = rows(&s, "DESC EXTENDED t");
    assert!(extended.contains(&"Comment\tadded comment".to_owned()));
    assert!(!extended.iter().any(|r| r.starts_with("Table Properties")));
}

#[test]
fn partitioned_ctas_keeps_dotted_column_names() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE d USING parquet PARTITIONED BY (p) AS SELECT 1 AS `a.b`, 'x' AS p",
    );
    assert_eq!(rows(&s, "SHOW PARTITIONS d"), ["p=x"]);
}

/// `DESCRIBE EXTENDED ... AS JSON` writes the keys in `DescribeTableJsonCommand`'s order (the `describe` golden's
/// shape); a table without buckets or partitions leaves those keys out, and nested column types are refused.
#[test]
fn describe_as_json_lists_the_keys_in_spark_order() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE j (a STRING, b DECIMAL(5,2)) USING parquet COMMENT 'c' TBLPROPERTIES (password 'x')",
    );
    let lines = rows(&s, "DESCRIBE EXTENDED j AS JSON");
    assert_eq!(lines.len(), 1);
    let json = lines.join("");
    let keys = [
        "table_name",
        "catalog_name",
        "namespace",
        "schema_name",
        "columns",
        "location",
        "created_time",
        "last_access",
        "created_by",
        "type",
        "provider",
        "comment",
        "table_properties",
        "collation",
    ];
    let mut at = 0;
    for k in keys {
        let found = json[at..]
            .find(&format!("\"{k}\":"))
            .unwrap_or_else(|| panic!("{k} in {json}"));
        at += found;
    }
    assert!(json.contains(
        r#"{"name":"b","type":{"name":"decimal","precision":5,"scale":2},"nullable":true}"#
    ));
    assert!(json.contains(r#""table_properties":{"password":"*********(redacted)"}"#));
    assert!(!json.contains("partition_columns") && !json.contains("num_buckets"));
    // `AS JSON` needs `EXTENDED` or `FORMATTED` and takes no column.
    assert_eq!(
        class(&s, "DESCRIBE j AS JSON"),
        "DESCRIBE_JSON_NOT_EXTENDED"
    );
    assert_eq!(
        class(&s, "DESC FORMATTED j a AS JSON"),
        "UNSUPPORTED_FEATURE.DESC_TABLE_COLUMN_JSON"
    );
    run(&s, "CREATE TABLE n (a ARRAY<INT>) USING parquet");
    assert_eq!(
        class(&s, "DESCRIBE EXTENDED n AS JSON"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn describe_as_json_values_follow_spark() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE jp (a INT COMMENT 'first' DEFAULT 1, token STRING, v VOID) USING parquet \
         PARTITIONED BY (token) COMMENT 'tbl'",
    );
    run(&s, "ALTER TABLE jp ADD PARTITION (token='t1')");
    let one = |sql: &str| -> serde_json::Value {
        let lines = rows(&s, sql);
        assert_eq!(lines.len(), 1, "{sql}");
        serde_json::from_str(&lines[0]).unwrap_or_else(|e| panic!("{sql}: {e}: {}", lines[0]))
    };
    let t = one("DESCRIBE EXTENDED jp AS JSON");
    assert_eq!(
        t["columns"][0],
        serde_json::json!({"name":"a","type":{"name":"int"},"nullable":true,
                           "comment":"first","default":"1"})
    );
    assert_eq!(t["columns"][1]["type"], serde_json::json!({"name":"void"}));
    let created = t["created_time"].as_str().unwrap();
    assert!(
        created.len() == 20 && created.as_bytes()[10] == b'T' && created.ends_with('Z'),
        "{created}"
    );
    let p = one("DESCRIBE EXTENDED jp PARTITION (token='t1') AS JSON");
    assert_eq!(p["partition_values"], serde_json::json!({"token":"t1"}));
    assert_eq!(p["comment"], "tbl");
    assert!(p["created_time"].as_str().unwrap().ends_with('Z'));
}

#[test]
fn describe_as_json_of_bucketed_tables_statistics_and_refusals() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE jb (a INT, b TIMESTAMP, c BOOLEAN, d TINYINT, e SMALLINT, f BIGINT, g FLOAT, \
         h DOUBLE, i BINARY, j DATE, k TIMESTAMP_NTZ, p INT) USING parquet PARTITIONED BY (p) \
         CLUSTERED BY (a) SORTED BY (b) INTO 4 BUCKETS",
    );
    run(&s, "ALTER TABLE jb ADD PARTITION (p=1)");
    run(&s, "ANALYZE TABLE jb PARTITION (p=1) COMPUTE STATISTICS");
    let one = |sql: &str| -> serde_json::Value {
        let lines = rows(&s, sql);
        assert_eq!(lines.len(), 1, "{sql}");
        serde_json::from_str(&lines[0]).unwrap()
    };
    let t = one("DESCRIBE EXTENDED jb AS JSON");
    assert_eq!(t["num_buckets"], 4);
    assert_eq!(t["bucket_columns"], serde_json::json!(["a"]));
    assert_eq!(t["sort_columns"], serde_json::json!(["b"]));
    let names: Vec<_> = t["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["type"]["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        names[..11],
        [
            "int",
            "timestamp_ltz",
            "boolean",
            "byte",
            "short",
            "long",
            "float",
            "double",
            "binary",
            "date",
            "timestamp_ntz"
        ]
    );
    let p = one("DESCRIBE EXTENDED jb PARTITION (p=1) AS JSON");
    assert!(
        p["partition_statistics"]
            .as_str()
            .unwrap()
            .contains("bytes")
    );
    assert_eq!(p["num_buckets"], 4);
    run(&s, "CREATE TEMPORARY VIEW tv AS SELECT 1 AS x");
    assert_eq!(
        class(&s, "DESCRIBE EXTENDED tv AS JSON"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    run(&s, "CREATE VIEW pv AS SELECT 1 AS x");
    assert_eq!(
        class(&s, "DESCRIBE EXTENDED pv AS JSON"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

/// A partitioned table's rows read back with their partition values: the `name=value` directories above its files
/// are its declared partition columns, last and typed as declared (Spark reads a catalog table's partition values
/// with its partition schema, never inferring them). NULL is the default partition; values come back unescaped.
#[test]
fn reading_a_partitioned_table_returns_its_partition_values() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE g (a INT, b STRING) USING parquet PARTITIONED BY (b)",
    );
    // Empty: no files, no partitions.
    assert!(rows(&s, "SELECT CAST(a AS STRING), b FROM g").is_empty());
    run(&s, "INSERT INTO g VALUES (1, 'x'), (2, NULL), (3, 'a/b')");
    assert_eq!(
        rows(&s, "SELECT CAST(a AS STRING), b FROM g ORDER BY a"),
        ["1\tx", "2", "3\ta/b"]
    );
    assert_eq!(
        rows(&s, "SELECT CAST(a AS STRING) FROM g WHERE b = 'x'"),
        ["1"]
    );
    // Two levels, an integer one: typed as declared (`q + 1` is integer arithmetic), and filterable.
    run(
        &s,
        "CREATE TABLE h (v STRING, p STRING, q INT) USING parquet PARTITIONED BY (p, q)",
    );
    run(
        &s,
        "INSERT INTO h VALUES ('r1', 'a', 1), ('r2', 'a', 2), ('r3', 'b', 2)",
    );
    assert_eq!(
        rows(
            &s,
            "SELECT v, p, CAST(q + 1 AS STRING), typeof(q) FROM h ORDER BY v"
        ),
        ["r1\ta\t2\tint", "r2\ta\t3\tint", "r3\tb\t3\tint"]
    );
    assert_eq!(
        rows(&s, "SELECT v FROM h WHERE q = 2 ORDER BY v"),
        ["r2", "r3"]
    );
    assert_eq!(rows(&s, "SELECT CAST(count(*) AS STRING) FROM h"), ["3"]);
    // A timestamp partition value is local time in the session zone, which the scan does not have yet: refused,
    // never read in another zone.
    run(
        &s,
        "CREATE TABLE z (a INT, t TIMESTAMP) USING parquet PARTITIONED BY (t)",
    );
    run(
        &s,
        "INSERT INTO z VALUES (1, TIMESTAMP'2024-01-02 03:04:05')",
    );
    assert_eq!(class(&s, "SELECT a FROM z"), "KRISHIV_UNSUPPORTED_FEATURE");
}

/// Spark reads a catalog table's registered partitions (`CatalogFileIndex`), not every directory under its root:
/// what a listing of the root cannot read the same way is refused, never read differently. A directory the catalog
/// does not know is not read (the commit log lists the table's own files; without a log the listing skips it, see
/// `a_table_over_foreign_files_skips_directories_the_catalog_does_not_know`); a partition registered at another
/// location is refused (the listing would not find its files).
#[test]
fn reads_follow_the_registered_partitions() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE g (a INT, b STRING) USING parquet PARTITIONED BY (b)",
    );
    run(&s, "INSERT INTO g VALUES (1, 'x')");
    // A partition directory written behind the catalog's back.
    let root = dir.path().join("g");
    std::fs::create_dir_all(root.join("b=y")).unwrap();
    let x = std::fs::read_dir(root.join("b=x"))
        .unwrap()
        .filter_map(Result::ok)
        .find(|e| e.file_name().to_string_lossy().starts_with("part-"))
        .unwrap()
        .path();
    std::fs::copy(&x, root.join("b=y").join(x.file_name().unwrap())).unwrap();
    // The table's commit log lists its own files only, so the unregistered directory is not read: Spark's answer.
    assert_eq!(
        rows(&s, "SELECT CAST(a AS STRING), b FROM g ORDER BY b"),
        ["1\tx"]
    );
    // A partition registered outside the table's directory.
    let elsewhere = tempfile::tempdir().unwrap();
    run(
        &s,
        &format!(
            "ALTER TABLE g ADD PARTITION (b = 'z') LOCATION '{}'",
            elsewhere.path().display()
        ),
    );
    assert_eq!(class(&s, "SELECT a FROM g"), "KRISHIV_UNSUPPORTED_FEATURE");
    // A declared type whose directory value is not read yet.
    run(
        &s,
        "CREATE TABLE f (a INT, x DOUBLE) USING parquet PARTITIONED BY (x)",
    );
    run(&s, "INSERT INTO f VALUES (1, 1.5)");
    assert_eq!(class(&s, "SELECT a FROM f"), "KRISHIV_UNSUPPORTED_FEATURE");
}

/// A table over files Krishiv did not write (no commit log: the listing reads its directory) reads its registered
/// partitions' exact locations only, as Spark's `CatalogFileIndex` (Spark 4.2.0, krishiv-internal
/// `research/spark/registered-partitions/probe.out`, table `e`, and `partition-followups/probe.out`, table `e`): an
/// unregistered `p=1`, a `tmp_x/` directory, a `P=2/` next to the registered `p=2`, a file at the root, and `p=007`
/// when `p=7` is registered are all skipped, not refused.
#[test]
fn a_table_over_foreign_files_skips_directories_the_catalog_does_not_know() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    // Parquet files written by Krishiv elsewhere; only the data files are copied, without its commit log.
    run(
        &s,
        "CREATE TABLE src (x INT, p INT) USING parquet PARTITIONED BY (p)",
    );
    run(&s, "INSERT INTO src VALUES (1, 1), (2, 2), (7, 7)");
    let part = |sub: &str| {
        std::fs::read_dir(dir.path().join("src").join(sub))
            .unwrap()
            .filter_map(Result::ok)
            .find(|e| e.file_name().to_string_lossy().starts_with("part-"))
            .unwrap()
            .path()
    };
    let loc = tempfile::tempdir().unwrap();
    let copies = [
        ("p=1", "p=1"),
        ("p=2", "p=2"),
        ("p=2", "tmp_x"),
        ("p=2", "P=2"),
        ("p=7", "p=007"),
    ];
    for (from, to) in copies {
        let file = part(from);
        std::fs::create_dir_all(loc.path().join(to)).unwrap();
        std::fs::copy(&file, loc.path().join(to).join(file.file_name().unwrap())).unwrap();
    }
    std::fs::copy(part("p=1"), loc.path().join("root.parquet")).unwrap();
    run(
        &s,
        &format!(
            "CREATE TABLE e (x INT, p INT) USING parquet PARTITIONED BY (p) LOCATION '{}'",
            loc.path().display()
        ),
    );
    let read = "SELECT CAST(x AS STRING), CAST(p AS STRING) FROM e ORDER BY x";
    assert!(rows(&s, read).is_empty());
    run(&s, "ALTER TABLE e ADD PARTITION (p = 2)");
    assert_eq!(rows(&s, "SHOW PARTITIONS e"), ["p=2"]);
    assert_eq!(rows(&s, read), ["2\t2"]);
    // `p=7` is registered, but its directory is `p=007`: not its location.
    run(&s, "ALTER TABLE e ADD PARTITION (p = 7)");
    assert_eq!(rows(&s, read), ["2\t2"]);
}

/// A `basePath` above a partitioned table's location puts directories that are not partition columns of the table
/// above its files (`q=1` over `p=2`): the read is refused, as before registered partitions, never an empty table
/// (#690 review).
#[test]
fn a_base_path_above_the_location_of_a_partitioned_table_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE src (x INT, p INT) USING parquet PARTITIONED BY (p)",
    );
    run(&s, "INSERT INTO src VALUES (2, 2)");
    let file = std::fs::read_dir(dir.path().join("src").join("p=2"))
        .unwrap()
        .filter_map(Result::ok)
        .find(|e| e.file_name().to_string_lossy().starts_with("part-"))
        .unwrap()
        .path();
    let data = tempfile::tempdir().unwrap();
    let at = data.path().join("q=1").join("p=2");
    std::fs::create_dir_all(&at).unwrap();
    std::fs::copy(&file, at.join(file.file_name().unwrap())).unwrap();
    run(
        &s,
        &format!(
            "CREATE TABLE e (x INT, p INT) USING parquet OPTIONS (basePath '{}') PARTITIONED BY (p) LOCATION '{}'",
            data.path().display(),
            data.path().join("q=1").display()
        ),
    );
    run(&s, "ALTER TABLE e ADD PARTITION (p = 2)");
    let e = err(&s, "SELECT x FROM e");
    assert!(e.to_string().contains("basePath"), "{e}");
}

/// An unpartitioned table over foreign files reads its root's files only: a subdirectory is not read (Spark 4.2.0,
/// krishiv-internal `research/spark/registered-partitions/probe.out`, table `u`: one row, `tmp_x/b.parquet` unread).
#[test]
fn an_unpartitioned_table_over_foreign_files_skips_a_subdirectory() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE TABLE src (x INT) USING parquet");
    run(&s, "INSERT INTO src VALUES (1)");
    let file = std::fs::read_dir(dir.path().join("src"))
        .unwrap()
        .filter_map(Result::ok)
        .find(|e| e.file_name().to_string_lossy().starts_with("part-"))
        .unwrap()
        .path();
    let loc = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(loc.path().join("tmp_x")).unwrap();
    std::fs::copy(&file, loc.path().join(file.file_name().unwrap())).unwrap();
    std::fs::copy(&file, loc.path().join("tmp_x").join("b.parquet")).unwrap();
    // A `name=value` subdirectory is not a partition of an unpartitioned table either (#690 review).
    std::fs::create_dir_all(loc.path().join("p=1")).unwrap();
    std::fs::copy(&file, loc.path().join("p=1").join("c.parquet")).unwrap();
    run(
        &s,
        &format!(
            "CREATE TABLE u (x INT) USING parquet LOCATION '{}'",
            loc.path().display()
        ),
    );
    assert_eq!(rows(&s, "SELECT CAST(x AS STRING) FROM u"), ["1"]);
}

/// A table partitioned by a BINARY column, as Spark 4.2.0 writes and reads it (krishiv-internal
/// `research/spark/binary-partition/probe1.out`, `probe2.out`): the directory is the value's bytes as text,
/// escaped like any value; NULL and the empty value are the default partition; the column stays `binary`.
#[test]
fn a_binary_partition_column_is_its_bytes_as_text() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE g (a INT, b BINARY) USING parquet PARTITIONED BY (b)",
    );
    run(
        &s,
        "INSERT INTO g VALUES (1, X'4142'), (2, NULL), (3, X''), (4, X'C3A9'), (5, X'2F3D25')",
    );
    for d in [
        "g/b=AB",
        "g/b=__HIVE_DEFAULT_PARTITION__",
        "g/b=é",
        "g/b=%2F%3D%25",
    ] {
        assert!(dir.path().join(d).is_dir(), "{d}");
    }
    assert_eq!(
        rows(&s, "SHOW PARTITIONS g"),
        ["b=%2F%3D%25", "b=AB", "b=__HIVE_DEFAULT_PARTITION__", "b=é"]
    );
    // A static value is the literal's text; DESC finds the partition by it and shows the column as binary.
    run(&s, "INSERT INTO g PARTITION (b = 'lit') VALUES (6)");
    assert!(dir.path().join("g/b=lit").is_dir());
    let desc = rows(&s, "DESC EXTENDED g PARTITION (b = 'AB')");
    assert!(desc.iter().any(|r| r.starts_with("b	binary")), "{desc:?}");
    assert!(
        desc.contains(&"Partition Values\t[b=AB]".to_owned()),
        "{desc:?}"
    );
    // Reading rows of a partitioned catalog table is not supported yet, whatever the column type.
}

/// A timestamp partition value renders in a region session zone, not only a fixed offset (Spark 4.2.0,
/// `research/spark/binary-partition/probe3.out`): the epoch in `America/Los_Angeles` is `1969-12-31 16:00:00`.
#[test]
fn a_timestamp_partition_renders_in_a_region_session_zone() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    s.conf()
        .set("spark.sql.session.timeZone", "America/Los_Angeles")
        .unwrap();
    run(
        &s,
        "CREATE TABLE t (a INT, c TIMESTAMP) USING parquet PARTITIONED BY (c)",
    );
    run(
        &s,
        "INSERT INTO t VALUES (1, CAST(0 AS TIMESTAMP)), (2, TIMESTAMP'2018-07-01 12:00:00')",
    );
    assert_eq!(
        rows(&s, "SHOW PARTITIONS t"),
        ["c=1969-12-31 16%3A00%3A00", "c=2018-07-01 12%3A00%3A00"]
    );
}

fn dropped(dir: &std::path::Path, relative: &str) {
    std::fs::remove_dir_all(dir.join(relative)).unwrap();
}

/// The rows below are a real Spark 4.2.0 run's (`RepairTableCommand`).
#[test]
fn msck_repair_drop_and_sync_forget_partitions_whose_directory_is_gone() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table pt(a int, p int, q string) using parquet partitioned by (p,q)",
    );
    run(&s, "insert into pt values (1,1,'x'),(2,2,'y'),(3,3,'z')");
    let root = dir.path().join("pt");
    dropped(&root, "p=2");
    run(&s, "msck repair table pt");
    assert_eq!(rows(&s, "show partitions pt").len(), 3);
    run(&s, "msck repair table pt drop partitions");
    assert_eq!(rows(&s, "show partitions pt"), ["p=1/q=x", "p=3/q=z"]);
    assert_eq!(
        rows(&s, "select cast(a as string) from pt order by a"),
        ["1", "3"]
    );
    // SYNC does both.
    dropped(&root, "p=3");
    std::fs::create_dir_all(root.join("p=9/q=w")).unwrap();
    run(&s, "repair table pt sync partitions");
    assert_eq!(rows(&s, "show partitions pt"), ["p=1/q=x", "p=9/q=w"]);
    // ADD never forgets one.
    dropped(&root, "p=9");
    run(&s, "msck repair table pt add partitions");
    assert_eq!(rows(&s, "show partitions pt"), ["p=1/q=x", "p=9/q=w"]);
    // Column statistics survive DROP and SYNC: a Spark 4.2.0 run keeps min, max and distinct_count as they were
    // after the partition holding the minimum is forgotten.
    run(
        &s,
        "create table ps(a int, p int) using parquet partitioned by (p)",
    );
    run(&s, "insert into ps values (10,1),(20,2),(30,3)");
    run(&s, "analyze table ps compute statistics for columns a");
    let ps = dir.path().join("ps");
    dropped(&ps, "p=1");
    run(&s, "msck repair table ps drop partitions");
    dropped(&ps, "p=2");
    run(&s, "msck repair table ps sync partitions");
    assert_eq!(rows(&s, "select cast(a as string) from ps"), ["30"]);
    let described = rows(&s, "describe extended ps a");
    for kept in ["min\t10", "max\t30", "distinct_count\t3"] {
        assert!(described.iter().any(|r| r == kept), "{described:?}");
    }
    run(&s, "create table pu(a int) using parquet");
    assert_eq!(
        class(&s, "msck repair table pu drop partitions"),
        "NOT_A_PARTITIONED_TABLE"
    );
}

/// The rows below are a real Spark 4.2.0 run's (`AlterTableRenamePartitionCommand`).
#[test]
fn rename_partition_moves_a_managed_partition() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table r1(a int, p int, q string) using parquet partitioned by (p,q)",
    );
    run(&s, "insert into r1 values (1,1,'x'),(2,2,'y')");
    run(
        &s,
        "alter table r1 partition (p=1,q='x') rename to partition (p=5,q='new val')",
    );
    assert_eq!(rows(&s, "show partitions r1"), ["p=2/q=y", "p=5/q=new val"]);
    assert_eq!(
        rows(&s, "select concat_ws('|', a, p, q) from r1 order by a"),
        ["1|5|new val", "2|2|y"]
    );
    assert_eq!(
        class(
            &s,
            "alter table r1 partition (p=1,q='x') rename to partition (p=7,q='a')"
        ),
        "PARTITIONS_NOT_FOUND"
    );
    assert_eq!(
        class(
            &s,
            "alter table r1 partition (p=2,q='y') rename to partition (p=5,q='new val')"
        ),
        "PARTITIONS_ALREADY_EXIST"
    );
    assert_eq!(
        class(
            &s,
            "alter table r1 partition (p=2) rename to partition (p=3,q='y')"
        ),
        "_LEGACY_ERROR_TEMP_1232"
    );
    assert_eq!(
        class(
            &s,
            "alter table r1 partition (p=2,q='y') rename to partition (p=3)"
        ),
        "_LEGACY_ERROR_TEMP_1232"
    );
    run(
        &s,
        "alter table r1 partition (P=2,Q='y') rename to partition (P=3,Q='Y')",
    );
    assert_eq!(rows(&s, "show partitions r1"), ["p=3/q=Y", "p=5/q=new val"]);
    assert_eq!(
        class(
            &s,
            "alter table r1 partition (p=3,q='Y') rename to partition (p=3,q='Y')"
        ),
        "PARTITIONS_ALREADY_EXIST"
    );
    assert_eq!(
        class(
            &s,
            "alter table r1 partition (p=3,q='Y') rename to partition (p='abc',q='Y')"
        ),
        "CAST_INVALID_INPUT"
    );
}

/// The `.json` entries of a table root's commit log.
fn log_entries(root: &std::path::Path) -> usize {
    std::fs::read_dir(root.join("_krishiv_log")).map_or(0, |d| {
        d.filter_map(Result::ok)
            .filter(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                name.ends_with(".json") && !name.ends_with(".checkpoint.json")
            })
            .count()
    })
}

/// A rename whose log commit fails moves nothing: the log is written before the directory moves, so the table
/// still reads its rows under the old spec.
#[cfg(unix)]
#[test]
fn a_rename_whose_log_commit_fails_leaves_the_table_readable() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table r2(a int, p int) using parquet partitioned by (p)",
    );
    run(&s, "insert into r2 values (1,1),(2,2)");
    let log = dir.path().join("r2").join("_krishiv_log");
    assert!(log.is_dir(), "the table root has a commit log");
    std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o555)).unwrap();
    // A user who can write a read-only directory (root) cannot make the commit fail this way.
    if std::fs::write(log.join("probe"), b"").is_ok() {
        std::fs::remove_file(log.join("probe")).unwrap();
        std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o755)).unwrap();
        return;
    }
    let failed = block_on(async {
        s.sql("alter table r2 partition (p=1) rename to partition (p=5)")
            .await?
            .collect()
            .await
    });
    std::fs::set_permissions(&log, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(failed.is_err(), "the commit cannot be written");
    assert!(dir.path().join("r2/p=1").is_dir());
    assert!(!dir.path().join("r2/p=5").exists());
    assert_eq!(rows(&s, "show partitions r2"), ["p=1", "p=2"]);
    assert_eq!(
        rows(&s, "select concat_ws('|', a, p) from r2 order by a"),
        ["1|1", "2|2"]
    );
}

/// A rename whose directory move fails after the log moved puts the log back.
#[test]
fn a_rename_whose_move_fails_puts_the_log_back() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table r3(a int, p int) using parquet partitioned by (p)",
    );
    run(&s, "insert into r3 values (1,1),(2,2)");
    let root = dir.path().join("r3");
    // A directory the catalog does not know, not empty: the move onto it fails.
    std::fs::create_dir_all(root.join("p=5")).unwrap();
    std::fs::write(root.join("p=5").join("_keep"), b"").unwrap();
    let before = log_entries(&root);
    let failed = block_on(async {
        s.sql("alter table r3 partition (p=1) rename to partition (p=5)")
            .await?
            .collect()
            .await
    });
    assert!(
        failed.is_err(),
        "the move cannot replace a non-empty directory"
    );
    assert_eq!(log_entries(&root), before + 2, "the move and its inverse");
    assert_eq!(rows(&s, "show partitions r3"), ["p=1", "p=2"]);
    assert_eq!(
        rows(&s, "select concat_ws('|', a, p) from r3 order by a"),
        ["1|1", "2|2"]
    );
}

/// The rows below are a real Spark 4.2.0 run's: `castPartitionSpec` casts with ANSI on under the default
/// `spark.sql.storeAssignmentPolicy`, whatever `spark.sql.ansi.enabled` says, and the value is re-rendered.
#[test]
fn rename_partition_values_cast_as_spark_casts_them() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table r(a int, p int, q string) using parquet partitioned by (p,q)",
    );
    run(&s, "insert into r values (1,1,'x'),(2,2,'y'),(3,3,'z')");
    s.conf().set("spark.sql.ansi.enabled", "false").unwrap();
    assert_eq!(
        class(
            &s,
            "alter table r partition (p=1,q='x') rename to partition (p='abc',q='x')"
        ),
        "CAST_INVALID_INPUT"
    );
    run(
        &s,
        "alter table r partition (p=1,q='x') rename to partition (p='007',q='x')",
    );
    assert_eq!(
        rows(&s, "show partitions r"),
        ["p=2/q=y", "p=3/q=z", "p=7/q=x"]
    );
    s.conf().set("spark.sql.ansi.enabled", "true").unwrap();
    s.conf()
        .set("spark.sql.storeAssignmentPolicy", "LEGACY")
        .unwrap();
    run(
        &s,
        "alter table r partition (p=2,q='y') rename to partition (p='abc',q='y')",
    );
    assert_eq!(
        rows(&s, "show partitions r"),
        ["p=3/q=z", "p=7/q=x", "p=__HIVE_DEFAULT_PARTITION__/q=y"]
    );
    run(
        &s,
        "alter table r partition (p='0003',q='z') rename to partition (p=4,q='z')",
    );
    assert_eq!(
        rows(&s, "show partitions r"),
        ["p=4/q=z", "p=7/q=x", "p=__HIVE_DEFAULT_PARTITION__/q=y"]
    );
    s.conf()
        .set("spark.sql.storeAssignmentPolicy", "ANSI")
        .unwrap();
    run(
        &s,
        "create table d(a int, p date) using parquet partitioned by (p)",
    );
    run(&s, "insert into d values (1, date'2024-01-02')");
    run(
        &s,
        "alter table d partition (p='2024-01-02') rename to partition (p='2024-1-3')",
    );
    assert_eq!(rows(&s, "show partitions d"), ["p=2024-01-03"]);
}

/// `MSCK REPAIR TABLE .. DROP PARTITIONS` takes every vanished partition out of the log in one entry.
#[test]
fn msck_repair_drop_forgets_vanished_partitions_in_one_log_entry() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table pv(a int, p int) using parquet partitioned by (p)",
    );
    run(&s, "insert into pv values (1,1),(2,2),(3,3)");
    let root = dir.path().join("pv");
    let before = log_entries(&root);
    dropped(&root, "p=1");
    dropped(&root, "p=2");
    run(&s, "msck repair table pv drop partitions");
    assert_eq!(log_entries(&root), before + 1);
    assert_eq!(rows(&s, "show partitions pv"), ["p=3"]);
    assert_eq!(rows(&s, "select cast(a as string) from pv"), ["3"]);
}

/// `ALTER TABLE .. DROP PARTITION` (Spark 4.2.0, krishiv-internal `research/spark/partition-followups/probe.{py,out}`):
/// values are cast to the column types, every spec must match a partition exactly (a partial one is not found),
/// all specs are checked before any is dropped, `IF EXISTS` skips the missing ones, a managed table's
/// directories go, and the column statistics are dropped (`updateTableStats`).
#[test]
fn drop_partition_drops_exact_partitions_of_a_managed_table() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table c(x int, p int, s string) using parquet partitioned by (p, s)",
    );
    run(
        &s,
        "insert into c values (1,1,'a'),(2,1,'b'),(3,2,'a'),(4,3,'c')",
    );
    run(&s, "analyze table c compute statistics for columns x");
    let root = dir.path().join("c");
    run(&s, "alter table c drop partition (p='01', s='a')");
    assert_eq!(
        rows(&s, "show partitions c"),
        ["p=1/s=b", "p=2/s=a", "p=3/s=c"]
    );
    assert!(!root.join("p=1/s=a").exists());
    assert!(root.join("p=1/s=b").is_dir());
    assert_eq!(
        rows(&s, "select concat_ws('|', x, p, s) from c order by x"),
        ["2|1|b", "3|2|a", "4|3|c"]
    );
    assert!(
        rows(&s, "describe extended c x")
            .iter()
            .any(|r| r == "min\tNULL"),
        "{:?}",
        rows(&s, "describe extended c x")
    );
    assert_eq!(
        params(&s, "alter table c drop partition (p=9, s='z')"),
        [
            "partitionList=PARTITION (`p` = 9, `s` = z)",
            "tableName=`default`.`c`"
        ]
    );
    run(&s, "alter table c drop if exists partition (p=9, s='z')");
    assert_eq!(
        params(&s, "alter table c drop partition (p=1)"),
        [
            "partitionList=PARTITION (`p` = 1)",
            "tableName=`default`.`c`"
        ]
    );
    // One missing spec drops nothing; with IF EXISTS the others go.
    assert_eq!(
        class(
            &s,
            "alter table c drop partition (p=2, s='a'), partition (p=9, s='z')"
        ),
        "PARTITIONS_NOT_FOUND"
    );
    assert_eq!(
        rows(&s, "show partitions c"),
        ["p=1/s=b", "p=2/s=a", "p=3/s=c"]
    );
    run(
        &s,
        "alter table c drop if exists partition (p=2, s='a'), partition (p=9, s='z')",
    );
    assert_eq!(rows(&s, "show partitions c"), ["p=1/s=b", "p=3/s=c"]);
    assert_eq!(
        params(&s, "alter table c drop partition (q=1)"),
        [
            "partitionList=`q`",
            "tableName=`spark_catalog`.`default`.`c`"
        ]
    );
    assert_eq!(
        class(&s, "alter table c drop partition (p='abc')"),
        "CAST_INVALID_INPUT"
    );
    run(&s, "alter table c drop partition (p=3, s='c') purge");
    assert_eq!(rows(&s, "show partitions c"), ["p=1/s=b"]);
    assert!(!root.join("p=3/s=c").exists());
    assert_eq!(rows(&s, "select cast(x as string) from c"), ["2"]);
    run(&s, "create table n(x int) using parquet");
    assert_eq!(
        class(&s, "alter table n drop partition (p=1)"),
        "_LEGACY_ERROR_TEMP_1251"
    );
}

/// More of Spark 4.2.0's `DROP PARTITION` (krishiv-internal `research/spark/partition-followups/probe2.{py,out}`):
/// a bare key is a parse error for every partition `ALTER`, every missing spec is listed, every spec is cast
/// before any is looked up, a repeated spec drops once, and `p=NULL` is the default partition.
#[test]
fn drop_partition_checks_every_spec_as_spark() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table c(x int, p int, s string) using parquet partitioned by (p, s)",
    );
    run(&s, "insert into c values (1,1,'a'),(2,2,'b'),(3,NULL,'c')");
    for (sql, key) in [
        ("alter table c drop partition (p, s)", "`p`"),
        ("alter table c drop partition (p=1, s)", "`s`"),
        ("alter table c add partition (p, s)", "`p`"),
        (
            "alter table c partition (p=1, s) rename to partition (p=5, s='x')",
            "`s`",
        ),
        (
            "alter table c partition (p=1, s) set location '/tmp/x'",
            "`s`",
        ),
    ] {
        assert_eq!(
            class(&s, sql),
            "INVALID_SQL_SYNTAX.EMPTY_PARTITION_VALUE",
            "{sql}"
        );
        assert_eq!(params(&s, sql), [format!("partKey={key}")], "{sql}");
    }
    // A repeated key comes first (probe3).
    for (sql, key) in [
        ("alter table c drop partition (p, p)", "`p`"),
        ("alter table c drop partition (p=1, p)", "`p`"),
        ("alter table c add partition (s, p, s)", "`s`"),
    ] {
        assert_eq!(class(&s, sql), "DUPLICATE_KEY", "{sql}");
        assert_eq!(params(&s, sql), [format!("keyColumn={key}")], "{sql}");
    }
    assert_eq!(
        params(
            &s,
            "alter table c drop partition (p=8, s='z'), partition (p=9, s='y')"
        ),
        [
            "partitionList=PARTITION (`p` = 8, `s` = z), PARTITION (`p` = 9, `s` = y)",
            "tableName=`default`.`c`"
        ]
    );
    assert_eq!(
        class(
            &s,
            "alter table c drop partition (p=9, s='z'), partition (p='abc', s='a')"
        ),
        "CAST_INVALID_INPUT"
    );
    assert_eq!(
        params(
            &s,
            "alter table c drop partition (p=9, s='z'), partition (q=1)"
        ),
        [
            "partitionList=`q`",
            "tableName=`spark_catalog`.`default`.`c`"
        ]
    );
    run(
        &s,
        "alter table c drop partition (p=1, s='a'), partition (p=1, s='a')",
    );
    run(&s, "alter table c drop partition (p=NULL, s='c')");
    assert_eq!(rows(&s, "show partitions c"), ["p=2/s=b"]);
    // The emptied parent `p=1` stays, as Spark leaves it.
    let root = dir.path().join("c");
    assert!(root.join("p=1").is_dir());
    assert!(!root.join("p=1/s=a").exists());
    assert_eq!(rows(&s, "select cast(x as string) from c"), ["2"]);
}

/// `spark.sql.legacy.skipTypeValidationOnAlterPartition` keeps `ALTER TABLE` partition values as written (Spark
/// 4.2.0); `INSERT`'s static values follow `storeAssignmentPolicy` alone, whatever `spark.sql.ansi.enabled` says.
#[test]
fn partition_value_casts_follow_the_legacy_settings_as_spark() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table c(x int, p int, s string) using parquet partitioned by (p, s)",
    );
    run(
        &s,
        "set spark.sql.legacy.skipTypeValidationOnAlterPartition=true",
    );
    run(&s, "alter table c add partition (p='abc', s='k')");
    run(&s, "alter table c add partition (p='007', s='k')");
    assert_eq!(rows(&s, "show partitions c"), ["p=007/s=k", "p=abc/s=k"]);
    run(&s, "alter table c drop partition (p='abc', s='k')");
    run(&s, "alter table c drop partition (p='007', s='k')");
    assert_eq!(rows(&s, "show partitions c"), Vec::<String>::new());
    run(
        &s,
        "set spark.sql.legacy.skipTypeValidationOnAlterPartition=false",
    );
    run(
        &s,
        "create table b(x int, p int) using parquet partitioned by (p)",
    );
    run(&s, "set spark.sql.storeAssignmentPolicy=LEGACY");
    run(&s, "insert into b partition (p='abc') select 1");
    run(&s, "insert into b partition (p='008') select 2");
    assert_eq!(
        rows(&s, "show partitions b"),
        ["p=8", "p=__HIVE_DEFAULT_PARTITION__"]
    );
    assert_eq!(
        rows(
            &s,
            "select concat_ws('|', x, coalesce(cast(p as string), 'null')) from b order by x"
        ),
        ["1|null", "2|8"]
    );
    run(&s, "set spark.sql.storeAssignmentPolicy=ANSI");
    run(&s, "set spark.sql.ansi.enabled=false");
    assert_eq!(
        class(&s, "insert into b partition (p='xyz') select 3"),
        "CAST_INVALID_INPUT"
    );
    assert_eq!(
        class(&s, "alter table b add partition (p='xyz')"),
        "CAST_INVALID_INPUT"
    );
    run(&s, "set spark.sql.ansi.enabled=true");
    run(&s, "insert into b partition (p=NULL) select 4");
    assert_eq!(
        rows(&s, "show partitions b"),
        ["p=8", "p=__HIVE_DEFAULT_PARTITION__"]
    );
}

/// A partition registered away from its default directory is never deleted: its directory may be anything (the
/// table root itself, another table's data), and reads of the table never list it.
#[test]
fn drop_partition_refuses_a_partition_outside_its_default_directory() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table o(x int, p int) using parquet partitioned by (p)",
    );
    run(&s, "insert into o values (1,1)");
    let root = dir.path().join("o");
    let elsewhere = tempfile::tempdir().unwrap();
    std::fs::write(elsewhere.path().join("keep.txt"), "x").unwrap();
    run(
        &s,
        &format!(
            "alter table o add partition (p=2) location '{}'",
            elsewhere.path().display()
        ),
    );
    run(
        &s,
        &format!(
            "alter table o add partition (p=3) location '{}'",
            root.display()
        ),
    );
    for p in [2, 3] {
        assert_eq!(
            class(&s, &format!("alter table o drop partition (p={p})")),
            "KRISHIV_UNSUPPORTED_FEATURE"
        );
    }
    assert!(elsewhere.path().join("keep.txt").is_file());
    assert!(root.join("p=1").is_dir());
    assert_eq!(rows(&s, "show partitions o"), ["p=1", "p=2", "p=3"]);
}

/// A partition whose default directory resolves outside the table through a symbolic link in a parent directory
/// (`t/p=1` linked elsewhere, partition `p=1/q=1`), or to another partition's directory, is not dropped: deleting it
/// would delete files that are not the partition's.
/// Refused before the log, the catalog or any file changes.
#[cfg(unix)]
#[test]
fn drop_partition_refuses_a_directory_resolved_outside_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table l(x int, p int, q int) using parquet partitioned by (p, q)",
    );
    run(&s, "insert into l values (1,2,2)");
    let root = dir.path().join("l");
    let outside = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(outside.path().join("q=1")).unwrap();
    std::fs::write(outside.path().join("q=1").join("keep.txt"), "x").unwrap();
    std::os::unix::fs::symlink(outside.path(), root.join("p=1")).unwrap();
    run(&s, "alter table l add partition (p=1, q=1)");
    assert_eq!(
        class(&s, "alter table l drop partition (p=1, q=1)"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert!(outside.path().join("q=1").join("keep.txt").is_file());
    // A link to another partition of the table (`p=4` -> `p=2`): dropping `p=4/q=2` would delete `p=2/q=2`'s files.
    std::os::unix::fs::symlink(root.join("p=2"), root.join("p=4")).unwrap();
    run(&s, "alter table l add partition (p=4, q=2)");
    assert_eq!(
        class(&s, "alter table l drop partition (p=4, q=2)"),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(
        rows(&s, "select cast(x as string) from l where p = 2"),
        ["1"]
    );
    assert_eq!(
        rows(&s, "show partitions l"),
        ["p=1/q=1", "p=2/q=2", "p=4/q=2"]
    );
    // A partition inside the table still drops.
    run(&s, "alter table l drop partition (p=2, q=2)");
    assert_eq!(rows(&s, "show partitions l"), ["p=1/q=1", "p=4/q=2"]);
}

/// RENAME PARTITION neither moves a source directory nor writes into a target that resolves away from its own path
/// in the table through a symbolic link in a parent directory: it would move another directory's files into the
/// table, or the partition's files out of it. Refused before the log or any file changes.
#[cfg(unix)]
#[test]
fn rename_partition_refuses_directories_resolved_away_from_their_path() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table rl(x int, p int, q int) using parquet partitioned by (p, q)",
    );
    run(&s, "insert into rl values (1,2,2)");
    let root = dir.path().join("rl");
    let outside = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(outside.path().join("q=1")).unwrap();
    std::fs::write(outside.path().join("q=1").join("keep.txt"), "x").unwrap();
    std::os::unix::fs::symlink(outside.path(), root.join("p=1")).unwrap();
    run(&s, "alter table rl add partition (p=1, q=1)");
    // The source resolves outside the table.
    assert_eq!(
        class(
            &s,
            "alter table rl partition (p=1, q=1) rename to partition (p=3, q=1)"
        ),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert!(outside.path().join("q=1").join("keep.txt").is_file());
    // The target's parent resolves outside the table.
    let elsewhere = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(elsewhere.path(), root.join("p=5")).unwrap();
    assert_eq!(
        class(
            &s,
            "alter table rl partition (p=2, q=2) rename to partition (p=5, q=2)"
        ),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert!(root.join("p=2").join("q=2").is_dir());
    assert!(!elsewhere.path().join("q=2").exists());
    assert_eq!(rows(&s, "show partitions rl"), ["p=1/q=1", "p=2/q=2"]);
    // A rename inside the table still works.
    run(
        &s,
        "alter table rl partition (p=2, q=2) rename to partition (p=6, q=2)",
    );
    assert_eq!(rows(&s, "show partitions rl"), ["p=1/q=1", "p=6/q=2"]);
}

/// A partition registered away from its default directory is not renamed: Spark 4.2.0 moves that directory, which
/// may be another table's (krishiv-internal `research/spark/partition-followups/probe9.py`); the engine refuses
/// before the log or any file changes, as DROP PARTITION does.
#[test]
fn rename_partition_refuses_a_partition_outside_its_default_directory() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "create table other(x int) using parquet");
    run(&s, "insert into other values (5)");
    let other = dir.path().join("other");
    run(
        &s,
        "create table ro(x int, p int) using parquet partitioned by (p)",
    );
    run(&s, "insert into ro values (1,1)");
    run(
        &s,
        &format!(
            "alter table ro add partition (p=2) location '{}'",
            other.display()
        ),
    );
    assert_eq!(
        class(
            &s,
            "alter table ro partition (p=2) rename to partition (p=3)"
        ),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(rows(&s, "show partitions ro"), ["p=1", "p=2"]);
    assert!(!dir.path().join("ro").join("p=3").exists());
    assert_eq!(rows(&s, "select cast(x as string) from other"), ["5"]);
    // A partition at its default directory still renames.
    run(
        &s,
        "alter table ro partition (p=1) rename to partition (p=4)",
    );
    assert_eq!(rows(&s, "show partitions ro"), ["p=2", "p=4"]);
}

/// Spark drops an external table's partition from the catalog and keeps its files; the table stops reading them (SQL-19).
#[test]
fn drop_partition_of_an_external_table_forgets_it_and_keeps_its_files() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    let ext = tempfile::tempdir().unwrap();
    run(
        &s,
        &format!(
            "create table e(x int, p int) using parquet partitioned by (p) location '{}'",
            ext.path().display()
        ),
    );
    run(&s, "insert into e values (1,1),(2,2)");
    run(&s, "alter table e drop partition (p=1)");
    assert_eq!(rows(&s, "show partitions e"), ["p=2"]);
    assert_eq!(
        rows(&s, "select cast(x as string) from e order by x"),
        ["2"]
    );
    assert!(
        std::fs::read_dir(ext.path().join("p=1"))
            .unwrap()
            .any(|f| f.is_ok_and(|f| f.path().extension().is_some_and(|e| e == "parquet"))),
        "the data files stay"
    );
    // A missing spec is still reported, and IF EXISTS over nothing changes nothing.
    assert_eq!(
        class(&s, "alter table e drop partition (p=9)"),
        "PARTITIONS_NOT_FOUND"
    );
    run(&s, "alter table e drop if exists partition (p=9)");
    // MSCK REPAIR TABLE registers the kept directory again
    run(&s, "msck repair table e");
    assert_eq!(rows(&s, "show partitions e"), ["p=1", "p=2"]);
}

/// The commit log forgets a dropped partition's files in one entry, so a read of the table does not list them.
#[test]
fn drop_partition_removes_the_files_from_the_log_in_one_entry() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table dl(a int, p int) using parquet partitioned by (p)",
    );
    run(&s, "insert into dl values (1,1),(2,2),(3,3)");
    let root = dir.path().join("dl");
    let before = log_entries(&root);
    run(&s, "alter table dl drop partition (p=1), partition (p=2)");
    assert_eq!(log_entries(&root), before + 1);
    assert_eq!(rows(&s, "select cast(a as string) from dl"), ["3"]);
    // A cached read sees the drop.
    run(&s, "cache table dl");
    run(&s, "alter table dl drop partition (p=3)");
    assert_eq!(
        rows(&s, "select cast(a as string) from dl"),
        Vec::<String>::new()
    );
}

/// Without a commit log a dropped partition's directory goes whole; one that cannot be removed is Spark's
/// `_LEGACY_ERROR_TEMP_2036` ("Unable to delete partition path"), naming the path.
#[cfg(unix)]
#[test]
fn drop_partition_whose_directory_cannot_be_removed_is_legacy_error_2036() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table du(a int, p int) using parquet partitioned by (p)",
    );
    run(&s, "insert into du values (1,1),(2,2)");
    let root = dir.path().join("du");
    std::fs::remove_dir_all(root.join("_krishiv_log")).unwrap();
    let part = root.join("p=1");
    std::fs::set_permissions(&part, std::fs::Permissions::from_mode(0o555)).unwrap();
    // A user who can write a read-only directory (root) cannot make the removal fail this way.
    if std::fs::write(part.join("probe"), b"").is_ok() {
        std::fs::remove_file(part.join("probe")).unwrap();
        std::fs::set_permissions(&part, std::fs::Permissions::from_mode(0o755)).unwrap();
        return;
    }
    let failed = block_on(async {
        s.sql("alter table du drop partition (p=1)")
            .await?
            .collect()
            .await
    });
    // Writable again before any assertion, so the temporary directory can be removed whatever happened.
    std::fs::set_permissions(&part, std::fs::Permissions::from_mode(0o755)).unwrap();
    let Err(e) = failed else {
        panic!("the partition directory cannot be removed");
    };
    assert_eq!(e.error_class().name(), "_LEGACY_ERROR_TEMP_2036", "{e}");
    assert!(e.to_string().contains("p=1"), "{e}");
    assert!(part.is_dir());
}

/// `ADD PARTITION` and the static values of `INSERT .. PARTITION` are cast to the partition column types
/// (`castPartitionSpec`, Spark 4.2.0): `'007'` is `7`, `'2024-1-5'` is `2024-01-05`, `'abc'` and `3.7` for an INT
/// are `CAST_INVALID_INPUT`; under `storeAssignmentPolicy=LEGACY` a value the cast makes NULL is the default
/// partition.
#[test]
fn added_and_inserted_partition_values_are_cast_to_their_types() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table a(x int, p int, d date) using parquet partitioned by (p, d)",
    );
    run(&s, "alter table a add partition (p='007', d='2024-01-05')");
    assert_eq!(
        class(&s, "alter table a add partition (p='abc', d='2024-01-05')"),
        "CAST_INVALID_INPUT"
    );
    run(&s, "alter table a add partition (p=8, d='2024-1-5')");
    assert_eq!(
        class(&s, "alter table a add partition (p='7', d='2024-01-05')"),
        "PARTITIONS_ALREADY_EXIST"
    );
    run(
        &s,
        "alter table a add if not exists partition (p='7', d='2024-01-05')",
    );
    assert_eq!(
        rows(&s, "show partitions a"),
        ["p=7/d=2024-01-05", "p=8/d=2024-01-05"]
    );
    assert!(dir.path().join("a/p=7/d=2024-01-05").is_dir());
    run(&s, "set spark.sql.storeAssignmentPolicy=LEGACY");
    run(&s, "alter table a add partition (p='abc', d='2024-01-06')");
    assert_eq!(
        rows(&s, "show partitions a"),
        [
            "p=7/d=2024-01-05",
            "p=8/d=2024-01-05",
            "p=__HIVE_DEFAULT_PARTITION__/d=2024-01-06"
        ]
    );
    run(&s, "set spark.sql.storeAssignmentPolicy=ANSI");
    run(
        &s,
        "create table b(x int, p int) using parquet partitioned by (p)",
    );
    run(&s, "insert into b partition (p='007') select 1");
    for v in ["'abc'", "3.7"] {
        assert_eq!(
            class(&s, &format!("insert into b partition (p={v}) select 2")),
            "CAST_INVALID_INPUT"
        );
    }
    assert_eq!(rows(&s, "show partitions b"), ["p=7"]);
    assert_eq!(rows(&s, "select concat_ws('|', x, p) from b"), ["1|7"]);
    assert!(dir.path().join("b/p=7").is_dir());
}

/// `ALTER TABLE .. PARTITION (..) SET LOCATION` casts its spec as ADD and RENAME do: the rows of a real Spark run
/// (3.5.9, 4.0.5, 4.1.4 and 4.2.0 agree; krishiv-internal `research/spark/partition-followups/probe4.py`).
#[test]
fn set_location_casts_its_partition_spec() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table c (x int, p int, s string) using parquet partitioned by (p, s)",
    );
    run(&s, "alter table c add partition (p=7, s='a')");
    let new1 = dir.path().join("new1");
    run(
        &s,
        &format!(
            "alter table c partition (p='007', s='a') set location '{}'",
            new1.display()
        ),
    );
    assert_eq!(rows(&s, "show partitions c"), ["p=7/s=a"]);
    let location = rows(&s, "describe formatted c partition (p=7, s='a')")
        .into_iter()
        .find(|r| r.starts_with("Location"))
        .unwrap();
    assert!(location.ends_with("new1"), "{location}");
    let other = dir.path().join("new2");
    assert_eq!(
        class(
            &s,
            &format!(
                "alter table c partition (p='abc', s='a') set location '{}'",
                other.display()
            )
        ),
        "CAST_INVALID_INPUT"
    );
    assert_eq!(
        params(
            &s,
            &format!(
                "alter table c partition (p='8', s='a') set location '{}'",
                other.display()
            )
        ),
        [
            "partitionList=PARTITION (`p` = 8, `s` = a)",
            "tableName=`default`.`c`"
        ]
    );
    s.conf()
        .set(
            "spark.sql.legacy.skipTypeValidationOnAlterPartition",
            "true",
        )
        .unwrap();
    for value in ["007", "abc"] {
        assert_eq!(
            params(
                &s,
                &format!(
                    "alter table c partition (p='{value}', s='a') set location '{}'",
                    other.display()
                )
            ),
            [
                format!("partitionList=PARTITION (`p` = {value}, `s` = a)"),
                "tableName=`default`.`c`".to_owned()
            ]
        );
    }
}

/// `SET LOCATION` and `RENAME` report a spec in the order it was written, and a rename written out of column
/// order still stores and names the partition in column order (Spark 4.2.0 and 3.5.9, krishiv-internal
/// `research/spark/partition-followups/probe5.py`).
#[test]
fn partition_errors_list_a_spec_as_written() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "create table c (x int, p int, s string) using parquet partitioned by (p, s)",
    );
    run(&s, "alter table c add partition (p=7, s='a')");
    run(&s, "alter table c add partition (p=9, s='b')");
    let missing = |list: &str| {
        vec![
            format!("partitionList=PARTITION ({list})"),
            "tableName=`default`.`c`".to_owned(),
        ]
    };
    let n1 = dir.path().join("n1");
    assert_eq!(
        params(
            &s,
            &format!(
                "alter table c partition (s='a', p=8) set location '{}'",
                n1.display()
            )
        ),
        missing("`s` = a, `p` = 8")
    );
    assert_eq!(
        params(
            &s,
            "alter table c partition (s='a', p=8) rename to partition (s='b', p=10)"
        ),
        missing("`s` = a, `p` = 8")
    );
    assert_eq!(
        params(
            &s,
            "alter table c partition (s='a', p=7) rename to partition (s='b', p='009')"
        ),
        missing("`s` = b, `p` = 9")
    );
    assert_eq!(
        class(
            &s,
            "alter table c partition (s='a', p=7) rename to partition (s='b', p='009')"
        ),
        "PARTITIONS_ALREADY_EXIST"
    );
    run(
        &s,
        "alter table c partition (s='a', p=7) rename to partition (s='c', p='010')",
    );
    assert_eq!(rows(&s, "show partitions c"), ["p=10/s=c", "p=9/s=b"]);
    assert!(dir.path().join("c").join("p=10").join("s=c").is_dir());
}
