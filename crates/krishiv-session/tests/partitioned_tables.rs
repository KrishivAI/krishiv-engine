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
