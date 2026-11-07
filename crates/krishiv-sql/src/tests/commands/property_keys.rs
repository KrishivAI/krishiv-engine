// SPDX-License-Identifier: Apache-2.0
//! Property lists (`TBLPROPERTIES`, `OPTIONS`, `DBPROPERTIES`, `SERDEPROPERTIES`, `WITH (...)`):
//! a repeated key is `DUPLICATE_KEY` at the list, keys without (or with) values are
//! `_LEGACY_ERROR_TEMP_0035` in Spark's map order, and both come where Spark's `AstBuilder` reads
//! the list. Every case is from a Spark 4.2.0 run (`research/spark/partition-spec-duplicate-key/`
//! `probe5.py` to `probe9.py` in krishiv-internal).

use super::*;
use krishiv_common::error::ErrorClass;

fn err(sql: &str) -> ParseError {
    match parse_statement(sql, &ParserOptions::default()) {
        Ok(s) => panic!("{sql} parsed: {s:?}"),
        Err(e) => e,
    }
}

fn assert_duplicate(sql: &str, key: &str) {
    let e = err(sql);
    assert_eq!(e.error.error_class(), ErrorClass::DuplicateKey, "{sql}");
    assert_eq!(e.error.sql_state(), Some("23505"), "{sql}");
    assert_eq!(
        e.error.message_parameters(),
        [("keyColumn", key.to_owned())],
        "{sql}"
    );
}

fn assert_message(sql: &str, message: &str) {
    let e = err(sql);
    assert_eq!(
        e.error.error_class(),
        ErrorClass::LegacyErrorTemp0035,
        "{sql}"
    );
    assert_eq!(
        e.error.message_parameters(),
        [("message", message.to_owned())],
        "{sql}"
    );
}

/// Every statement whose list goes through `visitPropertyList`.
#[test]
fn a_repeated_property_key_is_duplicate_key() {
    for sql in [
        "ALTER TABLE t SET TBLPROPERTIES ('k'='1', 'k'='2')",
        "ALTER TABLE t SET TBLPROPERTIES ('k', 'k')",
        "ALTER TABLE t SET TBLPROPERTIES ('k'='1', 'k')",
        "ALTER TABLE t SET TBLPROPERTIES ('j', 'k'='1', 'k'='2')",
        "ALTER TABLE t UNSET TBLPROPERTIES ('k', 'k')",
        "ALTER TABLE t UNSET TBLPROPERTIES IF EXISTS ('k', 'k')",
        "ALTER TABLE t UNSET TBLPROPERTIES ('k'='1', 'k')",
        "ALTER TABLE t SET SERDEPROPERTIES ('k'='1', 'k'='2')",
        "ALTER TABLE t PARTITION (p=1) SET SERDEPROPERTIES ('k'='1', 'k'='2')",
        "ALTER TABLE t PARTITION (p=1, p=2) SET SERDEPROPERTIES ('k'='1', 'k'='2')",
        "ALTER TABLE t PARTITION (p=1) SET SERDEPROPERTIES ('k'='1', 'k'='2', 'j')",
        "ALTER VIEW v SET TBLPROPERTIES ('k'='1', 'k'='2')",
        "ALTER VIEW v UNSET TBLPROPERTIES ('k', 'k')",
        "CREATE VIEW v TBLPROPERTIES ('k'='1', 'k'='2') AS SELECT 1",
        "CREATE VIEW v TBLPROPERTIES (k=1, k) AS SELECT 1",
        "CREATE VIEW v TBLPROPERTIES (k, k) AS SELECT 1",
        "CREATE VIEW v TBLPROPERTIES (k=1, k=2) AS SELECT CAST(1 AS INTX)",
        "CREATE TEMPORARY VIEW v TBLPROPERTIES ('k'='1', 'k'='2') AS SELECT 1",
        "CREATE TEMPORARY VIEW v WITH SCHEMA BINDING TBLPROPERTIES (k=1, k=2) AS SELECT 1",
        "CREATE TEMPORARY VIEW IF NOT EXISTS v TBLPROPERTIES (k=1, k=2) AS SELECT 1",
        "CREATE TEMPORARY VIEW a.b TBLPROPERTIES (k=1, k=2) AS SELECT 1",
        "CREATE OR REPLACE TEMPORARY VIEW v USING parquet OPTIONS ('k'='1', 'k'='2')",
        "CREATE OR REPLACE TEMPORARY VIEW v (a INT) USING parquet OPTIONS ('k'='1', 'k'='2')",
        "CREATE TEMPORARY VIEW v USING parquet OPTIONS (k=1, k)",
        "CREATE TEMPORARY VIEW v USING parquet OPTIONS (k, k)",
        "CREATE NAMESPACE n WITH DBPROPERTIES ('k'='1', 'k'='2')",
        "CREATE NAMESPACE n WITH PROPERTIES ('k'='1', 'k'='2')",
        "CREATE NAMESPACE n WITH DBPROPERTIES ('k'='1', 'k')",
        "CREATE NAMESPACE n WITH DBPROPERTIES ('k', 'k')",
        "CREATE NAMESPACE n WITH DBPROPERTIES ('k', 'j'=1, 'k')",
        "ALTER NAMESPACE n SET DBPROPERTIES ('k'='1', 'k'='2')",
        "ALTER NAMESPACE n SET PROPERTIES ('k'='1', 'k'='2')",
        "CACHE TABLE t OPTIONS ('k'='1', 'k'='2')",
        "CACHE TABLE t OPTIONS ('k'='1', 'k')",
        "CACHE LAZY TABLE c1 OPTIONS ('k'='1', 'k'='2') AS SELECT 1",
        "CACHE TABLE c1 OPTIONS ('k'='1', 'k'='2') AS SELECT CAST(1 AS INTX)",
        "INSERT INTO t WITH ('k'='1', 'k'='2') SELECT 1",
        "INSERT INTO t WITH (k=1, k=2) SELECT 1",
        "INSERT INTO t WITH (k, k) SELECT 1",
        "INSERT INTO t WITH (k=1, k) SELECT 1",
        "INSERT OVERWRITE t WITH (k=1, k=2) SELECT 1",
        "INSERT INTO TABLE t WITH (k=1, k=2) SELECT 1",
        "INSERT INTO t WITH (k=1, k=2) REPLACE WHERE CAST(1 AS INTX) = 1 SELECT 1",
        "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES ('k'='1', 'k'='2')",
        "CREATE TABLE t (a INT) STORED AS parquet TBLPROPERTIES (k=1, k=2)",
        "CREATE TABLE t (a INT, a INT) USING parquet TBLPROPERTIES (k=1, k=2)",
        "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k=1, k=2) STORED AS parquet",
        "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k=1, k=2) AS SELECT 1",
        "CREATE TABLE t USING parquet TBLPROPERTIES (k=1, k=2) AS SELECT CAST(1 AS INTX)",
        "CREATE TABLE t (a INT) USING parquet OPTIONS (p=1/0) TBLPROPERTIES (k=1, k=2)",
        "CREATE TABLE t (a INT) USING parquet OPTIONS (p) TBLPROPERTIES (k=1, k=2)",
        "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k=1, k=2) OPTIONS (p)",
        "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k=1, k=2, j)",
        "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k=1, k)",
        "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k=1, k=2) CLUSTER BY (a) PARTITIONED BY (a)",
        "REPLACE TABLE t (a INT) USING parquet TBLPROPERTIES (k=1, k=2)",
        "CREATE OR REPLACE TABLE t (a INT) USING parquet TBLPROPERTIES (k=1, k=2)",
        "CREATE TABLE t LIKE s TBLPROPERTIES (k=1, k=2)",
        "CREATE TABLE t LIKE s USING parquet TBLPROPERTIES (k, k)",
        "SELECT * FROM t WITH ('k'=1, 'k'=2)",
        "SELECT * FROM t WITH ('k'=1, 'k')",
        "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k'='1', 'k'='2')",
        "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k'='1', 'k')",
        "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k'='1', 'k'='2') AS SELECT 1",
        "CREATE TABLE t (a INT) USING parquet ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k'='1', 'k'='2')",
        "CREATE TABLE t LIKE s ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k'='1', 'k'='2')",
        "CREATE TABLE t LIKE s USING parquet ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k'='1', 'k'='2')",
    ] {
        assert_duplicate(sql, "`k`");
    }
    // Both lists of a `CREATE TABLE` repeat: the properties are read first, the options are not
    // checked for repeats at all.
    assert_duplicate(
        "CREATE TABLE t (a INT) USING parquet OPTIONS (k=1, k=2) TBLPROPERTIES (j=1, j=2)",
        "`j`",
    );
    assert_duplicate(
        "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (j=1, j=2) OPTIONS (k=1, k=2)",
        "`j`",
    );
    // A `CREATE TABLE` reads the SerDe's properties after its own, a `CREATE TABLE LIKE` before.
    assert_duplicate(
        "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k') TBLPROPERTIES (j=1, j=2)",
        "`j`",
    );
    assert_duplicate(
        "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES ('provider'='x', 'provider'='y')",
        "`provider`",
    );
    assert_duplicate(
        "CREATE NAMESPACE n WITH DBPROPERTIES ('location'='1', 'location'='2')",
        "`location`",
    );
    // The key Spark's `HashMap` iterates first.
    assert_duplicate(
        "ALTER TABLE t SET TBLPROPERTIES (a=1, a=2, b=1, b=2, c=1, c=2, d=1, d=2, e=1, e=2)",
        "`e`",
    );
}

/// The key is compared as written (no lower-casing), its parts unquoted, and named by `toSQLId`.
#[test]
fn a_property_key_is_compared_unquoted_and_named_by_to_sql_id() {
    let tp = "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES";
    let options = ParserOptions::default();
    assert!(parse_statement(&format!("{tp} ('k'='1', 'K'='2')"), &options).is_ok());
    for (list, key) in [
        ("(a.b=1, 'a.b'=2)", "`a`.`b`"),
        ("(`a`.b=1, a.b=2)", "`a`.`b`"),
        ("(`a`.b=1, `a`.b=2)", "`a`.`b`"),
        ("(`a.b`=1, `a.b`=2)", "`a`.`b`"),
        ("('my key'=1, 'my key'=2)", "`my key`"),
        ("(\"dq\"=1, \"dq\"=2)", "`dq`"),
        ("('x'=1, x=2)", "`x`"),
        ("(spark.sql.x=1, 'spark.sql.x'=2)", "`spark`.`sql`.`x`"),
        ("('' = 1, '' = 2)", "``"),
    ] {
        assert_duplicate(&format!("{tp} {list}"), key);
    }
    for (list, name) in [
        ("('a`b'=1, 'a`b'=2)", "a`b"),
        ("('.a'=1, '.a'=2)", ".a"),
        ("('a.'=1, 'a.'=2)", "a."),
        ("('a..b'=1, 'a..b'=2)", "a..b"),
        ("(`a``b`=1, `a``b`=2)", "a`b"),
    ] {
        let sql = format!("{tp} {list}");
        let e = err(&sql);
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidAttributeNameSyntax,
            "{sql}"
        );
        assert_eq!(
            e.error.message_parameters(),
            [("name", name.to_owned())],
            "{sql}"
        );
    }
    // A `CREATE TABLE`'s options are an expression list, which may repeat a key.
    for sql in [
        "CREATE TABLE t (a INT) USING parquet OPTIONS ('k'='1', 'k'='2')",
        "CREATE TABLE t (a INT) USING parquet OPTIONS (k 1, k 2)",
    ] {
        assert!(parse_statement(sql, &options).is_ok(), "{sql}");
    }
}

/// `DUPLICATE_KEY` spans the list from `(` to `)`, comments inside included.
#[test]
fn a_repeated_property_key_spans_the_list() {
    for (sql, list) in [
        (
            "ALTER TABLE t SET TBLPROPERTIES ('k'='1', 'k')",
            "('k'='1', 'k')",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES ('k'='1', 'k'='2')",
            "('k'='1', 'k'='2')",
        ),
        ("INSERT INTO t WITH (k=1, k=2) SELECT 1", "(k=1, k=2)"),
        ("SELECT * FROM t WITH ('k'=1, 'k'=2)", "('k'=1, 'k'=2)"),
        (
            "CACHE TABLE t OPTIONS ('k'='1', 'k'='2')",
            "('k'='1', 'k'='2')",
        ),
        (
            "CREATE NAMESPACE n WITH DBPROPERTIES ('k'='1', 'k'='2')",
            "('k'='1', 'k'='2')",
        ),
        (
            "ALTER TABLE t SET TBLPROPERTIES ( /* a */ k=1, k=2 -- b\n)",
            "( /* a */ k=1, k=2 -- b\n)",
        ),
    ] {
        assert_eq!(err(sql).span.text(sql), list, "{sql}");
    }
}

/// The errors Spark raises before it reads a property list.
#[test]
fn earlier_checks_come_before_the_property_list() {
    for (sql, class) in [
        (
            "CREATE TABLE t (a INTX) USING parquet TBLPROPERTIES (k=1, k=2)",
            ErrorClass::UnsupportedDatatype,
        ),
        (
            "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k=1, k=2) TBLPROPERTIES (j=1)",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k=1, k=2) COMMENT 'a' COMMENT 'b'",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k=1, k=2) CLUSTERED BY (a) INTO 0 BUCKETS",
            ErrorClass::InvalidBucketCount,
        ),
        (
            "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k) CLUSTERED BY (a) INTO 0 BUCKETS",
            ErrorClass::InvalidBucketCount,
        ),
        (
            "CREATE TABLE t (a INT) USING parquet OPTIONS (j) CLUSTERED BY (a) INTO 0 BUCKETS",
            ErrorClass::InvalidBucketCount,
        ),
        (
            "CREATE TABLE t (a INT) USING parquet PARTITIONED BY (b INTX) TBLPROPERTIES (k=1, k=2)",
            ErrorClass::UnsupportedDatatype,
        ),
        (
            "CREATE TABLE t (a INT) USING parquet PARTITIONED BY (b INTX) TBLPROPERTIES (k)",
            ErrorClass::UnsupportedDatatype,
        ),
        (
            "CREATE TABLE t (a INT) USING parquet SKEWED BY (a) ON (1) TBLPROPERTIES (k=1, k=2)",
            ErrorClass::InvalidStatementOrClause,
        ),
        (
            "CREATE TABLE t LIKE s TBLPROPERTIES (k=1, k=2) STORED AS parquet USING parquet",
            ErrorClass::InvalidStatementOrClause,
        ),
        (
            "CREATE TABLE t LIKE s TBLPROPERTIES (k) STORED AS parquet USING parquet",
            ErrorClass::InvalidStatementOrClause,
        ),
        (
            "CREATE TABLE t LIKE s TBLPROPERTIES (k=1, k=2) LOCATION 'a' LOCATION 'b'",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE OR REPLACE VIEW IF NOT EXISTS v TBLPROPERTIES (k=1, k=2) AS SELECT 1",
            ErrorClass::CreateViewWithIfNotExistsAndReplace,
        ),
        (
            "CREATE OR REPLACE VIEW IF NOT EXISTS v TBLPROPERTIES (k) AS SELECT 1",
            ErrorClass::CreateViewWithIfNotExistsAndReplace,
        ),
        (
            "CREATE VIEW v TBLPROPERTIES (k=1, k=2) TBLPROPERTIES (j=1) AS SELECT 1",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE VIEW v TBLPROPERTIES (k) TBLPROPERTIES (j=1) AS SELECT 1",
            ErrorClass::DuplicateClauses,
        ),
        (
            "CREATE VIEW v TBLPROPERTIES (k=1, k=2) PARTITIONED ON (a) AS SELECT 1",
            ErrorClass::InvalidStatementOrClause,
        ),
        (
            "CREATE VIEW v TBLPROPERTIES (k) PARTITIONED ON (a) AS SELECT 1",
            ErrorClass::InvalidStatementOrClause,
        ),
        (
            "CREATE TEMPORARY VIEW v (a INTX) USING parquet OPTIONS (k=1, k=2)",
            ErrorClass::UnsupportedDatatype,
        ),
        (
            "INSERT INTO t AS x WITH (k=1, k=2) REPLACE WHERE true SELECT 1",
            ErrorClass::InsertReplaceWhereTableAliasNotAllowed,
        ),
        (
            "INSERT INTO t WITH (k=1, k=2) SELECT CAST(1 AS INTX)",
            ErrorClass::UnsupportedDatatype,
        ),
        (
            "CREATE TABLE t (a INT) USING parquet OPTIONS (p=CAST('x' AS INTX), k)",
            ErrorClass::UnsupportedDatatype,
        ),
    ] {
        assert_eq!(err(sql).error.error_class(), class, "{sql}");
    }
}

/// Keys without a value (`SET` lists) or with one (`UNSET` lists), in the property map's order,
/// over the statement; a `CREATE TABLE` option without a value, in written order.
#[test]
fn misvalued_property_keys_follow_sparks_map_order() {
    for (sql, message) in [
        (
            "ALTER TABLE t SET TBLPROPERTIES (yoo, sgf, a=1, b=1, c=1)",
            "Values must be specified for key(s): [sgf,yoo]",
        ),
        (
            "ALTER TABLE t SET TBLPROPERTIES (yoo, sgf, a=1)",
            "Values must be specified for key(s): [yoo,sgf]",
        ),
        (
            "ALTER TABLE t UNSET TBLPROPERTIES (yoo=1, sgf=1, a, b, c)",
            "Values should not be specified for key(s): [sgf,yoo]",
        ),
        (
            "ALTER TABLE t UNSET TBLPROPERTIES (yoo=1, sgf=1, a)",
            "Values should not be specified for key(s): [yoo,sgf]",
        ),
        (
            "ALTER TABLE t SET TBLPROPERTIES (jee, hb, b, jbd=1, bg=1, cfi=1)",
            "Values must be specified for key(s): [b,jee,hb]",
        ),
        (
            "ALTER TABLE t UNSET TBLPROPERTIES (jee=1, hb=1, b=1, jbd, bg, cfi)",
            "Values should not be specified for key(s): [hb,b,jee]",
        ),
        (
            "ALTER TABLE t SET TBLPROPERTIES (i=1, fj=1, d, jg, jc)",
            "Values must be specified for key(s): [jg,d,jc]",
        ),
        (
            "ALTER TABLE t UNSET TBLPROPERTIES (i, fj, d=1, jg=1, jc=1)",
            "Values should not be specified for key(s): [jc,jg,d]",
        ),
        (
            "ALTER TABLE t PARTITION (p=1, p=2) SET SERDEPROPERTIES ('k'='1', 'j')",
            "Values must be specified for key(s): [j]",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k)",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (yoo, sgf, a=1, b=1, c=1)",
            "Values must be specified for key(s): [sgf,yoo]",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k) OPTIONS (j)",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet OPTIONS (j) TBLPROPERTIES (k)",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet OPTIONS ('k'='1', 'k'='2') TBLPROPERTIES (j)",
            "Values must be specified for key(s): [j]",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k) AS SELECT CAST(1 AS INTX)",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k) AS SELECT 1",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet TBLPROPERTIES (k) STORED AS parquet",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE TABLE t LIKE s USING parquet TBLPROPERTIES (k)",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE VIEW v TBLPROPERTIES (k) AS SELECT CAST(1 AS INTX)",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE TEMPORARY VIEW v TBLPROPERTIES (k) AS SELECT 1",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE VIEW v TBLPROPERTIES (yoo, sgf, a=1, b=1, c=1) AS SELECT 1",
            "Values must be specified for key(s): [sgf,yoo]",
        ),
        (
            "CREATE NAMESPACE n WITH DBPROPERTIES ('k')",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE NAMESPACE n WITH DBPROPERTIES ('location')",
            "Values must be specified for key(s): [location]",
        ),
        (
            "CREATE NAMESPACE n WITH DBPROPERTIES (yoo, sgf, a=1, b=1, c=1)",
            "Values must be specified for key(s): [sgf,yoo]",
        ),
        (
            "ALTER NAMESPACE n SET DBPROPERTIES ('k')",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CACHE TABLE t OPTIONS ('k')",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE TABLE t (a INT) TBLPROPERTIES (j) ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k'='1', 'k'='2')",
            "Values must be specified for key(s): [j]",
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k'='1', 'k'='2') OPTIONS (j)",
            "A value must be specified for the key: j.",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet OPTIONS (k, k)",
            "A value must be specified for the key: k.",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet OPTIONS (j, k)",
            "A value must be specified for the key: j.",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet OPTIONS (a.b)",
            "A value must be specified for the key: a.b.",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet OPTIONS (k, p=CAST('x' AS INTX))",
            "A value must be specified for the key: k.",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet OPTIONS (j) STORED AS parquet",
            "A value must be specified for the key: j.",
        ),
        (
            "CREATE TABLE t (a INT) USING parquet OPTIONS (j) AS SELECT 1",
            "A value must be specified for the key: j.",
        ),
    ] {
        assert_message(sql, message);
        assert_eq!(err(sql).span.range(), 0..sql.len(), "{sql}");
    }
    // A SerDe's properties: at the `ROW FORMAT` clause.
    for (sql, message) in [
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES (yoo, sgf, a=1, b=1, c=1)",
            "Values must be specified for key(s): [sgf,yoo]",
        ),
        (
            "CREATE TABLE t LIKE s ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k') TBLPROPERTIES (j=1, j=2)",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE TABLE t LIKE s TBLPROPERTIES (j=1, j=2) ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k')",
            "Values must be specified for key(s): [k]",
        ),
        (
            "CREATE TABLE t LIKE s USING parquet ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k')",
            "Values must be specified for key(s): [k]",
        ),
    ] {
        assert_message(sql, message);
        let span = err(sql).span.text(sql).to_owned();
        assert!(
            span.starts_with("ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES (") && span.ends_with(')'),
            "{sql}: {span}"
        );
    }
    // A table reference's options: the map order too, at the table name.
    let sql = "SELECT * FROM t WITH (yoo, sgf, a=1, b=1, c=1)";
    let e = err(sql);
    assert_eq!(
        e.error.message_parameters(),
        [(
            "message",
            "Values must be specified for key(s): [sgf,yoo]".to_owned()
        )],
    );
    assert_eq!(e.span.text(sql), "t");
    // A `CREATE TABLE` option key written as a string is not echoed (it may hold a secret).
    let e = err("CREATE TABLE t (a INT) USING parquet OPTIONS ('k')");
    assert_eq!(e.error.error_class(), ErrorClass::LegacyErrorTemp0035);
    assert_eq!(
        e.error.message_parameters(),
        [("message", krishiv_common::secret::REDACTED.to_owned())]
    );
}
