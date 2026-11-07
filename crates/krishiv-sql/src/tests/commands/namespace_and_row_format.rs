// SPDX-License-Identifier: Apache-2.0
//! `CREATE NAMESPACE`'s clause checks (`DUPLICATE_CLAUSES`, both `PROPERTIES` and `DBPROPERTIES`,
//! reserved property keys) and `ROW FORMAT` against `STORED AS`, in Spark's
//! `AstBuilder` order. Every case is from a Spark 4.2.0 run (`research/spark/row-format-file-format/`
//! `probe11.py`, `probe12.py` in krishiv-internal).

use super::*;
use krishiv_common::error::ErrorClass;

fn err(sql: &str) -> ParseError {
    match parse_statement(sql, &ParserOptions::default()) {
        Ok(s) => panic!("{sql} parsed: {s:?}"),
        Err(e) => e,
    }
}

fn ok(sql: &str) {
    if let Err(e) = parse_statement(sql, &ParserOptions::default()) {
        panic!("{sql}: {}", e.error);
    }
}

/// `sql` fails with `class` and `params`, at the whole statement unless `whole` is false.
fn assert_error(sql: &str, class: ErrorClass, params: &[(&str, &str)], whole: bool) {
    let e = err(sql);
    assert_eq!(e.error.error_class(), class, "{sql}");
    let params: Vec<(&str, String)> = params.iter().map(|(k, v)| (*k, (*v).to_owned())).collect();
    assert_eq!(e.error.message_parameters(), params, "{sql}");
    if whole {
        assert_eq!(e.span.range(), 0..sql.len(), "{sql}");
    }
}

#[test]
fn create_namespace_checks_follow_sparks_order() {
    let dup = |c| [("clauseName", c)];
    for (sql, clause) in [
        ("CREATE NAMESPACE n COMMENT 'a' COMMENT 'b'", "COMMENT"),
        (
            "CREATE NAMESPACE n LOCATION '/a' COMMENT 'a' LOCATION '/b' COMMENT 'b'",
            "COMMENT",
        ),
        (
            "CREATE NAMESPACE n WITH PROPERTIES (a=1) WITH PROPERTIES (b=1)",
            "WITH PROPERTIES",
        ),
        (
            "CREATE SCHEMA n WITH DBPROPERTIES (a=1) WITH DBPROPERTIES (b=1)",
            "WITH DBPROPERTIES",
        ),
        (
            "CREATE DATABASE n WITH DBPROPERTIES (a=1, a=2) LOCATION '/x' LOCATION '/y'",
            "LOCATION",
        ),
        (
            "CREATE NAMESPACE n DEFAULT COLLATION UTF8_LCASE DEFAULT COLLATION UTF8_LCASE \
             LOCATION '/a' LOCATION '/b'",
            "DEFAULT COLLATION",
        ),
        (
            "CREATE NAMESPACE n WITH PROPERTIES (a=1) WITH DBPROPERTIES (b=1) WITH PROPERTIES (c=1)",
            "WITH PROPERTIES",
        ),
    ] {
        assert_error(sql, ErrorClass::DuplicateClauses, &dup(clause), true);
    }
    for sql in [
        "CREATE NAMESPACE n WITH PROPERTIES (a=1) WITH DBPROPERTIES (b=1)",
        "CREATE NAMESPACE n WITH DBPROPERTIES (a=1) WITH PROPERTIES (a=1, a=2)",
    ] {
        assert_error(
            sql,
            ErrorClass::UnsupportedFeatureSetPropertiesAndDbproperties,
            &[],
            true,
        );
    }
    let location = "please use the LOCATION clause to specify it";
    let owner = "it will be set to the current user";
    let collation = "please use the DEFAULT COLLATION clause to specify it";
    for (sql, property, msg) in [
        (
            "CREATE NAMESPACE n WITH PROPERTIES (location='/x')",
            "location",
            location,
        ),
        (
            "CREATE NAMESPACE n WITH PROPERTIES ('location'='/x')",
            "location",
            location,
        ),
        (
            "CREATE NAMESPACE n WITH PROPERTIES (owner='me')",
            "owner",
            owner,
        ),
        (
            "CREATE NAMESPACE n WITH PROPERTIES (collation='UTF8_LCASE')",
            "collation",
            collation,
        ),
        (
            "CREATE NAMESPACE n WITH PROPERTIES (owner='me', location='/x')",
            "owner",
            owner,
        ),
        // Five keys: the property map's hash order.
        (
            "CREATE NAMESPACE n WITH PROPERTIES (k1=1,k2=2,k3=3,location='/x',owner='me')",
            "location",
            location,
        ),
        (
            "CREATE NAMESPACE n WITH PROPERTIES \
             (k5=1,k1=1,k2=2,k3=3,k4=4,owner='me',location='/x',collation='c')",
            "location",
            location,
        ),
        // Reserved keys before the collation clause.
        (
            "CREATE NAMESPACE n DEFAULT COLLATION UTF8_LCASE WITH PROPERTIES (owner='me')",
            "owner",
            owner,
        ),
        (
            "ALTER NAMESPACE default SET PROPERTIES (location='/x')",
            "location",
            location,
        ),
        (
            "ALTER NAMESPACE default SET DBPROPERTIES (owner='me')",
            "owner",
            owner,
        ),
    ] {
        assert_error(
            sql,
            ErrorClass::UnsupportedFeatureSetNamespaceProperty,
            &[("property", property), ("msg", msg)],
            true,
        );
    }
    // The key list's own checks come first.
    for sql in [
        "CREATE NAMESPACE n WITH PROPERTIES (location='/x', location='/y')",
        "ALTER NAMESPACE default SET PROPERTIES (a=1, a=2, location='/x')",
    ] {
        assert_eq!(
            err(sql).error.error_class(),
            ErrorClass::DuplicateKey,
            "{sql}"
        );
    }
    for sql in [
        "CREATE NAMESPACE n WITH PROPERTIES (location)",
        "ALTER NAMESPACE default SET PROPERTIES (location)",
    ] {
        assert_error(
            sql,
            ErrorClass::LegacyErrorTemp0035,
            &[("message", "Values must be specified for key(s): [location]")],
            false,
        );
    }
    // Reserved keys compare exactly; `comment` is not reserved.
    ok("CREATE NAMESPACE n WITH PROPERTIES (Location='/x')");
    ok("CREATE NAMESPACE n COMMENT 'x' WITH PROPERTIES (comment='y')");
    ok("CREATE NAMESPACE n COMMENT 'x' LOCATION '/a' WITH DBPROPERTIES (a=1)");
    // Spark refuses `DEFAULT COLLATION` with schema-level collations off; Krishiv accepts it.
    ok("CREATE NAMESPACE n DEFAULT COLLATION UTF8_LCASE");
}

#[test]
fn row_format_must_fit_the_file_format() {
    let serde = |f: &str| {
        format!("ROW FORMAT SERDE is incompatible with format '{f}', which also specifies a serde")
    };
    let delimited =
        |f: &str| format!("ROW FORMAT DELIMITED is only compatible with 'textfile', not '{f}'");
    let cases = [
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS parquet",
            serde("parquet"),
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS PARQUET",
            serde("parquet"),
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS bogus",
            serde("bogus"),
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS jsonfile",
            serde("jsonfile"),
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS ORC",
            delimited("orc"),
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT DELIMITED STORED AS Bogus",
            delimited("bogus"),
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT DELIMITED STORED AS sequencefile",
            delimited("sequencefile"),
        ),
        // Before the SerDe's properties.
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k'='1','k'='2') \
             STORED AS orc",
            serde("orc"),
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k') STORED AS orc",
            serde("orc"),
        ),
        // Before `USING` beside a serde clause.
        (
            "CREATE TABLE t (a INT) USING parquet ROW FORMAT SERDE 'x' STORED AS parquet",
            serde("parquet"),
        ),
        // A CTAS, and clauses in any order.
        (
            "CREATE TABLE t ROW FORMAT SERDE 'x' STORED AS parquet AS SELECT 1",
            serde("parquet"),
        ),
        (
            "CREATE TABLE t (a INT) STORED AS parquet ROW FORMAT SERDE 'x' PARTITIONED BY (b INT, b INT)",
            serde("parquet"),
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS parquet CLUSTER BY (a) \
             PARTITIONED BY (b)",
            serde("parquet"),
        ),
        (
            "CREATE TABLE t LIKE src ROW FORMAT SERDE 'x' STORED AS parquet",
            serde("parquet"),
        ),
        (
            "CREATE TABLE t LIKE src ROW FORMAT DELIMITED STORED AS orc",
            delimited("orc"),
        ),
        (
            "CREATE TABLE t LIKE src USING parquet ROW FORMAT SERDE 'x' STORED AS parquet",
            serde("parquet"),
        ),
    ];
    let combination = |r: &str, f: &str| format!("Unexpected combination of {r} and {f}");
    let cases = cases.into_iter().chain([
        // A quoted name is read unquoted.
        ("CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS `parquet`", serde("parquet")),
        (
            "CREATE TABLE t (a INT) ROW FORMAT DELIMITED LINES TERMINATED BY ';' STORED AS orc",
            delimited("orc"),
        ),
        // `STORED BY` beside any row format: the clauses' children, spaced as Spark's contexts.
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED BY 'h'",
            combination("ROW FORMAT SERDE 'x'", "STORED BY 'h'"),
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT  SERDE   'x' WITH SERDEPROPERTIES ('k'='v', 'q'='1') \
             STORED  BY  'h'",
            combination(
                "ROW FORMAT SERDE 'x' WITH SERDEPROPERTIES ('k'='v','q'='1')",
                "STORED BY 'h'",
            ),
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED BY 'h'",
            combination("ROW FORMAT DELIMITED FIELDS TERMINATED BY ','", "STORED BY 'h'"),
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT DELIMITED STORED BY 'h'",
            combination("ROW FORMAT DELIMITED", "STORED BY 'h'"),
        ),
        (
            r"CREATE TABLE t (a INT) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' ESCAPED BY '\\' LINES TERMINATED BY '\n' NULL DEFINED AS 'N' STORED BY 'h'",
            combination(
                r"ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' ESCAPED BY '\\' LINES TERMINATED BY '\n' NULL DEFINED AS 'N'",
                "STORED BY 'h'",
            ),
        ),
        (
            "CREATE TABLE t LIKE src ROW FORMAT SERDE 'x' STORED BY 'h'",
            combination("ROW FORMAT SERDE 'x'", "STORED BY 'h'"),
        ),
    ]);
    for (sql, message) in cases {
        assert_error(
            sql,
            ErrorClass::LegacyErrorTemp0035,
            &[("message", &message)],
            true,
        );
    }
    // Checks that come first.
    let e =
        err("CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS parquet SKEWED BY (a) ON (1)");
    assert_eq!(e.error.error_class(), ErrorClass::InvalidStatementOrClause);
    let e = err(
        "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS parquet TBLPROPERTIES ('k'='1','k'='2')",
    );
    assert_eq!(e.error.error_class(), ErrorClass::DuplicateKey);
    for (sql, message) in [
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS parquet OPTIONS (a)",
            "A value must be specified for the key: a.",
        ),
        (
            "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS parquet TBLPROPERTIES ('a')",
            "Values must be specified for key(s): [a]",
        ),
    ] {
        assert_error(
            sql,
            ErrorClass::LegacyErrorTemp0035,
            &[("message", message)],
            false,
        );
    }
    // The pairs Spark allows.
    for sql in [
        "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS sequencefile",
        "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS RCFile",
        "CREATE TABLE t (a INT) ROW FORMAT SERDE 'x' STORED AS textfile",
        "CREATE TABLE t (a INT) ROW FORMAT DELIMITED STORED AS TextFile",
        "CREATE TABLE t (a INT) ROW FORMAT DELIMITED",
        "CREATE TABLE t (a INT) STORED AS parquet",
    ] {
        ok(sql);
    }
}
