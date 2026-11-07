// SPDX-License-Identifier: Apache-2.0
//! `ROW FORMAT` against `STORED AS|BY`, in Spark's `AstBuilder` order. Every case is from a Spark
//! 4.2.0 run (`research/spark/row-format-file-format/probe11.py`, `probe12.py` in
//! krishiv-internal).

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
