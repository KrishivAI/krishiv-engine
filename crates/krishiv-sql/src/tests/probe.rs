// SPDX-License-Identifier: Apache-2.0
//! The parse probe recorded from a real Spark 4.2 run
//! (`tests/fixtures/spark-4.2.0/sql-parse-probe.json`): statements and type
//! strings with Spark's outcome, error class, parameters and position.

use super::*;

/// Statements where this parser and Spark 4.2 disagree, with the reason.
const SPARK_DEVIATIONS: &[(&str, &str)] = &[
    (
        "SELECT (SELECT 1",
        "ANTLR adds the hint `: missing ')'` (error quality, P2-T077)",
    ),
    (
        "SELECT a[1",
        "ANTLR adds the hint `: missing ']'` (error quality, P2-T077)",
    ),
    (
        "SELECT a FROM t ORDER BY",
        "ANTLR drops the trailing `BY` as extra input (P2-T077)",
    ),
    (
        "SELECT a, count(*) FROM t GROUP BY a HAVING",
        "ANTLR drops the trailing `HAVING` as extra input (P2-T077)",
    ),
    (
        "SELECT a FROM",
        "with enforced keywords ANTLR reports `FROM`, not the end of input (P2-T077)",
    ),
    (
        "SELECT 1 'secret'",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT * FROM t end",
        "with enforced keywords ANTLR adds no `extra input` hint here (P2-T077)",
    ),
    (
        "EXPLAIN ANALYZE TABLE t COMPUTE STATISTICS",
        "`ANALYZE TABLE` is not a supported statement",
    ),
    (
        "SELECT /*+ x(a b) */ * FROM t",
        "ANTLR drops `b` before `)` as extra input (P2-T077)",
    ),
    (
        "SELECT CAST(x AS array<int)",
        "ANTLR adds the hint `: missing ')'` (P2-T077)",
    ),
    (
        "SELECT CAST(x AS map<int>)",
        "ANTLR adds the hint `: missing ')'` (P2-T077)",
    ),
    (
        "SELECT CAST(x AS struct<a int,>)",
        "ANTLR adds the hint `: missing ')'` (P2-T077)",
    ),
    (
        "SELECT CAST(x AS decimal(10,2)",
        "ANTLR adds the hint `: missing ')'` (P2-T077)",
    ),
    (
        "SELECT CAST(x AS array<int>>)",
        "ANTLR drops the second `>` as extra input (P2-T077)",
    ),
    (
        "SELECT CAST(x AS foo x)",
        "ANTLR drops `x` before `)` as extra input (P2-T077)",
    ),
];

/// Type strings where this parser and Spark 4.2 disagree: `(mode, text,
/// reason)`.
const TYPE_DEVIATIONS: &[(&str, &str, &str)] = &[];

/// Error classes Spark raises while parsing a statement (not in
/// `AstBuilder`): the parser must raise the same class and parameters.
const PARSER_CLASSES: &[&str] = &[
    "PARSE_SYNTAX_ERROR",
    "PARSE_EMPTY_STATEMENT",
    "UNCLOSED_BRACKETED_COMMENT",
    "INVALID_IDENTIFIER",
    "SYNTAX_DISCONTINUED.BANG_EQUALS_NOT",
    "FAILED_TO_PARSE_TOO_COMPLEX",
    "INVALID_STATEMENT_OR_CLAUSE",
];

/// The case's position as a character offset, if Spark recorded one: the
/// query context's start, else a `ParseException`'s position on line 1.
fn spark_position(case: &serde_json::Value) -> Option<u64> {
    case["start"].as_u64().or_else(|| {
        (case["line"].as_u64() == Some(1))
            .then(|| case["pos"].as_u64())
            .flatten()
    })
}

/// Whether `ours` is Spark's error of `case`: class, parameters and, when
/// `position` is given, the character offset of its span.
fn same_error(
    sql: &str,
    ours: &ParseError,
    case: &serde_json::Value,
    position: Option<u64>,
) -> bool {
    let params: serde_json::Map<String, serde_json::Value> = ours
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone().into()))
        .collect();
    let at = sql.get(..ours.span.start).map(|p| p.chars().count() as u64);
    ours.error.error_class().name() == case["error_class"].as_str().unwrap_or_default()
        && Some(&params) == case["message_parameters"].as_object()
        && position.is_none_or(|p| at == Some(p))
}

/// Statements Spark parses must parse; errors Spark raises while parsing
/// must match in class, message parameters and (where Spark records a query
/// context) position; errors Spark raises when building the plan must parse
/// here (the post-parse validation, P2-T079, raises them). Type strings go
/// through the entry point of their mode and must match Spark's outcome,
/// with positions.
#[test]
fn agrees_with_spark_on_the_parse_probe() {
    let path = format!(
        "{}/../../tests/fixtures/spark-4.2.0/sql-parse-probe.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut deviations = vec![];
    let mut type_deviations = vec![];
    let mut validated = 0;
    for case in doc["cases"].as_array().unwrap() {
        let sql = case["sql"].as_str().unwrap();
        let conf = case["conf"].as_object().unwrap();
        let options = ParserOptions::from_conf(|k| conf.get(k).and_then(|v| v.as_str()));
        let parses = case["parses"].as_bool().unwrap();
        let mode = case["mode"].as_str().unwrap_or("plan");
        if mode != "plan" {
            let ours = match mode {
                "datatype" => parse_data_type(sql, &options).map(drop),
                "tableschema" => parse_table_schema(sql, &options).map(drop),
                "fromddl" => parse_ddl_type(sql, &options).map(drop),
                other => panic!("unknown probe mode {other}"),
            };
            let agrees = match &ours {
                Ok(()) => parses,
                Err(e) => !parses && same_error(sql, e, case, spark_position(case)),
            };
            if !agrees {
                type_deviations.push((
                    mode,
                    sql,
                    ours.err().map(|e| (e.error.to_string(), e.span.range())),
                ));
            }
            continue;
        }
        let ours = parse_statement(sql, &options);
        let agrees = if parses {
            ours.is_ok()
        } else if PARSER_CLASSES.contains(&case["error_class"].as_str().unwrap()) {
            ours.as_ref()
                .is_err_and(|e| same_error(sql, e, case, case["start"].as_u64()))
        } else {
            // Raised by the post-parse checks (P2-T079).
            validated += 1;
            ours.as_ref()
                .is_err_and(|e| same_error(sql, e, case, case["start"].as_u64()))
        };
        if !agrees {
            deviations.push(sql);
        }
    }
    for d in &type_deviations {
        eprintln!("TYPEDEV {d:?}");
    }
    let mut known: Vec<&str> = SPARK_DEVIATIONS.iter().map(|(sql, _)| *sql).collect();
    known.sort_unstable();
    deviations.sort_unstable();
    assert_eq!(deviations, known);
    let known: Vec<(&str, &str)> = TYPE_DEVIATIONS
        .iter()
        .map(|(m, sql, _)| (*m, *sql))
        .collect();
    let found: Vec<(&str, &str)> = type_deviations
        .iter()
        .map(|(m, sql, _)| (*m, *sql))
        .collect();
    assert_eq!(found, known, "{type_deviations:#?}");
    assert_eq!(validated, 20);
}

/// Every DDL type string of the Spark type-string fixture
/// (`type-strings.json`, `CatalystSqlParser.parseDataType`) gets Spark's
/// outcome from [`parse_data_type`]: the same canonical type, or the same
/// error class and parameters. Spark's unclassified `NumberFormatException`
/// for an out-of-range decimal argument is a decimal bound error here.
#[test]
fn agrees_with_spark_on_the_type_string_fixture() {
    let path = format!(
        "{}/../../tests/fixtures/spark-4.2.0/type-strings.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let d = ParserOptions::default();
    for case in doc["ddl"].as_array().unwrap() {
        let ddl = case["ddl"].as_str().unwrap();
        if ddl.to_ascii_lowercase().contains(" collate ") {
            // Collated string types arrive with `COLLATE` (P2-T075b).
            assert!(parse_data_type(ddl, &d).is_err(), "{ddl}");
            continue;
        }
        let ours = parse_data_type(ddl, &d);
        match case["error_class"].as_str() {
            None => {
                let t = ours.unwrap_or_else(|e| panic!("{ddl}: {}", e.error));
                // Out-of-scope types resolve once `krishiv-types` has them.
                if let Ok(resolved) = t.to_data_type() {
                    assert_eq!(resolved.simple_string(), case["simple_string"], "{ddl}");
                }
            }
            Some("UNCLASSIFIED") => {
                let class = ours.unwrap_err().error.error_class();
                assert!(
                    [
                        ErrorClass::DecimalPrecisionExceedsMaxPrecision,
                        ErrorClass::LegacyErrorTemp1228
                    ]
                    .contains(&class),
                    "{ddl}: {class:?}"
                );
            }
            Some(_) => {
                let e = ours.unwrap_err();
                assert!(same_error(ddl, &e, case, None), "{ddl}: {}", e.error);
            }
        }
    }
}

/// Every class the post-parse checks raise, on a statement Spark rejects
/// with it (the probe fixture holds Spark's runs of these shapes).
#[test]
fn post_parse_checks_raise_spark_classes() {
    let d = ParserOptions::default();
    let class = |sql: &str| match parse_statement(sql, &d) {
        Ok(_) => panic!("{sql} parsed"),
        Err(e) => e.error.error_class(),
    };
    for (sql, expected) in [
        ("SELECT CAST(x AS char)", ErrorClass::DatatypeMissingSize),
        (
            "SELECT CAST(x AS array)",
            ErrorClass::IncompleteTypeDefinitionArray,
        ),
        (
            "SELECT CAST(x AS map)",
            ErrorClass::IncompleteTypeDefinitionMap,
        ),
        (
            "SELECT CAST(x AS struct)",
            ErrorClass::IncompleteTypeDefinitionStruct,
        ),
        (
            "SELECT CAST(x AS time(7))",
            ErrorClass::UnsupportedTimePrecision,
        ),
        (
            "WITH t AS (SELECT 1), T AS (SELECT 2) SELECT 1",
            ErrorClass::DuplicatedCteNames,
        ),
        (
            "SELECT * FROM a NATURAL JOIN LATERAL (SELECT 1)",
            ErrorClass::IncompatibleJoinTypes,
        ),
        (
            "SELECT * FROM a JOIN LATERAL (SELECT 1) USING (c)",
            ErrorClass::UnsupportedFeatureLateralJoinUsing,
        ),
        (
            "VALUES (1, 2), (3)",
            ErrorClass::InvalidInlineTableNumColumnsMismatch,
        ),
        ("SELECT 128Y", ErrorClass::InvalidNumericLiteralRange),
        (
            "SELECT 1.20E-38BD",
            ErrorClass::DecimalPrecisionExceedsMaxPrecision,
        ),
        (
            "SELECT dateadd('DAY', 1, d)",
            ErrorClass::InvalidParameterValueDatetimeUnit,
        ),
        (
            "SELECT 1 IN ()",
            ErrorClass::InvalidSqlSyntaxEmptyInPredicate,
        ),
        (
            "SELECT 1 FROM t WINDOW w AS (), w AS ()",
            ErrorClass::InvalidSqlSyntaxRepetitiveWindowDefinition,
        ),
        ("SELECT DATE '2020-02-30'", ErrorClass::InvalidTypedLiteral),
        ("SELECT TIME '24:00'", ErrorClass::InvalidTypedLiteral),
        (
            "SELECT TIMESTAMP_NTZ '10:00'",
            ErrorClass::InvalidTypedLiteral,
        ),
        ("SELECT X'XY'", ErrorClass::InvalidTypedLiteral),
        ("SELECT INTERVAL '1 d'", ErrorClass::InvalidTypedLiteral),
        ("SELECT GEO 'x'", ErrorClass::UnsupportedTypedLiteral),
        (
            "SELECT INTERVAL '1' YEAR TO SECOND",
            ErrorClass::InvalidIntervalFormatUnsupportedFromToExpression,
        ),
        (
            "SELECT INTERVAL '10-12' YEAR TO MONTH",
            ErrorClass::InvalidIntervalFormatIntervalParsing,
        ),
        (
            "SELECT INTERVAL '2' YEAR TO MONTH",
            ErrorClass::InvalidIntervalFormatUnmatchedFormatString,
        ),
        (
            "SELECT INTERVAL '1 2' DAY TO SECOND",
            ErrorClass::InvalidIntervalFormatUnmatchedFormatStringWithNotice,
        ),
        (
            "SELECT INTERVAL '1 25:00:00' DAY TO SECOND",
            ErrorClass::LegacyErrorTemp0063,
        ),
        (
            "SELECT INTERVAL '106751991 23' DAY TO HOUR",
            ErrorClass::IntervalArithmeticOverflowWithoutSuggestion,
        ),
        (
            "SELECT INTERVAL 1.5 DAY",
            ErrorClass::InvalidIntervalFormatInvalidFraction,
        ),
        (
            "SELECT INTERVAL 1 YEAR '2-1' YEAR TO MONTH",
            ErrorClass::LegacyErrorTemp0024,
        ),
        (
            "SELECT INTERVAL '1 day' DAY",
            ErrorClass::LegacyErrorTemp0026,
        ),
        (
            "SELECT INTERVAL 1 YEAR TO MONTH",
            ErrorClass::LegacyErrorTemp0027,
        ),
        (
            "SELECT INTERVAL 1 YEAR 1 DAY",
            ErrorClass::LegacyErrorTemp0029,
        ),
        ("SELECT * FROM t WITH (k)", ErrorClass::LegacyErrorTemp0035),
        (
            "WITH r MAX RECURSION LEVEL 5 AS (SELECT 1) SELECT 1",
            ErrorClass::LegacyErrorTemp0035,
        ),
        (
            "SELECT a FROM t GROUP BY CUBE(())",
            ErrorClass::LegacyErrorTemp0051,
        ),
        ("SELECT a LIKE ANY ()", ErrorClass::LegacyErrorTemp0064),
        (
            "SELECT sum(a) OVER (ORDER BY a ROWS x PRECEDING) FROM t",
            ErrorClass::LegacyErrorTemp0064,
        ),
    ] {
        assert_eq!(class(sql), expected, "{sql}");
    }
    // The configs that relax the checks.
    let legacy = ParserOptions {
        legacy_interval: true,
        ..d
    };
    assert!(parse_statement("SELECT INTERVAL 1 YEAR 1 DAY", &legacy).is_ok());
    let ntz =
        ParserOptions::from_conf(|k| (k == "spark.sql.timestampType").then_some("TIMESTAMP_NTZ"));
    assert!(parse_statement("SELECT TIMESTAMP '10:00'", &ntz).is_err());
    assert!(parse_statement("SELECT TIMESTAMP '10:00'", &d).is_ok());
    // A string option key is never echoed.
    let e = parse_statement("SELECT * FROM t WITH ('secret')", &d).unwrap_err();
    assert!(!e.error.to_string().contains("secret"), "{}", e.error);
}
