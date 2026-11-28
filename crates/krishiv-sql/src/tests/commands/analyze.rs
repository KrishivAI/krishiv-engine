// SPDX-License-Identifier: Apache-2.0
//! `ANALYZE TABLE` and `ANALYZE TABLES`: the tree and the errors. Every error is from a Spark 4.3.0
//! run (`NOSCAN`'s place takes any identifier, and another word is refused after parsing).

use super::*;
use krishiv_common::error::ErrorClass;

fn analyze_tables(sql: &str) -> ast::AnalyzeTables {
    match parse_statement(sql, &ParserOptions::default()) {
        Ok(ast::Statement::AnalyzeTables(a)) => a,
        other => panic!("{sql}: {other:?}"),
    }
}

#[test]
fn analyze_tables_reads_an_optional_schema_and_noscan() {
    for (sql, schema, noscan) in [
        ("ANALYZE TABLES COMPUTE STATISTICS", None, false),
        ("ANALYZE TABLES COMPUTE STATISTICS NOSCAN", None, true),
        ("ANALYZE TABLES COMPUTE STATISTICS `noscan`", None, true),
        ("ANALYZE TABLES IN db COMPUTE STATISTICS", Some("db"), false),
        (
            "analyze tables from spark_catalog.Db compute statistics NoScan;",
            Some("spark_catalog.Db"),
            true,
        ),
    ] {
        let a = analyze_tables(sql);
        assert_eq!(a.noscan, noscan, "{sql}");
        assert_eq!(
            a.namespace.map(|n| n.span().text(sql).to_owned()),
            schema.map(str::to_owned),
            "{sql}"
        );
        assert_eq!(a.span.range(), 0..sql.trim_end_matches(';').len(), "{sql}");
    }
}

#[test]
fn a_word_other_than_noscan_is_refused_over_the_statement() {
    for (sql, word) in [
        ("ANALYZE TABLES COMPUTE STATISTICS MiXed", "MIXED"),
        ("ANALYZE TABLES COMPUTE STATISTICS `NoSuch`", "NOSUCH"),
        ("ANALYZE TABLES COMPUTE STATISTICS FOR", "FOR"),
        ("ANALYZE TABLES COMPUTE STATISTICS ALL", "ALL"),
        ("ANALYZE TABLES COMPUTE STATISTICS select", "SELECT"),
        ("ANALYZE TABLES IN db COMPUTE STATISTICS foo", "FOO"),
        ("ANALYZE TABLE t COMPUTE STATISTICS MiXed", "MIXED"),
        ("ANALYZE TABLE t COMPUTE STATISTICS FOR", "FOR"),
        ("ANALYZE TABLE t COMPUTE STATISTICS ALL", "ALL"),
        (
            "ANALYZE TABLE t PARTITION (p=1, p=2) COMPUTE STATISTICS foo",
            "FOO",
        ),
    ] {
        let e = err(sql);
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidSqlSyntaxAnalyzeTableUnexpectedNoscan,
            "{sql}"
        );
        assert_eq!(
            e.error.message_parameters(),
            [("ctx", word.to_owned())],
            "{sql}"
        );
        assert_eq!(e.span.range(), 0..sql.len(), "{sql}");
    }
}

#[test]
fn analyze_syntax_errors_are_at_sparks_positions() {
    for (sql, near, hint) in [
        (
            "ANALYZE TABLES default COMPUTE STATISTICS",
            "'default'",
            ": extra input 'default'",
        ),
        (
            "ANALYZE TABLES IN COMPUTE STATISTICS",
            "'STATISTICS'",
            ": missing 'COMPUTE'",
        ),
        ("ANALYZE TABLES", "end of input", ""),
        ("ANALYZE TABLES IN db", "end of input", ""),
        (
            "ANALYZE TABLES IN db COMPUTE",
            "end of input",
            ": missing 'STATISTICS'",
        ),
        ("ANALYZE TABLES LIKE 'x' COMPUTE STATISTICS", "'LIKE'", ""),
        (
            "ANALYZE TABLES COMPUTE STATISTICS FOR COLUMNS a",
            "'COLUMNS'",
            "",
        ),
        (
            "ANALYZE TABLES COMPUTE STATISTICS 'noscan'",
            // Spark echoes the literal; an error here never does (ADR 0038).
            "a string literal",
            ": extra input a string literal",
        ),
        (
            "ANALYZE TABLES COMPUTE STATISTICS NOSCAN NOSCAN",
            "'NOSCAN'",
            ": extra input 'NOSCAN'",
        ),
        (
            "ANALYZE TABLE t COMPUTE STATISTICS FOR ALL",
            "end of input",
            ": missing 'COLUMNS'",
        ),
        (
            "ANALYZE TABLE t COMPUTE STATISTICS FOR x",
            "'x'",
            ": extra input 'x'",
        ),
        (
            "ANALYZE TABLE t COMPUTE STATISTICS NOSCAN x",
            "'x'",
            ": extra input 'x'",
        ),
    ] {
        let e = err(sql);
        assert_eq!(e.error.error_class(), ErrorClass::ParseSyntaxError, "{sql}");
        assert_eq!(
            e.error.message_parameters(),
            [("error", near.to_owned()), ("hint", hint.to_owned())],
            "{sql}"
        );
    }
}
