// SPDX-License-Identifier: Apache-2.0
//! `TABLESAMPLE` against `tests/fixtures/spark-4.2.0/tablesample.json` (a
//! real Spark 4.2.0 run, generator `tests/scripts/gen_tablesample.py`): the
//! `Sample` node each case parses to, the error rows (class, message and
//! position) and the parser quirks.

use super::*;

fn fixture() -> serde_json::Value {
    let path = format!(
        "{}/../../tests/fixtures/spark-4.2.0/tablesample.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The lowered plan of `sql`, printed.
fn plan(sql: &str) -> String {
    match parse_and_lower(sql, &ParserOptions::default(), krishiv_plan::SourceId(0)) {
        Ok(p) => p.root().to_string(),
        Err(e) => panic!("{sql}: {}", e.error),
    }
}

/// Spark's `Sample` line without its tree prefix, resolution quote and the
/// 4.2 method (`+- 'Sample 0.0, 0.1, false, 7, Bernoulli` is
/// `Sample 0.0, 0.1, false, 7`).
fn spark_sample(line: &str) -> &str {
    let line = line.trim_start_matches("+- ").trim_start_matches('\'');
    line.strip_suffix(", Bernoulli").unwrap_or(line)
}

#[test]
fn cases_lower_to_spark_sample() {
    let doc = fixture();
    for case in doc["cases"].as_array().unwrap() {
        let sql = case["sql"].as_str().unwrap();
        let want = spark_sample(case["parsed_sample"].as_str().unwrap());
        let got = plan(sql);
        assert!(
            got.lines()
                .any(|l| l.trim_start_matches([' ', '+', '-', ':']) == want),
            "{sql}: want {want} in\n{got}"
        );
    }
}

#[test]
fn errors_match_spark() {
    let doc = fixture();
    for row in doc["errors"].as_array().unwrap() {
        let sql = row["sql"].as_str().unwrap();
        let e = err(sql, &ParserOptions::default());
        let message = row["message"].as_str().unwrap();
        let Some(class) = row["condition"].as_str() else {
            // Spark's unclassified `NumberFormatException` for a seed past
            // `Long`: refused as unsupported, naming the seed.
            assert_eq!(e.error.error_class(), ErrorClass::KrishivUnsupportedFeature);
            assert!(
                e.error.to_string().contains("99999999999999999999"),
                "{sql}"
            );
            continue;
        };
        assert_eq!(e.error.error_class().name(), class, "{sql}");
        let text = e.error.to_string();
        if let Some(rest) = message.strip_prefix("\n[PARSE_SYNTAX_ERROR] ") {
            // `Syntax error at or near 'x'. SQLSTATE: 42601 (line 1, pos N)`.
            let (what, tail) = rest.split_once(". SQLSTATE").unwrap();
            assert!(text.contains(what), "{sql}: {text} lacks {what}");
            let pos: usize = tail
                .split("pos ")
                .nth(1)
                .unwrap()
                .split(')')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(e.span.range().start, pos, "{sql}");
        } else {
            // The first message line, then the context block's position
            // (1-based) and carets.
            let first = message.trim_start_matches('\n').lines().next().unwrap();
            let first = first
                .split_once("] ")
                .map_or(first, |(_, m)| m)
                .trim_end_matches(" SQLSTATE: 0A000");
            assert!(text.contains(first), "{sql}: {text} lacks {first}");
            let position: usize = message
                .split("position ")
                .nth(1)
                .unwrap()
                .split(')')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            let carets = message.matches('^').count();
            assert_eq!(e.span.range().start + 1, position, "{sql}");
            assert_eq!(e.span.range().len(), carets, "{sql}");
        }
    }
}

#[test]
fn quirks_parse_as_spark() {
    // `REPEATABLE (abc)` is a table alias with a column alias, unseeded.
    let p = plan("SELECT * FROM t TABLESAMPLE (10 PERCENT) REPEATABLE (abc)");
    assert!(p.contains("Sample 0.0, 0.1, false\n"), "{p}");
    assert!(p.contains("SubqueryAlias REPEATABLE"), "{p}");
    // `ROWS` is a limit, its seed ignored.
    for sql in [
        "SELECT * FROM t TABLESAMPLE (10 ROWS) REPEATABLE (3)",
        "SELECT * FROM t TABLESAMPLE (10 ROWS)",
    ] {
        let p = plan(sql);
        assert!(p.contains("Limit 10") && !p.contains("Sample"), "{p}");
    }
    let p = plan("SELECT * FROM t TABLESAMPLE (BUCKET 0 OUT OF 10)");
    assert!(p.contains("Sample 0.0, 0.0, false\n"), "{p}");
    // The quirk rows the fixture pins are all here.
    assert_eq!(fixture()["quirks"].as_array().unwrap().len(), 6);
}
