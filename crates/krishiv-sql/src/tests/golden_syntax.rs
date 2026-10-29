// SPDX-License-Identifier: Apache-2.0
//! The syntax-negative statements of the vendored Spark 4.2.0 goldens
//! (`tests/spark-golden/v4.2.0/results/*.sql.out`): every statement whose
//! expected output is a parse-time error (`PARSE_SYNTAX_ERROR`,
//! `PARSE_EMPTY_STATEMENT`, or an `UNSUPPORTED_*` class raised by the parser)
//! must fail through the public parse entry point with Spark's error class,
//! SQLSTATE, message parameters and position.
//!
//! Position is Spark's query context when the golden records one (a
//! `startIndex`/`stopIndex` pair, 1-based start as Spark writes it), and the
//! offending token otherwise: the span must cover the expected `error` token
//! (`end of input` is the empty span at the end of the statement).

use super::*;
use std::collections::{BTreeMap, HashMap};

/// A collected syntax-negative statement.
struct GoldenCase {
    file: String,
    index: usize,
    sql: String,
    error_class: String,
    sql_state: Option<String>,
    params: BTreeMap<String, String>,
    /// `(start, stop)` from the golden's query context, when it has exactly
    /// one entry. `start` is 1-based as Spark writes it; `stop` is exclusive.
    context: Option<(u64, u64)>,
}

/// Statements where this parser and Spark 4.2 disagree, with the reason.
/// Keyed by the SQL text (one entry covers every file that runs it).
const GOLDEN_DEVIATIONS: &[(&str, &str)] = &[
    (
        "SELECT 1 FROM \"not_exist\"",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "USE SCHEMA \"not_exist\"",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "ALTER TABLE \"not_exist\" ADD COLUMN not_exist int",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "ALTER TABLE not_exist ADD COLUMN \"not_exist\" int",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT 1 AS \"not_exist\" FROM not_exist",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT 1 FROM not_exist AS X(\"hello\")",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT \"not_exist\"()",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT \"not_exist\".not_exist()",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "select 1 from \"not_\"\"exists\"",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "CREATE SCHEMA \"myschema\"",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "CREATE TEMPORARY VIEW \"myview\"(\"c1\") AS\n  WITH \"v\"(\"a\") AS (SELECT 1) SELECT \"a\" FROM \"v\"",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT \"a1\" AS \"a2\" FROM \"myview\" AS \"atab\"(\"a1\")",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "DROP TABLE \"myview\"",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "DROP SCHEMA \"myschema\"",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT INTERVAL \"1\" YEAR",
        "with double-quoted identifiers Spark reports the identifier, this parser reports the unit as extra input",
    ),
    (
        "EXECUTE IMMEDIATE 'SELCT Fa'",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'SELCT Fa' INTO res_id",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'SELECT id, name FROM tbl_view WHERE id = ?' INTO (a, b) USING 10",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE sql_string",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'SELECT 1 JOIN SELECT 2' INTO v1",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'SELECT 1 VALUES (1)' INTO v1",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'SELECT 1 alias garbage garbage garbage' INTO v1",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'SELECT 1 WITH abc' INTO v1",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'SELECT t1.*, t2.* FROM IDENTIFIER(:t1) t1 JOIN IDENTIFIER(:t2) t2 USING (IDENTIFIER(:col)) ORDER BY ALL'\n  USING 'integration_test' AS t1, 'integration_test2' AS t2, 'c1' AS col",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'INSERT INTO integration_test(IDENTIFIER(:col1), IDENTIFIER(:col2)) VALUES (:val1, :val2)'\n  USING 'c1' AS col1, 'c2' AS col2, 3 AS val1, 'c' AS val2",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'SELECT IDENTIFIER(concat(:schema, ''.'', :table, ''.c1'')) FROM VALUES(named_struct(''c1'', 100)) AS IDENTIFIER(:alias)(IDENTIFIER(:schema ''.'' :table))'\n  USING 'identifier_clause_test_schema' AS schema, 'my_table' AS table, 't' AS alias",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'WITH IDENTIFIER(:cte_name)(c1) AS (VALUES(1)) SELECT c1 FROM IDENTIFIER(:cte_name)'\n  USING 'my_cte' AS cte_name",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'CREATE OR REPLACE TEMPORARY VIEW IDENTIFIER(:view_name)(IDENTIFIER(:col_name)) AS VALUES(1)'\n  USING 'test_view' AS view_name, 'test_col' AS col_name",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'ALTER TABLE IDENTIFIER(:tab) ADD COLUMN IDENTIFIER(:new_col) INT'\n  USING 'integration_test' AS tab, 'c4' AS new_col",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'ALTER TABLE IDENTIFIER(:tab) RENAME COLUMN IDENTIFIER(:old_col) TO IDENTIFIER(:new_col)'\n  USING 'integration_test' AS tab, 'c4' AS old_col, 'c5' AS new_col",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'SELECT map(:key, :val).IDENTIFIER(:key) AS result'\n  USING 'mykey' AS key, 42 AS val",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'SELECT IDENTIFIER(:alias ''.c1'') FROM integration_test AS IDENTIFIER(:alias) ORDER BY ALL'\n  USING 't' AS alias",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "EXECUTE IMMEDIATE 'SELECT 1 AS IDENTIFIER(:schema ''.'' :col)'\n  USING 'identifier_clause_test_schema' AS schema, 'col1' AS col",
        "`EXECUTE IMMEDIATE` is not a supported statement: the error is at `EXECUTE`, not inside the immediate string",
    ),
    (
        "CREATE FUNCTION foo2a0() RETURNS TABLE() RETURN SELECT 1",
        "`CREATE FUNCTION` is not a supported statement: the error is at `FUNCTION`, not at the table return type",
    ),
    (
        "CREATE FUNCTION foo2c1() RETURNS TABLE(c1 INT DEFAULT 5) RETURN SELECT 1, 2",
        "`CREATE FUNCTION` is not a supported statement: the error is at `FUNCTION`, not at the default",
    ),
    (
        "CREATE TEMPORARY FUNCTION test_udf(IDENTIFIER('param1') INT, IDENTIFIER('param2') STRING)\nRETURNS INT\nRETURN IDENTIFIER('param1') + length(IDENTIFIER('param2'))",
        "`CREATE FUNCTION` is not a supported statement: the error is at `FUNCTION`, not at the parameter list",
    ),
    (
        "CREATE TEMPORARY FUNCTION test_table_udf(IDENTIFIER('input_val') INT)\nRETURNS TABLE(IDENTIFIER('col1') INT, IDENTIFIER('col2') STRING)\nRETURN SELECT IDENTIFIER('input_val'), 'result'",
        "`CREATE FUNCTION` is not a supported statement: the error is at `FUNCTION`, not at the parameter list",
    ),
    (
        "SELECT CASE WHEN foo3_1b(rand(0) * 0 < 1 THEN 1 ELSE -1 END",
        "the `CASE WHEN` function-argument recovery reports `THEN`, Spark reports the function name",
    ),
    (
        "SELECT :IDENTIFIER('param1') FROM VALUES(1) AS T(c1)",
        "parameter markers followed by `IDENTIFIER` are not supported: the error is at `:`, Spark echoes the string text",
    ),
    (
        "SELECT * FROM s.IDENTIFIER('tab')",
        "with the legacy identifier clause the dotted `IDENTIFIER` call is rejected at `.`, Spark reports the `(`",
    ),
    (
        "SELECT * FROM IDENTIFIER('s').IDENTIFIER('tab')",
        "with the legacy identifier clause the chained `IDENTIFIER` call is rejected at `.`, Spark reports `IDENTIFIER`",
    ),
    (
        "SELECT * FROM IDENTIFIER('s').tab",
        "with the legacy identifier clause the chained `IDENTIFIER` call is rejected at `.`, Spark reports `IDENTIFIER`",
    ),
    (
        "SELECT row_number() OVER win FROM VALUES(1) AS T(c1) WINDOW IDENTIFIER('win') AS (ORDER BY c1)",
        "with the legacy identifier clause the `WINDOW IDENTIFIER` name is rejected at `(`, Spark reports a missing `AS`",
    ),
    (
        "INSERT INTO tab(IDENTIFIER('c1')) VALUES(1)",
        "with the legacy identifier clause the `INSERT` column is rejected without Spark's missing `)` hint",
    ),
    (
        "SELECT * FROM VALUES (1, 2) AS IDENTIFIER('schema.table')(c1, c2)",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT row_number() OVER IDENTIFIER('x.win') FROM VALUES(1) AS T(c1) WINDOW win AS (ORDER BY c1)",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT map('a', 1).IDENTIFIER('a') FROM VALUES(1) AS T(c1)",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT named_struct('a', 1).IDENTIFIER('a') FROM VALUES(1) AS T(c1)",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT row_number() OVER IDENTIFIER('win') FROM VALUES(1) AS T(c1) WINDOW win AS (ORDER BY c1)",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT my_table.* FROM VALUES (1, 2) AS IDENTIFIER('my_table')(IDENTIFIER('c1'), IDENTIFIER('c2'))",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "WITH identifier('v')(identifier('c1')) AS (VALUES(1)) (SELECT c1 FROM v)",
        "the text of a string literal is never echoed in an error",
    ),
    (
        "SELECT EXTRACT(IDENTIFIER('YEAR') FROM DATE'2024-01-15')",
        "the default-mode case matches (`extract_identifier_reports_the_missing_paren`); only the `identifier-clause-legacy` case [234] still differs",
    ),
    (
        "DESC FORMATTED t a AS JSON",
        "`DESCRIBE ... AS JSON` without `EXTENDED` is `UNSUPPORTED_FEATURE.DESC_TABLE_COLUMN_JSON` at a different position here",
    ),
    (
        "select 3 as x, 4 as y\n|> aggregate group by all",
        "the pipe `GROUP BY ALL` unsupported case is reported at the whole statement in Spark, at the aggregate here",
    ),
    (
        "table courseSales\n|> aggregate sum(earnings) group by rollup(course, `year`)\n|> where course = 'dotNET' and `year` = '2013'",
        "the pipe `GROUP BY ROLLUP` unsupported case is reported at the whole statement in Spark, at the aggregate here",
    ),
    (
        "table courseSales\n|> aggregate sum(earnings) group by cube(course, `year`)\n|> where course = 'dotNET' and `year` = '2013'",
        "the pipe `GROUP BY CUBE` unsupported case is reported at the whole statement in Spark, at the aggregate here",
    ),
    (
        "table courseSales\n|> aggregate sum(earnings) group by course, `year` grouping sets(course, `year`)\n|> where course = 'dotNET' and `year` = '2013'",
        "the pipe `GROUPING SETS` unsupported case is reported at the whole statement in Spark, at the aggregate here",
    ),
    (
        "table courseSales\n|> aggregate sum(earnings), grouping(course) + 1\n   group by course\n|> where course = 'dotNET' and `year` = '2013'",
        "the pipe `grouping` unsupported case is reported at the whole statement in Spark, at the aggregate here",
    ),
    (
        "table courseSales\n|> aggregate sum(earnings), grouping_id(course)\n   group by course\n|> where course = 'dotNET' and `year` = '2013'",
        "the pipe `grouping_id` unsupported case is reported at the whole statement in Spark, at the aggregate here",
    ),
    (
        "table other\n|> aggregate sum(a) over () group by b",
        "the pipe `OVER` with `GROUP BY` unsupported case is reported at the whole statement in Spark, at the aggregate here",
    ),
    (
        "select rank() OVER (PARTITION BY four, ORDER BY ten) FROM tenk1",
        "in a window `PARTITION BY` list Spark reports the second `BY` as extra input, this parser reports the `ORDER`",
    ),
];

/// Whether `class` is a parse-time error this test collects: a syntax error,
/// an empty statement, or an `UNSUPPORTED_*` class raised by the parser
/// (a `ParseException` in the golden).
fn is_collected(exception: &str, class: &str) -> bool {
    if !exception.ends_with(".ParseException") {
        return false;
    }
    class == "PARSE_SYNTAX_ERROR"
        || class == "PARSE_EMPTY_STATEMENT"
        || class.starts_with("UNSUPPORTED_")
}

/// Parse the `--SET` / `--CONFIG_DIM` directives of an input file: the folder
/// defaults plus `--SET`, plus one entry per `--CONFIG_DIM` combination.
fn file_sessions(name: &str, input: &str) -> Vec<Vec<(String, String)>> {
    let kind_configs = |name: &str| {
        let mut configs = vec![(
            "spark.sql.session.timeZone".to_owned(),
            "America/Los_Angeles".to_owned(),
        )];
        if name.starts_with("nonansi/") {
            configs.push(("spark.sql.ansi.enabled".to_owned(), "false".to_owned()));
        } else if name.starts_with("postgreSQL/") {
            configs.push(("spark.sql.ansi.enabled".to_owned(), "true".to_owned()));
            configs.push((
                "spark.sql.legacy.interval.enabled".to_owned(),
                "true".to_owned(),
            ));
        } else if name.starts_with("timestampNTZ/") {
            configs.push((
                "spark.sql.timestampType".to_owned(),
                "TIMESTAMP_NTZ".to_owned(),
            ));
        } else {
            configs.push(("spark.sql.ansi.enabled".to_owned(), "true".to_owned()));
        }
        configs
    };
    let parse_pairs = |s: &str| {
        s.split(',')
            .filter_map(|kv| kv.split_once('='))
            .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
            .collect::<Vec<_>>()
    };
    let mut set = Vec::new();
    let mut dims: BTreeMap<String, Vec<Vec<(String, String)>>> = BTreeMap::new();
    for line in input.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("--SET ") {
            set.extend(parse_pairs(rest));
        } else if let Some(rest) = line.strip_prefix("--CONFIG_DIM") {
            let (dim, value) = rest.split_once(' ').unwrap_or((rest, ""));
            dims.entry(dim.to_owned())
                .or_default()
                .push(parse_pairs(value));
        }
    }
    let mut combos: Vec<Vec<(String, String)>> = vec![Vec::new()];
    for dim in dims.into_values() {
        combos = combos
            .iter()
            .flat_map(|c| {
                dim.iter().map(|alt| {
                    let mut next = c.clone();
                    next.extend(alt.iter().cloned());
                    next
                })
            })
            .collect();
    }
    combos
        .into_iter()
        .map(|combo| {
            let mut configs = kind_configs(name);
            configs.extend(set.iter().cloned());
            configs.extend(combo);
            configs
        })
        .collect()
}

/// Whether `sql` is a `SET` / `RESET` statement; when so, apply it to `conf`.
fn apply_set(sql: &str, conf: &mut HashMap<String, String>) -> bool {
    let trimmed = sql.trim().trim_end_matches(';').trim();
    let word = trimmed.split_whitespace().next().unwrap_or_default();
    if word.eq_ignore_ascii_case("RESET") {
        if let Some(key) = trimmed.get(5..).map(str::trim).filter(|k| !k.is_empty()) {
            conf.remove(key);
        } else {
            conf.clear();
        }
        return true;
    }
    if !word.eq_ignore_ascii_case("SET") {
        return false;
    }
    if let Some(rest) = trimmed.get(3..)
        && let Some((k, v)) = rest.split_once('=')
    {
        conf.insert(k.trim().to_owned(), v.trim().to_owned());
    }
    true
}

/// The blocks of a `.sql.out` result file: `(query, schema, output)`.
fn result_blocks(text: &str) -> Vec<(String, String, String)> {
    let mut blocks = Vec::new();
    let mut lines = text.lines().peekable();
    let mut more = false;
    for line in lines.by_ref() {
        if line == "-- !query" {
            more = true;
            break;
        }
    }
    while more {
        let mut query = Vec::new();
        for line in lines.by_ref() {
            if line == "-- !query schema" {
                break;
            }
            query.push(line);
        }
        let mut schema = Vec::new();
        for line in lines.by_ref() {
            if line == "-- !query output" {
                break;
            }
            schema.push(line);
        }
        let mut output = Vec::new();
        more = false;
        for line in lines.by_ref() {
            if line == "-- !query" {
                more = true;
                break;
            }
            output.push(line);
        }
        blocks.push((query.join("\n"), schema.join("\n"), output.join("\n")));
    }
    blocks
        .into_iter()
        .map(|(q, s, o)| (q.trim().to_owned(), s.trim().to_owned(), o))
        .collect()
}

/// Every syntax-negative statement of the vendored Spark 4.2.0 goldens must
/// fail through [`parse_statement`] with Spark's error class, SQLSTATE,
/// message parameters and position. The failing message names the file and
/// the statement index.
#[test]
fn golden_syntax_negatives_match_spark() {
    let results_dir = format!(
        "{}/../../tests/spark-golden/v4.2.0/results",
        env!("CARGO_MANIFEST_DIR")
    );
    let inputs_dir = format!(
        "{}/../../tests/spark-golden/v4.2.0/inputs",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut cases: Vec<GoldenCase> = Vec::new();
    let mut inputs: HashMap<String, String> = HashMap::new();
    fn collect_out_files(dir: &str, prefix: &str, out: &mut Vec<String>) {
        let mut entries: Vec<String> = std::fs::read_dir(format!("{dir}/{prefix}"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_owned())
            .collect();
        entries.sort();
        for name in entries {
            let rel = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            let full = format!("{dir}/{rel}");
            if std::fs::metadata(&full).unwrap().is_dir() {
                collect_out_files(dir, &rel, out);
            } else if rel.ends_with(".sql.out") {
                out.push(rel);
            }
        }
    }
    let mut names: Vec<String> = Vec::new();
    collect_out_files(&results_dir, "", &mut names);
    names.sort();
    for name in &names {
        let text = std::fs::read_to_string(format!("{results_dir}/{name}")).unwrap();
        let short = name.strip_suffix(".sql.out").unwrap().to_owned();
        let input = inputs
            .entry(short.clone())
            .or_insert_with(|| {
                std::fs::read_to_string(format!("{inputs_dir}/{short}.sql")).unwrap_or_default()
            })
            .clone();
        let _ = &input;
        for (index, (query, schema, output)) in result_blocks(&text).into_iter().enumerate() {
            if schema != "struct<>" {
                continue;
            }
            let mut output_lines = output.lines();
            let exception = output_lines.next().unwrap_or_default().trim().to_owned();
            if !exception.contains('.')
                || !(exception.ends_with("Exception") || exception.ends_with("Error"))
            {
                continue;
            }
            let body_text = output.trim_start_matches(&exception).trim();
            let body_text = body_text.trim_end_matches(|c: char| c.is_whitespace());
            let Ok(body) = serde_json::from_str::<serde_json::Value>(body_text) else {
                continue;
            };
            let class = body
                .get("errorClass")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned();
            if !is_collected(&exception, &class) {
                continue;
            }
            let sql_state = body
                .get("sqlState")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
            let mut params = BTreeMap::new();
            if let Some(map) = body.get("messageParameters").and_then(|v| v.as_object()) {
                for (k, v) in map {
                    params.insert(k.clone(), v.as_str().unwrap_or(&v.to_string()).to_owned());
                }
            }
            let context = body
                .get("queryContext")
                .and_then(|v| v.as_array())
                .and_then(|ctx| {
                    if ctx.len() == 1 {
                        let start = ctx[0]
                            .get("startIndex")
                            .and_then(serde_json::Value::as_u64)?;
                        let stop = ctx[0]
                            .get("stopIndex")
                            .and_then(serde_json::Value::as_u64)?;
                        Some((start, stop))
                    } else {
                        None
                    }
                });
            cases.push(GoldenCase {
                file: short.clone(),
                index,
                sql: query,
                error_class: class,
                sql_state,
                params,
                context,
            });
        }
    }
    assert!(
        cases.len() == 175,
        "syntax-negative case count changed: {}",
        cases.len()
    );
    let mut mismatches: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for case in &cases {
        let input =
            std::fs::read_to_string(format!("{inputs_dir}/{}.sql", case.file)).unwrap_or_default();
        let sessions = file_sessions(&case.file, &input);
        // The statements before this one in file order, to replay `SET`s.
        let ordered: Vec<String> = {
            let text =
                std::fs::read_to_string(format!("{results_dir}/{}.sql.out", case.file)).unwrap();
            result_blocks(&text)
                .into_iter()
                .map(|(q, _, _)| q)
                .collect()
        };
        let mut ok = true;
        for session in &sessions {
            let mut conf: HashMap<String, String> = session.iter().cloned().collect();
            for prev in ordered.iter().take(case.index) {
                apply_set(prev, &mut conf);
            }
            if apply_set(&case.sql, &mut conf) {
                continue;
            }
            let options = ParserOptions::from_conf(|k| conf.get(k).map(String::as_str));
            match parse_statement(&case.sql, &options) {
                Ok(_) => {
                    ok = false;
                    break;
                }
                Err(ours) => {
                    let ours_params: BTreeMap<String, String> = ours
                        .error
                        .message_parameters()
                        .iter()
                        .map(|(k, v)| ((*k).to_owned(), v.clone()))
                        .collect();
                    let want_params: BTreeMap<String, String> = case.params.clone();
                    let class_ok = ours.error.error_class().name() == case.error_class;
                    let state_ok = case
                        .sql_state
                        .as_ref()
                        .is_none_or(|st| ours.error.sql_state() == Some(st.as_str()));
                    let params_ok = ours_params == want_params;
                    let position_ok = match case.context {
                        Some((start, stop)) => {
                            let at = case
                                .sql
                                .get(..ours.span.start)
                                .map_or(0, |p| p.chars().count() as u64);
                            let end = case
                                .sql
                                .get(..ours.span.end)
                                .map_or(0, |p| p.chars().count() as u64);
                            at + 1 == start && end == stop
                        }
                        None => {
                            let want_error =
                                want_params.get("error").map(String::as_str).unwrap_or("");
                            if want_error == "end of input" {
                                ours.span.start == case.sql.len() && ours.span.end == case.sql.len()
                            } else {
                                let stripped = want_error
                                    .strip_prefix('\'')
                                    .and_then(|s| s.strip_suffix('\''))
                                    .unwrap_or(want_error);
                                case.sql
                                    .get(ours.span.range())
                                    .is_some_and(|got| got == stripped)
                            }
                        }
                    };
                    if !(class_ok && state_ok && params_ok && position_ok) {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if !ok {
            mismatches
                .entry(case.sql.clone())
                .or_default()
                .push(format!("{} [{}]", case.file, case.index));
        }
    }
    let mut known: Vec<&str> = GOLDEN_DEVIATIONS.iter().map(|(sql, _)| *sql).collect();
    known.sort_unstable();
    let mut found: Vec<&str> = mismatches.keys().map(String::as_str).collect();
    found.sort_unstable();
    let mut detail = String::new();
    for (sql, locations) in &mismatches {
        if !known.contains(&sql.as_str()) {
            detail.push_str(&format!("NEW MISMATCH {}: {sql:?}\n", locations.join(", ")));
        }
    }
    assert_eq!(found, known, "golden syntax deviations changed:\n{detail}");
}

/// `GROUP BY WITH ROLLUP` / `WITH CUBE` with no grouping expressions reports
/// the `ROLLUP` / `CUBE` as extra input, not the `WITH`.
#[test]
fn group_by_with_reports_the_rollup_cube() {
    let d = ParserOptions::default();
    for (sql, token, span) in [
        (
            "SELECT a, b, c, count(d) FROM grouping GROUP BY WITH ROLLUP",
            "'ROLLUP'",
            53..59,
        ),
        (
            "SELECT a, b, c, count(d) FROM grouping GROUP BY WITH CUBE",
            "'CUBE'",
            53..57,
        ),
    ] {
        let e = parse_statement(sql, &d).unwrap_err();
        assert_eq!(e.error.error_class().name(), "PARSE_SYNTAX_ERROR", "{sql}");
        assert_eq!(
            e.error.message_parameters(),
            [
                ("error", token.to_owned()),
                ("hint", format!(": extra input {token}"))
            ],
            "{sql}"
        );
        assert_eq!(e.span.range(), span, "{sql}");
    }
}

/// `FILTER (expr)` without `WHERE` reports the opening `(`.
#[test]
fn filter_without_where_reports_the_paren() {
    let d = ParserOptions::default();
    let sql = "SELECT sum(c1) FILTER (c1 != var1) FROM VALUES(1, 2), (2, 3) AS T(c1, c2)";
    let e = parse_statement(sql, &d).unwrap_err();
    assert_eq!(e.error.error_class().name(), "PARSE_SYNTAX_ERROR");
    assert_eq!(
        e.error.message_parameters(),
        [("error", "'('".to_owned()), ("hint", String::new())]
    );
    assert_eq!(e.span.range(), 22..23);
}

/// The higher-order function `filter(array, f)` is no aggregate `FILTER`: a
/// failure inside its arguments keeps its own position.
#[test]
fn higher_order_filter_keeps_its_error_position() {
    let d = ParserOptions::default();
    let sql = "SELECT filter(a, x -> x > 1 2) FROM t";
    let e = parse_statement(sql, &d).unwrap_err();
    assert_eq!(e.error.error_class().name(), "PARSE_SYNTAX_ERROR");
    assert_eq!(e.error.message_parameters()[0], ("error", "'2'".to_owned()));
    let at = sql.find('2').unwrap();
    assert_eq!(e.span.range(), at..at + 1);
}

/// `EXTRACT(IDENTIFIER('...') FROM ...)` reports `FROM` with the missing `)`.
#[test]
fn extract_identifier_reports_the_missing_paren() {
    let d = ParserOptions::default();
    let sql = "SELECT EXTRACT(IDENTIFIER('YEAR') FROM DATE'2024-01-15')";
    let e = parse_statement(sql, &d).unwrap_err();
    assert_eq!(e.error.error_class().name(), "PARSE_SYNTAX_ERROR");
    assert_eq!(
        e.error.message_parameters(),
        [
            ("error", "'FROM'".to_owned()),
            ("hint", ": missing ')'".to_owned())
        ]
    );
    assert_eq!(e.span.range(), 34..38);
}

/// `TABLE t |> EXTEND DISTINCT x AS z` reports the `AS`.
#[test]
fn pipe_extend_distinct_reports_the_as() {
    let d = ParserOptions::default();
    let sql = "table t\n|> extend distinct x as z";
    let e = parse_statement(sql, &d).unwrap_err();
    assert_eq!(e.error.error_class().name(), "PARSE_SYNTAX_ERROR");
    assert_eq!(
        e.error.message_parameters(),
        [("error", "'as'".to_owned()), ("hint", String::new())]
    );
    assert_eq!(e.span.range(), 29..31);
}
