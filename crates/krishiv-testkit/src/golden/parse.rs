// SPDX-License-Identifier: Apache-2.0
//! Parser for Spark `SQLQueryTestSuite` result files (`*.sql.out`).
//!
//! A result file is a header line followed by blocks of
//! `-- !query` / `<sql>` / `-- !query schema` / `<schema>` / `-- !query output` / `<output>`.
//! The output of a failed statement is the exception class name followed by Spark 4's JSON
//! error (`errorClass`, `sqlState`, `messageParameters`, ...), or plain text for older errors.

use serde_json::Value;

/// A malformed result file.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    /// A `-- !query` block lacks its schema or output marker.
    #[error("block {block}: missing `{marker}` marker")]
    MissingMarker {
        /// 1-based block number.
        block: usize,
        /// The marker that was expected.
        marker: &'static str,
    },
    /// Text before the first `-- !query` other than the generated-file header.
    #[error("unexpected text before the first block: `{0}`")]
    Preamble(String),
}

/// The Spark error of a failed statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedError {
    /// JVM exception class, e.g. `org.apache.spark.sql.AnalysisException`.
    pub exception: String,
    /// `errorClass` of the JSON body (`None` for plain-text errors).
    pub error_class: Option<String>,
    /// `sqlState` of the JSON body.
    pub sql_state: Option<String>,
    /// The body after the exception line (JSON or plain text), trimmed.
    pub body: String,
}

impl ExpectedError {
    /// Whether this is a SQL parser error (no query could be planned).
    #[must_use]
    pub fn is_parse_error(&self) -> bool {
        self.exception.ends_with(".ParseException")
    }
}

/// What Spark produced for a statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expected {
    /// Success; `output` is the tab-separated rows (already sorted by Spark when unordered).
    Rows(String),
    /// Failure.
    Error(ExpectedError),
}

/// One statement block of a result file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Statement {
    /// The SQL text, trimmed.
    pub query: String,
    /// Spark's `struct<...>` schema string.
    pub schema: String,
    /// The recorded outcome.
    pub expected: Expected,
}

impl Statement {
    /// Text fallback for "is the result order significant", used only when the engine does not
    /// report [`super::run::QueryOutput::sorted`]. Spark decides on the analyzed plan
    /// (`isSemanticallySorted`); this approximation says "sorted" for a leading `DESC`/`DESCRIBE`
    /// and for an `ORDER BY` at parenthesis depth 0, after dropping comments, string literals and
    /// quoted identifiers. An `ORDER BY` inside `OVER (...)` does not count, as in Spark. Known
    /// divergence: Spark also finds a sort in a subquery unless a join, aggregate, distinct,
    /// generate or sample sits above it; this approximation ignores every subquery.
    #[must_use]
    pub fn text_sorted(&self) -> bool {
        text_sorted(&self.query)
    }
}

/// See [`Statement::text_sorted`].
#[must_use]
pub fn text_sorted(query: &str) -> bool {
    let mut top = String::with_capacity(query.len());
    let mut depth = 0_usize;
    let mut chars = query.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' | '"' | '`' => {
                while let Some(n) = chars.next() {
                    if n == '\\' {
                        chars.next();
                    } else if n == c {
                        break;
                    }
                }
                top.push(' ');
            }
            '-' if chars.peek() == Some(&'-') => {
                for n in chars.by_ref() {
                    if n == '\n' {
                        break;
                    }
                }
                top.push(' ');
            }
            '/' if chars.peek() == Some(&'*') => {
                let mut prev = ' ';
                for n in chars.by_ref() {
                    if prev == '*' && n == '/' {
                        break;
                    }
                    prev = n;
                }
                top.push(' ');
            }
            '(' => {
                depth += 1;
                top.push(' ');
            }
            ')' => {
                depth = depth.saturating_sub(1);
                top.push(' ');
            }
            _ if depth == 0 => top.push(c.to_ascii_lowercase()),
            _ => {}
        }
    }
    let mut words = top
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .peekable();
    if matches!(words.peek(), Some(&"desc" | &"describe")) {
        return true;
    }
    while let Some(w) = words.next() {
        if w == "order" && words.peek() == Some(&"by") {
            return true;
        }
    }
    false
}

const QUERY: &str = "-- !query";
const SCHEMA: &str = "-- !query schema";
const OUTPUT: &str = "-- !query output";

/// Parse the text of a `.sql.out` file into its statements.
pub fn parse_results(text: &str) -> Result<Vec<Statement>, ParseError> {
    let mut statements = Vec::new();
    let mut lines = text.lines().peekable();
    let mut more = false;
    for line in lines.by_ref() {
        if line == QUERY {
            more = true;
            break;
        }
        if !line.trim().is_empty() && !line.starts_with("-- Automatically generated") {
            return Err(ParseError::Preamble(line.to_owned()));
        }
    }
    while more {
        let block = statements.len() + 1;
        let query = take_until(&mut lines, SCHEMA).ok_or(ParseError::MissingMarker {
            block,
            marker: SCHEMA,
        })?;
        let schema = take_until(&mut lines, OUTPUT).ok_or(ParseError::MissingMarker {
            block,
            marker: OUTPUT,
        })?;
        let mut output = Vec::new();
        more = false;
        for line in lines.by_ref() {
            if line == QUERY {
                more = true;
                break;
            }
            output.push(line);
        }
        let output = output.join("\n");
        statements.push(Statement {
            query: query.trim().to_owned(),
            schema: schema.trim().to_owned(),
            // Java `\s` (ASCII) trimming, as Spark: a trailing U+00A0 or
            // U+3000 is part of the row.
            expected: classify(
                schema.trim(),
                output.trim_end_matches(crate::golden::normalise::JAVA_SPACE),
            ),
        });
    }
    Ok(statements)
}

fn take_until<'a>(lines: &mut impl Iterator<Item = &'a str>, marker: &str) -> Option<String> {
    let mut taken = Vec::new();
    for line in lines {
        if line == marker {
            return Some(taken.join("\n"));
        }
        taken.push(line);
    }
    None
}

fn classify(schema: &str, output: &str) -> Expected {
    let (first, rest) = output.split_once('\n').unwrap_or((output, ""));
    if schema != "struct<>" || !is_exception_name(first) {
        return Expected::Rows(output.to_owned());
    }
    let body = rest.trim().to_owned();
    let json: Option<Value> = body
        .starts_with('{')
        .then(|| serde_json::from_str(&body).ok())
        .flatten();
    let field = |key: &str| {
        json.as_ref()
            .and_then(|j| j.get(key))
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    Expected::Error(ExpectedError {
        exception: first.to_owned(),
        error_class: field("errorClass"),
        sql_state: field("sqlState"),
        body,
    })
}

/// A fully-qualified JVM exception or error class name.
fn is_exception_name(line: &str) -> bool {
    line.contains('.')
        && (line.ends_with("Exception") || line.ends_with("Error"))
        && line
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '$')
        && line.starts_with(|c: char| c.is_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    type R = Result<(), Box<dyn std::error::Error>>;

    const SAMPLE: &str = "-- Automatically generated by SQLQueryTestSuite\n\
-- !query\ncreate view v as select 1 as a\n-- !query schema\nstruct<>\n-- !query output\n\n\n\n\
-- !query\nselect * from v\n-- !query schema\nstruct<a:int>\n-- !query output\n1\n\n\n\
-- !query\nselect * from\n-- !query schema\nstruct<>\n-- !query output\n\
org.apache.spark.sql.catalyst.parser.ParseException\n{\n  \"errorClass\" : \"PARSE_SYNTAX_ERROR\",\n  \"sqlState\" : \"42601\"\n}\n";

    #[test]
    fn parses_blocks_schema_rows_and_errors() -> R {
        let s = parse_results(SAMPLE)?;
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].expected, Expected::Rows(String::new()));
        assert_eq!(s[1].schema, "struct<a:int>");
        assert_eq!(s[1].expected, Expected::Rows("1".into()));
        let Expected::Error(e) = &s[2].expected else {
            return Err("expected an error block".into());
        };
        assert_eq!(e.error_class.as_deref(), Some("PARSE_SYNTAX_ERROR"));
        assert_eq!(e.sql_state.as_deref(), Some("42601"));
        assert!(e.is_parse_error());
        Ok(())
    }

    #[test]
    fn multi_line_query_and_ordering() -> R {
        let s = parse_results(
            "-- !query\nselect a\nfrom t\nORDER  BY a\n-- !query schema\nstruct<a:int>\n-- !query output\n1\n",
        )?;
        assert_eq!(s[0].query, "select a\nfrom t\nORDER  BY a");
        assert!(s[0].text_sorted());
        Ok(())
    }

    #[test]
    fn text_fallback_counts_only_top_level_order_by() {
        assert!(text_sorted("select * from t order by a"));
        assert!(text_sorted("DESCRIBE EXTENDED t"));
        assert!(text_sorted(
            "with c as (select 1) select * from c -- x\n order by 1"
        ));
        assert!(!text_sorted(
            "select a, row_number() over (order by a) from t"
        ));
        assert!(!text_sorted(
            "select count(*) from (select * from t order by a)"
        ));
        assert!(!text_sorted("select 'order by' as x, `order by` from t"));
        assert!(!text_sorted("select /* order by */ 1"));
    }

    #[test]
    fn missing_marker_and_preamble_are_errors() {
        assert_eq!(
            parse_results("-- !query\nselect 1\n"),
            Err(ParseError::MissingMarker {
                block: 1,
                marker: SCHEMA
            })
        );
        assert!(matches!(
            parse_results("junk\n-- !query\n"),
            Err(ParseError::Preamble(_))
        ));
    }

    #[test]
    fn empty_file_has_no_statements() -> R {
        assert!(parse_results("-- Automatically generated by SQLQueryTestSuite\n")?.is_empty());
        Ok(())
    }
}
