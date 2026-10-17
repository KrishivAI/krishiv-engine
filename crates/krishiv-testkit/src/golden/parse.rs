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
    /// Whether result order is significant: Spark keeps row order only for sorted queries and
    /// sorts both sides otherwise. A textual `ORDER BY` stands in for Spark's plan check.
    #[must_use]
    pub fn is_ordered(&self) -> bool {
        let q = self.query.to_ascii_lowercase();
        let mut words = q
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .filter(|w| !w.is_empty());
        while let Some(w) = words.next() {
            if w == "order" && words.next() == Some("by") {
                return true;
            }
        }
        false
    }
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
            expected: classify(schema.trim(), output.trim_end()),
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
        assert!(s[0].is_ordered());
        Ok(())
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
