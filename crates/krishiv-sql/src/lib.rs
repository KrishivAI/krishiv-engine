// SPDX-License-Identifier: Apache-2.0
//! Spark SQL parsing (layer L3).
//!
//! [`parse_query`] parses a Spark 4.2 query statement (`SELECT`, `WITH`,
//! `VALUES`, `TABLE`, `FROM ... SELECT`, set operations) into the syntax tree
//! of [`ast`], whose nodes carry source spans; [`parse_statement`] also
//! accepts `EXPLAIN`. [`parse_data_type`], [`parse_table_schema`] and
//! [`parse_ddl_type`] parse type strings with the same grammar. The parser is hand-written:
//! a lexer ([`lexer`]) and a recursive-descent parser with precedence
//! climbing for expressions. Errors are Spark's error classes
//! ([`ParseError`]).
//!
//! ```
//! use krishiv_sql::{parse_query, ParserOptions};
//!
//! let sql = "SELECT a, count(*) FROM t GROUP BY a";
//! let q = parse_query(sql, &ParserOptions::default()).unwrap();
//! assert_eq!(q.span.text(sql), sql);
//!
//! let e = parse_query("SELECT 1 +", &ParserOptions::default()).unwrap_err();
//! assert_eq!(
//!     e.error.to_string(),
//!     "[PARSE_SYNTAX_ERROR] Syntax error at or near end of input. SQLSTATE: 42601"
//! );
//! ```

#![forbid(unsafe_code)]

pub mod ast;
mod error;
mod frontend;
mod generated {
    pub(crate) mod letters;
}
mod keywords;
pub mod lexer;
mod lower;
mod parser;
mod redact;
mod types;
mod validate;

pub use error::ParseError;
pub use frontend::Frontend;
pub use lower::{lower, parse_and_lower};
pub use redact::redact_sql;

/// The deepest tree the parser builds, counted in nested expressions,
/// queries and relations; deeper statements fail with
/// `FAILED_TO_PARSE_TOO_COMPLEX` at the span of the whole statement (Spark's
/// query context for it). It equals the logical-plan IR's depth cap, and it
/// bounds the parser's recursion, so parsing never overflows the stack.
/// Spark has no fixed cap: its parser fails when the JVM stack runs out.
pub const MAX_DEPTH: usize = 1000;

/// The Spark session configs that change how SQL parses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParserOptions {
    /// `spark.sql.ansi.enabled` and `spark.sql.ansi.enforceReservedKeywords`:
    /// reserved keywords are never identifiers.
    pub enforce_reserved_keywords: bool,
    /// `spark.sql.ansi.enabled` and `spark.sql.ansi.doubleQuotedIdentifiers`:
    /// `"x"` is an identifier, not a string.
    pub double_quoted_identifiers: bool,
    /// `spark.sql.legacy.setopsPrecedence.enabled`: all set operators have
    /// one precedence.
    pub legacy_setops_precedence: bool,
    /// `spark.sql.parser.escapedStringLiterals`: backslashes in string
    /// literals are kept as written.
    pub escaped_string_literals: bool,
    /// `spark.sql.legacy.bangEqualsNot`: `!` may stand for `NOT` inside a
    /// predicate (`a ! IN (1)`).
    pub legacy_bang_equals_not: bool,
    /// `spark.sql.legacy.interval.enabled`: an interval literal may mix
    /// year-month and day-time fields.
    pub legacy_interval: bool,
    /// `spark.sql.timestampType = TIMESTAMP_NTZ`: a `TIMESTAMP '...'`
    /// literal without a zone is a `TIMESTAMP_NTZ`.
    pub timestamp_ntz: bool,
    /// `spark.sql.session.timeZone` as a fixed offset in seconds east of
    /// UTC (`UTC` is `Some(0)`); `None` for a region id (which needs the
    /// tz database) or when unset. Typed timestamp literals without a zone
    /// are evaluated in this zone at analysis time.
    pub session_offset: Option<i32>,
}

impl ParserOptions {
    /// Options from Spark session configs; `conf` returns a config's value
    /// if it is set. Unset configs take Spark 4.2's defaults.
    pub fn from_conf<'a>(conf: impl Fn(&str) -> Option<&'a str>) -> Self {
        let on = |key: &str, default: bool| {
            conf(key).map_or(default, |v| v.trim().eq_ignore_ascii_case("true"))
        };
        let ansi = on("spark.sql.ansi.enabled", true);
        Self {
            enforce_reserved_keywords: ansi && on("spark.sql.ansi.enforceReservedKeywords", false),
            double_quoted_identifiers: ansi && on("spark.sql.ansi.doubleQuotedIdentifiers", false),
            legacy_setops_precedence: on("spark.sql.legacy.setopsPrecedence.enabled", false),
            escaped_string_literals: on("spark.sql.parser.escapedStringLiterals", false),
            legacy_bang_equals_not: on("spark.sql.legacy.bangEqualsNot", false),
            legacy_interval: on("spark.sql.legacy.interval.enabled", false),
            timestamp_ntz: conf("spark.sql.timestampType")
                .is_some_and(|v| v.trim().eq_ignore_ascii_case("TIMESTAMP_NTZ")),
            session_offset: conf("spark.sql.session.timeZone").and_then(fixed_offset),
        }
    }
}

/// The offset of `UTC`, `GMT`, `Z` or `[+-]hh:mm`, in seconds east of UTC;
/// `None` for region ids (which need the tz database) and malformed zones.
fn fixed_offset(zone: &str) -> Option<i32> {
    let zone = zone.trim();
    if ["UTC", "GMT", "Z", "Etc/UTC"].contains(&zone) {
        return Some(0);
    }
    let rest = zone
        .strip_prefix("UTC")
        .or_else(|| zone.strip_prefix("GMT"))
        .unwrap_or(zone);
    let (sign, hm) = match rest.as_bytes().first()? {
        b'+' => (1, &rest[1..]),
        b'-' => (-1, &rest[1..]),
        _ => return None,
    };
    let (h, m) = hm.split_once(':').unwrap_or((hm, "0"));
    let (h, m): (i32, i32) = (h.parse().ok()?, m.parse().ok()?);
    ((0..=18).contains(&h) && (0..60).contains(&m)).then(|| sign * (h * 3600 + m * 60))
}

/// Parses one query statement, optionally followed by `;`.
///
/// # Errors
/// A [`ParseError`] with Spark's error class: `PARSE_SYNTAX_ERROR`,
/// `PARSE_EMPTY_STATEMENT`, `UNCLOSED_BRACKETED_COMMENT`,
/// `INVALID_IDENTIFIER`, `FAILED_TO_PARSE_TOO_COMPLEX` or
/// `SYNTAX_DISCONTINUED.BANG_EQUALS_NOT`; and the errors Spark raises
/// after parsing, while it builds the plan (invalid literals and types,
/// duplicate CTE names, ...).
///
/// ```
/// use krishiv_sql::{parse_query, ParserOptions};
///
/// let e = parse_query("SELECT DATE '2020-02-30'", &ParserOptions::default()).unwrap_err();
/// assert_eq!(e.error.error_class().name(), "INVALID_TYPED_LITERAL");
/// ```
pub fn parse_query(sql: &str, options: &ParserOptions) -> Result<ast::Query, ParseError> {
    parser::parse_query(sql, *options).map_err(|e| e.redacted_in(sql))
}

/// Parses one statement, optionally followed by `;`: a query, or
/// `EXPLAIN [EXTENDED | FORMATTED | CODEGEN | COST]` of a statement.
///
/// ```
/// use krishiv_sql::ast::{ExplainMode, Statement};
/// use krishiv_sql::{parse_statement, ParserOptions};
///
/// let sql = "EXPLAIN EXTENDED SELECT a FROM t";
/// let Statement::Explain(e) = parse_statement(sql, &ParserOptions::default()).unwrap() else {
///     unreachable!()
/// };
/// assert_eq!(e.mode, ExplainMode::Extended);
/// assert!(matches!(*e.statement, Statement::Query(_)));
/// ```
///
/// # Errors
/// The errors of [`parse_query`], and `INVALID_STATEMENT_OR_CLAUSE` for
/// `EXPLAIN LOGICAL`, which Spark parses and then rejects.
pub fn parse_statement(sql: &str, options: &ParserOptions) -> Result<ast::Statement, ParseError> {
    parser::parse_statement(sql, *options).map_err(|e| e.redacted_in(sql))
}

/// Parses a multipart identifier, as Spark's
/// `CatalystSqlParser.parseMultipartIdentifier` does for `spark.table(name)`:
/// dots separate parts outside backquotes.
///
/// ```
/// use krishiv_sql::{parse_multipart_identifier, ParserOptions};
///
/// let d = ParserOptions::default();
/// let name = parse_multipart_identifier("db.`my.table`", &d).unwrap();
/// let parts: Vec<&str> = name.parts.iter().map(|p| p.value.as_str()).collect();
/// assert_eq!(parts, ["db", "my.table"]);
/// let e = parse_multipart_identifier("a..b", &d).unwrap_err();
/// assert_eq!(e.error.error_class().name(), "PARSE_SYNTAX_ERROR");
/// ```
///
/// # Errors
/// The syntax errors of [`parse_query`] (`PARSE_EMPTY_STATEMENT` for an
/// empty name).
pub fn parse_multipart_identifier(
    sql: &str,
    options: &ParserOptions,
) -> Result<ast::ObjectName, ParseError> {
    parser::parse_multipart_identifier(sql, *options).map_err(|e| e.redacted_in(sql))
}

/// Parses a data type string, as Spark's `CatalystSqlParser.parseDataType`
/// does (`int`, `decimal(10, 2)`, `array<struct<a: int>>`), with the checks
/// Spark makes on the type it builds.
///
/// ```
/// use krishiv_sql::{parse_data_type, ParserOptions};
///
/// let d = ParserOptions::default();
/// assert_eq!(parse_data_type("ARRAY < INT >", &d).unwrap().to_string(), "array<int>");
/// let e = parse_data_type("a InvalidType", &d).unwrap_err();
/// assert_eq!(
///     e.error.to_string(),
///     "[PARSE_SYNTAX_ERROR] Syntax error at or near 'InvalidType': extra input 'InvalidType'. \
///      SQLSTATE: 42601"
/// );
/// ```
///
/// # Errors
/// The syntax errors of [`parse_query`], and `UNSUPPORTED_DATATYPE`,
/// `DATATYPE_MISSING_SIZE`, `INCOMPLETE_TYPE_DEFINITION.*`, the decimal
/// bound errors and `UNSUPPORTED_TIME_PRECISION` for a type Spark rejects
/// after parsing.
pub fn parse_data_type(
    sql: &str,
    options: &ParserOptions,
) -> Result<ast::DataTypeSpec, ParseError> {
    parser::parse_data_type(sql, *options).map_err(|e| e.redacted_in(sql))
}

/// Parses a table schema string, as Spark's
/// `CatalystSqlParser.parseTableSchema` does: `name type [NOT NULL]
/// [COMMENT '...']`, separated by commas.
///
/// # Errors
/// As for [`parse_data_type`], for the column list grammar.
pub fn parse_table_schema(
    sql: &str,
    options: &ParserOptions,
) -> Result<Vec<ast::StructField>, ParseError> {
    parser::parse_table_schema(sql, *options).map_err(|e| e.redacted_in(sql))
}

/// Parses a schema string as Spark's `DataType.fromDDL` does (the schema
/// argument of `from_json`, `from_csv` and `from_xml`): a data type, or else
/// a table schema, which becomes a `STRUCT` of its columns. When neither
/// parses, the error is the data type's.
///
/// ```
/// use krishiv_sql::{parse_ddl_type, ParserOptions};
///
/// let d = ParserOptions::default();
/// let t = parse_ddl_type("a int, b array<string>", &d).unwrap();
/// assert_eq!(t.to_string(), "struct<`a`:int,`b`:array<string>>");
/// assert_eq!(parse_ddl_type("map<string,int>", &d).unwrap().to_string(), "map<string,int>");
/// ```
///
/// # Errors
/// The error of [`parse_data_type`] when [`parse_table_schema`] fails too.
pub fn parse_ddl_type(sql: &str, options: &ParserOptions) -> Result<ast::DataTypeSpec, ParseError> {
    parse_data_type(sql, options).or_else(|e| {
        let columns = parse_table_schema(sql, options).map_err(|_| e)?;
        let start = columns.first().map_or(0, |c| c.span.start);
        let end = columns.last().map_or(sql.len(), |c| c.span.end);
        Ok(ast::DataTypeSpec {
            kind: ast::DataTypeKind::Struct(Some(columns)),
            span: ast::Span::new(start, end),
        })
    })
}

#[cfg(test)]
mod tests;
