// SPDX-License-Identifier: Apache-2.0
//! Spark SQL parsing (layer L3).
//!
//! [`parse_query`] parses a Spark 4.2 query statement (`SELECT`, `WITH`,
//! `VALUES`, `TABLE`, `FROM ... SELECT`, set operations) into the syntax tree
//! of [`ast`], whose nodes carry source spans; [`parse_statement`] also
//! accepts commands (`SET`, `RESET`, `CREATE`/`DROP` views, tables,
//! namespaces and SQL-bodied functions, `USE`) and `EXPLAIN`. [`parse_data_type`],
//! [`parse_table_schema`] and [`parse_ddl_type`] parse type strings with the
//! same grammar. The parser is hand-written:
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
mod classify;
mod error;
mod frontend;
mod generated {
    pub(crate) mod letters;
}
mod keywords;
pub use keywords::{SQL_KEYWORDS, sql_keywords};
pub mod lexer;
mod lower;
mod params;
mod parser;
mod redact;
mod types;
mod validate;

pub use classify::{
    ExplainKind, StatementClass, Target, TargetObject, classify, explain_kind, statement_queries,
    statement_targets,
};
pub use error::{ParseError, parse_exception_suffix};
pub use frontend::Frontend;
pub use lower::{
    lower, lower_expression, lower_select_item, lower_with, parse_and_lower,
    parse_and_lower_named_expression,
};
pub use params::{ParameterMarker, parameter_markers};
pub use redact::redact_sql;

/// Spark's check that a temporary function's name takes no database
/// (`INVALID_TEMP_OBJ_QUALIFIER`), for a name an `IDENTIFIER(...)` clause
/// supplied after parsing.
///
/// # Errors
/// `INVALID_TEMP_OBJ_QUALIFIER` at `span`.
pub fn check_temporary_function_name(
    name: &ast::ObjectName,
    span: ast::Span,
) -> Result<(), ParseError> {
    parser::check_temporary_function_name(name, span)
}

/// The deepest tree the parser builds, counted in nested expressions,
/// queries and relations; deeper statements fail with
/// `FAILED_TO_PARSE_TOO_COMPLEX` at the span of the whole statement (Spark's
/// query context for it). It equals the logical-plan IR's depth cap, and it
/// bounds the parser's recursion, so parsing never overflows the stack.
/// Spark has no fixed cap: its parser fails when the JVM stack runs out.
pub const MAX_DEPTH: usize = 1000;

/// Spark's default `spark.sql.sources.bucketing.maxBuckets`.
pub const DEFAULT_MAX_BUCKETS: u64 = 100_000;

/// The Spark session configs that change how SQL parses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// `spark.sql.legacy.identifierClause`: only the legacy
    /// `IDENTIFIER(expression)` function syntax is allowed;
    /// `IDENTIFIER('name')` is not an identifier.
    pub legacy_identifier_clause: bool,
    /// `spark.sql.legacy.interval.enabled`: an interval literal may mix
    /// year-month and day-time fields.
    pub legacy_interval: bool,
    /// `spark.sql.legacy.consecutiveStringLiterals.enabled`: doubled quotes
    /// inside string literals are dropped (`'a''b'` is `ab`), instead of
    /// read as one quote character (`a'b`). Adjacent literals concatenate
    /// either way.
    pub consecutive_string_literals: bool,
    /// `spark.sql.timestampType = TIMESTAMP_NTZ`: a `TIMESTAMP '...'`
    /// literal without a zone is a `TIMESTAMP_NTZ`.
    pub timestamp_ntz: bool,
    /// `spark.sql.sources.bucketing.maxBuckets`: a bucket count above it
    /// fails with `INVALID_BUCKET_COUNT`.
    pub max_buckets: u64,
    /// `spark.sql.parser.singleCharacterPipeOperator.enabled` (on by
    /// default): `|` followed by a pipe operator keyword is the pipe
    /// operator `|>`, never a bitwise or.
    pub single_character_pipe_operator: bool,
    /// `spark.sql.caseSensitive`: names in a script (labels, `FOR` variables, conditions) compare
    /// exactly instead of after lower-casing.
    pub case_sensitive: bool,
    /// `spark.sql.legacy.viewSchemaBindingMode` (on by default): a view may
    /// take `WITH SCHEMA ...`; off, the clause is `FEATURE_NOT_ENABLED`.
    pub view_schema_binding: bool,
}

impl Default for ParserOptions {
    /// Spark 4.2's defaults.
    fn default() -> Self {
        Self::from_conf(|_| None)
    }
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
            legacy_identifier_clause: on("spark.sql.legacy.identifierClause", false),
            legacy_interval: on("spark.sql.legacy.interval.enabled", false),
            consecutive_string_literals: on(
                "spark.sql.legacy.consecutiveStringLiterals.enabled",
                false,
            ),
            timestamp_ntz: conf("spark.sql.timestampType")
                .is_some_and(|v| v.trim().eq_ignore_ascii_case("TIMESTAMP_NTZ")),
            max_buckets: conf("spark.sql.sources.bucketing.maxBuckets")
                .and_then(|v| v.trim().parse::<u64>().ok())
                .filter(|v| (1..=i32::MAX as u64).contains(v))
                .unwrap_or(DEFAULT_MAX_BUCKETS),
            single_character_pipe_operator: on(
                "spark.sql.parser.singleCharacterPipeOperator.enabled",
                true,
            ),
            case_sensitive: on("spark.sql.caseSensitive", false),
            view_schema_binding: on("spark.sql.legacy.viewSchemaBindingMode", true),
        }
    }
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

/// Parses one statement, optionally followed by `;`: a query, `SET`,
/// `RESET`, `CREATE TEMPORARY VIEW`, `DROP VIEW`, `CREATE TABLE` (or
/// `CREATE TABLE ... LIKE`), `DROP TABLE`, `CREATE`/`DROP`
/// `DATABASE`|`SCHEMA`|`NAMESPACE`, `USE`, `ALTER TABLE`|`VIEW`|`NAMESPACE`,
/// `CREATE`/`DROP VIEW`, `INSERT`, `DELETE`, `UPDATE`, `MERGE`, `TRUNCATE`,
/// `SHOW`, `DESCRIBE`, `CACHE`/`UNCACHE`/`CLEAR CACHE`, `REFRESH`,
/// `ANALYZE`, `MSCK`/`REPAIR`, `ADD`/`LIST`, or `EXPLAIN [EXTENDED |
/// FORMATTED | CODEGEN | COST]` of a statement.
/// Commands beyond queries parse to syntax trees and are validated;
/// there is no lowering for them.
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

/// Parses one named expression (`namedExpression`), as Spark's
/// `CatalystSqlParser.parseExpression` does for `F.expr`,
/// `df.selectExpr("...")` (once per argument string) and
/// `df.filter("...")`: one expression with an optional alias, then the end
/// of the input. The expression grammar is that of [`parse_query`] (spans,
/// the [`MAX_DEPTH`] recursion limit and error classes); a statement or
/// trailing tokens are a `PARSE_SYNTAX_ERROR`, as in Spark.
///
/// ```
/// use krishiv_sql::{ParserOptions, parse_named_expression};
///
/// let d = ParserOptions::default();
/// let item = parse_named_expression("a AS b", &d).unwrap();
/// assert!(item.alias.is_some());
/// let trailing = parse_named_expression("a, b", &d).unwrap_err();
/// assert_eq!(
///     trailing.error.to_string(),
///     "[PARSE_SYNTAX_ERROR] Syntax error at or near ','. SQLSTATE: 42601"
/// );
/// ```
///
/// # Errors
/// The syntax errors of [`parse_query`] (`PARSE_EMPTY_STATEMENT` for an
/// empty string), and the errors Spark raises after parsing (invalid
/// literals and types, ...) for the expression it sits in.
pub fn parse_named_expression(
    sql: &str,
    options: &ParserOptions,
) -> Result<ast::SelectItem, ParseError> {
    parser::parse_named_expression(sql, *options).map_err(|e| e.redacted_in(sql))
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
            collation: None,
            span: ast::Span::new(start, end),
        })
    })
}

#[cfg(test)]
mod tests;
