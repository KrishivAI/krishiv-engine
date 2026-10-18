// SPDX-License-Identifier: Apache-2.0
//! Spark SQL parsing (layer L3).
//!
//! [`parse_query`] parses a Spark 4.2 query statement (`SELECT`, `WITH`,
//! `VALUES`, `TABLE`, `FROM ... SELECT`, set operations) into the syntax tree
//! of [`ast`], whose nodes carry source spans. The parser is hand-written:
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
mod generated {
    pub(crate) mod letters;
}
mod keywords;
pub mod lexer;
mod lower;
mod parser;
mod redact;
mod types;

pub use error::ParseError;
pub use lower::{lower, parse_and_lower};
pub use redact::redact_sql;

/// The deepest tree the parser builds, counted in nested expressions,
/// queries and relations; deeper statements fail with
/// `FAILED_TO_PARSE_TOO_COMPLEX`. It equals the logical-plan IR's depth
/// cap, and it bounds the parser's recursion, so parsing never overflows
/// the stack.
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
        }
    }
}

/// Parses one query statement, optionally followed by `;`.
///
/// # Errors
/// A [`ParseError`] with Spark's error class: `PARSE_SYNTAX_ERROR`,
/// `PARSE_EMPTY_STATEMENT`, `UNCLOSED_BRACKETED_COMMENT`,
/// `INVALID_IDENTIFIER`, `FAILED_TO_PARSE_TOO_COMPLEX` or
/// `SYNTAX_DISCONTINUED.BANG_EQUALS_NOT`.
pub fn parse_query(sql: &str, options: &ParserOptions) -> Result<ast::Query, ParseError> {
    parser::parse_query(sql, *options).map_err(|e| e.redacted_in(sql))
}

#[cfg(test)]
mod tests;
