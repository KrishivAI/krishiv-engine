// SPDX-License-Identifier: Apache-2.0
//! Parse errors: a Spark error class from the registry plus the source span
//! it refers to (Spark's query context).
//!
//! | Situation | Error class |
//! |---|---|
//! | unexpected token or end of input | `PARSE_SYNTAX_ERROR` |
//! | only whitespace | `PARSE_EMPTY_STATEMENT` |
//! | `/*` without `*/` | `UNCLOSED_BRACKETED_COMMENT` |
//! | unquoted identifier with non-ASCII characters, or `a-b` as a name | `INVALID_IDENTIFIER` |
//! | nesting deeper than [`crate::MAX_DEPTH`] (span: the whole statement) | `FAILED_TO_PARSE_TOO_COMPLEX` |
//! | `!` for `NOT` in `a ! IN (...)` and similar | `SYNTAX_DISCONTINUED.BANG_EQUALS_NOT` |
//! | `EXPLAIN LOGICAL` | `INVALID_STATEMENT_OR_CLAUSE` |
//!
//! A syntax error never echoes the text of a string literal (it may be a
//! secret): the token is described instead. An error at a secret that is
//! not a string (an unquoted password value) has its parameters redacted.

use krishiv_common::error::{
    CreateViewWithIfNotExistsAndReplace, FailedToParseTooComplex, InvalidIdentifier,
    InvalidResetCommandFormat, InvalidSetSyntax, InvalidStatementOrClause, InvalidTempObjQualifier,
    KrishivError, KrishivUnsupportedFeature, LegacyErrorTemp0053, ParseEmptyStatement,
    ParseSyntaxError, SyntaxDiscontinuedBangEqualsNot, UnclosedBracketedComment,
};

use crate::ast::Span;

/// The `ParseException` suffix for `error` over the (redacted) text `sql`
/// (`""` when `error` is not what parsing `sql` raises): Spark shows a
/// parse error with its line, 0-based position and the offending line, not
/// with a query-context summary (Spark's golden `PARSE_*` entries carry no
/// `queryContext`). Both user surfaces (the `krishiv sql` command and the
/// text of Python exceptions from `spark.sql`) use this for parse errors
/// and [`krishiv_common::error::render_with_context`] otherwise, so the two
/// agree and match PySpark.
///
/// The statement is parsed again here to recover the span, also when it is
/// empty (a syntax error at the end of the input), which a query context
/// does not show. The suffix is `""` unless re-parsing `sql` raises the
/// same error class, so every parser error class (including
/// `FAILED_TO_PARSE_TOO_COMPLEX`, whose span is the whole statement) takes
/// this layout without a class-prefix list to maintain. `line_offset` is
/// the number of input lines before `sql` (a multi-statement input counts
/// lines from its start on every surface).
#[must_use]
pub fn parse_exception_suffix(sql: &str, line_offset: usize, error: &KrishivError) -> String {
    let Err(parse) = crate::parse_statement(sql, &crate::ParserOptions::default()) else {
        return String::new();
    };
    if parse.error.error_class() != error.error_class() {
        return String::new();
    };
    let start = parse.span.start.min(sql.len());
    let before = &sql[..sql.floor_char_boundary(start)];
    let line_no = line_offset + before.matches('\n').count() + 1;
    let line_start = before.rfind('\n').map_or(0, |i| i + 1);
    let line = sql[line_start..].lines().next().unwrap_or("");
    let col = before[line_start..].chars().count();
    let width = sql[before.len()..parse.span.end.max(start).min(sql.len())]
        .chars()
        .take_while(|c| *c != '\n')
        .count()
        .max(1);
    format!(
        "\n(line {line_no}, pos {col})\n\n== SQL ==\n{line}\n{}{}",
        " ".repeat(col),
        "^".repeat(width)
    )
}

/// A statement Spark's parser would reject.
#[derive(Debug, Clone, Eq, thiserror::Error)]
#[error("{error}")]
pub struct ParseError {
    /// The error, with Spark's class, SQLSTATE and message parameters.
    pub error: KrishivError,
    /// The source text the error refers to (empty at the end of input).
    pub span: Span,
}

impl PartialEq for ParseError {
    /// The error and its position (unlike [`Span`] equality alone).
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error && self.span.range() == other.span.range()
    }
}

impl ParseError {
    /// This error with its message parameters redacted when its span
    /// touches a secret of `sql` (a token at a password echoes it).
    #[must_use]
    pub(crate) fn redacted_in(self, sql: &str) -> Self {
        let (start, end) = (self.span.start, self.span.end.max(self.span.start + 1));
        let secret = crate::redact::secret_ranges(sql)
            .iter()
            .any(|r| r.start < end && start < r.end);
        if secret {
            Self {
                error: self.error.redacted(),
                ..self
            }
        } else {
            self
        }
    }
}

impl From<ParseError> for KrishivError {
    fn from(e: ParseError) -> Self {
        e.error
    }
}

fn at(error: impl Into<KrishivError>, span: Span) -> ParseError {
    ParseError {
        error: error.into(),
        span,
    }
}

/// `PARSE_SYNTAX_ERROR` at `found` (a token as Spark displays it, or
/// `end of input`) with `hint` (empty, or `: missing ')'`).
pub(crate) fn syntax_error(found: String, hint: String, span: Span) -> ParseError {
    at(ParseSyntaxError { error: found, hint }, span)
}

pub(crate) fn unclosed_comment(span: Span) -> ParseError {
    at(UnclosedBracketedComment {}, span)
}

pub(crate) fn empty_statement(span: Span) -> ParseError {
    at(ParseEmptyStatement {}, span)
}

pub(crate) fn invalid_identifier(ident: &str, span: Span) -> ParseError {
    at(
        InvalidIdentifier {
            ident: ident.to_owned(),
        },
        span,
    )
}

pub(crate) fn too_complex(span: Span) -> ParseError {
    at(FailedToParseTooComplex {}, span)
}

pub(crate) fn bang_equals_not(span: Span) -> ParseError {
    at(
        SyntaxDiscontinuedBangEqualsNot {
            clause: "!".to_owned(),
        },
        span,
    )
}

/// `INVALID_STATEMENT_OR_CLAUSE` for `operation` (e.g. `EXPLAIN LOGICAL`).
pub(crate) fn invalid_statement(operation: &str, span: Span) -> ParseError {
    at(
        InvalidStatementOrClause {
            operation: operation.to_owned(),
        },
        span,
    )
}

/// `INVALID_SET_SYNTAX`: a `SET` that is not `SET`, `SET -v`, `SET key`
/// or `SET key = value` with a plain key.
pub(crate) fn invalid_set_syntax(span: Span) -> ParseError {
    at(InvalidSetSyntax {}, span)
}

/// `INVALID_RESET_COMMAND_FORMAT`: a `RESET` that is not `RESET` or
/// `RESET key` with a plain key.
pub(crate) fn invalid_reset_command_format(span: Span) -> ParseError {
    at(InvalidResetCommandFormat {}, span)
}

/// `_LEGACY_ERROR_TEMP_0053`: `IF NOT EXISTS` on a temporary view.
pub(crate) fn temp_view_if_not_exists(span: Span) -> ParseError {
    at(LegacyErrorTemp0053 {}, span)
}

/// `CREATE_VIEW_WITH_IF_NOT_EXISTS_AND_REPLACE`: `OR REPLACE` with
/// `IF NOT EXISTS` on a temporary view.
pub(crate) fn create_view_with_if_not_exists_and_replace(
    view_name: String,
    span: Span,
) -> ParseError {
    at(CreateViewWithIfNotExistsAndReplace { view_name }, span)
}

/// `INVALID_TEMP_OBJ_QUALIFIER`: a qualified temporary view name.
pub(crate) fn invalid_temp_obj_qualifier(
    object_type: &str,
    object_name: String,
    qualifier: String,
    span: Span,
) -> ParseError {
    at(
        InvalidTempObjQualifier {
            object_type: object_type.to_owned(),
            object_name,
            qualifier,
        },
        span,
    )
}

/// `KRISHIV_UNSUPPORTED_FEATURE` for the `SET TIME ZONE` form `form`
/// (e.g. an `INTERVAL` zone).
pub(crate) fn unsupported_set_time_zone(form: &str, span: Span) -> ParseError {
    at(
        KrishivUnsupportedFeature {
            feature: form.to_owned(),
        },
        span,
    )
}
