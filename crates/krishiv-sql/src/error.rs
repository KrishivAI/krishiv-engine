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
//! | nesting deeper than [`crate::MAX_DEPTH`] | `FAILED_TO_PARSE_TOO_COMPLEX` |
//! | `!` for `NOT` in `a ! IN (...)` and similar | `SYNTAX_DISCONTINUED.BANG_EQUALS_NOT` |
//! | `EXPLAIN LOGICAL` | `INVALID_STATEMENT_OR_CLAUSE` |
//!
//! A syntax error never echoes the text of a string literal (it may be a
//! secret): the token is described instead. An error at a secret that is
//! not a string (an unquoted password value) has its parameters redacted.

use krishiv_common::error::{
    FailedToParseTooComplex, InvalidIdentifier, InvalidStatementOrClause, KrishivError,
    ParseEmptyStatement, ParseSyntaxError, SyntaxDiscontinuedBangEqualsNot,
    UnclosedBracketedComment,
};

use crate::ast::Span;

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
