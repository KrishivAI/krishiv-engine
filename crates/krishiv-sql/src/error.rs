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

use krishiv_common::error::{
    FailedToParseTooComplex, InvalidIdentifier, KrishivError, ParseEmptyStatement,
    ParseSyntaxError, UnclosedBracketedComment,
};

use crate::ast::Span;

/// A statement Spark's parser would reject.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{error}")]
pub struct ParseError {
    /// The error, with Spark's class, SQLSTATE and message parameters.
    pub error: KrishivError,
    /// The source text the error refers to (empty at the end of input).
    pub span: Span,
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
