// SPDX-License-Identifier: Apache-2.0
//! The checks of `ANALYZE TABLE` and `ANALYZE TABLES`.

use krishiv_common::error::InvalidSqlSyntaxAnalyzeTableUnexpectedNoscan;

use super::at;
use crate::ast::{Ident, Span};
use crate::error::ParseError;

/// Spark's `AstBuilder` takes any identifier after `COMPUTE STATISTICS` and refuses all but
/// `NOSCAN`, over the whole statement, before it reads anything else of it.
pub(super) fn analyze_option(
    unexpected: Option<&Ident>,
    statement: Span,
) -> Result<(), ParseError> {
    unexpected.map_or(Ok(()), |word| {
        Err(at(
            InvalidSqlSyntaxAnalyzeTableUnexpectedNoscan {
                ctx: word.value.to_uppercase(),
            },
            statement,
        ))
    })
}
