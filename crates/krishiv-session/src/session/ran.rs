// SPDX-License-Identifier: Apache-2.0
//! [`Ran`]: what a statement run with `queries_only` did (P3-T101 Flight SQL refuses a command at GetFlightInfo).

use super::*;

/// What [`Session::sql_here_marked`] did with a statement.
#[expect(
    clippy::large_enum_variant,
    reason = "a return value matched at once; boxing the frame buys nothing"
)]
pub(crate) enum Ran {
    /// Its frame (a query's, analyzed; a command's result, already run) and whether it was a
    /// command.
    Frame(DataFrame, bool),
    /// Not a query, refused before it was authorized or run (`queries_only`), with its class.
    NotAQuery(krishiv_sql::StatementClass),
}

impl Ran {
    /// The frame, or `KRISHIV_FLIGHT_SQL_NOT_A_QUERY` for a refused statement.
    pub(crate) fn into_frame(self) -> Result<(DataFrame, bool), KrishivError> {
        match self {
            Self::Frame(df, command) => Ok((df, command)),
            Self::NotAQuery(class) => Err(not_a_query(class)),
        }
    }
}

/// `KRISHIV_FLIGHT_SQL_NOT_A_QUERY` (55000) for a statement of `class`.
pub(super) fn not_a_query(class: krishiv_sql::StatementClass) -> KrishivError {
    krishiv_common::error::KrishivFlightSqlNotAQuery {
        statement_class: class.name().to_owned(),
    }
    .into()
}
