// SPDX-License-Identifier: Apache-2.0
//! The SQL front end the analyzer calls for SQL text inside plans
//! (`spark.sql(...)` relations, later view text).

use krishiv_common::redact::RedactedSql;
use krishiv_plan::analyzer::{AnalysisError, SqlFrontend};
use krishiv_plan::{Origin, Plan, SourceId, Span};

use crate::{ParserOptions, parse_and_lower, redact_sql};

/// Parses and lowers SQL text with the session's parser options, and
/// redacts it from its tokens.
#[derive(Debug, Clone, Copy, Default)]
pub struct Frontend {
    /// The parser options of the session.
    pub options: ParserOptions,
}

impl SqlFrontend for Frontend {
    fn lower(&self, query: &str, source: SourceId) -> Result<Plan, AnalysisError> {
        parse_and_lower(query, &self.options, source).map_err(|e| {
            let at = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
            AnalysisError {
                error: e.error,
                origin: Origin::sql(source, Span::new(at(e.span.start), at(e.span.end))),
            }
        })
    }

    fn redact(&self, query: &str) -> RedactedSql {
        redact_sql(query)
    }

    /// The parser options: one bit per flag, mixed with the bucket limit.
    fn fingerprint(&self) -> u64 {
        let o = self.options;
        let flags = [
            o.enforce_reserved_keywords,
            o.double_quoted_identifiers,
            o.legacy_setops_precedence,
            o.escaped_string_literals,
            o.legacy_bang_equals_not,
            o.legacy_interval,
            o.timestamp_ntz,
            o.legacy_identifier_clause,
            o.single_character_pipe_operator,
        ]
        .iter()
        .fold(0, |acc, b| acc << 1 | u64::from(*b));
        flags.wrapping_add(o.max_buckets.wrapping_mul(0x9E3779B97F4A7C15))
    }

    fn keywords(&self) -> Vec<(String, bool)> {
        crate::keywords::sql_keywords(self.options.enforce_reserved_keywords)
    }
}
