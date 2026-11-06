// SPDX-License-Identifier: Apache-2.0
//! Lowering a statement whose `IDENTIFIER(..)` clauses the session substituted before parsing
//! (`krishiv-session` `identifier_clause.rs`): the substituted names' spans mark their columns
//! [`by_identifier`](krishiv_plan::ColumnRef::by_identifier), so the analyzer names a select
//! item of one as Spark names the clause (by the SQL of what it resolves to).

use krishiv_plan::{Plan, SourceId};

use super::{Lowerer, R, err};
use crate::ast::{Query, Span};

/// [`lower_with`] for a statement whose `IDENTIFIER(..)` clauses the caller substituted before
/// parsing: a column whose span is one of `identifier_spans` (the substituted names, in the
/// parsed text) is marked
/// [`ColumnRef::by_identifier`](krishiv_plan::ColumnRef::by_identifier), so the analyzer names
/// it as Spark names the clause.
///
/// # Errors
/// As for [`lower`].
pub fn lower_identified(
    query: &Query,
    source: SourceId,
    case_sensitive: bool,
    identifier_spans: &[Span],
) -> R<Plan> {
    let mut l = Lowerer::new(
        source,
        query.span,
        case_sensitive,
        identifier_spans.to_vec(),
    );
    let plan = l.query(query)?;
    Plan::try_new(plan).map_err(|e| err(e, query.span))
}

impl Lowerer {
    pub(super) fn new(
        source: SourceId,
        root: Span,
        case_sensitive: bool,
        identifier_spans: Vec<Span>,
    ) -> Self {
        Self {
            source,
            depth: 0,
            root,
            windows: Vec::new(),
            recursive: 0,
            case_sensitive,
            identifier_spans,
        }
    }

    /// Whether the column at `sp` is a name the session substituted for a clause.
    pub(super) fn by_identifier(&self, sp: Span) -> bool {
        // `Span`'s `==` ignores positions; compare the bytes.
        self.identifier_spans
            .iter()
            .any(|s| (s.start, s.end) == (sp.start, sp.end))
    }
}
