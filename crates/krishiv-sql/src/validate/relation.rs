// SPDX-License-Identifier: Apache-2.0
//! The checks Spark's `AstBuilder` makes on `NEAREST BY` joins and on the
//! `TABLE(...)` arguments of table-valued functions.

use krishiv_common::error::{
    InvalidSqlSyntaxInvalidTableFunctionIdentifierArgumentMissingParentheses, LegacyErrorTemp0064,
    NearestByJoinNumResultsOutOfRange, NearestByJoinUnsupportedJoinType,
    UnsupportedFeatureLateralJoinNearestBy,
};

use super::{Checker, R, at, quote};
use crate::ast::{
    DdlName, Join, JoinKind, NearestBy, Span, TableArgument, TableArgumentPartitioning,
    TableArgumentSource,
};

/// `NearestByJoin.MaxNumResults`.
const MAX_NUM_RESULTS: i64 = 100_000;

impl Checker<'_> {
    /// A `NEAREST BY` join: no `LATERAL`, an inner or left outer join, a
    /// result count in range, then the ranking expression.
    pub(super) fn nearest_by(&self, j: &Join, n: &NearestBy) -> R {
        if j.lateral {
            return Err(at(UnsupportedFeatureLateralJoinNearestBy {}, j.span));
        }
        let unsupported = match j.kind {
            JoinKind::Inner | JoinKind::Left => None,
            JoinKind::Cross => Some("CROSS"),
            JoinKind::Right => Some("RIGHT OUTER"),
            JoinKind::Full => Some("FULL OUTER"),
            JoinKind::LeftSemi => Some("LEFT SEMI"),
            JoinKind::LeftAnti => Some("LEFT ANTI"),
        };
        if let Some(join_type) = unsupported {
            return Err(at(
                NearestByJoinUnsupportedJoinType {
                    join_type: join_type.to_owned(),
                    supported: "'INNER', 'LEFT OUTER'".to_owned(),
                },
                j.span,
            ));
        }
        if let Some(text) = &n.num_results {
            let value = text.parse::<i64>().ok();
            if !value.is_some_and(|v| (1..=MAX_NUM_RESULTS).contains(&v)) {
                return Err(at(
                    NearestByJoinNumResultsOutOfRange {
                        num_results: value.map_or_else(|| text.clone(), |v| v.to_string()),
                        min: "1".to_owned(),
                        max: MAX_NUM_RESULTS.to_string(),
                    },
                    j.span,
                ));
            }
        }
        self.expr(&n.ranking)
    }

    /// A `TABLE(...)` argument: the name needs parentheses, then its source
    /// and partitioning.
    pub(super) fn table_argument(&self, t: &TableArgument) -> R {
        if t.unparenthesized
            && let TableArgumentSource::Table(name) = &t.source
        {
            let argument_name = match name {
                DdlName::Parts(n) => n
                    .parts
                    .iter()
                    .map(|p| quote(&p.value))
                    .collect::<Vec<_>>()
                    .join("."),
                DdlName::Dynamic(d) => quote(d.span.text(self.sql)),
            };
            return Err(at(
                InvalidSqlSyntaxInvalidTableFunctionIdentifierArgumentMissingParentheses {
                    argument_name,
                },
                t.span,
            ));
        }
        match &t.source {
            TableArgumentSource::Table(name) => self.ddl_name(name)?,
            TableArgumentSource::Query(q) => self.query(q)?,
        }
        t.partitioning
            .as_ref()
            .map_or(Ok(()), |p| self.table_argument_partitioning(p))
    }

    fn table_argument_partitioning(&self, p: &TableArgumentPartitioning) -> R {
        p.partition_by.iter().try_for_each(|e| self.expr(e))?;
        self.sort_items(&p.order_by)?;
        let invalid = |clause: &str, at_span: Span| {
            at(
                LegacyErrorTemp0064 {
                    msg: format!(
                        "The table function call includes a table argument with an invalid \
                         partitioning/ordering specification: the {clause} clause included \
                         multiple expressions without parentheses surrounding them; please add \
                         parentheses around these expressions and then retry the query again"
                    ),
                },
                at_span,
            )
        };
        if let Some(span) = p.extra_partition {
            return Err(invalid("PARTITION BY", span));
        }
        p.extra_order
            .map_or(Ok(()), |span| Err(invalid("ORDER BY", span)))
    }
}
