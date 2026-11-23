// SPDX-License-Identifier: Apache-2.0
//! The checks Spark's `AstBuilder` makes on `NEAREST BY` and `ASOF` joins and
//! on the `TABLE(...)` arguments of table-valued functions.

use krishiv_common::error::{
    AsofJoinMatchConditionInvalidOperator,
    InvalidSqlSyntaxInvalidTableFunctionIdentifierArgumentMissingParentheses, LegacyErrorTemp0064,
    NearestByJoinNumResultsOutOfRange, NearestByJoinUnsupportedJoinType,
    UnsupportedFeatureAsofJoin, UnsupportedFeatureLateralJoinNearestBy,
};

use super::{Checker, R, at, quote};
use crate::ast::{
    AsOfCriteria, AsOfFilter, BinaryOp, DdlName, Expr, ExprKind, Join, JoinKind, NearestBy, Span,
    TableArgument, TableArgumentPartitioning, TableArgumentSource,
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

    /// An `ASOF JOIN`, in the order of Spark 4.3's `withAsOfJoin`: the flag,
    /// the match condition and its operator (`>=`, `>`, `<=` or `<`), the
    /// `ON` condition, then the right side.
    pub(super) fn asof_join(&self, j: &Join, c: &AsOfCriteria) -> R {
        if !self.opts.asof_join {
            return Err(at(
                UnsupportedFeatureAsofJoin {
                    config: "\"spark.sql.join.asofJoin.enabled\"".to_owned(),
                },
                j.span,
            ));
        }
        self.expr(&c.match_condition)?;
        if let Some(operator) = asof_invalid_operator(self.sql, &c.match_condition) {
            return Err(at(
                AsofJoinMatchConditionInvalidOperator { operator },
                j.span,
            ));
        }
        if let Some(AsOfFilter::On(e)) = &c.filter {
            self.expr(e)?;
        }
        self.factor(&j.right)
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

/// The operator Spark 4.3 names when an `ASOF JOIN`'s match condition is not
/// a `>=`, `>`, `<=` or `<` comparison (`asOfMatchConditionInvalidOperatorText`),
/// or `None` when it is one: `AND` or `OR` at the top, else the first
/// comparison operator in the tree as written, else `IS [NOT] DISTINCT FROM`
/// at the top, else the condition's text.
pub(crate) fn asof_invalid_operator(sql: &str, e: &Expr) -> Option<String> {
    // Spark's builder drops the parentheses around an expression.
    let mut top = e;
    while let ExprKind::Nested(inner) = &top.kind {
        top = inner;
    }
    match &top.kind {
        ExprKind::Binary(_, BinaryOp::GtEq | BinaryOp::Gt | BinaryOp::LtEq | BinaryOp::Lt, _) => {
            return None;
        }
        ExprKind::Binary(_, BinaryOp::And, _) => return Some("AND".to_owned()),
        ExprKind::Binary(_, BinaryOp::Or, _) => return Some("OR".to_owned()),
        _ => {}
    }
    if let Some(op) = first_comparison(sql, e) {
        return Some(op);
    }
    if let ExprKind::IsDistinctFrom { negated, .. } = &top.kind {
        let text = if *negated {
            "IS NOT DISTINCT FROM"
        } else {
            "IS DISTINCT FROM"
        };
        return Some(text.to_owned());
    }
    Some(without_strings(e.span.text(sql).trim()))
}

/// `text` with each string literal's text as `?`: Spark echoes the condition as written, but an
/// error never echoes a string literal (this crate's invariant), so that part differs from
/// Spark's parameter. Everything else stays as written.
fn without_strings(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for t in crate::lexer::tokenize(text) {
        if t.kind.is_string_like() {
            out.push_str(text.get(at..t.span.start).unwrap_or(""));
            out.push('?');
            at = t.span.end;
        }
    }
    out.push_str(text.get(at..).unwrap_or(""));
    out
}

/// The first comparison operator in `e`, parents before children and left
/// to right (the order of Spark's parse-tree walk), as written.
fn first_comparison(sql: &str, e: &Expr) -> Option<String> {
    if let ExprKind::Binary(left, op, right) = &e.kind {
        let text = match op {
            BinaryOp::Eq => Some("="),
            BinaryOp::NullSafeEq => Some("<=>"),
            BinaryOp::Lt => Some("<"),
            BinaryOp::LtEq => Some("<="),
            BinaryOp::Gt => Some(">"),
            BinaryOp::GtEq => Some(">="),
            // `<>` and `!=` are one operator in the tree: read the text
            // between the operands.
            BinaryOp::NotEq => {
                let between = sql.get(left.span.end..right.span.start).unwrap_or("");
                Some(if between.contains("!=") { "!=" } else { "<>" })
            }
            _ => None,
        };
        if let Some(text) = text {
            return Some(text.to_owned());
        }
    }
    let mut found = None;
    e.for_each_child(&mut |child| {
        if found.is_none() {
            found = first_comparison(sql, child);
        }
    });
    found
}
