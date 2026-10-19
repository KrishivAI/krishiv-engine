// SPDX-License-Identifier: Apache-2.0
//! Hint resolution (Spark's `ResolveJoinStrategyHints` and
//! `RemoveAllHints`), once the plan is resolved.
//!
//! A join strategy hint wraps each relation of its subtree a parameter
//! names in a [`PlanNode::JoinHint`]: an alias (`FROM t AS x` is named
//! `x` only), a table (by a suffix of its name) or a `WITH` reference.
//! The search does not enter an alias, a table, a `WITH` scope or a
//! relation hinted before; a later hint on the same relation is
//! overridden by the outer one, which comes first in the text. Without
//! parameters the hint wraps its whole subtree. Every hint node is then
//! removed; what Spark only logs comes back as warnings.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::{FailedToParseTooComplex, LegacyErrorTemp1046};
use krishiv_common::stack::{STEP, on_big_stack};

use super::AnalysisError;
use super::bind::addr;
use super::exprs::pretty;
use crate::walk::{Node, visit_distinct};
use crate::{Expr, ExprNode, JoinStrategy, Literal, LogicalPlan, PlanNode, ReadSource};

type R<T> = Result<T, AnalysisError>;

/// `root` with its hints resolved, and the warnings they raised.
pub(super) fn resolve(
    root: &Arc<LogicalPlan>,
    case_sensitive: bool,
) -> R<(Arc<LogicalPlan>, Vec<String>)> {
    let found = visit_distinct(Node::Plan(root), |n| match n {
        Node::Plan(p) if matches!(p.node, PlanNode::Hint { .. }) => {
            std::ops::ControlFlow::Break(())
        }
        _ => std::ops::ControlFlow::Continue(()),
    });
    if found.is_continue() {
        return Ok((Arc::clone(root), Vec::new()));
    }
    let mut h = Hints {
        case_sensitive,
        done: HashMap::new(),
        warnings: Vec::new(),
        depth: 0,
    };
    let out = h.plan(root)?;
    Ok((out, h.warnings))
}

struct Hints {
    case_sensitive: bool,
    done: HashMap<usize, (Arc<LogicalPlan>, Arc<LogicalPlan>)>,
    warnings: Vec<String>,
    depth: usize,
}

impl Hints {
    fn plan(&mut self, root: &Arc<LogicalPlan>) -> R<Arc<LogicalPlan>> {
        if let Some((_, out)) = self.done.get(&addr(&**root)) {
            return Ok(Arc::clone(out));
        }
        self.depth += 1;
        let out = if self.depth.is_multiple_of(STEP) {
            on_big_stack(|| self.plan_here(root))
                .unwrap_or_else(|| Err(AnalysisError::new(FailedToParseTooComplex {}, root.origin)))
        } else {
            self.plan_here(root)
        };
        self.depth -= 1;
        let out = out?;
        self.done
            .insert(addr(&**root), (Arc::clone(root), Arc::clone(&out)));
        Ok(out)
    }

    fn plan_here(&mut self, root: &Arc<LogicalPlan>) -> R<Arc<LogicalPlan>> {
        // Inner hints first, as Spark resolves them bottom-up: the outer
        // hint, written first, then overrides them.
        let p = Arc::clone(root)
            .map_children(|c| self.plan(&c))?
            .map_expressions(|e| self.nested(e))?;
        let PlanNode::Hint {
            input,
            name,
            params,
        } = &p.node
        else {
            return Ok(p);
        };
        let shown = format!(
            "{name}({})",
            params
                .iter()
                .map(|e| shown(e))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let Some(strategy) = JoinStrategy::from_hint(name) else {
            self.warnings.push(format!("Unrecognized hint: {shown}"));
            return Ok(Arc::clone(input));
        };
        if params.is_empty() {
            return Ok(hinted(input, strategy));
        }
        let names = params
            .iter()
            .map(|e| relation_name(e).ok_or_else(|| bad_parameter(e)))
            .collect::<R<Vec<_>>>()?;
        let mut matched = vec![false; names.len()];
        let out = self.apply(input, &names, &mut matched, strategy);
        for (n, m) in names.iter().zip(matched) {
            if !m {
                self.warnings.push(format!(
                    "Count not find relation '{}' specified in hint '{shown}'.",
                    n.join(".")
                ));
            }
        }
        Ok(out)
    }

    /// `e` with the hints of its subquery plans resolved.
    fn nested(&mut self, e: Arc<Expr>) -> R<Arc<Expr>> {
        e.transform_up(&mut |x: Arc<Expr>| match &x.node {
            ExprNode::Subquery { kind, plan } => {
                let rewritten = self.plan(plan)?;
                if Arc::ptr_eq(&rewritten, plan) {
                    return Ok(x);
                }
                let node = ExprNode::Subquery {
                    kind: kind.clone(),
                    plan: rewritten,
                };
                Ok(Arc::new(Expr::new(node).with_origin(x.origin)))
            }
            _ => Ok(x),
        })
    }

    /// `p` with the relations `names` match hinted with `strategy`.
    fn apply(
        &mut self,
        p: &Arc<LogicalPlan>,
        names: &[Vec<String>],
        matched: &mut [bool],
        strategy: JoinStrategy,
    ) -> Arc<LogicalPlan> {
        let ident: Option<Vec<String>> = match &p.node {
            PlanNode::SubqueryAlias { alias, .. } => Some(vec![alias.clone()]),
            PlanNode::Scan(s) => match &s.source {
                ReadSource::Table { name, .. } => Some(name.clone()),
                ReadSource::Files { .. } => Some(Vec::new()),
            },
            PlanNode::CteRef(r) => Some(vec![r.name.clone()]),
            PlanNode::JoinHint {
                input,
                strategy: old,
            } => {
                let inner = match &input.node {
                    PlanNode::SubqueryAlias { alias, .. } => Some(vec![alias.clone()]),
                    PlanNode::Scan(s) => match &s.source {
                        ReadSource::Table { name, .. } => Some(name.clone()),
                        ReadSource::Files { .. } => None,
                    },
                    PlanNode::CteRef(r) => Some(vec![r.name.clone()]),
                    _ => None,
                };
                if let Some(ident) = inner
                    && self.mark(&ident, names, matched)
                {
                    self.warnings.push(format!(
                        "Hint (strategy={}) is overridden by another hint and will not take effect.",
                        old.name()
                    ));
                    return hinted(input, strategy);
                }
                return Arc::clone(p);
            }
            PlanNode::WithCte { .. } => return Arc::clone(p),
            _ => None,
        };
        if let Some(ident) = ident {
            if self.mark(&ident, names, matched) {
                return hinted(p, strategy);
            }
            return Arc::clone(p);
        }
        Arc::clone(p)
            .map_children(|c| {
                Ok::<_, std::convert::Infallible>(self.apply(&c, names, matched, strategy))
            })
            .unwrap_or_else(|never| match never {})
    }

    /// Whether a name of `names` matches `ident` (a suffix of it), marking
    /// each one that does.
    fn mark(&self, ident: &[String], names: &[Vec<String>], matched: &mut [bool]) -> bool {
        let mut any = false;
        for (n, m) in names.iter().zip(matched.iter_mut()) {
            let suffix = n.len() <= ident.len()
                && n.iter().zip(&ident[ident.len() - n.len()..]).all(|(a, b)| {
                    if self.case_sensitive {
                        a == b
                    } else {
                        a.eq_ignore_ascii_case(b)
                    }
                });
            if suffix && !n.is_empty() {
                *m = true;
                any = true;
            }
        }
        any
    }
}

fn hinted(input: &Arc<LogicalPlan>, strategy: JoinStrategy) -> Arc<LogicalPlan> {
    Arc::new(
        LogicalPlan::new(PlanNode::JoinHint {
            input: Arc::clone(input),
            strategy,
        })
        .with_origin(input.origin),
    )
}

/// The relation name a join strategy hint parameter gives: an identifier
/// or a string (split at dots).
fn relation_name(e: &Expr) -> Option<Vec<String>> {
    match &e.node {
        ExprNode::Column(c) if c.plan_id.is_none() => Some(c.name.clone()),
        ExprNode::Literal(Literal::String(s)) => Some(s.split('.').map(str::to_owned).collect()),
        _ => None,
    }
}

/// A hint parameter as written.
fn shown(e: &Expr) -> String {
    match &e.node {
        ExprNode::Column(c) => c.name.join("."),
        ExprNode::Literal(Literal::String(s)) => format!("'{s}'"),
        _ => pretty(e),
    }
}

/// `_LEGACY_ERROR_TEMP_1046` for a parameter that names no relation; the
/// class is Spark's for a literal, else its unresolved function.
fn bad_parameter(e: &Expr) -> AnalysisError {
    let class = match &e.node {
        ExprNode::Literal(_) => "org.apache.spark.sql.catalyst.expressions.Literal",
        _ => "org.apache.spark.sql.catalyst.analysis.UnresolvedFunction",
    };
    AnalysisError::new(
        LegacyErrorTemp1046 {
            unsupported: format!("\"{}\"", pretty(e)),
            class: format!("class {class}"),
        },
        e.origin,
    )
}
