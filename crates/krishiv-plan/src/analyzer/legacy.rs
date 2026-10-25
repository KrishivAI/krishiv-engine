// SPDX-License-Identifier: Apache-2.0
//! Legacy CTE precedence (`spark.sql.legacy.ctePrecedencePolicy`).
//!
//! * `LEGACY`: Spark 2.x inlining. A `WITH` level's definitions only see the
//!   previously defined siblings of their own level; a nested `WITH` in a
//!   definition's body is resolved on its own, while references in a nested
//!   main query are substituted with the enclosing levels' definitions,
//!   outermost first, before the nested level resolves. [`eliminate`] inlines
//!   every `WithCte` this way, so the normal (corrected) resolution of the
//!   result is the legacy answer. Table reads that stay unresolved (a name no
//!   enclosing level defines) fail later with `TABLE_OR_VIEW_NOT_FOUND`.
//! * `EXCEPTION`: [`nested_conflict`] reports a nested definition shadowing
//!   an outer one, as Spark's `assertNoNameConflictsInCTE` does; resolution
//!   itself is the corrected one.

use std::convert::Infallible;
use std::sync::Arc;

use krishiv_common::error::AmbiguousAliasInNestedCte;

use super::bind::same;
use super::catalog::quoted;
use super::exprs::SPARK_DOC_ROOT;
use super::{AnalysisError, CtePrecedencePolicy};
use crate::{Expr, ExprNode, LogicalPlan, Origin, PlanNode, ReadSource};

/// `r`, which never fails.
fn sure<T>(r: Result<T, Infallible>) -> T {
    match r {
        Ok(t) => t,
        Err(e) => match e {},
    }
}

/// What [`eliminate`] and [`nested_conflict`] need from the session.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CtePolicy {
    /// How nested definitions with the same name resolve.
    pub(crate) precedence: CtePrecedencePolicy,
    /// Whether `same` folds case.
    pub(crate) case_sensitive: bool,
}

/// The inner `WITH` definition shadowing an outer one, if any: Spark's
/// `assertNoNameConflictsInCTE` over the unresolved plan. `outer` holds the
/// enclosing definitions' names; `start_of_query` is false inside the main
/// query of a `WITH` and in a plain child (a `FROM` subquery), where nested
/// definitions may shadow freely.
pub(crate) fn nested_conflict(plan: &Arc<LogicalPlan>, policy: CtePolicy) -> Option<AnalysisError> {
    conflict(plan, &[], true, policy).map(|(name, origin)| {
        AnalysisError::new(
            AmbiguousAliasInNestedCte {
                name: quoted(std::slice::from_ref(&name)),
                config: "\"spark.sql.legacy.ctePrecedencePolicy\"".to_owned(),
                docroot: SPARK_DOC_ROOT.to_owned(),
            },
            origin,
        )
    })
}

fn conflict(
    plan: &Arc<LogicalPlan>,
    outer: &[String],
    start_of_query: bool,
    policy: CtePolicy,
) -> Option<(String, Origin)> {
    let cs = policy.case_sensitive;
    match &plan.node {
        PlanNode::WithCte { input, ctes } => {
            let mut names: Vec<String> = outer.to_vec();
            for c in ctes {
                if start_of_query && outer.iter().any(|o| same(o, &c.name, cs)) {
                    return Some((c.name.clone(), plan.origin));
                }
                if let Some(hit) = conflict(&c.plan, &names, true, policy) {
                    return Some(hit);
                }
                names.push(c.name.clone());
            }
            conflict(input, &names, false, policy)
        }
        _ => {
            let mut hit = None;
            plan.for_each_expr(&mut |e| {
                if hit.is_none() {
                    hit = conflict_expr(e, outer, policy);
                }
            });
            if hit.is_some() {
                return hit;
            }
            plan.for_each_child(&mut |c| {
                if hit.is_none() {
                    hit = conflict(c, outer, false, policy);
                }
            });
            hit
        }
    }
}

/// [`conflict`] for an expression tree: subquery plans keep the enclosing
/// definitions and start a query.
fn conflict_expr(e: &Arc<Expr>, outer: &[String], policy: CtePolicy) -> Option<(String, Origin)> {
    if let ExprNode::Subquery { plan, .. } = &e.node
        && let Some(hit) = conflict(plan, outer, true, policy)
    {
        return Some(hit);
    }
    let mut hit = None;
    e.for_each_child(&mut |c| {
        if hit.is_none() {
            hit = conflict_expr(c, outer, policy);
        }
    });
    hit
}

/// The definitions in scope, oldest first; the newest match wins.
type Defs = Vec<(String, Arc<LogicalPlan>)>;

fn find<'a>(defs: &'a Defs, name: &str, cs: bool) -> Option<&'a Arc<LogicalPlan>> {
    defs.iter()
        .rev()
        .find(|(n, _)| same(n, name, cs))
        .map(|(_, p)| p)
}

/// Every `WithCte` inlined per [`super::CtePrecedencePolicy::Legacy`]: one
/// [`traverse_once`] pass eliminates the definitions reachable without
/// crossing another definition, exposing the nested ones for the next pass.
pub(crate) fn eliminate(plan: &Arc<LogicalPlan>, policy: CtePolicy) -> Arc<LogicalPlan> {
    let mut plan = Arc::clone(plan);
    for _ in 0..=crate::MAX_DEPTH {
        let (next, changed) = traverse_once(&plan, policy);
        plan = next;
        if !changed {
            break;
        }
    }
    eliminate_in_exprs(&plan, policy)
}

/// One legacy pass: bottom-up over plan children (definitions and subquery
/// plans are not children). A `WithCte` resolves each definition's body with
/// the previous siblings only, substitutes the bodies into the main query,
/// and is replaced by it.
fn traverse_once(plan: &Arc<LogicalPlan>, policy: CtePolicy) -> (Arc<LogicalPlan>, bool) {
    let cs = policy.case_sensitive;
    match &plan.node {
        PlanNode::WithCte { input, ctes } => {
            let (input, _) = traverse_once(input, policy);
            let mut defs: Defs = Vec::with_capacity(ctes.len());
            for c in ctes {
                let body = substitute(&c.plan, &defs, cs);
                defs.push((c.name.clone(), body));
            }
            (substitute(&input, &defs, cs), true)
        }
        _ => {
            let mut changed = false;
            let plan = sure(Arc::clone(plan).map_children(|c| {
                let (next, hit) = traverse_once(&c, policy);
                changed |= hit;
                Ok::<_, Infallible>(next)
            }));
            (plan, changed)
        }
    }
}

/// [`substitute`] reached a `Read` of one name: the newest definition body
/// in scope, if any.
fn substitute_read(name: &[String], defs: &Defs, cs: bool) -> Option<Arc<LogicalPlan>> {
    match name {
        [one] => find(defs, one, cs).cloned(),
        _ => None,
    }
}

/// A definition body or main query with `defs` inlined: references reachable
/// without crossing another definition are replaced, newest definition
/// first. Nested definitions' bodies are never entered; nested main queries
/// are, so an outer definition wins over a nested shadowing one.
fn substitute(plan: &Arc<LogicalPlan>, defs: &Defs, cs: bool) -> Arc<LogicalPlan> {
    let plan = match &plan.node {
        PlanNode::Read(ReadSource::Table { name, .. }) => {
            return substitute_read(name, defs, cs).unwrap_or_else(|| Arc::clone(plan));
        }
        PlanNode::WithCte { input, ctes } => {
            let next = substitute(input, defs, cs);
            if Arc::ptr_eq(&next, input) {
                Arc::clone(plan)
            } else {
                Arc::new(LogicalPlan {
                    node: PlanNode::WithCte {
                        input: next,
                        ctes: ctes.clone(),
                    },
                    origin: plan.origin,
                })
            }
        }
        _ => sure(Arc::clone(plan).map_children(|c| Ok::<_, Infallible>(substitute(&c, defs, cs)))),
    };
    sure(Arc::clone(&plan).map_expressions(|e| Ok::<_, Infallible>(substitute_expr(&e, defs, cs))))
}

/// An expression with `defs` inlined: subquery plans get the same
/// definitions, then one [`traverse_once`] pass, as Spark substitutes and
/// re-runs the rule inside subqueries.
fn substitute_expr(e: &Arc<Expr>, defs: &Defs, cs: bool) -> Arc<Expr> {
    if let ExprNode::Subquery { kind, plan } = &e.node {
        let subbed = substitute(plan, defs, cs);
        let (once, _) = traverse_once(
            &subbed,
            CtePolicy {
                precedence: CtePrecedencePolicy::Legacy,
                case_sensitive: cs,
            },
        );
        if Arc::ptr_eq(&once, plan) {
            return Arc::clone(e);
        }
        return Arc::new(Expr {
            node: ExprNode::Subquery {
                kind: kind.clone(),
                plan: once,
            },
            origin: e.origin,
        });
    }
    sure(Arc::clone(e).map_children(|c| Ok::<_, Infallible>(substitute_expr(&c, defs, cs))))
}

/// Whatever [`traverse_once`] cannot reach sits inside an expression (Spark
/// analyzes such subqueries separately, with no outer definitions in
/// scope): eliminate it standalone.
fn eliminate_in_exprs(plan: &Arc<LogicalPlan>, policy: CtePolicy) -> Arc<LogicalPlan> {
    let plan = sure(
        Arc::clone(plan).map_children(|c| Ok::<_, Infallible>(eliminate_in_exprs(&c, policy))),
    );
    sure(Arc::clone(&plan).map_expressions(|e| Ok::<_, Infallible>(eliminate_expr(&e, policy))))
}

/// [`eliminate`] for an expression tree.
fn eliminate_expr(e: &Arc<Expr>, policy: CtePolicy) -> Arc<Expr> {
    if let ExprNode::Subquery { kind, plan } = &e.node {
        let once = eliminate(plan, policy);
        if Arc::ptr_eq(&once, plan) {
            return Arc::clone(e);
        }
        return Arc::new(Expr {
            node: ExprNode::Subquery {
                kind: kind.clone(),
                plan: once,
            },
            origin: e.origin,
        });
    }
    sure(Arc::clone(e).map_children(|c| Ok::<_, Infallible>(eliminate_expr(&c, policy))))
}
