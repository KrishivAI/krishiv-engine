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
//!   SQL text lowered through [`super::SqlFrontend`] is inlined the same way.
//! * `EXCEPTION`: [`nested_conflict`] reports a nested definition shadowing
//!   an outer one, as Spark's `assertNoNameConflictsInCTE` does; resolution
//!   itself is the corrected one.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::sync::Arc;

use krishiv_common::error::{AmbiguousAliasInNestedCte, FailedToParseTooComplex, KrishivError};
use krishiv_common::stack::{STEP, on_big_stack};

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

/// Runs `f` on a fresh stack every [`STEP`] levels, so no pass over a plan
/// needs more than `STEP` frames of the caller's stack.
fn hop<T: Send>(
    depth: usize,
    f: impl FnOnce() -> Result<T, KrishivError> + Send,
) -> Result<T, KrishivError> {
    if depth > 0 && depth.is_multiple_of(STEP) {
        on_big_stack(f).unwrap_or_else(|| Err(FailedToParseTooComplex {}.into()))
    } else {
        f()
    }
}

/// The nodes [`conflict`] has finished, by address, with the scope they were
/// read in (`start_of_query` and the enclosing names decide the result). A
/// node met again cannot add a hit the first visit missed, so it is skipped:
/// a shared subtree is walked once per scope, not once per path. The plan
/// being walked keeps every address alive.
#[derive(Default)]
struct Seen {
    plans: HashSet<(usize, bool, Vec<String>)>,
    exprs: HashSet<(usize, Vec<String>)>,
}

/// The inner `WITH` definition shadowing an outer one, if any: Spark's
/// `assertNoNameConflictsInCTE` over the unresolved plan. `outer` holds the
/// enclosing definitions' names; `start_of_query` is false inside the main
/// query of a `WITH` and in a plain child (a `FROM` subquery), where nested
/// definitions may shadow freely.
pub(crate) fn nested_conflict(
    plan: &Arc<LogicalPlan>,
    policy: CtePolicy,
) -> Result<Option<AnalysisError>, AnalysisError> {
    let found = conflict(plan, &[], true, policy, 0, &mut Seen::default())
        .map_err(|e| AnalysisError::new(e, plan.origin))?;
    Ok(found.map(|(name, origin)| {
        AnalysisError::new(
            AmbiguousAliasInNestedCte {
                name: quoted(std::slice::from_ref(&name)),
                config: "\"spark.sql.legacy.ctePrecedencePolicy\"".to_owned(),
                docroot: SPARK_DOC_ROOT.to_owned(),
            },
            origin,
        )
    }))
}

type Hit = Option<(String, Origin)>;

fn conflict(
    plan: &Arc<LogicalPlan>,
    outer: &[String],
    start_of_query: bool,
    policy: CtePolicy,
    depth: usize,
    seen: &mut Seen,
) -> Result<Hit, KrishivError> {
    if depth > crate::MAX_DEPTH {
        return Err(FailedToParseTooComplex {}.into());
    }
    if !seen
        .plans
        .insert((Arc::as_ptr(plan) as usize, start_of_query, outer.to_vec()))
    {
        return Ok(None);
    }
    hop(depth, || {
        let cs = policy.case_sensitive;
        match &plan.node {
            PlanNode::WithCte { input, ctes } => {
                let mut names: Vec<String> = outer.to_vec();
                for c in ctes {
                    if start_of_query && outer.iter().any(|o| same(o, &c.name, cs)) {
                        return Ok(Some((c.name.clone(), plan.origin)));
                    }
                    if let Some(hit) = conflict(&c.plan, &names, true, policy, depth + 1, seen)? {
                        return Ok(Some(hit));
                    }
                    names.push(c.name.clone());
                }
                conflict(input, &names, false, policy, depth + 1, seen)
            }
            _ => {
                let mut exprs = Vec::new();
                plan.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
                for e in &exprs {
                    if let Some(hit) = conflict_expr(e, outer, policy, depth + 1, seen)? {
                        return Ok(Some(hit));
                    }
                }
                let mut children = Vec::new();
                plan.for_each_child(&mut |c| children.push(Arc::clone(c)));
                for c in &children {
                    if let Some(hit) = conflict(c, outer, false, policy, depth + 1, seen)? {
                        return Ok(Some(hit));
                    }
                }
                Ok(None)
            }
        }
    })
}

/// [`conflict`] for an expression tree: subquery plans keep the enclosing
/// definitions and start a query.
fn conflict_expr(
    e: &Arc<Expr>,
    outer: &[String],
    policy: CtePolicy,
    depth: usize,
    seen: &mut Seen,
) -> Result<Hit, KrishivError> {
    if depth > crate::MAX_DEPTH {
        return Err(FailedToParseTooComplex {}.into());
    }
    if !seen.exprs.insert((Arc::as_ptr(e) as usize, outer.to_vec())) {
        return Ok(None);
    }
    hop(depth, || {
        if let ExprNode::Subquery { plan, .. } = &e.node
            && let Some(hit) = conflict(plan, outer, true, policy, depth + 1, seen)?
        {
            return Ok(Some(hit));
        }
        let mut children = Vec::new();
        e.for_each_child(&mut |c| children.push(Arc::clone(c)));
        for c in &children {
            if let Some(hit) = conflict_expr(c, outer, policy, depth + 1, seen)? {
                return Ok(Some(hit));
            }
        }
        Ok(None)
    })
}

/// The definitions in scope, oldest first; the newest match wins.
type Defs = Vec<(String, Arc<LogicalPlan>)>;

fn find<'a>(defs: &'a Defs, name: &str, cs: bool) -> Option<&'a Arc<LogicalPlan>> {
    defs.iter()
        .rev()
        .find(|(n, _)| same(n, name, cs))
        .map(|(_, p)| p)
}

/// Results of one rewrite pass, keyed by node address: the inlined plan is a
/// DAG (a definition read k times is one `Arc` in k places), so a walk
/// without a memo is exponential in the nesting. Each entry holds its key's
/// `Arc`, which keeps the address from being reused.
#[derive(Default)]
struct Memo {
    plans: HashMap<usize, (Arc<LogicalPlan>, Arc<LogicalPlan>)>,
    exprs: HashMap<usize, (Arc<Expr>, Arc<Expr>)>,
}

impl Memo {
    fn plan(&self, p: &Arc<LogicalPlan>) -> Option<Arc<LogicalPlan>> {
        self.plans
            .get(&(Arc::as_ptr(p) as usize))
            .map(|(_, out)| Arc::clone(out))
    }

    fn put_plan(&mut self, from: &Arc<LogicalPlan>, to: &Arc<LogicalPlan>) {
        self.plans.insert(
            Arc::as_ptr(from) as usize,
            (Arc::clone(from), Arc::clone(to)),
        );
    }

    fn expr(&self, e: &Arc<Expr>) -> Option<Arc<Expr>> {
        self.exprs
            .get(&(Arc::as_ptr(e) as usize))
            .map(|(_, out)| Arc::clone(out))
    }

    fn put_expr(&mut self, from: &Arc<Expr>, to: &Arc<Expr>) {
        self.exprs.insert(
            Arc::as_ptr(from) as usize,
            (Arc::clone(from), Arc::clone(to)),
        );
    }
}

/// Every `WithCte` inlined per [`super::CtePrecedencePolicy::Legacy`]: one
/// [`traverse_once`] pass eliminates the definitions reachable without
/// crossing another definition, exposing the nested ones for the next pass.
/// A definition whose body reads its own name is not an error here (Spark
/// raises `RECURSIVE_CTE_IN_LEGACY_MODE` only for `WITH RECURSIVE`): the name
/// resolves to an outer definition substituted later, or to a table.
///
/// # Errors
/// `FAILED_TO_PARSE_TOO_COMPLEX` when inlining nests deeper than
/// [`crate::MAX_DEPTH`].
pub(crate) fn eliminate(
    plan: &Arc<LogicalPlan>,
    policy: CtePolicy,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    eliminate_at(plan, policy, 0)
}

/// [`eliminate`] for a plan `depth` levels below the root of the walk.
fn eliminate_at(
    plan: &Arc<LogicalPlan>,
    policy: CtePolicy,
    depth: usize,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let mut plan = Arc::clone(plan);
    let mut settled = false;
    for _ in 0..=crate::MAX_DEPTH {
        let (next, changed) = traverse_once(&plan, policy, depth, &mut Memo::default())?;
        plan = next;
        if !changed {
            settled = true;
            break;
        }
    }
    if !settled {
        return Err(FailedToParseTooComplex {}.into());
    }
    let out = eliminate_in_exprs(&plan, policy, depth, &mut Memo::default())?;
    crate::check_depth(&out)?;
    Ok(out)
}

/// One legacy pass: bottom-up over plan children (definitions and subquery
/// plans are not children). A `WithCte` resolves each definition's body with
/// the previous siblings only, substitutes the bodies into the main query,
/// and is replaced by it.
fn traverse_once(
    plan: &Arc<LogicalPlan>,
    policy: CtePolicy,
    depth: usize,
    memo: &mut Memo,
) -> Result<(Arc<LogicalPlan>, bool), KrishivError> {
    hop(depth, || traverse_once_body(plan, policy, depth, memo))
}

fn traverse_once_body(
    plan: &Arc<LogicalPlan>,
    policy: CtePolicy,
    depth: usize,
    memo: &mut Memo,
) -> Result<(Arc<LogicalPlan>, bool), KrishivError> {
    // A node met again in this pass was counted when first rewritten.
    if let Some(hit) = memo.plan(plan) {
        return Ok((hit, false));
    }
    if depth > crate::MAX_DEPTH {
        return Err(FailedToParseTooComplex {}.into());
    }
    let cs = policy.case_sensitive;
    match &plan.node {
        PlanNode::WithCte { input, ctes } => {
            let (input, _) = traverse_once(input, policy, depth + 1, memo)?;
            let mut defs: Defs = Vec::with_capacity(ctes.len());
            for c in ctes {
                let body = substitute(&c.plan, &defs, cs, depth + 1, &mut Memo::default())?;
                crate::check_depth(&body)?;
                defs.push((c.name.clone(), body));
            }
            let out = substitute(&input, &defs, cs, depth + 1, &mut Memo::default())?;
            crate::check_depth(&out)?;
            memo.put_plan(plan, &out);
            Ok((out, true))
        }
        _ => {
            let original = Arc::clone(plan);
            let mut changed = false;
            let mut err: Option<KrishivError> = None;
            let plan = sure(Arc::clone(plan).map_children(|c| {
                match traverse_once(&c, policy, depth + 1, memo) {
                    Ok((next, hit)) => {
                        changed |= hit;
                        Ok::<_, Infallible>(next)
                    }
                    Err(e) => {
                        err = Some(e);
                        Ok::<_, Infallible>(Arc::clone(&c))
                    }
                }
            }));
            if let Some(e) = err {
                return Err(e);
            }
            memo.put_plan(&original, &plan);
            Ok((plan, changed))
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
/// are, so an outer definition wins over a nested shadowing one. `memo` is
/// for this `defs` only.
fn substitute(
    plan: &Arc<LogicalPlan>,
    defs: &Defs,
    cs: bool,
    depth: usize,
    memo: &mut Memo,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    hop(depth, || substitute_body(plan, defs, cs, depth, memo))
}

fn substitute_body(
    plan: &Arc<LogicalPlan>,
    defs: &Defs,
    cs: bool,
    depth: usize,
    memo: &mut Memo,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    if let Some(hit) = memo.plan(plan) {
        return Ok(hit);
    }
    if depth > crate::MAX_DEPTH {
        return Err(FailedToParseTooComplex {}.into());
    }
    let plan_in = plan;
    let plan = match &plan.node {
        PlanNode::Read(ReadSource::Table { name, .. }) => {
            let out = substitute_read(name, defs, cs).unwrap_or_else(|| Arc::clone(plan));
            memo.put_plan(plan_in, &out);
            return Ok(out);
        }
        PlanNode::WithCte { input, ctes } => {
            let next = substitute(input, defs, cs, depth + 1, memo)?;
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
        _ => {
            let mut err: Option<KrishivError> = None;
            let out = sure(Arc::clone(plan).map_children(|c| {
                match substitute(&c, defs, cs, depth + 1, memo) {
                    Ok(next) => Ok::<_, Infallible>(next),
                    Err(e) => {
                        err = Some(e);
                        Ok::<_, Infallible>(Arc::clone(&c))
                    }
                }
            }));
            if let Some(e) = err {
                return Err(e);
            }
            out
        }
    };
    let mut err: Option<KrishivError> = None;
    let out = sure(Arc::clone(&plan).map_expressions(|ex| {
        match substitute_expr(&ex, defs, cs, depth + 1, memo) {
            Ok(next) => Ok::<_, Infallible>(next),
            Err(fail) => {
                err = Some(fail);
                Ok::<_, Infallible>(Arc::clone(&ex))
            }
        }
    }));
    if let Some(e) = err {
        return Err(e);
    }
    memo.put_plan(plan_in, &out);
    Ok(out)
}

/// An expression with `defs` inlined: subquery plans get the same
/// definitions, then one [`traverse_once`] pass, as Spark substitutes and
/// re-runs the rule inside subqueries.
fn substitute_expr(
    e: &Arc<Expr>,
    defs: &Defs,
    cs: bool,
    depth: usize,
    memo: &mut Memo,
) -> Result<Arc<Expr>, KrishivError> {
    hop(depth, || substitute_expr_body(e, defs, cs, depth, memo))
}

fn substitute_expr_body(
    e: &Arc<Expr>,
    defs: &Defs,
    cs: bool,
    depth: usize,
    memo: &mut Memo,
) -> Result<Arc<Expr>, KrishivError> {
    if let Some(hit) = memo.expr(e) {
        return Ok(hit);
    }
    if depth > crate::MAX_DEPTH {
        return Err(FailedToParseTooComplex {}.into());
    }
    let out = if let ExprNode::Subquery { kind, plan } = &e.node {
        let subbed = substitute(plan, defs, cs, depth + 1, memo)?;
        let (once, _) = traverse_once(
            &subbed,
            CtePolicy {
                precedence: CtePrecedencePolicy::Legacy,
                case_sensitive: cs,
            },
            depth + 1,
            &mut Memo::default(),
        )?;
        if Arc::ptr_eq(&once, plan) {
            Arc::clone(e)
        } else {
            Arc::new(Expr {
                node: ExprNode::Subquery {
                    kind: kind.clone(),
                    plan: once,
                },
                origin: e.origin,
            })
        }
    } else {
        let mut err: Option<KrishivError> = None;
        let out = sure(Arc::clone(e).map_children(|c| {
            match substitute_expr(&c, defs, cs, depth + 1, memo) {
                Ok(next) => Ok::<_, Infallible>(next),
                Err(e) => {
                    err = Some(e);
                    Ok::<_, Infallible>(Arc::clone(&c))
                }
            }
        }));
        if let Some(e) = err {
            return Err(e);
        }
        out
    };
    memo.put_expr(e, &out);
    Ok(out)
}

/// Whatever [`traverse_once`] cannot reach sits inside an expression (Spark
/// analyzes such subqueries separately, with no outer definitions in
/// scope): eliminate it standalone.
fn eliminate_in_exprs(
    plan: &Arc<LogicalPlan>,
    policy: CtePolicy,
    depth: usize,
    memo: &mut Memo,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    hop(depth, || eliminate_in_exprs_body(plan, policy, depth, memo))
}

fn eliminate_in_exprs_body(
    plan: &Arc<LogicalPlan>,
    policy: CtePolicy,
    depth: usize,
    memo: &mut Memo,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    if let Some(hit) = memo.plan(plan) {
        return Ok(hit);
    }
    if depth > crate::MAX_DEPTH {
        return Err(FailedToParseTooComplex {}.into());
    }
    let mut err: Option<KrishivError> = None;
    let rebuilt = sure(Arc::clone(plan).map_children(|c| {
        match eliminate_in_exprs(&c, policy, depth + 1, memo) {
            Ok(next) => Ok::<_, Infallible>(next),
            Err(e) => {
                err = Some(e);
                Ok::<_, Infallible>(Arc::clone(&c))
            }
        }
    }));
    if let Some(e) = err {
        return Err(e);
    }
    let mut err: Option<KrishivError> = None;
    let out = sure(Arc::clone(&rebuilt).map_expressions(|ex| {
        match eliminate_expr(&ex, policy, depth + 1, memo) {
            Ok(next) => Ok::<_, Infallible>(next),
            Err(fail) => {
                err = Some(fail);
                Ok::<_, Infallible>(Arc::clone(&ex))
            }
        }
    }));
    if let Some(e) = err {
        return Err(e);
    }
    memo.put_plan(plan, &out);
    Ok(out)
}

/// [`eliminate`] for an expression tree.
fn eliminate_expr(
    e: &Arc<Expr>,
    policy: CtePolicy,
    depth: usize,
    memo: &mut Memo,
) -> Result<Arc<Expr>, KrishivError> {
    hop(depth, || eliminate_expr_body(e, policy, depth, memo))
}

fn eliminate_expr_body(
    e: &Arc<Expr>,
    policy: CtePolicy,
    depth: usize,
    memo: &mut Memo,
) -> Result<Arc<Expr>, KrishivError> {
    if let Some(hit) = memo.expr(e) {
        return Ok(hit);
    }
    if depth > crate::MAX_DEPTH {
        return Err(FailedToParseTooComplex {}.into());
    }
    let out = if let ExprNode::Subquery { kind, plan } = &e.node {
        let once = eliminate_at(plan, policy, depth + 1)?;
        if Arc::ptr_eq(&once, plan) {
            Arc::clone(e)
        } else {
            Arc::new(Expr {
                node: ExprNode::Subquery {
                    kind: kind.clone(),
                    plan: once,
                },
                origin: e.origin,
            })
        }
    } else {
        let mut err: Option<KrishivError> = None;
        let out = sure(Arc::clone(e).map_children(|c| {
            match eliminate_expr(&c, policy, depth + 1, memo) {
                Ok(next) => Ok::<_, Infallible>(next),
                Err(e) => {
                    err = Some(e);
                    Ok::<_, Infallible>(Arc::clone(&c))
                }
            }
        }));
        if let Some(e) = err {
            return Err(e);
        }
        out
    };
    memo.put_expr(e, &out);
    Ok(out)
}
