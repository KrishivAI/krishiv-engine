// SPDX-License-Identifier: Apache-2.0
//! Stage 2, Resolve: one bottom-up pass over the relations.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use krishiv_common::error::{FailedToParseTooComplex, TableOrViewNotFound};
use krishiv_common::stack::on_big_stack;

use super::bind::{Bindings, BoundTable, addr, same, unsupported};
use super::catalog::quoted;
use super::exprs::{Bound, Cx};
use super::ops::col_expr;
use super::{AnalysisContext, AnalysisError, SecurableUse};
use crate::{
    AttrId, BoundColumn, Cte, Expr, ExprNode, LogicalPlan, Origin, Plan, PlanId, PlanNode,
    ReadSource,
};

/// Relations resolved on the caller's stack before hopping to a fresh one.
///
/// Each nesting level keeps the frames of `rel`, `rel_on_stack`, `node`,
/// the operator helper, `bind`, `expr` and `expr_node` alive at once, and
/// debug and coverage-instrumented builds inflate those frames well past
/// their release size: 64 levels overflow a 2 MiB stack in debug builds and
/// abort under `cargo llvm-cov`. Hopping every 16 levels keeps at most 16
/// levels on the caller's stack; deeper levels run on fresh 64 MiB
/// segments. The test thread stays at 2 MiB, so deep plans are still proven
/// to resolve on a small stack.
pub(super) const RESOLVE_STEP: usize = 16;

/// An attribute in scope: the column, its qualifier and, for a table
/// column, the securable it belongs to and its catalog name.
#[derive(Debug, Clone)]
pub(super) struct Attr {
    pub(super) col: BoundColumn,
    pub(super) qual: Arc<[String]>,
    pub(super) source: Option<(usize, Arc<str>)>,
    /// A union input's re-aliased repeat of an earlier column of that input
    /// (`Resolver::distinct_output_ids`): Spark's `__is_duplicate`
    /// metadata. A name that matches it and another column resolves to the
    /// other. It follows the attribute wherever the attribute is copied
    /// (instances, CTE references, column aliases, an alias of the bare
    /// column); a new attribute computed from it does not have it.
    pub(super) duplicate: bool,
}

/// What an aggregate needs to resolve `HAVING` / `ORDER BY` over it.
#[derive(Clone)]
pub(super) struct AggInfo {
    pub(super) input: Rel,
    pub(super) grouping: Vec<Arc<Expr>>,
    /// The aggregate's own expressions and output, aligned.
    pub(super) aggregates: Vec<Arc<Expr>>,
    pub(super) out: Vec<Attr>,
    /// The grouping attributes, as attributes of the input.
    pub(super) keys: Vec<Attr>,
    /// Select items that use lateral column aliases or windows make
    /// projections (and window relations) above the aggregate: every
    /// select item, over the aggregate's output (`None` when no item
    /// does).
    pub(super) top: Option<Arc<[Bound]>>,
    /// The grouping sets, for an aggregate over `ROLLUP`, `CUBE` or
    /// `GROUPING SETS`: its input is then their expansion.
    pub(super) analytics: Option<Arc<super::analytics::Analytics>>,
    /// With windows: the `HAVING` filtering the aggregate below them.
    pub(super) having: Option<Arc<Expr>>,
    /// The second aggregate of `RewriteDistinctAggregates`: its input is the first aggregate's
    /// output, one row per distinct set, so an aggregate added to it later (a `HAVING` or `ORDER BY`
    /// call the select list does not hold) would run over those rows, not the query's.
    pub(super) distinct_rewritten: bool,
}

impl AggInfo {
    /// Whether the select list holds windows.
    pub(super) fn has_windows(&self) -> bool {
        self.top
            .as_ref()
            .is_some_and(|t| t.iter().any(|b| b.win.is_some()))
    }
}

/// How a relation gets back columns it dropped: Spark adds the attributes
/// an `ORDER BY` or a DataFrame filter names, but a projection below
/// dropped, to that projection (`ResolveMissingReferences`).
pub(super) enum Widen {
    /// A projection of `exprs` over `input`.
    Project { input: Rel, exprs: Vec<Arc<Expr>> },
    /// A filter, sort or limit over `child`: its columns pass through.
    Through { child: Rel },
    /// A pipe boundary over `child` (the same plan): only `child`'s hidden
    /// columns (a `USING` join's merged keys, a `_metadata`) come back
    /// through it, never a column a projection below dropped.
    Hidden { child: Rel },
    /// An aggregate (under the projections of its lateral aliases): it
    /// adds grouping attributes to its output.
    Aggregate(Arc<AggInfo>),
    /// A file scan: it adds its hidden `_metadata` column to its output.
    Scan,
    /// `inner` with every attribute id shifted by `offset` (the second
    /// side of a self-join): widening shifts what `inner` widens to.
    Shift { inner: Rel, offset: u64 },
    /// A join of `left` and `right` (their `hidden` kept): the hidden
    /// `_metadata` columns of either input come from widening it.
    Join {
        left: Rel,
        right: Rel,
        kind: crate::JoinKind,
        condition: crate::JoinCondition,
        /// Whether the left and right outputs turn nullable.
        nullable: (bool, bool),
    },
}

/// A resolved relation and its output scope.
#[derive(Clone)]
pub(super) struct Rel {
    pub(super) plan: Arc<LogicalPlan>,
    pub(super) out: Arc<[Attr]>,
    /// The span of attribute ids defined in the subtree.
    pub(super) ids: Option<(u64, u64)>,
    /// The outermost query level the subtree references, if any.
    pub(super) outer: Option<usize>,
    /// The outputs of the relations carrying each `plan_id` below.
    pub(super) carriers: Arc<BTreeMap<PlanId, Vec<Arc<[Attr]>>>>,
    pub(super) agg: Option<Arc<AggInfo>>,
    /// Where dropped columns come back from, if anywhere.
    pub(super) widen: Option<Arc<Widen>>,
    /// Attributes the output hides but a qualified name (`t.k`, `t.*`)
    /// still reaches: the key columns a `USING` join merges (Spark's
    /// hidden output). Filters, sorts and limits pass them on; a
    /// projection drops them.
    pub(super) hidden: Arc<[Attr]>,
    /// The aggregate and generator this relation is the generation of: a
    /// `HAVING` over it filters the aggregate, below its windows.
    pub(super) generated_from: Option<Arc<super::generate::GeneratedFrom>>,
}

/// An enclosing query's operator, as a subquery sees it.
#[derive(Clone)]
pub(super) struct Level {
    pub(super) scope: Arc<[Attr]>,
    /// The left side of a `JOIN LATERAL`: unlike a subquery's enclosing
    /// query, it stays in reach from further in.
    pub(super) lateral: bool,
    /// The columns a `USING` join of that query merged away: not
    /// resolvable by name (a lateral right side reaches them by a qualified
    /// name), but a qualified star still lists them.
    pub(super) hidden: Arc<[Attr]>,
    /// The input columns of the aggregate a `HAVING` subquery sits in that its
    /// output does not carry, searched first: Spark binds an outer name to the
    /// aggregate input before an output alias of the same name.
    pub(super) fallback: Arc<[Attr]>,
    /// The input of the aggregate whose `HAVING` the subquery sits in: an
    /// aggregate call of the subquery's `WHERE` over only these columns is
    /// that aggregate's (Spark's outer aggregate), not the subquery's.
    pub(super) having: Option<Arc<[Attr]>>,
}

/// A scope environment: the CTEs visible and the enclosing queries.
pub(super) struct Env {
    pub(super) id: usize,
    pub(super) ctes: Vec<(String, Rel)>,
    /// Identifies `ctes`: environments with one CTE scope resolve a
    /// relation without outer references alike.
    pub(super) ctes_id: usize,
    pub(super) outer: Vec<Level>,
    /// The span of the ids the enclosing queries define.
    pub(super) outer_ids: Option<(u64, u64)>,
}

pub(super) struct Resolver<'a> {
    pub(super) ctx: &'a AnalysisContext,
    /// The context with `spark.sql.ansi.enabled` inverted, swapped in with
    /// `ctx` while the body of a function captured under the other mode
    /// resolves.
    pub(super) flipped: &'a AnalysisContext,
    pub(super) b: &'a Bindings,
    pub(super) next_id: u64,
    next_env: usize,
    /// The recursive CTEs being resolved, innermost last: each loop's id
    /// and the attributes its recursive term reads (see `recursive`).
    pub(super) loops: Vec<(u32, Vec<Attr>)>,
    pub(super) memo: HashMap<(usize, usize), Rel>,
    /// Relations without outer references, by (relation, CTE scope): they
    /// resolve alike in every environment with that CTE scope, so a plan
    /// reached both as a child and inside a subquery is resolved once.
    pub(super) closed: HashMap<(usize, usize), Rel>,
    /// Each relation's representative among the relations equal to it
    /// (`encode::identity_keys`): resolution does not depend on how the
    /// input shares subtrees, so equal input keys give equal plans.
    pub(super) canon: HashMap<usize, Arc<LogicalPlan>>,
    /// The identity key of every relation of the input, kept from the first `canonicalize` (the root's).
    pub(super) keys: HashMap<usize, u128>,
    /// The input holds a node the prefix cache does not serve (a resolved scan, SQL text, a read by path).
    pub(super) uncacheable: bool,
    /// The names of the tables the input reads, as written.
    pub(super) read_tables: Vec<Vec<String>>,
    /// One shared expression per column id, used wherever a projection passes that column through
    /// unchanged. A chain of `withColumn` steps then holds one expression per column instead of one per
    /// column per step, and passes that skip an expression already seen skip it.
    pub(super) pass_exprs: HashMap<AttrId, Arc<crate::Expr>>,
    pub(super) columns: Vec<BTreeSet<String>>,
    /// The lambda scopes pushed by the higher-order calls whose bodies
    /// bind now, innermost last (see `hof`).
    pub(super) lambdas: Vec<super::hof::LambdaFrame>,
    depth: usize,
    /// Expression-binding recursion through higher-order calls (see
    /// `exprs::expr_in`): every [`RESOLVE_STEP`] levels run on a fresh
    /// stack.
    pub(super) expr_depth: usize,
    /// Nodes under the key argument of an AES call (ADR 0042), shared by
    /// every nested `expr_in` (lambda bodies, subqueries) so the secret
    /// context is inherited on all analysis paths; cleared when the
    /// outermost expression finishes. `secret_scope` is set while a secret
    /// node's children are bound, so a nested pass marks its root.
    pub(super) secret: std::collections::HashSet<usize>,
    pub(super) secret_scope: bool,
    op_outer: Option<usize>,
    root_env: Option<Arc<Env>>,
    /// Constant folding's values over this pass (see `fold::Memo`).
    pub(super) folds: std::cell::RefCell<super::fold::Memo>,
    /// The SQL function whose body is being resolved: what its parameters
    /// stand for (see `sqlfn`).
    pub(super) frame: Option<Arc<super::sqlfn::Frame>>,
    /// The SQL functions open, outermost first, to catch recursion.
    pub(super) fn_path: Vec<String>,
    /// SQL function calls inlined so far in this analysis (a budget).
    pub(super) sql_calls: usize,
    /// The first aggregate over a nondeterministic argument: Spark checks
    /// it after the operators' other rules, so any other error wins.
    pub(super) pending_nondeterministic: Option<AnalysisError>,
    /// The attributes that hold a `CHAR(n)` / `VARCHAR(n)` column, by id, with the catalog string they were
    /// declared with (see `char_varchar`).
    pub(super) declared: HashMap<AttrId, Arc<str>>,
    /// The attributes that carry a column comment, by id (Spark's `comment` field metadata; see `comments`).
    pub(super) comments: HashMap<AttrId, Arc<str>>,
}

type Resolved = (
    Arc<LogicalPlan>,
    Vec<BoundColumn>,
    Vec<SecurableUse>,
    Vec<String>,
    Vec<Option<String>>,
    Vec<Option<String>>,
    Option<super::prefix::PrefixEntry>,
);

pub(super) fn resolve(
    plan: &Plan,
    ctx: &AnalysisContext,
    b: &Bindings,
    warm: Option<&super::prefix::PrefixEntry>,
) -> Result<Resolved, AnalysisError> {
    let mut other = ctx.clone();
    other.conf.ansi = !ctx.conf.ansi;
    let mut r = Resolver {
        ctx,
        flipped: &other,
        b,
        next_id: 0,
        next_env: 0,
        loops: Vec::new(),
        memo: HashMap::new(),
        closed: HashMap::new(),
        canon: HashMap::new(),
        keys: HashMap::new(),
        uncacheable: false,
        read_tables: Vec::new(),
        pass_exprs: HashMap::new(),
        columns: vec![BTreeSet::new(); b.securables.len()],
        depth: 0,
        expr_depth: 0,
        secret: std::collections::HashSet::new(),
        secret_scope: false,
        op_outer: None,
        root_env: None,
        folds: std::cell::RefCell::default(),
        lambdas: Vec::new(),
        frame: None,
        fn_path: Vec::new(),
        sql_calls: 0,
        pending_nondeterministic: None,
        declared: HashMap::new(),
        comments: HashMap::new(),
    };
    // Fresh ids start past every id the input already holds (a plan that
    // is resolved in part, or re-analysed).
    r.next_id = first_free_id(plan.root())?;
    let start_id = r.next_id;
    let env = r.env(Vec::new(), 0, Vec::new());
    r.root_env = Some(Arc::clone(&env));
    r.canonicalize(plan.root());
    if let Some(entry) = warm {
        r.seed_prefix(entry, plan.root(), &env);
    }
    let rel = r.rel(plan.root(), &env)?;
    if let Some(error) = r.pending_nondeterministic.take() {
        return Err(error);
    }
    let entry = r.export_prefix(plan.root(), &rel, start_id);
    // Hints name relations of their resolved subtree.
    let (root, warnings) = super::hints::resolve(&rel.plan, ctx.conf.case_sensitive)?;
    // Subqueries to joins, after everything resolved (P2-T071c).
    let root = super::subquery::rewrite(
        &root,
        &mut r.next_id,
        &*ctx.functions,
        ctx.profile,
        &ctx.conf,
    )?;
    let output = rel.out.iter().map(|a| a.col.clone()).collect();
    // The output columns that are `CHAR(n)` / `VARCHAR(n)` columns (none under `charVarcharAsString`).
    let declared = rel
        .out
        .iter()
        .map(|a| {
            let kept = (!ctx.conf.char_varchar_as_string).then(|| r.declared.get(&a.col.id));
            kept.flatten().map(ToString::to_string)
        })
        .collect();
    let comments = rel
        .out
        .iter()
        .map(|a| r.comments.get(&a.col.id).map(ToString::to_string))
        .collect();
    let uses = b
        .securables
        .iter()
        .zip(r.columns)
        .map(|((securable, privilege), columns)| SecurableUse {
            securable: securable.clone(),
            privilege: *privilege,
            columns,
        })
        .collect();
    Ok((root, output, uses, warnings, declared, comments, entry))
}

/// One past the largest attribute id in `root` (a `Shift` counted at its
/// offset).
fn first_free_id(root: &LogicalPlan) -> Result<u64, AnalysisError> {
    use crate::walk::{Node, visit_distinct};
    use crate::{ExprNode, ResolvedExpr};
    let mut top = 0_u64;
    let mut shifts = 0_u64;
    let _ = visit_distinct(Node::Plan(root), |node| {
        let mut bump = |c: &BoundColumn| top = top.max(c.id.0.saturating_add(1));
        match node {
            Node::Plan(p) => match &p.node {
                PlanNode::Scan(s) => s.output.iter().for_each(&mut bump),
                PlanNode::CteRef(r) => r.output.iter().for_each(&mut bump),
                PlanNode::Shift { offset, .. } => shifts = shifts.saturating_add(*offset),
                // The `exists` column is defined by the join alone: an
                // unread one appears nowhere else.
                PlanNode::Join {
                    kind: crate::JoinKind::Existence(id),
                    ..
                } => top = top.max(id.0.saturating_add(1)),
                _ => {}
            },
            Node::Expr(e) => {
                if let ExprNode::Resolved(
                    ResolvedExpr::Column(c)
                    | ResolvedExpr::OuterColumn(c)
                    | ResolvedExpr::LambdaVar(c)
                    | ResolvedExpr::Alias { output: c, .. },
                ) = &e.node
                {
                    bump(c);
                }
                if let ExprNode::Lambda { params, .. } = &e.node {
                    // Unresolved lambdas carry placeholder identities (see
                    // `Expr::lambda`); only bound variables reserve ids.
                    params
                        .iter()
                        .filter(|c| c.id.0 != u64::MAX)
                        .for_each(&mut bump);
                }
            }
        }
        std::ops::ControlFlow::Continue(())
    });
    let next = top.saturating_add(shifts);
    if next > super::ops::MAX_ID {
        return Err(AnalysisError::new(FailedToParseTooComplex {}, root.origin));
    }
    Ok(next)
}

pub(super) fn hull(a: Option<(u64, u64)>, b: Option<(u64, u64)>) -> Option<(u64, u64)> {
    match (a, b) {
        (Some(x), Some(y)) => Some((x.0.min(y.0), x.1.max(y.1))),
        (x, None) | (None, x) => x,
    }
}

/// A CTE scope: every visible CTE (outer ones as `None`, this level's as
/// `Some(index)`); inner `WITH` names shadow as `None`.
type CteScope = Vec<(String, Option<usize>)>;

impl Resolver<'_> {
    pub(super) fn fresh(&mut self) -> AttrId {
        self.next_id += 1;
        AttrId(self.next_id - 1)
    }

    pub(super) fn env(
        &mut self,
        ctes: Vec<(String, Rel)>,
        ctes_id: usize,
        outer: Vec<Level>,
    ) -> Arc<Env> {
        self.next_env += 1;
        let outer_ids = outer
            .iter()
            .flat_map(|l| l.scope.iter())
            .fold(None, |h, a| hull(h, Some((a.col.id.0, a.col.id.0))));
        Arc::new(Env {
            id: self.next_env,
            ctes,
            ctes_id,
            outer,
            outer_ids,
        })
    }

    /// A new CTE scope id.
    pub(super) fn ctes_id(&mut self) -> usize {
        self.next_env += 1;
        self.next_env
    }

    /// Records the representative of every relation under `root`.
    pub(super) fn canonicalize(&mut self, root: &Arc<LogicalPlan>) {
        if self.canon.contains_key(&addr(&**root)) {
            return;
        }
        let keys = crate::encode::identity_keys(root);
        let first_call = self.keys.is_empty();
        let mut first: HashMap<u128, Arc<LogicalPlan>> = HashMap::new();
        let mut stack = vec![Arc::clone(root)];
        let mut exprs: Vec<Arc<crate::Expr>> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        while let Some(p) = stack.pop() {
            if !seen.insert(addr(&*p)) {
                continue;
            }
            let key = keys.get(&std::ptr::from_ref(&*p)).copied();
            let rep = match key {
                Some(k) => Arc::clone(first.entry(k).or_insert_with(|| Arc::clone(&p))),
                None => Arc::clone(&p),
            };
            self.canon.insert(addr(&*p), rep);
            if first_call {
                match &p.node {
                    PlanNode::Scan(_)
                    | PlanNode::Sql { .. }
                    | PlanNode::Read(ReadSource::Files { .. }) => self.uncacheable = true,
                    PlanNode::Read(ReadSource::Table { name, .. }) => {
                        self.read_tables.push(name.clone());
                    }
                    _ => {}
                }
            }
            p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
            p.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
            while let Some(e) = exprs.pop() {
                if !seen.insert(addr(&*e)) {
                    continue;
                }
                if let crate::ExprNode::Subquery { plan, .. } = &e.node {
                    stack.push(Arc::clone(plan));
                }
                e.for_each_child(&mut |c| exprs.push(Arc::clone(c)));
            }
        }
        if first_call {
            self.keys = keys.into_iter().map(|(p, k)| (p as usize, k)).collect();
        }
    }

    /// The shared expression that passes `col` through a projection (see `pass_exprs`).
    pub(super) fn pass_expr(&mut self, col: &BoundColumn) -> Arc<Expr> {
        let shared = self
            .pass_exprs
            .entry(col.id)
            .or_insert_with(|| col_expr(col, Origin::UNKNOWN));
        // The same id names the same column; a different shape builds its own expression.
        match &shared.node {
            ExprNode::Resolved(crate::ResolvedExpr::Column(c)) if c == col => Arc::clone(shared),
            _ => col_expr(col, Origin::UNKNOWN),
        }
    }

    pub(super) fn rep(&self, p: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        self.canon
            .get(&addr(&**p))
            .cloned()
            .unwrap_or_else(|| Arc::clone(p))
    }

    /// Resolves the relation `root` in `env`, every [`RESOLVE_STEP`]
    /// nested calls on a fresh stack.
    pub(super) fn rel(
        &mut self,
        root: &Arc<LogicalPlan>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        self.depth += 1;
        let r = if self.depth.is_multiple_of(RESOLVE_STEP) {
            on_big_stack(|| self.rel_on_stack(root, env))
                .unwrap_or_else(|| Err(AnalysisError::new(FailedToParseTooComplex {}, root.origin)))
        } else {
            self.rel_on_stack(root, env)
        };
        self.depth -= 1;
        r
    }

    /// Children first, with an explicit stack; one resolution per
    /// (relation, environment).
    fn rel_on_stack(
        &mut self,
        root: &Arc<LogicalPlan>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        self.canonicalize(root);
        let root = self.rep(root);
        let mut stack = vec![(Arc::clone(&root), false)];
        while let Some((p, expanded)) = stack.pop() {
            let key = (addr(&*p), env.id);
            if self.memo.contains_key(&key) {
                continue;
            }
            if !expanded {
                if let Some(r) = self.closed.get(&(key.0, env.ctes_id)).cloned() {
                    // Resolved before in another environment: the same
                    // relation, as a fresh instance where its ids would
                    // meet an enclosing query's.
                    let overlaps = matches!((r.ids, env.outer_ids),
                        (Some(a), Some(b)) if a.0 <= b.1 && b.0 <= a.1);
                    let r = if overlaps { self.instance(&r)? } else { r };
                    self.memo.insert(key, r);
                    continue;
                }
                stack.push((Arc::clone(&p), true));
                // A lateral join resolves its right side in its own
                // environment (the left side in scope), so it starts both.
                if !matches!(
                    p.node,
                    PlanNode::WithCte { .. }
                        | PlanNode::UnionLoop { .. }
                        | PlanNode::Join {
                            kind: crate::JoinKind::LateralInner | crate::JoinKind::LateralLeft,
                            ..
                        }
                ) {
                    let mut kids = Vec::new();
                    p.for_each_child(&mut |c| kids.push(self.rep(c)));
                    if matches!(p.node, PlanNode::Shift { .. }) {
                        // A shift reads its input's own resolution (see
                        // `shift_input`): resolved before, nothing to do.
                        kids.retain(|c| !self.closed.contains_key(&(addr(&**c), env.ctes_id)));
                    }
                    stack.extend(kids.into_iter().rev().map(|c| (c, false)));
                }
                continue;
            }
            let (lo, saved) = (self.next_id, self.op_outer.take());
            let mut r = match self.node(&p, env) {
                Ok(r) => r,
                // A select-list generator runs below its projection: see
                // `generator_over_lateral_alias`.
                Err(e) if matches!(p.node, PlanNode::Generate { .. }) => {
                    let list = stack.iter().rev().find_map(|(q, open)| match &q.node {
                        PlanNode::Generate { .. } if *open => None,
                        PlanNode::Project { exprs, .. } if *open => Some(Some(exprs)),
                        _ if *open => Some(None),
                        _ => None,
                    });
                    return Err(match list.flatten() {
                        Some(exprs) => self.generator_over_lateral_alias(exprs, e),
                        None => e,
                    });
                }
                Err(e) => return Err(e),
            };
            if self.next_id > lo {
                r.ids = hull(r.ids, Some((lo, self.next_id - 1)));
            }
            r.outer = [r.outer, self.op_outer].into_iter().flatten().min();
            self.op_outer = saved;
            if r.outer.is_none() {
                self.closed.insert((key.0, env.ctes_id), r.clone());
            }
            self.memo.insert(key, r);
        }
        self.memo
            .get(&(addr(&*root), env.id))
            .cloned()
            .ok_or_else(|| unsupported("a plan shape", root.origin))
    }

    pub(super) fn child(&self, c: &Arc<LogicalPlan>, env: &Env) -> Result<Rel, AnalysisError> {
        self.memo
            .get(&(addr(&*self.rep(c)), env.id))
            .cloned()
            .ok_or_else(|| unsupported("a plan shape", c.origin))
    }

    /// The input of a `Shift`, resolved as itself: the shift is the fresh
    /// instance, so it stands over the input's own resolution (the ids the
    /// input states, which it reads as `id + offset`), never over an
    /// instance made because those ids meet an enclosing query's. The
    /// input has no outer references, so its resolution is closed and
    /// shared by every environment with this CTE scope.
    fn shift_input(&self, input: &Arc<LogicalPlan>, env: &Env) -> Result<Rel, AnalysisError> {
        let rep = self.rep(input);
        match self.closed.get(&(addr(&*rep), env.ctes_id)) {
            Some(r) => Ok(r.clone()),
            None => {
                let r = self.child(&rep, env)?;
                if r.outer.is_some() {
                    return Err(unsupported("a shift over a correlated plan", input.origin));
                }
                Ok(r)
            }
        }
    }

    /// A relation over `kids` with output `out`, at `origin` (whose
    /// `plan_id` it carries).
    pub(super) fn make(
        &self,
        origin: Origin,
        node: PlanNode,
        out: Vec<Attr>,
        kids: &[&Rel],
    ) -> Rel {
        let passes = matches!(
            node,
            PlanNode::Filter { .. }
                | PlanNode::Sort { .. }
                | PlanNode::Limit { .. }
                | PlanNode::Offset { .. }
                | PlanNode::Repartition { .. }
        );
        let plan = Arc::new(LogicalPlan::new(node).with_origin(origin));
        let mut r = self.wrap(plan, origin.plan_id, out, kids);
        if let ([k], true) = (kids, passes) {
            r.hidden = Arc::clone(&k.hidden);
        }
        r
    }

    pub(super) fn wrap(
        &self,
        plan: Arc<LogicalPlan>,
        id: Option<PlanId>,
        out: Vec<Attr>,
        kids: &[&Rel],
    ) -> Rel {
        let out: Arc<[Attr]> = out.into();
        let own = out
            .iter()
            .fold(None, |h, a| hull(h, Some((a.col.id.0, a.col.id.0))));
        let ids = kids.iter().fold(own, |h, k| hull(h, k.ids));
        let mut carriers = match kids {
            [k] => Arc::clone(&k.carriers),
            _ => {
                let mut m: BTreeMap<PlanId, Vec<Arc<[Attr]>>> = BTreeMap::new();
                for k in kids {
                    for (id, v) in k.carriers.iter() {
                        m.entry(*id).or_default().extend(v.iter().cloned());
                    }
                }
                Arc::new(m)
            }
        };
        if let Some(id) = id {
            Arc::make_mut(&mut carriers).insert(id, vec![Arc::clone(&out)]);
        }
        Rel {
            plan,
            out,
            ids,
            outer: kids.iter().filter_map(|k| k.outer).min(),
            carriers,
            agg: None,
            widen: None,
            hidden: Arc::from(Vec::new()),
            generated_from: None,
        }
    }

    /// Spark's checks of a `LIMIT` or `OFFSET` (`name`) value: it must be
    /// foldable (`IS_UNFOLDABLE` when it is not: a column, a nondeterministic
    /// call, a subquery), to a value that is not NULL and not negative
    /// (cited at the expression, `at`); what folding raises (`2147483647 + 1`
    /// under ANSI) is raised as is. `unfoldable` reports the first check
    /// without folding. A foldable expression this folder cannot evaluate
    /// (a cast or call the kernels lack here) is not an error: it passes
    /// through for the executor to evaluate once at run time, as before.
    /// The folded value, when it is an integral.
    #[inline(never)]
    pub(super) fn check_limit(
        &self,
        b: &super::exprs::Bound,
        name: &str,
        unfoldable: bool,
        at: Origin,
        operator: Origin,
    ) -> Result<Option<i64>, AnalysisError> {
        use krishiv_common::error::{
            InvalidLimitLikeExpressionIsNegative, InvalidLimitLikeExpressionIsNull,
            InvalidLimitLikeExpressionIsUnfoldable,
        };
        // A nondeterministic expression is pulled out of the operator into
        // an attribute of this name before Spark checks it.
        let nondeterministic = {
            let mut found = false;
            let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(&b.expr), |n| {
                if let crate::walk::Node::Expr(x) = n
                    && let ExprNode::Resolved(crate::ResolvedExpr::Function(f)) = &x.node
                    && !f.deterministic
                {
                    found = true;
                }
                std::ops::ControlFlow::Continue(())
            });
            found
        };
        let text = if nondeterministic {
            "\"_nondeterministic\"".to_owned()
        } else {
            format!(
                "\"{}\"",
                super::exprs::pretty(&b.shown(), self.ctx.conf.session_zone)
            )
        };
        let name = name.to_owned();
        let folded = if unfoldable {
            None
        } else {
            self.fold(&b.expr, self.ctx.conf.ansi)
        };
        match folded {
            // An attribute carries no position of its own: Spark cites the
            // input there, the expression otherwise.
            // Not foldable to Spark either: `IS_UNFOLDABLE` without folding.
            None if unfoldable => Err(AnalysisError::new(
                InvalidLimitLikeExpressionIsUnfoldable { expr: text, name },
                if matches!(
                    b.expr.node,
                    ExprNode::Resolved(crate::ResolvedExpr::Column(_))
                ) {
                    operator
                } else {
                    b.expr.origin
                },
            )),
            Some(super::Constant::Error(e)) => Err(AnalysisError::new(e, b.expr.origin)),
            Some(super::Constant::Null) => Err(AnalysisError::new(
                InvalidLimitLikeExpressionIsNull { expr: text, name },
                at,
            )),
            Some(super::Constant::Integral(v)) if v < 0 => Err(AnalysisError::new(
                InvalidLimitLikeExpressionIsNegative {
                    expr: text,
                    name,
                    v: v.to_string(),
                },
                at,
            )),
            Some(super::Constant::Integral(v)) => Ok(Some(v)),
            // Foldable, but not evaluated here (or not an integral): pass
            // the expression through for the downstream folders and the
            // executor to read at run time.
            None | Some(_) => Ok(None),
        }
    }

    /// Binds a top-level expression of the current operator.
    pub(super) fn bind(&mut self, e: &Arc<Expr>, cx: &Cx<'_>) -> Result<Bound, AnalysisError> {
        let b = self.expr(e, cx)?;
        self.op_outer = [self.op_outer, b.outer].into_iter().flatten().min();
        Ok(b)
    }

    pub(super) fn cx<'s>(
        scope: &'s [Attr],
        rels: &'s [&'s Rel],
        env: &'s Arc<Env>,
        op: &'static str,
    ) -> Cx<'s> {
        Cx {
            scope,
            rels,
            lateral: None,
            aliases: &[],
            extra: None,
            agg_scope: None,
            grouping: false,
            lenient: false,
            outer: true,
            windows: super::exprs::WindowUse::Other,
            env,
            op,
        }
    }

    fn node(&mut self, p: &Arc<LogicalPlan>, env: &Arc<Env>) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        // The arms that recurse into children stay here; the others are
        // out of line in `leaf_node`, so this frame (on the recursion path
        // of nested subqueries) stays small.
        match &p.node {
            PlanNode::Read(ReadSource::Table { name, options }) => self.read(p, name, options, env),
            PlanNode::OneRowRelation => Ok(self.wrap(Arc::clone(p), o.plan_id, Vec::new(), &[])),
            PlanNode::Project { input, exprs } => self.project(p, input, exprs, env),
            PlanNode::Generate {
                input,
                kind,
                outer,
                arg,
                output,
                names,
                qualifier,
            } => self.generate_node(
                o,
                input,
                *kind,
                *outer,
                arg,
                output,
                names,
                qualifier.as_ref(),
                env,
            ),
            PlanNode::Filter { input, predicate } => self.filter(p, input, predicate, env),
            PlanNode::Aggregate {
                input,
                grouping,
                aggregates,
            } => self.aggregate(p, input, grouping, aggregates, env),
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => self.join(p, left, right, *kind, condition, env),
            PlanNode::Sort {
                input,
                order,
                global,
            } => self.sort(p, input, order, *global, env),
            PlanNode::SubqueryAlias {
                input,
                alias,
                columns,
            } => self.subquery_alias(p, input, alias, columns, env),
            PlanNode::Union {
                inputs,
                by_name,
                allow_missing_columns,
            } => self.union(p, inputs, *by_name, *allow_missing_columns, env),
            PlanNode::SetOp {
                left,
                right,
                op,
                all,
            } => self.set_op(p, left, right, *op, *all, env),
            PlanNode::InlineTable { rows, columns } => self.inline_table(p, rows, columns, env),
            PlanNode::WithColumns { input, columns } => self.with_columns(p, input, columns, env),
            PlanNode::Limit { input, limit } => self.limit_node(p, input, limit, false, env),
            PlanNode::Offset { input, offset } => self.limit_node(p, input, offset, true, env),
            PlanNode::WithCte { input, ctes } => self.with_cte(p, input, ctes, env),
            PlanNode::UnionLoop {
                id,
                anchor,
                recursive,
                max_level,
                limit,
                ..
            } => self.union_loop(p, *id, anchor, recursive, *max_level, *limit, env),
            PlanNode::UnionLoopRef { id, .. } => self.loop_ref(p, *id),
            PlanNode::Distinct { input } => self.distinct(p, input, env),
            PlanNode::Pivot {
                input,
                pivot_columns,
                values,
                aggregates,
            } => self.pivot(p, input, pivot_columns, values, aggregates, env),
            PlanNode::Unpivot {
                input,
                value_columns,
                name_column,
                column_sets,
                include_nulls,
            } => self.unpivot(
                p,
                input,
                value_columns,
                name_column,
                column_sets,
                *include_nulls,
                env,
            ),
            PlanNode::Window {
                input,
                exprs,
                partition_by,
                order_by,
            } => self.window_rel(p, input, exprs, partition_by, order_by, env),
            PlanNode::Hint {
                input,
                name,
                params,
            } => self.hint(p, input, name, params, env),
            PlanNode::PipeOperator { input } => self.pipe_operator(input, env),
            PlanNode::Repartition { input, exprs } => self.repartition(p, input, exprs, env),
            PlanNode::TableFunction { name, args } => self.table_function(p, name, args, env),
            // Listed, not wildcarded: a new `PlanNode` variant must be
            // placed here or in `leaf_node` to compile.
            PlanNode::Read(ReadSource::Files { .. })
            | PlanNode::Sql { .. }
            | PlanNode::Scan(_)
            | PlanNode::CteRef(_)
            | PlanNode::PartialAggregate { .. }
            | PlanNode::Shift { .. }
            | PlanNode::ReplicateRows { .. }
            | PlanNode::Expand { .. }
            | PlanNode::Range { .. }
            | PlanNode::JoinHint { .. } => self.leaf_node(p, env),
        }
    }

    #[inline(never)]
    fn leaf_node(&mut self, p: &Arc<LogicalPlan>, env: &Arc<Env>) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        match &p.node {
            PlanNode::Read(source @ ReadSource::Files { .. }) => {
                match self.b.files.iter().position(|f| f.source == *source) {
                    Some(i) => Ok(self.files(p, i)),
                    None => Err(unsupported("reading files by path", o)),
                }
            }
            PlanNode::Sql { .. } => {
                let root = self.b.sql.get(&addr(&**p)).cloned();
                let env0 = self.root_env.clone();
                match (root, env0) {
                    (Some(root), Some(env0)) => self.rel(&root, &env0),
                    _ => Err(unsupported("SQL text", o)),
                }
            }
            PlanNode::Scan(s) => {
                let name = match &s.source {
                    ReadSource::Table { name, .. } => name.clone(),
                    ReadSource::Files { .. } => Vec::new(),
                };
                let source = self.b.scans.get(&name).copied();
                // Bind checked every scan of a table; one it did not bind
                // never reaches a plan.
                if matches!(s.source, ReadSource::Table { .. }) && source.is_none() {
                    return Err(unsupported("a scan of an unchecked table", o));
                }
                if let Some(declared) = self.b.scan_declared.get(&name) {
                    for c in &s.output {
                        if let Some(d) = declared.get(&c.name) {
                            self.declared.insert(c.id, Arc::clone(d));
                        }
                    }
                }
                if let Some(comments) = self.b.scan_comments.get(&name) {
                    for c in &s.output {
                        if let Some(t) = comments.get(&c.name) {
                            self.comments.insert(c.id, Arc::clone(t));
                        }
                    }
                }
                let qual: Arc<[String]> = name.into();
                let out = s
                    .output
                    .iter()
                    .map(|c| Attr {
                        col: c.clone(),
                        qual: Arc::clone(&qual),
                        source: source.map(|i| (i, Arc::from(c.name.as_str()))),
                        duplicate: false,
                    })
                    .collect();
                Ok(self.wrap(Arc::clone(p), o.plan_id, out, &[]))
            }
            PlanNode::CteRef(r) => {
                // A reference in a stored analyzed plan (a DataFrame's temporary view) holds the declared
                // `CHAR`/`VARCHAR` types of its definition's columns, as a fresh reference does.
                for (source, out) in r.source.iter().zip(&r.output) {
                    self.carry_column_meta(*source, out.id);
                }
                let qual: Arc<[String]> = vec![r.name.clone()].into();
                let out = r
                    .output
                    .iter()
                    .map(|c| Attr {
                        col: c.clone(),
                        qual: Arc::clone(&qual),
                        source: None,
                        duplicate: false,
                    })
                    .collect();
                Ok(self.wrap(Arc::clone(p), o.plan_id, out, &[]))
            }
            PlanNode::Shift { input, offset } => {
                let c = self.shift_input(input, env)?;
                // The shifted node keeps its input: it may stand for the
                // resolved plan only when resolving left the input as it
                // was, with the attributes it states.
                if !Arc::ptr_eq(&c.plan, input) {
                    let same = input.is_resolved()
                        && input.resolved_output().is_some_and(|out| {
                            out.iter().map(|b| b.id).eq(c.out.iter().map(|a| a.col.id))
                        });
                    if !same {
                        return Err(unsupported("a shift over an unresolved plan", o));
                    }
                }
                self.shifted(&c, *offset, Some(Arc::clone(p)))
            }
            PlanNode::ReplicateRows { input, count } => {
                let c = self.child(input, env)?;
                let rels = [&c];
                let b = self.bind(count, &Self::cx(&c.out, &rels, env, "Generate"))?;
                let node = PlanNode::ReplicateRows {
                    input: Arc::clone(&c.plan),
                    count: b.expr,
                };
                Ok(self.make(o, node, c.out.to_vec(), &[&c]))
            }
            PlanNode::Expand {
                input,
                projections,
                output,
            } => self.expand_node(o, input, projections, output, env),
            PlanNode::JoinHint { input, strategy } => {
                let c = self.child(input, env)?;
                let node = PlanNode::JoinHint {
                    input: Arc::clone(&c.plan),
                    strategy: *strategy,
                };
                let mut r = self.make(o, node, c.out.to_vec(), &[&c]);
                r.widen = Some(Arc::new(Widen::Through { child: c }));
                Ok(r)
            }
            PlanNode::Range {
                start,
                end,
                step,
                num_slices,
                output,
            } => {
                // A built `Range` (the DataFrame API, Spark Connect) carries
                // an id of its builder's choosing, which two ranges of one
                // plan share; the leaf gets this analysis's own fresh id.
                let col = BoundColumn {
                    id: self.fresh(),
                    ..output.clone()
                };
                let node = PlanNode::Range {
                    start: *start,
                    end: *end,
                    step: *step,
                    num_slices: *num_slices,
                    output: col.clone(),
                };
                let qual: Arc<[String]> = Arc::from(Vec::new());
                let out = vec![Attr {
                    col,
                    qual,
                    source: None,
                    duplicate: false,
                }];
                let leaf = Arc::new(LogicalPlan::new(node).with_origin(o));
                Ok(self.wrap(leaf, o.plan_id, out, &[]))
            }
            // Handled in `node`, which never forwards them.
            PlanNode::Read(ReadSource::Table { .. })
            | PlanNode::OneRowRelation
            | PlanNode::Project { .. }
            | PlanNode::WithColumns { .. }
            | PlanNode::Filter { .. }
            | PlanNode::Aggregate { .. }
            | PlanNode::PartialAggregate { .. }
            | PlanNode::Join { .. }
            | PlanNode::Sort { .. }
            | PlanNode::Limit { .. }
            | PlanNode::Offset { .. }
            | PlanNode::SubqueryAlias { .. }
            | PlanNode::WithCte { .. }
            | PlanNode::Hint { .. }
            | PlanNode::PipeOperator { .. }
            | PlanNode::Repartition { .. }
            | PlanNode::Union { .. }
            | PlanNode::Distinct { .. }
            | PlanNode::Pivot { .. }
            | PlanNode::Unpivot { .. }
            | PlanNode::SetOp { .. }
            | PlanNode::TableFunction { .. }
            | PlanNode::InlineTable { .. }
            | PlanNode::Generate { .. }
            | PlanNode::UnionLoop { .. }
            | PlanNode::UnionLoopRef { .. }
            | PlanNode::Window { .. } => Err(unsupported("a plan shape", o)),
        }
    }

    /// A `WithColumns` as Spark resolves it: replaced columns keep their
    /// place, new ones are appended. Out of line: it keeps the recursive
    /// `node` frame small (deep plans resolve on a small stack).
    #[inline(never)]
    fn with_columns(
        &mut self,
        p: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
        columns: &[(String, Arc<Expr>)],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let rels = [&c];
        let mut cx = Self::cx(&c.out, &rels, env, "Project");
        cx.windows = super::exprs::WindowUse::Allowed;
        let mut news: Vec<(String, Bound)> = Vec::new();
        for (name, e) in columns {
            news.push((name.clone(), self.bind(e, &cx)?));
        }
        let cs = self.ctx.conf.case_sensitive;
        let mut exprs = Vec::new();
        let mut taken = HashSet::new();
        for a in c.out.iter() {
            match news.iter().rposition(|(n, _)| same(n, &a.col.name, cs)) {
                Some(i) => {
                    taken.insert(i);
                    exprs.push(Arc::new(Expr::new(ExprNode::Alias {
                        expr: Arc::clone(&news[i].1.expr),
                        name: news[i].0.clone(),
                    })));
                }
                None => exprs.push(self.pass_expr(&a.col)),
            }
        }
        for (i, (name, b)) in news.iter().enumerate() {
            let later = news[i + 1..].iter().any(|(n, _)| same(n, name, cs));
            if !taken.contains(&i) && !later {
                exprs.push(Arc::new(Expr::new(ExprNode::Alias {
                    expr: Arc::clone(&b.expr),
                    name: name.clone(),
                })));
            }
        }
        self.project_items(p, &c, &exprs, env)
    }

    /// The CTE definitions resolved in turn, then the input over them. Out
    /// of line: it keeps the recursive `node` frame small (deep plans
    /// resolve on a small stack).
    ///
    /// Only definitions the input needs (transitively) are resolved:
    /// Spark does not analyze a CTE the query never references, so an
    /// unused definition's column errors (like `AMBIGUOUS_REFERENCE`)
    /// never fail the query. Table existence still fails even when unused:
    /// every non-shadowed table read of an unused definition is checked
    /// against what Bind loaded, so a missing relation raises
    /// `TABLE_OR_VIEW_NOT_FOUND` at its read, as Spark does.
    #[inline(never)]
    fn with_cte(
        &mut self,
        p: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
        ctes: &[Cte],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        let needed = Self::needed_ctes(input, ctes, &env.ctes, self.ctx.conf.case_sensitive);
        let mut visible = env.ctes.clone();
        let mut scope = env.ctes_id;
        let mut defs = Vec::new();
        let mut rels = Vec::new();
        for (c, keep) in ctes.iter().zip(needed.iter()) {
            if !keep {
                // An unused definition is never resolved, but its tables
                // are: a missing relation fails the query even though no
                // column error of the definition does.
                self.check_unused_cte(c)?;
                continue;
            }
            let e = self.env(visible.clone(), scope, env.outer.clone());
            let r = self.rel(&c.plan, &e)?;
            defs.push(Cte {
                name: c.name.clone(),
                plan: Arc::clone(&r.plan),
            });
            visible.push((c.name.clone(), r.clone()));
            scope = self.ctes_id();
            rels.push(r);
        }
        if defs.is_empty() {
            return self.rel(input, env);
        }
        let e = self.env(visible, scope, env.outer.clone());
        let i = self.rel(input, &e)?;
        let node = PlanNode::WithCte {
            input: Arc::clone(&i.plan),
            ctes: defs,
        };
        let mut kids: Vec<&Rel> = rels.iter().collect();
        kids.push(&i);
        let mut r = self.make(o, node, i.out.to_vec(), &kids);
        r.carriers = Arc::clone(&i.carriers);
        Ok(r)
    }

    /// The tables of an unused CTE definition, checked as Spark checks
    /// them: every non-shadowed table read (Bind already resolved the
    /// shadowing, so a read it did not record is a CTE read) must name a
    /// table Bind found, else `TABLE_OR_VIEW_NOT_FOUND` at the read.
    /// Column errors are never raised here: the definition is not
    /// resolved, only walked for its reads (subquery plans and nested
    /// `WITH` bodies included).
    fn check_unused_cte(&self, c: &Cte) -> Result<(), AnalysisError> {
        use crate::walk::{Node, visit_distinct};
        use std::ops::ControlFlow;
        let mut missing: Option<AnalysisError> = None;
        // A `PlanNode::Sql` has no relation children: its reads live in the
        // lowered plan Bind stored, so walk those too.
        let mut roots: Vec<&LogicalPlan> = vec![&c.plan];
        while let Some(root) = roots.pop() {
            let _ = visit_distinct(Node::Plan(root), |node| {
                let Node::Plan(p) = node else {
                    return ControlFlow::Continue(());
                };
                if let PlanNode::Sql { .. } = &p.node
                    && let Some(lowered) = self.b.sql.get(&addr(p))
                {
                    roots.push(lowered);
                }
                if let PlanNode::Read(ReadSource::Table { name, .. }) = &p.node
                    && let Some(BoundTable::Missing) = self.b.tables.get(name)
                {
                    missing = Some(AnalysisError::new(
                        TableOrViewNotFound {
                            relation_name: quoted(name),
                            search_path: self.b.search_path.clone(),
                        },
                        p.origin,
                    ));
                    return ControlFlow::Break(());
                }
                ControlFlow::Continue(())
            });
            if missing.is_some() {
                break;
            }
        }
        if let Some(e) = missing {
            return Err(e);
        }
        Ok(())
    }

    /// Which definitions of `ctes` the input needs, transitively: a
    /// definition is needed when the input (or another needed definition)
    /// reads its name. Reads shadowed by an inner `WITH` never count, and
    /// a definition's own reads see only the outer CTEs and the earlier
    /// definitions, so a forward reference never marks a later definition.
    fn needed_ctes(
        input: &Arc<LogicalPlan>,
        ctes: &[Cte],
        outer: &[(String, Rel)],
        case_sensitive: bool,
    ) -> Vec<bool> {
        let n = ctes.len();
        let mut needed = vec![false; n];
        if n == 0 {
            return needed;
        }
        let outer_scope: Vec<(String, Option<usize>)> =
            outer.iter().map(|(name, _)| (name.clone(), None)).collect();
        let mut input_scope = outer_scope.clone();
        input_scope.extend(
            ctes.iter()
                .enumerate()
                .map(|(i, c)| (c.name.clone(), Some(i))),
        );
        let mut stack = Self::cte_refs(input, &input_scope, case_sensitive);
        let mut memo: Vec<Option<Vec<usize>>> = (0..n).map(|_| None).collect();
        while let Some(i) = stack.pop() {
            if i >= n || needed[i] {
                continue;
            }
            needed[i] = true;
            let refs = match &memo[i] {
                Some(r) => r.clone(),
                None => {
                    let mut scope = outer_scope.clone();
                    scope.extend(
                        ctes[..i]
                            .iter()
                            .enumerate()
                            .map(|(j, c)| (c.name.clone(), Some(j))),
                    );
                    let r = Self::cte_refs(&ctes[i].plan, &scope, case_sensitive);
                    memo[i] = Some(r.clone());
                    r
                }
            };
            for j in refs {
                if j < n && !needed[j] {
                    stack.push(j);
                }
            }
        }
        needed
    }

    /// Indices of this `WITH` level's definitions that `root` reads, with
    /// `scope` naming every visible CTE (outer ones as `None`, this level's
    /// as `Some(index)`). A single-part read resolves to the innermost
    /// scope entry matching its name; only reads resolving to this level
    /// are reported. Inner `WITH` names shadow as `None`.
    fn cte_refs(root: &Arc<LogicalPlan>, scope: &CteScope, case_sensitive: bool) -> Vec<usize> {
        enum Item<'a> {
            Plan(&'a Arc<LogicalPlan>, CteScope),
            Expr(&'a Arc<Expr>, CteScope),
        }
        let mut out = Vec::new();
        // Visits are keyed by scope contents, not scope length: the same
        // plan reached through two inner `WITH` scopes with different names
        // but equal lengths resolves its reads differently (a shadowing name
        // hides what the other scope reports), so each scope gets a visit.
        let mut seen: HashSet<(usize, CteScope)> = HashSet::new();
        let mut stack = vec![Item::Plan(root, scope.clone())];
        while let Some(item) = stack.pop() {
            match item {
                Item::Plan(p, sc) => {
                    if !seen.insert((addr(&**p), sc.clone())) {
                        continue;
                    }
                    match &p.node {
                        PlanNode::Read(ReadSource::Table { name, .. }) => {
                            if let [n] = name.as_slice() {
                                for (sname, idx) in sc.iter().rev() {
                                    if same(sname, n, case_sensitive) {
                                        if let Some(i) = idx
                                            && !out.contains(i)
                                        {
                                            out.push(*i);
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        PlanNode::CteRef(r) => {
                            // An already-resolved reference: re-analysis
                            // must keep the definition it names.
                            for (sname, idx) in sc.iter().rev() {
                                if same(sname, &r.name, case_sensitive) {
                                    if let Some(i) = idx
                                        && !out.contains(i)
                                    {
                                        out.push(*i);
                                    }
                                    break;
                                }
                            }
                        }
                        PlanNode::WithCte { input, ctes } => {
                            let mut ext = sc.clone();
                            ext.extend(ctes.iter().map(|c| (c.name.clone(), None)));
                            stack.push(Item::Plan(input, ext));
                            let mut cur = sc.clone();
                            for c in ctes {
                                stack.push(Item::Plan(&c.plan, cur.clone()));
                                cur.push((c.name.clone(), None));
                            }
                            continue;
                        }
                        _ => {}
                    }
                    p.for_each_expr(&mut |e| {
                        stack.push(Item::Expr(e, sc.clone()));
                    });
                    // `WithCte` already pushed its children above.
                    if !matches!(p.node, PlanNode::WithCte { .. }) {
                        p.for_each_child(&mut |c| {
                            stack.push(Item::Plan(c, sc.clone()));
                        });
                    }
                }
                Item::Expr(e, sc) => {
                    if !seen.insert((addr(&**e), sc.clone())) {
                        continue;
                    }
                    if let ExprNode::Subquery { plan, .. } = &e.node {
                        stack.push(Item::Plan(plan, sc.clone()));
                    }
                    e.for_each_child(&mut |c| {
                        stack.push(Item::Expr(c, sc.clone()));
                    });
                }
            }
        }
        out
    }
}
