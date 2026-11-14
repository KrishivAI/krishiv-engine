// SPDX-License-Identifier: Apache-2.0
//! The lineage walk over relations: one rule per `PlanNode` kind (design §3.1), memoized by (node address,
//! `Shift` offset), with the recursive CTE's fixed point.
//!
//! Attribute ids are unique within a resolved plan, and inside a `Shift` every id reads `id + offset`, so the
//! walk keeps one map from global attribute id to lineage: a node's expressions (and a subquery's outer
//! references) read their columns there, after the node's children put them in.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use krishiv_common::error::{FailedToParseTooComplex, InternalError, KrishivError};
use krishiv_common::stack::{STEP, on_big_stack};

use super::{ColumnLineage, FilterKind, Leaf, Lineage, LineageWarning, Source, Transform};
use crate::analyzer::AnalyzedPlan;
use crate::analyzer::views::ViewExpansion;
use crate::plan::{
    GeneratorKind, JoinCondition, JoinKind, LogicalPlan, PlanNode, ReadSource, Scan,
};
use crate::{
    AttrId, Builtin, Expr, ExprNode, FunctionId, Grouping, MAX_DEPTH, ResolvedExpr, generator_arg,
};

/// A map keyed by addresses or ids: the analyzer's multiply hasher (the keys are not attacker-chosen).
pub(super) type IdMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<crate::walk::PtrHasher>>;

/// A [`Source`] interned in the walk ([`Walker::source`]): sets of these are cheap to merge.
pub(super) type Sid = u32;

/// The lineage of one value: its leaves and the strongest step on any path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ColInfo {
    pub(super) sources: BTreeSet<Sid>,
    pub(super) transform: Transform,
}

/// A shared [`ColInfo`].
pub(super) type Info = Arc<ColInfo>;

/// A relation's output: its global attribute ids with their lineage, in order.
type Out = Arc<Vec<(AttrId, Info)>>;

/// What a node's rule gives: its input's output as it is (a pass-through), or a new one.
enum Rule {
    Same(Out),
    New(Vec<(AttrId, Info)>),
}

/// Rounds of a recursive CTE's fixed point before the walk gives up (each round's lineage is merged into
/// the last, so the sets only grow and are finite: it never gets near).
const LOOP_ROUNDS: usize = 10_000;

/// A memo entry, kept in order so a fixed-point round can drop what it computed.
enum Memo {
    Plan((usize, u64)),
    Expr((usize, u64)),
}

/// The walk's state.
pub(super) struct Walker<'a> {
    views: &'a [ViewExpansion],
    /// Global attribute id to lineage.
    pub(super) env: IdMap<AttrId, Info>,
    plans: IdMap<(usize, u64), Out>,
    pub(super) exprs: IdMap<(usize, u64), Info>,
    journal: Vec<Memo>,
    /// The current lineage of each open recursive CTE, by (loop id, offset).
    loops: IdMap<(u32, u64), Vec<Info>>,
    pub(super) filters: BTreeSet<(FilterKind, Sid)>,
    pub(super) warnings: BTreeSet<LineageWarning>,
    /// The interned sources, by [`Sid`], and back.
    sources: Vec<Source>,
    sids: HashMap<Source, Sid>,
    /// A source re-tagged with an expanded view, by (source, view entry).
    tagged: IdMap<(Sid, usize), Sid>,
    depth: usize,
    /// What the walk computed.
    pub(super) counts: Counts,
}

/// What a walk computed: relations and expressions, each (node, offset) at most once outside a fixed
/// point's rounds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    /// (Relation, offset) pairs.
    pub(crate) relations: usize,
    /// (Expression, offset) pairs.
    pub(crate) expressions: usize,
}

/// An `INTERNAL_ERROR` with `message`.
pub(super) fn internal(message: &str) -> KrishivError {
    InternalError {
        message: format!("lineage: {message}"),
    }
    .into()
}

/// The info of a value with no leaf.
pub(super) fn literal() -> Info {
    Arc::new(ColInfo {
        sources: BTreeSet::new(),
        transform: Transform::Literal,
    })
}

/// `a` and `b` where paths meet: the union of the sources, the strongest transform.
pub(super) fn merge(a: &Info, b: &Info) -> Info {
    let transform = a.transform.max(b.transform);
    if transform == a.transform && b.sources.is_subset(&a.sources) {
        return Arc::clone(a);
    }
    if transform == b.transform && a.sources.is_subset(&b.sources) {
        return Arc::clone(b);
    }
    Arc::new(ColInfo {
        sources: a.sources.union(&b.sources).copied().collect(),
        transform,
    })
}

/// The id `id` reads as under `offset`.
pub(super) fn shifted(id: AttrId, offset: u64) -> Result<AttrId, KrishivError> {
    id.0.checked_add(offset)
        .map(AttrId)
        .ok_or_else(|| internal("an attribute id overflows its shift"))
}

/// The address of `x`, a memo key.
pub(super) fn addr<T>(x: &T) -> usize {
    std::ptr::from_ref(x) as usize
}

/// The lineage of `analyzed`, and what the walk computed.
pub(super) fn run(analyzed: &AnalyzedPlan) -> Result<(Lineage, Counts), KrishivError> {
    // A view that is not expanded must be found (a node with its alias and origin in the plan itself, whatever
    // the entry's located ids say), or its base tables could show: fail closed.
    let views = &analyzed.views;
    if views.iter().any(|v| !v.expands()) {
        let mut found: Vec<&ViewExpansion> = Vec::new();
        let _ =
            crate::walk::visit_distinct(crate::walk::Node::Plan(analyzed.plan.root()), |node| {
                if let crate::walk::Node::Plan(p) = node
                    && let PlanNode::SubqueryAlias { alias, .. } = &p.node
                {
                    found.extend(views.iter().filter(|v| v.is_at(alias, p.origin)));
                }
                std::ops::ControlFlow::Continue(())
            });
        if views
            .iter()
            .any(|v| !v.expands() && !found.iter().any(|f| std::ptr::eq(*f, v)))
        {
            return Err(internal("a view that is not expanded is not in the plan"));
        }
    }
    let mut w = Walker {
        views: &analyzed.views,
        env: IdMap::default(),
        plans: IdMap::default(),
        exprs: IdMap::default(),
        journal: Vec::new(),
        loops: IdMap::default(),
        filters: BTreeSet::new(),
        warnings: BTreeSet::new(),
        sources: Vec::new(),
        sids: HashMap::new(),
        tagged: IdMap::default(),
        depth: 0,
        counts: Counts::default(),
    };
    let out = w.plan(analyzed.plan.root(), 0)?;
    if out.len() != analyzed.output.len()
        || out
            .iter()
            .zip(&analyzed.output)
            .any(|((id, _), c)| *id != c.id)
    {
        return Err(internal(
            "the plan's output does not match the analyzed output",
        ));
    }
    let columns = out
        .iter()
        .zip(&analyzed.output)
        .enumerate()
        .map(|(ordinal, ((_, info), col))| {
            Ok(ColumnLineage {
                ordinal,
                name: col.name.clone(),
                sources: info
                    .sources
                    .iter()
                    .map(|s| w.resolve(*s).cloned())
                    .collect::<Result<_, _>>()?,
                transform: info.transform,
            })
        })
        .collect::<Result<_, KrishivError>>()?;
    let filters = w
        .filters
        .iter()
        .map(|(k, s)| Ok((*k, w.resolve(*s)?.leaf.clone())))
        .collect::<Result<_, KrishivError>>()?;
    Ok((
        Lineage {
            columns,
            filters,
            warnings: w.warnings,
        },
        w.counts,
    ))
}

impl Walker<'_> {
    /// The interned id of `s`.
    pub(super) fn source(&mut self, s: Source) -> Result<Sid, KrishivError> {
        if let Some(id) = self.sids.get(&s) {
            return Ok(*id);
        }
        let id = Sid::try_from(self.sources.len()).map_err(|_| internal("too many sources"))?;
        self.sources.push(s.clone());
        self.sids.insert(s, id);
        Ok(id)
    }

    /// The source interned as `id`.
    pub(super) fn resolve(&self, id: Sid) -> Result<&Source, KrishivError> {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.sources.get(i))
            .ok_or_else(|| internal("a source the walk did not intern"))
    }

    /// Runs `f` one level deeper: `FAILED_TO_PARSE_TOO_COMPLEX` past [`MAX_DEPTH`], and on a fresh stack
    /// every [`STEP`] levels.
    pub(super) fn deeper<T: Send>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, KrishivError> + Send,
    ) -> Result<T, KrishivError> {
        if self.depth >= MAX_DEPTH {
            return Err(FailedToParseTooComplex {}.into());
        }
        self.depth += 1;
        let r = if self.depth.is_multiple_of(STEP) {
            on_big_stack(|| f(self)).unwrap_or_else(|| Err(FailedToParseTooComplex {}.into()))
        } else {
            f(self)
        };
        self.depth -= 1;
        r
    }

    /// The output of `p` under `offset`, computed once per (node, offset).
    fn plan(&mut self, p: &Arc<LogicalPlan>, offset: u64) -> Result<Out, KrishivError> {
        let key = (addr(&**p), offset);
        if let Some(out) = self.plans.get(&key) {
            return Ok(Arc::clone(out));
        }
        let rule = self.deeper(|w| w.rule(p, offset))?;
        self.counts.relations += 1;
        let out = match rule {
            // The input put these in the map already.
            Rule::Same(out) => out,
            Rule::New(v) => {
                for (id, info) in &v {
                    self.env.insert(*id, Arc::clone(info));
                }
                Arc::new(v)
            }
        };
        self.plans.insert(key, Arc::clone(&out));
        self.journal.push(Memo::Plan(key));
        Ok(out)
    }

    /// The lineage of global attribute `id`.
    pub(super) fn column(&self, id: AttrId) -> Result<Info, KrishivError> {
        self.env
            .get(&id)
            .cloned()
            .ok_or_else(|| internal("a column nothing defines"))
    }

    /// Records each leaf of `info` as a filter column of `kind`.
    pub(super) fn record(&mut self, kind: FilterKind, info: &Info) {
        for s in &info.sources {
            self.filters.insert((kind, *s));
        }
    }

    /// Evaluates `e` for the filter columns of `kind`.
    fn filter_by(
        &mut self,
        kind: FilterKind,
        e: &Arc<Expr>,
        offset: u64,
    ) -> Result<(), KrishivError> {
        let info = self.value(e, offset)?;
        self.record(kind, &info);
        Ok(())
    }

    /// An output expression of a projection or an aggregate: a column, or an alias.
    fn output(&mut self, e: &Arc<Expr>, offset: u64) -> Result<(AttrId, Info), KrishivError> {
        let id = match &e.node {
            crate::ExprNode::Resolved(
                crate::ResolvedExpr::Column(c) | crate::ResolvedExpr::Alias { output: c, .. },
            ) => c.id,
            _ => return Err(internal("an output expression without an attribute")),
        };
        Ok((shifted(id, offset)?, self.value(e, offset)?))
    }

    /// The rule of `p`'s kind. A node that passes its input's rows through gives the input's output as it
    /// is (no copy).
    #[deny(clippy::wildcard_enum_match_arm)]
    fn rule(&mut self, p: &LogicalPlan, offset: u64) -> Result<Rule, KrishivError> {
        match &p.node {
            PlanNode::Scan(s) => self.scan(s, offset).map(Rule::New),
            PlanNode::OneRowRelation => Ok(Rule::New(Vec::new())),
            PlanNode::Range { output, .. } => Ok(Rule::New(vec![(
                shifted(output.id, offset)?,
                Arc::new(ColInfo {
                    sources: BTreeSet::new(),
                    transform: Transform::Generated,
                }),
            )])),
            PlanNode::Project { input, exprs } => {
                self.plan(input, offset)?;
                exprs
                    .iter()
                    .map(|e| self.output(e, offset))
                    .collect::<Result<_, _>>()
                    .map(Rule::New)
            }
            PlanNode::Filter { input, predicate } => {
                let out = self.plan(input, offset)?;
                self.filter_by(FilterKind::Filter, predicate, offset)?;
                Ok(Rule::Same(out))
            }
            PlanNode::Aggregate {
                input,
                grouping,
                aggregates,
            } => {
                self.plan(input, offset)?;
                for g in grouping_exprs(grouping) {
                    self.filter_by(FilterKind::GroupBy, g, offset)?;
                }
                aggregates
                    .iter()
                    .map(|e| self.output(e, offset))
                    .collect::<Result<_, _>>()
                    .map(Rule::New)
            }
            PlanNode::PartialAggregate { .. } => Err(internal(
                "a partial aggregate (optimizer only) in an analyzed plan",
            )),
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => self
                .join(left, right, *kind, condition, offset)
                .map(Rule::New),
            PlanNode::Sort { input, order, .. } => {
                let out = self.plan(input, offset)?;
                for o in order {
                    self.filter_by(FilterKind::Sort, &o.expr, offset)?;
                }
                Ok(Rule::Same(out))
            }
            PlanNode::Repartition { input, exprs } => {
                let out = self.plan(input, offset)?;
                for e in exprs {
                    self.filter_by(FilterKind::Sort, e, offset)?;
                }
                Ok(Rule::Same(out))
            }
            PlanNode::Limit { input, .. }
            | PlanNode::Offset { input, .. }
            | PlanNode::JoinHint { input, .. }
            | PlanNode::ReplicateRows { input, .. } => Ok(Rule::Same(self.plan(input, offset)?)),
            PlanNode::SubqueryAlias { input, alias, .. } => self.alias(p, input, alias, offset),
            PlanNode::WithCte { input, ctes } => {
                for c in ctes {
                    self.plan(&c.plan, offset)?;
                }
                Ok(Rule::Same(self.plan(input, offset)?))
            }
            PlanNode::CteRef(r) => {
                if r.source.len() != r.output.len() {
                    return Err(internal("a CTE reference of another width"));
                }
                // A `CteRef::source` is the definition's id as written, never shifted (see
                // `PlanNode::Shift`). A `WithCte` outside the reference's `Shift` put the definition in the map
                // unshifted; one inside it (a shared DataFrame holding the whole `WITH`) was walked under the
                // offset, so its ids are in the map shifted: the fallback.
                r.source
                    .iter()
                    .zip(&r.output)
                    .map(|(s, o)| {
                        let info = match self.env.get(s) {
                            Some(i) => Arc::clone(i),
                            None => self.column(shifted(*s, offset)?)?,
                        };
                        Ok((shifted(o.id, offset)?, info))
                    })
                    .collect::<Result<_, _>>()
                    .map(Rule::New)
            }
            PlanNode::Shift { input, offset: by } => {
                let inner = offset
                    .checked_add(*by)
                    .ok_or_else(|| internal("a shift overflows"))?;
                Ok(Rule::Same(self.plan(input, inner)?))
            }
            PlanNode::Union {
                inputs, by_name, ..
            } => {
                if *by_name {
                    return Err(internal("an unresolved union by name"));
                }
                let mut out: Option<Vec<(AttrId, Info)>> = None;
                for i in inputs {
                    let o = self.plan(i, offset)?;
                    out = Some(match out {
                        None => o.to_vec(),
                        Some(acc) => positional(acc, &o)?,
                    });
                }
                out.map(Rule::New)
                    .ok_or_else(|| internal("a union without inputs"))
            }
            PlanNode::Window {
                input,
                exprs,
                partition_by,
                order_by,
            } => {
                let mut out = self.plan(input, offset)?.to_vec();
                for e in partition_by {
                    self.filter_by(FilterKind::Window, e, offset)?;
                }
                for o in order_by {
                    self.filter_by(FilterKind::Window, &o.expr, offset)?;
                }
                for e in exprs {
                    out.push(self.output(e, offset)?);
                }
                Ok(Rule::New(out))
            }
            PlanNode::Expand {
                input,
                projections,
                output,
            } => {
                self.plan(input, offset)?;
                let mut cols: Vec<Option<Info>> = vec![None; output.len()];
                for proj in projections {
                    if proj.len() != output.len() {
                        return Err(internal("an expand projection of another width"));
                    }
                    for (slot, e) in cols.iter_mut().zip(proj) {
                        let v = self.value(e, offset)?;
                        *slot = Some(match slot.take() {
                            None => v,
                            Some(acc) => merge(&acc, &v),
                        });
                    }
                }
                output
                    .iter()
                    .zip(cols)
                    .map(|(c, info)| Ok((shifted(c.id, offset)?, info.unwrap_or_else(literal))))
                    .collect::<Result<_, _>>()
                    .map(Rule::New)
            }
            PlanNode::Generate {
                input,
                kind,
                arg,
                output,
                ..
            } => {
                let mut out = self.plan(input, offset)?.to_vec();
                let a = self.value(arg, offset)?;
                // Every built-in generator: its columns come from the argument. (PR-26a or PR-26c adds the
                // Python UDTF's row.)
                let transform = match kind {
                    GeneratorKind::Explode
                    | GeneratorKind::PosExplode
                    | GeneratorKind::Inline
                    | GeneratorKind::VariantExplode => a.transform.max(Transform::Generated),
                };
                if let Some(id) = generator_arg(arg) {
                    let id = shifted(id, offset)?;
                    out.retain(|(c, _)| *c != id);
                }
                let generated = Arc::new(ColInfo {
                    sources: a.sources.clone(),
                    transform,
                });
                for c in output {
                    out.push((shifted(c.id, offset)?, Arc::clone(&generated)));
                }
                Ok(Rule::New(out))
            }
            PlanNode::UnionLoop {
                id,
                anchor,
                recursive,
                output,
                ..
            } => self
                .union_loop(*id, anchor, recursive, output, offset)
                .map(Rule::New),
            PlanNode::UnionLoopRef { id, output } => {
                let infos = self
                    .loops
                    .get(&(*id, offset))
                    .cloned()
                    .ok_or_else(|| internal("a recursive reference outside its loop"))?;
                if infos.len() != output.len() {
                    return Err(internal("a recursive reference of another width"));
                }
                output
                    .iter()
                    .zip(infos)
                    .map(|(c, i)| Ok((shifted(c.id, offset)?, i)))
                    .collect::<Result<_, _>>()
                    .map(Rule::New)
            }
            PlanNode::Read(_)
            | PlanNode::Sql { .. }
            | PlanNode::WithColumns { .. }
            | PlanNode::Hint { .. }
            | PlanNode::PipeOperator { .. }
            | PlanNode::SetOp { .. }
            | PlanNode::Pivot { .. }
            | PlanNode::Unpivot { .. }
            | PlanNode::InlineTable { .. }
            | PlanNode::TableFunction { .. }
            | PlanNode::Distinct { .. } => Err(internal("an unresolved relation")),
        }
    }

    /// A join: both sides pass through (the left only for semi and anti joins, plus the `exists` column of
    /// an existence join, which derives from the condition); the condition's columns are `join` filters.
    #[deny(clippy::wildcard_enum_match_arm)]
    fn join(
        &mut self,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: JoinKind,
        condition: &JoinCondition,
        offset: u64,
    ) -> Result<Vec<(AttrId, Info)>, KrishivError> {
        let l = self.plan(left, offset)?;
        let r = self.plan(right, offset)?;
        let cond = match condition {
            JoinCondition::None => literal(),
            JoinCondition::On(e) => {
                let info = self.value(e, offset)?;
                self.record(FilterKind::Join, &info);
                info
            }
            JoinCondition::Using(_) | JoinCondition::Natural => {
                return Err(internal("an unresolved join condition"));
            }
        };
        let mut out = if matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti)
            && set_op_condition(condition, &l, &r, offset)
        {
            // The `INTERSECT`/`EXCEPT DISTINCT` rewrite: both sides decide the rows, so each output column
            // reads both sides' column at its position, as `UNION` and the `ALL` forms do.
            positional(l.to_vec(), &r)?
        } else {
            l.to_vec()
        };
        match kind {
            JoinKind::Inner
            | JoinKind::LeftOuter
            | JoinKind::RightOuter
            | JoinKind::FullOuter
            | JoinKind::Cross
            | JoinKind::Single
            | JoinKind::LateralInner
            | JoinKind::LateralLeft => out.extend(r.iter().cloned()),
            JoinKind::LeftSemi | JoinKind::LeftAnti => {}
            JoinKind::Existence(id) => {
                // Whether a matching row exists: a value of the condition's columns, and of the right side's
                // rows even with no condition (an uncorrelated `EXISTS` is no constant).
                let transform = cond.transform.max(Transform::Expression);
                out.push((
                    shifted(id, offset)?,
                    Arc::new(ColInfo {
                        sources: cond.sources.clone(),
                        transform,
                    }),
                ));
            }
        }
        Ok(out)
    }

    /// A `SubqueryAlias`: a pass-through, unless it is a view read of the side table (an entry with its alias
    /// and origin). An expanded view tags its sources with the view (the outermost view wins); any other view
    /// is a leaf, and nothing under it is read. Entries of one key that disagree are not expanded.
    fn alias(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        alias: &str,
        offset: u64,
    ) -> Result<Rule, KrishivError> {
        let views = self.views;
        let mut matching = views.iter().filter(|v| v.is_at(alias, p.origin));
        let Some(first) = matching.next() else {
            return Ok(Rule::Same(self.plan(input, offset)?));
        };
        let hidden = std::iter::once(first)
            .chain(matching)
            .find(|v| !v.expands());
        let Some(view) = hidden else {
            let entry = views
                .iter()
                .position(|v| std::ptr::eq(v, first))
                .ok_or_else(|| internal("a view entry outside the side table"))?;
            let out = self.plan(input, offset)?;
            let mut tagged = Vec::with_capacity(out.len());
            for (id, info) in out.iter() {
                let mut sources = BTreeSet::new();
                for s in &info.sources {
                    sources.insert(self.tag(*s, entry, &first.name)?);
                }
                tagged.push((
                    *id,
                    Arc::new(ColInfo {
                        sources,
                        transform: info.transform,
                    }),
                ));
            }
            return Ok(Rule::New(tagged));
        };
        self.warnings.insert(LineageWarning::ViewNotExpanded);
        let cols = p
            .resolved_output()
            .ok_or_else(|| internal("an unresolved view"))?;
        cols.into_iter()
            .map(|c| {
                let leaf = Leaf::View {
                    name: view.name.clone(),
                    column: c.name,
                };
                let sid = self.source(Source {
                    leaf,
                    via_view: None,
                })?;
                Ok((
                    shifted(c.id, offset)?,
                    Arc::new(ColInfo {
                        sources: BTreeSet::from([sid]),
                        transform: Transform::View,
                    }),
                ))
            })
            .collect::<Result<_, _>>()
            .map(Rule::New)
    }

    /// `s` read through the expanded view entry `entry` (named `name`): the outermost view wins.
    fn tag(&mut self, s: Sid, entry: usize, name: &[String]) -> Result<Sid, KrishivError> {
        if let Some(t) = self.tagged.get(&(s, entry)) {
            return Ok(*t);
        }
        let leaf = self.resolve(s)?.leaf.clone();
        let t = self.source(Source {
            leaf,
            via_view: Some(name.to_vec()),
        })?;
        self.tagged.insert((s, entry), t);
        Ok(t)
    }

    /// A recursive CTE: the reference reads the anchor's lineage joined with the recursive term's, round
    /// after round until a round changes nothing (the sets are finite and only grow). Each round drops
    /// what the previous one computed.
    fn union_loop(
        &mut self,
        id: u32,
        anchor: &Arc<LogicalPlan>,
        recursive: &Arc<LogicalPlan>,
        output: &[crate::BoundColumn],
        offset: u64,
    ) -> Result<Vec<(AttrId, Info)>, KrishivError> {
        let a = self.plan(anchor, offset)?;
        if a.len() != output.len() {
            return Err(internal("a recursive CTE's anchor of another width"));
        }
        let mut current: Vec<Info> = a.iter().map(|(_, i)| Arc::clone(i)).collect();
        let mut settled = false;
        for _ in 0..LOOP_ROUNDS {
            self.loops.insert((id, offset), current.clone());
            let mark = self.journal.len();
            let r = self.plan(recursive, offset)?;
            if r.len() != output.len() {
                return Err(internal("a recursive term of another width"));
            }
            // Merged into the last round's lineage, so a rule that is not monotone (a `_metadata` field's
            // narrowing) cannot make the rounds oscillate.
            let next: Vec<Info> = current
                .iter()
                .zip(a.iter().zip(r.iter()))
                .map(|(c, ((_, x), (_, y)))| merge(c, &merge(x, y)))
                .collect();
            if next == current {
                settled = true;
                break;
            }
            current = next;
            for m in self.journal.drain(mark..) {
                match m {
                    Memo::Plan(k) => {
                        self.plans.remove(&k);
                    }
                    Memo::Expr(k) => {
                        self.exprs.remove(&k);
                    }
                }
            }
        }
        self.loops.remove(&(id, offset));
        if !settled {
            return Err(internal("a recursive CTE's lineage did not settle"));
        }
        output
            .iter()
            .zip(current)
            .map(|(c, i)| Ok((shifted(c.id, offset)?, i)))
            .collect()
    }

    /// Keeps the memo entry `key` of an expression for a fixed point's rounds.
    pub(super) fn remember_expr(&mut self, key: (usize, u64), info: &Info) {
        self.exprs.insert(key, Arc::clone(info));
        self.journal.push(Memo::Expr(key));
    }

    /// The output of the subquery plan `p` (relation memo and depth as any child).
    pub(super) fn subquery(
        &mut self,
        p: &Arc<LogicalPlan>,
        offset: u64,
    ) -> Result<Vec<Info>, KrishivError> {
        Ok(self
            .plan(p, offset)?
            .iter()
            .map(|(_, i)| Arc::clone(i))
            .collect())
    }
}

/// Whether `condition` of a semi or anti join between `l` and `r` is the set operations' rewrite: `l_i <=> r_i`
/// for every position `i`, joined by `AND` in order (`setops.rs`, `set_op_distinct`). Another semi join of
/// that exact shape reads as a set operation too, the conservative direction (more sources, never fewer).
fn set_op_condition(
    condition: &JoinCondition,
    l: &[(AttrId, Info)],
    r: &[(AttrId, Info)],
    offset: u64,
) -> bool {
    /// The column id of `e`, if it is a column.
    fn column(e: &Expr) -> Option<AttrId> {
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
            _ => None,
        }
    }
    /// The `<=>` pairs of `e`, left to right, if `e` is a conjunction of them.
    fn pairs(e: &Expr, out: &mut Vec<(AttrId, AttrId)>) -> bool {
        let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node else {
            return false;
        };
        match (&f.id, f.args.as_slice()) {
            (FunctionId::Builtin(Builtin::And), [x, y]) => pairs(x, out) && pairs(y, out),
            (FunctionId::Builtin(Builtin::EqNullSafe), [x, y]) => match (column(x), column(y)) {
                (Some(x), Some(y)) => {
                    out.push((x, y));
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }
    let JoinCondition::On(e) = condition else {
        // `x INTERSECT y` over relations with no column (`SELECT FROM ...` is not SQL, so only a DataFrame).
        return l.is_empty() && r.is_empty();
    };
    let mut found = Vec::new();
    // The condition's ids are as written; the outputs' are shifted by `offset`.
    pairs(e, &mut found)
        && found.len() == l.len()
        && l.len() == r.len()
        && found
            .iter()
            .zip(l.iter().zip(r))
            .all(|((x, y), ((lx, _), (ry, _)))| {
                shifted(*x, offset).is_ok_and(|x| x == *lx)
                    && shifted(*y, offset).is_ok_and(|y| y == *ry)
            })
}

/// `acc` with `o`'s columns joined in by position (a union's inputs); the ids stay the first input's.
fn positional(
    acc: Vec<(AttrId, Info)>,
    o: &[(AttrId, Info)],
) -> Result<Vec<(AttrId, Info)>, KrishivError> {
    if acc.len() != o.len() {
        return Err(internal("union inputs of different widths"));
    }
    Ok(acc
        .into_iter()
        .zip(o)
        .map(|((id, a), (_, b))| (id, merge(&a, b)))
        .collect())
}

/// Every grouping expression of `g`.
#[deny(clippy::wildcard_enum_match_arm)]
fn grouping_exprs(g: &Grouping) -> Vec<&Arc<Expr>> {
    match g {
        Grouping::GroupBy(e) | Grouping::Rollup(e) | Grouping::Cube(e) => e.iter().collect(),
        Grouping::GroupingSets { exprs, sets } => {
            exprs.iter().chain(sets.iter().flatten()).collect()
        }
    }
}

impl Walker<'_> {
    /// A scan's columns: leaves of the table, or of each path read (the `_metadata` column one leaf per
    /// field).
    #[deny(clippy::wildcard_enum_match_arm)]
    fn scan(&mut self, s: &Scan, offset: u64) -> Result<Vec<(AttrId, Info)>, KrishivError> {
        let leaves = |column: &str| -> Vec<Leaf> {
            match &s.source {
                ReadSource::Table { name, .. } => vec![Leaf::Table {
                    name: name.clone(),
                    column: column.to_owned(),
                }],
                ReadSource::Files { paths, .. } => paths
                    .iter()
                    .map(|p| Leaf::Path {
                        url: p.to_string(),
                        column: column.to_owned(),
                    })
                    .collect(),
            }
        };
        let metadata = s.metadata_column().map(|c| c.id);
        let mut out = Vec::with_capacity(s.output.len());
        for c in &s.output {
            let found = match (&c.data_type, metadata == Some(c.id)) {
                (krishiv_types::DataType::Struct(st), true) => st
                    .fields()
                    .iter()
                    .flat_map(|f| leaves(&format!("{}.{}", c.name, f.name())))
                    .collect::<Vec<_>>(),
                _ => leaves(&c.name),
            };
            let mut sources = BTreeSet::new();
            for leaf in found {
                sources.insert(self.source(Source {
                    leaf,
                    via_view: None,
                })?);
            }
            out.push((
                shifted(c.id, offset)?,
                Arc::new(ColInfo {
                    sources,
                    transform: Transform::Identity,
                }),
            ));
        }
        Ok(out)
    }
}
