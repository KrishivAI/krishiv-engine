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

use super::{
    ColumnLineage, FilterKind, Leaf, Lineage, LineageWarning, Source, Transform, ViewExpansion,
};
use crate::analyzer::AnalyzedPlan;
use crate::plan::{
    GeneratorKind, JoinCondition, JoinKind, LogicalPlan, PlanNode, ReadSource, Scan,
};
use crate::{AttrId, Expr, Grouping, MAX_DEPTH, generator_arg};

/// The lineage of one value: its leaves and the strongest step on any path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ColInfo {
    pub(super) sources: BTreeSet<Source>,
    pub(super) transform: Transform,
}

/// A shared [`ColInfo`].
pub(super) type Info = Arc<ColInfo>;

/// A relation's output: its global attribute ids with their lineage, in order.
type Out = Arc<Vec<(AttrId, Info)>>;

/// Rounds of a recursive CTE's fixed point before the walk gives up (the sets only grow and are finite,
/// so it never gets near).
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
    pub(super) env: HashMap<AttrId, Info>,
    plans: HashMap<(usize, u64), Out>,
    pub(super) exprs: HashMap<(usize, u64), Info>,
    journal: Vec<Memo>,
    /// The current lineage of each open recursive CTE, by (loop id, offset).
    loops: HashMap<(u32, u64), Vec<Info>>,
    pub(super) filters: BTreeSet<(FilterKind, Leaf)>,
    pub(super) warnings: BTreeSet<LineageWarning>,
    depth: usize,
    /// Relations computed (each (node, offset) at most once outside a fixed point's rounds).
    evaluations: usize,
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
    if a == b {
        return Arc::clone(a);
    }
    Arc::new(ColInfo {
        sources: a.sources.union(&b.sources).cloned().collect(),
        transform: a.transform.max(b.transform),
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

/// The lineage of `analyzed`, and the relations computed.
pub(super) fn run(analyzed: &AnalyzedPlan) -> Result<(Lineage, usize), KrishivError> {
    // A view that is not expanded must be found, or its base tables could show: fail closed.
    if analyzed
        .views
        .iter()
        .any(|v| !v.expands() && v.output.is_none())
    {
        return Err(internal("a view that is not expanded is not in the plan"));
    }
    let mut w = Walker {
        views: &analyzed.views,
        env: HashMap::new(),
        plans: HashMap::new(),
        exprs: HashMap::new(),
        journal: Vec::new(),
        loops: HashMap::new(),
        filters: BTreeSet::new(),
        warnings: BTreeSet::new(),
        depth: 0,
        evaluations: 0,
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
        .map(|(ordinal, ((_, info), col))| ColumnLineage {
            ordinal,
            name: col.name.clone(),
            sources: info.sources.clone(),
            transform: info.transform,
        })
        .collect();
    Ok((
        Lineage {
            columns,
            filters: w.filters,
            warnings: w.warnings,
        },
        w.evaluations,
    ))
}

impl Walker<'_> {
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
        let out = self.deeper(|w| w.node(p, offset))?;
        self.evaluations += 1;
        for (id, info) in &out {
            self.env.insert(*id, Arc::clone(info));
        }
        let out = Arc::new(out);
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
            self.filters.insert((kind, s.leaf.clone()));
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

    /// The rule of `p`'s kind.
    #[deny(clippy::wildcard_enum_match_arm)]
    fn node(&mut self, p: &LogicalPlan, offset: u64) -> Result<Vec<(AttrId, Info)>, KrishivError> {
        match &p.node {
            PlanNode::Scan(s) => scan(s, offset),
            PlanNode::OneRowRelation => Ok(Vec::new()),
            PlanNode::Range { output, .. } => Ok(vec![(
                shifted(output.id, offset)?,
                Arc::new(ColInfo {
                    sources: BTreeSet::new(),
                    transform: Transform::Generated,
                }),
            )]),
            PlanNode::Project { input, exprs } => {
                self.plan(input, offset)?;
                exprs.iter().map(|e| self.output(e, offset)).collect()
            }
            PlanNode::Filter { input, predicate } => {
                let out = self.plan(input, offset)?;
                self.filter_by(FilterKind::Filter, predicate, offset)?;
                Ok(out.to_vec())
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
                aggregates.iter().map(|e| self.output(e, offset)).collect()
            }
            PlanNode::PartialAggregate { .. } => Err(internal(
                "a partial aggregate (optimizer only) in an analyzed plan",
            )),
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => self.join(left, right, *kind, condition, offset),
            PlanNode::Sort { input, order, .. } => {
                let out = self.plan(input, offset)?;
                for o in order {
                    self.filter_by(FilterKind::Sort, &o.expr, offset)?;
                }
                Ok(out.to_vec())
            }
            PlanNode::Repartition { input, exprs } => {
                let out = self.plan(input, offset)?;
                for e in exprs {
                    self.filter_by(FilterKind::Sort, e, offset)?;
                }
                Ok(out.to_vec())
            }
            PlanNode::Limit { input, .. }
            | PlanNode::Offset { input, .. }
            | PlanNode::JoinHint { input, .. }
            | PlanNode::ReplicateRows { input, .. } => Ok(self.plan(input, offset)?.to_vec()),
            PlanNode::SubqueryAlias { input, alias, .. } => self.alias(p, input, alias, offset),
            PlanNode::WithCte { input, ctes } => {
                for c in ctes {
                    self.plan(&c.plan, offset)?;
                }
                Ok(self.plan(input, offset)?.to_vec())
            }
            PlanNode::CteRef(r) => {
                if r.source.len() != r.output.len() {
                    return Err(internal("a CTE reference of another width"));
                }
                // A `CteRef::source` is never shifted (see `PlanNode::Shift`).
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
                    .collect()
            }
            PlanNode::Shift { input, offset: by } => {
                let inner = offset
                    .checked_add(*by)
                    .ok_or_else(|| internal("a shift overflows"))?;
                Ok(self.plan(input, inner)?.to_vec())
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
                out.ok_or_else(|| internal("a union without inputs"))
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
                Ok(out)
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
                    .collect()
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
                Ok(out)
            }
            PlanNode::UnionLoop {
                id,
                anchor,
                recursive,
                output,
                ..
            } => self.union_loop(*id, anchor, recursive, output, offset),
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
                    .collect()
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
        let mut out = l.to_vec();
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
                let transform = if cond.sources.is_empty() {
                    Transform::Literal
                } else {
                    cond.transform.max(Transform::Expression)
                };
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

    /// A `SubqueryAlias`: a pass-through, unless it is a view of the side table. An expanded view tags its
    /// sources with the view (the outermost view wins); any other view is a leaf, and nothing under it is read.
    fn alias(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        alias: &str,
        offset: u64,
    ) -> Result<Vec<(AttrId, Info)>, KrishivError> {
        let view = if self.views.iter().any(|v| v.alias == alias) {
            let local: Option<Vec<AttrId>> = p
                .resolved_output()
                .map(|o| o.into_iter().map(|c| c.id).collect());
            self.views
                .iter()
                .find(|v| v.alias == alias && local.is_some() && v.output == local)
        } else {
            None
        };
        let Some(view) = view else {
            return Ok(self.plan(input, offset)?.to_vec());
        };
        if view.expands() {
            let name = view.name.clone();
            let out = self.plan(input, offset)?;
            return Ok(out
                .iter()
                .map(|(id, info)| {
                    let sources = info
                        .sources
                        .iter()
                        .map(|s| Source {
                            leaf: s.leaf.clone(),
                            via_view: Some(name.clone()),
                        })
                        .collect();
                    (
                        *id,
                        Arc::new(ColInfo {
                            sources,
                            transform: info.transform,
                        }),
                    )
                })
                .collect());
        }
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
                Ok((
                    shifted(c.id, offset)?,
                    Arc::new(ColInfo {
                        sources: BTreeSet::from([Source {
                            leaf,
                            via_view: None,
                        }]),
                        transform: Transform::View,
                    }),
                ))
            })
            .collect()
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
            let next: Vec<Info> = a
                .iter()
                .zip(r.iter())
                .map(|((_, x), (_, y))| merge(x, y))
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

/// A scan's columns: leaves of the table, or of each path read (the `_metadata` column one leaf per field).
#[deny(clippy::wildcard_enum_match_arm)]
fn scan(s: &Scan, offset: u64) -> Result<Vec<(AttrId, Info)>, KrishivError> {
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
    s.output
        .iter()
        .map(|c| {
            let sources: BTreeSet<Source> = match (&c.data_type, metadata == Some(c.id)) {
                (krishiv_types::DataType::Struct(st), true) => st
                    .fields()
                    .iter()
                    .flat_map(|f| leaves(&format!("{}.{}", c.name, f.name())))
                    .collect::<Vec<_>>(),
                _ => leaves(&c.name),
            }
            .into_iter()
            .map(|leaf| Source {
                leaf,
                via_view: None,
            })
            .collect();
            Ok((
                shifted(c.id, offset)?,
                Arc::new(ColInfo {
                    sources,
                    transform: Transform::Identity,
                }),
            ))
        })
        .collect()
}
