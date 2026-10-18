// SPDX-License-Identifier: Apache-2.0
//! Joins: hash joins on the equality conjuncts of the condition, with the
//! rest as the residual filter.
//!
//! - The **build** side is the right input; the optimizer's join ordering
//!   puts the smaller input of an inner join there. A right outer join is
//!   a left outer join with the sides swapped (the layout keeps the column
//!   order right).
//! - `NOT IN (subquery)` arrives as a left anti join on Spark's null-aware
//!   key `(a = b) OR isnull(a = b)` and runs as the null-aware anti join.
//! - A join without an equality conjunct (a cross join, `a < b`, a
//!   disjunction) is a nested-loop join building the right input, as
//!   Spark's broadcast nested-loop join.
//! - A full outer join needs the build-side outer join (P2-T052b) and is
//!   `KRISHIV_UNSUPPORTED_FEATURE` for now.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_plan::{
    AttrId, Builtin, Expr, ExprNode, FunctionId, JoinCondition, JoinKind, LogicalPlan,
    ResolvedExpr, ResolvedFunction,
};

use super::build::{Planned, Planner, concat_schemas, internal, unsupported};
use super::eval::{self, Compiled, Input};
use super::explain::ExplainNode;
use crate::error::Result;
use crate::join::{
    HashJoin, HashJoinNode, JoinSpec, JoinType, NestedLoopJoin, NestedLoopJoinNode, NestedLoopSpec,
};
use crate::operators::Predicate;

impl Planner<'_> {
    pub(crate) fn join(
        &mut self,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: JoinKind,
        condition: &JoinCondition,
    ) -> Result<Planned> {
        let on = match condition {
            JoinCondition::On(e) => Some(e),
            JoinCondition::None => None,
            _ => return Err(unsupported("JOIN ... USING in physical plans")),
        };
        let (probe, build, join_type) = match kind {
            JoinKind::Inner | JoinKind::Cross => (left, right, JoinType::Inner),
            JoinKind::LeftOuter => (left, right, JoinType::LeftOuter),
            JoinKind::RightOuter => (right, left, JoinType::LeftOuter),
            JoinKind::LeftSemi => (left, right, JoinType::LeftSemi),
            JoinKind::LeftAnti => (left, right, JoinType::LeftAnti),
            _ => return Err(unsupported("FULL OUTER JOIN")),
        };
        let probe = self.plan(probe)?;
        let build = self.plan(build)?;
        let probe_ids: HashSet<AttrId> = probe.layout.iter().copied().collect();
        let build_ids: HashSet<AttrId> = build.layout.iter().copied().collect();
        let mut conjuncts = Vec::new();
        if let Some(on) = on {
            split_and(on, &mut conjuncts);
        }
        let mut join_type = join_type;
        let mut keys: Vec<(Arc<Expr>, Arc<Expr>, bool)> = Vec::new();
        let mut residual: Vec<&Expr> = Vec::new();
        if join_type == JoinType::LeftAnti
            && let [only] = conjuncts.as_slice()
            && let Some((l, r)) = null_aware_key(only)
            && let Some((p, b)) = sides(l, r, &probe_ids, &build_ids)
        {
            join_type = JoinType::NullAwareAnti;
            keys.push((p, b, false));
        } else {
            for c in &conjuncts {
                let key = equality(c).and_then(|(l, r, null_safe)| {
                    sides(l, r, &probe_ids, &build_ids).map(|(p, b)| (p, b, null_safe))
                });
                match key {
                    Some(k) => keys.push(k),
                    None => residual.push(c),
                }
            }
            if keys.is_empty() {
                // `(k = x AND ...) OR (k = x AND ...)` (TPC-H Q19): an
                // equality every disjunct holds is a key; the disjunction
                // stays as the residual.
                for c in &residual {
                    for e in common_equalities(c) {
                        if let Some((l, r, null_safe)) = equality(e)
                            && let Some((p, b)) = sides(l, r, &probe_ids, &build_ids)
                        {
                            keys.push((p, b, null_safe));
                        }
                    }
                }
            }
        }
        if keys.is_empty() {
            return self.nested_loop(probe, build, join_type, &residual);
        }
        let probe_exprs: Vec<Arc<Expr>> = keys.iter().map(|k| Arc::clone(&k.0)).collect();
        let build_exprs: Vec<Arc<Expr>> = keys.iter().map(|k| Arc::clone(&k.1)).collect();
        let (probe, probe_keys) = self.with_columns(probe, &probe_exprs)?;
        let (build, build_keys) = self.with_columns(build, &build_exprs)?;
        let filter = self.residual(&probe, &build, &residual)?;
        let null_safe = keys.iter().map(|k| k.2).collect();
        let join = HashJoin::try_new(JoinSpec {
            join_type,
            probe_schema: Arc::clone(&probe.schema),
            build_schema: Arc::clone(&build.schema),
            probe_keys,
            build_keys,
            null_safe,
            filter,
            spill: self.cx.spill.clone(),
        })?;
        let mut layout = probe.layout.clone();
        if matches!(join_type, JoinType::Inner | JoinType::LeftOuter) {
            layout.extend(&build.layout);
        }
        let mut detail = format!(
            "{join_type:?}, [{}] = [{}]",
            text(&probe_exprs),
            text(&build_exprs)
        );
        if !residual.is_empty() {
            let r: Vec<String> = residual.iter().map(ToString::to_string).collect();
            detail.push_str(&format!(", residual: {}", r.join(" AND ")));
        }
        Ok(Planned {
            schema: join.schema(),
            node: Arc::new(HashJoinNode {
                join,
                build: build.node,
                probe: probe.node,
            }),
            layout,
            explain: ExplainNode::new("HashJoin", detail, vec![probe.explain, build.explain]),
        })
    }
}

impl Planner<'_> {
    /// The conjunction of `residual` over probe then build columns.
    fn residual(
        &self,
        probe: &Planned,
        build: &Planned,
        residual: &[&Expr],
    ) -> Result<Option<Arc<dyn Predicate>>> {
        if residual.is_empty() {
            return Ok(None);
        }
        let both = concat_schemas(&probe.schema, &build.schema);
        let layout: Vec<AttrId> = probe.layout.iter().chain(&build.layout).copied().collect();
        let input = Input {
            layout: &layout,
            schema: &both,
            env: &self.env,
            subqueries: &[],
        };
        let mut exprs = residual
            .iter()
            .map(|c| eval::compile(c, input))
            .collect::<Result<Vec<_>>>()?;
        let mut all = exprs.pop().ok_or_else(|| internal("residual"))?;
        while let Some(e) = exprs.pop() {
            all = eval::and(e, all);
        }
        Ok(Some(Arc::new(Compiled(all)) as Arc<dyn Predicate>))
    }

    /// A join without an equality conjunct: a nested-loop join building the
    /// right input.
    fn nested_loop(
        &mut self,
        probe: Planned,
        build: Planned,
        join_type: JoinType,
        residual: &[&Expr],
    ) -> Result<Planned> {
        let filter = self.residual(&probe, &build, residual)?;
        let join = NestedLoopJoin::try_new(NestedLoopSpec {
            join_type,
            probe_schema: Arc::clone(&probe.schema),
            build_schema: Arc::clone(&build.schema),
            filter,
        })?;
        let mut layout = probe.layout.clone();
        if matches!(join_type, JoinType::Inner | JoinType::LeftOuter) {
            layout.extend(&build.layout);
        }
        let r: Vec<String> = residual.iter().map(ToString::to_string).collect();
        let detail = if r.is_empty() {
            format!("{join_type:?}")
        } else {
            format!("{join_type:?}, {}", r.join(" AND "))
        };
        Ok(Planned {
            schema: join.schema(),
            node: Arc::new(NestedLoopJoinNode {
                join,
                build: build.node,
                probe: probe.node,
            }),
            layout,
            explain: ExplainNode::new("NestedLoopJoin", detail, vec![probe.explain, build.explain]),
        })
    }
}

fn text(exprs: &[Arc<Expr>]) -> String {
    exprs
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn builtin(e: &Expr) -> Option<(Builtin, &[Arc<Expr>])> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
            id: FunctionId::Builtin(b),
            args,
            ..
        })) => Some((*b, args.as_slice())),
        _ => None,
    }
}

fn split_and<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match builtin(e) {
        Some((Builtin::And, [l, r])) => {
            split_and(l, out);
            split_and(r, out);
        }
        _ => out.push(e),
    }
}

/// The conjuncts every disjunct of `e` has (compared by their text).
fn common_equalities(e: &Expr) -> Vec<&Expr> {
    let mut disjuncts = Vec::new();
    split_or(e, &mut disjuncts);
    let Some((first, rest)) = disjuncts.split_first() else {
        return Vec::new();
    };
    if rest.is_empty() {
        return Vec::new();
    }
    let mut common = Vec::new();
    split_and(first, &mut common);
    common.retain(|c| {
        let text = c.to_string();
        rest.iter().all(|d| {
            let mut cs = Vec::new();
            split_and(d, &mut cs);
            cs.iter().any(|x| x.to_string() == text)
        })
    });
    common
}

fn split_or<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match builtin(e) {
        Some((Builtin::Or, [l, r])) => {
            split_or(l, out);
            split_or(r, out);
        }
        _ => out.push(e),
    }
}

/// `l = r` or `l <=> r` (null-safe).
fn equality(e: &Expr) -> Option<(&Arc<Expr>, &Arc<Expr>, bool)> {
    match builtin(e)? {
        (Builtin::Eq, [l, r]) => Some((l, r, false)),
        (Builtin::EqNullSafe, [l, r]) => Some((l, r, true)),
        _ => None,
    }
}

/// The operands of Spark's null-aware anti-join key
/// `(a = b) OR isnull(a = b)`.
fn null_aware_key(e: &Expr) -> Option<(&Arc<Expr>, &Arc<Expr>)> {
    let (Builtin::Or, [eq, is_null]) = builtin(e)? else {
        return None;
    };
    let (Builtin::IsNull, [inner]) = builtin(is_null)? else {
        return None;
    };
    let (l, r, false) = equality(eq)? else {
        return None;
    };
    let (il, ir, false) = equality(inner)? else {
        return None;
    };
    (il.to_string() == l.to_string() && ir.to_string() == r.to_string()).then_some((l, r))
}

/// The attributes `e` reads.
fn reads(e: &Expr, out: &mut HashSet<AttrId>) {
    if let ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) = &e.node {
        out.insert(c.id);
    }
    if matches!(e.node, ExprNode::Subquery { .. }) {
        // A subquery is not a join key.
        out.insert(AttrId(u64::MAX));
    }
    e.for_each_child(&mut |c| reads(c, out));
}

/// `(probe side, build side)` of an equality, when each operand reads only
/// one side.
fn sides(
    l: &Arc<Expr>,
    r: &Arc<Expr>,
    probe: &HashSet<AttrId>,
    build: &HashSet<AttrId>,
) -> Option<(Arc<Expr>, Arc<Expr>)> {
    let of = |e: &Expr| {
        let mut ids = HashSet::new();
        reads(e, &mut ids);
        ids
    };
    let (li, ri) = (of(l), of(r));
    let within =
        |ids: &HashSet<AttrId>, side: &HashSet<AttrId>| !ids.is_empty() && ids.is_subset(side);
    if within(&li, probe) && within(&ri, build) {
        Some((Arc::clone(l), Arc::clone(r)))
    } else if within(&ri, probe) && within(&li, build) {
        Some((Arc::clone(r), Arc::clone(l)))
    } else {
        None
    }
}
