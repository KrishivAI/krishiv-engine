// SPDX-License-Identifier: Apache-2.0
//! Build-side projection (S4', design `P2b-S4p-q09-amount-below-join.md` in
//! krishiv-internal): an expression above an inner hash join that reads only
//! the build input is computed on the build input, so the hash table holds
//! its result instead of the columns it reads.
//!
//! TPC-H q09's `amount = l_extendedprice * (1 - l_discount) - ps_supplycost *
//! l_quantity` reads four 16-byte decimals of the orders join's build side and
//! gives one: the table loses 48 bytes a row (15.7 GB at SF1000).
//!
//! # How
//!
//! A consumer, a `Project` or an `Aggregate` directly above an inner join
//! (through projections that only select columns), leaves a request keyed by
//! the join's inputs ([`Planner::request_moves`]). It builds no logical node:
//! the planner's statistics are keyed by node address, and a new node would
//! read default estimates. The join plans its original inputs, chooses its
//! build side as before, and then ([`Planner::project_build`]) wraps the
//! planned build in a projection that computes the moved expressions and
//! drops the columns only they read. The consumer and the projections in
//! between then read the new columns ([`Planner::moved_exprs`]).
//!
//! # What moves
//!
//! The maximal subexpressions of the consumer's expressions (of an
//! aggregate's call arguments only: its keys stay) that read at least one
//! column, read only build columns, are not a bare column, and may run on any
//! row ([`krishiv_plan::runs_on_any_row`]: deterministic, no subquery, raises
//! on no row, which exact decimal arithmetic proves with ANSI on too). They
//! now run on build rows that match nothing, which cannot be observed. Their
//! results must be fixed-width. Candidates that free a common column form a
//! group; a group moves only if it frees a column and its results are
//! narrower than the columns it frees.
//!
//! Results, errors and the output schema are Spark's: the moved expressions
//! keep their resolved types and are evaluated on a superset of the rows
//! Spark evaluates them on. `krishiv.exec.join.buildProjection=false`
//! turns the rule off.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::sync::Arc;

use krishiv_common::config::{ConfigValue, EXEC_JOIN_BUILD_PROJECTION};
use krishiv_plan::{
    AttrId, BoundColumn, Expr, ExprNode, FunctionKind, Grouping, JoinKind, LogicalPlan, PlanNode,
    ResolvedExpr, identity_string, runs_on_any_row,
};
use krishiv_types::DataType;

use super::build::{Planned, Planner};
use super::join::default_size;
use crate::error::Result;

/// What a consumer above an inner join asks of it.
#[derive(Debug)]
pub(crate) struct Request {
    /// The consumer, by address.
    consumer: usize,
    /// The column-selection projections between the consumer and the join,
    /// by address.
    selections: Vec<usize>,
    /// The consumer's expressions.
    exprs: Vec<Arc<Expr>>,
    /// Only aggregate-call arguments may move (an aggregate consumer).
    calls_only: bool,
    /// Expressions of the consumer that read columns but never move (an
    /// aggregate's grouping keys).
    pinned: Vec<Arc<Expr>>,
}

/// What a join moved below itself, for a consumer or a selection above it.
#[derive(Debug)]
pub(crate) struct Applied {
    /// Build columns the projection no longer passes.
    dropped: HashSet<AttrId>,
    /// Each moved expression, by [`identity_string`], and the column it
    /// became.
    columns: Vec<(String, BoundColumn)>,
}

/// The pending requests and their results.
#[derive(Debug, Default)]
pub(crate) struct Moves {
    requests: HashMap<(usize, usize), Request>,
    applied: HashMap<usize, Arc<Applied>>,
}

fn addr<T>(p: &Arc<T>) -> usize {
    Arc::as_ptr(p).addr()
}

fn is_column(e: &Expr) -> bool {
    matches!(e.node, ExprNode::Resolved(ResolvedExpr::Column(_)))
}

impl Planner<'_> {
    /// Plans `Project { input, exprs }`, moving what it can below the join
    /// under it.
    pub(super) fn plan_project(
        &mut self,
        node: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
        exprs: &[Arc<Expr>],
    ) -> Result<Planned> {
        if !exprs.iter().all(|e| is_column(e)) {
            self.request_moves(node, input, exprs, false, Vec::new());
        }
        let planned = self.plan(input)?;
        let exprs = self.moved_exprs(node, exprs);
        self.project(planned, &exprs)
    }

    /// Plans `Aggregate { input, grouping, aggregates }`, moving call
    /// arguments below the join under it.
    pub(super) fn plan_aggregate(
        &mut self,
        node: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
        grouping: &Grouping,
        aggregates: &[Arc<Expr>],
    ) -> Result<Planned> {
        if let Grouping::GroupBy(keys) = grouping {
            self.request_moves(node, input, aggregates, true, keys.clone());
        }
        let planned = self.plan(input)?;
        let aggregates = self.moved_exprs(node, aggregates);
        self.aggregate(planned, grouping, &aggregates)
    }

    /// Leaves a request at the inner join under `input`, if there is one
    /// through column selections. The outermost consumer's request wins.
    fn request_moves(
        &mut self,
        node: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
        exprs: &[Arc<Expr>],
        calls_only: bool,
        pinned: Vec<Arc<Expr>>,
    ) {
        let on = self
            .cx
            .config
            .get(&EXEC_JOIN_BUILD_PROJECTION)
            .and_then(ConfigValue::as_bool)
            .unwrap_or(true);
        if !on {
            return;
        }
        let mut selections = Vec::new();
        let mut n = input;
        loop {
            match &n.node {
                PlanNode::Project { input, exprs } if exprs.iter().all(|e| is_column(e)) => {
                    selections.push(addr(n));
                    n = input;
                }
                PlanNode::Join {
                    left,
                    right,
                    kind: JoinKind::Inner,
                    ..
                } => {
                    self.moves
                        .requests
                        .entry((addr(left), addr(right)))
                        .or_insert_with(|| Request {
                            consumer: addr(node),
                            selections,
                            exprs: exprs.to_vec(),
                            calls_only,
                            pinned,
                        });
                    return;
                }
                _ => return,
            }
        }
    }

    /// The consumer's or selection's expressions over what the join below
    /// moved, or `exprs` when it moved nothing.
    fn moved_exprs(&mut self, node: &Arc<LogicalPlan>, exprs: &[Arc<Expr>]) -> Vec<Arc<Expr>> {
        let Some(applied) = self.moves.applied.remove(&addr(node)) else {
            return exprs.to_vec();
        };
        if exprs.iter().all(|e| is_column(e)) {
            // A selection: pass the new columns instead of the dropped ones.
            let mut out: Vec<Arc<Expr>> = exprs
                .iter()
                .filter(|e| match &e.node {
                    ExprNode::Resolved(ResolvedExpr::Column(c)) => !applied.dropped.contains(&c.id),
                    _ => true,
                })
                .cloned()
                .collect();
            out.extend(applied.columns.iter().map(|(_, c)| column(c)));
            return out;
        }
        let by_text: HashMap<&str, &BoundColumn> = applied
            .columns
            .iter()
            .map(|(k, c)| (k.as_str(), c))
            .collect();
        exprs
            .iter()
            .map(|e| {
                let rewritten: std::result::Result<Arc<Expr>, Infallible> = Arc::clone(e)
                    .transform_down(&mut |x| {
                        Ok(match by_text.get(identity_string(&x).as_str()) {
                            Some(c) => column(c),
                            None => x,
                        })
                    });
                match rewritten {
                    Ok(x) => x,
                    Err(never) => match never {},
                }
            })
            .collect()
    }

    /// Wraps the planned build side of the inner join of `left` and `right`
    /// in the projection its consumer's request allows, or returns it as it
    /// is. `logical` is the build input; `fixed` holds what the join itself
    /// reads (its keys and residual). The text, when it moved something, is
    /// for the join's `EXPLAIN` line (`computes on build: ...`).
    pub(super) fn project_build(
        &mut self,
        (left, right): (&Arc<LogicalPlan>, &Arc<LogicalPlan>),
        logical: &Arc<LogicalPlan>,
        build: Planned,
        fixed: &[&Expr],
    ) -> Result<(Planned, Option<String>)> {
        let Some(req) = self.moves.requests.remove(&(addr(left), addr(right))) else {
            return Ok((build, None));
        };
        let ansi = self.env.ansi;
        let build_ids: HashSet<AttrId> = build.layout.iter().copied().collect();
        let mut found = Found::default();
        for e in &req.exprs {
            found.walk(e, !req.calls_only, &build_ids, ansi);
        }
        for e in req
            .pinned
            .iter()
            .map(AsRef::as_ref)
            .chain(fixed.iter().copied())
        {
            found.outside(e);
        }
        if found.subquery || found.candidates.is_empty() {
            return Ok((build, None));
        }
        let Some(output) = logical.resolved_output() else {
            return Ok((build, None));
        };
        let moved = self.choose(&found, &build);
        if moved.is_empty() {
            return Ok((build, None));
        }
        let dropped: HashSet<AttrId> = moved
            .iter()
            .flat_map(|m| m.reads.iter().copied())
            .filter(|id| !found.read_outside.contains(id))
            .collect();
        let kept: Vec<&BoundColumn> = output
            .iter()
            .filter(|c| build_ids.contains(&c.id) && !dropped.contains(&c.id))
            .collect();
        // Every passed build column must be one of the logical outputs.
        if kept.len() + dropped.len() != build.layout.len() {
            return Ok((build, None));
        }
        let freed_bytes: u64 = dropped.iter().map(|id| column_width(&build, *id)).sum();
        let result_bytes: u64 = moved.iter().map(|m| m.width).sum();
        let mut exprs: Vec<Arc<Expr>> = kept.into_iter().map(column).collect();
        let mut columns = Vec::with_capacity(moved.len());
        for m in &moved {
            let out = match &m.expr.node {
                ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => output.clone(),
                _ => BoundColumn {
                    id: self.fresh_id(),
                    name: m.expr.to_string(),
                    data_type: m.data_type.clone(),
                    nullable: m.nullable,
                },
            };
            exprs.push(match &m.expr.node {
                ExprNode::Resolved(ResolvedExpr::Alias { .. }) => Arc::clone(&m.expr),
                _ => Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                    expr: Arc::clone(&m.expr),
                    output: out.clone(),
                }))),
            });
            columns.push((identity_string(&m.expr), out));
        }
        let names: Vec<String> = columns
            .iter()
            .map(|(_, c)| format!("{}#{}", c.name, c.id.0))
            .collect();
        let text = format!(
            "computes on build: {} ({freed_bytes} B -> {result_bytes} B a row)",
            names.join(", ")
        );
        let projected = self.project(build, &exprs)?;
        let applied = Arc::new(Applied { dropped, columns });
        self.moves
            .applied
            .insert(req.consumer, Arc::clone(&applied));
        for s in req.selections {
            self.moves.applied.insert(s, Arc::clone(&applied));
        }
        Ok((projected, Some(text)))
    }

    /// The candidates whose groups pass the width gate.
    fn choose<'f>(&self, found: &'f Found, build: &Planned) -> Vec<&'f Candidate> {
        let width = |id: &AttrId| column_width(build, *id);
        // Groups: candidates connected by a column only candidates read.
        let n = found.candidates.len();
        let mut group: Vec<usize> = (0..n).collect();
        let mut owner: HashMap<AttrId, usize> = HashMap::new();
        for (i, c) in found.candidates.iter().enumerate() {
            for id in c.reads.iter().filter(|id| !found.read_outside.contains(id)) {
                match owner.get(id) {
                    Some(&j) => {
                        let (a, b) = (root(&group, i), root(&group, j));
                        group[a] = b;
                    }
                    None => {
                        owner.insert(*id, i);
                    }
                }
            }
        }
        let mut by_group: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            by_group.entry(root(&group, i)).or_default().push(i);
        }
        let mut chosen = Vec::new();
        let mut groups: Vec<Vec<usize>> = by_group.into_values().collect();
        groups.sort_unstable();
        for members in groups {
            let freed: HashSet<AttrId> = members
                .iter()
                .flat_map(|&i| found.candidates[i].reads.iter().copied())
                .filter(|id| !found.read_outside.contains(id))
                .collect();
            let freed_bytes: u64 = freed.iter().map(width).sum();
            let result_bytes: u64 = members.iter().map(|&i| found.candidates[i].width).sum();
            if !freed.is_empty() && result_bytes < freed_bytes {
                chosen.extend(members.iter().map(|&i| &found.candidates[i]));
            }
        }
        chosen
    }
}

/// Bytes of build column `id` in the hash table: its fixed width, else
/// Spark's default size.
fn column_width(build: &Planned, id: AttrId) -> u64 {
    build.layout.iter().position(|a| *a == id).map_or(0, |i| {
        let t = build.schema.field(i).data_type();
        t.primitive_width()
            .and_then(|w| u64::try_from(w).ok())
            .unwrap_or_else(|| default_size(t))
    })
}

fn root(group: &[usize], mut i: usize) -> usize {
    while group[i] != i {
        i = group[i];
    }
    i
}

fn column(c: &BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
        c.clone(),
    ))))
}

/// One expression that may move.
#[derive(Debug)]
struct Candidate {
    expr: Arc<Expr>,
    /// The build columns it reads.
    reads: HashSet<AttrId>,
    data_type: DataType,
    nullable: bool,
    /// Bytes of its result in the hash table.
    width: u64,
}

/// The consumer's expressions split into candidates and the columns read
/// outside them.
#[derive(Debug, Default)]
struct Found {
    candidates: Vec<Candidate>,
    seen: HashSet<String>,
    read_outside: HashSet<AttrId>,
    /// A subquery reads columns this walk does not see: nothing moves.
    subquery: bool,
}

impl Found {
    fn walk(&mut self, e: &Arc<Expr>, movable: bool, build: &HashSet<AttrId>, ansi: bool) {
        if movable && let Some(c) = candidate(e, build, ansi) {
            if self.seen.insert(identity_string(e)) {
                self.candidates.push(c);
            }
            return;
        }
        match &e.node {
            ExprNode::Resolved(
                ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c) | ResolvedExpr::LambdaVar(c),
            ) => {
                self.read_outside.insert(c.id);
                return;
            }
            ExprNode::Subquery { .. } => {
                self.subquery = true;
                return;
            }
            _ => {}
        }
        let inner = movable
            || matches!(
                &e.node,
                ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate
            );
        e.for_each_child(&mut |c| self.walk(c, inner, build, ansi));
    }

    /// Records the columns `e` reads as read outside the candidates.
    fn outside(&mut self, e: &Expr) {
        match &e.node {
            ExprNode::Resolved(
                ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c) | ResolvedExpr::LambdaVar(c),
            ) => {
                self.read_outside.insert(c.id);
            }
            ExprNode::Subquery { .. } => self.subquery = true,
            _ => e.for_each_child(&mut |c| self.outside(c)),
        }
    }
}

/// `e` as a candidate: it reads only build columns, at least one, is not a
/// column (under casts or an alias), may run on any row and has a
/// fixed-width result.
fn candidate(e: &Arc<Expr>, build: &HashSet<AttrId>, ansi: bool) -> Option<Candidate> {
    if under_casts(e) {
        return None;
    }
    let mut reads = HashSet::new();
    if !build_only(e, build, &mut reads) || reads.is_empty() {
        return None;
    }
    let (data_type, nullable) = result_type(e)?;
    let width = fixed_width(&data_type)?;
    runs_on_any_row(e, ansi).then(|| Candidate {
        expr: Arc::clone(e),
        reads,
        data_type,
        nullable,
        width,
    })
}

fn under_casts(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(_)) => true,
        ExprNode::Cast { expr, .. } | ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => {
            under_casts(expr)
        }
        _ => false,
    }
}

/// Whether every column `e` reads is a build column (no outer reference, no
/// lambda variable, no subquery), collecting them.
fn build_only(e: &Expr, build: &HashSet<AttrId>, reads: &mut HashSet<AttrId>) -> bool {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            reads.insert(c.id);
            build.contains(&c.id)
        }
        ExprNode::Resolved(ResolvedExpr::OuterColumn(_) | ResolvedExpr::LambdaVar(_))
        | ExprNode::Subquery { .. } => false,
        _ => {
            let mut all = true;
            e.for_each_child(&mut |c| all &= build_only(c, build, reads));
            all
        }
    }
}

fn result_type(e: &Expr) -> Option<(DataType, bool)> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Scalar => {
            Some((f.data_type.clone(), f.nullable))
        }
        ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => {
            Some((output.data_type.clone(), output.nullable))
        }
        ExprNode::Cast { to, .. } => Some((to.clone(), true)),
        _ => None,
    }
}

/// Bytes of a fixed-width value of `t` in the hash table; `None` for the
/// variable-width types, which never move.
fn fixed_width(t: &DataType) -> Option<u64> {
    Some(match t {
        DataType::Boolean | DataType::Byte => 1,
        DataType::Short => 2,
        DataType::Integer | DataType::Float | DataType::Date => 4,
        DataType::Long | DataType::Double | DataType::Timestamp | DataType::TimestampNtz => 8,
        // The carriers: `Decimal64` up to 18 digits, `Decimal128` above.
        DataType::Decimal(d) if d.precision() <= 18 => 8,
        DataType::Decimal(_) => 16,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_plan::{Builtin, CastMode, FunctionId, Literal, ResolvedFunction};
    use krishiv_types::DecimalType;

    fn dec(p: u32, s: u32) -> DataType {
        DataType::Decimal(DecimalType::try_new(p, s).unwrap())
    }

    fn col(id: u64, t: DataType) -> Arc<Expr> {
        column(&BoundColumn {
            id: AttrId(id),
            name: format!("c{id}"),
            data_type: t,
            nullable: true,
        })
    }

    fn op(b: Builtin, args: Vec<Arc<Expr>>, t: DataType) -> Arc<Expr> {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Builtin(b),
                kind: FunctionKind::Scalar,
                args,
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: t,
                nullable: true,
                deterministic: true,
            },
        ))))
    }

    fn ids(v: &[u64]) -> HashSet<AttrId> {
        v.iter().copied().map(AttrId).collect()
    }

    #[test]
    fn an_exact_decimal_product_of_build_columns_is_a_candidate() {
        let e = op(
            Builtin::Multiply,
            vec![col(1, dec(15, 2)), col(2, dec(15, 2))],
            dec(31, 4),
        );
        let c = candidate(&e, &ids(&[1, 2]), true).unwrap();
        assert_eq!(c.reads, ids(&[1, 2]));
        assert_eq!(c.width, 16);
        let narrow = op(
            Builtin::Add,
            vec![col(1, dec(10, 2)), col(2, dec(10, 2))],
            dec(11, 2),
        );
        assert_eq!(candidate(&narrow, &ids(&[1, 2]), true).unwrap().width, 8);
    }

    #[test]
    fn columns_mixed_sides_and_raising_expressions_are_not_candidates() {
        let build = ids(&[1, 2]);
        assert!(candidate(&col(1, dec(15, 2)), &build, true).is_none());
        let cast = Arc::new(Expr::new(ExprNode::Cast {
            expr: col(1, dec(15, 2)),
            to: dec(20, 2),
            mode: CastMode::Ansi,
        }));
        assert!(candidate(&cast, &build, true).is_none());
        let mixed = op(
            Builtin::Multiply,
            vec![col(1, dec(15, 2)), col(3, dec(15, 2))],
            dec(31, 4),
        );
        assert!(candidate(&mixed, &build, true).is_none());
        // Integer addition may overflow under ANSI, not with it off.
        let int = op(
            Builtin::Add,
            vec![col(1, DataType::Integer), col(2, DataType::Integer)],
            DataType::Integer,
        );
        assert!(candidate(&int, &build, true).is_none());
        assert!(candidate(&int, &build, false).is_some());
        // No column at all.
        let lit = op(
            Builtin::Add,
            vec![
                Arc::new(Expr::new(ExprNode::Literal(Literal::Integer(1)))),
                Arc::new(Expr::new(ExprNode::Literal(Literal::Integer(2)))),
            ],
            DataType::Integer,
        );
        assert!(candidate(&lit, &build, false).is_none());
    }

    #[test]
    fn the_walk_splits_a_mixed_expression_and_records_outside_reads() {
        // (c1 * c2) + c3, with c3 on the probe side: c1 * c2 moves, c3 is read outside.
        let prod = op(
            Builtin::Multiply,
            vec![col(1, dec(15, 2)), col(2, dec(15, 2))],
            dec(31, 4),
        );
        let sum = op(
            Builtin::Add,
            vec![Arc::clone(&prod), col(3, dec(15, 2))],
            dec(32, 4),
        );
        let mut f = Found::default();
        f.walk(&sum, true, &ids(&[1, 2]), true);
        assert_eq!(f.candidates.len(), 1);
        assert_eq!(
            identity_string(&f.candidates[0].expr),
            identity_string(&prod)
        );
        assert_eq!(f.read_outside, ids(&[3]));
        // The same product twice is one candidate.
        let twice = op(Builtin::Add, vec![Arc::clone(&prod), prod], dec(32, 4));
        let mut g = Found::default();
        g.walk(&twice, true, &ids(&[1, 2]), true);
        assert_eq!(g.candidates.len(), 1);
    }

    #[test]
    fn an_aggregate_consumer_moves_call_arguments_only() {
        let prod = op(
            Builtin::Multiply,
            vec![col(1, dec(15, 2)), col(2, dec(15, 2))],
            dec(31, 4),
        );
        let mut f = Found::default();
        // Not movable at the top: a grouping-like expression stays.
        f.walk(&prod, false, &ids(&[1, 2]), true);
        assert!(f.candidates.is_empty());
        assert_eq!(f.read_outside, ids(&[1, 2]));
    }
}
