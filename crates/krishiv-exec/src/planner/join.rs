// SPDX-License-Identifier: Apache-2.0
//! Joins: hash joins on the equality conjuncts of the condition, with the
//! rest as the residual filter.
//!
//! - The **build** side is the right input; the optimizer's join ordering
//!   puts the smaller input of an inner join there. A right outer join is
//!   a left outer join with the sides swapped (the layout keeps the column
//!   order right).
//! - Join strategy hints follow Spark's `JoinSelection` (see
//!   [`Planner::strategy`]): `BROADCAST` and `SHUFFLE_HASH` pick the build
//!   side, `SHUFFLE_REPLICATE_NL` makes an inner join a nested loop over
//!   every pair of rows. A hint Spark would honour but a single-node plan
//!   cannot (`MERGE`: there is no sort-merge join; a build side this join
//!   type cannot build) is a warning, as Spark warns of the hints it
//!   ignores.
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
    AttrId, Builtin, Expr, ExprNode, FunctionId, JoinCondition, JoinKind, JoinStrategy,
    LogicalPlan, PlanNode, ResolvedExpr, ResolvedFunction,
};

use super::build::{Planned, Planner, unsupported};
use super::explain::ExplainNode;
use super::orient::{InputStats, StatsSource, orient};
use super::properties::Distribution;
use crate::error::Result;
use crate::join::JoinType;
use crate::physical::{
    HashJoinPlan, JoinDistribution, NestedLoopPlan, PhysNode, Residual, hash_join, nested_loop_join,
};
use crate::values::Producer;

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
        if kind == JoinKind::FullOuter {
            return Err(unsupported("FULL OUTER JOIN"));
        }
        let hints = (self.side_hint(left), self.side_hint(right));
        let logical = (Arc::clone(left), Arc::clone(right));
        let left = self.plan(left)?;
        let right = self.plan(right)?;
        let left_ids: HashSet<AttrId> = left.layout.iter().copied().collect();
        let right_ids: HashSet<AttrId> = right.layout.iter().copied().collect();
        let mut conjuncts = Vec::new();
        if let Some(on) = on {
            split_and(on, &mut conjuncts);
        }
        // Keys as (left side, right side, null-safe).
        let mut keys: Vec<(Arc<Expr>, Arc<Expr>, bool)> = Vec::new();
        let mut residual: Vec<&Expr> = Vec::new();
        let mut null_aware = false;
        if kind == JoinKind::LeftAnti
            && let [only] = conjuncts.as_slice()
            && let Some((l, r)) = null_aware_key(only)
            && let Some((p, b)) = sides(l, r, &left_ids, &right_ids)
        {
            null_aware = true;
            keys.push((p, b, false));
        } else {
            for c in &conjuncts {
                let key = equality(c).and_then(|(l, r, null_safe)| {
                    sides(l, r, &left_ids, &right_ids).map(|(p, b)| (p, b, null_safe))
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
                            && let Some((p, b)) = sides(l, r, &left_ids, &right_ids)
                        {
                            keys.push((p, b, null_safe));
                        }
                    }
                }
            }
        }
        let choice = self.strategy(kind, hints, !keys.is_empty());
        if choice.cartesian {
            // `SHUFFLE_REPLICATE_NL`: every pair of rows, the whole
            // condition as the filter.
            return self.nested_loop(left, right, JoinType::Inner, &conjuncts);
        }
        // The build side: right, else left (a right outer join is a left
        // outer join with the sides swapped).
        let build_left = choice.build_left || kind == JoinKind::RightOuter;
        let join_type = match kind {
            JoinKind::LeftOuter | JoinKind::RightOuter => JoinType::LeftOuter,
            JoinKind::LeftSemi => JoinType::LeftSemi,
            JoinKind::LeftAnti if null_aware => JoinType::NullAwareAnti,
            JoinKind::LeftAnti => JoinType::LeftAnti,
            _ => JoinType::Inner,
        };
        let (probe, build) = if build_left {
            for k in &mut keys {
                std::mem::swap(&mut k.0, &mut k.1);
            }
            (right, left)
        } else {
            (left, right)
        };
        if keys.is_empty() {
            return self.nested_loop(probe, build, join_type, &residual);
        }
        let probe_exprs: Vec<Arc<Expr>> = keys.iter().map(|k| Arc::clone(&k.0)).collect();
        let build_exprs: Vec<Arc<Expr>> = keys.iter().map(|k| Arc::clone(&k.1)).collect();
        let (probe, probe_keys) = self.with_columns(probe, &probe_exprs)?;
        let (build, build_keys) = self.with_columns(build, &build_exprs)?;
        let (probe_plan, build_plan) = if build_left {
            (&logical.1, &logical.0)
        } else {
            (&logical.0, &logical.1)
        };
        let build_rows = self.estimated_rows(build_plan);
        // Which side the statistics say to build (design P2-T166). The
        // joins that build their preserved side do not run yet: the plan
        // keeps the right build and shows the preference.
        let preferred = if choice.distribution.is_none() {
            let probe_stats = self.input_stats(probe_plan, &probe.schema);
            let build_stats = self.input_stats(build_plan, &build.schema);
            let other = orient(join_type, &probe_stats, &build_stats);
            (other != join_type).then_some((other, probe_stats, build_stats))
        } else {
            None
        };
        // A hint decides, else the build's estimated size. The null-aware
        // anti join reads whether the whole build has a NULL key, so its
        // build is always broadcast.
        let distribution = if join_type == JoinType::NullAwareAnti {
            JoinDistribution::Broadcast
        } else {
            choice.distribution.unwrap_or(
                if estimated_bytes(build_rows, &build.schema) <= BROADCAST_MAX_BYTES {
                    JoinDistribution::Broadcast
                } else {
                    JoinDistribution::Partitioned
                },
            )
        };
        let (probe, build) = match distribution {
            JoinDistribution::Broadcast => {
                let build = self.require(build, Distribution::Broadcast)?;
                (probe, build)
            }
            JoinDistribution::Partitioned => {
                self.co_partition(probe, &probe_keys, build, &build_keys)?
            }
        };
        let filter = residual_of(&probe, &build, &residual);
        let null_safe = keys.iter().map(|k| k.2).collect();
        // Inner and left semi joins drop unmatched probe rows: the build
        // publishes runtime filters to the probe side's scans.
        let mut published = Vec::new();
        let mut runtime_filters = Vec::new();
        if matches!(join_type, JoinType::Inner | JoinType::LeftSemi) {
            let key_types: Vec<_> = probe_keys
                .iter()
                .map(|&k| probe.schema.field(k).data_type().clone())
                .collect();
            let rf = self.runtime_filters(&probe, &probe_exprs, &key_types, build_rows);
            published = rf.explain;
            runtime_filters = rf.filters;
        }
        let mut layout = probe.layout.clone();
        let mut sites = probe.sites.clone();
        if matches!(join_type, JoinType::Inner | JoinType::LeftOuter) {
            layout.extend(&build.layout);
        }
        if join_type == JoinType::Inner {
            sites.extend(build.sites.iter().cloned());
        }
        // Rows stay where the probe placed them; an inner join's matched
        // rows are equally placed by the build's keys.
        let placement = match (distribution, join_type) {
            (JoinDistribution::Partitioned, JoinType::Inner) => {
                probe.placement.clone().union(build.placement.clone())
            }
            _ => probe.placement.clone(),
        }
        .restrict(&layout);
        let mut detail = format!(
            "{join_type:?}, [{}] = [{}]",
            text(&probe_exprs),
            text(&build_exprs)
        );
        if !residual.is_empty() {
            let r: Vec<String> = residual.iter().map(ToString::to_string).collect();
            detail.push_str(&format!(", residual: {}", r.join(" AND ")));
        }
        for p in published {
            detail.push_str(&format!(", {p}"));
        }
        if let Some((other, probe_stats, build_stats)) = preferred {
            detail.push_str(&format!(
                ", prefers {other:?} (probe: {}; build: {})",
                probe_stats.describe(),
                build_stats.describe()
            ));
        }
        let (probe_schema, build_schema) = (Arc::clone(&probe.schema), Arc::clone(&build.schema));
        let plan = HashJoinPlan {
            probe: Box::new(probe.node),
            build: Box::new(build.node),
            join_type,
            probe_keys,
            build_keys,
            null_safe,
            residual: filter,
            runtime_filters,
            distribution,
        };
        // Validates the join and gives its schema, over placeholder cells.
        let schema = hash_join(
            &probe_schema,
            &build_schema,
            &plan,
            &self.env,
            None,
            &self.placeholder_values(plan.runtime_filters.iter().map(|f| f.id))?,
            Producer::default(),
        )?
        .schema();
        Ok(Planned {
            schema,
            node: PhysNode::HashJoin(plan),
            layout,
            explain: ExplainNode::new("HashJoin", detail, vec![probe.explain, build.explain]),
            sites,
            placement,
        })
    }
}

/// The conjuncts `residual` over probe then build columns.
fn residual_of(probe: &Planned, build: &Planned, residual: &[&Expr]) -> Option<Residual> {
    if residual.is_empty() {
        return None;
    }
    Some(Residual {
        layout: probe.layout.iter().chain(&build.layout).copied().collect(),
        conjuncts: residual.iter().map(|c| Arc::new((*c).clone())).collect(),
    })
}

impl Planner<'_> {
    /// The statistics of a join input planned as `schema` from `plan`.
    fn input_stats(&self, plan: &Arc<LogicalPlan>, schema: &arrow_schema::Schema) -> InputStats {
        let rows = self.estimated_rows(plan);
        let source = if self.has_statistics(plan) {
            StatsSource::Statistics
        } else {
            StatsSource::Default
        };
        InputStats {
            rows,
            bytes: estimated_bytes(rows, schema),
            source,
        }
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
        // Every probe row meets every build row: the build is broadcast.
        let build = self.require(build, Distribution::Broadcast)?;
        let filter = residual_of(&probe, &build, residual);
        let mut layout = probe.layout.clone();
        let mut sites = probe.sites.clone();
        if matches!(join_type, JoinType::Inner | JoinType::LeftOuter) {
            layout.extend(&build.layout);
        }
        if join_type == JoinType::Inner {
            sites.extend(build.sites.iter().cloned());
        }
        let r: Vec<String> = residual.iter().map(ToString::to_string).collect();
        let detail = if r.is_empty() {
            format!("{join_type:?}")
        } else {
            format!("{join_type:?}, {}", r.join(" AND "))
        };
        let placement = probe.placement.clone().restrict(&layout);
        let (probe_schema, build_schema) = (Arc::clone(&probe.schema), Arc::clone(&build.schema));
        let plan = NestedLoopPlan {
            probe: Box::new(probe.node),
            build: Box::new(build.node),
            join_type,
            residual: filter,
        };
        let schema = nested_loop_join(&probe_schema, &build_schema, &plan, &self.env)?.schema();
        Ok(Planned {
            schema,
            node: PhysNode::NestedLoopJoin(plan),
            layout,
            explain: ExplainNode::new("NestedLoopJoin", detail, vec![probe.explain, build.explain]),
            placement,
            sites,
        })
    }
}

/// What the hints decided for one join.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Choice {
    /// Build the left input (an inner join), else the right.
    pub(crate) build_left: bool,
    /// A nested loop over every pair of rows (`SHUFFLE_REPLICATE_NL`).
    pub(crate) cartesian: bool,
    /// The distribution a hint asks for (`BROADCAST`; `SHUFFLE_HASH` and
    /// `MERGE`, Spark's shuffled joins).
    pub(crate) distribution: Option<JoinDistribution>,
}

/// The largest estimated build a hash join broadcasts without a hint: 10
/// MiB, the default of Spark's `spark.sql.autoBroadcastJoinThreshold`.
pub(crate) const BROADCAST_MAX_BYTES: f64 = 10.0 * 1024.0 * 1024.0;

/// The estimated bytes of `rows` rows of `schema`, after Spark's size
/// estimate: 8 bytes per row plus each column's default size.
pub(crate) fn estimated_bytes(rows: f64, schema: &arrow_schema::Schema) -> f64 {
    let width: u64 = schema
        .fields()
        .iter()
        .map(|f| default_size(f.data_type()))
        .fold(8, u64::saturating_add);
    #[expect(clippy::cast_precision_loss, reason = "an estimate")]
    let width = width as f64;
    rows * width
}

/// Spark's `DataType.defaultSize` (Spark 4.0 source) for the Arrow
/// carriers of Spark's types.
fn default_size(t: &arrow_schema::DataType) -> u64 {
    use arrow_schema::DataType as T;
    match t {
        T::Null | T::Boolean | T::Int8 => 1,
        T::Int16 => 2,
        T::Int32 | T::Float32 | T::Date32 => 4,
        T::Decimal128(p, _) | T::Decimal256(p, _) if *p > 18 => 16,
        T::Utf8 | T::LargeUtf8 | T::Utf8View => 20,
        T::Binary | T::LargeBinary | T::BinaryView => 100,
        T::List(f) | T::LargeList(f) | T::Map(f, _) => default_size(f.data_type()),
        T::Struct(fs) => fs
            .iter()
            .map(|f| default_size(f.data_type()))
            .fold(0, u64::saturating_add),
        T::Dictionary(_, v) => default_size(v),
        // Int64, Float64, timestamps, decimals of up to 18 digits.
        _ => 8,
    }
}

/// Spark's `JoinType.sql`.
fn kind_sql(kind: JoinKind) -> &'static str {
    match kind {
        JoinKind::LeftOuter => "LEFT OUTER",
        JoinKind::RightOuter => "RIGHT OUTER",
        JoinKind::FullOuter => "FULL OUTER",
        JoinKind::LeftSemi => "LEFT SEMI",
        JoinKind::LeftAnti => "LEFT ANTI",
        _ => "INNER",
    }
}

/// The sides this planner can build: the right of every join it runs, the
/// left of an inner or right outer join.
fn can_build(left: bool, kind: JoinKind) -> bool {
    if left {
        matches!(
            kind,
            JoinKind::Inner | JoinKind::Cross | JoinKind::RightOuter
        )
    } else {
        !matches!(kind, JoinKind::RightOuter | JoinKind::FullOuter)
    }
}

/// The sides Spark's shuffled hash join builds.
fn spark_shuffle_hash(left: bool, kind: JoinKind) -> bool {
    !left || !matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti)
}

impl Planner<'_> {
    /// The strategy hint of one join side: the first hint above its
    /// relation through single-input nodes (Spark's `EliminateResolvedHint`),
    /// a later different one overridden. The hints seen are consumed.
    fn side_hint(&mut self, p: &Arc<LogicalPlan>) -> Option<JoinStrategy> {
        let mut found: Option<JoinStrategy> = None;
        let mut cur = p;
        loop {
            match &cur.node {
                PlanNode::JoinHint { input, strategy } => {
                    self.consumed_hints.insert(Arc::as_ptr(cur) as usize);
                    match found {
                        None => found = Some(*strategy),
                        Some(f) if f != *strategy => self.warnings.push(format!(
                            "Hint (strategy={}) is overridden by another hint and will not take effect.",
                            strategy.name()
                        )),
                        Some(_) => {}
                    }
                    cur = input;
                }
                PlanNode::Project { input, .. }
                | PlanNode::Filter { input, .. }
                | PlanNode::Aggregate { input, .. }
                | PlanNode::Sort { input, .. }
                | PlanNode::Limit { input, .. }
                | PlanNode::Offset { input, .. }
                | PlanNode::SubqueryAlias { input, .. }
                | PlanNode::Shift { input, .. } => cur = input,
                _ => return found,
            }
        }
    }

    /// Spark's `JoinSelection` over the hints of the two sides, in its
    /// order: `BROADCAST`, `MERGE`, `SHUFFLE_HASH`, `SHUFFLE_REPLICATE_NL`.
    pub(crate) fn strategy(
        &mut self,
        kind: JoinKind,
        (left, right): (Option<JoinStrategy>, Option<JoinStrategy>),
        equi: bool,
    ) -> Choice {
        let mut choice = Choice::default();
        let hinted = |s: JoinStrategy| (left == Some(s), right == Some(s));
        let warn = |this: &mut Self, s: JoinStrategy, why: String| {
            this.warnings.push(format!(
                "Hint (strategy={}) is not supported in the query: {why}",
                s.name()
            ));
        };
        let build = |side_left: bool| if side_left { "left" } else { "right" };
        // BROADCAST: Spark builds what this planner can (a nested loop
        // builds either side in Spark).
        let (bl, br) = hinted(JoinStrategy::Broadcast);
        if bl || br {
            let spark = |l: bool| !equi || can_build(l, kind);
            let (sl, sr) = (bl && spark(true), br && spark(false));
            if sl || sr {
                let want_left = sl && !sr;
                if can_build(want_left, kind) {
                    choice.build_left = want_left;
                    choice.distribution = Some(JoinDistribution::Broadcast);
                } else {
                    let why = format!(
                        "build {} for {} join (a single-node nested-loop join cannot).",
                        build(want_left),
                        kind_sql(kind)
                    );
                    warn(self, JoinStrategy::Broadcast, why);
                }
                return choice;
            }
            for (on, l) in [(bl, true), (br, false)] {
                if on {
                    let why = format!("build {} for {} join.", build(l), kind_sql(kind));
                    warn(self, JoinStrategy::Broadcast, why);
                }
            }
        }
        if !equi {
            // Already a nested loop; Spark ignores the other hints here.
            return choice;
        }
        let (ml, mr) = hinted(JoinStrategy::ShuffleMerge);
        if ml || mr {
            warn(
                self,
                JoinStrategy::ShuffleMerge,
                "no sort-merge join in a single-node plan; a hash join runs instead.".to_owned(),
            );
            choice.distribution = Some(JoinDistribution::Partitioned);
            return choice;
        }
        let (hl, hr) = hinted(JoinStrategy::ShuffleHash);
        if hl || hr {
            let (sl, sr) = (
                hl && spark_shuffle_hash(true, kind),
                hr && spark_shuffle_hash(false, kind),
            );
            if sl || sr {
                let want_left = sl && !sr;
                if can_build(want_left, kind) {
                    choice.build_left = want_left;
                    choice.distribution = Some(JoinDistribution::Partitioned);
                } else {
                    let why = format!(
                        "build {} for {} join (a single-node hash join cannot).",
                        build(want_left),
                        kind_sql(kind)
                    );
                    warn(self, JoinStrategy::ShuffleHash, why);
                }
                return choice;
            }
            for (on, l) in [(hl, true), (hr, false)] {
                if on {
                    let why = format!("build {} for {} join.", build(l), kind_sql(kind));
                    warn(self, JoinStrategy::ShuffleHash, why);
                }
            }
        }
        let (nl, nr) = hinted(JoinStrategy::ShuffleReplicateNl);
        choice.cartesian = (nl || nr) && matches!(kind, JoinKind::Inner | JoinKind::Cross);
        choice
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType as T, Field, Fields, Schema};

    use super::{default_size, estimated_bytes};

    #[test]
    fn sizes_follow_sparks_default_sizes() {
        let item = |t: T| Arc::new(Field::new("item", t, true));
        for (t, size) in [
            (T::Null, 1),
            (T::Boolean, 1),
            (T::Int8, 1),
            (T::Int16, 2),
            (T::Int32, 4),
            (T::Float32, 4),
            (T::Date32, 4),
            (T::Int64, 8),
            (T::Float64, 8),
            (T::Decimal128(18, 2), 8),
            (T::Decimal128(19, 2), 16),
            (T::Utf8View, 20),
            (T::LargeUtf8, 20),
            (T::Binary, 100),
            (T::List(item(T::Int32)), 4),
            (
                T::Map(
                    item(T::Struct(Fields::from(vec![
                        Field::new("key", T::Int32, false),
                        Field::new("value", T::Utf8, true),
                    ]))),
                    false,
                ),
                24,
            ),
            (
                T::Struct(Fields::from(vec![
                    Field::new("a", T::Int16, true),
                    Field::new("b", T::Utf8, true),
                ])),
                22,
            ),
            (T::Dictionary(Box::new(T::Int32), Box::new(T::Utf8)), 20),
        ] {
            assert_eq!(default_size(&t), size, "{t}");
        }
        let schema = Schema::new(vec![
            Field::new("k", T::Int64, false),
            Field::new("s", T::Utf8, true),
        ]);
        // 8 per row, plus 8 and 20.
        assert!((estimated_bytes(1000.0, &schema) - 36_000.0).abs() < f64::EPSILON);
    }
}
