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

use arrow_schema::DataType as ArrowType;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, CastMode, Expr, ExprNode, FunctionId, JoinCondition, JoinKind,
    JoinStrategy, LogicalPlan, PlanNode, ResolvedExpr, ResolvedFunction,
};

use super::build::{Planned, Planner, internal, unsupported};
use super::eval::canonical_carrier;
use super::explain::ExplainNode;
use super::orient::{InputStats, StatsSource, builds_left, orient};
use super::properties::{Distribution, Placement};
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
            return self.nested_loop(left, right, JoinType::Inner, kind, &conjuncts);
        }
        // The build side: right, else left (a right outer join is a left
        // outer join with the sides swapped).
        let build_left =
            (choice.build_left && kind != JoinKind::Single) || kind == JoinKind::RightOuter;
        let join_type = match kind {
            JoinKind::LeftOuter | JoinKind::RightOuter | JoinKind::Single => JoinType::LeftOuter,
            JoinKind::FullOuter => JoinType::FullOuter,
            JoinKind::LeftSemi => JoinType::LeftSemi,
            JoinKind::LeftAnti if null_aware => JoinType::NullAwareAnti,
            JoinKind::LeftAnti => JoinType::LeftAnti,
            JoinKind::Existence(_) => JoinType::Existence,
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
            return self.nested_loop(probe, build, join_type, kind, &residual);
        }
        let probe_exprs: Vec<Arc<Expr>> = keys.iter().map(|k| Arc::clone(&k.0)).collect();
        let build_exprs: Vec<Arc<Expr>> = keys.iter().map(|k| Arc::clone(&k.1)).collect();
        let (probe, probe_keys) = self.with_columns(probe, &probe_exprs)?;
        let (build, build_keys) = self.with_columns(build, &build_exprs)?;
        // Keys of one Spark `STRING`/`BINARY` in different Arrow encodings
        // (a `Utf8` column joined to a `Utf8View` one) meet in the canonical
        // carrier: the executor rejects mixed key types, and its row
        // encoder needs one type per key. Keys of different Spark types are
        // left for the join builder to reject as before.
        let (probe, probe_keys, build, build_keys) =
            self.unify_join_keys(probe, probe_keys, build, build_keys)?;
        let (probe_plan, build_plan) = if build_left {
            (&logical.1, &logical.0)
        } else {
            (&logical.0, &logical.1)
        };
        let build_rows = self.estimated_rows(build_plan);
        // A `Single` join is a scalar subquery's single-match join;
        // orientation refuses to flip one.
        let single_match = kind == JoinKind::Single;
        // Which side the statistics say to build (design P2-T166): a semi,
        // anti or left outer join whose right input is much larger builds
        // its left input instead, as a right semi, right anti or right
        // outer join. A hint keeps Spark's side. The planner plans batch
        // queries only (unweighted rows): the right joins refuse weighted
        // inputs, so an incremental (IVM) planning mode must not flip.
        let build_side = self
            .cx
            .config
            .get(&krishiv_common::config::EXEC_JOIN_BUILD_SIDE)
            .and_then(krishiv_common::config::ConfigValue::as_bool)
            .unwrap_or(true);
        let preferred = if choice.distribution.is_none() && build_side {
            let probe_stats = self.input_stats(probe_plan, &probe.schema);
            let build_stats = self.input_stats(build_plan, &build.schema);
            let other = orient(join_type, single_match, &probe_stats, &build_stats);
            let flips = other != join_type
                || (join_type == JoinType::FullOuter && builds_left(&probe_stats, &build_stats));
            flips.then_some((other, probe_stats, build_stats))
        } else {
            None
        };
        let (probe, probe_keys, probe_exprs, build, build_keys, build_exprs, build_rows, join_type) =
            match &preferred {
                Some((other, ..)) => (
                    build,
                    build_keys,
                    build_exprs,
                    probe,
                    probe_keys,
                    probe_exprs,
                    self.estimated_rows(probe_plan),
                    *other,
                ),
                None => (
                    probe,
                    probe_keys,
                    probe_exprs,
                    build,
                    build_keys,
                    build_exprs,
                    build_rows,
                    join_type,
                ),
            };
        // A hint decides, else the build's estimated size. The null-aware
        // anti join reads whether the whole build has a NULL key, so its
        // build is always broadcast. An existence join reads only the build
        // rows with the probe row's key (its `exists` is never NULL), so,
        // like the other equi-joins, it is broadcast or partitioned by size.
        let distribution = if join_type == JoinType::NullAwareAnti {
            JoinDistribution::Broadcast
        } else if join_type.builds_preserved() {
            // A build whose rows the join returns is never broadcast: each
            // copy would return them (design P2-T174 decision 3).
            JoinDistribution::Partitioned
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
        // Inner, left semi and the right joins drop unmatched probe rows
        // (a right join returns build rows by their matches, and a probe
        // row that matches nothing marks nothing): the build publishes
        // runtime filters to the probe side's scans.
        let mut published = Vec::new();
        let mut runtime_filters = Vec::new();
        if join_type.drops_unmatched_probe() {
            let key_types: Vec<_> = probe_keys
                .iter()
                .map(|&k| probe.schema.field(k).data_type().clone())
                .collect();
            let rf =
                self.runtime_filters(&probe, &probe_exprs, &key_types, &build.node, build_rows);
            published = rf.explain;
            runtime_filters = rf.filters;
        }
        let mut layout = probe.layout.clone();
        let mut sites = probe.sites.clone();
        if matches!(
            join_type,
            JoinType::Inner | JoinType::LeftOuter | JoinType::RightOuter | JoinType::FullOuter
        ) {
            layout.extend(&build.layout);
        }
        if matches!(join_type, JoinType::RightSemi | JoinType::RightAnti) {
            layout.clone_from(&build.layout);
        }
        if let JoinKind::Existence(id) = kind {
            layout.push(id);
        }
        if join_type == JoinType::Inner {
            sites.extend(build.sites.iter().cloned());
        }
        if join_type.builds_preserved() {
            // The rows the join returns are build rows (a right outer
            // join's probe columns may be NULL).
            sites.clone_from(&build.sites);
        }
        if join_type == JoinType::FullOuter {
            // Neither side may be filtered at its scan.
            sites.clear();
        }
        // Rows stay where the probe placed them (the build for the joins
        // that return build rows); an inner join's matched rows are
        // equally placed by the build's keys.
        let placement = match (distribution, join_type) {
            (JoinDistribution::Partitioned, JoinType::Inner) => {
                probe.placement.clone().union(build.placement.clone())
            }
            // Either side's key may be NULL in a row of the other: nothing
            // is placed by a key.
            (_, JoinType::FullOuter) => Placement::Unknown,
            (_, t) if t.builds_preserved() => build.placement.clone(),
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
        if let Some((_, probe_stats, build_stats)) = preferred {
            // The statistics that chose to build the left input, labelled
            // by the logical inputs: `probe_stats` is the left input's and
            // `build_stats` the right one's (taken before the swap).
            detail.push_str(&format!(
                ", builds left (left: {}; right: {})",
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
            single_match,
            distinct_build: false,
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
        kind: JoinKind,
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
        if let JoinKind::Existence(id) = kind {
            layout.push(id);
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

    /// Each key pair's two columns in the canonical carrier of their Spark
    /// type where the sides store one Spark `STRING`/`BINARY` in different
    /// Arrow encodings: the side already there is kept, the other is
    /// projected through an ANSI cast. Pairs of equal or of different Spark
    /// types are untouched (the join builder rejects the latter as before).
    fn unify_join_keys(
        &mut self,
        mut probe: Planned,
        mut probe_keys: Vec<usize>,
        mut build: Planned,
        mut build_keys: Vec<usize>,
    ) -> Result<(Planned, Vec<usize>, Planned, Vec<usize>)> {
        for i in 0..probe_keys.len() {
            let pt = probe
                .schema
                .fields()
                .get(probe_keys[i])
                .map(|f| f.data_type().clone())
                .ok_or_else(|| internal("join key out of range"))?;
            let bt = build
                .schema
                .fields()
                .get(build_keys[i])
                .map(|f| f.data_type().clone())
                .ok_or_else(|| internal("join key out of range"))?;
            if pt == bt {
                continue;
            }
            let (Some(a), Some(b)) = (canonical_carrier(&pt), canonical_carrier(&bt)) else {
                continue;
            };
            if a != b {
                continue;
            }
            if pt != a {
                let (p, k) = self.recast_key(probe, probe_keys[i], &a)?;
                probe = p;
                probe_keys[i] = k;
            }
            if bt != a {
                let (b, k) = self.recast_key(build, build_keys[i], &a)?;
                build = b;
                build_keys[i] = k;
            }
        }
        Ok((probe, probe_keys, build, build_keys))
    }

    /// `input` with key column `key` projected through an ANSI cast to the
    /// canonical carrier `to` of its Spark type, and the new column. The
    /// cast is over the key column itself, so runtime-filter sites still
    /// resolve through it to the scan's column.
    fn recast_key(
        &mut self,
        input: Planned,
        key: usize,
        to: &ArrowType,
    ) -> Result<(Planned, usize)> {
        let field = input
            .schema
            .fields()
            .get(key)
            .ok_or_else(|| internal("join key out of range"))?;
        let stored = field.data_type().clone();
        let spark = krishiv_types::DataType::from_arrow(&stored)
            .map_err(|e| internal(&format!("a join key of unexpected type: {e}")))?;
        let target = krishiv_types::DataType::from_arrow(to)
            .map_err(|e| internal(&format!("a join key of unexpected type: {e}")))?;
        let attr = input
            .layout
            .get(key)
            .copied()
            .ok_or_else(|| internal("join key out of range"))?;
        let column = Expr::new(ExprNode::Resolved(ResolvedExpr::Column(BoundColumn {
            id: attr,
            name: field.name().clone(),
            data_type: spark,
            nullable: field.is_nullable(),
        })));
        let cast = Arc::new(Expr::new(ExprNode::Cast {
            expr: Arc::new(column),
            to: target,
            mode: CastMode::Ansi,
        }));
        let (planned, columns) = self.with_columns(input, &[cast])?;
        let key = columns
            .first()
            .copied()
            .ok_or_else(|| internal("key arity"))?;
        Ok((planned, key))
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
        JoinKind::Single => "SINGLE",
        JoinKind::RightOuter => "RIGHT OUTER",
        JoinKind::FullOuter => "FULL OUTER",
        JoinKind::LeftSemi => "LEFT SEMI",
        JoinKind::LeftAnti => "LEFT ANTI",
        JoinKind::Existence(_) => "EXISTENCE",
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
    !left
        || !matches!(
            kind,
            JoinKind::LeftSemi | JoinKind::LeftAnti | JoinKind::Existence(_)
        )
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
