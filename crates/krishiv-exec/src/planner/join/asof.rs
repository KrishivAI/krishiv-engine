// SPDX-License-Identifier: Apache-2.0
//! As-of joins (`PlanNode::AsOfJoin`, Spark 4.3) onto the sort-merge join's as-of merge
//! (`join/merge/asof.rs`).
//!
//! The condition's equalities between the two sides are the keys (none: one constant key, so
//! every row is in one group, Spark's `AllTuples`); the rest is the residual, which filters the
//! right rows before the nearest one is taken. Each input computes its as-of value as its last
//! column, and both are hash-partitioned on the keys (gathered to one partition when there are
//! none) and sorted on keys then value. Runtime-filter sites stay on the left side only: a filter
//! that dropped right rows would change which row is nearest; and none stays where the residual or
//! an evaluated distance may raise.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_plan::{
    AsOfDirection, AsOfMatch, AttrId, Expr, ExprNode, JoinCondition, JoinKind, Literal,
    LogicalPlan, ResolvedExpr,
};
use krishiv_types::DataType;

use super::super::build::{Planned, Planner, unsupported};
use super::super::explain::ExplainNode;
use super::super::properties::Distribution;
use super::{equality, residual_of, sides, split_and, text};
use crate::error::{Result, internal};
use crate::join::{AsOfMergeDirection, AsOfMergeSpec, AsOfSortColumns, JoinType};
use crate::physical::{PhysNode, SortMergeJoinPlan, sort_merge_join};

impl Planner<'_> {
    pub(crate) fn as_of_join(
        &mut self,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: JoinKind,
        condition: &JoinCondition,
        as_of: &AsOfMatch,
        demand: Option<HashSet<AttrId>>,
    ) -> Result<Planned> {
        let on = match condition {
            JoinCondition::On(e) => Some(e),
            JoinCondition::None => None,
            _ => {
                // The analyzer expands `USING` into `ON`.
                return Err(internal(
                    "an as-of join's USING reached the physical planner unexpanded".to_owned(),
                ));
            }
        };
        let direction = match as_of.direction {
            AsOfDirection::Backward => AsOfMergeDirection::Backward,
            AsOfDirection::Forward => AsOfMergeDirection::Forward,
            _ => return Err(unsupported("an as-of join in the nearest direction")),
        };
        if as_of.tolerance.is_some() {
            return Err(unsupported("an as-of join with a tolerance"));
        }
        let join_type = match kind {
            JoinKind::Inner => JoinType::Inner,
            JoinKind::LeftOuter => JoinType::LeftOuter,
            _ => return Err(unsupported("an as-of join other than inner or left outer")),
        };
        // Each side is asked for what the parent reads plus what the condition and the values read.
        let mut demand = demand;
        let (left_sort, right_sort) = as_of.sort_operands();
        for e in on
            .into_iter()
            .chain([&as_of.left, &as_of.right, left_sort, right_sort])
        {
            demand = super::super::build::with_columns_of(demand, e);
        }
        self.demand.clone_from(&demand);
        let probe = self.plan(left)?;
        self.demand = demand;
        let build = self.plan(right)?;
        let left_ids: HashSet<AttrId> = probe.layout.iter().copied().collect();
        let right_ids: HashSet<AttrId> = build.layout.iter().copied().collect();
        let mut conjuncts = Vec::new();
        if let Some(on) = on {
            split_and(on, &mut conjuncts);
        }
        let mut keys: Vec<(Arc<Expr>, Arc<Expr>, bool)> = Vec::new();
        let mut residual: Vec<&Expr> = Vec::new();
        for c in &conjuncts {
            let key = equality(c).and_then(|(l, r, null_safe)| {
                sides(l, r, &left_ids, &right_ids).map(|(p, b)| (p, b, null_safe))
            });
            match key {
                Some(k) => keys.push(k),
                None => residual.push(c),
            }
        }
        let constant = keys.is_empty();
        if constant {
            let zero = Arc::new(Expr::literal(Literal::Integer(0)));
            keys.push((Arc::clone(&zero), zero, false));
        }
        let probe_exprs: Vec<Arc<Expr>> = keys.iter().map(|k| Arc::clone(&k.0)).collect();
        let build_exprs: Vec<Arc<Expr>> = keys.iter().map(|k| Arc::clone(&k.1)).collect();
        let (probe, probe_keys) = self.with_columns(probe, &probe_exprs)?;
        let (build, build_keys) = self.with_columns(build, &build_exprs)?;
        let (probe, probe_keys, build, build_keys) =
            self.unify_join_keys(probe, probe_keys, build, build_keys)?;
        let (probe, probe_value) = self.with_columns(probe, std::slice::from_ref(&as_of.left))?;
        let (build, build_value) = self.with_columns(build, std::slice::from_ref(&as_of.right))?;
        // The values meet in one carrier as the keys do: the merge compares their sort rows.
        let (probe, probe_value, build, build_value) =
            self.unify_join_keys(probe, probe_value, build, build_value)?;
        let (Some(&left_column), Some(&right_column)) = (probe_value.first(), build_value.first())
        else {
            return Err(unsupported("an as-of join without its values"));
        };
        // Spark sorts each side by its operand before the comparison's coercion
        // (`matchSortExpressions`); the merge compares the coerced values. When both coercions keep
        // the order, the coerced values are in sort order and the merge walks a group once;
        // otherwise each left row reads its group from the first row, as Spark's scan does.
        let (probe, build, sort) = match &as_of.sort {
            None => (probe, build, None),
            Some((l, r)) => {
                let (probe, left) = self.with_columns(probe, std::slice::from_ref(l))?;
                let (build, right) = self.with_columns(build, std::slice::from_ref(r))?;
                let (Some(&left), Some(&right)) = (left.first(), right.first()) else {
                    return Err(unsupported("an as-of join without its sort operands"));
                };
                let fixed_zone = matches!(
                    self.env.zone_rules()?,
                    krishiv_kernels::datetime::ZoneRef::Fixed(_)
                );
                let keeps = |p: &Planned, from: usize, to: usize| {
                    keeps_order(
                        p.schema.field(from).data_type(),
                        p.schema.field(to).data_type(),
                        fixed_zone,
                    )
                };
                let monotone = keeps(&probe, left, left_column) && keeps(&build, right, right_column);
                (probe, build, Some(AsOfSortColumns { left, right, monotone }))
            }
        };
        if sort.is_some_and(|s| !s.monotone) {
            // Spark's forward scan picks by the distance, which unit 2 of the design builds for
            // every type; the Connect form has no sort operands.
            if direction == AsOfMergeDirection::Forward || as_of.distance_on_every_pair {
                return Err(unsupported(
                    "a forward as-of join whose comparison casts a value out of its order (as STRING against a number)",
                ));
            }
        }
        // Without keys every row meets in one group, so both sides gather to one partition
        // (I8): a hash exchange on the constant key would do the same with a hidden hot spot.
        let (probe, build) = if constant {
            (
                self.require(probe, Distribution::Single)?,
                self.require(build, Distribution::Single)?,
            )
        } else {
            self.co_partition(probe, &probe_keys, build, &build_keys)?
        };
        let filter = residual_of(&probe, &build, &residual);
        // The Connect form picks the row of the smallest distance (Spark's `min_by`); the SQL
        // scan forward compares the distances of the match and the next passing row. Without
        // ANSI an integral distance wraps and a decimal one that is not exact is NULL on
        // overflow, so their order need not be the as-of values' order, which the merge walks:
        // refuse rather than pick another row. Backward, the SQL scan evaluates no distance.
        // Timestamp, date and interval distances raise in Spark with ANSI off too (the distance
        // below plans that) and an exact decimal cannot overflow, so those run.
        if !self.env.ansi
            && (as_of.distance_on_every_pair || direction == AsOfMergeDirection::Forward)
            && as_of.distance.as_deref().is_some_and(distance_can_wrap)
        {
            return Err(unsupported(if as_of.distance_on_every_pair {
                "a Connect as-of join whose distance can overflow without ANSI mode"
            } else {
                "a forward as-of join whose distance can overflow without ANSI mode"
            }));
        }
        // A floating-point distance ranks unlike the values at the infinities (`-Infinity -
        // -Infinity` is NaN, which Spark orders above every number), so `min_by`, and the SQL
        // scan forward, can pick another row than the nearest value: refuse those forms on
        // floating-point values, ANSI or not (a double subtraction never raises).
        if (as_of.distance_on_every_pair || direction == AsOfMergeDirection::Forward)
            && as_of.distance.as_deref().is_some_and(distance_is_floating)
        {
            return Err(unsupported(if as_of.distance_on_every_pair {
                "a Connect as-of join on floating-point values"
            } else {
                "a forward as-of join on floating-point values"
            }));
        }
        // A `TIMESTAMP` distance subtracts local date-times in the session zone (Spark's
        // `subtractTimestamps`), so across a DST fallback a later right row can be nearer: Spark's
        // forward scan and `min_by` take it, the merge takes the first by value. Refuse those
        // forms where the session zone's local time can run backwards.
        if (as_of.distance_on_every_pair || direction == AsOfMergeDirection::Forward)
            && matches!(
                probe.schema.field(left_column).data_type(),
                arrow_schema::DataType::Timestamp(_, Some(_))
            )
            && !self.env.zone_rules()?.local_time_is_monotonic()
        {
            return Err(unsupported(if as_of.distance_on_every_pair {
                "a Connect as-of join on TIMESTAMP values in a session time zone whose UTC offset ever decreases (a fixed offset such as UTC, or TIMESTAMP_NTZ values, run)"
            } else {
                "a forward as-of join on TIMESTAMP values in a session time zone whose UTC offset ever decreases (a backward join, a fixed offset such as UTC, or TIMESTAMP_NTZ values, run)"
            }));
        }
        // Only a distance that can raise is evaluated: its value is never used.
        let distance = as_of
            .distance
            .as_deref()
            .filter(|d| krishiv_plan::may_raise(d, self.env.ansi))
            .and_then(|d| residual_of(&probe, &build, &[d]));
        let null_safe = keys.iter().map(|k| k.2).collect();
        let mut layout = probe.layout.clone();
        layout.extend(&build.layout);
        // A runtime filter a join above pushes into the left scan would drop rows before the
        // merge evaluates the residual or the distance on them: where either may raise, the
        // filter must not hide the row that raises (as `no_runtime_filters_before`).
        let raises = residual
            .iter()
            .copied()
            .chain(as_of.distance.as_deref().filter(|_| {
                // Spark's SQL scan evaluates no distance backward.
                as_of.distance_on_every_pair || direction == AsOfMergeDirection::Forward
            }))
            .any(|e| krishiv_plan::may_raise(e, self.env.ansi));
        let sites = if raises {
            Vec::new()
        } else {
            probe.sites.clone()
        };
        let placement = probe.placement.clone().restrict(&layout);
        let op = as_of
            .direction
            .sql_operator(as_of.allow_exact)
            .unwrap_or("nearest");
        let mut detail = format!("{join_type:?}, {} {op} {}", as_of.left, as_of.right);
        if !constant {
            detail.push_str(&format!(
                ", [{}] = [{}]",
                text(&probe_exprs),
                text(&build_exprs)
            ));
        }
        if !residual.is_empty() {
            let r: Vec<String> = residual.iter().map(ToString::to_string).collect();
            detail.push_str(&format!(", residual: {}", r.join(" AND ")));
        }
        let (probe_schema, build_schema) = (Arc::clone(&probe.schema), Arc::clone(&build.schema));
        let plan = SortMergeJoinPlan {
            probe: Box::new(probe.node),
            build: Box::new(build.node),
            join_type,
            probe_keys,
            build_keys,
            null_safe,
            residual: filter,
            single_match: false,
            as_of: Some(AsOfMergeSpec {
                left_column,
                right_column,
                direction,
                allow_exact: as_of.allow_exact,
                distance,
                distance_on_every_pair: as_of.distance_on_every_pair,
                sort,
            }),
        };
        // Validates the join and gives its schema.
        let schema =
            sort_merge_join(&probe_schema, &build_schema, &plan, &self.env, None)?.schema();
        let sorted = |input, keys: &[Arc<Expr>], value: &Arc<Expr>| {
            ExplainNode::new(
                "Sort",
                format!(
                    "[{}{}{value}]",
                    text(keys),
                    if constant { "" } else { ", " }
                ),
                vec![input],
            )
        };
        let (probe_keys_shown, build_keys_shown) = if constant {
            (Vec::new(), Vec::new())
        } else {
            (probe_exprs, build_exprs)
        };
        let inputs = vec![
            sorted(probe.explain, &probe_keys_shown, left_sort),
            sorted(build.explain, &build_keys_shown, right_sort),
        ];
        Ok(Planned {
            ordering: Vec::new(),
            schema,
            node: PhysNode::SortMergeJoin(plan),
            layout,
            explain: ExplainNode::new("SortMergeAsOfJoin", detail, inputs),
            sites,
            placement,
        })
    }
}

/// Whether an as-of distance can wrap or turn NULL without ANSI mode: an integral one, or a
/// decimal one [`krishiv_plan::exact_decimal`] does not prove exact.
fn distance_can_wrap(d: &Expr) -> bool {
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &d.node else {
        return false;
    };
    match f.data_type {
        DataType::Byte | DataType::Short | DataType::Integer | DataType::Long => true,
        DataType::Decimal(_) => !krishiv_plan::exact_decimal(d),
        _ => false,
    }
}

/// Whether an as-of distance is floating-point.
fn distance_is_floating(d: &Expr) -> bool {
    matches!(
        &d.node,
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if matches!(f.data_type, DataType::Float | DataType::Double)
    )
}

/// Whether a cast from `from` to `to` keeps the order of values (Spark's widening coercions):
/// the same type, an integral to a wider integral, a decimal or a double, a decimal to a decimal
/// or a double, a float to a double, a date to a timestamp, and a `TIMESTAMP_NTZ` to a
/// `TIMESTAMP` in a fixed-offset session zone (a region's DST gap moves local times forward). A cast
/// may map two values to one (a large `BIGINT` to `DOUBLE`) and still keep the order: ties sort as
/// Spark sorts them, by the value before the cast.
fn keeps_order(from: &arrow_schema::DataType, to: &arrow_schema::DataType, fixed_zone: bool) -> bool {
    use arrow_schema::DataType as A;
    let int_width = |t: &A| match t {
        A::Int8 => Some(1),
        A::Int16 => Some(2),
        A::Int32 => Some(4),
        A::Int64 => Some(8),
        _ => None,
    };
    let decimal = |t: &A| matches!(t, A::Decimal32(..) | A::Decimal64(..) | A::Decimal128(..) | A::Decimal256(..));
    if from == to {
        return true;
    }
    match (from, to) {
        (f, t) if int_width(f).is_some() => {
            int_width(t).is_some_and(|w| int_width(f).is_some_and(|v| v <= w))
                || decimal(t)
                || matches!(t, A::Float64)
        }
        (f, t) if decimal(f) => decimal(t) || matches!(t, A::Float64),
        (A::Float32, A::Float64) | (A::Date32, A::Timestamp(_, _)) => true,
        (A::Timestamp(a, None), A::Timestamp(b, Some(_))) => a == b && fixed_zone,
        _ => false,
    }
}
