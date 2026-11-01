// SPDX-License-Identifier: Apache-2.0
//! Eager aggregation: an aggregate's calls run below a join, and their states
//! merge above it (design P2-T167 unit 2; DuckDB #22572, Yan and Larson).
//!
//! `Aggregate(Expand?(Project?(S JOIN T)))` becomes the same aggregate merging
//! the states of `PartialAggregate(S)`, keyed by the columns of `S` that
//! anything above still reads. Each joined row merges its `S` group's state
//! once, which is what aggregating the duplicated rows gives, so `count`,
//! `sum`, `avg`, `min` and `max` need no duplication factor, and the same
//! accumulators run, so result types and rounding do not change.
//!
//! # When it applies
//!
//! Calls ([`refusal_of_call`]). Every aggregate call must be one that exports
//! states, and:
//!
//! 1. Its arguments read only `S`, are deterministic, hold no subquery and
//!    cannot raise under ANSI: the join may drop a row Spark never evaluated
//!    the argument on.
//! 2. It cannot be affected by summation order. Spark checks overflow in
//!    processing order, so a sum runs here only where no buffer can overflow:
//!    `sum` of at most 32-bit integers into `bigint` (fewer than 2^32 rows per
//!    group), and `sum` and `avg` of decimals of precision up to 28 (the buffer
//!    has 10 more digits, so fewer than 10^10 rows per group). `avg` of integers
//!    sums in a double, which is exact while every partial sum stays below 2^53;
//!    the column's range and `S`'s rows must prove that with 8× to spare.
//!    Floating-point `sum`, `avg`, `min` and `max` never run here.
//! 3. On the null-supplying side of an outer join an unmatched row must get
//!    what the empty state gives: only arguments that are a column under casts
//!    are NULL there. `count(*)` and `count(coalesce(x, 1))` count such a row,
//!    and stay above the join.
//!
//! The plan: inner or outer equi-join, with only column-preserving projections
//! and one `Expand` carrying the raw arguments between it and the aggregate.
//! Where the top join refuses and two inner joins allow it, the lower join's
//! dimension moves to the top ([`Shape::rotated`]): q22's `item`, whose join key
//! is the only key its aggregate needs, whichever order the near-tie gave.
//!
//! Gates ([`Gates`]), read from statistics only:
//!
//! - the join keeps at least [`MIN_RETENTION`] of `S`'s rows;
//! - the groups, the product of the keys' distinct counts, are at most
//!   `1 / `[`MIN_REDUCTION`] of `S`'s rows;
//! - at most one key beyond the join's own;
//! - the other side is a table read whole. A filtered dimension prunes the
//!   aggregated side through a runtime filter, which pays more than grouping
//!   and which a smaller aggregate could take away by becoming the join's build
//!   side; estimates cannot see this, as they read a key's matches from its
//!   distinct counts (TPC-DS q12, q43 and q63 ran 1.2–1.4x slower);
//! - every key's distinct count is known (never [`StatSource::Guess`]): Umbra
//!   measured this rewrite over 10× slower on TPC-H Q17 when it fires blind.
//!
//! The upper aggregate stays even when its keys are unique: states are local
//! to each task, so equal groups of different tasks still have to merge.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionKind, Grouping, JoinCondition, JoinKind,
    Literal, LogicalPlan, PlanNode, ResolvedExpr, ResolvedFunction, may_raise,
    partial_state_columns,
};
use krishiv_types::DataType;

use crate::dag;
use crate::estimate::Estimator;
use crate::expr::{
    Attrs, builtin, column, conjuncts, data_type, has_outer_reference, has_subquery,
    is_deterministic, lit, output_of, references, references_into, same,
};
use crate::restrict::{MAX_ID, max_id};
use crate::stats::{RelStats, Statistics};

/// The least fraction of `S`'s rows the join must keep.
pub(crate) const MIN_RETENTION: f64 = 0.7;

/// The least ratio of `S`'s rows to its groups.
pub(crate) const MIN_REDUCTION: f64 = 4.0;

/// The most keys beyond the join's own.
pub(crate) const MAX_EXTRA_KEYS: usize = 1;

/// 2^53: integers below it are exact in a double.
const EXACT_DOUBLE: f64 = 9_007_199_254_740_992.0;

/// The margin on the estimated rows when proving an `avg` exact.
const ROW_MARGIN: f64 = 8.0;

/// Pushes aggregates below joins, entering subquery plans. Also returns one
/// line per aggregate over a join: why it was pushed, or the gate that refused
/// it (`EXPLAIN VERBOSE`).
pub(crate) fn rewrite(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
) -> Result<(Arc<LogicalPlan>, Vec<String>), KrishivError> {
    let mut memo = HashMap::new();
    let mut next = max_id(&root, &mut memo).saturating_add(1);
    let mut est = Estimator::with_propagation(stats, propagate);
    let mut notes = Vec::new();
    let out = dag::rewrite_up(root, &mut |p| {
        let Some(shape) = Shape::of(&p) else {
            return Ok(p);
        };
        let tried = shape.push(shape.join, &mut est, &mut next).or_else(|why| {
            // The dimension the keys need may sit below another one.
            shape
                .rotated()
                .and_then(|join| shape.push(&join, &mut est, &mut next).ok())
                .ok_or(why)
        });
        match tried {
            Ok((pushed, note)) => {
                notes.push(note);
                if next > MAX_ID {
                    return Err(InternalError {
                        message: "attribute ids exhausted by eager aggregation".to_owned(),
                    }
                    .into());
                }
                Ok(pushed)
            }
            Err(why) => {
                notes.push(format!("not applied: {why}"));
                Ok(p)
            }
        }
    })?;
    Ok((out, notes))
}

/// An aggregate over projections and expansions over a join.
struct Shape<'a> {
    plan: &'a LogicalPlan,
    keys: &'a [Arc<Expr>],
    outputs: &'a [Arc<Expr>],
    /// Between the aggregate and the join, top first.
    spine: Vec<&'a LogicalPlan>,
    join: &'a Arc<LogicalPlan>,
}

impl<'a> Shape<'a> {
    fn of(plan: &'a LogicalPlan) -> Option<Self> {
        let PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(keys),
            aggregates: outputs,
        } = &plan.node
        else {
            return None;
        };
        let mut spine = Vec::new();
        let mut at = input;
        loop {
            match &at.node {
                PlanNode::Project { input, .. } | PlanNode::Expand { input, .. } => {
                    spine.push(at.as_ref());
                    at = input;
                }
                PlanNode::Join { .. } => {
                    return Some(Self {
                        plan,
                        keys,
                        outputs,
                        spine,
                        join: at,
                    });
                }
                _ => return None,
            }
        }
    }

    /// The rewritten aggregate and what the gates saw, or why it stays.
    fn push(
        &self,
        join: &Arc<LogicalPlan>,
        est: &mut Estimator<'_>,
        next: &mut u64,
    ) -> Result<(Arc<LogicalPlan>, String), String> {
        let calls = distinct_calls(self.outputs);
        if calls.is_empty() {
            return Err("no aggregate call".to_owned());
        }
        let mut args = Attrs::new();
        for (_, f) in &calls {
            for a in &f.args {
                references_into(a, &mut args);
            }
        }
        let need = self.read_above(&args)?;

        let PlanNode::Join {
            left,
            right,
            kind,
            condition,
        } = &join.node
        else {
            return Err("not a join".to_owned());
        };
        if !matches!(
            kind,
            JoinKind::Inner | JoinKind::LeftOuter | JoinKind::RightOuter
        ) {
            return Err(format!("a {kind:?} join"));
        }
        let JoinCondition::On(on) = condition else {
            return Err("a join without an ON condition".to_owned());
        };
        if !is_deterministic(on) || has_subquery(on) {
            return Err("the join condition is nondeterministic or holds a subquery".to_owned());
        }
        let (left_cols, right_cols) = (attrs_of(left)?, attrs_of(right)?);
        let in_left = args.iter().all(|a| contains(&left_cols, *a));
        let in_right = args.iter().all(|a| contains(&right_cols, *a));
        let left_side = match (in_left, in_right) {
            (true, false) => true,
            (false, true) => false,
            (false, false) => return Err("arguments read both join sides".to_owned()),
            // No arguments at all: the side with more rows has more to save.
            (true, true) => est.plan(left).rows >= est.plan(right).rows,
        };
        let (side, cols) = if left_side {
            (left, &left_cols)
        } else {
            (right, &right_cols)
        };
        let other = if left_side { right } else { left };
        if !plain_scan(other) {
            return Err("the other side is filtered or joined: its runtime filter is worth more than the aggregate".to_owned());
        }
        let null_supplying = matches!(
            (kind, left_side),
            (JoinKind::LeftOuter, false) | (JoinKind::RightOuter, true)
        );

        let side_est = est.plan(side);
        for (_, f) in &calls {
            refusal_of_call(f, &side_est, null_supplying)?;
        }

        // The columns of `S` that stay readable: the join condition's and
        // whatever is read above the join other than the calls' arguments.
        let equi = equi_columns(on, cols);
        if equi.is_empty() {
            return Err("no equality on a column of the aggregated side".to_owned());
        }
        let mut wanted = need;
        references_into(on, &mut wanted);
        let keys: Vec<BoundColumn> = cols
            .iter()
            .filter(|c| wanted.contains(&c.id))
            .cloned()
            .collect();
        let extra = keys.iter().filter(|k| !equi.contains(&k.id)).count();
        let gates = Gates {
            rows: side_est.rows,
            joined: est.plan(join).rows,
            groups: groups_of(&keys, &side_est)?,
            extra,
        };
        gates.check()?;

        // Build, bottom up.
        let mut states = Vec::new();
        let mut merged: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();
        for (c, f) in &calls {
            let raw = f
                .args
                .first()
                .and_then(|a| data_type(a))
                .unwrap_or(DataType::Long);
            let columns = partial_state_columns(f, &raw, AttrId(*next))
                .ok_or_else(|| format!("{} exports no states", f.id.name()))?;
            *next = next.saturating_add(columns.len() as u64);
            let mut upper = ResolvedFunction::clone(f);
            upper.args = vec![resolved(ResolvedExpr::StateInput {
                columns: columns.iter().map(column).collect(),
                raw_type: raw,
            })];
            merged.push((Arc::clone(c), resolved(ResolvedExpr::Function(upper))));
            states.extend(columns);
        }
        let partial = Arc::new(LogicalPlan::new(PlanNode::PartialAggregate {
            input: Arc::clone(side),
            keys: keys.iter().map(column).collect(),
            aggregates: calls.iter().map(|(c, _)| Arc::clone(c)).collect(),
            states: states.clone(),
        }));
        let (new_left, new_right) = if left_side {
            (partial, Arc::clone(right))
        } else {
            (Arc::clone(left), partial)
        };
        let mut below = Arc::new(
            LogicalPlan::new(PlanNode::Join {
                left: new_left,
                right: new_right,
                kind: *kind,
                condition: condition.clone(),
            })
            .with_origin(join.origin),
        );
        // The projections above: the arguments out, the states in.
        let mut read = self.read_by_aggregate();
        let mut plans = Vec::with_capacity(self.spine.len());
        for node in &self.spine {
            plans.push((*node, read.clone()));
            read = read_below(node, &read, &args)?;
        }
        for (node, read) in plans.into_iter().rev() {
            below = rebuilt(node, below, &read, &states)?;
        }
        let outputs = self
            .outputs
            .iter()
            .map(|e| replace_calls(e, &merged))
            .collect();
        let pushed = Arc::new(
            LogicalPlan::new(PlanNode::Aggregate {
                input: below,
                grouping: Grouping::GroupBy(self.keys.to_vec()),
                aggregates: outputs,
            })
            .with_origin(self.plan.origin),
        );
        let names: Vec<&str> = keys.iter().map(|k| k.name.as_str()).collect();
        Ok((pushed, format!("pushed by [{}]: {gates}", names.join(", "))))
    }

    /// `(X JOIN Y) JOIN C` as `(X JOIN C) JOIN Y`, all inner, where the calls
    /// read only `X` and `C` joins only `X`: the same rows, with `Y` on top.
    /// Join ordering breaks near-ties without regard to the aggregate, and only
    /// the top join's dimension leaves its keys out of `S`'s.
    fn rotated(&self) -> Option<Arc<LogicalPlan>> {
        let (left, right, on) = inner_join(self.join)?;
        let mut args = Attrs::new();
        for (_, f) in distinct_calls(self.outputs) {
            for a in &f.args {
                references_into(a, &mut args);
            }
        }
        if args.is_empty() {
            return None;
        }
        let holds = |p: &LogicalPlan| {
            p.resolved_output()
                .is_some_and(|cols| args.iter().all(|a| contains(&cols, *a)))
        };
        let (side, other) = if holds(left) {
            (left, right)
        } else if holds(right) {
            (right, left)
        } else {
            return None;
        };
        // A projection of plain columns only hides columns: nothing above
        // reads them, and pruning drops them again.
        let mut below = side;
        while let PlanNode::Project { input, exprs } = &below.node {
            if !exprs
                .iter()
                .all(|e| matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Column(_))))
            {
                return None;
            }
            below = input;
        }
        let (a, b, inner_on) = inner_join(below)?;
        let (x, y) = if holds(a) {
            (a, b)
        } else if holds(b) {
            (b, a)
        } else {
            return None;
        };
        let (x_cols, c_cols) = (x.resolved_output()?, other.resolved_output()?);
        if !references(on)
            .iter()
            .all(|id| contains(&x_cols, *id) || contains(&c_cols, *id))
        {
            return None;
        }
        let joined = |left: &Arc<LogicalPlan>, right: &Arc<LogicalPlan>, on: &Arc<Expr>| {
            Arc::new(LogicalPlan::new(PlanNode::Join {
                left: Arc::clone(left),
                right: Arc::clone(right),
                kind: JoinKind::Inner,
                condition: JoinCondition::On(Arc::clone(on)),
            }))
        };
        Some(joined(&joined(x, other, on), y, inner_on))
    }

    /// What the aggregate reads apart from its calls' arguments.
    fn read_by_aggregate(&self) -> Attrs {
        let mut read = Attrs::new();
        for k in self.keys {
            references_into(k, &mut read);
        }
        let none = lit(Literal::Null(None));
        for e in self.outputs {
            references_into(&without_calls(e, &none), &mut read);
        }
        read
    }

    /// What the join's output must keep apart from the arguments, walking the
    /// spine down; an error when an argument is computed or renamed on the way.
    fn read_above(&self, args: &Attrs) -> Result<Attrs, String> {
        if self
            .keys
            .iter()
            .chain(self.outputs)
            .any(|e| has_subquery(e) || has_outer_reference(e))
        {
            return Err("the aggregate holds a subquery or an outer reference".to_owned());
        }
        let mut read = self.read_by_aggregate();
        for node in &self.spine {
            read = read_below(node, &read, args)?;
        }
        Ok(read)
    }
}

/// Whether `p` is a table read whole, under projections.
fn plain_scan(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::Project { input, .. } => plain_scan(input),
        PlanNode::Scan(_) => true,
        _ => false,
    }
}

/// The inputs and condition of an inner join on a deterministic condition
/// without subqueries.
fn inner_join(p: &LogicalPlan) -> Option<(&Arc<LogicalPlan>, &Arc<LogicalPlan>, &Arc<Expr>)> {
    match &p.node {
        PlanNode::Join {
            left,
            right,
            kind: JoinKind::Inner,
            condition: JoinCondition::On(on),
        } if is_deterministic(on) && !has_subquery(on) => Some((left, right, on)),
        _ => None,
    }
}

/// What `node`'s input must provide, apart from the arguments, given what is
/// read of `node`. An item nothing reads goes (this runs before pruning).
fn read_below(node: &LogicalPlan, read: &Attrs, args: &Attrs) -> Result<Attrs, String> {
    let mut below = Attrs::new();
    let mut item = |out: &BoundColumn, values: &[&Arc<Expr>]| -> Result<(), String> {
        if !read.contains(&out.id) {
            if args.contains(&out.id) && !values.iter().all(|v| is_column(v, out.id)) {
                return Err(format!(
                    "the argument {} is computed above the join",
                    out.name
                ));
            }
            return Ok(());
        }
        for v in values {
            if has_subquery(v) || has_outer_reference(v) {
                return Err("a projection above the join holds a subquery".to_owned());
            }
            references_into(v, &mut below);
        }
        Ok(())
    };
    match &node.node {
        PlanNode::Project { exprs, .. } => {
            for e in exprs {
                let out = output_of(e).ok_or("an unnamed projection above the join")?;
                item(out, &[e])?;
            }
        }
        PlanNode::Expand {
            projections,
            output,
            ..
        } => {
            for (i, out) in output.iter().enumerate() {
                let values: Vec<&Arc<Expr>> = projections.iter().filter_map(|p| p.get(i)).collect();
                item(out, &values)?;
            }
        }
        _ => return Err("not a projection".to_owned()),
    }
    Ok(below)
}

/// `node` over `input`, without the items `read` drops and with `states`.
fn rebuilt(
    node: &LogicalPlan,
    input: Arc<LogicalPlan>,
    read: &Attrs,
    states: &[BoundColumn],
) -> Result<Arc<LogicalPlan>, String> {
    let state_columns: Vec<Arc<Expr>> = states.iter().map(column).collect();
    let next = match &node.node {
        PlanNode::Project { exprs, .. } => {
            let mut kept: Vec<Arc<Expr>> = exprs
                .iter()
                .filter(|e| output_of(e).is_some_and(|out| read.contains(&out.id)))
                .cloned()
                .collect();
            kept.extend(state_columns);
            PlanNode::Project { input, exprs: kept }
        }
        PlanNode::Expand {
            projections,
            output,
            ..
        } => {
            let at: Vec<usize> = (0..output.len())
                .filter(|&i| read.contains(&output[i].id))
                .collect();
            PlanNode::Expand {
                input,
                projections: projections
                    .iter()
                    .map(|p| {
                        at.iter()
                            .filter_map(|&i| p.get(i).cloned())
                            .chain(state_columns.iter().cloned())
                            .collect()
                    })
                    .collect(),
                output: at
                    .iter()
                    .map(|&i| output[i].clone())
                    .chain(states.iter().cloned())
                    .collect(),
            }
        }
        _ => return Err("not a projection".to_owned()),
    };
    Ok(Arc::new(LogicalPlan::new(next).with_origin(node.origin)))
}

/// What the gates read.
#[derive(Debug, Clone, Copy)]
struct Gates {
    /// `S`'s rows.
    rows: f64,
    /// The join's rows.
    joined: f64,
    /// `S`'s groups.
    groups: f64,
    /// Keys beyond the join's own.
    extra: usize,
}

impl Gates {
    fn check(self) -> Result<(), String> {
        if self.joined < MIN_RETENTION * self.rows {
            return Err(format!("retention below {MIN_RETENTION}: {self}"));
        }
        if self.rows < MIN_REDUCTION * self.groups {
            return Err(format!("reduction below {MIN_REDUCTION}: {self}"));
        }
        if self.extra > MAX_EXTRA_KEYS {
            return Err(format!("more than {MAX_EXTRA_KEYS} extra key: {self}"));
        }
        Ok(())
    }
}

impl std::fmt::Display for Gates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rows {:.0}, joined {:.0}, groups {:.0}, extra keys {}",
            self.rows, self.joined, self.groups, self.extra
        )
    }
}

/// The combined distinct count of `keys` limited to the side's rows, or which
/// key's count is only a guess.
fn groups_of(keys: &[BoundColumn], side: &RelStats) -> Result<f64, String> {
    let mut groups = 1.0_f64;
    for k in keys {
        let known = side.cols.get(&k.id).filter(|c| c.ndv_known());
        let Some(c) = known else {
            return Err(format!("the distinct count of {} is a guess", k.name));
        };
        groups *= c.ndv;
    }
    Ok(groups.min(side.rows))
}

/// Why `f` cannot run below the join, on a side with these statistics.
fn refusal_of_call(
    f: &ResolvedFunction,
    side: &RelStats,
    null_supplying: bool,
) -> Result<(), String> {
    let name = f.id.name();
    if f.args.iter().any(|a| {
        !is_deterministic(a)
            || has_subquery(a)
            || has_outer_reference(a)
            || is_state(a)
            || may_raise(a, true)
    }) {
        return Err(format!("{name}'s argument can raise or is not plain"));
    }
    if !f.deterministic {
        return Err(format!("{name} is nondeterministic"));
    }
    if null_supplying && (f.args.is_empty() || !f.args.iter().all(|a| under_casts(a).is_some())) {
        return Err(format!(
            "{name} on the null-supplying side counts unmatched rows"
        ));
    }
    let arg = f.args.first();
    let source = arg.map(|a| source_type(a));
    let order_free = match (name, &source) {
        ("count", _) => true,
        ("sum", Some(Some(t))) => small_integral(t) || decimal_fits(t),
        ("avg" | "mean", Some(Some(t))) if decimal_fits(t) => true,
        ("avg" | "mean", Some(Some(t))) if small_integral(t) => {
            arg.is_some_and(|a| avg_is_exact(a, t, side))
        }
        ("min" | "max", Some(Some(t))) => matches!(
            t,
            DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Decimal(_)
                | DataType::Date
                | DataType::Timestamp
                | DataType::TimestampNtz
                | DataType::String
        ),
        _ => false,
    };
    if !order_free {
        return Err(format!(
            "{name} over {source:?} can depend on the order of its rows"
        ));
    }
    // The registry's other aggregates, `DISTINCT`, `FILTER`, ordered sets.
    let raw = source.flatten().unwrap_or(DataType::Long);
    if f.kind != FunctionKind::Aggregate || partial_state_columns(f, &raw, AttrId(0)).is_none() {
        return Err(format!("{name} exports no states"));
    }
    Ok(())
}

fn small_integral(t: &DataType) -> bool {
    matches!(t, DataType::Byte | DataType::Short | DataType::Integer)
}

fn decimal_fits(t: &DataType) -> bool {
    matches!(t, DataType::Decimal(d) if d.precision() <= 28)
}

/// The type `e` reads, through casts between integers that widen it: their
/// values are the source's.
fn source_type(e: &Expr) -> Option<DataType> {
    if let ExprNode::Cast { expr, to, .. } = &e.node
        && let Some(from) = data_type(expr)
        && widens(&from, to)
    {
        return source_type(expr);
    }
    data_type(e)
}

fn widens(from: &DataType, to: &DataType) -> bool {
    let rank = |t: &DataType| match t {
        DataType::Byte => Some(1),
        DataType::Short => Some(2),
        DataType::Integer => Some(3),
        DataType::Long => Some(4),
        _ => None,
    };
    matches!((rank(from), rank(to)), (Some(a), Some(b)) if a <= b)
}

/// Whether every partial sum of `avg(arg)` stays exact in a double.
fn avg_is_exact(arg: &Expr, source: &DataType, side: &RelStats) -> bool {
    let of_type = match source {
        DataType::Byte => 128.0,
        DataType::Short => 32_768.0,
        _ => 2_147_483_648.0,
    };
    let magnitude = under_casts(arg)
        .and_then(|c| side.cols.get(&c.id))
        .and_then(|c| Some(c.min?.abs().max(c.max?.abs())))
        .map_or(of_type, |m: f64| m.min(of_type));
    magnitude * side.rows * ROW_MARGIN < EXACT_DOUBLE
}

/// The column `e` is, through casts.
fn under_casts(e: &Expr) -> Option<&BoundColumn> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c),
        ExprNode::Cast { expr, .. } => under_casts(expr),
        _ => None,
    }
}

fn is_column(e: &Expr, id: AttrId) -> bool {
    matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Column(c)) if c.id == id)
}

fn is_state(e: &Expr) -> bool {
    matches!(&e.node, ExprNode::Resolved(ResolvedExpr::StateInput { .. }))
}

/// The columns of `cols` that an equality conjunct of `on` compares directly.
fn equi_columns(on: &Arc<Expr>, cols: &[BoundColumn]) -> Attrs {
    let mut out = Attrs::new();
    for c in conjuncts(on) {
        if let Some((Builtin::Eq | Builtin::EqNullSafe, args, _)) = builtin(&c) {
            for a in args {
                if let ExprNode::Resolved(ResolvedExpr::Column(col)) = &a.node
                    && contains(cols, col.id)
                {
                    out.insert(col.id);
                }
            }
        }
    }
    out
}

fn contains(cols: &[BoundColumn], id: AttrId) -> bool {
    cols.iter().any(|c| c.id == id)
}

fn attrs_of(p: &LogicalPlan) -> Result<Vec<BoundColumn>, String> {
    p.resolved_output()
        .ok_or_else(|| "a join side without a resolved output".to_owned())
}

fn resolved(r: ResolvedExpr) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(r)))
}

fn is_call(e: &Expr) -> bool {
    matches!(
        &e.node,
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate
    )
}

/// The aggregate calls under `outputs`, once each.
fn distinct_calls(outputs: &[Arc<Expr>]) -> Vec<(&Arc<Expr>, &ResolvedFunction)> {
    let mut calls: Vec<(&Arc<Expr>, &ResolvedFunction)> = Vec::new();
    for e in outputs {
        let mut stack = vec![e];
        while let Some(x) = stack.pop() {
            match &x.node {
                ExprNode::Resolved(ResolvedExpr::Function(f))
                    if f.kind == FunctionKind::Aggregate =>
                {
                    if !calls.iter().any(|(c, _)| same(c, x)) {
                        calls.push((x, f));
                    }
                }
                _ => x.for_each_child(&mut |c| stack.push(c)),
            }
        }
    }
    calls
}

/// `e` with each call in `merged` replaced.
fn replace_calls(e: &Arc<Expr>, merged: &[(Arc<Expr>, Arc<Expr>)]) -> Arc<Expr> {
    let out = Arc::clone(e).transform_up(&mut |x| {
        Ok::<_, std::convert::Infallible>(if is_call(&x) {
            merged
                .iter()
                .find(|(c, _)| same(c, &x))
                .map_or(x, |(_, m)| Arc::clone(m))
        } else {
            x
        })
    });
    match out {
        Ok(v) => v,
        Err(never) => match never {},
    }
}

/// `e` with each call replaced by `none`.
fn without_calls(e: &Arc<Expr>, none: &Arc<Expr>) -> Arc<Expr> {
    let out = Arc::clone(e).transform_up(&mut |x| {
        Ok::<_, std::convert::Infallible>(if is_call(&x) { Arc::clone(none) } else { x })
    });
    match out {
        Ok(v) => v,
        Err(never) => match never {},
    }
}
