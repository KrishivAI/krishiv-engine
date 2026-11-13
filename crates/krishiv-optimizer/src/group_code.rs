// SPDX-License-Identifier: Apache-2.0
//! `GroupKeyCode` (design P2b D2): an aggregate's group keys that come from one
//! dimension `D` travel through the joins as one integer code, and come back
//! from `D`'s distinct values above the aggregate.
//!
//! ```text
//! Project [the aggregate's outputs; each key of D := dict.key]
//! +- Join Inner (code = dict.code)                       the decode join
//!    :- Aggregate [code, other keys], [code, other keys, calls]
//!    :  +- I'  (I with D replaced by D_c; the dropped keys left out)
//!    +- Shift(dict)                                      read 1
//! dict := Window [code := row_number() OVER (ORDER BY keys)]
//!         +- Aggregate [keys], [keys] over Shift(D)      one row per distinct key
//! D_c  := Shift 0 (Project [D's columns but the dropped keys, code]
//!                  +- Join Inner (D.key <=> dict.key, each key)
//!                     :- D
//!                     +- Shift(dict))                     read 2
//! ```
//!
//! `dict` numbers the distinct key tuples of `D` in one total order, so both
//! reads give each tuple the same code and distinct tuples distinct codes,
//! whatever the multiplicity of `D`'s join key; grouping by the code makes the
//! groups the keys made, with the same rows, and the decode join maps each code
//! back to its one tuple. A null key is one tuple of `dict`, joined by `<=>`.
//!
//! # When it applies (design section 2.2)
//!
//! 1. A plain `GROUP BY` of columns. From `D` up to the aggregate there are only
//!    filters, projections and joins, and `D`'s side of each join is preserved
//!    (inner; the left of a left outer, semi, anti or existence join; the right
//!    of a right outer join). `D`'s own join (the lowest on the path) is an inner
//!    equi-join whose keys on `D`'s side are not group keys.
//! 2. A key read on the way (a condition, a computed projection, an aggregate
//!    call) stays carried; the others are dropped from the payload. Outputs that
//!    read keys outside an aggregate call move to the top projection.
//! 3. Every key of `D` is integral, decimal, date, timestamp, boolean, binary or
//!    a string of the binary collation: grouping compares them as bytes.
//! 4. `D` is a scan with deterministic filters and column-only projections: it
//!    is read twice, and both reads see the same rows.
//! 5. Statistics say it pays: `NDV(keys) <= rows(I) / 4`, `NDV(keys) <= 10^7`
//!    (`dict` is numbered in one partition), `rows(I) >= 2 rows(D)`, and the
//!    dropped keys are wider than 8 bytes; an unknown estimate refuses.
//! 6. Exactly one input of the aggregate qualifies, and no key is already a
//!    code: a second optimization then finds nothing new.
//!
//! The fence: `D_c` is a `Shift` (offset 0, so every column above keeps its id),
//! which join ordering treats as one input, so `D` and `dict` are never folded
//! into the main join tree. Both reads of `dict` are `Shift`s of one body, so
//! `ShareCte` computes it once.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_functions::registry::{self, Lookup};
use krishiv_functions::signature::SparkProfile;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FrameUnits, FunctionId, FunctionKind, Grouping,
    JoinCondition, JoinKind, LogicalPlan, PlanNode, ResolvedExpr, ResolvedFunction, SortOrder,
    WindowExpr, WindowFrame, WindowSpec, WindowSpecRef, runs_on_any_row,
};
use krishiv_types::DataType;

use crate::dag;
use crate::estimate::Estimator;
use crate::expr::{
    and_all, builtin, call, column, conjuncts, equality, has_aggregate, has_outer_reference,
    has_subquery, is_deterministic, references, references_into, substitute,
};
use crate::plan_stats::default_size;
use crate::restrict::{MAX_ID, max_id};
use crate::stats::Statistics;

/// The aggregate's input must have at least this many rows per group.
const MIN_ROWS_PER_GROUP: f64 = 4.0;

/// The most distinct key tuples: `dict` is numbered in one partition.
const MAX_GROUPS: f64 = 10_000_000.0;

/// The aggregate's input must have at least this many rows per row of `D`:
/// the encoding join reads all of `D`.
const MIN_INPUT_RATIO: f64 = 2.0;

/// The name of the code column (an `int`: `dict` holds fewer than
/// [`MAX_GROUPS`] rows by estimate, far below `2^31`).
const CODE_NAME: &str = "krishiv_group_code";

/// The dropped keys must be wider than this many bytes (the code's width).
const CODE_WIDTH: u64 = 8;

/// Codes the dimension group keys of every aggregate where the statistics say
/// it pays, entering subquery plans. Also returns one line per aggregate with
/// keys from a dimension: what was coded, or why not (`EXPLAIN VERBOSE`).
pub(crate) fn rewrite(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
    force: bool,
) -> Result<(Arc<LogicalPlan>, Vec<String>), KrishivError> {
    let Some(row_number) = row_number() else {
        return Ok((root, Vec::new()));
    };
    let mut next = max_id(&root, &mut Default::default()).saturating_add(1);
    let mut est = Estimator::with_propagation(stats, propagate);
    let mut notes = Vec::new();
    let out = dag::rewrite_up(root, &mut |p| {
        if let Some(name) = collated_key(&p) {
            notes.push(format!(
                "not applied: {name} is a collated string, grouped by its collation key"
            ));
            return Ok(p);
        }
        let Some(shape) = Shape::of(&p) else {
            return Ok(p);
        };
        match shape.code(&mut est, &mut next, &row_number, force) {
            Ok(None) => Ok(p),
            Ok(Some((coded, note))) => {
                notes.push(note);
                Ok(coded)
            }
            Err(why) => {
                notes.push(format!("not applied: {why}"));
                Ok(p)
            }
        }
    })?;
    Ok((out, notes))
}

/// The column under a `__collation_key` group key of the aggregate `p`, if
/// any: such a key groups by the collation (condition 3), not by bytes.
fn collated_key(p: &LogicalPlan) -> Option<String> {
    let PlanNode::Aggregate {
        grouping: Grouping::GroupBy(keys),
        ..
    } = &p.node
    else {
        return None;
    };
    keys.iter().find_map(|k| match &k.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.id.name() == "__collation_key" => {
            match f.args.first().map(|a| &a.node) {
                Some(ExprNode::Resolved(ResolvedExpr::Column(c))) => Some(c.name.clone()),
                _ => Some("a key".to_owned()),
            }
        }
        _ => None,
    })
}

/// The registry's `row_number`.
fn row_number() -> Option<FunctionId> {
    match registry::builtins().lookup("row_number", SparkProfile::default()) {
        Lookup::Found(def) if def.kind == FunctionKind::Window => Some(FunctionId::Registry {
            name: def.name.to_owned(),
            id: def.id,
        }),
        _ => None,
    }
}

/// An aggregate grouping by columns.
struct Shape<'a> {
    plan: &'a LogicalPlan,
    input: &'a Arc<LogicalPlan>,
    keys: Vec<BoundColumn>,
    outputs: &'a [Arc<Expr>],
}

/// An input of the aggregate's filter, projection and join tree.
struct Leaf {
    plan: Arc<LogicalPlan>,
    /// The nodes from the aggregate's input down to the leaf's parent.
    path: Vec<Arc<LogicalPlan>>,
    /// Whether every join on the path keeps all of the leaf's rows.
    preserved: bool,
}

/// A dimension that qualifies.
struct Candidate {
    leaf: Leaf,
    /// The group keys it produces, in key order.
    keys: Vec<BoundColumn>,
    /// The keys nothing reads on the way: they leave the payload.
    dropped: HashSet<AttrId>,
    note: String,
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
        let keys = keys
            .iter()
            .map(|k| match &k.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.clone()),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        if keys.is_empty()
            || outputs
                .iter()
                .any(|e| has_subquery(e) || has_outer_reference(e))
        {
            return None;
        }
        Some(Self {
            plan,
            input,
            keys,
            outputs,
        })
    }

    /// The coded plan and what was decided, `None` when no key comes from a
    /// dimension, or why it stays.
    fn code(
        &self,
        est: &mut Estimator<'_>,
        next: &mut u64,
        row_number: &FunctionId,
        force: bool,
    ) -> Result<Option<(Arc<LogicalPlan>, String)>, String> {
        let mut leaves = Vec::new();
        walk(self.input, true, &mut Vec::new(), &mut leaves);
        let mut found = Vec::new();
        let mut refusals = Vec::new();
        let mut coded = false;
        for leaf in leaves {
            let out = leaf.plan.resolved_output().unwrap_or_default();
            let keys: Vec<BoundColumn> = self
                .keys
                .iter()
                .filter(|k| out.iter().any(|c| c.id == k.id))
                .cloned()
                .collect();
            if keys.is_empty() {
                continue;
            }
            coded |= is_code(&leaf.plan);
            match self.qualify(leaf, keys, est, force) {
                Ok(c) => found.push(c),
                Err(why) => refusals.push(why),
            }
        }
        if coded {
            return Err("a key is already a code".to_owned());
        }
        if found.len() > 1 {
            return Err(format!(
                "{} dimensions qualify (one per aggregate)",
                found.len()
            ));
        }
        let Some(c) = found.pop() else {
            return match refusals.into_iter().next() {
                Some(why) => Err(why),
                None => Ok(None),
            };
        };
        let note = c.note.clone();
        let plan = self.build(&c, next, row_number)?;
        Ok(Some((plan, note)))
    }

    /// Whether `leaf` qualifies as the dimension of `keys` (conditions 1-5).
    fn qualify(
        &self,
        leaf: Leaf,
        keys: Vec<BoundColumn>,
        est: &mut Estimator<'_>,
        force: bool,
    ) -> Result<Candidate, String> {
        let names = keys
            .iter()
            .map(|k| k.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        if !leaf.preserved {
            return Err(format!("[{names}] come from a null-padded side of a join"));
        }
        if !scan_chain(&leaf.plan) {
            return Err(format!("[{names}] do not come from a scan with filters"));
        }
        // `D` is read twice, and only the read under `D_c` can receive a
        // runtime filter: a filter that can raise (ANSI) could then run on a
        // row the original plan never evaluated it on; an outer reference
        // would put a correlated `dict` under the window.
        if !quiet_filters(&leaf.plan) {
            return Err(format!(
                "[{names}] come from a filter that can raise or reads an outer column"
            ));
        }
        // `D`'s own join: inner, on equalities whose `D` columns are not keys.
        let own = leaf
            .path
            .iter()
            .rev()
            .find(|p| matches!(p.node, PlanNode::Join { .. }))
            .ok_or_else(|| format!("[{names}] are not joined"))?;
        let d_cols = leaf.plan.resolved_output().unwrap_or_default();
        let in_d = |id: AttrId| d_cols.iter().any(|c| c.id == id);
        let PlanNode::Join {
            kind: JoinKind::Inner,
            condition: JoinCondition::On(on),
            ..
        } = &own.node
        else {
            return Err(format!("the join of [{names}] is not an inner join"));
        };
        let mut joined = false;
        for c in conjuncts(on) {
            if let Some([a, b]) = equality(&c)
                && in_d(a.id) != in_d(b.id)
            {
                let d = if in_d(a.id) { a } else { b };
                if keys.iter().any(|k| k.id == d.id) {
                    return Err(format!("{} is a group key and its join's key", d.name));
                }
                joined = true;
            }
        }
        if !joined {
            return Err(format!("the join of [{names}] is not an equi-join"));
        }
        if let Some(k) = keys.iter().find(|k| !codable(&k.data_type)) {
            return Err(format!(
                "{} is of type {}: grouping does not compare it as bytes",
                k.name,
                k.data_type.simple_string()
            ));
        }

        // The keys read on the way stay carried.
        let mut read = Attrs::default();
        for p in &leaf.path {
            match &p.node {
                PlanNode::Project { exprs, .. } => {
                    for e in exprs {
                        if !matches!(e.node, ExprNode::Resolved(ResolvedExpr::Column(_))) {
                            references_into(e, &mut read);
                        }
                    }
                }
                _ => p.for_each_expr(&mut |e| references_into(e, &mut read)),
            }
        }
        for e in self.outputs {
            if has_aggregate(e) {
                if references(&without_calls(e))
                    .iter()
                    .any(|id| keys.iter().any(|k| k.id == *id))
                {
                    return Err(format!(
                        "an output reads one of [{names}] beside an aggregate call"
                    ));
                }
                calls_read(e, &mut read);
            }
        }
        let dropped: HashSet<AttrId> = keys
            .iter()
            .map(|k| k.id)
            .filter(|id| !read.contains(id))
            .collect();
        let width: u64 = keys
            .iter()
            .filter(|k| dropped.contains(&k.id))
            .map(|k| default_size(&k.data_type))
            .sum();
        if width <= CODE_WIDTH {
            return Err(format!(
                "the keys of [{names}] it would drop are {width} bytes wide"
            ));
        }

        // Pays off.
        let d = est.plan(&leaf.plan);
        let i = est.plan(self.input);
        if !d.from_statistics || !i.from_statistics {
            return Err(format!("no statistics for the input of [{names}]"));
        }
        let mut groups = 1.0_f64;
        let mut guessed = None;
        for k in &keys {
            match d.cols.get(&k.id).filter(|c| c.ndv_known()) {
                Some(c) => groups *= c.ndv,
                None => guessed = guessed.or(Some(k.name.as_str())),
            }
        }
        let (rows_i, rows_d) = (gate_rows(est, self.input), d.rows_for_costing());
        if force {
            // Measurement only (`krishiv.experimental.groupKeyCodeForce`):
            // the shape qualifies, the statistics are not asked.
            let note = format!(
                "coded [{names}] (forced): input rows {rows_i:.0}, dimension rows {rows_d:.0}, \
                 {width} key bytes dropped"
            );
            return Ok(Candidate {
                leaf,
                keys,
                dropped,
                note,
            });
        }
        if let Some(name) = guessed {
            return Err(format!("the distinct count of {name} is a guess"));
        }
        let groups = groups.min(rows_d);
        let gates =
            format!("groups {groups:.0}, input rows {rows_i:.0}, dimension rows {rows_d:.0}");
        if groups * MIN_ROWS_PER_GROUP > rows_i {
            return Err(format!("[{names}]: too many groups ({gates})"));
        }
        if groups >= MAX_GROUPS {
            return Err(format!(
                "[{names}]: {MAX_GROUPS:.0} groups or more ({gates})"
            ));
        }
        if rows_i < MIN_INPUT_RATIO * rows_d {
            return Err(format!("[{names}]: the dimension is too large ({gates})"));
        }
        let note = format!("coded [{names}]: {gates}, {width} key bytes dropped");
        Ok(Candidate {
            leaf,
            keys,
            dropped,
            note,
        })
    }

    /// The rewrite of section 2.3.
    fn build(
        &self,
        c: &Candidate,
        next: &mut u64,
        row_number: &FunctionId,
    ) -> Result<Arc<LogicalPlan>, String> {
        let no_ids = || "the plan has no attribute ids left".to_owned();
        let d = &c.leaf.plan;
        let d_cols = d
            .resolved_output()
            .ok_or_else(|| "an unresolved dimension".to_owned())?;
        let top_d = max_id(d, &mut Default::default());

        // dict's body: the distinct keys of a fresh instance of D, numbered.
        let s0 = *next;
        *next = s0
            .checked_add(top_d)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(no_ids)?;
        let shifted_d = Arc::new(LogicalPlan::new(PlanNode::Shift {
            input: Arc::clone(d),
            offset: s0,
        }));
        let at = |k: &BoundColumn, offset: u64| -> Result<BoundColumn, String> {
            Ok(BoundColumn {
                id: AttrId(k.id.0.checked_add(offset).ok_or_else(no_ids)?),
                ..k.clone()
            })
        };
        let dict_keys = c
            .keys
            .iter()
            .map(|k| at(k, s0))
            .collect::<Result<Vec<_>, _>>()?;
        let distinct = Arc::new(LogicalPlan::new(PlanNode::Aggregate {
            input: shifted_d,
            grouping: Grouping::GroupBy(dict_keys.iter().map(column).collect()),
            aggregates: dict_keys.iter().map(column).collect(),
        }));
        let code = BoundColumn {
            id: AttrId(*next),
            name: CODE_NAME.to_owned(),
            data_type: DataType::Integer,
            nullable: false,
        };
        *next = next.checked_add(1).ok_or_else(no_ids)?;
        let order: Vec<SortOrder> = dict_keys
            .iter()
            .map(|k| SortOrder::asc(column(k)))
            .collect();
        let numbered = Arc::new(Expr::new(ExprNode::Window(WindowExpr {
            function: Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    id: row_number.clone(),
                    kind: FunctionKind::Window,
                    args: Vec::new(),
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
                    data_type: DataType::Integer,
                    nullable: false,
                    deterministic: true,
                },
            )))),
            window: WindowSpecRef::Spec(WindowSpec {
                partition_by: Vec::new(),
                order_by: order.clone(),
                frame: Some(WindowFrame::running(FrameUnits::Rows)),
            }),
        })));
        let body = Arc::new(LogicalPlan::new(PlanNode::Window {
            input: distinct,
            exprs: vec![Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: numbered,
                    output: code.clone(),
                },
            )))],
            partition_by: Vec::new(),
            order_by: order,
        }));
        // Two reads of the body under fresh ids each: every body id is at least `s0`.
        let read = |next: &mut u64| -> Result<(Arc<LogicalPlan>, u64), String> {
            let offset = next.checked_sub(s0).ok_or_else(no_ids)?;
            *next = code
                .id
                .0
                .checked_add(offset)
                .and_then(|n| n.checked_add(1))
                .filter(|n| *n <= MAX_ID)
                .ok_or_else(no_ids)?;
            let shift = Arc::new(LogicalPlan::new(PlanNode::Shift {
                input: Arc::clone(&body),
                offset,
            }));
            Ok((shift, offset))
        };
        let (decode_dict, o1) = read(next)?;
        let (encode_dict, o2) = read(next)?;

        // D_c: D with the code of its keys, under a fence.
        let on = and_all(
            c.keys
                .iter()
                .zip(&dict_keys)
                .map(|(k, dk)| {
                    Ok::<_, String>(call(
                        Builtin::EqNullSafe,
                        vec![column(k), column(&at(dk, o2)?)],
                        DataType::Boolean,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
        .ok_or_else(|| "no keys".to_owned())?;
        let encoded = Arc::new(LogicalPlan::new(PlanNode::Join {
            left: Arc::clone(d),
            right: encode_dict,
            kind: JoinKind::Inner,
            condition: JoinCondition::On(on),
        }));
        let row_code = at(&code, o2)?;
        let kept: Vec<Arc<Expr>> = d_cols
            .iter()
            .filter(|col| !c.dropped.contains(&col.id))
            .map(column)
            .chain(std::iter::once(column(&row_code)))
            .collect();
        let d_c = Arc::new(LogicalPlan::new(PlanNode::Shift {
            input: Arc::new(LogicalPlan::new(PlanNode::Project {
                input: encoded,
                exprs: kept,
            })),
            offset: 0,
        }));

        // I': the path rebuilt over D_c, the code carried and the dropped keys left out.
        let mut out = d_c;
        let mut below = Arc::clone(d);
        for node in c.leaf.path.iter().rev() {
            let rebuilt = match &node.node {
                PlanNode::Join {
                    left,
                    right,
                    kind,
                    condition,
                } => {
                    let (l, r) = if Arc::ptr_eq(left, &below) {
                        (out, Arc::clone(right))
                    } else {
                        (Arc::clone(left), out)
                    };
                    LogicalPlan::new(PlanNode::Join {
                        left: l,
                        right: r,
                        kind: *kind,
                        condition: condition.clone(),
                    })
                }
                PlanNode::Filter { predicate, .. } => LogicalPlan::new(PlanNode::Filter {
                    input: out,
                    predicate: Arc::clone(predicate),
                }),
                PlanNode::Project { exprs, .. } => {
                    let exprs = exprs
                        .iter()
                        .filter(|e| match &e.node {
                            ExprNode::Resolved(ResolvedExpr::Column(col)) => {
                                !c.dropped.contains(&col.id)
                            }
                            _ => true,
                        })
                        .cloned()
                        .chain(std::iter::once(column(&row_code)))
                        .collect();
                    LogicalPlan::new(PlanNode::Project { input: out, exprs })
                }
                _ => return Err("an unexpected node on the path".to_owned()),
            };
            below = Arc::clone(node);
            out = Arc::new(rebuilt.with_origin(node.origin));
        }

        // The aggregate by (code, the other keys); the outputs that read keys of
        // D outside a call move above the decode join.
        let rest: Vec<&BoundColumn> = self
            .keys
            .iter()
            .filter(|k| !c.keys.iter().any(|d| d.id == k.id))
            .collect();
        let is_d_key = |id: AttrId| c.keys.iter().any(|k| k.id == id);
        let mut agg_out: Vec<Arc<Expr>> = std::iter::once(column(&row_code))
            .chain(rest.iter().map(|k| column(k)))
            .collect();
        let decoded: std::collections::HashMap<AttrId, Arc<Expr>> = c
            .keys
            .iter()
            .zip(&dict_keys)
            .map(|(k, dk)| Ok((k.id, column(&at(dk, o1)?))))
            .collect::<Result<_, String>>()?;
        let mut top = Vec::with_capacity(self.outputs.len());
        for e in self.outputs {
            let plain = match &e.node {
                ExprNode::Resolved(ResolvedExpr::Column(col)) => Some(col),
                _ => None,
            };
            if let Some(col) = plain
                && rest.iter().any(|k| k.id == col.id)
            {
                top.push(Arc::clone(e));
            } else if has_aggregate(e) {
                agg_out.push(Arc::clone(e));
                let col = crate::expr::output_of(e)
                    .ok_or_else(|| "an aggregate output without a name".to_owned())?;
                top.push(column(col));
            } else if let Some(col) = plain
                && is_d_key(col.id)
            {
                let dict_col = decoded
                    .get(&col.id)
                    .ok_or_else(|| "a key without its dictionary column".to_owned())?;
                top.push(Arc::new(Expr::new(ExprNode::Resolved(
                    ResolvedExpr::Alias {
                        expr: Arc::clone(dict_col),
                        output: col.clone(),
                    },
                ))));
            } else {
                top.push(substitute(e, &decoded));
            }
        }
        let aggregate = Arc::new(
            LogicalPlan::new(PlanNode::Aggregate {
                input: out,
                grouping: Grouping::GroupBy(
                    std::iter::once(column(&row_code))
                        .chain(rest.iter().map(|k| column(k)))
                        .collect(),
                ),
                aggregates: agg_out,
            })
            .with_origin(self.plan.origin),
        );
        let decode = Arc::new(LogicalPlan::new(PlanNode::Join {
            left: aggregate,
            right: decode_dict,
            kind: JoinKind::Inner,
            condition: JoinCondition::On(call(
                Builtin::Eq,
                vec![column(&row_code), column(&at(&code, o1)?)],
                DataType::Boolean,
            )),
        }));
        Ok(Arc::new(
            LogicalPlan::new(PlanNode::Project {
                input: decode,
                exprs: top,
            })
            .with_origin(self.plan.origin),
        ))
    }
}

type Attrs = crate::expr::Attrs;

/// Collects the inputs of the filter, projection and join tree under `p`.
fn walk(
    p: &Arc<LogicalPlan>,
    preserved: bool,
    path: &mut Vec<Arc<LogicalPlan>>,
    out: &mut Vec<Leaf>,
) {
    if scan_chain(p) {
        out.push(Leaf {
            plan: Arc::clone(p),
            path: path.clone(),
            preserved,
        });
        return;
    }
    match &p.node {
        PlanNode::Filter { input, .. } | PlanNode::Project { input, .. } => {
            path.push(Arc::clone(p));
            walk(input, preserved, path, out);
            path.pop();
        }
        PlanNode::Join {
            left, right, kind, ..
        } => {
            let (keeps_left, keeps_right) = match kind {
                JoinKind::Inner | JoinKind::Cross => (true, true),
                JoinKind::LeftOuter
                | JoinKind::LeftSemi
                | JoinKind::LeftAnti
                | JoinKind::Existence(_) => (true, false),
                JoinKind::RightOuter => (false, true),
                _ => (false, false),
            };
            path.push(Arc::clone(p));
            walk(left, preserved && keeps_left, path, out);
            walk(right, preserved && keeps_right, path, out);
            path.pop();
        }
        _ => out.push(Leaf {
            plan: Arc::clone(p),
            path: path.clone(),
            preserved,
        }),
    }
}

/// The rows the gate reads as `rows(I)` (design section 2.2 condition 5): the
/// estimate of `p`, except under a null-aware `NOT IN` anti join on top (and a
/// column-only projection over one), which is read as its left input. The
/// estimator's anti join keeps `left - inner` rows (`estimate.rs`), and under
/// the null-aware condition `a = b OR isnull(a = b)` the `OR` makes `inner`
/// exceed `left`, so the estimate collapses to one row; the design reads TPC-H
/// q16's ratio "before its anti join". Every other semi, anti or existence
/// join is a filter of its own estimate (TPC-H q21's `EXISTS` / `NOT EXISTS`
/// keep a small part of their input) and is read as estimated.
fn gate_rows(est: &mut Estimator<'_>, p: &Arc<LogicalPlan>) -> f64 {
    match &p.node {
        PlanNode::Join {
            left,
            kind: JoinKind::LeftAnti,
            condition: JoinCondition::On(on),
            ..
        } if null_aware(on) => gate_rows(est, left),
        PlanNode::Project { input, exprs }
            if exprs
                .iter()
                .all(|e| matches!(e.node, ExprNode::Resolved(ResolvedExpr::Column(_))))
                && matches!(
                    &input.node,
                    PlanNode::Join {
                        kind: JoinKind::LeftAnti,
                        condition: JoinCondition::On(on),
                        ..
                    } if null_aware(on)
                ) =>
        {
            gate_rows(est, input)
        }
        _ => est.plan(p).rows_for_costing(),
    }
}

/// Whether `e` is Spark's null-aware `NOT IN` key `(a = b) OR isnull(a = b)`
/// over two columns (as the physical planner recognises it, `planner/join.rs`).
fn null_aware(e: &Expr) -> bool {
    let Some((Builtin::Or, [eq, is_null], _)) = builtin(e) else {
        return false;
    };
    let Some((Builtin::IsNull, [inner], _)) = builtin(is_null) else {
        return false;
    };
    match (equality(eq), equality(inner)) {
        (Some([a, b]), Some([c, d])) => a.id == c.id && b.id == d.id,
        _ => false,
    }
}

/// A scan under deterministic filters and column-only projections.
fn scan_chain(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::Scan(_) => true,
        PlanNode::Filter { input, predicate } => {
            is_deterministic(predicate) && !has_subquery(predicate) && scan_chain(input)
        }
        PlanNode::Project { input, exprs } => {
            exprs
                .iter()
                .all(|e| matches!(e.node, ExprNode::Resolved(ResolvedExpr::Column(_))))
                && scan_chain(input)
        }
        _ => false,
    }
}

/// Whether every filter of the scan chain `p` runs on any row without an
/// observable effect (`runs_on_any_row`, ANSI assumed) and reads no outer
/// column.
fn quiet_filters(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::Filter { input, predicate } => {
            runs_on_any_row(predicate, true)
                && !has_outer_reference(predicate)
                && quiet_filters(input)
        }
        PlanNode::Project { input, .. } => quiet_filters(input),
        _ => true,
    }
}

/// Whether `p` is a `D_c` this rule made: a `Shift` over a projection of a join
/// with a numbered dictionary.
fn is_code(p: &LogicalPlan) -> bool {
    let PlanNode::Shift { input, .. } = &p.node else {
        return false;
    };
    let mut at = input;
    while let PlanNode::Project { input, .. } | PlanNode::Filter { input, .. } = &at.node {
        at = input;
    }
    let PlanNode::Join { left, right, .. } = &at.node else {
        return false;
    };
    let numbered = |side: &LogicalPlan| {
        let mut s = side;
        loop {
            match &s.node {
                PlanNode::Shift { input, .. }
                | PlanNode::Project { input, .. }
                | PlanNode::Filter { input, .. } => s = input,
                PlanNode::Window { .. } => return true,
                // `ShareCte` made `dict` a `WITH` definition.
                PlanNode::CteRef(r) => return r.output.iter().any(|c| c.name == CODE_NAME),
                _ => return false,
            }
        }
    };
    numbered(left) || numbered(right)
}

/// Whether grouping compares values of `t` as their bytes.
fn codable(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Boolean
            | DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Decimal(_)
            | DataType::Date
            | DataType::Timestamp
            | DataType::TimestampNtz
            | DataType::Binary
            | DataType::String
    )
}

/// `e` with each aggregate call replaced by a null literal.
fn without_calls(e: &Arc<Expr>) -> Arc<Expr> {
    let none = Arc::new(Expr::literal(krishiv_plan::Literal::Null(None)));
    let out = Arc::clone(e).transform_down(&mut |x| {
        Ok::<_, std::convert::Infallible>(if is_call(&x) { Arc::clone(&none) } else { x })
    });
    match out {
        Ok(v) => v,
        Err(never) => match never {},
    }
}

/// Adds what the aggregate calls in `e` read to `out`.
fn calls_read(e: &Arc<Expr>, out: &mut Attrs) {
    if is_call(e) {
        references_into(e, out);
        return;
    }
    e.for_each_child(&mut |c| calls_read(c, out));
}

fn is_call(e: &Expr) -> bool {
    matches!(
        &e.node,
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate
    )
}
