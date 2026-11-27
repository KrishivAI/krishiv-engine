// SPDX-License-Identifier: Apache-2.0
//! `DerivedKeysAbove`: a group key computed from other group keys is
//! computed above the aggregate (P3-A part 2c; design
//! `design/P3-A-2c-derived-group-keys.md` in krishiv-internal). ClickBench
//! q36 groups by `ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3`: one
//! 4-byte key does the same grouping as four.
//!
//! ```text
//! Aggregate[K ∪ {f(K')}; calls](I)   where K' ⊆ K are bare columns, f deterministic
//!   =>  Project[outputs with f(K')](Aggregate[K; calls](I))
//! ```
//!
//! # Why the rows are the same
//!
//! Grouping by `K` and by `K ∪ {f(K')}` gives the same groups exactly when
//! equal `K` values give equal `f(K')`: `f` is deterministic and every column
//! it reads is a key whose grouping equality is identity. Each group's calls
//! see the same input rows, and `f` runs once per group on the value every
//! member row held (a NULL key gives one `f(NULL)` per group as before).
//!
//! # The two shapes
//!
//! A key is derived either as an expression of the aggregate (`GROUP BY c,
//! c - 1`: the analyzer leaves the expression in the key list and the outputs
//! name it again), or as a bare column the `Project` just below defines
//! (`SELECT c, d FROM (SELECT c, c - 1 AS d ..) GROUP BY c, d`). In the second
//! the column `d` leaves the keys, the project below is left to pruning, and
//! the `Project` above defines `d` again from `c`.
//!
//! # When it applies (each refusal is noted for `EXPLAIN VERBOSE`)
//!
//! - `Grouping::GroupBy` only (no sets, rollup, cube);
//! - the key reads only bare grouping columns of the aggregate that the
//!   `Project` below passes through, calls only built-in functions, is
//!   deterministic and holds no subquery or outer reference, and at least
//!   one bare key stays;
//! - every column it reads is of a type whose grouping equality is identity
//!   (integers, decimals, dates, timestamps, strings and binary of the
//!   binary collation; no floating point, whose grouping merges `-0.0` with
//!   `0.0` and every NaN, and no collation);
//! - errors: `f` now runs after the calls, once per group, so it must not
//!   raise. It is a column (or a widening integer cast of one) or integer
//!   `+`/`-` of key columns and integer literals: under ANSI only when the
//!   columns' ranges from statistics (never a type's range) keep every
//!   intermediate result strictly inside the result type, with
//!   `spark.sql.ansi.enabled` off integer `+`/`-` wrap and need no proof;
//! - an output that holds an aggregate call and also reads a moved key outside
//!   a call is not rewritten, nor is a derived column a call reads (the
//!   project below would keep defining it next to the new definition);
//! - several derived expressions as keys move together: a derived key
//!   expression that is refused keeps the whole aggregate as it was (its
//!   outputs name it again, which the aggregate would have to carry). A
//!   derived column key that is refused only stays a key.
//!
//! A derived key over a column the project below defines and that is refused (so it stays a key)
//! is not a base key: the whole aggregate is kept (a follow-up).
//!
//! Batch plans only: the session's batch planning is the optimizer's only
//! production caller and streaming plans never pass through `optimize*`; a
//! streaming caller must switch the rule off (`Options::derived_keys_above`).
//! It reads statistics only and runs once, before the rules that read the
//! aggregate's shape; the result is a fixed point (a second run finds only
//! bare keys).

use crate::expr::AttrMap;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Grouping, LogicalPlan,
    PlanNode, ResolvedExpr,
};
use krishiv_types::DataType;

use crate::dag;
use crate::estimate::Estimator;
use crate::expr::{
    Attrs, builtin, has_aggregate, has_outer_reference, has_subquery, is_deterministic, output_of,
    references, same, substitute,
};
use crate::restrict::{cannot_raise, key_type};
use crate::stats::{RelStats, Statistics, number};

/// Moves derived group keys above their aggregates, entering subquery plans.
/// Also returns one line per aggregate with a derived key: what moved, and
/// per refused key the gate that refused it (`EXPLAIN VERBOSE`).
///
/// `ansi` is the session's `spark.sql.ansi.enabled`: with it, `+` and `-` of
/// integers raise on overflow and need a range proof; without it they wrap.
pub(crate) fn rewrite(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
    ansi: bool,
) -> Result<(Arc<LogicalPlan>, Vec<String>), KrishivError> {
    let mut cx = Context {
        est: Estimator::with_propagation(stats, propagate),
        ansi,
        notes: Vec::new(),
    };
    let out = dag::rewrite_up(root, &mut |p| Ok(at_aggregate(p, &mut cx)))?;
    Ok((out, cx.notes))
}

/// What one pass carries between aggregates.
struct Context<'a> {
    est: Estimator<'a>,
    ansi: bool,
    /// `EXPLAIN VERBOSE` lines, each once.
    notes: Vec<String>,
}

impl Context<'_> {
    fn note(&mut self, line: String) {
        if !self.notes.contains(&line) {
            self.notes.push(line);
        }
    }
}

/// A group key that moves above its aggregate.
struct Moved {
    /// Its position among the aggregate's keys.
    index: usize,
    /// The expression computed above the aggregate.
    expr: Arc<Expr>,
    /// The column the project below defined it as (the second shape).
    column: Option<BoundColumn>,
}

/// `p` with its derived keys above it, or `p`.
fn at_aggregate(p: Arc<LogicalPlan>, cx: &mut Context<'_>) -> Arc<LogicalPlan> {
    let PlanNode::Aggregate {
        input,
        grouping,
        aggregates,
    } = &p.node
    else {
        return p;
    };
    // The analyzer expands ROLLUP, CUBE and GROUPING SETS into an Expand under
    // a plain GROUP BY, so only that form reaches here.
    let Grouping::GroupBy(keys) = grouping else {
        return p;
    };
    // The columns the project below defines (they leave the keys when they
    // move) and the bare keys it merely passes through.
    let defs: AttrMap<Arc<Expr>> = match &input.node {
        PlanNode::Project { exprs, .. } => exprs
            .iter()
            .filter_map(|e| match &e.node {
                ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
                    Some((output.id, Arc::clone(expr)))
                }
                _ => None,
            })
            .collect(),
        _ => AttrMap::default(),
    };
    let base: Attrs = keys
        .iter()
        .filter_map(|k| column_of(k))
        .filter(|c| !defs.contains_key(&c.id))
        .map(|c| c.id)
        .collect();
    let mut candidates = Vec::new();
    for (index, k) in keys.iter().enumerate() {
        match column_of(k) {
            Some(c) => {
                if let Some(f) = defs.get(&c.id) {
                    candidates.push(Moved {
                        index,
                        expr: Arc::clone(f),
                        column: Some(c.clone()),
                    });
                }
            }
            None => candidates.push(Moved {
                index,
                expr: Arc::clone(k),
                column: None,
            }),
        }
    }
    if candidates.is_empty() {
        return p;
    }
    let rel = cx.est.plan(input);
    let mut moved = Vec::new();
    for c in candidates {
        match gate(&c.expr, &base, &rel, cx.ansi) {
            Ok(()) => moved.push(c),
            Err(why) => {
                cx.note(format!(
                    "kept group key {}: {why}",
                    c.column
                        .as_ref()
                        .map_or_else(|| c.expr.to_string(), |c| c.name.clone())
                ));
                if c.column.is_none() {
                    return p;
                }
            }
        }
    }
    if moved.is_empty() {
        return p;
    }
    let calls_read: Attrs = aggregates
        .iter()
        .filter(|o| has_aggregate(o))
        .flat_map(|o| references(o))
        .collect();
    let moved_columns: Attrs = moved
        .iter()
        .filter_map(|m| m.column.as_ref().map(|c| c.id))
        .collect();
    let moved_exprs: Vec<Arc<Expr>> = moved
        .iter()
        .filter(|m| m.column.is_none())
        .map(|m| Arc::clone(&m.expr))
        .collect();
    if let Some(c) = moved
        .iter()
        .filter_map(|m| m.column.as_ref())
        .find(|c| calls_read.contains(&c.id))
    {
        cx.note(format!(
            "kept group key {}: an aggregate call reads it",
            c.name
        ));
        return p;
    }
    let substitutions: AttrMap<Arc<Expr>> = moved
        .iter()
        .filter_map(|m| m.column.as_ref().map(|c| (c.id, Arc::clone(&m.expr))))
        .collect();
    // The new aggregate: the keys that stay (as columns) and every output
    // that holds a call; the project above restores the outputs.
    let mut outputs: Vec<Arc<Expr>> = Vec::new();
    let mut seen = Attrs::default();
    let kept: Vec<Arc<Expr>> = keys
        .iter()
        .enumerate()
        .filter(|(i, _)| !moved.iter().any(|m| m.index == *i))
        .map(|(_, k)| Arc::clone(k))
        .collect();
    for k in &kept {
        match column_of(k) {
            Some(c) => {
                if seen.insert(c.id) {
                    outputs.push(Arc::clone(k));
                }
            }
            None => return p,
        }
    }
    if outputs.is_empty() {
        cx.note("kept the group keys: no bare key would stay".to_owned());
        return p;
    }
    let mut above = Vec::with_capacity(aggregates.len());
    for o in aggregates {
        let Some(out) = output_of(o) else {
            cx.note("kept the group keys: an output names no attribute".to_owned());
            return p;
        };
        if has_subquery(o) || has_outer_reference(o) {
            cx.note("kept the group keys: an output holds a subquery".to_owned());
            return p;
        }
        if has_aggregate(o) {
            if mentions(o, &moved_columns, &moved_exprs) {
                cx.note(format!(
                    "kept the group keys: output {} reads a moved key beside an aggregate call",
                    out.name
                ));
                return p;
            }
            outputs.push(Arc::clone(o));
            above.push(crate::expr::column(out));
            continue;
        }
        above.push(match &o.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) if substitutions.contains_key(&c.id) => {
                alias(
                    substitutions
                        .get(&c.id)
                        .map_or_else(|| Arc::clone(o), Arc::clone),
                    c,
                    o,
                )
            }
            ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
                alias(substitute(expr, &substitutions), output, o)
            }
            _ => substitute(o, &substitutions),
        });
    }
    for m in &moved {
        cx.note(format!(
            "moved group key {} above the aggregate",
            m.column.as_ref().map_or_else(
                || m.expr.to_string(),
                |c| format!("{} = {}", c.name, m.expr)
            )
        ));
    }
    let aggregate = Arc::new(
        LogicalPlan::new(PlanNode::Aggregate {
            input: Arc::clone(input),
            grouping: Grouping::GroupBy(kept),
            aggregates: outputs,
        })
        .with_origin(p.origin),
    );
    Arc::new(
        LogicalPlan::new(PlanNode::Project {
            input: aggregate,
            exprs: above,
        })
        .with_origin(p.origin),
    )
}

/// `expr AS output`, an output entry `like` names.
fn alias(expr: Arc<Expr>, output: &BoundColumn, like: &Arc<Expr>) -> Arc<Expr> {
    Arc::new(Expr {
        node: ExprNode::Resolved(ResolvedExpr::Alias {
            expr,
            output: output.clone(),
        }),
        origin: like.origin,
    })
}

fn column_of(e: &Expr) -> Option<&BoundColumn> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c),
        _ => None,
    }
}

/// Whether `e` reads one of `columns` or holds one of `exprs` outside an
/// aggregate call.
fn mentions(e: &Arc<Expr>, columns: &Attrs, exprs: &[Arc<Expr>]) -> bool {
    if matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate)
    {
        return false;
    }
    if column_of(e).is_some_and(|c| columns.contains(&c.id)) {
        return true;
    }
    if exprs.iter().any(|m| same(e, m)) {
        return true;
    }
    let mut hit = false;
    e.for_each_child(&mut |c| hit |= mentions(c, columns, exprs));
    hit
}

/// Why `e` may not be computed above the aggregate, if it may not.
fn gate(e: &Arc<Expr>, base: &Attrs, rel: &RelStats, ansi: bool) -> Result<(), String> {
    if has_subquery(e) || has_outer_reference(e) {
        return Err("it holds a subquery or an outer reference".to_owned());
    }
    if !is_deterministic(e) {
        return Err("it is not deterministic".to_owned());
    }
    let mut foreign = false;
    let mut columns = Vec::new();
    walk(e, &mut |x| match &x.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            foreign |= !matches!(f.id, FunctionId::Builtin(_));
        }
        ExprNode::Resolved(ResolvedExpr::Column(c)) => columns.push(c.clone()),
        _ => {}
    });
    if foreign {
        return Err("it reads a function that is not a built-in operator".to_owned());
    }
    for c in &columns {
        if !base.contains(&c.id) {
            return Err(format!(
                "it reads {}, which is not a bare group key of the aggregate",
                c.name
            ));
        }
        if !(key_type(&c.data_type) && !c.data_type.contains_collation()) {
            return Err(format!(
                "it reads {} of type {}, whose grouping is not identity",
                c.name,
                c.data_type.simple_string()
            ));
        }
    }
    if cannot_raise(e) {
        return Ok(());
    }
    bound(e, ansi.then_some(rel)).map(|_| ())
}

fn walk(e: &Arc<Expr>, f: &mut impl FnMut(&Arc<Expr>)) {
    f(e);
    e.for_each_child(&mut |c| walk(c, f));
}

/// The largest magnitude of `e`, an integer `+`/`-` tree of columns and
/// literals, where no intermediate result leaves its type; `rel` is the
/// ranges to prove it from (statistics only: a column without a minimum and
/// a maximum is not proven), `None` where `+` and `-` wrap and need none.
fn bound(e: &Arc<Expr>, rel: Option<&RelStats>) -> Result<f64, String> {
    match &e.node {
        ExprNode::Literal(l) => match l {
            krishiv_plan::Literal::Byte(_)
            | krishiv_plan::Literal::Short(_)
            | krishiv_plan::Literal::Integer(_)
            | krishiv_plan::Literal::Long(_) => number(l)
                .map(f64::abs)
                .ok_or_else(|| "it holds a literal of unknown size".to_owned()),
            _ => Err("it may raise (a literal that is not an integer)".to_owned()),
        },
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            if limit(&c.data_type).is_none() {
                return Err(format!("it may raise ({} is not an integer)", c.name));
            }
            match rel {
                None => Ok(0.0),
                Some(rel) => match rel.cols.get(&c.id) {
                    // A `count`'s maximum is the estimated input rows: no proof.
                    Some(s) if s.range_estimated => Err(format!(
                        "it may overflow: the range of {} is an estimate",
                        c.name
                    )),
                    s => s
                        .and_then(|s| Some(s.min?.abs().max(s.max?.abs())))
                        .filter(|m| m.is_finite())
                        .ok_or_else(|| {
                            format!(
                                "it may overflow: no minimum and maximum statistics for {}",
                                c.name
                            )
                        }),
                },
            }
        }
        ExprNode::Cast { expr, .. } if cannot_raise(e) => bound(expr, rel),
        _ => {
            let Some((Builtin::Add | Builtin::Subtract, [a, b], f)) = builtin(e) else {
                return Err("it may raise".to_owned());
            };
            let Some(l) = limit(&f.data_type) else {
                return Err("it may raise (the result is not an integer)".to_owned());
            };
            let m = bound(a, rel)? + bound(b, rel)?;
            if rel.is_some() && (m.is_nan() || m >= l * (1.0 - 1e-9)) {
                return Err(format!("it may overflow: |operands| <= {m:e}"));
            }
            Ok(m)
        }
    }
}

/// One past the largest magnitude an integer of type `t` holds.
fn limit(t: &DataType) -> Option<f64> {
    Some(match t {
        DataType::Byte => 128.0,
        DataType::Short => 32_768.0,
        DataType::Integer => 2_147_483_648.0,
        DataType::Long => 9_223_372_036_854_775_808.0,
        _ => return None,
    })
}
