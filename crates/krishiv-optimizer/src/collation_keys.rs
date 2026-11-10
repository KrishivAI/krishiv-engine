// SPDX-License-Identifier: Apache-2.0
//! The planner side of collations (ADR 0043, design `P2-collations.md` section 3.3, unit 3b):
//! every operator that compares values by bytes compares `__collation_key(c)` instead, and
//! carries the original `c` through. The key lives inside the operator's key expressions, so no
//! key column reaches an output schema; a second run finds nothing to do (a key is `BINARY`).
//!
//! - **Aggregates** group by the key and output the first value of each group (Spark's
//!   `RewriteGroupByCollation`, `First(c)`).
//! - **`min`/`max`** of a collated string take the key as a second argument, which the executor
//!   runs as a by-key extremum that keeps the first of equal keys (as Spark's `Max` keeps its
//!   buffer on a tie; `max_by`, which takes the last, keeps its own form with a keyed ordering).
//! - **`count(DISTINCT c)`**, `approx_count_distinct` count keys; the other `DISTINCT` aggregates
//!   first reduce the input to one row per key.
//! - **Sorts, windows and `DISTRIBUTE BY`** use the key; a hash over it is Spark's
//!   `CollationAwareMurmur3Hash`.
//!
//! Joins and scalar comparisons are keyed by the analyzer. What has no key (ICU) never gets here:
//! the analyzer's guard refuses it.

use std::sync::Arc;

use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_functions::registry::{Lookup, builtins};
use krishiv_functions::signature::SparkProfile;
use krishiv_plan::{
    BoundColumn, Expr, ExprNode, FunctionId, FunctionKind, Grouping, Literal, LogicalPlan,
    PlanNode, ResolvedExpr, ResolvedFunction, SortOrder, WindowExpr, WindowSpec, WindowSpecRef,
};
use krishiv_types::DataType;

use crate::dag;
use crate::expr::same;
use crate::restrict::max_id;

type Res<T> = Result<T, KrishivError>;

fn registry(name: &str) -> Res<FunctionId> {
    let profile = SparkProfile::ALL.into_iter().max();
    match profile.map(|p| builtins().lookup(name, p)) {
        Some(Lookup::Found(def)) => Ok(FunctionId::Registry {
            name: def.name.to_owned(),
            id: def.id,
        }),
        _ => Err(InternalError {
            message: format!("{name} is not registered"),
        }
        .into()),
    }
}

/// The type of `e`, when it is a kind the analyzer's guard can see.
fn data_type(e: &Expr) -> Option<DataType> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Field { data_type, .. }) => Some(data_type.clone()),
        _ => crate::expr::data_type(e),
    }
}

fn collated(e: &Expr) -> bool {
    data_type(e).is_some_and(|t| t.contains_collation())
}

fn collated_string(e: &Expr) -> bool {
    data_type(e).is_some_and(|t| t.is_collated_string())
}

fn resolved(f: ResolvedFunction, origin: krishiv_plan::Origin) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))).with_origin(origin))
}

/// `__collation_key(e, 'type of e')`.
fn key(e: &Arc<Expr>) -> Res<Arc<Expr>> {
    let Some(t) = data_type(e) else {
        return Ok(Arc::clone(e));
    };
    let name = Arc::new(
        Expr::new(ExprNode::Literal(Literal::String(t.catalog_string()))).with_origin(e.origin),
    );
    Ok(resolved(
        ResolvedFunction {
            id: registry("__collation_key")?,
            kind: FunctionKind::Scalar,
            args: vec![Arc::clone(e), name],
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: t.collation_key_type(),
            nullable: true,
            deterministic: true,
        },
        e.origin,
    ))
}

fn key_if_collated(e: &Arc<Expr>) -> Res<Arc<Expr>> {
    if collated(e) {
        key(e)
    } else {
        Ok(Arc::clone(e))
    }
}

/// `any_value(e)`: the first value of the group, as Spark's `First`.
fn first(e: &Arc<Expr>, t: DataType) -> Res<Arc<Expr>> {
    Ok(resolved(
        ResolvedFunction {
            id: registry("any_value")?,
            kind: FunctionKind::Aggregate,
            args: vec![Arc::clone(e)],
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: t,
            nullable: true,
            deterministic: true,
        },
        e.origin,
    ))
}

fn unsupported(what: &str, origin: krishiv_plan::Origin) -> KrishivError {
    let _ = origin;
    KrishivUnsupportedFeature {
        feature: what.to_owned(),
    }
    .into()
}

/// A `DISTINCT` aggregate whose values must survive (`listagg`, `collect_list`), so a key cannot
/// stand in for its argument.
fn distinct_keeping_values(f: &ResolvedFunction) -> bool {
    f.kind == FunctionKind::Aggregate
        && f.distinct
        && !matches!(f.id.name(), "count" | "approx_count_distinct")
        && f.args.first().is_some_and(|a| collated(peeled(a)))
}

/// `listagg` casts its value to a plain string: the collation that dedupes it is the child's.
fn peeled(e: &Arc<Expr>) -> &Arc<Expr> {
    match &e.node {
        ExprNode::Cast {
            expr,
            to: DataType::String,
            ..
        } if collated(expr) => expr,
        _ => e,
    }
}

/// The call `f` with its collated comparing arguments on keys.
fn keyed_call(f: &ResolvedFunction, origin: krishiv_plan::Origin) -> Res<Option<Arc<Expr>>> {
    if f.kind != FunctionKind::Aggregate {
        return Ok(None);
    }
    let with = |args: Vec<Arc<Expr>>| resolved(ResolvedFunction { args, ..f.clone() }, origin);
    let name = f.id.name();
    Ok(match name {
        "min" | "max" if f.args.len() == 1 && collated_string(&f.args[0]) => {
            Some(with(vec![Arc::clone(&f.args[0]), key(&f.args[0])?]))
        }
        "min_by" | "max_by" if f.args.len() == 2 && collated(&f.args[1]) => {
            Some(with(vec![Arc::clone(&f.args[0]), key(&f.args[1])?]))
        }
        "approx_count_distinct" if f.args.first().is_some_and(|a| collated(a)) => {
            let mut args = f.args.clone();
            args[0] = key(&args[0])?;
            Some(with(args))
        }
        "count" if f.distinct && f.args.iter().any(|a| collated(a)) => {
            let args = f
                .args
                .iter()
                .map(key_if_collated)
                .collect::<Res<Vec<_>>>()?;
            Some(with(args))
        }
        // The ordering keys follow the value and the delimiter.
        "listagg" if f.args.iter().skip(2).any(|a| collated(a)) => {
            let mut args = f.args.clone();
            for a in args.iter_mut().skip(2) {
                *a = key_if_collated(a)?;
            }
            Some(with(args))
        }
        _ => None,
    })
}

/// `e` with each group key outside an aggregate call read as the group's first value, and the
/// aggregate calls on keys.
fn output(e: &Arc<Expr>, keys: &[Arc<Expr>]) -> Res<Arc<Expr>> {
    if keys.iter().any(|k| same(e, k))
        && let Some(t) = data_type(e)
    {
        return first(e, t);
    }
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
        && f.kind == FunctionKind::Aggregate
    {
        return Ok(keyed_call(f, e.origin)?.unwrap_or_else(|| Arc::clone(e)));
    }
    Arc::clone(e).map_children(|c| output(&c, keys))
}

/// A select item of the aggregate: a bare group key keeps its attribute under an alias.
fn item(e: &Arc<Expr>, keys: &[Arc<Expr>]) -> Res<Arc<Expr>> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, output: col }) => Ok(Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                expr: output(expr, keys)?,
                output: col.clone(),
            }))
            .with_origin(e.origin),
        )),
        ExprNode::Resolved(ResolvedExpr::Column(col)) if keys.iter().any(|k| same(e, k)) => {
            Ok(Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                    expr: first(e, col.data_type.clone())?,
                    output: col.clone(),
                }))
                .with_origin(e.origin),
            ))
        }
        _ => output(e, keys),
    }
}

/// Every aggregate call under `e`.
fn calls<'a>(e: &'a Expr, out: &mut Vec<&'a ResolvedFunction>) {
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
        && f.kind == FunctionKind::Aggregate
    {
        out.push(f);
        return;
    }
    e.for_each_child(&mut |c| calls(c, out));
}

fn alias(expr: Arc<Expr>, output: BoundColumn) -> Arc<Expr> {
    let origin = expr.origin;
    Arc::new(
        Expr::new(ExprNode::Resolved(ResolvedExpr::Alias { expr, output })).with_origin(origin),
    )
}

fn column(c: &BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
        c.clone(),
    ))))
}

/// Replaces every occurrence of `from` in `e`.
fn substitute(e: &Arc<Expr>, from: &Arc<Expr>, to: &Arc<Expr>) -> Res<Arc<Expr>> {
    if same(e, from) {
        return Ok(Arc::clone(to));
    }
    Arc::clone(e).map_children(|c| substitute(&c, from, to))
}

/// A `DISTINCT` aggregate over a collated value as two aggregates: one row per group and key
/// (the first value of each key), then the call without `DISTINCT` over those rows.
fn distinct_over_keys(
    p: &Arc<LogicalPlan>,
    input: &Arc<LogicalPlan>,
    keys: &[Arc<Expr>],
    aggregates: &[Arc<Expr>],
    next: &mut u64,
) -> Res<Option<Arc<LogicalPlan>>> {
    let mut found = Vec::new();
    aggregates.iter().for_each(|a| calls(a, &mut found));
    if !found.iter().any(|f| distinct_keeping_values(f)) {
        return Ok(None);
    }
    let [call] = found.as_slice() else {
        return Err(unsupported(
            "a DISTINCT aggregate of a collated string beside another aggregate",
            p.origin,
        ));
    };
    let value = Arc::clone(&call.args[0]);
    let orders_are_value = call
        .args
        .iter()
        .skip(2)
        .all(|a| same(a, &value) || same(a, peeled(&value)));
    if call.filter.is_some() || !orders_are_value {
        return Err(unsupported(
            "a DISTINCT aggregate of a collated string with a filter or another ordering",
            p.origin,
        ));
    }
    let mut fresh = |name: &str, e: &Arc<Expr>| -> Res<BoundColumn> {
        let data_type = data_type(e).ok_or_else(|| unsupported("an untyped key", p.origin))?;
        *next += 1;
        Ok(BoundColumn {
            id: krishiv_plan::AttrId(*next - 1),
            name: name.to_owned(),
            data_type,
            nullable: true,
        })
    };
    let mut items = Vec::with_capacity(keys.len() + 1);
    let mut outer_keys = Vec::with_capacity(keys.len());
    let mut pairs = Vec::with_capacity(keys.len() + 1);
    for k in keys {
        let col = fresh("k", k)?;
        items.push(alias(first(k, col.data_type.clone())?, col.clone()));
        let read = column(&col);
        outer_keys.push(Arc::clone(&read));
        pairs.push((Arc::clone(k), read));
    }
    let val = fresh("v", &value)?;
    items.push(alias(first(&value, val.data_type.clone())?, val.clone()));
    // The ordering keys read the bare child when the value is its cast.
    let rep = column(&val);
    pairs.push((Arc::clone(&value), Arc::clone(&rep)));
    if !same(peeled(&value), &value) {
        pairs.push((Arc::clone(peeled(&value)), rep));
    }
    let mut grouping = keys.iter().map(key_if_collated).collect::<Res<Vec<_>>>()?;
    grouping.push(key(peeled(&value))?);
    let mut inner = LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::clone(input),
        grouping: Grouping::GroupBy(grouping),
        aggregates: items,
    });
    inner.origin = p.origin;
    let mut rewritten = Vec::with_capacity(aggregates.len());
    for a in aggregates {
        let mut e = plain_call(a);
        for (from, to) in &pairs {
            e = substitute(&e, from, to)?;
        }
        rewritten.push(e);
    }
    let mut outer = LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::new(inner),
        grouping: Grouping::GroupBy(outer_keys),
        aggregates: rewritten,
    });
    outer.origin = p.origin;
    aggregate(&Arc::new(outer), next).map(Some)
}

/// `e` with the `DISTINCT` of its aggregate call removed.
fn plain_call(e: &Arc<Expr>) -> Arc<Expr> {
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
        && f.kind == FunctionKind::Aggregate
    {
        return resolved(
            ResolvedFunction {
                distinct: false,
                ..f.clone()
            },
            e.origin,
        );
    }
    match Arc::clone(e).map_children(|c| Ok::<_, KrishivError>(plain_call(&c))) {
        Ok(out) => out,
        Err(_) => Arc::clone(e),
    }
}

fn aggregate(p: &Arc<LogicalPlan>, next: &mut u64) -> Res<Arc<LogicalPlan>> {
    let PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(keys),
        aggregates,
    } = &p.node
    else {
        return Ok(Arc::clone(p));
    };
    if let Some(done) = distinct_over_keys(p, input, keys, aggregates, next)? {
        return Ok(done);
    }
    let held: Vec<Arc<Expr>> = keys.iter().filter(|k| collated(k)).cloned().collect();
    let mut found = Vec::new();
    aggregates.iter().for_each(|a| calls(a, &mut found));
    let mut changes = !held.is_empty();
    for f in found {
        changes |= keyed_call(f, p.origin)?.is_some();
    }
    if !changes {
        return Ok(Arc::clone(p));
    }
    let grouping = keys.iter().map(key_if_collated).collect::<Res<Vec<_>>>()?;
    let aggregates = aggregates
        .iter()
        .map(|a| item(a, &held))
        .collect::<Res<Vec<_>>>()?;
    let mut out = LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::clone(input),
        grouping: Grouping::GroupBy(grouping),
        aggregates,
    });
    out.origin = p.origin;
    Ok(Arc::new(out))
}

fn keyed_order(order: &[SortOrder]) -> Res<Vec<SortOrder>> {
    order
        .iter()
        .map(|o| {
            Ok(SortOrder {
                expr: key_if_collated(&o.expr)?,
                ..o.clone()
            })
        })
        .collect()
}

/// A window expression with its window's keys and its aggregate's arguments keyed.
fn window_item(e: &Arc<Expr>) -> Res<Arc<Expr>> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
            Ok(alias(window_item(expr)?, output.clone()))
        }
        ExprNode::Window(w) => {
            let WindowSpecRef::Spec(spec) = &w.window else {
                return Ok(Arc::clone(e));
            };
            let function = match &w.function.node {
                ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                    keyed_call(f, w.function.origin)?.unwrap_or_else(|| Arc::clone(&w.function))
                }
                _ => Arc::clone(&w.function),
            };
            let window = WindowSpecRef::Spec(WindowSpec {
                partition_by: spec
                    .partition_by
                    .iter()
                    .map(key_if_collated)
                    .collect::<Res<Vec<_>>>()?,
                order_by: keyed_order(&spec.order_by)?,
                frame: spec.frame.clone(),
            });
            Ok(Arc::new(
                Expr::new(ExprNode::Window(WindowExpr { function, window })).with_origin(e.origin),
            ))
        }
        _ => Ok(Arc::clone(e)),
    }
}

fn step(p: Arc<LogicalPlan>, next: &mut u64) -> Res<Arc<LogicalPlan>> {
    let node = match &p.node {
        PlanNode::Aggregate { .. } => return aggregate(&p, next),
        PlanNode::Sort {
            input,
            order,
            global,
        } if order.iter().any(|o| collated(&o.expr)) => PlanNode::Sort {
            input: Arc::clone(input),
            order: keyed_order(order)?,
            global: *global,
        },
        PlanNode::Repartition { input, exprs } if exprs.iter().any(|e| collated(e)) => {
            PlanNode::Repartition {
                input: Arc::clone(input),
                exprs: exprs.iter().map(key_if_collated).collect::<Res<Vec<_>>>()?,
            }
        }
        PlanNode::Window {
            input,
            exprs,
            partition_by,
            order_by,
        } if partition_by.iter().any(|e| collated(e))
            || order_by.iter().any(|o| collated(&o.expr))
            || exprs.iter().any(|e| window_keys_a_call(e)) =>
        {
            PlanNode::Window {
                input: Arc::clone(input),
                exprs: exprs.iter().map(window_item).collect::<Res<Vec<_>>>()?,
                partition_by: partition_by
                    .iter()
                    .map(key_if_collated)
                    .collect::<Res<Vec<_>>>()?,
                order_by: keyed_order(order_by)?,
            }
        }
        _ => return Ok(p),
    };
    let mut out = LogicalPlan::new(node);
    out.origin = p.origin;
    Ok(Arc::new(out))
}

/// Whether a window expression's function has a comparing argument to key.
fn window_keys_a_call(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => window_keys_a_call(expr),
        ExprNode::Window(w) => matches!(
            &w.function.node,
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if matches!(keyed_call(f, w.function.origin), Ok(Some(_)))
        ),
        _ => false,
    }
}

/// Runs the rewrite over every relation of `root`, subquery plans included.
pub(crate) fn rewrite(root: Arc<LogicalPlan>) -> Res<Arc<LogicalPlan>> {
    let mut next = None;
    let top = Arc::clone(&root);
    dag::rewrite_up(root, &mut |p| {
        let free =
            next.get_or_insert_with(|| max_id(&top, &mut Default::default()).saturating_add(1));
        step(p, free)
    })
}
