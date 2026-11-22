// SPDX-License-Identifier: Apache-2.0
//! Generators (`explode`, `posexplode`, `inline` and their `_outer` forms) in the select lists
//! the DataFrame front ends build (native `select`/`selectExpr`/`withColumn` and the Connect
//! `Project`/`WithColumns` translation): each top-level generator call becomes a
//! [`PlanNode::Generate`] under the projection, the item reading the generated columns through
//! the reserved qualifier, as the SQL front end lowers them. Several generators chain (a cross
//! product). A list that also holds an aggregate call becomes a global [`PlanNode::Aggregate`]
//! instead, whose generators the analyzer extracts (`MULTI_GENERATOR`, `MISSING_GROUP_BY`). A
//! generator nested in another expression stays in place for the analyzer to refuse
//! (`UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS`).

use std::sync::Arc;

use crate::{
    Expr, ExprNode, GeneratorKind, Grouping, LogicalPlan, Origin, PlanNode, Star,
    generate_qualifier,
};

/// A generator call among select items: what it generates and the names written after `AS`.
struct Call {
    kind: GeneratorKind,
    outer: bool,
    arg: Arc<Expr>,
    names: Vec<String>,
    aliased: bool,
}

/// The generator a select item is, looking through `AS name` / `AS (a, b)`.
fn generator_call(item: &Expr) -> Option<Call> {
    let (inner, names, aliased) = match &item.node {
        ExprNode::Alias { expr, name } => (expr.as_ref(), vec![name.clone()], true),
        ExprNode::MultiAlias { expr, names } => (expr.as_ref(), names.clone(), false),
        _ => (item, Vec::new(), false),
    };
    let ExprNode::Function(call) = &inner.node else {
        return None;
    };
    if !call.qualifier.is_empty() || call.distinct || call.filter.is_some() {
        return None;
    }
    let (kind, outer) = GeneratorKind::of_name(&call.name.to_ascii_lowercase())?;
    let [arg] = &call.args[..] else {
        return None;
    };
    Some(Call {
        kind,
        outer,
        arg: Arc::clone(arg),
        names,
        aliased,
    })
}

/// Whether `e` calls a built-in aggregate function outside a window.
fn has_aggregate(e: &Expr) -> bool {
    if matches!(e.node, ExprNode::Window(_)) {
        return false;
    }
    if let ExprNode::Function(call) = &e.node
        && crate::analyzer::BuiltinFunctions::is_aggregate(&call.name)
    {
        return true;
    }
    let mut found = false;
    e.for_each_child(&mut |c| found = found || has_aggregate(c));
    found
}

fn star(qualifier: Vec<String>, origin: Origin) -> Arc<Expr> {
    Arc::new(
        Expr::new(ExprNode::Star(Star {
            qualifier,
            except: Vec::new(),
            replace: Vec::new(),
            position: None,
            regex: None,
            plan_id: None,
        }))
        .with_origin(origin),
    )
}

/// `Generate` over `input` for `call`, the `index`-th generator of its select list, and the star
/// reading the columns it adds.
fn generate(
    input: Arc<LogicalPlan>,
    call: Call,
    index: usize,
    item_origin: Origin,
    plan_origin: Origin,
) -> (Arc<LogicalPlan>, Arc<Expr>) {
    let qualifier = generate_qualifier(index, call.aliased);
    let plan = LogicalPlan::new(PlanNode::Generate {
        input,
        kind: call.kind,
        outer: call.outer,
        arg: call.arg,
        output: Vec::new(),
        names: call.names,
        qualifier: Some(qualifier.clone()),
    })
    .with_origin(plan_origin);
    (Arc::new(plan), star(vec![qualifier], item_origin))
}

/// `SELECT items` over `input`: the node to build, a `Project`, or when the list holds a
/// generator beside an aggregate a global `Aggregate`, or the `Project` over the `Generate`
/// nodes of its top-level generator calls. `origin` gives the origin of each node made below
/// the returned one.
pub fn select_with_generators(
    input: Arc<LogicalPlan>,
    items: Vec<Arc<Expr>>,
    mut origin: impl FnMut() -> Origin,
) -> PlanNode {
    if !items.iter().any(|i| generator_call(i).is_some()) {
        return PlanNode::Project {
            input,
            exprs: items,
        };
    }
    if items.iter().any(|i| has_aggregate(i)) {
        return PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(Vec::new()),
            aggregates: items,
        };
    }
    let mut plan = input;
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        match generator_call(&item) {
            Some(call) => {
                let (next, read) = generate(plan, call, index, item.origin, origin());
                plan = next;
                out.push(read);
            }
            None => out.push(item),
        }
    }
    PlanNode::Project {
        input: plan,
        exprs: out,
    }
}

/// `withColumn(name, value)` over `input` for a generator `value` (any alias it carries is
/// replaced by `name`, so a generator of several columns is `UDTF_ALIAS_NUMBER_MISMATCH`): the
/// node to build, or `None` when `value` is no generator call. The generated column replaces the
/// column called `name`, or follows the others.
pub fn with_column_generator(
    input: &Arc<LogicalPlan>,
    name: &str,
    value: &Arc<Expr>,
    mut origin: impl FnMut() -> Origin,
) -> Option<PlanNode> {
    let mut bare = value;
    while let ExprNode::Alias { expr, .. } | ExprNode::MultiAlias { expr, .. } = &bare.node {
        bare = expr;
    }
    // The generated column gets a name of its own, so it replaces no column of `name`'s.
    let generated = format!("__krishiv_generated_{name}");
    let named = Expr::new(ExprNode::Alias {
        expr: Arc::clone(bare),
        name: generated.clone(),
    })
    .with_origin(value.origin);
    let call = generator_call(&named)?;
    let qualifier = generate_qualifier(0, true);
    let (plan, _) = generate(Arc::clone(input), call, 0, value.origin, origin());
    let read = Expr::column([qualifier, generated]).with_origin(value.origin);
    let with = LogicalPlan::new(PlanNode::WithColumns {
        input: plan,
        columns: vec![(name.to_owned(), Arc::new(read))],
    })
    .with_origin(origin());
    // A bare `*` skips the generator's own column, so only the input's and `name` remain.
    Some(PlanNode::Project {
        input: Arc::new(with),
        exprs: vec![star(Vec::new(), value.origin)],
    })
}
