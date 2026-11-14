// SPDX-License-Identifier: Apache-2.0
//! Generators (`explode`, `posexplode`, `inline` and their `_outer` forms) in DataFrame select
//! lists, as the SQL front end lowers them: each top-level generator call of a `select`,
//! `selectExpr` or `withColumn` becomes a [`PlanNode::Generate`] under the projection, and the
//! select item reads the generated columns through the reserved qualifier. Several generators
//! chain (a cross product); a generator nested in another expression is left in place for the
//! analyzer to refuse (`UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS`).

use std::sync::Arc;

use krishiv_plan::{
    Expr, ExprNode, GeneratorKind, LogicalPlan, Origin, PlanNode, Star, generate_qualifier,
};

use super::DataFrame;

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

/// Whether the select item is, or aliases, a call of an aggregate function: a list with one is
/// an aggregation, whose generators the analyzer extracts itself.
fn is_aggregate_item(item: &Expr) -> bool {
    let inner = match &item.node {
        ExprNode::Alias { expr, .. } | ExprNode::MultiAlias { expr, .. } => expr.as_ref(),
        _ => item,
    };
    matches!(&inner.node, ExprNode::Function(call)
        if krishiv_plan::analyzer::BuiltinFunctions::is_aggregate(&call.name))
}

impl DataFrame {
    /// `Generate` over `input` for `call`, the `index`-th generator of its select list, and the
    /// star reading the columns it adds.
    fn generate(
        &self,
        input: Arc<LogicalPlan>,
        call: Call,
        index: usize,
        origin: Origin,
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
        .with_origin(Origin::plan(self.session.next_plan_id()));
        let star = Expr::new(ExprNode::Star(Star {
            qualifier: vec![qualifier],
            except: Vec::new(),
            replace: Vec::new(),
            position: None,
            regex: None,
            plan_id: None,
        }))
        .with_origin(origin);
        (Arc::new(plan), Arc::new(star))
    }

    /// `SELECT items`, the generator calls among them lifted into `Generate` nodes.
    pub(super) fn select_lifting_generators(&self, items: Vec<Arc<Expr>>) -> Self {
        let mut plan = Arc::clone(&self.plan);
        let mut out = Vec::with_capacity(items.len());
        let lift = !items.iter().any(|i| is_aggregate_item(i));
        for (index, item) in items.into_iter().enumerate() {
            match generator_call(&item).filter(|_| lift) {
                Some(call) => {
                    let (next, star) = self.generate(plan, call, index, item.origin);
                    plan = next;
                    out.push(star);
                }
                None => out.push(item),
            }
        }
        self.derive(PlanNode::Project {
            input: plan,
            exprs: out,
        })
    }

    /// `withColumn(name, value)` for a generator `value`: the columns it generates are named
    /// `name` (a generator of several columns under one name is
    /// `UDTF_ALIAS_NUMBER_MISMATCH`), and replace a column of that name or follow the others.
    pub(super) fn with_column_generator(&self, name: &str, value: &Arc<Expr>) -> Option<Self> {
        // The generated column gets a name of its own, so it replaces no column of `name`'s.
        let generated = format!("__krishiv_generated_{name}");
        let named = Expr::new(ExprNode::Alias {
            expr: Arc::clone(value),
            name: generated.clone(),
        })
        .with_origin(value.origin);
        let call = generator_call(&named)?;
        let qualifier = generate_qualifier(0, true);
        let (plan, _) = self.generate(Arc::clone(&self.plan), call, 0, value.origin);
        let read = Expr::column([qualifier, generated]).with_origin(value.origin);
        let with = LogicalPlan::new(PlanNode::WithColumns {
            input: plan,
            columns: vec![(name.to_owned(), Arc::new(read))],
        })
        .with_origin(Origin::plan(self.session.next_plan_id()));
        // A bare `*` skips the generator's own column, so only the input's and `name` remain.
        let all = Expr::new(ExprNode::Star(Star {
            qualifier: Vec::new(),
            except: Vec::new(),
            replace: Vec::new(),
            position: None,
            regex: None,
            plan_id: None,
        }))
        .with_origin(value.origin);
        Some(self.derive(PlanNode::Project {
            input: Arc::new(with),
            exprs: vec![Arc::new(all)],
        }))
    }
}
