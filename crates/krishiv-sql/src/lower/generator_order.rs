// SPDX-License-Identifier: Apache-2.0
//! The order a select list's generators run in. Spark resolves a generator's argument over the
//! select list's lateral aliases, so `explode(arr) AS col, explode(..) AS arr` runs the second
//! generator first: a generator that reads another generator's alias comes after it.

use std::sync::Arc;

use super::{
    Alias, Expr, ExprKind, ExprNode, IrExpr, LogicalPlan, Lowerer, PlanNode, R, SelectItem, Star,
    arc, unsupported,
};

/// Whether `e` reads the unqualified column `name`.
fn reads(e: &Expr, name: &str) -> bool {
    if let ExprKind::Column(parts) = &e.kind
        && let [only] = &parts[..]
    {
        return only.value.eq_ignore_ascii_case(name);
    }
    let mut found = false;
    e.for_each_child(&mut |c| found = found || reads(c, name));
    found
}

/// The positions of `projection`'s generators, in the order their nodes are chained: select-list
/// order, except that a generator reading the single alias of a later one follows it.
pub(super) fn chain(projection: &[SelectItem]) -> Vec<usize> {
    let gens: Vec<usize> = (0..projection.len())
        .filter(|&i| Lowerer::generator_of(&projection[i].expr).is_some())
        .collect();
    let waits_for = |g: usize, other: usize| match (
        Lowerer::generator_of(&projection[g].expr),
        &projection[other].alias,
    ) {
        (Some((.., args, _)), Some(Alias::Name(n))) => {
            args.iter().any(|a| reads(&a.expr, &n.value))
        }
        _ => false,
    };
    let mut order: Vec<usize> = Vec::with_capacity(gens.len());
    while order.len() < gens.len() {
        let next = gens.iter().copied().find(|g| {
            !order.contains(g)
                && gens
                    .iter()
                    .all(|&o| o == *g || order.contains(&o) || !waits_for(*g, o))
        });
        // A cycle has no valid order: the remaining ones keep theirs.
        match next.or_else(|| gens.iter().copied().find(|g| !order.contains(g))) {
            Some(g) => order.push(g),
            None => break,
        }
    }
    order
}

impl Lowerer {
    /// `generator(arg)` over `plan` (a [`PlanNode::Generate`]) and the
    /// select item standing for its columns: a star over the generator's
    /// own qualifier, in the place of the call.
    pub(super) fn select_generator(
        &mut self,
        plan: LogicalPlan,
        item: &SelectItem,
        index: usize,
    ) -> R<(LogicalPlan, Arc<IrExpr>)> {
        let Some((kind, outer, args, name)) = Self::generator_of(&item.expr) else {
            return Err(unsupported("a select item that is no generator", item.span));
        };
        let arg = self.generator_arg(name, args, item.span)?;
        let names = match &item.alias {
            None => Vec::new(),
            Some(Alias::Name(n)) => vec![n.value.clone()],
            Some(Alias::List(l)) => l.iter().map(|n| n.value.clone()).collect(),
        };
        // A reserved qualifier (no user table has it) the generated columns are read through.
        let aliased = matches!(item.alias, Some(Alias::Name(_)));
        let qualifier = krishiv_plan::generate_qualifier(index, aliased);
        let plan = self.node(
            PlanNode::Generate {
                input: arc(plan),
                kind,
                outer,
                arg,
                output: Vec::new(),
                names,
                qualifier: Some(qualifier.clone()),
            },
            item.span,
        );
        let star = self.ex(
            ExprNode::Star(Star {
                regex: None,
                replace: Vec::new(),
                position: None,
                qualifier: vec![qualifier],
                except: Vec::new(),
                plan_id: None,
            }),
            item.span,
        );
        Ok((plan, arc(star)))
    }
}
