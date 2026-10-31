// SPDX-License-Identifier: Apache-2.0
//! Traversals that never recurse: the depth cap, resolution state, and
//! `Drop`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::mem;
use std::ops::ControlFlow;
use std::sync::Arc;

use krishiv_common::error::{FailedToParseTooComplex, KrishivError};

use crate::{
    CastMode, Cte, Expr, ExprNode, FieldOp, FunctionCall, Grouping, JoinCondition, Literal,
    LogicalPlan, PlanNode, ReadSource, ResolvedExpr, ResolvedFunction, SortOrder, SqlArgs,
    SubqueryKind,
};

/// The deepest nesting the IR accepts, counting every relation and
/// expression node on a path from the root, subquery plans included.
///
/// Every producer (the Spark Connect decoder, SQL lowering, the native
/// DataFrame APIs) checks it with [`check_depth`] before handing a plan on,
/// as [`crate::Plan::try_new`] does; a deeper plan is an error, never a
/// stack overflow. Recursive traversals (the pretty-printer, the canonical
/// encoding, `transform_*`, `PartialEq`) are tested to run on a 2 MiB stack
/// at this depth. Checking, resolution state and `Drop` do not recurse, and
/// take time linear in the distinct nodes even when subtrees are shared.
pub const MAX_DEPTH: usize = 1000;

/// Checks that `plan` nests no deeper than [`MAX_DEPTH`], without
/// recursing.
///
/// # Errors
/// `FAILED_TO_PARSE_TOO_COMPLEX` (SQLSTATE 54001), Spark's error for a
/// statement too complex to process, when it does.
pub fn check_depth(plan: &LogicalPlan) -> Result<(), KrishivError> {
    if too_deep(Node::Plan(plan)) {
        return Err(FailedToParseTooComplex {}.into());
    }
    Ok(())
}

/// A relation or an expression.
#[derive(Clone, Copy)]
pub(crate) enum Node<'a> {
    Plan(&'a LogicalPlan),
    Expr(&'a Expr),
}

impl Node<'_> {
    /// The node's address: shared (`Arc`) subtrees have one address.
    pub(crate) fn key(self) -> *const () {
        match self {
            Node::Plan(p) => std::ptr::from_ref(p).cast(),
            Node::Expr(e) => std::ptr::from_ref(e).cast(),
        }
    }
}

/// Appends `node`'s children: a relation's expressions then its child
/// relations; an expression's subquery plan then its child expressions.
pub(crate) fn children<'a>(node: Node<'a>, out: &mut Vec<Node<'a>>) {
    match node {
        Node::Plan(p) => {
            p.for_each_expr(&mut |e| out.push(Node::Expr(e)));
            p.for_each_child(&mut |c| out.push(Node::Plan(c)));
        }
        Node::Expr(e) => {
            if let ExprNode::Subquery { plan, .. } = &e.node {
                out.push(Node::Plan(plan));
            }
            e.for_each_child(&mut |c| out.push(Node::Expr(c)));
        }
    }
}

/// Visits every distinct node under `root` once, in pre-order, with an
/// explicit stack: a subtree shared through `Arc` is visited once, so the
/// walk is linear in distinct nodes even when a plan is a DAG. Stops at the
/// first `Break`.
pub(crate) fn visit_distinct<'a>(
    root: Node<'a>,
    mut f: impl FnMut(Node<'a>) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let mut seen = HashSet::new();
    let mut stack = vec![root];
    let mut next = Vec::new();
    while let Some(node) = stack.pop() {
        if !seen.insert(node.key()) {
            continue;
        }
        f(node)?;
        children(node, &mut next);
        stack.extend(next.drain(..).rev());
    }
    ControlFlow::Continue(())
}

/// Whether `root` nests deeper than [`MAX_DEPTH`]: its height (the longest
/// path from it down, in nodes) exceeds the cap. Heights are memoised per
/// distinct node, so a shared subtree is measured once however many paths
/// reach it, and at the depth of the deepest one.
pub(crate) fn too_deep(root: Node<'_>) -> bool {
    let mut height: HashMap<*const (), usize> = HashMap::new();
    let mut stack = vec![(root, false)];
    let mut next = Vec::new();
    while let Some((node, expanded)) = stack.pop() {
        if height.contains_key(&node.key()) {
            continue;
        }
        children(node, &mut next);
        if expanded {
            let below = next
                .drain(..)
                .map(|c| height.get(&c.key()).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            if below + 1 > MAX_DEPTH {
                return true;
            }
            height.insert(node.key(), below + 1);
        } else {
            stack.push((node, true));
            stack.extend(next.drain(..).map(|c| (c, false)));
        }
    }
    false
}

/// Whether no unresolved node remains under `root`. A cast in
/// [`CastMode::Default`] is unresolved: the analyzer pins its mode.
pub(crate) fn all_resolved(root: Node<'_>) -> bool {
    visit_distinct(root, |node| {
        let resolved = match node {
            Node::Plan(p) => match &p.node {
                PlanNode::Read(_)
                | PlanNode::Sql { .. }
                | PlanNode::WithColumns { .. }
                | PlanNode::Hint { .. }
                | PlanNode::SetOp { .. }
                | PlanNode::Pivot { .. }
                | PlanNode::Unpivot { .. }
                | PlanNode::InlineTable { .. }
                | PlanNode::TableFunction { .. }
                | PlanNode::Distinct { .. } => false,
                PlanNode::Union { by_name, .. } => !by_name,
                PlanNode::Join { condition, .. } => match condition {
                    JoinCondition::Natural | JoinCondition::Using(_) => false,
                    JoinCondition::None | JoinCondition::On(_) => true,
                },
                PlanNode::Scan(_)
                | PlanNode::Range { .. }
                | PlanNode::OneRowRelation
                | PlanNode::CteRef(_)
                | PlanNode::Shift { .. }
                | PlanNode::Project { .. }
                | PlanNode::Filter { .. }
                | PlanNode::Aggregate { .. }
                | PlanNode::PartialAggregate { .. }
                | PlanNode::Sort { .. }
                | PlanNode::Limit { .. }
                | PlanNode::Offset { .. }
                | PlanNode::SubqueryAlias { .. }
                | PlanNode::WithCte { .. }
                | PlanNode::JoinHint { .. }
                | PlanNode::Expand { .. }
                | PlanNode::Window { .. }
                | PlanNode::ReplicateRows { .. } => true,
            },
            Node::Expr(e) => match &e.node {
                ExprNode::Column(_)
                | ExprNode::Star(_)
                | ExprNode::TypedLiteral { .. }
                | ExprNode::Function(_)
                | ExprNode::Field { .. }
                | ExprNode::UpdateFields { .. }
                | ExprNode::Alias { .. } => false,
                ExprNode::Cast { mode, .. } => match mode {
                    CastMode::Default => false,
                    CastMode::Legacy | CastMode::Ansi | CastMode::Try => true,
                },
                // A window is resolved once its frame is set (children are
                // visited on their own).
                ExprNode::Window(w) => matches!(
                    &w.window,
                    crate::WindowSpecRef::Spec(crate::WindowSpec { frame: Some(_), .. })
                ),
                // A lambda is resolved once its body is (visited on its
                // own); its parameters are plain names.
                ExprNode::Lambda { .. }
                | ExprNode::Literal(_)
                | ExprNode::Subquery { .. }
                | ExprNode::Resolved(_) => true,
            },
        };
        if resolved {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    })
    .is_continue()
}

/// Children detached from dropped nodes, dropped one at a time.
#[derive(Default)]
struct Teardown {
    plans: Vec<Arc<LogicalPlan>>,
    exprs: Vec<Arc<Expr>>,
}

/// A node without children or heap data, swapped in for a dropped node.
const fn empty_plan() -> PlanNode {
    PlanNode::Read(ReadSource::Table {
        name: Vec::new(),
        options: BTreeMap::new(),
    })
}

const fn empty_expr() -> ExprNode {
    ExprNode::Literal(Literal::Boolean(false))
}

/// `Drop` for [`LogicalPlan`]: detaches the children and drops every node
/// this plan owns alone, iteratively.
pub(crate) fn teardown_plan(plan: &mut LogicalPlan) {
    let mut t = Teardown::default();
    t.plan(mem::replace(&mut plan.node, empty_plan()));
    t.run();
}

/// `Drop` for [`Expr`]; see [`teardown_plan`].
pub(crate) fn teardown_expr(expr: &mut Expr) {
    let mut t = Teardown::default();
    t.expr(mem::replace(&mut expr.node, empty_expr()));
    t.run();
}

impl Teardown {
    fn run(&mut self) {
        loop {
            if let Some(p) = self.plans.pop() {
                if let Some(mut p) = Arc::into_inner(p) {
                    self.plan(mem::replace(&mut p.node, empty_plan()));
                }
            } else if let Some(e) = self.exprs.pop() {
                if let Some(mut e) = Arc::into_inner(e) {
                    self.expr(mem::replace(&mut e.node, empty_expr()));
                }
            } else {
                return;
            }
        }
    }

    fn plan(&mut self, node: PlanNode) {
        match node {
            PlanNode::Read(_)
            | PlanNode::Scan(_)
            | PlanNode::Range { .. }
            | PlanNode::CteRef(_)
            | PlanNode::OneRowRelation => {}
            PlanNode::Sql { args, .. } => match args {
                SqlArgs::None => {}
                SqlArgs::Positional(v) => self.exprs.extend(v),
                SqlArgs::Named(m) => self.exprs.extend(m.into_values()),
            },
            PlanNode::Project { input, exprs } => {
                self.plans.push(input);
                self.exprs.extend(exprs);
            }
            PlanNode::WithColumns { input, columns } => {
                self.plans.push(input);
                self.exprs.extend(columns.into_iter().map(|(_, e)| e));
            }
            PlanNode::Filter {
                input,
                predicate: e,
            }
            | PlanNode::Limit { input, limit: e }
            | PlanNode::Offset { input, offset: e }
            | PlanNode::ReplicateRows { input, count: e } => {
                self.plans.push(input);
                self.exprs.push(e);
            }
            PlanNode::Expand {
                input, projections, ..
            } => {
                self.plans.push(input);
                self.exprs.extend(projections.into_iter().flatten());
            }
            PlanNode::Pivot {
                input,
                pivot_columns,
                values,
                aggregates,
            } => {
                self.plans.push(input);
                self.exprs.extend(pivot_columns);
                self.exprs
                    .extend(values.into_iter().flat_map(|v| v.exprs.into_iter()));
                self.exprs.extend(aggregates);
            }
            PlanNode::Unpivot {
                input, column_sets, ..
            } => {
                self.plans.push(input);
                self.exprs
                    .extend(column_sets.into_iter().flat_map(|s| s.columns.into_iter()));
            }
            PlanNode::InlineTable { rows, .. } => {
                self.exprs.extend(rows.into_iter().flatten());
            }
            PlanNode::TableFunction { args, .. } => {
                self.exprs.extend(args);
            }
            PlanNode::PartialAggregate {
                input,
                keys,
                aggregates,
                ..
            } => {
                self.plans.push(input);
                self.exprs.extend(keys);
                self.exprs.extend(aggregates);
            }
            PlanNode::Aggregate {
                input,
                grouping,
                aggregates,
            } => {
                self.plans.push(input);
                match grouping {
                    Grouping::GroupBy(v) | Grouping::Rollup(v) | Grouping::Cube(v) => {
                        self.exprs.extend(v);
                    }
                    Grouping::GroupingSets { exprs, sets } => {
                        self.exprs
                            .extend(exprs.into_iter().chain(sets.into_iter().flatten()));
                    }
                }
                self.exprs.extend(aggregates);
            }
            PlanNode::Join {
                left,
                right,
                condition,
                ..
            } => {
                self.plans.extend([left, right]);
                match condition {
                    JoinCondition::On(e) => self.exprs.push(e),
                    JoinCondition::None | JoinCondition::Using(_) | JoinCondition::Natural => {}
                }
            }
            PlanNode::Sort { input, order, .. } => {
                self.plans.push(input);
                self.exprs
                    .extend(order.into_iter().map(|SortOrder { expr, .. }| expr));
            }
            PlanNode::Window {
                input,
                exprs,
                partition_by,
                order_by,
            } => {
                self.plans.push(input);
                self.exprs.extend(exprs);
                self.exprs.extend(partition_by);
                self.exprs
                    .extend(order_by.into_iter().map(|SortOrder { expr, .. }| expr));
            }
            PlanNode::SubqueryAlias { input, .. }
            | PlanNode::Shift { input, .. }
            | PlanNode::JoinHint { input, .. }
            | PlanNode::Distinct { input } => {
                self.plans.push(input);
            }
            PlanNode::Hint { input, params, .. } => {
                self.plans.push(input);
                self.exprs.extend(params);
            }
            PlanNode::SetOp { left, right, .. } => self.plans.extend([left, right]),
            PlanNode::Union { inputs, .. } => self.plans.extend(inputs),
            PlanNode::WithCte { input, ctes } => {
                self.plans.push(input);
                self.plans
                    .extend(ctes.into_iter().map(|Cte { plan, .. }| plan));
            }
        }
    }

    fn expr(&mut self, node: ExprNode) {
        match node {
            ExprNode::Column(_)
            | ExprNode::Literal(_)
            | ExprNode::TypedLiteral { .. }
            | ExprNode::Star(_)
            | ExprNode::Resolved(
                ResolvedExpr::Column(_) | ResolvedExpr::OuterColumn(_) | ResolvedExpr::LambdaVar(_),
            ) => {}
            ExprNode::Function(FunctionCall {
                args,
                filter,
                within_group,
                ..
            })
            | ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                args,
                filter,
                within_group,
                ..
            })) => {
                self.exprs.extend(args);
                self.exprs.extend(filter);
                self.exprs
                    .extend(within_group.into_iter().map(|SortOrder { expr, .. }| expr));
            }
            ExprNode::Resolved(ResolvedExpr::StateInput { columns, .. }) => {
                self.exprs.extend(columns);
            }
            ExprNode::Cast { expr, .. }
            | ExprNode::Alias { expr, .. }
            | ExprNode::Lambda { body: expr, .. }
            | ExprNode::Field { base: expr, .. }
            | ExprNode::Resolved(
                ResolvedExpr::Alias { expr, .. } | ResolvedExpr::Field { base: expr, .. },
            ) => self.exprs.push(expr),
            ExprNode::UpdateFields { base, ops } => {
                self.exprs.push(base);
                for op in ops {
                    if let FieldOp::With { value, .. } = op {
                        self.exprs.push(value);
                    }
                }
            }
            ExprNode::Subquery { kind, plan } => {
                self.plans.push(plan);
                match kind {
                    SubqueryKind::In(values) => self.exprs.extend(values),
                    SubqueryKind::Scalar | SubqueryKind::Exists => {}
                }
            }
            ExprNode::Window(crate::WindowExpr { function, window }) => {
                self.exprs.push(function);
                if let crate::WindowSpecRef::Spec(spec) = window {
                    self.exprs.extend(spec.partition_by);
                    self.exprs
                        .extend(spec.order_by.into_iter().map(|SortOrder { expr, .. }| expr));
                    if let Some(frame) = spec.frame {
                        for b in [frame.lower, frame.upper] {
                            if let crate::FrameBound::Offset(e) = b {
                                self.exprs.push(e);
                            }
                        }
                    }
                }
            }
        }
    }
}
