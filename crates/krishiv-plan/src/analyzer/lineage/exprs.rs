// SPDX-License-Identifier: Apache-2.0
//! The lineage of a resolved expression (design §3.1, "Expressions"): column references read the walk's
//! map, functions compose their arguments, a UDF is `opaque` over its arguments, a lambda variable reads
//! its higher-order function's other arguments, a scalar subquery is its output's lineage, and `IN` /
//! `EXISTS` subqueries are `join` filters. Memoized by (expression address, `Shift` offset).

use std::collections::BTreeSet;
use std::sync::Arc;

use krishiv_common::error::KrishivError;

use super::walk::{ColInfo, Info, Walker, addr, internal, literal, merge, shifted};
use super::{FilterKind, Leaf, LineageWarning, Transform};
use crate::{
    Builtin, Expr, ExprNode, FunctionId, FunctionKind, ResolvedExpr, ResolvedFunction,
    SubqueryKind, WindowSpecRef,
};

impl Walker<'_> {
    /// The lineage of the value of `e` under `offset`.
    pub(super) fn value(&mut self, e: &Arc<Expr>, offset: u64) -> Result<Info, KrishivError> {
        let key = (addr(&**e), offset);
        if let Some(i) = self.exprs.get(&key) {
            return Ok(Arc::clone(i));
        }
        let info = self.deeper(|w| w.expr(e, offset))?;
        self.remember_expr(key, &info);
        Ok(info)
    }

    /// The rule of `e`'s kind.
    #[deny(clippy::wildcard_enum_match_arm)]
    fn expr(&mut self, e: &Expr, offset: u64) -> Result<Info, KrishivError> {
        match &e.node {
            ExprNode::Literal(_) => Ok(literal()),
            ExprNode::Cast { expr, .. } => {
                let v = self.value(expr, offset)?;
                Ok(over(&v, Transform::Expression))
            }
            ExprNode::Lambda { body, .. } => self.value(body, offset),
            ExprNode::Subquery { kind, plan } => {
                let out = self.subquery(plan, offset)?;
                match kind {
                    SubqueryKind::Scalar => out
                        .first()
                        .cloned()
                        .ok_or_else(|| internal("a scalar subquery without a column")),
                    SubqueryKind::Exists => {
                        for o in &out {
                            self.record(FilterKind::Join, o);
                        }
                        Ok(Arc::new(ColInfo {
                            sources: BTreeSet::new(),
                            transform: Transform::Expression,
                        }))
                    }
                    SubqueryKind::In(values) => {
                        let mut acc = literal();
                        for v in values {
                            let v = self.value(v, offset)?;
                            acc = merge(&acc, &v);
                        }
                        for o in &out {
                            acc = merge(&acc, o);
                        }
                        self.record(FilterKind::Join, &acc);
                        Ok(over(&acc, Transform::Expression))
                    }
                }
            }
            ExprNode::Window(w) => {
                let f = self.value(&w.function, offset)?;
                let WindowSpecRef::Spec(spec) = &w.window else {
                    return Err(internal("an unresolved window"));
                };
                let mut keys = literal();
                for p in &spec.partition_by {
                    let v = self.value(p, offset)?;
                    keys = merge(&keys, &v);
                }
                for o in &spec.order_by {
                    let v = self.value(&o.expr, offset)?;
                    keys = merge(&keys, &v);
                }
                self.record(FilterKind::Window, &keys);
                // A window's value derives from its arguments and its keys (they change `rank()`).
                let transform = if f.transform > Transform::Aggregate {
                    f.transform
                } else {
                    Transform::Window.max(keys.transform)
                };
                Ok(Arc::new(ColInfo {
                    sources: f.sources.union(&keys.sources).cloned().collect(),
                    transform,
                }))
            }
            ExprNode::Resolved(r) => self.resolved(r, offset),
            ExprNode::Column(_)
            | ExprNode::TypedLiteral { .. }
            | ExprNode::Star(_)
            | ExprNode::Function(_)
            | ExprNode::Field { .. }
            | ExprNode::UpdateFields { .. }
            | ExprNode::StoredField { .. }
            | ExprNode::Collate { .. }
            | ExprNode::Alias { .. }
            | ExprNode::Pipe { .. }
            | ExprNode::MultiAlias { .. } => Err(internal("an unresolved expression")),
        }
    }

    /// The rule of a resolved expression's kind.
    #[deny(clippy::wildcard_enum_match_arm)]
    fn resolved(&mut self, r: &ResolvedExpr, offset: u64) -> Result<Info, KrishivError> {
        match r {
            ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c) | ResolvedExpr::LambdaVar(c) => {
                self.column(shifted(c.id, offset)?)
            }
            ResolvedExpr::Function(f) => self.function(f, offset),
            ResolvedExpr::Field { base, name, .. } => {
                let v = self.value(base, offset)?;
                Ok(over(&metadata_field(&v, name), Transform::Expression))
            }
            ResolvedExpr::Alias { expr, .. } | ResolvedExpr::EvalMode { expr, .. } => {
                self.value(expr, offset)
            }
            ResolvedExpr::StateInput { .. } => Err(internal(
                "a partial aggregate's state (optimizer only) in an analyzed plan",
            )),
        }
    }

    /// A call: the sources of its value arguments (a `CASE WHEN` or `IF` condition is a `conditional`
    /// filter instead), a lambda's variables bound to the call's other arguments, and the step of its kind.
    fn function(&mut self, f: &ResolvedFunction, offset: u64) -> Result<Info, KrishivError> {
        // A lambda's variables read the other arguments (the array, map or initial value).
        let mut inputs = literal();
        for a in f.args.iter().filter(|a| !is_lambda(a)) {
            let v = self.value(a, offset)?;
            inputs = merge(&inputs, &v);
        }
        for a in &f.args {
            if let ExprNode::Lambda { params, .. } = &a.node {
                for p in params {
                    self.env.insert(shifted(p.id, offset)?, Arc::clone(&inputs));
                }
            }
        }
        let conditions = conditions(f);
        let mut value = literal();
        let mut all_literal = true;
        let mut strongest = Transform::Literal;
        for (i, a) in f.args.iter().enumerate() {
            let v = self.value(a, offset)?;
            all_literal &= v.transform == Transform::Literal;
            strongest = strongest.max(v.transform);
            if conditions.contains(&i) {
                self.record(FilterKind::Conditional, &v);
            } else {
                value = merge(&value, &v);
            }
        }
        if let Some(filter) = &f.filter {
            let v = self.value(filter, offset)?;
            self.record(FilterKind::Filter, &v);
        }
        let step = match &f.id {
            FunctionId::Udf { .. } | FunctionId::PythonUdf { .. } => {
                self.warnings.insert(LineageWarning::LineageOpaqueUdf);
                Transform::Opaque
            }
            FunctionId::Builtin(_) | FunctionId::Registry { .. } => kind_step(f.kind),
        };
        // A deterministic scalar call of constants is a constant.
        let transform = if step == Transform::Expression && all_literal && f.deterministic {
            Transform::Literal
        } else {
            step.max(strongest)
        };
        Ok(Arc::new(ColInfo {
            sources: value.sources.clone(),
            transform,
        }))
    }
}

/// `v` one step `step` further: a constant stays a constant.
fn over(v: &Info, step: Transform) -> Info {
    if v.transform == Transform::Literal {
        return Arc::clone(v);
    }
    Arc::new(ColInfo {
        sources: v.sources.clone(),
        transform: v.transform.max(step),
    })
}

/// A field `name` of `v`: when `v` is a file scan's `_metadata` column (its leaves are `_metadata.<field>`),
/// only that field's leaves.
fn metadata_field(v: &Info, name: &str) -> Info {
    let column = |l: &Leaf| match l {
        Leaf::Table { column, .. } | Leaf::Path { column, .. } | Leaf::View { column, .. } => {
            column.clone()
        }
    };
    let field = format!("_metadata.{name}");
    let metadata = !v.sources.is_empty()
        && v.sources
            .iter()
            .all(|s| column(&s.leaf).starts_with("_metadata."));
    let kept: BTreeSet<_> = v
        .sources
        .iter()
        .filter(|s| column(&s.leaf) == field)
        .cloned()
        .collect();
    if !metadata || kept.is_empty() {
        return Arc::clone(v);
    }
    Arc::new(ColInfo {
        sources: kept,
        transform: v.transform,
    })
}

/// Whether `e` is a lambda (an argument of a higher-order function).
fn is_lambda(e: &Expr) -> bool {
    matches!(e.node, ExprNode::Lambda { .. })
}

/// The step a built-in of `kind` adds.
fn kind_step(kind: FunctionKind) -> Transform {
    match kind {
        FunctionKind::Aggregate => Transform::Aggregate,
        FunctionKind::Window => Transform::Window,
        FunctionKind::Generator => Transform::Generated,
        // A scalar; `FunctionKind` is another crate's non-exhaustive enum, so a new kind reads as one too.
        _ => Transform::Expression,
    }
}

/// The positions of `f`'s condition arguments: `CASE WHEN`'s (`when(c1, v1, ..., [else])`) and `IF`'s.
fn conditions(f: &ResolvedFunction) -> Vec<usize> {
    match &f.id {
        FunctionId::Builtin(Builtin::When) => (0..f.args.len() / 2).map(|i| 2 * i).collect(),
        FunctionId::Registry { name, .. } if name == "if" => vec![0],
        FunctionId::Builtin(_)
        | FunctionId::Registry { .. }
        | FunctionId::Udf { .. }
        | FunctionId::PythonUdf { .. } => Vec::new(),
    }
}
