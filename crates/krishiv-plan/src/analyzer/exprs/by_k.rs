// SPDX-License-Identifier: Apache-2.0
//! `max_by(x, y, k) OVER (..)` and `min_by(x, y, k) OVER (..)`: the `k` values of `x` with the
//! largest (smallest) non-NULL `y` in the window frame, best first. Without `OVER` the call is
//! rewritten after its arguments are bound (`node.rs`); a window needs the rewrite before they
//! are, because the window must wrap the `collect_list` the call becomes:
//! `__subscript(slice(sort_array(collect_list(named_struct('k', y, 'v', x))
//! FILTER (WHERE y IS NOT NULL) OVER (..), asc), 1, k), 'v')`.

use super::*;

impl Resolver<'_> {
    /// The rewritten window expression when `e` is such a call with a constant `k` in `1..=100000`;
    /// any other `k` keeps the plain path, which refuses the call.
    pub(super) fn window_by_k(&self, e: &Expr) -> Option<Arc<Expr>> {
        let ExprNode::Window(w) = &e.node else {
            return None;
        };
        let ExprNode::Function(f) = &w.function.node else {
            return None;
        };
        if !(f.unqualified().is_some() || super::super::bind::is_builtin_qualifier(&f.qualifier))
            || f.distinct
        {
            return None;
        }
        let ascending = if f.name.eq_ignore_ascii_case("min_by") {
            true
        } else if f.name.eq_ignore_ascii_case("max_by") {
            false
        } else {
            return None;
        };
        let [x, y, k] = &f.args[..] else {
            return None;
        };
        let ExprNode::Literal(Literal::Integer(k_value)) = &k.node else {
            return None;
        };
        if !(1..=100_000).contains(k_value) {
            return None;
        }
        let origin = e.origin;
        let lit = |l: Literal| Arc::new(Expr::new(ExprNode::Literal(l)).with_origin(origin));
        let call = |name: &str, args: Vec<Arc<Expr>>, filter: Option<Arc<Expr>>| {
            Arc::new(
                Expr::new(ExprNode::Function(FunctionCall {
                    name: name.to_owned(),
                    qualifier: Vec::new(),
                    args,
                    arg_names: Vec::new(),
                    distinct: false,
                    filter,
                    within_group: Vec::new(),
                }))
                .with_origin(origin),
            )
        };
        let pair = call(
            "named_struct",
            vec![
                lit(Literal::String("k".to_owned())),
                Arc::clone(y),
                lit(Literal::String("v".to_owned())),
                Arc::clone(x),
            ],
            None,
        );
        let present = call("isnotnull", vec![Arc::clone(y)], None);
        let filter = match &f.filter {
            Some(keep) => call("and", vec![present, Arc::clone(keep)], None),
            None => present,
        };
        let collected = Arc::new(
            Expr::new(ExprNode::Window(crate::WindowExpr {
                function: call("collect_list", vec![pair], Some(filter)),
                window: w.window.clone(),
            }))
            .with_origin(origin),
        );
        let sorted = call(
            "sort_array",
            vec![collected, lit(Literal::Boolean(ascending))],
            None,
        );
        let top = call(
            "slice",
            vec![
                sorted,
                lit(Literal::Integer(1)),
                lit(Literal::Integer(*k_value)),
            ],
            None,
        );
        Some(call(
            "__subscript",
            vec![top, lit(Literal::String("v".to_owned()))],
            None,
        ))
    }
}
