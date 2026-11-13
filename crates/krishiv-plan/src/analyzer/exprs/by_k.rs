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
        // A comparator on `k` only, so a map-valued `x` sorts as well (the entries order by `y`).
        let var = |name: &str| {
            Arc::new(
                Expr::new(ExprNode::Column(crate::ColumnRef::new(vec![
                    name.to_owned(),
                ])))
                .with_origin(origin),
            )
        };
        let key = |name: &str| {
            call(
                "__subscript",
                vec![var(name), lit(Literal::String("k".to_owned()))],
                None,
            )
        };
        let (before, after) = if ascending { (-1, 1) } else { (1, -1) };
        let compare = call(
            "if",
            vec![
                call("<", vec![key("l"), key("r")], None),
                lit(Literal::Integer(before)),
                call(
                    "if",
                    vec![
                        call(">", vec![key("l"), key("r")], None),
                        lit(Literal::Integer(after)),
                        lit(Literal::Integer(0)),
                    ],
                    None,
                ),
            ],
            None,
        );
        let comparator = Arc::new(
            Expr::lambda(vec!["l".to_owned(), "r".to_owned()], (*compare).clone())
                .with_origin(origin),
        );
        let sorted = call("array_sort", vec![Arc::clone(&collected), comparator], None);
        let top = call(
            "slice",
            vec![
                sorted,
                lit(Literal::Integer(1)),
                lit(Literal::Integer(*k_value)),
            ],
            None,
        );
        // NULL, not `[]`, when the frame holds no row with a `y` (Spark): `collected` is the window
        // result, used twice as one node (bound and planned once); `size` of it, not of the sorted
        // slice, so the comparator lambda is evaluated once per row.
        Some(call(
            "if",
            vec![
                call(
                    "<",
                    vec![
                        call("size", vec![collected], None),
                        lit(Literal::Integer(1)),
                    ],
                    None,
                ),
                lit(Literal::Null(None)),
                call(
                    "__subscript",
                    vec![top, lit(Literal::String("v".to_owned()))],
                    None,
                ),
            ],
            None,
        ))
    }

    /// The name of the rewritten call as written: the original call over the bound window spec
    /// (`max_by(emp, salary, 2) OVER (PARTITION BY dept ...)`), for the column name and for errors.
    pub(super) fn by_k_shown(original: &Expr, bound: &Bound) -> Option<Arc<Expr>> {
        let ExprNode::Window(w) = &original.node else {
            return None;
        };
        let inner = super::super::windows::first_window(&bound.shown())?;
        let ExprNode::Window(shown) = &inner.node else {
            return None;
        };
        Some(Arc::new(
            Expr::new(ExprNode::Window(crate::WindowExpr {
                function: Arc::clone(&w.function),
                window: shown.window.clone(),
            }))
            .with_origin(original.origin),
        ))
    }
}
