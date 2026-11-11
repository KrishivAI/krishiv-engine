// SPDX-License-Identifier: Apache-2.0
//! DISTINCT aggregates: the rewrite, the lift of calls and the aggregation check.

use super::*;

impl Resolver<'_> {
    /// Whether `items` (bound select items) hold what Spark's
    /// `RewriteDistinctAggregates` rewrites: several distinct argument
    /// sets, or a distinct aggregate with a filter. A single distinct set
    /// without a filter plans as one aggregate (the executor's
    /// two-phase distinct plan).
    pub(in crate::analyzer) fn needs_distinct_rewrite(items: &[Bound]) -> bool {
        let mut sets: Vec<Vec<u128>> = Vec::new();
        let mut memo = HashMap::new();
        let mut any_filtered = false;
        for (_, f) in Self::aggregate_calls(items) {
            if !f.distinct {
                continue;
            }
            any_filtered = any_filtered || f.filter.is_some();
            let mut set: Vec<u128> = f.args.iter().map(|a| semantic_key(a, &mut memo)).collect();
            set.sort_unstable();
            if !sets.contains(&set) {
                sets.push(set);
            }
        }
        sets.len() > 1 || any_filtered
    }

    /// The aggregate calls of bound select items, once each (keyed as the
    /// physical planner keys them: the whole call, filter included).
    pub(in crate::analyzer) fn aggregate_calls(items: &[Bound]) -> Vec<(String, ResolvedFunction)> {
        let mut calls: Vec<(String, ResolvedFunction)> = Vec::new();
        for b in items {
            let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(&b.expr), |n| {
                if let crate::walk::Node::Expr(x) = n
                    && let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
                    && f.kind == FunctionKind::Aggregate
                {
                    // The identity, not the display: a session UDF and a
                    // built-in of its name are two calls.
                    let key = crate::identity_string(x);
                    if !calls.iter().any(|(k, _)| *k == key) {
                        calls.push((key, f.clone()));
                    }
                }
                std::ops::ControlFlow::Continue(())
            });
        }
        calls
    }

    /// Spark's `RewriteDistinctAggregates`: `None` when the aggregate
    /// plans as one (see [`needs_distinct_rewrite`](Self::needs_distinct_rewrite)),
    /// else an `Expand` (one projection per distinct argument set, plus
    /// one for the other aggregates, told apart by a group id) and two
    /// ordinary aggregates. The second aggregate's output keeps the
    /// select list's attributes, so `HAVING` and `ORDER BY` above resolve
    /// as ever.
    ///
    /// The expansion evaluates every distinct argument (and every
    /// distinct filter) once per row, in its set's projection, and passes
    /// the input's columns through every projection. The first aggregate
    /// groups by the grouping expressions, the distinct arguments and the
    /// id, computing the other aggregates and, for each filtered
    /// distinct call, whether any row of the group passes its filter
    /// (`max` of it). The second groups by the grouping expressions,
    /// counting each distinct argument where the id is its set's (and the
    /// filter passed) and taking the other aggregates' values where the
    /// id is the plain one.
    /// Whether argument `n` of a `DISTINCT` call is a constant that
    /// `RewriteDistinctAggregates` leaves in place (Spark skips foldable
    /// children): any argument after the first, so a one-argument
    /// `count(DISTINCT 0.8)` still distincts its argument.
    pub(in crate::analyzer) fn constant_distinct_arg(&self, n: usize, a: &Arc<Expr>) -> bool {
        n >= 1 && self.fold(a, self.ctx.conf.ansi).is_some()
    }

    pub(in crate::analyzer) fn rewrite_distinct(
        &mut self,
        origin: Origin,
        c: &Rel,
        group: &[Bound],
        items: &[Bound],
        global: bool,
        env: &Arc<Env>,
    ) -> Result<Option<Rel>, AnalysisError> {
        // Items computed above this aggregate (lateral aliases, windows)
        // stay on the ordinary path, which rejects several distinct sets
        // loudly, as does anything this rewrite does not trigger on.
        if items
            .iter()
            .any(|b| !b.lateral.is_empty() || b.win.is_some())
            || !Self::needs_distinct_rewrite(items)
        {
            return Ok(None);
        }
        let calls = Self::aggregate_calls(items);
        // A `DISTINCT` call with an ordering has its keys among the arguments the expansion would
        // split, one keeping collated values dedupes by key: leave them to the ordinary path.
        if calls.iter().any(|(_, f)| {
            (f.distinct && !f.within_group.is_empty())
                || super::super::unexecutable::keeps_collated(f)
        }) {
            return Ok(None);
        }
        let mut memo = HashMap::new();
        // The distinct argument sets in order of first appearance, with
        // the indices of their calls; then the other calls.
        let mut sets: Vec<(Vec<u128>, Vec<usize>)> = Vec::new();
        let mut regular: Vec<usize> = Vec::new();
        for (i, (_, f)) in calls.iter().enumerate() {
            if !f.distinct {
                regular.push(i);
                continue;
            }
            let mut key: Vec<u128> = f
                .args
                .iter()
                .enumerate()
                .filter(|(n, a)| !self.constant_distinct_arg(*n, a))
                .map(|(_, a)| semantic_key(a, &mut memo))
                .collect();
            key.sort_unstable();
            match sets.iter_mut().find(|(k, _)| *k == key) {
                Some((_, v)) => v.push(i),
                None => sets.push((key, vec![i])),
            }
        }
        // A global aggregate over no rows evaluates each call on an empty
        // group; a regular call becomes `first` of its partial result, which
        // is NULL there. The count family is coalesced to 0 below, but an
        // aggregate UDF's empty-group value is unknown at analysis.
        if global
            && regular
                .iter()
                .any(|&i| matches!(calls[i].1.id, crate::FunctionId::Udf { .. }))
        {
            return Err(unsupported(
                "an aggregate UDF without GROUP BY next to a filtered DISTINCT aggregate or several DISTINCT argument sets",
                origin,
            ));
        }
        // Every distinct argument, once (in order of first appearance),
        // with its nullable expansion column.
        let zone = self.ctx.conf.session_zone;
        let empty = Self::cx(&[], &[], env, "Aggregate");
        let mut child_exprs: Vec<Arc<Expr>> = Vec::new();
        let mut child_keys: Vec<u128> = Vec::new();
        for (_, members) in &sets {
            for &i in members {
                for (n, a) in calls[i].1.args.iter().enumerate() {
                    if self.constant_distinct_arg(n, a) {
                        continue;
                    }
                    let k = semantic_key(a, &mut memo);
                    if !child_keys.contains(&k) {
                        child_keys.push(k);
                        child_exprs.push(Arc::clone(a));
                    }
                }
            }
        }
        let mut child_cols: Vec<Attr> = Vec::with_capacity(child_exprs.len());
        for a in &child_exprs {
            let ty = self.bind(a, &empty)?.ty;
            let col = self.attr(
                &pretty(a, zone),
                &ArgType {
                    data_type: ty.data_type.clone(),
                    nullable: true,
                },
            );
            child_cols.push(Attr {
                col,
                qual: no_qual(),
                source: None,
                duplicate: false,
            });
        }
        let gid_attr = Attr {
            col: self.attr(
                "gid",
                &ArgType::new(krishiv_types::DataType::Integer, false),
            ),
            qual: no_qual(),
            source: None,
            duplicate: false,
        };
        // The grouping expressions that are not plain input columns, once
        // each: computed below the expansion so every projection groups
        // alike (an inline non-deterministic key would group apart).
        let mut computed: Vec<(u128, Arc<Expr>, Attr)> = Vec::new();
        for b in group {
            if matches!(b.expr.node, ExprNode::Resolved(ResolvedExpr::Column(_))) {
                continue;
            }
            let k = semantic_key(&b.expr, &mut memo);
            if computed.iter().any(|(j, _, _)| *j == k) {
                continue;
            }
            let ty = self.bind(&b.expr, &empty)?.ty;
            let col = self.attr(
                &pretty(&b.expr, zone),
                &ArgType {
                    data_type: ty.data_type.clone(),
                    nullable: true,
                },
            );
            computed.push((
                k,
                Arc::clone(&b.expr),
                Attr {
                    col,
                    qual: no_qual(),
                    source: None,
                    duplicate: false,
                },
            ));
        }
        // A distinct call's filter that reads a subquery is computed below
        // the expansion too (the subquery rewrite joins it to the input
        // there); the projections read its column.
        let mut filter_cols: Vec<(u128, Attr)> = Vec::new();
        for (_, f) in &calls {
            let Some(filter) = &f.filter else {
                continue;
            };
            if !f.distinct || !has_subquery(filter) {
                continue;
            }
            let k = semantic_key(filter, &mut memo);
            if filter_cols.iter().any(|(j, _)| *j == k) {
                continue;
            }
            let col = self.attr(
                &pretty(filter, zone),
                &ArgType::new(krishiv_types::DataType::Boolean, true),
            );
            let attr = Attr {
                col,
                qual: no_qual(),
                source: None,
                duplicate: false,
            };
            computed.push((k, Arc::clone(filter), attr.clone()));
            filter_cols.push((k, attr));
        }
        // One nullable boolean column per filtered distinct call: whether
        // some row of the group passes its filter.
        let mut cond_cols: Vec<(usize, Attr)> = Vec::new();
        for (i, (_, f)) in calls.iter().enumerate() {
            if !f.distinct {
                continue;
            }
            let Some(filter) = &f.filter else {
                continue;
            };
            let col = self.attr(
                &pretty(filter, zone),
                &ArgType::new(krishiv_types::DataType::Boolean, true),
            );
            cond_cols.push((
                i,
                Attr {
                    col,
                    qual: no_qual(),
                    source: None,
                    duplicate: false,
                },
            ));
        }
        // The other aggregates' arguments that are not plain columns, once
        // each: evaluated in the plain projection.
        let mut regular_exprs: Vec<(u128, Arc<Expr>)> = Vec::new();
        for &i in &regular {
            for a in &calls[i].1.args {
                if matches!(a.node, ExprNode::Resolved(ResolvedExpr::Column(_))) {
                    continue;
                }
                let k = semantic_key(a, &mut memo);
                if regular_exprs.iter().any(|(j, _)| *j == k) {
                    continue;
                }
                regular_exprs.push((k, Arc::clone(a)));
            }
        }
        let mut regular_cols: Vec<(u128, Attr)> = Vec::with_capacity(regular_exprs.len());
        for (k, a) in &regular_exprs {
            let ty = self.bind(a, &empty)?.ty;
            let col = self.attr(
                &pretty(a, zone),
                &ArgType {
                    data_type: ty.data_type.clone(),
                    nullable: true,
                },
            );
            regular_cols.push((
                *k,
                Attr {
                    col,
                    qual: no_qual(),
                    source: None,
                    duplicate: false,
                },
            ));
        }
        // The expansion's input: the child's columns and the computed
        // grouping expressions.
        let computed_cols: Vec<(u128, Attr)> =
            computed.iter().map(|(k, _, a)| (*k, a.clone())).collect();
        let e0 = if computed.is_empty() {
            c.clone()
        } else {
            // The child's columns pass through, then the computed keys.
            let mut exprs: Vec<Arc<Expr>> = c
                .out
                .iter()
                .map(|a| col_expr(&a.col, Origin::UNKNOWN))
                .collect();
            let mut out: Vec<Attr> = c.out.to_vec();
            for (_, e, a) in &computed {
                exprs.push(Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: Arc::clone(e),
                        output: a.col.clone(),
                    }))
                    .with_origin(origin),
                ));
                out.push(a.clone());
            }
            self.projection(Origin::UNKNOWN, c, exprs, out)
        };
        // The expansion's output: the input's columns, the group id, one
        // column per distinct argument, distinct filter and computed
        // other-aggregate argument.
        let mut out: Vec<Attr> = e0.out.to_vec();
        out.push(gid_attr.clone());
        out.extend(child_cols.iter().cloned());
        out.extend(cond_cols.iter().map(|(_, a)| a.clone()));
        out.extend(regular_cols.iter().map(|(_, a)| a.clone()));
        let null_of = |a: &Attr| {
            Arc::new(Expr::new(ExprNode::Literal(Literal::Null(Some(
                a.col.data_type.clone(),
            )))))
        };
        let gid_lit = |g: i32| Arc::new(Expr::new(ExprNode::Literal(Literal::Integer(g))));
        let gid_set = |s: usize| {
            i32::try_from(s)
                .map_err(|_| unsupported("that many distinct sets", origin))
                .map(gid_lit)
        };
        let mut projections: Vec<Vec<Arc<Expr>>> = Vec::with_capacity(sets.len() + 1);
        if !regular.is_empty() {
            let mut row: Vec<Arc<Expr>> = e0
                .out
                .iter()
                .map(|a| col_expr(&a.col, Origin::UNKNOWN))
                .collect();
            row.push(gid_lit(0));
            row.extend(child_cols.iter().map(null_of));
            row.extend(cond_cols.iter().map(|(_, a)| null_of(a)));
            // The other aggregates' non-column arguments evaluate in
            // this projection (each once per row), into their columns;
            // plain columns read the input's own.
            for (_, a) in &regular_exprs {
                row.push(Arc::clone(a));
            }
            projections.push(row);
        }
        for (s, (_, members)) in sets.iter().enumerate() {
            let mut here: HashSet<u128> = HashSet::new();
            for &i in members {
                for (n, a) in calls[i].1.args.iter().enumerate() {
                    if !self.constant_distinct_arg(n, a) {
                        here.insert(semantic_key(a, &mut memo));
                    }
                }
            }
            let mut row: Vec<Arc<Expr>> = e0
                .out
                .iter()
                .map(|a| col_expr(&a.col, Origin::UNKNOWN))
                .collect();
            row.push(gid_set(s + 1)?);
            for (j, a) in child_exprs.iter().enumerate() {
                if here.contains(&child_keys[j]) {
                    row.push(Arc::clone(a));
                } else {
                    row.push(null_of(&child_cols[j]));
                }
            }
            // Only this set's filters evaluate; the rest are NULL.
            let mut filters: Vec<Arc<Expr>> = cond_cols.iter().map(|(_, a)| null_of(a)).collect();
            for &i in members {
                if let Some(p) = cond_cols.iter().position(|(j, _)| *j == i)
                    && let Some(f) = &calls[i].1.filter
                {
                    let k = semantic_key(f, &mut memo);
                    filters[p] = filter_cols
                        .iter()
                        .find(|(j, _)| *j == k)
                        .map_or_else(|| Arc::clone(f), |(_, a)| col_expr(&a.col, origin));
                }
            }
            row.extend(filters);
            row.extend(regular_cols.iter().map(|(_, a)| null_of(a)));
            projections.push(row);
        }
        let node = PlanNode::Expand {
            input: Arc::clone(&e0.plan),
            projections,
            output: out.iter().map(|a| a.col.clone()).collect(),
        };
        let expand = self.make(origin, node, out, &[&e0]);
        let expand_parents = [&expand];
        let expand_cx = Self::cx(&expand.out, &expand_parents, env, "Aggregate");
        // The first aggregate: grouping expressions, distinct arguments
        // and id; the other aggregates and each filter's `max`.
        let mut group1: Vec<Bound> = Vec::new();
        let mut key_attrs: Vec<Attr> = Vec::new();
        for b in group {
            let Some(a) = expand_col(&b.expr, &e0, &computed_cols, &mut memo) else {
                return Ok(None);
            };
            key_attrs.push(a.clone());
            group1.push(column_bound(&a, Origin::UNKNOWN));
        }
        group1.extend(child_cols.iter().map(|a| column_bound(a, Origin::UNKNOWN)));
        group1.push(column_bound(&gid_attr, Origin::UNKNOWN));
        let mut items1: Vec<Bound> = Vec::new();
        // The grouping columns ride as items (as in `SELECT a, max(x)
        // ... GROUP BY a`): the planner outputs only what the aggregate
        // lists, and the second aggregate reads the keys.
        for a in key_attrs.iter().chain(child_cols.iter()).chain([&gid_attr]) {
            items1.push(column_bound(a, Origin::UNKNOWN));
        }
        let mut max_of: HashMap<usize, Attr> = HashMap::new();
        for (i, a) in &cond_cols {
            let b = self.bind(
                &synth_call("max", vec![col_expr(&a.col, origin)], None, origin),
                &expand_cx,
            )?;
            let b = self.named(b, true)?;
            max_of.insert(*i, out_attr(&b, &expand.out));
            items1.push(b);
        }
        let mut partial_of: Vec<Attr> = Vec::with_capacity(regular.len());
        for &i in &regular {
            let f = &calls[i].1;
            let mut mapped = Vec::with_capacity(f.args.len());
            for (n, a) in f.args.iter().enumerate() {
                // A `listagg` delimiter and a `mode` tie-break flag must stay foldable literals.
                if n == 1 && matches!(f.id.name(), "listagg" | "mode") {
                    mapped.push(Arc::clone(a));
                    continue;
                }
                match expand_col(a, &e0, &regular_cols, &mut memo) {
                    Some(x) => mapped.push(col_expr(&x.col, origin)),
                    None => mapped.push(Arc::clone(a)),
                }
            }
            // A resolved `listagg` holds its ordering keys after its value
            // and delimiter: written again, they are the `WITHIN GROUP`
            // list. A resolved `mode` encodes its ordering in its arguments.
            let mut call = synth_call_of(f, Vec::new(), f.filter.clone(), origin);
            if let ExprNode::Function(c) = &mut Arc::make_mut(&mut call).node {
                let keys = if f.id.name() == "listagg" {
                    f.within_group.len().min(mapped.len())
                } else {
                    0
                };
                c.within_group = f
                    .within_group
                    .iter()
                    .zip(mapped.split_off(mapped.len() - keys))
                    .map(|(o, expr)| SortOrder {
                        expr,
                        direction: o.direction,
                        nulls: o.nulls,
                    })
                    .collect();
                c.args = mapped;
            }
            let b = self.bind(&call, &expand_cx)?;
            let b = self.named(b, true)?;
            partial_of.push(out_attr(&b, &expand.out));
            items1.push(b);
        }
        let r1 = self.build_aggregate(origin, &expand, group1, &items1, false, None, env)?;
        // The second aggregate's calls, keyed by the originals' text.
        let r1_parents = [&r1];
        let r1_cx = Self::cx(&r1.out, &r1_parents, env, "Aggregate");
        let mut new_call: HashMap<String, Arc<Expr>> = HashMap::new();
        for (s, (_, members)) in sets.iter().enumerate() {
            let g =
                i32::try_from(s + 1).map_err(|_| unsupported("that many distinct sets", origin))?;
            for &i in members {
                let f = &calls[i].1;
                let mut args = Vec::with_capacity(f.args.len());
                for (n, a) in f.args.iter().enumerate() {
                    // A constant stays as written (a quantile's percentage
                    // must stay foldable).
                    if self.constant_distinct_arg(n, a) {
                        args.push(Arc::clone(a));
                        continue;
                    }
                    let k = semantic_key(a, &mut memo);
                    let pos = child_keys.iter().position(|j| *j == k).ok_or_else(|| {
                        unsupported("a distinct argument the expansion lacks", origin)
                    })?;
                    args.push(col_expr(&child_cols[pos].col, origin));
                }
                let mut filter = self
                    .bind(
                        &synth_call(
                            "==",
                            vec![col_expr(&gid_attr.col, origin), gid_lit(g)],
                            None,
                            origin,
                        ),
                        &r1_cx,
                    )?
                    .expr;
                if let Some(m) = max_of.get(&i) {
                    filter = self
                        .bind(
                            &synth_call(
                                "and",
                                vec![filter, col_expr(&m.col, origin)],
                                None,
                                origin,
                            ),
                            &r1_cx,
                        )?
                        .expr;
                }
                let b = self.bind(&synth_call_of(f, args, Some(filter), origin), &r1_cx)?;
                new_call.insert(calls[i].0.clone(), b.expr);
            }
        }
        for (k, &i) in regular.iter().enumerate() {
            let filter = self
                .bind(
                    &synth_call(
                        "==",
                        vec![col_expr(&gid_attr.col, origin), gid_lit(0)],
                        None,
                        origin,
                    ),
                    &r1_cx,
                )?
                .expr;
            let b = self.bind(
                &synth_call(
                    "first",
                    vec![col_expr(&partial_of[k].col, origin)],
                    Some(filter),
                    origin,
                ),
                &r1_cx,
            )?;
            let mut replaced = b.expr;
            // A global count over no rows is 0, but `first` of the zero
            // partial rows the second phase sees is NULL: the count family
            // is coalesced to 0 (a grouped query always has a row per group).
            if matches!(calls[i].1.id, crate::FunctionId::Registry { .. })
                && matches!(calls[i].1.id.name(), "count" | "count_if")
            {
                replaced = self
                    .bind(
                        &synth_call(
                            "coalesce",
                            vec![replaced, Arc::new(Literal::Long(0).into())],
                            None,
                            origin,
                        ),
                        &r1_cx,
                    )?
                    .expr;
            }
            new_call.insert(calls[i].0.clone(), replaced);
        }
        // The grouping over the first aggregate, and the select list with
        // its calls and grouping expressions replaced. Keys match through
        // an alias: a key bound through an ordinal or a select-list alias
        // is the alias, while other items may name the expression bare.
        let mut key_map: HashMap<u128, Arc<Expr>> = HashMap::new();
        let mut group2: Vec<Bound> = Vec::new();
        for (b, a) in group.iter().zip(&key_attrs) {
            let col = col_expr(&a.col, origin);
            key_map.insert(semantic_key(&b.expr, &mut memo), Arc::clone(&col));
            if let ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) = &b.expr.node {
                key_map.insert(semantic_key(expr, &mut memo), col);
            }
            group2.push(column_bound(a, Origin::UNKNOWN));
        }
        let mut items2: Vec<Bound> = Vec::with_capacity(items.len());
        for b in items {
            let expr = match Arc::clone(&b.expr).transform_down(&mut |x| {
                Ok::<_, Infallible>(subst_distinct(&x, &new_call, &key_map, &mut memo))
            }) {
                Ok(e) => e,
                Err(void) => match void {},
            };
            items2.push(Bound {
                expr,
                ty: b.ty.clone(),
                ..b.clone()
            });
        }
        let mut r = self.build_aggregate(origin, &r1, group2, &items2, global, None, env)?;
        if let Some(info) = r.agg.as_mut() {
            Arc::make_mut(info).distinct_rewritten = true;
        }
        Ok(Some(r))
    }

    /// A select item that reads lateral aliases, bound over the
    /// aggregate's input, over its output instead: aggregate calls and
    /// grouping expressions read attributes of the aggregate (added to it
    /// when new); lateral aliases stay. Any other input column is
    /// `MISSING_AGGREGATION` (`MISSING_GROUP_BY` without `GROUP BY`).
    #[expect(
        clippy::too_many_arguments,
        reason = "one call site, all state of the rewrite"
    )]
    pub(in crate::analyzer) fn lift(
        &mut self,
        b: &Bound,
        info: &mut AggInfo,
        keys: &[(u128, ArgType)],
        lateral: &HashSet<AttrId>,
        memo: &mut HashMap<usize, u128>,
        global: bool,
        origin: Origin,
    ) -> Result<Arc<Expr>, AnalysisError> {
        let mut cx = Lift {
            info,
            keys,
            lateral,
            memo,
            global,
            origin,
        };
        self.lift_expr(Arc::clone(&b.expr), &mut cx)
    }

    /// [`Self::lift`] of one node, pre-order. A window's function stays a
    /// window call (its arguments are lifted), so `sum(sum(x)) OVER w`
    /// reads the aggregate's `sum(x)`.
    pub(in crate::analyzer) fn lift_expr(
        &mut self,
        x: Arc<Expr>,
        l: &mut Lift<'_>,
    ) -> Result<Arc<Expr>, AnalysisError> {
        if let ExprNode::Window(_) = &x.node {
            let mut first = true;
            return x.map_children(|c| {
                if std::mem::take(&mut first) {
                    c.map_children(|a| self.lift_expr(a, l))
                } else {
                    self.lift_expr(c, l)
                }
            });
        }
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node
            && l.lateral.contains(&c.id)
        {
            return Ok(x);
        }
        let k = semantic_key(&x, l.memo);
        let ty = match (l.keys.iter().find(|(kk, _)| *kk == k), &x.node) {
            (Some((_, ty)), _) => ty.clone(),
            (None, ExprNode::Resolved(ResolvedExpr::Function(f)))
                if f.kind == FunctionKind::Aggregate =>
            {
                ArgType {
                    data_type: Some(f.data_type.clone()),
                    nullable: f.nullable,
                }
            }
            (None, ExprNode::Resolved(ResolvedExpr::Column(_))) if l.global => {
                return Err(AnalysisError::new(MissingGroupBy {}, l.origin));
            }
            (None, ExprNode::Resolved(ResolvedExpr::Column(_))) => {
                let text = pretty(&x, self.ctx.conf.session_zone);
                return Err(AnalysisError::new(
                    MissingAggregation {
                        expression: format!("\"{text}\""),
                        expression_any_value: format!("\"any_value({text})\""),
                    },
                    Origin::UNKNOWN,
                ));
            }
            (None, _) => return x.map_children(|c| self.lift_expr(c, l)),
        };
        // An attribute of the aggregate computing `x`, added if new.
        let info = &mut *l.info;
        let found = info
            .aggregates
            .iter()
            .zip(&info.out)
            .find(|(e, _)| semantic_key(e, l.memo) == k)
            .map(|(_, a)| a.clone());
        let key_attr = match &x.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                info.keys.iter().find(|a| a.col.id == c.id).cloned()
            }
            _ => None,
        };
        let a = match (found, key_attr) {
            (Some(a), _) => a,
            (None, Some(a)) => {
                info.aggregates.push(col_expr(&a.col, Origin::UNKNOWN));
                info.out.push(a.clone());
                a
            }
            (None, None) => {
                let output = self.attr(&pretty(&x, self.ctx.conf.session_zone), &ty);
                info.aggregates.push(Arc::new(Expr::new(ExprNode::Resolved(
                    ResolvedExpr::Alias {
                        expr: Arc::clone(&x),
                        output: output.clone(),
                    },
                ))));
                let a = Attr {
                    col: output,
                    qual: no_qual(),
                    source: None,
                    duplicate: false,
                };
                info.out.push(a.clone());
                a
            }
        };
        Ok(col_expr(&a.col, x.origin))
    }

    /// Spark's `MISSING_AGGREGATION` / `MISSING_GROUP_BY`: outside
    /// aggregate calls, an item may use only grouping expressions.
    pub(in crate::analyzer) fn check_aggregation(
        &self,
        items: &[Bound],
        keys: &HashSet<u128>,
        memo: &mut HashMap<usize, u128>,
        origin: Origin,
        global: bool,
    ) -> Result<(), AnalysisError> {
        let mut seen = HashSet::new();
        // A lateral alias is an item of the list itself, checked there.
        let aliases: HashSet<AttrId> = items
            .iter()
            .flat_map(|b| b.lateral.iter().copied())
            .collect();
        for b in items {
            let mut stack = vec![Arc::clone(&b.expr)];
            while let Some(x) = stack.pop() {
                if !seen.insert(super::super::bind::addr(&*x))
                    || keys.contains(&semantic_key(&x, memo))
                {
                    continue;
                }
                if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node
                    && aliases.contains(&c.id)
                {
                    continue;
                }
                match &x.node {
                    ExprNode::Resolved(ResolvedExpr::Function(f))
                        if f.kind == FunctionKind::Aggregate => {}
                    ExprNode::Subquery { .. } => {}
                    ExprNode::Resolved(ResolvedExpr::Column(_)) if global => {
                        return Err(AnalysisError::new(MissingGroupBy {}, origin));
                    }
                    ExprNode::Resolved(ResolvedExpr::Column(_)) => {
                        let text = pretty(&x, self.ctx.conf.session_zone);
                        return Err(AnalysisError::new(
                            MissingAggregation {
                                expression: format!("\"{text}\""),
                                expression_any_value: format!("\"any_value({text})\""),
                            },
                            Origin::UNKNOWN,
                        ));
                    }
                    _ => x.for_each_child(&mut |c| stack.push(Arc::clone(c))),
                }
            }
        }
        Ok(())
    }
}
