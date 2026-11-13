// SPDX-License-Identifier: Apache-2.0
//! Aggregate binding: grouping keys, ordinals, GROUP BY checks and the aggregate node.

use super::*;

impl Resolver<'_> {
    /// Expressions over an aggregate (`HAVING`, `ORDER BY`): names resolve
    /// against its output, then its grouping attributes; aggregate-function
    /// arguments against its input. A name the output holds twice is looked
    /// up among the grouping attributes instead of failing as ambiguous
    /// (Spark resolves `HAVING`/`ORDER BY` to the grouping column there);
    /// when the grouping attributes hold no such name either, the duplicate
    /// outputs stay ambiguous rather than unresolved. Aggregate calls the
    /// aggregate does not compute yet, and grouping attributes it does not
    /// output, are added to it (and passed up through the projections of
    /// lateral aliases above it); the flag says whether its output grew.
    pub(in crate::analyzer) fn over_agg(
        &mut self,
        agg: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
    ) -> Result<(Vec<Bound>, Rel, bool), AnalysisError> {
        let Some(info) = agg.agg.clone() else {
            return Err(unsupported("a plan shape", agg.plan.origin));
        };
        let rels = [agg];
        let mut cx = Self::cx(&agg.out, &rels, env, op);
        cx.extra = Some(&info.keys);
        // A name the select list outputs twice is looked up among the
        // grouping attributes, as Spark looks a name an `ORDER BY` output
        // holds twice up below it: without leniency the duplicate outputs
        // are `AMBIGUOUS_REFERENCE` where Spark resolves to the grouping
        // column.
        cx.lenient = true;
        cx.agg_scope = Some(&info.input.out);
        cx.grouping = info.analytics.is_some();
        if op == "Filter" {
            cx.windows = WindowUse::Having;
        } else if op == "Qualify" {
            // `QUALIFY` resolves like `HAVING`, but windows are allowed
            // and extracted above the aggregate.
            cx.windows = WindowUse::Allowed;
        }
        let mut memo = HashMap::new();
        let mut known: Vec<(u128, Attr)> = info
            .aggregates
            .iter()
            .zip(info.out.iter())
            .map(|(e, a)| (semantic_key(e, &mut memo), a.clone()))
            .collect();
        let visible: HashSet<AttrId> = agg.out.iter().map(|a| a.col.id).collect();
        let mut grown = (*info).clone();
        // What the aggregate adds to the output it shows.
        let mut pass: Vec<Attr> = Vec::new();
        let mut bounds = Vec::new();
        for e in exprs {
            match self.bind(e, &cx) {
                Ok(b) => bounds.push(b),
                // A grouping expression written out in full (`GROUP BY x + 1
                // ORDER BY x + 1`) is the aggregate's output, wherever it
                // appears in the expression; names it does not cover stay
                // unresolved, as in Spark.
                Err(err)
                    if err
                        .error
                        .error_class()
                        .name()
                        .starts_with("UNRESOLVED_COLUMN") =>
                {
                    let mut st = Grouped {
                        info: &info,
                        env,
                        op,
                        visible: &visible,
                        memo: &mut memo,
                        known: &mut known,
                        grown: &mut grown,
                        pass: &mut pass,
                    };
                    let Some(sub) = self.group_substitute(e, &mut st) else {
                        return Err(err);
                    };
                    bounds.push(self.bind(&sub, &cx).map_err(|_| err)?);
                }
                Err(err) => return Err(err),
            }
        }
        if let Some(an) = &info.analytics {
            for b in &mut bounds {
                an.check_late_call(&b.expr)?;
                b.expr = self.resolve_grouping(&b.expr, an, env)?;
            }
        }
        let mut st = Grouped {
            info: &info,
            env,
            op,
            visible: &visible,
            memo: &mut memo,
            known: &mut known,
            grown: &mut grown,
            pass: &mut pass,
        };
        for b in &mut bounds {
            b.expr = Arc::clone(&b.expr).transform_down(&mut |x: Arc<Expr>| match &x.node {
                ExprNode::Resolved(ResolvedExpr::Function(f))
                    if f.kind == FunctionKind::Aggregate =>
                {
                    let k = semantic_key(&x, st.memo);
                    let a = self.aggregate_output(&x, k, &mut st)?;
                    Ok(col_expr(&a.col, x.origin))
                }
                ExprNode::Resolved(ResolvedExpr::Column(c))
                    if !st.visible.contains(&c.id) && !st.pass.iter().any(|a| a.col.id == c.id) =>
                {
                    if let Some(k) = st.info.keys.iter().find(|k| k.col.id == c.id) {
                        // An output that already computes the key (`x AS
                        // m`) stands for it, as in Spark: nothing grows.
                        let key = semantic_key(&x, st.memo);
                        if let Some((_, a)) = st.known.iter().find(|(kk, _)| *kk == key) {
                            return Ok(col_expr(&a.col, x.origin));
                        }
                        if !st.grown.out.iter().any(|o| o.col.id == c.id) {
                            st.grown.aggregates.push(col_expr(&k.col, Origin::UNKNOWN));
                            st.grown.out.push(k.clone());
                        }
                        st.pass.push(k.clone());
                    }
                    Ok(x)
                }
                _ => Ok(x),
            })?;
        }
        self.having_subqueries(&mut bounds, &mut st, agg.plan.origin)?;
        // An aggregate call over a merged-away `USING` column
        // (`ORDER BY max(t2.k)`) reads it from the input.
        grown.input = self.reveal_exprs(&grown.input, grown.aggregates.iter())?;
        if pass.is_empty() {
            return Ok((bounds, agg.clone(), false));
        }
        let r = self.aggregate_rel(agg.plan.origin, grown, &pass)?;
        Ok((bounds, r, true))
    }

    /// `e` with every subexpression that is one of the aggregate's grouping
    /// expressions or output expressions (equal after canonicalisation, as Spark's
    /// `semanticEquals`) replaced by the attribute the aggregate computes
    /// it as, added to the aggregate when it does not output it. `None`
    /// when nothing matches, or `e` holds a subquery.
    #[inline(never)]
    pub(in crate::analyzer) fn group_substitute(
        &mut self,
        e: &Arc<Expr>,
        st: &mut Grouped<'_>,
    ) -> Option<Arc<Expr>> {
        let info = st.info;
        let rels = [&info.input];
        let probe = Self::cx(&info.input.out, &rels, st.env, st.op);
        let grouping: HashSet<u128> = info
            .grouping
            .iter()
            .map(|g| semantic_key(g, st.memo))
            .collect();
        let mut subquery = false;
        let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
            if let crate::walk::Node::Expr(x) = n
                && matches!(x.node, ExprNode::Subquery { .. })
            {
                subquery = true;
            }
            std::ops::ControlFlow::Continue(())
        });
        if subquery {
            return None;
        }
        let mut changed = false;
        let out = Arc::clone(e)
            .transform_down(&mut |x: Arc<Expr>| -> Result<Arc<Expr>, Infallible> {
                // Leaves resolve by name; aggregate calls and the rest of
                // the output are bound by the caller.
                // A dotted name (`s.a`) may be a field of a grouped struct,
                // which is a grouping expression, so it is probed.
                let dotted = matches!(&x.node, ExprNode::Column(c) if c.name.len() > 1);
                if !dotted
                    && matches!(
                        x.node,
                        ExprNode::Column(_) | ExprNode::Literal(_) | ExprNode::Star(_)
                    )
                {
                    return Ok(x);
                }
                let Ok(b) = self.expr(&x, &probe) else {
                    return Ok(x);
                };
                if b.agg.is_some() || b.outer.is_some() || !b.lateral.is_empty() {
                    return Ok(x);
                }
                // The probe's expressions are dropped on return: their
                // addresses must not outlive them in a memo.
                let key = semantic_key(&b.expr, &mut HashMap::new());
                if !grouping.contains(&key) && !st.known.iter().any(|(k, _)| *k == key) {
                    return Ok(x);
                }
                let a = match st.known.iter().find(|(k, _)| *k == key) {
                    Some((_, a)) => a.clone(),
                    None => {
                        let output = self.attr(&pretty(&b.expr, self.ctx.conf.session_zone), &b.ty);
                        // The aggregate computes the grouping expression
                        // as written in `GROUP BY`: a commuted spelling
                        // (`1 + a` for `a + 1`) would read `a` ungrouped.
                        let computed = info
                            .grouping
                            .iter()
                            .find(|g| semantic_key(g, st.memo) == key)
                            .map_or_else(|| Arc::clone(&b.expr), Arc::clone);
                        st.grown
                            .aggregates
                            .push(Arc::new(Expr::new(ExprNode::Resolved(
                                ResolvedExpr::Alias {
                                    expr: computed,
                                    output: output.clone(),
                                },
                            ))));
                        let a = Attr {
                            col: output,
                            qual: no_qual(),
                            source: None,
                            duplicate: false,
                        };
                        st.grown.out.push(a.clone());
                        st.known.push((key, a.clone()));
                        a
                    }
                };
                if !st.visible.contains(&a.col.id) && !st.pass.iter().any(|p| p.col.id == a.col.id)
                {
                    st.pass.push(a.clone());
                }
                changed = true;
                Ok(col_expr(&a.col, x.origin))
            })
            .unwrap_or_else(|never| match never {});
        changed.then_some(out)
    }

    /// The aggregate of `info`, under its `HAVING` (with windows), its
    /// windows and the projections of its lateral aliases if it has any;
    /// `pass` (attributes of the aggregate) are output after the select
    /// items.
    pub(in crate::analyzer) fn aggregate_rel(
        &mut self,
        origin: Origin,
        info: AggInfo,
        pass: &[Attr],
    ) -> Result<Rel, AnalysisError> {
        // Over grouping sets, the aggregate reads their expansion; the
        // attributes it defines are part of the aggregate's id span.
        let (input, defined) = match info.analytics.as_deref().and_then(|a| a.expand.as_ref()) {
            Some(e) => (Arc::clone(&e.plan), e.ids),
            None => (Arc::clone(&info.input.plan), None),
        };
        let node = PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(info.grouping.clone()),
            aggregates: info.aggregates.clone(),
        };
        let mut r = match &info.top {
            None => self.make(origin, node, info.out.clone(), &[&info.input]),
            Some(top) => {
                let mut a = self.make(Origin::UNKNOWN, node, info.out.clone(), &[&info.input]);
                if let Some(h) = &info.having {
                    let f = PlanNode::Filter {
                        input: Arc::clone(&a.plan),
                        predicate: Arc::clone(h),
                    };
                    a = self.make(Origin::UNKNOWN, f, a.out.to_vec(), &[&a]);
                }
                if info.has_windows() {
                    self.window_project(origin, &a, top, pass)?
                } else {
                    self.lateral_project(origin, &a, top, pass)
                }
            }
        };
        r.ids = super::super::resolve::hull(r.ids, defined);
        let info = Arc::new(info);
        // A column the select list dropped comes back from the aggregate
        // (rebuilt with its windows and projections above it).
        r.widen = Some(Arc::new(Widen::Aggregate(Arc::clone(&info))));
        r.agg = Some(info);
        Ok(r)
    }

    pub(in crate::analyzer) fn aggregate(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        grouping: &Grouping,
        aggregates: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let Grouping::GroupBy(keys) = grouping else {
            return self.grouping_sets(p, input, grouping, aggregates, env);
        };
        let c = self.child(input, env)?;
        // A select item `stack(..)` is `inline` over its array of rows,
        // as in a plain select list.
        let stacked = self.stack_items(&c, aggregates, env)?;
        // `generator(..) AS (a, b)`: the names go to the generated columns,
        // the item resolves as the bare call.
        let mut written: Vec<String> = Vec::new();
        let stacked: Vec<Arc<Expr>> = stacked
            .iter()
            .map(|e| match &e.node {
                ExprNode::MultiAlias { expr, names }
                    if matches!(&expr.node, ExprNode::Function(f)
                        if super::super::generate::is_generator_name(&f.name)) =>
                {
                    written = names.clone();
                    Arc::clone(expr)
                }
                _ => Arc::clone(e),
            })
            .collect();
        let aggregates = &stacked[..];
        self.check_star_ordinals(keys, aggregates)?;
        // Spark resolves the select list first, then ordinals, then the
        // other keys.
        // Spark wraps a pipe item in `pipeexpression(..)`, so a generator
        // with several names (`AS (p, y)`) is nested in it there (4.2.0).
        for e in aggregates {
            if let ExprNode::Pipe { expr, .. } = &e.node
                && let ExprNode::MultiAlias { expr: g, .. } = &expr.node
                && matches!(&g.node, ExprNode::Function(f) if super::super::generate::is_generator_name(&f.name))
            {
                let zone = self.ctx.conf.session_zone;
                return Err(AnalysisError::new(
                    UnsupportedGeneratorNestedInExpressions {
                        expression: format!("\"pipeexpression(multialias({}))\"", pretty(g, zone)),
                    },
                    e.origin,
                ));
            }
        }
        let items = self.select_items(&c, aggregates, env, "Aggregate")?;
        crate::check_python_aggregates(items.iter().map(|b| &b.expr))
            .map_err(|e| AnalysisError::new(e, p.origin))?;
        // The declared `CHAR`/`VARCHAR` types follow the aliases of an explicit `GROUP BY` too.
        self.carry_declared(&items, aggregates);
        // Spark's `ExtractGenerator`: the generator's argument takes the
        // item's place in the aggregate, the generator runs above it.
        // A written alias of the generator (the analyzer names every item,
        // so the written one is read from the unresolved list).
        // A pipe `AGGREGATE` item is wrapped (`Alias(Pipe(generator))`).
        let unpiped = |e: &Arc<Expr>| match &e.node {
            ExprNode::Pipe { expr, .. } => Arc::clone(expr),
            _ => Arc::clone(e),
        };
        if let Some(name) = aggregates.iter().find_map(|e| match &e.node {
            ExprNode::Alias { name, expr }
                if matches!(&unpiped(expr).node, ExprNode::Function(f)
                if super::super::generate::is_generator_name(&f.name)) =>
            {
                Some(name.clone())
            }
            _ => None,
        }) {
            written = vec![name];
        }
        let (items, generator) = self.extract_generator(p.origin, items, written)?;
        // `GROUP BY ALL`: a lone unqualified `all` in SQL groups by the select
        // list, unless a column or a select-list alias named `all` takes it
        // (Spark resolves names first). A DataFrame `groupBy("all")` has no
        // SQL span and never expands (`UNRESOLVED_COLUMN`, as in Spark).
        let group = if p.origin.sql.is_some() && is_bare_all(keys) && !self.names_all(&c, &items) {
            self.group_by_all(&items, keys[0].origin)?
        } else {
            self.group_keys(&c, keys, &items, env)?
        };
        // No grouping (`HAVING` without `GROUP BY`): Spark's
        // `MISSING_GROUP_BY` for an ungrouped column.
        let global = group.is_empty();
        let c = self.reveal(&c, group.iter().chain(&items))?;
        let agg = self.build_aggregate(p.origin, &c, group, &items, global, None, env)?;
        match generator {
            None => Ok(agg),
            Some(g) => {
                if agg
                    .out
                    .get(g.index())
                    .is_none_or(|a| a.col.id != g.input_id())
                {
                    return Err(unsupported(
                        "a generator beside lateral aliases or windows in an aggregate",
                        p.origin,
                    ));
                }
                self.generate_over_aggregate(p.origin, &agg, &g)
            }
        }
    }

    /// `SELECT *` with `GROUP BY n` is an error (Spark's
    /// `STAR_GROUP_BY_POS`).
    pub(in crate::analyzer) fn check_star_ordinals(
        &self,
        keys: &[Arc<Expr>],
        aggregates: &[Arc<Expr>],
    ) -> Result<(), AnalysisError> {
        let star = aggregates
            .iter()
            .any(|e| matches!(e.node, ExprNode::Star(_)));
        if star && keys.iter().any(|k| self.ordinal(k).is_some()) {
            return Err(AnalysisError::new(StarGroupByPos {}, Origin::UNKNOWN));
        }
        Ok(())
    }

    /// `GROUP BY n` with ordinals on: `n`.
    pub(in crate::analyzer) fn ordinal(&self, k: &Expr) -> Option<i32> {
        match &k.node {
            ExprNode::Literal(Literal::Integer(n)) if self.ctx.conf.group_by_ordinal => Some(*n),
            _ => None,
        }
    }

    /// The grouping expressions `keys` over `c`: an ordinal names a select
    /// item, other names may be select-list aliases.
    pub(in crate::analyzer) fn group_keys(
        &mut self,
        c: &Rel,
        keys: &[Arc<Expr>],
        items: &[Bound],
        env: &Arc<Env>,
    ) -> Result<Vec<Bound>, AnalysisError> {
        let mut group: Vec<Option<Bound>> = vec![None; keys.len()];
        for (slot, k) in group.iter_mut().zip(keys) {
            if let Some(n) = self.ordinal(k) {
                *slot = Some(self.group_ordinal(n, k.origin, items, c)?);
            } else if let ExprNode::Star(s) = &k.node
                && s.position.is_some()
            {
                // `|> AGGREGATE ... GROUP BY n`: the n-th input column.
                let rels = [c];
                let cx = Self::cx(&c.out, &rels, env, "Aggregate");
                let attrs = self.expand(s, &cx, k.origin)?;
                *slot = attrs.first().map(|a| column_bound(a, k.origin));
            }
        }
        // Spark resolves the select list first, then ordinals, then the
        // other keys. A `GROUP BY` may name any select-list alias, explicit
        // or implicit (`SELECT 1 GROUP BY `1`` groups by the literal): the
        // aliases are the items' output names over the expressions that
        // compute them, searched after the input.
        let aliases: Vec<(String, Arc<Expr>)> = if self.ctx.conf.group_by_aliases {
            items
                .iter()
                .map(|b| {
                    let name = out_attr(b, &c.out).col.name;
                    let expr = match &b.expr.node {
                        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => Arc::clone(expr),
                        _ => Arc::clone(&b.expr),
                    };
                    (name, expr)
                })
                .collect()
        } else {
            Vec::new()
        };
        let rels = [c];
        for (slot, k) in group.iter_mut().zip(keys) {
            if slot.is_none() {
                let mut cx = Self::cx(&c.out, &rels, env, "Aggregate");
                cx.aliases = &aliases;
                *slot = Some(self.bind(k, &cx)?);
            }
        }
        let group: Vec<Bound> = group.into_iter().flatten().collect();
        self.check_grouping(&group, items)?;
        Ok(group)
    }

    /// Spark's `GROUP_BY_AGGREGATE` and
    /// `UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_GROUP_BY` for named
    /// keys: a grouping expression may not contain an aggregate call,
    /// however it was written (directly, or through a select-list alias
    /// resolving to one, which otherwise hides in the grouping-key set
    /// `check_aggregation` skips), and it may not compute what a lateral
    /// alias reader computes (as `group_ordinal` enforces for ordinals).
    pub(in crate::analyzer) fn check_grouping(
        &self,
        group: &[Bound],
        items: &[Bound],
    ) -> Result<(), AnalysisError> {
        let mut memo = HashMap::new();
        let mut lateral: Vec<u128> = Vec::new();
        for b in items {
            if !b.lateral.is_empty() {
                lateral.push(semantic_key(&b.expr, &mut memo));
            }
        }
        for b in group {
            if has_aggregate(&b.expr) {
                return Err(AnalysisError::new(
                    GroupByAggregate {
                        sql_expr: pretty(&b.expr, self.ctx.conf.session_zone),
                    },
                    b.expr.origin,
                ));
            }
            if lateral.contains(&semantic_key(&b.expr, &mut memo)) {
                return Err(AnalysisError::new(
                    UnsupportedFeatureLateralColumnAliasInGroupBy {},
                    Origin::UNKNOWN,
                ));
            }
        }
        Ok(())
    }

    /// `GROUP BY n`: the `n`-th select item, unaliased.
    pub(in crate::analyzer) fn group_ordinal(
        &self,
        n: i32,
        origin: Origin,
        items: &[Bound],
        c: &Rel,
    ) -> Result<Bound, AnalysisError> {
        let index = usize::try_from(n)
            .ok()
            .filter(|i| (1..=items.len()).contains(i));
        let Some(i) = index else {
            return Err(AnalysisError::new(
                GroupByPosOutOfRange {
                    index: n.to_string(),
                    size: items.len().to_string(),
                },
                origin,
            ));
        };
        let item = &items[i - 1];
        if item.win.is_some() {
            return Err(super::super::windows::misplaced(
                WindowUse::Other,
                item,
                origin,
            ));
        }
        if !item.lateral.is_empty() {
            return Err(AnalysisError::new(
                UnsupportedFeatureLateralColumnAliasInGroupBy {},
                Origin::UNKNOWN,
            ));
        }
        // `named` aliases every item with an aggregate.
        if let (Some(_), ExprNode::Resolved(ResolvedExpr::Alias { expr, output })) =
            (item.agg, &item.expr.node)
        {
            let text = pretty(&qualified(expr, &c.out), self.ctx.conf.session_zone);
            return Err(AnalysisError::new(
                GroupByPosAggregate {
                    index: n.to_string(),
                    agg_expr: format!("{text} AS {}", quote_if_needed(&output.name)),
                },
                origin,
            ));
        }
        let expr = match &item.expr.node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => Arc::clone(expr),
            _ => Arc::clone(&item.expr),
        };
        Ok(Bound {
            expr,
            ..item.clone()
        })
    }

    /// An aggregate of `items` grouped by `group`, validated. Items that
    /// read lateral column aliases are computed by projections above it,
    /// the aggregate calls and grouping expressions they use computed by
    /// the aggregate, as Spark plans them.
    #[expect(
        clippy::too_many_arguments,
        reason = "group, items and their scope travel together; seven call sites"
    )]
    pub(in crate::analyzer) fn build_aggregate(
        &mut self,
        origin: Origin,
        c: &Rel,
        group: Vec<Bound>,
        items: &[Bound],
        global: bool,
        analytics: Option<Arc<Analytics>>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        // Spark's `RewriteDistinctAggregates`: several distinct argument
        // sets, or a distinct aggregate with a filter, plan as an
        // expansion (one projection per distinct set, told apart by a
        // group id) and two ordinary aggregates. Over grouping sets the
        // aggregation already reads an expansion, which this rewrite
        // does not combine with.
        if analytics.is_some() {
            if Self::needs_distinct_rewrite(items) {
                return Err(unsupported(
                    "DISTINCT aggregates over different arguments with grouping sets",
                    origin,
                ));
            }
        } else if let Some(r) = self.rewrite_distinct(origin, c, &group, items, global, env)? {
            return Ok(r);
        }
        let mut memo = HashMap::new();
        let gkeys: Vec<(u128, ArgType)> = group
            .iter()
            .map(|b| (semantic_key(&b.expr, &mut memo), b.ty.clone()))
            .collect();
        let key_set: HashSet<u128> = gkeys.iter().map(|(k, _)| *k).collect();
        // Items with lateral aliases or windows are computed above the
        // aggregate, over its output.
        let base: Vec<Bound> = items
            .iter()
            .filter(|b| b.lateral.is_empty() && b.win.is_none())
            .cloned()
            .collect();
        self.check_aggregation(&base, &key_set, &mut memo, origin, global)?;
        // Over grouping sets, the grouping attributes are the expansion's.
        let (keys, scope): (Vec<Attr>, Vec<Attr>) = match &analytics {
            Some(a) => (
                a.keys(),
                c.out
                    .iter()
                    .chain(a.exprs.iter().map(|g| &g.attr))
                    .cloned()
                    .collect(),
            ),
            None => (
                group
                    .iter()
                    .filter_map(|b| match &b.expr.node {
                        ExprNode::Resolved(ResolvedExpr::Column(col)) => {
                            c.out.iter().find(|a| a.col.id == col.id).cloned()
                        }
                        _ => None,
                    })
                    .collect(),
                c.out.to_vec(),
            ),
        };
        let mut info = AggInfo {
            input: c.clone(),
            grouping: group.into_iter().map(|b| b.expr).collect(),
            aggregates: base.iter().map(|b| Arc::clone(&b.expr)).collect(),
            out: base.iter().map(|b| out_attr(b, &scope)).collect(),
            keys,
            top: None,
            analytics,
            having: None,
            distinct_rewritten: false,
        };
        if base.len() == items.len() {
            return self.aggregate_rel(origin, info, &[]);
        }
        for b in items.iter().filter(|b| !b.lateral.is_empty()) {
            in_aggregate_call(b)?;
        }
        let lateral: HashSet<AttrId> = items.iter().filter_map(|b| alias_id(&b.expr)).collect();
        let mut top = Vec::new();
        for b in items {
            if b.lateral.is_empty() && b.win.is_none() {
                let a = out_attr(b, &scope);
                top.push(Bound {
                    expr: col_expr(&a.col, b.expr.origin),
                    ..b.clone()
                });
                continue;
            }
            let expr = self.lift(b, &mut info, &gkeys, &lateral, &mut memo, global, origin)?;
            top.push(Bound { expr, ..b.clone() });
        }
        info.top = Some(top.into());
        self.aggregate_rel(origin, info, &[])
    }
}
