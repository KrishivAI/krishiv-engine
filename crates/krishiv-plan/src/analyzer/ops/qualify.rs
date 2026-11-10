// SPDX-License-Identifier: Apache-2.0
//! QUALIFY binding.

use super::*;

impl Resolver<'_> {
    /// A `QUALIFY` filter over the resolved select list `c`: `q` is the
    /// wrapped predicate, `input` its unresolved `Filter` node. The
    /// predicate reads the select output (aliases included) and the query
    /// input's other columns, as Spark's does; window functions in either
    /// go through [`Resolver::windowed`], the same extraction a `HAVING`
    /// with window functions uses. `None` when the shape is not a
    /// `QUALIFY`'s own (a wrapped predicate over anything but its query's
    /// projection or aggregate), so the caller falls back to the
    /// `WHERE`/`HAVING` paths.
    pub(in crate::analyzer) fn qualify(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        c: &Rel,
        q: &Arc<Expr>,
        env: &Arc<Env>,
    ) -> Result<Option<Rel>, AnalysisError> {
        match &input.node {
            PlanNode::Project { input: base, exprs } => {
                let base = self.child(base, env)?;
                let items = self.select_items(&base, exprs, env, "Project")?;
                if items.iter().any(|b| b.agg.is_some()) {
                    // A global aggregate (a `Project` lowering whose items
                    // aggregate): qualify over the aggregate, as below.
                    let agg =
                        self.build_aggregate(p.origin, &base, Vec::new(), &items, true, None, env)?;
                    return Ok(Some(self.qualify_agg(p, &agg, q, None, &agg.out, env)?));
                }
                Ok(Some(self.qualify_project(p, &base, &items, q, env)?))
            }
            PlanNode::Aggregate { .. } => {
                // `GROUP BY` without `HAVING`: `c` is the aggregate.
                if c.agg.is_none() {
                    return Ok(None);
                }
                let out = c.out.to_vec();
                Ok(Some(self.qualify_agg(p, c, q, None, &out, env)?))
            }
            PlanNode::Filter {
                input: mid,
                predicate,
            } => {
                // `HAVING` below `QUALIFY`: qualify over the aggregate the
                // inner filter resolved.
                if qualify_expression(predicate).is_some() {
                    return Ok(None);
                }
                let inner = self.child(mid, env)?;
                let Some(info) = inner.agg.clone() else {
                    return Ok(None);
                };
                let having = info.having.clone().or_else(|| match &c.plan.node {
                    PlanNode::Filter { predicate: h, .. } => Some(Arc::clone(h)),
                    _ => None,
                });
                let Some(having) = having else {
                    return Ok(None);
                };
                let out = c.out.to_vec();
                Ok(Some(self.qualify_agg(
                    p,
                    &inner,
                    q,
                    Some(&having),
                    &out,
                    env,
                )?))
            }
            _ => Ok(None),
        }
    }

    /// `QUALIFY` over a projection: the predicate reads the select output
    /// and the input's other columns. `c` is the resolved projection (its
    /// output is the select list); the filter is rebuilt below so the
    /// predicate's windows extract beside the select list's.
    pub(in crate::analyzer) fn qualify_project(
        &mut self,
        p: &LogicalPlan,
        base: &Rel,
        items: &[Bound],
        q: &Arc<Expr>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        Self::check_qualify_windows(q, items.iter().any(|b| b.win.is_some()), self)?;
        // The select list's windows over the input, as
        // [`Resolver::window_project`] computes them.
        let exprs: Vec<Arc<Expr>> = items.iter().map(|b| Arc::clone(&b.expr)).collect();
        let (below, exprs) = self.windowed(base, &exprs)?;
        let items: Vec<Bound> = items
            .iter()
            .zip(exprs)
            .map(|(b, expr)| Bound { expr, ..b.clone() })
            .collect();
        // The input's columns the select list drops pass through (the
        // predicate and its windows may name them); names the select list
        // outputs shadow the input's, as Spark resolves the output first.
        let cs = self.ctx.conf.case_sensitive;
        let outs: Vec<Attr> = items.iter().map(|b| out_attr(b, &base.out)).collect();
        let pass: Vec<Attr> = base
            .out
            .iter()
            .filter(|a| !outs.iter().any(|o| same(&o.col.name, &a.col.name, cs)))
            .cloned()
            .collect();
        let wide = self.lateral_project(Origin::UNKNOWN, &below, &items, &pass);
        let rels = [&wide];
        let mut cx = Self::cx(&wide.out, &rels, env, "Filter");
        cx.windows = WindowUse::Allowed;
        let qb = self.bind(q, &cx)?;
        boolean_filter(&qb, p.origin)?;
        let (wrel, exprs) = self.windowed(&wide, std::slice::from_ref(&qb.expr))?;
        let pred = exprs.into_iter().next().ok_or_else(|| {
            AnalysisError::new(krishiv_common::error::FailedToParseTooComplex {}, p.origin)
        })?;
        let node = PlanNode::Filter {
            input: Arc::clone(&wrel.plan),
            predicate: pred,
        };
        let mut f = self.make(p.origin, node, wrel.out.to_vec(), &[&wrel]);
        f.widen = Some(Arc::new(Widen::Through { child: wrel }));
        Ok(self.restore(&f, &wide.out[..outs.len()]))
    }

    /// `QUALIFY` over an aggregate `agg` (its output is the select list):
    /// the predicate resolves as a `HAVING` does (names read the output,
    /// then grouping attributes, then the input; missing aggregates and
    /// keys grow it), except windows are allowed and extracted above it.
    /// `having` re-applies a `HAVING` below the windows when one filtered
    /// the aggregate; `select_out` is the select list's output.
    pub(in crate::analyzer) fn qualify_agg(
        &mut self,
        p: &LogicalPlan,
        agg: &Rel,
        q: &Arc<Expr>,
        having: Option<&Arc<Expr>>,
        select_out: &[Attr],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let windows = agg.agg.as_ref().is_some_and(|i| i.has_windows());
        Self::check_qualify_windows(q, windows, self)?;
        let (bounds, grown, _) = self.over_agg(agg, std::slice::from_ref(q), env, "Qualify")?;
        let b = bounds.into_iter().next().ok_or_else(|| {
            AnalysisError::new(krishiv_common::error::FailedToParseTooComplex {}, p.origin)
        })?;
        boolean_filter(&b, p.origin)?;
        // With a `HAVING`, it filters the grown aggregate below the
        // predicate's windows (as [`Resolver::having_below_windows`] does
        // for the select list's).
        let base = match having {
            None => grown,
            Some(h) => {
                let info = grown.agg.as_ref().ok_or_else(|| {
                    AnalysisError::new(krishiv_common::error::FailedToParseTooComplex {}, p.origin)
                })?;
                let info = AggInfo {
                    having: Some(Arc::clone(h)),
                    ..(**info).clone()
                };
                self.aggregate_rel(Origin::UNKNOWN, info, &[])?
            }
        };
        let (wrel, exprs) = self.windowed(&base, std::slice::from_ref(&b.expr))?;
        let pred = exprs.into_iter().next().ok_or_else(|| {
            AnalysisError::new(krishiv_common::error::FailedToParseTooComplex {}, p.origin)
        })?;
        let node = PlanNode::Filter {
            input: Arc::clone(&wrel.plan),
            predicate: pred,
        };
        let mut f = self.make(p.origin, node, wrel.out.to_vec(), &[&wrel]);
        f.widen = Some(Arc::new(Widen::Through { child: wrel }));
        Ok(self.restore(&f, select_out))
    }

    /// Spark's `QUALIFY` checks on the predicate `q`: aggregates are not
    /// allowed in it, and at least one window function must appear in it
    /// or in the select list (`select_windows`).
    pub(in crate::analyzer) fn check_qualify_windows(
        q: &Arc<Expr>,
        select_windows: bool,
        r: &Resolver<'_>,
    ) -> Result<(), AnalysisError> {
        if let Some(a) = first_qualify_aggregate(q, r) {
            let zone = r.ctx.conf.session_zone;
            return Err(AnalysisError::new(
                QualifyAggregateNotAllowed {
                    aggregate_expr: format!("\"{}\"", pretty(&a, zone)),
                },
                a.origin,
            ));
        }
        if !select_windows && !has_qualify_window(q) {
            return Err(AnalysisError::new(
                QualifyRequiresWindowFunction {},
                q.origin,
            ));
        }
        Ok(())
    }
}
