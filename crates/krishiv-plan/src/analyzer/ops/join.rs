// SPDX-License-Identifier: Apache-2.0
//! Join binding (USING and NATURAL, hidden columns, widening) and subquery aliases.

use super::*;

impl Resolver<'_> {
    pub(in crate::analyzer) fn join(
        &mut self,
        p: &LogicalPlan,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: JoinKind,
        condition: &JoinCondition,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        self.join_with(p, left, right, kind, condition, None, env)
    }

    /// A join, or with `as_of` an as-of join: the same inputs, condition, `USING` expansion and
    /// output (Spark's `ResolveAsOfJoin` reuses the join's `USING` resolution), the match bound
    /// over the two inputs.
    #[expect(
        clippy::too_many_arguments,
        reason = "the join's parts, as the plan node holds them"
    )]
    pub(in crate::analyzer) fn join_with(
        &mut self,
        p: &LogicalPlan,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: JoinKind,
        condition: &JoinCondition,
        as_of: Option<&AsOfMatch>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let lateral = matches!(kind, JoinKind::LateralInner | JoinKind::LateralLeft);
        let l = if lateral {
            self.rel(left, env)?
        } else {
            self.child(left, env)?
        };
        // `JOIN LATERAL`: the right side reads the left side's columns as an
        // enclosing query's. The joined relation reads none of that level.
        let kind = match kind {
            JoinKind::LateralInner => JoinKind::Inner,
            JoinKind::LateralLeft => JoinKind::LeftOuter,
            k => k,
        };
        let mut r = if lateral {
            let mut outer = env.outer.clone();
            outer.push(super::super::resolve::Level {
                scope: Arc::clone(&l.out),
                lateral: true,
                hidden: Arc::clone(&l.hidden),
                fallback: Arc::from(Vec::new()),
                having: None,
            });
            let inner = self.env(env.ctes.clone(), env.ctes_id, outer);
            let mut r = self.rel(right, &inner)?;
            r.outer = r.outer.filter(|k| *k < env.outer.len());
            // Spark's `CheckAnalysis`: a lateral subquery holds no
            // nondeterministic expression.
            let mut nondeterministic = false;
            let _ = crate::walk::visit_distinct(crate::walk::Node::Plan(&r.plan), |n| {
                if let crate::walk::Node::Expr(e) = n
                    && matches!(
                        &e.node,
                        ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                            deterministic: false,
                            ..
                        }))
                    )
                {
                    nondeterministic = true;
                    return std::ops::ControlFlow::Break(());
                }
                std::ops::ControlFlow::<()>::Continue(())
            });
            if nondeterministic && !at_most_one_row(&l.plan) {
                return Err(AnalysisError::new(
                    UnsupportedSubqueryExpressionCategoryNonDeterministicLateralSubqueries {
                        tree_node: r.plan.to_string(),
                    },
                    p.origin,
                ));
            }
            r
        } else {
            self.child(right, env)?
        };
        // `JOIN LATERAL f(..)` of a Python UDTF: a generator over the left side.
        if lateral && let Some(g) = self.lateral_udtf(p.origin, &l, &r, kind, condition, env)? {
            return Ok(g);
        }
        if let (Some(a), Some(b)) = (l.ids, r.ids)
            && a.0 <= b.1
            && b.0 <= a.1
        {
            r = if r.outer.is_some_and(|k| k < env.outer.len()) {
                self.correlated_instance(&r)?
            } else {
                self.instance(&r)?
            };
        }
        let nullable = |a: &Attr, yes: bool| Attr {
            col: BoundColumn {
                nullable: a.col.nullable || yes,
                ..a.col.clone()
            },
            ..a.clone()
        };
        let (ln, rn) = match kind {
            JoinKind::LeftOuter | JoinKind::Single => (false, true),
            JoinKind::RightOuter => (true, false),
            JoinKind::FullOuter => (true, true),
            _ => (false, false),
        };
        let mut out: Vec<Attr> = l.out.iter().map(|a| nullable(a, ln)).collect();
        if let JoinKind::Existence(id) = kind {
            out.push(Attr {
                col: JoinKind::existence_column(id),
                qual: Arc::from(Vec::new()),
                source: None,
                duplicate: false,
            });
        } else if !matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti) {
            out.extend(r.out.iter().map(|a| nullable(a, rn)));
        }
        let rels = [&l, &r];
        // Spark binds the match before its `CheckAnalysis` checks the condition.
        let as_of = match as_of {
            Some(m) => Some(Arc::new(self.as_of_match(p.origin, m, &l, &r, env)?)),
            None => None,
        };
        let mut keys = None;
        let mut kind = kind;
        let condition = match condition {
            JoinCondition::None => JoinCondition::None,
            JoinCondition::Natural => {
                // `NATURAL JOIN`: the common column names of the two
                // inputs, then the `USING` path, so output order and
                // outer-join coalescing are exactly the `USING` ones. With
                // no common columns the join is a cross join, as Spark's
                // is.
                if matches!(
                    kind,
                    JoinKind::LeftSemi | JoinKind::LeftAnti | JoinKind::Existence(_)
                ) {
                    // Spark fails here with a bare `requirement failed:
                    // Unsupported natural join type LeftSemi` (no error class).
                    return Err(unsupported(
                        "NATURAL LEFT SEMI and LEFT ANTI joins",
                        p.origin,
                    ));
                }
                let commons = Self::common_columns(&l, &r, self.ctx.conf.case_sensitive);
                if commons.is_empty() {
                    if kind == JoinKind::Inner {
                        kind = JoinKind::Cross;
                    }
                    JoinCondition::None
                } else {
                    let (cond, k) = self.using(&commons, &l, &r, env, p.origin, true)?;
                    keys = Some(k);
                    JoinCondition::On(cond)
                }
            }
            JoinCondition::On(e) => {
                let scope: Vec<Attr> = l.out.iter().chain(r.out.iter()).cloned().collect();
                let b = self.bind(e, &Self::cx(&scope, &rels, env, "Join"))?;
                if b.ty.data_type != Some(krishiv_types::DataType::Boolean) {
                    return Err(AnalysisError::new(
                        JoinConditionIsNotBooleanType {
                            join_condition: format!(
                                "\"{}\"",
                                pretty(&b.shown(), self.ctx.conf.session_zone)
                            ),
                            condition_type: quoted_type(b.ty.data_type.as_ref()),
                        },
                        p.origin,
                    ));
                }
                // Spark: the condition of a lateral join holds no IN or EXISTS
                // subquery (the bound plan does not remember the LATERAL).
                if lateral && holds_in_or_exists(&b.expr) {
                    return Err(AnalysisError::new(
                        UnsupportedSubqueryExpressionCategoryUnsupportedInExistsSubquery {
                            tree_node: p.to_string(),
                        },
                        p.origin,
                    ));
                }
                // Spark's `CheckAnalysis`: a join condition holds no
                // nondeterministic expression.
                if lateral && !super::super::sqlfn::is_deterministic(&b.expr) {
                    return Err(AnalysisError::new(
                        UnsupportedSubqueryExpressionCategoryLateralJoinConditionNonDeterministic {
                            condition: pretty(&b.shown(), self.ctx.conf.session_zone),
                        },
                        p.origin,
                    ));
                }
                if !super::super::sqlfn::is_deterministic(&b.expr) {
                    return Err(AnalysisError::new(
                        InvalidNonDeterministicExpressions {
                            sql_exprs: format!(
                                "\"{}\"",
                                pretty(&b.shown(), self.ctx.conf.session_zone)
                            ),
                        },
                        p.origin,
                    ));
                }
                JoinCondition::On(b.expr)
            }
            JoinCondition::Using(cols) => {
                let (cond, k) = self.using(cols, &l, &r, env, p.origin, false)?;
                keys = Some(k);
                JoinCondition::On(cond)
            }
        };
        // Columns an earlier `USING` join merged away stay reachable above this join, and its
        // condition may read them (`a NATURAL JOIN b JOIN c ON b.k = c.k`): the inputs are
        // revealed, the join runs over the wide inputs and a projection restores the output.
        // The `_metadata` of a file table an `ON` condition or an as-of operand reads is widened
        // into its input first (Spark's `AddMetadataColumns` adds it to the side's output).
        let (l0, r0) = (l.clone(), r.clone());
        let read: Vec<&Arc<Expr>> = match &condition {
            JoinCondition::On(c) => Some(c),
            _ => None,
        }
        .into_iter()
        .chain(as_of.iter().flat_map(|m| [&m.left, &m.right]))
        .collect();
        let (l, r) = if read.is_empty() || matches!(kind, JoinKind::Existence(_)) {
            (l0.clone(), r0.clone())
        } else {
            (
                self.reveal_exprs(&l0, read.iter().copied())?,
                self.reveal_exprs(&r0, read.iter().copied())?,
            )
        };
        let wide = if !matches!(kind, JoinKind::Existence(_)) {
            let (wl, wr) = (self.reveal_hidden(&l)?, self.reveal_hidden(&r)?);
            (wl.out.len() != l0.out.len() || wr.out.len() != r0.out.len()).then_some((wl, wr))
        } else {
            None
        };
        let (jl, jr) = wide.as_ref().map_or((&l, &r), |(a, b)| (a, b));
        let node = match as_of {
            Some(as_of) => PlanNode::AsOfJoin {
                left: Arc::clone(&jl.plan),
                right: Arc::clone(&jr.plan),
                kind,
                condition,
                as_of,
            },
            None => PlanNode::Join {
                left: Arc::clone(&jl.plan),
                right: Arc::clone(&jr.plan),
                kind,
                condition,
            },
        };
        let Some(keys) = keys else {
            let Some((wl, wr)) = &wide else {
                let mut j = self.make(p.origin, node, out, &[&l, &r]);
                j.hidden = Self::carried_hidden(&l, &r, kind, (ln, rn)).into();
                j.widen = Self::join_widen(&l, &r, kind, &j, (ln, rn));
                return Ok(j);
            };
            let mut wide_out: Vec<Attr> = wl.out.iter().map(|a| nullable(a, ln)).collect();
            if !matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti) {
                wide_out.extend(wr.out.iter().map(|a| nullable(a, rn)));
            }
            let mut j = self.make(p.origin, node, wide_out, &[wl, wr]);
            j.widen = Self::join_widen(wl, wr, kind, &j, (ln, rn));
            let mut res = self.restore(&j, &out);
            res.hidden = Self::carried_hidden(wl, wr, kind, (ln, rn)).into();
            return Ok(res);
        };
        // Over wide inputs (an earlier `USING` join's merged-away columns revealed), the join
        // keeps them reachable; the projection shows only what the narrow join would.
        let (j, visible, hidden_in) = if let Some((wl, wr)) = &wide {
            let mut wide_out: Vec<Attr> = wl.out.iter().map(|a| nullable(a, ln)).collect();
            if !matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti) {
                wide_out.extend(wr.out.iter().map(|a| nullable(a, rn)));
            }
            let visible: HashSet<AttrId> = out.iter().map(|a| a.col.id).collect();
            let mut j = self.make(Origin::UNKNOWN, node, wide_out, &[wl, wr]);
            j.widen = Self::join_widen(wl, wr, kind, &j, (ln, rn));
            (j, Some(visible), true)
        } else {
            let mut j = self.make(Origin::UNKNOWN, node, out, &[&l, &r]);
            j.widen = Self::join_widen(&l, &r, kind, &j, (ln, rn));
            (j, None, false)
        };
        let mut res = self.using_project(p.origin, &j, kind, &keys, env, visible.as_ref())?;
        let shown: HashSet<AttrId> = res.out.iter().map(|a| a.col.id).collect();
        let hidden: Vec<Attr> = if hidden_in {
            // A file table's `_metadata` nobody asked for yet is in no
            // output, so it travels from the inputs' hidden lists.
            l.hidden
                .iter()
                .map(|a| nullable(a, ln))
                .chain(r.hidden.iter().map(|a| nullable(a, rn)))
                .filter(is_generated_metadata)
                .chain(j.out.iter().filter(|a| !shown.contains(&a.col.id)).cloned())
                .collect()
        } else {
            l.hidden
                .iter()
                .map(|a| nullable(a, ln))
                .chain(r.hidden.iter().map(|a| nullable(a, rn)))
                .chain(j.out.iter().filter(|a| !shown.contains(&a.col.id)).cloned())
                .collect()
        };
        res.hidden = hidden.into();
        Ok(res)
    }

    /// How `j`, a join of `l` and `r`, gets back a `_metadata` column of
    /// either input: `None` when neither has one.
    pub(in crate::analyzer) fn join_widen(
        l: &Rel,
        r: &Rel,
        kind: JoinKind,
        j: &Rel,
        nullable: (bool, bool),
    ) -> Option<Arc<Widen>> {
        let PlanNode::Join { condition, .. } = &j.plan.node else {
            return None;
        };
        let meta = |c: &Rel| c.hidden.iter().any(is_generated_metadata);
        (meta(l) || meta(r)).then(|| {
            Arc::new(Widen::Join {
                left: l.clone(),
                right: r.clone(),
                kind,
                condition: condition.clone(),
                nullable,
            })
        })
    }

    /// `c` with the columns its `hidden` list names in its output, as far as they can be reached.
    pub(in crate::analyzer) fn reveal_hidden(&mut self, c: &Rel) -> Result<Rel, AnalysisError> {
        if c.hidden.is_empty() {
            return Ok(c.clone());
        }
        // The `_metadata` of a file table is widened in when a query reads it.
        let exprs: Vec<Arc<Expr>> = c
            .hidden
            .iter()
            .filter(|a| !is_generated_metadata(a))
            .map(|a| col_expr(&a.col, Origin::UNKNOWN))
            .collect();
        self.reveal_exprs(c, exprs.iter())
    }

    /// The hidden columns a join above `l` and `r` passes on, nullable on the outer side; only
    /// those the (revealed) inputs output can be read again.
    pub(in crate::analyzer) fn carried_hidden(
        l: &Rel,
        r: &Rel,
        kind: JoinKind,
        nullable_sides: (bool, bool),
    ) -> Vec<Attr> {
        let side = |rel: &Rel, null: bool| -> Vec<Attr> {
            rel.hidden
                .iter()
                .filter(|a| {
                    is_generated_metadata(a) || rel.out.iter().any(|o| o.col.id == a.col.id)
                })
                .map(|a| Attr {
                    col: BoundColumn {
                        nullable: a.col.nullable || null,
                        ..a.col.clone()
                    },
                    ..a.clone()
                })
                .collect()
        };
        let mut out = side(l, nullable_sides.0);
        if !matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti) {
            out.extend(side(r, nullable_sides.1));
        }
        out
    }

    /// The common column names of a `NATURAL JOIN`'s inputs, as Spark's
    /// `canonicalizedIntersect` computes them: in the left input's order,
    /// each name as many times as both sides output it (a name the left
    /// outputs twice and the right once is common once; resolving a name
    /// always takes the first match, as Spark's key resolution does).
    /// `case_sensitive` applies.
    pub(in crate::analyzer) fn common_columns(
        l: &Rel,
        r: &Rel,
        case_sensitive: bool,
    ) -> Vec<String> {
        let mut right: Vec<&str> = r.out.iter().map(|a| a.col.name.as_str()).collect();
        let mut commons: Vec<String> = Vec::new();
        for a in l.out.iter() {
            if let Some(i) = right
                .iter()
                .position(|n| same(n, &a.col.name, case_sensitive))
            {
                right.remove(i);
                commons.push(a.col.name.clone());
            }
        }
        commons
    }

    /// The condition of `USING (cols)`, and each column's two attributes.
    /// A name resolves to the first match on each side when `lenient`
    /// (a repeated `NATURAL JOIN` name pairs the first matches again, as
    /// Spark's key resolution does); otherwise several matches are
    /// ambiguous.
    pub(in crate::analyzer) fn using(
        &mut self,
        cols: &[String],
        l: &Rel,
        r: &Rel,
        env: &Arc<Env>,
        origin: Origin,
        lenient: bool,
    ) -> Result<(Arc<Expr>, Vec<Keys>), AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let side = |rel: &Rel, which: &str, name: &String| {
            let mut hits: Vec<&Attr> = Vec::new();
            for a in rel.out.iter() {
                if same(&a.col.name, name, cs) && !hits.iter().any(|h| h.col.id == a.col.id) {
                    hits.push(a);
                }
            }
            // Spark resolves the key through `getCandidatesForResolution`:
            // a union input's repeat of a column does not count.
            if hits.len() > 1 {
                hits.retain(|a| !a.duplicate);
            }
            match hits[..] {
                [a] => Ok(a.clone()),
                [a, ..] if lenient => Ok(a.clone()),
                [] => Err(AnalysisError::new(
                    UnresolvedUsingColumnForJoin {
                        col_name: quoted(std::slice::from_ref(name)),
                        side: which.to_owned(),
                        suggestion: names(&rel.out, false),
                    },
                    Origin::UNKNOWN,
                )),
                _ => Err(AnalysisError::new(
                    AmbiguousReference {
                        name: quoted(std::slice::from_ref(name)),
                        reference_names: format!("[{}]", names(&rel.out, false)),
                    },
                    Origin::UNKNOWN,
                )),
            }
        };
        let mut keys = Vec::new();
        let mut cond: Option<Arc<Expr>> = None;
        for name in cols {
            let (la, ra) = (side(l, "left", name)?, side(r, "right", name)?);
            // Spark's errors cite the join for its keys' comparison.
            let eq = self.call(
                "==",
                vec![
                    col_expr(&la.col, Origin::UNKNOWN),
                    col_expr(&ra.col, Origin::UNKNOWN),
                ],
                env,
                origin,
            )?;
            cond = Some(match cond {
                None => eq,
                Some(c) => self.call("and", vec![c, eq], env, Origin::UNKNOWN)?,
            });
            keys.push((la, ra));
        }
        let cond = cond.ok_or_else(|| unsupported("USING ()", Origin::UNKNOWN))?;
        Ok((cond, keys))
    }

    /// A resolved call of `name` on resolved arguments.
    pub(in crate::analyzer) fn call(
        &mut self,
        name: &str,
        args: Vec<Arc<Expr>>,
        env: &Arc<Env>,
        origin: Origin,
    ) -> Result<Arc<Expr>, AnalysisError> {
        let call = Arc::new(
            Expr::new(ExprNode::Function(FunctionCall {
                name: name.to_owned(),
                qualifier: Vec::new(),
                arg_names: Vec::new(),
                args,
                distinct: false,
                filter: None,
                within_group: Vec::new(),
            }))
            .with_origin(origin),
        );
        Ok(self.bind(&call, &Self::cx(&[], &[], env, "Join"))?.expr)
    }

    /// The output of a `USING` join: the keys (from the side that keeps
    /// all rows; `coalesce` for a full join), then the other columns.
    pub(in crate::analyzer) fn using_project(
        &mut self,
        origin: Origin,
        j: &Rel,
        kind: JoinKind,
        keys: &[(Attr, Attr)],
        env: &Arc<Env>,
        visible: Option<&HashSet<AttrId>>,
    ) -> Result<Rel, AnalysisError> {
        // A key as the join outputs it (nullable as the join made it). A
        // semi or anti join outputs the left side only, its keys included
        // (Spark), so only the left key is looked up there.
        let find = |a: &Attr| {
            j.out
                .iter()
                .find(|x| x.col.id == a.col.id)
                .cloned()
                .unwrap_or_else(|| a.clone())
        };
        let mut exprs = Vec::new();
        let mut out = Vec::new();
        let mut key_ids = HashSet::new();
        for (la, ra) in keys {
            key_ids.extend([la.col.id, ra.col.id]);
            let l = find(la);
            match kind {
                JoinKind::RightOuter => {
                    let r = find(ra);
                    exprs.push(col_expr(&r.col, Origin::UNKNOWN));
                    out.push(r);
                }
                JoinKind::FullOuter => {
                    let r = find(ra);
                    let both = vec![
                        col_expr(&l.col, Origin::UNKNOWN),
                        col_expr(&r.col, Origin::UNKNOWN),
                    ];
                    let value = self.call("coalesce", both, env, Origin::UNKNOWN)?;
                    let named = Arc::new(Expr::new(ExprNode::Alias {
                        expr: value,
                        name: l.col.name.clone(),
                    }));
                    let b = self.bind(&named, &Self::cx(&[], &[], env, "Project"))?;
                    out.push(out_attr(&b, &[]));
                    exprs.push(b.expr);
                }
                _ => {
                    exprs.push(col_expr(&l.col, Origin::UNKNOWN));
                    out.push(l);
                }
            }
        }
        for a in j.out.iter().filter(|a| {
            !key_ids.contains(&a.col.id) && visible.is_none_or(|v| v.contains(&a.col.id))
        }) {
            exprs.push(col_expr(&a.col, Origin::UNKNOWN));
            out.push(a.clone());
        }
        // An `ORDER BY` may name a column the merge dropped.
        Ok(self.projection(origin, j, exprs, out))
    }

    pub(in crate::analyzer) fn subquery_alias(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        alias: &str,
        columns: &[String],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let inner = if columns.is_empty() {
            c
        } else {
            self.rename_by_position(p, input, c, columns, true)?
        };
        let qual: Arc<[String]> = alias
            .split(crate::plan::ALIAS_PART_SEP)
            .map(str::to_owned)
            .collect();
        let out = inner
            .out
            .iter()
            .map(|a| Attr {
                qual: Arc::clone(&qual),
                ..a.clone()
            })
            .collect();
        let node = PlanNode::SubqueryAlias {
            input: Arc::clone(&inner.plan),
            alias: alias.to_owned(),
            columns: Vec::new(),
        };
        let hidden: Vec<Attr> = inner
            .hidden
            .iter()
            .filter(|a| is_generated_metadata(a))
            .map(|a| Attr {
                qual: Arc::clone(&qual),
                ..a.clone()
            })
            .collect();
        let mut rel = self.make(p.origin, node, out, &[&inner]);
        // A table's `_metadata` column is reachable by the alias.
        if !hidden.is_empty() {
            rel.hidden = hidden.into();
            rel.widen = Some(Arc::new(Widen::Through { child: inner }));
        }
        Ok(rel)
    }

    /// `c` with its columns renamed by position and its qualifiers dropped (Spark's
    /// `UnresolvedSubqueryColumnAliases`: a derived table's column list and `toDF`).
    pub(in crate::analyzer) fn rename_by_position(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        c: Rel,
        columns: &[String],
        sql_table_function_form: bool,
    ) -> Result<Rel, AnalysisError> {
        if columns.len() != c.out.len() {
            // Only the SQL `FROM func(..) t(a, b)` form is Spark's `UnresolvedTVFAliases`; a
            // derived table's column list and `toDF` always raise the arity error below.
            let generator = if sql_table_function_form {
                generator_relation_name(input)
            } else {
                None
            };
            if let Some(func) = generator {
                return Err(AnalysisError::new(
                    NumTableValueAliasesMismatch {
                        func_name: format!("`{func}`"),
                        aliases_num: columns.len().to_string(),
                        out_cols_num: c.out.len().to_string(),
                    },
                    p.origin,
                ));
            }
            return Err(AnalysisError::new(
                AssignmentArityMismatch {
                    num_target: c.out.len().to_string(),
                    num_expr: columns.len().to_string(),
                },
                p.origin,
            ));
        }
        let mut exprs = Vec::new();
        let mut out = Vec::new();
        for (a, name) in c.out.iter().zip(columns) {
            let output = BoundColumn {
                id: self.fresh(),
                name: name.clone(),
                ..a.col.clone()
            };
            self.carry_column_meta(a.col.id, output.id);
            exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: col_expr(&a.col, Origin::UNKNOWN),
                    output: output.clone(),
                },
            ))));
            out.push(Attr {
                col: output,
                qual: no_qual(),
                source: a.source.clone(),
                // Spark's `Alias` keeps its child's metadata.
                duplicate: a.duplicate,
            });
        }
        let node = PlanNode::Project {
            input: Arc::clone(&c.plan),
            exprs: exprs.clone(),
        };
        let mut renamed = self.make(Origin::UNKNOWN, node, out, &[&c]);
        // A file table's `_metadata` stays reachable through the renaming.
        let meta: Vec<Attr> = c
            .hidden
            .iter()
            .filter(|a| is_generated_metadata(a))
            .cloned()
            .collect();
        if !meta.is_empty() {
            renamed.hidden = meta.into();
            renamed.widen = Some(Arc::new(Widen::Project { input: c, exprs }));
        }
        Ok(renamed)
    }

    /// A [`PlanNode::ColumnNames`] as a relation of its own: `renamed["x"]` resolves against it,
    /// also when the operation changed nothing and returned the child. Out of line so the
    /// resolver's dispatch frame, on its recursion path, stays small.
    #[inline(never)]
    pub(in crate::analyzer) fn column_names_node(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        op: &crate::ColumnNamesOp,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let mut r = self.column_names(p, input, op, env)?;
        if let Some(id) = p.origin.plan_id {
            let out = Arc::clone(&r.out);
            Arc::make_mut(&mut r.carriers).insert(id, vec![out]);
        }
        Ok(r)
    }

    /// A [`PlanNode::ColumnNames`]: the input's columns by name, as Spark's `toDF`,
    /// `withColumnsRenamed` and `drop(names)` resolve them. The node is not kept.
    pub(in crate::analyzer) fn column_names(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        op: &crate::ColumnNamesOp,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        use crate::ColumnNamesOp;
        let c = self.child(input, env)?;
        let cs = self.ctx.conf.case_sensitive;
        let same = |a: &str, b: &str| crate::analyzer::bind::same(a, b, cs);
        match op {
            ColumnNamesOp::ToDf(names) => self.rename_by_position(p, input, c, names, false),
            ColumnNamesOp::Rename(pairs) => {
                let mut names: Vec<String> = c.out.iter().map(|a| a.col.name.clone()).collect();
                let mut changed = vec![false; names.len()];
                for (from, to) in pairs {
                    for (n, ch) in names.iter_mut().zip(changed.iter_mut()) {
                        if same(n, from) {
                            n.clone_from(to);
                            *ch = true;
                        }
                    }
                }
                let exprs: Vec<Arc<Expr>> = c
                    .out
                    .iter()
                    .zip(names.iter().zip(&changed))
                    .map(|(a, (n, ch))| {
                        if *ch {
                            Arc::new(Expr::new(ExprNode::Alias {
                                expr: self.pass_expr(&a.col),
                                name: n.clone(),
                            }))
                        } else {
                            self.pass_expr(&a.col)
                        }
                    })
                    .collect();
                self.project_items(p, &c, &exprs, env)
            }
            ColumnNamesOp::Drop { names, columns } => {
                // A column reference resolves like any other (Spark's
                // `ResolveDataFrameDropColumns`): ambiguous is an error, not found is ignored,
                // found drops that one attribute by id.
                let mut dropped_ids = Vec::new();
                for name in columns {
                    if let Some((a, _)) =
                        self.find(&c.out, std::slice::from_ref(name), p.origin, false)?
                    {
                        dropped_ids.push(a.col.id);
                    }
                }
                let dropped = |a: &Attr| {
                    dropped_ids.contains(&a.col.id) || names.iter().any(|d| same(&a.col.name, d))
                };
                if !c.out.iter().any(&dropped) {
                    return Ok(c);
                }
                let exprs: Vec<Arc<Expr>> = c
                    .out
                    .iter()
                    .filter(|a| !dropped(a))
                    .map(|a| self.pass_expr(&a.col))
                    .collect();
                self.project_items(p, &c, &exprs, env)
            }
            ColumnNamesOp::DropNa {
                subset,
                min_non_nulls,
            } => self.drop_na(p, c, subset, *min_non_nulls, env),
            ColumnNamesOp::FillNa { cols, values } => self.fill_na(p, c, cols, values, env),
            ColumnNamesOp::Replace { cols, pairs } => self.replace_na(p, c, cols, pairs, env),
        }
    }
}
