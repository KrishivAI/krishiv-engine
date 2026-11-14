// SPDX-License-Identifier: Apache-2.0
//! Lowering the select list, GROUP BY and the FROM clause: relations, joins, PIVOT, UNPIVOT and sampling.

use super::*;

impl Lowerer {
    pub(super) fn select_list(&mut self, s: &Select, input: LogicalPlan) -> R<LogicalPlan> {
        let mut plan = self.lateral_views(input, &s.lateral_views)?;
        if let Some(w) = &s.selection {
            let predicate = arc(self.expr(w)?);
            plan = self.node(
                PlanNode::Filter {
                    input: arc(plan),
                    predicate,
                },
                s.span,
            );
        }
        let mut items = Vec::new();
        // A select list with an aggregate call is an aggregate even without
        // `GROUP BY`: its generators run above it, in the analyzer.
        let aggregates = s
            .projection
            .iter()
            .any(|i| Self::has_aggregate_call(&i.expr));
        let has_generator = s
            .projection
            .iter()
            .any(|i| Self::generator_of(&i.expr).is_some());
        // With GROUP BY or HAVING the analyzer extracts the generators from the aggregate
        // (Spark's `ExtractGenerator`).
        let mut stars = std::collections::HashMap::new();
        if s.group_by.is_none() && s.having.is_none() && !(aggregates && has_generator) {
            for index in generator_order::chain(&s.projection) {
                let taken =
                    std::mem::replace(&mut plan, LogicalPlan::new(PlanNode::OneRowRelation));
                let (generated, star) =
                    self.select_generator(taken, &s.projection[index], index)?;
                plan = generated;
                stars.insert(index, star);
            }
        }
        for (index, item) in s.projection.iter().enumerate() {
            items.push(match stars.remove(&index) {
                Some(star) => star,
                None => self.select_item(item)?,
            });
        }
        // `spark.sql.legacy.parser.havingWithoutGroupByAsWhere`: a `HAVING` with no `GROUP BY` over
        // a select list with no aggregate call filters the rows (as `WHERE` does; aggregates in
        // it are `INVALID_WHERE_CONDITION`) instead of making the query one global group.
        let having_filter = s.having_as_where
            && s.group_by.is_none()
            && s.having.is_some()
            && !aggregates
            && !has_generator;
        if s.group_by.is_none()
            && (s.having.is_none() || having_filter)
            && !(aggregates && has_generator)
        {
            plan = self.node(
                PlanNode::Project {
                    input: arc(plan),
                    exprs: items,
                },
                s.span,
            );
            if let Some(h) = s.having.as_ref().filter(|_| having_filter) {
                let predicate = self.having_predicate(h)?;
                plan = self.node(
                    PlanNode::Filter {
                        input: arc(plan),
                        predicate,
                    },
                    s.span,
                );
            }
        } else {
            let grouping = match &s.group_by {
                Some(g) => self.grouping(g)?,
                None => Grouping::GroupBy(Vec::new()),
            };
            plan = self.node(
                PlanNode::Aggregate {
                    input: arc(plan),
                    grouping,
                    aggregates: items,
                },
                s.span,
            );
            if let Some(h) = &s.having {
                let predicate = self.having_predicate(h)?;
                plan = self.node(
                    PlanNode::Filter {
                        input: arc(plan),
                        predicate,
                    },
                    s.span,
                );
            }
        }
        if let Some(q) = &s.qualify {
            // `QUALIFY`, as Spark's `AstBuilder` shapes it: a filter with
            // the qualify predicate over the select list's projection (the
            // analyzer resolves it like `HAVING`, extracting window
            // functions the same way). The predicate is wrapped so the
            // analyzer tells it apart from `WHERE` and `HAVING`; the
            // wrapper is stripped during analysis and never reaches the
            // executor.
            let inner = arc(self.expr(q)?);
            let predicate = arc(self.ex(
                ExprNode::Function(FunctionCall {
                    qualifier: Vec::new(),
                    name: "qualify".to_owned(),
                    args: vec![inner],
                    arg_names: Vec::new(),
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
                }),
                q.span,
            ));
            plan = self.node(
                PlanNode::Filter {
                    input: arc(plan),
                    predicate,
                },
                s.span,
            );
        }
        Ok(plan)
    }

    /// `GROUP BY`, as Spark's `AstBuilder` shapes it: plain keys, `WITH
    /// ROLLUP` / `WITH CUBE` over them, `GROUPING SETS (..)` after them (the
    /// keys only fix the grouping expressions), or items that are
    /// `ROLLUP`, `CUBE` and `GROUPING SETS` among keys, whose grouping sets
    /// are the cross product of the items' (a key is a set of itself).
    #[inline(never)]
    pub(super) fn grouping(&mut self, g: &GroupBy) -> R<Grouping> {
        let mut plain = Vec::with_capacity(g.items.len());
        for item in &g.items {
            if let GroupItemKind::Expr(e) = &item.kind {
                plain.push(arc(self.expr(e)?));
            }
        }
        let all_plain = plain.len() == g.items.len();
        match (&g.with, all_plain) {
            (None, true) => {
                // `GROUP BY ALL` reads as a column named `all` here; the
                // analyzer expands it (or resolves the column). The key
                // keeps the whole clause's span, so the analyzer's
                // `UNRESOLVED_ALL_IN_GROUP_BY` cites `group by all`.
                if plain.len() == 1 && Self::is_group_by_all(&g.items[0]) {
                    let node = plain[0].node.clone();
                    plain[0] = arc(IrExpr::new(node).with_origin(self.origin(g.span)));
                }
                return Ok(Grouping::GroupBy(plain));
            }
            (Some(GroupWith::Rollup), true) => return Ok(Grouping::Rollup(plain)),
            (Some(GroupWith::Cube), true) => return Ok(Grouping::Cube(plain)),
            (Some(GroupWith::GroupingSets(sets)), true) => {
                let sets = sets.iter().map(|s| self.set(s)).collect::<R<Vec<_>>>()?;
                return Ok(Grouping::GroupingSets { exprs: plain, sets });
            }
            (Some(_), false) => {
                return Err(unsupported(
                    "ROLLUP, CUBE and GROUPING SETS items with WITH ROLLUP, WITH CUBE or GROUPING SETS",
                    g.span,
                ));
            }
            (None, false) => {}
        }
        let mut exprs = Vec::new();
        let mut product: Vec<Vec<Arc<IrExpr>>> = vec![Vec::new()];
        for item in &g.items {
            let sets = self.item_sets(item, &mut exprs)?;
            if product.len().saturating_mul(sets.len()) > MAX_GROUPING_SETS {
                return Err(unsupported(
                    &format!("more than {MAX_GROUPING_SETS} grouping sets"),
                    g.span,
                ));
            }
            product = product
                .iter()
                .flat_map(|head| {
                    sets.iter()
                        .map(move |tail| head.iter().chain(tail).cloned().collect::<Vec<_>>())
                })
                .collect();
        }
        Ok(Grouping::GroupingSets {
            exprs,
            sets: product,
        })
    }

    /// The grouping sets one `GROUP BY` item stands for; the expressions it
    /// names are appended to `written` in order.
    pub(super) fn item_sets(
        &mut self,
        item: &GroupItem,
        written: &mut Vec<Arc<IrExpr>>,
    ) -> R<Vec<Vec<Arc<IrExpr>>>> {
        Ok(match &item.kind {
            GroupItemKind::Named(_) => return Err(unsupported("named grouping items", item.span)),
            GroupItemKind::Expr(e) => {
                let e = arc(self.expr(e)?);
                written.push(Arc::clone(&e));
                vec![vec![e]]
            }
            GroupItemKind::Set(s) => {
                let set = self.set(s)?;
                written.extend(set.iter().cloned());
                vec![set]
            }
            GroupItemKind::GroupingSets(items) => {
                let mut sets = Vec::new();
                let span = item.span;
                for inner in items {
                    sets.extend(self.deeper(|l| l.item_sets(inner, &mut *written))?);
                    if sets.len() > MAX_GROUPING_SETS {
                        return Err(unsupported(
                            &format!("more than {MAX_GROUPING_SETS} grouping sets"),
                            span,
                        ));
                    }
                }
                sets
            }
            GroupItemKind::Rollup(elements) | GroupItemKind::Cube(elements) => {
                let rollup = matches!(item.kind, GroupItemKind::Rollup(_));
                let mut parts = Vec::with_capacity(elements.len());
                for element in elements {
                    let part = self.set(element)?;
                    written.extend(part.iter().cloned());
                    parts.push(part);
                }
                // `ROLLUP (a, b)`: (a, b), (a), (); `CUBE (a, b)`: every
                // subset of the elements.
                let n = parts.len();
                if !rollup && n >= usize::BITS as usize - 1
                    || !rollup && (1_usize << n) > MAX_GROUPING_SETS
                {
                    return Err(unsupported(
                        &format!("more than {MAX_GROUPING_SETS} grouping sets"),
                        item.span,
                    ));
                }
                let chosen: Vec<Vec<usize>> = if rollup {
                    (0..=n).rev().map(|k| (0..k).collect()).collect()
                } else {
                    (0..1_usize << n)
                        .rev()
                        .map(|m| (0..n).filter(|i| m >> (n - 1 - i) & 1 == 1).collect())
                        .collect()
                };
                chosen
                    .into_iter()
                    .map(|c| {
                        c.into_iter()
                            .flat_map(|i| parts[i].iter().cloned())
                            .collect()
                    })
                    .collect()
            }
        })
    }

    /// The expressions of one grouping set.
    pub(super) fn set(&mut self, s: &GroupingSet) -> R<Vec<Arc<IrExpr>>> {
        s.exprs.iter().map(|e| Ok(arc(self.expr(e)?))).collect()
    }

    /// Whether a `GROUP BY` item is a bare `ALL` (`GROUP BY ALL`), written as
    /// a column of that name by the parser (see `column_named_all`).
    pub(super) fn is_group_by_all(item: &GroupItem) -> bool {
        let GroupItemKind::Expr(e) = &item.kind else {
            return false;
        };
        let ExprKind::Column(parts) = &e.kind else {
            return false;
        };
        matches!(&parts[..], [p] if p.value.eq_ignore_ascii_case("all"))
    }

    pub(super) fn from(&mut self, f: &FromClause) -> R<LogicalPlan> {
        let mut plan: Option<LogicalPlan> = None;
        for r in &f.relations {
            let right = self.relation(r)?;
            plan = Some(match plan {
                None => right,
                Some(left) => self.node(
                    PlanNode::Join {
                        left: arc(left),
                        right: arc(right),
                        kind: if r.lateral {
                            IrJoin::LateralInner
                        } else {
                            IrJoin::Inner
                        },
                        condition: JoinCondition::None,
                    },
                    r.span,
                ),
            });
        }
        let plan = plan.ok_or_else(|| unsupported("an empty FROM clause", f.span))?;
        let mut plan = self.lateral_views(plan, &f.lateral_views)?;
        if f.pivot.is_some() && f.unpivot.is_some() {
            return Err(err(
                krishiv_common::error::NotAllowedInFromUnpivotWithPivot {},
                f.span,
            ));
        }
        if let Some(p) = &f.pivot {
            plan = self.pivot(plan, p)?;
        }
        if let Some(u) = &f.unpivot {
            plan = self.unpivot(plan, u)?;
        }
        Ok(plan)
    }

    pub(super) fn relation(&mut self, r: &Relation) -> R<LogicalPlan> {
        // Spark rejects a relation carrying both clauses
        // (`NOT_ALLOWED_IN_FROM.UNPIVOT_WITH_PIVOT`, also raised for the
        // pipe-operator chain in the goldens): a `PIVOT ... UNPIVOT ...`
        // chain on one relation lowers to no plan.
        if r.extensions
            .iter()
            .any(|e| matches!(e, RelationExt::Pivot(_)))
            && r.extensions
                .iter()
                .any(|e| matches!(e, RelationExt::Unpivot(_)))
        {
            return Err(err(
                krishiv_common::error::NotAllowedInFromUnpivotWithPivot {},
                r.span,
            ));
        }
        let mut plan = self.factor(&r.primary)?;
        for ext in &r.extensions {
            match ext {
                RelationExt::Join(j) => {
                    plan = self.join(plan, j)?;
                }
                RelationExt::Pivot(p) => {
                    plan = self.pivot(plan, p)?;
                }
                RelationExt::Unpivot(u) => {
                    plan = self.unpivot(plan, u)?;
                }
            }
        }
        Ok(plan)
    }

    /// `[INNER] JOIN right constraint` onto `plan`.
    pub(super) fn join(&mut self, plan: LogicalPlan, j: &Join) -> R<LogicalPlan> {
        let kind = match j.kind {
            JoinKind::Inner | JoinKind::Cross if j.lateral => IrJoin::LateralInner,
            JoinKind::Left if j.lateral => IrJoin::LateralLeft,
            kind if j.lateral => {
                let join_type = match kind {
                    JoinKind::Right => "RIGHT OUTER",
                    JoinKind::Full => "FULL OUTER",
                    JoinKind::LeftSemi => "LEFT SEMI",
                    _ => "LEFT ANTI",
                };
                return Err(err(
                    krishiv_common::error::InvalidLateralJoinType {
                        join_type: join_type.to_owned(),
                    },
                    j.span,
                ));
            }
            JoinKind::Inner => IrJoin::Inner,
            JoinKind::Cross => IrJoin::Cross,
            JoinKind::Left => IrJoin::LeftOuter,
            JoinKind::Right => IrJoin::RightOuter,
            JoinKind::Full => IrJoin::FullOuter,
            JoinKind::LeftSemi => IrJoin::LeftSemi,
            JoinKind::LeftAnti => IrJoin::LeftAnti,
        };
        let right = self.factor(&j.right)?;
        let condition = if j.natural {
            // `NATURAL JOIN` has no condition in the syntax tree (the
            // parser leaves it empty); the analyzer computes the
            // common columns.
            JoinCondition::Natural
        } else {
            match &j.constraint {
                JoinConstraint::None => JoinCondition::None,
                JoinConstraint::On(e) => JoinCondition::On(arc(self.expr(e)?)),
                JoinConstraint::Using(cols) => {
                    JoinCondition::Using(cols.iter().map(|c| c.value.clone()).collect())
                }
                JoinConstraint::NearestBy(n) => {
                    return Err(unsupported("NEAREST BY joins", n.span));
                }
            }
        };
        Ok(self.node(
            PlanNode::Join {
                left: arc(plan),
                right: arc(right),
                kind,
                condition,
            },
            j.span,
        ))
    }

    /// `PIVOT (aggregates FOR columns IN (values)) [alias]` over `plan`,
    /// as Spark's parser builds it: the aggregates and values keep their
    /// aliases, several pivot columns stand for a struct value, and the
    /// alias becomes a `SubqueryAlias`.
    pub(super) fn pivot(&mut self, plan: LogicalPlan, p: &Pivot) -> R<LogicalPlan> {
        let mut aggregates = Vec::with_capacity(p.aggregates.len());
        for item in &p.aggregates {
            aggregates.push(self.select_item(item)?);
        }
        let mut pivot_columns = Vec::with_capacity(p.columns.len());
        for col in &p.columns {
            pivot_columns.push(arc(self.column(vec![col.value.clone()], col.span)));
        }
        let mut values = Vec::with_capacity(p.values.len());
        for item in &p.values {
            let alias = match &item.alias {
                None => None,
                Some(Alias::Name(name)) => Some(name.value.clone()),
                Some(_) => return Err(unsupported("multi-name aliases", item.span)),
            };
            // `(a, b)` is one struct value of two parts.
            let exprs = match &item.expr.kind {
                ExprKind::Row(items) => items
                    .iter()
                    .map(|i| self.expr(&i.expr).map(arc))
                    .collect::<R<Vec<_>>>()?,
                _ => vec![arc(self.expr(&item.expr)?)],
            };
            values.push(PivotValue { exprs, alias });
        }
        let plan = self.node(
            PlanNode::Pivot {
                input: arc(plan),
                pivot_columns,
                values,
                aggregates,
            },
            p.span,
        );
        Ok(match &p.alias {
            None => plan,
            Some(alias) => self.node(
                PlanNode::SubqueryAlias {
                    input: arc(plan),
                    alias: alias.value.clone(),
                    columns: Vec::new(),
                },
                p.span,
            ),
        })
    }

    /// `UNPIVOT [INCLUDE | EXCLUDE NULLS] (values FOR name IN (columns))
    /// [alias]` over `plan`, as Spark's parser builds it: the alias
    /// becomes a `SubqueryAlias`.
    pub(super) fn unpivot(&mut self, plan: LogicalPlan, u: &Unpivot) -> R<LogicalPlan> {
        let mut column_sets = Vec::with_capacity(u.columns.len());
        for set in &u.columns {
            let mut columns = Vec::with_capacity(set.columns.len());
            for col in &set.columns {
                columns.push(arc(self.column(
                    col.parts.iter().map(|p| p.value.clone()).collect(),
                    col.span,
                )));
            }
            column_sets.push(UnpivotColumnSet {
                columns,
                alias: set.alias.as_ref().map(|a| a.value.clone()),
            });
        }
        let plan = self.node(
            PlanNode::Unpivot {
                input: arc(plan),
                value_columns: u.value_columns.iter().map(|c| c.value.clone()).collect(),
                name_column: u.name_column.value.clone(),
                column_sets,
                include_nulls: u.include_nulls.unwrap_or(false),
            },
            u.span,
        );
        Ok(match &u.alias {
            None => plan,
            Some(alias) => self.node(
                PlanNode::SubqueryAlias {
                    input: arc(plan),
                    alias: alias.value.clone(),
                    columns: Vec::new(),
                },
                u.span,
            ),
        })
    }

    /// `TABLESAMPLE` over `plan`, as Spark's `AstBuilder.withSample` builds
    /// it (its checks are in `validate/sample.rs`): `n ROWS` is `LIMIT n`;
    /// `PERCENT` and `BUCKET x OUT OF y` are a Bernoulli sample of a
    /// fraction, which keeps every row at a fraction of 1 or more and none
    /// at 0 or less (Spark's `BernoulliCellSampler`). A fraction in between
    /// keeps rows by a random draw per row seeded by Spark's partition
    /// index, which Krishiv's partitions do not reproduce, so it stays
    /// unsupported, as does `SYSTEM` (block) sampling.
    pub(super) fn sample(&mut self, plan: LogicalPlan, s: &TableSample) -> R<LogicalPlan> {
        if s.sample_type == Some(SampleType::System) {
            return Err(unsupported("TABLESAMPLE SYSTEM", s.span));
        }
        let Some(method) = &s.method else {
            return Err(unsupported("TABLESAMPLE", s.span));
        };
        let limit = |l: &mut Self, plan: LogicalPlan, limit: Arc<IrExpr>| {
            l.node(
                PlanNode::Limit {
                    input: arc(plan),
                    limit,
                },
                s.span,
            )
        };
        let fraction = match &method.kind {
            SampleMethodKind::Rows(e) => {
                let n = arc(self.expr(e)?);
                return Ok(limit(self, plan, n));
            }
            SampleMethodKind::Percent { negative, value } => {
                let sign = if *negative { -1.0 } else { 1.0 };
                sign * value.parse::<f64>().unwrap_or(f64::NAN) / 100.0
            }
            SampleMethodKind::Bucket {
                numerator,
                denominator,
                on: None,
            } => {
                numerator.parse::<f64>().unwrap_or(f64::NAN)
                    / denominator.parse::<f64>().unwrap_or(f64::NAN)
            }
            // Validation refuses these.
            _ => return Err(unsupported("TABLESAMPLE", s.span)),
        };
        if fraction >= 1.0 {
            Ok(plan)
        } else if fraction <= 0.0 {
            Ok(limit(
                self,
                plan,
                arc(IrExpr::literal(IrLiteral::Integer(0))),
            ))
        } else {
            Err(unsupported(
                "TABLESAMPLE of a fraction between 0 and 1",
                s.span,
            ))
        }
    }

    pub(super) fn factor(&mut self, tf: &TableFactor) -> R<LogicalPlan> {
        let plan = self.factor_relation(tf)?;
        let plan = match &tf.sample {
            Some(sample) => self.sample(plan, sample)?,
            None => plan,
        };
        Ok(match &tf.alias {
            None => plan,
            Some(a) => self.with_alias(plan, a, tf.span),
        })
    }

    /// The relation of `tf`, before its sample and alias.
    pub(super) fn factor_relation(&mut self, tf: &TableFactor) -> R<LogicalPlan> {
        let plan = match &tf.kind {
            TableFactorKind::Table {
                name,
                temporal,
                options,
            } => {
                if let Some(t) = temporal {
                    return Err(unsupported("time travel", t.span));
                }
                let mut opts = BTreeMap::new();
                for o in options {
                    let key: Vec<&str> = o.key.parts.iter().map(|p| p.value.as_str()).collect();
                    let key = key.join(".");
                    let value = match o.value.as_ref().map(|v| &v.kind) {
                        Some(ExprKind::Literal(Literal::String(s))) => s.clone(),
                        Some(ExprKind::Literal(Literal::Number { text, .. })) => text.clone(),
                        Some(ExprKind::Literal(Literal::Boolean(b))) => b.to_string(),
                        _ => return Err(unsupported("this table option value", o.span)),
                    };
                    let value = OptionValue::from_user(&key, value);
                    opts.insert(key, value);
                }
                let parts: Vec<String> = name.parts.iter().map(|p| p.value.clone()).collect();
                let source = match &name.parts[..] {
                    // `parquet.`s3://...``: a path, kept typed so that its
                    // credentials never sit in a name.
                    [format, path] if path.quoted && is_path(&path.value) => ReadSource::Files {
                        // As written: Spark names it so in errors.
                        format: format.value.clone(),
                        paths: vec![OptionValue::from_user("path", path.value.clone())],
                        options: opts,
                        schema: None,
                    },
                    _ => ReadSource::Table {
                        name: parts,
                        options: opts,
                    },
                };
                self.node(PlanNode::Read(source), name.span)
            }
            TableFactorKind::Subquery(q) => {
                let inner = self.query(q)?;
                if tf.alias.is_none() {
                    return Ok(self.node(
                        PlanNode::SubqueryAlias {
                            input: arc(inner),
                            alias: "__auto_generated_subquery_name".to_owned(),
                            columns: Vec::new(),
                        },
                        tf.span,
                    ));
                }
                inner
            }
            TableFactorKind::Nested(r) => self.deeper(|l| l.relation(r))?,
            TableFactorKind::DynamicTable { name, .. } => {
                return Err(unsupported("IDENTIFIER()", name.span));
            }
            TableFactorKind::Values(rows) => self.inline_table(rows, tf.alias.as_ref(), tf.span)?,
            TableFactorKind::TableFunction {
                name,
                dynamic_name,
                args,
            } => {
                if dynamic_name.is_some() {
                    return Err(unsupported("table-valued functions", tf.span));
                }
                if name.parts.len() > 3 {
                    return Err(unsupported("table-valued functions", tf.span));
                }
                for a in args {
                    if matches!(a.expr.kind, crate::ast::ExprKind::TableArgument(_)) {
                        return Err(unsupported("table-valued functions", a.span));
                    }
                }
                // A named argument (`f(x => 1)`) travels as an `Alias`; the
                // analyzer binds it for a SQL-bodied function and refuses it
                // for any other table function.
                let lowered = args
                    .iter()
                    .map(|a| {
                        let value = self.arg(&a.expr)?;
                        Ok(match &a.name {
                            Some(n) => Arc::new(self.ex(
                                ExprNode::Alias {
                                    expr: value,
                                    name: n.value.clone(),
                                },
                                a.span,
                            )),
                            None => value,
                        })
                    })
                    .collect::<R<Vec<_>>>()?;
                // A generator as a table: `FROM explode(array(1, 2))`
                // generates from one empty row.
                let lname = name.parts[0].value.to_ascii_lowercase();
                let (kind, outer) = match lname.as_str() {
                    "stack" => (Some(krishiv_plan::GeneratorKind::Inline), false),
                    other => krishiv_plan::GeneratorKind::of_name(other)
                        .map_or((None, false), |(k, o)| (Some(k), o)),
                };
                // `FROM json_tuple(..)` stays a table function: the analyzer
                // runs it as the select list `SELECT json_tuple(..)` over one
                // row, and its alias-count error differs from a subquery's.
                if let (Some(kind), 1) = (kind, name.parts.len()) {
                    // `variant_explode` names its one parameter `input`
                    // (Spark's `FunctionSignature(InputParameter("input"))`).
                    let named = |a: &&crate::ast::FunctionArg| match &a.name {
                        Some(n) => {
                            kind != krishiv_plan::GeneratorKind::VariantExplode
                                || args.len() != 1
                                || !n.value.eq_ignore_ascii_case("input")
                        }
                        None => false,
                    };
                    if let Some(a) = args.iter().find(named) {
                        return Err(unsupported("table-valued functions", a.span));
                    }
                    let arg = &self.generator_arg(&name.parts[0], args, tf.span)?;
                    let one_row = self.node(PlanNode::OneRowRelation, tf.span);
                    self.node(
                        PlanNode::Generate {
                            input: arc(one_row),
                            kind,
                            outer,
                            arg: Arc::clone(arg),
                            output: Vec::new(),
                            names: Vec::new(),
                            qualifier: None,
                        },
                        tf.span,
                    )
                } else {
                    self.node(
                        PlanNode::TableFunction {
                            // A qualified name (`ns.f`) is kept as one string, its
                            // parts joined by `TABLE_FUNCTION_NAME_SEP`.
                            name: name
                                .parts
                                .iter()
                                .map(|p| p.value.as_str())
                                .collect::<Vec<_>>()
                                .join(&krishiv_plan::TABLE_FUNCTION_NAME_SEP.to_string()),
                            args: lowered,
                        },
                        tf.span,
                    )
                }
            }
        };
        Ok(plan)
    }

    /// `plan` under the `SubqueryAlias` of `a` (the one wrapper both the
    /// query-body and table-factor paths use). `AS (columns)` names no
    /// table: a generated name, as for no alias at all.
    pub(super) fn with_alias(
        &mut self,
        plan: LogicalPlan,
        a: &TableAlias,
        span: Span,
    ) -> LogicalPlan {
        self.node(
            PlanNode::SubqueryAlias {
                input: arc(plan),
                alias: a.name.as_ref().map_or_else(
                    || "__auto_generated_subquery_name".to_owned(),
                    |n| n.value.clone(),
                ),
                columns: a.columns.iter().map(|c| c.value.clone()).collect(),
            },
            span,
        )
    }

    /// `VALUES rows [AS alias(columns)]` at `span`: an unresolved
    /// [`PlanNode::InlineTable`] (Spark's `UnresolvedInlineTable`); the
    /// caller wraps it in a `SubqueryAlias` for the alias. A row is either
    /// a row constructor (one cell per item; item aliases are ignored, the
    /// schema comes from `columns`) or a single expression (one column, as
    /// Spark's grammar has it). Rows of different widths are rejected by
    /// [`crate::validate`] before lowering.
    pub(super) fn inline_table(
        &mut self,
        rows: &[Expr],
        alias: Option<&TableAlias>,
        span: Span,
    ) -> R<LogicalPlan> {
        let mut lowered = Vec::with_capacity(rows.len());
        for row in rows {
            let cells = match &row.kind {
                ExprKind::Row(items) => items
                    .iter()
                    .map(|i| self.arg(&i.expr))
                    .collect::<R<Vec<_>>>()?,
                _ => vec![self.arg(row)?],
            };
            lowered.push(cells);
        }
        let width = lowered.first().map_or(0, Vec::len);
        let columns = match alias.map(|a| &a.columns) {
            Some(cols) if !cols.is_empty() => cols.iter().map(|c| c.value.clone()).collect(),
            _ => (1..=width).map(|i| format!("col{i}")).collect(),
        };
        Ok(self.node(
            PlanNode::InlineTable {
                rows: lowered,
                columns,
            },
            span,
        ))
    }
}
