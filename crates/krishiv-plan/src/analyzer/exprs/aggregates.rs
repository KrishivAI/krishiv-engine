// SPDX-License-Identifier: Apache-2.0
//! Aggregate calls with their own resolution rules: `mode`, `listagg`/`string_agg` and the percentile family.

use super::*;

impl Resolver<'_> {
    /// `b` implicitly cast to `to`, if given (named and shown as `b`).
    /// `mode` with its `WITHIN GROUP` ordering and deterministic flag
    /// (Spark's `ModeBuilder` and `SupportsOrderingWithinGroup` checks):
    /// the ordering lowers to the value the executor reads, and the shown
    /// form keeps what was written (`mode() WITHIN GROUP (ORDER BY ...)`
    /// for the ordered forms, with Spark's swapped direction).
    ///
    /// `kids` holds the bound arguments, the filter, then the bound
    /// ordering keys; the executor reads the first resolved argument as
    /// the value and the second (a folded boolean literal, when present)
    /// as the tie-break (largest when true).
    #[inline(never)]
    pub(super) fn mode_call(
        &self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        f: &FunctionCall,
        kids: &[Bound],
        window_fn: bool,
    ) -> Result<Bound, AnalysisError> {
        let zone = self.ctx.conf.session_zone;
        let ansi = self.ctx.conf.ansi;
        let n_args = f.args.len();
        let at = |i: usize| kids.get(i);
        let filter_at = n_args + usize::from(f.filter.is_some());
        let orders: &[Bound] = kids.get(filter_at..).unwrap_or(&[]);
        let sql = |i: usize| at(i).map(|k| pretty(&k.shown(), zone));
        if f.distinct {
            return Err(AnalysisError::new(
                InvalidWithinGroupExpressionDistinctUnsupported {
                    func_name: "`mode`".to_owned(),
                },
                e.origin,
            ));
        }
        // The nested-aggregate and window checks, over the arguments, the
        // filter and the ordering keys alike.
        let mut checked: Vec<Bound> = kids[..filter_at.min(kids.len())].to_vec();
        checked.extend(orders.iter().cloned());
        if !window_fn {
            call_checks(entry, e, &checked, filter_at.min(checked.len()))?;
        }
        let wrong_arity = || {
            AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: "`mode`".to_owned(),
                    expected_num: entry.arity.expected().to_owned(),
                    actual_num: n_args.to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                e.origin,
            )
        };
        // The value the executor reads, the tie-break flag (largest when
        // true) and the shown ordering (Spark's swapped direction: an
        // ascending written order shows as descending and picks the
        // smallest, a descending one shows bare and picks the largest).
        enum Form {
            Plain,
            Smallest,
            Largest,
        }
        let (value, form): (&Bound, Form) = match (n_args, orders.len()) {
            (0, 0) => {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionWithinGroupMissing {
                        func_name: "`mode`".to_owned(),
                    },
                    e.origin,
                ));
            }
            (0, 1) => {
                let descending = f.within_group[0].direction == SortDirection::Descending;
                (
                    &orders[0],
                    if descending {
                        Form::Largest
                    } else {
                        Form::Smallest
                    },
                )
            }
            (0, n) => {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionWrongNumOrderings {
                        func_name: "`mode`".to_owned(),
                        expected_num: "1".to_owned(),
                        actual_num: n.to_string(),
                    },
                    e.origin,
                ));
            }
            (1, 0) => (at(0).ok_or_else(wrong_arity)?, Form::Plain),
            (2, 0) => {
                let flag = at(1).ok_or_else(wrong_arity)?;
                // The flag must fold (a column does not); a literal
                // without a folded value (binary, timestamp) still counts
                // as folded here, and fails the type check below.
                let folded = self.fold(&flag.expr, ansi);
                let literal = matches!(flag.expr.node, ExprNode::Literal(_));
                match (folded, literal) {
                    (None, false) => {
                        return Err(AnalysisError::new(
                            NonFoldableArgument {
                                func_name: "`mode`".to_owned(),
                                param_name: "`deterministic`".to_owned(),
                                param_type: "\"BOOLEAN\"".to_owned(),
                            },
                            e.origin,
                        ));
                    }
                    (Some(Constant::Error(err)), _) => {
                        return Err(AnalysisError::new(err, flag.expr.origin));
                    }
                    (Some(Constant::Null), _) => {
                        return Err(AnalysisError::new(
                            DatatypeMismatchUnexpectedNull {
                                sql_expr: format!("\"{}\"", sql(1).unwrap_or_default()),
                                expr_name: "`deterministic`".to_owned(),
                            },
                            e.origin,
                        ));
                    }
                    (Some(Constant::Boolean(true)), _) => {
                        (at(0).ok_or_else(wrong_arity)?, Form::Smallest)
                    }
                    (Some(Constant::Boolean(false)), _) => {
                        (at(0).ok_or_else(wrong_arity)?, Form::Plain)
                    }
                    // A literal without a folded value (binary, timestamp)
                    // or any other non-boolean foldable fails the type
                    // check, as Spark checks the type after folding.
                    _ if flag.ty.data_type != Some(DataType::Boolean) => {
                        let input = flag
                            .ty
                            .data_type
                            .clone()
                            .map_or_else(|| "VOID".to_owned(), |t| t.sql());
                        return Err(AnalysisError::new(
                            UnexpectedInputType {
                                param_index: "second".to_owned(),
                                function_name: "`mode`".to_owned(),
                                required_type: "\"BOOLEAN\"".to_owned(),
                                input_sql: format!("\"{}\"", sql(1).unwrap_or_default()),
                                input_type: format!("\"{input}\""),
                            },
                            e.origin,
                        ));
                    }
                    // A boolean-typed foldable that folds to no constant
                    // (unreachable through the current constant folder).
                    _ => (at(0).ok_or_else(wrong_arity)?, Form::Plain),
                }
            }
            (1 | 2, _) => {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionWrongNumOrderings {
                        func_name: "`mode`".to_owned(),
                        expected_num: "0".to_owned(),
                        actual_num: orders.len().to_string(),
                    },
                    e.origin,
                ));
            }
            _ => return Err(wrong_arity()),
        };
        let data_type = value.ty.data_type.clone().unwrap_or(DataType::Null);
        let mut args = vec![Arc::clone(&value.expr)];
        // The tie-break the executor reads (largest when true).
        let largest = match form {
            Form::Plain => None,
            Form::Smallest => Some(false),
            Form::Largest => Some(true),
        };
        if let Some(largest) = largest {
            args.push(Arc::new(
                Expr::new(ExprNode::Literal(Literal::Boolean(largest))).with_origin(e.origin),
            ));
        }
        // The shown ordering: Spark's swapped direction (ascending shows
        // as descending, descending shows bare).
        let shown_order = |o: &Bound, descending: bool, shown: bool| SortOrder {
            expr: if shown {
                o.shown()
            } else {
                Arc::clone(&o.expr)
            },
            direction: if descending {
                SortDirection::Ascending
            } else {
                SortDirection::Descending
            },
            nulls: NullOrdering::NullsFirst,
        };
        let within_group = match form {
            Form::Plain => Vec::new(),
            Form::Smallest => vec![shown_order(value, false, false)],
            Form::Largest => vec![shown_order(value, true, false)],
        };
        let call = ResolvedFunction {
            id: entry.id.clone(),
            kind: entry.kind,
            args,
            distinct: f.distinct,
            filter: at(n_args)
                .filter(|_| f.filter.is_some())
                .map(|k| Arc::clone(&k.expr)),
            within_group,
            data_type: data_type.clone(),
            nullable: true,
            deterministic: entry.deterministic,
        };
        let expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(call))).with_origin(e.origin),
        );
        // The shown form keeps what was written (the ordering for the
        // ordered forms, at Spark's swapped direction).
        let shown = Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: entry.id.clone(),
                kind: entry.kind,
                args: vec![value.shown()],
                distinct: f.distinct,
                filter: at(n_args).filter(|_| f.filter.is_some()).map(Bound::shown),
                within_group: match form {
                    Form::Plain => Vec::new(),
                    Form::Smallest => vec![shown_order(value, false, true)],
                    Form::Largest => vec![shown_order(value, true, true)],
                },
                data_type,
                nullable: true,
                deterministic: entry.deterministic,
            },
        )));
        let shown = Arc::new(shown.with_origin(e.origin));
        let bound = Bound {
            expr,
            ty: arg_type(value.ty.data_type.clone().unwrap_or(DataType::Null), true),
            agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                Some(e.origin)
            } else {
                kids.iter().find_map(|k| k.agg)
            },
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral: kids
                .iter()
                .flat_map(|k| k.lateral.iter().copied())
                .collect(),
            shown: Some(shown),
        };
        Ok(bound)
    }

    /// `listagg(expr[, delimiter])` with its `WITHIN GROUP` ordering
    /// (Spark's `ListAgg` checks): the delimiter is implicitly cast to
    /// the value's type by the declared inputs, `DISTINCT` needs the
    /// ordering to be the value itself (or a cast that preserves
    /// equality, as Spark's `hasDistinctOrderAmbiguity` computes it).
    ///
    /// `kids` holds the bound arguments, the filter, then the bound
    /// ordering keys; the executor reads the value, then the ordering
    /// keys, and the folded delimiter from the resolved call.
    #[inline(never)]
    pub(super) fn listagg_call(
        &self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        f: &FunctionCall,
        kids: &[Bound],
        window_fn: bool,
    ) -> Result<Bound, AnalysisError> {
        let n_args = f.args.len();
        let filter_at = n_args + usize::from(f.filter.is_some());
        let orders: &[Bound] = kids.get(filter_at..).unwrap_or(&[]);
        let zone = self.ctx.conf.session_zone;
        // The nested-aggregate and window checks, over the arguments, the
        // filter and the ordering keys alike.
        let mut checked: Vec<Bound> = kids[..filter_at.min(kids.len())].to_vec();
        checked.extend(orders.iter().cloned());
        if !window_fn {
            call_checks(entry, e, &checked, filter_at.min(checked.len()))?;
        }
        // The value and the delimiter, implicitly cast (Spark's arity
        // check, then `listagg`'s own delimiter checks in the registry).
        // The SQL layer fills a missing delimiter with NULL, so the call
        // always has both.
        if n_args == 0 {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: "`listagg`".to_owned(),
                    expected_num: entry.arity.expected().to_owned(),
                    actual_num: "0".to_owned(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                e.origin,
            ));
        }
        // A call built outside SQL (`call_function("listagg", [x])`) has no
        // delimiter: it is NULL, as the SQL layer writes it.
        let mut head: Vec<Bound> = kids[..n_args].to_vec();
        if n_args == 1 {
            head.push(default_argument(
                Literal::Null(None),
                DataType::Null,
                true,
                e.origin,
            ));
        }
        let (mut call_kids, ty) = self.typed_call(entry, e, &head)?;
        // `DISTINCT` with an ordering: the ordering must be the value
        // itself. When the value is a cast of the ordering key, the cast
        // must preserve equality (Spark's `hasDistinctOrderAmbiguity`,
        // with the looser cast rule off): anything else is ambiguous
        // (`MISMATCH_WITH_DISTINCT_INPUT`), an unsafe cast (`DOUBLE`,
        // `FLOAT`, `TIMESTAMP` to `STRING`) its
        // `MISMATCH_WITH_DISTINCT_INPUT_UNSAFE_CAST`. The value is the
        // implicitly cast one (a `DOUBLE` column reads as its `STRING`
        // cast); the error quotes what was written.
        if f.distinct && !orders.is_empty() {
            // Structural equality, ignoring source spans (either side may
            // be written elsewhere).
            let mut memo = std::collections::HashMap::new();
            let mut key = |e: &Arc<Expr>| semantic_key(e, &mut memo);
            let value = &call_kids[0].expr;
            let same =
                orders.len() == 1 && orders.first().is_some_and(|o| key(&o.expr) == key(value));
            if !same {
                let cast_of = match &call_kids[0].expr.node {
                    ExprNode::Cast { expr, to, .. }
                        if orders.len() == 1
                            && orders.first().is_some_and(|o| key(&o.expr) == key(expr))
                            && matches!(to, DataType::String | DataType::Binary) =>
                    {
                        Some(to.clone())
                    }
                    _ => None,
                };
                match cast_of {
                    // A cast that preserves equality deduplicates exactly.
                    Some(to)
                        if orders
                            .first()
                            .and_then(|o| o.ty.data_type.clone())
                            .is_some_and(|from| Self::cast_preserves_equality(&from, &to)) => {}
                    Some(to) => {
                        let input = orders
                            .first()
                            .and_then(|o| o.ty.data_type.clone())
                            .map_or_else(|| "VOID".to_owned(), |t| t.sql());
                        return Err(AnalysisError::new(
                            InvalidWithinGroupExpressionMismatchWithDistinctInputUnsafeCast {
                                func_name: "`listagg`".to_owned(),
                                input_type: format!("\"{input}\""),
                                cast_type: format!("\"{}\"", to.sql()),
                            },
                            e.origin,
                        ));
                    }
                    None => {
                        let arg = |k: &Bound| pretty(&k.shown(), zone);
                        return Err(AnalysisError::new(
                            InvalidWithinGroupExpressionMismatchWithDistinctInput {
                                func_name: "`listagg`".to_owned(),
                                func_arg: format!("\"{}\"", arg(&kids[0])),
                                ordering_expr: orders
                                    .iter()
                                    .map(|o| format!("\"{}\"", pretty(&o.shown(), zone)))
                                    .collect::<Vec<_>>()
                                    .join(", "),
                            },
                            e.origin,
                        ));
                    }
                }
            }
        }
        // A NULL delimiter stays NULL (it reads as empty): the implicit
        // cast must not give it a type, or the same-type check below
        // would compare the value against the cast.
        if head.len() > 1
            && matches!(
                head.get(1).map(|k| &k.expr.node),
                Some(ExprNode::Literal(Literal::Null(_)))
            )
        {
            call_kids[1] = default_argument(Literal::Null(None), DataType::Null, true, e.origin);
        }
        // The executor reads a literal delimiter: fold one here (the
        // shown form keeps what was written). Binary and other literals
        // the folder has no value for stay as written for the planner.
        if let Some(delim) = call_kids.get_mut(1) {
            let folded = match self.fold(&delim.expr, self.ctx.conf.ansi) {
                Some(Constant::String(s)) => Some((Literal::String(s), false)),
                Some(Constant::Null) => Some((Literal::Null(None), true)),
                Some(Constant::Boolean(b)) => Some((Literal::String(b.to_string()), false)),
                Some(Constant::Integral(i)) => Some((Literal::String(i.to_string()), false)),
                Some(Constant::Error(err)) => {
                    return Err(AnalysisError::new(err, delim.expr.origin));
                }
                Some(_) | None => None,
            };
            if let Some((lit, nullable)) = folded {
                let ty = arg_type(DataType::String, nullable);
                if delim.ty.data_type == Some(DataType::String) {
                    delim.expr =
                        Arc::new(Expr::new(ExprNode::Literal(lit)).with_origin(delim.expr.origin));
                    delim.ty = ty;
                }
            }
        }
        // The resolved ordering keys, with their written directions.
        let mut args: Vec<Arc<Expr>> = call_kids.iter().map(|k| Arc::clone(&k.expr)).collect();
        args.extend(orders.iter().map(|k| Arc::clone(&k.expr)));
        let resolved_orders: Vec<SortOrder> = orders
            .iter()
            .zip(&f.within_group)
            .map(|(k, o)| SortOrder {
                expr: Arc::clone(&k.expr),
                direction: o.direction,
                nulls: o.nulls,
            })
            .collect();
        let shown_orders: Vec<SortOrder> = orders
            .iter()
            .zip(&f.within_group)
            .map(|(k, o)| SortOrder {
                expr: k.shown(),
                direction: o.direction,
                nulls: o.nulls,
            })
            .collect();
        let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
        let call = ResolvedFunction {
            id: entry.id.clone(),
            kind: entry.kind,
            args,
            distinct: f.distinct,
            filter: kids
                .get(n_args)
                .filter(|_| f.filter.is_some())
                .map(|k| Arc::clone(&k.expr)),
            within_group: resolved_orders,
            data_type: data_type.clone(),
            nullable: ty.nullable,
            deterministic: entry.deterministic,
        };
        let expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(call))).with_origin(e.origin),
        );
        // The shown form keeps what was written (the delimiter default
        // included, as Spark's constructor writes it).
        let shown_args: Vec<Arc<Expr>> = call_kids.iter().map(Bound::shown).collect();
        let shown = Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: entry.id.clone(),
                kind: entry.kind,
                args: shown_args,
                distinct: f.distinct,
                filter: kids
                    .get(n_args)
                    .filter(|_| f.filter.is_some())
                    .map(Bound::shown),
                within_group: shown_orders,
                data_type,
                nullable: ty.nullable,
                deterministic: entry.deterministic,
            },
        )));
        let shown = Arc::new(shown.with_origin(e.origin));
        Ok(Bound {
            expr,
            ty,
            agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                Some(e.origin)
            } else {
                kids.iter().find_map(|k| k.agg)
            },
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral: kids
                .iter()
                .flat_map(|k| k.lateral.iter().copied())
                .collect(),
            shown: Some(shown),
        })
    }

    /// Whether casting `from` to string or binary preserves equality, so
    /// `listagg(DISTINCT CAST(x AS STRING)) WITHIN GROUP (ORDER BY x)`
    /// deduplicates exactly (Spark's `isCastEqualityPreserving` and
    /// `isCastTargetEqualityPreserving`, with the engine's types: no
    /// interval values, and every string is binary-collated).
    pub(super) fn cast_preserves_equality(from: &DataType, to: &DataType) -> bool {
        if !matches!(to, DataType::String | DataType::Binary) {
            return false;
        }
        matches!(
            from,
            DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Decimal(_)
                | DataType::Date
                | DataType::Time { .. }
                | DataType::TimestampNtz
                | DataType::Boolean
                | DataType::Binary
                | DataType::String,
        )
    }

    /// The ordered-set and quantile aggregates (`percentile`,
    /// `percentile_approx`, `approx_percentile`, `median`,
    /// `percentile_cont`, `percentile_disc`, `approx_count_distinct`),
    /// typed here rather than through the registry: Spark's builders fold
    /// the percentage, accuracy, frequency and `relativeSD` while typing,
    /// lower `WITHIN GROUP` to the ordering key, and fill the frequency
    /// and accuracy defaults. The executor reads the folded literals.
    ///
    /// The resolved call keeps the lowered arguments (the value, the cast
    /// percentage, the frequency or accuracy, the descending flag); the
    /// shown form keeps the call as written (with `WITHIN GROUP`), which
    /// names the output and quotes errors, as Spark's `toPrettySQL` does.
    #[inline(never)]
    pub(super) fn quantile_call(
        &mut self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        f: &FunctionCall,
        kids: &[Bound],
        n: usize,
        window_fn: bool,
    ) -> Result<Bound, AnalysisError> {
        use krishiv_functions::registry::{approx_percentile_result, exact_percentile_result};
        let name = entry.id.name();
        let origin = e.origin;
        let zone = self.ctx.conf.session_zone;
        let wrong_args = |actual: usize| {
            AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: format!("`{name}`"),
                    expected_num: entry.arity.expected().to_owned(),
                    actual_num: actual.to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                origin,
            )
        };
        if !entry.arity.accepts(f.args.len()) {
            return Err(wrong_args(f.args.len()));
        }
        let ordered = matches!(name, "percentile_cont" | "percentile_disc");
        if ordered {
            if f.within_group.is_empty() {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionWithinGroupMissing {
                        func_name: format!("`{name}`"),
                    },
                    origin,
                ));
            }
            if f.distinct {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionDistinctUnsupported {
                        func_name: format!("`{name}`"),
                    },
                    origin,
                ));
            }
            if f.within_group.len() != 1 {
                return Err(AnalysisError::new(
                    InvalidWithinGroupExpressionWrongNumOrderings {
                        func_name: format!("`{name}`"),
                        expected_num: "1".to_owned(),
                        actual_num: f.within_group.len().to_string(),
                    },
                    origin,
                ));
            }
        }
        // The nesting and window checks see the filter and the ordering key
        // too (a filter sits between the arguments and the ordering keys).
        call_checks(entry, e, kids, kids.len())?;
        let filter = kids.get(n).filter(|_| f.filter.is_some());
        let wstart = kids.len() - f.within_group.len();
        // The lowered call's SQL as Spark quotes it in errors: `name(args)`
        // over the shown arguments (the frequency and accuracy defaults
        // included, the descending flag excluded).
        let lowered_sql = |args: &[Arc<Expr>]| {
            format!(
                "\"{name}({})\"",
                args.iter()
                    .map(|a| pretty(a, zone))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        // The value (or ordering key): Spark's numeric input, which the
        // approximate path extends to dates and timestamps.
        let value = if ordered {
            kids[wstart].clone()
        } else {
            kids[0].clone()
        };
        let approx = matches!(name, "percentile_approx" | "approx_percentile");
        let temporal = approx
            && matches!(
                value.ty.data_type,
                Some(DataType::Date | DataType::Timestamp | DataType::TimestampNtz)
            );
        // `approx_count_distinct` hashes a value of any type as it is: no
        // numeric requirement and no cast (the executor refuses what it
        // cannot hash).
        let hashes_any_type = name == "approx_count_distinct";
        let child_target: Option<DataType> = if temporal || hashes_any_type {
            None
        } else {
            let given = [value.ty.clone()];
            let call = CallTypes::new(&given, &[]);
            let inputs = Inputs::Expects(vec![InputType::Numeric { intervals: true }]);
            let arg = arg_of(&value);
            match coerce::inputs(&inputs, &[arg], &call) {
                Ok(casts) => casts.into_iter().next().flatten(),
                Err(m) => {
                    let lowered = Arc::new(
                        Expr::new(ExprNode::Function(FunctionCall {
                            name: name.to_owned(),
                            qualifier: Vec::new(),
                            args: vec![value.shown()],
                            arg_names: Vec::new(),
                            distinct: false,
                            filter: None,
                            within_group: Vec::new(),
                        }))
                        .with_origin(origin),
                    );
                    return Err(self.mismatch(
                        m,
                        &lowered,
                        std::slice::from_ref(&value),
                        name,
                        origin,
                    ));
                }
            }
        };
        let mut value = value;
        if child_target.is_some() {
            value = self.implicit_cast(&value, child_target);
        }
        // `median` fixes its percentage at 0.5 and `approx_count_distinct`
        // takes none: neither reads the percentage argument.
        let median = name == "median";
        let appr = name == "approx_count_distinct";
        // Spark quotes errors over the lowered call, with defaulted
        // arguments filled in: the frequency, or the accuracy. The
        // ordered-set calls lower to two arguments.
        let third_shown: Option<Arc<Expr>> = if ordered || appr {
            None
        } else {
            let default = if approx {
                Literal::Integer(10000)
            } else {
                Literal::Long(1)
            };
            Some(if f.args.len() == 3 {
                kids[2].shown()
            } else {
                Arc::new(Expr::new(ExprNode::Literal(default)).with_origin(origin))
            })
        };
        // The percentage: a double or an array of doubles after the implicit
        // cast, folded here; the analyzer, not the executor, validates it.
        let pct_index = if ordered { 0 } else { 1 };
        // The lowered call as Spark quotes percentage errors: with the
        // defaulted third argument filled in.
        let pct_sql = |pct: &Bound| match &third_shown {
            Some(t) => lowered_sql(&[value.shown(), pct.shown(), Arc::clone(t)]),
            None => lowered_sql(&[value.shown(), pct.shown()]),
        };
        let half = || Bound {
            expr: Arc::new(Expr::new(ExprNode::Literal(Literal::Double(0.5))).with_origin(origin)),
            ty: arg_type(DataType::Double, false),
            agg: None,
            win: None,
            outer: None,
            lateral: Vec::new(),
            shown: None,
        };
        let (mut pct, mut array, mut percentages) = (half(), false, vec![0.5]);
        if !median && !appr {
            pct = kids[pct_index].clone();
            let pct_type = pct.ty.data_type.clone();
            array = matches!(pct_type, Some(DataType::Array(_)));
            if array {
                let (element, contains_null) = match &pct_type {
                    Some(DataType::Array(a)) => (a.element_type().clone(), a.contains_null()),
                    _ => (DataType::Null, true),
                };
                let castable = !contains_null
                    && matches!(
                        element,
                        DataType::Byte
                            | DataType::Short
                            | DataType::Integer
                            | DataType::Long
                            | DataType::Float
                            | DataType::Double
                            | DataType::Decimal(_)
                            | DataType::Null
                    )
                    || (!self.ctx.conf.ansi && element == DataType::String);
                // (Under ANSI Spark refuses an array of strings, where it casts a
                // string percentage.)
                if !castable {
                    let sql = pct_sql(&pct);
                    return Err(AnalysisError::new(
                        DatatypeMismatchUnexpectedInputType {
                            sql_expr: sql,
                            param_index: ordinal(1),
                            required_type: "\"ARRAY<DOUBLE>\"".to_owned(),
                            input_sql: format!("\"{}\"", pretty(&pct.shown(), zone)),
                            input_type: quoted_type(pct_type.as_ref()),
                        },
                        origin,
                    ));
                }
            } else {
                let castable = matches!(
                    pct_type,
                    None | Some(
                        DataType::Byte
                            | DataType::Short
                            | DataType::Integer
                            | DataType::Long
                            | DataType::Float
                            | DataType::Double
                            | DataType::Decimal(_)
                            | DataType::String
                            | DataType::Null
                    )
                );
                if !castable {
                    let sql = pct_sql(&pct);
                    return Err(AnalysisError::new(
                        DatatypeMismatchUnexpectedInputType {
                            sql_expr: sql,
                            param_index: ordinal(1),
                            required_type: "\"DOUBLE\"".to_owned(),
                            input_sql: format!("\"{}\"", pretty(&pct.shown(), zone)),
                            input_type: quoted_type(pct_type.as_ref()),
                        },
                        origin,
                    ));
                }
            }
            let pct_target = if array {
                DataType::Array(ArrayType::new(DataType::Double, false).into())
            } else {
                DataType::Double
            };
            let pct = self.implicit_cast(&pct, Some(pct_target));
            let conf = fold::Conf::new(
                self.ctx.conf.ansi,
                self.ctx.conf.decimal_precision_loss,
                self.ctx.conf.map_key_dedup_last_win,
                (
                    self.ctx.conf.legacy_java_charsets,
                    self.ctx.conf.legacy_coding_error_action,
                ),
                self.ctx.profile,
                self.ctx.conf.session_zone.and_then(SessionZone::fixed),
            );
            // An array percentage folds elementwise: a cast of the whole array
            // does not fold, so each element of the array call is cast to
            // double and folded on its own (as Spark converts its array).
            // Anything else folds as one value.
            let folded: Vec<Option<f64>> = if array {
                match array_elements(&pct.expr) {
                    Some(elements) => {
                        let mut out = Vec::with_capacity(elements.len());
                        for el in elements {
                            let cast = Arc::new(
                                Expr::new(ExprNode::Cast {
                                    expr: el,
                                    to: DataType::Double,
                                    mode: if self.ctx.conf.ansi {
                                        crate::CastMode::Ansi
                                    } else {
                                        crate::CastMode::Legacy
                                    },
                                })
                                .with_origin(origin),
                            );
                            match fold::doubles(
                                &cast,
                                conf,
                                &mut self.folds.borrow_mut(),
                                self.ctx.functions.as_ref(),
                            ) {
                                None => {
                                    return Err(non_foldable(
                                        name, &value, &pct, array, zone, origin,
                                    ));
                                }
                                Some(Err(e)) => return Err(AnalysisError::new(e, origin)),
                                Some(Ok(mut v)) => {
                                    if v.len() != 1 {
                                        return Err(non_foldable(
                                            name, &value, &pct, array, zone, origin,
                                        ));
                                    }
                                    out.append(&mut v);
                                }
                            }
                        }
                        out
                    }
                    None => {
                        return Err(non_foldable(name, &value, &pct, array, zone, origin));
                    }
                }
            } else {
                match fold::doubles(
                    &pct.expr,
                    conf,
                    &mut self.folds.borrow_mut(),
                    self.ctx.functions.as_ref(),
                ) {
                    None => {
                        return Err(non_foldable(name, &value, &pct, array, zone, origin));
                    }
                    Some(Err(e)) => return Err(AnalysisError::new(e, origin)),
                    Some(Ok(v)) => v,
                }
            };
            if folded.iter().any(Option::is_none) {
                let sql = pct_sql(&pct);
                return Err(AnalysisError::new(
                    DatatypeMismatchUnexpectedNull {
                        expr_name: "percentage".to_owned(),
                        sql_expr: sql,
                    },
                    origin,
                ));
            }
            percentages = folded.into_iter().flatten().collect();
            if percentages.iter().any(|&v| v < 0.0 || v > 1.0) {
                let sql = pct_sql(&pct);
                let current: Vec<String> = percentages
                    .iter()
                    .map(|v| format!("{}D", crate::spark_fmt::double(*v)))
                    .collect();
                return Err(AnalysisError::new(
                    DatatypeMismatchValueOutOfRange {
                        expr_name: "percentage".to_owned(),
                        value_range: "[0.0, 1.0]".to_owned(),
                        current_value: current.join(","),
                        sql_expr: sql,
                    },
                    origin,
                ));
            }
        }
        // The executor reads the folded percentage literals (Spark folds
        // them while typing too): a `double`, or an `array` of them.
        let pct_lit = if array {
            let elements: Vec<Arc<Expr>> = percentages
                .iter()
                .map(|v| {
                    Arc::new(Expr::new(ExprNode::Literal(Literal::Double(*v))).with_origin(origin))
                })
                .collect();
            let FunctionLookup::Found(entry) = self
                .ctx
                .functions
                .lookup(&["array".to_owned()], self.ctx.profile)
            else {
                return Err(super::super::bind::unsupported(
                    "the array function",
                    origin,
                ));
            };
            Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                    ResolvedFunction {
                        id: entry.id,
                        kind: FunctionKind::Scalar,
                        args: elements,
                        distinct: false,
                        filter: None,
                        within_group: Vec::new(),
                        data_type: DataType::Array(ArrayType::new(DataType::Double, false).into()),
                        nullable: false,
                        deterministic: true,
                    },
                )))
                .with_origin(origin),
            )
        } else {
            // An empty array is legal (Spark yields NULL for it); a scalar
            // percentage always folds to one value (NULL fails above).
            let v = percentages.first().copied().unwrap_or(f64::NAN);
            Arc::new(Expr::new(ExprNode::Literal(Literal::Double(v))).with_origin(origin))
        };
        // The frequency, accuracy, `relativeSD` and direction, per function.
        // The resolved call keeps the lowered arguments; the shown form keeps
        // the call as written (with `WITHIN GROUP`), which names the output.
        let mut resolved: Vec<Arc<Expr>> = vec![Arc::clone(&value.expr)];
        let mut shown: Vec<Arc<Expr>> = vec![value.shown()];
        let data_type: DataType;
        let mut nullable = true;
        if ordered {
            // The percentage (already validated) and the descending flag.
            // The shown form keeps the call as written: the percentage
            // argument, then `WITHIN GROUP`.
            resolved.push(Arc::clone(&pct_lit));
            shown = vec![kids[0].shown()];
            let descending = matches!(
                f.within_group[0].direction,
                crate::SortDirection::Descending
            );
            resolved.push(Arc::new(
                Expr::new(ExprNode::Literal(Literal::Boolean(descending))).with_origin(origin),
            ));
            // `percentile_disc` reads the session's legacy calculation
            // (`spark.sql.legacy.percentileDiscCalculation`); the executor
            // takes it as a literal like the descending flag.
            resolved.push(Arc::new(
                Expr::new(ExprNode::Literal(Literal::Boolean(
                    self.ctx.conf.legacy_percentile_disc,
                )))
                .with_origin(origin),
            ));
            data_type = exact_percentile_result(array);
        } else if median {
            data_type = exact_percentile_result(false);
        } else if appr {
            // `relativeSD`: a double (or decimal) literal, as Spark's
            // `validateDoubleLiteral` requires; the default is 0.05. It is
            // the second SQL argument, not the filter.
            let rsd = match (f.args.len() == 2).then(|| kids.get(1)).flatten() {
                None => Bound {
                    expr: Arc::new(
                        Expr::new(ExprNode::Literal(Literal::Double(0.05))).with_origin(origin),
                    ),
                    ty: arg_type(DataType::Double, false),
                    agg: None,
                    win: None,
                    outer: None,
                    lateral: Vec::new(),
                    shown: None,
                },
                Some(k) => k.clone(),
            };
            // The written value as a double (`0.5` is a decimal literal).
            let written = match &rsd.expr.node {
                ExprNode::Literal(Literal::Double(d)) => Some(*d),
                ExprNode::Literal(Literal::Decimal {
                    unscaled,
                    data_type,
                }) => Some(*unscaled as f64 / 10_f64.powi(i32::from(data_type.scale()))),
                _ => None,
            };
            let Some(written) = written else {
                return Err(AnalysisError::new(LegacyErrorTemp1104 {}, origin));
            };
            // Spark's `require(p >= 4)` on the addressing bits
            // (`p = ceil(2 * log2(1.106 / relativeSD))`) fails the call when
            // the error is above about 39%.
            let p = (2.0 * (1.106 / written).ln() / std::f64::consts::LN_2).ceil();
            if written.is_finite() && written > 0.0 && p < 4.0 {
                return Err(AnalysisError::new(
                    FailedFunctionCall {
                        func_name: "`approx_count_distinct`".to_owned(),
                    },
                    origin,
                ));
            }
            // The executor derives the addressing bits and refuses precisions
            // Spark's `require` refuses.
            resolved.push(Arc::clone(&rsd.expr));
            data_type = DataType::Long;
            nullable = false;
        } else {
            // `percentile` and `percentile_approx` share the percentage.
            resolved.push(Arc::clone(&pct_lit));
            shown.push(pct.shown());
            if approx {
                // The accuracy: the third argument, or 10000 (Spark's
                // builder default); it must be a foldable integral in
                // `(0, 2147483647]`. It is the third SQL argument, not the
                // filter.
                let accuracy = match (f.args.len() == 3).then(|| kids.get(2)).flatten() {
                    Some(k) => k.clone(),
                    None => Bound {
                        expr: Arc::new(
                            Expr::new(ExprNode::Literal(Literal::Integer(10000)))
                                .with_origin(origin),
                        ),
                        ty: arg_type(DataType::Integer, false),
                        agg: None,
                        win: None,
                        outer: None,
                        lateral: Vec::new(),
                        shown: None,
                    },
                };
                let accuracy_sql = lowered_sql(&[value.shown(), pct.shown(), accuracy.shown()]);
                let acc_type = accuracy.ty.data_type.clone();
                if !matches!(
                    acc_type,
                    None | Some(
                        DataType::Byte | DataType::Short | DataType::Integer | DataType::Long
                    )
                ) {
                    return Err(AnalysisError::new(
                        DatatypeMismatchUnexpectedInputType {
                            sql_expr: accuracy_sql,
                            param_index: ordinal(2),
                            required_type: "\"INTEGRAL\"".to_owned(),
                            input_sql: format!("\"{}\"", pretty(&accuracy.shown(), zone)),
                            input_type: quoted_type(acc_type.as_ref()),
                        },
                        origin,
                    ));
                }
                let n = match self.fold(&accuracy.expr, self.ctx.conf.ansi) {
                    Some(Constant::Integral(n)) => n,
                    Some(Constant::Null) => {
                        return Err(AnalysisError::new(
                            DatatypeMismatchUnexpectedNull {
                                expr_name: "accuracy".to_owned(),
                                sql_expr: accuracy_sql,
                            },
                            origin,
                        ));
                    }
                    Some(Constant::Error(e)) => return Err(AnalysisError::new(e, origin)),
                    _ => {
                        return Err(AnalysisError::new(
                            DatatypeMismatchNonFoldableInput {
                                sql_expr: accuracy_sql,
                                input_name: "`accuracy`".to_owned(),
                                input_type: quoted_type(acc_type.as_ref()),
                                input_expr: format!("\"{}\"", pretty(&accuracy.shown(), zone)),
                            },
                            origin,
                        ));
                    }
                };
                if n <= 0 || n > i64::from(i32::MAX) {
                    return Err(AnalysisError::new(
                        DatatypeMismatchValueOutOfRange {
                            expr_name: "accuracy".to_owned(),
                            value_range: "(0, 2147483647]".to_owned(),
                            current_value: n.to_string(),
                            sql_expr: accuracy_sql,
                        },
                        origin,
                    ));
                }
                // The executor reads the folded literal.
                resolved.push(Arc::new(
                    Expr::new(ExprNode::Literal(Literal::Long(n))).with_origin(origin),
                ));
                // Spark names the call with the defaulted accuracy.
                shown.push(accuracy.shown());
                let child = value.ty.data_type.clone().unwrap_or(DataType::Null);
                data_type = approx_percentile_result(&child, array);
            } else {
                // The frequency: the third argument, or 1; a per-row weight
                // which stays an expression (negative values fail per row).
                // It is the third SQL argument, not the filter.
                let frequency = match (f.args.len() == 3).then(|| kids.get(2)).flatten() {
                    Some(k) => k.clone(),
                    None => Bound {
                        expr: Arc::new(
                            Expr::new(ExprNode::Literal(Literal::Long(1))).with_origin(origin),
                        ),
                        ty: arg_type(DataType::Long, false),
                        agg: None,
                        win: None,
                        outer: None,
                        lateral: Vec::new(),
                        shown: None,
                    },
                };
                let freq_type = frequency.ty.data_type.clone();
                if !matches!(
                    freq_type,
                    None | Some(
                        DataType::Byte | DataType::Short | DataType::Integer | DataType::Long
                    )
                ) {
                    let sql = lowered_sql(&[value.shown(), pct.shown(), frequency.shown()]);
                    return Err(AnalysisError::new(
                        DatatypeMismatchUnexpectedInputType {
                            sql_expr: sql,
                            param_index: ordinal(2),
                            required_type: "\"INTEGRAL\"".to_owned(),
                            input_sql: format!("\"{}\"", pretty(&frequency.shown(), zone)),
                            input_type: quoted_type(freq_type.as_ref()),
                        },
                        origin,
                    ));
                }
                resolved.push(Arc::clone(&frequency.expr));
                // Spark names the call with the defaulted frequency.
                shown.push(frequency.shown());
                data_type = exact_percentile_result(array);
            }
        }
        let shown_within: Vec<SortOrder> = f
            .within_group
            .iter()
            .zip(kids[wstart..].iter())
            .map(|(o, k)| SortOrder {
                expr: k.shown(),
                ..o.clone()
            })
            .collect();
        let shown_call = Arc::new(
            Expr::new(ExprNode::Function(FunctionCall {
                name: f.name.clone(),
                qualifier: f.qualifier.clone(),
                args: shown,
                arg_names: Vec::new(),
                distinct: f.distinct,
                filter: filter.map(Bound::shown),
                within_group: shown_within,
            }))
            .with_origin(origin),
        );
        let expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    id: entry.id.clone(),
                    kind: entry.kind,
                    args: resolved,
                    distinct: f.distinct,
                    filter: filter.map(|b| Arc::clone(&b.expr)),
                    within_group: Vec::new(),
                    data_type: data_type.clone(),
                    nullable,
                    deterministic: entry.deterministic,
                },
            )))
            .with_origin(origin),
        );
        let mut lateral: Vec<AttrId> = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        Ok(Bound {
            expr,
            ty: arg_type(data_type, nullable),
            agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                Some(origin)
            } else {
                kids.iter().find_map(|k| k.agg)
            },
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral,
            shown: Some(shown_call),
        })
    }
}
