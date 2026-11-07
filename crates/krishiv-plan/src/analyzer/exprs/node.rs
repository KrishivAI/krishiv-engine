// SPDX-License-Identifier: Apache-2.0
//! `expr_node`: the resolution of one expression node from its resolved children.

use super::*;

impl Resolver<'_> {
    pub(super) fn expr_node_inner(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        in_agg: bool,
        window_fn: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        // The display-only `UTF8_BINARY` marker (ADR 0043, update 2026-11-07) is a plain STRING to
        // every operator and function: only an alias passes it on to the schema (calls re-mark
        // their result in `expr_node`).
        let plain_kids: Vec<Bound>;
        let reads_type = matches!(&e.node, ExprNode::Alias { .. })
            || matches!(&e.node, ExprNode::Function(f)
                if f.name.eq_ignore_ascii_case("typeof") || f.name.eq_ignore_ascii_case("collation"));
        let marked = |k: &Bound| {
            k.ty.data_type
                .as_ref()
                .is_some_and(|t| t.any_type(&DataType::is_binary_marker))
        };
        let kids = if !reads_type && kids.iter().any(marked) {
            plain_kids = kids
                .iter()
                .map(|k| {
                    let mut k = k.clone();
                    k.ty.data_type = k.ty.data_type.as_ref().map(DataType::without_binary_marker);
                    k
                })
                .collect();
            &plain_kids[..]
        } else {
            kids
        };
        let outer = kids.iter().filter_map(|k| k.outer).min();
        let agg = kids.iter().find_map(|k| k.agg);
        let win = kids.iter().find_map(|k| k.win);
        let mut lateral: Vec<AttrId> = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        let keep = |ty| Bound {
            expr: Arc::clone(e),
            ty,
            agg,
            win,
            outer,
            lateral: lateral.clone(),
            shown: None,
        };
        let rebuilt = |node: ExprNode| Arc::new(Expr::new(node).with_origin(e.origin));
        let args: Vec<Arc<Expr>> = kids.iter().map(|k| Arc::clone(&k.expr)).collect();
        let mut b = match &e.node {
            ExprNode::Column(c) => self.column(c, e.origin, in_agg, cx)?,
            ExprNode::Literal(
                Literal::YearMonthInterval { .. } | Literal::DayTimeInterval { .. },
            ) if self.ctx.conf.legacy_interval => {
                return Err(unsupported(
                    "interval literals under spark.sql.legacy.interval.enabled",
                    e.origin,
                ));
            }
            ExprNode::Literal(v) => keep(ArgType {
                nullable: matches!(v, Literal::Null(_)),
                data_type: literal_type(v),
            }),
            ExprNode::TypedLiteral {
                kind: TypedLiteralKind::Time,
                value,
            } => {
                if !self.ctx.conf.time_type {
                    return Err(AnalysisError::new(
                        krishiv_common::error::UnsupportedTimeType {},
                        e.origin,
                    ));
                }
                // A time is a string cast to `TIME(6)` (design P2-T181): the
                // value is checked again here, in case the plan came from a
                // front end other than the SQL parser.
                let Some(nanos) = krishiv_kernels::cast::parse_time(value) else {
                    return Err(AnalysisError::new(
                        krishiv_common::error::InvalidTypedLiteral {
                            value: format!("'{value}'"),
                            value_type: "\"TIME\"".to_owned(),
                        },
                        e.origin,
                    ));
                };
                let text = krishiv_kernels::cast::format_time(nanos);
                let text = Arc::new(Expr::from(Literal::String(text)).with_origin(e.origin));
                let cast = Expr::new(ExprNode::Cast {
                    expr: text,
                    to: DataType::Time { precision: 6 },
                    mode: CastMode::Ansi,
                })
                .with_origin(e.origin);
                Bound {
                    expr: Arc::new(cast),
                    shown: Some(Arc::clone(e)),
                    ..keep(ArgType {
                        nullable: false,
                        data_type: Some(DataType::Time { precision: 6 }),
                    })
                }
            }
            // `DATE 'today'` and `TIMESTAMP 'now'`: Spark's special values read
            // the clock; they bind as the clock calls they stand for (the
            // query-start instant), `epoch` as a constant.
            ExprNode::TypedLiteral { kind, value }
                if *kind != TypedLiteralKind::Time && is_special(value) =>
            {
                let special = value.trim().to_ascii_lowercase();
                if special == "epoch" {
                    let text = match special_kind(*kind, self.ctx.conf.timestamp_ntz) {
                        SpecialKind::Date => "1970-01-01",
                        SpecialKind::Instant => "1970-01-01 00:00:00+00:00",
                        SpecialKind::Local => "1970-01-01 00:00:00",
                    };
                    let literal = typed_literal(*kind, text, &self.ctx.conf, e.origin)?;
                    let ty = ArgType {
                        nullable: false,
                        data_type: literal_type(&literal),
                    };
                    return Ok(Bound {
                        expr: Arc::new(Expr::new(ExprNode::Literal(literal)).with_origin(e.origin)),
                        ..keep(ty)
                    });
                }
                let call =
                    special_clock(&special, special_kind(*kind, self.ctx.conf.timestamp_ntz))
                        .with_origin(e.origin);
                return self.expr(&Arc::new(call), cx);
            }
            // `TIMESTAMP '11:00'`: that time today, in the session zone (Spark reads a time of day alone
            // against the current date), bound as the clock calls it stands for.
            ExprNode::TypedLiteral { kind, value }
                if let Some(clock) = time_only_today(*kind, value, self.ctx.conf.timestamp_ntz) =>
            {
                return self.expr(&Arc::new(clock.with_origin(e.origin)), cx);
            }
            ExprNode::TypedLiteral { kind, value } => {
                let literal = typed_literal(*kind, value, &self.ctx.conf, e.origin)?;
                let ty = ArgType {
                    nullable: false,
                    data_type: literal_type(&literal),
                };
                // Shown as the resolved literal: Spark's names show the
                // normalized value (`TIMESTAMP '2016-03-11 20:54:00'` for
                // `.000`), formatted in the session zone by `pretty`.
                Bound {
                    expr: Arc::new(Expr::new(ExprNode::Literal(literal)).with_origin(e.origin)),
                    ..keep(ty)
                }
            }
            ExprNode::Star(_) => {
                return Err(AnalysisError::new(
                    InvalidUsageOfStarOrRegex {
                        elem: "'*'".to_owned(),
                        pretty_name: cx.op.to_owned(),
                    },
                    e.origin,
                ));
            }
            // A lambda outside a higher-order function's argument list:
            // lambdas in argument position are bound by the higher-order
            // call itself (see `hof`), never here. An already-bound lambda
            // (re-analysis of a resolved plan) is preserved as-is; its body
            // already holds `LambdaVar` leaves.
            ExprNode::Lambda { body, .. } => {
                let bound = crate::walk::visit_distinct(crate::walk::Node::Expr(body), |n| {
                    if let crate::walk::Node::Expr(x) = n
                        && matches!(&x.node, ExprNode::Resolved(ResolvedExpr::LambdaVar(_)))
                    {
                        return std::ops::ControlFlow::Break(());
                    }
                    std::ops::ControlFlow::Continue(())
                })
                .is_break();
                if bound && let Some(kid) = kids.first() {
                    keep(kid.ty.clone())
                } else {
                    return Err(unsupported(
                        "lambda functions outside a higher-order function",
                        e.origin,
                    ));
                }
            }
            // `m.k` on a map is the lookup `m['k']` (Spark's `ExtractValue`).
            ExprNode::Field { base: _, name }
                if matches!(kids[0].ty.data_type, Some(DataType::Map(_))) =>
            {
                let call = Expr::function(
                    "__subscript",
                    vec![
                        Expr::clone(&kids[0].expr),
                        Expr::new(ExprNode::Literal(Literal::String(name.clone())))
                            .with_origin(e.origin),
                    ],
                )
                .with_origin(e.origin);
                let mut b = self.expr(&Arc::new(call), cx)?;
                b.carry_markers(kids);
                return Ok(b);
            }
            ExprNode::Field { base: _, name } => {
                let (expr, ty) = self.struct_field(e, &kids[0], name)?;
                Bound {
                    expr,
                    ty: ty.clone(),
                    ..keep(ty)
                }
            }
            ExprNode::UpdateFields { ops, .. } => {
                return self.update_fields(e, ops, kids, cx);
            }
            ExprNode::Function(f) => {
                let n = f.args.len();
                // The filter is the child after the arguments, when the
                // call has one; any `WITHIN GROUP` ordering keys follow it
                // and are never the filter.
                if f.filter.is_some()
                    && let Some(filter) = kids.get(n)
                    && filter.ty.data_type != Some(DataType::Boolean)
                {
                    return Err(AnalysisError::new(
                        InvalidAggregateFilterNotBoolean {
                            filter_expr: format!(
                                "\"{}\"",
                                pretty(&filter.shown(), self.ctx.conf.session_zone)
                            ),
                        },
                        e.origin,
                    ));
                }
                if f.filter.is_some()
                    && let Some(filter) = kids.get(n)
                    && filter.win.is_some()
                {
                    return Err(super::super::windows::window_in_filter(filter, e.origin));
                }
                // `builtin.f` / `system.builtin.f` resolve to the builtin;
                // `session.f` / `system.session.f` reach a session UDF only;
                // any other qualifier is an unresolved routine.
                // Grouping functions are never qualified.
                // A function stored in a namespace of the session's catalog,
                // named by its qualifier (`default.f`, `spark_catalog.default.f`).
                if !f.qualifier.is_empty()
                    && !super::super::bind::is_session_qualifier(&f.qualifier)
                    && !super::super::bind::is_builtin_qualifier(&f.qualifier)
                {
                    let mut named = f.qualifier.clone();
                    named.push(f.name.clone());
                    if let FunctionLookup::Sql(def) =
                        self.ctx.functions.lookup(&named, self.ctx.profile)
                    {
                        return self.sql_call(&def, e, f, kids, cx.op);
                    }
                }
                super::super::bind::check_function_namespace(self.ctx, f, e.origin)?;
                let (parts, builtin) = qualified_parts(f, &self.b.search_path, e.origin)?;
                let session_qualified = super::super::bind::is_session_qualifier(&f.qualifier);
                let lookup_name = builtin.as_deref().unwrap_or(&f.name);
                // A SQL-bodied function of the session: the call is replaced by
                // its body. Built-ins and the operators rank first.
                if !session_qualified
                    && builtin.is_none()
                    && f.qualifier.is_empty()
                    && GroupingFn::from_name(lookup_name).is_none()
                    && builtin_op(lookup_name, n).is_none()
                    && let FunctionLookup::Sql(def) = self
                        .ctx
                        .functions
                        .lookup(std::slice::from_ref(&f.name), self.ctx.profile)
                {
                    return self.sql_call(&def, e, f, kids, cx.op);
                }
                // `session.f` / `system.session.f` reach a temporary SQL
                // function as well as a session UDF.
                if session_qualified
                    && let FunctionLookup::Sql(def) =
                        self.ctx.functions.lookup(&parts, self.ctx.profile)
                {
                    return self.sql_call(&def, e, f, kids, cx.op);
                }
                // A call of a function with a named-parameter signature is
                // rearranged into positional order (Spark's
                // `FunctionBuilderBase` rearrangement), filling defaults; any
                // other call with named arguments is unsupported.
                let signature = if session_qualified {
                    None
                } else {
                    named_signature(lookup_name)
                };
                let mut kids = kids.to_vec();
                if signature.is_some() {
                    let filter = kids.get(n).cloned();
                    // Any `WITHIN GROUP` ordering keys follow the filter;
                    // keep them after it (only `mask` and the unimplemented
                    // `count_min_sketch` rearrange, and both refuse the
                    // ordering, but the slice below must still count it).
                    let orders: Vec<Bound> = kids
                        .get(n + usize::from(f.filter.is_some())..)
                        .unwrap_or(&[])
                        .to_vec();
                    let mut reordered = self.rearrange_named(
                        &parts,
                        signature.unwrap_or(&[]),
                        f,
                        &kids[..n],
                        e.origin,
                    )?;
                    reordered.extend(filter);
                    reordered.extend(orders);
                    kids = reordered;
                } else if f.arg_names.iter().any(Option::is_some) {
                    return Err(AnalysisError::new(
                        NamedParametersNotSupported {
                            function_name: quoted(&parts),
                        },
                        e.origin,
                    ));
                }
                // The bound argument children: after any named-argument
                // rearrangement (which fills defaults, so `f.args.len()`
                // undercounts `mask`), minus the filter and any
                // `WITHIN GROUP` ordering keys (which follow it, so
                // `kids.len()` overcounts `mode`/`listagg`).
                let n = kids.len() - usize::from(f.filter.is_some()) - f.within_group.len();
                if !session_qualified && let Some(which) = GroupingFn::from_name(lookup_name) {
                    return self.grouping_call(which, e, &kids[..n], cx);
                }
                // `to_char(datetime, fmt)` and `to_varchar` are `date_format` to
                // Spark (`ToCharacterBuilder`), which names them so; a number or
                // a binary keeps `to_char`.
                if !session_qualified
                    && n == 2
                    && !window_fn
                    && (lookup_name.eq_ignore_ascii_case("to_char")
                        || lookup_name.eq_ignore_ascii_case("to_varchar"))
                    && matches!(
                        kids[0].ty.data_type,
                        Some(DataType::Date | DataType::Timestamp | DataType::TimestampNtz)
                    )
                {
                    // Spark builds `date_format` first, so its clauses are
                    // refused under that name.
                    let syntax = if f.distinct {
                        Some("DISTINCT")
                    } else if f.filter.is_some() {
                        Some("FILTER CLAUSE")
                    } else if !f.within_group.is_empty() {
                        Some("WITHIN GROUP (ORDER BY ...)")
                    } else {
                        None
                    };
                    if let Some(syntax) = syntax {
                        return Err(AnalysisError::new(
                            InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                                pretty_name: "`date_format`".to_owned(),
                                syntax: syntax.to_owned(),
                            },
                            e.origin,
                        ));
                    }
                    let call = Expr::function(
                        "date_format",
                        vec![Expr::clone(&kids[0].expr), Expr::clone(&kids[1].expr)],
                    )
                    .with_origin(e.origin);
                    let mut b = self.expr(&Arc::new(call), cx)?;
                    b.carry_markers(&kids[..n]);
                    b.shown = Some(Arc::new(
                        Expr::function(
                            "date_format",
                            vec![kids[0].shown(), kids[1].shown()]
                                .into_iter()
                                .map(|k| Expr::clone(&k))
                                .collect(),
                        )
                        .with_origin(e.origin),
                    ));
                    return Ok(b);
                }
                // Under `spark.sql.timestampType = TIMESTAMP_NTZ`, `to_timestamp`
                // and the numeric forms of `make_timestamp` produce a
                // `TIMESTAMP_NTZ` (a zone argument is ignored) and Spark
                // still names them as written.
                if !session_qualified
                    && self.ctx.conf.timestamp_ntz
                    && let Some((ntz, take)) = ntz_variant(lookup_name, &kids[..n])
                {
                    let call = Expr::function(
                        ntz,
                        kids[..take].iter().map(|k| Expr::clone(&k.expr)).collect(),
                    )
                    .with_origin(e.origin);
                    // The children are bound already, and rebinding a
                    // resolved child drops its markers: carry them over.
                    let mut b = self.expr(&Arc::new(call), cx)?;
                    b.carry_markers(&kids[..take]);
                    b.shown = Some(Arc::new(
                        Expr::function(
                            lookup_name.to_ascii_lowercase(),
                            kids[..n].iter().map(|k| Expr::clone(&k.shown())).collect(),
                        )
                        .with_origin(e.origin),
                    ));
                    return Ok(b);
                }
                if !session_qualified
                    && let Some(b) =
                        self.interval_operator(lookup_name, e.origin, &kids[..n], cx)?
                {
                    return Ok(b);
                }
                // `date + int`, `int + date` and `date - int` are `date_add`
                // and `date_sub` to Spark, which names them so.
                if !session_qualified
                    && n == 2
                    && let Some((name, swap)) =
                        date_with_days(lookup_name, &kids[0].ty, &kids[1].ty)
                {
                    let (first, second) = if swap { (1, 0) } else { (0, 1) };
                    let call = Expr::function(
                        name,
                        vec![
                            Expr::clone(&kids[first].expr),
                            Expr::clone(&kids[second].expr),
                        ],
                    )
                    .with_origin(e.origin);
                    // The children are bound already, and rebinding a
                    // resolved child drops its markers: carry them over.
                    let mut b = self.expr(&Arc::new(call), cx)?;
                    b.carry_markers(&kids[..n]);
                    // The `try_` forms are nullable, as Spark types them.
                    if lookup_name.starts_with("try_") {
                        b = make_nullable(b);
                    }
                    // Shown as Spark names it, over the operands as written
                    // (a session variable keeps its name).
                    b.shown = Some(Arc::new(
                        Expr::function(
                            name,
                            vec![kids[first].shown(), kids[second].shown()]
                                .into_iter()
                                .map(|k| bare_variable(&k))
                                .collect(),
                        )
                        .with_origin(e.origin),
                    ));
                    return Ok(b);
                }
                if !session_qualified && let Some(op) = builtin_op(lookup_name, n) {
                    // `ILIKE` and `LIKE|ILIKE ANY|SOME|ALL` are rewritten to
                    // the calls they denote, keeping Spark's display name.
                    if op == Builtin::ILike {
                        return self.ilike_call(e, &kids[..n]);
                    }
                    if op.quantified_like().is_some() {
                        return self.like_quantified_call(op, e, &kids[..n]);
                    }
                    return self.operator_call(op, e.origin, &kids[..n]);
                }
                let mut entry = if session_qualified {
                    match self.ctx.functions.lookup(&parts, self.ctx.profile) {
                        FunctionLookup::Found(entry) => entry,
                        FunctionLookup::KnownUnsupported => {
                            return Err(unsupported(
                                format!("function {}", parts.join(".")),
                                e.origin,
                            ));
                        }
                        FunctionLookup::NotFound | FunctionLookup::Sql(_) => {
                            return Err(AnalysisError::new(
                                UnresolvedRoutine {
                                    routine_name: quoted(&parts),
                                    search_path: self.b.search_path.clone(),
                                },
                                e.origin,
                            ));
                        }
                    }
                } else {
                    match self.ctx.functions.lookup(
                        std::slice::from_ref(&lookup_name.to_owned()),
                        self.ctx.profile,
                    ) {
                        // A builtin-qualified call (`builtin.f`,
                        // `system.builtin.f`) reaches built-ins only: a
                        // session UDF of the same name stays unresolved.
                        FunctionLookup::Found(entry)
                            if builtin.is_some() && matches!(entry.id, FunctionId::Udf { .. }) =>
                        {
                            return Err(AnalysisError::new(
                                UnresolvedRoutine {
                                    routine_name: quoted(&parts),
                                    search_path: self.b.search_path.clone(),
                                },
                                e.origin,
                            ));
                        }
                        FunctionLookup::Found(entry) => entry,
                        FunctionLookup::KnownUnsupported => {
                            return Err(unsupported(
                                format!("function {}", parts.join(".")),
                                e.origin,
                            ));
                        }
                        FunctionLookup::NotFound | FunctionLookup::Sql(_) => {
                            return Err(AnalysisError::new(
                                UnresolvedRoutine {
                                    routine_name: quoted(&parts),
                                    search_path: self.b.search_path.clone(),
                                },
                                e.origin,
                            ));
                        }
                    }
                };
                // A session aggregate UDF (ADR 0045): `DISTINCT` only when it
                // allows it (a window refuses `DISTINCT` for every aggregate).
                if !is_builtin_entry(&entry)
                    && entry.kind == FunctionKind::Aggregate
                    && f.distinct
                    && self.ctx.functions.rejects_distinct(entry.id.name())
                {
                    return Err(AnalysisError::new(
                        InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                            pretty_name: format!("`{lookup_name}`"),
                            syntax: "DISTINCT".to_owned(),
                        },
                        e.origin,
                    ));
                }
                // `array_agg` is Spark's spelling of `collect_list` (its
                // result column is named `collect_list(...)`): resolve it
                // as `collect_list` throughout.
                if is_builtin_entry(&entry) && entry.id.name() == "array_agg" {
                    entry = match self
                        .ctx
                        .functions
                        .lookup(&["collect_list".to_owned()], self.ctx.profile)
                    {
                        FunctionLookup::Found(entry) => entry,
                        _ => {
                            return Err(unsupported("function array_agg".to_owned(), e.origin));
                        }
                    };
                }
                // `string_agg` is Spark's alias of `listagg` (its result
                // column is named `listagg(...)`): resolve it as `listagg`
                // throughout.
                if is_builtin_entry(&entry) && entry.id.name() == "string_agg" {
                    entry = match self
                        .ctx
                        .functions
                        .lookup(&["listagg".to_owned()], self.ctx.profile)
                    {
                        FunctionLookup::Found(entry) => entry,
                        _ => {
                            return Err(unsupported("function string_agg".to_owned(), e.origin));
                        }
                    };
                }
                // `mode` and `listagg` take `WITHIN GROUP` orderings, which
                // lower into the resolved call (see `mode_call` and
                // `listagg_call`); any other function keeps Spark's
                // `FUNCTION_WITH_UNSUPPORTED_SYNTAX` error.
                if !f.within_group.is_empty()
                    && !(is_builtin_entry(&entry)
                        && matches!(
                            entry.id.name(),
                            "mode" | "listagg" | "percentile_cont" | "percentile_disc"
                        ))
                {
                    return Err(AnalysisError::new(
                        InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                            pretty_name: format!("`{lookup_name}`"),
                            syntax: "WITHIN GROUP (ORDER BY ...)".to_owned(),
                        },
                        e.origin,
                    ));
                }
                if !window_fn && entry.kind == FunctionKind::Aggregate {
                    self.nondeterministic_arguments(e, &kids[..n]);
                }
                if is_builtin_entry(&entry) && entry.id.name() == "mode" {
                    return self.mode_call(&entry, e, f, &kids, window_fn);
                }
                if is_builtin_entry(&entry) && entry.id.name() == "listagg" {
                    return self.listagg_call(&entry, e, f, &kids, window_fn);
                }
                if is_builtin_entry(&entry)
                    && matches!(
                        entry.id.name(),
                        "percentile"
                            | "percentile_approx"
                            | "approx_percentile"
                            | "median"
                            | "percentile_cont"
                            | "percentile_disc"
                            | "approx_count_distinct"
                    )
                {
                    return self.quantile_call(&entry, e, f, &kids, n, window_fn);
                }
                if !window_fn {
                    call_checks(&entry, e, &kids, n)?;
                }
                // `struct(a, ...)`: children with names keep them (a column
                // contributes its name, a struct field its field name), so a
                // call with any named child becomes `named_struct(name, ...)`
                // with `colN` defaults; all-default calls stay `struct`
                // (even one written with aliases: the alias is its default).
                // Only for the built-in: a qualified session UDF named
                // `struct` runs the UDF.
                if is_builtin_entry(&entry)
                    && entry.id.name() == "struct"
                    && let Some(rewritten) = self.struct_names(f, &kids[..n], e.origin)
                {
                    // The children are bound already, and rebinding a
                    // resolved child drops its markers: carry them over.
                    let mut b = self.expr(&rewritten, cx)?;
                    b.carry_markers(&kids[..n]);
                    // Shown as written (`struct(1 AS a)`), not the
                    // `named_struct` it binds as: aliases keep `AS name`.
                    let mut shown_kids = Vec::with_capacity(n);
                    for (arg, k) in f.args.iter().zip(&kids[..n]) {
                        if let ExprNode::Alias { name, .. } = &arg.node {
                            // The bound child is the alias itself: show what it names.
                            let shown = k.shown();
                            let inner = match &shown.node {
                                ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => {
                                    Arc::clone(expr)
                                }
                                _ => shown,
                            };
                            shown_kids.push(Arc::new(
                                Expr::new(ExprNode::Alias {
                                    expr: inner,
                                    name: name.clone(),
                                })
                                .with_origin(arg.origin),
                            ));
                        } else {
                            shown_kids.push(k.shown());
                        }
                    }
                    b.shown = Some(with_children(e, shown_kids));
                    return Ok(b);
                }
                // `max_by(x, y, k)` / `min_by(x, y, k)`: the `k` values of `x`
                // with the largest (smallest) non-NULL `y`, best first, as
                // `slice(sort_array(collect_list(named_struct('k', y, 'v', x))
                // FILTER (WHERE y IS NOT NULL), asc), 1, k).v`. The group's
                // pairs are held (Spark keeps a heap of `k`); ties order by `x`.
                if is_builtin_entry(&entry)
                    && matches!(entry.id.name(), "max_by" | "min_by")
                    && n == 3
                    && !window_fn
                    && !f.distinct
                {
                    // `k` is an INT to Spark (implicit cast): a fractional,
                    // decimal or string constant is validated as the INT it
                    // casts to, and the rewrite uses that cast.
                    // Only numbers and strings cast: an interval or other type stays
                    // as written and fails the signature check as before.
                    let castable = matches!(
                        kids[2].ty.data_type,
                        Some(
                            DataType::Byte
                                | DataType::Short
                                | DataType::Integer
                                | DataType::Long
                                | DataType::Float
                                | DataType::Double
                                | DataType::Decimal(_)
                                | DataType::String
                        ) | None
                    );
                    if !castable {
                        // Spark's `k` is an INT with implicit casts only from
                        // numbers and strings: anything else is a type mismatch.
                        let shown = with_children(e, kids[..n].iter().map(Bound::shown).collect());
                        return Err(AnalysisError::new(
                            DatatypeMismatchUnexpectedInputType {
                                sql_expr: format!(
                                    "\"{}\"",
                                    pretty(&shown, self.ctx.conf.session_zone)
                                ),
                                param_index: "third".to_owned(),
                                required_type: "\"INT\"".to_owned(),
                                input_sql: format!(
                                    "\"{}\"",
                                    pretty(&kids[2].shown(), self.ctx.conf.session_zone)
                                ),
                                input_type: quoted_type(kids[2].ty.data_type.as_ref()),
                            },
                            e.origin,
                        ));
                    }
                    let k_int = if castable {
                        self.implicit_cast(&kids[2], Some(DataType::Integer))
                    } else {
                        kids[2].clone()
                    };
                    // A NULL count (written, or a string that fails to cast) is
                    // as out of range as 0 is.
                    let bad_k = match self.fold(&k_int.expr, self.ctx.conf.ansi) {
                        Some(Constant::Integral(k)) if !(1..=100_000).contains(&k) => {
                            Some(k.to_string())
                        }
                        Some(Constant::Null) => Some("NULL".to_owned()),
                        // A count that is not a constant (a column, `rand()`) is refused as
                        // Spark does: `k` must be foldable.
                        None => {
                            let shown =
                                with_children(e, kids[..n].iter().map(Bound::shown).collect());
                            return Err(AnalysisError::new(
                                DatatypeMismatchNonFoldableInput {
                                    sql_expr: format!(
                                        "\"{}\"",
                                        pretty(&shown, self.ctx.conf.session_zone)
                                    ),
                                    input_name: "k".to_owned(),
                                    input_type: "int".to_owned(),
                                    // Spark shows the implicit cast to INT it added.
                                    input_expr: {
                                        let text =
                                            pretty(&kids[2].shown(), self.ctx.conf.session_zone);
                                        if matches!(kids[2].ty.data_type, Some(DataType::Integer)) {
                                            text
                                        } else {
                                            format!("CAST({text} AS INT)")
                                        }
                                    },
                                },
                                e.origin,
                            ));
                        }
                        _ => None,
                    };
                    if let Some(current) = bad_k {
                        let shown = with_children(e, kids[..n].iter().map(Bound::shown).collect());
                        return Err(AnalysisError::new(
                            DatatypeMismatchValueOutOfRange {
                                expr_name: "`k`".to_owned(),
                                value_range: "[1, 100000]".to_owned(),
                                current_value: current,
                                sql_expr: format!(
                                    "\"{}\"",
                                    pretty(&shown, self.ctx.conf.session_zone)
                                ),
                            },
                            e.origin,
                        ));
                    }
                    let origin = e.origin;
                    let lit =
                        |l: Literal| Arc::new(Expr::new(ExprNode::Literal(l)).with_origin(origin));
                    let mk = |name: &str, args: Vec<Arc<Expr>>, filter: Option<Arc<Expr>>| {
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
                    let (x, y, k) = (&kids[0].expr, &kids[1].expr, &k_int.expr);
                    let pair = mk(
                        "named_struct",
                        vec![
                            lit(Literal::String("k".to_owned())),
                            Arc::clone(y),
                            lit(Literal::String("v".to_owned())),
                            Arc::clone(x),
                        ],
                        None,
                    );
                    let keep = f
                        .filter
                        .is_some()
                        .then(|| kids.get(n).map(|b| Arc::clone(&b.expr)))
                        .flatten();
                    let present = mk("isnotnull", vec![Arc::clone(y)], None);
                    let filter = match keep {
                        Some(k) => mk("and", vec![present, k], None),
                        None => present,
                    };
                    let collected = mk("collect_list", vec![pair], Some(filter));
                    let ascending = entry.id.name() == "min_by";
                    // The entries order by `y` only: `x` is an opaque payload.
                    // `sort_array` orders the whole `(y, x)` struct (ties by
                    // `x`), which needs `x` orderable; a value holding a map
                    // sorts through a comparator on `y` instead.
                    let sorted = if kids[0].ty.data_type.as_ref().is_some_and(holds_map) {
                        let var = |name: &str| {
                            Arc::new(
                                Expr::new(ExprNode::Column(crate::ColumnRef::new(vec![
                                    name.to_owned(),
                                ])))
                                .with_origin(origin),
                            )
                        };
                        let key = |name: &str| {
                            mk(
                                "__subscript",
                                vec![var(name), lit(Literal::String("k".to_owned()))],
                                None,
                            )
                        };
                        let (before, after) = if ascending { (-1, 1) } else { (1, -1) };
                        let compare = mk(
                            "if",
                            vec![
                                mk("<", vec![key("l"), key("r")], None),
                                lit(Literal::Integer(before)),
                                mk(
                                    "if",
                                    vec![
                                        mk(">", vec![key("l"), key("r")], None),
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
                        mk("array_sort", vec![collected, comparator], None)
                    } else {
                        mk(
                            "sort_array",
                            vec![collected, lit(Literal::Boolean(ascending))],
                            None,
                        )
                    };
                    let top = mk(
                        "slice",
                        vec![sorted, lit(Literal::Integer(1)), Arc::clone(k)],
                        None,
                    );
                    let rewritten = mk(
                        "__subscript",
                        vec![top, lit(Literal::String("v".to_owned()))],
                        None,
                    );
                    let mut b = self.expr(&rewritten, cx)?;
                    b.carry_markers(&kids[..n]);
                    b.shown = Some(with_children(
                        e,
                        kids[..n].iter().map(Bound::shown).collect(),
                    ));
                    return Ok(b);
                }
                if is_builtin_entry(&entry) && entry.id.name() == "count" && n == 0 {
                    return self.parameterless_count(e, &entry, &kids, agg, window_fn);
                }
                // `assert_true(c)` is `assert_true(c, '<c>' is not true!)`:
                // Spark's constructor materializes the default message, so
                // the schema names it (`assert_true(true, 'true' is not
                // true!)`) and the kernel raises it. Only for the built-in:
                // a qualified session UDF named `assert_true` takes its
                // arguments as written.
                // Only for built-ins: a qualified session UDF named `assert_true`
                // takes its arguments as written, as do the other built-in
                // constructors below and the conditional rewrites.
                // The catalog probes up to six arguments, too few to see
                // these limits.
                if is_builtin_entry(&entry)
                    && let Some((least, total, _)) = interval_defaults(entry.id.name())
                    && n > total
                {
                    let counts: Vec<String> = (least..=total).map(|c| c.to_string()).collect();
                    return Err(AnalysisError::new(
                        WrongNumArgsWithoutSuggestion {
                            function_name: format!("`{}`", entry.id.name()),
                            expected_num: format!("[{}]", counts.join(", ")),
                            actual_num: n.to_string(),
                            docroot: SPARK_DOC_ROOT.to_owned(),
                        },
                        e.origin,
                    ));
                }
                // `aes_encrypt` takes two to six arguments (seven pass the six the catalog probes).
                if is_builtin_entry(&entry) && entry.id.name() == "aes_encrypt" && n > 6 {
                    return Err(AnalysisError::new(
                        WrongNumArgsWithoutSuggestion {
                            function_name: "`aes_encrypt`".to_owned(),
                            expected_num: "[2, 3, 4, 5, 6]".to_owned(),
                            actual_num: n.to_string(),
                            docroot: SPARK_DOC_ROOT.to_owned(),
                        },
                        e.origin,
                    ));
                }
                let mut owned: Vec<Bound>;
                let (kids, n) = if is_builtin_entry(&entry)
                    && entry.id.name() == "assert_true"
                    && n == 1
                {
                    owned = kids.clone();
                    owned.insert(
                        1,
                        assert_message(&owned[0], e.origin, self.ctx.conf.session_zone),
                    );
                    (&owned[..], 2)
                } else if is_builtin_entry(&entry)
                    && entry.id.name() == "mask"
                    && (1..5).contains(&n)
                {
                    owned = kids[..n].to_vec();
                    owned.extend(MASK_DEFAULTS.iter().skip(n - 1).map(|d| match d {
                        Some(text) => default_argument(
                            Literal::String((*text).to_owned()),
                            DataType::String,
                            false,
                            e.origin,
                        ),
                        None => default_argument(
                            Literal::Null(Some(DataType::String)),
                            DataType::String,
                            true,
                            e.origin,
                        ),
                    }));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], 5)
                } else if is_builtin_entry(&entry)
                    && matches!(entry.id.name(), "st_geomfromwkb" | "st_asbinary")
                    && n == 1
                {
                    // `st_geomfromwkb(wkb)` is `st_geomfromwkb(wkb, 0)` and
                    // `st_asbinary(g)` is `st_asbinary(g, 'NDR')`.
                    owned = kids[..n].to_vec();
                    owned.push(if entry.id.name() == "st_asbinary" {
                        default_argument(
                            Literal::String("NDR".to_owned()),
                            DataType::String,
                            false,
                            e.origin,
                        )
                    } else {
                        default_argument(Literal::Integer(0), DataType::Integer, false, e.origin)
                    });
                    (&owned[..], 2)
                } else if is_builtin_entry(&entry)
                    && matches!(entry.id.name(), "locate" | "position")
                    && n == 2
                {
                    // `locate(sub, str)` is `locate(sub, str, 1)`.
                    owned = kids[..n].to_vec();
                    owned.push(default_argument(
                        Literal::Integer(1),
                        DataType::Integer,
                        false,
                        e.origin,
                    ));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], 3)
                } else if is_builtin_entry(&entry)
                    && let Some((least, total, seconds)) = interval_defaults(entry.id.name())
                    && (least..total).contains(&n)
                {
                    // `make_dt_interval(1)` is `make_dt_interval(1, 0, 0,
                    // 0.000000)`; Spark names the column with the defaults.
                    owned = kids[..n].to_vec();
                    owned.extend((n..total).map(|at| {
                        if seconds && at + 1 == total {
                            let seconds =
                                DecimalType::try_new(18, 6).unwrap_or(DecimalType::USER_DEFAULT);
                            default_argument(
                                Literal::Decimal {
                                    unscaled: 0,
                                    data_type: seconds,
                                },
                                DataType::Decimal(seconds),
                                false,
                                e.origin,
                            )
                        } else {
                            default_argument(
                                Literal::Integer(0),
                                DataType::Integer,
                                false,
                                e.origin,
                            )
                        }
                    }));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], total)
                } else if is_builtin_entry(&entry) && entry.id.name() == "sort_array" && n == 1 {
                    // `sort_array(arr)` is `sort_array(arr, true)`.
                    owned = kids[..n].to_vec();
                    owned.push(default_argument(
                        Literal::Boolean(true),
                        DataType::Boolean,
                        false,
                        e.origin,
                    ));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], 2)
                } else if is_builtin_entry(&entry)
                    && let Some(default) = regexp_default_index(entry.id.name(), n)
                {
                    // `regexp_extract(str, re)` is `regexp_extract(str, re, 1)`,
                    // `regexp_instr(str, re)` takes index 0 and
                    // `regexp_replace(str, re, rep)` position 1; Spark names
                    // the column with the default shown.
                    owned = kids[..n].to_vec();
                    owned.push(default_argument(
                        Literal::Integer(default),
                        DataType::Integer,
                        false,
                        e.origin,
                    ));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], n + 1)
                } else if is_builtin_entry(&entry) && entry.id.name() == "split" && n == 2 {
                    // `split(str, pattern)` is `split(str, pattern, -1)`.
                    owned = kids[..n].to_vec();
                    owned.push(default_argument(
                        Literal::Integer(-1),
                        DataType::Integer,
                        false,
                        e.origin,
                    ));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], 3)
                } else if is_builtin_entry(&entry)
                    && let Some(at) = options_index(entry.id.name())
                    && n > at
                    && let Some(pairs) = literal_map_pairs(&kids[at].expr)
                {
                    // A `map('k', 'v', ...)` of string literals is the
                    // options literal Spark's `convertToMapData` reads at
                    // analysis; the executor binds it as a literal.
                    // Spark evaluates the options `map(...)`: a repeated key
                    // is `DUPLICATED_MAP_KEY` unless the session keeps the
                    // last one.
                    let mut distinct_pairs: Vec<(String, String)> = Vec::new();
                    for (key, value) in pairs {
                        if let Some(seen) = distinct_pairs.iter_mut().find(|(k, _)| *k == key) {
                            if !self.ctx.conf.map_key_dedup_last_win {
                                return Err(AnalysisError::new(
                                    DuplicatedMapKey {
                                        key,
                                        map_key_dedup_policy: "\"spark.sql.mapKeyDedupPolicy\""
                                            .to_owned(),
                                    },
                                    e.origin,
                                ));
                            }
                            seen.1 = value;
                        } else {
                            distinct_pairs.push((key, value));
                        }
                    }
                    let pairs = distinct_pairs;
                    owned = kids[..n].to_vec();
                    let original = Arc::clone(&owned[at].expr);
                    owned[at].shown = Some(Arc::clone(&original));
                    owned[at].expr = Arc::new(
                        Expr::new(ExprNode::Literal(Literal::Map(pairs)))
                            .with_origin(original.origin),
                    );
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], n)
                } else if is_builtin_entry(&entry)
                    && matches!(entry.id.name(), "vector_norm" | "vector_normalize")
                    && n == 1
                {
                    // `vector_norm(v)` is `vector_norm(v, 2.0)`.
                    owned = kids[..n].to_vec();
                    owned.push(default_argument(
                        Literal::Float(2.0),
                        DataType::Float,
                        false,
                        e.origin,
                    ));
                    if let Some(filter) = kids.get(n) {
                        owned.push(filter.clone());
                    }
                    (&owned[..], 2)
                } else if is_builtin_entry(&entry) && entry.id.name() == "named_struct" && n > 0 {
                    owned = kids.clone();
                    // A foldable name that is not a literal becomes one, so
                    // the executor binds it (the result rule already read it
                    // folded; the shown form keeps the written argument for
                    // error quotes).
                    for k in owned.iter_mut().step_by(2) {
                        if !matches!(k.expr.node, ExprNode::Literal(_))
                            && let Some(Constant::String(name)) =
                                self.fold(&k.expr, self.ctx.conf.ansi)
                        {
                            let lit = Arc::new(
                                Expr::new(ExprNode::Literal(Literal::String(name)))
                                    .with_origin(k.expr.origin),
                            );
                            k.shown = Some(Arc::clone(&k.expr));
                            k.expr = lit;
                        }
                    }
                    (&owned[..], n)
                } else {
                    (&kids[..], n)
                };
                // Only for built-ins: a qualified session UDF named like a
                // conditional (`session.nvl`, `system.session.nvl`, ...) runs
                // the UDF, while a bare `nvl` runs the built-in (which wins
                // the search path).
                if is_builtin_entry(&entry)
                    && let Some(which) = Conditional::of(lookup_name, n)
                {
                    return self.conditional_call(which, &entry, e, f, kids);
                }
                // A cast-named function (`float(x)`, `int(x)`, ...) is a
                // cast, never a call: the plan only ever holds the `Cast`
                // node. `DISTINCT` and `FILTER` have no cast form: Spark
                // rejects them as unsupported syntax on the function. Only for
                // built-ins: a qualified session UDF named `int` runs the UDF.
                if is_builtin_entry(&entry)
                    && let Some(to) = cast_target(&f.name)
                {
                    if f.distinct {
                        return Err(AnalysisError::new(
                            InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                                pretty_name: "`cast`".to_owned(),
                                syntax: "DISTINCT".to_owned(),
                            },
                            e.origin,
                        ));
                    }
                    if f.filter.is_some() {
                        return Err(AnalysisError::new(
                            InvalidSqlSyntaxFunctionWithUnsupportedSyntax {
                                pretty_name: "`cast`".to_owned(),
                                syntax: "FILTER CLAUSE".to_owned(),
                            },
                            e.origin,
                        ));
                    }
                    return self.cast_named_call(&entry, e, f, kids, to);
                }
                // `s['a']` on a struct is the field `a` (Spark's
                // `ExtractValue`): the extraction is a non-null string.
                if is_builtin_entry(&entry)
                    && entry.id.name() == "__subscript"
                    && n == 2
                    && matches!(kids[0].ty.data_type, Some(DataType::Struct(_)))
                {
                    let ExprNode::Literal(Literal::String(name)) = &kids[1].expr.node else {
                        return Err(AnalysisError::new(
                            InvalidExtractFieldType {
                                extraction: format!(
                                    "\"{}\"",
                                    pretty(&kids[1].shown(), self.ctx.conf.session_zone)
                                ),
                            },
                            e.origin,
                        ));
                    };
                    let (expr, ty) = self.struct_field(e, &kids[0], name)?;
                    return Ok(Bound {
                        expr,
                        ty: ty.clone(),
                        ..keep(ty)
                    });
                }
                // `array_of_structs['a']` is Spark's `GetArrayStructFields`: the
                // field of every element, a NULL element giving NULL. It is
                // built as `transform(arr, x -> x.a)` once the field is known.
                if is_builtin_entry(&entry)
                    && entry.id.name() == "__subscript"
                    && n == 2
                    && let Some(DataType::Array(arr)) = &kids[0].ty.data_type
                    && let DataType::Struct(st) = arr.element_type()
                    && let ExprNode::Literal(Literal::String(name)) = &kids[1].expr.node
                {
                    let cs = self.ctx.conf.case_sensitive;
                    let Some(field) = st.fields().iter().find(|f| same(f.name(), name, cs)) else {
                        return Err(unresolved_field(
                            &[
                                pretty(&kids[0].shown(), self.ctx.conf.session_zone),
                                name.clone(),
                            ],
                            st,
                            e.origin,
                        ));
                    };
                    let element = Expr::new(ExprNode::Column(ColumnRef::new(vec![
                        "__subscript_element".to_owned(),
                    ])))
                    .with_origin(e.origin);
                    let body = Expr::new(ExprNode::Field {
                        base: Arc::new(element),
                        name: field.name().to_owned(),
                    })
                    .with_origin(e.origin);
                    let call = Expr::function(
                        "transform",
                        vec![
                            Expr::clone(&kids[0].expr),
                            Expr::lambda(vec!["__subscript_element".to_owned()], body),
                        ],
                    )
                    .with_origin(e.origin);
                    let mut b = self.expr(&Arc::new(call), cx)?;
                    b.carry_markers(&kids[..n]);
                    // Spark names it as the field of the array (`x.a`).
                    b.shown = Some(Arc::new(
                        Expr::new(ExprNode::Field {
                            base: Arc::new(Expr::clone(&kids[0].shown())),
                            name: field.name().to_owned(),
                        })
                        .with_origin(e.origin),
                    ));
                    return Ok(b);
                }
                let (mut call_kids, mut ty) = self.typed_call(&entry, e, &kids[..n])?;
                // `arrays_zip(a, b)` names its struct fields after columns,
                // struct fields and aliases (`0`, `1`, ... otherwise): the
                // result is renamed and the names follow the arrays as
                // string literals, which the executor reads.
                let mut zip_names: Vec<Arc<Expr>> = Vec::new();
                // Under `spark.sql.legacy.negativeIndexInArrayInsert`,
                // `array_insert` carries the switch as a trailing literal.
                if is_builtin_entry(&entry)
                    && entry.id.name() == "array_insert"
                    && self.ctx.conf.legacy_negative_index_in_array_insert
                {
                    zip_names.push(Arc::new(
                        Expr::new(ExprNode::Literal(Literal::Boolean(true))).with_origin(e.origin),
                    ));
                }
                // `extract(SECOND FROM t)` reads a `time(p)` to `p` digits
                // whatever digits the value holds (a string cast keeps all
                // of them, and Arrow's `Time64` does not carry `p`): the
                // precision follows as a trailing literal.
                if is_builtin_entry(&entry)
                    && matches!(entry.id.name(), "extract" | "date_part" | "datepart")
                    && n == 2
                    && let Some(DataType::Time { precision }) = &call_kids[1].ty.data_type
                    && *precision < 6
                {
                    zip_names.push(Arc::new(
                        Expr::new(ExprNode::Literal(Literal::Integer(i32::from(*precision))))
                            .with_origin(e.origin),
                    ));
                }
                if is_builtin_entry(&entry)
                    && entry.id.name() == "arrays_zip"
                    && let Some(DataType::Array(a)) = &ty.data_type
                    && let DataType::Struct(st) = a.element_type()
                {
                    let names: Vec<String> = kids[..n]
                        .iter()
                        .enumerate()
                        .map(|(i, k)| match &k.expr.node {
                            ExprNode::Resolved(
                                ResolvedExpr::Column(c)
                                | ResolvedExpr::OuterColumn(c)
                                | ResolvedExpr::LambdaVar(c),
                            ) => c.name.clone(),
                            ExprNode::Resolved(ResolvedExpr::Field { name, .. }) => name.clone(),
                            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => {
                                output.name.clone()
                            }
                            _ => i.to_string(),
                        })
                        .collect();
                    let fields = st
                        .fields()
                        .iter()
                        .zip(&names)
                        .map(|(f, name)| {
                            StructField::new(name.clone(), f.data_type().clone(), f.nullable())
                        })
                        .collect::<Vec<_>>();
                    ty.data_type = Some(DataType::Array(
                        ArrayType::new(
                            DataType::Struct(StructType::new(fields)),
                            a.contains_null(),
                        )
                        .into(),
                    ));
                    zip_names = names
                        .into_iter()
                        .map(|name| {
                            Arc::new(
                                Expr::new(ExprNode::Literal(Literal::String(name)))
                                    .with_origin(e.origin),
                            )
                        })
                        .collect();
                }
                // `schema_of_csv(lit)` resolves below to the DDL literal.
                // The registry lookup is ASCII case-insensitive and
                // `id.name()` is the canonical lower-case name, so a
                // case-variant call (`SCHEMA_OF_CSV`) resolves here too.
                // Only for the built-in: a qualified session UDF of the
                // same name runs the UDF.
                let is_schema_of_csv = is_builtin_entry(&entry)
                    && matches!(
                        entry.id.name(),
                        "schema_of_csv" | "schema_of_xml" | "schema_of_json"
                    );
                let of_xml = entry.id.name() == "schema_of_xml";
                let of_json = entry.id.name() == "schema_of_json";
                // `from_csv` and `to_csv` show only their first argument.
                let is_csv = is_builtin_entry(&entry)
                    && matches!(
                        entry.id.name(),
                        "from_csv" | "to_csv" | "from_xml" | "to_xml"
                    );
                if is_builtin_entry(&entry) && entry.id.name() == "mask" {
                    self.check_mask_replacements(lookup_name, &kids[..n], &call_kids[..n])?;
                }
                let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
                call_kids.extend(kids.get(n).cloned());
                // `typeof(x)` is Spark's `RuntimeReplaceable`; its type
                // name is known once the call is typed. Only for the
                // built-in: a qualified session UDF named `typeof` runs
                // the UDF.
                let is_typeof = is_builtin_entry(&entry) && entry.id.name() == "typeof";
                let is_collation = is_builtin_entry(&entry) && entry.id.name() == "collation";
                let session_value_early = if is_builtin_entry(&entry) {
                    match entry.id.name() {
                        "current_schema" | "current_database" => {
                            Some(self.ctx.catalogs.current_namespace().join("."))
                        }
                        "current_path" => Some(super::super::current_path(&*self.ctx.catalogs)),
                        _ => None,
                    }
                } else {
                    None
                };
                let plain_args: Vec<Arc<Expr>> =
                    call_kids[..n].iter().map(|k| Arc::clone(&k.expr)).collect();
                let zipped = !zip_names.is_empty();
                let call = ResolvedFunction {
                    id: entry.id,
                    kind: entry.kind,
                    args: plain_args.iter().cloned().chain(zip_names).collect(),
                    distinct: f.distinct,
                    filter: call_kids.get(n).map(|k| Arc::clone(&k.expr)),
                    within_group: Vec::new(),
                    data_type,
                    nullable: ty.nullable,
                    deterministic: entry.deterministic,
                };
                // The names `arrays_zip` carries are not part of the call as
                // written.
                let plain = zipped.then(|| {
                    rebuilt(ExprNode::Resolved(ResolvedExpr::Function(
                        ResolvedFunction {
                            args: plain_args,
                            ..call.clone()
                        },
                    )))
                });
                let expr = rebuilt(ExprNode::Resolved(ResolvedExpr::Function(call)));
                let shown = match plain {
                    Some(plain) => self
                        .call_shown(&parts, &plain, kids, &call_kids)
                        .or(Some(plain)),
                    None => self.call_shown(&parts, &expr, kids, &call_kids),
                };
                // `typeof(x)` is Spark's `RuntimeReplaceable`: the analyzer
                // replaces it with the type name literal, keeping the call
                // as the shown form (the executor gets no type logic).
                // `current_schema()`, `current_database()` and `current_path()`
                // read the session: the analyzer replaces them with the
                // value, as the query sees it.
                let session_value = session_value_early;
                let (expr, shown) = if let Some(value) = session_value {
                    (
                        rebuilt(ExprNode::Literal(Literal::String(value))),
                        Some(expr),
                    )
                } else if is_collation {
                    // `collation(x)` is the qualified name of the collation of `x`
                    // (Spark's `RuntimeReplaceable` `Collation`); a plain string is `UTF8_BINARY`.
                    let child = call_kids.first();
                    let name = match child.and_then(|k| k.ty.data_type.as_ref()) {
                        Some(DataType::String) | None => {
                            krishiv_types::Collation::UTF8_BINARY.qualified_name()
                        }
                        Some(DataType::CollatedString(c)) => c.qualified_name(),
                        other => {
                            let shown = child
                                .map(|k| pretty(&k.shown(), self.ctx.conf.session_zone))
                                .unwrap_or_default();
                            return Err(AnalysisError::new(
                                krishiv_common::error::DatatypeMismatchUnexpectedInputType {
                                    sql_expr: format!("\"collation({shown})\""),
                                    param_index: "first".to_owned(),
                                    required_type: "\"STRING\"".to_owned(),
                                    input_sql: format!("\"{shown}\""),
                                    input_type: coerce::quoted_type(other),
                                },
                                e.origin,
                            ));
                        }
                    };
                    (
                        rebuilt(ExprNode::Literal(Literal::String(name))),
                        shown.or(Some(expr)),
                    )
                } else if is_typeof {
                    let name = call_kids
                        .first()
                        .and_then(|k| k.ty.data_type.clone())
                        .map_or_else(|| "void".to_owned(), |t| t.catalog_string());
                    (
                        rebuilt(ExprNode::Literal(Literal::String(name))),
                        shown.or(Some(expr)),
                    )
                } else {
                    (expr, shown)
                };
                let mut bound = Bound {
                    agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                        Some(e.origin)
                    } else {
                        agg
                    },
                    shown,
                    expr,
                    ..keep(ty)
                };
                // `schema_of_csv(lit)` resolves during analysis (Spark's
                // `SchemaOfCsv` evaluates its foldable child): the DDL
                // literal, shown as the call.
                if is_schema_of_csv {
                    self.resolve_schema_of_csv(e, &kids[..n], &mut bound, of_xml, of_json)?;
                }
                // `from_csv` and `to_csv` show only their first argument
                // (Spark's `toPrettySQL` drops the schema and options).
                if is_csv
                    && let Some(first) = kids.first()
                    && let ExprNode::Resolved(ResolvedExpr::Function(call)) = &bound.expr.node
                {
                    let mut new_call = call.clone();
                    new_call.args = vec![first.shown()];
                    bound.shown = Some(Arc::new(
                        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(new_call)))
                            .with_origin(bound.expr.origin),
                    ));
                }
                bound
            }
            ExprNode::Cast { to, mode, .. } => {
                let mode = match mode {
                    CastMode::Default if self.ctx.conf.ansi => CastMode::Ansi,
                    CastMode::Default => CastMode::Legacy,
                    m => *m,
                };
                // Spark's parser rejects the `TIME` type before any cast
                // is checked (`spark.sql.timeType.enabled` is false by
                // default; TIME cannot be evaluated here either way).
                if !self.ctx.conf.time_type && to.any_type(&|t| matches!(t, DataType::Time { .. }))
                {
                    return Err(AnalysisError::new(
                        krishiv_common::error::UnsupportedTimeType {},
                        e.origin,
                    ));
                }
                let child = &kids[0].ty;
                if let Err(bad) = coerce::can_cast(child.data_type.as_ref(), to, mode) {
                    return Err(bad_cast(
                        bad,
                        e.origin,
                        &kids[0],
                        to,
                        mode,
                        self.ctx.conf.session_zone,
                    ));
                }
                let nullable = child.nullable
                    || child.data_type.as_ref().is_some_and(|from| {
                        if mode == CastMode::Try {
                            coerce::try_nullable(from, to)
                        } else {
                            coerce::force_nullable(from, to)
                        }
                    });
                Bound {
                    expr: rebuilt(ExprNode::Cast {
                        expr: Arc::clone(&args[0]),
                        to: to.clone(),
                        mode,
                    }),
                    ..keep(arg_type(to.clone(), nullable))
                }
            }
            ExprNode::Collate {
                collation, name, ..
            } => {
                // Collations are the 4.2 profile's (ADR 0043 Decision 9).
                if self.ctx.profile < crate::analyzer::SparkProfile::V4_2 {
                    return Err(unsupported("COLLATE under this Spark profile", e.origin));
                }
                let child = &kids[0].ty;
                if !matches!(
                    child.data_type.as_ref(),
                    Some(DataType::String | DataType::CollatedString(_)) | None
                ) {
                    return Err(AnalysisError::new(
                        krishiv_common::error::DatatypeMismatchUnexpectedInputType {
                            sql_expr: format!(
                                "\"collate({}, {name})\"",
                                pretty(&kids[0].shown(), self.ctx.conf.session_zone)
                            ),
                            param_index: "first".to_owned(),
                            required_type: "\"STRING\"".to_owned(),
                            input_sql: format!(
                                "\"{}\"",
                                pretty(&kids[0].shown(), self.ctx.conf.session_zone)
                            ),
                            input_type: coerce::quoted_type(child.data_type.as_ref()),
                        },
                        e.origin,
                    ));
                }
                // A cast to the collated type: the carrier is unchanged, so it costs nothing at
                // run time. The unresolved node stays the display form.
                let to = DataType::collated_string(*collation);
                let mut bound = Bound {
                    expr: rebuilt(ExprNode::Cast {
                        expr: Arc::clone(&args[0]),
                        to: to.clone(),
                        mode: CastMode::Ansi,
                    }),
                    ..keep(arg_type(to, child.nullable))
                };
                bound.shown = Some(Arc::clone(e));
                bound
            }
            // Generators take their several names before this point
            // (`extract_generator`, lowering): any other expression with
            // a multi-part alias is an error.
            ExprNode::MultiAlias { expr, names } => {
                return Err(AnalysisError::new(
                    krishiv_common::error::MultiAliasWithoutGenerator {
                        names: names.join(", "),
                        expr: format!("\"{}\"", pretty(expr, self.ctx.conf.session_zone)),
                    },
                    e.origin,
                ));
            }
            ExprNode::Alias { name, .. } => {
                let output = self.attr(name, &kids[0].ty);
                Bound {
                    expr: rebuilt(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: Arc::clone(&args[0]),
                        output: output.clone(),
                    })),
                    ..keep(column(&output))
                }
            }
            ExprNode::Subquery { kind, plan } => {
                let env = self.sub_env(cx);
                // A subquery resolves without the enclosing lambda scopes:
                // its columns must not bind to enclosing lambda variables.
                let saved = std::mem::take(&mut self.lambdas);
                let rel = self.rel(plan, &env);
                let stale = std::mem::replace(&mut self.lambdas, saved);
                debug_assert!(stale.is_empty(), "a subquery leaks lambda scopes");
                let rel = rel?;
                let (data_type, nullable) = match kind {
                    SubqueryKind::Scalar => {
                        if rel.out.len() != 1 {
                            return Err(AnalysisError::new(
                                InvalidSubqueryExpressionScalarSubqueryReturnMoreThanOneOutputColumn {
                                    number: rel.out.len().to_string(),
                                },
                                e.origin,
                            ));
                        }
                        (
                            arg_type(rel.out[0].col.data_type.clone(), true).data_type,
                            true,
                        )
                    }
                    SubqueryKind::Exists => (Some(DataType::Boolean), false),
                    SubqueryKind::In(_) => (Some(DataType::Boolean), true),
                };
                let (kind, plan, values) = match kind {
                    SubqueryKind::In(_) => {
                        let (values, plan) = self.in_subquery(e, kids, &rel, cx)?;
                        let exprs = values.iter().map(|v| Arc::clone(&v.expr)).collect();
                        (SubqueryKind::In(exprs), plan, values)
                    }
                    k => (k.clone(), Arc::clone(&rel.plan), Vec::new()),
                };
                let level = cx.env.outer.len();
                let expr = rebuilt(ExprNode::Subquery { kind, plan });
                let original = Arc::clone(&expr);
                let shown = shown_call(&expr, kids, &values);
                let expr = self.null_in_subquery(expr);
                let expr = self.in_global_aggregate(expr);
                // A rewrite must not rename the output column: the written `IN`
                // stays the shown form when no coercion gave one.
                let shown = shown.or_else(|| (!Arc::ptr_eq(&original, &expr)).then_some(original));
                Bound {
                    shown,
                    expr,
                    outer: [outer, rel.outer.filter(|&l| l < level)]
                        .into_iter()
                        .flatten()
                        .min(),
                    ..keep(ArgType {
                        data_type,
                        nullable,
                    })
                }
            }
            ExprNode::Window(w) => self.window(e, w, kids, cx)?,
            ExprNode::Resolved(r) => keep({
                use ResolvedExpr::{Column, LambdaVar, OuterColumn};
                match r {
                    Column(c) | OuterColumn(c) | LambdaVar(c) => column(c),
                    ResolvedExpr::Alias { output, .. } => column(output),
                    ResolvedExpr::Function(f) => arg_type(f.data_type.clone(), f.nullable),
                    ResolvedExpr::StateInput { raw_type, .. } => arg_type(raw_type.clone(), true),
                    ResolvedExpr::Field {
                        data_type,
                        nullable,
                        ..
                    } => arg_type(data_type.clone(), *nullable),
                }
            }),
        };
        if b.shown.is_none() && kids.iter().any(|k| k.shown.is_some()) {
            b.shown = Some(with_children(
                &b.expr,
                kids.iter().map(Bound::shown).collect(),
            ));
        }
        Ok(b)
    }
}

impl Resolver<'_> {
    /// `v IN (correlated global aggregate)`: the aggregate has exactly one
    /// row for each outer row (a count of nothing is 0), so the `IN` is
    /// `v = (scalar subquery)`, which the scalar path decorrelates with
    /// its count-bug handling. Any other `IN` is returned as it came.
    fn in_global_aggregate(&self, e: Arc<Expr>) -> Arc<Expr> {
        let ExprNode::Subquery {
            kind: SubqueryKind::In(values),
            plan,
        } = &e.node
        else {
            return e;
        };
        let [value] = &values[..] else { return e };
        // Spark's legacy rule (`subquery_legacy.rs`) answers it otherwise.
        if self.ctx.conf.legacy_exists_in_count
            || !matches!(plan.node, crate::PlanNode::Aggregate { .. })
            || !crate::analyzer::subquery::one_row_aggregate(plan)
        {
            return e;
        }
        let scalar = Arc::new(
            Expr::new(ExprNode::Subquery {
                kind: SubqueryKind::Scalar,
                plan: Arc::clone(plan),
            })
            .with_origin(e.origin),
        );
        let nullable = crate::analyzer::subquery::typed(value).1
            || crate::analyzer::subquery::typed(&scalar).1;
        Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    id: FunctionId::Builtin(Builtin::Eq),
                    kind: FunctionKind::Scalar,
                    args: vec![Arc::clone(value), scalar],
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
                    data_type: DataType::Boolean,
                    nullable,
                    deterministic: true,
                },
            )))
            .with_origin(e.origin),
        )
    }

    /// `NULL IN (subquery)`: Spark folds it by whether the subquery has
    /// rows. It is `exists(subquery) AND NULL`: NULL over rows, false over
    /// none (NULL either way under
    /// `spark.sql.legacy.nullInEmptyListBehavior`). Any other `IN` is
    /// returned as it came.
    fn null_in_subquery(&self, e: Arc<Expr>) -> Arc<Expr> {
        let ExprNode::Subquery {
            kind: SubqueryKind::In(values),
            plan,
        } = &e.node
        else {
            return e;
        };
        let [value] = &values[..] else { return e };
        if !crate::analyzer::subquery::null_literal(value) {
            return e;
        }
        let null = Arc::new(Expr::new(ExprNode::Literal(Literal::Null(Some(
            DataType::Boolean,
        )))));
        if self.ctx.conf.legacy_null_in_empty_list {
            return null;
        }
        let exists = Arc::new(
            Expr::new(ExprNode::Subquery {
                kind: SubqueryKind::Exists,
                plan: Arc::clone(plan),
            })
            .with_origin(e.origin),
        );
        Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    id: FunctionId::Builtin(Builtin::And),
                    kind: FunctionKind::Scalar,
                    args: vec![exists, null],
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
                    data_type: DataType::Boolean,
                    nullable: true,
                    deterministic: true,
                },
            )))
            .with_origin(e.origin),
        )
    }
}

/// Whether `t` is or holds a map at any depth (Spark does not order it).
fn holds_map(t: &DataType) -> bool {
    match t {
        DataType::Map(_) => true,
        DataType::Array(a) => holds_map(a.element_type()),
        DataType::Struct(s) => s.fields().iter().any(|f| holds_map(f.data_type())),
        _ => false,
    }
}

/// `b` typed nullable (the expression's own flag too).
fn make_nullable(mut b: Bound) -> Bound {
    b.ty.nullable = true;
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &b.expr.node {
        let mut f = f.clone();
        f.nullable = true;
        b.expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))).with_origin(b.expr.origin),
        );
    }
    b
}
