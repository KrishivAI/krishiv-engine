// SPDX-License-Identifier: Apache-2.0
//! `expr_node`: the resolution of one expression node from its resolved children.

use super::*;

impl Resolver<'_> {
    pub(super) fn expr_node(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        in_agg: bool,
        window_fn: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
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
                // Spark's `SimpleDateFormat` time parser is not implemented: a
                // pattern function under it would answer with the new parser.
                if self.ctx.conf.legacy_time_parser
                    && !session_qualified
                    && is_pattern_function(lookup_name, &kids[..n])
                {
                    return Err(unsupported(
                        "spark.sql.legacy.timeParserPolicy=LEGACY (the SimpleDateFormat time parser)",
                        e.origin,
                    ));
                }
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
                    // Shown as Spark names it, over the operands as written
                    // (a session variable keeps its name).
                    b.shown = Some(Arc::new(
                        Expr::function(
                            name,
                            vec![kids[first].shown(), kids[second].shown()]
                                .into_iter()
                                .map(|k| Expr::clone(&k))
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
                // `array_agg` is Spark's spelling of `collect_list` (its
                // result column is named `collect_list(...)`): resolve it
                // as `collect_list` throughout.
                if entry.id.name() == "array_agg" {
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
                if entry.id.name() == "string_agg" {
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
                    && !matches!(
                        entry.id.name(),
                        "mode" | "listagg" | "percentile_cont" | "percentile_disc"
                    )
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
                if entry.id.name() == "mode" {
                    return self.mode_call(&entry, e, f, &kids, window_fn);
                }
                if entry.id.name() == "listagg" {
                    return self.listagg_call(&entry, e, f, &kids, window_fn);
                }
                if matches!(
                    entry.id.name(),
                    "percentile"
                        | "percentile_approx"
                        | "approx_percentile"
                        | "median"
                        | "percentile_cont"
                        | "percentile_disc"
                        | "approx_count_distinct"
                ) {
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
                            shown_kids.push(Arc::new(
                                Expr::new(ExprNode::Alias {
                                    expr: k.shown(),
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
                if entry.id.name() == "count" && n == 0 {
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
                    let element = Expr::new(ExprNode::Column(ColumnRef {
                        name: vec!["__subscript_element".to_owned()],
                        plan_id: None,
                    }))
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
                    && matches!(entry.id.name(), "extract" | "date_part")
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
                } else if is_typeof {
                    let name = call_kids
                        .first()
                        .and_then(|k| k.ty.data_type.clone())
                        .map_or_else(|| "void".to_owned(), |t| t.catalog_string());
                    (
                        rebuilt(ExprNode::Literal(Literal::String(name))),
                        Some(expr),
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
                Bound {
                    shown: shown_call(&expr, kids, &values),
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
