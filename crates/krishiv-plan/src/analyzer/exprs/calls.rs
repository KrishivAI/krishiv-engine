// SPDX-License-Identifier: Apache-2.0
//! Function calls: named arguments, typed and folded calls, `schema_of_csv`, `ntile` and the legacy day count.

use super::*;

/// The default message of `assert_true(cond)`: `'<cond>' is not true!`,
/// with casts stripped as Spark's `toPrettySQL` strips them.
pub(super) fn assert_message(cond: &Bound, origin: Origin, zone: Option<SessionZone>) -> Bound {
    let mut shown = cond.shown();
    while let ExprNode::Cast { expr, .. } = &shown.node {
        shown = Arc::clone(expr);
    }
    let message = format!("'{}' is not true!", pretty(&shown, zone));
    Bound {
        expr: Arc::new(Expr::new(ExprNode::Literal(Literal::String(message))).with_origin(origin)),
        ty: arg_type(DataType::String, false),
        agg: None,
        win: None,
        outer: None,
        lateral: Vec::new(),
        shown: None,
    }
}

/// The position of the options argument of the CSV and JSON functions.
pub(super) fn options_index(name: &str) -> Option<usize> {
    match name {
        "from_csv" | "from_json" | "from_xml" => Some(2),
        "to_csv" | "schema_of_csv" | "to_json" | "schema_of_json" | "to_xml" | "schema_of_xml" => {
            Some(1)
        }
        _ => None,
    }
}

/// The pairs of a `map('k', 'v', ...)` call whose arguments are all `STRING`
/// literals, if `e` is one (a call with any other argument keeps its call).
pub(super) fn literal_map_pairs(e: &Expr) -> Option<Vec<(String, String)>> {
    let ExprNode::Resolved(ResolvedExpr::Function(call)) = &e.node else {
        return None;
    };
    if call.id.name() != "map" || call.args.len() % 2 != 0 || call.filter.is_some() {
        return None;
    }
    let text = |x: &Arc<Expr>| match &x.node {
        ExprNode::Literal(Literal::String(s)) => Some(s.clone()),
        _ => None,
    };
    call.args
        .chunks(2)
        .map(|kv| Some((text(&kv[0])?, text(&kv[1])?)))
        .collect()
}

impl Resolver<'_> {
    /// A registry call's arguments after their implicit casts, and its
    /// result type: Spark's arity check, then the function's input types
    /// and result rule. Out of line: it keeps the recursive `expr` frame
    /// small (deep plans resolve on a small stack).
    #[inline(never)]
    pub(super) fn typed_call(
        &self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        kids: &[Bound],
    ) -> Result<(Vec<Bound>, ArgType), AnalysisError> {
        let n = kids.len();
        if !entry.arity.accepts(n) {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: format!("`{}`", entry.id.name()),
                    expected_num: entry.arity.expected().to_owned(),
                    actual_num: n.to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                e.origin,
            ));
        }
        let constants: Vec<Option<Constant>> = kids
            .iter()
            .map(|k| self.fold(&k.expr, self.ctx.conf.ansi))
            .collect();
        let literals: Vec<bool> = kids.iter().map(|k| fold::is_literal(&k.expr)).collect();
        // Rendered only if a rule's error quotes an argument.
        let zone = self.ctx.conf.session_zone;
        let sql = |i: usize| kids.get(i).map(|k| pretty(&k.shown(), zone));
        let given: Vec<ArgType> = kids.iter().map(|k| k.ty.clone()).collect();
        let ansi = self.ctx.conf.ansi;
        let loss = self.ctx.conf.decimal_precision_loss;
        let call = CallTypes::new(&given, &constants)
            .with_literals(&literals)
            .with_sql(&sql)
            .with_ansi(ansi)
            .with_decimal_loss(loss)
            .with_case_sensitive(self.ctx.conf.case_sensitive);
        let coerced: Vec<Arg<'_>> = kids.iter().map(arg_of).collect();
        let casts = coerce::inputs(&entry.inputs, &coerced, &call)
            .map_err(|m| self.mismatch(m, e, kids, entry.id.name(), e.origin))?;
        // `try_add('x', 1)` casts its arguments in TRY mode: NULL, not an error.
        let try_casts = matches!(entry.inputs, Inputs::TryOperator(_));
        let call_kids: Vec<Bound> = kids
            .iter()
            .zip(casts)
            .map(|(k, to)| self.implicit_cast_as(k, to, try_casts))
            .collect();
        let types: Vec<ArgType> = call_kids.iter().map(|k| k.ty.clone()).collect();
        let bound_sql = |i: usize| call_kids.get(i).map(|k| pretty(&k.expr, zone));
        let call = CallTypes::new(&types, &constants)
            .with_literals(&literals)
            .with_sql(&sql)
            .with_bound_sql(&bound_sql)
            .with_ansi(ansi)
            .with_decimal_loss(loss)
            .with_case_sensitive(self.ctx.conf.case_sensitive);
        let ty = (entry.result)(&call).map_err(|err| {
            // An error folding an argument is Spark's, cited at the argument.
            let origin = constants
                .iter()
                .zip(kids)
                .find_map(|(c, k)| match c {
                    Some(Constant::Error(folded)) if *folded == err => Some(k.expr.origin),
                    _ => None,
                })
                .unwrap_or(e.origin);
            AnalysisError::new(err, origin)
        })?;
        Ok((Self::folded_ntile(entry, &constants, call_kids), ty))
    }

    /// `count()` with no arguments: `WRONG_NUM_ARGS.WITH_SUGGESTION`,
    /// or `count(NULL)` — which counts no rows — when
    /// `spark.sql.legacy.allowParameterlessCount`. The output still shows
    /// as `count()`, as Spark names it.
    pub(super) fn parameterless_count(
        &self,
        e: &Arc<Expr>,
        entry: &FunctionEntry,
        kids: &[Bound],
        agg: Option<Origin>,
        window_fn: bool,
    ) -> Result<Bound, AnalysisError> {
        if !self.ctx.conf.allow_parameterless_count {
            return Err(AnalysisError::new(
                WrongNumArgsWithSuggestion {
                    function_name: format!("`{}`", entry.id.name()),
                    expected_num: entry.arity.expected().to_owned(),
                    actual_num: "0".to_owned(),
                    legacy_num: "0".to_owned(),
                    legacy_conf_key: "\"spark.sql.legacy.allowParameterlessCount\"".to_owned(),
                    legacy_conf_value: "\"true\"".to_owned(),
                },
                e.origin,
            ));
        }
        let null = Bound {
            expr: Arc::new(Expr::new(ExprNode::Literal(Literal::Null(None))).with_origin(e.origin)),
            ty: ArgType {
                data_type: Some(DataType::Null),
                nullable: true,
            },
            agg: None,
            win: None,
            outer: None,
            lateral: Vec::new(),
            shown: None,
        };
        let (mut call_kids, ty) = self.typed_call(entry, e, std::slice::from_ref(&null))?;
        let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
        // `kids.first()` is the `FILTER`, if any (`n == 0`).
        call_kids.extend(kids.first().cloned());
        let call = ResolvedFunction {
            within_group: Vec::new(),
            id: entry.id.clone(),
            kind: entry.kind,
            args: call_kids
                .first()
                .map(|k| vec![Arc::clone(&k.expr)])
                .unwrap_or_default(),
            distinct: match &e.node {
                ExprNode::Function(f) => f.distinct,
                _ => false,
            },
            filter: call_kids.get(1).map(|k| Arc::clone(&k.expr)),
            data_type,
            nullable: ty.nullable,
            deterministic: entry.deterministic,
        };
        // Shown as written: no argument, the `FILTER` after the call.
        let shown = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    args: Vec::new(),
                    ..call.clone()
                },
            )))
            .with_origin(e.origin),
        );
        let expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(call))).with_origin(e.origin),
        );
        // The `FILTER`'s markers ride along, as for any other call.
        let mut lateral: Vec<AttrId> = Vec::new();
        for id in kids.iter().flat_map(|k| k.lateral.iter()) {
            if !lateral.contains(id) {
                lateral.push(*id);
            }
        }
        Ok(Bound {
            expr,
            ty,
            agg: if entry.kind == FunctionKind::Aggregate && !window_fn {
                Some(e.origin)
            } else {
                agg
            },
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral,
            // Shown as written (`count()`), not as bound (`count(NULL)`).
            shown: Some(shown),
        })
    }

    /// `ntile`'s bucket count as the literal it folds to (`ntile(1 + 1)`):
    /// the window operator reads a literal. The rule above checked it is an
    /// `int` constant. Only for the built-in: a session UDF named `ntile`
    /// keeps its arguments as written.
    #[inline(never)]
    pub(super) fn folded_ntile(
        entry: &FunctionEntry,
        constants: &[Option<Constant>],
        mut kids: Vec<Bound>,
    ) -> Vec<Bound> {
        if is_builtin_entry(entry)
            && entry.id.name() == "ntile"
            && let (Some(Some(Constant::Integral(n))), Some(k)) =
                (constants.first(), kids.first_mut())
            && let Ok(n) = i32::try_from(*n)
        {
            k.expr = Arc::new(
                Expr::new(ExprNode::Literal(Literal::Integer(n))).with_origin(k.expr.origin),
            );
        }
        kids
    }

    /// `schema_of_csv(csv[, options])` as the DDL literal it resolves to (Spark's
    /// `SchemaOfCsv` evaluates its foldable child during analysis, keeping
    /// the call as the output name). The signature's result rule already
    /// rejected a non-foldable, null or non-string child, so anything but
    /// a folded string here (a custom registry's rule) keeps the call;
    /// that call has no scalar binding and fails at execution binding, so
    /// it never resolves silently. The options argument resolves when it is
    /// a map literal (the `map('k', 'v')` rewrite above); any other map
    /// keeps the call. Out of line: it is on the path of every call.
    #[inline(never)]
    pub(super) fn resolve_schema_of_csv(
        &self,
        e: &Arc<Expr>,
        kids: &[Bound],
        bound: &mut Bound,
        xml: bool,
        json: bool,
    ) -> Result<(), AnalysisError> {
        if kids.is_empty() || kids.len() > 2 {
            return Ok(());
        }
        let csv = kids
            .first()
            .and_then(|k| self.fold(&k.expr, self.ctx.conf.ansi));
        let Some(Constant::String(csv)) = csv else {
            return Ok(());
        };
        let ddl = if json {
            // `schema_of_json`: only a `map(...)` literal of options folds;
            // anything else keeps the call.
            let pairs = match kids.get(1) {
                None => Vec::new(),
                Some(opts) => match &opts.expr.node {
                    ExprNode::Literal(Literal::Map(pairs)) => pairs.clone(),
                    _ => return Ok(()),
                },
            };
            match krishiv_functions::scalar::schema_of_json_ddl(&csv, &pairs)
                .map_err(|err| AnalysisError::new(err, e.origin))?
            {
                Some(ddl) => ddl,
                None => return Ok(()),
            }
        } else if xml {
            // `schema_of_xml`: the document under the written options.
            let pairs = match kids.get(1) {
                None => Vec::new(),
                Some(opts) => match &opts.expr.node {
                    ExprNode::Literal(Literal::Map(pairs)) => pairs.clone(),
                    // Spark reads the options only from a `map(...)` call.
                    _ => {
                        return Err(AnalysisError::new(
                            krishiv_common::error::InvalidOptionsNonMapFunction {},
                            e.origin,
                        ));
                    }
                },
            };
            krishiv_functions::scalar::schema_of_xml_ddl(&csv, &pairs)
                .map_err(|err| AnalysisError::new(err, e.origin))?
        } else {
            // Options other than a map literal keep the call (the signature
            // accepts them; the executor has no binding for it).
            if kids
                .get(1)
                .is_some_and(|o| !matches!(o.expr.node, ExprNode::Literal(Literal::Map(_))))
            {
                return Ok(());
            }
            self.csv_ddl(e, &csv, kids.get(1))?
        };
        self.show_schema_literal(e, kids.len(), bound, ddl);
        Ok(())
    }

    /// The DDL `schema_of_csv(csv[, options])` resolves to (`None`: defaults).
    pub(super) fn csv_ddl(
        &self,
        e: &Arc<Expr>,
        csv: &str,
        options: Option<&Bound>,
    ) -> Result<String, AnalysisError> {
        let options = match options {
            None => krishiv_io::FormatOptions::defaults(krishiv_io::csv::option_schema()),
            Some(opts) => {
                // Options other than a literal map keep the call (the
                // signature accepts them; the executor has no binding).
                let ExprNode::Literal(Literal::Map(pairs)) = &opts.expr.node else {
                    return Err(AnalysisError::new(
                        InternalError {
                            message: "schema_of_csv options are not a map literal".to_owned(),
                        },
                        e.origin,
                    ));
                };
                // Option names are case-insensitive: of two spellings of one
                // name, the one written last wins (Spark's `CaseInsensitiveMap`).
                let mut user: std::collections::BTreeMap<
                    String,
                    krishiv_common::secret::OptionValue,
                > = std::collections::BTreeMap::new();
                for (k, v) in pairs {
                    user.retain(|seen, _| !seen.eq_ignore_ascii_case(k));
                    user.insert(
                        k.clone(),
                        krishiv_common::secret::OptionValue::from_user(k, v.clone()),
                    );
                }
                krishiv_io::csv::option_schema()
                    .parse(&user, &krishiv_common::config::Config::new())
                    .map_err(|err| AnalysisError::new(err, e.origin))?
                    .0
            }
        };
        krishiv_io::csv::ddl_of_string(csv, &options).map_err(|err| {
            AnalysisError::new(
                match err {
                    krishiv_io::FormatError::Other(e) => e,
                    other => InternalError {
                        message: other.to_string(),
                    }
                    .into(),
                },
                e.origin,
            )
        })
    }

    /// Replaces the resolved call by its DDL literal, shown as the call
    /// without its options.
    pub(super) fn show_schema_literal(
        &self,
        e: &Arc<Expr>,
        args: usize,
        bound: &mut Bound,
        ddl: String,
    ) {
        // Keep the form the call already shows (a session variable's name,
        // not its value); otherwise show the call itself.
        if bound.shown.is_none() {
            bound.shown = Some(Arc::clone(&bound.expr));
        }
        // Spark's output name drops the options (`schema_of_csv(1|abc)`).
        if args == 2
            && let Some(shown) = &bound.shown
            && let ExprNode::Resolved(ResolvedExpr::Function(call)) = &shown.node
        {
            let mut one = call.clone();
            one.args.truncate(1);
            bound.shown = Some(Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Function(one)))
                    .with_origin(shown.origin),
            ));
        }
        bound.expr =
            Arc::new(Expr::new(ExprNode::Literal(Literal::String(ddl))).with_origin(e.origin));
    }

    /// The shown form of the call `expr` (see [`shown_call`]); `date_add(d,
    /// ' 12 ')` and `date_add(d, '1' || '2')` without ANSI show the day count
    /// as an integer (`date_add(d, 12)`): Spark folds the string it casts to
    /// `INT`. Out of line: it is on the path of every call.
    #[inline(never)]
    pub(super) fn call_shown(
        &self,
        name: &[String],
        expr: &Expr,
        kids: &[Bound],
        cast: &[Bound],
    ) -> Option<Arc<Expr>> {
        let days = self.legacy_day_count(name, kids);
        shown_call(expr, days.as_deref().unwrap_or(kids), cast)
    }

    /// The value of `e` as a constant (`fold::constant`) under this
    /// session's settings, with `ansi` for the arithmetic: shared with
    /// every other fold of the pass.
    pub(in crate::analyzer) fn fold(&self, e: &Arc<Expr>, ansi: bool) -> Option<Constant> {
        let conf = fold::Conf::new(
            ansi,
            self.ctx.conf.decimal_precision_loss,
            self.ctx.conf.map_key_dedup_last_win,
            (
                self.ctx.conf.legacy_java_charsets,
                self.ctx.conf.legacy_coding_error_action,
            ),
            self.ctx.profile,
            self.ctx.conf.session_zone.and_then(SessionZone::fixed),
        );
        fold::constant(
            e,
            conf,
            &mut self.folds.borrow_mut(),
            self.ctx.functions.as_ref(),
        )
    }

    pub(super) fn legacy_day_count(&self, name: &[String], kids: &[Bound]) -> Option<Vec<Bound>> {
        let last = name.last().map(String::as_str).unwrap_or("");
        if self.ctx.conf.ansi
            || !["date_add", "dateadd", "date_sub"]
                .iter()
                .any(|n| last.eq_ignore_ascii_case(n))
        {
            return None;
        }
        let kid = kids.get(1)?;
        if kid.ty.data_type != Some(DataType::String) {
            return None;
        }
        let Some(Constant::String(text)) = self.fold(&kid.expr, false) else {
            return None;
        };
        let days = krishiv_functions::registry::legacy_day_count(&text)?;
        let mut shown = kids.to_vec();
        shown[1].shown = Some(Arc::new(Expr::new(ExprNode::Literal(Literal::String(
            days.to_string(),
        )))));
        Some(shown)
    }

    /// Rearranges a call of a function with a named-parameter signature
    /// into positional order (Spark's `FunctionBuilderBase`
    /// rearrangement): positional arguments first, then named arguments in
    /// signature order with defaults filled in. Short positional calls are
    /// expanded with defaults too.
    ///
    /// `parts` names the routine in errors, as written (qualified when the
    /// call was). `kids` holds the bound arguments (no filter).
    pub(super) fn rearrange_named(
        &self,
        parts: &[String],
        signature: &[NamedParam],
        f: &FunctionCall,
        kids: &[Bound],
        origin: Origin,
    ) -> Result<Vec<Bound>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let routine = || quoted(parts);
        let param = |name: &str| quoted(std::slice::from_ref(&name.to_owned()));
        // Spark's split: positional arguments first, then named ones; a
        // positional argument after a named one, and a twice-named
        // parameter, are errors.
        let mut seen: Vec<&str> = Vec::new();
        if let Some(first) = f.arg_names.iter().position(Option::is_some) {
            for (i, name) in f.arg_names.iter().enumerate().skip(first) {
                match name {
                    None => {
                        let prev = i
                            .checked_sub(1)
                            .and_then(|j| f.arg_names.get(j))
                            .and_then(Option::as_deref)
                            .unwrap_or_default();
                        return Err(AnalysisError::new(
                            UnexpectedPositionalArgument {
                                routine_name: routine(),
                                parameter_name: param(prev),
                            },
                            origin,
                        ));
                    }
                    Some(name) => {
                        if seen.iter().any(|s| same(s, name, cs)) {
                            return Err(AnalysisError::new(
                                DuplicateRoutineParameterAssignmentDoubleNamedArgumentReference {
                                    routine_name: routine(),
                                    parameter_name: param(name),
                                },
                                origin,
                            ));
                        }
                        seen.push(name);
                    }
                }
            }
        }
        let positional = f
            .arg_names
            .iter()
            .position(Option::is_some)
            .unwrap_or(kids.len());
        // Unrecognized names, and names that are both positional (by their
        // position) and named, are errors.
        for name in &seen {
            if !signature.iter().any(|p| same(p.name, name, cs)) {
                return Err(AnalysisError::new(
                    UnrecognizedParameterName {
                        routine_name: routine(),
                        argument_name: param(name),
                        proposal: suggestions(name, signature.iter().map(|p| p.name)),
                    },
                    origin,
                ));
            }
            if signature[..positional.min(signature.len())]
                .iter()
                .any(|p| same(p.name, name, cs))
            {
                return Err(AnalysisError::new(
                    DuplicateRoutineParameterAssignmentBothPositionalAndNamed {
                        routine_name: routine(),
                        parameter_name: param(name),
                    },
                    origin,
                ));
            }
        }
        // Spark's arity check over the valid sizes.
        let required = signature
            .iter()
            .take_while(|p| matches!(p.default, NamedDefault::Required))
            .count();
        if signature.len() < kids.len() {
            let valid: Vec<String> = (required..=signature.len())
                .map(|v| v.to_string())
                .collect();
            let expected = match valid.as_slice() {
                [only] => only.clone(),
                _ => format!("[{}]", valid.join(", ")),
            };
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: routine(),
                    expected_num: expected,
                    actual_num: kids.len().to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                origin,
            ));
        }
        // Positional arguments keep their order; the rest follows the
        // signature, filling defaults.
        let mut reordered: Vec<Bound> = kids[..positional.min(kids.len())].to_vec();
        for (i, p) in signature.iter().enumerate().skip(positional) {
            if let Some((_, kid)) =
                f.arg_names
                    .iter()
                    .zip(kids)
                    .find_map(|(name, kid)| match name {
                        Some(name) if same(name, p.name, cs) => Some((name, kid)),
                        _ => None,
                    })
            {
                reordered.push(kid.clone());
            } else if matches!(p.default, NamedDefault::Required) {
                return Err(AnalysisError::new(
                    RequiredParameterNotFound {
                        routine_name: routine(),
                        parameter_name: param(p.name),
                        index: (i - positional).to_string(),
                    },
                    origin,
                ));
            } else {
                let literal = match p.default {
                    NamedDefault::Null => Literal::Null(None),
                    NamedDefault::Text(s) => Literal::String(s.to_owned()),
                    NamedDefault::Required => {
                        return Err(unsupported("a required parameter default", origin));
                    }
                };
                let data_type = literal_type(&literal);
                let nullable = matches!(literal, Literal::Null(_));
                reordered.push(Bound {
                    expr: Arc::new(Expr::new(ExprNode::Literal(literal)).with_origin(origin)),
                    ty: ArgType {
                        data_type,
                        nullable,
                    },
                    agg: None,
                    win: None,
                    outer: None,
                    lateral: Vec::new(),
                    shown: None,
                });
            }
        }
        Ok(reordered)
    }

    /// `lower(b)` as a call of the registry's `lower` (`entry`), keeping the
    /// argument's aggregate, window and outer-reference bookkeeping.
    pub(super) fn lowered(
        &self,
        entry: &FunctionEntry,
        e: &Arc<Expr>,
        b: &Bound,
    ) -> Result<Bound, AnalysisError> {
        let (mut v, ty) = self.typed_call(entry, e, std::slice::from_ref(b))?;
        if v.is_empty() {
            return Err(unsupported("function lower", e.origin));
        }
        let arg = v.remove(0);
        let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
        Ok(Bound {
            expr: Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                    ResolvedFunction {
                        id: entry.id.clone(),
                        kind: entry.kind,
                        args: vec![Arc::clone(&arg.expr)],
                        distinct: false,
                        filter: None,
                        within_group: Vec::new(),
                        data_type,
                        nullable: ty.nullable,
                        deterministic: entry.deterministic,
                    },
                )))
                .with_origin(e.origin),
            ),
            ty,
            agg: arg.agg,
            win: arg.win,
            outer: arg.outer,
            lateral: arg.lateral,
            shown: None,
        })
    }
}
