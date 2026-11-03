// SPDX-License-Identifier: Apache-2.0
//! Pattern functions: `LIKE`/`ILIKE` and their quantified forms, `mask` and `decode`.

use super::*;

/// The datetime functions whose pattern the legacy time parser reads: the
/// formatters, the parsers with a format argument and `from_unixtime`.
/// `unix_timestamp(str)` and `to_unix_timestamp(str)` parse their string with
/// the default pattern, so a string argument alone counts too (a timestamp or
/// date argument has no text to parse).
pub(super) fn is_pattern_function(name: &str, args: &[Bound]) -> bool {
    match name.to_ascii_lowercase().as_str() {
        "date_format" | "from_unixtime" => true,
        "unix_timestamp" | "to_unix_timestamp" => {
            args.len() >= 2
                || args
                    .first()
                    .is_some_and(|a| a.ty.data_type == Some(DataType::String))
        }
        "to_date" | "try_to_date" | "to_timestamp" | "try_to_timestamp" | "to_timestamp_ntz"
        | "to_timestamp_ltz" => args.len() >= 2,
        _ => false,
    }
}

impl Resolver<'_> {
    /// `ilike(a, b[, escape])` as Spark's `ILike` replacement:
    /// `lower(a) LIKE lower(b)` with the same escape, shown as written.
    pub(super) fn ilike_call(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
    ) -> Result<Bound, AnalysisError> {
        let n = kids.len();
        if n != 2 && n != 3 {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: "`ilike`".to_owned(),
                    expected_num: "[2, 3]".to_owned(),
                    actual_num: n.to_string(),
                    docroot: SPARK_DOC_ROOT.to_owned(),
                },
                e.origin,
            ));
        }
        let FunctionLookup::Found(lower) = self
            .ctx
            .functions
            .lookup(&["lower".to_owned()], self.ctx.profile)
        else {
            return Err(unsupported("function lower", e.origin));
        };
        let origin = e.origin;
        let mut like_kids = vec![
            self.lowered(&lower, e, &kids[0])?,
            self.lowered(&lower, e, &kids[1])?,
        ];
        if n == 3 {
            like_kids.push(kids[2].clone());
        }
        let like = self.operator_call(Builtin::Like, origin, &like_kids)?;
        // Shown without the escape, as Spark shows `ILike`.
        let shown = Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Builtin(Builtin::ILike),
                kind: FunctionKind::Scalar,
                args: kids[..2].iter().map(Bound::shown).collect(),
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::Boolean,
                nullable: like.ty.nullable,
                deterministic: true,
            },
        )));
        Ok(Bound {
            shown: Some(Arc::new(shown.with_origin(origin))),
            ..like
        })
    }

    /// `x [NOT] LIKE|ILIKE ANY|SOME|ALL (patterns)` as an `OR` (`ANY`/`SOME`)
    /// or `AND` (`ALL`) chain of single `LIKE`s (the `NOT` inside, as Spark
    /// negates each pattern match), with the case-insensitive variants built
    /// from the `ILIKE` rewrite. Shown in the quantified form.
    pub(super) fn like_quantified_call(
        &mut self,
        which: Builtin,
        e: &Arc<Expr>,
        kids: &[Bound],
    ) -> Result<Bound, AnalysisError> {
        let Some(QuantifiedLike {
            insensitive,
            negated,
            all,
        }) = which.quantified_like()
        else {
            return Err(unsupported("a LIKE ANY, SOME or ALL form", e.origin));
        };
        let origin = e.origin;
        let Some((written, patterns)) = kids.split_first() else {
            return Err(unsupported("a LIKE ANY, SOME or ALL form", origin));
        };
        let child = written;
        let lower = if insensitive {
            let FunctionLookup::Found(entry) = self
                .ctx
                .functions
                .lookup(&["lower".to_owned()], self.ctx.profile)
            else {
                return Err(unsupported("function lower", origin));
            };
            Some(entry)
        } else {
            None
        };
        let wrap = |b: &Bound| match &lower {
            None => Ok(b.clone()),
            Some(entry) => self.lowered(entry, e, b),
        };
        let child = wrap(child)?;
        let mut lowered_patterns = Vec::with_capacity(patterns.len());
        for p in patterns {
            lowered_patterns.push(wrap(p)?);
        }
        let mut acc: Option<Bound> = None;
        for p in lowered_patterns {
            let like = self.operator_call(Builtin::Like, origin, &[child.clone(), p])?;
            let next = if negated {
                self.operator_call(Builtin::Not, origin, std::slice::from_ref(&like))?
            } else {
                like
            };
            acc = Some(match acc {
                None => next,
                Some(done) => self.operator_call(
                    if all { Builtin::And } else { Builtin::Or },
                    origin,
                    &[done, next],
                )?,
            });
        }
        let Some(acc) = acc else {
            return Err(unsupported("a LIKE ANY, SOME or ALL form", origin));
        };
        // Spark's name: with only non-NULL string literal patterns it builds
        // `LikeAny`/`LikeAll` (shown `likeany(child)`, the child lowered for
        // ILIKE); otherwise the chain of single `LIKE`s or `ilike` calls.
        let nullable = acc.ty.nullable;
        let shown_call = |op: Builtin, args: Vec<Arc<Expr>>| {
            Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                    ResolvedFunction {
                        id: FunctionId::Builtin(op),
                        kind: FunctionKind::Scalar,
                        args,
                        distinct: false,
                        filter: None,
                        within_group: Vec::new(),
                        data_type: DataType::Boolean,
                        nullable,
                        deterministic: true,
                    },
                )))
                .with_origin(origin),
            )
        };
        let ansi = self.ctx.conf.ansi;
        let literal_patterns = patterns
            .iter()
            .all(|p| matches!(self.fold(&p.expr, ansi), Some(Constant::String(_))));
        let shown = if literal_patterns {
            shown_call(which, vec![child.shown()])
        } else {
            let single = if insensitive {
                Builtin::ILike
            } else {
                Builtin::Like
            };
            patterns
                .iter()
                .map(|p| {
                    let one = shown_call(single, vec![written.shown(), p.shown()]);
                    if negated {
                        shown_call(Builtin::Not, vec![one])
                    } else {
                        one
                    }
                })
                .reduce(|done, next| {
                    shown_call(
                        if all { Builtin::And } else { Builtin::Or },
                        vec![done, next],
                    )
                })
                .unwrap_or_else(|| shown_call(which, vec![child.shown()]))
        };
        Ok(Bound {
            shown: Some(shown),
            ..acc
        })
    }

    /// Spark's checks of the replacement characters of a `mask` call: each
    /// must fold to a single character (or `NULL`), quoted and cited at the
    /// argument. `kids` are the call's bound arguments before their implicit
    /// casts, `call_kids` after.
    pub(super) fn check_mask_replacements(
        &self,
        lookup_name: &str,
        kids: &[Bound],
        call_kids: &[Bound],
    ) -> Result<(), AnalysisError> {
        const NAMES: [&str; 4] = ["upperChar", "lowerChar", "digitChar", "otherChar"];
        let zone = self.ctx.conf.session_zone;
        let ansi = self.ctx.conf.ansi;
        // The call as Spark quotes it: the rearranged positional form.
        let sql_expr = || {
            format!(
                "\"{}({})\"",
                lookup_name.to_lowercase(),
                call_kids
                    .iter()
                    .map(|k| pretty(&k.shown(), zone))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        for (i, name) in NAMES.iter().enumerate() {
            let (Some(arg), Some(written)) = (call_kids.get(1 + i), kids.get(1 + i)) else {
                return Err(unsupported(
                    "a mask call without its replacements",
                    call_kids
                        .first()
                        .map_or_else(Origin::default, |k| k.expr.origin),
                ));
            };
            match self.fold(&arg.expr, ansi) {
                Some(Constant::Null) => {}
                Some(Constant::String(s)) if s.chars().count() == 1 => {}
                Some(Constant::String(_)) => {
                    return Err(AnalysisError::new(
                        DatatypeMismatchInputSizeNotOne {
                            sql_expr: sql_expr(),
                            expr_name: (*name).to_owned(),
                        },
                        arg.expr.origin,
                    ));
                }
                Some(Constant::Error(err)) => {
                    return Err(AnalysisError::new(err, arg.expr.origin));
                }
                _ => {
                    return Err(AnalysisError::new(
                        DatatypeMismatchNonFoldableInput {
                            sql_expr: sql_expr(),
                            input_name: format!("`{name}`"),
                            input_type: "\"STRING\"".to_owned(),
                            input_expr: format!("\"{}\"", pretty(&written.shown(), zone)),
                        },
                        arg.expr.origin,
                    ));
                }
            }
        }
        Ok(())
    }

    /// `decode(expr, search1, result1, ..., [default])` (Oracle style): a
    /// `CASE` over null-safe equality, so a NULL `expr` matches a NULL
    /// search. A trailing unpaired argument is the default; without one the
    /// result is NULL. The column is named by the call as written.
    pub(super) fn decode_case(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
    ) -> Result<Bound, AnalysisError> {
        let origin = e.origin;
        let mut parts = Vec::with_capacity(kids.len());
        for pair in kids[1..].chunks(2) {
            if let [search, result] = pair {
                let cond = self.operator_call(
                    Builtin::EqNullSafe,
                    origin,
                    &[kids[0].clone(), search.clone()],
                )?;
                parts.push(cond);
                parts.push(result.clone());
            } else {
                parts.extend(pair.iter().cloned());
            }
        }
        // Without a default Spark ends the CASE with a string-typed NULL,
        // which takes part in the result type (`typeof(decode(1, 1, 2))` is
        // `string` outside ANSI mode).
        if kids.len() % 2 == 1 {
            let null = Arc::new(
                Expr::new(ExprNode::Literal(Literal::Null(Some(DataType::String))))
                    .with_origin(origin),
            );
            parts.push(Bound {
                expr: null,
                ty: arg_type(DataType::String, true),
                agg: None,
                win: None,
                outer: None,
                lateral: Vec::new(),
                shown: None,
            });
        }
        let mut b = self.operator_call(Builtin::When, origin, &parts)?;
        b.carry_markers(kids);
        b.shown = Some(with_children(e, kids.iter().map(Bound::shown).collect()));
        Ok(b)
    }
}
