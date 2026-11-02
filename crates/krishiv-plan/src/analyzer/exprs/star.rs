// SPDX-License-Identifier: Apache-2.0
//! Star expansion: `*`, `t.*`, `struct.*` and `* EXCEPT (...)`.

use super::*;

/// Whether `qualifier` matches the trailing parts of scope qualifier `qual`,
/// whole identifier by whole identifier (case-folded unless
/// `case_sensitive`). An empty qualifier matches everything; `t` matches
/// `db.t`; `db.t` matches `db.t` and `catalog.db.t` but not `t` alone or
/// `xdb.t`.
pub(super) fn qualifier_matches(
    qual: &[String],
    qualifier: &[String],
    case_sensitive: bool,
) -> bool {
    let q = qualifier.len();
    qual.len() >= q
        && qual[qual.len() - q..]
            .iter()
            .zip(qualifier.iter())
            .all(|(x, y)| same(x, y, case_sensitive))
}

/// One `EXCEPT` entry, resolved against its star's expansion.
#[derive(Debug, Clone)]
pub(super) enum ExceptTarget {
    /// A top-level column to drop.
    Top(Attr),
    /// A field below a top-level column: struct support is needed.
    Nested,
    /// Nothing by that name.
    Missing,
}

/// `name` folded for a case-insensitive comparison when `case_sensitive`
/// is off.
pub(super) fn fold_case(name: &str, case_sensitive: bool) -> String {
    if case_sensitive {
        name.to_owned()
    } else {
        name.to_lowercase()
    }
}

/// Whether folded path `a` is a prefix of (or equal to) folded path `b`.
pub(super) fn is_prefix(a: &[String], b: &[String]) -> bool {
    a.len() <= b.len() && a.iter().zip(b).all(|(x, y)| x == y)
}

impl Resolver<'_> {
    /// A call with `*` arguments, rewritten: `count(*)` is `count(1)`, any
    /// other call takes the input columns. `count(tbl.*)` with a single
    /// qualified star is `INVALID_USAGE_OF_STAR_WITH_TABLE_IDENTIFIER_IN_COUNT`
    /// unless `spark.sql.legacy.allowStarWithSingleTableIdentifierInCount`.
    /// A qualified star of a struct
    /// column in `struct`/`named_struct` (the `STRUCT(t.*)` syntax) takes
    /// the struct's fields instead; in `named_struct` it replaces a
    /// preceding placeholder name from that same syntax.
    pub(super) fn star_args(
        &mut self,
        e: &Expr,
        cx: &Cx<'_>,
    ) -> Result<Option<Arc<Expr>>, AnalysisError> {
        let ExprNode::Function(f) = &e.node else {
            return Ok(None);
        };
        if f.unqualified()
            .is_some_and(|n| n.eq_ignore_ascii_case("count"))
            && let [a] = &f.args[..]
            && let ExprNode::Star(s) = &a.node
            && !s.qualifier.is_empty()
            && s.plan_id.is_none()
            && !self
                .ctx
                .conf
                .allow_star_with_single_table_identifier_in_count
            && self.qualifier_matches_scope(&s.qualifier, cx)
        {
            return Err(AnalysisError::new(
                InvalidUsageOfStarWithTableIdentifierInCount {
                    table_name: quoted(&s.qualifier),
                },
                a.origin,
            ));
        }
        if !f.args.iter().any(|a| matches!(a.node, ExprNode::Star(_))) {
            return Ok(None);
        }
        let count_star = f
            .unqualified()
            .is_some_and(|n| n.eq_ignore_ascii_case("count"))
            && matches!(&f.args[..], [a] if matches!(&a.node, ExprNode::Star(s) if s.qualifier.is_empty() && s.plan_id.is_none()));
        let struct_star =
            f.name.eq_ignore_ascii_case("struct") || f.name.eq_ignore_ascii_case("named_struct");
        let mut args = Vec::new();
        let mut arg_names = Vec::new();
        for (i, a) in f.args.iter().enumerate() {
            let name = f.arg_names.get(i).cloned().flatten();
            match &a.node {
                _ if count_star => {
                    args.push(Arc::new(
                        Expr::literal(Literal::Integer(1)).with_origin(a.origin),
                    ));
                    arg_names.push(name);
                }
                ExprNode::Star(s) if struct_star => {
                    let is_named = f.name.eq_ignore_ascii_case("named_struct");
                    // A `STRUCT(t.*)` placeholder name precedes the star in
                    // a `named_struct` the syntax lowered: the expansion
                    // carries its own names.
                    if is_named
                        && matches!(
                            args.last().map(|a| &a.node),
                            Some(ExprNode::Literal(Literal::String(_)))
                        )
                    {
                        args.pop();
                        arg_names.pop();
                    }
                    for (name, field) in self.struct_star(s, cx, a.origin)? {
                        if is_named {
                            args.push(Arc::new(
                                Expr::new(ExprNode::Literal(Literal::String(name)))
                                    .with_origin(a.origin),
                            ));
                            arg_names.push(None);
                        }
                        args.push(field);
                        arg_names.push(None);
                    }
                }
                ExprNode::Star(s) => {
                    for attr in self.expand(s, cx, a.origin)? {
                        args.push(Arc::new(
                            Expr::new(ExprNode::Resolved(ResolvedExpr::Column(attr.col)))
                                .with_origin(a.origin),
                        ));
                        arg_names.push(None);
                    }
                }
                _ => {
                    args.push(Arc::clone(a));
                    arg_names.push(name);
                }
            }
        }
        // An all-positional call stays all-positional.
        if arg_names.iter().all(Option::is_none) {
            arg_names.clear();
        }
        let call = FunctionCall {
            within_group: Vec::new(),
            args,
            arg_names,
            ..f.clone()
        };
        Ok(Some(Arc::new(
            Expr::new(ExprNode::Function(call)).with_origin(e.origin),
        )))
    }

    /// Whether `qualifier` names the qualifier of an input attribute
    /// (a table or alias the scope knows): it must match the scope
    /// qualifier's trailing parts exactly, whole identifier by whole
    /// identifier (case-folded unless `case_sensitive`).
    pub(super) fn qualifier_matches_scope(&self, qualifier: &[String], cx: &Cx<'_>) -> bool {
        let cs = self.ctx.conf.case_sensitive;
        cx.scope
            .iter()
            .any(|a| qualifier_matches(&a.qual, qualifier, cs))
    }

    /// The fields of the struct `s` (a qualified `t.*`) names, as resolved
    /// field expressions. An unqualified star, a star with `EXCEPT` or a
    /// DataFrame star, a qualifier no column has, or a qualifier of a
    /// non-struct column falls back to the input columns
    /// ([`Self::expand_star_as_columns`]): an unqualified star is all of
    /// them, while a qualified one that is not a struct errors there when
    /// nothing matches.
    pub(super) fn struct_star(
        &mut self,
        s: &Star,
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Vec<(String, Arc<Expr>)>, AnalysisError> {
        let st = match s.qualifier[..] {
            [] => None,
            [..] if !s.except.is_empty() || s.plan_id.is_some() => None,
            _ => match self.find(cx.scope, &s.qualifier, origin, false)? {
                Some((a, used)) if used == s.qualifier.len() => match &a.col.data_type {
                    DataType::Struct(st) => Some((a.clone(), st.clone())),
                    _ => return self.columns_with_names(s, cx, origin),
                },
                _ => return self.columns_with_names(s, cx, origin),
            },
        };
        let Some((attr, st)) = st else {
            return self.columns_with_names(s, cx, origin);
        };
        self.used(&attr);
        Ok(st
            .fields()
            .iter()
            .enumerate()
            .map(|(ordinal, field)| {
                let nullable = field.nullable() || attr.col.nullable;
                let name = field.name().to_owned();
                let field = Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Field {
                        base: Arc::new(
                            Expr::new(ExprNode::Resolved(ResolvedExpr::Column(attr.col.clone())))
                                .with_origin(origin),
                        ),
                        ordinal,
                        name: name.clone(),
                        data_type: field.data_type().clone(),
                        nullable,
                    }))
                    .with_origin(origin),
                );
                (name, field)
            })
            .collect())
    }

    /// The field names of the struct column a qualified star in a select
    /// list names (`s.*`, `t._metadata.*`): in the input, else among the
    /// columns the relation hides (a file table's `_metadata`). `None` when
    /// the qualifier is no struct column.
    pub(in crate::analyzer) fn star_struct_fields(
        &self,
        s: &Star,
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Option<Vec<String>>, AnalysisError> {
        if s.qualifier.is_empty()
            || s.plan_id.is_some()
            || s.regex.is_some()
            || s.position.is_some()
            || !s.except.is_empty()
            || !s.replace.is_empty()
            // A table or alias of the name wins over a struct column of it.
            || self.qualifier_matches_scope(&s.qualifier, cx)
        {
            return Ok(None);
        }
        let hidden: &[Attr] = match cx.rels {
            [r] => &r.hidden[..],
            _ => &[],
        };
        for level in [cx.scope, hidden] {
            if let Some((a, used)) = self.find(level, &s.qualifier, origin, false)?
                && used == s.qualifier.len()
            {
                return Ok(match &a.col.data_type {
                    DataType::Struct(st) => {
                        Some(st.fields().iter().map(|f| f.name().to_owned()).collect())
                    }
                    _ => None,
                });
            }
        }
        Ok(None)
    }

    /// [`Self::expand`] as resolved column expressions with their names.
    pub(super) fn columns_with_names(
        &mut self,
        s: &Star,
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Vec<(String, Arc<Expr>)>, AnalysisError> {
        Ok(self
            .expand(s, cx, origin)?
            .into_iter()
            .map(|attr| {
                let name = attr.col.name.clone();
                let expr = Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Column(attr.col)))
                        .with_origin(origin),
                );
                (name, expr)
            })
            .collect())
    }

    /// The attributes `*`, `t.*` or `df["*"]` stands for.
    pub(in crate::analyzer) fn expand(
        &mut self,
        s: &Star,
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Vec<Attr>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let out: Vec<Attr> = match s.plan_id {
            Some(id) => carriers(cx, id)
                .into_iter()
                .filter(|a| cx.scope.iter().any(|b| b.col.id == a.col.id))
                .collect(),
            None => cx
                .scope
                .iter()
                .filter(|a| qualifier_matches(&a.qual, &s.qualifier, cs))
                // The columns of a select-list generator are read by its own
                // star, not by `*`.
                .filter(|a| {
                    !s.qualifier.is_empty()
                        || !a
                            .qual
                            .iter()
                            .any(|q| q.starts_with(crate::GENERATE_QUALIFIER_PREFIX))
                })
                .cloned()
                .collect(),
        };
        // A qualified star reaches the columns a `USING` join merged
        // away, ahead of the output columns as in Spark.
        let out = match cx.rels {
            [r] if s.plan_id.is_none()
                && s.regex.is_none()
                && !s.qualifier.is_empty()
                && cx.agg_scope.is_none()
                && r.hidden.iter().any(|a| {
                    a.col.name != crate::analyzer::ops::METADATA_COLUMN
                        && qualifier_matches(&a.qual, &s.qualifier, cs)
                }) =>
            {
                // Spark lists the merged-away columns first, then the output.
                // (A file table's `_metadata` is not one of them.)
                let mut all: Vec<_> = r
                    .hidden
                    .iter()
                    .filter(|a| {
                        a.col.name != crate::analyzer::ops::METADATA_COLUMN
                            && qualifier_matches(&a.qual, &s.qualifier, cs)
                    })
                    .cloned()
                    .collect();
                all.extend(out);
                all
            }
            _ => out,
        };
        // A quoted column regex keeps the columns whose names it matches in full (ignoring case
        // unless the session is case sensitive).
        let had_columns = !out.is_empty();
        let out = match &s.regex {
            Some(pattern) => {
                let matcher = krishiv_functions::scalar::FullMatch::new(pattern, !cs)
                    .map_err(|e| AnalysisError::new(e, origin))?;
                let mut kept = Vec::with_capacity(out.len());
                for a in out {
                    if matcher
                        .is_match(&a.col.name)
                        .map_err(|e| AnalysisError::new(e, origin))?
                    {
                        kept.push(a);
                    }
                }
                kept
            }
            None => out,
        };
        let out = match s.position {
            Some(n) if (1..=out.len()).contains(&n) => vec![out[n - 1].clone()],
            Some(n) => {
                return Err(AnalysisError::new(
                    GroupByPosOutOfRange {
                        index: n.to_string(),
                        size: out.len().to_string(),
                    },
                    origin,
                ));
            }
            None => out,
        };
        // A qualified regex that matches nothing selects nothing; only a qualifier no column
        // answers to cannot be expanded.
        if out.is_empty() && !s.qualifier.is_empty() && !(s.regex.is_some() && had_columns) {
            return Err(AnalysisError::new(
                CannotResolveStarExpand {
                    target_string: quoted(&s.qualifier),
                    columns: names(cx.scope, false),
                },
                origin,
            ));
        }
        out.iter().for_each(|a| self.used(a));
        if s.except.is_empty() {
            return Ok(out);
        }
        self.apply_except(s, &out, origin)
    }

    /// `EXCEPT_OVERLAPPING_COLUMNS` when one entry of `s.except` is a prefix
    /// of another.
    fn except_overlap(&self, s: &Star, origin: Origin) -> Result<(), AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let folded: Vec<Vec<String>> = s
            .except
            .iter()
            .map(|p| p.iter().map(|x| fold_case(x, cs)).collect())
            .collect();
        for (i, a) in folded.iter().enumerate() {
            for (j, b) in folded.iter().enumerate().take(i) {
                if is_prefix(b, a) || is_prefix(a, b) {
                    return Err(AnalysisError::new(
                        ExceptOverlappingColumns {
                            columns: format!(
                                "{}, {}",
                                s.except[j].join("."),
                                s.except[i].join(".")
                            ),
                        },
                        origin,
                    ));
                }
            }
        }
        Ok(())
    }

    /// `* EXCEPT (a.b)` and `s.* EXCEPT (...)` over struct columns as the
    /// items of a select list: each column a nested entry (or a struct
    /// qualifier) touches is rebuilt without the dropped fields (Spark's
    /// `DropField`, here [`ExprNode::UpdateFields`]), the others are
    /// kept. `None` when the star has no nested entry and no struct
    /// qualifier (the plain path handles it).
    pub(in crate::analyzer) fn nested_except_items(
        &mut self,
        s: &Star,
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Option<Vec<Arc<Expr>>>, AnalysisError> {
        if s.except.is_empty() || s.plan_id.is_some() || s.regex.is_some() || s.position.is_some() {
            return Ok(None);
        }
        let cs = self.ctx.conf.case_sensitive;
        let resolved = |c: &BoundColumn| {
            Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Column(c.clone()))).with_origin(origin),
            )
        };
        // The entries (name, value, type) and, per entry, the field paths
        // below it to drop (an empty path drops the entry itself).
        let mut entries: Vec<(String, Arc<Expr>, DataType)> = Vec::new();
        let mut drops: Vec<Vec<Vec<String>>> = Vec::new();
        // `data` or `data.s2`: a struct column, or a struct field below one.
        let struct_qualifier = if s.qualifier.is_empty() {
            None
        } else {
            match self.find(cx.scope, &s.qualifier, origin, false)? {
                Some((a, used)) => {
                    let mut base = resolved(&a.col);
                    let mut ty = a.col.data_type.clone();
                    for part in &s.qualifier[used..] {
                        let DataType::Struct(st) = &ty else {
                            return Ok(None);
                        };
                        let Some(f) = st.fields().iter().find(|f| same(f.name(), part, cs)) else {
                            return Ok(None);
                        };
                        base = Arc::new(
                            Expr::new(ExprNode::Field {
                                base,
                                name: f.name().to_owned(),
                            })
                            .with_origin(origin),
                        );
                        ty = f.data_type().clone();
                    }
                    match ty {
                        DataType::Struct(st) => Some((a, base, st)),
                        _ => None,
                    }
                }
                None => None,
            }
        };
        self.except_overlap(s, origin)?;
        if let Some((attr, root, st)) = struct_qualifier {
            self.used(&attr);
            for field in st.fields() {
                let value = Arc::new(
                    Expr::new(ExprNode::Field {
                        base: Arc::clone(&root),
                        name: field.name().to_owned(),
                    })
                    .with_origin(origin),
                );
                entries.push((field.name().to_owned(), value, field.data_type().clone()));
                drops.push(Vec::new());
            }
            for path in &s.except {
                let Some(i) = st
                    .fields()
                    .iter()
                    .position(|f| same(f.name(), &path[0], cs))
                else {
                    return Err(unresolved_field(path, &st, origin));
                };
                drops[i].push(path[1..].to_vec());
            }
        } else {
            let mut plain = s.clone();
            plain.except.clear();
            let out = self.expand(&plain, cx, origin)?;
            let mut any_nested = false;
            let mut targets: Vec<(usize, Vec<String>)> = Vec::new();
            for path in &s.except {
                let hit = (1..=path.len()).rev().find_map(|j| {
                    match self.except_top(&path[..j], &out, cs)[..] {
                        [ref a] => out
                            .iter()
                            .position(|o| o.col.id == a.col.id)
                            .map(|i| (i, j)),
                        _ => None,
                    }
                });
                let Some((i, j)) = hit else {
                    return Ok(None);
                };
                any_nested |= j < path.len();
                targets.push((i, path[j..].to_vec()));
            }
            if !any_nested {
                return Ok(None);
            }
            for a in &out {
                entries.push((
                    a.col.name.clone(),
                    resolved(&a.col),
                    a.col.data_type.clone(),
                ));
                drops.push(Vec::new());
            }
            for (i, rest) in targets {
                drops[i].push(rest);
            }
        }
        let mut items = Vec::new();
        for ((name, value, ty), rests) in entries.into_iter().zip(drops) {
            if rests.iter().any(Vec::is_empty) {
                continue;
            }
            let value = if rests.is_empty() {
                value
            } else {
                self.drop_nested(&value, &ty, &rests, origin)?
            };
            items.push(Arc::new(
                Expr::new(ExprNode::Alias { expr: value, name }).with_origin(origin),
            ));
        }
        Ok(Some(items))
    }

    /// `base` (a struct of type `ty`) without the fields `rests` name, each
    /// a path below it: an unknown name is `FIELD_NOT_FOUND`.
    fn drop_nested(
        &self,
        base: &Arc<Expr>,
        ty: &DataType,
        rests: &[Vec<String>],
        origin: Origin,
    ) -> Result<Arc<Expr>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let DataType::Struct(st) = ty else {
            return Err(unresolved_field(&rests[0], &StructType::new([]), origin));
        };
        let mut ops: Vec<FieldOp> = Vec::new();
        let mut seen: Vec<usize> = Vec::new();
        for rest in rests {
            let Some(i) = st
                .fields()
                .iter()
                .position(|f| same(f.name(), &rest[0], cs))
            else {
                return Err(unresolved_field(rest, st, origin));
            };
            if seen.contains(&i) {
                continue;
            }
            seen.push(i);
            let field = &st.fields()[i];
            let name = field.name().to_owned();
            let subs: Vec<Vec<String>> = rests
                .iter()
                .filter(|r| same(&r[0], &name, cs))
                .map(|r| r[1..].to_vec())
                .collect();
            if subs.iter().any(Vec::is_empty) {
                ops.push(FieldOp::Drop { name });
            } else {
                let inner = Arc::new(
                    Expr::new(ExprNode::Field {
                        base: Arc::clone(base),
                        name: name.clone(),
                    })
                    .with_origin(origin),
                );
                let value = self.drop_nested(&inner, field.data_type(), &subs, origin)?;
                ops.push(FieldOp::With { name, value });
            }
        }
        let dropped = ops
            .iter()
            .filter(|o| matches!(o, FieldOp::Drop { .. }))
            .count();
        if dropped == st.fields().len() {
            // Every field goes: `struct<>`, which a struct update refuses
            // but Spark's `EXCEPT` gives (null stays null).
            let call = |name: &str, args: Vec<Arc<Expr>>| {
                Arc::new(
                    Expr::new(ExprNode::Function(FunctionCall {
                        within_group: Vec::new(),
                        name: name.to_owned(),
                        qualifier: Vec::new(),
                        args,
                        arg_names: Vec::new(),
                        distinct: false,
                        filter: None,
                    }))
                    .with_origin(origin),
                )
            };
            let null = Arc::new(
                Expr::new(ExprNode::Cast {
                    expr: Arc::new(
                        Expr::new(ExprNode::Literal(Literal::Null(None))).with_origin(origin),
                    ),
                    to: DataType::Struct(StructType::new([])),
                    mode: CastMode::Default,
                })
                .with_origin(origin),
            );
            return Ok(call(
                "when",
                vec![
                    call("isnull", vec![Arc::clone(base)]),
                    null,
                    call("named_struct", vec![]),
                ],
            ));
        }
        Ok(Arc::new(
            Expr::new(ExprNode::UpdateFields {
                base: Arc::clone(base),
                ops,
            })
            .with_origin(origin),
        ))
    }

    /// What `* EXCEPT (cols)` drops from `out` (the star's own expansion).
    /// Each entry names one of those columns, bare or qualified with the
    /// table (as in `ids.id`); longer paths name nested struct fields,
    /// which need struct support (`KRISHIV_UNSUPPORTED_FEATURE`), and
    /// unknown names are `UNRESOLVED_COLUMN`. Overlapping entries are
    /// `EXCEPT_OVERLAPPING_COLUMNS`, checked before anything resolves (as
    /// Spark checks them).
    pub(super) fn apply_except(
        &mut self,
        s: &Star,
        out: &[Attr],
        origin: Origin,
    ) -> Result<Vec<Attr>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        self.except_overlap(s, origin)?;
        let mut dropped: Vec<(AttrId, String)> = Vec::new();
        for path in &s.except {
            let raw = path.join(".");
            match self.except_target(path, out, cs)? {
                ExceptTarget::Top(a) => {
                    if let Some((_, first)) = dropped.iter().find(|(id, _)| *id == a.col.id) {
                        return Err(AnalysisError::new(
                            ExceptOverlappingColumns {
                                columns: format!("{first}, {raw}"),
                            },
                            origin,
                        ));
                    }
                    dropped.push((a.col.id, raw));
                }
                ExceptTarget::Nested => {
                    return Err(AnalysisError::new(
                        KrishivUnsupportedFeature {
                            feature: format!("SELECT * EXCEPT with nested column `{raw}`"),
                        },
                        origin,
                    ));
                }
                ExceptTarget::Missing => {
                    return Err(unresolved(path, out, origin));
                }
            }
        }
        let gone: HashSet<AttrId> = dropped.into_iter().map(|(id, _)| id).collect();
        Ok(out
            .iter()
            .filter(|a| !gone.contains(&a.col.id))
            .cloned()
            .collect())
    }

    /// Records a column of a securable as referenced.
    pub(in crate::analyzer) fn used(&mut self, a: &Attr) {
        if let Some((i, name)) = &a.source
            && let Some(set) = self.columns.get_mut(*i)
        {
            set.insert(name.to_string());
        }
    }

    /// The column of `out` an `EXCEPT` entry names: a top-level column
    /// (`Top`), a field below one (`Nested`), or nothing (`Missing`). A
    /// bare name matches by column name; a longer path matches when its
    /// last part names the column and the leading parts match the
    /// qualifier (as in `ids.id`).
    pub(super) fn except_target(
        &self,
        path: &[String],
        out: &[Attr],
        cs: bool,
    ) -> Result<ExceptTarget, AnalysisError> {
        if path.is_empty() {
            return Err(unsupported("an empty EXCEPT entry", Origin::UNKNOWN));
        }
        match self.except_top(path, out, cs)[..] {
            [ref a] => Ok(ExceptTarget::Top(a.clone())),
            [] => {
                // A field below a top-level column when a strict prefix of
                // the entry names one (as in `t.id.sub`, or `id.sub`).
                let nested = path.len() > 1
                    && (1..path.len()).any(|j| !self.except_top(&path[..j], out, cs).is_empty());
                if nested {
                    Ok(ExceptTarget::Nested)
                } else {
                    Ok(ExceptTarget::Missing)
                }
            }
            _ => Err(AnalysisError::new(
                AmbiguousReference {
                    name: quoted(path),
                    reference_names: format!("[{}]", names(out, false)),
                },
                Origin::UNKNOWN,
            )),
        }
    }

    /// The columns of `out` an `EXCEPT` entry prefix names: a bare name
    /// matches by column name; a longer path matches when its last part
    /// names the column and the leading parts match the qualifier (as in
    /// `ids.id`).
    pub(super) fn except_top(&self, path: &[String], out: &[Attr], cs: bool) -> Vec<Attr> {
        let Some((last, leading)) = path.split_last() else {
            return Vec::new();
        };
        if leading.is_empty() {
            return out
                .iter()
                .filter(|a| same(&a.col.name, last, cs))
                .cloned()
                .collect();
        }
        out.iter()
            .filter(|a| {
                same(&a.col.name, last, cs)
                    && leading.len() <= a.qual.len()
                    && a.qual[a.qual.len() - leading.len()..]
                        .iter()
                        .zip(leading)
                        .all(|(x, y)| same(x, y, cs))
            })
            .cloned()
            .collect()
    }
}
