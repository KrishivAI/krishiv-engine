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
        if out.is_empty() && !s.qualifier.is_empty() {
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
