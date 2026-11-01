// SPDX-License-Identifier: Apache-2.0
//! Column and field resolution, and Spark's `UNRESOLVED_COLUMN`/`UNRESOLVED_FIELD` errors with their suggestions.

use super::*;

/// `FIELD_NOT_FOUND` for a struct field `parts` names below a struct with
/// `fields` (Spark's `GetStructField` by an unknown name, checked against
/// Spark 4.2.0 for `s.f`, `(s).f`, nested paths and `Column.getField`): the
/// missing name, and every field of the struct.
pub(in crate::analyzer) fn unresolved_field(
    parts: &[String],
    fields: &krishiv_types::StructType,
    origin: Origin,
) -> AnalysisError {
    let name = parts.last().map_or_else(String::new, |n| format!("`{n}`"));
    let list = fields
        .fields()
        .iter()
        .map(|f| format!("`{}`", f.name()))
        .collect::<Vec<_>>()
        .join(", ");
    AnalysisError::new(
        FieldNotFound {
            field_name: name,
            fields: list,
        },
        origin,
    )
}

/// `UNRESOLVED_COLUMN`, with up to five of `scope`'s columns ranked by edit
/// distance (qualified by their last qualifier part when `parts` is).
/// Distances are over the quoted candidate and the unquoted target with
/// ties in name order, as Spark's `orderSuggestedIdentifiersBySimilarity`
/// orders them (checked against Spark 4.2.0: `` `id`, `name`, `power`,
/// `city`, `country` `` for `non_exist`, `` `id`, `name`, `data` `` for
/// `invalid_column`).
pub(in crate::analyzer) fn unresolved(
    parts: &[String],
    scope: &[Attr],
    origin: Origin,
) -> AnalysisError {
    let qualified = parts.len() > 1;
    let candidates = scope
        .iter()
        .map(|a| {
            let mut p: Vec<String> = a
                .qual
                .last()
                .filter(|_| qualified)
                .cloned()
                .into_iter()
                .collect();
            p.push(a.col.name.clone());
            p
        })
        .collect();
    let ranked = closest(&parts.join("."), candidates);
    let object_name = quoted(parts);
    if ranked.is_empty() {
        return AnalysisError::new(UnresolvedColumnWithoutSuggestion { object_name }, origin);
    }
    let proposal = ranked
        .iter()
        .map(|p| quoted(p))
        .collect::<Vec<_>>()
        .join(", ");
    AnalysisError::new(
        UnresolvedColumnWithSuggestion {
            object_name,
            proposal,
        },
        origin,
    )
}

/// The quoted names of `attrs`, comma-separated.
pub(in crate::analyzer) fn names(attrs: &[Attr], _qualified: bool) -> String {
    attrs
        .iter()
        .map(|a| quoted(std::slice::from_ref(&a.col.name)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The `UNRECOGNIZED_PARAMETER_NAME` proposal for `name`: the parameter
/// names most similar to it, up to three, as Spark's
/// `orderSuggestedIdentifiersBySimilarity` orders them (Levenshtein over
/// the backticked names, ties in Scala-`Set` order).
pub(super) fn suggestions(name: &str, params: impl Iterator<Item = &'static str>) -> String {
    let params: Vec<&str> = params.collect();
    let order = scala_set_order(&params);
    let mut scored: Vec<(usize, usize, String)> = params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let quoted = format!("`{}`", p.replace('`', "``"));
            let rank = order.iter().position(|o| *o == i).unwrap_or(i);
            (utf16_distance(&quoted, name), rank, quoted)
        })
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(3)
        .map(|(_, _, q)| q)
        .collect::<Vec<_>>()
        .join(" ")
}

/// The Levenshtein distance of `a` and `b` in UTF-16 units, as Apache
/// Commons Text computes it.
pub(super) fn utf16_distance(a: &str, b: &str) -> usize {
    let a: Vec<u16> = a.encode_utf16().collect();
    let b: Vec<u16> = b.encode_utf16().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(x != y)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

/// Java's `String.hashCode` in UTF-16 units.
pub(super) fn java_hash(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(i32::from(c)))
}

/// Scala's hash-code spread for hash tries.
pub(super) fn improve(hcode: i32) -> i32 {
    let h = hcode.wrapping_add(!hcode.wrapping_shl(9));
    let h = h ^ (h.cast_unsigned() >> 14) as i32;
    let h = h.wrapping_add(h.wrapping_shl(4));
    h ^ (h.cast_unsigned() >> 10) as i32
}

/// The iteration order of a small Scala `Set` of `strings`, as position in
/// the returned order: a hash trie walked in ascending 5-bit chunks of the
/// improved hash.
pub(super) fn scala_set_order(strings: &[&str]) -> Vec<usize> {
    fn walk(items: Vec<(u32, usize)>, shift: u32, out: &mut Vec<usize>) {
        if items.len() <= 1 || shift >= 32 {
            out.extend(items.into_iter().map(|(_, i)| i));
            return;
        }
        let mut groups: Vec<(u32, Vec<(u32, usize)>)> = Vec::new();
        for (h, i) in items {
            let chunk = (h >> shift) & 31;
            match groups.iter_mut().find(|(c, _)| *c == chunk) {
                Some((_, g)) => g.push((h, i)),
                None => groups.push((chunk, vec![(h, i)])),
            }
        }
        groups.sort_by_key(|(c, _)| *c);
        for (_, g) in groups {
            walk(g, shift + 5, out);
        }
    }
    let items: Vec<(u32, usize)> = strings
        .iter()
        .enumerate()
        .map(|(i, s)| (improve(java_hash(s)).cast_unsigned(), i))
        .collect();
    let mut out = Vec::new();
    walk(items, 0, &mut out);
    out
}

impl Resolver<'_> {
    /// A new attribute named `name` of type `ty`.
    pub(in crate::analyzer) fn attr(&mut self, name: &str, ty: &ArgType) -> BoundColumn {
        let data_type = ty.data_type.clone().unwrap_or(DataType::Null);
        BoundColumn {
            id: self.fresh(),
            name: name.to_owned(),
            data_type,
            nullable: ty.nullable,
        }
    }

    /// A column reference, in Spark's order: the input, lateral column
    /// aliases (a select list), select-list aliases (a `GROUP BY`), the
    /// enclosing query (only the immediately enclosing one).
    pub(super) fn column(
        &mut self,
        c: &ColumnRef,
        origin: Origin,
        in_agg: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let bound = |expr: ExprNode, col: &BoundColumn, outer| Bound {
            expr: Arc::new(Expr::new(expr).with_origin(origin)),
            ty: column(col),
            agg: None,
            win: None,
            outer,
            lateral: Vec::new(),
            shown: None,
        };
        let renamed = |a: &Attr, parts: &[String], used: usize| {
            // A reference keeps the spelling it was written with.
            let mut col = a.col.clone();
            col.name.clone_from(&parts[used - 1]);
            col
        };
        if c.plan_id.is_none()
            && let [name] = &c.name[..]
            && let Some(var) = self.lambda_var(name)
        {
            // A lambda variable shadows the enclosing query's columns.
            let mut col = var;
            col.name.clone_from(name);
            return Ok(bound(
                ExprNode::Resolved(ResolvedExpr::LambdaVar(col.clone())),
                &col,
                None,
            ));
        }
        if let Some(id) = c.plan_id {
            let a = self.plan_column(id, &c.name, cx, origin)?;
            self.used(&a);
            let col = renamed(&a, &c.name, c.name.len());
            return Ok(bound(
                ExprNode::Resolved(ResolvedExpr::Column(col.clone())),
                &col,
                None,
            ));
        }
        let (scope, extra) = match cx.agg_scope {
            Some(input) if in_agg => (input, None),
            _ => (cx.scope, cx.extra),
        };
        // With leniency (an `ORDER BY` over duplicate outputs, a `HAVING`
        // or `ORDER BY` over an aggregate whose select list repeats a
        // name), duplicates in the first level are looked up below it; when
        // nothing below resolves them either, they stay ambiguous rather
        // than unresolved.
        let mut lenient_dup = false;
        for (i, level) in [Some(scope), extra].into_iter().enumerate() {
            let Some(level) = level else { continue };
            let lenient = cx.lenient && i == 0;
            match self.find(level, &c.name, origin, lenient) {
                Ok(Some((a, used))) => {
                    self.used(&a);
                    return self.resolve_attr(&a, used, &c.name, origin, None);
                }
                Ok(None) => {
                    lenient_dup = lenient_dup || lenient;
                }
                Err(e) => return Err(e),
            }
        }
        if lenient_dup {
            // Duplicates in the first level, nothing below resolved them:
            // they stay ambiguous rather than unresolved, before a
            // function, alias, outer column or session variable can match.
            self.find(scope, &c.name, origin, false)?;
        }
        // A name no input column has may be a function written without
        // parentheses (what the parser leaves a column unless ANSI reserved
        // keywords are on). Spark 4.2 tries it before lateral aliases,
        // GROUP BY aliases and outer references.
        if let [name] = &c.name[..]
            && !(self.ctx.conf.udf_parameter_shadows_function && self.has_parameter(name))
            && let Some(function) = LITERAL_FUNCTIONS
                .iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(name))
                .map(|(_, f)| *f)
            // A release without the function reads the name as a column.
            && matches!(
                self.ctx.functions.lookup(&[function.to_owned()], self.ctx.profile),
                FunctionLookup::Found(_) | FunctionLookup::KnownUnsupported
            )
        {
            let call = Arc::new(
                Expr::new(ExprNode::Function(FunctionCall {
                    name: function.to_owned(),
                    qualifier: Vec::new(),
                    arg_names: Vec::new(),
                    args: Vec::new(),
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
                }))
                .with_origin(origin),
            );
            return self.expr(&call, cx);
        }
        if let ([name], Some(lateral)) = (&c.name[..], cx.lateral) {
            let cs = self.ctx.conf.case_sensitive;
            let hits: Vec<&Attr> = lateral
                .iter()
                .filter(|a| same(&a.col.name, name, cs))
                .collect();
            match hits[..] {
                [] => {}
                [a] => {
                    let mut col = a.col.clone();
                    col.name.clone_from(name);
                    let mut b = bound(
                        ExprNode::Resolved(ResolvedExpr::Column(col.clone())),
                        &col,
                        None,
                    );
                    b.lateral.push(col.id);
                    return Ok(b);
                }
                _ => {
                    return Err(AnalysisError::new(
                        AmbiguousLateralColumnAlias {
                            name: quoted(std::slice::from_ref(name)),
                            n: hits.len().to_string(),
                        },
                        Origin::UNKNOWN,
                    ));
                }
            }
        }
        if let [name] = &c.name[..]
            && let Some((_, alias)) = cx
                .aliases
                .iter()
                .find(|(n, _)| same(n, name, self.ctx.conf.case_sensitive))
        {
            let inner = Cx {
                aliases: &[],
                ..*cx
            };
            return self.expr(alias, &inner);
        }
        // Only the immediately enclosing query, as Spark 4.2 resolves.
        let levels = cx.env.outer.iter().enumerate().rev().take(1);
        for (level, outer) in levels.filter(|_| cx.outer) {
            if let Some((a, used)) = self.find(&outer.scope, &c.name, origin, false)? {
                self.used(&a);
                return self.resolve_attr(&a, used, &c.name, origin, Some(level));
            }
        }
        // The parameters of the SQL function being resolved rank after
        // everything else, but before a session variable.
        if let Some(b) = self.parameter(&c.name, origin, cx.env.outer.len())? {
            return Ok(b);
        }
        // A name no column or outer reference has may be a session
        // variable (`v`, `session.v` or `system.session.v`): Spark
        // resolves columns and outer references first, then variables. A
        // name of another shape, or one naming no variable, stays an
        // unresolved column. The reference becomes the variable's current
        // value as a typed literal, so no physical, executor or plan-codec
        // change is needed; the expression keeps the spelling it was
        // written with for names and errors.
        if let Some(var) = self.ctx.variables.get(&c.name) {
            let Some(name) = c.name.last() else {
                debug_assert!(false, "a variable name is never empty");
                return Err(unresolved(&c.name, cx.scope, origin));
            };
            let col = BoundColumn {
                id: self.fresh(),
                name: name.clone(),
                data_type: var.data_type.clone(),
                nullable: matches!(var.value, Literal::Null(_)),
            };
            return Ok(Bound {
                expr: Arc::new(Expr::new(ExprNode::Literal(var.value.clone())).with_origin(origin)),
                ty: column(&col),
                agg: None,
                win: None,
                outer: None,
                lateral: Vec::new(),
                shown: Some(Arc::new(
                    Expr::new(ExprNode::Column(c.clone())).with_origin(origin),
                )),
            });
        }
        if self.parameter_attrs().is_empty() {
            return Err(unresolved(&c.name, cx.scope, origin));
        }
        let candidates: Vec<Attr> = cx
            .scope
            .iter()
            .chain(self.parameter_attrs())
            .cloned()
            .collect();
        Err(unresolved(&c.name, &candidates, origin))
    }

    /// `df["a"]`: the column `name` of the relations carrying `id` in the
    /// operator's subtree.
    pub(super) fn plan_column(
        &self,
        id: crate::PlanId,
        name: &[String],
        cx: &Cx<'_>,
        origin: Origin,
    ) -> Result<Attr, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let pick = |carried: Vec<Attr>, scope: &[Attr]| {
            let mut hits: Vec<Attr> = Vec::new();
            for a in carried {
                let visible = scope.iter().any(|s| s.col.id == a.col.id);
                if visible
                    && name.len() == 1
                    && same(&a.col.name, &name[0], cs)
                    && !hits.iter().any(|h| h.col.id == a.col.id)
                {
                    hits.push(a);
                }
            }
            hits
        };
        let sql_name = format!("\"{}\"", name.join("."));
        // The error names the DataFrame the reference belongs to.
        let origin = match origin.plan_id {
            Some(_) => origin,
            None => origin.with_plan_id(id),
        };
        // Only the operator's own subtree: Spark 4.2 does not resolve a
        // DataFrame reference against an enclosing query (golden case
        // `outer`).
        let mut hits = pick(carriers(cx, id), cx.scope);
        match hits.len() {
            1 => Ok(hits.remove(0)),
            0 => Err(AnalysisError::new(
                CannotResolveDataframeColumn { name: sql_name },
                origin,
            )),
            _ => Err(AnalysisError::new(
                AmbiguousColumnReference { name: sql_name },
                origin,
            )),
        }
    }

    /// The attribute `parts` names in `scope` and how many parts it used:
    /// with a qualifier of 3, 2 or 1 parts (a suffix of the attribute's),
    /// then by name alone; the first form with any match decides. Several
    /// matches are `AMBIGUOUS_REFERENCE`, or no match when `lenient`. A
    /// single match with leftover parts resolves them as struct fields in
    /// [`Self::resolve_attr`].
    pub(super) fn find(
        &self,
        scope: &[Attr],
        parts: &[String],
        origin: Origin,
        lenient: bool,
    ) -> Result<Option<(Attr, usize)>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        for q in (0..parts.len().min(4)).rev() {
            let mut hits: Vec<&Attr> = Vec::new();
            for a in scope {
                let qualified = a.qual.len() >= q
                    && a.qual[a.qual.len() - q..]
                        .iter()
                        .zip(&parts[..q])
                        .all(|(x, y)| same(x, y, cs));
                if qualified
                    && same(&a.col.name, &parts[q], cs)
                    && !hits.iter().any(|h| h.col.id == a.col.id)
                {
                    hits.push(a);
                }
            }
            match hits[..] {
                [] => continue,
                [a] if parts.len() > q + 1 => {
                    // A longer path may name struct fields below the
                    // column; the caller resolves them (or reports the
                    // base type when the column is not a struct).
                    return Ok(Some((a.clone(), q + 1)));
                }
                [a] => return Ok(Some((a.clone(), q + 1))),
                _ if lenient => return Ok(None),
                _ => {
                    let mut names: Vec<String> = hits
                        .iter()
                        .map(|a| {
                            let mut p = a.qual.to_vec();
                            p.push(a.col.name.clone());
                            quoted(&p)
                        })
                        .collect();
                    names.sort();
                    return Err(AnalysisError::new(
                        AmbiguousReference {
                            name: quoted(parts),
                            reference_names: format!("[{}]", names.join(", ")),
                        },
                        origin,
                    ));
                }
            }
        }
        Ok(None)
    }

    /// `struct` with named children as `named_struct`, or `None` when every
    /// child defaults to `colN`: a column contributes its (written) name, a
    /// struct field its field name, an alias its output name.
    pub(super) fn struct_names(
        &self,
        f: &FunctionCall,
        kids: &[Bound],
        origin: Origin,
    ) -> Option<Arc<Expr>> {
        let mut names = Vec::with_capacity(kids.len());
        for (i, k) in kids.iter().enumerate() {
            let name = match &k.expr.node {
                ExprNode::Resolved(
                    ResolvedExpr::Column(c)
                    | ResolvedExpr::OuterColumn(c)
                    | ResolvedExpr::LambdaVar(c),
                ) => c.name.clone(),
                ExprNode::Resolved(ResolvedExpr::Field { name, .. }) => name.clone(),
                ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => output.name.clone(),
                _ => format!("col{}", i + 1),
            };
            names.push(name);
        }
        if names
            .iter()
            .enumerate()
            .all(|(i, n)| *n == format!("col{}", i + 1))
            && !f
                .args
                .iter()
                .any(|a| matches!(a.node, ExprNode::Alias { .. }))
        {
            return None;
        }
        let mut args = Vec::with_capacity(2 * kids.len());
        for (name, k) in names.into_iter().zip(kids) {
            args.push(Arc::new(
                Expr::new(ExprNode::Literal(Literal::String(name))).with_origin(origin),
            ));
            args.push(Arc::clone(&k.expr));
        }
        Some(Arc::new(
            Expr::new(ExprNode::Function(FunctionCall {
                name: "named_struct".to_owned(),
                qualifier: Vec::new(),
                args,
                arg_names: Vec::new(),
                distinct: f.distinct,
                filter: None,
                within_group: Vec::new(),
            }))
            .with_origin(origin),
        ))
    }

    /// `UpdateFields(base, ops)` (PySpark `withField`/`dropFields`): the
    /// struct is rebuilt by `named_struct`, keeping every field the ops do
    /// not touch, so the executor and the kernels get nothing new. A `With`
    /// replaces every field matching its name (compared per
    /// `spark.sql.caseSensitive`) in place, else appends the field last; a
    /// `Drop` removes every matching field and ignores names that match
    /// nothing. The rewrite is `CASE WHEN base IS NULL THEN CAST(NULL AS
    /// <new type>) ELSE named_struct(...) END`, bound here, so its field
    /// order, names, types and nullability are the rewritten
    /// `named_struct`'s and no resolved plan holds the node. The base
    /// appears several times in the rewrite: a nondeterministic base would
    /// read a different value per use, and nothing below binds it once, so
    /// that is `KRISHIV_UNSUPPORTED_FEATURE`.
    pub(super) fn update_fields(
        &mut self,
        e: &Arc<Expr>,
        ops: &[FieldOp],
        kids: &[Bound],
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let base = &kids[0];
        let struct_type = match &base.ty.data_type {
            Some(DataType::Struct(st)) => st.clone(),
            _ => {
                let shown = with_children(e, kids.iter().map(Bound::shown).collect());
                return Err(AnalysisError::new(
                    DatatypeMismatchUnexpectedInputType {
                        sql_expr: format!("\"{}\"", pretty(&shown, self.ctx.conf.session_zone)),
                        param_index: ordinal(0),
                        required_type: "\"STRUCT\"".to_owned(),
                        input_sql: format!(
                            "\"{}\"",
                            pretty(&base.shown(), self.ctx.conf.session_zone)
                        ),
                        input_type: quoted_type(base.ty.data_type.as_ref()),
                    },
                    e.origin,
                ));
            }
        };
        if !is_deterministic(&base.expr) {
            return Err(unsupported(
                "withField/dropFields of a nondeterministic expression",
                e.origin,
            ));
        }
        let cs = self.ctx.conf.case_sensitive;
        // One output field per kept struct field (a field extraction of the
        // resolved base, as the SQL path reads it), then the appended ones.
        let mut out: Vec<(String, Arc<Expr>, DataType, bool)> = Vec::new();
        for (ordinal, field) in struct_type.fields().iter().enumerate() {
            let nullable = field.nullable() || base.ty.nullable;
            let value = Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Field {
                    base: Arc::clone(&base.expr),
                    ordinal,
                    name: field.name().to_owned(),
                    data_type: field.data_type().clone(),
                    nullable,
                }))
                .with_origin(e.origin),
            );
            out.push((
                field.name().to_owned(),
                value,
                field.data_type().clone(),
                nullable,
            ));
        }
        let mut values = kids[1..].iter();
        for op in ops {
            match op {
                FieldOp::With { name, .. } => {
                    let Some(v) = values.next() else {
                        return Err(unsupported("a struct update without its value", e.origin));
                    };
                    let updated = (
                        name.clone(),
                        Arc::clone(&v.expr),
                        v.ty.data_type.clone().unwrap_or(DataType::Null),
                        v.ty.nullable,
                    );
                    let mut matched = false;
                    for slot in out.iter_mut() {
                        if same(&slot.0, name, cs) {
                            *slot = updated.clone();
                            matched = true;
                        }
                    }
                    if !matched {
                        out.push(updated);
                    }
                }
                FieldOp::Drop { name } => {
                    out.retain(|slot| !same(&slot.0, name, cs));
                }
            }
        }
        if out.is_empty() {
            return Err(AnalysisError::new(
                DatatypeMismatchCannotDropAllFields {
                    sql_expr: format!(
                        "\"{}\"",
                        pretty(
                            &with_children(e, kids.iter().map(Bound::shown).collect()),
                            self.ctx.conf.session_zone
                        )
                    ),
                },
                e.origin,
            ));
        }
        let new_type = DataType::Struct(StructType::new(
            out.iter()
                .map(|(name, _, t, n)| StructField::new(name.clone(), t.clone(), *n)),
        ));
        // `named_struct(name0, val0, ...)`: bound here, so the output field
        // names, types and nullability are the analyzer's own.
        let mut struct_args: Vec<Arc<Expr>> = Vec::with_capacity(2 * out.len());
        for (name, value, _, _) in &out {
            struct_args.push(Arc::new(
                Expr::new(ExprNode::Literal(Literal::String(name.clone()))).with_origin(e.origin),
            ));
            struct_args.push(Arc::clone(value));
        }
        let struct_call = Arc::new(
            Expr::new(ExprNode::Function(FunctionCall {
                within_group: Vec::new(),
                name: "named_struct".to_owned(),
                qualifier: Vec::new(),
                args: struct_args,
                arg_names: Vec::new(),
                distinct: false,
                filter: None,
            }))
            .with_origin(e.origin),
        );
        let rebuilt = self.expr(&struct_call, cx)?;
        let cond = self.operator_call(Builtin::IsNull, e.origin, std::slice::from_ref(base))?;
        let null_branch = {
            let null =
                Arc::new(Expr::new(ExprNode::Literal(Literal::Null(None))).with_origin(e.origin));
            let cast = Arc::new(
                Expr::new(ExprNode::Cast {
                    expr: null,
                    to: new_type,
                    mode: CastMode::Default,
                })
                .with_origin(e.origin),
            );
            self.expr(&cast, cx)?
        };
        let mut b = self.operator_call(Builtin::When, e.origin, &[cond, null_branch, rebuilt])?;
        // The field values were bound before and rebinding them in the
        // `named_struct` dropped their markers (an aggregate, a window
        // expression, an outer or lateral reference): carry them over.
        b.carry_markers(kids);
        // Shown as written (`update_fields(...)`), not the rewrite.
        b.shown = Some(with_children(e, kids.iter().map(Bound::shown).collect()));
        Ok(b)
    }

    /// The attribute `a` matched with `used` of `parts`: the column itself,
    /// or the struct fields the leftover parts name below it (Spark's
    /// `GetStructField` chain), resolved under `spark.sql.caseSensitive`.
    /// `outer` is `Some` for a column of the enclosing query.
    pub(super) fn resolve_attr(
        &mut self,
        a: &Attr,
        used: usize,
        parts: &[String],
        origin: Origin,
        outer: Option<usize>,
    ) -> Result<Bound, AnalysisError> {
        // A reference keeps the spelling it was written with.
        let mut col = a.col.clone();
        col.name.clone_from(&parts[used - 1]);
        let node = match outer {
            None => ExprNode::Resolved(ResolvedExpr::Column(col.clone())),
            Some(_) => ExprNode::Resolved(ResolvedExpr::OuterColumn(col.clone())),
        };
        if used == parts.len() {
            return Ok(Bound {
                expr: Arc::new(Expr::new(node).with_origin(origin)),
                ty: column(&col),
                agg: None,
                win: None,
                outer,
                lateral: Vec::new(),
                shown: None,
            });
        }
        let cs = self.ctx.conf.case_sensitive;
        let mut ty = column(&col);
        let mut base = Arc::new(Expr::new(node).with_origin(origin));
        let mut parent = parts[used - 1].clone();
        for part in &parts[used..] {
            let st = match &ty.data_type {
                Some(DataType::Struct(st)) => st.clone(),
                _ => {
                    let other = ty
                        .data_type
                        .as_ref()
                        .map_or_else(|| "VOID".to_owned(), DataType::sql);
                    return Err(AnalysisError::new(
                        InvalidExtractBaseFieldType {
                            base: format!("\"{parent}\""),
                            other: format!("\"{other}\""),
                        },
                        origin,
                    ));
                }
            };
            let Some((ordinal, field)) = st
                .fields()
                .iter()
                .enumerate()
                .find(|(_, f)| same(f.name(), part, cs))
            else {
                return Err(unresolved_field(parts, &st, origin));
            };
            parent.clone_from(part);
            let field_ty = arg_type(field.data_type().clone(), field.nullable() || ty.nullable);
            base = Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Field {
                    base: Arc::clone(&base),
                    ordinal,
                    name: part.clone(),
                    data_type: field_ty.data_type.clone().unwrap_or(DataType::Null),
                    nullable: field_ty.nullable,
                }))
                .with_origin(origin),
            );
            ty = field_ty;
        }
        Ok(Bound {
            expr: base,
            ty,
            agg: None,
            win: None,
            outer,
            lateral: Vec::new(),
            shown: None,
        })
    }

    pub(in crate::analyzer) fn sub_env(&mut self, cx: &Cx<'_>) -> Arc<Env> {
        let mut outer = cx.env.outer.clone();
        // Over an aggregate, a subquery also sees its grouping attributes.
        let mut scope = cx.scope.to_vec();
        if cx.agg_scope.is_some() {
            scope.extend(cx.extra.unwrap_or_default().iter().cloned());
        }
        outer.push(super::super::resolve::Level {
            scope: scope.into(),
        });
        self.env(cx.env.ctes.clone(), cx.env.ctes_id, outer)
    }

    /// The field `name` of the struct `b`, as `b.name` or `b['name']`.
    pub(super) fn struct_field(
        &self,
        e: &Expr,
        b: &Bound,
        name: &str,
    ) -> Result<(Arc<Expr>, ArgType), AnalysisError> {
        let st = match &b.ty.data_type {
            Some(DataType::Struct(st)) => st.clone(),
            _ => {
                let other =
                    b.ty.data_type
                        .as_ref()
                        .map_or_else(|| "VOID".to_owned(), DataType::sql);
                return Err(AnalysisError::new(
                    InvalidExtractBaseFieldType {
                        base: format!("\"{}\"", pretty(&b.shown(), self.ctx.conf.session_zone)),
                        other: format!("\"{other}\""),
                    },
                    e.origin,
                ));
            }
        };
        let cs = self.ctx.conf.case_sensitive;
        let Some((ordinal, field)) = st
            .fields()
            .iter()
            .enumerate()
            .find(|(_, f)| same(f.name(), name, cs))
        else {
            return Err(unresolved_field(
                &[
                    pretty(&b.shown(), self.ctx.conf.session_zone),
                    name.to_owned(),
                ],
                &st,
                e.origin,
            ));
        };
        let ty = arg_type(field.data_type().clone(), field.nullable() || b.ty.nullable);
        let expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Field {
                base: Arc::clone(&b.expr),
                ordinal,
                name: name.to_owned(),
                data_type: ty.data_type.clone().unwrap_or(DataType::Null),
                nullable: ty.nullable,
            }))
            .with_origin(e.origin),
        );
        Ok((expr, ty))
    }
}
