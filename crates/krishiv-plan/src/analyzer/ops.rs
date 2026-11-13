// SPDX-License-Identifier: Apache-2.0
//! The relational operators of the Resolve pass and their rewrites.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::convert::Infallible;
use std::sync::Arc;

use krishiv_common::error::{
    AmbiguousReference, AssignmentArityMismatch, DatatypeMismatchFilterNotBoolean,
    DatatypeMismatchInvalidOrderingType, FailedToParseTooComplex, GroupByAggregate,
    GroupByPosAggregate, GroupByPosOutOfRange, InvalidLimitLikeExpressionDataType,
    InvalidNonDeterministicExpressions, InvalidWhereCondition, JoinConditionIsNotBooleanType,
    MissingAggregation, MissingGroupBy, NumTableValueAliasesMismatch, OrderByPosOutOfRange,
    QualifyAggregateNotAllowed, QualifyRequiresWindowFunction, StarGroupByPos,
    SumOfLimitAndOffsetExceedsMaxInt, TableOrViewNotFound, UdtfAliasNumberMismatch,
    UnresolvedUsingColumnForJoin, UnsupportedExprForOperator,
    UnsupportedFeatureLateralColumnAliasInAggregateFunc,
    UnsupportedFeatureLateralColumnAliasInGroupBy, UnsupportedGeneratorNestedInExpressions,
    UnsupportedSubqueryExpressionCategoryAggregateFunctionMixedOuterLocalReferences,
    UnsupportedSubqueryExpressionCategoryLateralJoinConditionNonDeterministic,
    UnsupportedSubqueryExpressionCategoryNonDeterministicLateralSubqueries,
    UnsupportedSubqueryExpressionCategoryUnsupportedInExistsSubquery,
};

use super::analytics::Analytics;
use super::bind::{BoundTable, same, unsupported};
use super::catalog::quoted;
use super::coerce::quoted_type;
use super::exprs::{
    Bound, Cx, WindowUse, names, pretty, semantic_key, synth_call, synth_call_of, unresolved,
};
use super::resolve::{AggInfo, Attr, Env, Rel, Resolver, Widen};
use super::{AnalysisError, ArgType, FileMetadata, TableKind};
use crate::{
    AttrId, BoundColumn, CteRef, Expr, ExprNode, FunctionCall, FunctionKind, Grouping,
    JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PlanNode, ReadSource, ResolvedExpr,
    ResolvedFunction, Scan, SortOrder,
};
use krishiv_types::DataType;

/// What [`Resolver::group_substitute`] reads and grows: the aggregate, the
/// attributes it shows, and the output it adds for grouping expressions the
/// select list does not name.
pub(super) struct Grouped<'a> {
    pub(super) info: &'a AggInfo,
    pub(super) env: &'a Arc<Env>,
    pub(super) op: &'static str,
    pub(super) visible: &'a HashSet<AttrId>,
    pub(super) memo: &'a mut HashMap<usize, u128>,
    pub(super) known: &'a mut Vec<(u128, Attr)>,
    pub(super) grown: &'a mut AggInfo,
    pub(super) pass: &'a mut Vec<Attr>,
}

/// The two attributes of a `USING` column.
type Keys = (Attr, Attr);

/// The largest attribute id: ids are numbered densely from 0, and a
/// self-join instance reserves its subtree's span; past this (about 2^62),
/// a plan is too complex.
pub(super) const MAX_ID: u64 = 1 << 62;

pub(super) fn col_expr(c: &BoundColumn, origin: Origin) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(c.clone()))).with_origin(origin))
}

pub(super) fn column_bound(a: &Attr, origin: Origin) -> Bound {
    Bound {
        expr: col_expr(&a.col, origin),
        ty: ArgType {
            data_type: Some(a.col.data_type.clone()),
            nullable: a.col.nullable,
        },
        agg: None,
        win: None,
        outer: None,
        lateral: Vec::new(),
        shown: None,
    }
}

pub(super) fn no_qual() -> Arc<[String]> {
    Arc::from(Vec::new())
}

impl Resolver<'_> {
    /// A table, view or CTE by name: CTEs shadow tables, inner ones outer.
    pub(super) fn read(
        &mut self,
        p: &LogicalPlan,
        name: &[String],
        options: &BTreeMap<String, krishiv_common::secret::OptionValue>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        if let [n] = name
            && let Some((cte, def)) = env.ctes.iter().rev().find(|(c, _)| same(c, n, cs))
        {
            let output: Vec<BoundColumn> = def
                .out
                .iter()
                .map(|a| BoundColumn {
                    id: self.fresh(),
                    ..a.col.clone()
                })
                .collect();
            // Each reference to a CTE holds the declared `CHAR`/`VARCHAR` types of its definition.
            for (a, c) in def.out.iter().zip(&output) {
                self.carry_column_meta(a.col.id, c.id);
            }
            let qual: Arc<[String]> = Arc::from(vec![n.clone()]);
            let out = output
                .iter()
                .zip(def.out.iter())
                .map(|(c, a)| Attr {
                    col: c.clone(),
                    qual: Arc::clone(&qual),
                    source: None,
                    // Spark's `CTERelationRef` keeps the metadata.
                    duplicate: a.duplicate,
                })
                .collect();
            let node = PlanNode::CteRef(CteRef {
                name: cte.clone(),
                source: def.out.iter().map(|a| a.col.id).collect(),
                output,
            });
            return Ok(self.make(p.origin, node, out, &[]));
        }
        if let Some(&BoundTable::Files(i)) = self.b.tables.get(name) {
            return Ok(self.files(p, i));
        }
        let Some(BoundTable::Found { meta, securable }) = self.b.tables.get(name) else {
            return Err(AnalysisError::new(
                TableOrViewNotFound {
                    relation_name: quoted(name),
                    search_path: self.b.search_path.clone(),
                },
                p.origin,
            ));
        };
        if !matches!(meta.kind, TableKind::Table) {
            return Err(unsupported("views", p.origin));
        }
        let parts = meta.ident.parts();
        let qual: Arc<[String]> = Arc::from(parts.clone());
        let mut out = Vec::new();
        for c in &meta.columns {
            let col = BoundColumn {
                id: self.fresh(),
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
            };
            if let Some(declared) = &c.char_varchar {
                self.declared.insert(col.id, Arc::from(declared.as_str()));
            }
            if let Some(comment) = meta.column_comments.get(&c.name) {
                self.comments.insert(col.id, Arc::from(comment.as_str()));
            }
            out.push(Attr {
                col,
                qual: Arc::clone(&qual),
                source: Some((*securable, Arc::from(c.name.as_str()))),
                duplicate: false,
            });
        }
        let node = PlanNode::Scan(Scan {
            source: ReadSource::Table {
                name: parts,
                options: options.clone(),
            },
            output: out.iter().map(|a| a.col.clone()).collect(),
            version: meta.version,
        });
        let mut rel = self.make(p.origin, node, out, &[]);
        self.with_metadata_column(&mut rel, meta.file_metadata, &qual);
        self.pad_char_columns(rel, p.origin)
    }

    /// `rel`, a scan of a file source, with the hidden `_metadata` column of
    /// Spark's file sources: a struct a query can read by name, left out of
    /// `*`, that the scan outputs only when a query reads it.
    pub(super) fn with_metadata_column(
        &mut self,
        rel: &mut Rel,
        metadata: FileMetadata,
        qual: &Arc<[String]>,
    ) {
        // A table column of the name hides it.
        if metadata == FileMetadata::None
            || rel
                .out
                .iter()
                .any(|a| a.col.name.eq_ignore_ascii_case(METADATA_COLUMN))
        {
            return;
        }
        let attr = Attr {
            col: BoundColumn {
                id: self.fresh(),
                name: METADATA_COLUMN.to_owned(),
                data_type: metadata_type(metadata == FileMetadata::WithRowIndex),
                nullable: false,
            },
            qual: Arc::clone(qual),
            source: None,
            duplicate: false,
        };
        rel.hidden = Arc::from(vec![attr]);
        rel.widen = Some(Arc::new(Widen::Scan));
    }

    /// The files of [`super::bind::Bindings::files`] entry `i`, read by
    /// `p`: a scan without a qualifier (Spark names neither the path nor
    /// the format), with fresh attributes.
    pub(super) fn files(&mut self, p: &LogicalPlan, i: usize) -> Rel {
        let bound = &self.b.files[i];
        let out: Vec<Attr> = bound
            .columns
            .iter()
            .map(|c| Attr {
                col: BoundColumn {
                    id: self.fresh(),
                    name: c.name.clone(),
                    data_type: c.data_type.clone(),
                    nullable: c.nullable,
                },
                qual: no_qual(),
                source: None,
                duplicate: false,
            })
            .collect();
        let node = PlanNode::Scan(Scan {
            source: bound.read.clone(),
            output: out.iter().map(|a| a.col.clone()).collect(),
            version: None,
        });
        let metadata = match &bound.read {
            ReadSource::Files { format, .. } => file_metadata(format),
            _ => FileMetadata::None,
        };
        let mut rel = self.make(p.origin, node, out, &[]);
        self.with_metadata_column(&mut rel, metadata, &no_qual());
        rel
    }

    /// `c` as an instance: every id inside shifted by `offset`; `plan` is
    /// the `Shift` node when it exists already. Ids past `MAX_ID` are an
    /// error, never an overflow.
    pub(super) fn shifted(
        &mut self,
        c: &Rel,
        offset: u64,
        plan: Option<Arc<LogicalPlan>>,
    ) -> Result<Rel, AnalysisError> {
        let (_, hi) = c.ids.unwrap_or((0, 0));
        if hi.checked_add(offset).is_none_or(|top| top > MAX_ID) {
            return Err(AnalysisError::new(
                FailedToParseTooComplex {},
                c.plan.origin,
            ));
        }
        // The declared `CHAR`/`VARCHAR` types follow the shifted ids.
        for a in c.out.iter().chain(c.hidden.iter()) {
            self.carry_column_meta(a.col.id, AttrId(a.col.id.0 + offset));
        }
        let sh = |a: &Attr| Attr {
            col: BoundColumn {
                id: AttrId(a.col.id.0 + offset),
                ..a.col.clone()
            },
            ..a.clone()
        };
        let carriers = c
            .carriers
            .iter()
            .map(|(id, outs)| {
                let outs = outs
                    .iter()
                    .map(|o| o.iter().map(sh).collect::<Vec<_>>().into())
                    .collect();
                (*id, outs)
            })
            .collect();
        let plan = plan.unwrap_or_else(|| {
            Arc::new(LogicalPlan::new(PlanNode::Shift {
                input: Arc::clone(&c.plan),
                offset,
            }))
        });
        // Hidden columns are widened in on demand, in the other instance's
        // ids: a file table's `_metadata`, and the keys a `USING` join
        // merged (`t1 JOIN (t2 JOIN t3 USING (a)) USING (a)` reading `t3.a`,
        // where the right side is a second instance of `t1`'s plan).
        let widen = (c.hidden.iter().any(is_generated_metadata)
            || (!c.hidden.is_empty() && c.widen.is_some()))
        .then(|| {
            Arc::new(Widen::Shift {
                inner: c.clone(),
                offset,
            })
        });
        Ok(Rel {
            plan,
            out: c.out.iter().map(sh).collect::<Vec<_>>().into(),
            ids: c.ids.map(|(lo, hi)| (lo + offset, hi + offset)),
            outer: None,
            carriers: Arc::new(carriers),
            agg: None,
            widen,
            hidden: c.hidden.iter().map(sh).collect::<Vec<_>>().into(),
            generated_from: None,
        })
    }

    /// A fresh instance of `c`, for the second side of a self-join.
    pub(super) fn instance(&mut self, c: &Rel) -> Result<Rel, AnalysisError> {
        let (lo, hi) = c.ids.unwrap_or((0, 0));
        // `next_id` is past every id of the input plan, so the instance's
        // ids are fresh.
        let offset = self.next_id.saturating_sub(lo);
        let r = self.shifted(c, offset, None)?;
        self.next_id += hi - lo + 1;
        Ok(r)
    }

    pub(super) fn project(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        self.project_items(p, &c, exprs, env)
    }

    pub(super) fn project_items(
        &mut self,
        p: &LogicalPlan,
        c: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let items = self.select_items(c, exprs, env, "Aggregate")?;
        // The declared `CHAR`/`VARCHAR` types follow the aliases whichever projection is built below.
        self.carry_declared(&items, exprs);
        let c = &self.reveal(c, &items)?;
        if items.iter().any(|b| has_generator(&b.expr)) {
            return self.generator_project(p.origin, c, exprs, &items);
        }
        if items.iter().any(|b| b.agg.is_some()) {
            return self.build_aggregate(p.origin, c, Vec::new(), &items, true, None, env);
        }
        if items.iter().any(|b| b.win.is_some()) {
            return self.window_project(p.origin, c, &items, &[]);
        }
        Ok(self.lateral_project(p.origin, c, &items, &[]))
    }

    /// `c` outputting the merged-away `USING` columns (`c.hidden`) the
    /// bound expressions read, so an operator above can read them; `c`
    /// itself when they read none.
    pub(super) fn reveal<'a>(
        &mut self,
        c: &Rel,
        bounds: impl IntoIterator<Item = &'a Bound>,
    ) -> Result<Rel, AnalysisError> {
        self.reveal_exprs(c, bounds.into_iter().map(|b| &b.expr))
    }

    /// [`Self::reveal`] for the columns `exprs` read.
    pub(super) fn reveal_exprs<'a>(
        &mut self,
        c: &Rel,
        exprs: impl IntoIterator<Item = &'a Arc<Expr>>,
    ) -> Result<Rel, AnalysisError> {
        if c.hidden.is_empty() {
            return Ok(c.clone());
        }
        let mut missing: Vec<Attr> = Vec::new();
        let mut seen = HashSet::new();
        let mut stack: Vec<Arc<Expr>> = exprs.into_iter().map(Arc::clone).collect();
        while let Some(x) = stack.pop() {
            if !seen.insert(super::bind::addr(&*x)) {
                continue;
            }
            if let ExprNode::Resolved(ResolvedExpr::Column(col)) = &x.node
                && !c.out.iter().any(|a| a.col.id == col.id)
                && !missing.iter().any(|m| m.col.id == col.id)
                && let Some(a) = c.hidden.iter().find(|a| a.col.id == col.id)
            {
                missing.push(a.clone());
            }
            x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
        }
        if missing.is_empty() {
            return Ok(c.clone());
        }
        let mut wide = self.widened(c, &missing)?;
        wide.hidden = Arc::clone(&c.hidden);
        Ok(wide)
    }

    /// A select list holding a generator call (today only `json_tuple`):
    /// Spark's `ExtractGenerator` for the one generator this engine
    /// implements. At most one top-level generator per select list; a
    /// generator nested in another expression is
    /// `UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS`, two are
    /// `UNSUPPORTED_GENERATOR.MULTI_GENERATOR`. The generator computes one
    /// struct per row in a projection below the select list, and the select
    /// list reads its fields (so the JSON is parsed once per row, not once
    /// per field). No generator node remains in the resolved plan.
    fn generator_project(
        &mut self,
        origin: Origin,
        base: &Rel,
        exprs: &[Arc<Expr>],
        items: &[Bound],
    ) -> Result<Rel, AnalysisError> {
        let zone = self.ctx.conf.session_zone;
        let mut top: Vec<(usize, String)> = Vec::new();
        for (i, b) in items.iter().enumerate() {
            let inner = trim_alias(&b.expr);
            if is_generator(inner) {
                top.push((i, pretty(inner, zone)));
                continue;
            }
            if has_generator(inner) {
                return Err(AnalysisError::new(
                    UnsupportedGeneratorNestedInExpressions {
                        expression: format!("\"{}\"", pretty(inner, zone)),
                    },
                    b.expr.origin,
                ));
            }
        }
        // Spark 4.2 accepts several generators in one plain select list (a
        // cross product; `MULTI_GENERATOR` is raised only beside
        // aggregates), which the one-struct rewrite below cannot express.
        if top.len() > 1 {
            return Err(unsupported("several generators in one SELECT list", origin));
        }
        let Some((index, _)) = top.first() else {
            return Err(unsupported("a generator", origin));
        };
        let index = *index;
        if items.iter().any(|b| b.agg.is_some() || b.win.is_some()) {
            return Err(unsupported(
                "a generator with aggregate or window expressions in one SELECT list",
                origin,
            ));
        }
        if items.iter().any(|b| !b.lateral.is_empty()) {
            return Err(unsupported(
                "a generator with lateral column aliases in one SELECT list",
                origin,
            ));
        }
        let ExprNode::Resolved(ResolvedExpr::Function(call)) = &trim_alias(&items[index].expr).node
        else {
            return Err(unsupported("a generator", origin));
        };
        let outputs = call.args.len() - 1;
        // An explicit single alias renames the one output column; several
        // outputs under one name do not match, as Spark's `ResolveGenerate`
        // reports them. Stars expand the item list, so the alias is matched
        // by call name rather than by position (as the analyzer matches
        // `struct` and `named_struct` by name elsewhere).
        // `|> SET x = json_tuple(..)` names the output by its target.
        let alias = exprs.iter().find_map(|e| match &e.node {
            ExprNode::Alias { name, expr } if is_json_tuple_call(expr) => Some(name.clone()),
            ExprNode::Star(s) => s
                .replace
                .iter()
                .find_map(|(name, expr)| is_json_tuple_call(expr).then(|| name.clone())),
            _ => None,
        });
        if let Some(name) = &alias
            && outputs != 1
        {
            return Err(AnalysisError::new(
                UdtfAliasNumberMismatch {
                    aliases_size: outputs.to_string(),
                    aliases_names: name.clone(),
                },
                items[index].expr.origin,
            ));
        }
        // The generator computes one struct per row under a fresh
        // attribute; the select list reads one field per output column.
        // The computation is an ordinary scalar call of the same kernel:
        // the rewrite leaves no generator node in the resolved plan, so
        // the executor, the optimizer and the plan codec all see plain
        // scalar expressions.
        if !matches!(&call.data_type, DataType::Struct(_)) {
            return Err(unsupported(
                format!("a generator of {}", call.data_type),
                items[index].expr.origin,
            ));
        }
        let data_type = call.data_type.clone();
        let compute_call = ResolvedFunction {
            kind: FunctionKind::Scalar,
            ..call.clone()
        };
        let compute_expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(compute_call)))
                .with_origin(items[index].expr.origin),
        );
        let compute_ty = ArgType {
            data_type: Some(data_type),
            nullable: call.nullable,
        };
        let compute = self.attr("json_tuple", &compute_ty);
        let inner_expr = Arc::new(
            Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                expr: compute_expr,
                output: compute.clone(),
            }))
            .with_origin(items[index].expr.origin),
        );
        let mut inner_exprs: Vec<Arc<Expr>> = base
            .out
            .iter()
            .map(|a| col_expr(&a.col, Origin::UNKNOWN))
            .collect();
        inner_exprs.push(inner_expr);
        let mut inner_out = base.out.to_vec();
        inner_out.push(Attr {
            col: compute.clone(),
            qual: no_qual(),
            source: None,
            duplicate: false,
        });
        let inner = self.projection(Origin::UNKNOWN, base, inner_exprs, inner_out);
        let mut outer_exprs = Vec::with_capacity(items.len() + outputs - 1);
        let mut outer_out = Vec::with_capacity(items.len() + outputs - 1);
        let compute_ref = col_expr(&compute, items[index].expr.origin);
        for (i, b) in items.iter().enumerate() {
            if i != index {
                outer_exprs.push(Arc::clone(&b.expr));
                outer_out.push(out_attr(b, &inner.out));
                continue;
            }
            for n in 0..outputs {
                let name = match &alias {
                    Some(name) => name.clone(),
                    None => format!("c{n}"),
                };
                let output = self.attr(
                    &name,
                    &ArgType {
                        data_type: Some(DataType::String),
                        nullable: true,
                    },
                );
                let field = Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Field {
                        base: Arc::clone(&compute_ref),
                        ordinal: n,
                        name: format!("c{n}"),
                        data_type: DataType::String,
                        nullable: true,
                    }))
                    .with_origin(b.expr.origin),
                );
                outer_exprs.push(Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: field,
                        output: output.clone(),
                    }))
                    .with_origin(b.expr.origin),
                ));
                outer_out.push(Attr {
                    col: output,
                    qual: no_qual(),
                    source: None,
                    duplicate: false,
                });
            }
        }
        Ok(self.projection(origin, &inner, outer_exprs, outer_out))
    }

    /// A select list: stars expanded, every item named. A name the input
    /// lacks may be an alias defined earlier in the list (a lateral column
    /// alias), when `spark.sql.lateralColumnAlias.enableImplicitResolution`.
    pub(super) fn select_items(
        &mut self,
        c: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
    ) -> Result<Vec<Bound>, AnalysisError> {
        self.select_items_in(c, exprs, env, op, None)
    }

    /// [`Self::select_items`] of an aggregate over grouping sets, which
    /// may call `grouping` and `grouping_id` and name `grouping__id`
    /// (`extra`, searched after the input).
    pub(super) fn select_items_in(
        &mut self,
        c: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
        extra: Option<&[Attr]>,
    ) -> Result<Vec<Bound>, AnalysisError> {
        let rels = [c];
        let mut items = Vec::new();
        let mut lateral: Vec<Attr> = Vec::new();
        let lca = self.ctx.conf.lateral_column_alias;
        for e in exprs {
            if let ExprNode::Star(s) = &e.node {
                let cx = Self::cx(&c.out, &rels, env, op);
                // `* EXCEPT (s.f)` and `s.* EXCEPT (...)` rebuild the structs.
                if let Some(nested) = self.nested_except_items(s, &cx, e.origin)? {
                    for item in nested {
                        let mut cx = Self::cx(&c.out, &rels, env, op);
                        cx.lateral = lca.then_some(&lateral[..]);
                        cx.windows = WindowUse::Allowed;
                        let bound = self.bind(&item, &cx)?;
                        if let ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) =
                            &bound.expr.node
                        {
                            lateral.push(Attr {
                                col: output.clone(),
                                qual: no_qual(),
                                source: None,
                                duplicate: false,
                            });
                        }
                        items.push(bound);
                    }
                    continue;
                }
                // `s.*` over a struct column: its fields.
                if let Some(fields) = self.star_struct_fields(s, &cx, e.origin)? {
                    for field in fields {
                        let mut name = s.qualifier.clone();
                        name.push(field.clone());
                        let item = Arc::new(
                            Expr::new(ExprNode::Alias {
                                expr: Arc::new(
                                    Expr::new(ExprNode::Column(crate::ColumnRef::new(name)))
                                        .with_origin(e.origin),
                                ),
                                name: field,
                            })
                            .with_origin(e.origin),
                        );
                        let mut cx = Self::cx(&c.out, &rels, env, op);
                        cx.lateral = lca.then_some(&lateral[..]);
                        cx.windows = WindowUse::Allowed;
                        let bound = self.bind(&item, &cx)?;
                        if let ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) =
                            &bound.expr.node
                        {
                            lateral.push(Attr {
                                col: output.clone(),
                                qual: no_qual(),
                                source: None,
                                duplicate: false,
                            });
                        }
                        items.push(bound);
                    }
                    continue;
                }
                // `t.*` in a subquery where `t` is the enclosing query's: after
                // the local forms above (a local struct column of that name wins).
                if let Some(outer) = self.outer_star(s, &cx, e.origin)? {
                    for b in outer {
                        items.push(self.named(b, true)?);
                    }
                    continue;
                }
                let expanded = self.expand(s, &cx, e.origin)?;
                let replaced = self.replacements(s, &expanded, e.origin)?;
                for (a, with) in expanded.iter().zip(replaced) {
                    let Some(with) = with else {
                        items.push(column_bound(a, e.origin));
                        continue;
                    };
                    // `|> SET name = expr`: the expression takes the
                    // column's place and name.
                    let item = Arc::new(
                        Expr::new(ExprNode::Alias {
                            expr: with,
                            name: a.col.name.clone(),
                        })
                        .with_origin(e.origin),
                    );
                    let mut cx = Self::cx(&c.out, &rels, env, op);
                    cx.lateral = lca.then_some(&lateral[..]);
                    cx.windows = WindowUse::Allowed;
                    let bound = self.bind(&item, &cx)?;
                    if let ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) = &bound.expr.node
                    {
                        lateral.push(Attr {
                            col: output.clone(),
                            qual: no_qual(),
                            source: None,
                            duplicate: false,
                        });
                    }
                    items.push(bound);
                }
                // A generator written with one alias is a lateral alias of the items after it
                // (its columns keep their generator qualifier: see `Resolver::column`).
                if s.qualifier.iter().any(|q| {
                    q.starts_with(crate::GENERATE_QUALIFIER_PREFIX)
                        && q.ends_with(crate::GENERATE_ALIAS_SUFFIX)
                }) {
                    lateral.extend(expanded.iter().cloned());
                }
                continue;
            }
            let b = {
                let mut cx = Self::cx(&c.out, &rels, env, op);
                cx.lateral = lca.then_some(&lateral[..]);
                cx.extra = extra;
                cx.grouping = extra.is_some();
                cx.windows = WindowUse::Allowed;
                self.bind(e, &cx)?
            };
            let written_identifier = matches!(&e.node, ExprNode::Column(c) if c.by_identifier);
            let b = Self::without_internal_identity_cast(self.named(b, !written_identifier)?, e);
            // Before the next item binds: it may read this alias (a lateral column alias).
            self.carry_declared_item(&b, Some(e));
            if let ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) = &b.expr.node {
                lateral.push(Attr {
                    col: output.clone(),
                    qual: no_qual(),
                    source: None,
                    duplicate: false,
                });
            }
            items.push(b);
        }
        Ok(items)
    }

    /// For each column of a star's expansion, the expression of its
    /// `|> SET` assignment (the last one naming it). A name that is not
    /// a column of the expansion is `UNRESOLVED_COLUMN`.
    fn replacements(
        &self,
        s: &crate::Star,
        out: &[Attr],
        origin: Origin,
    ) -> Result<Vec<Option<Arc<Expr>>>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        let mut with: Vec<Option<Arc<Expr>>> = vec![None; out.len()];
        for (name, expr) in &s.replace {
            let mut found = false;
            for (slot, a) in with.iter_mut().zip(out) {
                if super::bind::same(&a.col.name, name, cs) {
                    *slot = Some(Arc::clone(expr));
                    found = true;
                }
            }
            if !found {
                return Err(unresolved(std::slice::from_ref(name), out, origin));
            }
        }
        Ok(with)
    }

    /// `b` as a named expression: an attribute, or aliased by Spark's
    /// generated name (a cast of a column keeps the column's name, a
    /// lateral alias otherwise reads `lateralAliasReference(name)`).
    /// With `bare` false (an item written `IDENTIFIER(..)`), a variable or a field of a dotted
    /// path is named by its full shown form (`variablereference(..)`, `c1.c2`).
    fn named(&mut self, b: Bound, bare: bool) -> Result<Bound, AnalysisError> {
        // A column names itself, unless it stands for a call written in its
        // place (a SQL function whose body is its parameter, `f(c1)`).
        let written_call = b
            .shown
            .as_ref()
            .is_some_and(|s| matches!(s.node, ExprNode::Function(_)));
        match &b.expr.node {
            ExprNode::Resolved(ResolvedExpr::Alias { .. }) => return Ok(b),
            ExprNode::Resolved(ResolvedExpr::Column(_)) if !written_call => return Ok(b),
            _ => {}
        }
        let shown = b.shown();
        let bare = super::exprs::bare_name(&shown).filter(|_| bare);
        let zone = self.ctx.conf.session_zone;
        let name = match &shown.node {
            // A bare variable or a field of a dotted column path keeps its own name.
            _ if bare.is_some() => bare.unwrap_or_default().to_owned(),
            ExprNode::Cast { expr, .. }
                if matches!(expr.node, ExprNode::Resolved(ResolvedExpr::Column(_))) =>
            {
                pretty(expr, zone)
            }
            // A cast of an alias (`v:a::int`) keeps the alias's name.
            ExprNode::Cast { expr, .. }
                if matches!(expr.node, ExprNode::Resolved(ResolvedExpr::Alias { .. })) =>
            {
                match &expr.node {
                    ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => output.name.clone(),
                    _ => pretty(expr, zone),
                }
            }
            // A higher-order call names its output from the bound call:
            // lambdas read `namedlambdavariable()`, and a three-argument
            // `aggregate`/`reduce` shows the identity `finish` (see `hof`
            // and Spark's `toPrettySQL`).
            // (`max_by(x, y, k)` is bound as a `transform` but keeps its own name.)
            _ if !matches!(shown.node, ExprNode::Field { .. })
                && !matches!(&shown.node, ExprNode::Function(c) if matches!(c.name.as_str(), "max_by" | "min_by"))
                && matches!(&b.expr.node, ExprNode::Resolved(ResolvedExpr::Function(f)) if super::hof::Hof::of(f.id.name()).is_some()) =>
            {
                pretty(&lateral_refs(&b.expr, &b.lateral), zone)
            }
            _ => pretty(&lateral_refs(&shown, &b.lateral), zone),
        };
        let output = self.attr(&name, &b.ty);
        let origin = b.expr.origin;
        Ok(Bound {
            expr: Arc::new(
                Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                    expr: b.expr,
                    output,
                }))
                .with_origin(origin),
            ),
            ..b
        })
    }

    /// Select `items` over `base`, then the attributes `pass`. Items using
    /// lateral column aliases are planned as Spark plans them: an alias
    /// another item reads is computed by a projection below the items that
    /// read it (one projection per level of the chain), unless `base`
    /// outputs it already.
    pub(super) fn lateral_project(
        &self,
        origin: Origin,
        base: &Rel,
        items: &[Bound],
        pass: &[Attr],
    ) -> Rel {
        if pass.is_empty() && items.iter().all(|b| b.lateral.is_empty()) {
            let out = items.iter().map(|b| out_attr(b, &base.out)).collect();
            let exprs = items.iter().map(|b| Arc::clone(&b.expr)).collect();
            return self.projection(origin, base, exprs, out);
        }
        let present: HashSet<AttrId> = base.out.iter().map(|a| a.col.id).collect();
        let outs: Vec<Attr> = items.iter().map(|b| out_attr(b, &base.out)).collect();
        let read: HashSet<AttrId> = items.iter().flat_map(|b| b.lateral.clone()).collect();
        // The level an item is computed at: 0 for what `base` outputs, else
        // one above the highest alias it reads.
        let mut level = vec![0_usize; items.len()];
        for (i, b) in items.iter().enumerate() {
            if present.contains(&outs[i].col.id) {
                continue;
            }
            let reads = b.lateral.iter().filter_map(|id| {
                outs[..i]
                    .iter()
                    .rposition(|a| a.col.id == *id)
                    .map(|j| level[j])
            });
            level[i] = 1 + reads.max().unwrap_or(0);
        }
        let below = |i: usize| level[i] > 0 && read.contains(&outs[i].col.id);
        let top = (0..items.len())
            .filter(|&i| below(i))
            .map(|i| level[i])
            .max()
            .unwrap_or(0);
        let mut cur = base.clone();
        for l in 1..=top {
            let mut exprs: Vec<Arc<Expr>> = cur
                .out
                .iter()
                .map(|a| col_expr(&a.col, Origin::UNKNOWN))
                .collect();
            let mut out = cur.out.to_vec();
            for i in (0..items.len()).filter(|&i| level[i] == l && below(i)) {
                exprs.push(Arc::clone(&items[i].expr));
                out.push(outs[i].clone());
            }
            cur = self.projection(Origin::UNKNOWN, &cur, exprs, out);
        }
        let mut exprs = Vec::new();
        for (i, b) in items.iter().enumerate() {
            exprs.push(if below(i) {
                col_expr(&outs[i].col, b.expr.origin)
            } else {
                Arc::clone(&b.expr)
            });
        }
        exprs.extend(pass.iter().map(|a| col_expr(&a.col, Origin::UNKNOWN)));
        let mut out = outs;
        out.extend(pass.iter().cloned());
        self.projection(origin, &cur, exprs, out)
    }

    /// A projection of `exprs` (with output `out`) over `input`.
    pub(super) fn projection(
        &self,
        origin: Origin,
        input: &Rel,
        exprs: Vec<Arc<Expr>>,
        out: Vec<Attr>,
    ) -> Rel {
        let node = PlanNode::Project {
            input: Arc::clone(&input.plan),
            exprs: exprs.clone(),
        };
        let mut r = self.make(origin, node, out, &[input]);
        // Spark's `Project` keeps its child's hidden output: the columns a
        // `USING` join merged away stay reachable by a qualified name
        // through any projection (`|> extend`, `|> select`), up to an
        // alias, an aggregate or a set operation. A file table's
        // `_metadata` is not carried (it is widened from the scan).
        let hidden: Vec<Attr> = input
            .hidden
            .iter()
            .filter(|a| !is_generated_metadata(a) && !r.out.iter().any(|o| o.col.id == a.col.id))
            .cloned()
            .collect();
        if !hidden.is_empty() {
            r.hidden = hidden.into();
        }
        r.widen = Some(Arc::new(Widen::Project {
            input: input.clone(),
            exprs,
        }));
        r
    }

    /// Whether `r` is the projection over a `USING` join that merges its
    /// keys: one of its hidden columns is an output of its input.
    fn merges_using_keys(r: &Rel) -> bool {
        match r.widen.as_deref() {
            Some(Widen::Project { input, .. }) => r.hidden.iter().any(|h| {
                !is_generated_metadata(h) && input.out.iter().any(|o| o.col.id == h.col.id)
            }),
            _ => false,
        }
    }

    pub(super) fn filter(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        predicate: &Arc<Expr>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        // A `QUALIFY` filter: lowering wraps its predicate (see
        // `lower.rs`), so the analyzer tells it apart from `WHERE` and
        // `HAVING`. Only SQL lowers one: a DataFrame filter has a plan id
        // and no span, and never sits directly over its query's own
        // projection the way a `QUALIFY` does.
        if p.origin.sql.is_some()
            && let Some(q) = qualify_expression(predicate)
            && let Some(r) = self.qualify(p, input, &c, q, env)?
        {
            return Ok(r);
        }
        // A `HAVING` over an aggregate with windows filters below them
        // (Spark's `UnresolvedHaving`: it cannot name a window's alias). Only
        // SQL has a `HAVING`: lowering stamps the filter with the span of
        // its text, as every node it builds, and a DataFrame filter has a
        // plan id and no span. A DataFrame filter, and any further filter,
        // filters the output of the windows (checked against Spark 4.2.0:
        // `df.filter(col("r") > 1)` over `... rank() OVER (...) AS r` works,
        // `HAVING r > 1` is UNRESOLVED_COLUMN).
        if let Some(info) = c.agg.clone()
            && info.has_windows()
            && info.having.is_none()
            && p.origin.sql.is_some()
        {
            return self.having_below_windows(p, &info, predicate, env);
        }
        if let Some(r) = self.having_below_generated(p, &c, predicate, env)? {
            return Ok(r);
        }
        if c.agg.as_ref().is_some_and(|i| !i.has_windows()) {
            let (bounds, agg, extended) =
                self.over_agg(&c, std::slice::from_ref(predicate), env, "Filter")?;
            boolean_filter(&bounds[0], p.origin)?;
            let node = PlanNode::Filter {
                input: Arc::clone(&agg.plan),
                predicate: Arc::clone(&bounds[0].expr),
            };
            let mut f = self.make(p.origin, node, agg.out.to_vec(), &[&agg]);
            f.widen = Some(Arc::new(Widen::Through { child: agg }));
            return Ok(if extended {
                self.restore(&f, &c.out)
            } else {
                f
            });
        }
        let (mut bounds, wide) =
            self.bind_missing(&c, std::slice::from_ref(predicate), env, "Filter", false)?;
        let b = bounds.remove(0);
        if let Some(x) = super::having_subquery::mixed_aggregate(&b.expr) {
            return Err(AnalysisError::new(
                UnsupportedSubqueryExpressionCategoryAggregateFunctionMixedOuterLocalReferences {
                    function: super::having_subquery::mixed_sql(&x, &wide.out, env),
                },
                p.origin,
            ));
        }
        if b.agg.is_some() && !super::having_subquery::outer_aggregates_only(&b.expr, env) {
            return Err(AnalysisError::new(
                InvalidWhereCondition {
                    condition: format!("\"{}\"", pretty(&b.shown(), self.ctx.conf.session_zone)),
                    expression_list: super::aggregate_text::aggregate_sql(&b.expr, &wide.out),
                },
                p.origin,
            ));
        }
        boolean_filter(&b, p.origin)?;
        let wide = self.reveal(&wide, std::slice::from_ref(&b))?;
        let node = PlanNode::Filter {
            input: Arc::clone(&wide.plan),
            predicate: b.expr,
        };
        let mut f = self.make(p.origin, node, wide.out.to_vec(), &[&wide]);
        let extended = wide.out.len() > c.out.len();
        f.widen = Some(Arc::new(Widen::Through { child: wide }));
        Ok(if extended {
            let mut kept = self.restore(&f, &c.out);
            kept.hidden = Arc::clone(&c.hidden);
            kept
        } else {
            f
        })
    }

    /// A projection of `out` over `r`: drops what a rewrite added.
    pub(super) fn restore(&self, r: &Rel, out: &[Attr]) -> Rel {
        let exprs = out
            .iter()
            .map(|a| col_expr(&a.col, Origin::UNKNOWN))
            .collect();
        self.projection(Origin::UNKNOWN, r, exprs, out.to_vec())
    }

    /// Binds the top-level expressions of a filter or sort over `c`. Names
    /// `c` does not output, but a relation below it dropped (through
    /// projections, filters, sorts and limits, down to an aggregate's
    /// grouping attributes), are Spark's missing references: they are
    /// added back up to `c` and resolve. Enclosing queries come last, as
    /// in Spark: a dropped column hides an outer one of its name. Returns
    /// the bound expressions and the relation they read, wider than `c`
    /// when something was added. `lenient`: a name `c` outputs several
    /// times is looked up below (and, when the below holds no such name
    /// either, stays ambiguous rather than unresolved).
    pub(super) fn bind_missing(
        &mut self,
        c: &Rel,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
        lenient: bool,
    ) -> Result<(Vec<Bound>, Rel), AnalysisError> {
        let rels = [c];
        let mut cx = Self::cx(&c.out, &rels, env, op);
        cx.lenient = lenient;
        if op == "Filter" {
            cx.windows = WindowUse::Where;
        }
        // Without the enclosing queries when there is something below.
        let below = Self::below(c);
        // Names resolve among the nearest relations' columns first; the
        // columns a `USING` join merged away answer to qualified names
        // only after that (a bare name never meets them).
        let lookup = Self::below_of(c, true, false);
        let deep = Self::below_of(c, true, true);
        cx.outer = below.is_empty();
        let unresolved = |e: &AnalysisError| {
            e.error
                .error_class()
                .name()
                .starts_with("UNRESOLVED_COLUMN")
        };
        // With leniency, duplicates in the output are looked up below it:
        // the first try (without the below) reports them as ambiguous
        // rather than unresolved, so retry on either when there is a below.
        let ambiguous = |e: &AnalysisError| {
            lenient
                && e.error
                    .error_class()
                    .name()
                    .starts_with("AMBIGUOUS_REFERENCE")
        };
        // `ORDER BY grouping(a)` over an aggregate's windows reads the
        // grouping id, which goes up through them like a dropped key (Spark
        // resolves these calls before it extracts the windows).
        let analytics = c
            .agg
            .as_ref()
            .filter(|i| op == "Sort" && i.has_windows())
            .and_then(|i| i.analytics.clone());
        cx.grouping = analytics.is_some();
        let first = match self.bind_all(exprs, &cx) {
            Ok(bounds) if analytics.is_none() => return Ok((bounds, c.clone())),
            Ok(bounds) => Some(bounds),
            Err(e) if (!unresolved(&e) && !ambiguous(&e)) || lookup.is_empty() => return Err(e),
            Err(_) => None,
        };
        let mut bounds = match first {
            Some(bounds) => bounds,
            None => {
                cx.extra = Some(&lookup);
                let mut tried = self.bind_all(exprs, &cx);
                if tried.as_ref().is_err_and(unresolved) && deep.len() > lookup.len() {
                    cx.extra = Some(&deep);
                    tried = self.bind_all(exprs, &cx);
                }
                match tried {
                    Err(e) if unresolved(&e) => {
                        cx.outer = true;
                        self.bind_all(exprs, &cx)?
                    }
                    other => other?,
                }
            }
        };
        if let Some(an) = &analytics {
            for b in &mut bounds {
                an.check_late_call(&b.expr)?;
                b.expr = self.resolve_grouping(&b.expr, an, env)?;
            }
        }
        let have: HashSet<AttrId> = c.out.iter().map(|a| a.col.id).collect();
        let mut missing: Vec<Attr> = Vec::new();
        let mut seen = HashSet::new();
        for b in &bounds {
            let mut stack = vec![Arc::clone(&b.expr)];
            while let Some(x) = stack.pop() {
                if !seen.insert(super::bind::addr(&*x)) {
                    continue;
                }
                if let ExprNode::Resolved(ResolvedExpr::Column(col)) = &x.node
                    && !have.contains(&col.id)
                    && !missing.iter().any(|m| m.col.id == col.id)
                    && let Some(a) = below.iter().find(|a| a.col.id == col.id)
                {
                    missing.push(a.clone());
                }
                x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
            }
        }
        // The retry resolved a name only `below` holds, or a grouping call
        // reads the id; with neither there is nothing to widen.
        let wide = if missing.is_empty() {
            c.clone()
        } else {
            self.widened(c, &missing)?
        };
        Ok((bounds, wide))
    }

    /// The aggregate under `c`'s filters, sorts and limits (`HAVING`), and
    /// those layers, top first; `None` when `c` is not such a stack. A
    /// `HAVING` that had to grow the aggregate left a projection over its
    /// filter (`restore`), and that is not such a stack: Spark does not
    /// resolve a sort through it (`GROUP BY x + 1 HAVING sum(l) > 1 ORDER BY
    /// x + 1` is UNRESOLVED_COLUMN, checked against 4.2.0).
    fn through_to_agg(c: &Rel) -> Option<(Vec<Rel>, Rel)> {
        let mut layers = Vec::new();
        let mut cur = c.clone();
        loop {
            if cur.agg.is_some() {
                return Some((layers, cur));
            }
            // A `DISTRIBUTE BY` between ends the walk: Spark 4.2.0 resolves
            // an aggregate call in a sort only directly over the aggregate
            // or its `HAVING`, so `GROUP BY y [HAVING ...] DISTRIBUTE BY y
            // SORT BY max(x)` is UNRESOLVED_COLUMN there.
            if matches!(cur.plan.node, PlanNode::Repartition { .. }) {
                return None;
            }
            let child = match cur.widen.as_deref() {
                Some(Widen::Through { child }) => child.clone(),
                _ => return None,
            };
            layers.push(cur);
            cur = child;
        }
    }

    /// `layers` (from [`Self::through_to_agg`]) rebuilt over `inner`, a
    /// grown version of the aggregate they were over.
    fn rewrapped(&self, layers: &[Rel], inner: Rel) -> Rel {
        let mut built = inner;
        for rel in layers.iter().rev() {
            let child = built;
            let plan = Arc::clone(&rel.plan)
                .map_children(|_| Ok::<_, Infallible>(Arc::clone(&child.plan)))
                .unwrap_or_else(|never| match never {});
            let mut wide = self.wrap(plan, rel.plan.origin.plan_id, child.out.to_vec(), &[&child]);
            wide.widen = Some(Arc::new(Widen::Through { child }));
            built = wide;
        }
        built
    }

    /// Binds each of `exprs` in `cx`.
    fn bind_all(&mut self, exprs: &[Arc<Expr>], cx: &Cx<'_>) -> Result<Vec<Bound>, AnalysisError> {
        exprs.iter().map(|e| self.bind(e, cx)).collect()
    }

    /// The attributes relations below `r` dropped and could add back
    /// (see [`Self::bind_missing`]), nearest first.
    fn below(r: &Rel) -> Vec<Attr> {
        Self::below_of(r, false, true)
    }

    /// [`Self::below`]; with `keep_output`, attributes `r` already outputs
    /// are kept too (Spark looks a name the select list outputs twice up
    /// among them: `SELECT id, CAST(id AS STRING) AS id ... ORDER BY id`
    /// sorts by the first).
    fn below_of(r: &Rel, keep_output: bool, with_hidden: bool) -> Vec<Attr> {
        let mut out: Vec<Attr> = Vec::new();
        let mut seen: HashSet<AttrId> = if keep_output {
            HashSet::new()
        } else {
            r.out.iter().map(|a| a.col.id).collect()
        };
        let mut add = |a: &Attr, out: &mut Vec<Attr>| {
            if seen.insert(a.col.id) {
                out.push(a.clone());
            }
        };
        let mut cur = r.clone();
        while let Some(w) = cur.widen.clone() {
            // A `USING` join's own projection: what it merged away is
            // reached only with `with_hidden`. (Projections above it carry
            // the merged columns too, but only this one drops them.)
            if !with_hidden && Self::merges_using_keys(&cur) && !Arc::ptr_eq(&cur.plan, &r.plan) {
                break;
            }
            cur = match &*w {
                Widen::Aggregate(info) => {
                    info.keys.iter().for_each(|k| add(k, &mut out));
                    break;
                }
                Widen::Scan | Widen::Join { .. } | Widen::Shift { .. } => break,
                // Past a pipe boundary only the hidden columns come back.
                Widen::Hidden { child } => {
                    if with_hidden {
                        child.hidden.iter().for_each(|a| add(a, &mut out));
                    }
                    break;
                }
                Widen::Through { child } => child.clone(),
                Widen::Project { input, .. } => {
                    input.out.iter().for_each(|a| add(a, &mut out));
                    if with_hidden {
                        input.hidden.iter().for_each(|a| add(a, &mut out));
                    }
                    input.clone()
                }
            };
        }
        out
    }

    /// `r` outputting `missing` too (attributes [`Self::below`] found),
    /// each relation on the way down adding what the ones above need.
    fn widened(&mut self, r: &Rel, missing: &[Attr]) -> Result<Rel, AnalysisError> {
        // Down: each relation, how it widens and what it must add.
        let mut path: Vec<(Rel, Arc<Widen>, Vec<Attr>)> = Vec::new();
        let mut cur = r.clone();
        let mut need = missing.to_vec();
        while let Some(w) = cur.widen.clone() {
            let here = need.clone();
            let next = match &*w {
                Widen::Aggregate(_) | Widen::Scan | Widen::Join { .. } | Widen::Shift { .. } => {
                    None
                }
                Widen::Through { child } | Widen::Hidden { child } => Some(child.clone()),
                Widen::Project { input, .. } => {
                    need.retain(|a| !input.out.iter().any(|o| o.col.id == a.col.id));
                    (!need.is_empty()).then(|| input.clone())
                }
            };
            path.push((cur, w, here));
            match next {
                Some(next) => cur = next,
                None => break,
            }
        }
        // Up: rebuild each over the wider relation below it.
        let mut built: Option<Rel> = None;
        while let Some((rel, w, need)) = path.pop() {
            let mut out = rel.out.to_vec();
            out.extend(need.iter().cloned());
            built = Some(match &*w {
                Widen::Aggregate(info) => {
                    let mut grown = (**info).clone();
                    for a in &need {
                        if !grown.out.iter().any(|o| o.col.id == a.col.id) {
                            grown.aggregates.push(col_expr(&a.col, Origin::UNKNOWN));
                            grown.out.push(a.clone());
                        }
                    }
                    self.aggregate_rel(rel.plan.origin, grown, &need)?
                }
                Widen::Project { input, exprs } => {
                    let input = built.take().unwrap_or_else(|| input.clone());
                    let mut exprs = exprs.clone();
                    exprs.extend(need.iter().map(|a| col_expr(&a.col, Origin::UNKNOWN)));
                    self.projection(rel.plan.origin, &input, exprs, out)
                }
                // The same plan as its child: the widened child, still
                // letting only hidden columns cross.
                Widen::Hidden { child } => {
                    let below = built.take().unwrap_or_else(|| child.clone());
                    if Arc::ptr_eq(&rel.plan, &child.plan) {
                        let mut wide = below.clone();
                        wide.widen = Some(Arc::new(Widen::Hidden { child: below }));
                        wide.agg = None;
                        wide.generated_from = None;
                        wide
                    } else {
                        // A window over `child`: its input's columns, now
                        // wider, then the columns it adds.
                        let plan = Arc::clone(&rel.plan)
                            .map_children(|_| Ok::<_, Infallible>(Arc::clone(&below.plan)))
                            .unwrap_or_else(|never| match never {});
                        let mut out = below.out.to_vec();
                        out.extend(
                            rel.out
                                .iter()
                                .filter(|a| !child.out.iter().any(|c| c.col.id == a.col.id))
                                .cloned(),
                        );
                        let mut wide = self.wrap(plan, rel.plan.origin.plan_id, out, &[&below]);
                        wide.hidden = Arc::clone(&rel.hidden);
                        wide.widen = Some(Arc::new(Widen::Hidden { child: below }));
                        wide
                    }
                }
                Widen::Through { child } => {
                    let child = built.take().unwrap_or_else(|| child.clone());
                    let plan = Arc::clone(&rel.plan)
                        .map_children(|_| Ok::<_, Infallible>(Arc::clone(&child.plan)))
                        .unwrap_or_else(|never| match never {});
                    let mut wide = self.wrap(plan, rel.plan.origin.plan_id, out, &[&child]);
                    wide.widen = Some(Arc::new(Widen::Through { child }));
                    wide
                }
                Widen::Join {
                    left,
                    right,
                    kind,
                    condition,
                    nullable: (ln, rn),
                } => {
                    let side = |rel: &Rel| -> Vec<Attr> {
                        need.iter()
                            .filter(|a| {
                                rel.hidden.iter().any(|h| h.col.id == a.col.id)
                                    && !rel.out.iter().any(|o| o.col.id == a.col.id)
                            })
                            .cloned()
                            .collect()
                    };
                    let (nl, nr) = (side(left), side(right));
                    let wl = if nl.is_empty() {
                        left.clone()
                    } else {
                        let mut w = self.widened(left, &nl)?;
                        w.hidden = Arc::clone(&left.hidden);
                        w
                    };
                    let wr = if nr.is_empty() {
                        right.clone()
                    } else {
                        let mut w = self.widened(right, &nr)?;
                        w.hidden = Arc::clone(&right.hidden);
                        w
                    };
                    // Over the widened inputs, as the join itself was built.
                    let null = |a: &Attr, yes: bool| Attr {
                        col: BoundColumn {
                            nullable: a.col.nullable || yes,
                            ..a.col.clone()
                        },
                        ..a.clone()
                    };
                    let mut wide_out: Vec<Attr> = wl.out.iter().map(|a| null(a, *ln)).collect();
                    if !matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti) {
                        wide_out.extend(wr.out.iter().map(|a| null(a, *rn)));
                    }
                    let node = PlanNode::Join {
                        left: Arc::clone(&wl.plan),
                        right: Arc::clone(&wr.plan),
                        kind: *kind,
                        condition: condition.clone(),
                    };
                    let mut wide = self.make(rel.plan.origin, node, wide_out, &[&wl, &wr]);
                    wide.widen = Some(Arc::new(Widen::Join {
                        left: wl,
                        right: wr,
                        kind: *kind,
                        condition: condition.clone(),
                        nullable: (*ln, *rn),
                    }));
                    wide
                }
                Widen::Shift { inner, offset } => {
                    let back: Vec<Attr> = need
                        .iter()
                        .map(|a| Attr {
                            col: BoundColumn {
                                id: AttrId(a.col.id.0 - *offset),
                                ..a.col.clone()
                            },
                            ..a.clone()
                        })
                        .collect();
                    let mut w = self.widened(inner, &back)?;
                    w.hidden = Arc::clone(&inner.hidden);
                    let mut shifted = self.shifted(&w, *offset, None)?;
                    shifted.widen = Some(Arc::new(Widen::Shift {
                        inner: w,
                        offset: *offset,
                    }));
                    shifted
                }
                Widen::Scan => {
                    let PlanNode::Scan(scan) = &rel.plan.node else {
                        return Err(unsupported("a plan shape", rel.plan.origin));
                    };
                    let mut scan = scan.clone();
                    scan.output.extend(need.iter().map(|a| a.col.clone()));
                    let mut wide = self.wrap(
                        Arc::new(
                            LogicalPlan::new(PlanNode::Scan(scan)).with_origin(rel.plan.origin),
                        ),
                        rel.plan.origin.plan_id,
                        out,
                        &[],
                    );
                    wide.widen = Some(Arc::new(Widen::Scan));
                    wide
                }
            });
        }
        Ok(built.unwrap_or_else(|| r.clone()))
    }
}

/// Whether the expression tree of `e` holds an `IN` or `EXISTS` subquery of
/// its own. A subquery's plan is not entered: a scalar subquery holding an
/// `IN` inside is Spark-valid in a lateral join's condition.
fn holds_in_or_exists(e: &Arc<Expr>) -> bool {
    let mut stack = vec![Arc::clone(e)];
    let mut seen = HashSet::new();
    while let Some(x) = stack.pop() {
        if !seen.insert(super::bind::addr(&*x)) {
            continue;
        }
        if matches!(
            &x.node,
            ExprNode::Subquery {
                kind: crate::SubqueryKind::Exists | crate::SubqueryKind::In(_),
                ..
            }
        ) {
            return true;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    false
}

/// The name of the hidden column of file sources.
pub(super) const METADATA_COLUMN: &str = "_metadata";

/// Whether `a` is the `_metadata` struct the analyzer generates for a file
/// table, not a table column of that name (told by the provenance marker on
/// its fields).
pub(super) fn is_generated_metadata(a: &Attr) -> bool {
    a.col.name == METADATA_COLUMN
        && matches!(&a.col.data_type, DataType::Struct(s)
        if !s.fields().is_empty()
            && s.fields().iter().all(|f| {
                f.metadata().as_map().contains_key(crate::plan::GENERATED_METADATA_KEY)
            }))
}

/// The `_metadata` struct: Spark's field names, types and order; `row_index`
/// only for Parquet.
fn metadata_type(row_index: bool) -> DataType {
    use krishiv_types::{StructField, StructType, parse_metadata};
    let mark = || {
        parse_metadata(&format!(
            "{{\"{}\":true}}",
            crate::plan::GENERATED_METADATA_KEY
        ))
        .unwrap_or_else(|_| krishiv_types::Metadata::empty())
    };
    let field = |n: &str, t: DataType| StructField::new(n, t, false).with_metadata(mark());
    let mut fields = vec![
        field("file_path", DataType::String),
        field("file_name", DataType::String),
        field("file_size", DataType::Long),
        field("file_block_start", DataType::Long),
        field("file_block_length", DataType::Long),
        field("file_modification_time", DataType::Timestamp),
    ];
    if row_index {
        fields.push(field("row_index", DataType::Long));
    }
    DataType::Struct(StructType::new(fields))
}

/// The `_metadata` column of a file source named `format`.
fn file_metadata(format: &str) -> FileMetadata {
    match format.to_ascii_lowercase().as_str() {
        "parquet" => FileMetadata::WithRowIndex,
        "orc" | "csv" | "json" | "text" | "avro" | "xml" | "binaryfile" => FileMetadata::Basic,
        _ => FileMetadata::None,
    }
}

/// The function a table-valued generator relation was written as
/// (`FROM explode(..)`, `FROM stack(..)`, `FROM json_tuple(..)`), for the
/// alias-count error; `None` for any other relation.
fn generator_relation_name(plan: &LogicalPlan) -> Option<String> {
    match &plan.node {
        PlanNode::Generate {
            input,
            kind,
            outer,
            arg,
            ..
        } if matches!(input.node, PlanNode::OneRowRelation) => {
            if let ExprNode::Function(f) = &arg.node
                && f.unqualified() == Some("stack")
            {
                return Some("stack".to_owned());
            }
            Some(format!(
                "{}{}",
                kind.name(),
                if *outer { "_outer" } else { "" }
            ))
        }
        // The table functions that name themselves in the error.
        PlanNode::TableFunction { name, .. }
            if ["json_tuple", "range", "sql_keywords"]
                .iter()
                .any(|f| name.eq_ignore_ascii_case(f)) =>
        {
            Some(name.to_ascii_lowercase())
        }
        _ => None,
    }
}

mod lift;
pub(in crate::analyzer) use lift::*;

mod sortlimit;

mod join;

mod distinct;

mod aggregate;

mod qualify;
