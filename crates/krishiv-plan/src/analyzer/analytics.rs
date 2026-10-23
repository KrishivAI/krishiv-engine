// SPDX-License-Identifier: Apache-2.0
//! Grouping analytics: `GROUP BY ROLLUP`, `CUBE` and `GROUPING SETS`, and the
//! `grouping` and `grouping_id` functions.
//!
//! The rewrite is Spark's (the `ResolveGroupingAnalytics` rule and the
//! `Expand` operator): every distinct grouping expression gets a nullable
//! copy `g`, `Expand` repeats each input row once per grouping set (`g` is
//! the expression for a set that has it and `NULL` for a set that has not)
//! with the set's `spark_grouping_id`, and an ordinary aggregate groups by
//! the copies and the id. Bit `n - 1 - j` of the id is 1 when the `j`-th
//! distinct grouping expression (in order of first appearance) is absent
//! from the set. Select items, `HAVING` and `ORDER BY` read the copies where
//! they name a grouping expression (aggregate calls keep reading the
//! input's own columns, which stay in the expansion), `grouping(e)` is bit
//! `e` of the id (a `TINYINT`) and `grouping_id(..)` is the id (a `BIGINT`).
//!
//! Sets that are equal as sets stay separate rows (`GROUPING SETS ((a), (a))`
//! has two rows per group): the expansion then also numbers its sets
//! (`_gen_grouping_pos`) and the aggregate groups by that too. A single
//! empty set over no grouping expression is a grand total, an aggregate
//! without `GROUP BY` (one row over empty input) in which `grouping_id()` is 0.
//! The virtual column `grouping__id` names the id.
//!
//! Exact behaviour, Spark 4.2: `tests/spark-golden/v4.2.0` (`group-analytics`,
//! `grouping_set`, `postgreSQL/groupingsets`).

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::{
    GroupingColumnMismatch, GroupingIdColumnMismatch, GroupingSizeLimitExceeded,
    UnsupportedGroupingExpression, WrongNumArgsWithoutSuggestion,
};
use krishiv_types::DataType;

use super::bind::unsupported;
use super::exprs::{Bound, Cx, pretty, semantic_key};
use super::ops::col_expr;
use super::resolve::{Attr, Env, Rel, Resolver};
use super::{AnalysisError, ArgType};
use crate::{
    BoundColumn, CastMode, Expr, ExprNode, FunctionCall, FunctionKind, Grouping, Literal,
    LogicalPlan, Origin, PlanNode, ResolvedExpr,
};

/// The most grouping expressions one `GROUP BY` takes: the grouping id is a
/// `BIGINT` with one bit each.
const MAX_GROUPING: usize = 64;

/// The most expressions of a `CUBE`: 2^16 grouping sets (the SQL front end
/// bounds `CUBE (..)` items the same way).
const MAX_CUBE_EXPRS: usize = 16;

/// `grouping` or `grouping_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GroupingFn {
    Grouping,
    GroupingId,
}

impl GroupingFn {
    pub(super) fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("grouping") {
            Some(Self::Grouping)
        } else if name.eq_ignore_ascii_case("grouping_id") {
            Some(Self::GroupingId)
        } else {
            None
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Grouping => "grouping",
            Self::GroupingId => "grouping_id",
        }
    }

    const fn data_type(self) -> DataType {
        match self {
            Self::Grouping => DataType::Byte,
            Self::GroupingId => DataType::Long,
        }
    }
}

/// One distinct grouping expression.
pub(super) struct GroupExpr {
    /// Its semantic key.
    key: u128,
    /// Its attribute in the expansion: a nullable copy.
    pub(super) attr: Attr,
}

/// The expansion an aggregate reads.
pub(super) struct ExpandPlan {
    pub(super) plan: Arc<LogicalPlan>,
    /// The span of attribute ids the expansion defines.
    pub(super) ids: Option<(u64, u64)>,
}

/// An aggregate's grouping sets.
pub(super) struct Analytics {
    /// The distinct grouping expressions, in order of first appearance.
    pub(super) exprs: Vec<GroupExpr>,
    /// The grouping id as the query names it (`grouping__id`); `None` for a
    /// grand total, where it is 0.
    gid: Option<Attr>,
    /// The expansion; `None` for a grand total.
    pub(super) expand: Option<ExpandPlan>,
    /// Whether the expansion numbers its sets (some two are equal).
    numbered: bool,
}

impl Analytics {
    /// The attributes `HAVING` and `ORDER BY` may name: the grouping
    /// expressions' copies and `grouping__id`.
    pub(super) fn keys(&self) -> Vec<Attr> {
        self.exprs
            .iter()
            .map(|g| g.attr.clone())
            .chain(self.gid.iter().cloned())
            .collect()
    }

    /// `HAVING` and `ORDER BY` over numbered sets: Spark reads the last
    /// grouping key as the grouping id there, and that key is the set's
    /// number, so a `grouping` or `grouping_id` call is
    /// `UNSUPPORTED_GROUPING_EXPRESSION`.
    pub(super) fn check_late_call(&self, e: &Arc<Expr>) -> Result<(), AnalysisError> {
        if !self.numbered {
            return Ok(());
        }
        reject_grouping_calls(std::slice::from_ref(e))
    }

    /// The position of `e` among the grouping expressions: a copy of one, or
    /// an expression equal to one.
    fn position(&self, e: &Expr, memo: &mut HashMap<usize, u128>) -> Option<usize> {
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &e.node
            && let Some(j) = self.exprs.iter().position(|g| g.attr.col.id == c.id)
        {
            return Some(j);
        }
        let key = semantic_key(e, memo);
        self.exprs.iter().position(|g| g.key == key)
    }
}

/// `UNSUPPORTED_GROUPING_EXPRESSION` if any of `exprs` calls `grouping` or
/// `grouping_id`. Spark has no source position for it.
pub(super) fn reject_grouping_calls(exprs: &[Arc<Expr>]) -> Result<(), AnalysisError> {
    let mut seen = std::collections::HashSet::new();
    let mut stack: Vec<Arc<Expr>> = exprs.to_vec();
    while let Some(x) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&x) as usize) {
            continue;
        }
        if matches!(&x.node, ExprNode::Function(f) if GroupingFn::from_name(&f.name).is_some()) {
            return Err(AnalysisError::new(
                UnsupportedGroupingExpression {},
                Origin::UNKNOWN,
            ));
        }
        x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    Ok(())
}

/// A resolved call `e`, stated not nullable.
fn not_null(e: &Arc<Expr>) -> Arc<Expr> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            let mut f = f.clone();
            f.nullable = false;
            Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))).with_origin(e.origin))
        }
        _ => Arc::clone(e),
    }
}

fn literal(l: Literal) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Literal(l)))
}

fn arg_type(data_type: DataType, nullable: bool) -> ArgType {
    ArgType {
        data_type: Some(data_type),
        nullable,
    }
}

fn bound_of(col: &BoundColumn) -> Bound {
    Bound {
        expr: col_expr(col, Origin::UNKNOWN),
        ty: arg_type(col.data_type.clone(), col.nullable),
        agg: None,
        win: None,
        outer: None,
        lateral: Vec::new(),
        shown: None,
    }
}

/// Spark's text for an expression in a grouping error: a column carries its
/// attribute id.
fn with_id(e: &Expr) -> String {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => format!("{}#{}", c.name, c.id.0),
        _ => pretty(e),
    }
}

/// Whether a subquery is anywhere in `e`.
fn has_subquery(e: &Arc<Expr>) -> bool {
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if matches!(x.node, ExprNode::Subquery { .. }) {
            return true;
        }
        x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    false
}

/// The shape of a grouping: all its expressions in order, and each grouping
/// set as positions into them.
fn shape(grouping: &Grouping) -> (Vec<Arc<Expr>>, Vec<Vec<usize>>) {
    match grouping {
        Grouping::GroupBy(v) => (v.clone(), vec![(0..v.len()).collect()]),
        Grouping::Rollup(v) => (
            v.clone(),
            (0..=v.len()).rev().map(|n| (0..n).collect()).collect(),
        ),
        Grouping::Cube(v) => {
            // The full set first, then down to the empty one.
            let n = v.len();
            let sets = (0..1_usize << n)
                .rev()
                .map(|m| (0..n).filter(|i| m >> (n - 1 - i) & 1 == 1).collect())
                .collect();
            (v.clone(), sets)
        }
        Grouping::GroupingSets { exprs, sets } => {
            let mut flat = exprs.clone();
            let mut out = Vec::with_capacity(sets.len());
            for set in sets {
                out.push((flat.len()..flat.len() + set.len()).collect());
                flat.extend(set.iter().cloned());
            }
            (flat, out)
        }
    }
}

impl Resolver<'_> {
    /// `grouping(e)` or `grouping_id(..)` in a select list, `HAVING` or
    /// `ORDER BY`: a call the aggregate resolves once it knows its grouping
    /// sets ([`Self::resolve_grouping`]). Anywhere else it is
    /// `UNSUPPORTED_GROUPING_EXPRESSION`.
    #[inline(never)]
    pub(super) fn grouping_call(
        &mut self,
        which: GroupingFn,
        e: &Arc<Expr>,
        kids: &[Bound],
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        // Spark has no source position for the error in `HAVING` and
        // `ORDER BY`.
        let origin = if matches!(cx.op, "Filter" | "Sort") {
            Origin::UNKNOWN
        } else {
            e.origin
        };
        if !cx.grouping {
            return Err(AnalysisError::new(UnsupportedGroupingExpression {}, origin));
        }
        if which == GroupingFn::Grouping && kids.len() != 1 {
            return Err(AnalysisError::new(
                WrongNumArgsWithoutSuggestion {
                    function_name: "`grouping`".to_owned(),
                    expected_num: "1".to_owned(),
                    actual_num: kids.len().to_string(),
                    docroot: super::exprs::SPARK_DOC_ROOT.to_owned(),
                },
                e.origin,
            ));
        }
        let call = FunctionCall {
            name: which.name().to_owned(),
            args: kids.iter().map(|k| Arc::clone(&k.expr)).collect(),
            distinct: false,
            filter: None,
        };
        Ok(Bound {
            expr: Arc::new(Expr::new(ExprNode::Function(call)).with_origin(e.origin)),
            ty: arg_type(which.data_type(), false),
            agg: kids.iter().find_map(|k| k.agg),
            win: kids.iter().find_map(|k| k.win),
            outer: kids.iter().filter_map(|k| k.outer).min(),
            lateral: kids.iter().flat_map(|k| k.lateral.clone()).collect(),
            shown: None,
        })
    }

    /// A resolved `Expand` read again: its expressions bound over its
    /// input, its output as it states it.
    #[inline(never)]
    pub(super) fn expand_node(
        &mut self,
        origin: Origin,
        input: &Arc<LogicalPlan>,
        projections: &[Vec<Arc<Expr>>],
        output: &[BoundColumn],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let rels = [&c];
        let cx = Self::cx(&c.out, &rels, env, "Expand");
        let mut bound = Vec::with_capacity(projections.len());
        for row in projections {
            let mut exprs = Vec::with_capacity(row.len());
            for e in row {
                exprs.push(self.bind(e, &cx)?.expr);
            }
            bound.push(exprs);
        }
        let out = output
            .iter()
            .map(|col| Attr {
                col: col.clone(),
                qual: Arc::from(Vec::new()),
                source: None,
            })
            .collect();
        let node = PlanNode::Expand {
            input: Arc::clone(&c.plan),
            projections: bound,
            output: output.to_vec(),
        };
        Ok(self.make(origin, node, out, &[&c]))
    }

    /// `e` with its `grouping` and `grouping_id` calls resolved against
    /// `an`.
    #[inline(never)]
    pub(super) fn resolve_grouping(
        &mut self,
        e: &Arc<Expr>,
        an: &Analytics,
        env: &Arc<Env>,
    ) -> Result<Arc<Expr>, AnalysisError> {
        self.rewrite(e, an, false, env, &mut HashMap::new())
    }

    /// An aggregate over `ROLLUP`, `CUBE` or `GROUPING SETS`.
    #[inline(never)]
    pub(super) fn grouping_sets(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        grouping: &Grouping,
        aggregates: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        // A cube has a set per subset of its expressions: bound it before
        // building them (`WITH CUBE` and `DataFrame.cube` arrive here).
        if let Grouping::Cube(v) = grouping
            && v.len() > MAX_CUBE_EXPRS
        {
            return Err(AnalysisError::new(
                GroupingSizeLimitExceeded {
                    max_size: (1_u64 << MAX_CUBE_EXPRS).to_string(),
                },
                p.origin,
            ));
        }
        let c = self.child(input, env)?;
        let (flat, sets) = shape(grouping);
        self.check_star_ordinals(&flat, aggregates)?;
        // A single empty set over nothing is a grand total.
        let grand = flat.is_empty() && sets.len() == 1;
        let gid_col =
            (!grand).then(|| self.attr("spark_grouping_id", &arg_type(DataType::Long, false)));
        let gid = gid_col.as_ref().map(|col| Attr {
            col: BoundColumn {
                name: "grouping__id".to_owned(),
                ..col.clone()
            },
            qual: Arc::from(Vec::new()),
            source: None,
        });
        let keys: Vec<Attr> = gid.iter().cloned().collect();
        let items = self.select_items_in(&c, aggregates, env, "Aggregate", Some(&keys))?;
        let bound = self.group_keys(&c, &flat, aggregates, &items, env)?;
        let mut memo = HashMap::new();
        // The distinct expressions, and the sets as positions among them.
        let mut distinct: Vec<usize> = Vec::new();
        let mut keys_of: Vec<u128> = Vec::new();
        let mut at = Vec::with_capacity(bound.len());
        for (i, b) in bound.iter().enumerate() {
            let key = semantic_key(&b.expr, &mut memo);
            let j = match keys_of.iter().position(|k| *k == key) {
                Some(j) => j,
                None => {
                    keys_of.push(key);
                    distinct.push(i);
                    keys_of.len() - 1
                }
            };
            at.push(j);
        }
        let n = distinct.len();
        if n > MAX_GROUPING {
            return Err(AnalysisError::new(
                GroupingSizeLimitExceeded {
                    max_size: MAX_GROUPING.to_string(),
                },
                p.origin,
            ));
        }
        let sets: Vec<Vec<usize>> = sets
            .into_iter()
            .map(|s| {
                let mut s: Vec<usize> = s.into_iter().map(|i| at[i]).collect();
                s.sort_unstable();
                s.dedup();
                s
            })
            .collect();
        if grand {
            let an = Analytics {
                exprs: Vec::new(),
                gid: None,
                expand: None,
                numbered: false,
            };
            let items = self.rewrite_items(items, &an, env)?;
            return self.build_aggregate(
                p.origin,
                &c,
                Vec::new(),
                &items,
                true,
                Some(Arc::new(an)),
            );
        }
        let Some(gid_col) = gid_col else {
            return Err(unsupported("grouping sets without an id", p.origin));
        };
        let numbered = (0..sets.len()).any(|i| sets[..i].contains(&sets[i]));
        // The expansion's input: the child's columns, and the grouping
        // expressions that are not one of them.
        let mut passed: Vec<Arc<Expr>> = c
            .out
            .iter()
            .map(|a| col_expr(&a.col, Origin::UNKNOWN))
            .collect();
        let mut passed_out: Vec<Attr> = c.out.to_vec();
        let mut value: Vec<BoundColumn> = Vec::with_capacity(n);
        let mut copies: Vec<GroupExpr> = Vec::with_capacity(n);
        let mut computed = false;
        for (j, &i) in distinct.iter().enumerate() {
            let b = &bound[i];
            let column = match &b.expr.node {
                ExprNode::Resolved(ResolvedExpr::Column(col)) => {
                    c.out.iter().find(|a| a.col.id == col.id).cloned()
                }
                _ => None,
            };
            let name = match &column {
                Some(a) => a.col.name.clone(),
                None => pretty(&b.shown()),
            };
            let v = match &column {
                Some(a) => a.col.clone(),
                None => {
                    let v = self.attr(&name, &b.ty);
                    passed.push(Arc::new(Expr::new(ExprNode::Resolved(
                        ResolvedExpr::Alias {
                            expr: Arc::clone(&b.expr),
                            output: v.clone(),
                        },
                    ))));
                    passed_out.push(Attr {
                        col: v.clone(),
                        qual: Arc::from(Vec::new()),
                        source: None,
                    });
                    computed = true;
                    v
                }
            };
            let ty = arg_type(v.data_type.clone(), true);
            let copy = self.attr(&name, &ty);
            copies.push(GroupExpr {
                key: keys_of[j],
                attr: Attr {
                    col: copy,
                    qual: column
                        .as_ref()
                        .map_or_else(|| Arc::from(Vec::new()), |a| Arc::clone(&a.qual)),
                    source: column.and_then(|a| a.source),
                },
            });
            value.push(v);
        }
        let below = if computed {
            let out = passed_out.clone();
            self.projection(Origin::UNKNOWN, &c, passed, out)
        } else {
            c.clone()
        };
        let pos_col =
            numbered.then(|| self.attr("_gen_grouping_pos", &arg_type(DataType::Integer, false)));
        let mut output: Vec<BoundColumn> = passed_out.iter().map(|a| a.col.clone()).collect();
        output.extend(copies.iter().map(|g| g.attr.col.clone()));
        output.push(gid_col.clone());
        output.extend(pos_col.iter().cloned());
        let mut projections = Vec::with_capacity(sets.len());
        for (si, set) in sets.iter().enumerate() {
            let mut row: Vec<Arc<Expr>> = passed_out
                .iter()
                .map(|a| col_expr(&a.col, Origin::UNKNOWN))
                .collect();
            let mut mask: i64 = 0;
            for (j, v) in value.iter().enumerate() {
                if set.contains(&j) {
                    row.push(col_expr(v, Origin::UNKNOWN));
                } else {
                    row.push(literal(Literal::Null(Some(v.data_type.clone()))));
                    mask |= 1 << (n - 1 - j);
                }
            }
            row.push(literal(Literal::Long(mask)));
            if pos_col.is_some() {
                row.push(literal(Literal::Integer(i32::try_from(si).map_err(
                    |_| unsupported("that many grouping sets", p.origin),
                )?)));
            }
            projections.push(row);
        }
        let out_attrs: Vec<Attr> = passed_out
            .iter()
            .cloned()
            .chain(copies.iter().map(|g| g.attr.clone()))
            .chain(std::iter::once(Attr {
                col: gid_col.clone(),
                qual: Arc::from(Vec::new()),
                source: None,
            }))
            .chain(pos_col.iter().map(|col| Attr {
                col: col.clone(),
                qual: Arc::from(Vec::new()),
                source: None,
            }))
            .collect();
        let node = PlanNode::Expand {
            input: Arc::clone(&below.plan),
            projections,
            output,
        };
        let expand = self.make(Origin::UNKNOWN, node, out_attrs, &[&below]);
        let mut group: Vec<Bound> = copies.iter().map(|g| bound_of(&g.attr.col)).collect();
        group.push(bound_of(&gid_col));
        group.extend(pos_col.iter().map(bound_of));
        let an = Analytics {
            exprs: copies,
            gid,
            numbered,
            expand: Some(ExpandPlan {
                plan: Arc::clone(&expand.plan),
                ids: expand.ids,
            }),
        };
        let items = self.rewrite_items(items, &an, env)?;
        self.build_aggregate(p.origin, &c, group, &items, false, Some(Arc::new(an)))
    }

    /// The select `items` over the grouping sets: grouping expressions read
    /// their copies, `grouping` and `grouping_id` are resolved.
    #[inline(never)]
    fn rewrite_items(
        &mut self,
        items: Vec<Bound>,
        an: &Analytics,
        env: &Arc<Env>,
    ) -> Result<Vec<Bound>, AnalysisError> {
        let mut memo = HashMap::new();
        let mut out = Vec::with_capacity(items.len());
        for b in items {
            let expr = self.rewrite(&b.expr, an, true, env, &mut memo)?;
            let ty = if Arc::ptr_eq(&expr, &b.expr) {
                b.ty.clone()
            } else {
                self.type_of(&expr, env)?
            };
            out.push(Bound { expr, ty, ..b });
        }
        Ok(out)
    }

    /// The type of the resolved `e`.
    fn type_of(&mut self, e: &Arc<Expr>, env: &Arc<Env>) -> Result<ArgType, AnalysisError> {
        let inner = match &e.node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => expr,
            _ => e,
        };
        Ok(self.bind(inner, &Self::cx(&[], &[], env, "Aggregate"))?.ty)
    }

    /// `e` over the grouping sets: with `replace`, a subtree that is a
    /// grouping expression reads its copy (aggregate calls and subqueries
    /// are left as they are); `grouping` and `grouping_id` are resolved.
    /// Nodes above a change are typed again, as the copies are nullable.
    #[inline(never)]
    fn rewrite(
        &mut self,
        e: &Arc<Expr>,
        an: &Analytics,
        replace: bool,
        env: &Arc<Env>,
        memo: &mut HashMap<usize, u128>,
    ) -> Result<Arc<Expr>, AnalysisError> {
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
                // `grouping` calls inside aggregate arguments still resolve,
                // but ordinary columns stay on the input (no copy).
                let rebuilt =
                    Arc::clone(e).map_children(|c| self.rewrite(&c, an, false, env, memo))?;
                if Arc::ptr_eq(&rebuilt, e) {
                    return Ok(Arc::clone(e));
                }
                return self.retype(rebuilt, env);
            }
            ExprNode::Subquery { .. } => return Ok(Arc::clone(e)),
            ExprNode::Function(f) if GroupingFn::from_name(&f.name).is_some() => {
                return self.grouping_value(f, an, env, memo);
            }
            ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
                let inner = self.rewrite(expr, an, replace, env, memo)?;
                if Arc::ptr_eq(&inner, expr) {
                    return Ok(Arc::clone(e));
                }
                let ty = self.type_of(&inner, env)?;
                let output = BoundColumn {
                    nullable: ty.nullable,
                    ..output.clone()
                };
                return Ok(Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: inner,
                        output,
                    }))
                    .with_origin(e.origin),
                ));
            }
            _ => {}
        }
        if replace && let Some(j) = an.position(e, memo) {
            return Ok(col_expr(&an.exprs[j].attr.col, e.origin));
        }
        let rebuilt = Arc::clone(e).map_children(|c| self.rewrite(&c, an, replace, env, memo))?;
        if Arc::ptr_eq(&rebuilt, e) {
            return Ok(rebuilt);
        }
        self.retype(rebuilt, env)
    }

    /// `e`, whose children changed, typed again.
    fn retype(&mut self, e: Arc<Expr>, env: &Arc<Env>) -> Result<Arc<Expr>, AnalysisError> {
        let origin = e.origin;
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Function(f)) if !has_subquery(&e) => {
                let call = Arc::new(
                    Expr::new(ExprNode::Function(FunctionCall {
                        name: f.id.name().to_owned(),
                        args: f.args.clone(),
                        distinct: f.distinct,
                        filter: f.filter.clone(),
                    }))
                    .with_origin(origin),
                );
                Ok(self.bind(&call, &Self::cx(&[], &[], env, "Join"))?.expr)
            }
            ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                // A call over a subquery keeps its type, nullable.
                let mut f = f.clone();
                f.nullable = true;
                Ok(Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))).with_origin(origin),
                ))
            }
            ExprNode::Cast { .. } if !has_subquery(&e) => {
                Ok(self.bind(&e, &Self::cx(&[], &[], env, "Aggregate"))?.expr)
            }
            _ => Ok(e),
        }
    }

    /// The value of a `grouping` or `grouping_id` call.
    fn grouping_value(
        &mut self,
        f: &FunctionCall,
        an: &Analytics,
        env: &Arc<Env>,
        memo: &mut HashMap<usize, u128>,
    ) -> Result<Arc<Expr>, AnalysisError> {
        let n = an.exprs.len();
        let id = match &an.gid {
            Some(g) => col_expr(&g.col, Origin::UNKNOWN),
            None => literal(Literal::Long(0)),
        };
        let positions: Vec<Option<usize>> = f.args.iter().map(|a| an.position(a, memo)).collect();
        match GroupingFn::from_name(&f.name) {
            Some(GroupingFn::GroupingId) => {
                let exact = positions.len() == n
                    && positions.iter().enumerate().all(|(i, p)| *p == Some(i));
                if !f.args.is_empty() && !exact {
                    return Err(AnalysisError::new(
                        GroupingIdColumnMismatch {
                            grouping_id_column: f
                                .args
                                .iter()
                                .map(|a| with_id(a))
                                .collect::<Vec<_>>()
                                .join(","),
                            group_by_columns: an
                                .exprs
                                .iter()
                                .map(|g| format!("{}#{}", g.attr.col.name, g.attr.col.id.0))
                                .collect::<Vec<_>>()
                                .join(","),
                        },
                        Origin::UNKNOWN,
                    ));
                }
                Ok(id)
            }
            _ => {
                let (Some(_), Some(Some(j))) = (f.args.first(), positions.first()) else {
                    return Err(AnalysisError::new(
                        GroupingColumnMismatch {
                            grouping: f.args.first().map(|a| with_id(a)).unwrap_or_default(),
                            grouping_columns: an
                                .exprs
                                .iter()
                                .map(|g| format!("{}#{}", g.attr.col.name, g.attr.col.id.0))
                                .collect::<Vec<_>>()
                                .join(","),
                        },
                        Origin::UNKNOWN,
                    ));
                };
                // Bit `n - 1 - j` of the id, as a `TINYINT`.
                // With 64 expressions the id can be negative (bit 63 set),
                // where signed `div`/`%` on `i64::MIN` reads -1 instead of 1:
                // there the quotient is floored before `pmod(_, 2)` (bit 63
                // is the sign). Smaller ids stay non-negative and keep the
                // plain `div`/`%` extraction.
                let shift = n - 1 - j;
                let bit = if n < 64 {
                    // The divisors are literals that are not 0, so the id's
                    // bit is never null: `grouping` is not nullable (Spark).
                    let high = if shift == 0 {
                        id
                    } else {
                        let high = self.call(
                            "div",
                            vec![id, literal(Literal::Long(1_i64 << shift))],
                            env,
                            Origin::UNKNOWN,
                        )?;
                        not_null(&high)
                    };
                    self.call(
                        "%",
                        vec![high, literal(Literal::Long(2))],
                        env,
                        Origin::UNKNOWN,
                    )?
                } else if shift >= 63 {
                    let cond = self.call(
                        "<",
                        vec![Arc::clone(&id), literal(Literal::Long(0))],
                        env,
                        Origin::UNKNOWN,
                    )?;
                    self.call(
                        "when",
                        vec![cond, literal(Literal::Long(1)), literal(Literal::Long(0))],
                        env,
                        Origin::UNKNOWN,
                    )?
                } else {
                    let pow = 1_i64 << shift;
                    let high = self.call(
                        "div",
                        vec![Arc::clone(&id), literal(Literal::Long(pow))],
                        env,
                        Origin::UNKNOWN,
                    )?;
                    let high = not_null(&high);
                    let rem = self.call(
                        "%",
                        vec![Arc::clone(&id), literal(Literal::Long(pow))],
                        env,
                        Origin::UNKNOWN,
                    )?;
                    let neg = self.call(
                        "<",
                        vec![Arc::clone(&id), literal(Literal::Long(0))],
                        env,
                        Origin::UNKNOWN,
                    )?;
                    let zero = self.call(
                        "==",
                        vec![rem, literal(Literal::Long(0))],
                        env,
                        Origin::UNKNOWN,
                    )?;
                    let nonzero = self.call("not", vec![zero], env, Origin::UNKNOWN)?;
                    let adjust = self.call("and", vec![neg, nonzero], env, Origin::UNKNOWN)?;
                    let one = literal(Literal::Long(1));
                    let decremented =
                        self.call("-", vec![Arc::clone(&high), one], env, Origin::UNKNOWN)?;
                    let floor = self.call(
                        "when",
                        vec![adjust, decremented, Arc::clone(&high)],
                        env,
                        Origin::UNKNOWN,
                    )?;
                    // `pmod` (not `%`) so a negative quotient reads 0/1.
                    self.call(
                        "pmod",
                        vec![floor, literal(Literal::Long(2))],
                        env,
                        Origin::UNKNOWN,
                    )?
                };
                let bit = not_null(&bit);
                let cast = Arc::new(Expr::new(ExprNode::Cast {
                    expr: bit,
                    to: DataType::Byte,
                    mode: CastMode::Default,
                }));
                Ok(self
                    .bind(&cast, &Self::cx(&[], &[], env, "Aggregate"))?
                    .expr)
            }
        }
    }
}
