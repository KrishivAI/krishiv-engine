// SPDX-License-Identifier: Apache-2.0
//! `UNION ALL`, `INTERSECT`, `EXCEPT`, `DISTINCT` and hints.
//!
//! - The inputs of a set operation meet at Spark's wider type per column
//!   (`WidenSetOperationTypes`): an input whose column has another type
//!   gets a projection casting it, under the same name. The output is the
//!   first input's attributes; a union's is nullable where any input's
//!   column is, an intersection's where both are.
//! - `unionByName` matches columns by name (Spark's `ResolveUnion`): each
//!   input is projected to the first input's column order; with
//!   `allowMissingColumns`, a column some input lacks reads NULL there
//!   and columns only later inputs have come last.
//! - `INTERSECT` / `EXCEPT` become what Spark's optimizer makes of them:
//!   the `DISTINCT` forms a semi or anti join on the null-safe equality
//!   of every column under a distinct, the `ALL` forms each side's count
//!   of every distinct row (a union tagged per side, then an aggregate)
//!   and [`PlanNode::ReplicateRows`] of `min(left, right)` (intersect) or
//!   `left - right` (except) copies.
//! - `DISTINCT` is an aggregate grouping every column
//!   (`ReplaceDistinctWithAggregate`).
//! - A hint passes rows through; [`super::hints`] resolves it once the
//!   whole plan is resolved (a hint names relations of its subtree).

use std::sync::Arc;

use krishiv_common::error::{
    ColumnAlreadyExists, IncompatibleColumnType, KrishivInvalidPartitionNumber,
    LegacyErrorTemp1047, LegacyErrorTemp1048, NumColumnsMismatch, UnresolvedColumnAmongFieldNames,
};
use krishiv_types::DataType;

use super::bind::{same, unsupported};
use super::catalog::quoted;
use super::coerce::{self, Arg, ordinal, quoted_type};
use super::exprs::Bound;
use super::ops::{col_expr, column_bound};
use super::resolve::{Attr, Env, Rel, Resolver, Widen};
use super::{AnalysisError, ArgType};
use crate::{
    ColumnRef, Expr, ExprNode, FunctionCall, Grouping, JoinCondition, JoinKind, Literal,
    LogicalPlan, Origin, PlanNode, ResolvedExpr, SetOperation,
};

/// Spark's partitioning hints: checked, then no-ops on one node.
pub(super) const PARTITIONING_HINTS: [&str; 4] = [
    "COALESCE",
    "REPARTITION",
    "REPARTITION_BY_RANGE",
    "REBALANCE",
];

impl Resolver<'_> {
    pub(super) fn union(
        &mut self,
        p: &LogicalPlan,
        inputs: &[Arc<LogicalPlan>],
        by_name: bool,
        allow_missing_columns: bool,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        let mut kids = inputs
            .iter()
            .map(|i| self.child(i, env))
            .collect::<Result<Vec<_>, _>>()?;
        if kids.is_empty() {
            return Err(unsupported("a union without inputs", o));
        }
        if by_name {
            kids = self.by_name(&kids, allow_missing_columns, o)?;
        }
        let kids = self.widen_inputs(&kids, "UNION", o)?;
        let mut out: Vec<Attr> = kids[0]
            .out
            .iter()
            .map(|a| Attr {
                col: a.col.clone(),
                // The first input's attributes, qualifier included (Spark).
                qual: Arc::clone(&a.qual),
                source: None,
            })
            .collect();
        for k in &kids[1..] {
            for (a, b) in out.iter_mut().zip(k.out.iter()) {
                a.col.nullable |= b.col.nullable;
            }
        }
        let node = PlanNode::Union {
            inputs: kids.iter().map(|k| Arc::clone(&k.plan)).collect(),
            by_name: false,
            allow_missing_columns: false,
        };
        let refs: Vec<&Rel> = kids.iter().collect();
        Ok(self.make(o, node, out, &refs))
    }

    /// The inputs of a union, cast to the wider type of each column.
    /// `operator` names the operation in errors.
    fn widen_inputs(
        &mut self,
        kids: &[Rel],
        operator: &str,
        o: Origin,
    ) -> Result<Vec<Rel>, AnalysisError> {
        let first = &kids[0];
        let width = first.out.len();
        for (i, k) in kids.iter().enumerate().skip(1) {
            if k.out.len() != width {
                return Err(AnalysisError::new(
                    NumColumnsMismatch {
                        operator: operator.to_owned(),
                        first_num_columns: width.to_string(),
                        invalid_ordinal_num: ordinal(i),
                        invalid_num_columns: k.out.len().to_string(),
                    },
                    o,
                ));
            }
        }
        // The wider type of each column, input by input.
        let mut types = Vec::with_capacity(width);
        for c in 0..width {
            let mut acc = first.out[c].col.data_type.clone();
            for (i, k) in kids.iter().enumerate().skip(1) {
                let here = &k.out[c].col.data_type;
                let ansi = self.ctx.conf.ansi;
                let wider = coerce::wider_type(&arg(Some(&acc)), &arg(Some(here)), ansi);
                acc = wider.map(|t| t.unwrap_or(DataType::Null)).ok_or_else(|| {
                    AnalysisError::new(
                        IncompatibleColumnType {
                            operator: operator.to_owned(),
                            column_ordinal_number: ordinal(c),
                            table_ordinal_number: ordinal(i),
                            data_type1: quoted_type(Some(here)),
                            data_type2: quoted_type(Some(&first.out[c].col.data_type)),
                            hint: String::new(),
                        },
                        o,
                    )
                })?;
            }
            types.push(acc);
        }
        kids.iter().map(|k| self.cast_columns(k, &types)).collect()
    }

    /// `unionByName`'s inputs, each projected to the first input's columns
    /// by name (Spark folds the union pairwise, so errors cite the second
    /// input). With `allow_missing`, a column an input lacks is an untyped
    /// NULL there (the union's widening types it) and the columns only
    /// later inputs have follow the first input's.
    fn by_name(
        &mut self,
        kids: &[Rel],
        allow_missing: bool,
        o: Origin,
    ) -> Result<Vec<Rel>, AnalysisError> {
        let cs = self.ctx.conf.case_sensitive;
        for k in kids {
            no_duplicates(k, cs)?;
        }
        let mut names: Vec<String> = kids[0].out.iter().map(|a| a.col.name.clone()).collect();
        for k in &kids[1..] {
            if allow_missing {
                for a in k.out.iter() {
                    if !names.iter().any(|n| same(n, &a.col.name, cs)) {
                        names.push(a.col.name.clone());
                    }
                }
                continue;
            }
            if let Some(n) = names
                .iter()
                .find(|n| !k.out.iter().any(|a| same(&a.col.name, n, cs)))
            {
                let fields: Vec<&str> = k.out.iter().map(|a| a.col.name.as_str()).collect();
                return Err(AnalysisError::new(
                    UnresolvedColumnAmongFieldNames {
                        col_name: n.clone(),
                        field_names: fields.join(", "),
                    },
                    o,
                ));
            }
            if k.out.len() != names.len() {
                return Err(AnalysisError::new(
                    NumColumnsMismatch {
                        operator: "UNION".to_owned(),
                        first_num_columns: names.len().to_string(),
                        invalid_ordinal_num: ordinal(1),
                        invalid_num_columns: k.out.len().to_string(),
                    },
                    o,
                ));
            }
        }
        kids.iter().map(|k| self.by_names(k, &names, cs)).collect()
    }

    /// `k` projected to the columns `names` (a missing one as NULL).
    fn by_names(&mut self, k: &Rel, names: &[String], cs: bool) -> Result<Rel, AnalysisError> {
        let found: Vec<Option<&Attr>> = names
            .iter()
            .map(|n| k.out.iter().find(|a| same(&a.col.name, n, cs)))
            .collect();
        let identity = found.len() == k.out.len()
            && found
                .iter()
                .zip(k.out.iter())
                .all(|(f, a)| f.is_some_and(|f| f.col.id == a.col.id));
        if identity {
            return Ok(k.clone());
        }
        let mut exprs = Vec::with_capacity(names.len());
        let mut out = Vec::with_capacity(names.len());
        for (n, f) in names.iter().zip(found) {
            match f {
                Some(a) => {
                    exprs.push(col_expr(&a.col, Origin::UNKNOWN));
                    out.push(a.clone());
                }
                None => {
                    let null = Arc::new(Expr::new(ExprNode::Literal(Literal::Null(None))));
                    let output = self.attr(
                        n,
                        &ArgType {
                            data_type: Some(DataType::Null),
                            nullable: true,
                        },
                    );
                    exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                        ResolvedExpr::Alias {
                            expr: null,
                            output: output.clone(),
                        },
                    ))));
                    out.push(Attr {
                        col: output,
                        qual: Arc::from(Vec::new()),
                        source: None,
                    });
                }
            }
        }
        Ok(self.projection(Origin::UNKNOWN, k, exprs, out))
    }

    /// `INTERSECT` / `EXCEPT`, rewritten as Spark's optimizer rewrites
    /// them (see the module docs).
    pub(super) fn set_op(
        &mut self,
        p: &LogicalPlan,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        op: SetOperation,
        all: bool,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        let l = self.child(left, env)?;
        let mut r = self.child(right, env)?;
        // `df.intersect(df)`: the right side reads its own attributes.
        if let (Some(a), Some(b)) = (l.ids, r.ids)
            && a.0 <= b.1
            && b.0 <= a.1
        {
            if r.outer.is_some_and(|k| k < env.outer.len()) {
                return Err(unsupported("a self-join of a correlated subtree", o));
            }
            r = self.instance(&r)?;
        }
        let kids = self.widen_inputs(&[l, r], op.sql(all), o)?;
        let (l, r) = (&kids[0], &kids[1]);
        let rel = if all {
            self.set_op_all(l, r, op, env)?
        } else {
            self.set_op_distinct(l, r, op, env)?
        };
        // The left attributes; an intersection's are nullable only where
        // both sides' columns are.
        let out: Vec<Attr> = l
            .out
            .iter()
            .zip(r.out.iter())
            .map(|(a, b)| {
                let mut a = a.clone();
                a.col.nullable &= op == SetOperation::Except || b.col.nullable;
                a
            })
            .collect();
        let exprs = out
            .iter()
            .map(|a| col_expr(&a.col, Origin::UNKNOWN))
            .collect();
        let node = PlanNode::Project {
            input: Arc::clone(&rel.plan),
            exprs,
        };
        Ok(self.make(o, node, out, &[&rel]))
    }

    /// `DISTINCT`: the distinct left rows with (intersect) or without
    /// (except) a null-safe match on the right.
    fn set_op_distinct(
        &mut self,
        l: &Rel,
        r: &Rel,
        op: SetOperation,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let mut cond: Option<Arc<Expr>> = None;
        for (a, b) in l.out.iter().zip(r.out.iter()) {
            let args = vec![
                col_expr(&a.col, Origin::UNKNOWN),
                col_expr(&b.col, Origin::UNKNOWN),
            ];
            let eq = self.call("<=>", args, env, Origin::UNKNOWN)?;
            cond = Some(match cond {
                None => eq,
                Some(c) => self.call("and", vec![c, eq], env, Origin::UNKNOWN)?,
            });
        }
        let kind = match op {
            SetOperation::Intersect => JoinKind::LeftSemi,
            SetOperation::Except => JoinKind::LeftAnti,
        };
        let node = PlanNode::Join {
            left: Arc::clone(&l.plan),
            right: Arc::clone(&r.plan),
            kind,
            condition: cond.map_or(JoinCondition::None, JoinCondition::On),
        };
        let j = self.make(Origin::UNKNOWN, node, l.out.to_vec(), &[l, r]);
        Ok(self.distinct_rel(Origin::UNKNOWN, &j))
    }

    /// `ALL`: each distinct row's count per side (a union of both sides
    /// tagged with `1` and `0` per side, summed per row), then the row
    /// `min(left, right)` (intersect) or `left - right` (except) times.
    fn set_op_all(
        &mut self,
        l: &Rel,
        r: &Rel,
        op: SetOperation,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let tagged = |this: &mut Self, k: &Rel, tags: [i64; 2]| {
            let mut exprs: Vec<Arc<Expr>> = k
                .out
                .iter()
                .map(|a| col_expr(&a.col, Origin::UNKNOWN))
                .collect();
            for (name, v) in ["vcol_l", "vcol_r"].into_iter().zip(tags) {
                exprs.push(alias(lit(v), name));
            }
            let items = this.select_items(k, &exprs, env, "Project")?;
            Ok::<_, AnalysisError>(this.lateral_project(Origin::UNKNOWN, k, &items, &[]))
        };
        let lt = tagged(self, l, [1, 0])?;
        let rt = tagged(self, r, [0, 1])?;
        let n = l.out.len();
        let mut uout = lt.out.to_vec();
        for (a, b) in uout.iter_mut().zip(rt.out.iter()) {
            a.col.nullable |= b.col.nullable;
        }
        let node = PlanNode::Union {
            inputs: vec![Arc::clone(&lt.plan), Arc::clone(&rt.plan)],
            by_name: false,
            allow_missing_columns: false,
        };
        let u = self.make(Origin::UNKNOWN, node, uout, &[&lt, &rt]);
        let group: Vec<Bound> = u.out[..n]
            .iter()
            .map(|a| column_bound(a, Origin::UNKNOWN))
            .collect();
        let mut exprs: Vec<Arc<Expr>> = u.out[..n]
            .iter()
            .map(|a| col_expr(&a.col, Origin::UNKNOWN))
            .collect();
        for (name, tag) in ["l", "r"].into_iter().zip(&u.out[n..]) {
            let sum = Arc::new(Expr::new(ExprNode::Function(FunctionCall {
                name: "sum".to_owned(),
                args: vec![col_expr(&tag.col, Origin::UNKNOWN)],
                distinct: false,
                filter: None,
            })));
            exprs.push(alias(sum, name));
        }
        let items = self.select_items(&u, &exprs, env, "Aggregate")?;
        let agg = self.build_aggregate(Origin::UNKNOWN, &u, group, &items, false)?;
        let col = |i: usize| col_expr(&agg.out[i].col, Origin::UNKNOWN);
        let (lc, rc) = (col(n), col(n + 1));
        let (predicate, count) = match op {
            SetOperation::Intersect => {
                let lp = self.call(">", vec![Arc::clone(&lc), lit(0)], env, Origin::UNKNOWN)?;
                let rp = self.call(">", vec![Arc::clone(&rc), lit(0)], env, Origin::UNKNOWN)?;
                let both = self.call("and", vec![lp, rp], env, Origin::UNKNOWN)?;
                let more = self.call(
                    ">",
                    vec![Arc::clone(&lc), Arc::clone(&rc)],
                    env,
                    Origin::UNKNOWN,
                )?;
                let least = self.call("when", vec![more, rc, lc], env, Origin::UNKNOWN)?;
                (both, least)
            }
            SetOperation::Except => {
                let diff = self.call("-", vec![lc, rc], env, Origin::UNKNOWN)?;
                let kept = self.call(">", vec![Arc::clone(&diff), lit(0)], env, Origin::UNKNOWN)?;
                (kept, diff)
            }
        };
        let node = PlanNode::Filter {
            input: Arc::clone(&agg.plan),
            predicate,
        };
        let f = self.make(Origin::UNKNOWN, node, agg.out.to_vec(), &[&agg]);
        let node = PlanNode::ReplicateRows {
            input: Arc::clone(&f.plan),
            count,
        };
        Ok(self.make(Origin::UNKNOWN, node, f.out.to_vec(), &[&f]))
    }

    /// `k` with its columns cast to `types` where they differ (same names).
    fn cast_columns(&mut self, k: &Rel, types: &[DataType]) -> Result<Rel, AnalysisError> {
        if k.out.iter().zip(types).all(|(a, t)| a.col.data_type == *t) {
            return Ok(k.clone());
        }
        let mut exprs = Vec::with_capacity(types.len());
        let mut out = Vec::with_capacity(types.len());
        for (a, t) in k.out.iter().zip(types) {
            if a.col.data_type == *t {
                exprs.push(col_expr(&a.col, Origin::UNKNOWN));
                out.push(a.clone());
                continue;
            }
            let b = Bound {
                expr: col_expr(&a.col, Origin::UNKNOWN),
                ty: ArgType {
                    data_type: Some(a.col.data_type.clone()),
                    nullable: a.col.nullable,
                },
                agg: None,
                win: None,
                outer: None,
                lateral: Vec::new(),
                shown: None,
            };
            let cast = self.implicit_cast(&b, coerce::cast_to(Some(&a.col.data_type), Some(t)));
            let output = self.attr(&a.col.name, &cast.ty);
            exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: cast.expr,
                    output: output.clone(),
                },
            ))));
            // A cast column is a new attribute without a qualifier.
            out.push(Attr {
                col: output,
                qual: Arc::from(Vec::new()),
                source: None,
            });
        }
        Ok(self.projection(k.plan.origin, k, exprs, out))
    }

    pub(super) fn distinct(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        Ok(self.distinct_rel(p.origin, &c))
    }

    /// `c`'s distinct rows: an aggregate grouping every column.
    fn distinct_rel(&self, origin: Origin, c: &Rel) -> Rel {
        let cols: Vec<Arc<Expr>> = c
            .out
            .iter()
            .map(|a| col_expr(&a.col, Origin::UNKNOWN))
            .collect();
        let node = PlanNode::Aggregate {
            input: Arc::clone(&c.plan),
            grouping: Grouping::GroupBy(cols.clone()),
            aggregates: cols,
        };
        self.make(origin, node, c.out.to_vec(), &[c])
    }

    /// A hint: rows pass through; [`super::hints::resolve`] applies it
    /// once the plan is resolved. A partitioning hint's parameters are
    /// checked here, against the hinted relation.
    pub(super) fn hint(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        name: &str,
        params: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let upper = name.to_ascii_uppercase();
        let c = self.child(input, env)?;
        if PARTITIONING_HINTS.contains(&upper.as_str()) {
            self.partitioning_params(&upper, params, &c, env, p.origin)?;
        }
        let node = PlanNode::Hint {
            input: Arc::clone(&c.plan),
            name: name.to_owned(),
            params: params.to_vec(),
        };
        let mut r = self.make(p.origin, node, c.out.to_vec(), &[&c]);
        r.widen = Some(Arc::new(Widen::Through { child: c }));
        Ok(r)
    }
}

impl Resolver<'_> {
    /// Checks a partitioning hint's parameters as Spark's
    /// `ResolveCoalesceHints` does: `COALESCE(n)`; `REPARTITION`,
    /// `REPARTITION_BY_RANGE` and `REBALANCE` an optional partition number
    /// first, then columns (identifiers or strings), which must resolve
    /// against the hinted relation. A partition number is a positive
    /// `INT` literal. The hint itself changes nothing on one node:
    /// [`super::hints`] drops it with a warning.
    fn partitioning_params(
        &mut self,
        upper: &str,
        params: &[Arc<Expr>],
        c: &Rel,
        env: &Arc<Env>,
        o: Origin,
    ) -> Result<(), AnalysisError> {
        let number = |e: &Expr| match &e.node {
            ExprNode::Literal(Literal::Integer(n)) => Some(*n),
            _ => None,
        };
        let positive = |n: i32| {
            if n > 0 {
                Ok(())
            } else {
                Err(AnalysisError::new(
                    KrishivInvalidPartitionNumber {
                        num_partitions: n.to_string(),
                    },
                    o,
                ))
            }
        };
        if upper == "COALESCE" {
            return match params {
                [one] if let Some(n) = number(one) => positive(n),
                _ => Err(AnalysisError::new(
                    LegacyErrorTemp1048 {
                        hint_name: upper.to_owned(),
                    },
                    o,
                )),
            };
        }
        let (n, columns) = match params.split_first() {
            Some((first, rest)) if let Some(n) = number(first) => (Some(n), rest),
            _ => (None, params),
        };
        let column = |e: &Expr| match &e.node {
            ExprNode::Column(_) => Some(Arc::new(e.clone())),
            ExprNode::Literal(Literal::String(s)) => Some(Arc::new(
                Expr::new(ExprNode::Column(ColumnRef {
                    name: s.split('.').map(str::to_owned).collect(),
                    plan_id: None,
                }))
                .with_origin(e.origin),
            )),
            _ => None,
        };
        let invalid: Vec<String> = columns
            .iter()
            .filter(|e| column(e).is_none())
            .map(|e| format!("\"{}\"", hint_value(e)))
            .collect();
        if !invalid.is_empty() {
            return Err(AnalysisError::new(
                LegacyErrorTemp1047 {
                    hint_name: upper.to_owned(),
                    invalid_params: invalid.join(", "),
                },
                o,
            ));
        }
        if let Some(n) = n {
            positive(n)?;
        }
        // Spark resolves them as missing references too: a column a
        // projection below dropped resolves.
        let columns: Vec<Arc<Expr>> = columns.iter().filter_map(|e| column(e)).collect();
        self.bind_missing(c, &columns, env, "Hint", false)?;
        Ok(())
    }
}

/// A hint parameter as Spark's errors print it: a literal by its value.
fn hint_value(e: &Expr) -> String {
    match &e.node {
        ExprNode::Literal(Literal::Null(_)) => "NULL".to_owned(),
        ExprNode::Literal(Literal::Boolean(b)) => b.to_string(),
        ExprNode::Literal(Literal::Byte(n)) => n.to_string(),
        ExprNode::Literal(Literal::Short(n)) => n.to_string(),
        ExprNode::Literal(Literal::Integer(n)) => n.to_string(),
        ExprNode::Literal(Literal::Long(n)) => n.to_string(),
        _ => super::exprs::pretty(e, None),
    }
}

/// A non-literal argument of type `t` (`VOID` is an untyped NULL).
fn arg(t: Option<&DataType>) -> Arg<'_> {
    Arg {
        ty: t.filter(|t| **t != DataType::Null),
        nullable: false,
        literal: None,
    }
}

/// A `BIGINT` literal.
fn lit(v: i64) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Literal(Literal::Long(v))))
}

/// `e AS name`.
fn alias(e: Arc<Expr>, name: &str) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Alias {
        expr: e,
        name: name.to_owned(),
    }))
}

/// `COLUMN_ALREADY_EXISTS` for a name `k` outputs twice (compared as the
/// session compares names; Spark reports it in lower case unless names
/// are case-sensitive).
fn no_duplicates(k: &Rel, cs: bool) -> Result<(), AnalysisError> {
    for (i, a) in k.out.iter().enumerate() {
        if k.out[..i]
            .iter()
            .any(|b| same(&b.col.name, &a.col.name, cs))
        {
            let name = if cs {
                a.col.name.clone()
            } else {
                a.col.name.to_lowercase()
            };
            return Err(AnalysisError::new(
                ColumnAlreadyExists {
                    column_name: quoted(&[name]),
                },
                Origin::UNKNOWN,
            ));
        }
    }
    Ok(())
}
