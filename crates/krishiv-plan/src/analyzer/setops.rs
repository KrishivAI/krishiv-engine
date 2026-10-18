// SPDX-License-Identifier: Apache-2.0
//! `UNION ALL`, `DISTINCT` and hints.
//!
//! - A union's inputs meet at Spark's wider type per column
//!   (`WidenSetOperationTypes`): an input whose column has another type
//!   gets a projection casting it, under the same name. The output is the
//!   first input's attributes, nullable where any input's column is.
//! - `DISTINCT` is an aggregate grouping every column
//!   (`ReplaceDistinctWithAggregate`).
//! - A hint passes rows through; [`super::hints`] resolves it once the
//!   whole plan is resolved (a hint names relations of its subtree).

use std::sync::Arc;

use krishiv_common::error::{IncompatibleColumnType, NumColumnsMismatch};

use super::bind::unsupported;
use super::coerce::{self, Arg, ordinal, quoted_type};
use super::exprs::Bound;
use super::ops::col_expr;
use super::resolve::{Attr, Env, Rel, Resolver, Widen};
use super::{AnalysisError, ArgType};
use crate::{Expr, ExprNode, Grouping, LogicalPlan, Origin, PlanNode, ResolvedExpr};

/// Spark's partitioning hints: no single-node meaning yet.
const PARTITIONING_HINTS: [&str; 4] = [
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
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        if by_name {
            return Err(unsupported("UNION BY NAME", o));
        }
        let kids = inputs
            .iter()
            .map(|i| self.child(i, env))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(first) = kids.first() else {
            return Err(unsupported("a union without inputs", o));
        };
        let width = first.out.len();
        for (i, k) in kids.iter().enumerate().skip(1) {
            if k.out.len() != width {
                return Err(AnalysisError::new(
                    NumColumnsMismatch {
                        operator: "UNION".to_owned(),
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
                let wider = coerce::common_type(&arg(Some(&acc)), &arg(Some(here)));
                acc = wider
                    .map(|t| t.unwrap_or_else(|| acc.clone()))
                    .ok_or_else(|| {
                        AnalysisError::new(
                            IncompatibleColumnType {
                                operator: "UNION".to_owned(),
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
        let kids: Vec<Rel> = kids
            .iter()
            .map(|k| self.cast_columns(k, &types))
            .collect::<Result<_, _>>()?;
        let mut out: Vec<Attr> = kids[0]
            .out
            .iter()
            .map(|a| Attr {
                col: a.col.clone(),
                qual: Arc::from(Vec::new()),
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

    /// `k` with its columns cast to `types` where they differ (same names).
    fn cast_columns(
        &mut self,
        k: &Rel,
        types: &[krishiv_types::DataType],
    ) -> Result<Rel, AnalysisError> {
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
                outer: None,
                lateral: Vec::new(),
                shown: None,
            };
            let cast = self.implicit_cast(&b, coerce::cast_to(Some(&a.col.data_type), Some(t)));
            let output = self.attr(&a.col.name, &cast.ty, Origin::UNKNOWN)?;
            exprs.push(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: cast.expr,
                    output: output.clone(),
                },
            ))));
            out.push(Attr {
                col: output,
                qual: Arc::clone(&a.qual),
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
        Ok(self.make(p.origin, node, c.out.to_vec(), &[&c]))
    }

    /// A hint: rows pass through; [`super::hints::resolve`] applies it
    /// once the plan is resolved. Partitioning hints are not supported.
    pub(super) fn hint(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        name: &str,
        params: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let upper = name.to_ascii_uppercase();
        if PARTITIONING_HINTS.contains(&upper.as_str()) {
            return Err(unsupported(format!("the {upper} hint"), p.origin));
        }
        let c = self.child(input, env)?;
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

/// A non-literal argument of type `t`.
fn arg(t: Option<&krishiv_types::DataType>) -> Arg<'_> {
    Arg {
        ty: t,
        nullable: false,
        literal: None,
    }
}
