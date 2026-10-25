// SPDX-License-Identifier: Apache-2.0
//! `ResolveUnpivot`: an unresolved UNPIVOT node becomes an Expand node
//! (one projection per column set: the kept columns, the name literal, the
//! values), as Spark's `UnpivotTransformer` builds it, followed unless
//! `INCLUDE NULLS` by a Filter dropping rows whose values are all null (as
//! Spark's parser adds it).
//!
//! The kept columns are every input column no column set references. Each
//! value position takes the wider common type of its members (through
//! `coerce`, as Spark's `UnpivotTypeCoercion` widens them):
//! `UNPIVOT_VALUE_DATA_TYPE_MISMATCH` when positions share no type. A set
//! whose size differs from the value column count is
//! `UNPIVOT_VALUE_SIZE_MISMATCH`; no value columns at all is
//! `UNPIVOT_REQUIRES_VALUE_COLUMNS`.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use krishiv_common::error::{
    UnpivotRequiresAttributes, UnpivotRequiresValueColumns, UnpivotValueDataTypeMismatch,
    UnpivotValueSizeMismatch,
};
use krishiv_types::DataType;

use super::AnalysisError;
use super::catalog::quoted;
use super::coerce::{Arg, Rules, wider_type};
use super::exprs::{Bound, pretty};
use super::ops::{boolean_filter, col_expr, no_qual};
use super::resolve::{Attr, Env, Rel, Resolver, Widen};
use crate::{
    AttrId, BoundColumn, Builtin, CastMode, ColumnRef, Expr, ExprNode, FunctionCall, Literal,
    LogicalPlan, PlanNode, ResolvedExpr, UnpivotColumnSet,
};

impl Resolver<'_> {
    /// Resolves `p`: an `UNPIVOT` over `input` with `value_columns` named
    /// by `name_column`, unpivoting `column_sets`.
    #[expect(
        clippy::too_many_arguments,
        reason = "one call site, every field of the node"
    )]
    pub(super) fn unpivot(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        value_columns: &[String],
        name_column: &str,
        column_sets: &[UnpivotColumnSet],
        include_nulls: bool,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let rels = [&c];
        let cx = Self::cx(&c.out, &rels, env, "Unpivot");
        let mut sets: Vec<Vec<Bound>> = Vec::with_capacity(column_sets.len());
        for s in column_sets {
            let mut members = Vec::with_capacity(s.columns.len());
            for col in &s.columns {
                members.push(self.bind(col, &cx)?);
            }
            sets.push(members);
        }
        // Spark's `CheckAnalysis` order: attributes, then value columns,
        // then sizes, then types.
        let zone = self.ctx.conf.session_zone;
        let mut stray: Vec<String> = Vec::new();
        let mut member_ids: Vec<Vec<AttrId>> = Vec::with_capacity(sets.len());
        let mut member_names: Vec<Vec<String>> = Vec::with_capacity(sets.len());
        for members in &sets {
            let mut ids = Vec::with_capacity(members.len());
            let mut names = Vec::with_capacity(members.len());
            for b in members {
                match &b.expr.node {
                    ExprNode::Resolved(
                        ResolvedExpr::Column(col) | ResolvedExpr::OuterColumn(col),
                    ) => {
                        ids.push(col.id);
                        names.push(
                            c.out
                                .iter()
                                .find(|a| a.col.id == col.id)
                                .map_or_else(|| pretty(&b.expr, zone), |a| a.col.name.clone()),
                        );
                    }
                    _ => stray.push(format!(
                        "\"{}\"",
                        pretty(&b.expr, self.ctx.conf.session_zone)
                    )),
                }
            }
            member_ids.push(ids);
            member_names.push(names);
        }
        if !stray.is_empty() {
            return Err(AnalysisError::new(
                UnpivotRequiresAttributes {
                    given: "value".to_owned(),
                    empty: "id".to_owned(),
                    expressions: stray.join(", "),
                },
                p.origin,
            ));
        }
        if value_columns.is_empty() || column_sets.is_empty() {
            return Err(AnalysisError::new(UnpivotRequiresValueColumns {}, p.origin));
        }
        for members in &sets {
            if members.len() != value_columns.len() {
                return Err(AnalysisError::new(
                    UnpivotValueSizeMismatch {
                        names: value_columns.len().to_string(),
                    },
                    p.origin,
                ));
            }
        }
        let rules = Rules {
            ansi: self.ctx.conf.ansi,
            case_sensitive: self.ctx.conf.case_sensitive,
        };
        let used: HashSet<AttrId> = member_ids.iter().flatten().copied().collect();
        let kept: Vec<Attr> = c
            .out
            .iter()
            .filter(|a| !used.contains(&a.col.id))
            .cloned()
            .collect();
        // The wider common type of each value position, as Spark's
        // `UnpivotTypeCoercion` widens them.
        let mut wider: Vec<DataType> = Vec::with_capacity(value_columns.len());
        for i in 0..value_columns.len() {
            let mut acc: Option<DataType> = None;
            for members in &sets {
                let ty = members[i].ty.data_type.clone();
                let next = wider_type(
                    &Arg {
                        ty: acc.as_ref(),
                        nullable: false,
                        literal: None,
                    },
                    &Arg {
                        ty: ty.as_ref(),
                        nullable: members[i].ty.nullable,
                        literal: None,
                    },
                    rules,
                );
                match next {
                    Some(t) => acc = t,
                    None => {
                        return Err(AnalysisError::new(
                            UnpivotValueDataTypeMismatch {
                                types: unpivot_types(&sets, &member_names),
                            },
                            p.origin,
                        ));
                    }
                }
            }
            wider.push(acc.unwrap_or(DataType::Null));
        }
        // Each member at its position's type, cast when it differs (named
        // as Spark names the cast: the member's name).
        let mut values: Vec<Vec<Arc<Expr>>> = Vec::with_capacity(sets.len());
        for members in &sets {
            let mut row = Vec::with_capacity(members.len());
            for (i, b) in members.iter().enumerate() {
                if b.ty.data_type.as_ref() == Some(&wider[i]) {
                    row.push(Arc::clone(&b.expr));
                    continue;
                }
                let cast = Expr::new(ExprNode::Cast {
                    expr: Arc::clone(&b.expr),
                    to: wider[i].clone(),
                    mode: CastMode::Default,
                })
                .with_origin(b.expr.origin);
                row.push(self.bind(&Arc::new(cast), &cx)?.expr);
            }
            values.push(row);
        }
        let mut projections: Vec<Vec<Arc<Expr>>> = Vec::with_capacity(sets.len());
        for (j, s) in column_sets.iter().enumerate() {
            let mut row: Vec<Arc<Expr>> = kept.iter().map(|a| col_expr(&a.col, p.origin)).collect();
            let label = match &s.alias {
                Some(alias) => alias.clone(),
                None if value_columns.len() == 1 => member_names[j][0].clone(),
                None => member_names[j].join("_"),
            };
            row.push(Arc::new(
                Expr::new(ExprNode::Literal(Literal::String(label))).with_origin(p.origin),
            ));
            row.extend(values[j].iter().cloned());
            projections.push(row);
        }
        let mut output: Vec<BoundColumn> = kept.iter().map(|a| a.col.clone()).collect();
        output.push(BoundColumn {
            id: self.fresh(),
            name: name_column.to_owned(),
            data_type: DataType::String,
            nullable: false,
        });
        for (i, name) in value_columns.iter().enumerate() {
            let nullable = sets.iter().any(|members| members[i].ty.nullable);
            output.push(BoundColumn {
                id: self.fresh(),
                name: name.clone(),
                data_type: wider[i].clone(),
                nullable,
            });
        }
        let mut out: Vec<Attr> = kept;
        out.extend(
            output[output.len() - value_columns.len() - 1..]
                .iter()
                .map(|col| Attr {
                    col: col.clone(),
                    qual: no_qual(),
                    source: None,
                }),
        );
        let node = PlanNode::Expand {
            input: Arc::clone(&c.plan),
            projections,
            output,
        };
        let expanded = self.make(p.origin, node, out, &[&c]);
        if include_nulls {
            return Ok(expanded);
        }
        // Spark's parser filters `coalesce(values) IS NOT NULL` above the
        // unpivot: NULL values are excluded by default.
        let rels = [&expanded];
        let filter_cx = Self::cx(&expanded.out, &rels, env, "Filter");
        let coalesce = Expr::new(ExprNode::Function(FunctionCall {
            name: "coalesce".to_owned(),
            qualifier: Vec::new(),
            args: value_columns
                .iter()
                .map(|name| {
                    Arc::new(
                        Expr::new(ExprNode::Column(ColumnRef {
                            name: vec![name.clone()],
                            plan_id: None,
                        }))
                        .with_origin(p.origin),
                    )
                })
                .collect(),
            arg_names: Vec::new(),
            distinct: false,
            filter: None,
        }))
        .with_origin(p.origin);
        let predicate = Expr::new(ExprNode::Function(FunctionCall {
            name: Builtin::IsNotNull.name().to_owned(),
            qualifier: Vec::new(),
            args: vec![Arc::new(coalesce)],
            arg_names: Vec::new(),
            distinct: false,
            filter: None,
        }))
        .with_origin(p.origin);
        let b = self.bind(&Arc::new(predicate), &filter_cx)?;
        boolean_filter(&b, p.origin)?;
        let node = PlanNode::Filter {
            input: Arc::clone(&expanded.plan),
            predicate: b.expr,
        };
        let mut filtered = self.make(p.origin, node, expanded.out.to_vec(), &[&expanded]);
        filtered.widen = Some(Arc::new(Widen::Through { child: expanded }));
        Ok(filtered)
    }
}

/// The `types` parameter of `UNPIVOT_VALUE_DATA_TYPE_MISMATCH`: every value
/// grouped by data type, as Spark formats them.
fn unpivot_types(sets: &[Vec<Bound>], names: &[Vec<String>]) -> String {
    let mut by_type: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (members, labels) in sets.iter().zip(names) {
        for (b, name) in members.iter().zip(labels) {
            let ty =
                b.ty.data_type
                    .as_ref()
                    .map(DataType::sql)
                    .unwrap_or_else(|| "VOID".to_owned());
            by_type
                .entry(format!("\"{ty}\""))
                .or_default()
                .push(quoted(std::slice::from_ref(name)));
        }
    }
    by_type
        .into_iter()
        .map(|(ty, mut values)| {
            values.sort();
            values.dedup();
            if values.len() > 3 {
                values.truncate(3);
                values.push("...".to_owned());
            }
            format!("{ty} ({})", values.join(", "))
        })
        .collect::<Vec<_>>()
        .join(", ")
}
