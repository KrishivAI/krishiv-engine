// SPDX-License-Identifier: Apache-2.0
//! `GroupedData::pivot` (PySpark's `groupBy(...).pivot(col, values).agg(...)`): the
//! same [`PlanNode::Pivot`] the SQL `PIVOT` clause lowers to, so values, naming, casts and errors are
//! the analyzer's (`krishiv-plan` `analyzer/pivot.rs`). A child of `dataframe` so `dataframe.rs` does
//! not grow (`cargo xtask file-size`).
//!
//! The node groups by every input column neither the pivot column nor an aggregate reads, so the
//! input is first narrowed to the grouping keys, a copy of the pivot column and the columns the
//! aggregates read. The copy keeps an aggregate that reads the pivot column itself
//! (`count("course")`) from taking it out of the grouping.

use std::sync::Arc;

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_plan::{ColumnRef, Expr, ExprNode, LogicalPlan, PivotValue, PlanNode};

use super::{DataFrame, GroupedData, Sets};
use crate::column::Column;

/// The name the pivot column's copy has in the narrowed input (never an output column).
const PIVOT_COPY: &str = "__krishiv_pivot";

/// A pivot column and its values, set by [`GroupedData::pivot`].
#[derive(Debug, Clone)]
pub(super) struct Pivoting {
    column: Expr,
    values: Vec<PivotValue>,
}

/// The unresolved column references under `e`.
fn references(e: &Expr, out: &mut Vec<Expr>) {
    if let ExprNode::Column(_) = &e.node {
        out.push(e.clone());
    }
    e.for_each_child(&mut |child| references(child, out));
}

/// The last name part of a bare or aliased column, the name its output has.
fn output_name(e: &Expr) -> Option<String> {
    match &e.node {
        ExprNode::Column(c) => c.name.last().cloned(),
        ExprNode::Alias { name, .. } => Some(name.clone()),
        _ => None,
    }
}

impl GroupedData {
    /// Pivots on `column`: the next [`GroupedData::agg`] gives one output column per value (and
    /// aggregate), as PySpark's `pivot`. `values` are literal columns, each cast to the pivot
    /// column's type with the original value naming the output column.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit, sum};
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s
    ///     .table("t")
    ///     .group_by(["year"])
    ///     .pivot(col("course"), [lit("Java"), lit("dotNET")])
    ///     .agg([sum(col("earnings"))]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn pivot(&self, column: Column, values: impl IntoIterator<Item = Column>) -> Self {
        Self {
            pivot: Some(Pivoting {
                column: column.expr,
                values: values
                    .into_iter()
                    .map(|v| PivotValue {
                        exprs: vec![Arc::new(v.expr)],
                        alias: None,
                    })
                    .collect(),
            }),
            ..self.clone()
        }
    }

    /// The pivoted aggregation of `aggregates` ([`GroupedData::agg`] after [`GroupedData::pivot`]).
    pub(super) fn pivot_agg(&self, p: &Pivoting, aggregates: Vec<Column>) -> DataFrame {
        let aggregates: Vec<Arc<Expr>> = aggregates.into_iter().map(|c| Arc::new(c.expr)).collect();
        if !matches!(self.sets, Sets::GroupBy) {
            return self.refuse("PIVOT after ROLLUP, CUBE or GROUPING SETS");
        }
        let mut read = Vec::new();
        for a in &aggregates {
            references(a, &mut read);
        }
        let key_names: Vec<String> = self
            .keys
            .iter()
            .filter_map(output_name)
            .map(|n| n.to_lowercase())
            .collect();
        let mut seen: Vec<String> = Vec::new();
        let mut kept: Vec<Arc<Expr>> = self.keys.iter().cloned().map(Arc::new).collect();
        kept.push(Arc::new(p.column.clone().alias(PIVOT_COPY)));
        for r in read {
            let Some(name) = output_name(&r).map(|n| n.to_lowercase()) else {
                continue;
            };
            if key_names.contains(&name) {
                // The grouping would lose the key the aggregate reads.
                return self.refuse("a pivot aggregate that reads a grouping column");
            }
            if !seen.contains(&name) {
                seen.push(name);
                kept.push(Arc::new(r));
            }
        }
        let narrowed = self.df.derive(PlanNode::Project {
            input: Arc::clone(&self.df.plan),
            exprs: kept,
        });
        let pivot_column = Arc::new(Expr::new(ExprNode::Column(ColumnRef::new(vec![
            PIVOT_COPY.to_owned(),
        ]))));
        narrowed.derive(PlanNode::Pivot {
            input: Arc::clone(&narrowed.plan),
            pivot_columns: vec![pivot_column],
            values: p.values.clone(),
            aggregates,
        })
    }

    fn refuse(&self, feature: &str) -> DataFrame {
        let plan: Arc<LogicalPlan> = Arc::clone(&self.df.plan);
        let error: KrishivError = KrishivUnsupportedFeature {
            feature: feature.to_owned(),
        }
        .into();
        DataFrame::invalid(self.df.session.clone(), plan, error)
    }
}
