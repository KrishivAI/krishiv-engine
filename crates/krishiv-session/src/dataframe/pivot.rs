// SPDX-License-Identifier: Apache-2.0
//! `GroupedData::pivot` (PySpark's `groupBy(...).pivot(col, values).agg(...)`): the
//! same [`PlanNode::Pivot`] the SQL `PIVOT` clause lowers to, so values, naming, casts and errors are
//! the analyzer's (`krishiv-plan` `analyzer/pivot.rs`). A child of `dataframe` so `dataframe.rs` does
//! not grow (`cargo xtask file-size`).
//!
//! The narrowing the node needs lives in `krishiv-plan` (`grouped_pivot`), shared with Spark Connect.
//! The node groups by every input column neither the pivot column nor an aggregate reads, so the
//! input is first narrowed to the grouping keys, a copy of the pivot column and the columns the
//! aggregates read. The copy keeps an aggregate that reads the pivot column itself
//! (`count("course")`) from taking it out of the grouping; a grouping key an aggregate reads is
//! grouped by a hidden copy and renamed back.

use std::sync::Arc;

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_plan::{Expr, LogicalPlan, Origin, PivotValue};

use super::{DataFrame, GroupedData, Sets};
use crate::column::Column;

/// A pivot column and its values, set by [`GroupedData::pivot`].
#[derive(Debug, Clone)]
pub(super) struct Pivoting {
    column: Expr,
    values: Vec<PivotValue>,
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
    ///
    /// A grouping column an aggregate also reads (`sum(earnings * year)` grouped by `year`) would
    /// leave the grouping, so it is grouped by a hidden copy that is renamed back afterwards.
    pub(super) fn pivot_agg(&self, p: &Pivoting, aggregates: Vec<Column>) -> DataFrame {
        let aggregates: Vec<Arc<Expr>> = aggregates.into_iter().map(|c| Arc::new(c.expr)).collect();
        if !matches!(self.sets, Sets::GroupBy) {
            return self.refuse("PIVOT after ROLLUP, CUBE or GROUPING SETS");
        }
        let case_sensitive = self
            .df
            .session
            .conf()
            .get("spark.sql.caseSensitive")
            .is_some_and(|v| v.eq_ignore_ascii_case("true"));
        let session = self.df.session.clone();
        match krishiv_plan::grouped_pivot(
            &self.df.plan,
            &self.keys,
            &p.column,
            p.values.clone(),
            aggregates,
            case_sensitive,
            || Origin::plan(session.next_plan_id()),
        ) {
            Ok(node) => self.df.derive(node),
            Err(error) => DataFrame::invalid(session, Arc::clone(&self.df.plan), error),
        }
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
