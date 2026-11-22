// SPDX-License-Identifier: Apache-2.0
//! `GroupedData::pivot` (PySpark's `groupBy(...).pivot(col, values).agg(...)`): the
//! same [`PlanNode::Pivot`] the SQL `PIVOT` clause lowers to, so values, naming, casts and errors are
//! the analyzer's (`krishiv-plan` `analyzer/pivot.rs`). A child of `dataframe` so `dataframe.rs` does
//! not grow (`cargo xtask file-size`).
//!
//! The node groups by every input column neither the pivot column nor an aggregate reads, so the
//! input is first narrowed to the grouping keys, a copy of the pivot column and the columns the
//! aggregates read. The copy keeps an aggregate that reads the pivot column itself
//! (`count("course")`) from taking it out of the grouping; a grouping key an aggregate reads is
//! grouped by a hidden copy and renamed back.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::error::{InvalidUsageOfStarOrRegex, KrishivError, KrishivUnsupportedFeature};
use krishiv_plan::{ColumnRef, Expr, ExprNode, Literal, LogicalPlan, PivotValue, PlanNode, Star};

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

/// The unresolved column references under `e`, and whether a `*` is under it. Each distinct node is
/// visited once: expressions are DAGs, and a shared child would otherwise be walked once per path.
fn references(e: &Expr, out: &mut Vec<Expr>, star: &mut bool) {
    let mut seen: HashSet<*const Expr> = HashSet::new();
    let mut stack: Vec<&Expr> = vec![e];
    while let Some(e) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(e)) {
            continue;
        }
        match &e.node {
            ExprNode::Column(_) => out.push(e.clone()),
            ExprNode::Star(_) => *star = true,
            _ => {}
        }
        e.for_each_child(&mut |child| stack.push(child));
    }
}

/// The last name part of a bare or aliased column, the name its output has.
fn output_name(e: &Expr) -> Option<String> {
    match &e.node {
        ExprNode::Column(c) => c.name.last().cloned(),
        ExprNode::Alias { name, .. } => Some(name.clone()),
        _ => None,
    }
}

/// The column named `name` (one name part, so dots in it are kept).
fn named(name: &str) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Column(ColumnRef::new(vec![
        name.to_owned(),
    ]))))
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
        let mut read = Vec::new();
        let mut star = false;
        for a in &aggregates {
            references(a, &mut read, &mut star);
        }
        if star {
            // Spark's DataFrame pivot does not expand `count(*)` (SQL's PIVOT clause does).
            let error: KrishivError = InvalidUsageOfStarOrRegex {
                elem: "'*'".to_owned(),
                pretty_name: "Pivot".to_owned(),
            }
            .into();
            return DataFrame::invalid(self.df.session.clone(), Arc::clone(&self.df.plan), error);
        }
        if read
            .iter()
            .any(|r| matches!(&r.node, ExprNode::Column(c) if c.name.len() > 1))
        {
            // Narrowing the input by a dotted name would flatten `s.x` and drop the struct `s`.
            return self.refuse("a pivot aggregate that reads a nested field or qualified column");
        }
        let case_sensitive = self
            .df
            .session
            .conf()
            .get("spark.sql.caseSensitive")
            .is_some_and(|v| v.eq_ignore_ascii_case("true"));
        let fold = |n: String| if case_sensitive { n } else { n.to_lowercase() };
        let read_names: Vec<String> = read.iter().filter_map(output_name).map(fold).collect();
        if read_names.contains(&fold(PIVOT_COPY.to_owned())) {
            return self.refuse(
                "a pivot aggregate that reads a column named like the internal pivot copy",
            );
        }
        // Two references of one name from different DataFrames (a cross join's `left["v"]`, `right["v"]`)
        // cannot both survive a narrowing that keeps one column of each name.
        let source = |r: &Expr| match &r.node {
            ExprNode::Column(c) => c.plan_id.map(|id| id.0),
            _ => None,
        };
        for (i, a) in read.iter().enumerate() {
            let same_name = |b: &&Expr| output_name(b).map(fold) == output_name(a).map(fold);
            if read[..i]
                .iter()
                .filter(same_name)
                .any(|b| source(b) != source(a))
            {
                return self.refuse(
                    "a pivot aggregate that reads same-named columns of different DataFrames",
                );
            }
        }
        let names: Vec<Option<String>> = self.keys.iter().map(output_name).collect();
        let clash: Vec<bool> = names
            .iter()
            .map(|n| {
                n.as_ref()
                    .is_some_and(|n| read_names.contains(&fold(n.clone())))
            })
            .collect();
        let any_clash = clash.iter().any(|c| *c);
        if any_clash && names.iter().any(Option::is_none) {
            return self.refuse("a pivot aggregate that reads a grouping column beside an unnamed grouping expression");
        }
        let value_text: Vec<String> = p
            .values
            .iter()
            .filter_map(|v| match v.exprs.first().map(|e| &e.node) {
                Some(ExprNode::Literal(Literal::String(t))) => Some(fold(t.clone())),
                _ => None,
            })
            .collect();
        if names.iter().zip(&clash).any(|(n, c)| {
            *c && n
                .as_ref()
                .is_some_and(|n| value_text.contains(&fold(n.clone())))
        }) {
            // The restored key `h` would sit beside a pivot output named `h`, and a bare lookup of it is ambiguous.
            return self
                .refuse("a pivot value named like a grouping column an aggregate also reads");
        }
        let copy = |i: usize| format!("__krishiv_key_{i}");
        let mut kept: Vec<Arc<Expr>> = Vec::new();
        for (i, key) in self.keys.iter().enumerate() {
            if clash[i] {
                let inner = match &key.node {
                    ExprNode::Alias { expr, .. } => (**expr).clone(),
                    _ => key.clone(),
                };
                kept.push(Arc::new(inner.alias(copy(i))));
            } else {
                kept.push(Arc::new(key.clone()));
            }
        }
        kept.push(Arc::new(p.column.clone().alias(PIVOT_COPY)));
        let mut seen: Vec<String> = Vec::new();
        for r in read {
            let Some(name) = output_name(&r).map(fold) else {
                continue;
            };
            if !seen.contains(&name) {
                seen.push(name);
                kept.push(Arc::new(r));
            }
        }
        let narrowed = self.df.derive(PlanNode::Project {
            input: Arc::clone(&self.df.plan),
            exprs: kept,
        });
        let pivoted = narrowed.derive(PlanNode::Pivot {
            input: Arc::clone(&narrowed.plan),
            pivot_columns: vec![named(PIVOT_COPY)],
            values: p.values.clone(),
            aggregates,
        });
        if !any_clash {
            return pivoted;
        }
        // The keys back under their names, then the pivoted columns.
        let mut out: Vec<Arc<Expr>> = Vec::new();
        let mut except: Vec<Vec<String>> = Vec::new();
        for (i, name) in names.iter().enumerate() {
            let Some(name) = name else { continue };
            if clash[i] {
                out.push(Arc::new(Expr::clone(&named(&copy(i))).alias(name)));
                except.push(vec![copy(i)]);
            } else {
                out.push(named(name));
                except.push(vec![name.clone()]);
            }
        }
        out.push(Arc::new(Expr::new(ExprNode::Star(Star {
            qualifier: Vec::new(),
            except,
            replace: Vec::new(),
            position: None,
            regex: None,
            plan_id: None,
        }))));
        pivoted.derive(PlanNode::Project {
            input: Arc::clone(&pivoted.plan),
            exprs: out,
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
