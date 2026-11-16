// SPDX-License-Identifier: Apache-2.0
//! The analysis of a `MERGE INTO` before anything runs (`merge.rs`), following Spark 4.2's
//! parser checks and delta-spark 4.4.1's `ResolveDeltaMergeInto` and `PreprocessTableMerge`
//! (`tests/fixtures/delta-4.4.1/merge/MERGE.json` records each class):
//!
//! - Only the last clause of a kind may omit its condition: Spark's
//!   `NON_LAST_MATCHED_CLAUSE_OMIT_CONDITION`, `NON_LAST_NOT_MATCHED_BY_TARGET_CLAUSE_OMIT_CONDITION`
//!   and `NON_LAST_NOT_MATCHED_BY_SOURCE_CLAUSE_OMIT_CONDITION` (42613).
//! - The `ON` condition and `MATCHED` clauses resolve over the target and the source, `NOT
//!   MATCHED` clauses over the source only, `NOT MATCHED BY SOURCE` clauses over the target only
//!   (`ResolveDeltaMergeInto.scala:417-440`); an assignment key over the target. A name that does
//!   not resolve is the analyzer's error, which the table may raise as its own
//!   ([`RowLevelProblem::Unresolved`]: a Delta table raises delta-spark 4.4.1's `INTERNAL_ERROR`
//!   "Cannot find main error class 'DELTA_MERGE_UNRESOLVED_EXPRESSION'").
//! - `UPDATE SET *` and `INSERT *` assign every target column the source column of its name
//!   (`ResolveDeltaMergeInto.scala:285-305`); an extra source column is ignored.
//! - Conditions are deterministic (`DELTA_NON_DETERMINISTIC_FUNCTION_NOT_SUPPORTED`) and hold no
//!   subquery (`DELTA_UNSUPPORTED_SUBQUERY`; `PreprocessTableMerge.scala:66-85`).
//! - An `UPDATE` sets a column once (`DELTA_CONFLICT_SET_COLUMN`), an `INSERT` names it once
//!   (`DELTA_DUPLICATE_COLUMNS_ON_INSERT`, `PreprocessTableMerge.scala:136-162`); a column an
//!   `INSERT` omits is `NULL`.
//! - Each new value's type may be stored in its column (Spark's ANSI store assignment, as an
//!   `UPDATE` checks it).
//!
//! - Under `spark.databricks.delta.schema.autoMerge.enabled` a `MERGE` that would add a column
//!   is refused (schema evolution is not built); `*` takes the target columns the source has.
//!
//! The table raises its own classes for these ([`RowLevelTable::error`]); without one the
//! planner raises Spark's (the analyzer's) or refuses. The conditions' text in the messages is as written, where delta-spark
//! prints the resolved expression.

use std::sync::Arc;

use krishiv_common::error::{
    DeltaConflictSetColumn, DeltaDuplicateColumnsOnInsert,
    IncompatibleDataForTableCannotSafelyCast, KrishivError, KrishivUnsupportedFeature,
    NonLastMatchedClauseOmitCondition, NonLastNotMatchedBySourceClauseOmitCondition,
    NonLastNotMatchedByTargetClauseOmitCondition,
};
use krishiv_exec::planner::{RowLevelOperation, RowLevelProblem, RowLevelTable};
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::DataType;

use super::merge::{ClauseAction, ClauseKind, MergeClause, Resolved};
use super::row_level::ident;
use crate::dataframe::{DataFrame, JoinType};
use crate::functions::expr;

/// delta-spark's `usageReference` of `DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW_IN_MERGE`.
pub(super) const USAGE_REFERENCE: &str =
    "https://docs.delta.io/latest/delta-update.html#upsert-into-a-table-using-merge";

/// Only the last clause of each kind may omit its condition.
pub(super) fn clause_order(clauses: &[MergeClause]) -> Result<(), KrishivError> {
    for kind in [
        ClauseKind::Matched,
        ClauseKind::NotMatched,
        ClauseKind::NotMatchedBySource,
    ] {
        let of: Vec<&MergeClause> = clauses.iter().filter(|c| c.kind == kind).collect();
        let early = of
            .iter()
            .take(of.len().saturating_sub(1))
            .any(|c| c.condition.is_none());
        if early {
            return Err(match kind {
                ClauseKind::Matched => NonLastMatchedClauseOmitCondition {}.into(),
                ClauseKind::NotMatched => NonLastNotMatchedByTargetClauseOmitCondition {}.into(),
                ClauseKind::NotMatchedBySource => {
                    NonLastNotMatchedBySourceClauseOmitCondition {}.into()
                }
            });
        }
    }
    Ok(())
}

/// What the checks read.
pub(super) struct Checks<'a> {
    pub(super) target: &'a str,
    pub(super) source_alias: &'a str,
    pub(super) columns: &'a [TableColumn],
    pub(super) source_columns: &'a [String],
    /// The target, which raises its own classes ([`RowLevelTable::error`]).
    pub(super) table: &'a Arc<dyn RowLevelTable>,
    /// Schema evolution is on (`spark.databricks.delta.schema.autoMerge.enabled`).
    pub(super) evolve: bool,
}

/// The new values' types per assigned column, for the store-assignment overflow error.
pub(super) type ValueTypes = Vec<(String, DataType)>;

impl Checks<'_> {
    /// An analysis error of a clause's names as the table raises it.
    fn unresolved(&self, e: KrishivError) -> KrishivError {
        if !e.error_class().name().starts_with("UNRESOLVED_COLUMN") {
            return e;
        }
        self.table
            .error(RowLevelOperation::Merge, &RowLevelProblem::Unresolved(&e))
            .unwrap_or(e)
    }

    /// The refusal of a schema change, which needs schema evolution.
    fn evolution(&self, column: &str) -> KrishivError {
        KrishivUnsupportedFeature {
            feature: format!(
                "MERGE INTO with schema evolution (adding the column {})",
                ident(column)
            ),
        }
        .into()
    }

    /// A name that is no target column.
    fn no_column(&self, name: &str) -> KrishivError {
        self.unresolved(KrishivError::from(
            krishiv_common::error::UnresolvedColumnWithSuggestion {
                object_name: ident(name),
                proposal: self
                    .columns
                    .iter()
                    .map(|c| ident(&c.name))
                    .collect::<Vec<_>>()
                    .join(", "),
            },
        ))
    }

    /// The deterministic and subquery checks of a condition (`what`: delta-spark's name of it).
    fn condition(
        &self,
        text: &str,
        analysed: &krishiv_plan::Expr,
        what: &str,
    ) -> Result<(), KrishivError> {
        let refuse = |problem: RowLevelProblem<'_>, feature: &str| -> KrishivError {
            self.table
                .error(RowLevelOperation::Merge, &problem)
                .unwrap_or_else(|| {
                    KrishivUnsupportedFeature {
                        feature: feature.to_owned(),
                    }
                    .into()
                })
        };
        if !krishiv_plan::is_deterministic(analysed) {
            return Err(refuse(
                RowLevelProblem::NonDeterministic {
                    what,
                    condition: text,
                },
                "MERGE INTO with a non-deterministic condition",
            ));
        }
        if has_subquery(&expr(text)?.expr) {
            return Err(refuse(
                RowLevelProblem::Subquery {
                    what,
                    condition: text,
                },
                "MERGE INTO with a subquery in a condition",
            ));
        }
        Ok(())
    }

    /// The analysed expressions of `texts` over `frame`.
    async fn analyse(
        &self,
        frame: &DataFrame,
        texts: &[String],
    ) -> Result<(Vec<krishiv_plan::Expr>, Vec<DataType>), KrishivError> {
        if texts.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let probe = frame.select_expr(texts)?;
        let analysis = probe
            .action_analysis()
            .await
            .map_err(|e| self.unresolved(e))?;
        let exprs = match &analysis.analyzed.plan.root().node {
            krishiv_plan::PlanNode::Project { exprs, .. } => {
                exprs.iter().map(|e| (**e).clone()).collect()
            }
            _ => Vec::new(),
        };
        let types = analysis
            .analyzed
            .output
            .iter()
            .map(|c| c.data_type.clone())
            .collect();
        Ok((exprs, types))
    }

    /// The `ON` condition over the target and the source; the join the clauses of both read.
    pub(super) async fn on(
        &self,
        target: &DataFrame,
        source: &DataFrame,
        on: &str,
    ) -> Result<DataFrame, KrishivError> {
        let both = target.join(source, expr("true")?, JoinType::Inner);
        let (exprs, _) = self.analyse(&both, &[on.to_owned()]).await?;
        if let Some(e) = exprs.first() {
            self.condition(on, e, "search condition")?;
        }
        Ok(target.join(source, expr(on)?, JoinType::Inner))
    }

    /// Each clause resolved: its condition checked, its values per target column.
    pub(super) async fn clauses(
        &self,
        clauses: &[MergeClause],
        target: &DataFrame,
        source: &DataFrame,
        both: &DataFrame,
    ) -> Result<(Vec<Resolved>, ValueTypes), KrishivError> {
        let mut out = Vec::with_capacity(clauses.len());
        let mut value_types: ValueTypes = Vec::new();
        for clause in clauses {
            let frame = match clause.kind {
                ClauseKind::Matched => both,
                ClauseKind::NotMatched => source,
                ClauseKind::NotMatchedBySource => target,
            };
            let mut values: Vec<Option<String>> = vec![None; self.columns.len()];
            // `*`: each target column from the source column of its name. Under schema evolution
            // a target column the source lacks keeps its value (`UPDATE`) or is `NULL`
            // (`INSERT`), and a source column the target lacks would be added: refused
            // (`ResolveDeltaMergeInto.scala:285-345`).
            let star = || -> Result<Vec<Option<String>>, KrishivError> {
                if self.evolve
                    && let Some(extra) = self
                        .source_columns
                        .iter()
                        .find(|s| !self.columns.iter().any(|c| c.name.eq_ignore_ascii_case(s)))
                {
                    return Err(self.evolution(extra));
                }
                self.columns
                    .iter()
                    .map(|c| {
                        match self
                            .source_columns
                            .iter()
                            .find(|s| s.eq_ignore_ascii_case(&c.name))
                        {
                            Some(name) => Ok(Some(format!(
                                "{}.{}",
                                ident(self.source_alias),
                                ident(name)
                            ))),
                            None if self.evolve => Ok(None),
                            None => Err(self.no_column(&c.name)),
                        }
                    })
                    .collect()
            };
            let (delete, insert) = match &clause.action {
                ClauseAction::Delete => (true, false),
                ClauseAction::UpdateStar => {
                    values = star()?;
                    (false, false)
                }
                ClauseAction::InsertStar => {
                    values = star()?;
                    (false, true)
                }
                ClauseAction::Update(assignments) | ClauseAction::Insert(assignments) => {
                    let insert = matches!(clause.action, ClauseAction::Insert(_));
                    for a in assignments {
                        if a.column.contains('.') {
                            return Err(KrishivUnsupportedFeature {
                                feature: format!("MERGE INTO of the nested field {}", a.column),
                            }
                            .into());
                        }
                        let Some(i) = self
                            .columns
                            .iter()
                            .position(|c| c.name.eq_ignore_ascii_case(&a.column))
                        else {
                            // Under schema evolution a key naming a source column adds it.
                            if self.evolve
                                && self
                                    .source_columns
                                    .iter()
                                    .any(|s| s.eq_ignore_ascii_case(&a.column))
                            {
                                return Err(self.evolution(&a.column));
                            }
                            return Err(self.no_column(&a.column));
                        };
                        let Some(value) = &a.value else {
                            return Err(KrishivUnsupportedFeature {
                                feature: format!("MERGE INTO with {} = DEFAULT", a.column),
                            }
                            .into());
                        };
                        if values[i].is_some() {
                            return Err(if insert {
                                DeltaDuplicateColumnsOnInsert {}.into()
                            } else {
                                let repeated: Vec<String> = assignments
                                    .iter()
                                    .filter(|b| {
                                        assignments
                                            .iter()
                                            .filter(|c| c.column.eq_ignore_ascii_case(&b.column))
                                            .count()
                                            > 1
                                    })
                                    .map(|b| ident(&b.column))
                                    .collect();
                                DeltaConflictSetColumn {
                                    column_list: format!("[{}]", repeated.join(", ")),
                                }
                                .into()
                            });
                        }
                        values[i] = Some(value.clone());
                    }
                    (false, insert)
                }
            };
            let mut texts: Vec<String> = values.iter().flatten().cloned().collect();
            if let Some(c) = &clause.condition {
                texts.push(c.clone());
            }
            let (exprs, types) = self.analyse(frame, &texts).await?;
            if let (Some(c), Some(e)) = (&clause.condition, exprs.last()) {
                let what = match clause.action {
                    ClauseAction::Delete => "DELETE condition",
                    ClauseAction::Update(_) | ClauseAction::UpdateStar => "UPDATE condition",
                    _ => "INSERT condition",
                };
                self.condition(c, e, what)?;
            }
            let assigned = self
                .columns
                .iter()
                .zip(&values)
                .filter(|(_, v)| v.is_some())
                .map(|(c, _)| c);
            for (column, out_type) in assigned.zip(&types) {
                value_types.push((column.name.clone(), out_type.clone()));
                if !super::can_store(out_type, &column.data_type) {
                    return Err(IncompatibleDataForTableCannotSafelyCast {
                        table_name: self.target.to_owned(),
                        col_name: ident(&column.name),
                        src_type: format!("\"{}\"", out_type.catalog_string().to_uppercase()),
                        target_type: format!(
                            "\"{}\"",
                            column.data_type.catalog_string().to_uppercase()
                        ),
                    }
                    .into());
                }
            }
            out.push(Resolved {
                kind: clause.kind,
                condition: clause.condition.clone(),
                delete,
                insert,
                values,
            });
        }
        Ok((out, value_types))
    }
}

/// Whether `e` holds a subquery anywhere.
fn has_subquery(e: &krishiv_plan::Expr) -> bool {
    let mut stack: Vec<&krishiv_plan::Expr> = vec![e];
    while let Some(x) = stack.pop() {
        if matches!(x.node, krishiv_plan::ExprNode::Subquery { .. }) {
            return true;
        }
        x.for_each_child(&mut |c| stack.push(c));
    }
    false
}
