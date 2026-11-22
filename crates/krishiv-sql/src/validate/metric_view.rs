// SPDX-License-Identifier: Apache-2.0
//! The checks `SparkSqlParser.visitCreateMetricView` makes on `CREATE VIEW ... WITH METRICS` (P3
//! 4.2-MV).

use krishiv_common::error::{DuplicateClauses, LegacyErrorTemp0035};

use super::{Checker, R, at, ddl_label};
use crate::ast::CreateMetricView;
use crate::error;

impl Checker<'_> {
    /// `SparkSqlParser.visitCreateMetricView`'s order: a duplicate clause, `OR REPLACE` with `IF
    /// NOT EXISTS`, then the language and the properties, then the name. A missing `WITH METRICS`
    /// or `LANGUAGE` stops the checks here: its error depends on the Spark release (4.2 has no
    /// `MISSING_CLAUSES_FOR_OPERATION`), which the parser does not know, so the session raises it.
    pub(super) fn create_metric_view(&self, v: &CreateMetricView) -> R {
        if let Some(clause) = v.duplicate_clause {
            return Err(at(
                DuplicateClauses {
                    clause_name: clause.to_owned(),
                },
                v.span,
            ));
        }
        if v.if_not_exists && v.or_replace {
            return Err(error::create_view_with_if_not_exists_and_replace(
                ddl_label(&v.name, self.sql),
                v.span,
            ));
        }
        let (Some(language), Some(_)) = (&v.language, &v.with_metrics) else {
            return Ok(());
        };
        if !language.value.eq_ignore_ascii_case("YAML") {
            // `SQL` is a keyword token: Spark names it by the constant, any other name as written.
            let shown = if language.value.eq_ignore_ascii_case("SQL") {
                "SQL"
            } else {
                &language.value
            };
            return Err(at(
                LegacyErrorTemp0035 {
                    message: format!("Unsupported language for metric view: {shown}"),
                },
                v.span,
            ));
        }
        self.property_list(&v.tblproperties, true, v.span)?;
        v.tblproperties
            .iter()
            .filter_map(|o| o.value.as_ref())
            .try_for_each(|e| self.expr(e))?;
        self.ddl_name(&v.name)?;
        v.columns.iter().try_for_each(|c| self.column_name(&c.name))
    }
}
