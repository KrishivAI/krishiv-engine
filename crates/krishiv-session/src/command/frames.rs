// SPDX-License-Identifier: Apache-2.0
//! The result frames of the commands (`SET`, `SHOW`, `DESCRIBE`, `EXPLAIN`): literal rows under the column
//! tables of `analyze::command_schema`, so each column carries Spark 4.2.0's nullability.

use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{LogicalPlan, PlanNode};
use krishiv_types::DataType;

use crate::analyze::command_schema::CommandColumn;
use crate::dataframe::DataFrame;
use crate::functions::{col, lit, lit_null};
use crate::session::Session;

impl Session {
    /// A frame of the string columns `columns` with `rows` (`None` is
    /// NULL), as Spark's `SET` commands return. Rows come out ordered by
    /// `key`, as in Spark.
    ///
    /// Each column is as nullable as `columns` says (the tables of
    /// `analyze::command_schema`, Spark 4.2.0's), whatever the rows hold: a
    /// nullable column gets an empty branch of typed NULLs, and a NULL in a
    /// non-null column is an internal error rather than a frame whose
    /// schema says otherwise.
    pub(crate) async fn string_frame(
        &self,
        columns: &[CommandColumn],
        rows: Vec<Vec<Option<String>>>,
    ) -> Result<DataFrame, KrishivError> {
        for row in &rows {
            if let Some(c) = columns
                .iter()
                .zip(row)
                .find(|(c, v)| !c.nullable && v.is_none())
            {
                return Err(crate::session::internal(format!(
                    "a NULL in the non-null command column {}",
                    c.0.name
                )));
            }
        }
        let one = || {
            DataFrame::new(
                self.clone(),
                Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
            )
        };
        let row = |values: &[Option<String>]| {
            one().select(
                columns
                    .iter()
                    .zip(values.iter())
                    .map(|(c, value)| match value {
                        Some(value) => lit(value.as_str()).alias(c.name),
                        None => lit_null().cast(DataType::String).alias(c.name),
                    }),
            )
        };
        // No rows: one placeholder row with the schema, limited to nothing. Its
        // nullable columns are NULL, so a union with it reads them as nullable
        // (a union's column is nullable where any input's is).
        let placeholder: Vec<Option<String>> = columns
            .iter()
            .map(|c| (!c.nullable).then(String::new))
            .collect();
        let mut inputs: Vec<Arc<LogicalPlan>> =
            rows.iter().map(|r| Arc::clone(&row(r).plan)).collect();
        if inputs.is_empty() || columns.iter().any(|c| c.nullable) {
            inputs.push(Arc::clone(&row(&placeholder).limit(0).plan));
        }
        let df = if inputs.len() == 1 {
            let only = inputs.pop().ok_or_else(|| {
                crate::session::internal("a command frame without rows".to_owned())
            })?;
            DataFrame::new(self.clone(), only)
        } else {
            DataFrame::new(
                self.clone(),
                Arc::new(LogicalPlan::new(PlanNode::Union {
                    inputs,
                    by_name: false,
                    allow_missing_columns: false,
                })),
            )
        };
        // Rows come out ordered by key: unions read in any order, so the
        // frame sorts.
        let df = if columns.iter().any(|c| c.name == "key") {
            df.order_by([col("key")])
        } else {
            df // callers without a `key` column pass at most one row
        };
        let env = self.query_env();
        df.analyzed_in(env).await?;
        Ok(df)
    }
}
