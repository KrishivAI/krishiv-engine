// SPDX-License-Identifier: Apache-2.0
//! `CREATE MATERIALIZED VIEW`, `CREATE STREAMING TABLE`, `CREATE FLOW` and `LOAD DATA`: Spark parses them
//! and refuses to run them directly (`UNSUPPORTED_FEATURE.CREATE_*_QUERY_EXECUTION`, and for `LOAD DATA` on
//! a datasource table `_LEGACY_ERROR_TEMP_1261`), after resolving the table it names.

use krishiv_common::error::{
    LegacyErrorTemp1261, UnsupportedFeatureCreateFlowQueryExecution,
    UnsupportedFeatureCreatePipelineDatasetQueryExecution,
};
use krishiv_sql::ast::{
    CreateFlow, CreatePipelineDataset, DirectoryFormat, ExprKind, InsertDirectory, Literal,
    LoadData,
};

use super::*;

impl Session {
    /// A pipeline dataset is created by a pipeline, never by a direct statement.
    ///
    /// Spark raises `INTERNAL_ERROR` for `CREATE STREAMING TABLE st` without a query; here it is this
    /// refusal, like the form with a query.
    pub(crate) fn exec_create_pipeline_dataset(
        c: &CreatePipelineDataset,
    ) -> Result<DataFrame, KrishivError> {
        Err(UnsupportedFeatureCreatePipelineDatasetQueryExecution {
            pipeline_dataset_type: c.kind.name().to_owned(),
        }
        .into())
    }

    /// A flow is created by a pipeline. Its target is resolved first, as Spark's analysis does
    /// (`TABLE_OR_VIEW_NOT_FOUND` for a missing one).
    pub(crate) fn exec_create_flow(&self, f: &CreateFlow) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        for clause in &f.insert.clauses {
            let name = self.table_name(&clause.name, &env)?;
            self.table_for_write(&name, "INSERT", &env)?;
        }
        Err(UnsupportedFeatureCreateFlowQueryExecution {}.into())
    }

    /// `LOAD DATA` resolves its table (a missing one is `TABLE_OR_VIEW_NOT_FOUND`, a view
    /// `EXPECT_TABLE_NOT_VIEW`); every table here is a datasource table, which Spark refuses.
    pub(crate) fn exec_load_data(&self, l: &LoadData) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.table_name(&l.table, &env)?;
        self.table_for_write(&name, "LOAD DATA", &env)?;
        Err(LegacyErrorTemp1261 {
            table_ident_with_d_b: name.quoted(),
        }
        .into())
    }

    /// `INSERT OVERWRITE [LOCAL] DIRECTORY` with a data source (`USING provider [OPTIONS (..)]`): the
    /// query's rows written to the path as `df.write.format(provider).mode("overwrite")` writes them,
    /// with the writer's own access checks and the statement's path check already made by `sql`. The
    /// Hive form (`ROW FORMAT`, `STORED AS`) is refused by name.
    pub(crate) async fn exec_insert_directory(
        &self,
        d: &InsertDirectory,
        ids: &[krishiv_sql::ast::Span],
    ) -> Result<DataFrame, KrishivError> {
        let DirectoryFormat::Provider { provider, options } = &d.format else {
            return Err(unsupported(
                "INSERT OVERWRITE DIRECTORY with ROW FORMAT or STORED AS",
            ));
        };
        let path = match d.option_path() {
            Some(p) if !p.is_empty() => p.to_owned(),
            _ => d.path.clone().unwrap_or_default(),
        };
        let format = provider
            .parts
            .last()
            .map(|p| p.value.clone())
            .unwrap_or_default();
        let plan = krishiv_sql::lower_identified(
            &d.query,
            self.next_source_id(),
            self.parser_options().case_sensitive,
            ids,
        )
        .map_err(|e| e.error)?;
        let source = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        let mut writer = source.write().format(&format).mode("overwrite");
        for option in options {
            let key = option
                .key
                .parts
                .iter()
                .map(|p| p.value.as_str())
                .collect::<Vec<_>>()
                .join(".");
            if key.eq_ignore_ascii_case("path") {
                continue;
            }
            let value = match option.value.as_ref().map(|e| &e.kind) {
                Some(ExprKind::Literal(Literal::String(s))) => s.clone(),
                Some(ExprKind::Literal(Literal::Boolean(b))) => b.to_string(),
                Some(ExprKind::Literal(Literal::Number { text, .. })) => text.clone(),
                _ => return Err(unsupported("INSERT OVERWRITE DIRECTORY option values")),
            };
            writer = writer.option(&key, &value);
        }
        writer.save(&path).await?;
        self.empty_frame().await
    }
}
