// SPDX-License-Identifier: Apache-2.0
//! `CREATE MATERIALIZED VIEW`, `CREATE STREAMING TABLE`, `CREATE FLOW` and `LOAD DATA`: Spark parses them
//! and refuses to run them directly (`UNSUPPORTED_FEATURE.CREATE_*_QUERY_EXECUTION`, and for `LOAD DATA` on
//! a datasource table `_LEGACY_ERROR_TEMP_1261`), after resolving the table it names.

use krishiv_common::error::{
    LegacyErrorTemp1261, UnsupportedFeatureCreateFlowQueryExecution,
    UnsupportedFeatureCreatePipelineDatasetQueryExecution,
};
use krishiv_sql::ast::{CreateFlow, CreatePipelineDataset, LoadData};

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
}
