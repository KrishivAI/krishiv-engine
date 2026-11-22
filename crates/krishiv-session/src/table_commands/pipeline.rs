// SPDX-License-Identifier: Apache-2.0
//! `CREATE MATERIALIZED VIEW`, `CREATE STREAMING TABLE`, `CREATE FLOW` and `LOAD DATA`: Spark parses them
//! and refuses to run them directly (`UNSUPPORTED_FEATURE.CREATE_*_QUERY_EXECUTION`, and for `LOAD DATA` on
//! a datasource table `_LEGACY_ERROR_TEMP_1261`), after resolving the table it names. `INSERT OVERWRITE
//! DIRECTORY ... USING <format>` runs, as Spark's `InsertIntoDataSourceDirCommand` does: the query written
//! to the directory in overwrite mode (acceptance RS-10).

use krishiv_common::error::{
    LegacyErrorTemp1261, LegacyErrorTemp2233, NotSupportedCommandWithoutHiveSupport,
    UnsupportedFeatureCreateFlowQueryExecution,
    UnsupportedFeatureCreatePipelineDatasetQueryExecution,
};
use krishiv_sql::ast::{
    CreateFlow, CreatePipelineDataset, DirectoryFormat, InsertDirectory, LoadData,
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

    /// `INSERT OVERWRITE [LOCAL] DIRECTORY [path] USING <format> [OPTIONS (...)] <query>`: the query's
    /// rows written to the directory as `df.write.format(f).options(o).mode("overwrite").save(path)`,
    /// which is what Spark 4.2.0 runs (`InsertIntoDataSourceDirCommand`, `DataSource.planForWriting`
    /// with `SaveMode.Overwrite`): the directory is replaced, the result has no columns and no rows.
    /// The parser has checked that exactly one of the literal path and `OPTIONS ('path' ...)` is given.
    ///
    /// The Hive form (`STORED AS`, `ROW FORMAT`, no format at all) and `USING hive` are
    /// `NOT_SUPPORTED_COMMAND_WITHOUT_HIVE_SUPPORT`, as in a Spark session without Hive support; a
    /// provider that is not a file format is `_LEGACY_ERROR_TEMP_2233`. `LOCAL` changes nothing: the
    /// engine's paths are local unless they name a store. Difference from Spark, which deletes the
    /// directory and then fails reading it (`FAILED_READ_FILE.FILE_NOT_EXIST`, the data lost): a
    /// query that reads the directory it overwrites gets the writer's own refusal, before anything is
    /// deleted.
    pub(crate) async fn exec_insert_directory(
        &self,
        d: &InsertDirectory,
        original: &str,
        ids: &[krishiv_sql::ast::Span],
    ) -> Result<DataFrame, KrishivError> {
        let DirectoryFormat::Provider { provider, options } = &d.format else {
            return Err(hive_directory());
        };
        let format = provider
            .parts
            .iter()
            .map(|p| p.value.as_str())
            .collect::<Vec<_>>()
            .join(".");
        if format.eq_ignore_ascii_case("hive") {
            return Err(hive_directory());
        }
        if let Some(class) = non_file_provider(&format) {
            return Err(LegacyErrorTemp2233 {
                providing_class: format!("class {class}"),
            }
            .into());
        }
        let mut options = options_of(options, original);
        let option_path = options
            .keys()
            .find(|k| k.eq_ignore_ascii_case("path"))
            .cloned()
            .and_then(|k| options.remove(&k));
        let Some(path) = d.path.clone().filter(|p| !p.is_empty()).or(option_path) else {
            // The parser refuses a statement with no path (`_LEGACY_ERROR_TEMP_0049`).
            return Err(krishiv_common::error::InternalError {
                message: "INSERT OVERWRITE DIRECTORY without a path".to_owned(),
            }
            .into());
        };
        let plan = krishiv_sql::lower_identified(
            &d.query,
            self.next_source_id(),
            self.parser_options().case_sensitive,
            ids,
        )
        .map_err(|e| e.error)?;
        let rows = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        rows.write()
            .format(&format)
            .options(options.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .mode("overwrite")
            .save(&path)
            .await?;
        self.empty_frame().await
    }
}

/// Spark's refusal of the Hive form without Hive support (`HiveOnlyCheck`).
fn hive_directory() -> KrishivError {
    NotSupportedCommandWithoutHiveSupport {
        cmd: "INSERT OVERWRITE DIRECTORY with the Hive format".to_owned(),
    }
    .into()
}

/// The class Spark names for a built-in provider that is not a file format, which
/// `InsertIntoDataSourceDirCommand` refuses (Spark 4.2.0 run).
fn non_file_provider(format: &str) -> Option<&'static str> {
    const PROVIDERS: [(&str, &str); 4] = [
        (
            "jdbc",
            "org.apache.spark.sql.execution.datasources.jdbc.JdbcRelationProvider",
        ),
        (
            "noop",
            "org.apache.spark.sql.execution.datasources.noop.NoopDataSource",
        ),
        (
            "console",
            "org.apache.spark.sql.execution.streaming.ConsoleSinkProvider",
        ),
        (
            "rate",
            "org.apache.spark.sql.execution.streaming.sources.RateStreamProvider",
        ),
    ];
    PROVIDERS
        .iter()
        .find(|(name, class)| format.eq_ignore_ascii_case(name) || format == *class)
        .map(|(_, class)| *class)
}
