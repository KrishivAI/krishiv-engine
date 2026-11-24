// SPDX-License-Identifier: Apache-2.0
//! `DataFrame`'s temporary views (`createTempView` and its three siblings). A child of `dataframe`
//! so that it reads the frame's private fields and keeps `dataframe.rs` under the file-size limit
//! (`cargo xtask file-size`).

use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::TableColumn;
use krishiv_plan::{PlanNode, ReadSource};

use super::DataFrame;
use crate::catalog::TempTable;

impl DataFrame {
    /// Registers this DataFrame as the temporary view `name`, replacing any
    /// table or view of that name (PySpark's `createOrReplaceTempView`).
    /// A view over a read by path lists its files again at every query;
    /// any other view inlines its plan as a subquery wherever it is read.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let e = futures::executor::block_on(s.table("t").create_or_replace_temp_view("v")).unwrap_err();
    /// assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis errors, and `KRISHIV_RESERVED_TEMP_VIEW_NAME` when `name`
    /// starts with the reserved snapshot prefix `` `__krishiv_snapshot_` ``
    /// or the cache prefix `` `__krishiv_cache_` `` (in any letter case),
    /// which [`Session::sql`] cannot spell either.
    pub async fn create_or_replace_temp_view(&self, name: &str) -> Result<(), KrishivError> {
        self.temp_view(name, false, true).await
    }

    /// Registers this frame as the temporary view `name` (PySpark's `createTempView`). It fails with
    /// `TEMP_TABLE_OR_VIEW_ALREADY_EXISTS` when the view exists.
    ///
    /// # Errors
    /// As [`Self::create_or_replace_temp_view`], and `TEMP_TABLE_OR_VIEW_ALREADY_EXISTS`.
    pub async fn create_temp_view(&self, name: &str) -> Result<(), KrishivError> {
        self.temp_view(name, false, false).await
    }

    /// Registers this frame as the global temporary view `global_temp.name`, kept in this
    /// session's catalog, as `CREATE GLOBAL TEMPORARY VIEW` keeps it (PySpark's `createGlobalTempView`). It
    /// fails with `TEMP_TABLE_OR_VIEW_ALREADY_EXISTS` when the view exists.
    ///
    /// # Errors
    /// As [`Self::create_or_replace_temp_view`], and `TEMP_TABLE_OR_VIEW_ALREADY_EXISTS`.
    pub async fn create_global_temp_view(&self, name: &str) -> Result<(), KrishivError> {
        self.temp_view(name, true, false).await
    }

    /// Registers this frame as the global temporary view `global_temp.name`, replacing one of that
    /// name (PySpark's `createOrReplaceGlobalTempView`).
    ///
    /// # Errors
    /// As [`Self::create_or_replace_temp_view`].
    pub async fn create_or_replace_global_temp_view(&self, name: &str) -> Result<(), KrishivError> {
        self.temp_view(name, true, true).await
    }

    async fn temp_view(&self, name: &str, global: bool, replace: bool) -> Result<(), KrishivError> {
        let (df, name) = (self.clone(), name.to_owned());
        self.session
            .on_executor(async move { df.temp_view_here(&name, global, replace).await })
            .await
    }

    async fn temp_view_here(
        &self,
        view: &str,
        global: bool,
        replace: bool,
    ) -> Result<(), KrishivError> {
        // Spark parses the name as a multipart identifier: a parse failure is
        // `_LEGACY_ERROR_TEMP_1321`, more than one part is
        // `TEMP_VIEW_NAME_TOO_MANY_NAME_PARTS`, and backquotes are stripped.
        let parts = krishiv_sql::parse_multipart_identifier(view, &self.session.query_env().parser)
            .map_err(|_| {
                KrishivError::from(krishiv_common::error::LegacyErrorTemp1321 {
                    view_name: view.to_owned(),
                })
            })?
            .parts;
        let [part] = parts.as_slice() else {
            return Err(krishiv_common::error::TempViewNameTooManyNameParts {
                actual_name: view.to_owned(),
            }
            .into());
        };
        let view = part.value.as_str();
        if crate::catalog::is_reserved_view_name(view) {
            return Err(crate::catalog::reserved_temp_view_name(view));
        }
        let key = if global {
            crate::catalog::global_key(view)
        } else {
            view.to_owned()
        };
        let name = key.as_str();
        // Spark analyzes the view's plan first: its errors come first.
        let analysis = self.analysis().await?;
        // After the analysis, as Spark: a bad query is reported before an existing name.
        if !replace
            && self
                .session
                .state
                .catalog
                .exists(name, self.session.query_env().analysis.conf.case_sensitive)
        {
            let shown = if global {
                format!("`{}`.`{view}`", crate::catalog::GLOBAL_NAMESPACE)
            } else {
                format!("`{view}`")
            };
            return Err(krishiv_common::error::TempTableOrViewAlreadyExists {
                relation_name: shown,
            }
            .into());
        }
        let columns: Vec<TableColumn> =
            crate::char_write::refuse_indeterminate_schema(&analysis.analyzed)?
                .output
                .iter()
                .map(|c| TableColumn::new(c.name.clone(), c.data_type.clone(), c.nullable))
                .collect();
        // A view over files keeps listing them: files added or changed
        // later show up in the next query.
        if let PlanNode::Read(source @ ReadSource::Files { .. }) = &self.plan.node {
            self.session.state.catalog.put(
                name,
                TempTable::Files {
                    source: source.clone(),
                    columns,
                    known: false,
                    permanent: false,
                    partitions: Some(Default::default()),
                },
            );
            self.session.state.cache.forget_table(view);
            return Ok(());
        }
        // Any other view snapshots its input: the analyzed plan is stored,
        // with the temporary tables it reads pinned under hidden names, so
        // a view built from itself reads the old definition, as in Spark.
        // (SQL `CREATE VIEW` keeps the unresolved query instead, so it
        // follows the tables it reads.)
        let (expanded, pins, new_pins) = crate::catalog::pin_temp_tables(
            &self.session.state.catalog,
            analysis.analyzed.plan.root(),
            analysis.env.analysis.conf.case_sensitive,
        )?;
        self.session.state.catalog.put_view(
            name,
            expanded,
            columns,
            pins,
            new_pins,
            // A query that reads the view runs its observed nodes: it completes the same
            // observations. Published with the entry, under the same lock.
            crate::catalog::ViewKeeps(self.observation_handles_with(&analysis.env.observers)),
        );
        self.session.state.cache.forget_table(view);
        Ok(())
    }
}
