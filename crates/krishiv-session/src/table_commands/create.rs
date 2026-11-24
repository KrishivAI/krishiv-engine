// SPDX-License-Identifier: Apache-2.0
//! Creating, dropping, truncating and analyzing tables, and row-level changes.

use super::*;

impl Session {
    /// `CREATE [OR REPLACE] [EXTERNAL] TABLE ...`, with or without columns
    /// and a query.
    pub(crate) async fn exec_create_table(
        &self,
        t: &CreateTable,
        original: &str,
        ids: &[krishiv_sql::ast::Span],
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        #[cfg(feature = "delta")]
        if let Some(path) = self.delta_path_target(t, &env)? {
            self.create_delta_path_table(t, &path, original).await?;
            return self.empty_frame().await;
        }
        let warehouse = self.warehouse_dir()?;
        let Some(new) = self.new_table(t, &env)? else {
            return self.empty_frame().await;
        };
        let NewTable {
            name,
            mode,
            partition_names,
            typed_partitions,
            provider,
            provider_written,
        } = new;
        let mut options = options_of(refuse_secret_options(&t.options, original)?, original);
        // Stored store options redirect every later read of the table: a server-scoped key
        // (endpoint, region, account, key file) is `Manage` on the cluster, asked before the
        // table is stored (P3-T067b); readers are not asked again.
        self.authorize_store_options(
            krishiv_sql::StatementClass::Ddl,
            &self.state.conf.snapshot(),
            options.keys().map(String::as_str),
        )
        .await?;
        let properties = options_of(&t.tblproperties, original);
        #[cfg(feature = "delta")]
        let located =
            t.location.is_some() || options.keys().any(|k| k.eq_ignore_ascii_case("path"));
        let (external, location) = new_table_location(t, &mut options, &warehouse, &name)?;
        let mut table = CatalogTable {
            namespace: name.namespace.clone(),
            name: name.name.clone(),
            managed: !external,
            provider,
            provider_written,
            location,
            columns: Vec::new(),
            options,
            properties,
            comment: t.comment.clone(),
            created_ms: now_ms(),
            root: warehouse.clone(),
            view: None,
            layout: TableLayout::default(),
            column_stats: BTreeMap::new(),
            incarnation: 0,
            data_version: 0,
            stamp: 0,
        };
        #[cfg(feature = "delta")]
        if is_delta_table(&table) && t.as_query.is_none() {
            // delta-spark compares the whole declared schema, typed partition columns and
            // `NOT NULL` included, with the log's.
            let mut declared = self.declared_columns(t, original, &typed_partitions)?;
            super::delta::keep_not_null(&mut declared, t, &typed_partitions);
            self.create_delta_table(table, t, &declared, &name, &partition_names, mode, located)
                .await?;
            return self.empty_frame().await;
        }
        let layout = LayoutRequest {
            partition_by: &partition_names,
            bucket: t.bucket.as_ref(),
        };
        match &t.as_query {
            Some(query) => {
                self.create_table_as(table, query, mode, &layout, ids)
                    .await?;
            }
            None => {
                table.columns = self.declared_columns(t, original, &typed_partitions)?;
                // `CREATE TABLE t USING fmt LOCATION 'dir'` without columns: Spark reads the schema, and the partition
                // columns of `key=value` directories, from the files (SQL-19).
                let discovered: Vec<String> = if table.columns.is_empty() {
                    check_inferable(&layout, external, &table.provider_written)?;
                    let unable = || -> KrishivError {
                        krishiv_common::error::UnableToInferSchema {
                            format: table.provider_written.clone(),
                        }
                        .into()
                    };
                    let location = table.location.to_string_lossy().into_owned();
                    // Spark ignores a location that does not exist and finds no files to infer from.
                    table.columns = match self
                        .infer_file_columns(&table.provider, &[location.as_str()], &table.options)
                        .await
                    {
                        Ok(columns) => columns,
                        Err(e)
                            if e.error_class()
                                == krishiv_common::error::ErrorClass::PathNotFound =>
                        {
                            return Err(unable());
                        }
                        Err(e) => return Err(e),
                    };
                    if table.columns.is_empty() {
                        return Err(unable());
                    }
                    // The partition columns of `key=value` directories (hidden and `_`-prefixed entries are not).
                    crate::partition_keys::partition_keys(&table.location)?
                } else {
                    Vec::new()
                };
                let partition_by: &[String] = if discovered.is_empty() {
                    layout.partition_by
                } else {
                    &discovered
                };
                table.layout = layout_of(
                    &name,
                    &table.provider,
                    &mut table.columns,
                    partition_by,
                    layout.bucket,
                    env.analysis.conf.case_sensitive,
                )?;
                self.add_table(table, mode, false).await?;
            }
        }
        self.empty_frame().await
    }

    /// The checks of a `CREATE TABLE` up to the location of the new table, which change nothing: the
    /// name, provider and layout clauses Krishiv refuses, the namespace, `OR REPLACE` (refused), and
    /// whether the table exists (`TABLE_OR_VIEW_ALREADY_EXISTS`; `None` with `IF NOT EXISTS`, the
    /// statement does nothing). The run's first steps, and analyze's (`questions.rs`).
    pub(super) fn new_table<'t>(
        &self,
        t: &'t CreateTable,
        env: &QueryEnv,
    ) -> Result<Option<NewTable<'t>>, KrishivError> {
        if t.temporary {
            return Err(unsupported("CREATE TEMPORARY TABLE"));
        }
        if t.file_format.is_some() || t.row_format.is_some() {
            return Err(NotSupportedCommandWithoutHiveSupport {
                cmd: "CREATE Hive TABLE (AS SELECT)".to_owned(),
            }
            .into());
        }
        if t.skew.is_some() || !t.cluster_by.is_empty() {
            return Err(unsupported("skewed or clustered tables"));
        }
        let (partition_names, typed_partitions) = partition_fields(&t.partitioned_by)?;
        if t.as_query.is_some() && !typed_partitions.is_empty() {
            return Err(unsupported(
                "partition column types in CREATE TABLE ... AS SELECT",
            ));
        }
        let using = t.using.as_ref().map(|u| {
            u.parts
                .iter()
                .map(|p| p.value.as_str())
                .collect::<Vec<_>>()
                .join(".")
        });
        let provider_written = self.written_provider(using.clone());
        let provider = self.provider_of(using)?;
        let name = self.table_name(&t.name, env)?;
        self.refuse_collated_namespace(&name)?;
        // DeltaCatalog asks for the namespace before the session catalog checks the name
        // (`CREATE-PATH.json` `relative_path`, `no_slash`: `SCHEMA_NOT_FOUND`).
        #[cfg(feature = "delta")]
        if provider.eq_ignore_ascii_case(crate::delta::DELTA)
            && !self.state.catalog.store.has_namespace(&name.namespace)
        {
            return Err(SchemaNotFound {
                schema_name: format!("`spark_catalog`.`{}`", name.namespace),
            }
            .into());
        }
        check_new_name(&name.namespace)?;
        check_new_name(&name.name)?;
        self.check_namespace(&name)?;
        if t.or_replace {
            // Spark's file-source tables cannot be replaced either.
            return Err(UnsupportedFeatureTableOperation {
                table_name: name.quoted(),
                operation: if t.as_query.is_some() {
                    "REPLACE TABLE AS SELECT"
                } else {
                    "REPLACE TABLE"
                }
                .to_owned(),
            }
            .into());
        }
        let mode = if t.if_not_exists {
            CreateMode::IfNotExists
        } else {
            CreateMode::ErrorIfExists
        };
        if self
            .state
            .catalog
            .store
            .table(&name.namespace, &name.name)
            .is_none()
        {
            // A name taken by a temporary view is not a table; the table is
            // created beside it in its namespace, as in Spark.
        } else if mode == CreateMode::IfNotExists {
            return Ok(None);
        } else {
            // Before anything is written (a CTAS writes first).
            return Err(krishiv_common::error::TableOrViewAlreadyExists {
                relation_name: name.quoted(),
            }
            .into());
        }
        Ok(Some(NewTable {
            name,
            mode,
            partition_names,
            typed_partitions,
            provider,
            provider_written,
        }))
    }

    /// The declared columns of a `CREATE TABLE` without a query: its column elements, then its typed
    /// partition columns, checked for duplicates together.
    pub(super) fn declared_columns(
        &self,
        t: &CreateTable,
        original: &str,
        typed_partitions: &[&krishiv_sql::ast::PartitionColumn],
    ) -> Result<Vec<StoreColumn>, KrishivError> {
        let mut columns = self.columns_of(t, original)?;
        let mut typed = Vec::with_capacity(typed_partitions.len());
        for c in typed_partitions {
            let ColumnName::Ident(ident) = &c.name else {
                return Err(unsupported("IDENTIFIER() column names"));
            };
            typed.push(ident);
        }
        // A typed partition column is a column of the table's schema: checked with the others.
        check_column_name_duplication(
            columns
                .iter()
                .map(|c| c.name.as_str())
                .chain(typed.iter().map(|i| i.value.as_str())),
            self.query_env().analysis.conf.case_sensitive,
        )?;
        for (c, ident) in typed_partitions.iter().zip(typed) {
            columns.push(StoreColumn::declared(
                ident.value.clone(),
                &c.data_type,
                None,
                c.comment.clone(),
            )?);
        }
        Ok(columns)
    }

    /// The provider as written in `using`, or `spark.sql.sources.default`'s
    /// value: what a table's metadata prints (Spark keeps the case).
    pub(super) fn written_provider(&self, using: Option<String>) -> String {
        using.unwrap_or_else(|| {
            crate::conf::get(&self.state.conf.snapshot(), "spark.sql.sources.default")
                .unwrap_or_else(|| "parquet".to_owned())
        })
    }

    /// The provider named by `using`, or `spark.sql.sources.default`.
    pub(super) fn provider_of(&self, using: Option<String>) -> Result<String, KrishivError> {
        let provider = using.unwrap_or_else(|| {
            crate::conf::get(&self.state.conf.snapshot(), "spark.sql.sources.default")
                .unwrap_or_else(|| "parquet".to_owned())
        });
        let lower = provider.to_lowercase();
        if PROVIDERS.contains(&lower.as_str()) || crate::delta::handles(&lower) {
            Ok(lower)
        } else {
            Err(DataSourceNotFound { provider }.into())
        }
    }

    pub(super) fn check_namespace(&self, name: &TableName) -> Result<(), KrishivError> {
        if self.state.catalog.store.has_namespace(&name.namespace) {
            Ok(())
        } else {
            Err(SchemaNotFound {
                schema_name: format!("`{}`", name.namespace),
            }
            .into())
        }
    }

    /// The columns of a `CREATE TABLE` from its elements.
    pub(super) fn columns_of(
        &self,
        t: &CreateTable,
        original: &str,
    ) -> Result<Vec<StoreColumn>, KrishivError> {
        let mut columns: Vec<StoreColumn> = Vec::new();
        for element in &t.elements {
            let TableElement::Column(def) = element else {
                continue;
            };
            let ColumnName::Ident(ident) = &def.name else {
                return Err(unsupported("IDENTIFIER() column names"));
            };
            if def.generated.is_some() {
                return Err(unsupported("generated columns"));
            }
            // File-source tables are nullable whatever the DDL says (Spark 4.2 verified: NOT NULL is dropped and
            // NULLs are stored), so nothing is enforced on insert.
            let default = def.default.as_ref().map(|e| {
                original
                    .get(e.span.start..e.span.end)
                    .unwrap_or_default()
                    .to_owned()
            });
            columns.push(StoreColumn::declared(
                ident.value.clone(),
                &def.data_type,
                default,
                def.comment.clone(),
            )?);
        }
        check_column_name_duplication(
            columns.iter().map(|c| c.name.as_str()),
            self.query_env().analysis.conf.case_sensitive,
        )?;
        Ok(columns)
    }

    /// Adds `table` to the store, creating its directory first. A replaced
    /// managed table's directory is deleted after the metadata changed.
    pub(super) async fn add_table(
        &self,
        table: CatalogTable,
        mode: CreateMode,
        written: bool,
    ) -> Result<(), KrishivError> {
        let name = TableName {
            namespace: table.namespace.clone(),
            name: table.name.clone(),
            written: String::new(),
            bare: false,
        };
        if mode == CreateMode::ErrorIfExists
            && self
                .state
                .catalog
                .store
                .table(&table.namespace, &table.name)
                .is_some()
        {
            return Err(krishiv_common::error::TableOrViewAlreadyExists {
                relation_name: name.quoted(),
            }
            .into());
        }
        if !written && table.managed && table.location.exists() {
            return Err(LocationAlreadyExists {
                identifier: name.quoted(),
                location: table.location.display().to_string(),
            }
            .into());
        }
        if table.managed {
            self.allow_managed_location(&table.namespace, &table.name, &table.location);
        }
        let checked = self
            .authorize_location(&table.location, table.managed)
            .await?;
        // Spark creates the directory of a managed table only; an external table's location is
        // made by its first write. The directory made is the canonical one checked, verified level by
        // level, so a link swapped in after the check makes nothing outside it.
        if table.managed
            && let Some(checked) = &checked
        {
            self.create_checked_dir(
                &checked.path,
                &checked.shown,
                Privilege::Create,
                StatementClass::Ddl,
            )?;
        }
        // Tables are never replaced, so nothing comes back from the store to clean up.
        self.state.catalog.store.create_table(table, mode)?;
        Ok(())
    }

    /// `CREATE TABLE ... AS SELECT`: writes the query to the table's
    /// location, then adds the metadata (so a failed write leaves none).
    pub(super) async fn create_table_as(
        &self,
        table: CatalogTable,
        query: &Query,
        mode: CreateMode,
        layout: &LayoutRequest<'_>,
        ids: &[krishiv_sql::ast::Span],
    ) -> Result<(), KrishivError> {
        let plan = krishiv_sql::lower_identified(
            query,
            self.next_source_id(),
            self.parser_options().case_sensitive,
            ids,
        )
        .map_err(|e| e.error)?;
        let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        self.create_table_from(table, &df, mode, layout, &mut None)
            .await
    }

    /// Writes `df` to the location of the new `table` (whose columns are
    /// the frame's), then adds the metadata: the shared tail of `CREATE
    /// TABLE ... AS SELECT` and `writeTo(..).create()`.
    pub(super) async fn create_table_from(
        &self,
        mut table: CatalogTable,
        df: &DataFrame,
        mode: CreateMode,
        layout: &LayoutRequest<'_>,
        pre: &mut Option<Admitted>,
    ) -> Result<(), KrishivError> {
        let env = self.query_env();
        let analysis = df.analyzed_in(self.query_env()).await?;
        crate::char_write::refuse_indeterminate_schema(&analysis.analyzed)?;
        table.columns = crate::char_write::output_columns(&analysis.analyzed, Some(true));
        let name = TableName {
            namespace: table.namespace.clone(),
            name: table.name.clone(),
            written: String::new(),
            bare: false,
        };
        self.refuse_collated_namespace(&name)?;
        table.layout = crate::partitions::table_layout(
            &name,
            &mut table.columns,
            layout.partition_by,
            layout.bucket,
            env.analysis.conf.case_sensitive,
        )?;
        if table.managed && table.location.exists() {
            return Err(LocationAlreadyExists {
                identifier: quoted(&table.namespace, &table.name),
                location: table.location.display().to_string(),
            }
            .into());
        }
        // The partition columns go last, in the table's order.
        let df = if table.is_partitioned() {
            df.select(
                table
                    .columns
                    .iter()
                    .map(|c| col(&format!("`{}`", c.name.replace('`', "``")))),
            )
        } else {
            df.clone()
        };
        // A managed table's directory is made here, exclusively: that creation is the proof that this
        // statement owns it (a concurrent CTAS of the same table loses it and fails with
        // `LOCATION_ALREADY_EXISTS`), and the guard removes it again if the statement fails or is
        // cancelled, so a failed first write (and the commit log it started) leaves nothing and the
        // statement can be retried. The write goes into that empty directory (`append`).
        // An external location keeps `errorifexists`: a CTAS never replaces files already there.
        // The policy is asked about the destination first: a denied CTAS must not create the warehouse or
        // the table directory on the way to being refused (the writer asks again for the files it writes).
        let owned = if table.managed {
            self.allow_managed_location(&table.namespace, &table.name, &table.location);
            let checked = self
                .authorize_path(
                    &table.location,
                    &[Privilege::Modify, Privilege::Create],
                    StatementClass::Write,
                )
                .await?;
            // The canonical directory checked is the one made: its parents level by level, each verified,
            // then the directory itself, verified again (a link swapped in after the check makes nothing
            // that stays outside; the writer re-checks every path it writes).
            if let Some(parent) = checked.parent() {
                self.create_checked_dir(
                    &parent.path,
                    &parent.shown,
                    Privilege::Create,
                    StatementClass::Write,
                )?;
            }
            let owned = OwnedDirectory::create(checked.path.clone()).map_err(|e| match e {
                OwnedDirectoryError::Exists => KrishivError::from(LocationAlreadyExists {
                    identifier: quoted(&table.namespace, &table.name),
                    location: table.location.display().to_string(),
                }),
                OwnedDirectoryError::Io(message) => internal(message),
            })?;
            self.create_checked_dir(
                &checked.path,
                &checked.shown,
                Privilege::Create,
                StatementClass::Write,
            )?;
            Some(owned)
        } else {
            None
        };
        #[cfg(feature = "delta")]
        if is_delta_table(&table) {
            // A Delta table has no bucketing: refuse before anything is created.
            if table.layout.bucket.is_some() {
                return Err(super::delta::delta_write_refused(
                    quoted(&table.namespace, &table.name),
                    "bucketBy",
                ));
            }
            // Version 0 of a new Delta table with the rows (design section 4.1, "CTAS").
            drop(pre.take());
            let source =
                super::delta_source(&self.qualify_path(&table.location.display().to_string())?);
            // delta-spark refuses a CTAS into a location that holds anything, a Delta table
            // included (`WRITES.json` `ctas_existing_location`).
            let location = table.location.clone();
            let occupied = tokio::task::spawn_blocking(move || {
                std::fs::read_dir(&location).is_ok_and(|mut d| d.next().is_some())
            })
            .await
            .map_err(|e| internal(format!("reading a Delta location: {e}")))?;
            if !table.managed && (occupied || env.catalog.delta.exists(&source).await?) {
                let krishiv_plan::ReadSource::Files { paths, .. } = &source else {
                    return Err(internal("a Delta location that is not a path".to_owned()));
                };
                return Err(
                    krishiv_common::error::DeltaCreateTableWithNonEmptyLocation {
                        table_id: quoted(&table.namespace, &table.name),
                        table_location: paths
                            .first()
                            .and_then(|p| p.as_plain())
                            .unwrap_or_default()
                            .trim_end_matches('/')
                            .to_owned(),
                    }
                    .into(),
                );
            }
            self.write_delta(
                &df,
                super::DeltaTarget::Create {
                    source,
                    name: Some(format!("{}.{}", table.namespace, table.name)),
                    partition_by: table.layout.partition_columns.clone(),
                    properties: table.properties.clone(),
                },
                krishiv_exec::planner::WriteMode::Append,
                super::delta_writes::writer_options(&table.options),
                &quoted(&table.namespace, &table.name),
            )
            .await?;
            if let Some(owned) = owned {
                owned.keep();
            }
            // The idempotency key belongs to this one write, never to the table: stored, it would
            // deduplicate every later write of the table.
            table.options.retain(|k, _| {
                !k.eq_ignore_ascii_case(krishiv_exec::planner::IDEMPOTENCY_KEY_OPTION)
            });
            self.add_table(table, mode, true).await?;
            return Ok(());
        }
        let write_mode = if owned.is_some() {
            "append"
        } else {
            "errorifexists"
        };
        let key =
            super::explicit_key(&table.options, |v: &String| Some(v.as_str())).map(str::to_owned);
        let dirs = self
            .write_table_rows(&table, &df, write_mode, key.as_deref(), pre)
            .await?;
        if let Some(owned) = owned {
            owned.keep();
        }
        // The idempotency key belongs to this one write, never to the table: stored, it would
        // deduplicate every later write of the table.
        table
            .options
            .retain(|k, _| !k.eq_ignore_ascii_case(krishiv_exec::planner::IDEMPOTENCY_KEY_OPTION));
        self.add_table(table, mode, true).await?;
        // `RepairTableCommand`: the partitions written, with their file statistics.
        self.register_partitions(&name, &dirs, true);
        Ok(())
    }

    /// `CREATE TABLE target LIKE source`.
    pub(crate) async fn exec_create_table_like(
        &self,
        t: &CreateTableLike,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let warehouse = self.warehouse_dir()?;
        if t.file_format.is_some() || t.row_format.is_some() {
            return Err(NotSupportedCommandWithoutHiveSupport {
                cmd: "CREATE Hive TABLE (AS SELECT)".to_owned(),
            }
            .into());
        }
        let name = self.table_name(&t.name, &env)?;
        self.refuse_collated_namespace(&name)?;
        check_new_name(&name.namespace)?;
        check_new_name(&name.name)?;
        self.check_namespace(&name)?;
        let source = self.table_name(&t.source, &env)?;
        // Spark's `CreateTableLikeCommand` copies the partition columns, the bucket specification and the
        // storage options, under any provider.
        let mut layout = TableLayout::default();
        let mut options = BTreeMap::new();
        let (columns, source_provider) = if let Some(s) = self
            .state
            .catalog
            .store
            .table(&source.namespace, &source.name)
        {
            layout
                .partition_columns
                .clone_from(&s.layout.partition_columns);
            layout.bucket.clone_from(&s.layout.bucket);
            options.clone_from(&s.options);
            // A view has no provider of its own: Spark gives the new table
            // `spark.sql.sources.default` (`CreateTableLikeCommand`, `isViewLike`).
            let provider = s
                .view
                .is_none()
                .then(|| (s.provider.clone(), s.provider_written.clone()));
            (s.columns.clone(), provider)
        } else if self.temp_entry_of(&source, &env)
            && let Some(entry) = self
                .state
                .catalog
                .get(&source.name, env.analysis.conf.case_sensitive)
        {
            let columns = match &*entry {
                crate::catalog::TempTable::Memory { columns, .. }
                | crate::catalog::TempTable::Files { columns, .. }
                | crate::catalog::TempTable::Provider { columns, .. }
                | crate::catalog::TempTable::View { columns, .. } => columns,
            };
            (
                columns
                    .iter()
                    .map(|c| {
                        StoreColumn::plain(
                            c.name.clone(),
                            krishiv_plan::without_generated_marker(&c.data_type),
                            c.nullable,
                        )
                    })
                    .collect(),
                None,
            )
        } else {
            return Err(source.not_found(self.search_path()));
        };
        let (provider, provider_written) = match &t.using {
            Some(u) => {
                let using = u
                    .parts
                    .iter()
                    .map(|p| p.value.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                (self.provider_of(Some(using.clone()))?, using)
            }
            // `None`: a view, which has no provider of its own. The default
            // is looked up only here, so an explicit `USING` never depends on it.
            None => match source_provider {
                Some(p) => p,
                // Validated, and written as the setting has it (`CSV`).
                None => (self.provider_of(None)?, self.written_provider(None)),
            },
        };
        let external = t.location.is_some();
        let location = match &t.location {
            Some(l) => krishiv_exec::files::native_path(l),
            None => managed_location(&warehouse, &name.namespace, &name.name),
        };
        let table = CatalogTable {
            namespace: name.namespace.clone(),
            name: name.name.clone(),
            managed: !external,
            provider,
            provider_written,
            location,
            columns,
            options,
            properties: options_of(&t.tblproperties, original),
            comment: None,
            created_ms: now_ms(),
            root: warehouse.clone(),
            view: None,
            layout,
            column_stats: BTreeMap::new(),
            incarnation: 0,
            data_version: 0,
            stamp: 0,
        };
        #[cfg(feature = "delta")]
        if is_delta_table(&table) {
            return Err(delta_write_refused(name.quoted(), "CREATE TABLE LIKE"));
        }
        let mode = if t.if_not_exists {
            CreateMode::IfNotExists
        } else {
            CreateMode::ErrorIfExists
        };
        self.add_table(table, mode, false).await?;
        self.empty_frame().await
    }

    /// `DROP TABLE [IF EXISTS] name [PURGE]`.
    pub(crate) async fn exec_drop_table(&self, d: &DropTable) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.table_name(&d.name, &env)?;
        self.drop_named_table(&name, &env, d.if_exists).await?;
        self.empty_frame().await
    }

    /// `DROP TABLE` for a resolved name; also what `saveAsTable` in overwrite
    /// mode runs before it creates the table again.
    pub(super) async fn drop_named_table(
        &self,
        name: &TableName,
        env: &QueryEnv,
        if_exists: bool,
    ) -> Result<(), KrishivError> {
        // `DROP TABLE` on a permanent view is refused as Spark's `DropTableCommand` does.
        if self.temp_entry_of(name, env) {
            // A temporary view shadows a table of the same bare name, and `DROP TABLE` drops it, as in Spark.
            self.state.catalog.drop_table(&name.name);
            self.state.cache.invalidate_table(&name.name);
        } else if let Some(table) = self.state.catalog.store.table(&name.namespace, &name.name) {
            if table
                .view
                .as_ref()
                .is_some_and(crate::store::ViewDef::is_metric)
            {
                // Spark 4.2's `DropTableCommand` drops a metric view (4.2.1 and 4.3.0 goldens):
                // only a plain `VIEW` is refused.
                self.forget_cached(name);
                self.state
                    .catalog
                    .store
                    .drop_table(&name.namespace, &name.name);
                return Ok(());
            }
            if table.view.is_some() {
                return Err(WrongCommandForObjectType {
                    operation: "DROP TABLE".to_owned(),
                    required_type: "EXTERNAL or MANAGED".to_owned(),
                    // `CatalogTable.qualifiedName`, with the catalog (all releases).
                    object_name: format!(
                        "{}.{}.{}",
                        crate::catalog::CATALOG,
                        name.namespace,
                        name.name
                    ),
                    found_type: "VIEW".to_owned(),
                    alternative: "DROP VIEW".to_owned(),
                }
                .into());
            }
            // Checked before the metadata goes: a refused delete must not orphan the files.
            let dir = self.deletable_dir(&table).await?;
            self.forget_cached(name);
            self.state
                .catalog
                .store
                .drop_table(&name.namespace, &name.name);
            if let Some(dir) = dir {
                remove_dir(&dir)?;
            }
        } else if !if_exists {
            return Err(name.not_found(self.search_path()));
        }
        Ok(())
    }

    /// `TRUNCATE TABLE name`: removes the table's files and keeps its
    /// directory.
    pub(crate) async fn exec_truncate(&self, t: &Truncate) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        if t.partition.is_some() {
            return Err(unsupported("TRUNCATE TABLE ... PARTITION"));
        }
        let name = self.table_name(&t.table, &env)?;
        let table = self.table_for_write(&name, "TRUNCATE TABLE", &env)?;
        if !table.managed {
            return Err(KrishivUnsupportedFeature {
                feature: "TRUNCATE TABLE on an external table".to_owned(),
            }
            .into());
        }
        // Its deletes do not interleave with a rename's moves or a write's registration.
        let _ddl = if table.is_partitioned() {
            Some(
                self.state
                    .catalog
                    .store
                    .partition_ddl(&table.location)
                    .await,
            )
        } else {
            None
        };
        self.forget_cached(&name);
        let deleted = self.delete_table_files(&table, true).await;
        // Data may have changed even when the cleanup failed part way.
        self.state.catalog.store.bump();
        self.forget_cached(&name);
        self.forget_column_stats(&name);
        deleted?;
        self.empty_frame().await
    }

    /// `ANALYZE TABLE name COMPUTE STATISTICS [NOSCAN | FOR COLUMNS ...]`.
    ///
    /// The table (and the listed columns) must exist, as in Spark. Column
    /// statistics are computed and kept in the catalog; the engine keeps no
    /// table-level statistics, so the other forms only validate their target.
    pub(crate) async fn exec_analyze(
        &self,
        a: &Analyze,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let name = self.table_name(&a.table, &env)?;
        let table = self.table_for_read_command(&name, "ANALYZE TABLE", &env)?;
        if let Some(spec) = &a.partition {
            if !matches!(a.columns, AnalyzeColumns::All) {
                return Err(unsupported("ANALYZE TABLE ... PARTITION ... FOR COLUMNS"));
            }
            self.analyze_partitions(&name, &table, spec, a.noscan, original)
                .await?;
            return self.empty_frame().await;
        }
        let case_sensitive = env.analysis.conf.case_sensitive;
        let same = |x: &str, y: &str| {
            if case_sensitive {
                x == y
            } else {
                x.eq_ignore_ascii_case(y)
            }
        };
        // The columns whose statistics are asked for: the listed ones, or all.
        let chosen: Vec<&crate::store::StoreColumn> = match &a.columns {
            AnalyzeColumns::All => Vec::new(),
            AnalyzeColumns::ForAll => table.columns.iter().collect(),
            AnalyzeColumns::ForColumns(columns) => {
                let mut found = Vec::with_capacity(columns.len());
                for c in columns {
                    let Some(column) = table.columns.iter().find(|t| same(&t.name, &c.value))
                    else {
                        return Err(krishiv_common::error::ColumnNotFound {
                            col_name: format!("`{}`", c.value.replace('`', "``")),
                            case_sensitive_config: "\"spark.sql.caseSensitive\"".to_owned(),
                        }
                        .into());
                    };
                    found.push(column);
                }
                found
            }
            _ => return Err(unsupported("this ANALYZE TABLE form")),
        };
        // Spark gathers statistics of numbers, booleans, dates and times, strings and binary only.
        for column in &chosen {
            let supported = matches!(
                column.data_type,
                DataType::Boolean
                    | DataType::Byte
                    | DataType::Short
                    | DataType::Integer
                    | DataType::Long
                    | DataType::Float
                    | DataType::Double
                    | DataType::Decimal(_)
                    | DataType::String
                    | DataType::Binary
                    | DataType::Date
                    | DataType::Timestamp
                    | DataType::TimestampNtz
                    | DataType::Time { .. }
            );
            if !supported {
                return Err(
                    krishiv_common::error::UnsupportedFeatureAnalyzeUnsupportedColumnType {
                        column_type: format!("\"{}\"", column.data_type.sql()),
                        column_name: format!("`{}`", column.name.replace('`', "``")),
                        table_name: format!("`{}`.`{}`", table.namespace, table.name),
                    }
                    .into(),
                );
            }
        }
        if !chosen.is_empty() {
            self.store_column_stats(&name, &table, &chosen).await?;
        }
        self.empty_frame().await
    }

    /// `DELETE FROM`, `UPDATE` and `MERGE INTO` a file-source table: Spark's
    /// v1 tables support none of them, and says so once the table resolves.
    pub(crate) async fn exec_row_level_change(
        &self,
        table: &DdlName,
        alias: Option<&krishiv_sql::ast::Ident>,
        operation: &str,
        changes: &RowChanges<'_>,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let target = self
            .row_level_target_checked(table, operation, &env, changes, original)
            .await?;
        self.exec_row_level_change_on(target, alias, operation, changes, original)
            .await
    }

    /// [`Self::exec_row_level_change`] on a target `row_level_target_checked` already resolved
    /// and checked (`MERGE INTO` falls back here after its own check).
    pub(crate) async fn exec_row_level_change_on(
        &self,
        (name, quoted): (TableName, String),
        alias: Option<&krishiv_sql::ast::Ident>,
        operation: &str,
        changes: &RowChanges<'_>,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        // Spark analyzes the condition and the new values against the table before it finds that the
        // table cannot change: an unknown column is the error then.
        // Each assigned column is analyzed as a reference, so an unknown one is the analyzer's own
        // error (with its suggestions), not a message of ours.
        let plain = self.with_locals(Vec::new());
        for target in changes.targets {
            let written = target
                .parts
                .iter()
                .map(|p| format!("`{}`", p.value.replace('`', "``")))
                .collect::<Vec<_>>()
                .join(".");
            let sql = format!("SELECT {written} FROM {quoted}");
            plain.analyze_variable_query(&sql).await?;
        }
        let slice = |e: &krishiv_sql::ast::Expr| {
            original
                .get(e.span.start..e.span.end)
                .unwrap_or_default()
                .to_owned()
        };
        // `SET c = DEFAULT` takes the column's default: nothing to resolve.
        let mut select: Vec<String> = changes
            .values
            .iter()
            .filter(|e| !crate::command::is_bare_default(e))
            .map(|e| slice(e))
            .collect();
        if select.is_empty() {
            select.push("1".to_owned());
        }
        let mut sql = format!("SELECT {} FROM {quoted}", select.join(", "));
        if let Some(alias) = alias {
            sql.push_str(&format!(" AS `{}`", alias.value.replace('`', "``")));
        }
        if let Some(condition) = changes.condition {
            sql.push_str(&format!(" WHERE {}", slice(condition)));
        }
        // Spark 4.2 does not resolve a running script's local variables in these statements (the
        // record shows `UNRESOLVED_COLUMN` for one), so the analysis does not see them either.
        let analysis = self
            .with_locals(Vec::new())
            .analyze_variable_query(&sql)
            .await?;
        // A time travel clause in the condition or a value's subquery is refused where the relation resolves.
        analysis
            .env
            .catalog
            .check_reference_options(analysis.analyzed.plan.root())?;
        // A table whose provider offers a row-level table runs the statement (PR-DML).
        // The target's `WITH (...)` options reach the table's scan builder, and no table of the
        // session catalog takes them: the statement is refused as Spark refuses it for such a table.
        let op = match operation {
            _ if !changes.options.is_empty() => None,
            "DELETE" => Some(krishiv_exec::planner::RowLevelOperation::Delete),
            "UPDATE TABLE" => Some(krishiv_exec::planner::RowLevelOperation::Update),
            _ => None,
        };
        if let Some(op) = op {
            let alias = alias.map_or_else(|| name.name.clone(), |a| a.value.clone());
            let assignments: Vec<super::row_level::Assignment> = changes
                .targets
                .iter()
                .zip(changes.values)
                .map(|(target, value)| {
                    // `SET t.c = ..`: the qualifier names the target (its alias or table name).
                    let mut parts: Vec<&str> =
                        target.parts.iter().map(|p| p.value.as_str()).collect();
                    if parts.len() > 1
                        && (parts[0].eq_ignore_ascii_case(&alias)
                            || parts[0].eq_ignore_ascii_case(&name.name))
                    {
                        parts.remove(0);
                    }
                    super::row_level::Assignment {
                        nested: parts.len() > 1,
                        column: parts.join("."),
                        value: (!crate::command::is_bare_default(value)).then(|| slice(value)),
                    }
                })
                .collect();
            let condition = changes.condition.map(slice);
            // The statement's hidden tables go at its end, a cancelled one's too.
            let ran = super::row_level::Swept::new(
                Arc::clone(&self.state.catalog),
                self.run_row_level(op, &quoted, &alias, condition.as_deref(), &assignments),
            )
            .await;
            if let Some(done) = ran? {
                // The table's data changed: what read it is stale (as after an `INSERT`).
                self.state.catalog.store.bump();
                self.forget_column_stats(&name);
                self.forget_cached(&name);
                return Ok(done);
            }
        }
        Err(UnsupportedFeatureTableOperation {
            table_name: quoted,
            operation: operation.to_owned(),
        }
        .into())
    }

    /// The table a row-level statement changes and the name it is read under: a Delta table by
    /// path (`delta.`/path``, no catalog entry) as written, else a permanent table that may be
    /// written by `operation`. The outer error is the name's or the path's own check; the inner one
    /// is the target's refusal (`table_for_write`), which [`Self::row_level_target_checked`] orders
    /// after the subqueries' time travel check, as Spark does.
    async fn row_level_target(
        &self,
        table: &DdlName,
        operation: &str,
        env: &QueryEnv,
    ) -> Result<Result<(TableName, String), KrishivError>, KrishivError> {
        let name = self.table_name(table, env)?;
        // `delta.`/path``: a Delta table by path (no catalog entry), read as its scans read it.
        // The raw path beside its quoted form.
        #[cfg(feature = "delta")]
        let by_path = match crate::access::ddl_parts(table, env)?.as_slice() {
            [format, path]
                if format.eq_ignore_ascii_case("delta")
                    && path.contains('/')
                    && self
                        .state
                        .catalog
                        .store
                        .table(&name.namespace, &name.name)
                        .is_none() =>
            {
                // The path as written: a table name is case-folded, a path is not.
                Some((path.clone(), format!("delta.`{}`", path.replace('`', "``"))))
            }
            _ => None,
        };
        #[cfg(not(feature = "delta"))]
        let by_path: Option<(String, String)> = None;
        let quoted = match by_path {
            Some((path, quoted)) => {
                // The statement's own check named the target as a table; a table by path changes
                // the files at that path, so `Modify` on it is asked first, as a Delta save to the
                // path asks.
                #[cfg(feature = "delta")]
                {
                    let qualified = self.qualify_path(&path)?;
                    self.authorize_qualified_root(
                        &qualified,
                        &[Privilege::Modify],
                        krishiv_sql::StatementClass::Write,
                    )
                    .await?;
                }
                #[cfg(not(feature = "delta"))]
                let _ = path;
                quoted
            }
            None => {
                if let Err(refusal) = self.table_for_write(&name, operation, env) {
                    return Ok(Err(refusal));
                }
                name.quoted()
            }
        };
        Ok(Ok((name, quoted)))
    }

    /// [`Self::row_level_target`], then a time travel clause in the statement's subqueries: Spark
    /// analyzes the subqueries before it finds the target cannot change, so a time travel clause in
    /// one is the error then; a target that does not exist is reported first.
    pub(crate) async fn row_level_target_checked(
        &self,
        table: &DdlName,
        operation: &str,
        env: &QueryEnv,
        changes: &RowChanges<'_>,
        original: &str,
    ) -> Result<(TableName, String), KrishivError> {
        match self.row_level_target(table, operation, env).await? {
            Ok(target) => {
                self.refuse_time_travel_in_subqueries(changes, original)
                    .await?;
                Ok(target)
            }
            Err(refusal) => {
                if refusal.error_class().name() != "TABLE_OR_VIEW_NOT_FOUND" {
                    self.refuse_time_travel_in_subqueries(changes, original)
                        .await?;
                }
                Err(refusal)
            }
        }
    }

    /// Analyzes each subquery of a `DELETE`, `UPDATE` or `MERGE` on its own and refuses a time travel clause
    /// in it. Any other failure (a correlated column does not resolve alone) is left to the statement's own
    /// analysis.
    async fn refuse_time_travel_in_subqueries(
        &self,
        changes: &RowChanges<'_>,
        original: &str,
    ) -> Result<(), KrishivError> {
        let mut queries: Vec<&krishiv_sql::ast::Query> = changes.sources.to_vec();
        for e in changes
            .condition
            .into_iter()
            .chain(changes.values.iter().copied())
            .chain(changes.more.iter().copied())
        {
            super::all_subqueries(e, &mut queries);
        }
        let plain = self.with_locals(Vec::new());
        for q in queries {
            let Some(text) = original.get(q.span.start..q.span.end) else {
                continue;
            };
            match plain.analyze_variable_query(text).await {
                Ok(analysis) => analysis
                    .env
                    .catalog
                    .check_reference_options(analysis.analyzed.plan.root())?,
                Err(e) if e.error_class().name() == "UNSUPPORTED_FEATURE.TIME_TRAVEL" => {
                    return Err(e);
                }
                Err(_) => {}
            }
        }
        Ok(())
    }

    /// Drops the cached relations that read the permanent table `name`: its data or existence changed.
    pub(super) fn forget_cached(&self, name: &TableName) {
        self.state
            .cache
            .invalidate_table(&format!("{}.{}", name.namespace, name.name));
    }
}

/// A `CREATE TABLE` past [`Session::new_table`]'s checks: the table to add.
pub(super) struct NewTable<'t> {
    /// The new table's name.
    pub(super) name: TableName,
    /// `IF NOT EXISTS` or not.
    pub(super) mode: CreateMode,
    /// The `PARTITIONED BY` column names.
    pub(super) partition_names: Vec<String>,
    /// The typed `PARTITIONED BY` columns (declared there, not among the columns).
    pub(super) typed_partitions: Vec<&'t krishiv_sql::ast::PartitionColumn>,
    /// The provider's short name.
    pub(super) provider: String,
    /// The provider as written.
    pub(super) provider_written: String,
}

/// Whether the new table `name` is external and its location: `LOCATION`, else a `path` option (removed
/// from `options`: it is the location, as in Spark), else the managed location under `warehouse`.
///
/// # Errors
/// `INVALID_EMPTY_LOCATION` for an empty location.
pub(super) fn new_table_location(
    t: &CreateTable,
    options: &mut BTreeMap<String, String>,
    warehouse: &Path,
    name: &TableName,
) -> Result<(bool, PathBuf), KrishivError> {
    // A `path` option is the table's location, as in Spark.
    let path_key = options
        .keys()
        .find(|k| k.eq_ignore_ascii_case("path"))
        .cloned();
    let location_text = match (&t.location, path_key) {
        (Some(l), _) => Some(l.clone()),
        (None, Some(k)) => options.remove(&k),
        (None, None) => None,
    };
    let external = t.external || location_text.is_some();
    let location = match &location_text {
        Some(l) if l.is_empty() => {
            return Err(krishiv_common::error::InvalidEmptyLocation {
                location: l.clone(),
            }
            .into());
        }
        Some(l) => krishiv_exec::files::native_path(l),
        None => managed_location(warehouse, &name.namespace, &name.name),
    };
    Ok((external, location))
}

/// The refusals of a `CREATE TABLE` without columns before its schema is read from the files: Spark infers
/// the partitioning and the bucketing of such a table, so naming them is an error, and a managed table has
/// no files to read a schema from.
pub(super) fn check_inferable(
    layout: &LayoutRequest<'_>,
    external: bool,
    provider_written: &str,
) -> Result<(), KrishivError> {
    if !layout.partition_by.is_empty() {
        return Err(krishiv_common::error::SpecifyPartitionIsNotAllowed {}.into());
    }
    if layout.bucket.is_some() {
        return Err(krishiv_common::error::SpecifyBucketingIsNotAllowed {}.into());
    }
    if !external {
        return Err(krishiv_common::error::UnableToInferSchema {
            format: provider_written.to_owned(),
        }
        .into());
    }
    Ok(())
}

/// The layout of a new table with `columns` (the provider's schema check first), partitioned by
/// `partition_by` and bucketed by `bucket`.
pub(super) fn layout_of(
    name: &TableName,
    provider: &str,
    columns: &mut Vec<StoreColumn>,
    partition_by: &[String],
    bucket: Option<&krishiv_sql::ast::BucketSpec>,
    case_sensitive: bool,
) -> Result<TableLayout, KrishivError> {
    krishiv_io::check_declared_schema(
        provider,
        &arrow_schema::Schema::new(
            columns
                .iter()
                .map(|c| c.data_type.to_arrow_field(&c.name, true))
                .collect::<Vec<_>>(),
        ),
    )?;
    crate::partitions::table_layout(name, columns, partition_by, bucket, case_sensitive)
}
