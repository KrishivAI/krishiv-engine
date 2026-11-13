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
        let warehouse = self.warehouse_dir()?;
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
        let name = self.table_name(&t.name, &env)?;
        self.refuse_collated_namespace(&name)?;
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
            return self.empty_frame().await;
        } else {
            // Before anything is written (a CTAS writes first).
            return Err(krishiv_common::error::TableOrViewAlreadyExists {
                relation_name: name.quoted(),
            }
            .into());
        }
        let mut options = options_of(refuse_secret_options(&t.options, original)?, original);
        let properties = options_of(&t.tblproperties, original);
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
            Some(l) => PathBuf::from(l.strip_prefix("file:").unwrap_or(l)),
            None => managed_location(&warehouse, &name.namespace, &name.name),
        };
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
        };
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
                table.columns = self.columns_of(t, original)?;
                let mut typed = Vec::with_capacity(typed_partitions.len());
                for c in &typed_partitions {
                    let ColumnName::Ident(ident) = &c.name else {
                        return Err(unsupported("IDENTIFIER() column names"));
                    };
                    typed.push(ident);
                }
                // A typed partition column is a column of the table's schema: checked with the others.
                check_column_name_duplication(
                    table
                        .columns
                        .iter()
                        .map(|c| c.name.as_str())
                        .chain(typed.iter().map(|i| i.value.as_str())),
                    self.query_env().analysis.conf.case_sensitive,
                )?;
                for (c, ident) in typed_partitions.iter().zip(typed) {
                    table.columns.push(StoreColumn::declared(
                        ident.value.clone(),
                        &c.data_type,
                        None,
                        c.comment.clone(),
                    )?);
                }
                if table.columns.is_empty() {
                    return Err(unsupported("CREATE TABLE without columns"));
                }
                krishiv_io::check_declared_schema(
                    &table.provider,
                    &arrow_schema::Schema::new(
                        table
                            .columns
                            .iter()
                            .map(|c| c.data_type.to_arrow_field(&c.name, true))
                            .collect::<Vec<_>>(),
                    ),
                )?;
                table.layout = crate::partitions::table_layout(
                    &name,
                    &mut table.columns,
                    layout.partition_by,
                    layout.bucket,
                    env.analysis.conf.case_sensitive,
                )?;
                self.add_table(table, mode, false).await?;
            }
        }
        self.empty_frame().await
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
        if PROVIDERS.contains(&lower.as_str()) {
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
        let location = table.location.clone();
        self.authorize_path(&location, &[Privilege::Create], StatementClass::Ddl)
            .await?;
        // Spark creates the directory of a managed table only; an external table's location is
        // made by its first write.
        if table.managed {
            std::fs::create_dir_all(&location)
                .map_err(|e| internal(format!("cannot create {}: {e}", location.display())))?;
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
        if table.managed {
            self.authorize_path(
                &table.location,
                &[Privilege::Modify, Privilege::Create],
                StatementClass::Write,
            )
            .await?;
        }
        let owned = if table.managed {
            Some(
                OwnedDirectory::create(table.location.clone()).map_err(|e| match e {
                    OwnedDirectoryError::Exists => KrishivError::from(LocationAlreadyExists {
                        identifier: quoted(&table.namespace, &table.name),
                        location: table.location.display().to_string(),
                    }),
                    OwnedDirectoryError::Io(message) => internal(message),
                })?,
            )
        } else {
            None
        };
        let write_mode = if owned.is_some() {
            "append"
        } else {
            "errorifexists"
        };
        let dirs = self.write_table_rows(&table, &df, write_mode, pre).await?;
        if let Some(owned) = owned {
            owned.keep();
        }
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
            return Err(source.not_found());
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
            Some(l) => PathBuf::from(l.strip_prefix("file:").unwrap_or(l)),
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
        };
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
            self.state.cache.uncache_table(&name.name);
        } else if let Some(table) = self.state.catalog.store.table(&name.namespace, &name.name) {
            if table.view.is_some() {
                return Err(WrongCommandForObjectType {
                    operation: "DROP TABLE".to_owned(),
                    required_type: "EXTERNAL or MANAGED".to_owned(),
                    object_name: format!("{}.{}", name.namespace, name.name),
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
            return Err(name.not_found());
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
        let table = self.table_for_write(&name, "ANALYZE TABLE", &env)?;
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
        let name = self.table_name(table, &env)?;
        self.table_for_write(&name, operation, &env)?;
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
            let sql = format!("SELECT {written} FROM {}", name.quoted());
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
        let mut sql = format!("SELECT {} FROM {}", select.join(", "), name.quoted());
        if let Some(alias) = alias {
            sql.push_str(&format!(" AS `{}`", alias.value.replace('`', "``")));
        }
        if let Some(condition) = changes.condition {
            sql.push_str(&format!(" WHERE {}", slice(condition)));
        }
        // Spark 4.2 does not resolve a running script's local variables in these statements (the
        // record shows `UNRESOLVED_COLUMN` for one), so the analysis does not see them either.
        self.with_locals(Vec::new())
            .analyze_variable_query(&sql)
            .await?;
        Err(UnsupportedFeatureTableOperation {
            table_name: name.quoted(),
            operation: operation.to_owned(),
        }
        .into())
    }

    /// Drops the cached relations that read the permanent table `name`: its data or existence changed.
    pub(super) fn forget_cached(&self, name: &TableName) {
        self.state
            .cache
            .invalidate_table(&format!("{}.{}", name.namespace, name.name));
    }
}
