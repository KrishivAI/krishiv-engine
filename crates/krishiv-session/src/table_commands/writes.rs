// SPDX-License-Identifier: Apache-2.0
//! Writing rows: INSERT, INSERT OVERWRITE, saveAsTable, the v2 writer and their authorization.

use super::*;

impl Session {
    /// The permanent table `name` for a command that writes it.
    pub(crate) fn table_for_write(
        &self,
        name: &TableName,
        operation: &str,
        env: &QueryEnv,
    ) -> Result<Arc<CatalogTable>, KrishivError> {
        if let Some(table) = self.state.catalog.store.table(&name.namespace, &name.name)
            && !self.temp_entry_of(name, env)
        {
            if table.view.is_none() {
                return Ok(table);
            }
            return Err(krishiv_common::error::ExpectTableNotViewNoAlternative {
                operation: format!("`{operation}`"),
                view_name: name.quoted(),
            }
            .into());
        }
        let global = name
            .namespace
            .eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE)
            && self.state.catalog.exists(
                &crate::catalog::global_key(&name.name),
                env.analysis.conf.case_sensitive,
            );
        if self.temp_entry_of(name, env) || global {
            return Err(krishiv_common::error::ExpectTableNotViewNoAlternative {
                operation: format!("`{operation}`"),
                view_name: name.quoted(),
            }
            .into());
        }
        Err(name.not_found())
    }

    /// `INSERT INTO|OVERWRITE [TABLE] name [(columns)] [BY NAME] query`, or the multi-table
    /// `FROM source INSERT ... SELECT ... [INSERT ... SELECT ...]`.
    ///
    /// A multi-table insert writes each clause's rows of the shared source (the clause reads as
    /// `FROM source SELECT ...`). Every clause is resolved and checked before the first write, so
    /// a missing table, an unknown column or a column count mismatch in any clause writes nothing,
    /// as in Spark 4.2.0; the clauses are then written in order. Spark plans them as one statement,
    /// so a failure while one runs also leaves the others unwritten; here the clauses before it
    /// stay written (`docs/concepts/known-differences.md`).
    pub(crate) async fn exec_insert(
        &self,
        i: &Insert,
        original: &str,
        ids: &[krishiv_sql::ast::Span],
    ) -> Result<DataFrame, KrishivError> {
        let clauses: Vec<InsertClause> = match &i.source {
            None => i.clauses.clone(),
            Some(source) => i.clauses.iter().map(|c| from_source(source, c)).collect(),
        };
        if clauses.len() > 1 {
            // Spark resolves every target before any clause's columns: a missing table is the
            // error even when an earlier clause names an unknown column.
            let env = self.query_env();
            for clause in &clauses {
                let name = self.table_name(&clause.name, &env)?;
                self.table_for_write(&name, "INSERT", &env)?;
            }
            for clause in &clauses {
                self.insert_clause(clause, original, ids, true).await?;
            }
        }
        for clause in &clauses {
            self.insert_clause(clause, original, ids, false).await?;
        }
        self.empty_frame().await
    }

    /// `c` checked through to the rows it would write (`dry`), or written.
    pub(super) async fn insert_clause(
        &self,
        c: &InsertClause,
        original: &str,
        ids: &[krishiv_sql::ast::Span],
        dry: bool,
    ) -> Result<(), KrishivError> {
        let env = self.query_env();
        if c.replace_where.is_some() || c.replace_on.is_some() || c.replace_using.is_some() {
            return Err(unsupported("INSERT ... REPLACE"));
        }
        let name = self.table_name(&c.name, &env)?;
        let table = self.table_for_write(&name, "INSERT", &env)?;
        let statics = match &c.partition {
            None => Vec::new(),
            Some(spec) => {
                if c.overwrite || c.if_not_exists {
                    return Err(unsupported("INSERT OVERWRITE ... PARTITION"));
                }
                self.insert_partition_values(&table, spec, original).await?
            }
        };
        // Spark resolves a scalar subquery in the rows of `INSERT ... VALUES` (an unknown column is
        // the error then) and refuses it afterwards (a plain `VALUES` table reports the cell it
        // cannot evaluate instead).
        if let krishiv_sql::ast::SetExprKind::Values(values) = &c.query.body.kind {
            let mut subqueries = Vec::new();
            values
                .rows
                .iter()
                .for_each(|r| scalar_subqueries(r, &mut subqueries));
            for query in &subqueries {
                let plan = krishiv_sql::lower_identified(
                    query,
                    self.next_source_id(),
                    self.parser_options().case_sensitive,
                    ids,
                )
                .map_err(|e| e.error)?;
                DataFrame::new(self.clone(), Arc::clone(plan.root()))
                    .analyzed_in(self.query_env())
                    .await?;
            }
            if !subqueries.is_empty() {
                return Err(
                    krishiv_common::error::UnsupportedSubqueryExpressionCategoryScalarSubqueryInValues {}
                        .into(),
                );
            }
        }
        let plan = krishiv_sql::lower_identified(
            &c.query,
            self.next_source_id(),
            self.parser_options().case_sensitive,
            ids,
        )
        .map_err(|e| e.error)?;
        let source = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        let listed: Vec<String> = c
            .columns
            .iter()
            .map(|n| match n {
                ColumnName::Ident(i) => Ok(i.value.clone()),
                _ => Err(unsupported("IDENTIFIER() column names")),
            })
            .collect::<Result<_, _>>()?;
        self.insert_rows(
            &name,
            &table,
            &source,
            &InsertTarget {
                listed: &listed,
                by_name: c.by_name,
                overwrite: c.overwrite,
                statics: &statics,
                dry,
            },
        )
        .await
    }

    /// Inserts the rows of `source` into `table` (`INSERT INTO|OVERWRITE`,
    /// `writeTo(..).append()`): `listed` names the target columns (empty:
    /// all, by position, or by name with `by_name`).
    pub(super) async fn insert_rows(
        &self,
        name: &TableName,
        table: &Arc<CatalogTable>,
        source: &DataFrame,
        target: &InsertTarget<'_>,
    ) -> Result<(), KrishivError> {
        let InsertTarget {
            listed,
            by_name,
            overwrite,
            statics,
            dry,
        } = *target;
        let env = self.query_env();
        let analysis = source.analyzed_in(self.query_env()).await?;
        if overwrite
            && crate::catalog::reads_stored_table(
                analysis.analyzed.plan.root(),
                &name.namespace,
                &name.name,
                crate::catalog::NAMESPACE,
            )
        {
            return Err(UnsupportedOverwriteTable {
                table: name.quoted(),
            }
            .into());
        }
        let output: Vec<TableColumn> = analysis
            .analyzed
            .output
            .iter()
            .map(|c| {
                TableColumn::new(
                    c.name.clone(),
                    krishiv_plan::without_generated_marker(&c.data_type),
                    c.nullable,
                )
            })
            .collect();
        // The query provides every column but the static partition columns.
        let assigned: CatalogTable = if statics.is_empty() {
            (**table).clone()
        } else {
            let mut t = (**table).clone();
            t.columns
                .retain(|c| !statics.iter().any(|(k, _)| *k == c.name));
            t
        };
        let assignment = assign(
            &assigned,
            listed,
            by_name,
            &output,
            env.analysis.conf.case_sensitive,
        )?;
        let renamed: Vec<String> = (0..output.len())
            .map(|i| format!("__krishiv_c{i}"))
            .collect();
        let base = source.to_df(renamed.iter());
        let mut values = Vec::with_capacity(table.columns.len());
        let mut assigned_columns = assigned.columns.iter().zip(&assignment);
        for column in &table.columns {
            if let Some((_, value)) = statics.iter().find(|(k, _)| *k == column.name) {
                let value = if value == krishiv_io::partition::DEFAULT_PARTITION {
                    lit_null().cast(column.data_type.clone())
                } else {
                    crate::functions::lit(value.as_str()).cast_ansi(column.data_type.clone())
                };
                values.push((value, column));
                continue;
            }
            let Some((column, from)) = assigned_columns.next() else {
                return Err(internal("an INSERT lost a column".to_owned()));
            };
            let value = match from {
                Source::Position(i) => {
                    let value = col(&renamed[*i]);
                    if output[*i].data_type == column.data_type {
                        value
                    } else {
                        // Store assignment is ANSI whatever `spark.sql.ansi.enabled` says.
                        value.cast_ansi(column.data_type.clone())
                    }
                }
                Source::Default => match &column.default {
                    Some(text) => expr(text)?.cast_ansi(column.data_type.clone()),
                    None => lit_null().cast(column.data_type.clone()),
                },
            };
            values.push((value, column));
        }
        // The values are computed in one projection and checked in the next, so a check that reads its value
        // several times never evaluates a non-deterministic default (or query column) more than once.
        let staged: Vec<String> = (0..values.len())
            .map(|i| format!("__krishiv_v{i}"))
            .collect();
        let computed = base.select(
            values
                .iter()
                .zip(&staged)
                .map(|((value, _), name)| value.clone().alias(name))
                .collect::<Vec<_>>(),
        );
        let mut select = Vec::with_capacity(values.len());
        for ((_, column), name) in values.iter().zip(&staged) {
            let checked = crate::char_write::checked(col(name), column, &env.analysis.conf)?;
            select.push(checked.alias(&column.name));
        }
        let df = computed.select(select);
        if dry {
            // Every check of the write, without writing.
            return self.check_table_write(table, &df, overwrite).await;
        }
        let mode = if overwrite { "overwrite" } else { "append" };
        // A RENAME or DROP PARTITION waits until these rows are written and their partitions registered (it could
        // otherwise move or drop a partition between this write's log commit and its registration).
        let _partitions = if table.is_partitioned() {
            Some(
                self.state
                    .catalog
                    .store
                    .partition_write(&table.location)
                    .await,
            )
        } else {
            None
        };
        let written = self.write_table_rows(table, &df, mode, &mut None).await;
        // Data may have changed even when the write failed late.
        self.state.catalog.store.bump();
        self.forget_column_stats(name);
        self.forget_cached(name);
        let dirs =
            written.map_err(|e| overflow_in_table_insert(e, &assigned, &output, &assignment))?;
        // The partitions written (`AlterTableAddPartitionCommand` after an insert).
        self.register_partitions(name, &dirs, false);
        Ok(())
    }

    /// The access check the SQL path makes for the statement a table write
    /// stands for: `CREATE TABLE ... AS SELECT` (`Create` in the namespace) or
    /// `INSERT INTO` (`Modify` on the table). The DataFrame writers reach
    /// the command path without a statement, so the equivalent one is parsed
    /// and asked about through the same `authorize_statement`; the table name
    /// is quoted part by part, so it cannot add to the statement. The reads of
    /// the DataFrame are checked by its own analysis.
    pub(super) async fn authorize_table_write(
        &self,
        name: &TableName,
        create: bool,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        let text = if create {
            "CREATE TABLE {} AS SELECT 1"
        } else {
            "INSERT INTO {} SELECT 1"
        };
        self.authorize_equivalent(name, text, env).await
    }

    /// [`Self::authorize_table_write`] for any statement `text` (with `{}`
    /// standing for the quoted table name) the call stands for.
    pub(super) async fn authorize_equivalent(
        &self,
        name: &TableName,
        text: &str,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        let quoted = name.quoted_in_session_catalog();
        let text = text.replace("{}", &quoted);
        let statement = krishiv_sql::parse_statement(&text, &env.parser).map_err(|e| e.error)?;
        let class = krishiv_sql::classify(&statement);
        self.authorize_statement(&statement, class, env).await
    }

    /// What a `create` of a new table checks before it changes anything,
    /// so an overwrite can check it before it drops the old one.
    pub(super) fn validate_new_table(
        &self,
        name: &TableName,
        spec: &V2WriteSpec,
    ) -> Result<(), KrishivError> {
        if !spec.partitioned_by.is_empty() || !spec.cluster_by.is_empty() {
            return Err(unsupported("partitioned tables"));
        }
        self.warehouse_dir()?;
        self.provider_of(spec.provider.clone())?;
        check_new_name(&name.namespace)?;
        check_new_name(&name.name)?;
        self.check_namespace(name)?;
        // Before an overwrite drops the old table: the create below would refuse a collated schema too.
        self.refuse_collated_namespace(name)?;
        if spec.options.values().any(|v| v.as_plain().is_none()) {
            return Err(unsupported("secret table options"));
        }
        Ok(())
    }

    /// The table a `DataFrameWriter` call names.
    pub(super) fn writer_table_name(
        &self,
        table: &str,
        env: &QueryEnv,
    ) -> Result<TableName, KrishivError> {
        let parsed = krishiv_sql::parse_multipart_identifier(table, &self.parser_options())
            .map_err(|e| e.error)?;
        let parts: Vec<String> = parsed.parts.iter().map(|p| p.value.clone()).collect();
        TableName::resolve(&parts, &self.namespace_for_stored(&parts, env)?, env)
    }

    /// `DataFrameWriter.saveAsTable`: Spark's V1 save modes over a table. The
    /// table is created from the DataFrame's schema (an existing one in
    /// `errorifexists` mode fails with `TABLE_OR_VIEW_ALREADY_EXISTS`),
    /// `ignore` leaves an existing table alone, `append` adds the rows by
    /// column name (names that do not match the table's, a format other than
    /// its, or a `partition_by` fail with `KRISHIV_UNSUPPORTED_FEATURE`), and
    /// `overwrite` drops the table and creates it again, after the checks that
    /// can refuse the new table and the `UNSUPPORTED_OVERWRITE.TABLE` check
    /// (a query that reads the table it replaces); a temporary view of the
    /// same name makes it `KRISHIV_UNSUPPORTED_FEATURE`.
    pub(crate) async fn exec_save_as_table(
        &self,
        df: &DataFrame,
        spec: V2WriteSpec,
        mode: krishiv_io::SaveMode,
    ) -> Result<(), KrishivError> {
        use krishiv_io::SaveMode;
        let env = self.query_env();
        let name = self.writer_table_name(&spec.table, &env)?;
        let exists = self
            .state
            .catalog
            .store
            .table(&name.namespace, &name.name)
            .is_some();
        match (mode, exists) {
            (SaveMode::Ignore, true) => {
                // A no-op create is still authorized, as the SQL path does, so a denial
                // cannot tell an existing table from a missing one.
                self.authorize_table_write(&name, true, &env).await
            }
            (SaveMode::Append, true) => {
                // Asked first, as `insertInto` does: what is reported about a table the caller
                // may not write must not depend on its metadata.
                self.authorize_table_write(&name, false, &env).await?;
                let table = self.table_for_write(&name, "INSERT", &env)?;
                // Spark refuses an append whose format or partitioning differs
                // from the table's.
                if !spec.partitioned_by.is_empty() {
                    return Err(unsupported("partitioned tables"));
                }
                if spec
                    .provider
                    .as_deref()
                    .is_some_and(|p| !p.eq_ignore_ascii_case(&table.provider))
                {
                    return Err(unsupported(
                        "saveAsTable append with a format other than the table's",
                    ));
                }
                let schema = df.schema().await?;
                let case_sensitive = env.analysis.conf.case_sensitive;
                let same = |a: &str, b: &str| {
                    if case_sensitive {
                        a == b
                    } else {
                        a.eq_ignore_ascii_case(b)
                    }
                };
                let mut order = Vec::with_capacity(table.columns.len());
                for c in &table.columns {
                    let mut hits = schema
                        .fields()
                        .iter()
                        .map(|f| f.name().as_str())
                        .filter(|h| same(h, &c.name));
                    match (hits.next(), hits.next()) {
                        (Some(h), None) => order.push(h.to_owned()),
                        _ => break,
                    }
                }
                if order.len() != table.columns.len() || order.len() != schema.fields().len() {
                    return Err(unsupported(
                        "saveAsTable append with column names that differ from the table's",
                    ));
                }
                // Built directly: a name with dots or a star is one plain column.
                let by_name = df.select(order.iter().map(|h| {
                    crate::column::Column::new(krishiv_plan::Expr::new(
                        krishiv_plan::ExprNode::Column(krishiv_plan::ColumnRef::new(vec![
                            h.clone(),
                        ])),
                    ))
                }));
                self.exec_write_v2(&by_name, spec, V2Mode::Append, &mut None)
                    .await
            }
            (SaveMode::Overwrite, true) => {
                // Spark drops the table and creates it again. Everything that can
                // refuse the new table is checked before the old one goes.
                if self.temp_entry_of(&name, &env) {
                    return Err(unsupported(
                        "saveAsTable overwrite of a name a temporary view shadows",
                    ));
                }
                self.authorize_equivalent(&name, "DROP TABLE {}", &env)
                    .await?;
                self.authorize_table_write(&name, true, &env).await?;
                // `UNSUPPORTED_OVERWRITE.TABLE`: the query reads the table it replaces.
                let analysis = df.analysis().await?;
                if crate::catalog::reads_stored_table(
                    analysis.analyzed.plan.root(),
                    &name.namespace,
                    &name.name,
                    &self.state.catalog.current_namespace(),
                ) {
                    return Err(krishiv_common::error::UnsupportedOverwriteTable {
                        table: name.quoted(),
                    }
                    .into());
                }
                self.validate_new_table(&name, &spec)?;
                // The write refuses an indeterminate string column: not after the drop.
                crate::char_write::refuse_indeterminate_schema(&analysis.analyzed)?;
                // What the write itself would refuse (a format the engine does not
                // write, duplicate output columns, options the format rejects) must
                // not come after the drop.
                let provider = self.provider_of(spec.provider.clone())?;
                let schema = df.schema().await?;
                self.check_write_ready(&provider, &spec.options, &schema, &env)?;
                // The write asks the policy about the table's directory, in the form
                // `write_files` asks it (`locate_write_path`), and the create refuses
                // a directory that is already there other than the old table's own;
                // both before the drop.
                let location =
                    managed_location(&self.warehouse_dir()?, &name.namespace, &name.name);
                let at =
                    self.locate_write_path(&location.display().to_string(), &spec.options, &env)?;
                self.authorize_write_path(&at, &spec.options).await?;
                // A managed table's own directory goes with it; any other one in the way stays.
                let own = self
                    .state
                    .catalog
                    .store
                    .table(&name.namespace, &name.name)
                    .is_some_and(|t| t.managed && t.location == location);
                if location.exists() && !own {
                    return Err(LocationAlreadyExists {
                        identifier: name.quoted(),
                        location: location.display().to_string(),
                    }
                    .into());
                }
                // What is left after this point is the write itself (query errors, I/O): as in
                // Spark's V1 overwrite, a failure there leaves no table.
                // Admitted before the drop: a controller that refuses the write, or a caller that
                // gives up while it queues, must not have lost the old table. The write takes
                // this admission instead of asking again.
                let mut pre = Some(self.admit_query(&analysis.env, false).await?);
                let done = async {
                    // From the drop on, a cancel waits for the write's query
                    // instead of dropping the statement half-way (`action.rs`).
                    crate::action::commit_point();
                    self.drop_named_table(&name, &env, true).await?;
                    self.exec_write_v2(df, spec, V2Mode::Create, &mut pre).await
                }
                .await;
                // Not taken by a write (the drop or the create failed first): settle its row. A
                // statement dropped meanwhile drops `pre`, which closes its row `cancelled`.
                if let Some(unrun) = pre.take() {
                    match &done {
                        Ok(()) => unrun.finish_ok(),
                        Err(error) => unrun.fail(error),
                    }
                }
                done
            }
            // `ErrorIfExists` on an existing table fails in the create.
            _ => {
                self.exec_write_v2(df, spec, V2Mode::Create, &mut None)
                    .await
            }
        }
    }

    /// `DataFrameWriter.insertInto`: `errorifexists`, `ignore` and `append`
    /// are all `INSERT INTO` by position (Spark's `insertInto` ignores the
    /// first two); `overwrite` is `INSERT OVERWRITE` by position, as the SQL
    /// statement runs it.
    pub(crate) async fn exec_insert_into(
        &self,
        df: &DataFrame,
        table: &str,
        mode: krishiv_io::SaveMode,
    ) -> Result<(), KrishivError> {
        let overwrite = mode == krishiv_io::SaveMode::Overwrite;
        let env = self.query_env();
        let name = self.writer_table_name(table, &env)?;
        self.authorize_equivalent(
            &name,
            if overwrite {
                "INSERT OVERWRITE TABLE {} SELECT 1"
            } else {
                "INSERT INTO {} SELECT 1"
            },
            &env,
        )
        .await?;
        let found = self.table_for_write(
            &name,
            if overwrite {
                "INSERT OVERWRITE"
            } else {
                "INSERT"
            },
            &env,
        )?;
        let target = InsertTarget {
            listed: &[],
            by_name: false,
            overwrite,
            statics: &[],
            dry: false,
        };
        self.insert_rows(&name, &found, df, &target).await
    }

    /// Runs a [`DataFrameWriterV2`](crate::DataFrameWriterV2) request: `Create`
    /// is `CREATE TABLE name AS SELECT` over `df` (one code path, a failed
    /// write leaves no table), `Append` is `INSERT INTO name` by position.
    /// `Replace` and `CreateOrReplace` fail as `REPLACE TABLE AS SELECT` does
    /// on file-source tables; overwrites and partitioned or clustered tables
    /// are unsupported, as in `INSERT` and `CREATE TABLE`.
    pub(crate) async fn exec_write_v2(
        &self,
        df: &DataFrame,
        spec: V2WriteSpec,
        mode: V2Mode,
        pre: &mut Option<Admitted>,
    ) -> Result<(), KrishivError> {
        let env = self.query_env();
        let name = self.writer_table_name(&spec.table, &env)?;
        self.authorize_table_write(&name, mode != V2Mode::Append, &env)
            .await?;
        match mode {
            // An append ignores the layout the builder was given: the
            // existing table's layout rules.
            V2Mode::Append => {
                let table = self.table_for_write(&name, "INSERT", &env)?;
                let target = InsertTarget {
                    listed: &[],
                    by_name: false,
                    overwrite: false,
                    statics: &[],
                    dry: false,
                };
                return self.insert_rows(&name, &table, df, &target).await;
            }
            V2Mode::Replace | V2Mode::CreateOrReplace => {
                return Err(UnsupportedFeatureTableOperation {
                    table_name: name.quoted(),
                    operation: "REPLACE TABLE AS SELECT".to_owned(),
                }
                .into());
            }
            V2Mode::Create => {}
        }
        if !spec.partitioned_by.is_empty() || !spec.cluster_by.is_empty() {
            return Err(unsupported("partitioned tables"));
        }
        let warehouse = self.warehouse_dir()?;
        let provider_written = self.written_provider(spec.provider.clone());
        let provider = self.provider_of(spec.provider)?;
        check_new_name(&name.namespace)?;
        check_new_name(&name.name)?;
        self.check_namespace(&name)?;
        if self
            .state
            .catalog
            .store
            .table(&name.namespace, &name.name)
            .is_some()
        {
            return Err(krishiv_common::error::TableOrViewAlreadyExists {
                relation_name: name.quoted(),
            }
            .into());
        }
        let mut options = BTreeMap::new();
        for (k, v) in &spec.options {
            // Table metadata is stored as plain text: a secret never is.
            let Some(text) = v.as_plain() else {
                return Err(unsupported("secret table options"));
            };
            options.insert(k.clone(), text.to_owned());
        }
        let table = CatalogTable {
            namespace: name.namespace.clone(),
            name: name.name.clone(),
            managed: true,
            provider,
            provider_written,
            location: managed_location(&warehouse, &name.namespace, &name.name),
            columns: Vec::new(),
            options,
            properties: spec.properties,
            comment: None,
            created_ms: now_ms(),
            root: warehouse,
            view: None,
            layout: TableLayout::default(),
            column_stats: BTreeMap::new(),
            incarnation: 0,
            data_version: 0,
        };
        let layout = LayoutRequest {
            partition_by: &[],
            bucket: None,
        };
        self.create_table_from(table, df, CreateMode::ErrorIfExists, &layout, pre)
            .await
    }

    /// Writes `df` to the files of `table` with the writer's save `mode`, in partition directories when the
    /// table is partitioned, and returns the partition directories written. Bucketed files are not written:
    /// a bucketed table refuses the write.
    pub(super) async fn write_table_rows(
        &self,
        table: &CatalogTable,
        df: &DataFrame,
        mode: &str,
        pre: &mut Option<Admitted>,
    ) -> Result<Vec<String>, KrishivError> {
        table_writer(table, df, mode)?
            .save_reporting(&table.location.display().to_string(), pre.take())
            .await
    }

    /// The checks [`Self::write_table_rows`] makes before it writes, without writing.
    pub(super) async fn check_table_write(
        &self,
        table: &CatalogTable,
        df: &DataFrame,
        overwrite: bool,
    ) -> Result<(), KrishivError> {
        let mode = if overwrite { "overwrite" } else { "append" };
        table_writer(table, df, mode)?
            .check_save(&table.location.display().to_string())
            .await
    }
}
