// SPDX-License-Identifier: Apache-2.0
//! Delta tables in DDL (P3-T021 PR-08b part 2): `CREATE TABLE t [(columns)] USING delta
//! LOCATION 'p'` over an existing Delta table (its columns and partitioning from the log), a new
//! Delta table where there is none (PR-37a: version 0 of the declared columns; CTAS goes through
//! `create_table_from` and `delta_writes.rs`), `CREATE TEMPORARY VIEW v USING delta OPTIONS
//! (path 'p')`, and the refusals of the Delta DDL not supported yet (`ALTER`, `CREATE TABLE
//! LIKE`, `TRUNCATE`, bucketing).
//!
//! What Spark + delta-spark 4.4.1 does with each statement is recorded in
//! `tests/fixtures/delta-4.4.1/SESSION.json`: a column list must match the log
//! (`DELTA_CREATE_TABLE_SCHEME_MISMATCH`), and so must the partitioning
//! (`DELTA_CREATE_TABLE_WITH_DIFFERENT_PARTITIONING`, checked after the schema). `ALTER` succeeds
//! in Spark; Krishiv refuses it by name as `UNSUPPORTED_FEATURE.TABLE_OPERATION`. So are
//! `TBLPROPERTIES` and options other than `path`
//! on a `CREATE` over an existing table: delta-spark compares them with the log's
//! configuration, a rule no probe has pinned yet.

use super::*;
use crate::delta::DELTA;

/// The operation a refused Delta write names.
fn refused_operation(operation: &str) -> String {
    format!("{operation} on a Delta table (Delta writes and DDL are not supported yet)")
}

/// The refusal of `operation` on the Delta table `table` (quoted).
pub(crate) fn delta_write_refused(table: String, operation: &str) -> KrishivError {
    UnsupportedFeatureTableOperation {
        table_name: table,
        operation: refused_operation(operation),
    }
    .into()
}

/// Whether `table` is a Delta table.
pub(crate) fn is_delta_table(table: &CatalogTable) -> bool {
    table.provider.eq_ignore_ascii_case(DELTA)
}

/// Spark's `typeName` of a column type, as delta-spark's schema differences print it.
fn type_name(t: &DataType) -> String {
    t.type_name()
}

/// `NOT NULL` on the declared columns of `t` (its column list, then its typed `PARTITIONED BY`
/// columns, as [`Session::declared_columns`] lists them): a Delta table keeps it, where a file
/// source's columns are all nullable.
pub(super) fn keep_not_null(
    declared: &mut [StoreColumn],
    t: &CreateTable,
    typed_partitions: &[&krishiv_sql::ast::PartitionColumn],
) {
    let not_null = t
        .elements
        .iter()
        .filter_map(|e| match e {
            krishiv_sql::ast::TableElement::Column(def) => Some(def.not_null),
            _ => None,
        })
        .chain(typed_partitions.iter().map(|c| c.not_null));
    for (c, not_null) in declared.iter_mut().zip(not_null) {
        c.nullable = !not_null;
    }
}

/// Whether `path` is absolute as Hadoop's `Path.isAbsolute` reads it: a leading `/`, or a URI
/// scheme followed by one (`file:/t`, `s3://b/t`).
/// On Windows a drive path (`C:\Users\t`, `C:/Users/t`) is absolute too.
fn absolute(path: &str) -> bool {
    path.starts_with('/')
        || (cfg!(windows)
            && matches!(path.as_bytes(), [d, b':', b'\\' | b'/', ..] if d.is_ascii_alphabetic()))
        || path.split_once(':').is_some_and(|(scheme, rest)| {
            !scheme.is_empty()
                && scheme
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
                && rest.starts_with('/')
        })
}

/// The field a new table's log holds for the declared column `c`: its type in the client form
/// (a decimal on `Decimal128`, as the log's `decimal(p,s)` reads back; the engine's narrower
/// decimal carriers have no Delta type), its comment and its `CHAR`/`VARCHAR` type as Spark's
/// field metadata (`comment`, `__CHAR_VARCHAR_TYPE_STRING`).
fn log_field(c: &StoreColumn) -> arrow_schema::Field {
    let field = c.data_type.to_arrow_field(&c.name, c.nullable);
    let data_type = krishiv_types::arrow::client_arrow_type(field.data_type());
    let mut metadata = field.metadata().clone();
    if let Some(comment) = &c.comment {
        metadata.insert("comment".to_owned(), comment.clone());
    }
    if let Some(declared) = &c.char_varchar {
        metadata.insert("__CHAR_VARCHAR_TYPE_STRING".to_owned(), declared.clone());
    }
    field.with_data_type(data_type).with_metadata(metadata)
}

/// `SchemaUtils.reportDifferences(existing, specified)` for top-level columns: missing and
/// additional fields by exact name, then for each common field its nullability and type
/// differences. delta-spark lists the missing and additional fields in its set's order; here they
/// are in the schema's order (a known difference).
fn schema_differences(existing: &[TableColumn], specified: &[StoreColumn]) -> Vec<String> {
    let find_specified = |n: &str| specified.iter().find(|c| c.name == n);
    let find_existing = |n: &str| existing.iter().find(|c| c.name == n);
    let missing: Vec<&str> = existing
        .iter()
        .filter(|c| find_specified(&c.name).is_none())
        .map(|c| c.name.as_str())
        .collect();
    let additional: Vec<&str> = specified
        .iter()
        .filter(|c| find_existing(&c.name).is_none())
        .map(|c| c.name.as_str())
        .collect();
    let mut out = Vec::new();
    if !missing.is_empty() {
        out.push(format!(
            "- Specified schema is missing field(s): {}",
            missing.join(", ")
        ));
    }
    if !additional.is_empty() {
        out.push(format!(
            "- Specified schema has additional field(s): {}",
            additional.join(", ")
        ));
    }
    let non = |nullable: bool| if nullable { "" } else { "non-" };
    for s in specified {
        let Some(e) = find_existing(&s.name) else {
            continue;
        };
        if e.nullable != s.nullable {
            out.push(format!(
                "- Field {} is {}nullable in specified schema but {}nullable in existing schema.",
                s.name,
                non(s.nullable),
                non(e.nullable)
            ));
        }
        if e.data_type != s.data_type {
            out.push(format!(
                "- Specified type for {} is different from existing schema:\n  Specified: {}\n  \
                 Existing:  {}",
                s.name,
                type_name(&s.data_type),
                type_name(&e.data_type)
            ));
        }
    }
    out
}

impl Session {
    /// Asks `SELECT` on the Delta table root `path` before its log is read: qualified against the
    /// working directory, then a local root through [`Self::authorize_path`] (the canonical path,
    /// links resolved: stricter than a plain read by path, whose text is asked as written) and an
    /// object-store root as its qualified path, as [`Self::authorize_location`] asks `CREATE`.
    /// Returns the qualified text the table is opened with.
    async fn authorize_delta_root(&self, path: &str) -> Result<String, KrishivError> {
        let qualified = self.qualify_path(path)?;
        self.authorize_qualified_root(&qualified, &[Privilege::Select], StatementClass::Ddl)
            .await?;
        Ok(qualified)
    }

    /// Asks `privileges` on the already qualified Delta table root `qualified`: a `file:` root
    /// through [`Self::authorize_path`] (links resolved), an object-store root as its qualified
    /// path.
    pub(super) async fn authorize_qualified_root(
        &self,
        qualified: &str,
        privileges: &[Privilege],
        class: StatementClass,
    ) -> Result<(), KrishivError> {
        match qualified.strip_prefix("file:") {
            Some(local) => {
                self.authorize_path(&krishiv_exec::files::native_path(local), privileges, class)
                    .await?;
            }
            None => {
                let securable =
                    Securable::Path(OptionValue::from_user("path", qualified.to_owned()));
                self.authorize_securable(&securable, privileges, class)
                    .await?;
            }
        }
        Ok(())
    }

    /// `CREATE TABLE t [(columns)] USING delta [PARTITIONED BY (..)] LOCATION 'p'`: registers the
    /// Delta table at `p`, whose columns and partitioning come from its log. A column list or a
    /// partitioning that differs from the log's is delta-spark's error.
    ///
    /// # Errors
    /// The write refusals (CTAS, no location, a location without a Delta table, bucketing,
    /// `TBLPROPERTIES` or options other than `path`);
    /// `DELTA_CREATE_TABLE_SCHEME_MISMATCH`; `DELTA_CREATE_TABLE_WITH_DIFFERENT_PARTITIONING`;
    /// the table's read errors.
    #[expect(
        clippy::too_many_arguments,
        reason = "the CREATE TABLE dispatch hands over its resolved parts"
    )]
    pub(super) async fn create_delta_table(
        &self,
        mut table: CatalogTable,
        t: &CreateTable,
        specified: &[StoreColumn],
        name: &TableName,
        partition_names: &[String],
        mode: CreateMode,
        located: bool,
    ) -> Result<(), KrishivError> {
        let quoted = name.quoted();
        if t.bucket.is_some() {
            return Err(delta_write_refused(quoted, "CREATE TABLE ... CLUSTERED BY"));
        }
        if !located {
            return self
                .create_new_delta_table(table, specified, name, partition_names, mode)
                .await;
        }
        // The location is read before it is registered.
        match table.location.to_str() {
            Some(text) => {
                self.authorize_delta_root(text).await?;
            }
            // Refused there, audited, before the policy is asked.
            None => {
                self.authorize_path(&table.location, &[Privilege::Select], StatementClass::Ddl)
                    .await?;
            }
        }
        let env = self.query_env();
        let source = crate::catalog::table_source(&table);
        let shape = env
            .catalog
            .delta
            .shape(&source, format!("{}.{}", name.namespace, name.name))
            .await?;
        if !shape.existing {
            return self
                .create_new_delta_table(table, specified, name, partition_names, mode)
                .await;
        }
        if !specified.is_empty() {
            let differences = schema_differences(&shape.columns, specified);
            if !differences.is_empty() {
                let existing: Vec<(&str, &DataType, bool)> = shape
                    .columns
                    .iter()
                    .map(|c| (c.name.as_str(), &c.data_type, c.nullable))
                    .collect();
                let given: Vec<(&str, &DataType, bool)> = specified
                    .iter()
                    .map(|c| (c.name.as_str(), &c.data_type, c.nullable))
                    .collect();
                return Err(krishiv_common::error::DeltaCreateTableSchemeMismatch {
                    path: shape.path.clone(),
                    specified_schema: crate::introspection::tree_string(&given),
                    existing_schema: crate::introspection::tree_string(&existing),
                    schema_differences: differences.join("\n"),
                }
                .into());
            }
        }
        if (!specified.is_empty() || !partition_names.is_empty())
            && (partition_names.len() != shape.partition_columns.len()
                || partition_names
                    .iter()
                    .zip(&shape.partition_columns)
                    .any(|(a, b)| !a.eq_ignore_ascii_case(b)))
        {
            return Err(
                krishiv_common::error::DeltaCreateTableWithDifferentPartitioning {
                    path: shape.path.clone(),
                    specified_columns: partition_names.join(", "),
                    existing_columns: shape.partition_columns.join(", "),
                }
                .into(),
            );
        }
        // delta-spark compares given properties with the log's configuration; until a probe
        // pins its rule (and class), they are refused rather than stored unchecked (ADR 0008).
        if !table.properties.is_empty() {
            return Err(delta_write_refused(
                quoted,
                "CREATE TABLE ... TBLPROPERTIES over an existing Delta table",
            ));
        }
        // delta-spark compares a given comment with the log's `description`; refused until a
        // probe pins that comparison, as the properties are.
        if table.comment.is_some() {
            return Err(delta_write_refused(
                quoted,
                "CREATE TABLE ... COMMENT over an existing Delta table",
            ));
        }
        if table
            .options
            .keys()
            .any(|k| !k.eq_ignore_ascii_case("path"))
        {
            return Err(delta_write_refused(
                quoted,
                "CREATE TABLE ... OPTIONS over an existing Delta table",
            ));
        }
        table.columns = shape
            .columns
            .iter()
            .map(|c| StoreColumn::plain(c.name.clone(), c.data_type.clone(), c.nullable))
            .collect();
        table.layout = TableLayout {
            partition_columns: shape.partition_columns.clone(),
            ..TableLayout::default()
        };
        self.add_table(table, mode, false).await
    }

    /// `CREATE TABLE t [(columns)] USING delta [PARTITIONED BY (..)] [LOCATION 'p']` where no
    /// Delta table is: version 0 of a new table with the declared columns, partitioning and
    /// properties (the kernel's table-features protocol, design section 2.1), then the catalog
    /// entry.
    ///
    /// # Errors
    /// A table `COMMENT`, refused by name (the kernel writes no `description`); the kernel's
    /// create errors (an unsupported property); storage errors.
    async fn create_new_delta_table(
        &self,
        mut table: CatalogTable,
        specified: &[StoreColumn],
        name: &TableName,
        partition_names: &[String],
        mode: CreateMode,
    ) -> Result<(), KrishivError> {
        // A table whose name already exists is the store's error, before anything is written.
        if mode == CreateMode::IfNotExists
            && self
                .state
                .catalog
                .store
                .table(&name.namespace, &name.name)
                .is_some()
        {
            return Ok(());
        }
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
        if specified.is_empty() {
            // delta-spark creates a table of no columns; the kernel needs one.
            return Err(delta_write_refused(
                name.quoted(),
                "CREATE TABLE without columns where there is no Delta table",
            ));
        }
        // delta-spark writes the comment as the log's `description`; the kernel's create writes
        // none (delta_kernel 0.30.1), so it is refused rather than kept in the catalog alone.
        if table.comment.is_some() {
            return Err(delta_write_refused(
                name.quoted(),
                "CREATE TABLE ... COMMENT on a new Delta table",
            ));
        }
        self.authorize_path(&table.location, &[Privilege::Create], StatementClass::Ddl)
            .await?;
        let fields: Vec<_> = specified.iter().map(log_field).collect();
        let schema = std::sync::Arc::new(arrow_schema::Schema::new(fields));
        let source =
            super::delta_source(&self.qualify_path(&table.location.display().to_string())?);
        let env = self.query_env();
        let new = env.catalog.delta.new_table(
            &source,
            &schema,
            partition_names.to_vec(),
            table.properties.clone(),
            Some(format!("{}.{}", name.namespace, name.name)),
        )?;
        super::delta_writes::create_local_root(&source).await?;
        new.create_empty(&crate::session::job_uuid()).await?;
        table.columns = specified.to_vec();
        table.layout = TableLayout {
            partition_columns: partition_names.to_vec(),
            ..TableLayout::default()
        };
        self.add_table(table, mode, true).await
    }

    /// The Delta path a `CREATE TABLE delta.`<path>` .. USING delta` names, as written: `delta`
    /// and an absolute path (delta-spark's `isPathIdentifier`: Hadoop's `Path.isAbsolute`, so
    /// `delta.`rel/dir`` is a table `dir`.. in namespace `delta`), no catalog table of that name, `USING delta`. Any other target is
    /// `None` (a catalog table, or a name the name checks refuse).
    pub(super) fn delta_path_target(
        &self,
        t: &CreateTable,
        env: &QueryEnv,
    ) -> Result<Option<String>, KrishivError> {
        let using_delta = t.using.as_ref().is_some_and(
            |u| matches!(u.parts.as_slice(), [only] if only.value.eq_ignore_ascii_case(DELTA)),
        );
        if !using_delta {
            return Ok(None);
        }
        let name = self.table_name(&t.name, env)?;
        Ok(match crate::access::ddl_parts(&t.name, env)?.as_slice() {
            [format, path]
                if format.eq_ignore_ascii_case(DELTA)
                    && absolute(path)
                    && self
                        .state
                        .catalog
                        .store
                        .table(&name.namespace, &name.name)
                        .is_none() =>
            {
                Some(path.clone())
            }
            _ => None,
        })
    }

    /// `CREATE TABLE [IF NOT EXISTS] delta.`<path>` (columns) USING delta [PARTITIONED BY ..]
    /// [COMMENT ..] [TBLPROPERTIES ..]`: version 0 of a new Delta table at `path`, no catalog
    /// entry, as delta-spark 4.4.1 does (`tests/fixtures/delta-4.4.1/CREATE-PATH.json`). A Delta
    /// table or any file already at the path is `TABLE_OR_VIEW_ALREADY_EXISTS` naming
    /// `` `delta`.`<path>` `` (nothing with `IF NOT EXISTS`), checked before the column list.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_ALREADY_EXISTS`; refused by name (`UNSUPPORTED_FEATURE.TABLE_OPERATION`),
    /// where delta-spark creates a table: `OR REPLACE`, CTAS, no columns, a table `COMMENT` (the
    /// kernel writes no `description`), `OPTIONS`, a
    /// `LOCATION` other than the path, `CLUSTERED BY`; the access check (`CREATE` on the path);
    /// the kernel's create errors; storage errors.
    pub(super) async fn create_delta_path_table(
        &self,
        t: &CreateTable,
        path: &str,
        original: &str,
    ) -> Result<(), KrishivError> {
        let shown = format!("`delta`.`{}`", path.replace('`', "``"));
        if t.temporary {
            return Err(unsupported("CREATE TEMPORARY TABLE"));
        }
        if t.file_format.is_some() || t.row_format.is_some() {
            return Err(NotSupportedCommandWithoutHiveSupport {
                cmd: "CREATE Hive TABLE (AS SELECT)".to_owned(),
            }
            .into());
        }
        let refused = |operation: &str| Err(delta_write_refused(shown.clone(), operation));
        if t.or_replace {
            return refused("CREATE OR REPLACE TABLE by path");
        }
        if t.as_query.is_some() {
            return refused("CREATE TABLE ... AS SELECT by path");
        }
        if t.bucket.is_some() || t.skew.is_some() || !t.cluster_by.is_empty() {
            return refused("CREATE TABLE ... CLUSTERED BY");
        }
        if !t.options.is_empty() {
            return refused("CREATE TABLE ... OPTIONS by path");
        }
        let qualified = self.qualify_path(path)?;
        if let Some(location) = &t.location
            && self.qualify_path(location)? != qualified
        {
            return refused("CREATE TABLE by path with another LOCATION");
        }
        // A local root is acted on as the access check resolved it (links followed, canonical):
        // the occupancy check, the directory made and the table written all use that path.
        let checked = match qualified.strip_prefix("file:") {
            Some(local) => Some(
                self.authorize_path(Path::new(local), &[Privilege::Create], StatementClass::Ddl)
                    .await?,
            ),
            None => {
                self.authorize_qualified_root(
                    &qualified,
                    &[Privilege::Create],
                    StatementClass::Ddl,
                )
                .await?;
                None
            }
        };
        let root = match &checked {
            Some(dir) => krishiv_exec::files::file_display(&dir.path),
            None => qualified.clone(),
        };
        let source = super::delta_source(&root);
        let env = self.query_env();
        // delta-spark's `tableExists` for a path: a Delta table there, or anything at all in the
        // directory (a Parquet directory is "already exists" too; an empty directory is not).
        let occupied = env.catalog.delta.holds_files(&source).await?;
        if occupied {
            if t.if_not_exists {
                return Ok(());
            }
            return Err(krishiv_common::error::TableOrViewAlreadyExists {
                relation_name: shown,
            }
            .into());
        }
        let (partition_names, typed_partitions) = partition_fields(&t.partitioned_by)?;
        let mut declared = self.declared_columns(t, original, &typed_partitions)?;
        keep_not_null(&mut declared, t, &typed_partitions);
        if declared.is_empty() {
            return refused("CREATE TABLE without columns where there is no Delta table");
        }
        // delta-spark writes the comment as the log's `description`; the kernel's create
        // writes none, so it is refused rather than dropped.
        if t.comment.is_some() {
            return refused("CREATE TABLE ... COMMENT by path");
        }
        // The declared order, partition columns where they were written (delta-spark keeps it).
        let fields: Vec<_> = declared.iter().map(log_field).collect();
        let schema = std::sync::Arc::new(arrow_schema::Schema::new(fields));
        let new = env.catalog.delta.new_table(
            &source,
            &schema,
            partition_names,
            options_of(&t.tblproperties, original),
            None,
        )?;
        if let Some(dir) = &checked {
            self.create_checked_dir(
                &dir.path,
                &dir.shown,
                Privilege::Create,
                StatementClass::Ddl,
            )?;
        }
        new.create_empty(&crate::session::job_uuid()).await
    }

    /// `CREATE TEMPORARY VIEW v USING delta OPTIONS (path 'p')`: a read of the Delta table at
    /// `p`, its columns from the log now (each query reads the table's latest version).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for declared columns or no path; the table's read errors.
    pub(super) async fn create_delta_temp_view(
        &self,
        v: &CreatePermanentView,
        name: &str,
        env: &QueryEnv,
        original: &str,
    ) -> Result<(), KrishivError> {
        if !v.columns.is_empty() {
            return Err(unsupported("declared columns on a Delta temporary view"));
        }
        let mut options = options_of(refuse_secret_options(&v.options, original)?, original);
        // The view's options reach the store when it is opened here: a server-scoped key is
        // `Manage` on the cluster before that, as for `CREATE TABLE .. OPTIONS` (P3-T067b).
        self.authorize_store_options(
            krishiv_sql::StatementClass::Ddl,
            &self.state.conf.snapshot(),
            options.keys().map(String::as_str),
        )
        .await?;
        let path_key = options
            .keys()
            .find(|k| k.eq_ignore_ascii_case("path"))
            .cloned();
        let Some(path) = path_key.and_then(|k| options.remove(&k)) else {
            return Err(unsupported("a Delta temporary view without a path"));
        };
        let qualified = self.authorize_delta_root(&path).await?;
        let source = krishiv_plan::ReadSource::files(
            DELTA,
            vec![OptionValue::from_user("path", qualified)],
            options
                .into_iter()
                .map(|(k, val)| {
                    let val = OptionValue::from_user(&k, val);
                    (k, val)
                })
                .collect(),
        );
        let columns = env.catalog.delta.columns(&source, None).await?;
        self.state.catalog.put(
            name,
            crate::catalog::TempTable::Files {
                source,
                columns,
                known: true,
                permanent: false,
                partitions: Some(Default::default()),
            },
        );
        self.state.cache.invalidate_table(name);
        Ok(())
    }
}
