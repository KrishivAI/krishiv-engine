// SPDX-License-Identifier: Apache-2.0
//! Delta writes through the session (P3-T021 PR-37a part 2, design note section 4.1): every
//! Delta write is one query whose rows end in a
//! [`PhysNode::TableWrite`](krishiv_exec::physical::PhysNode::TableWrite) into the table's
//! writer, which commits once.
//!
//! - An existing table is opened through the writing query's own Delta reads, so a statement
//!   that also reads it (`INSERT INTO t SELECT .. FROM t`) reads and commits on one snapshot;
//!   such a statement is not a blind append (`READS_TARGET_OPTION`).
//! - A new table (`CREATE TABLE .. USING delta [AS SELECT]`, a first `save`, `saveAsTable`) is a
//!   hidden provider ([`NewDeltaTable`]) whose writer creates version 0 with the rows.
//! - `df.write.format("delta").save(path)`: Delta's save modes; `replaceWhere`,
//!   `partitionOverwriteMode` and `overwriteSchema` as Delta reads them.
//!
//! What stays refused, with the class the design names: replacing a table (`REPLACE TABLE`,
//! RTAS, `createOrReplace`, `overwriteSchema`) as `UNSUPPORTED_FEATURE.TABLE_OPERATION`; MERGE,
//! UPDATE and DELETE keep their refusals (PR-37b on).

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::StreamExt;
use krishiv_common::error::{
    DeltaMetadataMismatch, KrishivError, KrishivUnsupportedFeature,
    UnsupportedFeatureTableOperation,
};
use krishiv_common::secret::OptionValue;
use krishiv_connectors_delta::write::{PREDICATE_TEXT_OPTION, READS_TARGET_OPTION};
use krishiv_exec::physical::TableWriteRequest;
use krishiv_exec::planner::{TableProvider, WriteMode, WriteRequest};
use krishiv_plan::analyzer::TableColumn;
use krishiv_plan::{BoundColumn, ReadSource, Scan};

use crate::dataframe::DataFrame;
use crate::delta::DELTA;
use crate::functions::col;
use crate::session::{Session, internal};

/// Where a Delta write goes.
pub(crate) enum DeltaTarget {
    /// The existing table `source` reads, named `name` in errors (a catalog table's
    /// `namespace.table`; a path's Hadoop form when `None`).
    Existing {
        source: ReadSource,
        name: Option<String>,
    },
    /// A table created at the location `source` names.
    Create {
        source: ReadSource,
        name: Option<String>,
        partition_by: Vec<String>,
        properties: BTreeMap<String, String>,
    },
}

/// The read of the Delta table at `path` (already qualified).
pub(crate) fn delta_source(path: &str) -> ReadSource {
    ReadSource::files(
        DELTA,
        vec![OptionValue::from_user("path", path)],
        BTreeMap::new(),
    )
}

/// `name` as SQL text: a backquoted identifier.
fn ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// The bound columns of a scan reading `columns`.
fn bound(columns: &[TableColumn]) -> Vec<BoundColumn> {
    columns
        .iter()
        .zip(0_u64..)
        .map(|(c, i)| BoundColumn {
            id: krishiv_plan::AttrId(i),
            name: c.name.clone(),
            data_type: c.data_type.clone(),
            nullable: c.nullable,
        })
        .collect()
}

/// The refusal of a replace (design section 4.1, "Replace is refused").
pub(crate) fn replace_refused(table: String, what: &str) -> KrishivError {
    UnsupportedFeatureTableOperation {
        table_name: table,
        operation: format!(
            "{what} on a Delta table (replacing a Delta table is not supported: the kernel has \
             no replace transaction)"
        ),
    }
    .into()
}

impl Session {
    /// Writes the rows of `df` (the table's columns, in its order) into the Delta table
    /// `target` with `mode`; `quoted` names it in the write request.
    ///
    /// # Errors
    /// The table's open and write errors (the writer's refusals, conflicts, the replaceWhere
    /// check); analysis and execution errors.
    pub(crate) async fn write_delta(
        &self,
        df: &DataFrame,
        target: DeltaTarget,
        mode: WriteMode,
        options: BTreeMap<String, String>,
        quoted: &str,
    ) -> Result<(), KrishivError> {
        let (scan, df, opened) = match target {
            DeltaTarget::Existing { source, name } => {
                // The target is opened in the analysis that is planned: a frame that reads files
                // is analyzed afresh by each action, in a new query catalog.
                let a = df.action_analysis().await?;
                let table = a.env.catalog.delta.open(&source, name).await?;
                let columns = delta_columns(&table)?;
                let scan = Scan {
                    source,
                    output: bound(&columns),
                    version: None,
                };
                (scan, df.clone(), Some(a))
            }
            DeltaTarget::Create {
                source,
                name,
                partition_by,
                properties,
            } => {
                create_local_root(&source).await?;
                let a = df.action_analysis().await?;
                let columns: Vec<TableColumn> = a
                    .analyzed
                    .output
                    .iter()
                    .map(|c| {
                        // A query's columns are created nullable, as delta-spark creates them
                        // (`WRITES.json`: an append of NULL after the CTAS).
                        TableColumn::new(
                            c.name.clone(),
                            krishiv_plan::without_generated_marker(&c.data_type),
                            true,
                        )
                    })
                    .collect();
                let schema = arrow_schema(&columns)?;
                let env = self.query_env();
                let new = env.catalog.delta.new_table(
                    &source,
                    &schema,
                    partition_by,
                    properties,
                    name,
                )?;
                let (hidden, entry) = self
                    .state
                    .catalog
                    .put_hidden_provider(columns.clone(), Arc::new(new) as Arc<dyn TableProvider>);
                let scan = Scan {
                    source: ReadSource::Table {
                        name: vec![hidden],
                        options: BTreeMap::new(),
                    },
                    output: bound(&columns),
                    version: None,
                };
                // The rows, and no rows of the hidden table: its scan in the plan makes the
                // query load it, so the write's target resolves on the worker.
                let mut hidden = DataFrame::new(
                    self.clone(),
                    Arc::new(krishiv_plan::LogicalPlan::new(
                        krishiv_plan::PlanNode::Scan(scan.clone()),
                    )),
                );
                hidden.keep.push(entry);
                let fresh =
                    DataFrame::new(self.clone(), Arc::clone(&df.plan)).union(&hidden.limit(0));
                (scan, fresh, None)
            }
        };
        let analysis = match opened {
            Some(a) => a,
            None => df.action_analysis().await?,
        };
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        let mut request = WriteRequest::new(mode, quoted, Arc::clone(physical.schema()));
        request.options = options;
        super::row_level::apply_idempotency_key(&analysis.env.config, &mut request.options)?;
        request.write_id = Some(crate::session::job_uuid());
        let plan = physical.into_table_write(scan, TableWriteRequest::Write(request));
        let mut stream = Box::pin(self.run(&plan, &analysis.env)).await?;
        while let Some(batch) = stream.next().await {
            batch?;
        }
        drop(stream);
        self.refresh_files();
        Ok(())
    }

    /// `df` with the columns of `table` by name (ASCII case-insensitively), each stored in the
    /// table's type (Spark's ANSI store assignment), in the table's order.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` when the names differ from the table's (schema evolution
    /// on write is not supported yet).
    pub(crate) async fn delta_by_name(
        &self,
        df: &DataFrame,
        table: &[TableColumn],
        quoted: &str,
    ) -> Result<DataFrame, KrishivError> {
        let schema = df.schema().await?;
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        let mut select = Vec::with_capacity(table.len());
        for c in table {
            let mut hits = names.iter().filter(|n| n.eq_ignore_ascii_case(&c.name));
            let (Some(hit), None) = (hits.next(), hits.next()) else {
                return Err(columns_differ(quoted));
            };
            // Nested fields cast as nullable: the writer checks the table's NOT NULL fields
            // and raises delta-spark's `DELTA_NOT_NULL_CONSTRAINT_VIOLATED` (a cast into a
            // non-null struct field is refused before any row is read).
            select.push(
                col(&ident(hit))
                    .cast_ansi(c.data_type.as_nullable())
                    .alias(&c.name),
            );
        }
        if names.len() != table.len() {
            return Err(columns_differ(quoted));
        }
        Ok(df.select(select))
    }

    /// `df.write.format("delta").mode(mode).save(path)`: Delta's save modes over the table at
    /// `path` (created when there is none).
    ///
    /// # Errors
    /// `INVALID_SAVE_MODE`; the replace refusal for `overwriteSchema`; the oracle's errors for
    /// `errorifexists` on an existing table; the write's errors.
    pub(crate) async fn save_delta(
        &self,
        df: &DataFrame,
        path: &str,
        mode: &str,
        options: &BTreeMap<String, OptionValue>,
        partition_by: &[String],
    ) -> Result<(), KrishivError> {
        use krishiv_io::SaveMode;
        let mode = SaveMode::parse(mode)?;
        let qualified = self.qualify_path(path)?;
        let source = delta_source(&qualified);
        let env = self.query_env();
        let exists = env.catalog.delta.exists(&source).await?;
        let plain = plain_options(options)?;
        let quoted = format!("delta.{}", ident(&qualified));
        let option = |k: &str| {
            plain
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(k))
                .map(|(_, v)| v.clone())
        };
        // On a new path delta-spark simply creates the table; only replacing an existing
        // table's schema is refused.
        if exists
            && option("overwriteSchema").is_some_and(|v| v.eq_ignore_ascii_case("true"))
            && mode == SaveMode::Overwrite
        {
            return Err(replace_refused(quoted, "overwriteSchema"));
        }
        let privileges: &[krishiv_plan::analyzer::Privilege] = if exists {
            &[krishiv_plan::analyzer::Privilege::Modify]
        } else {
            &[krishiv_plan::analyzer::Privilege::Create]
        };
        self.authorize_qualified_root(&qualified, privileges, krishiv_sql::StatementClass::Write)
            .await?;
        // Every save mode creates a table where there is none.
        if !exists {
            return self
                .write_delta(
                    df,
                    DeltaTarget::Create {
                        source,
                        name: None,
                        partition_by: partition_by.to_vec(),
                        properties: BTreeMap::new(),
                    },
                    WriteMode::Append,
                    writer_options(plain.iter().map(|(k, v)| (k, v))),
                    &quoted,
                )
                .await;
        }
        let write_mode = match mode {
            SaveMode::Ignore => return Ok(()),
            SaveMode::ErrorIfExists => {
                // `tests/fixtures/delta-4.4.1/WRITES.json` `save_errorifexists`.
                return Err(krishiv_common::error::DeltaPathExists {
                    path: qualified.trim_end_matches('/').to_owned(),
                }
                .into());
            }
            SaveMode::Append => WriteMode::Append,
            SaveMode::Overwrite => match option("replaceWhere") {
                Some(text) => {
                    WriteMode::OverwriteByFilter(Arc::new(crate::functions::expr(&text)?.expr))
                }
                None => {
                    if crate::session::partition_overwrite_is_dynamic(&env.config, options)? {
                        WriteMode::OverwriteDynamic
                    } else {
                        WriteMode::Truncate
                    }
                }
            },
        };
        let mut request_options = writer_options(plain.iter().map(|(k, v)| (k, v)));
        if let Some(text) = option("replaceWhere") {
            request_options.insert(PREDICATE_TEXT_OPTION.to_owned(), text);
        }
        let columns = env.catalog.delta.columns(&source, None).await?;
        if !partition_by.is_empty() {
            let table = env.catalog.delta.open(&source, None).await?;
            let same = table.partition_columns().len() == partition_by.len()
                && table
                    .partition_columns()
                    .iter()
                    .zip(partition_by)
                    .all(|(a, b)| a.eq_ignore_ascii_case(b));
            if !same {
                return Err(KrishivUnsupportedFeature {
                    feature: format!(
                        "a Delta write to {quoted} partitioned other than the table (changing \
                         the partitioning replaces the table)"
                    ),
                }
                .into());
            }
        }
        drop(env);
        let aligned = self.delta_by_name(df, &columns, &quoted).await?;
        if write_reads_target(&aligned, &qualified).await {
            request_options.insert(READS_TARGET_OPTION.to_owned(), "true".to_owned());
        }
        self.write_delta(
            &aligned,
            DeltaTarget::Existing { source, name: None },
            write_mode,
            request_options,
            &quoted,
        )
        .await
    }

    /// v1 `saveAsTable(mode = overwrite)` on the existing Delta table `name` (design section
    /// 4.1): the same columns commit as a truncate of the data with no metadata change (the
    /// table keeps its partitioning, as delta-spark keeps it); other columns are delta-spark's
    /// `DELTA_METADATA_MISMATCH`.
    pub(crate) async fn delta_save_as_table_overwrite(
        &self,
        df: &DataFrame,
        table: &crate::store::CatalogTable,
        options: &BTreeMap<String, OptionValue>,
    ) -> Result<(), KrishivError> {
        let quoted = format!("`{}`.`{}`", table.namespace, table.name);
        let plain = plain_options(options)?;
        if plain.iter().any(|(k, v)| {
            k.eq_ignore_ascii_case("overwriteSchema") && v.eq_ignore_ascii_case("true")
        }) {
            return Err(replace_refused(
                quoted,
                "saveAsTable overwrite with overwriteSchema",
            ));
        }
        let source = crate::catalog::table_source(table);
        let name = Some(format!("{}.{}", table.namespace, table.name));
        let env = self.query_env();
        let columns = env.catalog.delta.columns(&source, name.clone()).await?;
        drop(env);
        let schema = df.schema().await?;
        let same = schema.fields().len() == columns.len()
            && schema.fields().iter().zip(&columns).all(|(f, c)| {
                f.name().eq_ignore_ascii_case(&c.name)
                    && krishiv_types::DataType::from_arrow_field(f).is_ok_and(|t| t == c.data_type)
            });
        if !same {
            // delta-spark raises the parent class bare, with no parameters
            // (`tests/fixtures/delta-4.4.1/WRITES.json` `save_as_table_overwrite_other_schema`).
            return Err(DeltaMetadataMismatch {}.into());
        }
        let aligned = self.delta_by_name(df, &columns, &quoted).await?;
        self.write_delta(
            &aligned,
            DeltaTarget::Existing { source, name },
            WriteMode::Truncate,
            writer_options(plain.iter().map(|(k, v)| (k, v))),
            &quoted,
        )
        .await?;
        self.state.catalog.store.bump();
        Ok(())
    }
}

impl Session {
    /// `INSERT INTO|OVERWRITE` into the Delta catalog table `table` of the rows of `df` (its
    /// columns, in its order): an append; an overwrite of the static partition `statics`
    /// (`replaceWhere` of `(p <=> 'v')`, as delta-spark records it); a dynamic partition
    /// overwrite under `spark.sql.sources.partitionOverwriteMode = dynamic`; else a truncate.
    /// The target's writer options give the files' `maxRecordsPerFile`.
    pub(super) async fn insert_delta(
        &self,
        name: &super::TableName,
        table: &crate::store::CatalogTable,
        df: &DataFrame,
        target: &super::InsertTarget<'_>,
        reads_target: bool,
    ) -> Result<(), KrishivError> {
        let (overwrite, statics) = (target.overwrite, target.statics);
        // `maxRecordsPerFile` of a DataFrame write into the table (`writeTo(..).append()`,
        // `saveAsTable` append, `insertInto`).
        let mut options = match target.options {
            // Only the option the writer reads is asked to be plain: an append ignores the
            // others, a secret one included.
            Some(o) => {
                let read: BTreeMap<String, OptionValue> = o
                    .iter()
                    .filter(|(k, _)| {
                        k.eq_ignore_ascii_case("maxRecordsPerFile")
                            || k.eq_ignore_ascii_case(krishiv_exec::planner::IDEMPOTENCY_KEY_OPTION)
                    })
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                writer_options(plain_options(&read)?.iter().map(|(k, v)| (k, v)))
            }
            None => BTreeMap::new(),
        };
        let mode = if !overwrite {
            WriteMode::Append
        } else if !statics.is_empty() {
            let mut filter: Option<krishiv_plan::Expr> = None;
            let mut texts = Vec::new();
            for (column, value) in statics {
                let (literal, text) = if value == krishiv_io::partition::DEFAULT_PARTITION {
                    (krishiv_plan::Literal::Null(None), "NULL".to_owned())
                } else {
                    (
                        krishiv_plan::Literal::String(value.clone()),
                        format!("'{}'", value.replace('\'', "\\'")),
                    )
                };
                texts.push(format!("({column} <=> {text})"));
                let eq = krishiv_plan::Expr::function(
                    "<=>",
                    vec![
                        krishiv_plan::Expr::column([column.as_str()]),
                        krishiv_plan::Expr::literal(literal),
                    ],
                );
                filter = Some(match filter {
                    None => eq,
                    Some(f) => krishiv_plan::Expr::function("and", vec![f, eq]),
                });
            }
            options.insert(PREDICATE_TEXT_OPTION.to_owned(), texts.join(" AND "));
            WriteMode::OverwriteByFilter(Arc::new(
                filter.ok_or_else(|| internal("a static overwrite without values".to_owned()))?,
            ))
        } else if table.is_partitioned()
            && crate::session::partition_overwrite_is_dynamic(
                &self.query_env().config,
                &BTreeMap::new(),
            )?
        {
            WriteMode::OverwriteDynamic
        } else {
            WriteMode::Truncate
        };
        if reads_target {
            options.insert(READS_TARGET_OPTION.to_owned(), "true".to_owned());
        }
        self.write_delta(
            df,
            DeltaTarget::Existing {
                source: crate::catalog::table_source(table),
                name: Some(format!("{}.{}", table.namespace, table.name)),
            },
            mode,
            options,
            &name.quoted(),
        )
        .await
    }
}

/// Whether `df`'s plan reads the Delta table at `path`.
async fn write_reads_target(df: &DataFrame, path: &str) -> bool {
    let Ok(a) = df.analysis().await else {
        return false;
    };
    crate::catalog::reads_files_at(a.analyzed.plan.root(), path)
}

fn columns_differ(quoted: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: format!(
            "a Delta write to {quoted} whose columns differ from the table's (schema evolution \
             on write)"
        ),
    }
    .into()
}

/// The write options the Delta writer reads from its request (`maxRecordsPerFile`, matched
/// ignoring case as Spark's case-insensitive options are), forwarded as written.
pub(crate) fn writer_options<'a>(
    options: impl IntoIterator<Item = (&'a String, &'a String)>,
) -> BTreeMap<String, String> {
    options
        .into_iter()
        .filter(|(k, _)| k.eq_ignore_ascii_case("maxRecordsPerFile"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

fn plain_options(
    options: &BTreeMap<String, OptionValue>,
) -> Result<Vec<(String, String)>, KrishivError> {
    options
        .iter()
        .map(|(k, v)| {
            v.as_plain()
                .map(|t| (k.clone(), t.to_owned()))
                .ok_or_else(|| {
                    KrishivError::from(KrishivUnsupportedFeature {
                        feature: "secret options on a Delta write".to_owned(),
                    })
                })
        })
        .collect()
}

/// Makes the local directory a new table at `source` is created in (the kernel's create wants
/// the table root to exist), on the blocking pool; nothing for an object store.
pub(crate) async fn create_local_root(source: &ReadSource) -> Result<(), KrishivError> {
    let ReadSource::Files { paths, .. } = source else {
        return Ok(());
    };
    let locals: Vec<String> = paths
        .iter()
        .filter_map(|p| {
            p.as_plain()
                .and_then(|t| t.strip_prefix("file:"))
                .map(str::to_owned)
        })
        .collect();
    if locals.is_empty() {
        return Ok(());
    }
    tokio::task::spawn_blocking(move || {
        for local in &locals {
            std::fs::create_dir_all(local)
                .map_err(|e| internal(format!("cannot create {local}: {e}")))?;
        }
        Ok(())
    })
    .await
    .map_err(|e| internal(format!("creating a Delta table's directory: {e}")))?
}

/// The table's columns from a provider's schema.
fn delta_columns(
    table: &krishiv_connectors_delta::table::DeltaTable,
) -> Result<Vec<TableColumn>, KrishivError> {
    table
        .schema()
        .fields()
        .iter()
        .map(|f| {
            Ok(TableColumn::new(
                f.name().clone(),
                krishiv_types::DataType::from_arrow_field(f)?,
                f.is_nullable(),
            ))
        })
        .collect()
}

/// The Arrow schema of `columns` (the engine's forms).
fn arrow_schema(columns: &[TableColumn]) -> Result<arrow_schema::SchemaRef, KrishivError> {
    let fields: Vec<_> = columns
        .iter()
        .map(|c| c.data_type.to_arrow_field(&c.name, c.nullable))
        .collect();
    Ok(Arc::new(arrow_schema::Schema::new(fields)))
}
