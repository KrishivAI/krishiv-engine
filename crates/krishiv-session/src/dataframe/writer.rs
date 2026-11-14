// SPDX-License-Identifier: Apache-2.0
//! DataFrameWriter (files) and DataFrameWriterV2 (tables): the write builders of a DataFrame.

use super::*;

/// Writes a DataFrame's rows to files ([`DataFrame::write`]; PySpark's
/// `DataFrameWriter`).
#[derive(Debug, Clone)]
#[must_use]
pub struct DataFrameWriter {
    pub(super) df: DataFrame,
    pub(super) format: String,
    pub(super) mode: String,
    pub(super) options: BTreeMap<String, OptionValue>,
    pub(super) partition_by: Vec<String>,
    /// `bucketBy`: the bucket count and columns.
    pub(super) bucket_by: Option<(i32, Vec<String>)>,
    /// `sortBy`.
    pub(super) sort_by: Option<Vec<String>>,
    /// Whether [`Self::format`] was called: a table writer without one uses
    /// `spark.sql.sources.default`.
    pub(super) format_set: bool,
}

impl DataFrameWriter {
    /// What to do when the path exists (Spark's save modes, ASCII
    /// case-insensitive): `errorifexists` (also `error`, `default`; the
    /// default) fails with `PATH_ALREADY_EXISTS`, `ignore` writes nothing,
    /// `append` adds files, `overwrite` replaces what is there once the
    /// new files are complete. Another name fails the write with
    /// `INVALID_SAVE_MODE`.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().mode("append");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn mode(mut self, mode: &str) -> Self {
        mode.clone_into(&mut self.mode);
        self
    }

    /// The file format (default `spark.sql.sources.default`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().format("parquet");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn format(mut self, format: &str) -> Self {
        format.clone_into(&mut self.format);
        self.format_set = true;
        self
    }

    /// A write option (`compression`, ...); values are typed as secrets
    /// where Spark redacts them.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().option("compression", "zstd");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options
            .insert(key.to_owned(), OptionValue::from_user(key, value));
        self
    }

    /// Several write options at once (PySpark's `options(**options)`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().options([("compression", "zstd")]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn options(
        mut self,
        options: impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>,
    ) -> Self {
        for (key, value) in options {
            self.options.insert(
                key.as_ref().to_owned(),
                OptionValue::from_user(key.as_ref(), value.as_ref()),
            );
        }
        self
    }

    /// Partition the output by `cols` (PySpark's `partitionBy`); each call
    /// replaces the columns of the one before, so an empty call clears them.
    /// Rows land in `<col>=<value>/` directories (Spark's Hive-style
    /// layout), ordered by the partition columns, holding only the other
    /// columns.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().partition_by(["a"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn partition_by(mut self, cols: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        self.partition_by = cols.into_iter().map(|c| c.as_ref().to_owned()).collect();
        self
    }

    /// Writes the rows to the directory `path` in the format. Dropping
    /// the future before the write commits cancels it and deletes its
    /// files.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let out = dir.path().join("t");
    /// futures::executor::block_on(s.table("t").write().format("parquet").save(out.to_str().unwrap()))?;
    /// assert!(out.join("_SUCCESS").exists());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis and planning errors; `INVALID_SAVE_MODE` (checked before
    /// partitioning, so a bad mode wins over a partitioned write);
    /// `PATH_ALREADY_EXISTS` (`errorifexists` on an existing path, or an
    /// append to a file); the format's option errors; write failures;
    /// `PARTITION_COLUMN_NOT_FOUND_IN_SCHEMA` for a `partition_by` column
    /// the output has not; `ALL_PARTITION_COLUMNS_NOT_ALLOWED` when every
    /// output column is a partition column.
    pub async fn save(self, path: &str) -> Result<(), KrishivError> {
        self.save_reporting(path, None).await.map(drop)
    }

    /// [`Self::save`], returning the partition directories (`col=value/...`
    /// below `path`, escaped) that hold the committed files.
    pub(crate) async fn save_reporting(
        self,
        path: &str,
        admitted: Option<crate::query_fold::Admitted>,
    ) -> Result<Vec<String>, KrishivError> {
        self.save_or_check(path, admitted, false).await
    }

    /// Every check [`Self::save_reporting`] makes before it changes anything (the rows resolve,
    /// the format and options, the path and its access check), without writing: a multi-table
    /// `INSERT` checks all its clauses before the first one writes.
    pub(crate) async fn check_save(self, path: &str) -> Result<(), KrishivError> {
        self.save_or_check(path, None, true).await.map(|_| ())
    }

    async fn save_or_check(
        self,
        path: &str,
        admitted: Option<crate::query_fold::Admitted>,
        dry: bool,
    ) -> Result<Vec<String>, KrishivError> {
        self.assert_not_bucketed("save")?;
        let target = WriteTarget {
            format: self.format,
            mode: self.mode,
            path: OptionValue::from_user("path", path),
            options: self.options,
            partition_by: self.partition_by,
        };
        let df = self.df;
        let session = df.session.clone();
        session
            .on_executor(async move {
                let mut admitted = admitted;
                let written = async {
                    let a = df.action_analysis().await?;
                    geo_gate::check(&a.analyzed.output, &target.format)?;
                    // A write reads the cache like any other action.
                    let (physical, fill) = df.physical_cached(&a).await?;
                    let _lease = fill.into_lease();
                    let types: Vec<_> = a
                        .analyzed
                        .output
                        .iter()
                        .map(|c| c.data_type.clone())
                        .collect();
                    df.session
                        .write_files(physical, &a.env, target, &types, &mut admitted, dry)
                        .await
                }
                .await;
                // An admission the write did not take (it ended before its query ran) is settled
                // with that outcome; dropped with the future, it closes its row `cancelled`.
                if let Some(unrun) = admitted.take() {
                    match &written {
                        Ok(_) => unrun.finish_ok(),
                        Err(error) => unrun.fail(error),
                    }
                }
                written
            })
            .await
    }

    /// Buckets the output by `cols` into `buckets` buckets (PySpark's
    /// `bucketBy`). Only `saveAsTable` takes a bucketed layout in Spark, and
    /// the engine has no bucketed tables yet: `save` and `insertInto` fail as
    /// in Spark and `save_as_table` with `KRISHIV_UNSUPPORTED_FEATURE`.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().bucket_by(4, ["a"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn bucket_by(
        mut self,
        buckets: i32,
        cols: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Self {
        self.bucket_by = Some((
            buckets,
            cols.into_iter().map(|c| c.as_ref().to_owned()).collect(),
        ));
        self
    }

    /// Sorts each bucket by `cols` (PySpark's `sortBy`); needs `bucket_by`.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().bucket_by(4, ["a"]).sort_by(["b"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn sort_by(mut self, cols: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        self.sort_by = Some(cols.into_iter().map(|c| c.as_ref().to_owned()).collect());
        self
    }

    /// Spark's `isBucketed` and `assertNotBucketed(operation)`: `sortBy`
    /// without `bucketBy` is `SORT_BY_WITHOUT_BUCKETING`; a bucketed write
    /// through `operation` is refused.
    pub(super) fn assert_not_bucketed(&self, operation: &str) -> Result<(), KrishivError> {
        match (&self.bucket_by, &self.sort_by) {
            (None, None) => Ok(()),
            (None, Some(_)) => Err(krishiv_common::error::SortByWithoutBucketing {}.into()),
            (Some(_), None) => Err(krishiv_common::error::LegacyErrorTemp1312 {
                operation: operation.to_owned(),
            }
            .into()),
            (Some(_), Some(_)) => Err(krishiv_common::error::LegacyErrorTemp1313 {
                operation: operation.to_owned(),
            }
            .into()),
        }
    }

    /// The table request of `saveAsTable` and `insertInto`.
    pub(super) fn table_spec(&self, table: &str) -> crate::table_commands::V2WriteSpec {
        crate::table_commands::V2WriteSpec {
            table: table.to_owned(),
            provider: self.format_set.then(|| self.format.clone()),
            options: self.options.clone(),
            properties: BTreeMap::new(),
            partitioned_by: self.partition_by.clone(),
            cluster_by: Vec::new(),
        }
    }

    /// Saves the rows as the table `table` (PySpark's `saveAsTable`). The
    /// mode is Spark's: `errorifexists` (the default) fails with
    /// `TABLE_OR_VIEW_ALREADY_EXISTS` when the table exists, `ignore` leaves
    /// it, `append` adds the rows by column name, and `overwrite` drops an
    /// existing table and creates it again as Spark does (a query that reads
    /// the table it replaces fails with `UNSUPPORTED_OVERWRITE.TABLE`, and
    /// whatever can refuse the new table is checked before the old one is
    /// dropped). The format is [`Self::format`] (default `parquet`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// futures::executor::block_on(s.table("t").write().save_as_table("u"))?;
    /// assert_eq!(futures::executor::block_on(s.table("u").count())?, 0);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `INVALID_SAVE_MODE`; `SORT_BY_WITHOUT_BUCKETING`;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for `bucket_by`, `partition_by`, an
    /// append that does not match the table;
    /// `UNSUPPORTED_OVERWRITE.TABLE`;
    /// `TABLE_OR_VIEW_ALREADY_EXISTS`; analysis and write errors.
    pub async fn save_as_table(self, table: &str) -> Result<(), KrishivError> {
        let mode = krishiv_io::SaveMode::parse(&self.mode)?;
        if let (None, Some(_)) = (&self.bucket_by, &self.sort_by) {
            return Err(krishiv_common::error::SortByWithoutBucketing {}.into());
        }
        if self.bucket_by.is_some() {
            return Err(KrishivUnsupportedFeature {
                feature: "bucketBy".to_owned(),
            }
            .into());
        }
        let spec = self.table_spec(table);
        let df = self.df;
        let session = df.session.clone();
        session
            .on_executor(async move { df.session.exec_save_as_table(&df, spec, mode).await })
            .await
    }

    /// Inserts the rows into the existing table `table` by position
    /// (PySpark's `insertInto`). `errorifexists`, `ignore` and `append` all
    /// append, as in Spark; `overwrite` is `INSERT OVERWRITE` by position.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// futures::executor::block_on(s.table("t").write().save_as_table("u"))?;
    /// futures::executor::block_on(s.table("t").write().insert_into("u"))?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `INVALID_SAVE_MODE`; the bucketing errors of `save`;
    /// `PARTITION_BY_NOT_ALLOWED_WITH_INSERT_INTO` after `partition_by`;
    /// `TABLE_OR_VIEW_NOT_FOUND`; the arity and write errors of `INSERT INTO`.
    pub async fn insert_into(self, table: &str) -> Result<(), KrishivError> {
        let mode = krishiv_io::SaveMode::parse(&self.mode)?;
        self.assert_not_bucketed("insertInto")?;
        if !self.partition_by.is_empty() {
            return Err(krishiv_common::error::PartitionByNotAllowedWithInsertInto {
                table_name: table.to_owned(),
            }
            .into());
        }
        let df = self.df;
        let table = table.to_owned();
        let session = df.session.clone();
        session
            .on_executor(async move { df.session.exec_insert_into(&df, &table, mode).await })
            .await
    }

    /// Writes the rows as text files under `path` (PySpark's `write.text`):
    /// one string column.
    ///
    /// # Errors
    /// As [`Self::save`].
    pub async fn text(self, path: &str) -> Result<(), KrishivError> {
        self.format("text").save(path).await
    }

    /// Writes the rows as Parquet files under `path` (PySpark's
    /// `write.parquet`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let out = dir.path().join("t");
    /// futures::executor::block_on(s.table("t").write().parquet(out.to_str().unwrap()))?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::save`].
    pub async fn parquet(self, path: &str) -> Result<(), KrishivError> {
        self.format("parquet").save(path).await
    }

    /// Writes the rows as CSV files under `path` (PySpark's `write.csv`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let out = dir.path().join("t");
    /// futures::executor::block_on(s.table("t").write().csv(out.to_str().unwrap()))?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::save`].
    pub async fn csv(self, path: &str) -> Result<(), KrishivError> {
        self.format("csv").save(path).await
    }

    /// Writes the rows as JSON files under `path` (PySpark's `write.json`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let out = dir.path().join("t");
    /// futures::executor::block_on(s.table("t").write().json(out.to_str().unwrap()))?;
    /// assert!(out.is_dir());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::save`].
    pub async fn json(self, path: &str) -> Result<(), KrishivError> {
        self.format("json").save(path).await
    }
}

/// Writes a DataFrame's rows to a table ([`DataFrame::write_to_table`];
/// PySpark's `DataFrameWriterV2`, as `df.writeTo` returns it).
///
/// The builder only collects the request; `create`, `replace` and
/// `create_or_replace` run it through the same write and metadata path as
/// `CREATE TABLE ... AS SELECT` (`Session::exec_write_v2`),
/// so `writeTo("t").create()` and `CREATE TABLE t AS SELECT ...` run one
/// code path. The table's schema is the DataFrame's; a failed write leaves
/// no table and no files behind. `append` is `INSERT INTO` by position;
/// `replace` and `create_or_replace` fail as `REPLACE TABLE AS SELECT` does
/// on file-source tables; `overwrite` and `overwrite_partitions` fail with
/// `KRISHIV_UNSUPPORTED_FEATURE` naming `INSERT OVERWRITE`; `partitioned_by`
/// and `cluster_by` fail at the write with `KRISHIV_UNSUPPORTED_FEATURE`.
#[derive(Debug, Clone)]
#[must_use]
pub struct DataFrameWriterV2 {
    pub(super) df: DataFrame,
    pub(super) table: String,
    pub(super) provider: Option<String>,
    pub(super) options: BTreeMap<String, OptionValue>,
    pub(super) properties: BTreeMap<String, String>,
    pub(super) partitioned_by: Vec<String>,
    pub(super) cluster_by: Vec<String>,
}

impl DataFrameWriterV2 {
    /// The file format (`using`): `parquet`, `csv`, `json` or `text`;
    /// without it the provider is `spark.sql.sources.default`. Anything
    /// else fails the write with `DATA_SOURCE_NOT_FOUND`.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write_to_table("u").using("parquet");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn using(mut self, provider: &str) -> Self {
        self.provider = Some(provider.to_owned());
        self
    }

    /// A table option (also a read option of the table); values are typed
    /// as secrets where Spark redacts them.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write_to_table("u").option("header", "true");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options
            .insert(key.to_owned(), OptionValue::from_user(key, value));
        self
    }

    /// Several table options at once.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write_to_table("u").options([("header", "true")]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn options(
        mut self,
        options: impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>,
    ) -> Self {
        for (key, value) in options {
            self.options.insert(
                key.as_ref().to_owned(),
                OptionValue::from_user(key.as_ref(), value.as_ref()),
            );
        }
        self
    }

    /// A `TBLPROPERTY` of the table.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write_to_table("u").table_property("k", "v");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn table_property(mut self, key: &str, value: &str) -> Self {
        self.properties.insert(key.to_owned(), value.to_owned());
        self
    }

    /// Partition the table by `cols` (each call replaces the columns of
    /// the one before, so an empty call clears them). The builder does not
    /// create partitioned tables yet: any column fails the write with
    /// `KRISHIV_UNSUPPORTED_FEATURE` (`CREATE TABLE ... PARTITIONED BY ...
    /// AS SELECT` creates one).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write_to_table("u").partitioned_by(["a"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn partitioned_by(mut self, cols: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        self.partitioned_by = cols.into_iter().map(|c| c.as_ref().to_owned()).collect();
        self
    }

    /// Cluster the table by `cols` (each call replaces the columns of the
    /// one before). Like [`Self::partitioned_by`], any column fails the
    /// write with `KRISHIV_UNSUPPORTED_FEATURE`.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write_to_table("u").cluster_by(["a"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn cluster_by(mut self, cols: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        self.cluster_by = cols.into_iter().map(|c| c.as_ref().to_owned()).collect();
        self
    }

    /// The request as the command path runs it.
    pub(super) fn spec(&self) -> crate::table_commands::V2WriteSpec {
        crate::table_commands::V2WriteSpec {
            table: self.table.clone(),
            provider: self.provider.clone(),
            options: self.options.clone(),
            properties: self.properties.clone(),
            partitioned_by: self.partitioned_by.clone(),
            cluster_by: self.cluster_by.clone(),
        }
    }

    /// Runs the request with `mode` on the session's executor.
    async fn run(self, mode: crate::table_commands::V2Mode) -> Result<(), KrishivError> {
        let spec = self.spec();
        let df = self.df;
        let session = df.session.clone();
        session
            .on_executor(async move { df.session.exec_write_v2(&df, spec, mode, &mut None).await })
            .await
    }

    /// Creates the table; an existing table fails with
    /// `TABLE_OR_VIEW_ALREADY_EXISTS`. Dropping the future before the write
    /// commits cancels it and deletes its files.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// futures::executor::block_on(s.table("t").write_to_table("u").create())?;
    /// assert_eq!(futures::executor::block_on(s.table("u").count())?, 0);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis and planning errors; `TABLE_OR_VIEW_ALREADY_EXISTS`;
    /// `SCHEMA_NOT_FOUND`; `DATA_SOURCE_NOT_FOUND`; write failures;
    /// `KRISHIV_UNSUPPORTED_FEATURE` when partitioning or clustering was
    /// requested.
    pub async fn create(self) -> Result<(), KrishivError> {
        self.run(crate::table_commands::V2Mode::Create).await
    }

    /// Replaces the table. File-source tables cannot be replaced, so this
    /// fails as `REPLACE TABLE AS SELECT` does, with
    /// `UNSUPPORTED_FEATURE.TABLE_OPERATION`.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let e = futures::executor::block_on(s.table("t").write_to_table("u").replace());
    /// assert!(e.is_err());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `UNSUPPORTED_FEATURE.TABLE_OPERATION`.
    pub async fn replace(self) -> Result<(), KrishivError> {
        self.run(crate::table_commands::V2Mode::Replace).await
    }

    /// Creates the table or replaces it: fails like [`Self::replace`].
    ///
    /// # Errors
    /// `UNSUPPORTED_FEATURE.TABLE_OPERATION`.
    pub async fn create_or_replace(self) -> Result<(), KrishivError> {
        self.run(crate::table_commands::V2Mode::CreateOrReplace)
            .await
    }

    /// Appends the rows to the existing table (`INSERT INTO`, by position).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// futures::executor::block_on(s.table("t").write_to_table("u").create())?;
    /// futures::executor::block_on(s.table("t").write_to_table("u").append())?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_NOT_FOUND`; the store-assignment errors of `INSERT`.
    pub async fn append(self) -> Result<(), KrishivError> {
        self.run(crate::table_commands::V2Mode::Append).await
    }

    /// Overwrites the rows `condition` selects (`INSERT OVERWRITE`): always
    /// fails naming it. The condition is
    /// checked by the caller (a column or SQL text); it never runs.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE`.
    pub async fn overwrite(self, _condition: &str) -> Result<(), KrishivError> {
        Err(KrishivUnsupportedFeature {
            feature: "INSERT OVERWRITE".to_owned(),
        }
        .into())
    }

    /// Overwrites the dynamic partitions (`INSERT OVERWRITE` with dynamic
    /// partitions): always fails naming it.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE`.
    pub async fn overwrite_partitions(self) -> Result<(), KrishivError> {
        Err(KrishivUnsupportedFeature {
            feature: "INSERT OVERWRITE with dynamic partitions".to_owned(),
        }
        .into())
    }
}
