// SPDX-License-Identifier: Apache-2.0
//! Delta tables in DDL (P3-T021 PR-08b part 2): `CREATE TABLE t [(columns)] USING delta
//! LOCATION 'p'` over an existing Delta table (its columns and partitioning from the log),
//! `CREATE TEMPORARY VIEW v USING delta OPTIONS (path 'p')`, and the refusals of every Delta
//! write and DDL until PR-37a.
//!
//! What Spark + delta-spark 4.4.1 does with each statement is recorded in
//! `tests/fixtures/delta-4.4.1/SESSION.json`: a column list must match the log
//! (`DELTA_CREATE_TABLE_SCHEME_MISMATCH`), and so must the partitioning
//! (`DELTA_CREATE_TABLE_WITH_DIFFERENT_PARTITIONING`, checked after the schema). The writes
//! (`INSERT`, CTAS, `CREATE` without a location or over a location without a Delta table,
//! `ALTER`) succeed in Spark; Krishiv refuses each by name as
//! `UNSUPPORTED_FEATURE.TABLE_OPERATION`. So are `TBLPROPERTIES` and options other than `path`
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

/// `SchemaUtils.reportDifferences(existing, specified)` for top-level columns: missing and
/// additional fields, then each type difference. delta-spark lists the missing and additional
/// fields in its set's order; here they are in the schema's order (a known difference).
fn schema_differences(existing: &[TableColumn], specified: &[StoreColumn]) -> Vec<String> {
    let find_specified = |n: &str| specified.iter().find(|c| c.name.eq_ignore_ascii_case(n));
    let find_existing = |n: &str| existing.iter().find(|c| c.name.eq_ignore_ascii_case(n));
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
    for s in specified {
        if let Some(e) = find_existing(&s.name)
            && e.data_type != s.data_type
        {
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
        original: &str,
        name: &TableName,
        partition_names: &[String],
        mode: CreateMode,
        located: bool,
    ) -> Result<(), KrishivError> {
        let quoted = name.quoted();
        if t.as_query.is_some() {
            return Err(delta_write_refused(quoted, "CREATE TABLE AS SELECT"));
        }
        if !located {
            return Err(delta_write_refused(
                quoted,
                "CREATE TABLE without LOCATION (a new Delta table)",
            ));
        }
        if t.bucket.is_some() {
            return Err(delta_write_refused(quoted, "CREATE TABLE ... CLUSTERED BY"));
        }
        let specified = self.columns_of(t, original)?;
        // The location is read before it is registered.
        self.authorize_path(&table.location, &[Privilege::Select], StatementClass::Ddl)
            .await?;
        let env = self.query_env();
        let source = crate::catalog::table_source(&table);
        let shape = env
            .catalog
            .delta
            .shape(&source, format!("{}.{}", name.namespace, name.name))
            .await?;
        if !shape.existing {
            return Err(delta_write_refused(
                quoted,
                "CREATE TABLE over a location without a Delta table (a new Delta table)",
            ));
        }
        if !specified.is_empty() {
            let differences = schema_differences(&shape.columns, &specified);
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
        let path_key = options
            .keys()
            .find(|k| k.eq_ignore_ascii_case("path"))
            .cloned();
        let Some(path) = path_key.and_then(|k| options.remove(&k)) else {
            return Err(unsupported("a Delta temporary view without a path"));
        };
        let qualified = self.qualify_path(&path)?;
        self.authorize_path(
            Path::new(&qualified),
            &[Privilege::Select],
            StatementClass::Ddl,
        )
        .await?;
        let source = krishiv_plan::ReadSource::Files {
            format: DELTA.to_owned(),
            paths: vec![OptionValue::from_user("path", qualified)],
            options: options
                .into_iter()
                .map(|(k, val)| {
                    let val = OptionValue::from_user(&k, val);
                    (k, val)
                })
                .collect(),
        };
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
