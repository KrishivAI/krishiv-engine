// SPDX-License-Identifier: Apache-2.0
//! The rest of `spark.catalog` (Spark 4.x's `Catalog` interface; the calls Spark Connect's `Catalog`
//! relation makes that the embedded API had not): `dropTable`, `dropView`, `createDatabase`,
//! `dropDatabase`, `truncateTable`, `analyzeTable`, `listPartitions`, `listViews`,
//! `getTableProperties`, `getCreateTableString` and the pattern form of `listCatalogs`. Each runs the
//! statement Spark's `CatalogImpl` runs (or filters the listing it shares), so the access checks and
//! errors are the statement's.

use std::collections::BTreeMap;

use arrow_array::{Array, RecordBatch, StringArray};
use krishiv_common::error::KrishivError;

use super::{Catalog, Table, literal, quote, quoted};

/// The strings of column `index` of `batches` (a null stays `None`; a column of another type is cast).
fn strings(batches: &[RecordBatch], index: usize) -> Result<Vec<Option<String>>, KrishivError> {
    let mut out = Vec::new();
    for batch in batches {
        let Some(column) = batch.columns().get(index) else {
            continue;
        };
        let cast = arrow_cast::cast(column, &arrow_schema::DataType::Utf8).map_err(|e| {
            krishiv_common::error::InternalError {
                message: format!("a catalog statement's column: {e}"),
            }
        })?;
        let Some(text) = cast.as_any().downcast_ref::<StringArray>() else {
            continue;
        };
        out.extend((0..text.len()).map(|i| (!text.is_null(i)).then(|| text.value(i).to_owned())));
    }
    Ok(out)
}

impl Catalog<'_> {
    /// `database.name` as one name for the one-name calls: the database is read as a (possibly
    /// dotted) namespace and the name as a single part, as Spark's two-argument overloads do.
    #[must_use]
    pub fn qualified(database: &str, name: &str) -> String {
        format!("{}.{}", quoted(database), quote(name))
    }

    /// The catalogs matching the `*`/`|` pattern, sorted (`listCatalogs(pattern)`).
    #[must_use]
    pub fn list_catalogs_matching(&self, pattern: Option<&str>) -> Vec<String> {
        let mut names = self.list_catalogs();
        if let Some(p) = pattern {
            names.retain(|n| crate::introspection::matches(p, n));
        }
        names
    }

    /// `DROP TABLE [IF EXISTS] name [PURGE]`.
    ///
    /// # Errors
    /// The statement's: `TABLE_OR_VIEW_NOT_FOUND`, `WRONG_COMMAND_FOR_OBJECT_TYPE` for a view.
    pub async fn drop_table(
        &self,
        name: &str,
        if_exists: bool,
        purge: bool,
    ) -> Result<(), KrishivError> {
        let mut sql = String::from("DROP TABLE ");
        if if_exists {
            sql.push_str("IF EXISTS ");
        }
        sql.push_str(&quoted(name));
        if purge {
            sql.push_str(" PURGE");
        }
        self.run(&sql).await
    }

    /// `DROP VIEW [IF EXISTS] name` (a temporary view too).
    ///
    /// # Errors
    /// The statement's.
    pub async fn drop_view(&self, name: &str, if_exists: bool) -> Result<(), KrishivError> {
        let ie = if if_exists { "IF EXISTS " } else { "" };
        self.run(&format!("DROP VIEW {ie}{}", quoted(name))).await
    }

    /// `CREATE DATABASE [IF NOT EXISTS] name [WITH DBPROPERTIES (...)]`.
    ///
    /// # Errors
    /// The statement's: `SCHEMA_ALREADY_EXISTS`.
    pub async fn create_database(
        &self,
        name: &str,
        if_not_exists: bool,
        properties: &BTreeMap<String, String>,
    ) -> Result<(), KrishivError> {
        let escaped = self.session.query_env().parser.escaped_string_literals;
        let mut sql = String::from("CREATE DATABASE ");
        if if_not_exists {
            sql.push_str("IF NOT EXISTS ");
        }
        sql.push_str(&quoted(name));
        if !properties.is_empty() {
            let list = properties
                .iter()
                .map(|(k, v)| Ok(format!("{} = {}", quote(k), literal(v, escaped)?)))
                .collect::<Result<Vec<_>, KrishivError>>()?;
            sql.push_str(&format!(" WITH DBPROPERTIES ({})", list.join(", ")));
        }
        self.run(&sql).await
    }

    /// `DROP DATABASE [IF EXISTS] name [CASCADE]`.
    ///
    /// # Errors
    /// The statement's: `SCHEMA_NOT_FOUND`, `SCHEMA_NOT_EMPTY`.
    pub async fn drop_database(
        &self,
        name: &str,
        if_exists: bool,
        cascade: bool,
    ) -> Result<(), KrishivError> {
        let ie = if if_exists { "IF EXISTS " } else { "" };
        let tail = if cascade { " CASCADE" } else { "" };
        self.run(&format!("DROP DATABASE {ie}{}{tail}", quoted(name)))
            .await
    }

    /// `TRUNCATE TABLE name`.
    ///
    /// # Errors
    /// The statement's.
    pub async fn truncate_table(&self, name: &str) -> Result<(), KrishivError> {
        self.run(&format!("TRUNCATE TABLE {}", quoted(name))).await
    }

    /// `ANALYZE TABLE name COMPUTE STATISTICS [NOSCAN]`.
    ///
    /// # Errors
    /// The statement's.
    pub async fn analyze_table(&self, name: &str, no_scan: bool) -> Result<(), KrishivError> {
        let tail = if no_scan { " NOSCAN" } else { "" };
        self.run(&format!(
            "ANALYZE TABLE {} COMPUTE STATISTICS{tail}",
            quoted(name)
        ))
        .await
    }

    /// The partitions of the table (`SHOW PARTITIONS`: `k=v/k2=v2`), in the statement's order.
    ///
    /// # Errors
    /// The statement's: `TABLE_OR_VIEW_NOT_FOUND`, `INVALID_PARTITION_OPERATION` for an unpartitioned table.
    pub async fn list_partitions(&self, name: &str) -> Result<Vec<String>, KrishivError> {
        let batches = self
            .session
            .sql(&format!("SHOW PARTITIONS {}", quoted(name)))
            .await?
            .collect()
            .await?;
        Ok(strings(&batches, 0)?.into_iter().flatten().collect())
    }

    /// The views (permanent and temporary) of `database` (the current one for `None`) matching
    /// `pattern`: [`Catalog::list_tables`] kept to views, as Spark's `SHOW VIEWS` lists them.
    ///
    /// # Errors
    /// The errors of [`Catalog::list_tables`].
    pub async fn list_views(
        &self,
        database: Option<&str>,
        pattern: Option<&str>,
    ) -> Result<Vec<Table>, KrishivError> {
        Ok(self
            .list_tables(database, pattern)
            .await?
            .into_iter()
            .filter(|t| t.table_type == "VIEW" || t.is_temporary)
            .collect())
    }

    /// The table's properties as `SHOW TBLPROPERTIES` lists them, by key.
    ///
    /// # Errors
    /// The statement's.
    pub async fn get_table_properties(
        &self,
        name: &str,
    ) -> Result<BTreeMap<String, String>, KrishivError> {
        let batches = self
            .session
            .sql(&format!("SHOW TBLPROPERTIES {}", quoted(name)))
            .await?
            .collect()
            .await?;
        let keys = strings(&batches, 0)?;
        let values = strings(&batches, 1)?;
        Ok(keys
            .into_iter()
            .zip(values)
            .filter_map(|(k, v)| Some((k?, v.unwrap_or_default())))
            .collect())
    }

    /// `SHOW CREATE TABLE name [AS SERDE]`: the statement text (empty when the statement answers no row).
    ///
    /// # Errors
    /// The statement's.
    pub async fn get_create_table_string(
        &self,
        name: &str,
        as_serde: bool,
    ) -> Result<String, KrishivError> {
        let tail = if as_serde { " AS SERDE" } else { "" };
        let batches = self
            .session
            .sql(&format!("SHOW CREATE TABLE {}{tail}", quoted(name)))
            .await?
            .collect()
            .await?;
        Ok(strings(&batches, 0)?
            .into_iter()
            .next()
            .flatten()
            .unwrap_or_default())
    }
}
