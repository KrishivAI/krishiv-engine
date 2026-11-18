// SPDX-License-Identifier: Apache-2.0
//! The write commands (P3-T012 group 11, PR-24), after Spark's `SparkConnectPlanner` at v4.3.0:
//! `transformWriteOperation` (`df.write`, `DataFrameWriter`) and `transformWriteOperationV2`
//! (`df.writeTo`, `DataFrameWriterV2`).
//!
//! The translator reads the writer state into a [`WriteSpec`] or a [`WriteV2Spec`]; the server
//! translates the command's relation and runs the write through `krishiv-session`'s writers, the
//! same builders the Python facade calls (`DataFrame::write_to_with`, `write_table_with`,
//! `write_to_table_with`), so a Connect write and an embedded one run one code path. The writer
//! state is applied in Spark's order: mode, options, `sortBy`, `bucketBy`, `partitionBy`, format.
//!
//! Refused with `KRISHIV_UNSUPPORTED_FEATURE`: `clusterBy` on a V1 writer (Krishiv's V1 writer has
//! none), `withSchemaEvolution` on either, and a V2 partitioning that is not a plain column (a
//! transform such as `years(col)`). An unspecified or unknown save method or V2 mode answers
//! Spark's `CONNECT_INVALID_PLAN.INVALID_ENUM`.

use std::collections::BTreeMap;
use std::fmt;

use super::Result;
use super::errors::{invalid_enum, unsupported_field};
use crate::proto::v4_3 as pb;
use crate::proto::v4_3::write_operation::{SaveMode, SaveType, save_table::TableSaveMethod};
use crate::proto::v4_3::write_operation_v2::Mode;

/// Where a V1 write goes.
#[derive(Clone, PartialEq, Eq)]
pub enum WriteTarget {
    /// `save(path)`; `None` for `save()`, which writes to the `path` option.
    Path(Option<String>),
    /// `saveAsTable(name)`.
    SaveAsTable(String),
    /// `insertInto(name)`.
    InsertInto(String),
}

/// A `write_operation` (`DataFrameWriter`), besides its relation.
#[derive(Clone, PartialEq, Eq)]
pub struct WriteSpec {
    /// `source`: the format, if the client set one (otherwise `spark.sql.sources.default`).
    pub source: Option<String>,
    /// The save mode's Spark name (`append`, `overwrite`, `errorifexists`, `ignore`), if set.
    pub mode: Option<&'static str>,
    /// `options`, sorted by key.
    pub options: Vec<(String, String)>,
    /// `sortBy` columns, if any.
    pub sort_by: Option<Vec<String>>,
    /// `bucketBy`: the bucket count and columns.
    pub bucket_by: Option<(i32, Vec<String>)>,
    /// `partitionBy` columns.
    pub partition_by: Vec<String>,
    /// Where the rows go.
    pub target: WriteTarget,
}

/// A `write_operation_v2` (`DataFrameWriterV2`), besides its relation.
#[derive(Clone, PartialEq, Eq)]
pub struct WriteV2Spec {
    /// `table_name`.
    pub table: String,
    /// `provider` (`using`); Spark reads it only for `create`, `replace` and `createOrReplace`.
    pub provider: Option<String>,
    /// `options`, sorted by key.
    pub options: Vec<(String, String)>,
    /// `table_properties`, sorted by key.
    pub properties: Vec<(String, String)>,
    /// `partitionedBy` columns.
    pub partitioned_by: Vec<String>,
    /// `clusterBy` columns.
    pub cluster_by: Vec<String>,
    /// The terminal action, as the session's `write_to_table_with` names it (`create`, `replace`,
    /// `createOrReplace`, `append`, `overwrite`, `overwritePartitions`).
    pub mode: &'static str,
}

// No paths, option values or names in a log (crate invariant: a plan can hold credentials).
impl fmt::Debug for WriteTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Path(None) => "Path(None)",
            Self::Path(Some(_)) => "Path(..)",
            Self::SaveAsTable(_) => "SaveAsTable(..)",
            Self::InsertInto(_) => "InsertInto(..)",
        })
    }
}

impl fmt::Debug for WriteSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteSpec")
            .field("source_set", &self.source.is_some())
            .field("mode", &self.mode)
            .field("options", &self.options.len())
            .field("sort_by", &self.sort_by.as_ref().map(Vec::len))
            .field(
                "bucket_by",
                &self.bucket_by.as_ref().map(|(n, c)| (n, c.len())),
            )
            .field("partition_by", &self.partition_by.len())
            .field("target", &self.target)
            .finish()
    }
}

impl fmt::Debug for WriteV2Spec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteV2Spec")
            .field("provider_set", &self.provider.is_some())
            .field("options", &self.options.len())
            .field("properties", &self.properties.len())
            .field("partitioned_by", &self.partitioned_by.len())
            .field("cluster_by", &self.cluster_by.len())
            .field("mode", &self.mode)
            .finish()
    }
}

fn sorted(map: &BTreeMap<String, String>) -> Vec<(String, String)> {
    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// Reads a `write_operation`'s writer state.
///
/// # Errors
/// `INVALID_ENUM` for an unknown save mode or an unspecified or unknown table save method;
/// `KRISHIV_UNSUPPORTED_FEATURE` for `clustering_columns` and `with_schema_evolution`.
pub fn write_spec(w: &pb::WriteOperation) -> Result<WriteSpec> {
    if !w.clustering_columns.is_empty() {
        return Err(unsupported_field(
            "WriteOperation",
            "clustering_columns",
            "clusterBy on a DataFrameWriter",
        ));
    }
    if w.with_schema_evolution {
        return Err(unsupported_field(
            "WriteOperation",
            "with_schema_evolution",
            "writes with schema evolution",
        ));
    }
    let mode = match SaveMode::try_from(w.mode) {
        Ok(SaveMode::Unspecified) => None,
        Ok(SaveMode::Append) => Some("append"),
        Ok(SaveMode::Overwrite) => Some("overwrite"),
        Ok(SaveMode::ErrorIfExists) => Some("errorifexists"),
        Ok(SaveMode::Ignore) => Some("ignore"),
        Err(_) => {
            return Err(invalid_enum(
                "spark.connect.WriteOperation.SaveMode",
                "UNRECOGNIZED",
                w.mode,
            ));
        }
    };
    let target = match &w.save_type {
        None => WriteTarget::Path(None),
        Some(SaveType::Path(path)) => WriteTarget::Path(Some(path.clone())),
        Some(SaveType::Table(table)) => {
            const METHOD: &str = "spark.connect.WriteOperation.SaveTable.TableSaveMethod";
            match TableSaveMethod::try_from(table.save_method) {
                Ok(TableSaveMethod::SaveAsTable) => {
                    WriteTarget::SaveAsTable(table.table_name.clone())
                }
                Ok(TableSaveMethod::InsertInto) => {
                    WriteTarget::InsertInto(table.table_name.clone())
                }
                Ok(TableSaveMethod::Unspecified) => {
                    return Err(invalid_enum(
                        METHOD,
                        "TABLE_SAVE_METHOD_UNSPECIFIED",
                        table.save_method,
                    ));
                }
                Err(_) => {
                    return Err(invalid_enum(METHOD, "UNRECOGNIZED", table.save_method));
                }
            }
        }
    };
    Ok(WriteSpec {
        source: w.source.clone(),
        mode,
        options: sorted(&w.options),
        sort_by: (!w.sort_column_names.is_empty()).then(|| w.sort_column_names.clone()),
        bucket_by: w
            .bucket_by
            .as_ref()
            .map(|b| (b.num_buckets, b.bucket_column_names.clone())),
        partition_by: w.partitioning_columns.clone(),
        target,
    })
}

/// Reads a `write_operation_v2`'s writer state.
///
/// # Errors
/// `INVALID_ENUM` for an unspecified or unknown mode; `KRISHIV_UNSUPPORTED_FEATURE` for
/// `with_schema_evolution` and for a partitioning expression that is not a column.
pub fn write_v2_spec(w: &pb::WriteOperationV2) -> Result<WriteV2Spec> {
    const MODE: &str = "spark.connect.WriteOperationV2.Mode";
    if w.with_schema_evolution {
        return Err(unsupported_field(
            "WriteOperationV2",
            "with_schema_evolution",
            "writes with schema evolution",
        ));
    }
    let (mode, reads_provider) = match Mode::try_from(w.mode) {
        Ok(Mode::Create) => ("create", true),
        Ok(Mode::Overwrite) => ("overwrite", false),
        Ok(Mode::OverwritePartitions) => ("overwritePartitions", false),
        Ok(Mode::Append) => ("append", false),
        Ok(Mode::Replace) => ("replace", true),
        Ok(Mode::CreateOrReplace) => ("createOrReplace", true),
        Ok(Mode::Unspecified) => {
            return Err(invalid_enum(MODE, "MODE_UNSPECIFIED", w.mode));
        }
        Err(_) => return Err(invalid_enum(MODE, "UNRECOGNIZED", w.mode)),
    };
    let partitioned_by = w
        .partitioning_columns
        .iter()
        .map(|e| match &e.expr_type {
            Some(pb::expression::ExprType::UnresolvedAttribute(a)) => {
                Ok(a.unparsed_identifier.clone())
            }
            _ => Err(unsupported_field(
                "WriteOperationV2",
                "partitioning_columns",
                "partition transforms",
            )),
        })
        .collect::<Result<_>>()?;
    Ok(WriteV2Spec {
        table: w.table_name.clone(),
        provider: if reads_provider {
            w.provider.clone()
        } else {
            None
        },
        options: sorted(&w.options),
        properties: sorted(&w.table_properties),
        partitioned_by,
        cluster_by: w.clustering_columns.clone(),
        mode,
    })
}

#[cfg(test)]
mod tests;
