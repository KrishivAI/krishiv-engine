// SPDX-License-Identifier: Apache-2.0
//! The write commands of `Plan.command` (P3-T012 group 11; Spark's `SparkConnectPlanner`
//! `transformWriteOperation` and `transformWriteOperationV2` at v4.3.0): the command's relation is
//! written through `krishiv-session`'s writers, the same calls the Python facade makes
//! (`DataFrame::write_to_with`, `write_table_with`, `write_to_table_with`), so a Connect write and
//! an embedded one share every check and error. The command answers no rows: the stream ends.

use krishiv_common::error::{KrishivError, LegacyErrorTemp1307, LegacyErrorTemp2050};
use krishiv_connect::translate::{ViewSpec, WriteSpec, WriteTarget, WriteV2Spec};
use krishiv_session::DataFrame;

/// Writes `source` as a `DataFrameWriter` with the state of `spec`; a path write without a
/// format takes `session`'s `spark.sql.sources.default`.
///
/// `save()` without a path writes to the `path` option, as Spark's `saveCommand(None)` reads it
/// (ASCII case-insensitive); with neither, Spark's `DataSource` answers `_LEGACY_ERROR_TEMP_2050`
/// ("Expected exactly one path to be specified, but got: ."), which this answers before reading
/// the relation.
///
/// `save(path)` with a `path` option is `_LEGACY_ERROR_TEMP_1307` (Spark's
/// `pathOptionNotSetCorrectlyWhenWritingError`) unless `spark.sql.legacy.pathOptionBehavior.enabled`.
///
/// # Errors
/// The session writer's errors; `_LEGACY_ERROR_TEMP_1307`; `_LEGACY_ERROR_TEMP_2050` for a path-less `save()` with no
/// `path` option.
pub(super) async fn write(
    session: &krishiv_session::Session,
    source: &DataFrame,
    spec: WriteSpec,
) -> Result<(), KrishivError> {
    let WriteSpec {
        source: format,
        mode,
        mut options,
        sort_by,
        bucket_by,
        partition_by,
        target,
    } = spec;
    // `errorifexists` is the writers' default, as Spark's `DataFrameWriter` mode default.
    let mode = mode.unwrap_or("errorifexists");
    match target {
        WriteTarget::Path(path) => {
            let path = match path {
                Some(path) => {
                    // Spark (`pathOptionBehavior.enabled=false`) refuses both forms at once.
                    if options.iter().any(|(k, _)| k.eq_ignore_ascii_case("path"))
                        && !session
                            .conf()
                            .get("spark.sql.legacy.pathOptionBehavior.enabled")
                            .is_some_and(|v| v.eq_ignore_ascii_case("true"))
                    {
                        return Err(LegacyErrorTemp1307 {
                            config: "spark.sql.legacy.pathOptionBehavior.enabled".to_owned(),
                        }
                        .into());
                    }
                    path
                }
                None => {
                    let at = options
                        .iter()
                        .position(|(k, _)| k.eq_ignore_ascii_case("path"))
                        .ok_or_else(|| -> KrishivError {
                            LegacyErrorTemp2050 {
                                paths: String::new(),
                            }
                            .into()
                        })?;
                    options.remove(at).1
                }
            };
            let format = match format {
                Some(format) => format,
                // Spark's `DataFrameWriter` source default.
                None => session
                    .conf()
                    .get("spark.sql.sources.default")
                    .unwrap_or_else(|| "parquet".to_owned()),
            };
            source
                .write_to_with(
                    &format,
                    mode,
                    &path,
                    options,
                    partition_by,
                    bucket_by,
                    sort_by,
                )
                .await
        }
        WriteTarget::SaveAsTable(table) => {
            Box::pin(source.write_table_with(
                false,
                &table,
                format.as_deref(),
                mode,
                options,
                partition_by,
                bucket_by,
                sort_by,
            ))
            .await
        }
        WriteTarget::InsertInto(table) => {
            Box::pin(source.write_table_with(
                true,
                &table,
                format.as_deref(),
                mode,
                options,
                partition_by,
                bucket_by,
                sort_by,
            ))
            .await
        }
    }
}

/// Writes `source` as a `DataFrameWriterV2` with the state of `spec`.
///
/// # Errors
/// The session writer's errors.
pub(super) async fn write_v2(source: &DataFrame, spec: WriteV2Spec) -> Result<(), KrishivError> {
    let WriteV2Spec {
        table,
        provider,
        options,
        properties,
        partitioned_by,
        cluster_by,
        mode,
    } = spec;
    source
        .write_to_table_with(
            &table,
            provider,
            options,
            properties,
            partitioned_by,
            cluster_by,
            mode,
            None,
        )
        .await
}

/// Registers `source` as the temporary view `spec` names: `createTempView`,
/// `createOrReplaceTempView`, `createGlobalTempView` and `createOrReplaceGlobalTempView`
/// (Spark's `transformCreateViewCommand`: a global view lives in `global_temp`).
///
/// # Errors
/// The session's: the view's analysis errors, then `TEMP_TABLE_OR_VIEW_ALREADY_EXISTS` for a
/// create that finds the name taken, and `KRISHIV_RESERVED_TEMP_VIEW_NAME`.
pub(super) async fn create_view(source: &DataFrame, spec: ViewSpec) -> Result<(), KrishivError> {
    let ViewSpec {
        name,
        is_global,
        replace,
    } = spec;
    match (is_global, replace) {
        (false, false) => source.create_temp_view(&name).await,
        (false, true) => source.create_or_replace_temp_view(&name).await,
        (true, false) => source.create_global_temp_view(&name).await,
        (true, true) => source.create_or_replace_global_temp_view(&name).await,
    }
}
