// SPDX-License-Identifier: Apache-2.0
//! Helpers of table and file writes: partition overwrite, the commit log, file counts and committed files.

use super::*;

/// Whether the write overwrites dynamically: the `partitionOverwriteMode`
/// write option (any case), else `spark.sql.sources.partitionOverwriteMode`,
/// is `DYNAMIC`.
///
/// # Errors
/// `KRISHIV_INVALID_FORMAT_OPTION_VALUE` for an option value other than
/// `STATIC` or `DYNAMIC`.
pub(super) fn partition_overwrite_is_dynamic(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
) -> Result<bool, KrishivError> {
    let from_option = options
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("partitionOverwriteMode"));
    if let Some((key, value)) = from_option {
        let text = value.as_plain().unwrap_or("<secret>");
        return match text.to_ascii_uppercase().as_str() {
            "DYNAMIC" => Ok(true),
            "STATIC" => Ok(false),
            _ => Err(krishiv_common::error::KrishivInvalidFormatOptionValue {
                value: format!("'{text}'"),
                option: format!("\"{key}\""),
                format: "\"partitioned write\"".to_owned(),
                reason: "expected STATIC or DYNAMIC".to_owned(),
            }
            .into()),
        };
    }
    Ok(config
        .get(&PARTITION_OVERWRITE_MODE)
        .and_then(ConfigValue::as_str)
        == Some("DYNAMIC"))
}

/// Whether this write goes through the commit log:
/// `krishiv.sql.files.commitProtocol` `LOG`, or `AUTO` where the store
/// creates objects only if absent (probed once per write; a store that
/// fails the probe falls back to the plain commit with a warning).
pub(super) async fn uses_commit_log(
    config: &Config,
    probes: &Mutex<HashMap<String, bool>>,
    io: &krishiv_io::IoContext,
    store: &StoreId,
    at: &krishiv_exec::files::LocatedPath,
    job: &str,
) -> Result<bool, KrishivError> {
    let protocol = config
        .get(&krishiv_common::config::FILES_COMMIT_PROTOCOL)
        .and_then(ConfigValue::as_str)
        .map(str::to_ascii_uppercase);
    let log = krishiv_io::commit_log::CommitLog::new(
        io,
        store.clone(),
        at.path.clone(),
        at.display.clone(),
    );
    // A directory that already has a log stays logged whatever the setting says: readers apply the
    // log, so a write that skipped it would succeed and stay invisible.
    if log.exists().await? {
        return Ok(true);
    }
    // A partition directory (`name=value`) below a logged root cannot be committed on its own: the
    // root's readers would never see its files, and a log of its own would shadow the root's. The
    // write goes through the root. The ancestors are probed through the store as the path names
    // it (`at.store`), not the one anchored at the target, which cannot see above it; the root
    // may be the store's own root (an empty path).
    let mut parts: Vec<String> = at.path.parts().map(|p| p.as_ref().to_owned()).collect();
    while parts
        .last()
        .is_some_and(|name| name.split_once('=').is_some_and(|(k, _)| !k.is_empty()))
    {
        parts.pop();
        let parent = krishiv_io::commit_log::CommitLog::new(
            io,
            at.store.clone(),
            object_store::path::Path::from_iter(parts.iter().map(String::as_str)),
            at.display.clone(),
        );
        if parent.exists().await? {
            return Err(krishiv_common::error::KrishivUnsupportedFeature {
                feature: format!(
                    "a write into a partition directory below a directory with a commit log ({}): write through the directory that has the log",
                    at.display
                ),
            }
            .into());
        }
    }
    if protocol.as_deref() == Some("NONE") {
        return Ok(false);
    }
    let known = probes
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(at.store.key())
        .copied();
    let supported = match known {
        Some(supported) => supported,
        None => {
            let supported = log.supports_put_if_absent(job).await?;
            probes
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(at.store.key().to_owned(), supported);
            supported
        }
    };
    if !supported {
        if protocol.as_deref() == Some("LOG") {
            return Err(krishiv_common::error::KrishivUnsupportedFeature {
                feature: format!(
                    "the commit log on a store without put-if-absent ({})",
                    at.display
                ),
            }
            .into());
        }
        tracing::warn!(
            path = %at.display,
            "the store does not create objects only if absent: writing without the commit log"
        );
    }
    Ok(supported)
}

/// The `maxRecordsPerFile` write option (`FILE_SOURCE_OPTIONS`), case
/// insensitive; 0 (unlimited) when unset.
pub(super) fn max_records_per_file(
    options: &BTreeMap<String, OptionValue>,
    format: &str,
) -> Result<u64, KrishivError> {
    let Some((key, value)) = options
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("maxRecordsPerFile"))
    else {
        return Ok(0);
    };
    let Some(raw) = value.as_plain() else {
        return Err(krishiv_common::error::KrishivInvalidFormatOptionValue {
            value: "<secret>".to_owned(),
            option: format!("\"{key}\""),
            format: format!("\"{format}\""),
            reason: "a secret is not allowed here".to_owned(),
        }
        .into());
    };
    raw.trim().parse::<u64>().map_err(|_| {
        krishiv_common::error::KrishivInvalidFormatOptionValue {
            value: format!("'{raw}'"),
            option: format!("\"{key}\""),
            format: format!("\"{format}\""),
            reason: "expected an integer in 0..=18446744073709551615".to_owned(),
        }
        .into()
    })
}

/// A write directory that could not be opened, as the path-based store
/// reports it (`KRISHIV_OBJECT_STORE_REQUEST_FAILED`).
pub(super) fn anchor_failed(at: &krishiv_exec::files::LocatedPath, reason: &str) -> KrishivError {
    krishiv_common::error::KrishivObjectStoreRequestFailed {
        path: at.display.clone(),
        attempts: "1".to_owned(),
        reason: reason.to_owned(),
    }
    .into()
}

/// Removes a write's private store registration when the write is over.
pub(super) struct StoreRegistration {
    pub(super) io: Arc<IoScheduler>,
    pub(super) id: StoreId,
}

impl Drop for StoreRegistration {
    fn drop(&mut self) {
        self.io.unregister_store(&self.id);
    }
}

/// The commit's files from the write plan's result rows (one per file:
/// `path`, `size`, `rows`).
pub(super) fn committed_files(
    batches: &[RecordBatch],
) -> Result<Vec<krishiv_io::WrittenFileInfo>, KrishivError> {
    use arrow_array::Int64Array;
    use arrow_array::StringViewArray;
    use krishiv_io::{FileStatistics, Precision};
    let mut out = Vec::new();
    for batch in batches {
        if batch.num_columns() != 3 {
            return Err(internal(
                "a write result without path, size and rows".to_owned(),
            ));
        }
        let path = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .ok_or_else(|| internal("a write result whose path is not a string".to_owned()))?;
        let size = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| internal("a write result whose size is not an integer".to_owned()))?;
        let rows = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| internal("a write result whose rows are not an integer".to_owned()))?;
        for i in 0..batch.num_rows() {
            let size = u64::try_from(size.value(i))
                .map_err(|_| internal(format!("a written file of {} bytes", size.value(i))))?;
            let rows = u64::try_from(rows.value(i))
                .map_err(|_| internal(format!("a written file of {} rows", rows.value(i))))?;
            out.push(krishiv_io::WrittenFileInfo {
                // As the writer made it (`Path::parse`: the escaped directory names stay as text), so it names
                // the key a listing returns; `Path::from` would encode the `%` of `%2F` a second time.
                path: object_store::path::Path::parse(path.value(i))
                    .map_err(|e| internal(format!("a written file with an invalid path ({e})")))?,
                size,
                rows,
                statistics: FileStatistics::with_rows(Precision::Exact(rows)),
            });
        }
    }
    Ok(out)
}

/// The `partitionBy` columns as `(index, name)` pairs in `partitionBy`
/// order.
///
/// # Errors
/// `COLUMN_ALREADY_EXISTS` for a repeated name (checked first);
/// `PARTITION_COLUMN_NOT_FOUND_IN_SCHEMA` for a name the schema has no
/// column for.
pub(super) fn resolve_partitions(
    schema: &SchemaRef,
    partition_by: &[String],
    case_sensitive: bool,
) -> Result<Vec<(usize, String)>, KrishivError> {
    // Spark rejects a repeated partition column before it looks any up
    // (`PartitioningUtils.validatePartitionColumn`); distinct names then
    // match distinct columns.
    check_column_name_duplication(partition_by.iter().map(String::as_str), case_sensitive)?;
    let mut out = Vec::new();
    for name in partition_by {
        let mut matches = schema.fields().iter().enumerate().filter(|(_, f)| {
            if case_sensitive {
                f.name() == name
            } else {
                f.name().eq_ignore_ascii_case(name)
            }
        });
        let Some((index, field)) = matches.next() else {
            return Err(PartitionColumnNotFoundInSchema {
                column: format!("`{}`", name.replace('`', "``")),
                schema: schema_simple_string(schema),
            }
            .into());
        };
        if matches.next().is_some() {
            return Err(internal(format!(
                "partition column `{name}` matches several columns"
            )));
        }
        // Checked here so that nothing is started for a column type that
        // has no partition path text.
        krishiv_io::check_partition_column(field.name(), field.data_type())?;
        out.push((index, field.name().clone()));
    }
    Ok(out)
}

/// `schema` as Spark prints one (`struct<id:int,...>`).
pub(super) fn schema_simple_string(schema: &SchemaRef) -> String {
    let fields: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| {
            let ty = DataType::from_arrow(f.data_type())
                .map(|t| t.simple_string())
                .unwrap_or_else(|_| f.data_type().to_string());
            format!("{}:{ty}", f.name())
        })
        .collect();
    format!("struct<{}>", fields.join(","))
}
