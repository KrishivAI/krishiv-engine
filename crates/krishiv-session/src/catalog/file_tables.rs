// SPDX-License-Identifier: Apache-2.0
//! Opening the file tables an analyzed plan scans: listing their files
//! (reads by path, file views, file-backed tables), reading the footers on the
//! executor's I/O runtime, and the plan-time distinct-value samples that
//! follow ([`super::ndv_sampling`]).

use std::collections::BTreeMap;
use std::sync::{Arc, PoisonError};

use arrow_schema::{Schema, SchemaRef};
use krishiv_common::error::KrishivError;
use krishiv_exec::planner::FileTable;
use krishiv_plan::{LogicalPlan, ReadSource};

use super::scans::collect_scans;
use super::{FileScans, FilesKey, QueryCatalog, TempTable, run_on};

/// A scan of files in an analyzed plan: what it reads, the table's columns,
/// the scan's column ids with their names, and a permanent table's
/// partition columns.
pub(super) type FileScanOf = (
    FilesKey,
    ReadSource,
    SchemaRef,
    Vec<(krishiv_plan::AttrId, String)>,
    Option<krishiv_exec::files::KnownPartitions>,
);

impl QueryCatalog {
    /// Lists the files of every read by path and file view that the
    /// analyzed `plan` scans (subqueries included), and opens them as file
    /// tables for the planner. A read the analyzer listed keeps that
    /// listing; a view lists its files now.
    ///
    /// # Errors
    /// Listing errors (`PATH_NOT_FOUND` once a view's files are gone, ...)
    /// and errors reading the files' footers.
    pub(crate) async fn open_files(&self, plan: &LogicalPlan) -> Result<(), KrishivError> {
        self.check_reference_options(plan)?;
        let scans = self.file_scans(plan);
        for (key, source, table, _, partitions) in &scans {
            if self.opened(key).is_some() {
                continue;
            }
            if crate::delta::is_delta(source) {
                // The table's log is read once per query, never listed as files.
                self.delta
                    .open_scanned(source, delta_name(key), table)
                    .await?;
                continue;
            }
            let ReadSource::Files {
                format,
                paths,
                options,
            } = source
            else {
                continue;
            };
            let listing = match key {
                FilesKey::Table(_) | FilesKey::Reference { known: true, .. } => {
                    let Some(partitions) = partitions else {
                        return Err(krishiv_common::error::KrishivUnsupportedFeature {
                            feature: "reading a table with a partition outside its default \
                                      directory (ADD PARTITION ... LOCATION, SET LOCATION)"
                                .to_owned(),
                        }
                        .into());
                    };
                    self.files
                        .listing_known(format, paths, options, Arc::clone(table), partitions)
                        .await?
                }
                _ => self.files.listing(format, paths, options).await?,
            };
            let opened = open_table(
                Arc::clone(&listing.format),
                self.scans.clone(),
                Arc::clone(table),
                listing.files.to_vec(),
                listing.options.clone(),
            )
            .await?
            .with_partition_columns(listing.partition_columns.to_vec());
            self.opened
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((key.clone(), Arc::new(opened)));
        }
        // Boxed: the sample's future would otherwise nest inside every
        // caller's state machine (the MCP server's overflows the layout depth).
        Box::pin(self.sample_distinct(plan, &scans)).await;
        Ok(())
    }

    /// Refuses a reference whose option has the key (any case) of one the catalog table stores
    /// with another value, as Spark's `DataSourceUtils.generateDatasourceOptions` does
    /// (`_LEGACY_ERROR_TEMP_1151`); the same value is accepted and the table's own is kept.
    fn check_reference_options(&self, plan: &LogicalPlan) -> Result<(), KrishivError> {
        let mut scans = Vec::new();
        collect_scans(plan, &mut scans);
        for scan in scans {
            let ReadSource::Table { name, options } = &scan.source else {
                continue;
            };
            let Some((_, table)) = self.entry_of(name) else {
                continue;
            };
            let TempTable::Files {
                source:
                    ReadSource::Files {
                        options: stored, ..
                    },
                permanent: true,
                ..
            } = &*table
            else {
                continue;
            };
            for (k, v) in options {
                let clash = stored
                    .iter()
                    .find(|(b, bv)| b.eq_ignore_ascii_case(k) && *bv != v);
                // Spark names the table's spelling of the key.
                if let Some((stored_key, _)) = clash {
                    return Err(krishiv_common::error::LegacyErrorTemp1151 {
                        table: format!(
                            "`{}`",
                            name.iter()
                                .map(String::as_str)
                                .collect::<Vec<_>>()
                                .join("`.`")
                        ),
                        key: stored_key.clone(),
                        config: "spark.sql.legacy.extraOptionsBehavior.enabled".to_owned(),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    /// The scans of files in `plan`: what each reads, and the columns the
    /// analyzer bound (the view's, for a view), which the files map to by
    /// name.
    pub(super) fn file_scans(&self, plan: &LogicalPlan) -> Vec<FileScanOf> {
        let mut scans = Vec::new();
        collect_scans(plan, &mut scans);
        scans
            .into_iter()
            .filter_map(|scan| {
                let (key, source, partitions) = match &scan.source {
                    ReadSource::Table { name, options } => {
                        let (key, table) = self.entry_of(name)?;
                        let TempTable::Files {
                            source,
                            known,
                            permanent,
                            partitions,
                            ..
                        } = &*table
                        else {
                            return None;
                        };
                        let (key, source) = reference_key(key, *known, *permanent, source, options);
                        (key, source, partitions.clone())
                    }
                    source @ ReadSource::Files { .. } => {
                        (FilesKey::Read(source.clone()), source.clone(), None)
                    }
                    _ => return None,
                };
                // The hidden `_metadata` column is the scan's, not the table's.
                let columns = scan.output.len() - usize::from(scan.metadata_column().is_some());
                let table = Arc::new(Schema::new(
                    scan.output[..columns]
                        .iter()
                        .map(|c| c.data_type.to_arrow_field(c.name.clone(), c.nullable))
                        .collect::<Vec<_>>(),
                ));
                let ids = scan.output[..columns]
                    .iter()
                    .map(|c| (c.id, c.name.clone()))
                    .collect();
                Some((key, source, table, ids, partitions))
            })
            .collect()
    }
}

/// How Delta errors name the table a scan of `key` reads: a permanent table by its
/// `namespace.table` (delta-spark's catalog form); a path or a temporary view by the path.
fn delta_name(key: &FilesKey) -> Option<String> {
    match key {
        FilesKey::Table(k) | FilesKey::Reference { key: k, .. } if k.contains('.') => {
            Some(k.clone())
        }
        _ => None,
    }
}

/// The key a reference to a file table scans under, and what it reads. A reference that carries
/// read options (`FROM t WITH ('k' = 'v')`) is a read of the table's files with those options over
/// the table's own, so two references with different options keep their own listing, decoder
/// options and scan; one without options shares the table's.
pub(super) fn reference_key(
    key: String,
    known: bool,
    permanent: bool,
    source: &ReadSource,
    options: &BTreeMap<String, krishiv_common::secret::OptionValue>,
) -> (FilesKey, ReadSource) {
    // Spark applies a reference's options to a catalog table's scan only (`resolveTempView`
    // drops them for a temporary view), and a key the table already stores keeps the table's
    // spelling and value: a different value is an error (`check_reference_options`).
    if let (false, true, ReadSource::Files { .. }) = (options.is_empty(), permanent, source) {
        let mut merged = source.clone();
        if let ReadSource::Files { options: base, .. } = &mut merged {
            for (k, v) in options {
                if !base.keys().any(|b| b.eq_ignore_ascii_case(k)) {
                    base.insert(k.clone(), v.clone());
                }
            }
        }
        let key = FilesKey::Reference {
            key,
            known,
            source: merged.clone(),
        };
        return (key, merged);
    }
    let key = if known {
        FilesKey::Table(key)
    } else {
        FilesKey::View(key)
    };
    (key, source.clone())
}

/// [`FileTable::open`] on the executor's I/O runtime: footer reads are
/// object-store requests, which belong there and not on the compute
/// runtime. Dropping the returned future aborts the open.
async fn open_table(
    format: Arc<dyn krishiv_io::FileFormat>,
    scans: FileScans,
    table: SchemaRef,
    files: Vec<krishiv_io::FileRef>,
    options: krishiv_io::FormatOptions,
) -> Result<FileTable, KrishivError> {
    let runtime = scans.runtime.clone();
    run_on(&runtime, async move {
        FileTable::open(
            format,
            scans.io,
            table,
            files,
            options,
            &scans.config,
            scans.slots,
        )
        .await
    })
    .await
}

/// The partitions a read of `table` lists: its partition columns and the
/// values of the partitions its catalog registered. `None` when a registered
/// partition is not at its default directory under the table (`ADD PARTITION
/// ... LOCATION`, `SET LOCATION`): Spark reads it there, a listing of the table
/// does not.
pub(super) fn known_partitions(
    table: &crate::store::CatalogTable,
) -> Option<krishiv_exec::files::KnownPartitions> {
    let columns = table.layout.partition_columns.clone();
    let mut registered = Vec::with_capacity(table.layout.partitions.len());
    for p in &table.layout.partitions {
        let mut dir = table.location.clone();
        let mut values = Vec::with_capacity(p.spec.len());
        for (k, v) in &p.spec {
            dir.push(format!(
                "{}={}",
                crate::partitions::escape_path_name(k),
                crate::partitions::escape_path_name(v)
            ));
            values.push((v != krishiv_io::DEFAULT_PARTITION).then(|| v.clone()));
        }
        if dir != p.location {
            return None;
        }
        registered.push(values);
    }
    Some(krishiv_exec::files::KnownPartitions {
        columns,
        registered,
    })
}
